/**
 * Aya inference server — minimal HTTP API.
 *
 * Endpoints:
 *   POST /generate   { "prompt": "...", "max_tokens": 256, "temperature": 0.7 }
 *   GET  /health
 *
 * INL 2025
 */
#include "gguf.h"
#include "model.h"
#include "tokenizer.h"
#include "sampler.h"
#include "http.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef _WIN32
  #pragma comment(lib, "ws2_32.lib")
#endif

static volatile int generating = 0;
static float *token_quality = NULL;  /* pre-computed token quality bias */

/* ---- Generation core ---- */

/* Build prompt token sequence based on architecture.
 * Cohere2: BOS + START_TURN + USER + <tokens> + END_TURN + START_TURN + CHATBOT
 * GPT-2/other: BOS + <tokens> */
static int *build_prompt(tokenizer_t *tk, const char *architecture,
                         const char *prompt, int *out_len) {
    int raw_len;
    int *raw_ids = tokenizer_encode(tk, prompt, 0, &raw_len);

    int is_cohere = (strcmp(architecture, "cohere2") == 0 ||
                     strcmp(architecture, "cohere") == 0 ||
                     architecture[0] == '\0');  /* default to Cohere for compat */

    if (is_cohere) {
        /* Cohere chat template */
        int n = 3 + raw_len + 3;
        int *ids = malloc(n * sizeof(int));
        ids[0] = 2;  /* BOS */
        ids[1] = 5;  /* START_OF_TURN */
        ids[2] = 7;  /* USER */
        memcpy(ids + 3, raw_ids, raw_len * sizeof(int));
        ids[3 + raw_len]     = 6;  /* END_OF_TURN */
        ids[3 + raw_len + 1] = 5;  /* START_OF_TURN */
        ids[3 + raw_len + 2] = 8;  /* CHATBOT */
        free(raw_ids);
        *out_len = n;
        printf("  Encoded %d tokens (Cohere chat template)\n", n);
        return ids;
    } else {
        /* GPT-2 / generic: BOS + raw tokens */
        int n = 1 + raw_len;
        int *ids = malloc(n * sizeof(int));
        ids[0] = tk->bos_id;
        memcpy(ids + 1, raw_ids, raw_len * sizeof(int));
        free(raw_ids);
        *out_len = n;
        printf("  Encoded %d tokens (raw + BOS)\n", n);
        return ids;
    }
}

/* Shared generation loop — works for both stream and non-stream.
 * Returns number of tokens generated. */
static int generate_tokens(model_t *model, kv_cache_t *cache, tokenizer_t *tk,
                           int *input_ids, int n_tokens, sample_params_t sp,
                           int max_tokens, int min_tokens, float rep_penalty,
                           float quality_alpha, float entropy_threshold,
                           int stream, SOCKET client) {
    int pos = 0;
    float *logits = NULL;

    /* Prefill */
    printf("  Prefill %d tokens...\n", n_tokens); fflush(stdout);
    for (int i = 0; i < n_tokens; i++) {
        if (logits) free(logits);
        logits = model_forward(model, cache, input_ids[i], pos);
        pos++;
        if ((i + 1) % 50 == 0 || i == n_tokens - 1) {
            printf("  Prefill %d/%d\n", i + 1, n_tokens);
            fflush(stdout);
        }
    }
    printf("  Prefill done, generating...\n"); fflush(stdout);

    /* Generation state */
    int *gen_ids = malloc(max_tokens * sizeof(int));
    int gen_count = 0;
    int consecutive_newlines = 0;
    int high_entropy_count = 0;
    int got_first_word = 0;  /* leading garbage strip: skip until first alpha token */

    /* Response buffer (non-stream) */
    char *response = NULL;
    int resp_len = 0;
    if (!stream) {
        response = calloc(max_tokens * 64 + 1, 1);
    } else {
        send_sse_start(client);
    }

    for (int t = 0; t < max_tokens; t++) {
        /* Repetition penalty */
        if (rep_penalty != 1.0f) {
            for (int g = 0; g < gen_count; g++) {
                int id = gen_ids[g];
                if (logits[id] > 0) logits[id] /= rep_penalty;
                else                 logits[id] *= rep_penalty;
            }
        }

        /* Token quality bias */
        apply_token_quality(logits, token_quality, model->vocab_size, quality_alpha);

        /* Entropy monitoring — detect degeneration */
        if (entropy_threshold > 0.0f && t >= min_tokens) {
            float H = compute_entropy(logits, model->vocab_size,
                                      sp.top_k > 0 ? sp.top_k : 64, sp.temperature);
            if (H > entropy_threshold) {
                high_entropy_count++;
                if (high_entropy_count >= 3) {
                    printf("  Entropy stop: H=%.2f (threshold=%.1f) at token %d\n",
                           H, entropy_threshold, t);
                    break;
                }
            } else {
                high_entropy_count = 0;
            }
        }

        int next;
        if (sp.temperature <= 0)
            next = argmax_fn(logits, model->vocab_size);
        else
            next = sample_advanced(logits, model->vocab_size, sp);

        /* EOS check */
        if ((next == tk->eos_id || next == 3 || next == 6) && t >= min_tokens) break;
        if (next == tk->eos_id || next == 3 || next == 6) {
            logits[next] = -1e30f;
            next = sample_advanced(logits, model->vocab_size, sp);
        }
        gen_ids[gen_count++] = next;

        /* Skip special tokens */
        const char *tok_text = tokenizer_decode(tk, next);
        if (next <= 9 || (tok_text[0] == '<' && tok_text[1] == '|')) {
            free(logits); logits = model_forward(model, cache, next, pos); pos++;
            continue;
        }

        char decoded[256];
        decode_token_str(tok_text, decoded, sizeof(decoded));

        /* Degeneration pattern stop — signals the useful response ended */
        if (t >= min_tokens) {
            if ((decoded[0] == '[' && decoded[1] == '[') ||
                (decoded[0] == '-' && decoded[1] == '-' && decoded[2] == '-') ||
                (decoded[0] == '#' && decoded[1] == '#') ||
                (decoded[0] == '*' && decoded[1] == '*' && (decoded[2] == '*' || decoded[2] == '[')) ||
                (decoded[0] == '<' && decoded[1] == '<' && decoded[2] == '<') ||
                (decoded[0] == '>' && decoded[1] == '>' && decoded[2] == '>') ||
                (strstr(decoded, "**Note") != NULL) ||
                (strstr(decoded, "Note:") != NULL && decoded[0] == 'N') ||
                (strstr(decoded, "\"*") != NULL)) {
                printf("  Pattern stop: '%s' at token %d\n", decoded, t);
                break;
            }
        }

        /* Leading garbage strip — skip tokens before first alphabetic content */
        if (!got_first_word) {
            int has_alpha = 0;
            for (int ci = 0; decoded[ci]; ci++) {
                unsigned char ch = (unsigned char)decoded[ci];
                if ((ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z')) {
                    has_alpha = 1; break;
                }
            }
            if (!has_alpha) {
                free(logits); logits = model_forward(model, cache, next, pos); pos++;
                continue;
            }
            got_first_word = 1;
        }

        /* Consecutive newline stop */
        if (strcmp(decoded, "\n") == 0 || strcmp(decoded, "\n\n") == 0)
            consecutive_newlines++;
        else
            consecutive_newlines = 0;
        if (consecutive_newlines >= 3 && t >= min_tokens) break;

        if (stream) {
            send_sse_token(client, decoded);
        } else {
            int dlen = (int)strlen(decoded);
            memcpy(response + resp_len, decoded, dlen);
            resp_len += dlen;
        }

        free(logits);
        logits = model_forward(model, cache, next, pos);
        pos++;
    }

    int tokens_generated = pos - n_tokens;

    if (stream) {
        send_sse_done(client);
    } else {
        response[resp_len] = '\0';

        /* Build JSON response */
        char *escaped = malloc(resp_len * 2 + 1);
        int ej = 0;
        for (int c = 0; c < resp_len; c++) {
            if (response[c] == '"') { escaped[ej++] = '\\'; escaped[ej++] = '"'; }
            else if (response[c] == '\\') { escaped[ej++] = '\\'; escaped[ej++] = '\\'; }
            else if (response[c] == '\n') { escaped[ej++] = '\\'; escaped[ej++] = 'n'; }
            else escaped[ej++] = response[c];
        }
        escaped[ej] = '\0';

        char *json_resp = malloc(ej + 256);
        snprintf(json_resp, ej + 256,
            "{\"text\":\"%s\",\"tokens_generated\":%d}", escaped, tokens_generated);
        send_response(client, 200, "OK", "application/json", json_resp);

        free(json_resp);
        free(escaped);
        free(response);
    }

    if (logits) free(logits);
    free(gen_ids);

    return tokens_generated;
}

/* ---- Main ---- */

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [port]\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    int port = argc > 2 ? atoi(argv[2]) : 8080;

    srand((unsigned)time(NULL));

#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
#endif

    printf("Loading GGUF: %s\n", model_path);
    fflush(stdout);
    gguf_file *gguf = gguf_open(model_path);
    if (!gguf) { fprintf(stderr, "Failed to open GGUF\n"); return 1; }

    printf("Loading model weights...\n");
    fflush(stdout);
    model_t *model = model_load(gguf);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    printf("Allocating KV cache (2048 tokens)...\n");
    fflush(stdout);
    kv_cache_t *cache = kv_cache_alloc(model, 2048);

    tokenizer_t *tk = tokenizer_from_gguf(gguf);
    printf("Tokenizer: %d tokens, %d merges, BOS=%d, EOS=%d\n",
           tk->vocab_size, tk->n_merges, tk->bos_id, tk->eos_id);
    printf("Architecture: %s\n", model->architecture);
    printf("Logit scale: %.6f\n", model->logit_scale);

    /* Verify special token IDs — show what the hardcoded IDs actually decode to */
    printf("Special token check:\n");
    for (int id = 0; id <= 9; id++) {
        const char *s = (id < tk->vocab_size && tk->tokens[id]) ? tk->tokens[id] : "(null)";
        printf("  ID %d = '%s'\n", id, s);
    }
    /* Search for Cohere chat tokens in vocab */
    {
        const char *names[] = {"<|START_OF_TURN_TOKEN|>", "<|END_OF_TURN_TOKEN|>",
                               "<|USER_TOKEN|>", "<|CHATBOT_TOKEN|>",
                               "<|SYSTEM_TOKEN|>"};
        for (int n = 0; n < 5; n++) {
            int id = tok_lookup(tk, names[n]);
            printf("  '%s' -> ID %d\n", names[n], id);
        }
    }
    fflush(stdout);

    /* Build token quality vector */
    token_quality = build_token_quality(tk);
    fflush(stdout);

    /* Quick tokenizer test */
    {
        int n_tok;
        int *ids = tokenizer_encode(tk, "Hello world", 1, &n_tok);
        printf("Test encode 'Hello world': %d tokens [", n_tok);
        for (int i = 0; i < n_tok; i++) printf("%s%d", i ? "," : "", ids[i]);
        printf("]\n");
        fflush(stdout);
        free(ids);
    }

    /* Create server socket */
    SOCKET server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == INVALID_SOCKET) {
        fprintf(stderr, "socket() failed\n"); return 1;
    }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, (const char *)&opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "bind() failed on port %d\n", port); return 1;
    }
    listen(server_fd, 5);
    printf("\n=== Aya inference server on http://localhost:%d ===\n\n", port);
    fflush(stdout);

    char *req_buf = malloc(65536);

    while (1) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        SOCKET client = accept(server_fd, (struct sockaddr *)&client_addr, &client_len);
        if (client == INVALID_SOCKET) continue;

        int total = 0;
        int r;
        while ((r = recv(client, req_buf + total, 65536 - total - 1, 0)) > 0) {
            total += r;
            req_buf[total] = '\0';
            if (strstr(req_buf, "\r\n\r\n")) {
                const char *cl = strstr(req_buf, "Content-Length:");
                if (!cl) cl = strstr(req_buf, "content-length:");
                if (cl) {
                    int content_len = atoi(cl + 15);
                    const char *body_start = strstr(req_buf, "\r\n\r\n") + 4;
                    int body_received = total - (int)(body_start - req_buf);
                    if (body_received >= content_len) break;
                } else {
                    break;
                }
            }
        }

        if (total <= 0) { CLOSE_SOCKET(client); continue; }

        char method[16], path[256];
        sscanf(req_buf, "%15s %255s", method, path);

        if (strcmp(method, "OPTIONS") == 0) {
            send_response(client, 204, "No Content", "text/plain", "");
            CLOSE_SOCKET(client);
            continue;
        }

        if (strcmp(path, "/health") == 0) {
            char health[256];
            snprintf(health, sizeof(health),
                "{\"status\":\"ok\",\"model\":\"aya-engine\",\"architecture\":\"%s\"}",
                model->architecture);
            send_response(client, 200, "OK", "application/json", health);
            CLOSE_SOCKET(client);
            continue;
        }

        if (strcmp(path, "/generate") == 0 && strcmp(method, "POST") == 0 && generating) {
            send_response(client, 503, "Service Busy", "application/json",
                "{\"error\":\"generation in progress, try again later\"}");
            CLOSE_SOCKET(client);
            continue;
        }

        if (strcmp(path, "/generate") == 0 && strcmp(method, "POST") == 0) {
            generating = 1;
            const char *body = strstr(req_buf, "\r\n\r\n");
            if (!body) { CLOSE_SOCKET(client); generating = 0; continue; }
            body += 4;

            char prompt[32768];
            if (!json_get_string(body, "prompt", prompt, sizeof(prompt))) {
                send_response(client, 400, "Bad Request", "application/json",
                    "{\"error\":\"missing prompt\"}");
                CLOSE_SOCKET(client);
                generating = 0;
                continue;
            }

            int max_tokens = json_get_int(body, "max_tokens", 256);
            float temperature = json_get_float(body, "temperature", 0.7f);
            int top_k = json_get_int(body, "top_k", 40);
            float top_p = json_get_float(body, "top_p", 0.9f);
            float min_p = json_get_float(body, "min_p", 0.05f);
            int min_tokens = json_get_int(body, "min_tokens", 8);
            int stream = json_get_int(body, "stream", 0);
            float rep_penalty = json_get_float(body, "repetition_penalty", 1.15f);
            float quality_alpha = json_get_float(body, "quality_alpha", 1.0f);
            float entropy_threshold = json_get_float(body, "entropy_threshold", 4.5f);

            sample_params_t sp = { top_k, top_p, min_p, temperature };

            printf("Generate: (max=%d, temp=%.2f, topk=%d, topp=%.2f, minp=%.2f, rep=%.2f, qa=%.2f, ent=%.1f)\n",
                   max_tokens, temperature, top_k, top_p, min_p, rep_penalty, quality_alpha, entropy_threshold);
            fflush(stdout);

            /* Reset KV cache */
            memset(cache->key_cache, 0,
                   (size_t)cache->num_layers * cache->max_seq * cache->kv_dim * sizeof(float));
            memset(cache->value_cache, 0,
                   (size_t)cache->num_layers * cache->max_seq * cache->kv_dim * sizeof(float));

            /* Build prompt with architecture-appropriate template */
            int n_tokens;
            int *input_ids = build_prompt(tk, model->architecture, prompt, &n_tokens);

            int gen = generate_tokens(model, cache, tk, input_ids, n_tokens,
                                      sp, max_tokens, min_tokens, rep_penalty,
                                      quality_alpha, entropy_threshold,
                                      stream, client);

            free(input_ids);
            generating = 0;
            printf("  Generated %d tokens\n", gen);
            fflush(stdout);

        } else if (strcmp(path, "/generate") == 0) {
            send_response(client, 405, "Method Not Allowed", "application/json",
                "{\"error\":\"POST required\"}");
        } else {
            send_response(client, 404, "Not Found", "application/json",
                "{\"error\":\"not found\"}");
        }

        CLOSE_SOCKET(client);
    }

    free(req_buf);
    kv_cache_free(cache);
    model_free(model);

#ifdef _WIN32
    WSACleanup();
#endif
    return 0;
}
