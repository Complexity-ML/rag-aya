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
#include "quant.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef _WIN32
  #include <winsock2.h>
  #include <ws2tcpip.h>
  #pragma comment(lib, "ws2_32.lib")
  typedef int socklen_t;
  #define CLOSE_SOCKET closesocket
#else
  #include <unistd.h>
  #include <sys/socket.h>
  #include <netinet/in.h>
  #include <arpa/inet.h>
  #define CLOSE_SOCKET close
  typedef int SOCKET;
  #define INVALID_SOCKET -1
#endif

/* ---- Simple hash table for token->id lookup ---- */

#define HT_SIZE (1 << 19)  /* 512K buckets */

typedef struct ht_entry {
    char *key;
    int   value;
    struct ht_entry *next;
} ht_entry;

typedef struct {
    ht_entry *buckets[HT_SIZE];
} hashtable;

static unsigned int ht_hash(const char *s) {
    unsigned int h = 5381;
    while (*s) h = h * 33 + (unsigned char)*s++;
    return h & (HT_SIZE - 1);
}

static void ht_put(hashtable *ht, const char *key, int value) {
    unsigned int idx = ht_hash(key);
    ht_entry *e = malloc(sizeof(ht_entry));
    e->key = strdup(key);
    e->value = value;
    e->next = ht->buckets[idx];
    ht->buckets[idx] = e;
}

static int ht_get(hashtable *ht, const char *key, int *out) {
    unsigned int idx = ht_hash(key);
    for (ht_entry *e = ht->buckets[idx]; e; e = e->next) {
        if (strcmp(e->key, key) == 0) { *out = e->value; return 1; }
    }
    return 0;
}

/* ---- Tokenizer ---- */

typedef struct {
    char     **tokens;
    int        vocab_size;
    int        bos_id;
    int        eos_id;
    hashtable  tok2id;
    /* BPE merge rules: merge_a[i] + merge_b[i] -> merge_result[i], priority = i */
    char     **merges;      /* raw merge strings "token_a token_b" */
    int        n_merges;
    hashtable  merge_rank;  /* "token_a\x00token_b" -> rank (lower = higher priority) */
} tokenizer_t;

static tokenizer_t *tokenizer_from_gguf(gguf_file *gguf) {
    tokenizer_t *tk = calloc(1, sizeof(tokenizer_t));
    tk->tokens     = gguf->vocab_tokens;
    tk->vocab_size = gguf->vocab_size;
    tk->bos_id     = gguf->bos_id;
    tk->eos_id     = gguf->eos_id;

    /* Build token->id hash table */
    printf("  Building vocab hash table (%d tokens)...\n", tk->vocab_size);
    for (int i = 0; i < tk->vocab_size; i++) {
        if (gguf->vocab_tokens[i]) {
            ht_put(&tk->tok2id, gguf->vocab_tokens[i], i);
        }
    }

    /* Build merge rank table */
    tk->merges = gguf->merges;
    tk->n_merges = gguf->n_merges;
    if (tk->n_merges > 0) {
        printf("  Building merge rank table (%d merges)...\n", tk->n_merges);
        for (int i = 0; i < tk->n_merges; i++) {
            /* Merge format: "token_a token_b" — store as "token_a\x01token_b" */
            char *m = tk->merges[i];
            char key[256];
            /* Find first space */
            char *sp = strchr(m, ' ');
            if (!sp) continue;
            int la = (int)(sp - m);
            int lb = (int)strlen(sp + 1);
            if (la + 1 + lb >= 255) continue;
            memcpy(key, m, la);
            key[la] = '\x01';  /* separator */
            memcpy(key + la + 1, sp + 1, lb);
            key[la + 1 + lb] = '\0';
            ht_put(&tk->merge_rank, key, i);
        }
    }

    return tk;
}

static int tok_lookup(tokenizer_t *tk, const char *s) {
    int id;
    if (ht_get(&tk->tok2id, s, &id)) return id;
    return -1;
}

/* BPE encode */
static int *tokenizer_encode(tokenizer_t *tk, const char *text,
                              int add_bos, int *out_len) {
    int cap = 4096;
    int *ids = malloc(cap * sizeof(int));
    int n = 0;

    if (add_bos) ids[n++] = tk->bos_id;

    /* Start with individual UTF-8 bytes/chars as tokens */
    int text_len = (int)strlen(text);
    int i = 0;
    while (i < text_len) {
        /* Try longest match first (up to 32 chars) */
        int best_len = 0, best_id = -1;
        int max_try = text_len - i;
        if (max_try > 32) max_try = 32;

        for (int len = max_try; len >= 1; len--) {
            char sub[64];
            if (len >= 64) continue;
            memcpy(sub, text + i, len);
            sub[len] = '\0';

            int id = tok_lookup(tk, sub);
            if (id >= 0) { best_len = len; best_id = id; break; }

            /* Try with sentencepiece space prefix (▁ = 0xE2 0x96 0x81) */
            if (i == 0 || (i > 0 && text[i - 1] == ' ')) {
                if (len + 3 < 64) {
                    char buf[64];
                    buf[0] = '\xe2'; buf[1] = '\x96'; buf[2] = '\x81';
                    memcpy(buf + 3, sub, len + 1);
                    id = tok_lookup(tk, buf);
                    if (id >= 0) { best_len = len; best_id = id; break; }
                }
            }
        }

        if (best_id >= 0) {
            if (n >= cap) { cap *= 2; ids = realloc(ids, cap * sizeof(int)); }
            ids[n++] = best_id;
            i += best_len;
        } else {
            /* Byte fallback */
            unsigned char byte = (unsigned char)text[i];
            char bytename[16];
            snprintf(bytename, sizeof(bytename), "<0x%02X>", byte);
            int id = tok_lookup(tk, bytename);
            if (id >= 0) {
                if (n >= cap) { cap *= 2; ids = realloc(ids, cap * sizeof(int)); }
                ids[n++] = id;
            }
            i++;
        }
    }

    /* BPE merge pass using merge ranks */
    if (tk->n_merges > 0) {
        int changed = 1;
        while (changed) {
            changed = 0;
            int best_rank = tk->n_merges; /* lower = higher priority */
            int best_idx = -1;

            for (int j = 0; j < n - 1; j++) {
                const char *a = tk->tokens[ids[j]];
                const char *b = tk->tokens[ids[j + 1]];
                if (!a || !b) continue;
                int la = (int)strlen(a), lb = (int)strlen(b);
                if (la + 1 + lb >= 255) continue;

                /* Build merge key "a\x01b" */
                char key[256];
                memcpy(key, a, la);
                key[la] = '\x01';
                memcpy(key + la + 1, b, lb);
                key[la + 1 + lb] = '\0';

                int rank;
                if (ht_get(&tk->merge_rank, key, &rank) && rank < best_rank) {
                    /* Check merged token exists in vocab */
                    char merged[256];
                    memcpy(merged, a, la);
                    memcpy(merged + la, b, lb);
                    merged[la + lb] = '\0';
                    int mid = tok_lookup(tk, merged);
                    if (mid >= 0) {
                        best_rank = rank;
                        best_idx = j;
                    }
                }
            }

            if (best_idx >= 0) {
                const char *a = tk->tokens[ids[best_idx]];
                const char *b = tk->tokens[ids[best_idx + 1]];
                char merged[256];
                int la = (int)strlen(a), lb = (int)strlen(b);
                memcpy(merged, a, la);
                memcpy(merged + la, b, lb);
                merged[la + lb] = '\0';
                ids[best_idx] = tok_lookup(tk, merged);
                memmove(ids + best_idx + 1, ids + best_idx + 2,
                        (n - best_idx - 2) * sizeof(int));
                n--;
                changed = 1;
            }
        }
    }

    *out_len = n;
    return ids;
}

/* Decode a single token to string (replaces ▁ with space) */
static const char *tokenizer_decode(tokenizer_t *tk, int id) {
    if (id < 0 || id >= tk->vocab_size || !tk->tokens[id]) return "";
    return tk->tokens[id];
}

/* ---- Sampling ---- */

static int argmax_fn(const float *logits, int n) {
    int best = 0;
    float best_val = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > best_val) { best_val = logits[i]; best = i; }
    }
    return best;
}

static int sample_topk(const float *logits, int vocab_size,
                        int top_k, float temperature) {
    int *indices = malloc(top_k * sizeof(int));
    float *vals  = malloc(top_k * sizeof(float));

    for (int i = 0; i < top_k; i++) { indices[i] = -1; vals[i] = -1e30f; }

    for (int v = 0; v < vocab_size; v++) {
        float val = logits[v];
        if (val > vals[top_k - 1]) {
            vals[top_k - 1] = val;
            indices[top_k - 1] = v;
            for (int j = top_k - 1; j > 0 && vals[j] > vals[j - 1]; j--) {
                float tv = vals[j]; vals[j] = vals[j - 1]; vals[j - 1] = tv;
                int ti = indices[j]; indices[j] = indices[j - 1]; indices[j - 1] = ti;
            }
        }
    }

    float max_val = vals[0];
    float sum = 0.0f;
    for (int i = 0; i < top_k && indices[i] >= 0; i++) {
        vals[i] = expf((vals[i] - max_val) / temperature);
        sum += vals[i];
    }
    for (int i = 0; i < top_k && indices[i] >= 0; i++) vals[i] /= sum;

    float r = (float)rand() / (float)RAND_MAX;
    float cum = 0.0f;
    int result = indices[0];
    for (int i = 0; i < top_k && indices[i] >= 0; i++) {
        cum += vals[i];
        if (r <= cum) { result = indices[i]; break; }
    }

    free(indices);
    free(vals);
    return result;
}

/* ---- JSON helpers (minimal) ---- */

static const char *json_get_string(const char *json, const char *key, char *buf, int buf_sz) {
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(json, pattern);
    if (!p) return NULL;
    p += strlen(pattern);
    while (*p == ' ' || *p == ':' || *p == '\t') p++;
    if (*p != '"') return NULL;
    p++;
    int i = 0;
    while (*p && *p != '"' && i < buf_sz - 1) {
        if (*p == '\\' && *(p + 1)) {
            p++;
            if (*p == 'n') buf[i++] = '\n';
            else if (*p == 't') buf[i++] = '\t';
            else if (*p == '"') buf[i++] = '"';
            else if (*p == '\\') buf[i++] = '\\';
            else buf[i++] = *p;
        } else {
            buf[i++] = *p;
        }
        p++;
    }
    buf[i] = '\0';
    return buf;
}

static int json_get_int(const char *json, const char *key, int default_val) {
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(json, pattern);
    if (!p) return default_val;
    p += strlen(pattern);
    while (*p == ' ' || *p == ':' || *p == '\t') p++;
    return atoi(p);
}

static float json_get_float(const char *json, const char *key, float default_val) {
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(json, pattern);
    if (!p) return default_val;
    p += strlen(pattern);
    while (*p == ' ' || *p == ':' || *p == '\t') p++;
    return (float)atof(p);
}

/* ---- GPT-2 BPE byte decoder ----
 * GPT-2 tokenizers map bytes to unicode codepoints:
 *   "safe" bytes (33-126, 161-172, 174-255) → identity
 *   other bytes (0-32, 127-160, 173) → U+0100 + index
 * We reverse this to get raw bytes back. */

static void decode_token_str(const char *tok_str, char *out, int out_sz) {
    int di = 0, i = 0;
    while (tok_str[i] && di < out_sz - 1) {
        unsigned char c = (unsigned char)tok_str[i];
        uint32_t cp;
        int nbytes;

        /* Parse UTF-8 codepoint */
        if (c < 0x80) {
            cp = c; nbytes = 1;
        } else if ((c & 0xE0) == 0xC0 && tok_str[i+1]) {
            cp = ((c & 0x1F) << 6) | ((unsigned char)tok_str[i+1] & 0x3F);
            nbytes = 2;
        } else if ((c & 0xF0) == 0xE0 && tok_str[i+1] && tok_str[i+2]) {
            cp = ((c & 0x0F) << 12) | (((unsigned char)tok_str[i+1] & 0x3F) << 6)
                 | ((unsigned char)tok_str[i+2] & 0x3F);
            nbytes = 3;
        } else {
            out[di++] = tok_str[i++];
            continue;
        }

        /* Reverse GPT-2 byte mapping */
        if ((cp >= 33 && cp <= 126) || (cp >= 161 && cp <= 172) || (cp >= 174 && cp <= 255)) {
            /* Identity-mapped "safe" bytes */
            out[di++] = (char)cp;
        } else if (cp >= 256) {
            /* Remapped bytes: 0-32 → U+0100..U+0120, 127-160 → U+0121..U+0142, 173 → U+0143 */
            int idx = (int)(cp - 256);
            uint8_t byte;
            if (idx <= 32)       byte = (uint8_t)idx;          /* bytes 0-32 (space=32) */
            else if (idx <= 66)  byte = (uint8_t)(127 + idx - 33); /* bytes 127-160 */
            else                 byte = 173;                    /* byte 173 */
            out[di++] = (char)byte;
        } else {
            /* Shouldn't happen in GPT-2 vocab, pass through as UTF-8 */
            for (int b = 0; b < nbytes && di < out_sz - 1; b++)
                out[di++] = tok_str[i + b];
        }
        i += nbytes;
    }
    out[di] = '\0';
}

/* ---- HTTP server ---- */

static void send_response(SOCKET sock, int status, const char *status_text,
                           const char *content_type, const char *body) {
    char header[512];
    int body_len = (int)strlen(body);
    snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %d\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: POST, GET, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type\r\n"
        "Connection: close\r\n"
        "\r\n",
        status, status_text, content_type, body_len);
    send(sock, header, (int)strlen(header), 0);
    send(sock, body, body_len, 0);
}

static void send_sse_start(SOCKET sock) {
    const char *header =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: keep-alive\r\n"
        "\r\n";
    send(sock, header, (int)strlen(header), 0);
}

static void send_sse_token(SOCKET sock, const char *token) {
    char buf[512];
    char escaped[256];
    int j = 0;
    for (int i = 0; token[i] && j < 250; i++) {
        if (token[i] == '"') { escaped[j++] = '\\'; escaped[j++] = '"'; }
        else if (token[i] == '\\') { escaped[j++] = '\\'; escaped[j++] = '\\'; }
        else if (token[i] == '\n') { escaped[j++] = '\\'; escaped[j++] = 'n'; }
        else escaped[j++] = token[i];
    }
    escaped[j] = '\0';
    snprintf(buf, sizeof(buf), "data: {\"token\":\"%s\"}\n\n", escaped);
    send(sock, buf, (int)strlen(buf), 0);
}

static void send_sse_done(SOCKET sock) {
    const char *msg = "data: [DONE]\n\n";
    send(sock, msg, (int)strlen(msg), 0);
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

    /* Dump some token strings to diagnose BPE encoding */
    {
        printf("\n=== Token dump (space-related) ===\n");
        /* Look for tokens containing common words with spaces */
        int sample_ids[] = {31621, 43791, 5, 10, 100, 500, 1000};
        int n_samples = sizeof(sample_ids) / sizeof(sample_ids[0]);
        for (int s = 0; s < n_samples; s++) {
            int id = sample_ids[s];
            if (id < tk->vocab_size && tk->tokens[id]) {
                const char *t = tk->tokens[id];
                printf("  token[%d] = \"", id);
                for (int c = 0; t[c]; c++) printf("%c", t[c]);
                printf("\" hex=[");
                for (int c = 0; t[c]; c++) printf("%02X ", (unsigned char)t[c]);
                printf("]\n");
            }
        }
        /* Check Cohere special tokens */
        const char *specials[] = {
            "<|START_OF_TURN_TOKEN|>", "<|END_OF_TURN_TOKEN|>",
            "<|USER_TOKEN|>", "<|CHATBOT_TOKEN|>", "<|SYSTEM_TOKEN|>",
            "<BOS_TOKEN>", "<EOS_TOKEN>"
        };
        for (int p = 0; p < 7; p++) {
            int id = tok_lookup(tk, specials[p]);
            printf("  lookup(\"%s\") = %d\n", specials[p], id);
        }
        printf("=================================\n\n");
        fflush(stdout);
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
            send_response(client, 200, "OK", "application/json",
                "{\"status\":\"ok\",\"model\":\"aya-integer\"}");
            CLOSE_SOCKET(client);
            continue;
        }

        if (strcmp(path, "/generate") == 0 && strcmp(method, "POST") == 0) {
            const char *body = strstr(req_buf, "\r\n\r\n");
            if (!body) { CLOSE_SOCKET(client); continue; }
            body += 4;

            char prompt[4096];
            if (!json_get_string(body, "prompt", prompt, sizeof(prompt))) {
                send_response(client, 400, "Bad Request", "application/json",
                    "{\"error\":\"missing prompt\"}");
                CLOSE_SOCKET(client);
                continue;
            }

            int max_tokens = json_get_int(body, "max_tokens", 256);
            float temperature = json_get_float(body, "temperature", 0.7f);
            int top_k = json_get_int(body, "top_k", 40);
            int stream = json_get_int(body, "stream", 0);

            printf("Generate: \"%s\" (max=%d, temp=%.2f, topk=%d)\n",
                   prompt, max_tokens, temperature, top_k);
            fflush(stdout);

            /* Reset KV cache for new request */
            memset(cache->key_cache, 0,
                   (size_t)cache->num_layers * cache->max_seq * cache->kv_dim * sizeof(float));
            memset(cache->value_cache, 0,
                   (size_t)cache->num_layers * cache->max_seq * cache->kv_dim * sizeof(float));

            /* Build Cohere chat template:
             * BOS(2) + START_TURN(5) + USER(7) + <prompt tokens> +
             * END_TURN(6) + START_TURN(5) + CHATBOT(8) */
            int raw_len;
            int *raw_ids = tokenizer_encode(tk, prompt, 0, &raw_len);  /* no BOS */
            int n_tokens = 3 + raw_len + 3;  /* prefix + prompt + suffix */
            int *input_ids = malloc(n_tokens * sizeof(int));
            input_ids[0] = 2;  /* BOS */
            input_ids[1] = 5;  /* START_OF_TURN */
            input_ids[2] = 7;  /* USER */
            memcpy(input_ids + 3, raw_ids, raw_len * sizeof(int));
            input_ids[3 + raw_len]     = 6;  /* END_OF_TURN */
            input_ids[3 + raw_len + 1] = 5;  /* START_OF_TURN */
            input_ids[3 + raw_len + 2] = 8;  /* CHATBOT */
            free(raw_ids);
            printf("  Encoded %d tokens (with chat template)\n", n_tokens);
            fflush(stdout);

            int pos = 0;
            float *logits = NULL;
            for (int i = 0; i < n_tokens; i++) {
                if (logits) free(logits);
                logits = model_forward(model, cache, input_ids[i], pos);
                pos++;
            }

            if (stream) {
                send_sse_start(client);
                for (int t = 0; t < max_tokens; t++) {
                    int next = (temperature <= 0)
                        ? argmax_fn(logits, model->vocab_size)
                        : sample_topk(logits, model->vocab_size, top_k, temperature);

                    /* Stop on EOS or special control tokens */
                    if (next == tk->eos_id || next == 3 || next == 6) break;
                    /* Skip special tokens in output (IDs 0-9 or <|...|> tokens) */
                    const char *tok_text = tokenizer_decode(tk, next);
                    if (next <= 9 || (tok_text[0] == '<' && tok_text[1] == '|')) {
                        free(logits); logits = model_forward(model, cache, next, pos); pos++; continue;
                    }

                    char decoded[256];
                    decode_token_str(tokenizer_decode(tk, next), decoded, sizeof(decoded));
                    send_sse_token(client, decoded);

                    free(logits);
                    logits = model_forward(model, cache, next, pos);
                    pos++;
                }
                send_sse_done(client);
            } else {
                char *response = calloc(max_tokens * 64 + 1, 1);
                int resp_len = 0;

                for (int t = 0; t < max_tokens; t++) {
                    int next = (temperature <= 0)
                        ? argmax_fn(logits, model->vocab_size)
                        : sample_topk(logits, model->vocab_size, top_k, temperature);

                    if (next == tk->eos_id || next == 3 || next == 6) break;
                    if (next <= 9) { free(logits); logits = model_forward(model, cache, next, pos); pos++; continue; }

                    char decoded[256];
                    decode_token_str(tokenizer_decode(tk, next), decoded, sizeof(decoded));
                    int dlen = (int)strlen(decoded);
                    memcpy(response + resp_len, decoded, dlen);
                    resp_len += dlen;

                    free(logits);
                    logits = model_forward(model, cache, next, pos);
                    pos++;
                }
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
                    "{\"text\":\"%s\",\"tokens_generated\":%d}", escaped, pos - n_tokens);
                send_response(client, 200, "OK", "application/json", json_resp);

                free(json_resp);
                free(escaped);
                free(response);
            }

            if (logits) free(logits);
            free(input_ids);
            printf("  Generated %d tokens\n", pos - n_tokens);
            fflush(stdout);

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
