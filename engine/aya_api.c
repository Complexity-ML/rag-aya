/**
 * Aya inference engine — clean C API implementation.
 *
 * Consolidates tokenizer, sampling, chat template, and decode logic
 * from main.c into a library-friendly interface with no HTTP or
 * platform-specific dependencies.
 *
 * INL 2025
 */
#include "aya_api.h"
#include "gguf.h"
#include "model.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ================================================================
 * Hash table for token<->id lookup
 * ================================================================ */

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
    ht_entry *e = (ht_entry *)malloc(sizeof(ht_entry));
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

static void ht_free(hashtable *ht) {
    for (int i = 0; i < HT_SIZE; i++) {
        ht_entry *e = ht->buckets[i];
        while (e) {
            ht_entry *next = e->next;
            free(e->key);
            free(e);
            e = next;
        }
        ht->buckets[i] = NULL;
    }
}

/* ================================================================
 * Tokenizer
 * ================================================================ */

typedef struct {
    char     **tokens;       /* borrowed from gguf — do not free */
    int        vocab_size;
    int        bos_id;
    int        eos_id;
    hashtable  tok2id;
    /* BPE merge rules */
    char     **merges;       /* borrowed from gguf */
    int        n_merges;
    hashtable  merge_rank;   /* "a\x01b" -> rank */
} tokenizer_t;

static tokenizer_t *tokenizer_from_gguf(gguf_file *gguf) {
    tokenizer_t *tk = (tokenizer_t *)calloc(1, sizeof(tokenizer_t));
    tk->tokens     = gguf->vocab_tokens;
    tk->vocab_size = gguf->vocab_size;
    tk->bos_id     = gguf->bos_id;
    tk->eos_id     = gguf->eos_id;

    /* Build token->id hash table */
    for (int i = 0; i < tk->vocab_size; i++) {
        if (gguf->vocab_tokens[i]) {
            ht_put(&tk->tok2id, gguf->vocab_tokens[i], i);
        }
    }

    /* Build merge rank table */
    tk->merges   = gguf->merges;
    tk->n_merges = gguf->n_merges;
    if (tk->n_merges > 0) {
        for (int i = 0; i < tk->n_merges; i++) {
            char *m = tk->merges[i];
            char key[256];
            char *sp = strchr(m, ' ');
            if (!sp) continue;
            int la = (int)(sp - m);
            int lb = (int)strlen(sp + 1);
            if (la + 1 + lb >= 255) continue;
            memcpy(key, m, la);
            key[la] = '\x01';
            memcpy(key + la + 1, sp + 1, lb);
            key[la + 1 + lb] = '\0';
            ht_put(&tk->merge_rank, key, i);
        }
    }
    return tk;
}

static void tokenizer_free(tokenizer_t *tk) {
    if (!tk) return;
    ht_free(&tk->tok2id);
    ht_free(&tk->merge_rank);
    free(tk);
}

static int tok_lookup(tokenizer_t *tk, const char *s) {
    int id;
    if (ht_get(&tk->tok2id, s, &id)) return id;
    return -1;
}

/* ================================================================
 * BPE encode
 * ================================================================ */

static int *tokenizer_encode(tokenizer_t *tk, const char *text,
                              int add_bos, int *out_len) {
    int cap = 4096;
    int *ids = (int *)malloc(cap * sizeof(int));
    int n = 0;

    if (add_bos) ids[n++] = tk->bos_id;

    int text_len = (int)strlen(text);
    int i = 0;
    while (i < text_len) {
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

            /* Try with sentencepiece space prefix */
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
            if (n >= cap) { cap *= 2; ids = (int *)realloc(ids, cap * sizeof(int)); }
            ids[n++] = best_id;
            i += best_len;
        } else {
            unsigned char byte = (unsigned char)text[i];
            char bytename[16];
            snprintf(bytename, sizeof(bytename), "<0x%02X>", byte);
            int id = tok_lookup(tk, bytename);
            if (id >= 0) {
                if (n >= cap) { cap *= 2; ids = (int *)realloc(ids, cap * sizeof(int)); }
                ids[n++] = id;
            }
            i++;
        }
    }

    /* BPE merge pass */
    if (tk->n_merges > 0) {
        int changed = 1;
        while (changed) {
            changed = 0;
            int best_rank = tk->n_merges;
            int best_idx = -1;

            for (int j = 0; j < n - 1; j++) {
                const char *a = tk->tokens[ids[j]];
                const char *b = tk->tokens[ids[j + 1]];
                if (!a || !b) continue;
                int la = (int)strlen(a), lb = (int)strlen(b);
                if (la + 1 + lb >= 255) continue;

                char key[256];
                memcpy(key, a, la);
                key[la] = '\x01';
                memcpy(key + la + 1, b, lb);
                key[la + 1 + lb] = '\0';

                int rank;
                if (ht_get(&tk->merge_rank, key, &rank) && rank < best_rank) {
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

/* ================================================================
 * Token decode — GPT-2 BPE byte decoder
 * ================================================================ */

static const char *tokenizer_decode_raw(tokenizer_t *tk, int id) {
    if (id < 0 || id >= tk->vocab_size || !tk->tokens[id]) return "";
    return tk->tokens[id];
}

static void decode_token_str(const char *tok_str, char *out, int out_sz) {
    int di = 0, i = 0;
    while (tok_str[i] && di < out_sz - 1) {
        unsigned char c = (unsigned char)tok_str[i];
        uint32_t cp;
        int nbytes;

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

        if ((cp >= 33 && cp <= 126) || (cp >= 161 && cp <= 172) || (cp >= 174 && cp <= 255)) {
            out[di++] = (char)cp;
        } else if (cp >= 256) {
            int idx = (int)(cp - 256);
            uint8_t byte;
            if (idx <= 32)       byte = (uint8_t)idx;
            else if (idx <= 66)  byte = (uint8_t)(127 + idx - 33);
            else                 byte = 173;
            out[di++] = (char)byte;
        } else {
            for (int b = 0; b < nbytes && di < out_sz - 1; b++)
                out[di++] = tok_str[i + b];
        }
        i += nbytes;
    }
    out[di] = '\0';
}

/* ================================================================
 * Sampling
 * ================================================================ */

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
    int *indices = (int *)malloc(top_k * sizeof(int));
    float *vals  = (float *)malloc(top_k * sizeof(float));

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

/* ================================================================
 * Cohere chat template special token IDs
 * ================================================================ */

#define COHERE_BOS_ID         2
#define COHERE_EOS_ID         3
#define COHERE_START_TURN_ID  5
#define COHERE_END_TURN_ID    6
#define COHERE_USER_ID        7
#define COHERE_CHATBOT_ID     8

/* ================================================================
 * Context struct
 * ================================================================ */

struct aya_context {
    gguf_file   *gguf;
    model_t     *model;
    kv_cache_t  *cache;
    tokenizer_t *tokenizer;
    int          max_seq;       /* KV cache capacity */
    int          seeded;        /* whether srand has been called */
};

/* ================================================================
 * Internal: shared init from a parsed gguf_file
 * ================================================================ */

static aya_context *aya_init_from_gguf(gguf_file *gguf) {
    if (!gguf) return NULL;

    model_t *model = model_load(gguf);
    if (!model) { gguf_close(gguf); return NULL; }

    int max_seq = 2048;
    kv_cache_t *cache = kv_cache_alloc(model, max_seq);
    if (!cache) { model_free(model); gguf_close(gguf); return NULL; }

    tokenizer_t *tk = tokenizer_from_gguf(gguf);
    if (!tk) { kv_cache_free(cache); model_free(model); gguf_close(gguf); return NULL; }

    aya_context *ctx = (aya_context *)calloc(1, sizeof(aya_context));
    ctx->gguf      = gguf;
    ctx->model     = model;
    ctx->cache     = cache;
    ctx->tokenizer = tk;
    ctx->max_seq   = max_seq;
    ctx->seeded    = 0;
    return ctx;
}

/* ================================================================
 * Public API
 * ================================================================ */

AYA_API aya_context *aya_init_file(const char *gguf_path) {
    if (!gguf_path) return NULL;
    gguf_file *gguf = gguf_open(gguf_path);
    return aya_init_from_gguf(gguf);
}

AYA_API aya_context *aya_init_buffer(const uint8_t *data, size_t size) {
    if (!data || size == 0) return NULL;
    gguf_file *gguf = gguf_open_buffer(data, size);
    return aya_init_from_gguf(gguf);
}

AYA_API char *aya_generate(aya_context *ctx,
                           const char *prompt,
                           int max_tokens,
                           float temperature,
                           int top_k,
                           aya_token_callback cb,
                           void *user_data) {
    if (!ctx || !prompt) return NULL;

    /* Lazy seed — once per context lifetime */
    if (!ctx->seeded) {
        srand((unsigned)time(NULL));
        ctx->seeded = 1;
    }

    model_t    *model = ctx->model;
    kv_cache_t *cache = ctx->cache;
    tokenizer_t *tk   = ctx->tokenizer;

    /* Reset KV cache for a fresh generation */
    memset(cache->key_cache, 0,
           (size_t)cache->num_layers * cache->max_seq * cache->kv_dim * sizeof(float));
    memset(cache->value_cache, 0,
           (size_t)cache->num_layers * cache->max_seq * cache->kv_dim * sizeof(float));

    /* Build Cohere chat template:
     *   BOS(2) + START_TURN(5) + USER(7) + <prompt tokens> +
     *   END_TURN(6) + START_TURN(5) + CHATBOT(8)
     */
    int raw_len;
    int *raw_ids = tokenizer_encode(tk, prompt, 0, &raw_len);  /* no BOS */
    int n_prompt = 3 + raw_len + 3;  /* prefix + prompt + suffix */
    int *input_ids = (int *)malloc(n_prompt * sizeof(int));
    if (!input_ids) { free(raw_ids); return NULL; }

    input_ids[0] = COHERE_BOS_ID;
    input_ids[1] = COHERE_START_TURN_ID;
    input_ids[2] = COHERE_USER_ID;
    memcpy(input_ids + 3, raw_ids, raw_len * sizeof(int));
    input_ids[3 + raw_len]     = COHERE_END_TURN_ID;
    input_ids[3 + raw_len + 1] = COHERE_START_TURN_ID;
    input_ids[3 + raw_len + 2] = COHERE_CHATBOT_ID;
    free(raw_ids);

    /* Prefill: run the prompt through the model */
    int pos = 0;
    float *logits = NULL;
    for (int i = 0; i < n_prompt; i++) {
        if (logits) free(logits);
        logits = model_forward(model, cache, input_ids[i], pos);
        pos++;
    }
    free(input_ids);

    if (!logits) return NULL;

    /* Autoregressive decode */
    size_t resp_cap = (size_t)max_tokens * 64 + 1;
    char *response = (char *)calloc(resp_cap, 1);
    int resp_len = 0;

    for (int t = 0; t < max_tokens; t++) {
        int next = (temperature <= 0.0f)
            ? argmax_fn(logits, model->vocab_size)
            : sample_topk(logits, model->vocab_size, top_k, temperature);

        /* Stop on EOS, <EOS_TOKEN>(3), or END_OF_TURN(6) */
        if (next == tk->eos_id || next == COHERE_EOS_ID || next == COHERE_END_TURN_ID)
            break;

        /* Skip special tokens: IDs 0-9 or <|...|> tokens */
        const char *tok_text = tokenizer_decode_raw(tk, next);
        if (next <= 9 || (tok_text[0] == '<' && tok_text[1] == '|')) {
            free(logits);
            logits = model_forward(model, cache, next, pos);
            pos++;
            continue;
        }

        /* Decode GPT-2 BPE bytes to raw UTF-8 */
        char decoded[256];
        decode_token_str(tok_text, decoded, sizeof(decoded));
        int dlen = (int)strlen(decoded);

        /* Append to response buffer */
        if ((size_t)(resp_len + dlen) >= resp_cap - 1) {
            resp_cap = resp_cap * 2 + dlen;
            response = (char *)realloc(response, resp_cap);
        }
        memcpy(response + resp_len, decoded, dlen);
        resp_len += dlen;
        response[resp_len] = '\0';

        /* Streaming callback */
        if (cb) {
            cb(decoded, user_data);
        }

        free(logits);
        logits = model_forward(model, cache, next, pos);
        pos++;
    }

    if (logits) free(logits);

    return response;
}

AYA_API void aya_free_string(char *s) {
    free(s);
}

AYA_API void aya_free(aya_context *ctx) {
    if (!ctx) return;
    tokenizer_free(ctx->tokenizer);
    kv_cache_free(ctx->cache);
    model_free(ctx->model);
    gguf_close(ctx->gguf);
    free(ctx);
}
