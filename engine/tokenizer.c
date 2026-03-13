/**
 * BPE tokenizer implementation.
 * INL 2025
 */
#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- Hash table ---- */

static unsigned int ht_hash(const char *s) {
    unsigned int h = 5381;
    while (*s) h = h * 33 + (unsigned char)*s++;
    return h & (HT_SIZE - 1);
}

void ht_put(hashtable *ht, const char *key, int value) {
    unsigned int idx = ht_hash(key);
    ht_entry *e = malloc(sizeof(ht_entry));
    e->key = strdup(key);
    e->value = value;
    e->next = ht->buckets[idx];
    ht->buckets[idx] = e;
}

int ht_get(hashtable *ht, const char *key, int *out) {
    unsigned int idx = ht_hash(key);
    for (ht_entry *e = ht->buckets[idx]; e; e = e->next) {
        if (strcmp(e->key, key) == 0) { *out = e->value; return 1; }
    }
    return 0;
}

/* ---- Tokenizer ---- */

tokenizer_t *tokenizer_from_gguf(gguf_file *gguf) {
    tokenizer_t *tk = calloc(1, sizeof(tokenizer_t));
    tk->tokens     = gguf->vocab_tokens;
    tk->vocab_size = gguf->vocab_size;
    tk->bos_id     = gguf->bos_id;
    tk->eos_id     = gguf->eos_id;

    printf("  Building vocab hash table (%d tokens)...\n", tk->vocab_size);
    for (int i = 0; i < tk->vocab_size; i++) {
        if (gguf->vocab_tokens[i]) {
            ht_put(&tk->tok2id, gguf->vocab_tokens[i], i);
        }
    }

    tk->merges = gguf->merges;
    tk->n_merges = gguf->n_merges;
    if (tk->n_merges > 0) {
        printf("  Building merge rank table (%d merges)...\n", tk->n_merges);
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

int tok_lookup(tokenizer_t *tk, const char *s) {
    int id;
    if (ht_get(&tk->tok2id, s, &id)) return id;
    return -1;
}

int *tokenizer_encode(tokenizer_t *tk, const char *text,
                      int add_bos, int *out_len) {
    int cap = 4096;
    int *ids = malloc(cap * sizeof(int));
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

            /* Sentencepiece space prefix */
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

const char *tokenizer_decode(tokenizer_t *tk, int id) {
    if (id < 0 || id >= tk->vocab_size || !tk->tokens[id]) return "";
    return tk->tokens[id];
}

/* ---- GPT-2 byte decoder ---- */

void decode_token_str(const char *tok_str, char *out, int out_sz) {
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
