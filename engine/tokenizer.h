/**
 * BPE tokenizer — supports Cohere (sentencepiece) and GPT-2 byte-level BPE.
 * INL 2025
 */
#ifndef TOKENIZER_H
#define TOKENIZER_H

#include "gguf.h"

/* Simple hash table for token <-> id lookups */
#define HT_SIZE (1 << 19)  /* 512K buckets */

typedef struct ht_entry {
    char *key;
    int   value;
    struct ht_entry *next;
} ht_entry;

typedef struct {
    ht_entry *buckets[HT_SIZE];
} hashtable;

void ht_put(hashtable *ht, const char *key, int value);
int  ht_get(hashtable *ht, const char *key, int *out);

/* Tokenizer */
typedef struct {
    char     **tokens;
    int        vocab_size;
    int        bos_id;
    int        eos_id;
    hashtable  tok2id;
    char     **merges;
    int        n_merges;
    hashtable  merge_rank;
} tokenizer_t;

/* Build tokenizer from parsed GGUF vocab */
tokenizer_t *tokenizer_from_gguf(gguf_file *gguf);

/* Lookup a token string -> id (-1 if not found) */
int tok_lookup(tokenizer_t *tk, const char *s);

/* Encode text to token ids (caller must free result) */
int *tokenizer_encode(tokenizer_t *tk, const char *text,
                      int add_bos, int *out_len);

/* Decode a single token id to its vocab string */
const char *tokenizer_decode(tokenizer_t *tk, int id);

/* GPT-2 byte decoder: convert vocab token string to raw bytes.
 * Reverses the GPT-2 unicode byte mapping. */
void decode_token_str(const char *tok_str, char *out, int out_sz);

#endif /* TOKENIZER_H */
