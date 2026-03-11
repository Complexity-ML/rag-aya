/**
 * GGUF file reader — header only.
 *
 * Parses .gguf metadata and provides mmap'd tensor access.
 * INL 2025
 */
#ifndef GGUF_H
#define GGUF_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/* GGML tensor types */
enum ggml_type {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
};

/* Block sizes */
static const int GGML_BLOCK_SIZE[] = {
    [GGML_TYPE_F32]  = 4,   [GGML_TYPE_F16]  = 2,
    [GGML_TYPE_Q4_0] = 18,  [GGML_TYPE_Q4_1] = 20,
    [GGML_TYPE_Q5_0] = 22,  [GGML_TYPE_Q5_1] = 24,
    [GGML_TYPE_Q8_0] = 34,  [GGML_TYPE_Q8_1] = 36,
    [GGML_TYPE_Q2_K] = 256, [GGML_TYPE_Q3_K] = 256,
    [GGML_TYPE_Q4_K] = 144, [GGML_TYPE_Q5_K] = 176,
    [GGML_TYPE_Q6_K] = 210, [GGML_TYPE_Q8_K] = 292,
};

static const int GGML_GROUP_SIZE[] = {
    [GGML_TYPE_F32]  = 1,   [GGML_TYPE_F16]  = 1,
    [GGML_TYPE_Q4_0] = 32,  [GGML_TYPE_Q4_1] = 32,
    [GGML_TYPE_Q5_0] = 32,  [GGML_TYPE_Q5_1] = 32,
    [GGML_TYPE_Q8_0] = 32,  [GGML_TYPE_Q8_1] = 32,
    [GGML_TYPE_Q2_K] = 256, [GGML_TYPE_Q3_K] = 256,
    [GGML_TYPE_Q4_K] = 256, [GGML_TYPE_Q5_K] = 256,
    [GGML_TYPE_Q6_K] = 256, [GGML_TYPE_Q8_K] = 256,
};

#define GGUF_MAGIC 0x46554747
#define MAX_TENSORS 4096
#define MAX_NAME_LEN 256
#define MAX_DIMS 4

typedef struct {
    char     name[MAX_NAME_LEN];
    int      n_dims;
    int64_t  shape[MAX_DIMS];
    int      type; /* enum ggml_type */
    uint64_t offset;
    int64_t  n_elements;
    size_t   data_size;
} gguf_tensor_info;

typedef struct {
    /* File mapping */
    uint8_t *data;       /* mmap'd file data (or external buffer) */
    size_t   file_size;
    int      owns_data;  /* 1 = mmap'd (must unmap), 0 = external buffer */
    uint8_t *tensor_data; /* pointer to start of tensor data */

    /* Model config (parsed from metadata) */
    int      hidden_size;
    int      num_layers;
    int      num_heads;
    int      num_kv_heads;
    int      intermediate_size;
    int      vocab_size;
    int      context_length;
    float    rope_theta;
    float    logit_scale;

    /* Tensors */
    int              n_tensors;
    gguf_tensor_info tensors[MAX_TENSORS];

    /* Tokenizer vocab (pointers into mmap'd data) */
    char   **vocab_tokens;
    float   *vocab_scores;
    char   **merges;       /* BPE merge rules */
    int      n_merges;
    int      bos_id;
    int      eos_id;
} gguf_file;

/* Open and parse a GGUF file (uses mmap) */
gguf_file *gguf_open(const char *path);

/* Open and parse a GGUF from an in-memory buffer (no mmap, no filesystem).
 * The buffer must remain valid for the lifetime of the gguf_file.
 * Useful for WASM or embedded scenarios. */
gguf_file *gguf_open_buffer(const uint8_t *data, size_t size);

/* Find tensor by name, returns NULL if not found */
gguf_tensor_info *gguf_find_tensor(gguf_file *f, const char *name);

/* Get pointer to tensor data */
uint8_t *gguf_tensor_data(gguf_file *f, gguf_tensor_info *t);

/* Close and free */
void gguf_close(gguf_file *f);

#endif /* GGUF_H */
