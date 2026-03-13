/**
 * Transformer model — forward pass with quantized weights.
 * INL 2025
 */
#ifndef MODEL_H
#define MODEL_H

#include "gguf.h"
#include <stdint.h>

typedef struct {
    /* Quantized weight pointers (into mmap'd data) */
    uint8_t *q_proj, *k_proj, *v_proj, *o_proj;
    uint8_t *gate_proj, *up_proj, *down_proj;
    int q_type, k_type, v_type, o_type;       /* per-tensor types */
    int gate_type, up_type, down_type;

    /* Combined QKV (GPT-2 style) — NULL if separate Q/K/V */
    uint8_t *qkv_proj;
    int qkv_type;

    /* Bias terms (dequantized to f32, NULL if absent) */
    float *q_bias, *k_bias, *v_bias, *o_bias;
    float *up_bias, *down_bias;
    float *qkv_bias;

    /* Norm weights (dequantized to f32) */
    float *attn_norm;
    float *ffn_norm;
    float *attn_norm_bias;   /* NULL for RMSNorm (no bias) */
    float *ffn_norm_bias;
} layer_weights;

typedef struct {
    int hidden_size, num_layers, num_heads, num_kv_heads;
    int intermediate_size, vocab_size, head_dim, kv_dim;
    float rope_theta;
    float logit_scale;

    /* Architecture: "cohere2", "llama", "gpt2", etc. */
    char architecture[64];

    /* Token embedding — kept quantized, dequant per-row on demand */
    uint8_t *embed_data;     /* raw quantized data */
    int      embed_type;     /* ggml type */
    size_t   embed_row_bytes;/* bytes per row */

    /* Position embedding (GPT-2) — NULL for RoPE models */
    uint8_t *pos_embed_data;
    int      pos_embed_type;
    size_t   pos_embed_row_bytes;

    /* Output projection — quantized or tied to embed */
    uint8_t *output_data;
    int      output_type;
    size_t   output_row_bytes;
    int      output_tied;    /* 1 if output == embed */

    /* Final norm (always f32) */
    float *output_norm;      /* [hidden_size] */
    float *output_norm_bias; /* NULL for RMSNorm */

    /* File bounds (for OOB detection) */
    uint8_t *file_data;
    size_t   file_size;

    layer_weights *layers;
} model_t;

typedef struct {
    float *key_cache;   /* [num_layers * max_seq * kv_dim] */
    float *value_cache;
    int max_seq;
    int kv_dim;
    int num_layers;
} kv_cache_t;

/* Load model from parsed GGUF */
model_t *model_load(gguf_file *gguf);
void     model_free(model_t *m);

/* Allocate KV cache */
kv_cache_t *kv_cache_alloc(model_t *m, int max_seq);
void        kv_cache_free(kv_cache_t *c);

/* Forward one token, returns logits[vocab_size] (caller must free) */
float *model_forward(model_t *m, kv_cache_t *cache, int token, int pos);

#endif /* MODEL_H */
