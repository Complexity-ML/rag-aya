/**
 * Transformer forward pass — integer-first inference.
 * INL 2025
 */
#include "model.h"
#include "quant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Dequantize a small tensor (norms) to f32 */
static float *dequant_tensor(gguf_file *gguf, const char *name) {
    gguf_tensor_info *t = gguf_find_tensor(gguf, name);
    if (!t) { fprintf(stderr, "Missing tensor: %s\n", name); return NULL; }

    uint8_t *data = gguf_tensor_data(gguf, t);
    int n = (int)t->n_elements;
    float *out = malloc(n * sizeof(float));

    if (t->type == GGML_TYPE_F32) {
        memcpy(out, data, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        const uint16_t *fp16 = (const uint16_t *)data;
        for (int i = 0; i < n; i++) out[i] = f16_to_f32(fp16[i]);
    } else {
        fprintf(stderr, "TODO: dequant type %d for %s\n", t->type, name);
        memset(out, 0, n * sizeof(float));
    }
    return out;
}

/* Compute bytes per row for a given quant type and number of columns */
static size_t bytes_per_row(int type, int cols) {
    int gs = GGML_GROUP_SIZE[type];
    int bs = GGML_BLOCK_SIZE[type];
    return ((size_t)cols / gs) * bs;
}

/* Dequantize a single row from a quantized embedding matrix */
static void embed_lookup(uint8_t *data, int type, int row, int cols, float *out) {
    size_t rbytes = bytes_per_row(type, cols);
    uint8_t *row_data = data + (size_t)row * rbytes;

    if (type == GGML_TYPE_F32) {
        memcpy(out, row_data, cols * sizeof(float));
    } else if (type == GGML_TYPE_F16) {
        const uint16_t *fp16 = (const uint16_t *)row_data;
        for (int i = 0; i < cols; i++) out[i] = f16_to_f32(fp16[i]);
    } else if (type == GGML_TYPE_Q6_K) {
        dequant_q6k_row(row_data, out, cols);
    } else if (type == GGML_TYPE_Q4_K) {
        int n_blocks = cols / 256;
        for (int b = 0; b < n_blocks; b++) {
            const uint8_t *block = row_data + b * 144;
            float d    = f16_to_f32(*(const uint16_t *)(block));
            float dmin = f16_to_f32(*(const uint16_t *)(block + 2));
            const uint8_t *sc_data = block + 4;
            uint8_t scales[8], mins[8];
            for (int j = 0; j < 4; j++) {
                scales[j] = sc_data[j] & 0x3F;
                mins[j]   = sc_data[4+j] & 0x3F;
            }
            for (int j = 0; j < 4; j++) {
                scales[4+j] = (sc_data[8+j] & 0x0F) | ((sc_data[j] >> 6) << 4);
                mins[4+j]   = (sc_data[8+j] >>   4) | ((sc_data[4+j] >> 6) << 4);
            }
            const uint8_t *quants = block + 16;
            float *ob = out + b * 256;
            /* 4 groups of 64: low nibble = first 32, high nibble = next 32 */
            for (int j = 0; j < 4; j++) {
                float sc0 = d * scales[j*2];
                float m0  = dmin * mins[j*2];
                float sc1 = d * scales[j*2+1];
                float m1  = dmin * mins[j*2+1];
                const uint8_t *qj = quants + j * 32;
                float *oj = ob + j * 64;
                for (int l = 0; l < 32; l++) {
                    oj[l]      = sc0 * (qj[l] & 0xF) - m0;
                    oj[l + 32] = sc1 * (qj[l] >>  4) - m1;
                }
            }
        }
    } else {
        fprintf(stderr, "Unsupported embed type %d\n", type);
        memset(out, 0, cols * sizeof(float));
    }
}

model_t *model_load(gguf_file *gguf) {
    model_t *m = calloc(1, sizeof(model_t));

    m->hidden_size      = gguf->hidden_size;
    m->num_layers       = gguf->num_layers;
    m->num_heads        = gguf->num_heads;
    m->num_kv_heads     = gguf->num_kv_heads;
    m->intermediate_size = gguf->intermediate_size;
    m->vocab_size       = gguf->vocab_size;
    m->head_dim         = m->hidden_size / m->num_heads;
    m->kv_dim           = m->num_kv_heads * m->head_dim;
    m->rope_theta       = gguf->rope_theta;
    m->logit_scale      = gguf->logit_scale;

    m->file_data = gguf->data;
    m->file_size = gguf->file_size;

    printf("  Config: H=%d, L=%d, NH=%d, NKV=%d, INTER=%d, V=%d, HD=%d\n",
           m->hidden_size, m->num_layers, m->num_heads, m->num_kv_heads,
           m->intermediate_size, m->vocab_size, m->head_dim);

    /* Embeddings — keep quantized */
    gguf_tensor_info *emb_t = gguf_find_tensor(gguf, "token_embd.weight");
    if (emb_t) {
        m->embed_data = gguf_tensor_data(gguf, emb_t);
        m->embed_type = emb_t->type;
        m->embed_row_bytes = bytes_per_row(emb_t->type, m->hidden_size);
        printf("  Embed: type=%d, row_bytes=%zu\n", m->embed_type, m->embed_row_bytes);
    }

    /* Output norm (small, dequant to f32) */
    m->output_norm = dequant_tensor(gguf, "output_norm.weight");

    /* Output weight — keep quantized or use tied embedding */
    gguf_tensor_info *out_t = gguf_find_tensor(gguf, "output.weight");
    if (out_t) {
        m->output_data = gguf_tensor_data(gguf, out_t);
        m->output_type = out_t->type;
        m->output_row_bytes = bytes_per_row(out_t->type, m->hidden_size);
        m->output_tied = 0;
        printf("  Output: type=%d (separate)\n", m->output_type);
    } else {
        m->output_data = m->embed_data;
        m->output_type = m->embed_type;
        m->output_row_bytes = m->embed_row_bytes;
        m->output_tied = 1;
        printf("  Output: tied to embedding\n");
    }

    /* Layers */
    m->layers = calloc(m->num_layers, sizeof(layer_weights));
    for (int l = 0; l < m->num_layers; l++) {
        char name[128];
        layer_weights *lw = &m->layers[l];

        #define FIND_T(field, type_field, suffix) do { \
            snprintf(name, sizeof(name), "blk.%d." suffix, l); \
            gguf_tensor_info *ti = gguf_find_tensor(gguf, name); \
            if (ti) { lw->field = gguf_tensor_data(gguf, ti); \
                       lw->type_field = ti->type; } \
            else fprintf(stderr, "WARN: missing %s\n", name); \
        } while(0)

        FIND_T(q_proj, q_type, "attn_q.weight");
        FIND_T(k_proj, k_type, "attn_k.weight");
        FIND_T(v_proj, v_type, "attn_v.weight");
        FIND_T(o_proj, o_type, "attn_output.weight");
        FIND_T(gate_proj, gate_type, "ffn_gate.weight");
        FIND_T(up_proj,   up_type,   "ffn_up.weight");
        FIND_T(down_proj, down_type, "ffn_down.weight");

        if (l == 0 || l == m->num_layers - 1) {
            printf("  L%d types: q=%d k=%d v=%d o=%d gate=%d up=%d down=%d\n",
                   l, lw->q_type, lw->k_type, lw->v_type, lw->o_type,
                   lw->gate_type, lw->up_type, lw->down_type);
            fflush(stdout);
        }

        /* Norms (always dequant to f32 — tiny) */
        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", l);
        lw->attn_norm = dequant_tensor(gguf, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", l);
        gguf_tensor_info *ffn_norm_t = gguf_find_tensor(gguf, name);
        if (ffn_norm_t) {
            lw->ffn_norm = dequant_tensor(gguf, name);
        } else {
            /* Cohere2: parallel attn+FFN, shared norm */
            lw->ffn_norm = lw->attn_norm;
        }

        if (l % 4 == 0) {
            printf("  Layer %d/%d (q=%d gate=%d up=%d down=%d)\n",
                   l, m->num_layers, lw->q_type, lw->gate_type,
                   lw->up_type, lw->down_type);
        }
    }

    printf("Model loaded.\n");
    return m;
}

void model_free(model_t *m) {
    if (!m) return;
    free(m->output_norm);
    for (int l = 0; l < m->num_layers; l++) {
        free(m->layers[l].attn_norm);
        if (m->layers[l].ffn_norm != m->layers[l].attn_norm)
            free(m->layers[l].ffn_norm);
    }
    free(m->layers);
    free(m);
}

kv_cache_t *kv_cache_alloc(model_t *m, int max_seq) {
    kv_cache_t *c = calloc(1, sizeof(kv_cache_t));
    c->max_seq = max_seq;
    c->kv_dim = m->kv_dim;
    c->num_layers = m->num_layers;
    c->key_cache   = calloc((size_t)m->num_layers * max_seq * m->kv_dim, sizeof(float));
    c->value_cache = calloc((size_t)m->num_layers * max_seq * m->kv_dim, sizeof(float));
    return c;
}

void kv_cache_free(kv_cache_t *c) {
    if (!c) return;
    free(c->key_cache);
    free(c->value_cache);
    free(c);
}

/* ---- Math ops ---- */

static void rms_norm(float *out, const float *x, const float *w, int n) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / n + 1e-6f);
    for (int i = 0; i < n; i++) out[i] = x[i] * ss * w[i];
}

static void silu_inplace(float *x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

static void rope(float *q, float *k, int pos, int head_dim,
                 int n_heads, int n_kv_heads, float theta) {
    for (int h = 0; h < n_heads; h++) {
        float *qh = q + h * head_dim;
        for (int i = 0; i < head_dim; i += 2) {
            float freq = 1.0f / powf(theta, (float)i / head_dim);
            float angle = pos * freq;
            float cos_a = cosf(angle), sin_a = sinf(angle);
            float q0 = qh[i], q1 = qh[i + 1];
            qh[i]     = q0 * cos_a - q1 * sin_a;
            qh[i + 1] = q0 * sin_a + q1 * cos_a;
        }
    }
    for (int h = 0; h < n_kv_heads; h++) {
        float *kh = k + h * head_dim;
        for (int i = 0; i < head_dim; i += 2) {
            float freq = 1.0f / powf(theta, (float)i / head_dim);
            float angle = pos * freq;
            float cos_a = cosf(angle), sin_a = sinf(angle);
            float k0 = kh[i], k1 = kh[i + 1];
            kh[i]     = k0 * cos_a - k1 * sin_a;
            kh[i + 1] = k0 * sin_a + k1 * cos_a;
        }
    }
}

/* Quantized matvec dispatch — bounds-safe */
static void qmatvec_safe(model_t *m, const uint8_t *w, int type,
                          const float *x, float *out, int rows, int cols) {
    if (!w) {
        memset(out, 0, (size_t)rows * sizeof(float));
        return;
    }

    /* Clamp rows if tensor data goes past file end */
    int safe_rows = rows;
    if (m->file_data) {
        size_t bpr = bytes_per_row(type, cols);
        size_t w_off = (size_t)(w - m->file_data);
        size_t needed = (size_t)rows * bpr;
        if (w_off + needed > m->file_size) {
            size_t avail = (w_off < m->file_size) ? m->file_size - w_off : 0;
            safe_rows = (int)(avail / bpr);
            if (safe_rows < rows) {
                fprintf(stderr, "WARN: tensor OOB, using %d/%d rows\n", safe_rows, rows);
            }
        }
    }

    if (type == GGML_TYPE_Q4_K) {
        matvec_q4k(w, x, out, safe_rows, cols);
    } else if (type == GGML_TYPE_Q6_K) {
        matvec_q6k(w, x, out, safe_rows, cols);
    } else {
        fprintf(stderr, "Unsupported quant type %d for matvec\n", type);
        safe_rows = 0;
    }
    /* Zero remaining rows */
    if (safe_rows < rows) {
        memset(out + safe_rows, 0, (size_t)(rows - safe_rows) * sizeof(float));
    }
}

float *model_forward(model_t *m, kv_cache_t *cache, int token, int pos) {
    int H = m->hidden_size;
    int HD = m->head_dim;
    int NH = m->num_heads;
    int NKV = m->num_kv_heads;
    int KVD = m->kv_dim;
    int INTER = m->intermediate_size;

    /* Allocate scratch (one-shot) */
    float *x      = malloc(H * sizeof(float));
    float *xnorm  = malloc(H * sizeof(float));
    float *q      = malloc(NH * HD * sizeof(float));
    float *k      = malloc(KVD * sizeof(float));
    float *v      = malloc(KVD * sizeof(float));
    float *attn_out = malloc(H * sizeof(float));
    float *proj   = malloc(H * sizeof(float));
    float *gate   = malloc(INTER * sizeof(float));
    float *up     = malloc(INTER * sizeof(float));
    float *mlp_out = malloc(H * sizeof(float));
    /* Pre-alloc scores for attention (max possible size) */
    float *scores = malloc((size_t)(pos + 1) * sizeof(float));

    if (!x || !xnorm || !q || !k || !v || !attn_out || !proj ||
        !gate || !up || !mlp_out || !scores) {
        fprintf(stderr, "ERROR: malloc failed in forward pass\n");
        fflush(stderr);
        free(x); free(xnorm); free(q); free(k); free(v);
        free(attn_out); free(proj); free(gate); free(up);
        free(mlp_out); free(scores);
        return NULL;
    }

    /* Token embedding — dequant single row on demand */
    embed_lookup(m->embed_data, m->embed_type, token, H, x);

    for (int l = 0; l < m->num_layers; l++) {
        layer_weights *lw = &m->layers[l];

        /* Attention norm */
        rms_norm(xnorm, x, lw->attn_norm, H);

        /* QKV projections */
        qmatvec_safe(m, lw->q_proj, lw->q_type, xnorm, q, NH * HD, H);
        qmatvec_safe(m, lw->k_proj, lw->k_type, xnorm, k, KVD, H);
        qmatvec_safe(m, lw->v_proj, lw->v_type, xnorm, v, KVD, H);

        /* RoPE */
        rope(q, k, pos, HD, NH, NKV, m->rope_theta);

        /* Store KV in cache */
        size_t layer_base = (size_t)l * cache->max_seq * KVD;
        size_t kv_off = layer_base + (size_t)pos * KVD;
        memcpy(cache->key_cache + kv_off, k, KVD * sizeof(float));
        memcpy(cache->value_cache + kv_off, v, KVD * sizeof(float));

        /* Multi-head attention */
        memset(attn_out, 0, H * sizeof(float));
        int kv_group = NH / NKV;
        float scale = 1.0f / sqrtf((float)HD);

        for (int h = 0; h < NH; h++) {
            int kvh = h / kv_group;
            float *qh = q + h * HD;

            float max_score = -1e30f;
            for (int t = 0; t <= pos; t++) {
                size_t k_off = layer_base + (size_t)t * KVD + kvh * HD;
                float s = 0.0f;
                for (int d = 0; d < HD; d++)
                    s += qh[d] * cache->key_cache[k_off + d];
                s *= scale;
                scores[t] = s;
                if (s > max_score) max_score = s;
            }

            float sum = 0.0f;
            for (int t = 0; t <= pos; t++) {
                scores[t] = expf(scores[t] - max_score);
                sum += scores[t];
            }
            float inv_sum = 1.0f / (sum + 1e-10f);
            for (int t = 0; t <= pos; t++) scores[t] *= inv_sum;

            float *oh = attn_out + h * HD;
            for (int t = 0; t <= pos; t++) {
                size_t v_off = layer_base + (size_t)t * KVD + kvh * HD;
                float s = scores[t];
                for (int d = 0; d < HD; d++)
                    oh[d] += s * cache->value_cache[v_off + d];
            }
        }

        /* Output projection */
        qmatvec_safe(m, lw->o_proj, lw->o_type, attn_out, proj, H, NH * HD);

        /* SwiGLU MLP (parallel: uses same xnorm as attention) */
        qmatvec_safe(m, lw->gate_proj, lw->gate_type, xnorm, gate, INTER, H);
        qmatvec_safe(m, lw->up_proj,   lw->up_type,   xnorm, up,   INTER, H);
        silu_inplace(gate, INTER);
        for (int i = 0; i < INTER; i++) gate[i] *= up[i];
        qmatvec_safe(m, lw->down_proj, lw->down_type, gate, mlp_out, H, INTER);

        /* Residual: x += attn_proj + mlp_out (parallel add) */
        for (int i = 0; i < H; i++) x[i] += proj[i] + mlp_out[i];
    }

    /* Final norm */
    rms_norm(x, x, m->output_norm, H);

    /* Logits — output matvec (quantized) */
    float *logits = malloc((size_t)m->vocab_size * sizeof(float));
    qmatvec_safe(m, m->output_data, m->output_type, x, logits, m->vocab_size, H);

    /* Apply logit scale (Cohere2) */
    if (m->logit_scale != 1.0f) {
        for (int i = 0; i < m->vocab_size; i++) logits[i] *= m->logit_scale;
    }

    free(xnorm); free(q); free(k); free(v);
    free(attn_out); free(proj); free(gate); free(up); free(mlp_out);
    free(scores); free(x);

    return logits;
}
