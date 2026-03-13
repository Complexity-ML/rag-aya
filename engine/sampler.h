/**
 * Token sampling — argmax, top-k, advanced (top-k + min-p + top-p).
 * INL 2025
 */
#ifndef SAMPLER_H
#define SAMPLER_H

/* Sampling parameters */
typedef struct {
    int   top_k;
    float top_p;
    float min_p;
    float temperature;
} sample_params_t;

/* Greedy argmax */
int argmax_fn(const float *logits, int n);

/* Top-k sampling with temperature */
int sample_topk(const float *logits, int vocab_size,
                int top_k, float temperature);

/* Advanced sampler: top-k -> softmax -> min-p -> top-p -> sample */
int sample_advanced(const float *logits, int vocab_size,
                    sample_params_t sp);

#endif /* SAMPLER_H */
