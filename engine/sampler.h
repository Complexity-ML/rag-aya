/**
 * Token sampling — argmax, top-k, advanced (top-k + min-p + top-p).
 * Token quality vector for logit biasing.
 * INL 2025
 */
#ifndef SAMPLER_H
#define SAMPLER_H

#include "tokenizer.h"

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

/* ---- Token quality vector ----
 *
 * Pre-computed bias for each token in the vocabulary.
 * Applied as: logits[i] += alpha * quality[i]  before sampling.
 *
 * Scoring heuristics:
 *   +0.3  complete word (starts with space/▁)
 *   +0.1  token length >= 3 chars (meaningful content)
 *   -2.0  byte fallback token <0xNN>
 *   -5.0  known artifact ([[ ]] <<< >>> etc.)
 *  -20.0  control / special token (id < 10 or <|...|>)
 *   -1.0  non-printable content
 *
 * Returns malloc'd float[vocab_size]. Caller must free. */
float *build_token_quality(tokenizer_t *tk);

/* Apply quality bias to logits in-place: logits[i] += alpha * quality[i] */
void apply_token_quality(float *logits, const float *quality,
                         int vocab_size, float alpha);

#endif /* SAMPLER_H */
