/**
 * Token sampling implementation.
 * INL 2025
 */
#include "sampler.h"
#include <stdlib.h>
#include <math.h>

int argmax_fn(const float *logits, int n) {
    int best = 0;
    float best_val = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > best_val) { best_val = logits[i]; best = i; }
    }
    return best;
}

int sample_topk(const float *logits, int vocab_size,
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

int sample_advanced(const float *logits, int vocab_size,
                    sample_params_t sp) {
    /* Step 1: find top-K candidates */
    int cap = sp.top_k > 0 ? sp.top_k : 256;
    if (cap > vocab_size) cap = vocab_size;
    if (sp.top_p > 0.0f && sp.top_p < 1.0f && cap < 1024)
        cap = vocab_size < 1024 ? vocab_size : 1024;

    int *indices = malloc(cap * sizeof(int));
    float *vals  = malloc(cap * sizeof(float));
    for (int i = 0; i < cap; i++) { indices[i] = -1; vals[i] = -1e30f; }

    for (int v = 0; v < vocab_size; v++) {
        float val = logits[v];
        if (val > vals[cap - 1]) {
            vals[cap - 1] = val;
            indices[cap - 1] = v;
            for (int j = cap - 1; j > 0 && vals[j] > vals[j - 1]; j--) {
                float tv = vals[j]; vals[j] = vals[j - 1]; vals[j - 1] = tv;
                int ti = indices[j]; indices[j] = indices[j - 1]; indices[j - 1] = ti;
            }
        }
    }

    /* Step 2: softmax with temperature */
    float max_val = vals[0];
    float sum = 0.0f;
    int count = 0;
    for (int i = 0; i < cap && indices[i] >= 0; i++) {
        vals[i] = expf((vals[i] - max_val) / sp.temperature);
        sum += vals[i];
        count++;
    }
    for (int i = 0; i < count; i++) vals[i] /= sum;

    /* Step 3: min-p filter */
    int n_keep = count;
    if (sp.min_p > 0.0f) {
        float threshold = sp.min_p * vals[0];
        n_keep = 0;
        for (int i = 0; i < count; i++) {
            if (vals[i] >= threshold) n_keep++;
            else break;
        }
        if (n_keep == 0) n_keep = 1;
    }

    /* Step 4: top-p (nucleus) filter */
    if (sp.top_p > 0.0f && sp.top_p < 1.0f) {
        float cum = 0.0f;
        int p_keep = 0;
        for (int i = 0; i < n_keep; i++) {
            cum += vals[i];
            p_keep++;
            if (cum >= sp.top_p) break;
        }
        n_keep = p_keep;
    }

    /* Step 5: re-normalize */
    float kept_sum = 0.0f;
    for (int i = 0; i < n_keep; i++) kept_sum += vals[i];
    for (int i = 0; i < n_keep; i++) vals[i] /= kept_sum;

    /* Step 6: sample */
    float r = (float)rand() / (float)RAND_MAX;
    float cum = 0.0f;
    int result = indices[0];
    for (int i = 0; i < n_keep; i++) {
        cum += vals[i];
        if (r <= cum) { result = indices[i]; break; }
    }

    free(indices);
    free(vals);
    return result;
}
