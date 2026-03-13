/**
 * Token sampling implementation.
 * INL 2025
 */
#include "sampler.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

/* ---- Token quality vector ---- */

/* Check if a string contains only printable ASCII/UTF-8 */
static int is_printable(const char *s) {
    while (*s) {
        unsigned char c = (unsigned char)*s;
        if (c < 0x20 && c != '\n' && c != '\t' && c != '\r') return 0;
        if (c == 0x7F) return 0;
        s++;
    }
    return 1;
}

float *build_token_quality(tokenizer_t *tk) {
    float *q = calloc(tk->vocab_size, sizeof(float));
    int n_penalized = 0, n_boosted = 0;

    for (int i = 0; i < tk->vocab_size; i++) {
        const char *tok = tk->tokens[i];
        if (!tok) { q[i] = -20.0f; n_penalized++; continue; }

        int len = (int)strlen(tok);
        float score = 0.0f;

        /* Control / special tokens */
        if (i < 10) {
            score = -20.0f;
            goto done;
        }
        if (tok[0] == '<' && tok[1] == '|') {
            score = -20.0f;
            goto done;
        }

        /* Byte fallback <0xNN> */
        if (len == 6 && tok[0] == '<' && tok[1] == '0' && tok[2] == 'x') {
            score = -2.0f;
            goto done;
        }

        /* Known artifacts */
        if (strstr(tok, "[[") || strstr(tok, "]]") ||
            strstr(tok, "<<<") || strstr(tok, ">>>") ||
            strstr(tok, "{{") || strstr(tok, "}}")) {
            score = -5.0f;
            goto done;
        }

        /* Repetitive single-char tokens like "aaaa", "nnnn" */
        if (len >= 4) {
            int all_same = 1;
            for (int j = 1; j < len; j++) {
                if (tok[j] != tok[0]) { all_same = 0; break; }
            }
            if (all_same) { score = -3.0f; goto done; }
        }

        /* Non-printable content */
        if (!is_printable(tok)) {
            score = -1.0f;
            goto done;
        }

        /* Boost: complete word (starts with sentencepiece ▁ = 0xE2 0x96 0x81) */
        if (len >= 3 && (unsigned char)tok[0] == 0xE2 &&
            (unsigned char)tok[1] == 0x96 && (unsigned char)tok[2] == 0x81) {
            score += 0.3f;
        }
        /* Or starts with space (GPT-2 style Ġ = 0xC4 0xA0) */
        if (len >= 2 && (unsigned char)tok[0] == 0xC4 &&
            (unsigned char)tok[1] == 0xA0) {
            score += 0.3f;
        }

        /* Meaningful length bonus */
        if (len >= 3) score += 0.1f;

    done:
        q[i] = score;
        if (score < -0.5f) n_penalized++;
        if (score > 0.05f) n_boosted++;
    }

    printf("  Token quality: %d penalized, %d boosted / %d total\n",
           n_penalized, n_boosted, tk->vocab_size);
    return q;
}

void apply_token_quality(float *logits, const float *quality,
                         int vocab_size, float alpha) {
    if (!quality || alpha == 0.0f) return;
    for (int i = 0; i < vocab_size; i++) {
        logits[i] += alpha * quality[i];
    }
}
