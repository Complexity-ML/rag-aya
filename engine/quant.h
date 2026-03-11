/**
 * Integer quantization — Q4_K & Q6_K with SIMD.
 * Supports: AVX2+FMA (x86), NEON (ARM), scalar fallback.
 * INL 2025
 */
#ifndef QUANT_H
#define QUANT_H

#include <stdint.h>
#include <stddef.h>
#include <math.h>

/* SIMD detection */
#ifdef __AVX2__
#include <immintrin.h>
#define HAS_AVX2 1
#else
#define HAS_AVX2 0
#endif

#if defined(__aarch64__)
#include <arm_neon.h>
#define HAS_NEON 1
#else
#define HAS_NEON 0
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

/* F16 to F32 conversion */
static inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t frac = h & 0x3FF;

    if (exp == 0) {
        if (frac == 0) return sign ? -0.0f : 0.0f;
        float f = (float)frac / 1024.0f * (1.0f / 16384.0f);
        return sign ? -f : f;
    }
    if (exp == 31) return frac == 0 ? (sign ? -INFINITY : INFINITY) : NAN;

    float f = (1.0f + (float)frac / 1024.0f) * ldexpf(1.0f, (int)exp - 15);
    return sign ? -f : f;
}

/* ================================================================
 * Q4_K dot product
 * Block: 256 elements, 144 bytes
 *   2B f16 d, 2B f16 dmin, 12B packed scales/mins, 128B quants
 * ================================================================ */

/* ggml Q4_K nibble layout: 4 groups of 64 elements, each group uses 32 bytes.
 * Low nibbles (byte & 0xF) = first 32 elements, high nibbles (byte >> 4) = next 32.
 * scales[j*2] applies to low nibbles, scales[j*2+1] to high nibbles. */

static inline float dot_q4k_f32(const uint8_t *qdata, const float *x, int n) {
    int blocks = n / 256;
    float total = 0.0f;

    for (int b = 0; b < blocks; b++) {
        const uint8_t *block = qdata + b * 144;
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
        const float *xb = x + b * 256;

        /* 4 groups of 64 elements, each with 32 bytes of quants */
        for (int j = 0; j < 4; j++) {
            float sc0 = (float)scales[j*2];
            float m0  = (float)mins[j*2];
            float sc1 = (float)scales[j*2+1];
            float m1  = (float)mins[j*2+1];
            const uint8_t *qj = quants + j * 32;
            const float *xj = xb + j * 64;

#if HAS_NEON
            float32x4_t vqsum0 = vdupq_n_f32(0.0f);
            float32x4_t vqsum1 = vdupq_n_f32(0.0f);
            float32x4_t vxsum0 = vdupq_n_f32(0.0f);
            float32x4_t vxsum1 = vdupq_n_f32(0.0f);
            for (int l = 0; l < 32; l += 4) {
                /* Load 4 quant bytes and extract low/high nibbles */
                uint8x8_t qbytes = vld1_u8(qj + l);  /* load 4 (only use low 4) */
                uint16x4_t qw = vget_low_u16(vmovl_u8(qbytes));
                /* low nibbles */
                float32x4_t q_lo = vcvtq_f32_u32(vmovl_u16(vand_u16(qw, vdup_n_u16(0xF))));
                /* high nibbles */
                float32x4_t q_hi = vcvtq_f32_u32(vmovl_u16(vshr_n_u16(qw, 4)));
                /* Load x vectors */
                float32x4_t vx0 = vld1q_f32(xj + l);
                float32x4_t vx1 = vld1q_f32(xj + l + 32);
                vqsum0 = vfmaq_f32(vqsum0, q_lo, vx0);
                vqsum1 = vfmaq_f32(vqsum1, q_hi, vx1);
                vxsum0 = vaddq_f32(vxsum0, vx0);
                vxsum1 = vaddq_f32(vxsum1, vx1);
            }
            float qsum0 = vaddvq_f32(vqsum0);
            float qsum1 = vaddvq_f32(vqsum1);
            float xsum0 = vaddvq_f32(vxsum0);
            float xsum1 = vaddvq_f32(vxsum1);
#else
            float qsum0 = 0.0f, qsum1 = 0.0f;
            float xsum0 = 0.0f, xsum1 = 0.0f;
            for (int l = 0; l < 32; l++) {
                qsum0 += (float)(qj[l] & 0xF) * xj[l];
                qsum1 += (float)(qj[l] >>  4) * xj[l + 32];
                xsum0 += xj[l];
                xsum1 += xj[l + 32];
            }
#endif
            total += d * sc0 * qsum0 - dmin * m0 * xsum0;
            total += d * sc1 * qsum1 - dmin * m1 * xsum1;
        }
    }
    return total;
}

static inline void matvec_q4k(const uint8_t *mat, const float *x,
                                float *out, int rows, int cols) {
    size_t bpr = (size_t)(cols / 256) * 144;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < rows; i++) {
        out[i] = dot_q4k_f32(mat + (size_t)i * bpr, x, cols);
    }
}

/* ================================================================
 * Q6_K dequantize row — ggml interleaved layout
 * Block: 256 elements, 210 bytes
 *   128B ql, 64B qh, 16B scales, 2B f16 d
 *
 * ggml layout: 2 halves of 128 elements each.
 * Per half (32 iterations of l=0..31):
 *   elem[l+0]  = (ql[l]    & 0xF) | ((qh[l]>>0 & 3) << 4)
 *   elem[l+32] = (ql[l+32] & 0xF) | ((qh[l]>>2 & 3) << 4)
 *   elem[l+64] = (ql[l]     >> 4) | ((qh[l]>>4 & 3) << 4)
 *   elem[l+96] = (ql[l+32]  >> 4) | ((qh[l]>>6 & 3) << 4)
 * Then ql += 64, qh += 32, sc += 8 for second half.
 * ================================================================ */

static inline void dequant_q6k_row(const uint8_t *block_data, float *out,
                                     int n_elements) {
    int n_blocks = n_elements / 256;
    for (int b = 0; b < n_blocks; b++) {
        const uint8_t *block = block_data + b * 210;
        const uint8_t *ql = block;
        const uint8_t *qh = block + 128;
        const int8_t  *sc = (const int8_t *)(block + 192);
        float d = f16_to_f32(*(const uint16_t *)(block + 208));

        float *y = out + b * 256;
        for (int half = 0; half < 2; half++) {
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int q1 = (int)(( ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)(( ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)(( ql[l]       >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)(( ql[l + 32]  >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            ql += 64;
            qh += 32;
            sc += 8;
            y  += 128;
        }
    }
}

/* ================================================================
 * Q6_K dot product — ggml interleaved layout
 * ================================================================ */

static inline float dot_q6k_f32(const uint8_t *qdata, const float *x, int n) {
    int n_blocks = n / 256;
    float total = 0.0f;

    for (int b = 0; b < n_blocks; b++) {
        const uint8_t *block = qdata + b * 210;
        const uint8_t *ql = block;
        const uint8_t *qh = block + 128;
        const int8_t  *sc = (const int8_t *)(block + 192);
        float d = f16_to_f32(*(const uint16_t *)(block + 208));

        const float *yb = x + b * 256;

        for (int half = 0; half < 2; half++) {
#if HAS_NEON
            /* Process 2 sub-groups of 16 elements each */
            for (int ig = 0; ig < 2; ig++) {
                float dsc0 = d * sc[ig + 0];
                float dsc2 = d * sc[ig + 2];
                float dsc4 = d * sc[ig + 4];
                float dsc6 = d * sc[ig + 6];
                float32x4_t vdsc0 = vdupq_n_f32(dsc0);
                float32x4_t vdsc2 = vdupq_n_f32(dsc2);
                float32x4_t vdsc4 = vdupq_n_f32(dsc4);
                float32x4_t vdsc6 = vdupq_n_f32(dsc6);
                float32x4_t acc0 = vdupq_n_f32(0.0f);
                float32x4_t acc1 = vdupq_n_f32(0.0f);
                float32x4_t acc2 = vdupq_n_f32(0.0f);
                float32x4_t acc3 = vdupq_n_f32(0.0f);
                int base = ig * 16;
                for (int l = base; l < base + 16; l += 4) {
                    /* Load 4 ql bytes from each half */
                    uint8x8_t ql_lo_raw = vld1_u8(ql + l);
                    uint8x8_t ql_hi_raw = vld1_u8(ql + l + 32);
                    uint8x8_t qh_raw    = vld1_u8(qh + l);
                    /* q1 = (ql[l] & 0xF) | ((qh[l]>>0 & 3) << 4) - 32 */
                    uint8x8_t lo_nib_a = vand_u8(ql_lo_raw, vdup_n_u8(0xF));
                    uint8x8_t qh_bits0 = vshl_n_u8(vand_u8(qh_raw, vdup_n_u8(3)), 4);
                    uint8x8_t raw1 = vorr_u8(lo_nib_a, qh_bits0);
                    int16x8_t q1_16 = vreinterpretq_s16_u16(vmovl_u8(raw1));
                    int16x4_t q1_lo = vget_low_s16(vsubq_s16(q1_16, vdupq_n_s16(32)));
                    float32x4_t fq1 = vcvtq_f32_s32(vmovl_s16(q1_lo));
                    /* q2 = (ql[l+32] & 0xF) | ((qh[l]>>2 & 3) << 4) - 32 */
                    uint8x8_t lo_nib_b = vand_u8(ql_hi_raw, vdup_n_u8(0xF));
                    uint8x8_t qh_bits2 = vshl_n_u8(vand_u8(vshr_n_u8(qh_raw, 2), vdup_n_u8(3)), 4);
                    uint8x8_t raw2 = vorr_u8(lo_nib_b, qh_bits2);
                    int16x8_t q2_16 = vreinterpretq_s16_u16(vmovl_u8(raw2));
                    int16x4_t q2_lo = vget_low_s16(vsubq_s16(q2_16, vdupq_n_s16(32)));
                    float32x4_t fq2 = vcvtq_f32_s32(vmovl_s16(q2_lo));
                    /* q3 = (ql[l] >> 4) | ((qh[l]>>4 & 3) << 4) - 32 */
                    uint8x8_t hi_nib_a = vshr_n_u8(ql_lo_raw, 4);
                    uint8x8_t qh_bits4 = vshl_n_u8(vand_u8(vshr_n_u8(qh_raw, 4), vdup_n_u8(3)), 4);
                    uint8x8_t raw3 = vorr_u8(hi_nib_a, qh_bits4);
                    int16x8_t q3_16 = vreinterpretq_s16_u16(vmovl_u8(raw3));
                    int16x4_t q3_lo = vget_low_s16(vsubq_s16(q3_16, vdupq_n_s16(32)));
                    float32x4_t fq3 = vcvtq_f32_s32(vmovl_s16(q3_lo));
                    /* q4 = (ql[l+32] >> 4) | ((qh[l]>>6 & 3) << 4) - 32 */
                    uint8x8_t hi_nib_b = vshr_n_u8(ql_hi_raw, 4);
                    uint8x8_t qh_bits6 = vshl_n_u8(vshr_n_u8(qh_raw, 6), 4);
                    uint8x8_t raw4 = vorr_u8(hi_nib_b, qh_bits6);
                    int16x8_t q4_16 = vreinterpretq_s16_u16(vmovl_u8(raw4));
                    int16x4_t q4_lo = vget_low_s16(vsubq_s16(q4_16, vdupq_n_s16(32)));
                    float32x4_t fq4 = vcvtq_f32_s32(vmovl_s16(q4_lo));
                    /* Load x vectors */
                    float32x4_t vx0 = vld1q_f32(yb + l);
                    float32x4_t vx1 = vld1q_f32(yb + l + 32);
                    float32x4_t vx2 = vld1q_f32(yb + l + 64);
                    float32x4_t vx3 = vld1q_f32(yb + l + 96);
                    acc0 = vfmaq_f32(acc0, fq1, vx0);
                    acc1 = vfmaq_f32(acc1, fq2, vx1);
                    acc2 = vfmaq_f32(acc2, fq3, vx2);
                    acc3 = vfmaq_f32(acc3, fq4, vx3);
                }
                total += dsc0 * vaddvq_f32(acc0);
                total += dsc2 * vaddvq_f32(acc1);
                total += dsc4 * vaddvq_f32(acc2);
                total += dsc6 * vaddvq_f32(acc3);
            }
#else
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int q1 = (int)(( ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)(( ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)(( ql[l]       >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)(( ql[l + 32]  >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
                total += d * sc[is + 0] * q1 * yb[l +  0];
                total += d * sc[is + 2] * q2 * yb[l + 32];
                total += d * sc[is + 4] * q3 * yb[l + 64];
                total += d * sc[is + 6] * q4 * yb[l + 96];
            }
#endif
            ql += 64;
            qh += 32;
            sc += 8;
            yb += 128;
        }
    }
    return total;
}

static inline void matvec_q6k(const uint8_t *mat, const float *x,
                                float *out, int rows, int cols) {
    size_t bpr = (size_t)(cols / 256) * 210;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < rows; i++) {
        out[i] = dot_q6k_f32(mat + (size_t)i * bpr, x, cols);
    }
}

/* ================================================================
 * F32 matvec
 * ================================================================ */

#if HAS_AVX2
static inline void matvec_f32(const float *mat, const float *x,
                               float *out, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        const float *row = mat + (size_t)i * cols;
        __m256 acc = _mm256_setzero_ps();
        int j;
        for (j = 0; j + 8 <= cols; j += 8) {
            __m256 r = _mm256_loadu_ps(row + j);
            __m256 v = _mm256_loadu_ps(x + j);
            acc = _mm256_fmadd_ps(r, v, acc);
        }
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 s = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        float sum = _mm_cvtss_f32(s);
        for (; j < cols; j++) sum += row[j] * x[j];
        out[i] = sum;
    }
}
#elif HAS_NEON
static inline void matvec_f32(const float *mat, const float *x,
                               float *out, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        const float *row = mat + (size_t)i * cols;
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        float32x4_t acc2 = vdupq_n_f32(0.0f);
        float32x4_t acc3 = vdupq_n_f32(0.0f);
        int j;
        for (j = 0; j + 16 <= cols; j += 16) {
            acc0 = vfmaq_f32(acc0, vld1q_f32(row + j),      vld1q_f32(x + j));
            acc1 = vfmaq_f32(acc1, vld1q_f32(row + j + 4),  vld1q_f32(x + j + 4));
            acc2 = vfmaq_f32(acc2, vld1q_f32(row + j + 8),  vld1q_f32(x + j + 8));
            acc3 = vfmaq_f32(acc3, vld1q_f32(row + j + 12), vld1q_f32(x + j + 12));
        }
        float sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
        for (; j < cols; j++) sum += row[j] * x[j];
        out[i] = sum;
    }
}
#else
static inline void matvec_f32(const float *mat, const float *x,
                               float *out, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float s = 0.0f;
        const float *row = mat + (size_t)i * cols;
        for (int j = 0; j < cols; j++) s += row[j] * x[j];
        out[i] = s;
    }
}
#endif

#endif /* QUANT_H */
