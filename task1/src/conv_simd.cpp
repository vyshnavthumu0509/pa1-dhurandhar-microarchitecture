// conv_simd.cpp  STAGE 4: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "convolution.h"

void conv_simd(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int i = 0; i < H * W; ++i) out[i] = 0.0f;

    for (int ky = 0; ky < K; ++ky) {
        for (int kx = 0; kx < K; ++kx) {
            const __m256 kv = _mm256_set1_ps(ker[ky * K + kx]);
            for (int oy = 0; oy < H; ++oy) {
                const float* in_row  = &in[(oy + ky) * in_stride + kx];
                float*       out_row = &out[oy * W];

                int ox = 0;
                for (; ox + 8 <= W; ox += 8) {
                    __m256 inv  = _mm256_loadu_ps(in_row + ox);
                    __m256 outv = _mm256_loadu_ps(out_row + ox);
                    outv = _mm256_fmadd_ps(inv, kv, outv);
                    _mm256_storeu_ps(out_row + ox, outv);
                }
                for (; ox < W; ++ox) {               
                    out_row[ox] += in_row[ox] * kv[0];
                }
            }
        }
    }
}