// conv_simd.cpp  STAGE 4: SIMD with AVX2 intrinsics (Corrected & Optimized)
#include <immintrin.h>
#include "convolution.h"

void conv_simd(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for(int oy = 0; oy < H; ++oy) {
        int ox = 0;
        
        // SIMD loop: process 8 pixels at a time
        // FIX: Changed from `ox + 8 < W` to `ox + 8 <= W`
        for(; ox + 8 <= W; ox += 8) {
            
            // Initialize the accumulator for 8 output pixels
            __m256 outv = _mm256_setzero_ps();

            // Compute the convolution
            for(int ky = 0; ky < K; ++ky) {
               for(int kx = 0; kx < K; kx++) {
                    __m256 inv = _mm256_loadu_ps(&in[(oy + ky) * in_stride + (ox + kx)]);
                    __m256 kv  = _mm256_set1_ps(ker[ky * K + kx]);
                    outv = _mm256_fmadd_ps(inv, kv, outv);
                }
            }
            
            // FIX: Write to memory exactly ONCE per 8-pixel block
            _mm256_storeu_ps(&out[oy * W + ox], outv);
        }
        
        // Tail loop: process the remaining pixels (if W is not a multiple of 8)
        for(; ox < W; ++ox) {
            float acc = 0.0f;
            
            for(int ky = 0; ky < K; ++ky) {
                for(int kx = 0; kx < K; ++kx) {
                    acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                }
            }
            
            // FIX: Write to memory exactly ONCE per pixel
            out[oy * W + ox] = acc;
        }
    }
}