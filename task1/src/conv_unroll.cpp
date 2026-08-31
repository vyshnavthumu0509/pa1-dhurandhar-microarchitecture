// conv_unroll.cpp  STAGE 2: LOOP UNROLLING
#include "convolution.h"

void conv_unroll(const float* in, float* out, const float* ker,
                 int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int i = 0; i < H * W; ++i) out[i] = 0.0f;

    for (int ky = 0; ky < K; ++ky) {
        for (int kx = 0; kx < K; ++kx) {
            const float kv = ker[ky * K + kx];
            for (int oy = 0; oy < H; ++oy) {
                const float* in_row  = &in[(oy + ky) * in_stride + kx];
                float*       out_row = &out[oy * W];

                int ox = 0;
                const int W4 = W & ~3;  
                for (; ox < W4; ox += 4) {
                    out_row[ox+0] += in_row[ox+0] * kv;
                    out_row[ox+1] += in_row[ox+1] * kv;
                    out_row[ox+2] += in_row[ox+2] * kv;
                    out_row[ox+3] += in_row[ox+3] * kv;
                }
                for (; ox < W; ++ox) {         
                    out_row[ox] += in_row[ox] * kv;
                }
            }
        }
    }
}
