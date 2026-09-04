#include "convolution.h"
#include <algorithm>
#include <cstring>

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    std::memset(out, 0, sizeof(float) * H * W);

    const int TILE_H = 4;
    
    // Core fix: Only tile the width if the working set exceeds the 32KB L1 cache.
    // 3 rows usually fit in L1, so we process the full width to minimize loop overhead.
    // 5+ rows exceed L1, so we clamp the width to 512 to prevent cache thrashing.
    const int TILE_W = (K <= 3) ? W : 512;

    for(int oy0 = 0; oy0 < H; oy0 += TILE_H){
        const int oy_end = std::min(oy0 + TILE_H, H);

        for(int ox0 = 0; ox0 < W; ox0 += TILE_W){
            const int ox_end = std::min(ox0 + TILE_W, W);
            
            // Standard nested loops preserved exactly as requested
            for(int oy = oy0; oy < oy_end; ++oy){
                for(int ox = ox0; ox < ox_end; ++ox){
                    float acc = 0.0f;
                    for(int ky = 0; ky < K; ++ky){
                        for(int kx = 0; kx < K; ++kx){
                            acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                        }
                    }
                    out[oy * W + ox] = acc;
                }
            }
        }
    }
}