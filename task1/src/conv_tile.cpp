// conv_tile.cpp  STAGE 3: CACHE TILING
#include "convolution.h"
#include <algorithm>
#include <cstring>

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    // zero-init output
    std::memset(out, 0, sizeof(float) * H * W);

    // Tile sizes: tune these for your L1/L2. Start here and measure.
    constexpr int TILE_H = 64;
    constexpr int TILE_W = 64;

    for (int oy0 = 0; oy0 < H; oy0 += TILE_H) {
        const int oy_end = std::min(oy0 + TILE_H, H);

        for (int ox0 = 0; ox0 < W; ox0 += TILE_W) {
            const int ox_end = std::min(ox0 + TILE_W, W);

            // For this (oy0,ox0) tile, sweep all kernel taps.
            // The 'in' region touched is [oy0, oy_end+K) x [ox0, ox_end+K),
            // and the 'out' region is [oy0,oy_end) x [ox0,ox_end) -- both
            // small enough to stay in L1/L2 across the K*K taps.
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    const float kv = ker[ky * K + kx];

                    for (int oy = oy0; oy < oy_end; ++oy) {
                        const float* in_row  = &in[(oy + ky) * in_stride + kx];
                        float*       out_row = &out[oy * W];

                        for (int ox = ox0; ox < ox_end; ++ox) {
                            out_row[ox] += in_row[ox] * kv;
                        }
                    }
                }
            }
        }
    }
}