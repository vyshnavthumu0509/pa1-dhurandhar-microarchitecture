// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
// Combines: cache tiling (Stage 3) + loop unrolling (Stage 2) + AVX2 SIMD (Stage 4).
//
// Why tiling still matters even though the K=3/K=5 kernels are tiny:
// each output row reads K input rows, and consecutive output rows share K-1 of
// those rows. If we sweep a full 2048-wide row before moving to the next row
// (as the flat K==3/K==5 loops below did), those shared input rows have long
// since been evicted from L1 (a 2048-float row is 8KB - fine alone, but K rows
// plus the streaming output blow well past a 32-48KB L1). Tiling narrows the
// column range (TILE_W) we work on before advancing oy, so the K input rows
// needed for a whole TILE_H-tall, TILE_W-wide block stay resident in cache
// while we reuse them.
//
// Tune TILE_H / TILE_W for your machine: measure with `./bin/conv optimized`
// and with `perf stat -e L1-dcache-load-misses` as the assignment asks.

#include <algorithm>
#include <immintrin.h>
#include "convolution.h"

namespace {
// Starting point: a TILE_H x (TILE_W + K-1 halo) strip of input rows should
// comfortably fit in a 32KB L1: TILE_H=8 output rows x ~(TILE_W+4) floats x 4B
// is a few KB. Tune these two and re-measure.
constexpr int TILE_H = 4;
constexpr int TILE_W = 32;  // must stay a multiple of 8 (AVX2 lane width)
}  // namespace

void conv_optimized(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    // -------------------------------------------------------------------------
    // OPTIMIZED PATH: Fully unrolled 3x3 kernel, tiled + AVX2
    // -------------------------------------------------------------------------
    if (K == 3) {
        __m256 k0 = _mm256_set1_ps(ker[0]);
        __m256 k1 = _mm256_set1_ps(ker[1]);
        __m256 k2 = _mm256_set1_ps(ker[2]);
        __m256 k3 = _mm256_set1_ps(ker[3]);
        __m256 k4 = _mm256_set1_ps(ker[4]);
        __m256 k5 = _mm256_set1_ps(ker[5]);
        __m256 k6 = _mm256_set1_ps(ker[6]);
        __m256 k7 = _mm256_set1_ps(ker[7]);
        __m256 k8 = _mm256_set1_ps(ker[8]);

        for (int oy0 = 0; oy0 < H; oy0 += TILE_H) {
            const int oy_end = std::min(oy0 + TILE_H, H);

            for (int ox0 = 0; ox0 < W; ox0 += TILE_W) {
                const int ox_end = std::min(ox0 + TILE_W, W);

                for (int oy = oy0; oy < oy_end; ++oy) {
                    int ox = ox0;

                    // SIMD main loop within the tile
                    for (; ox + 8 <= ox_end; ox += 8) {
                        __m256 outv = _mm256_setzero_ps();

                        // Row 0
                        outv = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + 0) * in_stride + (ox + 0)]), k0, outv);
                        outv = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + 0) * in_stride + (ox + 1)]), k1, outv);
                        outv = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + 0) * in_stride + (ox + 2)]), k2, outv);

                        // Row 1
                        outv = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + 1) * in_stride + (ox + 0)]), k3, outv);
                        outv = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + 1) * in_stride + (ox + 1)]), k4, outv);
                        outv = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + 1) * in_stride + (ox + 2)]), k5, outv);

                        // Row 2
                        outv = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + 2) * in_stride + (ox + 0)]), k6, outv);
                        outv = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + 2) * in_stride + (ox + 1)]), k7, outv);
                        outv = _mm256_fmadd_ps(_mm256_loadu_ps(&in[(oy + 2) * in_stride + (ox + 2)]), k8, outv);

                        _mm256_storeu_ps(&out[oy * W + ox], outv);
                    }

                    // Tail within the tile (only fires if TILE_W doesn't divide
                    // evenly into 8, or on the last, possibly-short tile column)
                    for (; ox < ox_end; ++ox) {
                        float acc = 0.0f;
                        acc += in[(oy + 0) * in_stride + (ox + 0)] * ker[0];
                        acc += in[(oy + 0) * in_stride + (ox + 1)] * ker[1];
                        acc += in[(oy + 0) * in_stride + (ox + 2)] * ker[2];
                        acc += in[(oy + 1) * in_stride + (ox + 0)] * ker[3];
                        acc += in[(oy + 1) * in_stride + (ox + 1)] * ker[4];
                        acc += in[(oy + 1) * in_stride + (ox + 2)] * ker[5];
                        acc += in[(oy + 2) * in_stride + (ox + 0)] * ker[6];
                        acc += in[(oy + 2) * in_stride + (ox + 1)] * ker[7];
                        acc += in[(oy + 2) * in_stride + (ox + 2)] * ker[8];
                        out[oy * W + ox] = acc;
                    }
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // OPTIMIZED PATH: Fully unrolled 5x5 kernel, tiled + AVX2
    // -------------------------------------------------------------------------
    else if (K == 5) {
        __m256 k[25];
        for (int i = 0; i < 25; ++i) k[i] = _mm256_set1_ps(ker[i]);

        for (int oy0 = 0; oy0 < H; oy0 += TILE_H) {
            const int oy_end = std::min(oy0 + TILE_H, H);

            for (int ox0 = 0; ox0 < W; ox0 += TILE_W) {
                const int ox_end = std::min(ox0 + TILE_W, W);

                for (int oy = oy0; oy < oy_end; ++oy) {
                    int ox = ox0;

                    for (; ox + 8 <= ox_end; ox += 8) {
                        __m256 outv = _mm256_setzero_ps();
                        // NOTE: the column offset within a kernel row is always
                        // kx (0..4) -- it must NOT continue counting up with the
                        // flat ky*K+kx kernel index. (That mismatch was the bug
                        // in the previous version of this file.)
                        for (int ky = 0; ky < 5; ++ky) {
                            const float* row = &in[(oy + ky) * in_stride + ox];
                            outv = _mm256_fmadd_ps(_mm256_loadu_ps(row + 0), k[ky * 5 + 0], outv);
                            outv = _mm256_fmadd_ps(_mm256_loadu_ps(row + 1), k[ky * 5 + 1], outv);
                            outv = _mm256_fmadd_ps(_mm256_loadu_ps(row + 2), k[ky * 5 + 2], outv);
                            outv = _mm256_fmadd_ps(_mm256_loadu_ps(row + 3), k[ky * 5 + 3], outv);
                            outv = _mm256_fmadd_ps(_mm256_loadu_ps(row + 4), k[ky * 5 + 4], outv);
                        }
                        _mm256_storeu_ps(&out[oy * W + ox], outv);
                    }

                    for (; ox < ox_end; ++ox) {
                        float acc = 0.0f;
                        for (int ky = 0; ky < 5; ++ky) {
                            for (int kx = 0; kx < 5; ++kx) {
                                acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * 5 + kx];
                            }
                        }
                        out[oy * W + ox] = acc;
                    }
                }
            }
        }
    }
    // -------------------------------------------------------------------------
    // DYNAMIC FALLBACK: For other odd K, tiled + AVX2
    // -------------------------------------------------------------------------
    else {
        for (int oy0 = 0; oy0 < H; oy0 += TILE_H) {
            const int oy_end = std::min(oy0 + TILE_H, H);

            for (int ox0 = 0; ox0 < W; ox0 += TILE_W) {
                const int ox_end = std::min(ox0 + TILE_W, W);

                for (int oy = oy0; oy < oy_end; ++oy) {
                    int ox = ox0;

                    for (; ox + 8 <= ox_end; ox += 8) {
                        __m256 outv = _mm256_setzero_ps();
                        for (int ky = 0; ky < K; ++ky) {
                            for (int kx = 0; kx < K; ++kx) {
                                __m256 inv = _mm256_loadu_ps(&in[(oy + ky) * in_stride + (ox + kx)]);
                                __m256 kv = _mm256_set1_ps(ker[ky * K + kx]);
                                outv = _mm256_fmadd_ps(inv, kv, outv);
                            }
                        }
                        _mm256_storeu_ps(&out[oy * W + ox], outv);
                    }

                    for (; ox < ox_end; ++ox) {
                        float acc = 0.0f;
                        for (int ky = 0; ky < K; ++ky) {
                            for (int kx = 0; kx < K; ++kx) {
                                acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                            }
                        }
                        out[oy * W + ox] = acc;
                    }
                }
            }
        }
    }
}