// matmul_optimized.cpp  STAGE 3: PUT IT ALL TOGETHER
//
// This is the graded function AND the kernel that gets injected into llama.cpp. Combine
// everything you have learned across the whole assignment  loop reordering, register
// blocking and unrolling (Task 1 / Stage 1 here), cache tiling and software prefetch
// (Stage 2)  and TUNE it to be as fast as you can. Your speedup over matmul_naive determines
// your score (see the tier table the harness prints), and this same function will power a
// real LLM inference via `make llama-demo`.

#include <immintrin.h>

#include "matmul.h"

void matmul_optimized(const float* A, const float* B, float* C,
                      int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your best combined implementation.
    for (int i = 0; i < M; ++i) {

        const float* a = A + static_cast<long>(i) * lda;

        for (int j = 0; j < N; ++j) {
            const float* b = B + static_cast<long>(j) * ldb;
            __m256 sum_vec = _mm256_setzero_ps();
            int p = 0;
            for (; p + 8 <= K; p += 8) {
                __m256 av = _mm256_loadu_ps(a + p);
                __m256 bv = _mm256_loadu_ps(b + p);
                sum_vec = _mm256_add_ps(
                    sum_vec,
                    _mm256_mul_ps(av, bv)
                );
            }
            float temp[8];
            _mm256_storeu_ps(temp, sum_vec);
            float acc = 0.0f;
            for (int x = 0; x < 8; ++x)
                acc += temp[x];
            for (; p < K; ++p) {
                acc += a[p] * b[p];
            }

            C[static_cast<long>(i) * ldc + j] = acc;
        }
    }
}
