#include <immintrin.h>
#include <algorithm>
#include "matmul.h"

void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
    
    // 1. CACHE BLOCKING PARAMETERS
    // You may need to tune these block sizes based on your L1/L2 cache capacities
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 64;

    // 2. PREFETCH DISTANCE PARAMETER
    // How many floats ahead the CPU should fetch. Tune this!
    constexpr int PREFETCH_DIST = 16; 

    for (int i = 0; i < M; i += BM) {
        int i_end = std::min(i + BM, M);
        
        for (int k = 0; k < K; k += BK) {
            int k_end = std::min(k + BK, K);
            
            for (int j = 0; j < N; j += BN) {
                int j_end = std::min(j + BN, N);
                
                // --- MICRO-KERNEL ---
                for (int ii = i; ii < i_end; ++ii) {
                    for (int kk = k; kk < k_end; ++kk) {
                        
                        // Prefetch A ahead of time. 
                        // _MM_HINT_T0 fetches into all levels of the cache hierarchy (L1, L2, L3).
                        _mm_prefetch((const char*)&A[ii * lda + kk + PREFETCH_DIST], _MM_HINT_T0);
                        
                        __m256 a_val = _mm256_set1_ps(A[ii * lda + kk]);
                        
                        int jj = j;
                        // SIMD Loop unrolled by 8
                        for (; jj + 8 <= j_end; jj += 8) {
                            
                            // Prefetch B ahead of time
                            _mm_prefetch((const char*)&B[kk * ldb + jj + PREFETCH_DIST], _MM_HINT_T0);
                            
                            // Load B and C
                            __m256 b_val = _mm256_loadu_ps(&B[kk * ldb + jj]);
                            __m256 c_val = _mm256_loadu_ps(&C[ii * ldc + jj]);
                            
                            // FMA: C = A * B + C
                            c_val = _mm256_fmadd_ps(a_val, b_val, c_val);
                            
                            // Store result back to C
                            _mm256_storeu_ps(&C[ii * ldc + jj], c_val);
                        }
                        
                        // Scalar tail loop for remainders
                        for (; jj < j_end; ++jj) {
                            C[ii * ldc + jj] += A[ii * lda + kk] * B[kk * ldb + jj];
                        }
                    }
                }
            }
        }
    }
}