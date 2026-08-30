#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>

#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4

// Tiled GEMM kernel with shared memory caching and register tiling
__global__ void gemm_optimized(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N)
{
    // Shared memory for tiles with padding to prevent bank conflicts
    __shared__ float s_A[BK][BM + 1];
    __shared__ float s_B[BK][BN + 1];

    int tx = threadIdx.x; // 0..15
    int ty = threadIdx.y; // 0..15
    int thread_id = ty * 16 + tx; // 0..255

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    // Registers for tile computation
    float accum[TM][TN] = {0.0f};
    float reg_a[TM];
    float reg_b[TN];

    // Compute indexing for 256 threads loading 1024 elements (4 per thread)
    int a_k0 = thread_id / BM;
    int a_m0 = thread_id % BM;
    int a_k1 = (thread_id + 256) / BM;
    int a_m1 = (thread_id + 256) % BM;
    int a_k2 = (thread_id + 512) / BM;
    int a_m2 = (thread_id + 512) % BM;
    int a_k3 = (thread_id + 768) / BM;
    int a_m3 = (thread_id + 768) % BM;

    int b_k0 = thread_id / BN;
    int b_n0 = thread_id % BN;
    int b_k1 = (thread_id + 256) / BN;
    int b_n1 = (thread_id + 256) % BN;
    int b_k2 = (thread_id + 512) / BN;
    int b_n2 = (thread_id + 512) % BN;
    int b_k3 = (thread_id + 768) / BN;
    int b_n3 = (thread_id + 768) % BN;

    for (int k_offset = 0; k_offset < N; k_offset += BK) {
        // Load s_A with boundary protection
        int row_a0 = block_row + a_m0;
        int col_a0 = k_offset + a_k0;
        s_A[a_k0][a_m0] = (row_a0 < N && col_a0 < N) ? A[row_a0 * N + col_a0] : 0.0f;

        int row_a1 = block_row + a_m1;
        int col_a1 = k_offset + a_k1;
        s_A[a_k1][a_m1] = (row_a1 < N && col_a1 < N) ? A[row_a1 * N + col_a1] : 0.0f;

        int row_a2 = block_row + a_m2;
        int col_a2 = k_offset + a_k2;
        s_A[a_k2][a_m2] = (row_a2 < N && col_a2 < N) ? A[row_a2 * N + col_a2] : 0.0f;

        int row_a3 = block_row + a_m3;
        int col_a3 = k_offset + a_k3;
        s_A[a_k3][a_m3] = (row_a3 < N && col_a3 < N) ? A[row_a3 * N + col_a3] : 0.0f;

        // Load s_B with boundary protection
        int row_b0 = k_offset + b_k0;
        int col_b0 = block_col + b_n0;
        s_B[b_k0][b_n0] = (row_b0 < N && col_b0 < N) ? B[row_b0 * N + col_b0] : 0.0f;

        int row_b1 = k_offset + b_k1;
        int col_b1 = block_col + b_n1;
        s_B[b_k1][b_n1] = (row_b1 < N && col_b1 < N) ? B[row_b1 * N + col_b1] : 0.0f;

        int row_b2 = k_offset + b_k2;
        int col_b2 = block_col + b_n2;
        s_B[b_k2][b_n2] = (row_b2 < N && col_b2 < N) ? B[row_b2 * N + col_b2] : 0.0f;

        int row_b3 = k_offset + b_k3;
        int col_b3 = block_col + b_n3;
        s_B[b_k3][b_n3] = (row_b3 < N && col_b3 < N) ? B[row_b3 * N + col_b3] : 0.0f;

        __syncthreads();

        // Compute sub-tile product
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                reg_a[m] = s_A[k][ty * TM + m];
            }
            #pragma unroll
            for (int n = 0; n < TN; ++n) {
                reg_b[n] = s_B[k][tx * TN + n];
            }

            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                #pragma unroll
                for (int n = 0; n < TN; ++n) {
                    accum[m][n] += reg_a[m] * reg_b[n];
                }
            }
        }

        __syncthreads();
    }

    // Write back results to global memory C
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        int r = block_row + ty * TM + m;
        if (r < N) {
            #pragma unroll
            for (int n = 0; n < TN; ++n) {
                int c = block_col + tx * TN + n;
                if (c < N) {
                    C[r * N + c] = accum[m][n];
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    int N = 1024;
    if (argc > 1) {
        N = std::atoi(argv[1]);
    }

    size_t size_bytes = (size_t)N * N * sizeof(float);

    std::vector<float> h_A(N * N);
    std::vector<float> h_B(N * N);
    std::vector<float> h_C(N * N, 0.0f);

    // Deterministic input generation with seed 12345
    std::srand(12345);
    for (size_t i = 0; i < (size_t)N * N; ++i) {
        h_A[i] = -1.0f + 2.0f * ((float)std::rand() / (float)RAND_MAX);
    }
    for (size_t i = 0; i < (size_t)N * N; ++i) {
        h_B[i] = -1.0f + 2.0f * ((float)std::rand() / (float)RAND_MAX);
    }

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaMalloc(&d_A, size_bytes);
    cudaMalloc(&d_B, size_bytes);
    cudaMalloc(&d_C, size_bytes);

    cudaMemcpy(d_A, h_A.data(), size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size_bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((N + BN - 1) / BN, (N + BM - 1) / BM);

    // 1 Warm-up execution
    gemm_optimized<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // 100 Measured kernel executions
    const int NUM_RUNS = 100;
    std::vector<float> times(NUM_RUNS);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < NUM_RUNS; ++i) {
        cudaEventRecord(start);
        gemm_optimized<<<grid, block>>>(d_A, d_B, d_C, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        times[i] = ms;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(h_C.data(), d_C, size_bytes, cudaMemcpyDeviceToHost);

    // Calculate median kernel execution time
    std::sort(times.begin(), times.end());
    float median_time = (times[49] + times[50]) / 2.0f;

    // Report required output fields
    std::cout << "SIZE=" << N << std::endl;
    std::cout << "KERNEL_TIME_MS=" << median_time << std::endl;
    std::cout << "RESULT_SAMPLE=";
    for (int i = 0; i < 10; ++i) {
        std::cout << h_C[i] << (i == 9 ? "" : ",");
    }
    std::cout << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}