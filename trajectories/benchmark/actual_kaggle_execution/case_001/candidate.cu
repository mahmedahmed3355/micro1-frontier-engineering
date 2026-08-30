#include <cuda_runtime.h>

__global__ void vector_add_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

extern "C" void launch_vector_add(
    const float* a,
    const float* b,
    float* out,
    int n
) {
    if (n <= 0) return;

    constexpr int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    vector_add_kernel<<<grid_size, block_size>>>(a, b, out, n);
}