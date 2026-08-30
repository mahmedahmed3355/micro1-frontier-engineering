#include <cuda_runtime.h>

__global__ void vector_add_kernel(
    const float* a,
    const float* b,
    float* out,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // BUG:
    // The kernel assumes the launch grid contains exactly enough
    // threads for the input. Non-multiple-of-block-size workloads
    // can access memory beyond the logical array boundary.
    out[idx] = a[idx] + b[idx];
}

extern "C" void launch_vector_add(
    const float* a,
    const float* b,
    float* out,
    int n
) {
    constexpr int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    vector_add_kernel<<<grid_size, block_size>>>(a, b, out, n);
}
