#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>

__global__ void vector_add_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int n
) {
    int n4 = n / 4;
    const float4* a4 = reinterpret_cast<const float4*>(a);
    const float4* b4 = reinterpret_cast<const float4*>(b);
    float4* out4 = reinterpret_cast<float4*>(out);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n4; i += stride) {
        float4 va = a4[i];
        float4 vb = b4[i];
        float4 vc;
        vc.x = va.x + vb.x;
        vc.y = va.y + vb.y;
        vc.z = va.z + vb.z;
        vc.w = va.w + vb.w;
        out4[i] = vc;
    }

    int tail_start = n4 * 4;
    for (int i = tail_start + idx; i < n; i += stride) {
        out[i] = a[i] + b[i];
    }
}

__global__ void vector_add_kernel_scalar(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        out[i] = a[i] + b[i];
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

    bool aligned = (reinterpret_cast<uintptr_t>(a) % 16 == 0) &&
                   (reinterpret_cast<uintptr_t>(b) % 16 == 0) &&
                   (reinterpret_cast<uintptr_t>(out) % 16 == 0);

    if (aligned) {
        int num_elements_4 = (n + 3) / 4;
        int grid_size = (num_elements_4 + block_size - 1) / block_size;
        if (grid_size > 65535) grid_size = 65535;
        if (grid_size < 1) grid_size = 1;
        vector_add_kernel<<<grid_size, block_size>>>(a, b, out, n);
    } else {
        int grid_size = (n + block_size - 1) / block_size;
        if (grid_size > 65535) grid_size = 65535;
        if (grid_size < 1) grid_size = 1;
        vector_add_kernel_scalar<<<grid_size, block_size>>>(a, b, out, n);
    }
}

int main(int argc, char** argv) {
    int n = 1024;
    if (argc > 1) {
        n = std::atoi(argv[1]);
    }

    size_t bytes = static_cast<size_t>(n) * sizeof(float);
    std::vector<float> h_a(n);
    std::vector<float> h_b(n);
    std::vector<float> h_out(n, 0.0f);

    for (int i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = 2.0f * static_cast<float>(i);
    }

    float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_out, bytes);

    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    // Warm-up kernel execution
    launch_vector_add(d_a, d_b, d_out, n);
    cudaDeviceSynchronize();

    constexpr int num_runs = 100;
    std::vector<float> times(num_runs, 0.0f);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < num_runs; ++i) {
        cudaEventRecord(start, 0);
        launch_vector_add(d_a, d_b, d_out, n);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        times[i] = ms;
    }

    cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost);

    std::sort(times.begin(), times.end());
    float median_time = times[num_runs / 2];

    std::printf("SIZE=%d\n", n);
    std::printf("KERNEL_TIME_MS=%.6f\n", median_time);

    std::printf("RESULT_SAMPLE=");
    for (int i = 0; i < 10; ++i) {
        float val = (i < n) ? h_out[i] : 0.0f;
        std::printf("%.6f%s", val, (i == 9) ? "" : ",");
    }
    std::printf("\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    return 0;
}