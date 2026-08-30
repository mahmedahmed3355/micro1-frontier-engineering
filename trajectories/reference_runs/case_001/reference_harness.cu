#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" void launch_vector_add(
    const float* a,
    const float* b,
    float* out,
    int n
);

int main(int argc, char** argv) {
    if (argc != 2) {
        return 10;
    }

    const int n = std::atoi(argv[1]);

    if (n <= 0) {
        return 11;
    }

    std::vector<float> host_a(n);
    std::vector<float> host_b(n);
    std::vector<float> host_out(n, 0.0f);

    for (int i = 0; i < n; ++i) {
        host_a[i] = static_cast<float>(i);
        host_b[i] = static_cast<float>(2 * i);
    }

    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_out = nullptr;

    if (cudaMalloc(
            reinterpret_cast<void**>(&device_a),
            n * sizeof(float)) != cudaSuccess) {
        return 20;
    }

    if (cudaMalloc(
            reinterpret_cast<void**>(&device_b),
            n * sizeof(float)) != cudaSuccess) {
        cudaFree(device_a);
        return 21;
    }

    if (cudaMalloc(
            reinterpret_cast<void**>(&device_out),
            n * sizeof(float)) != cudaSuccess) {
        cudaFree(device_a);
        cudaFree(device_b);
        return 22;
    }

    if (cudaMemcpy(
            device_a,
            host_a.data(),
            n * sizeof(float),
            cudaMemcpyHostToDevice) != cudaSuccess) {
        return 23;
    }

    if (cudaMemcpy(
            device_b,
            host_b.data(),
            n * sizeof(float),
            cudaMemcpyHostToDevice) != cudaSuccess) {
        return 24;
    }

    if (cudaMemset(
            device_out,
            0,
            n * sizeof(float)) != cudaSuccess) {
        return 25;
    }

    cudaEvent_t start;
    cudaEvent_t stop;

    if (cudaEventCreate(&start) != cudaSuccess) {
        return 26;
    }

    if (cudaEventCreate(&stop) != cudaSuccess) {
        cudaEventDestroy(start);
        return 27;
    }

    launch_vector_add(
        device_a,
        device_b,
        device_out,
        n
    );

    if (cudaDeviceSynchronize() != cudaSuccess) {
        return 28;
    }

    if (cudaEventRecord(start) != cudaSuccess) {
        return 29;
    }

    launch_vector_add(
        device_a,
        device_b,
        device_out,
        n
    );

    if (cudaEventRecord(stop) != cudaSuccess) {
        return 30;
    }

    if (cudaEventSynchronize(stop) != cudaSuccess) {
        return 31;
    }

    float kernel_time_ms = 0.0f;

    if (cudaEventElapsedTime(
            &kernel_time_ms,
            start,
            stop) != cudaSuccess) {
        return 32;
    }

    if (cudaMemcpy(
            host_out.data(),
            device_out,
            n * sizeof(float),
            cudaMemcpyDeviceToHost) != cudaSuccess) {
        return 33;
    }

    if (cudaDeviceSynchronize() != cudaSuccess) {
        return 34;
    }

    for (int i = 0; i < n; ++i) {
        const float expected =
            host_a[i] + host_b[i];

        if (host_out[i] != expected) {
            return 35;
        }
    }

    std::printf(
        "KERNEL_TIME_MS=%.6f\n",
        kernel_time_ms
    );

    std::printf(
        "SIZE=%d\n",
        n
    );

    std::printf(
        "RESULT_FIRST=%.6f\n",
        host_out.front()
    );

    std::printf(
        "RESULT_LAST=%.6f\n",
        host_out.back()
    );

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_out);

    return 0;
}
