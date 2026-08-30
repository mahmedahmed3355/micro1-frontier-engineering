#include <cuda_runtime.h>

extern "C" int run_transfer(
    const float* host_input,
    float* host_output,
    int n
) {
    float* device_buffer = nullptr;

    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    cudaError_t status = cudaMalloc(
        reinterpret_cast<void**>(&device_buffer),
        bytes
    );

    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }

    status = cudaMemcpy(
        device_buffer,
        host_input,
        bytes,
        cudaMemcpyHostToDevice
    );

    if (status != cudaSuccess) {
        cudaFree(device_buffer);
        return static_cast<int>(status);
    }

    status = cudaMemcpy(
        host_output,
        device_buffer,
        bytes,
        cudaMemcpyDeviceToHost
    );

    const cudaError_t free_status = cudaFree(device_buffer);

    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }

    return static_cast<int>(free_status);
}