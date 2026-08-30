#include <cuda_runtime.h>

extern "C" int run_lifetime_vector(
    const float* host_input,
    float* host_output,
    int n
) {
    float* device_input = nullptr;
    float* device_output = nullptr;

    cudaError_t status = cudaMalloc(
        reinterpret_cast<void**>(&device_input),
        static_cast<size_t>(n) * sizeof(float)
    );
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }

    status = cudaMalloc(
        reinterpret_cast<void**>(&device_output),
        static_cast<size_t>(n) * sizeof(float)
    );
    if (status != cudaSuccess) {
        cudaFree(device_input);
        return static_cast<int>(status);
    }

    status = cudaMemcpy(
        device_input,
        host_input,
        static_cast<size_t>(n) * sizeof(float),
        cudaMemcpyHostToDevice
    );
    if (status != cudaSuccess) {
        cudaFree(device_output);
        cudaFree(device_input);
        return static_cast<int>(status);
    }

    status = cudaMemcpy(
        host_output,
        device_output,
        static_cast<size_t>(n) * sizeof(float),
        cudaMemcpyDeviceToHost
    );
    if (status != cudaSuccess) {
        cudaFree(device_output);
        cudaFree(device_input);
        return static_cast<int>(status);
    }

    cudaError_t input_free_status = cudaFree(device_input);

    if (input_free_status != cudaSuccess) {
        return static_cast<int>(input_free_status);
    }

    return static_cast<int>(input_free_status);
}
