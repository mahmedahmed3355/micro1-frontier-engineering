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

    /*
     * BUG:
     * The device-to-host copy is initiated through an asynchronous
     * transfer path, but the function returns without establishing the
     * completion guarantee required by the caller.
     *
     * The caller expects host_output to contain the completed result
     * when this function returns.
     */
    status = cudaMemcpyAsync(
        host_output,
        device_buffer,
        bytes,
        cudaMemcpyDeviceToHost,
        0
    );

    cudaFree(device_buffer);

    return static_cast<int>(status);
}
