#include <cuda_runtime.h>

__global__ void add_scalar(
    const float* input,
    float* output,
    float value,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        output[idx] = input[idx] + value;
    }
}

extern "C" int run_stream_dependency(
    const float* host_input,
    float* host_output,
    float value,
    int n
) {
    float* device_buffer = nullptr;
    cudaStream_t producer_stream = nullptr;
    cudaStream_t consumer_stream = nullptr;

    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    cudaError_t status = cudaMalloc(
        reinterpret_cast<void**>(&device_buffer),
        bytes
    );

    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }

    status = cudaStreamCreate(&producer_stream);
    if (status != cudaSuccess) {
        cudaFree(device_buffer);
        return static_cast<int>(status);
    }

    status = cudaStreamCreate(&consumer_stream);
    if (status != cudaSuccess) {
        cudaStreamDestroy(producer_stream);
        cudaFree(device_buffer);
        return static_cast<int>(status);
    }

    status = cudaMemcpyAsync(
        device_buffer,
        host_input,
        bytes,
        cudaMemcpyHostToDevice,
        producer_stream
    );

    if (status != cudaSuccess) {
        cudaStreamDestroy(consumer_stream);
        cudaStreamDestroy(producer_stream);
        cudaFree(device_buffer);
        return static_cast<int>(status);
    }

    add_scalar<<<
        (n + 255) / 256,
        256,
        0,
        consumer_stream
    >>>(
        device_buffer,
        device_buffer,
        value,
        n
    );

    status = cudaGetLastError();
    if (status != cudaSuccess) {
        cudaStreamDestroy(consumer_stream);
        cudaStreamDestroy(producer_stream);
        cudaFree(device_buffer);
        return static_cast<int>(status);
    }

    status = cudaMemcpyAsync(
        host_output,
        device_buffer,
        bytes,
        cudaMemcpyDeviceToHost,
        consumer_stream
    );

    if (status != cudaSuccess) {
        cudaStreamDestroy(consumer_stream);
        cudaStreamDestroy(producer_stream);
        cudaFree(device_buffer);
        return static_cast<int>(status);
    }

    /*
     * BUG:
     * consumer_stream starts consuming device_buffer without an explicit
     * dependency on producer_stream.
     *
     * The host-to-device copy is ordered in producer_stream, while the
     * kernel and device-to-host copy are ordered in consumer_stream.
     * There is no CUDA event establishing the producer -> consumer
     * dependency.
     */

    status = cudaStreamSynchronize(consumer_stream);

    cudaStreamDestroy(consumer_stream);
    cudaStreamDestroy(producer_stream);
    cudaFree(device_buffer);

    return static_cast<int>(status);
}
