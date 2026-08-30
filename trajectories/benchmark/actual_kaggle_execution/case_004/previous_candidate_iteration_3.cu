#include <cuda_runtime.h>

__global__ void block_reduce(
    const float* input,
    float* output,
    int n
) {
    __shared__ float shared_values[1024];

    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;

    shared_values[tid] = (idx < n) ? input[idx] : 0.0f;

    __syncthreads();

    for (int s = blockDim.x; s > 1; ) {
        int half = s / 2;
        int stride = s - half;
        if (tid < half) {
            shared_values[tid] += shared_values[tid + stride];
        }
        __syncthreads();
        s = stride;
    }

    if (tid == 0) {
        output[blockIdx.x] = shared_values[0];
    }
}

extern "C" int run_reduction(
    const float* host_input,
    float* host_output,
    int n
) {
    if (n <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    const int block_size = 256;
    const int block_count = (n + block_size - 1) / block_size;

    const size_t input_bytes =
        static_cast<size_t>(n) * sizeof(float);

    const size_t output_bytes =
        static_cast<size_t>(block_count) * sizeof(float);

    float* device_input = nullptr;
    float* device_output = nullptr;

    cudaError_t status = cudaMalloc(
        reinterpret_cast<void**>(&device_input),
        input_bytes
    );

    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }

    status = cudaMalloc(
        reinterpret_cast<void**>(&device_output),
        output_bytes
    );

    if (status != cudaSuccess) {
        cudaFree(device_input);
        return static_cast<int>(status);
    }

    status = cudaMemcpy(
        device_input,
        host_input,
        input_bytes,
        cudaMemcpyHostToDevice
    );

    if (status != cudaSuccess) {
        cudaFree(device_output);
        cudaFree(device_input);
        return static_cast<int>(status);
    }

    block_reduce<<<block_count, block_size>>>(
        device_input,
        device_output,
        n
    );

    status = cudaGetLastError();

    if (status == cudaSuccess) {
        status = cudaDeviceSynchronize();
    }

    if (status == cudaSuccess) {
        status = cudaMemcpy(
            host_output,
            device_output,
            output_bytes,
            cudaMemcpyDeviceToHost
        );
    }

    cudaError_t output_free_status = cudaFree(device_output);
    cudaError_t input_free_status = cudaFree(device_input);

    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }

    if (output_free_status != cudaSuccess) {
        return static_cast<int>(output_free_status);
    }

    return static_cast<int>(input_free_status);
}
