
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

extern "C" void launch_vector_add(
    const float* a,
    const float* b,
    float* out,
    int n
);

static void check_cuda(cudaError_t err, const char* where) {
    if (err != cudaSuccess) {
        std::fprintf(stderr,
                     "CUDA_ERROR=%s:%s\n",
                     where,
                     cudaGetErrorString(err));
        std::exit(2);
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s SIZE\n", argv[0]);
        return 2;
    }

    const int n = std::atoi(argv[1]);

    if (n <= 0) {
        std::fprintf(stderr, "invalid size\n");
        return 2;
    }

    std::vector<float> h_a(n);
    std::vector<float> h_b(n);
    std::vector<float> h_out(n, -999.0f);

    for (int i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
    }

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_out = nullptr;

    check_cuda(cudaMalloc(&d_a, n * sizeof(float)), "cudaMalloc(a)");
    check_cuda(cudaMalloc(&d_b, n * sizeof(float)), "cudaMalloc(b)");
    check_cuda(cudaMalloc(&d_out, n * sizeof(float)), "cudaMalloc(out)");

    check_cuda(
        cudaMemcpy(
            d_a,
            h_a.data(),
            n * sizeof(float),
            cudaMemcpyHostToDevice),
        "cudaMemcpy(a)"
    );

    check_cuda(
        cudaMemcpy(
            d_b,
            h_b.data(),
            n * sizeof(float),
            cudaMemcpyHostToDevice),
        "cudaMemcpy(b)"
    );

    cudaEvent_t start;
    cudaEvent_t stop;

    check_cuda(cudaEventCreate(&start), "event_create_start");
    check_cuda(cudaEventCreate(&stop), "event_create_stop");

    check_cuda(cudaEventRecord(start), "event_record_start");

    launch_vector_add(d_a, d_b, d_out, n);

    check_cuda(cudaGetLastError(), "kernel_launch");
    check_cuda(cudaEventRecord(stop), "event_record_stop");
    check_cuda(cudaEventSynchronize(stop), "event_sync");

    float milliseconds = 0.0f;

    check_cuda(
        cudaEventElapsedTime(
            &milliseconds,
            start,
            stop),
        "event_elapsed"
    );

    check_cuda(
        cudaMemcpy(
            h_out.data(),
            d_out,
            n * sizeof(float),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy(out)"
    );

    std::printf("SIZE=%d\n", n);
    std::printf("KERNEL_TIME_MS=%.6f\n", milliseconds);

    std::printf("RESULT_BEGIN\n");

    for (int i = 0; i < n; ++i) {
        std::printf("%.9g", h_out[i]);

        if (i + 1 < n)
            std::printf(",");

        if ((i + 1) % 32 == 0)
            std::printf("\n");
    }

    std::printf("\nRESULT_END\n");

    check_cuda(cudaEventDestroy(start), "event_destroy_start");
    check_cuda(cudaEventDestroy(stop), "event_destroy_stop");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    return 0;
}
