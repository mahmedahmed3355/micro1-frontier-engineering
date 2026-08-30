
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

static void check_cuda(cudaError_t err, const char* where) {
    if (err != cudaSuccess) {
        std::fprintf(
            stderr,
            "CUDA_ERROR=%s:%s\n",
            where,
            cudaGetErrorString(err)
        );
        std::exit(2);
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        return 2;
    }

    const int n = std::atoi(argv[1]);

    if (n <= 0) {
        return 2;
    }

    std::vector<float> a(n);
    std::vector<float> b(n);
    std::vector<float> out(n, -999.0f);

    for (int i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(2 * i);
    }

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_out = nullptr;

    check_cuda(
        cudaMalloc(&d_a, n * sizeof(float)),
        "cudaMalloc(a)"
    );

    check_cuda(
        cudaMalloc(&d_b, n * sizeof(float)),
        "cudaMalloc(b)"
    );

    check_cuda(
        cudaMalloc(&d_out, n * sizeof(float)),
        "cudaMalloc(out)"
    );

    check_cuda(
        cudaMemcpy(
            d_a,
            a.data(),
            n * sizeof(float),
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy(a)"
    );

    check_cuda(
        cudaMemcpy(
            d_b,
            b.data(),
            n * sizeof(float),
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy(b)"
    );

    launch_vector_add(
        d_a,
        d_b,
        d_out,
        n
    );

    check_cuda(
        cudaGetLastError(),
        "kernel_launch"
    );

    check_cuda(
        cudaDeviceSynchronize(),
        "cudaDeviceSynchronize"
    );

    check_cuda(
        cudaMemcpy(
            out.data(),
            d_out,
            n * sizeof(float),
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy(out)"
    );

    std::printf("SIZE=%d\n", n);
    std::printf("RESULT_BEGIN\n");

    for (int i = 0; i < n; ++i) {
        std::printf("%.9g", out[i]);

        if (i + 1 < n)
            std::printf(",");

        if ((i + 1) % 32 == 0)
            std::printf("\n");
    }

    std::printf("\nRESULT_END\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    return 0;
}
