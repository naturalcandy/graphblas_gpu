#include <cuda_runtime.h>
#include <cstdio>

__global__ void test_kernel() {
    printf("CUDA kernel successfully executed! ThreadIdx.x = %d\n", threadIdx.x);
}

extern "C" void run_test_kernel() {
    test_kernel<<<1, 5>>>(); // Launch kernel with 5 threads
    cudaDeviceSynchronize(); // Wait for completion
}
