#include <cuda_runtime.h>
#include <stdio.h>

int main() {

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);

    printf("GPU %d: %s\n", i, props.name);
    printf("  SMs: %d\n", props.multiProcessorCount);
    // 64 for RTX 2080 (7.5)
    printf("  CUDA Cores: %d\n", props.multiProcessorCount * 64);
    printf("  Compute Capability: %d.%d\n", props.major, props.minor);
    printf("  Memory (VRAM): %.2f GB\n", props.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);

    printf("  Max Threads per SM: %d\n", props.maxThreadsPerMultiProcessor);
    printf("  Max Threads per Block: %d\n", props.maxThreadsPerBlock);
    printf("  Max Shared Memory per Block: %zu KB\n", props.sharedMemPerBlock / 1024);
    printf("  Max Registers per Block: %d\n", props.regsPerBlock);
    printf("  Max Grid Size (x, y, z): (%d, %d, %d)\n",
        props.maxGridSize[0], props.maxGridSize[1], props.maxGridSize[2]);
    printf("  Warp Size: %d\n", props.warpSize);
    
    return 0;
}