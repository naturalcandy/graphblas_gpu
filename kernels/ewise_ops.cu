#include "ewise_ops.h"

namespace graphblas_gpu {
namespace kernels {

// Kernel definitions
template <typename T>
__global__ void ewiseAddKernel(const T* a, const T* b, T* c, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

template <typename T>
__global__ void ewiseSubKernel(const T* a, const T* b, T* c, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] - b[idx];
    }
}

template <typename T>
__global__ void ewiseMulKernel(const T* a, const T* b, T* c, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] * b[idx];
    }
}

template <typename T>
__global__ void ewiseDivKernel(const T* a, const T* b, T* c, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] / b[idx];
    }
}


///////////////////////////////////////////////////////////////////////////////

// Host implementations
template <typename T>
void ewiseAdd(const T* a, const T* b, T* c, size_t size, cudaStream_t stream) {
    if (size == 0) return;
    
    constexpr int BLOCK_SIZE = 256;
    int grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    ewiseAddKernel<T><<<grid_size, BLOCK_SIZE, 0, stream>>>(a, b, c, size);
}

template <typename T>
void ewiseSub(const T* a, const T* b, T* c, size_t size, cudaStream_t stream) {
    if (size == 0) return;
    
    constexpr int BLOCK_SIZE = 256;
    int grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    ewiseSubKernel<T><<<grid_size, BLOCK_SIZE, 0, stream>>>(a, b, c, size);
}

template <typename T>
void ewiseMul(const T* a, const T* b, T* c, size_t size, cudaStream_t stream) {
    if (size == 0) return;
    
    constexpr int BLOCK_SIZE = 256;
    int grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    ewiseMulKernel<T><<<grid_size, BLOCK_SIZE, 0, stream>>>(a, b, c, size);
}

template <typename T>
void ewiseDiv(const T* a, const T* b, T* c, size_t size, cudaStream_t stream) {
    if (size == 0) return;
    
    constexpr int BLOCK_SIZE = 256;
    int grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    ewiseDivKernel<T><<<grid_size, BLOCK_SIZE, 0, stream>>>(a, b, c, size);
}

// Explicit instantiations for supported types
#define INSTANTIATE_EWISE_FUNCTIONS(TYPE) \
    template void ewiseAdd<TYPE>(const TYPE*, const TYPE*, TYPE*, size_t, cudaStream_t); \
    template void ewiseSub<TYPE>(const TYPE*, const TYPE*, TYPE*, size_t, cudaStream_t); \
    template void ewiseMul<TYPE>(const TYPE*, const TYPE*, TYPE*, size_t, cudaStream_t); \
    template void ewiseDiv<TYPE>(const TYPE*, const TYPE*, TYPE*, size_t, cudaStream_t);

INSTANTIATE_EWISE_FUNCTIONS(float)
INSTANTIATE_EWISE_FUNCTIONS(double)
INSTANTIATE_EWISE_FUNCTIONS(int)

} // namespace kernels
} // namespace graphblas_gpu