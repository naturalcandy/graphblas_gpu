#ifndef GRAPHBLAS_GPU_EWISE_OPS_H
#define GRAPHBLAS_GPU_EWISE_OPS_H

#include <cuda_runtime.h>
#include <cstddef>

namespace graphblas_gpu {
namespace kernels {

// Template declarations for element-wise operations
template <typename T>
void ewiseAdd(const T* a, const T* b, T* c, size_t size, cudaStream_t stream = 0);

template <typename T>
void ewiseSub(const T* a, const T* b, T* c, size_t size, cudaStream_t stream = 0);

template <typename T>
void ewiseMul(const T* a, const T* b, T* c, size_t size, cudaStream_t stream = 0);

template <typename T>
void ewiseDiv(const T* a, const T* b, T* c, size_t size, cudaStream_t stream = 0);

} // namespace kernels
} // namespace graphblas_gpu

#endif // GRAPHBLAS_GPU_EWISE_OPS_H