#ifndef GRAPHBLAS_GPU_EWISE_OPS_HPP
#define GRAPHBLAS_GPU_EWISE_OPS_HPP

#include <cuda_runtime.h>
#include <cstddef>

namespace graphblas_gpu {
namespace kernels {

template <typename T>
__device__ inline void ewise_add(const T* a, const T* b, T* c, 
                               size_t index);

template <typename T>
__device__ inline void ewise_sub(const T* a, const T* b, T* c, 
                               size_t index);

template <typename T>
__device__ inline void ewise_mul(const T* a, const T* b, T* c, 
                               size_t index);

template <typename T>
__device__ inline void ewise_div(const T* a, const T* b, T* c, 
                               size_t index);

} // namespace kernels
} // namespace graphblas_gpu

#endif // GRAPHBLAS_GPU_EWISE_OPS_HPP