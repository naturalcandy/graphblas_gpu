#ifndef GRAPHBLAS_GPU_COPY_HPP
#define GRAPHBLAS_GPU_COPY_HPP

#include <cuda_runtime.h>
#include <cstddef>

namespace graphblas_gpu {
namespace kernels {

// Copy elements from one vector to another
template <typename T>
__device__ void vector_copy(const T* src, T* dst, size_t index);

} // namespace kernels
} // namespace graphblas_gpu

#endif // GRAPHBLAS_GPU_COPY_HPP