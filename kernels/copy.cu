#include <graphblas_gpu/kernels/copy.hpp>

namespace graphblas_gpu {
namespace kernels {

template <typename T>
__device__ void vector_copy(const T* src, T* dst, size_t index) {
    dst[index] = src[index];
}

#define INSTANTIATE_VECTOR_COPY(TYPE) \
    template __device__ void vector_copy<TYPE>(const TYPE*, TYPE*, size_t);

INSTANTIATE_VECTOR_COPY(float)
INSTANTIATE_VECTOR_COPY(double)
INSTANTIATE_VECTOR_COPY(int)

} // namespace kernels
} // namespace graphblas_gpu