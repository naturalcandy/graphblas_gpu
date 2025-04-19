#include "ewise_ops.h"

namespace graphblas_gpu {
namespace kernels {

template <typename T>
__device__ inline void ewise_add(const T* a, const T* b, T* c, size_t index) {
    c[index] = a[index] + b[index];
}

template <typename T>
__device__ inline void ewise_sub(const T* a, const T* b, T* c, size_t index) {
    c[index] = a[index] - b[index];
}

template <typename T>
__device__ inline void ewise_mul(const T* a, const T* b, T* c, size_t index) {
    c[index] = a[index] * b[index];
}

template <typename T>
__device__ inline void ewise_div(const T* a, const T* b, T* c, size_t index) {
    c[index] = a[index] / b[index];
}

#define INSTANTIATE_EWISE_FUNCTIONS(TYPE) \
    template __device__ void ewise_add<TYPE>(const TYPE*, const TYPE*, TYPE*, size_t); \
    template __device__ void ewise_sub<TYPE>(const TYPE*, const TYPE*, TYPE*, size_t); \
    template __device__ void ewise_mul<TYPE>(const TYPE*, const TYPE*, TYPE*, size_t); \
    template __device__ void ewise_div<TYPE>(const TYPE*, const TYPE*, TYPE*, size_t);

INSTANTIATE_EWISE_FUNCTIONS(float)
INSTANTIATE_EWISE_FUNCTIONS(double)
INSTANTIATE_EWISE_FUNCTIONS(int)

} // namespace kernels
} // namespace graphblas_gpu