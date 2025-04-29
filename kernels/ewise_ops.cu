#include <graphblas_gpu/kernels/ewise_ops.hpp>

namespace graphblas_gpu {
namespace kernels {

// Regular element-wise operations
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

template <typename T>
__device__ inline void ewise_or(const T* a, const T* b,
                                T* c, size_t i)
{
    const T zero = T(0);
    const T one  = T(1);
    c[i] = (a[i] != zero || b[i] != zero) ? one : zero;
}


// In-place element-wise operations
template <typename T>
__device__ inline void ewise_add_inplace(T* a, const T* b, size_t index) {
    a[index] += b[index];
}

template <typename T>
__device__ inline void ewise_sub_inplace(T* a, const T* b, size_t index) {
    a[index] -= b[index];
}

template <typename T>
__device__ inline void ewise_mul_inplace(T* a, const T* b, size_t index) {
    a[index] *= b[index];
}

template <typename T>
__device__ inline void ewise_div_inplace(T* a, const T* b, size_t index) {
    a[index] /= b[index];
}

template <typename T>
__device__ inline void ewise_or_inplace(T* dst, const T* src, size_t i)
{
    const T zero = T(0);
    const T one  = T(1);
    if (src[i] != zero) dst[i] = one;
}

#define INSTANTIATE_EWISE_FUNCTIONS(TYPE) \
    template __device__ void ewise_add<TYPE>(const TYPE*, const TYPE*, TYPE*, size_t); \
    template __device__ void ewise_sub<TYPE>(const TYPE*, const TYPE*, TYPE*, size_t); \
    template __device__ void ewise_mul<TYPE>(const TYPE*, const TYPE*, TYPE*, size_t); \
    template __device__ void ewise_div<TYPE>(const TYPE*, const TYPE*, TYPE*, size_t); \
    template __device__ void ewise_add_inplace<TYPE>(TYPE*, const TYPE*, size_t); \
    template __device__ void ewise_sub_inplace<TYPE>(TYPE*, const TYPE*, size_t); \
    template __device__ void ewise_mul_inplace<TYPE>(TYPE*, const TYPE*, size_t); \
    template __device__ void ewise_div_inplace<TYPE>(TYPE*, const TYPE*, size_t); \
    template __device__ void ewise_or<TYPE>(const TYPE*, const TYPE*, TYPE*, size_t); \
    template __device__ void ewise_or_inplace<TYPE>(TYPE*, const TYPE*, size_t);

INSTANTIATE_EWISE_FUNCTIONS(float)
INSTANTIATE_EWISE_FUNCTIONS(double)
INSTANTIATE_EWISE_FUNCTIONS(int)

} // namespace kernels
} // namespace graphblas_gpu