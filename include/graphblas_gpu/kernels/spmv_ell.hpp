#ifndef GRAPHBLAS_GPU_SPMV_ELL_HPP
#define GRAPHBLAS_GPU_SPMV_ELL_HPP

namespace graphblas_gpu{
namespace kernels {

// Arithmetic semiring
template <typename T>
__device__ void spmv_ell(const int* col_indices,
                            const T* values,
                            const T* vector,
                            const T* mask,
                            bool mask_enabled,
                            T* output,
                            size_t num_rows,
                            size_t max_nnz_per_row);

// OR-AND semiring
template <typename T>
__device__ void spmv_ell_logical(const int* col_indices,
                                const T* values,
                                const T* vector,
                                const T* mask,
                                bool mask_enabled,
                                T* output,
                                size_t num_rows,
                                size_t max_nnz_per_row);

} // namespace kernels
} // namespace graphblas_gpu
#endif // GRAPHBLAS_GPU_SPMV_ELL_HPP