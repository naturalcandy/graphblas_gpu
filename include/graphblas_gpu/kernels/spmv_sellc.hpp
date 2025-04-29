#ifndef GRAPHBLAS_GPU_SPMV_SELLC_HPP
#define GRAPHBLAS_GPU_SPMV_SELLC_HPP

namespace graphblas_gpu{
namespace kernels {
    
// Arithmetic semiring
template <typename T>
__device__ void spmv_sell_c(const size_t* slice_offsets,
                            const int* col_indices,
                            const T* values,
                            size_t num_rows,
                            size_t c,
                            const T* vector,
                            const T* mask,
                            bool mask_enabled,
                            T* output
                            );

// Logical OR-AND semiring
template <typename T>
__device__ void spmv_sell_c_logical(const size_t* slice_offsets,
                                   const int* col_indices,
                                   const T* values,
                                   size_t num_rows,
                                   size_t c,
                                   const T* vector,
                                   const T* mask,
                                   bool mask_enabled,
                                   T* output);

    
} // namespace kernels
} // namespace graphblas_gpu
#endif // GRAPHBLAS_GPU_SPMV_SELLC_HPP