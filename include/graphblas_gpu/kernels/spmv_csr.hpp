#ifndef GRAPHBLAS_GPU_OPERATIONS_HPP
#define GRAPHBLAS_GPU_OPERATIONS_HPP


#include <cuda_runtime.h>
#include <cstddef>

namespace graphblas_gpu{
namespace kernels {

// Thread -> Row mapping

// Arithmetic semiring
template <typename T>
__device__ void spmv_csr(const size_t* row_offsets,
                        const int* col_indices,
                        const T* values,
                        const T* vector,
                        const T* mask,
                        bool mask_enabled,
                        T* output,
                        size_t num_rows);

// OR-AND Semiring
template <typename T>
__device__ void spmv_csr_logical(const size_t* row_offsets,
                                    const int* col_indices,
                                    const T* values,
                                    const T* vector,
                                    const T* mask,
                                    bool mask_enabled,
                                    T* output,
                                    size_t num_rows);


// Warp Cooperative
template <typename T, int THREADS_PER_ROW>
__device__ void spmv_csr_vector_arithmetic(const size_t* row_offsets,
                                           const int* col_indices,
                                           const T* values,
                                           const T* vector,
                                           const T* mask,
                                           bool mask_enabled,
                                           T* output,
                                           size_t num_rows);

// OR-AND Semiring, Warp Cooperative
template <typename T, int THREADS_PER_ROW>
__device__ void spmv_csr_vector_logical(const size_t* row_offsets,
                                        const int* col_indices,
                                        const T* values,
                                        const T* vector,
                                        const T* mask,
                                        bool mask_enabled,
                                        T* output,
                                        size_t num_rows);

} // namespace kernels
} // namespace graphblas_gpu

#endif // GRAPHBLAS_GPU_OPERATIONS_HPP
