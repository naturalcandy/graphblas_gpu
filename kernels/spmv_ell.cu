#include <graphblas_gpu/kernels/spmv_ell.hpp>

namespace graphblas_gpu{
namespace kernels {

template <typename T>
__device__ void spmv_ell(const int* col_indices,
                         const T* values,
                         const T* vector,
                         const T* mask,
                         bool mask_enabled,
                         T* output,
                         size_t num_rows,
                         size_t max_nnz_per_row) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    T sum = T(0);
    size_t row_offset = row * max_nnz_per_row;

    for (size_t i = 0; i < max_nnz_per_row; i++) {
        int col = col_indices[row_offset + i];
        if (col != -1) {
            sum += values[row_offset + i] * vector[col];
        }
    }
    if (mask_enabled) {
        output[row] = sum * T(mask[row] != T(0));
    } else {
        output[row] = sum;
    }
}


// Logical OR-AND semiring
template <typename T>
__device__ void spmv_ell_logical(const int* col_indices,
                                const T* values,
                                const T* vector,
                                const T* mask,
                                bool mask_enabled,
                                T* output,
                                size_t num_rows,
                                size_t max_nnz_per_row) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    bool result = false;
    size_t row_offset = row * max_nnz_per_row;

    for (size_t i = 0; i < max_nnz_per_row; i++) {
        int col = col_indices[row_offset + i];
        if (col != -1) {
            bool val = (values[row_offset + i] != 0) && (vector[col] != 0);
            result = result || val;
            if (result) break;  
        }
    }
    
    if (mask_enabled) {
        output[row] = static_cast<T>(result && (mask[row] != 0));
    } else {
        output[row] = static_cast<T>(result);
    }
}



} // namespace kernels
} // namespace graphblas_gpu