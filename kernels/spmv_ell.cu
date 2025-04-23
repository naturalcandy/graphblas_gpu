#include <graphblas_gpu/kernels/spmv_ell.hpp>

namespace graphblas_gpu{
namespace kernels {

template <typename T>
__device__ void spmv_ell(const size_t* col_indices,
                         const T* values,
                         const T* vector,
                         T* output,
                         size_t num_rows,
                         size_t max_nnz_per_row) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    T sum = T(0);
    for (size_t i = 0; i < max_nnz_per_row; i++) {
        size_t idx = row * max_nnz_per_row + i;
        size_t col = col_indices[idx];
        T val = values[idx];

        if (val != T(0)) {
            sum += val * vector[col];
        }
    }
    output[row] = sum;
}

} // namespace kernels
} // namespace graphblas_gpu