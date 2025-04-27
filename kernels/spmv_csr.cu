#include <graphblas_gpu/kernels/spmv_csr.hpp>

namespace graphblas_gpu{
namespace kernels {

// naive csr spmv implementation
template <typename T>
__device__ void spmv_csr(const size_t* row_offsets,
                        const int* col_indices,
                        const T* values,
                        const T* vector,
                        T* output,
                        size_t num_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;
    T sum = T(0);
    size_t row_start = row_offsets[row];
    size_t row_end = row_offsets[row + 1];
    for (size_t i = row_start; i < row_end; i++) {
        sum += values[i] * vector[col_indices[i]]; 
    }
    output[row] = sum;
}

} // namespace kernels
} // namespace graphblas_gpu