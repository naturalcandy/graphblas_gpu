#include <graphblas_gpu/kernels/spmv_ell.hpp>
#include <cuda_runtime.h>
#include <cuda.h>

namespace graphblas_gpu{
namespace kernels {

template <typename T>
__global__ void spmv_ell_kernel(const int* col_indices,
                         const T* values,
                         const T* vector,
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
    output[row] = sum;
}

template <typename T>
void spmv_ell(const int* col_indices,
    const T* values,
    const T* vector,
    T* output,
    size_t num_rows,
    size_t max_nnz_per_row){

        spmv_ell_kernel<T><<<(num_rows + 255) / 256, 256>>>(
            col_indices,
            values,
            vector,
            output,
            num_rows,
            max_nnz_per_row);
        cudaDeviceSynchronize();
    }
    
} // namespace kernels
} // namespace graphblas_gpu