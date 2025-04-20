#ifndef GRAPHBLAS_GPU_OPERATIONS_HPP
#define GRAPHBLAS_GPU_OPERATIONS_HPP

namespace graphblas_gpu{
namespace kernels {

void exclusive_scan(int* device_data, int length);

void matrix_to_csr(double* matrix,
    int num_rows,
    int num_cols,
    int** row_offsets,
    int** cols,
    double** vals,
    int* nnz,
    int threads_per_block);

} // namespace kernels
} // namespace graphblas_gpu

#endif // GRAPHBLAS_GPU_OPERATIONS_HPP
