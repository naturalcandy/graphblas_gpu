#include <cuda_runtime.h>
#include <cstdio>
#include <graphblas_gpu/kernels/spmv_csr.hpp>
#include <iostream>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

namespace graphblas_gpu{
namespace kernels {

__global__ void csr_mat_vec_kernel(const int num_rows,
                                  const int* row_offsets,
                                  const int* cols,
                                  const double* vals,
                                  const double* vec,
                                  double* output)
{
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < num_rows) {
    int start = row_offsets[row];
    int end = row_offsets[row+1];
    
    double sum = 0;
    for (int i = start; i < end; i++) {
      sum += vals[i] * vec[cols[i]];
    }
    
    output[row] = sum;
  }
}

void csr_mat_vec_mul(int num_rows,
                int threads_per_block,
                int* row_offsets,
                int* cols,
                double* vals, 
                double* x,
                double* y)
{
  int num_blocks = (num_rows + threads_per_block - 1) / threads_per_block;
  dim3 grid(num_blocks, 1, 1);
  dim3 block(threads_per_block, 1, 1);
  csr_mat_vec_kernel<<<grid, block>>>(num_rows, row_offsets, cols, vals, x, y);
}

__global__ void count_non_zero_kernel(const double* matrix,
                                      int num_rows,
                                      int num_cols,
                                      int* nnz_arr)
{
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < num_rows) {
    int count = 0;
    for (int col = 0; col < num_cols; col++) {
      if (matrix[row * num_cols + col] != 0) {
        count++;
      }
    }
    nnz_arr[row] = count;
  }
}

__global__ void csr_cols_vals_kernel(const double* matrix,  int num_rows, int num_cols, const int* row_offsets, int* cols, double* vals) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < num_rows) {
        int i = row_offsets[row];
        for (int col = 0; col < num_cols; ++col) {
            double val = matrix[row * num_cols + col];
            if (val != 0.0f) {
                vals[i] = val;
                cols[i] = col;
                i++;
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////

void matrix_to_csr(double* matrix,
    int num_rows,
    int num_cols,
    int** row_offsets,
    int** cols,
    double** vals,
    int* nnz,
    int threads_per_block)
{
int blocks = (num_rows + threads_per_block - 1) / threads_per_block;
dim3 grid(blocks, 1, 1);
dim3 block(threads_per_block, 1, 1);

*row_offsets = (int*)malloc((num_rows + 1) * sizeof(int));


// Device memory allocation
double* device_matrix;
cudaMalloc(&device_matrix, num_rows * num_cols * sizeof(double));
cudaMemcpy(device_matrix, matrix, num_rows * num_cols * sizeof(double), cudaMemcpyHostToDevice);

int* device_row_offsets;
cudaMalloc(&device_row_offsets, (num_rows + 1) * sizeof(int));

// Count non-zero entries per row
count_non_zero_kernel<<<grid, block>>>(device_matrix, num_rows, num_cols, device_row_offsets);
cudaDeviceSynchronize();

// Exclusive scan on device
thrust::device_ptr<int> dev_ptr(device_row_offsets);
thrust::exclusive_scan(dev_ptr, dev_ptr + (num_rows + 1), dev_ptr);

// Copy row_offsets back to host
cudaMemcpy(*row_offsets, device_row_offsets, (num_rows + 1) * sizeof(int), cudaMemcpyDeviceToHost);

*nnz = (*row_offsets)[num_rows];

// Allocate device memory for cols and vals
int* device_cols;
cudaMalloc(&device_cols, *nnz * sizeof(int));

double* device_vals;
cudaMalloc(&device_vals, *nnz * sizeof(double));

// Fill in cols and vals arrays
csr_cols_vals_kernel<<<grid, block>>>(device_matrix, num_rows, num_cols, device_row_offsets, device_cols, device_vals);
cudaDeviceSynchronize();

// Allocate host memory and copy results
*cols = (int*)malloc(*nnz * sizeof(int));
*vals = (double*)malloc(*nnz * sizeof(double));
cudaMemcpy(*cols, device_cols, *nnz * sizeof(int), cudaMemcpyDeviceToHost);
cudaMemcpy(*vals, device_vals, *nnz * sizeof(double), cudaMemcpyDeviceToHost);

// Free device memory
cudaFree(device_matrix);
cudaFree(device_row_offsets);
cudaFree(device_cols);
cudaFree(device_vals);
}



} // namespace kernels
} // namespace graphblas_gpu