#include <cuda_runtime.h>
#include <cstdio>

__global__ void csr_kernel(const int num_rows,
                                  const int* row_offsets,
                                  const int* cols,
                                  const double* vals,
                                  const double* vec,
                                  double* output)
{
  const int row = threadIdx.x + blockDim.x * blockIdx.x;
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

void scalar_csr(int num_rows,
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
  scalar_csr_kernel<<<grid, block>>>(num_rows, row_offsets, cols, vals, x, y);
}

/*Source: Assignment 2 exclusive scan*/
__global__ void exclusive_scan_upsweep_kernel(int* device_data, int length, int twod)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int twod1 = twod*2;
    int i = index*twod1;
    if (i+twod1-1 < length) {
        device_data[i+twod1-1] += device_data[i+twod-1];
    }
}

/*Source: Assignment 2 exclusive scan*/
__global__ void exclusive_scan_downsweep_kernel(int* device_data, int length, int twod)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int twod1 = twod*2;
    int i = index*twod1;
    if (i+twod1-1 < length){
        int t = device_data[i+twod-1];
        device_data[i+twod-1] = device_data[i+twod1-1];
        // change twod1 below to twod to reverse prefix sum.
        device_data[i+twod1-1] += t;
    }

}

/*Source: Assignment 2 exclusive scan*/
void exclusive_scan(int* device_data, int length)
{
    
    const int N = nextPow2(length);
    const int threadsPerBlock = 512;
    for (int twod = 1; twod < N; twod*=2){
        int twod1 = twod*2;
        int numThreads = (N + twod1 - 1) / twod1;
        int blocks = (numThreads + threadsPerBlock - 1) / threadsPerBlock;
        exclusive_scan_upsweep_kernel<<<blocks, threadsPerBlock>>>(device_data, N, twod);
        cudaDeviceSynchronize();

    }
    

    cudaMemset(device_data+N-1, 0, sizeof(int));

    for (int twod = N/2; twod >= 1; twod /= 2){
        int twod1 = twod*2;
        int numThreads = (N + twod1 - 1) / twod1;
        int blocks = (numThreads + threadsPerBlock - 1) / threadsPerBlock;
        exclusive_scan_downsweep_kernel<<<blocks, threadsPerBlock>>>(device_data, N, twod);
        cudaDeviceSynchronize();

    }
}

__global__ count_non_zeros_kernel(const double* matrix,
                                      int num_rows,
                                      int num_cols,
                                      int* nnz_arr)
{
  const int row = threadIdx.x + blockDim.x * blockIdx.x;
  if (row < num_rows) {
    int count = 0;
    for (int col = 0; col < num_cols; col++) {
      if (matrix[row * num_cols + col] != 0) {
        count++;
      }
    }
    nnz_arr[row + 1] = count;
  }
}

__global__ void csr_cols_vals_kernel(const double* matrix,  int num_rows, int num_cols, const int* row_offsets, int* cols, double* vals) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        int i = row_offsets[row];
        for (int col = 0; col < num_cols; ++col) {
            double val = matrix[row * num_cols + col];
            if (val != 0.0f) {
                vals[index] = val;
                cols[index] = col;
                i++;
            }
        }
    }
}

void matrix_to_csr( double* matrix,
                    int threads_per_block,
                    int num_rows,
                    int num_cols,
                    int* row_offsets,
                    int* cols,
                    double* vals)
{
    int blocks = (num_rows + threads_per_block - 1) / threads_per_block;
    dim3 grid(blocks, 1, 1);
    dim3 block(threads_per_block, 1, 1);

    double* device_matrix;
    cudaMalloc(&device_matrix, num_rows * num_cols * sizeof(double));
    cudaMemcpy(device_matrix, matrix, num_rows * num_cols * sizeof(double), cudaMemcpyHostToDevice);

    int* device_row_offsets;
    cudaMalloc(&device_row_offsets, num_rows * sizeof(int));

    count_non_zeros_kernel<<<grid, block>>>(device_matrix, num_rows, num_cols, device_row_offsets);
    exclusive_scan(device_row_offsets, num_rows);

    row_offsets = (int*)malloc(num_rows * sizeof(int));
    cudaMemcpy(row_offsets, device_row_offsets, num_rows * sizeof(int), cudaMemcpyDeviceToHost);
    int nnz = row_offsets[num_rows]; // number of non zeros

    int* device_cols;
    cudaMalloc(&device_cols, nnz * sizeof(int));

    double* device_vals;
    cudaMalloc(&device_vals, nnz * sizeof(double));

    csr_cols_vals_kernel<<<grid, block>>>(device_matrix, num_rows, num_cols, device_row_offsets, device_cols, device_vals);


    cols = (int*)malloc( * sizeof(int));
    vals = (double*)malloc( * sizeof(double));
    cudaMemcpy(cols, device_cols, num_rows * num_cols * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(vals, device_vals, num_rows * num_cols * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(device_matrix);
    cudaFree(device_row_offsets);
    cudaFree(device_cols);
}

