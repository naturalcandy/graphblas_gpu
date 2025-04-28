#include <graphblas_gpu/kernels/graphblas_kernels.hpp>
#include <graphblas_gpu/graph_classifier.hpp>
#include <cstdlib>  // for rand, srand
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda.h>


void generate_random_sparse_matrix(uint64_t num_rows, uint64_t num_cols, float sparsity,
                                   float*& dense_matrix) {
    dense_matrix = new float[num_rows * num_cols];
    std::srand(static_cast<unsigned>(std::time(nullptr)));  // Seed RNG

    for (uint64_t i = 0; i < num_rows; i++) {
        for (uint64_t j = 0; j < num_cols; ++j) {
            float r = static_cast<float>(rand()) / RAND_MAX;
            if (r < sparsity) {
                dense_matrix[i * num_cols + j] = 0.0f;
            } else {
                dense_matrix[i * num_cols + j] = 0.1f + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 9.9f));
            }
        }
    }
}

void generate_random_vector(uint64_t size, float*& vector) {
    vector = new float[size];
    std::srand(static_cast<unsigned>(std::time(nullptr)));  // Seed RNG

    for (uint64_t i = 0; i < size; i++) {
        // Generate values in range [0.1, 10.0]
        vector[i] = 0.1f + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 9.9f));
    }
}


float benchmark_csr_spmv(const uint64_t* row_offsets, const uint64_t* col_indices,
                        const float* values, const float* vector, float* output,
                        uint64_t num_rows, uint64_t nnz, size_t iterations) {
    // Device pointers
    uint64_t *d_row_offsets, *d_col_indices;
    float *d_values, *d_vector, *d_output;

    // Allocate device memory
    cudaMalloc(&d_row_offsets, (num_rows + 1) * sizeof(uint64_t));
    cudaMalloc(&d_col_indices, nnz * sizeof(uint64_t));
    cudaMalloc(&d_values, nnz * sizeof(float));
    cudaMalloc(&d_vector, num_rows * sizeof(float)); // assuming square matrix
    cudaMalloc(&d_output, num_rows * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_row_offsets, row_offsets, (num_rows + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_indices, col_indices, nnz * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values, nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector, vector, num_rows * sizeof(float), cudaMemcpyHostToDevice);

    // Launch config
    dim3 blockDim(256);
    dim3 gridDim((num_rows + blockDim.x - 1) / blockDim.x);

    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_time = 0.0f;

    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_output, 0, num_rows * sizeof(float));  // Reset output before each run

        cudaEventRecord(start);

        graphblas_gpu::kernels::spmv_csr<float>(
            d_row_offsets, d_col_indices, d_values, d_vector, d_output, num_rows);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        total_time += ms;
    }

    float avg_time = total_time / iterations;
    printf("CSR SpMV average execution time over %d iterations: %.3f ms\n", iterations, avg_time);

    // Copy result back to host (once, after final iteration)
    cudaMemcpy(output, d_output, num_rows * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_row_offsets);
    cudaFree(d_col_indices);
    cudaFree(d_values);
    cudaFree(d_vector);
    cudaFree(d_output);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return avg_time;
}

float benchmark_ell_spmv(const size_t* ell_col_indices, const float* ell_values,
                        const float* vector, float* output, uint64_t num_rows,
                        size_t max_nnz_per_row, size_t iterations) {
    // Device pointers
    size_t *d_ell_col_indices;
    float *d_ell_values, *d_vector, *d_output;

    // Allocate device memory
    cudaMalloc(&d_ell_col_indices, num_rows * max_nnz_per_row * sizeof(size_t));
    cudaMalloc(&d_ell_values, num_rows * max_nnz_per_row * sizeof(float));
    cudaMalloc(&d_vector, num_rows * sizeof(float)); // assuming square matrix
    cudaMalloc(&d_output, num_rows * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_ell_col_indices, ell_col_indices, num_rows * max_nnz_per_row * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ell_values, ell_values, num_rows * max_nnz_per_row * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector, vector, num_rows * sizeof(float), cudaMemcpyHostToDevice);

    // Launch config
    dim3 blockDim(256);
    dim3 gridDim((num_rows + blockDim.x - 1) / blockDim.x);

    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_time = 0.0f;

    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_output, 0, num_rows * sizeof(float));  // Reset output before each run

        cudaEventRecord(start);

        graphblas_gpu::kernels::spmv_ell<float>(
            d_ell_col_indices, d_ell_values, d_vector, d_output,
            num_rows, max_nnz_per_row);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        total_time += ms;
    }

    float avg_time = total_time / iterations;
    printf("ELL SpMV average execution time over %d iterations: %.3f ms\n", iterations, avg_time);

    // Copy result back to host
    cudaMemcpy(output, d_output, num_rows * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_ell_col_indices);
    cudaFree(d_ell_values);
    cudaFree(d_vector);
    cudaFree(d_output);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return avg_time;
}

float benchmark_sellc_spmv(const size_t* sell_col_indices, const float* sell_values,
                          const size_t* slice_ptrs, const size_t* slice_lengths,
                          const float* vector, float* output,
                          size_t num_rows, size_t c, size_t num_slices, size_t iterations) {
    // Device pointers
    size_t *d_sell_col_indices, *d_slice_ptrs, *d_slice_lengths;
    float *d_sell_values, *d_vector, *d_output;

    // Determine total number of elements in sell_values
    int total_vals = 0;
    for (int i = 0; i < num_slices; i++)
        total_vals += slice_lengths[i] * c;

    // Allocate device memory
    cudaMalloc(&d_sell_col_indices, total_vals * sizeof(size_t));
    cudaMalloc(&d_sell_values, total_vals * sizeof(float));
    cudaMalloc(&d_slice_ptrs, num_slices * sizeof(size_t));
    cudaMalloc(&d_slice_lengths, num_slices * sizeof(size_t));
    cudaMalloc(&d_vector, num_rows * sizeof(float));
    cudaMalloc(&d_output, num_rows * sizeof(float));

    // Copy to device
    cudaMemcpy(d_sell_col_indices, sell_col_indices, total_vals * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sell_values, sell_values, total_vals * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_slice_ptrs, slice_ptrs, num_slices * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_slice_lengths, slice_lengths, num_slices * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector, vector, num_rows * sizeof(float), cudaMemcpyHostToDevice);

    // CUDA launch config
    dim3 blockDim(256);
    dim3 gridDim((num_rows + blockDim.x - 1) / blockDim.x);

    // Timing setup
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_time = 0.0f;

    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_output, 0, num_rows * sizeof(float));

        cudaEventRecord(start);

        graphblas_gpu::kernels::spmv_sell_c<float>(
            d_slice_ptrs, d_sell_col_indices, d_sell_values,  num_rows, c,
            d_vector, d_output);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        total_time += ms;
    }

    float avg_time = total_time / iterations;
    printf("SELL-C SpMV average execution time over %d iterations: %.3f ms\n", iterations, avg_time);

    // Copy result back to host
    cudaMemcpy(output, d_output, num_rows * sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_sell_col_indices);
    cudaFree(d_sell_values);
    cudaFree(d_slice_ptrs);
    cudaFree(d_slice_lengths);
    cudaFree(d_vector);
    cudaFree(d_output);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return avg_time;
}





    // Copy result back to host (once, after
void benchmark_kernels_spmv(float sparsity, size_t num_rows, size_t num_cols, size_t iterations) {
    graphblas_gpu::GraphClassifier classifier;
    // Generate random sparse matrix in dense form
    float* dense_matrix = nullptr;
    generate_random_sparse_matrix(num_rows, num_cols, sparsity, dense_matrix);

    // Generate input vector
    float* vector = nullptr;
    generate_random_vector(num_cols, vector);

    // Allocate output buffer
    float* output = new float[num_rows];

    // ----- Convert to CSR -----
    uint64_t* csr_row_offsets = nullptr;
    uint64_t* csr_col_indices = nullptr;
    float* csr_values = nullptr;
    uint64_t csr_nnz = 0;
    classifier.dense_to_csr<float>(dense_matrix, num_rows, num_cols, csr_row_offsets, csr_col_indices, csr_values);

    // Benchmark CSR
    float csr_time = benchmark_csr_spmv(csr_row_offsets, csr_col_indices, csr_values,
                                        vector, output, num_rows, csr_nnz, iterations);
    printf("[CSR] Avg Time: %.3f ms\n", csr_time);

    // ----- Convert to ELL -----
    size_t* ell_col_indices = nullptr;
    float* ell_values = nullptr;
    size_t ell_max_cols = 0;
    classifier.dense_to_ell<float>(dense_matrix, num_rows, num_cols, ell_col_indices, ell_values, ell_max_cols);

    // Benchmark ELL
    float ell_time = benchmark_ell_spmv(ell_col_indices, ell_values, vector, output,
                                        num_rows, ell_max_cols, iterations);
    printf("[ELL] Avg Time: %.3f ms\n", ell_time);

    // ----- Convert to SELL-C -----
    size_t* sell_col_indices = nullptr;
    float* sell_values = nullptr;
    size_t* slice_ptrs = nullptr;
    size_t* slice_lengths = nullptr;
    size_t slice_height = 4;  // Or any desired C value

    // Number of slices: num_rows / slice_height
    size_t num_slices = (num_rows + slice_height - 1) / slice_height;

    // Convert dense matrix to SELL-C format
    classifier.dense_to_sell_c<float>(dense_matrix, num_rows, num_cols, slice_height,
                    sell_col_indices, sell_values, slice_ptrs, slice_lengths);

    // Benchmark SELL-C
    float sell_time = benchmark_sellc_spmv(sell_col_indices, sell_values,
                                           slice_ptrs, slice_lengths, vector, output,
                                           num_rows, slice_height, num_slices, iterations);
    printf("[SELL-C] Avg Time: %.3f ms\n", sell_time);

    // Cleanup
    delete[] dense_matrix;
    delete[] vector;
    delete[] output;

    delete[] csr_row_offsets;
    delete[] csr_col_indices;
    delete[] csr_values;

    delete[] ell_col_indices;
    delete[] ell_values;

    delete[] sell_col_indices;
    delete[] sell_values;
    delete[] slice_ptrs;
    delete[] slice_lengths;
}


int main() {
    // Parameters
    float sparsity = 0.1f; // Adjust as needed
    size_t num_rows = 1024;   // Adjust as needed
    size_t num_cols = 1024;   // Adjust as needed
    size_t iterations = 5;    // Number of iterations for benchmarking

    benchmark_kernels_spmv(sparsity, num_rows, num_cols, iterations);

    return 0;
}


