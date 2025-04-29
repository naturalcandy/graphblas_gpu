#include <graphblas_gpu/vector.hpp>
#include <graphblas_gpu/graph.hpp>
#include <graphblas_gpu/operation.hpp>
#include <graphblas_gpu/op_compiler.hpp>
#include <graphblas_gpu_utils/matrix_generators.hpp>

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <numeric>

__global__ void spmv_csr_kernel(const size_t* row_offsets, 
                                const int* col_indices, 
                                const float* values, 
                                const float* x, 
                                float* y, 
                                size_t num_rows)
{
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float sum = 0;
        for (size_t j = row_offsets[row]; j < row_offsets[row + 1]; ++j) {
            sum += values[j] * x[col_indices[j]];
        }
        y[row] = sum;
    }
}

float run_baseline(const std::vector<size_t>& row_offsets, const std::vector<int>& col_indices,
                   const std::vector<float>& values, const std::vector<float>& x, std::vector<float>& y, int iterations)
{
    size_t num_rows = row_offsets.size() - 1;
    size_t* d_row_offsets;
    int* d_col_indices;
    float* d_values;
    float* d_x;
    float* d_y;

    cudaMalloc(&d_row_offsets, row_offsets.size() * sizeof(size_t));
    cudaMalloc(&d_col_indices, col_indices.size() * sizeof(int));
    cudaMalloc(&d_values, values.size() * sizeof(float));
    cudaMalloc(&d_x, x.size() * sizeof(float));
    cudaMalloc(&d_y, num_rows * sizeof(float));

    cudaMemcpy(d_row_offsets, row_offsets.data(), row_offsets.size() * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_indices, col_indices.data(), col_indices.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (num_rows + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        spmv_csr_kernel<<<gridSize, blockSize>>>(d_row_offsets, d_col_indices, d_values, d_x, d_y, num_rows);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(y.data(), d_y, num_rows * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_row_offsets);
    cudaFree(d_col_indices);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

float run_graphblas(const std::vector<size_t>& row_offsets, const std::vector<int>& col_indices,
                    const std::vector<float>& values, const std::vector<float>& x, std::vector<float>& y, int iterations)
{
    using namespace graphblas_gpu;

    OpSequence::getInstance().clear();

    SparseMatrix<float> matrix(row_offsets.size() - 1, x.size(), row_offsets, col_indices, values);
    Vector<float> vec_x(x.size(), x);

    Vector<float> result = Operations<float>::spmv(matrix, vec_x, SemiringType::Arithmetic);

    OpCompiler& compiler = OpCompiler::getInstance();
    compiler.compile();
    compiler.copyHostToDevice(matrix);
    compiler.copyHostToDevice(vec_x);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    compiler.execute(iterations, &start, &stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    compiler.copyDeviceToHost(y.data(), result.bufferId(), result.bytes());

    return ms;
}

int main() {
    using T = float;

    std::vector<std::pair<std::string, std::pair<size_t, size_t>>> matrices = {
        {"Small", {512, 512}},
        {"Medium", {1024, 1024}},
        {"Large", {2048, 2048}},
        {"X-Large", {4096, 4096}}
    };

    std::vector<float> sparsities = {0.01f, 0.05f, 0.1f, 0.2f};
    std::vector<int> iterations_list = {100, 1000, 10000, 50000, 100000, 250000};

    std::ofstream file("benchmark_results.csv");
    file << "Matrix,Rows,Cols,Sparsity,Iterations,Baseline (ms),GraphBLAS (ms),Speedup\n";

    for (const auto& matrix_info : matrices) {
        for (float sparsity : sparsities) {
            for (int iterations : iterations_list) {

                size_t num_rows = matrix_info.second.first;
                size_t num_cols = matrix_info.second.second;

                std::cout << "\nTesting " << matrix_info.first
                          << " (" << num_rows << "x" << num_cols
                          << ", sparsity=" << sparsity << ", iterations=" << iterations << ")\n";

                std::vector<size_t> row_offsets;
                std::vector<int> col_indices;
                std::vector<float> values;

                graphblas_gpu::utils::generate_uniform_random_csr(num_rows, num_cols, sparsity,
                    row_offsets, col_indices, values);

                std::vector<float> x(num_cols, 1.0f);
                std::vector<float> y_baseline(num_rows, 0.0f);
                std::vector<float> y_graphblas(num_rows, 0.0f);

                const int num_trials = 3;

                std::vector<float> baseline_times, graphblas_times;

                for (int t = 0; t < num_trials; t++) {
                    baseline_times.push_back(run_baseline(row_offsets, col_indices, values, x, y_baseline, iterations));
                    graphblas_times.push_back(run_graphblas(row_offsets, col_indices, values, x, y_graphblas, iterations));
                }

                float avg_baseline = std::accumulate(baseline_times.begin(), baseline_times.end(), 0.0f) / num_trials;
                float avg_graphblas = std::accumulate(graphblas_times.begin(), graphblas_times.end(), 0.0f) / num_trials;
                float speedup = avg_baseline / avg_graphblas;

                file << matrix_info.first << ","
                     << num_rows << ","
                     << num_cols << ","
                     << sparsity << ","
                     << iterations << ","
                     << avg_baseline << ","
                     << avg_graphblas << ","
                     << speedup << "\n";

                std::cout << "Avg Baseline (ms): " << avg_baseline << "\n";
                std::cout << "Avg GraphBLAS (ms): " << avg_graphblas << "\n";
                std::cout << "Avg Speedup: " << speedup << "x\n";
            }
        }
    }

    file.close();
    return 0;
}
