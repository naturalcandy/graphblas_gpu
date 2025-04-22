#include <graphblas_gpu/vector.hpp>
#include <graphblas_gpu/operation.hpp>
#include <graphblas_gpu/graph.hpp>
#include <graphblas_gpu/op_compiler.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstring>
#include <string>
#include <algorithm>
#include <map>
#include <set>
#include <functional>

// ===== Matrix Generation Functions =====

// Generate a uniform random sparse matrix
void generate_uniform_random_csr(size_t num_rows, size_t num_cols, float sparsity, 
                              std::vector<size_t>& row_offsets,
                              std::vector<size_t>& col_indices,
                              std::vector<float>& values,
                              unsigned int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> val_dist(0.1f, 10.0f);
    std::uniform_int_distribution<size_t> col_dist(0, num_cols - 1);
    std::bernoulli_distribution sparsity_dist(sparsity);
    
    row_offsets.clear();
    col_indices.clear();
    values.clear();
    
    size_t estimated_nnz = num_rows * num_cols * sparsity;
    col_indices.reserve(estimated_nnz);
    values.reserve(estimated_nnz);
    row_offsets.resize(num_rows + 1);
    size_t nnz = 0;
    row_offsets[0] = 0;
    
    for (size_t row = 0; row < num_rows; row++) {
        for (size_t col = 0; col < num_cols; col++) {
            if (sparsity_dist(gen)) {
                col_indices.push_back(col);
                values.push_back(val_dist(gen));
                nnz++;
            }
        }
        row_offsets[row + 1] = nnz;
    }
    
    // At least one element per row...
    for (size_t row = 0; row < num_rows; row++) {
        if (row_offsets[row] == row_offsets[row + 1]) {
            size_t col = col_dist(gen);
            
            col_indices.insert(col_indices.begin() + row_offsets[row], col);
            values.insert(values.begin() + row_offsets[row], val_dist(gen));
            // Update all subsequent offsets
            for (size_t i = row + 1; i <= num_rows; i++) {
                row_offsets[i]++;
            }
        }
    }
}

// Generate a scale-free sparse matrix
void generate_powerlaw_csr(size_t num_rows, size_t num_cols, float sparsity,
                          std::vector<size_t>& row_offsets,
                          std::vector<size_t>& col_indices,
                          std::vector<float>& values,
                          unsigned int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> val_dist(0.1f, 10.0f);
    row_offsets.clear();
    col_indices.clear();
    values.clear();
    size_t estimated_nnz = num_rows * num_cols * sparsity;
    col_indices.reserve(estimated_nnz);
    values.reserve(estimated_nnz);
    row_offsets.resize(num_rows + 1);
    

    std::vector<float> column_weights(num_cols);
    for (size_t i = 0; i < num_cols; ++i) {
        column_weights[i] = 1.0f / std::pow(float(i + 1), 2.1f);
    }
    
    std::discrete_distribution<size_t> col_dist(column_weights.begin(), column_weights.end());
    
    size_t total_expected_edges = num_rows * num_cols * sparsity;
    size_t edges_per_row = total_expected_edges / num_rows;
    
    size_t nnz = 0;
    row_offsets[0] = 0;
    
    for (size_t row = 0; row < num_rows; row++) {
        std::set<size_t> row_cols; 
        
        while (row_cols.size() < edges_per_row && row_cols.size() < num_cols) {
            size_t col = col_dist(gen);
            row_cols.insert(col);
        }
        
        if (row_cols.empty() && num_cols > 0) {
            row_cols.insert(gen() % num_cols);
        }
         for (size_t col : row_cols) {
            col_indices.push_back(col);
            values.push_back(val_dist(gen));
            nnz++;
        }
        
        row_offsets[row + 1] = nnz;
    }
}

// Generate a diagonal-heavy sparse matrix 
void generate_diagonal_heavy_csr(size_t num_rows, size_t num_cols, float sparsity,
                              std::vector<size_t>& row_offsets,
                              std::vector<size_t>& col_indices,
                              std::vector<float>& values,
                              unsigned int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> val_dist(0.1f, 10.0f);
    std::bernoulli_distribution diagonal_dist(0.9); 
    std::bernoulli_distribution band_dist(0.4);    
    std::bernoulli_distribution random_dist(0.05);  
    
    row_offsets.clear();
    col_indices.clear();
    values.clear();
    
    size_t bandwidth = 5;
    size_t estimated_nnz = num_rows * (2 * bandwidth + 1) * sparsity;
    col_indices.reserve(estimated_nnz);
    values.reserve(estimated_nnz);
    row_offsets.resize(num_rows + 1); 
    size_t nnz = 0;
    row_offsets[0] = 0;
    
    for (size_t row = 0; row < num_rows; row++) {
        if (row < num_cols && diagonal_dist(gen)) {
            col_indices.push_back(row);
            values.push_back(val_dist(gen) * 10.0f);
            nnz++;
        }     
        for (size_t k = 1; k <= bandwidth; k++) {
            if (row >= k && row - k < num_cols && band_dist(gen)) {
                col_indices.push_back(row - k);
                values.push_back(val_dist(gen));
                nnz++;
            }
            
            if (row + k < num_cols && band_dist(gen)) {
                col_indices.push_back(row + k);
                values.push_back(val_dist(gen));
                nnz++;
            }
        }
        
        for (size_t col = 0; col < num_cols; col++) {
            if (std::abs(static_cast<long>(row) - static_cast<long>(col)) > bandwidth && random_dist(gen)) {
                col_indices.push_back(col);
                values.push_back(val_dist(gen) * 0.5f);
                nnz++;
            }
        }
        
        size_t row_start = row_offsets[row];
        size_t row_size = nnz - row_start;
        if (row_size > 0) {
            std::vector<std::pair<size_t, float>> row_data;
            for (size_t i = 0; i < row_size; i++) {
                row_data.push_back({col_indices[row_start + i], values[row_start + i]});
            }
            
            std::sort(row_data.begin(), row_data.end());
            
            for (size_t i = 0; i < row_size; i++) {
                col_indices[row_start + i] = row_data[i].first;
                values[row_start + i] = row_data[i].second;
            }
        }
        if (row_offsets[row] == nnz && num_cols > 0) {
            size_t col = std::min(row, num_cols - 1);
            col_indices.push_back(col);
            values.push_back(val_dist(gen));
            nnz++;
        }
        
        row_offsets[row + 1] = nnz;
    }
}

// Reference sequential implementation 
void spmv_ref(const std::vector<size_t>& row_offsets,
              const std::vector<size_t>& col_indices,
              const std::vector<float>& values,
              const std::vector<float>& x,
              std::vector<float>& y) {
    for (size_t row = 0; row < row_offsets.size() - 1; row++) {
        float sum = 0.0f;
        size_t row_start = row_offsets[row];
        size_t row_end = row_offsets[row + 1];
        
        for (size_t i = row_start; i < row_end; i++) {
            sum += values[i] * x[col_indices[i]];
        }
        
        y[row] = sum;
    }
}

std::string matrix_name(const std::string& size, const std::string& pattern, float sparsity) {
    char sparsity_str[32];
    snprintf(sparsity_str, sizeof(sparsity_str), "%.1f%%", (1.0f - sparsity) * 100.0f);
    return size + " " + pattern + " " + sparsity_str;
}

void csrToDevice(graphblas_gpu::OpCompiler& compiler, 
                          size_t buffer_id,
                          const std::vector<size_t>& row_offsets,
                          const std::vector<size_t>& col_indices,
                          const std::vector<float>& values) {
    size_t row_offsets_size = row_offsets.size() * sizeof(size_t);
    size_t col_indices_size = col_indices.size() * sizeof(size_t);
    size_t values_size = values.size() * sizeof(float);
    
    size_t total_size = row_offsets_size + col_indices_size + values_size;
    std::vector<char> buffer(total_size);
    
    char* ptr = buffer.data();
    std::memcpy(ptr, row_offsets.data(), row_offsets_size);
    ptr += row_offsets_size;
    std::memcpy(ptr, col_indices.data(), col_indices_size);
    ptr += col_indices_size;
    std::memcpy(ptr, values.data(), values_size);
    
    compiler.copyHostToDevice(buffer.data(), buffer_id, total_size);
}

struct TestMatrix {
    size_t rows;
    size_t cols;
    float sparsity;
    std::string pattern_name;
    std::string size_name;
    std::function<void(size_t, size_t, float, 
                    std::vector<size_t>&, 
                    std::vector<size_t>&, 
                    std::vector<float>&, 
                    unsigned int)> generator;
    unsigned int seed;
};

int main() {
    using namespace graphblas_gpu;
    
    std::map<std::string, std::function<void(size_t, size_t, float, 
                                          std::vector<size_t>&, 
                                          std::vector<size_t>&, 
                                          std::vector<float>&, 
                                          unsigned int)>> generators;
                                          
    generators["Uniform"] = generate_uniform_random_csr;
    generators["PowerLaw"] = generate_powerlaw_csr;
    generators["Diagonal"] = generate_diagonal_heavy_csr;
    std::vector<std::pair<std::string, std::pair<size_t, size_t>>> sizes = {
        {"Tiny", {16, 16}},
        {"Small", {64, 64}},
        {"Medium", {256, 256}},
        {"Large", {1024, 1024}},
        {"Tall", {1024, 256}},
        {"Wide", {256, 1024}}
    };
    
    std::vector<float> sparsities = {0.05f, 0.1f, 0.2f, 0.3f};  
        std::vector<TestMatrix> test_matrices;
    unsigned int base_seed = 42;
    
    for (const auto& size_pair : sizes) {
        const auto& size_name = size_pair.first;
        const auto& dims = size_pair.second;
        
        for (const auto& gen_pair : generators) {
            const auto& pattern_name = gen_pair.first;
            const auto& generator = gen_pair.second;
            
            for (float sparsity : sparsities) {
                if ((size_name == "Large" || size_name == "Tall" || size_name == "Wide") && 
                    sparsity > 0.1f) {
                    continue;
                }
                
                test_matrices.push_back({
                    dims.first,       
                    dims.second,       
                    sparsity,          
                    pattern_name,     
                    size_name,         
                    generator,         
                    base_seed++    
                });
            }
        }
    }
    
    size_t test_count = 0;
    size_t passed_count = 0;
    
    for (const auto& test : test_matrices) {
        test_count++;
        OpSequence::getInstance().clear();
        
        std::vector<size_t> row_offsets;
        std::vector<size_t> col_indices;
        std::vector<float> values;
        
        test.generator(test.rows, test.cols, test.sparsity, 
                     row_offsets, col_indices, values, test.seed);
        
        std::vector<float> h_x(test.cols, 1.0f);
        std::vector<float> h_result(test.rows, 0.0f);
        
        try {
            // Construct GraphBLAS objects
            SparseMatrix<float> matrix(test.rows, test.cols, row_offsets, col_indices, values);
            Vector<float> vec_x(test.cols, h_x);
            
            // Schedule the SpMV operation
            Vector<float> result = Operations<float>::spmv(matrix, vec_x, SemiringType::Arithmetic);
            
            // Compile operations
            OpCompiler& compiler = OpCompiler::getInstance();
            compiler.compile();
            
            csrToDevice(compiler, matrix.bufferId(), row_offsets, col_indices, values);
            
            // Copy vector to device
            compiler.copyHostToDevice(h_x.data(), vec_x.bufferId(), vec_x.bytes());
            
            // Execute our schedueled kernel
            compiler.execute(1);
            
            // Copy result back to host
            compiler.copyDeviceToHost(h_result.data(), result.bufferId(), result.bytes());
            
            // Compare with refsol
            std::vector<float> h_ref(test.rows, 0.0f);
            spmv_ref(row_offsets, col_indices, values, h_x, h_ref);
            
            // Verify
            bool all_correct = true;
            size_t error_index = 0;
            float max_error = 0.0f;
            
            for (size_t i = 0; i < test.rows; i++) {
                float error = std::abs(h_result[i] - h_ref[i]);
                if (error > max_error) {
                    max_error = error;
                    error_index = i;
                }
                
                if (error > 1e-5) {
                    all_correct = false;
                }
            }
            
            float actual_sparsity = 1.0f - (static_cast<float>(values.size()) / (test.rows * test.cols));
            
            std::string status = all_correct ? "PASSED" : "FAILED";
            std::string dim_str = std::to_string(test.rows) + "x" + std::to_string(test.cols);
            std::string sparsity_str = std::to_string(static_cast<int>(actual_sparsity * 100)) + "%";
            std::cout << test.size_name << " " << dim_str << " " 
                    << test.pattern_name << " (NNZ: " << values.size() 
                    << ", Sparsity: " << sparsity_str << "): " << status;

            if (!all_correct) {
                std::cout << " - Error at row " << error_index 
                        << ": " << h_result[error_index] 
                        << " vs " << h_ref[error_index];
            }
            std::cout << std::endl;

            if (all_correct) {
                passed_count++;
            }
            
        } catch (const std::exception& e) {
            std::string dim_str = std::to_string(test.rows) + "x" + std::to_string(test.cols);
            std::string sparsity_str = std::to_string(static_cast<int>((1.0f - test.sparsity) * 100)) + "%";

            std::cout << test.size_name << " " << dim_str << " " 
                    << test.pattern_name << " (NNZ: " << values.size() 
                    << ", Sparsity: " << sparsity_str << "): ERROR - " 
                    << e.what() << std::endl;
        }
    }
    
    std::cout << "\n-------------------------------------------\n";
    std::cout << "SUMMARY: " << passed_count << " of " << test_count << " tests passed\n";
    std::cout << "-------------------------------------------\n";
        
    return (passed_count == test_count) ? 0 : 1;
}