#include <graphblas_gpu/vector.hpp>
#include <graphblas_gpu/operation.hpp>
#include <graphblas_gpu/graph.hpp>
#include <graphblas_gpu/op_compiler.hpp>
#include <graphblas_gpu/graph_classifier.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <string>
#include <algorithm>
#include <map>
#include <set>
#include <functional>


// Generate a uniform random sparse matrix
void generate_uniform_random_csr(size_t num_rows, size_t num_cols, float sparsity, 
                              std::vector<size_t>& row_offsets,
                              std::vector<int>& col_indices,
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
                          std::vector<int>& col_indices,
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
                              std::vector<int>& col_indices,
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

// Sequential implementation
void spmv_ref(const std::vector<size_t>& row_offsets,
              const std::vector<int>& col_indices,
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


bool verify_results(const std::vector<float>& result, 
                    const std::vector<float>& reference,
                    size_t& error_index,
                    float& max_error,
                    float tolerance = 1e-5) {
    bool all_correct = true;
    max_error = 0.0f;
    
    for (size_t i = 0; i < result.size(); i++) {
        float error = std::abs(result[i] - reference[i]);
        if (error > max_error) {
            max_error = error;
            error_index = i;
        }
        
        if (error > tolerance) {
            all_correct = false;
        }
    }
    
    return all_correct;
}

struct TestMatrix {
    size_t rows;
    size_t cols;
    float sparsity;
    std::string pattern_name;
    std::string size_name;
    std::function<void(size_t, size_t, float, 
                    std::vector<size_t>&, 
                    std::vector<int>&, 
                    std::vector<float>&, 
                    unsigned int)> generator;
    unsigned int seed;
};

bool test_format(const std::string& format,
                const graphblas_gpu::SparseMatrix<float>& matrix,
                const std::vector<float>& x,
                const std::vector<float>& reference,
                const std::string& test_info) {
    using namespace graphblas_gpu;
    
    Vector<float> vec_x(x.size(), x);
    std::vector<float> result(reference.size(), 0.0f);
    
    try {
        // SpMV 
        Vector<float> gpu_result = Operations<float>::spmv(matrix, vec_x, SemiringType::Arithmetic);
        
        // Compile and execute
        OpCompiler& compiler = OpCompiler::getInstance();
        compiler.compile();
        
        // Copy host data to device
        compiler.copyHostToDevice(matrix);
        compiler.copyHostToDevice(vec_x);
        
        compiler.execute(1);
        
        // Get result
        compiler.copyDeviceToHost(result.data(), gpu_result.bufferId(), gpu_result.bytes());
        
        // Verify results
        size_t error_index = 0;
        float max_error = 0.0f;
        bool passed = verify_results(result, reference, error_index, max_error);
        
        if (passed) {
            std::cout << test_info << " with " << format << ": PASSED" << std::endl;
        } else {
            std::cout << test_info << " with " << format << ": FAILED";
            std::cout << " - Error at row " << error_index << ": " 
                     << result[error_index] << " vs " << reference[error_index] 
                     << " (error: " << max_error << ")" << std::endl;
        }
        
        return passed;
    }
    catch (const std::exception& e) {
        std::cerr << test_info << " with " << format 
                 << ": ERROR - " << e.what() << std::endl;
        return false;
    }
}

int main() {
    using namespace graphblas_gpu;
    
    std::map<std::string, std::function<void(size_t, size_t, float, 
                                          std::vector<size_t>&, 
                                          std::vector<int>&, 
                                          std::vector<float>&, 
                                          unsigned int)>> generators;
                                          
    generators["Uniform"] = generate_uniform_random_csr;
    generators["PowerLaw"] = generate_powerlaw_csr;
    generators["Diagonal"] = generate_diagonal_heavy_csr;
    
    // Matrix dimensions to test
    std::vector<std::pair<std::string, std::pair<size_t, size_t>>> sizes = {
        {"Tiny", {16, 16}},
        {"Small", {64, 64}},
        {"Medium", {256, 256}},
        {"Large", {1024, 1024}},
        {"Tall", {1024, 256}},
        {"Wide", {256, 1024}}
    };
    
    // Sparsity levels
    std::vector<float> sparsities = {0.05f, 0.1f, 0.2f, 0.3f};
    
    // Build test matrix list
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
    
    // Statistics
    size_t total_tests = 0;
    size_t csr_passed = 0, csr_total = 0;
    size_t ell_passed = 0, ell_total = 0;
    size_t sellc_passed = 0, sellc_total = 0;
    
    // Run tests
    for (const auto& test : test_matrices) {
        OpSequence::getInstance().clear();
        
        // Generate the test matrix in CSR format
        std::vector<size_t> row_offsets;
        std::vector<int> col_indices;
        std::vector<float> values;
        
        test.generator(test.rows, test.cols, test.sparsity, 
                      row_offsets, col_indices, values, test.seed);
        
        std::vector<float> x(test.cols, 1.0f);
        
        // Reference result
        std::vector<float> reference(test.rows, 0.0f);
        spmv_ref(row_offsets, col_indices, values, x, reference);
        
        float actual_sparsity = 1.0f - (static_cast<float>(values.size()) / (test.rows * test.cols));
        char sparsity_str[32];
        snprintf(sparsity_str, sizeof(sparsity_str), "%.1f%%", actual_sparsity * 100);
        
        std::string test_info = test.size_name + " " + 
                               std::to_string(test.rows) + "x" + std::to_string(test.cols) + 
                               " " + test.pattern_name + " (NNZ: " + std::to_string(values.size()) + 
                               ", Sparsity: " + sparsity_str + ")";
        
        std::cout << "\nTesting " << test_info << std::endl;
        total_tests++;
        
        // Test CSR format
        {
            csr_total++;
            SparseMatrix<float> matrix(test.rows, test.cols, row_offsets, col_indices, values);
            if (test_format("CSR", matrix, x, reference, test_info)) {
                csr_passed++;
            }
        }
        
        // Test ELL format
        {
            ell_total++;
            std::vector<int> ell_col_indices;
            std::vector<float> ell_values;
            size_t max_nnz_per_row;
            
            GraphClassifier::csr_to_ell(row_offsets, col_indices, values, test.rows, 
                                      ell_col_indices, ell_values, max_nnz_per_row);
            
            SparseMatrix<float> matrix(test.rows, test.cols, max_nnz_per_row, 
                                      ell_col_indices, ell_values);
            
            if (test_format("ELL", matrix, x, reference, test_info)) {
                ell_passed++;
            }
        }
        
        // Test SELLC format
        {
            sellc_total++;
            std::vector<size_t> slice_ptrs;
            std::vector<size_t> slice_lengths;
            std::vector<int> sell_col_indices;
            std::vector<float> sell_values;
            const size_t slice_size = 2;
            
            GraphClassifier::csr_to_sellc(row_offsets, col_indices, values, test.rows, slice_size,
                                         slice_ptrs, slice_lengths, sell_col_indices, sell_values);
            
            SparseMatrix<float> matrix(test.rows, test.cols, slice_size, slice_ptrs, slice_lengths, 
                                      sell_col_indices, sell_values);
            
            if (test_format("SELLC", matrix, x, reference, test_info)) {
                sellc_passed++;
            }
        }
    }
    
    // Print summary
    std::cout << "\n===== Test Summary =====" << std::endl;
    std::cout << "CSR:    " << csr_passed << "/" << csr_total 
             << " passed (" << (100.0f * csr_passed / csr_total) << "%)" << std::endl;
    std::cout << "ELL:    " << ell_passed << "/" << ell_total 
             << " passed (" << (100.0f * ell_passed / ell_total) << "%)" << std::endl;
    std::cout << "SELL-C: " << sellc_passed << "/" << sellc_total 
             << " passed (" << (100.0f * sellc_passed / sellc_total) << "%)" << std::endl;
    
    size_t total_passed = csr_passed + ell_passed + sellc_passed;
    size_t total = csr_total + ell_total + sellc_total;
    
    std::cout << "\nOverall: " << total_passed << "/" << total 
             << " passed (" << (100.0f * total_passed / total) << "%)" << std::endl;
    
    return (total_passed == total) ? 0 : 1;
}