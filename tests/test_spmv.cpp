#include <graphblas_gpu/vector.hpp>
#include <graphblas_gpu/operation.hpp>
#include <graphblas_gpu/graph.hpp>
#include <graphblas_gpu/op_compiler.hpp>
#include <graphblas_gpu/graph_classifier.hpp>
#include <graphblas_gpu_utils/matrix_generators.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <algorithm>
#include <map>
#include <set>
#include <functional>

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


bool verify(const std::vector<float>& result, 
            const std::vector<float>& reference) {
    for (size_t i = 0; i < result.size(); i++) {
        if (result[i] != reference[i]) {
            return false;
        }
    }
    return true;
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
        bool passed = verify(result, reference);
        
        if (passed) {
            std::cout << test_info << " with " << format << ": PASSED" << std::endl;
        } else {
            std::cout << test_info << " with " << format << ": FAILED" << std::endl;
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
                                          
    generators["Uniform"] = utils::generate_uniform_random_csr;
    generators["PowerLaw"] = utils::generate_powerlaw_csr;
    generators["Diagonal"] = utils::generate_diagonal_heavy_csr;
    
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