#include <graphblas_gpu/vector.hpp>
#include <graphblas_gpu/operation.hpp>
#include <graphblas_gpu/graph.hpp>
#include <graphblas_gpu/op_compiler.hpp>
#include <graphblas_gpu/semiring.hpp>
#include <iostream>
#include <vector>
#include <random>

// Sequential reference implementation of OR-AND semiring
void spmv_ref_logical(const std::vector<size_t>& row_offsets,
                      const std::vector<int>& col_indices,
                      const std::vector<float>& values,
                      const std::vector<float>& x,
                      std::vector<float>& y) {
    for (size_t row = 0; row < row_offsets.size() - 1; ++row) {
        bool result = false;
        for (size_t i = row_offsets[row]; i < row_offsets[row + 1]; ++i) {
            bool val = (values[i] != 0.0f) && (x[col_indices[i]] != 0.0f);
            result = result || val;
            if (result) break;  
        }
        y[row] = result ? 1.0f : 0.0f;
    }
}

bool verify(const std::vector<float>& result, 
                    const std::vector<float>& reference) {
    if (result.size() != reference.size()) {
        std::cerr << "Size mismatch: result[" << result.size() 
                 << "] vs reference[" << reference.size() << "]" << std::endl;
        return false;
    }
    
    bool correct = true;
    
    for (size_t i = 0; i < result.size(); i++) {
        if (result[i] != reference[i]) {
            correct = false;
        }
    }
    if (!correct) {
        std::cout << "Error" << std::endl;
    }
    
    return correct;
}

int main() {
    using namespace graphblas_gpu;
    
    // 0 1 0
    // 1 0 1
    // 0 1 1
    const size_t rows = 3;
    const size_t cols = 3;
    
    std::vector<size_t> row_offsets = {0, 1, 3, 5};
    std::vector<int> col_indices = {1, 0, 2, 1, 2};
    std::vector<float> values = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    
    std::vector<std::vector<float>> test_vectors = {
        {1.0f, 1.0f, 1.0f},   
        {0.0f, 1.0f, 0.0f},   
        {1.0f, 0.0f, 1.0f}   
    };
    
    std::vector<std::string> vector_descriptions = {
        "All ones",
        "Only middle column set",
        "First and third columns set"
    };
    
    for (size_t test_idx = 0; test_idx < test_vectors.size(); test_idx++) {
        OpSequence::getInstance().clear();
        
        const auto& test_vector = test_vectors[test_idx];
        std::cout << "\nTest #" << (test_idx + 1) << ": " << vector_descriptions[test_idx] << std::endl;
        
        // Calculate ref result
        std::vector<float> reference(rows, 0.0f);
        spmv_ref_logical(row_offsets, col_indices, values, test_vector, reference);
        
        std::cout << "Reference result: ";
        for (float val : reference) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
        
        SparseMatrix<float> matrix(rows, cols, row_offsets, col_indices, values);
        
        // Create GraphBLAS vectors
        Vector<float> vec_x(cols, test_vector);
        
        // Run logical OR-AND semiring operation
        Vector<float> gpu_result = Operations<float>::spmv(matrix, vec_x, SemiringType::LogicalOrAnd);
        
        // Compile
        OpCompiler& compiler = OpCompiler::getInstance();
        compiler.compile();
        
        // Copy data to device
        compiler.copyHostToDevice(matrix);
        compiler.copyHostToDevice(vec_x);
        
        // Execute
        compiler.execute(1);
        
        // Get results
        std::vector<float> result(rows, 0.0f);
        compiler.copyDeviceToHost(result.data(), gpu_result.bufferId(), gpu_result.bytes());
        
        std::cout << "GPU result: ";
        for (float val : result) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
        
        // Verify results
        bool passed = verify(result, reference);
        if (passed) {
            std::cout << "Test #" << (test_idx + 1) << " PASSED!" << std::endl;
        } else {
            std::cout << "Test #" << (test_idx + 1) << " FAILED!" << std::endl;
        }
    }
    
    return 0;
}