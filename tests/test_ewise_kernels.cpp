#include <graphblas_gpu/vector.hpp>
#include <graphblas_gpu/operation.hpp>
#include <graphblas_gpu/op_compiler.hpp>
#include <iostream>
#include <vector>

int main() {
    // Clear any previous operations
    graphblas_gpu::OpSequence::getInstance().clear();
    
    std::cout << "===== Testing Full Kernel Execution =====" << std::endl;
    // Create test vectors 
    const size_t size = 1024;
    std::vector<float> h_a(size, 1.0f);
    std::vector<float> h_b(size, 2.0f);
    std::vector<float> h_result_add(size);
    std::vector<float> h_result_sub(size);
    std::vector<float> h_result_mul(size);
    std::vector<float> h_result_div(size);
    
    // USe them to create our API vectors
    graphblas_gpu::Vector<float> vec_a(size, h_a);
    graphblas_gpu::Vector<float> vec_b(size, h_b);
    
    std::cout << "Created vectors with data" << std::endl;
    
    // Perform our operations
    graphblas_gpu::Vector<float> vec_add = vec_a + vec_b;
    graphblas_gpu::Vector<float> vec_sub = vec_a - vec_b;
    graphblas_gpu::Vector<float> vec_mul = vec_a * vec_b;
    graphblas_gpu::Vector<float> vec_div = vec_a / vec_b;
    
    std::cout << "Defined operations" << std::endl;
    
    graphblas_gpu::OpCompiler& compiler = graphblas_gpu::OpCompiler::getInstance();
    
    // Compile the operations
    compiler.compile();
    
    std::cout << "Compiled operations" << std::endl;
    
    // Copy input data to device
    compiler.copyHostToDevice(h_a.data(), vec_a.bufferId(), vec_a.bytes());
    compiler.copyHostToDevice(h_b.data(), vec_b.bufferId(), vec_b.bytes());
    
    std::cout << "Copied input data to device" << std::endl;
    
    // Execute the generated kernel
    compiler.execute(1);  // Run for 1 iteration
    
    std::cout << "Executed kernel" << std::endl;
    
    // Copy results back to host
    compiler.copyDeviceToHost(h_result_add.data(), vec_add.bufferId(), vec_add.bytes());
    compiler.copyDeviceToHost(h_result_sub.data(), vec_sub.bufferId(), vec_sub.bytes());
    compiler.copyDeviceToHost(h_result_mul.data(), vec_mul.bufferId(), vec_mul.bytes());
    compiler.copyDeviceToHost(h_result_div.data(), vec_div.bufferId(), vec_div.bytes());
    
    std::cout << "Copied results back to host" << std::endl;
    
    // Verify results
    bool all_correct = true;
    for (size_t i = 0; i < size; i++) {
        float expected_add = h_a[i] + h_b[i];
        float expected_sub = h_a[i] - h_b[i];
        float expected_mul = h_a[i] * h_b[i];
        float expected_div = h_a[i] / h_b[i];
        
        if ((h_result_add[i] != expected_add)||
            (h_result_sub[i] != expected_sub) ||
            (h_result_mul[i] != expected_mul)  ||
            (h_result_div[i] != expected_div)) {
            
            std::cout << "Error at index " << i << ":" << std::endl;
            std::cout << "  Add: " << h_result_add[i] << " (expected " << expected_add << ")" << std::endl;
            std::cout << "  Sub: " << h_result_sub[i] << " (expected " << expected_sub << ")" << std::endl;
            std::cout << "  Mul: " << h_result_mul[i] << " (expected " << expected_mul << ")" << std::endl;
            std::cout << "  Div: " << h_result_div[i] << " (expected " << expected_div << ")" << std::endl;
            
            all_correct = false;
            break;
        }
    }
    
    if (all_correct) {
        std::cout << "All results are correct!" << std::endl;
    } else {
        std::cout << "Some results are incorrect." << std::endl;
    }
    
    return 0;
}