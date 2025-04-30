#include <graphblas_gpu/operation.hpp>
#include <graphblas_gpu/op_compiler.hpp>
#include <iostream>
#include <vector>
#include <string>

int main() {
    graphblas_gpu::OpSequence::getInstance().clear();
    
    std::cout << "===== Testing Buffer Allocation & Op Scheduling =====" << std::endl;
    
    const size_t size = 1024;
    std::vector<float> h_a(size, 1.0f);
    std::vector<float> h_b(size, 2.0f);
    std::vector<double> h_c(size, 3.0);
    std::vector<int> h_d(size, 4);
    
    try {
        graphblas_gpu::Vector<float> vec_a(size, h_a);
        std::cout << "Created vec_a (float), buffer ID: " << vec_a.bufferId() << std::endl;
        
        graphblas_gpu::Vector<float> vec_b(size, h_b);
        std::cout << "Created vec_b (float), buffer ID: " << vec_b.bufferId() << std::endl;
        
        graphblas_gpu::Vector<double> vec_c(size, h_c);
        std::cout << "Created vec_c (double), buffer ID: " << vec_c.bufferId() << std::endl;
        
        graphblas_gpu::Vector<int> vec_d(size, h_d);
        std::cout << "Created vec_d (int), buffer ID: " << vec_d.bufferId() << std::endl;
        
        std::cout << "\nPerforming element-wise operations:" << std::endl;
        
        graphblas_gpu::Vector<float> result_add = vec_a + vec_b;
        std::cout << "vec_a + vec_b = result_add, buffer ID: " << result_add.bufferId() << std::endl;
        
        graphblas_gpu::Vector<float> result_sub = vec_a - vec_b;
        std::cout << "vec_a - vec_b = result_sub, buffer ID: " << result_sub.bufferId() << std::endl;
        
        graphblas_gpu::Vector<float> result_mul = vec_a * vec_b;
        std::cout << "vec_a * vec_b = result_mul, buffer ID: " << result_mul.bufferId() << std::endl;
        
        graphblas_gpu::Vector<float> result_div = vec_a / vec_b;
        std::cout << "vec_a / vec_b = result_div, buffer ID: " << result_div.bufferId() << std::endl;
        
        // Test chained operations
        std::cout << "\nTesting operation chaining (data dependencies):" << std::endl;
        graphblas_gpu::Vector<float> temp1 = vec_a + vec_b;
        std::cout << "temp1 = vec_a + vec_b, buffer ID: " << temp1.bufferId() << std::endl;
        
        graphblas_gpu::Vector<float> temp2 = vec_a - vec_b;
        std::cout << "temp2 = vec_a - vec_b, buffer ID: " << temp2.bufferId() << std::endl;
        
        graphblas_gpu::Vector<float> result_chain = temp1 * temp2;
        std::cout << "result_chain = temp1 * temp2, buffer ID: " << result_chain.bufferId() << std::endl;
        
        graphblas_gpu::Vector<float> complex_result = ((vec_a + vec_b) * (vec_a - vec_b)) / vec_b;
        std::cout << "complex_result = ((vec_a + vec_b) * (vec_a - vec_b)) / vec_b, buffer ID: " 
                  << complex_result.bufferId() << std::endl;
        
        // Compile operations to allocate the memory required
        std::cout << "\nCompiling operation sequence (calls allocateBuffers):" << std::endl;
        graphblas_gpu::OpCompiler& compiler = graphblas_gpu::OpCompiler::getInstance();
        compiler.compile();
        
        std::cout << "\nResetting compiler..." << std::endl;
        compiler.reset();
        
    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
    }
    
    return 0;
}