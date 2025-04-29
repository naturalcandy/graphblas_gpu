#include <graphblas_gpu/vector.hpp>
#include <graphblas_gpu/op_compiler.hpp>

#include <iostream>
#include <vector>
#include <cassert>

int main() {
    using namespace graphblas_gpu;

    // Create a Vector A with some host data
    std::vector<float> host_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    Vector<float> A(host_data.size(), host_data);

    // Create an empty Vector B (same size)
    Vector<float> B(host_data.size());

    // Stage the copy operation
    Vector<float>::copy(A, B);

    // Compile all staged operations
    auto& compiler = OpCompiler::getInstance();
    compiler.compile();

    // Copy the initial host data to device
    compiler.copyHostToDevice(A);

    // Execute the graph (run staged ops)
    compiler.execute(1); 

    // Fetch result from device
    std::vector<float> result_data(B.size());
    compiler.copyDeviceToHost(result_data, B);

    // Verify
    std::cout << "Result vector B: ";
    for (float val : result_data) {
        std::cout << val << " ";
    }
    std::cout << "\n";

    for (size_t i = 0; i < host_data.size(); ++i) {
        assert(result_data[i] == host_data[i]);
    }

    std::cout << "Test PASSED: Vector copy is correct!" << std::endl;

    return 0;
}
