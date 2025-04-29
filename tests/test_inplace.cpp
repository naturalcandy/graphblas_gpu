#include <graphblas_gpu/vector.hpp>
#include <graphblas_gpu/operation.hpp>
#include <graphblas_gpu/graph.hpp>
#include <graphblas_gpu/op_compiler.hpp>

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

int main() {
    using namespace graphblas_gpu;

    std::vector<size_t> row_offsets = {0, 2, 4, 5};
    std::vector<int> col_indices = {0, 2, 1, 2, 2};
    std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    SparseMatrix<float> A(3, 3, row_offsets, col_indices, values);

    std::vector<float> host_x = {1.0f, 1.0f, 1.0f};
    Vector<float> x_vec(host_x.size(), host_x);

    std::vector<float> host_y = {2.0f, 3.0f, 4.0f};
    Vector<float> y_vec(host_y.size(), host_y);

    Operations<float>::spmv(A, x_vec, x_vec, SemiringType::Arithmetic);

    // Stage EWise ops (all in-place on x_vec)
    x_vec += y_vec;  
    x_vec -= y_vec;  
    x_vec *= y_vec;  
    x_vec /= y_vec;  

    // Compile
    auto& compiler = OpCompiler::getInstance();
    compiler.compile();

    // Copy data to device
    compiler.copyHostToDevice(A);
    compiler.copyHostToDevice(x_vec);
    compiler.copyHostToDevice(y_vec);

    // Execute
    compiler.execute(1);

    // Copy result back
    std::vector<float> result(host_x.size());
    compiler.copyDeviceToHost(result, x_vec);

    // We expect to see this:
    // After SpMV: x = {3, 7, 5}
    // After adding y: x = {3+2, 7+3, 5+4} = {5, 10, 9}
    // After subbing y: x = {5-2, 10-3, 9-4} = {3, 7, 5}
    // After multiplying by y: x = {3*2, 7*3, 5*4} = {6, 21, 20}
    // After dividing by y: x = {6/2, 21/3, 20/4} = {3, 7, 5}

    std::vector<float> expected = {3.0f, 7.0f, 5.0f};

    for (size_t i = 0; i < expected.size(); ++i) {
        assert(result[i] == expected[i]);
    }

    std::cout << "Test PASSED" << std::endl;

    return 0;
}
