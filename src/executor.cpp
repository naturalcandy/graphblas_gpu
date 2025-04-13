#include <graphblas_gpu/api.hpp>
#include <iostream>

extern "C" void run_test_kernel();

void launch_example_kernel() {
    std::cout << "Launching kernel from CPU...\n";
    run_test_kernel();
}
