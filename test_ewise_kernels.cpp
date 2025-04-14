#include <cuda_runtime.h>
#include "kernels/ewise_ops.h"  
#include <iostream>
#include <vector>
#include <cmath>

#define CHECK_CUDA_ERROR(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << " at " \
                  << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

template <typename T>
void printArray(const std::vector<T>& arr, const std::string& name) {
    std::cout << name << " = [";
    for (size_t i = 0; i < arr.size(); ++i) {
        std::cout << arr[i];
        if (i < arr.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}

template <typename T>
bool verifyResults(const std::vector<T>& expected, const std::vector<T>& actual, T epsilon = 1e-5) {
    if (expected.size() != actual.size()) {
        std::cout << "Size mismatch: expected " << expected.size() 
                  << ", got " << actual.size() << std::endl;
        return false;
    }
    
    bool success = true;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::abs(expected[i] - actual[i]) > epsilon) {
            std::cout << "Mismatch at index " << i << ": expected " 
                      << expected[i] << ", got " << actual[i] << std::endl;
            success = false;
        }
    }
    return success;
}

int main() {
    // Test parameters
    const size_t size = 1024;
    
    // Allocate host memory
    std::vector<float> h_a(size);
    std::vector<float> h_b(size);
    std::vector<float> h_result(size);
    
    // Initialize test data
    for (size_t i = 0; i < size; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(size - i);
    }
    
    // Allocate device memory
    float *d_a, *d_b, *d_result;
    CHECK_CUDA_ERROR(cudaMalloc(&d_a, size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b, size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_result, size * sizeof(float)));
    
    // Copy input data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a.data(), size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b.data(), size * sizeof(float), cudaMemcpyHostToDevice));
    
    std::cout << "Testing element-wise kernel operations...\n";
    
    // Test element-wise addition
    graphblas_gpu::kernels::ewiseAdd<float>(d_a, d_b, d_result, size);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaMemcpy(h_result.data(), d_result, size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Compute expected results
    std::vector<float> expected_add(size);
    for (size_t i = 0; i < size; ++i) {
        expected_add[i] = h_a[i] + h_b[i];
    }
    
    // Verify addition results
    bool add_success = verifyResults(expected_add, h_result);
    std::cout << "Element-wise addition test: " 
              << (add_success ? "PASSED" : "FAILED") << std::endl;
    
    // Print a few values for verification
    std::cout << "First few values:\n";
    printArray(std::vector<float>(h_a.begin(), h_a.begin() + 5), "a");
    printArray(std::vector<float>(h_b.begin(), h_b.begin() + 5), "b");
    printArray(std::vector<float>(h_result.begin(), h_result.begin() + 5), "a + b");
    
    // Test element-wise subtraction
    graphblas_gpu::kernels::ewiseSub<float>(d_a, d_b, d_result, size);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaMemcpy(h_result.data(), d_result, size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Compute expected results
    std::vector<float> expected_sub(size);
    for (size_t i = 0; i < size; ++i) {
        expected_sub[i] = h_a[i] - h_b[i];
    }
    
    // Verify subtraction results
    bool sub_success = verifyResults(expected_sub, h_result);
    std::cout << "Element-wise subtraction test: " 
              << (sub_success ? "PASSED" : "FAILED") << std::endl;
    printArray(std::vector<float>(h_result.begin(), h_result.begin() + 5), "a - b");
    
    CHECK_CUDA_ERROR(cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFree(d_result));
    
    return 0;
}