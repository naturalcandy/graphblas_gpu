#ifndef GRAPHBLAS_GPU_OP_COMPILER_HPP
#define GRAPHBLAS_GPU_OP_COMPILER_HPP

#include <graphblas_gpu/op_sequence.hpp>
#include <graphblas_gpu/vector.hpp>
#include <graphblas_gpu/graph.hpp>
#include <memory>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <set>
#include <string>
#include <cuda.h>


namespace graphblas_gpu {

class KernelGenerator;

class OpCompiler {
public:
    static OpCompiler& getInstance();

    // Compile the current operation sequence
    void compile();
    
    // Execute compiled sequence
    void execute(int iterations = 1);
    
    // Clean up resources
    void reset();
    
    // Access for allocated memory and other resources
    void* getBuffer(size_t buffer_id);
    
    // Copy data between host and device
    void copyHostToDevice(const void* host_data, size_t buffer_id, size_t size_bytes);
    void copyDeviceToHost(void* host_data, size_t buffer_id, size_t size_bytes);
    
    template <typename T>
    void copyHostToDevice(const Vector<T>& vec);

    template <typename T>
    void copyHostToDevice(const SparseMatrix<T>& matrix);
    
private:
    OpCompiler();
    ~OpCompiler();
    
    
    OpCompiler(const OpCompiler&) = delete;
    OpCompiler& operator=(const OpCompiler&) = delete;
    
    void allocateBuffers();
    void generateKernel();
    bool compileAndLoadKernel(const std::string& kernel_code);
    
    bool is_compiled_;
    bool kernel_loaded_;
    
    // Map buffer IDs to GPU memory offsets
    std::unordered_map<size_t, size_t> buffer_offsets_;
    
    // Total memory size needed for the program
    size_t total_memory_bytes_;
    
    // GPU memory
    void* device_memory_;
    
    // Atomic for loop control
    int* iteration_flag_;

    // Runtime compilation
    std::string kernel_code_;
    std::string kernel_name_;
    
    // CUDA driver API resources
    CUmodule cuModule_;
    CUfunction kernel_function_;

    bool is_file_exists(const std::string& filename);
};

} // namespace graphblas_gpu

#endif // GRAPHBLAS_GPU_OP_COMPILER_HPP