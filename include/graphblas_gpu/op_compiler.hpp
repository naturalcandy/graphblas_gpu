#ifndef GRAPHBLAS_GPU_OP_COMPILER_HPP
#define GRAPHBLAS_GPU_OP_COMPILER_HPP

#include <graphblas_gpu/op_sequence.hpp>
#include <graphblas_gpu/vector.hpp>
#include <graphblas_gpu/graph.hpp>
#include <memory>
#include <unordered_map>
#include <vector>
#include <iostream>


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
    
private:
    OpCompiler();
    ~OpCompiler();
    
    
    OpCompiler(const OpCompiler&) = delete;
    OpCompiler& operator=(const OpCompiler&) = delete;
    
    void allocateBuffers();
    void generateKernel();
    
    bool is_compiled_;
    
    // Map buffer IDs to GPU memory offsets
    std::unordered_map<size_t, size_t> buffer_offsets_;
    
    // Total memory size needed for the program
    size_t total_memory_bytes_;
    
    // GPU memory
    void* device_memory_;
    
    // Kernel generation
    // std::unique_ptr<KernelGenerator> kernel_generator_;
    
    // Atomic for loop control
    int* iteration_flag_;
};

} // namespace graphblas_gpu

#endif // GRAPHBLAS_GPU_OP_COMPILER_HPP