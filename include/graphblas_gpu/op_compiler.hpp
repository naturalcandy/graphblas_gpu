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
#include <optional>
#include <cuda.h>
#include <cuda_runtime.h>



namespace graphblas_gpu {

class KernelGenerator;

class OpCompiler {
public:
    static OpCompiler& getInstance();

    // Compile the current operation sequence
    void compile();
    
    // Execute compiled sequence
    void execute(std::optional<int> iterations = 1, 
                cudaEvent_t* timing_start = nullptr, 
                cudaEvent_t* timing_stop = nullptr);
    
    // Clean up resources
    void reset();
    
    // Access for allocated memory
    void* getBuffer(size_t buffer_id);
    
    // Copy data between host and device
    void copyHostToDevice(const void* host_data, size_t buffer_id, size_t size_bytes);
    void copyDeviceToHost(void* host_data, size_t buffer_id, size_t size_bytes);
    
    template <typename T>
    void copyHostToDevice(const Vector<T>& vec);

    template <typename T>
    void copyHostToDevice(const SparseMatrix<T>& matrix);

    template <typename T>
    void copyDeviceToHost(std::vector<T>& host_vec, const Vector<T>& device_vec);
    

    
private:
    OpCompiler();
    ~OpCompiler();
    
    
    OpCompiler(const OpCompiler&) = delete;
    OpCompiler& operator=(const OpCompiler&) = delete;
    
    void planBufferLayout();
    void generateKernel();
    bool compileAndLoadKernel(const std::string& kernel_code);
    
    bool is_compiled_;
    bool kernel_loaded_;
    
    // Map buffer IDs to GPU memory offsets
    std::unordered_map<size_t, size_t> buffer_offsets_;
    
    // Total memory size needed
    size_t total_bytes_;
    
    // GPU memory
    void* device_memory_;
    
    // Loop control
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