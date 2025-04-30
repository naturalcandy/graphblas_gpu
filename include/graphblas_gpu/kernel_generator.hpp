#ifndef GRAPHBLAS_GPU_KERNEL_GENERATOR_HPP
#define GRAPHBLAS_GPU_KERNEL_GENERATOR_HPP

#include <vector>
#include <unordered_map>
#include <set>
#include <string>
#include <memory>
#include <sstream>
#include <graphblas_gpu/op_sequence.hpp>

namespace graphblas_gpu {

class KernelGenerator {
public:
    KernelGenerator(const std::vector<Op>& operations,
                    const std::unordered_map<size_t, size_t>& buffer_id_to_offset);
    
    ~KernelGenerator() = default;
    
    // Generate CUDA kernel code 
    std::string generateCode();
    
    // Get the kernel function name
    std::string getKernelName() const;
    
private:
    std::vector<Op> operations_;
    std::unordered_map<size_t, size_t> buffer_id_to_offset_;
    std::string kernel_name_;
};

} // namespace graphblas_gpu

#endif // GRAPHBLAS_GPU_KERNEL_GENERATOR_HPP