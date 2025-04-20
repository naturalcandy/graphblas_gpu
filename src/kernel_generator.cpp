#include <graphblas_gpu/kernel_generator.hpp>
#include <sstream>
#include <unordered_map>
#include <set>
#include <string>
#include <graphblas_gpu/op_sequence.hpp>

namespace graphblas_gpu {

KernelGenerator::KernelGenerator(const std::vector<Op>& operations,
                                const std::unordered_map<size_t, size_t>& buffer_id_to_offset,
                                const std::set<size_t>& extra_buffer_ids)
    : operations_(operations),
      buffer_id_to_offset_(buffer_id_to_offset),
      extra_buffer_ids_(extra_buffer_ids),
      kernel_name_("graphblas_gpu_kernel") {}

std::string KernelGenerator::getKernelName() const {
    return kernel_name_;
}

std::string KernelGenerator::generateCode() {
    std::stringstream ss;
    
    ss << "#include <graphblas_gpu/kernels/graphblas_kernels.hpp>\n\n";
  
    
    // Main kernel function
    ss << "extern \"C\" __global__ void " << kernel_name_ << "(char* buffer, int num_iterations";
    
    // Add extra buffer arguments
    for (auto buf_id : extra_buffer_ids_) {
        ss << ", void* ext_buf_" << buf_id;
    }
    
    ss << ") {\n";
    ss << "    // Get thread index\n";
    ss << "    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;\n";
    ss << "    size_t grid_size = gridDim.x * blockDim.x;\n\n";
    
    // Execute operations for each iteration if needed
    ss << "    for (int iter = 0; iter < num_iterations; iter++) {\n";
    
    // Generate code for each operation
    for (const auto& op : operations_) {
        // Generate code based on operation type
        switch (op.type) {
            case Op::Type::EWiseAdd:
            case Op::Type::EWiseSub:
            case Op::Type::EWiseMul:
            case Op::Type::EWiseDiv: {
                // fetch buffers & offsets
                size_t resId = op.buffer_ids[0];
                size_t aId   = op.buffer_ids[1];
                size_t bId   = op.buffer_ids[2];
                auto resOff  = buffer_id_to_offset_.at(resId);
                auto aOff    = buffer_id_to_offset_.at(aId);
                auto bOff    = buffer_id_to_offset_.at(bId);
    
                // direct lookup
                std::string dt = op.args.at("datatype");
                size_t n = std::stoul(op.args.at("size"));
    
                // pick correct func
                std::string fn = (op.type == Op::Type::EWiseAdd ? "ewise_add"
                                      : op.type == Op::Type::EWiseSub ? "ewise_sub"
                                      : op.type == Op::Type::EWiseMul ? "ewise_mul"
                                      :                                "ewise_div");
    
                ss << "    // " << fn << " (" << dt << ")  \n";
                ss << "    for (size_t i = idx; i < " << n << "; i += grid_size) {\n";
                ss << "      graphblas_gpu::kernels::" << fn << "<" << dt << ">(\n";
                ss << "          (" << dt << "*)(buffer + " << aOff << "),\n";
                ss << "          (" << dt << "*)(buffer + " << bOff << "),\n";
                ss << "          (" << dt << "*)(buffer + " << resOff << "),\n";
                ss << "          i);\n";
                ss << "    }\n\n";
                break;
            }
            
            case Op::Type::SpMV: {
                // TODO: Implement SpMV operation
                ss << "        // SpMV operation - Not yet implemented\n";
                break;
            }
            
            case Op::Type::SpMM: {
                // TODO: Implement SpMM operation
                ss << "        // SpMM operation - Not yet implemented\n";
                break;
            }
            
            default:
                break;
        }
    }
    
    // Add necessary synchronization between iterations
    ss << "        __syncthreads();\n";
    ss << "    }\n";  // End of iteration loop
    ss << "}\n";
    
    return ss.str();
}

} // namespace graphblas_gpu