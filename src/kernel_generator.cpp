#include <graphblas_gpu/kernel_generator.hpp>
#include <sstream>
#include <unordered_map>
#include "../kernels/ewise_ops.h"

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
    
    // Add includes
    ss << "#include <cuda_runtime.h>\n";
    ss << "#include <ewise_ops.h>\n";
    ss << "#include <ewise_ops.cu>\n\n";  // Include the implementation file
    
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
                // Get buffer IDs
                size_t result_id = op.buffer_ids[0];
                size_t lhs_id = op.buffer_ids[1];
                size_t rhs_id = op.buffer_ids[2];
                
                // Get buffer offsets
                size_t result_offset = buffer_id_to_offset_.at(result_id);
                size_t lhs_offset = buffer_id_to_offset_.at(lhs_id);
                size_t rhs_offset = buffer_id_to_offset_.at(rhs_id);
                
                // Get datatype and size
                std::string datatype = op.args.at("datatype");
                
                // Get the vector size from the lhs buffer information
                // For element-wise ops, we can just use either lhs or rhs size
                // since they should be the same
                size_t vec_size = 0;
                for (const auto& alloc_op : operations_) {
                    if (alloc_op.type == Op::Type::AllocVector && 
                        alloc_op.buffer_ids[0] == lhs_id) {
                        vec_size = std::stoul(alloc_op.args.at("size"));
                        break;
                    }
                }
                
                // If size wasn't found, we'll use the size from the operation args
                if (vec_size == 0 && op.args.find("size") != op.args.end()) {
                    vec_size = std::stoul(op.args.at("size"));
                }
                
                // Operation name for comments
                std::string op_name;
                std::string device_func;
                switch (op.type) {
                    case Op::Type::EWiseAdd: 
                        op_name = "EWiseAdd"; 
                        device_func = "ewise_add";
                        break;
                    case Op::Type::EWiseSub: 
                        op_name = "EWiseSub"; 
                        device_func = "ewise_sub";
                        break;
                    case Op::Type::EWiseMul: 
                        op_name = "EWiseMul"; 
                        device_func = "ewise_mul";
                        break;
                    case Op::Type::EWiseDiv: 
                        op_name = "EWiseDiv"; 
                        device_func = "ewise_div";
                        break;
                    default: 
                        op_name = "Unknown";
                        device_func = "unknown";
                }
                
                ss << "        // " << op_name << " operation\n";
                
                // Direct computation on elements with thread-based indexing
                if (datatype == "float") {
                    ss << "        for (size_t i = idx; i < " << vec_size << "; i += grid_size) {\n";
                    ss << "            graphblas_gpu::kernels::" << device_func << "<float>(\n";
                    ss << "                (float*)(buffer + " << lhs_offset << "),\n";
                    ss << "                (float*)(buffer + " << rhs_offset << "),\n";
                    ss << "                (float*)(buffer + " << result_offset << "),\n";
                    ss << "                i);\n";
                    ss << "        }\n";
                }
                else if (datatype == "double") {
                    ss << "        for (size_t i = idx; i < " << vec_size << "; i += grid_size) {\n";
                    ss << "            graphblas_gpu::kernels::" << device_func << "<double>(\n";
                    ss << "                (double*)(buffer + " << lhs_offset << "),\n";
                    ss << "                (double*)(buffer + " << rhs_offset << "),\n";
                    ss << "                (double*)(buffer + " << result_offset << "),\n";
                    ss << "                i);\n";
                    ss << "        }\n";
                }
                else if (datatype == "int" || datatype == "int32") {
                    ss << "        for (size_t i = idx; i < " << vec_size << "; i += grid_size) {\n";
                    ss << "            graphblas_gpu::kernels::" << device_func << "<int>(\n";
                    ss << "                (int*)(buffer + " << lhs_offset << "),\n";
                    ss << "                (int*)(buffer + " << rhs_offset << "),\n";
                    ss << "                (int*)(buffer + " << result_offset << "),\n";
                    ss << "                i);\n";
                    ss << "        }\n";
                }
                break;
            }
            
            // Handle other operations (SpMV, SpMM, etc.) similarly...
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