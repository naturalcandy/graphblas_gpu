#include <graphblas_gpu/kernel_generator.hpp>
#include <sstream>
#include <unordered_map>
#include <set>
#include <string>
#include <graphblas_gpu/op_sequence.hpp>
#include <graphblas_gpu/graph.hpp>

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
    // omit if we do tree based grid sync
    ss << "#include <cooperative_groups.h>\n\n";
    ss << "namespace cg = cooperative_groups;\n\n";
  
    
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
    // this should be ommitted if we are doing our tree based grid sync..
    ss << "    // Create grid-wide group for end of iteration sync\n";
    ss << "    auto grid = cg::this_grid();\n";
    ss << "    auto block = cg::this_thread_block();\n\n";
    
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
                size_t resultId = op.buffer_ids[0];
                size_t matrixId = op.buffer_ids[1];
                size_t vectorId = op.buffer_ids[2];
                
                // Get buffer offsets
                size_t resultOffset = buffer_id_to_offset_.at(resultId);
                size_t matrixOffset = buffer_id_to_offset_.at(matrixId);
                size_t vectorOffset = buffer_id_to_offset_.at(vectorId);

                // Get parameters
                std::string datatype = op.args.at("datatype");
                size_t num_rows = std::stoul(op.args.at("num_rows"));
                std::string format = op.args.at("format");
                
                if (format == "CSR") {
                    size_t row_offsets_size = (num_rows + 1) * sizeof(size_t);
                    size_t nnz = std::stoul(op.args.at("nnz"));
                    size_t col_indices_size = nnz * sizeof(int);

                    size_t row_offsets_offset = matrixOffset;
                    size_t col_indices_offset = matrixOffset + row_offsets_size;
                    size_t values_offset = col_indices_offset + col_indices_size;

                    // Generate SpMV code
                    ss << "        // SpMV operation\n";
                    ss << "        graphblas_gpu::kernels::spmv_csr<" << datatype << ">(\n";
                    ss << "            (size_t*)(buffer + " << row_offsets_offset << "),\n";
                    ss << "            (int*)(buffer + " << col_indices_offset << "),\n";
                    ss << "            (" << datatype << "*)(buffer + " << values_offset << "),\n";
                    ss << "            (" << datatype << "*)(buffer + " << vectorOffset << "),\n";
                    ss << "            (" << datatype << "*)(buffer + " << resultOffset << "),\n";
                    ss << "            " << num_rows << ");\n";
                } else if (format == "ELL") {
                    size_t max_nnz = std::stoul(op.args.at("max_nnz_per_row"));
                    size_t ell_size = num_rows * max_nnz;
                    
                    size_t col_idx_offset = matrixOffset;
                    size_t val_offset = matrixOffset + (ell_size * sizeof(int));
            
                    ss << "        // ELL SpMV operation\n";
                    ss << "        graphblas_gpu::kernels::spmv_ell<" << datatype << ">(\n";
                    ss << "            (int*)(buffer + " << col_idx_offset << "),\n";
                    ss << "            (" << datatype << "*)(buffer + " << val_offset << "),\n";
                    ss << "            (" << datatype << "*)(buffer + " << vectorOffset << "),\n";
                    ss << "            (" << datatype << "*)(buffer + " << resultOffset << "),\n";
                    ss << "            " << num_rows << ",\n";
                    ss << "            " << max_nnz << ");\n";
                }
                else if (format == "SELLC") {
                    size_t slice_size = std::stoul(op.args.at("slice_size"));
                    size_t num_slices = (num_rows + slice_size - 1) / slice_size;

                    size_t sellc_len = op.args.find("sellc_len") != op.args.end() ? 
                       std::stoul(op.args.at("sellc_len")) : 
                       std::stoul(op.args.at("nnz"));
                    
                    size_t sptr_size = (num_slices + 1) * sizeof(size_t);
                    size_t slen_size = num_slices * sizeof(size_t);
                    
                    size_t sptr_offset = matrixOffset;
                    size_t slice_lengths_offset = matrixOffset + sptr_size;
                    size_t col_indices_offset = matrixOffset + sptr_size + slen_size;
                    size_t values_offset = col_indices_offset + (sellc_len * sizeof(int));
            
                    ss << "        // SELL-C SpMV operation\n";
                    ss << "        graphblas_gpu::kernels::spmv_sell_c<" << datatype << ">(\n";
                    ss << "            (size_t*)(buffer + " << sptr_offset << "),\n";
                    ss << "            (int*)(buffer + " << col_indices_offset << "),\n";
                    ss << "            (" << datatype << "*)(buffer + " << values_offset << "),\n";
                    ss << "            " << num_rows << ",\n";
                    ss << "            " << slice_size << ",\n";
                    ss << "            (" << datatype << "*)(buffer + " << vectorOffset << "),\n";
                    ss << "            (" << datatype << "*)(buffer + " << resultOffset << "));\n";
                }
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
    // again here we should change how we do grid sync based on logic in
    // our op compiler. most likely if number of blocks are too large for us
    // to do a cooperative launch we use our own custom grid syc.
    ss << "        grid.sync();\n";
    ss << "    }\n";  // End of iteration loop
    ss << "}\n";
    
    return ss.str();
}

} // namespace graphblas_gpu