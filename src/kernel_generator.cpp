#include <graphblas_gpu/kernel_generator.hpp>
#include <sstream>
#include <unordered_map>
#include <set>
#include <string>
#include <graphblas_gpu/op_sequence.hpp>
#include <graphblas_gpu/graph.hpp>
#include <graphblas_gpu/semiring.hpp>
#include <graphblas_gpu/termination_condition.hpp>

namespace graphblas_gpu {

KernelGenerator::KernelGenerator(const std::vector<Op>& operations,
                                const std::unordered_map<size_t, size_t>& buffer_id_to_offset)
    : operations_(operations),
      buffer_id_to_offset_(buffer_id_to_offset),
      kernel_name_("graphblas_gpu_kernel") {}

std::string KernelGenerator::getKernelName() const {
    return kernel_name_;
}

std::string KernelGenerator::generateCode() {
    std::stringstream ss;
    
    ss << "#include <graphblas_gpu/kernels/graphblas_kernels.hpp>\n\n";
    ss << "#include <cooperative_groups.h>\n\n";
    ss << "namespace cg = cooperative_groups;\n\n";

    ss << "extern \"C\" __global__ void " << kernel_name_ << "(char* buffer, int num_iterations) {\n";
    
    ss << "    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;\n";
    ss << "    size_t grid_size = gridDim.x * blockDim.x;\n\n";
    ss << "    auto grid = cg::this_grid();\n";
    ss << "    auto block = cg::this_thread_block();\n\n";

    // Generate the operation body 
    auto generate_operation_body = [&](std::stringstream& ss_inner) {
        for (const auto& op : operations_) {
            switch (op.type) {
                case Op::Type::EWiseAdd:
                case Op::Type::EWiseSub:
                case Op::Type::EWiseMul:
                case Op::Type::EWiseDiv:
                case Op::Type::EWiseOr: {
                    size_t resId = op.buffer_ids[0];
                    size_t aId = op.buffer_ids[1];
                    size_t bId = op.buffer_ids[2];
                    auto resOff = buffer_id_to_offset_.at(resId);
                    auto aOff = buffer_id_to_offset_.at(aId);
                    auto bOff = buffer_id_to_offset_.at(bId);

                    std::string dt = op.args.at("datatype");
                    size_t n = std::stoul(op.args.at("size"));

                    std::string fn = (op.type == Op::Type::EWiseAdd ? "ewise_add"
                                    : op.type == Op::Type::EWiseSub ? "ewise_sub"
                                    : op.type == Op::Type::EWiseMul ? "ewise_mul"
                                    : op.type == Op::Type::EWiseDiv ? "ewise_div"
                                    : "ewise_or");

                    ss_inner << "    // " << fn << " (" << dt << ")\n";
                    ss_inner << "    for (size_t i = idx; i < " << n << "; i += grid_size) {\n";
                    ss_inner << "        graphblas_gpu::kernels::" << fn << "<" << dt << ">(\n";
                    ss_inner << "            (" << dt << "*)(buffer + " << aOff << "),\n";
                    ss_inner << "            (" << dt << "*)(buffer + " << bOff << "),\n";
                    ss_inner << "            (" << dt << "*)(buffer + " << resOff << "),\n";
                    ss_inner << "            i);\n";
                    ss_inner << "    }\n\n";
                    break;
                }
                case Op::Type::EWiseAddInPlace:
                case Op::Type::EWiseSubInPlace:
                case Op::Type::EWiseMulInPlace:
                case Op::Type::EWiseDivInPlace:
                case Op::Type::EWiseOrInPlace:  {
                    size_t aId = op.buffer_ids[0];  
                    size_t bId = op.buffer_ids[1];  
                    auto aOff = buffer_id_to_offset_.at(aId);
                    auto bOff = buffer_id_to_offset_.at(bId);

                    std::string dt = op.args.at("datatype");
                    size_t n = std::stoul(op.args.at("size"));

                    std::string fn = (op.type == Op::Type::EWiseAddInPlace ? "ewise_add_inplace"
                                    : op.type == Op::Type::EWiseSubInPlace ? "ewise_sub_inplace"
                                    : op.type == Op::Type::EWiseMulInPlace ? "ewise_mul_inplace"
                                    : op.type == Op::Type::EWiseDivInPlace ? "ewise_div_inplace"
                                    : "ewise_or_inplace");

                    ss_inner << "    // " << fn << " (" << dt << ")\n";
                    ss_inner << "    for (size_t i = idx; i < " << n << "; i += grid_size) {\n";
                    ss_inner << "        graphblas_gpu::kernels::" << fn << "<" << dt << ">(\n";
                    ss_inner << "            (" << dt << "*)(buffer + " << aOff << "),\n";
                    ss_inner << "            (" << dt << "*)(buffer + " << bOff << "),\n";
                    ss_inner << "            i);\n";
                    ss_inner << "    }\n\n";
                    break;
                }
                case Op::Type::SpMV: {
                    size_t resId = op.buffer_ids[0];
                    size_t matrixId = op.buffer_ids[1];
                    size_t vectorId = op.buffer_ids[2];
                    size_t resOffset = buffer_id_to_offset_.at(resId);
                    size_t matOffset = buffer_id_to_offset_.at(matrixId);
                    size_t vecOffset = buffer_id_to_offset_.at(vectorId);

                    std::string datatype = op.args.at("datatype");
                    size_t num_rows = std::stoul(op.args.at("num_rows"));
                    std::string format = op.args.at("format");

                    bool mask_enabled = (op.args.find("mask") != op.args.end() && op.args.at("mask") == "true");
                    bool in_place = (op.args.find("in_place") != op.args.end() && op.args.at("in_place") == "true");
                    size_t maskOffset = 0;
                    if (mask_enabled) {
                        size_t mask_buffer_id = op.buffer_ids[3];
                        maskOffset = buffer_id_to_offset_.at(mask_buffer_id);
                    }
                    if (in_place) {
                        resOffset = vecOffset;
                    }
                    if (format == "CSR") {
                        size_t row_offsets_offset = matOffset;
                        size_t col_indices_offset = matOffset + (num_rows + 1) * sizeof(size_t);
                        size_t values_offset = col_indices_offset + std::stoul(op.args.at("nnz")) * sizeof(int);
                        
                        int semiring = std::stoi(op.args.at("semiring"));
                        std::string func_name = (semiring == static_cast<int>(SemiringType::LogicalOrAnd)) ? "spmv_csr_logical" : "spmv_csr";

                        ss_inner << "    // SpMV CSR operation\n";
                        ss_inner << "    graphblas_gpu::kernels::" << func_name << "<" << datatype << ">(\n";
                        ss_inner << "        (size_t*)(buffer + " << row_offsets_offset << "),\n";
                        ss_inner << "        (int*)(buffer + " << col_indices_offset << "),\n";
                        ss_inner << "        (" << datatype << "*)(buffer + " << values_offset << "),\n";
                        ss_inner << "        (" << datatype << "*)(buffer + " << vecOffset << "),\n";
                        if (mask_enabled) {
                            ss_inner << "        (" << datatype << "*)(buffer + " << maskOffset << "),\n";
                            ss_inner << "        true,\n";
                        } else {
                            ss_inner << "        nullptr,\n";
                            ss_inner << "        false,\n";
                        }
                        ss_inner << "        (" << datatype << "*)(buffer + " << resOffset << "),\n";
                        ss_inner << "        " << num_rows << ");\n\n";
                    } 
                    else if (format == "ELL") {
                        size_t ell_size = num_rows * std::stoul(op.args.at("max_nnz_per_row"));
                        size_t col_idx_offset = matOffset;
                        size_t val_offset = col_idx_offset + ell_size * sizeof(int);

                        int semiring = std::stoi(op.args.at("semiring"));
                        std::string func_name = (semiring == static_cast<int>(SemiringType::LogicalOrAnd)) ? "spmv_ell_logical" : "spmv_ell";

                        ss_inner << "    // SpMV ELL operation\n";
                        ss_inner << "    graphblas_gpu::kernels::" << func_name << "<" << datatype << ">(\n";
                        ss_inner << "        (int*)(buffer + " << col_idx_offset << "),\n";
                        ss_inner << "        (" << datatype << "*)(buffer + " << val_offset << "),\n";
                        ss_inner << "        (" << datatype << "*)(buffer + " << vecOffset << "),\n";
                        if (mask_enabled) {
                            ss_inner << "        (" << datatype << "*)(buffer + " << maskOffset << "),\n";
                            ss_inner << "        true,\n";
                        } else {
                            ss_inner << "        nullptr,\n";
                            ss_inner << "        false,\n";
                        }
                        ss_inner << "        (" << datatype << "*)(buffer + " << resOffset << "),\n";
                        ss_inner << "        " << num_rows << ",\n";
                        ss_inner << "        " << std::stoul(op.args.at("max_nnz_per_row")) << ");\n\n";
                    }
                    else if (format == "SELLC") {
                        size_t slice_size = std::stoul(op.args.at("slice_size"));
                        size_t sptr_offset = matOffset;
                        size_t slice_lengths_offset = matOffset + (std::stoul(op.args.at("num_rows")) / slice_size + 1) * sizeof(size_t);
                        size_t col_indices_offset = slice_lengths_offset + (std::stoul(op.args.at("num_rows")) / slice_size) * sizeof(size_t);
                        size_t values_offset = col_indices_offset + std::stoul(op.args.at("sellc_len")) * sizeof(int);

                        int semiring = std::stoi(op.args.at("semiring"));
                        std::string func_name = (semiring == static_cast<int>(SemiringType::LogicalOrAnd)) ? "spmv_sell_c_logical" : "spmv_sell_c";

                        ss_inner << "    // SpMV SELLC operation\n";
                        ss_inner << "    graphblas_gpu::kernels::" << func_name << "<" << datatype << ">(\n";
                        ss_inner << "        (size_t*)(buffer + " << sptr_offset << "),\n";
                        ss_inner << "        (int*)(buffer + " << col_indices_offset << "),\n";
                        ss_inner << "        (" << datatype << "*)(buffer + " << values_offset << "),\n";
                        ss_inner << "        " << num_rows << ",\n";
                        ss_inner << "        " << slice_size << ",\n";
                        ss_inner << "        (" << datatype << "*)(buffer + " << vecOffset << "),\n";
                        if (mask_enabled) {
                            ss_inner << "        (" << datatype << "*)(buffer + " << maskOffset << "),\n";
                            ss_inner << "        true,\n";
                        } else {
                            ss_inner << "        nullptr,\n";
                            ss_inner << "        false,\n";
                        }
                        ss_inner << "        (" << datatype << "*)(buffer + " << resOffset << "));\n\n";
                    }
                    break;
                }

                case Op::Type::SpMM: {
                    ss_inner << "    // SpMM not yet implemented\n";
                    break;
                }
                case Op::Type::Copy: {
                    size_t dst_id = op.buffer_ids[0];
                    size_t src_id = op.buffer_ids[1];
                    auto dst_off = buffer_id_to_offset_.at(dst_id);
                    auto src_off = buffer_id_to_offset_.at(src_id);
                    
                    std::string dt = op.args.at("datatype");
                    size_t n = std::stoul(op.args.at("size"));
                    
                    ss << "    // Copy operation (" << dt << ")\n";
                    ss << "    for (size_t i = idx; i < " << n << "; i += grid_size) {\n";
                    ss << "      graphblas_gpu::kernels::vector_copy<" << dt << ">(\n";
                    ss << "          (" << dt << "*)(buffer + " << src_off << "),\n";
                    ss << "          (" << dt << "*)(buffer + " << dst_off << "),\n";
                    ss << "          i);\n";
                    ss << "    }\n\n";
                    break;
                }

                default:
                    break;
            }
        }
    };

    // Generate the main loop
    bool use_term = TerminationCondition::getInstance().isActive();
    bool need_all = TerminationCondition::getInstance().requireAllThreads();

    if (use_term) {
        // each block will havae a flag used to indicate the satisfaction of
        // a termination predicate.
        ss << "    __shared__ bool block_flag;\n";
        ss << "    __device__ static int  blocks_voted = 0;\n";
        ss << "    __device__ static bool global_done  = false;\n\n";
        
        // Loop indefinitely until we reach our terminiatio ncondition
        ss << "    while (true) {\n";
        
        // For an AND-reduction the logical identity is true. For OR it's false
        if (need_all) {                                   
            ss << "        if (threadIdx.x == 0) block_flag = true;\n";
        } else {                                          // or
            ss << "        if (threadIdx.x == 0) block_flag = false;\n";
        }
        ss << "        __syncthreads();\n\n";
    
        generate_operation_body(ss);
        ss << TerminationCondition::getInstance().getConditionCode() << "\n";
        
        // We need each thread to fold its local predicate into the block-wide flag
        // using the correct reduction logic.
        if (need_all) {
            ss << "        if (!should_terminate) block_flag = false;\n";
        } else {
            ss << "        if (should_terminate)  block_flag = true;\n";
        }
        ss << "        __syncthreads();\n\n";
        
        // We assign a thread to increment grid wide counter to singal that its
        // block meets the termination predicate
        ss << "        if (threadIdx.x == 0) {\n";
        ss << "            if (block_flag) atomicAdd(&blocks_voted, 1);\n";
        ss << "        }\n";
        ss << "        grid.sync();\n\n";
    
        ss << "        if (threadIdx.x == 0 && blockIdx.x == 0) {\n";
        if (need_all) {
            // if all blocks signal we meet termination predicate (AND) we return
            ss << "            if (blocks_voted == gridDim.x) global_done = true;\n";
        } else {
            // if there exist at laest one block that meets termination predicate (OR) we return
            ss << "            if (blocks_voted > 0)          global_done = true;\n";
        }
        ss << "            blocks_voted = 0;\n";
        ss << "        }\n";
        ss << "        grid.sync();\n";
        ss << "        if (global_done) break;\n";
        ss << "    }\n";
    } else {
        // we are just running for fixed number of iterations
        ss << "    for (int iter = 0; iter < num_iterations; ++iter) {\n";
        generate_operation_body(ss);
        ss << "        grid.sync();\n";
        ss << "    }\n";
    }
    ss << "}\n"; 

    std::string code = ss.str();

    // since we need to support dynamic termination predicates provided by the user
    // prior to compiling the actual kernel code, we should just do this string 
    // replace logic where any buffer referenced in the termination condition
    // can easily be later mapped to a GPU memory address by using the buffer id
    // as an identifier.
    if (use_term) {
        for (const auto& [buffer_id, offset] : buffer_id_to_offset_) {
            std::string old = "BUF" + std::to_string(buffer_id);
            std::string new = "(buffer + " + std::to_string(offset) + ")";
            size_t pos = 0;
            while ((pos = code.find(old, pos)) != std::string::npos) {
                code.replace(pos, old.length(), new);
                pos += new.length();
            }
        }
    }
    return code;
}

} // namespace graphblas_gpu