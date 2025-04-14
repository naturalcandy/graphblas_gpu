#include <graphblas_gpu/op_compiler.hpp>
#include <cuda_runtime.h>    
#include <cuda.h>           
#include <iostream>




namespace graphblas_gpu {

OpCompiler& OpCompiler::getInstance() {
    static OpCompiler instance;
    return instance;
}

OpCompiler::OpCompiler() 
    : is_compiled_(false), 
      total_memory_bytes_(0), 
      device_memory_(nullptr), 
      iteration_flag_(nullptr) {
}

OpCompiler::~OpCompiler() {
    reset();
}

void OpCompiler::compile() {
    if (is_compiled_) {
        reset();
    }
    // Analyze the operation sequence
    const auto& ops = OpSequence::getInstance().getOps();
    if (ops.empty()) {
        std::cerr << "Warning: No operations to compile" << std::endl;
        return;
    }
    
    // Allocate GPU memory for all buffers
    allocateBuffers();
    
    // Generate kernel for the operation sequence
    generateKernel();
    
    // Allocate host memory for iteration flag (for loop control)
    cudaHostAlloc(&iteration_flag_, sizeof(int), cudaHostAllocMapped);
    *iteration_flag_ = 0;
    
    is_compiled_ = true;
}

void OpCompiler::allocateBuffers() {
    const auto& ops = OpSequence::getInstance().getOps();
    total_memory_bytes_ = 0;

    std::cout << "\n==== OpCompiler::allocateBuffers() ====" << std::endl;
    std::cout << "Number of operations: " << ops.size() << std::endl;

    // Track buffer metadata (rows, cols, vector size)
    struct BufferInfo {
        size_t rows = 0;
        size_t cols = 0;
        size_t nnz = 0;
        size_t size_bytes = 0;
        DataType datatype = DataType(DataTypeEnum::Unknown);
        std::string origin; // Which operation created this buffer
    };

    std::unordered_map<size_t, BufferInfo> buffer_info_map;

    // First pass: analyze operations and determine buffer requirements
    for (size_t i = 0; i < ops.size(); i++) {
        const auto& op = ops[i];
        
        std::cout << "\nOperation " << i << ": " << op.name 
                  << " (Type: " << static_cast<int>(op.type) << ")" << std::endl;
        
        std::cout << "  Buffer IDs:";
        for (auto id : op.buffer_ids) {
            std::cout << " " << id;
        }
        std::cout << std::endl;
        
        std::cout << "  Arguments:" << std::endl;
        for (const auto& [key, value] : op.args) {
            std::cout << "    " << key << ": " << value << std::endl;
        }

        switch (op.type) {
            case Op::Type::AllocVector: {
                size_t buffer_id = op.buffer_ids[0];
                size_t vec_size = std::stoul(op.args.at("size"));
                std::string datatype_str = op.args.at("datatype");
                
                // Parse the datatype string to get a DataType object
                DataType datatype;
                try {
                    datatype = dataTypeFromString(datatype_str);
                } catch (const std::exception& e) {
                    std::cerr << "  WARNING: " << e.what() << " - Defaulting to Float" << std::endl;
                    datatype = DataType(DataTypeEnum::Float);
                }
                
                size_t buffer_size = vec_size * datatype.sizeInBytes();

                buffer_info_map[buffer_id] = {
                    .rows = vec_size,
                    .cols = 1,
                    .size_bytes = buffer_size,
                    .datatype = datatype,
                    .origin = "AllocVector"
                };
                
                std::cout << "  Allocated Vector Buffer ID " << buffer_id 
                          << ": size=" << vec_size 
                          << ", bytes=" << buffer_size 
                          << ", datatype=" << datatype.toString() << std::endl;
                break;
            }
            case Op::Type::AllocGraph: {
                size_t buffer_id = op.buffer_ids[0];
                size_t rows = std::stoul(op.args.at("rows"));
                size_t cols = std::stoul(op.args.at("cols"));
                size_t nnz = std::stoul(op.args.at("nnz"));
                std::string datatype_str = op.args.at("datatype");
                
                // Parse the datatype string to get a DataType object
                DataType datatype;
                try {
                    datatype = dataTypeFromString(datatype_str);
                } catch (const std::exception& e) {
                    std::cerr << "  WARNING: " << e.what() << " - Defaulting to Float" << std::endl;
                    datatype = DataType(DataTypeEnum::Float);
                }

                size_t buffer_size =
                    (rows + 1) * sizeof(size_t) +  // row_offsets
                    nnz * sizeof(size_t) +         // col_indices
                    nnz * datatype.sizeInBytes();  // values

                buffer_info_map[buffer_id] = {
                    .rows = rows,
                    .cols = cols,
                    .nnz = nnz,
                    .size_bytes = buffer_size,
                    .datatype = datatype,
                    .origin = "AllocGraph"
                };
                
                std::cout << "  Allocated Graph Buffer ID " << buffer_id 
                          << ": rows=" << rows 
                          << ", cols=" << cols
                          << ", nnz=" << nnz
                          << ", bytes=" << buffer_size 
                          << ", datatype=" << datatype.toString() << std::endl;
                break;
            }
            case Op::Type::SpMV: {
                size_t result_buffer_id = op.buffer_ids[0];
                size_t mat_buffer_id = op.buffer_ids[1];
                size_t vec_buffer_id = op.buffer_ids[2];
                std::string datatype_str = op.args.at("datatype");
                
                // Parse the datatype string to get a DataType object
                DataType datatype;
                try {
                    datatype = dataTypeFromString(datatype_str);
                } catch (const std::exception& e) {
                    std::cerr << "  WARNING: " << e.what() << " - Defaulting to Float" << std::endl;
                    datatype = DataType(DataTypeEnum::Float);
                }

                // Check if the input buffers exist
                if (buffer_info_map.find(mat_buffer_id) == buffer_info_map.end()) {
                    std::cerr << "  ERROR: Matrix buffer ID " << mat_buffer_id << " not found!" << std::endl;
                    continue;
                }
                if (buffer_info_map.find(vec_buffer_id) == buffer_info_map.end()) {
                    std::cerr << "  ERROR: Vector buffer ID " << vec_buffer_id << " not found!" << std::endl;
                    continue;
                }

                const auto& mat_info = buffer_info_map.at(mat_buffer_id);
                const auto& vec_info = buffer_info_map.at(vec_buffer_id);
                
                // Check dimensions compatibility
                if (mat_info.cols != vec_info.rows) {
                    std::cerr << "  ERROR: Matrix-vector dimension mismatch: "
                              << "matrix is " << mat_info.rows << "x" << mat_info.cols
                              << ", vector is " << vec_info.rows << "x1" << std::endl;
                }
                
                std::cout << "  SpMV Operation: Matrix(" << mat_buffer_id 
                          << ") * Vector(" << vec_buffer_id 
                          << ") -> Result(" << result_buffer_id << ")" << std::endl;
                std::cout << "    Matrix: " << mat_info.rows << "x" << mat_info.cols 
                          << ", Vector: " << vec_info.rows << "x1" << std::endl;
                
                size_t result_vec_size = mat_info.rows;  // m × 1 result
                size_t buffer_size = result_vec_size * datatype.sizeInBytes();

                buffer_info_map[result_buffer_id] = {
                    .rows = result_vec_size,
                    .cols = 1,
                    .size_bytes = buffer_size,
                    .datatype = datatype,
                    .origin = "SpMV"
                };
                
                std::cout << "  SpMV Result Vector Buffer ID " << result_buffer_id 
                          << ": size=" << result_vec_size 
                          << ", bytes=" << buffer_size 
                          << ", datatype=" << datatype.toString() << std::endl;
                break;
            }
            case Op::Type::EWiseAdd:
            case Op::Type::EWiseSub:
            case Op::Type::EWiseMul:
            case Op::Type::EWiseDiv: {
                std::string op_name;
                switch (op.type) {
                    case Op::Type::EWiseAdd: op_name = "EWiseAdd"; break;
                    case Op::Type::EWiseSub: op_name = "EWiseSub"; break;
                    case Op::Type::EWiseMul: op_name = "EWiseMul"; break;
                    case Op::Type::EWiseDiv: op_name = "EWiseDiv"; break;
                    default: op_name = "Unknown";
                }
                
                size_t result_buffer_id = op.buffer_ids[0];
                size_t lhs_buffer_id = op.buffer_ids[1];
                size_t rhs_buffer_id = op.buffer_ids[2];
                std::string datatype_str = op.args.at("datatype");
                
                // Parse the datatype string to get a DataType object
                DataType datatype;
                try {
                    datatype = dataTypeFromString(datatype_str);
                } catch (const std::exception& e) {
                    std::cerr << "  WARNING: " << e.what() << " - Defaulting to Float" << std::endl;
                    datatype = DataType(DataTypeEnum::Float);
                }

                // Check if the input buffers exist
                if (buffer_info_map.find(lhs_buffer_id) == buffer_info_map.end()) {
                    std::cerr << "  ERROR: LHS buffer ID " << lhs_buffer_id << " not found!" << std::endl;
                    continue;
                }
                if (buffer_info_map.find(rhs_buffer_id) == buffer_info_map.end()) {
                    std::cerr << "  ERROR: RHS buffer ID " << rhs_buffer_id << " not found!" << std::endl;
                    continue;
                }

                const auto& lhs_info = buffer_info_map.at(lhs_buffer_id);
                const auto& rhs_info = buffer_info_map.at(rhs_buffer_id);
                
                std::cout << "  " << op_name << " Operation: Vector(" << lhs_buffer_id 
                          << ", from " << lhs_info.origin << ") op Vector(" << rhs_buffer_id 
                          << ", from " << rhs_info.origin << ") -> Result(" << result_buffer_id << ")" << std::endl;
                std::cout << "    LHS: " << lhs_info.rows << "x" << lhs_info.cols 
                          << ", RHS: " << rhs_info.rows << "x" << rhs_info.cols << std::endl;
                
                // Verify dimensions are compatible
                if (lhs_info.rows != rhs_info.rows || lhs_info.cols != rhs_info.cols) {
                    std::cerr << "  WARNING: Dimension mismatch in element-wise operation!" << std::endl;
                }
                
                size_t result_size = lhs_info.rows * lhs_info.cols; // same as lhs/rhs
                size_t buffer_size = result_size * datatype.sizeInBytes();

                buffer_info_map[result_buffer_id] = {
                    .rows = lhs_info.rows,
                    .cols = lhs_info.cols,
                    .size_bytes = buffer_size,
                    .datatype = datatype,
                    .origin = op_name
                };
                
                std::cout << "  Result Vector Buffer ID " << result_buffer_id 
                          << ": size=" << result_size 
                          << ", bytes=" << buffer_size 
                          << ", datatype=" << datatype.toString() << std::endl;
                break;
            }
            default:
                std::cerr << "  WARNING: Operation type " << static_cast<int>(op.type) 
                          << " not supported yet" << std::endl;
        }
    }

    std::cout << "\n==== Buffer Information Summary ====" << std::endl;
    for (const auto& [buffer_id, buf_info] : buffer_info_map) {
        std::cout << "Buffer ID " << buffer_id << ":" 
                  << " rows=" << buf_info.rows
                  << " cols=" << buf_info.cols
                  << " nnz=" << buf_info.nnz
                  << " size_bytes=" << buf_info.size_bytes
                  << " datatype=" << buf_info.datatype.toString()
                  << " origin=" << buf_info.origin
                  << std::endl;
    }

    // Now we assign offsets and total buffer size to allocate on device
    std::cout << "\n==== Memory Layout ====" << std::endl;
    for (const auto& [buffer_id, buf_info] : buffer_info_map) {
        size_t aligned_size = (buf_info.size_bytes + 255) & ~255; // 256-byte alignment
        buffer_offsets_[buffer_id] = total_memory_bytes_;
        
        std::cout << "Buffer ID " << buffer_id 
                  << ": offset=" << total_memory_bytes_
                  << " bytes=" << buf_info.size_bytes
                  << " aligned_bytes=" << aligned_size
                  << " (padding=" << (aligned_size - buf_info.size_bytes) << " bytes)"
                  << std::endl;
        
        total_memory_bytes_ += aligned_size;
    }

    std::cout << "\nTotal memory to allocate: " << total_memory_bytes_ 
              << " bytes (" << (total_memory_bytes_ / (1024.0 * 1024.0)) << " MB)" << std::endl;

    // Validate the total memory calculation
    size_t expected_total = 0;
    for (const auto& [buffer_id, buf_info] : buffer_info_map) {
        size_t aligned_size = (buf_info.size_bytes + 255) & ~255;
        expected_total += aligned_size;
    }
    
    if (expected_total != total_memory_bytes_) {
        std::cerr << "ERROR: Memory calculation mismatch! Expected " 
                  << expected_total << " but got " << total_memory_bytes_ << std::endl;
    } else {
        std::cout << "Memory calculation verified: " << total_memory_bytes_ << " bytes" << std::endl;
    }

    if (total_memory_bytes_ > 0) {
        cudaError_t error = cudaMalloc(&device_memory_, total_memory_bytes_);
        if (error != cudaSuccess) {
            std::cerr << "CUDA error during memory allocation: " 
                      << cudaGetErrorString(error) << std::endl;
            throw std::runtime_error("Failed to allocate GPU memory");
        }
        std::cout << "Successfully allocated " << (total_memory_bytes_ / (1024.0 * 1024.0)) 
                  << " MB of GPU memory at " << device_memory_ << std::endl;
    } else {
        std::cout << "No memory allocation needed" << std::endl;
    }
    
    // Perform a simple sanity check on buffer offsets
    bool offsets_ok = true;
    for (const auto& [buffer_id, offset] : buffer_offsets_) {
        if (offset >= total_memory_bytes_) {
            std::cerr << "ERROR: Buffer ID " << buffer_id << " has invalid offset " 
                      << offset << " (exceeds total memory: " << total_memory_bytes_ << ")" << std::endl;
            offsets_ok = false;
        }
        
        // Check for overlapping buffers
        for (const auto& [other_id, other_offset] : buffer_offsets_) {
            if (buffer_id == other_id) continue;
            
            size_t buffer_size = (buffer_info_map[buffer_id].size_bytes + 255) & ~255;
            size_t other_size = (buffer_info_map[other_id].size_bytes + 255) & ~255;
            
            if ((offset < other_offset + other_size) && 
                (other_offset < offset + buffer_size)) {
                std::cerr << "ERROR: Buffer ID " << buffer_id << " (offset=" << offset 
                          << ", size=" << buffer_size << ") overlaps with Buffer ID " 
                          << other_id << " (offset=" << other_offset 
                          << ", size=" << other_size << ")" << std::endl;
                offsets_ok = false;
            }
        }
    }
    
    if (offsets_ok) {
        std::cout << "Buffer offset validation passed" << std::endl;
    }
    
    std::cout << "==== End of allocateBuffers() ====" << std::endl;
}


void OpCompiler::generateKernel() {
    // Create kernel generator
    /* 
    kernel_generator_ = std::make_unique<KernelGenerator>(
        OpSequence::getInstance().getOps(),
        buffer_offsets_
    );
    

    std::string kernel_code = kernel_generator_->generateCode();
    */
    
}

void OpCompiler::execute(int iterations) {
    if (!is_compiled_) {
        std::cerr << "Error: Call compile() before execute()" << std::endl;
        return;
    }
    
    // Set iteration flag for loop control
    *iteration_flag_ = iterations;
    
    // placeholders
    int block_size = 256;
    int grid_size = (total_memory_bytes_ + block_size - 1) / block_size;
    
    // Launch the kernel (do this l8ter)
    
    // Synchronize to ensure completion
    cudaDeviceSynchronize();
}

void OpCompiler::reset() {
    if (iteration_flag_) {
        cudaFreeHost(iteration_flag_);
        iteration_flag_ = nullptr;
    }
    
    if (device_memory_) {
        cudaFree(device_memory_);
        device_memory_ = nullptr;
    }
    
    //kernel_generator_.reset();
    buffer_offsets_.clear();
    total_memory_bytes_ = 0;
    is_compiled_ = false;
}

void* OpCompiler::getBuffer(size_t buffer_id) {
    if (!is_compiled_ || !device_memory_) {
        return nullptr;
    }
    
    auto it = buffer_offsets_.find(buffer_id);
    if (it == buffer_offsets_.end()) {
        return nullptr;
    }
    
    return static_cast<char*>(device_memory_) + it->second;
}

void OpCompiler::copyHostToDevice(const void* host_data, size_t buffer_id, size_t size_bytes) {
    if (!is_compiled_) {
        std::cerr << "Error: Call compile() before copying data" << std::endl;
        return;
    }
    
    auto it = buffer_offsets_.find(buffer_id);
    if (it == buffer_offsets_.end()) {
        std::cerr << "Error: Buffer ID not found" << std::endl;
        return;
    }
    
    void* device_ptr = static_cast<char*>(device_memory_) + it->second;
    cudaMemcpy(device_ptr, host_data, size_bytes, cudaMemcpyHostToDevice);
}

void OpCompiler::copyDeviceToHost(void* host_data, size_t buffer_id, size_t size_bytes) {
    if (!is_compiled_) {
        std::cerr << "Error: Call compile() before copying data" << std::endl;
        return;
    }
    
    auto it = buffer_offsets_.find(buffer_id);
    if (it == buffer_offsets_.end()) {
        std::cerr << "Error: Buffer ID not found" << std::endl;
        return;
    }
    
    void* device_ptr = static_cast<char*>(device_memory_) + it->second;
    cudaMemcpy(host_data, device_ptr, size_bytes, cudaMemcpyDeviceToHost);
}

} // namespace graphblas_gpu