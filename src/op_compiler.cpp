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

    // Track buffer metadata (rows, cols, vector size)
    struct BufferInfo {
        size_t rows = 0;
        size_t cols = 0;
        size_t nnz = 0;
        size_t size_bytes = 0;
        std::string datatype;
    };

    std::unordered_map<size_t, BufferInfo> buffer_info_map;

    auto datatype_size = [](const std::string& datatype) -> size_t {
        if (datatype == "float") return sizeof(float);
        if (datatype == "double") return sizeof(double);
        if (datatype == "int") return sizeof(int);
        if (datatype == "bool") return sizeof(bool);
        throw std::runtime_error("Unsupported datatype: " + datatype);
    };

    for (const auto& op : ops) {
        switch (op.type) {
            case Op::Type::AllocVector: {
                size_t buffer_id = op.buffer_ids[0];
                size_t vec_size = std::stoul(op.args.at("size"));
                std::string datatype = op.args.at("datatype");
                size_t buffer_size = vec_size * datatype_size(datatype);

                buffer_info_map[buffer_id] = {
                    .rows = vec_size,
                    .cols = 1,
                    .size_bytes = buffer_size,
                    .datatype = datatype
                };
                break;
            }
            case Op::Type::AllocGraph: {
                size_t buffer_id = op.buffer_ids[0];
                size_t rows = std::stoul(op.args.at("rows"));
                size_t cols = std::stoul(op.args.at("cols"));
                size_t nnz = std::stoul(op.args.at("nnz"));
                std::string datatype = op.args.at("datatype");

                size_t buffer_size =
                    (rows + 1) * sizeof(size_t) +  // row_offsets
                    nnz * sizeof(size_t) +         // col_indices
                    nnz * datatype_size(datatype); // values

                buffer_info_map[buffer_id] = {
                    .rows = rows,
                    .cols = cols,
                    .nnz = nnz,
                    .size_bytes = buffer_size,
                    .datatype = datatype
                };
                break;
            }
            case Op::Type::SpMV: {
                size_t result_buffer_id = op.buffer_ids[0];
                size_t mat_buffer_id = op.buffer_ids[1];
                std::string datatype = op.args.at("datatype");

                const auto& mat_info = buffer_info_map.at(mat_buffer_id);
                size_t result_vec_size = mat_info.rows;  // m × 1 result
                size_t buffer_size = result_vec_size * datatype_size(datatype);

                buffer_info_map[result_buffer_id] = {
                    .rows = result_vec_size,
                    .cols = 1,
                    .size_bytes = buffer_size,
                    .datatype = datatype
                };
                break;
            }
            case Op::Type::EWiseAdd:
            case Op::Type::EWiseSub:
            case Op::Type::EWiseMul:
            case Op::Type::EWiseDiv: {
                size_t result_buffer_id = op.buffer_ids[0];
                size_t lhs_buffer_id = op.buffer_ids[1];
                std::string datatype = op.args.at("datatype");

                const auto& lhs_info = buffer_info_map.at(lhs_buffer_id);
                size_t result_size = lhs_info.rows * lhs_info.cols; // same as lhs
                size_t buffer_size = result_size * datatype_size(datatype);

                buffer_info_map[result_buffer_id] = {
                    .rows = lhs_info.rows,
                    .cols = lhs_info.cols,
                    .size_bytes = buffer_size,
                    .datatype = datatype
                };
                break;
            }
            default:
                throw std::runtime_error("Operation not supported yet");
        }
    }

    // Now assign offsets and total size
    for (const auto& [buffer_id, buf_info] : buffer_info_map) {
        size_t aligned_size = (buf_info.size_bytes + 255) & ~255;
        buffer_offsets_[buffer_id] = total_memory_bytes_;
        total_memory_bytes_ += aligned_size;
    }

    if (total_memory_bytes_ > 0) {
        cudaMalloc(&device_memory_, total_memory_bytes_);
    }
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