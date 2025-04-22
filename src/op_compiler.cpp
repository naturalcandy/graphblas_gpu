#include <graphblas_gpu/op_compiler.hpp>
#include <graphblas_gpu/kernel_generator.hpp>
#include <cuda_runtime.h>
#include <cuda.h>
#include <nvrtc.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <unistd.h>
#include <iostream>
#include <algorithm>

namespace graphblas_gpu {

OpCompiler& OpCompiler::getInstance() {
    static OpCompiler instance;
    return instance;
}

OpCompiler::OpCompiler() 
    : is_compiled_(false),
      kernel_loaded_(false),
      total_memory_bytes_(0), 
      device_memory_(nullptr), 
      kernel_function_(nullptr),
      iteration_flag_(nullptr) {
    
    cuInit(0);
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
        DataType datatype = DataType(DataTypeEnum::Unknown);
        std::string origin; // Which operation created this buffer
    };

    std::unordered_map<size_t, BufferInfo> buffer_info_map;

    // First pass: analyze operations and determine buffer requirements
    for (size_t i = 0; i < ops.size(); i++) {
        const auto& op = ops[i];

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

                // Check if the input buffers exist and get their info
                if (buffer_info_map.find(mat_buffer_id) == buffer_info_map.end() ||
                    buffer_info_map.find(vec_buffer_id) == buffer_info_map.end()) {
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
                
                size_t result_vec_size = mat_info.rows;  // m × 1 result
                size_t buffer_size = result_vec_size * datatype.sizeInBytes();

                buffer_info_map[result_buffer_id] = {
                    .rows = result_vec_size,
                    .cols = 1,
                    .size_bytes = buffer_size,
                    .datatype = datatype,
                    .origin = "SpMV"
                };
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
                if (buffer_info_map.find(lhs_buffer_id) == buffer_info_map.end() ||
                    buffer_info_map.find(rhs_buffer_id) == buffer_info_map.end()) {
                    continue;
                }

                const auto& lhs_info = buffer_info_map.at(lhs_buffer_id);
                const auto& rhs_info = buffer_info_map.at(rhs_buffer_id);
                
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
                
                break;
            }
            default:
                std::cerr << "  WARNING: Operation type " << static_cast<int>(op.type) 
                          << " not supported yet" << std::endl;
        }
    }
    std::vector<size_t> buffer_ids;
    for (const auto& [buffer_id, _] : buffer_info_map) {
        buffer_ids.push_back(buffer_id);
    }
    std::sort(buffer_ids.begin(), buffer_ids.end()); 
    // Now we assign offsets and total buffer size to allocate on device
    // should we allocate buffers that end up not being used in computation
    // or is that user's responsibility..
    for (size_t buffer_id : buffer_ids) {
        const auto& buf_info = buffer_info_map[buffer_id];
        size_t aligned_size = (buf_info.size_bytes + 255) & ~255; 
        buffer_offsets_[buffer_id] = total_memory_bytes_;
        total_memory_bytes_ += aligned_size;

        std::cout << "BUFFER ID: " << buffer_id << ", OFFSET: " << buffer_offsets_[buffer_id] << std::endl;
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
                  
        // Initialize all memory to zero
        cudaMemset(device_memory_, 0, total_memory_bytes_);
    } else {
        std::cout << "No memory allocation needed" << std::endl;
    }
}

void OpCompiler::generateKernel() {
    // Create kernel generator with operations and buffer offsets
    KernelGenerator generator(OpSequence::getInstance().getOps(), buffer_offsets_);
    
    // Generate CUDA source code
    kernel_code_ = generator.generateCode();
    kernel_name_ = generator.getKernelName();
    
    
    // Compile the kernel
    kernel_loaded_ = compileAndLoadKernel(kernel_code_);
    
    if (!kernel_loaded_) {
        std::cerr << "ERROR: Failed to compile and load kernel" << std::endl;
    }
}

bool OpCompiler::compileAndLoadKernel(const std::string& kernel_code) {
    // Generate a unique filename based on a hash of the kernel code
    std::string hash = std::to_string(std::hash<std::string>{}(kernel_code));
    std::string temp_dir = "/tmp";
    std::string code_file_path = temp_dir + "/graphblas_kernel_" + hash + ".cu";
    std::string bin_file_path = temp_dir + "/graphblas_kernel_" + hash + ".cubin";
    
    // Check if we already have a cached binary for this kernel
    if (is_file_exists(bin_file_path)) {
        std::cout << "Using cached binary for kernel" << std::endl;
    } else {
        // Write the kernel code to a file
        std::ofstream code_file(code_file_path, std::ios::out | std::ios::trunc);
        if (!code_file.is_open()) {
            std::cerr << "ERROR: Failed to open file for writing: " << code_file_path << std::endl;
            return false;
        }
        code_file << kernel_code;
        code_file.close();
        
        // Build the compilation command
        std::stringstream compile_cmd;
        compile_cmd << "nvcc -cubin";
        
        // Add include paths
        compile_cmd << " -I/usr/local/cuda/include";
        compile_cmd << " -I" << PROJECT_SOURCE_DIR << "/include";
        compile_cmd << " -I" << PROJECT_SOURCE_DIR << "/kernels";
        
        // Add architecture
        compile_cmd << " -gencode arch=compute_75,code=sm_75";
        
        // Add input and output files
        compile_cmd << " -o " << bin_file_path;
        compile_cmd << " " << code_file_path;
        compile_cmd << " 2>&1";
        
        std::cout << "Compiling kernel with command: " << compile_cmd.str() << std::endl;
        
        // Execute the compilation command
        FILE* pipe = popen(compile_cmd.str().c_str(), "r");
        if (!pipe) {
            std::cerr << "ERROR: Failed to execute compilation command" << std::endl;
            return false;
        }
        
        // Read and display any compiler output
        char buffer[4096];
        std::string output;
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            output += buffer;
        }
        
        int status = pclose(pipe);
        if (status != 0) {
            std::cerr << "Compilation failed:\n" << output << std::endl;
            return false;
        }
        
        std::cout << "Kernel compilation successful" << std::endl;
    }
    
    // Load the compiled binary
    std::ifstream bin_file(bin_file_path, std::ios::binary | std::ios::ate);
    if (!bin_file.is_open()) {
        std::cerr << "ERROR: Failed to open binary file: " << bin_file_path << std::endl;
        return false;
    }
    
    size_t size = bin_file.tellg();
    bin_file.seekg(0, std::ios::beg);
    std::vector<char> binary(size);
    if (!bin_file.read(binary.data(), size)) {
        std::cerr << "ERROR: Failed to read binary file" << std::endl;
        return false;
    }
    bin_file.close();
    
    // Load the module
    CUresult result = cuModuleLoadData(&cuModule_, binary.data());
    if (result != CUDA_SUCCESS) {
        const char* error_str;
        cuGetErrorString(result, &error_str);
        std::cerr << "ERROR: Failed to load module: " << error_str << std::endl;
        return false;
    }
    
    // Get the kernel function
    result = cuModuleGetFunction(&kernel_function_, cuModule_, kernel_name_.c_str());
    if (result != CUDA_SUCCESS) {
        const char* error_str;
        cuGetErrorString(result, &error_str);
        std::cerr << "ERROR: Failed to get kernel function: " << error_str << std::endl;
        cuModuleUnload(cuModule_);
        return false;
    }
    
    return true;
}

bool OpCompiler::is_file_exists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good();
}

void OpCompiler::execute(int iterations) {
    if (!is_compiled_) {
        std::cerr << "Error: Call compile() before execute()" << std::endl;
        return;
    }
    
    if (!kernel_loaded_) {
        std::cerr << "Error: Kernel not loaded" << std::endl;
        return;
    }
    
    // Set iteration flag for loop control
    *iteration_flag_ = iterations;
    
    // Launch parameters
    int block_size = 256;
    int grid_size = std::min(65535, static_cast<int>((total_memory_bytes_ + block_size - 1) / block_size));
    
    // If grid_size is too small, make it at least 1
    if (grid_size < 1) grid_size = 1;
    
    std::cout << "Launching kernel with grid_size=" << grid_size 
              << ", block_size=" << block_size << std::endl;
    
    // Set up kernel arguments
    void* args[] = {&device_memory_, iteration_flag_};

    /* 
        
    // cooperative kernel has strong limitation on max threads..
    // perahps we can case on whether or not we do coopeartive
    // kernel launch or regular launch with our own custom grid sync.
    CUresult result = cuLaunchKernel(
        kernel_function_,
        grid_size, 1, 1,             // Grid dimensions
        block_size, 1, 1,            // Block dimensions
        0,                           // Shared mem size
        nullptr,                     // Stream
        args,                        // Args
        nullptr                      
    );
    
    */
    
    // Launch the kernel
    CUresult result = cuLaunchCooperativeKernel(
        kernel_function_,
        grid_size, 1, 1,             // Grid dimensions
        block_size, 1, 1,            // Block dimensions
        0,                           // Shared mem size
        nullptr,                     // Stream
        args                         // Args                    
    );
    
    if (result != CUDA_SUCCESS) {
        const char* error_str;
        cuGetErrorString(result, &error_str);
        std::cerr << "Error launching kernel: " << error_str << std::endl;
        return;
    }
    
    // Synchronize to ensure completion
    result = cuCtxSynchronize();
    if (result != CUDA_SUCCESS) {
        const char* error_str;
        cuGetErrorString(result, &error_str);
        std::cerr << "Error synchronizing context: " << error_str << std::endl;
    }
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
    
    if (kernel_loaded_) {
        cuModuleUnload(cuModule_);
        kernel_loaded_ = false;
        kernel_function_ = nullptr;
    }
    
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