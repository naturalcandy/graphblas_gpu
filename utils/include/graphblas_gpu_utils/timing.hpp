// utils/include/graphblas_gpu_utils/timing.hpp
#pragma once

#include <chrono>
#include <string>
#include <vector>
#include <cuda_runtime.h>

namespace graphblas_gpu {
namespace utils {

// CUDA event-based timer
class CudaTimer {
public:
    CudaTimer();
    ~CudaTimer();
    
    void start();
    void stop();
    float elapsed_ms() const;
    
private:
    cudaEvent_t start_event_;
    cudaEvent_t stop_event_;
    bool timing_active_;
};

// CPU high-resolution timer
class CpuTimer {
public:
    void start();
    void stop();
    double elapsed_ms() const;
    
private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point stop_time_;
    bool timing_active_ = false;
};

// Records benchmark results
struct BenchmarkResult {
    std::string name;
    std::vector<double> times_ms;
    double min_ms;
    double max_ms;
    double avg_ms;
    
    BenchmarkResult(const std::string& name);
    void add_time(double time_ms);
};

} // namespace utils
} // namespace graphblas_gpu