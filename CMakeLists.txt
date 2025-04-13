cmake_minimum_required(VERSION 3.22)

# Project setup explicitly targeting C++ and CUDA
project(graphblas_gpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# RTX 2080 GPU (Compute Capability 7.5)
set(CMAKE_CUDA_ARCHITECTURES 75)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(GRAPHBLAS_CPU_SRC
    src/op_sequence.cpp
    src/semiring.cpp
    src/operation.cpp
    src/executor.cpp
)

set(GRAPHBLAS_GPU_SRC
    kernels/spmv.cu
    kernels/spmm.cu
    kernels/kernel_generator.cpp
)

add_library(graphblas_gpu STATIC ${GRAPHBLAS_CPU_SRC} ${GRAPHBLAS_GPU_SRC})

target_compile_options(graphblas_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>
)

target_compile_options(graphblas_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
)

set_target_properties(graphblas_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

install(DIRECTORY include/ DESTINATION include)
install(TARGETS graphblas_gpu DESTINATION lib)

add_executable(test_graphblas test_main.cpp)
target_link_libraries(test_graphblas graphblas_gpu)
