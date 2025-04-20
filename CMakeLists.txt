cmake_minimum_required(VERSION 3.22)

project(graphblas_gpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_definitions(-DPROJECT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

# RTX 2080 GPU (Compute Capability 7.5)
set(CMAKE_CUDA_ARCHITECTURES 75)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

find_library(CUDA_DRIVER_LIB cuda PATHS ${CUDAToolkit_LIBRARY_DIR})

set(GRAPHBLAS_CPU_SRC
    src/op_sequence.cpp
    src/op_compiler.cpp
    src/kernel_generator.cpp
)

set(GRAPHBLAS_GPU_SRC
    kernels/spmv.cu
    kernels/spmm.cu
    kernels/ewise_ops.cu 
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
    CUDA_RESOLVE_DEVICE_SYMBOLS ON 
)

target_include_directories(graphblas_gpu PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(graphblas_gpu ${CUDA_DRIVER_LIB})

install(DIRECTORY include/ DESTINATION include)
install(TARGETS graphblas_gpu DESTINATION lib)

add_subdirectory(tests)