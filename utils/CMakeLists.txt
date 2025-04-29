set(UTILS_SOURCES
    src/matrix_generators.cpp
)

add_library(graphblas_gpu_utils ${UTILS_SOURCES})
target_include_directories(graphblas_gpu_utils
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link with CUDA if needed for timing utilities
target_link_libraries(graphblas_gpu_utils
    PUBLIC
        CUDA::cudart
)

# Set include path for dependent libraries
target_include_directories(graphblas_gpu_utils
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)