file(GLOB BENCHMARK_SRCS "*.cpp" "*.cu")

foreach(benchmark_src IN LISTS BENCHMARK_SRCS)
  get_filename_component(benchmark_name ${benchmark_src} NAME_WE)

  add_executable(${benchmark_name} ${benchmark_src})
  target_link_libraries(${benchmark_name} PRIVATE graphblas_gpu graphblas_gpu_utils) # <-- ADD THIS
  target_include_directories(${benchmark_name} PRIVATE ${CMAKE_SOURCE_DIR}/utils/include) # <-- ADD THIS TOO

  set_target_properties(${benchmark_name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmarks"
  )
endforeach()
