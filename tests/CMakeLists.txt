file(GLOB TEST_SRCS "*.cpp")

foreach(test_src IN LISTS TEST_SRCS)
  
  get_filename_component(test_name ${test_src} NAME_WE)

  add_executable(${test_name} ${test_src})
  target_link_libraries(${test_name} PRIVATE graphblas_gpu)

  set_target_properties(${test_name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
  )
endforeach()
