add_library(hprofiler_opencl SHARED opencl_hook.c)
target_compile_options(hprofiler_opencl PRIVATE -fPIC -O2 -Wall)
target_link_libraries(hprofiler_opencl PRIVATE dl pthread)
set_target_properties(hprofiler_opencl PROPERTIES
    OUTPUT_NAME hprofiler_opencl
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)
