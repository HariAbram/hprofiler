add_library(hprofiler_opencl SHARED opencl_hook.c)
target_compile_options(hprofiler_opencl PRIVATE -fPIC -O2 -Wall -fno-omit-frame-pointer)
target_link_libraries(hprofiler_opencl PRIVATE dl pthread hprofiler_callstack)
set_target_properties(hprofiler_opencl PROPERTIES
    OUTPUT_NAME hprofiler_opencl
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)
