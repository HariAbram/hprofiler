add_library(hprofiler_rocm SHARED rocm_hook.c)
target_compile_options(hprofiler_rocm PRIVATE -fPIC -O2 -Wall -fno-omit-frame-pointer)
target_link_libraries(hprofiler_rocm PRIVATE dl pthread hprofiler_callstack)
set_target_properties(hprofiler_rocm PROPERTIES
    OUTPUT_NAME hprofiler_rocm
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)
