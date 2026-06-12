add_library(hprofiler_cuda SHARED cuda_hook.c)
target_compile_options(hprofiler_cuda PRIVATE -fPIC -O2 -Wall -fno-omit-frame-pointer)
target_link_libraries(hprofiler_cuda PRIVATE dl pthread hprofiler_callstack)
set_target_properties(hprofiler_cuda PROPERTIES
    OUTPUT_NAME hprofiler_cuda
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)
# CUPTI PC sampling is loaded at runtime via dlopen — no compile-time linkage needed.
# The code is always compiled in; it silently does nothing if libcupti.so is absent.
