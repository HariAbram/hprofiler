add_library(hprofiler_cuda SHARED cuda_hook.c)
target_compile_options(hprofiler_cuda PRIVATE -fPIC -O2 -Wall -fno-omit-frame-pointer)
target_link_libraries(hprofiler_cuda PRIVATE dl pthread)
set_target_properties(hprofiler_cuda PROPERTIES
    OUTPUT_NAME hprofiler_cuda
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)
