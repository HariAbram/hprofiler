add_library(hprofiler_gomp SHARED gomp_hook.c)
target_compile_options(hprofiler_gomp PRIVATE -fPIC -O2 -Wall -fno-omit-frame-pointer)
target_link_libraries(hprofiler_gomp PRIVATE dl pthread hprofiler_callstack)
set_target_properties(hprofiler_gomp PROPERTIES
    OUTPUT_NAME hprofiler_gomp
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)
