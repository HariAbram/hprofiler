add_library(hprofiler_ompt SHARED ompt_tool.c)
target_compile_options(hprofiler_ompt PRIVATE -fPIC -O2 -Wall -fno-omit-frame-pointer)
target_link_libraries(hprofiler_ompt PRIVATE pthread)
set_target_properties(hprofiler_ompt PROPERTIES
    OUTPUT_NAME hprofiler_ompt
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)
