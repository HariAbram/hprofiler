add_library(hprofiler_rocm SHARED rocm_hook.c rocprof_trace.c)
target_compile_options(hprofiler_rocm PRIVATE -fPIC -O2 -Wall -fno-omit-frame-pointer)
target_link_libraries(hprofiler_rocm PRIVATE dl pthread hprofiler_callstack)
set_target_properties(hprofiler_rocm PROPERTIES
    OUTPUT_NAME hprofiler_rocm
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# ── Native device activity (ROCprofiler-SDK) ──────────────────────────────────
# Nothing is linked: rocprof_trace.c only defines rocprofiler_configure, which
# the SDK finds in the process when the HIP runtime starts, and resolves every
# SDK function with dlsym. The SDK HEADERS are needed at build time for the
# record layouts; without them (or if they lack the types used, test-compiled
# below) the file compiles to a stub and ROCm device timing stays on the
# hipEvent proxy. Point ROCPROFILER_SDK_INCLUDE_DIR at <rocm>/include to use
# a specific SDK.
find_path(ROCPROFILER_SDK_INCLUDE_DIR rocprofiler-sdk/rocprofiler.h
    HINTS $ENV{ROCM_PATH}/include $ENV{ROCM_HOME}/include /opt/rocm/include)

if(ROCPROFILER_SDK_INCLUDE_DIR AND NOT HPROFILER_NO_ROCPROFILER)
    include(CheckCSourceCompiles)
    set(CMAKE_REQUIRED_INCLUDES "${ROCPROFILER_SDK_INCLUDE_DIR}")
    # The SDK headers include HIP's, which require a platform macro.
    set(CMAKE_REQUIRED_DEFINITIONS -D__HIP_PLATFORM_AMD__)
    check_c_source_compiles([[
        #include <rocprofiler-sdk/rocprofiler.h>
        #include <rocprofiler-sdk/registration.h>
        int main(void) {
            rocprofiler_buffer_tracing_kernel_dispatch_record_t k;
            rocprofiler_buffer_tracing_memory_copy_record_t m;
            rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t s;
            rocprofiler_tool_configure_result_t c;
            (void)k.dispatch_info.grid_size; (void)k.correlation_id.external.value;
            (void)m.bytes; (void)s.kernel_name; (void)c.initialize;
            return (int)ROCPROFILER_BUFFER_POLICY_DISCARD + (int)ROCPROFILER_BUFFER_TRACING_MEMORY_COPY;
        }]] HPROFILER_ROCPROFILER_SDK_USABLE)
    unset(CMAKE_REQUIRED_INCLUDES)
    unset(CMAKE_REQUIRED_DEFINITIONS)
endif()

if(HPROFILER_ROCPROFILER_SDK_USABLE)
    message(STATUS "hprofiler: ROCprofiler-SDK headers found (${ROCPROFILER_SDK_INCLUDE_DIR}) — "
                   "native ROCm device-activity tracing compiled in")
    set_source_files_properties(rocprof_trace.c PROPERTIES
        COMPILE_DEFINITIONS "HP_HAVE_ROCPROFILER_SDK;__HIP_PLATFORM_AMD__"
        INCLUDE_DIRECTORIES "${ROCPROFILER_SDK_INCLUDE_DIR}")
else()
    message(STATUS "hprofiler: ROCprofiler-SDK headers not found — ROCm device timing "
                   "will use the interception proxy (hipEvent pairs)")
endif()
