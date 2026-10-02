add_library(hprofiler_cuda SHARED cuda_hook.c cupti_trace.c)
target_compile_options(hprofiler_cuda PRIVATE -fPIC -O2 -Wall -fno-omit-frame-pointer)
target_link_libraries(hprofiler_cuda PRIVATE dl pthread hprofiler_callstack)
set_target_properties(hprofiler_cuda PROPERTIES
    OUTPUT_NAME hprofiler_cuda
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# ── Native device activity (CUPTI) ────────────────────────────────────────────
# libcupti is loaded at run time with dlopen -- never linked -- so the hook
# builds and runs without it. Its HEADERS are needed at build time for the
# activity-record layouts; cupti_trace.c compiles to a stub (proxy timing
# only, reported as "built_without_cupti_headers") when they're missing or
# don't provide the record versions it reads (test-compiled below, so an
# unfamiliar toolkit disables native tracing instead of breaking the build).
set(_hp_cuda_roots $ENV{CUDA_HOME} $ENV{CUDA_PATH} $ENV{CUDA_ROOT} /usr/local/cuda /opt/cuda)
set(_hp_cupti_hints)
foreach(_r ${_hp_cuda_roots})
    list(APPEND _hp_cupti_hints "${_r}/extras/CUPTI/include" "${_r}/include")
endforeach()
find_path(CUPTI_INCLUDE_DIR cupti.h HINTS ${_hp_cupti_hints} PATHS /usr/include)
find_path(CUDA_DRIVER_INCLUDE_DIR cuda.h HINTS ${_hp_cupti_hints} PATHS /usr/include)

if(CUPTI_INCLUDE_DIR AND CUDA_DRIVER_INCLUDE_DIR AND NOT HPROFILER_NO_CUPTI)
    include(CheckCSourceCompiles)
    set(CMAKE_REQUIRED_INCLUDES "${CUPTI_INCLUDE_DIR}" "${CUDA_DRIVER_INCLUDE_DIR}")
    check_c_source_compiles([[
        #include <cupti.h>
        int main(void) {
            CUpti_ActivityKernel9 k; CUpti_ActivityMemset4 s; CUpti_ActivityMemcpyPtoP4 p;
            CUpti_CallbackData cb;
            (void)k.queued; (void)k.submitted; (void)k.graphId;
            (void)s.correlationId; (void)p.srcDeviceId; (void)cb.correlationData;
            return (int)CUPTI_API_VERSION + (int)CUPTI_ACTIVITY_KIND_MEMCPY2;
        }]] HPROFILER_CUPTI_HEADERS_USABLE)
    # The record version CUPTI delivers follows the toolkit: newer headers
    # document Memcpy6 / Synchronization2 (the older structs plus appended
    # fields). Decode with the newest one these headers have.
    check_c_source_compiles([[
        #include <cupti.h>
        int main(void) { CUpti_ActivityMemcpy6 m; (void)m.runtimeCorrelationId; return 0; }
        ]] HPROFILER_CUPTI_MEMCPY6)
    check_c_source_compiles([[
        #include <cupti.h>
        int main(void) { CUpti_ActivityMemcpy5 m; (void)m.runtimeCorrelationId; return 0; }
        ]] HPROFILER_CUPTI_MEMCPY5)
    check_c_source_compiles([[
        #include <cupti.h>
        int main(void) { CUpti_ActivitySynchronization2 y; (void)y.cudaEventId; return 0; }
        ]] HPROFILER_CUPTI_SYNC2)
    check_c_source_compiles([[
        #include <cupti.h>
        int main(void) { CUpti_ActivitySynchronization y; (void)y.cudaEventId; return 0; }
        ]] HPROFILER_CUPTI_SYNC1)
    unset(CMAKE_REQUIRED_INCLUDES)
    if(NOT (HPROFILER_CUPTI_MEMCPY6 OR HPROFILER_CUPTI_MEMCPY5) OR
       NOT (HPROFILER_CUPTI_SYNC2 OR HPROFILER_CUPTI_SYNC1))
        set(HPROFILER_CUPTI_HEADERS_USABLE OFF)
    endif()
endif()

if(HPROFILER_CUPTI_HEADERS_USABLE)
    message(STATUS "hprofiler: CUPTI headers found (${CUPTI_INCLUDE_DIR}) — "
                   "native CUDA device-activity tracing compiled in (libcupti loaded at run time)")
    set(_hp_cupti_defs HP_HAVE_CUPTI)
    if(HPROFILER_CUPTI_MEMCPY6)
        list(APPEND _hp_cupti_defs HP_CUPTI_MEMCPY6)
    endif()
    if(HPROFILER_CUPTI_SYNC2)
        list(APPEND _hp_cupti_defs HP_CUPTI_SYNC2)
    endif()
    # Prefer the libcupti shipped with these headers at run time (another,
    # older libcupti may come first on the loader path).
    get_filename_component(_hp_cupti_parent "${CUPTI_INCLUDE_DIR}" DIRECTORY)
    list(APPEND _hp_cupti_defs "HP_CUPTI_LIB_DIR=\"${_hp_cupti_parent}/lib64\"")
    set_source_files_properties(cupti_trace.c PROPERTIES
        COMPILE_DEFINITIONS "${_hp_cupti_defs}"
        INCLUDE_DIRECTORIES "${CUPTI_INCLUDE_DIR};${CUDA_DRIVER_INCLUDE_DIR}")
else()
    message(STATUS "hprofiler: usable CUPTI headers not found — CUDA device timing "
                   "will use the interception proxy (cudaEvent pairs)")
endif()
