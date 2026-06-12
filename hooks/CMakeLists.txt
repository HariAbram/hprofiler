# ── Optional libunwind for accurate C++ call-stack capture ────────────────────
# Without libunwind the hooks fall back to glibc backtrace(), which requires
# the profiled binary to be built with -fno-omit-frame-pointer.
# libunwind works without frame pointers and captures more accurate stacks for
# optimised C++ code.
find_path(LIBUNWIND_INCLUDE_DIR NAMES libunwind.h
    PATHS /usr/include /usr/local/include)
find_library(LIBUNWIND_LIBRARY NAMES unwind
    PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib /usr/lib64)

add_library(hprofiler_callstack INTERFACE)

if(LIBUNWIND_INCLUDE_DIR AND LIBUNWIND_LIBRARY)
    message(STATUS "hprofiler: libunwind found (${LIBUNWIND_LIBRARY}) — "
                   "accurate C++ stack unwinding enabled")
    target_compile_definitions(hprofiler_callstack INTERFACE HPROFILER_USE_LIBUNWIND)
    target_include_directories(hprofiler_callstack INTERFACE "${LIBUNWIND_INCLUDE_DIR}")
    target_link_libraries(hprofiler_callstack INTERFACE "${LIBUNWIND_LIBRARY}")
else()
    message(STATUS "hprofiler: libunwind not found — hooks will use glibc backtrace() "
                   "(profiled binary needs -fno-omit-frame-pointer for call stacks)")
endif()

add_subdirectory(cuda_hook)
add_subdirectory(opencl_hook)
add_subdirectory(ompt_tool)
add_subdirectory(rocm_hook)
add_subdirectory(mpi_hook)
add_subdirectory(nccl_hook)
