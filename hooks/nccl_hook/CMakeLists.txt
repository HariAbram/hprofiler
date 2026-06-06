add_library(hprofiler_nccl SHARED nccl_hook.c)
target_link_libraries(hprofiler_nccl PRIVATE pthread dl)
set_target_properties(hprofiler_nccl PROPERTIES
    OUTPUT_NAME hprofiler_nccl
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)
# NCCL headers are not required — the hook uses type stubs only.
# The library is loaded via LD_PRELOAD at runtime when libnccl.so is present.
message(STATUS "hprofiler: NCCL hook enabled (no NCCL headers required)")
