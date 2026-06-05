cmake_minimum_required(VERSION 3.16)
project(hprofiler C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Installation prefix for hook libs
set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}" CACHE PATH "Install prefix")

# Output all libraries to build/lib so the Python code can find them
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_subdirectory(hooks)

# ── Install hook libs into build/lib so Python runner can locate them ─────────
install(DIRECTORY "${CMAKE_BINARY_DIR}/lib/"
        DESTINATION "${CMAKE_SOURCE_DIR}/build/lib"
        FILES_MATCHING PATTERN "*.so*"
)
