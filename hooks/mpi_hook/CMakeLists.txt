# Locate mpi.h manually — avoids calling FindMPI.cmake which fails hard on
# partially-installed MPI packages (missing include path in cmake cache).
find_path(_MPI_INCLUDE mpi.h
    PATHS
        /usr/include
        /usr/include/mpi
        /usr/include/mpich
        /usr/include/openmpi
        /usr/local/include
        /opt/openmpi/include
        /opt/mpich/include
        /usr/lib/x86_64-linux-gnu/mpich/include
        /usr/lib/x86_64-linux-gnu/openmpi/include
    NO_CMAKE_SYSTEM_PATH
)

if(_MPI_INCLUDE)
    # Find the matching MPI library
    find_library(_MPI_LIB NAMES mpi mpich mpi_cxx
        PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib /opt/openmpi/lib
        NO_CMAKE_SYSTEM_PATH
    )
    add_library(hprofiler_mpi SHARED mpi_hook.c)
    target_compile_options(hprofiler_mpi PRIVATE -fPIC -O2 -Wall -fno-omit-frame-pointer)
    target_include_directories(hprofiler_mpi PRIVATE "${_MPI_INCLUDE}")
    if(_MPI_LIB)
        target_link_libraries(hprofiler_mpi PRIVATE "${_MPI_LIB}" pthread dl)
    else()
        target_link_libraries(hprofiler_mpi PRIVATE pthread dl)
    endif()
    set_target_properties(hprofiler_mpi PROPERTIES
        OUTPUT_NAME hprofiler_mpi
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    )
    message(STATUS "hprofiler: MPI hook enabled (headers: ${_MPI_INCLUDE})")
else()
    message(STATUS "hprofiler: MPI headers not found — MPI hook skipped")
endif()
