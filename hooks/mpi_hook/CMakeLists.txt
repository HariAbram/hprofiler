include(CheckCSourceCompiles)

set(_HPROFILER_MPI_FOUND FALSE)
set(_HPROFILER_MPI_CRAY_WRAPPER FALSE)

# On Cray systems (e.g. Dardel) MPI headers/libs are supplied automatically
# by the Cray compiler wrapper (`cc`) — there's often no mpicc and mpi.h
# doesn't live under /usr/include, so the plain find_path() below can't see
# it. Detect via $CRAY_MPICH_DIR and confirm the current compiler can
# already compile MPI code with no extra -I/-L before trusting it.
if(DEFINED ENV{CRAY_MPICH_DIR})
    message(STATUS "hprofiler: Cray MPICH detected: $ENV{CRAY_MPICH_DIR}")
    check_c_source_compiles("
        #include <mpi.h>
        int main(void) {
            int argc = 0; char **argv = 0, rank = 0;
            MPI_Init(&argc, &argv);
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Finalize();
            return rank;
        }
    " _HPROFILER_CRAY_MPI_WORKS)

    if(_HPROFILER_CRAY_MPI_WORKS)
        set(_HPROFILER_MPI_FOUND TRUE)
        set(_HPROFILER_MPI_CRAY_WRAPPER TRUE)
        message(STATUS "hprofiler: MPI available through Cray compiler wrapper")
    else()
        message(STATUS "hprofiler: Cray MPICH loaded, but ${CMAKE_C_COMPILER} "
                        "cannot compile MPI code directly — configure with CC=cc")
    endif()
endif()

# Conventional MPI installations (Ubuntu/Fedora/OpenMPI/MPICH/containers).
# Avoids calling FindMPI.cmake, which fails hard on partially-installed MPI
# packages (missing include path in cmake cache) — find_path/find_library
# degrade gracefully instead. $MPI_HOME/$MPI_ROOT/$CRAY_MPICH_DIR are
# consulted as hints for non-standard installs (e.g. cluster modules).
if(NOT _HPROFILER_MPI_FOUND)
    find_path(_MPI_INCLUDE mpi.h
        HINTS
            "$ENV{MPI_HOME}"
            "$ENV{MPI_ROOT}"
            "$ENV{CRAY_MPICH_DIR}"
        PATH_SUFFIXES
            include
            include/mpi
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
    )

    if(_MPI_INCLUDE)
        find_library(_MPI_LIB NAMES mpi mpich mpi_cxx
            HINTS
                "$ENV{MPI_HOME}"
                "$ENV{MPI_ROOT}"
                "$ENV{CRAY_MPICH_DIR}"
            PATH_SUFFIXES
                lib
                lib64
            PATHS
                /usr/lib
                /usr/lib/x86_64-linux-gnu
                /usr/lib64
                /usr/local/lib
                /opt/openmpi/lib
                /opt/mpich/lib
        )
        set(_HPROFILER_MPI_FOUND TRUE)
    endif()
endif()

if(_HPROFILER_MPI_FOUND)
    add_library(hprofiler_mpi SHARED mpi_hook.c)
    target_compile_options(hprofiler_mpi PRIVATE -fPIC -O2 -Wall -fno-omit-frame-pointer)

    if(_HPROFILER_MPI_CRAY_WRAPPER)
        # cc supplies mpi.h and links Cray MPICH automatically — no manual
        # -I$CRAY_MPICH_DIR or -lmpi needed (and adding them can conflict
        # with the wrapper's own flags).
        target_link_libraries(hprofiler_mpi PRIVATE pthread dl hprofiler_callstack)
    else()
        target_include_directories(hprofiler_mpi PRIVATE "${_MPI_INCLUDE}")
        if(_MPI_LIB)
            target_link_libraries(hprofiler_mpi PRIVATE "${_MPI_LIB}" pthread dl hprofiler_callstack)
        else()
            target_link_libraries(hprofiler_mpi PRIVATE pthread dl hprofiler_callstack)
        endif()
    endif()

    set_target_properties(hprofiler_mpi PROPERTIES
        OUTPUT_NAME hprofiler_mpi
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    )

    if(_HPROFILER_MPI_CRAY_WRAPPER)
        message(STATUS "hprofiler: MPI hook enabled (Cray compiler-wrapper MPI)")
    else()
        message(STATUS "hprofiler: MPI hook enabled (headers: ${_MPI_INCLUDE})")
    endif()
else()
    message(STATUS "hprofiler: MPI headers not found — MPI hook skipped")
endif()
