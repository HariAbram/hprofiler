include(CheckCSourceCompiles)

set(_HPROFILER_MPI_FOUND FALSE)
set(_HPROFILER_MPI_CRAY_WRAPPER FALSE)

# On Cray systems (e.g. Dardel) MPI headers/libs are supplied automatically
# by the Cray compiler wrapper (`cc`) — there's often no mpicc and mpi.h
# doesn't live under /usr/include, so the plain find_path() below can't see
# it. Detect via $CRAY_MPICH_DIR and confirm the current compiler can
# already compile MPI code with no extra -I/-L before trusting it. Inside a
# container the host's $CRAY_MPICH_DIR is often inherited while the Cray PE
# itself is not mounted, so the variable only counts if the directory exists.
if(DEFINED ENV{CRAY_MPICH_DIR} AND NOT IS_DIRECTORY "$ENV{CRAY_MPICH_DIR}")
    message(STATUS "hprofiler: ignoring CRAY_MPICH_DIR=$ENV{CRAY_MPICH_DIR} "
                   "(not present here, e.g. a host variable inside a container)")
elseif(DEFINED ENV{CRAY_MPICH_DIR})
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
                        "cannot compile MPI code directly (configure with CC=cc to use "
                        "Cray MPICH) — looking for another MPI installation")
    endif()
endif()

# An MPI compiler wrapper knows where its own mpi.h and libmpi live. Asking
# it finds module/Spack installs and distro layouts whose headers are not
# under the prefix (Fedora/RHEL Open MPI: /usr/include/openmpi-x86_64 with
# mpicc in /usr/lib64/openmpi/bin), and keeps header and library from the
# same MPI. Open MPI answers --showme:incdirs/--showme:libdirs; MPICH-derived
# wrappers (MPICH, Intel MPI, MVAPICH) reject that and answer -show instead.
set(_MPI_WRAPPER_INC "")
set(_MPI_WRAPPER_LIB "")
if(NOT _HPROFILER_MPI_FOUND)
    find_program(_HPROFILER_MPICC NAMES mpicc
        HINTS
            "$ENV{MPI_HOME}/bin"
            "$ENV{MPI_ROOT}/bin"
        PATHS
            /usr/lib64/openmpi/bin
            /usr/lib64/mpich/bin
    )
    if(_HPROFILER_MPICC)
        execute_process(COMMAND "${_HPROFILER_MPICC}" --showme:incdirs
            OUTPUT_VARIABLE _out RESULT_VARIABLE _rc ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE TIMEOUT 20)
        if(_rc EQUAL 0)
            separate_arguments(_MPI_WRAPPER_INC UNIX_COMMAND "${_out}")
            execute_process(COMMAND "${_HPROFILER_MPICC}" --showme:libdirs
                OUTPUT_VARIABLE _out RESULT_VARIABLE _rc ERROR_QUIET
                OUTPUT_STRIP_TRAILING_WHITESPACE TIMEOUT 20)
            if(_rc EQUAL 0)
                separate_arguments(_MPI_WRAPPER_LIB UNIX_COMMAND "${_out}")
            endif()
        else()
            execute_process(COMMAND "${_HPROFILER_MPICC}" -show
                OUTPUT_VARIABLE _out RESULT_VARIABLE _rc ERROR_QUIET
                OUTPUT_STRIP_TRAILING_WHITESPACE TIMEOUT 20)
            if(_rc EQUAL 0)
                separate_arguments(_tokens UNIX_COMMAND "${_out}")
                foreach(_tok IN LISTS _tokens)
                    if(_tok MATCHES "^-I(.+)")
                        list(APPEND _MPI_WRAPPER_INC "${CMAKE_MATCH_1}")
                    elseif(_tok MATCHES "^-L(.+)")
                        list(APPEND _MPI_WRAPPER_LIB "${CMAKE_MATCH_1}")
                    endif()
                endforeach()
            endif()
        endif()
        message(STATUS "hprofiler: MPI compiler wrapper ${_HPROFILER_MPICC} "
                       "(include dirs: ${_MPI_WRAPPER_INC})")
    endif()
endif()

# Conventional MPI installations (Ubuntu/Fedora/OpenMPI/MPICH/containers).
# Avoids calling FindMPI.cmake, which fails hard on partially-installed MPI
# packages (missing include path in cmake cache) — find_path/find_library
# degrade gracefully instead. $MPI_HOME/$MPI_ROOT, the wrapper's own
# directories and $CRAY_MPICH_DIR are consulted first, in that order.
if(NOT _HPROFILER_MPI_FOUND)
    find_path(_MPI_INCLUDE mpi.h
        HINTS
            "$ENV{MPI_HOME}"
            "$ENV{MPI_ROOT}"
            ${_MPI_WRAPPER_INC}
            "$ENV{CRAY_MPICH_DIR}"
        PATH_SUFFIXES
            include
            include/mpi
        PATHS
            /usr/include
            /usr/include/mpi
            /usr/include/mpich
            /usr/include/openmpi
            /usr/include/openmpi-x86_64
            /usr/include/openmpi-aarch64
            /usr/include/mpich-x86_64
            /usr/include/mpich-aarch64
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
                ${_MPI_WRAPPER_LIB}
                "$ENV{CRAY_MPICH_DIR}"
            PATH_SUFFIXES
                lib
                lib64
            PATHS
                /usr/lib
                /usr/lib/x86_64-linux-gnu
                /usr/lib64
                /usr/lib64/openmpi/lib
                /usr/lib64/mpich/lib
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
