cmake_minimum_required(VERSION 3.20)
project(Test2 VERSION 1.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# cmake/ lives two levels above src/Test2/ at the repo root
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/../../cmake)

# Find the vector library
find_package(Vector REQUIRED)

# Find MPI (required for MATAR's MPICArrayKokkos / CommunicationPlan types,
# which are compiled out entirely unless HAVE_MPI is defined below)
find_package(MPI REQUIRED)

# Set the compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${VECTOR_CXX_FLAGS}")


# --- Set build type if not already set ---
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
 # set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)


# --- Options ---
option(Test2_ENABLE_SERIAL "Enable Serial support in Kokkos" ON)
option(Test2_ENABLE_OPENMP "Enable OpenMP support in Kokkos" OFF)
option(Test2_ENABLE_PTHREADS "Enable PThreads support in Kokkos" OFF)
option(Test2_ENABLE_CUDA "Enable CUDA support in Kokkos" OFF)
option(Test2_ENABLE_HIP "Enable HIP support in Kokkos" OFF)

include(FetchContent)

# --- Dependency: Kokkos ---
if(NOT TARGET Kokkos::kokkos)
    set(Kokkos_ENABLE_SERIAL   ${Test2_ENABLE_SERIAL}   CACHE BOOL "" FORCE)
    set(Kokkos_ENABLE_OPENMP   ${Test2_ENABLE_OPENMP}   CACHE BOOL "" FORCE)
    set(Kokkos_ENABLE_PTHREADS ${Test2_ENABLE_PTHREADS} CACHE BOOL "" FORCE)
    set(Kokkos_ENABLE_CUDA     ${Test2_ENABLE_CUDA}     CACHE BOOL "" FORCE)
    set(Kokkos_ENABLE_HIP      ${Test2_ENABLE_HIP}      CACHE BOOL "" FORCE)
    
    FetchContent_Declare(
        Kokkos
        GIT_REPOSITORY https://github.com/kokkos/kokkos
        GIT_TAG        4.3.00
    )
    FetchContent_MakeAvailable(Kokkos)
endif()


# --- Dependency: MATAR ---
if(NOT TARGET matar::matar)
    FetchContent_Declare(
        MATAR
        GIT_REPOSITORY https://github.com/lanl/MATAR
        GIT_TAG        main
    )
    FetchContent_MakeAvailable(MATAR)

    if(TARGET matar)
        if(NOT TARGET matar::matar)
            add_library(matar::matar ALIAS matar)
        endif()
        
        FetchContent_GetProperties(MATAR SOURCE_DIR MATAR_SOURCE_DIR)
        get_target_property(_matar_includes matar INTERFACE_INCLUDE_DIRECTORIES)
        
        if(NOT "${MATAR_SOURCE_DIR}/src/include" IN_LIST _matar_includes)
            target_include_directories(matar INTERFACE 
                $<BUILD_INTERFACE:${MATAR_SOURCE_DIR}/src/include>
                $<BUILD_INTERFACE:${MATAR_SOURCE_DIR}/solvers>
            )
        endif()
        
        target_compile_definitions(matar INTERFACE HAVE_KOKKOS)
        if(MPI_FOUND)
            target_compile_definitions(matar INTERFACE HAVE_MPI)
        endif()
    endif()
endif()

# --- Dependency: ELEMENTS ---
# Declared after Kokkos/MATAR above so ELEMENTS's own FetchContent guards
# (if(NOT TARGET Kokkos::kokkos) / if(NOT TARGET matar::matar) in its
# top-level CMakeLists.txt) see those targets already exist and reuse them
# instead of fetching a second copy with different options.
#
# ELEMENTS's src/CMakeLists.txt defines an INTERFACE target "ELEMENTS" that
# already links Kokkos::kokkos, matar::matar, MPI::MPI_CXX, and the PT-Scotch
# targets it fetches/builds itself (ptscotch, scotch, ptscotcherrexit), and
# exposes src/ on the include path (for "ELEMENTS.h", "swage/...", etc.).
if(NOT TARGET ELEMENTS)
    set(ELEMENTS_ENABLE_SERIAL   ${Test2_ENABLE_SERIAL}   CACHE BOOL "" FORCE)
    set(ELEMENTS_ENABLE_OPENMP   ${Test2_ENABLE_OPENMP}   CACHE BOOL "" FORCE)
    set(ELEMENTS_ENABLE_PTHREADS ${Test2_ENABLE_PTHREADS} CACHE BOOL "" FORCE)
    set(ELEMENTS_ENABLE_CUDA     ${Test2_ENABLE_CUDA}     CACHE BOOL "" FORCE)
    set(ELEMENTS_ENABLE_HIP      ${Test2_ENABLE_HIP}      CACHE BOOL "" FORCE)
    set(ELEMENTS_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
    set(ELEMENTS_BUILD_TESTS     OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(
        ELEMENTS
        GIT_REPOSITORY https://github.com/lanl/ELEMENTS
        GIT_TAG        main
    )
    FetchContent_MakeAvailable(ELEMENTS)
endif()

FetchContent_GetProperties(ELEMENTS SOURCE_DIR ELEMENTS_SOURCE_DIR)

# --- Add Test2 Source ---
include_directories(include)
add_executable(main src/main.cpp)

# examples/decomp_example/include holds the mesh_io.h / state.h headers this
# example is built from; they are private to that example (not installed as
# part of the ELEMENTS library), so pull them in directly from the fetched
# ELEMENTS source tree, same as ELEMENTS's own examples/decomp_example/CMakeLists.txt does.
target_include_directories(main PRIVATE ${ELEMENTS_SOURCE_DIR}/examples/decomp_example/include)

target_link_libraries(main Kokkos::kokkos matar::matar MPI::MPI_CXX ELEMENTS)