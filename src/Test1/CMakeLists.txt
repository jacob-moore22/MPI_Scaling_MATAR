cmake_minimum_required(VERSION 3.20)
project(Test1 VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# cmake/ lives two levels above src/Test1/ at the repo root
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/../../cmake)

# Find the vector library
find_package(Vector REQUIRED)

# Set the compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${VECTOR_CXX_FLAGS}")


# --- Set build type if not already set ---
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
 # set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)


# --- Options ---
option(Test1_ENABLE_SERIAL "Enable Serial support in Kokkos" ON)
option(Test1_ENABLE_OPENMP "Enable OpenMP support in Kokkos" OFF)
option(Test1_ENABLE_PTHREADS "Enable PThreads support in Kokkos" OFF)
option(Test1_ENABLE_CUDA "Enable CUDA support in Kokkos" OFF)
option(Test1_ENABLE_HIP "Enable HIP support in Kokkos" OFF)

include(FetchContent)

# --- Dependency: Kokkos ---
if(NOT TARGET Kokkos::kokkos)
    set(Kokkos_ENABLE_SERIAL   ${Test1_ENABLE_SERIAL}   CACHE BOOL "" FORCE)
    set(Kokkos_ENABLE_OPENMP   ${Test1_ENABLE_OPENMP}   CACHE BOOL "" FORCE)
    set(Kokkos_ENABLE_PTHREADS ${Test1_ENABLE_PTHREADS} CACHE BOOL "" FORCE)
    set(Kokkos_ENABLE_CUDA     ${Test1_ENABLE_CUDA}     CACHE BOOL "" FORCE)
    set(Kokkos_ENABLE_HIP      ${Test1_ENABLE_HIP}      CACHE BOOL "" FORCE)
    
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

# --- Add Test1 Source ---
add_subdirectory(src)   