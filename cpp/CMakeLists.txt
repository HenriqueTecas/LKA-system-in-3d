cmake_minimum_required(VERSION 3.15)
project(lka_cpp_accelerators LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# Find required packages
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(OpenMP REQUIRED)

# Find or download pybind11
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Find or download Eigen
find_package(Eigen3 3.3 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        Eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE
    )
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(Eigen)
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/lane_detection.cpp
    src/mpc_controller.cpp
    src/hybrid_controller.cpp
    src/physics.cpp
    src/utils.cpp
)

# Create Python module
pybind11_add_module(lka_cpp_accelerators
    bindings/bindings.cpp
    ${SOURCES}
)

# Link libraries
target_link_libraries(lka_cpp_accelerators PRIVATE OpenMP::OpenMP_CXX)
if(TARGET Eigen3::Eigen)
    target_link_libraries(lka_cpp_accelerators PRIVATE Eigen3::Eigen)
else()
    target_link_libraries(lka_cpp_accelerators PRIVATE eigen)
endif()

# Enable warnings
target_compile_options(lka_cpp_accelerators PRIVATE
    -Wall -Wextra -Wpedantic
)

# Installation
install(TARGETS lka_cpp_accelerators
    LIBRARY DESTINATION ${Python_SITEARCH}
)
