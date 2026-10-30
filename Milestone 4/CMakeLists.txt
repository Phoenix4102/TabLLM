cmake_minimum_required(VERSION 3.22)
project(tabllm CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(tabllm main.cpp)

target_compile_options(tabllm PRIVATE
    # -O3
    # -Wall
    # -Wextra
)

option(TABLLM_OPENMP "Enable OpenMP multithreading" ON)

if(TABLLM_OPENMP)
    find_package(OpenMP REQUIRED)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(tabllm PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

include(CheckCXXCompilerFlag)

set(TABLLM_ARM_FLAG "-march=armv8.2-a+fp16+dotprod")

check_cxx_compiler_flag("${TABLLM_ARM_FLAG}" TABLLM_HAS_ARM_FLAG)

if(TABLLM_HAS_ARM_FLAG)
    target_compile_options(tabllm PRIVATE "${TABLLM_ARM_FLAG}")
endif()