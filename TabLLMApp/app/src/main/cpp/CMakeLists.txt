cmake_minimum_required(VERSION 3.22.1)
project("tabllm_jni")

add_library(tabllm_jni SHARED
    native-lib.cpp)

find_library(log-lib log)

target_link_libraries(tabllm_jni
    ${log-lib}
    android)

# 1. Optimization and 16KB page alignment
target_compile_options(tabllm_jni PRIVATE -O3)
target_link_options(tabllm_jni PRIVATE "-Wl,-z,max-page-size=16384")

# 2. OpenMP (Compile flag vs Linker flag)
target_compile_options(tabllm_jni PRIVATE -fopenmp)
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fopenmp -static-openmp")