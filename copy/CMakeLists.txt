cmake_minimum_required(VERSION 3.20)

project(copy C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 80 86 87 89 90 100 101 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
include_directories(../common ../thirdparty)

add_executable(copy main.cpp copy.cu)

target_compile_options(copy PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(copy PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(copy PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(copy PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
