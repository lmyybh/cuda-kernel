cmake_minimum_required(VERSION 3.20)

project(elementwise LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 80 86 87 89 90 100 101 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

include_directories(../common ../thirdparty)

add_executable(elementwise main.cpp elementwise_add.cu)

target_compile_options(elementwise PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(elementwise PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(elementwise PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(elementwise PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
