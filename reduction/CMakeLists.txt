include_directories(../common ../thirdparty)

add_executable(reduction main.cpp reduction.cu)
target_compile_options(reduction PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_compile_features(reduction PRIVATE cxx_std_17 cuda_std_17)
set_target_properties(reduction PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(reduction PUBLIC ${CUDAToolkit_INCLUDE_DIRS})

add_executable(reduction_cub reduction_cub.cu reduction.cu)
