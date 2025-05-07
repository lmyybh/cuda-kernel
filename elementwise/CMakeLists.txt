include_directories(../common ../thirdparty)

add_executable(elementwise main.cpp elementwise_add.cu)
target_compile_options(elementwise PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_compile_features(elementwise PRIVATE cxx_std_17 cuda_std_17)
set_target_properties(elementwise PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(elementwise PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
