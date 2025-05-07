include_directories(../common ../thirdparty)

add_executable(transpose main.cpp transpose.cu)
target_compile_options(transpose PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_compile_features(transpose PRIVATE cxx_std_17 cuda_std_17)
set_target_properties(transpose PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(transpose PUBLIC ${CUDAToolkit_INCLUDE_DIRS})

add_executable(transpose_cublas transpose_cublas.cu)
target_compile_options(transpose_cublas PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_compile_features(transpose_cublas PRIVATE cxx_std_17 cuda_std_17)
set_target_properties(transpose_cublas PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(transpose_cublas PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(transpose_cublas PRIVATE
    CUDA::cudart
    CUDA::cublas
)
