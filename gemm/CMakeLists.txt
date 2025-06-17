include_directories(../common ../thirdparty)

add_executable(gemm main.cpp gemm.cu)
target_compile_options(gemm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_compile_features(gemm PRIVATE cxx_std_17 cuda_std_17)
set_target_properties(gemm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(gemm PUBLIC ${CUDAToolkit_INCLUDE_DIRS})


add_executable(gemm_cublas gemm_cublas.cu)
target_compile_options(gemm_cublas PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_compile_features(gemm_cublas PRIVATE cxx_std_17 cuda_std_17)
set_target_properties(gemm_cublas PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(gemm_cublas PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(gemm_cublas PRIVATE
    CUDA::cudart
    CUDA::cublas
)

add_executable(case1 case.cu)
target_compile_options(case1 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_compile_features(case1 PRIVATE cxx_std_17 cuda_std_17)
set_target_properties(case1 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(case1 PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(case1 PRIVATE
    CUDA::cudart
    CUDA::cublas
)
