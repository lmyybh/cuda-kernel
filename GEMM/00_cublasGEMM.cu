#include <cublas_v2.h>
#include "utils.cuh"

int main()
{
    const int M = 1024;
    const int N = 1024;
    const int K = 512;
    float alpha = 1.3;
    float beta = 2.1;

    const int nBytesA = M * K * sizeof(float);
    const int nBytesB = K * N * sizeof(float);
    const int nBytesC = M * N * sizeof(float);

    // 初始化 host 数据
    float *A, *B, *C;
    A = (float *)malloc(nBytesA);
    B = (float *)malloc(nBytesB);
    C = (float *)malloc(nBytesC);

    initialRangeData(A, M * K, 1.0, 1.0);
    initialRangeData(B, K * N, 5.0, 0.5);
    initialRangeData(C, M * N, 2.3, 3.0);

    // 数据拷贝至 device
    float *d_A, *d_B, *d_C;
    cudaMalloc((float **)&d_A, nBytesA);
    cudaMalloc((float **)&d_B, nBytesB);
    cudaMalloc((float **)&d_C, nBytesC);
    cudaMemcpy(d_A, A, nBytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, nBytesB, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, nBytesC, cudaMemcpyHostToDevice);

    // host 计算
    double iStart = cpuSecond();
    cpuGEMM(A, B, C, alpha, beta, M, N, K);
    double iElaps = cpuSecond() - iStart;
    printf("cpuGEMM elapsed %f sec\n", iElaps);

    // device 计算
    cublasHandle_t handle;
    cublasCreate_v2(&handle);
    cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    cudaDeviceSynchronize();

    // 检查结果
    float *gpuC;
    gpuC = (float *)malloc(nBytesC);
    cudaMemcpy(gpuC, d_C, nBytesC, cudaMemcpyDeviceToHost);
    checkResult(C, gpuC, M * N);

    // 释放资源
    free(A);
    free(B);
    free(C);
    free(gpuC);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);

    return 0;
}
