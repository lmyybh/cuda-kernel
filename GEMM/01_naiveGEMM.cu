#include "utils.cuh"

__global__ void naiveGEMM(
    float *__restrict__ A, float *__restrict__ B, float *__restrict__ C,
    float alpha, float beta,
    const int M, const int N, const int K)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // 列号
    int j = blockIdx.y * blockDim.y + threadIdx.y; // 行号

    if (i < M && j < N)
    {
        float sum = 0.0;
        for (int q = 0; q < K; ++q)
        {
            sum += A[j * K + q] * B[q * N + i];
        }
        C[j * N + i] = alpha * sum + beta * C[j * N + i];
    }
}

int main(void)
{
    const int M = 1024;
    const int N = 1024;
    const int K = 512;
    float alpha = 1.3;
    float beta = 2.1;

    int BM = 32, BN = 32;

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
    dim3 block(BM, BN);
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    naiveGEMM<<<grid, block>>>(d_A, d_B, d_C, alpha, beta, M, N, K);
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

    return 0;
}
