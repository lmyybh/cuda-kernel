#include <stdio.h>
#include <time.h>
#include <sys/time.h>

#define CHECK(call)                                                             \
    {                                                                           \
        const cudaError_t error = call;                                         \
        if (error != cudaSuccess)                                               \
        {                                                                       \
            printf("Error: %s:%d, ", __FILE__, __LINE__);                       \
            printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
            exit(-10 * error);                                                  \
        }                                                                       \
    }

double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);

    return (double)tp.tv_sec + (double)tp.tv_usec * 1.e-6;
}

void initialRangeData(float *p, const int size, float start, float step)
{
    for (int i = 0; i < size; ++i)
    {
        p[i] = start + step * i;
    }
}

void printMatrix(float *p, const int size)
{
    for (int i = 0; i < size; ++i)
    {
        printf("%f, ", p[i]);
    }
    printf("\n");
}

void cpuGEMM(float *A, float *B, float *C, float alpha, float beta, const int M, const int N, const int K)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float sum = 0.0;
            for (int q = 0; q < K; ++q)
            {
                sum += A[i * K + q] * B[q * N + j];
            }
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}

void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double eps = 1.0E-8;
    int match = 1;
    for (int i = 0; i < N; ++i)
    {
        if (abs(hostRef[i] - gpuRef[i]) > eps)
        {
            match = 0;
            printf("Arrays don't match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
        }
        break;
    }

    if (match)
        printf("Arrays match.\n");
}
