#include <time.h>
#include <stdio.h>

void initialRangeData(float *p, int size, float start, float step) {
    for (int i = 0; i < size; ++i) {
        p[i] = start + step * i;
    }
}

void printMatrix(float *p, int size) {
    for (int i = 0; i < size; ++i) {
        printf("%f, ", p[i]);
    }
    printf("\n");
}

void cpuGEMM(float *A, float *B, float *C, float alpha, float beta, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0;
            for (int q = 0; q < K; ++q) {
                sum += A[i * K + q] * B[q * N + j];
            }
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}


int main() {
    int M = 4;
    int N = 3;
    int K = 2;

    float alpha = 1.3;
    float beta = 2.1;

    float *A, *B, *C;
    A = (float*)malloc(M * K * sizeof(float));
    B = (float*)malloc(K * N * sizeof(float));
    C = (float*)malloc(M * N * sizeof(float));

    initialRangeData(A, M*K, 1.0, 1.0);
    initialRangeData(B, K*N, 5.0, 0.5);
    initialRangeData(C, M*N, 2.3, 3.0);

    printMatrix(A, M*K);
    printMatrix(B, K*N);
    printMatrix(C, M*N);

    cpuGEMM(A, B, C, alpha, beta, M, N, K);

    printMatrix(C, M*N);

    free(A);
    free(B);
    free(C);

    return 0;
}