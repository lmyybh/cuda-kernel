#include "utils.cuh"

template <const int Bm, const int Bn, const int Bk, const int Tm, const int Tn>
__global__ void blockGEMM(
    float *__restrict__ A, float *__restrict__ B, float *__restrict__ C,
    float alpha, float beta,
    const int M, const int N, const int K)
{
    int bx = blockIdx.x; // 每个 block 在 grid 中的列号
    int by = blockIdx.y; // 每个 block 在 grid 中的行号

    int tx = threadIdx.x; // 每个 thread 在 block 中的列号
    int ty = threadIdx.y; // 每个 thread 在 block 中的行号

    // 为避免因参数设置导致的溢出，计算参与计算的有效线程的行数和列数
    int theard_x_per_block = (Bn + Tn - 1) / Tn; // 每个 block 中线程的列数
    int theard_y_per_block = (Bm + Tm - 1) / Tm; // 每个 block 中线程的行数
    int numberThreads = theard_x_per_block * theard_y_per_block;

    int tid = ty * theard_x_per_block + tx; // 每个 thread 在 block 中的序号

    __shared__ float As[Bm][Bk];
    __shared__ float Bs[Bk][Bn];

    float Ct[Tm][Tn] = {0.0};

    // 每个 Block 负责 Bm * Bn 大小的切块
    // 外层循环，每次读取 A 中 Bm * Bk 个元素，B 中 Bk * Bn 个元素，共循环 K / Bk 次
    for (int i = 0; i < (K + Bk - 1) / Bk; ++i)
    {

        // =============================
        // 读取 A 中元素到 shared memory
        // =============================

        // 计算 block 需要读取的 A 切块的左上角坐标
        int block_A_TILE_R0 = by * Bm;
        int block_A_TILE_C0 = Bk * i;

        // 计算每个线程需要读取 A 切块中的元素个数
        int numA = Bm * Bk / numberThreads;

        // 循环读取线程 tid 负责的 numA 个元素
        for (int n = tid * numA; n < (tid + 1) * numA; ++n)
        {
            // 计算第 n 个元素在 A_TILE [Bm, Bk] 中的位置
            int A_TILE_r = n / Bk; // 第 n 个元素在 A_TILE 中的行号
            int A_TILE_c = n % Bk; // 第 n 个元素在 A_TILE 中的列号

            // 计算第 n 个元素在 A [M, K] 中的位置
            int A_r = block_A_TILE_R0 + A_TILE_r;
            int A_c = block_A_TILE_C0 + A_TILE_c;

            // global -> shared
            As[A_TILE_r][A_TILE_c] = A[A_r * K + A_c];
        }

        // =============================
        // 读取 B 中元素到 shared memory
        // =============================

        // 计算 block 需要读取的 B 切块的左上角坐标
        int block_B_TILE_R0 = Bk * i;
        int block_B_TILE_C0 = bx * Bn;

        // 计算每个线程需要读取 B 切块中的元素个数
        int numB = Bk * Bn / numberThreads;

        // 循环读取线程 tid 负责的 num 个元素
        for (int n = tid * numB; n < (tid + 1) * numB; ++n)
        {
            // 计算第 n 个元素在 A_TILE [Bm, Bk] 中的位置
            int B_TILE_r = n / Bn; // 第 n 个元素在 B_TILE 中的行号
            int B_TILE_c = n % Bn; // 第 n 个元素在 B_TILE 中的列号

            // 计算第 n 个元素在 A [M, K] 中的位置
            int B_r = block_B_TILE_R0 + B_TILE_r;
            int B_c = block_B_TILE_C0 + B_TILE_c;

            // global -> shared
            Bs[B_TILE_r][B_TILE_c] = B[B_r * N + B_c];
        }
        __syncthreads(); // 等待所有线程读取完成

        // =============================
        // 计算每个线程对应的切块 Tm * Tn
        // =============================
        for (int k = 0; k < Bk; ++k)
        {
            for (int m = 0; m < Tm; ++m)
            {
                for (int n = 0; n < Tn; ++n)
                {
                    Ct[m][n] += As[ty * Tm + m][k] * Bs[k][tx * Tn + n];
                }
            }
        }
        __syncthreads(); // 等待所有线程计算完成
    }

    // Tm *Tn 切块的计算结果写回 C 中
    for (int m = 0; m < Tm; ++m)
    {
        int r = by * Bm + ty * Tm + m; // 切块第 m 行对应 C 中的行号
        for (int n = 0; n < Tn; ++n)
        {
            int c = bx * Bn + tx * Tn + n; // 切块第 n 列对应 C 中的列号
            C[r * N + c] = alpha * Ct[m][n] + beta * C[r * N + c];
        }
    }
}

int main(void)
{
    const int M = 1024;
    const int N = 1024;
    const int K = 512;
    float alpha = 1.3;
    float beta = 2.1;

    const int Bm = 128, Bn = 128, Bk = 8;
    const int Tm = 8, Tn = 8;

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
    dim3 block((Bm + Tm - 1) / Tm, (Bn + Tn - 1) / Tn);
    dim3 grid((M + Bm - 1) / Bm, (N + Bn - 1) / Bn);
    iStart = cpuSecond();
    blockGEMM<Bm, Bn, Bk, Tm, Tn><<<grid, block>>>(d_A, d_B, d_C, alpha, beta, M, N, K);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    printf("blockGEMM<<<(%d, %d), (%d, %d)>>> elapsed %f sec\n", grid.x, grid.y, block.x, block.y, iElaps);

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
