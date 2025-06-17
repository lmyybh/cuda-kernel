#include <stdio.h>

#define WARP_SIZE 32
#define CEIL(a, b) (((a) + (b) - 1) / (b))
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

// 每个 thread 负责计算 C 中的一个元素
__global__ void naiveGEMM(float* A, float* B, float* C, const int M, const int K, const int N) {
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.x * blockDim.x + threadIdx.x;

  if (r >= M || c >= N) { return; }

  float value = 0.f;
  for (int k = 0; k < K; ++k) { value += A[r * K + k] * B[k * N + c]; }
  C[r * N + c] = value;
}

// 每个 Block 负责计算 C 中 (Bm, Bn) 大小的分块 tileC
template<int Bm = 128, int Bn = 128, int Bk = 8, int blockSize = 256, int A_BLOCK_X = 8,
         int B_BLOCK_X = 32, int C_BLOCK_X = 16>
__global__ void blockTileGEMM(float* A, float* B, float* C, const int M, const int K, const int N) {
  __shared__ float As[Bm][Bk];  // 存储 tileA
  __shared__ float Bs[Bk][Bn];  // 存储 tileB

  // 计算 block 负责的 tileC 左上角元素的行列坐标
  int r0 = blockIdx.y * Bm;
  int c0 = blockIdx.x * Bn;

  // 当前 thread 的编号（默认为一维 block 配置）
  int tid = threadIdx.x;

  /*------ tileA ------*/
  // 写入 A tile 时，block 中 thread 排布尺寸为 (A_BLOCK_X, blockSize / A_BLOCK_X) = (8, 32)
  constexpr int A_BLOCK_Y = blockSize / A_BLOCK_X;

  // 对于 tid 号线程，其位于 blockA 中的行列坐标为 (tid / A_BLOCK_X, tid % A_BLOCK_X)
  int A_THREAD_Y = tid / A_BLOCK_X;
  int A_THREAD_X = tid % A_BLOCK_X;

  /*------ tileB ------*/
  // 写入 B tile 时，block 中 thread 排布尺寸为 (B_BLOCK_X, blockSize / B_BLOCK_X) = (32, 8)
  constexpr int B_BLOCK_Y = blockSize / B_BLOCK_X;

  // 对于 tid 号线程，其位于 blockB 中的行列坐标为 (tid / B_BLOCK_X, tid % B_BLOCK_X)
  int B_THREAD_Y = tid / B_BLOCK_X;
  int B_THREAD_X = tid % B_BLOCK_X;

  /*------ tileC ------*/
  constexpr int C_BLOCK_Y = blockSize / C_BLOCK_X;

  // 对于 tid 号线程，其位于 blockC 中的行列坐标为 (tid / C_BLOCK_X, tid % C_BLOCK_X)
  int C_THREAD_Y = tid / C_BLOCK_X;
  int C_THREAD_X = tid % C_BLOCK_X;

  // 每个 thread 负责 Tm * Tn 个元素计算
  constexpr int Tm = Bm / C_BLOCK_Y;
  constexpr int Tn = Bn / C_BLOCK_X;
  float Ct[Tm][Tn] = {0.0};

  // K- Loop
  for (int k = 0; k < K; k += Bk) {
    /* ------ 读取 global memory，存入 shared memory ------ */
    // 使用跨步循环，行方向的 stride 为 A_BLOCK_Y, 列方向的 stride 为 A_BLOCK_X
#pragma unroll
    for (int i = A_THREAD_Y; i < Bm; i += A_BLOCK_Y) {
      int r = r0 + i;
#pragma unroll
      for (int j = A_THREAD_X; j < Bk; j += A_BLOCK_X) {
        int c = k + j;
        As[i][j] = (r < M && c < K) ? A[r * K + c] : 0.f;
      }
    }

    // 使用跨步循环，行方向的 stride 为 B_BLOCK_Y, 列方向的 stride 为 B_BLOCK_X
#pragma unroll
    for (int i = B_THREAD_Y; i < Bk; i += B_BLOCK_Y) {
      int r = k + i;
#pragma unroll
      for (int j = B_THREAD_X; j < Bn; j += B_BLOCK_X) {
        int c = c0 + j;

        Bs[i][j] = (r < K && c < N) ? B[r * N + c] : 0.f;
      }
    }

    __syncthreads();

    /* ------ 计算 tileA * tileB ------ */
    // 先循环 k 维度，按向量外积的方式计算
#pragma unroll
    for (int p = 0; p < Bk; ++p) {
      // 使用跨步循环，行方向的 stride 为 C_BLOCK_Y, 列方向的 stride 为 C_BLOCK_X
#pragma unroll
      for (int i = 0; i < Tm; ++i) {
        int r = C_THREAD_Y + i * C_BLOCK_Y;
#pragma unroll
        for (int j = 0; j < Tn; ++j) {
          int c = C_THREAD_X + j * C_BLOCK_X;
          Ct[i][j] += As[r][p] * Bs[p][c];
        }
      }
    }

    __syncthreads();
  }

  // 将 Ct 写入 C
#pragma unroll
  for (int i = 0; i < Tm; ++i) {
    int r = r0 + C_THREAD_Y + i * C_BLOCK_Y;
#pragma unroll
    for (int j = 0; j < Tn; ++j) {
      int c = c0 + C_THREAD_X + j * C_BLOCK_X;

      if (r < M && c < N) { C[r * N + c] = Ct[i][j]; }
    }
  }
}

// 使用寄存器优化 shared memory 访问次数
template<int Bm = 128, int Bn = 128, int Bk = 8, int blockSize = 256, int A_BLOCK_X = 8,
         int B_BLOCK_X = 32, int C_BLOCK_X = 16>
__global__ void threadTileGEMM(float* A, float* B, float* C, const int M, const int K,
                               const int N) {
  __shared__ float As[Bm][Bk];  // 存储 tileA
  __shared__ float Bs[Bk][Bn];  // 存储 tileB

  // 计算 block 负责的 tileC 左上角元素的行列坐标
  int r0 = blockIdx.y * Bm;
  int c0 = blockIdx.x * Bn;

  // 当前 thread 的编号（默认为一维 block 配置）
  int tid = threadIdx.x;

  /*------ tileA ------*/
  // 写入 A tile 时，block 中 thread 排布尺寸为 (A_BLOCK_X, blockSize / A_BLOCK_X) = (8, 32)
  constexpr int A_BLOCK_Y = blockSize / A_BLOCK_X;

  // 对于 tid 号线程，其位于 blockA 中的行列坐标为 (tid / A_BLOCK_X, tid % A_BLOCK_X)
  int A_THREAD_Y = tid / A_BLOCK_X;
  int A_THREAD_X = tid % A_BLOCK_X;

  /*------ tileB ------*/
  // 写入 B tile 时，block 中 thread 排布尺寸为 (B_BLOCK_X, blockSize / B_BLOCK_X) = (32, 8)
  constexpr int B_BLOCK_Y = blockSize / B_BLOCK_X;

  // 对于 tid 号线程，其位于 blockB 中的行列坐标为 (tid / B_BLOCK_X, tid % B_BLOCK_X)
  int B_THREAD_Y = tid / B_BLOCK_X;
  int B_THREAD_X = tid % B_BLOCK_X;

  /*------ tileC ------*/
  constexpr int C_BLOCK_Y = blockSize / C_BLOCK_X;

  // 对于 tid 号线程，其位于 blockC 中的行列坐标为 (tid / C_BLOCK_X, tid % C_BLOCK_X)
  int C_THREAD_Y = tid / C_BLOCK_X;
  int C_THREAD_X = tid % C_BLOCK_X;

  // 每个 thread 负责 Tm * Tn 个元素计算
  constexpr int Tm = Bm / C_BLOCK_Y;
  constexpr int Tn = Bn / C_BLOCK_X;
  float Ct[Tm][Tn] = {0.0};

  // 存储 A 中列向量和 B 中行向量
  float regA[Tm] = {0.0f};
  float regB[Tn] = {0.0f};

  // K- Loop
  for (int k = 0; k < K; k += Bk) {
    /* ------ 读取 global memory，存入 shared memory ------ */
    // 使用跨步循环，行方向的 stride 为 A_BLOCK_Y, 列方向的 stride 为 A_BLOCK_X
#pragma unroll
    for (int i = A_THREAD_Y; i < Bm; i += A_BLOCK_Y) {
      int r = r0 + i;
#pragma unroll
      for (int j = A_THREAD_X; j < Bk; j += A_BLOCK_X) {
        int c = k + j;
        As[i][j] = (r < M && c < K) ? A[r * K + c] : 0.f;
      }
    }

    // 使用跨步循环，行方向的 stride 为 B_BLOCK_Y, 列方向的 stride 为 B_BLOCK_X
#pragma unroll
    for (int i = B_THREAD_Y; i < Bk; i += B_BLOCK_Y) {
      int r = k + i;
#pragma unroll
      for (int j = B_THREAD_X; j < Bn; j += B_BLOCK_X) {
        int c = c0 + j;

        Bs[i][j] = (r < K && c < N) ? B[r * N + c] : 0.f;
      }
    }

    __syncthreads();

    /* ------ 计算 tileA * tileB ------ */
    // p-Loop，先循环 k 维度，按向量外积的方式计算
#pragma unroll
    for (int p = 0; p < Bk; ++p) {
      // 存储 A 中列向量到 regA
#pragma unroll
      for (int i = 0; i < Tm; ++i) {
        int r = C_THREAD_Y + i * C_BLOCK_Y;
        regA[i] = As[r][p];
      }

      // 存储 B 中行向量到 regB
#pragma unroll
      for (int j = 0; j < Tn; ++j) {
        int c = C_THREAD_X + j * C_BLOCK_X;
        regB[j] = Bs[p][c];
      }

      // 计算 regA 与 regB 的向量外积
#pragma unroll
      for (int i = 0; i < Tm; ++i) {
#pragma unroll
        for (int j = 0; j < Tn; ++j) { Ct[i][j] += regA[i] * regB[j]; }
      }
    }

    __syncthreads();
  }

  // 将 Ct 写入 C
#pragma unroll
  for (int i = 0; i < Tm; ++i) {
    int r = r0 + C_THREAD_Y + i * C_BLOCK_Y;
#pragma unroll
    for (int j = 0; j < Tn; ++j) {
      int c = c0 + C_THREAD_X + j * C_BLOCK_X;

      if (r < M && c < N) { C[r * N + c] = Ct[i][j]; }
    }
  }
}

// 将 blockC 中的 warp 排列为 8*4
template<int Bm = 128, int Bn = 128, int Bk = 8, int blockSize = 256, int A_BLOCK_X = 8,
         int B_BLOCK_X = 32, int C_BLOCK_X = 16, int C_WARP_X = 8, int C_WARP_Y = 4>
__global__ void warpGEMM(float* A, float* B, float* C, const int M, const int K, const int N) {
  __shared__ float As[Bm][Bk];  // 存储 tileA
  __shared__ float Bs[Bk][Bn];  // 存储 tileB

  // 计算 block 负责的 tileC 左上角元素的行列坐标
  int r0 = blockIdx.y * Bm;
  int c0 = blockIdx.x * Bn;

  // 当前 thread 的编号（默认为一维 block 配置）
  int tid = threadIdx.x;

  /*------ tileA ------*/
  // 写入 A tile 时，block 中 thread 排布尺寸为 (A_BLOCK_X, blockSize / A_BLOCK_X) = (8, 32)
  constexpr int A_BLOCK_Y = blockSize / A_BLOCK_X;

  // 对于 tid 号线程，其位于 blockA 中的行列坐标为 (tid / A_BLOCK_X, tid % A_BLOCK_X)
  int A_THREAD_Y = tid / A_BLOCK_X;
  int A_THREAD_X = tid % A_BLOCK_X;

  /*------ tileB ------*/
  // 写入 B tile 时，block 中 thread 排布尺寸为 (B_BLOCK_X, blockSize / B_BLOCK_X) = (32, 8)
  constexpr int B_BLOCK_Y = blockSize / B_BLOCK_X;

  // 对于 tid 号线程，其位于 blockB 中的行列坐标为 (tid / B_BLOCK_X, tid % B_BLOCK_X)
  int B_THREAD_Y = tid / B_BLOCK_X;
  int B_THREAD_X = tid % B_BLOCK_X;

  /*------ tileC ------*/
  constexpr int C_BLOCK_Y = blockSize / C_BLOCK_X;

  // 按 8*4 排列 warp

  // 计算当前 thread 属于哪个 warp，第几个 lane
  int warpId = tid / WARP_SIZE;
  int laneId = tid % WARP_SIZE;
  // 计算总共有几行几列 warp
  constexpr int C_WARP_DIM_X = C_BLOCK_X / C_WARP_X;

  // 计算当前 thread 所在 warp 在 block 中的 x, y 坐标
  int warpX = warpId % C_WARP_DIM_X;
  int warpY = warpId / C_WARP_DIM_X;

  // 计算当前 thread 在 warp 中的 x, y 坐标
  int laneY = laneId / C_WARP_X;
  int laneX = laneId % C_WARP_X;

  // 当前 thread 在 blockC 中的行列坐标为 (warpY * C_WARP_Y + laneY, warpX * C_WARP_X + laneX)
  int C_THREAD_Y = warpY * C_WARP_Y + laneY;
  int C_THREAD_X = warpX * C_WARP_X + laneX;

  // 每个 thread 负责 Tm * Tn 个元素计算
  constexpr int Tm = Bm / C_BLOCK_Y;
  constexpr int Tn = Bn / C_BLOCK_X;
  float Ct[Tm][Tn] = {0.0};

  // 存储 A 中列向量和 B 中行向量
  float regA[Tm] = {0.0f};
  float regB[Tn] = {0.0f};

  // K- Loop
  for (int k = 0; k < K; k += Bk) {
    /* ------ 读取 global memory，存入 shared memory ------ */
    // 使用跨步循环，行方向的 stride 为 A_BLOCK_Y, 列方向的 stride 为 A_BLOCK_X
#pragma unroll
    for (int i = A_THREAD_Y; i < Bm; i += A_BLOCK_Y) {
      int r = r0 + i;
#pragma unroll
      for (int j = A_THREAD_X; j < Bk; j += A_BLOCK_X) {
        int c = k + j;
        As[i][j] = (r < M && c < K) ? A[r * K + c] : 0.f;
      }
    }

    // 使用跨步循环，行方向的 stride 为 B_BLOCK_Y, 列方向的 stride 为 B_BLOCK_X
#pragma unroll
    for (int i = B_THREAD_Y; i < Bk; i += B_BLOCK_Y) {
      int r = k + i;
#pragma unroll
      for (int j = B_THREAD_X; j < Bn; j += B_BLOCK_X) {
        int c = c0 + j;

        Bs[i][j] = (r < K && c < N) ? B[r * N + c] : 0.f;
      }
    }

    __syncthreads();

    /* ------ 计算 tileA * tileB ------ */
    // p-Loop，先循环 k 维度，按向量外积的方式计算
#pragma unroll
    for (int p = 0; p < Bk; ++p) {
      // 存储 A 中列向量到 regA
#pragma unroll
      for (int i = 0; i < Tm; ++i) {
        int r = C_THREAD_Y + i * C_BLOCK_Y;
        regA[i] = As[r][p];
      }

      // 存储 B 中行向量到 regB
#pragma unroll
      for (int j = 0; j < Tn; ++j) {
        int c = C_THREAD_X + j * C_BLOCK_X;
        regB[j] = Bs[p][c];
      }

      // 计算 regA 与 regB 的向量外积
#pragma unroll
      for (int i = 0; i < Tm; ++i) {
#pragma unroll
        for (int j = 0; j < Tn; ++j) { Ct[i][j] += regA[i] * regB[j]; }
      }
    }

    __syncthreads();
  }

  // 将 Ct 写入 C
#pragma unroll
  for (int i = 0; i < Tm; ++i) {
    int r = r0 + C_THREAD_Y + i * C_BLOCK_Y;
#pragma unroll
    for (int j = 0; j < Tn; ++j) {
      int c = c0 + C_THREAD_X + j * C_BLOCK_X;

      if (r < M && c < N) { C[r * N + c] = Ct[i][j]; }
    }
  }
}

// 向量化读取 shared memory
template<int Bm = 128, int Bn = 128, int Bk = 8, int blockSize = 256, int A_BLOCK_X = 8,
         int B_BLOCK_X = 32, int C_BLOCK_X = 16, int C_WARP_X = 8, int C_WARP_Y = 4>
__global__ void float4GEMM(float* A, float* B, float* C, const int M, const int K, const int N) {
  __shared__ float As[Bk][Bm];  // 存储转置后的 tileA
  __shared__ float Bs[Bk][Bn];  // 存储 tileB

  // 计算 block 负责的 tileC 左上角元素的行列坐标
  int r0 = blockIdx.y * Bm;
  int c0 = blockIdx.x * Bn;

  // 当前 thread 的编号（默认为一维 block 配置）
  int tid = threadIdx.x;

  /*------ tileA ------*/
  // 写入 A tile 时，block 中 thread 排布尺寸为 (A_BLOCK_X, blockSize / A_BLOCK_X) = (8, 32)
  constexpr int A_BLOCK_Y = blockSize / A_BLOCK_X;

  // 对于 tid 号线程，其位于 blockA 中的行列坐标为 (tid / A_BLOCK_X, tid % A_BLOCK_X)
  int A_THREAD_Y = tid / A_BLOCK_X;
  int A_THREAD_X = tid % A_BLOCK_X;

  /*------ tileB ------*/
  // 写入 B tile 时，block 中 thread 排布尺寸为 (B_BLOCK_X, blockSize / B_BLOCK_X) = (32, 8)
  constexpr int B_BLOCK_Y = blockSize / B_BLOCK_X;

  // 对于 tid 号线程，其位于 blockB 中的行列坐标为 (tid / B_BLOCK_X, tid % B_BLOCK_X)
  int B_THREAD_Y = tid / B_BLOCK_X;
  int B_THREAD_X = tid % B_BLOCK_X;

  /*------ tileC ------*/
  constexpr int C_BLOCK_Y = blockSize / C_BLOCK_X;

  // 按 8*4 排列 warp

  // 计算当前 thread 属于哪个 warp，第几个 lane
  int warpId = tid / WARP_SIZE;
  int laneId = tid % WARP_SIZE;
  // 计算总共有几行几列 warp
  constexpr int C_WARP_DIM_X = C_BLOCK_X / C_WARP_X;

  // 计算当前 thread 所在 warp 在 block 中的 x, y 坐标
  int warpX = warpId % C_WARP_DIM_X;
  int warpY = warpId / C_WARP_DIM_X;

  // 计算当前 thread 在 warp 中的 x, y 坐标
  int laneY = laneId / C_WARP_X;
  int laneX = laneId % C_WARP_X;

  // 当前 thread 在 blockC 中的行列坐标为 (warpY * C_WARP_Y + laneY, warpX * C_WARP_X + laneX)
  int C_THREAD_Y = warpY * C_WARP_Y + laneY;
  int C_THREAD_X = warpX * C_WARP_X + laneX;

  // 每个 thread 负责 Tm * Tn 个元素计算
  constexpr int Tm = Bm / C_BLOCK_Y;
  constexpr int Tn = Bn / C_BLOCK_X;
  float Ct[Tm][Tn] = {0.0};

  // 存储 A 中列向量和 B 中行向量
  float regA[Tm] = {0.0f};
  float regB[Tn] = {0.0f};

  // K- Loop
  for (int k = 0; k < K; k += Bk) {
    /* ------ 读取 global memory，存入 shared memory ------ */
    // 使用跨步循环，行方向的 stride 为 A_BLOCK_Y, 列方向的 stride 为 A_BLOCK_X
#pragma unroll
    for (int i = A_THREAD_Y; i < Bm; i += A_BLOCK_Y) {
      int r = r0 + i;
#pragma unroll
      for (int j = A_THREAD_X; j < Bk; j += A_BLOCK_X) {
        int c = k + j;
        As[j][i] = (r < M && c < K) ? A[r * K + c] : 0.f;  // 转置
      }
    }

    // 使用跨步循环，行方向的 stride 为 B_BLOCK_Y, 列方向的 stride 为 B_BLOCK_X
#pragma unroll
    for (int i = B_THREAD_Y; i < Bk; i += B_BLOCK_Y) {
      int r = k + i;
#pragma unroll
      for (int j = B_THREAD_X; j < Bn; j += B_BLOCK_X) {
        int c = c0 + j;

        Bs[i][j] = (r < K && c < N) ? B[r * N + c] : 0.f;
      }
    }

    __syncthreads();

    /* ------ 计算 tileA * tileB ------ */
    // p-Loop，先循环 k 维度，按向量外积的方式计算
#pragma unroll
    for (int p = 0; p < Bk; ++p) {
      // 向量化访存，存储 A 中列向量到 regA
#pragma unroll
      for (int i = 0; i < Tm / 4; ++i) {
        int r = (C_THREAD_Y + i * C_BLOCK_Y) * 4;
        FLOAT4(regA[i * 4]) = FLOAT4(As[p][r]);
      }

      // 向量化访存，存储 B 中行向量到 regB
#pragma unroll
      for (int j = 0; j < Tn / 4; ++j) {
        int c = (C_THREAD_X + j * C_BLOCK_X) * 4;
        FLOAT4(regB[j * 4]) = FLOAT4(Bs[p][c]);
      }

      // 计算 regA 与 regB 的向量外积
#pragma unroll
      for (int i = 0; i < Tm; ++i) {
#pragma unroll
        for (int j = 0; j < Tn; ++j) { Ct[i][j] += regA[i] * regB[j]; }
      }
    }

    __syncthreads();
  }

  // 将 Ct 写入 C
#pragma unroll
  for (int i = 0; i < Tm; ++i) {
    int r = r0 + 4 * C_THREAD_Y + i / 4 * 4 * C_BLOCK_Y + i % 4;
#pragma unroll
    for (int j = 0; j < Tn; ++j) {
      int c = c0 + 4 * C_THREAD_X + j / 4 * 4 * C_BLOCK_X + j % 4;

      if (r < M && c < N) { C[r * N + c] = Ct[i][j]; }
    }
  }
}

// 向量化读取 shared memory，解决 Bank Conflicts
template<int Bm = 128, int Bn = 128, int Bk = 8, int blockSize = 256, int A_BLOCK_X = 8,
         int B_BLOCK_X = 32, int C_BLOCK_X = 16, int C_WARP_X = 8, int C_WARP_Y = 4>
__global__ void float4GEMMnoBC(float* A, float* B, float* C, const int M, const int K,
                               const int N) {
  __shared__ float As[Bk][Bm];  // 存储转置后的 tileA
  __shared__ float Bs[Bk][Bn];  // 存储 tileB

  // 计算 block 负责的 tileC 左上角元素的行列坐标
  int r0 = blockIdx.y * Bm;
  int c0 = blockIdx.x * Bn;

  // 当前 thread 的编号（默认为一维 block 配置）
  int tid = threadIdx.x;

  /*------ tileA ------*/
  // 写入 A tile 时，block 中 thread 排布尺寸为 (A_BLOCK_X, blockSize / A_BLOCK_X) = (8, 32)
  constexpr int A_BLOCK_Y = blockSize / A_BLOCK_X;

  // 对于 tid 号线程，其位于 blockA 中的行列坐标为 (tid / A_BLOCK_X, tid % A_BLOCK_X)
  int A_THREAD_Y = tid / A_BLOCK_X;
  int A_THREAD_X = tid % A_BLOCK_X;

  /*------ tileB ------*/
  // 写入 B tile 时，block 中 thread 排布尺寸为 (B_BLOCK_X, blockSize / B_BLOCK_X) = (32, 8)
  constexpr int B_BLOCK_Y = blockSize / B_BLOCK_X;

  // 对于 tid 号线程，其位于 blockB 中的行列坐标为 (tid / B_BLOCK_X, tid % B_BLOCK_X)
  int B_THREAD_Y = tid / B_BLOCK_X;
  int B_THREAD_X = tid % B_BLOCK_X;

  /*------ tileC ------*/
  constexpr int C_BLOCK_Y = blockSize / C_BLOCK_X;

  // 按 8*4 排列 warp

  // 计算当前 thread 属于哪个 warp，第几个 lane
  int warpId = tid / WARP_SIZE;
  int laneId = tid % WARP_SIZE;
  // 计算总共有几行几列 warp
  constexpr int C_WARP_DIM_X = C_BLOCK_X / C_WARP_X;

  // 计算当前 thread 所在 warp 在 block 中的 x, y 坐标
  int warpX = warpId % C_WARP_DIM_X;
  int warpY = warpId / C_WARP_DIM_X;

  // 计算当前 thread 在 warp 中的 x, y 坐标
  int laneY = laneId / C_WARP_X;
  int laneX = laneId % C_WARP_X;

  // 当前 thread 在 blockC 中的行列坐标为 (warpY * C_WARP_Y + laneY, warpX * C_WARP_X + laneX)
  int C_THREAD_Y = warpY * C_WARP_Y + laneY;
  int C_THREAD_X = warpX * C_WARP_X + laneX;

  // 每个 thread 负责 Tm * Tn 个元素计算
  constexpr int Tm = Bm / C_BLOCK_Y;
  constexpr int Tn = Bn / C_BLOCK_X;
  float Ct[Tm][Tn] = {0.0};

  // 存储 A 中列向量和 B 中行向量
  float regA[Tm] = {0.0f};
  float regB[Tn] = {0.0f};

  // K- Loop
  for (int k = 0; k < K; k += Bk) {
    /* ------ 读取 global memory，存入 shared memory ------ */
    // 使用跨步循环，行方向的 stride 为 A_BLOCK_Y, 列方向的 stride 为 A_BLOCK_X
#pragma unroll
    for (int i = A_THREAD_Y; i < Bm; i += A_BLOCK_Y) {
      int r = r0 + i;
#pragma unroll
      for (int j = A_THREAD_X; j < Bk; j += A_BLOCK_X) {
        int c = k + j;
        As[j][i ^ (4 * j)] = (r < M && c < K) ? A[r * K + c] : 0.f;  // 转置
      }
    }

    // 使用跨步循环，行方向的 stride 为 B_BLOCK_Y, 列方向的 stride 为 B_BLOCK_X
#pragma unroll
    for (int i = B_THREAD_Y; i < Bk; i += B_BLOCK_Y) {
      int r = k + i;
#pragma unroll
      for (int j = B_THREAD_X; j < Bn; j += B_BLOCK_X) {
        int c = c0 + j;

        Bs[i][j] = (r < K && c < N) ? B[r * N + c] : 0.f;
      }
    }

    __syncthreads();

    /* ------ 计算 tileA * tileB ------ */
    // p-Loop，先循环 k 维度，按向量外积的方式计算
#pragma unroll
    for (int p = 0; p < Bk; ++p) {
      // 向量化访存，存储 A 中列向量到 regA
#pragma unroll
      for (int i = 0; i < Tm / 4; ++i) {
        int r = (C_THREAD_Y + i * C_BLOCK_Y) * 4;
        FLOAT4(regA[i * 4]) = FLOAT4(As[p][r ^ (4 * p)]);
      }

      // 向量化访存，存储 B 中行向量到 regB
#pragma unroll
      for (int j = 0; j < Tn / 4; ++j) {
        int c = (C_THREAD_X + j * C_BLOCK_X) * 4;
        FLOAT4(regB[j * 4]) = FLOAT4(Bs[p][c]);
      }

      // 计算 regA 与 regB 的向量外积
#pragma unroll
      for (int i = 0; i < Tm; ++i) {
#pragma unroll
        for (int j = 0; j < Tn; ++j) { Ct[i][j] += regA[i] * regB[j]; }
      }
    }

    __syncthreads();
  }

  // 将 Ct 写入 C
#pragma unroll
  for (int i = 0; i < Tm; ++i) {
    int r = r0 + 4 * C_THREAD_Y + i / 4 * 4 * C_BLOCK_Y + i % 4;
#pragma unroll
    for (int j = 0; j < Tn; ++j) {
      int c = c0 + 4 * C_THREAD_X + j / 4 * 4 * C_BLOCK_X + j % 4;

      if (r < M && c < N) { C[r * N + c] = Ct[i][j]; }
    }
  }
}

// z-order 排布线程
template<int Bm = 128, int Bn = 128, int Bk = 8, int blockSize = 256, int A_BLOCK_X = 8,
         int B_BLOCK_X = 32, int C_BLOCK_X = 16, int C_WARP_X = 8, int C_WARP_Y = 4>
__global__ void zorderGEMM(float* A, float* B, float* C, const int M, const int K, const int N) {
  __shared__ float As[Bk][Bm];  // 存储转置后的 tileA
  __shared__ float Bs[Bk][Bn];  // 存储 tileB

  // 计算 block 负责的 tileC 左上角元素的行列坐标
  int r0 = blockIdx.y * Bm;
  int c0 = blockIdx.x * Bn;

  // 当前 thread 的编号（默认为一维 block 配置）
  int tid = threadIdx.x;

  /*------ tileA ------*/
  // 写入 A tile 时，block 中 thread 排布尺寸为 (A_BLOCK_X, blockSize / A_BLOCK_X) = (8, 32)
  constexpr int A_BLOCK_Y = blockSize / A_BLOCK_X;

  // 对于 tid 号线程，其位于 blockA 中的行列坐标为 (tid / A_BLOCK_X, tid % A_BLOCK_X)
  int A_THREAD_Y = tid / A_BLOCK_X;
  int A_THREAD_X = tid % A_BLOCK_X;

  /*------ tileB ------*/
  // 写入 B tile 时，block 中 thread 排布尺寸为 (B_BLOCK_X, blockSize / B_BLOCK_X) = (32, 8)
  constexpr int B_BLOCK_Y = blockSize / B_BLOCK_X;

  // 对于 tid 号线程，其位于 blockB 中的行列坐标为 (tid / B_BLOCK_X, tid % B_BLOCK_X)
  int B_THREAD_Y = tid / B_BLOCK_X;
  int B_THREAD_X = tid % B_BLOCK_X;

  /*------ tileC ------*/
  constexpr int C_BLOCK_Y = blockSize / C_BLOCK_X;

  // 按 8*4 排列 warp

  // 计算当前 thread 属于哪个 warp，第几个 lane
  int warpId = tid / WARP_SIZE;
  int laneId = tid % WARP_SIZE;
  // 计算总共有几行几列 warp
  constexpr int C_WARP_DIM_X = C_BLOCK_X / C_WARP_X;

  // 计算当前 thread 所在 warp 在 block 中的 x, y 坐标
  int warpX = warpId % C_WARP_DIM_X;
  int warpY = warpId / C_WARP_DIM_X;

  // z-order 排布，计算当前 thread 在 warp 中的 x, y 坐标
  int laneY = laneId % 2 + laneId / 16 * 2;
  int laneX = laneId % 16 / 2;

  // 当前 thread 在 blockC 中的行列坐标为 (warpY * C_WARP_Y + laneY, warpX * C_WARP_X + laneX)
  int C_THREAD_Y = warpY * C_WARP_Y + laneY;
  int C_THREAD_X = warpX * C_WARP_X + laneX;

  // 每个 thread 负责 Tm * Tn 个元素计算
  constexpr int Tm = Bm / C_BLOCK_Y;
  constexpr int Tn = Bn / C_BLOCK_X;
  float Ct[Tm][Tn] = {0.0};

  // 存储 A 中列向量和 B 中行向量
  float regA[Tm] = {0.0f};
  float regB[Tn] = {0.0f};

  // K- Loop
  for (int k = 0; k < K; k += Bk) {
    /* ------ 读取 global memory，存入 shared memory ------ */
    // 使用跨步循环，行方向的 stride 为 A_BLOCK_Y, 列方向的 stride 为 A_BLOCK_X
#pragma unroll
    for (int i = A_THREAD_Y; i < Bm; i += A_BLOCK_Y) {
      int r = r0 + i;
#pragma unroll
      for (int j = A_THREAD_X; j < Bk; j += A_BLOCK_X) {
        int c = k + j;
        As[j][i ^ (4 * j)] = (r < M && c < K) ? A[r * K + c] : 0.f;  // 转置
      }
    }

    // 使用跨步循环，行方向的 stride 为 B_BLOCK_Y, 列方向的 stride 为 B_BLOCK_X
#pragma unroll
    for (int i = B_THREAD_Y; i < Bk; i += B_BLOCK_Y) {
      int r = k + i;
#pragma unroll
      for (int j = B_THREAD_X; j < Bn; j += B_BLOCK_X) {
        int c = c0 + j;

        Bs[i][j] = (r < K && c < N) ? B[r * N + c] : 0.f;
      }
    }

    __syncthreads();

    /* ------ 计算 tileA * tileB ------ */
    // p-Loop，先循环 k 维度，按向量外积的方式计算
#pragma unroll
    for (int p = 0; p < Bk; ++p) {
      // 向量化访存，存储 A 中列向量到 regA
#pragma unroll
      for (int i = 0; i < Tm / 4; ++i) {
        int r = (C_THREAD_Y + i * C_BLOCK_Y) * 4;
        FLOAT4(regA[i * 4]) = FLOAT4(As[p][r ^ (4 * p)]);
      }

      // 向量化访存，存储 B 中行向量到 regB
#pragma unroll
      for (int j = 0; j < Tn / 4; ++j) {
        int c = (C_THREAD_X + j * C_BLOCK_X) * 4;
        FLOAT4(regB[j * 4]) = FLOAT4(Bs[p][c]);
      }

      // 计算 regA 与 regB 的向量外积
#pragma unroll
      for (int i = 0; i < Tm; ++i) {
#pragma unroll
        for (int j = 0; j < Tn; ++j) { Ct[i][j] += regA[i] * regB[j]; }
      }
    }

    __syncthreads();
  }

  // 将 Ct 写入 C
#pragma unroll
  for (int i = 0; i < Tm; ++i) {
    int r = r0 + 4 * C_THREAD_Y + i / 4 * 4 * C_BLOCK_Y + i % 4;
#pragma unroll
    for (int j = 0; j < Tn; ++j) {
      int c = c0 + 4 * C_THREAD_X + j / 4 * 4 * C_BLOCK_X + j % 4;

      if (r < M && c < N) { C[r * N + c] = Ct[i][j]; }
    }
  }
}

// 写法优化
template<int Bm = 128, int Bn = 128, int Bk = 8, int blockSize = 256, int A_BLOCK_X = 8,
         int A_BLOCK_Y = 32, int B_BLOCK_X = 32, int C_BLOCK_X = 16, int C_BLOCK_Y = 16,
         int C_WARP_X = 8, int C_WARP_Y = 4, int C_WARP_DIM_X = 2, int Tm = 8, int Tn = 8>
__global__ void optimGEMM(float* A, float* B, float* C, const int M, const int K, const int N) {
  __shared__ float As[Bk][Bm];  // 存储转置后的 tileA
  __shared__ float Bs[Bk][Bn];  // 存储 tileB

  // 计算 block 负责的 tileC 左上角元素的行列坐标
  int r0 = blockIdx.y * Bm;
  int c0 = blockIdx.x * Bn;

  // 当前 thread 的编号（默认为一维 block 配置）
  int tid = threadIdx.x;

  /*------ tileA ------*/
  // 对于 tid 号线程，其位于 blockA 中的行列坐标为 (tid / A_BLOCK_X, tid % A_BLOCK_X)
  int A_THREAD_Y = tid / A_BLOCK_X;
  int A_THREAD_X = tid & (A_BLOCK_X - 1);

  /*------ tileB ------*/
  // 对于 tid 号线程，其位于 blockB 中的行列坐标为 (tid / B_BLOCK_X, tid % B_BLOCK_X)
  int B_THREAD_Y = tid / B_BLOCK_X;
  int B_THREAD_X = tid & (B_BLOCK_X - 1);

  /*------ tileC ------*/

  // 按 8*4 排列 warp

  // 计算当前 thread 属于哪个 warp，第几个 lane
  int warpId = tid / WARP_SIZE;
  int laneId = tid & (WARP_SIZE - 1);

  // 计算当前 thread 所在 warp 在 block 中的 x, y 坐标
  int warpX = warpId & (C_WARP_DIM_X - 1);
  int warpY = warpId / C_WARP_DIM_X;

  // z-order 排布，计算当前 thread 在 warp 中的 x, y 坐标
  int laneY = (laneId & 1) + ((laneId >> 4) << 1);
  int laneX = (laneId & 15) >> 1;

  // 当前 thread 在 blockC 中的行列坐标为 (warpY * C_WARP_Y + laneY, warpX * C_WARP_X + laneX)
  int C_THREAD_Y = warpY * C_WARP_Y + laneY;
  int C_THREAD_X = warpX * C_WARP_X + laneX;

  // 每个 thread 负责 Tm * Tn 个元素计算
  float Ct[Tm][Tn] = {0.0};

  // 存储 A 中列向量和 B 中行向量
  float regA[Tm] = {0.0f};
  float regB[Tn] = {0.0f};

  // K- Loop
  for (int k = 0; k < K; k += Bk) {
    /* ------ 读取 global memory，存入 shared memory ------ */
    // 使用跨步循环，行方向的 stride 为 A_BLOCK_Y, 列方向的 stride 为 A_BLOCK_X
    int c = k + A_THREAD_X;
#pragma unroll
    for (int i = 0; i < Bm; i += A_BLOCK_Y) {
      int r = r0 + i + A_THREAD_Y;
      As[A_THREAD_X][(i + A_THREAD_Y) ^ (A_THREAD_X << 2)] =
          (r < M && c < K) ? A[r * K + c] : 0.f;  // 转置
    }

    // 使用跨步循环，行方向的 stride 为 B_BLOCK_Y, 列方向的 stride 为 B_BLOCK_X
    int r = k + B_THREAD_Y;
#pragma unroll
    for (int j = 0; j < Bn; j += B_BLOCK_X) {
      c = c0 + j + B_THREAD_X;
      Bs[B_THREAD_Y][j + B_THREAD_X] = (r < K && c < N) ? B[r * N + c] : 0.f;
    }

    __syncthreads();

    /* ------ 计算 tileA * tileB ------ */
    // p-Loop，先循环 k 维度，按向量外积的方式计算
#pragma unroll
    for (int p = 0; p < Bk; ++p) {
      // 向量化访存，存储 A 中列向量到 regA
#pragma unroll
      for (int i = 0; i < (Tm >> 2); ++i) {
        int r = (C_THREAD_Y + i * C_BLOCK_Y) << 2;
        FLOAT4(regA[i << 2]) = FLOAT4(As[p][r ^ (p << 2)]);
      }

      // 向量化访存，存储 B 中行向量到 regB
#pragma unroll
      for (int j = 0; j < (Tn >> 2); ++j) {
        int c = (C_THREAD_X + j * C_BLOCK_X) << 2;
        FLOAT4(regB[j << 2]) = FLOAT4(Bs[p][c]);
      }

      // 计算 regA 与 regB 的向量外积
#pragma unroll
      for (int i = 0; i < Tm; ++i) {
#pragma unroll
        for (int j = 0; j < Tn; ++j) { Ct[i][j] += regA[i] * regB[j]; }
      }
    }

    __syncthreads();
  }

  // 将 Ct 写入 C
#pragma unroll
  for (int i = 0; i < Tm; ++i) {
    int r = r0 + (C_THREAD_Y << 2) + ((i >> 2) << 2) * C_BLOCK_Y + (i & 3);
#pragma unroll
    for (int j = 0; j < Tn; ++j) {
      int c = c0 + (C_THREAD_X << 2) + ((j >> 2) << 2) * C_BLOCK_X + (j & 3);

      if (r < M && c < N) { C[r * N + c] = Ct[i][j]; }
    }
  }
}

// double buffering
template<int Bm = 128, int Bn = 128, int Bk = 8, int blockSize = 256, int A_BLOCK_X = 8,
         int A_BLOCK_Y = 32, int B_BLOCK_X = 32, int C_BLOCK_X = 16, int C_BLOCK_Y = 16,
         int C_WARP_X = 8, int C_WARP_Y = 4, int C_WARP_DIM_X = 2, int Tm = 8, int Tn = 8>
__global__ void doublebufferingGEMM(float* A, float* B, float* C, const int M, const int K,
                                    const int N) {
  __shared__ float As[2][Bk][Bm];  // 存储转置后的 tileA
  __shared__ float Bs[2][Bk][Bn];  // 存储 tileB

  // 计算 block 负责的 tileC 左上角元素的行列坐标
  int r0 = blockIdx.y * Bm;
  int c0 = blockIdx.x * Bn;

  // 当前 thread 的编号（默认为一维 block 配置）
  int tid = threadIdx.x;

  /*------ tileA ------*/
  // 对于 tid 号线程，其位于 blockA 中的行列坐标为 (tid / A_BLOCK_X, tid % A_BLOCK_X)
  int A_THREAD_Y = tid / A_BLOCK_X;
  int A_THREAD_X = tid & (A_BLOCK_X - 1);

  /*------ tileB ------*/
  // 对于 tid 号线程，其位于 blockB 中的行列坐标为 (tid / B_BLOCK_X, tid % B_BLOCK_X)
  int B_THREAD_Y = tid / B_BLOCK_X;
  int B_THREAD_X = tid & (B_BLOCK_X - 1);

  // 按 8*4 排列 warp

  // 计算当前 thread 属于哪个 warp，第几个 lane
  int warpId = tid / WARP_SIZE;
  int laneId = tid & (WARP_SIZE - 1);

  // 计算当前 thread 所在 warp 在 block 中的 x, y 坐标
  int warpX = warpId & (C_WARP_DIM_X - 1);
  int warpY = warpId / C_WARP_DIM_X;

  // z-order 排布，计算当前 thread 在 warp 中的 x, y 坐标
  int laneY = (laneId & 1) + ((laneId >> 4) << 1);
  int laneX = (laneId & 15) >> 1;

  // 当前 thread 在 blockC 中的行列坐标为 (warpY * C_WARP_Y + laneY, warpX * C_WARP_X + laneX)
  int C_THREAD_Y = warpY * C_WARP_Y + laneY;
  int C_THREAD_X = warpX * C_WARP_X + laneX;

  // 每个 thread 负责 Tm * Tn 个元素计算
  float Ct[Tm][Tn] = {0.0};

  // 存储 A 中列向量和 B 中行向量
  float regA[2][Tm] = {0.0f};
  float regB[2][Tn] = {0.0f};

  int buffer_id = 0;

  // 预先读取 k = 0 数据
#pragma unroll
  for (int i = 0; i < Bm; i += A_BLOCK_Y) {
    int r = r0 + i + A_THREAD_Y;
    As[0][A_THREAD_X][(i + A_THREAD_Y) ^ (A_THREAD_X << 2)] =
        (r < M && A_THREAD_X < K) ? A[r * K + A_THREAD_X] : 0.0f;
  }

#pragma unroll
  for (int j = 0; j < Bm; j += B_BLOCK_X) {
    int c = c0 + j + B_THREAD_X;
    Bs[0][B_THREAD_Y][j + B_THREAD_X] = (B_THREAD_Y < K && c < N) ? B[B_THREAD_Y * N + c] : 0.0f;
  }

  __syncthreads();

  // K-Loop，跳过 k = 0 并且最后增加一次循环
  for (int k = Bk; k < K + Bk; k += Bk) {
    // 计算阶段：计算第 k - 1 次循环结果
    // 双缓冲 p-Loop
#pragma unroll
    for (int p = 0; p < Bk + 1; ++p) {
      // 计算阶段：计算第 p - 1 次循环的结果
      if (p > 0) {
#pragma unroll
        for (int i = 0; i < Tm; ++i) {
#pragma unroll
          for (int j = 0; j < Tn; ++j) { Ct[i][j] += regA[(p - 1) & 1][i] * regB[(p - 1) & 1][j]; }
        }
      }

      // 预取阶段：读取第 p 次循环的数据
      if (p < Bk) {
        // 读取 regA
#pragma unroll
        for (int i = 0; i < (Tm >> 2); ++i) {
          int r = (C_THREAD_Y + i * C_BLOCK_Y) << 2;
          FLOAT4(regA[p & 1][i << 2]) = FLOAT4(As[buffer_id][p][r ^ (p << 2)]);
        }

        // 读取 regB
#pragma unroll
        for (int j = 0; j < (Tn >> 2); ++j) {
          int c = (C_THREAD_X + j * C_BLOCK_X) << 2;
          FLOAT4(regB[p & 1][j << 2]) = FLOAT4(Bs[buffer_id][p][c]);
        }
      }
    }

    // 预取阶段：预取第 k 次循环数据
    if (k < K) {
      // 读取 tileA
      int c = k + A_THREAD_X;
#pragma unroll
      for (int i = 0; i < Bm; i += A_BLOCK_Y) {
        int r = r0 + i + A_THREAD_Y;
        As[buffer_id ^ 1][A_THREAD_X][(i + A_THREAD_Y) ^ (A_THREAD_X << 2)] =
            (r < M && c < K) ? A[r * K + c] : 0.f;  // 转置
      }

      // 读取 tileA
      int r = k + B_THREAD_Y;
#pragma unroll
      for (int j = 0; j < Bn; j += B_BLOCK_X) {
        c = c0 + j + B_THREAD_X;
        Bs[buffer_id ^ 1][B_THREAD_Y][j + B_THREAD_X] = (r < K && c < N) ? B[r * N + c] : 0.f;
      }

      __syncthreads();
    }

    buffer_id ^= 1;  // 切换缓冲区
  }

  // 将 Ct 写入 C
#pragma unroll
  for (int i = 0; i < Tm; ++i) {
    int r = r0 + (C_THREAD_Y << 2) + ((i >> 2) << 2) * C_BLOCK_Y + (i & 3);
#pragma unroll
    for (int j = 0; j < Tn; ++j) {
      int c = c0 + (C_THREAD_X << 2) + ((j >> 2) << 2) * C_BLOCK_X + (j & 3);

      if (r < M && c < N) { C[r * N + c] = Ct[i][j]; }
    }
  }
}

void call_gemm_host(float* A, float* B, float* out, const int M, const int K, const int N) {
  for (int r = 0; r < M; ++r) {
    for (int c = 0; c < N; ++c) {
      float value = 0.0f;
      for (int k = 0; k < K; ++k) { value += A[r * K + k] * B[k * N + c]; }
      out[r * N + c] = value;
    }
  }
}

void call_gemm_device(int whichKernel, float* A, float* B, float* out, const int M, const int K,
                      const int N) {
  void (*kernel)(float*, float*, float*, const int, const int, const int);
  const char* kernelName;
  dim3 block;
  dim3 grid;
  const int Bm = 128;
  const int Bn = 128;

  switch (whichKernel) {
    case 0:
      block = dim3(16, 16);
      grid = dim3(CEIL(N, block.x), CEIL(M, block.y));
      kernel = naiveGEMM;
      kernelName = "naiveGEMM";
      break;
    case 1:
      block = 256;
      grid = dim3(CEIL(N, Bn), CEIL(M, Bm));
      kernel = blockTileGEMM;
      kernelName = "blockTileGEMM";
      break;
    case 2:
      block = 256;
      grid = dim3(CEIL(N, Bn), CEIL(M, Bm));
      kernel = threadTileGEMM;
      kernelName = "threadTileGEMM";
      break;
    case 3:
      block = 256;
      grid = dim3(CEIL(N, Bn), CEIL(M, Bm));
      kernel = warpGEMM;
      kernelName = "warpGEMM";
      break;
    case 4:
      block = 256;
      grid = dim3(CEIL(N, Bn), CEIL(M, Bm));
      kernel = float4GEMM;
      kernelName = "float4GEMM";
      break;
    case 5:
      block = 256;
      grid = dim3(CEIL(N, Bn), CEIL(M, Bm));
      kernel = float4GEMMnoBC;
      kernelName = "float4GEMMnoBC";
      break;
    case 6:
      block = 256;
      grid = dim3(CEIL(N, Bn), CEIL(M, Bm));
      kernel = zorderGEMM;
      kernelName = "zorderGEMM";
      break;
    case 7:
      block = 256;
      grid = dim3(CEIL(N, Bn), CEIL(M, Bm));
      kernel = optimGEMM;
      kernelName = "optimGEMM";
      break;
    case 8:
      block = 256;
      grid = dim3(CEIL(N, Bn), CEIL(M, Bm));
      kernel = doublebufferingGEMM;
      kernelName = "doublebufferingGEMM";
      break;
    default: break;
  }

  printf("kernel: [%s], grid: [%d, %d], block: [%d, %d]\n", kernelName, grid.x, grid.y, block.x,
         block.y);
  kernel<<<grid, block>>>(A, B, out, M, K, N);
}
