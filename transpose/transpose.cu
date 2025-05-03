#include <stdio.h>

#define CEIL(a, b) ((a + b - 1) / (b))

// 朴素实现 NaiveRow：每个 thread 负责一个元素的转置，按行读取，按列写入
__global__ void transposeNaiveRow(float* A, float* B, const int M, const int N) {
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;

  if (iy < M && ix < N) { B[ix * M + iy] = A[iy * N + ix]; }
}

// 朴素实现 NaiveCol：每个 thread 负责一个元素的转置，按列读取，按行写入
__global__ void transposeNaiveCol(float* A, float* B, const int M, const int N) {
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;

  if (iy < N && ix < M) { B[iy * M + ix] = A[ix * N + iy]; }
}

// 每个 block 负责一个分块 tile，每个 thread 负责多个元素
template<int Bm, int Bn>
__global__ void transposeNaiveColNelements(float* A, float* B, const int M, const int N) {
  // (r0, c0) 表示 tile 内左上角元素的坐标
  int r0 = blockIdx.x * Bm;
  int c0 = blockIdx.y * Bn;

  // 循环中的 (x, y) 表示 thread 负责的元素在 tile 内的坐标 (以左上角为 (0, 0))
  // thread 负责的元素的实际坐标为: (r0 + x, c0 + y)
#pragma unroll
  for (int x = threadIdx.x; x < Bm; x += blockDim.x) {  // 在 x 方向，每次跨度为 blockDim.x
    int r = r0 + x;
    if (r >= M) break;

#pragma unroll
    for (int y = threadIdx.y; y < Bn; y += blockDim.y) {  // 在 y 方向，每次跨度为 blockDim.y
      int c = c0 + y;
      if (c < N) { B[c * M + r] = A[r * N + c]; }
    }
  }
}

template<int Bm, int Bn>
__global__ void transposeShared(float* A, float* B, const int M, const int N) {
  __shared__ float tile[Bm][Bn];

  /* -------- 读取阶段 -------- */
  // (r0, c0) 表示 matrixA 在 tile 内左上角元素的坐标
  int r0 = blockIdx.y * Bm;
  int c0 = blockIdx.x * Bn;

  // 循环中的 (y, x) 表示 thread 负责读取的 matrixA 元素在 tile 内的坐标 (以左上角为 (0, 0))
  // thread 负责的 matrixA 元素的实际坐标为: (r0 + y, c0 + x)
  for (int y = threadIdx.y; y < Bm; y += blockDim.y) {  // 在 y 方向，每次跨度为 blockDim.y
    int r = r0 + y;
    if (r >= M) break;

    for (int x = threadIdx.x; x < Bn; x += blockDim.x) {  // 在 x 方向，每次跨度为 blockDim.x
      int c = c0 + x;

      if (c < N) {
        tile[y][x] = A[r * N + c];  // 目标是将 A[r][c] 写入 tile[y][x]
      }
    }
  }

  __syncthreads();

  /* -------- 转置阶段: warp 按列读取 tile, 按行写入 matrixB -------- */
  for (int y = threadIdx.y; y < Bn; y += blockDim.y) {
    int c = c0 + y;
    if (c >= N) break;
    for (int x = threadIdx.x; x < Bm; x += blockDim.x) {
      int r = r0 + x;
      if (r < M) { B[c * M + r] = tile[x][y]; }  // 目标是将 tile[x][y] 写入 B[c][r]
    }
  }
}

template<int Bm, int Bn>
__global__ void transposeSharedNoBankConfilictsV1(float* A, float* B, const int M, const int N) {
  __shared__ float tile[Bm][Bn + 1];

  /* -------- 读取阶段: wrap 按行读取 matrixA, 按行写入 tile -------- */
  // (r0, c0) 表示 matrixA 在 tile 内左上角元素的坐标
  int r0 = blockIdx.y * Bm;
  int c0 = blockIdx.x * Bn;

  // 循环中的 (y, x) 表示 thread 负责读取的 matrixA 元素在 tile 内的坐标 (以左上角为 (0, 0))
  // thread 负责的 matrixA 元素的实际坐标为: (r0 + y, c0 + x)
  for (int y = threadIdx.y; y < Bm; y += blockDim.y) {  // 在 y 方向，每次跨度为 blockDim.y
    int r = r0 + y;
    if (r >= M) break;

    for (int x = threadIdx.x; x < Bn; x += blockDim.x) {  // 在 x 方向，每次跨度为 blockDim.x
      int c = c0 + x;

      if (c < N) {
        tile[y][x] = A[r * N + c];  // 目标是将 A[r][c] 写入 tile[y][x]
      }
    }
  }

  __syncthreads();

  /* -------- 转置阶段: warp 按列读取 tile, 按行写入 matrixB -------- */
  for (int y = threadIdx.y; y < Bn; y += blockDim.y) {
    int c = c0 + y;
    if (c >= N) break;
    for (int x = threadIdx.x; x < Bm; x += blockDim.x) {
      int r = r0 + x;
      if (r < M) { B[c * M + r] = tile[x][y]; }  // 目标是将 tile[x][y] 写入 B[c][r]
    }
  }
}

void call_transpose_host(float* A, float* B, const int M, const int N) {
  for (int r = 0; r < M; ++r) {
    for (int c = 0; c < N; ++c) { B[c * M + r] = A[r * N + c]; }
  }
}

void call_transpose_device(int whichKernel, float* A, float* B, const int M, const int N,
                           int blockDimX) {
  void (*kernel)(float*, float*, const int, const int) = nullptr;
  const char* kernelName = "";
  dim3 block, grid;
  const int Bm = 32, Bn = 32;

  switch (whichKernel) {
    case 0:
      kernel = transposeNaiveRow;
      kernelName = "transposeNaiveRow";
      block = dim3(blockDimX, 256 / blockDimX);
      grid = dim3(CEIL(N, block.x), CEIL(M, block.y));
      break;
    case 1:
      kernel = transposeNaiveCol;
      kernelName = "transposeNaiveCol";
      block = dim3(blockDimX, 256 / blockDimX);
      grid = dim3(CEIL(M, block.x), CEIL(N, block.y));
      break;
    case 2:
      kernel = transposeNaiveColNelements<32, 32>;
      kernelName = "transposeNaiveColNelements";
      block = dim3(blockDimX, 256 / blockDimX);
      grid = dim3(CEIL(N, Bn), CEIL(M, Bm));
      break;
    case 3:
      kernel = transposeShared<48, 32>;
      kernelName = "transposeShared";
      block = dim3(32, 8);
      grid = dim3(CEIL(N, 32), CEIL(M, 32));
      break;
    case 4:
      kernel = transposeSharedNoBankConfilictsV1<48, 32>;
      kernelName = "transposeSharedNoBankConfilictsV1";
      block = dim3(32, 8);
      grid = dim3(CEIL(N, 32), CEIL(M, 32));
      break;
    default: break;
  }

  printf("kernel: [%s], grid: <%d, %d>, block: <%d, %d>, M: [%d], N: [%d] \n", kernelName, grid.x,
         grid.y, block.x, block.y, M, N);

  kernel<<<grid, block>>>(A, B, M, N);
}