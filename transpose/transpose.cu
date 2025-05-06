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
  // (r0, c0) 表示 tile 内左上角元素在 matrixA 中的坐标
  int r0 = blockIdx.y * Bm;
  int c0 = blockIdx.x * Bn;

  // thread y 方向负责：矩阵 A 的行，shared memory 的行
  // thread x 方向负责：矩阵 A 的列，shared memory 的列
  // shared memory 中的元素 tile[y][x] = A[r0 + y, c0 + x]
#pragma unroll
  for (int y = threadIdx.y; y < Bm; y += blockDim.y) {  // 在 y 方向，每次跨度为 blockDim.y
    int r = r0 + y;
    if (r >= M) break;

#pragma unroll
    for (int x = threadIdx.x; x < Bn; x += blockDim.x) {  // 在 x 方向，每次跨度为 blockDim.x
      int c = c0 + x;
      if (c < N) {
        tile[y][x] = A[r * N + c];  // 将 A[r0 + y, c0 + x] 写入 tile[y][x]
      }
    }
  }

  __syncthreads();  // 同步线程块

/* -------- 写入阶段 -------- */
// (c0, r0) 表示 tile 内左上角元素在 matrixB 中的坐标
// thread y 方向负责：矩阵 B 的行，shared memory 的列
// thread x 方向负责：矩阵 B 的列，shared memory 的行
// shared memory 中的元素 tile[x][y] = B[c0 + y, r0 + x]
#pragma unroll
  for (int y = threadIdx.y; y < Bn; y += blockDim.y) {  // 在 y 方向，每次跨度为 blockDim.y
    int c = c0 + y;
    if (c >= N) break;

#pragma unroll
    for (int x = threadIdx.x; x < Bm; x += blockDim.x) {  // 在 x 方向，每次跨度为 blockDim.x
      int r = r0 + x;
      if (r < M) { B[c * M + r] = tile[x][y]; }  // 将 tile[x][y] 写入 B[c0 + y, r0 + x]
    }
  }
}

template<int Bm, int Bn>
__global__ void transposeSharedPadding(float* A, float* B, const int M, const int N) {
  __shared__ float tile[Bm][Bn + 1];  // padding

  /* -------- 读取阶段 -------- */
  // (r0, c0) 表示 tile 内左上角元素在 matrixA 中的坐标
  int r0 = blockIdx.y * Bm;
  int c0 = blockIdx.x * Bn;

  // thread y 方向负责：矩阵 A 的行，shared memory 的行
  // thread x 方向负责：矩阵 A 的列，shared memory 的列
  // shared memory 中的元素 tile[y][x] = A[r0 + y, c0 + x]
#pragma unroll
  for (int y = threadIdx.y; y < Bm; y += blockDim.y) {  // 在 y 方向，每次跨度为 blockDim.y
    int r = r0 + y;
    if (r >= M) break;

#pragma unroll
    for (int x = threadIdx.x; x < Bn; x += blockDim.x) {  // 在 x 方向，每次跨度为 blockDim.x
      int c = c0 + x;
      if (c < N) {
        tile[y][x] = A[r * N + c];  // 将 A[r0 + y, c0 + x] 写入 tile[y][x]
      }
    }
  }

  __syncthreads();  // 同步线程块

/* -------- 写入阶段 -------- */
// (c0, r0) 表示 tile 内左上角元素在 matrixB 中的坐标
// thread y 方向负责：矩阵 B 的行，shared memory 的列
// thread x 方向负责：矩阵 B 的列，shared memory 的行
// shared memory 中的元素 tile[x][y] = B[c0 + y, r0 + x]
#pragma unroll
  for (int y = threadIdx.y; y < Bn; y += blockDim.y) {  // 在 y 方向，每次跨度为 blockDim.y
    int c = c0 + y;
    if (c >= N) break;

#pragma unroll
    for (int x = threadIdx.x; x < Bm; x += blockDim.x) {  // 在 x 方向，每次跨度为 blockDim.x
      int r = r0 + x;
      if (r < M) { B[c * M + r] = tile[x][y]; }  // 将 tile[x][y] 写入 B[c0 + y, r0 + x]
    }
  }
}

template<int Bm, int Bn>
__global__ void transposeSharedSwizzling(float* A, float* B, const int M, const int N) {
  __shared__ float tile[Bm][Bn];

  /* -------- 读取阶段 -------- */
  // (r0, c0) 表示 tile 内左上角元素在 matrixA 中的坐标
  int r0 = blockIdx.y * Bm;
  int c0 = blockIdx.x * Bn;

  // thread y 方向负责：矩阵 A 的行，shared memory 的行
  // thread x 方向负责：矩阵 A 的列，shared memory 的列
  // shared memory 中的元素 tile[y][x] = A[r0 + y, c0 + x]
#pragma unroll
  for (int y = threadIdx.y; y < Bm; y += blockDim.y) {  // 在 y 方向，每次跨度为 blockDim.y
    int r = r0 + y;
    if (r >= M) break;

#pragma unroll
    for (int x = threadIdx.x; x < Bn; x += blockDim.x) {  // 在 x 方向，每次跨度为 blockDim.x
      int c = c0 + x;
      if (c < N) {
        tile[y][x ^ y] = A[r * N + c];  // 将 A[r0 + y, c0 + x] 写入 tile[y][x ^ y]
      }
    }
  }

  __syncthreads();  // 同步线程块

/* -------- 写入阶段 -------- */
// (c0, r0) 表示 tile 内左上角元素在 matrixB 中的坐标
// thread y 方向负责：矩阵 B 的行，shared memory 的列
// thread x 方向负责：矩阵 B 的列，shared memory 的行
// shared memory 中的元素 tile[x][y] = B[c0 + y, r0 + x]
#pragma unroll
  for (int y = threadIdx.y; y < Bn; y += blockDim.y) {  // 在 y 方向，每次跨度为 blockDim.y
    int c = c0 + y;
    if (c >= N) break;

#pragma unroll
    for (int x = threadIdx.x; x < Bm; x += blockDim.x) {  // 在 x 方向，每次跨度为 blockDim.x
      int r = r0 + x;
      if (r < M) { B[c * M + r] = tile[x][x ^ y]; }  // 将 tile[x][x ^ y] 写入 B[c0 + y, r0 + x]
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
      grid = dim3(CEIL(M, 32), CEIL(N, 32));
      break;
    case 3:
      kernel = transposeShared<Bm, Bn>;
      kernelName = "transposeShared";
      block = dim3(blockDimX, 256 / blockDimX);
      grid = dim3(CEIL(N, Bn), CEIL(M, Bm));
      break;
    case 4:
      kernel = transposeSharedPadding<Bm, Bn>;
      kernelName = "transposeSharedPadding";
      block = dim3(blockDimX, 256 / blockDimX);
      grid = dim3(CEIL(N, Bn), CEIL(M, Bm));
      break;
    case 5:
      kernel = transposeSharedSwizzling<Bm, Bn>;
      kernelName = "transposeSharedSwizzling";
      block = dim3(blockDimX, 256 / blockDimX);
      grid = dim3(CEIL(N, Bn), CEIL(M, Bm));
      break;
    default: break;
  }

  printf("kernel: [%s], grid: <%d, %d>, block: <%d, %d>, M: [%d], N: [%d] \n", kernelName, grid.x,
         grid.y, block.x, block.y, M, N);

  kernel<<<grid, block>>>(A, B, M, N);
}