#include <stdio.h>

#define CEIL(a, b) ((a + b - 1) / (b))

// 朴素实现：每个 thread 负责一个元素的拷贝，按行读取，按行写入
__global__ void copy_naive_row(float* A, float* B, const int M, const int N) {
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;

  if (iy < M && ix < N) { B[iy * N + ix] = A[iy * N + ix]; }
}

// 朴素实现：每个 thread 负责一个元素的拷贝，按列读取，按列写入
__global__ void copy_naive_col(float* A, float* B, const int M, const int N) {
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;

  if (iy < N && ix < M) { B[ix * N + iy] = A[ix * N + iy]; }
}

// 每个 block 负责一个矩阵块的转置，一个 thread 负责一个元素
template<int Bm, int Bn>
__global__ void copy_tile(float* A, float* B, const int M, const int N) {
  __shared__ float tile[Bm][Bn];

  int tx = threadIdx.x, ty = threadIdx.y;
  int bx = blockIdx.x, by = blockIdx.y;

  int r = by * Bm + ty;
  int c = bx * Bn + tx;

  if (r < M && c < N) { tile[ty][tx] = A[r * N + c]; }

  __syncthreads();

  r = bx * Bn + ty;
  c = by * Bm + tx;
  if (r < N && c < M) { B[r * M + c] = tile[tx][ty]; }
}

template<int Bm, int Bn>
__global__ void copy_tile_v2(float* A, float* B, const int M, const int N) {
  __shared__ float tile[Bm][Bn];

  int tx = threadIdx.x, ty = threadIdx.y;
  int bx = blockIdx.x, by = blockIdx.y;

  int iterX = Bn / blockDim.x;
  int iterY = Bm / blockDim.y;

  for (int i = 0; i < iterY; ++i) {
    int r = by * Bm + ty + i * blockDim.y;
    if (r >= M) break;
    for (int j = 0; j < iterX; ++j) {
      int c = bx * Bn + tx + j * blockDim.x;
      if (c >= N) break;
      tile[ty + i * blockDim.y][tx + j * blockDim.x] = A[r * N + c];
    }
  }

  __syncthreads();

  for (int i = 0; i < iterY; ++i) {
    int r = bx * Bn + ty + i * blockDim.y;
    if (r >= N) break;
    for (int j = 0; j < iterX; ++j) {
      int c = by * Bm + tx + j * blockDim.x;
      if (c >= M) break;
      B[r * M + c] = tile[tx + j * blockDim.x][ty + i * blockDim.y];
    }
  }
}

template<int Bm, int Bn>
__global__ void copy_tile_v3(float* A, float* B, const int M, const int N) {
  __shared__ float tile[Bm][Bn + 1];

  int tx = threadIdx.x, ty = threadIdx.y;
  int bx = blockIdx.x, by = blockIdx.y;

  int iterX = Bn / blockDim.x;
  int iterY = Bm / blockDim.y;

  for (int i = 0; i < iterY; ++i) {
    int r = by * Bm + ty + i * blockDim.y;
    if (r >= M) break;
    for (int j = 0; j < iterX; ++j) {
      int c = bx * Bn + tx + j * blockDim.x;
      if (c >= N) break;
      tile[ty + i * blockDim.y][tx + j * blockDim.x] = A[r * N + c];
    }
  }

  __syncthreads();

  for (int i = 0; i < iterY; ++i) {
    int r = bx * Bn + ty + i * blockDim.y;
    if (r >= N) break;
    for (int j = 0; j < iterX; ++j) {
      int c = by * Bm + tx + j * blockDim.x;
      if (c >= M) break;
      B[r * M + c] = tile[tx + j * blockDim.x][ty + i * blockDim.y];
    }
  }
}

void call_copy_device(int whichKernel, float* A, float* B, const int M, const int N,
                      int blockDimX) {
  void (*kernel)(float*, float*, const int, const int) = nullptr;
  const char* kernelName = "";
  dim3 block, grid;
  const int Bm = 32, Bn = 32;

  switch (whichKernel) {
    case 0:
      kernel = copy_naive_row;
      kernelName = "copy_naive_row";
      block = dim3(blockDimX, 256 / blockDimX);
      grid = dim3(CEIL(N, block.x), CEIL(M, block.y));
      break;
    case 1:
      kernel = copy_naive_col;
      kernelName = "copy_naive_col";
      block = dim3(blockDimX, 256 / blockDimX);
      grid = dim3(CEIL(M, block.x), CEIL(N, block.y));
      break;
    case 2:
      kernel = copy_tile<Bm, Bn>;
      kernelName = "copy_tile";
      block = dim3(Bn, Bm);
      grid = dim3(CEIL(N, block.x), CEIL(M, block.y));
      break;
    case 3:
      kernel = copy_tile_v2<Bm, Bn>;
      kernelName = "copy_tile_v2";
      block = dim3(Bn, Bm / 4);
      grid = dim3(CEIL(N, Bn), CEIL(M, Bm));
      break;
    case 4:
      kernel = copy_tile_v3<Bm, Bn>;
      kernelName = "copy_tile_v3";
      block = dim3(Bn, Bm / 4);
      grid = dim3(CEIL(N, Bn), CEIL(M, Bm));
      break;
    default: break;
  }

  printf("kernel: [%s], grid: <%d, %d>, block: <%d, %d>, M: [%d], N: [%d] \n", kernelName, grid.x,
         grid.y, block.x, block.y, M, N);

  kernel<<<grid, block>>>(A, B, M, N);
}