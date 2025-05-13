#include <stdio.h>

#define CEIL(a, b) (((a) + (b) - 1) / (b))

// version 0: 简单实现，存在线程束分化问题
__global__ void reduce0(float* d_A, const int N) {
  // 申请 shared memory，用于存放 block 负责的数据
  extern __shared__ float data[];

  // 读取数据到 shared memory
  int tid = threadIdx.x;
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  data[tid] = index < N ? d_A[index] : 0.f;

  // 同步，保证读取完成
  __syncthreads();

  // iter0 [s = 1] (t0, t2, t4, ...): t0 -> (0, 1) | t2 -> (2, 3) | t4 -> (4, 5) ...
  // iter1 [s = 2] (t0, t4, t8, ...): t0 -> (0, 2) | t4 -> (4, 6) | t8 -> (8, 10) ...
  for (int s = 1; s < blockDim.x; s <<= 1) {
    // 负责执行运算的 threadIdx 为 0, 2s, 4s, 8s, ...
    if ((tid % (s * 2)) == 0) { data[tid] += data[tid + s]; }

    // 进行同步，保证计算完成
    __syncthreads();
  }

  // block 负责数据的求和结果存储在 data[0]，由 0 号 thread 写入 d_A[blockIdx.x] 中
  if (tid == 0) { d_A[blockIdx.x] = data[0]; }
}

// version 0.5: 优化取余运算
__global__ void reduce0_5(float* d_A, const int N) {
  // 申请 shared memory，用于存放 block 负责的数据
  extern __shared__ float data[];

  // 读取数据到 shared memory
  int tid = threadIdx.x;
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  data[tid] = index < N ? d_A[index] : 0.f;

  // 同步，保证读取完成
  __syncthreads();

  // iter0 [s = 1] (t0, t2, t4, ...): t0 -> (0, 1) | t2 -> (2, 3) | t4 -> (4, 5) ...
  // iter1 [s = 2] (t0, t4, t8, ...): t0 -> (0, 2) | t4 -> (4, 6) | t8 -> (8, 10) ...
  for (int s = 1; s < blockDim.x; s <<= 1) {
    // 负责执行运算的 threadIdx 为 0, 2s, 4s, 8s, ...
    if ((tid & (s * 2 - 1)) == 0) { data[tid] += data[tid + s]; }

    // 进行同步，保证计算完成
    __syncthreads();
  }

  // block 负责数据的求和结果存储在 data[0]，由 0 号 thread 写入 d_A[blockIdx.x] 中
  if (tid == 0) { d_A[blockIdx.x] = data[0]; }
}

// version1: 使用连续 thread 负责计算，解决线程束分化，存在 bank conflicts
__global__ void reduce1(float* d_A, const int N) {
  extern __shared__ float data[];

  int tid = threadIdx.x;
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  data[tid] = i < N ? d_A[i] : 0.f;

  __syncthreads();

  // iter0 [s = 1] (t0 ~ tN/2): t0 -> (0, 1) | t1 -> (2, 3) | t2 -> (4, 5) ...
  // iter1 [s = 2] (t0 ~ tN/4): t0 -> (0, 2) | t1 -> (4, 6) | t2 -> (8, 10) ...
  for (int s = 1; s < blockDim.x; s <<= 1) {
    int index = tid * s * 2;

    if (index < blockDim.x) { data[index] += data[index + s]; }

    __syncthreads();
  }

  if (tid == 0) { d_A[blockIdx.x] = data[0]; }
}

// version2: 步长从大到小变化，解决 bank conflicts
__global__ void reduce2(float* d_A, const int N) {
  extern __shared__ float data[];

  int tid = threadIdx.x;
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  data[tid] = i < N ? d_A[i] : 0.f;

  __syncthreads();

  // iter0 [s = N/2] (t0 ~ tN/2-1): t0 -> (0, N/2) | t1 -> (1, N/2 + 1) | t2 -> (2, N/2 + 2) ...
  // iter1 [s = N/4] (t0 ~ tN/4-1): t0 -> (0, N/4) | t1 -> (1, N/4 + 1) | t2 -> (2, N/4 + 2) ...
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (tid < s) { data[tid] += data[tid + s]; }
    __syncthreads();
  }

  if (tid == 0) { d_A[blockIdx.x] = data[0]; }
}

// version3: 读取数据到 shared memory 时，进行一次加法计算
__global__ void reduce3(float* d_A, const int N) {
  extern __shared__ float data[];

  int tid = threadIdx.x;
  int i = 2 * blockDim.x * blockIdx.x + threadIdx.x;  // 每个 block 负责 2x threads 个元素

  float sum = i < N ? d_A[i] : 0.f;
  if (i + blockDim.x < N) { sum += d_A[i + blockDim.x]; }

  data[tid] = sum;

  __syncthreads();

  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (tid < s) { data[tid] += data[tid + s]; }
    __syncthreads();
  }

  if (tid == 0) { d_A[blockIdx.x] = data[0]; }
}

// version4: 最后 32 个数使用 warp shuffle 完成求和
__global__ void reduce4(float* d_A, const int N) {
  extern __shared__ float data[];

  int tid = threadIdx.x;
  int i = 2 * blockDim.x * blockIdx.x + threadIdx.x;

  float sum = i < N ? d_A[i] : 0.f;
  if (i + blockDim.x < N) { sum += d_A[i + blockDim.x]; }

  data[tid] = sum;

  __syncthreads();

  for (int s = blockDim.x >> 1; s >= 32; s >>= 1) {
    if (tid < s) { data[tid] = sum = sum + data[tid + s]; }
    __syncthreads();
  }

  // warp reduce
  if (tid < 32) {
#pragma unroll
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
      sum += __shfl_xor_sync(0xffffffff, sum, offset, warpSize);
    }
  }

  if (tid == 0) { d_A[blockIdx.x] = sum; }
}

// version5: 展开循环
template<int blockSize>  // 将 blockSize 作为模板参数，可以在编译期确定其数值，进而优化 if 分支
__global__ void reduce5(float* d_A, const int N) {
  extern __shared__ float data[];

  int tid = threadIdx.x;
  int i = 2 * blockSize * blockIdx.x + threadIdx.x;

  float sum = i < N ? d_A[i] : 0;
  if (i + blockSize < N) sum += d_A[i + blockSize];

  data[tid] = sum;

  __syncthreads();

  if (blockSize >= 1024 && tid < 512) { data[tid] = sum = sum + data[tid + 512]; }
  __syncthreads();

  if (blockSize >= 512 && tid < 256) { data[tid] = sum = sum + data[tid + 256]; }
  __syncthreads();

  if (blockSize >= 256 && tid < 128) { data[tid] = sum = sum + data[tid + 128]; }
  __syncthreads();

  if (blockSize >= 128 && tid < 64) { data[tid] = sum = sum + data[tid + 64]; }
  __syncthreads();

  if (blockSize >= 64 && tid < 32) { data[tid] = sum = sum + data[tid + 32]; }
  __syncthreads();

  if (tid < 32) {
#pragma unroll
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
      sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }
  }

  if (tid == 0) { d_A[blockIdx.x] = sum; }
}

template<int blockSize>
__global__ void reduce6(float* d_A, const int N) {
  extern __shared__ float data[];

  int tid = threadIdx.x;
  int i = blockSize * blockIdx.x + threadIdx.x;

  float sum = 0.f;
  for (int index = i; index < N; index += blockSize * gridDim.x) { sum += d_A[index]; }

  data[tid] = sum;

  __syncthreads();

  if (blockSize >= 1024 && tid < 512) { data[tid] = sum = sum + data[tid + 512]; }
  __syncthreads();

  if (blockSize >= 512 && tid < 256) { data[tid] = sum = sum + data[tid + 256]; }
  __syncthreads();

  if (blockSize >= 256 && tid < 128) { data[tid] = sum = sum + data[tid + 128]; }
  __syncthreads();

  if (blockSize >= 128 && tid < 64) { data[tid] = sum = sum + data[tid + 64]; }
  __syncthreads();

  if (blockSize >= 64 && tid < 32) { data[tid] = sum = sum + data[tid + 32]; }
  __syncthreads();

  if (tid < 32) {
#pragma unroll
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
      sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }
  }

  if (tid == 0) { d_A[blockIdx.x] = sum; }
}

// version7: 在最开始写入 shared memory 之前，进行一次 warp reduction
template<int blockSize>
__global__ void reduce7(float* d_A, const int N) {
  extern __shared__ float data[];

  int num_warps = CEIL(blockSize, warpSize);
  int tid = threadIdx.x;
  int i = blockSize * blockIdx.x + threadIdx.x;

  float sum = 0.f;
  for (int index = i; index < N; index += blockSize * gridDim.x) { sum += d_A[index]; }

#pragma unroll
  for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
    sum += __shfl_xor_sync(0xffffffff, sum, offset);
  }

  if ((tid & (warpSize - 1)) == 0) { data[tid / warpSize] = sum; }

  __syncthreads();

  if (tid < 32) {
    sum = tid < num_warps ? data[tid] : 0.f;
#pragma unroll
    for (int offset = num_warps >> 1; offset > 0; offset >>= 1) {
      sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }
  }

  if (tid == 0) { d_A[blockIdx.x] = sum; }
}

// GPU 求和
void call_reduction_sum_device(int whichKernel, float* d_A, const int N) {
  void (*kernel)(float*, const int);
  const char* kernelName = "";
  const int block = 256;

  switch (whichKernel) {
    case 0:
      kernel = reduce0;
      kernelName = "reduce0";
      break;
    case -1:
      kernel = reduce0_5;
      kernelName = "reduce0_5";
      break;
    case 1:
      kernel = reduce1;
      kernelName = "reduce1";
      break;
    case 2:
      kernel = reduce2;
      kernelName = "reduce2";
      break;
    case 3:
      kernel = reduce3;
      kernelName = "reduce3";
      break;
    case 4:
      kernel = reduce4;
      kernelName = "reduce4";
      break;
    case 5:
      kernel = reduce5<block>;
      kernelName = "reduce5";
      break;
    case 6:
      kernel = reduce6<block>;
      kernelName = "reduce6";
      break;
    case 7:
      kernel = reduce7<block>;
      kernelName = "reduce7";
      break;
    default: break;
  }

  int sharedBytes = block * sizeof(float);
  if (whichKernel == 7) { sharedBytes = CEIL(block, 32) * sizeof(float); }
  int size = N;

  auto getGrid = [](int whichKernel, int size, int block) -> int {
    if (whichKernel < 3) {
      return CEIL(size, block);
    } else if (whichKernel < 6) {
      return CEIL(size, block * 2);
    } else {
      return CEIL(size, block * 4);
    }
  };

  while (size > 1) {
    int grid = getGrid(whichKernel, size, block);
    kernel<<<grid, block, sharedBytes>>>(d_A, size);
    printf("kernel: [%s], size: [%d], grid: [%d], block: [%d]\n", kernelName, size, grid, block);

    size = grid;
  }
}

// CPU 求和
float pairwise_sum(float* A, int start, int stop) {
  if (stop == start) { return A[start]; }

  int mid = (start + stop) / 2;
  return pairwise_sum(A, start, mid) + pairwise_sum(A, mid + 1, stop);
}

float call_reduction_sum_host(float* A, const int N) { return pairwise_sum(A, 0, N - 1); }