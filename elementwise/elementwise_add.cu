#include "helper_cuda.cuh"

#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

__global__ void elementwise_add(float* A, float* B, float* C, const int N) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < N) { C[tid] = A[tid] + B[tid]; }
}

__global__ void elementwise_add_gsl(float* A, float* B, float* C, const int N) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < N; i += blockDim.x * gridDim.x) {
    C[i] = A[i] + B[i];
  }
}

__global__ void elementwise_add_vec4(float* A, float* B, float* C, const int N) {
  int index = 4 * (threadIdx.x + blockIdx.x * blockDim.x);

  if (index > N) { return; }

  if (index <= N - 4) {
    float4 a = FLOAT4(A[index]);
    float4 b = FLOAT4(B[index]);
    float4 c;

    c.x = a.x + b.x;
    c.y = a.y + b.y;
    c.z = a.z + b.z;
    c.w = a.w + b.w;
    FLOAT4(C[index]) = c;
  } else {
#pragma unroll
    for (int i = index; i < N; ++i) { C[i] = A[i] + B[i]; }
  }
}

__global__ void elementwise_add_vec4_gsl(float* A, float* B, float* C, const int N) {
  int index = 4 * (threadIdx.x + blockIdx.x * blockDim.x);

  for (int i = index; i < N; i += blockDim.x * gridDim.x * 4) {
    if (i < N - 4) {
      float4 a = FLOAT4(A[i]);
      float4 b = FLOAT4(B[i]);
      float4 c;

      c.x = a.x + b.x;
      c.y = a.y + b.y;
      c.z = a.z + b.z;
      c.w = a.w + b.w;
      FLOAT4(C[i]) = c;
    } else {
#pragma unroll
      for (int j = i; j < N; ++j) { C[j] = A[j] + B[j]; }
    }
  }
}

void call_add_f32_host(float* A, float* B, float* C, const int N) {
  for (int i = 0; i < N; ++i) { C[i] = A[i] + B[i]; }
}

void call_add_f32_device(int whichKernel, int blockSize, float* d_A, float* d_B, float* d_C,
                         const int N) {
  void (*kernel)(float*, float*, float*, const int);
  const char* kernelName = "";
  int gridSize = (N + blockSize - 1) / blockSize;

  switch (whichKernel) {
    case 0:
      kernel = elementwise_add;
      kernelName = "elementwise_add";
      break;
    case 1:
      kernel = elementwise_add_gsl;
      kernelName = "elementwise_add_gsl";
      break;
    case 2:
      gridSize = (N + blockSize * 4 - 1) / (blockSize * 4);
      kernel = elementwise_add_vec4;
      kernelName = "elementwise_add_vec4";
      break;
    case 3:
      gridSize = (N + blockSize * 4 - 1) / (blockSize * 4);
      kernel = elementwise_add_vec4_gsl;
      kernelName = "elementwise_add_vec4_gsl";
      break;
    default: break;
  }

  printf("kernel: [%s], grid: [%d], block: [%d], N: [%d] \n", kernelName, gridSize, blockSize, N);
  kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
}