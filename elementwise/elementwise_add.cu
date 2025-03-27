#include "helper_cuda.cuh"

#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define BlockSize 256

__global__ void elementwise_add(float* A, float* B, float* C, const int N) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < N) {
    C[tid] = A[tid] + B[tid];
  }
}

__global__ void elementwise_add_gsl(float* A, float* B, float* C, const int N) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < N; i += blockDim.x * gridDim.x) {
    C[i] = A[i] + B[i];
  }
}

__global__ void elementwise_add_vec4(float* A, float* B, float* C, const int N) {
  int index = 4 * (threadIdx.x + blockIdx.x * blockDim.x);

  if (index > N) {
    return;
  }

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
    for (int i = index; i < N; ++i) {
      C[i] = A[i] + B[i];
    }
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
      for (int j = i; j < N; ++j) {
        C[j] = A[j] + B[j];
      }
    }
  }
}

void call_add_f32_host(float* A, float* B, float* C, const int N) {
  for (int i = 0; i < N; ++i) {
    C[i] = A[i] + B[i];
  }
}

void call_add_f32_device(float* d_A, float* d_B, float* d_C, const int N) {
  elementwise_add<<<(N + BlockSize - 1) / BlockSize, BlockSize>>>(d_A, d_B, d_C, N);
}

void call_add_f32_gsl_device(float* d_A, float* d_B, float* d_C, const int N) {
  // int grid = (N + BlockSize - 1) / BlockSize;
  int grid = 1;
  elementwise_add_gsl<<<grid, BlockSize>>>(d_A, d_B, d_C, N);
}

void call_add_f32x4_device(float* d_A, float* d_B, float* d_C, const int N) {
  elementwise_add_vec4<<<(N + BlockSize * 4 - 1) / (BlockSize * 4), BlockSize>>>(d_A, d_B, d_C, N);
}

void call_add_f32x4_gsl_device(float* d_A, float* d_B, float* d_C, const int N) {
  int grid = (N + BlockSize * 4 - 1) / (BlockSize * 4);
  // int grid = 1;
  elementwise_add_vec4_gsl<<<grid, BlockSize>>>(d_A, d_B, d_C, N);
}
