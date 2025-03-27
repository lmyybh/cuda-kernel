#include <cuda_runtime.h>

#include "elementwise.h"
#include "helper_cuda.cuh"
#include "helper_data.h"

int main(int arc, char* argv[]) {
  int devID = 0;
  cudaDeviceProp deviceProps;
  checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
  printf("CUDA device [%s]\n", deviceProps.name);

  int N = 16 * 1024 * 1024;
  int nbytes = N * sizeof(float);

  float* A = nullptr;
  checkCudaErrors(cudaMallocHost((void**)&A, nbytes));
  initialRangeData(A, N, 0.0f, 0.01f);

  float* B = nullptr;
  checkCudaErrors(cudaMallocHost((void**)&B, nbytes));
  initialRangeData(B, N, 0.0f, -0.02f);

  float* C = nullptr;
  checkCudaErrors(cudaMallocHost((void**)&C, nbytes));
  memset(C, 0, nbytes);

  float* d_A = nullptr;
  checkCudaErrors(cudaMalloc((float**)&d_A, nbytes));
  checkCudaErrors(cudaMemcpy(d_A, A, nbytes, cudaMemcpyHostToDevice));

  float* d_B = nullptr;
  checkCudaErrors(cudaMalloc((float**)&d_B, nbytes));
  checkCudaErrors(cudaMemcpy(d_B, B, nbytes, cudaMemcpyHostToDevice));

  float* d_C = nullptr;
  checkCudaErrors(cudaMalloc((float**)&d_C, nbytes));
  checkCudaErrors(cudaMemcpy(d_C, C, nbytes, cudaMemcpyHostToDevice));

  float* d_C2 = nullptr;
  checkCudaErrors(cudaMalloc((float**)&d_C2, nbytes));
  checkCudaErrors(cudaMemcpy(d_C2, C, nbytes, cudaMemcpyHostToDevice));

  call_add_f32_host(A, B, C, N);

  call_add_f32_gsl_device(d_A, d_B, d_C, N);
  getLastCudaError("call_add_f32_gsl_device failed\n");

  call_add_f32x4_gsl_device(d_A, d_B, d_C2, N);
  getLastCudaError("call_add_f32x4_gsl_device failed\n");
  cudaDeviceSynchronize();

  // 检查结果
  float* gpuC = nullptr;
  checkCudaErrors(cudaMallocHost((void**)&gpuC, nbytes));
  cudaMemcpy(gpuC, d_C, nbytes, cudaMemcpyDeviceToHost);

  float* gpuC2 = nullptr;
  checkCudaErrors(cudaMallocHost((void**)&gpuC2, nbytes));
  cudaMemcpy(gpuC2, d_C2, nbytes, cudaMemcpyDeviceToHost);

  if (checkResult(C, gpuC, N)) {
    printf("Correct result: call_add_f32_gsl_device\n");
  } else {
    printf("Incorrect result: call_add_f32_gsl_device\n");
  }

  if (checkResult(C, gpuC2, N)) {
    printf("Correct result: call_add_f32x4_gsl_device\n");
  } else {
    printf("Incorrect result: call_add_f32x4_gsl_device\n");
  }

  checkCudaErrors(cudaFreeHost(A));
  checkCudaErrors(cudaFreeHost(B));
  checkCudaErrors(cudaFreeHost(C));
  checkCudaErrors(cudaFreeHost(gpuC));
  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_B));
  checkCudaErrors(cudaFree(d_C));

  return 0;
}
