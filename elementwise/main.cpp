#include <cuda_runtime.h>
#include <algorithm>

#include "elementwise.h"
#include "helper_cuda.cuh"
#include "helper_data.h"

int clip(int value, int amin, int amax) { return std::max(std::min(value, amax), amin); }

int main(int argc, char* argv[]) {
  // 运行示例: elementwise whichKernel blockSize N devID

  // 定义默认值
  int whichKernel = 0;
  int blockSize = 256;
  int N = 16 * 1024 * 1024;
  int devID = 0;

  // 处理输入参数
  for (int i = 1; i < argc; ++i) {
    if (i == 1) {
      whichKernel = clip(atoi(argv[i]), 0, 3);
    } else if (i == 2) {
      blockSize = clip(atoi(argv[i]), 64, 1024);
    } else if (i == 3) {
      N = clip(atoi(argv[i]), 1, 16 * 1024 * 1024);
    } else if (i == 4) {
      devID = atoi(argv[i]);
    }
  }

  // 获取设备信息
  cudaDeviceProp deviceProps;
  checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
  printf("CUDA device [%d]: [%s]\n", devID, deviceProps.name);

  // 初始化 host 数据
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

  // 数据拷贝: host -> device
  float* d_A = nullptr;
  checkCudaErrors(cudaMalloc((float**)&d_A, nbytes));
  checkCudaErrors(cudaMemcpy(d_A, A, nbytes, cudaMemcpyHostToDevice));

  float* d_B = nullptr;
  checkCudaErrors(cudaMalloc((float**)&d_B, nbytes));
  checkCudaErrors(cudaMemcpy(d_B, B, nbytes, cudaMemcpyHostToDevice));

  float* d_C = nullptr;
  checkCudaErrors(cudaMalloc((float**)&d_C, nbytes));
  checkCudaErrors(cudaMemcpy(d_C, C, nbytes, cudaMemcpyHostToDevice));

  // 调用 host
  call_add_f32_host(A, B, C, N);

  // 调用 device
  call_add_f32_device(whichKernel, blockSize, d_A, d_B, d_C, N);
  getLastCudaError("call_add_f32_device failed\n");
  cudaDeviceSynchronize();

  // 检查结果
  float* gpuC = nullptr;
  checkCudaErrors(cudaMallocHost((void**)&gpuC, nbytes));
  cudaMemcpy(gpuC, d_C, nbytes, cudaMemcpyDeviceToHost);

  if (checkResult(C, gpuC, N)) {
    printf("Correct result\n");
  } else {
    printf("Incorrect result\n");
  }

  // 释放资源
  checkCudaErrors(cudaFreeHost(A));
  checkCudaErrors(cudaFreeHost(B));
  checkCudaErrors(cudaFreeHost(C));
  checkCudaErrors(cudaFreeHost(gpuC));
  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_B));
  checkCudaErrors(cudaFree(d_C));

  return 0;
}