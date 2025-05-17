#include <cub/cub.cuh>
#include "cmdline.h"
#include "helper_cuda.cuh"
#include "helper_data.h"
#include "reduction.h"

int main(int argc, char* argv[]) {
  cmdline::parser args;
  args.add<int>("N", 'n', "number of elements", false, 16 * 1024 * 1024);
  args.add<unsigned int>("seed", 's', "random seed", false, 0);

  const int N = args.get<int>("N");
  const int nBytes = N * sizeof(float);

  // 初始化数据
  float* A = nullptr;
  checkCudaErrors(cudaMallocHost((void**)&A, nBytes));
  normalInitialData(A, N, args.get<unsigned int>("seed"));

  // CPU 求和
  float sum = call_reduction_sum_host(A, N);
  printf("CPU Sum: %f\n", sum);

  // host -> device
  float* d_A = nullptr;
  checkCudaErrors(cudaMalloc((void**)&d_A, nBytes));
  checkCudaErrors(cudaMemcpy(d_A, A, nBytes, cudaMemcpyHostToDevice));

  // 输出
  float* d_out;
  checkCudaErrors(cudaMalloc(&d_out, sizeof(float)));

  // 确定临时存储空间大小
  void* d_temp = nullptr;
  size_t temp_bytes = 0;
  checkCudaErrors(cub::DeviceReduce::Sum(d_temp, temp_bytes, d_A, d_out, N));
  checkCudaErrors(cudaMalloc(&d_temp, temp_bytes));

  // 分配临时存储空间
  checkCudaErrors(cub::DeviceReduce::Sum(d_temp, temp_bytes, d_A, d_out, N));

  // 执行规约求和
  float result;
  checkCudaErrors(cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost));
  printf("Sum: %f\n", result);

  // 释放资源
  checkCudaErrors(cudaFreeHost(A));
  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_out));
  checkCudaErrors(cudaFree(d_temp));

  return 0;
}