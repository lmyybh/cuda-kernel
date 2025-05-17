#include <cuda_runtime.h>
#include "cmdline.h"
#include "helper_cuda.cuh"
#include "helper_data.h"
#include "reduction.h"

int main(int argc, char* argv[]) {
  cmdline::parser args;
  args.add<int>("kernel", 'k', "which kernel", true, 0, cmdline::range(-2, 7));
  args.add<int>("N", 'n', "number of elements", false, 16 * 1024 * 1024);
  args.add<unsigned int>("seed", 's', "random seed", false, 0);
  args.add<int>("device", 'd', "gpu id", false, 0);
  args.parse_check(argc, argv);

  // 获取设备信息
  int devID = args.get<int>("device");
  cudaDeviceProp deviceProps;
  checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
  printf("Device [%d]: [%s]\n", devID, deviceProps.name);

  const int N = args.get<int>("N");
  const int nBytes = N * sizeof(float);

  // 初始化数据
  float* A = nullptr;
  checkCudaErrors(cudaMallocHost((void**)&A, nBytes));
  normalInitialData(A, N, args.get<unsigned int>("seed"));
  // print1D<float>(A, N);

  float sum = call_reduction_sum_host(A, N);
  printf("CPU Sum: %f\n", sum);

  float* d_A = nullptr;
  checkCudaErrors(cudaMalloc((void**)&d_A, nBytes));
  checkCudaErrors(cudaMemcpy(d_A, A, nBytes, cudaMemcpyHostToDevice));

  call_reduction_sum_device(args.get<int>("kernel"), d_A, N);
  checkCudaErrors(cudaMemcpy(A, d_A, nBytes, cudaMemcpyDeviceToHost));
  printf("GPU Sum: %f\n", A[0]);

  cudaDeviceSynchronize();

  checkCudaErrors(cudaFreeHost(A));
  checkCudaErrors(cudaFree(d_A));

  return 0;
}