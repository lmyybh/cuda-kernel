#include <cuda_runtime.h>
#include "cmdline.h"
#include "helper_cuda.cuh"
#include "helper_data.h"
#include "transpose.h"

int main(int argc, char* argv[]) {
  cmdline::parser args;
  args.add<int>("kernel", 'k', "which kernel", true, 0, cmdline::range(0, 4));
  args.add<int>("height", 'h', "heigth of matrix", false, 32 * 300);
  args.add<int>("width", 'w', "width of matrix", false, 32 * 300);
  args.add<int>("blockX", 'x', "blockDim.x", false, 16);
  args.add<int>("device", 'd', "gpu id", false, 0);
  args.parse_check(argc, argv);

  const int M = args.get<int>("height");
  const int N = args.get<int>("width");
  const int nBytes = M * N * sizeof(float);

  int devID = args.get<int>("device");
  cudaDeviceProp deviceProps;
  checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
  printf("CUDA device [%d]: [%s]\n", devID, deviceProps.name);

  float* A = nullptr;
  checkCudaErrors(cudaMallocHost((void**)&A, nBytes));
  float* B = nullptr;
  checkCudaErrors(cudaMallocHost((void**)&B, nBytes));
  initialRangeData(A, M * N, 0.0f, 1.0f);

  float* d_A = nullptr;
  checkCudaErrors(cudaMalloc((void**)&d_A, nBytes));
  float* d_B = nullptr;
  checkCudaErrors(cudaMalloc((void**)&d_B, nBytes));
  checkCudaErrors(cudaMemcpy(d_A, A, nBytes, cudaMemcpyHostToDevice));

  call_transpose_host(A, B, M, N);

  call_transpose_device(args.get<int>("kernel"), d_A, d_B, M, N, args.get<int>("blockX"));
  getLastCudaError("call transpose failed.\n");
  cudaDeviceSynchronize();

  float* gpuB = nullptr;
  checkCudaErrors(cudaMallocHost((void**)&gpuB, nBytes));
  checkCudaErrors(cudaMemcpy(gpuB, d_B, nBytes, cudaMemcpyDeviceToHost));

  if (checkResult(B, gpuB, M * N)) {
    printf("success\n");
  } else {
    printf("error\n");
  }

  // print2D<float>(A, M, N);
  // printf("-------------------------\n");
  // print2D<float>(B, N, M);
  // printf("-------------------------\n");
  // print2D<float>(gpuB, N, M);

  checkCudaErrors(cudaFreeHost(A));
  checkCudaErrors(cudaFreeHost(B));
  checkCudaErrors(cudaFreeHost(gpuB));
  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_B));

  return 0;
}