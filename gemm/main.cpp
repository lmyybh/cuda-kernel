#include <cuda_runtime.h>
#include "cmdline.h"
#include "helper_cuda.cuh"
#include "helper_data.h"
#include "gemm.h"

int main(int argc, char* argv[]) {
  cmdline::parser args;
  args.add<int>("kernel", 'x', "which kernel", true, 0, cmdline::range(0, 9));
  args.add<int>("M", 'm', "rows of matrix A", false, 2048);
  args.add<int>("K", 'k', "columns of matrix A", false, 2048);
  args.add<int>("N", 'n', "columns of matrix B", false, 2048);
  args.add<int>("check", 'c', "check result", false, 0);
  args.add<int>("device", 'd', "gpu id", false, 0);
  args.parse_check(argc, argv);

  // 获取设备信息
  int devID = args.get<int>("device");
  cudaDeviceProp deviceProps;
  checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
  printf("CUDA device [%d]: [%s]\n", devID, deviceProps.name);

  // 矩阵尺寸
  const int M = args.get<int>("M");
  const int K = args.get<int>("K");
  const int N = args.get<int>("N");
  printf("M: [%d], K: [%d], N: [%d]\n", M, K, N);

  const int A_Bytes = M * K * sizeof(float);
  const int B_Bytes = K * N * sizeof(float);
  const int out_Bytes = M * N * sizeof(float);

  // 初始化 host 数据
  float* A = nullptr;
  checkCudaErrors(cudaMallocHost((void**)&A, A_Bytes));
  normalInitialData(A, M * K, 1);

  float* B = nullptr;
  checkCudaErrors(cudaMallocHost((void**)&B, B_Bytes));
  normalInitialData(B, K * N, 2);

  float* out = nullptr;
  checkCudaErrors(cudaMallocHost((void**)&out, out_Bytes));
  initialRangeData(out, M * N, 0, 0);

  // 数据拷贝: host -> device
  float* d_A = nullptr;
  checkCudaErrors(cudaMalloc((void**)&d_A, A_Bytes));
  checkCudaErrors(cudaMemcpy(d_A, A, A_Bytes, cudaMemcpyHostToDevice));

  float* d_B = nullptr;
  checkCudaErrors(cudaMalloc((void**)&d_B, B_Bytes));
  checkCudaErrors(cudaMemcpy(d_B, B, B_Bytes, cudaMemcpyHostToDevice));

  float* d_out = nullptr;
  checkCudaErrors(cudaMalloc((void**)&d_out, out_Bytes));
  checkCudaErrors(cudaMemcpy(d_out, out, out_Bytes, cudaMemcpyHostToDevice));

  // 调用 device GEMM
  call_gemm_device(args.get<int>("kernel"), d_A, d_B, d_out, M, K, N);

  float* gpuOut = nullptr;
  checkCudaErrors(cudaMallocHost((void**)&gpuOut, out_Bytes));
  checkCudaErrors(cudaMemcpy(gpuOut, d_out, out_Bytes, cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaDeviceSynchronize());

  // printf("-----------------------------------------\n");
  // print2D<float>(A, M, K);
  // printf("-----------------------------------------\n");
  // print2D<float>(B, K, N);
  // printf("-----------------------------------------\n");
  // print2D<float>(out, M, N);
  // printf("-----------------------------------------\n");
  // print2D<float>(gpuOut, M, N);
  // printf("-----------------------------------------\n");

  if (args.get<int>("check")) {
    call_gemm_host(A, B, out, M, K, N);
    if (checkResult(out, gpuOut, M * N, 1.0E-3f)) {
      printf("success\n");
    } else {
      printf("error\n");
    }
  }

  // 释放资源
  checkCudaErrors(cudaFreeHost(A));
  checkCudaErrors(cudaFreeHost(B));
  checkCudaErrors(cudaFreeHost(out));
  checkCudaErrors(cudaFreeHost(gpuOut));

  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_B));
  checkCudaErrors(cudaFree(d_out));

  return 0;
}