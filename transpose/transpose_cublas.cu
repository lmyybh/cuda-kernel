#include <cublas_v2.h>
#include "cmdline.h"
#include "helper_cuda.cuh"
#include "helper_data.h"

int main(int argc, char* argv[]) {
  cmdline::parser args;
  args.add<int>("height", 'h', "heigth of matrix", false, 32 * 300);
  args.add<int>("width", 'w', "width of matrix", false, 32 * 300);
  args.add<int>("device", 'd', "gpu id", false, 0);
  args.parse_check(argc, argv);

  const int M = args.get<int>("height");
  const int N = args.get<int>("width");
  const int nBytes = M * N * sizeof(float);

  float* A = nullptr;
  checkCudaErrors(cudaMallocHost((void**)&A, nBytes));
  initialRangeData(A, M * N, 0.0f, 1.0f);

  float* A_colmajor = nullptr;
  checkCudaErrors(cudaMallocHost((void**)&A_colmajor, nBytes));

  // 转换为列主序
  for (int col = 0; col < N; ++col)
    for (int row = 0; row < M; ++row) { A_colmajor[col * M + row] = A[row * N + col]; }

  float* d_A = nullptr;
  checkCudaErrors(cudaMalloc((void**)&d_A, nBytes));
  float* d_B = nullptr;
  checkCudaErrors(cudaMalloc((void**)&d_B, nBytes));
  checkCudaErrors(cudaMemcpy(d_A, A_colmajor, nBytes, cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  cublasCreate(&handle);

  float alpha = 1.0f, beta = 0.0f;

  cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, &alpha, d_A, M, &beta, nullptr, N, d_B, N);

  cudaDeviceSynchronize();

  float* gpuB = nullptr;
  checkCudaErrors(cudaMallocHost((void**)&gpuB, nBytes));
  checkCudaErrors(cudaMemcpy(gpuB, d_B, nBytes, cudaMemcpyDeviceToHost));

  //   print2D<float>(A_colmajor, N, M);
  //   printf("-------------------------\n");
  //   print2D<float>(gpuB, M, N);

  checkCudaErrors(cudaFreeHost(A));
  checkCudaErrors(cudaFreeHost(A_colmajor));
  checkCudaErrors(cudaFreeHost(gpuB));
  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_B));

  return 0;
}