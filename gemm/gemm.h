#pragma once

void call_gemm_host(float* A, float* B, float* out, const int M, const int K, const int N);

void call_gemm_device(int whichKernel, float* A, float* B, float* out, const int M, const int K,
                      const int N);