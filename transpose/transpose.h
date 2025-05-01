#pragma once

void call_transpose_host(float* A, float* B, const int M, const int N);

void call_transpose_device(int whichKernel, float* d_A, float* d_B, const int M, const int N,
                           int blockDimX);