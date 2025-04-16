#pragma once

void call_add_f32_host(float* A, float* B, float* C, const int N);

void call_add_f32_device(int whichKernel, int blockSize, float* d_A, float* d_B, float* d_C,
                         const int N);