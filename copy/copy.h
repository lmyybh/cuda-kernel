#pragma once

void call_copy_device(int whichKernel, float* d_A, float* d_B, const int M, const int N,
                      int blockDimX);