#pragma once

float call_reduction_sum_host(float* A, const int N);

void call_reduction_sum_device(int whichKernel, float* d_A, const int N);