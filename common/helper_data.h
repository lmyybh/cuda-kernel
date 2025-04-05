#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void initialRangeData(float* p, const int size, float start, float step) {
  for (int i = 0; i < size; ++i) { p[i] = start + step * i; }
}

bool checkResult(float* hostRef, float* gpuRef, const int N) {
  double eps = 1.0E-8;
  bool match = true;
  for (int i = 0; i < N; ++i) {
    if (abs(hostRef[i] - gpuRef[i]) > eps) {
      match = false;
      break;
    }
  }

  return match;
}