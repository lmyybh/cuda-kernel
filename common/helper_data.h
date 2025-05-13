#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <string>
#include <random>

void initialRangeData(float* p, const int size, float start, float step) {
  for (int i = 0; i < size; ++i) { p[i] = start + step * i; }
}

void normalInitialData(float* p, const int size, unsigned int seed = 0, float mean = 0.0f,
                       float std = 1.0f) {
  std::default_random_engine generator(seed);
  std::normal_distribution<float> dist(mean, std);

  for (int i = 0; i < size; ++i) { p[i] = dist(generator); }
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

template<typename T>
void print1D(T* data, const int N) {
  for (int i = 0; i < N; ++i) { std::cout << data[i] << ","; }
  std::cout << std::endl;
}

template<typename T>
void print2D(T* matrix, const int rows, const int cols) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) { std::cout << matrix[i * cols + j] << ","; }
    std::cout << std::endl;
  }
}