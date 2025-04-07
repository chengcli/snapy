// C/C++
#include <cstdio>

// cuda
#include <cuda_runtime.h>

// snap
#include "cuda_utils.h"

int checkCudaError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error occured in : %s\n", msg);
    printf("Cause: %s\n", cudaGetErrorString(err));
    return 1;
  }

  return 0;
}
