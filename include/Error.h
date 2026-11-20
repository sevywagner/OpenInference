#ifndef IE_ERROR_H
#define IE_ERROR_H

#include "Common.h"
#include <cuda.h>
#include <cublasLt.h>

#include <stdio.h>
#include <stdexcept>

inline void cuda_error_check(cudaError_t err, const char *exp, const char *file, u64 line) {
  if (err != cudaSuccess) {
    fprintf(stderr, "Cuda error occurred: %s\n\tExpression: %s\n\tFile: %s\n\tLine: %llu", cudaGetErrorString(err), exp, file, line);
    throw ::std::runtime_error("CUDA Error");
  }
}

inline void cublas_error_check(cublasStatus_t stat, const char *expr, const char *file, u64 line) {
  if (stat != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "Cublas error: %u\n\tExpression: %s\n\tFile: %s\n\tLine: %llu", stat, expr, file, line);
    throw ::std::runtime_error("Cublas error");
  }
}

#define CUBLAS_CHECK(expr) (cublas_error_check(expr, #expr, __FILE__, __LINE__))
#define CUDA_CHECK(exp) (cuda_error_check(exp, #exp, __FILE__, __LINE__))

#endif
