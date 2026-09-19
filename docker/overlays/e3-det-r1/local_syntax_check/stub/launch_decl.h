#pragma once
#include <cuda_runtime_api.h>
extern "C" __host__ cudaError_t cudaConfigureCall(dim3 g, dim3 b, size_t s = 0, cudaStream_t st = 0);
