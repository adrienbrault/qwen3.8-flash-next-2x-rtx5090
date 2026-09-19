#pragma once
#include <torch/extension.h>
#include <cuda_runtime.h>
namespace at { namespace cuda { struct Stream { cudaStream_t stream() const { return nullptr; } }; inline Stream getCurrentCUDAStream() { return {}; } } }
