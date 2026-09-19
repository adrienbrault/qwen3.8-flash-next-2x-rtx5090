#pragma once
// Minimal stand-in for syntax-checking device code with clang (NOT the real torch API)
#include <cstdint>
#include <cstdio>
#include <vector>
#define TORCH_CHECK(cond, ...) do { if (!(cond)) { std::printf("check\n"); } } while (0)
namespace c10 { enum class ScalarType { Half, Float, Long, Int }; }
namespace at {
using ScalarType = c10::ScalarType;
constexpr ScalarType kHalf = ScalarType::Half, kFloat = ScalarType::Float, kLong = ScalarType::Long, kInt = ScalarType::Int;
struct Device { int index() const { return 0; } };
struct Tensor {
  void* data_ptr() const { return nullptr; }
  template <typename T> T* data_ptr() const { return nullptr; }
  int64_t size(int) const { return 0; }
  int64_t numel() const { return 0; }
  int64_t dim() const { return 0; }
  std::vector<int64_t> sizes() const { return {}; }
  bool is_cuda() const { return true; }
  bool is_contiguous() const { return true; }
  ScalarType scalar_type() const { return kHalf; }
  Device device() const { return {}; }
  int get_device() const { return 0; }
};
}
