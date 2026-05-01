#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <torch/torch.h>

namespace marr {
namespace vision {
namespace transforms {
namespace v2 {

class Error : public std::runtime_error {
 public:
  explicit Error(const std::string& message) : std::runtime_error(message) {}
};

class NotImplementedError : public Error {
 public:
  explicit NotImplementedError(const std::string& feature)
      : Error(feature + " is not implemented in marrvision.transforms.v2 yet") {}
};

enum class InterpolationMode {
  Nearest,
  NearestExact,
  Bilinear,
  Bicubic,
  Lanczos,
  Box,
  Hamming,
};

enum class AutoAugmentPolicy {
  ImageNet,
  Cifar10,
  Svhn,
};

enum class PadMode {
  Constant,
  Edge,
  Reflect,
  Symmetric,
};

enum class BoundingBoxFormat {
  XYXY,
  XYWH,
  CXCYWH,
};

struct Size2D {
  int64_t height = 0;
  int64_t width = 0;

  Size2D() = default;
  Size2D(int64_t h, int64_t w) : height(h), width(w) {
    if (height <= 0 || width <= 0) {
      throw Error("Size2D expects positive height and width");
    }
  }

  explicit Size2D(int64_t square) : Size2D(square, square) {}

  std::vector<int64_t> vec() const {
    return {height, width};
  }
};

struct Padding {
  int64_t left = 0;
  int64_t top = 0;
  int64_t right = 0;
  int64_t bottom = 0;

  Padding() = default;
  explicit Padding(int64_t all) : left(all), top(all), right(all), bottom(all) {}
  Padding(int64_t horizontal, int64_t vertical)
      : left(horizontal), top(vertical), right(horizontal), bottom(vertical) {}
  Padding(int64_t l, int64_t t, int64_t r, int64_t b)
      : left(l), top(t), right(r), bottom(b) {}

  std::vector<int64_t> torch_order() const {
    return {left, right, top, bottom};
  }
};

struct CropParams {
  int64_t top = 0;
  int64_t left = 0;
  int64_t height = 0;
  int64_t width = 0;
};

struct Chw {
  int64_t channels = 0;
  int64_t height = 0;
  int64_t width = 0;
};

struct ValueRange {
  double min = 0.0;
  double max = 0.0;

  double sample() const {
    if (min == max) {
      return min;
    }
    return torch::empty({1}).uniform_(min, max).item<double>();
  }
};

inline std::string to_string(InterpolationMode mode) {
  switch (mode) {
    case InterpolationMode::Nearest:
      return "nearest";
    case InterpolationMode::NearestExact:
      return "nearest-exact";
    case InterpolationMode::Bilinear:
      return "bilinear";
    case InterpolationMode::Bicubic:
      return "bicubic";
    case InterpolationMode::Lanczos:
      return "lanczos";
    case InterpolationMode::Box:
      return "box";
    case InterpolationMode::Hamming:
      return "hamming";
  }
  return "unknown";
}

inline bool is_floating_point(torch::ScalarType dtype) {
  return dtype == torch::kFloat16 || dtype == torch::kBFloat16 ||
         dtype == torch::kFloat32 || dtype == torch::kFloat64;
}

inline double max_value_for_dtype(torch::ScalarType dtype) {
  switch (dtype) {
    case torch::kUInt8:
      return 255.0;
    case torch::kInt8:
      return 127.0;
    case torch::kInt16:
      return static_cast<double>(std::numeric_limits<int16_t>::max());
    case torch::kInt32:
      return static_cast<double>(std::numeric_limits<int32_t>::max());
    case torch::kInt64:
      return static_cast<double>(std::numeric_limits<int64_t>::max());
    default:
      return 1.0;
  }
}

inline ValueRange check_non_negative_range(double value,
                                           double center,
                                           double lower_bound,
                                           const std::string& name) {
  if (value < 0.0) {
    throw Error(name + " must be non-negative");
  }
  return {std::max(lower_bound, center - value), center + value};
}

}  // namespace v2
}  // namespace transforms
}  // namespace vision
}  // namespace marr
