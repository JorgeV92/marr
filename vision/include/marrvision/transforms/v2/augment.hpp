#pragma once

#include <cmath>
#include <string>
#include <utility>

#include <torch/torch.h>

#include <marrvision/transforms/v2/functional.hpp>
#include <marrvision/transforms/v2/transform.hpp>

namespace marr {
namespace vision {
namespace transforms {
namespace v2 {

class RandomErasing : public Transform {
 public:
  RandomErasing(double p = 0.5,
                std::pair<double, double> scale = {0.02, 0.33},
                std::pair<double, double> ratio = {0.3, 3.3},
                double value = 0.0,
                bool inplace = false)
      : p_(p), scale_(scale), ratio_(ratio), value_(value), inplace_(inplace) {
    if (p_ < 0.0 || p_ > 1.0 || scale_.first <= 0.0 || scale_.first > scale_.second ||
        ratio_.first <= 0.0 || ratio_.first > ratio_.second) {
      throw Error("RandomErasing received invalid probability, scale, or ratio");
    }
  }

  torch::Tensor forward(torch::Tensor input) override {
    if (torch::rand({1}).item<double>() >= p_) {
      return input;
    }

    const auto size = functional::query_size(input);
    const double area = static_cast<double>(size.height * size.width);
    const double log_ratio_min = std::log(ratio_.first);
    const double log_ratio_max = std::log(ratio_.second);

    for (int attempt = 0; attempt < 10; ++attempt) {
      const double erase_area = area * torch::empty({1}).uniform_(scale_.first, scale_.second).item<double>();
      const double aspect =
          std::exp(torch::empty({1}).uniform_(log_ratio_min, log_ratio_max).item<double>());
      const auto height = static_cast<int64_t>(std::round(std::sqrt(erase_area * aspect)));
      const auto width = static_cast<int64_t>(std::round(std::sqrt(erase_area / aspect)));
      if (height > 0 && width > 0 && height < size.height && width < size.width) {
        const auto top = torch::randint(size.height - height + 1, {1}).item<int64_t>();
        const auto left = torch::randint(size.width - width + 1, {1}).item<int64_t>();
        return functional::erase(input, CropParams{top, left, height, width}, value_, inplace_);
      }
    }
    return input;
  }

  std::string name() const override {
    return "RandomErasing";
  }

 private:
  double p_;
  std::pair<double, double> scale_;
  std::pair<double, double> ratio_;
  double value_;
  bool inplace_;
};

MARRVISION_V2_UNSUPPORTED_TRANSFORM(CutMix);
MARRVISION_V2_UNSUPPORTED_TRANSFORM(MixUp);
MARRVISION_V2_UNSUPPORTED_TRANSFORM(JPEG);

}  // namespace v2
}  // namespace transforms
}  // namespace vision
}  // namespace marr
