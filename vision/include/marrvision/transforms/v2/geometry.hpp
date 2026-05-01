#pragma once

#include <cmath>
#include <optional>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include <marrvision/transforms/v2/functional.hpp>
#include <marrvision/transforms/v2/transform.hpp>

namespace marr {
namespace vision {
namespace transforms {
namespace v2 {

class RandomHorizontalFlip : public Transform {
 public:
  explicit RandomHorizontalFlip(double p = 0.5) : p_(p) {
    if (p_ < 0.0 || p_ > 1.0) {
      throw Error("RandomHorizontalFlip probability must be in [0, 1]");
    }
  }

  torch::Tensor forward(torch::Tensor input) override {
    return torch::rand({1}).item<double>() < p_ ? functional::horizontal_flip(input) : input;
  }

  std::string name() const override {
    return "RandomHorizontalFlip";
  }

 private:
  double p_;
};

class RandomVerticalFlip : public Transform {
 public:
  explicit RandomVerticalFlip(double p = 0.5) : p_(p) {
    if (p_ < 0.0 || p_ > 1.0) {
      throw Error("RandomVerticalFlip probability must be in [0, 1]");
    }
  }

  torch::Tensor forward(torch::Tensor input) override {
    return torch::rand({1}).item<double>() < p_ ? functional::vertical_flip(input) : input;
  }

  std::string name() const override {
    return "RandomVerticalFlip";
  }

 private:
  double p_;
};

class Resize : public Transform {
 public:
  explicit Resize(Size2D size,
                  InterpolationMode interpolation = InterpolationMode::Bilinear,
                  bool antialias = true)
      : fixed_size_(size), interpolation_(interpolation), antialias_(antialias) {}

  explicit Resize(int64_t shorter_edge,
                  InterpolationMode interpolation = InterpolationMode::Bilinear,
                  std::optional<int64_t> max_size = std::nullopt,
                  bool antialias = true)
      : shorter_edge_(shorter_edge),
        max_size_(max_size),
        interpolation_(interpolation),
        antialias_(antialias) {
    if (shorter_edge <= 0) {
      throw Error("Resize shorter edge must be positive");
    }
  }

  torch::Tensor forward(torch::Tensor input) override {
    if (fixed_size_.has_value()) {
      return functional::resize(input, *fixed_size_, interpolation_, antialias_);
    }
    return functional::resize_shorter_edge(input, *shorter_edge_, max_size_, interpolation_, antialias_);
  }

  std::string name() const override {
    return "Resize";
  }

 private:
  std::optional<Size2D> fixed_size_;
  std::optional<int64_t> shorter_edge_;
  std::optional<int64_t> max_size_;
  InterpolationMode interpolation_;
  bool antialias_;
};

class RandomResize : public Transform {
 public:
  RandomResize(int64_t min_size,
               int64_t max_size,
               InterpolationMode interpolation = InterpolationMode::Bilinear,
               bool antialias = true)
      : min_size_(min_size),
        max_size_(max_size),
        interpolation_(interpolation),
        antialias_(antialias) {
    if (min_size_ <= 0 || max_size_ < min_size_) {
      throw Error("RandomResize expects 0 < min_size <= max_size");
    }
  }

  torch::Tensor forward(torch::Tensor input) override {
    const auto size = torch::randint(min_size_, max_size_ + 1, {1}).item<int64_t>();
    return functional::resize_shorter_edge(input, size, std::nullopt, interpolation_, antialias_);
  }

  std::string name() const override {
    return "RandomResize";
  }

 private:
  int64_t min_size_;
  int64_t max_size_;
  InterpolationMode interpolation_;
  bool antialias_;
};

class RandomShortestSize : public Transform {
 public:
  explicit RandomShortestSize(std::vector<int64_t> min_sizes,
                              std::optional<int64_t> max_size = std::nullopt,
                              InterpolationMode interpolation = InterpolationMode::Bilinear,
                              bool antialias = true)
      : min_sizes_(std::move(min_sizes)),
        max_size_(max_size),
        interpolation_(interpolation),
        antialias_(antialias) {
    if (min_sizes_.empty()) {
      throw Error("RandomShortestSize expects at least one size");
    }
  }

  torch::Tensor forward(torch::Tensor input) override {
    const auto index = torch::randint(static_cast<int64_t>(min_sizes_.size()), {1}).item<int64_t>();
    return functional::resize_shorter_edge(
        input, min_sizes_[static_cast<size_t>(index)], max_size_, interpolation_, antialias_);
  }

  std::string name() const override {
    return "RandomShortestSize";
  }

 private:
  std::vector<int64_t> min_sizes_;
  std::optional<int64_t> max_size_;
  InterpolationMode interpolation_;
  bool antialias_;
};

class CenterCrop : public Transform {
 public:
  explicit CenterCrop(Size2D size) : size_(size) {}
  explicit CenterCrop(int64_t size) : size_(size) {}

  torch::Tensor forward(torch::Tensor input) override {
    return functional::center_crop(input, size_);
  }

  std::string name() const override {
    return "CenterCrop";
  }

 private:
  Size2D size_;
};

class RandomCrop : public Transform {
 public:
  explicit RandomCrop(Size2D size, std::optional<Padding> padding = std::nullopt, double fill = 0.0)
      : size_(size), padding_(padding), fill_(fill) {}

  explicit RandomCrop(int64_t size, std::optional<Padding> padding = std::nullopt, double fill = 0.0)
      : RandomCrop(Size2D{size}, padding, fill) {}

  torch::Tensor forward(torch::Tensor input) override {
    if (padding_.has_value()) {
      input = functional::pad(input, *padding_, fill_);
    }

    auto image_size = functional::query_size(input);
    if (size_.height > image_size.height || size_.width > image_size.width) {
      input = functional::pad(input,
                              Padding{0,
                                      0,
                                      std::max<int64_t>(size_.width - image_size.width, 0),
                                      std::max<int64_t>(size_.height - image_size.height, 0)},
                              fill_);
      image_size = functional::query_size(input);
    }

    const auto top_limit = image_size.height - size_.height + 1;
    const auto left_limit = image_size.width - size_.width + 1;
    const auto top = torch::randint(top_limit, {1}).item<int64_t>();
    const auto left = torch::randint(left_limit, {1}).item<int64_t>();
    return functional::crop(input, CropParams{top, left, size_.height, size_.width});
  }

  std::string name() const override {
    return "RandomCrop";
  }

 private:
  Size2D size_;
  std::optional<Padding> padding_;
  double fill_;
};

class RandomResizedCrop : public Transform {
 public:
  RandomResizedCrop(Size2D size,
                    std::pair<double, double> scale = {0.08, 1.0},
                    std::pair<double, double> ratio = {3.0 / 4.0, 4.0 / 3.0},
                    InterpolationMode interpolation = InterpolationMode::Bilinear,
                    bool antialias = true)
      : size_(size),
        scale_(scale),
        ratio_(ratio),
        interpolation_(interpolation),
        antialias_(antialias) {
    if (scale_.first > scale_.second || ratio_.first > ratio_.second || scale_.first <= 0.0 ||
        ratio_.first <= 0.0) {
      throw Error("RandomResizedCrop expects valid scale and ratio ranges");
    }
  }

  explicit RandomResizedCrop(int64_t size,
                             std::pair<double, double> scale = {0.08, 1.0},
                             std::pair<double, double> ratio = {3.0 / 4.0, 4.0 / 3.0},
                             InterpolationMode interpolation = InterpolationMode::Bilinear,
                             bool antialias = true)
      : RandomResizedCrop(Size2D{size}, scale, ratio, interpolation, antialias) {}

  torch::Tensor forward(torch::Tensor input) override {
    const auto image_size = functional::query_size(input);
    const double area = static_cast<double>(image_size.height * image_size.width);
    const double log_ratio_min = std::log(ratio_.first);
    const double log_ratio_max = std::log(ratio_.second);

    for (int attempt = 0; attempt < 10; ++attempt) {
      const double target_area = area * torch::empty({1}).uniform_(scale_.first, scale_.second).item<double>();
      const double aspect =
          std::exp(torch::empty({1}).uniform_(log_ratio_min, log_ratio_max).item<double>());

      const auto width = static_cast<int64_t>(std::round(std::sqrt(target_area * aspect)));
      const auto height = static_cast<int64_t>(std::round(std::sqrt(target_area / aspect)));

      if (height > 0 && width > 0 && height <= image_size.height && width <= image_size.width) {
        const auto top = torch::randint(image_size.height - height + 1, {1}).item<int64_t>();
        const auto left = torch::randint(image_size.width - width + 1, {1}).item<int64_t>();
        return functional::resized_crop(input,
                                        CropParams{top, left, height, width},
                                        size_,
                                        interpolation_,
                                        antialias_);
      }
    }

    const double in_ratio = static_cast<double>(image_size.width) / static_cast<double>(image_size.height);
    int64_t width = image_size.width;
    int64_t height = image_size.height;
    if (in_ratio < ratio_.first) {
      height = static_cast<int64_t>(std::round(static_cast<double>(width) / ratio_.first));
    } else if (in_ratio > ratio_.second) {
      width = static_cast<int64_t>(std::round(static_cast<double>(height) * ratio_.second));
    }

    return functional::resized_crop(input,
                                    CropParams{(image_size.height - height) / 2,
                                               (image_size.width - width) / 2,
                                               height,
                                               width},
                                    size_,
                                    interpolation_,
                                    antialias_);
  }

  std::string name() const override {
    return "RandomResizedCrop";
  }

 private:
  Size2D size_;
  std::pair<double, double> scale_;
  std::pair<double, double> ratio_;
  InterpolationMode interpolation_;
  bool antialias_;
};

class Pad : public Transform {
 public:
  explicit Pad(Padding padding, double fill = 0.0, PadMode padding_mode = PadMode::Constant)
      : padding_(padding), fill_(fill), padding_mode_(padding_mode) {}

  explicit Pad(int64_t padding, double fill = 0.0, PadMode padding_mode = PadMode::Constant)
      : Pad(Padding{padding}, fill, padding_mode) {}

  torch::Tensor forward(torch::Tensor input) override {
    return functional::pad(input, padding_, fill_, padding_mode_);
  }

  std::string name() const override {
    return "Pad";
  }

 private:
  Padding padding_;
  double fill_;
  PadMode padding_mode_;
};

class FiveCrop {
 public:
  explicit FiveCrop(Size2D size) : size_(size) {}
  explicit FiveCrop(int64_t size) : size_(size) {}

  std::vector<torch::Tensor> forward(const torch::Tensor& input) const {
    return functional::five_crop(input, size_);
  }

 private:
  Size2D size_;
};

class TenCrop {
 public:
  explicit TenCrop(Size2D size, bool vertical_flip = false)
      : size_(size), vertical_flip_(vertical_flip) {}
  explicit TenCrop(int64_t size, bool vertical_flip = false)
      : size_(size), vertical_flip_(vertical_flip) {}

  std::vector<torch::Tensor> forward(const torch::Tensor& input) const {
    return functional::ten_crop(input, size_, vertical_flip_);
  }

 private:
  Size2D size_;
  bool vertical_flip_;
};

MARRVISION_V2_UNSUPPORTED_TRANSFORM(RandomRotation);
MARRVISION_V2_UNSUPPORTED_TRANSFORM(RandomAffine);
MARRVISION_V2_UNSUPPORTED_TRANSFORM(RandomPerspective);
MARRVISION_V2_UNSUPPORTED_TRANSFORM(RandomIoUCrop);
MARRVISION_V2_UNSUPPORTED_TRANSFORM(RandomZoomOut);
MARRVISION_V2_UNSUPPORTED_TRANSFORM(ScaleJitter);
MARRVISION_V2_UNSUPPORTED_TRANSFORM(ElasticTransform);

}  // namespace v2
}  // namespace transforms
}  // namespace vision
}  // namespace marr
