#pragma once

#include <algorithm>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include <marrvision/transforms/v2/functional.hpp>
#include <marrvision/transforms/v2/transform.hpp>

namespace marr {
namespace vision {
namespace transforms {
namespace v2 {

class Grayscale : public Transform {
 public:
  explicit Grayscale(int64_t num_output_channels = 1) : num_output_channels_(num_output_channels) {}

  torch::Tensor forward(torch::Tensor input) override {
    return functional::grayscale(input, num_output_channels_);
  }

  std::string name() const override {
    return "Grayscale";
  }

 private:
  int64_t num_output_channels_;
};

class RGB : public Transform {
 public:
  torch::Tensor forward(torch::Tensor input) override {
    return functional::rgb(input);
  }

  std::string name() const override {
    return "RGB";
  }
};

class RandomGrayscale : public Transform {
 public:
  explicit RandomGrayscale(double p = 0.1) : p_(p) {
    if (p_ < 0.0 || p_ > 1.0) {
      throw Error("RandomGrayscale probability must be in [0, 1]");
    }
  }

  torch::Tensor forward(torch::Tensor input) override {
    if (torch::rand({1}).item<double>() >= p_) {
      return input;
    }
    const auto channels = input.size(input.dim() - 3);
    return functional::grayscale(input, channels);
  }

  std::string name() const override {
    return "RandomGrayscale";
  }

 private:
  double p_;
};

class ColorJitter : public Transform {
 public:
  explicit ColorJitter(double brightness = 0.0,
                       double contrast = 0.0,
                       double saturation = 0.0,
                       double hue = 0.0)
      : brightness_(check_non_negative_range(brightness, 1.0, 0.0, "brightness")),
        contrast_(check_non_negative_range(contrast, 1.0, 0.0, "contrast")),
        saturation_(check_non_negative_range(saturation, 1.0, 0.0, "saturation")),
        hue_{-hue, hue} {
    if (hue < 0.0 || hue > 0.5) {
      throw Error("hue must be in [0, 0.5]");
    }
  }

  torch::Tensor forward(torch::Tensor input) override {
    const auto brightness = brightness_.sample();
    const auto contrast = contrast_.sample();
    const auto saturation = saturation_.sample();
    const auto hue = hue_.sample();

    for (const auto op : functional::detail::shuffled_order(4)) {
      if (op == 0 && brightness != 1.0) {
        input = functional::adjust_brightness(input, brightness);
      } else if (op == 1 && contrast != 1.0) {
        input = functional::adjust_contrast(input, contrast);
      } else if (op == 2 && saturation != 1.0) {
        input = functional::adjust_saturation(input, saturation);
      } else if (op == 3 && hue != 0.0) {
        throw NotImplementedError("ColorJitter hue adjustment");
      }
    }
    return input;
  }

  std::string name() const override {
    return "ColorJitter";
  }

 private:
  ValueRange brightness_;
  ValueRange contrast_;
  ValueRange saturation_;
  ValueRange hue_;
};

class RandomInvert : public Transform {
 public:
  explicit RandomInvert(double p = 0.5) : p_(p) {}

  torch::Tensor forward(torch::Tensor input) override {
    return torch::rand({1}).item<double>() < p_ ? functional::invert(input) : input;
  }

  std::string name() const override {
    return "RandomInvert";
  }

 private:
  double p_;
};

class RandomSolarize : public Transform {
 public:
  explicit RandomSolarize(double threshold, double p = 0.5) : threshold_(threshold), p_(p) {}

  torch::Tensor forward(torch::Tensor input) override {
    return torch::rand({1}).item<double>() < p_ ? functional::solarize(input, threshold_) : input;
  }

  std::string name() const override {
    return "RandomSolarize";
  }

 private:
  double threshold_;
  double p_;
};

class RandomPosterize : public Transform {
 public:
  explicit RandomPosterize(int64_t bits, double p = 0.5) : bits_(bits), p_(p) {}

  torch::Tensor forward(torch::Tensor input) override {
    return torch::rand({1}).item<double>() < p_ ? functional::posterize(input, bits_) : input;
  }

  std::string name() const override {
    return "RandomPosterize";
  }

 private:
  int64_t bits_;
  double p_;
};

class RandomAutocontrast : public Transform {
 public:
  explicit RandomAutocontrast(double p = 0.5) : p_(p) {}

  torch::Tensor forward(torch::Tensor input) override {
    return torch::rand({1}).item<double>() < p_ ? functional::autocontrast(input) : input;
  }

  std::string name() const override {
    return "RandomAutocontrast";
  }

 private:
  double p_;
};

class RandomChannelPermutation : public Transform {
 public:
  torch::Tensor forward(torch::Tensor input) override {
    functional::detail::check_image_tensor(input);
    const auto channel_dim = input.dim() - 3;
    auto permutation = torch::randperm(input.size(channel_dim),
                                       torch::TensorOptions().dtype(torch::kLong).device(input.device()));
    return input.index_select(channel_dim, permutation);
  }

  std::string name() const override {
    return "RandomChannelPermutation";
  }
};

MARRVISION_V2_UNSUPPORTED_TRANSFORM(RandomAdjustSharpness);
MARRVISION_V2_UNSUPPORTED_TRANSFORM(RandomEqualize);
MARRVISION_V2_UNSUPPORTED_TRANSFORM(RandomPhotometricDistort);

}  // namespace v2
}  // namespace transforms
}  // namespace vision
}  // namespace marr
