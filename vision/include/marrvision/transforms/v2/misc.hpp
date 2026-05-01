#pragma once

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include <marrvision/transforms/v2/functional.hpp>
#include <marrvision/transforms/v2/transform.hpp>

namespace marr {
namespace vision {
namespace transforms {
namespace v2 {

class Identity : public Transform {
 public:
  torch::Tensor forward(torch::Tensor input) override {
    return input;
  }

  std::string name() const override {
    return "Identity";
  }
};

class Lambda : public Transform {
 public:
  explicit Lambda(std::function<torch::Tensor(torch::Tensor)> fn) : fn_(std::move(fn)) {}

  torch::Tensor forward(torch::Tensor input) override {
    return fn_(std::move(input));
  }

  std::string name() const override {
    return "Lambda";
  }

 private:
  std::function<torch::Tensor(torch::Tensor)> fn_;
};

class Normalize : public Transform {
 public:
  Normalize(std::vector<double> mean, std::vector<double> std)
      : mean_(std::move(mean)), std_(std::move(std)) {}

  torch::Tensor forward(torch::Tensor input) override {
    return functional::normalize(input, mean_, std_);
  }

  std::string name() const override {
    return "Normalize";
  }

 private:
  std::vector<double> mean_;
  std::vector<double> std_;
};

class ToDtype : public Transform {
 public:
  explicit ToDtype(torch::ScalarType dtype, bool scale = false) : dtype_(dtype), scale_(scale) {}

  torch::Tensor forward(torch::Tensor input) override {
    return functional::to_dtype(input, dtype_, scale_);
  }

  std::string name() const override {
    return "ToDtype";
  }

 private:
  torch::ScalarType dtype_;
  bool scale_;
};

class ConvertImageDtype : public ToDtype {
 public:
  explicit ConvertImageDtype(torch::ScalarType dtype) : ToDtype(dtype, true) {}
};

class GaussianNoise : public Transform {
 public:
  explicit GaussianNoise(double mean = 0.0, double sigma = 0.1) : mean_(mean), sigma_(sigma) {
    if (sigma_ < 0.0) {
      throw Error("GaussianNoise sigma must be non-negative");
    }
  }

  torch::Tensor forward(torch::Tensor input) override {
    return functional::gaussian_noise(input, mean_, sigma_);
  }

  std::string name() const override {
    return "GaussianNoise";
  }

 private:
  double mean_;
  double sigma_;
};

class LinearTransformation : public Transform {
 public:
  LinearTransformation(torch::Tensor transformation_matrix, torch::Tensor mean_vector)
      : transformation_matrix_(std::move(transformation_matrix)),
        mean_vector_(std::move(mean_vector)) {}

  torch::Tensor forward(torch::Tensor input) override {
    return functional::linear_transformation(input, transformation_matrix_, mean_vector_);
  }

  std::string name() const override {
    return "LinearTransformation";
  }

 private:
  torch::Tensor transformation_matrix_;
  torch::Tensor mean_vector_;
};

MARRVISION_V2_UNSUPPORTED_TRANSFORM(GaussianBlur);
MARRVISION_V2_UNSUPPORTED_TRANSFORM(SanitizeBoundingBoxes);
MARRVISION_V2_UNSUPPORTED_TRANSFORM(SanitizeKeyPoints);

}  // namespace v2
}  // namespace transforms
}  // namespace vision
}  // namespace marr
