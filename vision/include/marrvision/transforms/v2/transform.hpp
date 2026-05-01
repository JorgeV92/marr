#pragma once

#include <memory>
#include <string>

#include <torch/torch.h>

#include <marrvision/transforms/v2/types.hpp>

namespace marr {
namespace vision {
namespace transforms {
namespace v2 {

class Transform : public torch::nn::Module {
 public:
  ~Transform() override = default;

  virtual torch::Tensor forward(torch::Tensor input) = 0;

  virtual std::string name() const {
    return "Transform";
  }
};

using TransformPtr = std::shared_ptr<Transform>;

class UnsupportedTransform : public Transform {
 public:
  explicit UnsupportedTransform(std::string transform_name)
      : transform_name_(std::move(transform_name)) {}

  torch::Tensor forward(torch::Tensor input) override {
    (void)input;
    throw NotImplementedError(transform_name_);
  }

  std::string name() const override {
    return transform_name_;
  }

 private:
  std::string transform_name_;
};

#define MARRVISION_V2_UNSUPPORTED_TRANSFORM(ClassName)       \
  class ClassName : public UnsupportedTransform {            \
   public:                                                   \
    template <typename... Args>                              \
    explicit ClassName(Args&&...) : UnsupportedTransform(#ClassName) {} \
  }

}  // namespace v2
}  // namespace transforms
}  // namespace vision
}  // namespace marr
