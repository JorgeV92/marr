#pragma once

#include <torch/torch.h>

#include <marrvision/transforms/v2/functional.hpp>
#include <marrvision/transforms/v2/transform.hpp>

namespace marr {
namespace vision {
namespace transforms {
namespace v2 {

class ToImage : public Transform {
 public:
  torch::Tensor forward(torch::Tensor input) override {
    return input;
  }

  std::string name() const override {
    return "ToImage";
  }
};

class ToPureTensor : public Transform {
 public:
  torch::Tensor forward(torch::Tensor input) override {
    return input;
  }

  std::string name() const override {
    return "ToPureTensor";
  }
};

class ToTensor : public Transform {
 public:
  torch::Tensor forward(torch::Tensor input) override {
    return functional::to_dtype(input, torch::kFloat32, true);
  }

  std::string name() const override {
    return "ToTensor";
  }
};

MARRVISION_V2_UNSUPPORTED_TRANSFORM(PILToTensor);
MARRVISION_V2_UNSUPPORTED_TRANSFORM(ToPILImage);
MARRVISION_V2_UNSUPPORTED_TRANSFORM(CVCUDAToTensor);
MARRVISION_V2_UNSUPPORTED_TRANSFORM(ToCVCUDATensor);

}  // namespace v2
}  // namespace transforms
}  // namespace vision
}  // namespace marr
