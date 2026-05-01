#pragma once

#include <torch/torch.h>

#include <marrvision/transforms/v2/functional.hpp>
#include <marrvision/transforms/v2/transform.hpp>

namespace marr {
namespace vision {
namespace transforms {
namespace v2 {

class UniformTemporalSubsample : public Transform {
 public:
  explicit UniformTemporalSubsample(int64_t num_samples) : num_samples_(num_samples) {}

  torch::Tensor forward(torch::Tensor input) override {
    return functional::uniform_temporal_subsample(input, num_samples_);
  }

  std::string name() const override {
    return "UniformTemporalSubsample";
  }

 private:
  int64_t num_samples_;
};

}  // namespace v2
}  // namespace transforms
}  // namespace vision
}  // namespace marr
