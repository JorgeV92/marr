#pragma once

#include <algorithm>
#include <memory>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include <marrvision/transforms/v2/transform.hpp>

namespace marr {
namespace vision {
namespace transforms {
namespace v2 {

class Compose : public Transform {
 public:
  Compose() = default;
  explicit Compose(std::vector<TransformPtr> transforms) : transforms_(std::move(transforms)) {}

  torch::Tensor forward(torch::Tensor input) override {
    for (const auto& transform : transforms_) {
      input = transform->forward(std::move(input));
    }
    return input;
  }

  void push_back(TransformPtr transform) {
    transforms_.push_back(std::move(transform));
  }

  const std::vector<TransformPtr>& transforms() const {
    return transforms_;
  }

  std::string name() const override {
    return "Compose";
  }

 private:
  std::vector<TransformPtr> transforms_;
};

class RandomApply : public Transform {
 public:
  explicit RandomApply(std::vector<TransformPtr> transforms, double p = 0.5)
      : transforms_(std::move(transforms)), p_(p) {
    if (p_ < 0.0 || p_ > 1.0) {
      throw Error("RandomApply probability must be in [0, 1]");
    }
  }

  torch::Tensor forward(torch::Tensor input) override {
    if (torch::rand({1}).item<double>() >= p_) {
      return input;
    }
    for (const auto& transform : transforms_) {
      input = transform->forward(std::move(input));
    }
    return input;
  }

  std::string name() const override {
    return "RandomApply";
  }

 private:
  std::vector<TransformPtr> transforms_;
  double p_;
};

class RandomChoice : public Transform {
 public:
  explicit RandomChoice(std::vector<TransformPtr> transforms,
                        std::vector<double> probabilities = {})
      : transforms_(std::move(transforms)), probabilities_(std::move(probabilities)) {
    if (transforms_.empty()) {
      throw Error("RandomChoice expects at least one transform");
    }
    if (!probabilities_.empty() && probabilities_.size() != transforms_.size()) {
      throw Error("RandomChoice probabilities must match transform count");
    }
  }

  torch::Tensor forward(torch::Tensor input) override {
    int64_t index = 0;
    if (probabilities_.empty()) {
      index = torch::randint(static_cast<int64_t>(transforms_.size()), {1}).item<int64_t>();
    } else {
      auto weights = torch::tensor(probabilities_, torch::kDouble);
      index = torch::multinomial(weights, 1).item<int64_t>();
    }
    return transforms_[static_cast<size_t>(index)]->forward(std::move(input));
  }

  std::string name() const override {
    return "RandomChoice";
  }

 private:
  std::vector<TransformPtr> transforms_;
  std::vector<double> probabilities_;
};

class RandomOrder : public Transform {
 public:
  explicit RandomOrder(std::vector<TransformPtr> transforms) : transforms_(std::move(transforms)) {}

  torch::Tensor forward(torch::Tensor input) override {
    std::vector<size_t> order(transforms_.size());
    std::iota(order.begin(), order.end(), 0);
    static std::mt19937 generator(std::random_device{}());
    std::shuffle(order.begin(), order.end(), generator);

    for (const auto index : order) {
      input = transforms_[index]->forward(std::move(input));
    }
    return input;
  }

  std::string name() const override {
    return "RandomOrder";
  }

 private:
  std::vector<TransformPtr> transforms_;
};

}  // namespace v2
}  // namespace transforms
}  // namespace vision
}  // namespace marr
