#ifndef TINYVIT_HIER_HPP
#define TINYVIT_HIER_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <torch/torch.h>

namespace Marr {

using toorch::indexing::Slice;

struct TinyViTConfig {
  int64_t image_size = 32;
  int64_t in_channels = 1;
  int64_t num_classes = 4;
  std::vector<int64_t> embed_dims{32, 64, 96, 128};
  std::vector<int64_t> depths{2, 2, 2, 1};
  std::vector<int64_t> num_heads{1, 2, 3, 4};
  int64_t window_size = 4;
  int64_t mlp_ratio = 4;

  void validate() const {
    if (embed_dims.size() != 4 || depths.size() != 4 || num_heads.size() != 4) {
      throw std::invalid_argument(
          "TinyViTConfig expects exactly 4 stages for embed_dims/depths/num_heads");
    }
    if (image_size % 32 != 0) {
      throw std::invalid_argument(
          "image_size must be divisible by 32 so the 4-stage hierarchy stays well-formed");
    }
  }
};

inline torch::Tensor layer_norm_last_dim(const torch::Tensor& x,
                                         torch::nn::LayerNorm& norm) {
  return norm->forward(x);
}

struct ConvBNActImpl : torch::nn::Module {
  ConvBNActImpl(int64_t in_channels,
                int64_t out_channels,
                int64_t kernel_size,
                int64_t stride,
                int64_t padding,
                int64_t groups = 1,
                bool use_activation = true)
      : conv(register_module(
            "conv",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels,
                                                       out_channels,
                                                       kernel_size)
                                  .stride(stride)
                                  .padding(padding)
                                  .groups(groups)
                                  .bias(false)))),
        bn(register_module("bn", torch::nn::BatchNorm2d(out_channels))),
        use_activation_(use_activation) {}

  torch::Tensor forward(const torch::Tensor& x) {
    auto y = conv->forward(x);
    y = bn->forward(y);
    if (use_activation_) {
      y = torch::gelu(y, "tanh");
    }
    return y;
  }

  torch::nn::Conv2d conv{nullptr};
  torch::nn::BatchNorm2d bn{nullptr};
  bool use_activation_;
};
TORCH_MODULE(ConvBNAct);

struct ConvStemImpl : torch::nn::Module {
  ConvStemImpl(int64_t in_channels, int64_t embed_dim)
      : conv1(register_module("conv1", ConvBNAct(in_channels, embed_dim / 2, 3, 2, 1))),
        conv2(register_module("conv2", ConvBNAct(embed_dim / 2, embed_dim, 3, 2, 1))) {}

  torch::Tensor forward(const torch::Tensor& x) {
    auto y = conv1->forward(x);  // [B, C/2, H/2, W/2]
    y = conv2->forward(y);       // [B, C,   H/4, W/4]
    return y;
  }

  ConvBNAct conv1{nullptr};
  ConvBNAct conv2{nullptr};
};
TORCH_MODULE(ConvStem);

struct MBConvImpl : torch::nn::Module {
  MBConvImpl(int64_t channels, int64_t expand_ratio = 4)
      : hidden_dim_(channels * expand_ratio),
        expand(register_module("expand", ConvBNAct(channels, hidden_dim_, 1, 1, 0))),
        depthwise(register_module(
            "depthwise",
            ConvBNAct(hidden_dim_, hidden_dim_, 3, 1, 1, hidden_dim_))),
        project(register_module(
            "project",
            ConvBNAct(hidden_dim_, channels, 1, 1, 0, 1, false))) {}

  torch::Tensor forward(const torch::Tensor& x) {
    auto y = expand->forward(x);
    y = depthwise->forward(y);
    y = project->forward(y);
    return x + y;
  }

  int64_t hidden_dim_;
  ConvBNAct expand{nullptr};
  ConvBNAct depthwise{nullptr};
  ConvBNAct project{nullptr};
};
TORCH_MODULE(MBConv);

} // namespace Marr


#endif