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

struct ConvStageImpl : torch::nn::Module {
  explicit ConvStageImpl(int64_t channels, int64_t depth)
      : blocks(register_module("blocks", torch::nn::ModuleList())) {
    for (int64_t i = 0; i < depth; ++i) {
      blocks->push_back(MBConv(channels));
    }
  }

  torch::Tensor forward(torch::Tensor x) {
    for (auto& block : *blocks) {
      x = block->as<MBConv>()->forward(x);
    }
    return x;
  }

  torch::nn::ModuleList blocks{nullptr};
};
TORCH_MODULE(ConvStage);

struct PatchMergingImpl : torch::nn::Module {
  PatchMergingImpl(int64_t in_channels, int64_t out_channels)
      : reduce(register_module("reduce", ConvBNAct(in_channels, out_channels, 1, 1, 0))),
        downsample(register_module(
            "downsample",
            ConvBNAct(out_channels, out_channels, 3, 2, 1, out_channels))),
        project(register_module("project", ConvBNAct(out_channels, out_channels, 1, 1, 0))) {}

  torch::Tensor forward(const torch::Tensor& x) {
    auto y = reduce->forward(x);
    y = downsample->forward(y);
    y = project->forward(y);
    return y;
  }

  ConvBNAct reduce{nullptr};
  ConvBNAct downsample{nullptr};
  ConvBNAct project{nullptr};
};
TORCH_MODULE(PatchMerging);

struct MlpImpl : torch::nn::Module {
  MlpImpl(int64_t dim, int64_t mlp_ratio)
      : norm(register_module(
            "norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{dim})))),
        fc1(register_module("fc1", torch::nn::Linear(dim, dim * mlp_ratio))),
        fc2(register_module("fc2", torch::nn::Linear(dim * mlp_ratio, dim))) {}

  torch::Tensor forward(const torch::Tensor& x) {
    auto y = norm->forward(x);
    y = fc1->forward(y);
    y = torch::gelu(y, "tanh");
    y = fc2->forward(y);
    return y;
  }

  torch::nn::LayerNorm norm{nullptr};
  torch::nn::Linear fc1{nullptr};
  torch::nn::Linear fc2{nullptr};
};
TORCH_MODULE(Mlp);

inline torch::Tensor build_relative_position_index(int64_t window_size) {
  const int64_t tokens = window_size * window_size;
  auto coords_h = torch::arange(window_size, torch::kLong);
  auto coords_w = torch::arange(window_size, torch::kLong);
  auto mesh = torch::meshgrid({coords_h, coords_w}, "ij");
  auto coords = torch::stack({mesh[0], mesh[1]});                // [2, Ws, Ws]
  auto flat = coords.view({2, tokens});                          // [2, N]
  auto rel = flat.unsqueeze(2) - flat.unsqueeze(1);              // [2, N, N]
  rel = rel.permute({1, 2, 0}).contiguous();                     // [N, N, 2]
  rel.index_put_({Slice(), Slice(), 0}, rel.index({Slice(), Slice(), 0}) + window_size - 1);
  rel.index_put_({Slice(), Slice(), 1}, rel.index({Slice(), Slice(), 1}) + window_size - 1);
  rel.index_put_({Slice(), Slice(), 0}, rel.index({Slice(), Slice(), 0}) * (2 * window_size - 1));
  return rel.sum(-1);                                            // [N, N]
}

struct WindowAttentionImpl : torch::nn::Module {
  WindowAttentionImpl(int64_t dim, int64_t num_heads, int64_t window_size)
      : dim_(dim),
        num_heads_(num_heads),
        window_size_(window_size),
        head_dim_(dim / num_heads),
        scale_(1.0 / std::sqrt(static_cast<double>(head_dim_))),
        norm(register_module(
            "norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{dim})))),
        qkv(register_module("qkv", torch::nn::Linear(dim, 3 * dim))),
        proj(register_module("proj", torch::nn::Linear(dim, dim))) {
    if (dim % num_heads != 0) {
      throw std::invalid_argument("dim must be divisible by num_heads");
    }
    const int64_t bias_size = (2 * window_size - 1) * (2 * window_size - 1);
    relative_position_bias_table = register_parameter(
        "relative_position_bias_table",
        torch::zeros({bias_size, num_heads}, torch::TensorOptions().dtype(torch::kFloat32)));
    relative_position_index = register_buffer(
        "relative_position_index", build_relative_position_index(window_size));
    torch::nn::init::normal_(relative_position_bias_table, 0.0, 0.02);
  }

  torch::Tensor forward(const torch::Tensor& x) {
    // x: [B_windows, tokens_per_window, dim]
    const auto B = x.size(0);
    const auto N = x.size(1);

    auto y = norm->forward(x);
    auto qkv_out = qkv->forward(y)
                       .view({B, N, 3, num_heads_, head_dim_})
                       .permute({2, 0, 3, 1, 4})
                       .contiguous();

    auto q = qkv_out.select(0, 0);  // [B, heads, N, head_dim]
    auto k = qkv_out.select(0, 1);
    auto v = qkv_out.select(0, 2);

    auto attn = torch::matmul(q, k.transpose(-2, -1)) * scale_;  // [B, heads, N, N]

    auto bias_index = relative_position_index.view({-1});
    auto bias = relative_position_bias_table.index_select(0, bias_index)
                    .view({N, N, num_heads_})
                    .permute({2, 0, 1})
                    .contiguous();                                // [heads, N, N]
    attn = attn + bias.unsqueeze(0);
    attn = torch::softmax(attn, -1);

    auto context = torch::matmul(attn, v)
                       .transpose(1, 2)
                       .contiguous()
                       .view({B, N, dim_});
    return proj->forward(context);
  }

  int64_t dim_;
  int64_t num_heads_;
  int64_t window_size_;
  int64_t head_dim_;
  double scale_;
  torch::nn::LayerNorm norm{nullptr};
  torch::nn::Linear qkv{nullptr};
  torch::nn::Linear proj{nullptr};
  torch::Tensor relative_position_bias_table;
  torch::Tensor relative_position_index;
};
TORCH_MODULE(WindowAttention);

} // namespace Marr


#endif