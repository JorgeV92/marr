#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <optional>
#include <random>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include <marrvision/transforms/v2/types.hpp>

namespace marr {
namespace vision {
namespace transforms {
namespace v2 {
namespace functional {

namespace detail {

inline void check_image_tensor(const torch::Tensor& input) {
  if (!input.defined()) {
    throw Error("Expected a defined tensor");
  }
  if (input.dim() < 3) {
    throw Error("Expected image tensor with shape [..., C, H, W]");
  }
}

struct BatchedImage {
  torch::Tensor tensor;
  std::vector<int64_t> leading_shape;
  bool squeezed = false;
};

inline BatchedImage to_nchw(const torch::Tensor& input) {
  check_image_tensor(input);
  const auto dim = input.dim();
  const auto channels = input.size(dim - 3);
  const auto height = input.size(dim - 2);
  const auto width = input.size(dim - 1);

  if (dim == 3) {
    return {input.unsqueeze(0), {}, true};
  }

  int64_t batch = 1;
  std::vector<int64_t> leading_shape;
  leading_shape.reserve(static_cast<size_t>(dim - 3));
  for (int64_t i = 0; i < dim - 3; ++i) {
    leading_shape.push_back(input.size(i));
    batch *= input.size(i);
  }

  return {input.reshape({batch, channels, height, width}), leading_shape, false};
}

inline torch::Tensor from_nchw(const torch::Tensor& output, const BatchedImage& spec) {
  if (spec.squeezed) {
    return output.squeeze(0);
  }

  std::vector<int64_t> shape = spec.leading_shape;
  shape.push_back(output.size(1));
  shape.push_back(output.size(2));
  shape.push_back(output.size(3));
  return output.reshape(shape);
}

inline std::vector<int64_t> channel_view_shape(const torch::Tensor& input, int64_t channels) {
  std::vector<int64_t> shape(static_cast<size_t>(input.dim()), 1);
  shape[static_cast<size_t>(input.dim() - 3)] = channels;
  return shape;
}

inline torch::Tensor scalar_or_tensor_fill(const torch::Tensor& input, double value) {
  return torch::full_like(input, value);
}

inline torch::Tensor cast_like(const torch::Tensor& value, const torch::Tensor& like) {
  return value.to(like.scalar_type());
}

inline std::vector<int64_t> shuffled_order(int64_t n) {
  std::vector<int64_t> order(static_cast<size_t>(n));
  std::iota(order.begin(), order.end(), 0);
  static std::mt19937 generator(std::random_device{}());
  std::shuffle(order.begin(), order.end(), generator);
  return order;
}

}  // namespace detail

inline Chw query_chw(const torch::Tensor& input) {
  detail::check_image_tensor(input);
  return {input.size(input.dim() - 3), input.size(input.dim() - 2), input.size(input.dim() - 1)};
}

inline Size2D query_size(const torch::Tensor& input) {
  detail::check_image_tensor(input);
  return Size2D{input.size(input.dim() - 2), input.size(input.dim() - 1)};
}

inline Size2D compute_resized_output_size(Size2D input_size,
                                          int64_t shorter_edge,
                                          std::optional<int64_t> max_size = std::nullopt) {
  if (shorter_edge <= 0) {
    throw Error("Resize shorter edge must be positive");
  }

  const auto height = input_size.height;
  const auto width = input_size.width;
  const auto min_original = static_cast<double>(std::min(height, width));
  const auto max_original = static_cast<double>(std::max(height, width));
  double scale = static_cast<double>(shorter_edge) / min_original;

  if (max_size.has_value() && std::round(max_original * scale) > *max_size) {
    scale = static_cast<double>(*max_size) / max_original;
  }

  return Size2D{
      std::max<int64_t>(1, static_cast<int64_t>(std::round(static_cast<double>(height) * scale))),
      std::max<int64_t>(1, static_cast<int64_t>(std::round(static_cast<double>(width) * scale)))};
}

inline torch::Tensor resize(const torch::Tensor& input,
                            Size2D size,
                            InterpolationMode interpolation = InterpolationMode::Bilinear,
                            bool antialias = true) {
  (void)antialias;
  auto spec = detail::to_nchw(input);

  namespace F = torch::nn::functional;
  auto options = F::InterpolateFuncOptions().size(size.vec());

  switch (interpolation) {
    case InterpolationMode::Nearest:
    case InterpolationMode::NearestExact:
      options = options.mode(torch::kNearest);
      break;
    case InterpolationMode::Bilinear:
      options = options.mode(torch::kBilinear).align_corners(false);
      break;
    case InterpolationMode::Bicubic:
      options = options.mode(torch::kBicubic).align_corners(false);
      break;
    case InterpolationMode::Lanczos:
    case InterpolationMode::Box:
    case InterpolationMode::Hamming:
      throw NotImplementedError("resize interpolation mode " + to_string(interpolation));
  }

  return detail::from_nchw(F::interpolate(spec.tensor, options), spec);
}

inline torch::Tensor resize_shorter_edge(const torch::Tensor& input,
                                         int64_t shorter_edge,
                                         std::optional<int64_t> max_size = std::nullopt,
                                         InterpolationMode interpolation = InterpolationMode::Bilinear,
                                         bool antialias = true) {
  return resize(input,
                compute_resized_output_size(query_size(input), shorter_edge, max_size),
                interpolation,
                antialias);
}

inline torch::Tensor crop(const torch::Tensor& input, CropParams crop_params) {
  detail::check_image_tensor(input);
  const auto size = query_size(input);
  if (crop_params.top < 0 || crop_params.left < 0 || crop_params.height <= 0 ||
      crop_params.width <= 0 || crop_params.top + crop_params.height > size.height ||
      crop_params.left + crop_params.width > size.width) {
    throw Error("Crop parameters are outside the image bounds");
  }

  return input.narrow(input.dim() - 2, crop_params.top, crop_params.height)
      .narrow(input.dim() - 1, crop_params.left, crop_params.width);
}

inline torch::Tensor horizontal_flip(const torch::Tensor& input) {
  detail::check_image_tensor(input);
  return input.flip({input.dim() - 1});
}

inline torch::Tensor vertical_flip(const torch::Tensor& input) {
  detail::check_image_tensor(input);
  return input.flip({input.dim() - 2});
}

inline torch::Tensor pad(const torch::Tensor& input,
                         Padding padding,
                         double fill = 0.0,
                         PadMode padding_mode = PadMode::Constant) {
  detail::check_image_tensor(input);
  if (padding.left < 0 || padding.right < 0 || padding.top < 0 || padding.bottom < 0) {
    throw Error("Padding values must be non-negative");
  }

  if (padding_mode != PadMode::Constant) {
    throw NotImplementedError("non-constant padding modes");
  }

  return torch::constant_pad_nd(input, padding.torch_order(), fill);
}

inline torch::Tensor center_crop(const torch::Tensor& input, Size2D output_size) {
  detail::check_image_tensor(input);
  auto working = input;
  auto size = query_size(working);

  const auto pad_left = std::max<int64_t>((output_size.width - size.width) / 2, 0);
  const auto pad_right = std::max<int64_t>(output_size.width - size.width - pad_left, 0);
  const auto pad_top = std::max<int64_t>((output_size.height - size.height) / 2, 0);
  const auto pad_bottom = std::max<int64_t>(output_size.height - size.height - pad_top, 0);
  if (pad_left || pad_right || pad_top || pad_bottom) {
    working = pad(working, Padding{pad_left, pad_top, pad_right, pad_bottom});
    size = query_size(working);
  }

  return crop(working,
              CropParams{(size.height - output_size.height) / 2,
                         (size.width - output_size.width) / 2,
                         output_size.height,
                         output_size.width});
}

inline torch::Tensor resized_crop(const torch::Tensor& input,
                                  CropParams crop_params,
                                  Size2D size,
                                  InterpolationMode interpolation = InterpolationMode::Bilinear,
                                  bool antialias = true) {
  return resize(crop(input, crop_params), size, interpolation, antialias);
}

inline std::vector<torch::Tensor> five_crop(const torch::Tensor& input, Size2D size) {
  const auto image_size = query_size(input);
  if (size.height > image_size.height || size.width > image_size.width) {
    throw Error("Requested crop size is bigger than input size");
  }

  const auto bottom = image_size.height - size.height;
  const auto right = image_size.width - size.width;
  return {
      crop(input, CropParams{0, 0, size.height, size.width}),
      crop(input, CropParams{0, right, size.height, size.width}),
      crop(input, CropParams{bottom, 0, size.height, size.width}),
      crop(input, CropParams{bottom, right, size.height, size.width}),
      center_crop(input, size),
  };
}

inline std::vector<torch::Tensor> ten_crop(const torch::Tensor& input,
                                           Size2D size,
                                           bool vertical = false) {
  auto first = five_crop(input, size);
  auto flipped = five_crop(vertical ? vertical_flip(input) : horizontal_flip(input), size);
  first.insert(first.end(), flipped.begin(), flipped.end());
  return first;
}

inline torch::Tensor normalize(const torch::Tensor& input,
                               const std::vector<double>& mean,
                               const std::vector<double>& std) {
  detail::check_image_tensor(input);
  const auto channels = input.size(input.dim() - 3);
  if (static_cast<int64_t>(mean.size()) != channels || static_cast<int64_t>(std.size()) != channels) {
    throw Error("Normalize mean/std length must match the image channel count");
  }

  auto work = input.to(torch::kFloat32);
  auto options = work.options();
  auto mean_tensor = torch::tensor(mean, options).view(detail::channel_view_shape(work, channels));
  auto std_tensor = torch::tensor(std, options).view(detail::channel_view_shape(work, channels));
  return (work - mean_tensor) / std_tensor;
}

inline torch::Tensor to_dtype(const torch::Tensor& input, torch::ScalarType dtype, bool scale = false) {
  if (!scale) {
    return input.to(dtype);
  }

  const auto source = input.scalar_type();
  const auto source_float = is_floating_point(source);
  const auto target_float = is_floating_point(dtype);

  if (!source_float && target_float) {
    return input.to(dtype) / max_value_for_dtype(source);
  }
  if (source_float && !target_float) {
    return torch::clamp(input, 0.0, 1.0).mul(max_value_for_dtype(dtype)).round().to(dtype);
  }
  return input.to(dtype);
}

inline torch::Tensor grayscale(const torch::Tensor& input, int64_t num_output_channels = 1) {
  detail::check_image_tensor(input);
  const auto channel_dim = input.dim() - 3;
  const auto channels = input.size(channel_dim);
  if (channels == 1) {
    if (num_output_channels == 1) {
      return input;
    }
    if (num_output_channels == 3) {
      std::vector<int64_t> repeats(static_cast<size_t>(input.dim()), 1);
      repeats[static_cast<size_t>(channel_dim)] = 3;
      return input.repeat(repeats);
    }
  }
  if (channels != 3) {
    throw Error("Grayscale expects a 1-channel or 3-channel image");
  }
  if (num_output_channels != 1 && num_output_channels != 3) {
    throw Error("Grayscale output channels must be 1 or 3");
  }

  auto work = input.to(torch::kFloat32);
  auto gray = work.narrow(channel_dim, 0, 1).mul(0.2989) +
              work.narrow(channel_dim, 1, 1).mul(0.5870) +
              work.narrow(channel_dim, 2, 1).mul(0.1140);
  if (num_output_channels == 3) {
    std::vector<int64_t> repeats(static_cast<size_t>(input.dim()), 1);
    repeats[static_cast<size_t>(channel_dim)] = 3;
    gray = gray.repeat(repeats);
  }
  return detail::cast_like(gray, input);
}

inline torch::Tensor rgb(const torch::Tensor& input) {
  detail::check_image_tensor(input);
  const auto channel_dim = input.dim() - 3;
  const auto channels = input.size(channel_dim);
  if (channels == 3) {
    return input;
  }
  if (channels == 1) {
    std::vector<int64_t> repeats(static_cast<size_t>(input.dim()), 1);
    repeats[static_cast<size_t>(channel_dim)] = 3;
    return input.repeat(repeats);
  }
  throw Error("RGB expects a 1-channel or 3-channel image");
}

inline torch::Tensor adjust_brightness(const torch::Tensor& input, double factor) {
  return torch::clamp(input.to(torch::kFloat32) * factor, 0.0, max_value_for_dtype(input.scalar_type()))
      .to(input.scalar_type());
}

inline torch::Tensor adjust_contrast(const torch::Tensor& input, double factor) {
  detail::check_image_tensor(input);
  auto work = input.to(torch::kFloat32);
  auto mean = grayscale(work, 1).mean({input.dim() - 2, input.dim() - 1}, true);
  auto out = (work - mean) * factor + mean;
  return torch::clamp(out, 0.0, max_value_for_dtype(input.scalar_type())).to(input.scalar_type());
}

inline torch::Tensor adjust_saturation(const torch::Tensor& input, double factor) {
  detail::check_image_tensor(input);
  auto work = input.to(torch::kFloat32);
  auto gray = grayscale(work, 1);
  std::vector<int64_t> repeats(static_cast<size_t>(input.dim()), 1);
  repeats[static_cast<size_t>(input.dim() - 3)] = input.size(input.dim() - 3);
  gray = gray.repeat(repeats);
  auto out = (work - gray) * factor + gray;
  return torch::clamp(out, 0.0, max_value_for_dtype(input.scalar_type())).to(input.scalar_type());
}

inline torch::Tensor invert(const torch::Tensor& input) {
  return max_value_for_dtype(input.scalar_type()) - input;
}

inline torch::Tensor solarize(const torch::Tensor& input, double threshold) {
  return torch::where(input >= threshold, invert(input), input);
}

inline torch::Tensor posterize(const torch::Tensor& input, int64_t bits) {
  if (bits < 1 || bits > 8) {
    throw Error("Posterize bits must be in [1, 8]");
  }
  if (input.scalar_type() != torch::kUInt8) {
    throw Error("Posterize currently expects uint8 tensors");
  }
  const auto levels = std::pow(2.0, static_cast<double>(bits));
  const auto step = 256.0 / levels;
  return torch::floor(input.to(torch::kFloat32) / step).mul(step).to(torch::kUInt8);
}

inline torch::Tensor autocontrast(const torch::Tensor& input) {
  detail::check_image_tensor(input);
  auto work = input.to(torch::kFloat32);
  auto flat = work.flatten(work.dim() - 2, work.dim() - 1);
  auto min_values = std::get<0>(flat.min(-1, true));
  auto max_values = std::get<0>(flat.max(-1, true));
  auto view_shape = input.sizes().vec();
  view_shape[static_cast<size_t>(input.dim() - 2)] = 1;
  view_shape[static_cast<size_t>(input.dim() - 1)] = 1;
  min_values = min_values.view(view_shape);
  max_values = max_values.view(view_shape);
  auto scale = max_value_for_dtype(input.scalar_type()) / torch::clamp(max_values - min_values, 1e-12);
  auto out = (work - min_values) * scale;
  return torch::clamp(out, 0.0, max_value_for_dtype(input.scalar_type())).to(input.scalar_type());
}

inline torch::Tensor gaussian_noise(const torch::Tensor& input, double mean = 0.0, double sigma = 0.1) {
  auto noise = torch::randn_like(input.to(torch::kFloat32)).mul(sigma).add(mean);
  if (is_floating_point(input.scalar_type())) {
    return input + noise.to(input.scalar_type());
  }
  return torch::clamp(input.to(torch::kFloat32) + noise, 0.0, max_value_for_dtype(input.scalar_type()))
      .to(input.scalar_type());
}

inline torch::Tensor erase(const torch::Tensor& input,
                           CropParams region,
                           double value = 0.0,
                           bool inplace = false) {
  auto output = inplace ? input : input.clone();
  auto view = output.narrow(output.dim() - 2, region.top, region.height)
                  .narrow(output.dim() - 1, region.left, region.width);
  view.fill_(value);
  return output;
}

inline torch::Tensor linear_transformation(const torch::Tensor& input,
                                           const torch::Tensor& transformation_matrix,
                                           const torch::Tensor& mean_vector) {
  auto flat = input.flatten();
  if (flat.numel() != mean_vector.numel()) {
    throw Error("LinearTransformation mean_vector length must match flattened input");
  }
  return torch::matmul(transformation_matrix, flat - mean_vector).reshape_as(input);
}

inline torch::Tensor uniform_temporal_subsample(const torch::Tensor& input, int64_t num_samples) {
  if (num_samples <= 0) {
    throw Error("UniformTemporalSubsample expects a positive number of samples");
  }
  if (input.dim() < 4) {
    throw Error("UniformTemporalSubsample expects video shape [..., T, C, H, W]");
  }
  const auto temporal_dim = input.dim() - 4;
  const auto frames = input.size(temporal_dim);
  auto indices = torch::linspace(0,
                                 std::max<int64_t>(frames - 1, 0),
                                 num_samples,
                                 torch::TensorOptions().dtype(torch::kFloat32).device(input.device()))
                     .round()
                     .to(torch::kLong);
  return input.index_select(temporal_dim, indices);
}

}  // namespace functional
}  // namespace v2
}  // namespace transforms
}  // namespace vision
}  // namespace marr
