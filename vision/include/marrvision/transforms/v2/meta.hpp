#pragma once

#include <torch/torch.h>

#include <marrvision/transforms/v2/transform.hpp>
#include <marrvision/transforms/v2/types.hpp>

namespace marr {
namespace vision {
namespace transforms {
namespace v2 {

struct BoundingBoxes {
  torch::Tensor data;
  BoundingBoxFormat format = BoundingBoxFormat::XYXY;
  Size2D canvas_size;
};

struct KeyPoints {
  torch::Tensor data;
  Size2D canvas_size;
};

MARRVISION_V2_UNSUPPORTED_TRANSFORM(ClampBoundingBoxes);
MARRVISION_V2_UNSUPPORTED_TRANSFORM(ClampKeyPoints);
MARRVISION_V2_UNSUPPORTED_TRANSFORM(ConvertBoundingBoxFormat);
MARRVISION_V2_UNSUPPORTED_TRANSFORM(SetClampingMode);

}  // namespace v2
}  // namespace transforms
}  // namespace vision
}  // namespace marr
