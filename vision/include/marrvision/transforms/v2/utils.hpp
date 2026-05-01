#pragma once

#include <type_traits>

#include <torch/torch.h>

#include <marrvision/transforms/v2/functional.hpp>
#include <marrvision/transforms/v2/meta.hpp>
#include <marrvision/transforms/v2/types.hpp>

namespace marr {
namespace vision {
namespace transforms {
namespace v2 {

using functional::query_chw;
using functional::query_size;

template <typename T, typename U>
inline bool check_type(const U& value) {
  if constexpr (std::is_polymorphic_v<U>) {
    return dynamic_cast<const T*>(&value) != nullptr;
  } else {
    return std::is_same_v<std::decay_t<U>, T>;
  }
}

inline const BoundingBoxes& get_bounding_boxes(const BoundingBoxes& boxes) {
  return boxes;
}

inline const KeyPoints& get_keypoints(const KeyPoints& keypoints) {
  return keypoints;
}

template <typename... Args>
inline bool has_any(Args&&... args) {
  return ((args) || ...);
}

template <typename... Args>
inline bool has_all(Args&&... args) {
  return ((args) && ...);
}

}  // namespace v2
}  // namespace transforms
}  // namespace vision
}  // namespace marr
