#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace marr {

using Shape = std::vector<std::int64_t>;
using Sizes = Shape;
using Strides = Shape;

template <typename T>
class Tensor;

namespace detail {

template <typename T>
struct is_tensor : std::false_type {
};

template <typename T>
struct is_tensor<Tensor<T>> : std::true_type {
};

template <typename T>
inline constexpr bool is_tensor_v = is_tensor<std::remove_cvref_t<T>>::value;

template <typename Scalar, typename T>
concept TensorScalar = std::convertible_to<Scalar, T> && !is_tensor_v<Scalar>;

} // namespace detail

std::int64_t compute_numel(const Sizes& sizes);
Strides compute_contiguous_strides(const Sizes& sizes);

std::size_t compute_total_size(const Shape& shape);
Shape compute_row_major_strides(const Shape& shape);

} // namespace marr
