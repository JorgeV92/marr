#pragma once

#include <array>
#include <concepts>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

#include <marr/shape.hpp>

namespace marr::detail {

template <std::integral Index>
std::int64_t normalize_index(Index index)
{
    if constexpr (std::signed_integral<Index>) {
        if (index < 0) {
            throw std::out_of_range("Tensor index cannot be negative");
        }
    } else if (index > static_cast<Index>(std::numeric_limits<std::int64_t>::max())) {
        throw std::out_of_range("Tensor index is too large");
    }

    return static_cast<std::int64_t>(index);
}

template <std::integral... Indices>
std::array<std::int64_t, sizeof...(Indices)> make_index_array(Indices... indices)
{
    return {normalize_index(indices)...};
}

std::int64_t normalize_dim(std::int64_t dim, std::int64_t rank);
std::int64_t compute_offset(
    std::span<const std::int64_t> indices,
    const Sizes& sizes,
    const Strides& strides,
    std::int64_t storage_offset
);
std::int64_t compute_broadcast_offset(
    std::span<const std::int64_t> output_indices,
    const Sizes& input_sizes,
    const Strides& input_strides,
    std::int64_t storage_offset
);
Sizes unravel_index(std::int64_t flat_index, const Sizes& sizes, const Strides& strides);

} // namespace marr::detail
