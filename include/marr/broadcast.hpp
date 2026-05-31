#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include <marr/indexing.hpp>
#include <marr/tensor_core.hpp>

namespace marr {

Sizes broadcast_shapes(const Sizes& lhs, const Sizes& rhs);

template <typename T>
Tensor<T> sum_to_shape(const Tensor<T>& grad, const Sizes& target_shape)
{
    if (target_shape.size() > grad.sizes().size()) {
        throw std::invalid_argument("sum_to_shape target rank cannot exceed gradient rank");
    }

    for (std::size_t i = 0; i < target_shape.size(); ++i) {
        const std::size_t grad_dim = grad.sizes().size() - target_shape.size() + i;
        const std::int64_t target_extent = target_shape[i];
        const std::int64_t grad_extent = grad.sizes()[grad_dim];
        if (target_extent != grad_extent && target_extent != 1) {
            throw std::invalid_argument("sum_to_shape target shape is not broadcast-compatible");
        }
    }

    Tensor<T> result(target_shape, T{});
    const Strides grad_strides = compute_contiguous_strides(grad.sizes());
    const Strides target_strides = compute_contiguous_strides(target_shape);

    for (std::int64_t flat = 0; flat < grad.numel(); ++flat) {
        if (target_shape.empty()) {
            result[0] += grad[flat];
            continue;
        }

        const Sizes grad_indices = detail::unravel_index(flat, grad.sizes(), grad_strides);
        Sizes target_indices(target_shape.size(), 0);
        const std::size_t rank_offset = grad.sizes().size() - target_shape.size();

        for (std::size_t dim = 0; dim < target_shape.size(); ++dim) {
            target_indices[dim] = target_shape[dim] == 1 ? 0 : grad_indices[rank_offset + dim];
        }

        const std::int64_t target_flat = detail::compute_offset(
            target_indices,
            target_shape,
            target_strides,
            0
        );
        result[target_flat] += grad[flat];
    }

    return result;
}

} // namespace marr
