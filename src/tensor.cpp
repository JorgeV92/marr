#include <marr/tensor.hpp>

#include <limits>

// Tensor<T> is implemented in the header because it is a class template.
// The non-template shape and broadcasting helpers live here so the library
// target has a normal translation unit with real symbols.

namespace marr {

namespace {

void ensure_nonnegative_size(std::int64_t extent)
{
    if (extent < 0) {
        throw std::invalid_argument(
            "Tensor sizes must be non-negative, got " + std::to_string(extent)
        );
    }
}

void ensure_multiply_fits(std::int64_t lhs, std::int64_t rhs, const char* message)
{
    if (lhs != 0 && rhs > std::numeric_limits<std::int64_t>::max() / lhs) {
        throw std::invalid_argument(message);
    }
}

void ensure_add_fits(std::int64_t lhs, std::int64_t rhs, const char* message)
{
    if (rhs > std::numeric_limits<std::int64_t>::max() - lhs) {
        throw std::invalid_argument(message);
    }
}

std::int64_t checked_product(std::int64_t lhs, std::int64_t rhs, const char* message)
{
    ensure_multiply_fits(lhs, rhs, message);
    return lhs * rhs;
}

} // namespace

namespace detail {

std::int64_t normalize_dim(std::int64_t dim, std::int64_t rank)
{
    if (rank < 0) {
        throw std::invalid_argument("Tensor rank cannot be negative");
    }

    const std::int64_t normalized = dim < 0 ? dim + rank : dim;
    if (normalized < 0 || normalized >= rank) {
        throw std::out_of_range(
            "Tensor dimension out of range: dim " + std::to_string(dim) +
            " for rank " + std::to_string(rank)
        );
    }

    return normalized;
}

std::int64_t compute_offset(
    std::span<const std::int64_t> indices,
    const Sizes& sizes,
    const Strides& strides,
    std::int64_t storage_offset
)
{
    if (sizes.size() != strides.size()) {
        throw std::invalid_argument("Tensor sizes and strides rank mismatch");
    }

    if (indices.size() != sizes.size()) {
        throw std::invalid_argument(
            "Tensor index rank mismatch: got " + std::to_string(indices.size()) +
            " indices for rank " + std::to_string(sizes.size()) + " tensor"
        );
    }

    std::int64_t offset = storage_offset;
    for (std::size_t dim = 0; dim < indices.size(); ++dim) {
        const std::int64_t index = indices[dim];
        if (index < 0) {
            throw std::out_of_range(
                "Tensor index cannot be negative at dimension " + std::to_string(dim)
            );
        }
        if (index >= sizes[dim]) {
            throw std::out_of_range(
                "Tensor index out of bounds at dimension " + std::to_string(dim) +
                ": index " + std::to_string(index) +
                " >= size " + std::to_string(sizes[dim])
            );
        }

        const std::int64_t stride_offset = checked_product(
            index,
            strides[dim],
            "Tensor index offset overflows std::int64_t"
        );
        ensure_add_fits(offset, stride_offset, "Tensor index offset overflows std::int64_t");
        offset += stride_offset;
    }

    return offset;
}

std::int64_t compute_broadcast_offset(
    std::span<const std::int64_t> output_indices,
    const Sizes& input_sizes,
    const Strides& input_strides,
    std::int64_t storage_offset
)
{
    if (input_sizes.size() != input_strides.size()) {
        throw std::invalid_argument("Tensor sizes and strides rank mismatch");
    }
    if (input_sizes.size() > output_indices.size()) {
        throw std::invalid_argument("Cannot broadcast input with more dimensions than output");
    }

    std::int64_t offset = storage_offset;
    const std::size_t rank_offset = output_indices.size() - input_sizes.size();

    for (std::size_t dim = 0; dim < input_sizes.size(); ++dim) {
        const std::int64_t input_index = input_sizes[dim] == 1 ? 0 : output_indices[rank_offset + dim];
        if (input_index < 0 || input_index >= input_sizes[dim]) {
            throw std::out_of_range("Broadcast index is out of bounds for input tensor");
        }

        const std::int64_t stride_offset = checked_product(
            input_index,
            input_strides[dim],
            "Tensor broadcast offset overflows std::int64_t"
        );
        ensure_add_fits(offset, stride_offset, "Tensor broadcast offset overflows std::int64_t");
        offset += stride_offset;
    }

    return offset;
}

Sizes unravel_index(std::int64_t flat_index, const Sizes& sizes, const Strides& strides)
{
    if (flat_index < 0) {
        throw std::out_of_range("Tensor flat index cannot be negative");
    }
    if (sizes.size() != strides.size()) {
        throw std::invalid_argument("Tensor sizes and strides rank mismatch");
    }

    Sizes indices(sizes.size(), 0);
    std::int64_t remaining = flat_index;
    for (std::size_t dim = 0; dim < sizes.size(); ++dim) {
        if (strides[dim] == 0) {
            indices[dim] = 0;
        } else {
            indices[dim] = remaining / strides[dim];
            remaining %= strides[dim];
        }
    }

    return indices;
}

} // namespace detail

std::int64_t compute_numel(const Sizes& sizes)
{
    std::int64_t total = 1;

    for (const std::int64_t extent : sizes) {
        ensure_nonnegative_size(extent);
        if (extent == 0) {
            return 0;
        }

        total = checked_product(
            total,
            extent,
            "Tensor shape is too large: element count overflows std::int64_t"
        );
    }

    return total;
}

Strides compute_contiguous_strides(const Sizes& sizes)
{
    for (const std::int64_t extent : sizes) {
        ensure_nonnegative_size(extent);
    }

    Strides strides(sizes.size(), 1);
    if (sizes.empty()) {
        return strides;
    }

    // Row-major layout stores the last dimension contiguously. Its stride is 1.
    // Each earlier stride is the product of all extents to its right:
    // sizes {2, 3, 4} has strides {12, 4, 1}.
    for (std::size_t dim = sizes.size() - 1; dim > 0; --dim) {
        strides[dim - 1] = checked_product(
            sizes[dim],
            strides[dim],
            "Tensor shape is too large: stride computation overflows std::int64_t"
        );
    }

    return strides;
}

Sizes broadcast_shapes(const Sizes& lhs, const Sizes& rhs)
{
    const std::size_t result_rank = lhs.size() > rhs.size() ? lhs.size() : rhs.size();
    Sizes result(result_rank, 1);

    for (std::size_t i = 0; i < result_rank; ++i) {
        const std::int64_t lhs_dim = i < lhs.size() ? lhs[lhs.size() - 1 - i] : 1;
        const std::int64_t rhs_dim = i < rhs.size() ? rhs[rhs.size() - 1 - i] : 1;

        ensure_nonnegative_size(lhs_dim);
        ensure_nonnegative_size(rhs_dim);

        if (lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1) {
            throw std::invalid_argument(
                "Tensor shapes cannot broadcast: dimension from the right " +
                std::to_string(i) + " has sizes " + std::to_string(lhs_dim) +
                " and " + std::to_string(rhs_dim)
            );
        }

        result[result_rank - 1 - i] = lhs_dim > rhs_dim ? lhs_dim : rhs_dim;
    }

    return result;
}

std::size_t compute_total_size(const Shape& shape)
{
    return static_cast<std::size_t>(compute_numel(shape));
}

Shape compute_row_major_strides(const Shape& shape)
{
    return compute_contiguous_strides(shape);
}

} // namespace marr
