#pragma once

#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iosfwd>
#include <limits>
#include <ostream>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
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

} // namespace detail

std::int64_t compute_numel(const Sizes& sizes);
Strides compute_contiguous_strides(const Sizes& sizes);
Sizes broadcast_shapes(const Sizes& lhs, const Sizes& rhs);

std::size_t compute_total_size(const Shape& shape);
Shape compute_row_major_strides(const Shape& shape);

template <typename T>
class Tensor {
public:
    using value_type = T;
    using size_type = std::int64_t;

    Tensor() = default;

    explicit Tensor(Sizes sizes)
        : sizes_(std::move(sizes)),
          strides_(compute_contiguous_strides(sizes_)),
          storage_offset_(0),
          data_(static_cast<std::size_t>(compute_numel(sizes_)))
    {
    }

    Tensor(Sizes sizes, const T& fill_value)
        : sizes_(std::move(sizes)),
          strides_(compute_contiguous_strides(sizes_)),
          storage_offset_(0),
          data_(static_cast<std::size_t>(compute_numel(sizes_)), fill_value)
    {
    }

    Tensor(Sizes sizes, std::vector<T> data)
        : sizes_(std::move(sizes)),
          strides_(compute_contiguous_strides(sizes_)),
          storage_offset_(0),
          data_(std::move(data))
    {
        const std::int64_t expected_size = compute_numel(sizes_);
        if (static_cast<std::int64_t>(data_.size()) != expected_size) {
            throw std::invalid_argument(
                "Tensor data size mismatch: shape requires " + std::to_string(expected_size) +
                " elements but data has " + std::to_string(data_.size())
            );
        }
    }

    [[nodiscard]] std::int64_t dim() const noexcept
    {
        return static_cast<std::int64_t>(sizes_.size());
    }

    [[nodiscard]] std::int64_t ndim() const noexcept
    {
        return dim();
    }

    [[nodiscard]] std::int64_t numel() const noexcept
    {
        return static_cast<std::int64_t>(data_.size());
    }

    [[nodiscard]] std::int64_t size() const noexcept
    {
        return numel();
    }

    [[nodiscard]] std::int64_t size(std::int64_t dim) const
    {
        return sizes_.at(static_cast<std::size_t>(detail::normalize_dim(dim, this->dim())));
    }

    [[nodiscard]] std::int64_t stride(std::int64_t dim) const
    {
        return strides_.at(static_cast<std::size_t>(detail::normalize_dim(dim, this->dim())));
    }

    [[nodiscard]] const Sizes& sizes() const noexcept
    {
        return sizes_;
    }

    [[nodiscard]] const Shape& shape() const noexcept
    {
        return sizes_;
    }

    [[nodiscard]] const Strides& strides() const noexcept
    {
        return strides_;
    }

    [[nodiscard]] std::int64_t storage_offset() const noexcept
    {
        return storage_offset_;
    }

    [[nodiscard]] bool is_contiguous() const
    {
        return strides_ == compute_contiguous_strides(sizes_);
    }

    [[nodiscard]] T* data_ptr() noexcept
    {
        if (data_.empty()) {
            return data_.data();
        }
        return data_.data() + storage_offset_;
    }

    [[nodiscard]] const T* data_ptr() const noexcept
    {
        if (data_.empty()) {
            return data_.data();
        }
        return data_.data() + storage_offset_;
    }

    [[nodiscard]] T* data() noexcept
    {
        return data_ptr();
    }

    [[nodiscard]] const T* data() const noexcept
    {
        return data_ptr();
    }

    [[nodiscard]] bool empty() const noexcept
    {
        return numel() == 0;
    }

    template <std::integral Index>
    T& operator[](Index index)
    {
        return data_[checked_flat_index(detail::normalize_index(index))];
    }

    template <std::integral Index>
    const T& operator[](Index index) const
    {
        return data_[checked_flat_index(detail::normalize_index(index))];
    }

    template <std::integral... Indices>
    T& operator()(Indices... indices)
    {
        const auto index_array = detail::make_index_array(indices...);
        return at(std::span<const std::int64_t>(index_array));
    }

    template <std::integral... Indices>
    const T& operator()(Indices... indices) const
    {
        const auto index_array = detail::make_index_array(indices...);
        return at(std::span<const std::int64_t>(index_array));
    }

    T& operator()(std::initializer_list<std::int64_t> indices)
    {
        return at(indices);
    }

    const T& operator()(std::initializer_list<std::int64_t> indices) const
    {
        return at(indices);
    }

    T& operator()(const Sizes& indices)
    {
        return at(indices);
    }

    const T& operator()(const Sizes& indices) const
    {
        return at(indices);
    }

    T& at(std::initializer_list<std::int64_t> indices)
    {
        return at(std::span<const std::int64_t>(indices.begin(), indices.size()));
    }

    const T& at(std::initializer_list<std::int64_t> indices) const
    {
        return at(std::span<const std::int64_t>(indices.begin(), indices.size()));
    }

    T& at(const Sizes& indices)
    {
        return at(std::span<const std::int64_t>(indices));
    }

    const T& at(const Sizes& indices) const
    {
        return at(std::span<const std::int64_t>(indices));
    }

    T& at(std::span<const std::int64_t> indices)
    {
        return data_.at(static_cast<std::size_t>(
            detail::compute_offset(indices, sizes_, strides_, storage_offset_)
        ));
    }

    const T& at(std::span<const std::int64_t> indices) const
    {
        return data_.at(static_cast<std::size_t>(
            detail::compute_offset(indices, sizes_, strides_, storage_offset_)
        ));
    }

    Tensor reshape(Sizes new_sizes) const
    {
        return reshape_like(std::move(new_sizes), "reshape");
    }

    Tensor view(Sizes new_sizes) const
    {
        return reshape_like(std::move(new_sizes), "view");
    }

private:
    std::size_t checked_flat_index(std::int64_t index) const
    {
        if (index < 0) {
            throw std::out_of_range("Tensor flat index cannot be negative");
        }
        if (index >= numel()) {
            throw std::out_of_range(
                "Tensor flat index out of range: index " + std::to_string(index) +
                " >= numel " + std::to_string(numel())
            );
        }

        return static_cast<std::size_t>(storage_offset_ + index);
    }

    Tensor reshape_like(Sizes new_sizes, const char* operation) const
    {
        if (!is_contiguous()) {
            throw std::invalid_argument(
                std::string("Tensor ") + operation + " requires a contiguous tensor"
            );
        }

        const std::int64_t new_numel = compute_numel(new_sizes);
        if (new_numel != numel()) {
            throw std::invalid_argument(
                "Tensor " + std::string(operation) + " size mismatch: current tensor has " +
                std::to_string(numel()) + " elements but requested shape requires " +
                std::to_string(new_numel)
            );
        }

        Tensor result(new_sizes, data_);
        result.storage_offset_ = 0;
        return result;
    }

    Sizes sizes_;
    Strides strides_;
    std::int64_t storage_offset_ = 0;
    std::vector<T> data_;
};

template <typename T>
Tensor<T> empty(Sizes sizes)
{
    return Tensor<T>(std::move(sizes));
}

template <typename T>
Tensor<T> zeros(Sizes sizes)
{
    return Tensor<T>(std::move(sizes), T{});
}

template <typename T>
Tensor<T> ones(Sizes sizes)
{
    return Tensor<T>(std::move(sizes), static_cast<T>(1));
}

template <typename T>
Tensor<T> full(Sizes sizes, const T& value)
{
    return Tensor<T>(std::move(sizes), value);
}

namespace detail {

template <typename T, typename UnaryOperation>
Tensor<T> unary_tensor_operation(const Tensor<T>& input, UnaryOperation operation)
{
    Tensor<T> result(input.sizes());
    for (std::int64_t i = 0; i < input.numel(); ++i) {
        result[i] = operation(input[i]);
    }
    return result;
}

template <typename T, typename BinaryOperation>
Tensor<T> binary_tensor_operation(
    const Tensor<T>& lhs,
    const Tensor<T>& rhs,
    BinaryOperation operation
)
{
    const Sizes result_sizes = broadcast_shapes(lhs.sizes(), rhs.sizes());
    Tensor<T> result(result_sizes);

    if (result.numel() > 0 && (lhs.numel() == 0 || rhs.numel() == 0)) {
        throw std::invalid_argument("Tensor elementwise operation requires input data");
    }

    for (std::int64_t flat = 0; flat < result.numel(); ++flat) {
        const Sizes result_indices = unravel_index(flat, result.sizes(), result.strides());
        const std::int64_t lhs_offset = compute_broadcast_offset(
            result_indices,
            lhs.sizes(),
            lhs.strides(),
            lhs.storage_offset()
        );
        const std::int64_t rhs_offset = compute_broadcast_offset(
            result_indices,
            rhs.sizes(),
            rhs.strides(),
            rhs.storage_offset()
        );

        result[flat] = operation(
            lhs.data_ptr()[lhs_offset - lhs.storage_offset()],
            rhs.data_ptr()[rhs_offset - rhs.storage_offset()]
        );
    }

    return result;
}

template <typename T, typename Scalar, typename BinaryOperation>
Tensor<T> binary_tensor_scalar_operation(
    const Tensor<T>& tensor,
    Scalar scalar,
    BinaryOperation operation
)
{
    const T value = static_cast<T>(scalar);
    return unary_tensor_operation(tensor, [&](const T& element) {
        return operation(element, value);
    });
}

template <typename Scalar, typename T, typename BinaryOperation>
Tensor<T> binary_scalar_tensor_operation(
    Scalar scalar,
    const Tensor<T>& tensor,
    BinaryOperation operation
)
{
    const T value = static_cast<T>(scalar);
    return unary_tensor_operation(tensor, [&](const T& element) {
        return operation(value, element);
    });
}

} // namespace detail

template <typename T>
Tensor<T> operator+(const Tensor<T>& lhs, const Tensor<T>& rhs)
{
    return detail::binary_tensor_operation(lhs, rhs, [](const T& a, const T& b) {
        return a + b;
    });
}

template <typename T>
Tensor<T> operator-(const Tensor<T>& lhs, const Tensor<T>& rhs)
{
    return detail::binary_tensor_operation(lhs, rhs, [](const T& a, const T& b) {
        return a - b;
    });
}

template <typename T>
Tensor<T> operator*(const Tensor<T>& lhs, const Tensor<T>& rhs)
{
    return detail::binary_tensor_operation(lhs, rhs, [](const T& a, const T& b) {
        return a * b;
    });
}

template <typename T>
Tensor<T> operator/(const Tensor<T>& lhs, const Tensor<T>& rhs)
{
    return detail::binary_tensor_operation(lhs, rhs, [](const T& a, const T& b) {
        return a / b;
    });
}

template <typename T, detail::TensorScalar<T> Scalar>
Tensor<T> operator+(const Tensor<T>& tensor, Scalar scalar)
{
    return detail::binary_tensor_scalar_operation(tensor, scalar, [](const T& a, const T& b) {
        return a + b;
    });
}

template <typename T, detail::TensorScalar<T> Scalar>
Tensor<T> operator+(Scalar scalar, const Tensor<T>& tensor)
{
    return detail::binary_scalar_tensor_operation(scalar, tensor, [](const T& a, const T& b) {
        return a + b;
    });
}

template <typename T, detail::TensorScalar<T> Scalar>
Tensor<T> operator-(const Tensor<T>& tensor, Scalar scalar)
{
    return detail::binary_tensor_scalar_operation(tensor, scalar, [](const T& a, const T& b) {
        return a - b;
    });
}

template <typename T, detail::TensorScalar<T> Scalar>
Tensor<T> operator-(Scalar scalar, const Tensor<T>& tensor)
{
    return detail::binary_scalar_tensor_operation(scalar, tensor, [](const T& a, const T& b) {
        return a - b;
    });
}

template <typename T, detail::TensorScalar<T> Scalar>
Tensor<T> operator*(const Tensor<T>& tensor, Scalar scalar)
{
    return detail::binary_tensor_scalar_operation(tensor, scalar, [](const T& a, const T& b) {
        return a * b;
    });
}

template <typename T, detail::TensorScalar<T> Scalar>
Tensor<T> operator*(Scalar scalar, const Tensor<T>& tensor)
{
    return detail::binary_scalar_tensor_operation(scalar, tensor, [](const T& a, const T& b) {
        return a * b;
    });
}

template <typename T, detail::TensorScalar<T> Scalar>
Tensor<T> operator/(const Tensor<T>& tensor, Scalar scalar)
{
    return detail::binary_tensor_scalar_operation(tensor, scalar, [](const T& a, const T& b) {
        return a / b;
    });
}

template <typename T, detail::TensorScalar<T> Scalar>
Tensor<T> operator/(Scalar scalar, const Tensor<T>& tensor)
{
    return detail::binary_scalar_tensor_operation(scalar, tensor, [](const T& a, const T& b) {
        return a / b;
    });
}

template <typename T>
Tensor<T> operator-(const Tensor<T>& tensor)
{
    return detail::unary_tensor_operation(tensor, [](const T& value) {
        return -value;
    });
}

template <typename T>
Tensor<T> abs(const Tensor<T>& tensor)
{
    return detail::unary_tensor_operation(tensor, [](const T& value) {
        using std::abs;
        return static_cast<T>(abs(value));
    });
}

template <typename T>
Tensor<T> relu(const Tensor<T>& tensor)
{
    return detail::unary_tensor_operation(tensor, [](const T& value) {
        return value < T{} ? T{} : value;
    });
}

template <typename T>
Tensor<T> exp(const Tensor<T>& tensor)
{
    return detail::unary_tensor_operation(tensor, [](const T& value) {
        return static_cast<T>(std::exp(value));
    });
}

template <typename T>
Tensor<T> log(const Tensor<T>& tensor)
{
    return detail::unary_tensor_operation(tensor, [](const T& value) {
        return static_cast<T>(std::log(value));
    });
}

template <typename T>
std::ostream& operator<<(std::ostream& output, const Tensor<T>& tensor)
{
    output << "Tensor(sizes=[";
    for (std::int64_t dim = 0; dim < tensor.dim(); ++dim) {
        if (dim != 0) {
            output << ", ";
        }
        output << tensor.size(dim);
    }

    output << "], data=[";
    for (std::int64_t i = 0; i < tensor.numel(); ++i) {
        if (i != 0) {
            output << ", ";
        }
        output << tensor[i];
    }
    output << "])";
    return output;
}

} // namespace marr
