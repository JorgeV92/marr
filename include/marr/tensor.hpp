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

    [[nodiscard]] T value_at_flat_index(std::int64_t index) const
    {
        return (*this)[index];
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

template <typename Derived, typename T>
struct TensorExpr {
    using value_type = T;

    [[nodiscard]] const Derived& derived() const noexcept
    {
        return static_cast<const Derived&>(*this);
    }

    [[nodiscard]] std::int64_t dim() const noexcept
    {
        return static_cast<std::int64_t>(derived().sizes().size());
    }

    [[nodiscard]] std::int64_t ndim() const noexcept
    {
        return dim();
    }

    [[nodiscard]] std::int64_t size() const
    {
        return numel();
    }

    [[nodiscard]] std::int64_t size(std::int64_t dim) const
    {
        return derived().sizes().at(static_cast<std::size_t>(
            detail::normalize_dim(dim, this->dim())
        ));
    }

    [[nodiscard]] const Shape& shape() const noexcept
    {
        return derived().sizes();
    }

    [[nodiscard]] std::int64_t numel() const
    {
        return compute_numel(derived().sizes());
    }

    [[nodiscard]] bool empty() const
    {
        return numel() == 0;
    }

    template <std::integral Index>
    [[nodiscard]] value_type operator[](Index index) const
    {
        return derived().value_at_flat_index(detail::normalize_index(index));
    }

    template <std::integral... Indices>
    [[nodiscard]] value_type operator()(Indices... indices) const
    {
        const auto index_array = detail::make_index_array(indices...);
        return at(std::span<const std::int64_t>(index_array));
    }

    [[nodiscard]] value_type operator()(std::initializer_list<std::int64_t> indices) const
    {
        return at(indices);
    }

    [[nodiscard]] value_type operator()(const Sizes& indices) const
    {
        return at(indices);
    }

    [[nodiscard]] value_type at(std::initializer_list<std::int64_t> indices) const
    {
        return at(std::span<const std::int64_t>(indices.begin(), indices.size()));
    }

    [[nodiscard]] value_type at(const Sizes& indices) const
    {
        return at(std::span<const std::int64_t>(indices));
    }

    [[nodiscard]] value_type at(std::span<const std::int64_t> indices) const
    {
        const Strides strides = compute_contiguous_strides(derived().sizes());
        const std::int64_t flat_index = detail::compute_offset(
            indices,
            derived().sizes(),
            strides,
            0
        );
        return derived().value_at_flat_index(flat_index);
    }
};

template <typename Expr>
auto eval(const Expr& expr) -> Tensor<typename std::remove_cvref_t<Expr>::value_type>;

namespace detail {

template <typename E>
concept TensorExpression = requires(const std::remove_cvref_t<E>& expr, std::int64_t index) {
    typename std::remove_cvref_t<E>::value_type;
    { expr.sizes() } -> std::same_as<const Sizes&>;
    { expr.dim() } -> std::convertible_to<std::int64_t>;
    { expr.numel() } -> std::convertible_to<std::int64_t>;
    { expr.value_at_flat_index(index) } -> std::convertible_to<typename std::remove_cvref_t<E>::value_type>;
};

template <TensorExpression E>
using expr_value_t = typename std::remove_cvref_t<E>::value_type;

template <typename T>
class TensorRef : public TensorExpr<TensorRef<T>, T> {
public:
    using value_type = T;

    explicit TensorRef(const Tensor<T>& tensor) noexcept
        : tensor_(&tensor)
    {
    }

    [[nodiscard]] const Sizes& sizes() const noexcept
    {
        return tensor_->sizes();
    }

    [[nodiscard]] T value_at_flat_index(std::int64_t index) const
    {
        return tensor_->value_at_flat_index(index);
    }

private:
    const Tensor<T>* tensor_;
};

template <typename T>
TensorRef<T> as_operand(const Tensor<T>& tensor)
{
    return TensorRef<T>(tensor);
}

template <TensorExpression E>
requires (!is_tensor_v<std::remove_cvref_t<E>>)
std::remove_cvref_t<E> as_operand(const E& expression)
{
    return expression;
}

struct AddOperation {
    template <typename T>
    T operator()(const T& lhs, const T& rhs) const
    {
        return lhs + rhs;
    }
};

struct SubtractOperation {
    template <typename T>
    T operator()(const T& lhs, const T& rhs) const
    {
        return lhs - rhs;
    }
};

struct MultiplyOperation {
    template <typename T>
    T operator()(const T& lhs, const T& rhs) const
    {
        return lhs * rhs;
    }
};

struct DivideOperation {
    template <typename T>
    T operator()(const T& lhs, const T& rhs) const
    {
        return lhs / rhs;
    }
};

struct NegateOperation {
    template <typename T>
    T operator()(const T& value) const
    {
        return -value;
    }
};

struct AbsOperation {
    template <typename T>
    T operator()(const T& value) const
    {
        using std::abs;
        return static_cast<T>(abs(value));
    }
};

struct ReluOperation {
    template <typename T>
    T operator()(const T& value) const
    {
        return value < T{} ? T{} : value;
    }
};

struct ExpOperation {
    template <typename T>
    T operator()(const T& value) const
    {
        return static_cast<T>(std::exp(value));
    }
};

struct LogOperation {
    template <typename T>
    T operator()(const T& value) const
    {
        return static_cast<T>(std::log(value));
    }
};

template <TensorExpression E>
expr_value_t<E> value_at_broadcast_indices(
    const E& expression,
    std::span<const std::int64_t> output_indices
)
{
    const Strides expression_strides = compute_contiguous_strides(expression.sizes());
    const std::int64_t flat_index = compute_broadcast_offset(
        output_indices,
        expression.sizes(),
        expression_strides,
        0
    );
    return expression.value_at_flat_index(flat_index);
}

} // namespace detail

template <typename T>
class ScalarExpr : public TensorExpr<ScalarExpr<T>, T> {
public:
    using value_type = T;

    explicit ScalarExpr(T value)
        : value_(std::move(value))
    {
    }

    [[nodiscard]] const Sizes& sizes() const noexcept
    {
        return sizes_;
    }

    [[nodiscard]] T value_at_flat_index(std::int64_t index) const
    {
        if (index != 0) {
            throw std::out_of_range("Scalar expression flat index out of range");
        }
        return value_;
    }

    [[nodiscard]] operator Tensor<T>() const
    {
        return eval(*this);
    }

private:
    T value_;
    Sizes sizes_;
};

template <typename Op, typename L, typename R>
class BinaryExpr
    : public TensorExpr<
          BinaryExpr<Op, L, R>,
          std::common_type_t<typename L::value_type, typename R::value_type>> {
public:
    using value_type = std::common_type_t<typename L::value_type, typename R::value_type>;

    BinaryExpr(L lhs, R rhs)
        : lhs_(std::move(lhs)),
          rhs_(std::move(rhs)),
          sizes_(broadcast_shapes(lhs_.sizes(), rhs_.sizes())),
          strides_(compute_contiguous_strides(sizes_))
    {
        if (compute_numel(sizes_) > 0 && (lhs_.numel() == 0 || rhs_.numel() == 0)) {
            throw std::invalid_argument("Tensor elementwise operation requires input data");
        }
    }

    [[nodiscard]] const Sizes& sizes() const noexcept
    {
        return sizes_;
    }

    [[nodiscard]] value_type value_at_flat_index(std::int64_t index) const
    {
        if (index < 0 || index >= compute_numel(sizes_)) {
            throw std::out_of_range("Expression flat index out of range");
        }

        const Sizes output_indices = detail::unravel_index(index, sizes_, strides_);
        return operation_(
            static_cast<value_type>(detail::value_at_broadcast_indices(lhs_, output_indices)),
            static_cast<value_type>(detail::value_at_broadcast_indices(rhs_, output_indices))
        );
    }

    [[nodiscard]] operator Tensor<value_type>() const
    {
        return eval(*this);
    }

private:
    L lhs_;
    R rhs_;
    Sizes sizes_;
    Strides strides_;
    Op operation_;
};

template <typename Op, typename E>
class UnaryExpr : public TensorExpr<UnaryExpr<Op, E>, typename E::value_type> {
public:
    using value_type = typename E::value_type;

    explicit UnaryExpr(E expression)
        : expression_(std::move(expression)),
          sizes_(expression_.sizes())
    {
    }

    [[nodiscard]] const Sizes& sizes() const noexcept
    {
        return sizes_;
    }

    [[nodiscard]] value_type value_at_flat_index(std::int64_t index) const
    {
        if (index < 0 || index >= compute_numel(sizes_)) {
            throw std::out_of_range("Expression flat index out of range");
        }
        return operation_(expression_.value_at_flat_index(index));
    }

    [[nodiscard]] operator Tensor<value_type>() const
    {
        return eval(*this);
    }

private:
    E expression_;
    Sizes sizes_;
    Op operation_;
};

template <typename Expr>
auto eval(const Expr& expr) -> Tensor<typename std::remove_cvref_t<Expr>::value_type>
{
    static_assert(detail::TensorExpression<Expr>, "marr::eval expects a tensor expression");

    using T = typename std::remove_cvref_t<Expr>::value_type;
    Tensor<T> result(expr.sizes());
    for (std::int64_t flat = 0; flat < result.numel(); ++flat) {
        result[flat] = static_cast<T>(expr.value_at_flat_index(flat));
    }
    return result;
}

template <detail::TensorExpression L, detail::TensorExpression R>
requires std::same_as<detail::expr_value_t<L>, detail::expr_value_t<R>>
auto operator+(const L& lhs, const R& rhs)
{
    auto lhs_expr = detail::as_operand(lhs);
    auto rhs_expr = detail::as_operand(rhs);
    return BinaryExpr<detail::AddOperation, decltype(lhs_expr), decltype(rhs_expr)>(
        std::move(lhs_expr),
        std::move(rhs_expr)
    );
}

template <detail::TensorExpression E, typename Scalar>
requires detail::TensorScalar<Scalar, detail::expr_value_t<E>>
auto operator+(const E& expression, Scalar scalar)
{
    using T = detail::expr_value_t<E>;
    auto lhs_expr = detail::as_operand(expression);
    auto rhs_expr = ScalarExpr<T>(static_cast<T>(scalar));
    return BinaryExpr<detail::AddOperation, decltype(lhs_expr), decltype(rhs_expr)>(
        std::move(lhs_expr),
        std::move(rhs_expr)
    );
}

template <typename Scalar, detail::TensorExpression E>
requires detail::TensorScalar<Scalar, detail::expr_value_t<E>>
auto operator+(Scalar scalar, const E& expression)
{
    using T = detail::expr_value_t<E>;
    auto lhs_expr = ScalarExpr<T>(static_cast<T>(scalar));
    auto rhs_expr = detail::as_operand(expression);
    return BinaryExpr<detail::AddOperation, decltype(lhs_expr), decltype(rhs_expr)>(
        std::move(lhs_expr),
        std::move(rhs_expr)
    );
}

template <detail::TensorExpression L, detail::TensorExpression R>
requires std::same_as<detail::expr_value_t<L>, detail::expr_value_t<R>>
auto operator-(const L& lhs, const R& rhs)
{
    auto lhs_expr = detail::as_operand(lhs);
    auto rhs_expr = detail::as_operand(rhs);
    return BinaryExpr<detail::SubtractOperation, decltype(lhs_expr), decltype(rhs_expr)>(
        std::move(lhs_expr),
        std::move(rhs_expr)
    );
}

template <detail::TensorExpression E, typename Scalar>
requires detail::TensorScalar<Scalar, detail::expr_value_t<E>>
auto operator-(const E& expression, Scalar scalar)
{
    using T = detail::expr_value_t<E>;
    auto lhs_expr = detail::as_operand(expression);
    auto rhs_expr = ScalarExpr<T>(static_cast<T>(scalar));
    return BinaryExpr<detail::SubtractOperation, decltype(lhs_expr), decltype(rhs_expr)>(
        std::move(lhs_expr),
        std::move(rhs_expr)
    );
}

template <typename Scalar, detail::TensorExpression E>
requires detail::TensorScalar<Scalar, detail::expr_value_t<E>>
auto operator-(Scalar scalar, const E& expression)
{
    using T = detail::expr_value_t<E>;
    auto lhs_expr = ScalarExpr<T>(static_cast<T>(scalar));
    auto rhs_expr = detail::as_operand(expression);
    return BinaryExpr<detail::SubtractOperation, decltype(lhs_expr), decltype(rhs_expr)>(
        std::move(lhs_expr),
        std::move(rhs_expr)
    );
}

template <detail::TensorExpression L, detail::TensorExpression R>
requires std::same_as<detail::expr_value_t<L>, detail::expr_value_t<R>>
auto operator*(const L& lhs, const R& rhs)
{
    auto lhs_expr = detail::as_operand(lhs);
    auto rhs_expr = detail::as_operand(rhs);
    return BinaryExpr<detail::MultiplyOperation, decltype(lhs_expr), decltype(rhs_expr)>(
        std::move(lhs_expr),
        std::move(rhs_expr)
    );
}

template <detail::TensorExpression E, typename Scalar>
requires detail::TensorScalar<Scalar, detail::expr_value_t<E>>
auto operator*(const E& expression, Scalar scalar)
{
    using T = detail::expr_value_t<E>;
    auto lhs_expr = detail::as_operand(expression);
    auto rhs_expr = ScalarExpr<T>(static_cast<T>(scalar));
    return BinaryExpr<detail::MultiplyOperation, decltype(lhs_expr), decltype(rhs_expr)>(
        std::move(lhs_expr),
        std::move(rhs_expr)
    );
}

template <typename Scalar, detail::TensorExpression E>
requires detail::TensorScalar<Scalar, detail::expr_value_t<E>>
auto operator*(Scalar scalar, const E& expression)
{
    using T = detail::expr_value_t<E>;
    auto lhs_expr = ScalarExpr<T>(static_cast<T>(scalar));
    auto rhs_expr = detail::as_operand(expression);
    return BinaryExpr<detail::MultiplyOperation, decltype(lhs_expr), decltype(rhs_expr)>(
        std::move(lhs_expr),
        std::move(rhs_expr)
    );
}

template <detail::TensorExpression L, detail::TensorExpression R>
requires std::same_as<detail::expr_value_t<L>, detail::expr_value_t<R>>
auto operator/(const L& lhs, const R& rhs)
{
    auto lhs_expr = detail::as_operand(lhs);
    auto rhs_expr = detail::as_operand(rhs);
    return BinaryExpr<detail::DivideOperation, decltype(lhs_expr), decltype(rhs_expr)>(
        std::move(lhs_expr),
        std::move(rhs_expr)
    );
}

template <detail::TensorExpression E, typename Scalar>
requires detail::TensorScalar<Scalar, detail::expr_value_t<E>>
auto operator/(const E& expression, Scalar scalar)
{
    using T = detail::expr_value_t<E>;
    auto lhs_expr = detail::as_operand(expression);
    auto rhs_expr = ScalarExpr<T>(static_cast<T>(scalar));
    return BinaryExpr<detail::DivideOperation, decltype(lhs_expr), decltype(rhs_expr)>(
        std::move(lhs_expr),
        std::move(rhs_expr)
    );
}

template <typename Scalar, detail::TensorExpression E>
requires detail::TensorScalar<Scalar, detail::expr_value_t<E>>
auto operator/(Scalar scalar, const E& expression)
{
    using T = detail::expr_value_t<E>;
    auto lhs_expr = ScalarExpr<T>(static_cast<T>(scalar));
    auto rhs_expr = detail::as_operand(expression);
    return BinaryExpr<detail::DivideOperation, decltype(lhs_expr), decltype(rhs_expr)>(
        std::move(lhs_expr),
        std::move(rhs_expr)
    );
}

template <detail::TensorExpression E>
auto operator-(const E& expression)
{
    auto operand = detail::as_operand(expression);
    return UnaryExpr<detail::NegateOperation, decltype(operand)>(std::move(operand));
}

template <detail::TensorExpression E>
auto abs(const E& expression)
{
    auto operand = detail::as_operand(expression);
    return UnaryExpr<detail::AbsOperation, decltype(operand)>(std::move(operand));
}

template <detail::TensorExpression E>
auto relu(const E& expression)
{
    auto operand = detail::as_operand(expression);
    return UnaryExpr<detail::ReluOperation, decltype(operand)>(std::move(operand));
}

template <detail::TensorExpression E>
auto exp(const E& expression)
{
    auto operand = detail::as_operand(expression);
    return UnaryExpr<detail::ExpOperation, decltype(operand)>(std::move(operand));
}

template <detail::TensorExpression E>
auto log(const E& expression)
{
    auto operand = detail::as_operand(expression);
    return UnaryExpr<detail::LogOperation, decltype(operand)>(std::move(operand));
}

template <typename T>
Tensor<T> mm(const Tensor<T>& lhs, const Tensor<T>& rhs)
{
    if (lhs.dim() != 2 || rhs.dim() != 2) {
        throw std::invalid_argument("marr::mm expects both inputs to be 2D tensors");
    }
    if (lhs.size(1) != rhs.size(0)) {
        throw std::invalid_argument(
            "marr::mm inner dimensions must match: lhs size(1) is " +
            std::to_string(lhs.size(1)) + " but rhs size(0) is " + std::to_string(rhs.size(0))
        );
    }

    const std::int64_t rows = lhs.size(0);
    const std::int64_t inner = lhs.size(1);
    const std::int64_t columns = rhs.size(1);
    Tensor<T> result({rows, columns}, T{});

    for (std::int64_t row = 0; row < rows; ++row) {
        for (std::int64_t column = 0; column < columns; ++column) {
            T sum{};
            for (std::int64_t index = 0; index < inner; ++index) {
                sum += lhs(row, index) * rhs(index, column);
            }
            result(row, column) = sum;
        }
    }

    return result;
}

template <typename T>
Tensor<T> matmul(const Tensor<T>& lhs, const Tensor<T>& rhs)
{
    if (lhs.dim() == 1 && rhs.dim() == 1) {
        if (lhs.size(0) != rhs.size(0)) {
            throw std::invalid_argument(
                "marr::matmul vector dot product sizes must match: lhs size(0) is " +
                std::to_string(lhs.size(0)) + " but rhs size(0) is " + std::to_string(rhs.size(0))
            );
        }

        T sum{};
        for (std::int64_t index = 0; index < lhs.size(0); ++index) {
            sum += lhs(index) * rhs(index);
        }
        return Tensor<T>({}, sum);
    }

    if (lhs.dim() == 2 && rhs.dim() == 1) {
        if (lhs.size(1) != rhs.size(0)) {
            throw std::invalid_argument(
                "marr::matmul matrix-vector inner dimensions must match: lhs size(1) is " +
                std::to_string(lhs.size(1)) + " but rhs size(0) is " + std::to_string(rhs.size(0))
            );
        }

        Tensor<T> result({lhs.size(0)}, T{});
        for (std::int64_t row = 0; row < lhs.size(0); ++row) {
            T sum{};
            for (std::int64_t index = 0; index < lhs.size(1); ++index) {
                sum += lhs(row, index) * rhs(index);
            }
            result(row) = sum;
        }
        return result;
    }

    if (lhs.dim() == 1 && rhs.dim() == 2) {
        if (lhs.size(0) != rhs.size(0)) {
            throw std::invalid_argument(
                "marr::matmul vector-matrix inner dimensions must match: lhs size(0) is " +
                std::to_string(lhs.size(0)) + " but rhs size(0) is " + std::to_string(rhs.size(0))
            );
        }

        Tensor<T> result({rhs.size(1)}, T{});
        for (std::int64_t column = 0; column < rhs.size(1); ++column) {
            T sum{};
            for (std::int64_t index = 0; index < lhs.size(0); ++index) {
                sum += lhs(index) * rhs(index, column);
            }
            result(column) = sum;
        }
        return result;
    }

    if (lhs.dim() == 2 && rhs.dim() == 2) {
        return mm(lhs, rhs);
    }

    if (lhs.dim() == 3 && rhs.dim() == 3) {
        if (lhs.size(0) != rhs.size(0)) {
            throw std::invalid_argument(
                "marr::matmul batch dimensions must match: lhs size(0) is " +
                std::to_string(lhs.size(0)) + " but rhs size(0) is " + std::to_string(rhs.size(0))
            );
        }
        if (lhs.size(2) != rhs.size(1)) {
            throw std::invalid_argument(
                "marr::matmul batched matrix inner dimensions must match: lhs size(2) is " +
                std::to_string(lhs.size(2)) + " but rhs size(1) is " + std::to_string(rhs.size(1))
            );
        }

        const std::int64_t batch = lhs.size(0);
        const std::int64_t rows = lhs.size(1);
        const std::int64_t inner = lhs.size(2);
        const std::int64_t columns = rhs.size(2);
        Tensor<T> result({batch, rows, columns}, T{});

        for (std::int64_t batch_index = 0; batch_index < batch; ++batch_index) {
            for (std::int64_t row = 0; row < rows; ++row) {
                for (std::int64_t column = 0; column < columns; ++column) {
                    T sum{};
                    for (std::int64_t index = 0; index < inner; ++index) {
                        sum += lhs(batch_index, row, index) * rhs(batch_index, index, column);
                    }
                    result(batch_index, row, column) = sum;
                }
            }
        }

        return result;
    }

    throw std::invalid_argument(
        "marr::matmul supports only 1D/2D combinations and exact 3D batched matrix multiplication"
    );
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
