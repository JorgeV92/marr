#pragma once

#include <cmath>
#include <concepts>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <marr/autograd.hpp>
#include <marr/broadcast.hpp>
#include <marr/indexing.hpp>
#include <marr/tensor_core.hpp>

namespace marr {

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

    [[nodiscard]] const Tensor<T>& tensor() const noexcept
    {
        return *tensor_;
    }

    [[nodiscard]] Tensor<T>& mutable_tensor() const noexcept
    {
        return *const_cast<Tensor<T>*>(tensor_);
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
std::int64_t broadcast_flat_index(
    const E& expression,
    std::span<const std::int64_t> output_indices
)
{
    const Strides expression_strides = compute_contiguous_strides(expression.sizes());
    return compute_broadcast_offset(
        output_indices,
        expression.sizes(),
        expression_strides,
        0
    );
}

template <TensorExpression E>
expr_value_t<E> value_at_broadcast_indices(
    const E& expression,
    std::span<const std::int64_t> output_indices
)
{
    const std::int64_t flat_index = broadcast_flat_index(expression, output_indices);
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

    [[nodiscard]] const L& lhs() const noexcept
    {
        return lhs_;
    }

    [[nodiscard]] const R& rhs() const noexcept
    {
        return rhs_;
    }

    [[nodiscard]] const Strides& expression_strides() const noexcept
    {
        return strides_;
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

    [[nodiscard]] const E& expression() const noexcept
    {
        return expression_;
    }

private:
    E expression_;
    Sizes sizes_;
    Op operation_;
};

namespace detail {

template <typename T>
bool expression_requires_grad(const TensorRef<T>& expression)
{
    return expression.tensor().requires_grad();
}

template <typename T>
bool expression_requires_grad(const ScalarExpr<T>&)
{
    return false;
}

template <typename Op, typename L, typename R>
bool expression_requires_grad(const BinaryExpr<Op, L, R>& expression)
{
    return expression_requires_grad(expression.lhs()) ||
           expression_requires_grad(expression.rhs());
}

template <typename Op, typename E>
bool expression_requires_grad(const UnaryExpr<Op, E>& expression)
{
    return expression_requires_grad(expression.expression());
}

template <typename T>
void collect_expression_parents(const TensorRef<T>& expression, std::vector<Tensor<T>*>& parents)
{
    Tensor<T>& tensor = expression.mutable_tensor();
    if (tensor.requires_grad()) {
        parents.push_back(&tensor);
    }
}

template <typename T>
void collect_expression_parents(const ScalarExpr<T>&, std::vector<Tensor<T>*>&)
{
}

template <typename Op, typename L, typename R>
void collect_expression_parents(
    const BinaryExpr<Op, L, R>& expression,
    std::vector<Tensor<typename BinaryExpr<Op, L, R>::value_type>*>& parents
)
{
    collect_expression_parents(expression.lhs(), parents);
    collect_expression_parents(expression.rhs(), parents);
}

template <typename Op, typename E>
void collect_expression_parents(
    const UnaryExpr<Op, E>& expression,
    std::vector<Tensor<typename UnaryExpr<Op, E>::value_type>*>& parents
)
{
    collect_expression_parents(expression.expression(), parents);
}

template <typename T>
void backward_expression(const TensorRef<T>& expression, const Tensor<T>& grad_output)
{
    expression.mutable_tensor().accumulate_grad(grad_output);
}

template <typename T>
void backward_expression(const ScalarExpr<T>&, const Tensor<T>&)
{
}

template <typename Op, typename E>
void backward_expression(
    const UnaryExpr<Op, E>& expression,
    const Tensor<typename UnaryExpr<Op, E>::value_type>& grad_output
)
{
    using T = typename UnaryExpr<Op, E>::value_type;
    Tensor<T> input_grad(expression.sizes(), T{});

    for (std::int64_t flat = 0; flat < grad_output.numel(); ++flat) {
        const T upstream = grad_output[flat];
        const T input_value = expression.expression().value_at_flat_index(flat);

        if constexpr (std::same_as<Op, NegateOperation>) {
            input_grad[flat] += -upstream;
        } else if constexpr (std::same_as<Op, ReluOperation>) {
            input_grad[flat] += input_value > T{} ? upstream : T{};
        } else if constexpr (std::same_as<Op, AbsOperation>) {
            if (input_value > T{}) {
                input_grad[flat] += upstream;
            } else if (input_value < T{}) {
                input_grad[flat] += -upstream;
            }
        } else if constexpr (std::same_as<Op, ExpOperation>) {
            input_grad[flat] += upstream * static_cast<T>(std::exp(input_value));
        } else if constexpr (std::same_as<Op, LogOperation>) {
            input_grad[flat] += upstream / input_value;
        }
    }

    backward_expression(expression.expression(), input_grad);
}

template <typename Op, typename L, typename R>
void backward_expression(
    const BinaryExpr<Op, L, R>& expression,
    const Tensor<typename BinaryExpr<Op, L, R>::value_type>& grad_output
)
{
    using T = typename BinaryExpr<Op, L, R>::value_type;

    Tensor<T> lhs_grad(expression.lhs().sizes(), T{});
    Tensor<T> rhs_grad(expression.rhs().sizes(), T{});

    for (std::int64_t flat = 0; flat < grad_output.numel(); ++flat) {
        const Sizes output_indices = unravel_index(
            flat,
            expression.sizes(),
            expression.expression_strides()
        );
        const std::int64_t lhs_flat = broadcast_flat_index(expression.lhs(), output_indices);
        const std::int64_t rhs_flat = broadcast_flat_index(expression.rhs(), output_indices);
        const T lhs_value = static_cast<T>(expression.lhs().value_at_flat_index(lhs_flat));
        const T rhs_value = static_cast<T>(expression.rhs().value_at_flat_index(rhs_flat));
        const T upstream = grad_output[flat];

        if constexpr (std::same_as<Op, AddOperation>) {
            lhs_grad[lhs_flat] += upstream;
            rhs_grad[rhs_flat] += upstream;
        } else if constexpr (std::same_as<Op, SubtractOperation>) {
            lhs_grad[lhs_flat] += upstream;
            rhs_grad[rhs_flat] += -upstream;
        } else if constexpr (std::same_as<Op, MultiplyOperation>) {
            lhs_grad[lhs_flat] += upstream * rhs_value;
            rhs_grad[rhs_flat] += upstream * lhs_value;
        } else if constexpr (std::same_as<Op, DivideOperation>) {
            lhs_grad[lhs_flat] += upstream / rhs_value;
            rhs_grad[rhs_flat] += upstream * (-lhs_value / (rhs_value * rhs_value));
        }
    }

    backward_expression(expression.lhs(), lhs_grad);
    backward_expression(expression.rhs(), rhs_grad);
}

} // namespace detail

template <typename Expr>
auto eval(const Expr& expr) -> Tensor<typename std::remove_cvref_t<Expr>::value_type>
{
    static_assert(detail::TensorExpression<Expr>, "marr::eval expects a tensor expression");

    using T = typename std::remove_cvref_t<Expr>::value_type;
    auto expression = detail::as_operand(expr);
    Tensor<T> result(expression.sizes());
    for (std::int64_t flat = 0; flat < result.numel(); ++flat) {
        result[flat] = static_cast<T>(expression.value_at_flat_index(flat));
    }

    if (detail::grad_enabled() && detail::expression_requires_grad(expression)) {
        auto node = std::make_shared<AutogradNode<T>>();
        detail::collect_expression_parents(expression, node->parents);
        node->backward_fn = [expression](const Tensor<T>& grad_output) {
            detail::backward_expression(expression, grad_output);
        };
        result.set_grad_fn(std::move(node));
    }
    return result;
}

} // namespace marr
