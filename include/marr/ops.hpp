#pragma once

#include <concepts>
#include <cstdint>
#include <algorithm>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include <marr/detail/parallel_for.hpp>
#include <marr/expr.hpp>

namespace marr {

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

namespace detail {

template <typename T>
Tensor<T> sum_tensor_impl(
    const Tensor<T>& input,
    T scale,
    std::shared_ptr<Tensor<T>> owned_parent = nullptr
)
{
    if (input.numel() == 0) {
        throw std::invalid_argument("marr::sum and marr::mean require at least one element");
    }

    T total{};
    if (should_parallelize(input.numel())) {
        const std::int64_t workers = parallel_worker_count(input.numel());
        const std::int64_t chunk_size = (input.numel() + workers - 1) / workers;
        std::vector<T> partials(static_cast<std::size_t>(workers), T{});

        parallel_for(0, input.numel(), [&](std::int64_t flat) {
            const std::int64_t worker = std::min<std::int64_t>(flat / chunk_size, workers - 1);
            partials[static_cast<std::size_t>(worker)] += input[flat];
        });

        for (const T& partial : partials) {
            total += partial;
        }
    } else {
        for (std::int64_t flat = 0; flat < input.numel(); ++flat) {
            total += input[flat];
        }
    }

    Tensor<T> result({}, total * scale);
    if (grad_enabled() && input.requires_grad()) {
        auto node = std::make_shared<AutogradNode<T>>();
        Tensor<T>* parent = owned_parent ? owned_parent.get() : const_cast<Tensor<T>*>(&input);
        node->parents.push_back(parent);
        if (owned_parent) {
            node->owned_parents.push_back(std::move(owned_parent));
        }
        node->backward_fn = [parent, scale](const Tensor<T>& grad_output) {
            Tensor<T> parent_grad(parent->sizes(), grad_output[0] * scale);
            parent->accumulate_grad(parent_grad);
        };
        result.set_grad_fn(std::move(node));
    }

    return result;
}

} // namespace detail

template <typename T>
Tensor<T> sum(const Tensor<T>& input)
{
    return detail::sum_tensor_impl(input, static_cast<T>(1));
}

template <detail::TensorExpression E>
requires (!detail::is_tensor_v<std::remove_cvref_t<E>>)
auto sum(const E& expression) -> Tensor<detail::expr_value_t<E>>
{
    using T = detail::expr_value_t<E>;
    auto materialized = std::make_shared<Tensor<T>>(eval(expression));
    return detail::sum_tensor_impl(*materialized, static_cast<T>(1), materialized);
}

template <typename T>
Tensor<T> mean(const Tensor<T>& input)
{
    if (input.numel() == 0) {
        throw std::invalid_argument("marr::mean requires at least one element");
    }
    return detail::sum_tensor_impl(input, static_cast<T>(1) / static_cast<T>(input.numel()));
}

template <detail::TensorExpression E>
requires (!detail::is_tensor_v<std::remove_cvref_t<E>>)
auto mean(const E& expression) -> Tensor<detail::expr_value_t<E>>
{
    using T = detail::expr_value_t<E>;
    auto materialized = std::make_shared<Tensor<T>>(eval(expression));
    if (materialized->numel() == 0) {
        throw std::invalid_argument("marr::mean requires at least one element");
    }
    const T scale = static_cast<T>(1) / static_cast<T>(materialized->numel());
    return detail::sum_tensor_impl(*materialized, scale, materialized);
}

} // namespace marr
