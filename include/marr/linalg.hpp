#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <marr/autograd.hpp>
#include <marr/detail/parallel_for.hpp>
#include <marr/tensor_core.hpp>

namespace marr {

template <typename T>
Tensor<T> transpose(const Tensor<T>& input)
{
    if (input.dim() != 2) {
        throw std::invalid_argument("marr::transpose expects a 2D tensor");
    }

    Tensor<T> result({input.size(1), input.size(0)}, T{});
    for (std::int64_t row = 0; row < input.size(0); ++row) {
        for (std::int64_t column = 0; column < input.size(1); ++column) {
            result(column, row) = input(row, column);
        }
    }

    if (detail::grad_enabled() && input.requires_grad()) {
        auto node = std::make_shared<AutogradNode<T>>();
        Tensor<T>* parent = const_cast<Tensor<T>*>(&input);
        node->parents.push_back(parent);
        node->backward_fn = [parent](const Tensor<T>& grad_output) {
            NoGradGuard guard;
            parent->accumulate_grad(transpose(grad_output));
        };
        result.set_grad_fn(std::move(node));
    }

    return result;
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

    // Use one task per output element; each index writes a distinct result cell.
    detail::parallel_for(0, rows * columns, [&](std::int64_t flat) {
        const std::int64_t row = flat / columns;
        const std::int64_t column = flat % columns;

        T sum{};
        for (std::int64_t index = 0; index < inner; ++index) {
            sum += lhs(row, index) * rhs(index, column);
        }
        result(row, column) = sum;
    });

    if (detail::grad_enabled() && (lhs.requires_grad() || rhs.requires_grad())) {
        auto node = std::make_shared<AutogradNode<T>>();
        Tensor<T>* lhs_parent = const_cast<Tensor<T>*>(&lhs);
        Tensor<T>* rhs_parent = const_cast<Tensor<T>*>(&rhs);
        if (lhs.requires_grad()) {
            node->parents.push_back(lhs_parent);
        }
        if (rhs.requires_grad()) {
            node->parents.push_back(rhs_parent);
        }
        node->backward_fn = [lhs_parent, rhs_parent](const Tensor<T>& grad_output) {
            NoGradGuard guard;
            if (lhs_parent->requires_grad()) {
                lhs_parent->accumulate_grad(mm(grad_output, transpose(*rhs_parent)));
            }
            if (rhs_parent->requires_grad()) {
                rhs_parent->accumulate_grad(mm(transpose(*lhs_parent), grad_output));
            }
        };
        result.set_grad_fn(std::move(node));
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

} // namespace marr
