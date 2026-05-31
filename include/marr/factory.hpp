#pragma once

#include <utility>

#include <marr/tensor_core.hpp>

namespace marr {

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

} // namespace marr
