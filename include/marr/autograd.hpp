#pragma once

#include <functional>
#include <memory>
#include <vector>

#include <marr/shape.hpp>

namespace marr {

template <typename T>
struct AutogradNode {
    std::vector<Tensor<T>*> parents;
    std::vector<std::shared_ptr<Tensor<T>>> owned_parents;
    std::function<void(const Tensor<T>& grad_output)> backward_fn;
};

namespace detail {

inline bool& grad_enabled()
{
    static thread_local bool enabled = true;
    return enabled;
}

} // namespace detail

class NoGradGuard {
public:
    NoGradGuard()
        : previous_(detail::grad_enabled())
    {
        detail::grad_enabled() = false;
    }

    ~NoGradGuard()
    {
        detail::grad_enabled() = previous_;
    }

    NoGradGuard(const NoGradGuard&) = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;

private:
    bool previous_;
};

} // namespace marr
