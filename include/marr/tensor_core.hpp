#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <marr/autograd.hpp>
#include <marr/indexing.hpp>
#include <marr/shape.hpp>

namespace marr {

template <typename T>
class Tensor {
public:
    using value_type = T;
    using size_type = std::int64_t;

    Tensor() = default;

    static Tensor empty(Sizes sizes)
    {
        return Tensor(std::move(sizes));
    }

    static Tensor zeros(Sizes sizes)
    {
        return Tensor(std::move(sizes), T{});
    }

    static Tensor ones(Sizes sizes)
    {
        return Tensor(std::move(sizes), static_cast<T>(1));
    }

    static Tensor full(Sizes sizes, const T& value)
    {
        return Tensor(std::move(sizes), value);
    }

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

    [[nodiscard]] bool requires_grad() const noexcept
    {
        return requires_grad_;
    }

    void set_requires_grad(bool value)
    {
        requires_grad_ = value;
        if (!requires_grad_) {
            grad_.reset();
            grad_fn_.reset();
        }
    }

    [[nodiscard]] bool has_grad() const noexcept
    {
        return grad_ != nullptr;
    }

    [[nodiscard]] const Tensor& grad() const
    {
        if (!grad_) {
            throw std::runtime_error("Tensor has no gradient");
        }
        return *grad_;
    }

    Tensor& mutable_grad()
    {
        if (!requires_grad_) {
            throw std::runtime_error("Cannot create a gradient for a tensor that does not require grad");
        }
        if (!grad_) {
            grad_ = std::make_shared<Tensor>(sizes_, T{});
        }
        return *grad_;
    }

    void zero_grad()
    {
        grad_.reset();
    }

    void backward()
    {
        if (numel() != 1) {
            throw std::invalid_argument(
                "Tensor::backward currently requires a scalar-like tensor"
            );
        }
        if (!requires_grad_) {
            throw std::invalid_argument("Tensor::backward called on a tensor that does not require grad");
        }

        std::vector<Tensor*> topo;
        std::unordered_set<Tensor*> visited;
        build_topological_order(topo, visited);

        for (Tensor* tensor : topo) {
            if (tensor->grad_fn_) {
                tensor->grad_.reset();
            }
        }
        accumulate_grad(Tensor(sizes_, static_cast<T>(1)));

        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            Tensor* tensor = *it;
            if (tensor->grad_fn_ && tensor->grad_) {
                tensor->grad_fn_->backward_fn(*tensor->grad_);
            }
        }
    }

    [[nodiscard]] Tensor detach() const
    {
        return Tensor(sizes_, data_);
    }

    void set_grad_fn(std::shared_ptr<AutogradNode<T>> grad_fn)
    {
        grad_fn_ = std::move(grad_fn);
        requires_grad_ = grad_fn_ != nullptr;
    }

    [[nodiscard]] const std::shared_ptr<AutogradNode<T>>& grad_fn() const noexcept
    {
        return grad_fn_;
    }

    void accumulate_grad(const Tensor& incoming_grad)
    {
        if (!requires_grad_) {
            return;
        }
        if (incoming_grad.sizes() != sizes_) {
            throw std::invalid_argument("Gradient shape does not match tensor shape");
        }

        Tensor& current_grad = mutable_grad();
        // Gradient accumulation stays single-threaded to avoid racing writes when
        // multiple autograd paths contribute to the same tensor.
        for (std::int64_t i = 0; i < current_grad.numel(); ++i) {
            current_grad[i] += incoming_grad[i];
        }
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

    void build_topological_order(
        std::vector<Tensor*>& topo,
        std::unordered_set<Tensor*>& visited
    )
    {
        if (!visited.insert(this).second) {
            return;
        }

        if (grad_fn_) {
            for (Tensor* parent : grad_fn_->parents) {
                if (parent != nullptr) {
                    parent->build_topological_order(topo, visited);
                }
            }
            for (const auto& parent : grad_fn_->owned_parents) {
                if (parent) {
                    parent->build_topological_order(topo, visited);
                }
            }
        }

        topo.push_back(this);
    }

    Sizes sizes_;
    Strides strides_;
    std::int64_t storage_offset_ = 0;
    std::vector<T> data_;
    bool requires_grad_ = false;
    std::shared_ptr<Tensor> grad_;
    std::shared_ptr<AutogradNode<T>> grad_fn_;
};

} // namespace marr
