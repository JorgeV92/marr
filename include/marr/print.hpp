#pragma once

#include <cstdint>
#include <ostream>

#include <marr/tensor_core.hpp>

namespace marr {

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
