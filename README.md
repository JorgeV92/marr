# Marr 

This is a repo that holds different sections for working with deep learning models. 

### C++ tensor

Marr provides a small CPU-only C++20 tensor library through `marr::Tensor<T>`.
Use `#include <marr/tensor.hpp>` as the main public include; it pulls in the
smaller implementation headers under `include/marr/`.

```cpp
#include <iostream>
#include <vector>

#include <marr/tensor.hpp>

marr::Tensor<float> values({2, 2}, std::vector<float>{1, 2, 3, 4});
```

Phase 5 adds eager matrix multiplication:

```cpp
auto a = marr::ones<float>({2, 3});
auto b = marr::full<float>({3, 4}, 2.0f);

auto c = marr::mm(a, b);

std::cout << c.sizes()[0] << " " << c.sizes()[1] << "\n";
```

`marr::mm(a, b)` requires two 2D tensors. `marr::matmul(a, b)` supports a
small set of cases: vector-vector, matrix-vector, vector-matrix, matrix-matrix,
and exact 3D batched matrix multiplication. Vector-vector `matmul` returns a
scalar-like tensor with shape `{}`.

Phase 6 adds lazy elementwise expressions. Matrix multiplication is still eager,
but elementwise expressions can be fused into one evaluation pass with
`marr::eval`:

```cpp
auto x = marr::ones<float>({2, 3});
auto y = marr::full<float>({3}, 2.0f);

auto z = marr::eval(x + y * 3.0f - 1.0f);
```

This also works for direct tensor materialization:

```cpp
marr::Tensor<float> z = x + y * 3.0f - 1.0f;
```

Unevaluated expressions may hold references to tensor operands. Do not return or
store lazy expressions that refer to local tensors after those tensors go out of
scope.

## Phase 7: Autograd

Phase 7 adds a tiny dynamic autograd engine:

```cpp
#include <iostream>

#include <marr/tensor.hpp>

int main() {
    auto x = marr::full<float>({3}, 2.0f);
    x.set_requires_grad(true);

    auto y = x * x + 2.0f * x;
    auto loss = marr::sum(y);

    loss.backward();

    std::cout << x.grad() << "\n";
}
```

The expected gradient is `{6, 6, 6}` because
`d(x^2 + 2x)/dx = 2x + 2`, and `x = 2`.

Autograd builds a dynamic computation graph as tensor operations run.
`backward()` computes gradients from a scalar-like loss, and gradients
accumulate on tensors with `requires_grad == true`. Call `zero_grad()` to clear
stored gradients between optimization steps. Broadcasting gradients are reduced
back to input shapes with `sum_to_shape`.

This is inspired by PyTorch autograd, but it only implements a tiny CPU-only
subset: elementwise add/subtract/multiply/divide, negation, ReLU, `sum`, `mean`,
2D `mm`, `transpose`, `detach`, and `NoGradGuard`. Autograd expressions are
materialized when evaluated for now; matrix multiplication remains eager.

## Phase 8: Parallel CPU Backend

Large tensor loops can run in parallel on the CPU. The public include remains:

```cpp
#include <marr/tensor.hpp>
```

Users can control the simple CPU backend with:

```cpp
#include <marr/tensor.hpp>
#include <iostream>

int main() {
    marr::set_num_threads(4);

    auto a = marr::ones<float>({1000, 1000});
    auto b = marr::full<float>({1000, 1000}, 2.0f);

    auto c = a + b;

    std::cout << c(0, 0) << "\n";
}
```

Parallel execution is enabled by default. The default thread count is
`std::thread::hardware_concurrency()`, or `1` if that value is unavailable. Use
`marr::set_parallel_enabled(false)` to disable parallel execution globally.

For a scoped single-threaded section:

```cpp
{
    marr::NoParallelGuard guard;
    auto y = x + x; // runs single-threaded in this scope
}
```

### Computer vision 

**tinyViT**

This repository is meant to be a more faithful follow-up to a plain ViT implementation. Instead of a single flat sequence model with a `[CLS]` token, it moves closer to the structure used by TinyViT-style models:
