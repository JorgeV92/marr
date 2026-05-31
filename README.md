# Marr 

This is a repo that holds different sections for working with deep learning models. 

### C++ tensor

Marr provides a small CPU-only C++20 tensor library through `marr::Tensor<T>`.

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

### Computer vision 

**tinyViT**

This repository is meant to be a more faithful follow-up to a plain ViT implementation. Instead of a single flat sequence model with a `[CLS]` token, it moves closer to the structure used by TinyViT-style models:
