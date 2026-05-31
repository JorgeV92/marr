# Marr 

This is a repo that holds different sections for working with deep learning models. 

### C++ tensor

Marr now provides a small C++ tensor library through `marr::Tensor<T>`:

```cpp
#include <marr/tensor.hpp>

marr::Tensor<float> values({2, 2}, std::vector<float>{1, 2, 3, 4});
```

### Computer vision 

**tinyViT**

This repository is meant to be a more faithful follow-up to a plain ViT implementation. Instead of a single flat sequence model with a `[CLS]` token, it moves closer to the structure used by TinyViT-style models:
