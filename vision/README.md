# marrvision

This package is the start of marrvision's C++/LibTorch vision APIs plus a small
Python dataset layer.

The public namespace is:

```cpp
#include <marrvision/transforms/v2.hpp>
#include <marrvision/datasets.hpp>

namespace v2 = marr::vision::transforms::v2;
namespace datasets = marr::vision::datasets;
```

The first pass is intentionally tensor-first:

- Supported image tensors are `[..., C, H, W]`.
- Common classification transforms are implemented: `Compose`, `Resize`,
  `CenterCrop`, `RandomCrop`, `RandomResizedCrop`, random flips, `Pad`,
  `ToDtype`, `Normalize`, `ColorJitter`, grayscale/RGB conversion,
  `GaussianNoise`, `RandomErasing`, and temporal subsampling.
- The torchvision v2 public class names are present so downstream code can be
  shaped around the same API surface.
- Backends that need PIL, CVCUDA, image codecs, or torchvision `tv_tensors`
  metadata currently throw `NotImplementedError`.
- `datasets::MNIST` reads the same extracted IDX files as torchvision from
  `root/MNIST/raw`. Downloading is not implemented yet in C++.

Build:

```bash
cmake -S vision -B vision/build -DCMAKE_PREFIX_PATH=/path/to/libtorch
cmake --build vision/build
ctest --test-dir vision/build
```

Example:

```cpp
namespace v2 = marr::vision::transforms::v2;

auto pipeline = v2::Compose({
    std::make_shared<v2::RandomResizedCrop>(v2::Size2D{224, 224}),
    std::make_shared<v2::RandomHorizontalFlip>(0.5),
    std::make_shared<v2::ToDtype>(torch::kFloat32, true),
    std::make_shared<v2::Normalize>(
        std::vector<double>{0.485, 0.456, 0.406},
        std::vector<double>{0.229, 0.224, 0.225}),
});

auto image = torch::randint(0, 256, {3, 256, 256}, torch::kUInt8);
auto output = pipeline.forward(image);
```

MNIST dataset example:

```cpp
datasets::MNIST mnist("/path/to/data", true);
auto sample = mnist.get(0);

auto image = sample.data;   // uint8 tensor, [1, H, W]
auto target = sample.target.item<int64_t>();
```

Python dataset example:

```python
from marrvision.datasets import MNIST

mnist = MNIST("/path/to/data", train=True, download=True)
image, target = mnist[0]  # uint8 tensor [1, H, W], integer target
```

From the repository root, add `vision` to `PYTHONPATH` when using the Python
package directly:

```bash
PYTHONPATH=vision python vision/examples/python_mnist_dataset.py /path/to/data --download
```
