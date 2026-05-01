# marrvision transforms v2

This package is the C++/LibTorch start of a `torchvision.transforms.v2`-style
API for marrvision.

The public namespace is:

```cpp
#include <marrvision/transforms/v2.hpp>

namespace v2 = marr::vision::transforms::v2;
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
