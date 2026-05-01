#include <iostream>
#include <memory>
#include <vector>

#include <torch/torch.h>

#include <marrvision/transforms/v2.hpp>

int main() {
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

  std::cout << output.sizes() << " " << output.dtype() << "\n";
  return 0;
}
