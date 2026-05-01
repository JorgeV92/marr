#include <cassert>
#include <memory>
#include <vector>

#include <torch/torch.h>

#include <marrvision/transforms/v2.hpp>

int main() {
  namespace v2 = marr::vision::transforms::v2;

  auto image = torch::randint(0, 256, {3, 64, 80}, torch::kUInt8);

  auto resized = v2::Resize(v2::Size2D{32, 48}).forward(image);
  assert(resized.sizes() == torch::IntArrayRef({3, 32, 48}));

  auto cropped = v2::CenterCrop(24).forward(resized);
  assert(cropped.sizes() == torch::IntArrayRef({3, 24, 24}));

  auto flipped = v2::RandomHorizontalFlip(1.0).forward(cropped);
  assert(flipped.sizes() == cropped.sizes());

  auto pipeline = v2::Compose({
      std::make_shared<v2::ToDtype>(torch::kFloat32, true),
      std::make_shared<v2::Normalize>(
          std::vector<double>{0.5, 0.5, 0.5},
          std::vector<double>{0.5, 0.5, 0.5}),
  });
  auto normalized = pipeline.forward(cropped);
  assert(normalized.scalar_type() == torch::kFloat32);
  assert(normalized.sizes() == cropped.sizes());

  auto gray = v2::Grayscale(1).forward(image);
  assert(gray.sizes() == torch::IntArrayRef({1, 64, 80}));

  auto video = torch::rand({8, 3, 16, 16});
  auto sampled = v2::UniformTemporalSubsample(4).forward(video);
  assert(sampled.sizes() == torch::IntArrayRef({4, 3, 16, 16}));

  return 0;
}
