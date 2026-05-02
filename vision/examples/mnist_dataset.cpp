#include <iostream>
#include <marrvision/datasets.hpp>
#include <string>

int main(int argc, char **argv) {
  const std::string root = argc > 1 ? argv[1] : ".";

  marr::vision::datasets::MNIST dataset(root, true);
  auto sample = dataset.get(0);

  std::cout << "MNIST train samples: " << dataset.size().value() << "\n";
  std::cout << "First image shape: " << sample.data.sizes() << "\n";
  std::cout << "First target: " << sample.target.item<int64_t>() << "\n";

  return 0;
}
