#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

#include <torch/torch.h>

#include <marrvision/datasets.hpp>

namespace {

void write_be_u32(std::ofstream &output, uint32_t value) {
  const char bytes[] = {
      static_cast<char>((value >> 24) & 0xFF),
      static_cast<char>((value >> 16) & 0xFF),
      static_cast<char>((value >> 8) & 0xFF),
      static_cast<char>(value & 0xFF),
  };
  output.write(bytes, sizeof(bytes));
}

void write_idx_images(const std::filesystem::path &path, uint32_t count,
                      uint32_t rows, uint32_t cols,
                      const std::vector<uint8_t> &values) {
  std::ofstream output(path, std::ios::binary);
  write_be_u32(output, 0x00000803);
  write_be_u32(output, count);
  write_be_u32(output, rows);
  write_be_u32(output, cols);
  output.write(reinterpret_cast<const char *>(values.data()),
               static_cast<std::streamsize>(values.size()));
}

void write_idx_labels(const std::filesystem::path &path, uint32_t count,
                      const std::vector<uint8_t> &values) {
  std::ofstream output(path, std::ios::binary);
  write_be_u32(output, 0x00000801);
  write_be_u32(output, count);
  output.write(reinterpret_cast<const char *>(values.data()),
               static_cast<std::streamsize>(values.size()));
}

void write_split(const std::filesystem::path &raw_folder,
                 const std::string &prefix,
                 const std::vector<uint8_t> &image_values,
                 const std::vector<uint8_t> &label_values) {
  write_idx_images(raw_folder / (prefix + "-images-idx3-ubyte"), 2, 2, 3,
                   image_values);
  write_idx_labels(raw_folder / (prefix + "-labels-idx1-ubyte"), 2,
                   label_values);
}

}  // namespace

int main() {
  namespace datasets = marr::vision::datasets;

  const auto root =
      std::filesystem::temp_directory_path() / "marrvision_mnist_smoke";
  const auto raw_folder = root / "MNIST" / "raw";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(raw_folder);

  write_split(raw_folder, "train", {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
              {7, 3});
  write_split(raw_folder, "t10k",
              {12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}, {2, 1});

  datasets::MNIST train(root, true);
  assert(train.check_exists());
  assert(train.raw_folder() == raw_folder);
  assert(train.size().value() == 2);
  assert(train.data().sizes() == torch::IntArrayRef({2, 2, 3}));
  assert(train.targets().sizes() == torch::IntArrayRef({2}));
  assert(train.class_to_idx().at("7 - seven") == 7);

  auto train_sample = train.get(1);
  assert(train_sample.data.sizes() == torch::IntArrayRef({1, 2, 3}));
  assert(train_sample.data.scalar_type() == torch::kUInt8);
  assert(train_sample.data[0][0][0].item<int64_t>() == 6);
  assert(train_sample.target.item<int64_t>() == 3);

  datasets::MNIST test(root, false);
  auto test_sample = test.get(0);
  assert(test_sample.data[0][0][0].item<int64_t>() == 12);
  assert(test_sample.target.item<int64_t>() == 2);

  std::filesystem::remove_all(root);
  return 0;
}
