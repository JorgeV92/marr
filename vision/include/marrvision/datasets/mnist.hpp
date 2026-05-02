#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <torch/torch.h>

namespace marr {
namespace vision {
namespace datasets {

class Error : public std::runtime_error {
 public:
  explicit Error(const std::string &message) : std::runtime_error(message) {}
};

class NotImplementedError : public Error {
 public:
  explicit NotImplementedError(const std::string &feature)
      : Error(feature + " is not implemented in marrvision.datasets yet") {}
};

struct Resource {
  std::string filename;
  std::string md5;
};

struct MNISTOptions {
  std::filesystem::path root;
  bool train = true;
  bool download = false;
  bool add_channel_dimension = true;
};

namespace detail {

inline std::vector<uint8_t> read_binary_file(
    const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    throw Error("Could not open file: " + path.string());
  }

  const auto end = input.tellg();
  if (end < std::streampos(0)) {
    throw Error("Could not determine file size: " + path.string());
  }

  const auto size = static_cast<std::streamsize>(end);
  std::vector<uint8_t> data(static_cast<size_t>(size));
  input.seekg(0, std::ios::beg);
  if (!data.empty()) {
    input.read(reinterpret_cast<char *>(data.data()), size);
  }
  if (!input) {
    throw Error("Could not read file: " + path.string());
  }
  return data;
}

inline uint32_t read_be_u32(const std::vector<uint8_t> &data, size_t offset) {
  if (offset + 4 > data.size()) {
    throw Error("Unexpected end of IDX file");
  }
  return (static_cast<uint32_t>(data[offset]) << 24) |
         (static_cast<uint32_t>(data[offset + 1]) << 16) |
         (static_cast<uint32_t>(data[offset + 2]) << 8) |
         static_cast<uint32_t>(data[offset + 3]);
}

inline int64_t checked_product(const std::vector<int64_t> &shape) {
  return std::accumulate(
      shape.begin(), shape.end(), int64_t{1}, [](int64_t acc, int64_t value) {
        if (value < 0 ||
            (value != 0 && acc > std::numeric_limits<int64_t>::max() / value)) {
          throw Error("IDX tensor shape is invalid or too large");
        }
        return acc * value;
      });
}

}  // namespace detail

inline torch::Tensor read_sn3_pascalvincent_tensor(
    const std::filesystem::path &path, bool strict = true) {
  const auto bytes = detail::read_binary_file(path);
  if (bytes.size() < 8) {
    throw Error("IDX file is too small: " + path.string());
  }

  const auto magic = detail::read_be_u32(bytes, 0);
  const auto data_type = static_cast<uint8_t>((magic >> 8) & 0xFF);
  const auto dimensions = static_cast<uint8_t>(magic & 0xFF);

  if (dimensions < 1 || dimensions > 3) {
    throw Error("Expected IDX tensor with 1 to 3 dimensions");
  }
  if (data_type != 8) {
    throw Error("Only unsigned byte IDX tensors are supported");
  }

  std::vector<int64_t> shape;
  shape.reserve(dimensions);
  for (uint8_t i = 0; i < dimensions; ++i) {
    shape.push_back(
        static_cast<int64_t>(detail::read_be_u32(bytes, 4 * (i + 1))));
  }

  const auto header_size = static_cast<size_t>(4 * (dimensions + 1));
  const auto value_count = detail::checked_product(shape);
  const auto expected_size = header_size + static_cast<size_t>(value_count);

  if (bytes.size() < expected_size ||
      (strict && bytes.size() != expected_size)) {
    std::ostringstream message;
    message << "IDX payload size mismatch for " << path << ": expected "
            << expected_size << " bytes, got " << bytes.size();
    throw Error(message.str());
  }

  return torch::from_blob(const_cast<uint8_t *>(bytes.data() + header_size),
                          shape, torch::TensorOptions().dtype(torch::kUInt8))
      .clone();
}

inline torch::Tensor read_label_file(const std::filesystem::path &path) {
  auto tensor = read_sn3_pascalvincent_tensor(path, false);
  if (tensor.dim() != 1) {
    throw Error("MNIST label file must contain a 1D tensor");
  }
  return tensor.to(torch::kLong);
}

inline torch::Tensor read_image_file(const std::filesystem::path &path) {
  auto tensor = read_sn3_pascalvincent_tensor(path, false);
  if (tensor.dim() != 3) {
    throw Error("MNIST image file must contain a 3D tensor");
  }
  return tensor;
}

class MNIST : public torch::data::datasets::Dataset<MNIST> {
 public:
  inline static const std::vector<std::string> mirrors = {
      "https://ossci-datasets.s3.amazonaws.com/mnist/",
      "http://yann.lecun.com/exdb/mnist/",
  };

  inline static const std::vector<Resource> resources = {
      {"train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"},
      {"train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"},
      {"t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"},
      {"t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"},
  };

  inline static const std::array<std::string, 10> classes = {
      "0 - zero", "1 - one", "2 - two",   "3 - three", "4 - four",
      "5 - five", "6 - six", "7 - seven", "8 - eight", "9 - nine",
  };

  explicit MNIST(MNISTOptions options) : options_(std::move(options)) {
    if (options_.download) {
      download();
    }
    if (!check_exists()) {
      throw Error("Dataset not found. Place extracted MNIST files in " +
                  raw_folder().string());
    }

    auto loaded = load_data();
    data_ = std::move(loaded.first);
    targets_ = std::move(loaded.second);

    if (data_.size(0) != targets_.size(0)) {
      throw Error("MNIST image and label counts do not match");
    }
  }

  explicit MNIST(const std::filesystem::path &root, bool train = true,
                 bool download = false)
      : MNIST(MNISTOptions{root, train, download}) {}

  torch::data::Example<> get(size_t index) override {
    const auto i = static_cast<int64_t>(index);
    if (i < 0 || i >= data_.size(0)) {
      throw Error("MNIST index out of range");
    }

    auto image = data_.select(0, i);
    if (options_.add_channel_dimension) {
      image = image.unsqueeze(0);
    }
    return {image.clone(), targets_.select(0, i).clone()};
  }

  torch::optional<size_t> size() const override {
    return static_cast<size_t>(data_.size(0));
  }

  const std::filesystem::path &root() const { return options_.root; }

  bool train() const { return options_.train; }

  std::filesystem::path raw_folder() const {
    return options_.root / "MNIST" / "raw";
  }

  std::filesystem::path processed_folder() const {
    return options_.root / "MNIST" / "processed";
  }

  const torch::Tensor &data() const { return data_; }

  const torch::Tensor &targets() const { return targets_; }

  std::map<std::string, int64_t> class_to_idx() const {
    std::map<std::string, int64_t> mapping;
    for (int64_t i = 0; i < static_cast<int64_t>(classes.size()); ++i) {
      mapping.emplace(classes[static_cast<size_t>(i)], i);
    }
    return mapping;
  }

  bool check_exists() const {
    for (const auto &resource : resources) {
      if (!std::filesystem::is_regular_file(
              raw_folder() / uncompressed_filename(resource.filename))) {
        return false;
      }
    }
    return true;
  }

  void download() const {
    if (check_exists()) {
      return;
    }
    throw NotImplementedError(
        "MNIST download. Download and extract the files "
        "listed in MNIST::resources into " +
        raw_folder().string());
  }

 private:
  static std::string uncompressed_filename(const std::string &filename) {
    const std::string suffix = ".gz";
    if (filename.size() >= suffix.size() &&
        filename.compare(filename.size() - suffix.size(), suffix.size(),
                         suffix) == 0) {
      return filename.substr(0, filename.size() - suffix.size());
    }
    return filename;
  }

  std::pair<torch::Tensor, torch::Tensor> load_data() const {
    const auto prefix = options_.train ? "train" : "t10k";
    const auto image_file =
        raw_folder() / (std::string(prefix) + "-images-idx3-ubyte");
    const auto label_file =
        raw_folder() / (std::string(prefix) + "-labels-idx1-ubyte");
    return {read_image_file(image_file), read_label_file(label_file)};
  }

  MNISTOptions options_;
  torch::Tensor data_;
  torch::Tensor targets_;
};

}  // namespace datasets
}  // namespace vision
}  // namespace marr
