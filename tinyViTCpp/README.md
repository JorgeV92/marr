# tinyViTCpp

A tiny tiny example of tinyViT for vision transformers in c++.

run by provide correct path to `libtorch`

```bash
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH=$HOME/path/libtorch \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build -j2

./build/train_tinyvit_hier
```