cmake_minimum_required(VERSION 3.20)
project(tinyViTCpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Torch REQUIRED)

add_library(tinyViTCpp INTERFACE)
target_include_directories(tinyViTCpp INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/tinyvit)
target_link_libraries(tinyViTCpp INTERFACE ${TORCH_LIBRARIES})
target_compile_options(tinyViTCpp INTERFACE ${TORCH_CXX_FLAGS})

add_executable(train_tinyvit_hier examples/train_tinyvit_hier.cpp)
target_link_libraries(train_tinyvit_hier PRIVATE tinyViTCpp)