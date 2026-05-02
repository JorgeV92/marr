cmake_minimum_required(VERSION 3.20)
project(marrvision LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Torch REQUIRED)

add_library(marrvision INTERFACE)
target_include_directories(
  marrvision
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(marrvision INTERFACE ${TORCH_LIBRARIES})
target_compile_options(marrvision INTERFACE ${TORCH_CXX_FLAGS})

add_library(marrvision_transforms_v2 INTERFACE)
target_link_libraries(marrvision_transforms_v2 INTERFACE marrvision)

add_library(marrvision_datasets INTERFACE)
target_link_libraries(marrvision_datasets INTERFACE marrvision)

add_executable(marrvision_v2_basic examples/basic_pipeline.cpp)
target_link_libraries(marrvision_v2_basic PRIVATE marrvision)

add_executable(marrvision_mnist_basic examples/mnist_dataset.cpp)
target_link_libraries(marrvision_mnist_basic PRIVATE marrvision)

add_executable(marrvision_v2_smoke tests/smoke.cpp)
target_link_libraries(marrvision_v2_smoke PRIVATE marrvision)

add_executable(marrvision_mnist_smoke tests/mnist.cpp)
target_link_libraries(marrvision_mnist_smoke PRIVATE marrvision)

enable_testing()
add_test(NAME marrvision_v2_smoke COMMAND marrvision_v2_smoke)
add_test(NAME marrvision_mnist_smoke COMMAND marrvision_mnist_smoke)
