cmake_minimum_required(VERSION 3.20)
project(marrvision_transforms_v2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Torch REQUIRED)

add_library(marrvision_transforms_v2 INTERFACE)
target_include_directories(
  marrvision_transforms_v2
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(marrvision_transforms_v2 INTERFACE ${TORCH_LIBRARIES})
target_compile_options(marrvision_transforms_v2 INTERFACE ${TORCH_CXX_FLAGS})

add_executable(marrvision_v2_basic examples/basic_pipeline.cpp)
target_link_libraries(marrvision_v2_basic PRIVATE marrvision_transforms_v2)

add_executable(marrvision_v2_smoke tests/smoke.cpp)
target_link_libraries(marrvision_v2_smoke PRIVATE marrvision_transforms_v2)

enable_testing()
add_test(NAME marrvision_v2_smoke COMMAND marrvision_v2_smoke)
