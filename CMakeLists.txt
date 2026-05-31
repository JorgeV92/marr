cmake_minimum_required(VERSION 3.20)

project(marr VERSION 0.1.0 LANGUAGES CXX)

add_library(marr src/tensor.cpp)
add_library(marr::marr ALIAS marr)

target_compile_features(marr PUBLIC cxx_std_20)
target_include_directories(
    marr
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

option(MARR_BUILD_TESTS "Build Marr unit tests" ON)

if(MARR_BUILD_TESTS)
    enable_testing()

    add_executable(marr_tests tests/tensor_test.cpp)
    target_link_libraries(marr_tests PRIVATE marr::marr)

    add_test(NAME marr_tests COMMAND marr_tests)
endif()
