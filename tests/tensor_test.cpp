#include <marr/tensor.hpp>

#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

int failures = 0;

#define CHECK(expression)                                                                       \
    do {                                                                                        \
        if (!(expression)) {                                                                    \
            std::cerr << __FILE__ << ':' << __LINE__ << ": CHECK failed: " #expression << '\n'; \
            ++failures;                                                                         \
        }                                                                                       \
    } while (false)

template <typename Exception, typename Function>
void check_throws(Function function, std::string_view label)
{
    try {
        function();
    } catch (const Exception&) {
        return;
    } catch (const std::exception& error) {
        std::cerr << "Expected " << label << " to throw a different exception type, got: "
                  << error.what() << '\n';
        ++failures;
        return;
    }

    std::cerr << "Expected " << label << " to throw\n";
    ++failures;
}

template <typename T>
void check_tensor_values(const marr::Tensor<T>& tensor, const std::vector<T>& expected)
{
    CHECK(tensor.numel() == static_cast<std::int64_t>(expected.size()));
    for (std::int64_t i = 0; i < tensor.numel(); ++i) {
        CHECK(tensor[i] == expected[static_cast<std::size_t>(i)]);
    }
}

void test_default_construction()
{
    const marr::Tensor<int> tensor;

    CHECK(tensor.size() == 0);
    CHECK(tensor.numel() == 0);
    CHECK(tensor.dim() == 0);
    CHECK(tensor.ndim() == 0);
    CHECK(tensor.sizes().empty());
    CHECK(tensor.shape().empty());
    CHECK(tensor.strides().empty());
    CHECK(tensor.storage_offset() == 0);
    CHECK(tensor.empty());
}

void test_shape_stride_helpers()
{
    CHECK(marr::compute_numel({2, 3, 4}) == 24);
    CHECK(marr::compute_numel({2, 0, 4}) == 0);
    CHECK(marr::compute_numel({}) == 1);
    CHECK(marr::compute_total_size({2, 3, 4}) == 24);
    CHECK((marr::compute_contiguous_strides({2, 3, 4}) == marr::Strides{12, 4, 1}));
    CHECK((marr::compute_row_major_strides({2, 3, 4}) == marr::Shape{12, 4, 1}));
    CHECK((marr::compute_contiguous_strides({5}) == marr::Strides{1}));
    CHECK(marr::compute_contiguous_strides({}).empty());
}

void test_phase2_metadata()
{
    const marr::Tensor<int> tensor({2, 3, 4});

    CHECK(tensor.dim() == 3);
    CHECK(tensor.ndim() == 3);
    CHECK(tensor.numel() == 24);
    CHECK(tensor.size() == 24);
    CHECK((tensor.sizes() == marr::Sizes{2, 3, 4}));
    CHECK((tensor.shape() == marr::Shape{2, 3, 4}));
    CHECK((tensor.strides() == marr::Strides{12, 4, 1}));
    CHECK(tensor.size(0) == 2);
    CHECK(tensor.size(1) == 3);
    CHECK(tensor.size(2) == 4);
    CHECK(tensor.size(-1) == 4);
    CHECK(tensor.size(-2) == 3);
    CHECK(tensor.size(-3) == 2);
    CHECK(tensor.stride(0) == 12);
    CHECK(tensor.stride(1) == 4);
    CHECK(tensor.stride(2) == 1);
    CHECK(tensor.stride(-1) == 1);
    CHECK(tensor.stride(-2) == 4);
    CHECK(tensor.is_contiguous());
    CHECK(tensor.data_ptr() == tensor.data());

    check_throws<std::out_of_range>([&] { (void)tensor.size(3); }, "positive dim out of range");
    check_throws<std::out_of_range>([&] { (void)tensor.size(-4); }, "negative dim out of range");
}

void test_phase2_construction_and_indexing()
{
    const marr::Tensor<int> zeros({2, 3});
    CHECK(zeros.numel() == 6);
    CHECK(zeros[0] == 0);
    CHECK(zeros[5] == 0);

    const marr::Tensor<int> filled({2, 2}, 7);
    check_tensor_values(filled, {7, 7, 7, 7});

    marr::Tensor<int> tensor({2, 3}, std::vector<int>{1, 2, 3, 4, 5, 6});
    CHECK(tensor[0] == 1);
    CHECK(tensor[5] == 6);
    CHECK(tensor(0, 0) == 1);
    CHECK(tensor(0, 2) == 3);
    CHECK(tensor(1, 0) == 4);
    CHECK(tensor(1, 2) == 6);
    CHECK(tensor({1, 1}) == 5);
    CHECK(tensor.at({1, 2}) == 6);
    CHECK(tensor(marr::Sizes{0, 1}) == 2);

    tensor(0, 2) = 99;
    CHECK(tensor[2] == 99);

    const marr::Tensor<int>& const_tensor = tensor;
    CHECK(const_tensor.at({1, 2}) == 6);

    check_throws<std::out_of_range>([&] { (void)tensor[6]; }, "flat index past end");
    check_throws<std::out_of_range>([&] { (void)tensor[-1]; }, "negative flat index");
    check_throws<std::out_of_range>([&] { (void)tensor(2, 0); }, "out-of-bounds row");
    check_throws<std::out_of_range>([&] { (void)tensor(0, 3); }, "out-of-bounds column");
    check_throws<std::out_of_range>([&] { (void)tensor(-1, 0); }, "negative index");
    check_throws<std::invalid_argument>([&] { (void)tensor(0); }, "rank mismatch");
}

void test_phase2_reshape_and_view()
{
    const marr::Tensor<int> tensor({2, 3}, std::vector<int>{1, 2, 3, 4, 5, 6});

    const auto reshaped = tensor.reshape({3, 2});
    CHECK((reshaped.sizes() == marr::Sizes{3, 2}));
    CHECK((reshaped.strides() == marr::Strides{2, 1}));
    CHECK(reshaped.numel() == 6);
    CHECK(reshaped(0, 0) == 1);
    CHECK(reshaped(2, 1) == 6);
    CHECK((tensor.sizes() == marr::Sizes{2, 3}));

    const auto viewed = tensor.view({6});
    CHECK((viewed.sizes() == marr::Sizes{6}));
    CHECK((viewed.strides() == marr::Strides{1}));
    CHECK(viewed[5] == 6);

    check_throws<std::invalid_argument>([&] { (void)tensor.reshape({4, 2}); }, "reshape size mismatch");
    check_throws<std::invalid_argument>([&] { (void)tensor.view({5}); }, "view size mismatch");
}

void test_phase2_invalid_shape_and_data_errors()
{
    check_throws<std::invalid_argument>(
        [] { marr::Tensor<int> tensor({2, 3}, std::vector<int>{1, 2, 3}); },
        "data size mismatch"
    );

    check_throws<std::invalid_argument>(
        [] { marr::Tensor<int> tensor({-1, 3}); },
        "negative shape"
    );

    const std::int64_t max = std::numeric_limits<std::int64_t>::max();
    check_throws<std::invalid_argument>(
        [max] { marr::Tensor<int> tensor({max, 2}); },
        "shape element count overflow"
    );

    check_throws<std::invalid_argument>(
        [max] { (void)marr::compute_contiguous_strides({0, max, max}); },
        "stride overflow"
    );
}

void test_phase3_tensor_tensor_operations()
{
    const marr::Tensor<float> a({2, 2}, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
    const marr::Tensor<float> b({2, 2}, std::vector<float>{10.0f, 20.0f, 30.0f, 40.0f});

    check_tensor_values(a + b, {11.0f, 22.0f, 33.0f, 44.0f});
    check_tensor_values(b - a, {9.0f, 18.0f, 27.0f, 36.0f});
    check_tensor_values(a * b, {10.0f, 40.0f, 90.0f, 160.0f});
    check_tensor_values(b / a, {10.0f, 10.0f, 10.0f, 10.0f});
}

void test_phase3_tensor_scalar_operations()
{
    const marr::Tensor<float> a({2, 2}, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});

    check_tensor_values(a + 2.0f, {3.0f, 4.0f, 5.0f, 6.0f});
    check_tensor_values(2.0f + a, {3.0f, 4.0f, 5.0f, 6.0f});
    check_tensor_values(a - 2.0f, {-1.0f, 0.0f, 1.0f, 2.0f});
    check_tensor_values(2.0f - a, {1.0f, 0.0f, -1.0f, -2.0f});
    check_tensor_values(a * 2.0f, {2.0f, 4.0f, 6.0f, 8.0f});
    check_tensor_values(2.0f * a, {2.0f, 4.0f, 6.0f, 8.0f});
    check_tensor_values(a / 2.0f, {0.5f, 1.0f, 1.5f, 2.0f});
    check_tensor_values(8.0f / a, {8.0f, 4.0f, 8.0f / 3.0f, 2.0f});
}

void test_phase3_unary_operations()
{
    const marr::Tensor<float> tensor({4}, std::vector<float>{-2.0f, -1.0f, 0.0f, 3.0f});

    check_tensor_values(-tensor, {2.0f, 1.0f, -0.0f, -3.0f});
    check_tensor_values(marr::abs(tensor), {2.0f, 1.0f, 0.0f, 3.0f});
    check_tensor_values(marr::relu(tensor), {0.0f, 0.0f, 0.0f, 3.0f});
}

void test_phase4_broadcasting_matrix_with_vector()
{
    const marr::Tensor<float> a({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
    const marr::Tensor<float> b({3}, std::vector<float>{10, 20, 30});

    const auto c = a + b;
    CHECK((c.sizes() == marr::Sizes{2, 3}));
    check_tensor_values(c, {11, 22, 33, 14, 25, 36});
}

void test_phase4_broadcasting_three_dimensions()
{
    const marr::Tensor<float> a(
        {4, 1, 3},
        std::vector<float>{
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
            10, 11, 12,
        }
    );
    const marr::Tensor<float> b(
        {1, 5, 3},
        std::vector<float>{
            100, 200, 300,
            400, 500, 600,
            700, 800, 900,
            1000, 1100, 1200,
            1300, 1400, 1500,
        }
    );

    const auto c = a + b;
    CHECK((c.sizes() == marr::Sizes{4, 5, 3}));
    CHECK(c(0, 0, 0) == 101);
    CHECK(c(0, 4, 2) == 1503);
    CHECK(c(3, 0, 1) == 211);
    CHECK(c(3, 4, 2) == 1512);
}

void test_phase4_broadcasting_column_vector()
{
    const marr::Tensor<float> a({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
    const marr::Tensor<float> b({2, 1}, std::vector<float>{10, 20});

    const auto c = a + b;
    CHECK((c.sizes() == marr::Sizes{2, 3}));
    check_tensor_values(c, {11, 12, 13, 24, 25, 26});
}

void test_phase4_scalar_broadcasting()
{
    const marr::Tensor<float> a({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
    const auto scalar = marr::full<float>({}, 2.0f);

    const auto c = a + scalar;
    CHECK((c.sizes() == marr::Sizes{2, 3}));
    check_tensor_values(c, {3, 4, 5, 6, 7, 8});

    const auto d = scalar * a;
    CHECK((d.sizes() == marr::Sizes{2, 3}));
    check_tensor_values(d, {2, 4, 6, 8, 10, 12});
}

void test_phase4_incompatible_shapes_throw()
{
    const marr::Tensor<float> a({2, 3});
    const marr::Tensor<float> b({4});

    check_throws<std::invalid_argument>([&] { (void)(a + b); }, "incompatible broadcast");
}

void test_factories()
{
    const auto zero_tensor = marr::zeros<int>({2, 2});
    check_tensor_values(zero_tensor, {0, 0, 0, 0});

    const auto one_tensor = marr::ones<int>({2, 2});
    check_tensor_values(one_tensor, {1, 1, 1, 1});

    const auto full_tensor = marr::full<int>({2, 2}, 7);
    check_tensor_values(full_tensor, {7, 7, 7, 7});

    const auto empty_tensor = marr::empty<int>({2, 2});
    CHECK((empty_tensor.sizes() == marr::Sizes{2, 2}));
    CHECK(empty_tensor.numel() == 4);
}

} // namespace

int main()
{
    test_default_construction();
    test_shape_stride_helpers();
    test_phase2_metadata();
    test_phase2_construction_and_indexing();
    test_phase2_reshape_and_view();
    test_phase2_invalid_shape_and_data_errors();
    test_phase3_tensor_tensor_operations();
    test_phase3_tensor_scalar_operations();
    test_phase3_unary_operations();
    test_phase4_broadcasting_matrix_with_vector();
    test_phase4_broadcasting_three_dimensions();
    test_phase4_broadcasting_column_vector();
    test_phase4_scalar_broadcasting();
    test_phase4_incompatible_shapes_throw();
    test_factories();

    if (failures != 0) {
        std::cerr << failures << " test failure(s)\n";
        return 1;
    }

    std::cout << "All tensor tests passed\n";
    return 0;
}
