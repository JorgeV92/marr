#pragma once

#include <algorithm>
#include <cstdint>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include <marr/parallel.hpp>

namespace marr::detail {

inline constexpr std::int64_t PARALLEL_THRESHOLD = 32768;

inline bool should_parallelize(std::int64_t count)
{
    return count >= PARALLEL_THRESHOLD &&
           marr::is_parallel_enabled() &&
           marr::get_num_threads() > 1;
}

inline std::int64_t parallel_worker_count(std::int64_t count)
{
    if (count <= 0) {
        return 1;
    }
    return std::max<std::int64_t>(
        1,
        std::min<std::int64_t>(static_cast<std::int64_t>(marr::get_num_threads()), count)
    );
}

template <typename Fn>
void parallel_for(std::int64_t begin, std::int64_t end, Fn fn)
{
    if (end < begin) {
        throw std::invalid_argument("parallel_for end must be greater than or equal to begin");
    }

    const std::int64_t count = end - begin;
    if (!should_parallelize(count)) {
        for (std::int64_t i = begin; i < end; ++i) {
            fn(i);
        }
        return;
    }

    const std::int64_t workers = parallel_worker_count(count);
    const std::int64_t chunk_size = (count + workers - 1) / workers;

    std::mutex exception_mutex;
    std::exception_ptr first_exception;
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(workers));

    for (std::int64_t worker = 0; worker < workers; ++worker) {
        const std::int64_t chunk_begin = begin + worker * chunk_size;
        const std::int64_t chunk_end = std::min(end, chunk_begin + chunk_size);
        if (chunk_begin >= chunk_end) {
            break;
        }

        threads.emplace_back([&, chunk_begin, chunk_end] {
            try {
                for (std::int64_t i = chunk_begin; i < chunk_end; ++i) {
                    fn(i);
                }
            } catch (...) {
                std::lock_guard<std::mutex> lock(exception_mutex);
                if (!first_exception) {
                    first_exception = std::current_exception();
                }
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    if (first_exception) {
        std::rethrow_exception(first_exception);
    }
}

} // namespace marr::detail
