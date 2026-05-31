#pragma once

#include <atomic>
#include <cstddef>
#include <thread>

namespace marr::detail {

inline std::size_t default_num_threads()
{
    const unsigned int hardware_threads = std::thread::hardware_concurrency();
    return hardware_threads == 0 ? std::size_t{1} : static_cast<std::size_t>(hardware_threads);
}

inline std::atomic<std::size_t>& configured_num_threads()
{
    static std::atomic<std::size_t> threads{default_num_threads()};
    return threads;
}

inline std::atomic<bool>& configured_parallel_enabled()
{
    static std::atomic<bool> enabled{true};
    return enabled;
}

} // namespace marr::detail
