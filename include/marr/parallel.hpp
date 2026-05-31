#pragma once

#include <cstddef>

#include <marr/detail/thread_pool.hpp>

namespace marr {

inline void set_num_threads(std::size_t n)
{
    detail::configured_num_threads().store(n == 0 ? std::size_t{1} : n);
}

inline std::size_t get_num_threads()
{
    return detail::configured_num_threads().load();
}

inline void set_parallel_enabled(bool enabled)
{
    detail::configured_parallel_enabled().store(enabled);
}

inline bool is_parallel_enabled()
{
    return detail::configured_parallel_enabled().load();
}

class NoParallelGuard {
public:
    NoParallelGuard()
        : previous_(is_parallel_enabled())
    {
        set_parallel_enabled(false);
    }

    ~NoParallelGuard()
    {
        set_parallel_enabled(previous_);
    }

    NoParallelGuard(const NoParallelGuard&) = delete;
    NoParallelGuard& operator=(const NoParallelGuard&) = delete;

private:
    bool previous_;
};

} // namespace marr
