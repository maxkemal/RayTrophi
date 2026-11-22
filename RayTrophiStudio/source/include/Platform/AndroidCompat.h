// Force-included on Android only (scripts/player_android_manifest.json, cxx_flags_common).
// Fills two gaps of the NDK libc++/Bionic that Windows-first code relies on. Never included on
// Windows builds, so the MSVC code paths are unchanged.
#pragma once

#if defined(__ANDROID__) && defined(__cplusplus)

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <type_traits>
#include <utility>

// MSVC bounded string copy. count characters are copied at most, the result is always terminated,
// and 0 is returned (truncation is not reported as an error, matching _TRUNCATE-less callers here).
inline int strncpy_s(char* dest, std::size_t dest_size, const char* src, std::size_t count) {
    if (!dest || dest_size == 0) return 22;
    if (!src) { dest[0] = '\0'; return 22; }
    std::size_t n = 0;
    while (n < count && n + 1 < dest_size && src[n] != '\0') { dest[n] = src[n]; ++n; }
    dest[n] = '\0';
    return 0;
}
template <std::size_t N>
inline int strncpy_s(char (&dest)[N], const char* src, std::size_t count) {
    return strncpy_s(dest, N, src, count);
}

// libc++ in the NDK ships <execution> without the parallel policies. The policies below only
// select the overloads; the algorithms run sequentially (correct, just not parallel).
namespace std {
namespace execution {
struct rt_seq_fallback_policy {};
inline constexpr rt_seq_fallback_policy par{};
inline constexpr rt_seq_fallback_policy par_unseq{};
} // namespace execution

template <class It, class Fn>
inline void for_each(const execution::rt_seq_fallback_policy&, It first, It last, Fn fn) {
    std::for_each(first, last, fn);
}
template <class It, class Size, class Fn>
inline It for_each_n(const execution::rt_seq_fallback_policy&, It first, Size n, Fn fn) {
    return std::for_each_n(first, n, fn);
}
template <class It, class Pred>
inline It remove_if(const execution::rt_seq_fallback_policy&, It first, It last, Pred pred) {
    return std::remove_if(first, last, pred);
}
template <class It>
inline void sort(const execution::rt_seq_fallback_policy&, It first, It last) {
    std::sort(first, last);
}
template <class It, class Cmp>
inline void sort(const execution::rt_seq_fallback_policy&, It first, It last, Cmp cmp) {
    std::sort(first, last, cmp);
}
} // namespace std

#endif
