#pragma once

#include <cstddef>
#include <new>
#include <utility>

#if defined(_MSC_VER)
#include <malloc.h>
#else
#include <cstdlib>
#endif

namespace DNA {

    /**
     * @brief A 32-byte aligned allocator for AVX256/AVX512 vectorization.
     * Enforces memory alignment on dynamic allocations (like std::vector buffers).
     */
    template <typename T, size_t Alignment = 32>
    struct AlignedAllocator {
        using value_type = T;
        using size_type = size_t;
        using difference_type = ptrdiff_t;

        AlignedAllocator() noexcept = default;
        template <typename U> AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

        T* allocate(size_t n) {
            if (n == 0) return nullptr;
            
            void* ptr = nullptr;
#if defined(_MSC_VER)
            ptr = _aligned_malloc(n * sizeof(T), Alignment);
#else
            if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0) {
                ptr = nullptr;
            }
#endif
            if (!ptr) {
                throw std::bad_alloc();
            }
            return reinterpret_cast<T*>(ptr);
        }

        void deallocate(T* p, size_t) noexcept {
            if (!p) return;
#if defined(_MSC_VER)
            _aligned_free(p);
#else
            free(p);
#endif
        }

        template <typename U>
        struct rebind {
            using other = AlignedAllocator<U, Alignment>;
        };

        bool operator==(const AlignedAllocator&) const noexcept { return true; }
        bool operator!=(const AlignedAllocator&) const noexcept { return false; }
    };

} // namespace DNA
