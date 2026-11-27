#pragma once

// Zero-allocation dispatch lookup helpers for VulkanSimulationComputeBackend:
//   - FixedDescriptorKey / SimulationComputeDescriptorCache: avoids per-dispatch
//     std::vector<uint64_t> allocations and O(log N) std::map lookups.
//   - TransparentStringHash / PaintWetPipelineLookup: avoids per-dispatch
//     std::string allocations and "paint_wet." + std::to_string(mode) formatting.

#include "SimulationCompute.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace RayTrophiSim {

// Fixed-capacity buffer-handle key for descriptor-set caching on dispatch().
struct FixedDescriptorKey {
    static constexpr uint32_t kInlineCapacity = 32;

    uint64_t handles[kInlineCapacity]{};
    uint32_t count = 0;
    uint64_t hash = 0;

    static FixedDescriptorKey make(const ComputeBufferHandle* buffers,
                                   uint32_t buffer_count) noexcept {
        FixedDescriptorKey key;
        key.count = buffer_count;
        uint64_t h = 14695981039346656037ull;
        const uint32_t n = std::min(buffer_count, kInlineCapacity);
        for (uint32_t i = 0; i < n; ++i) {
            const uint64_t id = buffers[i].id;
            key.handles[i] = id;
            h ^= id + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        }
        h ^= static_cast<uint64_t>(buffer_count);
        key.hash = h;
        return key;
    }

    bool operator==(const FixedDescriptorKey& other) const noexcept {
        if (count != other.count || hash != other.hash) {
            return false;
        }
        for (uint32_t i = 0; i < count; ++i) {
            if (handles[i] != other.handles[i]) {
                return false;
            }
        }
        return true;
    }
};

struct FixedDescriptorKeyHash {
    std::size_t operator()(const FixedDescriptorKey& key) const noexcept {
        return static_cast<std::size_t>(key.hash);
    }
};

// Hash-cached descriptor set table with inline keys up to 32 bindings and
// clean std::map fallback above 32 bindings.
template <typename DescriptorSetT>
class SimulationComputeDescriptorCache {
public:
    bool find(const ComputeBufferHandle* buffers,
              uint32_t buffer_count,
              DescriptorSetT& out) const noexcept {
        if (buffer_count <= FixedDescriptorKey::kInlineCapacity) {
            const FixedDescriptorKey key =
                FixedDescriptorKey::make(buffers, buffer_count);
            const auto it = m_fast.find(key);
            if (it == m_fast.end()) {
                return false;
            }
            out = it->second;
            return true;
        }
        m_overflowKeyScratch.resize(buffer_count);
        for (uint32_t i = 0; i < buffer_count; ++i) {
            m_overflowKeyScratch[i] = buffers[i].id;
        }
        const auto it = m_overflow.find(m_overflowKeyScratch);
        if (it == m_overflow.end()) {
            return false;
        }
        out = it->second;
        return true;
    }

    void insert(const ComputeBufferHandle* buffers,
                uint32_t buffer_count,
                DescriptorSetT ds) {
        if (buffer_count <= FixedDescriptorKey::kInlineCapacity) {
            m_fast.emplace(FixedDescriptorKey::make(buffers, buffer_count), ds);
            return;
        }
        std::vector<uint64_t> key(buffer_count);
        for (uint32_t i = 0; i < buffer_count; ++i) {
            key[i] = buffers[i].id;
        }
        m_overflow.emplace(std::move(key), ds);
    }

    void clear() noexcept {
        m_fast.clear();
        m_overflow.clear();
    }

private:
    std::unordered_map<FixedDescriptorKey, DescriptorSetT, FixedDescriptorKeyHash> m_fast;
    mutable std::vector<uint64_t> m_overflowKeyScratch;
    std::map<std::vector<uint64_t>, DescriptorSetT> m_overflow;
};

struct TransparentStringHash {
    using is_transparent = void;

    std::size_t operator()(std::string_view sv) const noexcept {
        return std::hash<std::string_view>{}(sv);
    }

    std::size_t operator()(const std::string& s) const noexcept {
        return std::hash<std::string_view>{}(s);
    }

    std::size_t operator()(const char* s) const noexcept {
        return std::hash<std::string_view>{}(
            s ? std::string_view(s) : std::string_view{});
    }
};

inline constexpr std::array<std::string_view, 11> kPaintWetKernelNames = {
    "paint_wet.0",
    "paint_wet.1",
    "paint_wet.2",
    "paint_wet.3",
    "paint_wet.4",
    "paint_wet.5",
    "paint_wet.6",
    "paint_wet.7",
    "paint_wet.8",
    "paint_wet.9",
    "paint_wet.10",
};

// Resolves paint_wet.0..10 without allocating strings and caches the 11
// specialized pipeline pointers for O(1) lookup on hot wet-paint ticks.
template <typename PipelineEntryT>
struct PaintWetPipelineLookup {
    std::array<const PipelineEntryT*, 11> slots{};

    void clear() noexcept {
        slots.fill(nullptr);
    }

    template <typename PipelineMapT, typename EnsureFn>
    const PipelineEntryT* resolveAndFind(const ComputeDispatch& cmd,
                                         std::string_view& out_kernel,
                                         const PipelineMapT& pipelines,
                                         EnsureFn&& ensure_fn) {
        const std::string_view raw_kernel = cmd.kernel
                                                ? std::string_view(cmd.kernel)
                                                : std::string_view{};
        uint32_t paint_wet_mode = 0xFFFFFFFFu;
        out_kernel = raw_kernel;
        if (raw_kernel == "paint_wet" && cmd.constants_size >= sizeof(uint32_t)) {
            uint32_t mode = 0;
            std::memcpy(&mode, cmd.constants, sizeof(uint32_t));
            if (mode <= 10u) {
                paint_wet_mode = mode;
                out_kernel = kPaintWetKernelNames[mode];
                if (slots[mode] != nullptr) {
                    return slots[mode];
                }
            }
        }

        auto pit = pipelines.find(out_kernel);
        if (pit == pipelines.end()) {
            pit = ensure_fn(out_kernel);
        }
        if (pit != pipelines.end()) {
            if (paint_wet_mode <= 10u) {
                slots[paint_wet_mode] = &pit->second;
            }
            return &pit->second;
        }

        if (out_kernel.size() >= 10 && out_kernel.compare(0, 10, "paint_wet.") == 0) {
            out_kernel = "paint_wet";
            pit = pipelines.find(out_kernel);
            if (pit == pipelines.end()) {
                pit = ensure_fn(out_kernel);
            }
            if (pit != pipelines.end()) {
                return &pit->second;
            }
        }
        return nullptr;
    }
};

} // namespace RayTrophiSim
