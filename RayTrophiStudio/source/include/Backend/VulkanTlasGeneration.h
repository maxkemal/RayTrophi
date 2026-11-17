#pragma once

#include <atomic>
#include <cstdint>

namespace VulkanRT {

// ★★★★★ Identity of the TLAS OBJECT, not of its handle.
//
//   MEASURED 2026-11-17 (VK_EXT_device_fault + buffer ledger): RayFusion rebuilt
//   its TLAS, the driver handed the new object the SAME VkAccelerationStructureKHR
//   value and even the same storage VA (0x6770000 twice in a row), and every
//   consumer that decided "rewrite my descriptor?" by hashing the handle kept a
//   descriptor naming the destroyed object. The GPU then traversed freed memory:
//   read_invalid at offset 0 of the old TLAS buffer -> device lost.
//
//   Bumped whenever a TLAS object is created or destroyed (an in-place rebuild
//   into the same object does not bump: its descriptors stay valid). Consumers
//   mix this into their descriptor signature next to (not instead of) the handle.
inline std::atomic<uint64_t>& tlasObjectGenerationCounter() {
    static std::atomic<uint64_t> generation{1};
    return generation;
}
inline uint64_t tlasObjectGeneration() {
    return tlasObjectGenerationCounter().load(std::memory_order_acquire);
}
inline void bumpTlasObjectGeneration() {
    tlasObjectGenerationCounter().fetch_add(1, std::memory_order_acq_rel);
}

} // namespace VulkanRT
