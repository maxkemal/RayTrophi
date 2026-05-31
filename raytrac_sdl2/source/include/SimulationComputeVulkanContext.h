/*
 * SimulationComputeVulkanContext.h
 * Opaque context struct for passing Vulkan handles to the simulation compute
 * backend without including Vulkan headers in SimulationCompute.h.
 */
#pragma once
#include <cstdint>

namespace RayTrophiSim {

struct SimulationComputeVulkanContext {
    void*    device          = nullptr; // VkDevice
    void*    physical_device = nullptr; // VkPhysicalDevice
    void*    compute_queue   = nullptr; // VkQueue
    uint32_t queue_family_index = 0;
    // True only when VK_EXT_shader_atomic_float (shaderBufferFloat32AtomicAdd)
    // was actually ENABLED on the logical device above — not merely supported by
    // the physical device. The fluid P2G scatter + density splat kernels use
    // float atomicAdd; running them without this enabled is undefined behaviour
    // (wrong velocity field). The sim compute backend gates those kernels on it.
    bool     shader_atomic_float_enabled = false;
};

} // namespace RayTrophiSim
