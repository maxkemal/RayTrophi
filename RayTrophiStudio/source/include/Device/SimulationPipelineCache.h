#pragma once
// Persistent VkPipelineCache for the Vulkan simulation compute backend
// (docs/dev/PLAYER_ANDROID_PERF_PLAN.md, O7a).
//
// Measured problem: the backend compiled every registered compute kernel
// (~270) at construction with VK_NULL_HANDLE as the cache. On Mali-G52 that
// was the 22.6 s `prepare_particles` stage of a 47 s Player open.
//
// Contract:
//  - One blob per GPU (vendor/device id) plus a ".shaders" stamp file. The
//    stamp hashes name, size and write time of every registered SPIR-V file;
//    a blob saved for other shaders is deleted, never loaded (the driver keeps
//    every loaded entry, so stale blobs only grow; see VulkanRTPipelineBuild).
//  - The blob header is validated against the device (vendor, device, UUID)
//    before it reaches the driver. Any rejection starts an empty cache and
//    reports the reason; nothing fakes a hit.
//  - RT_VK_PIPELINE_CACHE=0 disables the disk side (the in-memory cache is
//    still used, so repeated creation within one process stays cheap).
//  - Windows: %LOCALAPPDATA%/RayTrophiStudio/vk_pipeline_cache.
//    Android: Platform::appDataRoot()/vk_pipeline_cache (internal storage).

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace RayTrophiSim {

struct SimulationPipelineCacheStatus {
    bool enabled = false;          // disk persistence active
    bool loaded = false;           // a validated blob was handed to the driver
    size_t loaded_bytes = 0;
    size_t saved_bytes = 0;
    std::string path;              // UTF-8, empty when persistence is off
    std::string reject_reason;     // why a blob on disk was not used
};

class SimulationPipelineCache {
public:
    SimulationPipelineCache() = default;
    SimulationPipelineCache(const SimulationPipelineCache&) = delete;
    SimulationPipelineCache& operator=(const SimulationPipelineCache&) = delete;
    ~SimulationPipelineCache();

    // Creates the cache, seeded from disk when the stamp and header match.
    // Returns false only when vkCreatePipelineCache itself fails; handle()
    // is then VK_NULL_HANDLE and pipelines are created uncached.
    bool open(VkDevice device, VkPhysicalDevice physical_device, uint64_t shader_stamp,
              const char* cache_prefix = "sim_compute");

    // Writes the cache back when pipelines were added since the last save.
    void save();

    // save() + vkDestroyPipelineCache. Safe to call twice.
    void close();

    // Call after every successful vkCreateComputePipelines with handle().
    void markDirty() { m_dirty = true; }

    VkPipelineCache handle() const { return m_cache; }
    const SimulationPipelineCacheStatus& status() const { return m_status; }

private:
    VkDevice m_device = VK_NULL_HANDLE;
    VkPipelineCache m_cache = VK_NULL_HANDLE;
    uint64_t m_stamp = 0;
    bool m_dirty = false;
    SimulationPipelineCacheStatus m_status;
};

// FNV-1a over (file name, size, last write time) of each SPIR-V file under
// `shader_dir`. Missing files contribute their name only, so adding a shader
// later still changes the stamp.
uint64_t simulationShaderStamp(const std::string& shader_dir,
                               const std::vector<std::string>& spv_files);

} // namespace RayTrophiSim
