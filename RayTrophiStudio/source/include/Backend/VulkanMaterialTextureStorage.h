#pragma once

// Single storage for material textures: every 2D texture lives in one layer of a
// 2D-array "page" (same format, extent and mip count). Bindless consumers (RT
// binding 6, IBL, shadow, matcap) keep using ImageHandle::view, which becomes a
// 2D view of that layer; the viewport samples the page's array view. One copy of
// the texels, so wet paint written into the layer is seen by both.
//
// Pages never grow (growing would invalidate views that are copied by value all
// over the backend). Instead compact() repacks every 2D texture into exactly
// sized pages at a quiet point (end of a texture batch / scene load), swaps the
// handles in place and destroys the previous storage after the device is idle.
// A texture uploaded after the last compaction stays standalone until the next
// one; refFor() reports it as kNoTexture so the viewport shows it as missing
// rather than reading the wrong layer.

#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

#include "Backend/VulkanMaterialTextureArrayImage.h"

namespace VulkanRT {
class VulkanDevice;
struct ImageHandle;
}

namespace Backend::MaterialTextureStorage {

struct Page {
    MaterialTexturePages::ArrayImage image;
    VkSampler sampler = VK_NULL_HANDLE;   // owned by the page
    uint32_t usedLayers = 0;
};

struct CompactionReport {
    VkResult result = VK_SUCCESS;
    uint32_t texturesPacked = 0;
    uint32_t texturesSkipped = 0;   // 3D, unsupported format, or over a device limit
    uint32_t pages = 0;
    uint32_t pagesOverViewportBudget = 0;
    uint64_t allocatedBytes = 0;
    uint64_t payloadBytes = 0;
};

// Creates the page sampler for a given mip count (the adapter owns the policy:
// address mode, anisotropy, maxLod). Must match what per-texture samplers use.
using SamplerFactory = std::function<VkSampler(uint32_t mipLevels)>;

class Storage {
public:
    Storage() = default;
    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;

    // Repacks all 2D textures in `images`. Synchronous: submits one command
    // buffer, waits for the device, then destroys the previous standalone images
    // and pages. On failure nothing in `images` changes. `keepStandalone(id)`
    // excludes textures bound outside the material slots (world environment).
    CompactionReport compact(VulkanRT::VulkanDevice& device,
                             std::unordered_map<int64_t, VulkanRT::ImageHandle>& images,
                             const SamplerFactory& makeSampler,
                             const std::function<bool(int64_t)>& keepStandalone = {});

    // Destroys every page. Call only after all handles that alias pages were
    // destroyed (their views/samplers) and the device no longer reads them.
    void destroyAll(VulkanRT::VulkanDevice& device);

    // Packed viewport reference (page << 16 | layer) for a handle, or kNoTexture
    // when it is standalone or its page is beyond the viewport page budget.
    uint32_t refFor(const VulkanRT::ImageHandle& handle) const;

    // ── Viewport descriptor resources (32 sampler2DArray pages + slot->ref table) ──
    // Creates the dummy page (bound to unused page slots) and a one-entry ref
    // table on first use. Returns false when the device cannot provide them.
    bool ensureDescriptorResources(VulkanRT::VulkanDevice& device);
    // Rebuilds the slot->ref table (index = material texture slot id). Call after
    // compaction with the device idle: the previous table buffer is destroyed.
    void rebuildRefTable(VulkanRT::VulkanDevice& device,
                         const std::unordered_map<int64_t, VulkanRT::ImageHandle>& images);
    // Fills all 32 page descriptors: live pages inside the budget, dummy elsewhere.
    void pageDescriptorImages(std::array<VkDescriptorImageInfo, 32>& out) const;
    VkDescriptorBufferInfo refTableDescriptor() const;
    bool descriptorResourcesReady() const { return m_dummy.view && m_dummySampler && m_refTable; }

    const std::vector<Page>& pages() const { return m_pages; }
    uint64_t generation() const { return m_generation; }
    const CompactionReport& lastReport() const { return m_last; }

private:
    std::vector<Page> m_pages;
    MaterialTexturePages::ArrayImage m_dummy;
    VkSampler m_dummySampler = VK_NULL_HANDLE;
    VkBuffer m_refTable = VK_NULL_HANDLE;
    VkDeviceMemory m_refTableMemory = VK_NULL_HANDLE;
    uint64_t m_refTableBytes = 0;
    void destroyDescriptorResources(VulkanRT::VulkanDevice& device);
    void destroyRefTable(VulkanRT::VulkanDevice& device);
    uint64_t m_generation = 0;
    CompactionReport m_last;
};

} // namespace Backend::MaterialTextureStorage
