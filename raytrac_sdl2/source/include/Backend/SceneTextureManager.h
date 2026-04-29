#ifndef BACKEND_SCENE_TEXTURE_MANAGER_H
#define BACKEND_SCENE_TEXTURE_MANAGER_H

#include "Backend/RenderCapabilities.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>

namespace Backend {

struct TextureHandle {
    uint32_t id = 0;

    bool isValid() const { return id != 0; }
};

struct VulkanBackingRecord {
    int64_t textureId = 0;
    uint64_t image = 0;
    uint64_t view = 0;
    uint64_t memory = 0;
    uint64_t sampler = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t format = 0;
    // Optional lifecycle callback: called by destroyAndClearVulkanBacking /
    // destroyAllVulkanBackingForOwner to destroy the physical Vulkan objects.
    // Must be safe to call once; cleared immediately after invocation.
    std::function<void()> destroyFn;

    bool isValid() const {
        return textureId != 0 && image != 0 && view != 0;
    }
};

struct TextureRecord {
    TextureHandle handle;
    std::string key;
    TextureConsumer consumers = TextureConsumer::None;
    uint32_t width = 0;
    uint32_t height = 0;
    uint64_t estimatedBytes = 0;
    bool hasVulkanBacking = false;
    bool hasOptixBacking = false;
    int64_t vulkanTextureId = 0;
    int64_t optixTextureId = 0;
    std::unordered_map<std::string, int64_t> vulkanTextureIdsByOwner;
    std::unordered_map<std::string, VulkanBackingRecord> vulkanBackingsByOwner;
    uint32_t registrationCount = 0;
    mutable uint64_t lastAccessCounter = 0; // updated on every successful resolve/get — used for LRU trim
};

class SceneTextureManager {
public:
    SceneTextureManager() = default;

    bool initialize(const RenderBackendCapabilities& capabilities, const std::string& ownerTag);
    void shutdown();

    const RenderBackendCapabilities& capabilities() const { return m_capabilities; }
    bool isInitialized() const { return m_initialized; }
    bool canServe(TextureConsumer consumer) const;

    TextureHandle registerTextureKey(const std::string& key,
                                     TextureConsumer consumer,
                                     uint32_t width = 0,
                                     uint32_t height = 0,
                                     uint64_t estimatedBytes = 0);

    bool tryGetHandleForKey(const std::string& key, TextureHandle& outHandle) const;
    bool tryGetTexture(const std::string& key, TextureRecord& outRecord) const;
    bool tryGetTexture(TextureHandle handle, TextureRecord& outRecord) const;
    bool tryGetVulkanTextureId(const std::string& key, int64_t& outTextureId) const;
    bool tryGetVulkanTextureId(TextureHandle handle, int64_t& outTextureId) const;
    void setVulkanTextureId(TextureHandle handle, int64_t textureId);
    void clearVulkanTextureId(int64_t textureId);
    bool tryGetVulkanTextureId(const std::string& key, const std::string& ownerTag, int64_t& outTextureId) const;
    bool tryGetVulkanTextureId(TextureHandle handle, const std::string& ownerTag, int64_t& outTextureId) const;
    void setVulkanTextureId(TextureHandle handle, const std::string& ownerTag, int64_t textureId);
    void clearVulkanTextureId(const std::string& ownerTag, int64_t textureId);
    bool tryGetVulkanBacking(TextureHandle handle, const std::string& ownerTag, VulkanBackingRecord& outBacking) const;
    bool tryGetVulkanBackingById(const std::string& ownerTag, int64_t textureId, VulkanBackingRecord& outBacking) const;
    void setVulkanBacking(TextureHandle handle, const std::string& ownerTag, const VulkanBackingRecord& backing);
    bool resolveOrCreateVulkanBacking(const std::string& key,
                                      const std::string& ownerTag,
                                      TextureConsumer consumers,
                                      uint32_t width,
                                      uint32_t height,
                                      uint64_t estimatedBytes,
                                      const std::function<bool(TextureHandle, VulkanBackingRecord&)>& createBacking,
                                      VulkanBackingRecord& outBacking,
                                      bool* outCreated = nullptr);
    void clearVulkanBacking(const std::string& ownerTag, int64_t textureId);
    void clearAllVulkanBackingForOwner(const std::string& ownerTag);
    // Destroy-and-clear: invokes destroyFn (if set) then clears the backing record.
    // Returns true if the record was found and processed (regardless of destroyFn presence).
    bool destroyAndClearVulkanBacking(const std::string& ownerTag, int64_t textureId);
    // Destroy-all: invokes destroyFn for every backing record owned by ownerTag, then clears them.
    // Must be called while the Vulkan device is still alive (before device shutdown).
    void destroyAllVulkanBackingForOwner(const std::string& ownerTag);
    bool tryGetOptixTextureId(const std::string& key, int64_t& outTextureId) const;
    bool tryGetOptixTextureId(TextureHandle handle, int64_t& outTextureId) const;
    void setOptixTextureId(TextureHandle handle, int64_t textureId);
    bool resolveOrCreateOptixTextureId(const std::string& key,
                                       TextureConsumer consumers,
                                       uint32_t width,
                                       uint32_t height,
                                       uint64_t estimatedBytes,
                                       const std::function<int64_t(TextureHandle)>& createTexture,
                                       int64_t& outTextureId,
                                       bool* outCreated = nullptr);
    void clearOptixTextureId(int64_t textureId);
    size_t textureCount() const;
    uint64_t totalEstimatedTextureBytes() const;
    uint64_t estimatedTextureBytesForOwner(const std::string& ownerTag) const;
    void logBudgetSummary(const std::string& context = {}) const;
    // Evict LRU Vulkan backings for ownerTag until estimatedTextureBytesForOwner(ownerTag) <= targetBytes.
    // Calls destroyFn for each evicted backing (removes from backend local maps + destroys VkImage).
    // Returns the number of textures evicted.
    size_t trimVulkanBackingLRU(const std::string& ownerTag, uint64_t targetBytes);

private:
    mutable std::mutex m_mutex;
    RenderBackendCapabilities m_capabilities;
    std::string m_ownerTag;
    std::unordered_map<std::string, TextureRecord> m_recordsByKey;
    std::unordered_map<uint32_t, std::string> m_keysByHandle;
    uint32_t m_nextHandleId = 1;
    mutable uint64_t m_accessCounter = 0; // monotonically increasing; stamped on every successful resolve
    bool m_initialized = false;
};

} // namespace Backend

#endif // BACKEND_SCENE_TEXTURE_MANAGER_H
