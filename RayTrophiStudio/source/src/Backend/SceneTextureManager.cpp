#include "Backend/SceneTextureManager.h"

#include "globals.h"

#include <algorithm>
#include <iomanip>
#include <sstream>

namespace Backend {
namespace {

std::string textureConsumersToString(TextureConsumer consumers) {
    if (consumers == TextureConsumer::None) {
        return "None";
    }

    std::ostringstream oss;
    bool first = true;
    auto append = [&](TextureConsumer bit, const char* label) {
        if (!hasTextureConsumer(consumers, bit)) return;
        if (!first) oss << '|';
        oss << label;
        first = false;
    };

    append(TextureConsumer::RasterPreview, "RasterPreview");
    append(TextureConsumer::VulkanRT, "VulkanRT");
    append(TextureConsumer::Optix, "Optix");
    return oss.str();
}

} // namespace

bool SceneTextureManager::initialize(const RenderBackendCapabilities& capabilities, const std::string& ownerTag) {
    std::lock_guard<std::mutex> guard(m_mutex);
    const bool firstInit = !m_initialized;
    m_capabilities.hasVulkan = m_capabilities.hasVulkan || capabilities.hasVulkan;
    m_capabilities.hasMaterialPreview = m_capabilities.hasMaterialPreview || capabilities.hasMaterialPreview;
    m_capabilities.hasVulkanRT = m_capabilities.hasVulkanRT || capabilities.hasVulkanRT;
    m_capabilities.hasOptix = m_capabilities.hasOptix || capabilities.hasOptix;
    m_capabilities.hasCUDA = m_capabilities.hasCUDA || capabilities.hasCUDA;
    if (m_ownerTag.empty()) {
        m_ownerTag = ownerTag;
    } else if (!ownerTag.empty() && m_ownerTag.find(ownerTag) == std::string::npos) {
        m_ownerTag += "|" + ownerTag;
    }
    m_initialized = true;

    if (firstInit) {
        SCENE_LOG_INFO(std::string("[SceneTextureManager] Initialized for ") + m_ownerTag +
                       " | Vulkan=" + (m_capabilities.hasVulkan ? std::string("yes") : std::string("no")) +
                       " | MaterialPreview=" + (m_capabilities.hasMaterialPreview ? std::string("yes") : std::string("no")) +
                       " | VulkanRT=" + (m_capabilities.hasVulkanRT ? std::string("yes") : std::string("no")) +
                       " | OptiX=" + (m_capabilities.hasOptix ? std::string("yes") : std::string("no")));
    }
    return true;
}

void SceneTextureManager::shutdown() {
    std::lock_guard<std::mutex> guard(m_mutex);
    m_recordsByKey.clear();
    m_keysByHandle.clear();
    m_nextHandleId = 1;
    m_initialized = false;
}

bool SceneTextureManager::canServe(TextureConsumer consumer) const {
    switch (consumer) {
        case TextureConsumer::RasterPreview:
            return m_capabilities.hasMaterialPreview;
        case TextureConsumer::VulkanRT:
            return m_capabilities.hasVulkanRT;
        case TextureConsumer::Optix:
            return m_capabilities.hasOptix;
        case TextureConsumer::None:
        default:
            return false;
    }
}

TextureHandle SceneTextureManager::registerTextureKey(const std::string& key,
                                                      TextureConsumer consumer,
                                                      uint32_t width,
                                                      uint32_t height,
                                                      uint64_t estimatedBytes) {
    std::lock_guard<std::mutex> guard(m_mutex);

    auto it = m_recordsByKey.find(key);
    if (it != m_recordsByKey.end()) {
        const TextureConsumer oldConsumers = it->second.consumers;
        it->second.consumers |= consumer;
        if (width != 0) it->second.width = width;
        if (height != 0) it->second.height = height;
        if (estimatedBytes != 0) it->second.estimatedBytes = estimatedBytes;
        it->second.hasVulkanBacking = it->second.hasVulkanBacking ||
            hasTextureConsumer(consumer, TextureConsumer::RasterPreview) ||
            hasTextureConsumer(consumer, TextureConsumer::VulkanRT);
        it->second.hasOptixBacking = it->second.hasOptixBacking ||
            hasTextureConsumer(consumer, TextureConsumer::Optix);
        ++it->second.registrationCount;
        if (oldConsumers != it->second.consumers) {
            SCENE_LOG_INFO("[SceneTextureManager] Rebound existing texture key='" + key +
                           "' | owner=" + m_ownerTag +
                           " | consumers=" + textureConsumersToString(it->second.consumers));
        }
        return it->second.handle;
    }

    TextureRecord rec;
    rec.handle.id = m_nextHandleId++;
    rec.key = key;
    rec.consumers = consumer;
    rec.width = width;
    rec.height = height;
    rec.estimatedBytes = estimatedBytes;
    rec.hasVulkanBacking = hasTextureConsumer(consumer, TextureConsumer::RasterPreview) ||
                           hasTextureConsumer(consumer, TextureConsumer::VulkanRT);
    rec.hasOptixBacking = hasTextureConsumer(consumer, TextureConsumer::Optix);
    rec.registrationCount = 1;

    m_recordsByKey.emplace(key, rec);
    m_keysByHandle.emplace(rec.handle.id, key);
   /* SCENE_LOG_INFO("[SceneTextureManager] Registered texture key='" + key +
                   "' | owner=" + m_ownerTag +
                   " | handle=" + std::to_string(rec.handle.id) +
                   " | consumers=" + textureConsumersToString(rec.consumers) +
                   " | size=" + std::to_string(rec.width) + "x" + std::to_string(rec.height) +
                   " | bytes~=" + std::to_string(rec.estimatedBytes));*/
    return rec.handle;
}

bool SceneTextureManager::tryGetTexture(const std::string& key, TextureRecord& outRecord) const {
    std::lock_guard<std::mutex> guard(m_mutex);
    auto it = m_recordsByKey.find(key);
    if (it == m_recordsByKey.end()) {
        return false;
    }

    outRecord = it->second;
    return true;
}

bool SceneTextureManager::tryGetHandleForKey(const std::string& key, TextureHandle& outHandle) const {
    std::lock_guard<std::mutex> guard(m_mutex);
    auto it = m_recordsByKey.find(key);
    if (it == m_recordsByKey.end()) {
        return false;
    }

    outHandle = it->second.handle;
    return true;
}

bool SceneTextureManager::tryGetTexture(TextureHandle handle, TextureRecord& outRecord) const {
    std::lock_guard<std::mutex> guard(m_mutex);
    const auto keyIt = m_keysByHandle.find(handle.id);
    if (keyIt == m_keysByHandle.end()) {
        return false;
    }

    const auto recordIt = m_recordsByKey.find(keyIt->second);
    if (recordIt == m_recordsByKey.end()) {
        return false;
    }

    outRecord = recordIt->second;
    return true;
}

bool SceneTextureManager::tryGetVulkanTextureId(const std::string& key, int64_t& outTextureId) const {
    std::lock_guard<std::mutex> guard(m_mutex);
    auto it = m_recordsByKey.find(key);
    if (it == m_recordsByKey.end() || it->second.vulkanTextureId == 0) {
        return false;
    }

    outTextureId = it->second.vulkanTextureId;
    return true;
}

bool SceneTextureManager::tryGetVulkanTextureId(const std::string& key,
                                                const std::string& ownerTag,
                                                int64_t& outTextureId) const {
    std::lock_guard<std::mutex> guard(m_mutex);
    auto it = m_recordsByKey.find(key);
    if (it == m_recordsByKey.end()) {
        return false;
    }

    auto ownerIt = it->second.vulkanTextureIdsByOwner.find(ownerTag);
    if (ownerIt == it->second.vulkanTextureIdsByOwner.end() || ownerIt->second == 0) {
        return false;
    }

    outTextureId = ownerIt->second;
    return true;
}

bool SceneTextureManager::tryGetVulkanTextureId(TextureHandle handle, int64_t& outTextureId) const {
    std::lock_guard<std::mutex> guard(m_mutex);
    const auto keyIt = m_keysByHandle.find(handle.id);
    if (keyIt == m_keysByHandle.end()) {
        return false;
    }

    const auto recordIt = m_recordsByKey.find(keyIt->second);
    if (recordIt == m_recordsByKey.end() || recordIt->second.vulkanTextureId == 0) {
        return false;
    }

    outTextureId = recordIt->second.vulkanTextureId;
    return true;
}

bool SceneTextureManager::tryGetVulkanTextureId(TextureHandle handle,
                                                const std::string& ownerTag,
                                                int64_t& outTextureId) const {
    std::lock_guard<std::mutex> guard(m_mutex);
    const auto keyIt = m_keysByHandle.find(handle.id);
    if (keyIt == m_keysByHandle.end()) {
        return false;
    }

    const auto recordIt = m_recordsByKey.find(keyIt->second);
    if (recordIt == m_recordsByKey.end()) {
        return false;
    }

    auto ownerIt = recordIt->second.vulkanTextureIdsByOwner.find(ownerTag);
    if (ownerIt == recordIt->second.vulkanTextureIdsByOwner.end() || ownerIt->second == 0) {
        return false;
    }

    outTextureId = ownerIt->second;
    return true;
}

void SceneTextureManager::setVulkanTextureId(TextureHandle handle, int64_t textureId) {
    std::lock_guard<std::mutex> guard(m_mutex);
    const auto keyIt = m_keysByHandle.find(handle.id);
    if (keyIt == m_keysByHandle.end()) {
        return;
    }

    const auto recordIt = m_recordsByKey.find(keyIt->second);
    if (recordIt == m_recordsByKey.end()) {
        return;
    }

    recordIt->second.vulkanTextureId = textureId;
}

void SceneTextureManager::setVulkanTextureId(TextureHandle handle,
                                             const std::string& ownerTag,
                                             int64_t textureId) {
    std::lock_guard<std::mutex> guard(m_mutex);
    const auto keyIt = m_keysByHandle.find(handle.id);
    if (keyIt == m_keysByHandle.end()) {
        return;
    }

    const auto recordIt = m_recordsByKey.find(keyIt->second);
    if (recordIt == m_recordsByKey.end()) {
        return;
    }

    recordIt->second.vulkanTextureIdsByOwner[ownerTag] = textureId;
}

bool SceneTextureManager::tryGetVulkanBacking(TextureHandle handle,
                                             const std::string& ownerTag,
                                             VulkanBackingRecord& outBacking) const {
    std::lock_guard<std::mutex> guard(m_mutex);
    const auto keyIt = m_keysByHandle.find(handle.id);
    if (keyIt == m_keysByHandle.end()) {
        return false;
    }

    const auto recordIt = m_recordsByKey.find(keyIt->second);
    if (recordIt == m_recordsByKey.end()) {
        return false;
    }

    auto ownerIt = recordIt->second.vulkanBackingsByOwner.find(ownerTag);
    if (ownerIt == recordIt->second.vulkanBackingsByOwner.end() || !ownerIt->second.isValid()) {
        return false;
    }

    recordIt->second.lastAccessCounter = ++m_accessCounter;
    outBacking = ownerIt->second;
    return true;
}

bool SceneTextureManager::tryGetVulkanBackingById(const std::string& ownerTag,
                                                  int64_t textureId,
                                                  VulkanBackingRecord& outBacking) const {
    if (textureId == 0) {
        return false;
    }

    std::lock_guard<std::mutex> guard(m_mutex);
    for (const auto& [key, record] : m_recordsByKey) {
        (void)key;
        auto ownerIt = record.vulkanBackingsByOwner.find(ownerTag);
        if (ownerIt != record.vulkanBackingsByOwner.end() &&
            ownerIt->second.textureId == textureId &&
            ownerIt->second.isValid()) {
            outBacking = ownerIt->second;
            return true;
        }
    }

    return false;
}

void SceneTextureManager::setVulkanBacking(TextureHandle handle,
                                           const std::string& ownerTag,
                                           const VulkanBackingRecord& backing) {
    if (!backing.isValid()) {
        return;
    }

    std::lock_guard<std::mutex> guard(m_mutex);
    const auto keyIt = m_keysByHandle.find(handle.id);
    if (keyIt == m_keysByHandle.end()) {
        return;
    }

    const auto recordIt = m_recordsByKey.find(keyIt->second);
    if (recordIt == m_recordsByKey.end()) {
        return;
    }

    recordIt->second.vulkanTextureId = backing.textureId;
    recordIt->second.vulkanTextureIdsByOwner[ownerTag] = backing.textureId;
    recordIt->second.vulkanBackingsByOwner[ownerTag] = backing;
}

bool SceneTextureManager::resolveOrCreateVulkanBacking(
    const std::string& key,
    const std::string& ownerTag,
    TextureConsumer consumers,
    uint32_t width,
    uint32_t height,
    uint64_t estimatedBytes,
    const std::function<bool(TextureHandle, VulkanBackingRecord&)>& createBacking,
    VulkanBackingRecord& outBacking,
    bool* outCreated) {
    if (outCreated) {
        *outCreated = false;
    }
    outBacking = VulkanBackingRecord{};
    if (key.empty() || ownerTag.empty()) {
        return false;
    }

    TextureHandle handle = registerTextureKey(key, consumers, width, height, estimatedBytes);
    if (!handle.isValid()) {
        return false;
    }

    if (tryGetVulkanBacking(handle, ownerTag, outBacking)) {
        return true;
    }

    if (!createBacking) {
        return false;
    }

    VulkanBackingRecord createdBacking{};
    if (!createBacking(handle, createdBacking) || !createdBacking.isValid()) {
        return false;
    }

   /* SCENE_LOG_INFO("[SceneTextureManager] Created Vulkan backing key='" + key +
                   "' | owner=" + ownerTag +
                   " | id=" + std::to_string(createdBacking.textureId) +
                   " | " + std::to_string(createdBacking.width) + "x" + std::to_string(createdBacking.height));*/
    setVulkanBacking(handle, ownerTag, createdBacking);
    outBacking = createdBacking;
    if (outCreated) {
        *outCreated = true;
    }
    return true;
}

void SceneTextureManager::clearVulkanTextureId(int64_t textureId) {
    if (textureId == 0) {
        return;
    }

    std::lock_guard<std::mutex> guard(m_mutex);
    for (auto& [key, record] : m_recordsByKey) {
        (void)key;
        if (record.vulkanTextureId == textureId) {
            record.vulkanTextureId = 0;
        }
    }
}

void SceneTextureManager::clearVulkanTextureId(const std::string& ownerTag, int64_t textureId) {
    if (textureId == 0) {
        return;
    }

    std::lock_guard<std::mutex> guard(m_mutex);
    for (auto& [key, record] : m_recordsByKey) {
        (void)key;
        if (record.vulkanTextureId == textureId) {
            record.vulkanTextureId = 0;
        }

        auto ownerIt = record.vulkanTextureIdsByOwner.find(ownerTag);
        if (ownerIt != record.vulkanTextureIdsByOwner.end() && ownerIt->second == textureId) {
            ownerIt->second = 0;
        }
    }
}

void SceneTextureManager::clearVulkanBacking(const std::string& ownerTag, int64_t textureId) {
    if (textureId == 0) {
        return;
    }

    std::lock_guard<std::mutex> guard(m_mutex);
    for (auto& [key, record] : m_recordsByKey) {
        (void)key;
        if (record.vulkanTextureId == textureId) {
            record.vulkanTextureId = 0;
        }

        auto ownerIdIt = record.vulkanTextureIdsByOwner.find(ownerTag);
        if (ownerIdIt != record.vulkanTextureIdsByOwner.end() && ownerIdIt->second == textureId) {
            ownerIdIt->second = 0;
        }

        auto ownerBackingIt = record.vulkanBackingsByOwner.find(ownerTag);
        if (ownerBackingIt != record.vulkanBackingsByOwner.end() &&
            ownerBackingIt->second.textureId == textureId) {
            ownerBackingIt->second = VulkanBackingRecord{};
        }
    }
}

void SceneTextureManager::clearAllVulkanBackingForOwner(const std::string& ownerTag) {
    if (ownerTag.empty()) {
        return;
    }

    std::lock_guard<std::mutex> guard(m_mutex);
    for (auto& [key, record] : m_recordsByKey) {
        (void)key;
        auto ownerIdIt = record.vulkanTextureIdsByOwner.find(ownerTag);
        if (ownerIdIt != record.vulkanTextureIdsByOwner.end()) {
            if (record.vulkanTextureId == ownerIdIt->second) {
                record.vulkanTextureId = 0;
            }
            ownerIdIt->second = 0;
        }

        auto ownerBackingIt = record.vulkanBackingsByOwner.find(ownerTag);
        if (ownerBackingIt != record.vulkanBackingsByOwner.end()) {
            ownerBackingIt->second = VulkanBackingRecord{};
        }
    }
}

bool SceneTextureManager::destroyAndClearVulkanBacking(const std::string& ownerTag,
                                                       int64_t textureId) {
    if (ownerTag.empty() || textureId == 0) {
        return false;
    }

    std::function<void()> destroyFn;
    bool found = false;

    {
        std::lock_guard<std::mutex> guard(m_mutex);
        for (auto& [key, record] : m_recordsByKey) {
            (void)key;
            auto ownerBackingIt = record.vulkanBackingsByOwner.find(ownerTag);
            if (ownerBackingIt == record.vulkanBackingsByOwner.end() ||
                ownerBackingIt->second.textureId != textureId) {
                continue;
            }

            found = true;
            if (ownerBackingIt->second.destroyFn) {
                destroyFn = std::move(ownerBackingIt->second.destroyFn);
            }
            ownerBackingIt->second = VulkanBackingRecord{};

            auto ownerIdIt = record.vulkanTextureIdsByOwner.find(ownerTag);
            if (ownerIdIt != record.vulkanTextureIdsByOwner.end() &&
                ownerIdIt->second == textureId) {
                ownerIdIt->second = 0;
            }
            if (record.vulkanTextureId == textureId) {
                record.vulkanTextureId = 0;
            }
            break;
        }
    }

    if (destroyFn) {
        destroyFn();
    }
    return found;
}

void SceneTextureManager::destroyAllVulkanBackingForOwner(const std::string& ownerTag) {
    if (ownerTag.empty()) {
        return;
    }

    std::vector<std::function<void()>> pendingDestroy;

    {
        std::lock_guard<std::mutex> guard(m_mutex);
        for (auto& [key, record] : m_recordsByKey) {
            (void)key;
            auto ownerIdIt = record.vulkanTextureIdsByOwner.find(ownerTag);
            if (ownerIdIt != record.vulkanTextureIdsByOwner.end()) {
                if (record.vulkanTextureId == ownerIdIt->second) {
                    record.vulkanTextureId = 0;
                }
                ownerIdIt->second = 0;
            }

            auto ownerBackingIt = record.vulkanBackingsByOwner.find(ownerTag);
            if (ownerBackingIt != record.vulkanBackingsByOwner.end()) {
                if (ownerBackingIt->second.destroyFn) {
                    pendingDestroy.push_back(std::move(ownerBackingIt->second.destroyFn));
                }
                ownerBackingIt->second = VulkanBackingRecord{};
            }
        }
    }

    for (auto& fn : pendingDestroy) {
        fn();
    }
}

bool SceneTextureManager::tryGetOptixTextureId(const std::string& key, int64_t& outTextureId) const {
    std::lock_guard<std::mutex> guard(m_mutex);
    auto it = m_recordsByKey.find(key);
    if (it == m_recordsByKey.end() || it->second.optixTextureId == 0) {
        return false;
    }

    outTextureId = it->second.optixTextureId;
    return true;
}

bool SceneTextureManager::tryGetOptixTextureId(TextureHandle handle, int64_t& outTextureId) const {
    std::lock_guard<std::mutex> guard(m_mutex);
    const auto keyIt = m_keysByHandle.find(handle.id);
    if (keyIt == m_keysByHandle.end()) {
        return false;
    }

    const auto recordIt = m_recordsByKey.find(keyIt->second);
    if (recordIt == m_recordsByKey.end() || recordIt->second.optixTextureId == 0) {
        return false;
    }

    outTextureId = recordIt->second.optixTextureId;
    return true;
}

void SceneTextureManager::setOptixTextureId(TextureHandle handle, int64_t textureId) {
    std::lock_guard<std::mutex> guard(m_mutex);
    const auto keyIt = m_keysByHandle.find(handle.id);
    if (keyIt == m_keysByHandle.end()) {
        return;
    }

    const auto recordIt = m_recordsByKey.find(keyIt->second);
    if (recordIt == m_recordsByKey.end()) {
        return;
    }

    recordIt->second.optixTextureId = textureId;
}

bool SceneTextureManager::resolveOrCreateOptixTextureId(
    const std::string& key,
    TextureConsumer consumers,
    uint32_t width,
    uint32_t height,
    uint64_t estimatedBytes,
    const std::function<int64_t(TextureHandle)>& createTexture,
    int64_t& outTextureId,
    bool* outCreated) {
    if (outCreated) {
        *outCreated = false;
    }
    outTextureId = 0;
    if (key.empty()) {
        return false;
    }

    // Fast path: single lock — avoids registerTextureKey overhead on every per-frame lookup
    {
        std::lock_guard<std::mutex> guard(m_mutex);
        auto it = m_recordsByKey.find(key);
        if (it != m_recordsByKey.end() && it->second.optixTextureId != 0) {
            outTextureId = it->second.optixTextureId;
            return true;
        }
    }

    // Slow path: first-time registration + creation
    TextureHandle handle = registerTextureKey(key, consumers, width, height, estimatedBytes);
    if (!handle.isValid()) {
        return false;
    }

    if (tryGetOptixTextureId(handle, outTextureId)) {
        return true;
    }

    if (!createTexture) {
        return false;
    }

    const int64_t createdTextureId = createTexture(handle);
    if (createdTextureId == 0) {
        return false;
    }

    /*SCENE_LOG_INFO("[SceneTextureManager] Created OptiX texture key='" + key +
                   "' | id=" + std::to_string(createdTextureId));*/
    setOptixTextureId(handle, createdTextureId);
    outTextureId = createdTextureId;
    if (outCreated) {
        *outCreated = true;
    }
    return true;
}

void SceneTextureManager::clearOptixTextureId(int64_t textureId) {
    if (textureId == 0) {
        return;
    }

    std::lock_guard<std::mutex> guard(m_mutex);
    for (auto& [key, record] : m_recordsByKey) {
        (void)key;
        if (record.optixTextureId == textureId) {
            record.optixTextureId = 0;
        }
    }
}

void SceneTextureManager::clearAllOptixTextureIds() {
    std::lock_guard<std::mutex> guard(m_mutex);
    for (auto& [key, record] : m_recordsByKey) {
        (void)key;
        record.optixTextureId = 0;
        record.hasOptixBacking = false;
    }
}

size_t SceneTextureManager::textureCount() const {
    std::lock_guard<std::mutex> guard(m_mutex);
    return m_recordsByKey.size();
}

uint64_t SceneTextureManager::totalEstimatedTextureBytes() const {
    std::lock_guard<std::mutex> guard(m_mutex);
    uint64_t total = 0;
    for (const auto& [key, record] : m_recordsByKey) {
        (void)key;
        total += record.estimatedBytes;
    }
    return total;
}

uint64_t SceneTextureManager::totalResidentTextureBytes() const {
    std::lock_guard<std::mutex> guard(m_mutex);
    uint64_t total = 0;
    for (const auto& [key, record] : m_recordsByKey) {
        (void)key;
        for (const auto& [owner, backing] : record.vulkanBackingsByOwner) {
            (void)owner;
            if (backing.isValid()) {
                total += (backing.allocatedBytes > 0) ? backing.allocatedBytes : record.estimatedBytes;
            }
        }
        if (record.optixTextureId != 0) {
            total += record.estimatedBytes;
        }
    }
    return total;
}

uint64_t SceneTextureManager::estimatedTextureBytesForOwner(const std::string& ownerTag) const {
    if (ownerTag.empty()) return 0;
    std::lock_guard<std::mutex> guard(m_mutex);
    uint64_t total = 0;
    for (const auto& [key, record] : m_recordsByKey) {
        (void)key;
        auto it = record.vulkanBackingsByOwner.find(ownerTag);
        if (it != record.vulkanBackingsByOwner.end() && it->second.isValid()) {
            total += (it->second.allocatedBytes > 0) ? it->second.allocatedBytes : record.estimatedBytes;
        }
    }
    return total;
}

uint64_t SceneTextureManager::estimatedOptixTextureBytes() const {
    std::lock_guard<std::mutex> guard(m_mutex);
    uint64_t total = 0;
    for (const auto& [key, record] : m_recordsByKey) {
        (void)key;
        if (record.optixTextureId != 0) {
            total += record.estimatedBytes;
        }
    }
    return total;
}

size_t SceneTextureManager::trimVulkanBackingLRU(const std::string& ownerTag, uint64_t targetBytes) {
    if (ownerTag.empty()) return 0;

    struct Candidate {
        std::string key;
        uint64_t lastAccess;
        uint64_t bytes;
        std::function<void()> destroyFn;
    };

    std::vector<std::function<void()>> toDestroy;
    size_t evictedCount = 0;

    {
        std::lock_guard<std::mutex> guard(m_mutex);

        // Collect trimable records: have a valid backing and a destroyFn for this owner.
        std::vector<Candidate> candidates;
        for (auto& [key, record] : m_recordsByKey) {
            auto backingIt = record.vulkanBackingsByOwner.find(ownerTag);
            if (backingIt == record.vulkanBackingsByOwner.end() ||
                !backingIt->second.isValid() ||
                !backingIt->second.destroyFn) {
                continue;
            }
            uint64_t bytes = (backingIt->second.allocatedBytes > 0) ? backingIt->second.allocatedBytes : record.estimatedBytes;
            candidates.push_back({key, record.lastAccessCounter,
                                  bytes, {}});
        }

        // Sort ascending by lastAccessCounter — oldest (LRU) first.
        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate& a, const Candidate& b) {
                      return a.lastAccess < b.lastAccess;
                  });

        // Compute current backing bytes for this owner.
        uint64_t currentBytes = 0;
        for (const auto& [key, record] : m_recordsByKey) {
            auto it = record.vulkanBackingsByOwner.find(ownerTag);
            if (it != record.vulkanBackingsByOwner.end() && it->second.isValid()) {
                currentBytes += (it->second.allocatedBytes > 0) ? it->second.allocatedBytes : record.estimatedBytes;
            }
        }

        // Evict until under targetBytes.
        for (auto& cand : candidates) {
            if (currentBytes <= targetBytes) break;
            auto recIt = m_recordsByKey.find(cand.key);
            if (recIt == m_recordsByKey.end()) continue;
            auto backingIt = recIt->second.vulkanBackingsByOwner.find(ownerTag);
            if (backingIt == recIt->second.vulkanBackingsByOwner.end() ||
                !backingIt->second.isValid()) {
                continue;
            }
            toDestroy.push_back(std::move(backingIt->second.destroyFn));
            backingIt->second = VulkanBackingRecord{};
            auto idIt = recIt->second.vulkanTextureIdsByOwner.find(ownerTag);
            if (idIt != recIt->second.vulkanTextureIdsByOwner.end()) {
                if (recIt->second.vulkanTextureId == idIt->second) {
                    recIt->second.vulkanTextureId = 0;
                }
                idIt->second = 0;
            }
            currentBytes -= cand.bytes;
            ++evictedCount;
        }
    }

    for (auto& fn : toDestroy) {
        fn();
    }

    if (evictedCount > 0) {
        SCENE_LOG_INFO("[SceneTextureManager] LRU trim | owner=" + ownerTag +
                       " | evicted=" + std::to_string(evictedCount) +
                       " | targetBytes=" + std::to_string(targetBytes / (1024 * 1024)) + " MB");
    }
    return evictedCount;
}

void SceneTextureManager::logBudgetSummary(const std::string& context) const {
    std::lock_guard<std::mutex> guard(m_mutex);

    uint64_t totalBytes = 0;
    uint64_t residentBytes = 0;
    size_t   totalCount = m_recordsByKey.size();
    size_t   residentRecordCount = 0;
    std::unordered_map<std::string, uint64_t> bytesByOwner;
    std::unordered_map<std::string, uint32_t> countByOwner;

    for (const auto& [key, record] : m_recordsByKey) {
        (void)key;
        totalBytes += record.estimatedBytes;
        bool anyBacking = false;
        for (const auto& [owner, backing] : record.vulkanBackingsByOwner) {
            if (backing.isValid()) {
                uint64_t b = (backing.allocatedBytes > 0) ? backing.allocatedBytes : record.estimatedBytes;
                bytesByOwner[owner] += b;
                countByOwner[owner]++;
                residentBytes += b;
                anyBacking = true;
            }
        }
        if (record.optixTextureId != 0) {
            bytesByOwner["optix"] += record.estimatedBytes;
            countByOwner["optix"]++;
            residentBytes += record.estimatedBytes;
            anyBacking = true;
        }
        if (anyBacking) ++residentRecordCount;
    }

    const auto mbStr = [](uint64_t b) -> std::string {
        std::ostringstream oss;
        oss.precision(1);
        oss << std::fixed << (b / (1024.0 * 1024.0)) << " MB";
        return oss.str();
    };

    std::string prefix = context.empty() ? "[SceneTextureManager]" : "[SceneTextureManager:" + context + "]";
    SCENE_LOG_INFO(prefix + " Budget summary | total=" +
                   std::to_string(totalCount) + " textures (~" + mbStr(totalBytes) +
                   ") | resident=" + std::to_string(residentRecordCount) +
                   " (~" + mbStr(residentBytes) + " across all owners)");
    for (const auto& [owner, bytes] : bytesByOwner) {
        SCENE_LOG_INFO(prefix + "  owner=" + owner +
                       " | count=" + std::to_string(countByOwner[owner]) +
                       " | ~" + mbStr(bytes));
    }
}

} // namespace Backend
