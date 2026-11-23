#pragma once

#include "VulkanMaterialTexturePages.h"

#include <vulkan/vulkan.h>

namespace Backend::MaterialTexturePages {

struct ImageProfile {
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageUsageFlags usage = 0;
    VkImageCreateFlags flags = 0;
};

struct ImageProfileResult {
    ImageProfile profile;
    VkResult result;
    VkImageFormatProperties properties;
};

struct ImageLimits {
    uint32_t maxDimension2D = 0;
    uint32_t maxLayers = 0;
    std::vector<ImageProfileResult> profiles;

    uint32_t layersFor(const Key& key) const {
        if (key.compatibility >= profiles.size()) {
            return 0;
        }
        const ImageProfileResult& entry = profiles[key.compatibility];
        const VkImageFormatProperties& image = entry.properties;
        if (entry.result != VK_SUCCESS || !(image.sampleCounts & VK_SAMPLE_COUNT_1_BIT)) {
            return 0;
        }
        if (!key.width || !key.height || !key.mipLevels ||
            key.width > maxDimension2D || key.height > maxDimension2D ||
            key.width > image.maxExtent.width || key.height > image.maxExtent.height ||
            key.mipLevels > image.maxMipLevels) {
            return 0;
        }
        uint32_t extent = (std::max)(key.width, key.height);
        uint32_t geometricMipLevels = 1;
        while (extent > 1) {
            extent >>= 1;
            ++geometricMipLevels;
        }
        if (key.mipLevels > geometricMipLevels) {
            return 0;
        }
        return layerLimit(maxLayers, image.maxArrayLayers);
    }
};

// The order of profiles defines the compatibility IDs used by the planner.
inline ImageLimits queryImageLimits(
    VkPhysicalDevice device,
    const VkPhysicalDeviceLimits& deviceLimits,
    const std::vector<ImageProfile>& profiles,
    PFN_vkGetPhysicalDeviceImageFormatProperties query = vkGetPhysicalDeviceImageFormatProperties) {
    ImageLimits limits;
    limits.maxDimension2D = deviceLimits.maxImageDimension2D;
    limits.maxLayers = deviceLimits.maxImageArrayLayers;
    limits.profiles.reserve(profiles.size());
    for (const ImageProfile& profile : profiles) {
        VkImageFormatProperties properties{};
        const VkResult result = query(device, profile.format, VK_IMAGE_TYPE_2D,
                                      VK_IMAGE_TILING_OPTIMAL, profile.usage,
                                      profile.flags, &properties);
        // Output properties are undefined on error. Keep the call arguments and result.
        limits.profiles.push_back({profile, result, result == VK_SUCCESS
                                                       ? properties
                                                       : VkImageFormatProperties{}});
    }
    return limits;
}

template <typename PayloadBytes>
Plan makeVulkanPlan(const std::vector<Texture>& textures,
                    const ImageLimits& limits,
                    PayloadBytes payloadBytes) {
    return makePlan(textures, limits.maxLayers, [&](const Key& key) {
        const uint32_t layers = limits.layersFor(key);
        if (!layers) {
            return 0u;
        }
        const uint64_t layerBytes = payloadBytes(key);
        const auto& image = limits.profiles[key.compatibility].properties;
        if (!layerBytes) {
            return 0u;
        }
        // Payload is a lower bound. The actual allocation can still be larger.
        return static_cast<uint32_t>((std::min)(
            static_cast<uint64_t>(layers), image.maxResourceSize / layerBytes));
    }, payloadBytes);
}

} // namespace Backend::MaterialTexturePages
