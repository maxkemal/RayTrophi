#pragma once

#include "VulkanMaterialTextureImageLimits.h"

namespace Backend::MaterialTexturePages {

struct ArrayImage {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize allocatedBytes = 0;
    Key key;
    uint32_t layers = 0;
};

// The caller retires this allocation after its last transfer and reader complete.
template <typename Device>
void destroyArrayImage(Device& owner, ArrayImage& image) {
    const VkDevice device = owner.getDevice();
    if (image.view) {
        vkDestroyImageView(device, image.view, nullptr);
    }
    if (image.image) {
        vkDestroyImage(device, image.image, nullptr);
    }
    if (image.memory) {
        owner.freeTrackedMemory(image.memory);
    }
    image = {};
}

// Records initialization in the caller's upload batch. Publish the view only
// after initialization and its initial uploads are ordered before sampling.
template <typename Device, typename Category>
VkResult allocateArrayImage(Device& owner, const ImageLimits& limits, const Key& key,
                            uint32_t layers, Category category, VkCommandBuffer command,
                            VkPipelineStageFlags samplingStages, ArrayImage& output) {
    if (!command || !samplingStages || output.image || output.memory || output.view) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    if (!layers || layers > limits.layersFor(key)) {
        return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }
    const ImageProfileResult& profile = limits.profiles[key.compatibility];
    const VkImageUsageFlags required = VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                       VK_IMAGE_USAGE_SAMPLED_BIT;
    if ((profile.profile.usage & required) != required) {
        return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }

    const VkDevice device = owner.getDevice();
    ArrayImage image;
    image.key = key;
    image.layers = layers;
    VkImageCreateInfo create{};
    create.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    create.flags = profile.profile.flags;
    create.imageType = VK_IMAGE_TYPE_2D;
    create.format = profile.profile.format;
    create.extent = {key.width, key.height, 1};
    create.mipLevels = key.mipLevels;
    create.arrayLayers = layers;
    create.samples = VK_SAMPLE_COUNT_1_BIT;
    create.tiling = VK_IMAGE_TILING_OPTIMAL;
    create.usage = profile.profile.usage;
    create.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    create.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkResult result = vkCreateImage(device, &create, nullptr, &image.image);
    if (result != VK_SUCCESS) {
        return result;
    }

    VkMemoryRequirements memory{};
    vkGetImageMemoryRequirements(device, image.image, &memory);
    const uint32_t memoryType = owner.findMemoryType(
        memory.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (memoryType == UINT32_MAX || memory.size > profile.properties.maxResourceSize) {
        destroyArrayImage(owner, image);
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }

    VkMemoryAllocateInfo allocation{};
    allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation.allocationSize = memory.size;
    allocation.memoryTypeIndex = memoryType;
    result = vkAllocateMemory(device, &allocation, nullptr, &image.memory);
    if (result != VK_SUCCESS) {
        image.memory = VK_NULL_HANDLE;
        destroyArrayImage(owner, image);
        return result;
    }
    owner.noteMemoryAllocated(image.memory, memory.size, memoryType, category);
    image.allocatedBytes = memory.size;
    result = vkBindImageMemory(device, image.image, image.memory, 0);
    if (result != VK_SUCCESS) {
        destroyArrayImage(owner, image);
        return result;
    }

    VkImageViewCreateInfo view{};
    view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view.image = image.image;
    view.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    view.format = create.format;
    view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, key.mipLevels, 0, layers};
    result = vkCreateImageView(device, &view, nullptr, &image.view);
    if (result != VK_SUCCESS) {
        image.view = VK_NULL_HANDLE;
        destroyArrayImage(owner, image);
        return result;
    }

    VkImageMemoryBarrier initialize{};
    initialize.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    initialize.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    initialize.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    initialize.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    initialize.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    initialize.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    initialize.image = image.image;
    initialize.subresourceRange = view.subresourceRange;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, samplingStages,
                         0, 0, nullptr, 0, nullptr, 1, &initialize);
    output = image;
    return VK_SUCCESS;
}

// Records a complete mip chain for a new, unpublished layer on the transfer-capable
// graphics/compute queue. For an edit, the caller reserves a different layer and
// retires the old layer only after the last reader's fence. The staging packer
// supplies mip offsets and the selected format's texel-block size in bytes.
// The caller keeps staging alive until submission completion.
inline bool recordArrayLayerUpload(VkCommandBuffer command, VkBuffer staging,
                                   const ArrayImage& image, uint32_t layer,
                                   const std::vector<VkDeviceSize>& mipOffsets,
                                   VkDeviceSize texelBlockBytes,
                                   VkPipelineStageFlags samplingStages) {
    if (!command || !staging || !image.image || !samplingStages || !texelBlockBytes ||
        layer >= image.layers || mipOffsets.size() != image.key.mipLevels ||
        mipOffsets.empty()) {
        return false;
    }

    std::vector<VkBufferImageCopy> regions;
    regions.reserve(mipOffsets.size());
    uint32_t width = image.key.width;
    uint32_t height = image.key.height;
    for (uint32_t mip = 0; mip < image.key.mipLevels; ++mip) {
        if (mipOffsets[mip] % texelBlockBytes != 0) {
            return false;
        }
        VkBufferImageCopy copy{};
        copy.bufferOffset = mipOffsets[mip];
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, layer, 1};
        copy.imageExtent = {width, height, 1};
        regions.push_back(copy);
        width = (std::max)(1u, width / 2);
        height = (std::max)(1u, height / 2);
    }

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image.image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, image.key.mipLevels, layer, 1};
    vkCmdPipelineBarrier(command, samplingStages, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &barrier);
    vkCmdCopyBufferToImage(command, staging, image.image, barrier.newLayout,
                           static_cast<uint32_t>(regions.size()), regions.data());
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, samplingStages,
                         0, 0, nullptr, 0, nullptr, 1, &barrier);
    return true;
}

} // namespace Backend::MaterialTexturePages
