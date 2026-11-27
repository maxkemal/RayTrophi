#pragma once

#include "Backend/VulkanBackend.h"

#include <array>

namespace Backend::ParticleBillboardCompaction {

constexpr uint32_t kFirstBinding = kPulledStreamCount;
constexpr uint32_t kBindingCount = 3;
constexpr uint32_t kBlockSize = 64;
// Vulkan guarantees at least 65535 workgroups in X.
constexpr uint32_t kMaxCapacity = 65535 * kBlockSize;

struct Params {
    uint32_t capacity;
    uint32_t lookupOffset;
    uint32_t lookupCount;
    uint32_t phase;
};
static_assert(sizeof(Params) == 16);
static_assert(sizeof(VkDrawIndirectCommand) == 16);

inline uint32_t blockCount(uint32_t capacity) {
    return (capacity + kBlockSize - 1) / kBlockSize;
}

inline bool needsAllocation(uint32_t requested, uint32_t allocated, uint32_t attempted) {
    return requested > allocated && requested <= kMaxCapacity && requested != attempted;
}

inline void release(VulkanRT::VulkanDevice& device,
                    VulkanRT::BufferHandle (&buffers)[kBindingCount]) {
    for (auto& buffer : buffers) {
        if (buffer.buffer != VK_NULL_HANDLE) {
            device.destroyBuffer(buffer);
            buffer = {};
        }
    }
}

inline bool allocate(VulkanRT::VulkanDevice& device, VkDescriptorSet set, uint32_t capacity,
                     VulkanRT::BufferHandle (&buffers)[kBindingCount]) {
    const std::array<VkDeviceSize, kBindingCount> sizes{
        VkDeviceSize{capacity} * 2 * sizeof(uint32_t),
        VkDeviceSize{blockCount(capacity)} * 2 * sizeof(uint32_t),
        2 * sizeof(VkDrawIndirectCommand)
    };
    VulkanRT::BufferHandle replacement[kBindingCount]{};
    VkDescriptorBufferInfo infos[kBindingCount]{};
    VkWriteDescriptorSet writes[kBindingCount]{};
    for (uint32_t i = 0; i < kBindingCount; ++i) {
        VulkanRT::BufferCreateInfo info{};
        info.size = sizes[i];
        info.usage = VulkanRT::BufferUsage::STORAGE;
        info.location = VulkanRT::MemoryLocation::GPU_ONLY;
        if (i == 2) {
            info.usage = info.usage | VulkanRT::BufferUsage::INDIRECT;
        }
        replacement[i] = device.createBuffer(info);
        if (replacement[i].buffer == VK_NULL_HANDLE) {
            release(device, replacement);
            return false;
        }
        infos[i] = {replacement[i].buffer, 0, sizes[i]};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = kFirstBinding + i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(device.getDevice(), kBindingCount, writes, 0, nullptr);
    release(device, buffers);
    for (uint32_t i = 0; i < kBindingCount; ++i) {
        buffers[i] = replacement[i];
    }
    return true;
}

inline void barrier(VkCommandBuffer cmd, VkPipelineStageFlags sourceStage,
                    VkPipelineStageFlags targetStage, VkAccessFlags sourceAccess,
                    VkAccessFlags targetAccess) {
    VkMemoryBarrier memory{};
    memory.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memory.srcAccessMask = sourceAccess;
    memory.dstAccessMask = targetAccess;
    vkCmdPipelineBarrier(cmd, sourceStage, targetStage, 0, 1, &memory, 0, nullptr, 0, nullptr);
}

} // namespace Backend::ParticleBillboardCompaction
