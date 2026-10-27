#include "Backend/VulkanRawGuard.h"
#include "Backend/VulkanBackend.h"

#include <vector>

namespace VulkanRT {
namespace {
void barrier(VkCommandBuffer cmd, VkPipelineStageFlags src, VkPipelineStageFlags dst,
             VkAccessFlags srcAccess, VkAccessFlags dstAccess) {
    VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    b.srcAccessMask = srcAccess;
    b.dstAccessMask = dstAccess;
    vkCmdPipelineBarrier(cmd, src, dst, 0, 1, &b, 0, nullptr, 0, nullptr);
}

void copyImage(VkCommandBuffer cmd, const ImageHandle& src, const ImageHandle& dst) {
    VkImageCopy region{};
    region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.dstSubresource = region.srcSubresource;
    region.extent = {src.width, src.height, 1};
    vkCmdCopyImage(cmd, src.image, VK_IMAGE_LAYOUT_GENERAL,
                   dst.image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
}
} // namespace

struct VulkanRawGuard::Impl {
    ImageHandle raw;
    ImageHandle backup;
    bool saved = false;
};

VulkanRawGuard::VulkanRawGuard() : m_impl(std::make_unique<Impl>()) {}
VulkanRawGuard::~VulkanRawGuard() = default;

const ImageHandle& VulkanRawGuard::backup() const {
    return m_impl->backup;
}

bool VulkanRawGuard::ensure(VulkanDevice& device, const ImageHandle& raw) {
    auto& s = *m_impl;
    if (!raw.image) {
        return false;
    }
    if (s.backup.image && s.raw.view == raw.view && s.backup.width == raw.width &&
        s.backup.height == raw.height && s.backup.format == raw.format) {
        return true;
    }
    // Only allocation / image replacement waits. Normal frames remain asynchronous.
    device.waitIdle();
    destroy(device);
    s.raw = raw;
    s.backup = device.createImage2D(raw.width, raw.height, raw.format,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
        VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    if (!s.backup.image) {
        s.raw = {};
        return false;
    }
    std::vector<VulkanDevice::ImageClearRequest> clears{{&s.backup, 0, 0, 0, 0}};
    device.clearImages(clears);
    return true;
}

void VulkanRawGuard::save(VkCommandBuffer cmd) {
    auto& s = *m_impl;
    s.saved = false;
    if (!s.backup.image) {
        return;
    }
    barrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT,
        VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
    copyImage(cmd, s.raw, s.backup);
    barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT,
        VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
    s.saved = true;
}

void VulkanRawGuard::restore(VkCommandBuffer cmd) {
    auto& s = *m_impl;
    if (!s.saved) {
        return;
    }
    barrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT,
        VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
    copyImage(cmd, s.backup, s.raw);
    barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
    s.saved = false;
}

void VulkanRawGuard::destroy(VulkanDevice& device) {
    auto& s = *m_impl;
    if (s.backup.image) {
        device.destroyImage(s.backup);
    }
    s.backup = {};
    s.raw = {};
    s.saved = false;
}

} // namespace VulkanRT
