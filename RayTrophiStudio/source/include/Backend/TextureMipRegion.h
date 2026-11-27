#pragma once

#include "Backend/VulkanBackend.h"

#include <algorithm>

namespace Backend {

// Rebuild the affected mip footprint for one array layer. Even extents use
// aligned 2x2 source blocks. Odd extents retain the whole-level blit.
inline void regenerateMipChainRegion(VkCommandBuffer cmd, const VulkanRT::ImageHandle& image,
                                     int32_t x, int32_t y, int32_t width, int32_t height) {
    if (cmd == VK_NULL_HANDLE || !image.image || image.mipLevels <= 1) {
        return;
    }

    VkImageMemoryBarrier base{};
    base.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    base.image = image.image;
    base.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    base.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    base.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, image.arrayLayer, 1};

    // All higher levels start read-only and become blit destinations.
    // Transition them together, along with the just-updated base level.
    VkImageMemoryBarrier begin[2] = {base, base};
    begin[0].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    begin[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    begin[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    begin[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    begin[1].oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    begin[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    begin[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    begin[1].subresourceRange.baseMipLevel = 1;
    begin[1].subresourceRange.levelCount = image.mipLevels - 1;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 2, begin);

    int32_t mipW = static_cast<int32_t>(image.width);
    int32_t mipH = static_cast<int32_t>(image.height);
    // Region on the current SOURCE level, half-open.
    int32_t rx0 = std::clamp(x, 0, mipW), ry0 = std::clamp(y, 0, mipH);
    int32_t rx1 = std::clamp(x + width, rx0, mipW), ry1 = std::clamp(y + height, ry0, mipH);
    bool wholeLevels = false;
    VkImageMemoryBarrier b = base;

    for (uint32_t i = 1; i < image.mipLevels; ++i) {
        const int32_t nextW = std::max(1, mipW / 2);
        const int32_t nextH = std::max(1, mipH / 2);
        // Exact 2:1 only on even extents; anything else is blitted whole.
        if ((mipW & 1) || (mipH & 1)) {
            wholeLevels = true;
        }
        int32_t sx0 = 0;
        int32_t sy0 = 0;
        int32_t sx1 = mipW;
        int32_t sy1 = mipH;
        int32_t dx0 = 0;
        int32_t dy0 = 0;
        int32_t dx1 = nextW;
        int32_t dy1 = nextH;
        if (!wholeLevels) {
            sx0 = rx0 & ~1; sy0 = ry0 & ~1;
            sx1 = std::min(mipW, (rx1 + 1) & ~1); sy1 = std::min(mipH, (ry1 + 1) & ~1);
            dx0 = sx0 / 2; dy0 = sy0 / 2; dx1 = sx1 / 2; dy1 = sy1 / 2;
        }

        if (sx1 > sx0 && sy1 > sy0) {
            VkImageBlit blit{};
            blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i - 1, image.arrayLayer, 1};
            blit.srcOffsets[0] = {sx0, sy0, 0};
            blit.srcOffsets[1] = {sx1, sy1, 1};
            blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i, image.arrayLayer, 1};
            blit.dstOffsets[0] = {dx0, dy0, 0};
            blit.dstOffsets[1] = {dx1, dy1, 1};
            vkCmdBlitImage(cmd,
                image.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1, &blit, VK_FILTER_LINEAR);
        }
        rx0 = dx0;
        ry0 = dy0;
        rx1 = dx1;
        ry1 = dy1;

        // mip i TRANSFER_DST -> TRANSFER_SRC (next iteration's
        // source) for non-final levels; final level goes straight to
        // SHADER_READ_ONLY in the closing barrier below.
        if (i + 1 < image.mipLevels) {
            b.subresourceRange.baseMipLevel = i;
            b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &b);
        }

        mipW = nextW;
        mipH = nextH;
    }

    // Levels that became sources and the final destination have distinct
    // old layouts, but share one synchronization point before shader reads.
    VkImageMemoryBarrier finish[2] = {base, base};
    finish[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    finish[0].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    finish[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    finish[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    finish[0].subresourceRange.levelCount = image.mipLevels - 1;
    finish[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    finish[1].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    finish[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    finish[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    finish[1].subresourceRange.baseMipLevel = image.mipLevels - 1;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 2, finish);
}

} // namespace Backend
