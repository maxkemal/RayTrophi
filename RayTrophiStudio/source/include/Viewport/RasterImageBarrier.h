#pragma once
// Raster viewport görüntü layout/görünürlük bariyeri — TEK GÖVDE.
//
// ★★★ 2026-09-08'de buraya taşındı. Önce `VulkanViewportBackend.cpp` içinde
//   ANONİM namespace'teydi, yani ikinci bir çeviri birimi ona ulaşamıyordu ve
//   RT gölge geçişi kendi kopyasını çıkarmak zorunda kalacaktı. Bu depoda
//   kopyalanan gövde tekrar tekrar sessiz arızaya döndü (en son: raster
//   geometri kuyruğunun kopyası bir çağrıyı düşürdü ve GPU culling HİÇ
//   açılmadı, bkz. docs/dev/RASTER_GPU_CULLING_NEVER_ENABLED.md). Bir bariyer
//   sarmalayıcısı küçük bir gövde, ama iki kopyanın ayrışması burada
//   "bazen bozuluyor" sınıfı üretir — senkronizasyon hataları kare kare
//   değişir ve tekrar üretilemez.

#include <vulkan/vulkan.h>

namespace Backend {

inline void rasterImageBarrier(VkCommandBuffer cmd, VkImage image,
                               VkImageAspectFlags aspect,
                               VkImageLayout oldLayout, VkImageLayout newLayout,
                               VkAccessFlags srcAccess, VkAccessFlags dstAccess,
                               VkPipelineStageFlags srcStage,
                               VkPipelineStageFlags dstStage) {
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {aspect, 0, 1, 0, 1};
    barrier.srcAccessMask = srcAccess;
    barrier.dstAccessMask = dstAccess;
    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

}  // namespace Backend
