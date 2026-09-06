#include "Viewport/RasterViewportFrameRing.h"

#include "Backend/VulkanBackend.h"

#include <array>
#include <chrono>
#include <cstring>
#include <limits>

namespace Backend {

namespace {
constexpr std::uint32_t kNoSlot = (std::numeric_limits<std::uint32_t>::max)();
}

struct RasterViewportFrameRing::Impl {
    struct Slot {
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        VulkanRT::BufferHandle readback;
        void* mapped = nullptr;
        std::uint64_t serial = 0;
        bool inFlight = false;
    };

    VulkanRT::VulkanDevice* device = nullptr;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    std::array<Slot, kSlotCount> slots{};
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t submitSlot = 0;
    std::uint32_t activeSlot = kNoSlot;
    std::uint64_t nextSerial = 1;
    std::uint64_t consumedSerial = 0;
    VkResult lastResult = VK_SUCCESS;
    Stats stats{};

    std::size_t byteSize() const {
        return static_cast<std::size_t>(width) *
               static_cast<std::size_t>(height) * 4u;
    }

    void destroy() {
        if (!device) return;

        VkDevice vkDevice = device->getDevice();
        for (Slot& slot : slots) {
            if (slot.inFlight && slot.fence != VK_NULL_HANDLE) {
                vkWaitForFences(vkDevice, 1, &slot.fence, VK_TRUE, UINT64_MAX);
            }
            if (slot.mapped && slot.readback.buffer) {
                device->unmapBuffer(slot.readback);
                slot.mapped = nullptr;
            }
            if (slot.readback.buffer) {
                device->destroyBuffer(slot.readback);
            }
            if (slot.fence != VK_NULL_HANDLE) {
                vkDestroyFence(vkDevice, slot.fence, nullptr);
                slot.fence = VK_NULL_HANDLE;
            }
            if (slot.commandBuffer != VK_NULL_HANDLE && commandPool != VK_NULL_HANDLE) {
                vkFreeCommandBuffers(vkDevice, commandPool, 1, &slot.commandBuffer);
                slot.commandBuffer = VK_NULL_HANDLE;
            }
            slot = Slot{};
        }
        if (commandPool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(vkDevice, commandPool, nullptr);
            commandPool = VK_NULL_HANDLE;
        }

        device = nullptr;
        width = 0;
        height = 0;
        submitSlot = 0;
        activeSlot = kNoSlot;
        nextSerial = 1;
        consumedSerial = 0;
        lastResult = VK_SUCCESS;
        stats = Stats{};
    }
};

RasterViewportFrameRing::RasterViewportFrameRing()
    : impl_(std::make_unique<Impl>()) {}

RasterViewportFrameRing::~RasterViewportFrameRing() {
    reset();
}

bool RasterViewportFrameRing::ensure(VulkanRT::VulkanDevice& device,
                                     std::uint32_t width,
                                     std::uint32_t height) {
    if (width == 0 || height == 0 || !device.isInitialized() ||
        !device.supportsGraphicsQueue()) {
        return false;
    }

    if (impl_->device == &device && impl_->width == width &&
        impl_->height == height && impl_->commandPool != VK_NULL_HANDLE) {
        return true;
    }

    impl_->destroy();
    impl_->device = &device;
    impl_->width = width;
    impl_->height = height;

    VkDevice vkDevice = device.getDevice();
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    // Single-queue device: the "compute" queue is the one that carries raster
    // submissions too (see VulkanDevice::endSingleTimeCommands). A pool from any
    // other family would be rejected at submit time.
    poolInfo.queueFamilyIndex = device.getComputeQueueFamily();
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(vkDevice, &poolInfo, nullptr, &impl_->commandPool) != VK_SUCCESS) {
        impl_->destroy();
        return false;
    }

    const std::size_t bytes = impl_->byteSize();
    for (Impl::Slot& slot : impl_->slots) {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = impl_->commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(vkDevice, &allocInfo, &slot.commandBuffer) != VK_SUCCESS) {
            impl_->destroy();
            return false;
        }

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        if (vkCreateFence(vkDevice, &fenceInfo, nullptr, &slot.fence) != VK_SUCCESS) {
            impl_->destroy();
            return false;
        }

        VulkanRT::BufferCreateInfo bufferInfo{};
        bufferInfo.size = bytes;
        bufferInfo.usage = VulkanRT::BufferUsage::TRANSFER_DST;
        bufferInfo.location = VulkanRT::MemoryLocation::GPU_TO_CPU;
        bufferInfo.initialData = nullptr;
        slot.readback = device.createBuffer(bufferInfo);
        if (!slot.readback.buffer) {
            impl_->destroy();
            return false;
        }
        // Persistently mapped for the ring lifetime: a per-frame map/unmap is
        // exactly the serial host cost this phase exists to remove.
        slot.mapped = device.mapBuffer(slot.readback);
        if (!slot.mapped) {
            impl_->destroy();
            return false;
        }
    }

    return true;
}

VkCommandBuffer RasterViewportFrameRing::beginFrame(double* slotWaitMs) {
    if (slotWaitMs) *slotWaitMs = 0.0;
    impl_->lastResult = VK_SUCCESS;
    if (!impl_->device || impl_->commandPool == VK_NULL_HANDLE) {
        impl_->lastResult = VK_ERROR_INITIALIZATION_FAILED;
        return VK_NULL_HANDLE;
    }
    if (impl_->activeSlot != kNoSlot) {
        impl_->lastResult = VK_NOT_READY;
        return VK_NULL_HANDLE;
    }

    Impl::Slot& slot = impl_->slots[impl_->submitSlot];
    VkDevice vkDevice = impl_->device->getDevice();
    if (slot.inFlight) {
        // ★★ Sayac YALNIZCA GERCEKTEN BLOKLANDIGINDA artar. Eskiden yuva
        //   `inFlight` isaretli oldugu HER karede artiyordu; iki yuvali bir
        //   halkada bu neredeyse her karedir, cunku iki kare onceki yuvanin
        //   fence'i coktan sinyallenmistir ve bekleme SIFIR surer. Yani sayac
        //   "halka darbogazda" gibi okunan bir CAGRI SAYISI raporluyordu --
        //   ve yanindaki `slot_wait_ms` 0.0 ile kendini zaten yalanliyordu.
        //   Ayni arizanin ucuncusu icin bkz. waitAll ve noteStalePresent.
        const bool alreadySignalled =
            vkGetFenceStatus(vkDevice, slot.fence) == VK_SUCCESS;
        const auto waitStart = std::chrono::steady_clock::now();
        const VkResult waitResult =
            vkWaitForFences(vkDevice, 1, &slot.fence, VK_TRUE, UINT64_MAX);
        const auto waitEnd = std::chrono::steady_clock::now();
        if (slotWaitMs) {
            *slotWaitMs = std::chrono::duration<double, std::milli>(
                waitEnd - waitStart).count();
        }
        if (!alreadySignalled) ++impl_->stats.slotWaits;
        if (waitResult != VK_SUCCESS) {
            impl_->lastResult = waitResult;
            return VK_NULL_HANDLE;
        }
        slot.inFlight = false;
    }

    const VkResult resetFenceResult = vkResetFences(vkDevice, 1, &slot.fence);
    if (resetFenceResult != VK_SUCCESS) {
        impl_->lastResult = resetFenceResult;
        return VK_NULL_HANDLE;
    }
    const VkResult resetCommandResult = vkResetCommandBuffer(slot.commandBuffer, 0);
    if (resetCommandResult != VK_SUCCESS) {
        impl_->lastResult = resetCommandResult;
        return VK_NULL_HANDLE;
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    const VkResult beginResult = vkBeginCommandBuffer(slot.commandBuffer, &beginInfo);
    if (beginResult != VK_SUCCESS) {
        impl_->lastResult = beginResult;
        return VK_NULL_HANDLE;
    }

    impl_->activeSlot = impl_->submitSlot;
    return slot.commandBuffer;
}

bool RasterViewportFrameRing::submitFrame(const VulkanRT::ImageHandle& colorImage,
                                          double* submitMs) {
    if (submitMs) *submitMs = 0.0;
    impl_->lastResult = VK_SUCCESS;
    if (!impl_->device || impl_->activeSlot == kNoSlot || !colorImage.image) {
        impl_->lastResult = VK_ERROR_INITIALIZATION_FAILED;
        return false;
    }

    Impl::Slot& slot = impl_->slots[impl_->activeSlot];
    VkCommandBuffer cmd = slot.commandBuffer;

    VkImageMemoryBarrier toTransfer{};
    toTransfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toTransfer.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                               VK_ACCESS_SHADER_WRITE_BIT;
    toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    toTransfer.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    toTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransfer.image = colorImage.image;
    toTransfer.subresourceRange = {
        VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1
    };
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &toTransfer);

    VkBufferImageCopy copy{};
    copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy.imageSubresource.mipLevel = 0;
    copy.imageSubresource.baseArrayLayer = 0;
    copy.imageSubresource.layerCount = 1;
    copy.imageExtent = {colorImage.width, colorImage.height, 1};
    vkCmdCopyImageToBuffer(cmd, colorImage.image,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           slot.readback.buffer, 1, &copy);

    VkImageMemoryBarrier toGeneral{};
    toGeneral.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toGeneral.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    toGeneral.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                              VK_ACCESS_SHADER_READ_BIT;
    toGeneral.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    toGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toGeneral.image = colorImage.image;
    toGeneral.subresourceRange = {
        VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1
    };
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &toGeneral);

    const VkResult endResult = vkEndCommandBuffer(cmd);
    if (endResult != VK_SUCCESS) {
        impl_->lastResult = endResult;
        impl_->activeSlot = kNoSlot;
        return false;
    }

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    const auto submitStart = std::chrono::steady_clock::now();
    const VkResult submitResult = vkQueueSubmit(impl_->device->getComputeQueue(), 1,
                                                &submitInfo, slot.fence);
    const auto submitEnd = std::chrono::steady_clock::now();
    if (submitMs) {
        *submitMs = std::chrono::duration<double, std::milli>(
            submitEnd - submitStart).count();
    }
    if (submitResult != VK_SUCCESS) {
        impl_->lastResult = submitResult;
        impl_->activeSlot = kNoSlot;
        return false;
    }

    slot.serial = impl_->nextSerial++;
    slot.inFlight = true;
    ++impl_->stats.framesSubmitted;
    impl_->submitSlot = (impl_->activeSlot + 1u) % kSlotCount;
    impl_->activeSlot = kNoSlot;
    return true;
}

bool RasterViewportFrameRing::consumeNewestReady(void* dst,
                                                 std::size_t dstBytes,
                                                 std::uint64_t* consumedSerial) {
    if (!impl_->device || !dst || dstBytes < impl_->byteSize()) return false;

    VkDevice vkDevice = impl_->device->getDevice();
    Impl::Slot* newest = nullptr;
    for (Impl::Slot& slot : impl_->slots) {
        if (!slot.inFlight || slot.serial <= impl_->consumedSerial) continue;
        if (vkGetFenceStatus(vkDevice, slot.fence) != VK_SUCCESS) continue;
        if (!newest || slot.serial > newest->serial) newest = &slot;
    }
    if (!newest || !newest->mapped) return false;

    // GPU_TO_CPU is HOST_CACHED and is not guaranteed HOST_COHERENT; without
    // this the memcpy can read a stale cache line and the viewport would show a
    // frame that was never rendered.
    VkMappedMemoryRange range{};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = newest->readback.memory;
    range.offset = 0;
    range.size = VK_WHOLE_SIZE;
    vkInvalidateMappedMemoryRanges(vkDevice, 1, &range);
    std::memcpy(dst, newest->mapped, impl_->byteSize());

    impl_->consumedSerial = newest->serial;
    const std::uint64_t newestSubmitted = impl_->nextSerial - 1u;
    impl_->stats.lastPresentLatencyFrames =
        (newestSubmitted > newest->serial)
            ? static_cast<std::uint32_t>(newestSubmitted - newest->serial)
            : 0u;
    ++impl_->stats.framesConsumed;
    if (consumedSerial) *consumedSerial = newest->serial;
    return true;
}

void RasterViewportFrameRing::noteStalePresent() {
    ++impl_->stats.stalePresents;
}

bool RasterViewportFrameRing::hasPendingFrame() const {
    if (!impl_->device) return false;
    for (const Impl::Slot& slot : impl_->slots) {
        if (slot.inFlight && slot.serial > impl_->consumedSerial) return true;
    }
    return false;
}

void RasterViewportFrameRing::waitAll(bool countAsResourceDrain) {
    if (!impl_->device) return;
    VkDevice vkDevice = impl_->device->getDevice();
    // ★★ Sayac YALNIZCA GERCEKTEN BLOKLANDIGINDA artar. Eskiden `inFlight`
    //   isaretli her yuva icin artiyordu, ama waitAll `inFlight`'i TEMIZLEMIYOR:
    //   ayni kare icinde ikinci ve sonraki cagrilar zaten sinyallenmis bir
    //   fence'i "bekleyip" sayaci sisiriyordu. Yani resource_drains bir maliyet
    //   degil, bir CAGRI SAYISI raporluyordu -- olcu aletinin kendi arizasi.
    bool waited = false;
    for (Impl::Slot& slot : impl_->slots) {
        if (slot.inFlight && slot.fence != VK_NULL_HANDLE) {
            if (vkGetFenceStatus(vkDevice, slot.fence) == VK_SUCCESS) continue;
            vkWaitForFences(vkDevice, 1, &slot.fence, VK_TRUE, UINT64_MAX);
            waited = true;
        }
    }
    if (!waited) return;
    if (countAsResourceDrain) ++impl_->stats.resourceDrains;
    else ++impl_->stats.blockingWaits;
}

void RasterViewportFrameRing::reset() {
    if (impl_) impl_->destroy();
}

bool RasterViewportFrameRing::isReady() const {
    return impl_->device != nullptr && impl_->commandPool != VK_NULL_HANDLE;
}

VkResult RasterViewportFrameRing::lastResult() const { return impl_->lastResult; }

std::uint32_t RasterViewportFrameRing::width() const { return impl_->width; }
std::uint32_t RasterViewportFrameRing::height() const { return impl_->height; }
const RasterViewportFrameRing::Stats& RasterViewportFrameRing::stats() const {
    return impl_->stats;
}

} // namespace Backend
