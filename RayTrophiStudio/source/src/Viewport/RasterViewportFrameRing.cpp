#include "Viewport/RasterViewportFrameRing.h"
#include "Backend/VulkanGpuBreadcrumbs.h"

#include "Backend/VulkanBackend.h"
#include "Backend/VulkanFailureReporting.h"
#include "Viewport/ParticleBillboardFrameBuffer.h"

#include <array>
#include <chrono>
#include <cstring>
#include <limits>
#include <vector>

namespace Backend {

namespace {
constexpr std::uint32_t kNoSlot = (std::numeric_limits<std::uint32_t>::max)();
}

struct RasterViewportFrameRing::Impl {
    struct Slot {
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        VulkanRT::BufferHandle readback;
        ParticleBillboardFrameBuffer billboards;
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
    // A GPU-bound producer may wait for a slot only AFTER the caller's
    // opportunistic consume. Preserve that completed image before reusing its
    // fence/readback buffer; otherwise every pass can discard its only ready frame.
    std::vector<std::uint8_t> readyPixels;
    std::uint64_t readySerial = 0;
    VkResult lastResult = VK_SUCCESS;
    Stats stats{};

    std::size_t byteSize() const {
        return static_cast<std::size_t>(width) *
               static_cast<std::size_t>(height) * 4u;
    }

    void copyReadback(const Slot& slot, void* destination) const {
        // GPU_TO_CPU is HOST_CACHED, not necessarily HOST_COHERENT.
        VkMappedMemoryRange range{};
        range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range.memory = slot.readback.memory;
        range.offset = 0;
        range.size = VK_WHOLE_SIZE;
        vkInvalidateMappedMemoryRanges(device->getDevice(), 1, &range);
        std::memcpy(destination, slot.mapped, byteSize());
    }

    void preserveCompletedImage(const Slot& slot) {
        if (!slot.mapped || slot.serial <= consumedSerial || slot.serial <= readySerial) {
            return;
        }
        readyPixels.resize(byteSize());
        copyReadback(slot, readyPixels.data());
        readySerial = slot.serial;
    }

    void destroy() {
        if (!device) return;

        VkDevice vkDevice = device->getDevice();
        for (Slot& slot : slots) {
            if (slot.inFlight && slot.fence != VK_NULL_HANDLE && !g_vulkan_device_lost) {
                // ★★★ Never UINT64_MAX here. MEASURED 2026-11-17 (cdb on a hung
                //   Studio after a device loss): the NVIDIA driver spun forever
                //   in this wait (SwitchToThread loop) instead of returning
                //   VK_ERROR_DEVICE_LOST, so the app could neither recover to CPU
                //   nor close. A lost device skips the wait (nothing will signal);
                //   a live one gets a bounded wait.
                const VkResult waited =
                    vkWaitForFences(vkDevice, 1, &slot.fence, VK_TRUE, 2000000000ull);
                if (waited != VK_SUCCESS) {
                    VulkanRT::reportVulkanDeviceFailure(waited, "RasterViewportFrameRing::destroy/vkWaitForFences");
                    SCENE_LOG_WARN("[Viewport] frame ring teardown: fence wait returned " +
                                   std::to_string(static_cast<int>(waited)) +
                                   "; releasing the slot without completion");
                }
            }
            if (slot.mapped && slot.readback.buffer) {
                device->unmapBuffer(slot.readback);
                slot.mapped = nullptr;
            }
            if (slot.readback.buffer) {
                device->destroyBuffer(slot.readback);
            }
            slot.billboards.release(*device);
            if (slot.fence != VK_NULL_HANDLE) {
                if (device->acousticPass) device->acousticPass->retireFence(slot.fence);
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
        readySerial = 0;
        readyPixels.clear();
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
            VulkanRT::waitFenceBounded(vkDevice, slot.fence, "RasterViewportFrameRing.cpp:234");
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
        // The fence is now complete. Do not reset it and submit over this
        // staging image until a CPU-owned copy can survive the slot reuse.
        impl_->preserveCompletedImage(slot);
        slot.inFlight = false;
    }

    if (impl_->device->acousticPass) impl_->device->acousticPass->collect(slot.fence);
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
    // This frame reads simulation compute results: vertex-pulled GPU particles,
    // the wet-paint atlas, indirect counts. The Player submits that compute work
    // asynchronously on this same queue (no host fence any more), so order it
    // here: every earlier compute/transfer write is visible to every stage of
    // this frame. One global barrier covers every earlier submission on the queue.
    {
        VkMemoryBarrier simReads{};
        simReads.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        simReads.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        simReads.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT |
                                 VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT |
                                 VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                                 VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(slot.commandBuffer,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT |
                                 VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 1, &simReads, 0, nullptr, 0, nullptr);
    }

    impl_->activeSlot = impl_->submitSlot;
    return slot.commandBuffer;
}

ParticleBillboardFrameBuffer* RasterViewportFrameRing::recordingBillboards(
    VkCommandBuffer commands) {
    if (impl_->activeSlot == kNoSlot) {
        return nullptr;
    }
    auto& slot = impl_->slots[impl_->activeSlot];
    return slot.commandBuffer == commands ? &slot.billboards : nullptr;
}

bool RasterViewportFrameRing::submitFrame(const VulkanRT::ImageHandle& colorImage,
                                          double* submitMs,
                                          bool captureReadback) {
    if (submitMs) *submitMs = 0.0;
    impl_->lastResult = VK_SUCCESS;
    if (!impl_->device || impl_->activeSlot == kNoSlot || !colorImage.image) {
        impl_->lastResult = VK_ERROR_INITIALIZATION_FAILED;
        return false;
    }

    Impl::Slot& slot = impl_->slots[impl_->activeSlot];
    VkCommandBuffer cmd = slot.commandBuffer;

    if (captureReadback) {
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
    }

    if (impl_->device->acousticPass) impl_->device->acousticPass->record(cmd, slot.fence);
    {
        VulkanRT::GpuBreadcrumbTrail& crumbs = VulkanRT::rasterFrameBreadcrumbs();
        const char* stage = captureReadback ? "readback_copy " : "raster_frame ";
        crumbs.mark(cmd, std::string(stage) + VulkanRT::kEndOfCommandBuffer);
        crumbs.commitSubmit();
    }
    const VkResult endResult = vkEndCommandBuffer(cmd);
    if (endResult != VK_SUCCESS) {
        if (impl_->device->acousticPass) {
            impl_->device->acousticPass->submitted(slot.fence, false);
        }
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
    if (impl_->device->acousticPass) {
        impl_->device->acousticPass->submitted(slot.fence, submitResult == VK_SUCCESS);
    }
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

    slot.serial = captureReadback ? impl_->nextSerial : 0;
    ++impl_->nextSerial;
    slot.inFlight = true;
    ++impl_->stats.framesSubmitted;
    impl_->submitSlot = (impl_->activeSlot + 1u) % kSlotCount;
    impl_->activeSlot = kNoSlot;
    return true;
}

bool RasterViewportFrameRing::submitAcousticOnly() {
    if (!impl_->device || !impl_->device->acousticPass || impl_->activeSlot != kNoSlot) {
        return false;
    }
    auto& device = *impl_->device;
    auto& pass = *device.acousticPass;
    if (!pass.hasPending()) {
        return false;
    }
    for (auto& slot : impl_->slots) {
        if (slot.serial > impl_->consumedSerial) {
            continue;
        }
        if (slot.inFlight && vkGetFenceStatus(device.getDevice(), slot.fence) != VK_SUCCESS) {
            continue;
        }
        pass.collect(slot.fence);
        if (vkResetCommandBuffer(slot.commandBuffer, 0) != VK_SUCCESS) {
            return false;
        }
        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(slot.commandBuffer, &begin) != VK_SUCCESS) {
            return false;
        }
        const bool recorded = pass.record(slot.commandBuffer, slot.fence);
        const bool ended = vkEndCommandBuffer(slot.commandBuffer) == VK_SUCCESS;
        if (!recorded || !ended) {
            pass.submitted(slot.fence, false);
            return false;
        }
        if (vkResetFences(device.getDevice(), 1, &slot.fence) != VK_SUCCESS) {
            pass.submitted(slot.fence, false);
            return false;
        }
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &slot.commandBuffer;
        const auto result = VulkanRT::reportVulkanDeviceFailure(
            vkQueueSubmit(device.getComputeQueue(), 1, &submit, slot.fence),
            "RasterViewportFrameRing/submitAcousticOnly");
        impl_->lastResult = result;
        pass.submitted(slot.fence, result == VK_SUCCESS);
        slot.inFlight = result == VK_SUCCESS;
        slot.serial = 0; // No image was produced; never publish old staging as a new frame.
        return result == VK_SUCCESS;
    }
    return false; // Busy ring: keep pending work without blocking rendering.
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
    const bool savedReady = impl_->readySerial > impl_->consumedSerial;
    if (savedReady && (!newest || impl_->readySerial > newest->serial)) {
        std::memcpy(dst, impl_->readyPixels.data(), impl_->byteSize());
        impl_->consumedSerial = impl_->readySerial;
    } else if (newest && newest->mapped) {
        impl_->copyReadback(*newest, dst);
        impl_->consumedSerial = newest->serial;
    } else {
        return false;
    }
    // Keep the allocation for reuse, but never publish an older saved image.
    if (impl_->readySerial <= impl_->consumedSerial) {
        impl_->readySerial = 0;
    }
    const std::uint64_t newestSubmitted = impl_->nextSerial - 1u;
    impl_->stats.lastPresentLatencyFrames =
        (newestSubmitted > impl_->consumedSerial)
            ? static_cast<std::uint32_t>(newestSubmitted - impl_->consumedSerial)
            : 0u;
    ++impl_->stats.framesConsumed;
    if (consumedSerial) *consumedSerial = impl_->consumedSerial;
    return true;
}

void RasterViewportFrameRing::noteStalePresent() {
    ++impl_->stats.stalePresents;
}

bool RasterViewportFrameRing::hasPendingFrame() const {
    if (!impl_->device) return false;
    if (impl_->readySerial > impl_->consumedSerial) return true;
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
            VulkanRT::waitFenceBounded(vkDevice, slot.fence, "RasterViewportFrameRing.cpp:489");
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
