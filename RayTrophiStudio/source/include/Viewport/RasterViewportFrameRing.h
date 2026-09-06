#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include <vulkan/vulkan.h>

#include "Viewport/RasterFrameTelemetry.h"

namespace VulkanRT {
class VulkanDevice;
struct ImageHandle;
}

namespace Backend {

// Two-slot raster presentation scheduler. It mirrors the proven Vulkan RT
// submit/consume ownership model without coupling raster recording to RT passes:
// GPU writes slot N while the host consumes the newest completed older slot.
// Waiting is allowed only when a slot is about to be reused.
class RasterViewportFrameRing final {
public:
    static constexpr std::uint32_t kSlotCount = 2;

    // Ring-ömrü sayaçları. Değer; hiçbir handle sınırı geçmez.
    struct Stats {
        std::uint64_t framesSubmitted = 0;
        std::uint64_t framesConsumed = 0;
        // Raster work ran but no completed slot was ready, so the host had to
        // re-publish older pixels. Lives here, next to the counters it must
        // stay comparable with: a resize recreates the ring and resets them all
        // together. Kept outside, it would keep counting against a fresh
        // frames_submitted and read as a permanently sick bridge.
        std::uint64_t stalePresents = 0;
        std::uint64_t slotWaits = 0;
        std::uint64_t blockingWaits = 0;
        std::uint64_t resourceDrains = 0;
        // consumeNewestReady son çağrıda kaç kare geriden sunum yaptı.
        std::uint32_t lastPresentLatencyFrames = 0;
    };

    RasterViewportFrameRing();
    ~RasterViewportFrameRing();

    RasterViewportFrameRing(const RasterViewportFrameRing&) = delete;
    RasterViewportFrameRing& operator=(const RasterViewportFrameRing&) = delete;

    bool ensure(VulkanRT::VulkanDevice& device,
                std::uint32_t width,
                std::uint32_t height);

    // Returns an already-begun persistent command buffer. slotWaitMs measures
    // the bounded wait needed only when both GPU slots are still occupied.
    VkCommandBuffer beginFrame(double* slotWaitMs = nullptr);

    // Appends GENERAL -> TRANSFER_SRC -> readback -> GENERAL to the same command
    // buffer as raster rendering, then submits without a host wait. submitMs
    // measures the vkQueueSubmit itself, which must NOT include a fence wait.
    bool submitFrame(const VulkanRT::ImageHandle& colorImage,
                     double* submitMs = nullptr);

    // Copies the newest completed, not-yet-consumed slot to dst. Never waits.
    // false means the host should keep presenting its cached previous frame.
    bool consumeNewestReady(void* dst,
                            std::size_t dstBytes,
                            std::uint64_t* consumedSerial = nullptr);

    // The caller decides what "stale" means (it owns the presentation target),
    // so it reports the event rather than the ring inferring it.
    void noteStalePresent();

    bool hasPendingFrame() const;
    // Waits for submitted slots without destroying them. Resource mutation
    // paths call this before replacing a buffer referenced by raster draws.
    // countAsResourceDrain=false is used for deliberate presentation seeding so
    // the two very different reasons for blocking stay separable in telemetry.
    void waitAll(bool countAsResourceDrain = true);
    void reset();

    bool isReady() const;
    // Exact Vulkan result from the latest begin/submit operation. Callers must
    // distinguish transient slot state from VK_ERROR_DEVICE_LOST before trying
    // a compatibility submit on the same queue.
    VkResult lastResult() const;
    std::uint32_t width() const;
    std::uint32_t height() const;
    const Stats& stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace Backend
