/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          Backend/VulkanMotionAov.cpp
* License:       MIT
* =========================================================================
*
* Motion AOV status, options and readback for VulkanBackendAdapter
* (docs/dev/RT_REALTIME_DENOISE_UPSCALE_PLAN.md B2). The per-dispatch work
* lives in renderProgressiveImpl and raygen.rgen / raster_taa.comp; this file
* is what render.motion_stats and render.set_motion reach.
*/
#include "Backend/VulkanBackend.h"

#include <cmath>
#include <cstring>
#include <vector>

namespace Backend {

namespace {

// IEEE 754 binary16 -> float for the RGBA16F motion images.
float motionHalfToFloat(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x03FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int e = -14;
            uint32_t m = mant;
            while ((m & 0x0400u) == 0u) { m <<= 1; --e; }
            bits = sign | uint32_t(e + 127) << 23 | ((m & 0x03FFu) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

} // namespace

void VulkanBackendAdapter::getMotionAovStatus(VulkanRT::MotionAovStatus& out) const {
    out = {};
    out.rt_available = m_motionHistory.motionImage() && m_motionHistory.motionImage()->image;
    out.rt_enabled = m_motionHistory.enabled();
    out.halton_jitter = m_motionHistory.haltonJitter();
    out.rt_history_valid = m_motionHistory.hasHistory();
    out.jitter[0] = m_motionHistory.currentJitter()[0];
    out.jitter[1] = m_motionHistory.currentJitter()[1];
    out.width = m_motionHistory.width();
    out.height = m_motionHistory.height();
    out.counters = m_motionHistory.counters();

    out.raster_available = m_viewportMode == ViewportMode::MaterialPreview &&
                           m_interactiveViewport.taaMotionImage.image != VK_NULL_HANDLE &&
                           m_interactiveViewport.taaPipeline != VK_NULL_HANDLE && m_taaEnabled;
    out.raster_objects_available = out.raster_available &&
        m_viewportMode == ViewportMode::MaterialPreview && m_rasterUseGlobalInstBuffer &&
        m_rasterGlobalInstBuf && m_rasterGlobalInstBuf->previousValid();
    if (m_viewportMode != ViewportMode::MaterialPreview) {
        out.raster_reason = "raster motion runs in Material/RayFusion mode";
    } else if (!out.raster_available) {
        out.raster_reason = m_interactiveViewport.taaPipeline == VK_NULL_HANDLE
            ? "no raster TAA pipeline on this device"
            : (!m_taaEnabled ? "raster TAA is disabled (viewport.set_taa); the TAA pass writes the raster motion"
                             : "raster motion image is not allocated");
    }
}

void VulkanBackendAdapter::setMotionAovOptions(bool enabled, bool haltonJitter) {
    m_motionHistory.setEnabled(enabled);
    if (m_motionHistory.haltonJitter() != haltonJitter) {
        m_motionHistory.setHaltonJitter(haltonJitter);
        // The primary-ray sub-pixel pattern changed: accumulated samples were
        // taken with the other pattern.
        resetAccumulation();
    }
}

bool VulkanBackendAdapter::sampleMotionImage(bool raster, VulkanRT::MotionImageSummary& out) {
    out = {};
    if (!m_device || !m_device->isInitialized()) { out.reason = "no Vulkan device"; return false; }
    if (raster) {
        VulkanRT::MotionAovStatus status;
        getMotionAovStatus(status);
        if (!status.raster_available) {
            out.reason = status.raster_reason;
            return false;
        }
    }
    const VulkanRT::ImageHandle* img = raster ? &m_interactiveViewport.taaMotionImage
                                              : m_motionHistory.motionImage();
    if (!img || !img->image || img->width == 0 || img->height == 0) {
        out.reason = raster ? "raster motion image is not allocated" : "RT motion image is not allocated";
        return false;
    }
    const uint32_t sampleWidth = raster ? img->width : m_upscale.renderWidth(img->width);
    const uint32_t sampleHeight = raster ? img->height : m_upscale.renderHeight(img->height);
    const uint64_t pixels = uint64_t(sampleWidth) * sampleHeight;
    const uint64_t allocationPixels = uint64_t(img->width) * img->height;
    VulkanRT::BufferCreateInfo bci;
    bci.size = allocationPixels * 8ull;   // RGBA16F
    bci.usage = VulkanRT::BufferUsage::TRANSFER_DST;
    bci.location = VulkanRT::MemoryLocation::GPU_TO_CPU;
    VulkanRT::BufferHandle staging = m_device->createBuffer(bci);
    if (!staging.buffer) { out.reason = "staging allocation failed"; return false; }
    m_device->copyImageToBuffer(*img, staging);   // single-time submit, waits
    std::vector<uint16_t> half(size_t(allocationPixels) * 4u);
    m_device->downloadBuffer(staging, half.data(), bci.size);
    m_device->destroyBuffer(staging);

    uint64_t rejected = 0, reactive = 0, accepted = 0;
    double sx = 0.0, sy = 0.0, sLen = 0.0, maxLen = 0.0;
    for (uint64_t sample = 0; sample < pixels; ++sample) {
        const uint64_t i = (sample / sampleWidth) * img->width + sample % sampleWidth;
        const float mx = motionHalfToFloat(half[i * 4 + 0]);
        const float my = motionHalfToFloat(half[i * 4 + 1]);
        const float rz = motionHalfToFloat(half[i * 4 + 2]);
        const float rw = motionHalfToFloat(half[i * 4 + 3]);
        if (rz > 0.0f) ++reactive;
        if (rw > 0.5f) { ++rejected; continue; }
        if (!std::isfinite(mx) || !std::isfinite(my)) continue;
        ++accepted;
        sx += mx;
        sy += my;
        const double len = std::sqrt(double(mx) * mx + double(my) * my);
        sLen += len;
        if (len > maxLen) maxLen = len;
    }
    out.sampled = true;
    out.pixels = pixels;
    out.rejected_fraction = double(rejected) / double(pixels);
    out.reactive_fraction = double(reactive) / double(pixels);
    if (accepted > 0) {
        out.mean_dx = sx / double(accepted);
        out.mean_dy = sy / double(accepted);
        out.mean_motion_px = sLen / double(accepted);
    }
    out.max_motion_px = maxLen;
    return true;
}

} // namespace Backend
