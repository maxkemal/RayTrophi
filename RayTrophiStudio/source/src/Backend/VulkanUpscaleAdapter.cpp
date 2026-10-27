#include "Backend/VulkanBackend.h"
#include "globals.h"

namespace Backend {
bool VulkanBackendAdapter::configureUpscale(const VulkanRT::UpscaleSettings& settings,
    bool moving, bool renderVisible) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    m_upscale.configure(settings, moving && renderVisible);
    if (!renderVisible) {
        m_upscale.setEligible(false, "Rendered viewport is not active");
    }
    if (m_upscale.needsReset() && renderVisible) {
        requestPresentRefresh();
    }
    return m_upscale.needsReset() && renderVisible;
}

VulkanRT::UpscaleStats VulkanBackendAdapter::upscaleStats(bool synchronize) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    auto stats = m_upscale.stats(*m_device, synchronize);
    const auto svgf = m_realtimeDenoiser.stats(*m_device, synchronize);
    stats.input_provider = stats.dispatches > 0 ? m_upscaleInput :
        (::render_settings.realtime_denoiser.enabled ? "svgf" : "none");
    stats.svgf_filter_ms = stats.input_provider == "svgf" ?
        svgf.temporal_ms + svgf.variance_ms + svgf.atrous_ms : 0.0;
    return stats;
}

void VulkanBackendAdapter::syncUpscaleFrame(uint32_t width, uint32_t height,
    bool resourcesReady) {
    if (!resourcesReady) {
        m_upscaleResetApplied = false;
    }
    // Auto (0) and Pro (1) both support the standard perspective projection.
    // Cinema (2) remains excluded until its lens model has matching motion vectors.
    const char* blockedReason = nullptr;
    if (m_viewportMode != ViewportMode::Rendered) {
        blockedReason = "Rendered viewport is not active";
    } else if (::render_settings.is_final_render_mode) {
        blockedReason = "final render is active";
    } else if (m_debugView != 0) {
        blockedReason = "renderer debug view is active";
    } else if (m_camera.orthographic) {
        blockedReason = "orthographic camera is unsupported";
    } else if (m_camera.camera_mode < 0 || m_camera.camera_mode > 1) {
        blockedReason = "camera mode requires unsupported lens motion vectors";
    } else if (m_camera.distortion != 0.0f) {
        blockedReason = "camera lens distortion is active";
    } else if (m_camera.chromaticAberrationEnabled) {
        blockedReason = "camera chromatic aberration is enabled";
    } else if (m_camera.shake_enabled) {
        blockedReason = "camera shake is enabled";
    } else if (::render_settings.realtime_denoiser.enabled &&
               ::render_settings.realtime_denoiser.debug_history) {
        blockedReason = "SVGF history debug view is active";
    } else if (!::render_settings.motion_aov_enabled) {
        blockedReason = "motion AOV is disabled";
    } else if (!m_device->hasTonemapPipeline()) {
        blockedReason = "tonemap pipeline is unavailable";
    }
    const bool eligible = blockedReason == nullptr;
    m_upscale.setEligible(eligible, eligible ? "" : blockedReason);
    if (resourcesReady) {
        // Composition resources (raw guard, and the input provider's when one
        // is enabled) must be known-good BEFORE selecting a reduced RT extent.
        // SVGF is not required: with it off FSR consumes the raw accumulation
        // (upscale_stats input_provider "none").
        const char* resourceReason = nullptr;
        const auto* motion = m_motionHistory.motionImage();
        if (::render_settings.upscale.mode != VulkanRT::UpscaleMode::Off && eligible) {
            if (!motion || !m_rawGuard.ensure(*m_device, m_outputImage)) {
                resourceReason = "raw display backup unavailable; full-resolution fallback";
            } else if (::render_settings.realtime_denoiser.enabled &&
                !m_realtimeDenoiser.ensure(*m_device, m_outputImage, m_rawGuard.backup(),
                    m_varianceImage, m_denoiserAlbedoImage, m_denoiserNormalImage,
                    m_denoiserPositionImage, *motion, *m_motionHistory.historySlot(0),
                    *m_motionHistory.historySlot(1))) {
                resourceReason = "SVGF resources unavailable; full-resolution fallback";
            }
        }
        m_upscale.setEligible(eligible && !resourceReason, !eligible ? blockedReason :
            (resourceReason ? resourceReason : ""));
        if (motion) {
            m_upscale.prepare(*m_device, m_outputImage, m_denoiserPositionImage, *motion);
        }
        float jitter[2]{};
        const bool external = m_upscale.jitter(m_motionHistory.counters().dispatches, jitter);
        m_motionHistory.setExternalJitter(external, jitter);
    }
    if (m_upscale.needsReset()) {
        if (!m_upscaleResetApplied) {
            // Never consume a slot tagged with the previous render extent.
            for (uint32_t slot = 0; slot < VulkanRT::VulkanDevice::kFrameSlotCount; ++slot) {
                if (m_tonemappedSlotInFlight[slot]) {
                    m_device->waitFrameSlot(slot);
                    m_tonemappedSlotInFlight[slot] = false;
                }
            }
            m_tonemappedFrameSlot = 0;
            resetAccumulation();
            m_upscaleResetApplied = true;
        }
        m_motionHistory.invalidate();
        m_realtimeDenoiser.invalidate();
        m_upscale.acceptReset(m_accumulationEpoch);
        // The fresh trace consumes the wake-up request; do not accidentally
        // force a different final sample than the unchanged native path.
        m_tonemapRefreshPending = false;
    }
    m_motionHistory.setActiveExtent(m_upscale.renderWidth(width), m_upscale.renderHeight(height));
    m_realtimeDenoiser.setActiveExtent(m_upscale.renderWidth(width), m_upscale.renderHeight(height));
}

bool VulkanBackendAdapter::upscaleSlotValid(uint32_t slot) const {
    return m_upscale.slotValid(slot);
}

void VulkanBackendAdapter::releaseUpscaleResources() {
    m_device->waitIdle();
    m_upscale.destroy(*m_device);
    // Both processors retain descriptors of the raw/AOV views. Release them
    // before resize can destroy/reuse those Vulkan handles.
    m_realtimeDenoiser.destroy(*m_device);
    m_rawGuard.destroy(*m_device);
    float unused[2]{};
    m_motionHistory.setExternalJitter(false, unused);
}
} // namespace Backend
