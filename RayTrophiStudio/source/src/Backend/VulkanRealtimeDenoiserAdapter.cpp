#include "Backend/VulkanBackend.h"
#include "globals.h"

#include <algorithm>

namespace Backend {
namespace {
// Two in a row: one reset (a slider, a single transform edit, an upscale
// extent switch) is a new still image, not animation. Requiring a streak also
// keeps the FSR extent switch that scene motion triggers from feeding itself.
constexpr int kSceneMotionStreak = 2;
} // namespace

void VulkanBackendAdapter::configureRealtimeDenoiser(
    const VulkanRT::RealtimeDenoiserSettings& settings, bool moving) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    // Replaces a TLAS-serial comparison that could never fire for skinning:
    // the serial and its "seen" copy were both advanced inside the same
    // frame's render, so by the next policy call they were always equal and a
    // skinned scene ran the 1 spp still filter with its history wiped per frame.
    const bool sceneMotion = settings.scene_motion && !moving &&
        m_sceneResetStreak >= kSceneMotionStreak;
    m_policyCameraMoving = moving;
    m_motionSource = moving ? "camera" : (sceneMotion ? "scene" : "none");
    m_realtimeDenoiser.configure(settings, moving || sceneMotion);
}

VulkanRT::RealtimeDenoiserStats VulkanBackendAdapter::realtimeDenoiserStats(bool synchronize) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    auto stats = m_realtimeDenoiser.stats(*m_device, synchronize);
    stats.motion_source = m_motionSource;
    stats.scene_reset_streak = m_sceneResetStreak;
    return stats;
}

bool VulkanBackendAdapter::realtimeDenoiserOwnsViewport() const {
    return m_upscale.protectsDenoiser() || m_realtimeDenoiser.ownsViewport();
}

bool VulkanBackendAdapter::prepareRealtimeDenoiser() {
    // Still-frame SVGF only when viewport OIDN is off: OIDN owns still frames
    // otherwise and would overwrite the display anyway.
    const bool svgfWanted = m_realtimeDenoiser.wantsFrame(m_accumulationEpoch,
            m_currentSamples + 1, m_targetSamples,
            ::render_settings.is_final_render_mode, m_debugView != 0,
            !::render_settings.use_denoiser);
    const bool bridge = m_currentSamples + 1 < m_targetSamples - 1 &&
        m_upscale.bridgeWanted(m_currentSamples + 1,
            ::render_settings.is_final_render_mode, m_debugView != 0);
    m_svgfThisFrame = false;
    if (!svgfWanted && !m_upscale.active() && !bridge) {
        return false;
    }
    // The guard is the composition contract; SVGF is one optional input
    // provider. The upscaler and its bridge run without SVGF (provider "none":
    // the raw accumulation feeds FSR directly).
    if (!m_rawGuard.ensure(*m_device, m_outputImage)) {
        return false;
    }
    const auto* motion = m_motionHistory.motionImage();
    const auto* record0 = m_motionHistory.historySlot(0);
    const auto* record1 = m_motionHistory.historySlot(1);
    // SVGF also runs under the upscaler / bridge when enabled (forced moving
    // or warm), exactly as before the provider split.
    m_svgfThisFrame = ::render_settings.realtime_denoiser.enabled &&
        !::render_settings.is_final_render_mode &&
        motion && record0 && record1 && m_motionHistory.enabled() &&
        m_realtimeDenoiser.ensure(*m_device, m_outputImage, m_rawGuard.backup(),
            m_varianceImage, m_denoiserAlbedoImage, m_denoiserNormalImage,
            m_denoiserPositionImage, *motion, *record0, *record1);
    // A still SVGF frame with nothing else to compose: without SVGF the
    // guard would only copy raw twice.
    return m_svgfThisFrame || m_upscale.active() || bridge;
}
bool VulkanBackendAdapter::submitRealtimeTrace(uint32_t slot, uint32_t width, uint32_t height,
    const std::function<void(VkCommandBuffer)>& prepass) {
    std::function<void(VkCommandBuffer)> before, after;
    // A trace that starts a new accumulation while the camera is still is a
    // scene change; a streak of them is animation (configureRealtimeDenoiser).
    const bool reset = m_accumulationEpoch != m_sceneMotionEpoch;
    m_sceneMotionEpoch = m_accumulationEpoch;
    m_sceneResetStreak = reset && !m_policyCameraMoving ?
        std::min(m_sceneResetStreak + 1, 1 << 20) : 0;
    m_upscale.setSlotValid(slot, true);
    if (prepareRealtimeDenoiser()) {
        before = [this, slot](VkCommandBuffer cmd) {
            m_rawGuard.save(cmd);
            if (m_svgfThisFrame) {
                // Recorded before m_motionHistory.advance(): writeIndex() is the slot
                // raygen writes in this same submission.
                float camera[4];
                m_motionHistory.cameraFootprint(camera);
                m_realtimeDenoiser.begin(*m_device, cmd, slot,
                    m_accumulationEpoch, m_currentSamples + 1,
                    m_motionHistory.counters().dispatches + 1,
                    m_motionHistory.hasHistory(), m_motionHistory.writeIndex(), camera,
                    m_upscale.active());
            }
            float vectors[4][4];
            m_motionHistory.cameraVectors(vectors);
            if (m_upscale.active()) {
                m_upscaleInput = m_svgfThisFrame ? "svgf" : "none";
                const bool valid = m_upscale.record(*m_device, cmd, slot,
                    m_accumulationEpoch, m_currentSamples + 1, vectors, m_camera.fov,
                    1.0f, m_motionHistory.hasHistory());
                m_upscale.setSlotValid(slot, valid);
            } else if (m_currentSamples + 1 < m_targetSamples - 1 &&
                m_upscale.bridgeWanted(m_currentSamples + 1,
                    ::render_settings.is_final_render_mode, m_debugView != 0)) {
                m_upscale.recordBridge(cmd, m_accumulationEpoch, m_currentSamples + 1, vectors);
            }
        };
        after = [this](VkCommandBuffer cmd) {
            m_rawGuard.restore(cmd);
            m_realtimeDenoiser.end(cmd);
        };
    }
    const bool submitted = m_device->submitTraceTonemapAsync(slot, width, height, m_outputImage,
        m_tonemappedImage, m_tonemappedStagings[slot], true, prepass, before, after,
        m_upscale.renderWidth(width), m_upscale.renderHeight(height));
    if (submitted && m_targetSamples == 1 && !m_upscale.active() &&
        (::render_settings.upscale.mode != VulkanRT::UpscaleMode::Off ||
         m_upscale.stats(*m_device, false).dispatches > 0)) {
        // After an extent transition a one-sample target has no previous full
        // slot to consume. Pump tonemap once instead of leaving the old size.
        requestPresentRefresh();
    }
    if (!submitted) {
        m_upscale.setSlotValid(slot, false);
        m_upscale.invalidate();
        m_realtimeDenoiser.invalidate();
    }
    return submitted;
}
} // namespace Backend
