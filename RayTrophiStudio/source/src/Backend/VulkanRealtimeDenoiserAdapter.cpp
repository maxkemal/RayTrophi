#include "Backend/VulkanBackend.h"
#include "globals.h"

namespace Backend {

void VulkanBackendAdapter::configureRealtimeDenoiser(
    const VulkanRT::RealtimeDenoiserSettings& settings, bool moving) {
    const bool objectMotion = m_device &&
        m_device->motionInstanceSerial() != m_motionSeenInstanceSerial;
    m_realtimeDenoiser.configure(settings, moving || objectMotion);
}

VulkanRT::RealtimeDenoiserStats VulkanBackendAdapter::realtimeDenoiserStats(bool synchronize) {
    return m_realtimeDenoiser.stats(*m_device, synchronize);
}

bool VulkanBackendAdapter::realtimeDenoiserOwnsViewport() const {
    return m_realtimeDenoiser.ownsViewport();
}

bool VulkanBackendAdapter::prepareRealtimeDenoiser() {
    // Still-frame SVGF only when viewport OIDN is off: OIDN owns still frames
    // otherwise and would overwrite the display anyway.
    if (!m_realtimeDenoiser.wantsFrame(m_currentSamples + 1, m_targetSamples,
            ::render_settings.is_final_render_mode, m_debugView != 0,
            !::render_settings.use_denoiser)) {
        return false;
    }
    const auto* motion = m_motionHistory.motionImage();
    const auto* record0 = m_motionHistory.historySlot(0);
    const auto* record1 = m_motionHistory.historySlot(1);
    return motion && record0 && record1 && m_motionHistory.enabled() &&
        m_realtimeDenoiser.ensure(*m_device, m_outputImage, m_denoiserAlbedoImage,
            m_denoiserNormalImage, m_denoiserPositionImage, *motion, *record0, *record1);
}
bool VulkanBackendAdapter::submitRealtimeTrace(uint32_t slot, uint32_t width, uint32_t height,
    const std::function<void(VkCommandBuffer)>& prepass) {
    std::function<void(VkCommandBuffer)> before, after;
    if (prepareRealtimeDenoiser()) {
        before = [this, slot](VkCommandBuffer cmd) {
            // Recorded before m_motionHistory.advance(): writeIndex() is the slot
            // raygen writes in this same submission.
            float camera[4];
            m_motionHistory.cameraFootprint(camera);
            m_realtimeDenoiser.begin(*m_device, cmd, slot,
                m_accumulationEpoch, m_currentSamples + 1,
                m_motionHistory.counters().dispatches + 1,
                m_motionHistory.hasHistory(), m_motionHistory.writeIndex(), camera);
        };
        after = [this](VkCommandBuffer cmd) {
            m_realtimeDenoiser.end(cmd);
        };
    }
    const bool submitted = m_device->submitTraceTonemapAsync(slot, width, height, m_outputImage,
        m_tonemappedImage, m_tonemappedStagings[slot], true, prepass, before, after);
    if (!submitted) {
        m_realtimeDenoiser.invalidate();
    }
    return submitted;
}
} // namespace Backend
