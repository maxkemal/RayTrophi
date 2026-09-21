// Backend side of the per-pass raster frame timing.
//
// ★★★ The CPU and GPU halves are closed at the SAME call site (markRasterStage
//   writes both), because the one way this instrument could lie quietly is by
//   letting its two clocks describe different boundaries. A host stage that
//   ends at the draw call and a GPU stage that ends after the next pass would
//   still produce two plausible tables that could not be compared with each
//   other, and nobody would report that as a bug.

#include "Backend/VulkanBackend.h"
#include "globals.h"

#include <algorithm>
#include <string>

namespace Backend {

void VulkanBackendAdapter::beginRasterStageTimings(VkCommandBuffer cmd) {
    m_rasterCpuStageMs.fill(0.0);
    if (!m_device || cmd == VK_NULL_HANDLE) return;
    m_rasterGpuTimers.ensure(*m_device);
    m_rasterGpuTimers.beginFrame(cmd);
    m_rasterStageClock = std::chrono::steady_clock::now();
}

void VulkanBackendAdapter::markRasterStage(VkCommandBuffer cmd, RasterStage stage,
                                            bool ran) {
    if (cmd == VK_NULL_HANDLE) return;
    const auto now = std::chrono::steady_clock::now();
    const std::size_t index = static_cast<std::size_t>(stage);
    if (index < m_rasterCpuStageMs.size()) {
        // Same gap-filling contract as the GPU marks: whatever host time has
        // passed since the previous mark belongs to THIS stage, including the
        // stages skipped over on this path. They report it as ~0 because no
        // time passed in them.
        m_rasterCpuStageMs[index] =
            std::chrono::duration<double, std::milli>(now - m_rasterStageClock).count();
    }
    m_rasterStageClock = now;
    m_rasterGpuTimers.mark(cmd, stage, ran);
}

RasterAppliedState VulkanBackendAdapter::sampleRasterAppliedState() const {
    RasterAppliedState out;
    out.screen_gi = screenGiStatus();
    out.reflection = reflectionStatus();
    switch (m_viewportMode) {
        case ViewportMode::Solid:           out.shading = "solid"; break;
        case ViewportMode::MaterialPreview: out.shading = "material"; break;
        case ViewportMode::Matcap:          out.shading = "matcap"; break;
        case ViewportMode::Rendered:        out.shading = "rendered"; break;
        default:                            out.shading = "unknown"; break;
    }
    switch (::render_settings.material_preview_lighting_preset) {
        case MaterialPreviewLightingPreset::Scene: out.lighting_preset = "scene"; break;
        default:                                   out.lighting_preset = "three_point"; break;
    }
    switch (::render_settings.raster_viewport_quality_preset) {
        case RasterViewportQualityPreset::Auto:        out.quality_preset = "auto"; break;
        case RasterViewportQualityPreset::Performance: out.quality_preset = "performance"; break;
        case RasterViewportQualityPreset::Balanced:    out.quality_preset = "balanced"; break;
        case RasterViewportQualityPreset::Quality:     out.quality_preset = "quality"; break;
        case RasterViewportQualityPreset::Full:        out.quality_preset = "full"; break;
        default:                                       out.quality_preset = "unknown"; break;
    }
    out.width = static_cast<std::uint32_t>((std::max)(m_interactiveViewport.width, 0));
    out.height = static_cast<std::uint32_t>((std::max)(m_interactiveViewport.height, 0));

    // ★★★★ APPLIED, never requested. Both of these have shipped as a silent
    //   divergence in this repo: the depth prepass arm on with no pipeline
    //   built, and GPU culling requested but dropped by a layout override.
    //   Reading the request here would make the instrument agree with the panel
    //   and disagree with the frame.
    out.depth_prepass = m_rasterDepthPrepassRan;
    out.gpu_culling = m_rasterGpuCullActive;
    out.global_instance_buffer = m_rasterUseGlobalInstBuffer;

    // Through the same accessor the IPC surface uses, so the timing window and
    // viewport.rt_shadow can never disagree about whether the pass was live.
    bool rtSupported = false, rtReady = false;
    uint32_t rtRays = 0, rtReplaced = 0;
    std::string rtReason;
    getRtShadowStatus(rtSupported, rtReady, rtRays, rtReplaced, rtReason);
    out.rt_shadow_requested = m_rtShadowAllowed;
    out.rt_shadow_ready = rtReady;
    out.rt_cascades_replaced = rtReplaced;
    out.directional_cascades = static_cast<std::uint32_t>(
        rasterDirectionalShadowCascades(::render_settings.raster_viewport_quality_preset));
    bool atlasActive = false;
    materialPreviewShadowAtlasState(atlasActive, out.shadowed_lights);
    out.scene_lights = m_device ? m_device->m_lightCount : 0u;
    out.volume_count = m_device ? m_device->m_volumeCount : 0u;
    out.total_instances = static_cast<std::uint32_t>(m_rasterInstances.size());
    return out;
}

void VulkanBackendAdapter::publishRasterStageTimings(VkCommandBuffer cmd,
                                                      double frameCpuMs,
                                                      uint64_t visibleTriangles,
                                                      uint32_t drawCalls) {
    if (cmd == VK_NULL_HANDLE) return;
    m_rasterGpuTimers.endFrame(cmd);

    // The GPU numbers that land here belong to an EARLIER frame -- marks are
    // read without waiting, so the newest retired slot is one or two frames
    // back. Over a window that is irrelevant; for a single frame it is not, and
    // frames_with_gpu lagging frames is how the caller sees it.
    RasterGpuFrameTimings gpu{};
    if (m_device) gpu = m_rasterGpuTimers.collect(*m_device);

    RasterAppliedState applied = sampleRasterAppliedState();
    applied.visible_triangles = visibleTriangles;
    applied.draw_calls = drawCalls;

    m_rasterStageAccum.record(m_rasterCpuStageMs.data(), frameCpuMs,
                              gpu.stageMs.data(), gpu.stageRan.data(),
                              gpu.totalMs, gpu.available, applied);
}

bool VulkanBackendAdapter::getRasterStageTimings(RasterStageTimings& out) const {
    out = m_rasterStageAccum.snapshot(m_rasterGpuTimers.supported(),
                                      m_rasterGpuTimers.unsupportedReason());
    return true;
}

bool VulkanBackendAdapter::resetRasterStageTimings() {
    m_rasterStageAccum.reset();
    // The viewport renders only when dirty. Without this a reset followed by a
    // read on a still camera returns an empty window and reads as "measured
    // nothing" when it should read as "here is the first frame".
    m_interactiveViewport.dirty = true;
    return true;
}

} // namespace Backend
