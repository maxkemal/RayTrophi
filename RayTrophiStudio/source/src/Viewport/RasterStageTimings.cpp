#include "Viewport/RasterStageTimings.h"
#include "Viewport/RasterGpuTimers.h"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace Backend {

// The accumulator cannot include the enum's header from ITS header without a
// cycle, so the two counts are checked against each other here, where both are
// visible. Adding a stage and forgetting the array size would otherwise drop
// the last stage from every window with no error.
static_assert(RasterStageAccumulator::kStageCount ==
                  static_cast<std::size_t>(RasterStage::Count),
              "RasterStageAccumulator::kStageCount must track RasterStage::Count");

namespace {

double nowMs() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// 95th percentile of a small sample, nearest-rank. Not interpolated on purpose:
// with 30 frames an interpolated p95 invents a value between two real frames,
// and every number this file publishes should be a frame that actually happened.
double percentile95(std::vector<double>& values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const std::size_t rank = static_cast<std::size_t>(
        std::ceil(0.95 * static_cast<double>(values.size())));
    return values[std::min(values.size(), std::max<std::size_t>(rank, 1u)) - 1u];
}

bool sameApplied(const RasterAppliedState& a, const RasterAppliedState& b) {
    // rt_cascades_replaced, visible_triangles, draw_calls and the instance count
    // are deliberately EXCLUDED: they move with the camera every frame, and
    // treating them as a settings change would make every orbit report itself
    // as contaminated. What is compared is what a person or a script CHOSE.
    return a.screen_gi.settings.enabled == b.screen_gi.settings.enabled &&
           a.screen_gi.settings.samples == b.screen_gi.settings.samples &&
           a.screen_gi.settings.filterRadius == b.screen_gi.settings.filterRadius &&
           a.screen_gi.settings.maxDistance == b.screen_gi.settings.maxDistance &&
           a.screen_gi.ready == b.screen_gi.ready &&
           // ★ Yansima AYARLARI karsilastirmaya girer, SAYACLARI girmez: sayaclar
           //   kamerayla her karede degisir ve onlari bir ayar degisimi saymak,
           //   her yorungeyi "kirlenmis olcum" olarak raporlamak olurdu.
           a.reflection.settings.enabled == b.reflection.settings.enabled &&
           a.reflection.settings.samples == b.reflection.settings.samples &&
           a.reflection.settings.roughnessGate == b.reflection.settings.roughnessGate &&
           a.reflection.settings.weightGate == b.reflection.settings.weightGate &&
           a.reflection.settings.maxDistance == b.reflection.settings.maxDistance &&
           a.reflection.ready == b.reflection.ready &&
           a.shading == b.shading &&
           a.quality_preset == b.quality_preset &&
           a.lighting_preset == b.lighting_preset &&
           a.width == b.width && a.height == b.height &&
           a.depth_prepass == b.depth_prepass &&
           a.gpu_culling == b.gpu_culling &&
           a.global_instance_buffer == b.global_instance_buffer &&
           a.rt_shadow_requested == b.rt_shadow_requested &&
           a.rt_shadow_ready == b.rt_shadow_ready &&
           a.directional_cascades == b.directional_cascades &&
           a.shadowed_lights == b.shadowed_lights &&
           a.scene_lights == b.scene_lights &&
           a.volume_count == b.volume_count;
}

} // namespace

void RasterStageAccumulator::reset() {
    m_count = 0;
    m_next = 0;
    m_appliedSeen = false;
    m_settingsChanged = false;
    m_applied = RasterAppliedState{};
    m_firstFrameStampMs = 0.0;
    m_lastFrameStampMs = 0.0;
}

void RasterStageAccumulator::record(const double* cpuStageMs, double cpuFrameMs,
                                    const double* gpuStageMs, const bool* gpuStageRan,
                                    double gpuFrameMs, bool gpuAvailable,
                                    const RasterAppliedState& applied) {
    Frame& f = m_frames[m_next];
    f = Frame{};
    for (std::size_t s = 0; s < kStageCount; ++s) {
        f.cpuStage[s] = cpuStageMs ? cpuStageMs[s] : 0.0;
        if (gpuAvailable && gpuStageMs) f.gpuStage[s] = gpuStageMs[s];
        if (gpuAvailable && gpuStageRan) f.gpuRan[s] = gpuStageRan[s];
    }
    f.cpuFrameMs = cpuFrameMs;
    f.gpuFrameMs = gpuAvailable ? gpuFrameMs : 0.0;
    f.gpuAvailable = gpuAvailable;

    m_next = (m_next + 1u) % kCapacity;
    if (m_count < kCapacity) ++m_count;

    const double stamp = nowMs();
    if (!m_appliedSeen) {
        m_applied = applied;
        m_appliedSeen = true;
        m_firstFrameStampMs = stamp;
    } else if (!sameApplied(m_applied, applied)) {
        m_settingsChanged = true;
        // Keep the LATEST, so a reader who ignores the warning at least sees
        // the configuration the window ended in rather than one it left behind.
        m_applied = applied;
    }
    m_lastFrameStampMs = stamp;
}

RasterStageTimings RasterStageAccumulator::snapshot(
        bool gpuSupported, const char* gpuUnsupportedReason) const {
    RasterStageTimings out;
    out.gpu_supported = gpuSupported;
    if (!gpuSupported && gpuUnsupportedReason)
        out.gpu_unsupported_reason = gpuUnsupportedReason;
    if (m_count == 0) {
        out.warnings.push_back(
            "no raster frame was measured since the last reset; the viewport "
            "renders only when it is marked dirty, so a still camera produces "
            "an empty window");
        return out;
    }

    out.available = true;
    out.frames = static_cast<std::uint32_t>(m_count);
    out.applied = m_applied;
    out.window_wall_ms = m_lastFrameStampMs - m_firstFrameStampMs;

    std::vector<double> cpuFrame, gpuFrame;
    cpuFrame.reserve(m_count);
    gpuFrame.reserve(m_count);
    std::vector<std::vector<double>> cpuStage(kStageCount), gpuStage(kStageCount);
    std::vector<std::uint32_t> ranCount(kStageCount, 0u);

    for (std::size_t i = 0; i < m_count; ++i) {
        const Frame& f = m_frames[i];
        cpuFrame.push_back(f.cpuFrameMs);
        for (std::size_t s = 0; s < kStageCount; ++s)
            cpuStage[s].push_back(f.cpuStage[s]);
        if (!f.gpuAvailable) continue;
        ++out.frames_with_gpu;
        gpuFrame.push_back(f.gpuFrameMs);
        for (std::size_t s = 0; s < kStageCount; ++s) {
            gpuStage[s].push_back(f.gpuStage[s]);
            if (f.gpuRan[s]) ++ranCount[s];
        }
    }

    auto mean = [](const std::vector<double>& v) {
        if (v.empty()) return 0.0;
        double sum = 0.0;
        for (double x : v) sum += x;
        return sum / static_cast<double>(v.size());
    };

    out.frame_cpu_mean_ms = mean(cpuFrame);
    out.frame_gpu_mean_ms = mean(gpuFrame);
    out.frame_cpu_p95_ms = percentile95(cpuFrame);
    out.frame_gpu_p95_ms = percentile95(gpuFrame);

    out.stages.reserve(kStageCount);
    for (std::size_t s = 0; s < kStageCount; ++s) {
        RasterStageSample sample;
        sample.name = rasterStageName(static_cast<RasterStage>(s));
        sample.cpu_mean_ms = mean(cpuStage[s]);
        sample.cpu_p95_ms = percentile95(cpuStage[s]);
        sample.gpu_mean_ms = mean(gpuStage[s]);
        sample.gpu_p95_ms = percentile95(gpuStage[s]);
        sample.frames_ran = ranCount[s];
        out.stages.push_back(std::move(sample));
    }

    // ── The instrument's own reasons to distrust it ─────────────────────────
    if (m_settingsChanged) {
        out.warnings.push_back(
            "a viewport setting changed DURING this window; the means average "
            "two different configurations and must not be compared with another "
            "window - reset and measure again");
    }
    if (!gpuSupported) {
        out.warnings.push_back(
            "GPU timestamps are unavailable on this device or queue, so every "
            "gpu_* number is ABSENT rather than zero; the cpu_* half is real");
    } else if (out.frames_with_gpu == 0u) {
        out.warnings.push_back(
            "no frame's GPU marks had retired when this was read; drive a few "
            "more frames before reading");
    } else if (out.frames_with_gpu * 2u < out.frames) {
        out.warnings.push_back(
            "fewer than half the frames carried GPU timings, so the gpu_* means "
            "are drawn from a small subset of the window");
    }
    if (out.frames < 8u) {
        out.warnings.push_back(
            "fewer than 8 frames in the window; a p95 over this few frames is "
            "one frame, not a percentile");
    }
    // ★ A window that spans far more wall time than its frames account for was
    //   not a frame rate: it was a camera being nudged once per round trip.
    //   Quoting it as ms/frame is right, quoting it as FPS is not.
    if (out.frames > 1u && out.window_wall_ms >
            3.0 * out.frame_cpu_mean_ms * static_cast<double>(out.frames)) {
        out.warnings.push_back(
            "the window spans much more wall time than its frames cost: these "
            "are individually driven frames, not a sustained frame rate");
    }
    if (out.applied.shading != "material" && out.applied.shading != "solid" &&
        out.applied.shading != "matcap") {
        out.warnings.push_back(
            "the viewport is not in a raster shading mode; these timings do not "
            "describe the raster path");
    }
    if (out.applied.rt_shadow_requested && !out.applied.rt_shadow_ready) {
        out.warnings.push_back(
            "the RT shadow pass was requested but never built during this "
            "window, so the cascade cost it normally replaces is still in these "
            "numbers - read viewport.rt_shadow 'reason'");
    }
    return out;
}

} // namespace Backend
