#pragma once

// What one measured window of raster viewport frames cost, per pass, on both
// sides of the bus -- VALUES only, no handles.
//
// ★★★★★ THE POINT OF THE `applied` BLOCK. A timing is meaningless without the
//   configuration it was taken under, and in this app the configuration a panel
//   REQUESTS and the one the frame APPLIES are different things: the depth
//   prepass can be switched on with no pipeline built, GPU culling can be
//   requested and dropped by a layout override, the RT shadow can be enabled
//   and decline the light, scatter proxies can be disabled by the quality
//   preset. Every one of those has already shipped as a silent divergence here.
//   So the window carries what the frames ACTUALLY did, sampled from the same
//   frames that produced the timings -- never re-read afterwards, and never
//   taken from the request side.
//
// ★★★ And it carries `settings_changed`: if any of it moved DURING the window,
//   the mean is an average over two different machines and must not be
//   compared with anything. The instrument says so itself rather than leaving
//   the reader to notice.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include "RayFusion/ScreenGi.h"
#include "RayFusion/Reflection.h"

namespace Backend {

struct RasterStageSample {
    std::string name;
    // Mean over the frames in the window, and the 95th percentile, which is
    // what a hitch lives in. A mean alone hides the stall that makes a viewport
    // feel broken; a max alone reports one unlucky frame as the truth.
    double cpu_mean_ms = 0.0;
    double cpu_p95_ms = 0.0;
    double gpu_mean_ms = 0.0;
    double gpu_p95_ms = 0.0;
    // Frames in the window where this stage recorded work. 0 with a non-zero
    // gpu_mean_ms would be a contradiction; 0 with 0 ms means the stage was
    // skipped every frame, which is a RESULT (the RT shadow handoff), not an
    // absence of data.
    std::uint32_t frames_ran = 0;
};

// The configuration the measured frames actually ran under.
struct RasterAppliedState {
    RayFusion::ScreenGiStatus screen_gi;
    // ★ Yansima da BURADAN gorunur olmak zorunda: kapisi sahneye bagli oldugu
    //   icin maliyeti ve kapsami sahneden sahneye degisir, ve "neden yansima
    //   yok" sorusunun cevabi `reason` alanidir.
    RayFusion::ReflectionStatus reflection;
    std::string shading;            // solid | material | rendered | matcap
    std::string quality_preset;
    std::string lighting_preset;
    std::uint32_t width = 0, height = 0;
    bool depth_prepass = false;     // APPLIED, not requested
    bool gpu_culling = false;       // APPLIED
    bool global_instance_buffer = false;
    bool rt_shadow_requested = false;
    bool rt_shadow_ready = false;   // built AND recorded, not just asked for
    std::uint32_t rt_cascades_replaced = 0;
    std::uint32_t directional_cascades = 0;
    std::uint32_t shadowed_lights = 0;
    std::uint32_t scene_lights = 0;
    std::uint32_t volume_count = 0;
    std::uint64_t visible_triangles = 0;
    std::uint32_t total_instances = 0;
    std::uint32_t draw_calls = 0;
};

struct RasterStageTimings {
    // false = no raster frame has been measured since the last reset. Every
    // number below is ABSENT. This is the state a still camera produces: the
    // viewport renders only when dirty, so a window with nothing moving in it
    // legitimately contains zero frames.
    bool available = false;
    // false = this device or queue cannot write timestamps; CPU numbers are
    // still real, GPU numbers are absent. Never faked to zero.
    bool gpu_supported = false;
    std::string gpu_unsupported_reason;

    std::uint32_t frames = 0;            // frames with CPU timings
    std::uint32_t frames_with_gpu = 0;   // of those, frames whose GPU marks retired
    double frame_cpu_mean_ms = 0.0;      // host: the whole render call
    double frame_cpu_p95_ms = 0.0;
    double frame_gpu_mean_ms = 0.0;      // GPU: first mark to last
    double frame_gpu_p95_ms = 0.0;
    // Wall time the window spanned, so an agent can tell "10 frames in 0.5 s"
    // from "10 frames in 30 s" -- the second is a still camera being nudged,
    // not a frame rate.
    double window_wall_ms = 0.0;

    std::vector<RasterStageSample> stages;
    RasterAppliedState applied;

    // ★★ The instrument's own reasons to distrust it. Empty is the only clean
    //   result; anything here has to be resolved before the numbers are quoted.
    std::vector<std::string> warnings;
};


// Rolling window over the last N measured frames.
//
// ★★ A window, not a running total since startup: the question is always "what
//   does a frame cost UNDER THESE SETTINGS", and a lifetime average silently
//   mixes in every configuration the session has been through. Reset it, drive
//   the camera, read it.
class RasterStageAccumulator final {
public:
    static constexpr std::size_t kCapacity = 256;
    // 2026-09-13: 13 -> 14 (Reflection), 14 -> 15 (ReflectionFilter).
    // 2026-09-14: 15 -> 16 (Taa). Elle tutulur ve `RasterStageTimings.cpp`
    // icindeki static_assert onu RasterStage::Count ile karsilastirir -- yeni
    // bir stage ekleyip burayi unutmak DERLEME hatasi verir, sessiz bir kayma
    // degil. (Bu satir 2026-09-14'te tam olarak oyle yakalandi.)
    static constexpr std::size_t kStageCount = 16; // RasterStage::Count

    void reset();
    // One frame. `gpuAvailable` false means this frame's marks had not retired
    // yet -- the CPU half is still recorded, and frames_with_gpu stays behind
    // frames, which is normal and reported rather than hidden.
    void record(const double* cpuStageMs, double cpuFrameMs,
                const double* gpuStageMs, const bool* gpuStageRan,
                double gpuFrameMs, bool gpuAvailable,
                const RasterAppliedState& applied);

    RasterStageTimings snapshot(bool gpuSupported,
                                const char* gpuUnsupportedReason) const;
    bool empty() const { return m_count == 0; }

private:
    struct Frame {
        double cpuStage[kStageCount]{};
        double gpuStage[kStageCount]{};
        bool   gpuRan[kStageCount]{};
        double cpuFrameMs = 0.0;
        double gpuFrameMs = 0.0;
        bool   gpuAvailable = false;
    };
    Frame m_frames[kCapacity]{};
    std::size_t m_count = 0;
    std::size_t m_next = 0;
    RasterAppliedState m_applied{};
    bool m_appliedSeen = false;
    bool m_settingsChanged = false;
    double m_firstFrameStampMs = 0.0;
    double m_lastFrameStampMs = 0.0;
};

} // namespace Backend
