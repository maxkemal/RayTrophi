/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          Api/RtApiViewport.cpp
 * Author:        Kemal Demirtas
 * License:       MIT
 * =========================================================================
 * Viewport measurement surface: what the agent can ASK about the frame it is
 * looking at, rather than what it can save to disk and squint at.
 *
 * ★★★ Why this file exists. On 2026-08-16 a volume-box re-entry bug (black band
 * plus a pathtrace cost explosion) took an entire session, and two of the lost
 * rounds were pure instrumentation gaps, not physics:
 *
 *   - Every counter had to be copied out of the panel by a human. Driving the
 *     simulation over IPC left the viewport idle, so `volume_rays` read 0 and
 *     that zero was indistinguishable from "the scene is cheap".
 *   - One full round was spent analysing a metrics dump taken from a frame with
 *     no fire in it. Nothing in the reachable API could have caught that.
 *   - "Is there a black band?" was answered by eye. The question was numeric all
 *     along: what fraction of pixels in this region is below 0.001 luminance?
 *
 * CLAUDE.md rule 1 calls a capability that only exists in a panel untestable.
 * That is precisely what the viewport was.
 *
 * ★★ Data-model rule (docs/dev/IPC_SECURITY_PERFORMANCE.md): only names, ids
 * and VALUES cross the IPC boundary. Nothing here returns a handle, an SDL
 * surface or a backend pointer. The captured frame is copied into a buffer this
 * module owns, so no caller can outlive the engine's own image.
 */

#include "Api/RtApiInternal.h"
#include "Backend/IBackend.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <mutex>
#include <vector>

namespace rtapi {

namespace {

// The captured frame. Guarded because the display loop writes it and IPC
// queries read it; both run on the UI thread today, but the lock costs nothing
// and removes the assumption.
// Read by the display loop every frame to decide whether to pay for a surface
// conversion, so it must not require the mutex.
std::atomic<bool>   g_capture_requested{false};
std::mutex          g_frame_mutex;
std::vector<uint8_t> g_frame_rgba;     // tightly packed RGBA8
int                 g_frame_width = 0;
int                 g_frame_height = 0;
bool                g_capture_enabled = false;
bool                g_frame_available = false;

// Rec.709 luma on the already-tonemapped display frame. This deliberately
// measures WHAT THE VIEWER SEES, not scene-referred radiance: the failure being
// hunted is "the pixel is black on screen".
inline float luma(uint8_t r, uint8_t g, uint8_t b) {
    return (0.2126f * static_cast<float>(r) +
            0.7152f * static_cast<float>(g) +
            0.0722f * static_cast<float>(b)) / 255.0f;
}

const char* backendName(const Backend::IBackend* b) {
    if (!b) return "";
    switch (b->getInfo().type) {
        case Backend::BackendType::VULKAN_RT:      return "vulkan";
        case Backend::BackendType::VULKAN_COMPUTE: return "vulkan_compute";
        case Backend::BackendType::OPTIX:          return "optix";
        case Backend::BackendType::METAL:          return "metal";
        default:                                   return "cpu";
    }
}

} // namespace

void publishViewportFrame(const void* pixels, int width, int height,
                          int pitch_bytes) {
    if (!pixels || width <= 0 || height <= 0) return;
    std::lock_guard<std::mutex> lock(g_frame_mutex);
    if (!g_capture_enabled) return;   // opt-in: no copy cost unless asked for

    const int packed_pitch = width * 4;
    const int pitch = pitch_bytes > 0 ? pitch_bytes : packed_pitch;
    g_frame_rgba.resize(static_cast<size_t>(packed_pitch) * static_cast<size_t>(height));
    const uint8_t* src = static_cast<const uint8_t*>(pixels);
    for (int y = 0; y < height; ++y) {
        std::memcpy(g_frame_rgba.data() + static_cast<size_t>(y) * packed_pitch,
                    src + static_cast<size_t>(y) * pitch,
                    static_cast<size_t>(packed_pitch));
    }
    g_frame_width = width;
    g_frame_height = height;
    g_frame_available = true;
}

bool viewportCaptureEnabled() {
    return g_capture_requested.load(std::memory_order_relaxed);
}

Result setViewportCapture(bool enabled) {
    g_capture_requested.store(enabled, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(g_frame_mutex);
    g_capture_enabled = enabled;
    if (!enabled) {
        // Drop the buffer AND the availability flag together. Leaving a stale
        // frame readable after capture is switched off is exactly the "a
        // default is not a measurement" trap: the next probe would silently
        // describe an old frame.
        g_frame_rgba.clear();
        g_frame_rgba.shrink_to_fit();
        g_frame_available = false;
        g_frame_width = 0;
        g_frame_height = 0;
    }
    return Result::success();
}

ViewportStatusInfo viewportStatus() {
    ViewportStatusInfo out;
    {
        std::lock_guard<std::mutex> lock(g_frame_mutex);
        out.capture_enabled = g_capture_enabled;
        out.frame_available = g_frame_available;
        out.width = g_frame_width;
        out.height = g_frame_height;
    }
    if (!g_ctx) return out;                 // available stays false
    Backend::IBackend* backend = g_ctx->backend_ptr;
    if (!backend) return out;

    out.available = true;
    out.backend = backendName(backend);
    out.samples = backend->getCurrentSampleCount();
    out.accumulation_complete = backend->isAccumulationComplete();
    out.ms_per_sample = backend->getMillisecondsPerSample();
    out.rendering_active = g_ctx->render_settings.is_rendering_active;
    return out;
}

ViewportProbeInfo probeViewportFrame(const ViewportProbeRegion& region,
                                     float black_threshold) {
    ViewportProbeInfo out;
    std::lock_guard<std::mutex> lock(g_frame_mutex);
    if (!g_frame_available || g_frame_width <= 0 || g_frame_height <= 0)
        return out;                          // available stays false

    int x0 = std::clamp(region.x, 0, g_frame_width);
    int y0 = std::clamp(region.y, 0, g_frame_height);
    int w  = region.width  > 0 ? region.width  : g_frame_width  - x0;
    int h  = region.height > 0 ? region.height : g_frame_height - y0;
    w = std::clamp(w, 0, g_frame_width  - x0);
    h = std::clamp(h, 0, g_frame_height - y0);
    if (w <= 0 || h <= 0) return out;

    const float threshold = std::max(black_threshold, 0.0f);
    double sum = 0.0;
    float lo = 1.0f, hi = 0.0f;
    uint32_t black = 0, nans = 0, counted = 0;

    for (int y = y0; y < y0 + h; ++y) {
        const uint8_t* row = g_frame_rgba.data() +
                             (static_cast<size_t>(y) * g_frame_width + x0) * 4;
        for (int x = 0; x < w; ++x) {
            const uint8_t* p = row + static_cast<size_t>(x) * 4;
            const float l = luma(p[0], p[1], p[2]);
            // ★ A NaN is neither black nor lit, and it vanishes inside a mean.
            // Count it separately or a whole failure class stays invisible.
            if (!std::isfinite(l)) { ++nans; continue; }
            sum += l;
            lo = std::min(lo, l);
            hi = std::max(hi, l);
            if (l <= threshold) ++black;
            out.histogram[std::min<size_t>(static_cast<size_t>(l * 8.0f), 7u)]++;
            ++counted;
        }
    }

    const uint32_t total = static_cast<uint32_t>(w) * static_cast<uint32_t>(h);
    out.available = true;
    out.width = w;
    out.height = h;
    out.pixels = total;
    out.mean_luminance = counted ? static_cast<float>(sum / counted) : 0.0f;
    out.min_luminance = counted ? lo : 0.0f;
    out.max_luminance = counted ? hi : 0.0f;
    // Denominator is the FULL region for both fractions, so black_fraction and
    // nan_fraction stay comparable and sum meaningfully.
    out.black_fraction = total ? static_cast<float>(black) / static_cast<float>(total) : 0.0f;
    out.nan_fraction   = total ? static_cast<float>(nans)  / static_cast<float>(total) : 0.0f;
    return out;
}

} // namespace rtapi
