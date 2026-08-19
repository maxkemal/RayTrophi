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
#include "Backend/IViewportBackend.h"
#include "globals.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstring>
#include <mutex>
#include <vector>

// The interactive raster viewport. Owned by the UI layer; declared there as a
// file-local extern, so it is re-declared rather than pulled from a header.
extern std::unique_ptr<Backend::IViewportBackend> g_viewport_backend;

namespace rtapi {

namespace {

// One table, both directions. Two separate switch statements is how a name and
// an int drift apart.
struct ShadingEntry {
    int mode;                        // SceneUI::ViewportDisplaySettings::shading_mode
    const char* name;                // canonical name crossing the boundary
    Backend::ViewportMode backend;   // what the backend has to support
};
constexpr ShadingEntry kShadingModes[] = {
    { 0, "solid",    Backend::ViewportMode::Solid },
    { 1, "material", Backend::ViewportMode::MaterialPreview },
    { 2, "rendered", Backend::ViewportMode::Rendered },
    { 3, "matcap",   Backend::ViewportMode::Matcap },
};

const char* shadingName(int mode) {
    for (const ShadingEntry& e : kShadingModes)
        if (e.mode == mode) return e.name;
    return "unknown";
}

const ShadingEntry* shadingEntry(std::string name) {
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    // The panel labels mode 1 "Preview"; someone reading the UI will type that.
    if (name == "preview") name = "material";
    if (name == "render")  name = "rendered";
    for (const ShadingEntry& e : kShadingModes)
        if (name == e.name) return &e;
    return nullptr;
}

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

ViewportRenderResult renderViewportFrames(int count) {
    ViewportRenderResult out;
    if (!g_ctx) {
        out.error = "Engine context not bound";
        return out;
    }
    if (count <= 0) {
        out.error = "Count must be positive";
        return out;
    }
    if (!g_ctx->backend_ptr) {
        out.error = "No backend available";
        return out;
    }
    
    Backend::IBackend* backend = g_ctx->backend_ptr;
    
    // We are on the IPC handler thread, which evaluates via enqueueResult on the main thread.
    // We can block here and pump the backend for `count` progressive passes.
    // However, since we aren't calling SDL_RenderPresent, the UI will freeze for this duration.
    // For 16 frames this is practically instantaneous on modern GPUs.
    
    auto start_time = std::chrono::steady_clock::now();
    
    for (int i = 0; i < count; ++i) {
        if (backend->isAccumulationComplete()) {
            out.converged = true;
            break;
        }
        // Force the renderer to step. We pass nullptr for surface/window since we're just
        // accumulating internally in the backend, not presenting to SDL right this moment.
        // The display loop will catch up on the next natural frame.
        g_ctx->renderer.render_progressive_pass(nullptr, nullptr, g_ctx->scene, 1, 0);
        out.samples_rendered++;
    }
    
    auto end_time = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    out.ms_per_frame = (out.samples_rendered > 0) ? static_cast<float>(total_ms / out.samples_rendered) : 0.0f;
    out.success = true;
    return out;
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
    if (g_ctx->scene_ui_ptr)
        out.shading = shadingName(g_ctx->scene_ui_ptr->viewport_settings.shading_mode);
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

ViewportShadingInfo viewportShading() {
    ViewportShadingInfo out;
    if (!g_ctx || !g_ctx->scene_ui_ptr) return out;   // mode stays empty
    const auto& vs = g_ctx->scene_ui_ptr->viewport_settings;
    out.mode = shadingName(vs.shading_mode);
    out.matcap_preset = vs.matcap_preset;
    out.interactive_available =
        (g_viewport_backend != nullptr) ||
        (g_ctx->backend_ptr &&
         g_ctx->backend_ptr->supportsViewportMode(Backend::ViewportMode::Solid));
    return out;
}

Result setViewportShading(const std::string& mode, int matcap_preset) {
    if (!g_ctx) return Result::fail("Engine context not bound");
    if (!g_ctx->scene_ui_ptr) return Result::fail("No UI bound");

    const ShadingEntry* entry = shadingEntry(mode);
    if (!entry)
        return Result::fail("Unknown shading mode '" + mode +
                               "'. Valid: solid, material, rendered, matcap.");

    // Same support test the panel buttons run. Rendered always exists — it is
    // the pathtracer itself; the other three need a raster viewport.
    const bool supported =
        (entry->mode == 2) ||
        (g_viewport_backend != nullptr) ||
        (g_ctx->backend_ptr && g_ctx->backend_ptr->supportsViewportMode(entry->backend));
    if (!supported) {
        // ★ Refuse loudly instead of silently falling back to Rendered like the
        // panel does. A caller that asked for solid and got rendered without
        // being told would go on to measure the wrong image.
        return Result::fail(
            std::string("Shading mode '") + entry->name +
            "' needs the interactive raster viewport, which is not available on "
            "this machine (no Vulkan viewport backend). Only 'rendered' works here.");
    }

    if (matcap_preset >= 0) {
        if (matcap_preset > 9)
            return Result::fail("matcap_preset must be 0..9");
        g_ctx->scene_ui_ptr->viewport_settings.matcap_preset = matcap_preset;
        Backend::IBackend* matcapBackend = g_ctx->backend_ptr;
        if (entry->mode != 2 && g_viewport_backend &&
            g_viewport_backend.get() != g_ctx->backend_ptr) {
            matcapBackend = g_viewport_backend.get();
        }
        if (matcapBackend) matcapBackend->setInteractiveViewportMatcapPreset(matcap_preset);
    }

    g_ctx->scene_ui_ptr->viewport_settings.shading_mode = entry->mode;
    if (entry->mode != 2 && g_viewport_backend != nullptr)
        g_viewport_raster_rebuild_pending = true;

    // ★★ Without this the next probe measures the frame accumulated in the mode
    // you just LEFT, and reports it as a valid measurement of the new one.
    g_ctx->start_render = true;
    g_ctx->renderer.resetCPUAccumulation();
    if (g_ctx->backend_ptr) g_ctx->backend_ptr->resetAccumulation();
    return Result::success();
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
