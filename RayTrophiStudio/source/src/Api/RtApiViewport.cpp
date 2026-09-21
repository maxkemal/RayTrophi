#include "Viewport/MaterialPreviewVolumeShadow.h"
#include "Api/RtApiRasterDiagnostics.h"
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
#include <functional>
#include <mutex>
#include <string>
#include <vector>
#include "stb_image_write.h"

// The interactive raster viewport. Owned by the UI layer; declared there as a
// file-local extern, so it is re-declared rather than pulled from a header.
extern std::unique_ptr<Backend::IViewportBackend> g_viewport_backend;

// scene_ui.h'deki sahne-yukleme kalkani. Basligi cekmek yerine burada
// bildiriliyor: RtApiViewport.cpp UI basliklarina bagli degil ve oyle kalmali.
extern bool g_scene_load_solid_guard;

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

// Ayni sozlesme: sinirdan ISIM gecer, panelin tam sayisi degil.
struct QualityEntry {
    RasterViewportQualityPreset preset;
    const char* name;
};
constexpr QualityEntry kQualityPresets[] = {
    { RasterViewportQualityPreset::Auto,        "auto" },
    { RasterViewportQualityPreset::Performance, "performance" },
    { RasterViewportQualityPreset::Balanced,    "balanced" },
    { RasterViewportQualityPreset::Quality,     "quality" },
    { RasterViewportQualityPreset::Full,        "full" },
};

const char* qualityName(RasterViewportQualityPreset p) {
    for (const QualityEntry& e : kQualityPresets)
        if (e.preset == p) return e.name;
    return "unknown";
}

const QualityEntry* qualityEntry(std::string name) {
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    // Panelde "Full (no proxy)" yaziyor; UI'yi okuyan bunu yazar.
    if (name == "full (no proxy)" || name == "no_proxy" || name == "noproxy")
        name = "full";
    for (const QualityEntry& e : kQualityPresets)
        if (name == e.name) return &e;
    return nullptr;
}

struct PreviewLightingEntry {
    MaterialPreviewLightingPreset preset;
    const char* name;
};
constexpr PreviewLightingEntry kPreviewLightingPresets[] = {
    { MaterialPreviewLightingPreset::Classic, "three_point" },
    { MaterialPreviewLightingPreset::Scene,   "scene"       },
};

const char* previewLightingName(MaterialPreviewLightingPreset p) {
    return p == MaterialPreviewLightingPreset::Scene ? "scene" : "three_point";
}

const PreviewLightingEntry* previewLightingEntry(std::string name) {
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    // UI etiketleri ve eski API adlari dosya/proje uyumlulugu icin alias'tir.
    if (name == "scene (realtime)" || name == "scene (real lights)" ||
        name == "real" || name == "real_lights" || name == "realtime")
        name = "scene";
    if (name == "3 point" || name == "3-point" ||
        name == "3 point (material preview)" || name == "three point" ||
        name == "classic" || name == "studio" || name == "outdoor")
        name = "three_point";
    for (const PreviewLightingEntry& e : kPreviewLightingPresets)
        if (name == e.name) return &e;
    return nullptr;
}

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

// Every backend that could own the raster viewport. g_viewport_backend is the
// dedicated one; on machines without it the render backend serves Solid itself.
// Missing either of them is how a setting reaches "the other" object and looks
// like it did nothing.
// Not file-local any more: RtApiRayFusion.cpp asks the same question, and a
// second copy of "which objects might be the viewport" is exactly how one
// caller ends up reading a backend the other never looked at.
void forEachViewportBackend(const std::function<void(Backend::IBackend&)>& fn) {
    Backend::IBackend* viewport = g_viewport_backend.get();  // IViewportBackend : IBackend
    if (viewport) fn(*viewport);
    if (g_ctx && g_ctx->backend_ptr && g_ctx->backend_ptr != viewport) {
        fn(*g_ctx->backend_ptr);
    }
}

Result setViewportCapture(bool enabled) {
    g_capture_requested.store(enabled, std::memory_order_relaxed);
    // ★★★ Asenkron sunum ile probe DOGRUDAN CELISIR: halka bilerek ESKI ama
    // tamamlanmis bir slotu yayinlar, yani capture acikken bir script kendi
    // sahne duzenlemesinden ONCE kaydedilmis bir kareyi olcebilir ve goruntude
    // bunu soyleyen hicbir sey olmaz. Capture acikken gecikmeyi belirlilige
    // takas ediyoruz.
    forEachViewportBackend([enabled](Backend::IBackend& b) {
        b.setInteractiveViewportSynchronousPresent(enabled);
    });
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

namespace {
// Ana dongu yazar, IPC okur. Iki ayri thread, o yuzden kilit.
std::mutex g_display_timing_mutex;
DisplayPathTiming g_display_timing;
uint64_t g_display_timing_frames = 0;
}  // namespace

void noteDisplayPathTiming(const DisplayPathTiming& t) {
    std::lock_guard<std::mutex> lock(g_display_timing_mutex);
    g_display_timing = t;
    ++g_display_timing_frames;
}

ViewportFrameTelemetryInfo viewportFrameTelemetry() {
    ViewportFrameTelemetryInfo out;

    // ★★ Ekran yolu once doldurulur: raster viewport hic kosmamis olsa bile
    // (Rendered modu) ana dongu pikselleri yine CPU'da tasiyor. Bunu `found`
    // erken donusunun arkasina koymak, olcumun tam da gerekli oldugu modda
    // kaybolmasi demek olurdu.
    {
        std::lock_guard<std::mutex> lock(g_display_timing_mutex);
        if (g_display_timing_frames > 0) {
            out.display_available = true;
            out.display_post_ms = g_display_timing.post_ms;
            out.display_texture_upload_ms = g_display_timing.texture_upload_ms;
            out.display_loop_period_ms = g_display_timing.loop_period_ms;
            out.display_post_was_noop_copy = g_display_timing.post_was_noop_copy;
            out.display_frames = g_display_timing_frames;
        }
    }

    Backend::RasterFrameTelemetry t;
    bool found = false;
    forEachViewportBackend([&](Backend::IBackend& b) {
        if (found) return;
        found = b.getInteractiveViewportFrameTelemetry(t);
    });
    if (!found) return out;                 // available stays false

    out.available = true;
    out.async_present = t.async_present;
    out.synchronous_present = t.synchronous_present;
    out.slot_count = static_cast<int>(t.slot_count);
    out.width = static_cast<int>(t.width);
    out.height = static_cast<int>(t.height);
    out.frame_ms = t.frame_ms;
    out.cpu_record_ms = t.cpu_record_ms;
    out.slot_wait_ms = t.slot_wait_ms;
    out.submit_ms = t.submit_ms;
    out.image_readback_ms = t.image_readback_ms;
    out.host_read_ms = t.host_read_ms;
    out.present_ms = t.present_ms;
    out.frames_submitted = t.frames_submitted;
    out.frames_consumed = t.frames_consumed;
    out.stale_presents = t.stale_presents;
    out.slot_waits = t.slot_waits;
    out.blocking_seeds = t.blocking_seeds;
    out.resource_drains = t.resource_drains;
    out.present_latency_frames = static_cast<int>(t.present_latency_frames);
    out.global_instance_buffer  = t.global_instance_buffer;
    out.gpu_culling             = t.gpu_culling;
    out.depth_prepass           = t.depth_prepass;
    out.total_instances         = t.total_instances;
    out.cull_mesh_count         = t.cull_mesh_count;
    out.draw_calls              = t.draw_calls;
    out.visible_triangles       = t.visible_triangles;
    out.full_triangles          = t.full_triangles;
    out.proxy_triangles         = t.proxy_triangles;
    out.full_instances          = t.full_instances;
    out.proxy_instances         = t.proxy_instances;
    out.scatter_triangle_target = t.scatter_triangle_target;
    // Teshis sayaclari: `available` false iken de tasinir -- surucu kaybi
    // ring'i olduren seyin ta kendisi, yani "olcum yok" dedigi an tam da
    // okunmasi gereken andir.
    out.stale_descset_rebuilds  = t.stale_descset_rebuilds;
    out.device_lost             = t.device_lost;
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

Result setRtShadow(bool enabled) {
    bool applied = false;
    forEachViewportBackend([enabled, &applied](Backend::IBackend& b) {
        applied = b.setRtShadow(enabled) || applied;
    });
    if (!applied)
        return Result::fail("no viewport backend owns an RT shadow pass");
    SCENE_LOG_INFO(enabled
        ? "[Viewport] Same-frame RT directional shadows requested; viewport.rt_shadow reports readiness."
        : "[Viewport] RT shadows disabled; cascade shadows active.");
    return Result::success();
}

RtShadowInfo rtShadow() {
    RtShadowInfo out;
    forEachViewportBackend([&out](Backend::IBackend& b) {
        bool supported = false, ready = false;
        uint32_t rays = 0, cascadesReplaced = 0;
        std::string reason;
        b.getRtShadowStatus(supported, ready, rays, cascadesReplaced, reason);
        out.supported = out.supported || supported;
        out.ready = out.ready || ready;
        out.enabled = out.enabled || b.rtShadowAllowed();
        if (rays > out.rays) out.rays = rays;
        if (cascadesReplaced > out.cascades_replaced)
            out.cascades_replaced = cascadesReplaced;
        if (out.reason.empty()) out.reason = reason;
    });
    return out;
}

RasterTimingInfo rasterTimings() {
    RasterTimingInfo out;
    bool found = false;
    forEachViewportBackend([&out, &found](Backend::IBackend& b) {
        if (found) return; // first viewport backend owns the window
        Backend::RasterStageTimings t;
        if (!b.getRasterStageTimings(t)) return;
        found = true;
        out.available = t.available;
        out.gpu_supported = t.gpu_supported;
        out.gpu_unsupported_reason = t.gpu_unsupported_reason;
        out.frames = t.frames;
        out.frames_with_gpu = t.frames_with_gpu;
        out.frame_cpu_mean_ms = t.frame_cpu_mean_ms;
        out.frame_cpu_p95_ms = t.frame_cpu_p95_ms;
        out.frame_gpu_mean_ms = t.frame_gpu_mean_ms;
        out.frame_gpu_p95_ms = t.frame_gpu_p95_ms;
        out.window_wall_ms = t.window_wall_ms;
        for (const auto& stage : t.stages) {
            RasterStageInfo s;
            s.name = stage.name;
            s.cpu_mean_ms = stage.cpu_mean_ms;
            s.cpu_p95_ms = stage.cpu_p95_ms;
            s.gpu_mean_ms = stage.gpu_mean_ms;
            s.gpu_p95_ms = stage.gpu_p95_ms;
            s.frames_ran = stage.frames_ran;
            out.stages.push_back(std::move(s));
        }
        out.shading = t.applied.shading;
        out.screen_gi = t.applied.screen_gi;
        out.reflection = t.applied.reflection;
        out.quality_preset = t.applied.quality_preset;
        out.lighting_preset = t.applied.lighting_preset;
        out.width = t.applied.width;
        out.height = t.applied.height;
        out.depth_prepass = t.applied.depth_prepass;
        out.gpu_culling = t.applied.gpu_culling;
        out.global_instance_buffer = t.applied.global_instance_buffer;
        out.rt_shadow_requested = t.applied.rt_shadow_requested;
        out.rt_shadow_ready = t.applied.rt_shadow_ready;
        out.rt_cascades_replaced = t.applied.rt_cascades_replaced;
        out.directional_cascades = t.applied.directional_cascades;
        out.shadowed_lights = t.applied.shadowed_lights;
        out.scene_lights = t.applied.scene_lights;
        out.volume_count = t.applied.volume_count;
        out.visible_triangles = t.applied.visible_triangles;
        out.total_instances = t.applied.total_instances;
        out.draw_calls = t.applied.draw_calls;
        out.warnings = t.warnings;
    });
    if (!found)
        out.warnings.push_back("no viewport backend owns a raster timing window");
    return out;
}

Result resetRasterTimings() {
    bool applied = false;
    forEachViewportBackend([&applied](Backend::IBackend& b) {
        applied = b.resetRasterStageTimings() || applied;
    });
    if (!applied)
        return Result::fail("no viewport backend owns a raster timing window");
    return Result::success();
}

Result setRasterDepthPrepass(bool enabled) {
    bool applied = false;
    forEachViewportBackend([enabled, &applied](Backend::IBackend& b) {
        applied = b.setRasterDepthPrepass(enabled) || applied;
    });
    if (!applied)
        return Result::fail("no viewport backend owns a raster depth prepass");
    if (!enabled) {
        // ★★★ Aranarak bulunmali: kapaliyken sahne DOGRU cizilir, yalniz her
        //   ortusen foliage katmani tam scene shader'ini yeniden kosar.
        SCENE_LOG_INFO("[Viewport] Derinlik on gecisi istegi KAPALI; RT golge "
                       "gerektiginde on gecisi zorlar. Son kaydedilen kare icin "
                       "viewport.raster_depth_prepass effective/forced_by_rt_shadow alanlarini okuyun.");
    } else {
        SCENE_LOG_INFO("[Viewport] Derinlik on gecisi ACIK.");
    }
    return Result::success();
}

bool rasterDepthPrepass() {
    bool enabled = false;
    forEachViewportBackend([&enabled](Backend::IBackend& b) {
        enabled = b.rasterDepthPrepassAllowed() || enabled;
    });
    return enabled;
}

RasterDepthPrepassInfo rasterDepthPrepassStatus() {
    RasterDepthPrepassInfo out;
    out.enabled = rasterDepthPrepass();
    const auto timings = rasterTimings();
    out.observed = timings.available && timings.frames > 0;
    out.effective = out.observed && timings.depth_prepass;
    out.forced_by_rt_shadow = out.effective && timings.rt_shadow_ready;
    return out;
}

Result setRasterGpuInstancing(bool enabled) {
    bool applied = false;
    forEachViewportBackend([enabled, &applied](Backend::IBackend& b) {
        applied = b.setRasterGpuInstancing(enabled) || applied;
    });
    if (!applied)
        return Result::fail("no viewport backend owns a raster instance layout");
    if (!enabled) {
        // ★★★★ Aranarak bulunmali. Kapali oldugunda sahne DOGRU cizilir, yalniz
        //   yavas: GPU culling ve scatter LOD proxy'leri devre disi kalir ve
        //   gorunur kume her degistiginde tam kare halkasi drenaji odenir.
        //   Aylar sonra "realtime neden yavas" diye sorulursa cevap burasi.
        SCENE_LOG_WARN("[Viewport] Raster GPU instancing KAPATILDI: global instance "
                       "buffer, GPU culling ve scatter LOD proxy'leri devre disi. "
                       "Sahne dogru gorunur, yalnizca yavastir. "
                       "viewport.frame_telemetry icinde gpu_culling, visible_triangles "
                       "ve resource_drains alanlarini okuyun.");
    } else {
        SCENE_LOG_INFO("[Viewport] Raster GPU instancing ACIK.");
    }
    return Result::success();
}

bool rasterGpuInstancing() {
    bool enabled = false;
    forEachViewportBackend([&enabled](Backend::IBackend& b) {
        enabled = b.rasterGpuInstancingAllowed() || enabled;
    });
    return enabled;
}

Result setSceneLoadGuard(bool enabled) {
    // ★ Durum degismediyse de basarilidir; cagiran idempotent kurabilsin.
    const bool previous = g_scene_load_solid_guard;
    g_scene_load_solid_guard = enabled;
    if (previous != enabled) {
        if (enabled) {
            SCENE_LOG_INFO("[Viewport] Sahne yukleme kalkani ACIK: acilislar Solid'e duser.");
        } else {
            // ★★★★ Bu satir ARANARAK bulunmali. Kapali kalkan, aylar sonra
            //   gelen bir device-lost raporunun sessiz sebebi olabilir.
            SCENE_LOG_WARN(
                "[Viewport] Sahne yukleme kalkani KAPATILDI. Proje/template acilislari artik "
                "Material/Rendered bagliyken kosacak -- bu BILEREK ariza uretmek icindir "
                "(device-lost tekrar uretimi). viewport.frame_telemetry icindeki "
                "stale_descset_rebuilds ve device_lost alanlarini okuyun.");
        }
    }
    return Result::success();
}

bool sceneLoadGuard() {
    return g_scene_load_solid_guard;
}

ViewportQualityInfo viewportQuality() {
    ViewportQualityInfo out;
    const RasterViewportQualityPreset preset =
        ::render_settings.raster_viewport_quality_preset;
    out.preset = qualityName(preset);
    out.scatter_lod_split =
        (preset != RasterViewportQualityPreset::Full);
    out.raster_viewport_available =
        (g_viewport_backend != nullptr) ||
        (g_ctx && g_ctx->backend_ptr &&
         g_ctx->backend_ptr->supportsViewportMode(Backend::ViewportMode::Solid));
    out.shadow_atlas_resolution = rasterShadowAtlasSize();
    out.shadow_tile_resolution = rasterShadowTileSize(preset);
    const auto volumeBudget = Backend::volumeShadowBudget(preset);
    out.volume_shadow_tile_resolution = static_cast<int>(volumeBudget.tileSize);
    out.volume_shadow_depth_layers = static_cast<int>(volumeBudget.layers);
    out.volume_shadow_steps = static_cast<int>(volumeBudget.steps);
    const int tilesPerRow = out.shadow_atlas_resolution / out.shadow_tile_resolution;
    out.shadow_tile_capacity = tilesPerRow * tilesPerRow;
    out.shadow_light_budget = rasterShadowLightBudget(preset);
    out.shadow_pcf_samples = rasterShadowPcfSamples(preset);
    out.directional_shadow_cascades =
        rasterDirectionalShadowCascades(preset);
    out.scene_pbr_shader = "ggx";
    out.opaque_core_parity = "rt_aligned";
    out.material_graph_surface = "bounded";
    out.clearcoat = "iridescent_lobe";
    out.subsurface = "radius_profile_approx";
    out.translucency = "thin_surface_approx";
    out.surface_anisotropy = "unsupported_abi_conflict";
    out.transparency = "unsorted_alpha";
    out.transmission = "screen_space_thickness";
    out.resin_interior = "procedural_interior_approx";
    out.sdf_surface = "shared_nanovdb_depth_pbr";
    out.volumes = "shared_vdb_dense_single_scatter";
    return out;
}

Result setViewportQuality(const std::string& preset) {
    if (!g_ctx) return Result::fail("Engine context not bound");

    const QualityEntry* entry = qualityEntry(preset);
    if (!entry)
        return Result::fail("Unknown viewport quality preset '" + preset +
                            "'. Valid: auto, performance, balanced, quality, full.");

    ::render_settings.raster_viewport_quality_preset = entry->preset;

    // ★★ Preset scatter LOD ayrimini degistirir, ve o ayrim MESH KURULUMUNDA
    //   karara baglanir (hangi cizim yuvasi var, hangi proxy baglanir). Sahneyi
    //   yeniden kurmadan degistirmek, bir sonraki karede eski bagi kullanir --
    //   yani cagri basarili doner ve HICBIR SEY degismez.
    if (g_viewport_backend != nullptr)
        g_viewport_raster_rebuild_pending = true;

    // Panel combo'sunun yaptigi ile ayni: birikimi sifirla ki bir sonraki probe
    // ONCEKI preset'te birikmis kareyi olcup yeni preset'in sonucu sanmasin.
    g_ctx->start_render = true;
    g_ctx->renderer.resetCPUAccumulation();
    if (g_ctx->backend_ptr) g_ctx->backend_ptr->resetAccumulation();
    return Result::success();
}

Result setViewportDepthOfField(bool enabled, float max_coc_pixels, int max_taps) {
    if (!g_ctx) return Result::fail("Engine context not bound");
    // ★★★ Degerler KIRPILMAZ, REDDEDILIR: sessizce kirpilan bir kadran, ajanin
    //   "ayarladim" deyip baska bir degeri olcmesi demektir.
    if (!(max_coc_pixels >= 1.0f && max_coc_pixels <= 128.0f))
        return Result::fail("max_coc_pixels must be in [1, 128] pixels");
    if (max_taps < 8 || max_taps > 128)
        return Result::fail("max_taps must be in [8, 128]");

    ::render_settings.realtime_depth_of_field = enabled;
    ::render_settings.realtime_dof_max_coc = max_coc_pixels;
    ::render_settings.realtime_dof_max_taps = max_taps;

    // Post gecisi her karede push-constant'tan okur; birikim sifirlanmasi
    // gerekmez, ama durgun viewport yeni kare cizmez -> dirty.
    g_ctx->start_render = true;
    if (g_ctx->backend_ptr) g_ctx->backend_ptr->resetAccumulation();
    g_ctx->renderer.resetCPUAccumulation();
    return Result::success();
}

Result setViewportTaa(bool enabled, int samples) {
    if (!g_ctx) return Result::fail("Engine context not bound");
    // ★★★ KIRPMA YOK, REDDETME VAR: sessizce kirpilan bir kadran, cagiranin
    //   "64 ornek istedim" deyip 16'yi olcmesi demektir.
    if (samples < 1 || samples > 256)
        return Result::fail("samples must be in [1, 256]");
    ::render_settings.realtime_taa = enabled;
    ::render_settings.realtime_taa_samples = samples;
    // Birikimi backend sifirlar (setRenderParams); burada yalnizca durgun
    // viewport'un yeni bir kare cizmesini istiyoruz.
    g_ctx->start_render = true;
    return Result::success();
}

ViewportTaaInfo viewportTaa() {
    ViewportTaaInfo info;
    info.enabled = ::render_settings.realtime_taa;
    info.target_samples = ::render_settings.realtime_taa_samples;
    if (!g_ctx) {
        info.inactive_reason = "Engine context not bound";
        return info;
    }
    // ★★★★ OLCUM raster viewport'u SUREN adapter'dan okunur ve backend'ler
    //   uzerinde OR'lanMAZ: RT adapter'inin TAA'si yoktur ve olsaydi bile
    //   ekranda o cizmiyor olurdu. Bu alanin tamami "ayar ile ekranda olan
    //   ayrisabilir mi" sorusunu cevaplamak icin var.
    bool reported = false;
    forEachViewportBackend([&](Backend::IBackend& backend) {
        if (reported) return;
        Backend::IBackend::RasterTaaStatus st{};
        if (!backend.getRasterTaaStatus(st)) return;
        reported = true;
        info.supported = st.supported;
        info.accumulated_samples = static_cast<int>(st.accumulated_samples);
        info.converged = st.converged;
        info.last_ms = static_cast<float>(st.last_ms);
        info.inactive_reason = st.inactive_reason;
    });
    if (!reported) info.inactive_reason = "no raster viewport backend is alive";
    return info;
}

// ★★ Nokta sayisi alan modundan TURETILIR; iki yerde tutulursa "Zone 21
//   seciliyken 9 nokta" gibi kendi icinde celisen bir rapor cikardi.
//   (Overlay 3x3 cizer, mod 2 ise 5x5.)
static int afPointCount(int area_mode) { return (area_mode == 2) ? 25 : 9; }

Result setViewportAf(bool enabled, int area_mode, int focus_mode, int selected_point) {
    if (!g_ctx) return Result::fail("Engine context not bound");
    if (!g_ctx->scene_ui_ptr) return Result::fail("UI is not available");
    // ★★★ KIRPMA YOK, REDDETME VAR: sessizce kirpilan bir indeks, ajanin
    //   "21. noktayi sectim" deyip merkeze odaklanmasi demektir.
    if (area_mode < 0 || area_mode > 4)
        return Result::fail("area_mode must be in [0, 4] (single, zone9, zone21, wide, center_weighted)");
    if (focus_mode < 0 || focus_mode > 2)
        return Result::fail("focus_mode must be in [0, 2] (mf, af_s, af_c)");
    const int count = afPointCount(area_mode);
    if (selected_point < 0 || selected_point >= count)
        return Result::fail("selected_point must be in [0, " + std::to_string(count - 1) +
                            "] for this area_mode");

    auto& vs = g_ctx->scene_ui_ptr->viewport_settings;
    vs.show_af_points = enabled;
    vs.af_mode = area_mode;
    vs.focus_mode = focus_mode;
    vs.af_selected_point = selected_point;
    g_ctx->start_render = true;
    return Result::success();
}

ViewportAfInfo viewportAf() {
    ViewportAfInfo info;
    if (!g_ctx || !g_ctx->scene_ui_ptr) {
        info.inactive_reason = "UI is not available";
        return info;
    }
    const auto& vs = g_ctx->scene_ui_ptr->viewport_settings;
    info.enabled = vs.show_af_points;
    info.area_mode = vs.af_mode;
    info.focus_mode = vs.focus_mode;
    info.selected_point = vs.af_selected_point;
    info.point_count = afPointCount(vs.af_mode);

    // ★★★ Kapilar overlay'in KENDI erken cikislariyla ayni sirada olmali;
    //   ayri tutulursa rapor ile cizim ayrisir ve "acik ama gorunmuyor"
    //   sorusunun cevabi hicbir yerde olmaz.
    if (!info.enabled)                 info.inactive_reason = "disabled by setting";
    else if (!vs.show_camera_hud)      info.inactive_reason = "camera HUD is off (viewport draws no overlay)";
    else if (!g_ctx->scene.camera)     info.inactive_reason = "no active camera";
    else if (!g_ctx->scene.bvh)        info.inactive_reason = "scene has no BVH - AF cannot measure distance";
    else if (g_ctx->is_animation_mode) info.inactive_reason = "sequence render is running (AF probes are disabled)";
    else                               info.active = true;
    return info;
}

ViewportDepthOfFieldInfo viewportDepthOfField() {
    ViewportDepthOfFieldInfo info;
    if (!g_ctx) return info;
    info.enabled = ::render_settings.realtime_depth_of_field;
    info.max_coc_pixels = ::render_settings.realtime_dof_max_coc;
    info.max_taps = ::render_settings.realtime_dof_max_taps;

    const Camera* cam = g_ctx->scene.camera.get();
    if (cam) {
        // ★★★ ETKIN aciklik: kamera anahtari kapaliysa 0. Fiziksel degeri
        //   raporlamak, "aciklik 0.2 ama ekran keskin" gibi kendi icinde
        //   celisen bir rapor uretirdi.
        info.camera_aperture = cam->effectiveAperture();
        info.camera_depth_of_field = cam->depth_of_field;
        info.camera_focus_distance = static_cast<float>(cam->focus_dist);
    }
    // ★★★ "Aktif mi" sorusunun cevabi TEK BIR bayrak degildir; dort kapi var
    //   ve hangisinin kapali oldugunu SOYLEMEK zorundayiz. Yalnizca `false`
    //   dondurmek, cagiran tarafta "bilgi eksikligi"ni "olctum, sifir"a
    //   cevirir -- bu deponun adi konmus hata sinifi.
    // ★★ shading_mode: 0=Solid, 1=Material/Realtime raster, 2=Rendered, 3=Matcap.
    //   DoF gecisi RASTER yolundadir, yani mod 1. Mod 2 (Rendered) path
    //   tracer'dir ve DoF'u zaten lens ornekleyerek yapar -- oradaki bulaniklik
    //   bu ayardan etkilenmez, karistirmak "acik ama calismiyor" uretir.
    const bool materialMode = g_ctx->scene_ui_ptr &&
                              g_ctx->scene_ui_ptr->viewport_settings.shading_mode == 1;
    if (!info.enabled)                    info.inactive_reason = "disabled by setting";
    else if (!materialMode)               info.inactive_reason = "shading mode is not Material";
    else if (cam && cam->orthographic)    info.inactive_reason = "orthographic camera has no lens";
    else if (cam && !cam->depth_of_field)
        // ★★★★ BESINCI KAPI (2026-09-06 II). Kamera anahtari kapaliyken
        //   aciklik dolu olabilir; "aperture is 0" demek YANLIS bir teshis
        //   olurdu ve kullaniciyi f-stop kadranini cevirmeye gonderirdi --
        //   yani hicbir sey degistirmeyen bir eyleme.
        info.inactive_reason = "camera depth of field is off (camera.set_depth_of_field)";
    else if (info.camera_aperture <= 1e-5f)
        info.inactive_reason = "camera aperture is 0 (open a lens with camera.set_aperture)";
    else if (info.camera_focus_distance <= 1e-4f)
        info.inactive_reason = "camera focus distance is 0";
    else                                  info.active = true;
    return info;
}

ViewportPreviewLightingInfo viewportPreviewLighting() {
    ViewportPreviewLightingInfo out;

    // ★★★ GPU'ya GERCEKTEN giden goruntuleme donusumu. Kaynak `post.get`
    //   DEGIL, shader'larin okudugu aynanin ta kendisi -- yoksa bu alan
    //   "ayar boyle" der ve onizlemenin onu gorup gormedigini soylemez.
    out.display_exposure = g_display_post.exposure;
    out.display_gamma = g_display_post.gamma;
    out.display_saturation = g_display_post.saturation;
    out.display_color_temperature = g_display_post.color_temperature;
    out.display_vignette_strength = g_display_post.vignette_strength;
    out.display_vignette_enabled = g_display_post.vignette_enabled != 0;
    switch (g_display_post.tone_mapping) {
        case 0: out.display_tone_mapping = "agx"; break;
        case 1: out.display_tone_mapping = "aces_fitted"; break;
        case 2: out.display_tone_mapping = "uncharted"; break;
        case 3: out.display_tone_mapping = "filmic"; break;
        case 5: out.display_tone_mapping = "reinhard"; break;
        default: out.display_tone_mapping = "linear"; break;
    }

    const MaterialPreviewLightingPreset preset =
        ::render_settings.material_preview_lighting_preset;
    out.preset = previewLightingName(preset);
    out.uses_scene_lights = (preset == MaterialPreviewLightingPreset::Scene);
    // Scene owns the bounded shared shadow atlas. The three-point rig remains
    // shadow-free so material inspection does not depend on scene geometry.
    out.shadows = (preset == MaterialPreviewLightingPreset::Scene);

    if (g_ctx) {
        // setLights yalnizca GORUNUR isiklari yukler; ayni olcutu kullaniyoruz
        // ki bildirilen sayi shader'in gerceklik payiyla ayni olsun.
        int visible = 0;
        for (const auto& l : g_ctx->scene.lights) {
            if (l && l->visible) ++visible;
        }
        out.scene_light_total = visible;
        out.scene_light_count =
            (visible > kMaterialPreviewMaxSceneLights)
                ? kMaterialPreviewMaxSceneLights : visible;
        if (preset == MaterialPreviewLightingPreset::Scene) {
            const ViewportQualityInfo quality = viewportQuality();
            const int shadowTileCapacity = quality.shadow_tile_capacity;
            const int shadowLightBudget = quality.shadow_light_budget;
            const bool nishitaWorld =
                g_ctx->renderer.world.getMode() == WORLD_MODE_NISHITA;
            const NishitaSkyParams sky = g_ctx->renderer.world.getNishitaParams();
            out.world_background = true;
            out.world_sun_direct = nishitaWorld && sky.sun_intensity > 0.0f;
            out.world_sun_shadow = out.world_sun_direct;
            const int directionalCascades =
                quality.directional_shadow_cascades;
            int tiles = out.world_sun_shadow ? directionalCascades : 0;
            int previewLights = 0;
            for (const auto& l : g_ctx->scene.lights) {
                if (!l || !l->visible) continue;
                if (previewLights++ >= kMaterialPreviewMaxSceneLights) break;
                const int need = l->type() == LightType::Point ? 6 :
                    (l->type() == LightType::Directional
                        ? directionalCascades : 1);
                if (out.shadowed_light_count >= shadowLightBudget ||
                    tiles + need > shadowTileCapacity) continue;
                tiles += need;
                ++out.shadowed_light_count;
            }
            out.world_ambient = true;
            // Physical Sky now also has a prefiltered producer, so the report
            // may no longer be gated on HDRI: gating it there would show the
            // sky path as a permanent fallback while it was actually running.
            {
                Backend::MaterialPreviewIblStatus ibl{};
                // ★★★★ `forEachViewportBackend` ONCE raster viewport adapter'ini,
                //   sonra render backend'ini verir. Diger alanlar OR'lanir (bir
                //   yetenek herhangi birinde varsa vardir), ama ARKA PLAN
                //   OR'lanamaz: arizanin tanimi zaten "RT adapter dogru, raster
                //   adapter degil". OR, tam olarak aranan ayrismayi gizlerdi.
                bool backgroundTaken = false;
                forEachViewportBackend([&](Backend::IBackend& backend) {
                    Backend::MaterialPreviewIblStatus candidate{};
                    if (backend.getMaterialPreviewIblStatus(candidate)) {
                        ibl.supported = ibl.supported || candidate.supported;
                        ibl.sky_capture_supported = ibl.sky_capture_supported ||
                            candidate.sky_capture_supported;
                        if (candidate.ready && !ibl.ready) {
                            ibl.ready = true;
                            ibl.source = candidate.source;
                        }
                        if (!backgroundTaken) {
                            ibl.background_source = candidate.background_source;
                            backgroundTaken = true;
                        }
                    }
                });
                const bool worldNeedsIbl =
                    g_ctx->renderer.world.getMode() == WORLD_MODE_HDRI ||
                    g_ctx->renderer.world.getMode() == WORLD_MODE_NISHITA;
                out.world_ibl_supported = ibl.supported;
                out.world_sky_capture_supported = ibl.sky_capture_supported;
                out.world_ibl_ready = ibl.ready;
                out.world_ibl_fallback = worldNeedsIbl && !ibl.ready;
                out.world_ibl_source = ibl.ready ? ibl.source : "none";
                out.world_background_source = ibl.background_source;
            }
        }
        if (g_ctx->scene_ui_ptr)
            out.material_preview_active =
                (g_ctx->scene_ui_ptr->viewport_settings.shading_mode == 1);
    }
    return out;
}

Result setViewportPreviewLighting(const std::string& preset) {
    if (!g_ctx) return Result::fail("Engine context not bound");

    const PreviewLightingEntry* entry = previewLightingEntry(preset);
    if (!entry)
        return Result::fail("Unknown preview lighting preset '" + preset +
                            "'. Valid: three_point, scene (legacy classic/studio/outdoor aliases are accepted).");

    ::render_settings.material_preview_lighting_preset = entry->preset;

    // Panel combo'sunun yaptigi ile ayni. Preset push constant'tan okundugu
    // icin mesh yeniden kurulumu GEREKMEZ -- ama birikim sifirlanmali, yoksa
    // bir sonraki probe onceki preset'te birikmis kareyi olcer.
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

namespace {
    void write_to_vector_func(void* context, void* data, int size) {
        auto* vec = static_cast<std::vector<unsigned char>*>(context);
        const unsigned char* bytes = static_cast<const unsigned char*>(data);
        vec->insert(vec->end(), bytes, bytes + size);
    }
}

std::string getViewportScreenshotAsBase64() {
    std::lock_guard<std::mutex> lock(g_frame_mutex);
    if (!g_frame_available || g_frame_width <= 0 || g_frame_height <= 0 || g_frame_rgba.empty())
        return "";

    std::vector<unsigned char> jpeg_bytes;
    int quality = 80;
    stbi_write_jpg_to_func(write_to_vector_func, &jpeg_bytes, g_frame_width, g_frame_height, 4, g_frame_rgba.data(), quality);

    if (jpeg_bytes.empty()) return "";

    static const char base64_chars[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789+/";

    std::string ret;
    int i = 0, j = 0;
    unsigned char char_array_3[3];
    unsigned char char_array_4[4];

    for (unsigned char byte : jpeg_bytes) {
        char_array_3[i++] = byte;
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;

            for (i = 0; i < 4; i++) ret += base64_chars[char_array_4[i]];
            i = 0;
        }
    }

    if (i > 0) {
        for (j = i; j < 3; j++) char_array_3[j] = '\0';
        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
        char_array_4[3] = char_array_3[2] & 0x3f;

        for (j = 0; j < i + 1; j++) ret += base64_chars[char_array_4[j]];
        while (i++ < 3) ret += '=';
    }

    return ret;
}

} // namespace rtapi
