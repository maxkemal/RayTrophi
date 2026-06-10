/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          globals.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#ifndef GLOBALS_H
#define GLOBALS_H

#include <mutex>
#include <atomic>
#include <limits>
#include <cmath>
#include <vector>
#include <memory>
#include "Vec3.h"
#include "Backend/RenderCapabilities.h"
#include "SimulationComputeVulkanContext.h"
#include <filesystem>
#include <fstream>
#include <iostream>

namespace Backend {
class SceneTextureManager;
}

// Quality presets for easy selection
enum class QualityPreset {
    Preview = 0,      // Fast, noisy - good for quick previews
    Production = 1,   // Balanced quality/speed
    Cinematic = 2     // Highest quality, slowest
};

// Timeline playback quality presets
enum class TimelineQualityPreset {
    Draft = 0,        // 1 sample - fastest, for scrubbing
    Low = 1,          // 4 samples - basic preview
    Medium = 2,       // 16 samples - balanced
    High = 3          // 64 samples - high quality preview
};

enum class RasterViewportQualityPreset {
    Auto = 0,
    Performance = 1,
    Balanced = 2,
    Quality = 3
};

enum class MaterialPreviewLightingPreset {
    Classic = 0,
    Studio = 1,
    Outdoor = 2
};

// Helper to get sample count from timeline quality preset
inline int getTimelineSamplesFromPreset(TimelineQualityPreset preset) {
    switch (preset) {
        case TimelineQualityPreset::Draft:  return 1;
        case TimelineQualityPreset::Low:    return 4;
        case TimelineQualityPreset::Medium: return 16;
        case TimelineQualityPreset::High:   return 64;
        default: return 1;
    }
}

// Resolution source for final render
enum class ResolutionSource {
    Native = 0,       // Use current window/viewport size
    Custom = 1,       // Manual width x height
    FromAspect = 2    // Calculate from aspect ratio + base height
};

// Per-frame OIDN timing telemetry ([OIDN][CUDA] / [OIDN][Vulkan] log lines plus
// the chrono/CUDA-event timers that feed them). Off by default — measurement is
// done; leaving the timers in costs CPU + a CUDA event sync every frame. Flip to
// 1 to bring the instrumentation back when investigating a denoiser regression.
#ifndef RT_OIDN_PROFILING
#define RT_OIDN_PROFILING 0
#endif

enum class DenoiserMode {
    Fast = 0,         // beauty only
    Quality = 1       // beauty + albedo + normal
};

// OIDN model/quality tier (orthogonal to DenoiserMode, which selects AOV usage).
// Maps to OIDN_QUALITY_FAST/BALANCED/HIGH. The viewport path uses this; final
// renders are forced to High regardless. Fast is the cheapest model (~half the
// GPU cost of Balanced on a mid-range card) and is the right default for an
// interactive, still-converging preview.
enum class DenoiserQuality {
    Fast = 0,
    Balanced = 1,
    High = 2
};

struct RenderSettings {
    // Input / Interaction
    float mouse_sensitivity = 0.4f;

    // Quality Preset
    QualityPreset quality_preset = QualityPreset::Preview;
    RasterViewportQualityPreset raster_viewport_quality_preset = RasterViewportQualityPreset::Auto;
    MaterialPreviewLightingPreset material_preview_lighting_preset = MaterialPreviewLightingPreset::Classic;
    
    // Sampling
    int samples_per_pixel = 1;
    int samples_per_pass = 1;
    int max_bounces = 10;
    int diffuse_bounces = 4;
    int transmission_bounces = 8;
    bool show_scene_stats_hud = true;

    // Adaptive Sampling
    bool use_adaptive_sampling = true;
    int min_samples = 1;
    int max_samples = 128;
    float variance_threshold = 0.01f;

    // Denoiser
    bool use_denoiser = false;        // Viewport Denoiser
    bool render_use_denoiser = false;  // Final Render Denoiser
    float denoiser_blend_factor = 1.0f;
    DenoiserMode denoiser_mode = DenoiserMode::Quality;
    DenoiserQuality denoiser_quality = DenoiserQuality::Fast; // Viewport OIDN model tier; final render forced to High
    // Set per-frame from Renderer::stylizeMode.enabled so the OptiX backend allocates the
    // albedo/normal/position AOV buffers for Stylize even when the denoiser is off.
    bool stylize_enabled = false;

    // Backend
    bool use_optix = false;
    bool use_vulkan = false;
    bool backend_changed = false;
    bool UI_use_embree = true;
  
    // Animation
    float animation_duration = 1.0f;
    int animation_fps = 24;
    int animation_samples_per_frame = 1; // Samples per frame during playback (updated by preset)
    TimelineQualityPreset timeline_quality_preset = TimelineQualityPreset::Draft; // Easy quality selection
    bool timeline_use_denoiser = false;  // Apply denoiser during timeline playback
    bool start_animation_render = false;
    bool animation_render_locked = false;  // Lock viewport/camera during animation render
    bool save_image_requested = false;
    int animation_start_frame = 0;
    int animation_end_frame = 100;      // Default to 100 frames (sensible default)
    int animation_current_frame = 0;
    int animation_total_frames = 0;     // For progress tracking
    std::string animation_output_folder = "";
    
    // Animation playback (timeline icin)
    bool animation_is_playing = false;
    int animation_playback_frame = 0;
    bool realtime_weather_preview = false;
    
    // Render progress tracking (for UI display)
    int render_current_samples = 0;
    int render_target_samples = 256;
    int final_render_samples = 128;
    int final_render_width = 1280;
    int final_render_height = 720; // Specific for F12 Output
    
    // Resolution Source (Native/Custom/FromAspect)
    ResolutionSource resolution_source = ResolutionSource::Custom;
    int aspect_base_height = 1080;        // Base height for aspect ratio calculation
    int aspect_ratio_index = 0;           // Index into CameraPresets::ASPECT_RATIOS
    
    float render_progress = 0.0f;
    bool is_rendering_active = false;
    bool is_render_paused = false;
    bool is_final_render_mode = false;
    
    // Render time estimation
    float render_elapsed_seconds = 0.0f;
    float render_estimated_remaining = 0.0f;
    float avg_sample_time_ms = 0.0f;
    float avg_total_frame_time_ms = 0.0f;
    float avg_total_frame_fps = 0.0f;
    
    // Viewport Grid Settings
    bool grid_enabled = false;
    bool show_background = true;       // NEW: Toggle background visibility
    float grid_fade_distance = 500.0f;  // Units where grid fades out completely
    float viewport_near_clip = 0.01f;   // Objects closer than this won't be seen
    float viewport_far_clip = 1000000.0f; // Keep far volumes/clouds visible in large-scale scenes
    bool persistent_tonemap = false;     // NEW: Persistent tonemapping (Renamed from tonemap_auto_apply)
};
enum class LogLevel { Info, Warning, Error };

struct LogEntry {
    std::string msg;
    LogLevel level;
};

// Makrolar
#define SCENE_LOG_INFO(msg) g_sceneLog.add(msg, LogLevel::Info)
#define SCENE_LOG_WARN(msg) g_sceneLog.add(msg, LogLevel::Warning)
#define SCENE_LOG_ERROR(msg) g_sceneLog.add(msg, LogLevel::Error)

class UILogger {
public:
    UILogger() {
        logFilePath = "SceneLog.txt";
        // Program EXE’nin yanına log dosyası oluşturur
        logFile.open(logFilePath, std::ios::out | std::ios::app);
        if (!logFile.is_open()) {
            std::cerr << "[LOGGER ERROR] Failed to open log file!\n";
        }
    }

    ~UILogger() {
        if (logFile.is_open()) {
            logFile.flush();
            logFile.close();
        }
    }

    void initLogLocation() {
        std::lock_guard<std::mutex> guard(lock);
        if (logFile.is_open()) {
            logFile.close();
        }
        try {
            logFilePath = std::filesystem::absolute("SceneLog.txt").string();
        } catch(...) {
            logFilePath = "SceneLog.txt";
        }
        // Re-open in the absolute path
        logFile.open(logFilePath, std::ios::out | std::ios::app);
    }

    void add(const std::string& msg, LogLevel level = LogLevel::Info) {
        std::lock_guard<std::mutex> guard(lock);

        const char* prefix = logLevelToString(level);

        // UI için hafızaya
        lines.push_back({ msg, level });
        // Bound in-memory log growth to avoid unbounded RAM usage during long sessions
        // (e.g. repeated backend switches generating thousands of Vulkan/OptiX lines).
        static constexpr size_t kMaxInMemoryLines = 5000;
        if (lines.size() > kMaxInMemoryLines) {
            const size_t excess = lines.size() - kMaxInMemoryLines;
            lines.erase(lines.begin(), lines.begin() + excess);
        }

        // Dosyaya — CRASH olsa bile satır kaybolmaz (Öncelikli Yazım)
        if (logFile.is_open()) {
            logFile << "[" << prefix << "] " << msg << std::endl;
            logFile.flush();
        }

        // Konsola (Gelistirici takibi icin). Sim compute telemetry stays in
        // the UI/file log only; the app may run with its console hidden/closed.
        if (msg.rfind("[SimCompute]", 0) != 0) {
            std::cout << "[" << prefix << "] " << msg << std::endl;
        }
    }

    void clear() {
        std::lock_guard<std::mutex> guard(lock);
        lines.clear();

        // Dosyayı da sıfırlayalım
        if (logFile.is_open()) {
            logFile.close();
            logFile.open(logFilePath, std::ios::out | std::ios::trunc);
        }
    }

    void getLines(std::vector<LogEntry>& out) {
        std::lock_guard<std::mutex> guard(lock);
        out = lines;
    }
    void closeLogFile() {
        std::lock_guard<std::mutex> guard(lock);
        if (logFile.is_open())
            logFile.close();
    }
private:
    std::mutex lock;
    std::vector<LogEntry> lines;
    std::ofstream logFile;
    std::string logFilePath;

    const char* logLevelToString(LogLevel level) const {
        switch (level) {
        case LogLevel::Info:    return "INFO";
        case LogLevel::Warning: return "WARNING";
        case LogLevel::Error:   return "ERROR";
        }
        return "INFO";
    }
};

// Global
extern UILogger g_sceneLog;



// Yalnızca bildirim:
extern RenderSettings render_settings;
extern std::atomic<int> completed_pixels;

extern  float aspect_ratio; // Sabit olarak double türünde tanımlıyoruz
extern  int image_width;
extern  int image_height;
extern const float EPSILON;
extern std::atomic<int> next_row;
extern const float infinity;
extern  std::string baseDirectory;
extern const float aperture;
extern const float focusdistance;
extern const float gamma;
extern const float exposure ;
extern const float saturation ;
const float BASE_EPSILON = 1e-6f;
const float MIN_EPSILON = 1e-8f;
const float MAX_EPSILON = 1e-4f;
extern bool globalreflectance ;
extern bool is_normal_map;
extern  int hitcount;
extern bool use_embree;
extern bool g_hasOptix ;
extern bool g_hasVulkan;
extern bool g_hasVulkanRT;
extern bool g_hasCUDA; // General CUDA availability (independent of OptiX)
extern bool g_hasVulkanComputeSim; // Vulkan sim compute backend initialized successfully
extern std::atomic<int> g_cuda_texture_upload_scope_depth;

class ScopedCudaTextureUpload {
public:
    ScopedCudaTextureUpload() {
        g_cuda_texture_upload_scope_depth.fetch_add(1, std::memory_order_acq_rel);
    }
    ~ScopedCudaTextureUpload() {
        g_cuda_texture_upload_scope_depth.fetch_sub(1, std::memory_order_acq_rel);
    }

    ScopedCudaTextureUpload(const ScopedCudaTextureUpload&) = delete;
    ScopedCudaTextureUpload& operator=(const ScopedCudaTextureUpload&) = delete;
};

inline bool isCudaTextureUploadAllowed() {
    return g_cuda_texture_upload_scope_depth.load(std::memory_order_acquire) > 0;
}
Backend::RenderBackendCapabilities captureRuntimeRenderCapabilities();
std::shared_ptr<Backend::SceneTextureManager> getSharedSceneTextureManager();
// Clears any cached OptiX textureId entries equal to `textureId` from the shared
// SceneTextureManager. Call this BEFORE cudaDestroyTextureObject so concurrent
// lookups never receive a stale (about-to-be-destroyed) handle.
void notifyOptixTextureDestroyed(int64_t textureId);
extern float last_render_time_ms;
extern bool pending_resolution_change;
extern int pending_width;
extern int pending_height;
extern float pending_aspect_ratio;
extern float light_radius;
extern bool render_finished;
extern std::atomic<bool> rendering_in_progress;
extern std::atomic<bool> rendering_stopped_gpu;
extern std::atomic<bool> rendering_stopped_cpu;
extern std::atomic<bool> rendering_paused;  // Pause animation render (P key or button)

// ===========================================================================
// DIRTY FLAGS - Set to true when respective data changes
// Prevents unnecessary GPU buffer updates when data is unchanged
// ===========================================================================
extern bool g_camera_dirty;
extern bool g_lights_dirty;
extern bool g_world_dirty;

// Granular scene-sync dirty flags (for syncActiveRenderBackendScene optimization)
// Each flag tracks whether a specific subsystem needs re-uploading to the GPU.
// Set to true by scene loaders, editors, and mode switches that modify data.
extern bool g_geometry_dirty;        // Mesh/triangle data changed (requires rebuildBackendGeometry)
extern bool g_materials_dirty;       // Material properties changed (requires updateBackendMaterials)
extern bool g_gas_volumes_dirty;     // Gas/VDB volumes changed
// Set after a bake hot-reload so uploadMaterials evicts the pool AFTER vkDeviceWaitIdle.
// Never evict the pool from the UI thread directly — GPU may still reference the old views.
extern bool g_texture_pool_dirty;

// Scene geometry generation counter — monotonically increasing.
// Incremented whenever scene geometry actually changes (load, add, delete, edit).
// Viewport/backend code compares against its last-seen generation to skip
// redundant rebuilds when geometry hasn't changed between mode switches.
extern std::atomic<uint64_t> g_scene_geometry_generation;

// ===========================================================================
// DEFERRED REBUILD FLAGS - For optimized batched rebuilds
// Set these instead of calling rebuild immediately, Main loop handles them
// ===========================================================================
extern bool g_bvh_rebuild_pending;      // CPU BVH needs rebuild
extern bool g_gpu_refit_pending;        // GPU Geometry needs update (Deferred)
extern bool g_vulkan_rebuild_pending;    // GPU Vulkan geometry needs rebuild
extern bool g_vulkan_geometry_append_pending; // Additive-only mutation hint — Main loop tries incremental TLAS refit before falling back to full rebuild
extern bool g_vulkan_geometry_deform_pending; // A physics body deformed its source mesh this frame — Main loop refits ONLY those BLAS in place (vs full-scene teardown)
extern bool g_viewport_raster_rebuild_pending; // Interactive raster viewport needs rebuild
extern bool g_optix_rebuild_pending;
// Discrete-particle render bridge changed its instance set (structural or motion).
// Consumed ONLY when the CPU reference backend is active (Main loop translates it
// into g_bvh_rebuild_pending) so a moving sim re-expands particle HittableInstances
// into the CPU BVH each frame. Left unconsumed during GPU sessions (those refit via
// g_gpu_refit_pending / g_*_rebuild_pending), so it never burns CPU off-path.
extern bool g_particle_cpu_geometry_dirty;
extern bool g_solid_viewport_active; // true in Solid/Matcap shading (not Rendered); fluid bridge shows splat-sphere proxy
// Simulation drive mode for grid-domain gas. true = Timeline (default): the sim
// is driven by the timeline (play bakes into the memory cache, scrub restores,
// stopped = frozen/idle → cheap). false = Live Update: continuous free-run
// interactive preview (always simulating + resetting accumulation; heavier).
extern bool g_sim_timeline_mode;
// Experimental: route all simulation compute (grid solver, APIC fluid forces/P2G,
// NanoVDB density bridge, ...) through the GPU backend. CUDA today, additional
// backends later. Default off; CPU is the reference path.
extern bool g_sim_use_gpu_solver;
// Populated by VulkanBackend when the Vulkan device is initialized.
// Used by scene_data::syncSimulationWorld to create the Vulkan sim compute backend.
extern RayTrophiSim::SimulationComputeVulkanContext g_vulkan_sim_compute_ctx;
extern std::atomic<bool> g_optix_rebuild_in_progress; // True while TLAS rebuild is happening - blocks render // GPU OptiX geometry needs rebuild
extern std::atomic<bool> g_viewport_rebuild_in_progress; // True while viewport backend resources are being torn down/rebuilt - blocks denoiser device access
// Set by:
//   (a) animation worker thread's cleanup lambda when the previous animation was aborted
//       (rendering_stopped_gpu/cpu was true) — the in-flight TLAS/BLAS build may have
//       left the RT backend in a partial state with no dirty flag raised, so the next
//       animation start needs a forced full sync to avoid a driver-side crash inside
//       createTLAS.
//   (b) backend-swap paths after the previous backend was destroyed and a fresh one
//       installed — the granular dirty flags may have been cleared during swap.
// Checked + cleared by the animation start path before launching the worker.
extern std::atomic<bool> g_anim_backend_needs_full_rebuild;
extern bool g_mesh_cache_dirty;         // UI mesh cache needs rebuild
extern bool g_cpu_bvh_refit_pending;    // CPU BVH fast refit (Embree only)
extern int g_bvh_rebuild_deferred_frames; // Delay CPU BVH rebuild briefly after heavy topology edits in GPU modes

// ===========================================================================
// SCENE LOADING FLAGS - Thread safety for project load/save operations
// ===========================================================================
extern std::atomic<bool> g_scene_loading_in_progress;  // Prevents concurrent load operations
extern std::atomic<bool> g_needs_geometry_rebuild;   // Set by loader thread, main loop does actual rebuild
extern std::atomic<bool> g_needs_optix_sync;         // Set by loader thread, main loop syncs backend buffers

// ===========================================================================
// UI INTERACTION FLAGS
// ===========================================================================
extern bool g_viewport_hovered;         // True when mouse is over the main RenderView

// Vulkan runtime device-loss indicator (set by Vulkan backend when fatal errors occur)
extern bool g_vulkan_device_lost;
extern std::string g_vulkan_device_lost_msg;
// Vulkan memory-pressure indicator (set by Vulkan backend on OOM/alloc pressure).
// Main loop performs a safe backend recreate at the next synchronization point.
extern std::atomic<bool> g_vulkan_trim_recreate_requested;

#endif // GLOBALS_H



