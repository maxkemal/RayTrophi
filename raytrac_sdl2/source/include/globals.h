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
#include "Vec3.h"
#include <filesystem>
#include <fstream>

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

struct RenderSettings {
    // Input / Interaction
    float mouse_sensitivity = 0.4f;

    // Quality Preset
    QualityPreset quality_preset = QualityPreset::Preview;
    
    // Sampling
    int samples_per_pixel = 1;
    int samples_per_pass = 1;
    int max_bounces = 10;
    int transparent_max_bounces = 10;

    // Adaptive Sampling
    bool use_adaptive_sampling = true;
    int min_samples = 1;
    int max_samples = 32;
    float variance_threshold = 0.1f;

    // Denoiser
    bool use_denoiser = false;        // Viewport Denoiser
    bool render_use_denoiser = true;  // Final Render Denoiser
    float denoiser_blend_factor = 1.0f;

    // Backend
    bool use_optix = false;
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
    
    // Viewport Grid Settings
    bool grid_enabled = false;
    bool show_background = true;       // NEW: Toggle background visibility
    float grid_fade_distance = 500.0f;  // Units where grid fades out completely
    float viewport_near_clip = 0.01f;   // Objects closer than this won't be seen
    float viewport_far_clip = 5000.0f; // Objects further than this won't be seen
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
        // Program EXE’nin yanına log dosyası oluşturur
        logFile.open("SceneLog.txt", std::ios::out | std::ios::app);
        if (!logFile.is_open()) {
            std::cerr << "[LOGGER ERROR] Log dosyası açılamadı!\n";
        }
    }

    ~UILogger() {
        if (logFile.is_open()) {
            logFile.flush();
            logFile.close();
        }
    }

    void add(const std::string& msg, LogLevel level = LogLevel::Info) {
        std::lock_guard<std::mutex> guard(lock);

        const char* prefix = logLevelToString(level);

        // UI için hafızaya
        lines.push_back({ msg, level });

        // Dosyaya — CRASH olsa bile satır kaybolmaz (Öncelikli Yazım)
        if (logFile.is_open()) {
            logFile << "[" << prefix << "] " << msg << std::endl;
            logFile.flush();
        }

        // Konsola (Geliştirici takibi için)
        std::cout << "[" << prefix << "] " << msg << std::endl;
    }

    void clear() {
        std::lock_guard<std::mutex> guard(lock);
        lines.clear();

        // Dosyayı da sıfırlayalım
        if (logFile.is_open()) {
            logFile.close();
            logFile.open("SceneLog.txt", std::ios::out | std::ios::trunc);
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
extern const float min_distance;
extern const float max_distance;
extern  float aspect_ratio; // Sabit olarak double türünde tanımlıyoruz
extern  int image_width;
extern  int image_height;
extern const float EPSILON;
extern std::atomic<int> next_row;
extern const float infinity;
extern  std::string baseDirectory;
extern bool atmosferic_effect_enabled;
extern const float max_normal_distance;
extern const float max_normal_strength;
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
extern bool g_hasCUDA; // General CUDA availability (independent of OptiX)
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

// ===========================================================================
// DEFERRED REBUILD FLAGS - For optimized batched rebuilds
// Set these instead of calling rebuild immediately, Main loop handles them
// ===========================================================================
extern bool g_bvh_rebuild_pending;      // CPU BVH needs rebuild
extern bool g_gpu_refit_pending;        // GPU Geometry needs update (Deferred)
extern bool g_optix_rebuild_pending;
extern bool g_optix_rebuild_in_progress; // True while TLAS rebuild is happening - blocks render    // GPU OptiX geometry needs rebuild
extern bool g_mesh_cache_dirty;         // UI mesh cache needs rebuild
extern bool g_cpu_bvh_refit_pending;    // CPU BVH fast refit (Embree only)

// ===========================================================================
// SCENE LOADING FLAGS - Thread safety for project load/save operations
// ===========================================================================
extern std::atomic<bool> g_scene_loading_in_progress;  // Prevents concurrent load operations
extern bool g_needs_geometry_rebuild;   // Set by loader thread, main loop does actual rebuild
extern bool g_needs_optix_sync;         // Set by loader thread, main loop syncs OptiX buffers

#endif // GLOBALS_H



