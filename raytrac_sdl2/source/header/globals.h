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

struct RenderSettings {
    // Sampling
    int samples_per_pixel;
    int samples_per_pass;
    int max_bounces;

    // Adaptive Sampling
    bool use_adaptive_sampling;
    int min_samples;
    int max_samples;
    float variance_threshold;

    // Denoiser
    bool use_denoiser;
    float denoiser_blend_factor;

    // Backend
    bool use_optix;
    bool UI_use_embree;
    // Animation
    float animation_duration;
    int animation_fps;
    bool start_animation_render = false;
    bool save_image_requested = false;
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
        // Program EXE’nin yanýna log dosyasý oluþturur
        logFile.open("SceneLog.txt", std::ios::out | std::ios::app);
        if (!logFile.is_open()) {
            std::cerr << "[LOGGER ERROR] Log dosyasý açýlamadý!\n";
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

        // UI için hafýzaya
        lines.push_back({ msg, level });

        // Konsola
        std::cout << "[" << prefix << "] " << msg << std::endl;

        // Dosyaya — CRASH olsa bile satýr kaybolmaz
        if (logFile.is_open()) {
            logFile << "[" << prefix << "] " << msg << std::endl;
            logFile.flush();
        }
    }

    void clear() {
        std::lock_guard<std::mutex> guard(lock);
        lines.clear();

        // Dosyayý da sýfýrlayalým
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



// Yalnýzca bildirim:
extern RenderSettings render_settings;
extern std::atomic<int> completed_pixels;
extern const double min_distance;
extern const double max_distance;
extern  float aspect_ratio; // Sabit olarak double türünde tanýmlýyoruz
extern  int image_width;
extern  int image_height;
extern const float EPSILON;
extern std::atomic<int> next_row;
extern const double infinity;
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
extern float last_render_time_ms;
extern bool pending_resolution_change;
extern int pending_width;
extern int pending_height;
extern float pending_aspect_ratio;
extern float light_radius;
extern bool render_finished;
extern bool rendering_in_progress;
extern bool rendering_stopped_gpu;
extern std::atomic<bool> rendering_stopped_cpu;
#endif // GLOBALS_H
