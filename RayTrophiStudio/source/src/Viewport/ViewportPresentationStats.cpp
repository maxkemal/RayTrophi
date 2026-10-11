#include "Viewport/ViewportPresentationStats.h"

#include "Backend/IBackend.h"
#include "globals.h"
#include <SDL.h>
#include <chrono>
#include <cmath>
#include <cstdio>

namespace Viewport {
namespace {

using Clock = std::chrono::steady_clock;
Clock::time_point lastPresentation;
Clock::time_point lastTitle;
Backend::ViewportMode previousMode = Backend::ViewportMode::Rendered;

} // namespace

void resetPresentationStats() {
    lastPresentation = Clock::time_point{};
    lastTitle = Clock::time_point{};
}

double notePresentationStats(RenderSettings& settings, SDL_Window* window,
                             const std::string& projectPath, Backend::ViewportMode mode,
                             const char* renderBackend) {
    const auto now = Clock::now();
    if (lastPresentation == Clock::time_point{} || mode != previousMode) {
        lastPresentation = now;
        previousMode = mode;
        settings.avg_total_frame_time_ms = 0.0f;
        settings.avg_total_frame_fps = 0.0f;
        lastTitle = Clock::time_point{};
        return 0.0;
    }

    const double periodMs = std::chrono::duration<double, std::milli>(
        now - lastPresentation).count();
    lastPresentation = now;
    if (!std::isfinite(periodMs) || periodMs <= 0.0) {
        return 0.0;
    }

    // Time-based smoothing responds at the same speed at 3 FPS and 60 FPS.
    const float intervalMs = static_cast<float>(periodMs);
    const float blend = static_cast<float>(1.0 - std::exp(-periodMs / 500.0));
    if (settings.avg_total_frame_time_ms <= 0.0f) {
        settings.avg_total_frame_time_ms = intervalMs;
    } else {
        settings.avg_total_frame_time_ms +=
            blend * (intervalMs - settings.avg_total_frame_time_ms);
    }
    // Reciprocal of the same averaged interval; do not average reciprocal FPS.
    settings.avg_total_frame_fps = 1000.0f / settings.avg_total_frame_time_ms;

    if (window && (lastTitle == Clock::time_point{} ||
        std::chrono::duration<double, std::milli>(now - lastTitle).count() >= 100.0)) {
        lastTitle = now;
        std::string projectName = projectPath;
        if (projectName.empty() || projectName == "Untitled") {
            projectName = "Untitled Project";
        } else {
            const auto slash = projectName.find_last_of("\\/");
            if (slash != std::string::npos) projectName = projectName.substr(slash + 1);
            const auto dot = projectName.find_last_of('.');
            if (dot != std::string::npos) projectName = projectName.substr(0, dot);
        }
        const char* modeName = mode == Backend::ViewportMode::Solid ? "Solid" :
            mode == Backend::ViewportMode::Matcap ? "Matcap" :
            mode == Backend::ViewportMode::MaterialPreview ? "RayFusion" : renderBackend;
        char timing[160];
        std::snprintf(timing, sizeof(timing), " - %.0f ms/frame - %.1f fps",
                      settings.avg_total_frame_time_ms, settings.avg_total_frame_fps);
        std::string title = "RayTrophi Studio [" + projectName + "] - " + modeName;
        if (mode == Backend::ViewportMode::Rendered) {
            title += " - Sample " + std::to_string(settings.render_current_samples) +
                "/" + std::to_string(settings.render_target_samples);
        }
        title += timing;
        SDL_SetWindowTitle(window, title.c_str());
    }
    return periodMs;
}

} // namespace Viewport
