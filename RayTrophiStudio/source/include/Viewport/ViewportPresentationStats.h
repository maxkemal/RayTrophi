#pragma once

#include <cstdint>
#include <string>

struct RenderSettings;
struct SDL_Window;

namespace Backend {
enum class ViewportMode : uint8_t;
}

namespace Viewport {

// Called after presentation; returns the measured interval for IPC telemetry.
double notePresentationStats(RenderSettings& settings, SDL_Window* window,
                             const std::string& projectPath, Backend::ViewportMode mode,
                             const char* renderBackend);
void resetPresentationStats();

} // namespace Viewport
