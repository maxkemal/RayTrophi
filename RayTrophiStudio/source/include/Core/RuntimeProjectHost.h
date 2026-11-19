#pragma once

#include "Api/RtApi.h"
#include "json.hpp"

#include <optional>
#include <filesystem>
#include <string>

struct UIContext;

namespace AppRuntime {

std::filesystem::path projectPathFromUtf8(const std::string& path);
std::string projectPathToUtf8(const std::filesystem::path& path);

// Main-thread extension of the existing project.open operation. Studio keeps
// its editor finalization; Player installs its scene-viewer finalization.
class ProjectHost {
public:
    virtual ~ProjectHost() = default;
    virtual rtapi::Result openProject(const std::string& path) = 0;
    virtual nlohmann::json status() const = 0;
};

void bindProjectHost(UIContext& context, ProjectHost& host);
void unbindProjectHost(ProjectHost& host);
std::optional<rtapi::Result> tryOpenRuntimeProject(UIContext& context,
                                                const std::string& path);
nlohmann::json runtimeStatus(UIContext& context);

// Executable-owned display surface (window mode, render resolution, frame
// pacing, frame timing). The Player installs one; Studio has none, so the
// player.display / player.set_display IPC calls fail there with an explicit
// error instead of pretending. `params` keys: fullscreen (bool), width/height
// (windowed size, px), render_scale (0.25..1), pacing_hz (0 = uncapped), hud.
class DisplayHost {
public:
    virtual ~DisplayHost() = default;
    virtual nlohmann::json displayStatus() const = 0;
    virtual rtapi::Result setDisplay(const nlohmann::json& params) = 0;
};

void bindDisplayHost(DisplayHost& host);
void unbindDisplayHost(DisplayHost& host);
rtapi::Result runtimeDisplayStatus(nlohmann::json& out);
rtapi::Result setRuntimeDisplay(const nlohmann::json& params);
bool playerBodyOverlayEnabled();
bool playerVideoCardEnabled();

} // namespace AppRuntime
