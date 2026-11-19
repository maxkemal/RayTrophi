#include "Core/RuntimeProjectHost.h"

namespace AppRuntime {
namespace {
UIContext* host_context = nullptr;
ProjectHost* project_host = nullptr;
DisplayHost* display_host = nullptr;
bool body_overlay = false;
bool video_card = false;
} // namespace

std::filesystem::path projectPathFromUtf8(const std::string& path) {
    return std::filesystem::path(std::u8string(path.begin(), path.end()));
}

std::string projectPathToUtf8(const std::filesystem::path& path) {
    const auto encoded = path.u8string();
    return std::string(encoded.begin(), encoded.end());
}

void bindProjectHost(UIContext& context, ProjectHost& host) {
    host_context = &context;
    project_host = &host;
}

void unbindProjectHost(ProjectHost& host) {
    if (project_host == &host) {
        project_host = nullptr;
        host_context = nullptr;
    }
}

std::optional<rtapi::Result> tryOpenRuntimeProject(UIContext& context,
                                                const std::string& path) {
    if (!project_host || host_context != &context) {
        return std::nullopt;
    }
    return project_host->openProject(path);
}

nlohmann::json runtimeStatus(UIContext& context) {
    if (!project_host || host_context != &context) {
        return {{"host_active", false}, {"host", "studio"}};
    }
    return project_host->status();
}

void bindDisplayHost(DisplayHost& host) {
    display_host = &host;
}

void unbindDisplayHost(DisplayHost& host) {
    if (display_host == &host) {
        display_host = nullptr;
        body_overlay = false;
        video_card = false;
    }
}

rtapi::Result runtimeDisplayStatus(nlohmann::json& out) {
    if (!display_host) {
        return rtapi::Result::fail("player.display requires the RayTrophi Player runtime");
    }
    out = display_host->displayStatus();
    out["body_overlay"] = body_overlay;
    out["video_card"] = video_card;
    return rtapi::Result::success();
}

rtapi::Result setRuntimeDisplay(const nlohmann::json& params) {
    if (!display_host) {
        return rtapi::Result::fail("player.set_display requires the RayTrophi Player runtime");
    }
    if (!params.is_object()) {
        return rtapi::Result::fail("player.set_display expects an object");
    }
    const auto overlay = params.find("body_overlay");
    if (overlay != params.end() && !overlay->is_null() && !overlay->is_boolean()) {
        return rtapi::Result::fail("body_overlay must be a boolean");
    }
    const auto card = params.find("video_card");
    if (card != params.end() && !card->is_null() && !card->is_boolean()) {
        return rtapi::Result::fail("video_card must be a boolean");
    }
    const auto result = display_host->setDisplay(params);
    if (result.ok && overlay != params.end() && overlay->is_boolean()) {
        body_overlay = overlay->get<bool>();
    }
    if (result.ok && card != params.end() && card->is_boolean()) {
        video_card = card->get<bool>();
    }
    return result;
}

bool playerBodyOverlayEnabled() {
    return body_overlay;
}

bool playerVideoCardEnabled() {
    return video_card;
}

} // namespace AppRuntime
