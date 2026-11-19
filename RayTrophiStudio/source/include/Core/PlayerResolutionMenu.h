#pragma once

#include <SDL.h>
#include "imgui.h"
#include "json.hpp"
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace AppRuntime {

// Render choices are independent of the presentation window. Querying another
// monitor refreshes the choices, but never changes an already selected extent.
struct PlayerResolutionChoices {
    std::vector<std::pair<int, int>> modes;
    std::pair<int, int> recommended{0, 0};

    bool supports(int width, int height) const {
        return std::find(modes.begin(), modes.end(), std::make_pair(width, height)) !=
               modes.end();
    }

    nlohmann::json status() const {
        auto entries = nlohmann::json::array();
        for (const auto& mode : modes) {
            entries.push_back({mode.first, mode.second});
        }
        return {{"resolutions", entries},
                {"recommended_resolution", {recommended.first, recommended.second}}};
    }
};

inline PlayerResolutionChoices playerResolutionChoices(SDL_Window* window) {
    PlayerResolutionChoices result;
    const int display = window ? SDL_GetWindowDisplayIndex(window) : -1;
    SDL_DisplayMode desktop{};
    if (display < 0 || SDL_GetDesktopDisplayMode(display, &desktop) != 0) {
        return result; // Do not invent supported modes if the OS query failed.
    }
    const auto add = [&](int width, int height) {
        if (width >= 320 && height >= 240 && width <= 16384 && height <= 16384) {
            result.modes.emplace_back(width, height);
        }
    };
    add(desktop.w, desktop.h);
    const int count = SDL_GetNumDisplayModes(display);
    for (int i = 0; i < count; ++i) {
        SDL_DisplayMode mode{};
        if (SDL_GetDisplayMode(display, i, &mode) == 0) {
            add(mode.w, mode.h);
        }
    }
    std::sort(result.modes.begin(), result.modes.end(), [](const auto& a, const auto& b) {
        const auto area_a = static_cast<long long>(a.first) * a.second;
        const auto area_b = static_cast<long long>(b.first) * b.second;
        return area_a == area_b ? a < b : area_a < area_b;
    });
    result.modes.erase(std::unique(result.modes.begin(), result.modes.end()),
                       result.modes.end());
    if (!result.modes.empty()) {
        const auto desktop_size = std::make_pair(desktop.w, desktop.h);
        result.recommended = result.supports(1920, 1080) ? std::make_pair(1920, 1080) :
            (result.supports(desktop.w, desktop.h) ? desktop_size : result.modes.back());
    }
    return result;
}

// Returns a selection only; PlayerDisplay::setDisplay owns validation/mutation.
inline std::pair<int, int> drawPlayerResolutionMenu(
    bool& open, const PlayerResolutionChoices& choices, std::pair<int, int> current) {
    std::pair<int, int> selected{0, 0};
    const auto* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + viewport->WorkSize.x * 0.5f,
                                 viewport->WorkPos.y + viewport->WorkSize.y * 0.5f),
                            ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(370.0f, 380.0f), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Render resolution", &open, ImGuiWindowFlags_NoSavedSettings)) {
        ImGui::TextWrapped("Window resizing and fullscreen keep this resolution.");
        ImGui::Separator();
        if (choices.modes.empty()) {
            ImGui::TextWrapped("Display modes are unavailable. Current resolution is kept.");
        }
        for (const auto& mode : choices.modes) {
            std::string label = std::to_string(mode.first) + " x " +
                                std::to_string(mode.second);
            if (mode == choices.recommended) {
                label += " (Recommended)";
            }
            if (ImGui::Selectable(label.c_str(), mode == current)) {
                selected = mode;
                open = false;
            }
        }
    }
    ImGui::End();
    return selected;
}

} // namespace AppRuntime
