#pragma once

#include "Api/RtApiPlayerOperator.h"
#include "Core/Localization.h"
#include "Core/PlayerSceneLoad.h"
#include "Core/RuntimeProjectHost.h"
#include "imgui.h"
#include <SDL.h>
#include <array>
#include <cmath>
#include <string>
#include <vector>

namespace PlayerConsole {
using json = nlohmann::json;
inline const char *tr(const char *key, const char *english) {
    return Localization::text(key, english);
}
inline const char *trId(const char *key, const char *english) {
    return Localization::widget(key, english);
}
inline std::string translatedMessage(const std::string &message) {
    return Localization::message(message);
}
inline std::string stateLabel(const std::string &code) {
    return Localization::text("state." + code, code.c_str());
}

inline constexpr ImU32 ink = IM_COL32(232, 239, 250, 255);
inline constexpr ImU32 muted = IM_COL32(143, 158, 183, 255);
inline constexpr ImU32 mint = IM_COL32(91, 226, 189, 255);
inline constexpr ImU32 violet = IM_COL32(166, 147, 255, 255);
inline constexpr ImU32 panel = IM_COL32(24, 32, 49, 255);
inline constexpr ImU32 border = IM_COL32(49, 62, 84, 255);

struct Console {
    bool open = true;
    bool game_settings_dirty = false;
    std::string page = "games";
    std::string error;
    std::string selected_scene;
    json status;
    double next_refresh = 0;
    ImFont *heading = nullptr;
    ImFont *title = nullptr;
    char search[160]{};
    char folder[1024]{};
    char mapping_path[1024]{};
    std::string avatar;
    int person_id = 0;
    bool calibrating = false;
    bool camera_enumeration_requested = false;
    std::vector<ImVec2> corners;
    float room[2] = {4.0f, 3.0f};
    float angles[3]{};
};
extern Console ui;

inline ImVec2 add(ImVec2 p, float x, float y) {
    return ImVec2(p.x + x, p.y + y);
}
inline ImVec4 color(ImU32 value) {
    return ImGui::ColorConvertU32ToFloat4(value);
}
inline void label(const char *text, ImU32 tint = ink) {
    ImGui::PushStyleColor(ImGuiCol_Text, color(tint));
    ImGui::TextUnformatted(text);
    ImGui::PopStyleColor();
}
inline void heading(const char *text, const char *description) {
    if (ui.heading) {
        ImGui::PushFont(ui.heading);
    }
    label(text);
    if (ui.heading) {
        ImGui::PopFont();
    }
    ImGui::Spacing();
    ImGui::PushStyleColor(ImGuiCol_Text, color(muted));
    ImGui::TextWrapped("%s", description);
    ImGui::PopStyleColor();
    ImGui::Spacing();
    ImGui::Spacing();
}
inline bool action(const char *text, ImVec2 size = ImVec2(0, 38), bool primary = false) {
    ImGui::PushStyleColor(ImGuiCol_Button,
                          primary ? ImVec4(.25f, .67f, .57f, 1) : ImVec4(.15f, .20f, .29f, 1));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                          primary ? ImVec4(.31f, .77f, .65f, 1) : ImVec4(.21f, .28f, .39f, 1));
    ImGui::PushStyleColor(ImGuiCol_Text, primary ? ImVec4(.04f, .10f, .12f, 1) : color(ink));
    if (size.x > 0) {
        size.x = std::max(size.x, ImGui::CalcTextSize(text, nullptr, true).x + 24.0f);
    }
    const bool clicked = ImGui::Button(text, size);
    ImGui::PopStyleColor(3);
    return clicked;
}
inline bool call(const std::string &operation, const json &config = json::object()) {
    if (AppRuntime::playerSceneLoad().presenting()) {
        return false;
    }
    json output;
    const auto result = rtapi::playerOperator(operation, config, output);
    ui.next_refresh = 0;
    ui.error = result.ok ? "" : result.error;
    return result.ok;
}
inline void languagePicker(float width = 180.0f) {
    const auto language = ui.status.value("language", nlohmann::json::object());
    const auto current = language.value("locale", std::string("en"));
    std::string title = current;
    for (const auto &item : language.value("available", nlohmann::json::array())) {
        if (item.value("code", "") == current) {
            title = item.value("name", current);
        }
    }
    ImGui::SetNextItemWidth(width);
    if (ImGui::BeginCombo("##interface_language", title.c_str())) {
        for (const auto &item : language.value("available", nlohmann::json::array())) {
            const auto code = item.value("code", "");
            if (ImGui::Selectable(item.value("name", code).c_str(), current == code)) {
                call("language.set", {{"locale", code}});
            }
        }
        ImGui::EndCombo();
    }
}

// Native vector icons follow the app's outlined icon language. They do not
// depend on supplementary Unicode glyphs absent from Player's font atlas.
inline void icon(ImDrawList *draw, ImVec2 c, const std::string &kind, ImU32 tint, float r = 12) {
    constexpr float line = 1.8f;
    if (kind == "settings") {
        draw->AddCircle(c, r * .65f, tint, 20, line);
        draw->AddCircle(c, r * .25f, tint, 16, line);
        for (int tooth = 0; tooth < 8; ++tooth) {
            const float angle = tooth * 0.785398163f;
            draw->AddLine(add(c, std::cos(angle) * r * .65f, std::sin(angle) * r * .65f),
                          add(c, std::cos(angle) * r, std::sin(angle) * r), tint, line);
        }
    } else if (kind == "camera") {
        draw->AddRect(add(c, -r, -r * .65f), add(c, r, r * .7f), tint, 4, 0, line);
        draw->AddCircle(c, r * .37f, tint, 20, line);
        draw->AddLine(add(c, -r * .6f, -r * .65f), add(c, -r * .3f, -r), tint, line);
        draw->AddLine(add(c, -r * .3f, -r), add(c, r * .25f, -r), tint, line);
    } else if (kind == "display") {
        draw->AddRect(add(c, -r, -r * .75f), add(c, r, r * .5f), tint, 3, 0, line);
        draw->AddLine(add(c, 0, r * .5f), add(c, 0, r), tint, line);
        draw->AddLine(add(c, -r * .5f, r), add(c, r * .5f, r), tint, line);
    } else if (kind == "players") {
        draw->AddCircle(add(c, 0, -r * .55f), r * .35f, tint, 18, line);
        draw->AddBezierCubic(add(c, -r * .8f, r), add(c, -r, -r * .25f), add(c, r, -r * .25f),
                             add(c, r * .8f, r), tint, line);
    } else if (kind == "audio") {
        draw->AddRect(add(c, -r, -r * .3f), add(c, -r * .5f, r * .3f), tint, 1, 0, line);
        const ImVec2 points[] = {add(c, -r * .5f, -r * .3f), add(c, r * .1f, -r * .8f),
                                 add(c, r * .1f, r * .8f), add(c, -r * .5f, r * .3f)};
        draw->AddPolyline(points, 4, tint, 0, line);
        draw->AddBezierCubic(add(c, r * .4f, -r * .65f), add(c, r * 1.2f, -r * .5f),
                             add(c, r * 1.2f, r * .5f), add(c, r * .4f, r * .65f), tint, line);
    } else {
        draw->AddRect(add(c, -r, -r * .65f), add(c, r, r * .65f), tint, 5, 0, line);
        draw->AddLine(add(c, -r * .7f, 0), add(c, -r * .15f, 0), tint, line);
        draw->AddLine(add(c, -r * .42f, -r * .26f), add(c, -r * .42f, r * .26f), tint, line);
        draw->AddCircleFilled(add(c, r * .4f, -r * .2f), 2, tint);
        draw->AddCircleFilled(add(c, r * .65f, r * .2f), 2, tint);
    }
}
void games(SDL_Renderer *renderer);
void gameSettings();
void hiddenObjects(json& values, json& patch);
bool applyGameSettingsDraft(bool save);
void camera(SDL_Renderer *renderer);
void players();
void display();
void audio();
void releaseArtwork();
SDL_Texture *brand(SDL_Renderer *renderer);
} // namespace PlayerConsole
