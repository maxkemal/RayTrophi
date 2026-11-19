#include "UI/PlayerOperatorUI.h"
#include "PlayerOperatorWidgets.h"
#include <algorithm>

namespace PlayerConsole {
Console ui;
namespace {
void badge(const char *title, bool ready) {
    const auto point = ImGui::GetCursorScreenPos();
    const float width = ImGui::CalcTextSize(title).x + 30;
    ImGui::Dummy(ImVec2(width, 25));
    auto *draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled(point, add(point, width, 25),
                        ready ? IM_COL32(37, 66, 64, 255) : IM_COL32(55, 49, 65, 255), 12);
    draw->AddCircleFilled(add(point, 11, 12), 3, ready ? mint : IM_COL32(241, 188, 112, 255));
    draw->AddText(add(point, 20, 4), ready ? mint : muted, title);
}
} // namespace
} // namespace PlayerConsole

namespace AppRuntime {
using namespace PlayerConsole;

void loadPlayerOperatorFonts() {
    ImFontConfig config;
    config.OversampleH = 2;
    config.OversampleV = 2;
    static const ImWchar glyphs[] = {0x0020, 0x017F, 0};
    ui.heading = ImGui::GetIO().Fonts->AddFontFromFileTTF("C:/Windows/Fonts/segoeuib.ttf", 27,
                                                          &config, glyphs);
    ui.title = ImGui::GetIO().Fonts->AddFontFromFileTTF("C:/Windows/Fonts/segoeuib.ttf", 21,
                                                        &config, glyphs);
}
void showPlayerOperator(const std::string &page) {
    ui.open = true;
    ui.page = page;
    ui.next_refresh = 0;
}
bool playerOperatorOpen() {
    return ui.open;
}
void closePlayerOperator() {
    ui.open = false;
}
void releasePlayerOperatorUI() {
    releaseArtwork();
}
void drawPlayerOperator(SDL_Renderer *renderer) {
    if (!ui.open) {
        return;
    }
    if (ImGui::GetTime() >= ui.next_refresh) {
        json status;
        const auto result = rtapi::playerOperator("status", json::object(), status);
        if (result.ok) {
            if (!ui.status.is_null() &&
                ui.status.value("project", "") != status.value("project", "")) {
                ui.avatar.clear();
                ui.corners.clear();
                ui.calibrating = false;
                ui.game_settings_dirty = false;
            }
            ui.status = std::move(status);
        } else {
            ui.error = result.error;
        }
        ui.next_refresh = ImGui::GetTime() + .25;
    }
    if (ui.status.is_null()) {
        return;
    }
    const auto *viewport = ImGui::GetMainViewport();
    const float width = std::min(1160.0f, viewport->WorkSize.x - 24);
    const float height = std::min(780.0f, viewport->WorkSize.y - 24);
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(.025f, .04f, .07f, .78f));
    const auto flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                       ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoDocking;
    ImGui::Begin("##RayTrophiOperator", nullptr, flags);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 18);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 8);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(12, 8));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(12, 10));
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(.055f, .075f, .12f, 1));
    ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(.10f, .14f, .21f, 1));
    ImGui::PushStyleColor(ImGuiCol_Text, color(ink));
    ImGui::SetCursorPos(
        ImVec2((viewport->WorkSize.x - width) * .5f, (viewport->WorkSize.y - height) * .5f));
    ImGui::BeginChild("shell", ImVec2(width, height), ImGuiChildFlags_Borders,
                      ImGuiWindowFlags_NoScrollbar);
    const auto origin = ImGui::GetCursorScreenPos();
    ImGui::GetWindowDrawList()->AddRectFilled(origin, add(origin, width, 3), mint, 2);
    ImGui::SetCursorPos(ImVec2(24, 20));
    if (auto *logo = brand(renderer)) {
        ImGui::Image(static_cast<ImTextureID>(reinterpret_cast<intptr_t>(logo)), ImVec2(42, 42));
        ImGui::SameLine();
    }
    ImGui::BeginGroup();
    if (ui.title) {
        ImGui::PushFont(ui.title);
    }
    label(tr("player.raytrophi_player", "RayTrophi Player"));
    if (ui.title) {
        ImGui::PopFont();
    }
    label(width < 950 ? tr("player.operator_console", "OPERATOR CONSOLE")
                      : tr("player.operator_console_developer_edition",
                           "OPERATOR CONSOLE  /  DEVELOPER EDITION"),
          muted);
    ImGui::EndGroup();
    if (width >= 760) {
        ImGui::SetCursorPos(ImVec2(width - 318, 25));
        languagePicker(175);
    }
    ImGui::SetCursorPos(ImVec2(width - 122, 24));
    if (action(trId("player.close_esc", "Close / Esc"), ImVec2(100, 34))) {
        ui.open = false;
    }
    const bool compact = width < 650;
    const float nav_width = compact ? 52.0f : 190.0f;
    ImGui::SetCursorPos(ImVec2(14, 92));
    ImGui::BeginChild("navigation", ImVec2(nav_width, height - 178));
    const char *ids[] = {"games", "settings", "camera", "players", "display", "audio"};
    const char *titles[] = {
        tr("player.game_library", "Game library"), tr("player.game_settings", "Game settings"),
        tr("player.camera_room", "Camera & room"), tr("player.players", "Players"),
        tr("player.display", "Display"),           tr("player.room_audio", "Room audio")};
    for (int i = 0; i < 6; ++i) {
        const auto point = ImGui::GetCursorScreenPos();
        const bool chosen = ui.page == ids[i];
        if (ImGui::InvisibleButton(ids[i], ImVec2(nav_width, 56))) {
            ui.page = ids[i];
            ui.next_refresh = 0;
        }
        auto *navigation = ImGui::GetWindowDrawList();
        if (chosen || ImGui::IsItemHovered()) {
            navigation->AddRectFilled(point, add(point, nav_width, 56),
                                      chosen ? IM_COL32(34, 62, 65, 255) : panel, 10);
        }
        icon(navigation, add(point, 22, 28), ids[i], chosen ? mint : muted, 10);
        if (!compact) {
            navigation->AddText(add(point, 43, 20), chosen ? mint : muted, titles[i]);
        } else if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s", titles[i]);
        }
    }
    if (!compact) {
        ImGui::Spacing();
        ImGui::Separator();
        label(tr("player.quick_check", "QUICK CHECK"), muted);
        const auto camera_state = ui.status["camera"].value("state", "stopped");
        const bool project_ready = !ui.status.value("project", "").empty();
        badge(project_ready ? tr("player.game_loaded", "Game loaded")
                            : tr("player.choose_game", "Choose game"),
              project_ready);
        badge(camera_state == "tracking" ? tr("player.tracking_live", "Tracking live")
                                         : tr("player.camera_optional", "Camera optional"),
              camera_state == "tracking");
        const bool calibrated = ui.status.value("calibration_ready", false);
        badge(calibrated ? tr("player.floor_saved", "Floor saved")
                         : tr("player.floor_not_set", "Floor not set"),
              calibrated);
    }
    ImGui::EndChild();
    ImGui::SetCursorPos(ImVec2(nav_width + 42, 94));
    ImGui::BeginChild("content", ImVec2(width - nav_width - 68, height - 192));
    if (ui.page == "games") {
        games(renderer);
    } else if (ui.page == "settings") {
        gameSettings();
    } else if (ui.page == "camera") {
        camera(renderer);
    } else if (ui.page == "players") {
        players();
    } else if (ui.page == "display") {
        display();
    } else {
        audio();
    }
    ImGui::EndChild();
    ImGui::SetCursorPos(ImVec2(24, height - 82));
    const auto error = ui.error.empty() ? ui.status.value("error", "") : ui.error;
    if (!error.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, .68f, .5f, 1));
        ImGui::PushTextWrapPos(width - 410);
        ImGui::TextUnformatted(translatedMessage(error).c_str());
        ImGui::PopTextWrapPos();
        ImGui::PopStyleColor();
    } else {
        if (width >= 900) {
            label(tr("player.ready_the_room_let_imagination_play",
                     "Ready the room. Let imagination play."),
                  mint);
            label(tr("player.f2_console_f6_library_esc_return_to_game",
                     "F2 console  /  F6 library  /  Esc return to game"),
                  muted);
        }
    }
    ImGui::SetCursorPos(ImVec2(width - 372, height - 70));
    ImGui::BeginDisabled(ui.status.value("project", "").empty());
    if (ui.page == "settings") {
        const bool ready = ui.status.value("game_settings", json::object()).value("ready", false);
        ImGui::BeginDisabled(!ready);
        if (action(trId("player.apply", "Apply"), ImVec2(72, 44))) {
            applyGameSettingsDraft(false);
        }
        ImGui::SameLine();
        if (action(trId("player.save", "Save"), ImVec2(72, 44))) {
            applyGameSettingsDraft(true);
        }
        ImGui::EndDisabled();
    } else {
        if (action(trId("player.stop", "Stop"), ImVec2(72, 44))) {
            call("game.stop");
        }
        ImGui::SameLine();
        if (action(trId("player.reset", "Reset"), ImVec2(72, 44))) {
            call("game.reset");
        }
    }
    ImGui::SameLine();
    ImGui::BeginDisabled(ui.game_settings_dirty);
    if (action(trId("player.play_experience", "Play experience"), ImVec2(180, 44), true) &&
        call("game.start")) {
        ui.open = false;
    }
    ImGui::EndDisabled();
    if (ui.game_settings_dirty && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::SetTooltip(tr("player.apply_save_or_discard_game_settings_edits_before_playing",
                             "Apply, save or discard Game settings edits before playing."));
    }
    ImGui::EndDisabled();
    ImGui::EndChild();
    ImGui::PopStyleColor(3);
    ImGui::PopStyleVar(4);
    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar();
}
} // namespace AppRuntime
