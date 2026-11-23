#include "UI/PlayerOperatorUI.h"
#include "UI/PlayerKidLauncher.h"
#include "PlayerOperatorWidgets.h"
#include <algorithm>
#include <filesystem>

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

namespace {
#ifdef __ANDROID__
const char *systemBoldFont() {
    // Same candidates as the Player body font (PlayerMain), bold first. Not every Android
    // image ships every file; a missing font leaves heading/title null, which the widgets
    // already handle.
    for (const char *path :
         {"/system/fonts/RobotoStatic-Bold.ttf", "/system/fonts/Roboto-Bold.ttf",
          "/system/fonts/NotoSans-Bold.ttf", "/system/fonts/RobotoStatic-Regular.ttf",
          "/system/fonts/Roboto-Regular.ttf", "/system/fonts/RobotoFlex-Regular.ttf",
          "/system/fonts/NotoSans-Regular.ttf"}) {
        std::error_code error;
        if (std::filesystem::is_regular_file(path, error)) {
            return path;
        }
    }
    return nullptr;
}
// Phones are ~2-3x denser than the desktop layout assumes. Layout pixels scale with the display
// density (a little under 1:1 so the 1160x780 console still fits a landscape phone) and every
// finger target is at least 48 dp.
void updateTouchScale() {
    float dpi = 0.0f;
    if (SDL_GetDisplayDPI(0, &dpi, nullptr, nullptr) != 0 || dpi < 80.0f) {
        dpi = 320.0f;
    }
    const float density = dpi / 160.0f;
    ui.scale = std::clamp(density * 0.6f, 1.0f, 3.0f);
    ui.touch_min = 48.0f * density;
}
#endif
} // namespace

float playerOperatorScale() {
    return ui.scale;
}

bool playerConsoleOpen() {
    return ui.open;
}

void loadPlayerOperatorFonts() {
    ImFontConfig config;
    config.OversampleH = 2;
    config.OversampleV = 2;
    static const ImWchar glyphs[] = {0x0020, 0x017F, 0};
#ifdef __ANDROID__
    if (const char *path = systemBoldFont()) {
        ui.heading = ImGui::GetIO().Fonts->AddFontFromFileTTF(path, 27, &config, glyphs);
        ui.title = ImGui::GetIO().Fonts->AddFontFromFileTTF(path, 21, &config, glyphs);
    }
#else
    ui.heading = ImGui::GetIO().Fonts->AddFontFromFileTTF("C:/Windows/Fonts/segoeuib.ttf", 27,
                                                          &config, glyphs);
    ui.title = ImGui::GetIO().Fonts->AddFontFromFileTTF("C:/Windows/Fonts/segoeuib.ttf", 21,
                                                        &config, glyphs);
#endif
}
void showPlayerOperator(const std::string &page) {
    ui.open = true;
    ui.page = page;
    ui.next_refresh = 0;
}
bool playerOperatorOpen() {
#ifdef __ANDROID__
    return ui.open || kidLauncherActive();
#else
    return ui.open;
#endif
}
std::string playerOperatorPage() {
    return ui.page;
}
void closePlayerOperator() {
    ui.open = false;
}
void releasePlayerOperatorUI() {
    releaseArtwork();
}
void drawPlayerOperator(SDL_Renderer *renderer) {
#ifdef __ANDROID__
    if (!ui.open) {
        // Android's visible library/loading/pause screens (an IPC-opened console wins).
        kidLauncherDraw(renderer);
        return;
    }
    updateTouchScale();
#else
    if (!ui.open) {
        return;
    }
#endif
    const bool presenting_load = playerSceneLoad().presenting();
    if (!presenting_load && ImGui::GetTime() >= ui.next_refresh) {
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
    const float width = std::min(S(1160.0f), viewport->WorkSize.x - S(24));
    const float height = std::min(S(780.0f), viewport->WorkSize.y - S(24));
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(.025f, .04f, .07f, .78f));
    const ImGuiWindowFlags input_flags = presenting_load ? ImGuiWindowFlags_NoInputs : 0;
    const auto flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                       ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoDocking | input_flags;
    // Font scale is applied to every window opened below (children, popups); restored after End.
    const float previous_font_scale = ImGui::GetStyle().FontScaleMain;
    ImGui::GetStyle().FontScaleMain = previous_font_scale * ui.scale;
    ImGui::Begin("##RayTrophiOperator", nullptr, flags);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, S(18));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, S(8));
    // Frame height (checkbox, slider, combo) reaches the finger-target minimum on touch screens.
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding,
                        ImVec2(S(12), std::max(S(8), (ui.touch_min - ImGui::GetFontSize()) * .5f)));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(S(12), S(10)));
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(.055f, .075f, .12f, 1));
    ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(.10f, .14f, .21f, 1));
    ImGui::PushStyleColor(ImGuiCol_Text, color(ink));
    ImGui::SetCursorPos(
        ImVec2((viewport->WorkSize.x - width) * .5f, (viewport->WorkSize.y - height) * .5f));
    ImGui::BeginChild("shell", ImVec2(width, height), ImGuiChildFlags_Borders,
                      ImGuiWindowFlags_NoScrollbar | input_flags);
    const auto origin = ImGui::GetCursorScreenPos();
    ImGui::GetWindowDrawList()->AddRectFilled(origin, add(origin, width, 3), mint, 2);
    // Touch screens (landscape phones are short): a slimmer header, no side status, and the
    // action row only on the settings page.
    const bool touch = ui.touch_min > 0;
    const float top = touch ? std::max(S(60), ui.touch_min + S(14)) : S(92);
    const bool actions = !touch || ui.page == "settings";
    const float foot = actions ? std::max(S(44), ui.touch_min) + S(26) : S(14);
    ImGui::SetCursorPos(ImVec2(S(24), touch ? S(12) : S(20)));
    if (auto *logo = brand(renderer)) {
        ImGui::Image(static_cast<ImTextureID>(reinterpret_cast<intptr_t>(logo)), ImVec2(S(42), S(42)));
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
#ifdef __ANDROID__
    label(tr("player.game_menu", "GAME MENU"), muted);
#else
    label(width < S(950) ? tr("player.operator_console", "OPERATOR CONSOLE")
                      : tr("player.operator_console_developer_edition",
                           "OPERATOR CONSOLE  /  DEVELOPER EDITION"),
          muted);
#endif
    ImGui::EndGroup();
    // On touch screens the language lives on the Display page only (the header is too short).
    if (width >= S(760) && !touch) {
        ImGui::SetCursorPos(ImVec2(width - S(318), S(25)));
        languagePicker(S(175));
    }
    ImGui::SetCursorPos(ImVec2(width - S(122), touch ? S(10) : S(24)));
#ifdef __ANDROID__
    if (action(trId("player.close", "Close"), ImVec2(100, 34))) {
#else
    if (action(trId("player.close_esc", "Close / Esc"), ImVec2(100, 34))) {
#endif
        ui.open = false;
    }
    const bool compact = touch || width < S(650);
    const float nav_width = compact ? std::max(S(52.0f), ui.touch_min) : S(190.0f);
    ImGui::SetCursorPos(ImVec2(S(14), top));
    const float nav_height = height - (touch ? top + S(12) : S(178));
    ImGui::BeginChild("navigation", ImVec2(nav_width, nav_height), 0, input_flags);
    const char *ids[] = {"games", "settings", "camera", "players", "display", "audio"};
    const char *titles[] = {
        tr("player.game_library", "Game library"), tr("player.game_settings", "Game settings"),
        tr("player.camera_room", "Camera & room"), tr("player.players", "Players"),
        tr("player.display", "Display"),           tr("player.room_audio", "Room audio")};
    // Four pages on a touch screen: shrink the rows below 48 dp only on screens too short to
    // fit them (the emulator window), never on a normal phone.
    const float row = touch ? std::min(ui.touch_min, (nav_height - S(10) * 3.0f - S(4)) / 4.0f)
                            : S(56);
    for (int i = 0; i < 6; ++i) {
#ifdef __ANDROID__
        if (ids[i] == std::string("camera") || ids[i] == std::string("players")) {
            continue; // camera/perception bridge is not part of the Android build
        }
#endif
        const auto point = ImGui::GetCursorScreenPos();
        const bool chosen = ui.page == ids[i];
        if (ImGui::InvisibleButton(ids[i], ImVec2(nav_width, row))) {
            ui.page = ids[i];
            ui.next_refresh = 0;
        }
        auto *navigation = ImGui::GetWindowDrawList();
        if (chosen || ImGui::IsItemHovered()) {
            navigation->AddRectFilled(point, add(point, nav_width, row),
                                      chosen ? IM_COL32(34, 62, 65, 255) : panel, S(10));
        }
        icon(navigation, add(point, S(22), row * .5f), ids[i], chosen ? mint : muted, S(10));
        if (!compact) {
            navigation->AddText(add(point, S(43), (row - ImGui::GetFontSize()) * .5f),
                                chosen ? mint : muted, titles[i]);
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
    ImGui::SetCursorPos(ImVec2(nav_width + S(42), top + S(2)));
    ImGui::BeginChild("content",
                      ImVec2(width - nav_width - S(68),
                             touch ? height - top - S(2) - foot : height - S(192)),
                      0, input_flags);
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
    ImGui::SetCursorPos(ImVec2(S(24), touch ? height - foot + S(6) : height - S(82)));
    const auto error = ui.error.empty() ? ui.status.value("error", "") : ui.error;
    if (!error.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, .68f, .5f, 1));
        ImGui::PushTextWrapPos(width - S(410));
        ImGui::TextUnformatted(translatedMessage(error).c_str());
        ImGui::PopTextWrapPos();
        ImGui::PopStyleColor();
    } else {
        if (width >= S(900) && !touch) {
            label(tr("player.ready_the_room_let_imagination_play",
                     "Ready the room. Let imagination play."),
                  mint);
#ifdef __ANDROID__
            label(tr("player.touch_hint", "The Back button closes this menu"), muted);
#else
            label(tr("player.f2_console_f6_library_esc_return_to_game",
                     "F2 console  /  F6 library  /  Esc return to game"),
                  muted);
#endif
        }
    }
    // Bottom-right action row; 44 + 26 = 70 on Windows, taller on a touch screen.
    ImGui::SetCursorPos(
        ImVec2(width - S(372), height - std::max(S(44), ui.touch_min) - S(26)));
    if (actions) {
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
        tooltipText(tr("player.apply_save_or_discard_game_settings_edits_before_playing",
                             "Apply, save or discard Game settings edits before playing."));
    }
    ImGui::EndDisabled();
    }
    ImGui::EndChild();
    ImGui::PopStyleColor(3);
    ImGui::PopStyleVar(4);
    ImGui::End();
    ImGui::GetStyle().FontScaleMain = previous_font_scale;
    ImGui::PopStyleColor();
    ImGui::PopStyleVar();
}
} // namespace AppRuntime
