#include "PlayerOperatorWidgets.h"
#include <fstream>

namespace PlayerConsole {
void players() {
    heading(tr("player.player_appearance", "Player appearance"),
            tr("player.use_a_live_camera_cutout_or_a_prepared_3d_character_driven_by_body_tracking",
               "Use a live camera cutout or a prepared 3D character driven by body tracking."));
    const bool active = ui.status["camera"].value("active", false);
    ImGui::BeginDisabled(active);
    if (action(trId("player.camera_card", "Camera card"), ImVec2(180, 60),
               ui.status["preferences"]["appearance"] == "camera")) {
        call("appearance.camera");
    }
    ImGui::SameLine();
    label(tr("player.cutout_from_the_camera_existing_video_card_renderer",
             "Cutout from the camera\nExisting video-card renderer"),
          muted);
    ImGui::Spacing();
    ImGui::Separator();
    label(tr("player.3d_avatars", "3D AVATARS"), violet);
    const auto characters = ui.status.value("characters", json::array());
    if (characters.empty()) {
        ImGui::TextWrapped(tr("player.this_scene_has_no_character_prepare_a_rigged_character_in_"
                              "studio_and_save_t_5b1f2051",
                              "This scene has no character. Prepare a rigged character in Studio "
                              "and save the scene to use avatar mode."));
    }
    for (const auto &character : characters) {
        const auto name = character["name"].get<std::string>();
        const auto title = name + (character.value("ready", false)
                                       ? tr("player.prepared", "  / Prepared")
                                       : tr("player.needs_body_mapping", "  / Needs body mapping"));
        ImGui::PushID(name.c_str());
        if (ImGui::Selectable((title + "###avatar").c_str(), ui.avatar == name, 0, ImVec2(0, 34))) {
            ui.avatar = name;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s", translatedMessage(character.value("reason", "")).c_str());
        }
        ImGui::PopID();
    }
    ImGui::SetNextItemWidth(160);
    ImGui::InputInt(trId("player.tracked_person_id", "Tracked person ID"), &ui.person_id);
    ImGui::TextWrapped(tr("player.person_ids_appear_in_the_live_tracking_check_below_each_avatar_"
                          "binds_to_one_42cbe919",
                          "Person IDs appear in the live tracking check below. Each avatar binds "
                          "to one person; it is not copied for every player."));
    ImGui::BeginDisabled(ui.avatar.empty());
    if (action(trId("player.use_selected_avatar", "Use selected avatar"), ImVec2(210, 42), true)) {
        call("avatar.bind", {{"character", ui.avatar}, {"person_id", ui.person_id}});
    }
    ImGui::EndDisabled();
    if (ImGui::CollapsingHeader(trId("player.developer_import_prepared_rig_mapping",
                                     "Developer / import prepared rig mapping"))) {
        ImGui::InputTextWithHint(
            "##mapping", tr("player.path_to_body_mapping_json", "Path to body mapping JSON"),
            ui.mapping_path, sizeof(ui.mapping_path));
        if (action(trId("player.bind_from_mapping_file", "Bind from mapping file"))) {
            std::ifstream stream(AppRuntime::projectPathFromUtf8(ui.mapping_path),
                                 std::ios::binary);
            const auto document = json::parse(stream, nullptr, false);
            if (document.is_discarded() || !document.is_object()) {
                ui.error = "Choose a valid mapping JSON object";
            } else {
                call(
                    "avatar.bind",
                    {{"character", ui.avatar}, {"person_id", ui.person_id}, {"mapping", document}});
            }
        }
    }
    ImGui::EndDisabled();
    if (active) {
        label(tr("player.stop_camera_input_before_changing_appearance",
                 "Stop camera input before changing appearance."),
              muted);
    }
    ImGui::Spacing();
    label(tr("player.live_tracking_check", "LIVE TRACKING CHECK"), mint);
    const auto persons = ui.status["body"].value("persons", json::array());
    ImGui::TextUnformatted(Localization::format("player.tracked_count", "{count} player(s) tracked",
                                                {{"count", persons.size()}})
                               .c_str());
    for (const auto &person : persons) {
        ImGui::Text(tr("player.person_d_confidence_0f", "Person %d / confidence %.0f%%"),
                    person["id"].get<int>(), person["confidence"].get<double>() * 100);
    }
    if (persons.empty()) {
        ImGui::TextWrapped(
            tr("player.start_tracking_in_camera_room_then_stand_inside_the_calibrated_floor_area",
               "Start tracking in Camera & room, then stand inside the calibrated floor area."));
    }
}

void display() {
    heading(tr("player.display_performance", "Display & performance"),
            tr("player.set_up_the_game_screen_or_projector_then_choose_the_render_size",
               "Set up the game screen or projector, then choose the render size."));
    label(tr("common.language", "Language / Dil"));
    languagePicker(230);
    const auto locale_error = ui.status.value("language", json::object()).value("error", "");
    if (!locale_error.empty()) {
        ImGui::TextWrapped("%s", translatedMessage(locale_error).c_str());
    }
    ImGui::TextWrapped("%s", tr("common.language_help",
                                "The interface language changes immediately and is remembered."));
    ImGui::Spacing();
    const auto status = ui.status["display"];
    for (const auto &monitor : ui.status.value("monitors", json::array())) {
        const int index = monitor["index"];
        const auto title =
            std::string(tr("player.screen", "Screen ")) + std::to_string(index + 1) + " / " +
            monitor.value("name", tr("player.display", "Display")) +
            (monitor.value("selected", false) ? tr("player.active", "  / Active") : "");
        if (action((title + "###monitor." + std::to_string(index)).c_str(), ImVec2(-1, 38))) {
            call("display.monitor", {{"index", index}});
        }
    }
    ImGui::Spacing();
    bool fullscreen = status.value("fullscreen", false);
    if (ImGui::Checkbox(trId("player.fullscreen_presentation", "Fullscreen presentation"),
                        &fullscreen)) {
        call("display.configure", {{"fullscreen", fullscreen}});
    }
    const auto size = status.value("render_base", json::array({0, 0}));
    const auto title =
        std::to_string(size[0].get<int>()) + " x " + std::to_string(size[1].get<int>());
    ImGui::SetNextItemWidth(260);
    if (ImGui::BeginCombo(trId("player.game_resolution", "Game resolution"), title.c_str())) {
        for (const auto &mode : status.value("resolutions", json::array())) {
            const auto name =
                std::to_string(mode[0].get<int>()) + " x " + std::to_string(mode[1].get<int>());
            if (ImGui::Selectable(name.c_str(), mode == size)) {
                call("display.configure", {{"render_width", mode[0]}, {"render_height", mode[1]}});
            }
        }
        ImGui::EndCombo();
    }
    float scale = status.value("render_scale", 1.0f);
    ImGui::SetNextItemWidth(260);
    if (ImGui::SliderFloat(trId("player.render_scale", "Render scale"), &scale, .25f, 1, "%.2f")) {
        call("display.configure", {{"render_scale", scale}});
    }
    const int pacing = status.value("pacing_hz", 60);
    ImGui::SetNextItemWidth(260);
    if (ImGui::BeginCombo(trId("player.frame_pacing", "Frame pacing"),
                          pacing ? (std::to_string(pacing) + " Hz").c_str()
                                 : tr("player.uncapped", "Uncapped"))) {
        for (int hz : {30, 60, 120, 0}) {
            const auto name = hz ? std::to_string(hz) + " Hz" : tr("player.uncapped", "Uncapped");
            if (ImGui::Selectable(name.c_str(), hz == pacing)) {
                call("display.configure", {{"pacing_hz", hz}});
            }
        }
        ImGui::EndCombo();
    }
    bool hud = status.value("hud", true);
    if (ImGui::Checkbox(trId("player.developer_performance_hud", "Developer performance HUD"),
                        &hud)) {
        call("display.configure", {{"hud", hud}});
    }
    label(tr("player.window_resizing_keeps_the_chosen_game_resolution",
             "Window resizing keeps the chosen game resolution."),
          muted);
}

void audio() {
    heading(tr("player.room_audio", "Room audio"),
            tr("player.set_a_comfortable_level_before_the_children_enter_the_play_area",
               "Set a comfortable level before the children enter the play area."));
    if (!ui.status.contains("audio")) {
        label(tr("player.audio_service_unavailable", "Audio service unavailable"), muted);
        return;
    }
    for (const auto &bus : ui.status["audio"].value("buses", json::array())) {
        if (bus.value("name", "") != "Master") {
            continue;
        }
        float gain = bus.value("gain", 1.0f);
        bool muted = bus.value("muted", false);
        ImGui::SetNextItemWidth(320);
        if (ImGui::SliderFloat(trId("player.master_volume", "Master volume"), &gain, 0, 1,
                               "%.2f")) {
            call("audio.configure", {{"gain", gain}});
        }
        if (ImGui::Checkbox(trId("player.mute_room_audio", "Mute room audio"), &muted)) {
            call("audio.configure", {{"muted", muted}});
        }
    }
    ImGui::Spacing();
    label(tr("player.output_device_windows_default", "Output device: Windows default"), mint);
    ImGui::TextWrapped(
        tr("player.select_speakers_or_hdmi_audio_in_windows_player_uses_the_shared_audio_service",
           "Select speakers or HDMI audio in Windows. Player uses the shared audio service."));
}
} // namespace PlayerConsole
