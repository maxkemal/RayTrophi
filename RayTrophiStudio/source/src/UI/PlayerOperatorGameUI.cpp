#include "PlayerOperatorWidgets.h"
#include <algorithm>

namespace PlayerConsole {
namespace {
struct Draft {
    std::string project;
    std::uint64_t revision = 0;
    json values = json::object();
    json patch = json::object();
    std::string message;
};
Draft draft;
float number(const char *key, float fallback) {
    return draft.values.contains(key) && draft.values[key].is_number()
               ? draft.values[key].get<float>()
               : fallback;
}
void change(const char *key, const json &value) {
    draft.values[key] = value;
    draft.patch[key] = value;
    ui.game_settings_dirty = true;
    draft.message.clear();
}
void hint(const char *text) {
    ImGui::PushStyleColor(ImGuiCol_Text, color(muted));
    ImGui::TextWrapped("%s", text);
    ImGui::PopStyleColor();
}
void field(const char *key, const char *title, float low, float high, float fallback,
           const char *format, const char *explanation) {
    float value = number(key, fallback);
    label(title);
    ImGui::SetNextItemWidth(std::min(330.0f, ImGui::GetContentRegionAvail().x));
    const auto widget_id = std::string("##") + key;
    if (ImGui::SliderFloat(widget_id.c_str(), &value, low, high, format)) {
        change(key, value);
    }
    if (draft.values.contains(key) && draft.values[key].is_null()) {
        label(tr("player.mixed_across_surfaces_changing_this_applies_to_every_paint_surface",
                 "Mixed across surfaces; changing this applies to every paint surface."),
              violet);
    }
    hint(explanation);
}
void timer(const char *key, const char *title, float minimum, float maximum, float fallback,
           const char *explanation) {
    ImGui::PushID(key);
    bool enabled = number(key, 0) > 0;
    if (ImGui::Checkbox((std::string(title) + "###timer.enabled." + key).c_str(), &enabled)) {
        change(key, enabled ? fallback : 0.0f);
    }
    ImGui::BeginDisabled(!enabled);
    float seconds = std::max(minimum, number(key, fallback));
    ImGui::SetNextItemWidth(std::min(330.0f, ImGui::GetContentRegionAvail().x));
    if (ImGui::SliderFloat(trId("player.seconds", "Seconds"), &seconds, minimum, maximum,
                           tr("player.1f_s", "%.1f s"))) {
        change(key, enabled ? seconds : 0.0f);
    }
    ImGui::EndDisabled();
    hint(explanation);
    ImGui::PopID();
}
bool settingsCall(const std::string &operation, const json &config) {
    json output;
    const auto result = rtapi::playerOperator(operation, config, output);
    ui.next_refresh = 0;
    ui.error = result.ok ? "" : result.error;
    if (!result.ok) {
        return false;
    }
    draft.patch = json::object();
    ui.game_settings_dirty = false;
    draft.message = operation == "game.settings.save" ? "Saved for this game on this Player."
                                                      : "Applied to this session.";
    if (output.value("wet_paint_cleared", false)) {
        draft.message += " Wet paint was cleared.";
    }
    return true;
}
void presets() {
    label(tr("player.room_mood", "ROOM MOOD"), mint);
    const char *names[] = {"calm", "balanced", "lively"};
    const char *titles[] = {tr("player.calm", "Calm"), tr("player.balanced", "Balanced"),
                            tr("player.lively", "Lively")};
    const char *notes[] = {tr("player.fewer_balloons", "Fewer balloons"),
                           tr("player.an_everyday_balance", "An everyday balance"),
                           tr("player.faster_returns", "Faster returns")};
    const ImU32 tints[] = {mint, violet, IM_COL32(246, 184, 135, 255)};
    const float width = (ImGui::GetContentRegionAvail().x - 20) / 3;
    for (int i = 0; i < 3; ++i) {
        if (i) {
            ImGui::SameLine(0, 10);
        }
        ImGui::PushID(names[i]);
        const auto p = ImGui::GetCursorScreenPos();
        const bool chosen = ImGui::InvisibleButton("mood", ImVec2(width, 106));
        auto *draw = ImGui::GetWindowDrawList();
        draw->AddRectFilled(p, add(p, width, 106), panel, 12);
        draw->AddRect(p, add(p, width, 106), ImGui::IsItemHovered() ? tints[i] : border, 12);
        draw->AddCircleFilled(add(p, 26, 27), 13, tints[i]);
        draw->AddCircleFilled(add(p, 22, 23), 3, IM_COL32(255, 255, 255, 100));
        draw->AddBezierCubic(add(p, 26, 40), add(p, 20, 47), add(p, 31, 49), add(p, 26, 57),
                             tints[i], 1.2f);
        draw->PushClipRect(p, add(p, width, 106), true);
        draw->AddText(add(p, 14, 62), ink, titles[i]);
        draw->AddText(add(p, 14, 83), muted, notes[i]);
        draw->PopClipRect();
        if (chosen) {
            json result;
            const auto applied =
                rtapi::playerOperator("game.settings.preset", {{"name", names[i]}}, result);
            if (applied.ok) {
                ui.game_settings_dirty = true;
                draft.patch.update(result["patch"]);
                draft.values.update(result["patch"]);
                draft.message = "Preset selected. Apply or save when ready.";
            } else {
                ui.error = applied.error;
            }
        }
        ImGui::PopID();
    }
    hint(tr("player.presets_do_not_turn_on_automatic_popping_or_change_texture_resolution",
            "Presets do not turn on automatic popping or change texture resolution."));
}
} // namespace
bool applyGameSettingsDraft(bool save) {
    const auto state = ui.status.value("game_settings", json::object());
    if (!state.value("ready", false) || draft.project != state.value("project", "") ||
        draft.revision != state.value("revision", std::uint64_t{0})) {
        ui.error = "Wait for this game's settings to finish loading";
        return false;
    }
    return settingsCall(save ? "game.settings.save" : "game.settings.apply", draft.patch);
}

void gameSettings() {
    heading(tr("player.game_settings", "Game settings"),
            tr("player.prepare_the_pace_and_paint_feel_for_today_s_group",
               "Prepare the pace and paint feel for today's group."));
    const auto status = ui.status.value("game_settings", json::object());
    if (!status.value("ready", false)) {
        hint(tr("player.open_a_prepared_paint_game_and_wait_for_loading_to_finish",
                "Open a prepared paint game and wait for loading to finish."));
        if (status.contains("error")) {
            ImGui::TextWrapped("%s", translatedMessage(status["error"].get<std::string>()).c_str());
        }
        return;
    }
    const auto project = status.value("project", "");
    const auto revision = status.value("revision", std::uint64_t{0});
    if (draft.project != project || draft.revision != revision) {
        draft = Draft{};
        draft.project = project;
        draft.revision = revision;
        ui.game_settings_dirty = false;
    }
    if (draft.patch.empty()) {
        draft.values = status["values"];
    }
    presets();
    ImGui::Spacing();
    if (draft.values.contains("autostart")) {
        label(tr("player.play_mode", "PLAY MODE"), mint);
        bool automatic = draft.values.value("autostart", false);
        if (ImGui::RadioButton(trId("player.players_pop_the_balloons", "Players pop the balloons"),
                               !automatic)) {
            change("autostart", false);
        }
        if (ImGui::GetContentRegionAvail().x >= 430) {
            ImGui::SameLine();
        }
        if (ImGui::RadioButton(trId("player.automatic_demo", "Automatic demo"), automatic)) {
            change("autostart", true);
        }
        hint(tr(
            "player.player_mode_uses_mouse_or_body_input_automatic_demo_also_pops_random_balloons",
            "Player mode uses mouse or body input. Automatic demo also pops random balloons."));
        ImGui::Separator();
        label(tr("player.balloons", "BALLOONS"), violet);
        int count = static_cast<int>(number("balloons", 3));
        label(tr("player.balloon_count", "Balloon count"));
        ImGui::SetNextItemWidth(std::min(330.0f, ImGui::GetContentRegionAvail().x));
        if (ImGui::SliderInt("##balloon_count", &count, 3,
                             std::max(3, status.value("max_balloons", 3)))) {
            change("balloons", count);
        }
        hint(tr("player.limited_by_the_scene_s_prepared_balloon_meshes_and_particle_budget",
                "Limited by the scene's prepared balloon meshes and particle budget."));
        field("respawn_seconds", tr("player.return_after_a_pop", "Return after a pop"), .25f, 60,
              3.5f, tr("player.2f_s", "%.2f s"),
              tr("player.how_long_a_popped_balloon_waits_before_appearing_again",
                 "How long a popped balloon waits before appearing again."));
        field(
            "birth_seconds", tr("player.inflate_playful_birth", "Inflate / playful birth"), 0, 3,
            .75f, tr("player.2f_s", "%.2f s"),
            tr("player.mesh_balloons_grow_from_tiny_bounce_and_settle_0_means_an_instant_"
               "appearance",
               "Mesh balloons grow from tiny, bounce and settle. 0 means an instant appearance."));
        ImGui::BeginDisabled(!draft.values.value("autostart", false));
        field("interval_min", tr("player.automatic_pop_minimum", "Automatic pop / minimum"), 2, 30,
              2.5f, tr("player.1f_s", "%.1f s"),
              tr("player.only_used_in_automatic_demo_mode", "Only used in automatic demo mode."));
        field("interval_max", tr("player.automatic_pop_maximum", "Automatic pop / maximum"), 2, 30,
              5, tr("player.1f_s", "%.1f s"),
              tr("player.must_be_at_least_the_minimum_interval",
                 "Must be at least the minimum interval."));
        ImGui::EndDisabled();
    }
    if (draft.values.contains("lifetime_seconds")) {
        ImGui::Separator();
        label(tr("player.paint_feel_all_surfaces", "PAINT FEEL / ALL SURFACES"), mint);
        field("lifetime_seconds", tr("player.drying_time", "Drying time"), .05f, 120, 12,
              tr("player.2f_s", "%.2f s"),
              tr("player.longer_keeps_paint_wet_for_longer_it_does_not_erase_the_dry_colour",
                 "Longer keeps paint wet for longer; it does not erase the dry colour."));
        field("diffusion", tr("player.paint_spread", "Paint spread"), 0, 2, .24f, "%.2f",
              tr("player.higher_lets_wet_colour_spread_farther",
                 "Higher lets wet colour spread farther."));
        field("runoff", tr("player.paint_flow", "Paint flow"), 0, 2, .35f, "%.2f",
              tr("player.higher_makes_wet_paint_flow_more_strongly_along_surfaces",
                 "Higher makes wet paint flow more strongly along surfaces."));
        const int sizes[] = {512, 1024, 2048};
        const auto current = draft.values["resolution"];
        const auto title =
            current.is_null() ? std::string(tr("player.mixed_surface_sizes", "Mixed surface sizes"))
                              : std::to_string(current.get<int>());
        ImGui::SetNextItemWidth(240);
        if (ImGui::BeginCombo(trId("player.texture_resolution", "Texture resolution"),
                              title.c_str())) {
            for (int size : sizes) {
                if (ImGui::Selectable(std::to_string(size).c_str(), current == size)) {
                    change("resolution", size);
                }
            }
            ImGui::EndCombo();
        }
        hint(tr(
            "player.applies_to_every_paint_surface_changing_resolution_clears_transient_wet_paint",
            "Applies to every paint surface. Changing resolution clears transient wet paint."));
    }
    if (draft.values.contains("idle_reset_seconds")) {
        ImGui::Separator();
        label(tr("player.room_reset", "ROOM RESET"), violet);
        timer(
            "idle_reset_seconds", tr("player.reset_after_inactivity", "Reset after inactivity"), 10,
            3600, 60,
            tr("player.restarts_the_room_after_this_long_without_a_player_pop_off_means_no_idle_"
               "reset",
               "Restarts the room after this long without a player pop. Off means no idle reset."));
        timer("finale_reset_seconds", tr("player.reset_after_the_finale", "Reset after the finale"),
              3, 120, 10,
              tr("player.delay_after_the_room_goal_celebration_off_keeps_the_finished_room",
                 "Delay after the room-goal celebration. Off keeps the finished room."));
        bool goal = number("room_goal", 0) > 0;
        if (ImGui::Checkbox(
                trId("player.enable_room_progress_finale", "Enable room progress / finale"),
                &goal)) {
            change("room_goal", goal ? .5f : 0.0f);
        }
        ImGui::BeginDisabled(!goal);
        field("room_goal", tr("player.painted_coverage_goal", "Painted coverage goal"), .05f, 1,
              .5f, "%.2f",
              tr("player.the_finale_happens_when_this_fraction_of_the_paintable_area_has_been_"
                 "coloured",
                 "The finale happens when this fraction of the paintable area has been coloured."));
        ImGui::EndDisabled();
    }
    ImGui::Separator();
    hint(tr("player.only_edited_fields_are_applied_demo_changes_renew_balloons_and_clear_in_fli_"
            "d0535529",
            "Only edited fields are applied. Demo changes renew balloons and clear in-flight "
            "effects; texture resizing clears wet paint."));
    if (action(trId("player.apply_to_this_session", "Apply to this session"), ImVec2(205, 40),
               true)) {
        settingsCall("game.settings.apply", draft.patch);
    }
    if (ImGui::GetContentRegionAvail().x >= 400) {
        ImGui::SameLine();
    }
    if (action(trId("player.save_for_this_game", "Save for this game"), ImVec2(185, 40))) {
        settingsCall("game.settings.save", draft.patch);
    }
    if (!draft.patch.empty()) {
        label(tr("player.unapplied_edits_apply_or_save_before_playing",
                 "Unapplied edits / Apply or save before playing."),
              violet);
        if (action(trId("player.discard_edits", "Discard edits"), ImVec2(150, 32))) {
            draft.patch = json::object();
            draft.values = status["values"];
            ui.game_settings_dirty = false;
            draft.message.clear();
        }
    }
    hint(tr("player.saved_locally_for_this_game_on_this_player_the_original_rtp_is_unchanged",
            "Saved locally for this game on this Player. The original .rtp is unchanged."));
    if (!draft.message.empty()) {
        label(translatedMessage(draft.message).c_str(), mint);
    }
    if (status.value("saved", false)) {
        label(tr("player.saved_player_defaults_exist_for_this_game",
                 "Saved Player defaults exist for this game."),
              violet);
    }
    if (!status.value("error", "").empty()) {
        ImGui::TextWrapped("%s", translatedMessage(status["error"].get<std::string>()).c_str());
    }
    ImGui::Separator();
    label(tr("player.quick_room_actions", "QUICK ROOM ACTIONS"), mint);
    if (action(trId("player.clear_paint_f8", "Clear paint / F8"), ImVec2(180, 36))) {
        call("game.settings.clear_paint");
    }
    if (ImGui::GetContentRegionAvail().x >= 400) {
        ImGui::SameLine();
    }
    ImGui::BeginDisabled(!draft.values.contains("balloons"));
    if (action(trId("player.renew_popped_balloons", "Renew popped balloons"), ImVec2(215, 36))) {
        call("game.settings.renew_balloons");
    }
    ImGui::EndDisabled();
    hint(tr("player.clear_paint_keeps_the_current_game_and_balloons_new_drops_can_still_paint_a_"
            "0ba46c95",
            "Clear paint keeps the current game and balloons. New drops can still paint as they "
            "land."));
    if (action(trId("player.reset_room_f9", "Reset room / F9"), ImVec2(180, 36))) {
        call("game.reset");
    }
    if (ImGui::TreeNode(trId("player.restore_the_authored_game", "Restore the authored game"))) {
        hint(tr("player.removes_this_game_s_saved_player_profile_and_reloads_the_original_scene_"
                "cur_94ac2970",
                "Removes this game's saved Player profile and reloads the original scene. Current "
                "paint and session changes are lost. Stop camera input first."));
        if (action(trId("player.restore_and_reload", "Restore and reload"), ImVec2(205, 36))) {
            if (call("game.settings.restore")) {
                draft = Draft{};
            }
        }
        ImGui::TreePop();
    }
}
} // namespace PlayerConsole
