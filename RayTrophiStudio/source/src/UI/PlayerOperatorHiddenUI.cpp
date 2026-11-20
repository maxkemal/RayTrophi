#include "PlayerOperatorWidgets.h"
#include <algorithm>

namespace PlayerConsole {
void hiddenObjects(json& values, json& patch) {
    if (!values.contains("hidden_objects")) {
        return;
    }
    auto design = values["hidden_objects"];
    auto& rows = design["objects"];
    const auto live = ui.status.value("hidden_objects", json::object());
    ImGui::Separator();
    label(tr("player.hidden_objects", "HIDDEN OBJECTS"), mint);
    ImGui::Text(tr("player.hidden_count", "%d objects / %d active"),
                static_cast<int>(rows.size()), live.value("active_count", 0));
    ImGui::TextWrapped("%s", tr("player.hidden_help",
        "Paint reveals the object. Shown keeps it visible. Hidden removes it from the round. "
        "Reset moves enabled objects inside the room. Apply or save your edits before resetting."));
    bool changed = false;
    for (auto& row : rows) {
        const auto name = row.value("object", std::string());
        ImGui::PushID(name.c_str());
        if (ImGui::TreeNode("##object", "%s", name.c_str())) {
            const char* modes[] = {"paint", "shown", "hidden"};
            const char* titles[] = {tr("player.hidden_paint", "Reveal with paint"),
                                    tr("player.hidden_shown", "Always visible"),
                                    tr("player.hidden_off", "Hidden / excluded")};
            const auto current = row.value("visibility", std::string("paint"));
            int selected = current == "shown" ? 1 : current == "hidden" ? 2 : 0;
            if (ImGui::BeginCombo(trId("player.hidden_visibility", "Visibility"), titles[selected])) {
                for (int i = 0; i < 3; ++i) {
                    if (ImGui::Selectable(titles[i], i == selected)) {
                        row["visibility"] = modes[i];
                        changed = true;
                    }
                }
                ImGui::EndCombo();
            }
            bool randomize = row.value("randomize_on_reset", true);
            if (ImGui::Checkbox(trId("player.hidden_randomize", "Move on room reset"), &randomize)) {
                row["randomize_on_reset"] = randomize;
                changed = true;
            }
            float yaw[2] = {row.value("yaw_min", -15.0f), row.value("yaw_max", 15.0f)};
            if (ImGui::SliderFloat2(trId("player.hidden_yaw", "Y rotation range"), yaw, -45, 45,
                                     "%.0f deg")) {
                row["yaw_min"] = std::min(yaw[0], yaw[1]);
                row["yaw_max"] = std::max(yaw[0], yaw[1]);
                changed = true;
            }
            bool player = row.value("blocks_player", false);
            if (ImGui::Checkbox(trId("player.hidden_blocks_player", "Blocks avatar / camera card"),
                                 &player)) {
                row["blocks_player"] = player;
                changed = true;
            }
            bool balloon = row.value("blocks_balloon", false);
            if (ImGui::Checkbox(trId("player.hidden_blocks_balloon", "Blocks carried balloons"),
                                 &balloon)) {
                row["blocks_balloon"] = balloon;
                changed = true;
            }
            ImGui::TextWrapped("%s", tr("player.hidden_passage_help",
                "Leave both obstacle options off for free passage. Paint still hits the object."));
            ImGui::TreePop();
        }
        ImGui::PopID();
    }
    if (ImGui::TreeNode(trId("player.hidden_room", "Placement bounds"))) {
        auto& room = design["room"];
        bool grounded = room.value("placement", std::string("walls_floor")) == "walls_floor";
        if (ImGui::Checkbox(trId("player.hidden_grounded", "Keep on floor near walls"), &grounded)) {
            room["placement"] = grounded ? "walls_floor" : "free";
            changed = true;
        }
        float band = room.value("wall_band", 0.65f);
        ImGui::BeginDisabled(!grounded);
        if (ImGui::SliderFloat(trId("player.hidden_wall_band", "Distance band from wall"),
                               &band, 0.1f, 2, "%.2f m")) {
            room["wall_band"] = band;
            changed = true;
        }
        ImGui::EndDisabled();
        float clearance = room.value("camera_clearance", 3.0f);
        if (ImGui::SliderFloat(trId("player.hidden_camera_clearance", "Keep away from camera"),
                               &clearance, 0, 10, "%.2f m")) {
            room["camera_clearance"] = clearance;
            changed = true;
        }
        const auto detected = live.value("room", json::object());
        bool automatic = room.value("mode", std::string("auto")) == "auto";
        if (ImGui::Checkbox(trId("player.hidden_auto", "Use room wall bounds"), &automatic)) {
            const auto previous = room;
            if (automatic) {
                room = {{"mode", "auto"}};
            } else {
                room = {{"mode", "manual"},
                        {"min", detected.value("min", json::array({-3, 0, -3}))},
                        {"max", detected.value("max", json::array({3, 3, 3}))}};
                if (room["min"].is_null() || room["max"].is_null()) {
                    room["min"] = json::array({-3, 0, -3});
                    room["max"] = json::array({3, 3, 3});
                }
            }
            room["placement"] = previous.value("placement", std::string("walls_floor"));
            room["wall_band"] = previous.value("wall_band", 0.65f);
            room["camera_clearance"] = previous.value("camera_clearance", 3.0f);
            changed = true;
        }
        if (!automatic) {
            float low[3], high[3];
            for (int i = 0; i < 3; ++i) {
                low[i] = room["min"][i].get<float>();
                high[i] = room["max"][i].get<float>();
            }
            const bool low_changed = ImGui::InputFloat3(trId("player.hidden_min", "Minimum XYZ"), low);
            const bool high_changed = ImGui::InputFloat3(trId("player.hidden_max", "Maximum XYZ"), high);
            if (low_changed || high_changed) {
                room["min"] = json::array({low[0], low[1], low[2]});
                room["max"] = json::array({high[0], high[1], high[2]});
                changed = true;
            }
        }
        if (!detected.value("valid", false)) {
            label(tr("player.hidden_invalid_room", "Set valid room bounds before resetting."), violet);
        }
        ImGui::TreePop();
    }
    if (!live.value("error", std::string()).empty()) {
        ImGui::TextWrapped("%s", live["error"].get<std::string>().c_str());
    }
    if (changed) {
        values["hidden_objects"] = design;
        patch["hidden_objects"] = std::move(design);
        ui.game_settings_dirty = true;
    }
}
} // namespace PlayerConsole
