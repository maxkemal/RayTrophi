#include "PlayerOperatorWidgets.h"
#include "Core/PlayerOperator.h"
#include <algorithm>

namespace PlayerConsole {
namespace {
void calibration(const json &settings, bool active) {
    if (!ImGui::CollapsingHeader(
            trId("player.floor_calibration_tracking", "Floor calibration & tracking"))) {
        return;
    }
    ImGui::TextWrapped(
        tr("player.start_test_camera_mark_the_measured_floor_rectangle_in_order_0_0_width_0_wi_"
           "ee21bddc",
           "Start Test camera. Mark the measured floor rectangle in order: (0,0), (width,0), "
           "(width,length), (0,length). Camera orientation must match the room."));
    ImGui::SetNextItemWidth(200);
    ImGui::InputFloat2(trId("player.room_width_length_m", "Room width / length (m)"), ui.room,
                       "%.2f");
    ImGui::SetNextItemWidth(240);
    ImGui::InputFloat3(trId("player.camera_yaw_pitch_roll", "Camera yaw / pitch / roll"), ui.angles,
                       "%.1f");
    if (action(ui.calibrating ? tr("player.clear_four_corners", "Clear four corners")
                              : tr("player.mark_floor_corners", "Mark floor corners"),
               ImVec2(180, 32))) {
        ui.calibrating = true;
        ui.corners.clear();
    }
    ImGui::SameLine();
    ImGui::BeginDisabled(ui.corners.size() != 4 || ui.room[0] <= 0 || ui.room[1] <= 0);
    if (action(trId("player.save_calibration", "Save calibration"), ImVec2(155, 32))) {
        json points = json::array();
        const float x[] = {0, ui.room[0], ui.room[0], 0};
        const float z[] = {0, 0, ui.room[1], ui.room[1]};
        for (int i = 0; i < 4; ++i) {
            points.push_back(
                {{"pixel", {ui.corners[i].x, ui.corners[i].y}}, {"room_xz", {x[i], z[i]}}});
        }
        if (call("calibration.save", {{"points", points},
                                      {"camera_yaw_degrees", ui.angles[0]},
                                      {"camera_pitch_degrees", ui.angles[1]},
                                      {"camera_roll_degrees", ui.angles[2]}})) {
            ui.calibrating = false;
        }
    }
    ImGui::EndDisabled();
    ImGui::TextWrapped(tr("player.saved_calibration_s", "Saved calibration: %s"),
                       settings.value("calibration", "").empty()
                           ? tr("player.not_ready", "Not ready")
                           : tr("player.ready_for_this_camera_and_resolution",
                                "Ready for this camera and resolution"));
    ImGui::BeginDisabled(active);
    bool mirror = settings.value("mirror", false);
    if (ImGui::Checkbox(trId("player.mirror_operator_preview", "Mirror operator preview"),
                        &mirror)) {
        call("camera.configure", {{"mirror", mirror}});
    }
    ImGui::TextWrapped(tr("player.preview_mirror_does_not_change_calibrated_room_coordinates",
                          "Preview mirror does not change calibrated room coordinates."));
    int people = settings.value("people", 4);
    ImGui::SetNextItemWidth(180);
    if (ImGui::SliderInt(trId("player.maximum_players", "Maximum players"), &people, 1, 10)) {
        call("camera.configure", {{"people", people}});
    }
    float confidence = settings.value("confidence", .5f);
    ImGui::SetNextItemWidth(180);
    if (ImGui::SliderFloat(trId("player.tracking_confidence", "Tracking confidence"), &confidence,
                           .1f, 1, "%.2f")) {
        call("camera.configure", {{"confidence", confidence}});
    }
    ImGui::EndDisabled();
}
void preview(SDL_Renderer *renderer, const json &settings, const json &camera) {
    const float available = ImGui::GetContentRegionAvail().x;
    const float height = std::min(270.0f, available * 360.0f / 640.0f);
    const float width = height * 640.0f / 360.0f;
    const auto start = ImGui::GetCursorScreenPos();
    auto *texture = AppRuntime::playerOperatorPreview(renderer);
    if (texture) {
        ImGui::Image(static_cast<ImTextureID>(reinterpret_cast<intptr_t>(texture)),
                     ImVec2(width, height));
    } else {
        ImGui::InvisibleButton("preview", ImVec2(width, height));
        auto *draw = ImGui::GetWindowDrawList();
        draw->AddRectFilled(start, add(start, width, height), IM_COL32(15, 22, 35, 255), 12);
        icon(draw, add(start, width * .5f, height * .43f), "camera", muted, 22);
        draw->PushClipRect(start, add(start, width, height), true);
        draw->AddText(add(start, 20, height - 36), muted,
                      tr("player.live_preview_appears_here_frames_stay_in_memory",
                         "Live preview appears here. Frames stay in memory."));
        draw->PopClipRect();
    }
    if (!texture || !ui.calibrating || camera.value("state", "") != "preview") {
        return;
    }
    if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left) &&
        ui.corners.size() < 4) {
        const auto mouse = ImGui::GetIO().MousePos;
        float u = (mouse.x - start.x) / width;
        if (settings.value("mirror", false)) {
            u = 1 - u;
        }
        const float pixel_x = std::clamp(u * settings["width"].get<float>(), 0.0f,
                                         settings["width"].get<float>() - 1);
        const float pixel_y =
            std::clamp((mouse.y - start.y) / height * settings["height"].get<float>(), 0.0f,
                       settings["height"].get<float>() - 1);
        ui.corners.emplace_back(pixel_x, pixel_y);
    }
    auto *draw = ImGui::GetWindowDrawList();
    for (size_t i = 0; i < ui.corners.size(); ++i) {
        float u = ui.corners[i].x / settings["width"].get<float>();
        if (settings.value("mirror", false)) {
            u = 1 - u;
        }
        const auto p =
            add(start, u * width, ui.corners[i].y / settings["height"].get<float>() * height);
        draw->AddCircleFilled(p, 11, mint);
        draw->AddText(add(p, -4, -8), IM_COL32(10, 25, 30, 255), std::to_string(i + 1).c_str());
    }
}
} // namespace

void camera(SDL_Renderer *renderer) {
    heading(tr("player.camera_room", "Camera & room"),
            tr("player.choose_the_right_lens_check_the_picture_then_prepare_body_tracking",
               "Choose the right lens, check the picture, then prepare body tracking."));
    const auto camera = ui.status["camera"];
    const auto settings = ui.status["preferences"]["camera"];
    const bool active = camera.value("active", false);
    if (!ui.camera_enumeration_requested) {
        ui.camera_enumeration_requested = true;
        if (!active && !camera.value("scanning", false) && camera["catalog"]["devices"].empty()) {
            call("camera.refresh");
        }
    }
    ImGui::BeginDisabled(active || camera.value("scanning", false));
    ImGui::SetNextItemWidth(-140);
    const auto selected_id = settings.value("device_id", "");
    std::string selected_name =
        selected_id.empty() ? tr("player.select_a_camera", "Select a camera")
                            : tr("player.saved_camera_not_detected", "Saved camera (not detected)");
    const auto devices = camera["catalog"]["devices"];
    for (const auto &device : devices) {
        if (device["id"] == selected_id) {
            selected_name = device.value("name", tr("player.camera", "Camera"));
        }
    }
    if (ImGui::BeginCombo("##camera", selected_name.c_str())) {
        for (const auto &device : devices) {
            const auto name = device.value("name", tr("player.camera", "Camera")) + "  #" +
                              std::to_string(device.value("index", 0) + 1);
            if (ImGui::Selectable(name.c_str(), device["id"] == selected_id)) {
                if (call("camera.configure", {{"device_id", device["id"]}})) {
                    call("camera.refresh");
                    ui.corners.clear();
                }
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("%s", device["id"].get<std::string>().c_str());
            }
        }
        ImGui::EndCombo();
    }
    ImGui::SameLine();
    if (action(trId("player.find_cameras", "Find cameras"), ImVec2(128, 32))) {
        call("camera.refresh");
    }
    const auto mode_name = std::to_string(settings.value("width", 1280)) + " x " +
                           std::to_string(settings.value("height", 720)) + " / " +
                           std::to_string(static_cast<int>(settings.value("fps", 30.0))) + " fps";
    ImGui::SetNextItemWidth(-1);
    if (ImGui::BeginCombo(trId("player.capture_mode", "Capture mode"), mode_name.c_str())) {
        for (const auto &device : devices) {
            if (device["id"] != selected_id) {
                continue;
            }
            for (const auto &mode : device["modes"]) {
                const auto title = std::to_string(mode["width"].get<int>()) + " x " +
                                   std::to_string(mode["height"].get<int>()) + " / " +
                                   std::to_string(mode["fps"].get<double>()) + " fps";
                if (ImGui::Selectable(title.c_str())) {
                    call("camera.configure", {{"width", mode["width"]},
                                              {"height", mode["height"]},
                                              {"fps", mode["fps"]}});
                    ui.corners.clear();
                }
            }
        }
        ImGui::EndCombo();
    }
    ImGui::EndDisabled();
    if (devices.empty()) {
        label(camera.value("scanning", false)
                  ? tr("player.looking_for_cameras", "Looking for cameras...")
                  : tr("player.no_camera_detected_connect_one_and_choose_find_cameras",
                       "No camera detected. Connect one and choose Find cameras."),
              muted);
    }
    const auto error = camera.value("error", camera["catalog"].value("error", ""));
    if (!error.empty()) {
        ImGui::TextWrapped("%s", translatedMessage(error).c_str());
    }
    for (const auto &device : devices) {
        if (device["id"] == selected_id && device.contains("error")) {
            ImGui::TextWrapped("%s", translatedMessage(device["error"].get<std::string>()).c_str());
        }
    }
    ImGui::Spacing();
    preview(renderer, settings, camera);
    ImGui::Spacing();
    ImGui::BeginDisabled(active || camera.value("scanning", false));
    if (action(trId("player.test_camera", "Test camera"), ImVec2(130, 38))) {
        call("camera.preview");
    }
    ImGui::SameLine();
    if (action(trId("player.start_tracking", "Start tracking"), ImVec2(145, 38), true)) {
        call("camera.start");
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(!active);
    if (action(trId("player.stop", "Stop"), ImVec2(80, 38))) {
        call("camera.stop");
    }
    ImGui::EndDisabled();
    label((std::string(tr("player.camera_f27869", "Camera: ")) +
           stateLabel(camera.value("state", "stopped")))
              .c_str(),
          mint);
    if (camera.contains("processed_fps")) {
        ImGui::Text(tr("player.processed_1f_fps_requested_1f_fps",
                       "Processed %.1f fps / requested %.1f fps"),
                    camera["processed_fps"].get<double>(), camera.value("requested_fps", 0.0));
    }
    calibration(settings, active);
}
} // namespace PlayerConsole
