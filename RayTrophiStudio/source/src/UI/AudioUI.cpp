#include "UI/AudioUI.h"
#include "Audio/AudioService.h"
#include "Audio/AudioAuthoring.h"
#include "Audio/AudioWeatherBridge.h"
#include "UI/AudioReactionUI.h"
#include "scene_ui.h"
#include "scene_data.h"
#include "Camera.h"
#include "SceneSelection.h"
#include "AssetRegistry.h"
#include "Api/RtApi.h"
#include "Api/RtPython.h"
#include "imgui.h"
#include "imgui_internal.h"
#include "ImGuizmo.h"
#include <cmath>
#include <cstdio>
#include <cstring>

namespace AudioUI {
namespace {
using json = nlohmann::json;
std::string selected;
std::string error;
char path[1024]{};
json dragBefore;
bool dragging = false;

std::string chooseAudioFile() {
    const auto chosen = SceneUI::openFileDialogW(
        L"Audio Files (WAV, FLAC, MP3)\0*.wav;*.flac;*.mp3\0"
        L"WAV Files\0*.wav\0FLAC Files\0*.flac\0MP3 Files\0*.mp3\0");
    if (!chosen.empty()) {
        std::snprintf(path, sizeof(path), "%s", chosen.c_str());
    }
    return chosen;
}

json call(UIContext& ctx, const std::string& op, const json& p = json::object()) {
    try {
        auto result = audio::author(ctx, ctx.scene_ui_ptr ? &ctx.scene_ui_ptr->history : nullptr, op, p);
        error.clear();
        return result;
    } catch (const std::exception& e) {
        error = e.what();
        return {};
    }
}

void select(UIContext& ctx, const std::string& id) {
    const auto result = call(ctx, "select", {{"id", id}});
    if (result.contains("id")) {
        selected = id;
    }
}
void connectEnvironment(UIContext& ctx, const json& params = json::object()) {
    const auto installed = call(ctx, "environment_setup", params);
    if (installed.contains("wind_source_id")) {
        select(ctx, installed.at("wind_source_id"));
    }
}
}

void addSource(UIContext& ctx) {
    const auto e = call(ctx, "create");
    if (e.contains("id")) {
        select(ctx, e.at("id"));
    }
}

void appendAsset(UIContext& ctx, const AssetRecord& asset, float x, float y, float z) {
    const auto source = call(ctx, "append", {
        {"path", audio::pathUtf8(asset.entry_path)}, {"license", asset.license},
        {"author", asset.author}, {"source", asset.source}, {"tags", asset.tags},
        {"name", asset.name}, {"position", json::array({x, y, z})}});
    if (source.contains("id")) {
        select(ctx, source.at("id"));
    }
}

bool editSelection(UIContext& ctx, bool duplicate) {
    if (ctx.selection.selected.type != SelectableType::AudioSource) {
        return false;
    }
    const auto result = call(ctx, duplicate ? "duplicate" : "remove",
                             {{"id", ctx.selection.selected.audio_source_id}});
    if (result.contains("id")) {
        select(ctx, result.at("id"));
    } else if (result.value("ok", false)) {
        ctx.selection.clearSelection();
        selected.clear();
    }
    return true;
}

void drawAssetActions(UIContext& ctx, const AssetRecord& asset) {
    if (asset.asset_kind != "audio") {
        return;
    }
    ImGui::Text("%.2f seconds, %u channels, %u Hz", asset.audio_duration,
                asset.audio_channels, asset.audio_sample_rate);
    if (ImGui::Button("Audio preview")) {
        call(ctx, "preview_file", {{"path", audio::pathUtf8(asset.entry_path)}});
    }
    ImGui::SameLine();
    if (ImGui::Button("Stop audio preview")) {
        call(ctx, "preview_stop");
    }
    if (!error.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.3f, 1.0f));
        ImGui::TextWrapped("Audio: %s", error.c_str());
        ImGui::PopStyleColor();
    }
}

void drawHierarchy(UIContext& ctx) {
    if (!ImGui::TreeNode("Audio Sources")) {
        return;
    }
    const auto data = ctx.scene.audio->snapshot();
    for (const auto& e : data.at("emitters")) {
        const std::string id = e.at("id");
        const bool active = ctx.selection.selected.type == SelectableType::AudioSource &&
                            ctx.selection.selected.audio_source_id == id;
        if (ImGui::Selectable((e.at("name").get<std::string>() + "##audio_tree_" + id).c_str(), active)) {
            select(ctx, id);
        }
    }
    ImGui::TreePop();
}

void drawProperties(UIContext& ctx) {
    if (ctx.selection.selected.type == SelectableType::AudioSource) {
        selected = ctx.selection.selected.audio_source_id;
    } else {
        selected.clear();
    }
    if (!ImGui::CollapsingHeader("Scene Audio")) {
        return;
    }
    if (!error.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.3f, 1.0f));
        ImGui::TextWrapped("Audio: %s", error.c_str());
        ImGui::PopStyleColor();
        if (ImGui::SmallButton("Dismiss audio error")) {
            error.clear();
        }
    }
    auto data = ctx.scene.audio->snapshot();
    if (ImGui::Button("Connect starter wind / rain")) {
        connectEnvironment(ctx);
    }
    ImGui::SameLine();
    if (ImGui::Button("Choose environment Assets folder")) {
        const auto folder = SceneUI::selectFolderDialogW(L"Select Assets root containing audio/cc0_starter");
        if (!folder.empty()) {
            connectEnvironment(ctx, {{"assets_root", folder}});
        }
    }
    ImGui::TextWrapped("Connect creates automatic weather sources from the CC0 starter pack. "
                       "For domains, select a sound source and choose Domain Contact in Reactive Audio.");
    ImGui::InputText("Selected audio file", path, sizeof(path), ImGuiInputTextFlags_ReadOnly);
    if (ImGui::Button("Import WAV / FLAC / MP3")) {
        const auto chosen = chooseAudioFile();
        if (!chosen.empty()) {
            call(ctx, "asset_import", {{"path", chosen}});
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Add Audio Source")) {
        addSource(ctx);
    }
    auto status = ctx.scene.audio->invoke("status", json::object());
    bool mute = status.at("preview_muted");
    if (ImGui::Checkbox("Mute preview", &mute)) {
        call(ctx, "preview_mute", {{"muted", mute}});
    }
    ImGui::SameLine();
    if (ImGui::Button("Stop all audio")) {
        call(ctx, "stop_all");
    }
    ImGui::Text("Device: %s", status.at("device").get<std::string>().c_str());
    if (ImGui::TreeNode("Audio assets")) {
        for (const auto& asset : data.at("assets")) {
            const std::string id = asset.at("id");
            ImGui::PushID(id.c_str());
            ImGui::Text("%s (%.2fs, %d ch, %d Hz)", asset.at("name").get<std::string>().c_str(),
                        asset.at("duration").get<double>(), asset.at("channels").get<int>(),
                        asset.at("sample_rate").get<int>());
            if (ImGui::SmallButton("Preview")) {
                call(ctx, "preview", {{"id", id}});
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("Stop preview")) {
                call(ctx, "preview_stop");
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("Relink to file")) {
                const auto chosen = chooseAudioFile();
                if (!chosen.empty()) {
                    call(ctx, "asset_relink", {{"id", id}, {"path", chosen}});
                }
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("Remove asset")) {
                call(ctx, "asset_remove", {{"id", id}});
            }
            ImGui::PopID();
        }
        ImGui::TreePop();
    }
    for (const auto& e : data.at("emitters")) {
        const std::string id = e.at("id");
        if (ImGui::Selectable((e.at("name").get<std::string>() + "##" + id).c_str(), selected == id)) {
            select(ctx, id);
        }
    }
    for (const auto& e : data.at("emitters")) {
        if (e.at("id") != selected) {
            continue;
        }
        const auto id = selected;
        const std::string assetId = e.at("asset_id");
        bool stereoAsset = false;
        for (const auto& asset : data.at("assets")) {
            if (asset.at("id") == assetId) {
                stereoAsset = asset.at("channels") == 2;
            }
        }
        if (assetId.empty()) {
            ImGui::TextWrapped("No audio file is linked. Import a file, then choose it in Asset.");
        } else if (stereoAsset) {
            ImGui::TextWrapped("Stereo asset: ambience playback is enabled. "
                               "For 3D positioning, choose a mono asset.");
        }
        char parent[256]{};
        std::snprintf(parent, sizeof(parent), "%s", e.value("parent", std::string()).c_str());
        if (ImGui::InputText("Parent object", parent, sizeof(parent), ImGuiInputTextFlags_EnterReturnsTrue)) {
            call(ctx, "update", {{"id", id}, {"parent", parent}});
        }
        char name[256]{};
        const auto text = e.at("name").get<std::string>();
        std::snprintf(name, sizeof(name), "%s", text.c_str());
        if (ImGui::InputText("Source name", name, sizeof(name), ImGuiInputTextFlags_EnterReturnsTrue)) {
            call(ctx, "update", {{"id", id}, {"name", name}});
        }
        if (ImGui::BeginCombo("Asset", e.at("asset_id").get<std::string>().c_str())) {
            for (const auto& asset : data.at("assets")) {
                if (ImGui::Selectable(asset.at("name").get<std::string>().c_str())) {
                    call(ctx, "update", {{"id", id}, {"asset_id", asset.at("id")},
                                         {"spatial", asset.at("channels") == 1}});
                }
            }
            if (ImGui::Selectable("Unbound")) {
                call(ctx, "update", {{"id", id}, {"asset_id", ""}});
            }
            ImGui::EndCombo();
        }
        for (const char* key : {"position", "direction"}) {
            float v[3] = {e.at(key)[0], e.at(key)[1], e.at(key)[2]};
            if (ImGui::InputFloat3(key, v, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue)) {
                call(ctx, "update", {{"id", id}, {key, json::array({v[0], v[1], v[2]})}});
            }
        }
        for (const char* key : {"gain", "pitch", "min_distance", "max_distance", "rolloff",
                                "cone_inner", "cone_outer", "cone_gain"}) {
            float value = e.at(key);
            if (ImGui::InputFloat(key, &value, 0, 0, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue)) {
                call(ctx, "update", {{"id", id}, {key, value}});
            }
        }
        for (const char* key : {"loop", "spatial"}) {
            bool value = e.at(key);
            const bool spatial = std::strcmp(key, "spatial") == 0;
            ImGui::BeginDisabled(spatial && stereoAsset);
            if (ImGui::Checkbox(spatial ? "3D positional (mono)" : "Loop", &value)) {
                call(ctx, "update", {{"id", id}, {key, value}});
            }
            ImGui::EndDisabled();
        }
        if (ImGui::BeginCombo("Bus", e.at("bus").get<std::string>().c_str())) {
            for (const char* bus : {"Master", "Weather", "Effects", "Ambience", "Music"}) {
                if (ImGui::Selectable(bus, e.at("bus") == bus)) {
                    call(ctx, "update", {{"id", id}, {"bus", bus}});
                }
            }
            ImGui::EndCombo();
        }
        for (const char* op : {"play", "pause", "stop", "duplicate", "remove"}) {
            ImGui::BeginDisabled(std::strcmp(op, "play") == 0 && assetId.empty());
            if (ImGui::Button(op)) {
                const auto result = call(ctx, op, {{"id", id}});
                if (op == std::string("duplicate") && result.contains("id")) {
                    select(ctx, result.at("id"));
                } else if (op == std::string("remove") && result.value("ok", false)) {
                    selected.clear();
                    ctx.selection.clearSelection();
                }
            }
            ImGui::EndDisabled();
            ImGui::SameLine();
        }
        ImGui::NewLine();
        AudioReactionUI::draw(ctx, e);
    }
}

void updateListener(UIContext& ctx) {
    audio::syncScene(ctx);
    audio::tickWeatherReactions(ctx);
    if (!ctx.scene.camera) {
        return;
    }
    const auto& camera = *ctx.scene.camera;
    const Vec3 direction = camera.lookat - camera.lookfrom;
    ctx.scene.audio->listener(camera.lookfrom.x, camera.lookfrom.y, camera.lookfrom.z,
        direction.x, direction.y, direction.z, camera.vup.x, camera.vup.y, camera.vup.z);
}

void drawViewport(UIContext& ctx, bool ortho, bool showGizmos, bool& hit) {
    selected = ctx.selection.selected.type == SelectableType::AudioSource ?
        ctx.selection.selected.audio_source_id : std::string();
    if (!showGizmos || !ctx.scene.camera) {
        return;
    }
    if (rtpython::wantsInputCapture() || rtapi::renderOutputPending()) {
        dragging = false;
        return;
    }
    const auto data = ctx.scene.audio->snapshot();
    const auto& cam = *ctx.scene.camera;
    const auto size = ImGui::GetIO().DisplaySize;
    if (size.x <= 0 || size.y <= 0) {
        return;
    }
    const Vec3 f = (cam.lookat - cam.lookfrom).normalize();
    const Vec3 r = f.cross(cam.vup).normalize();
    const Vec3 u = r.cross(f);
    const float aspect = size.x / size.y;
    const float tangent = std::tan(cam.vfov * 0.00872664626f);
    auto project = [&](const Vec3& p, ImVec2& screen) {
        const auto delta = p - cam.lookfrom;
        const float depth = delta.dot(f);
        if (!ortho && depth <= 0.1f) {
            return false;
        }
        const float half = ortho ? cam.ortho_height * 0.5f : depth * tangent;
        if (half <= 1e-6f) {
            return false;
        }
        screen = ImVec2((0.5f + delta.dot(r) / (2 * half * aspect)) * size.x,
                        (0.5f - delta.dot(u) / (2 * half)) * size.y);
        return true;
    };
    auto* draw = ImGui::GetBackgroundDrawList();
    const auto* hovered = ImGui::GetCurrentContext()->HoveredWindow;
    const bool viewportInput = !ImGui::GetIO().WantCaptureMouse ||
        (hovered && (std::strcmp(hovered->Name, "##ViewportAssetDropTarget") == 0 ||
                     std::strcmp(hovered->Name, "##RayTrophiDockHost") == 0));
    for (const auto& e : data.at("emitters")) {
        const std::string id = e.at("id");
        json pose;
        try {
            pose = audio::worldPose(ctx, e);
        } catch (const std::exception&) {
            continue;
        }
        const Vec3 p(pose.at("position")[0], pose.at("position")[1], pose.at("position")[2]);
        ImVec2 screen;
        if (!project(p, screen)) {
            continue;
        }
        draw->AddCircleFilled(screen, 7, selected == id ? IM_COL32(255, 185, 70, 255) :
                                                       IM_COL32(80, 205, 255, 255));
        draw->AddText(ImVec2(screen.x + 10, screen.y - 8), IM_COL32(80, 205, 255, 255), "Audio");
        const auto mouse = ImGui::GetIO().MousePos;
        const float dx = mouse.x - screen.x;
        const float dy = mouse.y - screen.y;
        if (dx * dx + dy * dy < 144 && viewportInput && !ImGuizmo::IsUsing()) {
            hit = true;
            if (ImGui::IsMouseClicked(0)) {
                select(ctx, id);
            }
        }
        if (ctx.selection.selected.type != SelectableType::AudioSource ||
            ctx.selection.selected.audio_source_id != id) {
            continue;
        }
        if (e.at("spatial").get<bool>()) {
            for (const char* key : {"min_distance", "max_distance"}) {
                const float radius = e.at(key);
                ImVec2 last;
                bool valid = false;
                for (int segment = 0; segment <= 64; ++segment) {
                    const float angle = segment * 0.09817477042f;
                    ImVec2 point;
                    const bool current = project(p + Vec3(std::cos(angle) * radius, 0,
                                                         std::sin(angle) * radius), point);
                    if (current && valid) {
                        draw->AddLine(last, point, IM_COL32(80, 205, 255, 100));
                    }
                    last = point;
                    valid = current;
                }
            }
        }
        const Vec3 direction(pose.at("direction")[0], pose.at("direction")[1], pose.at("direction")[2]);
        ImVec2 tip;
        if (project(p + direction.normalize() * 2, tip)) {
            draw->AddLine(screen, tip, IM_COL32(255, 185, 70, 200), 2);
        }
        const Vec3 axis = direction.normalize();
        const Vec3 reference = std::abs(axis.y) > 0.99f ? Vec3(1, 0, 0) : Vec3(0, 1, 0);
        const Vec3 right = axis.cross(reference).normalize();
        const Vec3 up = right.cross(axis).normalize();
        if (e.at("spatial").get<bool>()) {
            for (const char* key : {"cone_inner", "cone_outer"}) {
                const float degrees = e.at(key);
                if (degrees >= 360) {
                    continue;
                }
                const float halfAngle = degrees * 0.00872664626f;
                const Vec3 center = p + axis * (2 * std::cos(halfAngle));
                const float radius = 2 * std::sin(halfAngle);
                ImVec2 last;
                bool valid = false;
                for (int segment = 0; segment <= 32; ++segment) {
                    const float angle = segment * 0.19634954085f;
                    ImVec2 point;
                    const bool current = project(center + right * (radius * std::cos(angle)) +
                                                  up * (radius * std::sin(angle)), point);
                    if (current && valid) {
                        draw->AddLine(last, point, IM_COL32(255, 185, 70, 120));
                    }
                    if (current && segment % 8 == 0) {
                        draw->AddLine(screen, point, IM_COL32(255, 185, 70, 120));
                    }
                    last = point;
                    valid = current;
                }
            }
        }
        if (ctx.selection.transform_mode == TransformMode::Scale) {
            continue;
        }
        float view[16] = {r.x, u.x, -f.x, 0, r.y, u.y, -f.y, 0, r.z, u.z, -f.z, 0,
                          -r.dot(cam.lookfrom), -u.dot(cam.lookfrom), f.dot(cam.lookfrom), 1};
        float projection[16]{};
        if (ortho) {
            projection[0] = 2 / (cam.ortho_height * aspect);
            projection[5] = 2 / cam.ortho_height;
            projection[10] = -2 / 9999.9f;
            projection[14] = -10000.1f / 9999.9f;
            projection[15] = 1;
        } else {
            projection[0] = 1 / (aspect * tangent);
            projection[5] = 1 / tangent;
            projection[10] = -10000.1f / 9999.9f;
            projection[11] = -1;
            projection[14] = -2000 / 9999.9f;
        }
        float matrix[16] = {right.x, right.y, right.z, 0, up.x, up.y, up.z, 0,
                            -axis.x, -axis.y, -axis.z, 0, p.x, p.y, p.z, 1};
        ImGuizmo::PushID(8271);
        ImGuizmo::SetDrawlist(draw);
        ImGuizmo::SetOrthographic(ortho);
        ImGuizmo::SetRect(0, 0, size.x, size.y);
        const bool rotate = ctx.selection.transform_mode == TransformMode::Rotate;
        const bool changed = ImGuizmo::Manipulate(view, projection,
                                                  rotate ? ImGuizmo::ROTATE : ImGuizmo::TRANSLATE,
                                                  ImGuizmo::WORLD, matrix);
        if (ImGuizmo::IsOver() || ImGuizmo::IsUsing()) {
            hit = true;
        }
        if (changed) {
            if (!dragging) {
                dragBefore = data;
                dragging = true;
            }
            try {
                const auto local = audio::localPosition(ctx, e,
                    json::array({matrix[12], matrix[13], matrix[14]}));
                const auto target = audio::localPosition(ctx, e,
                    json::array({matrix[12] - matrix[8], matrix[13] - matrix[9], matrix[14] - matrix[10]}));
                ctx.scene.audio->invoke("update", {{"id", id}, {"position", local},
                    {"direction", json::array({target[0].get<float>() - local[0].get<float>(),
                        target[1].get<float>() - local[1].get<float>(),
                        target[2].get<float>() - local[2].get<float>()})}});
            } catch (const std::exception& e) {
                error = e.what();
            }
        }
        if (dragging && !ImGuizmo::IsUsing()) {
            audio::recordEdit(ctx, ctx.scene_ui_ptr ? &ctx.scene_ui_ptr->history : nullptr,
                              dragBefore, "Transform Audio Source");
            dragging = false;
        }
        ImGuizmo::PopID();
    }
}
}
