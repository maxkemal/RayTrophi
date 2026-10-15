#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "UI/AudioTrackPanels.h"
#include "UI/AudioWorkspaceUI.h"
#include "UI/AudioPropertyLayout.h"
#include "Audio/AudioService.h"
#include "scene_data.h"
#include "imgui.h"
#include <algorithm>
#include <iterator>
#include <cstdio>
#include "scene_ui.h"
namespace AudioUI {
namespace {
using json = nlohmann::json;
json mutate(UIContext& ctx, const char* operation, const json& params) {
    finishAudioTimelineGesture(ctx);
    return workspaceCall(ctx, operation, params);
}
} // namespace

void drawAudioTrackTools(UIContext& ctx) {
    const auto tracks = ctx.scene.audio->invoke("track_list", json::object());
    const auto view = ctx.scene.audio->invoke("workspace_get", json::object());
    if (ImGui::SmallButton("Add track")) {
        const auto added = mutate(ctx, "track_create", json::object());
        if (added.contains("id")) {
            workspaceCall(ctx, "workspace_update", {{"selected_track", added.at("id")},
                {"selected_clip", ""}});
        }
    }
    for (std::size_t index = 0; index < tracks.size(); ++index) {
        const auto& track = tracks[index];
        const auto id = track.at("id").get<std::string>();
        ImGui::PushID(id.c_str());
        bool muted = track.at("muted"), solo = track.at("solo");
        if (ImGui::Checkbox("M", &muted)) {
            mutate(ctx, "track_update", {{"id", id}, {"muted", muted}});
        }
        ImGui::SameLine();
        if (ImGui::Checkbox("S", &solo)) {
            mutate(ctx, "track_update", {{"id", id}, {"solo", solo}});
        }
        if (ImGui::Selectable(track.at("name").get<std::string>().c_str(),
            view.value("selected_track", std::string()) == id)) {
            workspaceCall(ctx, "workspace_update", {{"selected_track", id}, {"selected_clip", ""}});
        }
        if (ImGui::BeginPopupContextItem("TrackActions")) {
            int direction = 0;
            if (ImGui::MenuItem("Move up", nullptr, false, index > 0)) {
                direction = -1;
            }
            if (ImGui::MenuItem("Move down", nullptr, false, index + 1 < tracks.size())) {
                direction = 1;
            }
            if (direction != 0) {
                json ids = json::array();
                for (const auto& entry : tracks) {
                    ids.push_back(entry.at("id"));
                }
                const auto target = static_cast<std::size_t>(static_cast<int>(index) + direction);
                std::swap(ids[index], ids[target]);
                mutate(ctx, "track_reorder", {{"ids", ids}});
            }
            if (ImGui::MenuItem("Remove track; keep clips on bus")) {
                mutate(ctx, "track_remove", {{"id", id}, {"clip_policy", "detach"}});
            }
            if (ImGui::MenuItem("Remove track and its clips")) {
                mutate(ctx, "track_remove", {{"id", id}, {"clip_policy", "delete"}});
            }
            ImGui::EndPopup();
        }
        ImGui::PopID();
    }
}

void drawAudioClipSelectionTools(UIContext& ctx) {
    const auto view = ctx.scene.audio->invoke("workspace_get", json::object());
    const auto scene = ctx.scene.audio->snapshot();
    const auto selected = view.at("selected_clips");
    if (ImGui::CollapsingHeader("Clips / multi-selection")) {
        for (const auto& clip : scene.at("clips")) {
            const auto id = clip.at("id").get<std::string>();
            const auto label = clip.at("name").get<std::string>() + "##" + id;
            const auto item = std::find(selected.begin(), selected.end(), json(id));
            if (ImGui::Selectable(label.c_str(), item != selected.end())) {
                auto ids = selected;
                if (item == selected.end()) {
                    ids.push_back(id);
                } else {
                    ids.erase(ids.begin() + std::distance(selected.begin(), item));
                }
                selectAudioClips(ctx, ids);
            }
        }
    }
    ImGui::Text("Selected: %zu", selected.size());
    ImGui::BeginDisabled(selected.empty());
    if (ImGui::BeginCombo("Move selection to track", "Choose track...")) {
        for (const auto& track : scene.at("tracks")) {
            if (ImGui::Selectable(track.at("name").get<std::string>().c_str())) {
                mutate(ctx, "clip_batch", {{"ids", selected}, {"action", "update"},
                    {"patch", {{"track_id", track.at("id")}}}});
            }
        }
        ImGui::EndCombo();
    }
    ImGui::EndDisabled();
}

bool drawSelectedAudioTrackProperties(UIContext& ctx) {
    const auto view = ctx.scene.audio->invoke("workspace_get", json::object());
    if (!view.at("selected_clip").get<std::string>().empty()) {
        return false;
    }
    const auto id = view.value("selected_track", std::string());
    const auto tracks = ctx.scene.audio->invoke("track_list", json::object());
    for (const auto& track : tracks) {
        if (track.at("id").get<std::string>() != id) {
            continue;
        }
        char name[257]{};
        std::snprintf(name, sizeof(name), "%s", track.at("name").get<std::string>().c_str());
        if (ImGui::InputText(property("Track name").c_str(), name, sizeof(name),
            ImGuiInputTextFlags_EnterReturnsTrue)) {
            mutate(ctx, "track_update", {{"id", id}, {"name", name}});
        }
        if (ImGui::BeginCombo(property("Output bus").c_str(), track.at("bus").get<std::string>().c_str())) {
            for (const char* bus : {"Master", "Weather", "Effects", "Ambience", "Music"}) {
                if (ImGui::Selectable(bus, track.at("bus").get<std::string>() == bus)) {
                    mutate(ctx, "track_update", {{"id", id}, {"bus", bus}});
                }
            }
            ImGui::EndCombo();
        }
        float gain = track.at("gain"), pan = track.at("pan");
        if (ImGui::InputFloat(property("Track gain (linear 0..16)").c_str(), &gain,
            0, 0, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue)) {
            mutate(ctx, "track_update", {{"id", id}, {"gain", gain}});
        }
        if (ImGui::InputFloat(property("Track pan (-1..1)").c_str(), &pan,
            0, 0, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue)) {
            mutate(ctx, "track_update", {{"id", id}, {"pan", pan}});
        }
        ImGui::TextWrapped("Track controls multiply clip gain and add to automated pan. "
                           "Bus and Master mixing follow afterward.");
        return true;
    }
    return false;
}
} // namespace AudioUI
