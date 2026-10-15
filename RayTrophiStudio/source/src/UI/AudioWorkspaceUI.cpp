#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "UI/AudioWorkspaceUI.h"
#include "UI/AudioSoundDesignerUI.h"
#include "UI/AudioPropertyLayout.h"
#include "UI/AudioUI.h"
#include "UI/AudioThunderUI.h"
#include "Audio/AudioAuthoring.h"
#include "Audio/AudioService.h"
#include "Api/RtApi.h"
#include "scene_ui.h"
#include "scene_data.h"
#include "SceneSelection.h"
#include "imgui.h"
#include "ui_modern.h"
#include "../../../external/imgui-knobs/imgui-knobs.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <utility>

namespace AudioUI {
namespace {
using json = nlohmann::json;
std::string error;
ImGuiTextFilter sourceFilter;

void openBottom(const char* name) {
    const auto result = rtapi::setBottomEditor(name);
    if (!result.ok) {
        error = result.error;
    }
}

void editNumber(UIContext& ctx, const json& clip, const char* key, const char* label) {
    double value = clip.at(key);
    if (ImGui::InputDouble(property(label).c_str(), &value, 0, 0, "%.4f",
                          ImGuiInputTextFlags_EnterReturnsTrue)) {
        if (std::string(key) == "duration_seconds") {
            workspaceCall(ctx, "clip_resize", {{"id", clip.at("id")},
                {"end_seconds", clip.at("start_seconds").get<double>() + value},
                {"auto_loop", true}});
        } else {
            workspaceCall(ctx, "clip_update", {{"id", clip.at("id")}, {key, value}});
        }
    }
}

void editCurve(UIContext& ctx, const json& clip, const char* key, const char* label,
               double initialValue) {
    if (!ImGui::TreeNode(key, "%s", label)) {
        return;
    }
    const auto& curve = clip.at(key);
    ImGui::TextWrapped("Time is relative to the original clip envelope.");
    for (std::size_t index = 0; index < curve.size(); ++index) {
        ImGui::PushID(static_cast<int>(index));
        double time = curve[index][0], value = curve[index][1];
        bool changed = ImGui::InputDouble(property("Time").c_str(), &time, 0, 0, "%.3f",
                                         ImGuiInputTextFlags_EnterReturnsTrue);
        changed = ImGui::InputDouble(property("Value").c_str(), &value, 0, 0, "%.3f",
                                    ImGuiInputTextFlags_EnterReturnsTrue) || changed;
        if (changed) {
            auto updated = curve;
            updated[index] = {time, value};
            workspaceCall(ctx, "clip_update", {{"id", clip.at("id")}, {key, updated}});
        }
        if (ImGui::SmallButton("Remove knot")) {
            auto updated = curve;
            updated.erase(updated.begin() + index);
            workspaceCall(ctx, "clip_update", {{"id", clip.at("id")}, {key, updated}});
        }
        ImGui::PopID();
    }
    if (ImGui::SmallButton("Add knot at playhead")) {
        const double fps = std::max(1, ctx.render_settings.animation_fps);
        const double time = ctx.scene.timeline.current_frame / fps -
            clip.at("start_seconds").get<double>() + clip.at("envelope_offset_seconds").get<double>();
        auto updated = curve;
        if (time >= 0 && time <= clip.at("envelope_duration_seconds").get<double>()) {
            bool replaced = false;
            for (auto& knot : updated) {
                if (std::abs(knot[0].get<double>() - time) < 1e-8) {
                    knot[1] = initialValue;
                    replaced = true;
                }
            }
            if (!replaced) {
                updated.push_back({time, initialValue});
                std::sort(updated.begin(), updated.end(), [](const json& a, const json& b) {
                    return a[0].get<double>() < b[0].get<double>();
                });
            }
            workspaceCall(ctx, "clip_update", {{"id", clip.at("id")}, {key, updated}});
        } else {
            error = "Playhead must be within this clip's envelope.";
        }
    }
    ImGui::TreePop();
}

void drawClipInspector(UIContext& ctx, const json& clip) {
    const auto id = clip.at("id");
    ImGui::TextUnformatted("Audio clip");
    char name[257]{};
    std::snprintf(name, sizeof(name), "%s", clip.at("name").get<std::string>().c_str());
    if (ImGui::InputText(property("Name").c_str(), name, sizeof(name), ImGuiInputTextFlags_EnterReturnsTrue)) {
        workspaceCall(ctx, "clip_update", {{"id", id}, {"name", name}});
    }
    const auto data = ctx.scene.audio->snapshot();
    if (ImGui::BeginCombo(property("Track").c_str(),
        clip.value("track_id", std::string()).c_str())) {
        if (ImGui::Selectable("Legacy bus / no track")) {
            workspaceCall(ctx, "clip_update", {{"id", id}, {"track_id", ""}});
        }
        for (const auto& track : data.at("tracks")) {
            if (ImGui::Selectable(track.at("name").get<std::string>().c_str())) {
                workspaceCall(ctx, "clip_update", {{"id", id}, {"track_id", track.at("id")}});
            }
        }
        ImGui::EndCombo();
    }
    if (ImGui::BeginCombo(property("Asset").c_str(), clip.at("asset_id").get<std::string>().c_str())) {
        for (const auto& asset : data.at("assets")) {
            if (ImGui::Selectable(asset.at("name").get<std::string>().c_str(),
                                  clip.at("asset_id") == asset.at("id"))) {
                workspaceCall(ctx, "clip_update", {{"id", id}, {"asset_id", asset.at("id")}});
            }
        }
        ImGui::EndCombo();
    }
    editNumber(ctx, clip, "start_seconds", "Start (seconds)");
    editNumber(ctx, clip, "duration_seconds", "Duration");
    if (ImGui::TreeNode("Source range")) {
        editNumber(ctx, clip, "source_in_seconds", "Source in");
        editNumber(ctx, clip, "source_out_seconds", "Source out");
        editNumber(ctx, clip, "phase_seconds", "Source phase");
        ImGui::TreePop();
    }
    editNumber(ctx, clip, "gain", "Linear gain");
    editNumber(ctx, clip, "pitch", "Pitch ratio");
    // Keep a drag draft until release; one canonical mutation and one undo entry.
    static std::string panId;
    static float panDraft = 0;
    static bool panEditing = false;
    const auto currentId = id.get<std::string>();
    if (!panEditing || panId != currentId) {
        panDraft = clip.at("pan");
        panId = currentId;
        panEditing = false;
    }
    ImGuiKnobs::Knob("Pan", &panDraft, -1, 1, 0.01f, "%.2f",
        ImGuiKnobVariant_Wiper, 52, ImGuiKnobFlags_NoInput | ImGuiKnobFlags_AlwaysClamp);
    if (ImGui::IsItemActivated()) {
        panEditing = true;
    }
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        workspaceCall(ctx, "clip_update", {{"id", id}, {"pan", panDraft}});
        panEditing = false;
    }
    editNumber(ctx, clip, "pan", "Pan value");
    editNumber(ctx, clip, "fade_in_seconds", "Fade in");
    editNumber(ctx, clip, "fade_out_seconds", "Fade out");
    for (const char* key : {"loop", "muted"}) {
        bool value = clip.at(key);
        if (ImGui::Checkbox(key, &value)) {
            workspaceCall(ctx, "clip_update", {{"id", id}, {key, value}});
        }
    }
    ImGui::BeginDisabled(!clip.value("track_id", std::string()).empty());
    if (ImGui::BeginCombo(property("Bus").c_str(), clip.at("bus").get<std::string>().c_str())) {
        for (const char* bus : {"Master", "Weather", "Effects", "Ambience", "Music"}) {
            if (ImGui::Selectable(bus, clip.at("bus") == bus)) {
                workspaceCall(ctx, "clip_update", {{"id", id}, {"bus", bus}});
            }
        }
        ImGui::EndCombo();
    }
    ImGui::EndDisabled();
    editCurve(ctx, clip, "gain_curve", "Gain automation", 1);
    editCurve(ctx, clip, "pitch_curve", "Pitch automation", 1);
    editCurve(ctx, clip, "pan_curve", "Pan automation", 0);
    if (ImGui::Button("Duplicate")) {
        const auto added = workspaceCall(ctx, "clip_duplicate", {{"id", id}});
        if (added.contains("id")) {
            selectClip(ctx, added.at("id"));
        }
    }
    if (ImGui::GetContentRegionAvail().x + ImGui::GetCursorScreenPos().x -
        ImGui::GetItemRectMax().x > ImGui::CalcTextSize("Delete clip").x + 24) {
        ImGui::SameLine();
    }
    if (ImGui::Button("Delete clip")) {
        workspaceCall(ctx, "clip_remove", {{"id", id}});
        workspaceCall(ctx, "workspace_update", {{"selected_clip", ""}});
    }
}
}

json workspaceCall(UIContext& ctx, const std::string& operation, const json& params) {
    try {
        auto result = audio::author(ctx, ctx.scene_ui_ptr ? &ctx.scene_ui_ptr->history : nullptr,
                                    operation, params);
        error.clear();
        return result;
    } catch (const std::exception& exception) {
        error = exception.what();
        return json::object();
    }
}

void drawSelectedAudioClipProperties(UIContext& ctx) {
    const auto view = ctx.scene.audio->invoke("workspace_get", json::object());
    const auto data = ctx.scene.audio->snapshot();
    for (const auto& clip : data.at("clips")) {
        if (clip.at("id") == view.at("selected_clip")) {
            drawClipInspector(ctx, clip);
            return;
        }
    }
    ImGui::TextWrapped("Select a clip on the arrangement canvas.");
}

void selectAudioClips(UIContext& ctx, const json& ids) {
    const auto id = ids.empty() ? std::string() : ids.back().get<std::string>();
    std::string trackId;
    const auto data = ctx.scene.audio->snapshot();
    for (const auto& clip : data.at("clips")) {
        if (clip.at("id").get<std::string>() == id) {
            trackId = clip.value("track_id", std::string());
            break;
        }
    }
    json patch = {{"selected_clip", id}, {"selected_clips", ids}, {"selected_track", trackId}};
    if (!ids.empty()) {
        double first = 86400, last = 0;
        for (const auto& clip : data.at("clips")) {
            if (std::find(ids.begin(), ids.end(), clip.at("id")) != ids.end()) {
                first = std::min(first, clip.at("start_seconds").get<double>());
                last = std::max(last, clip.at("start_seconds").get<double>() +
                    clip.at("duration_seconds").get<double>());
            }
        }
        first = std::min(first, 86400 - 0.05);
        patch["audition_start_seconds"] = first;
        patch["audition_end_seconds"] = std::min(86400.0,
            std::max(first + 0.05, std::min(last, first + 30)));
    }
    const auto result = workspaceCall(ctx, "workspace_update", patch);
    if (result.contains("selected_clip") && ctx.scene_ui_ptr) {
        ctx.selection.clearSelection();
        ctx.scene_ui_ptr->tab_to_focus = "Audio";
    }
}

void selectClip(UIContext& ctx, const std::string& id) {
    selectAudioClips(ctx, id.empty() ? json::array() : json::array({id}));
}

void addTimelineAsset(UIContext& ctx, const std::string& assetId) {
    const double fps = std::max(1, ctx.render_settings.animation_fps);
    const auto view = ctx.scene.audio->invoke("workspace_get", json::object());
    const auto track = view.value("designer_open", false) &&
        view.value("designer_edit_view", std::string("source")) == "arrange" ?
        view.value("selected_track", std::string()) : std::string();
    const auto added = workspaceCall(ctx, "clip_create", {
        {"track_id", track}, {"asset_id", assetId}, {"start_seconds", ctx.scene.timeline.current_frame / fps}});
    if (added.contains("id")) {
        selectClip(ctx, added.at("id"));
        openBottom("audio");
    }
}

void drawSynthesisControls(UIContext& ctx) {
    if (ImGui::Button("Design sound...")) {
        openSoundDesigner(ctx);
    }
}

void drawAudioEditorNavigation(UIContext& ctx) {
    const auto state = rtapi::editorState();
    for (const auto& entry : {std::pair{"Timeline", "audio"},
                              std::pair{"Mixer", "audio_mixer"}}) {
        if (state.bottom_editor != "audio" && !ImGui::GetIO().MouseDown[0]) {
            finishAudioTimelineGesture(ctx);
        }
        const bool active = state.bottom_editor == entry.second &&
            !ctx.scene.audio->invoke("workspace_get", json::object()).value("designer_open", false);
        if (ImGui::Selectable(entry.first, active, 0, ImVec2(85, 0))) {
            finishAudioTimelineGesture(ctx);
            workspaceCall(ctx, "workspace_update", {{"designer_open", false}});
            workspaceCall(ctx, "design_stop");
            openBottom(entry.second);
        }
        ImGui::SameLine();
    }
    const auto view = ctx.scene.audio->invoke("workspace_get", json::object());
    if (ImGui::Selectable("Designer", view.value("designer_open", false), 0, ImVec2(85, 0))) {
        openSoundDesigner(ctx);
    }
    if (view.value("designer_open", false)) {
        ImGui::SameLine();
    } else {
        ImGui::Separator();
    }
}

void drawAudioBottomTabs(UIContext& ctx) {
    const auto state = rtapi::editorState();
    const bool active = state.bottom_editor == "audio" || state.bottom_editor == "audio_mixer";
    if (UIWidgets::HorizontalTab("Audio", UIWidgets::IconType::Audio, active)) {
        finishAudioTimelineGesture(ctx);
        openBottom(active ? "none" : "audio");
        if (ctx.scene_ui_ptr && !active) {
            ctx.scene_ui_ptr->tab_to_focus = "Audio";
        }
    }
    ImGui::SameLine();
    const auto status = ctx.scene.audio->invoke("status", json::object());
    if (status.at("capture").at("state") == "recording") {
        ImGui::TextColored(ImVec4(1, 0.3f, 0.2f, 1), "REC");
        ImGui::SameLine();
    }
    if (status.at("export").at("state") == "running") {
        ImGui::TextDisabled("Audio export");
        ImGui::SameLine();
    }
}

void drawOutputControls(UIContext& ctx) {
    const auto status = ctx.scene.audio->invoke("status", json::object());
    const auto& capture = status.at("capture");
    ImGui::Text("Capture: %s", capture.at("state").get<std::string>().c_str());
    const bool running = capture.at("state") == "recording" || capture.at("state") == "ready";
    if (!running) {
        if (ImGui::Button("Start scene capture")) {
            workspaceCall(ctx, "capture_start");
        }
    } else {
        if (ImGui::Button("Finish capture")) {
            const auto result = workspaceCall(ctx, "capture_finish");
            if (result.contains("clips") && !result.at("clips").empty()) {
                selectClip(ctx, result.at("clips")[0].at("id"));
                openBottom("audio");
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel capture")) {
            workspaceCall(ctx, "capture_cancel");
        }
    }
    if (!capture.at("error").get<std::string>().empty()) {
        ImGui::TextWrapped("%s", capture.at("error").get<std::string>().c_str());
    }
    ImGui::TextWrapped("Capture follows timeline Play and records played scene sources into clips.");
    if (ImGui::Button("Export timeline audio...")) {
        ImGui::OpenPopup("Export audio");
    }
    if (ImGui::BeginPopupModal("Export audio", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        static char output[1024] = "";
        static bool overwrite = false;
        ImGui::Text("Range: frames %d..%d inclusive, %d fps",
            ctx.render_settings.animation_start_frame, ctx.render_settings.animation_end_frame,
            ctx.render_settings.animation_fps);
        ImGui::SetNextItemWidth(440);
        ImGui::InputText("WAV path", output, sizeof(output));
        if (ImGui::Button("Choose folder")) {
            const auto folder = SceneUI::selectFolderDialogW(L"Select audio export folder");
            if (!folder.empty()) {
                std::snprintf(output, sizeof(output), "%s/scene_audio.wav", folder.c_str());
            }
        }
        ImGui::Checkbox("Replace existing output", &overwrite);
        ImGui::TextWrapped("Exports authored and captured clips. Capture reactive sources first.");
        if (ImGui::Button("Export")) {
            const auto result = workspaceCall(ctx, "export_start", {{"path", output},
                {"start_frame", ctx.render_settings.animation_start_frame},
                {"end_frame_exclusive", ctx.render_settings.animation_end_frame + 1},
                {"fps_num", std::max(1, ctx.render_settings.animation_fps)},
                {"fps_den", 1}, {"overwrite", overwrite}});
            if (result.contains("state")) {
                ImGui::CloseCurrentPopup();
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Close")) {
            ImGui::CloseCurrentPopup();
        }
        if (!error.empty()) {
            ImGui::TextWrapped("%s", error.c_str());
        }
        ImGui::EndPopup();
    }
    const auto& job = status.at("export");
    ImGui::Text("Export: %s", job.at("state").get<std::string>().c_str());
    if (job.at("state") == "running") {
        if (job.contains("frames") && job.at("frames").get<double>() > 0) {
            const float progress = static_cast<float>(job.at("written_frames").get<double>() /
                                                       job.at("frames").get<double>());
            ImGui::ProgressBar(progress, ImVec2(-1, 0));
        }
        if (ImGui::Button("Cancel audio export")) {
            workspaceCall(ctx, "export_cancel");
        }
    }
    if (job.contains("error")) {
        ImGui::TextWrapped("%s", job.at("error").get<std::string>().c_str());
    }
}

void drawWorkspace(UIContext& ctx) {
    ImGui::TextUnformatted("Scene Audio");
    if (ImGui::Button("Add source")) {
        addSource(ctx);
    }
    ImGui::SameLine();
    if (ImGui::Button("Import sound")) {
        const auto path = SceneUI::openFileDialogW(
            L"Audio Files\0*.wav;*.flac;*.mp3\0");
        if (!path.empty()) {
            workspaceCall(ctx, "asset_import", {{"path", path}});
        }
    }
    if (ImGui::Button("Audio timeline")) {
        openBottom("audio");
    }
    ImGui::SameLine();
    if (ImGui::Button("Mixer")) {
        openBottom("audio_mixer");
    }
    const auto workspace = ctx.scene.audio->invoke("workspace_get", json::object());
    bool inspectorOpen = workspace.at("inspector_open");
    if (ImGui::Checkbox("Audio inspector", &inspectorOpen)) {
        workspaceCall(ctx, "workspace_update", {{"inspector_open", inspectorOpen}});
    }
    const auto status = ctx.scene.audio->invoke("status", json::object());
    bool previewMuted = status.at("preview_muted");
    if (ImGui::Checkbox("Mute preview output", &previewMuted)) {
        workspaceCall(ctx, "preview_mute", {{"muted", previewMuted}});
    }
    if (ImGui::Button("Stop all")) {
        workspaceCall(ctx, "stop_all");
    }
    ImGui::SameLine();
    if (ImGui::Button("Resume automatic")) {
        workspaceCall(ctx, "resume_automatic");
    }
    ImGui::Separator();
    const auto data = ctx.scene.audio->snapshot();
    if (ImGui::BeginTabBar("AudioWorkspaceSections")) {
        if (ImGui::BeginTabItem("Sources")) {
            sourceFilter.Draw("Search sources", -1);
            for (const auto& source : data.at("emitters")) {
                const auto name = source.at("name").get<std::string>();
                const auto id = source.at("id").get<std::string>();
                if (!sourceFilter.PassFilter(name.c_str())) {
                    continue;
                }
                ImGui::PushID(id.c_str());
                const bool selected = ctx.selection.selected.type == SelectableType::AudioSource &&
                    ctx.selection.selected.audio_source_id == id;
                if (ImGui::Selectable(name.c_str(), selected)) {
                    workspaceCall(ctx, "select", {{"id", id}});
                    workspaceCall(ctx, "workspace_update", {{"selected_clip", ""}});
                }
                ImGui::TextDisabled("%s | %s", source.at("bus").get<std::string>().c_str(),
                    source.at("asset_id").get<std::string>().empty() ? "Unbound" :
                    source.at("asset_id").get<std::string>().c_str());
                for (const auto& voice : status.at("voices")) {
                    if (voice.at("id") == id) {
                        ImGui::TextDisabled("%s", voice.at("playing").get<bool>() ? "Playing" : "Paused");
                    }
                }
                ImGui::PopID();
            }
            if (data.at("emitters").empty()) {
                ImGui::TextWrapped("Add a positional source or connect the atmosphere below.");
            }
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Assets")) {
            drawSynthesisControls(ctx);
            for (const auto& asset : data.at("assets")) {
                const auto id = asset.at("id").get<std::string>();
                ImGui::PushID(id.c_str());
                ImGui::TextWrapped("%s", asset.at("name").get<std::string>().c_str());
                ImGui::TextDisabled("%.2fs | %d channels", asset.at("duration").get<double>(),
                                    asset.at("channels").get<int>());
                if (ImGui::SmallButton("Preview")) {
                    workspaceCall(ctx, "preview", {{"id", id}});
                }
                ImGui::SameLine();
                if (ImGui::SmallButton("Design / process")) {
                    openSoundDesigner(ctx, id);
                }
                if (asset.value("source", std::string()).rfind("RayTrophi sound design v", 0) == 0 &&
                    ImGui::SmallButton("Reuse recipe")) {
                    reuseSoundRecipe(ctx, id);
                }
                if (ImGui::SmallButton("Add clip")) {
                    addTimelineAsset(ctx, id);
                }
                if (ImGui::SmallButton("Add scene source")) {
                    const auto added = workspaceCall(ctx, "append", {
                        {"path", asset.at("path")}, {"name", asset.at("name")},
                        {"license", asset.value("license", "")}, {"author", asset.value("author", "")},
                        {"source", asset.value("source", "")}, {"tags", asset.value("tags", json::array())}});
                    if (added.contains("id")) {
                        workspaceCall(ctx, "select", {{"id", added.at("id")}});
                    }
                }
                if (ImGui::SmallButton("Relink")) {
                    const auto path = SceneUI::openFileDialogW(L"Audio Files\0*.wav;*.flac;*.mp3\0");
                    if (!path.empty()) {
                        workspaceCall(ctx, "asset_relink", {{"id", id}, {"path", path}});
                    }
                }
                ImGui::SameLine();
                if (ImGui::SmallButton("Remove asset")) {
                    workspaceCall(ctx, "asset_remove", {{"id", id}});
                }
                ImGui::Separator();
                ImGui::PopID();
            }
            if (ImGui::Button("Stop audition")) {
                workspaceCall(ctx, "preview_stop");
            }
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Scene")) {
            const auto telemetry = ctx.scene.audio->invoke("telemetry_get", json::object());
            bool collect = telemetry.at("enabled");
            if (ImGui::Checkbox("Collect physics audio telemetry", &collect)) {
                workspaceCall(ctx, "telemetry_update", {{"enabled", collect}});
            }
            ImGui::TextWrapped(collect ?
                "Domain physics measurements are enabled. DEM readback scales with grain count." :
                "Off: skips additional physics audio collection. Enable manually when using domain sounds.");
            ImGui::Separator();
            const auto mode = data.at("playback_mode").get<std::string>();
            if (ImGui::BeginCombo("Playback", mode.c_str())) {
                for (const char* entry : {"timeline", "continuous"}) {
                    if (ImGui::Selectable(entry, mode == entry)) {
                        workspaceCall(ctx, "playback_update", {{"mode", entry}});
                    }
                }
                ImGui::EndCombo();
            }
            if (ImGui::Button("Connect atmosphere wind / rain")) {
                workspaceCall(ctx, "environment_setup");
            }
            drawThunderAudio(ctx);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Output")) {
            if (ctx.scene.audio->invoke("workspace_get", json::object()).value("designer_open", false)) {
                if (ImGui::CollapsingHeader("Designed sound / Save", ImGuiTreeNodeFlags_DefaultOpen)) {
                    drawSoundDesignInspector(ctx);
                }
            }
            drawOutputControls(ctx);
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    if (!error.empty()) {
        ImGui::TextColored(ImVec4(1, 0.4f, 0.3f, 1), "Audio operation failed");
        ImGui::TextWrapped("%s", error.c_str());
    }
    if (ImGui::TreeNode("Diagnostics")) {
        ImGui::Text("Device: %s", status.at("device").get<std::string>().c_str());
        ImGui::TextWrapped("%s", status.at("clip_runtime").dump(2).c_str());
        ImGui::TreePop();
    }
}

void drawInspectorDock(UIContext& ctx) {
    if (!ctx.scene_ui_ptr) {
        return;
    }
    const auto view = ctx.scene.audio->invoke("workspace_get", json::object());
    bool open = view.at("inspector_open");
    if (!open) {
        return;
    }
    ImGui::SetNextWindowSizeConstraints(ImVec2(180, 180), ImVec2(FLT_MAX, FLT_MAX));
    ImGui::SetNextWindowSize(ImVec2(360, 520), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Audio Inspector", &open)) {
        ImGui::SetScrollX(0);
        ImGui::PushItemWidth(-FLT_MIN);
        const auto data = ctx.scene.audio->snapshot();
        bool found = false;
        for (const auto& clip : data.value("clips", json::array())) {
            if (!found && clip.at("id") == view.at("selected_clip")) {
                drawClipInspector(ctx, clip);
                found = true;
                break;
            }
        }
        if (!found && ctx.selection.selected.type == SelectableType::AudioSource) {
            drawSourceInspector(ctx);
        } else if (!found) {
            ImGui::TextWrapped("Select a scene source or a timeline clip to edit its properties.");
        }
        ImGui::PopItemWidth();
        if (!error.empty()) {
            ImGui::TextWrapped("%s", error.c_str());
        }
    }
    ImGui::End();
    if (!open) {
        workspaceCall(ctx, "workspace_update", {{"inspector_open", false}});
    }
}
}
