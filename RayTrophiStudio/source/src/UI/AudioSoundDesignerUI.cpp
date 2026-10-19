#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "UI/AudioSoundDesignerUI.h"
#include "UI/AudioHistoryUI.h"
#include "UI/AudioLiveAnalysisUI.h"
#include "UI/AudioDesignPanels.h"
#include "UI/AudioDesignView.h"
#include "UI/AudioTrackPanels.h"
#include "UI/AudioClipEnvelopeEditor.h"
#include "UI/AudioWorkspaceUI.h"
#include "UI/AudioPropertyLayout.h"
#include "UI/AudioEditorChrome.h"
#include "Audio/AudioService.h"
#include "Audio/AudioAuthoring.h"
#include "Audio/AudioDesignLimits.h"
#include "Api/RtApi.h"
#include "scene_data.h"
#include "scene_ui.h"
#include "imgui.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <utility>
#include <map>

namespace AudioUI {
namespace {
using json = nlohmann::json;
struct DesignerUI {
    const audio::AudioService* owner = nullptr;
    std::uint64_t generation = 0;
    json recipe = {{"mode", "synth"}, {"preset", "metal"}};
    bool dirty = true;
    std::uint64_t renderJob = 0;
    json requestedRecipe;
    bool previewWhenReady = false;
    bool saveWhenReady = false;
    bool loop = false;
    bool autoPreview = true;
    bool insertClip = true;
    bool original = false;
    int envelopeHandle = -1;
    int fadeHandle = -1;
    bool fitCanvasRequested = true;
    json rendered = json::object();
    std::uint64_t renderedRevision = 0;
    char name[128] = "Metal impact";
    std::string folder;
    std::string savedPath;
    std::string error;
    std::string targetTrack;
    double targetStart = 0;
    bool targetInitialized = false;
    std::string savedAsset;
    std::string savedClip;
    bool clipPending = false;
    std::string clipError;
    bool keepEffects = false;
};
DesignerUI state;

struct DesignerDraft {
    json recipe;
    std::string name;
};
std::map<std::string, DesignerDraft> drafts;

std::string draftKey() {
    return state.recipe.value("mode", "synth") == "asset" ?
        state.recipe.value("asset_id", std::string()) : "@synthetic";
}

void retainDraft() {
    drafts[draftKey()] = {state.recipe, state.name};
}

json call(UIContext& ctx, const char* operation, const json& params = json::object()) {
    try {
        auto result = audio::author(ctx, ctx.scene_ui_ptr ? &ctx.scene_ui_ptr->history : nullptr,
                                    operation, params);
        if (std::string(operation) != "design_destination") {
            state.error.clear();
        }
        return result;
    } catch (const std::exception& error) {
        state.error = error.what();
        return json::object();
    }
}

void sync(UIContext& ctx) {
    if (state.owner != ctx.scene.audio.get() || state.generation != ctx.scene.audio->generation()) {
        state = {};
        drafts.clear();
        state.owner = ctx.scene.audio.get();
        state.generation = ctx.scene.audio->generation();
        const auto report = ctx.scene.audio->invoke("design_status", json::object());
        state.rendered = report;
        state.renderedRevision = report.at("revision");
        if (report.at("ready")) {
            state.recipe = report.at("recipe");
            state.dirty = false;
        }
    }
}

json currentReport(UIContext& ctx) {
    const auto transport = ctx.scene.audio->invoke("design_status", {{"details", false}});
    if (state.rendered.empty() || state.renderedRevision != transport.at("revision").get<std::uint64_t>()) {
        state.rendered = ctx.scene.audio->invoke("design_status", json::object());
        state.renderedRevision = transport.at("revision");
    }
    auto result = state.rendered;
    for (auto field = transport.begin(); field != transport.end(); ++field) {
        result[field.key()] = field.value();
    }
    return result;
}

void insertSavedClip(UIContext& ctx) {
    const auto result = call(ctx, "clip_create", {
        {"asset_id", state.savedAsset}, {"track_id", state.targetTrack},
        {"start_seconds", state.targetStart}});
    if (result.contains("id")) {
        state.savedClip = result.at("id").get<std::string>();
        state.clipPending = false;
        state.clipError.clear();
        selectClip(ctx, state.savedClip);
    } else {
        state.clipPending = true;
        state.clipError = state.error;
    }
}

void saveAsset(UIContext& ctx) {
    json params = {{"name", state.name}};
    if (!state.folder.empty()) {
        params["folder"] = state.folder;
    }
    const auto result = call(ctx, "design_save", params);
    if (result.contains("design")) {
        state.savedPath = result.at("design").at("path");
    }
    if (result.contains("asset")) {
        state.savedAsset = result.at("asset").at("id").get<std::string>();
        state.savedClip.clear();
        state.clipPending = false;
        state.clipError.clear();
        if (state.insertClip) {
            insertSavedClip(ctx);
        }
    }
}

void requestRender(UIContext& ctx, bool preview) {
    state.saveWhenReady = false;
    const auto job = call(ctx, "design_job_start", {{"kind", "render"}, {"recipe", state.recipe}});
    if (job.contains("id")) {
        state.renderJob = job.at("id");
        state.requestedRecipe = state.recipe;
        state.previewWhenReady = preview;
    }
}

void followRecipeJob(const json& job) {
    if (!job.contains("id") || !job.contains("recipe")) {
        return;
    }
    state.recipe = job.at("recipe");
    state.requestedRecipe = state.recipe;
    state.renderJob = job.at("id");
    state.dirty = true;
    state.previewWhenReady = false;
    state.saveWhenReady = false;
}

void pollRender(UIContext& ctx) {
    if (state.renderJob == 0) {
        return;
    }
    const auto job = ctx.scene.audio->invoke("design_job_status", json::object());
    if (job.at("id").get<std::uint64_t>() != state.renderJob) {
        state.renderJob = 0;
        return;
    }
    if (state.recipe != state.requestedRecipe) {
        call(ctx, "design_job_cancel");
        state.renderJob = 0;
        return;
    }
    const auto phase = job.at("state").get<std::string>();
    if (phase == "completed") {
        const auto result = call(ctx, "design_job_apply");
        state.renderJob = 0;
        if (result.contains("design")) {
            state.rendered = result.at("design");
            state.renderedRevision = state.rendered.at("revision");
            state.recipe = state.rendered.at("recipe");
            state.dirty = false;
            if (state.saveWhenReady && !rtapi::renderOutputPending()) {
                saveAsset(ctx);
            }
            state.saveWhenReady = false;
            if (state.previewWhenReady && !rtapi::renderOutputPending()) {
                call(ctx, "design_preview", {{"original", state.original}, {"loop", state.loop}});
            }
        } else {
            state.dirty = true;
            state.saveWhenReady = false;
            state.error = "Render context changed; refresh the current recipe.";
        }
    } else if (phase == "failed" || phase == "cancelled" || phase == "consumed") {
        state.renderJob = 0;
        if (phase == "failed") {
            state.error = job.value("error", std::string("Audio render failed"));
        }
    }
}

void play(UIContext& ctx, bool original) {
    state.original = original;
    if (state.dirty || !currentReport(ctx).value("ready", false)) {
        requestRender(ctx, true);
        return;
    }
    call(ctx, "design_preview", {{"original", original}, {"loop", state.loop}});
}

void playArrangement(UIContext& ctx) {
    commitAudioClipEnvelope(ctx);
    const auto view = ctx.scene.audio->invoke("workspace_get", json::object());
    finishAudioTimelineGesture(ctx);
    call(ctx, "arrange_preview", {{"ids", view.at("selected_clips")},
        {"start_seconds", view.at("audition_start_seconds")},
        {"end_seconds", view.at("audition_end_seconds")}, {"loop", state.loop},
        {"background", true}});
}

bool slider(const char* key, const char* label, double fallback, float low, float high,
            const char* format = "%.2f", bool logarithmic = false) {
    float value = static_cast<float>(state.recipe.value(key, fallback));
    const bool changed = ImGui::SliderFloat(property(label).c_str(), &value, low, high, format,
        logarithmic ? ImGuiSliderFlags_Logarithmic : ImGuiSliderFlags_None);
    if (changed) {
        state.recipe[key] = value;
        if (std::string(key) == "gain_db") {
            state.recipe.erase("gain");
        }
        state.dirty = true;
    }
    return ImGui::IsItemDeactivatedAfterEdit();
}

void choosePreset(const char* preset, const char* name) {
    const std::string kind(preset);
    const double frequency = kind == "wood" ? 180 : kind == "tone" ? 440 :
                             kind == "wind" ? 220 : 330;
    state.recipe = {{"mode", "synth"}, {"preset", preset},
        {"frequency_hz", frequency}, {"decay_seconds", kind == "wood" ? 0.15 : 0.8}};
    std::snprintf(state.name, sizeof(state.name), "%s", name);
    state.dirty = true;
    state.original = false;
}

void selectSyntheticDraft(UIContext& ctx) {
    retainDraft();
    if (state.renderJob != 0) {
        call(ctx, "design_job_cancel");
        state.renderJob = 0;
    }
    call(ctx, "design_stop");
    choosePreset("metal", "Metal impact");
    const auto draft = drafts.find("@synthetic");
    if (draft != drafts.end()) {
        state.recipe = draft->second.recipe;
        std::snprintf(state.name, sizeof(state.name), "%s", draft->second.name.c_str());
    }
    state.previewWhenReady = false;
    state.saveWhenReady = false;
    state.dirty = true;
    state.savedPath.clear();
    state.savedAsset.clear();
    state.savedClip.clear();
    state.clipPending = false;
    state.clipError.clear();
}

void waveform(const json& report, const ImVec2& size, double cursor, bool playing) {
    ImGui::InvisibleButton("##designer_wave", size);
    const auto start = ImGui::GetItemRectMin(), end = ImGui::GetItemRectMax();
    auto* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled(start, end, ImGui::GetColorU32(ImGuiCol_FrameBg), 6);
    const auto mid = (start.y + end.y) * 0.5f;
    draw->AddLine(ImVec2(start.x, mid), ImVec2(end.x, mid),
                  ImGui::GetColorU32(ImGuiCol_Border));
    const auto& bins = report.at("waveform");
    const float scale = size.y * 0.44f;
    const float peak = std::max(1.0f, report.at("peak").get<float>());
    for (std::size_t bin = 0; bin < bins.size(); ++bin) {
        const float x = start.x + (bin + 0.5f) / bins.size() * size.x;
        draw->AddLine(ImVec2(x, mid - bins[bin][1].get<float>() / peak * scale),
            ImVec2(x, mid - bins[bin][0].get<float>() / peak * scale),
            ImGui::GetColorU32(ImGuiCol_PlotHistogram), std::max(1.0f, size.x / bins.size()));
    }
    if (playing) {
        const double duration = report.at("duration_seconds");
        const float x = start.x + static_cast<float>(std::fmod(cursor, duration) / duration) * size.x;
        draw->AddLine(ImVec2(x, start.y), ImVec2(x, end.y),
                      ImGui::GetColorU32(ImGuiCol_Text), 2);
    }
}

bool fadeControls(json& recipe, const json& report) {
    const auto first = ImGui::GetItemRectMin(), last = ImGui::GetItemRectMax();
    const float width = last.x - first.x;
    const double duration = report.at("duration_seconds");
    const double fadeIn = recipe.value("fade_in_seconds", 0.003);
    const double fadeOut = recipe.value("fade_out_seconds", 0.02);
    const ImVec2 handles[] = {
        ImVec2(first.x + static_cast<float>(std::min(1.0, fadeIn / duration)) * width, first.y + 8),
        ImVec2(last.x - static_cast<float>(std::min(1.0, fadeOut / duration)) * width, first.y + 8)};
    auto* draw = ImGui::GetWindowDrawList();
    draw->AddLine(ImVec2(first.x, last.y), handles[0], ImGui::GetColorU32(ImGuiCol_SliderGrabActive));
    draw->AddLine(handles[1], last, ImGui::GetColorU32(ImGuiCol_SliderGrabActive));
    const auto mouse = ImGui::GetIO().MousePos;
    for (int handle = 0; handle < 2; ++handle) {
        const auto point = handles[handle];
        draw->AddRectFilled(ImVec2(point.x - 5, point.y - 5), ImVec2(point.x + 5, point.y + 5),
                            ImGui::GetColorU32(ImGuiCol_SliderGrabActive));
        if (ImGui::IsItemHovered() && std::abs(mouse.x - point.x) < 12 &&
            std::abs(mouse.y - point.y) < 16) {
            ImGui::SetTooltip("Drag fade %s (%.3f s)", handle == 0 ? "in" : "out",
                               handle == 0 ? fadeIn : fadeOut);
            if (ImGui::IsMouseClicked(0)) {
                state.fadeHandle = handle;
            }
        }
    }
    if (state.fadeHandle >= 0 && ImGui::IsItemActive()) {
        const double position = std::clamp((mouse.x - first.x) / width, 0.0f, 1.0f);
        recipe[state.fadeHandle == 0 ? "fade_in_seconds" : "fade_out_seconds"] =
            std::min(10.0, (state.fadeHandle == 0 ? position : 1 - position) * duration);
    }
    if (state.fadeHandle >= 0 && ImGui::IsItemDeactivated()) {
        state.fadeHandle = -1;
        return true;
    }
    return false;
}

void curve(const char* id, const std::vector<float>& values, const ImVec2& size) {
    ImGui::InvisibleButton(id, size);
    const auto start = ImGui::GetItemRectMin(), end = ImGui::GetItemRectMax();
    auto* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled(start, end, ImGui::GetColorU32(ImGuiCol_FrameBg), 5);
    for (int line = 1; line < 4; ++line) {
        const float y = start.y + size.y * line / 4;
        draw->AddLine(ImVec2(start.x, y), ImVec2(end.x, y),
                      ImGui::GetColorU32(ImGuiCol_Border));
    }
    for (std::size_t index = 1; index < values.size(); ++index) {
        const auto point = [&](std::size_t at) {
            return ImVec2(start.x + size.x * at / (values.size() - 1),
                          end.y - size.y * std::clamp(values[at], 0.0f, 1.0f));
        };
        draw->AddLine(point(index - 1), point(index),
                      ImGui::GetColorU32(ImGuiCol_PlotLines), 2);
    }
}

bool envelopeEditor(UIContext& ctx, const json& report, float width) {
    ImGui::TextWrapped("Amplitude envelope | drag Attack, Sustain or Release");
    curve("##designer_envelope", report.at("envelope").get<std::vector<float>>(),
           ImVec2(width, designViewPixels(70)));
    const auto start = ImGui::GetItemRectMin(), end = ImGui::GetItemRectMax();
    const double duration = state.recipe.value("duration_seconds", 2.0);
    const double attack = state.recipe.value("attack_seconds", 0.003);
    const double release = state.recipe.value("release_seconds", 0.1);
    const double sustain = state.recipe.value("sustain", 0.7);
    const ImVec2 handles[] = {
        ImVec2(start.x + static_cast<float>(std::min(1.0, attack / duration)) * width, start.y + 5),
        ImVec2(start.x + width * 0.5f, end.y - static_cast<float>(sustain) * designViewPixels(70)),
        ImVec2(end.x - static_cast<float>(std::min(0.5, release / duration)) * width, end.y - 5)};
    const char* names[] = {"Attack", "Sustain", "Release"};
    const auto mouse = ImGui::GetIO().MousePos;
    for (int handle = 0; handle < 3; ++handle) {
        ImGui::GetWindowDrawList()->AddCircleFilled(handles[handle], 5,
            ImGui::GetColorU32(ImGuiCol_SliderGrabActive));
        const float dx = mouse.x - handles[handle].x, dy = mouse.y - handles[handle].y;
        if (ImGui::IsItemHovered() && dx * dx + dy * dy < 225) {
            ImGui::SetTooltip("Drag %s", names[handle]);
            if (ImGui::IsMouseClicked(0)) {
                state.envelopeHandle = handle;
            }
        }
    }
    if (state.envelopeHandle >= 0 && ImGui::IsItemActive()) {
        const double position = std::clamp((mouse.x - start.x) / width, 0.0f, 1.0f);
        if (state.envelopeHandle == 0) {
            state.recipe["attack_seconds"] = std::clamp(position * duration, 0.001, 5.0);
        } else if (state.envelopeHandle == 1) {
            state.recipe["sustain"] = std::clamp((end.y - mouse.y) / designViewPixels(70), 0.0f, 1.0f);
        } else {
            state.recipe["release_seconds"] = std::clamp((1 - position) * duration, 0.001,
                                                        std::min(5.0, duration * 0.5));
        }
        state.dirty = true;
    }
    if (state.envelopeHandle >= 0 && ImGui::IsItemDeactivated()) {
        state.envelopeHandle = -1;
        return true;
    }
    return false;
}
} // namespace

bool isSoundDesignerOpen(UIContext& ctx) {
    return ctx.scene.audio->invoke("workspace_get", json::object()).value("designer_open", false);
}

bool canSoundDesignerHistory(UIContext& ctx, bool redo) {
    sync(ctx);
    pollRender(ctx);
    if (rtapi::renderOutputPending() || state.renderJob != 0) {
        return false;
    }
    return currentReport(ctx).value(redo ? "redo_available" : "undo_available", false);
}

void stepSoundDesignerHistory(UIContext& ctx, bool redo) {
    if (canSoundDesignerHistory(ctx, redo)) {
        followRecipeJob(call(ctx, redo ? "design_redo" : "design_undo", {{"background", true}}));
    }
}

void openSoundDesigner(UIContext& ctx, const std::string& assetId, bool equalizer) {
    sync(ctx);
    if (!assetId.empty()) {
        const auto scene = ctx.scene.audio->snapshot();
        for (const auto& asset : scene.at("assets")) {
            if (asset.at("id").get<std::string>() == assetId) {
                retainDraft();
                const auto effects = state.recipe.value("effects", json::array());
                if (state.renderJob != 0) {
                    call(ctx, "design_job_cancel");
                    state.renderJob = 0;
                }
                state.previewWhenReady = false;
                state.saveWhenReady = false;
                state.savedPath.clear();
                state.savedAsset.clear();
                state.savedClip.clear();
                state.clipPending = false;
                state.clipError.clear();
                state.error.clear();
                state.recipe = {{"mode", "asset"}, {"asset_id", assetId},
                    {"source_in_seconds", 0},
                    {"source_out_seconds", std::min(audio::designMaxSeconds,
                        asset.at("duration").get<double>())}};
                if (equalizer) {
                    const auto catalogue = ctx.scene.audio->invoke("design_catalogue", json::object());
                    state.recipe["effects"] = json::array({{
                        {"type", "eq"}, {"enabled", true}, {"mix", 1}, {"output_db", 0},
                        {"bands", catalogue.at("eq_presets")[0].at("bands")}}});
                }
                std::snprintf(state.name, sizeof(state.name), "%s edited",
                               asset.at("name").get<std::string>().c_str());
                const auto draft = drafts.find(assetId);
                if (!equalizer && !state.keepEffects && draft != drafts.end()) {
                    state.recipe = draft->second.recipe;
                    std::snprintf(state.name, sizeof(state.name), "%s",
                        draft->second.name.c_str());
                }
                if (!equalizer && state.keepEffects) {
                    state.recipe["effects"] = effects;
                }
                state.dirty = true;
                state.original = false;
                call(ctx, "design_stop");
                break;
            }
        }
    }
    finishAudioTimelineGesture(ctx);
    workspaceCall(ctx, "workspace_update", {{"designer_open", true},
        {"designer_page", equalizer ? "effects" :
            state.recipe.value("mode", "synth") == "asset" ? "edit" : "generate"},
        {"designer_edit_view", "source"}, {"designer_effect", 0}});
    rtapi::setBottomEditor("audio");
    if (ctx.scene_ui_ptr) {
        ctx.scene_ui_ptr->tab_to_focus = "Audio";
    }
}

void reuseSoundRecipe(UIContext& ctx, const std::string& assetId) {
    sync(ctx);
    retainDraft();
    const auto result = call(ctx, "design_reuse", {{"id", assetId}, {"background", true}});
    if (result.contains("id")) {
        followRecipeJob(result);
        state.savedPath.clear();
        state.savedAsset.clear();
        state.savedClip.clear();
        state.clipPending = false;
        state.clipError.clear();
        const auto scene = ctx.scene.audio->snapshot();
        for (const auto& asset : scene.at("assets")) {
            if (asset.at("id") == assetId) {
                std::snprintf(state.name, sizeof(state.name), "%s",
                    asset.at("name").get<std::string>().c_str());
                break;
            }
        }
        openSoundDesigner(ctx);
    }
}

static void drawSaveAssetControls(UIContext& ctx) {
    ImGui::Separator();
    ImGui::InputText(property("Sound name").c_str(), state.name, sizeof(state.name));
    const auto destination = call(ctx, "design_destination", state.folder.empty() ?
        json::object() : json{{"folder", state.folder}});
    const auto folder = destination.value("folder", std::string());
    ImGui::TextWrapped("Folder: %s", folder.c_str());
    if (ImGui::Button("Choose folder...")) {
        const auto folder = SceneUI::selectFolderDialogW(L"Save designed sounds in");
        if (!folder.empty()) {
            state.folder = folder;
        }
    }
    if (ImGui::Button("Use project folder")) {
        state.folder.clear();
    }
    ImGui::Checkbox("Create timeline clip", &state.insertClip);
    ImGui::TextWrapped("Saves a new asset; existing clips keep their current source.");
    if (!state.targetInitialized) {
        const auto view = ctx.scene.audio->invoke("workspace_get", json::object());
        state.targetTrack = view.value("selected_track", std::string());
        state.targetStart = ctx.scene.timeline.current_frame /
            static_cast<double>(std::max(1, ctx.render_settings.animation_fps));
        state.targetInitialized = true;
    }
    if (state.insertClip || state.clipPending) {
        const auto scene = ctx.scene.audio->snapshot();
        std::string targetLabel = state.targetTrack.empty() ? "Unassigned / legacy bus" :
            "Missing track";
        for (const auto& track : scene.at("tracks")) {
            if (track.at("id") == state.targetTrack) {
                targetLabel = track.at("name").get<std::string>();
            }
        }
        if (ImGui::BeginCombo("Target track", targetLabel.c_str())) {
            if (ImGui::Selectable("Unassigned / legacy bus", state.targetTrack.empty())) {
                state.targetTrack.clear();
            }
            for (const auto& track : scene.at("tracks")) {
                const auto id = track.at("id").get<std::string>();
                const auto label = track.at("name").get<std::string>() + "##" + id;
                if (ImGui::Selectable(label.c_str(), state.targetTrack == id)) {
                    state.targetTrack = id;
                }
            }
            ImGui::EndCombo();
        }
        ImGui::InputDouble("Start time (s)", &state.targetStart, 0, 0, "%.3f");
        if (ImGui::SmallButton("Use current playhead")) {
            state.targetStart = ctx.scene.timeline.current_frame /
                static_cast<double>(std::max(1, ctx.render_settings.animation_fps));
        }
    }
    ImGui::BeginDisabled(rtapi::renderOutputPending() || state.name[0] == 0 ||
        state.renderJob != 0 ||
        state.clipPending || (state.insertClip &&
            (!std::isfinite(state.targetStart) || state.targetStart < 0)));
    if (ImGui::Button(state.insertClip ? "Save and create clip" : "Save as new asset",
            ImVec2(-1, 0))) {
        if (state.dirty) {
            requestRender(ctx, false);
            state.saveWhenReady = state.renderJob != 0;
        } else {
            saveAsset(ctx);
        }
    }
    ImGui::EndDisabled();
    ImGui::TextWrapped("Names are numbered automatically when a file already exists.");
    if (!state.savedPath.empty()) {
        ImGui::TextWrapped("Asset saved: %s", state.savedPath.c_str());
    }
    if (!state.savedClip.empty()) {
        ImGui::TextWrapped("Timeline clip created: %s", state.savedClip.c_str());
    }
    if (state.clipPending) {
        ImGui::TextWrapped("Asset saved; clip was not created: %s", state.clipError.c_str());
        ImGui::BeginDisabled(rtapi::renderOutputPending() || state.renderJob != 0);
        if (ImGui::Button("Retry clip creation only")) {
            insertSavedClip(ctx);
        }
        ImGui::EndDisabled();
    }
    if (!state.error.empty()) {
        ImGui::TextWrapped("%s", state.error.c_str());
    }
}

static void drawSoundDesignControls(UIContext& ctx, bool primary = false) {
    sync(ctx);
    pollRender(ctx);
    ImGui::TextUnformatted("Sound design");
    bool finished = false;
    const auto before = state.recipe;
    const auto view = ctx.scene.audio->invoke("workspace_get", json::object());
    const auto page = view.value("designer_page", std::string("generate"));
    const bool synth = state.recipe.value("mode", "synth") == "synth";
    if (primary && ImGui::BeginCombo(property("Input").c_str(), synth ? "Synthetic" : "Audio asset")) {
        if (ImGui::Selectable("Synthetic", synth)) {
            selectSyntheticDraft(ctx);
            finished = true;
        }
        const auto scene = ctx.scene.audio->snapshot();
        for (const auto& asset : scene.at("assets")) {
            const auto label = asset.at("name").get<std::string>() + "##" +
                               asset.at("id").get<std::string>();
            if (ImGui::Selectable(label.c_str(), !synth &&
                                  state.recipe.value("asset_id", std::string()) ==
                                      asset.at("id").get<std::string>())) {
                openSoundDesigner(ctx, asset.at("id"));
                finished = true;
            }
        }
        ImGui::EndCombo();
    }
    if (!primary) {
        if (page == "generate" && state.recipe.value("mode", "synth") == "synth") {
            const auto library = ctx.scene.audio->invoke("design_catalogue", json::object());
            if (ImGui::BeginCombo(property("Sound recipe").c_str(), "Choose complete preset...")) {
                for (const auto& entry : library.at("presets")) {
                    if (ImGui::Selectable(entry.at("name").get<std::string>().c_str())) {
                        state.recipe = entry.at("recipe");
                        std::snprintf(state.name, sizeof(state.name), "%s", entry.at("name").get<std::string>().c_str());
                        state.dirty = true;
                        finished = true;
                    }
                }
                ImGui::EndCombo();
            }
            if (ImGui::CollapsingHeader("Generator details")) {
                const auto catalogue = ctx.scene.audio->invoke("synth_presets", json::object());
                const auto preset = state.recipe.value("preset", "metal");
                const bool layered = !state.recipe.value("layers", json::array()).empty();
                ImGui::BeginDisabled(layered);
                if (ImGui::BeginCombo(property("Character").c_str(), preset.c_str())) {
                    for (const auto& entry : catalogue.at("presets")) {
                        if (ImGui::Selectable(entry.at("name").get<std::string>().c_str(),
                                              entry.at("id").get<std::string>() == preset)) {
                            choosePreset(entry.at("id").get<std::string>().c_str(),
                                         entry.at("name").get<std::string>().c_str());
                            finished = true;
                        }
                    }
                    ImGui::EndCombo();
                }
                ImGui::EndDisabled();
                finished |= slider("duration_seconds", "Duration (s)", 2, 0.05f,
                    static_cast<float>(audio::designMaxSeconds));
                ImGui::BeginDisabled(layered);
                finished |= slider("frequency_hz", "Body frequency (Hz)", 330, 20, 4000, "%.0f", true);
                ImGui::EndDisabled();
                finished |= slider("decay_seconds", "Decay / damping (s)", 0.8, 0.01f, 10, "%.3f", true);
                ImGui::BeginDisabled(layered);
                finished |= slider("brightness", "Brightness", 0.5, 0, 1);
                finished |= slider("noise_mix", "Noise / texture", preset == "wind" ? 1 : 0.15, 0, 1);
                ImGui::EndDisabled();
                if (ImGui::CollapsingHeader("Envelope")) {
                    finished |= slider("attack_seconds", "Attack (s)", 0.003, 0.001f, 5, "%.3f", true);
                    finished |= slider("sustain", "Sustain", 0.7, 0, 1);
                    finished |= slider("release_seconds", "Release (s)", 0.1, 0.001f, 5, "%.3f", true);
                }
                if (ImGui::CollapsingHeader("Oscillator / modulation")) {
                    const auto wave = state.recipe.value("wave", "sine");
                    ImGui::BeginDisabled(layered || (preset != "tone" && preset != "wind"));
                    if (ImGui::BeginCombo(property("Wave").c_str(), wave.c_str())) {
                        for (const char* item : {"sine", "triangle", "saw", "square"}) {
                            if (ImGui::Selectable(item, wave == item)) {
                                state.recipe["wave"] = item;
                                state.dirty = true;
                                finished = true;
                            }
                        }
                        ImGui::EndCombo();
                    }
                    ImGui::EndDisabled();
                    ImGui::BeginDisabled(layered || preset != "tone");
                    finished |= slider("detune_cents", "Tone detune (cents)", 0, -100, 100);
                    ImGui::EndDisabled();
                    finished |= slider("modulation_hz", "Amplitude modulation (Hz)", 0.7, 0.05f, 20);
                    finished |= slider("modulation_depth", "Modulation depth", 0, 0, 1);
                    std::uint32_t seed = state.recipe.value("seed", std::uint32_t(1));
                    if (ImGui::InputScalar(property("Variation seed").c_str(), ImGuiDataType_U32, &seed)) {
                        state.recipe["seed"] = seed;
                        state.dirty = true;
                    }
                    finished |= ImGui::IsItemDeactivatedAfterEdit();
                }

            }
        } else if (page == "edit" && state.recipe.value("mode", "synth") == "asset") {
            const auto scene = ctx.scene.audio->snapshot();
            double duration = 0;
            for (const auto& asset : scene.at("assets")) {
                if (asset.at("id") == state.recipe.at("asset_id")) {
                    duration = asset.at("duration");
                }
            }
            finished |= slider("source_in_seconds", "Source start (s)", 0, 0,
                                static_cast<float>(duration));
            finished |= slider("source_out_seconds", "Source end (s)", std::min(10.0, duration),
                                0, static_cast<float>(duration));
            ImGui::TextWrapped("Input/output: up to 600 s, subject to PCM memory budget. "
                "Original file preserved.");
        }
        if (page == "edit" && ImGui::CollapsingHeader("Pitch / duration")) {
            finished |= slider("pitch_semitones", "Pitch (semitones)", 0, -24, 24);
            finished |= slider("stretch_ratio", "Duration multiplier", 1, 0.5f, 2);
            finished |= slider("formant_semitones", "Formant shift (semitones)", 0, -12, 12);
            finished |= slider("formant_base_hz", "Formant base (Hz)", 200, 20, 2000, "%.0f", true);
            bool compensate = state.recipe.value("formant_compensate", false);
            if (ImGui::Checkbox("Compensate formants for pitch", &compensate)) {
                state.recipe["formant_compensate"] = compensate;
                finished = true;
            }
            bool reverse = state.recipe.value("reverse", false);
            if (ImGui::Checkbox("Reverse input", &reverse)) {
                state.recipe["reverse"] = reverse;
                finished = true;
            }
            ImGui::TextWrapped("Pitch and duration are independent. Very short ranges may be rejected.");
        }
        if ((page == "edit" || page == "effects") &&
            ImGui::CollapsingHeader("Output / shaping")) {
            finished |= slider("highpass_hz", "Highpass (Hz)", 20, 20, 20000, "%.0f", true);
            finished |= slider("lowpass_hz", "Lowpass (Hz)", 20000, 20, 20000, "%.0f", true);
            finished |= slider("drive", "Saturation", 0, 0, 1);
            finished |= slider("fade_in_seconds", "Edge fade in (s)", 0.003, 0, 10, "%.3f");
            finished |= slider("fade_out_seconds", "Edge fade out (s)", 0.02, 0, 10, "%.3f");
            finished |= slider("loop_crossfade_seconds", "Loop seam crossfade (s)", 0, 0, 1, "%.3f");
            ImGui::TextWrapped("Loop crossfade joins the tail to the head and shortens the result "
                               "by its overlap.");
        }

        if (page != "analyze" && ImGui::CollapsingHeader("Master output")) {
            const double defaultGainDb = 20 * std::log10(std::max(1e-3,
                state.recipe.value("gain", synth ? 0.5 : 1.0)));
            finished |= slider("gain_db", "Output level (dB)", defaultGainDb, -60, 6, "%.1f");
            finished |= slider("tail_seconds", "Effect tail (s)", 0, 0, 5);
        }
        if (page == "analyze") {
            ImGui::TextWrapped("Analysis controls are in the bottom canvas. Peak/RMS, clipping, DC, "
                               "FFT spectrum and spectrogram inspect the selected original/processed sound.");
        }
    }
    if (before != state.recipe) {
        state.dirty = true;
    }
    ImGui::Checkbox("Preview after releasing a control", &state.autoPreview);
    if (finished && state.autoPreview && !rtapi::renderOutputPending()) {
        play(ctx, false);
    }
    if (!primary) {
        return;
    }
    drawSaveAssetControls(ctx);
}

void drawSoundDesignInspector(UIContext& ctx) {
    // Give Designer fields their own focus scope rather than claiming the whole rail.
    if (ImGui::BeginChild("SoundDesignerHistoryInspector", ImVec2(0, 0),
            ImGuiChildFlags_AutoResizeY, ImGuiWindowFlags_NoScrollbar)) {
        registerAudioHistoryWindow(ctx, true);
        drawSoundDesignControls(ctx, true);
    }
    ImGui::EndChild();
}

void drawSoundDesigner(UIContext& ctx) {
    registerAudioHistoryWindow(ctx);
    sync(ctx);
    pollRender(ctx);
    const auto analysisView = ctx.scene.audio->invoke("workspace_get", json::object());
    if (analysisView.value("designer_page", "generate") == "analyze" &&
        analysisView.value("analysis_mode", "live") == "live") {
        ImGui::NewLine();
        ImGui::BeginChild("LiveAudioAnalysisWorkspace", ImVec2(0, 0), false);
        drawAudioAnalysisWorkspace(ctx, currentReport(ctx), state.original, state.recipe);
        ImGui::EndChild();
        return;
    }
    // Keep the toolbar to one compact row; narrow docks scroll horizontally.
    ImGui::NewLine();
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(6, 2));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(6, 3));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 3.0f);
    const bool compact = ImGui::GetContentRegionAvail().x < 1000;
    const float toolbarHeight = ImGui::GetFrameHeight() +
        (compact ? ImGui::GetStyle().ScrollbarSize : 0.0f) + 4;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 2));
    ImGui::BeginChild("DesignerToolbar", ImVec2(0, toolbarHeight), false,
        ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::PopStyleVar();
    const auto groupSeparator = [] {
        ImGui::SameLine(0, 8);
        const auto p = ImGui::GetCursorScreenPos();
        const float height = ImGui::GetFrameHeight();
        ImGui::GetWindowDrawList()->AddLine(ImVec2(p.x, p.y + 3),
            ImVec2(p.x, p.y + height - 3), ImGui::GetColorU32(ImGuiCol_Border));
        ImGui::Dummy(ImVec2(1, height));
        ImGui::SameLine(0, 8);
    };
    const auto scene = ctx.scene.audio->snapshot();
    const bool synthetic = state.recipe.value("mode", "synth") == "synth";
    std::string sourceLabel = synthetic ? "Synthetic draft" : "Missing source asset";
    std::string sourcePath;
    for (const auto& asset : scene.at("assets")) {
        if (!synthetic && asset.at("id") ==
                state.recipe.value("asset_id", std::string())) {
            sourceLabel = asset.at("name").get<std::string>();
            sourcePath = asset.at("path").get<std::string>();
            break;
        }
    }
    ImGui::BeginDisabled(rtapi::renderOutputPending());
    ImGui::SetNextItemWidth(compact ? 120.0f : 170.0f);
    ImGui::InputText("##DesignName", state.name, sizeof(state.name));
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Sound name / saved asset name");
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(compact ? 130.0f : 190.0f);
    if (ImGui::BeginCombo("##DesignSource", sourceLabel.c_str())) {
        ImGui::Checkbox("Keep effects when switching assets", &state.keepEffects);
        ImGui::Separator();
        if (ImGui::Selectable("Synthetic draft", synthetic)) {
            selectSyntheticDraft(ctx);
        }
        for (const auto& asset : scene.at("assets")) {
            const auto id = asset.at("id").get<std::string>();
            const auto label = asset.at("name").get<std::string>() + "##" + id;
            if (ImGui::Selectable(label.c_str(), !synthetic &&
                    state.recipe.value("asset_id", std::string()) == id)) {
                openSoundDesigner(ctx, id);
            }
        }
        ImGui::EndCombo();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("%s\nSwitch source; drafts are retained during this session.\n"
            "Audio context reset clears drafts; save assets to keep your work.",
            sourcePath.empty() ? sourceLabel.c_str() : sourcePath.c_str());
    }
    ImGui::EndDisabled();
    auto status = currentReport(ctx);
    auto workspace = ctx.scene.audio->invoke("workspace_get", json::object());
    const bool arrange = workspace.value("designer_page", "generate") == "edit" &&
        workspace.value("designer_edit_view", "source") == "arrange";
    if (arrange) {
        const auto transport = ctx.scene.audio->invoke("arrange_status", json::object());
        status["playing"] = transport.at("playing");
        status["loop"] = transport.at("loop");
        status["rendering"] = transport.value("rendering", false);
        const auto job = transport.value("job", json::object());
        if (job.value("state", "idle") == "failed") {
            state.error = job.value("error", "Arrangement render failed");
        }
    }
    if (status.at("playing")) {
        if (!arrange) {
            state.original = status.at("audition_original");
        }
        state.loop = status.at("loop");
    }
    if (status.at("ready") && !state.dirty && status.at("recipe") != state.recipe) {
        state.recipe = status.at("recipe");
    }
    groupSeparator();
    ImGui::BeginDisabled(rtapi::renderOutputPending());
    if (ImGui::Button("Play")) {
        if (arrange) {
            playArrangement(ctx);
        } else {
            play(ctx, state.original);
        }
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("Stop")) {
        call(ctx, arrange ? "arrange_stop" : "design_stop");
        state.renderJob = 0;
        state.previewWhenReady = false;
        state.saveWhenReady = false;
    }
    auto view = ctx.scene.audio->invoke("workspace_get", json::object());
    auto page = view.value("designer_page", std::string("generate"));
    float panelZoom = view.value("designer_panel_zoom", 1.0f);
    groupSeparator();
    ImGui::SetNextItemWidth(105);
    if (ImGui::BeginCombo("##DesignerWorkflow", page.c_str())) {
        for (const char* entry : {"generate", "edit", "effects", "analyze"}) {
            if (ImGui::Selectable(entry, page == entry)) {
                finishAudioTimelineGesture(ctx);
                workspaceCall(ctx, "workspace_update", {{"designer_page", entry}});
                page = entry;
            }
        }
        ImGui::EndCombo();
    }
    if (page == "edit") {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(100);
        const auto editView = view.value("designer_edit_view", std::string("source"));
        if (ImGui::BeginCombo("##DesignerEditView", editView.c_str())) {
            for (const char* mode : {"source", "arrange"}) {
                if (ImGui::Selectable(mode, editView == mode)) {
                    finishAudioTimelineGesture(ctx);
                    workspaceCall(ctx, "workspace_update", {{"designer_edit_view", mode}});
                }
            }
            ImGui::EndCombo();
        }
    }
    groupSeparator();
    if (ImGui::Button("Save...")) {
        ImGui::OpenPopup("Save designed asset");
    }
    if (ImGui::BeginPopup("Save designed asset")) {
        ImGui::PushItemWidth(240);
        drawSaveAssetControls(ctx);
        ImGui::PopItemWidth();
        ImGui::EndPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("Options...")) {
        ImGui::OpenPopup("Designer options");
    }
    if (ImGui::BeginPopup("Designer options")) {
        ImGui::TextDisabled("Playback");
        ImGui::BeginDisabled(rtapi::renderOutputPending());
        if (ImGui::Checkbox("Loop", &state.loop) && status.at("playing")) {
            if (arrange) {
                playArrangement(ctx);
            } else {
                play(ctx, state.original);
            }
        }
        if (!arrange) {
            ImGui::Separator();
            ImGui::TextDisabled("Compare / render");
            if (ImGui::RadioButton("Original", state.original)) {
                state.original = true;
                if (status.at("playing")) {
                    play(ctx, true);
                }
            }
            if (ImGui::RadioButton("Processed", !state.original)) {
                state.original = false;
                if (status.at("playing")) {
                    play(ctx, false);
                }
            }
            if (ImGui::Button("Refresh")) {
                requestRender(ctx, false);
            }
        }
        ImGui::EndDisabled();
        ImGui::Separator();
        ImGui::TextDisabled("View");
        ImGui::SetNextItemWidth(110);
        if (ImGui::SliderFloat("##DesignerPanelZoom", &panelZoom, 0.5f, 1.5f, "Panel %.2fx")) {
            workspaceCall(ctx, "workspace_update", {{"designer_panel_zoom", panelZoom}});
            view["designer_panel_zoom"] = panelZoom;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Whole panel scale / Alt+wheel: zoom / Middle drag: pan");
        }
        ImGui::SameLine();
        if (ImGui::Button("Fit")) {
            state.fitCanvasRequested = true;
            workspaceCall(ctx, "workspace_update", {{"designer_canvas_zoom", 1},
                {"designer_panel_zoom", 1}, {"designer_panel_pan_x", 0}, {"designer_panel_pan_y", 0},
                {"designer_canvas_pan_x", 0}, {"designer_canvas_pan_y", 0},
                {"designer_zoom", 1}, {"designer_pan", 0}});
            view = ctx.scene.audio->invoke("workspace_get", json::object());
            panelZoom = 1;
        }
        bool showTools = view.value("designer_tools_open", true);
        bool showProperties = view.value("designer_properties_open", true);
        if (ImGui::Checkbox("Tools panel", &showTools)) {
            workspaceCall(ctx, "workspace_update", {{"designer_tools_open", showTools}});
            view["designer_tools_open"] = showTools;
        }
        if (ImGui::Checkbox("Properties panel", &showProperties)) {
            workspaceCall(ctx, "workspace_update", {{"designer_properties_open", showProperties}});
            view["designer_properties_open"] = showProperties;
        }
        ImGui::EndPopup();
    }
    ImGui::SameLine(0, 12);
    ImGui::AlignTextToFramePadding();
    ImGui::TextDisabled("%s", state.renderJob != 0 ||
        (arrange && status.value("rendering", false)) ? "Rendering..." :
        state.dirty ? "Changed" : "Rendered");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("%s", state.dirty ?
            "Changes are waiting to render. Play renders and auditions the current sound." :
            "Current sound is rendered and ready to audition or save.");
    }
    ImGui::EndChild();
    ImGui::PopStyleVar(3);
    ImGui::Separator();
    const bool toolsOpen = view.value("designer_tools_open", true);
    const bool propertiesOpen = view.value("designer_properties_open", true);
    const bool wide = ImGui::GetContentRegionAvail().x >= 720;
    const auto before = state.recipe;
    bool completed = false;
    static float toolsWidth = 220;
    static float propertiesWidth = 250;
    const float workspaceWidth = ImGui::GetContentRegionAvail().x;
    const float maxSideWidth = std::max(160.0f, (workspaceWidth - 300) * 0.5f);
    const float visibleToolsWidth = std::min(toolsWidth, maxSideWidth);
    const float visiblePropertiesWidth = std::min(propertiesWidth, maxSideWidth);
    if (wide && toolsOpen) {
        ImGui::BeginChild("DesignerTools", ImVec2(visibleToolsWidth, 0), false);
        if (page == "generate" || page == "effects") {
            completed |= drawDesignModules(ctx, state.recipe, page == "effects", 1);
        } else if (page == "edit") {
            const auto editView = view.value("designer_edit_view", std::string("source"));
            for (const char* mode : {"source", "arrange"}) {
                if (ImGui::Selectable(mode, editView == mode)) {
                    finishAudioTimelineGesture(ctx);
                    workspaceCall(ctx, "workspace_update", {{"designer_edit_view", mode}});
                }
            }
            if (editView == "arrange") {
                drawAudioTrackTools(ctx);
                drawAudioClipSelectionTools(ctx);
                const auto data = ctx.scene.audio->snapshot();
                if (ImGui::BeginCombo("Add asset clip", "Choose asset...")) {
                    for (const auto& asset : data.at("assets")) {
                        if (ImGui::Selectable(asset.at("name").get<std::string>().c_str())) {
                            addTimelineAsset(ctx, asset.at("id"));
                        }
                    }
                    ImGui::EndCombo();
                }
                ImGui::TextWrapped("Ctrl+click: multi-select. A: select. B: razor. Drag clips/edges/gain/fades. "
                                   "Right-click: split, duplicate, delete. Track rows route into mixer buses.");
                if (ImGui::SmallButton("Scene transport")) {
                    finishAudioTimelineGesture(ctx);
                    call(ctx, "design_stop");
                    workspaceCall(ctx, "workspace_update", {{"designer_open", false}});
                }
                const auto audition = ctx.scene.audio->invoke("arrange_status", json::object());
                ImGui::TextWrapped("Audition %.3f..%.3f s | selected clips, or all when selection is empty",
                    view.at("audition_start_seconds").get<double>(),
                    view.at("audition_end_seconds").get<double>());
                if (audition.value("ready", false)) {
                    ImGui::Text("Mix peak %.3f | RMS %.3f", audition.at("peak").get<double>(),
                        audition.at("rms").get<double>());
                }
                if (ImGui::CollapsingHeader("Audition range")) {
                    double start = view.at("audition_start_seconds"), end = view.at("audition_end_seconds");
                    bool changed = ImGui::InputDouble("Start", &start, 0, 0, "%.3f",
                        ImGuiInputTextFlags_EnterReturnsTrue);
                    changed |= ImGui::InputDouble("End", &end, 0, 0, "%.3f",
                        ImGuiInputTextFlags_EnterReturnsTrue);
                    if (changed) {
                        workspaceCall(ctx, "workspace_update", {{"audition_start_seconds", start},
                            {"audition_end_seconds", end}});
                    }
                }
            }
        } else {
            ImGui::TextWrapped("Wheel scrolls; Ctrl+wheel zooms; middle drag pans; Fit resets the canvas.");
        }
        ImGui::EndChild();
    }
    if (wide) {
        if (toolsOpen) {
            audioPanelSplitter("##DesignerToolsSplitter", toolsWidth, true, 160, maxSideWidth);
            ImGui::SameLine(0, 0);
        }
        if (audioPanelToggleStrip("##DesignerToolsToggle", toolsOpen, true, "tools")) {
            workspaceCall(ctx, "workspace_update", {{"designer_tools_open", !toolsOpen}});
        }
        ImGui::SameLine(0, 0);
    }
    const float rightReserve = wide ? 13 +
        (propertiesOpen ? visiblePropertiesWidth + 6 : 0) : 0;
    const float canvasWidth = wide ?
        std::max(200.0f, ImGui::GetContentRegionAvail().x - rightReserve) : 0;
    const auto canvasPosition = ImGui::GetCursorScreenPos();
    if (wide) {
        ImGui::SetCursorScreenPos(ImVec2(canvasPosition.x + canvasWidth, canvasPosition.y));
        if (audioPanelToggleStrip("##DesignerPropertiesToggle", propertiesOpen, false,
                "properties")) {
            workspaceCall(ctx, "workspace_update", {{"designer_properties_open", !propertiesOpen}});
        }
        if (propertiesOpen) {
            audioPanelSplitter("##DesignerPropertiesSplitter", propertiesWidth, false,
                160, maxSideWidth);
            ImGui::SameLine(0, 0);
            ImGui::BeginChild("DesignerProperties", ImVec2(visiblePropertiesWidth, 0), false);
            if (page == "generate" || page == "effects") {
                completed |= drawDesignModules(ctx, state.recipe, page == "effects", 2);
            }
            // Mark shared edits before the controls poll any outstanding render.
            if (before != state.recipe) {
                state.dirty = true;
            }
            if (arrange) {
                if (!drawSelectedAudioTrackProperties(ctx)) {
                    drawSelectedAudioClipProperties(ctx);
                }
            } else {
                drawSoundDesignControls(ctx);
            }
            ImGui::EndChild();
        }
    }
    ImGui::SetCursorScreenPos(canvasPosition);
    status = currentReport(ctx);
    const auto viewportSize = ImVec2(wide ? canvasWidth : ImGui::GetContentRegionAvail().x,
        ImGui::GetContentRegionAvail().y);
    float canvasZoom = view.value("designer_canvas_zoom", 1.0f);
    ImGui::SetNextWindowContentSize(ImVec2(
        std::max(1.0f, viewportSize.x - 20) *
            (arrange ? 1.0f : panelZoom * (page == "effects" ? 1.0f : canvasZoom)), 0));
    ImGui::BeginChild("SoundDesignCanvas", ImVec2(canvasWidth, 0), false,
        ImGuiWindowFlags_HorizontalScrollbar);
    designPanelScale = arrange ? 1.0f : panelZoom;
    ImGui::SetWindowFontScale(designPanelScale);
    static std::string appliedPanelView;
    const auto panelViewKey = std::to_string(ctx.scene.audio->generation()) + ":" +
        std::to_string(panelZoom) + ":" +
        std::to_string(view.value("designer_panel_pan_x", 0.0f)) + ":" +
        std::to_string(view.value("designer_panel_pan_y", 0.0f));
    if (!arrange && appliedPanelView != panelViewKey) {
        ImGui::SetScrollX(view.value("designer_panel_pan_x", 0.0f) * ImGui::GetScrollMaxX());
        ImGui::SetScrollY(view.value("designer_panel_pan_y", 0.0f) * ImGui::GetScrollMaxY());
        appliedPanelView = panelViewKey;
    }
    const float graphBudget = std::max(60.0f, viewportSize.y -
        (page == "analyze" ? 190.0f : page == "generate" ? 180.0f : 150.0f));
    const float nominalGraphs = page == "generate" ? 455.0f :
        page == "effects" ? 335.0f : page == "analyze" ? 225.0f : 230.0f;
    designViewScale = std::clamp(graphBudget / nominalGraphs, 0.2f, 1.0f) *
        (page == "effects" ? panelZoom : canvasZoom * panelZoom);
    static std::string appliedView;
    const auto viewKey = std::to_string(canvasZoom) + ":" +
        std::to_string(view.value("designer_canvas_pan_x", 0.0f)) + ":" +
        std::to_string(view.value("designer_canvas_pan_y", 0.0f));
    if (page != "effects" && (appliedView != viewKey || state.fitCanvasRequested)) {
        ImGui::SetScrollX(view.value("designer_canvas_pan_x", 0.0f) * ImGui::GetScrollMaxX());
        ImGui::SetScrollY(view.value("designer_canvas_pan_y", 0.0f) * ImGui::GetScrollMaxY());
        appliedView = viewKey;
        state.fitCanvasRequested = false;
    }
    if (!wide && ImGui::CollapsingHeader("Tools", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (page == "generate" || page == "effects") {
            completed |= drawDesignModules(ctx, state.recipe, page == "effects", 1);
        }
    }
    if (!wide && ImGui::CollapsingHeader("Properties", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (page == "generate" || page == "effects") {
            completed |= drawDesignModules(ctx, state.recipe, page == "effects", 2);
        }
        if (arrange) {
            if (!drawSelectedAudioTrackProperties(ctx)) {
                drawSelectedAudioClipProperties(ctx);
            }
        } else {
            drawSoundDesignControls(ctx);
        }
        status = currentReport(ctx);
    }
    if (state.recipe.value("mode", "synth") == "asset") {
        const auto assets = ctx.scene.audio->snapshot().at("assets");
        for (const auto& asset : assets) {
            if (asset.at("id") == state.recipe.at("asset_id")) {
                ImGui::TextWrapped("Source: %s | %.2f - %.2f s | original file preserved",
                    asset.at("name").get<std::string>().c_str(),
                    state.recipe.value("source_in_seconds", 0.0),
                    state.recipe.value("source_out_seconds", 0.0));
                if (asset.at("duration").get<double>() > audio::designMaxSeconds) {
                    ImGui::TextWrapped("Selected input is limited to 600 seconds. "
                        "Choose the source range in Edit before applying effects.");
                }
                break;
            }
        }
    }
    if (page == "generate") {
        completed |= drawDesignGenerationCanvas(ctx, state.recipe);
    }
    if (arrange) {
        const float firstFrame = static_cast<float>(ctx.render_settings.animation_start_frame);
        const float lastFrame = static_cast<float>(ctx.render_settings.animation_end_frame);
        const float span = std::max(1.0f, lastFrame - firstFrame);
        float arrangementZoom = canvasZoom;
        float arrangementPan = view.value("designer_canvas_pan_x", 0.0f) * span;
        const bool envelopeOpen = view.value("designer_envelope", std::string("none")) != "none";
        const float laneHeight = std::max(80.0f, ImGui::GetContentRegionAvail().y -
            (envelopeOpen ? 180.0f : 35.0f));
        ImGui::BeginChild("ArrangementLanes", ImVec2(0, laneHeight), false);
        drawAudioTimeline(ctx, arrangementZoom, arrangementPan, 100, true);
        ImGui::EndChild();
        drawAudioClipEnvelope(ctx);
        if (arrangementZoom != canvasZoom ||
            arrangementPan != view.value("designer_canvas_pan_x", 0.0f) * span) {
            workspaceCall(ctx, "workspace_update", {
                {"designer_canvas_zoom", std::clamp(arrangementZoom, 1.0f, 8.0f)},
                {"designer_canvas_pan_x", std::clamp(arrangementPan / span, 0.0f, 1.0f)}});
        }
    }
    if (page == "edit" && !arrange) {
        completed |= drawDesignSourceEditor(ctx, state.recipe);
    }
    if (page == "effects") {
        completed |= drawDesignEffectGraph(ctx, state.recipe);
        completed |= drawDesignEffectPad(ctx, state.recipe);
    }
    if (arrange) {
        // The embedded arrangement consumes the canvas and retains its own clip gestures.
    } else if (page == "analyze") {
        completed |= drawAudioAnalysisWorkspace(ctx, status, state.original, state.recipe);
    } else if (!status.at("ready")) {
        ImGui::TextWrapped("Choose layers in Tools, a recipe in Properties or an asset in the main panel. "
                           "Play renders and auditions without creating files.");
    } else {
        if (status.value("output_muted", false) || status.value("audition_output_gain", 1.0) <= 0) {
            ImGui::TextWrapped("Audition is muted by preview output or Master. Check the main Audio/Mixer controls.");
        }
        const bool original = status.at("playing").get<bool>() ?
                              status.at("audition_original").get<bool>() : state.original;
        const auto& report = status.at(original ? "original" : "processed");
        if (!original && status.contains("processing") &&
            status.at("processing").value("clamped_samples_before_loop_join", 0ULL) > 0) {
            const auto& processing = status.at("processing");
            ImGui::TextColored(ImVec4(1, 0.45f, 0.25f, 1), "Rendered output clipped");
            ImGui::TextWrapped("Before output clipping: %.1f dBFS. Lower EQ output trim "
                "or the final output level and render again.",
                processing.value("peak_before_output_clip_dbfs", 0.0));
        }
        if (report.at("rms").get<double>() < 1e-7) {
            ImGui::TextWrapped("Rendered audio is silent. Check enabled layers, source range, "
                               "effect mix and output level.");
        }
        ImGui::TextWrapped("%s | %.2f s | %d channels | peak %.3f | RMS %.3f%s",
            original ? "Original" : "Processed", report.at("duration_seconds").get<double>(),
            report.at("channels").get<int>(), report.at("peak").get<double>(),
            report.at("rms").get<double>(), state.dirty ? " | changes pending" : "");
        const float width = std::max(1.0f, ImGui::GetContentRegionAvail().x);
        if (page != "effects" || ImGui::CollapsingHeader("Rendered waveform")) {
            waveform(report, ImVec2(width, designViewPixels(page == "generate" ? 120 : 80)),
                status.at("cursor_seconds"), status.at("playing"));
        }
        if (page == "edit" && !original) {
            completed |= fadeControls(state.recipe, report);
        }
        if (page == "generate" && !status.at("envelope").empty() &&
            state.recipe.value("mode", "synth") == "synth") {
            completed |= envelopeEditor(ctx, status, width);
        }
    }
    if (before != state.recipe) {
        state.dirty = true;
    }
    if (completed && state.autoPreview && !rtapi::renderOutputPending()) {
        play(ctx, false);
    }
    if (!designPanelsError().empty()) {
        ImGui::TextWrapped("%s", designPanelsError().c_str());
    }
    if (!state.error.empty()) {
        ImGui::TextWrapped("%s", state.error.c_str());
    }
    if (!arrange && ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows) &&
        !ImGui::GetIO().KeyShift) {
        const auto& io = ImGui::GetIO();
        if (io.KeyAlt && io.MouseWheel != 0) {
            panelZoom = std::clamp(panelZoom * std::pow(1.1f, io.MouseWheel), 0.5f, 1.5f);
            workspaceCall(ctx, "workspace_update", {{"designer_panel_zoom", panelZoom}});
        }
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Middle)) {
            const float x = ImGui::GetScrollMaxX() > 0 ? std::clamp(
                (ImGui::GetScrollX() - io.MouseDelta.x) / ImGui::GetScrollMaxX(), 0.0f, 1.0f) : 0;
            const float y = ImGui::GetScrollMaxY() > 0 ? std::clamp(
                (ImGui::GetScrollY() - io.MouseDelta.y) / ImGui::GetScrollMaxY(), 0.0f, 1.0f) : 0;
            workspaceCall(ctx, "workspace_update", {{"designer_panel_pan_x", x},
                {"designer_panel_pan_y", y}});
            ImGui::SetScrollX(x * ImGui::GetScrollMaxX());
            ImGui::SetScrollY(y * ImGui::GetScrollMaxY());
        }
    }
    if (!arrange && page != "effects" && !ImGui::GetIO().KeyAlt && ImGui::IsWindowHovered()) {
        const auto& io = ImGui::GetIO();
        if (io.KeyCtrl && io.MouseWheel != 0) {
            canvasZoom = std::clamp(canvasZoom * std::pow(1.25f, io.MouseWheel), 1.0f, 8.0f);
            workspaceCall(ctx, "workspace_update", {{"designer_canvas_zoom", canvasZoom}});
        }
        if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Middle)) {
            state.fitCanvasRequested = true;
            workspaceCall(ctx, "workspace_update", {{"designer_canvas_zoom", 1},
                {"designer_canvas_pan_x", 0}, {"designer_canvas_pan_y", 0}});
        }
    }
    designViewScale = 1;
    designPanelScale = 1;
    ImGui::SetWindowFontScale(1);
    ImGui::EndChild();

}
} // namespace AudioUI
