#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "UI/AudioSoundDesignerUI.h"
#include "UI/AudioWorkspaceUI.h"
#include "UI/AudioPropertyLayout.h"
#include "Audio/AudioService.h"
#include "Audio/AudioAuthoring.h"
#include "Api/RtApi.h"
#include "scene_data.h"
#include "scene_ui.h"
#include "imgui.h"
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace AudioUI {
namespace {
using json = nlohmann::json;
struct DesignerUI {
    const audio::AudioService* owner = nullptr;
    std::uint64_t generation = 0;
    json recipe = {{"mode", "synth"}, {"preset", "metal"}};
    bool dirty = true;
    bool loop = false;
    bool autoPreview = true;
    bool insertClip = true;
    bool original = false;
    int envelopeHandle = -1;
    char name[128] = "Metal impact";
    std::string folder;
    std::string savedPath;
    std::string error;
};
DesignerUI state;

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
        state.owner = ctx.scene.audio.get();
        state.generation = ctx.scene.audio->generation();
        const auto report = ctx.scene.audio->invoke("design_status", json::object());
        if (report.at("ready")) {
            state.recipe = report.at("recipe");
            state.dirty = false;
        }
    }
}

bool render(UIContext& ctx) {
    const auto result = call(ctx, "design_render", {{"recipe", state.recipe}});
    if (!result.value("ready", false)) {
        return false;
    }
    state.recipe = result.at("recipe");
    state.dirty = false;
    return true;
}

void play(UIContext& ctx, bool original) {
    if (state.dirty && !render(ctx)) {
        return;
    }
    const auto report = call(ctx, "design_preview", {{"original", original}, {"loop", state.loop}});
    if (report.contains("playing")) {
        state.original = original;
    }
}

bool slider(const char* key, const char* label, double fallback, float low, float high,
            const char* format = "%.2f", bool logarithmic = false) {
    float value = static_cast<float>(state.recipe.value(key, fallback));
    const bool changed = ImGui::SliderFloat(property(label).c_str(), &value, low, high, format,
        logarithmic ? ImGuiSliderFlags_Logarithmic : ImGuiSliderFlags_None);
    if (changed) {
        state.recipe[key] = value;
        state.dirty = true;
    }
    return ImGui::IsItemDeactivatedAfterEdit();
}

void nextButton(const char* label) {
    const float right = ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x;
    const float width = ImGui::CalcTextSize(label).x + ImGui::GetStyle().FramePadding.x * 2;
    if (right - ImGui::GetItemRectMax().x > width + ImGui::GetStyle().ItemSpacing.x) {
        ImGui::SameLine();
    }
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

void envelopeEditor(UIContext& ctx, const json& report, float width) {
    ImGui::TextWrapped("Amplitude envelope | drag Attack, Sustain or Release");
    curve("##designer_envelope", report.at("envelope").get<std::vector<float>>(),
           ImVec2(width, 70));
    const auto start = ImGui::GetItemRectMin(), end = ImGui::GetItemRectMax();
    const double duration = state.recipe.value("duration_seconds", 2.0);
    const double attack = state.recipe.value("attack_seconds", 0.003);
    const double release = state.recipe.value("release_seconds", 0.1);
    const double sustain = state.recipe.value("sustain", 0.7);
    const ImVec2 handles[] = {
        ImVec2(start.x + static_cast<float>(std::min(1.0, attack / duration)) * width, start.y + 5),
        ImVec2(start.x + width * 0.5f, end.y - static_cast<float>(sustain) * 70),
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
            state.recipe["sustain"] = std::clamp((end.y - mouse.y) / 70, 0.0f, 1.0f);
        } else {
            state.recipe["release_seconds"] = std::clamp((1 - position) * duration, 0.001,
                                                        std::min(5.0, duration * 0.5));
        }
        state.dirty = true;
    }
    if (state.envelopeHandle >= 0 && ImGui::IsItemDeactivated()) {
        state.envelopeHandle = -1;
        if (state.autoPreview && !rtapi::renderOutputPending()) {
            play(ctx, false);
        }
    }
}
} // namespace

void openSoundDesigner(UIContext& ctx, const std::string& assetId) {
    sync(ctx);
    if (!assetId.empty()) {
        const auto scene = ctx.scene.audio->snapshot();
        for (const auto& asset : scene.at("assets")) {
            if (asset.at("id").get<std::string>() == assetId) {
                state.recipe = {{"mode", "asset"}, {"asset_id", assetId},
                    {"source_in_seconds", 0},
                    {"source_out_seconds", std::min(10.0, asset.at("duration").get<double>())}};
                std::snprintf(state.name, sizeof(state.name), "%s edited",
                               asset.at("name").get<std::string>().c_str());
                state.dirty = true;
                state.original = false;
                break;
            }
        }
    }
    finishAudioTimelineGesture(ctx);
    workspaceCall(ctx, "workspace_update", {{"designer_open", true}, {"inspector_open", true}});
    rtapi::setBottomEditor("audio");
    if (ctx.scene_ui_ptr) {
        ctx.scene_ui_ptr->tab_to_focus = "Audio";
    }
}

void drawSoundDesignInspector(UIContext& ctx) {
    sync(ctx);
    ImGui::TextUnformatted("Sound design");
    bool finished = false;
    const bool synth = state.recipe.value("mode", "synth") == "synth";
    if (ImGui::BeginCombo(property("Input").c_str(), synth ? "Synthetic" : "Audio asset")) {
        if (ImGui::Selectable("Synthetic", synth)) {
            choosePreset("metal", "Metal impact");
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
    if (state.recipe.value("mode", "synth") == "synth") {
        const auto catalogue = ctx.scene.audio->invoke("synth_presets", json::object());
        const auto preset = state.recipe.value("preset", "metal");
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
        finished |= slider("duration_seconds", "Duration (s)", 2, 0.05f, 10);
        finished |= slider("frequency_hz", "Body frequency (Hz)", 330, 20, 4000, "%.0f", true);
        finished |= slider("decay_seconds", "Decay / damping (s)", 0.8, 0.01f, 10, "%.3f", true);
        finished |= slider("brightness", "Brightness", 0.5, 0, 1);
        finished |= slider("noise_mix", "Noise / texture", preset == "wind" ? 1 : 0.15, 0, 1);
        if (ImGui::CollapsingHeader("Envelope", ImGuiTreeNodeFlags_DefaultOpen)) {
            finished |= slider("attack_seconds", "Attack (s)", 0.003, 0.001f, 5, "%.3f", true);
            finished |= slider("sustain", "Sustain", 0.7, 0, 1);
            finished |= slider("release_seconds", "Release (s)", 0.1, 0.001f, 5, "%.3f", true);
        }
        if (ImGui::CollapsingHeader("Oscillator / modulation")) {
            const auto wave = state.recipe.value("wave", "sine");
            ImGui::BeginDisabled(preset != "tone" && preset != "wind");
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
            ImGui::BeginDisabled(preset != "tone");
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
    } else {
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
        ImGui::TextWrapped("Select up to 30 seconds. The original file is preserved.");
    }
    if (ImGui::CollapsingHeader("Pitch / duration")) {
        finished |= slider("pitch_semitones", "Pitch (semitones)", 0, -24, 24);
        finished |= slider("stretch_ratio", "Duration multiplier", 1, 0.5f, 2);
        ImGui::TextWrapped("Pitch and duration are independent. Very short ranges may be rejected.");
    }
    if (ImGui::CollapsingHeader("Filter / shaping", ImGuiTreeNodeFlags_DefaultOpen)) {
        finished |= slider("highpass_hz", "Highpass (Hz)", 20, 20, 20000, "%.0f", true);
        finished |= slider("lowpass_hz", "Lowpass (Hz)", 20000, 20, 20000, "%.0f", true);
        finished |= slider("drive", "Saturation", 0, 0, 1);
        finished |= slider("gain", "Output gain", synth ? 0.5 : 1, 0, 2);
        finished |= slider("fade_in_seconds", "Edge fade in (s)", 0.003, 0, 10, "%.3f");
        finished |= slider("fade_out_seconds", "Edge fade out (s)", 0.02, 0, 10, "%.3f");
        finished |= slider("loop_crossfade_seconds", "Loop seam crossfade (s)", 0, 0, 1, "%.3f");
        ImGui::TextWrapped("Loop crossfade joins the tail to the head and shortens the result "
                           "by its overlap.");
    }
    ImGui::Checkbox("Preview after releasing a control", &state.autoPreview);
    if (finished && state.autoPreview && !rtapi::renderOutputPending()) {
        play(ctx, false);
    }
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
    ImGui::Checkbox("Insert clip at playhead", &state.insertClip);
    ImGui::BeginDisabled(rtapi::renderOutputPending() || state.name[0] == 0);
    if (ImGui::Button("Save asset", ImVec2(-1, 0))) {
        if (!state.dirty || render(ctx)) {
            json params = {{"name", state.name}};
            if (!state.folder.empty()) {
                params["folder"] = state.folder;
            }
            const auto result = call(ctx, "design_save", params);
            if (result.contains("design")) {
                state.savedPath = result.at("design").at("path");
            }
            if (result.contains("asset") && state.insertClip) {
                addTimelineAsset(ctx, result.at("asset").at("id"));
            }
        }
    }
    ImGui::EndDisabled();
    ImGui::TextWrapped("Names are numbered automatically when a file already exists.");
    if (!state.savedPath.empty()) {
        ImGui::TextWrapped("Saved: %s", state.savedPath.c_str());
    }
    if (!state.error.empty()) {
        ImGui::TextWrapped("%s", state.error.c_str());
    }
}

void drawSoundDesigner(UIContext& ctx) {
    sync(ctx);
    auto status = ctx.scene.audio->invoke("design_status", json::object());
    if (status.at("ready") && !state.dirty && status.at("recipe") != state.recipe) {
        state.recipe = status.at("recipe");
    }
    ImGui::TextUnformatted("Sound designer");
    nextButton("Loop preview");
    if (ImGui::Checkbox("Loop preview", &state.loop) && status.at("playing")) {
        play(ctx, status.at("audition_original"));
    }
    ImGui::BeginDisabled(rtapi::renderOutputPending());
    if (ImGui::Button(state.dirty ? "Render + listen" : "Listen processed")) {
        play(ctx, false);
    }
    nextButton("Listen original");
    if (ImGui::Button("Listen original")) {
        play(ctx, true);
    }
    nextButton("Update visuals");
    if (ImGui::Button("Update visuals")) {
        render(ctx);
    }
    ImGui::EndDisabled();
    if (ImGui::Button("Stop")) {
        call(ctx, "design_stop");
    }
    nextButton("Controls / Save");
    if (ImGui::Button("Controls / Save")) {
        workspaceCall(ctx, "workspace_update", {{"inspector_open", true}});
    }
    status = ctx.scene.audio->invoke("design_status", json::object());
    ImGui::BeginChild("SoundDesignCanvas", ImVec2(0, 0), false);
    if (!status.at("ready")) {
        ImGui::TextWrapped("Choose a character or audio asset in the inspector, then Render + listen. "
                           "Preview creates no files or scene assets.");
    } else {
        const bool original = status.at("audition_original");
        const auto& report = status.at(original ? "original" : "processed");
        ImGui::TextWrapped("%s | %.2f s | %d channels | peak %.3f | RMS %.3f%s",
            original ? "Original" : "Processed", report.at("duration_seconds").get<double>(),
            report.at("channels").get<int>(), report.at("peak").get<double>(),
            report.at("rms").get<double>(), state.dirty ? " | changes pending" : "");
        const float width = std::max(1.0f, ImGui::GetContentRegionAvail().x);
        waveform(report, ImVec2(width, 110), status.at("cursor_seconds"), status.at("playing"));
        if (!status.at("envelope").empty() && state.recipe.value("mode", "synth") == "synth") {
            envelopeEditor(ctx, status, width);
        }
        ImGui::TextWrapped("Filter response | 20 Hz - 20 kHz | -60 to 0 dB");
        std::vector<float> response;
        for (const auto& point : status.at("filter_response")) {
            response.push_back(static_cast<float>((point[1].get<double>() + 60) / 60));
        }
        curve("##designer_filter", response, ImVec2(width, 55));
        ImGui::TextWrapped("Spectrum of loudest sampled window | 20 Hz - 20 kHz | -120 to 0 dB");
        std::vector<float> spectrum;
        for (const auto& point : report.at("spectrum")) {
            spectrum.push_back(static_cast<float>((point[1].get<double>() + 120) / 120));
        }
        curve("##designer_spectrum", spectrum, ImVec2(width, 70));
    }
    if (!state.error.empty()) {
        ImGui::TextWrapped("%s", state.error.c_str());
    }
    ImGui::EndChild();
}
} // namespace AudioUI
