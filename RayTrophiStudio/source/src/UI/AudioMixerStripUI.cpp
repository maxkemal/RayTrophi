#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "UI/AudioWorkspaceUI.h"
#include "Audio/AudioAuthoring.h"
#include "Audio/AudioService.h"
#include "Api/RtApi.h"
#include "scene_ui.h"
#include "scene_data.h"
#include "imgui.h"
#include "../../../external/imgui-knobs/imgui-knobs.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <stdexcept>

namespace AudioUI {
namespace {
using json = nlohmann::json;
struct Edit {
    json before;
    bool active = false;
    float value = 0;
    std::weak_ptr<audio::AudioService> owner;
};
std::map<std::string, Edit> edits;
std::string mixerError;

void applyDrag(UIContext& ctx, Edit& edit, bool changed, const std::string& bus,
               const char* key, float value) {
    if (ImGui::IsItemActivated() || (changed && !edit.active)) {
        edit.before = ctx.scene.audio->snapshot();
        edit.owner = ctx.scene.audio;
        edit.active = true;
    }
    if (changed) {
        try {
            if (rtapi::renderOutputPending()) {
                throw std::runtime_error("Audio mixer is locked by the render job");
            }
            ctx.scene.audio->invoke("mixer_update", {{"bus", bus}, {key, value}});
            mixerError.clear();
        } catch (const std::exception& exception) {
            mixerError = exception.what();
        }
    }
    if (edit.active && ImGui::IsItemDeactivated()) {
        if (edit.owner.lock() == ctx.scene.audio) {
            audio::recordEdit(ctx, ctx.scene_ui_ptr ? &ctx.scene_ui_ptr->history : nullptr,
                              edit.before, std::string("Audio mixer ") + key);
        }
        edit.active = false;
    }
}

void meter(const json& reading, float height) {
    const auto origin = ImGui::GetCursorScreenPos();
    const float width = 23;
    auto* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled(origin, ImVec2(origin.x + width, origin.y + height),
                         ImGui::GetColorU32(ImGuiCol_FrameBg), 3);
    if (reading.at("ready").get<bool>()) {
        const auto yAt = [&](float db) {
            return origin.y + height * (1 - std::clamp((db + 60) / 60, 0.0f, 1.0f));
        };
        const float peak = reading.at("peak_dbfs"), rms = reading.at("rms_dbfs");
        draw->AddRectFilled(ImVec2(origin.x + 3, yAt(rms)),
                             ImVec2(origin.x + width - 3, origin.y + height),
                             IM_COL32(72, 191, 117, 255));
        draw->AddLine(ImVec2(origin.x + 2, yAt(peak)),
                       ImVec2(origin.x + width - 2, yAt(peak)), IM_COL32(236, 176, 64, 255), 2);
        const float hold = reading.at("peak_hold_dbfs");
        draw->AddLine(ImVec2(origin.x + 1, yAt(hold)),
                       ImVec2(origin.x + width - 1, yAt(hold)),
                       ImGui::GetColorU32(ImGuiCol_Text), 1);
    }
    ImGui::Dummy(ImVec2(width, height));
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Peak %.1f dBFS\nRMS %.1f dBFS\nHold %.1f dBFS",
            reading.at("peak_dbfs").get<float>(), reading.at("rms_dbfs").get<float>(),
            reading.at("peak_hold_dbfs").get<float>());
    }
}
}

void drawMixerContent(UIContext& ctx) {
    const auto view = ctx.scene.audio->invoke("mixer_get", json::object());
    ImGui::TextDisabled("Gain in dB | stereo balance | meters before preview mute");
    if (ImGui::Button("Reset peak holds")) {
        workspaceCall(ctx, "mixer_reset_peaks");
    }
    ImGui::SameLine();
    bool previewMuted = view.at("output_muted");
    if (ImGui::Checkbox("Mute preview output", &previewMuted)) {
        workspaceCall(ctx, "preview_mute", {{"muted", previewMuted}});
    }
    ImGui::BeginChild("AudioMixerStrips", ImVec2(0, 0), false,
                      ImGuiWindowFlags_HorizontalScrollbar);
    const float height = std::clamp(ImGui::GetContentRegionAvail().y - 185, 70.0f, 210.0f);
    for (const auto& bus : view.at("buses")) {
        const auto name = bus.at("name").get<std::string>();
        ImGui::PushID(name.c_str());
        ImGui::BeginGroup();
        ImGui::TextUnformatted(name.c_str());
        bool muted = bus.at("muted"), solo = bus.at("solo");
        if (ImGui::Checkbox("M", &muted)) {
            workspaceCall(ctx, "mixer_update", {{"bus", name}, {"muted", muted}});
        }
        ImGui::SameLine();
        ImGui::BeginDisabled(name == "Master");
        if (ImGui::Checkbox("S", &solo)) {
            workspaceCall(ctx, "mixer_update", {{"bus", name}, {"solo", solo}});
        }
        ImGui::EndDisabled();
        auto& gainEdit = edits[name + "/gain"];
        if (!gainEdit.active || gainEdit.owner.lock() != ctx.scene.audio) {
            const float gain = bus.at("gain");
            gainEdit.value = gain > 0 ? std::max(-60.0f, 20 * std::log10(gain)) : -60;
            gainEdit.active = false;
        }
        const bool changed = ImGui::VSliderFloat("##GainFader", ImVec2(30, height),
            &gainEdit.value, -60, 12.0412f, "", ImGuiSliderFlags_AlwaysClamp);
        applyDrag(ctx, gainEdit, changed, name, "gain",
            gainEdit.value <= -60 ? 0 : std::min(4.0f, std::pow(10.0f, gainEdit.value / 20)));
        ImGui::SameLine();
        meter(bus.at("meter"), height);
        float gainDb = gainEdit.value;
        ImGui::SetNextItemWidth(82);
        if (ImGui::InputFloat("##GainDb", &gainDb, 0, 0, "%.2f dB",
                              ImGuiInputTextFlags_EnterReturnsTrue)) {
            workspaceCall(ctx, "mixer_update", {{"bus", name},
                {"gain", gainDb <= -60 ? 0 : std::pow(10.0f, gainDb / 20)}});
        }
        auto& panEdit = edits[name + "/pan"];
        if (!panEdit.active || panEdit.owner.lock() != ctx.scene.audio) {
            panEdit.value = bus.value("pan", 0.0f);
            panEdit.active = false;
        }
        const bool panChanged = ImGuiKnobs::Knob("Pan", &panEdit.value, -1, 1, 0.01f, "%.2f",
            ImGuiKnobVariant_Wiper, 48,
            ImGuiKnobFlags_NoInput | ImGuiKnobFlags_AlwaysClamp | ImGuiKnobFlags_ValueTooltip);
        applyDrag(ctx, panEdit, panChanged, name, "pan", panEdit.value);
        if (bus.at("meter").at("overload_blocks").get<unsigned>() > 0) {
            ImGui::TextColored(ImVec4(1, 0.3f, 0.2f, 1), "OVER");
        } else {
            ImGui::TextDisabled("Peak %.1f", bus.at("meter").at("peak_dbfs").get<float>());
        }
        ImGui::EndGroup();
        ImGui::PopID();
        ImGui::SameLine(0, 24);
    }
    ImGui::NewLine();
    if (!mixerError.empty()) {
        ImGui::TextWrapped("%s", mixerError.c_str());
    }
    ImGui::EndChild();
}
}
