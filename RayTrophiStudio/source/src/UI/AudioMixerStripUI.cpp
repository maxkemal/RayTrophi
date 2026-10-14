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
#include <cstdio>
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
    const float width = 32;
    auto* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled(origin, ImVec2(origin.x + width, origin.y + height),
                         ImGui::GetColorU32(ImGuiCol_FrameBg), 3);
    if (reading.at("ready").get<bool>()) {
        const auto yAt = [&](float db) {
            return origin.y + height * (1 - std::clamp((db + 60) / 60, 0.0f, 1.0f));
        };
        const float peak = reading.at("peak_dbfs"), rms = reading.at("rms_dbfs");
        // Segmented level display: energy (RMS), transient peak and hold stay distinct.
        for (float db = -60; db < 0; db += 3) {
            const auto color = db >= -6 ? IM_COL32(231, 93, 87, 255) :
                db >= -18 ? IM_COL32(222, 183, 87, 255) : IM_COL32(68, 193, 156, 255);
            draw->AddRectFilled(ImVec2(origin.x + 3, yAt(db + 2)),
                ImVec2(origin.x + width - 3, yAt(db)),
                db <= rms ? color : IM_COL32(45, 53, 60, 150), 1);
        }
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
    drawAudioEditorNavigation(ctx);
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
    const float height = std::clamp(ImGui::GetContentRegionAvail().y - 225, 65.0f, 260.0f);
    const float stripWidth = std::clamp((ImGui::GetContentRegionAvail().x - 48) / 5,
                                       136.0f, 200.0f);
    for (const auto& bus : view.at("buses")) {
        const auto name = bus.at("name").get<std::string>();
        ImGui::PushID(name.c_str());
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(14, 10));
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImGui::GetStyleColorVec4(ImGuiCol_FrameBg));
        ImGui::BeginChild("Strip", ImVec2(stripWidth, height + 215), true,
                          ImGuiWindowFlags_NoScrollbar);
        auto* draw = ImGui::GetWindowDrawList();
        const auto origin = ImGui::GetCursorScreenPos();
        const auto accent = ImGui::GetColorU32(name == "Master" ?
            ImGuiCol_PlotLinesHovered : ImGuiCol_SliderGrabActive);
        draw->AddRectFilled(origin, ImVec2(origin.x + stripWidth - 28, origin.y + 3),
                            accent, 2);
        ImGui::Dummy(ImVec2(0, 6));
        ImGui::TextUnformatted(name.c_str());
        ImGui::TextDisabled(name == "Master" ? "Stereo output" : "Bus > Master");
        bool muted = bus.at("muted"), solo = bus.at("solo");
        if (ImGui::Selectable("Mute", muted, 0, ImVec2(48, 0))) {
            muted = !muted;
            workspaceCall(ctx, "mixer_update", {{"bus", name}, {"muted", muted}});
        }
        ImGui::SameLine();
        ImGui::BeginDisabled(name == "Master");
        if (ImGui::Selectable("Solo", solo, 0, ImVec2(44, 0))) {
            solo = !solo;
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
        const auto meterOrigin = ImGui::GetCursorScreenPos();
        meter(bus.at("meter"), height);
        for (float db : {0.0f, -6.0f, -12.0f, -24.0f, -36.0f, -60.0f}) {
            char label[16];
            std::snprintf(label, sizeof(label), "%.0f", db);
            draw->AddText(ImVec2(meterOrigin.x + 36,
                meterOrigin.y + height * (1 - (db + 60) / 60) - 5),
                ImGui::GetColorU32(ImGuiCol_TextDisabled), label);
        }
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
        ImGui::EndChild();
        ImGui::PopStyleColor();
        ImGui::PopStyleVar();
        ImGui::PopID();
        ImGui::SameLine(0, 10);
    }
    ImGui::NewLine();
    if (!mixerError.empty()) {
        ImGui::TextWrapped("%s", mixerError.c_str());
    }
    ImGui::EndChild();
}
}
