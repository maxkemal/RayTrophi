#include "UI/AudioMixerUI.h"
#include "UI/AudioLiveAnalysisUI.h"
#include "UI/AudioWorkspaceUI.h"
#include "UI/AudioPropertyLayout.h"
#include "Api/RtApi.h"
#include "Audio/AudioAuthoring.h"
#include "Audio/AudioService.h"
#include "scene_ui.h"
#include "scene_data.h"
#include "imgui.h"
#include <algorithm>
#include <cmath>
#include <string>

namespace AudioUI {
void drawInspectorMixerLink(UIContext& ctx, const nlohmann::json& source) {
    using json = nlohmann::json;
    const auto name = source.at("bus").get<std::string>();
    ImGui::Separator();
    ImGui::Text("Mixer / %s", name.c_str());
    const auto view = ctx.scene.audio->invoke("mixer_get", json::object());
    for (const auto& bus : view.at("buses")) {
        if (bus.at("name") != name) {
            continue;
        }
        ImGui::TextWrapped("Channel controls affect every sound routed to %s.", name.c_str());
        const float gain = bus.at("gain");
        float db = gain > 0 ? std::max(-60.0f, 20.0f * std::log10(gain)) : -60;
        if (slider("Channel level", db, -60.0f, 12.0412f, "%.1f dB")) {
            workspaceCall(ctx, "mixer_update", {{"bus", name},
                {"gain", db <= -60 ? 0 : std::min(4.0f, std::pow(10.0f, db / 20))}});
        }
        bool muted = bus.at("muted");
        if (ImGui::Checkbox("Mute channel", &muted)) {
            workspaceCall(ctx, "mixer_update", {{"bus", name}, {"muted", muted}});
        }
        const float peak = bus.at("meter").at("peak_dbfs");
        ImGui::TextDisabled("Channel peak %.1f dBFS / before output mute", peak);
        ImGui::ProgressBar(std::clamp((peak + 60) / 60, 0.0f, 1.0f), ImVec2(-1, 6), "");
        if (ImGui::IsItemClicked()) {
            openLiveAnalysis(ctx, name);
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Click for this channel's live spectrum and levels.");
        }
    }
    if (ImGui::Button("Show channel in Mixer")) {
        const auto updated = workspaceCall(ctx, "workspace_update",
            {{"mixer_bus", name}, {"designer_open", false}});
        if (!updated.is_null() && !updated.empty()) {
            rtapi::setBottomEditor("audio_mixer");
        }
    }
}

void drawMixer(UIContext& ctx) {
    if (!ImGui::TreeNode("Audio mixer")) {
        return;
    }
    static std::string error;
    auto call = [&](const std::string& op, const nlohmann::json& p) {
        try {
            audio::author(ctx, ctx.scene_ui_ptr ? &ctx.scene_ui_ptr->history : nullptr, op, p);
            error.clear();
        } catch (const std::exception& e) {
            error = e.what();
        }
    };
    const auto view = ctx.scene.audio->invoke("mixer_get", nlohmann::json::object());
    ImGui::TextWrapped("Enter applies gain in dB (-60 = off, maximum +12.04). "
        "Child solos combine; mute wins. Asset audition bypasses these scene buses.");
    if (ImGui::BeginTable("SceneAudioBuses", 5, ImGuiTableFlags_BordersInnerV |
                         ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("Bus");
        ImGui::TableSetupColumn("Gain dB");
        ImGui::TableSetupColumn("Mute");
        ImGui::TableSetupColumn("Solo");
        ImGui::TableSetupColumn("Peak / RMS");
        ImGui::TableHeadersRow();
        for (const auto& bus : view.at("buses")) {
            const std::string name = bus.at("name");
            ImGui::PushID(name.c_str());
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextUnformatted(name.c_str());
            ImGui::TableSetColumnIndex(1);
            const float gain = bus.at("gain");
            float gainDb = gain > 0 ? std::max(-60.0f, 20.0f * std::log10(gain)) : -60;
            ImGui::SetNextItemWidth(-1);
            if (ImGui::InputFloat("##Gain", &gainDb, 0, 0, "%.2f",
                                  ImGuiInputTextFlags_EnterReturnsTrue)) {
                const float value = gainDb <= -60 ? 0 : std::pow(10.0f, gainDb / 20.0f);
                call("mixer_update", {{"bus", name}, {"gain", value}});
            }
            ImGui::TableSetColumnIndex(2);
            bool muted = bus.at("muted");
            if (ImGui::Checkbox("##Mute", &muted)) {
                call("mixer_update", {{"bus", name}, {"muted", muted}});
            }
            ImGui::TableSetColumnIndex(3);
            bool solo = bus.at("solo");
            ImGui::BeginDisabled(name == "Master");
            if (ImGui::Checkbox("##Solo", &solo)) {
                call("mixer_update", {{"bus", name}, {"solo", solo}});
            }
            ImGui::EndDisabled();
            ImGui::TableSetColumnIndex(4);
            const auto& meter = bus.at("meter");
            if (!meter.at("ready").get<bool>()) {
                ImGui::TextDisabled("No device yet");
            } else {
                const float peak = meter.at("peak_dbfs");
                const float rms = meter.at("rms_dbfs");
                const bool overload = meter.at("overload_blocks").get<unsigned>() > 0;
                if (overload) {
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, 0.35f, 0.25f, 1));
                }
                ImGui::Text("%.1f / %.1f", peak, rms);
                ImGui::ProgressBar(std::clamp((peak + 60) / 60, 0.0f, 1.0f), ImVec2(-1, 5), "");
                ImGui::TextDisabled("Hold %.1f%s", meter.at("peak_hold_dbfs").get<float>(),
                                    overload ? " OVER" : "");
                if (overload) {
                    ImGui::PopStyleColor();
                }
            }
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
    if (ImGui::Button("Reset meter holds")) {
        call("mixer_reset_peaks", nlohmann::json::object());
    }
    if (view.at("output_muted").get<bool>()) {
        ImGui::TextWrapped("Preview output is muted. Bus meters still measure the scene mix.");
    }
    if (!error.empty()) {
        ImGui::TextWrapped("Mixer: %s", error.c_str());
    }
    ImGui::TreePop();
}
}
