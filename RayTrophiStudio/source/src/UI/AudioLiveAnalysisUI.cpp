#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "UI/AudioLiveAnalysisUI.h"
#include "UI/AudioAnalysisGraph.h"
#include "UI/AudioDesignPanels.h"
#include "UI/AudioWorkspaceUI.h"
#include "Audio/AudioService.h"
#include "Audio/AudioMixer.h"
#include "Api/RtApi.h"
#include "scene_ui.h"
#include "scene_data.h"
#include "imgui.h"
#include <algorithm>
#include <cmath>
#include <exception>
#include <utility>

namespace AudioUI {
namespace {
using json = nlohmann::json;

void liveStatus(const json& report) {
    const bool stale = !report.value("fresh", false);
    const bool silent = report.value("silent", false);
    const bool muted = report.value("output_muted", false);
    const bool clipped = report.value("clip_samples_window", 0u) != 0;
    std::string label = stale ? "No new audio frames" :
        silent ? "Channel silent" : "Signal active";
    std::string explanation = stale ?
        "No new audio frames. Showing the last received snapshot." :
        silent ? "This channel is currently silent. Check playback and channel mute." :
        "Receiving live audio from this mixer channel.";
    if (muted) {
        label += " | Output muted";
        explanation += "\nOutput muted: the graph shows the signal before output mute.";
    }
    if (clipped) {
        label += " | Full scale";
        explanation += "\nSignal reaches full scale. Lower the channel level if distortion is audible.";
    }
    // One fixed, non-wrapping row for every signal state, including active audio.
    // Keep graph origin/height and the mouse zoom anchor stable as notices change.
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::BeginChild("LiveAnalysisStatus", ImVec2(0, ImGui::GetTextLineHeight()), false,
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::PopStyleVar();
    const auto color = clipped ? ImVec4(1, 0.4f, 0.25f, 1) :
        muted ? ImVec4(1, 0.7f, 0.3f, 1) : ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled);
    ImGui::TextColored(color, "%s", label.c_str());
    if (ImGui::IsWindowHovered()) {
        ImGui::SetTooltip("%s", explanation.c_str());
    }
    ImGui::EndChild();
}

void live(UIContext& ctx, const json& workspace) {
    const auto bus = workspace.value("mixer_bus", std::string("Master"));
    const auto label = bus == "Master" ? std::string("All audio / Master") : bus;
    ImGui::SetNextItemWidth(220);
    if (ImGui::BeginCombo("Analyze", label.c_str())) {
        for (const auto& name : audio::AudioMixer::names()) {
            if (ImGui::Selectable(name == "Master" ? "All audio / Master" : name.c_str(),
                    name == bus)) {
                workspaceCall(ctx, "workspace_update", {{"mixer_bus", name}});
            }
        }
        ImGui::EndCombo();
    }
    if (ImGui::GetContentRegionAvail().x > 320) {
        ImGui::SameLine();
    }
    const auto graphView = workspace.value("analysis_view", std::string("line"));
    const std::pair<const char*, const char*> views[] = {
        {"line", "Spectrum / line"}, {"filled", "Spectrum / filled"},
        {"bands", "Six bands"}, {"spectrogram", "Spectrogram / time"}};
    const char* graphLabel = views[0].second;
    for (const auto& view : views) {
        if (graphView == view.first) {
            graphLabel = view.second;
        }
    }
    ImGui::SetNextItemWidth(200);
    if (ImGui::BeginCombo("View", graphLabel)) {
        for (const auto& view : views) {
            if (ImGui::Selectable(view.second, graphView == view.first)) {
                workspaceCall(ctx, "workspace_update", {{"analysis_view", view.first}});
            }
        }
        ImGui::EndCombo();
    }
    const bool details = graphView == "spectrogram";
    // Only cache presentation data; the source/bus selection is canonical workspace state.
    static const audio::AudioService* owner = nullptr;
    static std::uint64_t generation = 0;
    static std::string shownBus;
    static double refresh = 0;
    static bool shownDetails = false;
    static json report;
    static std::string error;
    if (owner != ctx.scene.audio.get() || generation != ctx.scene.audio->generation() ||
        shownBus != bus || shownDetails != details || ImGui::GetTime() >= refresh) {
        owner = ctx.scene.audio.get();
        generation = ctx.scene.audio->generation();
        shownBus = bus;
        shownDetails = details;
        refresh = ImGui::GetTime() + 0.07;
        try {
            report = ctx.scene.audio->invoke("analysis_get", {{"bus", bus}, {"details", details}});
            error.clear();
        } catch (const std::exception& exception) {
            report = json::object();
            error = exception.what();
        }
    }
    ImGui::TextWrapped("LIVE | %s", label.c_str());
    ImGui::TextDisabled("Ctrl+wheel: zoom at pointer | Middle drag: pan graph");
    if (!error.empty()) {
        ImGui::TextWrapped("Analysis unavailable: %s", error.c_str());
        return;
    }
    if (!report.value("device_ready", false)) {
        ImGui::TextWrapped("Audio device is not active yet. Play a scene sound to start it.");
        return;
    }
    if (report.value("ready", false)) {
        liveStatus(report);
        ImGui::Text("Peak %.1f dBFS | Average %.1f dBFS",
            report.at("peak_dbfs").get<double>(), report.at("rms_dbfs").get<double>());
        drawLiveAnalysisGraph(ctx, report, workspace);
        ImGui::TextUnformatted("Live bands (dBFS) | bass to treble");
        if (ImGui::BeginTable("LiveBands", 6, ImGuiTableFlags_SizingStretchSame)) {
            for (const auto& band : report.at("bands")) {
                ImGui::TableNextColumn();
                const int hz = band.at("frequency_hz").get<int>();
                ImGui::Text("%s", hz == 1000 ? "1k" : hz == 2000 ? "2k" :
                    hz == 4000 ? "4k" : std::to_string(hz).c_str());
                const float db = band.at("dbfs");
                ImGui::ProgressBar(std::clamp((db + 96) / 96, 0.0f, 1.0f), ImVec2(-1, 8), "");
                ImGui::TextDisabled("%.0f", db);
            }
            ImGui::EndTable();
        }
    } else {
        ImGui::TextWrapped("Waiting for a short live audio window...");
    }
    ImGui::SeparatorText("Channels now (dBFS)");
    if (ImGui::BeginTable("LiveChannels", 3,
            ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("Channel");
        ImGui::TableSetupColumn("Peak");
        ImGui::TableSetupColumn("Average");
        ImGui::TableHeadersRow();
        for (const auto& name : audio::AudioMixer::names()) {
            const auto& meter = report.at("bus_meters").at(name);
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            if (ImGui::Selectable(name.c_str(), name == bus)) {
                workspaceCall(ctx, "workspace_update", {{"mixer_bus", name}});
            }
            ImGui::TableSetColumnIndex(1);
            const float db = meter.at("peak_dbfs");
            ImGui::Text("%.1f", db);
            ImGui::ProgressBar(std::clamp((db + 60) / 60, 0.0f, 1.0f), ImVec2(-1, 5), "");
            ImGui::TableSetColumnIndex(2);
            ImGui::Text("%.1f", meter.at("rms_dbfs").get<double>());
        }
        ImGui::EndTable();
    }
    ImGui::TextWrapped("Master includes scene sounds and any active preview. "
        "Channels are measured after their fader, before output mute; child channels are before Master.");
    if (ImGui::CollapsingHeader("How to read this")) {
        ImGui::TextWrapped("Peak is the loudest instant; Average is the overall level. "
            "0 dBFS is full scale. These are real mixer samples, independent of Designer renders.");
        ImGui::TextWrapped("The six bands measure the mixed signal, not separate direct, "
            "transmitted or reflected acoustic energies.");
        if (report.value("ready", false)) {
            ImGui::Text("%.0f Hz | %u samples | window %.1f ms",
                report.at("sample_rate").get<double>(), report.at("window_frames").get<unsigned>(),
                1000 * report.at("window_frames").get<double>() /
                    report.at("sample_rate").get<double>());
        }
    }
}
} // namespace

void openLiveAnalysis(UIContext& ctx, const std::string& bus) {
    finishAudioTimelineGesture(ctx);
    const auto updated = workspaceCall(ctx, "workspace_update", {
        {"designer_open", true}, {"designer_page", "analyze"},
        {"analysis_mode", "live"}, {"mixer_bus", bus}});
    if (!updated.empty()) {
        rtapi::setBottomEditor("audio");
        if (ctx.scene_ui_ptr) {
            ctx.scene_ui_ptr->tab_to_focus = "Audio";
        }
    }
}

bool drawAudioAnalysisWorkspace(UIContext& ctx, const json& designStatus,
                                bool original, json& recipe) {
    const auto workspace = ctx.scene.audio->invoke("workspace_get", json::object());
    const auto mode = workspace.value("analysis_mode", std::string("live"));
    ImGui::TextUnformatted("Audio / Analyze");
    if (ImGui::SmallButton("Designer")) {
        workspaceCall(ctx, "workspace_update", {{"designer_page", "generate"}});
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("Mixer")) {
        workspaceCall(ctx, "workspace_update", {{"designer_open", false}});
        rtapi::setBottomEditor("audio_mixer");
    }
    if (ImGui::RadioButton("Live scene / channels", mode == "live")) {
        workspaceCall(ctx, "workspace_update", {{"analysis_mode", "live"}});
    }
    if (ImGui::GetContentRegionAvail().x > 420) {
        ImGui::SameLine();
    }
    if (ImGui::RadioButton("Designer: last render", mode == "design")) {
        workspaceCall(ctx, "workspace_update", {{"analysis_mode", "design"}});
    }
    ImGui::Separator();
    if (mode == "live") {
        live(ctx, workspace);
        return false;
    }
    ImGui::TextWrapped("LAST RENDER | Designer only. This is a whole-sound analysis; "
        "it does not follow live playback or mixer channels.");
    return drawDesignAnalysis(ctx, designStatus, original, recipe);
}

} // namespace AudioUI
