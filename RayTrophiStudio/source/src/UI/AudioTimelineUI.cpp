#include "UI/AudioSoundDesignerUI.h"
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "UI/AudioWorkspaceUI.h"
#include "Audio/AudioClips.h"
#include "Audio/AudioAuthoring.h"
#include "Audio/AudioService.h"
#include "Api/RtApi.h"
#include "scene_ui.h"
#include "scene_data.h"
#include "imgui.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <vector>
#include <stdexcept>

namespace AudioUI {
namespace {
using json = nlohmann::json;
struct Drag {
    json before;
    std::string id;
    json sceneBefore, lastApplied;
    int mode = 0; // Move, trim left/right, gain, fade in/out.
    float mouseY = 0;
    std::string bus;
    std::vector<std::string> laneBuses;
    std::map<std::string, std::size_t> lanes;
    std::size_t targetLane = 0;
    float mouse = 0;
    double left = 0, right = 0;
    std::weak_ptr<audio::AudioService> owner;
};
Drag drag;
std::string timelineError;
bool razor = false;
struct CachedWave {
    std::string signature;
    json data;
};
std::map<std::string, CachedWave> waves;
std::weak_ptr<audio::AudioService> waveOwner;
int lastDecodeFrame = -1;

const json* waveform(UIContext& ctx, const json& clip, const json& assets) {
    if (waveOwner.lock() != ctx.scene.audio) {
        waves.clear();
        waveOwner = ctx.scene.audio;
    }
    const auto id = clip.at("asset_id").get<std::string>();
    for (const auto& asset : assets) {
        if (asset.at("id") != id) {
            continue;
        }
        const auto signature = asset.at("path").get<std::string>() + "/" + asset.at("duration").dump();
        if (auto found = waves.find(id); found != waves.end() && found->second.signature == signature) {
            return found->second.data.contains("points") ? &found->second.data : nullptr;
        }
        // Decode at most one new visible asset per UI frame.
        if (lastDecodeFrame == ImGui::GetFrameCount()) {
            return nullptr;
        }
        lastDecodeFrame = ImGui::GetFrameCount();
        auto data = workspaceCall(ctx, "waveform", {{"id", id}, {"points", 1024}});
        if (waves.size() >= 32) {
            waves.erase(waves.begin());
        }
        waves[id] = {signature, std::move(data)};
        return waves[id].data.contains("points") ? &waves[id].data : nullptr;
    }
    return nullptr;
}

double snapped(double seconds, double fps, bool snap) {
    return snap ? std::round(seconds * fps) / fps : seconds;
}
}

void finishAudioTimelineGesture(UIContext& ctx) {
    if (!drag.id.empty()) {
        if (drag.owner.lock() == ctx.scene.audio && ctx.scene_ui_ptr) {
            audio::recordEdit(ctx, &ctx.scene_ui_ptr->history, drag.sceneBefore,
                              "Audio clip gesture");
        }
        drag = {};
    }
}

void drawAudioTimeline(UIContext& ctx, float& zoom, float& pan, float legendWidth) {
    if (!ctx.scene_ui_ptr) {
        return;
    }
    drawAudioEditorNavigation(ctx);
    if (ctx.scene.audio->invoke("workspace_get", json::object()).value("designer_open", false)) {
        drawSoundDesigner(ctx);
        return;
    }
    auto& timeline = ctx.scene_ui_ptr->timeline;
    const double fps = std::max(1, ctx.render_settings.animation_fps);
    const double first = ctx.render_settings.animation_start_frame;
    const double last = std::max(ctx.render_settings.animation_end_frame,
                                 ctx.render_settings.animation_start_frame + 1);
    auto data = ctx.scene.audio->snapshot();
    const auto view = ctx.scene.audio->invoke("workspace_get", json::object());
    const auto selected = view.at("selected_clip").get<std::string>();
    bool snap = view.at("snap"), showWaves = view.at("waveforms");
    if (ImGui::Selectable(razor ? "Razor [B]" : "Select [A]", razor, 0, ImVec2(90, 0))) {
        razor = !razor;
    }
    ImGui::SameLine();
    drawSynthesisControls(ctx);
    ImGui::SameLine();
    if (ImGui::Button("Add clip...")) {
        ImGui::OpenPopup("Audio clip assets");
    }
    if (ImGui::BeginPopup("Audio clip assets")) {
        ImGui::TextDisabled("Existing scene assets");
        for (const auto& asset : data.at("assets")) {
            const auto label = asset.at("name").get<std::string>() + "##" +
                asset.at("id").get<std::string>();
            if (ImGui::Selectable(label.c_str())) {
                addTimelineAsset(ctx, asset.at("id"));
            }
        }
        ImGui::Separator();
        ImGui::TextDisabled("Use a scene source's bound sound");
        for (const auto& source : data.at("emitters")) {
            if (source.at("asset_id").get<std::string>().empty()) {
                continue;
            }
            const auto label = source.at("name").get<std::string>() + "##" +
                source.at("id").get<std::string>();
            if (ImGui::Selectable(label.c_str())) {
                const auto added = workspaceCall(ctx, "clip_create", {
                    {"asset_id", source.at("asset_id")}, {"name", source.at("name")},
                    {"bus", source.at("bus")}, {"gain", source.at("gain")},
                    {"pitch", source.at("pitch")},
                    {"start_seconds", timeline.getCurrentFrame() / fps}});
                if (added.contains("id")) {
                    selectClip(ctx, added.at("id"));
                }
            }
        }
        ImGui::TextWrapped("A source clip uses its sound and base controls. Capture records reactions.");
        ImGui::EndPopup();
    }
    ImGui::SameLine();
    bool inspectorOpen = view.at("inspector_open");
    if (ImGui::Checkbox("Inspector", &inspectorOpen)) {
        workspaceCall(ctx, "workspace_update", {{"inspector_open", inspectorOpen}});
    }
    ImGui::SameLine();
    if (ImGui::Checkbox("Snap frames", &snap)) {
        workspaceCall(ctx, "workspace_update", {{"snap", snap}});
    }
    ImGui::SameLine();
    if (ImGui::Checkbox("Waveform", &showWaves)) {
        workspaceCall(ctx, "workspace_update", {{"waveforms", showWaves}});
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("Fit range")) {
        zoom = 1;
        pan = 0;
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("Refresh waveforms")) {
        waves.clear();
    }
    const json* selectedClip = nullptr;
    for (const auto& clip : data.at("clips")) {
        if (clip.at("id") == selected) {
            selectedClip = &clip;
        }
    }
    ImGui::SameLine();
    ImGui::BeginDisabled(!selectedClip);
    if (ImGui::SmallButton("Split at playhead")) {
        const auto split = workspaceCall(ctx, "clip_split", {{"id", selected},
            {"time_seconds", timeline.getCurrentFrame() / fps}});
        if (split.contains("right")) {
            selectClip(ctx, split.at("right").at("id"));
        }
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("Duplicate")) {
        const auto copied = workspaceCall(ctx, "clip_duplicate", {{"id", selected}});
        if (copied.contains("id")) {
            selectClip(ctx, copied.at("id"));
        }
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("Delete")) {
        workspaceCall(ctx, "clip_remove", {{"id", selected}});
        workspaceCall(ctx, "workspace_update", {{"selected_clip", ""}});
    }
    ImGui::EndDisabled();
    if (data.at("clips").empty()) {
        ImGui::TextDisabled("Add a clip from Audio > Assets, or capture played scene sources.");
    }
    if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) &&
        !ImGui::GetIO().WantTextInput && selectedClip &&
        ImGui::IsKeyPressed(ImGuiKey_Delete)) {
        workspaceCall(ctx, "clip_remove", {{"id", selected}});
        workspaceCall(ctx, "workspace_update", {{"selected_clip", ""}});
    }
    if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) &&
        !ImGui::GetIO().WantTextInput && drag.id.empty()) {
        if (ImGui::IsKeyPressed(ImGuiKey_B)) {
            razor = true;
        }
        if (ImGui::IsKeyPressed(ImGuiKey_A)) {
            razor = false;
        }
    }
    if (!timelineError.empty()) {
        ImGui::TextWrapped("%s", timelineError.c_str());
    }
    ImGui::BeginChild("AudioTimelineCanvas", ImVec2(0, 0), false,
                      ImGuiWindowFlags_HorizontalScrollbar);
    auto origin = ImGui::GetCursorScreenPos();
    const float available = std::max(80.0f, ImGui::GetContentRegionAvail().x - legendWidth);
    const float x0 = origin.x + legendWidth;
    const double scale = available * zoom / (last - first);
    const auto xAt = [&](double seconds) {
        return x0 + static_cast<float>((seconds * fps - first - pan) * scale);
    };
    auto* draw = ImGui::GetWindowDrawList();
    const auto& io = ImGui::GetIO();
    const float ruler = 27, rowHeight = 80;
    const auto secondsAt = [&](float x) { return ((x - x0) / scale + first + pan) / fps; };
    draw->AddRectFilled(origin, ImVec2(x0 + available, origin.y + ruler),
                         ImGui::GetColorU32(ImGuiCol_FrameBg));
    const int step = std::max(1, static_cast<int>(std::ceil(75 / scale)));
    const int visibleFirst = std::max(0, static_cast<int>(first + pan));
    for (int frame = (visibleFirst / step) * step;
         frame <= visibleFirst + static_cast<int>(available / scale) + step; frame += step) {
        const float x = xAt(frame / fps);
        if (x < x0 || x > x0 + available) {
            continue;
        }
        char label[32];
        std::snprintf(label, sizeof(label), "%d", frame);
        draw->AddText(ImVec2(x + 3, origin.y + 3), ImGui::GetColorU32(ImGuiCol_Text), label);
        draw->AddLine(ImVec2(x, origin.y + ruler - 6), ImVec2(x, origin.y + ruler),
                       ImGui::GetColorU32(ImGuiCol_Border));
    }
    ImGui::SetCursorScreenPos(ImVec2(x0, origin.y));
    ImGui::InvisibleButton("AudioRuler", ImVec2(available, ruler));
    if (ImGui::IsItemActive() && !rtapi::renderOutputPending()) {
        const int frame = static_cast<int>(std::llround(secondsAt(io.MousePos.x) * fps));
        rtapi::setFrame(std::clamp(frame, static_cast<int>(first), static_cast<int>(last)));
    }
    struct Row {
        std::string bus;
        std::vector<json> clips;
        double end = -1;
    };
    std::vector<Row> rows;
    const auto buildRows = [&]() {
        rows.clear();
        if (!drag.id.empty() && drag.owner.lock() == ctx.scene.audio) {
            // Freeze lane geometry while held: overlap packing must not move drop targets.
            for (const auto& bus : drag.laneBuses) {
                rows.push_back({bus});
            }
            for (const auto& clip : data.at("clips")) {
                const auto id = clip.at("id").get<std::string>();
                const auto lane = id == drag.id ? drag.targetLane : drag.lanes.at(id);
                rows.at(lane).clips.push_back(clip);
            }
            return;
        }
        for (const char* bus : {"Master", "Weather", "Effects", "Ambience", "Music"}) {
            const auto base = rows.size();
            rows.push_back({bus});
            std::vector<json> clips;
            for (const auto& clip : data.at("clips")) {
                if (clip.at("bus") == bus) {
                    clips.push_back(clip);
                }
            }
            std::sort(clips.begin(), clips.end(), [](const json& a, const json& b) {
                return a.at("start_seconds").get<double>() < b.at("start_seconds").get<double>();
            });
            for (const auto& clip : clips) {
                std::size_t row = base;
                const double start = clip.at("start_seconds");
                while (row < rows.size() && rows[row].end > start) {
                    ++row;
                }
                if (row == rows.size()) {
                    rows.push_back({bus});
                }
                rows[row].clips.push_back(clip);
                rows[row].end = start + clip.at("duration_seconds").get<double>();
            }
        }
    };
    if (!drag.id.empty()) {
        bool sameCollection = data.at("clips").size() == drag.lanes.size();
        for (const auto& clip : data.at("clips")) {
            sameCollection = sameCollection && drag.lanes.contains(clip.at("id").get<std::string>());
        }
        if (!sameCollection) {
            finishAudioTimelineGesture(ctx);
        }
    }
    buildRows();
    // Update the canonical clip before drawing so the visual and runtime agree.
    if (!drag.id.empty() && drag.owner.lock() != ctx.scene.audio) {
        drag = {};
    }
    if (!drag.id.empty()) {
        auto current = std::find_if(data["clips"].begin(), data["clips"].end(),
            [&](const json& clip) { return clip.at("id") == drag.id; });
        if (current == data["clips"].end() || *current != drag.lastApplied) {
            // A script/IPC edit wins; never overwrite concurrent authoring.
            audio::recordEdit(ctx, &ctx.scene_ui_ptr->history, drag.sceneBefore,
                              "Audio clip gesture");
            drag = {};
        } else if (rtapi::renderOutputPending()) {
            audio::recordEdit(ctx, &ctx.scene_ui_ptr->history, drag.sceneBefore,
                              "Audio clip gesture");
            drag = {};
        } else {
            const double initial = drag.before.at("start_seconds");
            const double duration = drag.before.at("duration_seconds");
            const double finish = initial + duration;
            const double delta = (io.MousePos.x - drag.mouse) / (scale * fps);
            try {
                json patch = {{"id", drag.id}};
                if (drag.mode == 1) {
                    drag.left = std::clamp(snapped(initial + delta, fps, snap),
                                           0.0, 86400.0 - duration);
                    drag.right = drag.left + duration;
                    const int target = static_cast<int>(std::floor(
                        (io.MousePos.y - origin.y - ruler) / rowHeight));
                    if (target >= 0 && target < static_cast<int>(rows.size()) &&
                        io.MousePos.x >= x0 && io.MousePos.x <= x0 + available) {
                        drag.bus = rows[target].bus;
                        drag.targetLane = static_cast<std::size_t>(target);
                    }
                    patch["start_seconds"] = drag.left;
                    patch["bus"] = drag.bus;
                } else if (drag.mode == 2 || drag.mode == 3) {
                    auto candidate = drag.sceneBefore;
                    if (drag.mode == 2) {
                        drag.left = std::clamp(snapped(initial + delta, fps, snap),
                                               initial, finish - 1.0 / 48000);
                        drag.right = finish;
                        patch = audio::AudioClips::mutate(candidate, "clip_trim", {
                            {"id", drag.id}, {"start_seconds", drag.left},
                            {"end_seconds", drag.right}});
                    } else {
                        drag.left = initial;
                        drag.right = std::clamp(snapped(finish + delta, fps, snap),
                                                initial + 1.0 / 48000, 86400.0);
                        patch = audio::AudioClips::mutate(candidate, "clip_resize", {
                            {"id", drag.id}, {"end_seconds", drag.right}, {"auto_loop", true}});
                    }
                    patch.erase("pitch_integral");
                } else if (drag.mode == 4) {
                    const double initialGain = drag.before.at("gain");
                    const double db = initialGain > 0 ? 20 * std::log10(initialGain) : -60;
                    const double adjusted = std::clamp(db + (drag.mouseY - io.MousePos.y) * 0.25,
                                                       -60.0, 24.0823996531);
                    patch["gain"] = adjusted <= -60 ? 0 : std::min(16.0, std::pow(10.0, adjusted / 20));
                } else {
                    const double offset = drag.before.at("envelope_offset_seconds");
                    const double horizon = drag.before.at("envelope_duration_seconds");
                    if (drag.mode == 5) {
                        const double value = std::clamp(snapped(
                            drag.before.at("fade_in_seconds").get<double>() + delta, fps, snap),
                            0.0, offset + duration);
                        patch["fade_in_seconds"] = value;
                    } else {
                        const double value = std::clamp(snapped(
                            drag.before.at("fade_out_seconds").get<double>() - delta, fps, snap),
                            0.0, horizon - offset);
                        patch["fade_out_seconds"] = value;
                    }
                }
                json proposed = *current;
                proposed.update(patch);
                if (proposed != *current) {
                    drag.lastApplied = ctx.scene.audio->invoke("clip_update", patch);
                    timelineError.clear();
                    data = ctx.scene.audio->snapshot();
                    buildRows();
                }
            } catch (const std::exception& exception) {
                timelineError = exception.what();
            }
            if (!io.MouseDown[0]) {
                audio::recordEdit(ctx, &ctx.scene_ui_ptr->history, drag.sceneBefore,
                                  "Audio clip gesture");
                drag = {};
                buildRows();
            }
        }
    }
    const float height = ruler + rows.size() * rowHeight;
    // Drive gestures from stored state even when the clip leaves the visible range.
    for (std::size_t index = 0; index < rows.size(); ++index) {
        const auto& row = rows[index];
        const float top = origin.y + ruler + index * rowHeight;
        const bool destination = drag.mode == 1 && drag.targetLane == index;
        draw->AddRectFilled(ImVec2(origin.x, top), ImVec2(x0 + available, top + rowHeight),
            ImGui::GetColorU32(destination ? ImGuiCol_Header :
                index % 2 == 0 ? ImGuiCol_TableRowBg : ImGuiCol_TableRowBgAlt));
        draw->AddText(ImVec2(origin.x + 6, top + 10), ImGui::GetColorU32(ImGuiCol_Text),
                       row.bus.c_str());
        draw->AddLine(ImVec2(origin.x, top + rowHeight), ImVec2(x0 + available, top + rowHeight),
                       ImGui::GetColorU32(ImGuiCol_Border));
        for (const auto& clip : row.clips) {
            const auto id = clip.at("id").get<std::string>();
            double start = clip.at("start_seconds");
            double end = start + clip.at("duration_seconds").get<double>();
            const float left = xAt(start), right = xAt(end);
            const auto& displayClip = clip;
            const float visibleLeft = std::max(left, x0);
            const float visibleRight = std::min(right, x0 + available);
            if (visibleRight <= visibleLeft) {
                continue;
            }
            const ImVec2 a(visibleLeft, top + 3), b(visibleRight, top + rowHeight - 3);
            const auto fill = clip.at("muted").get<bool>() ? ImGuiCol_FrameBg : ImGuiCol_Header;
            draw->AddRectFilled(a, b, ImGui::GetColorU32(id == selected ? ImGuiCol_HeaderActive : fill), 4);
            draw->PushClipRect(a, b, true);
            if (showWaves) {
                if (const auto* wave = waveform(ctx, clip, data.at("assets"))) {
                    const auto& bins = wave->at("points");
                    const double frames = wave->at("frames");
                    for (float x = visibleLeft; x < visibleRight; x += 3) {
                        const auto sample = audio::AudioClips::evaluate(displayClip, secondsAt(x));
                        if (!sample.active || bins.empty()) {
                            continue;
                        }
                        const double fraction = sample.source_seconds * 48000 / frames;
                        const auto bin = std::min(bins.size() - 1,
                            static_cast<std::size_t>(std::max(0.0, fraction) * bins.size()));
                        float low = 1, high = -1;
                        for (std::size_t channel = 0; channel < bins[bin]["min"].size(); ++channel) {
                            low = std::min(low, bins[bin]["min"][channel].get<float>());
                            high = std::max(high, bins[bin]["max"][channel].get<float>());
                        }
                        const float center = top + 48;
                        const float amplitude = std::min(3.0f, sample.gain);
                        draw->AddLine(ImVec2(x, center - high * 20 * amplitude), ImVec2(x, center - low * 20 * amplitude),
                                       ImGui::GetColorU32(ImGuiCol_TextDisabled));
                    }
                }
            }
            if (clip.at("loop").get<bool>()) {
                draw->AddText(ImVec2(visibleRight - 38, top + 5),
                    ImGui::GetColorU32(ImGuiCol_TextDisabled), "LOOP");
            }
            draw->AddText(ImVec2(left + 7, top + 5), ImGui::GetColorU32(ImGuiCol_Text),
                           clip.at("name").get<std::string>().c_str());
            const double offset = clip.at("envelope_offset_seconds");
            const double horizon = clip.at("envelope_duration_seconds");
            const float fadeIn = std::clamp(xAt(start +
                clip.at("fade_in_seconds").get<double>() - offset), left, right);
            const float fadeOut = std::clamp(xAt(start + horizon - offset -
                clip.at("fade_out_seconds").get<double>()), left, right);
            const float handleY = top + 24;
            const double gain = clip.at("gain");
            const float db = gain > 0 ? static_cast<float>(20 * std::log10(gain)) : -60;
            const float gainY = top + 32 + (12 - std::clamp(db, -36.0f, 12.0f)) * 0.65f;
            const auto accent = ImGui::GetColorU32(ImGuiCol_SliderGrabActive);
            draw->AddLine(ImVec2(left, b.y), ImVec2(fadeIn, handleY), accent, 2);
            draw->AddLine(ImVec2(fadeOut, handleY), ImVec2(right, b.y), accent, 2);
            draw->AddLine(ImVec2(visibleLeft, gainY), ImVec2(visibleRight, gainY), accent);
            draw->AddRectFilled(ImVec2(fadeIn - 4, handleY - 4),
                                ImVec2(fadeIn + 4, handleY + 4), accent, 2);
            draw->AddRectFilled(ImVec2(fadeOut - 4, handleY - 4),
                                ImVec2(fadeOut + 4, handleY + 4), accent, 2);
            draw->PopClipRect();
            ImGui::PushID(id.c_str());
            ImGui::SetCursorScreenPos(a);
            ImGui::InvisibleButton("Clip", ImVec2(b.x - a.x, b.y - a.y));
            const bool nearIn = std::abs(io.MousePos.x - fadeIn) < 8 &&
                                std::abs(io.MousePos.y - handleY) < 8;
            const bool nearOut = std::abs(io.MousePos.x - fadeOut) < 8 &&
                                 std::abs(io.MousePos.y - handleY) < 8;
            const bool nearGain = std::abs(io.MousePos.y - gainY) < 5;
            const int mode = nearIn ? 5 : nearOut ? 6 :
                io.MousePos.x - left < 7 ? 2 : right - io.MousePos.x < 7 ? 3 :
                nearGain ? 4 : 1;
            if (ImGui::IsItemActivated() && !rtapi::renderOutputPending()) {
                selectClip(ctx, id);
                if (razor) {
                    const double cut = snapped(secondsAt(io.MousePos.x), fps, snap);
                    if (cut > start && cut < end) {
                        const auto result = workspaceCall(ctx, "clip_split",
                            {{"id", id}, {"time_seconds", cut}});
                        if (result.contains("right")) {
                            selectClip(ctx, result.at("right").at("id"));
                        }
                    }
                } else {
                    drag = {};
                    drag.before = drag.lastApplied = clip;
                    drag.sceneBefore = ctx.scene.audio->snapshot();
                    drag.id = id;
                    drag.mode = mode;
                    drag.mouse = io.MousePos.x;
                    drag.mouseY = io.MousePos.y;
                    drag.left = start;
                    drag.right = end;
                    drag.bus = clip.at("bus").get<std::string>();
                    drag.owner = ctx.scene.audio;
                    drag.targetLane = index;
                    for (std::size_t lane = 0; lane < rows.size(); ++lane) {
                        drag.laneBuses.push_back(rows[lane].bus);
                        for (const auto& entry : rows[lane].clips) {
                            drag.lanes[entry.at("id").get<std::string>()] = lane;
                        }
                    }
                }
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetMouseCursor(razor ? ImGuiMouseCursor_Hand :
                    mode == 4 ? ImGuiMouseCursor_ResizeNS :
                    mode == 1 ? ImGuiMouseCursor_ResizeAll : ImGuiMouseCursor_ResizeEW);
                ImGui::SetTooltip("%s | %.1f dB\n%s",
                    clip.at("bus").get<std::string>().c_str(), db,
                    razor ? "Click to cut" : mode == 4 ? "Drag gain line up/down" :
                    mode == 5 || mode == 6 ? "Drag fade handle" :
                    mode == 2 || mode == 3 ? "Drag edge to trim; right edge extends and loops past source end" :
                    "Drag to move in time and across buses");
                if (razor) {
                    draw->AddLine(ImVec2(io.MousePos.x, a.y),
                                  ImVec2(io.MousePos.x, b.y), accent, 2);
                }
            }
            ImGui::PopID();
        }
    }
    const float playhead = xAt(timeline.getCurrentFrame() / fps);
    if (playhead >= x0 && playhead <= x0 + available) {
        draw->AddLine(ImVec2(playhead, origin.y), ImVec2(playhead, origin.y + height),
                       ImGui::GetColorU32(ImGuiCol_PlotLinesHovered), 2);
    }
    if (drag.id.empty() && ImGui::IsWindowHovered() && io.MouseWheel != 0) {
        if (io.KeyCtrl) {
            zoom = std::clamp(zoom * std::pow(1.2f, io.MouseWheel), 0.1f, 100.0f);
        } else {
            pan = std::max(0.0f, pan - io.MouseWheel * static_cast<float>(50 / scale));
        }
    }
    ImGui::SetCursorScreenPos(ImVec2(origin.x, origin.y + height));
    ImGui::Dummy(ImVec2(legendWidth + available, 2));
    ImGui::EndChild();
}
}
