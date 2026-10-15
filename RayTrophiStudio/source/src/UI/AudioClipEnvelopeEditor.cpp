#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "UI/AudioClipEnvelopeEditor.h"
#include "UI/AudioWorkspaceUI.h"
#include "Audio/AudioService.h"
#include "scene_data.h"
#include "imgui.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include "scene_ui.h"

namespace AudioUI {
namespace {
using json = nlohmann::json;
struct Draft {
    const audio::AudioService* owner = nullptr;
    std::uint64_t generation = 0;
    std::string id, field;
    json clip, points;
    int point = -1;
};
Draft draft;
} // namespace

void commitAudioClipEnvelope(UIContext& ctx) {
    if (draft.point < 0 || draft.owner != ctx.scene.audio.get() ||
        draft.generation != ctx.scene.audio->generation()) {
        return;
    }
    const auto scene = ctx.scene.audio->snapshot();
    for (const auto& clip : scene.at("clips")) {
        if (clip.at("id").get<std::string>() == draft.id && clip == draft.clip &&
            draft.points != clip.at(draft.field)) {
            workspaceCall(ctx, "clip_update", {{"id", draft.id}, {draft.field, draft.points}});
            break;
        }
    }
    draft = {};
}

void drawAudioClipEnvelope(UIContext& ctx) {
    const auto view = ctx.scene.audio->invoke("workspace_get", json::object());
    auto mode = view.value("designer_envelope", std::string("none"));
    for (const char* value : {"none", "gain", "pan"}) {
        if (ImGui::RadioButton(value, mode == value)) {
            workspaceCall(ctx, "workspace_update", {{"designer_envelope", value}});
            mode = value;
        }
        if (std::string(value) != "pan") {
            ImGui::SameLine();
        }
    }
    if (mode == "none") {
        draft = {};
        return;
    }
    const auto scene = ctx.scene.audio->snapshot();
    const auto id = view.at("selected_clip").get<std::string>();
    const auto found = std::find_if(scene.at("clips").begin(), scene.at("clips").end(),
        [&](const json& clip) { return clip.at("id").get<std::string>() == id; });
    if (found == scene.at("clips").end()) {
        draft = {};
        ImGui::TextDisabled("Select a clip to draw its gain or pan automation.");
        return;
    }
    const auto field = mode == "gain" ? std::string("gain_curve") : std::string("pan_curve");
    if (draft.owner != ctx.scene.audio.get() || draft.generation != ctx.scene.audio->generation() ||
        draft.id != id || draft.field != field || (draft.point >= 0 && draft.clip != *found)) {
        draft = {};
    }
    if (draft.point < 0) {
        draft.owner = ctx.scene.audio.get();
        draft.generation = ctx.scene.audio->generation();
        draft.id = id;
        draft.field = field;
        draft.clip = *found;
        draft.points = found->at(field);
    }
    const double offset = found->at("envelope_offset_seconds");
    const double duration = found->at("duration_seconds");
    const double minimum = mode == "gain" ? 0 : -1;
    const double maximum = mode == "gain" ? 16 : 1;
    const float width = std::max(1.0f, ImGui::GetContentRegionAvail().x);
    ImGui::TextWrapped("%s | double-click to add, drag points, right-click to remove. "
                       "Commit on release; Escape cancels.", mode == "gain" ? "Gain -60..24 dB (linear multiplier stored)" : "Pan -1..1");
    ImGui::InvisibleButton("ClipEnvelope", ImVec2(width, 110));
    const auto first = ImGui::GetItemRectMin(), last = ImGui::GetItemRectMax();
    const auto mouse = ImGui::GetIO().MousePos;
    auto* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled(first, last, ImGui::GetColorU32(ImGuiCol_FrameBg), 4);
    const auto pointAt = [&](double time, double value) {
        const double vertical = mode == "gain" ?
            (20 * std::log10(std::max(0.001, value)) + 60) / 84.0823996531 :
            (value - minimum) / (maximum - minimum);
        return ImVec2(first.x + static_cast<float>((time - offset) / duration) * width,
            last.y - static_cast<float>(vertical) * 110);
    };
    draw->PushClipRect(first, last, true);
    int hovered = -1;
    ImVec2 previous;
    for (std::size_t index = 0; index < draft.points.size(); ++index) {
        const auto point = pointAt(draft.points[index][0], draft.points[index][1]);
        if (index == 0) {
            draw->AddLine(ImVec2(first.x, point.y), point, ImGui::GetColorU32(ImGuiCol_PlotLines), 2);
        }
        if (index > 0) {
            draw->AddLine(previous, point, ImGui::GetColorU32(ImGuiCol_PlotLines), 2);
        }
        draw->AddCircleFilled(point, 5, ImGui::GetColorU32(ImGuiCol_SliderGrabActive));
        const float dx = mouse.x - point.x, dy = mouse.y - point.y;
        if (dx * dx + dy * dy < 100) {
            hovered = static_cast<int>(index);
        }
        previous = point;
    }
    if (!draft.points.empty()) {
        draw->AddLine(previous, ImVec2(last.x, previous.y), ImGui::GetColorU32(ImGuiCol_PlotLines), 2);
    }
    if (draft.points.empty()) {
        const auto point = pointAt(offset, mode == "gain" ? 1 : 0);
        draw->AddLine(ImVec2(first.x, point.y), ImVec2(last.x, point.y),
            ImGui::GetColorU32(ImGuiCol_PlotLines), 2);
    }
    draw->PopClipRect();
    const double time = std::clamp(offset + (mouse.x - first.x) / width * duration,
        offset, offset + duration);
    const double normalized = std::clamp((last.y - mouse.y) / 110, 0.0f, 1.0f);
    const double db = -60 + normalized * 84.0823996531;
    const double value = mode == "gain" ? (db <= -60 ? 0 : std::min(16.0, std::pow(10.0, db / 20))) :
        minimum + normalized * (maximum - minimum);
    if (ImGui::IsItemHovered()) {
        if (hovered >= 0 && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            draft.point = hovered;
        } else if (hovered < 0 && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) &&
                   draft.points.size() < 4096) {
            const auto position = std::lower_bound(draft.points.begin(), draft.points.end(), time,
                [](const json& knot, double value) { return knot[0].get<double>() < value; });
            if (position == draft.points.end() || (*position)[0].get<double>() != time) {
                draft.points.insert(position, json::array({time, value}));
                workspaceCall(ctx, "clip_update", {{"id", id}, {field, draft.points}});
            }
        }
        if (hovered >= 0 && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
            draft.points.erase(draft.points.begin() + hovered);
            workspaceCall(ctx, "clip_update", {{"id", id}, {field, draft.points}});
            draft.point = -1;
        }
    }
    if (draft.point >= 0) {
        if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
            draft = {};
        } else if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            const auto index = static_cast<std::size_t>(draft.point);
            const double low = index == 0 ? offset : std::max(offset,
                std::nextafter(draft.points[index - 1][0].get<double>(),
                               std::numeric_limits<double>::infinity()));
            const double high = index + 1 == draft.points.size() ? offset + duration :
                std::min(offset + duration, std::nextafter(draft.points[index + 1][0].get<double>(),
                    -std::numeric_limits<double>::infinity()));
            if (low <= high) {
                draft.points[index] = json::array({std::clamp(time, low, high), value});
            }
        } else {
            if (draft.points != found->at(field)) {
                workspaceCall(ctx, "clip_update", {{"id", id}, {field, draft.points}});
            }
            draft.point = -1;
        }
    }
}
} // namespace AudioUI
