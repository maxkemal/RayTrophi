#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "UI/AudioEqEditorUI.h"
#include "UI/AudioDesignView.h"
#include "UI/AudioPropertyLayout.h"
#include "Audio/AudioService.h"
#include "scene_data.h"
#include "imgui.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <exception>
#include <utility>
#include "scene_ui.h"

namespace AudioUI {
namespace {
using json = nlohmann::json;
json band(double frequency, double gain = 0, const char* mode = "peak") {
    return {{"frequency_hz", frequency}, {"gain_db", gain}, {"q", 0.707},
        {"mode", mode}, {"enabled", true}};
}
ImVec4 bandColor(std::size_t index, float alpha = 1.0f) {
    const ImVec4 colors[] = {
        {0.30f, 0.78f, 1.0f, 1}, {0.43f, 0.88f, 0.60f, 1},
        {1.0f, 0.76f, 0.30f, 1}, {1.0f, 0.43f, 0.48f, 1},
        {0.73f, 0.53f, 1.0f, 1}, {0.30f, 0.88f, 0.83f, 1},
        {1.0f, 0.55f, 0.27f, 1}, {0.88f, 0.55f, 0.82f, 1}
    };
    auto color = colors[index % 8];
    color.w = alpha;
    return color;
}

// Reuse the canonical background analysis service. Never replace another job.
const json& spectrumForEq(UIContext& ctx, bool original, std::string& message) {
    static std::weak_ptr<audio::AudioService> owner;
    static std::uint64_t generation = 0;
    static std::uint64_t jobId = 0;
    static std::string requestedKey, appliedKey, failedKey, failure;
    static json spectrum = json::array();
    if (owner.lock() != ctx.scene.audio || generation != ctx.scene.audio->generation()) {
        owner = ctx.scene.audio;
        generation = ctx.scene.audio->generation();
        jobId = 0;
        requestedKey.clear();
        appliedKey.clear();
        failedKey.clear();
        failure.clear();
        spectrum = json::array();
    }
    std::string currentKey;
    try {
        const auto status = ctx.scene.audio->invoke("design_status", {{"details", false}});
        if (!status.value("ready", false)) {
            spectrum = json::array();
            message = "Render a sound to see its spectrum";
            return spectrum;
        }
        const auto key = std::to_string(status.at("revision").get<std::uint64_t>()) +
            (original ? ":original" : ":processed");
        currentKey = key;
        if (failedKey == key) {
            message = failure + " / render again to retry";
            return spectrum;
        }
        const auto job = ctx.scene.audio->invoke("design_job_status", json::object());
        const auto phase = job.value("state", std::string("idle"));
        if (jobId != 0 && job.value("id", std::uint64_t(0)) == jobId) {
            if (phase == "completed") {
                const auto result = ctx.scene.audio->invoke("design_job_apply", json::object());
                if (result.contains("analysis") && requestedKey == key) {
                    spectrum = result.at("analysis").at("spectrum");
                    appliedKey = key;
                }
                jobId = 0;
            } else if (phase != "queued" && phase != "running") {
                jobId = 0;
                if (phase == "failed") {
                    failedKey = requestedKey;
                    failure = job.value("error", std::string("Spectrum analysis failed"));
                    spectrum = json::array();
                    message = failure;
                    return spectrum;
                }
            }
        } else {
            jobId = 0;
        }
        if (appliedKey != key) {
            spectrum = json::array();
            message = "Analyzing last render...";
            if (jobId == 0 && (phase == "idle" || phase == "consumed" ||
                phase == "failed" || phase == "cancelled")) {
                const auto started = ctx.scene.audio->invoke("design_job_start", {
                    {"kind", "analysis"}, {"options", {{"original", original},
                    {"fft_size", 4096}, {"channel", "mix"}, {"points", 256},
                    {"time_bins", 16}}}});
                jobId = started.at("id");
                requestedKey = key;
            }
        } else {
            message = original ? "Original render spectrum / dBFS" :
                "Processed render spectrum / dBFS";
        }
    } catch (const std::exception& exception) {
        spectrum = json::array();
        failedKey = currentKey;
        failure = exception.what();
        message = failure;
    }
    return spectrum;
}

bool rotary(json& object, const char* key, const char* label, float low, float high,
            const char* format, const ImVec4& color, bool logarithmic = false) {
    ImGui::PushID(key);
    ImGui::BeginGroup();
    float value = object.value(key, key == std::string("q") ? 0.707f :
        key == std::string("frequency_hz") ? 1000.0f : 0.0f);
    const auto normalized = [&](float number) {
        return logarithmic ? std::log(number / low) / std::log(high / low) :
            (number - low) / (high - low);
    };
    const float scale = designPanelScale;
    ImGui::InvisibleButton("##dial", ImVec2(72 * scale, 60 * scale));
    const auto p = ImGui::GetItemRectMin();
    const bool hovered = ImGui::IsItemHovered();
    const bool active = ImGui::IsItemActive();
    bool completed = ImGui::IsItemDeactivatedAfterEdit();
    // InvisibleButton has no edited state: remember an actual drag until release.
    const auto changedId = ImGui::GetID("dragChanged");
    auto* storage = ImGui::GetStateStorage();
    if (ImGui::IsItemActivated()) {
        storage->SetBool(changedId, false);
    }
    if (active && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 1)) {
        const auto& io = ImGui::GetIO();
        const float delta = (io.MouseDelta.x - io.MouseDelta.y) *
            (io.KeyShift ? 0.00035f : 0.0035f);
        if (delta != 0) {
            const float amount = std::clamp(normalized(value) + delta, 0.0f, 1.0f);
            value = logarithmic ? low * std::pow(high / low, amount) :
                low + amount * (high - low);
            object[key] = value;
            storage->SetBool(changedId, true);
        }
    }
    if (ImGui::IsItemDeactivated()) {
        completed |= storage->GetBool(changedId, false);
        storage->SetBool(changedId, false);
    }
    if (hovered) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
        ImGui::SetTooltip("%s: drag to turn / Shift: fine / double-click: type value", label);
        if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
            ImGui::OpenPopup("Exact value");
        }
    }
    const ImVec2 center(p.x + 36 * scale, p.y + 29 * scale);
    constexpr float pi = 3.14159265359f;
    constexpr float begin = 0.75f * pi;
    constexpr float end = 2.25f * pi;
    auto* draw = ImGui::GetWindowDrawList();
    draw->AddCircleFilled(center, 19 * scale, ImGui::GetColorU32(ImGuiCol_FrameBg));
    draw->PathArcTo(center, 24 * scale, begin, end, 36);
    draw->PathStroke(ImGui::GetColorU32(ImGuiCol_Border), 0, 3);
    const float angle = begin + std::clamp(normalized(value), 0.0f, 1.0f) * (end - begin);
    draw->PathArcTo(center, 24 * scale, begin, angle, 36);
    draw->PathStroke(ImGui::GetColorU32(color), 0, active ? 4 : 3);
    draw->AddLine(center, ImVec2(center.x + std::cos(angle) * 16 * scale,
        center.y + std::sin(angle) * 16 * scale), ImGui::GetColorU32(color), 2);
    char text[40];
    std::snprintf(text, sizeof(text), format, value);
    ImGui::TextUnformatted(text);
    ImGui::TextDisabled("%s", label);
    if (ImGui::BeginPopup("Exact value")) {
        ImGui::SetNextItemWidth(150);
        if (ImGui::InputFloat("##number", &value, 0, 0, format,
                ImGuiInputTextFlags_EnterReturnsTrue)) {
            if (std::isfinite(value)) {
                object[key] = std::clamp(value, low, high);
                completed = true;
                ImGui::CloseCurrentPopup();
            }
        }
        ImGui::EndPopup();
    }
    ImGui::EndGroup();
    ImGui::PopID();
    return completed;
}
}

bool drawParametricEqEditor(UIContext& ctx, json& effect) {
    static std::weak_ptr<audio::AudioService> owner;
    static ImGuiID lastId = 0;
    const auto editorView = ctx.scene.audio->invoke("workspace_get", json::object());
    const int initialBand = editorView.value("designer_eq_band", 0);
    int selected = initialBand;
    static int dragging = -1;
    static bool dragChanged = false;
    static int contextBand = -1;
    float range = editorView.value("designer_eq_range_db", 24.0f);
    bool autoRange = editorView.value("designer_eq_auto_range", true);
    static json cachedEffect, visual;
    static std::string error;
    const auto id = ImGui::GetID("Parametric EQ editor");
    if (owner.lock() != ctx.scene.audio || lastId != id) {
        owner = ctx.scene.audio;
        lastId = id;
        dragging = -1;
        dragChanged = false;
        cachedEffect = nullptr;
    }
    if (!effect.contains("bands")) {
        effect["bands"] = json::array();
    }
    auto& bands = effect["bands"];
    selected = bands.empty() ? -1 : std::clamp(selected, 0, static_cast<int>(bands.size()) - 1);
    bool completed = false;
    const float toolbarHeight = ImGui::GetFrameHeightWithSpacing() +
        (ImGui::GetContentRegionAvail().x < 850 ? ImGui::GetStyle().ScrollbarSize : 0);
    ImGui::BeginChild("EQ toolbar", ImVec2(0, toolbarHeight), false,
        ImGuiWindowFlags_HorizontalScrollbar);
    bool enabled = effect.value("enabled", true);
    if (ImGui::Checkbox("EQ enabled", &enabled)) {
        effect["enabled"] = enabled;
        completed = true;
    }
    ImGui::SameLine();
    ImGui::BeginDisabled(bands.size() >= 8);
    if (ImGui::Button("Add band")) {
        bands.push_back(band(1000));
        selected = static_cast<int>(bands.size()) - 1;
        completed = true;
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("Tone setup...")) {
        ImGui::OpenPopup("Tone setup");
    }
    if (ImGui::BeginPopup("Tone setup")) {
        ImGui::TextWrapped("Replaces the current bands; output trim and mix are retained.");
        const auto catalogue = ctx.scene.audio->invoke("design_catalogue", json::object());
        for (const auto& preset : catalogue.at("eq_presets")) {
            if (ImGui::MenuItem(preset.at("name").get<std::string>().c_str())) {
                bands = preset.at("bands");
                selected = 0;
                completed = true;
            }
        }
        if (ImGui::MenuItem("Clear all bands")) {
            bands = json::array();
            selected = -1;
            completed = true;
        }
        ImGui::EndPopup();
    }
    if (cachedEffect != effect) {
        try {
            visual = ctx.scene.audio->invoke("design_effect_visual", {{"effect", effect}});
            cachedEffect = effect;
            error.clear();
        } catch (const std::exception& exception) {
            error = exception.what();
            visual = nullptr;
        }
    }
    if (visual.is_null()) {
        ImGui::EndChild();
        ImGui::TextWrapped("%s", error.c_str());
        return completed;
    }
    if (autoRange) {
        double extent = 6;
        for (const auto& point : visual.at("points")) {
            extent = std::max(extent, std::abs(point[1].get<double>()));
        }
        for (const auto& value : bands) {
            extent = std::max(extent, std::abs(value.value("gain_db", 0.0)));
        }
        range = std::clamp(static_cast<float>(std::ceil(extent * 1.15 / 6) * 6), 6.0f, 288.0f);
    }
    const auto setRange = [&](float value, bool fit) {
        range = std::clamp(value, 6.0f, 288.0f);
        autoRange = fit;
        ctx.scene.audio->invoke("workspace_update", {{"designer_eq_range_db", range},
            {"designer_eq_auto_range", fit}});
    };
    ImGui::SameLine();
    ImGui::SetNextItemWidth(105);
    float gainZoom = 288 / range;
    if (ImGui::SliderFloat("##GainZoom", &gainZoom, 1, 48, "Zoom %.1fx",
            ImGuiSliderFlags_Logarithmic)) {
        setRange(288 / gainZoom, false);
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Gain axis +/- %.0f dB / Shift+wheel over graph", range);
    }
    ImGui::SameLine();
    if (ImGui::Checkbox("Auto", &autoRange)) {
        setRange(range, autoRange);
    }
    ImGui::SameLine();
    float graphHeight = editorView.value("designer_eq_height", 280.0f);
    ImGui::SetNextItemWidth(75);
    if (ImGui::DragFloat("##GraphHeight", &graphHeight, 1, 180, 600, "H %.0f")) {
        ctx.scene.audio->invoke("workspace_update", {{"designer_eq_height", graphHeight}});
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Graph height / drag to resize");
    }
    static bool showSpectrum = true;
    static bool originalSpectrum = false;
    ImGui::SameLine();
    ImGui::Checkbox("Spectrum", &showSpectrum);
    ImGui::SameLine();
    ImGui::BeginDisabled(!showSpectrum);
    ImGui::SetNextItemWidth(75);
    int spectrumSource = originalSpectrum ? 1 : 0;
    if (ImGui::Combo("##SpectrumSource", &spectrumSource, "After\0Before\0")) {
        originalSpectrum = spectrumSource == 1;
    }
    ImGui::EndDisabled();
    ImGui::EndChild();
    std::string spectrumMessage;
    json spectrum = json::array();
    if (showSpectrum) {
        spectrum = spectrumForEq(ctx, originalSpectrum, spectrumMessage);
    }
    if (showSpectrum && spectrum.empty()) {
        ImGui::TextDisabled("%s", spectrumMessage.c_str());
    }
    // Band selectors share the handle colors; the graph footer holds only three dials.
    for (std::size_t index = 0; index < bands.size(); ++index) {
        if (index > 0 && ImGui::GetContentRegionAvail().x > 65) {
            ImGui::SameLine();
        }
        ImGui::PushID(static_cast<int>(index));
        ImGui::PushStyleColor(ImGuiCol_Text, bandColor(index));
        if (ImGui::Selectable(("Band " + std::to_string(index + 1)).c_str(),
                selected == static_cast<int>(index), 0, ImVec2(62, 0))) {
            selected = static_cast<int>(index);
        }
        ImGui::PopStyleColor();
        ImGui::PopID();
    }
    const float height = graphHeight * designPanelScale;
    const auto view = ctx.scene.audio->invoke("workspace_get", json::object());
    float zoom = view.value("designer_canvas_zoom", 1.0f);
    float pan = view.value("designer_canvas_pan_x", 0.0f);
    const double span = 1.0 / zoom;
    const double start = pan * (1 - span);
    const float width = std::max(120.0f, ImGui::GetContentRegionAvail().x);
    ImGui::InvisibleButton("EQ frequency canvas", ImVec2(width, height - 105 * designPanelScale),
        ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonMiddle);
    if (!ImGui::GetIO().KeyAlt) {
        ImGui::SetItemKeyOwner(ImGuiKey_MouseWheelY);
    }
    const auto outer = ImGui::GetItemRectMin();
    const ImVec2 first(outer.x + 42, outer.y + 12);
    const ImVec2 last(outer.x + width - 40, outer.y + height - 148 * designPanelScale);
    const float plotWidth = std::max(1.0f, last.x - first.x);
    const float plotHeight = std::max(1.0f, last.y - first.y);
    const bool hovered = ImGui::IsItemHovered();
    const bool active = ImGui::IsItemActive();
    const bool released = ImGui::IsItemDeactivated();
    const auto mouse = ImGui::GetIO().MousePos;
    auto* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled(outer, ImVec2(outer.x + width, outer.y + height),
        ImGui::GetColorU32(ImGuiCol_FrameBg), 6);
    const auto xAt = [&](double hz) {
        const double normalized = std::log(hz / 20) / std::log(1000.0);
        return first.x + static_cast<float>((normalized - start) / span) * plotWidth;
    };
    const auto hzAt = [&](float x) {
        return 20 * std::pow(1000.0, start + span *
            std::clamp((x - first.x) / plotWidth, 0.0f, 1.0f));
    };
    const auto yAt = [&](double db) {
        return first.y + static_cast<float>((range - db) / (2 * range)) * plotHeight;
    };
    for (const float hz : {20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000}) {
        if (width < 600 && (hz == 50 || hz == 200 || hz == 500 ||
            hz == 2000 || hz == 5000)) {
            continue;
        }
        const float x = xAt(hz);
        if (x < first.x - 0.5f || x > last.x + 0.5f) {
            continue;
        }
        draw->AddLine(ImVec2(x, first.y), ImVec2(x, last.y), ImGui::GetColorU32(ImGuiCol_Border));
        char label[16];
        std::snprintf(label, sizeof(label), hz >= 1000 ? "%.0fk" : "%.0f", hz >= 1000 ? hz / 1000 : hz);
        const float textWidth = ImGui::CalcTextSize(label).x;
        draw->AddText(ImVec2(std::clamp(x - textWidth / 2, outer.x, outer.x + width - textWidth),
            last.y + 5), ImGui::GetColorU32(ImGuiCol_TextDisabled), label);
    }
    for (const float db : {-range, -range / 2, 0.0f, range / 2, range}) {
        const float y = yAt(db);
        draw->AddLine(ImVec2(first.x, y), ImVec2(last.x, y),
            ImGui::GetColorU32(db == 0 ? ImGuiCol_TextDisabled : ImGuiCol_Border));
        char label[16];
        std::snprintf(label, sizeof(label), "%+.0f", db);
        draw->AddText(ImVec2(outer.x + 3, y - 6), ImGui::GetColorU32(ImGuiCol_TextDisabled), label);
    }
    // Spectrum uses its own -96..0 dBFS scale, independent from EQ gain.
    draw->PushClipRect(first, last, true);
    for (std::size_t index = 1; index < spectrum.size(); ++index) {
        const auto point = [&](std::size_t i) {
            const float db = spectrum[i][1].get<float>();
            return ImVec2(xAt(spectrum[i][0]), last.y -
                std::clamp((db + 96) / 96, 0.0f, 1.0f) * plotHeight);
        };
        const auto a = point(index - 1), b = point(index);
        const ImVec2 area[] = {ImVec2(a.x, last.y), a, b, ImVec2(b.x, last.y)};
        draw->AddConvexPolyFilled(area, 4, IM_COL32(105, 155, 175, 30));
        draw->AddLine(a, b, IM_COL32(130, 185, 205, 100));
    }
    draw->PopClipRect();
    if (showSpectrum) {
        for (const float db : {-96.0f, -48.0f, 0.0f}) {
            char label[16];
            std::snprintf(label, sizeof(label), "%.0f", db);
            draw->AddText(ImVec2(last.x + 5,
                last.y - (db + 96) / 96 * plotHeight - 6),
                ImGui::GetColorU32(ImGuiCol_TextDisabled), label);
        }
    }
    const char* regions[] = {"Bass", "Body", "Clarity", "Air"};
    const double regionHz[] = {65, 350, 2500, 12000};
    for (int index = 0; index < 4; ++index) {
        if (xAt(regionHz[index]) < first.x || xAt(regionHz[index]) > last.x) {
            continue;
        }
        const auto textSize = ImGui::CalcTextSize(regions[index]);
        draw->AddText(ImVec2(xAt(regionHz[index]) - textSize.x / 2, last.y + 23),
            ImGui::GetColorU32(ImGuiCol_TextDisabled), regions[index]);
    }
    const auto curve = [&](const json& points, ImU32 color, float thickness) {
        draw->PushClipRect(first, last, true);
        for (std::size_t index = 1; index < points.size(); ++index) {
            draw->AddLine(ImVec2(xAt(points[index - 1][0]), yAt(points[index - 1][1])),
                ImVec2(xAt(points[index][0]), yAt(points[index][1])), color, thickness);
        }
        draw->PopClipRect();
    };
    for (std::size_t index = 0; index < visual.at("band_curves").size(); ++index) {
        const auto& entry = visual.at("band_curves")[index];
        const bool bandActive = bands[index].value("enabled", true) && enabled;
        const auto& points = entry.at("points");
        const auto color = bandColor(index, bandActive ? 0.8f : 0.18f);
        if (bandActive) {
            draw->PushClipRect(first, last, true);
            for (std::size_t i = 1; i < points.size(); ++i) {
                const ImVec2 a(xAt(points[i - 1][0]), yAt(points[i - 1][1]));
                const ImVec2 b(xAt(points[i][0]), yAt(points[i][1]));
                // Two triangles remain valid even where the response crosses zero.
                const auto fill = ImGui::ColorConvertFloat4ToU32(
                    bandColor(index, index == selected ? 0.12f : 0.045f));
                draw->AddTriangleFilled(ImVec2(a.x, yAt(0)), a, b, fill);
                draw->AddTriangleFilled(ImVec2(a.x, yAt(0)), b,
                    ImVec2(b.x, yAt(0)), fill);
            }
            draw->PopClipRect();
        }
        curve(points, ImGui::ColorConvertFloat4ToU32(color), index == selected ? 2.0f : 1.0f);
    }
    curve(visual.at("points"), IM_COL32(230, 245, 240, 255), 2.5f);
    int hit = -1;
    for (std::size_t index = 0; index < bands.size(); ++index) {
        const auto& value = bands[index];
        const auto mode = value.value("mode", std::string("peak"));
        const float gain = mode == "lowpass" || mode == "highpass" ? 0 : value.value("gain_db", 0.0f);
        const ImVec2 point(xAt(value.value("frequency_hz", 1000.0)),
            std::clamp(yAt(gain), first.y, last.y));
        if (point.x < first.x || point.x > last.x) {
            continue;
        }
        const float dx = mouse.x - point.x, dy = mouse.y - point.y;
        if (hovered && dx * dx + dy * dy < 144) {
            hit = static_cast<int>(index);
        }
        const auto color = ImGui::ColorConvertFloat4ToU32(
            bandColor(index, value.value("enabled", true) && enabled ? 1.0f : 0.3f));
        draw->AddCircleFilled(point, index == selected ? 8 : 6, color);
        if (index == selected) {
            draw->AddCircle(point, 11, ImGui::GetColorU32(ImGuiCol_Text));
        }
        char number[8];
        std::snprintf(number, sizeof(number), "%u", static_cast<unsigned>(index + 1));
        draw->AddText(ImVec2(point.x + 11, point.y - 7), ImGui::GetColorU32(ImGuiCol_Text), number);
    }
    if (hit >= 0) {
        ImGui::SetTooltip("Band %d / %.0f Hz / %.1f dB / Q %.2f", hit + 1,
            bands[hit].value("frequency_hz", 1000.0), bands[hit].value("gain_db", 0.0),
            bands[hit].value("q", 0.707));
    }
    if (hovered && ImGui::IsMouseClicked(0)) {
        selected = hit >= 0 ? hit : selected;
        dragging = hit;
        dragChanged = false;
    }
    const auto& io = ImGui::GetIO();
    if (hovered && !io.KeyAlt && io.MouseWheel != 0) {
        if (hit >= 0 && io.KeyCtrl) {
            bands[hit]["q"] = std::clamp(bands[hit].value("q", 0.707) *
                std::pow(1.1, io.MouseWheel), 0.1, 10.0);
            selected = hit;
            completed = true;
        } else if (io.KeyShift) {
            setRange(range * std::pow(1.25f, -io.MouseWheel), false);
        } else {
            const double fraction = std::clamp((mouse.x - first.x) / plotWidth, 0.0f, 1.0f);
            const double anchor = start + fraction * span;
            zoom = std::clamp(zoom * std::pow(1.25f, io.MouseWheel), 1.0f, 8.0f);
            const double nextSpan = 1.0 / zoom;
            pan = nextSpan < 1 ? static_cast<float>(std::clamp(
                (anchor - fraction * nextSpan) / (1 - nextSpan), 0.0, 1.0)) : 0;
            ctx.scene.audio->invoke("workspace_update", {{"designer_canvas_zoom", zoom},
                {"designer_canvas_pan_x", pan}});
        }
    }
    if (active && io.KeyShift && !io.KeyAlt && ImGui::IsMouseDragging(ImGuiMouseButton_Middle)) {
        pan = span < 1 ? static_cast<float>(std::clamp(
            pan - io.MouseDelta.x / plotWidth * span / (1 - span), 0.0, 1.0)) : 0;
        ctx.scene.audio->invoke("workspace_update", {{"designer_canvas_pan_x", pan}});
    }
    if (hovered && !io.KeyAlt && !io.KeyCtrl &&
        ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Middle)) {
        ctx.scene.audio->invoke("workspace_update", {{"designer_canvas_zoom", 1},
            {"designer_canvas_pan_x", 0}, {"designer_canvas_pan_y", 0}});
    }
    if (hovered && hit < 0 && ImGui::IsMouseDoubleClicked(0) && bands.size() < 8) {
        bands.push_back(band(hzAt(mouse.x)));
        selected = static_cast<int>(bands.size()) - 1;
        completed = true;
    }
    if (hovered && hit >= 0 && ImGui::IsMouseClicked(1)) {
        contextBand = hit;
        selected = hit;
        ImGui::OpenPopup("EQ band actions");
    }
    if (ImGui::BeginPopup("EQ band actions")) {
        if (contextBand >= 0 && contextBand < static_cast<int>(bands.size())) {
            bool activeBand = bands[contextBand].value("enabled", true);
            if (ImGui::MenuItem("Band enabled", nullptr, &activeBand)) {
                bands[contextBand]["enabled"] = activeBand;
                completed = true;
            }
            const auto mode = bands[contextBand].value("mode", std::string("peak"));
            if (ImGui::BeginMenu("Filter shape")) {
                for (const auto& choice : {std::pair{"peak", "Bell"},
                    std::pair{"lowshelf", "Bass shelf"},
                    std::pair{"highshelf", "Treble shelf"},
                    std::pair{"highpass", "Low cut"},
                    std::pair{"lowpass", "High cut"}}) {
                    if (ImGui::MenuItem(choice.second, nullptr, mode == choice.first)) {
                        bands[contextBand]["mode"] = choice.first;
                        completed = true;
                    }
                }
                ImGui::EndMenu();
            }
            if (ImGui::MenuItem("Reset band gain", nullptr, false,
                mode != "highpass" && mode != "lowpass")) {
                bands[contextBand]["gain_db"] = 0;
                completed = true;
            }
            if (ImGui::MenuItem("Remove band")) {
                bands.erase(bands.begin() + contextBand);
                selected = -1;
                dragging = -1;
                completed = true;
            }
        }
        ImGui::EndPopup();
    }
    if (dragging >= 0 && dragging < static_cast<int>(bands.size()) && active &&
        ImGui::IsMouseDragging(0, 1.0f)) {
        dragChanged = true;
        bands[dragging]["frequency_hz"] = hzAt(mouse.x);
        const auto mode = bands[dragging].value("mode", std::string("peak"));
        if (mode != "highpass" && mode != "lowpass") {
            bands[dragging]["gain_db"] = std::clamp(range -
                (mouse.y - first.y) / plotHeight * 2 * range, -24.0f, 24.0f);
        }
    }
    if (released && dragging >= 0) {
        dragging = -1;
        completed |= dragChanged;
        dragChanged = false;
    }
    if (selected >= 0 && selected < static_cast<int>(bands.size())) {
        auto& value = bands[selected];
        const auto color = bandColor(selected);
        completed |= rotary(value, "frequency_hz", "Frequency", 20, 20000, "%.0f Hz", color, true);
        ImGui::SameLine();
        const auto mode = value.value("mode", std::string("peak"));
        ImGui::BeginDisabled(mode == "highpass" || mode == "lowpass");
        completed |= rotary(value, "gain_db", "Gain", -24, 24, "%+.1f dB", color);
        ImGui::EndDisabled();
        ImGui::SameLine();
        completed |= rotary(value, "q", "Width / Q", 0.1f, 10, "%.2f", color, true);
        if (ImGui::GetContentRegionAvail().x > 180) {
            ImGui::SameLine();
        }
        ImGui::BeginGroup();
        bool bandEnabled = value.value("enabled", true);
        if (ImGui::Checkbox("Band active", &bandEnabled)) {
            value["enabled"] = bandEnabled;
            completed = true;
        }
        ImGui::SetNextItemWidth(160);
        const char* shape = mode == "peak" ? "Bell" : mode == "lowshelf" ? "Bass shelf" :
            mode == "highshelf" ? "Treble shelf" : mode == "highpass" ? "Low cut" : "High cut";
        if (ImGui::BeginCombo("##BandShape", shape)) {
            for (const auto& choice : {std::pair{"peak", "Bell"},
                std::pair{"lowshelf", "Bass shelf"}, std::pair{"highshelf", "Treble shelf"},
                std::pair{"highpass", "Low cut"}, std::pair{"lowpass", "High cut"}}) {
                if (ImGui::Selectable(choice.second, mode == choice.first)) {
                    value["mode"] = choice.first;
                    completed = true;
                }
            }
            ImGui::EndCombo();
        }
        ImGui::TextDisabled("Precision sliders in Properties");
        ImGui::EndGroup();
    }
    ImGui::TextDisabled("White: total EQ / colors: bands / left: dB gain / right: dBFS");
    if (visual.value("estimated_max_boost_db", 0.0) > 6) {
        ImGui::TextColored(ImVec4(1, 0.75f, 0.3f, 1),
            "Boost up to %.1f dB: check rendered output headroom",
            visual.value("estimated_max_boost_db", 0.0));
    }
    if (selected >= 0 && selected != initialBand) {
        ctx.scene.audio->invoke("workspace_update", {{"designer_eq_band", selected}});
    }
    if (autoRange && range != editorView.value("designer_eq_range_db", 24.0f)) {
        ctx.scene.audio->invoke("workspace_update", {{"designer_eq_range_db", range}});
    }
    return completed;
}
}
