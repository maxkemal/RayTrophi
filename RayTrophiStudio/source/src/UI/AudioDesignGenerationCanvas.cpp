#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "UI/AudioDesignPanels.h"
#include "UI/AudioDesignView.h"
#include "UI/AudioWorkspaceUI.h"
#include "Audio/AudioService.h"
#include "scene_data.h"
#include "imgui.h"
#include <algorithm>
#include <cmath>
#include "scene_ui.h"
namespace AudioUI {
namespace {
using json = nlohmann::json;
constexpr double pi = 3.14159265358979323846;

// A one-period shape guide, never labelled as rendered PCM or spectrum.
float shape(const json& module, double phase) {
    const auto type = module.value("type", std::string("oscillator"));
    if (type == "fm") {
        return static_cast<float>(std::sin(2 * pi * phase + module.value("index", 1.0) *
            std::sin(2 * pi * phase * module.value("ratio", 2.0))));
    }
    if (type == "resonator") {
        return static_cast<float>(std::sin(2 * pi * phase * 5) * std::exp(-phase * 3));
    }
    if (type == "noise") {
        return static_cast<float>((std::sin(phase * 71) + std::sin(phase * 137)) * 0.4);
    }
    const auto wave = module.value("wave", std::string("sine"));
    if (wave == "saw") {
        return static_cast<float>(2 * phase - 1);
    }
    if (wave == "triangle") {
        return static_cast<float>(1 - 4 * std::abs(phase - 0.5));
    }
    if (wave == "square") {
        return phase < module.value("pulse_width", 0.5) ? 1 : -1;
    }
    return static_cast<float>(std::sin(2 * pi * phase));
}

bool pad(json& module, const json& specs, const char* horizontal, const char* vertical) {
    const auto& xSpec = specs.at(horizontal);
    const auto& ySpec = specs.at(vertical);
    const double xMin = xSpec.at("min"), xMax = xSpec.at("max");
    const double yMin = ySpec.at("min"), yMax = ySpec.at("max");
    const bool log = xSpec.at("logarithmic").get<bool>();
    const double x = module.value(horizontal, xSpec.at("default")).get<double>();
    const double y = module.value(vertical, ySpec.at("default")).get<double>();
    ImGui::TextWrapped("%s: %.2f %s | %s: %.2f %s", xSpec.at("label").get<std::string>().c_str(),
        x, xSpec.at("unit").get<std::string>().c_str(),
        ySpec.at("label").get<std::string>().c_str(), y,
        ySpec.at("unit").get<std::string>().c_str());
    const float width = std::max(1.0f, ImGui::GetContentRegionAvail().x);
    const float height = designViewPixels(105);
    ImGui::InvisibleButton("ParameterPad", ImVec2(width, height));
    const auto first = ImGui::GetItemRectMin(), last = ImGui::GetItemRectMax();
    auto* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled(first, last, ImGui::GetColorU32(ImGuiCol_FrameBg), 4);
    for (int line = 1; line < 4; ++line) {
        const float t = line / 4.0f;
        draw->AddLine(ImVec2(first.x + width * t, first.y),
            ImVec2(first.x + width * t, last.y), ImGui::GetColorU32(ImGuiCol_Border));
        draw->AddLine(ImVec2(first.x, first.y + height * t),
            ImVec2(last.x, first.y + height * t), ImGui::GetColorU32(ImGuiCol_Border));
    }
    const double nx = log ? std::log(x / xMin) / std::log(xMax / xMin) :
                           (x - xMin) / (xMax - xMin);
    const double ny = (y - yMin) / (yMax - yMin);
    draw->AddCircleFilled(ImVec2(first.x + static_cast<float>(std::clamp(nx, 0.0, 1.0)) * width,
        last.y - static_cast<float>(std::clamp(ny, 0.0, 1.0)) * height), 6,
        ImGui::GetColorU32(ImGuiCol_SliderGrabActive));
    if (ImGui::IsItemActive()) {
        const auto mouse = ImGui::GetIO().MousePos;
        const double px = std::clamp((mouse.x - first.x) / width, 0.0f, 1.0f);
        const double py = std::clamp((last.y - mouse.y) / height, 0.0f, 1.0f);
        module[horizontal] = log ? xMin * std::pow(xMax / xMin, px) : xMin + px * (xMax - xMin);
        module[vertical] = yMin + py * (yMax - yMin);
    }
    return ImGui::IsItemDeactivated();
}
} // namespace

bool drawDesignGenerationCanvas(UIContext& ctx, json& recipe) {
    if (recipe.value("mode", "synth") != "synth") {
        return false;
    }
    const auto catalogue = ctx.scene.audio->invoke("design_catalogue", json::object());
    const auto workspace = ctx.scene.audio->invoke("workspace_get", json::object());
    if (!recipe.contains("layers") || !recipe.at("layers").is_array() || recipe.at("layers").empty()) {
        ImGui::TextWrapped("Character engine active. Add production layers in Tools for visual module editing.");
        bool completed = false;
        if (ImGui::BeginCombo("Sound recipe", "Choose layered sound...")) {
            for (const auto& preset : catalogue.at("presets")) {
                if (ImGui::Selectable(preset.at("name").get<std::string>().c_str())) {
                    recipe = preset.at("recipe");
                    completed = true;
                }
            }
            ImGui::EndCombo();
        }
        return completed;
    }
    auto& layers = recipe["layers"];
    const auto index = std::min<std::size_t>(workspace.value("designer_layer", 0u), layers.size() - 1);
    auto& layer = layers[index];
    const auto type = layer.at("type").get<std::string>();
    const auto& definitions = catalogue.at("layers");
    const auto definition = std::find_if(definitions.begin(), definitions.end(), [&](const json& item) {
        return item.at("type").get<std::string>() == type;
    });
    if (definition == definitions.end()) {
        return false;
    }
    const auto& specs = definition->at("fields");
    ImGui::Text("Layer %zu / %zu: %s", index + 1, layers.size(),
        definition->at("name").get<std::string>().c_str());
    bool completed = false;
    if (type != "asset") {
        ImGui::TextDisabled("Shape guide; processed waveform below shows the actual sound");
        const float width = std::max(1.0f, ImGui::GetContentRegionAvail().x);
        ImGui::InvisibleButton("LayerShapeGuide", ImVec2(width, designViewPixels(55)));
        const auto first = ImGui::GetItemRectMin();
        auto* draw = ImGui::GetWindowDrawList();
        ImVec2 previous(first.x, first.y + designViewPixels(27.5f - shape(layer, 0) * 24));
        for (int point = 1; point <= 192; ++point) {
            const double phase = point / 192.0;
            const ImVec2 next(first.x + static_cast<float>(phase) * width,
                first.y + designViewPixels(27.5f - shape(layer, phase) * 24));
            draw->AddLine(previous, next, ImGui::GetColorU32(ImGuiCol_PlotLines), 2);
            previous = next;
        }
        const char* x = type == "noise" ? "cutoff_hz" : "frequency_hz";
        const char* y = type == "fm" ? "index" : type == "resonator" ? "damping" : "gain_db";
        ImGui::PushID("Tone");
        completed |= pad(layer, specs, x, y);
        ImGui::PopID();
    }
    ImGui::PushID("Placement");
    auto placement = specs;
    placement["start_seconds"]["max"] = std::max(0.001,
        recipe.value("duration_seconds", 2.0) - 1.0 / 48000);
    completed |= pad(layer, placement, "start_seconds", "pan");
    ImGui::PopID();
    ImGui::TextWrapped("Drag the points. Tools selects a layer; Properties exposes its remaining parameters.");
    return completed;
}

bool drawDesignEffectPad(UIContext& ctx, json& recipe) {
    if (!recipe.contains("effects") || !recipe.at("effects").is_array() ||
        recipe.at("effects").empty()) {
        return false;
    }
    const auto view = ctx.scene.audio->invoke("workspace_get", json::object());
    auto& effects = recipe["effects"];
    const auto index = std::min<std::size_t>(view.value("designer_effect", 0u), effects.size() - 1);
    auto& effect = effects[index];
    const auto type = effect.at("type").get<std::string>();
    const char* x = nullptr;
    const char* y = nullptr;
    if (type == "delay") {
        x = "time_seconds";
        y = "feedback";
    } else if (type == "reverb") {
        x = "size";
        y = "decay_seconds";
    } else if (type == "chorus" || type == "phaser") {
        x = "rate_hz";
        y = "depth";
    } else if (type == "saturation") {
        x = "drive";
        y = "mix";
    } else if (type == "compressor") {
        x = "ratio";
        y = "knee_db";
    } else {
        return false;
    }
    const auto catalogue = ctx.scene.audio->invoke("design_catalogue", json::object());
    for (const auto& definition : catalogue.at("effects")) {
        if (definition.at("type").get<std::string>() == type) {
            ImGui::PushID("EffectPad");
            const bool completed = pad(effect, definition.at("fields"), x, y);
            ImGui::PopID();
            return completed;
        }
    }
    return false;
}
} // namespace AudioUI
