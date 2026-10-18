#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "UI/AudioDesignPanels.h"
#include "UI/AudioEqEditorUI.h"
#include "UI/AudioDesignView.h"
#include "UI/AudioWorkspaceUI.h"
#include "UI/AudioPropertyLayout.h"
#include "Audio/AudioService.h"
#include "scene_data.h"
#include "imgui.h"
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <exception>
#include "scene_ui.h"
namespace AudioUI {
namespace {
using json = nlohmann::json;
std::string error;
json catalogue;

json read(UIContext& ctx, const char* method, const json& params = json::object()) {
    try {
        auto value = ctx.scene.audio->invoke(method, params);
        error.clear();
        return value;
    } catch (const std::exception& exception) {
        error = exception.what();
        return json::object();
    }
}

const json& definitions(UIContext& ctx) {
    if (catalogue.empty()) {
        catalogue = read(ctx, "design_catalogue");
    }
    return catalogue;
}

bool fields(json& object, const json& specs) {
    bool completed = false;
    for (auto spec = specs.begin(); spec != specs.end(); ++spec) {
        const auto key = spec.key();
        const auto& definition = spec.value();
        const auto label = definition.at("label").get<std::string>();
        if (definition.at("kind").get<std::string>() == "choice") {
            const auto value = object.value(key, definition.at("default")).get<std::string>();
            if (ImGui::BeginCombo(property(label.c_str()).c_str(), value.c_str())) {
                for (const auto& entry : definition.at("choices")) {
                    const auto choice = entry.get<std::string>();
                    if (ImGui::Selectable(choice.c_str(), choice == value)) {
                        object[key] = choice;
                        completed = true;
                    }
                }
                ImGui::EndCombo();
            }
        } else {
            float value = object.value(key, definition.at("default")).get<float>();
            const auto unit = definition.at("unit").get<std::string>();
            const auto title = label + (unit.empty() ? "" : " (" + unit + ")");
            const auto mode = object.value("mode", std::string());
            ImGui::BeginDisabled(key == "gain_db" && (mode == "lowpass" || mode == "highpass"));
            if (ImGui::SliderFloat(property(title.c_str()).c_str(), &value,
                definition.at("min").get<float>(), definition.at("max").get<float>(), "%.3f",
                definition.at("logarithmic").get<bool>() ? ImGuiSliderFlags_Logarithmic : 0)) {
                object[key] = value;
            }
            completed |= ImGui::IsItemDeactivatedAfterEdit();
            ImGui::EndDisabled();
        }
    }
    return completed;
}

void grid(const ImVec2& first, const ImVec2& last) {
    auto* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled(first, last, ImGui::GetColorU32(ImGuiCol_FrameBg), 5);
    for (int step = 1; step < 6; ++step) {
        const float x = first.x + (last.x - first.x) * step / 6;
        const float y = first.y + (last.y - first.y) * step / 6;
        draw->AddLine(ImVec2(x, first.y), ImVec2(x, last.y), ImGui::GetColorU32(ImGuiCol_Border));
        draw->AddLine(ImVec2(first.x, y), ImVec2(last.x, y), ImGui::GetColorU32(ImGuiCol_Border));
    }
}

void linePlot(const json& points, const ImVec2& first, const ImVec2& last,
              double minX, double maxX, double minY, double maxY, bool logX) {
    auto* draw = ImGui::GetWindowDrawList();
    const auto position = [&](const json& value) {
        const double x = logX ? std::log(value[0].get<double>() / minX) / std::log(maxX / minX) :
                               (value[0].get<double>() - minX) / (maxX - minX);
        const double y = (value[1].get<double>() - minY) / (maxY - minY);
        return ImVec2(first.x + static_cast<float>(std::clamp(x, 0.0, 1.0)) * (last.x - first.x),
                      last.y - static_cast<float>(std::clamp(y, 0.0, 1.0)) * (last.y - first.y));
    };
    for (std::size_t point = 1; point < points.size(); ++point) {
        draw->AddLine(position(points[point - 1]), position(points[point]),
                      ImGui::GetColorU32(ImGuiCol_PlotLines), 2);
    }
}

struct SourceView {
    const audio::AudioService* owner = nullptr;
    std::uint64_t generation = 0;
    std::string key;
    json waveform;
    int handle = -1;
    double initialStart = 0, initialEnd = 0, mouseStart = 0;
};
SourceView source;

struct AnalysisView {
    std::uint64_t job = 0;
    std::string requestedKey;
    const audio::AudioService* owner = nullptr;
    std::uint64_t generation = 0;
    std::string key;
    json report;
    int fftSize = 2048;
    std::string channel = "mix";
    float floorDb = -100;
    std::uint64_t selectionRevision = 0;
    bool selecting = false;
    bool selectionReady = false;
    ImVec2 selectionStart{}, selectionEnd{};
    float regionGainDb = -24;
    float feather = 0.25f;
    int shape = 0;
};
AnalysisView analyzer;
} // namespace

const std::string& designPanelsError() {
    return error;
}

bool drawDesignModules(UIContext& ctx, json& recipe, bool effects, int layout) {
    const auto& library = definitions(ctx);
    if (library.empty()) {
        return false;
    }
    const char* key = effects ? "effects" : "layers";
    if (!recipe.contains(key) || !recipe.at(key).is_array()) {
        recipe[key] = json::array();
    }
    auto& modules = recipe[key];
    bool completed = false;
    if (layout != 2) {
        ImGui::BeginDisabled(modules.size() >= 8);
        if (ImGui::BeginCombo(effects ? "Add effect" : "Add layer", "Choose module...")) {
            for (const auto& definition : library.at(key)) {
                const auto type = definition.at("type").get<std::string>();
                const auto scene = ctx.scene.audio->snapshot();
                const bool sample = definition.value("requires_asset", false);
                ImGui::BeginDisabled(sample && scene.at("assets").empty());
                if (ImGui::Selectable(definition.at("name").get<std::string>().c_str())) {
                    json module = {{"type", type}, {"enabled", true}};
                    for (auto field = definition.at("fields").begin();
                         field != definition.at("fields").end(); ++field) {
                        module[field.key()] = field.value().at("default");
                    }
                    if (sample) {
                        module["asset_id"] = scene.at("assets")[0].at("id");
                        module["source_out_seconds"] = std::min(1.0,
                            scene.at("assets")[0].at("duration").get<double>());
                    }
                    if (type == "eq") {
                        module["bands"] = json::array({{{"frequency_hz", 1000}, {"gain_db", 0},
                                                        {"q", 0.707}, {"mode", "peak"}, {"enabled", true}}});
                    }
                    modules.push_back(module);
                    workspaceCall(ctx, "workspace_update", {{effects ? "designer_effect" :
                        "designer_layer", modules.size() - 1}});
                    completed = true;
                }
                ImGui::EndDisabled();
            }
            ImGui::EndCombo();
        }
        ImGui::EndDisabled();
    }
    const char* selectionKey = effects ? "designer_effect" : "designer_layer";
    const auto view = ctx.scene.audio->invoke("workspace_get", json::object());
    const auto selected = modules.empty() ? 0 : std::min<std::size_t>(
        view.value(selectionKey, 0u), modules.size() - 1);
    if (effects && layout == 1 && !modules.empty()) {
        ImGui::TextDisabled("Signal chain / top to bottom");
        ImGui::BeginDisabled(selected == 0);
        if (ImGui::SmallButton("Move selected up")) {
            std::swap(modules[selected], modules[selected - 1]);
            workspaceCall(ctx, "workspace_update", {{selectionKey, selected - 1}});
            ImGui::EndDisabled();
            return true;
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::BeginDisabled(selected + 1 >= modules.size());
        if (ImGui::SmallButton("Move selected down")) {
            std::swap(modules[selected], modules[selected + 1]);
            workspaceCall(ctx, "workspace_update", {{selectionKey, selected + 1}});
            ImGui::EndDisabled();
            return true;
        }
        ImGui::EndDisabled();
    }
    if (!effects && modules.empty()) {
        ImGui::TextWrapped("No explicit layers: the character generator is active. Adding a layer "
                           "switches to the layered engine, with the same global envelope.");
    }
    for (std::size_t index = 0; index < modules.size(); ++index) {
        if (layout == 2 && index != selected) {
            continue;
        }
        ImGui::PushID(static_cast<int>(index));
        auto& module = modules[index];
        const auto type = module.at("type").get<std::string>();
        const auto& entries = library.at(key);
        auto definition = std::find_if(entries.begin(), entries.end(), [&](const json& value) {
            return value.at("type").get<std::string>() == type;
        });
        if (definition == entries.end()) {
            ImGui::PopID();
            continue;
        }
        bool enabled = module.value("enabled", true);
        if (ImGui::Checkbox("Enabled", &enabled)) {
            module["enabled"] = enabled;
            completed = true;
        }
        ImGui::SameLine();
        const auto title = std::to_string(index + 1) + ". " +
            definition->at("name").get<std::string>() +
            (effects && !module.value("enabled", true) ? " / Bypassed" : "");
        if (ImGui::Selectable(title.c_str(), index == selected)) {
            workspaceCall(ctx, "workspace_update", {{selectionKey, index}});
        }
        if (layout == 1 && ImGui::BeginPopupContextItem("ModuleActions")) {
            bool changed = false;
            if (ImGui::MenuItem("Move up", nullptr, false, index > 0)) {
                std::swap(modules[index], modules[index - 1]);
                workspaceCall(ctx, "workspace_update", {{selectionKey, index - 1}});
                changed = true;
            }
            if (!changed && ImGui::MenuItem("Move down", nullptr, false, index + 1 < modules.size())) {
                std::swap(modules[index], modules[index + 1]);
                workspaceCall(ctx, "workspace_update", {{selectionKey, index + 1}});
                changed = true;
            }
            if (!changed && ImGui::MenuItem("Remove")) {
                modules.erase(modules.begin() + index);
                changed = true;
            }
            ImGui::EndPopup();
            if (changed) {
                ImGui::PopID();
                return true;
            }
        }
        if (layout != 1) {
            if (effects && ImGui::SmallButton(type == "eq" ? "Open visual equalizer" : "Visual controls")) {
                workspaceCall(ctx, "workspace_update", {{"designer_effect", index}, {"designer_page", "effects"}});
            }
            if (ImGui::SmallButton("Up") && index > 0) {
                std::swap(modules[index], modules[index - 1]);
                workspaceCall(ctx, "workspace_update", {{selectionKey, index - 1}});
                ImGui::PopID();
                return true;
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("Down") && index + 1 < modules.size()) {
                std::swap(modules[index], modules[index + 1]);
                workspaceCall(ctx, "workspace_update", {{selectionKey, index + 1}});
                ImGui::PopID();
                return true;
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("Remove")) {
                modules.erase(modules.begin() + index);
                ImGui::PopID();
                return true;
            }
        }
        const bool showParameters = type == "eq" ?
            (layout != 1 && ImGui::CollapsingHeader("EQ precision parameters")) :
            (layout == 2 || (layout == 0 && ImGui::CollapsingHeader("Parameters")));
        if (showParameters) {
            completed |= fields(module, definition->at("fields"));
            if (type == "asset") {
                const auto scene = ctx.scene.audio->snapshot();
                const auto id = module.value("asset_id", std::string());
                if (ImGui::BeginCombo(property("Sample asset").c_str(), id.c_str())) {
                    for (const auto& asset : scene.at("assets")) {
                        if (ImGui::Selectable(asset.at("name").get<std::string>().c_str(),
                            asset.at("id").get<std::string>() == id)) {
                            module["asset_id"] = asset.at("id");
                            module["source_in_seconds"] = 0;
                            module["source_out_seconds"] = std::min(1.0, asset.at("duration").get<double>());
                            completed = true;
                        }
                    }
                    ImGui::EndCombo();
                }
            }
            if (type == "eq") {
                if (!module.contains("bands")) {
                    module["bands"] = json::array();
                }
                auto& bands = module["bands"];
                ImGui::BeginDisabled(bands.size() >= 8);
                if (ImGui::Button("Add EQ band")) {
                    bands.push_back({{"frequency_hz", 1000}, {"gain_db", 0}, {"q", 0.707},
                                      {"mode", "peak"}, {"enabled", true}});
                    completed = true;
                }
                ImGui::EndDisabled();
                for (std::size_t band = 0; band < bands.size(); ++band) {
                    ImGui::PushID(static_cast<int>(band));
                    const bool selectedBand = view.value("designer_eq_band", 0u) == band;
                    const bool bandOpen = ImGui::TreeNodeEx("Band",
                        selectedBand ? ImGuiTreeNodeFlags_Selected : ImGuiTreeNodeFlags_None,
                        "Band %zu", band + 1);
                    const bool bandClicked = ImGui::IsItemClicked();
                    if (bandClicked) {
                        workspaceCall(ctx, "workspace_update", {{"designer_eq_band", band}});
                    }
                    if (bandOpen) {
                        const auto beforeBand = bands[band];
                        bool active = bands[band].value("enabled", true);
                        if (ImGui::Checkbox("Band enabled", &active)) {
                            bands[band]["enabled"] = active;
                            completed = true;
                        }
                        completed |= fields(bands[band], library.at("eq_band_fields"));
                        if (bands[band] != beforeBand) {
                            workspaceCall(ctx, "workspace_update", {{"designer_eq_band", band}});
                        }
                        if (ImGui::Button("Remove band")) {
                            bands.erase(bands.begin() + band);
                            ImGui::TreePop();
                            ImGui::PopID();
                            ImGui::PopID();
                            return true;
                        }
                        ImGui::TreePop();
                    }
                    ImGui::PopID();
                }
            }
        }
        ImGui::Separator();
        ImGui::PopID();
    }
    return completed;
}

bool drawDesignSourceEditor(UIContext& ctx, json& recipe) {
    if (recipe.value("mode", "synth") != "asset") {
        ImGui::TextWrapped("Choose an imported audio asset to select its range visually. "
                           "Synthetic output uses its generated duration.");
        return false;
    }
    if (source.owner != ctx.scene.audio.get() || source.generation != ctx.scene.audio->generation()) {
        source = {};
        source.owner = ctx.scene.audio.get();
        source.generation = ctx.scene.audio->generation();
    }
    const auto assets = ctx.scene.audio->snapshot().at("assets");
    const auto id = recipe.at("asset_id").get<std::string>();
    auto asset = std::find_if(assets.begin(), assets.end(), [&](const json& entry) {
        return entry.at("id").get<std::string>() == id;
    });
    if (asset == assets.end()) {
        ImGui::TextWrapped("The selected source asset is unavailable.");
        return false;
    }
    const double duration = asset->at("duration");
    if (duration < 0.05) {
        ImGui::TextWrapped("Designer source editing needs at least 0.05 seconds of audio.");
        return false;
    }
    auto view = ctx.scene.audio->invoke("workspace_get", json::object());
    const float zoom = view.value("designer_zoom", 1.0f);
    const float pan = view.value("designer_pan", 0.0f);
    const double span = std::min(duration, std::max(16.0 / 48000, duration / zoom));
    const double begin = (duration - span) * pan, finish = begin + span;
    const json params = {{"id", id}, {"start_seconds", begin}, {"end_seconds", finish},
        {"points", 512}, {"background", true}};
    const auto key = params.dump() + asset->at("path").get<std::string>();
    const auto data = read(ctx, "design_waveform", params);
    if (!data.contains("points")) {
        source.waveform = json::object();
        if (data.value("state", "idle") == "failed") {
            ImGui::TextWrapped("%s", data.value("error", "Source waveform failed").c_str());
        } else {
            ImGui::TextWrapped("%s", error.empty() ? "Loading source waveform..." :
                               error.c_str());
        }
        return false;
    }
    source.waveform = data;
    source.key = key;
    if (source.waveform.empty()) {
        ImGui::TextWrapped("%s", error.c_str());
        return false;
    }
    const double selectedStart = recipe.value("source_in_seconds", 0.0);
    const double selectedEnd = recipe.value("source_out_seconds", std::min(duration, 10.0));
    if (selectedStart < 0 || selectedEnd > duration + 1e-6 ||
        selectedEnd < selectedStart + 0.05 || selectedEnd - selectedStart > 30) {
        ImGui::TextWrapped("Adjust source start/end to a valid 0.05..30 second range in Controls.");
        return false;
    }
    ImGui::Text("Source %.3f - %.3f s | selection %.3f - %.3f s", begin, finish,
                selectedStart, selectedEnd);
    ImGui::TextWrapped("Drag range edges or its center. Wheel: canvas zoom. Middle drag: canvas pan. "
                       "Ctrl+wheel/middle drag: source range view. Source file stays intact.");
    const float width = std::max(1.0f, ImGui::GetContentRegionAvail().x);
    const float height = designViewPixels(150);
    ImGui::InvisibleButton("SourceRangeCanvas", ImVec2(width, height),
        ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonMiddle);
    const auto first = ImGui::GetItemRectMin(), last = ImGui::GetItemRectMax();
    grid(first, last);
    auto* draw = ImGui::GetWindowDrawList();
    const auto xOf = [&](double time) {
        return first.x + static_cast<float>(std::clamp((time - begin) / span, 0.0, 1.0)) * width;
    };
    const auto& bins = source.waveform.at("points");
    const unsigned channels = source.waveform.at("channels");
    for (unsigned channel = 0; channel < channels; ++channel) {
        const float channelHeight = height / channels;
        const float middle = first.y + channelHeight * (channel + 0.5f);
        draw->AddLine(ImVec2(first.x, middle), ImVec2(last.x, middle), ImGui::GetColorU32(ImGuiCol_Border));
        for (std::size_t bin = 0; bin < bins.size(); ++bin) {
            const float x = first.x + (bin + 0.5f) / bins.size() * width;
            const float low = std::clamp(bins[bin].at("min")[channel].get<float>(), -1.0f, 1.0f);
            const float high = std::clamp(bins[bin].at("max")[channel].get<float>(), -1.0f, 1.0f);
            draw->AddLine(ImVec2(x, middle - high * channelHeight * 0.44f),
                ImVec2(x, middle - low * channelHeight * 0.44f),
                ImGui::GetColorU32(ImGuiCol_PlotHistogram), std::max(1.0f, width / bins.size()));
        }
    }
    const float left = xOf(selectedStart), right = xOf(selectedEnd);
    if (selectedEnd >= begin && selectedStart <= finish) {
        draw->AddRectFilled(ImVec2(left, first.y), ImVec2(right, last.y),
                            ImGui::GetColorU32(ImGuiCol_Header, 0.35f));
    }
    for (const float edge : {left, right}) {
        draw->AddLine(ImVec2(edge, first.y), ImVec2(edge, last.y),
                      ImGui::GetColorU32(ImGuiCol_SliderGrabActive), 2);
        draw->AddRectFilled(ImVec2(edge - 4, first.y), ImVec2(edge + 4, first.y + 14),
                            ImGui::GetColorU32(ImGuiCol_SliderGrabActive));
    }
    const auto mouse = ImGui::GetIO().MousePos;
    const double time = std::clamp(begin + (mouse.x - first.x) / width * span, 0.0, duration);
    if (ImGui::IsItemActive() && ImGui::GetIO().KeyCtrl && ImGui::IsMouseDragging(ImGuiMouseButton_Middle) &&
        duration > span) {
        workspaceCall(ctx, "workspace_update", {{"designer_pan",
            std::clamp(pan - ImGui::GetIO().MouseDelta.x / width * span / (duration - span),
                       0.0, 1.0)}});
    }
    if (ImGui::IsItemHovered()) {
        if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
            workspaceCall(ctx, "workspace_update", {{"designer_zoom", 1}, {"designer_pan", 0}});
        }
        ImGui::SetTooltip("%.6f s | frame %.0f", time, time * 48000);
        if (ImGui::GetIO().KeyCtrl && ImGui::GetIO().MouseWheel != 0) {
            const double nextZoom = std::clamp(zoom * std::pow(1.5, ImGui::GetIO().MouseWheel), 1.0, 2048.0);
            const double nextSpan = std::min(duration, std::max(16.0 / 48000, duration / nextZoom));
            const double nextBegin = time - (mouse.x - first.x) / width * nextSpan;
            workspaceCall(ctx, "workspace_update", {{"designer_zoom", nextZoom},
                {"designer_pan", duration > nextSpan ?
                    std::clamp(nextBegin / (duration - nextSpan), 0.0, 1.0) : 0}});
        }
        if (ImGui::IsMouseClicked(0)) {
            source.handle = std::abs(mouse.x - left) < 12 ? 0 :
                std::abs(mouse.x - right) < 12 ? 1 : mouse.x > left && mouse.x < right ? 2 : -1;
            source.initialStart = selectedStart;
            source.initialEnd = selectedEnd;
            source.mouseStart = time;
        }
    }
    if (source.handle >= 0 && ImGui::IsItemActive() && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        double start = source.initialStart, end = source.initialEnd;
        const double position = std::round(time * 48000) / 48000;
        if (source.handle == 0) {
            start = std::clamp(position, std::max(0.0, end - 30), end - 0.05);
        } else if (source.handle == 1) {
            end = std::clamp(position, start + 0.05, std::min(duration, start + 30));
        } else {
            const double length = end - start;
            start = std::clamp(start + position - source.mouseStart, 0.0, duration - length);
            end = start + length;
        }
        recipe["source_in_seconds"] = start;
        recipe["source_out_seconds"] = end;
    }
    if (source.handle >= 0 && ImGui::IsItemDeactivated()) {
        source.handle = -1;
        return true;
    }
    return false;
}

bool drawDesignEffectGraph(UIContext& ctx, json& recipe) {
    if (!recipe.contains("effects") || !recipe.at("effects").is_array() ||
        recipe.at("effects").empty()) {
        ImGui::TextWrapped("Add an effect in the inspector. The rack order is the processing order.");
        return false;
    }
    auto& effects = recipe["effects"];
    auto view = ctx.scene.audio->invoke("workspace_get", json::object());
    const auto selected = std::min<std::size_t>(view.value("designer_effect", 0u), effects.size() - 1);
    auto& effect = effects[selected];
    if (effect.at("type") == "eq") {
        ImGui::PushID(static_cast<int>(selected));
        const bool completed = drawParametricEqEditor(ctx, effect);
        ImGui::PopID();
        return completed;
    }
    static json visualEffect;
    static json visual;
    if (visualEffect != effect || visual.empty()) {
        visual = read(ctx, "design_effect_visual", {{"effect", effect}});
        visualEffect = effect;
    }
    if (!visual.contains("points")) {
        ImGui::TextWrapped("%s", error.c_str());
        return false;
    }
    const auto type = effect.at("type").get<std::string>();
    const auto kind = visual.at("kind").get<std::string>();
    if (kind == "spectral_region") {
        ImGui::TextWrapped("Spectral mask: %.3f - %.3f s | %.0f - %.0f Hz. "
                           "Edit bounds in Properties or select another region in Analyze. "
                           "2048-frame overlap-add reconstruction; edges have finite resolution.",
            effect.value("start_seconds", 0.0), effect.value("end_seconds", 1.0),
            effect.value("low_hz", 200.0), effect.value("high_hz", 2000.0));
        return false;
    }
    ImGui::Text("%zu. %s | %s", selected + 1, type.c_str(),
                effect.value("enabled", true) ? "Enabled" : "Bypassed");
    if (kind == "eq") {
        ImGui::TextWrapped("20 Hz - 20 kHz | -24 to +24 dB. Drag bands; Ctrl+wheel adjusts Q. "
                           "Double-click empty graph: add band. Right-click band: remove.");
    } else if (kind == "filter") {
        ImGui::TextWrapped("Small-signal filter response | 20 Hz - 20 kHz | -60 to +12 dB. "
                           "Drag cutoff horizontally and resonance vertically.");
    } else if (kind == "transfer") {
        ImGui::TextWrapped("Steady-state transfer | input/output dB. Drag threshold or ceiling; "
                           "attack and release apply during rendering.");
    } else if (kind == "tail") {
        ImGui::TextWrapped("Feedback/decay guide | actual tail is retained by the Tail setting.");
    } else {
        ImGui::TextWrapped("Modulation guide. Actual processing is shown in waveform and analysis.");
    }
    const float width = std::max(1.0f, ImGui::GetContentRegionAvail().x), height = designViewPixels(150);
    ImGui::InvisibleButton("EffectGraph", ImVec2(width, height));
    const auto first = ImGui::GetItemRectMin(), last = ImGui::GetItemRectMax();
    grid(first, last);
    const auto& points = visual.at("points");
    const double maxTime = points.back()[0].get<double>();
    const bool frequencyPlot = kind == "eq" || kind == "filter";
    linePlot(points, first, last, frequencyPlot ? 20 : kind == "transfer" ? -60 : 0,
             frequencyPlot ? 20000 : kind == "transfer" ? 6 : maxTime,
             kind == "eq" ? -24 : kind == "filter" || kind == "transfer" ? -60 :
             kind == "modulation" ? -1 : -120,
             kind == "eq" ? 24 : kind == "filter" ? 12 : kind == "transfer" ? 6 :
             kind == "modulation" ? 1 : 0, frequencyPlot);
    bool completed = false;
    const auto mouse = ImGui::GetIO().MousePos;
    if ((type == "compressor" || type == "limiter") && ImGui::IsItemActive()) {
        effect[type == "compressor" ? "threshold_db" : "ceiling_db"] =
            std::clamp(-60.0 + (mouse.x - first.x) / width * 66,
                       type == "compressor" ? -60.0 : -12.0, 0.0);
    } else if (type == "filter" && ImGui::IsItemActive()) {
        effect["cutoff_hz"] = std::min(16000.0, 20 * std::pow(1000.0,
            std::clamp((mouse.x - first.x) / width, 0.0f, 1.0f)));
        effect["resonance"] = std::clamp((last.y - mouse.y) / height * 0.95f, 0.0f, 0.95f);
    }
    completed |= ImGui::IsItemDeactivatedAfterEdit();
    if ((type == "compressor" || type == "limiter" || type == "filter") && ImGui::IsItemDeactivated()) {
        completed = true;
    }
    return completed;
}

bool drawDesignAnalysis(UIContext& ctx, const json& status, bool original, json& recipe) {
    if (!status.at("ready").get<bool>()) {
        ImGui::TextWrapped("Render a sound to inspect its spectrum, spectrogram and channel levels.");
        return false;
    }
    if (analyzer.owner != ctx.scene.audio.get() || analyzer.generation != ctx.scene.audio->generation()) {
        analyzer = {};
        analyzer.owner = ctx.scene.audio.get();
        analyzer.generation = ctx.scene.audio->generation();
    }
    const int sizes[] = {512, 1024, 2048, 4096, 8192};
    ImGui::SetNextItemWidth(110);
    if (ImGui::BeginCombo("FFT", std::to_string(analyzer.fftSize).c_str())) {
        for (const int size : sizes) {
            if (ImGui::Selectable(std::to_string(size).c_str(), size == analyzer.fftSize)) {
                analyzer.fftSize = size;
            }
        }
        ImGui::EndCombo();
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(110);
    if (ImGui::BeginCombo("Channel", analyzer.channel.c_str())) {
        for (const char* channel : {"mix", "left", "right"}) {
            ImGui::BeginDisabled(std::string(channel) == "right" &&
                status.at("processed").at("channels").get<unsigned>() == 1);
            if (ImGui::Selectable(channel, analyzer.channel == channel)) {
                analyzer.channel = channel;
            }
            ImGui::EndDisabled();
        }
        ImGui::EndCombo();
    }
    if (status.at("processed").at("channels").get<unsigned>() == 1 && analyzer.channel == "right") {
        analyzer.channel = "mix";
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(130);
    ImGui::SliderFloat("Floor", &analyzer.floorDb, -120, -40, "%.0f dB");
    const json params = {{"original", original}, {"fft_size", analyzer.fftSize},
        {"channel", analyzer.channel}, {"points", 256}, {"time_bins", 64}};
    const auto key = std::to_string(status.at("revision").get<std::uint64_t>()) + params.dump();
    auto job = read(ctx, "design_job_status");
    const auto phase = job.value("state", std::string("idle"));
    const bool renderBusy = job.value("kind", std::string()) == "render" &&
        (phase == "queued" || phase == "running" || phase == "completed");
    if (!renderBusy && key != analyzer.key) {
        if (analyzer.job != 0 && analyzer.requestedKey == key &&
            job.value("id", std::uint64_t(0)) == analyzer.job) {
            if (phase == "completed") {
                const auto result = read(ctx, "design_job_apply");
                if (result.contains("analysis")) {
                    analyzer.report = result.at("analysis");
                    analyzer.key = key;
                }
                analyzer.job = 0;
            } else if (phase == "failed" || phase == "cancelled" || phase == "consumed") {
                error = job.value("error", std::string("Analysis cancelled"));
                analyzer.job = 0;
                analyzer.key = key;
            }
        } else {
            job = read(ctx, "design_job_start", {{"kind", "analysis"}, {"options", params}});
            analyzer.job = job.value("id", std::uint64_t(0));
            analyzer.requestedKey = key;
            analyzer.report = json::object();
        }
    }
    if (analyzer.report.empty() || analyzer.key != key) {
        ImGui::TextWrapped("%s", error.empty() ? "Analyzing..." : error.c_str());
        return false;
    }
    const auto& report = analyzer.report;
    if (!original && status.contains("processing")) {
        ImGui::Text("Output clamp events before loop join: %llu",
            status.at("processing").at("clamped_samples_before_loop_join").get<unsigned long long>());
    }
    for (std::size_t channel = 0; channel < report.at("channel_stats").size(); ++channel) {
        const auto& stats = report.at("channel_stats")[channel];
        ImGui::TextWrapped("%s: peak %.1f dBFS | RMS %.1f dBFS | crest %.1f dB | DC %.5f | full-scale hits %llu",
            channel == 0 ? "L / mono" : "R", stats.at("peak_dbfs").get<double>(),
            stats.at("rms_dbfs").get<double>(), stats.at("crest_db").get<double>(),
            stats.at("dc_offset").get<double>(), stats.at("clip_samples").get<unsigned long long>());
    }
    ImGui::TextWrapped("Peak spectrum across all overlapping windows | resolution %.2f Hz | 20 Hz - 20 kHz",
                       report.at("frequency_resolution_hz").get<double>());
    const float width = std::max(1.0f, ImGui::GetContentRegionAvail().x);
    ImGui::InvisibleButton("AnalyzerSpectrum", ImVec2(width, designViewPixels(85)));
    grid(ImGui::GetItemRectMin(), ImGui::GetItemRectMax());
    linePlot(report.at("spectrum"), ImGui::GetItemRectMin(), ImGui::GetItemRectMax(),
             20, 20000, analyzer.floorDb, 0, true);
    ImGui::TextWrapped("Spectrogram | full time coverage, peak in each column | %.0f to 0 dBFS",
                       analyzer.floorDb);
    ImGui::InvisibleButton("AnalyzerSpectrogram", ImVec2(width, designViewPixels(140)));
    const auto first = ImGui::GetItemRectMin(), last = ImGui::GetItemRectMax();
    auto* draw = ImGui::GetWindowDrawList();
    const auto& map = report.at("spectrogram");
    constexpr unsigned rows = 64;
    for (unsigned column = 0; column < map.size(); ++column) {
        for (unsigned row = 0; row < rows; ++row) {
            double db = -120;
            const auto begin = map[column].size() * row / rows;
            const auto end = map[column].size() * (row + 1) / rows;
            for (auto bin = begin; bin < end; ++bin) {
                db = std::max(db, map[column][bin].get<double>());
            }
            const float level = static_cast<float>(std::clamp((db - analyzer.floorDb) /
                                                             -analyzer.floorDb, 0.0, 1.0));
            const auto color = ImGui::ColorConvertFloat4ToU32(ImVec4(
                std::clamp(level * 2, 0.0f, 1.0f), std::clamp((level - 0.4f) * 1.7f, 0.0f, 1.0f),
                0.15f + std::sin(level * 3.14159265f) * 0.6f, 1));
            draw->AddRectFilled(ImVec2(first.x + width * column / map.size(),
                                     last.y - designViewPixels(140) * (row + 1) / rows),
                ImVec2(first.x + width * (column + 1) / map.size(), last.y - designViewPixels(140) * row / rows), color);
        }
    }
    if (ImGui::IsItemHovered()) {
        const auto mouse = ImGui::GetIO().MousePos;
        const double position = std::clamp((mouse.x - first.x) / width, 0.0f, 1.0f);
        const double frequency = 20 * std::pow(1000.0, std::clamp((last.y - mouse.y) / designViewPixels(140), 0.0f, 1.0f));
        ImGui::SetTooltip("%.3f s | %.0f Hz", position * report.at("duration_seconds").get<double>(), frequency);
    }
    const auto revision = status.at("revision").get<std::uint64_t>();
    if (analyzer.selectionRevision != revision) {
        analyzer.selectionRevision = revision;
        analyzer.selecting = false;
        analyzer.selectionReady = false;
    }
    const bool editable = !original && recipe == status.at("recipe") && !renderBusy;
    const auto position = [&] {
        const auto mouse = ImGui::GetIO().MousePos;
        return ImVec2(std::clamp((mouse.x - first.x) / width, 0.0f, 1.0f),
            std::clamp((last.y - mouse.y) / (last.y - first.y), 0.0f, 1.0f));
    };
    if (editable && ImGui::IsItemActivated()) {
        analyzer.selectionStart = position();
        analyzer.selectionEnd = analyzer.selectionStart;
        analyzer.selecting = true;
        analyzer.selectionReady = false;
    }
    if (analyzer.selecting) {
        if (!editable || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
            analyzer.selecting = false;
            analyzer.selectionReady = false;
        } else {
            analyzer.selectionEnd = position();
            if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
                analyzer.selecting = false;
                analyzer.selectionReady =
                    std::abs(analyzer.selectionStart.x - analyzer.selectionEnd.x) * width >= 2 &&
                    std::abs(analyzer.selectionStart.y - analyzer.selectionEnd.y) *
                        (last.y - first.y) >= 2;
            }
        }
    }
    if (analyzer.selecting || analyzer.selectionReady) {
        const ImVec2 low(first.x + width * std::min(analyzer.selectionStart.x,
                             analyzer.selectionEnd.x),
            last.y - (last.y - first.y) * std::max(analyzer.selectionStart.y,
                             analyzer.selectionEnd.y));
        const ImVec2 high(first.x + width * std::max(analyzer.selectionStart.x,
                              analyzer.selectionEnd.x),
            last.y - (last.y - first.y) * std::min(analyzer.selectionStart.y,
                              analyzer.selectionEnd.y));
        draw->AddRectFilled(low, high, IM_COL32(80, 180, 255, 45));
        draw->AddRect(low, high, IM_COL32(120, 210, 255, 255), 0, 0, 2);
    }
    ImGui::TextWrapped("Drag a time/frequency region to change its sound. Escape cancels selection. "
                       "Masks are saved in the effect rack; source files stay intact.");
    if (original) {
        ImGui::TextWrapped("Select Processed to author spectral masks.");
    }
    ImGui::SetNextItemWidth(150);
    ImGui::SliderFloat("Region gain", &analyzer.regionGainDb, -96, 12, "%.1f dB");
    ImGui::SetNextItemWidth(150);
    ImGui::SliderFloat("Edge softness", &analyzer.feather, 0, 1);
    ImGui::SetNextItemWidth(150);
    ImGui::Combo("Mask shape", &analyzer.shape, "Rectangle\0Ellipse\0");
    const auto count = recipe.value("effects", json::array()).size();
    ImGui::BeginDisabled(!editable || !analyzer.selectionReady || count >= 8);
    const bool apply = ImGui::Button("Apply spectral mask");
    ImGui::EndDisabled();
    if (count >= 8) {
        ImGui::TextWrapped("The effect rack is full (8 modules). Remove an effect before adding a mask.");
    }
    if (!apply) {
        return false;
    }
    const double duration = report.at("duration_seconds");
    const double begin = std::min(analyzer.selectionStart.x, analyzer.selectionEnd.x) * duration;
    const double end = std::max(analyzer.selectionStart.x, analyzer.selectionEnd.x) * duration;
    const double low = 20 * std::pow(1000.0,
        std::min(analyzer.selectionStart.y, analyzer.selectionEnd.y));
    const double high = 20 * std::pow(1000.0,
        std::max(analyzer.selectionStart.y, analyzer.selectionEnd.y));
    if (!recipe.contains("effects")) {
        recipe["effects"] = json::array();
    }
    recipe["effects"].push_back({{"type", "spectral_region"}, {"enabled", true}, {"mix", 1},
        {"start_seconds", begin}, {"end_seconds", std::max(begin + 0.001, end)},
        {"low_hz", low}, {"high_hz", high}, {"gain_db", analyzer.regionGainDb},
        {"feather", analyzer.feather}, {"shape", analyzer.shape == 0 ? "rectangle" : "ellipse"},
        {"channel", analyzer.channel == "mix" ? "all" : analyzer.channel}});
    analyzer.selectionReady = false;
    return true;
}
} // namespace AudioUI
