#include "UI/AudioReactionUI.h"
#include "Audio/AudioAuthoring.h"
#include "Audio/AudioService.h"
#include "Audio/AudioReaction.h"
#include "scene_ui.h"
#include "scene_data.h"
#include "imgui.h"
#include <cstdio>
#include <exception>
#include <string>

namespace AudioReactionUI {
namespace {
std::string error;
void call(UIContext& ctx, const std::string& op, const nlohmann::json& p) {
    try {
        audio::author(ctx, ctx.scene_ui_ptr ? &ctx.scene_ui_ptr->history : nullptr, op, p);
        error.clear();
    } catch (const std::exception& exception) {
        error = exception.what();
    }
}
}
void draw(UIContext& ctx, const nlohmann::json& emitter) {
    using json = nlohmann::json;
    if (!ImGui::TreeNode("Reactive Audio")) {
        return;
    }
    const std::string id = emitter.at("id");
    auto config = audio::AudioReaction::validate(emitter.value("reaction", json::object()));
    if (ImGui::BeginCombo("Driver", config.at("driver").get<std::string>().c_str())) {
        for (const char* driver : {"manual", "wind", "rain", "signal"}) {
            if (ImGui::Selectable(driver, config.at("driver") == driver)) {
                config["driver"] = driver;
                config["input_max"] = std::string(driver) == "wind" ? 30.0f : 1.0f;
                if (std::string(driver) == "signal" && config.at("signal_key") == "") {
                    config["signal_key"] = "contact";
                }
                call(ctx, "update", {{"id", id}, {"reaction", config}});
            }
        }
        ImGui::EndCombo();
    }
    if (config.at("driver") != "manual") {
        ImGui::TextWrapped("Choose a suitable sound asset, enable loop, then Play. "
                           "The driver changes gain/pitch without editing source settings.");
        for (const char* key : {"input_min", "input_max", "gain_min", "gain_max",
                                "pitch_min", "pitch_max", "response_seconds"}) {
            float value = config.at(key);
            if (ImGui::InputFloat(key, &value, 0, 0, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue)) {
                auto edited = config;
                edited[key] = value;
                call(ctx, "update", {{"id", id}, {"reaction", edited}});
            }
        }
        if (config.at("driver") == "signal") {
            for (const char* key : {"signal_key", "source_material", "target_material"}) {
                char text[257]{};
                std::snprintf(text, sizeof(text), "%s", config.at(key).get<std::string>().c_str());
                if (ImGui::InputText(key, text, sizeof(text), ImGuiInputTextFlags_EnterReturnsTrue)) {
                    auto edited = config;
                    edited[key] = text;
                    call(ctx, "update", {{"id", id}, {"reaction", edited}});
                }
            }
            bool follow = config.at("follow_position");
            if (ImGui::Checkbox("Follow signal world position", &follow)) {
                auto edited = config;
                edited["follow_position"] = follow;
                call(ctx, "update", {{"id", id}, {"reaction", edited}});
            }
            if (ImGui::TreeNode("Signal preview (external telemetry input)")) {
                static float flow = 1;
                static float speed = 1;
                static float position[3]{};
                ImGui::InputFloat("Mass flow kg/s", &flow);
                ImGui::InputFloat("Contact speed m/s", &speed);
                ImGui::InputFloat3("Contact world position", position);
                if (ImGui::Button("Publish 2-second signal")) {
                    call(ctx, "signal_set", {{"key", config.at("signal_key")},
                        {"flow_kg_s", flow}, {"speed_mps", speed}, {"ttl_seconds", 2},
                        {"position", json::array({position[0], position[1], position[2]})},
                        {"source_material", config.at("source_material")},
                        {"target_material", config.at("target_material")}});
                }
                ImGui::SameLine();
                if (ImGui::Button("Clear signals")) {
                    call(ctx, "signals_clear", json::object());
                }
                ImGui::TreePop();
            }
        }
        const auto runtime = ctx.scene.audio->invoke("status", json::object());
        const auto& outputs = runtime.at("reaction_runtime").at("outputs");
        if (outputs.contains(id)) {
            const auto& output = outputs.at(id);
            ImGui::Text("Input %.3f / gain x%.3f / pitch x%.3f",
                        output.at("input").get<float>(), output.at("gain_multiplier").get<float>(),
                        output.at("pitch_multiplier").get<float>());
        }
    }
    if (!error.empty()) {
        ImGui::TextWrapped("%s", error.c_str());
    }
    ImGui::TreePop();
}
}
