#include "UI/AudioReactionUI.h"
#include "UI/AudioPropertyLayout.h"
#include "Audio/AudioAuthoring.h"
#include "Audio/AudioService.h"
#include "Audio/AudioReaction.h"
#include "Audio/AudioDomainBridge.h"
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
    if (ImGui::BeginCombo(AudioUI::property("Driver").c_str(), config.at("driver").get<std::string>().c_str())) {
        for (const char* driver : {"manual", "wind", "rain", "signal", "domain_contact"}) {
            if (ImGui::Selectable(driver, config.at("driver") == driver)) {
                config["driver"] = driver;
                config["input_metric"] = "energy_j";
                config["input_max"] = std::string(driver) == "wind" ? 30.0f : 1.0f;
                if (std::string(driver) == "signal" && config.at("signal_key") == "") {
                    config["signal_key"] = "contact";
                }
                if (std::string(driver) == "manual") {
                    config["automatic"] = false;
                }
                if (std::string(driver) == "domain_contact") {
                    const auto domains = audio::domainAudioInputs(ctx);
                    if (!domains.empty()) {
                        config["system_id"] = domains[0].at("system_id");
                        config["domain_name"] = domains[0].at("domain_name");
                    }
                    config["source_material"] = "";
                    config["target_material"] = "";
                    config["follow_position"] = true;
                }
                call(ctx, "update", {{"id", id}, {"reaction", config}});
            }
        }
        ImGui::EndCombo();
    }
    if (config.at("driver") != "manual") {
        ImGui::TextWrapped("Bind a sound asset and enable Automatic to start/stop with activity. "
                           "Pause/Stop suspend automation until Play or reconfiguration.");
        const auto source = ctx.scene.audio->invoke("source_runtime", {{"id", id}});
        const auto& position = source.at("position");
        ImGui::Text("World position: %.2f, %.2f, %.2f", position.at(0).get<float>(),
                    position.at(1).get<float>(), position.at(2).get<float>());
        if (source.at("position_origin") == "reaction_world") {
            ImGui::TextWrapped("Position follows the physics/signal input. Disable Follow position "
                               "to place the source manually. Scene acoustics uses this same position.");
        }
        bool automatic = config.at("automatic");
        if (ImGui::Checkbox(AudioUI::property("Automatic start / stop").c_str(), &automatic)) {
            auto edited = config;
            edited["automatic"] = automatic;
            call(ctx, "update", {{"id", id}, {"reaction", edited}, {"loop", true}});
        }
        for (const char* key : {"input_min", "input_max", "gain_min", "gain_max",
                                "pitch_min", "pitch_max", "response_seconds", "start_threshold",
                                "stop_threshold"}) {
            float value = config.at(key);
            if (ImGui::InputFloat(AudioUI::property(key).c_str(), &value, 0, 0, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue)) {
                auto edited = config;
                edited[key] = value;
                call(ctx, "update", {{"id", id}, {"reaction", edited}});
            }
        }
        if (config.at("driver") == "domain_contact") {
            const auto policy = ctx.scene.audio->invoke("telemetry_get", json::object());
            bool collect = policy.at("enabled");
            if (ImGui::Checkbox(AudioUI::property("Collect physics audio telemetry (scene)").c_str(), &collect)) {
                call(ctx, "telemetry_update", {{"enabled", collect}});
            }
            if (!collect) {
                ImGui::TextWrapped("Physics audio collection is off. Binding a sound does not "
                                   "enable particle telemetry; opt in manually when needed.");
            }
            const std::string channel = config.at("domain_channel");
            if (ImGui::BeginCombo(AudioUI::property("Physics input").c_str(), channel.c_str())) {
                for (const char* choice : {"mixed_contact", "dem_contact", "dem_friction",
                                           "dem_rolling", "fluid_flow", "gas_flow",
                                           "mpm_deformation", "mpm_fracture"}) {
                    if (ImGui::Selectable(choice, channel == choice)) {
                        auto edited = config;
                        edited["domain_channel"] = choice;
                        if (std::string(choice) == "mpm_fracture") {
                            edited["input_metric"] = "value";
                        } else if (std::string(choice) == "fluid_flow" ||
                                   std::string(choice) == "gas_flow" ||
                                   std::string(choice) == "mpm_deformation") {
                            edited["input_metric"] = "power_w";
                        }
                        call(ctx, "update", {{"id", id}, {"reaction", edited}});
                    }
                }
                ImGui::EndCombo();
            }
            const std::string metric = config.at("input_metric");
            if (ImGui::BeginCombo(AudioUI::property("Contact magnitude").c_str(), metric.c_str())) {
                  for (const char* choice : {"energy_j", "power_w", "value"}) {
                    if (ImGui::Selectable(choice, metric == choice)) {
                        auto edited = config;
                        edited["input_metric"] = choice;
                        call(ctx, "update", {{"id", id}, {"reaction", edited}});
                    }
                }
                ImGui::EndCombo();
            }
            const auto domains = audio::domainAudioInputs(ctx);
            const std::string label = std::to_string(config.at("system_id").get<int>()) +
                                      "/" + config.at("domain_name").get<std::string>();
            if (ImGui::BeginCombo(AudioUI::property("Domain binding").c_str(), label.c_str())) {
                for (const auto& domain : domains) {
                    if (domain.at("domain_channel") != channel) {
                        continue;
                    }
                    const std::string choice = domain.at("system_name").get<std::string>() +
                                              " / " + domain.at("domain_name").get<std::string>();
                    if (ImGui::Selectable(choice.c_str())) {
                        auto edited = config;
                        edited["system_id"] = domain.at("system_id");
                        edited["domain_name"] = domain.at("domain_name");
                        call(ctx, "update", {{"id", id}, {"reaction", edited}});
                    }
                }
                ImGui::EndCombo();
            }
            ImGui::TextWrapped("Energy: J per solver step. Power: energy / measured step duration. "
                               "Unavailable power stays silent. Position uses the measured "
                               "contact centroid when available, otherwise domain center. "
                               "Flow/deformation report end-state power samples; fracture reports "
                               "damage growth per second. Use Value for fracture. Material pairs "
                               "are unavailable.");
            for (const auto& domain : domains) {
                if (domain.at("system_id") == config.at("system_id") &&
                    domain.at("domain_name") == config.at("domain_name") &&
                    domain.at("domain_channel") == channel) {
                    ImGui::Text("Report: %s", domain.at("measured").get<bool>() ?
                                "measured" : "unavailable");
                    ImGui::Text("Energy: %s / %.4f J", domain.at("energy_measured").get<bool>() ?
                                "measured" : "unavailable", domain.at("contact_energy_j").get<double>());
                    ImGui::Text("Value: %.4f %s", domain.at("value").get<double>(),
                                domain.at("value_units").get<std::string>().c_str());
                    ImGui::TextWrapped("%s", domain.at("reason").get<std::string>().c_str());
                    if (!domain.at("measured").get<bool>()) {
                        ImGui::TextWrapped("Binding is configured, but the selected physics input "
                                           "has no measurement. Other inputs in this domain do "
                                           "not substitute for it.");
                    }
                    ImGui::Text("Power: %s / %.4f W", domain.at("power_measured").get<bool>() ?
                                "measured" : "unavailable",
                                domain.at("contact_power_w").get<double>());
                    ImGui::TextWrapped("Position: %s",
                        domain.at("position_semantics").get<std::string>().c_str());
                    ImGui::TextWrapped("Measurement: %s",
                        domain.at("measurement").get<std::string>().c_str());
                    if (domain.at("audio_error") != "") {
                        ImGui::TextWrapped("Telemetry: %s",
                                           domain.at("audio_error").get<std::string>().c_str());
                    }
                }
            }
        }
        if (config.at("driver") == "signal") {
            for (const char* key : {"signal_key", "source_material", "target_material"}) {
                char text[257]{};
                std::snprintf(text, sizeof(text), "%s", config.at(key).get<std::string>().c_str());
                if (ImGui::InputText(AudioUI::property(key).c_str(), text, sizeof(text), ImGuiInputTextFlags_EnterReturnsTrue)) {
                    auto edited = config;
                    edited[key] = text;
                    call(ctx, "update", {{"id", id}, {"reaction", edited}});
                }
            }
            bool follow = config.at("follow_position");
            if (ImGui::Checkbox(AudioUI::property("Follow signal world position").c_str(), &follow)) {
                auto edited = config;
                edited["follow_position"] = follow;
                call(ctx, "update", {{"id", id}, {"reaction", edited}});
            }
            if (ImGui::TreeNode("Signal preview (external telemetry input)")) {
                static float flow = 1;
                static float speed = 1;
                static float position[3]{};
                ImGui::InputFloat(AudioUI::property("Mass flow kg/s").c_str(), &flow);
                ImGui::InputFloat(AudioUI::property("Contact speed m/s").c_str(), &speed);
                ImGui::InputFloat3(AudioUI::property("Contact world position").c_str(), position);
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
        if (runtime.at("automatic_errors").contains(id)) {
            ImGui::TextWrapped("Automatic playback: %s",
                runtime.at("automatic_errors").at(id).get<std::string>().c_str());
        }
        for (const auto& suspended : runtime.at("automatic_suspended")) {
            if (suspended == id) {
                ImGui::TextWrapped("Automatic playback is suspended. "
                                   "Use Play or change its reaction settings.");
            }
        }
        const auto& outputs = runtime.at("reaction_runtime").at("outputs");
        if (outputs.contains(id)) {
            const auto& output = outputs.at(id);
            ImGui::TextWrapped("Input %.3f / gain x%.3f / pitch x%.3f",
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
