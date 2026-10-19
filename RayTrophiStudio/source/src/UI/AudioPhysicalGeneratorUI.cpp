#include "UI/AudioPhysicalGeneratorUI.h"
#include "UI/AudioPropertyLayout.h"
#include "Audio/AudioService.h"
#include "Audio/AudioAuthoring.h"
#include "scene_ui.h"
#include "scene_data.h"
#include "imgui.h"
#include <algorithm>
#include <cstdint>
#include <exception>
#include <string>

namespace AudioUI {
namespace {
using json = nlohmann::json;
std::string error;

void configure(UIContext& ctx, const json& source, const json& recipe, bool clearHistory = false) {
    try {
        audio::author(ctx, ctx.scene_ui_ptr ? &ctx.scene_ui_ptr->history : nullptr,
            "physical_configure", {{"id", source.at("id")}, {"generator", recipe},
                {"clear_history", clearHistory}});
        error.clear();
    } catch (const std::exception& exception) {
        error = exception.what();
    }
}
}

void drawPhysicalGenerator(UIContext& ctx, const json& source) {
    if (!ImGui::CollapsingHeader("Synthetic generator")) {
        return;
    }
    static std::string previousSource;
    const auto sourceId = source.at("id").get<std::string>();
    if (previousSource != sourceId) {
        previousSource = sourceId;
        error.clear();
    }
    const auto catalogue = ctx.scene.audio->invoke("physical_catalogue", json::object());
    const auto current = source.value("generator", json());
    const bool generated = !current.is_null();
    const auto model = generated ? current.at("model").get<std::string>() : std::string("Audio asset");
    std::string modelLabel = model;
    for (const auto& entry : catalogue.at("models")) {
        if (entry.at("model") == model) {
            modelLabel = entry.at("label").get<std::string>();
        }
    }
    if (ImGui::BeginCombo(property("Model").c_str(), modelLabel.c_str())) {
        for (const auto& entry : catalogue.at("models")) {
            const auto name = entry.at("model").get<std::string>();
            const auto label = entry.at("label").get<std::string>();
            if (ImGui::Selectable(label.c_str(), generated && name == model) && name != model) {
                configure(ctx, source, entry.at("default_generator"));
            }
        }
        ImGui::EndCombo();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Select a procedural source instead of its audio file. "
            "Source identity and routing stay the same. Incompatible domain bindings/history reject.");
    }
    if (!generated) {
        ImGui::TextWrapped("Choose a model to generate sound without an audio file.");
    } else {
        auto recipe = current;
        const int version = current.at("version");
        if (ImGui::BeginCombo(property("Sound engine").c_str(), version == 1 ? "v1 / original" :
                "v2 / layered texture")) {
            for (int value : {1, 2}) {
                if (ImGui::Selectable(value == 1 ? "v1 / original" : "v2 / layered texture",
                        version == value)) {
                    recipe["version"] = value;
                    if (value == 1) {
                        for (const char* key : {"texture", "variation", "body", "character_revision", "surface"}) {
                            recipe.erase(key);
                        }
                    }
                    configure(ctx, source, recipe);
                    recipe = current;
                }
            }
            ImGui::EndCombo();
        }
        if (version == 2) {
            const int revision = current.value("character_revision", 1);
            if (ImGui::BeginCombo(property("Sound character").c_str(),
                    revision == 1 ? "Original" : "Tuned / material character")) {
                for (int value : {1, 2}) {
                    if (ImGui::Selectable(value == 1 ? "Original" : "Tuned / material character",
                            value == revision)) {
                        recipe["character_revision"] = value;
                        if (value == 1) {
                            recipe.erase("surface");
                        }
                        configure(ctx, source, recipe);
                        recipe = current;
                    }
                }
                ImGui::EndCombo();
            }
            if (current.contains("surface")) {
                const auto selected = current.at("surface").get<std::string>();
                if (ImGui::BeginCombo(property("Friction surface").c_str(), selected.c_str())) {
                    for (const auto& entry : catalogue.at("friction_surfaces")) {
                        const auto name = entry.get<std::string>();
                        if (ImGui::Selectable(name.c_str(), name == selected)) {
                            recipe["surface"] = name;
                            configure(ctx, source, recipe);
                            recipe = current;
                        }
                    }
                    ImGui::EndCombo();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Authored surface character; not automatic material-pair detection.");
                }
            }
        }
        for (const char* key : {"brightness", "texture", "variation", "body"}) {
            if (version == 1 && std::string(key) != "brightness") {
                continue;
            }
            float value = current.value(key, 0.5f);
            const std::string field(key);
            const char* label = field == "brightness" ? "Brightness" : field == "texture" ?
                "Texture" : field == "variation" ? "Variation" : "Body";
            if (ImGui::InputFloat(property(label).c_str(), &value, 0, 0, "%.3f",
                    ImGuiInputTextFlags_EnterReturnsTrue)) {
                recipe[key] = value;
                configure(ctx, source, recipe);
                recipe = current;
            }
        }
        // Numeric entry commits once, so editing doesn't rebuild an active voice every frame.
        auto seed = current.at("seed").get<std::uint32_t>();
        if (ImGui::InputScalar(property("Seed").c_str(), ImGuiDataType_U32, &seed,
                nullptr, nullptr, "%u", ImGuiInputTextFlags_EnterReturnsTrue)) {
            recipe["seed"] = seed;
            configure(ctx, source, recipe);
            recipe = current;
        }
        if (current.contains("resonator")) {
            const auto resonator = current.at("resonator").get<std::string>();
            if (ImGui::BeginCombo(property("Resonator").c_str(), resonator.c_str())) {
                for (const auto& entry : catalogue.at("resonators")) {
                    const auto name = entry.get<std::string>();
                    if (ImGui::Selectable(name.c_str(), name == resonator)) {
                        recipe["resonator"] = name;
                        configure(ctx, source, recipe);
                        recipe = current;
                    }
                }
                ImGui::EndCombo();
            }
            for (const char* key : {"size_m", "decay_seconds", "reference_energy_j"}) {
                if (!current.contains(key)) {
                    continue;
                }
                float value = current.at(key);
                const std::string field(key);
                const char* label = field == "size_m" ? "Size (m)" :
                    field == "decay_seconds" ? "Decay (s)" : "Excitation reference (J)";
                if (ImGui::InputFloat(property(label).c_str(), &value, 0, 0, "%.4g",
                        ImGuiInputTextFlags_EnterReturnsTrue)) {
                    recipe[key] = value;
                    configure(ctx, source, recipe);
                    recipe = current;
                }
            }
        }
        if (!source.value("generator_history", json()).is_null()) {
            ImGui::TextDisabled("Stored take available");
            if (ImGui::SmallButton("Clear stored take")) {
                configure(ctx, source, current, true);
            }
        }
        if (model == "impact") {
            static float energy = 0.01f;
            ImGui::InputFloat(property("Test energy (J)").c_str(), &energy, 0, 0, "%.4g");
            if (ImGui::Button("Excite impact")) {
                try {
                    const auto runtime = ctx.scene.audio->invoke("source_runtime",
                        {{"id", source.at("id")}});
                    const auto previous = runtime.at("generator_runtime").value(
                        "last_accepted_sequence", json());
                    const auto sequence = previous.is_null() ? 1 : previous.get<std::uint64_t>() + 1;
                    const auto result = audio::author(ctx,
                        ctx.scene_ui_ptr ? &ctx.scene_ui_ptr->history : nullptr, "physical_impact",
                        {{"id", source.at("id")}, {"sequence", sequence}, {"energy_j", energy}});
                    error = result.value("accepted", false) ? "" : "Impact event was not accepted.";
                } catch (const std::exception& exception) {
                    error = exception.what();
                }
            }
        }
        ImGui::TextDisabled("Press Enter to apply numeric values. Texture / variation / body: 0..1.");
        ImGui::TextWrapped("Activity-driven synthetic sound. Microtexture is artistic; "
            "it is not measured splash, bubble or fracture events.");
        if (model != "impact") {
            ImGui::TextWrapped("Activity connection controls the sound. Bind the matching domain "
                "channel or publish its signal; zero activity stays silent.");
        }
    }
    if (!error.empty()) {
        ImGui::TextWrapped("Generator: %s", error.c_str());
    }
}
}
