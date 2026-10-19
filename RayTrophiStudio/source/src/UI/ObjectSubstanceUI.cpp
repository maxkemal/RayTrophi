#include "ObjectSubstance.h"
#include "scene_ui.h"
#include "scene_data.h"
#include "SubstanceLibrary.h"
#include "scene_ui_substance_editor.hpp"
#include "imgui.h"
#include <stdexcept>

namespace matter {
bool drawObjectSubstance(UIContext& ctx, const std::string& object) {
    using json = nlohmann::json;
    static std::string error;
    const auto state = objectSubstanceState(ctx.scene, object);
    if (!state.at("exists").get<bool>()) {
        return false;
    }
    ImGui::PushID("ObjectSubstance");
    if (!ImGui::CollapsingHeader("Physical material", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::PopID();
        return true;
    }
    const auto call = [&](const char* operation, const json& params) {
        try {
            invokeObjectSubstance(ctx, operation, params);
            error.clear();
        } catch (const std::exception& exception) {
            error = exception.what();
        }
    };
    const auto source = state.at("source").get<std::string>();
    const auto& binding = state.at("binding");
    const auto label = binding.is_null() ? std::string("Unassigned") :
        binding.at("substance").get<std::string>();
    ImGui::TextWrapped("What this object is made of. Shared by simulation consumers; "
                       "assigning it does not enable a collider or simulation.");
    if (source == "conflict") {
        ImGui::TextWrapped("Collider materials disagree. Choose the object's material explicitly.");
        for (const auto& collider : state.at("colliders")) {
            ImGui::TextWrapped("%s: %s", collider.at("name").get<std::string>().c_str(),
                collider.at("binding").at("substance").get<std::string>().c_str());
        }
    } else if (source == "legacy_collider") {
        ImGui::TextWrapped("Inherited from existing collider settings.");
        if (ImGui::Button("Use as object material")) {
            call("adopt_object", {{"object", object}});
        }
    }
    ImGui::TextWrapped("Assignment: %s", object.c_str());
    auto chosen = binding.is_null() ? std::string() : label;
    if (SubstanceEditorUI::drawPicker("Substance", chosen, "Unassigned")) {
        if (chosen.empty()) {
            call("clear_object", {{"object", object}});
        } else {
            auto updated = binding.is_null() ? json::object() : binding;
            updated["substance"] = chosen;
            call("set_object", {{"object", object}, {"binding", updated}});
        }
        // This frame's readback describes the old binding; do not edit it below.
        ImGui::PopID();
        return true;
    }
    if (source == "object") {
        if (state.contains("acoustics") && !state.at("acoustics").value("defined", false)) {
            ImGui::TextWrapped("Acoustic response is not defined for this substance yet.");
        }
        auto assigned = label;
        if (SubstanceEditorUI::draw("ObjectSubstanceProfile", assigned) && assigned != label) {
            auto changed = binding;
            changed["substance"] = assigned;
            call("set_object", {{"object", object}, {"binding", changed}});
            ImGui::PopID();
            return true;
        }
        if (ImGui::Button("Clear object assignment")) {
            call("clear_object", {{"object", object}});
            ImGui::PopID();
            return true;
        }
        static const SceneData* owner = nullptr;
        static std::string key;
        static json draft;
        const auto nextKey = object + "\n" + std::to_string(ctx.scene.load_counter) +
            "\n" + binding.dump();
        if (owner != &ctx.scene || key != nextKey) {
            owner = &ctx.scene;
            key = nextKey;
            draft = binding;
        }
        if (!ImGui::TreeNode("This object's combustion overrides")) {
            if (!error.empty()) {
                ImGui::TextWrapped("%s", error.c_str());
            }
            ImGui::PopID();
            return true;
        }
        auto& values = draft["overrides"];
        bool overrideIgnition = values.at("override_ignition");
        bool commit = ImGui::Checkbox("Override ignition", &overrideIgnition);
        values["override_ignition"] = overrideIgnition;
        for (const auto* field : {"ignition_kelvin", "burn_rate_scale", "fuel_capacity_scale"}) {
            float value = values.at(field);
            const char* label = std::string(field) == "ignition_kelvin" ? "Ignition (K)" :
                std::string(field) == "burn_rate_scale" ? "Burn rate multiplier" :
                                                       "Fuel capacity multiplier";
            ImGui::BeginDisabled(std::string(field) == "ignition_kelvin" && !overrideIgnition);
            ImGui::DragFloat(label, &value, 0.01f, 0,
                std::string(field) == "ignition_kelvin" ? 100000.0f : 1000000.0f);
            values[field] = value;
            commit = ImGui::IsItemDeactivatedAfterEdit() || commit;
            ImGui::EndDisabled();
        }
        if (commit) {
            call("set_object", {{"object", object}, {"binding", draft}});
        }
        ImGui::TreePop();
    }
    if (!error.empty()) {
        ImGui::TextWrapped("%s", error.c_str());
    }
    ImGui::PopID();
    return true;
}
} // namespace matter
