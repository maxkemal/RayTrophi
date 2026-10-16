#include "Audio/AcousticBoundary.h"
#include "Audio/AudioAuthoring.h"
#include "SubstanceLibrary.h"
#include "scene_ui.h"
#include "imgui.h"
#include "UI/AudioPropertyLayout.h"
#include <exception>
#include <string>

namespace audio {
void drawAcousticBoundaryProbe(UIContext& ctx) {
    if (!ImGui::TreeNode("Acoustic interface probe")) {
        return;
    }
    static std::string from, to, error, resultKey;
    static float angle = 0, distance = 1;
    static nlohmann::json result;
    auto choose = [](const char* label, std::string& selected) {
        if (ImGui::BeginCombo(AudioUI::property(label).c_str(),
            selected.empty() ? "Choose defined medium" : selected.c_str())) {
            for (const auto* profile : RayTrophiSim::substanceProfiles()) {
                if (!profile->acoustic_medium_defined) {
                    continue;
                }
                if (ImGui::Selectable(profile->name.c_str(), selected == profile->name)) {
                    selected = profile->name;
                }
            }
            ImGui::EndCombo();
        }
    };
    ImGui::TextWrapped("Define acoustic medium density/speed on a derived Substance, then "
        "enable acoustic_medium_defined. Equivalent fluid only; no elastic shear modes.");
    choose("Incident medium", from);
    choose("Transmitted medium", to);
    ImGui::DragFloat(AudioUI::property("Angle from normal (deg)").c_str(),
        &angle, 0.1f, 0, 89.9f);
    ImGui::DragFloat(AudioUI::property("Transmitted path (m)").c_str(),
        &distance, 0.01f, 0, 1000000, "%.3f");
    const nlohmann::json params = {{"from_substance", from}, {"to_substance", to},
        {"incidence_degrees", angle}, {"transmitted_distance_meters", distance}};
    const auto key = params.dump() + ":" +
        std::to_string(RayTrophiSim::substanceLibraryRevision());
    ImGui::BeginDisabled(from.empty() || to.empty());
    if (ImGui::Button("Evaluate interface")) {
        try {
            result = author(ctx, nullptr, "acoustics_boundary", params);
            resultKey = key;
            error.clear();
        } catch (const std::exception& exception) {
            result = nullptr;
            error = exception.what();
        }
    }
    ImGui::EndDisabled();
    ImGui::TextWrapped("This probe does not bend viewport paths or simulate edge diffraction.");
    if (!error.empty()) {
        ImGui::TextWrapped("%s", error.c_str());
    }
    if (!result.is_null() && resultKey != key) {
        ImGui::TextDisabled("Inputs/materials changed; evaluate again.");
    } else if (!result.is_null()) {
        ImGui::Text("Energy R %.6f | T %.6f", result.at("reflection_energy").get<double>(),
            result.at("transmission_energy").get<double>());
        if (result.at("total_internal_reflection").get<bool>()) {
            ImGui::TextWrapped("Total internal reflection; transmitted field is evanescent.");
        } else {
            ImGui::Text("Refracted angle %.3f deg | delay %.3f ms",
                result.at("transmitted_angle_degrees").get<double>(),
                result.at("delay_seconds").get<double>() * 1000);
        }
        const auto phase = result.at("reflected_pressure").at("phase_radians");
        if (!phase.is_null()) {
            ImGui::Text("Reflected pressure phase %.4f rad", phase.get<double>());
        }
        for (const auto& band : result.at("bands")) {
            ImGui::Text("%u Hz | wavelength %.3f m | transmitted energy %.5f",
                band.at("frequency_hz").get<unsigned>(),
                band.at("transmitted_wavelength_meters").get<double>(),
                band.at("transmitted_energy_after_distance").get<double>());
        }
    }
    ImGui::TreePop();
}
} // namespace audio
