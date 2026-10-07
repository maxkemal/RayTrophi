#include "Fluid/MatterPoreAuthoring.h"
#include "../Api/RtMatterModels.h"
#include "ui_modern.h"
#include "imgui.h"
#include "DomainPanelWidgets.h"

#include <exception>
#include <map>

namespace RayTrophiSim::Fluid {

void drawMatterPoreControls(const std::string& domain_name, bool grains) {
    // With the grain solver the MPM pore physics is off (grains hold their own
    // film water, Wet grains), but the wet look reads the same per-particle
    // saturation (pore_water_mass_kg / pore_capacity_kg), so only it is shown.
    if (!UIWidgets::CollapsingHeader(grains ? "Wet Look" : "Pore Water")) {
        return;
    }
    ImGui::PushID(domain_name.c_str());
    static std::map<std::string, std::string> errors;
    auto& error = errors[domain_name];
    try {
        const auto state = rtapi::getMatterModels(domain_name, false).at("pore_exchange");
        const auto settings = state.at("settings");
        nlohmann::json patch = nlohmann::json::object();
        auto slider = [&](const char* label, const char* key, float low, float high,
                          const char* format, const char* tip) {
            float value = settings.at(key).get<float>();
            ImGui::SetNextItemWidth(DomainUi::itemWidth());
            if (ImGui::SliderFloat(label, &value, low, high, format,
                                  ImGuiSliderFlags_Logarithmic)) {
                patch[key] = value;
            }
            DomainUi::tooltip(tip);
        };
        if (!grains) {
        bool enabled = settings.at("enabled").get<bool>();
        if (ImGui::Checkbox("Enable water absorption / drainage", &enabled)) {
            patch["enabled"] = enabled;
        }
        DomainUi::tooltip("MPM granular carriers absorb water from liquid parcels into their pores\n"
            "and drain it back; water is held once (pore_water_mass_kg).\n"
            "Needs Closed Vulkan Matter; not with the grain solver.\n"
            "Script: fluid.set_pore_exchange(enabled=...)");
        slider("Porosity", "porosity", 0.01f, 0.8f, "%.3f",
            "Pore volume fraction of a granular carrier (0.01..0.8; sand ~0.35-0.4).\n"
            "Sets how much water a carrier can hold.\nScript: fluid.set_pore_exchange(porosity=...)");
        slider("Permeability (m2)", "permeability_m2", 0.0f, 1e-6f, "%.3e",
            "Intrinsic permeability (Darcy), m^2: sand ~1e-11..1e-10.\n"
            "Higher drains faster.\nScript: fluid.set_pore_exchange(permeability_m2=...)");
        slider("Water viscosity (Pa.s)", "viscosity_pa_s", 1e-5f, 10.0f, "%.4g",
            "Dynamic viscosity of the pore water (water 1e-3 Pa s) in the Darcy flux.\n"
            "Script: fluid.set_pore_exchange(viscosity_pa_s=...)");
        slider("Gravity drive (m/s2)", "gravity_m_s2", 0.0f, 100.0f, "%.3f",
            "Gravity that drives drainage through the pores (9.81 m/s^2).\n"
            "Script: fluid.set_pore_exchange(gravity_m_s2=...)");
        slider("Drainage scale", "drainage_scale", 0.0f, 100.0f, "%.3f",
            "Multiplier on the Darcy drainage rate (1 = physical; 0 holds water).\n"
            "Script: fluid.set_pore_exchange(drainage_scale=...)");
        ImGui::Separator();
        bool wet_physics = settings.at("wet_response_enabled").get<bool>();
        if (ImGui::Checkbox("Saturation affects granular strength", &wet_physics)) {
            patch["wet_response_enabled"] = wet_physics;
        }
        DomainUi::tooltip("Saturation changes the MPM friction, dilatancy and capillary cohesion\n"
            "(multipliers below). Physics, not look.\n"
            "Script: fluid.set_pore_exchange(wet_response_enabled=...)");
        }
        bool wet_appearance = settings.at("wet_appearance_enabled").get<bool>();
        if (ImGui::Checkbox("Saturation darkens Principled granular splats", &wet_appearance)) {
            patch["wet_appearance_enabled"] = wet_appearance;
        }
        DomainUi::tooltip(grains
            ? "Look only: grains render darker and smoother as they take up water\n"
              "(eight bands of film water / Water capacity, Wet grains on).\n"
              "Script: fluid.set_pore_exchange(wet_appearance_enabled=...)"
            : "Look only: wet carriers render darker and smoother (eight bands).\n"
              "Physics and saturation are unchanged.\n"
              "Script: fluid.set_pore_exchange(wet_appearance_enabled=...)");
        if (!grains) {
        slider("Saturated friction multiplier", "wet_friction_scale", 0.0f, 1.0f, "%.3f",
            "Friction multiplier at full saturation (0..1).\nScript: fluid.set_pore_exchange(wet_friction_scale=...)");
        slider("Saturated dilatancy multiplier", "wet_dilatancy_scale", 0.0f, 1.0f, "%.3f",
            "Dilatancy multiplier at full saturation (0..1).\nScript: fluid.set_pore_exchange(wet_dilatancy_scale=...)");
        slider("Peak capillary cohesion (Pa)", "capillary_cohesion_pa", 0.0f, 100000.0f, "%.2f",
            "Peak capillary cohesion at intermediate saturation, Pa (damp sand ~1e3).\nScript: fluid.set_pore_exchange(capillary_cohesion_pa=...)");
        slider("Local pore pressure multiplier", "pore_pressure_scale", 0.0f, 10.0f, "%.3f",
            "Multiplier on the empirical cell-height pore pressure head (0..10).\nScript: fluid.set_pore_exchange(pore_pressure_scale=...)");
        }
        slider("Saturated color multiplier", "wet_color_scale", 0.05f, 1.0f, "%.3f",
            "Look: base color multiplier at full wetness (0.05..1).\nScript: fluid.set_pore_exchange(wet_color_scale=...)");
        slider("Saturated roughness multiplier", "wet_roughness_scale", 0.05f, 1.0f, "%.3f",
            "Look: roughness multiplier at full wetness (0.05..1).\nScript: fluid.set_pore_exchange(wet_roughness_scale=...)");
        slider("Full wet look at pore saturation", "wet_appearance_full_saturation",
            0.001f, 1.0f, "%.3f",
            "Look: pore saturation at which the wet look is complete (0.001..1).\n"
            "Script: fluid.set_pore_exchange(wet_appearance_full_saturation=...)");
        ImGui::TextWrapped("Full wet look at %.1f%% pore filling. Lower values make "
            "early wetting more visible; physical saturation and strength stay unchanged.",
            settings.at("wet_appearance_full_saturation").get<float>() * 100.0f);
        if (!patch.empty()) {
            rtapi::setMatterPoreExchange(domain_name, patch);
            error.clear();
        }
        if (grains) {
            ImGui::TextWrapped("Grain saturation is film water over Water capacity (Wet grains).");
            ImGui::PopID();
            return;
        }
        ImGui::Text("Stored / capacity: %.6g / %.6g kg",
            state.at("pore_water_kg").get<double>(), state.at("capacity_kg").get<double>());
        ImGui::Text("Maximum saturation: %.4f", state.at("maximum_saturation").get<float>());
        ImGui::Text("Last absorbed / drained: %.6g / %.6g kg",
            state.at("absorbed_kg").get<double>(), state.at("drained_kg").get<double>());
        ImGui::Text("Carriers waiting for free particle slots: %llu",
            static_cast<unsigned long long>(
                state.at("drainage_budget_blocked_carriers").get<std::size_t>()));
        ImGui::TextWrapped("%s", state.at("status").get<std::string>().c_str());
        ImGui::TextWrapped(
            "Requires Closed Vulkan Matter with Water and Sand/Gravel/Soil. "
            "Drainage refills local Water parcels or reserves one birth per cell. "
            "Wet strength uses an empirical cell-height pressure head; this is not a "
            "pore-pressure projection. Principled splats use eight saturation bands; "
            "positive saturation selects at least the first wet band, zero remains dry.");
    } catch (const std::exception& failure) {
        error = failure.what();
    }
    if (!error.empty()) {
        ImGui::TextWrapped("%s", error.c_str());
    }
    ImGui::PopID();
}

} // namespace RayTrophiSim::Fluid
