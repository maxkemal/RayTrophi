#include "Fluid/MatterPoreAuthoring.h"
#include "../Api/RtMatterModels.h"
#include "imgui.h"

#include <exception>

namespace RayTrophiSim::Fluid {

void drawMatterPoreControls(const std::string& domain_name) {
    if (!ImGui::CollapsingHeader("Pore Water")) {
        return;
    }
    ImGui::PushID(domain_name.c_str());
    static std::string error;
    try {
        const auto state = rtapi::getMatterModels(domain_name, false).at("pore_exchange");
        const auto settings = state.at("settings");
        nlohmann::json patch = nlohmann::json::object();
        bool enabled = settings.at("enabled").get<bool>();
        if (ImGui::Checkbox("Enable water absorption / drainage", &enabled)) {
            patch["enabled"] = enabled;
        }
        auto slider = [&](const char* label, const char* key, float low, float high,
                          const char* format) {
            float value = settings.at(key).get<float>();
            ImGui::SetNextItemWidth(-1.0f);
            if (ImGui::SliderFloat(label, &value, low, high, format,
                                  ImGuiSliderFlags_Logarithmic)) {
                patch[key] = value;
            }
        };
        slider("Porosity", "porosity", 0.01f, 0.8f, "%.3f");
        slider("Permeability (m2)", "permeability_m2", 0.0f, 1e-6f, "%.3e");
        slider("Water viscosity (Pa.s)", "viscosity_pa_s", 1e-5f, 10.0f, "%.4g");
        slider("Gravity drive (m/s2)", "gravity_m_s2", 0.0f, 100.0f, "%.3f");
        slider("Drainage scale", "drainage_scale", 0.0f, 100.0f, "%.3f");
        ImGui::Separator();
        bool wet_physics = settings.at("wet_response_enabled").get<bool>();
        if (ImGui::Checkbox("Saturation affects granular strength", &wet_physics)) {
            patch["wet_response_enabled"] = wet_physics;
        }
        bool wet_appearance = settings.at("wet_appearance_enabled").get<bool>();
        if (ImGui::Checkbox("Saturation darkens Principled granular splats", &wet_appearance)) {
            patch["wet_appearance_enabled"] = wet_appearance;
        }
        slider("Saturated friction multiplier", "wet_friction_scale", 0.0f, 1.0f, "%.3f");
        slider("Saturated dilatancy multiplier", "wet_dilatancy_scale", 0.0f, 1.0f, "%.3f");
        slider("Peak capillary cohesion (Pa)", "capillary_cohesion_pa", 0.0f, 100000.0f, "%.2f");
        slider("Local pore pressure multiplier", "pore_pressure_scale", 0.0f, 10.0f, "%.3f");
        slider("Saturated color multiplier", "wet_color_scale", 0.05f, 1.0f, "%.3f");
        slider("Saturated roughness multiplier", "wet_roughness_scale", 0.05f, 1.0f, "%.3f");
        slider("Full wet look at pore saturation", "wet_appearance_full_saturation",
            0.001f, 1.0f, "%.3f");
        ImGui::TextWrapped("Full wet look at %.1f%% pore filling. Lower values make "
            "early wetting more visible; physical saturation and strength stay unchanged.",
            settings.at("wet_appearance_full_saturation").get<float>() * 100.0f);
        if (!patch.empty()) {
            rtapi::setMatterPoreExchange(domain_name, patch);
            error.clear();
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
