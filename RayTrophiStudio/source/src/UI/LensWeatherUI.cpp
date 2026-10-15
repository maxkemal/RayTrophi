#include "UI/LensWeatherUI.h"
#include "Atmosphere/LensWeatherService.h"
#include "Atmosphere/AtmosphereClimate.h"
#include "Camera.h"
#include "imgui.h"
#include <exception>
#include <string>

namespace LensWeatherUI {
void draw(UIContext& context, Camera& camera) {
    if (!ImGui::TreeNodeEx("Camera glass weather (Lens weather)",
                           ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }
    static std::string error;
    static nlohmann::json measurement = nlohmann::json::object();
    static const Camera* measuredCamera = nullptr;
    if (measuredCamera != &camera) {
        measurement = nlohmann::json::object();
        measuredCamera = &camera;
    }
    const auto edit = [&](const nlohmann::json& patch) {
        try {
            atmosphere::setLensWeather(context, camera, patch);
            error.clear();
        } catch (const std::exception& exception) {
            error = exception.what();
        }
    };
    bool enabled = camera.lens_weather.enabled;
    if (ImGui::Checkbox("Precipitation on camera glass", &enabled)) {
        edit({{"enabled", enabled}});
    }
    ImGui::TextWrapped("Uses atmospheric rain, snow and ice. Play the timeline to accumulate; "
                       "pausing freezes the glass state.");
    const float temperature = atmosphere::sampleAmbient(camera.lookfrom).temperature_k - 273.15f;
    ImGui::Text("Atmosphere at camera: %.1f C", temperature);
    ImGui::TextDisabled("Temperature is controlled by Atmosphere / Climate.");
    bool hydrophobic = camera.lens_weather.hydrophobic;
    if (ImGui::Checkbox("Hydrophobic coating", &hydrophobic)) {
        edit({{"coating", hydrophobic ? "hydrophobic" : "bare"}});
    }
    float hood = camera.lens_weather.hood_deg;
    if (ImGui::InputFloat("Hood angle (deg)", &hood, 1, 5, "%.1f",
                           ImGuiInputTextFlags_EnterReturnsTrue)) {
        edit({{"hood_deg", hood}});
    }
    float diameter = camera.lens_weather.diameter_mm;
    if (ImGui::InputFloat("Glass diameter (mm)", &diameter, 1, 5, "%.1f",
                           ImGuiInputTextFlags_EnterReturnsTrue)) {
        edit({{"diameter_mm", diameter}});
    }
    auto seed = camera.lens_weather.seed;
    if (ImGui::InputScalar("Drop seed", ImGuiDataType_U32, &seed, nullptr, nullptr,
                            "%u", ImGuiInputTextFlags_EnterReturnsTrue)) {
        edit({{"seed", seed}});
    }
    if (ImGui::Button("Clear glass")) {
        atmosphere::resetLensWeather(context, camera);
    }
    if (ImGui::Button("Measure active viewport glass")) {
        try {
            measurement = atmosphere::sampleLensWeather(context, "viewport");
            error.clear();
            if (!measurement.value("pipelines_ready", false)) {
                error = "Lens shaders are unavailable. Rebuild shaders and reopen Studio.";
            }
        } catch (const std::exception& exception) {
            error = exception.what();
        }
    }
    if (measurement.value("initialized", false) &&
        measurement.value("state_matches_camera", false)) {
        ImGui::Text("Drops: %u | impacts: %.2f /s | coverage estimate: %.1f%%",
            measurement.value("drop_count", 0u), measurement.value("impact_rate_per_s", 0.0),
            measurement.value("coverage", 0.0) * 100);
        ImGui::Text("Snow: %.1f%% | ice: %.1f%% (water mass)",
            measurement.value("snow_fraction", 0.0) * 100,
            measurement.value("ice_fraction", 0.0) * 100);
        ImGui::Text("Optical field: water %.2f | snow %.2f | ice %.2f",
            measurement.value("field_max_coverage", 0.0),
            measurement.value("field_max_snow", 0.0),
            measurement.value("field_max_ice", 0.0));
        if (measurement.value("history_misses", 0u) > 0) {
            ImGui::TextWrapped("Seek exceeded recent history; glass restarted dry.");
        }
    }
    if (!error.empty()) {
        ImGui::TextWrapped("%s", error.c_str());
    }
    ImGui::TextDisabled("Vulkan RT / RayFusion; 256 drops, 10 s recent history.");
    ImGui::TreePop();
}
} // namespace LensWeatherUI
