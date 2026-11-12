#include "UI/DomainVisualUI.h"
#include "Api/DomainVisualApi.h"
#include "ParticleSimulation.h"
#include "imgui.h"

#include <map>

namespace RayTrophiSim::Fluid {

void drawDomainVisualControls(const SimulationGridDomainDesc& domain) {
    if (!ImGui::TreeNode("Field samples")) {
        return;
    }
    static std::map<std::string, std::string> errors;
    auto& error = errors[domain.name];
    auto set = [&](const nlohmann::json& patch) {
        const auto result = rtapi::setDomainVisualSamples(domain.name, patch);
        error = result.ok ? "" : result.error;
    };
    bool enabled = domain.fluid_visual_samples.enabled;
    if (ImGui::Checkbox("Dense field samples", &enabled)) {
        set({{"enabled", enabled}});
    }
    ImGui::TextWrapped("Adds visual grains without adding physical particles. "
        "Uses the existing splat geometry and scene-object selections. "
        "Contact grains and spray keep their physical representation.");
    if (enabled) {
        const auto& settings = domain.fluid_visual_samples.sampling;
        float diameter = static_cast<float>(std::cbrt(6.0 * settings.target_volume /
            3.14159265358979323846));
        if (ImGui::DragFloat("Target grain diameter (m)", &diameter, 0.0001f,
                             0.0001f, 10.0f, "%.5f")) {
            const double d = diameter;
            set({{"target_volume", 3.14159265358979323846 * d * d * d / 6.0}});
        }
        int count = static_cast<int>(settings.max_samples);
        if (ImGui::DragInt("Visual sample limit", &count, 1000.0f, 1, 4000000)) {
            set({{"max_samples", count}});
        }
        int budget = static_cast<int>(domain.fluid_visual_samples.working_budget_mb);
        if (ImGui::DragInt("Memory budget (MiB)", &budget, 16.0f, 16, 16384)) {
            set({{"working_budget_mb", budget}});
        }
        if (ImGui::Button("Reset visual samples")) {
            const auto result = rtapi::resetDomainVisualSamples(domain.name);
            error = result.ok ? "" : result.error;
        }
        nlohmann::json report;
        const auto result = rtapi::getDomainVisualSamples(domain.name, report);
        if (result.ok) {
            const auto& stats = report.at("stats");
            ImGui::Text("Physical: %llu   Visual: %llu",
                static_cast<unsigned long long>(stats.at("physical_carriers").get<uint64_t>()),
                static_cast<unsigned long long>(stats.at("samples").get<uint64_t>()));
            ImGui::Text("Sample update: %.2f ms", stats.at("update_ms").get<double>());
            ImGui::TextWrapped("%s", stats.at("status").get<std::string>().c_str());
            const double missing = stats.at("unrepresented_volume").get<double>();
            if (missing > 0.0) {
                ImGui::TextWrapped("Unrepresented volume: %.6g m3; increase the sample limit.",
                                   missing);
            }
        } else {
            ImGui::TextWrapped("%s", result.error.c_str());
        }
    }
    if (!error.empty()) {
        ImGui::TextWrapped("%s", error.c_str());
    }
    ImGui::TreePop();
}

} // namespace RayTrophiSim::Fluid
