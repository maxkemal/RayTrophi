#include "MatterMaterialResolutionUI.h"
#include "Fluid/MatterMaterialResolution.h"
#include "imgui.h"

namespace RayTrophiSim::Fluid {

void drawMatterMaterialResolution(const SimulationGridDomainDesc& domain,
    const SimulationGridDomainState* state) {
    if (!ImGui::TreeNode("Material resolution")) {
        return;
    }
    nlohmann::json report;
    std::string error;
    if (!inspectMatterMaterialResolution(domain, state, report, error)) {
        ImGui::TextWrapped("%s", error.c_str());
        ImGui::TreePop();
        return;
    }
    const auto& viscosity = report.at("viscosity");
    ImGui::TextWrapped("Default: %s / Requested: %s, %s",
        report.at("default_substance").get_ref<const std::string&>().c_str(),
        report.at("requested_model").get_ref<const std::string&>().c_str(),
        report.at("requested_transport").get_ref<const std::string&>().c_str());
    ImGui::Text("Requested backend: %s",
        report.at("backend_requested").get_ref<const std::string&>().c_str());
    ImGui::TextDisabled("Configuration and carrier snapshot; execution is not measured here.");
    const auto& mechanics = report.at("phase_mechanics");
    ImGui::Text("Particle definitions: solid %s / liquid %s",
        mechanics.at("solid").get_ref<const std::string&>().c_str(),
        mechanics.at("liquid").get_ref<const std::string&>().c_str());
    if (!mechanics.at("solid_deformation_defined").get<bool>()) {
        ImGui::TextDisabled("No deformable solid law is defined by this particle profile.");
    }
    ImGui::Text("Viscosity [m^2/s]: requested %.6g / policy %.6g / configured %.6g",
        viscosity.at("requested_m2_s").get<double>(),
        viscosity.at("policy_effective_m2_s").get<double>(),
        viscosity.at("configured_m2_s").get<double>());
    ImGui::Text("Viscosity policy: %s",
        viscosity.at("reason").get_ref<const std::string&>().c_str());
    if (!viscosity.at("configuration_matches_policy").get<bool>()) {
        ImGui::TextWrapped("Configured viscosity differs from the current substance policy.");
    }
    ImGui::TextDisabled("Thermal per-particle viscosity is separate from this domain base.");
    ImGui::TextDisabled("MPM material parameters have shared domain scope.");
    if (ImGui::BeginTable("MaterialCarriers", 5,
            ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Substance");
        ImGui::TableSetupColumn("Model / transport");
        ImGui::TableSetupColumn("Particles");
        ImGui::TableSetupColumn("Mean T [K]");
        ImGui::TableSetupColumn("Mean solid share");
        ImGui::TableHeadersRow();
        for (const auto& carrier : report.at("carriers")) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextUnformatted(carrier.at("substance").get_ref<const std::string&>().c_str());
            if (carrier.at("identity_resolution") == "unknown_tag_domain_fallback") {
                ImGui::TextWrapped("Unknown tag: using domain default");
            }
            if (ImGui::IsItemHovered()) {
                const auto phase_text = carrier.at("phase_counts").dump();
                const auto basis_text = carrier.at("selection_basis_counts").dump();
                const auto phase_basis = carrier.at("phase_basis_counts").dump();
                ImGui::SetTooltip("Phase counts: %s\nPhase basis: %s\nSelection basis: %s",
                    phase_text.c_str(), phase_basis.c_str(), basis_text.c_str());
            }
            ImGui::TableSetColumnIndex(1);
            ImGui::Text("%s / %s",
                carrier.at("model").get_ref<const std::string&>().c_str(),
                carrier.at("transport").get_ref<const std::string&>().c_str());
            if (carrier.at("model_mismatch_particles").get<std::size_t>() > 0) {
                ImGui::TextWrapped("%llu differ from current phase selection",
                    carrier.at("model_mismatch_particles").get<unsigned long long>());
            }
            const auto& thermal = carrier.at("elastic_thermal_response");
            if (thermal.at("known_particles").get<std::size_t>() > 0 && ImGui::IsItemHovered()) {
                const auto response = thermal.at("at_minimum_softening").dump(2);
                ImGui::SetTooltip("At minimum softening, before stability caps/damage:\n%s",
                    response.c_str());
            }
            ImGui::TableSetColumnIndex(2);
            ImGui::Text("%llu", carrier.at("particles").get<unsigned long long>());
            ImGui::TableSetColumnIndex(3);
            if (carrier.at("mean_temperature_k").is_null()) {
                ImGui::TextDisabled("Unknown");
            } else {
                ImGui::Text("%.4g (%llu known)",
                    carrier.at("mean_temperature_k").get<double>(),
                    carrier.at("temperature_known_particles").get<unsigned long long>());
            }
            ImGui::TableSetColumnIndex(4);
            if (carrier.at("mean_fusion_solid_fraction").is_null()) {
                ImGui::TextDisabled("Unknown");
            } else {
                ImGui::Text("%.4g (%llu known)",
                    carrier.at("mean_fusion_solid_fraction").get<double>(),
                    carrier.at("fusion_known_particles").get<unsigned long long>());
            }
        }
        ImGui::EndTable();
    }
    if (report.contains("transport_ready") && !report.at("transport_ready").get<bool>()) {
        ImGui::TextWrapped("%s",
            report.at("transport_reason").get_ref<const std::string&>().c_str());
    }
    for (const auto& note : report.at("notes")) {
        ImGui::TextWrapped("%s", note.get_ref<const std::string&>().c_str());
    }
    ImGui::TreePop();
}

} // namespace RayTrophiSim::Fluid
