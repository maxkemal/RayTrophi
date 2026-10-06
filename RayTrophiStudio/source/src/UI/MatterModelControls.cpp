#include "Fluid/MatterModelService.h"
#include "Api/RtApi.h"
#include "Fluid/FluidViewResolver.h"
#include "ParticleSimulation.h"
#include "imgui.h"

#include <algorithm>

namespace RayTrophiSim::Fluid {

void drawMatterPoolControls(const SimulationGridDomainDesc& domain,
                           const SimulationGridDomainState* state) {
    if (domain.type != SimulationDomainType::Matter ||
        !ImGui::CollapsingHeader("Shared Particle Pool")) {
        return;
    }
    ImGui::PushID(domain.name.c_str());
    ImGui::Text("Live particles: %llu / %llu",
        static_cast<unsigned long long>(state && state->valid ? state->particles.size() : 0),
        static_cast<unsigned long long>(domain.fluid_max_particles));
    ImGui::TextWrapped(
        "All liquid emitters share the domain capacity. When free slots are limited, "
        "weights divide them between sources. Unused shares return to the pool; "
        "weights are not reservations or lifetime particle limits.");
    std::vector<rtapi::SimulationFlowSourceInfo> sources;
    const auto listed = rtapi::listSimulationFlowSources(sources);
    if (!listed.ok) {
        ImGui::TextWrapped("%s", listed.error.c_str());
        ImGui::PopID();
        return;
    }
    static std::string error;
    for (auto source : sources) {
        if (source.domain != domain.name || source.phase != "liquid") {
            continue;
        }
        ImGui::PushID(source.name.c_str());
        ImGui::Separator();
        ImGui::TextUnformatted(source.name.c_str());
        ImGui::SetNextItemWidth(-1.0f);
        if (ImGui::DragFloat("Pool weight", &source.particle_pool_weight,
                             0.05f, 0.001f, 1000.0f, "%.3f")) {
            const auto result = rtapi::updateSimulationFlowSource(source.name, source);
            error = result.ok ? "" : result.error;
        }
        ImGui::TextDisabled("Last injection requested / allocated: %d / %d",
            source.pool_requested_particles, source.pool_granted_particles);
        ImGui::PopID();
    }
    if (!error.empty()) {
        ImGui::TextWrapped("%s", error.c_str());
    }
    ImGui::PopID();
}

void drawMatterOutputControls(const SimulationGridDomainDesc& domain) {
    if (domain.type != SimulationDomainType::Matter ||
        !ImGui::CollapsingHeader("Matter Output")) {
        return;
    }
    ImGui::PushID(domain.name.c_str());
    ImGui::TextWrapped(
        "Each substance chooses its appearance independently. "
        "Changing output or scene material does not change physics or combustion.");
    ImGui::TextDisabled("Granular wet strength and appearance: see Pore Water controls.");
    ImGui::TextWrapped(
        "Output bindings are saved independently of emitters. "
        "Renaming or deleting an emitter leaves its output binding available here.");

    std::vector<rtapi::SimulationFlowSourceInfo> sources;
    const auto source_result = rtapi::listSimulationFlowSources(sources);
    std::vector<std::string> substances;
    for (const auto& binding : domain.fluid_substance_materials) {
        substances.push_back(binding.substance);
    }
    if (source_result.ok) {
        for (const auto& source : sources) {
            if (source.domain != domain.name || source.phase != "liquid" ||
                source.fluid_substance.empty()) {
                continue;
            }
            if (std::find(substances.begin(), substances.end(), source.fluid_substance) ==
                substances.end()) {
                substances.push_back(source.fluid_substance);
            }
        }
    } else {
        ImGui::TextWrapped("%s", source_result.error.c_str());
    }

    const auto materials = rtapi::listMaterials();
    const auto plan = resolveFluidViews(domain, {});
    static std::string error;
    for (const auto& substance : substances) {
        ImGui::PushID(substance.c_str());
        ImGui::Separator();
        ImGui::TextUnformatted(substance.c_str());
        const auto found = std::find_if(
            domain.fluid_substance_materials.begin(), domain.fluid_substance_materials.end(),
            [&](const auto& binding) { return binding.substance == substance; });
        const bool has_binding = found != domain.fluid_substance_materials.end();
        // Snapshot before API calls: creating a binding may reallocate the table.
        const auto representation = found != domain.fluid_substance_materials.end()
            ? found->representation : SubstanceRepresentation::Inherit;
        const int material_id = found != domain.fluid_substance_materials.end()
            ? found->material_id : -1;
        int selected = static_cast<int>(representation);
        const char* choices[] = {"Domain fallback", "Splat", "SDF", "Fog"};
        const char* names[] = {"inherit", "splat", "sdf", "fog"};
        ImGui::SetNextItemWidth(-1.0f);
        if (ImGui::Combo("Output", &selected, choices, 4)) {
            const std::string route = names[selected];
            const auto result = rtapi::setFluidSubstanceMaterial(
                domain.name, substance, "", &route, nullptr, nullptr, nullptr, nullptr);
            error = result.ok ? "" : result.error;
        }

        std::string current = "Domain material";
        for (const auto& material : materials) {
            if (static_cast<int>(material.id) == material_id) {
                current = material.name;
                break;
            }
        }
        ImGui::SetNextItemWidth(-1.0f);
        if (ImGui::BeginCombo("Scene material", current.c_str())) {
            if (ImGui::Selectable("Domain material", material_id < 0)) {
                const auto result = rtapi::setFluidSubstanceMaterial(
                    domain.name, substance, "dielectric",
                    nullptr, nullptr, nullptr, nullptr, nullptr);
                error = result.ok ? "" : result.error;
            }
            for (const auto& material : materials) {
                if (ImGui::Selectable(material.name.c_str(),
                                      static_cast<int>(material.id) == material_id)) {
                    const auto result = rtapi::setFluidSubstanceMaterial(
                        domain.name, substance, material.name,
                        nullptr, nullptr, nullptr, nullptr, nullptr);
                    error = result.ok ? "" : result.error;
                }
            }
            ImGui::EndCombo();
        }
        if (ImGui::Button("Create material from substance")) {
            std::string created;
            auto result = rtapi::createMaterial(
                "substance:" + substance, substance + " Material", created);
            if (result.ok) {
                result = rtapi::setFluidSubstanceMaterial(
                    domain.name, substance, created,
                    nullptr, nullptr, nullptr, nullptr, nullptr);
            }
            error = result.ok ? "" : result.error;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(
                "Creates an editable scene material from the catalogue substance.\n"
                "It does not change output, temperature or physical behaviour.");
        }
        if (has_binding && ImGui::Button("Remove output binding")) {
            const auto result = rtapi::setFluidSubstanceMaterial(
                domain.name, substance, "",
                nullptr, nullptr, nullptr, nullptr, nullptr);
            error = result.ok ? "" : result.error;
        }
        if (has_binding && ImGui::IsItemHovered()) {
            ImGui::SetTooltip(
                "Restores domain defaults for this substance.\n"
                "An emitter referencing it keeps the substance listed here.");
        }
        ImGui::TextDisabled("Substance output: %s",
            fluidViewName(plan.viewForTag(substanceTag(substance))));
        for (const auto& source : sources) {
            if (source.domain == domain.name && source.phase == "liquid" &&
                source.fluid_substance == substance) {
                ImGui::TextDisabled("Emitter: %s / initial model: %s",
                    source.name.c_str(), source.initial_constitutive_model.c_str());
            }
        }
        ImGui::PopID();
    }
    if (substances.empty()) {
        ImGui::TextDisabled("Assign a substance to a liquid emitter to configure its output.");
    }
    ImGui::TextWrapped(
        "Spray, foam, bubble and mist can use the existing state-label output routes. "
        "Fog uses the domain volume shader; a surface material does not replace it.");
    if (!error.empty()) {
        ImGui::TextWrapped("%s", error.c_str());
    }
    ImGui::PopID();
}

void drawMatterModelSummary(const SimulationGridDomainDesc& domain,
                           const SimulationGridDomainState* state) {
    if (domain.type != SimulationDomainType::Matter ||
        !ImGui::CollapsingHeader("Active Matter")) {
        return;
    }
    MatterTransferFrame frame;
    std::string error;
    if (!inspectMatterModels(domain, state, false, frame, error)) {
        ImGui::TextWrapped("%s", error.c_str());
        return;
    }
    if (!state || !state->valid) {
        ImGui::TextDisabled("No synchronized particle state.");
        return;
    }
    static const char* names[] = {"Fluid", "Granular", "Elastic", "Unresolved"};
    if (ImGui::BeginTable("##MatterModels", 3, ImGuiTableFlags_Borders)) {
        ImGui::TableSetupColumn("Model");
        ImGui::TableSetupColumn("Particles");
        ImGui::TableSetupColumn("Mass (kg)");
        ImGui::TableHeadersRow();
        for (std::size_t model = 0; model < frame.totals.size(); ++model) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextUnformatted(names[model]);
            ImGui::TableSetColumnIndex(1);
            ImGui::Text("%llu", static_cast<unsigned long long>(frame.totals[model].particles));
            ImGui::TableSetColumnIndex(2);
            ImGui::Text("%.6g", frame.totals[model].mass_kg);
        }
        ImGui::EndTable();
    }
    if (frame.totals[0].particles > 0 && frame.totals[1].particles > 0) {
        ImGui::TextWrapped("%s", state->fluid_stats.gpu_status.c_str());
        if (state->fluid_stats.mixed_model_step) {
            ImGui::Text("Common substeps: %d / Contact pairs: %llu",
                state->fluid_stats.mixed_common_substeps,
                static_cast<unsigned long long>(state->fluid_stats.mixed_contact_pairs));
        }
    }
    if (frame.totals[1].particles > 0) {
        const auto& stats = state->fluid_stats;
        if (stats.granular_load_measured) {
            ImGui::Text("Granular load estimate: %.6g Pa / Required Young: %.6g Pa",
                stats.granular_overburden_pressure, stats.granular_young_modulus_for_load);
            ImGui::Text("Wave / strain substeps: %d / %d",
                stats.granular_wave_substeps, stats.granular_strain_substeps);
            ImGui::TextWrapped("Extent-based rho*g*h estimate; contact pressure and "
                "repose are separate acceptance tests.");
        } else {
            ImGui::TextDisabled("Granular load estimate has not been published.");
        }
    }
}

} // namespace RayTrophiSim::Fluid
