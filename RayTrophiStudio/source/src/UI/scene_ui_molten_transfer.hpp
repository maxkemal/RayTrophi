#pragma once

#include <algorithm>

namespace MoltenTransferUI {
inline void draw(SceneData& scene, RayTrophiSim::ParticleSimulationSystem& runtime,
                 RayTrophiSim::ParticleColliderDesc& collider) {
    const auto& profile = RayTrophiSim::findSubstance(collider.msf_substance);
    if (!profile.meltable) return;
    ImGui::SeparatorText("Molten APIC Transfer");
    bool changed = false;
    changed |= ImGui::Checkbox("Local Molten Surface Flow##MsfMeltFlow",
                               &collider.msf_melt_flow_enabled);
    ImGui::BeginDisabled(!collider.msf_melt_flow_enabled);
    changed |= ImGui::DragFloat("Maximum Height Loss##MsfMeltFlow",
        &collider.msf_melt_height_loss, 0.01f, 0.0f, 0.92f, "%.2f");
    changed |= ImGui::DragFloat("Maximum Pool Spread##MsfMeltFlow",
        &collider.msf_melt_spread, 0.02f, 0.0f, 2.5f, "%.2f");
    ImGui::TextDisabled("Local pool conserves solid + untransferred molten mass.\n"
                        "Pyrolysis and accepted APIC mass shrink the source volume.");
    const bool mesh_sdf = collider.source_mode ==
        RayTrophiSim::ParticleColliderSourceMode::ObjectMeshSDF;
    ImGui::BeginDisabled(!mesh_sdf);
    changed |= ImGui::Checkbox("Refresh SDF While Melting##MsfMeltSdf",
                               &collider.msf_melt_sdf_refresh);
    ImGui::BeginDisabled(!collider.msf_melt_sdf_refresh);
    int refresh_interval = static_cast<int>(collider.msf_melt_sdf_revision_interval);
    if (ImGui::DragInt("SDF Refresh Interval##MsfMeltSdf", &refresh_interval,
                       1.0f, 1, 60, "%d state updates")) {
        collider.msf_melt_sdf_revision_interval =
            static_cast<uint32_t>(std::clamp(refresh_interval, 1, 60));
        changed = true;
    }
    changed |= ImGui::DragFloat("SDF Shape Threshold##MsfMeltSdf",
        &collider.msf_melt_sdf_change_threshold, 0.002f, 0.001f, 0.25f, "%.3f");
    ImGui::EndDisabled();
    ImGui::TextDisabled(mesh_sdf
        ? "Throttled asynchronous cook; never rebuilds every render frame."
        : "Dynamic refresh requires Mesh SDF collider mode.");
    ImGui::EndDisabled();
    ImGui::EndDisabled();
    changed |= ImGui::Checkbox("Automatic Transfer##MsfAutoTransfer",
                               &collider.msf_auto_transfer);
    ImGui::BeginDisabled(!collider.msf_auto_transfer);
    const char* preview = collider.msf_transfer_domain.empty()
        ? "Select Fluid domain" : collider.msf_transfer_domain.c_str();
    if (ImGui::BeginCombo("Target Fluid Domain##MsfTransferDomain", preview)) {
        for (const auto& domain : runtime.gridDomains()) {
            if (domain.type != RayTrophiSim::SimulationDomainType::Fluid) continue;
            const bool selected = collider.msf_transfer_domain == domain.name;
            if (ImGui::Selectable(domain.name.c_str(), selected)) {
                collider.msf_transfer_domain = domain.name;
                changed = true;
            }
            if (selected) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    changed |= ImGui::DragFloat("Transfer Rate (kg/s)##MsfTransfer", &collider.msf_transfer_rate_kg_s,
                                0.005f, 0.0f, 10.0f, "%.3f");
    changed |= ImGui::DragFloat("Start Reservoir (kg)##MsfTransfer", &collider.msf_transfer_min_mass_kg,
                                0.001f, 0.0f, 10.0f, "%.3f");
    changed |= ImGui::DragFloat("Particles / kg##MsfTransfer", &collider.msf_transfer_particles_per_kg,
                                16.0f, 1.0f, 100000.0f, "%.0f");
    int batch = static_cast<int>(collider.msf_transfer_max_batch_particles);
    if (ImGui::DragInt("Max Batch Particles##MsfTransfer", &batch, 1.0f, 1, 4096)) {
        collider.msf_transfer_max_batch_particles = static_cast<uint32_t>(batch);
        changed = true;
    }
    changed |= ImGui::DragFloat3("Initial Velocity##MsfTransfer",
        &collider.msf_transfer_velocity.x, 0.01f, -20.0f, 20.0f, "%.2f");
    ImGui::TextDisabled("Target is forced to Surface SDF on first transfer.\n"
                        "Plastic uses viscous melt; iron/steel stay low-viscosity and non-burning.\n"
                        "Hard batch cap protects Vulkan/TLAS and APIC working sets.");
    ImGui::EndDisabled();
    const auto& stats = runtime.moltenMassTransferStats();
    ImGui::TextDisabled("queued %llu | completed %llu | APIC %.4f kg | %llu particles",
        static_cast<unsigned long long>(stats.queued),
        static_cast<unsigned long long>(stats.completed),
        stats.transferred_mass,
        static_cast<unsigned long long>(stats.spawned_particles));
    if (stats.deferred_no_domain || stats.deferred_no_capacity) {
        ImGui::TextColored(ImVec4(1.0f, 0.68f, 0.25f, 1.0f),
            "Deferred: domain/chemistry %llu | capacity %llu",
            static_cast<unsigned long long>(stats.deferred_no_domain),
            static_cast<unsigned long long>(stats.deferred_no_capacity));
    }
    if (changed) {
        collider.msf_transfer_rate_kg_s = std::max(0.0f, collider.msf_transfer_rate_kg_s);
        collider.msf_transfer_min_mass_kg = std::max(0.0f, collider.msf_transfer_min_mass_kg);
        collider.msf_transfer_particles_per_kg = std::clamp(
            collider.msf_transfer_particles_per_kg, 1.0f, 100000.0f);
        collider.msf_melt_height_loss = std::clamp(collider.msf_melt_height_loss, 0.0f, 0.92f);
        collider.msf_melt_spread = std::clamp(collider.msf_melt_spread, 0.0f, 2.5f);
        collider.msf_melt_sdf_change_threshold = std::clamp(
            collider.msf_melt_sdf_change_threshold, 0.001f, 0.25f);
        scene.clearSimFrameCache();
    }
}
} // namespace MoltenTransferUI
