#include "scene_ui_fluid_labels.h"
#include "Fluid/FluidParticleLabels.h"
#include "imgui.h"

namespace ForceFieldUI {

void drawFluidParticleLabels(const RayTrophiSim::SimulationGridDomainState& state) {
    using namespace RayTrophiSim::Fluid;
    if (!ImGui::CollapsingHeader("Particle State Labels")) {
        return;
    }
    const auto report = inspectParticleLabels(&state);
    if (!report.available) {
        ImGui::TextDisabled("No live liquid particle state.");
        return;
    }
    ImGui::TextDisabled("Simulation labels; display still follows substance bindings.");
    if (ImGui::BeginTable("ParticleLabelCounts", 3, ImGuiTableFlags_BordersInnerV)) {
        ImGui::TableSetupColumn("State");
        ImGui::TableSetupColumn("Primary parcels");
        ImGui::TableSetupColumn("Secondary whitewater");
        ImGui::TableHeadersRow();
        for (std::size_t i = 0; i < kParticleLabelCount; ++i) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(particleLabelName(static_cast<ParticleLabel>(i)));
            ImGui::TableNextColumn();
            ImGui::Text("%llu", static_cast<unsigned long long>(report.primary[i]));
            ImGui::TableNextColumn();
            ImGui::Text("%llu", static_cast<unsigned long long>(report.secondary[i]));
        }
        ImGui::EndTable();
    }
    ImGui::Text("Last classification: %.3f ms / %llu parcels", report.last_step.milliseconds,
        static_cast<unsigned long long>(report.last_step.particles));
    ImGui::Text("Classifier: %s", report.last_step.on_gpu ? "GPU" : "CPU");
    ImGui::Text("Bins: %.3f ms / classification: %.3f ms", report.last_step.bin_milliseconds,
        report.last_step.classify_milliseconds);
    ImGui::Text("Resolved in own cell: %llu",
        static_cast<unsigned long long>(report.last_step.center_resolved));
    ImGui::TextDisabled("Unknown includes new/cache parcels and unsupported solid states.");
    ImGui::TextDisabled("Secondary whitewater carries no liquid mass.");
    ImGui::TextDisabled("Mist transfers its remaining mass into an overlapping gas phase.");
}

} // namespace ForceFieldUI
