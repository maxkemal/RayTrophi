#pragma once
#include "Fluid/MixedGpuStageStats.h"
#include "imgui.h"

namespace RayTrophiSim::Fluid {
inline void drawMixedGpuStages(const MixedGpuStageStats& stats) {
    if (!stats.measured || !ImGui::TreeNode("Mixed GPU stage counters")) {
        return;
    }
    ImGui::Text("Physical substeps: %d; grid solves: %d; liquid solves: %d%s",
        stats.common_substeps, stats.grid_steps, stats.liquid_steps,
        stats.liquid_multirate ? " (own clock)" : "");
    ImGui::TextWrapped("Host times include GPU waits; use GPU kernel timings for device cost.");
    if (ImGui::BeginTable("MixedGpuStages", 4, ImGuiTableFlags_BordersInnerV)) {
        ImGui::TableSetupColumn("Stage");
        ImGui::TableSetupColumn("Host ms");
        ImGui::TableSetupColumn("Dispatches");
        ImGui::TableSetupColumn("Batches / syncs");
        ImGui::TableHeadersRow();
        for (std::size_t i = 0; i < stats.stages.size(); ++i) {
            const auto& entry = stats.stages[i];
            if (entry.visits == 0) {
                continue;
            }
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(mixedGpuStageNames[i]);
            ImGui::TableNextColumn();
            ImGui::Text("%.3f", entry.host_wall_ms);
            ImGui::TableNextColumn();
            ImGui::Text("%u", entry.transfer.dispatch_calls);
            ImGui::TableNextColumn();
            ImGui::Text("%u / %u", entry.transfer.batch_end_calls,
                entry.transfer.synchronize_calls);
        }
        ImGui::EndTable();
    }
    ImGui::TreePop();
}
} // namespace RayTrophiSim::Fluid
