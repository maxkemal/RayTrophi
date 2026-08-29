#pragma once

#include "NodeSystem/Graph.h"
#include "imgui.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace TerrainNodesV2 {

inline bool terrainPinConnected(const NodeSystem::GraphBase& graph, uint32_t pinId) {
    for (const auto& link : graph.links)
        if (link.startPinId == pinId || link.endPinId == pinId) return true;
    return false;
}

inline bool drawTerrainNodePortPanel(NodeSystem::NodeBase& node,
                                     const NodeSystem::GraphBase& graph) {
    bool hasOptional = false;
    for (const auto& pin : node.inputs)
        hasOptional |= pin.exposure != NodeSystem::PinExposure::Primary;
    for (const auto& pin : node.outputs)
        hasOptional |= pin.exposure != NodeSystem::PinExposure::Primary;
    if (!hasOptional) return false;

    ImGui::Spacing();
    if (!ImGui::CollapsingHeader("Ports", ImGuiTreeNodeFlags_DefaultOpen)) return false;
    ImGui::TextDisabled("Expose only the data this graph genuinely uses.");

    bool changed = false;
    const auto drawDirection = [&](const char* label, std::vector<NodeSystem::Pin>& pins) {
        bool any = false;
        for (const auto& pin : pins)
            any |= pin.exposure != NodeSystem::PinExposure::Primary;
        if (!any) return;

        ImGui::TextDisabled("%s", label);
        for (auto& pin : pins) {
            if (pin.exposure == NodeSystem::PinExposure::Primary) continue;
            const bool connected = terrainPinConnected(graph, pin.id);
            bool visible = !pin.hidden || connected;
            ImGui::PushID(static_cast<int>(pin.id));
            if (connected) ImGui::BeginDisabled();
            if (ImGui::Checkbox(pin.name.c_str(), &visible)) {
                pin.hidden = !visible;
                changed = true;
            }
            if (connected) ImGui::EndDisabled();
            if (ImGui::IsItemHovered()) {
                const char* tier = pin.exposure == NodeSystem::PinExposure::Diagnostic
                    ? "Diagnostic" : "Optional";
                ImGui::SetTooltip("%s%s%s\nPort key: %s%s",
                    tier,
                    pin.section.empty() ? "" : " / ",
                    pin.section.c_str(),
                    pin.stableKey.c_str(),
                    connected ? "\nConnected ports remain visible." : "");
            }
            ImGui::PopID();
        }
    };

    drawDirection("Inputs", node.inputs);
    drawDirection("Outputs", node.outputs);
    return changed;
}

} // namespace TerrainNodesV2
