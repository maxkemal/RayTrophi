#include "UI/SourceSplatGeometryUI.h"
#include "Fluid/SourceSplatGeometryJson.h"
#include "Api/RtApi.h"
#include "Api/SourceSplatGeometryApi.h"
#include "TriangleMesh.h"
#include "scene_data.h"
#include <imgui.h>
#include <algorithm>
#include <unordered_map>

namespace RayTrophiSim::Fluid {
namespace {
struct Editor {
    SourceSplatGeometry draft;
    std::string committed;
    std::string error;
    ImGuiTextFilter filter;
};
std::unordered_map<std::string, Editor> editors;
}

void drawSubstanceSplatGeometry(SceneData& scene, const std::string& domain,
    const std::string& substance) {
    SourceSplatGeometry applied;
    const auto read = rtapi::getSubstanceSplatGeometry(domain, substance, applied);
    if (!read.ok) {
        ImGui::TextWrapped("%s", read.error.c_str());
        return;
    }
    const std::string key = domain + "\n" + substance;
    ImGui::PushID(key.c_str());
    auto& editor = editors[key];
    const std::string snapshot = nlohmann::json(applied).dump();
    if (editor.committed != snapshot) {
        editor.committed = snapshot;
        editor.draft = applied;
        editor.error.clear();
    }
    auto& draft = editor.draft;
    ImGui::SeparatorText("Splat Geometry");
    ImGui::TextWrapped("All %s splat particles use this geometry, independent of emitters.",
        substance.c_str());
    int geometry = draft.mode == "sphere" ? 0 : 1;
    const char* choices[] = {"Sphere", "Scene Objects (multiple selection)"};
    if (ImGui::Combo("Represented by", &geometry, choices, 2)) {
        draft.mode = geometry == 0 ? "sphere" : "scene_object";
    }
    if (geometry == 1) {
        std::vector<std::string> nodes;
        for (const auto& object : scene.world.objects) {
            const auto mesh = std::dynamic_pointer_cast<TriangleMesh>(object);
            if (!mesh || !mesh->geometry || mesh->nodeName.empty() ||
                mesh->nodeName.front() == '[' ||
                scene.isEditorPendingDeleteObjectName(mesh->nodeName)) {
                continue;
            }
            nodes.push_back(mesh->nodeName);
        }
        std::sort(nodes.begin(), nodes.end());
        nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
        editor.filter.Draw("Filter scene objects");
        ImGui::Text("Selected geometries: %zu", draft.objects.size());
        if (ImGui::BeginListBox("##GeometryObjects",
                ImVec2(-1.0f, ImGui::GetTextLineHeightWithSpacing() * 6))) {
            for (const auto& node : nodes) {
                if (!editor.filter.PassFilter(node.c_str())) {
                    continue;
                }
                const auto found = std::find(draft.objects.begin(), draft.objects.end(), node);
                bool selected = found != draft.objects.end();
                if (ImGui::Checkbox(node.c_str(), &selected)) {
                    if (selected) {
                        draft.objects.push_back(node);
                    } else {
                        draft.objects.erase(found);
                    }
                }
            }
            ImGui::EndListBox();
        }
        ImGui::TextWrapped("Each particle uses a stable choice from these scene objects. "
            "Create objects independently with Add / Procedural Generator.");
        for (const auto& node : draft.objects) {
            if (!std::binary_search(nodes.begin(), nodes.end(), node)) {
                ImGui::TextWrapped("Missing geometry: %s", node.c_str());
                ImGui::PushID(node.c_str());
                if (ImGui::SmallButton("Remove missing geometry")) {
                    draft.objects.erase(std::find(draft.objects.begin(), draft.objects.end(), node));
                    ImGui::PopID();
                    break;
                }
                ImGui::PopID();
            }
        }
        ImGui::Checkbox("Use Object Face Materials", &draft.source_materials);
        bool near_only = draft.mode == "mesh_grains";
        if (ImGui::Checkbox("Use spheres beyond distance", &near_only)) {
            draft.mode = near_only ? "mesh_grains" : "scene_object";
        }
        if (near_only) {
            ImGui::DragFloat("Mesh Distance (m)", &draft.near_distance,
                0.1f, 0.01f, 1000.0f, "%.2f");
            int budget = static_cast<int>(std::min(draft.mesh_budget, 16000000u));
            if (ImGui::DragInt("Mesh Budget", &budget, 1000.0f, 0, 16000000)) {
                draft.mesh_budget = static_cast<uint32_t>(std::max(budget, 0));
            }
        } else {
            ImGui::TextDisabled("Selected scene geometries are used at all distances.");
        }
    }
    const bool changed = nlohmann::json(draft).dump() != editor.committed;
    ImGui::BeginDisabled(!changed);
    if (ImGui::Button("Apply Geometry")) {
        const auto result = rtapi::setSubstanceSplatGeometry(domain, substance, draft);
        editor.error = result.ok ? "" : result.error;
    }
    ImGui::SameLine();
    if (ImGui::Button("Revert")) {
        draft = applied;
        editor.error.clear();
    }
    ImGui::EndDisabled();
    if (changed) {
        ImGui::TextDisabled("Selection is pending until Apply Geometry.");
    }
    if (!editor.error.empty()) {
        ImGui::TextWrapped("%s", editor.error.c_str());
    }
    ImGui::PopID();
}
}
