#include "UI/GrainMeshLibraryUI.h"
#include "UI/GrainGeneratorUI.h"

#include "Api/RtApi.h"
#include "Fluid/GrainMeshNear.h"
#include "TriangleMesh.h"
#include "scene_data.h"
#include <imgui.h>

#include <algorithm>
#include <string>
#include <vector>

namespace RayTrophiSim::Fluid {

void drawGrainMeshNearControls(SceneData& scene, const SimulationGridDomainDesc& domain) {
    if (!ImGui::CollapsingHeader("Near Grain Meshes##DomainFluid")) {
        return;
    }
    rtapi::FluidSplatGeometryPatch patch;
    bool changed = false;
    bool enabled = domain.fluid_grain_mesh_near_enabled;
    if (ImGui::Checkbox("Mesh Grains Near Camera##DomainFluid", &enabled)) {
        patch.near_mesh_enabled = enabled;
        changed = true;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Draws nearby DEM, MPM and fluid splat particles as mesh instances.\n"
                          "Distant particles stay spheres. While enabled, the primary view uses\n"
                          "one shape per carrier; whitewater refinement stays independent.\n"
                          "This changes appearance only, not the solver or collision shape.");
    }
    if (ImGui::Button("Create Grain / Crystal Library...")) {
        GrainGeneratorUI::requestOpen(domain.name);
    }
    if (domain.fluid_grain_mesh_near_shape != "scene") {
        ImGui::TextWrapped("This scene uses a legacy procedural library. Create an editable "
                           "library above, or use the selected scene objects below.");
        if (ImGui::SmallButton("Use Selected Scene Objects")) {
            patch.near_mesh_shape = "scene";
            changed = true;
        }
    }
    bool source_materials = domain.fluid_grain_mesh_near_source_materials;
    if (ImGui::Checkbox("Use Exemplar Face Materials##NearLibrary", &source_materials)) {
        patch.near_mesh_source_materials = source_materials;
        changed = true;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Every selected object keeps its original face materials and UVs.\n"
                          "This explicitly replaces domain, substance and wet-band materials\n"
                          "on near meshes. Distant spheres keep their existing materials.");
    }
    std::vector<std::string> nodes;
    for (const auto& object : scene.world.objects) {
        const auto mesh = std::dynamic_pointer_cast<TriangleMesh>(object);
        if (!mesh || !mesh->geometry) {
            continue;
        }
        const std::string& name = mesh->nodeName;
        if (name.empty() || name.front() == '[' ||
            scene.isEditorPendingDeleteObjectName(name)) {
            continue;
        }
        nodes.push_back(name);
    }
    std::sort(nodes.begin(), nodes.end());
    nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
    auto sources = domain.fluid_grain_mesh_near_sources;
    if (ImGui::BeginListBox("Exemplar Grains##DomainFluid")) {
        for (const auto& name : nodes) {
            const auto found = std::find(sources.begin(), sources.end(), name);
            bool selected = found != sources.end();
            if (ImGui::Checkbox(name.c_str(), &selected)) {
                if (selected) {
                    sources.push_back(name);
                } else {
                    sources.erase(found);
                }
                patch.near_mesh_sources = sources;
                changed = true;
            }
        }
        ImGui::EndListBox();
    }
    for (const auto& name : sources) {
        if (!std::binary_search(nodes.begin(), nodes.end(), name)) {
            ImGui::TextDisabled("Missing exemplar: %s", name.c_str());
        }
    }
    float distance = domain.fluid_grain_mesh_near_distance;
    if (ImGui::DragFloat("Near Distance (m)##DomainFluid", &distance,
                         0.05f, 0.01f, 1000.0f, "%.2f")) {
        patch.near_mesh_distance = distance;
        changed = true;
    }
    int budget = static_cast<int>(std::min<uint32_t>(
        domain.fluid_grain_mesh_near_budget, 0x7fffffffu));
    if (ImGui::DragInt("Near Mesh Budget##DomainFluid", &budget, 1000.0f, 0, 16000000)) {
        patch.near_mesh_budget = static_cast<uint32_t>(std::max(budget, 0));
        changed = true;
    }
    if (changed) {
        const auto result = rtapi::setFluidSplatGeometry(domain.name, patch);
        if (!result.ok) {
            ImGui::TextDisabled("%s", result.error.c_str());
        }
    }
    GrainMeshNearStats stats;
    if (!readGrainMeshNearStats(domain.name, stats)) {
        return;
    }
    ImGui::TextDisabled("Status: %s", stats.status.c_str());
    if (stats.status == "active") {
        ImGui::TextDisabled("Meshes %llu, spheres %llu, over budget %llu",
            static_cast<unsigned long long>(stats.near_instances),
            static_cast<unsigned long long>(stats.sphere_instances),
            static_cast<unsigned long long>(stats.near_over_budget));
        ImGui::TextDisabled("Exemplars %u (%llu triangles), mesh TLAS slots %llu",
            stats.exemplars, static_cast<unsigned long long>(stats.exemplar_triangles),
            static_cast<unsigned long long>(stats.mesh_slots));
        ImGui::TextDisabled("Mean volume scale %.2f, unmeasured %u",
            stats.mean_volume_scale, stats.exemplars_without_volume);
    }
}

} // namespace RayTrophiSim::Fluid
