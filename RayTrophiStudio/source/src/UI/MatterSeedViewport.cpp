#include "MatterSeedViewport.h"
#include "Api/MatterSeedSourcesApi.h"
#include "ImGuizmo.h"
#include "MatterSeedSourcesUI.h"
#include "ParticleSimulation.h"
#include "scene_data.h"
#include "scene_ui.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

namespace RayTrophiSim::Fluid {
namespace {
std::map<std::string, std::string> errors;
}

uint64_t selectedMatterSeed(const SimulationGridDomainDesc &domain) {
    return domain.matter_seed_selected_source;
}

void selectMatterSeed(const std::string &domain, uint64_t id) {
    nlohmann::json result;
    const auto status =
        rtapi::matterSeedSources(domain, "select", id, nlohmann::json::object(), result);
    errors[domain] = status.ok ? std::string() : status.error;
}

void drawMatterSeedAuthoring(
    UIContext &ctx, const std::function<bool(const Vec3 &, ImVec2 &)> &project,
    const std::function<void(const Vec3 &, const Vec3 &, ImU32, float)> &draw_box) {
    const auto runtime = ctx.scene.getParticleSimulationSystem();
    if (!runtime) {
        return;
    }
    for (const auto &domain : runtime->gridDomains()) {
        if (domain.type == SimulationDomainType::Matter && domain.matter_seed_sources_enabled) {
            drawMatterSeedSourceBounds(domain, draw_box);
            drawMatterSeedLabels(ctx, domain, project);
        }
    }
}

void drawMatterSeedLabels(UIContext &ctx, const SimulationGridDomainDesc &domain,
                          const std::function<bool(const Vec3 &, ImVec2 &)> &project) {
    if (!domain.matter_seed_sources_enabled) {
        return;
    }
    std::vector<ImVec2> placed;
    for (const auto &source : domain.matter_seed_sources) {
        if (!source.enabled) {
            continue;
        }
        Vec3 low = source.bounds_min;
        Vec3 high = source.bounds_max;
        if (!source.mesh_object.empty()) {
            const auto mesh = ctx.scene.resolveMeshVolumeForSimulation(source.mesh_object);
            if (!mesh || mesh->empty()) {
                continue;
            }
            low = mesh->boundsMin();
            high = mesh->boundsMax();
        }
        const auto label = source.name + " | " + source.substance;
        const auto textSize = ImGui::CalcTextSize(label.c_str());
        const ImVec2 labelSize(textSize.x + 8, textSize.y + 4);
        ImVec2 screen;
        bool visible = false;
        // Place the label outside the projected upper-right box corner, rather
        // than over the move/scale handles at the source center.
        for (int corner = 0; corner < 8; ++corner) {
            ImVec2 point;
            if (project(Vec3((corner & 1) ? high.x : low.x, (corner & 2) ? high.y : low.y,
                             (corner & 4) ? high.z : low.z),
                        point)) {
                if (!visible) {
                    screen = point;
                    visible = true;
                } else {
                    screen.x = std::max(screen.x, point.x);
                    screen.y = std::min(screen.y, point.y);
                }
            }
        }
        if (!visible && !project((low + high) * 0.5f, screen)) {
            continue;
        }
        screen.x += 10;
        screen.y -= labelSize.y + 8;
        const auto display = ImGui::GetIO().DisplaySize;
        screen.x = std::clamp(screen.x, 8.0f, std::max(8.0f, display.x - labelSize.x - 8));
        screen.y = std::clamp(screen.y, 8.0f, std::max(8.0f, display.y - labelSize.y - 8));
        for (const auto &previous : placed) {
            if (std::abs(screen.x - previous.x) < labelSize.x + 12 &&
                std::abs(screen.y - previous.y) < labelSize.y + 4) {
                screen.y = previous.y + labelSize.y + 6;
            }
        }
        placed.push_back(screen);
        const bool active = selectedMatterSeed(domain) == source.id;
        // A selected box's label is purely informational. It must never steal
        // mouse input from its handles, even when a small box projects to a dot.
        const bool passThrough =
            (active && source.mesh_object.empty()) || ImGuizmo::IsOver() || ImGuizmo::IsUsing();
        const auto key = "SeedLabel##" + domain.name + ":" + std::to_string(source.id);
        ImGui::SetNextWindowPos(screen, ImGuiCond_Always);
        ImGui::SetNextWindowBgAlpha(active ? 0.25f : 0.14f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(4, 2));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowMinSize, ImVec2(0, 0));
        ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0, 0, 0, 0));
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(0.4f, 0.55f, 0.7f, 0.2f));
        ImGui::PushStyleColor(ImGuiCol_Text, active ? ImVec4(1.0f, 0.8f, 0.4f, 0.95f)
                                                    : ImGui::GetStyleColorVec4(ImGuiCol_Text));
        auto flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                     ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoNav |
                     ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoMove;
        if (passThrough) {
            flags |= ImGuiWindowFlags_NoInputs;
        }
        if (ImGui::Begin(key.c_str(), nullptr, flags)) {
            if (ImGui::Selectable(label.c_str(), active) && !passThrough) {
                selectMatterSeed(domain.name, source.id);
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    source.mesh_object.empty()
                        ? "Select layer. Use Move / Scale to edit the box; Generate updates "
                          "particles."
                        : "Mesh volume: edit the source object's transform to change its shape.");
            }
            if (!source.mesh_object.empty() && ImGui::BeginPopupContextItem("Source mesh")) {
                if (ImGui::MenuItem("Edit source mesh")) {
                    const auto result = rtapi::selectObject(source.mesh_object);
                    errors[domain.name] = result.ok ? std::string() : result.error;
                }
                ImGui::EndPopup();
            }
        }
        ImGui::End();
        ImGui::PopStyleColor(3);
        ImGui::PopStyleVar(2);
    }
}

bool manipulateMatterSeed(const SimulationGridDomainDesc &domain, const float *view,
                          const float *projection, bool scale, bool blocked) {
    const auto id = selectedMatterSeed(domain);
    const auto source =
        std::find_if(domain.matter_seed_sources.begin(), domain.matter_seed_sources.end(),
                     [id](const auto &item) { return item.id == id && item.enabled; });
    if (!domain.matter_seed_sources_enabled || source == domain.matter_seed_sources.end()) {
        return false;
    }
    if (!source->mesh_object.empty()) {
        return true;
    }
    const auto center = (source->bounds_min + source->bounds_max) * 0.5f;
    const auto size = source->bounds_max - source->bounds_min;
    float matrix[16] = {size.x, 0, 0,      0, 0,        size.y,   0,        0,
                        0,      0, size.z, 0, center.x, center.y, center.z, 1};
    if (!blocked &&
        ImGuizmo::Manipulate(view, projection, scale ? ImGuizmo::SCALE : ImGuizmo::TRANSLATE,
                             ImGuizmo::LOCAL, matrix)) {
        const Vec3 newCenter(matrix[12], matrix[13], matrix[14]);
        const Vec3 newSize(std::max(0.0001f, std::abs(matrix[0])),
                           std::max(0.0001f, std::abs(matrix[5])),
                           std::max(0.0001f, std::abs(matrix[10])));
        const Vec3 low = newCenter - newSize * 0.5f;
        const Vec3 high = newCenter + newSize * 0.5f;
        nlohmann::json result;
        const auto status = rtapi::matterSeedSources(
            domain.name, "update", id,
            {{"bounds_min", {low.x, low.y, low.z}}, {"bounds_max", {high.x, high.y, high.z}}},
            result);
        errors[domain.name] = status.ok ? std::string() : status.error;
    }
    const auto error = errors.find(domain.name);
    if (error != errors.end() && !error->second.empty()) {
        ImGui::GetForegroundDrawList()->AddText(ImVec2(20, 70), IM_COL32(255, 110, 75, 255),
                                                error->second.c_str());
    }
    return true;
}

bool manipulateSelectedMatterSeed(UIContext &ctx, const float *view, const float *projection,
                                  bool blocked) {
    auto &selection = ctx.selection;
    const auto &item = selection.selected;
    if (item.type != SelectableType::SimulationDomain || item.particle_system_index < 0 ||
        item.particle_system_index >= static_cast<int>(ctx.scene.particle_systems.size())) {
        return false;
    }
    const auto &system = ctx.scene.particle_systems[item.particle_system_index];
    if (!system.runtime || item.simulation_domain_index < 0 ||
        item.simulation_domain_index >= static_cast<int>(system.runtime->gridDomains().size())) {
        return false;
    }
    const auto &domain = system.runtime->gridDomains()[item.simulation_domain_index];
    if (!selectedMatterSeed(domain)) {
        return false;
    }
    if (!blocked && !ImGui::GetIO().WantTextInput) {
        if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
            selectMatterSeed(domain.name, 0);
            return false;
        }
        if (ImGui::IsKeyPressed(ImGuiKey_G)) {
            selection.transform_mode = TransformMode::Translate;
        } else if (ImGui::IsKeyPressed(ImGuiKey_S) && !ImGui::GetIO().KeyCtrl &&
                   !ImGui::GetIO().KeyShift) {
            selection.transform_mode = TransformMode::Scale;
        }
    }
    return manipulateMatterSeed(domain, view, projection,
                                selection.transform_mode == TransformMode::Scale, blocked);
}
} // namespace RayTrophiSim::Fluid
