#include "Fluid/MatterPhaseConfig.h"
#include "Fluid/FluidGridResourceBudget.h"
#include "imgui.h"
#include "DomainPanelWidgets.h"

#include <map>

namespace RayTrophiSim::Fluid {

bool drawPhaseGridControls(SimulationGridDomainDesc& domain) {
    if (domain.type != SimulationDomainType::Matter ||
        !ImGui::CollapsingHeader("Gas / Liquid Phase Grids")) {
        return false;
    }
    struct Draft {
        Vec3 lo;
        Vec3 hi;
        float voxel = 0.1f;
        bool inherit = true;
        bool initialized = false;
        uint64_t configuration = 0;
        Vec3 origin;
        std::string error;
    };
    // Unapplied edits, keyed by domain name and phase: a domain list
    // reallocation moves the object, a name keeps naming the same domain.
    static std::map<std::string, Draft> drafts;
    ImGui::PushID(domain.name.c_str());
    bool changed = false;
    const Vec3 origin = logicalGridOrigin(domain);
    const auto layouts = previewPhaseLayouts(domain);
    ImGui::TextWrapped("Independent grids follow the same domain. Changes restart the simulation.");
    for (GridPhase phase : {GridPhase::Gas, GridPhase::Liquid}) {
        const bool gas = phase == GridPhase::Gas;
        const auto& setting = gas ? domain.gas_phase_grid : domain.liquid_phase_grid;
        const auto& layout = gas ? layouts.gas : layouts.liquid;
        ImGui::PushID(gas ? "gas_phase" : "liquid_phase");
        Draft& draft = drafts[domain.name + (gas ? "/gas" : "/liquid")];
        const uint64_t configuration = hashPhaseSettings(0, domain);
        if (!draft.initialized || draft.configuration != configuration) {
            draft.inherit = !setting.override_enabled;
            draft.lo = setting.override_enabled ? origin + setting.offset_min : layout.origin;
            draft.hi = setting.override_enabled
                ? origin + setting.offset_max : layout.requested_max;
            draft.voxel = setting.override_enabled ? setting.voxel_size : domain.voxel_size;
            draft.configuration = configuration;
            draft.origin = origin;
            draft.initialized = true;
            draft.error.clear();
        } else {
            const Vec3 delta = origin - draft.origin;
            draft.lo += delta;
            draft.hi += delta;
            draft.origin = origin;
        }
        ImGui::Separator();
        ImGui::TextUnformatted(gas ? "Gas" : "Liquid / Granular");
        ImGui::Checkbox("Inherit domain grid", &draft.inherit);
        DomainUi::tooltip("This phase uses the domain's bounds and voxel size.\n"
            "Off: the phase gets its own grid (gas wide/coarse, liquid narrow/fine).\n"
            "Script: fluid.set_phase_grid(domain, phase, inherit=...)");
        ImGui::BeginDisabled(draft.inherit);
        ImGui::SetNextItemWidth(DomainUi::itemWidth());
        ImGui::InputFloat3("Bounds min", &draft.lo.x);
        DomainUi::tooltip("World-space minimum corner of this phase's grid, m.\n"
            "Applied with 'Apply phase grid'.\nScript: fluid.set_phase_grid(..., bounds_min=[x, y, z])");
        ImGui::SetNextItemWidth(DomainUi::itemWidth());
        ImGui::InputFloat3("Bounds max", &draft.hi.x);
        DomainUi::tooltip("World-space maximum corner of this phase's grid, m.\n"
            "Applied with 'Apply phase grid'.\nScript: fluid.set_phase_grid(..., bounds_max=[x, y, z])");
        ImGui::SetNextItemWidth(DomainUi::itemWidth());
        ImGui::InputFloat("Voxel (m)", &draft.voxel, 0.001f, 0.01f, "%.6f");
        DomainUi::tooltip("Cell size of this phase's grid, m. Memory grows as 1/voxel^3.\n"
            "Applied with 'Apply phase grid'.\nScript: fluid.set_phase_grid(..., voxel=...)");
        ImGui::EndDisabled();
        if (ImGui::Button("Apply phase grid")) {
            draft.error.clear();
            changed |= setPhaseGrid(domain, phase, draft.inherit, draft.lo, draft.hi,
                                    draft.voxel, draft.error);
        }
        if (!draft.error.empty()) {
            ImGui::TextWrapped("%s", draft.error.c_str());
        }
        ImGui::Text("Effective: %d x %d x %d | %.6f m | %.1f MiB",
                    layout.nx, layout.ny, layout.nz, layout.voxel,
                    static_cast<double>(layout.cells() *
                        (gas ? kGasWorkingBytesPerCell : kLiquidWorkingBytesPerCell)) /
                        (1024.0 * 1024.0));
        if (layout.budget_clamped) {
            ImGui::TextUnformatted("Combined resource budget reduced this phase grid.");
        }
        ImGui::PopID();
    }
    ImGui::PopID();
    return changed;
}

void drawPhaseResourceSummary(const SimulationGridDomainDesc& domain) {
    const auto layouts = previewPhaseLayouts(domain);
    ImGui::Text("Grid working set: %.1f MiB | gas %zu + liquid %zu cells",
                static_cast<double>(layouts.working_bytes) / (1024.0 * 1024.0),
                layouts.gas.cells(), layouts.liquid.cells());
    if (domain.enforce_resource_budget && domain.resource_budget_mb > 0) {
        ImGui::Text("Combined grid budget: %u MiB", domain.resource_budget_mb);
    }
    if (layouts.gas.budget_clamped || layouts.liquid.budget_clamped) {
        ImGui::TextWrapped("Grid resolution was reduced to fit the combined phase budget.");
    }
}

} // namespace RayTrophiSim::Fluid
