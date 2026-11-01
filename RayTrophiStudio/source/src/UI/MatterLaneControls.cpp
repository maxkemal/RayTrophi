#include "UI/MatterLaneControls.h"
#include "Api/MatterDomainLanesApi.h"
#include "ParticleSimulation.h"
#include "DomainPanelWidgets.h"
#include "imgui.h"

#include <exception>
#include <string>

namespace RayTrophiSim::Fluid {
bool drawMatterLaneControls(const SimulationGridDomainDesc& domain) {
    static std::string error;
    static std::string error_domain;
    const bool gas = simulationDomainHasGas(domain);
    const bool liquid = simulationDomainHasLiquid(domain);
    int selected = gas && liquid ? 2 : liquid ? 1 : 0;
    const char* presets[] = {"Gas", "Liquid", "Matter (gas + liquid / granular)"};
    ImGui::SetNextItemWidth(-1.0f);
    if (ImGui::Combo("Lane preset##MatterLanes", &selected, presets, 3)) {
        try {
            rtapi::setMatterDomainLanes(domain.name, selected != 1, selected != 0);
            error.clear();
            return true;
        } catch (const std::exception& exception) {
            error_domain = domain.name;
            error = exception.what();
        }
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("One Matter domain; enable gas, liquid, or both lanes.\n"
            "Disabled lanes allocate no grid and take no simulation steps.\n"
            "Reset live content and retarget sources before disabling a lane.\n"
            "Python / IPC: fluid.get_lanes, fluid.set_lanes.");
    }
    if (!error.empty() && error_domain == domain.name) {
        DomainUi::Reason(error.c_str());
    }
    return false;
}

bool drawMatterSourcePhaseControl(const SimulationGridDomainDesc& domain,
    SimulationFlowSourceDesc& source) {
    int phase = source.phase == SimulationFlowSourceDesc::Phase::Liquid ? 1 : 0;
    const char* names[] = {"Gas", "Liquid"};
    if (!ImGui::Combo("Emission Phase##FlowPhase", &phase, names, 2)) {
        return false;
    }
    bool liquid = false;
    std::string error;
    if (!resolveMatterSourcePhase(domain, phase == 1 ? "liquid" : "gas", liquid, error)) {
        DomainUi::Reason(error.c_str());
        return false;
    }
    source.phase = liquid ? SimulationFlowSourceDesc::Phase::Liquid
                          : SimulationFlowSourceDesc::Phase::Gas;
    return true;
}
} // namespace RayTrophiSim::Fluid
