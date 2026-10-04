#pragma once

#include "Fluid/MatterPhaseGrid.h"
#include "json.hpp"

#include <string>

namespace RayTrophiSim::Fluid {

enum class GridPhase { Gas, Liquid };

struct PhaseLayout {
    Vec3 origin;
    Vec3 requested_max;
    int nx = 0;
    int ny = 0;
    int nz = 0;
    float voxel = 0.0f;
    bool budget_clamped = false;
    std::size_t cells() const;
};

struct PhaseLayouts {
    PhaseLayout gas;
    PhaseLayout liquid;
    std::size_t working_bytes = 0;
};

Vec3 logicalGridOrigin(const SimulationGridDomainDesc& domain);
bool parseGridPhase(const std::string& name, GridPhase& phase, std::string& error);
bool setPhaseGrid(SimulationGridDomainDesc& domain, GridPhase phase, bool inherit,
                  const Vec3& bounds_min, const Vec3& bounds_max, float voxel,
                  std::string& error);
PhaseLayouts resolvePhaseLayouts(const SimulationGridDomainDesc& domain,
                                const Vec3& logical_min, const Vec3& logical_max,
                                int nx, int ny, int nz, float voxel);
PhaseLayouts previewPhaseLayouts(const SimulationGridDomainDesc& domain);
bool phaseStorageMatches(const SimulationGridDomainState& state,
                         const SimulationGridDomainDesc& domain);
void synchronizePhaseStorage(SimulationGridDomainState& state,
                             const SimulationGridDomainDesc& domain,
                             const PhaseLayouts& layouts);
nlohmann::json phaseSettingsJson(const SimulationGridDomainDesc& domain);
void loadPhaseSettings(const nlohmann::json& object, SimulationGridDomainDesc& domain);
nlohmann::json phaseGridInfo(const SimulationGridDomainDesc& domain,
                            const SimulationGridDomainState* state);
uint64_t hashPhaseSettings(uint64_t hash, const SimulationGridDomainDesc& domain);
void adoptPresetPhaseGrids(SimulationGridDomainDesc& matter,
                           const SimulationGridDomainDesc& gas,
                           const SimulationGridDomainDesc& liquid);
bool drawPhaseGridControls(SimulationGridDomainDesc& domain);
void drawPhaseResourceSummary(const SimulationGridDomainDesc& domain);

} // namespace RayTrophiSim::Fluid
