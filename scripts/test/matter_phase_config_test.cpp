// User-run C++ source test, linked with MatterPhaseConfig.cpp in the project environment.
#include "Fluid/MatterPhaseConfig.h"
#include "Fluid/FluidGridResourceBudget.h"

#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>

using namespace RayTrophiSim;
namespace Phase = RayTrophiSim::Fluid;

int main() {
    SimulationGridDomainDesc domain;
    domain.type = SimulationDomainType::Matter;
    domain.bounds_min = Vec3(0.0f);
    domain.bounds_max = Vec3(10.0f);
    std::string error;
    assert(Phase::setPhaseGrid(domain, Phase::GridPhase::Gas, false,
                               Vec3(0.0f), Vec3(10.0f), 0.5f, error));
    assert(Phase::setPhaseGrid(domain, Phase::GridPhase::Liquid, false,
                               Vec3(1.0f), Vec3(3.0f), 0.1f, error));
    auto layout = Phase::previewPhaseLayouts(domain);
    assert(layout.gas.nx == 20 && layout.liquid.nx == 20);
    assert(layout.gas.voxel == 0.5f && layout.liquid.voxel == 0.1f);
    assert(layout.liquid.origin.x == 1.0f);
    assert(layout.working_bytes == layout.gas.cells() * Phase::kGasWorkingBytesPerCell +
                                   layout.liquid.cells() * Phase::kLiquidWorkingBytesPerCell);
    const auto saved = Phase::phaseSettingsJson(domain);
    const uint64_t hash = Phase::hashPhaseSettings(0, domain);
    assert(!Phase::setPhaseGrid(domain, Phase::GridPhase::Liquid, false,
                                Vec3(3.0f), Vec3(1.0f), 0.1f, error));
    assert(!Phase::setPhaseGrid(domain, Phase::GridPhase::Liquid, false,
                                Vec3(0.0f), Vec3(1.0f),
                                std::numeric_limits<float>::quiet_NaN(), error));
    assert(Phase::phaseSettingsJson(domain) == saved);
    assert(Phase::hashPhaseSettings(0, domain) == hash);

    domain.bounds_min += Vec3(5.0f, 0.0f, 0.0f);
    domain.bounds_max += Vec3(5.0f, 0.0f, 0.0f);
    layout = Phase::previewPhaseLayouts(domain);
    assert(layout.gas.origin.x == 5.0f && layout.liquid.origin.x == 6.0f);
    assert(Phase::hashPhaseSettings(0, domain) == hash);

    SimulationGridDomainDesc loaded = domain;
    loaded.gas_phase_grid = {};
    loaded.liquid_phase_grid = {};
    Phase::loadPhaseSettings(nlohmann::json{{"phase_grids", saved}}, loaded);
    assert(Phase::phaseSettingsJson(loaded) == saved);
    auto invalid = saved;
    invalid["liquid"]["voxel"] = -1.0f;
    bool rejected = false;
    try {
        Phase::loadPhaseSettings(nlohmann::json{{"phase_grids", invalid}}, loaded);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    assert(rejected && Phase::phaseSettingsJson(loaded) == saved);
    domain.resource_budget_mb = 1;
    layout = Phase::previewPhaseLayouts(domain);
    assert(layout.working_bytes <= 1024u * 1024u);
    assert(layout.gas.budget_clamped && layout.liquid.budget_clamped);
    assert(layout.gas.origin.x + layout.gas.nx * layout.gas.voxel >= 15.0f);
    assert(layout.liquid.origin.x + layout.liquid.nx * layout.liquid.voxel >= 8.0f);
    SimulationGridDomainState state;
    Phase::synchronizePhaseStorage(state, domain, layout);
    state.type = domain.type;
    state.valid = true;
    state.phase_config_hash = Phase::hashPhaseSettings(0, domain);
    assert(Phase::phaseStorageMatches(state, domain));
    state.matter_liquid_grid.voxel_size *= 2.0f;
    assert(!Phase::phaseStorageMatches(state, domain));
    state.matter_liquid_grid.voxel_size = layout.liquid.voxel;
    assert(state.grid.nx == layout.gas.nx);
    assert(state.matter_liquid_grid.nx == layout.liquid.nx);
    const auto info = Phase::phaseGridInfo(domain, &state);
    assert(info["gas"]["measured"].get<bool>());
    assert(info["liquid"]["cells"].get<std::size_t>() == layout.liquid.cells());

    for (auto type : {SimulationDomainType::Gas, SimulationDomainType::Fluid}) {
        domain.type = type;
        layout = Phase::previewPhaseLayouts(domain);
        assert((type == SimulationDomainType::Gas ? layout.liquid : layout.gas).cells() == 0);
        Phase::synchronizePhaseStorage(state, domain, layout);
        assert(state.matter_liquid_grid.getCellCount() == 0);
        const auto missing = type == SimulationDomainType::Gas
            ? Phase::GridPhase::Liquid : Phase::GridPhase::Gas;
        assert(!Phase::setPhaseGrid(domain, missing, true, Vec3(0.0f), Vec3(1.0f),
                                    0.1f, error));
    }
    SimulationGridDomainDesc legacy;
    Phase::loadPhaseSettings(nlohmann::json::object(), legacy);
    assert(!legacy.gas_phase_grid.override_enabled && !legacy.liquid_phase_grid.override_enabled);
}
