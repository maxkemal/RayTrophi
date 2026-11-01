#include "ParticleSimulation.h"
#include "SimulationDomainLanes.h"
#include "Fluid/MatterPhaseGrid.h"
#include "Fluid/MatterPhaseConfig.h"
#include "Fluid/MatterCacheRestore.h"
#include <cassert>
#include <map>
#include <stdexcept>
#include <unordered_set>

using namespace RayTrophiSim;

namespace {
struct CacheRuntime {
    std::vector<SimulationGridDomainDesc> domains;
    const auto& gridDomains() const { return domains; }
};
struct CacheSystem {
    uint32_t id;
    std::shared_ptr<CacheRuntime> runtime;
};
struct CachedDomain {
    SimulationGridDomainState meta;
};

void checkCacheFramePreflight() {
    std::vector<CacheSystem> systems;
    for (uint32_t id = 1; id <= 2; ++id) {
        auto runtime = std::make_shared<CacheRuntime>();
        runtime->domains.resize(1);
        runtime->domains[0].name = "gas";
        runtime->domains[0].liquid_lane_enabled = false;
        systems.push_back({id, runtime});
    }
    SimulationGridDomainState gas;
    gas.liquid_lane_enabled = false;
    std::map<int, std::vector<std::vector<CachedDomain>>> ram;
    ram[7] = {{{gas}}, {{gas}}};
    std::string error;
    assert(MatterCacheRestore::validateRamFrame(systems, ram, 7, error));
    ram[7][1][0].meta.liquid_lane_enabled = true;
    assert(!MatterCacheRestore::validateRamFrame(systems, ram, 7, error));
    assert(error.find("rebake") != std::string::npos);
    assert(MatterCacheRestore::validateRamFrame(systems, ram, 8, error));
    assert(error.empty());

    std::unordered_set<uint32_t> ids{1, 2};
    std::vector<MatterCacheRestore::DiskSystemFrame> output(1);
    output[0].domains.resize(1);
    output[0].domains[0].type = SimulationDomainType::LegacyFluid;
    int reads = 0;
    const auto mismatched = [&](const std::string&, uint32_t id, int,
                               auto& loaded, auto&) {
        ++reads;
        loaded = {gas};
        loaded[0].liquid_lane_enabled = id == 2;
        return true;
    };
    assert(!MatterCacheRestore::stageDiskFrame(
        systems, ids, "unused", 7, output, error, mismatched));
    assert(reads == 2);
    assert(output.size() == 1 && output[0].domains[0].type == SimulationDomainType::LegacyFluid);
    const auto missing = [&](const std::string&, uint32_t id, int,
                            auto& loaded, auto&) {
        loaded = {gas};
        return id == 1;
    };
    assert(!MatterCacheRestore::stageDiskFrame(
        systems, ids, "unused", 7, output, error, missing));
    assert(error.find("system 2 frame 7") != std::string::npos);
    assert(output.size() == 1 && output[0].domains[0].type == SimulationDomainType::LegacyFluid);
    const auto valid = [&](const std::string&, uint32_t, int, auto& loaded, auto&) {
        loaded = {gas};
        return true;
    };
    assert(MatterCacheRestore::stageDiskFrame(
        systems, ids, "unused", 7, output, error, valid));
    assert(output.size() == 2 && error.empty());
    assert(!output[1].domains[0].liquid_lane_enabled);
}
}

int main() {
    checkCacheFramePreflight();
    assert(parseSavedSimulationDomainType("Gas") == SimulationDomainType::LegacyGas);
    assert(parseSavedSimulationDomainType("Fluid") == SimulationDomainType::LegacyFluid);
    assert(parseSavedSimulationDomainType("Matter") == SimulationDomainType::Matter);
    assert(parseSavedSimulationDomainType(0) == SimulationDomainType::LegacyGas);
    assert(parseSavedSimulationDomainType(1) == SimulationDomainType::LegacyFluid);
    assert(parseSavedSimulationDomainType(2) == SimulationDomainType::Matter);
    for (const auto value : {-1, 3, 255}) {
        bool rejected = false;
        try {
            parseSavedSimulationDomainType(value);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        assert(rejected);
    }
    bool unknown_name_rejected = false;
    try {
        parseSavedSimulationDomainType("unknown_domain_contract");
    } catch (const std::invalid_argument&) {
        unknown_name_rejected = true;
    }
    assert(unknown_name_rejected);
    // Reproduce a saved two-domain scene whose old liquid source carried the
    // historical default Gas phase. Migration must not turn it into smoke.
    std::vector<SimulationGridDomainDesc> domains(2);
    domains[0].type = SimulationDomainType::LegacyGas;
    domains[0].shader = std::make_shared<VolumeShader>();
    const auto shader = domains[0].shader;
    domains[1].type = SimulationDomainType::LegacyFluid;
    std::vector<SimulationFlowSourceDesc> sources(2);
    sources[0].domain_index = 0;
    sources[1].domain_index = 1;
    sources[1].phase = SimulationFlowSourceDesc::Phase::Gas;
    canonicalizeMatterDomain(domains[0], sources, 0);
    canonicalizeMatterDomain(domains[1], sources, 1);
    assert(domains[0].type == SimulationDomainType::Matter);
    assert(domains[1].type == SimulationDomainType::Matter);
    assert(simulationDomainHasGas(domains[0]) && !simulationDomainHasLiquid(domains[0]));
    assert(!simulationDomainHasGas(domains[1]) && simulationDomainHasLiquid(domains[1]));
    assert(domains[0].shader == shader && domains[0].fluid_fog_shader == shader);
    assert(sources[1].phase == SimulationFlowSourceDesc::Phase::Liquid);
    const auto saved = Fluid::phaseSettingsJson(domains[1]);
    SimulationGridDomainDesc loaded;
    Fluid::loadPhaseSettings({{"phase_grids", saved}}, loaded);
    assert(!loaded.gas_lane_enabled && loaded.liquid_lane_enabled);

    SimulationGridDomainState state;
    state.gas_lane_enabled = false;
    state.liquid_lane_enabled = true;
    assert(!Fluid::liquidUsesSecondaryGrid(state));
    state.gas_lane_enabled = true;
    assert(Fluid::liquidUsesSecondaryGrid(state));
    std::string error;
    assert(!validateMatterCacheLanes({state}, {domains[1]}, error));
    assert(!error.empty());

    std::vector<SimulationGridDomainState> states(2);
    for (std::size_t i = 0; i < states.size(); ++i) {
        states[i].valid = true;
        states[i].gas_lane_enabled = domains[i].gas_lane_enabled;
        states[i].liquid_lane_enabled = domains[i].liquid_lane_enabled;
        states[i].grid.resize(8, 8, 8, 0.1f, Vec3(0.0f));
    }
    assert(resolveMatterGasExchangeOwners(domains, states)[1] == 0);
    domains[1].gas_lane_enabled = true;
    states[1].gas_lane_enabled = true;
    states[1].matter_liquid_grid = states[1].grid;
    // An earlier external gas cannot steal a Matter domain's own liquid heat.
    assert(resolveMatterGasExchangeOwners(domains, states)[1] == 1);
}
