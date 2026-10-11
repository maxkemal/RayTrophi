#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

namespace RayTrophiSim {
class MatterExchangeLedger;
struct MaterialTemperatureScale;
struct WorldThermalState;
struct SimulationGridDomainDesc;
struct SimulationGridDomainState;
struct SimulationGridDomainComputeBuffers;
class SimulationComputeContext;
namespace Fluid {

// Host parcel SoA remains the liquid publication contract. Only compact surface
// parcel records/results cross this boundary; gas kg/J is read and written on GPU.
bool exchangeGasSurfaceGpu(
    const SimulationGridDomainDesc& fluid_domain, SimulationGridDomainState& fluid,
    const SimulationGridDomainDesc& gas_domain, SimulationGridDomainState& gas,
    SimulationGridDomainComputeBuffers& gas_buffers, SimulationComputeContext& compute,
    const MaterialTemperatureScale& scale, float ambient_kelvin,
    float gas_ambient_normalized, float drying_scale, float dt,
    uint64_t event_id_base, MatterExchangeLedger& ledger);

bool exchangeGasSurfaceGpuDomains(
    const std::vector<std::size_t>& targets,
    const std::vector<SimulationGridDomainDesc>& domains,
    std::vector<SimulationGridDomainState>& states, std::size_t gas_index,
    SimulationGridDomainComputeBuffers& buffers, SimulationComputeContext& compute,
    const WorldThermalState& world, float dt, uint64_t step, MatterExchangeLedger& ledger);

} // namespace Fluid
} // namespace RayTrophiSim
