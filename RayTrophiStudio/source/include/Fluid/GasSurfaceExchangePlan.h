#pragma once

#include "Vec3.h"
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace RayTrophiSim {
struct SimulationGridDomainDesc;
struct SimulationGridDomainState;
struct MaterialTemperatureScale;
class MatterExchangeLedger;
namespace Fluid {
// Buffer uploads copy bytes; explicit fields preserve the shader stride without
// extended host alignment, which MSVC stable_sort's temporary storage rejects.
struct GasSurfaceParcel {
    uint32_t particle;
    uint32_t air_cell;
    float kelvin;
    float rest_mass;
    float fraction;
    float specific_heat;
    float boiling;
    float latent;
    float molar;
    float share;
    double capacity;
    float velocity[4];
};
struct GasSurfaceGroup {
    uint32_t first;
    uint32_t count;
    uint32_t cell;
    uint32_t padding = 0;
};
struct GasSurfaceResult {
    double heat_to_gas;
    double sensible;
    double latent;
    double mass;
    float kelvin;
    float fraction;
    float exchanged;
    float error;
    // Rain water the parcel absorbed from its air cell: kg and its sensible
    // energy J (no phase change; kelvin above already includes it).
    float rain_kg = 0.0f;
    float rain_energy_j = 0.0f;
    float padding[2]{};
};
struct GasSurfaceExchangePlan {
    std::string error;
    std::vector<GasSurfaceParcel> records;
    std::vector<GasSurfaceGroup> groups;
    std::map<uint32_t, std::string> substances;
};
GasSurfaceExchangePlan buildGasSurfaceExchangePlan(
    const SimulationGridDomainDesc& liquid_domain, SimulationGridDomainState& liquid,
    const SimulationGridDomainDesc& gas_domain, const SimulationGridDomainState& gas);
bool validateGasSurfaceResults(const GasSurfaceExchangePlan& plan,
                               const std::vector<GasSurfaceResult>& results);
void publishGasSurfaceResults(const GasSurfaceExchangePlan& plan,
                              const std::vector<GasSurfaceResult>& results,
                              const SimulationGridDomainDesc& liquid_domain,
                              SimulationGridDomainState& liquid,
                              const SimulationGridDomainDesc& gas_domain,
                              SimulationGridDomainState& gas,
                              uint64_t event_id, MatterExchangeLedger& ledger,
                              bool inventory_committed);
bool exchangeGasSurfaceCpu(
    const SimulationGridDomainDesc& liquid_domain, SimulationGridDomainState& liquid,
    const SimulationGridDomainDesc& gas_domain, SimulationGridDomainState& gas,
    const MaterialTemperatureScale& scale, float ambient_kelvin,
    float gas_ambient_normalized, float drying_scale, float dt,
    uint64_t event_id, MatterExchangeLedger& ledger);
}
}
