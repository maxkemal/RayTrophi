#pragma once

#include "Vec3.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace FluidSim { class FluidGrid; }
namespace RayTrophiSim {
class SimulationComputeContext;
struct MatterExchangeRecord;
namespace Fluid {
class FluidParticles;

struct MatterPoreParams {
    bool enabled = false;
    float porosity = 0.35f;
    float permeability_m2 = 1.0e-10f;
    float viscosity_pa_s = 0.001f;
    float gravity_m_s2 = 9.81f;
    float drainage_scale = 1.0f;
    bool wet_response_enabled = false;
    bool wet_appearance_enabled = false;
    float wet_friction_scale = 0.6f;
    float wet_dilatancy_scale = 0.25f;
    float capillary_cohesion_pa = 250.0f;
    float pore_pressure_scale = 1.0f;
    float wet_color_scale = 0.55f;
    float wet_roughness_scale = 0.65f;
    // Surface appearance can become fully wet before the pore volume fills.
    float wet_appearance_full_saturation = 0.05f;
};

struct MatterPoreReport {
    bool measured = false;
    bool held = false;
    std::string status = "Disabled";
    double free_water_kg = 0.0;
    double pore_water_kg = 0.0;
    double capacity_kg = 0.0;
    double absorbed_kg = 0.0;
    double drained_kg = 0.0;
    double absorbed_energy_j = 0.0;
    double drained_energy_j = 0.0;
    Vec3 absorbed_momentum;
    Vec3 drained_momentum;
    double mass_error_kg = 0.0;
    double momentum_error_kg_m_s = 0.0;
    double thermal_energy_error_j = 0.0;
    std::size_t drainage_births = 0;
    std::size_t drainage_refills = 0;
    std::size_t drainage_budget_blocked_carriers = 0;
    std::size_t budget_bytes = 0;
};

bool validateMatterPoreParams(const MatterPoreParams& params, std::string& error);
uint64_t matterPoreSettingsHash(const MatterPoreParams& params);
bool needsMatterPoreTransport(const FluidParticles& particles, const MatterPoreParams& params);
// Transport mass includes pore water; canonical dry rest_mass_kg is never rewritten.
void matterPoreTransportMasses(const FluidParticles& particles, bool legacy_granular,
                              std::vector<float>& rest, std::vector<float>& fraction);

// GPU cell-owned exchange; host batches births/refills per cell with conserved moments.
// The caller provides a candidate and only commits it when this returns true.
bool exchangeMatterPoresGpu(FluidParticles& particles, const FluidSim::FluidGrid& grid,
                           const MatterPoreParams& params, float dt,
                           std::size_t particle_limit, std::size_t budget_bytes,
                           SimulationComputeContext& compute, MatterPoreReport& report,
                           std::vector<MatterExchangeRecord>& events,
                           std::string& error);

} // namespace Fluid
} // namespace RayTrophiSim
