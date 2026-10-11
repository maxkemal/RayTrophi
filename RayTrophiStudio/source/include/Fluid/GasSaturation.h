#pragma once

// ═══════════════════════════════════════════════════════════════════════════
// FOG WATER — saturation adjustment of a gas cell's water (2026-10-11).
// ═══════════════════════════════════════════════════════════════════════════
// The gas phase carries water in two pools per cell, in the units the
// atmosphere layer uses (kg/m^3 = kg / cell volume):
//   gas_phase_mass_kg        vapour
//   gas_phase_condensate_kg  fog/cloud water: liquid droplets riding the air
// gas_phase_energy_j is the energy of all of it (vapour c*T + L per kg,
// fog water c*T per kg).
//
// Each step, every cell is brought to equilibrium with its air temperature:
// vapour above saturation condenses to fog water and its latent heat warms
// the air; fog water in sub-saturated air evaporates and cools it. The
// latent heat feeds back on the saturation point, so the amount is solved
// (bisection on f(d) = m - d + m_bg - m_sat(T0 + L d / C)), not guessed:
// the cell never overshoots. C is the air's heat capacity plus the water's.
// m_sat follows Clausius-Clapeyron from the substance's Tb, L and M, with no
// 1 atm cap (a cell holding more vapour than 1 atm is over-pressured: the gas
// solver has no expansion source for steam, and capping would heat it without
// bound). m_bg is the world humidity's vapour, a background with no mass.
//
// Rain (2026-10-11, part 3). The air cannot hold liquid water without limit:
// the first live run kept 41.7 kg/m^3 of fog suspended over a boiling tank.
//   * Autoconversion (Kessler): fog water above the domain's onset
//     (fog_autoconversion_kg_m3) coalesces into RAIN water at
//     fog_autoconversion_rate (1/s, implicit: (w - onset)(1 - e^(-k dt))),
//     and whatever is still above fog_ceiling_kg_m3 rains at once. The
//     water's sensible energy c*Ta moves from the shared energy pool into
//     the rain's own (gas_phase_rain_energy_j).
//   * Drift and fall: a drop moves with the air plus its terminal fall. The
//     rain pools ride the gas velocity in the inventory advection (wind drift,
//     and an updraft holds rain up; 2026-10-11 part 4), then fall relative to
//     the air at the Kessler terminal speed 36.34 (rho_r [g/cm^3])^0.1364 m/s
//     (5.5 m/s at 1 g/m^3, capped at 10), column by column, and stop above a
//     solid cell, at the grid floor, or in a cell the surface exchange stamped
//     as lying on liquid (gas_rain_landing).
//   * Landing: in a stamped cell the surface exchange hands the rain to the
//     liquid parcels below by surface share (no phase change). Where rain
//     pools to one liquid parcel (rho h^3 / ppc of the matter domain's
//     liquid grid) inside that grid, it becomes a parcel at its own
//     temperature (rain-out). Both are ledger Precipitation `:rain -> liquid`.
//
// Runs on a Matter domain with its thermal chain on, and on any gas domain
// that holds fog water (mist parcels deposited there). The condensable
// substance is the domain's Default Substance when it has vapour data
// (boiling_kelvin, latent_heat_vaporization, vapor_molar_mass_kg_mol) and is
// not flammable, else Water. Every kg of the vapour pool is that substance;
// fuel vapour and combustion products have their own non-condensable pool
// (gas_phase_foreign_kg, 2026-10-11 part 4) and never condense here.
//
// ★ VIRTUAL TEMPERATURE. The gas temperature field drives buoyancy, so boiling
// writes steam into it at its virtual temperature Tv = T Ma / M_mix (pure
// steam at 373 K reads ~600 K: as light as 600 K air). Readers that need the
// AIR temperature (this adjustment, the surface exchange) convert with the
// cell's vapour mass (airKelvinFromVirtual) and write their change back as a
// Tv DIFFERENCE (virtualKelvinFromAir(after) - virtualKelvinFromAir(before)),
// so the field's own cooling toward ambient is never undone. Before part 4 the
// adjustment read 600 K above the source and held vapour there as dry.
//
// The GPU path (Vulkan, resident inventory) runs sim_inventory_saturation and
// reads back only the totals and rain records; the CPU path is the reference.

#include <cstdint>
#include <string>
#include <vector>

namespace RayTrophiSim {
class MatterExchangeLedger;
class SimulationComputeContext;
struct MaterialTemperatureScale;
struct SimulationGridDomainDesc;
struct SimulationGridDomainState;
struct SimulationGridDomainComputeBuffers;

namespace Fluid {

struct CondensableVapour {
    std::string substance;
    uint32_t tag = 0;              // kSubstanceUntagged when it is the default
    float boiling_kelvin = 0.0f;
    float latent_j_kg = 0.0f;
    float molar_kg_mol = 0.0f;
    float specific_heat = 0.0f;
    bool valid() const {
        return boiling_kelvin > 0.0f && latent_j_kg > 0.0f && molar_kg_mol > 0.0f &&
               specific_heat > 0.0f;
    }
};

CondensableVapour resolveCondensableVapour(const SimulationGridDomainDesc& domain);

// Virtual temperature of air at air_kelvin holding vapour_kg of a vapour of
// molar mass vapour_molar in cell_volume, at 1 atm: the temperature dry air
// would need for the same density. Mole fraction x = n_v R T / (P V), capped
// at 1 (pure vapour). molar <= 0 or no vapour: Tv = T.
float virtualKelvinFromAir(float air_kelvin, float vapour_kg, float cell_volume,
                           float vapour_molar);
// Inverse, closed form: T = Tv Ma / (Ma + Tv k (Ma - Mv)), k = m R / (Mv P V);
// pure vapour (x >= 1) gives T = Tv Mv / Ma. The result is never below
// min(Tv, floor_kelvin): a cell whose Tv the field has already cooled toward
// ambient (gas temperature dissipation stands for entrainment of ambient air)
// does not read as air colder than that ambient.
float airKelvinFromVirtual(float virtual_kelvin, float vapour_kg, float cell_volume,
                           float vapour_molar, float floor_kelvin);

struct GasSaturationInputs {
    const MaterialTemperatureScale* scale = nullptr;
    float ambient_kelvin = 293.15f;       // K of the gas solver's ambient value
    float gas_ambient_normalized = 0.0f;
    float humidity = 0.0f;                // world relative humidity 0..1
    float world_ambient_kelvin = 293.15f; // where the background vapour sits
    float dt = 0.0f;                      // s, this gas step
    uint64_t event_id_base = 0;
};

// False only on a device failure (the caller holds the inventory). A domain
// with nothing to do returns true and leaves gas_stats.saturation_measured off.
bool adjustGasSaturation(const SimulationGridDomainDesc& domain,
                         SimulationGridDomainState& state,
                         const GasSaturationInputs& inputs,
                         SimulationComputeContext* compute,
                         SimulationGridDomainComputeBuffers* buffers,
                         MatterExchangeLedger& ledger);

// One cell, host side (the CPU path; also the reference for the shader and the
// regression test). Returns d: > 0 condensed, < 0 evaporated (kg). Updates m,
// w, e and the air temperature in kelvin.
float adjustCellSaturation(float& vapour_kg, float& water_kg, float& energy_j,
                           float& air_kelvin, float cell_volume,
                           const CondensableVapour& vapour, float background_kg);

// Autoconversion of one cell (CPU reference of the shader's step 2). onset and
// ceiling are kg per cell, conversion = 1 - exp(-rate dt). Returns the kg moved
// from fog water to rain water; c*Ta per kg moves from energy_j to rain_energy_j
// (no more than energy_j holds).
float autoconvertCellFog(float& water_kg, float& energy_j, float& rain_kg,
                         float& rain_energy_j, float air_kelvin, float specific_heat,
                         float onset_kg, float conversion, float ceiling_kg);

// Kessler terminal speed of rain water at rain_kg_m3, m/s (capped at 10).
float rainTerminalSpeed(float rain_kg_m3);

// One column sweep of the rain fall (CPU reference of sim_inventory_rain_fall)
// over a whole grid. landing/solid may be empty (nothing stops the rain but
// the floor).
void fallRain(std::vector<float>& rain_kg, std::vector<float>& rain_energy_j,
              const std::vector<uint32_t>& landing, uint32_t stamp,
              const std::vector<uint8_t>& solid, int nx, int ny, int nz,
              float voxel, float dt);

} // namespace Fluid
} // namespace RayTrophiSim
