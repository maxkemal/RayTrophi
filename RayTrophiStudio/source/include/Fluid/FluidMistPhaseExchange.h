#pragma once

#include <cstddef>
#include <cstdint>

namespace RayTrophiSim {

class MatterExchangeLedger;
struct MaterialTemperatureScale;
struct SimulationGridDomainDesc;
struct SimulationGridDomainState;

namespace GridFluid {
struct SolverParams;
}

namespace Fluid {

struct FluidMistPhaseExchangeStats {
    bool ran = false;
    std::size_t particles_removed = 0;
    double transferred_mass_kg = 0.0;
    double transferred_energy_j = 0.0;
};

// Moves every labelled mist parcel inside one gas domain into the gas phase's
// fog-water (condensate) pool: liquid water carried by the air, no phase
// change and no latent heat here. The remaining APIC mass is debited exactly
// once and recorded as a balanced mist_to_gas exchange; it evaporates later
// through adjustGasSaturation, which takes the latent heat from the air.
FluidMistPhaseExchangeStats transferMistToGas(
    const SimulationGridDomainDesc& fluid_domain,
    SimulationGridDomainState& fluid_state,
    const SimulationGridDomainDesc& gas_domain,
    SimulationGridDomainState& gas_state,
    uint64_t event_id_base,
    MatterExchangeLedger& ledger);

// True when some parcel is above its boiling point: wakes an otherwise idle
// gas step so the steam has somewhere to go.
bool hasBoilingParcel(const SimulationGridDomainDesc& fluid_domain,
                      const SimulationGridDomainState& fluid_state);

struct FluidBoilingExchangeStats {
    bool ran = false;
    std::size_t boiling_particles = 0;   // parcels that lost mass to boiling
    std::size_t particles_removed = 0;   // parcels boiled away completely
    double vaporized_mass_kg = 0.0;
    double latent_energy_j = 0.0;
    std::size_t bubbles_emitted = 0;     // whitewater Bubble parcels born
};

// Boiling: a liquid parcel above its substance's boiling_kelvin spends the
// superheat m*c*(T - Tb) on latent heat, so dm = min(m, m*c*(T - Tb) / L)
// leaves as vapour and the parcel is clamped to Tb. No rate constant: the
// heat source sets the pace. The vapour joins the gas phase of the
// overlapping gas domain (the matter domain's own) as kg/J inventory and a
// gas density/temperature tracer, recorded as a balanced `vaporization`
// exchange. Any fluid-model, unfrozen liquid whose substance has boiling data
// boils; a flammable one's vapour also feeds the gas fuel channel. Unknown
// vapour molar mass reads as air (neutral).
// Parcels that shrink to mist size leave through transferMistToGas.
// Nucleate boiling: the vapour first occupies the parcel's volume, so the
// parcel is lifted by its void fraction, dv = -g * Vv/(Vv+Vl) * dt (the rolling
// boil), and past a 0.2 void fraction a whitewater Bubble is seeded there
// sparsely (chance void*0.05 per step, <= 512 per step, 0.75 s life); the foam
// solver lifts it and turns it to foam at the surface.
// Buoyancy: the gas temperature tracer is written as the vapour's VIRTUAL
// temperature Tv = Tb * M_air / M_vapour in the solver's normalized scale
// (`scale`), so steam (18 g/mol) rises like a ~600 K parcel even at 373 K,
// because at equal pressure gas density goes as M/T.
FluidBoilingExchangeStats transferBoilingToGas(
    const SimulationGridDomainDesc& fluid_domain,
    SimulationGridDomainState& fluid_state,
    const SimulationGridDomainDesc& gas_domain,
    SimulationGridDomainState& gas_state,
    const MaterialTemperatureScale& scale,
    float dt,
    uint64_t event_id_base,
    MatterExchangeLedger& ledger);

// Liquid surface <-> gas, per substance, energy conserved (Matter domains:
// the gas grid IS the air above the liquid). For every parcel in the top
// liquid cell of its column, with the gas cell directly above:
//  * Heat: the two bodies (parcel C_p = m*c of its substance, air cell
//    C_a = rho_air(T)*V*c_air) relax toward their shared equilibrium through a
//    natural-convection film, h = 20 W/(m^2 K) over the parcel's share of the
//    cell face. Exact two-body solution: never overshoots, C_p dT_p = -C_a dT_a.
//  * Evaporation below boiling (Hertz-Knudsen with Clausius-Clapeyron
//    saturation from the substance's Tb, L and M; humidity from the
//    atmosphere and from the vapour already in the cell). The latent heat is
//    taken from the parcel, the vapour joins the gas phase inventory and the
//    ledger as `vaporization`.
// Air temperature = ambient_kelvin + (gas.temperature - gas_ambient_normalized)
// * kelvin_per_unit: the gas solver's buoyancy anomaly, placed on the liquid
// domain's own ambient (so a domain override holds in empty air too).
// Writes fluid_state.thermal_stats gas fields.
void exchangeLiquidGasSurface(
    const SimulationGridDomainDesc& fluid_domain,
    SimulationGridDomainState& fluid_state,
    const SimulationGridDomainDesc& gas_domain,
    SimulationGridDomainState& gas_state,
    const MaterialTemperatureScale& scale,
    float ambient_kelvin,
    float gas_ambient_normalized,
    float drying_scale,
    float dt,
    uint64_t event_id_base,
    MatterExchangeLedger& ledger);

// Carries physical gas kg/J with the solved gas velocity and the same advection
// scheme and boundary rule as the gas scalar channels. Visual dissipation does
// not destroy physical inventory; only transport through an open boundary can.
bool advectGasPhaseInventory(
    SimulationGridDomainState& gas_state,
    const GridFluid::SolverParams& params,
    float dt);

} // namespace Fluid
} // namespace RayTrophiSim
