// Source regression for the user's test build: fog-water saturation adjustment
// (Fluid::adjustCellSaturation, the CPU reference of sim_inventory_saturation).
// Expected values: Python oracle of the same formula (2026-10-11).
#include "Fluid/GasSaturation.h"

#include <cassert>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

using namespace RayTrophiSim::Fluid;

int main() {
    CondensableVapour water;
    water.substance = "Water";
    water.boiling_kelvin = 373.15f;
    water.latent_j_kg = 2.26e6f;
    water.molar_kg_mol = 0.018015f;
    water.specific_heat = 4180.0f;
    const float volume = 0.05f * 0.05f * 0.05f;
    // Humid air (0.03 kg/m^3) at 293 K: condenses ~5.3 g/m^3 and the latent
    // heat warms it ~9 K; the warming is what stops it. (Constant-L
    // Clausius-Clapeyron gives 0.0209 kg/m^3 saturation at 293 K; tables 0.0173.)
    float m = 0.03f * volume, w = 0.0f, e = 1.0e3f, t = 293.15f;
    // 50% world humidity at 293.15 K (saturation 0.0173 kg/m^3) as background.
    const float background = 0.5f * 0.0173f * volume;
    float d = adjustCellSaturation(m, w, e, t, volume, water, background);
    assert(d > 0.0f && std::abs(d / volume * 1000.0f - 5.33f) < 0.3f);
    assert(std::abs(t - 302.2f) < 0.6f);
    assert(std::abs((m + w) - 0.03f * volume) < 1.0e-9f);           // water conserved
    assert(std::abs(e - (1.0e3f - d * water.latent_j_kg)) < 1.0e-2f); // latent left the water

    // Fog (5 g/m^3) in half-saturated air evaporates and cools the air.
    m = 0.0f; w = 0.005f * volume; e = 10.0f; t = 293.15f;
    d = adjustCellSaturation(m, w, e, t, volume, water, background);
    assert(d < 0.0f && std::abs(-d / volume * 1000.0f - 4.53f) < 0.3f);
    assert(t < 293.15f - 5.0f);

    // Over-pressured steam (8.5 kg/m^3): no runaway, it settles hot but bounded.
    m = 8.5f * volume; w = 0.0f; e = 1.0e6f; t = 373.15f;
    d = adjustCellSaturation(m, w, e, t, volume, water, background);
    assert(d > 0.0f && d < m && t > 373.15f && t < 600.0f);

    // Autoconversion (part 3): fog water above the ceiling rains at once and
    // carries c*Ta into the rain's own energy; water is conserved.
    const float onset = 1.0e-3f * volume, ceiling = 5.0e-3f * volume;
    float fog = 0.03f * volume, energy = 1.0f, rain = 0.0f, rain_energy = 0.0f;
    const float before = fog;
    const float conversion = -std::expm1(-1.0e-3f / 60.0f);
    float converted = autoconvertCellFog(fog, energy, rain, rain_energy, 300.0f,
                                         water.specific_heat, onset, conversion, ceiling);
    assert(std::abs(fog - ceiling) < 1.0e-9f);
    assert(std::abs(fog + rain - before) < 1.0e-9f && std::abs(converted - rain) < 1.0e-12f);
    assert(rain_energy <= 1.0f + 1.0e-6f && energy >= 0.0f);   // never more than the pool held
    // Between onset and ceiling only the Kessler share converts.
    fog = 3.0e-3f * volume; energy = 1.0e3f; rain = rain_energy = 0.0f;
    converted = autoconvertCellFog(fog, energy, rain, rain_energy, 300.0f,
                                   water.specific_heat, onset, conversion, ceiling);
    assert(converted > 0.0f && std::abs(converted - 2.0e-3f * volume * conversion) < 1.0e-12f);
    assert(std::abs(rain_energy - converted * water.specific_heat * 300.0f) < 1.0e-6f);

    // Kessler terminal speed: ~5.5 m/s at 1 g/m^3, capped at 10.
    assert(std::abs(rainTerminalSpeed(1.0e-3f) - 5.51f) < 0.1f);
    assert(rainTerminalSpeed(1.0e3f) == 10.0f && rainTerminalSpeed(0.0f) == 0.0f);

    // Fall: a 1x8x1 column, rain at the top, one second: it reaches the floor,
    // mass and energy conserved; a landing stamp at y=3 stops it there.
    std::vector<float> column(8, 0.0f), column_energy(8, 0.0f);
    column[7] = 1.0e-4f; column_energy[7] = 120.0f;
    fallRain(column, column_energy, {}, 1u, {}, 1, 8, 1, 0.05f, 1.0f);
    float total = 0.0f, total_energy = 0.0f;
    for (int y = 0; y < 8; ++y) { total += column[y]; total_energy += column_energy[y]; }
    assert(std::abs(total - 1.0e-4f) < 1.0e-9f && std::abs(total_energy - 120.0f) < 1.0e-3f);
    assert(column[0] > 0.99f * 1.0e-4f);
    std::fill(column.begin(), column.end(), 0.0f);
    std::fill(column_energy.begin(), column_energy.end(), 0.0f);
    column[7] = 1.0e-4f;
    std::vector<uint32_t> landing(8, 0u);
    landing[3] = 5u;
    fallRain(column, column_energy, landing, 5u, {}, 1, 8, 1, 0.05f, 1.0f);
    assert(column[3] > 0.99f * 1.0e-4f && column[0] == 0.0f);
    return 0;
}
