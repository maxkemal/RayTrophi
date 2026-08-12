#pragma once

namespace MaterialMassUI {
inline void drawBudget(UIWidgets::PerfBlock& block,
                       const RayTrophiSim::MaterialStateField& field) {
    const auto mass =
        RayTrophiSim::MaterialStateFieldSystem::summarizeMassBudget(field);
    if (!mass.valid) return;
    block.Value("  mass initial / solid", "%.4f / %.4f",
                mass.initial_mass, mass.solid_mass);
    block.Value("    pyrolyzed / molten", "%.4f / %.4f",
                mass.pyrolyzed_mass, mass.molten_reservoir_mass);
    block.Value("    APIC transferred", "%.4f", mass.transferred_mass);
    // ★ Measured on the raw field, so a non-zero value here is a real solver
    // fault, not rounding. The four masses above are clamped for display and
    // will always look consistent even when this does not.
    block.Value("    conservation error", "%.8f", mass.conservation_error);
    if (mass.conservation_error > 0.0f || mass.invalid_elements > 0u) {
        block.Value("    !! overflow / negative", "%.6f / %.6f",
                    mass.budget_overflow_mass, mass.negative_mass);
        block.Value("    !! non-finite elements", "%u", mass.invalid_elements);
    }
}
} // namespace MaterialMassUI
