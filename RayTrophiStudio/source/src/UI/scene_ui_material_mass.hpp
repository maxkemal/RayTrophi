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
    block.Value("    conservation error", "%.8f", mass.conservation_error);
}
} // namespace MaterialMassUI
