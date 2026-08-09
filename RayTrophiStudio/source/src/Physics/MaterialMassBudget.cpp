#include "MaterialStateField.h"

#include <algorithm>
#include <cmath>

namespace RayTrophiSim {
MaterialMassBudgetSummary MaterialStateFieldSystem::summarizeMassBudget(
    const MaterialStateField& field) {
    MaterialMassBudgetSummary out;
    if (field.elementCount() == 0 ||
        field.state.size() != field.elementCount() * MaterialStateField::kStateStride)
        return out;
    const auto& profile = findSubstance(field.substance_name);
    const MaterialSubstance substance = MaterialSubstance::fromProfile(
        profile, MaterialTemperatureScale{}, field.overrides);
    const float capacity_per_area = substance.mass_capacity;
    if (capacity_per_area <= 0.0f) return out;

    double initial = 0.0, solid = 0.0, pyro = 0.0, molten = 0.0, transferred = 0.0;
    for (std::size_t i = 0; i < field.elementCount(); ++i) {
        const float area = std::max(field.centers[i * 4u + 3u], 0.0f);
        const float mass = capacity_per_area * area;
        const float burned = std::clamp(
            field.state[i * MaterialStateField::kStateStride + 6u], 0.0f, mass);
        const float melt_fraction = std::clamp(
            field.state[i * MaterialStateField::kStateStride + 5u], 0.0f, 1.0f);
        const float melted = std::min(melt_fraction * mass, mass - burned);
        const float moved = std::clamp(
            field.state[i * MaterialStateField::kStateStride + 7u],
            0.0f, mass - burned - melted);
        initial += mass;
        pyro += burned;
        molten += melted;
        transferred += moved;
        solid += std::max(mass - burned - melted - moved, 0.0f);
    }
    out.valid = initial > 0.0;
    out.initial_mass = static_cast<float>(initial);
    out.solid_mass = static_cast<float>(solid);
    out.pyrolyzed_mass = static_cast<float>(pyro);
    out.molten_reservoir_mass = static_cast<float>(molten);
    out.transferred_mass = static_cast<float>(transferred);
    out.conservation_error = std::abs(out.initial_mass -
        (out.solid_mass + out.pyrolyzed_mass + out.molten_reservoir_mass +
         out.transferred_mass));
    return out;
}

bool MaterialStateFieldSystem::consumeMoltenMass(
    const std::string& object_key, float requested_mass,
    SimulationComputeContext& compute, float& out_consumed_mass) {
    out_consumed_mass = 0.0f;
    if (!(requested_mass > 0.0f) || !std::isfinite(requested_mass)) return false;
    auto it = fields_.find(object_key);
    if (it == fields_.end()) return false;
    MaterialStateField& field = it->second;
    if (!refreshHostState(compute, field)) return false;

    const auto& profile = findSubstance(field.substance_name);
    const float capacity_per_area = MaterialSubstance::fromProfile(
        profile, readback_scale_, field.overrides).mass_capacity;
    if (!(capacity_per_area > 0.0f)) return false;

    double available = 0.0;
    for (std::size_t i = 0; i < field.elementCount(); ++i) {
        const float mass = capacity_per_area * std::max(field.centers[i * 4u + 3u], 0.0f);
        available += std::clamp(field.state[i * MaterialStateField::kStateStride + 5u],
                                0.0f, 1.0f) * mass;
    }
    const float consume = std::min(requested_mass, static_cast<float>(available));
    if (!(consume > 0.0f)) return false;

    const std::vector<float> before = field.state;
    double remaining = consume;
    for (std::size_t i = 0; i < field.elementCount() && remaining > 0.0; ++i) {
        const std::size_t base = i * MaterialStateField::kStateStride;
        const float mass = capacity_per_area * std::max(field.centers[i * 4u + 3u], 0.0f);
        const double element_molten =
            std::clamp(field.state[base + 5u], 0.0f, 1.0f) * mass;
        if (!(element_molten > 0.0)) continue;
        const double take = std::min(remaining,
            consume * element_molten / available);
        field.state[base + 5u] = std::max(0.0f,
            field.state[base + 5u] - static_cast<float>(take / mass));
        field.state[base + 7u] += static_cast<float>(take);
        remaining -= take;
    }
    out_consumed_mass = consume - static_cast<float>(std::max(remaining, 0.0));
    if (!(out_consumed_mass > 0.0f)) return false;
    if (field.gpu_state.valid() &&
        !compute.uploadBuffer(field.gpu_state, field.state.data(),
                              field.state.size() * sizeof(float))) {
        field.state = before;
        out_consumed_mass = 0.0f;
        return false;
    }
    scatterCharMask(field, readback_scale_);
    return true;
}
} // namespace RayTrophiSim
