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
    double overflow = 0.0, negative = 0.0;
    uint32_t invalid = 0u;
    for (std::size_t i = 0; i < field.elementCount(); ++i) {
        const std::size_t base = i * MaterialStateField::kStateStride;
        const float area = std::max(field.centers[i * 4u + 3u], 0.0f);
        const float mass = capacity_per_area * area;

        // ── The real invariant, read straight off the buffer ──────────────────
        // Everything below the clamps is what the SHADER wrote. The shader's own
        // contract is `burned + melted + moved <= capacity`, each term >= 0. If
        // that ever breaks, two processes have spent the same kilogram, and this
        // is the only place in the pipeline that can still see it — one line
        // further down the clamps make the violation unrepresentable.
        const float raw_burned = field.state[base + 6u];
        const float raw_melt_fraction = field.state[base + 5u];
        const float raw_moved = field.state[base + 7u];
        if (!std::isfinite(raw_burned) || !std::isfinite(raw_melt_fraction) ||
            !std::isfinite(raw_moved)) {
            ++invalid;
        } else if (mass > 0.0f) {
            const float raw_melted = raw_melt_fraction * mass;
            overflow += std::max(0.0f,
                (raw_burned + raw_melted + raw_moved) - mass);
            negative += std::max(0.0f, -raw_burned) +
                        std::max(0.0f, -raw_melted) +
                        std::max(0.0f, -raw_moved);
        }

        // ── Reported masses: clamped, because every consumer downstream needs a
        // physically representable number even on a frame where the invariant is
        // violated. These deliberately sum to `mass`; that is a presentation
        // guarantee, NOT evidence of conservation.
        const float burned = std::clamp(raw_burned, 0.0f, mass);
        const float melt_fraction = std::clamp(raw_melt_fraction, 0.0f, 1.0f);
        const float melted = std::min(melt_fraction * mass, mass - burned);
        const float moved = std::clamp(raw_moved, 0.0f, mass - burned - melted);
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
    out.budget_overflow_mass = static_cast<float>(overflow);
    out.negative_mass = static_cast<float>(negative);
    out.invalid_elements = invalid;
    out.conservation_error = out.budget_overflow_mass + out.negative_mass;
    return out;
}

void MaterialStateFieldSystem::collectDamageSamples(
    const MaterialStateField& field,
    std::vector<MaterialDamageSample>& out, float minimum_weight) {
    const std::size_t count = std::min(
        field.elementCount(), field.state.size() / MaterialStateField::kStateStride);
    if (count == 0u || field.centers.size() < count * 4u) return;
    const auto& profile = findSubstance(field.substance_name);
    const MaterialSubstance substance = MaterialSubstance::fromProfile(
        profile, MaterialTemperatureScale{}, field.overrides);
    const float capacity_per_area = std::max(substance.mass_capacity, 0.0f);
    if (!(capacity_per_area > 0.0f)) return;

    // ★ THE SAME DEFINITION OF DAMAGE summarizeIntegrity uses: mass that has
    // left the solid, by any route. Deriving it a second way here would let the
    // object break somewhere its own integrity summary says it is still strong.
    for (std::size_t i = 0; i < count; ++i) {
        const std::size_t base = i * MaterialStateField::kStateStride;
        const float area = std::max(field.centers[i * 4u + 3u], 1e-8f);
        const float capacity = std::max(capacity_per_area * area, 1e-8f);
        const float burned = std::clamp(field.state[base + 6u], 0.0f, capacity);
        const float melted = std::clamp(field.state[base + 5u], 0.0f, 1.0f) * capacity;
        const float moved = std::clamp(field.state[base + 7u], 0.0f, capacity);
        const float weight = std::clamp((burned + melted + moved) / capacity, 0.0f, 1.0f);
        if (weight < minimum_weight) continue;
        MaterialDamageSample sample;
        sample.position = Vec3(field.centers[i * 4u + 0u],
                               field.centers[i * 4u + 1u],
                               field.centers[i * 4u + 2u]);
        sample.weight = weight;
        out.push_back(sample);
    }
}

bool MaterialStateFieldSystem::consumeMoltenMass(
    const std::string& object_key, float requested_mass,
    SimulationComputeContext& compute, float& out_consumed_mass) {
    out_consumed_mass = 0.0f;
    if (!(requested_mass > 0.0f) || !std::isfinite(requested_mass)) return false;
    auto it = fields_.find(object_key);
    if (it == fields_.end()) return false;
    MaterialStateField& field = it->second;
    // ★ The per-frame readback (flushReadback) already ran before the transfer
    // pass and left the mirror current. Downloading again here cost a second
    // submit+fence stall per transferring object per frame for identical bytes.
    if (!field.host_state_fresh && !refreshHostState(compute, field)) return false;

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
        // ★ Read and write the SAME clamped fraction. Subtracting a debit derived
        // from the clamped value out of an unclamped one would leave a residual
        // that the next pass sees as still-molten mass, i.e. the same kilogram
        // handed to APIC twice.
        const float melt_fraction = std::clamp(field.state[base + 5u], 0.0f, 1.0f);
        const double element_molten = melt_fraction * mass;
        if (!(element_molten > 0.0)) continue;
        const double take = std::min(remaining,
            consume * element_molten / available);
        field.state[base + 5u] = std::max(0.0f,
            melt_fraction - static_cast<float>(take / mass));
        field.state[base + 7u] += static_cast<float>(take);
        remaining -= take;
    }
    out_consumed_mass = consume - static_cast<float>(std::max(remaining, 0.0));
    if (!(out_consumed_mass > 0.0f)) return false;
    if (field.gpu_state.valid() &&
        !compute.uploadBuffer(field.gpu_state, field.state.data(),
                              field.elementCount() *
                                  MaterialStateField::kStateStride *
                                  sizeof(float))) {
        field.state = before;
        out_consumed_mass = 0.0f;
        return false;
    }
    // ★ NO scatterCharMask here, deliberately.
    //
    // The mask is only ever re-read by consumers gated on `mask_revision`, and a
    // debit does not bump it — so rebuilding the mask now produced bytes nobody
    // would look at before the next readback rebuilt them again. That was a full
    // CPU mask rebuild per transferring object per frame, thrown away.
    //
    // The debit also cannot change what the mask encodes in a way geometry cares
    // about: melt leaves the reservoir and reappears as `transferred`, so the
    // surface has not un-melted. Melt geometry must keep showing the loss.
    //
    // Host and device now hold the same bytes, so the mirror stays usable.
    field.host_state_fresh = true;
    return true;
}
} // namespace RayTrophiSim
