#pragma once

// One reader of the per-particle constitutive state for every backend.
//
// ★★★ There used to be three copies of this loop (CPU step, single-model GPU
// step, nothing on the Matter path). The Matter path is where Snow runs, so a
// 4 m/s impact reported 0 yielded, 0 damaged, 0 plastic for 120 steps: the
// device state was downloaded every frame and never counted. A zero from an
// uncounted backend is indistinguishable from "the material never yielded",
// which is why `granular_material_measured` exists and why every backend now
// calls this one function.

#include "APICFluidSolver.h"
#include "FluidParticles.h"
#include "GranularConstitutive.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace RayTrophiSim::Fluid::Granular {

// `include(i)` selects the granular parcels; a mixed domain also carries liquid
// parcels whose granular sidecars are never written and must not dilute means.
template <class Include>
inline void countMaterialState(const FluidParticles& particles, APICSolverStats& out,
                               Include include) {
    out.granular_yielded_particles = 0;
    out.granular_detached_particles = 0;
    out.granular_invalid_particles = 0;
    out.granular_sleeping_particles = 0;
    out.granular_strain_limited_particles = 0;
    out.granular_compaction_capped_particles = 0;
    out.granular_hardening_capped_particles = 0;
    out.granular_damaged_particles = 0;
    out.granular_damage_over_10_particles = 0;
    out.granular_damage_over_50_particles = 0;
    out.granular_damage_over_90_particles = 0;
    out.granular_max_yield_value = 0.0f;
    out.granular_max_plastic_increment = 0.0f;
    out.granular_max_accumulated_plastic = 0.0f;
    out.granular_mean_accumulated_plastic = 0.0f;
    out.granular_max_fracture_history = 0.0f;
    out.granular_mean_fracture_history = 0.0f;
    out.granular_max_damage = 0.0f;
    out.granular_mean_damage = 0.0f;
    out.granular_min_plastic_jacobian = 1.0f;
    out.granular_max_plastic_jacobian = 1.0f;
    out.granular_mean_plastic_jacobian = 1.0f;
    out.granular_counted_particles = 0;

    const std::size_t n = particles.size();
    if (particles.granular_material_flags.size() != n ||
        particles.granular_damage.size() != n ||
        particles.granular_yield_value.size() != n ||
        particles.granular_plastic_increment.size() != n ||
        particles.granular_hardening.size() != n ||
        particles.granular_fracture_history.size() != n ||
        particles.granular_plastic_volume.size() != n) {
        out.granular_material_measured = false;
        return;
    }
    double damage_sum = 0.0, plastic_sum = 0.0, bond_sum = 0.0, jp_sum = 0.0;
    std::size_t counted = 0, jp_samples = 0;
    float jp_min = 0.0f, jp_max = 0.0f;
    for (std::size_t p = 0; p < n; ++p) {
        if (!include(p)) continue;
        ++counted;
        const uint32_t f = particles.granular_material_flags[p];
        out.granular_yielded_particles += (f & kFlagYielded) != 0u;
        out.granular_detached_particles += (f & kFlagDetached) != 0u;
        out.granular_invalid_particles += (f & kFlagInvalid) != 0u;
        out.granular_sleeping_particles += (f & kFlagSleeping) != 0u;
        out.granular_strain_limited_particles += (f & kFlagStrainLimited) != 0u;
        out.granular_compaction_capped_particles += (f & kFlagCompactionCapped) != 0u;
        out.granular_hardening_capped_particles += (f & kFlagHardeningCapped) != 0u;
        const float dmg = particles.granular_damage[p];
        if (std::isfinite(dmg)) {
            out.granular_max_damage = std::max(out.granular_max_damage, dmg);
            out.granular_damaged_particles += dmg > 1.0e-4f;
            out.granular_damage_over_10_particles += dmg >= 0.10f;
            out.granular_damage_over_50_particles += dmg >= 0.50f;
            out.granular_damage_over_90_particles += dmg >= 0.90f;
            damage_sum += dmg;
        }
        const float yv = particles.granular_yield_value[p];
        if (std::isfinite(yv))
            out.granular_max_yield_value = std::max(out.granular_max_yield_value, yv);
        const float pi = particles.granular_plastic_increment[p];
        if (std::isfinite(pi))
            out.granular_max_plastic_increment = std::max(out.granular_max_plastic_increment, pi);
        const float ap = particles.granular_hardening[p];
        if (std::isfinite(ap)) {
            out.granular_max_accumulated_plastic =
                std::max(out.granular_max_accumulated_plastic, ap);
            plastic_sum += ap;
        }
        const float bo = particles.granular_fracture_history[p];
        if (std::isfinite(bo)) {
            out.granular_max_fracture_history = std::max(out.granular_max_fracture_history, bo);
            bond_sum += bo;
        }
        // plastic_volume is compactness (> 1 compacted), so J_P = 1 / pv.
        const float pv = particles.granular_plastic_volume[p];
        if (std::isfinite(pv) && pv > 0.0f) {
            const float jp = 1.0f / pv;
            jp_min = jp_samples ? std::min(jp_min, jp) : jp;
            jp_max = jp_samples ? std::max(jp_max, jp) : jp;
            jp_sum += jp;
            ++jp_samples;
        }
    }
    out.granular_counted_particles = counted;
    out.granular_material_measured = true;
    if (counted) {
        const double inv = 1.0 / static_cast<double>(counted);
        out.granular_mean_damage = static_cast<float>(damage_sum * inv);
        out.granular_mean_accumulated_plastic = static_cast<float>(plastic_sum * inv);
        out.granular_mean_fracture_history = static_cast<float>(bond_sum * inv);
    }
    if (jp_samples) {
        out.granular_min_plastic_jacobian = jp_min;
        out.granular_max_plastic_jacobian = jp_max;
        out.granular_mean_plastic_jacobian =
            static_cast<float>(jp_sum / static_cast<double>(jp_samples));
    }
}

inline void countMaterialState(const FluidParticles& particles, APICSolverStats& out) {
    countMaterialState(particles, out, [](std::size_t) { return true; });
}

} // namespace RayTrophiSim::Fluid::Granular
