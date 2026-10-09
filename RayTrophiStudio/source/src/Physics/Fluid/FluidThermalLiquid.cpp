// Thermal liquid: cooling, temperature-dependent viscosity, freezing.
// See include/Fluid/FluidThermalLiquid.h for the model and the call order.
#include "Fluid/FluidThermalLiquid.h"
#include "Fluid/MatterSubstanceState.h"
#include "Fluid/SubstanceTag.h"
#include "MaterialStateField.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace RayTrophiSim {
namespace Fluid {

namespace {

inline bool particleCell(const FluidSim::FluidGrid& grid, const Vec3& wp,
                         float inv_h, int& i, int& j, int& k) {
    if (!std::isfinite(wp.x) || !std::isfinite(wp.y) || !std::isfinite(wp.z)) return false;
    const Vec3 g = (wp - grid.origin) * inv_h;
    i = static_cast<int>(std::floor(g.x));
    j = static_cast<int>(std::floor(g.y));
    k = static_cast<int>(std::floor(g.z));
    return i >= 0 && i < grid.nx && j >= 0 && j < grid.ny && k >= 0 && k < grid.nz;
}

constexpr int kFaceOffsets[6][3] = {
    {-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}
};

// A parcel whose mass burned/evaporated away still sits in the arrays until
// compaction. It must not cool, freeze or hold a cell of support.
inline bool liveParcel(const FluidParticles& parts, std::size_t p) {
    return p >= parts.mass_fraction.size() || parts.mass_fraction[p] > 0.02f;
}

void measureTemperatures(const FluidParticles& parts, ThermalLiquidStats& stats) {
    const std::size_t n = std::min(parts.size(), parts.temperature.size());
    double sum = 0.0;
    std::size_t count = 0;
    float lo = std::numeric_limits<float>::max();
    float hi = -std::numeric_limits<float>::max();
    for (std::size_t p = 0; p < n; ++p) {
        if (!liveParcel(parts, p)) continue;
        const float t = parts.temperature[p];
        if (!std::isfinite(t)) continue;
        lo = std::min(lo, t);
        hi = std::max(hi, t);
        sum += t;
        ++count;
    }
    if (count == 0) return;
    stats.min_kelvin = lo;
    stats.max_kelvin = hi;
    stats.mean_kelvin = static_cast<float>(sum / static_cast<double>(count));
}

} // namespace

void coolThermalLiquid(FluidParticles& particles,
                       const FluidSim::FluidGrid& grid,
                       const APICSolverParams& params,
                       float ambient_kelvin,
                       float dt,
                       ThermalLiquidStats& stats) {
    const std::size_t n = particles.size();
    const std::size_t cells = static_cast<std::size_t>(grid.nx) *
                              static_cast<std::size_t>(grid.ny) *
                              static_cast<std::size_t>(grid.nz);
    if (!params.thermal_liquid_enabled || n == 0 || cells == 0 ||
        grid.voxel_size <= 0.0f || dt <= 0.0f) {
        return;
    }
    // ★ A parcel with no temperature is not a cold parcel, it is an unwritten
    // one (see FluidParticles::temperature). Give it the ambient rather than
    // letting it read as 0 K and freeze on its first contact.
    if (particles.temperature.size() < n)
        particles.temperature.resize(n, ambient_kelvin);

    // Cell occupancy: which cells hold liquid. An EMPTY, non-solid neighbour
    // cell is air, and a parcel in a cell with an air face is at the surface.
    static std::vector<uint8_t> s_occupied;
    s_occupied.assign(cells, 0u);
    const float inv_h = 1.0f / grid.voxel_size;
    for (std::size_t p = 0; p < n; ++p) {
        if (!liveParcel(particles, p)) continue;
        int i, j, k;
        if (!particleCell(grid, particles.position[p], inv_h, i, j, k)) continue;
        s_occupied[grid.cellIndex(i, j, k)] = 1u;
    }

    const bool closed_walls =
        params.boundary == APICSolverParams::BoundaryMode::Closed;
    const bool has_solid = grid.solid.size() == cells;
    const float a_air = 1.0f - std::exp(-std::max(0.0f, params.thermal_air_cooling_rate) * dt);
    const float a_contact =
        1.0f - std::exp(-std::max(0.0f, params.thermal_contact_cooling_rate) * dt);

    int64_t air_count = 0;
    int64_t contact_count = 0;
    const int64_t count = static_cast<int64_t>(n);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+:air_count, contact_count) if(count > 32768)
#endif
    for (int64_t raw = 0; raw < count; ++raw) {
        const std::size_t p = static_cast<std::size_t>(raw);
        if (!liveParcel(particles, p)) continue;
        int i, j, k;
        if (!particleCell(grid, particles.position[p], inv_h, i, j, k)) continue;
        bool air = false;
        bool contact = false;
        for (const auto& o : kFaceOffsets) {
            const int ni = i + o[0], nj = j + o[1], nk = k + o[2];
            if (ni < 0 || ni >= grid.nx || nj < 0 || nj >= grid.ny ||
                nk < 0 || nk >= grid.nz) {
                // A closed wall is a container surface at ambient; an open or
                // periodic one is more air (or more liquid) beyond the box.
                if (closed_walls) contact = true; else air = true;
                continue;
            }
            const std::size_t nc = grid.cellIndex(ni, nj, nk);
            // ★ Only the COLLIDER bit counts as a cold surface. The solid-phase
            // overlay (value 2) is the liquid's own frozen parcels, and cooling
            // a parcel "by contact" with its own kind would make every layer
            // quench the next one instantly.
            if (has_solid && grid.solid[nc] == FluidSim::FluidGrid::kSolidCollider) {
                contact = true;
            } else if (!(has_solid && grid.solid[nc] != 0u) && s_occupied[nc] == 0u) {
                air = true;
            }
        }
        if (!air && !contact) continue;
        float t = particles.temperature[p];
        if (!std::isfinite(t)) t = ambient_kelvin;
        if (contact) { t += (ambient_kelvin - t) * a_contact; ++contact_count; }
        if (air)     { t += (ambient_kelvin - t) * a_air;     ++air_count; }
        particles.temperature[p] = t;
    }
    stats.air_cooled_particles = static_cast<std::size_t>(air_count);
    stats.contact_cooled_particles = static_cast<std::size_t>(contact_count);
    stats.measured = true;
    measureTemperatures(particles, stats);
}

void updateThermalFreeze(FluidParticles& particles,
                         const FluidSim::FluidGrid& grid,
                         const APICSolverParams& params,
                         ThermalLiquidStats& stats) {
    const std::size_t n = particles.size();
    if (particles.flags.size() < n) particles.flags.resize(n, 0u);

    // Disabled or granular: no parcel may stay frozen. Clearing here (rather
    // than leaving the flag and ignoring it) is what makes the switch honest —
    // a stale flag would keep the parcel pinned through the solid-phase path,
    // which reads the flag without knowing about this feature.
    if (!params.thermal_liquid_enabled || params.granular_enabled) {
        for (std::size_t p = 0; p < n; ++p) particles.flags[p] &= ~kParticleFlagFrozen;
        return;
    }
    const std::size_t cells = static_cast<std::size_t>(grid.nx) *
                              static_cast<std::size_t>(grid.ny) *
                              static_cast<std::size_t>(grid.nz);
    if (n == 0 || cells == 0 || grid.voxel_size <= 0.0f ||
        particles.temperature.size() < n) {
        return;
    }

    // Support from parcels ALREADY frozen at the start of this pass. Freezing
    // does not feed back within the pass, so the front advances at most one
    // cell per frame — from the contact surface outward.
    static std::vector<uint8_t> s_frozen_cell;
    s_frozen_cell.assign(cells, 0u);
    const float inv_h = 1.0f / grid.voxel_size;
    for (std::size_t p = 0; p < n; ++p) {
        if (!isFrozenParticle(particles, p) || !liveParcel(particles, p)) continue;
        int i, j, k;
        if (!particleCell(grid, particles.position[p], inv_h, i, j, k)) continue;
        s_frozen_cell[grid.cellIndex(i, j, k)] = 1u;
    }

    const bool closed_walls =
        params.boundary == APICSolverParams::BoundaryMode::Closed;
    const bool has_solid = grid.solid.size() == cells;
    const auto freezeKelvinFor = [&](std::size_t p) {
        const uint32_t tag = p < particles.substance_tag.size()
            ? particles.substance_tag[p] : kSubstanceUntagged;
        return substanceFreezeKelvin(tag, params);
    };

    auto supported = [&](int i, int j, int k) -> bool {
        if (s_frozen_cell[grid.cellIndex(i, j, k)]) return true;
        for (const auto& o : kFaceOffsets) {
            const int ni = i + o[0], nj = j + o[1], nk = k + o[2];
            if (ni < 0 || ni >= grid.nx || nj < 0 || nj >= grid.ny ||
                nk < 0 || nk >= grid.nz) {
                if (closed_walls) return true;
                continue;
            }
            const std::size_t nc = grid.cellIndex(ni, nj, nk);
            if (s_frozen_cell[nc]) return true;
            if (has_solid && grid.solid[nc] == FluidSim::FluidGrid::kSolidCollider) return true;
        }
        return false;
    };

    int64_t froze = 0, melted = 0, frozen_total = 0, cold_unsupported = 0;
    const int64_t count = static_cast<int64_t>(n);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+:froze, melted, frozen_total, cold_unsupported) if(count > 32768)
#endif
    for (int64_t raw = 0; raw < count; ++raw) {
        const std::size_t p = static_cast<std::size_t>(raw);
        const auto model = p < particles.constitutive_model.size()
            ? static_cast<MatterConstitutiveModel>(particles.constitutive_model[p])
            : MatterConstitutiveModel::Auto;
        if (!liveParcel(particles, p) || model == MatterConstitutiveModel::Granular ||
            model == MatterConstitutiveModel::Elastic) {
            particles.flags[p] &= ~kParticleFlagFrozen;
            continue;
        }
        const float t = particles.temperature[p];
        const float freeze_k = freezeKelvinFor(p);
        uint32_t& f = particles.flags[p];
        // A non-meltable parcel has no freeze point (-inf). It must never carry
        // the frozen flag: with melt_k = -inf the melt test below would clear it
        // on the first step, so the flag would flicker for nothing. Clear it here
        // and leave the parcel to the particle solver untouched.
        if (!std::isfinite(freeze_k)) {
            f &= ~kParticleFlagFrozen;
            continue;
        }
        const uint32_t tag = p < particles.substance_tag.size()
            ? particles.substance_tag[p] : kSubstanceUntagged;
        const float melt_k = substanceMeltReleaseKelvin(tag, params);
        if (f & kParticleFlagFrozen) {
            if (std::isfinite(t) && t > melt_k) {
                f &= ~kParticleFlagFrozen;
                ++melted;
                continue;
            }
            ++frozen_total;
        } else {
            if (!(std::isfinite(t) && t < freeze_k)) continue;
            int i, j, k;
            if (!particleCell(grid, particles.position[p], inv_h, i, j, k) ||
                !supported(i, j, k)) {
                ++cold_unsupported;
                continue;
            }
            f |= kParticleFlagFrozen;
            ++froze;
            ++frozen_total;
        }
        // Pinned: a frozen parcel neither flows nor falls. Zeroed here for the
        // host arrays; Fluid::step re-zeroes after its force stage, because
        // gravity is added again every step.
        if (p < particles.velocity.size()) particles.velocity[p] = Vec3(0.0f, 0.0f, 0.0f);
        if (p < particles.affine.size()) particles.affine[p] = AffineC{};
    }
    stats.froze_this_frame = static_cast<std::size_t>(froze);
    stats.melted_this_frame = static_cast<std::size_t>(melted);
    stats.frozen_particles = static_cast<std::size_t>(frozen_total);
    stats.cold_unsupported = static_cast<std::size_t>(cold_unsupported);
    stats.measured = true;
}

bool buildThermalViscosityField(const FluidParticles& particles,
                                const FluidSim::FluidGrid& grid,
                                const APICSolverParams& params,
                                const std::vector<float>* base,
                                std::vector<float>& viscosity_out,
                                ThermalLiquidStats* stats) {
    const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    const std::size_t cells = static_cast<std::size_t>(nx) *
                              static_cast<std::size_t>(ny) *
                              static_cast<std::size_t>(nz);
    const std::size_t n = particles.size();
    if (!params.thermal_liquid_enabled || params.granular_enabled ||
        n == 0 || cells == 0 || grid.voxel_size <= 0.0f ||
        particles.temperature.size() < n) {
        viscosity_out.clear();
        return false;
    }
    const bool have_base = base != nullptr && base->size() == cells;
    const float hot_scalar = std::max(0.0f, params.kinematic_viscosity);

    // Parcel viscosity gathered onto cell centres with the transfer's
    // trilinear support. Each parcel uses its own temperature and substance.
    static std::vector<float> s_nu_sum;
    static std::vector<float> s_weight;
    s_nu_sum.assign(cells, 0.0f);
    s_weight.assign(cells, 0.0f);
    const float inv_h = 1.0f / grid.voxel_size;
    for (std::size_t p = 0; p < n; ++p) {
        if (!liveParcel(particles, p)) continue;
        const auto model = p < particles.constitutive_model.size()
            ? static_cast<MatterConstitutiveModel>(particles.constitutive_model[p])
            : MatterConstitutiveModel::Auto;
        if (model == MatterConstitutiveModel::Granular || model == MatterConstitutiveModel::Elastic) {
            continue;
        }
        const float t = particles.temperature[p];
        const Vec3& wp = particles.position[p];
        if (!std::isfinite(t) || !std::isfinite(wp.x) || !std::isfinite(wp.y) ||
            !std::isfinite(wp.z)) continue;
        const uint32_t tag = p < particles.substance_tag.size()
            ? particles.substance_tag[p] : kSubstanceUntagged;
        const float nu = substanceThermalViscosity(tag, t, params);
        const Vec3 c = (wp - grid.origin) * inv_h - Vec3(0.5f, 0.5f, 0.5f);
        const int i0 = static_cast<int>(std::floor(c.x));
        const int j0 = static_cast<int>(std::floor(c.y));
        const int k0 = static_cast<int>(std::floor(c.z));
        const float fx = c.x - static_cast<float>(i0);
        const float fy = c.y - static_cast<float>(j0);
        const float fz = c.z - static_cast<float>(k0);
        for (int dk = 0; dk <= 1; ++dk)
        for (int dj = 0; dj <= 1; ++dj)
        for (int di = 0; di <= 1; ++di) {
            const int i = i0 + di, j = j0 + dj, k = k0 + dk;
            if (i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k >= nz) continue;
            const float w = (di ? fx : 1.0f - fx) * (dj ? fy : 1.0f - fy) *
                            (dk ? fz : 1.0f - fz);
            if (w <= 0.0f) continue;
            const std::size_t ci = grid.cellIndex(i, j, k);
            s_nu_sum[ci] += w * nu;
            s_weight[ci] += w;
        }
    }

    // Evaluate each parcel's own material curve BEFORE gathering. Averaging
    // temperatures first and applying the domain curve freezes Water and Wax
    // at the same threshold, even when the per-cell base viscosity is correct.
    viscosity_out.resize(cells);
    const int64_t cell_count = static_cast<int64_t>(cells);
    // No min/max reduction clause: MSVC's OpenMP 2.0 has none. The readout is
    // taken in a separate serial pass below.
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t raw = 0; raw < cell_count; ++raw) {
        const std::size_t ci = static_cast<std::size_t>(raw);
        const float hot = have_base ? std::max(0.0f, (*base)[ci]) : hot_scalar;
        if (s_weight[ci] <= 1e-8f) {
            // No liquid here: the hot value, exactly what the field would have
            // been without this feature. Zero would be the inviscid answer and
            // carve a frictionless pocket wherever the splat did not reach.
            viscosity_out[ci] = hot;
            continue;
        }
        viscosity_out[ci] = s_nu_sum[ci] / s_weight[ci];
    }
    float lo = std::numeric_limits<float>::max();
    float hi = 0.0f;
    for (std::size_t ci = 0; ci < cells; ++ci) {
        if (s_weight[ci] <= 1e-8f) continue;   // only cells liquid actually holds
        lo = std::min(lo, viscosity_out[ci]);
        hi = std::max(hi, viscosity_out[ci]);
    }
    if (stats) {
        stats->viscosity_field_built = true;
        stats->min_viscosity = (hi >= lo) ? lo : 0.0f;
        stats->max_viscosity = hi;
    }
    return true;
}

} // namespace Fluid
} // namespace RayTrophiSim
