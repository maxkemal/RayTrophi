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
                       ThermalLiquidStats& stats,
                       const std::vector<MaterialStateFieldSystem::ThermalSource>*
                           heat_sources,
                       bool air_through_gas,
                       const std::vector<float>* solid_capacity) {
    const std::size_t n = particles.size();
    const std::size_t cells = static_cast<std::size_t>(grid.nx) *
                              static_cast<std::size_t>(grid.ny) *
                              static_cast<std::size_t>(grid.nz);
    const bool fields = heat_sources && !heat_sources->empty();
    if ((!params.thermal_liquid_enabled && !fields) || n == 0 || cells == 0 ||
        grid.voxel_size <= 0.0f || dt <= 0.0f) {
        return;
    }
    const bool chain = params.thermal_liquid_enabled;
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
    // Local ambient = domain ambient + every field's delta at the parcel;
    // `heated` is set when any field reaches it.
    const auto localAmbient = [&](std::size_t p, bool& heated) {
        heated = false;
        float local = ambient_kelvin;
        if (!fields) return local;
        const auto& x = particles.position[p];
        for (const auto& s : *heat_sources) {
            const float w = thermalSourceWeight(s, x.x, x.y, x.z);
            if (w <= 0.0f) continue;
            local += s.delta_kelvin * w;
            heated = true;
        }
        return std::max(1.0f, local);
    };

    const bool closed_walls =
        params.boundary == APICSolverParams::BoundaryMode::Closed;
    const bool has_solid = grid.solid.size() == cells;
    const float a_air = 1.0f - std::exp(-std::max(0.0f, params.thermal_air_cooling_rate) * dt);
    const float a_contact =
        1.0f - std::exp(-std::max(0.0f, params.thermal_contact_cooling_rate) * dt);

    int64_t air_count = 0;
    int64_t contact_count = 0;
    int64_t heated_count = 0;
    const int64_t count = static_cast<int64_t>(n);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+:air_count, contact_count, heated_count) if(count > 32768)
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
                // A surface with an MSF shell trades real heat in
                // exchangeLiquidSolidContact instead of being an ambient sink.
                if (!(solid_capacity && solid_capacity->size() == cells &&
                      (*solid_capacity)[nc] > 0.0f)) {
                    contact = true;
                }
            } else if (!(has_solid && grid.solid[nc] != 0u) && s_occupied[nc] == 0u) {
                air = true;
            }
        }
        bool heated = false;
        const float local = localAmbient(p, heated);
        if (!chain) { air = false; contact = false; }
        // A Matter domain's air is its own gas grid: the surface exchange
        // happens there (exchangeLiquidGasSurface), with energy conserved,
        // instead of against an infinite ambient reservoir here.
        if (air_through_gas) air = false;
        if (!air && !contact && !heated) continue;
        float t = particles.temperature[p];
        if (!std::isfinite(t)) t = ambient_kelvin;
        if (heated && !contact) { t += (local - t) * a_contact; ++heated_count; }
        if (contact) { t += (local - t) * a_contact; ++contact_count; }
        if (air)     { t += (local - t) * a_air;     ++air_count; }
        particles.temperature[p] = t;
    }
    stats.air_cooled_particles = static_cast<std::size_t>(air_count);
    stats.contact_cooled_particles = static_cast<std::size_t>(contact_count);
    stats.field_heated_particles = static_cast<std::size_t>(heated_count);
    stats.measured = true;
    measureTemperatures(particles, stats);
}

void exchangeLiquidSolidContact(FluidParticles& particles,
                                const FluidSim::FluidGrid& grid,
                                const APICSolverParams& params,
                                float dt,
                                const std::vector<float>& solid_capacity,
                                std::vector<float>& solid_kelvin,
                                std::vector<float>& solid_heat_j,
                                ThermalLiquidStats& stats) {
    const std::size_t n = std::min(particles.size(), particles.temperature.size());
    const std::size_t cells = static_cast<std::size_t>(grid.nx) * grid.ny * grid.nz;
    solid_heat_j.assign(cells, 0.0f);
    if (n == 0 || cells == 0 || solid_capacity.size() != cells ||
        solid_kelvin.size() != cells || !(grid.voxel_size > 0.0f) || !(dt > 0.0f)) {
        return;
    }
    constexpr double kContactFilm = MaterialStateFieldSystem::kMsfContactFilmWm2K;
    const float h = grid.voxel_size;
    const float inv_h = 1.0f / h;
    // Parcels per cell share the cell's contact face.
    static std::vector<uint16_t> s_count;
    s_count.assign(cells, 0u);
    for (std::size_t p = 0; p < n; ++p) {
        if (!liveParcel(particles, p)) continue;
        int i, j, k;
        if (!particleCell(grid, particles.position[p], inv_h, i, j, k)) continue;
        uint16_t& c = s_count[grid.cellIndex(i, j, k)];
        if (c < 0xffffu) ++c;
    }
    uint32_t cached_tag = ~0u;
    const SubstanceProfile* profile = nullptr;
    std::size_t touched = 0;
    double heat_out = 0.0;
    for (std::size_t p = 0; p < n; ++p) {
        if (!liveParcel(particles, p)) continue;
        double tp = particles.temperature[p];
        if (!std::isfinite(tp) || tp <= 0.0) continue;
        int i, j, k;
        if (!particleCell(grid, particles.position[p], inv_h, i, j, k)) continue;
        const std::size_t own = grid.cellIndex(i, j, k);
        // The shell cell: the parcel's own, else the first face neighbour.
        std::size_t shell = cells;
        if (solid_capacity[own] > 0.0f) shell = own;
        for (const auto& o : kFaceOffsets) {
            if (shell != cells) break;
            const int ni = i + o[0], nj = j + o[1], nk = k + o[2];
            if (ni < 0 || ni >= grid.nx || nj < 0 || nj >= grid.ny || nk < 0 || nk >= grid.nz) continue;
            const std::size_t nc = grid.cellIndex(ni, nj, nk);
            if (solid_capacity[nc] > 0.0f) shell = nc;
        }
        if (shell == cells) continue;
        const uint32_t tag = p < particles.substance_tag.size()
            ? particles.substance_tag[p] : kSubstanceUntagged;
        if (tag != cached_tag) {
            profile = particleSubstance(tag, params);
            cached_tag = tag;
        }
        if (!profile) continue;
        const double cp = parcelHeatCapacity(particles, p, *profile, h, params.particles_per_cell);
        const double cs = solid_capacity[shell];
        if (!(cp > 0.0) || !(cs > 0.0)) continue;
        const double ts = solid_kelvin[shell];
        const double area = static_cast<double>(h) * h / std::max<int>(1, s_count[own]);
        const double g = kContactFilm * area;
        const double decay = 1.0 - std::exp(-g * dt * (1.0 / cp + 1.0 / cs));
        const double eq = (cp * tp + cs * ts) / (cp + cs);
        const double dtp = (eq - tp) * decay;
        const double q_to_solid = -cp * dtp;
        particles.temperature[p] = static_cast<float>(tp + dtp);
        solid_kelvin[shell] = static_cast<float>(ts + q_to_solid / cs);
        solid_heat_j[shell] += static_cast<float>(q_to_solid);
        heat_out += q_to_solid;
        ++touched;
    }
    stats.solid_contact_particles = touched;
    stats.heat_to_solid_j = heat_out;
}

double parcelHeatCapacity(const FluidParticles& particles, std::size_t p,
                          const SubstanceProfile& profile, float voxel_size,
                          int particles_per_cell) {
    const float fraction = p < particles.mass_fraction.size()
        ? std::clamp(particles.mass_fraction[p], 0.0f, 1.0f) : 1.0f;
    double mass = 0.0;
    if (p < particles.rest_mass_kg.size() && particles.rest_mass_kg[p] > 0.0f) {
        mass = static_cast<double>(particles.rest_mass_kg[p]) * fraction;
    } else {
        const double rho = profile.liquid_density > 0.0f ? profile.liquid_density
                         : (profile.density > 0.0f ? profile.density : 1000.0f);
        mass = rho * voxel_size * voxel_size * voxel_size /
               std::max(1, particles_per_cell) * fraction;
    }
    const double c = profile.specific_heat > 0.0f ? profile.specific_heat : 1000.0;
    return mass * c;
}

void conductThermalLiquid(FluidParticles& particles,
                          const FluidSim::FluidGrid& grid,
                          const APICSolverParams& params,
                          float dt,
                          ThermalLiquidStats& stats) {
    const std::size_t n = std::min(particles.size(), particles.temperature.size());
    const std::size_t cells = static_cast<std::size_t>(grid.nx) * grid.ny * grid.nz;
    if (n == 0 || cells == 0 || !(grid.voxel_size > 0.0f) || !(dt > 0.0f)) return;
    const float h = grid.voxel_size;
    const float inv_h = 1.0f / h;

    // Gather per cell: heat capacity C (J/K), energy C*T, capacity-weighted k.
    static std::vector<double> s_cap, s_energy, s_kcap;
    static std::vector<int> s_cell_of;
    s_cap.assign(cells, 0.0);
    s_energy.assign(cells, 0.0);
    s_kcap.assign(cells, 0.0);
    s_cell_of.assign(n, -1);
    uint32_t cached_tag = ~0u;
    const SubstanceProfile* profile = nullptr;
    for (std::size_t p = 0; p < n; ++p) {
        if (!liveParcel(particles, p)) continue;
        const float t = particles.temperature[p];
        if (!std::isfinite(t) || t <= 0.0f) continue;
        int i, j, k;
        if (!particleCell(grid, particles.position[p], inv_h, i, j, k)) continue;
        const uint32_t tag = p < particles.substance_tag.size()
            ? particles.substance_tag[p] : kSubstanceUntagged;
        if (tag != cached_tag) {
            profile = particleSubstance(tag, params);
            cached_tag = tag;
        }
        if (!profile) continue;
        const double cap = parcelHeatCapacity(particles, p, *profile, h,
                                              params.particles_per_cell);
        if (!(cap > 0.0)) continue;
        const std::size_t c = grid.cellIndex(i, j, k);
        s_cell_of[p] = static_cast<int>(c);
        s_cap[c] += cap;
        s_energy[c] += cap * t;
        s_kcap[c] += cap * std::max(0.0f, profile->conductivity);
    }

    // Cell temperatures before and after; faces solved pairwise, implicitly:
    // for two bodies C_a, C_b joined by conductance G the exact exchange over
    // dt is dT = (Tb - Ta) * (1 - exp(-G dt (1/Ca + 1/Cb))) / (1 + Ca/Cb).
    static std::vector<double> s_t0, s_t;
    s_t0.assign(cells, 0.0);
    s_t.assign(cells, 0.0);
    std::size_t live_cells = 0;
    for (std::size_t c = 0; c < cells; ++c) {
        if (s_cap[c] > 0.0) {
            s_t0[c] = s_t[c] = s_energy[c] / s_cap[c];
            ++live_cells;
        }
    }
    if (live_cells < 2) return;
    const double area = static_cast<double>(h) * h;
    // Three positive-direction faces per cell; red-black would be symmetric,
    // a single sweep is first-order and conservative, enough at these rates.
    const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const std::size_t a = grid.cellIndex(i, j, k);
                if (!(s_cap[a] > 0.0)) continue;
                const int nb[3][3] = {{i + 1, j, k}, {i, j + 1, k}, {i, j, k + 1}};
                for (const auto& o : nb) {
                    if (o[0] >= nx || o[1] >= ny || o[2] >= nz) continue;
                    const std::size_t b = grid.cellIndex(o[0], o[1], o[2]);
                    if (!(s_cap[b] > 0.0)) continue;
                    const double ka = s_kcap[a] / s_cap[a];
                    const double kb = s_kcap[b] / s_cap[b];
                    if (!(ka > 0.0) || !(kb > 0.0)) continue;
                    const double kf = 2.0 * ka * kb / (ka + kb);       // harmonic mean
                    const double g = kf * area / h;                    // W/K
                    const double ca = s_cap[a], cb = s_cap[b];
                    const double decay = 1.0 - std::exp(-g * dt * (1.0 / ca + 1.0 / cb));
                    const double eq = (ca * s_t[a] + cb * s_t[b]) / (ca + cb);
                    s_t[a] += (eq - s_t[a]) * decay;
                    s_t[b] += (eq - s_t[b]) * decay;
                }
            }
    for (std::size_t p = 0; p < n; ++p) {
        const int c = s_cell_of[p];
        if (c < 0) continue;
        particles.temperature[p] += static_cast<float>(s_t[c] - s_t0[c]);
    }
    stats.conducting_cells = live_cells;
}

std::size_t applyThermalConvection(FluidParticles& particles,
                                   const APICSolverParams& params,
                                   float dt,
                                   ThermalLiquidStats& stats) {
    const std::size_t n = std::min(particles.size(), particles.temperature.size());
    if (n == 0 || !(dt > 0.0f)) return 0;
    static std::vector<float> s_beta;
    s_beta.assign(n, 0.0f);
    uint32_t cached_tag = ~0u;
    float cached_beta = 0.0f;
    double sum = 0.0;
    std::size_t count = 0;
    for (std::size_t p = 0; p < n; ++p) {
        if (!liveParcel(particles, p)) continue;
        if (p < particles.flags.size() && (particles.flags[p] & kParticleFlagFrozen)) continue;
        if (p < particles.constitutive_model.size() &&
            particles.constitutive_model[p] != static_cast<uint8_t>(MatterConstitutiveModel::Fluid) &&
            particles.constitutive_model[p] != static_cast<uint8_t>(MatterConstitutiveModel::Auto)) {
            continue;
        }
        const float t = particles.temperature[p];
        if (!std::isfinite(t)) continue;
        const uint32_t tag = p < particles.substance_tag.size()
            ? particles.substance_tag[p] : kSubstanceUntagged;
        if (tag != cached_tag) {
            const SubstanceProfile* profile = particleSubstance(tag, params);
            cached_tag = tag;
            cached_beta = profile ? std::max(0.0f, profile->thermal_expansion_per_k) : 0.0f;
        }
        if (!(cached_beta > 0.0f)) continue;
        s_beta[p] = cached_beta;
        sum += t;
        ++count;
    }
    if (count < 2) return 0;
    const float mean = static_cast<float>(sum / static_cast<double>(count));
    const Vec3 g = params.gravity;
    std::size_t kicked = 0;
    for (std::size_t p = 0; p < n; ++p) {
        const float beta = s_beta[p];
        if (!(beta > 0.0f)) continue;
        const float anomaly = beta * (particles.temperature[p] - mean);
        if (std::fabs(anomaly) < 1.0e-7f) continue;
        particles.velocity[p] -= g * (anomaly * dt);
        ++kicked;
    }
    stats.convecting_particles = kicked;
    return kicked;
}

void updateThermalFreeze(FluidParticles& particles,
                         const FluidSim::FluidGrid& grid,
                         const APICSolverParams& params,
                         ThermalLiquidStats& stats,
                         bool heat_fields_present,
                         std::vector<float>* solid_change) {
    const std::size_t n = particles.size();
    if (particles.flags.size() < n) particles.flags.resize(n, 0u);
    // dF per parcel for the ledger. A parcel resolved for the first time this
    // pass reports only the change after resolution, never its birth state.
    if (solid_change) solid_change->assign(n, 0.0f);
    std::vector<float>& solid = particles.fusion_solid_fraction;
    solid.resize(n, std::numeric_limits<float>::quiet_NaN());

    // Liquid-model parcels freeze only with the chain on or a placed hot/cold
    // field. Off, no parcel may stay frozen: clearing here (rather than leaving
    // the flag and ignoring it) is what makes the switch honest - a stale flag
    // would keep the parcel pinned through the solid-phase path.
    // ★ Not gated on the domain's granular_enabled: a Matter domain that once
    // held snow kept its water from ever freezing.
    // Skeleton carriers (Snow, Ice) always pay their latent heat: whatever
    // warmed them (a field, the gas film) did it with real energy.
    const bool liquid_freezing = params.thermal_liquid_enabled || heat_fields_present;
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
    // Liquid occupancy: ice also nucleates at the FREE SURFACE (air above),
    // the way a pond freezes from the top. Without it a tank cooled from
    // above never froze - only wall/collider contact could start the front.
    static std::vector<uint8_t> s_liquid_cell;
    s_liquid_cell.assign(cells, 0u);
    const float inv_h = 1.0f / grid.voxel_size;
    for (std::size_t p = 0; p < n; ++p) {
        if (!liveParcel(particles, p)) continue;
        int i, j, k;
        if (!particleCell(grid, particles.position[p], inv_h, i, j, k)) continue;
        s_liquid_cell[grid.cellIndex(i, j, k)] = 1u;
    }
    for (std::size_t p = 0; p < n; ++p) {
        if (!isFrozenParticle(particles, p) || !liveParcel(particles, p)) continue;
        int i, j, k;
        if (!particleCell(grid, particles.position[p], inv_h, i, j, k)) continue;
        s_frozen_cell[grid.cellIndex(i, j, k)] = 1u;
    }

    const bool closed_walls =
        params.boundary == APICSolverParams::BoundaryMode::Closed;
    const bool has_solid = grid.solid.size() == cells;

    auto supported = [&](int i, int j, int k) -> bool {
        if (s_frozen_cell[grid.cellIndex(i, j, k)]) return true;
        if (j + 1 < grid.ny) {
            const std::size_t up = grid.cellIndex(i, j + 1, k);
            if (!s_liquid_cell[up] && !(has_solid && grid.solid[up] != 0u)) {
                return true;   // free surface: air above
            }
        }
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

    int64_t froze = 0, melted = 0, frozen_total = 0, cold_unsupported = 0, mushy = 0;
    double released_j = 0.0, absorbed_j = 0.0;
    const int64_t count = static_cast<int64_t>(n);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+:froze, melted, frozen_total, cold_unsupported, mushy, released_j, absorbed_j) if(count > 32768)
#endif
    for (int64_t raw = 0; raw < count; ++raw) {
        const std::size_t p = static_cast<std::size_t>(raw);
        uint32_t& f = particles.flags[p];
        float& fraction = solid[p];
        const uint32_t tag = p < particles.substance_tag.size()
            ? particles.substance_tag[p] : kSubstanceUntagged;
        const SubstanceProfile* profile = particleSubstance(tag, params);
        const float melt_k = substanceFreezeKelvin(tag, params);
        const auto model = p < particles.constitutive_model.size()
            ? static_cast<MatterConstitutiveModel>(particles.constitutive_model[p])
            : MatterConstitutiveModel::Auto;
        const bool skeleton = substanceHasSolidSkeleton(profile);
        // No freeze point (non-meltable, unknown substance) or a liquid run by
        // the legacy granular switch: no fusion state at all. A flag left here
        // would flicker or pin a parcel nothing can melt.
        if (!profile || !std::isfinite(melt_k) ||
            (!skeleton && (model == MatterConstitutiveModel::Granular ||
                           model == MatterConstitutiveModel::Elastic))) {
            f &= ~kParticleFlagFrozen;
            fraction = 0.0f;
            continue;
        }
        if (!std::isfinite(fraction)) {
            // First pass for this parcel: at birth (or before this sidecar
            // existed) it is all solid or all liquid.
            fraction = skeleton
                ? (model == MatterConstitutiveModel::Fluid ? 0.0f : 1.0f)
                : ((f & kParticleFlagFrozen) != 0u ? 1.0f : 0.0f);
        }
        if (!skeleton && !liquid_freezing) {
            f &= ~kParticleFlagFrozen;
            fraction = 0.0f;
            continue;
        }
        if (!liveParcel(particles, p)) {
            f &= ~kParticleFlagFrozen;
            continue;
        }
        float t = particles.temperature[p];
        if (!std::isfinite(t)) continue;

        const float before = fraction;
        const float latent = std::max(0.0f, profile->latent_heat_fusion);
        const float c = profile->specific_heat > 0.0f ? profile->specific_heat : 1000.0f;
        // Melting needs no support. Freezing a liquid needs a nucleus: a
        // support, or crystals already in the parcel. Without one the parcel
        // supercools, and that is reported, not hidden.
        bool nucleated = true;
        if (t < melt_k && fraction < 1.0f && !skeleton && !(fraction > 0.0f)) {
            int i, j, k;
            nucleated = particleCell(grid, particles.position[p], inv_h, i, j, k) &&
                supported(i, j, k);
            if (!nucleated) ++cold_unsupported;
        }
        if (!(latent > 0.0f)) {
            // No latent data: the state switches at the melt point, for free.
            if (t > melt_k) fraction = 0.0f;
            else if (t < melt_k && nucleated) fraction = 1.0f;
        } else if (t > melt_k && fraction > 0.0f) {
            // Superheat pays the latent heat; the parcel holds at the melt point.
            const float df = std::min(fraction, c * (t - melt_k) / latent);
            fraction -= df;
            t -= df * latent / c;
        } else if (t < melt_k && fraction < 1.0f && nucleated) {
            // Undercooling is paid back by the heat crystallisation releases.
            const float df = std::min(1.0f - fraction, c * (melt_k - t) / latent);
            fraction += df;
            t += df * latent / c;
        }
        if (fraction < 1.0e-6f) fraction = 0.0f;
        if (fraction > 1.0f - 1.0e-6f) fraction = 1.0f;
        particles.temperature[p] = t;
        const float changed = fraction - before;
        if (solid_change) (*solid_change)[p] = changed;
        if (changed != 0.0f && latent > 0.0f) {
            const float rest = p < particles.rest_mass_kg.size() ? particles.rest_mass_kg[p] : 0.0f;
            const float share = p < particles.mass_fraction.size()
                ? std::clamp(particles.mass_fraction[p], 0.0f, 1.0f) : 1.0f;
            const double heat = static_cast<double>(rest) * share * std::fabs(changed) * latent;
            if (changed > 0.0f) released_j += heat; else absorbed_j += heat;
        }
        if (fraction > 0.0f && fraction < 1.0f) ++mushy;

        const bool was_solid = before >= kFusionSolidThreshold;
        const bool is_solid = fraction >= kFusionSolidThreshold;
        if (is_solid && !was_solid) ++froze;
        if (!is_solid && was_solid) ++melted;
        if (skeleton) {
            // Solid and liquid are two constitutive models of this parcel;
            // refreshMatterConstitutiveModels switches them at the next step.
            f &= ~kParticleFlagFrozen;
            continue;
        }
        if (!is_solid) {
            f &= ~kParticleFlagFrozen;
            continue;
        }
        f |= kParticleFlagFrozen;
        ++frozen_total;
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
    stats.mushy_particles = static_cast<std::size_t>(mushy);
    stats.fusion_released_j = released_j;
    stats.fusion_absorbed_j = absorbed_j;
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
