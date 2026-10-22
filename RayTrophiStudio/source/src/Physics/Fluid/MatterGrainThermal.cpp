// Grain carriers in the thermal chain. See include/Fluid/MatterGrainThermal.h.
#include "Fluid/MatterGrainThermal.h"

#include "Fluid/FluidParticles.h"
#include "Fluid/GasSaturation.h"
#include "Fluid/MatterSubstanceState.h"
#include "Fluid/SubstanceTag.h"
#include "FluidGrid.h"
#include "MaterialStateField.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

namespace RayTrophiSim::Fluid {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kSigma = 5.670374e-8;
constexpr double kLiquidFilm = MaterialStateFieldSystem::kMsfContactFilmWm2K;
constexpr double kAirFilm = MaterialStateFieldSystem::kMsfGasFilmWm2K;
// rho_air = P M / (R T) at 1 atm: 101325 * 0.028965 / 8.314462.
constexpr double kAirDensityTimesKelvin = 352.98;
constexpr double kAirSpecificHeat = 1005.0;
// A grain bed on a plate: point contacts with air between, not a wetting film.
constexpr double kGrainWallFilm = 200.0;

// Continuum: the cell's pool the grains wet into - liquid parcels and, since
// 2026-10-20, MPM-owned granular parcels (Snow beside DEM Sand).
enum class Role : uint8_t { None, Continuum, Grain };

Role roleOf(const FluidParticles& p, std::size_t i, const APICSolverParams& params) {
    if (i < p.mass_fraction.size() && !(p.mass_fraction[i] > 0.02f)) return Role::None;
    if (i < p.flags.size() && (p.flags[i] & kParticleFlagFrozen)) return Role::None;
    const auto model = i < p.constitutive_model.size()
        ? static_cast<MatterConstitutiveModel>(p.constitutive_model[i])
        : MatterConstitutiveModel::Auto;
    if (model == MatterConstitutiveModel::Fluid || model == MatterConstitutiveModel::Auto) {
        return Role::Continuum;
    }
    if (model != MatterConstitutiveModel::Granular || !params.grain.enabled) return Role::None;
    const uint32_t tag = i < p.substance_tag.size() ? p.substance_tag[i] : kSubstanceUntagged;
    // ★ An MPM granular parcel used to be None: outside every heat path, so
    // 600 K sand gave 263 K snow nothing and it never melted (2026-10-20,
    // user scene). It joins the cell's pool; its melt then pays latent heat
    // in updateThermalFreeze like any skeleton carrier.
    return substanceTransportOwner(tag, model, params) == MatterTransportOwner::Grain
        ? Role::Grain : Role::Continuum;
}

bool cellOf(const FluidSim::FluidGrid& g, const Vec3& x, int& i, int& j, int& k) {
    if (!std::isfinite(x.x) || !std::isfinite(x.y) || !std::isfinite(x.z)) return false;
    const float inv = 1.0f / g.voxel_size;
    i = static_cast<int>(std::floor((x.x - g.origin.x) * inv));
    j = static_cast<int>(std::floor((x.y - g.origin.y) * inv));
    k = static_cast<int>(std::floor((x.z - g.origin.z) * inv));
    return i >= 0 && i < g.nx && j >= 0 && j < g.ny && k >= 0 && k < g.nz;
}

// Exact two-body relaxation through conductance g (W/K) over dt. Returns the
// heat a -> b (J) and moves both temperatures by exactly that energy.
double relax(double& ta, double ca, double& tb, double cb, double g, double dt) {
    if (!(g > 0.0) || !(ca > 0.0) || !(cb > 0.0)) return 0.0;
    const double decay = 1.0 - std::exp(-g * dt * (1.0 / ca + 1.0 / cb));
    const double eq = (ca * ta + cb * tb) / (ca + cb);
    const double dta = (eq - ta) * decay;
    const double q = -ca * dta;
    ta += dta;
    tb += q / cb;
    return q;
}

// A lumped sphere seen through its own interior: the film and the
// centre-to-surface resistance r / (5 k) in series (Biot correction for a
// sphere). Without it a 1 cm gravel grain in water (Bi ~ 14) settles ~4x too
// fast, because the film alone pretends the grain is all at its surface.
double seriesFilm(double film, double radius, double conductivity) {
    if (!(film > 0.0)) return 0.0;
    if (!(conductivity > 0.0) || !(radius > 0.0)) return film;
    return 1.0 / (1.0 / film + radius / (5.0 * conductivity));
}

struct AirCell {
    double capacity = 0.0;   // J/K, air left beside the grains
    double kelvin = 0.0;     // air T (what the films trade with)
    double kelvin0 = 0.0;
    double tv_per_t = 1.0;   // Tv / T of the cell: a dT moves the field by dT * this
    double grain_volume = 0.0;
    float flame = 0.0f;
    bool ready = false;
    std::size_t cell = 0;    // gas grid cell
};

} // namespace

void exchangeMatterGrainHeat(FluidParticles& particles, const FluidSim::FluidGrid& grid,
                             const APICSolverParams& params, FluidSim::FluidGrid* gas,
                             const MatterGrainGasAir& air, float wall_kelvin,
                             const std::vector<float>* shell_capacity, float dt,
                             ThermalLiquidStats& stats) {
    const auto start = std::chrono::steady_clock::now();
    const std::size_t n = std::min(particles.size(), particles.temperature.size());
    const std::size_t cells = static_cast<std::size_t>(grid.nx) * grid.ny * grid.nz;
    if (!params.grain.enabled || n == 0 || cells == 0 || !(grid.voxel_size > 0.0f) ||
        !(dt > 0.0f)) {
        return;
    }
    const double h = grid.voxel_size;
    const double cell_volume = h * h * h;
    const double radius = std::max(1.0e-5, double(params.grain.radius_m));
    // Films see the grains the simulated one stands for (2026-10-20). A 1 cm
    // simulated grain representing 0.3 mm sand has 33x the surface per kg;
    // with the simulated radius hot sand took minutes to quench in water
    // (tau ~200 s at 2.5 cm). Grain-to-grain contact keeps the simulated
    // radius: it is the simulated contact network.
    const double film_radius = params.grain.represented_grain_radius_m > 0.0f
        ? std::max(1.0e-6, double(params.grain.represented_grain_radius_m)) : radius;

    // Pass 1: roles, cells, the liquid pool of every cell and the grains' room.
    static std::vector<int> s_cell;
    static std::vector<Role> s_role;
    static std::vector<double> s_liquid_cap, s_liquid_t0, s_liquid_t, s_liquid_volume;
    static std::vector<double> s_grain_volume;
    s_cell.assign(n, -1);
    s_role.assign(n, Role::None);
    s_liquid_cap.assign(cells, 0.0);
    s_liquid_t0.assign(cells, 0.0);
    s_liquid_volume.assign(cells, 0.0);
    s_grain_volume.assign(cells, 0.0);
    std::vector<std::size_t> grains;
    uint32_t cached_tag = ~0u;
    const SubstanceProfile* profile = nullptr;
    for (std::size_t p = 0; p < n; ++p) {
        const Role role = roleOf(particles, p, params);
        if (role == Role::None) continue;
        const float t = particles.temperature[p];
        if (!std::isfinite(t) || t <= 0.0f) continue;
        int i, j, k;
        if (!cellOf(grid, particles.position[p], i, j, k)) continue;
        const uint32_t tag = p < particles.substance_tag.size()
            ? particles.substance_tag[p] : kSubstanceUntagged;
        if (tag != cached_tag) {
            profile = particleSubstance(tag, params);
            cached_tag = tag;
        }
        if (!profile) continue;
        const std::size_t c = grid.cellIndex(i, j, k);
        const double fraction = p < particles.mass_fraction.size()
            ? std::clamp(double(particles.mass_fraction[p]), 0.0, 1.0) : 1.0;
        const double mass = p < particles.rest_mass_kg.size()
            ? double(particles.rest_mass_kg[p]) * fraction : 0.0;
        s_cell[p] = static_cast<int>(c);
        s_role[p] = role;
        if (role == Role::Continuum) {
            const double cap = parcelHeatCapacity(particles, p, *profile, grid.voxel_size,
                                                  params.particles_per_cell);
            if (!(cap > 0.0)) { s_role[p] = Role::None; continue; }
            s_liquid_cap[c] += cap;
            s_liquid_t0[c] += cap * t;
            // Volume in the parcel's own state: snow at its bulk density, not
            // at the density of the water it melts into.
            const bool solid_state = p < particles.constitutive_model.size() &&
                particles.constitutive_model[p] ==
                    static_cast<uint8_t>(MatterConstitutiveModel::Granular);
            const double rho = !solid_state && profile->liquid_density > 0.0f
                ? profile->liquid_density
                : (profile->density > 0.0f ? profile->density : 1000.0f);
            s_liquid_volume[c] += mass / rho;
        } else {
            const double rho = profile->density > 0.0f ? profile->density : 1000.0f;
            s_grain_volume[c] += mass / rho;
            grains.push_back(p);
        }
    }
    stats.grain_count = grains.size();
    if (grains.empty()) return;
    for (std::size_t c = 0; c < cells; ++c) {
        if (s_liquid_cap[c] > 0.0) s_liquid_t0[c] /= s_liquid_cap[c];
    }
    s_liquid_t = s_liquid_t0;

    // The domain's air. Its temperature field holds Tv in normalized units.
    const bool air_live = gas && air.kelvin_per_unit > 0.0f && gas->voxel_size > 0.0f &&
        gas->temperature.size() ==
            static_cast<std::size_t>(gas->nx) * gas->ny * gas->nz;
    const double gas_cell_volume = air_live
        ? double(gas->voxel_size) * gas->voxel_size * gas->voxel_size : 0.0;
    std::unordered_map<std::size_t, AirCell> air_cells;
    std::vector<int64_t> grain_air(grains.size(), -1);
    if (air_live) {
        air_cells.reserve(grains.size() / 4 + 16);
        for (std::size_t g = 0; g < grains.size(); ++g) {
            const std::size_t p = grains[g];
            int i, j, k;
            if (!cellOf(*gas, particles.position[p], i, j, k)) continue;
            const std::size_t c = gas->cellIndex(i, j, k);
            if (c < gas->solid.size() && gas->solid[c] != 0u) continue;
            const uint32_t tag = p < particles.substance_tag.size()
                ? particles.substance_tag[p] : kSubstanceUntagged;
            const SubstanceProfile* sp = particleSubstance(tag, params);
            const double rho = sp && sp->density > 0.0f ? sp->density : 1000.0;
            const double fraction = p < particles.mass_fraction.size()
                ? std::clamp(double(particles.mass_fraction[p]), 0.0, 1.0) : 1.0;
            air_cells[c].grain_volume += double(particles.rest_mass_kg[p]) * fraction / rho;
            grain_air[g] = static_cast<int64_t>(c);
        }
    }

    // Pass 2 (P4, 2026-10-19: parallel). Each grain's constants first; then
    // phase A over LIQUID cells (liquid pool, walls, grain contact - a cell's
    // pool and grains belong to that cell only) and phase B over AIR cells
    // (the gas grid has its own cells, shared by grains of several liquid
    // cells). Both phases run their groups in parallel and each group in a
    // stable order, so the result does not depend on the thread count.
    const std::size_t ng = grains.size();
    std::vector<double> tg(ng), cg(ng, 0.0), kg(ng, 0.0), ag(ng, 0.0), wg(ng, 0.0), eg(ng, 0.0);
    const bool closed_walls = params.boundary == APICSolverParams::BoundaryMode::Closed;
    const bool has_solid = grid.solid.size() == cells;
    const bool have_shells = shell_capacity && shell_capacity->size() == cells;
    // Faces of a cell that are a container at wall_kelvin.
    const auto wallFaces = [&](std::size_t c) {
        const int i = static_cast<int>(c % grid.nx);
        const int j = static_cast<int>((c / grid.nx) % grid.ny);
        const int k = static_cast<int>(c / (static_cast<std::size_t>(grid.nx) * grid.ny));
        constexpr int offsets[6][3] = {{-1, 0, 0}, {1, 0, 0}, {0, -1, 0},
                                       {0, 1, 0}, {0, 0, -1}, {0, 0, 1}};
        int faces = 0;
        for (const auto& o : offsets) {
            const int ni = i + o[0], nj = j + o[1], nk = k + o[2];
            if (ni < 0 || ni >= grid.nx || nj < 0 || nj >= grid.ny || nk < 0 || nk >= grid.nz) {
                if (closed_walls) ++faces;
                continue;
            }
            const std::size_t nc = grid.cellIndex(ni, nj, nk);
            if (has_solid && grid.solid[nc] == FluidSim::FluidGrid::kSolidCollider &&
                !(have_shells && (*shell_capacity)[nc] > 0.0f)) {
                ++faces;
            }
        }
        return faces;
    };
    const int64_t grain_count = static_cast<int64_t>(ng);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(grain_count > 4096)
#endif
    for (int64_t gi = 0; gi < grain_count; ++gi) {
        const std::size_t g = static_cast<std::size_t>(gi);
        const std::size_t p = grains[g];
        tg[g] = particles.temperature[p];
        const uint32_t tag = p < particles.substance_tag.size()
            ? particles.substance_tag[p] : kSubstanceUntagged;
        const SubstanceProfile* sp = particleSubstance(tag, params);
        if (!sp) continue;
        const double cap = parcelHeatCapacity(particles, p, *sp, grid.voxel_size,
                                              params.particles_per_cell);
        if (!(cap > 0.0)) continue;
        const double rho = sp->density > 0.0f ? sp->density : 1000.0;
        const double fraction = p < particles.mass_fraction.size()
            ? std::clamp(double(particles.mass_fraction[p]), 0.0, 1.0) : 1.0;
        cg[g] = cap;
        kg[g] = std::max(0.0f, sp->conductivity);
        ag[g] = 3.0 * double(particles.rest_mass_kg[p]) * fraction / (rho * film_radius);
        eg[g] = std::clamp(double(sp->emissivity), 0.0, 1.0);
        const std::size_t c = static_cast<std::size_t>(s_cell[p]);
        const double room = std::max(cell_volume - s_grain_volume[c], 0.1 * cell_volume);
        wg[g] = std::clamp(s_liquid_volume[c] / room, 0.0, 1.0);
    }

    // Groups of `order` sharing key(g): [starts[k], starts[k+1]).
    const auto groupsOf = [&](std::vector<std::size_t>& order, auto key) {
        order.resize(ng);
        for (std::size_t g = 0; g < ng; ++g) order[g] = g;
        std::stable_sort(order.begin(), order.end(),
                         [&](std::size_t x, std::size_t y) { return key(x) < key(y); });
        std::vector<std::size_t> starts;
        for (std::size_t o = 0; o < ng; ++o) {
            if (o == 0 || key(order[o]) != key(order[o - 1])) starts.push_back(o);
        }
        starts.push_back(ng);
        return starts;
    };

    // Phase A: liquid pool, walls, grain <-> grain, per liquid cell.
    std::vector<std::size_t> order;
    const auto cell_starts = groupsOf(order, [&](std::size_t g) { return s_cell[grains[g]]; });
    // Each cell's grain temperatures before the exchange: the contact deposit
    // below may not carry a parcel past them (2026-10-21).
    static std::vector<double> s_grain_hi, s_grain_lo;
    s_grain_hi.assign(cells, -std::numeric_limits<double>::max());
    s_grain_lo.assign(cells, std::numeric_limits<double>::max());
    double to_liquid = 0.0, to_walls = 0.0;
    int64_t wetted = 0;
    const int64_t cell_groups = static_cast<int64_t>(cell_starts.size()) - 1;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 64) reduction(+:to_liquid, to_walls, wetted) if(cell_groups > 256)
#endif
    for (int64_t gr = 0; gr < cell_groups; ++gr) {
        const std::size_t begin = cell_starts[static_cast<std::size_t>(gr)];
        const std::size_t end = cell_starts[static_cast<std::size_t>(gr) + 1];
        const std::size_t c = static_cast<std::size_t>(s_cell[grains[order[begin]]]);
        const int walls = std::isfinite(wall_kelvin) && wall_kelvin > 0.0f ? wallFaces(c) : 0;
        double cell_cap = 0.0, cell_energy = 0.0;
        for (std::size_t o = begin; o < end; ++o) {
            const std::size_t g = order[o];
            const double cap = cg[g];
            if (!(cap > 0.0)) continue;
            s_grain_hi[c] = std::max(s_grain_hi[c], tg[g]);
            s_grain_lo[c] = std::min(s_grain_lo[c], tg[g]);
            // Wetted share: against the cell's liquid as one pool.
            if (wg[g] > 0.0 && s_liquid_cap[c] > 0.0) {
                const double film = seriesFilm(kLiquidFilm, film_radius, kg[g]);
                to_liquid += relax(tg[g], cap, s_liquid_t[c], s_liquid_cap[c],
                                   film * ag[g] * wg[g], dt);
                ++wetted;
            }
            // Walls: an infinite container at wall_kelvin, a sixth of the
            // grain's surface per touching face.
            if (walls > 0) {
                const double film = seriesFilm(kGrainWallFilm, film_radius, kg[g]);
                const double conductance = film * ag[g] * (walls / 6.0);
                const double before = tg[g];
                tg[g] = wall_kelvin + (tg[g] - wall_kelvin) * std::exp(-conductance * dt / cap);
                to_walls += cap * (before - tg[g]);
            }
            cell_cap += cap;
            cell_energy += cap * tg[g];
        }
        // Grain <-> grain inside the cell: contact conduction toward the
        // cell's grain mean, G = k 4 pi r. Rates differ with capacity, so the
        // energy the relaxation leaves over is spread back by capacity: the
        // pass conserves exactly.
        if (end - begin > 1 && cell_cap > 0.0) {
            const double mean = cell_energy / cell_cap;
            double drift = 0.0;
            for (std::size_t o = begin; o < end; ++o) {
                const std::size_t g = order[o];
                if (!(cg[g] > 0.0)) continue;
                const double conductance = kg[g] * 4.0 * kPi * radius;
                const double d = (mean - tg[g]) * (1.0 - std::exp(-conductance * dt / cg[g]));
                tg[g] += d;
                drift += cg[g] * d;
            }
            for (std::size_t o = begin; o < end; ++o) {
                const std::size_t g = order[o];
                if (cg[g] > 0.0) tg[g] -= drift / cell_cap;
            }
        }
    }

    // Phase B: the dry share against the cell's air, per air cell. The air
    // cells are resolved serially first (their own T, Tv/T and capacity).
    std::vector<AirCell> air_list;
    std::vector<int64_t> grain_air_slot(ng, -1);
    if (air_live) {
        std::unordered_map<std::size_t, std::size_t> slot_of;
        slot_of.reserve(air_cells.size());
        for (auto& [ac, a] : air_cells) {
            double tv = air.ambient_kelvin +
                (double(gas->temperature[ac]) - air.gas_ambient) * air.kelvin_per_unit;
            tv = std::isfinite(tv) ? std::max(tv, 1.0) : air.ambient_kelvin;
            // The field holds Tv; the film trades with the air's T.
            double t_air = tv;
            if (air.vapour_kg && ac < air.vapour_cells && air.vapour_molar > 0.0f) {
                t_air = airKelvinFromVirtual(static_cast<float>(tv), air.vapour_kg[ac],
                    air.cell_volume, air.vapour_molar, 1.0f);
            }
            a.kelvin = a.kelvin0 = std::max(t_air, 1.0);
            a.tv_per_t = tv / a.kelvin;
            const double volume = std::max(gas_cell_volume - a.grain_volume,
                                           0.1 * gas_cell_volume);
            // rho = P Ma / (R Tv), exactly, vapour or not.
            a.capacity = kAirDensityTimesKelvin / tv * kAirSpecificHeat * volume;
            a.flame = ac < gas->interaction.size()
                ? std::clamp(gas->interaction[ac], 0.0f, 1.0f) : 0.0f;
            a.ready = true;
            a.cell = ac;
            slot_of[ac] = air_list.size();
            air_list.push_back(a);
        }
        for (std::size_t g = 0; g < ng; ++g) {
            if (grain_air[g] >= 0 && wg[g] < 1.0 && cg[g] > 0.0) {
                grain_air_slot[g] = static_cast<int64_t>(
                    slot_of[static_cast<std::size_t>(grain_air[g])]);
            }
        }
    }
    double to_gas = 0.0;
    if (!air_list.empty()) {
        std::vector<std::size_t> air_order;
        const auto air_starts = groupsOf(air_order, [&](std::size_t g) { return grain_air_slot[g]; });
        const int64_t air_groups = static_cast<int64_t>(air_starts.size()) - 1;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 64) reduction(+:to_gas) if(air_groups > 256)
#endif
        for (int64_t gr = 0; gr < air_groups; ++gr) {
            const std::size_t begin = air_starts[static_cast<std::size_t>(gr)];
            const std::size_t end = air_starts[static_cast<std::size_t>(gr) + 1];
            const int64_t slot = grain_air_slot[air_order[begin]];
            if (slot < 0) continue;  // the group of grains with no air share
            AirCell& a = air_list[static_cast<std::size_t>(slot)];
            for (std::size_t o = begin; o < end; ++o) {
                const std::size_t g = air_order[o];
                const double ts = tg[g], ta = a.kelvin;
                const double radiation = a.flame * eg[g] * kSigma *
                    (ts * ts + ta * ta) * (ts + ta);
                const double film = seriesFilm(kAirFilm + radiation, film_radius, kg[g]);
                to_gas += relax(tg[g], cg[g], a.kelvin, a.capacity,
                                film * ag[g] * (1.0 - wg[g]), dt);
            }
        }
    }

    // Write back: grains own their temperature; every liquid parcel of a cell
    // moves by its pool's change (sum C dT = C_pool dT_pool, exactly).
    float lo = std::numeric_limits<float>::max(), hi = -std::numeric_limits<float>::max();
    double sum = 0.0, residual = 0.0;
    std::size_t counted = 0;
    for (std::size_t g = 0; g < grains.size(); ++g) {
        const std::size_t p = grains[g];
        residual += cg[g] * (tg[g] - double(particles.temperature[p]));
        particles.temperature[p] = static_cast<float>(tg[g]);
        if (!(cg[g] > 0.0)) continue;
        const float t = particles.temperature[p];
        lo = std::min(lo, t);
        hi = std::max(hi, t);
        sum += t;
        ++counted;
    }
    // ★ The pool's energy change lands where the grains touch it (2026-10-20).
    // Spread evenly, 600 K sand only warmed its cell's water by a fraction of a
    // kelvin: nothing boiled on contact and snow under hot sand melted no
    // faster at the contact than across the cell. Each continuum parcel takes
    // a share by its proximity to the cell's grains (sum over them of
    // 1 / (d^2 + (h/4)^2)), so the parcels at the contact reach the boiling or
    // melting point first and the existing boiling / fusion passes take it
    // from there. sum C dT is still exactly the pool's change.
    // ★ Capped at the cell's grains (2026-10-21): a parcel close to the grains
    // took so large a share that it ran past them (629.8 K beside 600 K sand,
    // the ledger closed so no total showed it). Heat cannot carry a parcel
    // past its source, so a parcel stops at the hottest grain's starting
    // temperature (coldest, when the grains cool the pool) and its unused
    // share goes to the others by their weights. The pool mean never passes
    // the grains (relax ends between them), so the room always suffices; a
    // remainder only appears for parcels that already started past the
    // bound, and it is spread evenly as before.
    static std::vector<int64_t> s_grain_begin;
    static std::vector<double> s_parcel_weight, s_parcel_cap;
    s_grain_begin.assign(cells, -1);
    s_parcel_weight.assign(n, 0.0);
    s_parcel_cap.assign(n, 0.0);
    for (std::size_t o = 0; o < order.size(); ++o) {
        const int cell = s_cell[grains[order[o]]];
        if (cell >= 0 && (o == 0 || s_cell[grains[order[o - 1]]] != cell)) {
            s_grain_begin[static_cast<std::size_t>(cell)] = static_cast<int64_t>(o);
        }
    }
    const double soft = 0.0625 * h * h;
    std::vector<std::size_t> touched;  // parcels of cells whose pool changed, by cell
    for (std::size_t p = 0; p < n; ++p) {
        if (s_role[p] != Role::Continuum || s_cell[p] < 0) continue;
        const std::size_t c = static_cast<std::size_t>(s_cell[p]);
        if (s_liquid_t[c] == s_liquid_t0[c]) continue;
        touched.push_back(p);
        if (s_grain_begin[c] < 0) continue;
        double w = 0.0;
        for (std::size_t o = static_cast<std::size_t>(s_grain_begin[c]);
             o < order.size() && s_cell[grains[order[o]]] == static_cast<int>(c); ++o) {
            const Vec3 d = particles.position[p] - particles.position[grains[order[o]]];
            w += 1.0 / (double(d.x) * d.x + double(d.y) * d.y + double(d.z) * d.z + soft);
        }
        const uint32_t tag = p < particles.substance_tag.size()
            ? particles.substance_tag[p] : kSubstanceUntagged;
        const SubstanceProfile* sp = particleSubstance(tag, params);
        s_parcel_weight[p] = w;
        s_parcel_cap[p] = sp ? parcelHeatCapacity(particles, p, *sp, grid.voxel_size,
                                                  params.particles_per_cell) : 0.0;
    }
    std::stable_sort(touched.begin(), touched.end(),
                     [&](std::size_t x, std::size_t y) { return s_cell[x] < s_cell[y]; });
    std::vector<std::size_t> open;
    for (std::size_t b = 0; b < touched.size();) {
        const std::size_t c = static_cast<std::size_t>(s_cell[touched[b]]);
        std::size_t e = b;
        while (e < touched.size() && s_cell[touched[e]] == static_cast<int>(c)) ++e;
        const double d = s_liquid_t[c] - s_liquid_t0[c];
        // C_pool dT_pool = dE, handed out by weight: sum_p C_p dT_p = dE.
        double remaining = s_liquid_cap[c] * d;
        const double bound = d > 0.0 ? s_grain_hi[c] : s_grain_lo[c];
        const bool capped = std::isfinite(bound) && std::abs(bound) <
            std::numeric_limits<double>::max();
        open.clear();
        for (std::size_t q = b; q < e; ++q) {
            const std::size_t p = touched[q];
            if (s_parcel_weight[p] > 0.0 && s_parcel_cap[p] > 0.0) open.push_back(p);
        }
        // Water filling: clamp every parcel whose share passes the bound, hand
        // the rest out again; each round closes at least one parcel.
        while (!open.empty() && remaining != 0.0) {
            double weight = 0.0;
            for (const std::size_t p : open) weight += s_parcel_weight[p] * s_parcel_cap[p];
            if (!(weight > 0.0)) break;
            std::size_t kept = 0;
            for (const std::size_t p : open) {
                const double share = remaining * s_parcel_weight[p] * s_parcel_cap[p] / weight;
                const double room = capped
                    ? std::max(0.0, (bound - double(particles.temperature[p])) *
                                    (d > 0.0 ? 1.0 : -1.0)) * s_parcel_cap[p]
                    : std::numeric_limits<double>::max();
                if (std::abs(share) > room) {
                    const double given = d > 0.0 ? room : -room;
                    particles.temperature[p] += static_cast<float>(given / s_parcel_cap[p]);
                    remaining -= given;
                } else {
                    open[kept++] = p;
                }
            }
            if (kept == open.size()) {
                for (const std::size_t p : open) {
                    particles.temperature[p] += static_cast<float>(
                        remaining * s_parcel_weight[p] / weight);
                }
                remaining = 0.0;
                break;
            }
            open.resize(kept);
        }
        if (remaining != 0.0) {
            double capacity = 0.0;
            for (std::size_t q = b; q < e; ++q) capacity += s_parcel_cap[touched[q]];
            for (std::size_t q = b; q < e; ++q) {
                const std::size_t p = touched[q];
                particles.temperature[p] += static_cast<float>(
                    capacity > 0.0 ? remaining / capacity : d);
            }
        }
        b = e;
    }
    for (std::size_t c = 0; c < cells; ++c) {
        if (s_liquid_cap[c] > 0.0) residual += s_liquid_cap[c] * (s_liquid_t[c] - s_liquid_t0[c]);
    }
    // The walls are a reservoir outside the pass: what they took is booked.
    residual += to_walls;
    for (const AirCell& a : air_list) {
        const double d = a.kelvin - a.kelvin0;
        residual += a.capacity * d;
        if (d != 0.0) {
            gas->temperature[a.cell] += static_cast<float>(d * a.tv_per_t / air.kelvin_per_unit);
        }
    }
    stats.grain_measured = true;
    stats.grain_air_humid = air_live && air.vapour_kg && air.vapour_molar > 0.0f;
    stats.grain_wetted = static_cast<std::size_t>(wetted);
    stats.grain_heat_to_liquid_j = to_liquid;
    stats.grain_heat_to_gas_j = to_gas;
    stats.grain_heat_to_walls_j = to_walls;
    stats.grain_energy_residual_j = residual;
    if (counted > 0) {
        stats.grain_min_kelvin = lo;
        stats.grain_max_kelvin = hi;
        stats.grain_mean_kelvin = static_cast<float>(sum / double(counted));
    }
    stats.grain_thermal_ms = std::chrono::duration<float, std::milli>(
        std::chrono::steady_clock::now() - start).count();
}

} // namespace RayTrophiSim::Fluid
