#include "Fluid/FluidSurfaceWater.h"

#include "Fluid/APICFluidSolver.h"
#include "Fluid/FluidParticles.h"
#include "Fluid/MatterSubstanceState.h"
#include "Fluid/SubstanceTag.h"
#include "FluidGrid.h"
#include "MaterialStateField.h"

#include <algorithm>
#include <cmath>

namespace RayTrophiSim::Fluid {
namespace {

constexpr int kFaceOffsets[6][3] = {
    {-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}
};
// Below this fraction a parcel is dead until compaction (liveParcel in
// FluidThermalLiquid.cpp uses 0.02); a debit never leaves one in between.
constexpr float kKeepFraction = 0.03f;

bool cellOf(const FluidSim::FluidGrid& grid, const Vec3& x, int& i, int& j, int& k) {
    if (!std::isfinite(x.x) || !std::isfinite(x.y) || !std::isfinite(x.z)) return false;
    const Vec3 g = (x - grid.origin) / grid.voxel_size;
    i = static_cast<int>(std::floor(g.x));
    j = static_cast<int>(std::floor(g.y));
    k = static_cast<int>(std::floor(g.z));
    return i >= 0 && i < grid.nx && j >= 0 && j < grid.ny && k >= 0 && k < grid.nz;
}

} // namespace

SurfaceWaterExchangeStats exchangeLiquidSurfaceWater(
    FluidParticles& particles, const FluidSim::FluidGrid& grid,
    const APICSolverParams& params, float dt,
    const std::vector<float>& cell_deficit_kg, const std::vector<float>& cell_rate_kg_s,
    const std::vector<float>& cell_kelvin,
    std::vector<float>& cell_absorbed_kg, std::vector<float>& cell_absorbed_heat_j) {
    SurfaceWaterExchangeStats stats;
    const std::size_t cells = static_cast<std::size_t>(grid.nx) * grid.ny * grid.nz;
    cell_absorbed_kg.assign(cells, 0.0f);
    cell_absorbed_heat_j.assign(cells, 0.0f);
    const std::size_t n = particles.size();
    if (n == 0 || cells == 0 || !(dt > 0.0f) || !(grid.voxel_size > 0.0f) ||
        cell_deficit_kg.size() != cells || cell_rate_kg_s.size() != cells ||
        cell_kelvin.size() != cells || particles.rest_mass_kg.size() < n ||
        particles.mass_fraction.size() < n) {
        return stats;
    }

    // Pass 1: each water parcel's surface cell and what it can give.
    std::vector<int64_t> parcel_cell(n, -1);
    std::vector<double> offered(cells, 0.0);
    uint32_t cached_tag = ~0u;
    bool cached_water = false;
    for (std::size_t p = 0; p < n; ++p) {
        const float fraction = particles.mass_fraction[p];
        if (!(fraction > 0.02f)) continue;
        const uint32_t tag = p < particles.substance_tag.size()
            ? particles.substance_tag[p] : kSubstanceUntagged;
        if (tag != cached_tag) {
            const SubstanceProfile* profile = particleSubstance(tag, params);
            cached_water = profile && profile->name == "Water";
            cached_tag = tag;
        }
        if (!cached_water) continue;
        int i, j, k;
        if (!cellOf(grid, particles.position[p], i, j, k)) continue;
        std::size_t surface = grid.cellIndex(i, j, k);
        if (!(cell_deficit_kg[surface] > 0.0f)) {
            surface = cells;
            for (const auto& o : kFaceOffsets) {
                const int ni = i + o[0], nj = j + o[1], nk = k + o[2];
                if (ni < 0 || ni >= grid.nx || nj < 0 || nj >= grid.ny || nk < 0 ||
                    nk >= grid.nz) {
                    continue;
                }
                const std::size_t c = grid.cellIndex(ni, nj, nk);
                if (cell_deficit_kg[c] > 0.0f) { surface = c; break; }
            }
        }
        if (surface == cells) continue;
        parcel_cell[p] = static_cast<int64_t>(surface);
        offered[surface] += 0.5 * double(particles.rest_mass_kg[p]) * fraction;
    }

    // Pass 2: each cell takes min(deficit, rate dt, what its parcels offer),
    // split among them by their offer.
    std::vector<double> take_share(cells, 0.0);
    for (std::size_t c = 0; c < cells; ++c) {
        if (!(offered[c] > 0.0)) continue;
        const double want = std::min(double(cell_deficit_kg[c]),
                                     double(cell_rate_kg_s[c]) * dt);
        take_share[c] = std::clamp(want / offered[c], 0.0, 1.0);
    }
    for (std::size_t p = 0; p < n; ++p) {
        if (parcel_cell[p] < 0) continue;
        const std::size_t c = static_cast<std::size_t>(parcel_cell[p]);
        if (!(take_share[c] > 0.0)) continue;
        const double rest = particles.rest_mass_kg[p];
        const double mass = rest * particles.mass_fraction[p];
        double dm = 0.5 * mass * take_share[c];
        // Never strand a parcel between "live" and "dead".
        const double keep = rest * kKeepFraction;
        if (mass - dm < keep) dm = std::max(0.0, mass - keep);
        if (!(dm > 0.0) || !(rest > 0.0)) continue;
        particles.mass_fraction[p] = static_cast<float>((mass - dm) / rest);
        const double kelvin = p < particles.temperature.size() &&
            std::isfinite(particles.temperature[p]) ? particles.temperature[p] : 293.15;
        const SubstanceProfile* profile = particleSubstance(
            p < particles.substance_tag.size() ? particles.substance_tag[p] : kSubstanceUntagged,
            params);
        const double c_w = profile && profile->specific_heat > 0.0f
            ? profile->specific_heat : 4186.0;
        cell_absorbed_kg[c] += static_cast<float>(dm);
        const double heat = c_w * dm * (kelvin - double(cell_kelvin[c]));
        cell_absorbed_heat_j[c] += static_cast<float>(heat);
        stats.absorbed_kg += dm;
        stats.energy_j += c_w * dm * kelvin;
        stats.heat_to_surface_j += heat;
        ++stats.parcels;
    }
    return stats;
}

} // namespace RayTrophiSim::Fluid
