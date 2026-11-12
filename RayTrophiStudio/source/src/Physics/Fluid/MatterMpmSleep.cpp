#include "Fluid/MatterMpmSleep.h"

#include "Fluid/APICFluidSolver.h"
#include "Fluid/GranularConstitutive.h"
#include "Fluid/MatterConstitutive.h"
#include "FluidGrid.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace RayTrophiSim::Fluid {

namespace {

uint64_t mix(uint64_t h, uint64_t v) {
    return h ^ (v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
}

uint64_t mixFloat(uint64_t h, float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(u));
    return mix(h, u);
}

} // namespace

void computeMpmSleep(const FluidParticles& particles, const FluidSim::FluidGrid& grid,
                     const APICSolverParams& params, bool legacy_granular,
                     MpmSleepState& state, MpmSleepResult& out) {
    out = MpmSleepResult{};
    const std::size_t n = particles.size();
    if (!params.grain.sleep || n == 0 || particles.granular_material_flags.size() != n) {
        state.context_valid = false;
        state.rest_frames = 0;
        return;
    }

    // ★ Colliders count only NEAR the MPM parcels (2026-11-12). A moving
    // collider anywhere in the domain used to keep resting snow awake (any
    // nonzero solid_vel), and its voxel signature, which changes every frame
    // it moves, reset the rest count too: a sphere still 2 m above a resting
    // pack ran the full 44-substep frame (150 ms; 23 ms without the snow).
    // The region is the parcels' cell box grown by 3 cells plus twice the
    // collider footprint's motion since last frame, so a collider about to
    // arrive wakes the domain the frame before it gets there.
    const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    int plo[3] = {nx, ny, nz}, phi[3] = {-1, -1, -1};
    const float inv_h = grid.voxel_size > 0.0f ? 1.0f / grid.voxel_size : 0.0f;
    for (std::size_t p = 0; p < n; ++p) {
        const Vec3 g = (particles.position[p] - grid.origin) * inv_h;
        const int c[3] = {static_cast<int>(std::floor(g.x)), static_cast<int>(std::floor(g.y)),
                          static_cast<int>(std::floor(g.z))};
        for (int a = 0; a < 3; ++a) { plo[a] = std::min(plo[a], c[a]); phi[a] = std::max(phi[a], c[a]); }
    }
    int sweep = 0;
    for (int a = 0; a < 3; ++a) {
        if (grid.collider_prev_hi[a] >= grid.collider_prev_lo[a] &&
            grid.collider_cur_hi[a] >= grid.collider_cur_lo[a]) {
            sweep = std::max(sweep, std::abs(grid.collider_cur_lo[a] - grid.collider_prev_lo[a]));
            sweep = std::max(sweep, std::abs(grid.collider_cur_hi[a] - grid.collider_prev_hi[a]));
        }
    }
    const int margin = 3 + 2 * sweep;
    int rlo[3], rhi[3];
    const int dims[3] = {nx, ny, nz};
    for (int a = 0; a < 3; ++a) {
        rlo[a] = std::clamp(plo[a] - margin, 0, dims[a] - 1);
        rhi[a] = std::clamp(phi[a] + margin, 0, dims[a] - 1);
    }
    // Moving solid cells and the solid layout, inside the region only.
    bool moving_collider = false;
    uint64_t local_solid = 1469598103934665603ULL;
    const bool have_solid = grid.solid.size() == static_cast<std::size_t>(nx) * ny * nz;
    const bool have_vel = grid.solid_vel.size() == grid.solid.size();
    if (have_solid && phi[0] >= plo[0]) {
        for (int k = rlo[2]; k <= rhi[2]; ++k)
        for (int j = rlo[1]; j <= rhi[1]; ++j)
        for (int i = rlo[0]; i <= rhi[0]; ++i) {
            const std::size_t c = grid.cellIndex(i, j, k);
            if (!grid.solid[c]) continue;
            local_solid = mix(local_solid, c);
            if (have_vel && grid.solid_vel[c].length_squared() > 1e-12f) moving_collider = true;
        }
    }
    // A collider whose footprint (now or one step ahead) reaches the region
    // wakes it even before it stamps solid cells there.
    if (!moving_collider && grid.collider_cur_hi[0] >= grid.collider_cur_lo[0] && sweep > 0) {
        bool overlap = true;
        for (int a = 0; a < 3; ++a) {
            const int next_lo = grid.collider_cur_lo[a] + (grid.collider_cur_lo[a] - grid.collider_prev_lo[a]);
            const int next_hi = grid.collider_cur_hi[a] + (grid.collider_cur_hi[a] - grid.collider_prev_hi[a]);
            const int lo = std::min(grid.collider_cur_lo[a], next_lo);
            const int hi = std::max(grid.collider_cur_hi[a], next_hi);
            overlap = overlap && hi >= rlo[a] && lo <= rhi[a];
        }
        moving_collider = overlap;
    }
    uint64_t context = 1469598103934665603ULL;
    context = mixFloat(context, params.gravity.x);
    context = mixFloat(context, params.gravity.y);
    context = mixFloat(context, params.gravity.z);
    context = mix(context, static_cast<uint64_t>(grid.nx));
    context = mix(context, static_cast<uint64_t>(grid.ny));
    context = mix(context, static_cast<uint64_t>(grid.nz));
    context = mixFloat(context, grid.voxel_size);
    context = mixFloat(context, grid.origin.x);
    context = mixFloat(context, grid.origin.y);
    context = mixFloat(context, grid.origin.z);
    context = mix(context, local_solid);   // solids near the parcels, not the whole set
    context = mix(context, static_cast<uint64_t>(n));
    const bool same_context = state.context_valid && state.context == context;
    state.context = context;
    state.context_valid = true;
    if (!same_context || moving_collider) {
        state.rest_frames = 0;
        return;
    }

    const float tolerance = 0.25f * params.gravity.length();
    bool all_rest = true;
    for (std::size_t p = 0; p < n; ++p) {
        const auto model = p < particles.constitutive_model.size()
            ? static_cast<MatterConstitutiveModel>(particles.constitutive_model[p])
            : MatterConstitutiveModel::Auto;
        // Elastic bodies run no settle, liquid is never at rest: both keep
        // the domain awake.
        const bool mpm = model == MatterConstitutiveModel::Granular ||
            (model == MatterConstitutiveModel::Auto && legacy_granular);
        if (!mpm) { all_rest = false; continue; }
        const float speed2 = particles.velocity[p].length_squared();
        const bool slept =
            (particles.granular_material_flags[p] & Granular::kFlagSleeping) != 0u;
        const bool supported = p < particles.granular_support_accel.size() &&
            particles.granular_support_accel[p] < tolerance;
        const bool rests = speed2 < kMpmWakeSpeed * kMpmWakeSpeed &&
            (slept || (supported && speed2 < kMpmRestSpeed * kMpmRestSpeed));
        if (rests) ++out.resting_particles;
        else all_rest = false;
    }
    state.rest_frames = all_rest ? state.rest_frames + 1 : 0;
    out.domain_asleep = state.rest_frames >= kMpmSleepFrames;
}

} // namespace RayTrophiSim::Fluid
