#include "Fluid/MatterMpmSleep.h"

#include "Fluid/APICFluidSolver.h"
#include "Fluid/GranularConstitutive.h"
#include "Fluid/MatterConstitutive.h"
#include "FluidGrid.h"

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

    // Context: anything that changes what "at rest" means wakes everything.
    bool moving_collider = false;
    for (const Vec3& v : grid.solid_vel) {
        if (v.length_squared() > 1e-12f) { moving_collider = true; break; }
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
    context = mix(context, grid.collider_voxel_sig);
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
