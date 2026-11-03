#pragma once

// Domain sleep for MPM (docs/dev/MPM_UYUYAN_KAROLAR.md, stage B).
//
// When every parcel of a Matter domain has RESTED for kMpmSleepFrames frames,
// the step runs no substep for the frame (gpu_status "matter_asleep"); a
// resting MPM domain then costs ~nothing. Any parcel that does not rest keeps
// the whole domain awake and exact.
//
// A parcel RESTS when it is slower than kMpmWakeSpeed and either the settle put
// it to sleep or it is slower than kMpmRestSpeed and supported (stage A: mean
// acceleration along gravity under 0.25 g). Settle only sleeps parcels under
// pressure, so a pile's surface never slept; and ~1 % of a resting pile shows
// a rest-jitter moment above the sleep speed at any time.
//
// Nothing rests (and the frame counter restarts) on a context change: gravity,
// domain grid, collider set, particle count; nor while a collider moves, with
// a force field (MatterGpuStep), with liquid or elastic parcels, or with the
// domain's sleep switch off (grain settings `sleep`, shared with DEM sleeping
// grains).
//
// Partial freezing was built, measured and removed (2026-11-04): frozen cells
// left both lanes and stood as still walls. Right, it kept the result within
// 3 mm on a 2 m pile but made the frame 5-7 % SLOWER: a resting MPM frame is
// bound by its serial substep chain (~50 substeps x ~36 dispatches, each
// behind a full barrier), not by parcel work - 384 parcels and 100k parcels
// cost the same ~100 ms of GPU, and removing work only let the GPU clock down.
// Its lessons, should the chain ever shorten enough to make parcel work count:
//  * a frozen cell supports like a wall but carries no load, so awake material
//    must never be under or beside one (the first rule unloaded snow under a
//    frozen top; when the top woke the plastic snow crushed, the pile ended
//    0.39 m lower): calm 26-neighbourhood, and support swept up from the floor
//    through frozen cells only;
//  * a parcel the force stage still integrates wakes at g * frame_dt next
//    frame unless its velocity is taken back.

#include "FluidParticles.h"
#include <cstdint>

namespace FluidSim { class FluidGrid; }

namespace RayTrophiSim::Fluid {

struct APICSolverParams;

// Faster than this, a parcel is moving (m/s): above the settle sleep speed
// and the snow rest jitter, well below a body that is going somewhere.
constexpr float kMpmWakeSpeed = 0.1f;
// A supported parcel slower than this rests even unslept (the settle sleep speed).
constexpr float kMpmRestSpeed = 0.03f;
// Frames every parcel must rest before the domain sleeps.
constexpr int kMpmSleepFrames = 3;

// Carried across frames by the domain's Matter GPU runtime.
struct MpmSleepState {
    uint64_t context = 0;
    bool context_valid = false;
    int rest_frames = 0;
};

struct MpmSleepResult {
    std::size_t resting_particles = 0;
    bool domain_asleep = false;
};

void computeMpmSleep(const FluidParticles& particles, const FluidSim::FluidGrid& grid,
                     const APICSolverParams& params, bool legacy_granular,
                     MpmSleepState& state, MpmSleepResult& out);

} // namespace RayTrophiSim::Fluid
