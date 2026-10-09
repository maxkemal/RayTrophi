#pragma once

#include <cstdint>

namespace RayTrophiSim::Fluid {

// One substep: rebuild the Verlet neighbour lists if the previous substep
// raised this substep's flag (list_clear, hash and list_build return at once
// otherwise), then the fused contact/integrate step, which raises the next
// substep's flag once a grain has moved half the skin. The host forces a build
// on substep 0 of every frame (docs/dev/DEM_VERLET_LISTESI.md).
template <class Dispatch>
bool dispatchMatterGrainSubstep(uint32_t substep, Dispatch&& dispatch) {
    return dispatch("sim_matter_grain_list_clear", substep) &&
        dispatch("sim_matter_grain_hash", substep) &&
        dispatch("sim_matter_grain_list_build", substep) &&
        dispatch("sim_matter_grain_step", substep);
}

// Frame: one substep group per substep. State ping-pongs between the
// canonical buffers (bank 0) and grain scratch (bank 1), so the substep count
// must be even for the frame to end in the canonical buffers. Each dispatch
// observes the complete previous one through the backend barrier. Contact
// history needs no prologue: it is updated in place in each grain's own block
// (sim_matter_grain.glsl). `dispatch(kernel, substep)` receives the substep index.
template <class Dispatch>
bool dispatchMatterGrainStages(uint32_t substeps, Dispatch&& dispatch) {
    if (!substeps || (substeps & 1u)) {
        return false;
    }
    for (uint32_t step = 0; step < substeps; ++step) {
        if (!dispatchMatterGrainSubstep(step, dispatch)) {
            return false;
        }
    }
    return true;
}

} // namespace RayTrophiSim::Fluid
