#pragma once

#include <cstdint>

namespace RayTrophiSim::Fluid {

// Frame: clear both generation hash tables, hash the canonical state, then one
// fused contact/integrate/hash dispatch per substep. State ping-pongs between
// the canonical buffers (bank 0) and grain scratch (bank 1), so the substep
// count must be even for the frame to end in the canonical buffers. Each
// dispatch observes the complete previous substep through the backend barrier.
// `dispatch(kernel, substep)` receives the substep index (0 for the prologue).
template <class Dispatch>
bool dispatchMatterGrainStages(uint32_t substeps, Dispatch&& dispatch) {
    if (!substeps || (substeps & 1u) || !dispatch("sim_matter_grain_clear", 0u) ||
        !dispatch("sim_matter_grain_hash", 0u)) {
        return false;
    }
    for (uint32_t step = 0; step < substeps; ++step) {
        if (!dispatch("sim_matter_grain_step", step)) {
            return false;
        }
    }
    return true;
}

} // namespace RayTrophiSim::Fluid
