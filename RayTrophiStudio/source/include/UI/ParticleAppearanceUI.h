#pragma once

#include <cstdint>

struct UIContext;
namespace RayTrophiSim {
struct ParticleEmitterDesc;
}

namespace ParticleAppearanceUI {

// Emitter > Appearance: profile picker, "new / duplicate / remove" and the
// editor of the picked profile (blend, colour ramp, opacity / size / emission
// curves) with a strip preview of the baked LUT. Every write goes through
// rtapi (updateParticleEmitter / *ParticleAppearance), the same service IPC
// and Python use, so the panel cannot hold state of its own.
void drawEmitterAppearance(UIContext& ctx, uint32_t system_id,
                           const RayTrophiSim::ParticleEmitterDesc& emitter);

} // namespace ParticleAppearanceUI
