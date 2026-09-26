#pragma once

#include "Viewport/ParticleBillboardData.h"

struct SceneData;

namespace ParticleBillboardBuilder {

// Collects the raster billboards of every visible, non-emitter-only particle
// system and of every fluid domain in Particles render mode. Each particle
// system contributes one LUT row per appearance profile; each fluid domain one
// constant row. The profile's blend picks the additive or alpha group.
void build(const SceneData& scene, ParticleBillboardUpload& out);

} // namespace ParticleBillboardBuilder
