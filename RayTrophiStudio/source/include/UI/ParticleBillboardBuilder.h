#pragma once

#include "Viewport/ParticleBillboardData.h"

struct SceneData;

namespace ParticleBillboardBuilder {

// Collects the raster billboards of every visible, non-emitter-only particle
// system and of every fluid domain in Particles render mode. Each particle
// system contributes one LUT row per appearance profile; each fluid domain one
// constant row. The profile's blend picks the additive or alpha group.
//
// `viewport_device` is the VkDevice of the backend that will draw. A
// device-resident system on that same device becomes a pulled draw (no host
// copy); any other system is expanded into CPU quads from host state.
void build(const SceneData& scene, void* viewport_device, ParticleBillboardUpload& out);

} // namespace ParticleBillboardBuilder
