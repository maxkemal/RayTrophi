#pragma once

#include "Fluid/FluidLevelSet.h"

namespace RayTrophiSim {
struct SimulationGridDomainDesc;
namespace Fluid {

// One resolver for CPU and GPU composition. Authored source/domain materials win.
inline int surfaceMaterialForParticle(const FluidParticles& particles, std::size_t index,
    const SubstanceMaterialEntry* entries, std::size_t count, int fallback) {
    const uint32_t tag = index < particles.substance_tag.size()
        ? particles.substance_tag[index] : 0u;
    for (std::size_t i = 0; i < count; ++i) {
        if (!entries[i].frozen_only && entries[i].tag == tag && entries[i].material_id >= 0) {
            return entries[i].material_id;
        }
    }
    if (fallback >= 0) {
        return fallback;
    }
    const bool frozen = index < particles.flags.size() &&
        (particles.flags[index] & kParticleFlagFrozen) != 0u;
    if (frozen) {
        for (std::size_t i = 0; i < count; ++i) {
            if (entries[i].frozen_only) {
                return entries[i].material_id;
            }
        }
    }
    return fallback;
}

// Render publication may lazily create the editable Ice asset. Read-only parity
// probes resolve existing assets without changing the material registry.
std::size_t collectPhaseSurfaceMaterialEntries(const SimulationGridDomainDesc& domain,
    const FluidParticles& particles, SubstanceMaterialEntry* entries,
    bool createDefault = false);
uint64_t frozenSurfaceMaterialRegistrySignature();

} // namespace Fluid
} // namespace RayTrophiSim
