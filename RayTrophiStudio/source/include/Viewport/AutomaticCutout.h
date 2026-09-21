#pragma once
#include "../../shaders/surface_coverage.h"

namespace Backend {
// Applied to GPU copies only. Authored material flags and offline cutout stay intact.
template<class Material>
void applyAutomaticViewportCutout(Material& m, bool enabled) {
    m.flags &= ~MATERIAL_FLAG_VIEWPORT_ALPHA_CUTOUT;
    const bool optical = !(m.transmission <= 0.0f) || m.transmission_tex != 0u ||
        (m.flags & ((1u << 17u) | (1u << 19u) | (1u << 24u))) != 0u;
    if (enabled && !optical && (m.opacity_tex != 0u || m.opacity < 0.999f))
        m.flags |= MATERIAL_FLAG_VIEWPORT_ALPHA_CUTOUT;
}
}
