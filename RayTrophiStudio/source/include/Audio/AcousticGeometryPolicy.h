#pragma once

#include "TriangleMesh.h"
#include "VDBVolume.h"

namespace audio {
// The current boundary model covers authored flat surfaces. Simulation visuals
// are not acoustic boundaries; in particular a mist volume is not a solid wall.
inline bool isAcousticSimulationVisual(const Hittable& object) {
    if (const auto* mesh = dynamic_cast<const TriangleMesh*>(&object)) {
        return mesh->transient;
    }
    if (const auto* volume = dynamic_cast<const VDBVolume*>(&object)) {
        return volume->transient;
    }
    return false;
}

} // namespace audio
