#pragma once

#include <cstddef>

#include "scene_data.h"

namespace ParticleSystemUsage {

inline bool setEmitterOnly(SceneData& scene, std::size_t system_index, bool emitter_only) {
    if (system_index >= scene.particle_systems.size()) {
        return false;
    }

    scene.particle_systems[system_index].render.emitter_only = emitter_only;
    return true;
}

} // namespace ParticleSystemUsage
