#pragma once

#include "Vec3.h"

namespace Backend {
class IBackend;
}

struct SceneData;

struct FoliageWindUpdateStats {
    bool any_cpu_update = false;
    bool gpu_deform_applied = false;
    bool used_cpu_fallback = false;
    int enabled_group_count = 0;
};

class FoliageWindSystem {
public:
    static FoliageWindUpdateStats update(SceneData& scene, float time, Backend::IBackend* backend);
};
