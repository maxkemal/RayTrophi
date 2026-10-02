#pragma once

#include "Vec3.h"
#include "InstanceGroup.h"

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
    // The settings the animation actually runs with: authored values, or --
    // with inherit_atmosphere -- the climate's wind applied to them (see
    // InstanceGroup::WindSettings). The panel and scatter.get_wind report
    // this, never the raw fields, when a group inherits.
    static InstanceGroup::WindSettings effectiveSettings(const InstanceGroup::WindSettings& authored);
};
