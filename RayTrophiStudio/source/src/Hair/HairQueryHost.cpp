// Player hair rendering consumes the shared strand buffers through Vulkan.
// Offline CPU ray/brush queries do not construct an Embree cache in this target.
#include "Hair/HairSystem.h"

namespace Hair {
void HairSystem::buildBVHImpl(bool) {
}

void HairSystem::releaseCpuBVH() {
    m_embreeScene = nullptr;
}

bool HairSystem::intersect(const Vec3&, const Vec3&, float, float, HairHitInfo&) const {
    return false;
}

bool HairSystem::intersectVolumetric(const Vec3&, const Vec3&, float, float, float,
                                     HairHitInfo&) const {
    return false;
}

bool HairSystem::occluded(const Vec3&, const Vec3&, float, float) const {
    return false;
}
} // namespace Hair
