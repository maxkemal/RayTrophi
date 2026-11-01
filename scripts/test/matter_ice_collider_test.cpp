#include "Fluid/MatterIceCollider.h"
#include "Fluid/FluidParticles.h"
#include "FluidGrid.h"
#include "SurfaceMeshCache.h"
#include <cassert>
#include <cmath>

int main() {
    using namespace RayTrophiSim::Fluid;
    using RayTrophiSim::SurfaceMeshTriangle;
    FluidSim::FluidGrid grid(4, 4, 4, .1f, Vec3(0.0f));
    FluidParticles particles;
    particles.emit(Vec3(.15f, .15f, .15f), Vec3(0.0f));
    particles.flags[0] |= kParticleFlagFrozen;
    std::vector<SurfaceMeshTriangle> faces;
    assert(appendMatterIceColliderFaces(particles, grid, faces) == 12);
    const Vec3 center(.15f, .15f, .15f);
    for (const auto& face : faces) {
        const Vec3 normal = Vec3::cross(face.p1 - face.p0, face.p2 - face.p0);
        const Vec3 offset = face.p0 - center;
        assert(normal.x * offset.x + normal.y * offset.y + normal.z * offset.z > 0.0f);
    }
    // Two cells share an internal face; duplicate parcels do not add faces.
    particles.emit(Vec3(.25f, .15f, .15f), Vec3(0.0f));
    particles.emit(Vec3(.16f, .15f, .15f), Vec3(0.0f));
    particles.flags[1] |= kParticleFlagFrozen;
    particles.flags[2] |= kParticleFlagFrozen;
    faces.clear();
    assert(appendMatterIceColliderFaces(particles, grid, faces) == 20);
    // No interior face at x=.2; both outer faces retain exact grid-node bounds.
    for (const auto& face : faces) {
        const bool on_internal = face.p0.x == .2f && face.p1.x == .2f && face.p2.x == .2f;
        assert(!on_internal);
    }
    for (auto& flag : particles.flags) {
        flag &= ~kParticleFlagFrozen;
    }
    faces.clear();
    assert(appendMatterIceColliderFaces(particles, grid, faces) == 0);
}
