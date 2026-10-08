#include "Fluid/MatterGrainGeometry.h"
#include "TriangleMesh.h"
#include "Transform.h"
#include "SurfaceMeshCache.h"
#include "ParticleSimulation.h"

#include <cmath>

namespace RayTrophiSim::Fluid {

bool collectFlatGrainCollider(const std::vector<std::shared_ptr<Hittable>>& objects,
    const std::string& name, std::vector<SurfaceMeshTriangle>& triangles) {
    std::vector<SurfaceMeshTriangle> result;
    for (const auto& object : objects) {
        const auto* mesh = dynamic_cast<const TriangleMesh*>(object.get());
        if (!mesh || mesh->nodeName != name || !mesh->geometry) {
            continue;
        }
        const auto& g = *mesh->geometry;
        const auto* points = g.get_attribute_data<Vec3>("P");
        const auto* local = g.get_attribute_data<Vec3>("P_orig");
        if (!points || g.indices.size() % 3 != 0) {
            return false;
        }
        const auto matrix = mesh->transform ? mesh->transform->getFinal() : Matrix4x4::identity();
        for (std::size_t f = 0; f < g.indices.size(); f += 3) {
            SurfaceMeshTriangle t;
            Vec3* vertices[] = {&t.p0, &t.p1, &t.p2};
            for (std::size_t corner = 0; corner < 3; ++corner) {
                const auto index = g.indices[f + corner];
                if (index >= g.get_vertex_count()) {
                    return false;
                }
                *vertices[corner] = local && mesh->transform
                    ? matrix.transform_point(local[index]) : points[index];
            }
            // A zero-area face has no surface to touch: a UV sphere's pole
            // rows are exactly that, and the BVH rejects them (the whole step
            // was held for a plain sphere primitive). Same threshold as the BVH;
            // nonfinite corners still reach it and fail there.
            const float area = (t.p1 - t.p0).cross(t.p2 - t.p0).length();
            if (std::isfinite(area) && area < 1e-10f) {
                continue;
            }
            result.push_back(t);
            if (result.size() > 4096) {
                return false;
            }
        }
    }
    if (result.empty()) {
        return false;
    }
    triangles = std::move(result);
    return true;
}

} // namespace RayTrophiSim::Fluid

namespace RayTrophiSim {
void ParticleSimulationSystem::setGrainColliderMeshResolver(
    std::function<bool(const ParticleColliderDesc&, std::vector<SurfaceMeshTriangle>&,
                       uint64_t&)> resolver) {
    grain_collider_mesh_resolver_ = std::move(resolver);
}
}
