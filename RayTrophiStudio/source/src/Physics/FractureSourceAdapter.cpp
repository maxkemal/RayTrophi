#include "FractureSourceAdapter.h"

#include "Triangle.h"
#include "TriangleMesh.h"

namespace RayTrophiSim {
bool isFractureSourceObject(const std::shared_ptr<Hittable>& object,
                            const std::string& node) {
    if (auto tri = std::dynamic_pointer_cast<Triangle>(object))
        return tri->getNodeName() == node;
    if (auto mesh = std::dynamic_pointer_cast<TriangleMesh>(object))
        return mesh->nodeName == node;
    return false;
}

bool gatherFractureSource(
    const std::vector<std::shared_ptr<Hittable>>& objects,
    const std::string& node, std::vector<FractureInputTri>& out,
    uint16_t& out_material) {
    out.clear();
    out_material = 0xFFFFu;
    for (const auto& object : objects) {
        if (auto tri = std::dynamic_pointer_cast<Triangle>(object)) {
            if (tri->getNodeName() != node) continue;
            out.push_back({tri->getVertexPosition(0), tri->getVertexPosition(1),
                           tri->getVertexPosition(2)});
            if (out_material == 0xFFFFu) out_material = tri->getMaterialID();
            continue;
        }
        auto mesh = std::dynamic_pointer_cast<TriangleMesh>(object);
        if (!mesh || mesh->nodeName != node || !mesh->geometry) continue;
        auto* geometry = mesh->geometry.get();
        const Vec3* positions = geometry->get_attribute_data<Vec3>("P");
        const Vec3* original = geometry->get_attribute_data<Vec3>("P_orig");
        const uint16_t* materials =
            geometry->get_attribute_data<uint16_t>("materialID");
        const Matrix4x4 transform = mesh->transform
            ? mesh->transform->getFinal() : Matrix4x4::identity();
        const auto& indices = geometry->indices;
        const std::size_t vertex_count = geometry->get_vertex_count();
        if ((!positions && !original) || indices.size() < 3u) continue;
        for (std::size_t i = 0; i + 2u < indices.size(); i += 3u) {
            const uint32_t ia = indices[i], ib = indices[i + 1u], ic = indices[i + 2u];
            if (ia >= vertex_count || ib >= vertex_count || ic >= vertex_count) continue;
            const Vec3 a = original ? transform.transform_point(original[ia]) : positions[ia];
            const Vec3 b = original ? transform.transform_point(original[ib]) : positions[ib];
            const Vec3 c = original ? transform.transform_point(original[ic]) : positions[ic];
            if (Vec3::cross(b - a, c - a).length() <= 1e-8f) continue;
            out.push_back({a, b, c});
            if (out_material == 0xFFFFu && materials) out_material = materials[ia];
        }
    }
    if (out_material == 0xFFFFu) out_material = 0u;
    return !out.empty();
}
} // namespace RayTrophiSim
