#pragma once

#include "Fluid/GrainMeshNear.h"
#include "TriangleMesh.h"

namespace RayTrophiSim::Fluid {

inline std::shared_ptr<TriangleMesh> buildDomainVisualFlatMesh(const GrainExemplar& exemplar,
    int material, const std::string& name, float volume_scale = 1.0f) {
    auto mesh = std::make_shared<TriangleMesh>();
    mesh->nodeName = name;
    mesh->transient = true;
    mesh->geometry = std::make_shared<DNA::GeometryDetail>();
    auto& geometry = *mesh->geometry;
    geometry.resize_vertices(exemplar.faces.size() * 3);
    geometry.add_attribute<Vec3>("P");
    geometry.add_attribute<Vec3>("P_orig");
    geometry.add_attribute<Vec3>("N");
    geometry.add_attribute<Vec3>("N_orig");
    geometry.add_attribute<Vec2>("uv");
    geometry.add_attribute<uint16_t>("materialID");
    auto* p = geometry.get_attribute_data_mut<Vec3>("P");
    auto* original = geometry.get_attribute_data_mut<Vec3>("P_orig");
    auto* n = geometry.get_attribute_data_mut<Vec3>("N");
    auto* original_n = geometry.get_attribute_data_mut<Vec3>("N_orig");
    auto* uv = geometry.get_attribute_data_mut<Vec2>("uv");
    auto* materials = geometry.get_attribute_data_mut<uint16_t>("materialID");
    for (std::size_t face = 0; face < exemplar.faces.size(); ++face) {
        const auto& source = exemplar.faces[face];
        for (std::size_t corner = 0; corner < 3; ++corner) {
            const auto vertex = face * 3 + corner;
            p[vertex] = source.p[corner] * volume_scale;
            original[vertex] = p[vertex];
            n[vertex] = source.n[corner];
            original_n[vertex] = n[vertex];
            uv[vertex] = source.uv[corner];
            materials[vertex] = material < 0 ? source.material : static_cast<uint16_t>(material);
            geometry.indices.push_back(static_cast<uint32_t>(vertex));
        }
    }
    return mesh;
}

} // namespace RayTrophiSim::Fluid
