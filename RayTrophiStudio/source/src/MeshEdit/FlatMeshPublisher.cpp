#include "MeshEdit/FlatMeshPublisher.h"

#include "MeshEdit/FlatMeshValidator.h"
#include "TriangleMesh.h"
#include "Transform.h"

#include <algorithm>
#include <cmath>
#include <memory>

namespace MeshEdit {

namespace {

Vec3 safeNormal(const Vec3& value) {
    const float length = value.length();
    return length > 1.0e-12f ? value / length : Vec3(0.0f, 1.0f, 0.0f);
}

} // namespace

MeshOperationReport publishHalfEdgeMeshToFlat(TriangleMesh& mesh,
                                              const HalfEdgeMesh& topology,
                                              const DNA::GeometryDetail* source) {
    MeshOperationReport report;
    report.operation_id = "mesh.publish.half_edge_to_flat";
    report.undo_group = "mesh_topology_publish";

    std::string error;
    if (!topology.validate(&error)) {
        report.addError("topology_invalid", "publish rejected: " + error);
        return report;
    }

    std::vector<std::array<HEIndex, 3>> triangles;
    topology.triangulate(triangles);
    if (triangles.empty() && topology.liveFaceCount() != 0) {
        report.addError("triangulation_failed", "validated topology produced no triangles");
        return report;
    }

    auto next = std::make_shared<DNA::GeometryDetail>();
    const size_t vertex_count = topology.vertices.size();
    next->resize_vertices(vertex_count);
    next->add_attribute<Vec3>("P");
    next->add_attribute<Vec3>("N");
    next->add_attribute<Vec3>("P_orig");
    next->add_attribute<Vec3>("N_orig");
    next->add_attribute<Vec2>("uv");
    next->add_attribute<uint16_t>("materialID");

    Vec3* p_orig = next->get_attribute_data_mut<Vec3>("P_orig");
    Vec3* p = next->get_attribute_data_mut<Vec3>("P");
    Vec3* n_orig = next->get_attribute_data_mut<Vec3>("N_orig");
    Vec3* n = next->get_attribute_data_mut<Vec3>("N");
    Vec2* uv = next->get_attribute_data_mut<Vec2>("uv");
    uint16_t* material = next->get_attribute_data_mut<uint16_t>("materialID");
    if (!p_orig || !p || !n_orig || !n || !uv || !material) {
        report.addError("attribute_allocation_failed", "flat publish could not allocate core attributes");
        return report;
    }

    const size_t copied = source ? (std::min)(source->get_vertex_count(), vertex_count) : 0;
    const Vec2* source_uv = source ? source->get_attribute_data<Vec2>("uv") : nullptr;
    const uint16_t* source_material = source ? source->get_attribute_data<uint16_t>("materialID") : nullptr;
    for (size_t i = 0; i < vertex_count; ++i) {
        p_orig[i] = topology.vertices[i].position;
        p[i] = topology.vertices[i].position;
        n_orig[i] = Vec3(0.0f, 1.0f, 0.0f);
        n[i] = n_orig[i];
        if (i < copied && source_uv) uv[i] = source_uv[i];
        else uv[i] = Vec2(0.0f, 0.0f);
        if (i < copied && source_material) material[i] = source_material[i];
        else material[i] = 0;
    }

    std::vector<Vec3> normal_accum(vertex_count, Vec3(0.0f, 0.0f, 0.0f));
    next->indices.reserve(triangles.size() * 3);
    for (const auto& tri : triangles) {
        if (tri[0] < 0 || tri[1] < 0 || tri[2] < 0 ||
            static_cast<size_t>(tri[0]) >= vertex_count ||
            static_cast<size_t>(tri[1]) >= vertex_count ||
            static_cast<size_t>(tri[2]) >= vertex_count) {
            report.addError("index_out_of_range", "triangulation produced an invalid flat index");
            return report;
        }
        const Vec3 a = topology.vertices[static_cast<size_t>(tri[0])].position;
        const Vec3 b = topology.vertices[static_cast<size_t>(tri[1])].position;
        const Vec3 c = topology.vertices[static_cast<size_t>(tri[2])].position;
        const Vec3 face_normal = (b - a).cross(c - a);
        normal_accum[static_cast<size_t>(tri[0])] += face_normal;
        normal_accum[static_cast<size_t>(tri[1])] += face_normal;
        normal_accum[static_cast<size_t>(tri[2])] += face_normal;
        next->indices.push_back(static_cast<uint32_t>(tri[0]));
        next->indices.push_back(static_cast<uint32_t>(tri[1]));
        next->indices.push_back(static_cast<uint32_t>(tri[2]));
    }
    for (size_t i = 0; i < vertex_count; ++i) {
        n_orig[i] = safeNormal(normal_accum[i]);
        const Matrix4x4 normal_matrix = mesh.transform
            ? mesh.transform->getNormalTransform() : Matrix4x4::identity();
        n[i] = safeNormal(normal_matrix.transform_vector(n_orig[i]));
        if (mesh.transform) p[i] = mesh.transform->getFinal().transform_point(p_orig[i]);
    }

    mesh.geometry = std::move(next);
    report.changed.vertices_changed = vertex_count;
    report.changed.faces_changed = topology.liveFaceCount();
    report.changed.triangles_changed = triangles.size();
    report.ok = true;
    if (copied < vertex_count) {
        report.addWarning("attribute_defaults", "new topology vertices received default UV/material attributes");
    }
    return report;
}

bool runFlatMeshPublisherSelfTest(std::string& report) {
    HalfEdgeMesh topology;
    const std::vector<Vec3> positions = {
        Vec3(-1.0f, -1.0f, 0.0f), Vec3(1.0f, -1.0f, 0.0f),
        Vec3(1.0f, 1.0f, 0.0f), Vec3(-1.0f, 1.0f, 0.0f)
    };
    if (!topology.buildFromPolygons(positions, {{0, 1, 2, 3}})) {
        report = "topology build failed";
        return false;
    }
    TriangleMesh mesh;
    const MeshOperationReport result = publishHalfEdgeMeshToFlat(mesh, topology);
    if (!result.ok || !mesh.geometry) {
        report = "publish failed";
        return false;
    }
    const auto validation = validateFlatMesh(mesh);
    if (!validation.valid) {
        report = "published mesh validation failed";
        return false;
    }
    report = "ok: vertices=" + std::to_string(mesh.num_vertices()) +
             ", triangles=" + std::to_string(mesh.num_triangles());
    return true;
}

} // namespace MeshEdit
