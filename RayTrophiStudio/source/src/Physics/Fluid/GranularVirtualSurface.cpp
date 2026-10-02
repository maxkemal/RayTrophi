#include "Fluid/GranularVirtualSurface.h"

#include "Fluid/GranularVirtualRepresentation.h"
#include "TriangleMesh.h"

#include <algorithm>
#include <iterator>
#include <limits>

namespace RayTrophiSim::Fluid {
namespace {

std::size_t columnIndex(int x, int z, int nx) {
    return static_cast<std::size_t>(x) +
           static_cast<std::size_t>(z) * static_cast<std::size_t>(nx);
}

Vec3 surfaceNormal(
    const GranularVirtualRepresentation& representation,
    int x,
    int z) {
    const int nx = representation.resolution_x;
    const int nz = representation.resolution_z;
    const std::size_t center = columnIndex(x, z, nx);
    if (representation.column_valid[center] == 0) {
        return Vec3(0.0f, 1.0f, 0.0f);
    }
    const auto sample = [&](int sx, int sz) {
        sx = std::clamp(sx, 0, nx - 1);
        sz = std::clamp(sz, 0, nz - 1);
        const std::size_t index = columnIndex(sx, sz, nx);
        return representation.column_valid[index] != 0
            ? representation.height[index]
            : representation.height[center];
    };
    const float dx = sample(x + 1, z) - sample(x - 1, z);
    const float dz = sample(x, z + 1) - sample(x, z - 1);
    return Vec3(-dx, 2.0f * representation.cell_size, -dz).normalize();
}

} // namespace

bool updateGranularVirtualSurface(
    const GranularVirtualRepresentation& representation,
    std::uint16_t material_id,
    const std::string& node_name,
    std::shared_ptr<TriangleMesh>& mesh,
    bool& topology_changed,
    std::string& error) {
    topology_changed = false;
    error.clear();
    const int nx = representation.resolution_x;
    const int nz = representation.resolution_z;
    if (nx < 2 || nz < 2 || representation.cell_size <= 0.0f) {
        error = "Granular virtual surface requires at least a 2x2 height grid.";
        return false;
    }
    const std::size_t columns =
        static_cast<std::size_t>(nx) * static_cast<std::size_t>(nz);
    if (representation.height.size() != columns ||
        representation.column_valid.size() != columns) {
        error = "Granular virtual surface height buffers do not match their dimensions.";
        return false;
    }

    const std::size_t cells =
        static_cast<std::size_t>(nx - 1) * static_cast<std::size_t>(nz - 1);
    if (cells > static_cast<std::size_t>(
                    std::numeric_limits<std::uint32_t>::max() / 4u)) {
        error = "Granular virtual surface exceeds the 32-bit mesh-index contract.";
        return false;
    }
    const std::size_t vertex_count = cells * 4u;
    const std::size_t index_count = cells * 6u;
    if (!mesh || !mesh->geometry ||
        mesh->geometry->get_vertex_count() != vertex_count ||
        mesh->geometry->indices.size() != index_count) {
        mesh = std::make_shared<TriangleMesh>();
        mesh->nodeName = node_name;
        mesh->transient = true;
        auto& geometry = *mesh->geometry;
        geometry.resize_vertices(vertex_count);
        geometry.add_attribute<Vec3>("P");
        geometry.add_attribute<Vec3>("N");
        geometry.add_attribute<Vec3>("P_orig");
        geometry.add_attribute<Vec3>("N_orig");
        geometry.add_attribute<Vec2>("uv");
        geometry.add_attribute<std::uint16_t>("materialID");
        geometry.indices.resize(index_count);
        for (std::size_t cell = 0; cell < cells; ++cell) {
            const std::uint32_t base = static_cast<std::uint32_t>(cell * 4u);
            const std::size_t index = cell * 6u;
            geometry.indices[index + 0] = base + 0u;
            geometry.indices[index + 1] = base + 3u;
            geometry.indices[index + 2] = base + 1u;
            geometry.indices[index + 3] = base + 0u;
            geometry.indices[index + 4] = base + 2u;
            geometry.indices[index + 5] = base + 3u;
        }
        topology_changed = true;
    }

    mesh->nodeName = node_name;
    mesh->transient = true;
    auto& geometry = *mesh->geometry;
    Vec3* positions = geometry.get_attribute_data_mut<Vec3>("P");
    Vec3* normals = geometry.get_attribute_data_mut<Vec3>("N");
    Vec3* original_positions = geometry.get_attribute_data_mut<Vec3>("P_orig");
    Vec3* original_normals = geometry.get_attribute_data_mut<Vec3>("N_orig");
    Vec2* uvs = geometry.get_attribute_data_mut<Vec2>("uv");
    std::uint16_t* materials =
        geometry.get_attribute_data_mut<std::uint16_t>("materialID");
    if (!positions || !normals || !original_positions || !original_normals ||
        !uvs || !materials) {
        error = "Granular virtual surface could not allocate canonical mesh attributes.";
        return false;
    }

    const auto point = [&](int x, int z) {
        const std::size_t column = columnIndex(x, z, nx);
        return Vec3(
            representation.bounds_min.x +
                (static_cast<float>(x) + 0.5f) * representation.cell_size,
            representation.height[column],
            representation.bounds_min.z +
                (static_cast<float>(z) + 0.5f) * representation.cell_size);
    };
    for (int z = 0; z < nz - 1; ++z) {
        for (int x = 0; x < nx - 1; ++x) {
            const std::size_t cell =
                static_cast<std::size_t>(x) +
                static_cast<std::size_t>(z) * static_cast<std::size_t>(nx - 1);
            const std::size_t base = cell * 4u;
            const bool valid =
                representation.column_valid[columnIndex(x, z, nx)] != 0 &&
                representation.column_valid[columnIndex(x + 1, z, nx)] != 0 &&
                representation.column_valid[columnIndex(x, z + 1, nx)] != 0 &&
                representation.column_valid[columnIndex(x + 1, z + 1, nx)] != 0;
            Vec3 cell_positions[4] = {
                point(x, z),
                point(x + 1, z),
                point(x, z + 1),
                point(x + 1, z + 1)
            };
            Vec3 cell_normals[4] = {
                surfaceNormal(representation, x, z),
                surfaceNormal(representation, x + 1, z),
                surfaceNormal(representation, x, z + 1),
                surfaceNormal(representation, x + 1, z + 1)
            };
            if (!valid) {
                const Vec3 collapsed(
                    representation.bounds_min.x +
                        (static_cast<float>(x) + 0.5f) * representation.cell_size,
                    representation.bounds_min.y,
                    representation.bounds_min.z +
                        (static_cast<float>(z) + 0.5f) * representation.cell_size);
                std::fill(std::begin(cell_positions), std::end(cell_positions), collapsed);
                std::fill(
                    std::begin(cell_normals),
                    std::end(cell_normals),
                    Vec3(0.0f, 1.0f, 0.0f));
            }
            for (std::size_t corner = 0; corner < 4u; ++corner) {
                const std::size_t vertex = base + corner;
                positions[vertex] = cell_positions[corner];
                original_positions[vertex] = cell_positions[corner];
                normals[vertex] = cell_normals[corner];
                original_normals[vertex] = cell_normals[corner];
                uvs[vertex] = Vec2(
                    static_cast<float>(x + (corner == 1u || corner == 3u)) /
                        static_cast<float>(nx - 1),
                    static_cast<float>(z + (corner >= 2u)) /
                        static_cast<float>(nz - 1));
                materials[vertex] = material_id;
            }
        }
    }
    return true;
}

} // namespace RayTrophiSim::Fluid
