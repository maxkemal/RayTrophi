// SPDX-License-Identifier: MIT
#include "DNA/GeometryDetail.h"
#include "DNA/GeometryRemap.h"
#include "UV/UvLayout.h"

#include <iostream>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

void require(bool value, const char* message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

UV::Layout plane() {
    UV::Layout layout;
    layout.positions = {Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(1, 1, 0), Vec3(0, 1, 0)};
    layout.coordinates = {Vec2(0, 0), Vec2(1, 0), Vec2(1, 1), Vec2(0, 1)};
    layout.indices = {0, 1, 2, 0, 2, 3};
    UV::buildIslands(layout);
    return layout;
}

std::vector<Vec2> corners(const UV::Layout& layout) {
    std::vector<Vec2> result;
    for (uint32_t vertex : layout.indices) {
        result.push_back(layout.coordinates[vertex]);
    }
    return result;
}

void cornerCases() {
    auto original = plane();
    auto values = corners(original);
    std::string error;
    auto same = original;
    require(UV::applyCornerCoordinates(same, values, error), "identity corner write failed");
    require(same.indices == original.indices, "identity changed topology");
    require(UV::fingerprint(same) == UV::fingerprint(original), "identity changed UVs");
    auto split = original;
    values[3] = Vec2(0.25f, 0.5f);
    require(UV::applyCornerCoordinates(split, values, error), "corner split failed");
    require(split.coordinates.size() == 5, "expected exactly one appended vertex");
    require(split.indices == std::vector<uint32_t>({0, 1, 2, 4, 2, 3}), "face order changed");
    require(split.source_vertices == std::vector<uint32_t>({0, 1, 2, 3, 0}), "bad source map");
    require(split.positions[4].x == original.positions[0].x, "3D position changed");
    require(split.coordinates[0] == original.coordinates[0], "unselected face changed");
    auto repeated = corners(split);
    repeated[4] = Vec2(0.75f, 0.75f);
    require(UV::applyCornerCoordinates(split, repeated, error), "composed remap failed");
    require(split.source_vertices.back() == 2, "composed map lost the original vertex");

    auto invalid = original;
    values.pop_back();
    require(!UV::applyCornerCoordinates(invalid, values, error), "bad count accepted");
    require(UV::fingerprint(invalid) == UV::fingerprint(original), "bad count changed layout");
    values = corners(original);
    values[0].u = (std::numeric_limits<float>::quiet_NaN)();
    require(!UV::applyCornerCoordinates(invalid, values, error), "NaN corner accepted");
    require(UV::fingerprint(invalid) == UV::fingerprint(original), "NaN changed layout");

    auto unused = original;
    unused.positions.push_back(Vec3(7, 8, 9));
    unused.coordinates.push_back(Vec2(0.8f, 0.9f));
    require(UV::applyCornerCoordinates(unused, corners(original), error), "unused vertex failed");
    require(unused.coordinates.size() == 5 && unused.coordinates[4] == Vec2(0.8f, 0.9f),
            "unused attribute slot was removed or modified");

    // Coincident geometry/UV values on separate source vertices must not weld.
    auto coincident = original;
    coincident.positions[3] = coincident.positions[0];
    coincident.coordinates[3] = coincident.coordinates[0];
    require(UV::applyCornerCoordinates(coincident, corners(coincident), error), "coincident failed");
    require(coincident.indices == original.indices, "separate source vertices were welded");
}

void attributeCases() {
    auto layout = plane();
    DNA::GeometryDetail geometry;
    geometry.resize_vertices(4);
    geometry.indices.assign(layout.indices.begin(), layout.indices.end());
    geometry.add_attribute<Vec3>("P");
    geometry.add_attribute<Vec3>("P_orig");
    geometry.add_attribute<Vec3>("N");
    geometry.add_attribute<Vec3>("N_orig");
    geometry.add_attribute<Vec2>("uv");
    geometry.add_attribute<Vec2>("uv1");
    geometry.add_attribute<uint16_t>("materialID");
    geometry.add_attribute<float>("paint_mask");
    geometry.add_attribute<std::string>("custom_label");
    geometry.skin_weights.resize(4);
    geometry.last_skinned_pose_hash = 123;
    for (uint32_t i = 0; i < 4; ++i) {
        geometry.get_positions_mut()[i] = layout.positions[i];
        geometry.get_attribute_data_mut<Vec3>("P_orig")[i] = layout.positions[i];
        geometry.get_normals_mut()[i] = Vec3(0, 0, 1);
        geometry.get_attribute_data_mut<Vec3>("N_orig")[i] = Vec3(0, 0, 1);
        geometry.get_uvs_mut()[i] = layout.coordinates[i];
        geometry.get_attribute_data_mut<Vec2>("uv1")[i] = Vec2(float(i), -float(i));
        geometry.get_attribute_data_mut<uint16_t>("materialID")[i] = uint16_t(10 + i);
        geometry.get_attribute_data_mut<float>("paint_mask")[i] = float(i);
        geometry.get_attribute_data_mut<std::string>("custom_label")[i] = "vertex " + std::to_string(i);
        geometry.skin_weights[i] = {{int(i), 0.25f}, {int(i + 1), 0.75f}};
    }
    geometry.push_delta<float>("paint_mask", {0}, {0.9f}, false);
    geometry.get_attribute_data_mut<float>("paint_mask")[0] = 0.7f;
    auto values = corners(layout);
    values[3] = Vec2(0.25f, 0.5f);
    std::string error;
    require(UV::applyCornerCoordinates(layout, values, error), "fixture split failed");
    auto result = DNA::remapGeometry(geometry, layout.source_vertices, layout.indices, error);
    require(bool(result), error.c_str());
    require(result->get_vertex_count() == 5, "remap did not resize all channels");
    require(geometry.get_vertex_count() == 4 && geometry.delta_count() == 1, "source mutated");
    require(geometry.last_skinned_pose_hash == 123 && result->last_skinned_pose_hash == 0,
            "skin pose invalidation changed source or skipped the new topology");
    require(result->delta_count() == 0, "detached result still has a stale delta stack");
    require(result->get_attribute_data<float>("paint_mask")[4] == 0.7f,
            "remap discarded evaluated in-place attribute changes");
    require(result->get_attribute_data<Vec2>("uv1")[4] == Vec2(0, 0), "extra UV set lost");
    require(result->get_material_ids()[4] == 10, "material ID lost");
    require(result->get_normals()[4].z == 1, "normal lost");
    require(result->skin_weights[4] == geometry.skin_weights[0], "skin weights lost");
    require(result->get_attribute_data<std::string>("custom_label")[4] == "vertex 0",
            "non-trivial custom attribute lost");
    result->get_attribute_data_mut<std::string>("custom_label")[4] = "changed";
    require(geometry.get_attribute_data<std::string>("custom_label")[0] == "vertex 0",
            "custom attribute copy aliases source storage");
    require(!DNA::remapGeometry(geometry, {99}, {0, 0, 0}, error), "invalid source accepted");
    require(!DNA::remapGeometry(geometry, {0}, {0, 1, 0}, error), "invalid index accepted");
    geometry.skin_weights.pop_back();
    require(!DNA::remapGeometry(geometry, {0}, {0, 0, 0}, error), "short skin channel accepted");
}

void vertexTransformCases() {
    auto layout = plane();
    const auto before = layout;
    const std::vector<uint32_t> edge = {0, 1};
    std::string error;
    require(UV::transform(layout, {}, Vec2(0.25f, -0.5f), Vec2(1, 1), Vec2(),
        0, false, error, false, &edge), "edge endpoint transform failed");
    require(layout.coordinates[0] == Vec2(0.25f, -0.5f) &&
        layout.coordinates[1] == Vec2(1.25f, -0.5f), "edge endpoints not transformed");
    require(layout.coordinates[2] == before.coordinates[2] &&
        layout.coordinates[3] == before.coordinates[3], "unselected vertices moved");
    require(layout.indices == before.indices, "vertex move changed topology");
    const auto edited = UV::fingerprint(layout);
    for (const auto vertices : {std::vector<uint32_t>{}, std::vector<uint32_t>{99}}) {
        require(!UV::transform(layout, {}, Vec2(1, 1), Vec2(1, 1), Vec2(),
            0, false, error, false, &vertices), "invalid vertices accepted");
        require(UV::fingerprint(layout) == edited, "invalid selection partially mutated UVs");
    }
    require(!UV::transform(layout, {0}, Vec2(), Vec2(1, 1), Vec2(),
        0, false, error, false, &edge), "mixed selection accepted");
    require(UV::fingerprint(layout) == edited, "mixed selection mutated UVs");
    auto rotate = before;
    const std::vector<uint32_t> vertex = {1};
    require(UV::transform(rotate, {}, Vec2(), Vec2(2, 2), Vec2(),
        90, false, error, false, &vertex), "single vertex rotation failed");
    require(std::abs(rotate.coordinates[1].u) < 1.0e-5f &&
        std::abs(rotate.coordinates[1].v - 2) < 1.0e-5f, "rotation/scale pivot incorrect");
    require(rotate.coordinates[0] == before.coordinates[0] &&
        rotate.coordinates[2] == before.coordinates[2], "rotation moved unselected UVs");
}

void atlasCases() {
    std::string error;
    uint32_t preserved = 0;
    auto degenerate = plane();
    degenerate.indices.insert(degenerate.indices.end(), {0, 0, 1});
    const auto old_uvs = corners(degenerate);
    require(UV::atlas(degenerate, true, 128, 2, error, &preserved), error.c_str());
    require(preserved == 1 && degenerate.indices.size() == 9,
            "geometric degenerate face was dropped or not reported");
    const auto fresh_uvs = corners(degenerate);
    for (size_t i = 6; i < 9; ++i) {
        require(fresh_uvs[i] == old_uvs[i], "unmapped face UVs were overwritten");
    }
    auto flat_uv = plane();
    flat_uv.coordinates[3] = flat_uv.coordinates[2];
    const auto before_pack = corners(flat_uv);
    require(UV::atlas(flat_uv, false, 128, 2, error, &preserved), error.c_str());
    require(preserved == 1, "UV-degenerate face was not reported during pack");
    const auto packed = corners(flat_uv);
    for (size_t i = 3; i < 6; ++i) {
        require(packed[i] == before_pack[i], "pack changed ignored face UVs");
    }
    UV::Layout cube;
    cube.positions = {Vec3(-1,-1,-1), Vec3(1,-1,-1), Vec3(1,1,-1), Vec3(-1,1,-1),
                      Vec3(-1,-1,1), Vec3(1,-1,1), Vec3(1,1,1), Vec3(-1,1,1)};
    cube.coordinates.resize(8);
    cube.indices = {0,2,1, 0,3,2, 4,5,6, 4,6,7, 0,1,5, 0,5,4,
                    1,2,6, 1,6,5, 2,3,7, 2,7,6, 3,0,4, 3,4,7};
    const auto original = cube;
    require(UV::atlas(cube, true, 128, 2, error, &preserved), error.c_str());
    require(preserved == 0 && cube.coordinates.size() > 8, "welded cube did not split");
    require(cube.indices.size() == original.indices.size(), "atlas changed face count");
    for (size_t corner = 0; corner < cube.indices.size(); ++corner) {
        require(cube.source_vertices[cube.indices[corner]] == original.indices[corner],
                "atlas changed face/corner source order");
    }
}

} // namespace

int main() {
    try {
        cornerCases();
        attributeCases();
        vertexTransformCases();
        atlasCases();
        std::cout << "PASS: UV remap, vertex transforms, attributes and atlas regressions\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
