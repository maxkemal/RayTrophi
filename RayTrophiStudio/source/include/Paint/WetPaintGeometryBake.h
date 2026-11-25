#pragma once
// Parallel CPU geometry baker for WetPaintGpu surfaces (O7c).
//
// Extracted from WetPaintGpu.cpp (>2000 lines) to parallelize UV-space triangle
// rasterization, skip redundant gutter passes/copies when a surface already
// fills its UV page (valid_texels == W*H), and parallelize tile AABB reduction.
// Output layout and numerical rules match the original serial baker.

#include "Vec2.h"
#include "Vec3.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace Paint {

struct WetPaintBakeFace {
    std::array<Vec3, 3> positions;
    std::array<Vec3, 3> normals;
    std::array<Vec2, 3> uvs;
};

struct WetPaintBakeTile {
    uint32_t surface = 0;
    uint32_t tx = 0;
    uint32_t ty = 0;
    uint32_t valid_texels = 0;
    float bmin[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float bmax[4] = {0.0f, 0.0f, 0.0f, 0.0f};
};
static_assert(sizeof(WetPaintBakeTile) == 48, "WetPaintBakeTile must match GpuTile");

struct WetPaintBakeResult {
    std::vector<float> pos_valid;
    std::vector<float> flow_normal;
    std::vector<Vec2> normal_spans;
    std::vector<WetPaintBakeTile> tiles;
    uint32_t tiles_x = 0;
    uint32_t tiles_y = 0;
    uint32_t valid_texels = 0;
    uint32_t gutter_texels = 0;
    uint32_t triangles = 0;
};

bool bakeWetPaintGeometry(const std::vector<WetPaintBakeFace>& faces,
                          uint32_t width,
                          uint32_t height,
                          uint32_t tile_size,
                          int gutter_texels,
                          const std::string& object_name,
                          int uv_set,
                          WetPaintBakeResult& out,
                          std::string& error);

} // namespace Paint
