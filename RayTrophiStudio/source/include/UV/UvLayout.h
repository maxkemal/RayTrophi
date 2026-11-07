// SPDX-License-Identifier: MIT
#pragma once

#include "Vec2.h"
#include "Vec3.h"

#include <cstdint>
#include <string>
#include <vector>

namespace UV {

// A detached, flat snapshot. No Triangle facades or scene/UI ownership.
struct Layout {
    std::vector<Vec3> positions;
    std::vector<Vec2> coordinates;
    std::vector<uint32_t> indices;
    std::vector<uint32_t> islands; // per face, derived from shared indexed edges
    uint32_t island_count = 0;
    std::string revision;
    // Optional new vertex -> original vertex map, produced by corner edits/atlas.
    std::vector<uint32_t> source_vertices;
};

bool validate(const Layout& layout, std::string& error);
void buildIslands(Layout& layout);
std::string fingerprint(const Layout& layout);
bool transform(Layout& layout, const std::vector<uint32_t>& islands,
               Vec2 offset, Vec2 scale, Vec2 pivot, float degrees,
               bool normalize, std::string& error, bool swap_axes = false,
               const std::vector<uint32_t>* vertices = nullptr);
bool projectFromView(Layout& layout, const std::vector<Vec3>& world_positions,
                     Vec3 eye, Vec3 forward, Vec3 up, std::string& error,
                     bool orthographic = false);
bool xatlasAvailable();
bool applyCornerCoordinates(Layout& layout, const std::vector<Vec2>& corners,
                            std::string& error);
// Preserves face/corner order. Carries a vertex remap when UV seams split vertices.
bool atlas(Layout& layout, bool unwrap, uint32_t resolution, uint32_t padding,
           std::string& error, uint32_t* preserved_faces = nullptr);

} // namespace UV
