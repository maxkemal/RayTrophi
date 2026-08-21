#pragma once

#include "Vec3.h"

#include <array>
#include <vector>

namespace MeshEdit {

// Ear-clips a planar polygon and returns indices into the original point list.
// Unlike a fan, this remains valid for the concave vertex patches produced by
// bevel and prevents overlap/sliver triangles at multi-face corners.
std::vector<std::array<int, 3>> triangulatePlanarPolygon(
    const std::vector<Vec3>& points, const Vec3& reference_normal);

} // namespace MeshEdit
