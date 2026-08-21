#include "MeshEdit/SplineObject.h"

#include <algorithm>

namespace MeshEdit {

bool SplineObject::bounding_box(float, float, AABB& output_box) const {
    if (spline.points.empty()) return false;
    Vec3 localMin = spline.points.front().position;
    Vec3 localMax = localMin;
    for (const auto& point : spline.points) {
        localMin = Vec3((std::min)(localMin.x, point.position.x),
                        (std::min)(localMin.y, point.position.y),
                        (std::min)(localMin.z, point.position.z));
        localMax = Vec3((std::max)(localMax.x, point.position.x),
                        (std::max)(localMax.y, point.position.y),
                        (std::max)(localMax.z, point.position.z));
    }
    if (!transform) { output_box = AABB(localMin, localMax); return true; }
    const Matrix4x4 matrix = transform->getFinal();
    const Vec3 corners[8] = {
        {localMin.x, localMin.y, localMin.z}, {localMax.x, localMin.y, localMin.z},
        {localMin.x, localMax.y, localMin.z}, {localMax.x, localMax.y, localMin.z},
        {localMin.x, localMin.y, localMax.z}, {localMax.x, localMin.y, localMax.z},
        {localMin.x, localMax.y, localMax.z}, {localMax.x, localMax.y, localMax.z}
    };
    Vec3 worldMin = matrix * corners[0];
    Vec3 worldMax = worldMin;
    for (int i = 1; i < 8; ++i) {
        const Vec3 p = matrix * corners[i];
        worldMin = Vec3((std::min)(worldMin.x, p.x), (std::min)(worldMin.y, p.y), (std::min)(worldMin.z, p.z));
        worldMax = Vec3((std::max)(worldMax.x, p.x), (std::max)(worldMax.y, p.y), (std::max)(worldMax.z, p.z));
    }
    output_box = AABB(worldMin, worldMax);
    return true;
}

} // namespace MeshEdit
