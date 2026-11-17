#pragma once

#include "Vec2.h"
#include "Vec3.h"
#include <algorithm>
#include <cmath>

namespace Paint::HeightNormalMath {

// Existing paint height->normal metric: two texels in world units, derived
// from the UV Jacobian, not neighbour barycentric extrapolation across seams.
inline Vec2 twoTexelWorldSpan(const Vec3& p0, const Vec3& p1, const Vec3& p2,
                             const Vec2& uv0, const Vec2& uv1, const Vec2& uv2,
                             int width, int height) {
    const Vec3 e1 = p1 - p0;
    const Vec3 e2 = p2 - p0;
    const float du1 = uv1.u - uv0.u;
    const float dv1 = uv1.v - uv0.v;
    const float du2 = uv2.u - uv0.u;
    const float dv2 = uv2.v - uv0.v;
    const float determinant = du1 * dv2 - du2 * dv1;
    float horizontal = 1.0f;
    float vertical = 1.0f;
    if (std::abs(determinant) >= 1e-10f) {
        const Vec3 dPdU = (e1 * dv2 - e2 * dv1) * (1.0f / determinant);
        const Vec3 dPdV = (e2 * du1 - e1 * du2) * (1.0f / determinant);
        horizontal = dPdU.length() * (width > 1 ? 2.0f / (width - 1) : 0.0f);
        vertical = dPdV.length() * (height > 1 ? 2.0f / (height - 1) : 0.0f);
    }
    const float average_edge = (e1.length() + e2.length() + (e1 - e2).length()) / 3.0f;
    const float minimum = std::max(1e-5f, average_edge * 1e-4f);
    return Vec2(std::max(minimum, horizontal), std::max(minimum, vertical));
}

inline Vec3 combinePD(const Vec3& base_normal, const Vec3& detail_normal) {
    const Vec3 base = base_normal.normalize();
    const Vec3 detail = detail_normal.normalize();
    return Vec3(base.x / std::max(0.15f, base.z) + detail.x / std::max(0.15f, detail.z),
                base.y / std::max(0.15f, base.z) + detail.y / std::max(0.15f, detail.z),
                1.0f).normalize();
}

} // namespace Paint::HeightNormalMath
