#pragma once

// Faz 1 adapter: conversions between RayTrophi math types (global ::Vec3,
// ::Matrix4x4) and Jolt math types (JPH::Vec3/RVec3/Quat/Mat44).
//
// This header pulls in Jolt headers, so it may ONLY be included from .cpp
// translation units that are already Jolt-aware (JoltWorld.cpp and friends).
// Public consumers use JoltWorld.h, which speaks pure RayTrophi types.
//
// Convention note: both ::Matrix4x4 and JPH::Mat44 use column-vector / basis-
// in-columns semantics (transform is M * v, translation in the last column),
// so the 3x3 basis maps DIRECTLY with no transpose. ::Matrix4x4 stores
// m[row][col]; JPH::Mat44 column j element i == m[i][j].

#include <Jolt/Jolt.h>
#include <Jolt/Math/Vec3.h>
#include <Jolt/Math/Vec4.h>
#include <Jolt/Math/Quat.h>
#include <Jolt/Math/Mat44.h>

#include <cmath>

#include "Vec3.h"
#include "Matrix4x4.h"

namespace RayTrophiSim {
namespace JoltIntegration {

// ---- Vector conversions --------------------------------------------------

inline JPH::Vec3 toJolt(const ::Vec3& v) {
    return JPH::Vec3(v.x, v.y, v.z);
}

inline JPH::RVec3 toJoltR(const ::Vec3& v) {
    return JPH::RVec3((JPH::Real)v.x, (JPH::Real)v.y, (JPH::Real)v.z);
}

inline ::Vec3 toRT(JPH::Vec3Arg v) {
    return ::Vec3(v.GetX(), v.GetY(), v.GetZ());
}

#ifdef JPH_DOUBLE_PRECISION
// Only a distinct overload when RVec3 != Vec3 (double precision build); in the
// default single-precision build RVec3Arg IS Vec3Arg, so the above covers it.
inline ::Vec3 toRT(JPH::RVec3Arg v) {
    return ::Vec3((float)v.GetX(), (float)v.GetY(), (float)v.GetZ());
}
#endif

// ---- Matrix decomposition (RT -> Jolt) -----------------------------------
// Splits a RayTrophi transform into the position + rotation a Jolt body wants,
// plus the residual scale (Jolt bodies are rigid; scale must go into the shape
// at creation time, never into the body pose).

inline ::Vec3 extractScale(const ::Matrix4x4& m) {
    const float sx = std::sqrt(m.m[0][0]*m.m[0][0] + m.m[1][0]*m.m[1][0] + m.m[2][0]*m.m[2][0]);
    const float sy = std::sqrt(m.m[0][1]*m.m[0][1] + m.m[1][1]*m.m[1][1] + m.m[2][1]*m.m[2][1]);
    const float sz = std::sqrt(m.m[0][2]*m.m[0][2] + m.m[1][2]*m.m[1][2] + m.m[2][2]*m.m[2][2]);
    return ::Vec3(sx, sy, sz);
}

inline void decomposeToJolt(const ::Matrix4x4& m, JPH::RVec3& outPos, JPH::Quat& outRot,
                            ::Vec3* outScale = nullptr) {
    outPos = JPH::RVec3((JPH::Real)m.m[0][3], (JPH::Real)m.m[1][3], (JPH::Real)m.m[2][3]);

    ::Vec3 s = extractScale(m);
    if (outScale) *outScale = s;

    const float ix = (s.x > 1e-8f) ? 1.0f / s.x : 0.0f;
    const float iy = (s.y > 1e-8f) ? 1.0f / s.y : 0.0f;
    const float iz = (s.z > 1e-8f) ? 1.0f / s.z : 0.0f;

    // Scale-free basis columns (column j is the transformed axis j).
    JPH::Mat44 r = JPH::Mat44(
        JPH::Vec4(m.m[0][0]*ix, m.m[1][0]*ix, m.m[2][0]*ix, 0.0f),
        JPH::Vec4(m.m[0][1]*iy, m.m[1][1]*iy, m.m[2][1]*iy, 0.0f),
        JPH::Vec4(m.m[0][2]*iz, m.m[1][2]*iz, m.m[2][2]*iz, 0.0f),
        JPH::Vec4(0.0f, 0.0f, 0.0f, 1.0f));
    outRot = r.GetQuaternion().Normalized();
}

// ---- Compose (Jolt -> RT) ------------------------------------------------
// Builds a rotation+translation RayTrophi matrix (no scale) from a Jolt body
// pose. Callers that need scale re-apply it themselves.

inline ::Matrix4x4 composeRT(JPH::RVec3Arg pos, JPH::QuatArg rot) {
    JPH::Mat44 rm = JPH::Mat44::sRotation(rot);
    ::Matrix4x4 out = ::Matrix4x4::identity();
    for (int j = 0; j < 3; ++j) {
        JPH::Vec4 col = rm.GetColumn4(j);
        out.m[0][j] = col.GetX();
        out.m[1][j] = col.GetY();
        out.m[2][j] = col.GetZ();
    }
    out.m[0][3] = (float)pos.GetX();
    out.m[1][3] = (float)pos.GetY();
    out.m[2][3] = (float)pos.GetZ();
    return out;
}

} // namespace JoltIntegration
} // namespace RayTrophiSim
