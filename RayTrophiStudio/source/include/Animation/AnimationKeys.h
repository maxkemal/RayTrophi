/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          Animation/AnimationKeys.h
* Author:        Kemal Demirtas
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*
* ANIMATION KEY TYPES - RayTrophi's own, not Assimp's.
*
* WHY THIS EXISTS (Assimp import replacement, Faz 0)
* -------------------------------------------------
* Assimp was never just a loader in this repo: its TYPES had moved into the
* core scene structures. AnimationData stored std::vector<aiVectorKey> /
* aiQuatKey, AnimatedObject::interpolate* and AnimationNodes::sample*Key took
* them by reference, and the .rtp project serializer read and wrote them. So
* "stop using Assimp" meant changing the animation key type across the WHOLE
* animation system - which is the expensive half of the job, and it has to
* happen BEFORE a second reader exists.
*
* ★ Writing a glTF reader first would mean two key representations alive at
* once, and that failure class has bitten this repo repeatedly (see
* feedback_flat_soa_is_the_geometry_model). One representation, then a reader.
*
* THIS FILE CHANGES NO BEHAVIOUR. It is a pure type migration: same layout
* (double time + value), same sampling maths, same results. The field names are
* deliberately NOT mTime/mValue - a rename makes it impossible for stale code to
* keep compiling against the old meaning by accident (CLAUDE.md rule 5).
* =========================================================================
*/
#pragma once

#include <cmath>
#include <vector>

#include "Vec3.h"
#include "Quaternion.h"
#include "Matrix4x4.h"

namespace RayTrophi {

// One position/scale key. `time` is in TICKS, the unit AnimationData::duration
// and ticksPerSecond are expressed in - NOT seconds. Keeping the double here is
// not decoration: FBX exports routinely carry tick counts that lose keys at
// float precision on long clips.
struct VectorKey {
    double time = 0.0;
    Vec3   value;

    VectorKey() = default;
    VectorKey(double t, const Vec3& v) : time(t), value(v) {}
};

// One rotation key. Quaternion is stored (w, x, y, z) as elsewhere in the engine.
struct QuatKey {
    double     time = 0.0;
    Quaternion value;

    QuatKey() = default;
    QuatKey(double t, const Quaternion& v) : time(t), value(v) {}
};

using VectorKeys = std::vector<VectorKey>;
using QuatKeys   = std::vector<QuatKey>;

// ---------------------------------------------------------------------------
// TRS decomposition, replacing aiMatrix4x4::Decompose().
//
// The bind-pose fallback in AnimationData::calculateAnimationTransform used to
// build an aiMatrix4x4 purely to call Decompose() on it. That single call was
// the reason <assimp/matrix4x4.h> had to be reachable from the animation core.
//
// Matches Matrix4x4::decompose()'s convention exactly (scale = COLUMN lengths,
// rotation = the matrix with those columns normalised), but returns the
// rotation as a quaternion instead of Euler angles - Euler would introduce a
// gimbal-order round trip the Assimp path never had.
// ---------------------------------------------------------------------------
inline void decomposeTRS(const Matrix4x4& mat, Vec3& position, Quaternion& rotation, Vec3& scale) {
    position = Vec3(mat.m[0][3], mat.m[1][3], mat.m[2][3]);

    float sx = std::sqrt(mat.m[0][0] * mat.m[0][0] + mat.m[1][0] * mat.m[1][0] + mat.m[2][0] * mat.m[2][0]);
    float sy = std::sqrt(mat.m[0][1] * mat.m[0][1] + mat.m[1][1] * mat.m[1][1] + mat.m[2][1] * mat.m[2][1]);
    float sz = std::sqrt(mat.m[0][2] * mat.m[0][2] + mat.m[1][2] * mat.m[1][2] + mat.m[2][2] * mat.m[2][2]);
    scale = Vec3(sx, sy, sz);

    if (sx < 1e-8f) sx = 1e-8f;
    if (sy < 1e-8f) sy = 1e-8f;
    if (sz < 1e-8f) sz = 1e-8f;

    Matrix4x4 rot = Matrix4x4::identity();
    rot.m[0][0] = mat.m[0][0] / sx; rot.m[0][1] = mat.m[0][1] / sy; rot.m[0][2] = mat.m[0][2] / sz;
    rot.m[1][0] = mat.m[1][0] / sx; rot.m[1][1] = mat.m[1][1] / sy; rot.m[1][2] = mat.m[1][2] / sz;
    rot.m[2][0] = mat.m[2][0] / sx; rot.m[2][1] = mat.m[2][1] / sy; rot.m[2][2] = mat.m[2][2] / sz;

    rotation = Quaternion::fromMatrix(rot);
}

} // namespace RayTrophi
