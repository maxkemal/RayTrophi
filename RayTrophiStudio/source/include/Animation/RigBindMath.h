#pragma once
#include "Matrix4x4.h"
#include "Animation/SkinWeightContract.h"
namespace RigAuthoring {
struct SkinSegment { int bone;Vec3 start,end; };
// Stable affine inverse in double precision (actor scales can be .0001).
bool bindAffineInverse(const Matrix4x4&,Matrix4x4&);
double segmentDistanceSquared(const Vec3&,const SkinSegment&);
VertexInfluences distanceSkinWeights(const Vec3&,const std::vector<SkinSegment>&,double distanceFloor,double& nearestDistance);
}
