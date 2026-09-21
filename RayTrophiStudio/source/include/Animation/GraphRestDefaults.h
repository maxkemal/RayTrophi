#pragma once
#include "Matrix4x4.h"
#include <string>
struct BoneData;
namespace AnimationGraph {
struct BoneTransform;
Matrix4x4 graphRestLocalMatrix(const BoneData&, const std::string& name);
BoneTransform graphRestLocalTRS(const BoneData&, const std::string& name);
}
