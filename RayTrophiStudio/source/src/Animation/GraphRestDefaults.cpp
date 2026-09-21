#include "Animation/GraphRestDefaults.h"
#include "AnimationNodes.h"
#include "Animation/AnimationKeys.h"
namespace AnimationGraph {
Matrix4x4 graphRestLocalMatrix(const BoneData& bones, const std::string& name) {
    const auto found = bones.boneDefaultTransforms.find(name);
    return found == bones.boneDefaultTransforms.end() ? Matrix4x4::identity() : found->second;
}
BoneTransform graphRestLocalTRS(const BoneData& bones, const std::string& name) {
    BoneTransform result;
    RayTrophi::decomposeTRS(graphRestLocalMatrix(bones, name), result.translation, result.rotation, result.scale);
    return result;
}
}
