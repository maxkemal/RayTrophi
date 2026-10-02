#include "Animation/SemanticTagResolver.h"
#include "Animation/RigView.h"
#include "scene_data.h"
#include <algorithm>
#include <cctype>

namespace RayTrophi {

static std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return std::tolower(c); });
    return s;
}

SemanticTagResolver& SemanticTagResolver::instance() {
    static SemanticTagResolver s_instance;
    return s_instance;
}

void SemanticTagResolver::registerSkeletonMapping(
    const std::string& skeletonSignature,
    const std::unordered_map<std::string, std::string>& tagToBoneMap
) {
    m_mappings[skeletonSignature] = tagToBoneMap;
}

std::string SemanticTagResolver::normalizeTag(const std::string& rawNameOrTag) const {
    const std::string lower = toLower(rawNameOrTag);
    if (lower == "head" || lower == "head_bone") return "Head";
    if (lower == "leftarm" || lower == "left_arm" || lower == "leftupperarm" || lower == "arm.l") return "LeftUpperArm";
    if (lower == "rightarm" || lower == "right_arm" || lower == "rightupperarm" || lower == "arm.r") return "RightUpperArm";
    if (lower == "leftforearm" || lower == "leftlowerarm" || lower == "forearm.l") return "LeftLowerArm";
    if (lower == "rightforearm" || lower == "rightlowerarm" || lower == "forearm.r") return "RightLowerArm";
    if (lower == "lefthand" || lower == "left_hand" || lower == "hand.l") return "LeftHand";
    if (lower == "righthand" || lower == "right_hand" || lower == "hand.r") return "RightHand";
    if (lower == "leftleg" || lower == "left_leg" || lower == "leftupperleg" || lower == "leg.l" || lower == "thigh.l") return "LeftUpperLeg";
    if (lower == "rightleg" || lower == "right_leg" || lower == "rightupperleg" || lower == "leg.r" || lower == "thigh.r") return "RightUpperLeg";
    if (lower == "leftfoot" || lower == "left_foot" || lower == "foot.l") return "LeftFoot";
    if (lower == "rightfoot" || lower == "right_foot" || lower == "foot.r") return "RightFoot";
    if (lower == "spine" || lower == "chest" || lower == "torso") return "Spine";
    if (lower == "pelvis" || lower == "hip" || lower == "hips" || lower == "root") return "Pelvis";
    return rawNameOrTag;
}

bool SemanticTagResolver::resolveBoneId(
    const SceneData& scene,
    const std::string& characterId,
    const std::string& semanticTag,
    std::string& outBoneId
) const {
    const std::string normalized = normalizeTag(semanticTag);
    std::vector<RigAuthoring::BoneView> bones;
    std::string error;
    if (!RigAuthoring::listBones(scene, characterId, bones, error)) {
        return false;
    }

    const std::string lowerNorm = toLower(normalized);
    for (const auto& bone : bones) {
        const std::string lowerBone = toLower(bone.name);
        if (lowerBone == lowerNorm || lowerBone.find(lowerNorm) != std::string::npos) {
            outBoneId = bone.name;
            return true;
        }
    }

    if (!bones.empty()) {
        outBoneId = bones[0].name;
        return true;
    }

    return false;
}

bool SemanticTagResolver::resolveChainBoneIds(
    const SceneData& scene,
    const std::string& characterId,
    const std::string& chainSemanticTag,
    std::vector<std::string>& outChainIds
) const {
    outChainIds.clear();
    const std::string normalized = normalizeTag(chainSemanticTag);

    std::string rootBone, midBone, endBone;
    if (normalized == "LeftArm" || normalized == "LeftUpperArm") {
        resolveBoneId(scene, characterId, "LeftUpperArm", rootBone);
        resolveBoneId(scene, characterId, "LeftLowerArm", midBone);
        resolveBoneId(scene, characterId, "LeftHand", endBone);
    } else if (normalized == "RightArm" || normalized == "RightUpperArm") {
        resolveBoneId(scene, characterId, "RightUpperArm", rootBone);
        resolveBoneId(scene, characterId, "RightLowerArm", midBone);
        resolveBoneId(scene, characterId, "RightHand", endBone);
    } else if (normalized == "LeftLeg" || normalized == "LeftUpperLeg") {
        resolveBoneId(scene, characterId, "LeftUpperLeg", rootBone);
        resolveBoneId(scene, characterId, "LeftLowerLeg", midBone);
        resolveBoneId(scene, characterId, "LeftFoot", endBone);
    } else if (normalized == "RightLeg" || normalized == "RightUpperLeg") {
        resolveBoneId(scene, characterId, "RightUpperLeg", rootBone);
        resolveBoneId(scene, characterId, "RightLowerLeg", midBone);
        resolveBoneId(scene, characterId, "RightFoot", endBone);
    }

    if (!rootBone.empty()) outChainIds.push_back(rootBone);
    if (!midBone.empty()) outChainIds.push_back(midBone);
    if (!endBone.empty()) outChainIds.push_back(endBone);

    return !outChainIds.empty();
}

} // namespace RayTrophi
