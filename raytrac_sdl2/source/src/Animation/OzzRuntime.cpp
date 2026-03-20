#include "OzzRuntime.h"

#include "AssimpLoader.h"
#include "Matrix4x4.h"

#include "ozz/animation/offline/animation_builder.h"
#include "ozz/animation/offline/raw_animation.h"
#include "ozz/animation/offline/raw_skeleton.h"
#include "ozz/animation/offline/skeleton_builder.h"
#include "ozz/animation/runtime/blending_job.h"
#include "ozz/animation/runtime/local_to_model_job.h"
#include "ozz/animation/runtime/sampling_job.h"
#include "ozz/base/maths/simd_math.h"
#include "ozz/base/maths/quaternion.h"
#include "ozz/base/maths/transform.h"
#include "ozz/base/maths/vec_float.h"

#include <algorithm>
#include <set>
#include <unordered_map>

namespace OzzRuntime {

namespace {

std::string makePrefix(const std::string& importName) {
    if (importName.empty()) {
        return std::string();
    }
    return importName + "_";
}

bool matchesImportPrefix(const std::string& importName, const std::string& boneName) {
    if (importName.empty()) {
        return true;
    }
    const std::string prefix = makePrefix(importName);
    return boneName.rfind(prefix, 0) == 0;
}

bool endsWithNodeName(const std::string& candidate, const std::string& nodeName) {
    if (candidate == nodeName) {
        return true;
    }
    if (candidate.size() <= nodeName.size()) {
        return false;
    }
    const size_t offset = candidate.size() - nodeName.size();
    return candidate[offset - 1] == '_' && candidate.compare(offset, nodeName.size(), nodeName) == 0;
}

Vec3 toVec3(const aiVector3D& value) {
    return Vec3(value.x, value.y, value.z);
}

Quaternion toQuaternion(const aiQuaternion& value) {
    return Quaternion(value.w, value.x, value.y, value.z);
}

float toSeconds(double ticks, double ticksPerSecond) {
    if (ticksPerSecond <= 0.0) {
        return 0.0f;
    }
    return static_cast<float>(ticks / ticksPerSecond);
}

aiMatrix4x4 toAiMatrix(const Matrix4x4& matrix) {
    aiMatrix4x4 out;
    out.a1 = matrix.m[0][0]; out.a2 = matrix.m[0][1]; out.a3 = matrix.m[0][2]; out.a4 = matrix.m[0][3];
    out.b1 = matrix.m[1][0]; out.b2 = matrix.m[1][1]; out.b3 = matrix.m[1][2]; out.b4 = matrix.m[1][3];
    out.c1 = matrix.m[2][0]; out.c2 = matrix.m[2][1]; out.c3 = matrix.m[2][2]; out.c4 = matrix.m[2][3];
    out.d1 = matrix.m[3][0]; out.d2 = matrix.m[3][1]; out.d3 = matrix.m[3][2]; out.d4 = matrix.m[3][3];
    return out;
}

ozz::math::Transform toOzzTransform(const Matrix4x4& matrix) {
    aiVector3D scale;
    aiQuaternion rotation;
    aiVector3D position;
    aiMatrix4x4 aiMatrix = toAiMatrix(matrix);
    aiMatrix.Decompose(scale, rotation, position);

    ozz::math::Transform result = ozz::math::Transform::identity();
    result.translation = ozz::math::Float3(position.x, position.y, position.z);
    result.rotation = ozz::math::Quaternion(rotation.x, rotation.y, rotation.z, rotation.w);
    result.scale = ozz::math::Float3(scale.x, scale.y, scale.z);
    return result;
}

void buildRawSkeletonChildren(
    const std::string& boneName,
    const BoneData& boneData,
    const std::unordered_map<std::string, std::vector<std::string>>& childrenMap,
    ozz::animation::offline::RawSkeleton::Joint::Children* outChildren) {
    auto jointIt = boneData.boneDefaultTransforms.find(boneName);
    if (jointIt == boneData.boneDefaultTransforms.end()) {
        return;
    }

    ozz::animation::offline::RawSkeleton::Joint joint;
    joint.name = boneName.c_str();
    joint.transform = toOzzTransform(jointIt->second);

    auto childIt = childrenMap.find(boneName);
    if (childIt != childrenMap.end()) {
        for (const std::string& childName : childIt->second) {
            buildRawSkeletonChildren(childName, boneData, childrenMap, &joint.children);
        }
    }

    outChildren->push_back(std::move(joint));
}

std::string resolveClipJointName(const std::string& runtimeJointName, const AnimationData& clip) {
    if (clip.positionKeys.find(runtimeJointName) != clip.positionKeys.end() ||
        clip.rotationKeys.find(runtimeJointName) != clip.rotationKeys.end() ||
        clip.scalingKeys.find(runtimeJointName) != clip.scalingKeys.end()) {
        return runtimeJointName;
    }

    std::set<std::string> candidates;
    for (const auto& [name, _] : clip.positionKeys) candidates.insert(name);
    for (const auto& [name, _] : clip.rotationKeys) candidates.insert(name);
    for (const auto& [name, _] : clip.scalingKeys) candidates.insert(name);

    for (const std::string& candidate : candidates) {
        if (endsWithNodeName(runtimeJointName, candidate)) {
            return candidate;
        }
    }
    return std::string();
}

ozz::animation::offline::RawSkeleton buildRawSkeletonForImport(
    const std::string& importName,
    const BoneData& boneData,
    const std::set<std::string>& clipNodeNames,
    std::vector<std::string>* orderedRuntimeJointNames) {
    ozz::animation::offline::RawSkeleton rawSkeleton;
    std::unordered_map<std::string, std::vector<std::string>> childrenMap;
    std::vector<std::string> roots;
    std::set<std::string> includedBones;

    auto shouldIncludeBone = [&](const std::string& boneName) {
        const bool prefixMatch = matchesImportPrefix(importName, boneName);
        const bool clipMatch = std::any_of(clipNodeNames.begin(), clipNodeNames.end(), [&](const std::string& nodeName) {
            return endsWithNodeName(boneName, nodeName);
        });
        return prefixMatch || clipMatch;
    };

    for (const auto& [boneName, _] : boneData.boneDefaultTransforms) {
        if (!shouldIncludeBone(boneName)) {
            continue;
        }
        std::string current = boneName;
        while (!current.empty()) {
            auto insertIt = includedBones.insert(current);
            auto parentIt = boneData.boneParents.find(current);
            if (parentIt == boneData.boneParents.end()) {
                break;
            }
            if (!insertIt.second && parentIt->second == current) {
                break;
            }
            current = parentIt->second;
        }
    }

    for (const auto& [boneName, _] : boneData.boneDefaultTransforms) {
        if (includedBones.find(boneName) == includedBones.end()) {
            continue;
        }

        auto parentIt = boneData.boneParents.find(boneName);
        if (parentIt == boneData.boneParents.end()) {
            roots.push_back(boneName);
            continue;
        }
        if (includedBones.find(parentIt->second) == includedBones.end()) {
            roots.push_back(boneName);
            continue;
        }
        childrenMap[parentIt->second].push_back(boneName);
    }

    std::sort(roots.begin(), roots.end());
    for (auto& [parent, children] : childrenMap) {
        (void)parent;
        std::sort(children.begin(), children.end());
    }

    for (const std::string& rootName : roots) {
        buildRawSkeletonChildren(rootName, boneData, childrenMap, &rawSkeleton.roots);
    }

    orderedRuntimeJointNames->clear();
    ozz::animation::offline::IterateJointsDF(rawSkeleton, [&](const ozz::animation::offline::RawSkeleton::Joint& joint, const ozz::animation::offline::RawSkeleton::Joint*) {
        orderedRuntimeJointNames->push_back(joint.name.c_str());
    });

    return rawSkeleton;
}

ozz::animation::offline::RawAnimation buildRawAnimation(
    const AnimationData& clip,
    const std::vector<std::string>& runtimeJointNames) {
    ozz::animation::offline::RawAnimation rawAnimation;
    rawAnimation.name = clip.name.c_str();
    rawAnimation.duration = (clip.ticksPerSecond > 0.0)
        ? static_cast<float>(clip.duration / clip.ticksPerSecond)
        : 0.0f;
    rawAnimation.tracks.resize(runtimeJointNames.size());

    for (size_t jointIndex = 0; jointIndex < runtimeJointNames.size(); ++jointIndex) {
        const std::string& jointName = runtimeJointNames[jointIndex];
        const std::string clipJointName = resolveClipJointName(jointName, clip);
        auto& track = rawAnimation.tracks[jointIndex];

        auto posIt = clipJointName.empty() ? clip.positionKeys.end() : clip.positionKeys.find(clipJointName);
        if (posIt != clip.positionKeys.end()) {
            for (const auto& key : posIt->second) {
                ozz::animation::offline::RawAnimation::TranslationKey outKey;
                outKey.time = toSeconds(key.mTime, clip.ticksPerSecond);
                outKey.value = ozz::math::Float3(key.mValue.x, key.mValue.y, key.mValue.z);
                track.translations.push_back(outKey);
            }
        }

        auto rotIt = clipJointName.empty() ? clip.rotationKeys.end() : clip.rotationKeys.find(clipJointName);
        if (rotIt != clip.rotationKeys.end()) {
            for (const auto& key : rotIt->second) {
                ozz::animation::offline::RawAnimation::RotationKey outKey;
                outKey.time = toSeconds(key.mTime, clip.ticksPerSecond);
                outKey.value = ozz::math::Quaternion(key.mValue.x, key.mValue.y, key.mValue.z, key.mValue.w);
                track.rotations.push_back(outKey);
            }
        }

        auto scaleIt = clipJointName.empty() ? clip.scalingKeys.end() : clip.scalingKeys.find(clipJointName);
        if (scaleIt != clip.scalingKeys.end()) {
            for (const auto& key : scaleIt->second) {
                ozz::animation::offline::RawAnimation::ScaleKey outKey;
                outKey.time = toSeconds(key.mTime, clip.ticksPerSecond);
                outKey.value = ozz::math::Float3(key.mValue.x, key.mValue.y, key.mValue.z);
                track.scales.push_back(outKey);
            }
        }
    }

    return rawAnimation;
}

void sortTranslationKeys(std::vector<TranslationKey>& keys) {
    std::sort(keys.begin(), keys.end(), [](const TranslationKey& a, const TranslationKey& b) {
        return a.timeSeconds < b.timeSeconds;
    });
}

void sortRotationKeys(std::vector<RotationKey>& keys) {
    std::sort(keys.begin(), keys.end(), [](const RotationKey& a, const RotationKey& b) {
        return a.timeSeconds < b.timeSeconds;
    });
}

void sortScaleKeys(std::vector<ScaleKey>& keys) {
    std::sort(keys.begin(), keys.end(), [](const ScaleKey& a, const ScaleKey& b) {
        return a.timeSeconds < b.timeSeconds;
    });
}

Matrix4x4 toMatrix4x4(const ozz::math::Float4x4& matrix) {
    alignas(16) float cols[16];
    ozz::math::StorePtrU(matrix.cols[0], cols + 0);
    ozz::math::StorePtrU(matrix.cols[1], cols + 4);
    ozz::math::StorePtrU(matrix.cols[2], cols + 8);
    ozz::math::StorePtrU(matrix.cols[3], cols + 12);

    return Matrix4x4(
        cols[0], cols[4], cols[8],  cols[12],
        cols[1], cols[5], cols[9],  cols[13],
        cols[2], cols[6], cols[10], cols[14],
        cols[3], cols[7], cols[11], cols[15]);
}

float computeNormalizedRatio(const ozz::animation::Animation& animation, float timeSeconds) {
    const float duration = animation.duration();
    if (duration <= 0.0f) {
        return 0.0f;
    }
    float ratio = std::fmod(timeSeconds, duration) / duration;
    if (ratio < 0.0f) {
        ratio += 1.0f;
    }
    return ratio;
}

bool sampleAnimationToLocalPose(
    const AnimationSet& animationSet,
    size_t clipIndex,
    float timeSeconds,
    ozz::span<ozz::math::SoaTransform> outLocalPose) {
    if (!animationSet.runtimeSkeleton || clipIndex >= animationSet.runtimeAnimations.size()) {
        return false;
    }

    const auto& animation = animationSet.runtimeAnimations[clipIndex];
    if (!animation) {
        return false;
    }

    auto& mutableSet = const_cast<AnimationSet&>(animationSet);
    mutableSet.samplingContext.Resize(animationSet.runtimeSkeleton->num_joints());

    ozz::animation::SamplingJob samplingJob;
    samplingJob.animation = animation.get();
    samplingJob.context = &mutableSet.samplingContext;
    samplingJob.ratio = computeNormalizedRatio(*animation, timeSeconds);
    samplingJob.output = outLocalPose;
    return samplingJob.Run();
}

bool localPoseToModelMatrices(
    const AnimationSet& animationSet,
    ozz::span<const ozz::math::SoaTransform> localPose,
    std::vector<Matrix4x4>* outModelMatrices) {
    if (!outModelMatrices || !animationSet.runtimeSkeleton) {
        return false;
    }

    auto& mutableSet = const_cast<AnimationSet&>(animationSet);
    if (mutableSet.modelMatrices.size() < static_cast<size_t>(animationSet.runtimeSkeleton->num_joints())) {
        mutableSet.modelMatrices.resize(animationSet.runtimeSkeleton->num_joints());
    }

    ozz::animation::LocalToModelJob ltmJob;
    ltmJob.skeleton = animationSet.runtimeSkeleton.get();
    ltmJob.input = localPose;
    ltmJob.output = ozz::span<ozz::math::Float4x4>(mutableSet.modelMatrices.data(), mutableSet.modelMatrices.size());
    if (!ltmJob.Run()) {
        return false;
    }

    outModelMatrices->resize(mutableSet.modelMatrices.size(), Matrix4x4::identity());
    for (size_t i = 0; i < mutableSet.modelMatrices.size(); ++i) {
        (*outModelMatrices)[i] = toMatrix4x4(mutableSet.modelMatrices[i]);
    }
    return true;
}

} // namespace

bool isCompiledIn() {
#if defined(RAYTROPHI_ENABLE_OZZ)
    return true;
#else
    return false;
#endif
}

const char* backendLabel() {
#if defined(RAYTROPHI_ENABLE_OZZ)
    return "Ozz";
#else
    return "Stub";
#endif
}

std::shared_ptr<AnimationSet> buildStubAnimationSet(
    const std::string& importName,
    const BoneData& boneData,
    const std::vector<std::shared_ptr<AnimationData>>& clips) {
    auto runtime = std::make_shared<AnimationSet>();
    runtime->state = isCompiledIn() ? IntegrationState::StubReady : IntegrationState::Disabled;
    runtime->sourceImportName = importName;

    if (boneData.boneNameToIndex.empty()) {
        return runtime;
    }

    std::vector<std::string> runtimeJointNames;
    std::set<std::string> clipNodeNames;
    for (const auto& clip : clips) {
        if (!clip) {
            continue;
        }
        for (const auto& [name, _] : clip->positionKeys) clipNodeNames.insert(name);
        for (const auto& [name, _] : clip->rotationKeys) clipNodeNames.insert(name);
        for (const auto& [name, _] : clip->scalingKeys) clipNodeNames.insert(name);
    }

    auto rawSkeleton = buildRawSkeletonForImport(importName, boneData, clipNodeNames, &runtimeJointNames);
    runtime->skeleton.jointCount = runtimeJointNames.size();
    runtime->skeleton.jointNames = runtimeJointNames;
    runtime->skeleton.parents.assign(runtimeJointNames.size(), -1);
    runtime->skeleton.skinnedBoneCount = 0;
    runtime->sceneBoneToRuntimeJoint.assign(boneData.boneIndexToName.size(), -1);
    runtime->bindPoseMatrices.assign(runtimeJointNames.size(), Matrix4x4::identity());

    std::unordered_map<std::string, int> runtimeIndexByName;
    for (size_t i = 0; i < runtimeJointNames.size(); ++i) {
        runtimeIndexByName[runtimeJointNames[i]] = static_cast<int>(i);

        auto boneIt = boneData.boneNameToIndex.find(runtimeJointNames[i]);
        if (boneIt != boneData.boneNameToIndex.end()) {
            if (boneIt->second < runtime->sceneBoneToRuntimeJoint.size()) {
                runtime->sceneBoneToRuntimeJoint[boneIt->second] = static_cast<int>(i);
            }
            if (boneData.weightedBoneNames.find(runtimeJointNames[i]) != boneData.weightedBoneNames.end()) {
                ++runtime->skeleton.skinnedBoneCount;
            }
        }

        auto bindIt = boneData.boneDefaultTransforms.find(runtimeJointNames[i]);
        if (bindIt != boneData.boneDefaultTransforms.end()) {
            runtime->bindPoseMatrices[i] = bindIt->second;
        }
    }

    for (size_t i = 0; i < runtimeJointNames.size(); ++i) {
        auto parentIt = boneData.boneParents.find(runtimeJointNames[i]);
        if (parentIt == boneData.boneParents.end()) {
            continue;
        }
        auto runtimeParentIt = runtimeIndexByName.find(parentIt->second);
        if (runtimeParentIt != runtimeIndexByName.end()) {
            runtime->skeleton.parents[i] = runtimeParentIt->second;
        }
    }

    if (rawSkeleton.Validate()) {
        ozz::animation::offline::SkeletonBuilder skeletonBuilder;
        runtime->runtimeSkeleton = skeletonBuilder(rawSkeleton);
    }

    runtime->clips.reserve(clips.size());
    runtime->clipRuntimes.reserve(clips.size());
    runtime->runtimeAnimations.reserve(clips.size());
    for (const auto& clip : clips) {
        if (!clip) {
            continue;
        }

        ClipInfo info;
        info.name = clip->name;
        info.durationSeconds = (clip->ticksPerSecond > 0.0)
            ? static_cast<float>(clip->duration / clip->ticksPerSecond)
            : 0.0f;
        info.ticksPerSecond = static_cast<float>(clip->ticksPerSecond);
        runtime->clips.push_back(info);

        ClipRuntime clipRuntime;
        clipRuntime.info = info;
        clipRuntime.jointTracks.reserve(runtime->skeleton.jointNames.size());

        for (size_t jointIndex = 0; jointIndex < runtime->skeleton.jointNames.size(); ++jointIndex) {
            JointTrack track;
            track.jointIndex = static_cast<int>(jointIndex);
            track.jointName = runtime->skeleton.jointNames[jointIndex];

            if (jointIndex < runtime->bindPoseMatrices.size()) {
                track.defaultLocalTransform = runtime->bindPoseMatrices[jointIndex];
            }

        const std::string clipJointName = resolveClipJointName(track.jointName, *clip);

        auto posIt = clipJointName.empty() ? clip->positionKeys.end() : clip->positionKeys.find(clipJointName);
        if (posIt != clip->positionKeys.end()) {
                track.translations.reserve(posIt->second.size());
                for (const auto& key : posIt->second) {
                    track.translations.push_back({toSeconds(key.mTime, clip->ticksPerSecond), toVec3(key.mValue)});
                }
                sortTranslationKeys(track.translations);
            }

        auto rotIt = clipJointName.empty() ? clip->rotationKeys.end() : clip->rotationKeys.find(clipJointName);
            if (rotIt != clip->rotationKeys.end()) {
                track.rotations.reserve(rotIt->second.size());
                for (const auto& key : rotIt->second) {
                    track.rotations.push_back({toSeconds(key.mTime, clip->ticksPerSecond), toQuaternion(key.mValue)});
                }
                sortRotationKeys(track.rotations);
            }

        auto scaleIt = clipJointName.empty() ? clip->scalingKeys.end() : clip->scalingKeys.find(clipJointName);
            if (scaleIt != clip->scalingKeys.end()) {
                track.scales.reserve(scaleIt->second.size());
                for (const auto& key : scaleIt->second) {
                    track.scales.push_back({toSeconds(key.mTime, clip->ticksPerSecond), toVec3(key.mValue)});
                }
                sortScaleKeys(track.scales);
            }

            clipRuntime.jointTracks.push_back(std::move(track));
        }

        runtime->clipRuntimes.push_back(std::move(clipRuntime));

        if (runtime->runtimeSkeleton) {
            auto rawAnimation = buildRawAnimation(*clip, runtimeJointNames);
            if (rawAnimation.Validate()) {
                ozz::animation::offline::AnimationBuilder animationBuilder;
                animationBuilder.iframe_interval = 0.1f;
                runtime->runtimeAnimations.push_back(animationBuilder(rawAnimation));
            } else {
                runtime->runtimeAnimations.push_back(nullptr);
            }
        }
    }

    if (runtime->skeleton.jointCount == 0 && !clipNodeNames.empty()) {
        runtime->state = isCompiledIn() ? IntegrationState::StubReady : IntegrationState::Disabled;
    }

    if (runtime->runtimeSkeleton && !runtime->runtimeAnimations.empty()) {
        runtime->state = IntegrationState::Ready;
        runtime->samplingContext.Resize(runtime->runtimeSkeleton->num_joints());
        runtime->localSoaPose.resize(runtime->runtimeSkeleton->num_soa_joints(), ozz::math::SoaTransform::identity());
        runtime->modelMatrices.resize(runtime->runtimeSkeleton->num_joints());
    }

    return runtime;
}

bool sampleAnimationToModelMatrices(
    const AnimationSet& animationSet,
    size_t clipIndex,
    float timeSeconds,
    std::vector<Matrix4x4>* outModelMatrices) {
    if (!outModelMatrices || !animationSet.runtimeSkeleton) {
        return false;
    }
    if (clipIndex >= animationSet.runtimeAnimations.size()) {
        return false;
    }

    auto& mutableSet = const_cast<AnimationSet&>(animationSet);
    if (mutableSet.localSoaPose.size() < static_cast<size_t>(animationSet.runtimeSkeleton->num_soa_joints())) {
        mutableSet.localSoaPose.resize(animationSet.runtimeSkeleton->num_soa_joints(), ozz::math::SoaTransform::identity());
    }
    if (!sampleAnimationToLocalPose(animationSet, clipIndex, timeSeconds,
        ozz::span<ozz::math::SoaTransform>(mutableSet.localSoaPose.data(), mutableSet.localSoaPose.size()))) {
        return false;
    }

    if (!localPoseToModelMatrices(animationSet,
        ozz::span<const ozz::math::SoaTransform>(mutableSet.localSoaPose.data(), mutableSet.localSoaPose.size()),
        outModelMatrices)) {
        return false;
    }

    return true;
}

bool sampleBlendedAnimationToModelMatrices(
    const AnimationSet& animationSet,
    const std::vector<BlendLayerInput>& layers,
    std::vector<Matrix4x4>* outModelMatrices) {
    if (!outModelMatrices || !animationSet.runtimeSkeleton || layers.empty()) {
        return false;
    }

    const size_t numSoaJoints = static_cast<size_t>(animationSet.runtimeSkeleton->num_soa_joints());
    if (numSoaJoints == 0) {
        return false;
    }

    std::vector<std::vector<ozz::math::SoaTransform>> sampledLayerPoses;
    sampledLayerPoses.reserve(layers.size());
    std::vector<ozz::animation::BlendingJob::Layer> blendLayers;
    blendLayers.reserve(layers.size());

    for (const BlendLayerInput& layerInput : layers) {
        if (layerInput.layerWeight <= 0.0f || layerInput.clipIndexA < 0) {
            continue;
        }
        if (layerInput.blendMode != BlendMode::Replace) {
            return false;
        }

        sampledLayerPoses.emplace_back(numSoaJoints, ozz::math::SoaTransform::identity());
        std::vector<ozz::math::SoaTransform>& finalLayerPose = sampledLayerPoses.back();

        const bool hasBlendTarget = layerInput.clipIndexB >= 0 && layerInput.blendWeight > 0.0f;
        if (hasBlendTarget) {
            std::vector<ozz::math::SoaTransform> poseA(numSoaJoints, ozz::math::SoaTransform::identity());
            std::vector<ozz::math::SoaTransform> poseB(numSoaJoints, ozz::math::SoaTransform::identity());
            if (!sampleAnimationToLocalPose(animationSet, static_cast<size_t>(layerInput.clipIndexA), layerInput.timeA,
                ozz::span<ozz::math::SoaTransform>(poseA.data(), poseA.size()))) {
                return false;
            }
            if (!sampleAnimationToLocalPose(animationSet, static_cast<size_t>(layerInput.clipIndexB), layerInput.timeB,
                ozz::span<ozz::math::SoaTransform>(poseB.data(), poseB.size()))) {
                return false;
            }

            ozz::animation::BlendingJob::Layer crossfadeLayers[2];
            crossfadeLayers[0].weight = std::max(0.0f, 1.0f - layerInput.blendWeight);
            crossfadeLayers[0].transform = ozz::span<const ozz::math::SoaTransform>(poseA.data(), poseA.size());
            crossfadeLayers[1].weight = std::max(0.0f, layerInput.blendWeight);
            crossfadeLayers[1].transform = ozz::span<const ozz::math::SoaTransform>(poseB.data(), poseB.size());

            ozz::animation::BlendingJob crossfadeJob;
            crossfadeJob.threshold = 0.1f;
            crossfadeJob.layers = ozz::span<const ozz::animation::BlendingJob::Layer>(crossfadeLayers, 2);
            crossfadeJob.rest_pose = animationSet.runtimeSkeleton->joint_rest_poses();
            crossfadeJob.output = ozz::span<ozz::math::SoaTransform>(finalLayerPose.data(), finalLayerPose.size());
            if (!crossfadeJob.Run()) {
                return false;
            }
        } else {
            if (!sampleAnimationToLocalPose(animationSet, static_cast<size_t>(layerInput.clipIndexA), layerInput.timeA,
                ozz::span<ozz::math::SoaTransform>(finalLayerPose.data(), finalLayerPose.size()))) {
                return false;
            }
        }

        ozz::animation::BlendingJob::Layer layer;
        layer.weight = layerInput.layerWeight;
        layer.transform = ozz::span<const ozz::math::SoaTransform>(finalLayerPose.data(), finalLayerPose.size());
        blendLayers.push_back(layer);
    }

    if (blendLayers.empty()) {
        return false;
    }

    auto& mutableSet = const_cast<AnimationSet&>(animationSet);
    if (mutableSet.localSoaPose.size() < numSoaJoints) {
        mutableSet.localSoaPose.resize(numSoaJoints, ozz::math::SoaTransform::identity());
    }

    ozz::animation::BlendingJob blendJob;
    blendJob.threshold = 0.1f;
    blendJob.layers = ozz::span<const ozz::animation::BlendingJob::Layer>(blendLayers.data(), blendLayers.size());
    blendJob.rest_pose = animationSet.runtimeSkeleton->joint_rest_poses();
    blendJob.output = ozz::span<ozz::math::SoaTransform>(mutableSet.localSoaPose.data(), mutableSet.localSoaPose.size());
    if (!blendJob.Run()) {
        return false;
    }

    return localPoseToModelMatrices(animationSet,
        ozz::span<const ozz::math::SoaTransform>(mutableSet.localSoaPose.data(), mutableSet.localSoaPose.size()),
        outModelMatrices);
}

} // namespace OzzRuntime
