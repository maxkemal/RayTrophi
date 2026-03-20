#pragma once

#include "Matrix4x4.h"
#include "Quaternion.h"
#include "Vec3.h"
#include "AnimationController.h"

#include "ozz/animation/runtime/animation.h"
#include "ozz/animation/runtime/blending_job.h"
#include "ozz/animation/runtime/local_to_model_job.h"
#include "ozz/animation/runtime/sampling_job.h"
#include "ozz/animation/runtime/skeleton.h"
#include "ozz/base/maths/soa_float4x4.h"
#include "ozz/base/maths/soa_transform.h"
#include "ozz/base/memory/unique_ptr.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

struct AnimationData;
struct BoneData;

namespace OzzRuntime {

enum class IntegrationState {
    Disabled,
    StubReady,
    Ready
};

struct ClipInfo {
    std::string name;
    float durationSeconds = 0.0f;
    float ticksPerSecond = 0.0f;
};

struct TranslationKey {
    float timeSeconds = 0.0f;
    Vec3 value = Vec3(0.0f, 0.0f, 0.0f);
};

struct RotationKey {
    float timeSeconds = 0.0f;
    Quaternion value = Quaternion(1.0f, 0.0f, 0.0f, 0.0f);
};

struct ScaleKey {
    float timeSeconds = 0.0f;
    Vec3 value = Vec3(1.0f, 1.0f, 1.0f);
};

struct JointTrack {
    int jointIndex = -1;
    std::string jointName;
    Matrix4x4 defaultLocalTransform = Matrix4x4::identity();
    std::vector<TranslationKey> translations;
    std::vector<RotationKey> rotations;
    std::vector<ScaleKey> scales;
};

struct ClipRuntime {
    ClipInfo info;
    std::vector<JointTrack> jointTracks;
};

struct SkeletonInfo {
    size_t jointCount = 0;
    size_t skinnedBoneCount = 0;
    std::vector<std::string> jointNames;
    std::vector<int> parents;
};

struct AnimationSet {
    IntegrationState state = IntegrationState::Disabled;
    SkeletonInfo skeleton;
    std::vector<ClipInfo> clips;
    std::vector<ClipRuntime> clipRuntimes;
    std::vector<int> sceneBoneToRuntimeJoint;
    std::vector<Matrix4x4> bindPoseMatrices;
    std::string sourceImportName;
    ozz::unique_ptr<ozz::animation::Skeleton> runtimeSkeleton;
    std::vector<ozz::unique_ptr<ozz::animation::Animation>> runtimeAnimations;
    mutable ozz::animation::SamplingJob::Context samplingContext;
    mutable std::vector<ozz::math::SoaTransform> localSoaPose;
    mutable std::vector<ozz::math::Float4x4> modelMatrices;

    bool isUsable() const {
        return state == IntegrationState::Ready && runtimeSkeleton != nullptr;
    }

    bool hasScaffoldData() const {
        return state != IntegrationState::Disabled;
    }
};

struct BlendLayerInput {
    int clipIndexA = -1;
    int clipIndexB = -1;
    float timeA = 0.0f;
    float timeB = 0.0f;
    float blendWeight = 0.0f;
    float layerWeight = 1.0f;
    BlendMode blendMode = BlendMode::Replace;
};

bool isCompiledIn();
const char* backendLabel();

std::shared_ptr<AnimationSet> buildStubAnimationSet(
    const std::string& importName,
    const BoneData& boneData,
    const std::vector<std::shared_ptr<AnimationData>>& clips);

bool sampleAnimationToModelMatrices(
    const AnimationSet& animationSet,
    size_t clipIndex,
    float timeSeconds,
    std::vector<Matrix4x4>* outModelMatrices);

bool sampleBlendedAnimationToModelMatrices(
    const AnimationSet& animationSet,
    const std::vector<BlendLayerInput>& layers,
    std::vector<Matrix4x4>* outModelMatrices);

} // namespace OzzRuntime
