#pragma once
#include <string>
#include <vector>
#include <memory>
#include <cstdint>

namespace RayTrophi {

enum class AnimationSourceType {
    ImportedClip,
    MotionRecipe,
    PoseSequence,
    ProceduralMotion,
    BakedAnimation
};

struct ContactMarkers {
    bool leftFoot = false;
    bool rightFoot = false;
    bool leftHand = false;
    bool rightHand = false;
    float leftFootPhase = 0.0f;
    float rightFootPhase = 0.0f;
};

struct AnimationSourceMetadata {
    std::string id;
    std::string name;
    AnimationSourceType type = AnimationSourceType::MotionRecipe;
    
    float durationSeconds = 1.0f;
    float frameRate = 30.0f;
    bool isLoopable = true;
    bool hasRootMotion = false;
    
    std::string skeletonSignature;
    std::vector<std::string> tags;
    ContactMarkers contactMarkers;
};

class IAnimationSource {
public:
    virtual ~IAnimationSource() = default;

    virtual const AnimationSourceMetadata& getMetadata() const = 0;
    virtual bool evaluate(float timeSeconds, uint32_t boneMask, void* outPoseData) = 0;
    virtual bool isNull() const { return false; }
};

using AnimationSourcePtr = std::shared_ptr<IAnimationSource>;

} // namespace RayTrophi
