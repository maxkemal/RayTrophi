/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          AnimationController.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include <unordered_map>
#include "Matrix4x4.h"
#include "Vec3.h"
#include "Quaternion.h"
#include "AssimpLoader.h"

// Forward declarations
struct AnimationData;
struct BoneData;

// ============================================================================
// ANIMATION CLIP - Wrapper around AnimationData with playback state
// ============================================================================
struct AnimationClip {
    std::string name;
    float duration = 0.0f;           // In seconds
    float ticksPerSecond = 24.0f;
    bool loop = true;
    
    // Source animation data (reference)
    std::shared_ptr<AnimationData> sourceData = nullptr;
    
    // Root motion settings
    bool extractRootMotion = false;
    std::string rootBoneName = "Hips"; // Usually hips or root
    
    // Cached frame range (for timeline display)
    int startFrame = 0;
    int endFrame = 0;
    
    AnimationClip() = default;
    AnimationClip(std::shared_ptr<AnimationData> data);
    
    float getDurationInSeconds() const {
        return (ticksPerSecond > 0) ? (float)(duration / ticksPerSecond) : 0.0f;
    }
};

enum class BlendMode {
    Replace,      // B replaces A completely
    Additive,     // B is added on top of A
    Override      // B overrides A where B has keys
};

struct AnimationBlendState {
    const AnimationClip* clipA = nullptr;
    const AnimationClip* clipB = nullptr;
    
    float timeA = 0.0f;            // Playback time in clip A
    float timeB = 0.0f;            // Playback time in clip B
    float blendWeight = 0.0f;      // 0 = 100% A, 1 = 100% B
    float blendDuration = 0.3f;    // Crossfade duration in seconds
    
    BlendMode mode = BlendMode::Replace;
    
    bool isBlending() const { return blendWeight > 0.0f && blendWeight < 1.0f; }
};

struct AnimationQueueItem {
    std::string clipName;
    float blendInTime = 0.3f;      // Crossfade from previous
    bool waitForEnd = true;         // Wait for current to finish before starting
    int repeatCount = 1;           // Number of times to play (0 = infinite)
    
    // Callbacks
    std::function<void()> onStart;
    std::function<void()> onEnd;
};

struct RootMotionDelta {
    Vec3 positionDelta = Vec3(0,0,0);
    Quaternion rotationDelta;
    bool hasPosition = false;
    bool hasRotation = false;
    
    RootMotionDelta() : rotationDelta(1,0,0,0) {}
};

struct AnimationLayer {
    std::string name = "Base";
    float weight = 1.0f;
    BlendMode blendMode = BlendMode::Replace;
    std::vector<std::string> affectedBones;
    AnimationBlendState blendState;
    std::vector<AnimationQueueItem> queue;
};

class AnimationController {
public:
    AnimationController();
    ~AnimationController() = default;
    
    AnimationController(const AnimationController&) = default;
    AnimationController& operator=(const AnimationController&) = default;
    
    static AnimationController& getInstance() {
        static AnimationController instance;
        return instance;
    }

    // Full reset - clears all cached bone matrices, clips, layers, etc.
    void clear();
    
    void registerClips(const std::vector<std::shared_ptr<AnimationData>>& animationDataList);
    AnimationClip* getClip(const std::string& name);
    const std::vector<AnimationClip>& getAllClips() const { return clips; }
    
    void play(const std::string& clipName, float blendTime = 0.3f, int layer = 0);
    void queue(const std::string& clipName, float blendTime = 0.3f, int layer = 0);
    void queue(const AnimationQueueItem& item, int layer = 0);
    void stop(int layer = 0, float blendOutTime = 0.3f);
    void stopAll(float blendOutTime = 0.3f);
    
    void pause(int layer = 0);
    void resume(int layer = 0);
    void setPaused(bool paused);
    void setTime(float time, int layer = 0);
    void setSpeed(float speed, int layer = 0);
    void setLoop(bool loop, int layer = 0);
    
    void setLayerWeight(int layer, float weight);
    void setLayerBlendMode(int layer, BlendMode mode);
    void setLayerBoneMask(int layer, const std::vector<std::string>& bones);
    
    void setRootMotionEnabled(bool enabled, const std::string& rootBone = "Hips");
    RootMotionDelta consumeRootMotion();
    std::string findBestRootMotionBone(const std::string& clipName);
    
    bool update(float deltaTime, const BoneData& boneData);
    
    const std::vector<Matrix4x4>& getFinalBoneMatrices() const { return cachedFinalBoneMatrices; }
    bool areBoneMatricesDirty() const { return boneMatricesDirty; }
    void clearDirtyFlag() { boneMatricesDirty = false; }
    
    bool isPlaying(int layer = 0) const;
    bool isPaused() const { return globalPaused; }
    bool isBlending(int layer = 0) const;
    float getCurrentTime(int layer = 0) const;
    float getNormalizedTime(int layer = 0) const;
    const std::string& getCurrentClipName(int layer = 0) const;
    
    using AnimationEventCallback = std::function<void(const std::string& clipName, const std::string& event)>;
    void setEventCallback(AnimationEventCallback callback) { eventCallback = callback; }

private:
    void updateLayer(AnimationLayer& layer, float deltaTime, const BoneData& boneData);
    void processQueue(AnimationLayer& layer);
    Matrix4x4 calculateNodeTransform(std::shared_ptr<AnimationData> anim, float timeInTicks, const std::string& nodeName, const Matrix4x4& defaultTransform, bool wrap = true) const;
    Matrix4x4 getAnimatedGlobalTransform(const std::string& boneName, const BoneData& boneData, std::shared_ptr<AnimationData> animA, float timeA, std::shared_ptr<AnimationData> animB, float timeB, float blendWeight, BlendMode mode, std::unordered_map<std::string, Matrix4x4>& cache) const;
    Matrix4x4 blendTransforms(const Matrix4x4& a, const Matrix4x4& b, float weight, BlendMode mode) const;
    void extractRootMotion(std::shared_ptr<AnimationData> anim, float prevTime, float currentTime, const std::string& rootBone);
    
    std::vector<AnimationClip> clips;
    std::map<std::string, size_t> clipNameToIndex;
    std::vector<AnimationLayer> layers;
    static constexpr int MAX_LAYERS = 4;
    
    bool rootMotionEnabled = false;
    std::string rootMotionBone = "Hips";
    RootMotionDelta accumulatedRootMotion;
    Vec3 lastRootPosition;
    Quaternion lastRootRotation;
    
    std::vector<Matrix4x4> cachedFinalBoneMatrices;
    bool boneMatricesDirty = true;
    float lastUpdateTime = -1.0f;
    bool globalPaused = false;
    float globalSpeed = 1.0f;
    AnimationEventCallback eventCallback;
    static const std::string emptyString;
};

namespace AnimationUtils {
    inline float smoothstep(float t) { return t * t * (3.0f - 2.0f * t); }
    inline float exponentialDecay(float current, float target, float speed, float dt) { return current + (target - current) * (1.0f - std::exp(-speed * dt)); }
    inline float ticksToSeconds(double ticks, double ticksPerSecond) { return (ticksPerSecond > 0) ? (float)(ticks / ticksPerSecond) : 0.0f; }
    inline float wrapTime(float time, float duration) { if (duration <= 0.0f) return 0.0f; return fmodf(time, duration); }
    inline float clampTime(float time, float duration) { return std::max(0.0f, std::min(time, duration)); }
}
