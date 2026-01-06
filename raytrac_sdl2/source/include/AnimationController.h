#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include "Matrix4x4.h"
#include "Vec3.h"
#include "Quaternion.h"
#include "AssimpLoader.h"

// ============================================================================
// ANIMATION CONTROLLER - Advanced Animation Management System
// ============================================================================
// Features:
// - Animation blending (crossfade between clips)
// - Animation chaining (queue system)
// - Root motion extraction
// - Animation layers
// - Performance optimizations (caching, dirty flags)
// ============================================================================

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
    const AnimationData* sourceData = nullptr;
    
    // Root motion settings
    bool extractRootMotion = false;
    std::string rootBoneName = "Hips"; // Usually hips or root
    
    // Cached frame range (for timeline display)
    int startFrame = 0;
    int endFrame = 0;
    
    AnimationClip() = default;
    AnimationClip(const AnimationData* data);
    
    float getDurationInSeconds() const {
        return (ticksPerSecond > 0) ? (float)(duration / ticksPerSecond) : 0.0f;
    }
};

// ============================================================================
// ANIMATION BLEND NODE - For blending two animations
// ============================================================================
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

// ============================================================================
// ANIMATION QUEUE ITEM - For chaining animations
// ============================================================================
struct AnimationQueueItem {
    std::string clipName;
    float blendInTime = 0.3f;      // Crossfade from previous
    bool waitForEnd = true;         // Wait for current to finish before starting
    int repeatCount = 1;           // Number of times to play (0 = infinite)
    
    // Callbacks
    std::function<void()> onStart;
    std::function<void()> onEnd;
};

// ============================================================================
// ROOT MOTION DATA - Extracted motion for object movement
// ============================================================================
struct RootMotionDelta {
    Vec3 positionDelta = Vec3(0, 0, 0);   // World-space movement this frame
    Quaternion rotationDelta;              // Rotation change this frame
    bool hasPosition = false;
    bool hasRotation = false;
    
    RootMotionDelta() : rotationDelta(1, 0, 0, 0) {}
};

// ============================================================================
// ANIMATION LAYER - For layered animation (upper body, lower body, etc.)
// ============================================================================
struct AnimationLayer {
    std::string name = "Base";
    float weight = 1.0f;
    BlendMode blendMode = BlendMode::Replace;
    
    // Bone mask - which bones this layer affects
    // Empty = all bones, otherwise only listed bones
    std::vector<std::string> affectedBones;
    
    // Current playback state
    AnimationBlendState blendState;
    std::vector<AnimationQueueItem> queue;
};

// ============================================================================
// ANIMATION CONTROLLER - Main manager class
// ============================================================================
class AnimationController {
public:
    // Singleton access
    static AnimationController& getInstance() {
        static AnimationController instance;
        return instance;
    }
    
    // ========================================================================
    // Clip Management
    // ========================================================================
    
    // Register animation clips from scene data
    void registerClips(const std::vector<AnimationData>& animationDataList);
    
    // Get clip by name
    AnimationClip* getClip(const std::string& name);
    const std::vector<AnimationClip>& getAllClips() const { return clips; }
    
    // ========================================================================
    // Playback Control
    // ========================================================================
    
    // Play animation immediately (with optional crossfade)
    void play(const std::string& clipName, float blendTime = 0.3f, int layer = 0);
    
    // Queue animation to play after current finishes
    void queue(const std::string& clipName, float blendTime = 0.3f, int layer = 0);
    void queue(const AnimationQueueItem& item, int layer = 0);
    
    // Stop animation on layer
    void stop(int layer = 0, float blendOutTime = 0.3f);
    void stopAll(float blendOutTime = 0.3f);
    
    // Pause/Resume
    void pause(int layer = 0);
    void resume(int layer = 0);
    void setPaused(bool paused);
    
    // Playback control
    void setTime(float time, int layer = 0);
    void setSpeed(float speed, int layer = 0);
    void setLoop(bool loop, int layer = 0);
    
    // ========================================================================
    // Layer Management
    // ========================================================================
    
    void setLayerWeight(int layer, float weight);
    void setLayerBlendMode(int layer, BlendMode mode);
    void setLayerBoneMask(int layer, const std::vector<std::string>& bones);
    
    // ========================================================================
    // Root Motion
    // ========================================================================
    
    void setRootMotionEnabled(bool enabled, const std::string& rootBone = "Hips");
    RootMotionDelta consumeRootMotion(); // Get and clear accumulated motion
    
    // ========================================================================
    // Update (call each frame)
    // ========================================================================
    
    // Main update function - returns true if animation state changed
    bool update(float deltaTime, const BoneData& boneData);
    
    // Get final bone matrices after update
    const std::vector<Matrix4x4>& getFinalBoneMatrices() const { return cachedFinalBoneMatrices; }
    bool areBoneMatricesDirty() const { return boneMatricesDirty; }
    void clearDirtyFlag() { boneMatricesDirty = false; }
    
    // ========================================================================
    // State Queries
    // ========================================================================
    
    bool isPlaying(int layer = 0) const;
    bool isBlending(int layer = 0) const;
    float getCurrentTime(int layer = 0) const;
    float getNormalizedTime(int layer = 0) const; // 0-1 range
    const std::string& getCurrentClipName(int layer = 0) const;
    
    // ========================================================================
    // Events & Callbacks
    // ========================================================================
    
    using AnimationEventCallback = std::function<void(const std::string& clipName, const std::string& event)>;
    void setEventCallback(AnimationEventCallback callback) { eventCallback = callback; }
    
private:
    AnimationController() = default;
    ~AnimationController() = default;
    AnimationController(const AnimationController&) = delete;
    AnimationController& operator=(const AnimationController&) = delete;
    
    // ========================================================================
    // Internal Methods
    // ========================================================================
    
    void updateLayer(AnimationLayer& layer, float deltaTime, const BoneData& boneData);
    void processQueue(AnimationLayer& layer);
    
    // Calculate transform from animation at given time
    Matrix4x4 calculateNodeTransform(
        const AnimationData* anim,
        float timeInTicks,
        const std::string& nodeName,
        const Matrix4x4& defaultTransform
    ) const;
    
    // Blend two transforms
    Matrix4x4 blendTransforms(
        const Matrix4x4& a,
        const Matrix4x4& b,
        float weight,
        BlendMode mode
    ) const;
    
    // Extract root motion from animation
    void extractRootMotion(
        const AnimationData* anim,
        float prevTime,
        float currentTime,
        const std::string& rootBone
    );
    
    // ========================================================================
    // Data
    // ========================================================================
    
    std::vector<AnimationClip> clips;
    std::map<std::string, size_t> clipNameToIndex;
    
    std::vector<AnimationLayer> layers;
    static constexpr int MAX_LAYERS = 4;
    
    // Root motion
    bool rootMotionEnabled = false;
    std::string rootMotionBone = "Hips";
    RootMotionDelta accumulatedRootMotion;
    Vec3 lastRootPosition;
    Quaternion lastRootRotation;
    
    // Cached bone matrices (performance optimization)
    std::vector<Matrix4x4> cachedFinalBoneMatrices;
    bool boneMatricesDirty = true;
    float lastUpdateTime = -1.0f;
    
    // Playback state
    bool globalPaused = false;
    float globalSpeed = 1.0f;
    
    // Callbacks
    AnimationEventCallback eventCallback;
    
    // Empty string for queries
    static const std::string emptyString;
};

// ============================================================================
// UTILITY: Animation Blending Functions
// ============================================================================
namespace AnimationUtils {
    
    // Smooth step for blend curves
    inline float smoothstep(float t) {
        return t * t * (3.0f - 2.0f * t);
    }
    
    // Exponential decay for natural-feeling blends
    inline float exponentialDecay(float current, float target, float speed, float dt) {
        return current + (target - current) * (1.0f - std::exp(-speed * dt));
    }
    
    // Convert FBX animation time to seconds
    inline float ticksToSeconds(double ticks, double ticksPerSecond) {
        return (ticksPerSecond > 0) ? (float)(ticks / ticksPerSecond) : 0.0f;
    }
    
    // Wrap time for looping animations
    inline float wrapTime(float time, float duration) {
        if (duration <= 0.0f) return 0.0f;
        return fmodf(time, duration);
    }
    
    // Clamp time for non-looping animations
    inline float clampTime(float time, float duration) {
        return std::max(0.0f, std::min(time, duration));
    }
}
