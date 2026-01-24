#include "AnimationController.h"
#include "globals.h"
#include <cmath>
#include <algorithm>

// Static member initialization
const std::string AnimationController::emptyString = "";

// ============================================================================
// ANIMATION CLIP Implementation
// ============================================================================

AnimationClip::AnimationClip(const AnimationData* data) : sourceData(data) {
    if (data) {
        name = data->name;
        duration = (float)data->duration;
        ticksPerSecond = (float)(data->ticksPerSecond > 0 ? data->ticksPerSecond : 24.0);
        startFrame = data->startFrame;
        endFrame = data->endFrame;
    }
}

// ============================================================================
// ANIMATION CONTROLLER Implementation
// ============================================================================

void AnimationController::registerClips(const std::vector<AnimationData>& animationDataList) {
    clips.clear();
    clipNameToIndex.clear();
    
    for (size_t i = 0; i < animationDataList.size(); ++i) {
        AnimationClip clip(&animationDataList[i]);
        clipNameToIndex[clip.name] = clips.size();
        clips.push_back(clip);
        
        SCENE_LOG_INFO("[AnimController] Registered clip: " + clip.name + 
                      " (duration: " + std::to_string(clip.getDurationInSeconds()) + "s)");
    }
    
    // Initialize default layer if empty
    if (layers.empty()) {
        layers.resize(1);
        layers[0].name = "Base";
    }
    
    boneMatricesDirty = true;
}

AnimationClip* AnimationController::getClip(const std::string& name) {
    auto it = clipNameToIndex.find(name);
    if (it != clipNameToIndex.end() && it->second < clips.size()) {
        return &clips[it->second];
    }
    return nullptr;
}

// ========================================================================
// Playback Control
// ========================================================================

void AnimationController::play(const std::string& clipName, float blendTime, int layer) {
    // Ensure layer exists
    while (layers.size() <= static_cast<size_t>(layer)) {
        AnimationLayer newLayer;
        newLayer.name = "Layer_" + std::to_string(layers.size());
        layers.push_back(newLayer);
    }
    
    AnimationClip* clip = getClip(clipName);
    if (!clip) {
        SCENE_LOG_WARN("[AnimController] Clip not found: " + clipName);
        return;
    }
    
    AnimationLayer& layerRef = layers[layer];
    
    // Setup blend state
    if (blendTime > 0.0f && layerRef.blendState.clipA != nullptr) {
        // Crossfade from current (A) to new (B)
        layerRef.blendState.clipB = clip;
        layerRef.blendState.timeB = 0.0f;
        layerRef.blendState.blendWeight = 0.0f;
        layerRef.blendState.blendDuration = blendTime;
    } else {
        // Instant switch
        layerRef.blendState.clipA = clip;
        layerRef.blendState.timeA = 0.0f;
        layerRef.blendState.clipB = nullptr;
        layerRef.blendState.blendWeight = 0.0f;
    }
    
    // Clear queue when starting new animation directly
    layerRef.queue.clear();
    
    boneMatricesDirty = true;
    
    SCENE_LOG_INFO("[AnimController] Playing: " + clipName + " on layer " + std::to_string(layer));
}

void AnimationController::queue(const std::string& clipName, float blendTime, int layer) {
    AnimationQueueItem item;
    item.clipName = clipName;
    item.blendInTime = blendTime;
    item.waitForEnd = true;
    item.repeatCount = 1;
    
    queue(item, layer);
}

void AnimationController::queue(const AnimationQueueItem& item, int layer) {
    while (layers.size() <= static_cast<size_t>(layer)) {
        AnimationLayer newLayer;
        newLayer.name = "Layer_" + std::to_string(layers.size());
        layers.push_back(newLayer);
    }
    
    layers[layer].queue.push_back(item);
    SCENE_LOG_INFO("[AnimController] Queued: " + item.clipName);
}

void AnimationController::stop(int layer, float blendOutTime) {
    if (layer >= static_cast<int>(layers.size())) return;
    
    // TODO: Implement blend out to bind pose
    layers[layer].blendState = AnimationBlendState();
    layers[layer].queue.clear();
    boneMatricesDirty = true;
}

void AnimationController::stopAll(float blendOutTime) {
    for (size_t i = 0; i < layers.size(); ++i) {
        stop(static_cast<int>(i), blendOutTime);
    }
}

void AnimationController::pause(int layer) {
    // Per-layer pause could be implemented with a flag
    // For now, use global pause
    globalPaused = true;
}

void AnimationController::resume(int layer) {
    globalPaused = false;
}

void AnimationController::setPaused(bool paused) {
    globalPaused = paused;
}

void AnimationController::setTime(float time, int layer) {
    if (layer >= static_cast<int>(layers.size())) return;
    
    layers[layer].blendState.timeA = time;
    boneMatricesDirty = true;
}

void AnimationController::setSpeed(float speed, int layer) {
    globalSpeed = speed;
}

void AnimationController::setLoop(bool loop, int layer) {
    if (layer >= static_cast<int>(layers.size())) return;
    
    auto& state = layers[layer].blendState;
    if (state.clipA) {
        // Note: Clips are const, so we'd need non-const access to modify loop
        // For now, loop is per-clip setting
    }
}

// ========================================================================
// Layer Management
// ========================================================================

void AnimationController::setLayerWeight(int layer, float weight) {
    if (layer >= static_cast<int>(layers.size())) return;
    layers[layer].weight = std::max(0.0f, std::min(1.0f, weight));
}

void AnimationController::setLayerBlendMode(int layer, BlendMode mode) {
    if (layer >= static_cast<int>(layers.size())) return;
    layers[layer].blendMode = mode;
}

void AnimationController::setLayerBoneMask(int layer, const std::vector<std::string>& bones) {
    if (layer >= static_cast<int>(layers.size())) return;
    layers[layer].affectedBones = bones;
}

// ========================================================================
// Root Motion
// ========================================================================

void AnimationController::setRootMotionEnabled(bool enabled, const std::string& rootBone) {
    rootMotionEnabled = enabled;
    rootMotionBone = rootBone;
    accumulatedRootMotion = RootMotionDelta();
}

RootMotionDelta AnimationController::consumeRootMotion() {
    RootMotionDelta result = accumulatedRootMotion;
    accumulatedRootMotion = RootMotionDelta();
    return result;
}

// ========================================================================
// Main Update
// ========================================================================

bool AnimationController::update(float deltaTime, const BoneData& boneData) {
    if (globalPaused || deltaTime <= 0.0f) {
        return false;
    }
    
    deltaTime *= globalSpeed;
    
    bool stateChanged = false;
    
    // Update each layer
    for (auto& layer : layers) {
        updateLayer(layer, deltaTime, boneData);
    }
    
    // Recalculate bone matrices if dirty
    if (boneMatricesDirty) {
        // Resize matrices if needed
        size_t numBones = boneData.boneNameToIndex.size();
        if (cachedFinalBoneMatrices.size() != numBones) {
            cachedFinalBoneMatrices.resize(numBones);
        }
        
        // Initialize to identity
        std::fill(cachedFinalBoneMatrices.begin(), cachedFinalBoneMatrices.end(), Matrix4x4::identity());
        
        // Process layers from bottom to top
        for (const auto& layer : layers) {
            if (layer.weight <= 0.0f) continue;
            
            const auto& state = layer.blendState;
            const AnimationClip* activeClip = state.clipA;
            float activeTime = state.timeA;
            
            if (!activeClip || !activeClip->sourceData) continue;
            
            // Calculate blend if transitioning
            const AnimationData* animA = activeClip->sourceData;
            const AnimationData* animB = (state.clipB && state.clipB->sourceData) ? state.clipB->sourceData : nullptr;
            
            // Calculate bone transforms
            for (const auto& [boneName, boneIndex] : boneData.boneNameToIndex) {
                // Skip if bone mask is set and bone not in mask
                if (!layer.affectedBones.empty()) {
                    auto it = std::find(layer.affectedBones.begin(), layer.affectedBones.end(), boneName);
                    if (it == layer.affectedBones.end()) continue;
                }
                
                // Get offset matrix
                auto offsetIt = boneData.boneOffsetMatrices.find(boneName);
                if (offsetIt == boneData.boneOffsetMatrices.end()) continue;
                
                // Calculate animated transform from clip A
                float timeInTicksA = activeTime * activeClip->ticksPerSecond;
                Matrix4x4 transformA = calculateNodeTransform(animA, timeInTicksA, boneName, Matrix4x4::identity());
                
                // Blend with clip B if blending
                Matrix4x4 finalTransform = transformA;
                if (animB && state.isBlending()) {
                    float timeInTicksB = state.timeB * state.clipB->ticksPerSecond;
                    Matrix4x4 transformB = calculateNodeTransform(animB, timeInTicksB, boneName, Matrix4x4::identity());
                    finalTransform = blendTransforms(transformA, transformB, state.blendWeight, state.mode);
                }
                
                // CRITICAL FIX: Apply globalInverseTransform for correct axis transformation
                // This is essential for FBX files which use Z-up coordinate system.
                // Without this, skinned meshes will appear rotated and scaled incorrectly
                // when loaded from a saved project (where loader context is not available).
                Matrix4x4 boneMatrix = boneData.globalInverseTransform * finalTransform * offsetIt->second;
                
                // Blend with existing matrix based on layer weight
                if (layer.weight < 1.0f && layer.blendMode == BlendMode::Replace) {
                    cachedFinalBoneMatrices[boneIndex] = blendTransforms(
                        cachedFinalBoneMatrices[boneIndex], 
                        boneMatrix, 
                        layer.weight, 
                        BlendMode::Replace
                    );
                } else {
                    cachedFinalBoneMatrices[boneIndex] = boneMatrix;
                }
            }
        }
        
        stateChanged = true;
    }
    
    return stateChanged;
}

void AnimationController::updateLayer(AnimationLayer& layer, float deltaTime, const BoneData& boneData) {
    auto& state = layer.blendState;
    
    if (!state.clipA) {
        processQueue(layer);
        return;
    }
    
    // Update playback time
    float durationA = state.clipA->getDurationInSeconds();
    state.timeA += deltaTime;
    
    // Handle looping
    if (state.clipA->loop) {
        if (durationA > 0.0f && state.timeA >= durationA) {
            state.timeA = AnimationUtils::wrapTime(state.timeA, durationA);
        }
    } else {
        // Check if animation ended
        if (state.timeA >= durationA) {
            state.timeA = durationA;
            processQueue(layer);
        }
    }
    
    // Update blend if in progress
    if (state.isBlending() && state.clipB) {
        state.blendWeight += deltaTime / state.blendDuration;
        state.timeB += deltaTime;
        
        // Handle B clip looping
        float durationB = state.clipB->getDurationInSeconds();
        if (state.clipB->loop && durationB > 0.0f && state.timeB >= durationB) {
            state.timeB = AnimationUtils::wrapTime(state.timeB, durationB);
        }
        
        if (state.blendWeight >= 1.0f) {
            // Blend complete - switch to B
            state.clipA = state.clipB;
            state.timeA = state.timeB;
            state.clipB = nullptr;
            state.blendWeight = 0.0f;
        }
    }
    
    boneMatricesDirty = true;
}

void AnimationController::processQueue(AnimationLayer& layer) {
    if (layer.queue.empty()) return;
    
    AnimationQueueItem item = layer.queue.front();
    layer.queue.erase(layer.queue.begin());
    
    // Execute callback
    if (item.onStart) item.onStart();
    
    // Play the queued animation
    AnimationClip* clip = getClip(item.clipName);
    if (!clip) return;
    
    auto& state = layer.blendState;
    
    if (item.blendInTime > 0.0f && state.clipA != nullptr) {
        state.clipB = clip;
        state.timeB = 0.0f;
        state.blendWeight = 0.0f;
        state.blendDuration = item.blendInTime;
    } else {
        state.clipA = clip;
        state.timeA = 0.0f;
        state.clipB = nullptr;
        state.blendWeight = 0.0f;
    }
    
    // Handle repeat
    if (item.repeatCount > 1) {
        item.repeatCount--;
        layer.queue.insert(layer.queue.begin(), item);
    } else if (item.repeatCount == 0) {
        // Infinite repeat
        layer.queue.insert(layer.queue.begin(), item);
    }
}

Matrix4x4 AnimationController::calculateNodeTransform(
    const AnimationData* anim,
    float timeInTicks,
    const std::string& nodeName,
    const Matrix4x4& defaultTransform
) const {
    if (!anim) return defaultTransform;
    
    // Wrap time within animation duration
    double wrappedTime = fmod((double)timeInTicks, anim->duration);
    if (wrappedTime < 0) wrappedTime += anim->duration;
    
    Matrix4x4 translation = Matrix4x4::identity();
    Matrix4x4 rotation = Matrix4x4::identity();
    Matrix4x4 scale = Matrix4x4::identity();
    
    // Position interpolation
    auto posIt = anim->positionKeys.find(nodeName);
    if (posIt != anim->positionKeys.end() && !posIt->second.empty()) {
        const auto& keys = posIt->second;
        
        // Find surrounding keyframes
        size_t keyIndex = 0;
        for (size_t i = 0; i < keys.size() - 1; ++i) {
            if (wrappedTime < keys[i + 1].mTime) {
                keyIndex = i;
                break;
            }
        }
        
        size_t nextKey = (keyIndex + 1) % keys.size();
        double deltaTime = keys[nextKey].mTime - keys[keyIndex].mTime;
        if (deltaTime < 0) deltaTime += anim->duration;
        
        float t = (deltaTime > 0) ? (float)((wrappedTime - keys[keyIndex].mTime) / deltaTime) : 0.0f;
        t = std::max(0.0f, std::min(1.0f, t));
        
        const auto& start = keys[keyIndex].mValue;
        const auto& end = keys[nextKey].mValue;
        Vec3 pos(
            start.x + (end.x - start.x) * t,
            start.y + (end.y - start.y) * t,
            start.z + (end.z - start.z) * t
        );
        translation = Matrix4x4::translation(pos);
    }
    
    // Rotation interpolation (using quaternion slerp)
    auto rotIt = anim->rotationKeys.find(nodeName);
    if (rotIt != anim->rotationKeys.end() && !rotIt->second.empty()) {
        const auto& keys = rotIt->second;
        
        size_t keyIndex = 0;
        for (size_t i = 0; i < keys.size() - 1; ++i) {
            if (wrappedTime < keys[i + 1].mTime) {
                keyIndex = i;
                break;
            }
        }
        
        size_t nextKey = (keyIndex + 1) % keys.size();
        double deltaTime = keys[nextKey].mTime - keys[keyIndex].mTime;
        if (deltaTime < 0) deltaTime += anim->duration;
        
        float t = (deltaTime > 0) ? (float)((wrappedTime - keys[keyIndex].mTime) / deltaTime) : 0.0f;
        t = std::max(0.0f, std::min(1.0f, t));
        
        Quaternion start(keys[keyIndex].mValue.w, keys[keyIndex].mValue.x, 
                        keys[keyIndex].mValue.y, keys[keyIndex].mValue.z);
        Quaternion end(keys[nextKey].mValue.w, keys[nextKey].mValue.x,
                      keys[nextKey].mValue.y, keys[nextKey].mValue.z);
        
        Quaternion result = Quaternion::slerp(start, end, t);
        rotation = result.toMatrix();
    }
    
    // Scale interpolation
    auto sclIt = anim->scalingKeys.find(nodeName);
    if (sclIt != anim->scalingKeys.end() && !sclIt->second.empty()) {
        const auto& keys = sclIt->second;
        
        size_t keyIndex = 0;
        for (size_t i = 0; i < keys.size() - 1; ++i) {
            if (wrappedTime < keys[i + 1].mTime) {
                keyIndex = i;
                break;
            }
        }
        
        size_t nextKey = (keyIndex + 1) % keys.size();
        double deltaTime = keys[nextKey].mTime - keys[keyIndex].mTime;
        if (deltaTime < 0) deltaTime += anim->duration;
        
        float t = (deltaTime > 0) ? (float)((wrappedTime - keys[keyIndex].mTime) / deltaTime) : 0.0f;
        t = std::max(0.0f, std::min(1.0f, t));
        
        const auto& start = keys[keyIndex].mValue;
        const auto& end = keys[nextKey].mValue;
        Vec3 scl(
            start.x + (end.x - start.x) * t,
            start.y + (end.y - start.y) * t,
            start.z + (end.z - start.z) * t
        );
        scale = Matrix4x4::scaling(scl);
    }
    
    return translation * rotation * scale;
}

Matrix4x4 AnimationController::blendTransforms(
    const Matrix4x4& a,
    const Matrix4x4& b,
    float weight,
    BlendMode mode
) const {
    // Simple linear blend for now
    // TODO: Use proper quaternion slerp for rotation component
    Matrix4x4 result;
    float wa = 1.0f - weight;
    float wb = weight;
    
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            result.m[i][j] = a.m[i][j] * wa + b.m[i][j] * wb;
        }
    }
    
    return result;
}

void AnimationController::extractRootMotion(
    const AnimationData* anim,
    float prevTime,
    float currentTime,
    const std::string& rootBone
) {
    if (!anim || !rootMotionEnabled) return;
    
    // TODO: Implement root motion extraction
    // Extract position delta from root bone keyframes
    // Store in accumulatedRootMotion
}

// ========================================================================
// State Queries
// ========================================================================

bool AnimationController::isPlaying(int layer) const {
    if (layer >= static_cast<int>(layers.size())) return false;
    return layers[layer].blendState.clipA != nullptr;
}

bool AnimationController::isBlending(int layer) const {
    if (layer >= static_cast<int>(layers.size())) return false;
    return layers[layer].blendState.isBlending();
}

float AnimationController::getCurrentTime(int layer) const {
    if (layer >= static_cast<int>(layers.size())) return 0.0f;
    return layers[layer].blendState.timeA;
}

float AnimationController::getNormalizedTime(int layer) const {
    if (layer >= static_cast<int>(layers.size())) return 0.0f;
    
    const auto& state = layers[layer].blendState;
    if (!state.clipA) return 0.0f;
    
    float duration = state.clipA->getDurationInSeconds();
    return (duration > 0.0f) ? (state.timeA / duration) : 0.0f;
}

const std::string& AnimationController::getCurrentClipName(int layer) const {
    if (layer >= static_cast<int>(layers.size())) return emptyString;
    
    const auto& state = layers[layer].blendState;
    if (!state.clipA) return emptyString;
    
    return state.clipA->name;
}
