#include "AnimationController.h"
#include "globals.h"
#include <cmath>
#include <algorithm>

// Static member initialization
const std::string AnimationController::emptyString = "";

// ============================================================================
// ANIMATION CLIP Implementation
// ============================================================================

AnimationClip::AnimationClip(std::shared_ptr<AnimationData> data) : sourceData(data) {
    if (data) {
        name = data->name;
        duration = (float)data->duration;
        ticksPerSecond = (float)(data->ticksPerSecond > 0 ? data->ticksPerSecond : 24.0);
        startFrame = data->startFrame;
        endFrame = data->endFrame;
    }
}

AnimationController::AnimationController() {
    layers.resize(1);
    layers[0].name = "Base";
    globalSpeed = 1.0f;
    globalPaused = false;
    boneMatricesDirty = true;
}

// ============================================================================
// FULL RESET - Clears all animation cache and state
// ============================================================================

void AnimationController::clear() {
    // Clear all cached bone matrices
    cachedFinalBoneMatrices.clear();
    
    // Clear clips and index map
    clips.clear();
    clipNameToIndex.clear();
    
    // Reset layers to initial state
    layers.clear();
    layers.resize(1);
    layers[0] = AnimationLayer();
    layers[0].name = "Base";
    
    // Reset root motion state
    rootMotionEnabled = false;
    rootMotionBone = "Hips";
    accumulatedRootMotion = RootMotionDelta();
    lastRootPosition = Vec3(0, 0, 0);
    lastRootRotation = Quaternion();
    
    // Reset flags
    boneMatricesDirty = true;
    lastUpdateTime = -1.0f;
    globalPaused = false;
    globalSpeed = 1.0f;
    
    // Clear callback (optional - user may want to keep)
    eventCallback = nullptr;
    
    SCENE_LOG_INFO("[AnimController] Fully cleared - bone matrices, clips, and layers reset.");
}

// ============================================================================
// ANIMATION CONTROLLER Implementation
// ============================================================================

void AnimationController::registerClips(const std::vector<std::shared_ptr<AnimationData>>& animationDataList) {
    // We want to avoid full clear if we are appending, 
    // but the clip pointers in layers[i].blendState might still POINT to items in our 'clips' vector.
    // If 'clips' vector reallocates, those pointers break.
    
    // To be safe, we'll store old clip names that were playing
    std::vector<std::pair<int, std::string>> playingClips;
    for (int i = 0; i < layers.size(); ++i) {
        if (layers[i].blendState.clipA) {
            playingClips.push_back({i, layers[i].blendState.clipA->name});
        }
    }

    clips.clear();
    clipNameToIndex.clear();
    
    for (size_t i = 0; i < animationDataList.size(); ++i) {
        AnimationClip clip(animationDataList[i]);
        clipNameToIndex[clip.name] = clips.size();
        clips.push_back(clip);
        
        SCENE_LOG_INFO("[AnimController] Registered clip: " + clip.name + 
                      " (duration: " + std::to_string(clip.getDurationInSeconds()) + "s)");
    }
    
    // RE-LINK pointers in layers
    for (auto& pair : playingClips) {
        int layerIdx = pair.first;
        const std::string& clipName = pair.second;
        layers[layerIdx].blendState.clipA = getClip(clipName);
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
    if (rootMotionEnabled == enabled && rootMotionBone == rootBone) return;
    
    rootMotionEnabled = enabled;
    rootMotionBone = rootBone;
    accumulatedRootMotion = RootMotionDelta();
}

RootMotionDelta AnimationController::consumeRootMotion() {
    RootMotionDelta result = accumulatedRootMotion;
    accumulatedRootMotion = RootMotionDelta();
    return result;
}

std::string AnimationController::findBestRootMotionBone(const std::string& clipName) {
    AnimationClip* clip = getClip(clipName);
    if (!clip || !clip->sourceData) return "RootNode";
    
    auto& posKeys = clip->sourceData->positionKeys;
    
    // 1. Check direct RootNode
    if (posKeys.count("RootNode")) return "RootNode";
    
    // 2. Search for common names exactly
    static const std::vector<std::string> common = {"Hips", "Pelvis", "base_link", "root", "Armature"};
    for (const auto& name : common) {
        if (posKeys.count(name)) return name;
    }
    
    // 3. Search for names containing root-like strings (handles prefixes like "mixamorig:Hips")
    for (auto& pair : posKeys) {
        std::string nameLower = pair.first;
        for (char& c : nameLower) c = (char)tolower(c);
        
        if (nameLower.find("hips") != std::string::npos || 
            nameLower.find("pelvis") != std::string::npos ||
            nameLower.find("root") != std::string::npos) {
            return pair.first;
        }
    }

    return "RootNode";
}

// ========================================================================
// Main Update
// ========================================================================

bool AnimationController::update(float deltaTime, const BoneData& boneData) {
    if (globalPaused || (deltaTime <= 0.0f && !boneMatricesDirty)) {
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
        
        // Cache for global transforms during this update pass
        std::unordered_map<std::string, Matrix4x4> globalTransformCache;
        
        // Process layers from bottom to top
        for (const auto& layer : layers) {
            if (layer.weight <= 0.0f) continue;
            
            const auto& state = layer.blendState;
            const AnimationClip* activeClip = state.clipA;
            if (!activeClip || !activeClip->sourceData) continue;
            
            std::shared_ptr<AnimationData> animA = activeClip->sourceData;
            std::shared_ptr<AnimationData> animB = (state.clipB && state.clipB->sourceData) ? state.clipB->sourceData : nullptr;
            float weight = state.blendWeight;
            
            // Re-calculate the whole hierarchy for this layer's animations
            globalTransformCache.clear();
            
            for (const auto& [boneName, boneIndex] : boneData.boneNameToIndex) {
                // MODEL ISOLATION: Only update bones that belong to this animation's model
                // This prevents multiple animated models from overwriting each other's poses
                if (!animA->modelName.empty()) {
                    // Check if bone belongs to this model (prefixed with "modelName_")
                    if (boneName.find(animA->modelName + "_") != 0) {
                        continue; // Skip bones from other models
                    }
                }

                // Skip if bone mask is set and bone not in mask
                if (!layer.affectedBones.empty()) {
                    auto it = std::find(layer.affectedBones.begin(), layer.affectedBones.end(), boneName);
                    if (it == layer.affectedBones.end()) continue;
                }
                
                // Get global animated transform (recursively)
                Matrix4x4 animatedGlobal = getAnimatedGlobalTransform(
                    boneName, boneData, 
                    animA, state.timeA, 
                    animB, state.timeB, 
                    weight, state.mode, 
                    globalTransformCache);
                
                // Final Bone Matrix = GlobalInverse * GlobalAnimated * Offset
                auto offsetIt = boneData.boneOffsetMatrices.find(boneName);
                Matrix4x4 offset = (offsetIt != boneData.boneOffsetMatrices.end()) ? offsetIt->second : Matrix4x4::identity();
                
                // Retrieve correct Global Inverse for this model
                Matrix4x4 globalInv = boneData.globalInverseTransform;
                if (!animA->modelName.empty()) {
                     auto invIt = boneData.perModelInverses.find(animA->modelName);
                     if (invIt != boneData.perModelInverses.end()) {
                         globalInv = invIt->second;
                     }
                }
                
                // Proper Skinning Formula: GlobalInverse * BoneGlobal * BoneOffset
                Matrix4x4 boneMatrix = globalInv * animatedGlobal * offset;
                
                // Blend with existing matrix (from previous layers)
                // Since we isolated models, layers for DIFFERENT models won't compete for the same boneIndex.
                // Multiple layers for the SAME model will still blend correctly.
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
    
    // Extract root motion from clipA
    if (rootMotionEnabled && state.clipA->sourceData) {
        float prevTime = state.timeA - deltaTime;
        float currTime = state.timeA;
        if (state.clipA->loop && durationA > 0.0f && currTime >= durationA) {
            extractRootMotion(state.clipA->sourceData, prevTime, durationA, rootMotionBone);
            extractRootMotion(state.clipA->sourceData, 0.0f, fmodf(currTime, durationA), rootMotionBone);
        } else {
            extractRootMotion(state.clipA->sourceData, prevTime, currTime, rootMotionBone);
        }
    }

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
    std::shared_ptr<AnimationData> anim,
    float timeInTicks,
    const std::string& nodeName,
    const Matrix4x4& defaultTransform,
    bool wrap
) const {
    if (!anim) return defaultTransform;
    
    // Time sampling logic
    double sampleTime = (double)timeInTicks;
    if (wrap) {
        sampleTime = fmod(sampleTime, anim->duration);
        if (sampleTime < 0) sampleTime += anim->duration;
    } else {
        // Clamp to animation range if not wrapping
        sampleTime = std::max(0.0, std::min(sampleTime, anim->duration));
    }
    
    // Use sampleTime for interpolation
    double wrappedTime = sampleTime; 
    
    Matrix4x4 translation = Matrix4x4::identity();
    Matrix4x4 rotation = Matrix4x4::identity();
    Matrix4x4 scale = Matrix4x4::identity();
    
    // Position interpolation
    auto posIt = anim->positionKeys.find(nodeName);
    if (posIt != anim->positionKeys.end() && !posIt->second.empty()) {
        const auto& keys = posIt->second;
        
        size_t keyIndex = 0;
        if (wrappedTime >= keys.back().mTime) {
            keyIndex = keys.size() - 1;
        } else {
            for (size_t i = 0; i < keys.size() - 1; ++i) {
                if (wrappedTime < keys[i + 1].mTime) {
                    keyIndex = i;
                    break;
                }
            }
        }
        
        size_t nextKey = keyIndex + 1;
        float t = 0.0f;
        
        if (nextKey < keys.size()) {
            double deltaTime = keys[nextKey].mTime - keys[keyIndex].mTime;
            t = (deltaTime > 0) ? (float)((wrappedTime - keys[keyIndex].mTime) / deltaTime) : 0.0f;
        } else if (wrap) {
            nextKey = 0;
            double deltaTime = (anim->duration - keys[keyIndex].mTime) + keys[nextKey].mTime;
            double elapsed = wrappedTime - keys[keyIndex].mTime;
            t = (deltaTime > 0) ? (float)(elapsed / deltaTime) : 0.0f;
        } else {
            nextKey = keyIndex; // Clamp to end
            t = 0.0f;
        }
        
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
        if (wrappedTime >= keys.back().mTime) {
            keyIndex = keys.size() - 1;
        } else {
            for (size_t i = 0; i < keys.size() - 1; ++i) {
                if (wrappedTime < keys[i + 1].mTime) {
                    keyIndex = i;
                    break;
                }
            }
        }
        
        size_t nextKey = keyIndex + 1;
        float t = 0.0f;
        
        if (nextKey < keys.size()) {
            double deltaTime = keys[nextKey].mTime - keys[keyIndex].mTime;
            t = (deltaTime > 0) ? (float)((wrappedTime - keys[keyIndex].mTime) / deltaTime) : 0.0f;
        } else if (wrap) {
            nextKey = 0;
            double deltaTime = (anim->duration - keys[keyIndex].mTime) + keys[nextKey].mTime;
            double elapsed = wrappedTime - keys[keyIndex].mTime;
            t = (deltaTime > 0) ? (float)(elapsed / deltaTime) : 0.0f;
        } else {
            nextKey = keyIndex;
            t = 0.0f;
        }
        
        t = std::max(0.0f, std::min(1.0f, t));
        Quaternion q_start(keys[keyIndex].mValue.w, keys[keyIndex].mValue.x, 
                        keys[keyIndex].mValue.y, keys[keyIndex].mValue.z);
        Quaternion q_end(keys[nextKey].mValue.w, keys[nextKey].mValue.x,
                      keys[nextKey].mValue.y, keys[nextKey].mValue.z);
        
        Quaternion result = Quaternion::slerp(q_start, q_end, t);
        rotation = result.toMatrix();
    }
    
    // Scale interpolation
    auto sclIt = anim->scalingKeys.find(nodeName);
    if (sclIt != anim->scalingKeys.end() && !sclIt->second.empty()) {
        const auto& keys = sclIt->second;
        
        size_t keyIndex = 0;
        if (wrappedTime >= keys.back().mTime) {
            keyIndex = keys.size() - 1;
        } else {
            for (size_t i = 0; i < keys.size() - 1; ++i) {
                if (wrappedTime < keys[i + 1].mTime) {
                    keyIndex = i;
                    break;
                }
            }
        }
        
        size_t nextKey = keyIndex + 1;
        float t = 0.0f;
        
        if (nextKey < keys.size()) {
            double deltaTime = keys[nextKey].mTime - keys[keyIndex].mTime;
            t = (deltaTime > 0) ? (float)((wrappedTime - keys[keyIndex].mTime) / deltaTime) : 0.0f;
        } else if (wrap) {
            nextKey = 0;
            double deltaTime = (anim->duration - keys[keyIndex].mTime) + keys[nextKey].mTime;
            double elapsed = wrappedTime - keys[keyIndex].mTime;
            t = (deltaTime > 0) ? (float)(elapsed / deltaTime) : 0.0f;
        } else {
            nextKey = keyIndex;
            t = 0.0f;
        }
        
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

Matrix4x4 AnimationController::getAnimatedGlobalTransform(
    const std::string& boneName,
    const BoneData& boneData,
    std::shared_ptr<AnimationData> animA, float timeA,
    std::shared_ptr<AnimationData> animB, float timeB,
    float blendWeight, BlendMode mode,
    std::unordered_map<std::string, Matrix4x4>& cache
) const {
    // 1. Check cache first
    auto it = cache.find(boneName);
    if (it != cache.end()) return it->second;

    // 2. Get local transform (interpolated from animation or default bind pose)
    Matrix4x4 localDefault = Matrix4x4::identity();
    auto defIt = boneData.boneDefaultTransforms.find(boneName);
    if (defIt != boneData.boneDefaultTransforms.end()) {
        localDefault = defIt->second;
    }

    float ticksPerSecA = animA ? (float)animA->ticksPerSecond : 24.0f;
    Matrix4x4 localA = calculateNodeTransform(animA, timeA * ticksPerSecA, boneName, localDefault);
    
    // ROOT MOTION ISOLATION: If this is the root bone and RM is enabled, we remove translation
    // to prevent the character from moving twice (double transformation).
    if (rootMotionEnabled && boneName == rootMotionBone) {
        localA.m[0][3] = 0; localA.m[1][3] = 0; localA.m[2][3] = 0;
    }

    Matrix4x4 localFinal = localA;
    if (animB && blendWeight > 0.0f) {
        float ticksPerSecB = animB ? (float)animB->ticksPerSecond : 24.0f;
        Matrix4x4 localB = calculateNodeTransform(animB, timeB * ticksPerSecB, boneName, localDefault);
        
        if (rootMotionEnabled && boneName == rootMotionBone) {
            localB.m[0][3] = 0; localB.m[1][3] = 0; localB.m[2][3] = 0;
        }

        localFinal = blendTransforms(localA, localB, blendWeight, mode);
    }

    // 3. Get parent's global transform
    Matrix4x4 parentGlobal = Matrix4x4::identity();
    auto parentIt = boneData.boneParents.find(boneName);
    if (parentIt != boneData.boneParents.end()) {
        parentGlobal = getAnimatedGlobalTransform(parentIt->second, boneData, animA, timeA, animB, timeB, blendWeight, mode, cache);
    }

    // 4. Combine: Global = ParentGlobal * Local
    Matrix4x4 globalResult = parentGlobal * localFinal;
    
    // 5. Cache and return
    cache[boneName] = globalResult;
    return globalResult;
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
    std::shared_ptr<AnimationData> anim,
    float prevTime,
    float currentTime,
    const std::string& rootBone
) {
    if (!anim || !rootMotionEnabled) return;
    
    double ticksPerSec = anim->ticksPerSecond;
    
    // CRITICAL: Disable wrapping here so we sample pos(duration) and pos(0) correctly
    // instead of both wrapping to pos(0).
    Matrix4x4 matPrev = calculateNodeTransform(anim, (float)(prevTime * ticksPerSec), rootBone, Matrix4x4::identity(), false);
    Matrix4x4 matCurr = calculateNodeTransform(anim, (float)(currentTime * ticksPerSec), rootBone, Matrix4x4::identity(), false);
    
    Vec3 posPrev(matPrev.m[0][3], matPrev.m[1][3], matPrev.m[2][3]);
    Vec3 posCurr(matCurr.m[0][3], matCurr.m[1][3], matCurr.m[2][3]);
    
    Vec3 delta = posCurr - posPrev;
    
    accumulatedRootMotion.positionDelta = accumulatedRootMotion.positionDelta + delta;
    accumulatedRootMotion.hasPosition = true;
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
