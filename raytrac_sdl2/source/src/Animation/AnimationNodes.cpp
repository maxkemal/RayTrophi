#include "AnimationNodes.h"
#include "globals.h"
#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace AnimationGraph {
    
    // Helper for std::string with ImGui
    static bool InputTextString(const char* label, std::string& str) {
        char buf[256];
        memset(buf, 0, sizeof(buf));
        strncpy(buf, str.c_str(), sizeof(buf) - 1);
        if (ImGui::InputText(label, buf, sizeof(buf))) {
            str = buf;
            return true;
        }
        return false;
    }

    // ============================================================================
    // CURVE DATA Implementation
    // ============================================================================
    
    float CurveData::evaluate(float time) const {
        if (keyframes.empty()) return 0.0f;
        if (keyframes.size() == 1) return keyframes[0].second;
        
        // Handle looping
        float t = time;
        if (loop && !keyframes.empty()) {
            float duration = keyframes.back().first - keyframes.front().first;
            if (duration > 0) {
                t = fmodf(time - keyframes.front().first, duration) + keyframes.front().first;
            }
        }
        
        // Clamp to range
        if (t <= keyframes.front().first) return keyframes.front().second;
        if (t >= keyframes.back().first) return keyframes.back().second;
        
        // Find surrounding keyframes
        for (size_t i = 0; i < keyframes.size() - 1; ++i) {
            if (t >= keyframes[i].first && t < keyframes[i + 1].first) {
                float localT = (t - keyframes[i].first) / (keyframes[i + 1].first - keyframes[i].first);
                return keyframes[i].second + (keyframes[i + 1].second - keyframes[i].second) * localT;
            }
        }
        
        return keyframes.back().second;
    }
    
    // ============================================================================
    // ANIM NODE BASE Implementation
    // ============================================================================
    
    PoseData AnimNodeBase::getInputPose(int inputIndex, AnimationEvalContext& ctx) {
        if (!ctx.graph || inputIndex >= inputs.size()) return PoseData{};
        
        uint32_t pinId = inputs[inputIndex].id;
        
        // Find link connected to this input pin
        for (const auto& link : ctx.graph->links) {
            if (link.endPinId == pinId) {
                // Find the node on the other side
                AnimNodeBase* upstreamNode = ctx.graph->findNodeByPinId(link.startPinId);
                if (upstreamNode) {
                    // Find which output index it is
                    int outputIdx = 0;
                    for (size_t i = 0; i < upstreamNode->outputs.size(); ++i) {
                        if (upstreamNode->outputs[i].id == link.startPinId) {
                            outputIdx = (int)i;
                            break;
                        }
                    }
                    return upstreamNode->computePose(ctx);
                }
            }
        }
        
        return PoseData{};
    }
    
    // ============================================================================
    // ANIM CLIP NODE Implementation
    // ============================================================================
    
    AnimClipNode::AnimClipNode() {
        metadata.displayName = "Animation Clip";
        metadata.description = "Plays a single animation clip";
        metadata.category = "Input";
        metadata.typeId = "AnimClip";
        metadata.headerColor = COLOR_INPUT;
        
        isPlaying = true;  // Default to playing for immediate feedback
        loop = true;       // Standard for clips
        
        setupPins();
    }
    
    void AnimClipNode::setupPins() {
        // Inputs
        addInput("Time Override", NodeSystem::DataType::Float).optional = true;
        addInput("Speed", NodeSystem::DataType::Float).optional = true;
        
        // Outputs
        addOutput("Pose", NodeSystem::DataType::Custom);  // Custom = Pose type
        addOutput("Normalized Time", NodeSystem::DataType::Float);
        addOutput("Remaining", NodeSystem::DataType::Float);
    }
    
    PoseData AnimClipNode::computePose(AnimationEvalContext& ctx) {
        PoseData result;
        
        if (clipName.empty() || !ctx.clipsPtr || !ctx.boneData) {
            return result;
        }
        
        // 1. Find the clip
        const AnimationClip* clip = nullptr;
        for (auto& c : *ctx.clipsPtr) {
            if (c.name == clipName) { clip = &c; break; }
        }
        
        if (!clip || !clip->sourceData) return result;
        
        // 2. Update time (Unreal Style)
        float prevTime = currentTime;
        float currTimeTemp = currentTime;
        
        if (isPlaying) {
            currTimeTemp += ctx.deltaTime * playbackSpeed;
            float duration = clip->getDurationInSeconds();
            
            // --- ROOT MOTION EXTRACTION ---
            if (ctx.useRootMotion && clip->sourceData) {
                double tps = clip->sourceData->ticksPerSecond;
                auto extractRM = [&](float startT, float endT) {
                    float startTicks = startT * (float)tps;
                    float endTicks = endT * (float)tps;
                    auto posIt = clip->sourceData->positionKeys.find(ctx.rootMotionBone);
                    if (posIt != clip->sourceData->positionKeys.end() && !posIt->second.empty()) {
                        Vec3 posPrev = sampleVectorKey(posIt->second, startTicks, clip->sourceData->duration, false);
                        Vec3 posCurr = sampleVectorKey(posIt->second, endTicks, clip->sourceData->duration, false);
                        result.rootMotion.positionDelta = result.rootMotion.positionDelta + (posCurr - posPrev);
                        result.rootMotion.hasPosition = true;
                    }
                };
                
                if (loop && duration > 0.0f && currTimeTemp >= duration) {
                    extractRM(prevTime, duration);
                    extractRM(0.0f, fmodf(currTimeTemp, duration));
                } else if (duration > 0.0f) {
                    extractRM(prevTime, currTimeTemp);
                }
            }
            
            currentTime = currTimeTemp;
            if (duration > 0.0f) {
                if (loop) currentTime = fmodf(currentTime, duration);
                else currentTime = std::clamp(currentTime, 0.0f, duration);
            }
        }
        
        // 3. Sample Animation to TRS (Industry Standard)
        size_t boneCount = ctx.boneData->boneNameToIndex.size();
        result.trsTransforms.resize(boneCount);
        result.boneTransforms.resize(boneCount);
        result.wasUpdated = isPlaying && (ctx.deltaTime > 0.0f);
        
        float timeInTicks = currentTime * clip->ticksPerSecond;
        auto anim = clip->sourceData;
        
        for (const auto& [boneName, boneIndex] : ctx.boneData->boneNameToIndex) {
            BoneTransform trs;
            // Match AnimationController: when a clip is active, we build from tracks directly.
            // Omitted tracks remain Identity. Do not fallback to Bind Pose values, as 
            // the legacy controller ignores defaultTransform if the clip exists.

            // Override with Animation Keys if they exist
            auto posIt = anim->positionKeys.find(boneName);
            if (posIt != anim->positionKeys.end() && !posIt->second.empty())
                trs.translation = sampleVectorKey(posIt->second, timeInTicks, anim->duration);
            
            auto rotIt = anim->rotationKeys.find(boneName);
            if (rotIt != anim->rotationKeys.end() && !rotIt->second.empty())
                trs.rotation = sampleQuatKey(rotIt->second, timeInTicks, anim->duration);
            
            auto sclIt = anim->scalingKeys.find(boneName);
            if (sclIt != anim->scalingKeys.end() && !sclIt->second.empty())
                trs.scale = sampleVectorKey(sclIt->second, timeInTicks, anim->duration, true);
            
            // ROOT MOTION ZEROING
            if (ctx.useRootMotion && boneName == ctx.rootMotionBone) {
                trs.translation = Vec3(0, 0, 0);
            }
            
            result.trsTransforms[boneIndex] = trs;
            result.boneTransforms[boneIndex] = trs.toMatrix();
        }
        
        // Include animated nodes that aren't in boneNameToIndex (e.g. Armature, RootNode that lack skin weights)
        std::unordered_set<std::string> extraNodes;
        for (const auto& pair : anim->positionKeys) extraNodes.insert(pair.first);
        for (const auto& pair : anim->rotationKeys) extraNodes.insert(pair.first);
        for (const auto& pair : anim->scalingKeys) extraNodes.insert(pair.first);
        
        for (const std::string& nodeName : extraNodes) {
             if (ctx.boneData->boneNameToIndex.count(nodeName) > 0) continue;
             
             BoneTransform trs;
             // Match AnimationController: tracks only.
             
             auto posIt = anim->positionKeys.find(nodeName);
             if (posIt != anim->positionKeys.end() && !posIt->second.empty())
                 trs.translation = sampleVectorKey(posIt->second, timeInTicks, anim->duration);
             
             auto rotIt = anim->rotationKeys.find(nodeName);
             if (rotIt != anim->rotationKeys.end() && !rotIt->second.empty())
                 trs.rotation = sampleQuatKey(rotIt->second, timeInTicks, anim->duration);
             
             auto sclIt = anim->scalingKeys.find(nodeName);
             if (sclIt != anim->scalingKeys.end() && !sclIt->second.empty())
                 trs.scale = sampleVectorKey(sclIt->second, timeInTicks, anim->duration, true);
                 
             // ROOT MOTION ZEROING for Extra Nodes too (e.g. root bone without skin weights)
             if (ctx.useRootMotion && nodeName == ctx.rootMotionBone) {
                 trs.translation = Vec3(0, 0, 0);
             }
                 
             result.extraTransforms[nodeName] = trs;
        }

        if (clip->getDurationInSeconds() > 0.0f) {
            result.normalizedTime = currentTime / clip->getDurationInSeconds();
        }
        
        return result;
    }
    
    void AnimClipNode::drawContent() {
        // Dropdown to select from all available clips in the scene
        static std::vector<const char*> clipNames;
        clipNames.clear();
        
        extern std::vector<std::shared_ptr<AnimationData>>* g_uiClipsRef; 
        
        if (g_uiClipsRef) {
            for (auto& clipData : *g_uiClipsRef) {
                if (clipData) clipNames.push_back(clipData->name.c_str());
            }
        }

        if (!clipNames.empty()) {
            int currentIdx = -1;
            for (int i=0; i < (int)clipNames.size(); ++i) {
                if (clipName == clipNames[i]) { currentIdx = i; break; }
            }

            ImGui::PushItemWidth(150);
            if (ImGui::Combo("Select Clip", &currentIdx, clipNames.data(), (int)clipNames.size())) {
                clipName = clipNames[currentIdx];
                reset(); // Start from beginning on change
            }
            ImGui::PopItemWidth();
        } else {
            InputTextString("Clip Name", clipName);
            ImGui::TextDisabled("(No clips loaded in scene)");
        }
        
        ImGui::Separator();
        
        // --- Playback Visualizer ---
        if (!clipName.empty()) {
            float normTime = 0.0f;
            // Get current duration from clips if possible
            if (g_uiClipsRef) {
                for(auto& c : *g_uiClipsRef) {
                    if(c && c->name == clipName && c->duration > 0) {
                        float duration = c->duration / (c->ticksPerSecond > 0 ? c->ticksPerSecond : 24.0f);
                        normTime = (duration > 0) ? (currentTime / duration) : 0.0f;
                        if (loop) normTime = fmodf(normTime, 1.0f);
                        else normTime = std::clamp(normTime, 0.0f, 1.0f);
                        break;
                    }
                }
            }
            
            char overlay[32];
            snprintf(overlay, sizeof(overlay), "%.2fs", currentTime);
            ImGui::ProgressBar(normTime, ImVec2(-1, 15), overlay);
        }

        ImGui::SliderFloat("Speed", &playbackSpeed, -2.0f, 2.0f);
        ImGui::Checkbox("Loop", &loop);
        
        // Playback controls
        if (ImGui::Button(isPlaying ? "Pause" : "Play")) {
            isPlaying = !isPlaying;
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset")) {
            reset();
        }

        ImGui::Text("Time: %.2fs", currentTime);
    }
    
    // ============================================================================
    // HELPER: Sample animation keys
    // ============================================================================
    
    Vec3 sampleVectorKey(const std::vector<aiVectorKey>& keys, float time, double duration, bool wrap) {
        if (keys.empty()) return Vec3(0, 0, 0);
        if (keys.size() == 1) {
            return Vec3(keys[0].mValue.x, keys[0].mValue.y, keys[0].mValue.z);
        }
        
        double t = (double)time;
        if (wrap) {
            t = fmod(t, duration);
            if (t < 0) t += duration;
        } else {
            t = std::max(0.0, std::min(t, duration));
        }
        
        // Find surrounding keyframes
        size_t keyIndex = 0;
        for (size_t i = 0; i < keys.size() - 1; ++i) {
            if (t < keys[i + 1].mTime) {
                keyIndex = i;
                break;
            }
        }
        
        size_t nextKey = (keyIndex + 1) % keys.size();
        double deltaTime = keys[nextKey].mTime - keys[keyIndex].mTime;
        if (deltaTime < 0) deltaTime += duration;
        
        float factor = (deltaTime > 0) ? (float)((t - keys[keyIndex].mTime) / deltaTime) : 0.0f;
        factor = std::max(0.0f, std::min(1.0f, factor));
        
        const auto& start = keys[keyIndex].mValue;
        const auto& end = keys[nextKey].mValue;
        
        return Vec3(
            start.x + (end.x - start.x) * factor,
            start.y + (end.y - start.y) * factor,
            start.z + (end.z - start.z) * factor
        );
    }
    
    Quaternion sampleQuatKey(const std::vector<aiQuatKey>& keys, float time, double duration) {
        if (keys.empty()) return Quaternion(1, 0, 0, 0);
        if (keys.size() == 1) {
            return Quaternion(keys[0].mValue.w, keys[0].mValue.x, keys[0].mValue.y, keys[0].mValue.z);
        }
        
        double t = fmod((double)time, duration);
        if (t < 0) t += duration;
        
        size_t keyIndex = 0;
        for (size_t i = 0; i < keys.size() - 1; ++i) {
            if (t < keys[i + 1].mTime) {
                keyIndex = i;
                break;
            }
        }
        
        size_t nextKey = (keyIndex + 1) % keys.size();
        double deltaTime = keys[nextKey].mTime - keys[keyIndex].mTime;
        if (deltaTime < 0) deltaTime += duration;
        
        float factor = (deltaTime > 0) ? (float)((t - keys[keyIndex].mTime) / deltaTime) : 0.0f;
        factor = std::max(0.0f, std::min(1.0f, factor));
        
        Quaternion start(keys[keyIndex].mValue.w, keys[keyIndex].mValue.x, 
                        keys[keyIndex].mValue.y, keys[keyIndex].mValue.z);
        Quaternion end(keys[nextKey].mValue.w, keys[nextKey].mValue.x,
                      keys[nextKey].mValue.y, keys[nextKey].mValue.z);
        
        return Quaternion::slerp(start, end, factor);
    }
    
    // ============================================================================
    // POSE SNAPSHOT NODE Implementation
    // ============================================================================
    
    PoseSnapshotNode::PoseSnapshotNode() {
        metadata.displayName = "Pose Snapshot";
        metadata.category = "Input";
        metadata.typeId = "PoseSnapshot";
        metadata.headerColor = COLOR_INPUT;
        setupPins();
    }
    
    void PoseSnapshotNode::setupPins() {
        addInput("Capture", NodeSystem::DataType::Bool);
        addOutput("Pose", NodeSystem::DataType::Custom);
    }
    
    PoseData PoseSnapshotNode::computePose(AnimationEvalContext& ctx) {
        return capturedPose;
    }
    
    void PoseSnapshotNode::drawContent() {
        if (ImGui::Button("Capture Now")) {
            hasCapture = true; // Simplified placeholder
        }
        ImGui::SameLine();
        if (ImGui::Button("Clear")) {
            clear();
        }
        if (hasCapture) ImGui::TextColored(ImVec4(0,1,0,1), "Pose Captured");
        else ImGui::TextDisabled("No pose captured");
    }
    
    void PoseSnapshotNode::capture(const PoseData& pose) {
        capturedPose = pose;
        hasCapture = true;
    }
    
    void PoseSnapshotNode::clear() {
        capturedPose = PoseData{};
        hasCapture = false;
    }

    // ============================================================================
    // BLEND NODE Implementation
    // ============================================================================
    
    BlendNode::BlendNode() {
        metadata.displayName = "Blend";
        metadata.description = "Blends two poses based on alpha";
        metadata.category = "Blend";
        metadata.typeId = "AnimBlend";
        metadata.headerColor = COLOR_BLEND;
        
        setupPins();
    }
    
    void BlendNode::setupPins() {
        addInput("Pose A", NodeSystem::DataType::Custom);
        addInput("Pose B", NodeSystem::DataType::Custom);
        addInput("Alpha", NodeSystem::DataType::Float);
        
        addOutput("Pose", NodeSystem::DataType::Custom);
    }
    
    PoseData BlendNode::computePose(AnimationEvalContext& ctx) {
        PoseData poseA = getInputPose(0, ctx);
        PoseData poseB = getInputPose(1, ctx);
        
        // Get alpha from input or use property
        float blendAlpha = useInputAlpha ? alpha : alpha;  // TODO: Get from input
        
        return blendPoses(poseA, poseB, blendAlpha, ctx);
    }
    
    PoseData BlendNode::blendPoses(const PoseData& a, const PoseData& b, float t, const AnimationEvalContext& ctx) {
        PoseData result;
        
        if (!a.isValid()) return b;
        if (!b.isValid()) return a;
        
        size_t count = std::max(a.boneCount(), b.boneCount());
        result.trsTransforms.resize(count);
        result.boneTransforms.resize(count);
        result.boneNames.resize(count);
        result.blendWeight = 1.0f;
        
        // Blend Root Motion
        if (a.rootMotion.hasPosition && b.rootMotion.hasPosition) {
            result.rootMotion.positionDelta = Vec3::lerp(a.rootMotion.positionDelta, b.rootMotion.positionDelta, t);
            result.rootMotion.hasPosition = true;
        } else if (a.rootMotion.hasPosition) {
            result.rootMotion.positionDelta = a.rootMotion.positionDelta * (1.0f - t);
            result.rootMotion.hasPosition = true;
        } else if (b.rootMotion.hasPosition) {
            result.rootMotion.positionDelta = b.rootMotion.positionDelta * t;
            result.rootMotion.hasPosition = true;
        }
        
        // TRS Interpolation (Smooth and Correct)
        for (size_t i = 0; i < count; ++i) {
            BoneTransform trsA = (i < a.trsTransforms.size()) ? a.trsTransforms[i] : BoneTransform::identity();
            BoneTransform trsB = (i < b.trsTransforms.size()) ? b.trsTransforms[i] : BoneTransform::identity();
            
            result.trsTransforms[i].translation = Vec3::lerp(trsA.translation, trsB.translation, t);
            result.trsTransforms[i].scale = Vec3::lerp(trsA.scale, trsB.scale, t);
            result.trsTransforms[i].rotation = Quaternion::slerp(trsA.rotation, trsB.rotation, t);
            
            // Get bone name for offset lookup
            std::string boneName = "";
            if (i < a.boneNames.size()) boneName = a.boneNames[i];
            else if (i < b.boneNames.size()) boneName = b.boneNames[i];
            result.boneNames[i] = boneName;
            
            // Generate matrix and apply offset
            Matrix4x4 localMat = result.trsTransforms[i].toMatrix();
            auto offsetIt = ctx.boneData->boneOffsetMatrices.find(boneName);
            if (offsetIt != ctx.boneData->boneOffsetMatrices.end()) {
                result.boneTransforms[i] = localMat * offsetIt->second;
            } else {
                result.boneTransforms[i] = localMat;
            }
            
        }
        
        for (const auto& [nodeName, trsA] : a.extraTransforms) {
            auto itB = b.extraTransforms.find(nodeName);
            if (itB != b.extraTransforms.end()) {
                 const BoneTransform& trsB = itB->second;
                 BoneTransform blended;
                 blended.translation = Vec3::lerp(trsA.translation, trsB.translation, t);
                 blended.rotation = Quaternion::slerp(trsA.rotation, trsB.rotation, t);
                 blended.scale = Vec3::lerp(trsA.scale, trsB.scale, t);
                 result.extraTransforms[nodeName] = blended;
            } else {
                 result.extraTransforms[nodeName] = trsA;
            }
        }
        for (const auto& [nodeName, trsB] : b.extraTransforms) {
            if (a.extraTransforms.find(nodeName) == a.extraTransforms.end()) {
                 result.extraTransforms[nodeName] = trsB;
            }
        }
        
        return result;
    }

    
    void BlendNode::drawContent() {
        ImGui::Checkbox("Use Input Alpha", &useInputAlpha);
        if (!useInputAlpha) {
            ImGui::SliderFloat("Alpha", &alpha, 0.0f, 1.0f);
        }
    }
    
    // ============================================================================
    // ADDITIVE BLEND NODE Implementation
    // ============================================================================
    
    AdditiveBlendNode::AdditiveBlendNode() {
        metadata.displayName = "Additive Blend";
        metadata.category = "Blend";
        metadata.typeId = "AdditiveBlend";
        metadata.headerColor = COLOR_BLEND;
        setupPins();
    }
    
    void AdditiveBlendNode::setupPins() {
        addInput("Base", NodeSystem::DataType::Custom);
        addInput("Additive", NodeSystem::DataType::Custom);
        addOutput("Pose", NodeSystem::DataType::Custom);
    }
    
    PoseData AdditiveBlendNode::computePose(AnimationEvalContext& ctx) {
        PoseData base = getInputPose(0, ctx);
        PoseData additive = getInputPose(1, ctx);
        if (!base.isValid()) return additive;
        if (!additive.isValid()) return base;
        
        PoseData result = base;
        size_t count = std::min(base.boneCount(), additive.boneCount());
        
        for (size_t i = 0; i < count; ++i) {
            BoneTransform trsBase = (i < base.trsTransforms.size()) ? base.trsTransforms[i] : BoneTransform::identity();
            BoneTransform trsAdd = (i < additive.trsTransforms.size()) ? additive.trsTransforms[i] : BoneTransform::identity();
            
            result.trsTransforms[i].translation = trsBase.translation + trsAdd.translation;
            result.trsTransforms[i].rotation = trsAdd.rotation * trsBase.rotation;
            result.trsTransforms[i].scale = trsBase.scale * trsAdd.scale;
            
            result.boneTransforms[i] = result.trsTransforms[i].toMatrix();
            // TODO: properly adjust base offsets if needed, but in standard additive, 
            // base offsets are applied AFTER blending, so TRS is enough.
        }
        
        // Additive Blend Root Motion
        if (additive.rootMotion.hasPosition) {
            result.rootMotion.positionDelta = base.rootMotion.positionDelta + additive.rootMotion.positionDelta;
            result.rootMotion.hasPosition = true;
        }
        
        for (const auto& [nodeName, trsAdd] : additive.extraTransforms) {
            // Get bone name for offset lookup
            std::string boneName = (i < result.boneNames.size()) ? result.boneNames[i] : "";
            
            Matrix4x4 localMat = result.trsTransforms[i].toMatrix();
            auto offsetIt = ctx.boneData->boneOffsetMatrices.find(boneName);
            if (offsetIt != ctx.boneData->boneOffsetMatrices.end()) {
                result.boneTransforms[i] = localMat * offsetIt->second;
            } else {
                result.boneTransforms[i] = localMat;
            }
        }
        
        for (const auto& [nodeName, trsAdditive] : additive.extraTransforms) {
            auto baseIt = base.extraTransforms.find(nodeName);
            if (baseIt != base.extraTransforms.end()) {
                const BoneTransform& trsBase = baseIt->second;
                BoneTransform resultTrs;
                resultTrs.translation = trsBase.translation + trsAdditive.translation;
                resultTrs.rotation = trsAdditive.rotation * trsBase.rotation;
                resultTrs.scale = trsBase.scale * trsAdditive.scale;
                result.extraTransforms[nodeName] = resultTrs;
            } else {
                result.extraTransforms[nodeName] = trsAdditive;
            }
        }
        
        return result;
    }
    
    void AdditiveBlendNode::drawContent() {
        ImGui::SliderFloat("Alpha", &alpha, 0.0f, 1.0f);
    }

    // ============================================================================
    // LAYERED BLEND NODE Implementation
    // ============================================================================
    
    LayeredBlendNode::LayeredBlendNode() {
        metadata.displayName = "Layered Blend";
        metadata.description = "Per-bone masked blending for upper/lower body";
        metadata.category = "Blend";
        metadata.typeId = "LayeredBlend";
        metadata.headerColor = COLOR_BLEND;
        
        setupPins();
    }
    
    void LayeredBlendNode::setupPins() {
        addInput("Base Pose", NodeSystem::DataType::Custom);
        addInput("Layer Pose", NodeSystem::DataType::Custom);
        addInput("Alpha", NodeSystem::DataType::Float);
        
        addOutput("Pose", NodeSystem::DataType::Custom);
    }
    
    PoseData LayeredBlendNode::computePose(AnimationEvalContext& ctx) {
        PoseData basePose = getInputPose(0, ctx);
        PoseData layerPose = getInputPose(1, ctx);
        
        if (!basePose.isValid()) return layerPose;
        if (!layerPose.isValid()) return basePose;
        
        PoseData result = basePose;  // Start with base
        
        // Layered Blend Root Motion (assuming root bone is in the mask, interpolate it)
        float rootWeight = 0.0f;
        auto getRootWeight = [&]() -> float {
            if (affectedBones.empty()) return 1.0f;
            for (const auto& bone : affectedBones) {
                if (bone == ctx.rootMotionBone || bone == "RootNode") return 1.0f;
            }
            return 0.0f;
        };
        rootWeight = getRootWeight() * alpha;
        
        if (layerPose.rootMotion.hasPosition) {
            result.rootMotion.positionDelta = Vec3::lerp(basePose.rootMotion.positionDelta, layerPose.rootMotion.positionDelta, rootWeight);
            result.rootMotion.hasPosition = true;
        }
        
        // Blend in layer poses wherever weight > 0
        // Ensure result has TRS and matrices
        size_t count = std::max(basePose.boneCount(), layerPose.boneCount());
        result.trsTransforms.resize(count);
        result.boneTransforms.resize(count);
        
        // Check if bone is affected
        auto isAffected = [this](const std::string& boneName) -> float {
            if (affectedBones.empty()) return 1.0f;  // All bones affected if no list
            
            // Check if bone or parent is in list (simple name-based check for now)
            for (const auto& affected : affectedBones) {
                if (boneName.find(affected) != std::string::npos) {
                    return 1.0f;
                }
            }
            return 0.0f;
        };
        
        // Blend affected bones in TRS space
        for (size_t i = 0; i < count; ++i) {
            BoneTransform trsBase = (i < basePose.trsTransforms.size()) ? basePose.trsTransforms[i] : BoneTransform::identity();
            BoneTransform trsLayer = (i < layerPose.trsTransforms.size()) ? layerPose.trsTransforms[i] : BoneTransform::identity();
            
            std::string boneName = (i < result.boneNames.size()) ? result.boneNames[i] : "";
            float weight = isAffected(boneName) * alpha;
            
            if (weight > 0.0f) {
                // Blend TRS components
                result.trsTransforms[i].translation = Vec3::lerp(trsBase.translation, trsLayer.translation, weight);
                result.trsTransforms[i].scale = Vec3::lerp(trsBase.scale, trsLayer.scale, weight);
                result.trsTransforms[i].rotation = Quaternion::slerp(trsBase.rotation, trsLayer.rotation, weight);
                
                // Update matrix and apply offset
                Matrix4x4 localMat = result.trsTransforms[i].toMatrix();
                auto offsetIt = ctx.boneData->boneOffsetMatrices.find(boneName);
                if (offsetIt != ctx.boneData->boneOffsetMatrices.end()) {
                    result.boneTransforms[i] = localMat * offsetIt->second;
                } else {
                    result.boneTransforms[i] = localMat;
                }
            } else {
                result.trsTransforms[i] = trsBase;
                result.boneTransforms[i] = (i < basePose.boneTransforms.size()) ? basePose.boneTransforms[i] : Matrix4x4::identity();
            }
        }
        
        for (const auto& [nodeName, trsBase] : basePose.extraTransforms) {
            float weight = isAffected(nodeName) * alpha;
            auto itLayer = layerPose.extraTransforms.find(nodeName);
            
            if (weight > 0.0f && itLayer != layerPose.extraTransforms.end()) {
                const BoneTransform& trsLayer = itLayer->second;
                BoneTransform blended;
                blended.translation = Vec3::lerp(trsBase.translation, trsLayer.translation, weight);
                blended.scale = Vec3::lerp(trsBase.scale, trsLayer.scale, weight);
                blended.rotation = Quaternion::slerp(trsBase.rotation, trsLayer.rotation, weight);
                result.extraTransforms[nodeName] = blended;
            } else {
                result.extraTransforms[nodeName] = trsBase;
            }
        }
        
        for (const auto& [nodeName, trsLayer] : layerPose.extraTransforms) {
            float weight = isAffected(nodeName) * alpha;
            if (weight > 0.0f && basePose.extraTransforms.find(nodeName) == basePose.extraTransforms.end()) {
                result.extraTransforms[nodeName] = trsLayer; // Assume base is identity effectively? No, rely on fallback if needed, or just insert it.
            }
        }
        
        return result;
    }
    
    void LayeredBlendNode::drawContent() {
        ImGui::SliderFloat("Alpha", &alpha, 0.0f, 1.0f);
        
        ImGui::Text("Affected Bones:");
        for (size_t i = 0; i < affectedBones.size(); ++i) {
            ImGui::BulletText("%s", affectedBones[i].c_str());
        }
        
        // TODO: Add bone picker UI
    }
    
    // ============================================================================
    // STATE MACHINE NODE Implementation
    // ============================================================================
    
    bool StateMachineNode::Transition::evaluate(const AnimationEvalContext& ctx) const {
        switch (conditionType) {
            case ConditionType::Bool:
                return ctx.getBoolParam(parameterName, false);
                
            case ConditionType::FloatGreater:
                return ctx.getFloatParam(parameterName, 0.0f) > compareValue;
                
            case ConditionType::FloatLess:
                return ctx.getFloatParam(parameterName, 0.0f) < compareValue;
                
            case ConditionType::Trigger:
                // Triggers are consumed, so we check if it exists
                return ctx.triggerParams.find(parameterName) != ctx.triggerParams.end();
                
            default:
                return false;
        }
    }
    
    StateMachineNode::StateMachineNode() {
        metadata.displayName = "State Machine";
        metadata.description = "Animation state machine with transitions";
        metadata.category = "State";
        metadata.typeId = "StateMachine";
        metadata.headerColor = COLOR_STATE;
        
        setupPins();
    }
    
    void StateMachineNode::setupPins() {
        addOutput("Pose", NodeSystem::DataType::Custom);
        addOutput("Current State", NodeSystem::DataType::String);
    }
    
    PoseData StateMachineNode::computePose(AnimationEvalContext& ctx) {
        if (!ctx.graph) return PoseData{};

        // 1. Get current state pose FIRST (to know its normalized time)
        auto getStatePose = [&](const std::string& stateName) -> PoseData {
            for (size_t i = 0; i < states.size(); ++i) {
                if (states[i].name == stateName) {
                    // Start reading from inputs. Assume pins were added in order of states.
                    return getInputPose(i, ctx);
                }
            }
            return PoseData{};
        };

        PoseData currentPose = getStatePose(currentStateName);

        // 2. Check for transitions (passing current pose for Exit Time check)
        if (!isTransitioning) {
            for (const auto& trans : transitions) {
                if (trans.fromState == currentStateName) {
                    bool conditionMet = trans.evaluate(ctx);
                    bool exitTimeMet = !trans.hasExitTime || (currentPose.normalizedTime >= trans.exitTime);
                    
                    if (conditionMet && exitTimeMet) {
                        targetStateName = trans.toState;
                        transitionProgress = 0.0f;
                        isTransitioning = true;
                        for (const auto& state : states) {
                            if (state.name == currentStateName && state.onExit) state.onExit();
                        }
                        SCENE_LOG_INFO("[StateMachine] Transition: " + currentStateName + " -> " + targetStateName);
                        break;
                    }
                }
            }
        }
        
        // 3. Update transition progress
        if (isTransitioning) {
            updateTransition(ctx.deltaTime);
            PoseData targetPose = getStatePose(targetStateName);
            
            float t = std::clamp(transitionProgress, 0.0f, 1.0f);
            return BlendNode::blendPoses(currentPose, targetPose, t, ctx);
        }
        
        return currentPose;
    }
    
    void StateMachineNode::updateTransition(float deltaTime) {
        float blendTime = 0.3f;
        for (const auto& trans : transitions) {
            if (trans.fromState == currentStateName && trans.toState == targetStateName) {
                blendTime = trans.blendTime;
                break;
            }
        }
        transitionProgress += deltaTime / std::max(0.001f, blendTime);
        if (transitionProgress >= 1.0f) {
            currentStateName = targetStateName;
            targetStateName.clear();
            transitionProgress = 0.0f;
            isTransitioning = false;
            for (const auto& state : states) {
                if (state.name == currentStateName && state.onEnter) state.onEnter();
            }
        }
    }
    
    void StateMachineNode::addState(const std::string& name, uint32_t poseNodeId, bool isDefault) {
        State state;
        state.name = name;
        state.nodeId = poseNodeId;
        state.isDefault = isDefault;
        
        states.push_back(state);
        
        if (isDefault || currentStateName.empty()) {
            currentStateName = name;
        }
        
        // Add an input pin for this state so it can be connected
        addInput(name + " Pose", NodeSystem::DataType::Custom);
    }
    
    void StateMachineNode::addTransition(const Transition& transition) {
        transitions.push_back(transition);
    }
    
    void StateMachineNode::forceState(const std::string& stateName) {
        currentStateName = stateName;
        isTransitioning = false;
        targetStateName.clear();
    }
    
    void StateMachineNode::drawContent() {
        ImGui::Text("Current: %s", currentStateName.c_str());
        
        if (isTransitioning) {
            ImGui::Text("-> %s (%.0f%%)", targetStateName.c_str(), transitionProgress * 100.0f);
        }
        
        ImGui::Separator();
        ImGui::Text("States: %zu", states.size());
        ImGui::Text("Transitions: %zu", transitions.size());
    }
    
    // ============================================================================
    // ANIM PARAMETER NODE Implementation
    // ============================================================================
    
    AnimParameterNode::AnimParameterNode() {
        metadata.displayName = "Parameter";
        metadata.category = "Input";
        metadata.typeId = "AnimParameter";
        metadata.headerColor = COLOR_INPUT;
        setupPins();
    }
    
    void AnimParameterNode::setupPins() {
        addOutput("Value", NodeSystem::DataType::Float);
    }
    
    NodeSystem::PinValue AnimParameterNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        return NodeSystem::PinValue{}; 
    }
    
    void AnimParameterNode::drawContent() {
        InputTextString("Param Name", parameterName);
        const char* types[] = { "Float", "Bool", "Int", "Trigger" };
        int typeIdx = (int)paramType;
        if (ImGui::Combo("Type", &typeIdx, types, 4)) {
            paramType = (ParamType)typeIdx;
        }
    }

    // ============================================================================
    // BLEND SPACE 1D NODE Implementation
    // ============================================================================
    
    BlendSpace1DNode::BlendSpace1DNode() {
        metadata.displayName = "Blend Space 1D";
        metadata.category = "Blend";
        metadata.typeId = "BlendSpace1D";
        metadata.headerColor = COLOR_BLEND;
        setupPins();
    }
    
    void BlendSpace1DNode::setupPins() {
        addInput("Value", NodeSystem::DataType::Float);
        addOutput("Pose", NodeSystem::DataType::Custom);
    }
    
    PoseData BlendSpace1DNode::computePose(AnimationEvalContext& ctx) {
        float val = ctx.getFloatParam(parameterName, 0.0f);
        if (blendPoints.empty()) return PoseData{};
        return PoseData{}; // Placeholder for full logic
    }
    
    void BlendSpace1DNode::drawContent() {
        InputTextString("Parameter", parameterName);
        if (ImGui::Button("Add Point")) {
            addBlendPoint("", 0.0f);
        }
        for (size_t i = 0; i < blendPoints.size(); ++i) {
            ImGui::PushID((int)i);
            InputTextString("Clip", blendPoints[i].clipName);
            ImGui::SameLine();
            ImGui::DragFloat("Val", &blendPoints[i].paramValue, 0.1f);
            ImGui::PopID();
        }
    }

    void BlendSpace1DNode::addBlendPoint(const std::string& clip, float value) {
        blendPoints.push_back({clip, value, 0.0f});
    }

    void BlendSpace1DNode::removeBlendPoint(size_t index) {
        if (index < blendPoints.size()) {
            blendPoints.erase(blendPoints.begin() + index);
        }
    }

    // ============================================================================
    // LOGIC NODES Implementation
    // ============================================================================
    
    ConditionNode::ConditionNode() {
        metadata.displayName = "Condition";
        metadata.category = "Logic";
        metadata.typeId = "AnimCondition";
        metadata.headerColor = COLOR_LOGIC;
        setupPins();
    }
    
    void ConditionNode::setupPins() {
        addOutput("Result", NodeSystem::DataType::Bool);
    }
    
    NodeSystem::PinValue ConditionNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        return NodeSystem::PinValue{};
    }

    void ConditionNode::drawContent() {
        InputTextString("Param", parameterName);
        const char* ops[] = { ">", "<", "==", "!=", ">=", "<=" };
        int opIdx = (int)compareType;
        if (ImGui::Combo("Op", &opIdx, ops, 6)) compareType = (CompareType)opIdx;
        ImGui::DragFloat("Value", &compareValue);
    }

    PoseSwitchNode::PoseSwitchNode() {
        metadata.displayName = "Switch";
        metadata.category = "Logic";
        metadata.typeId = "PoseSwitch";
        metadata.headerColor = COLOR_LOGIC;
        setupPins();
    }
    
    void PoseSwitchNode::setupPins() {
        addInput("Index", NodeSystem::DataType::Int);
        for(int i=0; i<poseCount; ++i) addInput("Pose " + std::to_string(i), NodeSystem::DataType::Custom);
        addOutput("Pose", NodeSystem::DataType::Custom);
    }

    PoseData PoseSwitchNode::computePose(AnimationEvalContext& ctx) {
        return getInputPose(activeIndex + 1, ctx);
    }

    void PoseSwitchNode::drawContent() {
        if (ImGui::InputInt("Pose Count", &poseCount)) {
            poseCount = std::clamp(poseCount, 1, 10);
            
            // Rebuild pins safely without destroying all of them if possible. 
            // The first input is "Index". Keep it.
            if (inputs.size() > 0) {
                // Resize to poseCount + 1 (1 for Index, poseCount for Poses)
                size_t targetSize = poseCount + 1;
                if (inputs.size() > targetSize) {
                    inputs.resize(targetSize);
                } else {
                    for (int i = (int)inputs.size() - 1; i < poseCount; ++i) {
                        NodeSystem::Pin p = NodeSystem::Pin::createInput("Pose " + std::to_string(i), NodeSystem::DataType::Custom);
                        // We must assign a pin ID to prevent failure, though we don't have graph access here.
                        // Rely on save/load or graph update mechanism, but giving it a dummy ID is dangerous.
                        // Actually, animation graphs usually sweep for 0 IDs and assign them on next interact.
                        inputs.push_back(std::move(p));
                    }
                }
            } else {
                setupPins(); // fallback
            }
        }
        ImGui::SliderInt("Active", &activeIndex, 0, poseCount - 1);
    }

    // ============================================================================
    // UTILITY NODES Implementation
    // ============================================================================

    TimeNode::TimeNode() {
        metadata.displayName = "Time";
        metadata.category = "Utility";
        metadata.typeId = "AnimTime";
        metadata.headerColor = COLOR_UTILITY;
        setupPins();
    }

    void TimeNode::setupPins() {
        addOutput("Delta", NodeSystem::DataType::Float);
        addOutput("Global", NodeSystem::DataType::Float);
    }

    NodeSystem::PinValue TimeNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        return NodeSystem::PinValue{};
    }

    void TimeNode::drawContent() {
        ImGui::Text("Provides time data");
    }

    SpeedNode::SpeedNode() {
        metadata.displayName = "Speed";
        metadata.category = "Utility";
        metadata.typeId = "AnimSpeed";
        metadata.headerColor = COLOR_UTILITY;
        setupPins();
    }

    void SpeedNode::setupPins() {
        addInput("Pose", NodeSystem::DataType::Custom);
        addInput("Multiplier", NodeSystem::DataType::Float);
        addOutput("Pose", NodeSystem::DataType::Custom);
    }

    PoseData SpeedNode::computePose(AnimationEvalContext& ctx) {
        float oldDT = ctx.deltaTime;
        ctx.deltaTime *= speed;
        PoseData result = getInputPose(0, ctx);
        ctx.deltaTime = oldDT;
        return result;
    }

    void SpeedNode::drawContent() {
        ImGui::DragFloat("Multiplier", &speed, 0.1f, 0.0f, 10.0f);
    }

    PoseCacheNode::PoseCacheNode() {
        metadata.displayName = "Pose Cache";
        metadata.category = "Utility";
        metadata.typeId = "PoseCache";
        metadata.headerColor = COLOR_UTILITY;
        setupPins();
    }

    void PoseCacheNode::setupPins() {
        addInput("Pose", NodeSystem::DataType::Custom);
        addOutput("Pose", NodeSystem::DataType::Custom);
    }

    PoseData PoseCacheNode::computePose(AnimationEvalContext& ctx) {
        if (cacheValid && cacheFrame == ctx.currentFrame) return cachedPose;
        cachedPose = getInputPose(0, ctx);
        cacheValid = true;
        cacheFrame = ctx.currentFrame;
        return cachedPose;
    }

    void PoseCacheNode::drawContent() {
        ImGui::Text("Optimizes graph by caching pose");
    }

    // ============================================================================
    // FINAL POSE NODE Implementation
    // ============================================================================
    
    FinalPoseNode::FinalPoseNode() {
        metadata.displayName = "Final Pose";
        metadata.description = "Output node - connects to skeleton";
        metadata.category = "Output";
        metadata.typeId = "FinalPose";
        metadata.headerColor = COLOR_OUTPUT;
        
        setupPins();
    }
    
    void FinalPoseNode::setupPins() {
        addInput("Pose", NodeSystem::DataType::Custom);
    }
    
    PoseData FinalPoseNode::computePose(AnimationEvalContext& ctx) {
        PoseData localPose = getInputPose(0, ctx);
        if (!localPose.isValid() || !ctx.boneData) return localPose;
        
        // --- THE MISSING LINK: GLOBAL HIERARCHY RESOLUTION ---
        // We need to transform all local TRS hits into global matrices 
        // using the character's actual bone hierarchy.
        
        size_t count = localPose.boneCount();
        PoseData finalPose = localPose;
        finalPose.boneTransforms.resize(count);
        
        // Cache for global matrices during hierarchy traversal
        std::unordered_map<std::string, Matrix4x4> globalCache;
        
        // Match AnimationController logic: Use globalInverseTransform directly
        // This matrix (Scale 100/Inverse Root) combined with the AnimatedGlobal (which likely lacks Root)
        // produces the correct result in the legacy system. We mirror it here.
        Matrix4x4 globalCorrection = ctx.globalInverseTransform;

        for (const auto& [boneName, boneIndex] : ctx.boneData->boneNameToIndex) {
            // 1. Calculate the pure animated global transform (World space relative to character origin)
            // This traversal simulates the skeleton hierarchy
            Matrix4x4 animatedGlobal = resolveGlobalMatrix(
                boneName, ctx.boneData, localPose, globalCache, ctx);
            
            // 2. APPLY GLOBAL CORRECTION (Legacy Standard)
            Matrix4x4 finalMat = globalCorrection * animatedGlobal;
            
            // 3. APPLY BONE OFFSET (Bind Pose Inverse)
            // Corrects the bone from Bind Pose to Origin for skinning
            auto offsetIt = ctx.boneData->boneOffsetMatrices.find(boneName);
            if (offsetIt != ctx.boneData->boneOffsetMatrices.end()) {
                finalMat = finalMat * offsetIt->second;
            }
            
            finalPose.boneTransforms[boneIndex] = finalMat;
        }
        
        ctx.outputPose = finalPose; // Final result for the renderer
        return finalPose;
    }

    Matrix4x4 FinalPoseNode::resolveGlobalMatrix(
        const std::string& boneName, 
        const BoneData* boneData, 
        const PoseData& localPose,
        std::unordered_map<std::string, Matrix4x4>& cache,
        const AnimationEvalContext& ctx
    ) {
        // 1. Check cache
        auto it = cache.find(boneName);
        if (it != cache.end()) return it->second;
        
        // 2. Get local transform from graph result
        Matrix4x4 localMat = Matrix4x4::identity();
        auto idxIt = boneData->boneNameToIndex.find(boneName);
        if (idxIt != boneData->boneNameToIndex.end() && idxIt->second < localPose.trsTransforms.size()) {
            localMat = localPose.trsTransforms[idxIt->second].toMatrix();
        } else {
            auto extraIt = localPose.extraTransforms.find(boneName);
            if (extraIt != localPose.extraTransforms.end()) {
                 localMat = extraIt->second.toMatrix();
            } else {
                 // Match AnimationController: if animation is playing, missing tracks yield Identity,
                 // NOT Bind Pose, because bone offset and existing tracks already contain the space transforms.
                 localMat = Matrix4x4::identity();
            }
        }
        
        // 3. Get parent's global transform
        // BASE CASE: Skeleton root starts from Identity
        // 3. Get parent's global transform
        // BASE CASE: Skeleton root starts from Identity
        Matrix4x4 parentGlobal = Matrix4x4::identity(); 
        
        auto parentIt = boneData->boneParents.find(boneName);
        if (parentIt != boneData->boneParents.end()) {
            parentGlobal = resolveGlobalMatrix(parentIt->second, boneData, localPose, cache, ctx);
        }
        
        // 4. Combine: Global = ParentGlobal * Local
        Matrix4x4 globalResult = parentGlobal * localMat;
        cache[boneName] = globalResult;
        return globalResult;
    }
    
    void FinalPoseNode::drawContent() {
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Output Node");
    }
    
    // ============================================================================
    // ANIMATION NODE GRAPH Implementation
    // ============================================================================
    
    bool AnimationNodeGraph::connect(uint32_t outputPinId, uint32_t inputPinId) {
        NodeSystem::Pin* outPin = findPinById(outputPinId);
        NodeSystem::Pin* inPin = findPinById(inputPinId);
        
        if (!outPin || !inPin) return false;
        
        // Enforce output -> input direction
        if (outPin->kind == NodeSystem::PinKind::Input && inPin->kind == NodeSystem::PinKind::Output) {
            std::swap(outPin, inPin);
            std::swap(outputPinId, inputPinId);
        }
        
        // Validation
        if (outPin->kind != NodeSystem::PinKind::Output || inPin->kind != NodeSystem::PinKind::Input) {
            return false;
        }
        
        if (!outPin->canConnectTo(*inPin)) {
            return false;
        }
        
        // Remove existing links to the input pin if it does not allow multiple connections
        if (!inPin->allowMultipleConnections) {
            links.erase(std::remove_if(links.begin(), links.end(),
                [inputPinId](const NodeSystem::Link& l) { return l.endPinId == inputPinId; }),
                links.end());
        }

        NodeSystem::Link link;
        link.id = nextLinkId++;
        link.startPinId = outputPinId;
        link.endPinId = inputPinId;
        
        links.push_back(link);
        needsRebuild = true;
        return true;
    }
    
    bool AnimationNodeGraph::disconnect(uint32_t linkId) {
        auto it = std::find_if(links.begin(), links.end(), 
            [linkId](const NodeSystem::Link& l) { return l.id == linkId; });
        
        if (it != links.end()) {
            links.erase(it);
            needsRebuild = true;
            return true;
        }
        return false;
    }
    
    void AnimationNodeGraph::removeNode(uint32_t nodeId) {
        // Remove connected links
        links.erase(std::remove_if(links.begin(), links.end(),
            [this, nodeId](const NodeSystem::Link& l) {
                auto* startNode = findNodeByPinId(l.startPinId);
                auto* endNode = findNodeByPinId(l.endPinId);
                return (startNode && startNode->id == nodeId) || 
                       (endNode && endNode->id == nodeId);
            }), links.end());
        
        // Remove node
        auto it = std::find_if(nodes.begin(), nodes.end(),
            [nodeId](const std::unique_ptr<AnimNodeBase>& n) { return n->id == nodeId; });
            
        if (it != nodes.end()) {
            if (it->get() == outputNode) {
                outputNode = nullptr;
            }
            nodes.erase(it);
        }
        
        needsRebuild = true;
    }
    
    AnimNodeBase* AnimationNodeGraph::findNodeById(uint32_t id) {
        for (auto& node : nodes) {
            if (node->id == id) return node.get();
        }
        return nullptr;
    }
    
    NodeSystem::Pin* AnimationNodeGraph::findPinById(uint32_t id) {
        for (auto& node : nodes) {
            for (auto& pin : node->inputs) {
                if (pin.id == id) return &pin;
            }
            for (auto& pin : node->outputs) {
                if (pin.id == id) return &pin;
            }
        }
        return nullptr;
    }
    
    AnimNodeBase* AnimationNodeGraph::findNodeByPinId(uint32_t pinId) {
        for (auto& node : nodes) {
            for (auto& pin : node->inputs) {
                if (pin.id == pinId) return node.get();
            }
            for (auto& pin : node->outputs) {
                if (pin.id == pinId) return node.get();
            }
        }
        return nullptr;
    }
    
    PoseData AnimationNodeGraph::evaluate(float deltaTime, const BoneData& boneData) {
        evalContext.deltaTime = deltaTime;
        evalContext.globalTime += deltaTime;
        evalContext.currentFrame++;
        evalContext.boneData = &boneData;
        evalContext.graph = this;
        
        if (!outputNode) {
            return PoseData{};
        }
        
        return outputNode->computePose(evalContext);
    }
    

    std::unique_ptr<AnimNodeBase> AnimationNodeGraph::createNodeByType(const std::string& typeId) {
        if (typeId == "AnimClip") return std::make_unique<AnimClipNode>();
        if (typeId == "AnimBlend") return std::make_unique<BlendNode>();
        if (typeId == "LayeredBlend") return std::make_unique<LayeredBlendNode>();
        if (typeId == "StateMachine") return std::make_unique<StateMachineNode>();
        if (typeId == "FinalPose") return std::make_unique<FinalPoseNode>();
        if (typeId == "AnimParameter") return std::make_unique<AnimParameterNode>();
        if (typeId == "BlendSpace1D") return std::make_unique<BlendSpace1DNode>();
        if (typeId == "AnimCondition") return std::make_unique<ConditionNode>();
        if (typeId == "PoseSwitch") return std::make_unique<PoseSwitchNode>();
        if (typeId == "AnimTime") return std::make_unique<TimeNode>();
        if (typeId == "AnimSpeed") return std::make_unique<SpeedNode>();
        if (typeId == "PoseCache") return std::make_unique<PoseCacheNode>();
        if (typeId == "PoseSnapshot") return std::make_unique<PoseSnapshotNode>();
        if (typeId == "AdditiveBlend") return std::make_unique<AdditiveBlendNode>();
        
        return nullptr;
    }
    
    std::vector<std::pair<std::string, std::string>> AnimationNodeGraph::getAvailableNodeTypes() {
        return {
            // Input
            {"AnimClip", "Animation Clip"},
            {"AnimParameter", "Parameter"},
            {"PoseSnapshot", "Pose Snapshot"},
            
            // Blend
            {"AnimBlend", "Blend"},
            {"AdditiveBlend", "Additive Blend"},
            {"LayeredBlend", "Layered Blend"},
            {"BlendSpace1D", "Blend Space 1D"},
            
            // State
            {"StateMachine", "State Machine"},
            
            // Logic
            {"PoseSwitch", "Pose Switch"},
            {"AnimCondition", "Condition"},
            
            // Output
            {"FinalPose", "Final Pose"},
            
            // Utility
            {"AnimTime", "Time"},
            {"AnimSpeed", "Speed"},
            {"PoseCache", "Pose Cache"}
        };
    }
    
    void AnimationNodeGraph::saveToJson(nlohmann::json& j) const {
        j["version"] = 1;
        j["nextNodeId"] = nextNodeId;
        j["nextPinId"] = nextPinId;
        j["nextLinkId"] = nextLinkId;
        
        // Save nodes
        j["nodes"] = nlohmann::json::array();
        for (const auto& node : nodes) {
            nlohmann::json nodeJson;
            nodeJson["id"] = node->id;
            nodeJson["type"] = node->getTypeId();
            nodeJson["x"] = node->x;
            nodeJson["y"] = node->y;
            nodeJson["name"] = node->metadata.displayName;
            
            // Save Pin IDs
            nlohmann::json inputsJson = nlohmann::json::array();
            for(const auto& pin : node->inputs) inputsJson.push_back(pin.id);
            nodeJson["inputIDs"] = inputsJson;
            
            nlohmann::json outputsJson = nlohmann::json::array();
            for(const auto& pin : node->outputs) outputsJson.push_back(pin.id);
            nodeJson["outputIDs"] = outputsJson;
            
            // Allow node to save custom data
            node->onSave(nodeJson);
            
            j["nodes"].push_back(nodeJson);
        }
        
        // Save links
        j["links"] = nlohmann::json::array();
        for (const auto& link : links) {
            nlohmann::json linkJson;
            linkJson["id"] = link.id;
            linkJson["start"] = link.startPinId;
            linkJson["end"] = link.endPinId;
            j["links"].push_back(linkJson);
        }
    }
    
    void AnimationNodeGraph::loadFromJson(const nlohmann::json& j) {
        nodes.clear();
        links.clear();
        
        nextNodeId = j.value("nextNodeId", 1u);
        nextPinId = j.value("nextPinId", 1u);
        nextLinkId = j.value("nextLinkId", 1u);
        
        // Load nodes
        if (j.contains("nodes")) {
            for (const auto& nodeJson : j["nodes"]) {
                std::string typeId = nodeJson.value("type", "");
                auto node = createNodeByType(typeId);
                if (node) {
                    node->id = nodeJson.value("id", 0u);
                    node->x = nodeJson.value("x", 0.0f);
                    node->y = nodeJson.value("y", 0.0f);
                    
                    // Restore Pin IDs if available
                    if (nodeJson.contains("inputIDs")) {
                        auto ids = nodeJson["inputIDs"];
                        for(size_t i=0; i<node->inputs.size() && i<ids.size(); ++i) {
                            node->inputs[i].id = ids[i];
                            node->inputs[i].nodeId = node->id;
                        }
                    } else {
                        // Fallback (legacy/fresh)
                        for (auto& pin : node->inputs) {
                            pin.id = nextPinId++;
                            pin.nodeId = node->id;
                        }
                    }
                    
                    if (nodeJson.contains("outputIDs")) {
                        auto ids = nodeJson["outputIDs"];
                        for(size_t i=0; i<node->outputs.size() && i<ids.size(); ++i) {
                            node->outputs[i].id = ids[i];
                            node->outputs[i].nodeId = node->id;
                        }
                    } else {
                        // Fallback
                        for (auto& pin : node->outputs) {
                            pin.id = nextPinId++;
                            pin.nodeId = node->id;
                        }
                    }
                    
                    // Allow node to load custom data
                    node->onLoad(nodeJson);
                    
                    if (typeId == "FinalPose") {
                        outputNode = static_cast<FinalPoseNode*>(node.get());
                    }
                    
                    uint32_t currentId = node->id;
                    nodes.push_back(std::move(node));
                    
                    nextNodeId = std::max(nextNodeId, currentId + 1);
                }
            }
        }
        
        // Load links
        if (j.contains("links")) {
            for (const auto& linkJson : j["links"]) {
                NodeSystem::Link link;
                link.id = linkJson.value("id", 0u);
                link.startPinId = linkJson.value("start", 0u);
                link.endPinId = linkJson.value("end", 0u);
                links.push_back(link);
                
                nextLinkId = std::max(nextLinkId, link.id + 1);
            }
        }
        
        // Recalculate nextPinId to be safe
        for(const auto& node : nodes) {
            for(const auto& pin : node->inputs) nextPinId = std::max(nextPinId, pin.id + 1);
            for(const auto& pin : node->outputs) nextPinId = std::max(nextPinId, pin.id + 1);
        }
    }

} // namespace AnimationGraph
