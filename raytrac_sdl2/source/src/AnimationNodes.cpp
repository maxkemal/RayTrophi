#include "AnimationNodes.h"
#include "globals.h"
#include <algorithm>
#include <cmath>

namespace AnimationGraph {

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
        // TODO: Traverse graph to connected node
        // For now, return empty pose
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
        
        if (clipName.empty() || !ctx.clips) {
            return result;
        }
        
        // Find the clip
        AnimationClip* clip = nullptr;
        for (auto& c : *ctx.clips) {
            if (c.name == clipName) {
                clip = &c;
                break;
            }
        }
        
        if (!clip || !clip->sourceData) {
            return result;
        }
        
        // Update time
        if (isPlaying) {
            currentTime += ctx.deltaTime * playbackSpeed;
            
            // Handle looping
            float duration = clip->getDurationInSeconds();
            if (duration > 0.0f) {
                if (loop) {
                    currentTime = fmodf(currentTime, duration);
                    if (currentTime < 0) currentTime += duration;
                } else {
                    currentTime = std::max(0.0f, std::min(currentTime, duration));
                }
            }
        }
        
        // Sample animation at current time
        if (!ctx.boneData) {
            return result;
        }
        
        float timeInTicks = currentTime * clip->ticksPerSecond;
        const AnimationData* anim = clip->sourceData;
        
        // Initialize result
        size_t boneCount = ctx.boneData->boneNameToIndex.size();
        result.boneTransforms.resize(boneCount, Matrix4x4::identity());
        result.boneNames.resize(boneCount);
        
        // Sample each bone
        for (const auto& [boneName, boneIndex] : ctx.boneData->boneNameToIndex) {
            result.boneNames[boneIndex] = boneName;
            
            // Get animated transform
            Matrix4x4 transform = Matrix4x4::identity();
            bool hasAnimation = false;
            
            // Position
            auto posIt = anim->positionKeys.find(boneName);
            if (posIt != anim->positionKeys.end() && !posIt->second.empty()) {
                const auto& keys = posIt->second;
                Vec3 pos = sampleVectorKey(keys, timeInTicks, anim->duration);
                transform = Matrix4x4::translation(pos);
                hasAnimation = true;
            }
            
            // Rotation
            auto rotIt = anim->rotationKeys.find(boneName);
            if (rotIt != anim->rotationKeys.end() && !rotIt->second.empty()) {
                const auto& keys = rotIt->second;
                Quaternion rot = sampleQuatKey(keys, timeInTicks, anim->duration);
                transform = transform * rot.toMatrix();
                hasAnimation = true;
            }
            
            // Scale
            auto sclIt = anim->scalingKeys.find(boneName);
            if (sclIt != anim->scalingKeys.end() && !sclIt->second.empty()) {
                const auto& keys = sclIt->second;
                Vec3 scl = sampleVectorKey(keys, timeInTicks, anim->duration);
                transform = transform * Matrix4x4::scaling(scl);
                hasAnimation = true;
            }
            
            if (hasAnimation) {
                // Apply offset matrix
                auto offsetIt = ctx.boneData->boneOffsetMatrices.find(boneName);
                if (offsetIt != ctx.boneData->boneOffsetMatrices.end()) {
                    result.boneTransforms[boneIndex] = transform * offsetIt->second;
                } else {
                    result.boneTransforms[boneIndex] = transform;
                }
            }
        }
        
        return result;
    }
    
    void AnimClipNode::drawContent() {
        ImGui::Text("Clip: %s", clipName.empty() ? "(none)" : clipName.c_str());
        
        ImGui::SliderFloat("Speed", &playbackSpeed, -2.0f, 2.0f);
        ImGui::Checkbox("Loop", &loop);
        
        // Playback controls
        if (ImGui::Button(isPlaying ? "||" : ">")) {
            isPlaying = !isPlaying;
        }
        ImGui::SameLine();
        if (ImGui::Button("<<")) {
            reset();
        }
        
        ImGui::Text("Time: %.2fs", currentTime);
    }
    
    // ============================================================================
    // HELPER: Sample animation keys
    // ============================================================================
    
    Vec3 sampleVectorKey(const std::vector<aiVectorKey>& keys, float time, double duration) {
        if (keys.empty()) return Vec3(0, 0, 0);
        if (keys.size() == 1) {
            return Vec3(keys[0].mValue.x, keys[0].mValue.y, keys[0].mValue.z);
        }
        
        // Wrap time
        double t = fmod((double)time, duration);
        if (t < 0) t += duration;
        
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
        
        return blendPoses(poseA, poseB, blendAlpha);
    }
    
    PoseData BlendNode::blendPoses(const PoseData& a, const PoseData& b, float t) {
        PoseData result;
        
        if (!a.isValid()) return b;
        if (!b.isValid()) return a;
        
        size_t count = std::max(a.boneTransforms.size(), b.boneTransforms.size());
        result.boneTransforms.resize(count);
        result.boneNames.resize(count);
        result.blendWeight = 1.0f;
        
        // Linear interpolation of matrices
        // NOTE: For proper animation, we should decompose to TRS and slerp rotations
        for (size_t i = 0; i < count; ++i) {
            const Matrix4x4& matA = (i < a.boneTransforms.size()) ? a.boneTransforms[i] : Matrix4x4::identity();
            const Matrix4x4& matB = (i < b.boneTransforms.size()) ? b.boneTransforms[i] : Matrix4x4::identity();
            
            // Simple matrix lerp (not ideal but fast)
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 4; ++col) {
                    result.boneTransforms[i].m[row][col] = 
                        matA.m[row][col] * (1.0f - t) + matB.m[row][col] * t;
                }
            }
            
            // Use A's bone name
            if (i < a.boneNames.size()) {
                result.boneNames[i] = a.boneNames[i];
            } else if (i < b.boneNames.size()) {
                result.boneNames[i] = b.boneNames[i];
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
        
        // Check if bone is affected
        auto isAffected = [this](const std::string& boneName) -> float {
            if (affectedBones.empty()) return 1.0f;  // All bones affected
            
            // Check if bone or parent is in list
            for (const auto& affected : affectedBones) {
                if (boneName.find(affected) != std::string::npos) {
                    return 1.0f;
                }
            }
            return 0.0f;
        };
        
        // Blend affected bones
        for (size_t i = 0; i < result.boneTransforms.size(); ++i) {
            if (i >= layerPose.boneTransforms.size()) break;
            
            std::string boneName = (i < result.boneNames.size()) ? result.boneNames[i] : "";
            float weight = isAffected(boneName) * alpha;
            
            if (weight > 0.0f) {
                // Blend this bone
                const Matrix4x4& base = result.boneTransforms[i];
                const Matrix4x4& layer = layerPose.boneTransforms[i];
                
                for (int row = 0; row < 4; ++row) {
                    for (int col = 0; col < 4; ++col) {
                        result.boneTransforms[i].m[row][col] = 
                            base.m[row][col] * (1.0f - weight) + layer.m[row][col] * weight;
                    }
                }
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
        // Check for transitions
        if (!isTransitioning) {
            checkTransitions(ctx);
        }
        
        // Update transition progress
        if (isTransitioning) {
            updateTransition(ctx.deltaTime);
        }
        
        // Get current state pose
        // TODO: Look up and evaluate the pose node for current state
        
        PoseData result;
        return result;
    }
    
    void StateMachineNode::checkTransitions(AnimationEvalContext& ctx) {
        for (const auto& trans : transitions) {
            if (trans.fromState == currentStateName) {
                if (trans.evaluate(ctx)) {
                    // Start transition
                    targetStateName = trans.toState;
                    transitionProgress = 0.0f;
                    isTransitioning = true;
                    
                    // Fire exit event
                    for (const auto& state : states) {
                        if (state.name == currentStateName && state.onExit) {
                            state.onExit();
                        }
                    }
                    
                    SCENE_LOG_INFO("[StateMachine] Transition: " + currentStateName + " -> " + targetStateName);
                    break;
                }
            }
        }
    }
    
    void StateMachineNode::updateTransition(float deltaTime) {
        // Find transition to get blend time
        float blendTime = 0.3f;  // Default
        for (const auto& trans : transitions) {
            if (trans.fromState == currentStateName && trans.toState == targetStateName) {
                blendTime = trans.blendTime;
                break;
            }
        }
        
        transitionProgress += deltaTime / blendTime;
        
        if (transitionProgress >= 1.0f) {
            // Transition complete
            currentStateName = targetStateName;
            targetStateName.clear();
            transitionProgress = 0.0f;
            isTransitioning = false;
            
            // Fire enter event
            for (const auto& state : states) {
                if (state.name == currentStateName && state.onEnter) {
                    state.onEnter();
                }
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
        PoseData result = getInputPose(0, ctx);
        ctx.outputPose = result;  // Store in context
        return result;
    }
    
    void FinalPoseNode::drawContent() {
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Output Node");
    }
    
    // ============================================================================
    // ANIMATION NODE GRAPH Implementation
    // ============================================================================
    
    bool AnimationNodeGraph::connect(uint32_t outputPinId, uint32_t inputPinId) {
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
                auto* node = findNodeByPinId(l.startPinId);
                if (node && node->id == nodeId) return true;
                node = findNodeByPinId(l.endPinId);
                if (node && node->id == nodeId) return true;
                return false;
            }), links.end());
        
        // Remove node
        nodes.erase(std::remove_if(nodes.begin(), nodes.end(),
            [nodeId](const std::unique_ptr<AnimNodeBase>& n) { return n->id == nodeId; }), nodes.end());
        
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
        // Add more node types...
        
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
        
        // Save nodes
        j["nodes"] = nlohmann::json::array();
        for (const auto& node : nodes) {
            nlohmann::json nodeJson;
            nodeJson["id"] = node->id;
            nodeJson["type"] = node->getTypeId();
            nodeJson["x"] = node->x;
            nodeJson["y"] = node->y;
            nodeJson["name"] = node->metadata.displayName;
            // TODO: Save node-specific properties
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
        
        // Load nodes
        if (j.contains("nodes")) {
            for (const auto& nodeJson : j["nodes"]) {
                std::string typeId = nodeJson.value("type", "");
                auto node = createNodeByType(typeId);
                if (node) {
                    node->id = nodeJson.value("id", 0u);
                    node->x = nodeJson.value("x", 0.0f);
                    node->y = nodeJson.value("y", 0.0f);
                    
                    // Reassign pin IDs
                    for (auto& pin : node->inputs) {
                        pin.id = nextPinId++;
                        pin.nodeId = node->id;
                    }
                    for (auto& pin : node->outputs) {
                        pin.id = nextPinId++;
                        pin.nodeId = node->id;
                    }
                    
                    if (typeId == "FinalPose") {
                        outputNode = static_cast<FinalPoseNode*>(node.get());
                    }
                    
                    nodes.push_back(std::move(node));
                    
                    nextNodeId = std::max(nextNodeId, node->id + 1);
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
    }

} // namespace AnimationGraph
