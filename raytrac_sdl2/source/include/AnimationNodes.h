#pragma once

/**
 * @file AnimationNodes.h
 * @brief Node-based animation control system
 * 
 *--------
 * Uses the existing NodeSystem infrastructure for visual programming
 * of character animations.
 * 
 * Node Types:
 * - Input Nodes: Animation Clip, Parameter, Pose Snapshot
 * - Blend Nodes: Blend, Additive Blend, Layered Blend
 * - State Nodes: State Machine, Transition
 * - Logic Nodes: Switch, Select, Condition
 * - Output Nodes: Final Pose, IK Target
 * - Utility Nodes: Time, Speed, Mirror, Cache
 */

#include "NodeSystem/Node.h"
#include "NodeSystem/NodeCore.h"
#include "AnimationController.h"
#include "Matrix4x4.h"
#include "Quaternion.h"
#include <vector>
#include <string>
#include <functional>
#include <assimp/anim.h>  // For aiVectorKey, aiQuatKey

namespace AnimationGraph {

    // Forward declarations for animation sampling helpers
    Vec3 sampleVectorKey(const std::vector<aiVectorKey>& keys, float time, double duration);
    Quaternion sampleQuatKey(const std::vector<aiQuatKey>& keys, float time, double duration);

    // ============================================================================
    // ANIMATION-SPECIFIC DATA TYPES
    // ============================================================================
    
    /**
     * @brief Extended data types for animation system
     */
    enum class AnimDataType : uint8_t {
        Pose = 100,          ///< Bone transforms array
        Curve,               ///< Animation curve (float over time)
        BoneReference,       ///< Single bone identifier
        BoneMask,            ///< Bone weight mask
        AnimClipRef,         ///< Reference to animation clip
        AnimState,           ///< State machine state
        Transform            ///< Single transform (pos/rot/scale)
    };
    
    /**
     * @brief Pose data - array of bone transforms
     */
    struct PoseData {
        std::vector<Matrix4x4> boneTransforms;
        std::vector<std::string> boneNames;  // For debugging
        float blendWeight = 1.0f;
        
        bool isValid() const { return !boneTransforms.empty(); }
        size_t boneCount() const { return boneTransforms.size(); }
    };
    
    /**
     * @brief Bone mask for layered blending
     */
    struct BoneMaskData {
        std::vector<float> weights;  // Per-bone weights (0-1)
        std::vector<std::string> boneNames;
        std::string maskName;
        
        float getWeight(size_t boneIndex) const {
            return (boneIndex < weights.size()) ? weights[boneIndex] : 1.0f;
        }
    };
    
    /**
     * @brief Animation curve data
     */
    struct CurveData {
        std::vector<std::pair<float, float>> keyframes;  // (time, value)
        bool loop = false;
        
        float evaluate(float time) const;
    };
    
    // ============================================================================
    // ANIMATION EVALUATION CONTEXT
    // ============================================================================
    
    /**
     * @brief Animation-specific evaluation context
     */
    struct AnimationEvalContext {
        // Time info
        float deltaTime = 0.0f;
        float globalTime = 0.0f;
        int currentFrame = 0;
        
        // Bone data reference
        const BoneData* boneData = nullptr;
        
        // Animation clips (from scene)
        std::vector<AnimationClip>* clips = nullptr;
        
        // Parameters (from user/gameplay)
        std::unordered_map<std::string, float> floatParams;
        std::unordered_map<std::string, bool> boolParams;
        std::unordered_map<std::string, int> intParams;
        std::unordered_map<std::string, std::string> triggerParams;
        
        // Output
        PoseData outputPose;
        RootMotionDelta rootMotion;
        
        // Helper methods
        float getFloatParam(const std::string& name, float defaultValue = 0.0f) const {
            auto it = floatParams.find(name);
            return (it != floatParams.end()) ? it->second : defaultValue;
        }
        
        bool getBoolParam(const std::string& name, bool defaultValue = false) const {
            auto it = boolParams.find(name);
            return (it != boolParams.end()) ? it->second : defaultValue;
        }
        
        int getIntParam(const std::string& name, int defaultValue = 0) const {
            auto it = intParams.find(name);
            return (it != intParams.end()) ? it->second : defaultValue;
        }
        
        bool consumeTrigger(const std::string& name) {
            auto it = triggerParams.find(name);
            if (it != triggerParams.end()) {
                triggerParams.erase(it);
                return true;
            }
            return false;
        }
    };
    
    // ============================================================================
    // BASE ANIMATION NODE
    // ============================================================================
    
    /**
     * @brief Base class for all animation nodes
     */
    class AnimNodeBase : public NodeSystem::NodeBase {
    public:
        // Category colors for visual distinction
        static constexpr ImU32 COLOR_INPUT = IM_COL32(80, 140, 80, 255);     // Green
        static constexpr ImU32 COLOR_BLEND = IM_COL32(80, 80, 160, 255);     // Blue
        static constexpr ImU32 COLOR_STATE = IM_COL32(160, 80, 80, 255);     // Red
        static constexpr ImU32 COLOR_LOGIC = IM_COL32(140, 120, 80, 255);    // Yellow
        static constexpr ImU32 COLOR_OUTPUT = IM_COL32(120, 80, 140, 255);   // Purple
        static constexpr ImU32 COLOR_UTILITY = IM_COL32(100, 100, 100, 255); // Gray
        
        // Animation-specific compute
        virtual PoseData computePose(AnimationEvalContext& ctx) {
            return PoseData{};
        }
        
        // Get pose from input pin
        PoseData getInputPose(int inputIndex, AnimationEvalContext& ctx);
        
        // For compatibility with generic node system
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override {
            // Animation nodes use computePose instead
            return NodeSystem::PinValue{};
        }
    };
    
    // ============================================================================
    // INPUT NODES
    // ============================================================================
    
    /**
     * @brief Animation Clip Player Node
     * 
     * Plays a single animation clip with time control.
     * 
     * Inputs:
     * - Time Override (Float, optional)
     * - Speed (Float, optional)
     * 
     * Outputs:
     * - Pose
     * - Normalized Time (0-1)
     * - Remaining Time
     */
    class AnimClipNode : public AnimNodeBase {
    public:
        // Properties
        std::string clipName;
        float playbackSpeed = 1.0f;
        bool loop = true;
        float startTime = 0.0f;
        
        // Runtime state
        float currentTime = 0.0f;
        bool isPlaying = true;
        
        AnimClipNode();
        void setupPins();
        
        PoseData computePose(AnimationEvalContext& ctx) override;
        void drawContent() override;
        
        std::string getTypeId() const override { return "AnimClip"; }
        
        // Playback control
        void play() { isPlaying = true; }
        void pause() { isPlaying = false; }
        void reset() { currentTime = startTime; }
    };
    
    /**
     * @brief Parameter Input Node
     * 
     * Exposes a parameter value from the animation context.
     * Used to drive blends and state transitions.
     */
    class AnimParameterNode : public AnimNodeBase {
    public:
        enum class ParamType { Float, Bool, Int, Trigger };
        
        std::string parameterName;
        ParamType paramType = ParamType::Float;
        float defaultFloat = 0.0f;
        bool defaultBool = false;
        int defaultInt = 0;
        
        AnimParameterNode();
        void setupPins();
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        
        std::string getTypeId() const override { return "AnimParameter"; }
    };
    
    /**
     * @brief Pose Snapshot Node
     * 
     * Captures the current pose for use in blending.
     * Useful for transitions from ragdoll or procedural animation.
     */
    class PoseSnapshotNode : public AnimNodeBase {
    public:
        PoseData capturedPose;
        bool hasCapture = false;
        
        PoseSnapshotNode();
        void setupPins();
        
        PoseData computePose(AnimationEvalContext& ctx) override;
        void drawContent() override;
        
        void capture(const PoseData& pose);
        void clear();
        
        std::string getTypeId() const override { return "PoseSnapshot"; }
    };
    
    // ============================================================================
    // BLEND NODES
    // ============================================================================
    
    /**
     * @brief Two-Pose Blend Node
     * 
     * Blends between two poses based on alpha value.
     * 
     * Inputs:
     * - Pose A
     * - Pose B
     * - Alpha (Float, 0-1)
     * 
     * Output:
     * - Blended Pose
     */
    class BlendNode : public AnimNodeBase {
    public:
        float alpha = 0.5f;
        bool useInputAlpha = true;
        
        BlendNode();
        void setupPins();
        
        PoseData computePose(AnimationEvalContext& ctx) override;
        void drawContent() override;
        
        std::string getTypeId() const override { return "AnimBlend"; }
        
    private:
        PoseData blendPoses(const PoseData& a, const PoseData& b, float t);
    };
    
    /**
     * @brief Additive Blend Node
     * 
     * Applies additive animation on top of base pose.
     * 
     * Inputs:
     * - Base Pose
     * - Additive Pose
     * - Alpha (influence)
     */
    class AdditiveBlendNode : public AnimNodeBase {
    public:
        float alpha = 1.0f;
        
        AdditiveBlendNode();
        void setupPins();
        
        PoseData computePose(AnimationEvalContext& ctx) override;
        void drawContent() override;
        
        std::string getTypeId() const override { return "AdditiveBlend"; }
    };
    
    /**
     * @brief Layered Blend Per Bone Node
     * 
     * Blends poses with per-bone masks.
     * Perfect for upper/lower body independent animation.
     * 
     * Inputs:
     * - Base Pose
     * - Layer Pose
     * - Bone Mask
     * - Alpha
     */
    class LayeredBlendNode : public AnimNodeBase {
    public:
        float alpha = 1.0f;
        std::vector<std::string> affectedBones;  // If empty, use mask input
        
        LayeredBlendNode();
        void setupPins();
        
        PoseData computePose(AnimationEvalContext& ctx) override;
        void drawContent() override;
        
        std::string getTypeId() const override { return "LayeredBlend"; }
    };
    
    /**
     * @brief Blend Space 1D Node
     * 
     * Blends between multiple animations based on a single parameter.
     * Like Unreal's BlendSpace1D.
     * 
     * Inputs:
     * - Parameter Value (Float)
     * 
     * Internal:
     * - List of clips with parameter values
     */
    class BlendSpace1DNode : public AnimNodeBase {
    public:
        struct BlendPoint {
            std::string clipName;
            float paramValue;
            float currentTime = 0.0f;  // Per-clip playback state
        };
        
        std::vector<BlendPoint> blendPoints;
        std::string parameterName;
        bool syncAnimations = true;  // Keep animations in sync
        
        BlendSpace1DNode();
        void setupPins();
        
        PoseData computePose(AnimationEvalContext& ctx) override;
        void drawContent() override;
        
        void addBlendPoint(const std::string& clip, float value);
        void removeBlendPoint(size_t index);
        
        std::string getTypeId() const override { return "BlendSpace1D"; }
    };
    
    // ============================================================================
    // STATE MACHINE NODES
    // ============================================================================
    
    /**
     * @brief State Machine Node
     * 
     * Contains states and transitions for complex animation logic.
     * Each state outputs a pose, transitions control flow.
     */
    class StateMachineNode : public AnimNodeBase {
    public:
        struct State {
            std::string name;
            uint32_t nodeId = 0;      // Which node produces this state's pose
            bool isDefault = false;
            
            // Entry/Exit events
            std::function<void()> onEnter;
            std::function<void()> onExit;
        };
        
        struct Transition {
            std::string fromState;
            std::string toState;
            float blendTime = 0.3f;
            
            // Condition
            std::string parameterName;
            enum class ConditionType { Bool, FloatGreater, FloatLess, Trigger };
            ConditionType conditionType = ConditionType::Bool;
            float compareValue = 0.0f;
            
            bool evaluate(const AnimationEvalContext& ctx) const;
        };
        
        std::vector<State> states;
        std::vector<Transition> transitions;
        std::string currentStateName;
        std::string targetStateName;  // For transitions
        float transitionProgress = 0.0f;
        bool isTransitioning = false;
        
        StateMachineNode();
        void setupPins();
        
        PoseData computePose(AnimationEvalContext& ctx) override;
        void drawContent() override;
        
        void addState(const std::string& name, uint32_t poseNodeId, bool isDefault = false);
        void addTransition(const Transition& transition);
        void forceState(const std::string& stateName);
        
        std::string getTypeId() const override { return "StateMachine"; }
        
    private:
        void checkTransitions(AnimationEvalContext& ctx);
        void updateTransition(float deltaTime);
    };
    
    // ============================================================================
    // LOGIC NODES
    // ============================================================================
    
    /**
     * @brief Pose Switch Node
     * 
     * Selects one pose from multiple inputs based on index.
     */
    class PoseSwitchNode : public AnimNodeBase {
    public:
        int activeIndex = 0;
        int poseCount = 2;
        
        PoseSwitchNode();
        void setupPins();
        
        PoseData computePose(AnimationEvalContext& ctx) override;
        void drawContent() override;
        
        std::string getTypeId() const override { return "PoseSwitch"; }
    };
    
    /**
     * @brief Condition Node
     * 
     * Outputs bool based on parameter comparison.
     */
    class ConditionNode : public AnimNodeBase {
    public:
        enum class CompareType { Greater, Less, Equal, NotEqual, GreaterEqual, LessEqual };
        
        std::string parameterName;
        CompareType compareType = CompareType::Greater;
        float compareValue = 0.0f;
        
        ConditionNode();
        void setupPins();
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        
        std::string getTypeId() const override { return "AnimCondition"; }
    };
    
    // ============================================================================
    // OUTPUT NODES
    // ============================================================================
    
    /**
     * @brief Final Pose Output Node
     * 
     * The end of the animation graph. Only one per graph.
     * 
     * Input:
     * - Final Pose
     * 
     * Writes to AnimationEvalContext::outputPose
     */
    class FinalPoseNode : public AnimNodeBase {
    public:
        FinalPoseNode();
        void setupPins();
        
        PoseData computePose(AnimationEvalContext& ctx) override;
        void drawContent() override;
        
        std::string getTypeId() const override { return "FinalPose"; }
    };
    
    // ============================================================================
    // UTILITY NODES
    // ============================================================================
    
    /**
     * @brief Time Controller Node
     * 
     * Provides time-based outputs for driving animations.
     * 
     * Outputs:
     * - Delta Time
     * - Global Time
     * - Frame Number
     */
    class TimeNode : public AnimNodeBase {
    public:
        TimeNode();
        void setupPins();
        
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
        void drawContent() override;
        
        std::string getTypeId() const override { return "AnimTime"; }
    };
    
    /**
     * @brief Playback Speed Node
     * 
     * Modifies playback speed of input pose.
     */
    class SpeedNode : public AnimNodeBase {
    public:
        float speed = 1.0f;
        
        SpeedNode();
        void setupPins();
        
        PoseData computePose(AnimationEvalContext& ctx) override;
        void drawContent() override;
        
        std::string getTypeId() const override { return "AnimSpeed"; }
    };
    
    /**
     * @brief Pose Cache Node
     * 
     * Caches the input pose to prevent recomputation.
     * Useful when the same pose is used multiple times.
     */
    class PoseCacheNode : public AnimNodeBase {
    public:
        PoseData cachedPose;
        bool cacheValid = false;
        int cacheFrame = -1;
        
        PoseCacheNode();
        void setupPins();
        
        PoseData computePose(AnimationEvalContext& ctx) override;
        void drawContent() override;
        
        void invalidateCache() { cacheValid = false; }
        
        std::string getTypeId() const override { return "PoseCache"; }
    };
    
    // ============================================================================
    // ANIMATION GRAPH
    // ============================================================================
    
    /**
     * @brief Animation node graph manager
     * 
     * Contains all animation nodes and handles evaluation.
     */
    class AnimationNodeGraph {
    public:
        // Node management
        std::vector<std::unique_ptr<AnimNodeBase>> nodes;
        std::vector<NodeSystem::Link> links;
        
        // References
        FinalPoseNode* outputNode = nullptr;
        
        // ID generation
        uint32_t nextNodeId = 1;
        uint32_t nextPinId = 1;
        uint32_t nextLinkId = 1;
        
        // Evaluation
        AnimationEvalContext evalContext;
        bool needsRebuild = false;
        
        // ========================================================================
        // Graph Operations
        // ========================================================================
        
        template<typename T>
        T* addNode() {
            static_assert(std::is_base_of<AnimNodeBase, T>::value, "T must derive from AnimNodeBase");
            
            auto node = std::make_unique<T>();
            node->id = nextNodeId++;
            
            // Assign pin IDs
            for (auto& pin : node->inputs) {
                pin.id = nextPinId++;
                pin.nodeId = node->id;
            }
            for (auto& pin : node->outputs) {
                pin.id = nextPinId++;
                pin.nodeId = node->id;
            }
            
            T* ptr = node.get();
            nodes.push_back(std::move(node));
            
            // Track output node
            if constexpr (std::is_same_v<T, FinalPoseNode>) {
                outputNode = ptr;
            }
            
            needsRebuild = true;
            return ptr;
        }
        
        bool connect(uint32_t outputPinId, uint32_t inputPinId);
        bool disconnect(uint32_t linkId);
        void removeNode(uint32_t nodeId);
        
        AnimNodeBase* findNodeById(uint32_t id);
        NodeSystem::Pin* findPinById(uint32_t id);
        AnimNodeBase* findNodeByPinId(uint32_t pinId);
        
        // ========================================================================
        // Evaluation
        // ========================================================================
        
        /**
         * @brief Evaluate the entire graph
         * 
         * @param deltaTime Time since last frame
         * @param boneData Scene bone data
         * @return Final pose output
         */
        PoseData evaluate(float deltaTime, const BoneData& boneData);
        
        /**
         * @brief Set a float parameter
         */
        void setFloatParam(const std::string& name, float value) {
            evalContext.floatParams[name] = value;
        }
        
        /**
         * @brief Set a bool parameter
         */
        void setBoolParam(const std::string& name, bool value) {
            evalContext.boolParams[name] = value;
        }
        
        /**
         * @brief Fire a trigger
         */
        void triggerParam(const std::string& name) {
            evalContext.triggerParams[name] = name;
        }
        
        // ========================================================================
        // Serialization
        // ========================================================================
        
        void saveToJson(nlohmann::json& j) const;
        void loadFromJson(const nlohmann::json& j);
        
        // ========================================================================
        // Node Factory
        // ========================================================================
        
        static std::unique_ptr<AnimNodeBase> createNodeByType(const std::string& typeId);
        static std::vector<std::pair<std::string, std::string>> getAvailableNodeTypes();
    };

} // namespace AnimationGraph
