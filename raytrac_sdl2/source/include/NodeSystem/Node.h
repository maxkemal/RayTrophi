/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          Node.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once

/**
 * @file Node.h
 * @brief Enhanced Node base class with pull-based evaluation
 * 
 * Nodes implement the compute() method which is called when
 * downstream nodes request data. This is the Gaea-inspired
 * lazy evaluation pattern.
 */

#include "NodeCore.h"
#include "EvaluationContext.h"
#include <vector>
#include <memory>
#include <algorithm>

namespace NodeSystem {

    // ============================================================================
    // NODE BASE CLASS
    // ============================================================================
    
    /**
     * @brief Base class for all nodes in the graph system
     * 
     * Key features:
     * - Typed pins with automatic visual configuration
     * - Pull-based compute() for lazy evaluation
     * - Rich metadata for UI and organization
     * - Group membership for visual organization
     */
    class NodeBase {
    public:
        // Identity
        uint32_t id = 0;
        
        // Pins
        std::vector<Pin> inputs;
        std::vector<Pin> outputs;
        
        // Position
        float x = 0.0f;
        float y = 0.0f;
        
        // State
        bool dirty = true;          ///< Needs recomputation
        bool selected = false;      ///< Currently selected in UI
        bool collapsed = false;     ///< Minimized view
        bool enabled = true;        ///< Can be disabled to bypass
        
        // Metadata
        NodeMetadata metadata;
        
        // Visual organization
        uint32_t groupId = 0;       ///< 0 = no group
        int zOrder = 0;             ///< Drawing order
        
        // Destructor
        virtual ~NodeBase() = default;
        
        // ========================================================================
        // PIN MANAGEMENT
        // ========================================================================
        
        /**
         * @brief Add an input pin with automatic ID assignment (done by Graph)
         */
        Pin& addInput(const std::string& name, DataType type, 
                      ImageSemantic semantic = ImageSemantic::Generic,
                      bool optional = false) {
            Pin pin;
            pin.name = name;
            pin.kind = PinKind::Input;
            pin.dataType = type;
            pin.imageSemantic = semantic;
            pin.optional = optional;
            pin.updateVisualCache();
            inputs.push_back(std::move(pin));
            return inputs.back();
        }
        
        /**
         * @brief Add an output pin
         */
        Pin& addOutput(const std::string& name, DataType type,
                       ImageSemantic semantic = ImageSemantic::Generic) {
            Pin pin;
            pin.name = name;
            pin.kind = PinKind::Output;
            pin.dataType = type;
            pin.imageSemantic = semantic;
            pin.updateVisualCache();
            outputs.push_back(std::move(pin));
            return outputs.back();
        }
        
        /**
         * @brief Get input pin by index
         */
        Pin* getInputPin(size_t index) {
            return (index < inputs.size()) ? &inputs[index] : nullptr;
        }
        
        /**
         * @brief Get output pin by index
         */
        Pin* getOutputPin(size_t index) {
            return (index < outputs.size()) ? &outputs[index] : nullptr;
        }
        
        /**
         * @brief Get input pin by name
         */
        Pin* getInputPinByName(const std::string& name) {
            auto it = std::find_if(inputs.begin(), inputs.end(), 
                [&](const Pin& p) { return p.name == name; });
            return (it != inputs.end()) ? &(*it) : nullptr;
        }
        
        /**
         * @brief Get output pin by name
         */
        Pin* getOutputPinByName(const std::string& name) {
            auto it = std::find_if(outputs.begin(), outputs.end(),
                [&](const Pin& p) { return p.name == name; });
            return (it != outputs.end()) ? &(*it) : nullptr;
        }
        
        // ========================================================================
        // EVALUATION
        // ========================================================================
        
        /**
         * @brief Compute the value for a specific output pin
         * 
         * This is the core method that derived nodes must implement.
         * It requests input values from the context, performs computation,
         * and returns the result.
         * 
         * @param outputIndex Which output pin to compute (usually 0)
         * @param ctx Evaluation context with cache and domain data
         * @return Computed value for the output pin
         */
        virtual PinValue compute(int outputIndex, EvaluationContext& ctx) {
            // Default: return empty
            return PinValue{};
        }
        
        /**
         * @brief Request a value from this node's output
         * 
         * Uses caching to prevent redundant computation.
         * This is called by the evaluation system, not directly by other nodes.
         */
        PinValue requestOutput(int outputIndex, EvaluationContext& ctx) {
            // Check cache first
            if (!dirty && ctx.hasCachedValue(id, outputIndex)) {
                return ctx.getCachedValue(id, outputIndex);
            }
            
            // Check if disabled (bypass)
            if (!enabled) {
                return getBypassValue(outputIndex, ctx);
            }
            
            // Compute
            PinValue result = compute(outputIndex, ctx);
            
            // Cache result
            ctx.setCachedValue(id, outputIndex, result);
            dirty = false;
            
            return result;
        }
        
        /**
         * @brief Get bypass value when node is disabled
         * 
         * Override to provide pass-through behavior.
         * Default returns empty value.
         */
        virtual PinValue getBypassValue(int outputIndex, EvaluationContext& ctx) {
            // Default: try to pass through first matching input
            if (!inputs.empty() && outputIndex < (int)outputs.size()) {
                // Find input with matching type
                DataType outType = outputs[outputIndex].dataType;
                for (size_t i = 0; i < inputs.size(); i++) {
                    if (inputs[i].dataType == outType) {
                        return getInputValue((int)i, ctx);
                    }
                }
            }
            return PinValue{};
        }
        
        /**
         * @brief Get value from an input pin
         * 
         * This requests the value from the connected upstream node.
         */
        PinValue getInputValue(int inputIndex, EvaluationContext& ctx);
        
        // ========================================================================
        // UI CUSTOMIZATION
        // ========================================================================
        
        /**
         * @brief Draw custom content inside the node body
         * 
         * Override to add sliders, checkboxes, preview images, etc.
         * Use ImGui calls here.
         */
        virtual void drawContent() {}
        
        /**
         * @brief Draw custom header content (after title)
         */
        virtual void drawHeaderExtra() {}
        
        /**
         * @brief Get custom node width (0 = auto)
         */
        virtual float getCustomWidth() const { return 0.0f; }
        
        /**
         * @brief Called when selection state changes
         */
        virtual void onSelectionChanged(bool isSelected) {}
        
        // ========================================================================
        // SERIALIZATION HELPERS
        // ========================================================================
        
        /**
         * @brief Get node type identifier for serialization
         */
        virtual std::string getTypeId() const {
            return metadata.typeId.empty() ? "unknown" : metadata.typeId;
        }
        
        // ========================================================================
        // LEGACY COMPATIBILITY
        // ========================================================================
        
        // For backward compatibility with old NodeSystem::Node users
        std::string name;  // Deprecated: use metadata.displayName
        ImVec4 headerColor = ImVec4(0.2f, 0.3f, 0.4f, 1.0f);  // Deprecated: use metadata.headerColor
        
        // Legacy evaluate() - calls compute() for output 0
        virtual void evaluate() {
            // No-op in new system; use compute()
        }
    };

    // ============================================================================
    // HELPER TEMPLATE FOR TYPED NODES
    // ============================================================================
    
    /**
     * @brief Helper to create nodes with specific domain context
     * 
     * Usage:
     *   class MyTerrainNode : public TypedNode<TerrainObject> { ... };
     */
    template<typename DomainContextT>
    class TypedNode : public NodeBase {
    public:
        /**
         * @brief Get domain context with type safety
         */
        DomainContextT* getContext(EvaluationContext& ctx) const {
            if (ctx.hasDomainContext<DomainContextT>()) {
                return ctx.getDomainContext<DomainContextT>();
            }
            return nullptr;
        }
    };

} // namespace NodeSystem

