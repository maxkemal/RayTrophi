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
        float uiWidth = 0.0f;       ///< Optional manual width in graph-space units
        
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
                      bool optional = false,
                      ImageUnit unit = ImageUnit::Unknown) {
            Pin pin;
            pin.name = name;
            pin.kind = PinKind::Input;
            pin.dataType = type;
            pin.imageSemantic = semantic;
            pin.optional = optional;
            pin.imageUnit = unit;
            pin.updateVisualCache();
            inputs.push_back(std::move(pin));
            return inputs.back();
        }
        
        /**
         * @brief Add an output pin
         */
        Pin& addOutput(const std::string& name, DataType type,
                       ImageSemantic semantic = ImageSemantic::Generic,
                       ImageUnit unit = ImageUnit::Unknown) {
            Pin pin;
            pin.name = name;
            pin.kind = PinKind::Output;
            pin.dataType = type;
            pin.imageSemantic = semantic;
            pin.imageUnit = unit;
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
                ctx.markNodeCached(id);
                return ctx.getCachedValue(id, outputIndex);
            }
            
            // Check if disabled (bypass)
            if (!enabled) {
                return getBypassValue(outputIndex, ctx);
            }
            
            // Compute — bracketed so the node-editor UI can show which node is
            // currently active and derive a coarse completion fraction (see
            // EvaluationContext::beginNode/endNode) without every node subtype
            // needing to report progress itself.
            ctx.beginNode(id);
            PinValue result = compute(outputIndex, ctx);
            ctx.endNode();

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
         * @brief Opt in to rendering drawContent() INSIDE the node body on the
         * canvas (NodeEditorUIV2 draws it below the pins when zoom is close
         * enough). Off by default — properties-panel-only nodes are unaffected.
         * Widgets are NOT zoom-scaled; keep content compact.
         */
        virtual bool wantsInlineContent() const { return false; }

        /// Measured screen-pixel height of the last inline drawContent() render
        /// (set by NodeEditorUIV2; consumed next frame to size the node body).
        float inlineContentHeight_ = 0.0f;

        // ---- Socket sections + inline pin values (Blender-style parameter node) ----
        // A node with 30 sockets (the uber-material Output) is unreadable as one flat pin
        // list, and pushing its values into the properties panel means the node shows a
        // column of names with no numbers on it. These three hooks let a node group its
        // input pins under headings drawn ON the node and put each unconnected pin's value
        // widget on the pin's own row. Every hook is a no-op by default, so no other graph
        // (terrain, geo, anim) changes shape.

        /// Section heading to draw ABOVE input pin `index`, or nullptr for "no new section".
        /// Only consulted for pins that are actually visible.
        virtual const char* inputSectionLabel(int index) const { (void)index; return nullptr; }

        /// Is the section that starts at input pin `index` expanded? Collapsed sections hide
        /// their unconnected pins (the node itself decides that via Pin::hidden); this only
        /// drives the header's chevron and click target.
        virtual bool isInputSectionOpen(int index) const { (void)index; return true; }

        /// User clicked the section header that starts at input pin `index`.
        virtual void toggleInputSection(int index) { (void)index; }

        /// Optional ENABLE toggle for the section starting at input pin `index` (nullptr =
        /// none). Drawn as a small checkbox right-aligned in the section's header row; the
        /// editor sets `dirty` when it flips. For sections that ARE a feature (a thin-shell
        /// bubble, a coat) rather than a bundle of always-on parameters — without this, the
        /// enable flag needs its own row somewhere else, under a second copy of the same
        /// heading, which is exactly the "why are there two Bubbles" confusion it replaces.
        virtual bool* inputSectionToggle(int index) { (void)index; return nullptr; }

        /// Does the section starting at input pin `index` have overflow parameters that don't
        /// fit on pin rows? Advertised as a "..." button on the section's own header; clicking
        /// opens a popup whose body is drawInputSectionExtra. This keeps a group's parameters
        /// in ONE place — a shared "advanced" block elsewhere on the node splits the group in
        /// two under the same heading, and which copy is authoritative becomes a guess.
        virtual bool inputSectionHasExtra(int index) const { (void)index; return false; }

        /// Popup body for the section's overflow parameters. Same contract as drawContent:
        /// set `dirty` yourself when a value changes.
        virtual void drawInputSectionExtra(int index) { (void)index; }

        /// Draw the value editor for UNCONNECTED input pin `index`, right-aligned on the
        /// pin's row, within `width` screen px. Return true if the value changed.
        /// Only called when wantsInlinePinWidgets() is on and the zoom is near 1:1.
        virtual bool drawInputInlineWidget(int index, float width) { (void)index; (void)width; return false; }

        /// Opt in to the two hooks above. Also widens the pin rows to fit real ImGui frames.
        virtual bool wantsInlinePinWidgets() const { return false; }

        /// Pin row height in unscaled px (0 = editor default). A node drawing real widgets on
        /// its pin rows needs a taller row than a bare label does.
        virtual float pinRowHeight() const { return 0.0f; }


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

