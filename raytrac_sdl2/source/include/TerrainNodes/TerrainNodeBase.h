#pragma once

/**
 * @file TerrainNodeBase.h
 * @brief Base class for terrain nodes using the new node system
 * 
 * This file provides TerrainNodeV2 which extends the new NodeBase
 * and provides terrain-specific functionality while maintaining
 * compatibility with the existing evaluation patterns.
 */

#include "NodeSystem/Node.h"
#include "NodeSystem/Graph.h"
#include "TerrainNodes/TerrainContext.h"
#include "TerrainManager.h"
#include <vector>

namespace TerrainNodes {

    // Forward declaration for compatibility
    enum class NodeType;

    // ============================================================================
    // TERRAIN NODE V2 BASE
    // ============================================================================
    
    /**
     * @brief Modern terrain node base class
     * 
     * Extends NodeSystem::NodeBase with terrain-specific functionality:
     * - Height/mask data caching
     * - TerrainObject access
     * - Compatibility bridge to legacy evaluateTerrain()
     */
    class TerrainNodeV2 : public NodeSystem::NodeBase {
    public:
        // Legacy compatibility
        NodeType terrainNodeType;
        
        // Height data cache (for intermediate results)
        std::vector<float> cachedHeightData;
        bool hasCachedHeight = false;
        
        // Mask data cache
        std::vector<float> cachedMaskData;
        bool hasCachedMask = false;
        
        // ========================================================================
        // PIN HELPERS
        // ========================================================================
        
        /**
         * @brief Add a height input pin
         */
        NodeSystem::Pin& addHeightInput(const std::string& pinName = "Height In", bool optional = false) {
            return addInput(pinName, NodeSystem::DataType::Image2D, 
                           NodeSystem::ImageSemantic::Height, optional);
        }
        
        /**
         * @brief Add a height output pin
         */
        NodeSystem::Pin& addHeightOutput(const std::string& pinName = "Height Out") {
            return addOutput(pinName, NodeSystem::DataType::Image2D,
                            NodeSystem::ImageSemantic::Height);
        }
        
        /**
         * @brief Add a mask input pin
         */
        NodeSystem::Pin& addMaskInput(const std::string& pinName = "Mask", bool optional = true) {
            return addInput(pinName, NodeSystem::DataType::Image2D,
                           NodeSystem::ImageSemantic::Mask, optional);
        }
        
        /**
         * @brief Add a mask output pin
         */
        NodeSystem::Pin& addMaskOutput(const std::string& pinName = "Mask Out") {
            return addOutput(pinName, NodeSystem::DataType::Image2D,
                            NodeSystem::ImageSemantic::Mask);
        }
        
        /**
         * @brief Add a float input pin with default value
         */
        NodeSystem::Pin& addFloatInput(const std::string& pinName, float defaultVal = 0.0f, bool optional = true) {
            auto& pin = addInput(pinName, NodeSystem::DataType::Float, 
                                NodeSystem::ImageSemantic::Generic, optional);
            pin.defaultValue = defaultVal;
            return pin;
        }
        
        // ========================================================================
        // INPUT DATA HELPERS
        // ========================================================================
        
        /**
         * @brief Get height data from an input pin
         * @returns Pointer to height data vector, or nullptr if not connected/invalid
         */
        const std::vector<float>* getInputHeightData(int inputIndex, NodeSystem::EvaluationContext& ctx) {
            NodeSystem::PinValue value = getInputValue(inputIndex, ctx);
            
            NodeSystem::Image2DData img;
            if (NodeSystem::tryGetImage(value, img) && img.isValid()) {
                // Store in temp for caller to use
                tempInputData_ = *img.data;
                return &tempInputData_;
            }
            return nullptr;
        }
        
        /**
         * @brief Get mask data from an input pin
         */
        const std::vector<float>* getInputMaskData(int inputIndex, NodeSystem::EvaluationContext& ctx) {
            return getInputHeightData(inputIndex, ctx); // Same logic, different semantic
        }
        
        /**
         * @brief Get float value from an input pin
         */
        float getInputFloat(int inputIndex, NodeSystem::EvaluationContext& ctx, float defaultVal = 0.0f) {
            NodeSystem::PinValue value = getInputValue(inputIndex, ctx);
            float result = defaultVal;
            NodeSystem::tryGetFloat(value, result);
            return result;
        }
        
        /**
         * @brief Get terrain from evaluation context
         */
        TerrainObject* getTerrain(NodeSystem::EvaluationContext& ctx) {
            return getTerrainFromContext(ctx);
        }
        
        /**
         * @brief Get resolution from terrain context
         */
        int getResolution(NodeSystem::EvaluationContext& ctx) {
            TerrainObject* terrain = getTerrain(ctx);
            if (terrain) {
                return terrain->heightmap.width;
            }
            return 0;
        }
        
        // ========================================================================
        // OUTPUT HELPERS
        // ========================================================================
        
        /**
         * @brief Create a height output PinValue from cached data
         */
        NodeSystem::PinValue makeHeightOutput(NodeSystem::EvaluationContext& ctx) {
            int res = getResolution(ctx);
            return NodeSystem::wrapImageValue(
                std::make_shared<std::vector<float>>(cachedHeightData),
                res, res, 1, NodeSystem::ImageSemantic::Height
            );
        }
        
        /**
         * @brief Create a mask output PinValue from cached data
         */
        NodeSystem::PinValue makeMaskOutput(NodeSystem::EvaluationContext& ctx) {
            int res = getResolution(ctx);
            return NodeSystem::wrapImageValue(
                std::make_shared<std::vector<float>>(cachedMaskData),
                res, res, 1, NodeSystem::ImageSemantic::Mask
            );
        }
        
        // ========================================================================
        // EVALUATION
        // ========================================================================
        
        /**
         * @brief Main compute method (override in derived classes)
         * 
         * Default implementation calls legacy evaluateTerrain() for compatibility.
         */
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override {
            TerrainObject* terrain = getTerrain(ctx);
            if (!terrain) {
                ctx.addError(id, "No terrain in context");
                return NodeSystem::PinValue{};
            }
            
            // Call terrain-specific evaluation (implement in derived)
            computeTerrain(terrain, ctx);
            
            // Return appropriate output based on index
            if (outputIndex == 0 && hasCachedHeight) {
                return makeHeightOutput(ctx);
            } else if (outputIndex == 1 && hasCachedMask) {
                return makeMaskOutput(ctx);
            }
            
            return NodeSystem::PinValue{};
        }
        
        /**
         * @brief Terrain-specific computation (override this instead of compute())
         */
        virtual void computeTerrain(TerrainObject* terrain, NodeSystem::EvaluationContext& ctx) {
            // Default: no-op
        }
        
        // ========================================================================
        // CACHE MANAGEMENT
        // ========================================================================
        
        void clearCache() {
            hasCachedHeight = false;
            hasCachedMask = false;
            cachedHeightData.clear();
            cachedMaskData.clear();
        }
        
    protected:
        // Temporary storage for input data (used by getInputHeightData)
        std::vector<float> tempInputData_;
    };

    // ============================================================================
    // TERRAIN GRAPH V2
    // ============================================================================
    
    /**
     * @brief Modern terrain graph using new node system
     */
    class TerrainGraphV2 : public NodeSystem::GraphBase {
    public:
        /**
         * @brief Evaluate the graph for a terrain object
         */
        void evaluateForTerrain(TerrainObject* terrain) {
            // Create context
            TerrainContext tctx(terrain);
            tctx.resolution = terrain ? terrain->heightmap.width : 0;
            
            // Create evaluation context
            NodeSystem::EvaluationContext ctx(static_cast<NodeSystem::GraphBase*>(this));
            ctx.setDomainContext(&tctx);
            
            // Run evaluation
            evaluate(ctx);
            
            // Report errors if any
            if (ctx.hasErrors()) {
                for (const auto& err : ctx.getErrors()) {
                    // Could log these or show in UI
                    // For now just marking them
                }
            }
        }
        
        /**
         * @brief Add a terrain node by type
         */
        template<typename T>
        T* addTerrainNode(float x = 0, float y = 0) {
            T* node = addNode<T>();
            node->x = x;
            node->y = y;
            return node;
        }
    };

} // namespace TerrainNodes
