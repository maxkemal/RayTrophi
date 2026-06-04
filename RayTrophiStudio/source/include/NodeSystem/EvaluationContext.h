/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          EvaluationContext.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once

/**
 * @file EvaluationContext.h
 * @brief Pull-based evaluation engine for node graphs
 * 
 * Inspired by Gaea's lazy evaluation model:
 * - Output nodes request data from inputs
 * - Results are cached to prevent redundant computation
 * - Domain context passed via std::any for flexibility
 */

#include "NodeCore.h"
#include <unordered_map>
#include <functional>
#include <any>
#include <atomic>
#include <chrono>
#include <stdexcept>
#include <algorithm>

        
namespace NodeSystem {

    // Forward declarations
    struct Node;
    class GraphBase;

    // ============================================================================
    // EVALUATION CONTEXT
    // ============================================================================
    
    /**
     * @brief Context for evaluating a node graph
     * 
     * Holds cached values, domain-specific data, and progress tracking.
     * A new context should be created for each full evaluation pass.
     */
    class EvaluationContext {
    public:
        // Constructor
        explicit EvaluationContext(class GraphBase* graph) : graph_(graph) {}
        
        // ========================================================================
        // DOMAIN CONTEXT
        // ========================================================================
        
        /**
         * @brief Set domain-specific context (TerrainObject*, MaterialContext*, etc.)
         */
        template<typename T>
        void setDomainContext(T* ctx) {
            domainContext_ = ctx;
        }
        
        /**
         * @brief Get domain-specific context
         * @throws std::bad_any_cast if wrong type
         */
        template<typename T>
        T* getDomainContext() const {
            return std::any_cast<T*>(domainContext_);
        }
        
        /**
         * @brief Check if domain context is set and matches type
         */
        template<typename T>
        bool hasDomainContext() const {
            try {
                std::any_cast<T*>(domainContext_);
                return true;
            } catch (...) {
                return false;
            }
        }
        
        // ========================================================================
        // VALUE CACHE
        // ========================================================================
        
        /**
         * @brief Check if an output pin has a cached value
         */
        bool hasCachedValue(uint32_t nodeId, int outputIndex) const {
            return cache_.find(makeCacheKey(nodeId, outputIndex)) != cache_.end();
        }
        
        /**
         * @brief Get cached value for an output pin
         * @returns Empty PinValue (monostate) if not cached
         */
        PinValue getCachedValue(uint32_t nodeId, int outputIndex) const {
            auto it = cache_.find(makeCacheKey(nodeId, outputIndex));
            if (it != cache_.end()) {
                return it->second;
            }
            return PinValue{}; // monostate
        }
        
        /**
         * @brief Store computed value in cache
         */
        void setCachedValue(uint32_t nodeId, int outputIndex, const PinValue& value) {
            cache_[makeCacheKey(nodeId, outputIndex)] = value;
        }
        
        /**
         * @brief Clear all cached values
         */
        void clearCache() {
            cache_.clear();
        }
        
        /**
         * @brief Clear cache for a specific node
         */
        void clearNodeCache(uint32_t nodeId) {
            for (auto it = cache_.begin(); it != cache_.end(); ) {
                if ((it->first >> 32) == nodeId) {
                    it = cache_.erase(it);
                } else {
                    ++it;
                }
            }
        }
        
        // ========================================================================
        // PROGRESS & CANCELLATION
        // ========================================================================
        
        /**
         * @brief Report progress (0.0 to 1.0)
         */
        void setProgress(float progress) {
            progress_.store(std::clamp(progress, 0.0f, 1.0f));
        }
        
        float getProgress() const {
            return progress_.load();
        }
        
        /**
         * @brief Request cancellation of evaluation
         */
        void requestCancel() {
            cancelled_.store(true);
        }
        
        bool isCancelled() const {
            return cancelled_.load();
        }
        
        // ========================================================================
        // ERROR HANDLING
        // ========================================================================
        
        /**
         * @brief Record an error during evaluation
         */
        void addError(uint32_t nodeId, const std::string& message) {
            errors_.push_back({ nodeId, message });
        }
        
        struct EvaluationError {
            uint32_t nodeId;
            std::string message;
        };
        
        const std::vector<EvaluationError>& getErrors() const {
            return errors_;
        }
        
        bool hasErrors() const {
            return !errors_.empty();
        }
        
        void clearErrors() {
            errors_.clear();
        }
        
        // ========================================================================
        // TIMING
        // ========================================================================
        
        void startTiming() {
            startTime_ = std::chrono::high_resolution_clock::now();
        }
        
        double getElapsedMs() const {
            auto now = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::milli>(now - startTime_).count();
        }
        
        // ========================================================================
        // GRAPH ACCESS
        // ========================================================================
        
        GraphBase* getGraph() const { return graph_; }
        
    private:
        static uint64_t makeCacheKey(uint32_t nodeId, int outputIndex) {
            return (static_cast<uint64_t>(nodeId) << 32) | static_cast<uint32_t>(outputIndex);
        }
        
        GraphBase* graph_ = nullptr;
        std::any domainContext_;
        std::unordered_map<uint64_t, PinValue> cache_;
        std::vector<EvaluationError> errors_;
        
        std::atomic<float> progress_{0.0f};
        std::atomic<bool> cancelled_{false};
        
        std::chrono::high_resolution_clock::time_point startTime_;
    };

    // ============================================================================
    // NODE GROUP (Visual Organization)
    // ============================================================================
    
    /**
     * @brief Visual grouping of nodes (Gaea's "Frame" concept)
     */
    struct NodeGroup {
        uint32_t id = 0;
        std::string name;
        std::string comment;            ///< Optional description
        
        ImVec2 position{0, 0};          ///< Top-left corner
        ImVec2 size{200, 150};          ///< Dimensions
        ImU32 color = IM_COL32(80, 80, 100, 100);
        
        std::vector<uint32_t> nodeIds;  ///< Contained nodes
        bool collapsed = false;
        bool locked = false;            ///< Prevent accidental movement
        
        bool containsPoint(const ImVec2& p) const {
            return p.x >= position.x && p.x <= position.x + size.x &&
                   p.y >= position.y && p.y <= position.y + size.y;
        }
    };

    // ============================================================================
    // WIRE PORTAL (Gaea-inspired teleport connections)
    // ============================================================================
    
    /**
     * @brief Virtual connection point for long-distance links
     * 
     * Instead of drawing long bezier curves, portals can "teleport"
     * connections between distant nodes or different graph tabs.
     */
    struct WirePortal {
        uint32_t id = 0;
        std::string name;               ///< Display name
        uint32_t linkedPinId = 0;       ///< The actual pin this represents
        
        ImVec2 position{0, 0};
        ImU32 color = IM_COL32(200, 150, 50, 255);
        
        PinKind kind = PinKind::Input;  ///< Visual direction
    };

} // namespace NodeSystem

