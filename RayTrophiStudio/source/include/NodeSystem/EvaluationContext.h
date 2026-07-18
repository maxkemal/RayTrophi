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
#include <mutex>
#include <unordered_set>
#include <limits>

        
namespace NodeSystem {

    enum class NodeEvaluationState : uint8_t {
        Idle = 0,
        Running,
        Completed,
        Cached,
        Failed
    };

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
            const uint64_t key = makeCacheKey(nodeId, outputIndex);
            auto it = cache_.find(key);
            if (it != cache_.end()) {
                cacheLastUse_[key] = ++cacheUseCounter_;
                return it->second;
            }
            return PinValue{}; // monostate
        }
        
        /**
         * @brief Store computed value in cache
         */
        void setCachedValue(uint32_t nodeId, int outputIndex, const PinValue& value) {
            const uint64_t key = makeCacheKey(nodeId, outputIndex);
            cache_[key] = value;
            cacheLastUse_[key] = ++cacheUseCounter_;
        }
        
        /**
         * @brief Clear all cached values
         */
        void clearCache() {
            cache_.clear();
            cacheLastUse_.clear();
        }
        
        /**
         * @brief Clear cache for a specific node
         */
        void clearNodeCache(uint32_t nodeId) {
            for (auto it = cache_.begin(); it != cache_.end(); ) {
                if ((it->first >> 32) == nodeId) {
                    cacheLastUse_.erase(it->first);
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
         * @brief Total node count expected to be visited this evaluation pass
         * (set once up front so per-node completion can derive a fraction).
         */
        void setTotalNodes(int total) {
            totalNodes_.store(total);
            completedNodes_.store(0);
            progress_.store(0.0f);
            currentNodeId_.store(0);
            activeNodeStack_.clear();
            std::lock_guard<std::mutex> lock(nodeStatesMutex_);
            nodeStates_.clear();
        }

        /**
         * @brief Mark a node as the one currently being computed (read by the
         * node-editor UI to draw a per-node "active" indicator).
         */
        void beginNode(uint32_t nodeId) {
            activeNodeStack_.push_back(nodeId);
            currentNodeId_.store(nodeId);
            std::lock_guard<std::mutex> lock(nodeStatesMutex_);
            nodeStates_[nodeId] = NodeEvaluationState::Running;
        }

        /**
         * @brief Mark the current node as finished; advances the coarse
         * completed/total progress fraction.
         */
        void endNode() {
            uint32_t finishedNode = 0;
            if (!activeNodeStack_.empty()) {
                finishedNode = activeNodeStack_.back();
                activeNodeStack_.pop_back();
            }
            if (finishedNode != 0) {
                std::lock_guard<std::mutex> lock(nodeStatesMutex_);
                auto it = nodeStates_.find(finishedNode);
                if (it == nodeStates_.end() || it->second != NodeEvaluationState::Failed) {
                    nodeStates_[finishedNode] = NodeEvaluationState::Completed;
                }
            }
            currentNodeId_.store(activeNodeStack_.empty() ? 0u : activeNodeStack_.back());

            int total = totalNodes_.load();
            if (total > 0) {
                int done = completedNodes_.fetch_add(1) + 1;
                setProgress(static_cast<float>(done) / static_cast<float>(total));
            }
        }

        size_t approximateImageCacheBytes() const {
            std::unordered_set<const void*> seen;
            size_t bytes = 0;
            for (const auto& [key, value] : cache_) {
                (void)key;
                const auto* image = std::get_if<Image2DData>(&value);
                if (!image || !image->data || !seen.insert(image->data.get()).second) continue;
                bytes += image->data->size() * sizeof(float);
            }
            return bytes;
        }

        // Budgeted persistent preview cache. Evicts least-recently-used image
        // outputs but keeps the selected node's payload pinned. Scalar values are
        // tiny and remain cached.
        void enforceImageCacheBudget(size_t maxBytes, uint32_t protectedNodeId = 0) {
            while (approximateImageCacheBytes() > maxBytes) {
                auto victim = cache_.end();
                uint64_t oldestUse = (std::numeric_limits<uint64_t>::max)();
                for (auto it = cache_.begin(); it != cache_.end(); ++it) {
                    if ((it->first >> 32) == protectedNodeId) continue;
                    const auto* image = std::get_if<Image2DData>(&it->second);
                    if (!image || !image->data) continue;
                    const auto useIt = cacheLastUse_.find(it->first);
                    const uint64_t use = useIt != cacheLastUse_.end() ? useIt->second : 0;
                    if (use < oldestUse) {
                        oldestUse = use;
                        victim = it;
                    }
                }
                // A selected RGBA payload can exceed the whole budget by itself
                // (for example an 8K erosion map). Prefer keeping it, but never
                // let the protected entry turn the budget into an unbounded pin.
                if (victim == cache_.end() && protectedNodeId != 0) {
                    for (auto it = cache_.begin(); it != cache_.end(); ++it) {
                        const auto* image = std::get_if<Image2DData>(&it->second);
                        if (!image || !image->data) continue;
                        const auto useIt = cacheLastUse_.find(it->first);
                        const uint64_t use = useIt != cacheLastUse_.end() ? useIt->second : 0;
                        if (use < oldestUse) {
                            oldestUse = use;
                            victim = it;
                        }
                    }
                }
                if (victim == cache_.end()) break;
                cacheLastUse_.erase(victim->first);
                cache_.erase(victim);
            }
        }

        uint32_t getCurrentNodeId() const { return currentNodeId_.load(); }

        void markNodeCached(uint32_t nodeId) {
            std::lock_guard<std::mutex> lock(nodeStatesMutex_);
            auto it = nodeStates_.find(nodeId);
            if (it == nodeStates_.end() || it->second == NodeEvaluationState::Idle) {
                nodeStates_[nodeId] = NodeEvaluationState::Cached;
            }
        }

        void markNodeFailed(uint32_t nodeId) {
            std::lock_guard<std::mutex> lock(nodeStatesMutex_);
            nodeStates_[nodeId] = NodeEvaluationState::Failed;
        }

        NodeEvaluationState getNodeState(uint32_t nodeId) const {
            std::lock_guard<std::mutex> lock(nodeStatesMutex_);
            auto it = nodeStates_.find(nodeId);
            return it != nodeStates_.end() ? it->second : NodeEvaluationState::Idle;
        }

        /**
         * @brief Report FRACTIONAL progress (0..1) of the node currently being
         * computed (between its beginNode()/endNode() calls). Blends with the
         * coarse node-count progress so a single long-running node (e.g. an
         * erosion droplet loop) shows visible movement instead of the progress
         * bar sitting still for the node's entire duration. No-op if
         * setTotalNodes() was never called (e.g. synchronous non-terrain graphs).
         */
        void reportNodeProgress(float fraction) {
            int total = totalNodes_.load();
            if (total <= 0) return;
            int completed = completedNodes_.load();
            float frac = std::clamp(fraction, 0.0f, 1.0f);
            setProgress((static_cast<float>(completed) + frac) / static_cast<float>(total));
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
            markNodeFailed(nodeId);
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
        mutable std::unordered_map<uint64_t, uint64_t> cacheLastUse_;
        mutable uint64_t cacheUseCounter_ = 0;
        std::vector<EvaluationError> errors_;
        
        std::atomic<float> progress_{0.0f};
        std::atomic<bool> cancelled_{false};
        std::atomic<uint32_t> currentNodeId_{0};
        std::atomic<int> totalNodes_{0};
        std::atomic<int> completedNodes_{0};
        std::vector<uint32_t> activeNodeStack_;
        mutable std::mutex nodeStatesMutex_;
        std::unordered_map<uint32_t, NodeEvaluationState> nodeStates_;
        
        std::chrono::high_resolution_clock::time_point startTime_;
    };

    // ============================================================================
    // NODE GROUP (Visual Organization)
    // ============================================================================
    
    enum class LayerPortDirection : uint8_t {
        Input = 0,
        Output = 1
    };

    // Persistent interface contract for a semantic graph layer. Endpoint ids
    // are refreshed when a boundary link is spliced, while id/name survive so
    // the artist-facing port does not disappear because topology changed.
    struct LayerInterfacePort {
        uint32_t id = 0;
        LayerPortDirection direction = LayerPortDirection::Input;
        std::string name;
        uint32_t internalPinId = 0;
        uint32_t externalPinId = 0;
        DataType dataType = DataType::None;
        ImageSemantic imageSemantic = ImageSemantic::Generic;
        int imageChannels = 1;
        bool connected = false;
    };

    /**
     * @brief Visual grouping plus persistent layer interface contract
     */
    struct NodeGroup {
        uint32_t id = 0;
        std::string name;
        std::string comment;            ///< Optional description
        
        ImVec2 position{0, 0};          ///< Top-left corner
        ImVec2 size{200, 150};          ///< Dimensions
        ImU32 color = IM_COL32(80, 80, 100, 100);
        
        std::vector<uint32_t> nodeIds;  ///< Contained nodes
        std::vector<LayerInterfacePort> interfacePorts;
        uint32_t nextInterfacePortId = 1;
        bool publicationEnabled = true; ///< Enables this layer's explicit sink/output publication
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

