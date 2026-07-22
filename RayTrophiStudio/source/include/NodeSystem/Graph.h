/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          Graph.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once

/**
 * @file Graph.h
 * @brief Enhanced Graph container with topological evaluation
 * 
 * The Graph manages nodes, links, groups, and portals.
 * It provides the pull-based evaluation traversal.
 */

#include "Node.h"
#include <algorithm>
#include <queue>
#include <unordered_set>

namespace NodeSystem {

    // ============================================================================
    // GRAPH CLASS
    // ============================================================================
    
    /**
     * @brief Container and manager for a node graph
     * 
     * Provides:
     * - Node/link/group management
     * - Topological evaluation
     * - Query utilities
     */
    class GraphBase {
    public:
        virtual ~GraphBase() = default;  // Enable polymorphic usage
        
        // Data containers
        std::vector<std::shared_ptr<NodeBase>> nodes;
        std::vector<Link> links;
        std::vector<NodeGroup> groups;
        std::vector<WirePortal> portals;
        
        // ID generators
        uint32_t nextNodeId = 1;
        uint32_t nextPinId = 1;
        uint32_t nextLinkId = 1;
        uint32_t nextGroupId = 1;
        uint32_t nextPortalId = 1;
        // Deserialization may temporarily preserve links authored before image
        // semantics became strict. The editor never enables this for new links.
        bool allowLegacyImageSemanticLinks = false;
        
        // ========================================================================
        // NODE MANAGEMENT
        // ========================================================================
        
        /**
         * @brief Add a node to the graph
         * @tparam T Node type (must derive from NodeBase)
         * @returns Pointer to the created node
         */
        template<typename T, typename... Args>
        T* addNode(Args&&... args) {
            auto node = std::make_shared<T>(std::forward<Args>(args)...);
            return static_cast<T*>(registerNode(std::move(node)));
        }
        
        /**
         * @brief Register an existing node
         */
        NodeBase* registerNode(std::shared_ptr<NodeBase> node) {
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
            
            nodes.push_back(std::move(node));
            return nodes.back().get();
        }
        
        /**
         * @brief Remove a node and all its connections
         */
        void removeNode(uint32_t nodeId) {
            std::unordered_set<uint32_t> removedPinIds;
            if (NodeBase* removedNode = getNode(nodeId)) {
                for (const Pin& pin : removedNode->inputs) removedPinIds.insert(pin.id);
                for (const Pin& pin : removedNode->outputs) removedPinIds.insert(pin.id);
            }

            // Remove all links connected to this node
            links.erase(std::remove_if(links.begin(), links.end(), [this, nodeId](const Link& l) {
                auto* startOwner = getPinOwner(l.startPinId);
                auto* endOwner = getPinOwner(l.endPinId);
                return (startOwner && startOwner->id == nodeId) ||
                       (endOwner && endOwner->id == nodeId);
            }), links.end());
            
            // Remove from groups
            for (auto& group : groups) {
                group.nodeIds.erase(std::remove(group.nodeIds.begin(), group.nodeIds.end(), nodeId),
                                    group.nodeIds.end());
            }

            // Wire junctions are editor proxies for a real output pin.
            portals.erase(std::remove_if(portals.begin(), portals.end(),
                [&removedPinIds](const WirePortal& portal) {
                    return removedPinIds.find(portal.linkedPinId) != removedPinIds.end();
                }), portals.end());
            
            // Remove node
            nodes.erase(std::remove_if(nodes.begin(), nodes.end(), 
                [nodeId](const std::shared_ptr<NodeBase>& n) { return n->id == nodeId; }),
                nodes.end());
        }
        
        /**
         * @brief Get node by ID
         */
        NodeBase* getNode(uint32_t id) const {
            for (auto& n : nodes) {
                if (n->id == id) return n.get();
            }
            return nullptr;
        }
        
        /**
         * @brief Get node by ID with type cast
         */
        template<typename T>
        T* getNodeAs(uint32_t id) const {
            return dynamic_cast<T*>(getNode(id));
        }
        
        // ========================================================================
        // LINK MANAGEMENT
        // ========================================================================
        
        /**
         * @brief Would producer(startPinId) -> consumer(endPinId) close a CYCLE?
         *
         * Nothing else in the system defends against one. Every walker over a graph recurses
         * on its inputs with no visited set — the evaluator, and the material program
         * compiler's subtreeSupported / subtreeSpatial / compileNode / compileMatSlot. A
         * single loop drawn in the editor therefore does not misbehave, it recurses until the
         * stack is gone. And a Windows stack overflow (0xC00000FD) is unrecoverable: the
         * guard page is consumed, so even the top-level catch(...) handler faults as soon as
         * it calls anything deep — the crash surfaces as an access violation inside the crash
         * LOGGER, with the real reason never written down.
         *
         * The link closes a loop exactly when `end`'s node already feeds `start`'s node, so
         * walk UPSTREAM from start and see whether we reach end. Iterative + visited, so it
         * stays safe even if a malformed graph somehow already contains a cycle.
         *
         * The UI uses this too, so a pin that would close a loop is never offered as a snap
         * target in the first place.
         */
        bool wouldCreateCycle(uint32_t startPinId, uint32_t endPinId) {
            NodeBase* src = getPinOwner(startPinId);
            NodeBase* dst = getPinOwner(endPinId);
            if (!src || !dst) return false;
            if (src == dst) return true;   // self-link

            std::vector<NodeBase*> stack{ src };
            std::unordered_set<NodeBase*> seen{ src };
            while (!stack.empty()) {
                NodeBase* n = stack.back();
                stack.pop_back();
                if (n == dst) return true;
                for (const auto& l : links) {
                    if (getPinOwner(l.endPinId) != n) continue;
                    NodeBase* producer = getPinOwner(l.startPinId);
                    if (producer && seen.insert(producer).second) stack.push_back(producer);
                }
            }
            return false;
        }

        /**
         * @brief Create a link between two pins
         * @returns Link ID, or 0 if failed
         */
        virtual uint32_t addLink(uint32_t startPinId, uint32_t endPinId) {
            Pin* start = findPin(startPinId);
            Pin* end = findPin(endPinId);
            
            if (!start || !end) return 0;
            
            // Enforce Output -> Input direction
            if (start->kind == PinKind::Input && end->kind == PinKind::Output) {
                std::swap(startPinId, endPinId);
                std::swap(start, end);
            }
            
            // Validation
            if (start->kind != PinKind::Output || end->kind != PinKind::Input) {
                return 0; // Must be output -> input
            }
            
            if (!start->canConnectTo(*end)) {
                const bool legacyImageLink = allowLegacyImageSemanticLinks &&
                    start->dataType == DataType::Image2D && end->dataType == DataType::Image2D &&
                    (start->imageChannels == 0 || end->imageChannels == 0 ||
                     start->imageChannels == end->imageChannels);
                if (!legacyImageLink) return 0; // Type/semantic/unit incompatible
            }

            if (wouldCreateCycle(startPinId, endPinId)) {
                return 0;
            }

            // Avoid duplicate links for the same pin pair.
            for (const auto& existing : links) {
                if (existing.startPinId == startPinId && existing.endPinId == endPinId) {
                    return existing.id;
                }
            }
            
            // Remove existing link to this input (unless multi-input allowed)
            if (!end->allowMultipleConnections) {
                removeLinkToInput(endPinId);
            } else {
                links.erase(std::remove_if(links.begin(), links.end(),
                    [startPinId, endPinId](const Link& l) {
                        return l.startPinId == startPinId && l.endPinId == endPinId;
                    }),
                    links.end());
            }
            
            // Create link
            Link link;
            link.id = nextLinkId++;
            link.startPinId = startPinId;
            link.endPinId = endPinId;
            links.push_back(link);
            
            // Mark consumer as dirty
            NodeBase* consumer = getPinOwner(endPinId);
            if (consumer) {
                markDirtyDownstream(consumer->id);
            }
            
            return link.id;
        }
        
        /**
         * @brief Remove a link by ID
         */
        void removeLink(uint32_t linkId) {
            auto it = std::find_if(links.begin(), links.end(), 
                [linkId](const Link& l) { return l.id == linkId; });
            
            if (it != links.end()) {
                NodeBase* consumer = getPinOwner(it->endPinId);
                if (consumer) {
                    markDirtyDownstream(consumer->id);
                }
                links.erase(it);
            }
        }
        
        /**
         * @brief Remove all links to an input pin
         */
        void removeLinkToInput(uint32_t inputPinId) {
            links.erase(std::remove_if(links.begin(), links.end(),
                [inputPinId](const Link& l) { return l.endPinId == inputPinId; }),
                links.end());
        }
        
        /**
         * @brief Get link by ID
         */
        Link* getLink(uint32_t id) {
            for (auto& l : links) {
                if (l.id == id) return &l;
            }
            return nullptr;
        }
        
        // ========================================================================
        // PIN QUERIES
        // ========================================================================
        
        /**
         * @brief Find a pin by ID
         */
        Pin* findPin(uint32_t pinId) const {
            for (auto& n : nodes) {
                for (auto& p : n->inputs) if (p.id == pinId) return const_cast<Pin*>(&p);
                for (auto& p : n->outputs) if (p.id == pinId) return const_cast<Pin*>(&p);
            }
            return nullptr;
        }
        
        /**
         * @brief Get the node that owns a pin
         */
        NodeBase* getPinOwner(uint32_t pinId) const {
            for (auto& n : nodes) {
                for (auto& p : n->inputs) if (p.id == pinId) return n.get();
                for (auto& p : n->outputs) if (p.id == pinId) return n.get();
            }
            return nullptr;
        }
        
        /**
         * @brief Get the source output pin connected to an input pin
         */
        Pin* getInputSource(uint32_t inputPinId) const {
            for (auto& l : links) {
                if (l.endPinId == inputPinId) {
                    return findPin(l.startPinId);
                }
            }
            return nullptr;
        }
        
        /**
         * @brief Get the source node connected to an input pin
         */
        NodeBase* getInputSourceNode(uint32_t inputPinId) const {
            Pin* source = getInputSource(inputPinId);
            return source ? getPinOwner(source->id) : nullptr;
        }
        
        // ========================================================================
        // DIRTY PROPAGATION
        // ========================================================================
        
        /**
         * @brief Mark a node and all downstream nodes as dirty
         */
        void markDirtyDownstream(uint32_t nodeId) {
            std::queue<uint32_t> toProcess;
            std::unordered_set<uint32_t> visited;
            
            toProcess.push(nodeId);
            
            while (!toProcess.empty()) {
                uint32_t current = toProcess.front();
                toProcess.pop();
                
                if (visited.count(current)) continue;
                visited.insert(current);
                
                NodeBase* node = getNode(current);
                if (!node) continue;
                
                node->dirty = true;
                
                // Find downstream nodes
                for (auto& outPin : node->outputs) {
                    for (auto& link : links) {
                        if (link.startPinId == outPin.id) {
                            NodeBase* consumer = getPinOwner(link.endPinId);
                            if (consumer && !visited.count(consumer->id)) {
                                toProcess.push(consumer->id);
                            }
                        }
                    }
                }
            }
        }
        
        /**
         * @brief Mark all nodes as dirty
         */
        void markAllDirty() {
            for (auto& n : nodes) {
                n->dirty = true;
            }
        }
        
        // ========================================================================
        // EVALUATION
        // ========================================================================
        
        /**
         * @brief Evaluate the graph by pulling from output nodes
         * 
         * Finds all "terminal" nodes (nodes with no outgoing connections)
         * and requests their output values.
         */
        void evaluate(EvaluationContext& ctx) {
            ctx.startTiming();
            ctx.clearErrors();
            ctx.clearCache();  // Clear cached values for fresh evaluation
            
            // Mark all nodes dirty to force recomputation
            markAllDirty();
            
            // Find terminal nodes (outputs or nodes with unused outputs)
            std::vector<NodeBase*> terminals;
            for (auto& node : nodes) {
                if (isTerminalNode(node.get())) {
                    terminals.push_back(node.get());
                }
            }
            
            // Evaluate each terminal
            float progressStep = terminals.empty() ? 0 : 1.0f / terminals.size();
            float progress = 0;
            
            for (auto* terminal : terminals) {
                if (ctx.isCancelled()) break;
                
                // Request all outputs
                for (size_t i = 0; i < terminal->outputs.size(); i++) {
                    terminal->requestOutput((int)i, ctx);
                }
                
                // Even if no outputs, some nodes are "sinks" (apply to external state)
                if (terminal->outputs.empty()) {
                    terminal->compute(0, ctx);
                }
                
                progress += progressStep;
                ctx.setProgress(progress);
            }
            
            ctx.setProgress(1.0f);
        }

        // ========================================================================
        // ASYNC EVALUATION STATE (generic accessors for the node-editor UI)
        // ========================================================================
        // Default no-ops for graphs that always evaluate synchronously. A graph
        // subtype that supports background evaluation (e.g. TerrainNodeGraphV2)
        // overrides these so NodeEditorUIV2::drawNode() can draw a per-node
        // "currently active" indicator + overall progress without needing to know
        // about that subtype.
        virtual bool isEvaluatingAsync() const { return false; }
        virtual uint32_t currentAsyncNodeId() const { return 0; }
        virtual float asyncEvalProgress() const { return 0.0f; }
        virtual NodeEvaluationState asyncNodeState(uint32_t) const { return NodeEvaluationState::Idle; }

        /**
         * @brief Check if a node is terminal (no downstream connections)
         */
        bool isTerminalNode(NodeBase* node) const {
            for (auto& outPin : node->outputs) {
                for (auto& link : links) {
                    if (link.startPinId == outPin.id) {
                        return false; // Has downstream connection
                    }
                }
            }
            return true;
        }
        
        // ========================================================================
        // GROUP MANAGEMENT
        // ========================================================================
        
        /**
         * @brief Create a new node group
         */
        uint32_t createGroup(const std::string& name, const ImVec2& pos, const ImVec2& size) {
            NodeGroup group;
            group.id = nextGroupId++;
            group.name = name;
            group.position = pos;
            group.size = size;
            groups.push_back(std::move(group));
            return groups.back().id;
        }
        
        /**
         * @brief Add a node to a group
         */
        void addNodeToGroup(uint32_t nodeId, uint32_t groupId) {
            for (auto& group : groups) {
                if (group.id == groupId) {
                    if (std::find(group.nodeIds.begin(), group.nodeIds.end(), nodeId) == group.nodeIds.end()) {
                        group.nodeIds.push_back(nodeId);
                    }
                    // Also update node's groupId
                    if (NodeBase* node = getNode(nodeId)) {
                        node->groupId = groupId;
                    }
                    return;
                }
            }
        }
        
        /**
         * @brief Remove a node from all groups
         */
        void removeNodeFromGroups(uint32_t nodeId) {
            for (auto& group : groups) {
                group.nodeIds.erase(std::remove(group.nodeIds.begin(), group.nodeIds.end(), nodeId),
                                    group.nodeIds.end());
            }
            if (NodeBase* node = getNode(nodeId)) {
                node->groupId = 0;
            }
        }
        
        /**
         * @brief Delete a group (does not delete contained nodes)
         */
        void deleteGroup(uint32_t groupId) {
            // Clear groupId from nodes
            for (auto& node : nodes) {
                if (node->groupId == groupId) {
                    node->groupId = 0;
                }
            }
            
            groups.erase(std::remove_if(groups.begin(), groups.end(),
                [groupId](const NodeGroup& g) { return g.id == groupId; }),
                groups.end());
        }
        
        /**
         * @brief Get group by ID
         */
        NodeGroup* getGroup(uint32_t id) {
            for (auto& g : groups) {
                if (g.id == id) return &g;
            }
            return nullptr;
        }
        
        // ========================================================================
        // UTILITIES
        // ========================================================================
        
        /**
         * @brief Clear all nodes, links, groups
         */
        void clear() {
            nodes.clear();
            links.clear();
            groups.clear();
            portals.clear();
            nextNodeId = 1;
            nextPinId = 1;
            nextLinkId = 1;
            nextGroupId = 1;
            nextPortalId = 1;
        }
        
        /**
         * @brief Get count of nodes
         */
        size_t nodeCount() const { return nodes.size(); }
        
        /**
         * @brief Get count of links
         */
        size_t linkCount() const { return links.size(); }
    };

    // ============================================================================
    // NODE INPUT VALUE IMPLEMENTATION
    // ============================================================================
    
    // Implement the getInputValue method that was declared in Node.h
    inline PinValue NodeBase::getInputValue(int inputIndex, EvaluationContext& ctx) {
        if (inputIndex < 0 || inputIndex >= (int)inputs.size()) {
            return PinValue{}; // Invalid index
        }
        
        Pin& inputPin = inputs[inputIndex];
        GraphBase* graph = ctx.getGraph();
        
        if (!graph) {
            ctx.addError(id, "No graph in evaluation context");
            return inputPin.defaultValue;
        }
        
        // Find source connection
        Pin* sourcePin = graph->getInputSource(inputPin.id);
        if (!sourcePin) {
            // No connection - use default or return empty
            if (inputPin.hasDefaultValue()) {
                return inputPin.defaultValue;
            }
            if (!inputPin.optional) {
                ctx.addError(id, "Required input '" + inputPin.name + "' not connected");
            }
            return PinValue{};
        }
        
        // Get the source node
        NodeBase* sourceNode = graph->getPinOwner(sourcePin->id);
        if (!sourceNode) {
            ctx.addError(id, "Source node not found for input '" + inputPin.name + "'");
            return PinValue{};
        }
        
        // Find output index
        int outputIndex = -1;
        for (size_t i = 0; i < sourceNode->outputs.size(); i++) {
            if (sourceNode->outputs[i].id == sourcePin->id) {
                outputIndex = (int)i;
                break;
            }
        }
        
        if (outputIndex < 0) {
            ctx.addError(id, "Output pin not found on source node");
            return PinValue{};
        }
        
        // Request value from source
        return sourceNode->requestOutput(outputIndex, ctx);
    }

} // namespace NodeSystem

