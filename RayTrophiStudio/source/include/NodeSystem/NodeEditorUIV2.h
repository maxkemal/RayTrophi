/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          NodeEditorUIV2.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once

/**
 * @file NodeEditorUIV2.h
 * @brief Modern node editor UI for the new node system
 * 
 * Features:
 * - Color-coded pins by DataType
 * - Shape-based pin rendering (circle, square, diamond)
 * - NodeGroup (Frame) rendering
 * - Minimap support
 * - Improved visual design inspired by Gaea
 */

#include "NodeCore.h"
#include "EvaluationContext.h"
#include "Node.h"
#include "NodeEditorChrome.h"
#include "Graph.h"
#include "imgui.h"
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <functional>
#include <utility>

namespace NodeSystem {

    /**
     * @brief Modern node editor rendering class
     */
    class NodeEditorUIV2 {
    public:
        // ========================================================================
        // CONFIGURATION
        // ========================================================================
        
        struct Config {
            // Grid
            ImU32 gridColorMajor = IM_COL32(58, 64, 78, 180);
            ImU32 gridColorMinor = IM_COL32(42, 46, 58, 120);
            float gridSizeMajor = 128.0f;
            float gridSizeMinor = 32.0f;
            ImU32 bgColor = IM_COL32(22, 24, 30, 255);
            
            // Node
            ImU32 nodeBodyColor = IM_COL32(34, 37, 45, 245);
            ImU32 nodeBorderColor = IM_COL32(66, 74, 92, 210);
            ImU32 nodeSelectedColor = IM_COL32(109, 180, 255, 255);
            float nodeRounding = 4.0f;
            float nodeBorderWidth = 1.5f;
            
            // Link
            float linkThickness = 1.5f;
            float linkSelectedThickness = 2.5f;
            
            // Group
            ImU32 groupBorderColor = IM_COL32(100, 100, 120, 180);
            ImU32 groupTitleColor = IM_COL32(255, 255, 255, 200);
            
            // Minimap
            bool showMinimap = false;
            float minimapSize = 150.0f;
            ImU32 minimapBgColor = IM_COL32(20, 20, 25, 200);

            // Inline node-body content (NodeBase::wantsInlineContent).
            // Widgets don't zoom-scale, so they only render close to 1:1.
            bool inlineNodeContent = true;
            float inlineContentMinZoom = 0.8f;
        } config;

        // ========================================================================
        // STATE
        // ========================================================================
        
        float zoom = 1.0f;
        float scrollX = 0.0f;
        float scrollY = 0.0f;
        
        uint32_t selectedNodeId = 0;
        std::vector<uint32_t> selectedNodeIds;
        uint32_t selectedLinkId = 0;
        uint32_t selectedPortalId = 0;
        uint32_t selectedGroupId = 0;
        uint32_t draggingNodeId = 0;
        uint32_t draggingGroupId = 0;
        uint32_t resizingNodeId = 0;
        
        ImVec2 mousePosOnRightClick = ImVec2(0.0f, 0.0f);
        uint32_t releasedLinkPinId = 0;

        // Right-click context menu resolution: the background canvas's own hover+click check
        // (handleInput(), runs FIRST in draw()) can't yet know whether a node/group drawn LATER
        // in the same frame will also claim that same click, since node/group hit-testing hasn't
        // happened yet at that point. Opening "LocalGraphContextPopup" immediately there meant
        // right-clicking a node or group always opened the background/group-tools popup instead
        // of the node's or group's own popup (background's OpenPopup call, being first in the
        // frame, wins ImGui's same-frame-multiple-OpenPopup resolution). Fixed by deferring:
        // background only marks a request here; node/group loops (drawn after) get first refusal
        // and flip contextMenuClaimedByNodeOrGroup_ when THEY open their own popup; the actual
        // OpenPopup("LocalGraphContextPopup") call happens once, right before drawPopups(), only
        // if nothing claimed the click in between.
        bool backgroundContextMenuRequested_ = false;
        bool contextMenuClaimedByNodeOrGroup_ = false;
        bool linkCreateMenuRequested_ = false;
        bool linkCreatePopupWasOpen_ = false;
        // Node/group right-click detection happens INSIDE a local ImGui::PushID(node.id/group.id
        // + offset) scope (needed so multiple nodes'/groups' identically-named InvisibleButtons
        // don't collide). ImGui::OpenPopup()/BeginPopup() resolve their id from the CURRENT id
        // stack at the time each is called — calling OpenPopup while still inside that per-node
        // PushID scope registers it under a DIFFERENT id than drawPopups()'s BeginPopup (called
        // at the un-pushed root scope), so it could never actually be found/shown. These flags
        // let the click be detected inside the PushID scope but the actual OpenPopup call happen
        // right after the matching PopID(), at the same scope BeginPopup uses.
        bool nodeContextMenuRequested_ = false;
        bool groupContextMenuRequested_ = false;
        uint32_t resizingGroupId = 0;
        uint32_t draggingPortalId = 0;
        
        bool isCreatingLink = false;
        uint32_t linkStartPinId = 0;
        DataType linkStartType = DataType::None;
        
        // Box selection state
        bool isBoxSelecting = false;
        ImVec2 boxSelectStartPos;
        
        // Callbacks
        std::function<void(uint32_t nodeId)> onNodeSelected;
        std::function<void(uint32_t linkId)> onLinkSelected;
        std::function<void()> onBackgroundContextMenu;
        std::function<void(uint32_t nodeId)> onNodeContextMenu;
        std::function<void()> onDrawBackgroundMenu;
        // Domain-specific searchable node picker shown when a link is released
        // over empty canvas. The domain owns node creation; onNodeAdded() then
        // performs the type-compatible auto connection.
        std::function<void()> onDrawLinkCreateMenu;
        std::function<void()> onGraphModified;

        // 0 = complete graph. A non-zero group id turns the editor into a layer
        // workspace: only that group's nodes are shown and cross-layer links are
        // represented by explicit input/output boundary ports.
        uint32_t focusedGroupId = 0;

        /**
         * @brief Reset editor state (zoom, scroll, selection)
         */
        void reset() {
            zoom = 1.0f;
            scrollX = 0.0f;
            scrollY = 0.0f;
            selectedNodeId = 0;
            selectedNodeIds.clear();
            selectedLinkId = 0;
            selectedPortalId = 0;
            selectedGroupId = 0;
            focusedGroupId = 0;
            draggingNodeId = 0;
            draggingGroupId = 0;
            resizingNodeId = 0;
            resizingGroupId = 0;
            draggingPortalId = 0;
            isCreatingLink = false;
            linkStartPinId = 0;
            isBoxSelecting = false;
            releasedLinkPinId = 0;
            linkCreateMenuRequested_ = false;
            linkCreatePopupWasOpen_ = false;
            mousePosOnRightClick = ImVec2(0.0f, 0.0f);
        }

    private:
        struct LinkScreenGeometry {
            ImVec2 p1, cp1, cp2, p2;
        };
        std::unordered_map<uint32_t, ImVec2> pinPositions_;
        std::unordered_map<uint32_t, LinkScreenGeometry> linkScreenGeometry_;
        std::unordered_map<uint32_t, ImVec2> portalSourcePositions_;

    public:
        ImVec2 canvasPos_;
        ImVec2 canvasSize_;

        std::string fitTextToWidth(const std::string& text, float maxWidth) const {
            if (text.empty() || maxWidth <= 8.0f) return {};
            if (ImGui::CalcTextSize(text.c_str()).x <= maxWidth) return text;

            static const char* kEllipsis = "...";
            std::string result = text;
            while (!result.empty()) {
                result.pop_back();
                const std::string candidate = result + kEllipsis;
                if (ImGui::CalcTextSize(candidate.c_str()).x <= maxWidth) {
                    return candidate;
                }
            }
            return {};
        }
        
    public:
        NodeEditorUIV2() = default;

        // ========================================================================
        // MAIN DRAW
        // ========================================================================
        
        void draw(GraphBase& graph, const ImVec2& size = ImVec2(0, 0)) {
            ImGui::BeginChild("NodeEditorV2Canvas", size, true, 
                ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoMove);
            
            ImDrawList* dl = ImGui::GetWindowDrawList();
            canvasPos_ = ImGui::GetCursorScreenPos();
            canvasSize_ = ImGui::GetContentRegionAvail();

            // Keep custom graph primitives inside the child canvas. Oversized frames used
            // to bleed into adjacent panels and read as translucent "ghost" geometry.
            dl->PushClipRect(canvasPos_,
                ImVec2(canvasPos_.x + canvasSize_.x, canvasPos_.y + canvasSize_.y), true);
            
            // Background
            dl->AddRectFilled(canvasPos_, 
                ImVec2(canvasPos_.x + canvasSize_.x, canvasPos_.y + canvasSize_.y), 
                config.bgColor);
            
            // Grid
            drawGrid(dl);

            // Keep the serialized layer interface contract synchronized with
            // cross-layer topology before any hit testing or rendering.
            synchronizeLayerInterfaces(graph);
            
            // Input handling
            handleInput(graph);
            
            // Reset pin cache
            pinPositions_.clear();
            linkScreenGeometry_.clear();
            portalSourcePositions_.clear();
            
            // Draw groups (behind nodes)
            for (auto& group : graph.groups) {
                if (focusedGroupId != 0) continue;
                drawGroup(dl, group, graph);
            }
            
            // Draw nodes
            for (auto& node : graph.nodes) {
                if (!isNodeVisible(graph, *node)) continue;
                drawNode(dl, *node, graph);
            }

            if (focusedGroupId == 0) prepareWirePortals(graph);
            
            // Draw internal links. In layer view, crossing links terminate at explicit
            // canvas-edge interface ports instead of disappearing or spanning other layers.
            std::vector<Link*> incomingBoundaryLinks;
            std::vector<Link*> outgoingBoundaryLinks;
            for (auto& link : graph.links) {
                Pin* startPin = graph.findPin(link.startPinId);
                Pin* endPin = graph.findPin(link.endPinId);
                NodeBase* startNode = startPin ? graph.getNode(startPin->nodeId) : nullptr;
                NodeBase* endNode = endPin ? graph.getNode(endPin->nodeId) : nullptr;
                const bool startVisible = startNode && isNodeVisible(graph, *startNode);
                const bool endVisible = endNode && isNodeVisible(graph, *endNode);
                if (startVisible && endVisible) {
                    drawLink(dl, link, graph);
                } else if (focusedGroupId != 0 && startVisible != endVisible) {
                    (endVisible ? incomingBoundaryLinks : outgoingBoundaryLinks).push_back(&link);
                } else if (focusedGroupId == 0) {
                    // A collapsed group is a visual proxy for its hidden nodes. Its
                    // interface pins were registered by drawGroup(), so crossing
                    // links can keep their real graph pin ids while terminating on
                    // the compact group tile. Links wholly inside one collapsed
                    // group remain hidden.
                    const NodeGroup* startGroup = startNode ? graph.getGroup(startNode->groupId) : nullptr;
                    const NodeGroup* endGroup = endNode ? graph.getGroup(endNode->groupId) : nullptr;
                    const bool sameCollapsedGroup = startGroup && endGroup &&
                        startGroup->id == endGroup->id && startGroup->collapsed;
                    if (!sameCollapsedGroup &&
                        pinPositions_.find(link.startPinId) != pinPositions_.end() &&
                        pinPositions_.find(link.endPinId) != pinPositions_.end()) {
                        drawLink(dl, link, graph);
                    }
                }
            }

            auto drawBoundarySet = [&](std::vector<Link*>& boundaryLinks, bool incoming) {
                std::sort(boundaryLinks.begin(), boundaryLinks.end(), [this, &graph, incoming](const Link* a, const Link* b) {
                    const LayerInterfacePort* aPort = findLayerInterfacePort(graph, *a, incoming);
                    const LayerInterfacePort* bPort = findLayerInterfacePort(graph, *b, incoming);
                    const uint32_t aKey = aPort ? aPort->id : (incoming ? a->startPinId : a->endPinId);
                    const uint32_t bKey = bPort ? bPort->id : (incoming ? b->startPinId : b->endPinId);
                    if (aKey != bKey) return aKey < bKey;
                    return a->id < b->id;
                });
                uint32_t previousPortKey = 0;
                int row = -1;
                for (Link* link : boundaryLinks) {
                    LayerInterfacePort* port = findLayerInterfacePort(graph, *link, incoming);
                    const uint32_t portKey = port ? port->id : (incoming ? link->startPinId : link->endPinId);
                    const bool firstForPort = portKey != previousPortKey;
                    if (firstForPort) {
                        ++row;
                        previousPortKey = portKey;
                    }
                    drawLayerBoundaryLink(dl, *link, graph, incoming, row, firstForPort, port);
                }
                return row + 1;
            };
            int incomingRows = drawBoundarySet(incomingBoundaryLinks, true);
            int outgoingRows = drawBoundarySet(outgoingBoundaryLinks, false);
            if (NodeGroup* focused = graph.getGroup(focusedGroupId)) {
                for (const LayerInterfacePort& port : focused->interfacePorts) {
                    if (port.connected) continue;
                    const bool incoming = port.direction == LayerPortDirection::Input;
                    drawDisconnectedLayerPort(dl, graph, port, incoming,
                                              incoming ? incomingRows++ : outgoingRows++);
                }
            }

            if (focusedGroupId == 0) drawWirePortals(dl, graph);
            
            // Draw creating link
            if (isCreatingLink && linkStartPinId != 0) {
                drawCreatingLink(dl);
            }
            
            // Minimap
            if (config.showMinimap) {
                drawMinimap(dl, graph);
            }

            dl->PopClipRect();
            
            // Resolve the deferred background context menu request now that node/group
            // hit-testing (which can claim the same click) has already run this frame.
            if (backgroundContextMenuRequested_ && !contextMenuClaimedByNodeOrGroup_) {
                ImGui::OpenPopup("LocalGraphContextPopup");
            }
            backgroundContextMenuRequested_ = false;
            contextMenuClaimedByNodeOrGroup_ = false;

            if (linkCreateMenuRequested_) {
                ImGui::OpenPopup("LocalLinkCreatePopup");
                linkCreatePopupWasOpen_ = true;
            }
            linkCreateMenuRequested_ = false;

            // Context menu popups for group rename/color/delete
            drawPopups(graph);

            ImGui::EndChild();
        }
        void onNodeAdded(GraphBase& graph, NodeBase* newNode) {
            if (releasedLinkPinId != 0 && newNode) {
                Pin* dragPin = graph.findPin(releasedLinkPinId);
                if (dragPin) {
                    if (dragPin->kind == PinKind::Output) {
                        for (auto& inPin : newNode->inputs) {
                            if (dragPin->canConnectTo(inPin)) {
                                graph.addLink(dragPin->id, inPin.id);
                                if (onGraphModified) onGraphModified();
                                break;
                            }
                        }
                    }
                    else if (dragPin->kind == PinKind::Input) {
                        for (auto& outPin : newNode->outputs) {
                            if (outPin.canConnectTo(*dragPin)) {
                                graph.addLink(outPin.id, dragPin->id);
                                if (onGraphModified) onGraphModified();
                                break;
                            }
                        }
                    }
                }
                releasedLinkPinId = 0; // Reset
            }
        }

        void autoArrangeGroup(GraphBase& graph, uint32_t groupId) {
            if (NodeGroup* group = graph.getGroup(groupId)) {
                autoArrangeGroupContents(graph, *group);
                if (onGraphModified) onGraphModified();
            }
        }

        Link* findVisibleLinkAt(GraphBase& graph, const ImVec2& mouse, float threshold = 15.0f) {
            for (auto& link : graph.links) {
                const auto geometry = linkScreenGeometry_.find(link.id);
                if (geometry == linkScreenGeometry_.end()) continue;
                ImVec2 previous = geometry->second.p1;
                for (int step = 1; step <= 24; ++step) {
                    const float t = static_cast<float>(step) / 24.0f;
                    const ImVec2 point = getBezierPoint(
                        geometry->second.p1, geometry->second.cp1,
                        geometry->second.cp2, geometry->second.p2, t);
                    const ImVec2 v(point.x - previous.x, point.y - previous.y);
                    const ImVec2 w(mouse.x - previous.x, mouse.y - previous.y);
                    const float len2 = v.x * v.x + v.y * v.y;
                    const float projection = len2 > 0.0001f
                        ? std::clamp((w.x * v.x + w.y * v.y) / len2, 0.0f, 1.0f) : 0.0f;
                    const ImVec2 closest(previous.x + v.x * projection, previous.y + v.y * projection);
                    if (std::hypot(mouse.x - closest.x, mouse.y - closest.y) <= threshold) {
                        return &link;
                    }
                    previous = point;
                }
            }
            return nullptr;
        }

        void synchronizeLayerInterfaces(GraphBase& graph) {
            if (syncLayerInterfaces(graph) && onGraphModified) onGraphModified();
        }


    private:
        LayerInterfacePort* findLayerInterfacePort(GraphBase& graph, const Link& link, bool incoming) {
            NodeGroup* group = graph.getGroup(focusedGroupId);
            if (!group) return nullptr;
            const LayerPortDirection direction = incoming ? LayerPortDirection::Input : LayerPortDirection::Output;
            const uint32_t internalPinId = incoming ? link.endPinId : link.startPinId;
            const uint32_t externalPinId = incoming ? link.startPinId : link.endPinId;
            for (auto& port : group->interfacePorts) {
                if (port.direction == direction && port.internalPinId == internalPinId &&
                    port.externalPinId == externalPinId) return &port;
            }
            for (auto& port : group->interfacePorts) {
                if (port.direction != direction) continue;
                if ((incoming && port.externalPinId == externalPinId) ||
                    (!incoming && port.internalPinId == internalPinId)) return &port;
            }
            return nullptr;
        }

        bool syncLayerInterfaces(GraphBase& graph) {
            bool changed = false;
            for (NodeGroup& group : graph.groups) {
                std::vector<bool> wasConnected;
                wasConnected.reserve(group.interfacePorts.size());
                for (auto& port : group.interfacePorts) {
                    wasConnected.push_back(port.connected);
                    port.connected = false;
                }

                for (const Link& link : graph.links) {
                    NodeBase* startNode = graph.getPinOwner(link.startPinId);
                    NodeBase* endNode = graph.getPinOwner(link.endPinId);
                    if (!startNode || !endNode) continue;
                    const bool startInside = startNode->groupId == group.id;
                    const bool endInside = endNode->groupId == group.id;
                    if (startInside == endInside) continue;

                    const LayerPortDirection direction = endInside
                        ? LayerPortDirection::Input : LayerPortDirection::Output;
                    const uint32_t internalPinId = endInside ? link.endPinId : link.startPinId;
                    const uint32_t externalPinId = endInside ? link.startPinId : link.endPinId;
                    Pin* internalPin = graph.findPin(internalPinId);
                    if (!internalPin) continue;

                    LayerInterfacePort* matched = nullptr;
                    enum class MatchKind { None, Exact, Primary, Secondary } matchKind = MatchKind::None;
                    for (auto& port : group.interfacePorts) {
                        if (port.direction != direction) continue;
                        if (port.internalPinId == internalPinId && port.externalPinId == externalPinId) {
                            matched = &port; matchKind = MatchKind::Exact; break;
                        }
                    }
                    if (!matched) {
                        for (auto& port : group.interfacePorts) {
                            if (port.direction != direction) continue;
                            const bool primaryMatch = direction == LayerPortDirection::Input
                                ? port.externalPinId == externalPinId
                                : port.internalPinId == internalPinId;
                            if (primaryMatch) { matched = &port; matchKind = MatchKind::Primary; break; }
                        }
                    }
                    if (!matched) {
                        for (auto& port : group.interfacePorts) {
                            if (port.direction != direction) continue;
                            const bool secondaryMatch = direction == LayerPortDirection::Input
                                ? port.internalPinId == internalPinId
                                : port.externalPinId == externalPinId;
                            if (secondaryMatch) { matched = &port; matchKind = MatchKind::Secondary; break; }
                        }
                    }

                    if (!matched) {
                        LayerInterfacePort port;
                        port.id = group.nextInterfacePortId++;
                        port.direction = direction;
                        port.name = internalPin->name;
                        port.internalPinId = internalPinId;
                        port.externalPinId = externalPinId;
                        port.dataType = internalPin->dataType;
                        port.imageSemantic = internalPin->imageSemantic;
                        port.imageChannels = internalPin->imageChannels;
                        port.imageUnit = internalPin->imageUnit;
                        group.interfacePorts.push_back(std::move(port));
                        matched = &group.interfacePorts.back();
                        matchKind = MatchKind::Exact;
                        changed = true;
                    }

                    // Splicing inside a layer changes the internal endpoint;
                    // splicing outside changes the external endpoint. Preserve
                    // the port id/name and refresh only the moved side. For
                    // fan-out links the first crossing remains the representative.
                    if (matchKind == MatchKind::Secondary ||
                        (matchKind == MatchKind::Primary && !matched->connected)) {
                        if (matched->internalPinId != internalPinId || matched->externalPinId != externalPinId) {
                            matched->internalPinId = internalPinId;
                            matched->externalPinId = externalPinId;
                            changed = true;
                        }
                    }
                    matched->dataType = internalPin->dataType;
                    matched->imageSemantic = internalPin->imageSemantic;
                    matched->imageChannels = internalPin->imageChannels;
                    matched->imageUnit = internalPin->imageUnit;
                    matched->connected = true;
                }

                for (size_t i = 0; i < group.interfacePorts.size(); ++i) {
                    const bool oldConnected = i < wasConnected.size() ? wasConnected[i] : false;
                    if (oldConnected != group.interfacePorts[i].connected) changed = true;
                }
            }
            return changed;
        }

        void drawPopups(GraphBase& graph) {
            // Searchable, domain-provided node picker opened by dropping a link
            // on empty canvas. It deliberately has its own popup so the normal
            // right-click graph menu remains unchanged.
            if (ImGui::BeginPopup("LocalLinkCreatePopup")) {
                if (onDrawLinkCreateMenu) onDrawLinkCreateMenu();
                ImGui::EndPopup();
            } else if (linkCreatePopupWasOpen_) {
                // Escape/click-away cancels the pending auto connection.
                releasedLinkPinId = 0;
                linkCreatePopupWasOpen_ = false;
            }

            // Background Context Menu
            if (ImGui::BeginPopup("LocalGraphContextPopup")) {
                if (ImGui::MenuItem("Create Group Frame", "Ctrl+G", nullptr, !selectedNodeIds.empty())) {
                    // Create group wrapping selection
                    float minX = FLT_MAX, minY = FLT_MAX, maxX = -FLT_MAX, maxY = -FLT_MAX;
                    for (uint32_t nid : selectedNodeIds) {
                        NodeBase* n = graph.getNode(nid);
                        if (n) {
                            minX = std::min(minX, n->x);
                            minY = std::min(minY, n->y);
                            maxX = std::max(maxX, n->x + 160.0f);
                            maxY = std::max(maxY, n->y + 100.0f);
                        }
                    }
                    if (minX != FLT_MAX) {
                        uint32_t gid = graph.createGroup("New Group", 
                            ImVec2(minX - 16.0f, minY - 32.0f), 
                            ImVec2(maxX - minX + 32.0f, maxY - minY + 48.0f));
                        for (uint32_t nid : selectedNodeIds) {
                            graph.addNodeToGroup(nid, gid);
                        }
                        if (onGraphModified) onGraphModified();
                    }
                }
                
                if (ImGui::MenuItem("Create Empty Group Frame")) {
                    graph.createGroup("New Group", mousePosOnRightClick, ImVec2(240, 160));
                    if (onGraphModified) onGraphModified();
                }

                // Domain-specific extra items (e.g. "Add Base Mesh" / "Add Subdivide" for the
                // Geo-DAG) appended into this SAME background popup instead of opening a second,
                // competing one — was declared but never invoked before this.
                if (onDrawBackgroundMenu) {
                    ImGui::Separator();
                    onDrawBackgroundMenu();
                }

                ImGui::EndPopup();
            }
            
            // Node Context Menu
            if (ImGui::BeginPopup("LocalNodeContextPopup")) {
                NodeBase* activeNode = graph.getNode(selectedNodeId);
                if (activeNode) {
                    ImGui::Text("Node: %s", activeNode->metadata.displayName.c_str());
                    ImGui::Separator();
                    
                    if (activeNode->groupId != 0) {
                        if (ImGui::BeginMenu("Move to Layer")) {
                            for (auto& g : graph.groups) {
                                if (g.id == activeNode->groupId) continue;
                                if (ImGui::MenuItem(g.name.c_str())) {
                                    const uint32_t oldGroupId = activeNode->groupId;
                                    graph.removeNodeFromGroups(activeNode->id);
                                    graph.addNodeToGroup(activeNode->id, g.id);
                                    if (NodeGroup* oldGroup = graph.getGroup(oldGroupId)) {
                                        fitGroupToContents(graph, *oldGroup);
                                    }
                                    fitGroupToContents(graph, g);
                                    if (onGraphModified) onGraphModified();
                                }
                            }
                            ImGui::EndMenu();
                        }
                        if (ImGui::MenuItem("Remove from Group")) {
                            graph.removeNodeFromGroups(activeNode->id);
                            if (onGraphModified) onGraphModified();
                        }
                    } else {
                        if (ImGui::BeginMenu("Add to Group")) {
                            for (auto& g : graph.groups) {
                                if (ImGui::MenuItem(g.name.c_str())) {
                                    graph.addNodeToGroup(activeNode->id, g.id);
                                    if (onGraphModified) onGraphModified();
                                }
                            }
                            ImGui::EndMenu();
                        }
                    }
                    
                    if (ImGui::MenuItem("Delete Node")) {
                        const uint32_t deletedId = activeNode->id;
                        graph.removeNode(deletedId);
                        selectedNodeId = 0;
                        selectedNodeIds.erase(std::remove(selectedNodeIds.begin(), selectedNodeIds.end(), deletedId), selectedNodeIds.end());
                        if (onGraphModified) onGraphModified();
                    }
                }
                ImGui::EndPopup();
            }
            
            // Group Context Menu
            if (ImGui::BeginPopup("LocalGroupContextPopup")) {
                NodeGroup* group = graph.getGroup(selectedGroupId);
                if (group) {
                    ImGui::Text("Group: %s", group->name.c_str());
                    ImGui::Separator();
                    
                    static char renameBuf[128] = "";
                    if (ImGui::IsWindowAppearing()) {
                        strncpy_s(renameBuf, group->name.c_str(), sizeof(renameBuf) - 1);
                    }
                    if (ImGui::InputText("Rename", renameBuf, sizeof(renameBuf))) {
                        group->name = renameBuf;
                        if (onGraphModified) onGraphModified();
                    }
                    
                    static ImVec4 colorPicker;
                    if (ImGui::IsWindowAppearing()) {
                        colorPicker = ImGui::ColorConvertU32ToFloat4(group->color);
                    }
                    if (ImGui::ColorEdit4("Color", &colorPicker.x, ImGuiColorEditFlags_NoInputs)) {
                        group->color = ImGui::ColorConvertFloat4ToU32(colorPicker);
                        if (onGraphModified) onGraphModified();
                    }
                    
                    if (ImGui::MenuItem("Select All Nodes in Group")) {
                        selectedNodeIds = group->nodeIds;
                        if (!selectedNodeIds.empty()) {
                            selectedNodeId = selectedNodeIds.back();
                        }
                    }

                    if (ImGui::MenuItem(group->collapsed ? "Expand Layer" : "Collapse Layer")) {
                        group->collapsed = !group->collapsed;
                        if (onGraphModified) onGraphModified();
                    }

                    if (ImGui::MenuItem("Fit Frame to Nodes", nullptr, false, !group->nodeIds.empty())) {
                        fitGroupToContents(graph, *group);
                        if (onGraphModified) onGraphModified();
                    }

                    if (ImGui::MenuItem("Auto Arrange Layer", nullptr, false, !group->nodeIds.empty())) {
                        autoArrangeGroupContents(graph, *group);
                        if (onGraphModified) onGraphModified();
                    }
                    
                    if (ImGui::MenuItem("Remove All Nodes")) {
                        for (uint32_t nid : group->nodeIds) {
                            NodeBase* n = graph.getNode(nid);
                            if (n) n->groupId = 0;
                        }
                        group->nodeIds.clear();
                        if (onGraphModified) onGraphModified();
                    }
                    
                    if (ImGui::MenuItem("Delete Group Frame (Keep Nodes)")) {
                        graph.deleteGroup(group->id);
                        selectedGroupId = 0;
                        if (onGraphModified) onGraphModified();
                    }
                }
                ImGui::EndPopup();
            }
        }
        // ========================================================================
        // GRID
        // ========================================================================

        bool isNodeVisible(GraphBase& graph, const NodeBase& node) const {
            if (focusedGroupId != 0) return node.groupId == focusedGroupId;
            if (node.groupId == 0) return true;
            const NodeGroup* group = graph.getGroup(node.groupId);
            return !(group && group->collapsed);
        }

        static void fitGroupToContents(GraphBase& graph, NodeGroup& group) {
            float minX = FLT_MAX, minY = FLT_MAX, maxX = -FLT_MAX, maxY = -FLT_MAX;
            for (uint32_t nodeId : group.nodeIds) {
                NodeBase* node = graph.getNode(nodeId);
                if (!node) continue;
                const float customWidth = node->getCustomWidth();
                const float width = node->uiWidth > 0.0f ? node->uiWidth
                    : (customWidth > 0.0f ? customWidth : 180.0f);
                int visibleInputs = 0, visibleOutputs = 0;
                for (const Pin& pin : node->inputs) if (!pin.hidden) ++visibleInputs;
                for (const Pin& pin : node->outputs) if (!pin.hidden) ++visibleOutputs;
                const float height = node->collapsed ? 34.0f
                    : 54.0f + 22.0f * static_cast<float>(std::max(visibleInputs, visibleOutputs));
                minX = std::min(minX, node->x);
                minY = std::min(minY, node->y);
                maxX = std::max(maxX, node->x + width);
                maxY = std::max(maxY, node->y + height);
            }
            if (minX == FLT_MAX) return;
            constexpr float sidePad = 24.0f;
            constexpr float topPad = 44.0f;
            constexpr float bottomPad = 24.0f;
            group.position = ImVec2(minX - sidePad, minY - topPad);
            group.size = ImVec2((maxX - minX) + sidePad * 2.0f,
                                (maxY - minY) + topPad + bottomPad);
        }

        static void autoArrangeGroupContents(GraphBase& graph, NodeGroup& group) {
            if (group.nodeIds.empty()) return;
            constexpr float topPad = 46.0f;
            constexpr float sidePad = 24.0f;
            constexpr float columnGap = 34.0f;
            constexpr float rowGap = 26.0f;
            const int columns = group.nodeIds.size() > 2 ? 2 : 1;
            float cellWidth = 180.0f;
            auto nodeSize = [](const NodeBase& node) {
                int inCount = 0, outCount = 0;
                for (const Pin& pin : node.inputs) if (!pin.hidden) ++inCount;
                for (const Pin& pin : node.outputs) if (!pin.hidden) ++outCount;
                const float custom = node.getCustomWidth();
                const float width = node.uiWidth > 0.0f ? node.uiWidth
                    : (custom > 0.0f ? custom : 180.0f);
                const float height = node.collapsed ? 34.0f
                    : 54.0f + 22.0f * static_cast<float>(std::max(inCount, outCount));
                return ImVec2(width, height);
            };
            for (uint32_t nodeId : group.nodeIds) {
                if (NodeBase* node = graph.getNode(nodeId)) {
                    cellWidth = std::max(cellWidth, nodeSize(*node).x);
                }
            }
            float y[2] = {group.position.y + topPad, group.position.y + topPad};
            for (uint32_t nodeId : group.nodeIds) {
                NodeBase* node = graph.getNode(nodeId);
                if (!node) continue;
                const ImVec2 size = nodeSize(*node);
                const int column = columns == 1 ? 0 : (y[0] <= y[1] ? 0 : 1);
                node->x = group.position.x + sidePad + column * (cellWidth + columnGap);
                node->y = y[column];
                y[column] += size.y + rowGap;
            }
            fitGroupToContents(graph, group);
        }
        
        void drawGrid(ImDrawList* dl) {
            float minorStep = config.gridSizeMinor * zoom;
            float majorStep = config.gridSizeMajor * zoom;
            
            // Minor grid (drawn as dots for a premium modern aesthetic, saving draw calls + look & feel)
            if (minorStep > 12.0f) {
                ImU32 dotColor = config.gridColorMinor;
                float startX = fmodf(scrollX, minorStep);
                float startY = fmodf(scrollY, minorStep);
                
                for (float x = startX; x < canvasSize_.x; x += minorStep) {
                    for (float y = startY; y < canvasSize_.y; y += minorStep) {
                        // Skip if it aligns closely with the major grid to keep it clean
                        if (fmodf(x - scrollX + 1.0f, majorStep) < minorStep * 0.5f ||
                            fmodf(y - scrollY + 1.0f, majorStep) < minorStep * 0.5f) {
                            continue;
                        }
                        dl->AddCircleFilled(ImVec2(canvasPos_.x + x, canvasPos_.y + y), 1.0f * std::clamp(zoom, 0.5f, 1.2f), dotColor);
                    }
                }
            }
            
            // Major grid (drawn as subtle lines)
            for (float x = fmodf(scrollX, majorStep); x < canvasSize_.x; x += majorStep) {
                dl->AddLine(
                    ImVec2(canvasPos_.x + x, canvasPos_.y),
                    ImVec2(canvasPos_.x + x, canvasPos_.y + canvasSize_.y),
                    config.gridColorMajor,
                    1.0f
                );
            }
            for (float y = fmodf(scrollY, majorStep); y < canvasSize_.y; y += majorStep) {
                dl->AddLine(
                    ImVec2(canvasPos_.x, canvasPos_.y + y),
                    ImVec2(canvasPos_.x + canvasSize_.x, canvasPos_.y + y),
                    config.gridColorMajor,
                    1.0f
                );
            }
        }

        // ========================================================================
        // INPUT
        // ========================================================================
        
        void handleInput(GraphBase& graph) {
            ImGui::SetCursorScreenPos(canvasPos_);
            ImGui::SetNextItemAllowOverlap(); // must precede the item in ImGui 1.92+
            ImGui::InvisibleButton("##CanvasInput", canvasSize_,
                ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonMiddle | ImGuiButtonFlags_MouseButtonRight);

            // Popups and text search own navigation while open. In particular,
            // a trackball/middle-button gesture or wheel used over the searchable
            // link-create menu must never leak through to the graph canvas.
            const bool canvasGestureBlocked =
                ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId) || ImGui::GetIO().WantTextInput;
            
            // Pan (middle mouse) & Box Selection start
            if (!canvasGestureBlocked && ImGui::IsItemActive()) {
                if (ImGui::IsMouseDragging(ImGuiMouseButton_Middle)) {
                    scrollX += ImGui::GetIO().MouseDelta.x;
                    scrollY += ImGui::GetIO().MouseDelta.y;
                } else if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                    // Clicking on background clears selection unless Shift is held
                    if (!ImGui::GetIO().KeyShift) {
                        selectedNodeIds.clear();
                        selectedNodeId = 0;
                    }
                    isBoxSelecting = true;
                    boxSelectStartPos = ImGui::GetMousePos();
                }
            }
            
            // Box selection processing
            if (isBoxSelecting) {
                ImDrawList* dl = ImGui::GetWindowDrawList();
                ImVec2 mousePos = ImGui::GetMousePos();
                
                // Draw selection rectangle (dotted/dashed style)
                dl->AddRect(boxSelectStartPos, mousePos, IM_COL32(255, 180, 50, 180), 0.0f, 0, 1.0f);
                dl->AddRectFilled(boxSelectStartPos, mousePos, IM_COL32(255, 180, 50, 20));
                
                if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
                    isBoxSelecting = false;
                    
                    float minX = std::min(boxSelectStartPos.x, mousePos.x);
                    float maxX = std::max(boxSelectStartPos.x, mousePos.x);
                    float minY = std::min(boxSelectStartPos.y, mousePos.y);
                    float maxY = std::max(boxSelectStartPos.y, mousePos.y);
                    
                    // Only select if the drag distance is significant (not a click)
                    if (std::abs(maxX - minX) > 4.0f || std::abs(maxY - minY) > 4.0f) {
                        for (auto& node : graph.nodes) {
                            if (!isNodeVisible(graph, *node)) continue;
                            ImVec2 nPos = nodeToScreen(node->x, node->y);
                            // Estimate node center
                            ImVec2 nCenter(nPos.x + 80.0f * zoom, nPos.y + 40.0f * zoom);
                            
                            if (nCenter.x >= minX && nCenter.x <= maxX &&
                                nCenter.y >= minY && nCenter.y <= maxY) {
                                if (std::find(selectedNodeIds.begin(), selectedNodeIds.end(), node->id) == selectedNodeIds.end()) {
                                    selectedNodeIds.push_back(node->id);
                                }
                            }
                        }
                        if (!selectedNodeIds.empty()) {
                            selectedNodeId = selectedNodeIds.back();
                            if (onNodeSelected) onNodeSelected(selectedNodeId);
                        }
                    }
                }
            }
            
            // Zoom is a canvas-window gesture. Testing only the background item made the
            // wheel stop whenever a node/group InvisibleButton overlapped the pointer.
            const ImVec2 mouseNow = ImGui::GetMousePos();
            const bool mouseInsideCanvas =
                mouseNow.x >= canvasPos_.x && mouseNow.y >= canvasPos_.y &&
                mouseNow.x < canvasPos_.x + canvasSize_.x &&
                mouseNow.y < canvasPos_.y + canvasSize_.y;
            const bool canvasHovered = mouseInsideCanvas &&
                ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem |
                                       ImGuiHoveredFlags_ChildWindows);
            if (canvasHovered && !canvasGestureBlocked) {
                float wheel = ImGui::GetIO().MouseWheel;
                if (wheel != 0) {
                    float oldZoom = zoom;
                    zoom += wheel * 0.1f;
                    zoom = std::clamp(zoom, 0.2f, 3.0f);
                    
                    // Zoom towards mouse
                    ImVec2 mouse = ImGui::GetMousePos();
                    ImVec2 mouseRel = ImVec2(mouse.x - canvasPos_.x - scrollX,
                                              mouse.y - canvasPos_.y - scrollY);
                    scrollX -= mouseRel.x * (zoom - oldZoom) / oldZoom;
                    scrollY -= mouseRel.y * (zoom - oldZoom) / oldZoom;
                }
            }
            
            // Context menu (right click on background) — deferred, see
            // backgroundContextMenuRequested_'s comment: only actually opens (below, right
            // before drawPopups()) if no node/group claims this same click afterward.
            if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
                selectedGroupId = 0;
                backgroundContextMenuRequested_ = true;
                releasedLinkPinId = 0;
                mousePosOnRightClick = ImVec2((ImGui::GetMousePos().x - canvasPos_.x - scrollX) / zoom,
                                               (ImGui::GetMousePos().y - canvasPos_.y - scrollY) / zoom);
            }
            
            // Key presses & shortcuts
            if (ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows)) {
                // Ctrl+G Grouping Shortcut
                if (ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_G)) {
                    if (!selectedNodeIds.empty()) {
                        float minX = FLT_MAX, minY = FLT_MAX, maxX = -FLT_MAX, maxY = -FLT_MAX;
                        for (uint32_t nid : selectedNodeIds) {
                            NodeBase* n = graph.getNode(nid);
                            if (n) {
                                minX = std::min(minX, n->x);
                                minY = std::min(minY, n->y);
                                maxX = std::max(maxX, n->x + 160.0f);
                                maxY = std::max(maxY, n->y + 100.0f);
                            }
                        }
                        if (minX != FLT_MAX) {
                            uint32_t gid = graph.createGroup("New Group", 
                                ImVec2(minX - 16.0f, minY - 32.0f), 
                                ImVec2(maxX - minX + 32.0f, maxY - minY + 48.0f));
                            for (uint32_t nid : selectedNodeIds) {
                                graph.addNodeToGroup(nid, gid);
                            }
                            if (onGraphModified) onGraphModified();
                        }
                    }
                }
                
                // Delete keys
                if (ImGui::IsKeyPressed(ImGuiKey_Delete) || ImGui::IsKeyPressed(ImGuiKey_Backspace)) {
                    if (selectedPortalId != 0) {
                        graph.portals.erase(std::remove_if(graph.portals.begin(), graph.portals.end(),
                            [this](const WirePortal& portal) { return portal.id == selectedPortalId; }),
                            graph.portals.end());
                        selectedPortalId = 0;
                        draggingPortalId = 0;
                        if (onGraphModified) onGraphModified();
                    } else if (selectedLinkId != 0) {
                        graph.removeLink(selectedLinkId);
                        selectedLinkId = 0;
                        if (onGraphModified) onGraphModified();
                    } else if (!selectedNodeIds.empty()) {
                        for (uint32_t nid : selectedNodeIds) {
                            graph.removeNode(nid);
                        }
                        selectedNodeIds.clear();
                        selectedNodeId = 0;
                        if (onGraphModified) onGraphModified();
                    } else if (selectedNodeId != 0) {
                        graph.removeNode(selectedNodeId);
                        selectedNodeId = 0;
                        if (onGraphModified) onGraphModified();
                    }
                }
            }
            
            // Cancel link creation
            if (isCreatingLink) {
                if (ImGui::IsMouseClicked(ImGuiMouseButton_Right) || 
                    ImGui::IsKeyPressed(ImGuiKey_Escape)) {
                    isCreatingLink = false;
                    linkStartPinId = 0;
                }
                
                // Finish link on release
                if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
                    ImVec2 mouse = ImGui::GetMousePos();
                    uint32_t targetPin = findClosestPin(graph, mouse, 20.0f * zoom);
                    
                    if (targetPin != 0 && targetPin != linkStartPinId) {
                        graph.addLink(linkStartPinId, targetPin);
                        if (onGraphModified) onGraphModified();
                    } else if (targetPin == 0) {
                        releasedLinkPinId = linkStartPinId;
                        mousePosOnRightClick = ImVec2((mouse.x - canvasPos_.x - scrollX) / zoom,
                                                      (mouse.y - canvasPos_.y - scrollY) / zoom);
                        if (onDrawLinkCreateMenu) {
                            linkCreateMenuRequested_ = true;
                        } else {
                            // Backward-compatible fallback for domains that have
                            // not supplied a link-drop node picker yet.
                            backgroundContextMenuRequested_ = true;
                        }
                    }
                    
                    isCreatingLink = false;
                    linkStartPinId = 0;
                }
            }
            
            // Dragging interactions
            // Dragging interactions
            if (resizingNodeId != 0) {
                if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
                    NodeBase* node = graph.getNode(resizingNodeId);
                    if (node) {
                        node->uiWidth += ImGui::GetIO().MouseDelta.x / zoom;
                        node->uiWidth = std::clamp(node->uiWidth, 110.0f, 360.0f);
                    }
                } else {
                    resizingNodeId = 0;
                }
            } else if (resizingGroupId != 0) {
                if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
                    NodeGroup* group = graph.getGroup(resizingGroupId);
                    if (group) {
                        group->size.x += ImGui::GetIO().MouseDelta.x / zoom;
                        group->size.y += ImGui::GetIO().MouseDelta.y / zoom;
                        group->size.x = std::max(100.0f, group->size.x);
                        group->size.y = std::max(80.0f, group->size.y);
                    }
                } else {
                    resizingGroupId = 0;
                }
            } else if (draggingNodeId != 0) {
                if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
                    if (ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.0f)) {
                        ImVec2 delta = ImVec2(ImGui::GetIO().MouseDelta.x / zoom, ImGui::GetIO().MouseDelta.y / zoom);
                        
                        bool isDraggedNodeSelected = std::find(selectedNodeIds.begin(), selectedNodeIds.end(), draggingNodeId) != selectedNodeIds.end();
                        if (isDraggedNodeSelected) {
                            // Move all selected nodes together
                            for (uint32_t nid : selectedNodeIds) {
                                NodeBase* n = graph.getNode(nid);
                                if (n) {
                                    n->x += delta.x;
                                    n->y += delta.y;
                                }
                            }
                        } else {
                            // Only move the single dragged node
                            NodeBase* n = graph.getNode(draggingNodeId);
                            if (n) {
                                n->x += delta.x;
                                n->y += delta.y;
                            }
                        }
                    }
                } else {
                    // Release! Check if we should add/remove the dragged nodes from groups
                    bool isDraggedNodeSelected = std::find(selectedNodeIds.begin(), selectedNodeIds.end(), draggingNodeId) != selectedNodeIds.end();
                    std::vector<uint32_t> nodesToProcess = isDraggedNodeSelected ? selectedNodeIds : std::vector<uint32_t>{draggingNodeId};
                    
                    for (uint32_t nid : nodesToProcess) {
                        NodeBase* node = graph.getNode(nid);
                        if (node) {
                            checkNodeDroppedOnLink(graph, *node);
                            NodeGroup* currentGroup = node->groupId ? graph.getGroup(node->groupId) : nullptr;
                            const bool logicalLayer = currentGroup &&
                                (currentGroup->comment == "Auto-managed terrain layer" ||
                                 focusedGroupId == currentGroup->id);
                            if (logicalLayer) {
                                // Layer ownership is semantic, never a hit-test side effect.
                                // Moving anywhere on the infinite layer canvas keeps membership;
                                // the All-view frame simply refits to the new member bounds.
                                fitGroupToContents(graph, *currentGroup);
                                continue;
                            }
                            float nodeCenterX = node->x + 80.0f;
                            float nodeCenterY = node->y + 40.0f;
                            uint32_t newGroupId = 0;
                            
                            for (auto& g : graph.groups) {
                                // Auto-managed layers accept nodes only through explicit
                                // Move to Layer / active-layer creation, never spatial overlap.
                                if (g.comment == "Auto-managed terrain layer") continue;
                                if (nodeCenterX >= g.position.x && nodeCenterX <= g.position.x + g.size.x &&
                                    nodeCenterY >= g.position.y && nodeCenterY <= g.position.y + g.size.y) {
                                    newGroupId = g.id;
                                    break;
                                }
                            }
                            
                            if (node->groupId != newGroupId) {
                                graph.removeNodeFromGroups(node->id);
                                if (newGroupId != 0) {
                                    graph.addNodeToGroup(node->id, newGroupId);
                                }
                                if (onGraphModified) onGraphModified();
                            }
                        }
                    }
                    draggingNodeId = 0;
                }
            } else if (draggingGroupId != 0) {
                if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
                    if (ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.0f)) {
                        NodeGroup* group = graph.getGroup(draggingGroupId);
                        if (group) {
                            float dx = ImGui::GetIO().MouseDelta.x / zoom;
                            float dy = ImGui::GetIO().MouseDelta.y / zoom;
                            group->position.x += dx;
                            group->position.y += dy;
                            
                            // Move all nodes associated with this group
                            for (uint32_t nodeId : group->nodeIds) {
                                NodeBase* n = graph.getNode(nodeId);
                                if (n) {
                                    n->x += dx;
                                    n->y += dy;
                                }
                            }
                        }
                    }
                } else {
                    draggingGroupId = 0;
                }
            }
        }

        // ========================================================================
        // NODE
        // ========================================================================
        
        static int visiblePinCount(const std::vector<Pin>& pins) {
            int n = 0;
            for (const auto& p : pins) if (!p.hidden) ++n;
            return n;
        }

        /// Row layout of a node's INPUT side: every visible pin gets a row, and every section
        /// (NodeBase::inputSectionLabel) gets a header row above its first pin.
        ///
        /// The header row is emitted even when the section is COLLAPSED and all of its pins
        /// are therefore hidden — otherwise a section you close disappears entirely and there
        /// is nothing left on the node to click to bring it back.
        ///
        /// Both the drawing path and the CULLED path go through this. They used to compute
        /// pin Y independently as index*spacing, and any divergence between the two shows up
        /// as links detaching from their sockets the moment a node scrolls off screen.
        struct InputRowLayout {
            std::vector<int> rowOfInput;                  ///< -1 = hidden
            std::vector<std::pair<int, int>> sections;    ///< (input index that starts it, header row)
            int totalRows = 0;
        };

        static InputRowLayout buildInputRowMap(const NodeBase& node) {
            InputRowLayout L;
            L.rowOfInput.assign(node.inputs.size(), -1);
            int row = 0;
            for (size_t i = 0; i < node.inputs.size(); ++i) {
                const int idx = static_cast<int>(i);
                if (node.inputSectionLabel(idx) != nullptr) {
                    L.sections.emplace_back(idx, row);
                    ++row;
                }
                if (node.inputs[i].hidden) continue;
                L.rowOfInput[i] = row++;
            }
            L.totalRows = row;
            return L;
        }

        /// One socket-section heading on the node body: a chevron, the label, and a hairline
        /// to the node's right edge. DRAWING ONLY — the click target is created after
        /// ChannelsMerge (see drawNode), because every other ImGui item on the node is too
        /// and mixing real widgets into a split draw channel is how you get a node that
        /// renders under its own body.
        /// Shared geometry for the section-header adornments (enable toggle + "..." overflow).
        /// The visual pass and the interaction pass both need the same rects — two copies of
        /// this arithmetic is how a hit target drifts off its pixels.
        void sectionAdornMetrics(NodeBase& node, int secFirst, float& toggleW, float& extraW) {
            const float tR = std::clamp(4.0f * zoom, 3.0f, 5.5f);
            toggleW = (node.inputSectionToggle(secFirst) != nullptr) ? tR * 2.0f + 8.0f : 0.0f;
            extraW  = node.inputSectionHasExtra(secFirst) ? std::max(12.0f, 14.0f * zoom) : 0.0f;
        }

        /// `rightInset` reserves space for the header adornments so the hairline stops before
        /// them; `labelAlpha` < 1 dims a DISABLED feature-group's title — the "this whole
        /// group is off" cue you can read without finding the toggle first.
        void drawInputSectionHeaderVisual(ImDrawList* dl, bool open, const char* label,
                                          ImVec2 rowTopLeft, float nodeWidth,
                                          float rowHeight, float padding,
                                          float rightInset = 0.0f, float labelAlpha = 1.0f) {
            const float cy = rowTopLeft.y + rowHeight * 0.5f;
            const float x0 = rowTopLeft.x + padding;

            const float s = std::max(3.0f, 3.5f * zoom);
            const int la = static_cast<int>(255.0f * labelAlpha);
            const ImU32 chev = IM_COL32(190, 195, 205, la);
            if (open) {
                dl->AddTriangleFilled(ImVec2(x0, cy - s * 0.6f), ImVec2(x0 + s * 2.0f, cy - s * 0.6f),
                                      ImVec2(x0 + s, cy + s * 0.8f), chev);
            } else {
                dl->AddTriangleFilled(ImVec2(x0, cy - s), ImVec2(x0 + s * 1.6f, cy),
                                      ImVec2(x0, cy + s), chev);
            }

            const float tx = x0 + s * 2.0f + 6.0f * zoom;
            const ImVec2 ts = ImGui::CalcTextSize(label);
            dl->AddText(ImVec2(tx, cy - ts.y * 0.5f), IM_COL32(205, 210, 220, la), label);

            const float lineX = tx + ts.x + 6.0f * zoom;
            const float rightX = rowTopLeft.x + nodeWidth - padding - rightInset;
            if (rightX > lineX) {
                dl->AddLine(ImVec2(lineX, cy), ImVec2(rightX, cy), IM_COL32(255, 255, 255, 22), 1.0f);
            }
        }

        void drawNode(ImDrawList* dl, NodeBase& node, GraphBase& graph) {
            ImVec2 pos = nodeToScreen(node.x, node.y);
            float padding = scaleNodeChromeMetric(zoom, 10.0f, 7.0f, 14.0f);

            // Culling must use the SAME dynamic layout as rendering. A fixed 400px
            // estimate dropped tall field/output nodes while their lower sockets were still
            // visible, making them appear to vanish permanently during pan/zoom.
            float customW = node.getCustomWidth();
            float titleW = ImGui::CalcTextSize(node.metadata.displayName.c_str()).x + padding * 2;
            if (titleW < 10) titleW = ImGui::CalcTextSize(node.name.c_str()).x + padding * 2;
            const bool showInlineContent = config.inlineNodeContent && !node.collapsed &&
                zoom >= config.inlineContentMinZoom && node.wantsInlineContent();
            const InputRowLayout rows = buildInputRowMap(node);
            const int inputRows = rows.totalRows;
            NodeChromeLayout chrome = buildNodeChromeLayout(node, zoom,
                customW > 0.0f ? customW * zoom : 160.0f * zoom,
                static_cast<size_t>(inputRows), visiblePinCount(node.outputs), titleW,
                showInlineContent ? node.inlineContentHeight_ : 0.0f);
            chrome.cornerRadius = scaleNodeChromeMetric(zoom, config.nodeRounding, 3.0f, 12.0f);
            const float cullMargin = std::max(8.0f, chrome.shadowOffset + 3.0f);
            
            // Frustum culling - skip nodes completely outside canvas
            if (pos.x + chrome.width + cullMargin < canvasPos_.x ||
                pos.x - cullMargin > canvasPos_.x + canvasSize_.x ||
                pos.y + chrome.height + cullMargin < canvasPos_.y ||
                pos.y - cullMargin > canvasPos_.y + canvasSize_.y) {
                // Still need to cache pin positions for link drawing
                const float inputStartY = getNodePinStartY(chrome, pos.y, static_cast<size_t>(inputRows));
                const float outputStartY = getNodePinStartY(chrome, pos.y, visiblePinCount(node.outputs));
                const float rowStep = node.collapsed ? chrome.collapsedPinSpacing : chrome.pinSpacing;
                int visibleInput = 0;
                for (int i = 0; i < (int)node.inputs.size(); i++) {
                    if (node.inputs[i].hidden) continue;
                    const int row = node.collapsed ? visibleInput : rows.rowOfInput[i];
                    pinPositions_[node.inputs[i].id] = ImVec2(pos.x, inputStartY + row * rowStep);
                    ++visibleInput;
                }
                int visOut = 0;
                for (int i = 0; i < (int)node.outputs.size(); i++) {
                    if (node.outputs[i].hidden) continue;
                    pinPositions_[node.outputs[i].id] = ImVec2(
                        pos.x + chrome.width, outputStartY + visOut * rowStep);
                    ++visOut;
                }
                return; // Skip full rendering
            }

            float finalWidth = chrome.width;
            float finalHeight = chrome.height;
            float headerH = chrome.headerHeight;
            float cornerRadius = chrome.cornerRadius;
            float shadowOffset = chrome.shadowOffset;
            bool showTitle = chrome.showTitle;
            bool showPinLabels = chrome.showPinLabels;
            bool isCollapsed = chrome.collapsed;

            // Two-pass rendering
            dl->ChannelsSplit(2);
            dl->ChannelsSetCurrent(1);

            float inputPinStartY = getNodePinStartY(chrome, pos.y, static_cast<size_t>(inputRows));
            float outputPinStartY = getNodePinStartY(chrome, pos.y, visiblePinCount(node.outputs));

            // Inline pin value widgets are real ImGui items: only near 1:1 zoom, where an
            // unscaled frame still lines up with its (zoom-scaled) pin row.
            const bool showPinWidgets = !isCollapsed && node.wantsInlinePinWidgets() &&
                zoom >= config.inlineContentMinZoom;

            const float rowStep = isCollapsed ? chrome.collapsedPinSpacing : chrome.pinSpacing;
            int visIn = 0;
            for (int i = 0; i < (int)node.inputs.size(); i++) {
                Pin& pin = node.inputs[i];
                if (pin.hidden) continue;  // collapsed pin group — not drawn, not interactable
                const int row = isCollapsed ? visIn : rows.rowOfInput[i];
                ImVec2 pinPos(pos.x, inputPinStartY + row * rowStep);
                ++visIn;
                pinPositions_[pin.id] = pinPos;
                drawPin(dl, pinPos, pin, true, (!isCollapsed && showPinLabels) ? chrome.labelWidth : 0.0f);
            }

            // Section headings (a collapsed node is a title bar with sockets — headings there
            // would be noise, so they are skipped along with the labels).
            if (!isCollapsed && showTitle) {
                for (const auto& sec : rows.sections) {
                    const char* label = node.inputSectionLabel(sec.first);
                    if (!label) continue;
                    float tw = 0.0f, ew = 0.0f;
                    sectionAdornMetrics(node, sec.first, tw, ew);
                    const bool* t = node.inputSectionToggle(sec.first);
                    drawInputSectionHeaderVisual(dl, node.isInputSectionOpen(sec.first), label,
                                                 ImVec2(pos.x, inputPinStartY + sec.second * rowStep),
                                                 finalWidth, rowStep, padding,
                                                 tw + ew + ((tw + ew) > 0.0f ? 4.0f : 0.0f),
                                                 (t && !*t) ? 0.45f : 1.0f);
                }
            }

            int visOut = 0;
            for (int i = 0; i < (int)node.outputs.size(); i++) {
                Pin& pin = node.outputs[i];
                if (pin.hidden) continue;
                ImVec2 pinPos(pos.x + finalWidth, outputPinStartY + visOut * (isCollapsed ? chrome.collapsedPinSpacing : chrome.pinSpacing));
                ++visOut;
                pinPositions_[pin.id] = pinPos;
                drawPin(dl, pinPos, pin, false, (!isCollapsed && showPinLabels) ? chrome.labelWidth : 0.0f);
            }

            if (showTitle) {
                const std::string fullTitle = node.metadata.displayName.empty()
                    ? node.name
                    : node.metadata.displayName;
                const std::string title = fitTextToWidth(fullTitle, finalWidth - padding * 2.0f);
                if (!title.empty()) {
                    ImVec2 titleMin(pos.x + padding, pos.y + 4.0f);
                    ImVec2 titleMax(pos.x + finalWidth - padding, pos.y + headerH - 4.0f);
                    dl->PushClipRect(titleMin, titleMax, true);
                    dl->AddText(ImVec2(pos.x + padding, pos.y + (headerH - ImGui::GetTextLineHeight()) * 0.5f - 1.0f),
                        IM_COL32(245, 247, 250, 255), title.c_str());
                    dl->PopClipRect();
                }
            }
            
            // Background (Channel 0)
            dl->ChannelsSetCurrent(0);
            
            bool isSelected = std::find(selectedNodeIds.begin(), selectedNodeIds.end(), node.id) != selectedNodeIds.end();
            
            ImU32 headerCol = node.metadata.headerColor;
            if (headerCol == 0) {
                headerCol = ImGui::ColorConvertFloat4ToU32(node.headerColor);
            }
            
            ImU32 borderCol = isSelected ? config.nodeSelectedColor : config.nodeBorderColor;
            float borderW = isSelected ? config.nodeBorderWidth * 1.25f : config.nodeBorderWidth;
            
            // Multi-layered soft drop shadows
            dl->AddRectFilled(
                ImVec2(pos.x + shadowOffset * 1.5f, pos.y + shadowOffset * 1.5f),
                ImVec2(pos.x + finalWidth + shadowOffset * 1.5f, pos.y + finalHeight + shadowOffset * 1.5f),
                IM_COL32(0, 0, 0, 25), cornerRadius);
            dl->AddRectFilled(
                ImVec2(pos.x + shadowOffset, pos.y + shadowOffset),
                ImVec2(pos.x + finalWidth + shadowOffset, pos.y + finalHeight + shadowOffset),
                IM_COL32(0, 0, 0, 45), cornerRadius);

            // Body
            dl->AddRectFilled(pos, ImVec2(pos.x + finalWidth, pos.y + finalHeight),
                config.nodeBodyColor, cornerRadius);

            // Section group cards — what actually makes 30 rows read as 8 GROUPS instead of
            // one wall of sockets: each section sits on its own faint panel, its header row a
            // touch brighter so it reads as a title. Pure draw-list; the hairline + chevron
            // from drawInputSectionHeaderVisual land on top of this.
            if (!isCollapsed && showTitle && !rows.sections.empty()) {
                for (size_t s = 0; s < rows.sections.size(); ++s) {
                    const int startRow = rows.sections[s].second;
                    // A collapsed section's card is just its header row (its pins are hidden,
                    // so the NEXT section's header row bounds it immediately).
                    const int endRow = (s + 1 < rows.sections.size())
                        ? rows.sections[s + 1].second - 1
                        : inputRows - 1;
                    const float y0 = inputPinStartY + startRow * chrome.pinSpacing;
                    const float y1 = inputPinStartY + endRow * chrome.pinSpacing + chrome.pinSpacing * 0.55f;
                    dl->AddRectFilled(ImVec2(pos.x + 3.0f, y0), ImVec2(pos.x + finalWidth - 3.0f, y1),
                        IM_COL32(255, 255, 255, 7), 4.0f);
                    dl->AddRectFilled(ImVec2(pos.x + 3.0f, y0),
                        ImVec2(pos.x + finalWidth - 3.0f, y0 + chrome.pinSpacing * 0.92f),
                        IM_COL32(255, 255, 255, 11), 4.0f, ImDrawFlags_RoundCornersTop);
                }
            }

            // Modern category color strip at the very top of the node (thin, Gaea/Houdini-style)
            // Drawn first with a taller height to prevent ImGui from clamping the corner rounding due to height constraints
            float stripeHeight = 3.5f * zoom;
            float stripeDrawHeight = std::max(stripeHeight, cornerRadius);
            dl->AddRectFilled(pos, ImVec2(pos.x + finalWidth, pos.y + stripeDrawHeight),
                headerCol, cornerRadius, ImDrawFlags_RoundCornersTop);
            
            // Header Base (Integrated Dark Charcoal)
            // Drawn on top of the stripe, starting from stripeHeight, to overlay/clip the excess height of the stripe
            dl->AddRectFilled(ImVec2(pos.x, pos.y + stripeHeight), ImVec2(pos.x + finalWidth, pos.y + headerH),
                IM_COL32(26, 28, 33, 250), isCollapsed ? cornerRadius : 0.0f, isCollapsed ? ImDrawFlags_RoundCornersBottom : 0);
            
            // Subtle premium accent line just below the color stripe
            dl->AddLine(
                ImVec2(pos.x + 1.0f, pos.y + stripeHeight),
                ImVec2(pos.x + finalWidth - 1.0f, pos.y + stripeHeight),
                IM_COL32(255, 255, 255, 30),
                1.0f
            );

            dl->AddLine(
                ImVec2(pos.x + 1.0f, pos.y + headerH),
                ImVec2(pos.x + finalWidth - 1.0f, pos.y + headerH),
                IM_COL32(255, 255, 255, 12),
                1.0f);
            
            // Border
            dl->AddRect(pos, ImVec2(pos.x + finalWidth, pos.y + finalHeight),
                borderCol, cornerRadius, 0, std::max(1.0f, borderW * zoom));

            // Background-evaluation indicator: pulsing border + overall-progress
            // fill bar along the bottom edge, shown only on the node the graph
            // reports as currently active (see GraphBase::isEvaluatingAsync /
            // currentAsyncNodeId / asyncEvalProgress — default no-ops for graphs
            // that don't support background evaluation, so this is a no-op there).
            const NodeEvaluationState evalState = graph.asyncNodeState(node.id);
            const bool nodeRunning = graph.isEvaluatingAsync() &&
                (evalState == NodeEvaluationState::Running || graph.currentAsyncNodeId() == node.id);
            if (nodeRunning) {
                float pulse = 0.5f + 0.5f * sinf((float)ImGui::GetTime() * 6.0f);
                ImU32 glowCol = IM_COL32(120, 220, 255, (int)(120 + 100 * pulse));
                dl->AddRect(ImVec2(pos.x - 2.0f, pos.y - 2.0f), ImVec2(pos.x + finalWidth + 2.0f, pos.y + finalHeight + 2.0f),
                    glowCol, cornerRadius, 0, std::max(2.0f, borderW * zoom * 1.5f));

                float barH = std::max(3.0f, 4.0f * zoom);
                float progress = std::clamp(graph.asyncEvalProgress(), 0.0f, 1.0f);
                ImVec2 barMin(pos.x + 1.0f, pos.y + finalHeight - barH - 1.0f);
                dl->AddRectFilled(barMin, ImVec2(pos.x + finalWidth - 1.0f, pos.y + finalHeight - 1.0f),
                    IM_COL32(20, 24, 30, 200));
                dl->AddRectFilled(barMin, ImVec2(pos.x + 1.0f + (finalWidth - 2.0f) * progress, pos.y + finalHeight - 1.0f),
                    IM_COL32(120, 220, 255, 235));
            }

            // Persist per-node state after the active cursor moves on. This makes
            // a pull evaluation readable as a sequence across the whole DAG:
            // blue = processing, green = completed, cyan = cache hit, red = failed.
            const bool showDirtyState = !graph.isEvaluatingAsync() && node.dirty;
            if (evalState != NodeEvaluationState::Idle || showDirtyState) {
                ImU32 stateColor = IM_COL32(120, 220, 255, 255);
                if (evalState == NodeEvaluationState::Idle && showDirtyState) stateColor = IM_COL32(240, 170, 70, 255);
                else if (evalState == NodeEvaluationState::Completed) stateColor = IM_COL32(90, 220, 135, 255);
                else if (evalState == NodeEvaluationState::Cached) stateColor = IM_COL32(80, 205, 225, 255);
                else if (evalState == NodeEvaluationState::Failed) stateColor = IM_COL32(245, 85, 85, 255);

                const float radius = std::clamp(4.0f * zoom, 3.0f, 6.0f);
                const ImVec2 center(pos.x + finalWidth - padding * 0.75f,
                                    pos.y + headerH * 0.5f);
                dl->AddCircleFilled(center, radius + 1.5f, IM_COL32(12, 14, 18, 230));
                dl->AddCircleFilled(center, radius, stateColor);
                if (evalState == NodeEvaluationState::Completed || evalState == NodeEvaluationState::Cached) {
                    dl->AddRect(ImVec2(pos.x - 1.0f, pos.y - 1.0f),
                                ImVec2(pos.x + finalWidth + 1.0f, pos.y + finalHeight + 1.0f),
                                (evalState == NodeEvaluationState::Cached)
                                    ? IM_COL32(80, 205, 225, 80)
                                    : IM_COL32(90, 220, 135, 65),
                                cornerRadius, 0, 1.0f);
                }
            }

            dl->ChannelsMerge();
            
            // Interaction - only if node overlaps canvas
            ImVec2 nodeEnd(pos.x + finalWidth, pos.y + finalHeight);
            bool overlapsCanvas = (pos.x < canvasPos_.x + canvasSize_.x && nodeEnd.x > canvasPos_.x &&
                                    pos.y < canvasPos_.y + canvasSize_.y && nodeEnd.y > canvasPos_.y);
            
            if (overlapsCanvas) {
                float toggleSize = chrome.toggleSize;
                ImVec2 toggleMin(
                    pos.x + finalWidth - toggleSize - padding * 0.6f,
                    pos.y + (headerH - toggleSize) * 0.5f);
                ImVec2 toggleMax(toggleMin.x + toggleSize, toggleMin.y + toggleSize);

                ImGui::SetCursorScreenPos(toggleMin);
                ImGui::PushID(node.id);
                ImGui::InvisibleButton("NodeCollapseToggle", ImVec2(toggleSize, toggleSize));
                bool toggleHovered = ImGui::IsItemHovered();
                if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
                    node.collapsed = !node.collapsed;
                }
                if (toggleHovered) {
                    ImGui::SetTooltip("%s", node.collapsed ? "Expand node" : "Collapse node");
                }
                ImGui::PopID();

                dl->AddRectFilled(toggleMin, toggleMax,
                    toggleHovered ? IM_COL32(255, 255, 255, 28) : IM_COL32(0, 0, 0, 26),
                    toggleSize * 0.28f);
                dl->AddLine(
                    ImVec2(toggleMin.x + 4.0f, toggleMin.y + toggleSize * 0.5f),
                    ImVec2(toggleMax.x - 4.0f, toggleMin.y + toggleSize * 0.5f),
                    IM_COL32(245, 247, 250, 220), 1.4f);
                if (node.collapsed) {
                    dl->AddLine(
                        ImVec2(toggleMin.x + toggleSize * 0.5f, toggleMin.y + 4.0f),
                        ImVec2(toggleMin.x + toggleSize * 0.5f, toggleMax.y - 4.0f),
                        IM_COL32(245, 247, 250, 220), 1.4f);
                }

                float resizeHandleWidth = chrome.resizeHandleWidth;
                ImVec2 resizeMin(pos.x + finalWidth - resizeHandleWidth * 0.5f, pos.y + headerH);
                ImVec2 resizeMax(pos.x + finalWidth + resizeHandleWidth * 0.5f, pos.y + finalHeight);
                ImGui::SetCursorScreenPos(ImVec2(resizeMin.x, resizeMin.y));
                ImGui::PushID(node.id + 600000u);
                ImGui::InvisibleButton("NodeResizeHandle", ImVec2(resizeMax.x - resizeMin.x, resizeMax.y - resizeMin.y));
                bool resizeHovered = ImGui::IsItemHovered();
                if (resizeHovered || resizingNodeId == node.id) {
                    ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
                }
                if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.0f)) {
                    resizingNodeId = node.id;
                }
                if (resizeHovered) {
                    ImGui::SetTooltip("Drag to resize node");
                }
                ImGui::PopID();
                dl->AddLine(
                    ImVec2(pos.x + finalWidth - 1.0f, pos.y + headerH + 4.0f),
                    ImVec2(pos.x + finalWidth - 1.0f, pos.y + finalHeight - 4.0f),
                    resizeHovered || resizingNodeId == node.id ? IM_COL32(255, 255, 255, 90) : IM_COL32(255, 255, 255, 34),
                    1.0f);

                // Clamp interaction rect to canvas
                ImVec2 interactPos = pos;
                ImVec2 interactSize(std::max(24.0f, finalWidth - toggleSize - padding * 1.4f), headerH);
                
                // Don't set cursor outside canvas bounds
                interactPos.x = std::max(interactPos.x, canvasPos_.x);
                interactPos.y = std::max(interactPos.y, canvasPos_.y);
                interactPos.x = std::min(interactPos.x, canvasPos_.x + canvasSize_.x - 10);
                interactPos.y = std::min(interactPos.y, canvasPos_.y + canvasSize_.y - 10);
                
                ImGui::SetCursorScreenPos(interactPos);
                ImGui::PushID(node.id + 1000000u);
                ImGui::InvisibleButton("NodeHeader", ImVec2(std::min(interactSize.x, 200.0f * zoom), interactSize.y));
            
            if (ImGui::IsItemActive()) {
                if (draggingNodeId == 0) {
                    draggingNodeId = node.id;
                }
            }
            
            if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
                bool isShift = ImGui::GetIO().KeyShift;
                auto it = std::find(selectedNodeIds.begin(), selectedNodeIds.end(), node.id);
                if (isShift) {
                    if (it != selectedNodeIds.end()) {
                        selectedNodeIds.erase(it);
                        if (selectedNodeId == node.id) {
                            selectedNodeId = selectedNodeIds.empty() ? 0 : selectedNodeIds.back();
                        }
                    } else {
                        selectedNodeIds.push_back(node.id);
                        selectedNodeId = node.id;
                    }
                } else {
                    if (it == selectedNodeIds.end()) {
                        selectedNodeIds.clear();
                        selectedNodeIds.push_back(node.id);
                    }
                    selectedNodeId = node.id;
                }
                selectedLinkId = 0;
                if (onNodeSelected) onNodeSelected(selectedNodeId);
            }
            
            // Double-click to collapse/expand
            if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                node.collapsed = !node.collapsed;
            }
            
            if (ImGui::IsItemClicked(ImGuiMouseButton_Right)) {
                // Right-clicking a node now also selects it and opens the built-in
                // LocalNodeContextPopup (Delete Node / group management, see drawPopups()) —
                // previously that popup body existed but nothing ever called OpenPopup for it,
                // so right-click on a node did nothing but invoke onNodeContextMenu (if a caller
                // bothered to set it up their own popup). Mirrors the group right-click pattern
                // below (selectedGroupId set immediately before its OpenPopup call).
                // NOTE: the actual OpenPopup() call is deferred to just after PopID() below —
                // see nodeContextMenuRequested_'s comment for why (id-stack scope mismatch).
                selectedNodeId = node.id;
                if (std::find(selectedNodeIds.begin(), selectedNodeIds.end(), node.id) == selectedNodeIds.end()) {
                    selectedNodeIds.clear();
                    selectedNodeIds.push_back(node.id);
                }
                nodeContextMenuRequested_ = true;
                if (onNodeContextMenu) onNodeContextMenu(node.id);
            }

            ImGui::PopID();
            if (nodeContextMenuRequested_) {
                nodeContextMenuRequested_ = false;
                ImGui::OpenPopup("LocalNodeContextPopup");
                contextMenuClaimedByNodeOrGroup_ = true;  // suppress the background popup below
            }
                // ── Socket sections + inline pin values ──────────────────────
                // Real ImGui items, so they belong here (after ChannelsMerge) with the rest
                // of the node's interaction — the pin loop above only drew their pixels.
                if (!isCollapsed && showTitle) {
                    for (const auto& sec : rows.sections) {
                        // Header adornments (enable toggle, "..." overflow popup). Drawn as
                        // node chrome — draw-list glyphs over invisible hit targets — NOT as
                        // stock ImGui widgets: a raw Checkbox on the header ignored the zoom,
                        // dwarfed the 3px chevron next to it, and read as a foreign control
                        // pasted on top of the node. They stay OUT of the click-to-collapse
                        // strip (two overlapping items both take the same click), so the
                        // strip ends where they begin.
                        bool* secToggle = node.inputSectionToggle(sec.first);
                        const bool hasExtra = node.inputSectionHasExtra(sec.first);
                        float toggleW = 0.0f, extraW = 0.0f;
                        sectionAdornMetrics(node, sec.first, toggleW, extraW);
                        const float reserved = (toggleW + extraW > 0.0f) ? toggleW + extraW + padding : 0.0f;
                        const float headerY = inputPinStartY + sec.second * rowStep;
                        const float cy = headerY + rowStep * 0.5f;

                        ImGui::SetCursorScreenPos(ImVec2(pos.x + 2.0f, headerY));
                        ImGui::PushID(static_cast<int>(node.id + 4000000u + static_cast<uint32_t>(sec.first)));
                        ImGui::InvisibleButton("##section",
                            ImVec2(std::max(8.0f, finalWidth - 4.0f - reserved), std::max(6.0f, rowStep)));
                        if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) node.toggleInputSection(sec.first);
                        if (ImGui::IsItemHovered()) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                        ImGui::PopID();

                        float rightX = pos.x + finalWidth - padding;
                        if (secToggle) {
                            rightX -= toggleW;
                            const ImVec2 c(rightX + toggleW * 0.5f, cy);
                            const float tR = std::clamp(4.0f * zoom, 3.0f, 5.5f);

                            ImGui::SetCursorScreenPos(ImVec2(rightX, cy - rowStep * 0.4f));
                            ImGui::PushID(static_cast<int>(node.id + 4500000u + static_cast<uint32_t>(sec.first)));
                            ImGui::InvisibleButton("##secOn", ImVec2(toggleW, rowStep * 0.8f));
                            const bool hov = ImGui::IsItemHovered();
                            if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
                                *secToggle = !*secToggle;
                                node.dirty = true;
                            }
                            if (hov) {
                                const char* lbl = node.inputSectionLabel(sec.first);
                                ImGui::SetTooltip("%s: %s (click to toggle)",
                                                  lbl ? lbl : "Feature", *secToggle ? "on" : "off");
                            }
                            ImGui::PopID();

                            // Power glyph: ring + tick. On = accent ring with a lit core, off =
                            // dim hollow ring — state readable without the tooltip.
                            const ImU32 col = *secToggle
                                ? (hov ? IM_COL32(150, 240, 190, 255) : IM_COL32(110, 215, 165, 235))
                                : (hov ? IM_COL32(200, 205, 215, 220) : IM_COL32(125, 130, 140, 150));
                            const float th = std::max(1.2f, 1.4f * zoom);
                            dl->AddCircle(c, tR, col, 0, th);
                            dl->AddLine(ImVec2(c.x, c.y - tR * 1.35f), ImVec2(c.x, c.y - tR * 0.15f), col, th);
                            if (*secToggle) dl->AddCircleFilled(c, tR * 0.4f, col);
                            rightX -= 2.0f;
                        }
                        if (hasExtra) {
                            rightX -= extraW;
                            ImGui::SetCursorScreenPos(ImVec2(rightX, cy - rowStep * 0.4f));
                            ImGui::PushID(static_cast<int>(node.id + 4600000u + static_cast<uint32_t>(sec.first)));
                            ImGui::InvisibleButton("##secExtra", ImVec2(extraW, rowStep * 0.8f));
                            const bool hov = ImGui::IsItemHovered();
                            if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) ImGui::OpenPopup("##secExtraPopup");
                            if (hov) ImGui::SetTooltip("More parameters");
                            if (ImGui::BeginPopup("##secExtraPopup")) {
                                node.drawInputSectionExtra(sec.first);
                                ImGui::EndPopup();
                            }

                            const ImU32 col = hov ? IM_COL32(220, 225, 235, 255) : IM_COL32(150, 155, 165, 190);
                            const float dR = std::max(1.1f, 1.4f * zoom);
                            const float dSp = std::max(3.5f, 4.2f * zoom);
                            const ImVec2 dc(rightX + extraW * 0.5f, cy);
                            dl->AddCircleFilled(ImVec2(dc.x - dSp, dc.y), dR, col);
                            dl->AddCircleFilled(dc, dR, col);
                            dl->AddCircleFilled(ImVec2(dc.x + dSp, dc.y), dR, col);
                            ImGui::PopID();
                        }
                    }

                    // Value editor for each UNCONNECTED pin, on the pin's own row. This is what
                    // puts the node's NUMBERS on the node instead of a bare column of socket
                    // names. A connected pin has no default to show — the wire drives it.
                    if (showPinWidgets) {
                        for (int i = 0; i < (int)node.inputs.size(); i++) {
                            const Pin& pin = node.inputs[i];
                            if (pin.hidden || graph.getInputSource(pin.id) != nullptr) continue;
                            const float labelW = ImGui::CalcTextSize(pin.name.c_str()).x;
                            const float wStart = pos.x + padding + labelW + 10.0f;
                            const float wWidth = (pos.x + finalWidth - padding - 4.0f) - wStart;
                            if (wWidth < 40.0f) continue;
                            const float pinY = inputPinStartY + rows.rowOfInput[i] * rowStep;
                            const float frameH = ImGui::GetFrameHeight();
                            ImGui::SetCursorScreenPos(ImVec2(wStart, pinY - frameH * 0.5f));
                            ImGui::PushID(static_cast<int>(node.id + 3000000u + static_cast<uint32_t>(i)));
                            if (node.drawInputInlineWidget(i, wWidth)) node.dirty = true;
                            ImGui::PopID();
                        }
                    }
                }

                // ── Inline node-body content ─────────────────────────────────
                // Rendered as real ImGui widgets below the pin block. The whole
                // canvas sits behind a SetNextItemAllowOverlap InvisibleButton,
                // so these items take click priority the same way the header
                // button does. Height is measured and cached for next frame's
                // chrome layout (1-frame lag on first show — invisible).
                if (showInlineContent) {
                    // inputRows, not the pin count: section headers occupy rows too, and the
                    // body content has to start below the LAST of them.
                    const int maxVis = std::max(inputRows, visiblePinCount(node.outputs));
                    const float contentTop = pos.y + headerH + chrome.bodyPadding +
                        static_cast<float>(maxVis) * chrome.pinSpacing + 2.0f;
                    const ImVec2 clipMin(pos.x + 2.0f, contentTop);
                    // First frame the height is unmeasured (0): clip generously so
                    // the measuring pass can run; correct height applies next frame.
                    const float clipBottom = (node.inlineContentHeight_ > 0.0f)
                        ? std::max(pos.y + finalHeight - 2.0f, clipMin.y + 1.0f)
                        : clipMin.y + 400.0f;
                    const ImVec2 clipMax(pos.x + finalWidth - 2.0f, clipBottom);
                    ImGui::SetCursorScreenPos(ImVec2(pos.x + padding, contentTop));
                    ImGui::PushID(static_cast<int>(node.id + 2000000u));
                    ImGui::PushClipRect(clipMin, clipMax, true);
                    ImGui::BeginGroup();
                    node.drawContent();
                    ImGui::EndGroup();
                    ImGui::PopClipRect();
                    node.inlineContentHeight_ = ImGui::GetItemRectSize().y + 8.0f;
                    ImGui::PopID();
                } else if (!node.wantsInlineContent()) {
                    node.inlineContentHeight_ = 0.0f;
                }
            } // end overlapsCanvas
        }

        // ========================================================================
        // PIN (Color-coded by type)
        // ========================================================================
        
        void drawPin(ImDrawList* dl, ImVec2 center, Pin& pin, bool isInput, float maxLabelWidth = 0.0f) {
            float r = scaleNodeChromeMetric(zoom, 6.0f, 4.5f, 9.0f);
            
            // Get color and shape from pin's cached values or compute
            ImU32 col = pin.cachedColor;
            PinShape shape = pin.cachedShape;
            
            if (col == 0) {
                auto visual = getDataTypeVisual(pin.dataType, pin.imageSemantic);
                col = visual.color;
                shape = visual.shape;
            }
            
            // Draw shape
            switch (shape) {
                case PinShape::Circle:
                    dl->AddCircleFilled(center, r, col);
                    dl->AddCircle(center, r, IM_COL32(255, 255, 255, 180), 0, 1.0f * zoom);
                    break;
                    
                case PinShape::Square: {
                    ImVec2 p1(center.x - r * 0.7f, center.y - r * 0.7f);
                    ImVec2 p2(center.x + r * 0.7f, center.y + r * 0.7f);
                    dl->AddRectFilled(p1, p2, col);
                    dl->AddRect(p1, p2, IM_COL32(255, 255, 255, 180), 0, 0, 1.0f * zoom);
                    break;
                }
                    
                case PinShape::Diamond: {
                    ImVec2 pts[4] = {
                        {center.x, center.y - r},
                        {center.x + r, center.y},
                        {center.x, center.y + r},
                        {center.x - r, center.y}
                    };
                    dl->AddConvexPolyFilled(pts, 4, col);
                    dl->AddPolyline(pts, 4, IM_COL32(255, 255, 255, 180), 
                        ImDrawFlags_Closed, 1.0f * zoom);
                    break;
                }

                case PinShape::Arrow: {
                    ImVec2 pts[3] = {
                        {center.x - r * 0.75f, center.y - r},
                        {center.x + r, center.y},
                        {center.x - r * 0.75f, center.y + r}
                    };
                    dl->AddConvexPolyFilled(pts, 3, col);
                    dl->AddPolyline(pts, 3, IM_COL32(255, 255, 255, 180),
                                    ImDrawFlags_Closed, 1.0f * zoom);
                    break;
                }
                    
                default:
                    dl->AddCircleFilled(center, r, col);
                    break;
            }
            
            // Interaction
            ImGui::SetCursorScreenPos(ImVec2(center.x - r * 2, center.y - r * 2));
            ImGui::PushID(pin.id);
            ImGui::InvisibleButton("Pin", ImVec2(r * 4, r * 4));
            
            if (ImGui::IsItemHovered()) {
                dl->AddCircle(center, r * 1.6f, IM_COL32(255, 255, 255, 100), 0, 2.0f);
                
                // Tooltip
                const auto visual = getDataTypeVisual(pin.dataType, pin.imageSemantic);
                ImGui::BeginTooltip();
                if (!pin.tooltip.empty()) ImGui::TextUnformatted(pin.tooltip.c_str());
                ImGui::TextDisabled("%s%s%s", visual.displayName,
                    pin.imageUnit != ImageUnit::Unknown ? " | " : "",
                    getImageUnitName(pin.imageUnit));
                ImGui::EndTooltip();
            }
            
            if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.0f)) {
                if (!isCreatingLink) {
                    isCreatingLink = true;
                    linkStartPinId = pin.id;
                    linkStartType = pin.dataType;
                }
            }
            
            ImGui::PopID();
            
            // Label
            if (zoom > 0.4f && maxLabelWidth > 6.0f) {
                const std::string label = fitTextToWidth(pin.name, maxLabelWidth);
                if (!label.empty()) {
                    ImVec2 txtSz = ImGui::CalcTextSize(label.c_str());
                    ImVec2 txtPos = isInput
                        ? ImVec2(center.x + r + 5 * zoom, center.y - txtSz.y * 0.5f)
                        : ImVec2(center.x - r - 5 * zoom - txtSz.x, center.y - txtSz.y * 0.5f);
                    ImVec2 clipMin = isInput
                        ? ImVec2(center.x + r + 3.0f * zoom, center.y - txtSz.y)
                        : ImVec2(center.x - r - 5.0f * zoom - maxLabelWidth, center.y - txtSz.y);
                    ImVec2 clipMax = isInput
                        ? ImVec2(center.x + r + 5.0f * zoom + maxLabelWidth, center.y + txtSz.y)
                        : ImVec2(center.x - r - 3.0f * zoom, center.y + txtSz.y);
                    dl->PushClipRect(clipMin, clipMax, true);
                    dl->AddText(txtPos, IM_COL32(220, 224, 230, 255), label.c_str());
                    dl->PopClipRect();
                }
            }
        }

        // ========================================================================
        // LINK
        // ========================================================================

        WirePortal* findWirePortalForPin(GraphBase& graph, uint32_t pinId) {
            for (WirePortal& portal : graph.portals) {
                if (portal.linkedPinId == pinId) return &portal;
            }
            return nullptr;
        }

        void createWireJunction(GraphBase& graph, uint32_t sourcePinId,
                                const ImVec2& screenPosition) {
            Pin* sourcePin = graph.findPin(sourcePinId);
            if (!sourcePin || sourcePin->kind != PinKind::Output) return;
            if (WirePortal* existing = findWirePortalForPin(graph, sourcePinId)) {
                selectedPortalId = existing->id;
                selectedLinkId = 0;
                return;
            }

            WirePortal portal;
            portal.id = graph.nextPortalId++;
            portal.name = "Junction " + std::to_string(portal.id);
            portal.linkedPinId = sourcePinId;
            portal.position = ImVec2(
                (screenPosition.x - canvasPos_.x - scrollX) / zoom,
                (screenPosition.y - canvasPos_.y - scrollY) / zoom);
            portal.color = sourcePin->cachedColor
                ? sourcePin->cachedColor : IM_COL32(200, 150, 50, 255);
            portal.kind = PinKind::Output;
            graph.portals.push_back(std::move(portal));
            selectedPortalId = graph.portals.back().id;
            selectedLinkId = 0;
            if (onGraphModified) onGraphModified();
        }

        void prepareWirePortals(GraphBase& graph) {
            for (WirePortal& portal : graph.portals) {
                Pin* pin = graph.findPin(portal.linkedPinId);
                auto sourceIt = pinPositions_.find(portal.linkedPinId);
                if (!pin || pin->kind != PinKind::Output || sourceIt == pinPositions_.end()) continue;
                if (portalSourcePositions_.find(portal.linkedPinId) != portalSourcePositions_.end()) continue;
                portalSourcePositions_[portal.linkedPinId] = sourceIt->second;
                const ImVec2 center = nodeToScreen(portal.position.x, portal.position.y);
                pinPositions_[portal.linkedPinId] = ImVec2(center.x + 14.0f * zoom, center.y);
            }
        }

        void drawWirePortals(ImDrawList* dl, GraphBase& graph) {
            for (WirePortal& portal : graph.portals) {
                Pin* pin = graph.findPin(portal.linkedPinId);
                const auto sourceIt = portalSourcePositions_.find(portal.linkedPinId);
                if (!pin || sourceIt == portalSourcePositions_.end()) continue;

                const ImVec2 center = nodeToScreen(portal.position.x, portal.position.y);
                const ImVec2 trunkEnd(center.x - 8.0f * zoom, center.y);
                const ImVec2 source = sourceIt->second;
                ImU32 color = pin->cachedColor ? pin->cachedColor : portal.color;
                const float cpDistance = std::max(45.0f * zoom,
                    std::abs(trunkEnd.x - source.x) * 0.45f);
                dl->AddBezierCubic(source, ImVec2(source.x + cpDistance, source.y),
                    ImVec2(trunkEnd.x - cpDistance, trunkEnd.y), trunkEnd, color,
                    std::max(1.0f, config.linkThickness * zoom));

                const float bodyRadius = std::max(6.0f, 7.0f * zoom);
                const bool selected = selectedPortalId == portal.id;
                ImVec2 diamond[4] = {
                    ImVec2(center.x, center.y - bodyRadius),
                    ImVec2(center.x + bodyRadius, center.y),
                    ImVec2(center.x, center.y + bodyRadius),
                    ImVec2(center.x - bodyRadius, center.y)
                };
                dl->AddConvexPolyFilled(diamond, 4, IM_COL32(32, 36, 46, 255));
                dl->AddPolyline(diamond, 4, selected ? IM_COL32(255, 220, 120, 255) : color,
                                ImDrawFlags_Closed, selected ? 2.5f : 1.5f);

                ImGui::SetCursorScreenPos(ImVec2(center.x - bodyRadius * 1.5f,
                                                 center.y - bodyRadius * 1.5f));
                ImGui::PushID(static_cast<int>(portal.id) + 9200000);
                ImGui::InvisibleButton("WireJunctionBody",
                    ImVec2(bodyRadius * 3.0f, bodyRadius * 3.0f));
                if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
                    selectedPortalId = portal.id;
                    selectedLinkId = 0;
                    selectedNodeId = 0;
                    selectedNodeIds.clear();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("%s | drag to move | Delete to remove",
                                      portal.name.c_str());
                }
                if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                    draggingPortalId = portal.id;
                    portal.position.x += ImGui::GetIO().MouseDelta.x / zoom;
                    portal.position.y += ImGui::GetIO().MouseDelta.y / zoom;
                    if (onGraphModified) onGraphModified();
                } else if (draggingPortalId == portal.id &&
                           ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
                    draggingPortalId = 0;
                }
                ImGui::PopID();

                // The small socket is the real source output mirrored at the
                // junction. Dragging it creates another fan-out connection.
                const ImVec2 socket(center.x + 14.0f * zoom, center.y);
                const float socketRadius = std::max(4.5f, 5.0f * zoom);
                dl->AddLine(ImVec2(center.x + bodyRadius, center.y), socket, color,
                            std::max(1.0f, zoom));
                dl->AddCircleFilled(socket, socketRadius, color);
                ImGui::SetCursorScreenPos(ImVec2(socket.x - socketRadius * 2.0f,
                                                 socket.y - socketRadius * 2.0f));
                ImGui::PushID(static_cast<int>(portal.id) + 9300000);
                ImGui::InvisibleButton("WireJunctionOutput",
                    ImVec2(socketRadius * 4.0f, socketRadius * 4.0f));
                if (ImGui::IsItemHovered()) {
                    dl->AddCircle(socket, socketRadius * 1.6f,
                                  IM_COL32(255, 255, 255, 135), 0, 2.0f);
                    ImGui::SetTooltip("Drag to branch this cable");
                }
                if (ImGui::IsItemActive() &&
                    ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.0f) && !isCreatingLink) {
                    isCreatingLink = true;
                    linkStartPinId = portal.linkedPinId;
                    linkStartType = pin->dataType;
                }
                ImGui::PopID();
            }
        }
        
        ImVec2 getBezierPoint(const ImVec2& p1, const ImVec2& cp1, const ImVec2& cp2, const ImVec2& p2, float t) {
            float u = 1.0f - t;
            float tt = t * t;
            float uu = u * u;
            float uuu = uu * u;
            float ttt = tt * t;
            
            ImVec2 p;
            p.x = uuu * p1.x + 3.0f * uu * t * cp1.x + 3.0f * u * tt * cp2.x + ttt * p2.x;
            p.y = uuu * p1.y + 3.0f * uu * t * cp1.y + 3.0f * u * tt * cp2.y + ttt * p2.y;
            return p;
        }

        void drawLink(ImDrawList* dl, Link& link, GraphBase& graph) {
            auto itStart = pinPositions_.find(link.startPinId);
            auto itEnd = pinPositions_.find(link.endPinId);
            if (itStart == pinPositions_.end() || itEnd == pinPositions_.end()) return;
            
            ImVec2 p1 = itStart->second;
            ImVec2 p2 = itEnd->second;
            
            // Get color from source pin
            Pin* sourcePin = graph.findPin(link.startPinId);
            Pin* targetPin = graph.findPin(link.endPinId);
            ImU32 col = sourcePin && sourcePin->cachedColor 
                ? sourcePin->cachedColor 
                : IM_COL32(150, 150, 150, 255);

            // Old projects retain formerly-permitted Image2D links. Make any
            // semantic/unit mismatch unmistakable instead of silently treating
            // it as a valid conversion.
            const bool semanticWarning = sourcePin && targetPin &&
                sourcePin->dataType == DataType::Image2D &&
                targetPin->dataType == DataType::Image2D &&
                !sourcePin->canConnectTo(*targetPin);
            if (semanticWarning) col = IM_COL32(255, 170, 45, 255);
            
            if (link.colorOverride != 0) {
                col = link.colorOverride;
            }
            
            // Bezier control points
            float dist = std::abs(p1.x - p2.x);
            float cpDist = std::max(dist * 0.5f, 50.0f * zoom);
            ImVec2 cp1(p1.x + cpDist, p1.y);
            ImVec2 cp2(p2.x - cpDist, p2.y);
            linkScreenGeometry_[link.id] = {p1, cp1, cp2, p2};
            
            bool isSelected = (selectedLinkId == link.id);
            bool isHovered = isLinkHovered(p1, cp1, cp2, p2);
            
            float thickness = isSelected ? config.linkSelectedThickness : config.linkThickness;
            thickness = std::max(1.0f, thickness * zoom);
            
            // 1. Draw glowing background shadow if selected or hovered
            if (isSelected) {
                dl->AddBezierCubic(p1, cp1, cp2, p2, (col & 0x00FFFFFF) | 0x30000000, thickness + 3.0f * zoom);
                dl->AddBezierCubic(p1, cp1, cp2, p2, (col & 0x00FFFFFF) | 0x60000000, thickness + 1.2f * zoom);
            } else if (isHovered) {
                dl->AddBezierCubic(p1, cp1, cp2, p2, (col & 0x00FFFFFF) | 0x20000000, thickness + 2.0f * zoom);
            }
            
            // 2. Draw core link line
            dl->AddBezierCubic(p1, cp1, cp2, p2, col, thickness);
            
            // 3. Draw moving data flow particles (only if there is active data flowing through the link)
            bool isFlowActive = sourcePin && sourcePin->hasValue();
            if (isFlowActive) {
                double time = ImGui::GetTime();
                float speed = 0.4f; // softer speed
                float progress = fmodf(static_cast<float>(time * speed), 1.0f);
                
                // Draw 2 dots moving along the line
                for (int k = 0; k < 2; k++) {
                    float t = progress + k * 0.5f;
                    if (t > 1.0f) t -= 1.0f;
                    
                    ImVec2 dotPos = getBezierPoint(p1, cp1, cp2, p2, t);
                    ImU32 haloCol = (col & 0x00FFFFFF) | 0x24000000; // soft glow halo
                    dl->AddCircleFilled(dotPos, 4.0f * zoom, haloCol);
                    dl->AddCircleFilled(dotPos, 1.5f * zoom, IM_COL32(255, 255, 255, 150)); // smaller and softer dot
                }
            }
            
            // Hit test and click selection
            if (isHovered) {
                dl->AddBezierCubic(p1, cp1, cp2, p2, IM_COL32(255, 255, 255, 50), 
                    thickness + 1.0f * zoom);
                
                if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                    selectedLinkId = link.id;
                    selectedPortalId = 0;
                    selectedNodeId = 0;
                    if (onLinkSelected) onLinkSelected(link.id);
                }
                if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                    createWireJunction(graph, link.startPinId, ImGui::GetMousePos());
                }
                if (semanticWarning) {
                    ImGui::SetTooltip("Legacy semantic/unit mismatch: insert an explicit conversion node");
                }
            }
        }

        void drawLayerBoundaryLink(ImDrawList* dl, Link& link, GraphBase& graph,
                                   bool incoming, int row, bool drawPortCard,
                                   const LayerInterfacePort* interfacePort) {
            Pin* sourcePin = graph.findPin(link.startPinId);
            Pin* targetPin = graph.findPin(link.endPinId);
            if (!sourcePin || !targetPin) return;
            const uint32_t internalPinId = incoming ? link.endPinId : link.startPinId;
            const auto internalIt = pinPositions_.find(internalPinId);
            if (internalIt == pinPositions_.end()) return;

            constexpr float cardWidth = 132.0f;
            constexpr float rowHeight = 24.0f;
            constexpr float topOffset = 30.0f;
            const float y = canvasPos_.y + topOffset + 22.0f + row * rowHeight;
            const ImVec2 cardMin = incoming
                ? ImVec2(canvasPos_.x + 6.0f, y)
                : ImVec2(canvasPos_.x + canvasSize_.x - cardWidth - 6.0f, y);
            const ImVec2 cardMax(cardMin.x + cardWidth, cardMin.y + rowHeight - 3.0f);
            const ImVec2 boundaryPin = incoming
                ? ImVec2(cardMax.x, (cardMin.y + cardMax.y) * 0.5f)
                : ImVec2(cardMin.x, (cardMin.y + cardMax.y) * 0.5f);
            const ImVec2 internalPin = internalIt->second;
            const ImVec2 p1 = incoming ? boundaryPin : internalPin;
            const ImVec2 p2 = incoming ? internalPin : boundaryPin;

            ImU32 color = sourcePin->cachedColor ? sourcePin->cachedColor
                                                  : IM_COL32(150, 150, 160, 255);
            if (link.colorOverride != 0) color = link.colorOverride;
            const float cp = std::max(55.0f * zoom, std::abs(p2.x - p1.x) * 0.42f);
            const ImVec2 cp1(p1.x + cp, p1.y);
            const ImVec2 cp2(p2.x - cp, p2.y);
            linkScreenGeometry_[link.id] = {p1, cp1, cp2, p2};
            dl->AddBezierCubic(p1, cp1, cp2, p2,
                               color, std::max(1.0f, config.linkThickness * zoom));

            const bool hovered = isLinkHovered(p1, cp1, cp2, p2);
            if (hovered) {
                dl->AddBezierCubic(p1, cp1, cp2, p2, IM_COL32(255, 255, 255, 55),
                                   std::max(2.0f, (config.linkThickness + 1.0f) * zoom));
                if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                    selectedLinkId = link.id;
                    selectedPortalId = 0;
                    selectedNodeId = 0;
                    if (onLinkSelected) onLinkSelected(link.id);
                }
                if (focusedGroupId == 0 && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                    createWireJunction(graph, link.startPinId, ImGui::GetMousePos());
                }
            }

            if (!drawPortCard) return;
            dl->AddRectFilled(cardMin, cardMax, IM_COL32(24, 28, 36, 238), 4.0f);
            dl->AddRect(cardMin, cardMax, (color & 0x00FFFFFF) | 0x90000000, 4.0f);
            dl->AddCircleFilled(boundaryPin, 4.5f, color);

            // The boundary socket is a real interaction proxy for the hidden
            // external endpoint. Registering that actual pin id gives layer
            // ports the same drag/type/cycle/replacement behavior as node pins.
            if (interfacePort) {
                drawLayerPortInteraction(dl, graph, *interfacePort, boundaryPin, color, true);
            }

            NodeBase* externalNode = graph.getPinOwner(incoming ? link.startPinId : link.endPinId);
            std::string label = interfacePort && !interfacePort->name.empty()
                ? interfacePort->name : (incoming ? sourcePin->name : targetPin->name);
            if (!interfacePort && externalNode) {
                const std::string& owner = externalNode->metadata.displayName.empty()
                    ? externalNode->name : externalNode->metadata.displayName;
                label += incoming ? "  < " : "  > ";
                label += owner;
            }
            if (interfacePort) label += "  [P" + std::to_string(interfacePort->id) + "]";
            label = fitTextToWidth(label, cardWidth - 16.0f);
            const ImVec2 textSize = ImGui::CalcTextSize(label.c_str());
            const float textX = incoming ? cardMin.x + 7.0f : cardMax.x - 7.0f - textSize.x;
            dl->AddText(ImVec2(textX, cardMin.y + (cardMax.y - cardMin.y - textSize.y) * 0.5f),
                        IM_COL32(220, 225, 235, 255), label.c_str());

            if (row == 0) {
                const char* heading = incoming ? "LAYER INPUTS" : "LAYER OUTPUTS";
                const ImVec2 headingSize = ImGui::CalcTextSize(heading);
                const float headingX = incoming ? cardMin.x : cardMax.x - headingSize.x;
                dl->AddText(ImVec2(headingX, canvasPos_.y + 8.0f),
                            IM_COL32(125, 175, 225, 220), heading);
            }
        }

        void drawDisconnectedLayerPort(ImDrawList* dl, GraphBase& graph,
                                       const LayerInterfacePort& port,
                                       bool incoming, int row) {
            constexpr float cardWidth = 132.0f;
            constexpr float rowHeight = 24.0f;
            constexpr float topOffset = 30.0f;
            const float y = canvasPos_.y + topOffset + 22.0f + row * rowHeight;
            const ImVec2 cardMin = incoming
                ? ImVec2(canvasPos_.x + 6.0f, y)
                : ImVec2(canvasPos_.x + canvasSize_.x - cardWidth - 6.0f, y);
            const ImVec2 cardMax(cardMin.x + cardWidth, cardMin.y + rowHeight - 3.0f);
            const ImVec2 boundaryPin = incoming
                ? ImVec2(cardMax.x, (cardMin.y + cardMax.y) * 0.5f)
                : ImVec2(cardMin.x, (cardMin.y + cardMax.y) * 0.5f);
            dl->AddRectFilled(cardMin, cardMax, IM_COL32(42, 25, 29, 238), 4.0f);
            dl->AddRect(cardMin, cardMax, IM_COL32(220, 75, 75, 210), 4.0f, 0, 1.5f);
            Pin* proxyPin = graph.findPin(port.externalPinId);
            const ImU32 pinColor = proxyPin && proxyPin->cachedColor
                ? proxyPin->cachedColor : IM_COL32(220, 95, 95, 255);
            dl->AddCircleFilled(boundaryPin, 4.5f, pinColor);
            drawLayerPortInteraction(dl, graph, port, boundaryPin, pinColor, false);
            std::string label = port.name + "  [P" + std::to_string(port.id) + "] !";
            label = fitTextToWidth(label, cardWidth - 16.0f);
            const ImVec2 textSize = ImGui::CalcTextSize(label.c_str());
            const float textX = incoming ? cardMin.x + 7.0f : cardMax.x - 7.0f - textSize.x;
            dl->AddText(ImVec2(textX, cardMin.y + (cardMax.y - cardMin.y - textSize.y) * 0.5f),
                        IM_COL32(255, 175, 175, 255), label.c_str());
        }

        void drawLayerPortInteraction(ImDrawList* dl, GraphBase& graph,
                                      const LayerInterfacePort& port,
                                      const ImVec2& center, ImU32 color,
                                      bool connected) {
            Pin* proxyPin = graph.findPin(port.externalPinId);
            if (!proxyPin) return;

            // The external node is hidden in focused-layer view, so its real pin
            // id can safely occupy the boundary position in the common pin map.
            // findClosestPin()/drawCreatingLink() then require no special cases.
            pinPositions_[proxyPin->id] = center;

            const float radius = std::max(5.0f, 5.0f * zoom);
            dl->AddCircle(center, radius, color, 0, 1.25f);
            ImGui::SetCursorScreenPos(ImVec2(center.x - radius * 2.0f,
                                             center.y - radius * 2.0f));
            ImGui::PushID(static_cast<int>(focusedGroupId));
            ImGui::PushID(static_cast<int>(port.id));
            ImGui::InvisibleButton("LayerPort", ImVec2(radius * 4.0f, radius * 4.0f));
            if (ImGui::IsItemHovered()) {
                dl->AddCircle(center, radius * 1.55f, IM_COL32(255, 255, 255, 130), 0, 2.0f);
                ImGui::SetTooltip("%s layer %s: drag to %s",
                    connected ? "Connected" : "Disconnected",
                    port.direction == LayerPortDirection::Input ? "input" : "output",
                    port.direction == LayerPortDirection::Input ? "an internal input" : "an internal output");
            }
            if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.0f) &&
                !isCreatingLink) {
                isCreatingLink = true;
                linkStartPinId = proxyPin->id;
                linkStartType = proxyPin->dataType;
            }
            ImGui::PopID();
            ImGui::PopID();
        }

        void drawCreatingLink(ImDrawList* dl) {
            auto it = pinPositions_.find(linkStartPinId);
            if (it == pinPositions_.end()) return;
            
            ImVec2 p1 = it->second;
            ImVec2 p2 = ImGui::GetMousePos();
            
            float dist = std::abs(p1.x - p2.x);
            float cpDist = std::max(dist * 0.5f, 50.0f * zoom);
            ImVec2 cp1(p1.x + cpDist, p1.y);
            ImVec2 cp2(p2.x - cpDist, p2.y);
            
            dl->AddBezierCubic(p1, cp1, cp2, p2, IM_COL32(255, 200, 100, 200), 
                config.linkThickness * zoom);
        }

        // ========================================================================
        // GROUP (Frame)
        // ========================================================================

        void drawCollapsedGroupPorts(ImDrawList* dl, GraphBase& graph, NodeGroup& group,
                                     const ImVec2& pos, const ImVec2& size) {
            int inputRow = 0;
            int outputRow = 0;
            const float radius = std::max(4.0f, 4.5f * zoom);
            const float firstRowY = pos.y + 43.0f * zoom;
            const float rowStep = 22.0f * zoom;

            for (const LayerInterfacePort& port : group.interfacePorts) {
                if (!port.connected) continue;
                const bool incoming = port.direction == LayerPortDirection::Input;
                const int row = incoming ? inputRow++ : outputRow++;
                const ImVec2 center(
                    incoming ? pos.x : pos.x + size.x,
                    firstRowY + static_cast<float>(row) * rowStep);

                // The interface socket represents the actual hidden internal pin.
                // Keeping that id means link editing, cycle checks and evaluation
                // remain identical whether the group is expanded or collapsed.
                Pin* internalPin = graph.findPin(port.internalPinId);
                if (!internalPin) continue;
                pinPositions_[internalPin->id] = center;

                ImU32 color = internalPin->cachedColor
                    ? internalPin->cachedColor : IM_COL32(150, 150, 165, 255);
                dl->AddCircleFilled(center, radius, color);
                dl->AddCircle(center, radius, IM_COL32(245, 248, 255, 210), 0,
                              std::max(1.0f, zoom));

                ImGui::SetCursorScreenPos(ImVec2(center.x - radius * 2.0f,
                                                 center.y - radius * 2.0f));
                ImGui::PushID(static_cast<int>(group.id) + 8100000);
                ImGui::PushID(static_cast<int>(port.id));
                ImGui::InvisibleButton("CollapsedGroupPort", ImVec2(radius * 4.0f, radius * 4.0f));
                if (ImGui::IsItemHovered()) {
                    dl->AddCircle(center, radius * 1.65f, IM_COL32(255, 255, 255, 135), 0, 2.0f);
                    ImGui::SetTooltip("%s | %s group %s",
                        port.name.c_str(), group.name.c_str(), incoming ? "input" : "output");
                }
                if (ImGui::IsItemActive() &&
                    ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.0f) && !isCreatingLink) {
                    isCreatingLink = true;
                    linkStartPinId = internalPin->id;
                    linkStartType = internalPin->dataType;
                }
                ImGui::PopID();
                ImGui::PopID();

                if (zoom > 0.45f) {
                    const float labelWidth = size.x * 0.43f;
                    const std::string label = fitTextToWidth(port.name, labelWidth);
                    const ImVec2 textSize = ImGui::CalcTextSize(label.c_str());
                    const float textX = incoming
                        ? center.x + radius + 5.0f * zoom
                        : center.x - radius - 5.0f * zoom - textSize.x;
                    dl->AddText(ImVec2(textX, center.y - textSize.y * 0.5f),
                                IM_COL32(222, 228, 238, 235), label.c_str());
                }
            }
        }
        
        void drawGroup(ImDrawList* dl, NodeGroup& group, GraphBase& graph) {
            ImVec2 pos = nodeToScreen(group.position.x, group.position.y);
            int collapsedInputCount = 0;
            int collapsedOutputCount = 0;
            if (group.collapsed) {
                for (const LayerInterfacePort& port : group.interfacePorts) {
                    if (!port.connected) continue;
                    if (port.direction == LayerPortDirection::Input) ++collapsedInputCount;
                    else ++collapsedOutputCount;
                }
            }
            const int collapsedRows = std::max(collapsedInputCount, collapsedOutputCount);
            const float collapsedWidth = std::max(190.0f,
                42.0f + ImGui::CalcTextSize(group.name.c_str()).x);
            const float collapsedHeight = std::max(38.0f,
                34.0f + static_cast<float>(collapsedRows) * 22.0f);
            ImVec2 size = group.collapsed
                ? ImVec2(collapsedWidth * zoom, collapsedHeight * zoom)
                : ImVec2(group.size.x * zoom, group.size.y * zoom);
            
            float radius = 8.0f * zoom;
            
            ImVec4 colF = ImGui::ColorConvertU32ToFloat4(group.color);
            // Expanded frames preserve grid readability; collapsed groups become compact,
            // clearly visible subsystem-layer tiles.
            ImU32 bgCol = ImGui::ColorConvertFloat4ToU32(
                ImVec4(colF.x, colF.y, colF.z, group.collapsed ? 0.20f : 0.035f));
            
            // Background
            dl->AddRectFilled(pos, ImVec2(pos.x + size.x, pos.y + size.y), bgCol, radius);
            
            // Border
            bool isSelected = (selectedGroupId == group.id);
            ImU32 borderCol = isSelected 
                ? config.nodeSelectedColor 
                : ImGui::ColorConvertFloat4ToU32(ImVec4(colF.x, colF.y, colF.z, 0.30f));
                
            dl->AddRect(pos, ImVec2(pos.x + size.x, pos.y + size.y), borderCol, radius, 0, 
                isSelected ? 1.8f * zoom : 1.0f * zoom);

            if (group.collapsed) {
                drawCollapsedGroupPorts(dl, graph, group, pos, size);
            }
            
            // Resize Handle (Gaea-style)
            if (!group.collapsed) {
                ImVec2 resizeMin(pos.x + size.x - 12.0f * zoom, pos.y + size.y - 12.0f * zoom);
                ImVec2 resizeMax(pos.x + size.x, pos.y + size.y);
                ImU32 handleCol = (resizingGroupId == group.id) ? config.nodeSelectedColor : IM_COL32(160, 160, 170, 120);
                dl->AddTriangleFilled(
                    ImVec2(resizeMax.x, resizeMin.y),
                    ImVec2(resizeMin.x, resizeMax.y),
                    resizeMax,
                    handleCol
                );

                ImGui::SetCursorScreenPos(resizeMin);
                ImGui::PushID((int)group.id + 7000000);
                ImGui::InvisibleButton("GroupResize", ImVec2(resizeMax.x - resizeMin.x, resizeMax.y - resizeMin.y));
                if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                    resizingGroupId = group.id;
                }
                ImGui::PopID();
            }
            
            // Title Badge (translucent tab at the top-left)
            if (zoom > 0.3f && !group.name.empty()) {
                ImVec2 txtSz = ImGui::CalcTextSize(group.name.c_str());
                float badgePaddingX = 8.0f * zoom;
                float badgePaddingY = 4.0f * zoom;
                
                ImVec2 badgeMin = pos;
                ImVec2 badgeMax = ImVec2(pos.x + txtSz.x + badgePaddingX * 2, pos.y + txtSz.y + badgePaddingY * 2);
                
                // Draw a colored top-left tab/badge
                ImU32 badgeCol = ImGui::ColorConvertFloat4ToU32(ImVec4(colF.x, colF.y, colF.z, 0.18f));
                dl->AddRectFilled(badgeMin, badgeMax, badgeCol, radius, ImDrawFlags_RoundCornersTopLeft | ImDrawFlags_RoundCornersBottomRight);
                dl->AddRect(badgeMin, badgeMax, borderCol, radius, ImDrawFlags_RoundCornersTopLeft | ImDrawFlags_RoundCornersBottomRight, 1.0f);
                
                dl->AddText(ImVec2(pos.x + badgePaddingX, pos.y + badgePaddingY), 
                    IM_COL32(230, 235, 245, 230), group.name.c_str());

                // Group drag interaction area
                ImGui::SetCursorScreenPos(badgeMin);
                ImGui::PushID((int)group.id + 5000000);
                ImGui::InvisibleButton("GroupDrag", ImVec2(badgeMax.x - badgeMin.x, badgeMax.y - badgeMin.y));
                if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                    selectedGroupId = group.id;
                    draggingGroupId = group.id;
                }
                if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
                    selectedGroupId = group.id;
                    selectedNodeIds.clear();
                    selectedNodeId = 0;
                }
                if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                    group.collapsed = !group.collapsed;
                    if (onGraphModified) onGraphModified();
                }
                if (ImGui::IsItemClicked(ImGuiMouseButton_Right)) {
                    // OpenPopup() deferred to just after PopID() below — see
                    // groupContextMenuRequested_'s comment for why (id-stack scope mismatch).
                    selectedGroupId = group.id;
                    groupContextMenuRequested_ = true;
                }
                ImGui::PopID();
                if (groupContextMenuRequested_) {
                    groupContextMenuRequested_ = false;
                    ImGui::OpenPopup("LocalGroupContextPopup");
                    contextMenuClaimedByNodeOrGroup_ = true;  // suppress the background popup below
                }
            }
        }

        // ========================================================================
        // MINIMAP
        // ========================================================================
        
        void drawMinimap(ImDrawList* dl, GraphBase& graph) {
            float mmSize = config.minimapSize;
            ImVec2 mmPos(canvasPos_.x + canvasSize_.x - mmSize - 10, 
                         canvasPos_.y + canvasSize_.y - mmSize - 10);
            
            // Background
            dl->AddRectFilled(mmPos, ImVec2(mmPos.x + mmSize, mmPos.y + mmSize), 
                config.minimapBgColor, 4.0f);
            dl->AddRect(mmPos, ImVec2(mmPos.x + mmSize, mmPos.y + mmSize), 
                IM_COL32(100, 100, 120, 150), 4.0f);
            
            if (graph.nodes.empty()) return;
            
            // Calculate bounds
            float minX = FLT_MAX, minY = FLT_MAX, maxX = -FLT_MAX, maxY = -FLT_MAX;
            for (auto& node : graph.nodes) {
                if (!isNodeVisible(graph, *node)) continue;
                minX = std::min(minX, node->x);
                minY = std::min(minY, node->y);
                maxX = std::max(maxX, node->x + 200);
                maxY = std::max(maxY, node->y + 100);
            }
            if (minX == FLT_MAX) return;
            
            float rangeX = maxX - minX + 100;
            float rangeY = maxY - minY + 100;
            float scale = std::min(mmSize / rangeX, mmSize / rangeY) * 0.8f;
            
            // Draw nodes
            for (auto& node : graph.nodes) {
                if (!isNodeVisible(graph, *node)) continue;
                float nx = mmPos.x + 10 + (node->x - minX) * scale;
                float ny = mmPos.y + 10 + (node->y - minY) * scale;
                ImU32 col = node->metadata.headerColor ? node->metadata.headerColor 
                    : ImGui::ColorConvertFloat4ToU32(node->headerColor);
                dl->AddRectFilled(ImVec2(nx, ny), ImVec2(nx + 10, ny + 6), col, 2.0f);
            }
            
            // Draw viewport
            float vpX = mmPos.x + 10 + (-scrollX / zoom - minX) * scale;
            float vpY = mmPos.y + 10 + (-scrollY / zoom - minY) * scale;
            float vpW = (canvasSize_.x / zoom) * scale;
            float vpH = (canvasSize_.y / zoom) * scale;
            dl->AddRect(ImVec2(vpX, vpY), ImVec2(vpX + vpW, vpY + vpH), 
                IM_COL32(255, 255, 255, 150), 0, 0, 1.5f);
        }

        // ========================================================================
        // UTILITIES
        // ========================================================================
        
        ImVec2 nodeToScreen(float x, float y) {
            return ImVec2(canvasPos_.x + x * zoom + scrollX, 
                          canvasPos_.y + y * zoom + scrollY);
        }
        
       
        void checkNodeDroppedOnLink(GraphBase& graph, NodeBase& node) {
            // Auto-splice is a placement convenience for a fresh/unconnected
            // node only. Repositioning an already wired node must never mutate
            // its graph merely because its body crosses another cable.
            const bool nodeAlreadyConnected = std::any_of(
                graph.links.begin(), graph.links.end(),
                [&graph, &node](const Link& link) {
                    const NodeBase* startOwner = graph.getPinOwner(link.startPinId);
                    const NodeBase* endOwner = graph.getPinOwner(link.endPinId);
                    return (startOwner && startOwner->id == node.id) ||
                           (endOwner && endOwner->id == node.id);
                });
            if (nodeAlreadyConnected) return;

            float customW = node.getCustomWidth();
            float padding = scaleNodeChromeMetric(zoom, 10.0f, 7.0f, 14.0f);
            float titleW = ImGui::CalcTextSize(node.metadata.displayName.c_str()).x + padding * 2;
            if (titleW < 10) titleW = ImGui::CalcTextSize(node.name.c_str()).x + padding * 2;
            NodeChromeLayout chrome = buildNodeChromeLayout(node, zoom, customW > 0.0f ? customW * zoom : 160.0f * zoom,
                visiblePinCount(node.inputs), visiblePinCount(node.outputs), titleW);

            ImVec2 nodeScreenPos = nodeToScreen(node.x, node.y);
            ImVec2 center = ImVec2(nodeScreenPos.x + chrome.width * 0.5f, nodeScreenPos.y + chrome.height * 0.5f);
            
            float threshold = 18.0f * zoom; // distance threshold
            
            for (auto& link : graph.links) {
                NodeBase* startOwner = graph.getPinOwner(link.startPinId);
                NodeBase* endOwner = graph.getPinOwner(link.endPinId);
                if (!startOwner || !endOwner) continue;
                if (startOwner->id == node.id || endOwner->id == node.id) continue;
                
                const auto geometry = linkScreenGeometry_.find(link.id);
                if (geometry == linkScreenGeometry_.end()) continue;

                const ImVec2 p1 = geometry->second.p1;
                const ImVec2 p2 = geometry->second.p2;
                const ImVec2 cp1 = geometry->second.cp1;
                const ImVec2 cp2 = geometry->second.cp2;
                
                ImVec2 prev = p1;
                bool onLink = false;
                for (int i = 1; i <= 20; i++) {
                    float t = (float)i / 20.0f;
                    float u = 1.0f - t;
                    ImVec2 p;
                    p.x = u*u*u*p1.x + 3*u*u*t*cp1.x + 3*u*t*t*cp2.x + t*t*t*p2.x;
                    p.y = u*u*u*p1.y + 3*u*u*t*cp1.y + 3*u*t*t*cp2.y + t*t*t*p2.y;
                    
                    float d = pointSegmentDistance(center, prev, p);
                    if (d < threshold) {
                        onLink = true;
                        break;
                    }
                    prev = p;
                }
                
                if (onLink) {
                    Pin* outPin = graph.findPin(link.startPinId);
                    Pin* inPin = graph.findPin(link.endPinId);
                    if (!outPin || !inPin) continue;
                    
                    auto compatibilityScore = [](const Pin& candidate, const Pin& reference) {
                        if (!candidate.canConnectTo(reference) && !reference.canConnectTo(candidate)) return -1;
                        int score = candidate.dataType == reference.dataType ? 100 : 0;
                        if (candidate.dataType == DataType::Image2D && reference.dataType == DataType::Image2D) {
                            if (candidate.imageSemantic == reference.imageSemantic) score += 50;
                            else if (candidate.imageSemantic == ImageSemantic::Generic ||
                                     reference.imageSemantic == ImageSemantic::Generic) score += 10;
                            if (candidate.imageChannels == reference.imageChannels) score += 20;
                        }
                        return score;
                    };

                    // Find the semantically closest compatible input/output.
                    Pin* compatibleIn = nullptr;
                    int bestInputScore = -1;
                    for (auto& nIn : node.inputs) {
                        const int score = compatibilityScore(nIn, *outPin);
                        if (score > bestInputScore) {
                            bestInputScore = score;
                            compatibleIn = &nIn;
                        }
                    }

                    Pin* compatibleOut = nullptr;
                    int bestOutputScore = -1;
                    for (auto& nOut : node.outputs) {
                        const int score = compatibilityScore(nOut, *inPin);
                        if (score > bestOutputScore) {
                            bestOutputScore = score;
                            compatibleOut = &nOut;
                        }
                    }

                    if (compatibleIn && compatibleOut) {
                        const uint32_t oldLinkId = link.id;
                        const uint32_t oldStartPinId = outPin->id;
                        const uint32_t oldEndPinId = inPin->id;
                        graph.removeLink(oldLinkId);
                        const uint32_t incomingId = graph.addLink(oldStartPinId, compatibleIn->id);
                        const uint32_t outgoingId = graph.addLink(compatibleOut->id, oldEndPinId);
                        if (incomingId == 0 || outgoingId == 0) {
                            if (incomingId != 0) graph.removeLink(incomingId);
                            if (outgoingId != 0) graph.removeLink(outgoingId);
                            graph.addLink(oldStartPinId, oldEndPinId);
                        } else if (onGraphModified) {
                            onGraphModified();
                        }
                        break; // Inserted successfully
                    }
                }
            }
        }
        
        uint32_t findClosestPin(GraphBase& graph, const ImVec2& mouse, float maxDist) {
            // Type-aware two-tier snap. The old plain nearest-pin snap with a
            // 20px radius exceeded the ~18px pin spacing, so a release between
            // two sockets silently landed on the neighbor — e.g. a texture
            // meant for Base Color binding to Metallic — which reads as "wrong
            // texture / wrong UV" in the render. Now:
            //   1) pins the dragged link cannot connect to are skipped entirely,
            //   2) an EXACT type match beats a merely convertible one
            //      (Float<->Vector3) regardless of distance.
            // Follow-up: "exact type wins REGARDLESS of distance" over-corrected. With a
            // column of mixed-type sockets, aiming straight at a convertible pin still
            // snapped to an exact-type pin one slot away — the link visibly jumped to the
            // socket ABOVE the one under the cursor. Distance decides now; an exact type
            // match only wins as a TIEBREAK, when it is within kExactBias of the nearest
            // convertible one (well under a pin spacing, so it can never reach a neighbour).
            //
            // Also skipped outright: the dragged pin's own node, and any pin that would
            // close a CYCLE. A cycle is not a cosmetic mistake here — every recursive walk
            // over the graph runs without a visited set, so one loop is a stack overflow
            // (0xC00000FD) and takes the process down. Not offering the pin is the cheapest
            // place to stop it; GraphBase::addLink refuses it again as a backstop.
            constexpr float kExactBias = 6.0f;

            Pin* startPin = linkStartPinId ? graph.findPin(linkStartPinId) : nullptr;
            NodeBase* startNode = linkStartPinId ? graph.getPinOwner(linkStartPinId) : nullptr;

            uint32_t closestExact = 0;
            float bestExact = maxDist;
            uint32_t closestConv = 0;
            float bestConv = maxDist;

            for (auto& [pinId, pos] : pinPositions_) {
                if (pinId == linkStartPinId) continue;
                const float d = std::hypot(pos.x - mouse.x, pos.y - mouse.y);
                if (d >= maxDist) continue;

                if (!startPin) {
                    if (d < bestConv) { bestConv = d; closestConv = pinId; }
                    continue;
                }
                Pin* cand = graph.findPin(pinId);
                if (!cand) continue;
                if (startNode && graph.getPinOwner(pinId) == startNode) continue;   // same node
                if (!startPin->canConnectTo(*cand) && !cand->canConnectTo(*startPin)) continue;

                // Resolve producer -> consumer for the cycle test, whichever end was dragged.
                const bool startIsOutput = (startPin->kind == PinKind::Output);
                const uint32_t producerPin = startIsOutput ? linkStartPinId : pinId;
                const uint32_t consumerPin = startIsOutput ? pinId : linkStartPinId;
                if (graph.wouldCreateCycle(producerPin, consumerPin)) continue;

                if (cand->dataType == startPin->dataType) {
                    if (d < bestExact) { bestExact = d; closestExact = pinId; }
                } else {
                    if (d < bestConv) { bestConv = d; closestConv = pinId; }
                }
            }

            if (closestExact && closestConv) {
                return (bestExact <= bestConv + kExactBias) ? closestExact : closestConv;
            }
            return closestExact ? closestExact : closestConv;
        }
        
        bool isLinkHovered(const ImVec2& p1, const ImVec2& cp1, 
                          const ImVec2& cp2, const ImVec2& p2) {
            ImVec2 mouse = ImGui::GetMousePos();

            // Links are custom draw-list primitives rather than ImGui items, so
            // their hit test must explicitly obey the canvas interaction domain.
            // Without this guard, an off-canvas Bezier/control point could pass
            // beneath the Properties panel; clicking a slider then selected that
            // link and cleared selectedNodeId.
            const bool mouseInsideCanvas =
                mouse.x >= canvasPos_.x && mouse.y >= canvasPos_.y &&
                mouse.x < canvasPos_.x + canvasSize_.x &&
                mouse.y < canvasPos_.y + canvasSize_.y;
            if (!mouseInsideCanvas ||
                !ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem |
                                        ImGuiHoveredFlags_ChildWindows)) {
                return false;
            }

            float threshold = 10.0f;
            
            ImVec2 prev = p1;
            for (int i = 1; i <= 20; i++) {
                float t = (float)i / 20.0f;
                float u = 1.0f - t;
                ImVec2 p;
                p.x = u*u*u*p1.x + 3*u*u*t*cp1.x + 3*u*t*t*cp2.x + t*t*t*p2.x;
                p.y = u*u*u*p1.y + 3*u*u*t*cp1.y + 3*u*t*t*cp2.y + t*t*t*p2.y;
                
                float d = pointSegmentDistance(mouse, prev, p);
                if (d < threshold) return true;
                prev = p;
            }
            return false;
        }
        
        float pointSegmentDistance(const ImVec2& p, const ImVec2& a, const ImVec2& b) {
            ImVec2 v(b.x - a.x, b.y - a.y);
            ImVec2 w(p.x - a.x, p.y - a.y);
            float len2 = v.x * v.x + v.y * v.y;
            if (len2 < 0.001f) return std::hypot(p.x - a.x, p.y - a.y);
            float t = std::clamp((w.x * v.x + w.y * v.y) / len2, 0.0f, 1.0f);
            ImVec2 proj(a.x + t * v.x, a.y + t * v.y);
            return std::hypot(p.x - proj.x, p.y - proj.y);
        }
    };

} // namespace NodeSystem

