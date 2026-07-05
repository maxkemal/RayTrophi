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
        std::function<void()> onGraphModified;

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
            selectedGroupId = 0;
            draggingNodeId = 0;
            draggingGroupId = 0;
            resizingNodeId = 0;
            resizingGroupId = 0;
            isCreatingLink = false;
            linkStartPinId = 0;
            isBoxSelecting = false;
            releasedLinkPinId = 0;
            mousePosOnRightClick = ImVec2(0.0f, 0.0f);
        }

    private:
        std::unordered_map<uint32_t, ImVec2> pinPositions_;

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
            
            // Background
            dl->AddRectFilled(canvasPos_, 
                ImVec2(canvasPos_.x + canvasSize_.x, canvasPos_.y + canvasSize_.y), 
                config.bgColor);
            
            // Grid
            drawGrid(dl);
            
            // Input handling
            handleInput(graph);
            
            // Reset pin cache
            pinPositions_.clear();
            
            // Draw groups (behind nodes)
            for (auto& group : graph.groups) {
                drawGroup(dl, group);
            }
            
            // Draw nodes
            for (auto& node : graph.nodes) {
                drawNode(dl, *node, graph);
            }
            
            // Draw links
            for (auto& link : graph.links) {
                drawLink(dl, link, graph);
            }
            
            // Draw creating link
            if (isCreatingLink && linkStartPinId != 0) {
                drawCreatingLink(dl);
            }
            
            // Minimap
            if (config.showMinimap) {
                drawMinimap(dl, graph);
            }
            
            // Resolve the deferred background context menu request now that node/group
            // hit-testing (which can claim the same click) has already run this frame.
            if (backgroundContextMenuRequested_ && !contextMenuClaimedByNodeOrGroup_) {
                ImGui::OpenPopup("LocalGraphContextPopup");
            }
            backgroundContextMenuRequested_ = false;
            contextMenuClaimedByNodeOrGroup_ = false;

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


    private:
        void drawPopups(GraphBase& graph) {
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
            
            // Pan (middle mouse) & Box Selection start
            if (ImGui::IsItemActive()) {
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
            
            // Zoom (scroll wheel)
            if (ImGui::IsItemHovered()) {
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
                    if (selectedLinkId != 0) {
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
                    uint32_t targetPin = findClosestPin(mouse, 20.0f * zoom);
                    
                    if (targetPin != 0 && targetPin != linkStartPinId) {
                        graph.addLink(linkStartPinId, targetPin);
                        if (onGraphModified) onGraphModified();
                    } else if (targetPin == 0) {
                        releasedLinkPinId = linkStartPinId;
                        mousePosOnRightClick = ImVec2((mouse.x - canvasPos_.x - scrollX) / zoom,
                                                      (mouse.y - canvasPos_.y - scrollY) / zoom);
                        backgroundContextMenuRequested_ = true;
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
                            float nodeCenterX = node->x + 80.0f;
                            float nodeCenterY = node->y + 40.0f;
                            uint32_t newGroupId = 0;
                            
                            for (auto& g : graph.groups) {
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
        
        void drawNode(ImDrawList* dl, NodeBase& node, GraphBase& graph) {
            ImVec2 pos = nodeToScreen(node.x, node.y);
            float padding = scaleNodeChromeMetric(zoom, 10.0f, 7.0f, 14.0f);
            
            // Estimate node size for culling (use max possible size)
            float estWidth = 500.0f * zoom;
            float estHeight = 400.0f * zoom;
            
            // Frustum culling - skip nodes completely outside canvas
            if (pos.x + estWidth < canvasPos_.x || pos.x > canvasPos_.x + canvasSize_.x ||
                pos.y + estHeight < canvasPos_.y || pos.y > canvasPos_.y + canvasSize_.y) {
                // Still need to cache pin positions for link drawing
                const float headerH = scaleNodeChromeMetric(zoom, 26.0f, 22.0f, 38.0f);
                const float pinSpacing = scaleNodeChromeMetric(zoom, 22.0f, 16.0f, 30.0f);
                const float minWidth = scaleNodeChromeMetric(zoom, 160.0f, 110.0f, 240.0f);
                float pinStartY = pos.y + headerH + padding;
                
                for (int i = 0; i < (int)node.inputs.size(); i++) {
                    pinPositions_[node.inputs[i].id] = ImVec2(pos.x, pinStartY + i * pinSpacing + pinSpacing * 0.5f);
                }
                for (int i = 0; i < (int)node.outputs.size(); i++) {
                    pinPositions_[node.outputs[i].id] = ImVec2(pos.x + minWidth, pinStartY + i * pinSpacing + pinSpacing * 0.5f);
                }
                return; // Skip full rendering
            }
            
            float customW = node.getCustomWidth();
            
            // Calculate title width (capped)
            float titleW = ImGui::CalcTextSize(node.metadata.displayName.c_str()).x + padding * 2;
            if (titleW < 10) titleW = ImGui::CalcTextSize(node.name.c_str()).x + padding * 2;
            NodeChromeLayout chrome = buildNodeChromeLayout(node, zoom, customW > 0.0f ? customW * zoom : 160.0f * zoom,
                node.inputs.size(), node.outputs.size(), titleW);
            chrome.cornerRadius = scaleNodeChromeMetric(zoom, config.nodeRounding, 3.0f, 12.0f);

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

            float inputPinStartY = getNodePinStartY(chrome, pos.y, node.inputs.size());
            float outputPinStartY = getNodePinStartY(chrome, pos.y, node.outputs.size());

            for (int i = 0; i < (int)node.inputs.size(); i++) {
                Pin& pin = node.inputs[i];
                ImVec2 pinPos(pos.x, inputPinStartY + i * (isCollapsed ? chrome.collapsedPinSpacing : chrome.pinSpacing));
                pinPositions_[pin.id] = pinPos;
                drawPin(dl, pinPos, pin, true, (!isCollapsed && showPinLabels) ? chrome.labelWidth : 0.0f);
            }

            for (int i = 0; i < (int)node.outputs.size(); i++) {
                Pin& pin = node.outputs[i];
                ImVec2 pinPos(pos.x + finalWidth, outputPinStartY + i * (isCollapsed ? chrome.collapsedPinSpacing : chrome.pinSpacing));
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
            if (graph.isEvaluatingAsync() && graph.currentAsyncNodeId() == node.id) {
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
                if (!pin.tooltip.empty()) {
                    ImGui::SetTooltip("%s", pin.tooltip.c_str());
                }
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
            ImU32 col = sourcePin && sourcePin->cachedColor 
                ? sourcePin->cachedColor 
                : IM_COL32(150, 150, 150, 255);
            
            if (link.colorOverride != 0) {
                col = link.colorOverride;
            }
            
            // Bezier control points
            float dist = std::abs(p1.x - p2.x);
            float cpDist = std::max(dist * 0.5f, 50.0f * zoom);
            ImVec2 cp1(p1.x + cpDist, p1.y);
            ImVec2 cp2(p2.x - cpDist, p2.y);
            
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
                    selectedNodeId = 0;
                    if (onLinkSelected) onLinkSelected(link.id);
                }
            }
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
        
        void drawGroup(ImDrawList* dl, NodeGroup& group) {
            ImVec2 pos = nodeToScreen(group.position.x, group.position.y);
            ImVec2 size(group.size.x * zoom, group.size.y * zoom);
            
            float radius = 8.0f * zoom;
            
            ImVec4 colF = ImGui::ColorConvertU32ToFloat4(group.color);
            // Soft translucent group backdrop
            ImU32 bgCol = ImGui::ColorConvertFloat4ToU32(ImVec4(colF.x, colF.y, colF.z, 0.05f));
            
            // Background
            dl->AddRectFilled(pos, ImVec2(pos.x + size.x, pos.y + size.y), bgCol, radius);
            
            // Border
            bool isSelected = (selectedGroupId == group.id);
            ImU32 borderCol = isSelected 
                ? config.nodeSelectedColor 
                : ImGui::ColorConvertFloat4ToU32(ImVec4(colF.x, colF.y, colF.z, 0.30f));
                
            dl->AddRect(pos, ImVec2(pos.x + size.x, pos.y + size.y), borderCol, radius, 0, 
                isSelected ? 1.8f * zoom : 1.0f * zoom);
            
            // Resize Handle (Gaea-style)
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
                minX = std::min(minX, node->x);
                minY = std::min(minY, node->y);
                maxX = std::max(maxX, node->x + 200);
                maxY = std::max(maxY, node->y + 100);
            }
            
            float rangeX = maxX - minX + 100;
            float rangeY = maxY - minY + 100;
            float scale = std::min(mmSize / rangeX, mmSize / rangeY) * 0.8f;
            
            // Draw nodes
            for (auto& node : graph.nodes) {
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
            float customW = node.getCustomWidth();
            float padding = scaleNodeChromeMetric(zoom, 10.0f, 7.0f, 14.0f);
            float titleW = ImGui::CalcTextSize(node.metadata.displayName.c_str()).x + padding * 2;
            if (titleW < 10) titleW = ImGui::CalcTextSize(node.name.c_str()).x + padding * 2;
            NodeChromeLayout chrome = buildNodeChromeLayout(node, zoom, customW > 0.0f ? customW * zoom : 160.0f * zoom,
                node.inputs.size(), node.outputs.size(), titleW);
            
            ImVec2 nodeScreenPos = nodeToScreen(node.x, node.y);
            ImVec2 center = ImVec2(nodeScreenPos.x + chrome.width * 0.5f, nodeScreenPos.y + chrome.height * 0.5f);
            
            float threshold = 18.0f * zoom; // distance threshold
            
            for (auto& link : graph.links) {
                NodeBase* startOwner = graph.getPinOwner(link.startPinId);
                NodeBase* endOwner = graph.getPinOwner(link.endPinId);
                if (!startOwner || !endOwner) continue;
                if (startOwner->id == node.id || endOwner->id == node.id) continue;
                
                auto itStart = pinPositions_.find(link.startPinId);
                auto itEnd = pinPositions_.find(link.endPinId);
                if (itStart == pinPositions_.end() || itEnd == pinPositions_.end()) continue;
                
                ImVec2 p1 = itStart->second;
                ImVec2 p2 = itEnd->second;
                
                float dist = std::abs(p1.x - p2.x);
                float cpDist = std::max(dist * 0.5f, 50.0f * zoom);
                ImVec2 cp1(p1.x + cpDist, p1.y);
                ImVec2 cp2(p2.x - cpDist, p2.y);
                
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
                    
                    // Find compatible input on node
                    Pin* compatibleIn = nullptr;
                    for (auto& nIn : node.inputs) {
                        if (outPin->canConnectTo(nIn)) {
                            compatibleIn = &nIn;
                            break;
                        }
                    }
                    
                    // Find compatible output on node
                    Pin* compatibleOut = nullptr;
                    for (auto& nOut : node.outputs) {
                        if (nOut.canConnectTo(*inPin)) {
                            compatibleOut = &nOut;
                            break;
                        }
                    }
                    
                    if (compatibleIn && compatibleOut) {
                        graph.removeLink(link.id);
                        graph.addLink(outPin->id, compatibleIn->id);
                        graph.addLink(compatibleOut->id, inPin->id);
                        
                        if (onGraphModified) onGraphModified();
                        break; // Inserted successfully
                    }
                }
            }
        }
        
        uint32_t findClosestPin(const ImVec2& mouse, float maxDist) {
            uint32_t closest = 0;
            float bestDist = maxDist;
            
            for (auto& [pinId, pos] : pinPositions_) {
                if (pinId == linkStartPinId) continue;
                float d = std::hypot(pos.x - mouse.x, pos.y - mouse.y);
                if (d < bestDist) {
                    bestDist = d;
                    closest = pinId;
                }
            }
            
            return closest;
        }
        
        bool isLinkHovered(const ImVec2& p1, const ImVec2& cp1, 
                          const ImVec2& cp2, const ImVec2& p2) {
            ImVec2 mouse = ImGui::GetMousePos();
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

