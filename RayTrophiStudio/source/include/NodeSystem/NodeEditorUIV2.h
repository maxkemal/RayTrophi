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
            float nodeRounding = 10.0f;
            float nodeBorderWidth = 1.5f;
            
            // Link
            float linkThickness = 2.5f;
            float linkSelectedThickness = 4.0f;
            
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
        uint32_t selectedLinkId = 0;
        uint32_t selectedGroupId = 0;
        uint32_t draggingNodeId = 0;
        uint32_t resizingNodeId = 0;
        
        bool isCreatingLink = false;
        uint32_t linkStartPinId = 0;
        DataType linkStartType = DataType::None;
        
        // Callbacks
        std::function<void(uint32_t nodeId)> onNodeSelected;
        std::function<void(uint32_t linkId)> onLinkSelected;
        std::function<void()> onBackgroundContextMenu;
        std::function<void(uint32_t nodeId)> onNodeContextMenu;
        std::function<void()> onGraphModified;

        /**
         * @brief Reset editor state (zoom, scroll, selection)
         */
        void reset() {
            zoom = 1.0f;
            scrollX = 0.0f;
            scrollY = 0.0f;
            selectedNodeId = 0;
            selectedLinkId = 0;
            selectedGroupId = 0;
            draggingNodeId = 0;
            resizingNodeId = 0;
            isCreatingLink = false;
            linkStartPinId = 0;
        }

    private:
        std::unordered_map<uint32_t, ImVec2> pinPositions_;
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
            
            ImGui::EndChild();
        }

    private:
        // ========================================================================
        // GRID
        // ========================================================================
        
        void drawGrid(ImDrawList* dl) {
            float minorStep = config.gridSizeMinor * zoom;
            float majorStep = config.gridSizeMajor * zoom;
            
            // Minor grid
            if (minorStep > 8.0f) {
                for (float x = fmodf(scrollX, minorStep); x < canvasSize_.x; x += minorStep) {
                    dl->AddLine(
                        ImVec2(canvasPos_.x + x, canvasPos_.y),
                        ImVec2(canvasPos_.x + x, canvasPos_.y + canvasSize_.y),
                        config.gridColorMinor
                    );
                }
                for (float y = fmodf(scrollY, minorStep); y < canvasSize_.y; y += minorStep) {
                    dl->AddLine(
                        ImVec2(canvasPos_.x, canvasPos_.y + y),
                        ImVec2(canvasPos_.x + canvasSize_.x, canvasPos_.y + y),
                        config.gridColorMinor
                    );
                }
            }
            
            // Major grid
            for (float x = fmodf(scrollX, majorStep); x < canvasSize_.x; x += majorStep) {
                dl->AddLine(
                    ImVec2(canvasPos_.x + x, canvasPos_.y),
                    ImVec2(canvasPos_.x + x, canvasPos_.y + canvasSize_.y),
                    config.gridColorMajor
                );
            }
            for (float y = fmodf(scrollY, majorStep); y < canvasSize_.y; y += majorStep) {
                dl->AddLine(
                    ImVec2(canvasPos_.x, canvasPos_.y + y),
                    ImVec2(canvasPos_.x + canvasSize_.x, canvasPos_.y + y),
                    config.gridColorMajor
                );
            }
        }

        // ========================================================================
        // INPUT
        // ========================================================================
        
        void handleInput(GraphBase& graph) {
            ImGui::SetCursorScreenPos(canvasPos_);
            ImGui::InvisibleButton("##CanvasInput", canvasSize_,
                ImGuiButtonFlags_MouseButtonMiddle | ImGuiButtonFlags_MouseButtonRight);
            ImGui::SetItemAllowOverlap();
            
            // Pan (middle mouse)
            if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Middle)) {
                scrollX += ImGui::GetIO().MouseDelta.x;
                scrollY += ImGui::GetIO().MouseDelta.y;
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
            
            // Context menu (right click on background)
            if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
                if (onBackgroundContextMenu) onBackgroundContextMenu();
            }
            
            // Delete key
            if (ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows)) {
                if (ImGui::IsKeyPressed(ImGuiKey_Delete) || ImGui::IsKeyPressed(ImGuiKey_Backspace)) {
                    if (selectedLinkId != 0) {
                        graph.removeLink(selectedLinkId);
                        selectedLinkId = 0;
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
                    }
                    
                    isCreatingLink = false;
                    linkStartPinId = 0;
                }
            }
            
            // Node dragging
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
            } else if (draggingNodeId != 0) {
                if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
                    if (ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.0f)) {
                        NodeBase* node = graph.getNode(draggingNodeId);
                        if (node) {
                            node->x += ImGui::GetIO().MouseDelta.x / zoom;
                            node->y += ImGui::GetIO().MouseDelta.y / zoom;
                        }
                    }
                } else {
                    draggingNodeId = 0;
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
            chrome.cornerRadius = scaleNodeChromeMetric(zoom, config.nodeRounding, 8.0f, 16.0f);

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
            
            bool isSelected = (selectedNodeId == node.id);
            
            ImU32 headerCol = node.metadata.headerColor;
            if (headerCol == 0) {
                headerCol = ImGui::ColorConvertFloat4ToU32(node.headerColor);
            }
            
            ImU32 borderCol = isSelected ? config.nodeSelectedColor : config.nodeBorderColor;
            float borderW = isSelected ? config.nodeBorderWidth * 2 : config.nodeBorderWidth;
            
            dl->AddRectFilled(
                ImVec2(pos.x + shadowOffset, pos.y + shadowOffset),
                ImVec2(pos.x + finalWidth + shadowOffset, pos.y + finalHeight + shadowOffset),
                IM_COL32(0, 0, 0, 55), cornerRadius);

            // Body
            dl->AddRectFilled(pos, ImVec2(pos.x + finalWidth, pos.y + finalHeight),
                config.nodeBodyColor, cornerRadius);
            
            // Header
            dl->AddRectFilled(pos, ImVec2(pos.x + finalWidth, pos.y + headerH),
                headerCol, cornerRadius, ImDrawFlags_RoundCornersTop);

            dl->AddLine(
                ImVec2(pos.x + 1.0f, pos.y + headerH),
                ImVec2(pos.x + finalWidth - 1.0f, pos.y + headerH),
                IM_COL32(255, 255, 255, 22),
                1.0f);
            
            // Border
            dl->AddRect(pos, ImVec2(pos.x + finalWidth, pos.y + finalHeight),
                borderCol, cornerRadius, 0, std::max(1.0f, borderW * zoom));
            
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
            
            if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                selectedNodeId = node.id;
                draggingNodeId = node.id;
                node.x += ImGui::GetIO().MouseDelta.x / zoom;
                node.y += ImGui::GetIO().MouseDelta.y / zoom;
            }
            
            if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
                selectedNodeId = node.id;
                selectedLinkId = 0;
                if (onNodeSelected) onNodeSelected(node.id);
            }
            
            // Double-click to collapse/expand
            if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                node.collapsed = !node.collapsed;
            }
            
            if (ImGui::IsItemClicked(ImGuiMouseButton_Right)) {
                if (onNodeContextMenu) onNodeContextMenu(node.id);
            }
            
            ImGui::PopID();
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
            
            bool isSelected = (selectedLinkId == link.id);
            float thickness = isSelected ? config.linkSelectedThickness : config.linkThickness;
            thickness = std::max(1.5f, thickness * zoom);
            
            // Bezier control points
            float dist = std::abs(p1.x - p2.x);
            float cpDist = std::max(dist * 0.5f, 50.0f * zoom);
            ImVec2 cp1(p1.x + cpDist, p1.y);
            ImVec2 cp2(p2.x - cpDist, p2.y);
            
            // Draw glow if selected
            if (isSelected) {
                dl->AddBezierCubic(p1, cp1, cp2, p2, IM_COL32(100, 150, 255, 60), 
                    thickness + 4.0f * zoom);
            }
            
            dl->AddBezierCubic(p1, cp1, cp2, p2, col, thickness);
            
            // Hit test
            if (isLinkHovered(p1, cp1, cp2, p2)) {
                dl->AddBezierCubic(p1, cp1, cp2, p2, IM_COL32(255, 255, 255, 80), 
                    thickness + 2.0f * zoom);
                
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
            
            // Background
            dl->AddRectFilled(pos, ImVec2(pos.x + size.x, pos.y + size.y), group.color, 4.0f);
            
            // Border
            bool isSelected = (selectedGroupId == group.id);
            ImU32 borderCol = isSelected ? config.nodeSelectedColor : config.groupBorderColor;
            dl->AddRect(pos, ImVec2(pos.x + size.x, pos.y + size.y), borderCol, 4.0f, 0, 
                isSelected ? 2.0f : 1.0f);
            
            // Title
            if (zoom > 0.3f && !group.name.empty()) {
                dl->AddText(ImVec2(pos.x + 8, pos.y + 4), config.groupTitleColor, group.name.c_str());
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

