#pragma once

/**
 * @file scene_ui_nodeeditor.hpp
 * @brief Terrain Node Editor UI using V2 NodeSystem
 * 
 * This file implements the terrain node editor using the modern V2 node system
 * with NodeEditorUIV2 for rendering and TerrainNodesV2 for terrain operations.
 */

#include "imgui.h"
#include "TerrainNodesV2.h"
#include "TerrainManager.h"
#include "NodeSystem/NodeEditorUIV2.h"
#include "ProjectManager.h" // Added for markModified
#include <functional> // Added for callback
#include <string>
#include <unordered_map>
#include <algorithm>
// #include <functional>
// #include <scene_ui.h> // Dependency removed // Dependency removed to fix circular include

extern bool g_bvh_rebuild_pending;
extern bool g_optix_rebuild_pending;

namespace TerrainNodesV2 {

// Node categories for library panel
struct NodeCategory {
    const char* name;
    ImVec4 color;
    std::vector<std::pair<NodeType, const char*>> nodes;
};

class TerrainNodeEditorUI {
public:
    // Callback for file dialog to prevent circular dependency
    std::function<std::string(const wchar_t*)> onOpenFileDialog;
    std::function<std::string(const wchar_t*, const wchar_t*)> onSaveFileDialog; // filter, default name
    TerrainObject* currentTerrain = nullptr;  // For export operations
    
    // Resizable properties panel width
    float propertiesWidth = 220.0f;
    float minPropertiesWidth = 150.0f;
    float maxPropertiesWidth = 400.0f;
    bool isResizingProperties = false;
    
    // Mask preview cache and controls
    uint32_t lastPreviewedNodeId = 0;
    std::vector<float> lastPreviewedData;
    int lastPreviewedWidth = 0;
    int lastPreviewedHeight = 0;
    float previewZoom = 1.0f;
    float previewPanX = 0.0f;
    float previewPanY = 0.0f;
    bool isPanningPreview = false;

    TerrainNodeEditorUI() {
        // Setup node categories
        categories = {
            {"Input", ImVec4(0.2f, 0.7f, 0.3f, 1.0f), {
                {NodeType::HeightmapInput, "Heightmap"},
                {NodeType::NoiseGenerator, "Noise Generator"}
            }},
            {"Erosion", ImVec4(0.3f, 0.5f, 0.9f, 1.0f), {
                {NodeType::ErosionWizard, "Erosion Wizard"},
                {NodeType::HydraulicErosion, "Hydraulic"},
                {NodeType::ThermalErosion, "Thermal"},
                {NodeType::FluvialErosion, "Fluvial"},
                {NodeType::WindErosion, "Wind"},
                {NodeType::SedimentDeposition, "Sediment Deposit"},
                {NodeType::AlluvialFan, "Alluvial Fan"},
                {NodeType::DeltaFormation, "Delta Formation"}
            }},
            {"Filter", ImVec4(0.4f, 0.6f, 0.8f, 1.0f), {
                {NodeType::Smooth, "Smooth"},
                {NodeType::Normalize, "Normalize"},
                {NodeType::Terrace, "Terrace"}
            }},
            {"Mask", ImVec4(0.7f, 0.4f, 0.8f, 1.0f), {
                {NodeType::HeightMask, "Height Mask"},
                {NodeType::SlopeMask, "Slope Mask"},
                {NodeType::CurvatureMask, "Curvature Mask"},
                {NodeType::FlowMask, "Flow / Soil"},
                {NodeType::ExposureMask, "Sun Exposure"},
                {NodeType::MaskCombine, "Mask Combine"},
                {NodeType::MaskPaint, "Mask Paint"},
                {NodeType::MaskImage, "Mask Image"}
            }},
            {"Math", ImVec4(0.8f, 0.7f, 0.3f, 1.0f), {
                {NodeType::Add, "Add"},
                {NodeType::Subtract, "Subtract"},
                {NodeType::Multiply, "Multiply"},
                {NodeType::Blend, "Blend"},
                {NodeType::Clamp, "Clamp"},
                {NodeType::Invert, "Invert"}
            }},
            {"Blend Modes", ImVec4(0.6f, 0.4f, 0.7f, 1.0f), {
                {NodeType::Overlay, "Overlay"},
                {NodeType::Screen, "Screen"}
            }},
            {"Output", ImVec4(0.9f, 0.3f, 0.3f, 1.0f), {
                {NodeType::HeightOutput, "Height Output"},
                {NodeType::SplatOutput, "Splat Output"}
            }},
            {"Texture", ImVec4(0.8f, 0.6f, 0.2f, 1.0f), {
                {NodeType::AutoSplat, "Auto Splat"}
            }},
            {"Geology", ImVec4(0.7f, 0.45f, 0.35f, 1.0f), {
                {NodeType::Fault, "Fault Line"},
                {NodeType::Mesa, "Mesa / Plateau"},
                {NodeType::Shear, "Shear Zone"}
            }}
        };
        
        // Initialize V2 Editor
        editor.config.gridSizeMinor = 32.0f;
        editor.config.showMinimap = true;
        
        // Context Menu Callback
        editor.onBackgroundContextMenu = []() {
            ImGui::OpenPopup("NodeGraphContext");
        };
        
        // Node selected callback
        editor.onNodeSelected = [](uint32_t nodeId) {
            // Could trigger property panel refresh
        };
        
        // Graph modification callback (Mark Project Dirty)
        editor.onGraphModified = []() {
            ProjectManager::getInstance().markModified();
        };
    }
    
    // Templated Context to avoid circular dependency with scene_ui.h
    template<typename ContextT>
    void draw(ContextT& ctx, TerrainNodeGraphV2& graph, TerrainObject* terrain) {
        if (!terrain) {
            ImGui::TextColored(ImVec4(1, 0.5f, 0.3f, 1), "Select a terrain in Terrain tab first");
            return;
        }
        
        // Update current terrain reference for export operations
        currentTerrain = terrain;
        
        float libraryWidth = 160.0f;
        
        // Left Panel - Node Library
        ImGui::BeginChild("NodeLibrary", ImVec2(libraryWidth, 0), true);
        drawNodeLibrary(graph, terrain);
        ImGui::EndChild();
        
        ImGui::SameLine();
        
        // Middle Panel - Node Canvas (will take remaining space minus properties panel)
        float availWidth = ImGui::GetContentRegionAvail().x;
        float canvasWidth = availWidth - propertiesWidth - 8; // 8 for spacing
        
        ImGui::BeginChild("NodeCanvas", ImVec2(canvasWidth, 0), true, ImGuiWindowFlags_NoScrollbar);
        
        drawToolbar(ctx, graph, terrain);
        
        // Get canvas position before drawing (for drop coordinate calculation)
        ImVec2 canvasPos = ImGui::GetCursorScreenPos();
        ImVec2 canvasSize = ImGui::GetContentRegionAvail();
        
        // Render V2 Node Graph
        editor.draw(graph, canvasSize);
        
        // Drop target for node library drag-drop
        // Make the whole canvas area accept drops
        if (ImGui::BeginDragDropTarget()) {
            if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("TERRAIN_NODE_TYPE")) {
                NodeType droppedType = *(const NodeType*)payload->Data;
                
                // Calculate spawn position from mouse position
                ImVec2 mousePos = ImGui::GetMousePos();
                float spawnX = (mousePos.x - canvasPos.x - editor.scrollX) / editor.zoom;
                float spawnY = (mousePos.y - canvasPos.y - editor.scrollY) / editor.zoom;
                
                // Check if dropped on a link - insert node in between
                NodeSystem::Link* targetLink = findLinkNearMouse(graph, mousePos, canvasPos);
                
                if (targetLink) {
                    // Create the new node
                    auto* newNode = graph.addTerrainNode(droppedType, spawnX, spawnY);
                    
                    if (newNode && tryInsertNodeIntoLink(graph, newNode, targetLink)) {
                        // Successfully inserted!
                        ProjectManager::getInstance().markModified();
                    }
                    // If insertion failed, node is still created at position
                } else {
                    // Normal drop - just create node
                    graph.addTerrainNode(droppedType, spawnX, spawnY);
                    ProjectManager::getInstance().markModified();
                }
            }
            ImGui::EndDragDropTarget();
        }
        
        // Render Context Menu
        drawContextMenu(graph);
        
        ImGui::EndChild();
        
        ImGui::SameLine();
        
        // ========================================================================
        // RESIZABLE SPLITTER
        // ========================================================================
        ImGui::InvisibleButton("##PropertiesSplitter", ImVec2(6.0f, ImGui::GetContentRegionAvail().y));
        if (ImGui::IsItemActive()) {
            isResizingProperties = true;
            propertiesWidth -= ImGui::GetIO().MouseDelta.x;
            propertiesWidth = std::clamp(propertiesWidth, minPropertiesWidth, maxPropertiesWidth);
        }
        if (ImGui::IsItemHovered() || isResizingProperties) {
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
        }
        if (!ImGui::IsMouseDown(0)) {
            isResizingProperties = false;
        }
        
        // Draw splitter visual feedback
        ImVec2 splitterMin = ImGui::GetItemRectMin();
        ImVec2 splitterMax = ImGui::GetItemRectMax();
        ImDrawList* dl = ImGui::GetWindowDrawList();
        ImU32 splitterColor = ImGui::IsItemHovered() || isResizingProperties 
            ? IM_COL32(100, 150, 255, 200) 
            : IM_COL32(80, 80, 90, 150);
        dl->AddRectFilled(splitterMin, splitterMax, splitterColor);
        
        ImGui::SameLine();
        
        // ========================================================================
        // RIGHT PANEL - PROPERTIES (with border)
        // ========================================================================
        ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 1.0f);
        ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.4f, 0.4f, 0.5f, 1.0f));
        ImGui::BeginChild("NodeProperties", ImVec2(propertiesWidth, 0), true);
        drawPropertiesPanel(graph);
        ImGui::EndChild();
        ImGui::PopStyleColor();
        ImGui::PopStyleVar();
    }
    
    void drawPropertiesPanel(TerrainNodeGraphV2& graph) {
        ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.8f, 1.0f), "Node Properties");
        ImGui::Separator();

        // Get selected node
        uint32_t selectedId = editor.selectedNodeId;
        if (selectedId == 0) {
            ImGui::TextDisabled("Select a node to edit");
            ImGui::TextDisabled("its properties here.");
            return;
        }

        NodeSystem::NodeBase* node = graph.getNode(selectedId);
        if (!node) {
            ImGui::TextDisabled("Node not found");
            return;
        }

        // Node Info Header
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 0.8f, 1.0f, 1.0f));
        ImGui::Text("%s", node->metadata.displayName.empty() ? node->name.c_str() : node->metadata.displayName.c_str());
        ImGui::PopStyleColor();

        // Category
        if (!node->metadata.category.empty()) {
            ImGui::TextDisabled("Category: %s", node->metadata.category.c_str());
        }
        ImGui::TextDisabled("ID: %u", node->id);

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // Parameters header
        ImGui::TextColored(ImVec4(0.9f, 0.9f, 0.6f, 1.0f), "Parameters");
        ImGui::Separator();
        ImGui::Spacing();

        // Draw node's content (parameters) with proper width
        float labelWidth = 80.0f;
        ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - labelWidth);
        
        bool wasDirty = node->dirty;
        node->drawContent();
        if (node->dirty && !wasDirty) {
            ProjectManager::getInstance().markModified();
        }
        
        ImGui::PopItemWidth();

        // Handle HeightmapInputNode file dialog request
        auto* heightmapNode = dynamic_cast<TerrainNodesV2::HeightmapInputNode*>(node);
        if (heightmapNode && heightmapNode->browseForHeightmap) {
            heightmapNode->browseForHeightmap = false;

            if (onOpenFileDialog) {
                std::string path = onOpenFileDialog(L"Heightmap Files\0*.png;*.jpg;*.jpeg;*.bmp;*.raw;*.r16\0");
                if (!path.empty()) {
                    strncpy(heightmapNode->filePath, path.c_str(), sizeof(heightmapNode->filePath) - 1);
                    heightmapNode->loadHeightmapFromFile();
                }
            }
            else {
                ImGui::TextColored(ImVec4(1, 0, 0, 1), "Error: File dialog callback not set");
            }
        }
        
        // Handle MaskImageNode file dialog request
        auto* maskImageNode = dynamic_cast<TerrainNodesV2::MaskImageNode*>(node);
        if (maskImageNode && maskImageNode->browseForMask) {
            maskImageNode->browseForMask = false;

            if (onOpenFileDialog) {
                std::string path = onOpenFileDialog(L"Mask Files\0*.png;*.jpg;*.jpeg;*.bmp\0");
                if (!path.empty()) {
                    strncpy(maskImageNode->filePath, path.c_str(), sizeof(maskImageNode->filePath) - 1);
                    maskImageNode->loadMaskFromFile();
                }
            }
        }
        
        // Handle SplatOutputNode export dialog request
        auto* splatOutputNode = dynamic_cast<TerrainNodesV2::SplatOutputNode*>(node);
        if (splatOutputNode && splatOutputNode->browseForExport) {
            splatOutputNode->browseForExport = false;
            
            if (onSaveFileDialog) {
                std::string path = onSaveFileDialog(L"PNG Files\0*.png\0", L"splat_map.png");
                if (!path.empty()) {
                    strncpy(splatOutputNode->exportPath, path.c_str(), sizeof(splatOutputNode->exportPath) - 1);
                    if (currentTerrain) {
                        splatOutputNode->exportSplatMap(currentTerrain);
                    }
                }
            }
        }
        
        // ========================================================================
        // MASK NODE PREVIEW
        // ========================================================================
        drawMaskPreview(graph, node);
    }
    
    void drawMaskPreview(TerrainNodeGraphV2& graph, NodeSystem::NodeBase* node) {
        auto* terrainNode = dynamic_cast<TerrainNodesV2::TerrainNodeBase*>(node);
        if (!terrainNode) return;
        
        // Check if this is a mask-producing node
        bool isMaskNode = false;
        int maskOutputIndex = -1;
        
        for (size_t i = 0; i < node->outputs.size(); i++) {
            if (node->outputs[i].imageSemantic == NodeSystem::ImageSemantic::Mask) {
                isMaskNode = true;
                maskOutputIndex = static_cast<int>(i);
                break;
            }
        }
        
        if (!isMaskNode || maskOutputIndex < 0) return;
        
        // Skip preview for heavy nodes (Wizard)
        if (terrainNode->terrainNodeType == TerrainNodesV2::NodeType::ErosionWizard) {
            ImGui::TextDisabled("Preview disabled for heavy node");
            return;
        }
        
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        
        // Check if we need to update preview (node changed or refresh clicked)
        bool nodeChanged = (lastPreviewedNodeId != node->id);
        
        // Wrap preview in collapsed header (Closed by default to prevent freeze)
        if (ImGui::CollapsingHeader("Mask Preview (Click to Show)", ImGuiTreeNodeFlags_None)) {
            
            // Refresh button inside the header
            ImGui::SameLine(ImGui::GetContentRegionAvail().x - 60);
            bool refreshClicked = ImGui::SmallButton("Refresh");
            
            bool needsUpdate = nodeChanged || refreshClicked || (lastPreviewedWidth == 0);
            
            if (needsUpdate && currentTerrain) {
                // Create temporary evaluation context to compute this node's output
                TerrainNodesV2::TerrainContext tctx(currentTerrain);
                NodeSystem::EvaluationContext ctx(&graph);
                ctx.setDomainContext(&tctx);
                
                // Request the mask output from this node
                NodeSystem::PinValue result = node->requestOutput(maskOutputIndex, ctx);
                
                // Extract Image2DData from the result
                auto* imgData = std::get_if<NodeSystem::Image2DData>(&result);
                if (imgData && imgData->isValid()) {
                    lastPreviewedData = *(imgData->data);
                    lastPreviewedNodeId = node->id;
                    lastPreviewedWidth = imgData->width;
                    lastPreviewedHeight = imgData->height;
                    // Reset pan when switching nodes
                    if (nodeChanged) {
                        previewPanX = 0.0f;
                        previewPanY = 0.0f;
                        previewZoom = 1.0f;
                    }
                }
            }
        
        // Draw preview if we have data
        if (!lastPreviewedData.empty() && lastPreviewedNodeId == node->id && lastPreviewedWidth > 0) {
            float previewSize = ImGui::GetContentRegionAvail().x;
            
            int srcW = lastPreviewedWidth;
            int srcH = lastPreviewedHeight;
            
            // Normalize values to 0-1 for display
            float minVal = *std::min_element(lastPreviewedData.begin(), lastPreviewedData.end());
            float maxVal = *std::max_element(lastPreviewedData.begin(), lastPreviewedData.end());
            float range = maxVal - minVal;
            if (range < 0.0001f) range = 1.0f;
            
            // Wrap preview in a child window to capture scroll events
            ImGui::BeginChild("##MaskPreviewChild", ImVec2(previewSize, previewSize + 50), 
                             false, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
            
            ImVec2 p = ImGui::GetCursorScreenPos();
            ImDrawList* dl = ImGui::GetWindowDrawList();
            
            // Draw preview box background
            ImVec2 previewRect(previewSize, previewSize);
            dl->AddRectFilled(p, ImVec2(p.x + previewRect.x, p.y + previewRect.y), 
                             IM_COL32(20, 20, 25, 255));
            
            // Higher resolution preview with zoom/pan
            int displayRes = std::min(128, std::max(srcW, srcH)); // Higher quality
            float cellSize = previewSize / displayRes;
            
            // Calculate visible area based on zoom and pan
            float viewSize = 1.0f / previewZoom;
            float viewX = previewPanX - viewSize * 0.5f + 0.5f;
            float viewY = previewPanY - viewSize * 0.5f + 0.5f;
            
            for (int py = 0; py < displayRes; py++) {
                for (int px = 0; px < displayRes; px++) {
                    // Calculate source UV with zoom/pan
                    float u = viewX + (float(px) / displayRes) * viewSize;
                    float v = viewY + (float(py) / displayRes) * viewSize;
                    
                    // Clamp to valid range
                    u = std::clamp(u, 0.0f, 1.0f);
                    v = std::clamp(v, 0.0f, 1.0f);
                    
                    // Sample with bilinear interpolation
                    float fx = u * (srcW - 1);
                    float fy = v * (srcH - 1);
                    int x0 = static_cast<int>(fx);
                    int y0 = static_cast<int>(fy);
                    int x1 = std::min(x0 + 1, srcW - 1);
                    int y1 = std::min(y0 + 1, srcH - 1);
                    float tx = fx - x0;
                    float ty = fy - y0;
                    
                    int idx00 = y0 * srcW + x0;
                    int idx10 = y0 * srcW + x1;
                    int idx01 = y1 * srcW + x0;
                    int idx11 = y1 * srcW + x1;
                    
                    float v00 = lastPreviewedData[idx00];
                    float v10 = lastPreviewedData[idx10];
                    float v01 = lastPreviewedData[idx01];
                    float v11 = lastPreviewedData[idx11];
                    
                    float val = v00 * (1 - tx) * (1 - ty) + v10 * tx * (1 - ty) +
                                v01 * (1 - tx) * ty + v11 * tx * ty;
                    
                    val = (val - minVal) / range;
                    val = std::clamp(val, 0.0f, 1.0f);
                    
                    int gray = static_cast<int>(val * 255.0f);
                    ImU32 color = IM_COL32(gray, gray, gray, 255);
                    
                    ImVec2 cellMin(p.x + px * cellSize, p.y + py * cellSize);
                    ImVec2 cellMax(cellMin.x + cellSize, cellMin.y + cellSize);
                    dl->AddRectFilled(cellMin, cellMax, color);
                }
            }
            
            // Border
            dl->AddRect(p, ImVec2(p.x + previewRect.x, p.y + previewRect.y), 
                       IM_COL32(80, 80, 90, 255));
            
            // Handle mouse interaction for pan/zoom
            ImGui::InvisibleButton("##PreviewInteract", previewRect);
            bool isHovered = ImGui::IsItemHovered();
            if (isHovered) {
                // Zoom with scroll wheel - consume the event
                float wheel = ImGui::GetIO().MouseWheel;
                if (wheel != 0.0f) {
                    previewZoom *= (wheel > 0) ? 1.2f : (1.0f / 1.2f);
                    previewZoom = std::clamp(previewZoom, 0.5f, 8.0f);
                    // Clear scroll to prevent parent from receiving it
                    ImGui::GetIO().MouseWheel = 0.0f;
                }
                
                // Pan with middle mouse or left drag
                if (ImGui::IsMouseDragging(ImGuiMouseButton_Left) || 
                    ImGui::IsMouseDragging(ImGuiMouseButton_Middle)) {
                    ImVec2 delta = ImGui::GetIO().MouseDelta;
                    previewPanX -= delta.x / previewSize / previewZoom;
                    previewPanY -= delta.y / previewSize / previewZoom;
                }
            }
            
            // Zoom controls (inside child window)
            ImGui::Text("Zoom: %.1fx", previewZoom);
            ImGui::SameLine();
            if (ImGui::SmallButton("Reset")) {
                previewZoom = 1.0f;
                previewPanX = 0.0f;
                previewPanY = 0.0f;
            }
            
            ImGui::EndChild();
            
            // Stats (outside child window)
            ImGui::TextDisabled("Size: %dx%d", srcW, srcH);
            ImGui::TextDisabled("Range: %.3f - %.3f", minVal, maxVal);
        } else {
            ImGui::TextDisabled("No preview available");
            ImGui::TextDisabled("Select terrain first");
        }
    } // End CollapsingHeader
    } // End drawMaskPreview
    // Access graph for external use  
    TerrainNodeGraphV2& getGraph() { return ownedGraph; }
    
private:
    std::vector<NodeCategory> categories;
    char searchBuffer[128] = "";
    
    // Drag-drop state
    NodeType pendingDragNodeType = NodeType::HeightmapInput;
    bool isDragging = false;
    
    // V2 Editor Instance
    NodeSystem::NodeEditorUIV2 editor;
    
    // Owned graph instance (if not passed externally)
    TerrainNodeGraphV2 ownedGraph;

    // ========================================================================
    // LINK INSERTION HELPERS
    // ========================================================================
    
    /**
     * @brief Find a link near the mouse position for drop-on-link insertion
     */
    NodeSystem::Link* findLinkNearMouse(TerrainNodeGraphV2& graph, ImVec2 mousePos, ImVec2 canvasPos) {
        const float threshold = 15.0f;  // Distance threshold for hit detection
        
        for (auto& link : graph.links) {
            // Get pin positions - need to calculate bezier curve
            NodeSystem::Pin* startPin = graph.findPin(link.startPinId);
            NodeSystem::Pin* endPin = graph.findPin(link.endPinId);
            if (!startPin || !endPin) continue;
            
            NodeSystem::NodeBase* startNode = graph.getPinOwner(link.startPinId);
            NodeSystem::NodeBase* endNode = graph.getPinOwner(link.endPinId);
            if (!startNode || !endNode) continue;
            
            // Approximate pin screen positions
            ImVec2 p1(canvasPos.x + startNode->x * editor.zoom + editor.scrollX + 160 * editor.zoom,
                      canvasPos.y + startNode->y * editor.zoom + editor.scrollY + 50 * editor.zoom);
            ImVec2 p2(canvasPos.x + endNode->x * editor.zoom + editor.scrollX,
                      canvasPos.y + endNode->y * editor.zoom + editor.scrollY + 50 * editor.zoom);
            
            // Check if mouse is near the bezier curve
            float dist = std::abs(p1.x - p2.x);
            float cpDist = std::max(dist * 0.5f, 50.0f * editor.zoom);
            ImVec2 cp1(p1.x + cpDist, p1.y);
            ImVec2 cp2(p2.x - cpDist, p2.y);
            
            // Sample bezier and check distance
            ImVec2 prev = p1;
            for (int i = 1; i <= 20; i++) {
                float t = (float)i / 20.0f;
                float u = 1.0f - t;
                ImVec2 p;
                p.x = u*u*u*p1.x + 3*u*u*t*cp1.x + 3*u*t*t*cp2.x + t*t*t*p2.x;
                p.y = u*u*u*p1.y + 3*u*u*t*cp1.y + 3*u*t*t*cp2.y + t*t*t*p2.y;
                
                float d = pointSegmentDistance(mousePos, prev, p);
                if (d < threshold) {
                    return &link;
                }
                prev = p;
            }
        }
        return nullptr;
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
    
    /**
     * @brief Try to insert a node into an existing link
     * @returns true if successfully inserted
     */
    bool tryInsertNodeIntoLink(TerrainNodeGraphV2& graph, NodeSystem::NodeBase* newNode, 
                               NodeSystem::Link* link) {
        if (!newNode || !link) return false;
        
        // Check if node has compatible pins
        // Need: at least one input that matches link's output type
        //       at least one output that matches link's input type
        
        NodeSystem::Pin* sourcePin = graph.findPin(link->startPinId);
        NodeSystem::Pin* destPin = graph.findPin(link->endPinId);
        
        if (!sourcePin || !destPin) return false;
        
        // Find matching input on new node (to receive data from source)
        NodeSystem::Pin* matchingInput = nullptr;
        for (auto& pin : newNode->inputs) {
            if (pin.canConnectTo(*sourcePin) || sourcePin->canConnectTo(pin)) {
                matchingInput = &pin;
                break;
            }
        }
        
        // Find matching output on new node (to send data to dest)
        NodeSystem::Pin* matchingOutput = nullptr;
        for (auto& pin : newNode->outputs) {
            if (pin.canConnectTo(*destPin) || destPin->canConnectTo(pin)) {
                matchingOutput = &pin;
                break;
            }
        }
        
        // Both must match for insertion
        if (!matchingInput || !matchingOutput) return false;
        
        // Store old link info
        uint32_t oldStartPinId = link->startPinId;
        uint32_t oldEndPinId = link->endPinId;
        uint32_t oldLinkId = link->id;
        
        // Remove old link
        graph.removeLink(oldLinkId);
        
        // Create new links
        // Source -> NewNode input
        graph.addLink(oldStartPinId, matchingInput->id);
        // NewNode output -> Destination
        graph.addLink(matchingOutput->id, oldEndPinId);
        
        return true;
    }

    void drawContextMenu(TerrainNodeGraphV2& graph) {
        if (ImGui::BeginPopup("NodeGraphContext")) {
            for (const auto& cat : categories) {
                ImGui::PushStyleColor(ImGuiCol_Text, cat.color);
                bool menuOpen = ImGui::BeginMenu(cat.name);
                ImGui::PopStyleColor();

                if (menuOpen) {
                    for (const auto& nodePair : cat.nodes) {
                        if (ImGui::MenuItem(nodePair.second)) {
                            // Spawn relative to view center
                            float spawnX = (-editor.scrollX + ImGui::GetWindowSize().x * 0.5f) / editor.zoom;
                            float spawnY = (-editor.scrollY + ImGui::GetWindowSize().y * 0.5f) / editor.zoom;
                            graph.addTerrainNode(nodePair.first, spawnX, spawnY);
                            ProjectManager::getInstance().markModified();
                        }
                    }
                    ImGui::EndMenu();
                }
            }
            ImGui::EndPopup();
        }
    }

    void drawNodeLibrary(TerrainNodeGraphV2& graph, TerrainObject* terrain) {
        ImGui::TextColored(ImVec4(0.7f, 0.9f, 0.7f, 1.0f), "Node Library");
        ImGui::Separator();
        
        // Search box
        ImGui::PushItemWidth(-1);
        ImGui::InputTextWithHint("##Search", "Search...", searchBuffer, sizeof(searchBuffer));
        ImGui::PopItemWidth();
        ImGui::Spacing();
        
        std::string searchStr = searchBuffer;
        std::transform(searchStr.begin(), searchStr.end(), searchStr.begin(), ::tolower);
        
        for (auto& category : categories) {
            ImGui::PushStyleColor(ImGuiCol_Header, category.color);
            bool categoryOpen = ImGui::CollapsingHeader(category.name, ImGuiTreeNodeFlags_DefaultOpen);
            ImGui::PopStyleColor();
            
            if (categoryOpen) {
                ImGui::Indent(8);
                for (auto& [type, name] : category.nodes) {
                    std::string nameLower = name;
                    std::transform(nameLower.begin(), nameLower.end(), nameLower.begin(), ::tolower);
                    
                    if (!searchStr.empty() && nameLower.find(searchStr) == std::string::npos) continue;
                    
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.2f, 0.25f, 1.0f));
                    
                    // Make button a drag source
                    ImGui::Button(name, ImVec2(-1, 0));
                    
                    // Begin drag source when button is clicked and dragged
                    if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)) {
                        // Store the node type being dragged
                        ImGui::SetDragDropPayload("TERRAIN_NODE_TYPE", &type, sizeof(NodeType));
                        
                        // Show preview tooltip
                        ImGui::Text("+ %s", name);
                        ImGui::EndDragDropSource();
                    }
                    
                    // Tooltip on hover
                    if (ImGui::IsItemHovered() && !ImGui::IsMouseDragging(0)) {
                        ImGui::SetTooltip("Drag to canvas");
                    }
                    
                    ImGui::PopStyleColor();
                }
                ImGui::Unindent(8);
            }
        }
    }
    
    template<typename ContextT>
    void drawToolbar(ContextT& ctx, TerrainNodeGraphV2& graph, TerrainObject* terrain) {
        if (ImGui::Button("Reset Graph")) {
            graph.createDefaultGraph(terrain);
            editor.scrollX = 0; 
            editor.scrollY = 0;
            editor.zoom = 1.0f;
        }
        ImGui::SameLine();
        
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.3f, 1.0f));
        if (ImGui::Button("Evaluate")) {
            // Mark all nodes dirty
            graph.markAllDirty();
            
            // Evaluate using V2 system
            graph.evaluateTerrain(terrain, ctx.scene);
            
            // Trigger GPU rebuild
            if (terrain) {
                ctx.renderer.resetCPUAccumulation();
                g_bvh_rebuild_pending = true;
                g_optix_rebuild_pending = true;
                if (ctx.optix_gpu_ptr) {
                    cudaDeviceSynchronize();
                    ctx.optix_gpu_ptr->resetAccumulation();
                }
            }
        }
        ImGui::PopStyleColor();
        
        ImGui::SameLine();
        ImGui::TextDisabled("|");
        ImGui::SameLine();
        ImGui::Text("Nodes: %d  Links: %d", (int)graph.nodeCount(), (int)graph.linkCount());
        ImGui::SameLine();
        ImGui::Text("Zoom: %.1f", editor.zoom);
        
        ImGui::Separator();
    }
};

} // namespace TerrainNodesV2
