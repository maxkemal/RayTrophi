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

    TerrainNodeEditorUI() {
        // Setup node categories
        categories = {
            {"Input", ImVec4(0.2f, 0.7f, 0.3f, 1.0f), {
                {NodeType::HeightmapInput, "Heightmap"},
                {NodeType::NoiseGenerator, "Noise Generator"}
            }},
            {"Erosion", ImVec4(0.3f, 0.5f, 0.9f, 1.0f), {
                {NodeType::HydraulicErosion, "Hydraulic"},
                {NodeType::ThermalErosion, "Thermal"},
                {NodeType::FluvialErosion, "Fluvial"},
                {NodeType::WindErosion, "Wind"}
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
        float propertiesWidth = 200.0f;
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
                    }
                    // If insertion failed, node is still created at position
                } else {
                    // Normal drop - just create node
                    graph.addTerrainNode(droppedType, spawnX, spawnY);
                }
            }
            ImGui::EndDragDropTarget();
        }
        
        // Render Context Menu
        drawContextMenu(graph);
        
        ImGui::EndChild();
        
        ImGui::SameLine();
        
        // Right Panel - Properties
        ImGui::BeginChild("NodeProperties", ImVec2(propertiesWidth, 0), true);
        drawPropertiesPanel(graph);
        ImGui::EndChild();
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
        node->drawContent();
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
    }
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
