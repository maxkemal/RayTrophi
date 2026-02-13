/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          scene_ui_material_nodeeditor.hpp
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once

/**
 * @file scene_ui_material_nodeeditor.hpp
 * @brief Material Node Editor UI
 * 
 * Adapts the generic NodeEditorUIV2 for MaterialNodes.
 */

#include "imgui.h"
#include "MaterialNodes.h"
#include "NodeSystem/NodeEditorUIV2.h"
#include "ProjectManager.h"
#include <functional>
#include <vector>
#include <string>

// External rebuild flags (from globals)
extern bool g_optix_rebuild_pending;

namespace MaterialNodes {

struct MatNodeCategory {
    const char* name;
    ImVec4 color;
    std::vector<std::pair<NodeType, const char*>> nodes;
};

class MaterialNodeEditorUI {
public:
    std::function<std::string(const wchar_t*)> onOpenFileDialog;
    
    // UI State
    float propertiesWidth = 250.0f;
    bool isResizingProperties = false;
    
    // Editor Instance
    NodeSystem::NodeEditorUIV2 editor;
    
    // Categories
    std::vector<MatNodeCategory> categories;
    
    MaterialNodeEditorUI() {
        // Setup categories
        categories = {
            {"Input", ImVec4(0.8f, 0.3f, 0.3f, 1.0f), {
                {NodeType::Value, "Value"},
                {NodeType::RGB, "RGB Input"},
                {NodeType::TextureCoordinate, "Texture Coordinate"}
            }},
            {"Texture", ImVec4(0.8f, 0.5f, 0.2f, 1.0f), {
                {NodeType::ImageTexture, "Image Texture"},
                {NodeType::NoiseTexture, "Noise Texture"},
                {NodeType::CheckerTexture, "Checker Texture"}
            }},
            {"Color", ImVec4(0.8f, 0.8f, 0.2f, 1.0f), {
                {NodeType::MixRGB, "Mix RGB"},
                {NodeType::Invert, "Separate RGB"},  // Uses Invert type
                {NodeType::Gamma, "Combine RGB"},    // Uses Gamma type
                {NodeType::ColorRamp, "Color Ramp"}
            }},
            {"Vector", ImVec4(0.4f, 0.4f, 0.8f, 1.0f), {
                {NodeType::Mapping, "Mapping"}
            }},
            {"Math", ImVec4(0.4f, 0.6f, 0.8f, 1.0f), {
                {NodeType::Math, "Math"}
            }},
            {"Output", ImVec4(0.2f, 0.2f, 0.2f, 1.0f), {
                {NodeType::MaterialOutput, "Material Output"}
            }}
        };
        
        // Config adjustments
        editor.config.gridColorMajor = IM_COL32(50, 50, 60, 255);
        editor.config.bgColor = IM_COL32(25, 25, 30, 255);
        
        editor.onGraphModified = []() {
             ProjectManager::getInstance().markModified();
        };
    }
    
    /**
     * @brief Reset editor state
     */
    void reset() {
        editor.reset();
    }
    
    void draw(MaterialNodeGraph& graph) {
        float libraryWidth = 160.0f;
        
        // 1. Library Panel
        ImGui::BeginChild("MatNodeLib", ImVec2(libraryWidth, 0), true);
        drawLibrary(graph);
        ImGui::EndChild();
        
        ImGui::SameLine();
        
        // 2. Canvas
        float availW = ImGui::GetContentRegionAvail().x;
        float canvasW = availW - propertiesWidth - 8;
        
        editor.draw(graph, ImVec2(canvasW, 0));
        
        // Handle Drops
        if (ImGui::BeginDragDropTarget()) {
            if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("MAT_NODE_TYPE")) {
                 NodeType type = *(const NodeType*)payload->Data;
                 
                 // Spawn at mouse
                 ImVec2 mouse = ImGui::GetMousePos();
                 ImVec2 canvasPos = ImGui::GetItemRectMin(); // approximate
                 // Better approach: use editor state (but direct access is consistent here)
                 float spawnX = (mouse.x - canvasPos.x - editor.scrollX) / editor.zoom;
                 float spawnY = (mouse.y - canvasPos.y - editor.scrollY) / editor.zoom;
                 
                 spawnNode(graph, type, spawnX, spawnY);
                 ProjectManager::getInstance().markModified();
            }
            ImGui::EndDragDropTarget();
        }
        
        ImGui::SameLine();
        
        // 3. Splitter
        ImGui::InvisibleButton("##Splitter", ImVec2(6, ImGui::GetContentRegionAvail().y));
        if (ImGui::IsItemActive()) {
            propertiesWidth -= ImGui::GetIO().MouseDelta.x;
            isResizingProperties = true;
        }
        if (!ImGui::IsMouseDown(0)) isResizingProperties = false;
        
        ImGui::SameLine();
        
        // 4. Properties
        ImGui::BeginChild("MatNodeProps", ImVec2(propertiesWidth, 0), true);
        drawProperties(graph);
        ImGui::EndChild();
    }
    
    void spawnNode(MaterialNodeGraph& graph, NodeType type, float x, float y) {
        MaterialNodeBase* node = nullptr;
        
        switch(type) {
            case NodeType::Value: node = graph.addNode<ValueNode>(); break;
            case NodeType::RGB: node = graph.addNode<RGBNode>(); break;
            case NodeType::TextureCoordinate: node = graph.addNode<TextureCoordinateNode>(); break;
            case NodeType::ImageTexture: node = graph.addNode<ImageTextureNode>(); break;
            case NodeType::NoiseTexture: node = graph.addNode<NoiseTextureNode>(); break;
            case NodeType::MixRGB: node = graph.addNode<MixRGBNode>(); break;
            case NodeType::Math: node = graph.addNode<MathNode>(); break;
            case NodeType::Invert: node = graph.addNode<SeparateRGBNode>(); break; // SeparateRGB uses Invert type
            case NodeType::Gamma: node = graph.addNode<CombineRGBNode>(); break;   // CombineRGB uses Gamma type
            case NodeType::MaterialOutput: node = graph.addNode<MaterialOutputNode>(); break;
            default: 
                // Generic fallback
                node = graph.addNode<MaterialNodeBase>();
                node->name = "Unknown";
                node->matNodeType = type;
                break;
        }
        
        if (node) {
            node->x = x;
            node->y = y;
        }
    }
    
    void drawLibrary(MaterialNodeGraph& graph) {
        ImGui::TextDisabled("Nodes");
        ImGui::Separator();
        
        for(auto& cat : categories) {
             ImGui::PushStyleColor(ImGuiCol_Header, cat.color);
             if (ImGui::CollapsingHeader(cat.name, ImGuiTreeNodeFlags_DefaultOpen)) {
                 for(auto& pair : cat.nodes) {
                      ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.2f, 0.25f, 1.0f));
                      ImGui::Button(pair.second, ImVec2(-1, 0));
                      
                      if (ImGui::BeginDragDropSource()) {
                           ImGui::SetDragDropPayload("MAT_NODE_TYPE", &pair.first, sizeof(NodeType));
                           ImGui::Text("+ %s", pair.second);
                           ImGui::EndDragDropSource();
                      }
                      ImGui::PopStyleColor();
                 }
             }
             ImGui::PopStyleColor();
        }
    }
    
    void drawProperties(MaterialNodeGraph& graph) {
        ImGui::TextDisabled("Properties");
        ImGui::Separator();
        
        if (editor.selectedNodeId == 0) {
            ImGui::Text("Select a node.");
            return;
        }
        
        auto* node = graph.getNodeAs<MaterialNodeBase>(editor.selectedNodeId);
        if (!node) return;
        
        ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "%s", node->name.c_str());
        ImGui::Spacing();
        
        // File Dialog Handler (Propagated from Node)
        if (auto* imgNode = dynamic_cast<ImageTextureNode*>(node)) {
            if (imgNode->browseForFile && onOpenFileDialog) {
                 imgNode->browseForFile = false;
                 std::string path = onOpenFileDialog(L"Image Files\0*.png;*.jpg;*.jpeg;*.bmp;*.tga\0");
                 if (!path.empty()) {
                     imgNode->filePath = path;
                     imgNode->loadFile();
                     imgNode->dirty = true;
                 }
            }
        }
        
        node->drawContent();
        
        ImGui::Spacing();
        ImGui::Separator();
        
        // Preview
        ImGui::Text("Preview");
        // TODO: Render small preview logic here
    }
};

} // namespace MaterialNodes

