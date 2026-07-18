/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          scene_ui_nodeeditor.hpp
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
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
#include "Backend/VulkanBackend.h" // For the partial BLAS/raster-mesh refit path (same-topology re-evaluate)
#include <SDL.h>
#include <functional> // Added for callback
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cctype>
#include <cstdio>
// #include <functional>
// #include <scene_ui.h> // Dependency removed // Dependency removed to fix circular include

extern bool g_bvh_rebuild_pending;
extern bool g_optix_rebuild_pending;
extern std::unique_ptr<Backend::IViewportBackend> g_viewport_backend;
extern std::unique_ptr<Backend::IBackend> g_backend;
extern SDL_Renderer* renderer;

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
    std::function<void(TerrainObject*, const std::vector<int>&)> onFoliageScattered;
    std::function<ImTextureID(const std::string&, int&, int&)> onFoliageThumbnail;
    TerrainObject* currentTerrain = nullptr;  // For export operations
    
    // Resizable properties panel width
    float propertiesWidth = 220.0f;
    float minPropertiesWidth = 150.0f;
    float maxPropertiesWidth = 400.0f;
    bool isResizingProperties = false;
    float libraryWidth = 160.0f;
    bool showLibrary = true;
    bool showProperties = true;
    
    // Mask preview cache and controls
    uint32_t lastPreviewedNodeId = 0;
    std::vector<float> lastPreviewedData;
    int lastPreviewedWidth = 0;
    int lastPreviewedHeight = 0;
    float lastPreviewMin = 0.0f;
    float lastPreviewMax = 1.0f;
    uint64_t maskPreviewRevision = 1;
    uint64_t lastPreviewedRevision = 0;
    std::vector<ImU32> previewDisplayColors;
    static constexpr int MASK_PREVIEW_TEXTURE_SIZE = 512;
    SDL_Texture* maskPreviewTexture = nullptr;
    std::vector<uint8_t> maskPreviewTexturePixels;
    float cachedPreviewZoom = -1.0f;
    float cachedPreviewPanX = 0.0f;
    float cachedPreviewPanY = 0.0f;
    bool propertyEditActiveThisFrame = false;
    float previewZoom = 1.0f;
    float previewPanX = 0.0f;
    float previewPanY = 0.0f;
    bool isPanningPreview = false;
    bool autoPreviewSelectedNode = true;
    uint32_t pendingAutoPreviewNodeId = 0;
    double pendingAutoPreviewSince = 0.0;
    std::string setupStatus;
    double setupStatusUntil = 0.0;
    bool requestSnowyMountainValleyPopup = false;

    TerrainNodeEditorUI() {
        // Setup node categories
        categories = {
            {"Input", ImVec4(0.2f, 0.7f, 0.3f, 1.0f), {
                {NodeType::HeightmapInput, "Heightmap"},
                {NodeType::NoiseGenerator, "Noise Generator"},
                {NodeType::HardnessInput, "Hardness Input"}
            }},
            {"Erosion", ImVec4(0.3f, 0.5f, 0.9f, 1.0f), {
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
                {NodeType::MaskAdjust, "Mask Adjust"},
                {NodeType::Remap, "Remap / Levels"},
                {NodeType::Terrace, "Terrace"},
                {NodeType::EdgeFalloff, "Edge Falloff"}
            }},
            {"Mask", ImVec4(0.7f, 0.4f, 0.8f, 1.0f), {
                {NodeType::HeightMask, "Height Mask"},
                {NodeType::SlopeMask, "Slope Mask"},
                {NodeType::CurvatureMask, "Curvature Mask"},
                {NodeType::FlowMask, "Flow / Soil"},
                {NodeType::ExposureMask, "Sun Exposure"},
                {NodeType::MaskCombine, "Mask Combine"},
                {NodeType::MaskMorphology, "Dilate / Erode / Blur"},
                {NodeType::MaskPaint, "Mask Paint"},
                {NodeType::MaskImage, "Mask Image"}
            }},
            {"Data Maps", ImVec4(0.25f, 0.55f, 0.72f, 1.0f), {
                {NodeType::TerrainAnalysis, "Terrain Analysis"},
                {NodeType::BiomeComposer, "Biome Composer"},
                {NodeType::WetnessMap, "Wetness Map"},
                {NodeType::SoilDepth, "Soil Depth"}
            }},
            {"Foliage", ImVec4(0.24f, 0.56f, 0.29f, 1.0f), {
                {NodeType::FoliageLayer, "Foliage Layer"},
                {NodeType::FoliageSet, "Foliage Set / Biome"}
            }},
            {"Hydrology", ImVec4(0.16f, 0.59f, 0.80f, 1.0f), {
                {NodeType::WatershedAnalysis, "Watershed Analysis"},
                {NodeType::LakeBasin, "Lake Basin"},
                {NodeType::RiverNetwork, "River Network"},
                {NodeType::RiverHydraulics, "River Hydraulics"},
                {NodeType::RiverBedCarve, "River Bed Carve"}
            }},
            {"Snow & Ice", ImVec4(0.42f, 0.72f, 0.88f, 1.0f), {
                {NodeType::SnowClimate, "Snow Layer (Easy)"},
                {NodeType::Climate, "Climate"},
                {NodeType::Snowfall, "Snowfall"},
                {NodeType::SnowSettle, "Snow Settle / Avalanche"},
                {NodeType::SnowMeltFreeze, "Snow Melt / Freeze"},
                {NodeType::GlacierFlow, "Glacier Flow"}
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
                {NodeType::SplatOutput, "Splat Output"},
                {NodeType::HardnessOutput, "Hardness Output"},
                {NodeType::TerrainFieldsOutput, "Terrain Fields Output"},
                {NodeType::FoliageOutput, "Foliage Output"},
                {NodeType::LakeSurfaceOutput, "Lake Surface Output"},
                {NodeType::RiverSplineOutput, "River Spline Output"}
            }},
            {"Texture", ImVec4(0.8f, 0.6f, 0.2f, 1.0f), {
                {NodeType::AutoSplat, "Auto Splat"},
                {NodeType::SplatCompose, "Splat Compose"},
                {NodeType::SurfaceComposer, "Surface Composer"}
            }},
            {"Utility", ImVec4(0.35f, 0.62f, 0.72f, 1.0f), {
                {NodeType::Resample, "Resample"},
                {NodeType::ChannelExtract, "Channel Extract"}
            }},
            {"Geology", ImVec4(0.7f, 0.45f, 0.35f, 1.0f), {
                {NodeType::Fault, "Fault Line"},
                {NodeType::Mesa, "Mesa / Plateau"},
                {NodeType::Shear, "Shear Zone"},
                {NodeType::Lithology, "Lithology"},
                {NodeType::Strata, "Strata"}
            }}
        };
        
        // Initialize V2 Editor
        editor.config.gridSizeMinor = 32.0f;
        editor.config.showMinimap = true;
        
        // Context Menu Callback (Fallback)
        editor.onBackgroundContextMenu = []() {
            // Unused as we unified popups, but kept for compatibility
        };
        
        // Node selected callback
        editor.onNodeSelected = [this](uint32_t nodeId) {
            pendingAutoPreviewNodeId = nodeId;
            pendingAutoPreviewSince = ImGui::GetTime();
        };
        
        // Graph modification callback (Mark Project Dirty)
        editor.onGraphModified = []() {
            ProjectManager::getInstance().markModified();
        };
    }

    /**
     * @brief Reset editor and UI state
     */
    void reset() {
        releaseMaskPreviewTexture();
        lastPreviewedNodeId = 0;
        lastPreviewedData.clear();
        lastPreviewedWidth = 0;
        lastPreviewedHeight = 0;
        lastPreviewMin = 0.0f;
        lastPreviewMax = 1.0f;
        maskPreviewRevision = 1;
        lastPreviewedRevision = 0;
        previewDisplayColors.clear();
        cachedPreviewZoom = -1.0f;
        propertyEditActiveThisFrame = false;
        previewZoom = 1.0f;
        previewPanX = 0.0f;
        previewPanY = 0.0f;
        isPanningPreview = false;
        pendingAutoPreviewNodeId = 0;
        pendingAutoPreviewSince = 0.0;
        setupStatus.clear();
        setupStatusUntil = 0.0;
        requestSnowyMountainValleyPopup = false;
        
        editor.reset();
    }

    void releaseMaskPreviewTexture() {
        if (maskPreviewTexture) {
            SDL_DestroyTexture(maskPreviewTexture);
            maskPreviewTexture = nullptr;
        }
        maskPreviewTexturePixels.clear();
    }

    bool updateMaskPreviewTexture() {
        if (!renderer || lastPreviewedData.empty() ||
            lastPreviewedWidth < 1 || lastPreviewedHeight < 1) {
            releaseMaskPreviewTexture();
            return false;
        }

        if (!maskPreviewTexture) {
            maskPreviewTexture = SDL_CreateTexture(
                renderer, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING,
                MASK_PREVIEW_TEXTURE_SIZE, MASK_PREVIEW_TEXTURE_SIZE);
            if (!maskPreviewTexture) return false;
            SDL_SetTextureBlendMode(maskPreviewTexture, SDL_BLENDMODE_NONE);
#if SDL_VERSION_ATLEAST(2, 0, 12)
            SDL_SetTextureScaleMode(maskPreviewTexture, SDL_ScaleModeLinear);
#endif
        }

        constexpr int dstSize = MASK_PREVIEW_TEXTURE_SIZE;
        maskPreviewTexturePixels.resize(static_cast<size_t>(dstSize) * dstSize * 4u);
        const float range = std::max(1.0e-6f, lastPreviewMax - lastPreviewMin);
        for (int y = 0; y < dstSize; ++y) {
            const float fy = (static_cast<float>(y) + 0.5f) / dstSize * lastPreviewedHeight - 0.5f;
            const int y0 = std::clamp(static_cast<int>(std::floor(fy)), 0, lastPreviewedHeight - 1);
            const int y1 = std::min(y0 + 1, lastPreviewedHeight - 1);
            const float ty = std::clamp(fy - std::floor(fy), 0.0f, 1.0f);
            for (int x = 0; x < dstSize; ++x) {
                const float fx = (static_cast<float>(x) + 0.5f) / dstSize * lastPreviewedWidth - 0.5f;
                const int x0 = std::clamp(static_cast<int>(std::floor(fx)), 0, lastPreviewedWidth - 1);
                const int x1 = std::min(x0 + 1, lastPreviewedWidth - 1);
                const float tx = std::clamp(fx - std::floor(fx), 0.0f, 1.0f);
                const float v00 = lastPreviewedData[static_cast<size_t>(y0) * lastPreviewedWidth + x0];
                const float v10 = lastPreviewedData[static_cast<size_t>(y0) * lastPreviewedWidth + x1];
                const float v01 = lastPreviewedData[static_cast<size_t>(y1) * lastPreviewedWidth + x0];
                const float v11 = lastPreviewedData[static_cast<size_t>(y1) * lastPreviewedWidth + x1];
                const float value = (v00 * (1.0f - tx) + v10 * tx) * (1.0f - ty) +
                                    (v01 * (1.0f - tx) + v11 * tx) * ty;
                const uint8_t gray = static_cast<uint8_t>(
                    std::clamp((value - lastPreviewMin) / range, 0.0f, 1.0f) * 255.0f + 0.5f);
                const size_t pixel = (static_cast<size_t>(y) * dstSize + x) * 4u;
                maskPreviewTexturePixels[pixel + 0] = gray;
                maskPreviewTexturePixels[pixel + 1] = gray;
                maskPreviewTexturePixels[pixel + 2] = gray;
                maskPreviewTexturePixels[pixel + 3] = 255;
            }
        }

        if (SDL_UpdateTexture(maskPreviewTexture, nullptr, maskPreviewTexturePixels.data(), dstSize * 4) != 0) {
            releaseMaskPreviewTexture();
            return false;
        }
        return true;
    }
    
    // Templated Context to avoid circular dependency with scene_ui.h
    template<typename ContextT>
    void draw(ContextT& ctx, TerrainNodeGraphV2& graph, TerrainObject* terrain) {
        if (!terrain) {
            ImGui::TextColored(ImVec4(1, 0.5f, 0.3f, 1), "Select a terrain in Terrain tab first");
            return;
        }

        // Dynamically bind background custom menu choices (requires active graph parameter)
        editor.onDrawBackgroundMenu = [this, &graph]() {
            for (const auto& cat : categories) {
                ImGui::PushStyleColor(ImGuiCol_Text, cat.color);
                bool menuOpen = ImGui::BeginMenu(cat.name);
                ImGui::PopStyleColor();

                if (menuOpen) {
                    for (const auto& nodePair : cat.nodes) {
                        if (ImGui::MenuItem(nodePair.second)) {
                            ImVec2 mousePos = ImGui::GetMousePos();
                            float spawnX = (mousePos.x - editor.canvasPos_.x - editor.scrollX) / editor.zoom;
                            float spawnY = (mousePos.y - editor.canvasPos_.y - editor.scrollY) / editor.zoom;
                            NodeSystem::NodeBase* node = graph.addTerrainNode(nodePair.first, spawnX, spawnY);
                            addNodeToActiveLayer(graph, node);
                            ProjectManager::getInstance().markModified();
                        }
                    }
                    ImGui::EndMenu();
                }
            }
        };

        // A link released over empty canvas opens a compact, searchable list of
        // only pin-compatible terrain nodes. The callback is rebound per draw
        // because the active graph can change with the selected terrain.
        editor.onDrawLinkCreateMenu = [this, &graph]() {
            drawLinkCreateMenu(graph);
        };
        
        // Update current terrain reference for export operations
        currentTerrain = terrain;
        
        auto verticalSplitter = [](const char* id, float& width, bool leftPanel,
                                   float minWidth, float maxWidth) {
            ImGui::SameLine(0.0f, 0.0f);
            ImGui::InvisibleButton(id, ImVec2(6.0f, ImGui::GetContentRegionAvail().y));
            if (ImGui::IsItemActive()) {
                width += leftPanel ? ImGui::GetIO().MouseDelta.x : -ImGui::GetIO().MouseDelta.x;
                width = std::clamp(width, minWidth, maxWidth);
            }
            if (ImGui::IsItemHovered() || ImGui::IsItemActive()) {
                ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
            }
            const ImU32 color = (ImGui::IsItemHovered() || ImGui::IsItemActive())
                ? IM_COL32(100, 150, 255, 200) : IM_COL32(80, 80, 90, 120);
            ImGui::GetWindowDrawList()->AddRectFilled(
                ImGui::GetItemRectMin(), ImGui::GetItemRectMax(), color);
        };

        if (showLibrary) {
            ImGui::BeginChild("NodeLibrary", ImVec2(libraryWidth, 0), true);
            drawNodeLibrary(graph, terrain);
            ImGui::EndChild();
            verticalSplitter("##TerrainLibrarySplitter", libraryWidth, true, 110.0f, 360.0f);
        }

        if (showLibrary) ImGui::SameLine(0.0f, 0.0f);
        drawPanelToggleStrip("##TerrainLibraryToggle", showLibrary, true,
                             "node library");
        ImGui::SameLine(0.0f, 0.0f);

        // Canvas takes all remaining width except the optional inspector and its edge strip.
        const float availWidth = ImGui::GetContentRegionAvail().x;
        const float rightReserve = 13.0f + (showProperties ? propertiesWidth + 6.0f : 0.0f);
        const float canvasWidth = std::max(200.0f, availWidth - rightReserve);
        
        ImGui::BeginChild("NodeCanvas", ImVec2(canvasWidth, 0), true, ImGuiWindowFlags_NoScrollbar);
        
        drawToolbar(ctx, graph, terrain);
        
        // Get canvas position before drawing (for drop coordinate calculation)
        ImVec2 canvasPos = ImGui::GetCursorScreenPos();
        ImVec2 canvasSize = ImGui::GetContentRegionAvail();

        // The background worker thread (see TerrainNodeGraphV2::evaluateTerrainAsync)
        // reads `nodes`/pin connections while computing — structural edits (add/
        // remove/connect node) or parameter edits from the UI thread during that
        // window would race it. Block the mutating entry points below (and grey
        // out canvas interaction) while an evaluation is in flight; viewing/panning
        // still works via editor.draw() itself.
        const bool graphEvaluating = graph.isEvaluatingAsync();

        // Render V2 Node Graph
        ImGui::BeginDisabled(graphEvaluating);
        editor.draw(graph, canvasSize);
        ImGui::EndDisabled();

        if (graphEvaluating) {
            ImVec2 overlayPos = canvasPos;
            ImGui::SetCursorScreenPos(ImVec2(overlayPos.x + 8.0f, overlayPos.y + 8.0f));
            ImGui::TextColored(ImVec4(0.5f, 0.85f, 1.0f, 1.0f), "Evaluating... graph is read-only until it finishes");
        }

        // Drop target for node library drag-drop
        // Make the whole canvas area accept drops
        if (!graphEvaluating && ImGui::BeginDragDropTarget()) {
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
                    addNodeToActiveLayer(graph, newNode);

                    if (newNode && tryInsertNodeIntoLink(graph, newNode, targetLink)) {
                        // Successfully inserted!
                        ProjectManager::getInstance().markModified();
                    }
                    // If insertion failed, node is still created at position
                } else {
                    // Normal drop - just create node
                    NodeSystem::NodeBase* node = graph.addTerrainNode(droppedType, spawnX, spawnY);
                    addNodeToActiveLayer(graph, node);
                    ProjectManager::getInstance().markModified();
                }
            }
            ImGui::EndDragDropTarget();
        }

        // Render Context Menu (structural edits — disabled during evaluation)
        if (!graphEvaluating) {
            drawContextMenu(graph);
        }

        ImGui::EndChild();
        ImGui::SameLine(0.0f, 0.0f);
        drawPanelToggleStrip("##TerrainPropertiesToggle", showProperties, false,
                             "node properties");

        if (showProperties) {
            verticalSplitter("##TerrainPropertiesSplitter", propertiesWidth, false,
                             minPropertiesWidth, maxPropertiesWidth);
            ImGui::SameLine(0.0f, 0.0f);
            ImGui::BeginChild("NodeProperties", ImVec2(0, 0), true);
            if (graphEvaluating) {
                // Parameter UI remains dormant while the worker owns node state.
                ImGui::TextColored(ImVec4(0.5f, 0.85f, 1.0f, 1.0f), "Evaluating terrain...");
                ImGui::TextDisabled("Node parameters and mask preview are paused.");
            } else {
                drawPropertiesPanel(graph);
            }
            ImGui::EndChild();
        }
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

        // Node Info Header (with text wrapping to prevent clipping)
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 0.8f, 1.0f, 1.0f));
        ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x);
        ImGui::TextWrapped("%s", node->metadata.displayName.empty() ? node->name.c_str() : node->metadata.displayName.c_str());
        ImGui::PopTextWrapPos();
        ImGui::PopStyleColor();

        // Category (wrapped to prevent overflow)
        if (!node->metadata.category.empty()) {
            ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x);
            ImGui::TextDisabled("Category: %s", node->metadata.category.c_str());
            ImGui::PopTextWrapPos();
        }
        ImGui::TextDisabled("ID: %u", node->id);

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // Parameters header
        ImGui::TextColored(ImVec4(0.9f, 0.9f, 0.6f, 1.0f), "Parameters");
        ImGui::Separator();
        ImGui::Spacing();

        // ========================================================================
        // USER-FRIENDLY SLIDER/INPUT STYLING & WIDTH MANAGEMENT
        // ========================================================================
        // Problem 1: When panel is wide, sliders become too long
        // Problem 2: Input fields blend into panel background, hard to distinguish
        // Solution: Limit slider width + Apply distinct visual styling
        
        float availWidth = ImGui::GetContentRegionAvail().x;
        
        // Slider/input width: Capped at 140px for usability, minimum 80px
        float maxSliderWidth = 140.0f;
        float minSliderWidth = 80.0f;
        float sliderWidth = std::clamp(availWidth * 0.6f, minSliderWidth, maxSliderWidth);
        
        ImGui::PushItemWidth(sliderWidth);
        
        // ── DISTINCT INPUT FIELD STYLING ──────────────────────────────────────
        // Make sliders/inputs visually distinct from panel (darker, with border)
        ImVec4 nodeFrameBg = ImVec4(0.12f, 0.12f, 0.15f, 1.0f);
        ImVec4 nodeFrameBgHovered = ImVec4(0.18f, 0.20f, 0.25f, 1.0f);
        ImVec4 nodeFrameBgActive = ImVec4(0.22f, 0.25f, 0.30f, 1.0f);
        float nodeFrameRounding = 3.0f;
        
        ImVec4 nodeSliderGrab = ImVec4(0.45f, 0.55f, 0.75f, 1.0f);
        ImVec4 nodeSliderGrabActive = ImVec4(0.55f, 0.65f, 0.85f, 1.0f);
        if (ThemeManager::instance().getIconSettings().overridePanelAccentsWithTheme) {
            const auto& curTheme = ThemeManager::instance().current();
            nodeSliderGrab = ImVec4(curTheme.colors.accent.x, curTheme.colors.accent.y, curTheme.colors.accent.z, 0.92f);
            nodeSliderGrabActive = ImVec4((std::min)(1.0f, curTheme.colors.accent.x + 0.10f), (std::min)(1.0f, curTheme.colors.accent.y + 0.10f), (std::min)(1.0f, curTheme.colors.accent.z + 0.10f), 1.0f);
            
            nodeFrameBg = curTheme.colors.surface;
            nodeFrameBgHovered = UIWidgets::ScaleColor(curTheme.colors.surface, 1.3f);
            nodeFrameBgActive = UIWidgets::ScaleColor(curTheme.colors.surface, 1.5f);
            nodeFrameRounding = curTheme.style.frameRounding;
        }
        ImGui::PushStyleColor(ImGuiCol_FrameBg, nodeFrameBg);
        ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, nodeFrameBgHovered);
        ImGui::PushStyleColor(ImGuiCol_FrameBgActive, nodeFrameBgActive);
        ImGui::PushStyleColor(ImGuiCol_SliderGrab, nodeSliderGrab);
        ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, nodeSliderGrabActive);
        ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.35f, 0.38f, 0.45f, 0.8f));         // Subtle border
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, nodeFrameRounding);      // Rounded corners
        ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.0f);    // Visible border
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(6, 3)); // Compact padding
        
        // `dirty` is a level, not an edit event. Temporarily clear it around the
        // UI draw so repeated edits are detectable even when this node was
        // already invalidated by an earlier change or an upstream node.
        const bool wasDirty = node->dirty;
        node->dirty = false;
        if (auto* terrainNode = dynamic_cast<TerrainNodeBase*>(node);
            terrainNode && isPublicationSink(*terrainNode)) {
            if (ImGui::Checkbox("Publish Output", &terrainNode->publicationEnabled)) {
                terrainNode->dirty = true;
            }
            ImGui::TextDisabled(terrainNode->publicationEnabled
                ? "This sink participates in Evaluate"
                : "Disabled: upstream branch is not pulled by this sink");
            ImGui::Separator();
        }
        auto* foliageLayer = dynamic_cast<FoliageLayerNode*>(node);
        if (foliageLayer) foliageLayer->propertyThumbnailProvider = onFoliageThumbnail;
        node->drawContent();
        if (foliageLayer) foliageLayer->propertyThumbnailProvider = {};
        const bool parameterChanged = node->dirty;
        node->dirty = wasDirty || parameterChanged;
        // IsAnyItemActive() is application-global.  Without scoping it to this
        // properties window, editing an unrelated material/layer slider keeps
        // re-arming the terrain auto-preview debounce while this node is dirty.
        const bool editingTerrainNodeProperty =
            ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows) &&
            ImGui::IsAnyItemActive();
        propertyEditActiveThisFrame = editingTerrainNodeProperty;
        if (parameterChanged) {
            // Propagate every edit event so cached downstream values cannot
            // survive a second/third slider or preset change.
            graph.markDirtyDownstream(node->id);
            ProjectManager::getInstance().markModified();
            pendingAutoPreviewNodeId = node->id;
            pendingAutoPreviewSince = ImGui::GetTime();
            ++maskPreviewRevision;
        }
        
        // Pop all styling
        ImGui::PopStyleVar(3);
        ImGui::PopStyleColor(6);
        ImGui::PopItemWidth();

        // Handle HeightmapInputNode file dialog request
        auto* heightmapNode = dynamic_cast<TerrainNodesV2::HeightmapInputNode*>(node);
        if (heightmapNode && heightmapNode->browseForHeightmap) {
            heightmapNode->browseForHeightmap = false;

            if (onOpenFileDialog) {
                std::string path = onOpenFileDialog(L"Heightmap Files\0*.png;*.jpg;*.jpeg;*.bmp;*.raw;*.r16\0");
                if (!path.empty()) {
                    strncpy(heightmapNode->filePath, path.c_str(), sizeof(heightmapNode->filePath) - 1);
                    heightmapNode->filePath[sizeof(heightmapNode->filePath) - 1] = '\0';
                    heightmapNode->loadHeightmapFromFile();
                    graph.markDirtyDownstream(heightmapNode->id);
                    ProjectManager::getInstance().markModified();
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
                    graph.markDirtyDownstream(maskImageNode->id);
                    ProjectManager::getInstance().markModified();
                    pendingAutoPreviewNodeId = maskImageNode->id;
                    pendingAutoPreviewSince = ImGui::GetTime();
                    ++maskPreviewRevision;
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
            
            const bool revisionChanged = lastPreviewedRevision != maskPreviewRevision;
            const bool debounceElapsed =
                (ImGui::GetTime() - pendingAutoPreviewSince) >= 0.15;
            bool needsUpdate = nodeChanged || refreshClicked ||
                               (revisionChanged && debounceElapsed && !propertyEditActiveThisFrame);
            
            if (needsUpdate && currentTerrain && !graph.isEvaluatingAsync()) {
                NodeSystem::Image2DData image;
                bool imageAvailable = graph.getCachedImageOutput(node->id, maskOutputIndex, image);
                if (!imageAvailable && refreshClicked) {
                    // An explicit refresh may need to pull the upstream DAG, but
                    // erosion nodes mutate their TerrainObject while computing.
                    // Run that pull on a detached data-only terrain so inspector
                    // work can never dirty the live mesh or wake BVH rebuilds.
                    TerrainObject scratchTerrain;
                    scratchTerrain.id = currentTerrain->id;
                    scratchTerrain.name = currentTerrain->name + "##MaskPreview";
                    scratchTerrain.heightmap = currentTerrain->heightmap;
                    scratchTerrain.original_heightmap_data = currentTerrain->original_heightmap_data;
                    scratchTerrain.hardnessMap = currentTerrain->hardnessMap;
                    scratchTerrain.flowMap = currentTerrain->flowMap;
                    scratchTerrain.erosionMapRGBA = currentTerrain->erosionMapRGBA;
                    scratchTerrain.defer_mesh_updates = true;

                    TerrainNodesV2::TerrainContext tctx(&scratchTerrain);
                    tctx.publishTerrainState = false;
                    NodeSystem::EvaluationContext ctx(&graph);
                    ctx.setDomainContext(&tctx);

                    // requestOutput() updates NodeBase::dirty even with a local
                    // EvaluationContext. Preserve graph authoring state so this
                    // inspector pull cannot make the next real Evaluate reuse a
                    // stale live-terrain result.
                    std::vector<std::pair<NodeSystem::NodeBase*, bool>> dirtySnapshot;
                    dirtySnapshot.reserve(graph.nodes.size());
                    for (const auto& graphNode : graph.nodes) {
                        dirtySnapshot.emplace_back(graphNode.get(), graphNode->dirty);
                    }
                    NodeSystem::PinValue result = node->requestOutput(maskOutputIndex, ctx);
                    for (const auto& [graphNode, wasDirty] : dirtySnapshot) {
                        graphNode->dirty = wasDirty;
                    }
                    if (const auto* computed = std::get_if<NodeSystem::Image2DData>(&result)) {
                        image = *computed;
                        imageAvailable = image.isValid();
                    }
                }

                if (imageAvailable && image.isValid()) {
                    const auto* imgData = &image;
                    const size_t pixels = imgData->pixelCount();
                    lastPreviewedData.resize(pixels);
                    if (imgData->channels == 1) {
                        lastPreviewedData = *(imgData->data);
                    } else {
                        // Multi-channel masks (Auto Splat / packed erosion maps)
                        // preview channel 0. Use Channel Extract for explicit
                        // R/G/B/A authoring and inspection.
                        for (size_t pixel = 0; pixel < pixels; ++pixel) {
                            lastPreviewedData[pixel] = (*imgData->data)[pixel * imgData->channels];
                        }
                    }
                    lastPreviewedNodeId = node->id;
                    lastPreviewedWidth = imgData->width;
                    lastPreviewedHeight = imgData->height;
                    lastPreviewedRevision = maskPreviewRevision;
                    auto minMax = std::minmax_element(lastPreviewedData.begin(), lastPreviewedData.end());
                    lastPreviewMin = *minMax.first;
                    lastPreviewMax = *minMax.second;
                    updateMaskPreviewTexture();
                    previewDisplayColors.clear();
                    cachedPreviewZoom = -1.0f;
                    // Reset pan when switching nodes
                    if (nodeChanged) {
                        previewPanX = 0.0f;
                        previewPanY = 0.0f;
                        previewZoom = 1.0f;
                    }
                } else {
                    // A slider drag invalidates the node before the debounced
                    // material-only evaluation publishes its new cache. Keep
                    // the last valid image and leave the revision pending; the
                    // next frame after Apply Material will refresh it in place.
                    const bool hasPreviousImageForNode =
                        lastPreviewedNodeId == node->id &&
                        lastPreviewedWidth > 0 && !lastPreviewedData.empty();
                    if (!hasPreviousImageForNode || nodeChanged || refreshClicked) {
                        lastPreviewedNodeId = node->id;
                        lastPreviewedWidth = 0;
                        lastPreviewedHeight = 0;
                        lastPreviewedRevision = maskPreviewRevision;
                        lastPreviewedData.clear();
                        releaseMaskPreviewTexture();
                    }
                }
            }
        
        // Draw preview if we have data
        if (!lastPreviewedData.empty() && lastPreviewedNodeId == node->id && lastPreviewedWidth > 0) {
            float previewSize = ImGui::GetContentRegionAvail().x;
            
            int srcW = lastPreviewedWidth;
            int srcH = lastPreviewedHeight;
            
            // Range is computed only when preview data changes, not every frame.
            float minVal = lastPreviewMin;
            float maxVal = lastPreviewMax;
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
            
            // The inspector is diagnostic, not the authored mask storage. Keep
            // its persistent ImGui geometry bounded; full-resolution values are
            // still retained and bilinearly sampled while zooming.
            int displayRes = std::min(64, std::max(srcW, srcH));
            float cellSize = previewSize / displayRes;
            
            // Calculate visible area based on zoom and pan
            float viewSize = 1.0f / previewZoom;
            float viewX = previewPanX - viewSize * 0.5f + 0.5f;
            float viewY = previewPanY - viewSize * 0.5f + 0.5f;

            if (maskPreviewTexture) {
                const ImVec2 uv0(std::clamp(viewX, 0.0f, 1.0f),
                                 std::clamp(viewY, 0.0f, 1.0f));
                const ImVec2 uv1(std::clamp(viewX + viewSize, 0.0f, 1.0f),
                                 std::clamp(viewY + viewSize, 0.0f, 1.0f));
                dl->AddImage(reinterpret_cast<ImTextureID>(maskPreviewTexture), p,
                             ImVec2(p.x + previewRect.x, p.y + previewRect.y), uv0, uv1);
            } else {
            // Bounded fallback for renderer teardown/unsupported texture paths.
            const bool rebuildDisplayCache =
                previewDisplayColors.size() != static_cast<size_t>(displayRes * displayRes) ||
                cachedPreviewZoom != previewZoom || cachedPreviewPanX != previewPanX ||
                cachedPreviewPanY != previewPanY;
            if (rebuildDisplayCache) {
                previewDisplayColors.resize(static_cast<size_t>(displayRes * displayRes));
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
                        previewDisplayColors[static_cast<size_t>(py * displayRes + px)] =
                            IM_COL32(gray, gray, gray, 255);
                    }
                }
                cachedPreviewZoom = previewZoom;
                cachedPreviewPanX = previewPanX;
                cachedPreviewPanY = previewPanY;
            }

            for (int py = 0; py < displayRes; py++) {
                for (int px = 0; px < displayRes; px++) {
                    ImVec2 cellMin(p.x + px * cellSize, p.y + py * cellSize);
                    ImVec2 cellMax(cellMin.x + cellSize, cellMin.y + cellSize);
                    dl->AddRectFilled(cellMin, cellMax,
                        previewDisplayColors[static_cast<size_t>(py * displayRes + px)]);
                }
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
            ImGui::TextDisabled("Preview: %dx%d GPU texture", MASK_PREVIEW_TEXTURE_SIZE,
                                MASK_PREVIEW_TEXTURE_SIZE);
            ImGui::TextDisabled("Range: %.3f - %.3f", minVal, maxVal);
        } else if (lastPreviewedNodeId == node->id && lastPreviewedWidth == 0) {
            ImGui::TextDisabled("No evaluated mask cache.");
            ImGui::TextDisabled("Press Refresh to compute an isolated preview.");
        } else {
            ImGui::TextDisabled("No preview available");
        }
    } // End CollapsingHeader
    } // End drawMaskPreview
    // Access graph for external use  
    TerrainNodeGraphV2& getGraph() { return ownedGraph; }
    
private:
    std::vector<NodeCategory> categories;
    char searchBuffer[128] = "";
    char linkCreateSearchBuffer[128] = "";
    char newLayerName[64] = "River";
    std::vector<NodeType> compatibleLinkNodeTypes;
    
    // Drag-drop state
    NodeType pendingDragNodeType = NodeType::HeightmapInput;
    bool isDragging = false;
    
    // V2 Editor Instance
    NodeSystem::NodeEditorUIV2 editor;
    
    // Owned graph instance (if not passed externally)
    TerrainNodeGraphV2 ownedGraph;

    static bool isPublicationSink(const TerrainNodeBase& node) {
        const std::string type = node.getTypeId();
        return type == "TerrainV2.SplatOutput" ||
               type == "TerrainV2.HardnessOutput" ||
               type == "TerrainV2.TerrainFieldsOutput" ||
               type == "TerrainV2.FoliageOutput" ||
               type == "TerrainV2.LakeBasin" ||
               type == "TerrainV2.LakeSurfaceOutput" ||
               type == "TerrainV2.RiverSplineOutput";
    }

    void addNodeToActiveLayer(TerrainNodeGraphV2& graph, NodeSystem::NodeBase* node) {
        if (!node || editor.focusedGroupId == 0 || !graph.getGroup(editor.focusedGroupId)) return;
        graph.addNodeToGroup(node->id, editor.focusedGroupId);
    }

    void drawPanelToggleStrip(const char* id, bool& visible, bool leftPanel,
                              const char* panelName) {
        constexpr float stripWidth = 13.0f;
        ImGui::BeginChild(id, ImVec2(stripWidth, 0), false, ImGuiWindowFlags_NoScrollbar);
        const ImVec2 p = ImGui::GetCursorScreenPos();
        const ImVec2 size = ImGui::GetContentRegionAvail();
        ImGui::InvisibleButton("##toggle", ImVec2(stripWidth, std::max(24.0f, size.y)));
        const bool hovered = ImGui::IsItemHovered();
        if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) visible = !visible;
        if (hovered) {
            ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
            ImGui::SetTooltip("%s %s", visible ? "Hide" : "Show", panelName);
        }
        ImDrawList* drawList = ImGui::GetWindowDrawList();
        if (hovered) {
            drawList->AddRectFilled(p, ImVec2(p.x + stripWidth, p.y + size.y),
                                    IM_COL32(90, 130, 200, 60));
        }
        const float cx = p.x + stripWidth * 0.5f;
        const float cy = p.y + size.y * 0.5f;
        constexpr float arrow = 3.5f;
        const ImU32 color = hovered ? IM_COL32(220, 230, 245, 255)
                                    : IM_COL32(130, 135, 145, 255);
        // Arrow points toward the space the panel will move into when clicked.
        const bool pointLeft = leftPanel ? visible : !visible;
        if (pointLeft) {
            drawList->AddTriangleFilled(ImVec2(cx + arrow * 0.7f, cy - arrow),
                ImVec2(cx + arrow * 0.7f, cy + arrow), ImVec2(cx - arrow * 0.9f, cy), color);
        } else {
            drawList->AddTriangleFilled(ImVec2(cx - arrow * 0.7f, cy - arrow),
                ImVec2(cx - arrow * 0.7f, cy + arrow), ImVec2(cx + arrow * 0.9f, cy), color);
        }
        ImGui::EndChild();
    }

    void activateLayer(TerrainNodeGraphV2& graph, uint32_t groupId) {
        editor.focusedGroupId = groupId;
        editor.selectedNodeId = 0;
        editor.selectedNodeIds.clear();
        editor.selectedLinkId = 0;
        if (groupId == 0) return;
        NodeSystem::NodeGroup* group = graph.getGroup(groupId);
        if (!group || editor.canvasSize_.x <= 1.0f || editor.canvasSize_.y <= 1.0f) return;

        // Reserve horizontal room for the virtual Layer Input/Output contracts.
        const float fitWidth = group->size.x + 320.0f;
        const float fitHeight = group->size.y + 100.0f;
        editor.zoom = std::clamp(std::min(editor.canvasSize_.x / std::max(1.0f, fitWidth),
                                          editor.canvasSize_.y / std::max(1.0f, fitHeight)),
                                 0.35f, 1.15f);
        const float centerX = group->position.x + group->size.x * 0.5f;
        const float centerY = group->position.y + group->size.y * 0.5f;
        editor.scrollX = editor.canvasSize_.x * 0.5f - centerX * editor.zoom;
        editor.scrollY = editor.canvasSize_.y * 0.5f - centerY * editor.zoom;
    }

    void frameAllNodes(TerrainNodeGraphV2& graph) {
        editor.focusedGroupId = 0;
        if (graph.nodes.empty() || editor.canvasSize_.x <= 1.0f || editor.canvasSize_.y <= 1.0f) return;
        float minX = FLT_MAX, minY = FLT_MAX, maxX = -FLT_MAX, maxY = -FLT_MAX;
        for (const auto& node : graph.nodes) {
            int inputs = 0, outputs = 0;
            for (const auto& pin : node->inputs) if (!pin.hidden) ++inputs;
            for (const auto& pin : node->outputs) if (!pin.hidden) ++outputs;
            const float custom = node->getCustomWidth();
            const float width = node->uiWidth > 0.0f ? node->uiWidth
                : (custom > 0.0f ? custom : 180.0f);
            const float height = node->collapsed ? 34.0f
                : 54.0f + 22.0f * static_cast<float>(std::max(inputs, outputs));
            minX = std::min(minX, node->x); minY = std::min(minY, node->y);
            maxX = std::max(maxX, node->x + width); maxY = std::max(maxY, node->y + height);
        }
        const float width = std::max(1.0f, maxX - minX + 140.0f);
        const float height = std::max(1.0f, maxY - minY + 140.0f);
        editor.zoom = std::clamp(std::min(editor.canvasSize_.x / width,
                                          editor.canvasSize_.y / height), 0.20f, 1.10f);
        const float centerX = (minX + maxX) * 0.5f;
        const float centerY = (minY + maxY) * 0.5f;
        editor.scrollX = editor.canvasSize_.x * 0.5f - centerX * editor.zoom;
        editor.scrollY = editor.canvasSize_.y * 0.5f - centerY * editor.zoom;
    }

    void drawLayerTabs(TerrainNodeGraphV2& graph) {
        // Toolbar/audit is drawn before the canvas, so reconcile here as well;
        // audit results always describe the current topology, not the prior frame.
        editor.synchronizeLayerInterfaces(graph);
        if (editor.focusedGroupId != 0) {
            NodeSystem::NodeGroup* focused = graph.getGroup(editor.focusedGroupId);
            // A preset may legitimately move the last shared node to its canonical layer.
            // Never leave the editor focused on an empty, therefore invisible tab.
            if (!focused || focused->nodeIds.empty()) editor.focusedGroupId = 0;
        }
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(3.0f, 3.0f));
        auto tab = [&](const char* label, uint32_t id, ImU32 color) {
            const bool active = editor.focusedGroupId == id;
            const ImVec4 tint = ImGui::ColorConvertU32ToFloat4(color);
            ImGui::PushStyleColor(ImGuiCol_Button, active
                ? ImVec4(tint.x, tint.y, tint.z, 0.72f) : ImVec4(0.12f, 0.13f, 0.16f, 0.92f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(tint.x, tint.y, tint.z, 0.55f));
            if (ImGui::SmallButton(label)) activateLayer(graph, id);
            ImGui::PopStyleColor(2);
        };

        tab("All", 0, IM_COL32(100, 110, 130, 255));
        for (auto& group : graph.groups) {
            if (group.nodeIds.empty()) continue;
            ImGui::SameLine();
            const char* label = group.name.c_str();
            if (group.name.size() > 3 && std::isdigit(static_cast<unsigned char>(group.name[0])) &&
                std::isdigit(static_cast<unsigned char>(group.name[1])) && group.name[2] == ' ') {
                label += 3;
            }
            tab(label, group.id, group.color);
            if (ImGui::IsItemHovered()) {
                int incoming = 0, outgoing = 0;
                for (const auto& link : graph.links) {
                    const NodeSystem::NodeBase* a = graph.getPinOwner(link.startPinId);
                    const NodeSystem::NodeBase* b = graph.getPinOwner(link.endPinId);
                    const bool aInside = a && a->groupId == group.id;
                    const bool bInside = b && b->groupId == group.id;
                    if (!aInside && bInside) ++incoming;
                    if (aInside && !bInside) ++outgoing;
                }
                ImGui::SetTooltip("%zu nodes | %d inputs | %d outputs",
                                  group.nodeIds.size(), incoming, outgoing);
            }
        }
        if (editor.focusedGroupId != 0) {
            ImGui::SameLine();
            if (ImGui::SmallButton("Arrange")) {
                editor.autoArrangeGroup(graph, editor.focusedGroupId);
                if (NodeSystem::NodeGroup* group = graph.getGroup(editor.focusedGroupId)) {
                    activateLayer(graph, group->id);
                }
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("One-shot tidy for this layer; manual placement remains free afterwards");
            }
            if (NodeSystem::NodeGroup* group = graph.getGroup(editor.focusedGroupId)) {
                ImGui::SameLine();
                if (ImGui::Checkbox("Publish##TerrainLayer", &group->publicationEnabled)) {
                    ProjectManager::getInstance().markModified();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Enable explicit output/sink publication in this layer");
                }
            }
        }
        ImGui::SameLine();
        ImGui::BeginDisabled(editor.selectedNodeIds.empty() || graph.isEvaluatingAsync());
        if (ImGui::SmallButton("+ Layer")) {
            std::snprintf(newLayerName, sizeof(newLayerName), "%s", "River");
            ImGui::OpenPopup("CreateTerrainLayerFromSelection");
        }
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            ImGui::SetTooltip(editor.selectedNodeIds.empty()
                ? "Select the nodes that belong to the new layer first"
                : "Create a semantic layer from the selected nodes");
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("Audit")) ImGui::OpenPopup("TerrainGraphAuditPopup");
        ImGui::PopStyleVar();

        if (ImGui::BeginPopup("CreateTerrainLayerFromSelection")) {
            ImGui::TextUnformatted("Create layer from selection");
            ImGui::SetNextItemWidth(190.0f);
            ImGui::InputText("Name", newLayerName, sizeof(newLayerName));
            const bool valid = newLayerName[0] != '\0' && !editor.selectedNodeIds.empty();
            ImGui::BeginDisabled(!valid);
            if (ImGui::Button("Create and Open")) {
                createLayerFromSelection(graph, newLayerName);
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndDisabled();
            ImGui::SameLine();
            if (ImGui::Button("Cancel")) ImGui::CloseCurrentPopup();
            ImGui::EndPopup();
        }

        if (ImGui::BeginPopup("TerrainGraphAuditPopup")) {
            drawGraphAudit(graph);
            ImGui::EndPopup();
        }
    }

    void drawGraphAudit(TerrainNodeGraphV2& graph) {
        int danglingLinks = 0;
        int missingRequiredInputs = 0;
        int membershipErrors = 0;
        int heightOutputs = 0;
        int disconnectedPorts = 0;
        int enabledPublishers = 0;
        int disabledPublishers = 0;

        for (const auto& link : graph.links) {
            if (!graph.findPin(link.startPinId) || !graph.findPin(link.endPinId)) ++danglingLinks;
        }
        for (const auto& node : graph.nodes) {
            if (node->getTypeId() == "TerrainV2.HeightOutput") ++heightOutputs;
            const auto* terrainNode = dynamic_cast<const TerrainNodeBase*>(node.get());
            const bool publicationSink = terrainNode && isPublicationSink(*terrainNode);
            const NodeSystem::NodeGroup* publicationGroup = node->groupId ? graph.getGroup(node->groupId) : nullptr;
            const bool publisherEnabled = !publicationSink ||
                (terrainNode->publicationEnabled && (!publicationGroup || publicationGroup->publicationEnabled));
            for (const auto& input : node->inputs) {
                if (!publisherEnabled || input.optional || input.hasDefaultValue()) continue;
                const bool connected = std::any_of(graph.links.begin(), graph.links.end(),
                    [&input](const NodeSystem::Link& link) { return link.endPinId == input.id; });
                if (!connected) ++missingRequiredInputs;
            }
            if (node->groupId != 0) {
                const NodeSystem::NodeGroup* group = graph.getGroup(node->groupId);
                if (!group || std::find(group->nodeIds.begin(), group->nodeIds.end(), node->id) == group->nodeIds.end()) {
                    ++membershipErrors;
                }
            }
            if (publicationSink) {
                if (publisherEnabled) ++enabledPublishers;
                else ++disabledPublishers;
            }
        }
        for (const auto& group : graph.groups) {
            for (uint32_t nodeId : group.nodeIds) {
                const NodeSystem::NodeBase* node = graph.getNode(nodeId);
                if (!node || node->groupId != group.id) ++membershipErrors;
            }
            disconnectedPorts += static_cast<int>(std::count_if(
                group.interfacePorts.begin(), group.interfacePorts.end(),
                [](const NodeSystem::LayerInterfacePort& port) { return !port.connected; }));
        }

        const int hardErrors = danglingLinks + membershipErrors + (heightOutputs == 1 ? 0 : 1);
        const bool contractValid = hardErrors == 0 && missingRequiredInputs == 0 && disconnectedPorts == 0;
        const ImVec4 statusColor = contractValid
            ? ImVec4(0.35f, 0.85f, 0.45f, 1.0f) : ImVec4(1.0f, 0.58f, 0.25f, 1.0f);
        ImGui::TextColored(statusColor, contractValid ? "Graph contract: valid" : "Graph contract: needs attention");
        ImGui::Separator();
        ImGui::Text("Nodes %d | Links %d | Layers %d", static_cast<int>(graph.nodeCount()),
                    static_cast<int>(graph.linkCount()), static_cast<int>(graph.groups.size()));
        ImGui::Text("Publishers: %d enabled | %d disabled", enabledPublishers, disabledPublishers);
        ImGui::Text("Height Output: %d", heightOutputs);
        ImGui::Text("Missing required inputs: %d", missingRequiredInputs);
        ImGui::Text("Dangling links: %d", danglingLinks);
        ImGui::Text("Layer membership errors: %d", membershipErrors);
        ImGui::Text("Disconnected saved ports: %d", disconnectedPorts);

        if (editor.focusedGroupId != 0) {
            if (NodeSystem::NodeGroup* group = graph.getGroup(editor.focusedGroupId)) {
                ImGui::SeparatorText(group->name.c_str());
                if (group->interfacePorts.empty()) ImGui::TextDisabled("No cross-layer interface ports");
                for (const auto& port : group->interfacePorts) {
                    const char* direction = port.direction == NodeSystem::LayerPortDirection::Input ? "IN " : "OUT";
                    ImGui::TextColored(port.connected ? ImVec4(0.65f, 0.85f, 1.0f, 1.0f)
                                                        : ImVec4(1.0f, 0.45f, 0.32f, 1.0f),
                                       "%s  %s  [P%u]", direction, port.name.c_str(), port.id);
                }
                const bool hasDisconnected = std::any_of(group->interfacePorts.begin(), group->interfacePorts.end(),
                    [](const NodeSystem::LayerInterfacePort& port) { return !port.connected; });
                ImGui::BeginDisabled(!hasDisconnected);
                if (ImGui::Button("Prune Disconnected Ports")) {
                    group->interfacePorts.erase(std::remove_if(group->interfacePorts.begin(), group->interfacePorts.end(),
                        [](const NodeSystem::LayerInterfacePort& port) { return !port.connected; }),
                        group->interfacePorts.end());
                    ProjectManager::getInstance().markModified();
                }
                ImGui::EndDisabled();
            }
        } else {
            ImGui::TextDisabled("Open a layer tab to inspect its named interface ports.");
        }
    }

    void createLayerFromSelection(TerrainNodeGraphV2& graph, const std::string& requestedName) {
        std::vector<NodeSystem::NodeBase*> selected;
        selected.reserve(editor.selectedNodeIds.size());
        for (uint32_t nodeId : editor.selectedNodeIds) {
            if (NodeSystem::NodeBase* node = graph.getNode(nodeId)) selected.push_back(node);
        }
        if (selected.empty()) return;

        std::string uniqueName = requestedName;
        int suffix = 2;
        auto nameExists = [&graph](const std::string& name) {
            return std::any_of(graph.groups.begin(), graph.groups.end(),
                [&name](const NodeSystem::NodeGroup& group) { return group.name == name; });
        };
        while (nameExists(uniqueName)) uniqueName = requestedName + " " + std::to_string(suffix++);

        float minX = selected.front()->x;
        float minY = selected.front()->y;
        float maxX = minX + 180.0f;
        float maxY = minY + 100.0f;
        for (const NodeSystem::NodeBase* node : selected) {
            const float customWidth = node->getCustomWidth();
            const float width = node->uiWidth > 0.0f ? node->uiWidth
                : (customWidth > 0.0f ? customWidth : 180.0f);
            const float height = node->collapsed ? 34.0f
                : 54.0f + 22.0f * static_cast<float>(std::max(node->inputs.size(), node->outputs.size()));
            minX = std::min(minX, node->x);
            minY = std::min(minY, node->y);
            maxX = std::max(maxX, node->x + width);
            maxY = std::max(maxY, node->y + height);
        }

        const uint32_t groupId = graph.createGroup(uniqueName, ImVec2(minX - 35.0f, minY - 55.0f),
            ImVec2(maxX - minX + 70.0f, maxY - minY + 90.0f));
        NodeSystem::NodeGroup* group = graph.getGroup(groupId);
        if (!group) return;
        group->comment = "Auto-managed terrain layer";
        group->color = IM_COL32(55, 115, 175, 105);
        for (NodeSystem::NodeBase* node : selected) {
            graph.removeNodeFromGroups(node->id);
            graph.addNodeToGroup(node->id, groupId);
        }
        ProjectManager::getInstance().markModified();
        activateLayer(graph, groupId);
    }

    static bool pinsCompatibleIgnoringOwner(const NodeSystem::Pin& first,
                                            const NodeSystem::Pin& second) {
        NodeSystem::Pin a = first;
        NodeSystem::Pin b = second;
        a.nodeId = 1;
        b.nodeId = 2;
        return a.canConnectTo(b);
    }

    bool nodeAcceptsPendingLink(TerrainNodeGraphV2& probeGraph,
                                NodeType type,
                                const NodeSystem::Pin& draggedPin) const {
        NodeSystem::NodeBase* probe = probeGraph.addTerrainNode(type, 0.0f, 0.0f);
        if (!probe) return false;

        if (draggedPin.kind == NodeSystem::PinKind::Output) {
            for (const auto& input : probe->inputs) {
                if (pinsCompatibleIgnoringOwner(draggedPin, input)) return true;
            }
        } else {
            for (const auto& output : probe->outputs) {
                if (pinsCompatibleIgnoringOwner(output, draggedPin)) return true;
            }
        }
        return false;
    }

    void rebuildCompatibleLinkNodeTypes(TerrainNodeGraphV2& graph) {
        compatibleLinkNodeTypes.clear();
        const NodeSystem::Pin* draggedPin = graph.findPin(editor.releasedLinkPinId);
        if (!draggedPin) return;

        // Constructors only declare metadata and pins; a scratch graph lets the
        // picker use those declarations as the single source of compatibility.
        TerrainNodeGraphV2 probeGraph;
        for (const auto& category : categories) {
            for (const auto& node : category.nodes) {
                if (nodeAcceptsPendingLink(probeGraph, node.first, *draggedPin)) {
                    compatibleLinkNodeTypes.push_back(node.first);
                }
            }
        }
    }

    bool isCompatibleLinkNodeType(NodeType type) const {
        return std::find(compatibleLinkNodeTypes.begin(), compatibleLinkNodeTypes.end(), type)
            != compatibleLinkNodeTypes.end();
    }

    void createAndConnectPendingNode(TerrainNodeGraphV2& graph, NodeType type) {
        NodeSystem::NodeBase* node = graph.addTerrainNode(
            type, editor.mousePosOnRightClick.x, editor.mousePosOnRightClick.y);
        if (!node) return;

        addNodeToActiveLayer(graph, node);

        editor.onNodeAdded(graph, node);
        editor.selectedNodeId = node->id;
        editor.selectedNodeIds.clear();
        editor.selectedNodeIds.push_back(node->id);
        if (editor.onNodeSelected) editor.onNodeSelected(node->id);
        ProjectManager::getInstance().markModified();
        ImGui::CloseCurrentPopup();
    }

    void drawLinkCreateMenu(TerrainNodeGraphV2& graph) {
        if (ImGui::IsWindowAppearing()) {
            linkCreateSearchBuffer[0] = '\0';
            rebuildCompatibleLinkNodeTypes(graph);
            ImGui::SetKeyboardFocusHere();
        }

        const NodeSystem::Pin* draggedPin = graph.findPin(editor.releasedLinkPinId);
        if (!draggedPin) {
            ImGui::CloseCurrentPopup();
            return;
        }

        ImGui::SetNextItemWidth(260.0f);
        const bool createFirstMatch = ImGui::InputTextWithHint(
            "##LinkCreateSearch", "Search compatible nodes...",
            linkCreateSearchBuffer, sizeof(linkCreateSearchBuffer),
            ImGuiInputTextFlags_EnterReturnsTrue);
        ImGui::Separator();

        std::string query = linkCreateSearchBuffer;
        std::transform(query.begin(), query.end(), query.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

        bool hasMatch = false;
        NodeType firstMatch = NodeType::HeightmapInput;
        bool createSelected = false;
        NodeType selectedType = firstMatch;

        ImGui::BeginChild("##CompatibleNodeList", ImVec2(300.0f, 320.0f), false,
                          ImGuiWindowFlags_AlwaysVerticalScrollbar);
        for (const auto& category : categories) {
            bool categoryShown = false;
            for (const auto& [type, name] : category.nodes) {
                if (!isCompatibleLinkNodeType(type)) continue;

                std::string lowerName = name;
                std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(),
                    [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                if (!query.empty() && lowerName.find(query) == std::string::npos) continue;

                if (!hasMatch) {
                    firstMatch = type;
                    hasMatch = true;
                }
                if (!categoryShown) {
                    ImGui::TextColored(category.color, "%s", category.name);
                    categoryShown = true;
                }
                ImGui::PushID(static_cast<int>(type));
                if (ImGui::Selectable(name)) {
                    selectedType = type;
                    createSelected = true;
                }
                ImGui::PopID();
            }
            if (categoryShown) ImGui::Spacing();
        }
        if (!hasMatch) ImGui::TextDisabled("No compatible nodes");
        ImGui::EndChild();

        if (createSelected) {
            createAndConnectPendingNode(graph, selectedType);
        } else if (createFirstMatch && hasMatch) {
            createAndConnectPendingNode(graph, firstMatch);
        }
    }

    // ========================================================================
    // LINK INSERTION HELPERS
    // ========================================================================
    
    /**
     * @brief Find a link near the mouse position for drop-on-link insertion
     */
    NodeSystem::Link* findLinkNearMouse(TerrainNodeGraphV2& graph, ImVec2 mousePos, ImVec2 canvasPos) {
        (void)canvasPos;
        // Use the exact geometry drawn this frame. This also covers virtual
        // layer-boundary cables, whose hidden endpoint has no ordinary node pin
        // position in the focused layer.
        return editor.findVisibleLinkAt(graph, mousePos, 15.0f);
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
        
        auto compatibilityScore = [](const NodeSystem::Pin& candidate,
                                     const NodeSystem::Pin& reference) {
            if (!candidate.canConnectTo(reference) && !reference.canConnectTo(candidate)) return -1;
            int score = candidate.dataType == reference.dataType ? 100 : 0;
            if (candidate.dataType == NodeSystem::DataType::Image2D &&
                reference.dataType == NodeSystem::DataType::Image2D) {
                if (candidate.imageSemantic == reference.imageSemantic) score += 50;
                else if (candidate.imageSemantic == NodeSystem::ImageSemantic::Generic ||
                         reference.imageSemantic == NodeSystem::ImageSemantic::Generic) score += 10;
                if (candidate.imageChannels == reference.imageChannels) score += 20;
            }
            return score;
        };

        // A node with many Image2D pins must not splice through the first merely
        // type-compatible socket. Prefer the same semantic (Height, Mask, ...).
        NodeSystem::Pin* matchingInput = nullptr;
        int bestInputScore = -1;
        for (auto& pin : newNode->inputs) {
            const int score = compatibilityScore(pin, *sourcePin);
            if (score > bestInputScore) {
                bestInputScore = score;
                matchingInput = &pin;
            }
        }

        NodeSystem::Pin* matchingOutput = nullptr;
        int bestOutputScore = -1;
        for (auto& pin : newNode->outputs) {
            const int score = compatibilityScore(pin, *destPin);
            if (score > bestOutputScore) {
                bestOutputScore = score;
                matchingOutput = &pin;
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
        
        // Replace atomically from the user's point of view. If either half is
        // rejected, remove the partial splice and restore the original edge.
        const uint32_t incomingLinkId = graph.addLink(oldStartPinId, matchingInput->id);
        const uint32_t outgoingLinkId = graph.addLink(matchingOutput->id, oldEndPinId);
        if (incomingLinkId == 0 || outgoingLinkId == 0) {
            if (incomingLinkId != 0) graph.removeLink(incomingLinkId);
            if (outgoingLinkId != 0) graph.removeLink(outgoingLinkId);
            graph.addLink(oldStartPinId, oldEndPinId);
            return false;
        }

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
                            NodeSystem::NodeBase* node = graph.addTerrainNode(nodePair.first, spawnX, spawnY);
                            addNodeToActiveLayer(graph, node);
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
        const bool structuralEditBlocked = graph.isEvaluatingAsync();
        ImGui::BeginDisabled(structuralEditBlocked);
        if (ImGui::Button("Reset Graph")) {
            graph.createDefaultGraph(terrain);
            editor.scrollX = 0; 
            editor.scrollY = 0;
            editor.zoom = 1.0f;
        }
        ImGui::SameLine();

        if (ImGui::Button("Setups")) ImGui::OpenPopup("TerrainSetupsMenu");
        if (ImGui::BeginPopup("TerrainSetupsMenu")) {
            if (ImGui::MenuItem("Add Snow Layer")) {
                if (graph.addSnowLayerSetup()) {
                    frameAllNodes(graph);
                    setupStatus = "Snow layer added: base masks preserved";
                    ProjectManager::getInstance().markModified();
                } else {
                    setupStatus = "Snow setup needs a connected Height Output";
                }
                setupStatusUntil = ImGui::GetTime() + 4.0;
            }
            if (ImGui::MenuItem("Add River Network")) {
                if (graph.addRiverNetworkSetup()) {
                    frameAllNodes(graph);
                    setupStatus = "Hydrology ready: lakes, river network and RiverSpline";
                    ProjectManager::getInstance().markModified();
                } else {
                    setupStatus = "River setup needs a connected Height Output";
                }
                setupStatusUntil = ImGui::GetTime() + 4.0;
            }
            if (ImGui::BeginMenu("Add Biome Fields")) {
                auto addBiomePreset = [&](const char* label, BiomeClimatePreset preset) {
                    if (!ImGui::MenuItem(label)) return;
                    if (graph.addBiomeFieldsSetup(preset)) {
                        frameAllNodes(graph);
                        setupStatus = std::string("Biome fields ready: ") +
                            BiomeComposerNode::getPresetName(preset);
                        ProjectManager::getInstance().markModified();
                    } else {
                        setupStatus = "Biome setup needs a connected Height Output";
                    }
                    setupStatusUntil = ImGui::GetTime() + 4.0;
                };
                addBiomePreset("Temperate Mixed", BiomeClimatePreset::TemperateMixed);
                addBiomePreset("Lush Valleys", BiomeClimatePreset::LushValleys);
                addBiomePreset("Alpine Tundra", BiomeClimatePreset::AlpineTundra);
                addBiomePreset("Arid Highlands", BiomeClimatePreset::AridHighlands);
                addBiomePreset("Boreal Mountains", BiomeClimatePreset::BorealMountains);
                ImGui::EndMenu();
            }
            if (ImGui::MenuItem("Add Biome Foliage")) {
                if (graph.addBiomeFoliageSetup()) {
                    frameAllNodes(graph);
                    setupStatus = "Biome foliage ready: bind any unmatched foliage layers";
                    ProjectManager::getInstance().markModified();
                } else {
                    setupStatus = "Could not create biome foliage setup";
                }
                setupStatusUntil = ImGui::GetTime() + 4.0;
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Example: Snowy Mountain Valley...")) {
                requestSnowyMountainValleyPopup = true;
            }
            ImGui::EndPopup();
        }
        ImGui::EndDisabled();
        ImGui::SameLine();

        if (requestSnowyMountainValleyPopup) {
            ImGui::OpenPopup("Create Snowy Mountain Valley");
            requestSnowyMountainValleyPopup = false;
        }

        if (ImGui::BeginPopupModal("Create Snowy Mountain Valley", nullptr,
                                   ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("Replace the current graph with the example setup?");
            ImGui::TextDisabled("Ridge noise, erosion, base masks, snow geometry and splat layers");
            ImGui::Spacing();
            ImGui::BeginDisabled(graph.isEvaluatingAsync());
            if (ImGui::Button("Create", ImVec2(110.0f, 0.0f))) {
                graph.createSnowyMountainValleyGraph(terrain);
                editor.scrollX = 0.0f;
                editor.scrollY = 0.0f;
                editor.zoom = 0.85f;
                setupStatus = "Snowy Mountain Valley example created";
                setupStatusUntil = ImGui::GetTime() + 4.0;
                ProjectManager::getInstance().markModified();
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndDisabled();
            ImGui::SameLine();
            if (ImGui::Button("Cancel", ImVec2(110.0f, 0.0f))) {
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
        
        // Once the background height-data phase finishes, run the (main-thread-only)
        // GPU-texture + mesh/BVH finalize phases and fire the same rebuild flags the
        // old synchronous handler set inline. Polled every frame this toolbar draws
        // (i.e. while the terrain node-editor tab is open). If the user switches to
        // a different tab mid-evaluation, the worker thread still runs to completion
        // in the background — finalize (GPU upload + mesh rebuild) simply happens on
        // the next frame this tab is visible again, since it needs ImGui/GPU context
        // that's only meaningful while this panel is drawing.
        if (graph.pollEvaluateAsync()) {
            const std::vector<int> scatteredGroups = graph.getLastScatteredFoliageGroupIds();
            if (onFoliageScattered && !scatteredGroups.empty()) {
                onFoliageScattered(terrain, scatteredGroups);
            }
            ctx.renderer.resetCPUAccumulation();
            // Auxiliary outputs can update the terrain splat texture without a
            // topology change. Push that dirty texture to both render and Solid
            // viewport consumers immediately instead of waiting for a later
            // geometry/material rebuild to happen incidentally.
            ctx.renderer.updateBackendMaterials(ctx.scene);
            if (g_viewport_backend) g_viewport_backend->resetAccumulation();
            // CPU Embree BVH rebuild and OptiX rebuild are already dispatched via
            // std::async by their Main.cpp consumers (non-blocking), so requesting
            // them unconditionally costs nothing extra on the main thread.
            g_bvh_rebuild_pending = true;
            g_optix_rebuild_pending = true;

            extern bool g_geometry_dirty;
            extern std::atomic<uint64_t> g_scene_geometry_generation;
            extern std::atomic<bool> g_needs_optix_sync;
            extern bool g_mesh_cache_dirty;
            extern bool g_viewport_raster_rebuild_pending;
            extern bool g_vulkan_rebuild_pending;

            g_geometry_dirty = true;
            g_scene_geometry_generation.fetch_add(1, std::memory_order_release);
            g_needs_optix_sync.store(true, std::memory_order_release);
            g_mesh_cache_dirty = true;

            // g_viewport_raster_rebuild_pending / g_vulkan_rebuild_pending each
            // trigger a FULL re-upload of every object's geometry to the GPU
            // (buildRasterGeometry / rebuildAccelerationStructure+updateGeometry) —
            // the one synchronous, main-thread-blocking cost left after
            // backgrounding the height-data compute. When this evaluate only
            // updated terrain positions in place (same topology — the common
            // "tweak a parameter and re-evaluate" case), try the cheap partial
            // refit the terrain sculpt-brush path already uses in production
            // (scene_ui_terrain.hpp's commitTerrainStroke/updateTerrainBLASPartial)
            // instead of re-uploading the whole scene. Topology changes (first
            // evaluate, resolution edits) still take the proven full-rebuild path.
            if (graph.lastFinalizeWasFullRebuild() || !terrain) {
                g_viewport_raster_rebuild_pending = true;
                g_vulkan_rebuild_pending = true;
            } else {
                bool vulkanHandled = false;
                if (auto* vkRtBackend = dynamic_cast<Backend::VulkanBackendAdapter*>(g_backend.get())) {
                    vulkanHandled = vkRtBackend->updateTerrainBLASPartial(terrain->name, terrain);
                    if (vulkanHandled) vkRtBackend->resetAccumulation();
                }
                if (!vulkanHandled) {
                    g_vulkan_rebuild_pending = true;
                }

                // Terrain is now a single flat (SoA) mesh — there is no facade triangle
                // list left to feed the incremental raster patch, so fall back to a full
                // raster rebuild here (RT/BLAS refit above stays incremental either way).
                g_viewport_raster_rebuild_pending = true;
            }

            if (ctx.optix_gpu_ptr) {
                if (g_hasCUDA) cudaDeviceSynchronize();
                ctx.optix_gpu_ptr->resetAccumulation();
            }
        }

        bool evaluating = graph.isEvaluatingAsync();
        auto publishMaterialOnlyUpdate = [&]() {
            ctx.renderer.resetCPUAccumulation();
            ctx.renderer.updateBackendMaterials(ctx.scene);
            if (g_viewport_backend) g_viewport_backend->resetAccumulation();
            // Wake an already-open mask inspector after the debounced cache has
            // been replaced. Without this, a drag could leave the preview at
            // the pre-drag revision until the node was selected again.
            ++maskPreviewRevision;
            setupStatus = "Material branch updated (geometry/BVH unchanged)";
            setupStatusUntil = ImGui::GetTime() + 3.0;
        };

        ImGui::Checkbox("Auto Preview", &autoPreviewSelectedNode);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Selecting a Height node previews that node and its upstream chain.");
        }
        ImGui::SameLine();
        ImGui::BeginDisabled(evaluating || !graph.isPreviewActive());
        if (ImGui::Button("Final Output")) {
            if (graph.restoreTerrainPreview(terrain, ctx.scene)) {
                ctx.renderer.resetCPUAccumulation();
                g_bvh_rebuild_pending = true;
                g_optix_rebuild_pending = true;
                extern bool g_geometry_dirty;
                extern std::atomic<uint64_t> g_scene_geometry_generation;
                extern std::atomic<bool> g_needs_optix_sync;
                extern bool g_mesh_cache_dirty;
                extern bool g_viewport_raster_rebuild_pending;
                extern bool g_vulkan_rebuild_pending;
                g_geometry_dirty = true;
                g_scene_geometry_generation.fetch_add(1, std::memory_order_release);
                g_needs_optix_sync.store(true, std::memory_order_release);
                g_mesh_cache_dirty = true;
                g_viewport_raster_rebuild_pending = true;
                g_vulkan_rebuild_pending = true;
            }
        }
        ImGui::EndDisabled();
        ImGui::SameLine();

        // Debounce rapid graph browsing and parameter drags. Only Height-capable
        // nodes become 3D preview terminals; mask nodes keep their heat-map panel.
        if (autoPreviewSelectedNode && pendingAutoPreviewNodeId != 0 && !evaluating &&
            !propertyEditActiveThisFrame &&
            (ImGui::GetTime() - pendingAutoPreviewSince) >= 0.15) {
            NodeSystem::NodeBase* previewNode = graph.getNode(pendingAutoPreviewNodeId);
            bool supportsHeightPreview = dynamic_cast<HeightOutputNode*>(previewNode) != nullptr;
            bool hasMaskOutput = false;
            if (previewNode && !supportsHeightPreview) {
                for (const auto& output : previewNode->outputs) {
                    if (output.dataType == NodeSystem::DataType::Image2D &&
                        output.imageSemantic == NodeSystem::ImageSemantic::Mask) {
                        hasMaskOutput = true;
                    }
                    if (output.dataType == NodeSystem::DataType::Image2D &&
                        output.imageSemantic == NodeSystem::ImageSemantic::Height) {
                        supportsHeightPreview = true;
                    }
                }
            } else if (previewNode) {
                for (const auto& output : previewNode->outputs) {
                    if (output.dataType == NodeSystem::DataType::Image2D &&
                        output.imageSemantic == NodeSystem::ImageSemantic::Mask) {
                        hasMaskOutput = true;
                        break;
                    }
                }
            }
            const uint32_t requestedNode = pendingAutoPreviewNodeId;
            pendingAutoPreviewNodeId = 0;
            // Mixed Height+Mask nodes use the cheap heat-map inspector by
            // default. Selecting/opening Mask Show must never silently rebuild
            // the terrain mesh and its acceleration structures.
            if (supportsHeightPreview && !hasMaskOutput) {
                graph.evaluateTerrainPreviewAsync(requestedNode, terrain, ctx.scene);
                evaluating = graph.isEvaluatingAsync();
            } else if (graph.hasEvaluationCache() &&
                       graph.classifyDirtyEvaluationImpact() ==
                           TerrainNodeGraphV2::DirtyEvaluationImpact::MaterialOnly &&
                       graph.evaluateDirtyMaterialOutputs(terrain, ctx.scene)) {
                publishMaterialOnlyUpdate();
            } else if (graph.hasEvaluationCache() &&
                       graph.classifyDirtyEvaluationImpact() ==
                           TerrainNodeGraphV2::DirtyEvaluationImpact::FoliageOnly &&
                       graph.evaluateDirtyFoliageOutputs(terrain)) {
                const std::vector<int> scatteredGroups = graph.getLastScatteredFoliageGroupIds();
                if (onFoliageScattered && !scatteredGroups.empty()) {
                    onFoliageScattered(terrain, scatteredGroups);
                }
                setupStatus = "Foliage settings applied live";
                setupStatusUntil = ImGui::GetTime() + 2.0;
            }
        }

        ImGui::PushStyleColor(ImGuiCol_Button, evaluating ? ImVec4(0.65f, 0.25f, 0.20f, 1.0f)
                                                           : ImVec4(0.2f, 0.6f, 0.3f, 1.0f));
        if (evaluating) {
            ImGui::BeginDisabled(!graph.activeEvalContext);
            if (ImGui::Button("Cancel Evaluate") && graph.activeEvalContext) {
                graph.activeEvalContext->requestCancel();
            }
            ImGui::EndDisabled();
            if (graph.activeEvalContext) {
                ImGui::SameLine();
                ImGui::TextDisabled("%.0f%%", graph.activeEvalContext->getProgress() * 100.0f);
            }
        } else {
            const auto dirtyImpact = graph.classifyDirtyEvaluationImpact();
            const bool canApplyMaterialOnly = graph.hasEvaluationCache() &&
                dirtyImpact == TerrainNodeGraphV2::DirtyEvaluationImpact::MaterialOnly;
            const bool canApplyFoliageOnly = graph.hasEvaluationCache() &&
                dirtyImpact == TerrainNodeGraphV2::DirtyEvaluationImpact::FoliageOnly;
            const char* evaluateLabel = canApplyMaterialOnly ? "Apply Material"
                : (canApplyFoliageOnly ? "Apply Foliage" : "Evaluate");
            if (ImGui::Button(evaluateLabel)) {
            // Mark all nodes dirty and kick off the height-data compute (the
            // expensive part — noise/erosion) on a worker thread. Splat/hardness
            // GPU upload + mesh rebuild run later on the main thread once
            // pollEvaluateAsync() (above) detects completion.
                if (canApplyMaterialOnly && graph.evaluateDirtyMaterialOutputs(terrain, ctx.scene)) {
                    publishMaterialOnlyUpdate();
                } else if (canApplyFoliageOnly && graph.evaluateDirtyFoliageOutputs(terrain)) {
                    const std::vector<int> scatteredGroups = graph.getLastScatteredFoliageGroupIds();
                    if (onFoliageScattered && !scatteredGroups.empty()) {
                        onFoliageScattered(terrain, scatteredGroups);
                    }
                    setupStatus = "Foliage layer settings applied without terrain rebuild";
                    setupStatusUntil = ImGui::GetTime() + 3.0;
                } else if (graph.hasEvaluationCache() &&
                           dirtyImpact == TerrainNodeGraphV2::DirtyEvaluationImpact::None) {
                    setupStatus = "No enabled output depends on the edited node";
                    setupStatusUntil = ImGui::GetTime() + 3.0;
                } else {
                    graph.evaluateTerrainAsync(terrain, ctx.scene);
                }
            }
        }
        ImGui::PopStyleColor();

        ImGui::SameLine();
        ImGui::TextDisabled("|");
        ImGui::SameLine();
        drawLayerTabs(graph);

        ImGui::SameLine();
        ImGui::TextDisabled("|");
        ImGui::SameLine();
        ImGui::Text("Nodes: %d  Links: %d", (int)graph.nodeCount(), (int)graph.linkCount());
        ImGui::SameLine();
        ImGui::Text("Zoom: %.1f", editor.zoom);
        if (!setupStatus.empty() && ImGui::GetTime() < setupStatusUntil) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.55f, 0.85f, 1.0f, 1.0f), "%s", setupStatus.c_str());
        }

        ImGui::Separator();
    }
};

} // namespace TerrainNodesV2

