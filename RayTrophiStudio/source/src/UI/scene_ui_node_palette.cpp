/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          scene_ui_node_palette.cpp
* Author:        Kemal Demirtaş
* Date:          2026
* License:       Proprietary / RayTrophi Studio
* =========================================================================
*/
#include "scene_ui_node_palette.hpp"
#include "imgui_internal.h"
#include <cctype>

namespace TerrainNodesV2 {

ModernTerrainNodePalette::ModernTerrainNodePalette() {
    initializeCategories();
}

void ModernTerrainNodePalette::initializeCategories() {
    categories_ = {
        {"Input", ImVec4(0.22f, 0.72f, 0.32f, 1.0f), {
            {NodeType::HeightmapInput, "Heightmap", "Imports 16-bit heightmap image", UIWidgets::IconType::Terrain},
            {NodeType::NoiseGenerator, "Noise Generator", "Procedural Perlin/Simplex noise source", UIWidgets::IconType::Noise},
            {NodeType::HardnessInput, "Hardness Input", "Material structural resistance map", UIWidgets::IconType::ClayTool},
            {NodeType::CurveInput, "Curve Input", "Immutable scene spline snapshot", UIWidgets::IconType::Gizmo}
        }, true},
        {"Landform", ImVec4(0.62f, 0.45f, 0.32f, 1.0f), {
            {NodeType::MountainRange, "Mountain Range", "Orogeny & tectonic uplift ridges", UIWidgets::IconType::DrawTool},
            {NodeType::BasinValley, "Basin & Valley", "Depression and valley formation", UIWidgets::IconType::FlattenTool},
            {NodeType::TerrainDetail, "Terrain Detail", "Micro-topography detail accentuation", UIWidgets::IconType::DrawSharpTool},
            {NodeType::RoadCarve, "Road Carve", "Grade-limited road terrain and infrastructure fields", UIWidgets::IconType::Sculpt},
            {NodeType::RoadNetwork, "Road Network", "Carves every curve with a road assignment in one solve", UIWidgets::IconType::Sculpt}
        }, true},
        {"Erosion", ImVec4(0.32f, 0.52f, 0.92f, 1.0f), {
            {NodeType::HydraulicErosion, "Hydraulic", "Water stream sediment transport erosion", UIWidgets::IconType::Water},
            {NodeType::ThermalErosion, "Thermal", "Talus slope & scree settling erosion", UIWidgets::IconType::ScrapeTool},
            {NodeType::FluvialErosion, "Fluvial", "River network carving erosion", UIWidgets::IconType::Vortex},
            {NodeType::WindErosion, "Wind", "Aeolian sand & wind abrasion", UIWidgets::IconType::Wind}
        }, true},
        {"Filter", ImVec4(0.42f, 0.62f, 0.82f, 1.0f), {
            {NodeType::Smooth, "Smooth", "Gaussian heightmap blurring filter", UIWidgets::IconType::SmoothTool},
            {NodeType::Normalize, "Normalize", "Rescales height range to 0..1 span", UIWidgets::IconType::ScaleAxis},
            {NodeType::MaskAdjust, "Mask Adjust", "Contrast and curve response tuning", UIWidgets::IconType::DodgeTool},
            {NodeType::Remap, "Remap / Levels", "Min/Max level remapping", UIWidgets::IconType::Settings},
            {NodeType::Terrace, "Terrace", "Step terracing & sedimentary ledge effect", UIWidgets::IconType::ShadeFlatTool},
            {NodeType::EdgeFalloff, "Edge Falloff", "Border fade & margin falloff mask", UIWidgets::IconType::PinchTool}
        }, true},
        {"Mask", ImVec4(0.72f, 0.42f, 0.82f, 1.0f), {
            {NodeType::HeightMask, "Height Mask", "Selects elevation altitude range", UIWidgets::IconType::MaskTool},
            {NodeType::SlopeMask, "Slope Mask", "Selects surface incline steepness", UIWidgets::IconType::Rotate},
            {NodeType::CurvatureMask, "Curvature Mask", "Convexity and concavity mask", UIWidgets::IconType::CreaseTool},
            {NodeType::FlowMask, "Flow", "Hydraulic accumulation flow paths", UIWidgets::IconType::Water},
            {NodeType::ExposureMask, "Sun Exposure", "Solar aspect and lighting exposure", UIWidgets::IconType::LightDir},
            {NodeType::MaskCombine, "Mask Combine", "Logical/arithmetic mask operator", UIWidgets::IconType::MergeVertices},
            {NodeType::MaskMorphology, "Morphology", "Dilate, erode, and morphological blur", UIWidgets::IconType::InflateTool},
            {NodeType::MaskPaint, "Mask Paint", "Interactive brush painting mask", UIWidgets::IconType::PaintTool},
            {NodeType::MaskImage, "Mask Image", "External mask texture file source", UIWidgets::IconType::Assets},
            {NodeType::PaintMaskCombine, "Paint Mask Combine", "Layered paint mask compositor", UIWidgets::IconType::LayerTool},
            {NodeType::GrassMask, "Grass Mask", "Vegetation growth probability map", UIWidgets::IconType::Hair},
            {NodeType::SurfaceMasks, "Surface Masks", "Multi-channel sediment & debris mask", UIWidgets::IconType::ViewSolid},
            {NodeType::CurveToMask, "Curve to Mask", "Metric curve stroke or closed footprint", UIWidgets::IconType::MaskTool}
        }, true},
        {"Data Maps", ImVec4(0.25f, 0.58f, 0.75f, 1.0f), {
            {NodeType::TerrainAnalysis, "Terrain Analysis", "Integrated slope/aspect/flow field", UIWidgets::IconType::Graph},
            {NodeType::BiomeComposer, "Biome Composer", "Multi-field biome distribution map", UIWidgets::IconType::World},
            {NodeType::WetnessMap, "Wetness Map", "Topographic wetness index (TWI)", UIWidgets::IconType::EyedropperTool},
            {NodeType::SoilDepth, "Soil Depth", "Sediment thickness & soil accumulation", UIWidgets::IconType::ClayStripsTool}
        }, true},
        {"Foliage", ImVec4(0.26f, 0.60f, 0.32f, 1.0f), {
            {NodeType::FoliageLayer, "Foliage Layer", "Scatter instance rule layer", UIWidgets::IconType::HairCombTool},
            {NodeType::FoliageSet, "Foliage Set", "Biome vegetation species palette", UIWidgets::IconType::HairClumpTool}
        }, true},
        {"Hydrology", ImVec4(0.18f, 0.62f, 0.82f, 1.0f), {
            {NodeType::RiverLakeEasy, "River & Lake (Easy)", "Unified river network & lake hydrology setup", UIWidgets::IconType::Water},
            {NodeType::WatershedAnalysis, "Watershed", "Drainage basin catchment areas", UIWidgets::IconType::Console},
            {NodeType::LakeBasin, "Lake Basin", "Water surface level & flooding basin", UIWidgets::IconType::FillTool},
            {NodeType::RiverNetwork, "River Network", "Procedural river spline graph", UIWidgets::IconType::AnimGraph},
            {NodeType::RiverHydraulics, "River Hydraulics", "Velocity field & bed shear stress", UIWidgets::IconType::Water},
            {NodeType::RiverBedCarve, "River Bed Carve", "Depression river channel incision", UIWidgets::IconType::Sculpt}
        }, true},
        {"Snow & Ice", ImVec4(0.45f, 0.75f, 0.90f, 1.0f), {
            {NodeType::SnowClimate, "Snow Layer", "Easy snow accumulation & temperature", UIWidgets::IconType::Wind},
            {NodeType::Climate, "Climate", "Temperature and precipitation model", UIWidgets::IconType::Volumetric},
            {NodeType::Snowfall, "Snowfall", "Atmospheric snowfall distribution", UIWidgets::IconType::Volumetric},
            {NodeType::SnowSettle, "Snow Settle", "Avalanching and angle of repose", UIWidgets::IconType::SmudgeTool},
            {NodeType::SnowMeltFreeze, "Melt / Freeze", "Thermal thaw & re-freezing cycle", UIWidgets::IconType::BurnTool},
            {NodeType::GlacierFlow, "Glacier Flow", "Viscous ice movement & glacial carve", UIWidgets::IconType::ElasticDeformTool}
        }, true},
        {"Math", ImVec4(0.82f, 0.72f, 0.32f, 1.0f), {
            {NodeType::Add, "Add", "Sum two height/field inputs", UIWidgets::IconType::AddKey},
            {NodeType::Subtract, "Subtract", "Difference between two field inputs", UIWidgets::IconType::RemoveKey},
            {NodeType::Multiply, "Multiply", "Scale field by coefficient or mask", UIWidgets::IconType::ScaleAxis},
            {NodeType::Blend, "Blend", "Linear interpolation between fields", UIWidgets::IconType::LayerTool},
            {NodeType::Clamp, "Clamp", "Restrict field to min/max range", UIWidgets::IconType::PivotEdit},
            {NodeType::Invert, "Invert", "Invert mask (1.0 - input)", UIWidgets::IconType::ViewSolid}
        }, true},
        {"Blend Modes", ImVec4(0.62f, 0.42f, 0.72f, 1.0f), {
            {NodeType::Overlay, "Overlay", "High contrast blend mode", UIWidgets::IconType::LayerTool},
            {NodeType::Screen, "Screen", "Lightening screen blend mode", UIWidgets::IconType::ViewRendered}
        }, true},
        {"Output", ImVec4(0.92f, 0.32f, 0.32f, 1.0f), {
            {NodeType::HeightOutput, "Height Output", "Primary terrain heightmap output", UIWidgets::IconType::Render},
            {NodeType::SplatOutput, "Splat Output", "Multi-layer texture splat map", UIWidgets::IconType::ViewMatcap},
            {NodeType::SatMapOutput, "SatMap Output", "Textured color composite output", UIWidgets::IconType::ViewRendered},
            {NodeType::HardnessOutput, "Hardness Output", "Structural hardness map export", UIWidgets::IconType::ShadeFlatTool},
            {NodeType::TerrainFieldsOutput, "Terrain Fields", "Multi-channel physical field export", UIWidgets::IconType::Console},
            {NodeType::PublishField, "Publish Field", "Publish one field under your own name for mask consumers", UIWidgets::IconType::Console},
            {NodeType::RoadFieldsOutput, "Road Fields", "Publish one validated road snapshot for materials, hydrology and foliage", UIWidgets::IconType::Console},
            {NodeType::FoliageOutput, "Foliage Output", "Scatter instance transform list", UIWidgets::IconType::HairAddTool},
            {NodeType::LakeSurfaceOutput, "Lake Output", "Water surface geometry mesh output", UIWidgets::IconType::ViewPreview},
            {NodeType::RiverSplineOutput, "River Spline", "River spline path export", UIWidgets::IconType::Gizmo}
        }, true},
        {"Texture", ImVec4(0.82f, 0.62f, 0.22f, 1.0f), {
            {NodeType::AutoSplat, "Auto Splat", "Automatic slope/height material rule", UIWidgets::IconType::PaintTool},
            {NodeType::SplatCompose, "Splat Compose", "Multi-channel splat layer composer", UIWidgets::IconType::LayerTool},
            {NodeType::SurfaceComposer, "Surface Composer", "PBR surface material binder", UIWidgets::IconType::ViewMatcap}
        }, true},
        {"Utility", ImVec4(0.38f, 0.65f, 0.75f, 1.0f), {
            {NodeType::Resample, "Resample", "Grid resolution up/downsampling", UIWidgets::IconType::Sensitivity},
            {NodeType::ChannelExtract, "Channel Extract", "Extract single scalar field component", UIWidgets::IconType::EyedropperTool},
            {NodeType::Transform, "Transform", "Offset/scale/rotate a field in the XZ plane", UIWidgets::IconType::Move}
        }, true},
        {"SatMap", ImVec4(0.88f, 0.38f, 0.68f, 1.0f), {
            {NodeType::SatMapColorRamp, "SatMap ColorRamp", "Multi-stop gradient color ramp", UIWidgets::IconType::EyedropperTool},
            {NodeType::SatMapBlend, "SatMap Blend", "Layered satellite texture color blend", UIWidgets::IconType::LayerTool}
        }, true},
        {"Geology", ImVec4(0.72f, 0.48f, 0.38f, 1.0f), {
            {NodeType::Fault, "Fault Line", "Displacement along tectonic fault line", UIWidgets::IconType::DissolveTopology},
            {NodeType::Mesa, "Mesa / Plateau", "Flat tableland and cliff cap rock", UIWidgets::IconType::FlattenTool},
            {NodeType::Shear, "Shear Zone", "Lateral strike-slip shear deformation", UIWidgets::IconType::NudgeTool},
            {NodeType::PlateTectonics, "Plate Tectonics", "Macro-crustal compression uplift", UIWidgets::IconType::GrabTool},
            {NodeType::Fold, "Fold / Compression", "Syncline & anticline rock folding", UIWidgets::IconType::SnakeHookTool},
            {NodeType::Lithology, "Lithology", "Stratified rock resistance profile", UIWidgets::IconType::ViewMatcap},
            {NodeType::Strata, "Strata", "Sedimentary rock layer bedding", UIWidgets::IconType::FaceMode},
            {NodeType::CraterCaldera, "Crater / Caldera", "Impact crater or volcanic caldera", UIWidgets::IconType::BlobTool}
        }, true}
    };
}


std::vector<std::string> ModernTerrainNodePalette::getCategoryNames() const {
    std::vector<std::string> names;
    names.reserve(categories_.size());
    for (const auto& cat : categories_) {
        names.push_back(cat.name);
    }
    return names;
}

std::vector<std::string> ModernTerrainNodePalette::getNodeNamesInCategory(const std::string& categoryName) const {
    std::vector<std::string> names;
    for (const auto& cat : categories_) {
        if (cat.name == categoryName) {
            for (const auto& n : cat.nodes) {
                names.push_back(n.label);
            }
            break;
        }
    }
    return names;
}

void ModernTerrainNodePalette::renderNodeCard(const ModernNodeCategory& category, const ModernNodeDescriptor& nodeDesc, float itemWidth) {
    ImGui::PushID(static_cast<int>(nodeDesc.type));

    const ImVec2 cardSize(itemWidth, itemWidth); // Square Icon Card
    const ImVec2 cursorScreen = ImGui::GetCursorScreenPos();
    
    // Check hover / active state
    ImGui::InvisibleButton(nodeDesc.label, cardSize);
    const bool isHovered = ImGui::IsItemHovered();
    const bool isActive = ImGui::IsItemActive();

    ImDrawList* drawList = ImGui::GetWindowDrawList();
    const ImVec2 pMin = cursorScreen;
    const ImVec2 pMax = ImVec2(pMin.x + cardSize.x, pMin.y + cardSize.y);

    // Dynamic Theme Manager Color Integration
    ThemeManager& tm = ThemeManager::instance();
    const ImVec4 catColor = tm.getCategoryColor(category.name, category.defaultColor);
    const ImU32 stripeCol = ImGui::ColorConvertFloat4ToU32(catColor);

    const ImVec4 currentSurface = tm.current().colors.surface;
    const ImU32 bgCol = isHovered 
        ? ImGui::ColorConvertFloat4ToU32(ImVec4(currentSurface.x + 0.08f, currentSurface.y + 0.08f, currentSurface.z + 0.10f, 0.95f))
        : ImGui::ColorConvertFloat4ToU32(ImVec4(currentSurface.x * 0.85f, currentSurface.y * 0.85f, currentSurface.z * 0.90f, 0.60f));

    const ImU32 borderCol = isHovered 
        ? stripeCol 
        : ImGui::ColorConvertFloat4ToU32(ImVec4(0.30f, 0.30f, 0.35f, 0.35f));

    // Card background & rounded border
    drawList->AddRectFilled(pMin, pMax, bgCol, 4.0f);
    drawList->AddRect(pMin, pMax, borderCol, 4.0f, 0, isHovered ? 1.5f : 1.0f);

    // Top accent category color bar (instead of left)
    drawList->AddRectFilled(pMin, ImVec2(pMax.x, pMin.y + 4.0f), stripeCol, 4.0f, ImDrawFlags_RoundCornersTop);

    // Badge Icon (Large, Centered, Programmatic)
    const ImU32 badgeTextCol = IM_COL32(230, 230, 235, 220);
    const float iconDrawSize = cardSize.x * 0.46f;
    const ImVec2 iconDrawPos(pMin.x + (cardSize.x - iconDrawSize) * 0.5f,
                             pMin.y + cardSize.y * 0.24f);
    UIWidgets::DrawIcon(nodeDesc.iconType, iconDrawPos, iconDrawSize, stripeCol, 1.8f);

    // Node Label (Small, at the bottom, truncated if necessary)
    std::string shortLabel = nodeDesc.label;
    const float maxLabelWidth = cardSize.x - 8.0f;
    ImVec2 labelSize = ImGui::CalcTextSize(shortLabel.c_str());
    if (labelSize.x > maxLabelWidth && shortLabel.length() > 3) {
        // Simple truncation
        while (labelSize.x > maxLabelWidth && shortLabel.length() > 3) {
            shortLabel.pop_back();
            labelSize = ImGui::CalcTextSize((shortLabel + "..").c_str());
        }
        shortLabel += "..";
    }
    const ImVec2 labelPos(pMin.x + (cardSize.x - labelSize.x) * 0.5f, pMax.y - labelSize.y - 6.0f);
    drawList->AddText(labelPos, badgeTextCol, shortLabel.c_str());

    // Drag and Drop payload source
    if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)) {
        ImGui::SetDragDropPayload("TERRAIN_NODE_TYPE", &nodeDesc.type, sizeof(NodeType));

        // Modern Drag Tooltip Card Preview
        ImGui::PushStyleColor(ImGuiCol_PopupBg, ImVec4(0.12f, 0.14f, 0.18f, 0.95f));
        ImGui::BeginTooltip();
        
        ImGui::TextColored(catColor, "%s", nodeDesc.label);
        ImGui::TextDisabled("%s", nodeDesc.description);
        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 0.85f), "Drop onto graph canvas or cable link");
        
        ImGui::EndTooltip();
        ImGui::PopStyleColor();
        
        ImGui::EndDragDropSource();
    }

    // Tooltip on hover (when not dragging)
    if (isHovered && !ImGui::IsMouseDragging(0)) {
        ImGui::SetTooltip("%s\nCategory: %s", nodeDesc.description, category.name.c_str());
    }

    ImGui::PopID();
}

void ModernTerrainNodePalette::render(TerrainNodeGraphV2& graph, TerrainObject* terrain, char* searchBuffer, size_t searchBufferSize) {
    (void)graph;
    (void)terrain;

    ThemeManager& tm = ThemeManager::instance();
    const ImVec4 textMuted = tm.current().colors.textMuted;
    const ImVec4 primary = tm.current().colors.primary;

    // Header Title
    ImGui::TextColored(ImVec4(0.85f, 0.90f, 0.95f, 1.0f), "Node Palette");
    ImGui::SameLine();
    ImGui::TextColored(textMuted, "(Library)");
    ImGui::Separator();

    // Search bar with clear button
    ImGui::PushItemWidth(-1);
    ImGui::InputTextWithHint("##TerrainNodeSearch", "Search nodes...", searchBuffer, searchBufferSize);
    ImGui::PopItemWidth();
    ImGui::Spacing();

    std::string query = searchBuffer ? searchBuffer : "";
    std::transform(query.begin(), query.end(), query.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    // Check if any category popup is currently open to enable "hover-to-open" mode
    bool anyCategoryOpen = false;
    for (int i = 0; i < static_cast<int>(categories_.size()); ++i) {
        ImGui::PushID(i);
        if (ImGui::IsPopupOpen("CatPopup")) {
            anyCategoryOpen = true;
        }
        ImGui::PopID();
    }

    // ----------------------------------------------------
    // POPUP MENU LAYOUT (Gaea 2 Style)
    // ----------------------------------------------------
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 4));
    
    for (int i = 0; i < static_cast<int>(categories_.size()); ++i) {
        auto& category = categories_[i];
        const ImVec4 catColor = tm.getCategoryColor(category.name, category.defaultColor);
        const ImU32 color32 = ImGui::ColorConvertFloat4ToU32(catColor);
        
        // Count matching nodes in category
        int matchCount = 0;
        for (const auto& nodeDesc : category.nodes) {
            std::string labelLower = nodeDesc.label;
            std::string descLower = nodeDesc.description;
            std::transform(labelLower.begin(), labelLower.end(), labelLower.begin(), ::tolower);
            std::transform(descLower.begin(), descLower.end(), descLower.begin(), ::tolower);

            if (query.empty() || labelLower.find(query) != std::string::npos || descLower.find(query) != std::string::npos) {
                matchCount++;
            }
        }

        if (!query.empty() && matchCount == 0) {
            continue; // Skip categories with no search hits
        }

        ImGui::PushID(i);
        
        // Category Button Layout (Full width of the docked panel)
        const ImVec2 catSize(ImGui::GetContentRegionAvail().x, 32.0f);
        const ImVec2 catPos = ImGui::GetCursorScreenPos();
        
        if (ImGui::InvisibleButton("##catbtn", catSize)) {
            ImGui::OpenPopup("CatPopup");
        }
        // Allow hover detection even if another category's popup is currently blocking the window
        bool isHovered = ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenBlockedByPopup);
        
        // Auto-open menu on hover ONLY if the user has already clicked to open a menu
        if (isHovered && anyCategoryOpen && !ImGui::IsPopupOpen("CatPopup")) {
            ImGui::OpenPopup("CatPopup");
        }
        
        bool isPopupOpen = ImGui::IsPopupOpen("CatPopup");
        
        ImDrawList* drawList = ImGui::GetWindowDrawList();
        
        // Background: Subtle highlight on hover or open
        ImU32 bgCol = isPopupOpen ? ImGui::ColorConvertFloat4ToU32(ImVec4(catColor.x, catColor.y, catColor.z, 0.2f)) : 
                      isHovered   ? ImGui::ColorConvertFloat4ToU32(ImVec4(1.0f, 1.0f, 1.0f, 0.08f)) : 
                                    IM_COL32(0, 0, 0, 0);
        drawList->AddRectFilled(catPos, ImVec2(catPos.x + catSize.x, catPos.y + catSize.y), bgCol, 4.0f);
        
        // Accent line if popup is open
        if (isPopupOpen) {
            drawList->AddRectFilled(catPos, ImVec2(catPos.x + 3.0f, catPos.y + catSize.y), color32, 4.0f, ImDrawFlags_RoundCornersLeft);
        }
        
        // Draw category name (Elegant text, no big letters)
        ImVec2 textPos(catPos.x + (isPopupOpen ? 12.0f : 8.0f), catPos.y + (catSize.y - ImGui::GetTextLineHeight()) * 0.5f);
        drawList->AddText(textPos, isHovered || isPopupOpen ? color32 : IM_COL32(200, 200, 205, 255), category.name.c_str());
        
        // Draw ">" arrow indicator on the right
        const char* arrow = ">";
        ImVec2 arrowSize = ImGui::CalcTextSize(arrow);
        drawList->AddText(ImVec2(catPos.x + catSize.x - arrowSize.x - 8.0f, textPos.y), IM_COL32(120, 120, 125, 255), arrow);
        
        // Estimate popup height to prevent clipping at the bottom of the screen
        float estHeight = category.nodes.size() * 22.0f + 40.0f; // 22px per node + 40px for header/padding
        float popY = catPos.y;
        float screenBottom = ImGui::GetIO().DisplaySize.y - 10.0f; // 10px safety margin
        if (popY + estHeight > screenBottom) {
            // Shift the popup upwards so it fits on screen, but don't go above the top of the screen (10px margin)
            popY = std::max(10.0f, screenBottom - estHeight);
        }
        
        ImGui::SetNextWindowPos(ImVec2(catPos.x + catSize.x + 4.0f, popY));
        ImGui::SetNextWindowSizeConstraints(ImVec2(170.0f, 0.0f), ImVec2(FLT_MAX, FLT_MAX)); // Reduced width slightly
        ImGui::PushStyleColor(ImGuiCol_PopupBg, ImVec4(0.12f, 0.12f, 0.13f, 0.98f));
        ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(catColor.x * 0.4f, catColor.y * 0.4f, catColor.z * 0.4f, 0.8f));
        
        if (ImGui::BeginPopup("CatPopup")) {
            // Auto-close if mouse leaves the popup and button area
            // ImGui usually handles closing on click outside, but to be robust with hover leaving:
            if (!ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow | ImGuiHoveredFlags_AllowWhenBlockedByPopup)) {
                // To prevent immediate close, wait for a click, or use standard ImGui behavior.
                // Standard ImGui behavior (click outside to close) is preferred for accessibility.
            }

            ImGui::TextColored(catColor, "%s", category.name.c_str());
            ImGui::Separator();
            
            for (const auto& nodeDesc : category.nodes) {
                // Search filter check for items inside the popup
                std::string labelLower = nodeDesc.label;
                std::string descLower = nodeDesc.description;
                std::transform(labelLower.begin(), labelLower.end(), labelLower.begin(), ::tolower);
                std::transform(descLower.begin(), descLower.end(), descLower.begin(), ::tolower);
                if (!query.empty() && labelLower.find(query) == std::string::npos && descLower.find(query) == std::string::npos) {
                    continue;
                }
                
                ImGui::PushID(static_cast<int>(nodeDesc.type));
                
                // Draw as a Selectable to ensure Drag and Drop works flawlessly without auto-closing issues
                
                // Draw icon inline before the selectable label
                {
                    const ImVec2 curPos = ImGui::GetCursorScreenPos();
                    const float iconSz = ImGui::GetTextLineHeight() * 1.30f;
                    UIWidgets::DrawIcon(nodeDesc.iconType, curPos, iconSz,
                        ImGui::ColorConvertFloat4ToU32(catColor), 1.5f);
                    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + iconSz + 6.0f);
                }

                // Selectable for the hit box and drag source
                bool clicked = ImGui::Selectable(nodeDesc.label, false, ImGuiSelectableFlags_DontClosePopups | ImGuiSelectableFlags_AllowOverlap);
                
                // Attach Drag and Drop IMMEDIATELY to the Selectable
                if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)) {
                    ImGui::SetDragDropPayload("TERRAIN_NODE_TYPE", &nodeDesc.type, sizeof(NodeType));
                    ImGui::TextColored(catColor, "%s", nodeDesc.label);
                    ImGui::TextDisabled("%s", nodeDesc.description);
                    ImGui::EndDragDropSource();
                }

                if (clicked) {
                    if (onNodeClicked) {
                        onNodeClicked(nodeDesc.type);
                    }
                    ImGui::CloseCurrentPopup();
                }
                
                if (ImGui::IsItemHovered() && !ImGui::IsMouseDragging(0)) {
                    ImGui::SetTooltip("%s", nodeDesc.description);
                }
                
                ImGui::PopID();
            }
            ImGui::EndPopup();
        }
        
        ImGui::PopStyleColor(2);
        ImGui::PopID();
    }
    
    ImGui::PopStyleVar();
}

} // namespace TerrainNodesV2
