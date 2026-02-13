/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          HairUI.h
 * Description:   ImGui-based Hair/Fur editing panel
 * =========================================================================
 */
#ifndef HAIR_UI_H
#define HAIR_UI_H

#include "Hair/HairSystem.h"
#include "Hair/HairBSDF.h"
#include <imgui.h>
#include <functional>
#include <string>
#include <vector>
#include <memory>
#include "Renderer.h"

// Forward declarations
class Triangle;


namespace Hair {

/**
 * @brief Hair/Fur editing UI panel
 * 
 * Features:
 *   - Groom creation/deletion
 *   - Generation parameters (length, density, curl, etc.)
 *   - Hair material editing (color, melanin, roughness)
 *   - Paint mode (add, remove, cut, comb brushes)
 *   - Preview controls
 */


/**
 * @brief Paint brush mode for interactive hair editing
 */
enum class HairPaintMode {
    NONE,       // Not painting
    ADD,        // Add new strands
    REMOVE,     // Remove strands
    CUT,        // Cut strands shorter
    COMB,       // Comb/style direction
    LENGTH,     // Adjust length
    DENSITY,    // Adjust local density
    CLUMP,      // Adjust clumpiness
    PUFF,       // Add volume/puff
    WAVE,       // Add waves/curls procedural
    FRIZZ,      // Add random detail/frizz
    SMOOTH,     // Straighten/relax hair
    PINCH,      // Bunch points together
    SPREAD      // Spread points apart
};

// ODR-safe helper for brush mode info
struct HairBrushModeInfo {
    HairPaintMode mode;
    const char* name;
    const char* icon;
    const char* tooltip;
};

/**
 * @brief Paint brush settings
 */
struct HairBrushSettings {
    float radius = 0.05f;       // Brush radius in world units
    float strength = 0.5f;      // Effect strength (0-1)
    float falloff = 0.5f;       // Falloff curve (0=hard, 1=soft)
    bool affectGuides = true;   // Affect guide strands
    bool affectChildren = false; // Affect child strands (slower)
    
    // Cut mode specific
    float cutLength = 0.5f;     // Relative cut position (0=root, 1=tip)
    bool cutAtHitPoint = false; // Scissors mode: cut exactly where clicked
    
    // Comb mode specific  
    Vec3 combDirection = Vec3(0, -1, 0);
    bool useViewDirection = true; // Use camera as comb direction
    
    // Procedural (Wave/Curl/Frizz)
    float frequency = 1.0f;
    float amplitude = 0.5f;

    // Mirroring
    bool mirrorX = false;
    bool mirrorY = false;
    bool mirrorZ = false;
};


class HairUI {
public:
    // File dialog callback (injected by SceneUI)
    std::function<std::string(const wchar_t*)> onOpenFileDialog;

    HairUI();
    ~HairUI() = default;
    
    /**
     * @brief Render the Hair panel in ImGui
     * @param hairSystem Reference to the hair system
     * @param selectedMeshTriangles Currently selected mesh (for hair generation target)
     */
    void render(
        HairSystem& hairSystem,
        const std::vector<std::shared_ptr<Triangle>>* selectedMeshTriangles = nullptr,
        Renderer* renderer = nullptr,
        std::function<void()> onPreGenerate = nullptr
    );
    
    /**
     * @brief Get hair material for rendering
     */
    const HairMaterialParams& getMaterial() const { return m_currentMaterial; }
    
    /**
     * @brief Check if any parameter changed (for BVH rebuild)
     */
    bool isDirty() const { return m_needsRebuild; }
    void clearDirty() { m_needsRebuild = false; }
    void clear() {
        m_selectedGroomName = "";
        m_lastSelectedMeshName = "";
        m_groomNameBuffer[0] = '\0';
        m_paintMode = HairPaintMode::NONE;
        
        // Reset to reasonable defaults
        m_editParams = HairGenerationParams(); 
        m_brushSettings = HairBrushSettings();
        m_currentMaterial = HairMaterialParams();
        m_needsRebuild = false;
        m_showAdvanced = false;
        m_previewMode = false;
        m_hideChildrenDuringPaint = true;

        // Restore defaults
        m_editParams.guideCount = 5000;
        m_editParams.interpolatedPerGuide = 3;
        m_editParams.pointsPerStrand = 8;
        m_editParams.length = 0.05f;
        m_editParams.lengthVariation = 0.2f;
        m_editParams.rootRadius = 0.0005f;
        m_editParams.tipRadius = 0.0001f;
        m_editParams.clumpiness = 0.5f;
        m_editParams.useBSpline = true;
        m_editParams.subdivisions = 3;
        
        m_currentMaterial.colorMode = HairMaterialParams::ColorMode::DIRECT_COLORING;
        m_currentMaterial.color = Vec3(0.35f, 0.22f, 0.12f);
        m_currentMaterial.roughness = 0.3f;
        m_currentMaterial.radialRoughness = 0.3f;
        m_currentMaterial.ior = 1.55f;

        m_presets.clear();
        initializePresets();
    }

    
    /**
     * @brief Paint mode access for viewport interaction
     */
    HairPaintMode getPaintMode() const { return m_paintMode; }
    bool isPainting() const { return m_paintMode != HairPaintMode::NONE; }
    bool shouldHideChildren() const { return m_paintMode != HairPaintMode::NONE && m_hideChildrenDuringPaint; }
    const HairBrushSettings& getBrushSettings() const { return m_brushSettings; }
    
    HairGroom* getSelectedGroom(HairSystem& sys) {
        if (m_selectedGroomName.empty()) return nullptr;
        return sys.getGroom(m_selectedGroomName);
    }

    const HairGroom* getSelectedGroom(const HairSystem& sys) const {
        if (m_selectedGroomName.empty()) return nullptr;
        return sys.getGroom(m_selectedGroomName);
    }
    
    /**
     * @brief Apply brush at world position (called from viewport)
     */
    void applyBrush(HairSystem& hairSystem, const Vec3& worldPos, const Vec3& normal, float deltaTime, 
                    const Vec3& customDir = Vec3(0, 0, 0));
    
    // Set an optional surface projector for the brush (e.g. for snapping to complex geometry)
    void setSurfaceProjector(std::function<bool(Vec3&, Vec3&)> projector) { m_projector = projector; }

    // Helper to validate if the hit surface matches the groom's target
    bool isSurfaceValid(const HairSystem& sys, const std::string& hitMeshName) const {
        if (m_selectedGroomName.empty()) return false;
        const HairGroom* g = sys.getGroom(m_selectedGroomName);
        return g && g->boundMeshName == hitMeshName;
    }

    bool isGroomValid(const std::string& groomName) const {
        return !m_selectedGroomName.empty() && m_selectedGroomName == groomName;
    }
    
private:
    // Current editing state
    std::string m_selectedGroomName;
    HairGenerationParams m_editParams;
    HairMaterialParams m_currentMaterial;
    
    // Paint mode state
    HairPaintMode m_paintMode = HairPaintMode::NONE;
    HairBrushSettings m_brushSettings;
    
    // UI state
    bool m_needsRebuild = false;
    bool m_showAdvanced = false;
    bool m_previewMode = false;
    bool m_hideChildrenDuringPaint = true;
    HairPaintMode m_lastPaintMode = HairPaintMode::NONE;
    int m_currentTab = 0; // 0=Gen, 1=Mat, 2=Paint, 3=Presets

    std::function<bool(Vec3&, Vec3&)> m_projector = nullptr;
    
    // Presets
    struct HairPreset {
        std::string name;
        HairGenerationParams genParams;
        HairMaterialParams matParams;
    };
    std::vector<HairPreset> m_presets;
    
    // Internal methods
    void drawGenerationPanel(HairSystem& hairSystem, 
                            const std::vector<std::shared_ptr<Triangle>>* triangles,
                            Renderer* renderer = nullptr,
                            std::function<void()> onPreGenerate = nullptr);
    void drawMaterialPanel(Renderer* renderer = nullptr);
    void drawGroomList(HairSystem& hairSystem, Renderer* renderer = nullptr);

    void drawPresets();
    void drawPaintPanel(HairSystem& hairSystem);
    void drawStats(const HairSystem& hairSystem);
    
    void initializePresets();
    void applyPreset(const HairPreset& preset);
    void syncToGroom(HairSystem& hairSystem);
    void applyBrushInternal(HairSystem& hairSystem, const Vec3& pos, const Vec3& normal, 
                            float effectStrength, float radius, const Vec3& combDir, float deltaTime);


    
    // Tracking state
    std::string m_lastSelectedMeshName;
    char m_groomNameBuffer[64];
};


// ============================================================================
// Implementation
// ============================================================================

inline HairUI::HairUI() {
    // Default generation params
    m_editParams.guideCount = 5000;
    m_editParams.interpolatedPerGuide = 3;
    m_editParams.pointsPerStrand = 8;
    m_editParams.length = 0.05f;
    m_editParams.lengthVariation = 0.2f;
    m_editParams.rootRadius = 0.0005f;
    m_editParams.tipRadius = 0.0001f;
    m_editParams.clumpiness = 0.5f;
    m_editParams.curlFrequency = 0.0f;
    m_editParams.curlRadius = 0.01f;
    m_editParams.frizz = 0.0f;
    m_editParams.gravity = 0.3f;
    m_editParams.useBSpline = true;
    m_editParams.subdivisions = 3;  // High quality by default (8x tessellation)
    
    // Default material (medium brown hair - visible default)
    m_currentMaterial.colorMode = HairMaterialParams::ColorMode::DIRECT_COLORING;
    m_currentMaterial.color = Vec3(0.35f, 0.22f, 0.12f); // Medium brown, much more visible
    m_currentMaterial.roughness = 0.3f;
    m_currentMaterial.radialRoughness = 0.3f;
    m_currentMaterial.ior = 1.55f;
    
    m_groomNameBuffer[0] = '\0';
    initializePresets();
}


inline void HairUI::initializePresets() {
    // ═══════════════════════════════════════════════════════════════════════════
    // HUMAN HAIR PRESETS
    // ═══════════════════════════════════════════════════════════════════════════
    {
        HairPreset preset;
        preset.name = "Blonde Hair";
        preset.genParams = m_editParams;
        preset.genParams.length = 0.15f;
        preset.matParams.colorMode = HairMaterialParams::ColorMode::MELANIN;
        preset.matParams.melanin = 0.15f;
        preset.matParams.melaninRedness = 0.2f;
        preset.matParams.roughness = 0.25f;
        m_presets.push_back(preset);
    }
    {
        HairPreset preset;
        preset.name = "Brown Hair";
        preset.genParams = m_editParams;
        preset.genParams.length = 0.15f;
        preset.matParams.colorMode = HairMaterialParams::ColorMode::MELANIN;
        preset.matParams.melanin = 0.55f;
        preset.matParams.melaninRedness = 0.35f;
        preset.matParams.roughness = 0.3f;
        m_presets.push_back(preset);
    }
    {
        HairPreset preset;
        preset.name = "Black Hair";
        preset.genParams = m_editParams;
        preset.genParams.length = 0.15f;
        preset.matParams.colorMode = HairMaterialParams::ColorMode::MELANIN;
        preset.matParams.melanin = 0.98f;
        preset.matParams.melaninRedness = 0.05f;
        preset.matParams.roughness = 0.35f;
        m_presets.push_back(preset);
    }
    {
        HairPreset preset;
        preset.name = "Red Hair";
        preset.genParams = m_editParams;
        preset.genParams.length = 0.15f;
        preset.matParams.colorMode = HairMaterialParams::ColorMode::MELANIN;
        preset.matParams.melanin = 0.35f;
        preset.matParams.melaninRedness = 0.95f;
        preset.matParams.roughness = 0.28f;
        m_presets.push_back(preset);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // ARCHITECTURAL / NATURE PRESETS
    // ═══════════════════════════════════════════════════════════════════════════
    {
        HairPreset preset;
        preset.name = "Luxury Carpet";
        preset.genParams = m_editParams;
        preset.genParams.length = 0.012f;
        preset.genParams.guideCount = 120000; 
        preset.genParams.interpolatedPerGuide = 4;
        preset.genParams.clumpiness = -0.15f; // Slightly spread tips for "shag" look
        preset.genParams.frizz = 0.25f;       // Technical fiber jitter
        preset.genParams.gravity = 0.0f; 
        preset.genParams.roughness = 0.1f;    // Global noise low
        preset.matParams.colorMode = HairMaterialParams::ColorMode::ROOT_UV_MAP;
        preset.matParams.roughness = 0.7f;
        preset.matParams.radialRoughness = 0.8f;
        m_presets.push_back(preset);
    }
    {
        HairPreset preset;
        preset.name = "Wild Grass";
        preset.genParams = m_editParams;
        preset.genParams.length = 0.12f;
        preset.genParams.lengthVariation = 0.6f;
        preset.genParams.guideCount = 15000;
        preset.genParams.interpolatedPerGuide = 8;
        preset.genParams.rootRadius = 0.002f;
        preset.genParams.tipRadius = 0.0004f;
        preset.genParams.clumpiness = -0.4f;  // Spread at tips like grass blades
        preset.genParams.waveFrequency = 8.0f;
        preset.genParams.waveAmplitude = 0.02f;
        preset.genParams.frizz = 0.1f;
        preset.genParams.gravity = 0.3f;
        preset.matParams.colorMode = HairMaterialParams::ColorMode::ROOT_UV_MAP;
        preset.matParams.roughness = 0.5f;
        m_presets.push_back(preset);
    }
    {
        HairPreset preset;
        preset.name = "Dense Moss";
        preset.genParams = m_editParams;
        preset.genParams.length = 0.005f;
        preset.genParams.guideCount = 150000;
        preset.genParams.interpolatedPerGuide = 2;
        preset.genParams.frizz = 0.4f;
        preset.genParams.clumpiness = 0.1f;
        preset.matParams.colorMode = HairMaterialParams::ColorMode::ROOT_UV_MAP;
        preset.matParams.color = Vec3(0.1f, 0.3f, 0.05f);
        m_presets.push_back(preset);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // ANIMAL FUR PRESETS
    // ═══════════════════════════════════════════════════════════════════════════
    {
        HairPreset preset;
        preset.name = "Velvet Fur";
        preset.genParams = m_editParams;
        preset.genParams.length = 0.008f;
        preset.genParams.guideCount = 40000;
        preset.genParams.gravity = 0.05f;
        preset.matParams.colorMode = HairMaterialParams::ColorMode::MELANIN;
        preset.matParams.melanin = 0.6f;
        preset.matParams.roughness = 0.4f;
        preset.matParams.coat = 0.2f;
        m_presets.push_back(preset);
    }
    {
        HairPreset preset;
        preset.name = "Long Wolf Fur";
        preset.genParams = m_editParams;
        preset.genParams.length = 0.08f;
        preset.genParams.guideCount = 12000;
        preset.genParams.gravity = 0.7f;
        preset.genParams.frizz = 0.2f;
        preset.genParams.clumpiness = 0.6f;
        preset.matParams.colorMode = HairMaterialParams::ColorMode::MELANIN;
        preset.matParams.melanin = 0.4f;
        preset.matParams.melaninRedness = 0.2f;
        preset.matParams.roughness = 0.5f;
        preset.matParams.coat = 0.5f;
        m_presets.push_back(preset);
    }
}

inline void HairUI::applyPreset(const HairPreset& preset) {
    m_editParams = preset.genParams;
    m_currentMaterial = preset.matParams;
    m_needsRebuild = true;
}

inline void HairUI::syncToGroom(HairSystem& hairSystem) {
    if (m_selectedGroomName.empty()) return;
    if (HairGroom* g = hairSystem.getGroom(m_selectedGroomName)) {
        g->params = m_editParams;
        g->material = m_currentMaterial;
    }
}


inline void HairUI::render(
    HairSystem& hairSystem,
    const std::vector<std::shared_ptr<Triangle>>* selectedMeshTriangles,
    Renderer* renderer,
    std::function<void()> onPreGenerate
) {

    // 1. Handle Selection Change
    if (selectedMeshTriangles && !selectedMeshTriangles->empty()) {
        std::string meshName = (*selectedMeshTriangles)[0]->getNodeName();
        if (meshName != m_lastSelectedMeshName) {
            m_lastSelectedMeshName = meshName;
            
            // Try to find existing groom for this mesh
            // Only auto-switch if strictly necessary (different mesh)
            bool currentGroomBelongsToNewMesh = false;
            if (!m_selectedGroomName.empty()) {
                if (HairGroom* g = hairSystem.getGroom(m_selectedGroomName)) {
                    if (g->boundMeshName == meshName) currentGroomBelongsToNewMesh = true;
                }
            }

            if (!currentGroomBelongsToNewMesh) {
                HairGroom* existing = hairSystem.getGroomByMesh(meshName);
                if (existing) {
                    m_selectedGroomName = existing->name;
                    m_editParams = existing->params;
                    m_currentMaterial = existing->material;
                    if (renderer) renderer->setHairMaterial(m_currentMaterial);
                    strncpy(m_groomNameBuffer, existing->name.c_str(), sizeof(m_groomNameBuffer));
                }
            }


                // Suggest a new name
                std::string suggested = meshName + "_Hair";
                strncpy(m_groomNameBuffer, suggested.c_str(), sizeof(m_groomNameBuffer));
            
        }
    } else if (selectedMeshTriangles && selectedMeshTriangles->empty()) {
        m_lastSelectedMeshName = "";
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // MODERN NARROW SIDEBAR (Icon Based)
    // ═══════════════════════════════════════════════════════════════════════════
    
    float sidebar_w = 42.0f;
    ImGui::BeginChild("HairSidebar", ImVec2(sidebar_w, 0), true, ImGuiWindowFlags_NoScrollbar);
    
    auto DrawIconButton = [&](int index, UIWidgets::IconType icon, const char* tooltip) {
        bool selected = (m_currentTab == index);
        ImVec2 pos = ImGui::GetCursorScreenPos();
        float size = 32.0f;
        float margin = (sidebar_w - size) * 0.5f;
        
        ImGui::SetCursorPosX(margin);
        ImGui::PushID(index);
        
        if (selected) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(1, 1, 1, 0.08f));
        } else {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
        }

        if (ImGui::Button("##tab", ImVec2(size, size))) {
            m_currentTab = index;
        }
        
        // Draw the modern icon
        ImU32 iconCol = selected ? ImGui::ColorConvertFloat4ToU32(ImVec4(0.4f, 0.8f, 1.0f, 1.0f)) 
                                 : ImGui::ColorConvertFloat4ToU32(ImVec4(0.7f, 0.7f, 0.7f, 0.8f));
        
        UIWidgets::DrawIcon(icon, ImVec2(pos.x + size*0.5f, pos.y + size*0.5f), size * 0.6f, iconCol);
        
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tooltip);
        
        ImGui::PopStyleColor();
        ImGui::PopID();
        ImGui::Spacing();
    };

    DrawIconButton(0, UIWidgets::IconType::Mesh,    "Groom Management");
    DrawIconButton(1, UIWidgets::IconType::Render,  "Hair Material");
    DrawIconButton(2, UIWidgets::IconType::Magnet,  "Interactive Paint");
    DrawIconButton(3, UIWidgets::IconType::Graph,   "Presets & Styles");
    
    ImGui::EndChild();

    ImGui::SameLine();

    // Main Content Area
    ImGui::BeginChild("HairContent", ImVec2(0, 0), false);
    
    switch (m_currentTab) {
        case 0: drawGenerationPanel(hairSystem, selectedMeshTriangles, renderer, onPreGenerate); break;
        case 1: drawMaterialPanel(renderer); break;
        case 2: drawPaintPanel(hairSystem); break;
        case 3: drawPresets(); break;
    }
    
    ImGui::EndChild();
    
    // Show paint mode indicator if active
    if (m_paintMode != HairPaintMode::NONE) {
        ImGui::Separator();
        const char* modeNames[] = { 
            "None", "Add", "Remove", "Cut", "Comb", "Length", "Density", "Clump", "Puff",
            "Wave", "Frizz", "Smooth", "Pinch", "Spread"
        };
        ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.2f, 1.0f), 
                          "PAINT MODE: %s (ESC to exit)", modeNames[static_cast<int>(m_paintMode)]);
    }
    
    ImGui::Separator();
    drawStats(hairSystem);
    
    // Final sync of any UI edits back to the underlying groom model
    syncToGroom(hairSystem);
}


inline void HairUI::drawGenerationPanel(
    HairSystem& hairSystem,
    const std::vector<std::shared_ptr<Triangle>>* triangles,
    Renderer* renderer,
    std::function<void()> onPreGenerate
) {
    ImGui::Text("Target Mesh: %s", 
                triangles && !triangles->empty() ? "Selected" : "None (select mesh first)");
    
    ImGui::Separator();
    
    bool structuralChanged = false;
    bool stylingChanged = false;

    // Strand Count
    ImGui::Text("Strand Density");
    int guideCount = static_cast<int>(m_editParams.guideCount);
    if (ImGui::SliderInt("Guide Strands", &guideCount, 100, 100000)) {
        m_editParams.guideCount = static_cast<uint32_t>(guideCount);
        structuralChanged = true;
    }
    
    int interpolated = static_cast<int>(m_editParams.interpolatedPerGuide);
    // Increased range to support dense fur (DragInt allows manual entry > 128 if needed)
    if (ImGui::DragInt("Children per Guide", &interpolated, 0.5f, 0, 128)) {
        m_editParams.interpolatedPerGuide = static_cast<uint32_t>(interpolated);
        stylingChanged = true;
    }
    
    int points = static_cast<int>(m_editParams.pointsPerStrand);
    if (ImGui::SliderInt("Points per Strand", &points, 2, 16)) {
        m_editParams.pointsPerStrand = static_cast<uint32_t>(points);
        structuralChanged = true;
    }

    
    ImGui::Separator();
    ImGui::Text("Physical Properties");
    
    // Length
    float length_cm = m_editParams.length * 100.0f;
    if (ImGui::SliderFloat("Length (cm)", &length_cm, 0.1f, 100.0f)) {
        m_editParams.length = length_cm / 100.0f;
        stylingChanged = true;
    }
    
    if (ImGui::SliderFloat("Length Variation", &m_editParams.lengthVariation, 0.0f, 1.0f)) stylingChanged = true;
    
    // Radius
    float rootRadius_mm = m_editParams.rootRadius * 1000.0f;
    if (ImGui::SliderFloat("Root Radius (mm)", &rootRadius_mm, 0.01f, 2.0f)) {
        m_editParams.rootRadius = rootRadius_mm / 1000.0f;
        stylingChanged = true;
    }
    
    float tipRadius_mm = m_editParams.tipRadius * 1000.0f;
    if (ImGui::SliderFloat("Tip Radius (mm)", &tipRadius_mm, 0.001f, 1.0f)) {
        m_editParams.tipRadius = tipRadius_mm / 1000.0f;
        stylingChanged = true;
    }

    
    ImGui::Separator();
    ImGui::Text("Styling & Shaping");
    
    // Child Radius (Standard control in XGen/Blender)
    float childRadius_mm = m_editParams.childRadius * 1000.0f;
    if (ImGui::SliderFloat("Child Spread (mm)", &childRadius_mm, 0.1f, 100.0f)) {
        m_editParams.childRadius = childRadius_mm / 1000.0f;
        stylingChanged = true;
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Uniform spread radius of children around guides at clumpiness = 0");

    // Better clumpiness (-1 to 1)
    if (ImGui::SliderFloat("Clumpiness", &m_editParams.clumpiness, -1.0f, 1.0f, "%.2f")) stylingChanged = true;
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Positive: Tips converge to guide. Negative: Tips flare out.");

    if (ImGui::SliderFloat("Gravity", &m_editParams.gravity, 0.0f, 2.0f)) stylingChanged = true;
    
    // Curly / Wavy support
    if (ImGui::CollapsingHeader("Waves & Curls")) {
        if (ImGui::SliderFloat("Curl Freq", &m_editParams.curlFrequency, 0.0f, 50.0f)) stylingChanged = true;
        if (ImGui::SliderFloat("Curl Radius", &m_editParams.curlRadius, 0.0f, 0.1f)) stylingChanged = true;
        
        ImGui::Separator();
        
        if (ImGui::SliderFloat("Wave Freq", &m_editParams.waveFrequency, 0.0f, 50.0f)) stylingChanged = true;
        if (ImGui::SliderFloat("Wave Amp", &m_editParams.waveAmplitude, 0.0f, 0.1f)) stylingChanged = true;
    }


    if (ImGui::CollapsingHeader("Noise & Roughness")) {
        if (ImGui::SliderFloat("Roughness", &m_editParams.roughness, 0.0f, 1.0f)) stylingChanged = true;
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Large scale low-frequency noise");
        
        if (ImGui::SliderFloat("Frizz", &m_editParams.frizz, 0.0f, 1.0f)) stylingChanged = true;
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Small scale high-frequency jitter");
    }

    if (ImGui::CollapsingHeader("Force Field Physics")) {
        if (ImGui::SliderFloat("Force Influence", &m_editParams.forceInfluence, 0.0f, 2.0f, "%.2f")) stylingChanged = true;
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("How much external force fields (Wind, Turbulence, etc.) affect this groom.\n0 = Ignore forces, 1 = Normal, 2 = Double effect");
    }


    // Live update for styling/structural changes
    if ((stylingChanged || structuralChanged) && !m_selectedGroomName.empty()) {
        if (HairGroom* g = hairSystem.getGroom(m_selectedGroomName)) {
            g->params = m_editParams;
            g->material = m_currentMaterial; // Sync material changes too
            
            // NOTE: restyleGroom handles PointsPerStrand resize and ChildCount changes
            // For GuideCount changes, one still needs to click "Generate Full" 
            // but we at least mark it for rebuild if any change happens.
            hairSystem.restyleGroom(m_selectedGroomName);
            
            if (renderer) renderer->resetCPUAccumulation();
            m_needsRebuild = true;
        }
    }


    
    ImGui::Separator();
    ImGui::Text("Quality & Curves");
    
    ImGui::Checkbox("Use B-Spline Interpolation", &m_editParams.useBSpline);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Use cubic splines for much smoother strands (Requires more CPU/BVH time)");
    }
    
    if (m_editParams.useBSpline) {
        int subdiv = static_cast<int>(m_editParams.subdivisions);
        if (ImGui::SliderInt("Subdivisions (Steps)", &subdiv, 0, 4, "%d (Geometric detail)")) {
            m_editParams.subdivisions = static_cast<uint32_t>(subdiv);
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Tessellation level for cylindrical smoothing.\n0 = Low (Fastest)\n2 = Medium (Balanced)\n4 = High (Ultra smooth)");
        }
    }
    
    ImGui::Separator();
    
    // Generation Buttons
    ImGui::InputText("Groom Name", m_groomNameBuffer, sizeof(m_groomNameBuffer));
    
    bool canGenerate = triangles && !triangles->empty();
    if (!canGenerate) ImGui::BeginDisabled();
    
    ImGui::Columns(2, "GenButtons", false);
    
    if (ImGui::Button("Generate Full", ImVec2(-1, 40))) {
        if (onPreGenerate) onPreGenerate(); // [FIX] Sync CPU BVH before generating to avoid offset hair
        std::string groomNameStr = m_groomNameBuffer[0] == '\0' ? "hair_groom" : m_groomNameBuffer;
        hairSystem.generateOnMesh(*triangles, m_editParams, groomNameStr);
        if (HairGroom* g = hairSystem.getGroom(groomNameStr)) {
            g->material = m_currentMaterial;
        }
        hairSystem.buildBVH();
        m_selectedGroomName = groomNameStr;
        m_needsRebuild = true;
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Distribute guide strands across entire mesh based on count");

    ImGui::NextColumn();

    if (ImGui::Button("Create Empty", ImVec2(-1, 40))) {
        if (onPreGenerate) onPreGenerate(); // [FIX] Sync here too
        std::string groomNameStr = m_groomNameBuffer[0] == '\0' ? "hair_layer" : m_groomNameBuffer;
        
        // Auto-increment name if exists to prevent overwriting
        int counter = 1;
        std::string finalName = groomNameStr;
        while (hairSystem.exists(finalName)) {
            finalName = groomNameStr + "_" + std::to_string(counter++);
        }

        Hair::HairGenerationParams emptyParams = m_editParams;
        emptyParams.guideCount = 0; // Create empty
        hairSystem.generateOnMesh(*triangles, emptyParams, finalName);
        if (HairGroom* g = hairSystem.getGroom(finalName)) {
            g->material = m_currentMaterial;
        }
        m_selectedGroomName = finalName;
        strncpy(m_groomNameBuffer, finalName.c_str(), sizeof(m_groomNameBuffer));
        m_needsRebuild = true;
    }

    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Create an empty hair layer bound to mesh. Perfect for painting from scratch!");

    ImGui::Columns(1);
    
    if (!canGenerate) {
        ImGui::EndDisabled();
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "[!] Select a mesh to bind hair layers.");
    }

    // --- Groom Management Section ---
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("Groom Management");
    drawGroomList(hairSystem, renderer);
}

inline void HairUI::drawMaterialPanel(Renderer* renderer) {
    bool changed = false;
    
    // Color Mode
    const char* colorModes[] = { "Direct Color", "Melanin (Physical)", "Absorption", "Root UV Map" };
    int currentMode = static_cast<int>(m_currentMaterial.colorMode);
    if (ImGui::Combo("Color Mode", &currentMode, colorModes, 4)) {
        m_currentMaterial.colorMode = static_cast<HairMaterialParams::ColorMode>(currentMode);
        changed = true;
    }
    
    ImGui::Separator();
    
    switch (m_currentMaterial.colorMode) {
        case HairMaterialParams::ColorMode::ROOT_UV_MAP: {
             ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Mode: Inheritance");
             ImGui::TextWrapped("Hair color is sampled from the scalp mesh texture using UV coordinates.");
             break;
        }

        case HairMaterialParams::ColorMode::DIRECT_COLORING: {
            float color[3] = {m_currentMaterial.color.x, 
                             m_currentMaterial.color.y, 
                             m_currentMaterial.color.z};
            if (ImGui::ColorEdit3("Hair Color", color)) {
                m_currentMaterial.color = Vec3(color[0], color[1], color[2]);
                changed = true;
            }
            break;
        }
        
        case HairMaterialParams::ColorMode::MELANIN: {
            if (ImGui::SliderFloat("Melanin", &m_currentMaterial.melanin, 0.0f, 1.0f,
                             "%.2f (0=Blonde, 1=Black)")) changed = true;
            if (ImGui::SliderFloat("Redness", &m_currentMaterial.melaninRedness, 0.0f, 1.0f,
                             "%.2f (0=Brown, 1=Red)")) changed = true;
            
            // Preview color
            Vec3 sigma = HairBSDF::melaninToAbsorption(
                m_currentMaterial.melanin, 
                m_currentMaterial.melaninRedness
            );
            Vec3 previewColor(
                std::exp(-sigma.x * 0.5f),
                std::exp(-sigma.y * 0.5f),
                std::exp(-sigma.z * 0.5f)
            );
            ImGui::ColorButton("Preview", ImVec4(previewColor.x, previewColor.y, previewColor.z, 1.0f),
                             0, ImVec2(ImGui::GetContentRegionAvail().x, 30));
            break;
        }
        
        case HairMaterialParams::ColorMode::ABSORPTION: {
            float sigma[3] = {m_currentMaterial.absorptionCoefficient.x,
                             m_currentMaterial.absorptionCoefficient.y,
                             m_currentMaterial.absorptionCoefficient.z};
            if (ImGui::SliderFloat3("Absorption (σa)", sigma, 0.0f, 5.0f)) {
                m_currentMaterial.absorptionCoefficient = Vec3(sigma[0], sigma[1], sigma[2]);
                changed = true;
            }
            break;
        }
    }
    
    // Global Tint (Works on all modes)
    ImGui::Separator();
    ImGui::Text("Artistic Tint");
    if (ImGui::SliderFloat("Tint Strength", &m_currentMaterial.tint, 0.0f, 1.0f)) changed = true;
    float tintCol[3] = { m_currentMaterial.tintColor.x, m_currentMaterial.tintColor.y, m_currentMaterial.tintColor.z };
    if (ImGui::ColorEdit3("Tint Color", tintCol)) {
        m_currentMaterial.tintColor = Vec3(tintCol[0], tintCol[1], tintCol[2]);
        changed = true;
    }
    
    
    ImGui::Separator();
    ImGui::Text("Custom Textures (Independent)");
    ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Overrides Mesh Material when set.");

    // Helper to get filename
    auto GetFileName = [](const std::string& path) {
        size_t lastSlash = path.find_last_of("/\\");
        return (lastSlash != std::string::npos) ? path.substr(lastSlash + 1) : path;
    };

    // Albedo Map
    static char albedoBuf[256] = ""; 
    ImGui::Text("Albedo Map:");
    if (m_currentMaterial.customAlbedoTexture) {
         std::string shortName = GetFileName(m_currentMaterial.customAlbedoTexture->name);
         ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Active: %s", shortName.c_str());
         if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", m_currentMaterial.customAlbedoTexture->name.c_str());
         
         ImGui::SameLine();
         if (ImGui::Button("X##RemoveAlbedo")) { 
             m_currentMaterial.customAlbedoTexture = nullptr; 
             changed = true; 
         }
    } else {
         if (ImGui::Button("Browse...##Albedo")) {
             if (onOpenFileDialog) {
                 std::string path = onOpenFileDialog(L"Image Files\0*.png;*.jpg;*.jpeg;*.tga;*.bmp\0All Files\0*.*\0");
                 if (!path.empty()) {
                      m_currentMaterial.customAlbedoTexture = std::make_shared<Texture>(path, TextureType::Albedo);
                      changed = true;
                 }
             }
         }
    }

    // Roughness Map
    ImGui::Text("Roughness Map:");
    if (m_currentMaterial.customRoughnessTexture) {
         std::string shortName = GetFileName(m_currentMaterial.customRoughnessTexture->name);
         ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Active: %s", shortName.c_str());
         if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", m_currentMaterial.customRoughnessTexture->name.c_str());

         ImGui::SameLine();
         if (ImGui::Button("X##RemoveRough")) { 
             m_currentMaterial.customRoughnessTexture = nullptr; 
             changed = true; 
         }
    } else {
         if (ImGui::Button("Browse...##Rough")) {
             if (onOpenFileDialog) {
                 std::string path = onOpenFileDialog(L"Image Files\0*.png;*.jpg;*.jpeg;*.tga;*.bmp\0All Files\0*.*\0");
                 if (!path.empty()) {
                      m_currentMaterial.customRoughnessTexture = std::make_shared<Texture>(path, TextureType::Roughness);
                      changed = true;
                 }
             }
         }
    }

    ImGui::Separator();
    ImGui::Text("Surface Properties");
    
    if (ImGui::SliderFloat("Roughness", &m_currentMaterial.roughness, 0.01f, 1.0f)) changed = true;
    if (ImGui::SliderFloat("Radial Roughness", &m_currentMaterial.radialRoughness, 0.01f, 1.0f)) changed = true;
    if (ImGui::SliderFloat("Cuticle Angle (°)", &m_currentMaterial.cuticleAngle, 0.0f, 10.0f)) changed = true;
    if (ImGui::SliderFloat("IOR", &m_currentMaterial.ior, 1.0f, 2.5f)) changed = true;
    
    ImGui::Separator();
    ImGui::Text("Coat (Fur)");
    
    if (ImGui::SliderFloat("Coat Strength", &m_currentMaterial.coat, 0.0f, 1.0f)) changed = true;
    if (m_currentMaterial.coat > 0.0f) {
        float coatTint[3] = {m_currentMaterial.coatTint.x,
                            m_currentMaterial.coatTint.y,
                            m_currentMaterial.coatTint.z};
        if (ImGui::ColorEdit3("Coat Tint", coatTint)) {
            m_currentMaterial.coatTint = Vec3(coatTint[0], coatTint[1], coatTint[2]);
            changed = true;
        }
    }
    
    // Random variation
    ImGui::Separator();
    ImGui::Text("Variation");
    if (ImGui::SliderFloat("Random Hue", &m_currentMaterial.randomHue, 0.0f, 0.2f)) changed = true;
    if (ImGui::SliderFloat("Random Brightness", &m_currentMaterial.randomValue, 0.0f, 0.3f)) changed = true;

    // Sync with renderer if something changed
    if (changed) {
        if (!m_selectedGroomName.empty()) {
            // Need a way to get the hair system here, but drawMaterialPanel doesn't take it.
            // However, Renderer has it!
            if (renderer) {
                if (HairGroom* g = renderer->getHairSystem().getGroom(m_selectedGroomName)) {
                    g->material = m_currentMaterial;
                }
                renderer->setHairMaterial(m_currentMaterial);
                // Ensure instant visual feedback by clearing old samples
                renderer->resetCPUAccumulation();
            }
        } else if (renderer) {
            renderer->setHairMaterial(m_currentMaterial);
            renderer->resetCPUAccumulation();
        }
    }
}


inline void HairUI::drawGroomList(HairSystem& hairSystem, Renderer* renderer) {
    ImGui::Text("Active Grooms: %zu", hairSystem.getGroomCount());
    ImGui::Separator();
    
    std::string groomToDelete;
    std::vector<std::string> groomNames = hairSystem.getGroomNames();

    if (ImGui::BeginChild("GroomScroll", ImVec2(0, 250), true)) {
        for (const auto& name : groomNames) {
            HairGroom* g = hairSystem.getGroom(name);
            if (!g) continue;

            bool isSelected = (name == m_selectedGroomName);
            ImGui::PushID(name.c_str());
            
            // Format: [MeshName] GroomName (Count)
            char label[128];
            snprintf(label, sizeof(label), "[%s] %s (%zu)", 
                     g->boundMeshName.c_str(), name.c_str(), g->guides.size());

            if (ImGui::Selectable(label, isSelected)) {
                m_selectedGroomName = name;
                m_editParams = g->params;
                m_currentMaterial = g->material;
                if (renderer) renderer->setHairMaterial(m_currentMaterial);
                strncpy(m_groomNameBuffer, name.c_str(), sizeof(m_groomNameBuffer));
            }



            
            if (ImGui::BeginPopupContextItem()) {
                if (ImGui::MenuItem("Delete Groom")) {
                    groomToDelete = name;
                }
                ImGui::EndPopup();
            }
            ImGui::PopID();
        }
    }
    ImGui::EndChild();

    if (!groomToDelete.empty()) {
        hairSystem.removeGroom(groomToDelete);
        if (m_selectedGroomName == groomToDelete) {
            m_selectedGroomName = "";
            m_groomNameBuffer[0] = '\0'; // Clear buffer so new name can be typed
        }
        m_needsRebuild = true;
    }
    
    ImGui::Separator();
    if (ImGui::Button("Clear All Hair", ImVec2(-1, 30))) {
        hairSystem.clearAll();
        m_selectedGroomName = "";
        m_needsRebuild = true;
    }
}



inline void HairUI::drawPresets() {
    ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Style Library");
    ImGui::TextDisabled("Select to apply all settings instantly.");
    ImGui::Separator();
    
    // Grouping presets by index (matches initializePresets order)
    auto drawPresetItem = [&](size_t i) {
        bool selected = false; // Could track if current matches this
        if (ImGui::Selectable(m_presets[i].name.c_str(), selected)) {
            applyPreset(m_presets[i]);
            m_needsRebuild = true;
        }
    };

    if (ImGui::BeginChild("PresetsList")) {
        // 1. Human (Indices 0-3)
        if (ImGui::CollapsingHeader("Human Hair", ImGuiTreeNodeFlags_DefaultOpen)) {
            for (size_t i = 0; i < 4 && i < m_presets.size(); ++i) drawPresetItem(i);
        }

        // 2. ArchViz / Props (Indices 4-6)
        if (ImGui::CollapsingHeader("Interior & Nature", ImGuiTreeNodeFlags_DefaultOpen)) {
            for (size_t i = 4; i < 7 && i < m_presets.size(); ++i) drawPresetItem(i);
        }

        // 3. Animals (Indices 7+)
        if (ImGui::CollapsingHeader("Fur & Creatures", ImGuiTreeNodeFlags_DefaultOpen)) {
            for (size_t i = 7; i < m_presets.size(); ++i) drawPresetItem(i);
        }
        
        ImGui::EndChild();
    }
}

inline void HairUI::drawPaintPanel(HairSystem& hairSystem) {
    ImGui::Text("Interactive Hair Painting");
    ImGui::TextWrapped("Select a brush mode and paint directly on the mesh in the viewport.");
    


    ImGui::Separator();
    
    // Brush Mode Selection
    ImGui::Text("Brush Mode:");
    
    // Mode buttons with icons - STACK SAFE IMPLEMENTATION
    float btnW = (ImGui::GetContentRegionAvail().x - 3 * ImGui::GetStyle().ItemSpacing.x) / 4.0f;

    auto DrawSafeBtn = [&](HairPaintMode mode, const char* label, const char* tooltip, int idx) {
        if (idx % 4 != 0) ImGui::SameLine();
        bool isCurrent = (m_paintMode == mode);
        if (isCurrent) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.2f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.7f, 0.3f, 1.0f));
        }
        if (ImGui::Button(label, ImVec2(btnW, 35))) {
            m_paintMode = isCurrent ? HairPaintMode::NONE : mode;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tooltip);
        if (isCurrent) ImGui::PopStyleColor(2);
    };

    DrawSafeBtn(HairPaintMode::ADD,     "+ Add",     "Add new hair strands", 0);
    DrawSafeBtn(HairPaintMode::REMOVE,  "- Rem",     "Remove strands", 1);
    DrawSafeBtn(HairPaintMode::CUT,     "/ Cut",     "Cut strands shorter", 2);
    DrawSafeBtn(HairPaintMode::COMB,    "~ Comb",    "Comb/style hair direction", 3);
    DrawSafeBtn(HairPaintMode::LENGTH,  "| Len",     "Adjust strand length", 4);
    DrawSafeBtn(HairPaintMode::DENSITY, "# Den",     "Adjust local density", 5);
    DrawSafeBtn(HairPaintMode::CLUMP,   "* Clp",     "Increase clumpiness", 6);
    DrawSafeBtn(HairPaintMode::PUFF,    "o Puf",     "Add volume/puff", 7);
    
    DrawSafeBtn(HairPaintMode::WAVE,    "S Wav",     "Add procedural waves", 8);
    DrawSafeBtn(HairPaintMode::FRIZZ,   "~ Frz",     "Add random detail", 9);
    DrawSafeBtn(HairPaintMode::SMOOTH,  "_ Smth",    "Relax/Straighten hair", 10);
    DrawSafeBtn(HairPaintMode::PINCH,   "> Pnc",     "Bunch points together", 11);
    DrawSafeBtn(HairPaintMode::SPREAD,  "< Spr",     "Spread points apart", 12);
    
    // Exit button
    if (m_paintMode != HairPaintMode::NONE) {
        ImGui::Separator();
        
        if (ImGui::Checkbox("Hide Children During Paint", &m_hideChildrenDuringPaint)) {
            m_needsRebuild = true;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Improves performance by only showing guide strands while grooming.");

        if (ImGui::Button("Exit Paint Mode (ESC)", ImVec2(-1, 30))) {
            m_paintMode = HairPaintMode::NONE;
            m_needsRebuild = true;
        }
    }
    
    if (m_paintMode != m_lastPaintMode) {
        if (m_hideChildrenDuringPaint) {
            m_needsRebuild = true; 
        }
        m_lastPaintMode = m_paintMode;
    }
    
    ImGui::Separator();
    ImGui::Text("Brush Settings:");
    
    // Brush radius
    float radiusCm = m_brushSettings.radius * 100.0f;
    if (ImGui::SliderFloat("Radius (cm)", &radiusCm, 0.5f, 50.0f)) {
        m_brushSettings.radius = radiusCm / 100.0f;
    }
    
    // Brush strength
    ImGui::SliderFloat("Strength", &m_brushSettings.strength, 0.0f, 1.0f);
    
    // Falloff
    ImGui::SliderFloat("Falloff", &m_brushSettings.falloff, 0.0f, 1.0f, "%.2f (0=Hard, 1=Soft)");
    
    // Mode-specific options
    ImGui::Separator();
    
    switch (m_paintMode) {
        case HairPaintMode::CUT:
            ImGui::Checkbox("Trim at Hit Point (Scissors)", &m_brushSettings.cutAtHitPoint);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("If enabled, hair is cut exactly at the brush intersection point.");
            if (!m_brushSettings.cutAtHitPoint) {
                ImGui::SliderFloat("Cut Rate", &m_brushSettings.cutLength, 0.1f, 0.9f, 
                                  "%.2f (Gentle shortening)");
            }
            break;
            
        case HairPaintMode::COMB:
            ImGui::Checkbox("Use View Direction", &m_brushSettings.useViewDirection);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Comb in the hit direction. Affects points closest to brush first.");
            if (!m_brushSettings.useViewDirection) {
                float dir[3] = { m_brushSettings.combDirection.x, 
                                m_brushSettings.combDirection.y, 
                                m_brushSettings.combDirection.z };
                if (ImGui::SliderFloat3("Comb Direction", dir, -1.0f, 1.0f)) {
                    m_brushSettings.combDirection = Vec3(dir[0], dir[1], dir[2]).normalize();
                }
            }
            break;
            
        case HairPaintMode::WAVE:
        case HairPaintMode::FRIZZ:
            ImGui::SliderFloat("Frequency", &m_brushSettings.frequency, 0.1f, 50.0f);
            ImGui::SliderFloat("Amplitude", &m_brushSettings.amplitude, 0.0f, 1.0f);
            break;
            
        case HairPaintMode::ADD:
            ImGui::Checkbox("Affect Only Guides", &m_brushSettings.affectGuides);
            break;
            
        default:
            break;
    }
    
    ImGui::Separator();
    ImGui::Text("Mirror Brush:");
    ImGui::Checkbox("Mirror X", &m_brushSettings.mirrorX); ImGui::SameLine();
    ImGui::Checkbox("Mirror Y", &m_brushSettings.mirrorY); ImGui::SameLine();
    ImGui::Checkbox("Mirror Z", &m_brushSettings.mirrorZ);

    ImGui::Separator();
    ImGui::Text("Target:");
    ImGui::Checkbox("Affect Guide Strands", &m_brushSettings.affectGuides);
    ImGui::Checkbox("Affect Children (slower)", &m_brushSettings.affectChildren);

    
    // Keyboard shortcut hint
    ImGui::Separator();
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Shortcuts:");
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "  [ / ] - Decrease/Increase Radius");
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "  Shift+[ / ] - Decrease/Increase Strength");
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "  ESC - Exit Paint Mode");
}

inline void HairUI::applyBrush(HairSystem& hairSystem, const Vec3& worldPos, const Vec3& normal, float deltaTime, const Vec3& customDir) {
    if (m_paintMode == HairPaintMode::NONE) {
        return;
    }
    if (m_selectedGroomName.empty()) return;
    
    HairGroom* groom = hairSystem.getGroom(m_selectedGroomName);
    if (!groom) return;


    // Apply primary brush
    applyBrushInternal(hairSystem, worldPos, normal, m_brushSettings.strength, m_brushSettings.radius, m_brushSettings.combDirection, deltaTime);
    // Apply mirrored brushes relative to Object Pivot
    if (m_brushSettings.mirrorX || m_brushSettings.mirrorY || m_brushSettings.mirrorZ) {
        Matrix4x4 localToWorld = groom->transform;
        Matrix4x4 worldToLocal = localToWorld.inverse();
        
        Vec3 localPos = worldToLocal.transform_point(worldPos);
        Vec3 localNormal = worldToLocal.transform_vector(normal).normalize();

        for (int i = 1; i < 8; ++i) {
            bool mx = (i & 1) && m_brushSettings.mirrorX;
            bool my = (i & 2) && m_brushSettings.mirrorY;
            bool mz = (i & 4) && m_brushSettings.mirrorZ;
            
            if ((i & 1) && !m_brushSettings.mirrorX) continue;
            if ((i & 2) && !m_brushSettings.mirrorY) continue;
            if ((i & 4) && !m_brushSettings.mirrorZ) continue;
            
            Vec3 mirroredLocalPos = localPos;
            Vec3 mirroredLocalNormal = localNormal;
            Vec3 mirroredCombDir = m_brushSettings.combDirection;
            
            if (mx) { mirroredLocalPos.x *= -1.0f; mirroredLocalNormal.x *= -1.0f; mirroredCombDir.x *= -1.0f; }
            if (my) { mirroredLocalPos.y *= -1.0f; mirroredLocalNormal.y *= -1.0f; mirroredCombDir.y *= -1.0f; }
            if (mz) { mirroredLocalPos.z *= -1.0f; mirroredLocalNormal.z *= -1.0f; mirroredCombDir.z *= -1.0f; }
            
            Vec3 mirroredWorldPos = localToWorld.transform_point(mirroredLocalPos);
            Vec3 mirroredWorldNormal = localToWorld.transform_vector(mirroredLocalNormal).normalize();
            
            if (m_projector) {
                if (!m_projector(mirroredWorldPos, mirroredWorldNormal)) continue;
            }
            
            if (std::isfinite(mirroredWorldPos.x) && std::isfinite(mirroredWorldPos.y) && std::isfinite(mirroredWorldPos.z)) {
                applyBrushInternal(hairSystem, mirroredWorldPos, mirroredWorldNormal, m_brushSettings.strength, m_brushSettings.radius, mirroredCombDir, deltaTime);
            }
        }
    }
}


inline void HairUI::applyBrushInternal(HairSystem& hairSystem, const Vec3& worldPos, const Vec3& normal, 
                                      float effectStrength, float brushRadius, const Vec3& combDirection, float deltaTime) {
    if (deltaTime <= 1e-6f || effectStrength <= 0.0f) return;

    // Avoid accumulation spikes (e.g. first frame or lag)
    float effectiveDelta = std::min(deltaTime, 0.1f);



    switch (m_paintMode) {
        case HairPaintMode::ADD: {
            // Scale by time for smooth continuous addition
            // Rate: hairs per second
            float rate = effectStrength * 100.0f; 
            float countAccum = rate * effectiveDelta;
            
            // Probabilistic addition to handle fractional counts and multiple spikes
            int strandCount = static_cast<int>(countAccum);
            float remainder = countAccum - (float)strandCount;
            if (((float)rand() / (float)RAND_MAX) < remainder) strandCount++;
            
            if (strandCount > 0) {
                hairSystem.addStrandsAtPosition(m_selectedGroomName, worldPos, normal, brushRadius, strandCount);
                m_needsRebuild = true;
            }
            break;
        }

        case HairPaintMode::DENSITY: {
            float rate = effectStrength * 150.0f;
            float countAccum = rate * effectiveDelta; 
            
            int strandCount = static_cast<int>(countAccum);
            float remainder = countAccum - (float)strandCount;
            if (((float)rand() / (float)RAND_MAX) < remainder) strandCount++;

            if (strandCount > 0) {
                hairSystem.addStrandsAtPosition(m_selectedGroomName, worldPos, normal, brushRadius, strandCount);
                m_needsRebuild = true;
            }
            break;
        }
        
        case HairPaintMode::REMOVE: {
            hairSystem.removeStrandsAtPosition(m_selectedGroomName, worldPos, brushRadius);
            m_needsRebuild = true;
            break;
        }
        
        case HairPaintMode::CUT: {
            if (HairGroom* groom = hairSystem.getGroom(m_selectedGroomName)) {
                float brushRadiusSq = brushRadius * brushRadius;
                Matrix4x4 localToWorld = groom->transform;

                for (auto& strand : groom->guides) {
                    // OPTIMIZATION: Quick root-distance early exit
                    Vec3 strandRootWorld = localToWorld.transform_point(strand.baseRootPos);
                    float distToRoot = (strandRootWorld - worldPos).length();
                    if (distToRoot > (brushRadius + strand.baseLength * 1.2f)) continue;

                    // Modern Volume Hit: Check distance to any point on the strand
                    float minDistSq = 1e30f;
                    for (const auto& p : strand.groomedPositions) {
                        Vec3 worldP = localToWorld.transform_point(p);
                        float d2 = (worldP - worldPos).length_squared();
                        if (d2 < minDistSq) minDistSq = d2;
                    }

                    if (minDistSq < brushRadiusSq) {
                        float dist = std::sqrt(minDistSq);
                        float falloff = 1.0f - dist / brushRadius;
                        
                        if (m_brushSettings.cutAtHitPoint) {
                            // OPTIMIZED Scissors Mode: Use v-parameter from closest point
                            // Since we already found minDistSq, we can just track the index
                            float closestV = 0.0f;
                            float dMin = 1e30f;
                            for (size_t i = 0; i < strand.groomedPositions.size(); ++i) {
                                float d = (localToWorld.transform_point(strand.groomedPositions[i]) - worldPos).length();
                                if (d < dMin) { dMin = d; closestV = (float)i / (strand.groomedPositions.size() - 1); }
                            }
                            strand.baseLength = std::min(strand.baseLength, strand.baseLength * closestV);
                        } else {
                            float cutRate = effectStrength * falloff * 2.5f; 
                            strand.baseLength = std::max(0.001f, strand.baseLength * (1.0f - cutRate * effectiveDelta));
                        }
                        
                        // Sync groomed positions to new length
                        Vec3 root = strand.groomedPositions[0];
                        float segmentLen = (strand.groomedPositions.size() > 1) ? (strand.baseLength / (strand.groomedPositions.size() - 1)) : 0.0f;
                        for (size_t i = 1; i < strand.groomedPositions.size(); ++i) {
                            Vec3 dir = (strand.groomedPositions[i] - strand.groomedPositions[i-1]);
                            float l = dir.length();
                            if (l > 1e-5f) strand.groomedPositions[i] = strand.groomedPositions[i-1] + (dir / l) * segmentLen;
                            else strand.groomedPositions[i] = strand.groomedPositions[i-1] + strand.rootNormal * segmentLen;
                        }
                    }
                }
                groom->isDirty = true;
                hairSystem.restyleGroom(m_selectedGroomName);
            }
            m_needsRebuild = true;
            break;
        }

        
        case HairPaintMode::COMB: {
            if (HairGroom* groom = hairSystem.getGroom(m_selectedGroomName)) {
                Vec3 combDir = combDirection;
                
                // [FIX] Space Correction: World -> Local
                Matrix4x4 localToWorld = groom->transform;
                Matrix4x4 worldToLocal = localToWorld.inverse();
                
                // Project world comb dir to surface tangent, then move to local
                float dotN = combDir.dot(normal);
                combDir = (combDir - normal * dotN).normalize();
                Vec3 localCombDir = worldToLocal.transform_vector(combDir).normalize();

                float brushRadiusSq = brushRadius * brushRadius;
                for (auto& strand : groom->guides) {
                    // OPTIMIZATION: Quick box/root check
                    Vec3 strandRootWorld = localToWorld.transform_point(strand.baseRootPos);
                    if ((strandRootWorld - worldPos).length() > (brushRadius + strand.baseLength * 1.5f)) continue;

                    // Modern Volume Hit Check
                    float minDistSq = 1e30f;
                    for (const auto& p : strand.groomedPositions) {
                        Vec3 worldP = localToWorld.transform_point(p);
                        float d2 = (worldP - worldPos).length_squared();
                        if (d2 < minDistSq) minDistSq = d2;
                    }
                    
                    if (minDistSq < brushRadiusSq && strand.groomedPositions.size() > 1) {
                        float segmentTargetLen = strand.baseLength / (strand.groomedPositions.size() - 1);

                        for (size_t i = 1; i < strand.groomedPositions.size(); ++i) {
                            Vec3 worldSegP = localToWorld.transform_point(strand.groomedPositions[i]);
                            float segDist = (worldSegP - worldPos).length();
                            
                            // Quadratic falloff for smoother feel
                            float normDist = std::clamp(segDist / brushRadius, 0.0f, 1.0f);
                            float segFalloff = (1.0f - normDist) * (1.0f - normDist);
                            
                            // Magnetic bending: Apply falloff per control point based on hit proximity
                            float segmentStrength = effectStrength * segFalloff * effectiveDelta * 10.0f; 
                            
                            // 1. Tangential displacement (LOCAL SPACE)
                            strand.groomedPositions[i] = strand.groomedPositions[i] + localCombDir * segmentStrength;
                            
                            // 2. [COLLISION] Keep hair ABOVE the emitter surface
                            Vec3 rel = strand.groomedPositions[i] - strand.groomedPositions[0];
                            float distFromRootPlane = rel.dot(strand.rootNormal);
                            if (distFromRootPlane < 0.002f) {
                                strand.groomedPositions[i] = strand.groomedPositions[i] + strand.rootNormal * (0.0025f - distFromRootPlane);
                            }

                            // 3. Length Constraint (Per segment)
                            Vec3 prev = strand.groomedPositions[i-1];
                            Vec3 segDir = strand.groomedPositions[i] - prev;
                            float currentLen = segDir.length();
                            if (currentLen > 1e-6f) {
                                strand.groomedPositions[i] = prev + (segDir / currentLen) * segmentTargetLen;
                            }
                        }

                        // [NEW] Relaxation pass to prevent jagged strands
                        for (int iter = 0; iter < 2; ++iter) {
                            for (size_t i = 1; i < strand.groomedPositions.size() - 1; ++i) {
                                Vec3 avg = (strand.groomedPositions[i-1] + strand.groomedPositions[i+1]) * 0.5f;
                                strand.groomedPositions[i] = strand.groomedPositions[i] * 0.9f + avg * 0.1f;
                                
                                // Re-apply length constraint after smoothing
                                Vec3 prev = strand.groomedPositions[i-1];
                                Vec3 segDir = (strand.groomedPositions[i] - prev);
                                float l = segDir.length();
                                if (l > 1e-6f) strand.groomedPositions[i] = prev + (segDir / l) * segmentTargetLen;
                            }
                        }
                    }
                }
                groom->isDirty = true;
                // [FIX] Bake manual edits to rest pose immediately to persist against skinning updates
                hairSystem.bakeGroomToRest(m_selectedGroomName);
                hairSystem.restyleGroom(m_selectedGroomName);
            }
            m_needsRebuild = true;
            break;
        }
        
        case HairPaintMode::LENGTH: {
            if (HairGroom* groom = hairSystem.getGroom(m_selectedGroomName)) {
                float brushRadiusSq = brushRadius * brushRadius;
                Matrix4x4 localToWorld = groom->transform;
                for (auto& strand : groom->guides) {
                    float minDistSq = 1e30f;
                    for (const auto& p : strand.groomedPositions) {
                        Vec3 worldP = localToWorld.transform_point(p);
                        float d2 = (worldP - worldPos).length_squared();
                        if (d2 < minDistSq) minDistSq = d2;
                    }

                    if (minDistSq < brushRadiusSq && strand.groomedPositions.size() > 1) {
                        float dist = std::sqrt(minDistSq);
                        float falloff = 1.0f - dist / brushRadius;
                        
                        // Scale length change by time for smooth growth
                        float growRate = effectStrength * falloff * 2.0f; // 200% length per second max
                        float amount = growRate * effectiveDelta;
                        
                        float scaleFactor = 1.0f + amount;
                        strand.baseLength *= scaleFactor;
                        Vec3 root = strand.groomedPositions[0];
                        for (size_t i = 1; i < strand.groomedPositions.size(); ++i) {
                            strand.groomedPositions[i] = root + (strand.groomedPositions[i] - root) * scaleFactor;
                        }
                    }
                }
                groom->isDirty = true;
                hairSystem.restyleGroom(m_selectedGroomName);
            }
            m_needsRebuild = true;
            break;
        }
        
        case HairPaintMode::PUFF: {
            if (HairGroom* groom = hairSystem.getGroom(m_selectedGroomName)) {
                float brushRadiusSq = brushRadius * brushRadius;
                Matrix4x4 localToWorld = groom->transform;
                for (auto& strand : groom->guides) {
                    float minDistSq = 1e30f;
                    for (const auto& p : strand.groomedPositions) {
                        Vec3 worldP = localToWorld.transform_point(p);
                        float d2 = (worldP - worldPos).length_squared();
                        if (d2 < minDistSq) minDistSq = d2;
                    }

                    if (minDistSq < brushRadiusSq && strand.groomedPositions.size() > 1) {
                        float dist = std::sqrt(minDistSq);
                        float falloff = 1.0f - dist / brushRadius;
                        Vec3 rootNormal = strand.rootNormal;

                        for (size_t i = 1; i < strand.groomedPositions.size(); ++i) {
                            Vec3 worldPoint = localToWorld.transform_point(strand.groomedPositions[i]);
                            float pDist = (worldPoint - worldPos).length();
                            float pFalloff = 1.0f - std::clamp(pDist / brushRadius, 0.0f, 1.0f);
                            
                            // Refined Puff: Move points along normal but with per-point falloff 
                            // and a much smaller multiplier to avoid "explosive" growth.
                            float displacement = effectStrength * pFalloff * effectiveDelta * 2.0f;
                            strand.groomedPositions[i] = strand.groomedPositions[i] + rootNormal * displacement;
                        }
                    }
                }
                groom->isDirty = true;
                hairSystem.restyleGroom(m_selectedGroomName);
            }
            m_needsRebuild = true;
            break;
        }

        case HairPaintMode::CLUMP: {
            if (HairGroom* groom = hairSystem.getGroom(m_selectedGroomName)) {
                float brushRadiusSq = brushRadius * brushRadius;
                Matrix4x4 localToWorld = groom->transform;
                for (auto& strand : groom->guides) {
                    float minDistSq = 1e30f;
                    for (const auto& p : strand.groomedPositions) {
                        Vec3 worldP = localToWorld.transform_point(p);
                        float d2 = (worldP - worldPos).length_squared();
                        if (d2 < minDistSq) minDistSq = d2;
                    }

                    if (minDistSq < brushRadiusSq) {
                        float dist = std::sqrt(minDistSq);
                        float falloff = 1.0f - dist / brushRadius;
                        // Scale clump change by time
                        strand.clumpScale = std::max(0.0f, strand.clumpScale + effectStrength * falloff * effectiveDelta * 100.0f);
                    }
                }
                groom->isDirty = true;
                hairSystem.restyleGroom(m_selectedGroomName);
            }
            m_needsRebuild = true;
            break;
        }

        case HairPaintMode::WAVE: {
            if (HairGroom* groom = hairSystem.getGroom(m_selectedGroomName)) {
                float brushRadiusSq = brushRadius * brushRadius;
                Matrix4x4 localToWorld = groom->transform;
                Matrix4x4 worldToLocal = localToWorld.inverse();
                Vec3 localPos = worldToLocal.transform_point(worldPos);

                for (auto& strand : groom->guides) {
                    float minDistSq = 1e30f;
                    for (const auto& p : strand.groomedPositions) {
                        Vec3 worldP = localToWorld.transform_point(p);
                        float d2 = (worldP - worldPos).length_squared();
                        if (d2 < minDistSq) minDistSq = d2;
                    }

                    if (minDistSq < brushRadiusSq && strand.groomedPositions.size() > 1) {
                        float dist = std::sqrt(minDistSq);
                        float falloff = 1.0f - dist / brushRadius;
                        Vec3 normal = strand.rootNormal;
                        Vec3 bitangent = (std::abs(normal.y) < 0.9f) ? Vec3::cross(normal, Vec3(0,1,0)) : Vec3::cross(normal, Vec3(1,0,0));
                        bitangent.normalize();

                        for (size_t i = 1; i < strand.groomedPositions.size(); ++i) {
                            float t = (float)i / (strand.groomedPositions.size() - 1);
                            float wave = std::sin(t * m_brushSettings.frequency * 6.283f + strand.randomSeed * 6.283f);
                            float displacement = m_brushSettings.amplitude * wave * falloff * effectStrength * effectiveDelta * 5.0f * t;
                            strand.groomedPositions[i] = strand.groomedPositions[i] + bitangent * displacement;
                        }
                    }
                }
                groom->isDirty = true;
                hairSystem.restyleGroom(m_selectedGroomName);
            }
            m_needsRebuild = true;
            break;
        }

        case HairPaintMode::FRIZZ: {
            if (HairGroom* groom = hairSystem.getGroom(m_selectedGroomName)) {
                float brushRadiusSq = brushRadius * brushRadius;
                Matrix4x4 localToWorld = groom->transform;
                for (auto& strand : groom->guides) {
                    float minDistSq = 1e30f;
                    for (const auto& p : strand.groomedPositions) {
                        Vec3 worldP = localToWorld.transform_point(p);
                        float d2 = (worldP - worldPos).length_squared();
                        if (d2 < minDistSq) minDistSq = d2;
                    }

                    if (minDistSq < brushRadiusSq && strand.groomedPositions.size() > 1) {
                        float dist = std::sqrt(minDistSq);
                        float falloff = 1.0f - dist / brushRadius;
                        
                        for (size_t i = 1; i < strand.groomedPositions.size(); ++i) {
                            float t = (float)i / (strand.groomedPositions.size() - 1);
                            // Simple stable pseudo-random jitter
                            uint32_t seed = strand.strandID ^ (i * 12345);
                            float rx = (float)((seed * 1103515245 + 12345) & 0x7FFFFFFF) / 2147483647.0f - 0.5f;
                            float ry = (float)((seed * 22695477 + 1) & 0x7FFFFFFF) / 2147483647.0f - 0.5f;
                            float rz = (float)((seed * 65539) & 0x7FFFFFFF) / 2147483647.0f - 0.5f;
                            
                            Vec3 jitter(rx, ry, rz);
                            float displacement = m_brushSettings.amplitude * falloff * effectStrength * effectiveDelta * 2.0f * t;
                            strand.groomedPositions[i] = strand.groomedPositions[i] + jitter * displacement;
                        }
                    }
                }
                groom->isDirty = true;
                hairSystem.restyleGroom(m_selectedGroomName);
            }
            m_needsRebuild = true;
            break;
        }

        case HairPaintMode::SMOOTH: {
            if (HairGroom* groom = hairSystem.getGroom(m_selectedGroomName)) {
                float brushRadiusSq = brushRadius * brushRadius;
                Matrix4x4 localToWorld = groom->transform;
                for (auto& strand : groom->guides) {
                    float minDistSq = 1e30f;
                    for (const auto& p : strand.groomedPositions) {
                        Vec3 worldP = localToWorld.transform_point(p);
                        float d2 = (worldP - worldPos).length_squared();
                        if (d2 < minDistSq) minDistSq = d2;
                    }

                    if (minDistSq < brushRadiusSq && strand.groomedPositions.size() > 2) {
                        float dist = std::sqrt(minDistSq);
                        float falloff = 1.0f - dist / brushRadius;
                        float smoothAmount = effectStrength * falloff * effectiveDelta * 5.0f;

                        // Laplace smoothing on conditioned positions
                        std::vector<Vec3> nextPos = strand.groomedPositions;
                        for (size_t i = 1; i < strand.groomedPositions.size() - 1; ++i) {
                            Vec3 avg = (strand.groomedPositions[i-1] + strand.groomedPositions[i+1]) * 0.5f;
                            nextPos[i] = Vec3::mix(strand.groomedPositions[i], avg, std::min(1.0f, smoothAmount));
                        }
                        strand.groomedPositions = nextPos;
                    }
                }
                groom->isDirty = true;
                hairSystem.restyleGroom(m_selectedGroomName);
            }
            m_needsRebuild = true;
            break;
        }

        case HairPaintMode::PINCH: {
            if (HairGroom* groom = hairSystem.getGroom(m_selectedGroomName)) {
                float brushRadiusSq = brushRadius * brushRadius;
                Matrix4x4 localToWorld = groom->transform;
                Matrix4x4 worldToLocal = localToWorld.inverse();
                Vec3 localBrushPos = worldToLocal.transform_point(worldPos);

                for (auto& strand : groom->guides) {
                    float minDistSq = 1e30f;
                    for (const auto& p : strand.groomedPositions) {
                        Vec3 worldP = localToWorld.transform_point(p);
                        float d2 = (worldP - worldPos).length_squared();
                        if (d2 < minDistSq) minDistSq = d2;
                    }

                    if (minDistSq < brushRadiusSq) {
                        float dist = std::sqrt(minDistSq);
                        float falloff = 1.0f - dist / brushRadius;
                        float pinchAmt = effectStrength * falloff * effectiveDelta * 5.0f;

                        for (size_t i = 1; i < strand.groomedPositions.size(); ++i) {
                            Vec3 toBrush = localBrushPos - strand.groomedPositions[i];
                            strand.groomedPositions[i] = strand.groomedPositions[i] + toBrush * std::min(1.0f, pinchAmt);
                        }
                    }
                }
                groom->isDirty = true;
                hairSystem.restyleGroom(m_selectedGroomName);
            }
            m_needsRebuild = true;
            break;
        }

        case HairPaintMode::SPREAD: {
            if (HairGroom* groom = hairSystem.getGroom(m_selectedGroomName)) {
                float brushRadiusSq = brushRadius * brushRadius;
                Matrix4x4 localToWorld = groom->transform;
                Matrix4x4 worldToLocal = localToWorld.inverse();
                Vec3 localBrushPos = worldToLocal.transform_point(worldPos);

                for (auto& strand : groom->guides) {
                    float minDistSq = 1e30f;
                    for (const auto& p : strand.groomedPositions) {
                        Vec3 worldP = localToWorld.transform_point(p);
                        float d2 = (worldP - worldPos).length_squared();
                        if (d2 < minDistSq) minDistSq = d2;
                    }

                    if (minDistSq < brushRadiusSq) {
                        float dist = std::sqrt(minDistSq);
                        float falloff = 1.0f - dist / brushRadius;
                        float spreadAmt = effectStrength * falloff * effectiveDelta * 5.0f;

                        for (size_t i = 1; i < strand.groomedPositions.size(); ++i) {
                            Vec3 awayFromBrush = strand.groomedPositions[i] - localBrushPos;
                            float l = awayFromBrush.length();
                            if (l > 0.001f) {
                                strand.groomedPositions[i] = strand.groomedPositions[i] + (awayFromBrush / l) * spreadAmt * 0.01f;
                            }
                        }
                    }
                }
                groom->isDirty = true;
                hairSystem.restyleGroom(m_selectedGroomName);
            }
            m_needsRebuild = true;
            break;
        }
        default:
            break;
    }
}



inline void HairUI::drawStats(const HairSystem& hairSystem) {
    size_t strandCount = hairSystem.getTotalStrandCount();
    size_t pointCount = hairSystem.getTotalPointCount();
    
    ImGui::Text("Stats: %zu strands, %zu points", strandCount, pointCount);
    
    // Memory estimate
    size_t memoryBytes = pointCount * sizeof(HairPoint) + strandCount * sizeof(HairStrand);
    float memoryMB = memoryBytes / (1024.0f * 1024.0f);
    ImGui::Text("Estimated Memory: %.2f MB", memoryMB);
}

} // namespace Hair

#endif // HAIR_UI_H
