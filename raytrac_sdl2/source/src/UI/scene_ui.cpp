// ═══════════════════════════════════════════════════════════════════════════════
// SCENE UI - MAIN ENTRY POINT
// ═══════════════════════════════════════════════════════════════════════════════
// NOTE: This file has been split into multiple modules for better maintainability.
// Implementations are located in:
//   - scene_ui_camera.cpp    : Camera settings (drawCameraContent)
//   - scene_ui_materials.cpp : Material editor (drawMaterialPanel)
//   - scene_ui_hierarchy.cpp : Scene tree (drawSceneHierarchy)
//   - scene_ui_lights.cpp    : Lights panel (drawLightsContent)
//   - scene_ui_gizmos.cpp    : 3D gizmos & bounding boxes
//   - scene_ui_viewport.cpp  : Overlays (Focus/Zoom/Exposure/Dolly)
//   - scene_ui_selection.cpp : Selection logic & Marquee
//   - scene_ui_world.cpp     : World environment settings
// ═══════════════════════════════════════════════════════════════════════════════

#include "scene_ui.h"
#include <thread>
#include <filesystem>
#include <algorithm>
#include <fstream>
#include "json.hpp"
#include "ProjectManager.h"
#include "TerrainManager.h"
#include "SceneSerializer.h"
#include "renderer.h"
#include "OptixWrapper.h"
#include "ColorProcessingParams.h"
#include "SceneSelection.h"
#include "scene_data.h"    // Added explicit include
#include "ui_modern.h"
#include "imgui.h"
#include "ImGuizmo.h"  // Transform gizmo
#include <string>
#include <memory>  // For std::make_unique
#include "KeyframeSystem.h"   // For keyframe animation
#include "scene_ui_guides.hpp" // Viewport guides (safe areas, letterbox, grids)
#include "TimelineWidget.h"   // Custom timeline widget
#include "scene_data.h"
#include "scene_ui_water.hpp"   // Water panel implementation
#include "scene_ui_river.hpp"   // River spline editor
#include "scene_ui_terrain.hpp" // Terrain panel implementation
#include "scene_ui_animgraph.hpp" // Animation Graph Editor
#include "scene_ui_gas.hpp"     // Gas Simulation panel
#include "scene_ui_forcefield.hpp" // Force Field panel
#include "ParallelBVHNode.h"
#include "Triangle.h"  // For object hierarchy
#include "PointLight.h"
#include "DirectionalLight.h"
#include "SpotLight.h"
#include "AreaLight.h"
#include "PrincipledBSDF.h" // For material editing
#include "Volumetric.h"     // For volumetric material
#include "VDBVolume.h"      // For VDB volume UI panel
#include "AssimpLoader.h"  // For scene rebuild after object changes
#include "SceneCommand.h"  // For undo/redo
#include "default_scene_creator.hpp"
#include "SceneSerializer.h"
#include "ProjectManager.h"  // Project system
#include "MaterialManager.h"  // For material editing
#include <map>  // For mesh grouping
#include <unordered_set>  // For fast deletion lookup
#include <windows.h>
#include <commdlg.h>
#include <shlobj.h>  // SHBrowseForFolder için
#include <chrono>  // Playback timing için
#include <filesystem>  // Frame dosyalarını kontrol için

bool show_documentation_window = false; // Global toggle (unused now, kept for linkage if needed or remove)



static int new_width = image_width;
static int new_height = image_height;
static int aspect_w = 16;
static int aspect_h = 9;
static bool modelLoaded = false;
static bool loadFeedback = false; // geçici hata geri bildirimi
static float feedbackTimer = 0.0f;
// show_animation_panel is now a member of SceneUI class

// Pivot Mode State: 0=Median Point (Group), 1=Individual Origins
// Pivot Mode State is now a member of SceneUI class (see scene_ui.h) 

// Not: ScaleColor ve HelpMarker artık UIWidgets namespace'inde tanımlı

struct ResolutionPreset {
    const char* name;
    int w, h;
    int bw, bh;
};

static ResolutionPreset presets[] = {
    { "Custom", 0,0,0,0 },
    { "HD 720p", 1280,720, 16,9 },
    { "Full HD 1080p", 1920,1080, 16,9 },
    { "1440p", 2560,1440, 16,9 },
    { "4K UHD", 3840,2160, 16,9 },
    { "DCI 2K", 2048,1080, 19,10 },
    { "DCI 4K", 4096,2160, 19,10 },
    { "CinemaScope 4K", 4096,1716, 239,100 },
    { "Scope HD", 1920,804, 239,100 },
    { "2.35:1 HD", 1920,817, 235,100 },
    { "Vertical 1080x1920", 1080,1920, 9,16 }
};

static int preset_index = 0;




std::string SceneUI::openFileDialogW(const wchar_t* filter, const std::string& initialDir, const std::string& defaultFilename) {
    wchar_t filename[MAX_PATH] = L"";
    wchar_t initialDirW[MAX_PATH] = L"";
    
    // Convert initial directory to wide string if provided
    if (!initialDir.empty()) {
        MultiByteToWideChar(CP_UTF8, 0, initialDir.c_str(), -1, initialDirW, MAX_PATH);
    }

    // Convert default filename to wide string if provided
    if (!defaultFilename.empty()) {
        MultiByteToWideChar(CP_UTF8, 0, defaultFilename.c_str(), -1, filename, MAX_PATH);
    }
    
    OPENFILENAMEW ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = nullptr;
    ofn.lpstrFilter = filter;
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
    ofn.lpstrTitle = L"Select a file";
    ofn.hwndOwner = GetActiveWindow();
    ofn.lpstrInitialDir = initialDir.empty() ? nullptr : initialDirW;
    
    if (GetOpenFileNameW(&ofn)) {
        int size_needed = WideCharToMultiByte(CP_UTF8, 0, filename, -1, nullptr, 0, nullptr, nullptr);
        std::string utf8_path(size_needed, 0);
        WideCharToMultiByte(CP_UTF8, 0, filename, -1, utf8_path.data(), size_needed, nullptr, nullptr);
        utf8_path.resize(size_needed - 1); // null terminatörü çıkar
        return utf8_path;
    }
    return "";
}

std::string SceneUI::saveFileDialogW(const wchar_t* filter, const wchar_t* defExt) {
    wchar_t filename[MAX_PATH] = L"";
    OPENFILENAMEW ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = GetActiveWindow();
    ofn.lpstrFilter = filter;
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH; // Initialize buffer with 0
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT | OFN_NOCHANGEDIR;
    ofn.lpstrDefExt = defExt;
    
    if (GetSaveFileNameW(&ofn)) {
        int size_needed = WideCharToMultiByte(CP_UTF8, 0, filename, -1, nullptr, 0, nullptr, nullptr);
        std::string utf8_path(size_needed, 0);
        WideCharToMultiByte(CP_UTF8, 0, filename, -1, utf8_path.data(), size_needed, nullptr, nullptr);
        utf8_path.resize(size_needed - 1); 
        return utf8_path;
    }
    return "";
}

std::string SceneUI::selectFolderDialogW(const wchar_t* title) {
    BROWSEINFOW bi = { 0 };
    bi.lpszTitle = title;
    bi.ulFlags = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE | BIF_USENEWUI;
    bi.hwndOwner = GetActiveWindow();

    LPITEMIDLIST pidl = SHBrowseForFolderW(&bi);
    if (pidl != nullptr) {
        wchar_t path[MAX_PATH];
        if (SHGetPathFromIDListW(pidl, path)) {
            int size_needed = WideCharToMultiByte(CP_UTF8, 0, path, -1, nullptr, 0, nullptr, nullptr);
            std::string utf8_path(size_needed, 0);
            WideCharToMultiByte(CP_UTF8, 0, path, -1, utf8_path.data(), size_needed, nullptr, nullptr);
            utf8_path.resize(size_needed - 1);

            // Free pidl
            IMalloc* imalloc = nullptr;
            if (SUCCEEDED(SHGetMalloc(&imalloc))) {
                imalloc->Free(pidl);
                imalloc->Release();
            }

            return utf8_path;
        }
    }
    return "";
}

static std::string active_model_path = "No file selected yet.";

// ═══════════════════════════════════════════════════════════
// GLOBAL UI WIDGETS IMPLEMENTATION
// ═══════════════════════════════════════════════════════════

// Initialize default theme (Retro Monochrome)
SceneUI::LCDTheme SceneUI::currentTheme = {
    IM_COL32(200, 200, 200, 255), // Lit: White-ish
    IM_COL32(40, 45, 50, 255),    // Off: Dark Gray
    IM_COL32(20, 20, 20, 255),    // Bg: Almost Black
    IM_COL32(180, 230, 255, 255), // Text: Light Cyan
    false
};

// Initialize default style
SceneUI::UISliderStyle SceneUI::globalSliderStyle = SceneUI::UISliderStyle::Modern;

bool SceneUI::DrawSmartFloat(const char* id, const char* label, float* value, float min, float max, 
                           const char* format, bool keyed, 
                           std::function<void()> onKeyframeClick, int segments) 
{
    if (globalSliderStyle == UISliderStyle::RetroLCD) {
        return DrawLCDSlider(id, label, value, min, max, format, keyed, onKeyframeClick, segments);
    }
    else {
        // Modern / Standard Style
        ImGui::PushID(id);
        bool changed = false;
        
        // Keyframe Button (if callback provided)
        if (onKeyframeClick) {
            float s = ImGui::GetFrameHeight();
            ImVec2 kf_pos = ImGui::GetCursorScreenPos();
            bool kf_clicked = ImGui::InvisibleButton("kf", ImVec2(s, s));
            
            // Standard diamond drawing
            ImU32 kf_bg = keyed ? IM_COL32(255, 200, 0, 255) : IM_COL32(40, 40, 40, 255);
            ImU32 kf_border = IM_COL32(180, 180, 180, 255);
            if (ImGui::IsItemHovered()) {
                kf_border = IM_COL32(255, 255, 255, 255);
                ImGui::SetTooltip(keyed ? "Click to REMOVE keyframe" : "Click to ADD keyframe");
            }
            ImDrawList* dl = ImGui::GetWindowDrawList();
            float cx = kf_pos.x + s * 0.5f;
            float cy = kf_pos.y + s * 0.5f;
            float r = s * 0.22f;
            ImVec2 p[4] = { ImVec2(cx, cy - r), ImVec2(cx + r, cy), ImVec2(cx, cy + r), ImVec2(cx - r, cy) };
            dl->AddQuadFilled(p[0], p[1], p[2], p[3], kf_bg);
            dl->AddQuad(p[0], p[1], p[2], p[3], kf_border, 1.0f);
            
            if (kf_clicked) onKeyframeClick();
            ImGui::SameLine();
        }

        // Standard Slider
        // Adjust width if keyframe button exists or not to align
        if (ImGui::SliderFloat(label, value, min, max, format)) {
            changed = true;
        }
        
        ImGui::PopID();
        return changed;
    }
}


bool SceneUI::DrawLCDSlider(const char* id, const char* label, float* value, float min, float max, 
                          const char* format, bool keyed, 
                          std::function<void()> onKeyframeClick, int segments) 
{
    ImGui::PushID(id);
    bool changed = false;
    float t = (*value - min) / (max - min);
    int lit = (int)(t * segments);
    
    // Keyframe diamond button (Only if callback provided)
    if (onKeyframeClick) {
        float s = ImGui::GetFrameHeight();
        ImVec2 kf_pos = ImGui::GetCursorScreenPos();
        bool kf_clicked = ImGui::InvisibleButton("kf", ImVec2(s, s));
        
        ImU32 kf_bg = keyed ? IM_COL32(100, 180, 255, 255) : IM_COL32(40, 40, 40, 255);
        if (currentTheme.isRetroGreen) {
            kf_bg = keyed ? IM_COL32(50, 255, 50, 255) : IM_COL32(20, 50, 20, 255);
        }
        
        ImU32 kf_border = ImGui::IsItemHovered() ? IM_COL32(255, 255, 255, 255) : IM_COL32(150, 150, 150, 255);
        if (ImGui::IsItemHovered()) {
            if (!currentTheme.isRetroGreen)
                kf_bg = keyed ? IM_COL32(120, 200, 255, 255) : IM_COL32(70, 70, 70, 255);
            ImGui::SetTooltip(keyed ? "%s: Click to REMOVE keyframe" : "%s: Click to ADD keyframe", label);
        }
        
        ImDrawList* dl = ImGui::GetWindowDrawList();
        float cx = kf_pos.x + s * 0.5f;
        float cy = kf_pos.y + s * 0.5f;
        float r = s * 0.22f;
        dl->AddQuadFilled(ImVec2(cx, cy-r), ImVec2(cx+r, cy), ImVec2(cx, cy+r), ImVec2(cx-r, cy), kf_bg);
        dl->AddQuad(ImVec2(cx, cy-r), ImVec2(cx+r, cy), ImVec2(cx, cy+r), ImVec2(cx-r, cy), kf_border, 1.0f);
        
        if (kf_clicked) onKeyframeClick();
        
        ImGui::SameLine();
    } else {
        // Just add some spacing if no keyframe button
        ImGui::Dummy(ImVec2(ImGui::GetFrameHeight(), ImGui::GetFrameHeight()));
        ImGui::SameLine();
    }
    
    // Label (fixed width)
    ImGui::Text("%-6s", label);
    ImGui::SameLine();
    
    // LCD Bar
    ImVec2 bar_pos = ImGui::GetCursorScreenPos();
    float segW = 6.0f;
    float segH = 14.0f;
    float gap = 2.0f;
    float totalW = segments * (segW + gap);
    
    ImDrawList* dl = ImGui::GetWindowDrawList();
    
    for (int i = 0; i < segments; i++) {
        float x = bar_pos.x + i * (segW + gap);
        ImU32 color;
        if (i < lit) {
            color = currentTheme.litColor;
        } else {
            color = currentTheme.offColor;
        }
        dl->AddRectFilled(ImVec2(x, bar_pos.y), ImVec2(x + segW, bar_pos.y + segH), color, 1.0f);
        dl->AddRect(ImVec2(x, bar_pos.y), ImVec2(x + segW, bar_pos.y + segH), currentTheme.bgColor, 1.0f);
    }
    
    // Invisible slider over the bar
    ImGui::SetCursorScreenPos(bar_pos);
    ImGui::InvisibleButton("bar", ImVec2(totalW, segH));
    if (ImGui::IsItemActive()) {
        float mx = ImGui::GetIO().MousePos.x - bar_pos.x;
        float newT = std::clamp(mx / totalW, 0.0f, 1.0f);
        *value = min + newT * (max - min);
        changed = true;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
        ImGui::SetTooltip("Drag to adjust value");
    }
    
    ImGui::SameLine();
    
    // Value Input
    ImGui::PushItemWidth(60);
    char inputId[64];
    snprintf(inputId, sizeof(inputId), "##input_%s", id);
    
    // Style input specific to theme
    ImGui::PushStyleColor(ImGuiCol_Text, currentTheme.textValColor);
    if (ImGui::InputFloat(inputId, value, 0.0f, 0.0f, format)) {
        *value = std::clamp(*value, min, max); // Clamp manual input
        changed = true;
    }
    ImGui::PopStyleColor();
    ImGui::PopItemWidth();
    
    ImGui::PopID();
    return changed;
}


// Helper Methods Implementation
void SceneUI::ClampWindowToDisplay()
{
    ImGuiIO& io = ImGui::GetIO();
    ImVec2 disp = io.DisplaySize;

    ImVec2 win_pos = ImGui::GetWindowPos();
    ImVec2 win_size = ImGui::GetWindowSize();

    // Eğer pencere invisible veya 0 boyutluysa çık
    if (win_size.x <= 0.0f || win_size.y <= 0.0f) return;

    float x = win_pos.x;
    float y = win_pos.y;

    // Sağ/bottom taşmaları düzelt
    if (x + win_size.x > disp.x) x = disp.x - win_size.x;
    if (y + win_size.y > disp.y) y = disp.y - win_size.y;

    // Negatif değerlere izin verme
    if (x < 0.0f) x = 0.0f;
    if (y < 0.0f) y = 0.0f;

    // Pozisyon değiştiyse uygula
    if (x != win_pos.x || y != win_pos.y) {
        ImGui::SetWindowPos(ImVec2(x, y), ImGuiCond_Always);
    }

    // Eğer pencere ekran boyutuna göre taşarsa, boyutu da düzelt
    bool size_changed = false;
    float new_width = win_size.x;
    float new_height = win_size.y;

    if (win_size.x > disp.x) { new_width = disp.x; size_changed = true; }
    if (win_size.y > disp.y) { new_height = disp.y; size_changed = true; }

    if (size_changed) {
        ImGui::SetWindowSize(ImVec2(new_width, new_height), ImGuiCond_Always);
    }
}

// Timeline Panel - Blender-style Custom Timeline Widget
void SceneUI::drawTimelineContent(UIContext& ctx)
{
    // Use the timeline member widget
    timeline.draw(ctx);
}


// Wrapper for compatibility (if needed) but essentially deprecated as a window creator
void SceneUI::drawTimelinePanel(UIContext& ctx, float screen_y) {
    drawTimelineContent(ctx);
}

// Eski drawAnimationSettings metodunu kaldırdık, artık kullanılmıyor
void SceneUI::drawAnimationSettings(UIContext& ctx)
{
    // Bu metod artık kullanılmıyor - timeline panel'e taşındı
}

void SceneUI::drawLogPanelEmbedded()
{
    ImFont* tinyFont = ImGui::GetIO().Fonts->Fonts.back();
    ImGui::PushFont(tinyFont);

    // Başlık reset zamanlayıcıları (global/statik olarak zaten tanımlı olmalı)
    if (titleChanged && ImGui::GetTime() > titleResetTime) {
        logTitle = "Scene Log";
        titleChanged = false;
    }

    // Başlık rengi varsa uygulayıp geri al
    if (titleChanged)
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.3f, 1.0f, 0.3f, 1.0f));

    // AllowItemOverlap ile header oluşturuyoruz — böylece aynı satırda başka butonlar çalışır
    ImGuiTreeNodeFlags hdrFlags = ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_AllowItemOverlap;
    bool open = ImGui::CollapsingHeader(logTitle.c_str(), hdrFlags);

    if (titleChanged)
        ImGui::PopStyleColor();

    // Header ile aynı satıra Copy butonunu koy
    // İtem örtüşmesine izin verdiğimiz için buton tıklanabilir kalır
    float avail = ImGui::GetContentRegionAvail().x;
    ImGui::SameLine(avail - 60.0f); // butonu sağa sabitliyoruz (60 px boşluk bırak)
    if (ImGui::SmallButton("Copy"))
    {
        std::vector<LogEntry> lines;
        g_sceneLog.getLines(lines);

        std::string total;
        total.reserve(lines.size() * 64);
        for (auto& e : lines) {
            const char* prefix =
                (e.level == LogLevel::Info) ? "INFO" :
                (e.level == LogLevel::Warning) ? "WARN" : "ERROR";
            total += "[" + std::string(prefix) + "] " + e.msg + "\n";
        }

        ImGui::SetClipboardText(total.c_str());

        // Başlığa kısa süreli bildirim ver
        logTitle = "Scene Log  (Copied)";
        titleResetTime = ImGui::GetTime() + 2.0f;
        titleChanged = true;
    }

    // Eğer header açıksa logu göster
    if (open)
    {
        // Mevcut boşluğu doldur (en az 100px)
        float avail_y = ImGui::GetContentRegionAvail().y;
        if (avail_y < 100.0f) avail_y = 150.0f; 

        ImGui::BeginChild("scroll_log", ImVec2(0, avail_y), true);

        static size_t lastCount = 0;
        std::vector<LogEntry> lines;
        g_sceneLog.getLines(lines);

        for (auto& e : lines)
        {
            ImVec4 color =
                (e.level == LogLevel::Info) ? ImVec4(1, 1, 1, 1) :
                (e.level == LogLevel::Warning) ? ImVec4(1, 1, 0, 1) :
                ImVec4(1, 0, 0, 1);

            const char* prefix =
                (e.level == LogLevel::Info) ? "INFO" :
                (e.level == LogLevel::Warning) ? "WARN" : "ERROR";

            ImGui::TextColored(color, "[%s] %s", prefix, e.msg.c_str());
        }

        if (lines.size() > lastCount)
            ImGui::SetScrollHereY(1.0f);
        lastCount = lines.size();

        ImGui::EndChild();
    }

    ImGui::PopFont();
}

void SceneUI::drawThemeSelector() {
    UIWidgets::DrawThemeSelector(panel_alpha);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // ═══════════════════════════════════════════════════════════
    // LCD WIDGET THEME
    // ═══════════════════════════════════════════════════════════
    if (ImGui::CollapsingHeader("LCD Widget Theme", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Indent();
        
        static bool init_theme = true;
        
        // ── SLIDER STYLE SELECTOR ──
        int styleIdx = (globalSliderStyle == UISliderStyle::Modern) ? 0 : 1;
        if (ImGui::Combo("Slider Style", &styleIdx, "Modern (Standard)\0Retro LCD\0")) {
            globalSliderStyle = (styleIdx == 0) ? UISliderStyle::Modern : UISliderStyle::RetroLCD;
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        
        static int theme_preset = 0;
        
        // Ensure initialization on first run to prevent zero-alpha bugs
        if (init_theme) {
             theme_preset = 0;
             currentTheme.litColor = IM_COL32(200, 200, 200, 255);
             currentTheme.offColor = IM_COL32(40, 45, 50, 255);
             currentTheme.bgColor = IM_COL32(20, 20, 20, 255);
             currentTheme.textValColor = IM_COL32(180, 230, 255, 255);
             currentTheme.isRetroGreen = false;
             init_theme = false;
        }

        if (ImGui::Combo("Preset", &theme_preset, "Retro Monochrome\0Classic Green\0Amber\0Cyberpunk Blue\0Custom\0")) {
            switch (theme_preset) {
                case 0: // Mono
                    currentTheme.litColor = IM_COL32(200, 200, 200, 255);
                    currentTheme.offColor = IM_COL32(40, 45, 50, 255);
                    currentTheme.bgColor = IM_COL32(20, 20, 20, 255);
                    currentTheme.textValColor = IM_COL32(180, 230, 255, 255);
                    currentTheme.isRetroGreen = false;
                    break;
                case 1: // Green
                    currentTheme.litColor = IM_COL32(50, 255, 50, 255);
                    currentTheme.offColor = IM_COL32(20, 50, 20, 255);
                    currentTheme.bgColor = IM_COL32(10, 20, 10, 255);
                    currentTheme.textValColor = IM_COL32(100, 255, 100, 255);
                    currentTheme.isRetroGreen = true;
                    break;
                case 2: // Amber
                    currentTheme.litColor = IM_COL32(255, 180, 20, 255);
                    currentTheme.offColor = IM_COL32(60, 40, 10, 255);
                    currentTheme.bgColor = IM_COL32(20, 15, 5, 255);
                    currentTheme.textValColor = IM_COL32(255, 200, 100, 255);
                    currentTheme.isRetroGreen = false;
                    break;
                case 3: // Cyberpunk
                    currentTheme.litColor = IM_COL32(0, 255, 255, 255);
                    currentTheme.offColor = IM_COL32(0, 50, 60, 255);
                    currentTheme.bgColor = IM_COL32(5, 10, 20, 255);
                    currentTheme.textValColor = IM_COL32(0, 255, 255, 255);
                    currentTheme.isRetroGreen = false;
                    break;
                case 4: // Custom
                    // Ensure alpha is full if coming from an uninit state (just safety)
                    if ((currentTheme.litColor & 0xFF000000) == 0) currentTheme.litColor |= 0xFF000000;
                    if ((currentTheme.offColor & 0xFF000000) == 0) currentTheme.offColor |= 0xFF000000;
                    if ((currentTheme.bgColor & 0xFF000000) == 0) currentTheme.bgColor |= 0xFF000000;
                    if ((currentTheme.textValColor & 0xFF000000) == 0) currentTheme.textValColor |= 0xFF000000;
                    break;
            }
        }
        
        // Manual Color Overrides
        ImGui::Text("Custom Colors");
        ImGui::Spacing();

        bool custom_changed = false;
        ImVec4 colLit = ImGui::ColorConvertU32ToFloat4(currentTheme.litColor);
        if (ImGui::ColorEdit3("Lit Color", &colLit.x)) { 
            currentTheme.litColor = ImGui::ColorConvertFloat4ToU32(colLit); 
            theme_preset = 4; // Switch to Custom
        }

        ImVec4 colOff = ImGui::ColorConvertU32ToFloat4(currentTheme.offColor);
        if (ImGui::ColorEdit3("Off Color", &colOff.x)) { 
            currentTheme.offColor = ImGui::ColorConvertFloat4ToU32(colOff); 
            theme_preset = 4;
        }
        
        ImVec4 colBg = ImGui::ColorConvertU32ToFloat4(currentTheme.bgColor);
        if (ImGui::ColorEdit3("Background", &colBg.x)) { 
            currentTheme.bgColor = ImGui::ColorConvertFloat4ToU32(colBg); 
            theme_preset = 4;
        }

        ImVec4 colTxt = ImGui::ColorConvertU32ToFloat4(currentTheme.textValColor);
        if (ImGui::ColorEdit3("Text Color", &colTxt.x)) { 
            currentTheme.textValColor = ImGui::ColorConvertFloat4ToU32(colTxt); 
            theme_preset = 4;
        }

        ImGui::Checkbox("Retro Keyframe Style", &currentTheme.isRetroGreen);
        
        ImGui::Unindent();
    }
}
void SceneUI::drawResolutionPanel(UIContext& ctx)
{
    if (UIWidgets::BeginSection("System & Output", ImVec4(0.4f, 0.8f, 0.6f, 1.0f))) {
        

        UIWidgets::ColoredHeader("Resolution", ImVec4(0.7f, 0.9f, 0.8f, 1.0f));
        
        if (UIWidgets::BeginSection("Display Settings", ImVec4(0.5f, 0.5f, 0.6f, 1.0f))) {
            if (ImGui::Combo("Presets", &preset_index,
                [](void* data, int idx, const char** out_text) {
                    *out_text = ((ResolutionPreset*)data)[idx].name;
                    return true;
                }, presets, IM_ARRAYSIZE(presets)))
            {
                if (preset_index != 0) {
                    new_width = presets[preset_index].w;
                    new_height = presets[preset_index].h;
                    aspect_w = presets[preset_index].bw;
                    aspect_h = presets[preset_index].bh;
                }
            }

            ImGui::Spacing();
            ImGui::PushItemWidth(150);
            ImGui::InputInt("Width", &new_width);
            ImGui::InputInt("Height", &new_height);
            ImGui::PopItemWidth();
            
            ImGui::Spacing();
            ImGui::PushItemWidth(100);
            ImGui::InputInt("Aspect W", &aspect_w);
            ImGui::SameLine();
            ImGui::InputInt("Aspect H", &aspect_h);
            ImGui::PopItemWidth();

            bool resolution_changed =
                (new_width != last_applied_width) ||
                (new_height != last_applied_height) ||
                (aspect_w != last_applied_aspect_w) ||
                (aspect_h != last_applied_aspect_h);

            ImGui::Spacing();
            
            if (UIWidgets::PrimaryButton("Apply Resolution", ImVec2(150, 0), resolution_changed))
            {
                float ar = aspect_h ? float(aspect_w) / aspect_h : 1.0f;
                pending_aspect_ratio = ar;
                pending_width = new_width;
                pending_height = new_height;
                aspect_ratio = ar;
                pending_resolution_change = true;

                last_applied_width = new_width;
                last_applied_height = new_height;
                last_applied_aspect_w = aspect_w;
                last_applied_aspect_h = aspect_h;
            }
            
            UIWidgets::EndSection();
        }

        UIWidgets::EndSection();
    }
}


static void DrawRenderWindowToneMapControls(UIContext& ctx) {
    UIWidgets::ColoredHeader("Post-Processing Controls", ImVec4(1.0f, 0.65f, 0.6f, 1.0f));
    UIWidgets::Divider();

    // -------- Main Parameters --------
    if (UIWidgets::BeginSection("Main Post-Processing", ImVec4(0.8f, 0.6f, 0.5f, 1.0f))) {
        UIWidgets::SliderWithHelp("Gamma", &ctx.color_processor.params.global_gamma, 
                                   0.5f, 3.0f, "Controls overall image brightness curve");
        UIWidgets::SliderWithHelp("Exposure", &ctx.color_processor.params.global_exposure, 
                                   0.1f, 5.0f, "Adjusts overall brightness level");
        UIWidgets::SliderWithHelp("Saturation", &ctx.color_processor.params.saturation, 
                                   0.0f, 2.0f, "Controls color intensity");
        UIWidgets::SliderWithHelp("Temperature (K)", &ctx.color_processor.params.color_temperature, 
                                   1000.0f, 10000.0f, "Color temperature in Kelvin", "%.0f");
        UIWidgets::EndSection();
    }

    // -------- Tonemapping Type --------
    if (UIWidgets::BeginSection("Tonemapping Type", ImVec4(0.6f, 0.7f, 0.9f, 1.0f))) {
        const char* tone_names[] = { "AGX", "ACES", "Uncharted", "Filmic", "None" };
        int selected_tone = static_cast<int>(ctx.color_processor.params.tone_mapping_type);
        if (ImGui::Combo("Tonemapping", &selected_tone, tone_names, IM_ARRAYSIZE(tone_names))) {
            ctx.color_processor.params.tone_mapping_type = static_cast<ToneMappingType>(selected_tone);
        }
        UIWidgets::HelpMarker("AGX: Balanced look | ACES: Cinema standard | Filmic: Classic film");
        UIWidgets::EndSection();
    }

    // -------- Effects --------
    if (UIWidgets::BeginSection("Effects", ImVec4(0.7f, 0.5f, 0.8f, 1.0f))) {
        ImGui::Checkbox("Vignette", &ctx.color_processor.params.enable_vignette);
        if (ctx.color_processor.params.enable_vignette) {
            UIWidgets::SliderWithHelp("Vignette Strength", &ctx.color_processor.params.vignette_strength, 
                                       0.0f, 2.0f, "Darkening around image edges");
        }
        UIWidgets::EndSection();
    }

    // -------- Actions --------
    UIWidgets::Divider();
    
    if (UIWidgets::PrimaryButton("Apply Tonemap", ImVec2(120, 0))) 
        ctx.apply_tonemap = true;
    ImGui::SameLine();
    if (UIWidgets::SecondaryButton("Reset", ImVec2(80, 0))) 
        ctx.reset_tonemap = true;
}



// drawWorldContent moved to scene_ui_world.cpp

void SceneUI::drawRenderSettingsPanel(UIContext& ctx, float screen_y)
{
    // Dinamik yükseklik hesabı
    bool bottom_visible = show_animation_panel || show_scene_log;
    float bottom_margin = bottom_visible ? (bottom_panel_height + 24.0f) : 24.0f; // Panel + StatusBar

    float menu_height = 19.0f; 
    float target_height = screen_y - menu_height - bottom_margin;

    // Panel ayarları
    // Lock Height to target_height (MinY = MaxY), allow Width resize (300-800)
    ImGui::SetNextWindowSizeConstraints(
        ImVec2(300, target_height),                 
        ImVec2(800, target_height) 
    );

    // LEFT SIDE DOCKING
    ImGuiIO& io = ImGui::GetIO();
    
    // Position at (0, menu_height) -> TOP LEFT
    ImGui::SetNextWindowPos(ImVec2(0, menu_height), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(side_panel_width, target_height), ImGuiCond_FirstUseEver);

    // Remove TitleBar and Resize for a seamless docked look
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse;

    // Add frame styling
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.0f);
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.3f, 0.3f, 0.35f, 1.0f));

    if (ImGui::Begin("Properties", nullptr, flags))
    {
        ImDrawList* parent_dl = ImGui::GetWindowDrawList();
        ImVec2 win_pos = ImGui::GetWindowPos();
        ImVec2 win_size = ImGui::GetWindowSize();
        // Update width if user resized
        side_panel_width = ImGui::GetWindowWidth();

        // ─────────────────────────────────────────────────────────────────────────
        // MODERN VERTICAL TAB NAVIGATION
        // ─────────────────────────────────────────────────────────────────────────

        // Sync tab_to_focus with vertical tabs
        if (tab_to_focus == "Scene Edit") { active_properties_tab = 0; tab_to_focus = ""; }
        if (tab_to_focus == "Render")     { active_properties_tab = 1; tab_to_focus = ""; }
        if (tab_to_focus == "Terrain")    { active_properties_tab = 2; tab_to_focus = ""; }
        if (tab_to_focus == "Water")      { active_properties_tab = 3; tab_to_focus = ""; }
        if (tab_to_focus == "Volumetric" || tab_to_focus == "VDB" || tab_to_focus == "Gas") { active_properties_tab = 4; tab_to_focus = ""; }
        if (tab_to_focus == "Force Field"){ active_properties_tab = 5; tab_to_focus = ""; }
        if (tab_to_focus == "World")      { active_properties_tab = 6; tab_to_focus = ""; }
        if (tab_to_focus == "System")     { active_properties_tab = 7; tab_to_focus = ""; }

        float sidebar_width = 46.0f;
        
        // --- 1. SIDEBAR (Fixed Width) ---
        ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 8));
        
        // Sidebar background: slightly darker than main window for contrast
        ImVec4 sidebarBg = ImGui::GetStyleColorVec4(ImGuiCol_WindowBg);
        sidebarBg.x *= 0.85f; sidebarBg.y *= 0.85f; sidebarBg.z *= 0.85f;
        ImGui::PushStyleColor(ImGuiCol_ChildBg, sidebarBg);
        
        ImGui::BeginChild("PropSidebar", ImVec2(sidebar_width, 0), false, ImGuiWindowFlags_NoScrollbar);
        
        // Add a vertical line to separate sidebar - Use Parent DL to avoid clipping
        parent_dl->AddLine(
            ImVec2(win_pos.x + sidebar_width - 1, win_pos.y),
            ImVec2(win_pos.x + sidebar_width - 1, win_pos.y + win_size.y),
            ImGui::ColorConvertFloat4ToU32(ImVec4(1.0f, 1.0f, 1.0f, 0.08f)) // Faint border
        );

        auto drawTabButton = [&](int index, UIWidgets::IconType icon, const char* tooltip) {
            bool is_active = (active_properties_tab == index);
            ImGui::PushID(index);
            
            ImVec2 pos = ImGui::GetCursorScreenPos();
            float size = 36.0f; // Slightly smaller buttons
            float margin = (sidebar_width - size) * 0.5f;

            ImGui::SetCursorPosX(margin);
            
            if (is_active) {
                // Connection Bridge: Use Parent DL to bleed across the child border
                parent_dl->AddRectFilled(
                    ImVec2(pos.x - margin, pos.y), 
                    ImVec2(pos.x + sidebar_width + 2, pos.y + size), 
                    ImGui::ColorConvertFloat4ToU32(ImGui::GetStyleColorVec4(ImGuiCol_WindowBg)),
                    0.0f
                );

                // Indicator on the right edge of the sidebar
                parent_dl->AddRectFilled(
                    ImVec2(win_pos.x + sidebar_width - 3, pos.y + 4), 
                    ImVec2(win_pos.x + sidebar_width, pos.y + size - 4), 
                    ImGui::ColorConvertFloat4ToU32(ImVec4(0.05f, 0.75f, 0.65f, 1.0f)),
                    2.0f
                );

                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(1,1,1, 0.05f));
            } else {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0)); 
            }

            if (ImGui::Button("##tab", ImVec2(size, size))) {
                active_properties_tab = index;
            }
            
            // Draw Icon
            ImU32 iconCol = is_active ? ImGui::ColorConvertFloat4ToU32(ImVec4(0.1f, 0.9f, 0.8f, 1.0f)) : ImGui::ColorConvertFloat4ToU32(ImVec4(0.55f, 0.55f, 0.6f, 1.0f));
            UIWidgets::DrawIcon(icon, ImVec2(pos.x + (size-20)*0.5f, pos.y + (size-20)*0.5f), 20.0f, iconCol, 1.5f);

            if (ImGui::IsItemHovered()) {
                ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                ImGui::BeginTooltip();
                ImGui::TextUnformatted(tooltip);
                ImGui::EndTooltip();
            }
            
            ImGui::PopStyleColor(1);
            ImGui::PopID();
        };

        ImGui::Spacing(); // Top spacing
        drawTabButton(0, UIWidgets::IconType::Scene,      "Scene / Hierarchy");
        drawTabButton(1, UIWidgets::IconType::Render,     "Render Settings");
        if (show_terrain_tab)    drawTabButton(2, UIWidgets::IconType::Terrain,    "Terrain Editor");
        if (show_water_tab)      drawTabButton(3, UIWidgets::IconType::Water,      "Water & Rivers");
        if (show_volumetric_tab) drawTabButton(4, UIWidgets::IconType::Volumetric, "Volumetrics");
        if (show_forcefield_tab) drawTabButton(5, UIWidgets::IconType::Force,      "Force Fields");
        if (show_world_tab)      drawTabButton(6, UIWidgets::IconType::World,      "World & Sky");
        if (show_system_tab)     drawTabButton(7, UIWidgets::IconType::System,     "System & UI");
        
        ImGui::EndChild();
        ImGui::PopStyleColor();
        ImGui::PopStyleVar(2);
        
        ImGui::SameLine(0, 0);
        
        // --- 2. CONTENT AREA (Inspector Style) ---
        ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 0.0f);
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImGui::GetStyleColorVec4(ImGuiCol_WindowBg)); // Base background
        
        ImGui::BeginChild("PropContentArea", ImVec2(0, 0), false, ImGuiWindowFlags_NoScrollbar);
        
        // ── TAB HEADER (Sticky & Flush) ──
        const char* tab_names[] = { "SCENE HIERARCHY", "RENDER SETTINGS", "TERRAIN EDITOR", "WATER & RIVERS", "VOLUMETRICS", "FORCE FIELDS", "WORLD & SKY", "SYSTEM & UI" };
        UIWidgets::IconType tab_icons[] = { 
            UIWidgets::IconType::Scene, UIWidgets::IconType::Render, UIWidgets::IconType::Terrain, 
            UIWidgets::IconType::Water, UIWidgets::IconType::Volumetric, UIWidgets::IconType::Force, 
            UIWidgets::IconType::World, UIWidgets::IconType::System 
        };

        ImGui::BeginChild("TabHeader", ImVec2(0, 42), false, ImGuiWindowFlags_NoScrollbar);
        
        ImVec2 headerPos = ImGui::GetCursorScreenPos();
        // Background for the sticky header
        ImGui::GetWindowDrawList()->AddRectFilled(headerPos, ImVec2(headerPos.x + ImGui::GetContentRegionAvail().x, headerPos.y + 42), 
            ImGui::ColorConvertFloat4ToU32(ImVec4(1,1,1, 0.02f)));

        ImGui::SetCursorPos(ImVec2(12, 11)); // Precise alignment
        UIWidgets::DrawIcon(tab_icons[active_properties_tab], ImGui::GetCursorScreenPos(), 20, ImGui::ColorConvertFloat4ToU32(ImVec4(0.1f, 0.9f, 0.8f, 1.0f)));
        
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 30);
        ImGui::TextColored(ImVec4(0.9f, 0.9f, 1.0f, 1.0f), "%s", tab_names[active_properties_tab]);
        
        // Solid separation line
        ImGui::GetWindowDrawList()->AddLine(
            ImVec2(ImGui::GetWindowPos().x, ImGui::GetWindowPos().y + 41),
            ImVec2(ImGui::GetWindowPos().x + ImGui::GetWindowWidth(), ImGui::GetWindowPos().y + 41),
            ImGui::ColorConvertFloat4ToU32(ImVec4(1,1,1, 0.12f))
        );
        
        ImGui::EndChild();

        // ── MAIN CONTENT (Flush Scroll Area) ──
        ImGui::BeginChild("PropScrollArea", ImVec2(0, 0), false, ImGuiWindowFlags_AlwaysVerticalScrollbar);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0)); // FLUSH LEFT
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8, 12));
        
        // Start Global Indent for controls (leaving headers flush)
        ImGui::Indent(8.0f); 
        ImGui::Spacing();
        ImGui::Unindent(8.0f);

        switch (active_properties_tab) {
            case 0: drawSceneHierarchy(ctx); break;
            case 1: 
                {
                    // ─────────────────────────────────────────────────────────────────────────
                    // ENGINE & BACKEND
                    // ─────────────────────────────────────────────────────────────────────────
                    if (UIWidgets::BeginSection("Render Engine", ImVec4(0.4f, 0.7f, 1.0f, 1.0f))) {
                        if (ImGui::Checkbox("Use OptiX (GPU Acceleration)", &ctx.render_settings.use_optix)) {
                            if (!ctx.render_settings.use_optix) { extern bool g_cpu_sync_pending; g_cpu_sync_pending = true; }
                            ctx.start_render = true;
                        }
                        
                        if (!ctx.render_settings.use_optix) {
                            const char* bvh_items[] = { "Custom RayTrophi BVH", "Intel Embree (Recommended)" };
                            int current_bvh = ctx.render_settings.UI_use_embree ? 1 : 0;
                            if (ImGui::Combo("CPU BVH Type", &current_bvh, bvh_items, 2)) {
                                ctx.render_settings.UI_use_embree = (current_bvh == 1);
                                ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
                                ctx.start_render = true;
                            }
                          
                        }
                        UIWidgets::EndSection();
                    }

                    // ─────────────────────────────────────────────────────────────────────────
                    // SAMPLING (Cycles Style)
                    // ─────────────────────────────────────────────────────────────────────────
                    if (UIWidgets::BeginSection("Sampling", ImVec4(0.5f, 0.9f, 0.6f, 1.0f))) {
                        UIWidgets::ColoredHeader("Viewport", ImVec4(0.7f, 0.7f, 0.7f, 1.0f));
                        ImGui::Checkbox("Use Adaptive Sampling##view", &ctx.render_settings.use_adaptive_sampling);
                        if (ctx.render_settings.use_adaptive_sampling) {
                            ImGui::DragFloat("Noise Threshold", &ctx.render_settings.variance_threshold, 0.001f, 0.001f, 0.5f, "%.3f");
                            ImGui::DragInt("Min Samples##view", &ctx.render_settings.min_samples, 1, 1, 512);
                        }
                        ImGui::DragInt("Max Samples##view", &ctx.render_settings.max_samples, 1, 1, 10000);
                        
                        UIWidgets::Divider();
                        UIWidgets::ColoredHeader("Final Render (F12)", ImVec4(0.7f, 0.7f, 0.7f, 1.0f));
                        ImGui::DragInt("Samples##final", &ctx.render_settings.final_render_samples, 1, 1, 100000);
                        ImGui::Checkbox("Apply Denoiser##final", &ctx.render_settings.render_use_denoiser);
                        
                        UIWidgets::EndSection();
                    }

                    // ─────────────────────────────────────────────────────────────────────────
                    // LIGHT PATHS (Bounces)
                    // ─────────────────────────────────────────────────────────────────────────
                    if (UIWidgets::BeginSection("Light Paths", ImVec4(1.0f, 0.8f, 0.3f, 1.0f))) {
                        ImGui::DragInt("Total Bounces", &ctx.render_settings.max_bounces, 1, 0, 64);
                        ImGui::DragInt("Transparent Bounces", &ctx.render_settings.transparent_max_bounces, 1, 0, 64);
                        
                        UIWidgets::HelpMarker("Higher bounces increase realism for glass and interiors but slow down rendering.");
                        UIWidgets::EndSection();
                    }

                    // ─────────────────────────────────────────────────────────────────────────
                    // DENOISING
                    // ─────────────────────────────────────────────────────────────────────────
                    if (UIWidgets::BeginSection("Denoising", ImVec4(0.8f, 0.5f, 1.0f, 1.0f))) {
                        ImGui::Checkbox("Enable Viewport Denoising", &ctx.render_settings.use_denoiser);
                        if (ctx.render_settings.use_denoiser) {
                            ImGui::SliderFloat("Blend Factor", &ctx.render_settings.denoiser_blend_factor, 0.0f, 1.0f);
                        }
                        UIWidgets::EndSection();
                    }

                    // ─────────────────────────────────────────────────────────────────────────
                    // FORMAT & OUTPUT (Previously in System)
                    // ─────────────────────────────────────────────────────────────────────────
                    if (UIWidgets::BeginSection("Resolution & Output", ImVec4(0.9f, 0.4f, 0.5f, 1.0f))) {
                        // Presets
                        if (ImGui::Combo("Resolution Preset", &preset_index,
                            [](void* data, int idx, const char** out_text) {
                                *out_text = ((ResolutionPreset*)data)[idx].name;
                                return true;
                            }, presets, IM_ARRAYSIZE(presets)))
                        {
                            if (preset_index != 0) {
                                new_width = presets[preset_index].w;
                                new_height = presets[preset_index].h;
                                aspect_w = presets[preset_index].bw;
                                aspect_h = presets[preset_index].bh;
                            }
                        }

                        ImGui::DragInt("Width", &new_width, 1, 1, 8192);
                        ImGui::DragInt("Height", &new_height, 1, 1, 8192);
                        
                        ImGui::PushItemWidth(80);
                        ImGui::DragInt("Aspect W", &aspect_w, 1, 1, 100); ImGui::SameLine();
                        ImGui::DragInt("Aspect H", &aspect_h, 1, 1, 100);
                        ImGui::PopItemWidth();

                        bool resolution_changed = (new_width != last_applied_width) || (new_height != last_applied_height);
                        
                        if (UIWidgets::PrimaryButton("Apply Settings", ImVec2(-1, 30), resolution_changed)) {
                            float ar = aspect_h ? float(aspect_w) / aspect_h : 1.0f;
                            pending_aspect_ratio = ar;
                            pending_width = new_width;
                            pending_height = new_height;
                            aspect_ratio = ar;
                            pending_resolution_change = true;
                            last_applied_width = new_width; last_applied_height = new_height;
                        }

                        UIWidgets::Divider();
                        if (UIWidgets::SecondaryButton("Open Dedicated Render Window", ImVec2(-1, 30))) {
                            extern bool show_render_window;
                            show_render_window = true;
                        }
                        UIWidgets::EndSection();
                    }

                    // ─────────────────────────────────────────────────────────────────────────
                    // ANIMATION RENDER (Sequence Export)
                    // ─────────────────────────────────────────────────────────────────────────
                    if (UIWidgets::BeginSection("Animation Render", ImVec4(1.0f, 0.4f, 0.7f, 1.0f))) {
                        UIWidgets::ColoredHeader("Frame Range & Speed", ImVec4(0.8f, 0.8f, 0.8f, 1.0f));
                        ImGui::PushItemWidth(80);
                        ImGui::DragInt("Start", &ctx.render_settings.animation_start_frame, 1, 0, ctx.render_settings.animation_end_frame);
                        ImGui::SameLine();
                        ImGui::DragInt("End", &ctx.render_settings.animation_end_frame, 1, ctx.render_settings.animation_start_frame, 10000);
                        ImGui::SameLine();
                        ImGui::DragInt("FPS", &ctx.render_settings.animation_fps, 1, 1, 120);
                        ImGui::PopItemWidth();
                        
                        UIWidgets::Divider();
                        UIWidgets::ColoredHeader("Quality", ImVec4(0.8f, 0.8f, 0.8f, 1.0f));
                        ImGui::DragInt("Samples Per Frame", &ctx.render_settings.animation_samples_per_frame, 1, 1, 10000);
                        
                        UIWidgets::Divider();
                        UIWidgets::ColoredHeader("Output Style (PNG Sequence)", ImVec4(0.8f, 0.8f, 0.8f, 1.0f));
                        
                        // Output Path Display & Browse
                        ImGui::PushItemWidth(-50); // Leave room for browse button
                        char folder_buf[512];
                        strncpy(folder_buf, ctx.render_settings.animation_output_folder.c_str(), 511);
                        if (ImGui::InputText("##outdir", folder_buf, 512)) {
                            ctx.render_settings.animation_output_folder = folder_buf;
                        }
                        ImGui::PopItemWidth();
                        ImGui::SameLine();
                        if (ImGui::Button("...##browse")) {
                            std::string path = selectFolderDialogW(L"Select Animation Output Folder");
                            if (!path.empty()) ctx.render_settings.animation_output_folder = path;
                        }
                        
                        ImGui::Spacing();
                        bool can_render = !ctx.render_settings.animation_output_folder.empty();
                        if (UIWidgets::PrimaryButton("RENDER ANIMATION SEQUENCE", ImVec2(-1, 36), can_render)) {
                            ctx.render_settings.start_animation_render = true;
                            SCENE_LOG_INFO("Animation render sequence triggered...");
                        }
                        
                        if (!can_render) {
                            ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1.0f), "Set output folder to enable rendering.");
                        } else {
                            int total_frames = ctx.render_settings.animation_end_frame - ctx.render_settings.animation_start_frame + 1;
                            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Total: %d frames", total_frames);
                        }

                        UIWidgets::EndSection();
                    }
                }
                break;
            case 2: if (show_terrain_tab) drawTerrainPanel(ctx); break;
            case 3: if (show_water_tab) { drawWaterPanel(ctx); ImGui::Separator(); drawRiverPanel(ctx); } break;
            case 4: if (show_volumetric_tab) drawVolumetricPanel(ctx); break;
            case 5: if (show_forcefield_tab) ForceFieldUI::drawForceFieldPanel(ctx, ctx.scene); break;
            case 6: if (show_world_tab) drawWorldContent(ctx); break;
            case 7: drawThemeSelector(); drawResolutionPanel(ctx); break;
        }

        // Safety: Disable brushes if tab changed
        if (active_properties_tab != 2) terrain_brush.enabled = false;
        
        ImGui::PopStyleVar(2);  // WindowPadding, ItemSpacing
        ImGui::EndChild();      // End PropScrollArea
        ImGui::EndChild();      // End PropContentArea
        ImGui::PopStyleColor(); // ChildBg
        ImGui::PopStyleVar();   // ChildRounding
    }
    ImGui::End();
    ImGui::PopStyleColor(); // Border
    ImGui::PopStyleVar();   // BorderSize
}

// Main Menu Bar implementation moved to separate file: scene_ui_menu.hpp check end of file

#include "scene_ui_menu.hpp"


void SceneUI::draw(UIContext& ctx)
{
    // Export Popup Logic
    if (SceneExporter::getInstance().drawExportPopup(ctx.scene)) {
         std::wstring filter = SceneExporter::getInstance().settings.binary_mode ? L"GLTF Binary (.glb)\0*.glb\0" : L"GLTF Text (.gltf)\0*.gltf\0";
         std::wstring defExt = SceneExporter::getInstance().settings.binary_mode ? L"glb" : L"gltf";
         
         std::string filepath = saveFileDialogW(filter.c_str(), defExt.c_str());
         
         if (!filepath.empty()) {
             // Enforce extension
             std::string ext = SceneExporter::getInstance().settings.binary_mode ? ".glb" : ".gltf";
             if (!std::string(filepath).ends_with(ext)) {
                 filepath += ext;
             }

             rendering_stopped_cpu = true;
             rendering_stopped_gpu = true;
             
             // Show "Exporting..." modal or message? 
             addViewportMessage("Exporting Scene... Check Console...", 10.0f, ImVec4(1, 1, 0, 1));
             SCENE_LOG_INFO("[Export] Thread starting for: " + filepath);
             
             // Capture SceneData via pointer
             SceneData* pScene = &ctx.scene;
             
             // Capture Selection (Convert SelectableItem to shared_ptr<Hittable>)
             std::vector<std::shared_ptr<Hittable>> selected_hittables;
             if (SceneExporter::getInstance().settings.export_selected_only) {
                 for (const auto& item : ctx.selection.multi_selection) {
                     if (item.type == SelectableType::Object && item.object) {
                         selected_hittables.push_back(item.object);
                     }
                 }
             }

             std::thread export_thread([filepath, pScene, selected_hittables]() {
                 SCENE_LOG_INFO("[Export] Thread running...");
                 ExportSettings settings = SceneExporter::getInstance().settings;
                 
                 try {
                     bool success = SceneExporter::getInstance().exportScene(filepath, *pScene, settings, selected_hittables);
                     if (success) {
                         SCENE_LOG_INFO("[Export] SUCCESS: " + filepath);
                     } else {
                         SCENE_LOG_ERROR("[Export] FAILED (Check logs)");
                     }
                 } catch (const std::exception& e) {
                     SCENE_LOG_ERROR("[Export] EXCEPTION: " + std::string(e.what()));
                 } catch (...) {
                     SCENE_LOG_ERROR("[Export] UNKNOWN EXCEPTION");
                 }

                 rendering_stopped_cpu = false;
                 rendering_stopped_gpu = false;
             });
             export_thread.detach();
         }
    }

    world_params_changed_this_frame = false; // Reset flag
    ImGuiIO& io = ImGui::GetIO();
    float screen_x = io.DisplaySize.x;
    float screen_y = io.DisplaySize.y;

    drawMainMenuBar(ctx);
    handleEditorShortcuts(ctx);

    float left_offset = 0.0f;
    drawPanels(ctx);
    left_offset = showSidePanel ? side_panel_width : 0.0f;

    drawStatusAndBottom(ctx, screen_x, screen_y, left_offset);

    bool gizmo_hit = drawOverlays(ctx);
    
    // --- ANIMATION UPDATE ---
    processAnimations(ctx);

    drawSelectionGizmos(ctx);
    drawCameraGizmos(ctx);  // Draw camera frustum icons
    drawRiverGizmos(ctx, gizmo_hit);  // Draw river spline control points
    drawViewportControls(ctx);  // Blender-style viewport overlay
    
    // --- BACKGROUND SAVE STATUS POLL ---
    static int last_save_state = 0;
    int save_state = bg_save_state.load();
    
    if (save_state != last_save_state) {
        if (save_state == 1) { // Saving...
            addViewportMessage("Saving...", 300.0f, ImVec4(1.0f, 0.9f, 0.2f, 1.0f));
        }
        else if (save_state == 2) { // Done
            clearViewportMessages();
            addViewportMessage("Project saved", 2.0f, ImVec4(0.2f, 1.0f, 0.4f, 1.0f));
            bg_save_state = 0; // Reset
        }
        else if (save_state == 3) { // Error
            clearViewportMessages();
            addViewportMessage("Save failed", 4.0f, ImVec4(1.0f, 0.2f, 0.2f, 1.0f));
            bg_save_state = 0; // Reset
        }
        last_save_state = save_state;
    }

    drawViewportMessages(ctx, left_offset); // Messages/HUD (e.g. Async Rebuild)

    // Hide HUD overlays if exit confirmation is open
    // Otherwise draw them (they use ForegroundDrawList so they appear on top)
    // Note: They might overlay panels like Graph, but visibility is priority.
    if (!show_exit_confirmation) {
        drawFocusIndicator(ctx);
        drawZoomRing(ctx);
        drawExposureInfo(ctx);  // Now includes lens info below the triangle
    }
    
    // Scatter Brush System
    handleScatterBrush(ctx);   // Handle brush painting input
    drawBrushPreview(ctx);     // Draw brush circle preview
    
    // Terrain Sculpting
    handleTerrainBrush(ctx);
    handleTerrainFoliageBrush(ctx);  // Foliage painting brush

    handleSceneInteraction(ctx, gizmo_hit);
    processDeferredSceneUpdates(ctx);
    
    // Update Water Animation
    if (WaterManager::getInstance().update(io.DeltaTime)) {
         if (ctx.optix_gpu_ptr) {
             ctx.renderer.updateOptiXMaterialsOnly(ctx.scene, ctx.optix_gpu_ptr);
             ctx.optix_gpu_ptr->resetAccumulation();
         }
         // Also reset CPU accumulation if needed, though water FFT is mostly for GPU?
         // If CPU supports it (sampleOceanHeight), maybe reset CPU too.
         ctx.renderer.resetCPUAccumulation();
    }

    drawAuxWindows(ctx);
    
    // Global Sun Sync (Light -> Nishita)
    processSunSync(ctx);
    drawRenderWindow(ctx);
    drawExitConfirmation(ctx);
    
    // NOTE: Animation Graph Editor is now in the bottom panel (like Terrain Graph)
    // The floating window version (drawAnimationEditorPanel) is kept for optional use via View menu
        }
    
void SceneUI::handleEditorShortcuts(UIContext& ctx)
{
    ImGuiIO& io = ImGui::GetIO();

    if (!io.WantCaptureKeyboard && ctx.selection.hasSelection()) {
        handleDeleteShortcut(ctx);
    }

    // F12 Render
    if (ImGui::IsKeyPressed(ImGuiKey_F12)) {
        extern bool show_render_window;
        show_render_window = !show_render_window;
        if (show_render_window) ctx.start_render = true;
    }

    // N key - Toggle Properties Panel (Blender-style sidebar toggle)
    // Use WantTextInput instead of WantCaptureKeyboard so it works even when panel has focus
    // Only block when actively typing in a text field
    if (!io.WantTextInput && ImGui::IsKeyPressed(ImGuiKey_N) && !io.KeyCtrl && !io.KeyShift && !io.KeyAlt) {
        showSidePanel = !showSidePanel;
        SCENE_LOG_INFO(showSidePanel ? "Properties panel shown (N)" : "Properties panel hidden (N)");
    }

    // Undo / Redo
    if (ImGui::IsKeyPressed(ImGuiKey_Z) && io.KeyCtrl && !io.KeyShift) {
        if (history.canUndo()) {
            history.undo(ctx);
            rebuildMeshCache(ctx.scene.world.objects);
            ctx.selection.updatePositionFromSelection();
            ctx.selection.selected.has_cached_aabb = false;
        }
    }

    if ((ImGui::IsKeyPressed(ImGuiKey_Y) && io.KeyCtrl) ||
        (ImGui::IsKeyPressed(ImGuiKey_Z) && io.KeyCtrl && io.KeyShift)) {
        if (history.canRedo()) {
            history.redo(ctx);
            rebuildMeshCache(ctx.scene.world.objects);
            ctx.selection.updatePositionFromSelection();
            ctx.selection.selected.has_cached_aabb = false;
        }
    }
}
void SceneUI::drawPanels(UIContext& ctx)
{
    ImGuiIO& io = ImGui::GetIO();
    float screen_y = io.DisplaySize.y;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 4.0f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg,
        ImVec4(0.1f, 0.1f, 0.13f, panel_alpha));

    if (showSidePanel) {
        drawRenderSettingsPanel(ctx, screen_y);
    }

    ImGui::PopStyleColor();
    ImGui::PopStyleVar();
}
void SceneUI::drawStatusAndBottom(UIContext& ctx,
    float screen_x,
    float screen_y,
    float left_offset)
{
    // ---------------- STATUS BAR ----------------
    float status_bar_height = 24.0f;

    // EXPORT PROGRESS HUD
    if (SceneExporter::getInstance().is_exporting) {
        ImGui::OpenPopup("Exporting...");
    }

    if (ImGui::BeginPopupModal("Exporting...", NULL, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar)) {
        if (!SceneExporter::getInstance().is_exporting) {
            ImGui::CloseCurrentPopup();
        } else {
            ImGui::Text("Exporting Scene...");
            ImGui::Separator();
            
            // Spinner or Bar
            ImGui::Text("Please wait, data is being processed.");
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "%s", SceneExporter::getInstance().current_export_status.c_str());
            
            ImGui::Separator();
        }
        ImGui::EndPopup();
    }

    ImGui::SetNextWindowPos(ImVec2(0, screen_y - status_bar_height), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(screen_x, status_bar_height), ImGuiCond_Always);

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(4, 2));
    // Use theme color with alpha
    ImGui::SetNextWindowBgAlpha(panel_alpha);

    if (ImGui::Begin("StatusBar", nullptr,
        ImGuiWindowFlags_NoDecoration |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoBringToFrontOnFocus))
    {
        ImGui::SetCursorPosX(8); // Small left margin
        
        if (UIWidgets::HorizontalTab("Timeline", UIWidgets::IconType::Timeline, show_animation_panel))
        {
            show_animation_panel = !show_animation_panel;
            if (show_animation_panel) show_scene_log = false;
        }

        if (UIWidgets::HorizontalTab("Console", UIWidgets::IconType::Console, show_scene_log))
        {
            show_scene_log = !show_scene_log;
            if (show_scene_log) { 
                show_animation_panel = false; 
                show_terrain_graph = false; 
            }
        }
        
        if (UIWidgets::HorizontalTab("Graph", UIWidgets::IconType::Graph, show_terrain_graph))
        {
            show_terrain_graph = !show_terrain_graph;
            if (show_terrain_graph) {
                show_scene_log = false;
                show_animation_panel = false;
                show_anim_graph = false;
            }
        }
        
        if (UIWidgets::HorizontalTab("AnimGraph", UIWidgets::IconType::AnimGraph, show_anim_graph))
        {
            show_anim_graph = !show_anim_graph;
            if (show_anim_graph) {
                show_scene_log = false;
                show_animation_panel = false;
                show_terrain_graph = false;
            }
        }

        ImGui::SameLine();
        

        ImGui::SameLine();
        ImGui::TextDisabled("|");
        ImGui::SameLine();

        if (ctx.scene.initialized) {
            auto drawPill = [&](const char* label, const ImVec4& color, bool highlight = false) {
                ImVec2 p = ImGui::GetCursorScreenPos();
                ImVec2 textSize = ImGui::CalcTextSize(label);
                ImVec2 pillSize = ImVec2(textSize.x + 16, 18);
                ImDrawList* dlist = ImGui::GetWindowDrawList();
                
                // Pill Background
                dlist->AddRectFilled(p, ImVec2(p.x + pillSize.x, p.y + pillSize.y), highlight ? IM_COL32(26, 230, 204, 40) : IM_COL32(255, 255, 255, 15), 9.0f);
                if (highlight) dlist->AddRect(p, ImVec2(p.x + pillSize.x, p.y + pillSize.y), IM_COL32(26, 230, 204, 80), 9.0f);

                ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 8);
                ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 2);
                ImGui::TextColored(highlight ? ImVec4(0.1f, 0.9f, 0.8f, 1.0f) : ImVec4(0.8f, 0.8f, 0.85f, 1.0f), "%s", label);
                
                ImGui::SetCursorScreenPos(ImVec2(p.x + pillSize.x + 6, p.y));
            };

            // Scene Info Pills
            int obj_count = (int)mesh_cache.size();
            std::string obj_str = "Scene: " + std::to_string(obj_count) + " Objects";
            drawPill(obj_str.c_str(), ImVec4(0.8f, 0.8f, 0.8f, 1.0f));
            
            std::string light_str = std::to_string(ctx.scene.lights.size()) + " Lights";
            drawPill(light_str.c_str(), ImVec4(0.8f, 0.8f, 0.8f, 1.0f));
            
            // Selection Pill
            if (ctx.selection.hasSelection()) {
                std::string sel_str = "Selected: " + ctx.selection.selected.name;
                drawPill(sel_str.c_str(), ImVec4(0.4f, 0.8f, 1.0f, 1.0f), true);
            }
        }
        else {
            ImGui::Text("Ready");
        }

        if (rendering_in_progress) {
            float p = ctx.render_settings.render_progress * 100.0f;
            std::string prog = "Rendering: " + std::to_string((int)p) + "%";
            float w = ImGui::CalcTextSize(prog.c_str()).x;
            ImGui::SameLine(screen_x - w - 20);
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "%s", prog.c_str());
        }
    }
    ImGui::End();

    // ImGui::End(); // Removed redundant End
    
    // ImGui::PopStyleColor(); // Removed hardcoded color push
    ImGui::PopStyleVar(2);

    // ---------------- BOTTOM PANEL (Resizable) ----------------
    bool show_bottom = (show_animation_panel || show_scene_log || show_terrain_graph || show_anim_graph);
    if (!show_bottom) return;

    // Use class member for persistent height
    // static float bottom_height = 280.0f; <-- REMOVED
    const float min_height = 100.0f;
    const float max_height = screen_y * 0.6f;  // Max 60% of screen
    const float resize_handle_height = 6.0f;
    
    // Clamp height to valid range
    bottom_panel_height = std::clamp(bottom_panel_height, min_height, max_height);

    // Calculate panel position
    float panel_top = screen_y - bottom_panel_height - status_bar_height;
    
    // --- RESIZE HANDLE (invisible button at top edge) ---
    ImGui::SetNextWindowPos(ImVec2(0, panel_top - resize_handle_height / 2), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(screen_x, resize_handle_height), ImGuiCond_Always);
    
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0));  // Transparent
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    
    if (ImGui::Begin("##BottomPanelResizer", nullptr,
        ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoScrollbar |
        ImGuiWindowFlags_NoSavedSettings))
    {
        // Draw a subtle resize indicator line
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        ImVec2 p1(0, panel_top);
        ImVec2 p2(screen_x, panel_top);
        draw_list->AddLine(p1, p2, IM_COL32(100, 100, 100, 180), 2.0f);
        
        // Handle resize dragging
        ImGui::InvisibleButton("##ResizeHandle", ImVec2(screen_x, resize_handle_height));
        if (ImGui::IsItemHovered()) {
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
        }
        if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
            bottom_panel_height -= ImGui::GetIO().MouseDelta.y;
            bottom_panel_height = std::clamp(bottom_panel_height, min_height, max_height);
        }
    }
    ImGui::End();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();

    // --- MAIN BOTTOM PANEL ---
    ImGui::SetNextWindowPos(ImVec2(0, panel_top), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(screen_x, bottom_panel_height), ImGuiCond_Always);

    // Use theme color with alpha
    ImGui::SetNextWindowBgAlpha(panel_alpha);
    // ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.12f, 0.12f, 0.15f, 1.0f)); // Removed hardcoded

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0)); // No padding for thin look
    if (ImGui::Begin("BottomPanel", nullptr,
        ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoCollapse))
    {
        if (show_animation_panel) {
            drawTimelineContent(ctx);
        }
        else if (show_scene_log) {
            drawLogPanelEmbedded();
        }
        else if (show_terrain_graph) {
            // Terrain Node Graph Editor
            // Disable edit tool when using node graph (performance optimization)
            terrain_brush.enabled = false;
            
            TerrainObject* activeTerrain = nullptr;
            if (terrain_brush.active_terrain_id != -1) {
                activeTerrain = TerrainManager::getInstance().getTerrain(terrain_brush.active_terrain_id);
            }
            // Auto-create default graph if empty
            if (activeTerrain && terrainNodeGraph.nodes.empty()) {
                terrainNodeGraph.createDefaultGraph(activeTerrain);
            }
            
            // Set callback if not set to enable file dialogs
            if (!terrainNodeEditorUI.onOpenFileDialog) {
                 terrainNodeEditorUI.onOpenFileDialog = [](const wchar_t* filter) -> std::string {
                     return SceneUI::openFileDialogW(filter);
                 };
            }
            if (!terrainNodeEditorUI.onSaveFileDialog) {
                 terrainNodeEditorUI.onSaveFileDialog = [](const wchar_t* filter, const wchar_t* defName) -> std::string {
                     // defName string'i saveFileDialogW'ye defExt olarak geçiliyor, 
                     // ancak SceneUI::saveFileDialogW aslında uzantı bekliyor olabilir mi?
                     // Bakalım: SceneUI::saveFileDialogW, lpstrDefExt kullanıyor.
                     // Eğer defName "splat.png" gelirse, lpstrDefExt sadece uzantı ister genelde.
                     // Ama varsayılan dosya adı (lpstrFile) da ayarlanabilir.
                     // Mevcut SceneUI::saveFileDialogW implementasyonu lpstrFile'ı temizliyor.
                     // Orayı da güncellemek gerekebilir ama şimdilik exportPath'i UI tarafında set ediyoruz.
                     
                     // Not: SceneUI::saveFileDialogW(filter, defExt) imzası var.
                     // Bizim lambda (filter, defName) alıyor.
                     // SplatOutputNode'da "splat_map.png" gönderiyoruz.
                     // Bu durumda defExt olarak "png" göndermek daha doğru olur.
                     
                     return SceneUI::saveFileDialogW(filter, L"png");
                 };
            }
            terrainNodeEditorUI.draw(ctx, terrainNodeGraph, activeTerrain);
        }
        else if (show_anim_graph) {
            // Animation Node Graph Editor
            drawAnimationGraphPanel(ctx);
        }
    }
    ImGui::End();
    ImGui::PopStyleVar(); // Pop WindowPadding
    // ImGui::PopStyleVar(); // Removed redundant PopStyleVar
    // ImGui::PopStyleColor(); // Removed hardcoded color push
}
bool SceneUI::drawOverlays(UIContext& ctx)
{
    // === UI SETTINGS SERIALIZATION ===
    // 1. LOAD from SceneData if a new project was loaded
    static int last_local_counter = -1;
    if (ctx.scene.load_counter != last_local_counter) {
        if (!ctx.scene.ui_settings_json_str.empty()) {
            try {
                auto j = nlohmann::json::parse(ctx.scene.ui_settings_json_str);
                
                // Pro Camera Settings
                if (j.contains("viewport_settings")) {
                    auto& vs = j["viewport_settings"];
                    viewport_settings.show_histogram = vs.value("show_histogram", false);
                    viewport_settings.histogram_mode = vs.value("histogram_mode", 0);
                    viewport_settings.histogram_opacity = vs.value("histogram_opacity", 0.5f);
                    viewport_settings.show_focus_peaking = vs.value("show_focus_peaking", false);
                    viewport_settings.focus_peaking_color = vs.value("focus_peaking_color", 0);
                    viewport_settings.focus_peaking_threshold = vs.value("focus_peaking_threshold", 0.15f);
                    viewport_settings.show_zebra = vs.value("show_zebra", false);
                    viewport_settings.zebra_threshold = vs.value("zebra_threshold", 0.95f);
                    viewport_settings.show_af_points = vs.value("show_af_points", false);
                    viewport_settings.af_mode = vs.value("af_mode", 0);
                    viewport_settings.af_selected_point = vs.value("af_selected_point", 4);
                    viewport_settings.focus_mode = vs.value("focus_mode", 1);
                }
            } catch (...) {}
        }
        last_local_counter = ctx.scene.load_counter;
    }

    // 2. SAVE to SceneData if settings changed
    static ViewportDisplaySettings last_vs_check;
    static bool first_run = true;
    if (first_run) {
        std::memset(&last_vs_check, 0, sizeof(last_vs_check));
        first_run = false;
    }
    
    if (std::memcmp(&viewport_settings, &last_vs_check, sizeof(ViewportDisplaySettings)) != 0) {
        try {
            nlohmann::json root;
            if (!ctx.scene.ui_settings_json_str.empty()) {
                try { root = nlohmann::json::parse(ctx.scene.ui_settings_json_str); } catch(...) {}
            }
            
            nlohmann::json& vs = root["viewport_settings"];
            vs["show_histogram"] = viewport_settings.show_histogram;
            vs["histogram_mode"] = viewport_settings.histogram_mode;
            vs["histogram_opacity"] = viewport_settings.histogram_opacity;
            vs["show_focus_peaking"] = viewport_settings.show_focus_peaking;
            vs["focus_peaking_color"] = viewport_settings.focus_peaking_color;
            vs["focus_peaking_threshold"] = viewport_settings.focus_peaking_threshold;
            vs["show_zebra"] = viewport_settings.show_zebra;
            vs["zebra_threshold"] = viewport_settings.zebra_threshold;
            vs["show_af_points"] = viewport_settings.show_af_points;
            vs["af_mode"] = viewport_settings.af_mode;
            vs["af_selected_point"] = viewport_settings.af_selected_point;
            vs["focus_mode"] = viewport_settings.focus_mode;
            
            ctx.scene.ui_settings_json_str = root.dump();
            std::memcpy(&last_vs_check, &viewport_settings, sizeof(ViewportDisplaySettings));
        } catch (...) {}
    }

    bool gizmo_hit = false;

    // Draw Viewport HUDs
    // Render status is now integrated into drawViewportMessages

    if (ctx.scene.camera && ctx.selection.show_gizmo) {
        drawLightGizmos(ctx, gizmo_hit);
        drawForceFieldGizmos(ctx, gizmo_hit);
    }

    // === PRO CAMERA HUD OVERLAYS ===
    // These are drawn on top of everything else
    drawHistogramOverlay(ctx);
    drawFocusPeakingOverlay(ctx);
    drawZebraOverlay(ctx);
    drawAFPointsOverlay(ctx);

    // === VIEWPORT EDGE FRAME ===
    // Draw a subtle border on the right edge of the viewport to clearly delineate the area
    {
        ImGuiIO& io = ImGui::GetIO();
        ImDrawList* draw_list = ImGui::GetForegroundDrawList();
        
        // Right edge border (1px dark line)
        float border_x = io.DisplaySize.x - 1.0f;
        ImU32 border_color = IM_COL32(40, 40, 50, 200);  // Dark subtle border
        draw_list->AddLine(
            ImVec2(border_x, 0), 
            ImVec2(border_x, io.DisplaySize.y), 
            border_color, 
            1.0f
        );
        
        // Optional: Add a subtle highlight line just inside
        ImU32 highlight_color = IM_COL32(60, 60, 70, 100);  // Very subtle highlight
        draw_list->AddLine(
            ImVec2(border_x - 1.0f, 0), 
            ImVec2(border_x - 1.0f, io.DisplaySize.y), 
            highlight_color, 
            1.0f
        );
    }

    return gizmo_hit;
}



void SceneUI::handleSceneInteraction(UIContext& ctx, bool gizmo_hit)
{
    if (ctx.scene.initialized && !gizmo_hit) {
        handleMouseSelection(ctx);
        handleMarqueeSelection(ctx);
    }
}
void SceneUI::processDeferredSceneUpdates(UIContext& ctx)
{
    if (is_bvh_dirty && !ImGuizmo::IsUsing()) {
        ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
        ctx.renderer.resetCPUAccumulation();
        
        // OPTIMIZATION: Only update OptiX geometry when OptiX rendering is enabled
        if (ctx.render_settings.use_optix && ctx.optix_gpu_ptr) {
            ctx.optix_gpu_ptr->updateGeometry(ctx.scene.world.objects);
            ctx.optix_gpu_ptr->resetAccumulation();
        }
        is_bvh_dirty = false;
    }
}
void SceneUI::drawAuxWindows(UIContext& ctx)
{
    if (show_controls_window) {
        ImGui::SetNextWindowSize(ImVec2(500, 600), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Controls & Help", &show_controls_window)) {
            drawControlsContent();
        }
        ImGui::End();
    }
}

void SceneUI::drawControlsContent()
{
     UIWidgets::ColoredHeader("Camera Controls", ImVec4(1.0f, 0.9f, 0.4f, 1.0f));
     UIWidgets::Divider();
     
     ImGui::BulletText("Rotate: Middle Mouse Drag");
     ImGui::BulletText("Pan: Shift + Middle Mouse Drag");
     ImGui::BulletText("Zoom: Mouse Wheel OR Ctrl + Middle Mouse Drag");
     ImGui::BulletText("Move Forward/Back: Arrow Up / Arrow Down");
     ImGui::BulletText("Move Left/Right: Arrow Left / Arrow Right");
     ImGui::BulletText("Move Up/Down: PageUp / PageDown");
     
     ImGui::Spacing();
     UIWidgets::ColoredHeader("Shortcuts", ImVec4(0.6f, 0.8f, 1.0f, 1.0f));
     UIWidgets::Divider();
     ImGui::BulletText("Toggle Properties Panel: N");
     ImGui::BulletText("Toggle Render Window: F12");
     ImGui::BulletText("Toggle Help Window: F1");
     ImGui::BulletText("Save Image: Ctrl + S"); 
     ImGui::BulletText("Undo: Ctrl + Z");
     ImGui::BulletText("Redo: Ctrl + Y or Ctrl + Shift + Z");
     ImGui::BulletText("Delete Object: Delete or X");
     ImGui::BulletText("Duplicate Object: Shift + D");
     
     ImGui::Spacing();
     UIWidgets::ColoredHeader("Gizmo Controls", ImVec4(1.0f, 0.6f, 0.4f, 1.0f));
     UIWidgets::Divider();
     ImGui::BulletText("Move Mode: G (Translate)");
     ImGui::BulletText("Rotate Mode: R");
     ImGui::BulletText("Scale Mode: S");
     ImGui::BulletText("Switch Mode: W (Cycle through modes)");
     ImGui::TextDisabled("  * Click and drag gizmo handles to transform");
     ImGui::TextDisabled("  * Changes apply immediately to selected object");
     
     ImGui::Spacing();
     UIWidgets::ColoredHeader("Selection Controls", ImVec4(0.8f, 0.4f, 1.0f, 1.0f));
     UIWidgets::Divider();
     ImGui::BulletText("Select Object: Left Click");
     ImGui::BulletText("Multi-Select: Shift + Left Click");
     ImGui::BulletText("Box Selection: Right Mouse Drag");
     ImGui::BulletText("Duplicate Selection: Shift + D");

     ImGui::Spacing();
     UIWidgets::ColoredHeader("Interface Guide", ImVec4(0.4f, 1.0f, 0.6f, 1.0f));
     UIWidgets::Divider();
     
     if (ImGui::CollapsingHeader("Render Settings")) {
         ImGui::BulletText("Quality Preset: Quick setup for Preview (Fast) vs Cinematic (High Quality).");
         ImGui::BulletText("Use OptiX: Enable NVIDIA GPU acceleration (Requires RTX card).");
         ImGui::BulletText("Use Denoiser: Clean up noise using AI (OIDN).");
         ImGui::BulletText("Start/Stop: Control the rendering process.");
         ImGui::TextDisabled("  * Pausing allows you to resume later.");
         ImGui::TextDisabled("  * Stopping resets the progress.");
     }
     
     if (ImGui::CollapsingHeader("Sampling")) {
         ImGui::BulletText("Adaptive Sampling: Focuses rays on noisy areas for efficiency.");
         ImGui::BulletText("Max Samples: The target quality level (higher = less noise).");
         ImGui::BulletText("Max Bounces: Light reflection depth (higher = more realistic light).");
     }
     
     if (ImGui::CollapsingHeader("Timeline & Animation")) {
         ImGui::BulletText("Play/Pause: Preview animation movement.");
         ImGui::BulletText("Scrubbing: Drag the timeline handle to jump to frames.");
         ImGui::BulletText("Render Animation: Renders the full sequence to the output folder.");
     }
     
     if (ImGui::CollapsingHeader("Physical Camera System")) {
         ImGui::TextColored(ImVec4(0.5f, 0.9f, 0.7f, 1.0f), "Camera Body Selection:");
         ImGui::BulletText("Select real camera bodies (Canon, Sony, Pentax, RED, ARRI...)");
         ImGui::BulletText("Each body has a specific sensor size and crop factor.");
         ImGui::BulletText("APS-C sensors have 1.5x-1.6x crop, narrowing FOV.");
         ImGui::BulletText("Medium Format has 0.79x crop, widening FOV.");
         ImGui::BulletText("Custom mode: Manual crop factor slider (0.5x - 2.5x).");
         
         ImGui::Separator();
         ImGui::TextColored(ImVec4(0.9f, 0.8f, 0.5f, 1.0f), "Lens Selection:");
         ImGui::BulletText("Professional lenses from Canon, Sony, Sigma, Zeiss, Pentax...");
         ImGui::BulletText("Each lens provides: Focal length, Max aperture, Blade count.");
         ImGui::BulletText("Blade count affects bokeh shape (more = rounder).");
         ImGui::BulletText("Effective focal = Base focal x Crop factor.");
         
         ImGui::Separator();
         ImGui::TextColored(ImVec4(0.8f, 0.6f, 0.9f, 1.0f), "Depth of Field:");
         ImGui::BulletText("F-Stop presets: f/1.4 (blur) to f/16 (sharp).");
         ImGui::BulletText("Focus Distance: Distance to the sharpest plane.");
         ImGui::BulletText("Focus to Selection: Auto-set focus to selected object.");
         
         ImGui::Separator();
         ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1.0f), "Keyframing:");
         ImGui::BulletText("Click [K] buttons to insert keyframes for properties.");
         ImGui::BulletText("Key All: Insert keyframe for all camera properties.");
     }
     
     if (ImGui::CollapsingHeader("Lights Panel")) {
         ImGui::BulletText("Lists all light sources in the scene.");
         ImGui::BulletText("Allows adjusting Intensity (Brightness) and Color.");
         ImGui::BulletText("Supports Point, Directional, Area, and Spot lights.");
     }
     
     if (ImGui::CollapsingHeader("World Panel")) {
         ImGui::BulletText("Sky Model: Switch between Solid Color, HDRI, or Raytrophi Spectral Sky.");
         ImGui::BulletText("Raytrophi Spectral Sky: Realistic day/night cycle with Sun & Moon controls.");
         ImGui::BulletText("Atmosphere: Adjust Air, Dust, and Ozone density for realistic scattering.");
         ImGui::BulletText("Volumetric Clouds: Enable 3D clouds with various weather presets.");
         ImGui::BulletText("HDRI: Load external HDR enviromaps for realistic reflections.");
         ImGui::BulletText("Light Sync: Sync Sun position with the main directional light.");
     }
     
     if (ImGui::CollapsingHeader("Post-FX Panel")) {
         ImGui::BulletText("Main: Adjust Gamma, Exposure, Saturation, and Color Temperature.");
         ImGui::BulletText("Tonemapping: Choose from AGX, ACES, Filmic, etc.");
         ImGui::BulletText("Effects: Add Vignette (dark corners) to frame the image.");
         ImGui::BulletText("Apply/Reset: Post-processing is applied AFTER rendering.");
     }

     if (ImGui::CollapsingHeader("System Panel")) {
         ImGui::BulletText("Theme: Switch between Dark/Light/Classic themes.");
         ImGui::BulletText("Resolution: Set render resolution (Presets like 720p, 1080p, 4K).");
         ImGui::BulletText("Animation Panel: Toggle visibility of the timeline.");
     }
}


void SceneUI::rebuildMeshCache(const std::vector<std::shared_ptr<Hittable>>& objects) {
    mesh_cache.clear();
    mesh_ui_cache.clear();
    bbox_cache.clear();  // Clear bounding box cache too
    material_slots_cache.clear();  // Clear material slots cache
    
    for (size_t i = 0; i < objects.size(); ++i) {
        auto tri = std::dynamic_pointer_cast<Triangle>(objects[i]);
        if (tri) {
            std::string name = tri->nodeName.empty() ? "Unnamed" : tri->nodeName;
            mesh_cache[name].push_back({(int)i, tri});
        }
    }
    
    // Transfer to sequential vector for ImGui Clipper AND calculate bounding boxes AND material slots
    mesh_ui_cache.reserve(mesh_cache.size());
    for (auto& kv : mesh_cache) {
        mesh_ui_cache.push_back(kv);
        
        // Calculate LOCAL bounding box from ORIGINAL vertices (not transformed!)
        // This allows us to properly apply the transform matrix when drawing
        Vec3 bb_min(1e10f, 1e10f, 1e10f);
        Vec3 bb_max(-1e10f, -1e10f, -1e10f);
        
        // Collect unique material IDs for this object
        std::vector<uint16_t> mat_ids;
        
        for (auto& pair : kv.second) {
            auto& tri = pair.second;
            // Use ORIGINAL vertices (local space) - not getV0() which returns transformed!
            Vec3 v0 = tri->getOriginalVertexPosition(0);
            Vec3 v1 = tri->getOriginalVertexPosition(1);
            Vec3 v2 = tri->getOriginalVertexPosition(2);
            
            bb_min.x = fminf(bb_min.x, fminf(v0.x, fminf(v1.x, v2.x)));
            bb_min.y = fminf(bb_min.y, fminf(v0.y, fminf(v1.y, v2.y)));
            bb_min.z = fminf(bb_min.z, fminf(v0.z, fminf(v1.z, v2.z)));
            bb_max.x = fmaxf(bb_max.x, fmaxf(v0.x, fmaxf(v1.x, v2.x)));
            bb_max.y = fmaxf(bb_max.y, fmaxf(v0.y, fmaxf(v1.y, v2.y)));
            bb_max.z = fmaxf(bb_max.z, fmaxf(v0.z, fmaxf(v1.z, v2.z)));
            
            // Collect material ID (check for duplicates - usually few materials)
            uint16_t mid = tri->getMaterialID();
            bool found = false;
            for (uint16_t existing : mat_ids) {
                if (existing == mid) { found = true; break; }
            }
            if (!found) mat_ids.push_back(mid);
        }
        
        bbox_cache[kv.first] = {bb_min, bb_max};
        material_slots_cache[kv.first] = std::move(mat_ids);
    }
    
    mesh_cache_valid = true;
}

// Update bounding box for a specific object (after transform)
// NOTE: Since we now store LOCAL bbox (from original vertices), this may not need
// to be called unless the mesh geometry itself changes. Transform is applied at draw time.
void SceneUI::updateBBoxCache(const std::string& objectName) {
    auto it = mesh_cache.find(objectName);
    if (it == mesh_cache.end()) return;
    
    Vec3 bb_min(1e10f, 1e10f, 1e10f);
    Vec3 bb_max(-1e10f, -1e10f, -1e10f);
    
    for (auto& pair : it->second) {
        auto& tri = pair.second;
        // Use ORIGINAL vertices (local space) for consistency with rebuildMeshCache
        Vec3 v0 = tri->getOriginalVertexPosition(0);
        Vec3 v1 = tri->getOriginalVertexPosition(1);
        Vec3 v2 = tri->getOriginalVertexPosition(2);
        
        bb_min.x = fminf(bb_min.x, fminf(v0.x, fminf(v1.x, v2.x)));
        bb_min.y = fminf(bb_min.y, fminf(v0.y, fminf(v1.y, v2.y)));
        bb_min.z = fminf(bb_min.z, fminf(v0.z, fminf(v1.z, v2.z)));
        bb_max.x = fmaxf(bb_max.x, fmaxf(v0.x, fmaxf(v1.x, v2.x)));
        bb_max.y = fmaxf(bb_max.y, fmaxf(v0.y, fmaxf(v1.y, v2.y)));
        bb_max.z = fmaxf(bb_max.z, fmaxf(v0.z, fmaxf(v1.z, v2.z)));
    }
    
    bbox_cache[objectName] = {bb_min, bb_max};
}

// Lazy CPU Sync - called before mouse picking to ensure vertices are up to date
// This is much more efficient than updating on gizmo release because:
// 1. Gizmo release is instant (no freeze)
// 2. Sync only happens when user actually tries to pick something
// 3. If user moves object multiple times without picking, we only sync once
void SceneUI::ensureCPUSyncForPicking() {
    if (objects_needing_cpu_sync.empty()) return;
    
    // Update all pending objects
    for (const auto& name : objects_needing_cpu_sync) {
        auto it = mesh_cache.find(name);
        if (it != mesh_cache.end()) {
            for (auto& pair : it->second) {
                pair.second->updateTransformedVertices();
            }
        }
    }
    
    size_t count = objects_needing_cpu_sync.size();
    objects_needing_cpu_sync.clear();
    
    // Trigger BVH rebuild for accurate picking
    extern bool g_bvh_rebuild_pending;
    g_bvh_rebuild_pending = true;
    
    SCENE_LOG_INFO("Lazy CPU sync completed for " + std::to_string(count) + " objects");
}

// Global flag for Render Window visibility
bool show_render_window = false;

void SceneUI::drawRenderWindow(UIContext& ctx) {
    if (!show_render_window) return;

    ImGui::SetNextWindowSize(ImVec2(800, 600), ImGuiCond_FirstUseEver);
    
    // Auto-Stop Logic
    // Auto-Stop Logic (ONLY for Single Frame Render)
    // For animation, the render thread manages the loop and frame progression.
    int current_samples = ctx.renderer.getCPUAccumulatedSamples();
    int target_samples = ctx.render_settings.final_render_samples;

    if (!ctx.is_animation_mode && ctx.render_settings.is_final_render_mode && current_samples >= target_samples) {
        ctx.render_settings.is_final_render_mode = false; // Finish
        extern std::atomic<bool> rendering_stopped_cpu;
        rendering_stopped_cpu = true; 
    }

    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.08f, 0.08f, 0.08f, 1.0f)); // Opaque Background
    
    // Enable Resize and Collapse/Maximize (removed NoCollapse)
    ImGuiWindowFlags win_flags = ImGuiWindowFlags_None; 
    
    if (ImGui::Begin("Render Result", &show_render_window, win_flags)) {
        
        static bool show_sidebar = true;

        // Progress Info
        float progress = (float)current_samples / (float)target_samples;
        if (progress > 1.0f) progress = 1.0f;

        // Header and Toolbar
        // ---------------------------------------------------------
        if (ctx.is_animation_mode) {            
             ImGui::Text("Animation:");
             ImGui::SameLine();
             
             if (rendering_in_progress) ImGui::TextColored(ImVec4(1, 1, 0, 1), "[RENDERING...]");
             else ImGui::TextColored(ImVec4(0, 1, 0, 1), "[FINISHED / STOPPED]");
             
             ImGui::SameLine();
             int cur_frame = ctx.render_settings.animation_current_frame;
             int end_frame = ctx.render_settings.animation_end_frame;
             int start_frame = ctx.render_settings.animation_start_frame;
             int total = end_frame - start_frame + 1;
             int current_idx = cur_frame - start_frame + 1;
             if (current_idx < 0) current_idx = 0;
             
             float progress = (total > 0) ? (float)current_idx / (float)total : 0.0f;
             char buf[64];
             sprintf(buf, "Frame: %d / %d", cur_frame, end_frame);
             ImGui::ProgressBar(progress, ImVec2(200, 0), buf);
        }
        else {
            ImGui::Text("Status:");
            ImGui::SameLine();
            if (current_samples >= target_samples) {
                ImGui::TextColored(ImVec4(0,1,0,1), "[FINISHED]");
            } else if (ctx.render_settings.is_final_render_mode) {
                 ImGui::TextColored(ImVec4(1,1,0,1), "[RENDERING...]");
            } else {
                 ImGui::TextColored(ImVec4(0.7f,0.7f,0.7f,1), "[IDLE]");
            }

            // Progress Bar
            ImGui::SameLine();
            char buf[32];
            sprintf(buf, "%d / %d", current_samples, target_samples);
            ImGui::ProgressBar(progress, ImVec2(200, 0), buf);
        }

        ImGui::SameLine();
        ImGui::TextDisabled("|");
        ImGui::SameLine();

        // Toolbar Buttons
        if (ImGui::Button("Save Image")) {
             std::string filename = "Render_" + std::to_string(time(0)) + ".png";
             ctx.render_settings.save_image_requested = true;
        }
        
        ImGui::SameLine();
        if (ctx.is_animation_mode) {
            
             if (rendering_in_progress) {
                 if (UIWidgets::DangerButton("Stop Anim", ImVec2(80, 0))) {
                     extern std::atomic<bool> rendering_stopped_cpu;
                     extern std::atomic<bool> rendering_stopped_gpu;
                     rendering_stopped_cpu = true;
                     rendering_stopped_gpu = true;
                 }
             } else {
                 if (ImGui::Button("Close")) {
                     ctx.is_animation_mode = false;
                     show_render_window = false;
                 }
             }
        }
        else if (ctx.render_settings.is_final_render_mode) {
            if (UIWidgets::DangerButton("Stop", ImVec2(60, 0))) {
                ctx.render_settings.is_final_render_mode = false;
                extern std::atomic<bool> rendering_stopped_cpu; // Also stop loop!
                rendering_stopped_cpu = true;
            }
        } else {
            if (UIWidgets::PrimaryButton("Render", ImVec2(60, 0))) {
                ctx.renderer.resetCPUAccumulation();
                if (ctx.optix_gpu_ptr) ctx.optix_gpu_ptr->resetAccumulation();
                ctx.render_settings.is_final_render_mode = true;
                ctx.start_render = true; 
            }
        }

        ImGui::SameLine();
        ImGui::TextDisabled("|");
        ImGui::SameLine();

        // Zoom & Fit
        static float zoom = 1.0f;
        if (ImGui::Button("Fit")) {
             extern int image_width, image_height;
             ImVec2 avail = ImGui::GetContentRegionAvail();
             // Account for sidebar if visible
             float avail_w = avail.x - (show_sidebar ? 305.0f : 0.0f);
             
             if (image_width > 0 && image_height > 0) {
                 float rX = avail_w / image_width;
                 float rY = avail.y / image_height;
                 zoom = (rX < rY) ? rX : rY;
             }
        }
        ImGui::SameLine();
        ImGui::SetNextItemWidth(80);
        if (SceneUI::DrawSmartFloat("zoom_res", "Zoom", &zoom, 0.1f, 5.0f, "%.1fx", false, nullptr, 16)) {}

        // Sidebar Toggle
        ImGui::SameLine();
        float avail_width_right = ImGui::GetContentRegionAvail().x;
        // Align to right
        ImGui::SameLine(ImGui::GetWindowWidth() - 110.0f);
        if (UIWidgets::SecondaryButton(show_sidebar ? "Hide Panel >>" : "<< Options", ImVec2(90, 0))) {
            show_sidebar = !show_sidebar;
        }

        ImGui::Separator();

        // Layout: Left (Image) | Right (Settings)
        // ---------------------------------------------------------
        float sidebar_width = 300.0f;
        float content_w = ImGui::GetContentRegionAvail().x;
        float image_view_w = show_sidebar ? (content_w - sidebar_width - 8.0f) : content_w;

        // 1. Image Viewer (Left)
        ImGui::BeginChild("RenderView", ImVec2(image_view_w, 0), true, ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_NoMove);
        {
            extern SDL_Texture* raytrace_texture; 
            extern int image_width, image_height;
            
            if (raytrace_texture && image_width > 0 && image_height > 0) {
                 ImGuiIO& io = ImGui::GetIO();
                 
                 // Handle Zoom/Pan if hovered
                 if (ImGui::IsWindowHovered()) {
                     if (io.MouseWheel != 0.0f) {
                         zoom += io.MouseWheel * 0.1f * zoom;
                         if (zoom < 0.1f) zoom = 0.1f;
                         if (zoom > 10.0f) zoom = 10.0f;
                     }
                     if (ImGui::IsMouseDragging(ImGuiMouseButton_Middle)) {
                         ImVec2 delta = io.MouseDelta;
                         ImGui::SetScrollX(ImGui::GetScrollX() - delta.x);
                         ImGui::SetScrollY(ImGui::GetScrollY() - delta.y);
                     }
                 }

                 float w = (float)image_width * zoom;
                 float h = (float)image_height * zoom;
                 
                 // Center logic
                 ImVec2 avail = ImGui::GetContentRegionAvail();
                 float offX = (avail.x > w) ? (avail.x - w) * 0.5f : 0.0f;
                 float offY = (avail.y > h) ? (avail.y - h) * 0.5f : 0.0f;
                 
                 if (offX > 0) ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offX);
                 if (offY > 0) ImGui::SetCursorPosY(ImGui::GetCursorPosY() + offY);

                 SDL_Texture* display_tex = raytrace_texture;
                 if (ctx.is_animation_mode && ctx.animation_preview_texture) {
                     display_tex = ctx.animation_preview_texture;
                     // Use Animation Preview Dimensions
                     w = (float)ctx.animation_preview_width * zoom;
                     h = (float)ctx.animation_preview_height * zoom;
                 }
                 
                 ImGui::Image((ImTextureID)display_tex, ImVec2(w, h));
                 
                 // HUD calls removed from here (moved back to main draw function)
                 
                 if (ImGui::IsItemHovered()) {
                     ImGui::SetTooltip("Res: %dx%d | Zoom: %.1f%%", image_width, image_height, zoom * 100.0f);
                 }
            } else {
                 ImGui::TextColored(ImVec4(1,0,0,1), "No Render Output Available");
            }
        }
        ImGui::EndChild();

        // 2. Sidebar (Right)
        if (show_sidebar) {
            ImGui::SameLine();
            ImGui::BeginChild("RenderSettingsSidebar", ImVec2(sidebar_width, 0), true);
            
            ImGui::TextDisabled("Render Adjustment");
            ImGui::Separator();
            
            // Draw controls
            DrawRenderWindowToneMapControls(ctx);
            
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::TextDisabled("Stats:");
            ImGui::Text("Samples: %d", current_samples);
            ImGui::Text("Resolution: %dx%d", ctx.render_settings.final_render_width, ctx.render_settings.final_render_height);
            
            ImGui::EndChild();
        }
    }
    ImGui::End();
    ImGui::PopStyleColor();
    
    // Safety: If window is closed, ensure we exit final render mode
    if (!show_render_window && ctx.render_settings.is_final_render_mode) {
        ctx.render_settings.is_final_render_mode = false;
    }
    
    // Sync F12 trigger from main loop
    if (show_render_window && ctx.start_render && !ctx.render_settings.is_final_render_mode && current_samples < 5) {
         // Detect if F12 just opened this
         ctx.render_settings.is_final_render_mode = true;
    }
}

void SceneUI::tryExit() {
    if (ProjectManager::getInstance().hasUnsavedChanges()) {
        pending_action = PendingAction::Exit;
        show_exit_confirmation = true;
    } else {
        extern bool quit;
        quit = true;
    }
}

void SceneUI::tryNew(UIContext& ctx) {
    if (ProjectManager::getInstance().hasUnsavedChanges()) {
        pending_action = PendingAction::NewProject;
        show_exit_confirmation = true;
    } else {
        performNewProject(ctx);
    }
}

void SceneUI::tryOpen(UIContext& ctx) {
    if (ProjectManager::getInstance().hasUnsavedChanges()) {
        pending_action = PendingAction::OpenProject;
        show_exit_confirmation = true;
    } else {
        performOpenProject(ctx);
    }
}

void SceneUI::drawExitConfirmation(UIContext& ctx) {
    if (!show_exit_confirmation) return;

    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

    ImGui::OpenPopup("Unsaved Changes?");

    if (ImGui::BeginPopupModal("Unsaved Changes?", &show_exit_confirmation, ImGuiWindowFlags_AlwaysAutoResize)) {
        
        std::string actionName = "exiting";
        if (pending_action == PendingAction::NewProject) actionName = "creating a new project";
        else if (pending_action == PendingAction::OpenProject) actionName = "opening a project";
        else if (pending_action == PendingAction::Exit) actionName = "exiting";

        ImGui::Text("You have unsaved changes.");
        ImGui::Text("Do you want to save them before %s?", actionName.c_str());
        ImGui::Separator();

        // Save & Continue
        if (ImGui::Button("Save & Continue", ImVec2(140, 0))) {
            std::string path = ProjectManager::getInstance().getCurrentFilePath();
            if (path.empty()) {
                 path = saveFileDialogW(L"RayTrophi Project (.rtp)\0*.rtp\0", L"rtp");
            }

            if (!path.empty()) {
                rendering_stopped_cpu = true;
                bool success = ProjectManager::getInstance().saveProject(path, ctx.scene, ctx.render_settings, ctx.renderer);
                
                try {
                    std::string auxPath = path + ".aux.json";
                    nlohmann::json rootJson;
                    rootJson["terrain_graph"] = terrainNodeGraph.toJson();
                     rootJson["viewport_settings"] = {
                        {"shading_mode", viewport_settings.shading_mode},
                        {"show_gizmos", viewport_settings.show_gizmos},
                        {"show_camera_hud", viewport_settings.show_camera_hud},
                        {"show_focus_ring", viewport_settings.show_focus_ring},
                        {"show_zoom_ring", viewport_settings.show_zoom_ring}
                    };
                    rootJson["guide_settings"] = {
                        {"show_safe_areas", guide_settings.show_safe_areas},
                        {"safe_area_type", guide_settings.safe_area_type},
                        {"show_letterbox", guide_settings.show_letterbox},
                        {"aspect_ratio_index", guide_settings.aspect_ratio_index},
                        {"show_grid", guide_settings.show_grid},
                        {"grid_type", guide_settings.grid_type},
                        {"show_center", guide_settings.show_center}
                    };
                    rootJson["sync_sun_with_light"] = sync_sun_with_light;
                    std::ofstream auxFile(auxPath);
                    if (auxFile.is_open()) {
                        auxFile << rootJson.dump(2);
                        auxFile.close();
                    }
                } catch (...) {}
                
                rendering_stopped_cpu = false;

                if (success) {
                    ImGui::CloseCurrentPopup();
                    show_exit_confirmation = false;
                    
                    if (pending_action == PendingAction::Exit) {
                         extern bool quit; quit = true;
                    } else if (pending_action == PendingAction::NewProject) {
                         performNewProject(ctx);
                    } else if (pending_action == PendingAction::OpenProject) {
                         performOpenProject(ctx);
                    }
                    pending_action = PendingAction::None;
                }
            }
        }

        ImGui::SameLine();

        // Discard & Continue
        ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(0.0f, 0.6f, 0.6f));
        if (ImGui::Button("Discard & Continue", ImVec2(140, 0))) {
            ImGui::CloseCurrentPopup();
            show_exit_confirmation = false;
            
            if (pending_action == PendingAction::Exit) {
                extern bool quit; quit = true;
            } else if (pending_action == PendingAction::NewProject) {
                 performNewProject(ctx);
            } else if (pending_action == PendingAction::OpenProject) {
                 performOpenProject(ctx);
            }
            pending_action = PendingAction::None;
        }
        ImGui::PopStyleColor();

        ImGui::SameLine();

        // Cancel
        if (ImGui::Button("Cancel", ImVec2(120, 0))) {
            ImGui::CloseCurrentPopup();
            show_exit_confirmation = false;
            pending_action = PendingAction::None;
        }
        
        ImGui::EndPopup();
    }
}




void SceneUI::performNewProject(UIContext& ctx) {
     // Clear selection to remove references to objects about to be deleted
     ctx.selection.clearSelection();

     // Reset Foliage Brush to prevent crashes (referencing deleted terrain)
     foliage_brush.enabled = false;
     foliage_brush.active_group_id = -1;

     // Stop rendering while resetting
     rendering_stopped_cpu = true;
     rendering_stopped_gpu = true;
     
     // 1. Reset Global Project System
     g_ProjectManager.newProject(ctx.scene, ctx.renderer);
     ctx.scene.clear();
     
     // 2. Reset UI-Side Persistent Data (Node Graphs, History, Cache)
     terrainNodeGraph.clear();
     history.clear();
     timeline.reset();
     active_messages.clear();
     invalidateCache();
     
     // 3. Reset Viewport & Guide Settings
     viewport_settings = ViewportDisplaySettings();
     guide_settings = GuideSettings();
     sync_sun_with_light = true;
     
     // 4. Reset Rendering State
     ctx.sample_count = 0;
     ctx.renderer.resetCPUAccumulation();
     if (ctx.optix_gpu_ptr) {
         ctx.optix_gpu_ptr->resetAccumulation();
         // Explicitly clear GPU VDB buffer
         syncVDBVolumesToGPU(ctx); // Sends empty list since scene.vdb_volumes is cleared
     }
     
     // 5. Create Default Scene (Camera, Ground Plane, default light)
     createDefaultScene(ctx.scene, ctx.renderer, ctx.optix_gpu_ptr);
     
     ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
     if (ctx.optix_gpu_ptr) {
         ctx.renderer.rebuildOptiXGeometry(ctx.scene, ctx.optix_gpu_ptr);
     }
     if(ctx.scene.camera) ctx.scene.camera->update_camera_vectors();
     
     active_model_path = "Untitled";
     ctx.start_render = true;
     
     SCENE_LOG_INFO("New project created.");
     addViewportMessage("New Project Created");
     
     g_ProjectManager.getProjectData().is_modified = false;

     pending_action = PendingAction::None;
     show_exit_confirmation = false;
     
     // Reset Animation Graph UI
     g_animGraphUI = AnimGraphUIState();
}

void SceneUI::performOpenProject(UIContext& ctx) {
    if (g_scene_loading_in_progress.load()) {
        SCENE_LOG_WARN("Already loading a project. Please wait...");
        return;
    }
    
    std::string filepath = openFileDialogW(L"RayTrophi Project (.rtp;.rts)\0*.rtp;*.rts\0All Files\0*.*\0");
    if (!filepath.empty()) {
        // Clear selection to remove references to old objects (Fixes ghost camera issue)
        ctx.selection.clearSelection();

        // Reset Foliage Brush
        foliage_brush.enabled = false;
        foliage_brush.active_group_id = -1;

        // 1. Reset UI-Side Persistent Data before loading new project
        terrainNodeGraph.clear();
        history.clear();
        timeline.reset();
        active_messages.clear();
        active_messages.clear();
        invalidateCache();
        
        // Reset Animation Graph UI
        g_animGraphUI = AnimGraphUIState();


        g_scene_loading_in_progress = true;
        rendering_stopped_cpu = true;
        rendering_stopped_gpu = true;
        
        scene_loading = true;
        scene_loading_done = false;
        scene_loading_progress = 0;
        ctx.sample_count = 0;
        scene_loading_stage = "Opening project...";
        
        std::thread loader_thread([this, filepath, &ctx]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            
            std::string ext = filepath.substr(filepath.find_last_of('.'));
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            
            if (ext == ".rtp") {
                g_ProjectManager.openProject(filepath, ctx.scene, ctx.render_settings, ctx.renderer, ctx.optix_gpu_ptr,
                    [this](int p, const std::string& s) {
                        scene_loading_progress = p;
                        scene_loading_stage = s;
                    });
                
                {
                    std::string auxPath = filepath + ".aux.json";
                    std::string oldGraphPath = filepath + ".nodegraph.json";
                    bool loaded = false;

                    if (std::filesystem::exists(auxPath)) {
                        try {
                            std::ifstream file(auxPath);
                            if (file.is_open()) {
                                nlohmann::json rootJson;
                                file >> rootJson;
                                file.close();

                                TerrainObject* terrain = nullptr;
                                auto& terrains = TerrainManager::getInstance().getTerrains();
                                if (!terrains.empty()) terrain = &terrains[0];

                                if (rootJson.contains("terrain_graph")) {
                                     terrainNodeGraph.fromJson(rootJson["terrain_graph"], terrain);
                                }

                                if (rootJson.contains("viewport_settings")) {
                                    auto& vs = rootJson["viewport_settings"];
                                    viewport_settings.shading_mode = vs.value("shading_mode", 1);
                                    viewport_settings.show_gizmos = vs.value("show_gizmos", true);
                                    viewport_settings.show_camera_hud = vs.value("show_camera_hud", true);
                                    viewport_settings.show_focus_ring = vs.value("show_focus_ring", true);
                                    viewport_settings.show_zoom_ring = vs.value("show_zoom_ring", true);
                                }

                                if (rootJson.contains("guide_settings")) {
                                    auto& gs = rootJson["guide_settings"];
                                    guide_settings.show_safe_areas = gs.value("show_safe_areas", false);
                                    guide_settings.safe_area_type = gs.value("safe_area_type", 0);
                                    guide_settings.show_letterbox = gs.value("show_letterbox", false);
                                    guide_settings.aspect_ratio_index = gs.value("aspect_ratio_index", 0);
                                    guide_settings.show_grid = gs.value("show_grid", false);
                                    guide_settings.grid_type = gs.value("grid_type", 0);
                                    guide_settings.show_center = gs.value("show_center", false);
                                }

                                if (rootJson.contains("sync_sun_with_light")) {
                                    sync_sun_with_light = rootJson["sync_sun_with_light"].get<bool>();
                                }
                                
                                SCENE_LOG_INFO("[Load] Auxiliary settings loaded.");
                                loaded = true;
                            }
                        } catch (...) {}
                    } 
                    
                    if (!loaded && std::filesystem::exists(oldGraphPath)) {
                        try {
                            std::ifstream ngFile(oldGraphPath);
                            if (ngFile.is_open()) {
                                nlohmann::json graphJson;
                                ngFile >> graphJson;
                                ngFile.close();
                                TerrainObject* terrain = nullptr;
                                auto& terrains = TerrainManager::getInstance().getTerrains();
                                if (!terrains.empty()) terrain = &terrains[0];
                                terrainNodeGraph.fromJson(graphJson, terrain);
                            }
                        } catch (...) {}
                    }
                }
            } else {
                SceneSerializer::Deserialize(ctx.scene, ctx.render_settings, ctx.renderer, ctx.optix_gpu_ptr, filepath);
            }
            
            invalidateCache();
            active_model_path = g_ProjectManager.getProjectName();
            
            if (ctx.optix_gpu_ptr) cudaDeviceSynchronize();
            
            // Register animation clips with AnimationController
            // This ensures bone animations work immediately after project load
            if (!ctx.scene.animationDataList.empty()) {
                auto& animCtrl = AnimationController::getInstance();
                animCtrl.registerClips(ctx.scene.animationDataList);
                
                // Auto-play first animation clip
                const auto& clips = animCtrl.getAllClips();
                if (!clips.empty()) {
                    animCtrl.play(clips[0].name, 0.0f);  // Instant start, no blend
                    SCENE_LOG_INFO("[SceneUI] Auto-playing animation: " + clips[0].name);
                }
                
                SCENE_LOG_INFO("[SceneUI] Registered " + std::to_string(ctx.scene.animationDataList.size()) + " animation clips after project load.");
            }
            
            g_ProjectManager.getProjectData().is_modified = false;

            // Ensure VDB parameters are synchronized to GPU after load
            SceneUI::syncVDBVolumesToGPU(ctx);

            scene_loading = false;
            scene_loading_done = true;
            g_scene_loading_in_progress = false;
            rendering_stopped_cpu = false;
            rendering_stopped_gpu = false;
        });
        loader_thread.detach();
    }
    
    pending_action = PendingAction::None;
    show_exit_confirmation = false;
}

// ═══════════════════════════════════════════════════════════════════════════════
// SHARED VOLUMETRIC UI (VDB & Gas Simulation)
// ═══════════════════════════════════════════════════════════════════════════════

#include "VDBVolume.h"
#include "GasVolume.h"

bool SceneUI::drawVolumeShaderUI(UIContext& ctx, std::shared_ptr<VolumeShader> shader, VDBVolume* vdb, GasVolume* gas) {
    if (!shader) return false;
    
    bool changed = false;
    ImGui::PushID(shader.get());
    
    // ─────────────────────────────────────────────────────────────────────────
    // DENSITY
    // ─────────────────────────────────────────────────────────────────────────
    // ─────────────────────────────────────────────────────────────────────────
    // DENSITY
    // ─────────────────────────────────────────────────────────────────────────
    if (UIWidgets::BeginSection("Density", ImVec4(0.4f, 0.7f, 1.0f, 1.0f))) {
        ImGui::Indent();
        // Channel Selection
        std::vector<std::string> grids;
        if (vdb) grids = vdb->getAvailableGrids();
        else if (gas) grids = {"density", "fuel", "temperature", "interaction"}; // Standard Gas grids
        
        if (!grids.empty() && ImGui::BeginCombo("Channel", shader->density.channel.c_str())) {
            for (const auto& g : grids) {
                if (ImGui::Selectable(g.c_str(), shader->density.channel == g)) {
                    shader->density.channel = g;
                    changed = true;
                }
            }
            ImGui::EndCombo();
        }
        
        if (ImGui::SliderFloat("Multiplier", &shader->density.multiplier, 0.0f, 100.0f)) changed = true;
        if (ImGui::DragFloatRange2("Remap", &shader->density.remap_low, &shader->density.remap_high, 0.01f, 0.0f, 1.0f)) changed = true;
        if (ImGui::SliderFloat("Edge Falloff", &shader->density.edge_falloff, 0.0f, 2.0f)) changed = true;
        
        ImGui::Unindent();
        UIWidgets::EndSection();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SCATTERING & ABSORPTION
    // ─────────────────────────────────────────────────────────────────────────
    // ─────────────────────────────────────────────────────────────────────────
    // SCATTERING & ABSORPTION
    // ─────────────────────────────────────────────────────────────────────────
    if (UIWidgets::BeginSection("Scattering & Absorption", ImVec4(0.8f, 0.5f, 1.0f, 1.0f))) {
        ImGui::Indent();
        float col[3] = { shader->scattering.color.x, shader->scattering.color.y, shader->scattering.color.z };
        if (ImGui::ColorEdit3("Scattering Color", col)) {
            shader->scattering.color = Vec3(col[0], col[1], col[2]);
            changed = true;
        }
        if (ImGui::DragFloat("Scattering Strength", &shader->scattering.coefficient, 0.1f, 0.0f, 100.0f)) changed = true;
        if (ImGui::SliderFloat("Anisotropy (G)", &shader->scattering.anisotropy, -0.99f, 0.99f)) changed = true;
        
        ImGui::Separator();
        float abs_col[3] = { shader->absorption.color.x, shader->absorption.color.y, shader->absorption.color.z };
        if (ImGui::ColorEdit3("Absorption Color", abs_col)) {
            shader->absorption.color = Vec3(abs_col[0], abs_col[1], abs_col[2]);
            changed = true;
        }
        if (ImGui::DragFloat("Absorption Coeff", &shader->absorption.coefficient, 0.1f, 0.0f, 100.0f)) changed = true;
        
        if (ImGui::TreeNode("Advanced Scattering")) {
            if (ImGui::SliderFloat("Back Scatter G", &shader->scattering.anisotropy_back, -0.99f, 0.0f)) changed = true;
            if (ImGui::SliderFloat("Lobe Mix", &shader->scattering.lobe_mix, 0.0f, 1.0f)) changed = true;
            if (ImGui::SliderFloat("Multi-Scatter", &shader->scattering.multi_scatter, 0.0f, 1.0f)) changed = true;
            ImGui::TreePop();
        }
        ImGui::Unindent();
        UIWidgets::EndSection();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // EMISSION (FIRE CONTROLS)
    // ─────────────────────────────────────────────────────────────────────────
    // ─────────────────────────────────────────────────────────────────────────
    // EMISSION (FIRE CONTROLS)
    // ─────────────────────────────────────────────────────────────────────────
    if (UIWidgets::BeginSection("Emission / Fire", ImVec4(1.0f, 0.5f, 0.2f, 1.0f))) {
        ImGui::Indent();
        const char* modes[] = { "None", "Constant", "Blackbody", "Channel" };
        int mode = static_cast<int>(shader->emission.mode);
        if (ImGui::Combo("Emission Mode", &mode, modes, 4)) {
            shader->emission.mode = static_cast<VolumeEmissionMode>(mode);
            changed = true;
        }

        if (shader->emission.mode == VolumeEmissionMode::Constant) {
            float ecol[3] = { shader->emission.color.x, shader->emission.color.y, shader->emission.color.z };
            if (ImGui::ColorEdit3("Color", ecol)) {
                shader->emission.color = Vec3(ecol[0], ecol[1], ecol[2]);
                changed = true;
            }
            if (ImGui::DragFloat("Intensity", &shader->emission.intensity, 0.1f, 0.0f, 1000.0f)) changed = true;
        }
        else if (shader->emission.mode == VolumeEmissionMode::Blackbody) {
            std::vector<std::string> grids;
            if (vdb) grids = vdb->getAvailableGrids();
            else if (gas) grids = {"temperature", "fuel", "density"}; 

            if (!grids.empty() && ImGui::BeginCombo("Temp Channel", shader->emission.temperature_channel.c_str())) {
                for (const auto& g : grids) {
                    if (ImGui::Selectable(g.c_str(), shader->emission.temperature_channel == g)) {
                         shader->emission.temperature_channel = g;
                         changed = true;
                    }
                }
                ImGui::EndCombo();
            }
            
            if (ImGui::SliderFloat("Temp Scale", &shader->emission.temperature_scale, 0.1f, 10.0f)) changed = true;
            if (ImGui::SliderFloat("Blackbody Intensity", &shader->emission.blackbody_intensity, 0.0f, 100.0f)) changed = true;
            
            // Temperature range for color mapping
            ImGui::Separator();
            ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.3f, 1.0f), "Temperature Range (K above ambient)");
            if (ImGui::DragFloat("Temp Min", &shader->emission.temperature_min, 10.0f, 0.0f, 2000.0f, "%.0f K")) changed = true;
            if (ImGui::DragFloat("Temp Max", &shader->emission.temperature_max, 50.0f, 100.0f, 5000.0f, "%.0f K")) changed = true;
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Fire typically ranges 500-1500K above ambient.\nExplosions can reach 2000-3000K.");
            }
            ImGui::Separator();
            
            // ═══════════════════════════════════════════════════════════════
            // INTERACTIVE COLOR RAMP EDITOR (Now Shared!)
            // ═══════════════════════════════════════════════════════════════
            ImGui::Spacing();
            if (ImGui::Checkbox("Use Interactive Color Ramp", &shader->emission.color_ramp.enabled)) changed = true;
            
            if (shader->emission.color_ramp.enabled) {
                auto& ramp = shader->emission.color_ramp;
                ImDrawList* draw_list = ImGui::GetWindowDrawList();
                ImVec2 p = ImGui::GetCursorScreenPos();
                float width = (std::max)(100.0f, ImGui::GetContentRegionAvail().x);
                float height = 24.0f;
                float marker_size = 6.0f;
                
                static int selected_stop = -1;
                static int dragging_stop = -1;

                ImGui::InvisibleButton("gradient_bar", ImVec2(width, height + marker_size * 2));
                bool is_clicked = ImGui::IsItemClicked(0);
                ImVec2 mouse_pos = ImGui::GetIO().MousePos;
                float mouse_t = (std::max)(0.0f, (std::min)(1.0f, (mouse_pos.x - p.x) / width));

                if (is_clicked) {
                    bool hit = false;
                    for(int i=0; i<(int)ramp.stops.size(); ++i) {
                        if(fabs(p.x + ramp.stops[i].position * width - mouse_pos.x) < 8) {
                            selected_stop = i; dragging_stop = i; hit = true; break;
                        }
                    }
                    if(!hit) {
                        ColorRampStop stop; stop.position = mouse_t; stop.color = ramp.sample(mouse_t);
                        ramp.stops.push_back(stop);
                        std::sort(ramp.stops.begin(), ramp.stops.end(), [](auto& a, auto& b){ return a.position < b.position; });
                        changed = true;
                    }
                }
                
                if (dragging_stop != -1 && ImGui::IsMouseDown(0)) {
                    ramp.stops[dragging_stop].position = mouse_t;
                    std::sort(ramp.stops.begin(), ramp.stops.end(), [](auto& a, auto& b){ return a.position < b.position; });
                    for(int i=0; i<(int)ramp.stops.size(); ++i) if(ramp.stops[i].position == mouse_t) dragging_stop = i;
                    changed = true;
                } else dragging_stop = -1;

                // Draw Ramp
                for(int i=0; i<(int)width; ++i) {
                    Vec3 c = ramp.sample((float)i/width);
                    draw_list->AddRectFilled(ImVec2(p.x+i, p.y), ImVec2(p.x+i+1, p.y+height), IM_COL32(c.x*255, c.y*255, c.z*255, 255));
                }
                // Draw Markers
                for(int i=0; i<(int)ramp.stops.size(); ++i) {
                    float x = p.x + ramp.stops[i].position * width;
                    draw_list->AddTriangleFilled(ImVec2(x, p.y+height+marker_size*2), ImVec2(x-marker_size, p.y+height), ImVec2(x+marker_size, p.y+height), (i==selected_stop)?IM_COL32(255,255,0,255):IM_COL32(255,255,255,255));
                }
                ImGui::Dummy(ImVec2(width, marker_size * 2 + 5));

                if (selected_stop >= 0 && selected_stop < (int)ramp.stops.size()) {
                    if (ImGui::SliderFloat("Stop Pos", &ramp.stops[selected_stop].position, 0, 1)) changed = true;
                    float c[3] = { ramp.stops[selected_stop].color.x, ramp.stops[selected_stop].color.y, ramp.stops[selected_stop].color.z };
                    if (ImGui::ColorEdit3("Stop Color", c)) { ramp.stops[selected_stop].color = Vec3(c[0],c[1],c[2]); changed = true; }
                    if (ramp.stops.size() > 2 && ImGui::Button("Delete Stop")) { ramp.stops.erase(ramp.stops.begin()+selected_stop); selected_stop = -1; changed = true; }
                }
            }
        }
        ImGui::Unindent();
        UIWidgets::EndSection();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // RAY MARCHING QUALITY
    // ─────────────────────────────────────────────────────────────────────────
    // ─────────────────────────────────────────────────────────────────────────
    // RAY MARCHING QUALITY
    // ─────────────────────────────────────────────────────────────────────────
    if (UIWidgets::BeginSection("Ray Marching Quality", ImVec4(0.7f, 0.7f, 0.7f, 1.0f), false)) { // Closed by default
        ImGui::Indent();
        if (ImGui::SliderFloat("Step Size", &shader->quality.step_size, 0.001f, 1.0f, "%.3f")) changed = true;
        if (ImGui::SliderInt("Max Steps", &shader->quality.max_steps, 8, 4096)) changed = true;
        if (ImGui::SliderInt("Shadow Steps", &shader->quality.shadow_steps, 0, 128)) changed = true;
        if (ImGui::SliderFloat("Shadow Strength", &shader->quality.shadow_strength, 0.0f, 1.0f)) changed = true;
        ImGui::Unindent();
        UIWidgets::EndSection();
    }

    // NOTE: Shader presets moved to Gas Simulation panel (scene_ui_gas.hpp)
    // The "Quick Presets" there configure BOTH simulation AND shader for best results

    ImGui::PopID();

    if (changed) {
        ctx.renderer.resetCPUAccumulation();
        if (ctx.optix_gpu_ptr) {
            syncVDBVolumesToGPU(ctx); 
            ctx.optix_gpu_ptr->resetAccumulation();
        }
    }

    return changed;
}
