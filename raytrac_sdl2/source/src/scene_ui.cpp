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
#include "ui_modern.h"
#include "imgui.h"
#include "ImGuizmo.h"  // Transform gizmo
#include <string>
#include <memory>  // For std::make_unique
#include "KeyframeSystem.h"   // For keyframe animation
#include "scene_ui_guides.hpp" // Viewport guides (safe areas, letterbox, grids)
#include "TimelineWidget.h"   // Custom timeline widget
#include "scene_data.h"
#include "ParallelBVHNode.h"
#include "Triangle.h"  // For object hierarchy
#include "PointLight.h"
#include "DirectionalLight.h"
#include "SpotLight.h"
#include "AreaLight.h"
#include "PrincipledBSDF.h" // For material editing
#include "Volumetric.h"     // For volumetric material
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

static std::string active_model_path = "No file selected yet.";


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
}
void SceneUI::drawResolutionPanel(UIContext& ctx)
{
    if (UIWidgets::BeginSection("System & Output", ImVec4(0.4f, 0.8f, 0.6f, 1.0f))) {
        

        UIWidgets::ColoredHeader("Resolution", ImVec4(0.7f, 0.9f, 0.8f, 1.0f));
        
        if (ImGui::TreeNodeEx("Display Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
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
            
            ImGui::TreePop();
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

    // Remove NoResize flag, allow collapse for minimizing panel
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus;

    // Add frame styling
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.0f);
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.3f, 0.3f, 0.35f, 1.0f));

    if (ImGui::Begin("Properties", nullptr, flags))
    {
        // Update width if user resized
        side_panel_width = ImGui::GetWindowWidth();

        if (ImGui::BeginTabBar("MainPropertiesTabs"))
        {
            // -------------------------------------------------------------
            // TAB: RENDER (Render Controls & Settings)
            // -------------------------------------------------------------

            if (ImGui::BeginTabItem("Render")) {
                
                // Scene Status (File menu handles loading now)
                UIWidgets::ColoredHeader("Scene Status", ImVec4(1.0f, 0.8f, 0.6f, 1.0f));
                
                if (ctx.scene.initialized) {
                    ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Scene Active");
                    ImGui::SameLine();
                    ImGui::TextDisabled("| %d Objects | %d Lights", 
                        (int)ctx.scene.world.objects.size(), 
                        (int)ctx.scene.lights.size());
                    
                    ImGui::TextDisabled("File: %s", active_model_path.c_str());
                } else {
                    ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.3f, 1.0f), "No Scene Loaded");
                    ImGui::TextDisabled("Use File > Import Model or Open Project");
                }

                // ---------------------------------------------------------
                // RENDER CONFIGURATION & SAMPLING
                // ---------------------------------------------------------
                UIWidgets::Divider();
                UIWidgets::ColoredHeader("Render Engine & Quality", ImVec4(0.6f, 0.8f, 1.0f, 1.0f));

                // 1. Backend Selection
                if (ImGui::TreeNodeEx("Backend & Acceleration", ImGuiTreeNodeFlags_DefaultOpen)) {
                    // GPU Selection
                    bool optix_available = g_hasOptix;
                    if (!optix_available) ImGui::BeginDisabled();
                    if (ImGui::Checkbox("Use OptiX (GPU)", &ctx.render_settings.use_optix)) {
                         ctx.start_render = true; // Trigger restart
                    }
                    if (!optix_available) { 
                        ImGui::EndDisabled();
                        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) 
                            ImGui::SetTooltip("No NVIDIA RTX GPU detected.");
                    }

                    // CPU Selection (Embree)
                    if (!ctx.render_settings.use_optix) {
                        ImGui::Indent();
                        
                        // BVH Mode Selection (Combo Box)
                        const char* bvh_items[] = { "RayTrophi BVH (Custom)", "Embree BVH (Intel)" };
                        int current_bvh = ctx.render_settings.UI_use_embree ? 1 : 0;
                        
                        ImGui::PushItemWidth(200);
                        if (ImGui::Combo("CPU BVH Mode", &current_bvh, bvh_items, IM_ARRAYSIZE(bvh_items))) {
                            ctx.render_settings.UI_use_embree = (current_bvh == 1);
                            
                            // Sync global variable
                            extern bool use_embree;
                            use_embree = ctx.render_settings.UI_use_embree; 
                            
                            // Rebuild BVH immediately
                            ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
                            ctx.renderer.resetCPUAccumulation();
                            
                            SCENE_LOG_INFO("Switched CPU BVH to: " + std::string(bvh_items[current_bvh]));
                        }
                        ImGui::PopItemWidth();
                        
                        ImGui::SameLine();
                        UIWidgets::HelpMarker("Select the acceleration structure backend for CPU rendering.\\nEmbree is faster for complex scenes but requires rebuilding.");
                        
                        // Informational Message
                        if (ctx.render_settings.UI_use_embree) {
                             ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Embree Active: High Performance");
                        } else {
                             ImGui::TextColored(ImVec4(0.5f, 0.8f, 1.0f, 1.0f), "RayTrophi BVH Active");
                        }
                        
                        ImGui::Unindent();
                    }
                    ImGui::TreePop();
                }

                // 2. Settings Tabs (Viewport vs Final)
                ImGui::Spacing();
                if (ImGui::BeginTabBar("RenderSettingsTabs")) {
                    
                    // --- VIEWPORT TAB ---
                    if (ImGui::BeginTabItem("Viewport")) {
                        ImGui::TextDisabled("Realtime Preview Settings");
                        ImGui::Separator();
                        ImGui::Spacing();
                        
                        // Sampling
                        if (ImGui::TreeNodeEx("Viewport Sampling", ImGuiTreeNodeFlags_DefaultOpen)) {
                            ImGui::PushItemWidth(150);
                            ImGui::SliderInt("Max Samples", &ctx.render_settings.max_samples, 1, 64);
                            ImGui::PopItemWidth();
                            UIWidgets::HelpMarker("Viewport accumulation limit. Set high for continuous refinement.");
                            ImGui::TreePop();
                        }

                        ImGui::Spacing();
                        
                        // Denoising
                        if (ImGui::TreeNodeEx("Viewport Denoising", ImGuiTreeNodeFlags_DefaultOpen)) {
                            ImGui::Checkbox("Enable Denoising", &ctx.render_settings.use_denoiser);
                            
                            if (ctx.render_settings.use_denoiser) {
                                ImGui::Spacing();
                                ImGui::PushItemWidth(150);
                                ImGui::SliderFloat("Blend Factor", &ctx.render_settings.denoiser_blend_factor, 0.0f, 1.0f);
                                ImGui::PopItemWidth();
                                UIWidgets::HelpMarker("0.0 = original, 1.0 = fully denoised");
                            }
                            ImGui::TreePop();
                        }
                        
                        ImGui::EndTabItem();
                    }

                    // --- FINAL RENDER TAB ---
                    if (ImGui::BeginTabItem("Final Render")) {
                        ImGui::TextDisabled("Output Settings");
                        ImGui::Separator();
                        ImGui::Spacing();
                        
                        // === RESOLUTION GROUP ===
                        if (ImGui::TreeNodeEx("Output Resolution", ImGuiTreeNodeFlags_DefaultOpen)) {
                                // Resolution Source dropdown
                                const char* source_names[] = { "Native (Window)", "Custom", "From Aspect Ratio" };
                                int source_idx = (int)ctx.render_settings.resolution_source;
                                
                                ImGui::PushItemWidth(200);
                                if (ImGui::Combo("Source", &source_idx, source_names, 3)) {
                                    ctx.render_settings.resolution_source = (ResolutionSource)source_idx;
                                }
                                ImGui::PopItemWidth();
                                
                                ImGui::Spacing();
                                
                                if (ctx.render_settings.resolution_source == ResolutionSource::Native) {
                                    // Native - use current window size
                                    extern int image_width, image_height;
                                    ctx.render_settings.final_render_width = image_width;
                                    ctx.render_settings.final_render_height = image_height;
                                    ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1.0f), 
                                        "Output: %d x %d (current viewport)", image_width, image_height);
                                }
                                else if (ctx.render_settings.resolution_source == ResolutionSource::Custom) {
                                    // Custom - manual input
                                    const char* res_items[] = { "1280x720 (HD)", "1920x1080 (FHD)", "2560x1440 (2K)", "3840x2160 (4K)", "Custom" };
                                    static int current_res_item = 1; // Default 1080p
                                    
                                    ImGui::PushItemWidth(200);
                                    if (ImGui::Combo("Preset", &current_res_item, res_items, IM_ARRAYSIZE(res_items))) {
                                        if (current_res_item == 0) { ctx.render_settings.final_render_width = 1280; ctx.render_settings.final_render_height = 720; }
                                        if (current_res_item == 1) { ctx.render_settings.final_render_width = 1920; ctx.render_settings.final_render_height = 1080; }
                                        if (current_res_item == 2) { ctx.render_settings.final_render_width = 2560; ctx.render_settings.final_render_height = 1440; }
                                        if (current_res_item == 3) { ctx.render_settings.final_render_width = 3840; ctx.render_settings.final_render_height = 2160; }
                                    }
                                    ImGui::PopItemWidth();
                                    
                                    ImGui::PushItemWidth(100);
                                    if (ImGui::InputInt("Width", &ctx.render_settings.final_render_width)) current_res_item = 4;
                                    if (ImGui::InputInt("Height", &ctx.render_settings.final_render_height)) current_res_item = 4;
                                    ImGui::PopItemWidth();
                                }
                                else if (ctx.render_settings.resolution_source == ResolutionSource::FromAspect) {
                                    // From Aspect Ratio - calculate width from height and aspect
                                    ImGui::PushItemWidth(200);
                                    if (ImGui::BeginCombo("Aspect Ratio", 
                                        CameraPresets::ASPECT_RATIOS[ctx.render_settings.aspect_ratio_index].name)) 
                                    {
                                        for (size_t i = 0; i < CameraPresets::ASPECT_RATIO_COUNT; ++i) {
                                            bool is_selected = (ctx.render_settings.aspect_ratio_index == (int)i);
                                            std::string label = std::string(CameraPresets::ASPECT_RATIOS[i].name) + 
                                                " - " + CameraPresets::ASPECT_RATIOS[i].usage;
                                            if (ImGui::Selectable(label.c_str(), is_selected)) {
                                                ctx.render_settings.aspect_ratio_index = (int)i;
                                            }
                                            if (is_selected) ImGui::SetItemDefaultFocus();
                                        }
                                        ImGui::EndCombo();
                                    }
                                    ImGui::PopItemWidth();
                                    
                                    // Base height input
                                    ImGui::PushItemWidth(100);
                                    ImGui::InputInt("Base Height", &ctx.render_settings.aspect_base_height);
                                    ctx.render_settings.aspect_base_height = std::clamp(ctx.render_settings.aspect_base_height, 480, 8192);
                                    ImGui::PopItemWidth();
                                    
                                    // Calculate final dimensions
                                    float ratio = CameraPresets::ASPECT_RATIOS[ctx.render_settings.aspect_ratio_index].ratio;
                                    if (ratio > 0.01f) {
                                        ctx.render_settings.final_render_width = (int)(ctx.render_settings.aspect_base_height * ratio);
                                        ctx.render_settings.final_render_height = ctx.render_settings.aspect_base_height;
                                    } else {
                                        // Native ratio - use base_height as height, calculate natural aspect
                                        extern int image_width, image_height;
                                        float native_ratio = (float)image_width / (float)image_height;
                                        ctx.render_settings.final_render_width = (int)(ctx.render_settings.aspect_base_height * native_ratio);
                                        ctx.render_settings.final_render_height = ctx.render_settings.aspect_base_height;
                                    }
                                    
                                    // Display calculated resolution
                                    ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.7f, 1.0f), 
                                        "Output: %d x %d", 
                                        ctx.render_settings.final_render_width, 
                                        ctx.render_settings.final_render_height);
                                }
                                
                                ImGui::TextDisabled("(Viewport unaffected until render starts)");
                                ImGui::TreePop();
                        }

                        ImGui::Spacing();
                        
                        // === QUALITY GROUP ===
                        if (ImGui::TreeNodeEx("Render Quality", ImGuiTreeNodeFlags_DefaultOpen)) {
                                ImGui::PushItemWidth(150);
                                ImGui::DragInt("Target Samples", &ctx.render_settings.final_render_samples, 16, 16, 65536);
                                ImGui::PopItemWidth();
                                
                                ImGui::Spacing();
                                ImGui::Checkbox("Render Denoising", &ctx.render_settings.render_use_denoiser);
                                ImGui::TreePop();
                        }
                        
                        ImGui::Spacing();
                        ImGui::Separator();
                        ImGui::Spacing();
                        
                        // === START RENDER BUTTON (CENTERED) ===
                        float button_width = 180.0f;
                        float offset = (ImGui::GetContentRegionAvail().x - button_width) * 0.5f;
                        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offset);
                        
                        if (UIWidgets::PrimaryButton("Start Final Render (F12)", ImVec2(button_width, 28))) {
                             extern bool show_render_window;
                             show_render_window = true;
                             ctx.render_settings.is_final_render_mode = true;
                             ctx.start_render = true;
                        }

                        ImGui::Spacing();
                        ImGui::Separator();
                        ImGui::Spacing();
                        
                        // === ANIMATION SEQUENCE GROUP (VERTICAL) ===
                        if (ImGui::TreeNodeEx("Animation Sequence", ImGuiTreeNodeFlags_DefaultOpen)) {
                                ImGui::PushItemWidth(150);
                                ImGui::InputInt("Start Frame", &ctx.render_settings.animation_start_frame);
                                ImGui::InputInt("End Frame", &ctx.render_settings.animation_end_frame);
                                ImGui::InputInt("FPS", &ctx.render_settings.animation_fps);
                                ImGui::PopItemWidth();
                                
                                ImGui::Spacing();
                                
                                // Center button like Final Render
                                float anim_button_width = 180.0f;
                                float anim_offset = (ImGui::GetContentRegionAvail().x - anim_button_width) * 0.5f;
                                ImGui::SetCursorPosX(ImGui::GetCursorPosX() + anim_offset);
                                
                                if (UIWidgets::PrimaryButton("Render Animation", ImVec2(anim_button_width, 28))) {
                                     ctx.render_settings.start_animation_render = true;
                                     ctx.is_animation_mode = true; // Enable unified render window for animation
                                     extern bool show_render_window;
                                     show_render_window = true;
                                     ctx.show_animation_preview = true; // Keep for internal logic (texture update)
                                }
                                ImGui::Spacing();
                                // REMOVED: Separate Checkbox (integrated into main window)
                                // ImGui::SetCursorPosX(ImGui::GetCursorPosX() + anim_offset);
                                // ImGui::Checkbox("Show Preview", &ctx.show_animation_preview);
                                ImGui::TreePop();
                        }
                        
                        ImGui::EndTabItem();
                    }
                    
                    ImGui::EndTabBar();
                }

                // 3. Global Settings (Path Tracing)
                ImGui::Spacing();
                if (ImGui::TreeNodeEx("Light Paths & Optimization", ImGuiTreeNodeFlags_DefaultOpen)) {
                        ImGui::PushItemWidth(150);
                        ImGui::DragInt("Max Bounces", &ctx.render_settings.max_bounces, 1, 1, 32);
                        ImGui::PopItemWidth();
                        UIWidgets::HelpMarker("Maximum ray bounce depth for global illumination");
                        
                        ImGui::Spacing();
                        ImGui::Checkbox("Adaptive Sampling", &ctx.render_settings.use_adaptive_sampling);
                        if (ctx.render_settings.use_adaptive_sampling) {
                            ImGui::Spacing();
                            ImGui::PushItemWidth(150);
                            ImGui::DragFloat("Noise Threshold", &ctx.render_settings.variance_threshold, 0.001f, 0.0001f, 1.0f, "%.4f");
                            UIWidgets::HelpMarker("Stop sampling when pixel variance is below this value");
                            ImGui::DragInt("Min Samples", &ctx.render_settings.min_samples, 1, 1, 64);
                            UIWidgets::HelpMarker("Minimum samples per pixel before adaptive sampling kicks in");
                            ImGui::PopItemWidth();
                        }
                    ImGui::TreePop();
                }

                // ---------------------------------------------------------
                // CONTROL PANEL (Progress & Action)
                // ---------------------------------------------------------
                ImGui::Spacing();
                ImGui::Separator();
                
                // Progress Bar with overlay text
                float progress = ctx.render_settings.render_progress;
                int current = ctx.render_settings.render_current_samples;
                int target = ctx.render_settings.render_target_samples;
                
                // Colorize progress bar based on state
                ImVec4 progress_color = ImVec4(0.2f, 0.7f, 0.3f, 1.0f); // Green
                if (ctx.render_settings.is_render_paused) progress_color = ImVec4(0.8f, 0.6f, 0.2f, 1.0f); // Orange
                
                ImGui::PushStyleColor(ImGuiCol_PlotHistogram, progress_color);
                char overlay[64];
                snprintf(overlay, sizeof(overlay), "%d / %d Samples (%.1f%%)", current, target, progress * 100.0f);
                ImGui::ProgressBar(progress, ImVec2(-1, 24), overlay);
                ImGui::PopStyleColor();

                ImGui::Spacing();
                
                // Control Buttons
                bool is_active = ctx.render_settings.is_rendering_active;
                bool is_paused = ctx.render_settings.is_render_paused;
                
                float btn_w = ImGui::GetContentRegionAvail().x / 2.0f - 4.0f;
                
                // Start/Pause Button
                if (is_active && !is_paused) {
                    if (UIWidgets::SecondaryButton("Pause", ImVec2(btn_w, 30))) {
                         ctx.render_settings.is_render_paused = true;
                    }
                } else {
                    const char* btn_label = is_paused ? "Resume Run" : "Start Render";
                    if (UIWidgets::PrimaryButton(btn_label, ImVec2(btn_w, 30))) {
                        ctx.render_settings.is_render_paused = false;
                        if (!is_active) ctx.start_render = true;
                    }
                }
                
                ImGui::SameLine();
                
                // Stop/Reset Button
                if (UIWidgets::DangerButton("Stop & Reset", ImVec2(btn_w, 30))) {
                    rendering_stopped_cpu = true;
                    rendering_stopped_gpu = true;
                    ctx.render_settings.render_progress = 0.0f;
                    ctx.render_settings.render_current_samples = 0;
                }
                
                ImGui::Spacing();
                if (ImGui::Button("Save Final Image...", ImVec2(-1, 0))) {
                    ctx.render_settings.save_image_requested = true;
                }

                ImGui::EndTabItem();
            }

            // -------------------------------------------------------------
            // TAB: WORLD (Environment, Sky, Lights)
            // -------------------------------------------------------------
            if (ImGui::BeginTabItem("World")) {
                drawWorldContent(ctx);

                ImGui::EndTabItem();
            }

            // NOTE: Camera tab removed - camera settings now in Scene Edit > Selection Properties

            // -------------------------------------------------------------
            // TAB: SCENE (Hierarchy)
            // -------------------------------------------------------------
            ImGuiTabItemFlags tab_flags = 0;
            if (focus_scene_edit_tab) {
                tab_flags = ImGuiTabItemFlags_SetSelected;
                focus_scene_edit_tab = false;
            }
            if (ImGui::BeginTabItem("Scene Edit", nullptr, tab_flags)) {
                drawSceneHierarchy(ctx);
                // Material Editor is now drawn inside drawSceneHierarchy
                ImGui::EndTabItem();
            }

            // -------------------------------------------------------------
            // TAB: SYSTEM (App settings)
            // -------------------------------------------------------------
            if (ImGui::BeginTabItem("System")) {
                drawThemeSelector();
                drawResolutionPanel(ctx);
                
                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }
    }
    ImGui::End();
    ImGui::PopStyleColor(); // Border
    ImGui::PopStyleVar();   // BorderSize
}

// Main Menu Bar implementation moved to separate file: scene_ui_menu.hpp check end of file

#include "scene_ui_menu.hpp"


void SceneUI::draw(UIContext& ctx)
{
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
    drawSelectionGizmos(ctx);
    drawCameraGizmos(ctx);  // Draw camera frustum icons
    drawViewportControls(ctx);  // Blender-style viewport overlay
    drawFocusIndicator(ctx);  // Split-prism focus aid
    drawZoomRing(ctx);  // FOV zoom control
    drawExposureInfo(ctx);  // Cinema camera exposure bar

    handleSceneInteraction(ctx, gizmo_hit);
    processDeferredSceneUpdates(ctx);
    drawAuxWindows(ctx);
    
    // Global Sun Sync (Light -> Nishita)
    processSunSync(ctx);

    extern void DrawRenderWindow(UIContext & ctx);
    DrawRenderWindow(ctx);

    // --- ANIMATION PREVIEW PANEL (REMOVED - Integrated into Render Result) ---
    /*
    if (ctx.animation_preview_texture && ctx.show_animation_preview) {
        if (ImGui::Begin("Animation Preview", &ctx.show_animation_preview, ImGuiWindowFlags_None)) {
             // ...
        }
        ImGui::End();
    }
    */
             int w = ctx.animation_preview_width;
             int h = ctx.animation_preview_height;
             
             if (w > 0 && h > 0) {
                 ImVec2 avail = ImGui::GetContentRegionAvail();
                 if (avail.x < 50) avail.x = 50;
                 if (avail.y < 50) avail.y = 50;
                 
                 float aspect = (float)w / (float)h;
                 float display_w = avail.x;
                 float display_h = display_w / aspect;
                 
                 // If height exceeds available space, clamp by height
                 if (display_h > avail.y) {
                     display_h = avail.y;
                     display_w = display_h * aspect;
                 }

                 ImGui::Image((ImTextureID)ctx.animation_preview_texture, ImVec2(display_w, display_h));
                 
                 if (ctx.render_settings.start_animation_render) {
                    ImGui::Text("Initializing...");
                 } else {
                   
                    if (rendering_in_progress) {
                        ImGui::Text("Rendering Frame: %d / %d", ctx.render_settings.animation_current_frame, ctx.render_settings.animation_end_frame);
                    } else {
                        ImGui::Text("Last Rendered Frame: %d", ctx.render_settings.animation_current_frame);
                    }
                 }
             }
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

    ImGui::SetNextWindowPos(ImVec2(0, screen_y - status_bar_height));
    ImGui::SetNextWindowSize(ImVec2(screen_x, status_bar_height));

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
        bool anim_active = show_animation_panel;

        if (UIWidgets::StateButton(
            anim_active ? "[Timeline]" : "Timeline",
            anim_active,
            ImVec4(0.3f, 0.6f, 1.0f, 1.0f),
            ImVec4(0.2f, 0.2f, 0.2f, 1.0f),
            ImVec2(80, 20)))
        {
            show_animation_panel = !show_animation_panel;
            if (show_animation_panel) show_scene_log = false;
        }

        ImGui::SameLine();

        bool log_active = show_scene_log;
        if (UIWidgets::StateButton(
            log_active ? "[Console]" : "Console",
            log_active,
            ImVec4(0.3f, 0.6f, 1.0f, 1.0f),
            ImVec4(0.2f, 0.2f, 0.2f, 1.0f),
            ImVec2(80, 20)))
        {
            show_scene_log = !show_scene_log;
            if (show_scene_log) show_animation_panel = false;
        }

        ImGui::SameLine();
        ImGui::TextDisabled("|");
        ImGui::SameLine();

        if (ctx.scene.initialized) {
            // Use mesh_cache size for logical object count instead of primitive count
            int obj_count = (int)mesh_cache.size();
            // If cache not ready (e.g. startup), show primitive count but indicate it
            if (obj_count == 0 && !ctx.scene.world.objects.empty()) {
                ImGui::Text("Scene: %d Primitives", (int)ctx.scene.world.objects.size());
            } else {
                ImGui::Text("Scene: %d Objects", obj_count);
            }
            
            ImGui::SameLine();
            ImGui::Text("%d Lights", (int)ctx.scene.lights.size());
            
            ImGui::SameLine();
            ImGui::TextDisabled("|");
            ImGui::SameLine();

            // Show selected item name
            if (ctx.selection.hasSelection()) {
                ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), 
                    "Selected: %s", ctx.selection.selected.name.c_str());
            } else {
                ImGui::TextDisabled("No Selection");
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
    bool show_bottom = (show_animation_panel || show_scene_log);
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
    }
    ImGui::End();
    ImGui::PopStyleVar(); // Pop WindowPadding
    // ImGui::PopStyleVar(); // Removed redundant PopStyleVar
    // ImGui::PopStyleColor(); // Removed hardcoded color push
}
bool SceneUI::drawOverlays(UIContext& ctx)
{
    bool gizmo_hit = false;

    if (ctx.scene.camera && ctx.selection.show_gizmo) {
        drawLightGizmos(ctx, gizmo_hit);
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
        if (ctx.optix_gpu_ptr) {
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
    
    for (size_t i = 0; i < objects.size(); ++i) {
        auto tri = std::dynamic_pointer_cast<Triangle>(objects[i]);
        if (tri) {
            std::string name = tri->nodeName.empty() ? "Unnamed" : tri->nodeName;
            mesh_cache[name].push_back({(int)i, tri});
        }
    }
    
    // Transfer to sequential vector for ImGui Clipper
    mesh_ui_cache.reserve(mesh_cache.size());
    for (auto& kv : mesh_cache) {
        mesh_ui_cache.push_back(kv);
    }
    
    mesh_cache_valid = true;
}



// Global flag for Render Window visibility
bool show_render_window = false;

void DrawRenderWindow(UIContext& ctx) {
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
        ImGui::SliderFloat("Zoom", &zoom, 0.1f, 5.0f, "%.1fx");

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
    
