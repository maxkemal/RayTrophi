#include "scene_ui.h"
#include "ui_modern.h"
#include "imgui.h"
#include "ImGuizmo.h"  // Transform gizmo
#include <string>
#include "scene_data.h"
#include "ParallelBVHNode.h"
#include "Triangle.h"  // For object hierarchy
#include "PointLight.h"
#include "DirectionalLight.h"
#include "SpotLight.h"
#include "AreaLight.h"
#include "PrincipledBSDF.h" // For material editing
#include "AssimpLoader.h"  // For scene rebuild after object changes
#include "SceneCommand.h"  // For undo/redo
#include "default_scene_creator.hpp"
#include "SceneSerializer.h"
#include "ProjectManager.h"  // Project system
#include <map>  // For mesh grouping
#include <windows.h>
#include <commdlg.h>
#include <shlobj.h>  // SHBrowseForFolder için
#include <string>

#include <chrono>  // Playback timing için
#include <filesystem>  // Frame dosyalarını kontrol için

static int new_width = image_width;
static int new_height = image_height;
static int aspect_w = 16;
static int aspect_h = 9;
static bool modelLoaded = false;
static bool loadFeedback = false; // geçici hata geri bildirimi
static float feedbackTimer = 0.0f;
bool show_animation_panel = false; // Default closed as requested

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




std::string openFileDialogW(const wchar_t* filter = L"All Files\0*.*\0") {
    wchar_t filename[MAX_PATH] = L"";
    OPENFILENAMEW ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = nullptr;
    ofn.lpstrFilter = filter;
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;
    ofn.lpstrTitle = L"Select a file";
    ofn.hwndOwner = GetActiveWindow();
    if (GetOpenFileNameW(&ofn)) {
        int size_needed = WideCharToMultiByte(CP_UTF8, 0, filename, -1, nullptr, 0, nullptr, nullptr);
        std::string utf8_path(size_needed, 0);
        WideCharToMultiByte(CP_UTF8, 0, filename, -1, utf8_path.data(), size_needed, nullptr, nullptr);
        utf8_path.resize(size_needed - 1); // null terminatörü çıkar
        return utf8_path;
    }
    return "";
}

std::string saveFileDialogW(const wchar_t* filter = L"All Files\0*.*\0", const wchar_t* defExt = L"rts") {
    wchar_t filename[MAX_PATH] = L"";
    OPENFILENAMEW ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = GetActiveWindow();
    ofn.lpstrFilter = filter;
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH; // Initialize buffer with 0
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT;
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

// Timeline Panel - Embedded Content
void SceneUI::drawTimelineContent(UIContext& ctx)
{
    static int start_frame = 0;
    static int end_frame = 100;
    static bool frame_range_initialized = false;
    static bool is_playing = false;
    static int playback_frame = 0;
    static auto last_frame_time = std::chrono::steady_clock::now();
    
    // NOT: Window creation code moved out to allow embedding

    if (!ctx.scene.animationDataList.empty()) {
        auto& anim = ctx.scene.animationDataList[0];
        
        // Detect if we switched to a different animation
        static int cached_source_start = -1;
        static int cached_source_end = -1;

        if (anim.startFrame != cached_source_start || anim.endFrame != cached_source_end) {
            frame_range_initialized = false;
            cached_source_start = anim.startFrame;
            cached_source_end = anim.endFrame;
        }

        // Initialize defaults
        if (!frame_range_initialized) {
            start_frame = anim.startFrame;
            end_frame = anim.endFrame;
            playback_frame = start_frame;
            ctx.render_settings.animation_start_frame = start_frame;
            ctx.render_settings.animation_end_frame = end_frame;
            frame_range_initialized = true;
        }
        
        int total_frames = end_frame - start_frame + 1;
        bool is_rendering = rendering_in_progress;
        
        // --- ÜST SATIR: Frame Range ve Ayarlar ---
        ImGui::PushItemWidth(150);
        if (ImGui::SliderInt("Start", &start_frame, anim.startFrame, anim.endFrame)) {
            ctx.render_settings.animation_start_frame = start_frame;
            if (playback_frame < start_frame) playback_frame = start_frame;
        }
        ImGui::SameLine();
        if (ImGui::SliderInt("End", &end_frame, anim.startFrame, anim.endFrame)) {
            ctx.render_settings.animation_end_frame = end_frame;
            if (playback_frame > end_frame) playback_frame = end_frame;
        }
        ImGui::PopItemWidth();
        
        if (start_frame > end_frame) {
            end_frame = start_frame;
            ctx.render_settings.animation_end_frame = end_frame;
        }
        
        // Settings Group
        ImGui::SameLine();
        ImGui::TextDisabled("|");
        ImGui::SameLine();
        
        ImGui::PushItemWidth(100);
        ImGui::SliderInt("FPS", &ctx.render_settings.animation_fps, 1, 60);
        ImGui::PopItemWidth();
        
        ImGui::SameLine();
        ImGui::TextDisabled("(%d frames)", total_frames);
        
        // --- ORTA SATIR: Playback Controls ---
        ImGui::Spacing();
        
        // Play/Pause
        if (UIWidgets::SecondaryButton(is_playing ? "||" : "|>", ImVec2(40, 24))) {
            is_playing = !is_playing;
            if (is_playing) {
                last_frame_time = std::chrono::steady_clock::now();
            }
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(is_playing ? "Pause" : "Play");
        
        ImGui::SameLine();
        // Stop
        if (UIWidgets::SecondaryButton("[]", ImVec2(40, 24))) {
            is_playing = false;
            playback_frame = start_frame;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Stop & Reset");
        
        ImGui::SameLine();
        ImGui::TextDisabled("|");
        ImGui::SameLine();
        
        // Scrubbing / Current Frame
        ImGui::AlignTextToFramePadding();
        ImGui::Text("Frame:");
        ImGui::SameLine();
        
        ImGui::PushItemWidth(80);
        if (ImGui::InputInt("##CurrentFrame", &playback_frame, 0, 0)) {
            playback_frame = std::clamp(playback_frame, start_frame, end_frame);
        }
        ImGui::PopItemWidth();
        
        // Timeline Slider (Scrubber)
        ImGui::SameLine();
        ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 10);
        if (ImGui::SliderInt("##Scrubber", &playback_frame, start_frame, end_frame, "")) {
             // Just scrubber
        }
        ImGui::PopItemWidth();
        
        // Logic Update
        if (is_playing && !is_rendering) {
            auto now = std::chrono::steady_clock::now();
            float elapsed = std::chrono::duration<float>(now - last_frame_time).count();
            float frame_duration = 1.0f / (float)ctx.render_settings.animation_fps;
            
            if (elapsed >= frame_duration) {
                playback_frame++;
                if (playback_frame > end_frame) playback_frame = start_frame;
                last_frame_time = now;
            }
        }
        
        // Store user state
        ctx.render_settings.animation_is_playing = is_playing;
        ctx.render_settings.animation_playback_frame = playback_frame;
        ctx.render_settings.animation_current_frame = playback_frame; // For renderer preview
        
    } else {
        ImGui::TextColored(ImVec4(1,1,0,1), "No animation data found in scene.");
    }
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
void SceneUI::drawResolutionPanel()
{
    if (UIWidgets::BeginSection("Resolution Settings", ImVec4(0.4f, 0.8f, 0.6f, 1.0f))) {
        
        UIWidgets::ColoredHeader("Preset Resolution", ImVec4(0.7f, 0.9f, 0.8f, 1.0f));
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

        UIWidgets::Divider();
        UIWidgets::ColoredHeader("Custom Resolution", ImVec4(0.7f, 0.9f, 0.8f, 1.0f));
        
        ImGui::InputInt("Width", &new_width);
        ImGui::InputInt("Height", &new_height);
        
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
        
        if (UIWidgets::PrimaryButton("Apply", ImVec2(100, 0), resolution_changed))
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
}


void SceneUI::drawToneMapContent(UIContext& ctx) {
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

void SceneUI::drawCameraContent(UIContext& ctx)
{
    // Camera selector for multi-camera support
    if (ctx.scene.cameras.size() > 1) {
        UIWidgets::ColoredHeader("Camera Selection", ImVec4(0.3f, 0.8f, 0.6f, 1.0f));
        
        std::string current_label = "Camera #" + std::to_string(ctx.scene.active_camera_index);
        if (ImGui::BeginCombo("Active Camera", current_label.c_str())) {
            for (size_t i = 0; i < ctx.scene.cameras.size(); ++i) {
                bool is_selected = (i == ctx.scene.active_camera_index);
                std::string label = "Camera #" + std::to_string(i);
                if (ImGui::Selectable(label.c_str(), is_selected)) {
                    ctx.scene.setActiveCamera(i);
                    // Update GPU with new camera
                    if (ctx.optix_gpu_ptr && ctx.scene.camera) {
                        ctx.optix_gpu_ptr->setCameraParams(*ctx.scene.camera);
                        ctx.optix_gpu_ptr->resetAccumulation();
                    }
                    ctx.renderer.resetCPUAccumulation();
                    SCENE_LOG_INFO("Switched to Camera #" + std::to_string(i));
                }
                if (is_selected) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        ImGui::Text("Total Cameras: %zu", ctx.scene.cameras.size());
        ImGui::Separator();
    } else if (ctx.scene.cameras.size() == 1) {
        ImGui::TextDisabled("1 Camera in scene");
    }
    
    if (!ctx.scene.camera) {
        UIWidgets::StatusIndicator("No Camera Available", UIWidgets::StatusType::Warning);
        return;
    }

    Vec3 pos = ctx.scene.camera->lookfrom;
    Vec3 target = ctx.scene.camera->lookat;
    float fov = (float)ctx.scene.camera->vfov;
    float& aperture = ctx.scene.camera->aperture;
    float& focus_dist = ctx.scene.camera->focus_dist;

    // -------- Position & Target --------
    static bool targetLock = true;
    if (UIWidgets::BeginSection("Position & Target", ImVec4(0.4f, 0.7f, 1.0f, 1.0f))) {
        ImGui::PushItemWidth(280);
        bool pos_changed = ImGui::DragFloat3("Position", &pos.x, 0.01f);
        UIWidgets::HelpMarker("Camera world position (X, Y, Z)");

        bool target_changed = false;
        if (!targetLock) {
            target_changed = ImGui::DragFloat3("Target", &target.x, 0.01f);
            UIWidgets::HelpMarker("Point the camera looks at");
        } else {
            ImGui::BeginDisabled();
            ImGui::DragFloat3("Target", &target.x, 0.1f);
            ImGui::EndDisabled();
        }

        ImGui::PopItemWidth();
        ImGui::Checkbox("Lock Target", &targetLock);
        UIWidgets::HelpMarker("When locked, camera orbits around target");

        if (pos_changed) {
            if (targetLock)
                ctx.scene.camera->moveToTargetLocked(pos);
            else {
                ctx.scene.camera->lookfrom = pos;
                ctx.scene.camera->origin = pos;
                ctx.scene.camera->update_camera_vectors();
            }
        }
        if (target_changed) {
            ctx.scene.camera->lookat = target;
            ctx.scene.camera->update_camera_vectors();
        }
        UIWidgets::EndSection();
    }

    // -------- View Parameters --------
    if (UIWidgets::BeginSection("View Parameters", ImVec4(0.5f, 0.8f, 0.5f, 1.0f))) {
        if (UIWidgets::SliderWithHelp("FOV", &fov, 10.0f, 120.0f, 
                                       "Field of View - lower values for telephoto, higher for wide angle")) {
            ctx.scene.camera->vfov = fov;
            ctx.scene.camera->fov = fov;
            ctx.scene.camera->update_camera_vectors();
        }
        UIWidgets::EndSection();
    }

    // -------- Depth of Field --------
    if (UIWidgets::BeginSection("Depth of Field & Bokeh", ImVec4(0.8f, 0.6f, 0.9f, 1.0f))) {
        // DOF Enable checkbox (if aperture is 0, DOF is effectively off anyway)
        static bool dof_enabled = false;
        if (ImGui::Checkbox("Enable DOF", &dof_enabled)) {
            if (!dof_enabled) {
                // Disable DOF by setting aperture to 0
                aperture = 0.0f;
                ctx.scene.camera->aperture = 0.0f;
                ctx.scene.camera->lens_radius = 0.0f;
            } else {
                // Enable DOF with a reasonable default aperture
                if (aperture < 0.01f) {
                    aperture = 0.1f;
                    ctx.scene.camera->aperture = 0.1f;
                    ctx.scene.camera->lens_radius = 0.05f;
                }
            }
            ctx.scene.camera->update_camera_vectors();
        }
        UIWidgets::HelpMarker("Enable/disable depth of field effect");
        
        // Sync dof_enabled with current aperture state
        dof_enabled = (aperture > 0.001f);
        
        if (dof_enabled) {
            bool aperture_changed = UIWidgets::SliderWithHelp("Aperture", &aperture, 0.01f, 5.0f,
                                                               "Lens aperture size - affects blur amount");
            
            // Focus Distance with Focus to Selection button
            bool focus_changed = UIWidgets::DragFloatWithHelp("Focus Distance", &focus_dist, 0.05f, 0.01f, 100.0f,
                                                               "Distance to sharp focus plane");
            
            // Focus to Selection button - sets focus to selected object's distance
            bool has_selection = ctx.selection.hasSelection();
            if (!has_selection) ImGui::BeginDisabled();
            
            if (ImGui::Button("Focus to Selection")) {
                Vec3 selection_pos = ctx.selection.selected.position;
                Vec3 cam_pos = ctx.scene.camera->lookfrom;
                float distance = (selection_pos - cam_pos).length();
                
                if (distance > 0.01f) {
                    ctx.scene.camera->focus_dist = distance;
                    focus_dist = distance;
                    ctx.scene.camera->update_camera_vectors();
                    SCENE_LOG_INFO("Focus distance set to: " + std::to_string(distance));
                    
                    // Update GPU if available
                    if (ctx.optix_gpu_ptr) {
                        ctx.optix_gpu_ptr->setCameraParams(*ctx.scene.camera);
                        ctx.optix_gpu_ptr->resetAccumulation();
                    }
                    ctx.renderer.resetCPUAccumulation();
                }
            }
            
            if (!has_selection) ImGui::EndDisabled();
            UIWidgets::HelpMarker("Select an object first, then click to set focus distance to that object");
            
            if (aperture_changed || focus_changed) {
                ctx.scene.camera->lens_radius = aperture * 0.5f;
                ctx.scene.camera->update_camera_vectors();
            }

            ImGui::SliderInt("Blade Count", &ctx.scene.camera->blade_count, 3, 12);
            UIWidgets::HelpMarker("Number of aperture blades - affects bokeh shape");
        } else {
            ImGui::TextDisabled("DOF disabled (Aperture = 0)");
        }
        
        UIWidgets::EndSection();
    }

    // -------- Mouse Control --------
    if (UIWidgets::BeginSection("Mouse Control", ImVec4(0.9f, 0.7f, 0.4f, 1.0f))) {
        ImGui::Checkbox("Enable Mouse Look", &ctx.mouse_control_enabled);
        UIWidgets::HelpMarker("Use mouse to rotate camera view");
        
        if (ctx.mouse_control_enabled) {
            UIWidgets::SliderWithHelp("Sensitivity", &ctx.mouse_sensitivity, 0.01f, 5.0f,
                                       "Mouse movement sensitivity", "%.3f");
        }
        UIWidgets::EndSection();
    }

    // -------- Actions --------
    UIWidgets::Divider();
    if (UIWidgets::SecondaryButton("Reset Camera", ImVec2(120, 0))) {
        ctx.scene.camera->reset();
        ctx.start_render = true; // Trigger re-render
        SCENE_LOG_INFO("Camera reset to initial state.");
    }
}

void SceneUI::drawLightsContent(UIContext& ctx)
{
    ImGui::TextColored(ImVec4(0.6f, 0.9f, 1.0f, 1), "Scene Lights");
    bool changed = false;

    for (size_t i = 0; i < ctx.scene.lights.size(); ++i) {
        auto light = ctx.scene.lights[i];
        if (!light) continue;

        std::string label = "Light #" + std::to_string(i);

        if (ImGui::TreeNode(label.c_str())) {
            const char* names[] = { "Point", "Directional", "Spot", "Area" };
            int index = (int)light->type();
            if (index >= 0 && index < 4)
                ImGui::Text("Type: %s", names[index]);
            else
                ImGui::Text("Type: Unknown");

            if (ImGui::DragFloat3("Position", &light->position.x, 0.1f)) changed = true;

            if (light->type() == LightType::Directional || light->type() == LightType::Spot)
                if (ImGui::DragFloat3("Direction", &light->direction.x, 0.1f)) changed = true;

            if (ImGui::ColorEdit3("Color", &light->color.x)) changed = true;
            if (ImGui::DragFloat("Intensity", &light->intensity, 0.1f, 0, 1000.0f)) changed = true;

            if (light->type() == LightType::Point ||
                light->type() == LightType::Area ||
                light->type() == LightType::Directional)
                if (ImGui::DragFloat("Radius", &light->radius, 0.01f, 0.01f, 100.0f)) changed = true;

            if (auto sl = std::dynamic_pointer_cast<SpotLight>(light)) {
                 float angle = sl->getAngleDegrees();
                 if (ImGui::DragFloat("Cone Angle", &angle, 0.5f, 1.0f, 89.0f)) {
                     sl->setAngleDegrees(angle);
                     changed = true;
                 }
                 float falloff = sl->getFalloff();
                 if (ImGui::SliderFloat("Falloff", &falloff, 0.0f, 1.0f)) {
                     sl->setFalloff(falloff);
                     changed = true;
                 }
            }
            else if (auto al = std::dynamic_pointer_cast<AreaLight>(light)) {
                 if (ImGui::DragFloat("Width", &al->width, 0.05f, 0.01f, 100.0f)) {
                     al->u = al->u.normalize() * al->width;
                     changed = true;
                 }
                 if (ImGui::DragFloat("Height", &al->height, 0.05f, 0.01f, 100.0f)) {
                     al->v = al->v.normalize() * al->height;
                     changed = true;
                 }
            }

            ImGui::TreePop();
        }
    }
    
    if (changed) {
        ctx.renderer.resetCPUAccumulation();
        if (ctx.optix_gpu_ptr) {
            ctx.optix_gpu_ptr->setLightParams(ctx.scene.lights);
            ctx.optix_gpu_ptr->resetAccumulation();
        }
    }
}
void SceneUI::drawWorldContent(UIContext& ctx) {
    World& world = ctx.renderer.world;
    WorldMode current_mode = world.getMode();
    
    UIWidgets::ColoredHeader("Environment Settings", ImVec4(0.3f, 0.7f, 1.0f, 1.0f));
    UIWidgets::Divider();
    
    bool changed = false;
    
    // ═══════════════════════════════════════════════════════════
    // Sky Model Selection
    // ═══════════════════════════════════════════════════════════
    if (UIWidgets::BeginSection("Sky Model", ImVec4(0.4f, 0.6f, 0.9f, 1.0f))) {
        const char* modes[] = { "Solid Color", "HDRI Environment", "Nishita Sky" };
        int mode_idx = static_cast<int>(current_mode);
        
        ImGui::PushItemWidth(-1);
        if (ImGui::Combo("##SkyModel", &mode_idx, modes, IM_ARRAYSIZE(modes))) {
            world.setMode(static_cast<WorldMode>(mode_idx));
            changed = true;
        }
        ImGui::PopItemWidth();
        UIWidgets::EndSection();
    }
    
    ImGui::Spacing();
    
    // ═══════════════════════════════════════════════════════════
    // Solid Color Mode
    // ═══════════════════════════════════════════════════════════
    if (current_mode == WORLD_MODE_COLOR) {
        if (UIWidgets::BeginSection("Background", ImVec4(0.6f, 0.4f, 0.8f, 1.0f))) {
            Vec3 color = world.getColor();
            if (ImGui::ColorEdit3("Color", &color.x)) {
                world.setColor(color);
                ctx.scene.background_color = color;
                changed = true;
            }
            
            float intensity = world.getColorIntensity();
            if (ImGui::SliderFloat("Intensity", &intensity, 0.0f, 10.0f)) {
                world.setColorIntensity(intensity);
                changed = true;
            }
            UIWidgets::EndSection();
        }
    }
    // ═══════════════════════════════════════════════════════════
    // HDRI Environment Mode
    // ═══════════════════════════════════════════════════════════
    else if (current_mode == WORLD_MODE_HDRI) {
        if (UIWidgets::BeginSection("HDRI Map", ImVec4(0.2f, 0.7f, 0.5f, 1.0f))) {
            // Load Button
            if (UIWidgets::PrimaryButton("Load Environment", ImVec2(-1, 0))) {
#ifdef _WIN32
                std::string file = openFileDialogW(L"Environment Maps\0*.hdr;*.exr;*.jpg;*.jpeg;*.png\0HDR/EXR\0*.hdr;*.exr\0All Files\0*.*\0");
                if (!file.empty()) {
                    world.setHDRI(file);
                    changed = true;
                }
#endif
            }
            
            // Current file display
            std::string path = world.getHDRIPath();
            if (!path.empty()) {
                size_t lastSlash = path.find_last_of("/\\");
                std::string filename = (lastSlash != std::string::npos) ? path.substr(lastSlash + 1) : path;
                ImGui::TextColored(ImVec4(0.5f, 0.8f, 0.5f, 1.0f), "%s", filename.c_str());
            } else {
                ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "(No HDRI loaded)");
            }
            UIWidgets::EndSection();
        }
        
        if (UIWidgets::BeginSection("Transform", ImVec4(0.5f, 0.6f, 0.8f, 1.0f))) {
            // Rotation - Use SliderFloat for better 360 control
            float rotation = world.getHDRIRotation();
            if (ImGui::SliderFloat("Rotation", &rotation, 0.0f, 360.0f, "%.1f deg")) {
                world.setHDRIRotation(rotation);
                changed = true;
            }
            UIWidgets::HelpMarker("Rotate the environment map around Y axis (0-360 degrees)");
            
            // Intensity
            float intensity = world.getHDRIIntensity();
            if (ImGui::SliderFloat("Intensity", &intensity, 0.0f, 10.0f, "%.2f")) {
                world.setHDRIIntensity(intensity);
                changed = true;
            }
            UIWidgets::HelpMarker("Brightness multiplier for the environment");
            UIWidgets::EndSection();
        }
    }
    // ═══════════════════════════════════════════════════════════
    // Nishita Sky Mode
    // ═══════════════════════════════════════════════════════════
    else if (current_mode == WORLD_MODE_NISHITA) {
        NishitaSkyParams params = world.getNishitaParams();
        static bool syncWithDirectionalLight = false;
        
        // Sync with Directional Light Section
        if (UIWidgets::BeginSection("Light Sync", ImVec4(0.6f, 0.8f, 1.0f, 1.0f))) {
            if (ImGui::Checkbox("Sync with Scene Light", &syncWithDirectionalLight)) {
                // Checkbox toggled
            }
            UIWidgets::HelpMarker("Automatically sync sun direction with the first directional light in the scene");
            
            if (syncWithDirectionalLight) {
                // Find first directional light
                bool foundDirLight = false;
                for (const auto& light : ctx.scene.lights) {
                    if (light && light->type() == LightType::Directional) {
                        // Convert light direction to sun direction
                        // Note: light->direction is the direction light TRAVELS (sun to ground)
                        // We need direction TO the sun, so negate it
                        Vec3 dir = -(light->direction.normalize());
                        
                        // Elevation: angle from horizon (Y component)
                        float elevation = asinf(dir.y) * 180.0f / M_PI;
                        
                        // Azimuth: horizontal angle (XZ plane)
                        float azimuth = atan2f(dir.x, dir.z) * 180.0f / M_PI;
                        if (azimuth < 0) azimuth += 360.0f;
                        
                        // Update params if different
                        if (fabsf(params.sun_elevation - elevation) > 0.1f || 
                            fabsf(params.sun_azimuth - azimuth) > 0.1f) {
                            params.sun_elevation = elevation;
                            params.sun_azimuth = azimuth;
                            changed = true;
                        }
                        
                        foundDirLight = true;
                        ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Synced with Directional Light");
                        break;
                    }
                }
                
                if (!foundDirLight) {
                    ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.3f, 1.0f), "No directional light found");
                }
            }
            UIWidgets::EndSection();
        }
        
        // Sun Position Section (controls both sun and directional light when synced)
        const char* sunPosTitle = syncWithDirectionalLight ? "Sun/Light Position" : "Sun Position";
        if (UIWidgets::BeginSection(sunPosTitle, ImVec4(1.0f, 0.7f, 0.3f, 1.0f))) {
            if (ImGui::SliderFloat("Elevation", &params.sun_elevation, -10.0f, 90.0f, "%.1f deg")) {
                changed = true;
            }
            UIWidgets::HelpMarker(syncWithDirectionalLight ? 
                "Sun/Light height above horizon - controls both sky sun and directional light" :
                "Sun height above horizon (0 = horizon, 90 = zenith)");
            
            if (ImGui::SliderFloat("Azimuth", &params.sun_azimuth, 0.0f, 360.0f, "%.1f deg")) {
                changed = true;
            }
            UIWidgets::HelpMarker(syncWithDirectionalLight ?
                "Sun/Light horizontal rotation - controls both sky sun and directional light" :
                "Sun horizontal rotation (compass direction)");
            UIWidgets::EndSection();
        }
        
        // Sun Appearance Section
        if (UIWidgets::BeginSection("Sun", ImVec4(1.0f, 0.5f, 0.3f, 1.0f))) {
            if (ImGui::SliderFloat("Intensity", &params.sun_intensity, 0.0f, 100.0f, "%.1f")) {
                changed = true;
            }
            UIWidgets::HelpMarker("Brightness of the sun");
            
            // Calculate automatic sun size based on elevation (atmospheric magnification)
            // Sun appears larger near horizon due to refraction
            float autoSunSize = params.sun_size;
            float elevationFactor = 1.0f;
            if (params.sun_elevation < 15.0f) {
                // Below 15 degrees, sun starts appearing larger
                // At 0 degrees, size is ~1.5x, at -5 degrees, ~2x
                elevationFactor = 1.0f + (15.0f - fmaxf(params.sun_elevation, -10.0f)) * 0.04f;
            }
            float displaySize = params.sun_size * elevationFactor;
            
            if (ImGui::SliderFloat("Size", &params.sun_size, 0.1f, 5.0f, "%.3f deg")) {
                changed = true;
            }
            if (elevationFactor > 1.01f) {
                ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.5f, 1.0f), "Effective: %.3f deg (horizon boost)", displaySize);
            }
            UIWidgets::HelpMarker("Angular diameter of the sun disc (real sun = 0.545°). Automatically increases near horizon.");
            UIWidgets::EndSection();
        }
        
        // Atmosphere Section (Blender-style parameters)
        if (UIWidgets::BeginSection("Atmosphere", ImVec4(0.4f, 0.7f, 1.0f, 1.0f))) {
            if (ImGui::SliderFloat("Air", &params.air_density, 0.0f, 10.0f, "%.2f")) {
                changed = true;
            }
            UIWidgets::HelpMarker("Density of air molecules (Rayleigh scattering). Higher = denser atmosphere, more color shift at horizon");
            
            if (ImGui::SliderFloat("Dust", &params.dust_density, 0.0f, 10.0f, "%.2f")) {
                changed = true;
            }
            UIWidgets::HelpMarker("Density of dust/aerosols (Mie scattering). Higher = more haze and sun glow");
            
            if (ImGui::SliderFloat("Ozone", &params.ozone_density, 0.0f, 10.0f, "%.2f")) {
                changed = true;
            }
            UIWidgets::HelpMarker("Ozone layer density. Higher = bluer sky, affects color saturation");
            
            // Altitude in UI is shown in km, but stored in meters
            float altitudeKm = params.altitude / 1000.0f;
            if (ImGui::SliderFloat("Altitude", &altitudeKm, 0.0f, 60.0f, "%.1f km")) {
                params.altitude = altitudeKm * 1000.0f;  // Convert back to meters
                changed = true;
            }
            UIWidgets::HelpMarker("Camera altitude above sea level. 0 = beach, 10 = airplane, 60 = edge of atmosphere");
            UIWidgets::EndSection();
        }
        
        // Advanced Physics (Collapsible)
        if (ImGui::CollapsingHeader("Advanced Physics")) {
            ImGui::Indent();
            
            if (ImGui::SliderFloat("Mie Anisotropy", &params.mie_anisotropy, 0.0f, 0.99f, "%.2f")) {
                changed = true;
            }
            UIWidgets::HelpMarker("Sun glow directionality (0 = uniform, 0.8+ = strong forward scatter)");
            
            ImGui::Unindent();
        }
        
        // Night Sky Section (Stars)
        if (UIWidgets::BeginSection("Night Sky", ImVec4(0.3f, 0.3f, 0.7f, 1.0f))) {
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 1.0f, 1.0f), "Stars (visible when sun is low)");
            
            if (ImGui::SliderFloat("Stars Intensity", &params.stars_intensity, 0.0f, 5.0f, "%.2f")) {
                changed = true;
            }
            UIWidgets::HelpMarker("Brightness of stars. Set to 0 to disable.");
            
            if (ImGui::SliderFloat("Stars Density", &params.stars_density, 0.0f, 1.0f, "%.2f")) {
                changed = true;
            }
            UIWidgets::HelpMarker("How many stars appear (0 = few, 1 = many)");
            UIWidgets::EndSection();
        }
        
        // Moon Section
        if (UIWidgets::BeginSection("Moon", ImVec4(0.8f, 0.8f, 0.5f, 1.0f))) {
            bool moonEnabled = params.moon_enabled != 0;
            if (ImGui::Checkbox("Show Moon", &moonEnabled)) {
                params.moon_enabled = moonEnabled ? 1 : 0;
                changed = true;
            }
            
            if (params.moon_enabled) {
                if (ImGui::SliderFloat("Moon Elevation", &params.moon_elevation, -10.0f, 90.0f, "%.1f deg")) {
                    changed = true;
                }
                
                if (ImGui::SliderFloat("Moon Azimuth", &params.moon_azimuth, 0.0f, 360.0f, "%.1f deg")) {
                    changed = true;
                }
                
                if (ImGui::SliderFloat("Moon Intensity", &params.moon_intensity, 0.0f, 5.0f, "%.2f")) {
                    changed = true;
                }
                
                if (ImGui::SliderFloat("Moon Size", &params.moon_size, 0.1f, 3.0f, "%.2f deg")) {
                    changed = true;
                }
                UIWidgets::HelpMarker("Real moon = 0.52 degrees");
                
                if (ImGui::SliderFloat("Moon Phase", &params.moon_phase, 0.0f, 1.0f, "%.2f")) {
                    changed = true;
                }
                UIWidgets::HelpMarker("0 = New Moon, 0.5 = Full Moon, 1 = New Moon");
            }
            UIWidgets::EndSection();
        }


        
        // Update sun direction from elevation/azimuth
        if (changed) {
            float elevationRad = params.sun_elevation * M_PI / 180.0f;
            float azimuthRad = params.sun_azimuth * M_PI / 180.0f;
            params.sun_direction = make_float3(
                cosf(elevationRad) * sinf(azimuthRad),
                sinf(elevationRad),
                cosf(elevationRad) * cosf(azimuthRad)
            );
            world.setNishitaParams(params);
            
            // If sync is enabled, also update the directional light
            if (syncWithDirectionalLight) {
                for (auto& light : ctx.scene.lights) {
                    if (light && light->type() == LightType::Directional) {
                        // Light direction is opposite of sun direction (sun to ground)
                        light->direction = Vec3(
                            -params.sun_direction.x,
                            -params.sun_direction.y,
                            -params.sun_direction.z
                        );
                        break;
                    }
                }
            }
        }
    }
    
    // ═══════════════════════════════════════════════════════════
    // Global Atmosphere (Clouds)
    // ═══════════════════════════════════════════════════════════
    ImGui::Spacing();
    if (UIWidgets::BeginSection("Global Atmosphere (Clouds)", ImVec4(0.5f, 0.6f, 0.7f, 1.0f))) {
        NishitaSkyParams cloudParams = world.getNishitaParams();
        bool cloudParamsChanged = false;

        bool cloudsEnabled = cloudParams.clouds_enabled != 0;
        if (ImGui::Checkbox("Enable Volumetric Clouds", &cloudsEnabled)) {
            cloudParams.clouds_enabled = cloudsEnabled ? 1 : 0;
            cloudParamsChanged = true;
        }
        UIWidgets::HelpMarker("Render volumetric clouds over any sky model.\nNote: High Performance Cost!");

        if (cloudParams.clouds_enabled) {
            ImGui::Spacing();
            
            // === CLOUD QUALITY (Performance) ===
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.4f, 1.0f), "Quality:");
            static int quality_preset = 0; // Normal
            const char* quality_presets[] = { "Fast Preview", "Low", "Normal", "High", "Ultra" };
            float quality_values[] = { 0.25f, 0.5f, 1.0f, 1.5f, 2.5f };
            
            ImGui::PushItemWidth(120);
            if (ImGui::Combo("##CloudQuality", &quality_preset, quality_presets, IM_ARRAYSIZE(quality_presets))) {
                cloudParams.cloud_quality = quality_values[quality_preset];
                cloudParamsChanged = true;
            }
            ImGui::PopItemWidth();
            ImGui::SameLine();
            ImGui::TextDisabled("(%.0f-%.0f steps)", 
                48.0f * cloudParams.cloud_quality, 
                96.0f * cloudParams.cloud_quality);
            UIWidgets::HelpMarker("Lower quality = faster render\nHigher quality = better details");
            
            ImGui::Spacing();
            
            // === WEATHER PRESETS (Combo Box) ===
            static int weather_preset_index = 7; // Default to Custom
            const char* weather_presets[] = { 
                "Clear Sky", 
                "Cirrus (Wispy)",
                "Cumulus (Puffy)",
                "Stratocumulus",
                "Overcast", 
                "Cumulonimbus (Storm)",
                "Fog/Low Clouds",
                "Custom" 
            };
            
            ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1.0f), "Weather Type:");
            ImGui::PushItemWidth(-1);
            if (ImGui::Combo("##WeatherType", &weather_preset_index, weather_presets, IM_ARRAYSIZE(weather_presets))) {
                switch (weather_preset_index) {
                    case 0: // Clear Sky
                        cloudParams.clouds_enabled = 0;
                        cloudParamsChanged = true;
                        break;
                    case 1: // Cirrus (Wispy high clouds) - VERY THIN layer
                        cloudParams.cloud_height_min = 8000.0f;   // 8km - high altitude
                        cloudParams.cloud_height_max = 8500.0f;   // Only 500m thick!
                        cloudParams.cloud_scale = 15.0f;          // Very stretched/wispy
                        cloudParams.cloud_coverage = 0.35f;       // Sparse
                        cloudParams.cloud_density = 0.15f;        // Very transparent
                        cloudParamsChanged = true;
                        break;
                    case 2: // Cumulus (Classic puffy clouds)
                        cloudParams.cloud_height_min = 1000.0f;
                        cloudParams.cloud_height_max = 3000.0f;
                        cloudParams.cloud_scale = 3.0f;
                        cloudParams.cloud_coverage = 0.4f;
                        cloudParams.cloud_density = 1.5f;
                        cloudParamsChanged = true;
                        break;
                    case 3: // Stratocumulus (Layered puffy)
                        cloudParams.cloud_height_min = 500.0f;
                        cloudParams.cloud_height_max = 2000.0f;
                        cloudParams.cloud_scale = 2.0f;
                        cloudParams.cloud_coverage = 0.88f;
                        cloudParams.cloud_density = 1.2f;
                        cloudParamsChanged = true;
                        break;
                    case 4: // Overcast (Stratus)
                        cloudParams.cloud_height_min = 200.0f;
                        cloudParams.cloud_height_max = 600.0f;
                        cloudParams.cloud_scale = 0.5f;
                        cloudParams.cloud_coverage = 0.9f;
                        cloudParams.cloud_density = 2.5f;
                        cloudParamsChanged = true;
                        break;
                    case 5: // Cumulonimbus (Storm/Rain clouds)
                        cloudParams.cloud_height_min = 500.0f;
                        cloudParams.cloud_height_max = 10000.0f;
                        cloudParams.cloud_scale = 5.0f;
                        cloudParams.cloud_coverage = 0.9f;
                        cloudParams.cloud_density = 4.0f;
                        cloudParamsChanged = true;
                        break;
                    case 6: // Fog/Low Clouds
                        cloudParams.cloud_height_min = 0.0f;
                        cloudParams.cloud_height_max = 200.0f;
                        cloudParams.cloud_scale = 0.3f;
                        cloudParams.cloud_coverage = 0.9f;
                        cloudParams.cloud_density = 3.0f;
                        cloudParamsChanged = true;
                        break;
                    case 7: // Custom
                        break;
                }
            }
            ImGui::PopItemWidth();
            UIWidgets::HelpMarker("Cloud type presets:\n"
                "- Cirrus: High wispy clouds\n"
                "- Cumulus: Classic puffy clouds\n"
                "- Stratocumulus: Layered clouds\n"
                "- Overcast: Gray flat sky\n"
                "- Cumulonimbus: Towering storm clouds\n"
                "- Fog: Ground-level clouds");
            
            ImGui::Spacing();
            ImGui::Separator();
            
            // Show detailed controls in a collapsible section or when Custom is selected
            bool show_details = (weather_preset_index == 7); // Custom mode
            
            if (show_details || ImGui::CollapsingHeader("Cloud Details")) {
                // Coverage
                if (ImGui::SliderFloat("Coverage", &cloudParams.cloud_coverage, 0.0f, 1.0f, "%.2f")) {
                    cloudParamsChanged = true;
                    weather_preset_index = 7; // Switch to Custom when manually editing
                }
                UIWidgets::HelpMarker("Cloud coverage (0 = clear sky, 1 = overcast)");

                // Density
                if (ImGui::DragFloat("Density", &cloudParams.cloud_density, 0.05f, 0.0f, 10.0f, "%.2f")) {
                    cloudParamsChanged = true;
                    weather_preset_index = 7;
                }
                UIWidgets::HelpMarker("Cloud opacity multiplier (higher = more opaque clouds)");
                
                // Scale
                if (ImGui::DragFloat("Scale", &cloudParams.cloud_scale, 0.1f, 0.1f, 100.0f, "%.2f")) {
                    cloudParamsChanged = true;
                    weather_preset_index = 7;
                }
                UIWidgets::HelpMarker("Cloud size.\nHigher = Larger clouds\nLower = Smaller, detailed clouds");

                // Wind / Seed Offsets
                if (ImGui::DragFloat("Offset X (Wind)", &cloudParams.cloud_offset_x, 10.0f, -100000.0f, 100000.0f, "%.0f m")) {
                    cloudParamsChanged = true;
                }
                if (ImGui::DragFloat("Offset Z (Wind)", &cloudParams.cloud_offset_z, 10.0f, -100000.0f, 100000.0f, "%.0f m")) {
                    cloudParamsChanged = true;
                }
                UIWidgets::HelpMarker("Shift cloud pattern for variety or wind animation");
            }
            
            // ═══════════════════════════════════════════════════════════
            // CLOUD LIGHTING SETTINGS
            // ═══════════════════════════════════════════════════════════
            if (ImGui::CollapsingHeader("Cloud Lighting")) {
                ImGui::TextColored(ImVec4(1.0f, 0.9f, 0.5f, 1.0f), "Self-Shadowing:");
                
                // Light Steps (0 = disabled for performance)
                if (ImGui::SliderInt("Light Steps", &cloudParams.cloud_light_steps, 0, 12)) {
                    cloudParamsChanged = true;
                }
                UIWidgets::HelpMarker("Number of light marching steps.\n0 = Disabled (fast)\n4-6 = Normal\n8-12 = High quality");
                
                // Shadow Strength
                if (ImGui::SliderFloat("Shadow Strength", &cloudParams.cloud_shadow_strength, 0.0f, 2.0f, "%.2f")) {
                    cloudParamsChanged = true;
                }
                UIWidgets::HelpMarker("Cloud self-shadowing intensity.\n0 = No shadows\n1 = Normal\n2 = Dark, dramatic shadows");
                
                ImGui::Spacing();
                ImGui::TextColored(ImVec4(1.0f, 0.9f, 0.5f, 1.0f), "Lighting Effects:");
                
                // Silver Lining
                if (ImGui::SliderFloat("Silver Lining", &cloudParams.cloud_silver_intensity, 0.0f, 3.0f, "%.2f")) {
                    cloudParamsChanged = true;
                }
                UIWidgets::HelpMarker("Bright rim effect when backlit by sun.\n0 = Off\n1 = Normal\n2+ = Strong glow");
                
                // Ambient Strength
                if (ImGui::SliderFloat("Ambient Light", &cloudParams.cloud_ambient_strength, 0.0f, 2.0f, "%.2f")) {
                    cloudParamsChanged = true;
                }
                UIWidgets::HelpMarker("Sky ambient light contribution.\nLower = Darker shadows\nHigher = Softer look");
                
                // Absorption
                if (ImGui::SliderFloat("Absorption", &cloudParams.cloud_absorption, 0.2f, 3.0f, "%.2f")) {
                    cloudParamsChanged = true;
                }
                UIWidgets::HelpMarker("Light absorption rate.\n0.5 = Thin, transparent\n1.0 = Normal\n2.0 = Thick, opaque");
            }

            // === LAYER 1 ALTITUDE ===
            if (ImGui::TreeNode("Layer 1 Altitude")) {
                ImGui::TextColored(ImVec4(0.5f, 0.8f, 0.5f, 1.0f), "Layer 1: %.0f - %.0f m (%.0f m thick)", 
                    cloudParams.cloud_height_min, cloudParams.cloud_height_max,
                    cloudParams.cloud_height_max - cloudParams.cloud_height_min);
                
                if (ImGui::DragFloat("Min Height##L1", &cloudParams.cloud_height_min, 10.0f, 0.0f, 20000.0f, "%.0f m")) {
                    cloudParamsChanged = true;
                }
                if (ImGui::DragFloat("Max Height##L1", &cloudParams.cloud_height_max, 10.0f, 0.0f, 20000.0f, "%.0f m")) {
                    cloudParamsChanged = true;
                }
                ImGui::TreePop();
            }
            
            ImGui::Spacing();
            ImGui::Separator();
            
            // ==================================================================
            // CLOUD LAYER 2 (Secondary/High Altitude)
            // ==================================================================
            bool layer2Enabled = cloudParams.cloud_layer2_enabled != 0;
            if (ImGui::Checkbox("Enable Cloud Layer 2", &layer2Enabled)) {
                cloudParams.cloud_layer2_enabled = layer2Enabled ? 1 : 0;
                cloudParamsChanged = true;
            }
            UIWidgets::HelpMarker("Enable a second cloud layer.\nUseful for high cirrus above low cumulus.");
            
            if (cloudParams.cloud_layer2_enabled) {
                ImGui::Indent();
                ImGui::TextColored(ImVec4(0.8f, 0.6f, 1.0f, 1.0f), "Layer 2 Settings:");
                
                // Layer 2 presets
                static int layer2_preset = 0;
                const char* layer2_presets[] = { "Custom", "High Cirrus", "Mid Stratus", "Low Fog" };
                if (ImGui::Combo("Layer 2 Preset", &layer2_preset, layer2_presets, IM_ARRAYSIZE(layer2_presets))) {
                    switch (layer2_preset) {
                        case 1: // High Cirrus
                            cloudParams.cloud2_height_min = 8000.0f;
                            cloudParams.cloud2_height_max = 9000.0f;
                            cloudParams.cloud2_scale = 12.0f;
                            cloudParams.cloud2_coverage = 0.25f;
                            cloudParams.cloud2_density = 0.2f;
                            cloudParamsChanged = true;
                            break;
                        case 2: // Mid Stratus
                            cloudParams.cloud2_height_min = 2500.0f;
                            cloudParams.cloud2_height_max = 3500.0f;
                            cloudParams.cloud2_scale = 3.0f;
                            cloudParams.cloud2_coverage = 0.5f;
                            cloudParams.cloud2_density = 0.8f;
                            cloudParamsChanged = true;
                            break;
                        case 3: // Low Fog
                            cloudParams.cloud2_height_min = 0.0f;
                            cloudParams.cloud2_height_max = 150.0f;
                            cloudParams.cloud2_scale = 0.5f;
                            cloudParams.cloud2_coverage = 0.7f;
                            cloudParams.cloud2_density = 2.0f;
                            cloudParamsChanged = true;
                            break;
                    }
                }
                
                if (ImGui::SliderFloat("Coverage##L2", &cloudParams.cloud2_coverage, 0.0f, 1.0f, "%.2f")) {
                    cloudParamsChanged = true;
                    layer2_preset = 0;
                }
                if (ImGui::DragFloat("Density##L2", &cloudParams.cloud2_density, 0.05f, 0.0f, 5.0f, "%.2f")) {
                    cloudParamsChanged = true;
                    layer2_preset = 0;
                }
                if (ImGui::DragFloat("Scale##L2", &cloudParams.cloud2_scale, 0.1f, 0.1f, 50.0f, "%.2f")) {
                    cloudParamsChanged = true;
                    layer2_preset = 0;
                }
                
                ImGui::TextColored(ImVec4(0.8f, 0.6f, 1.0f, 1.0f), "Layer 2: %.0f - %.0f m", 
                    cloudParams.cloud2_height_min, cloudParams.cloud2_height_max);
                if (ImGui::DragFloat("Min Height##L2", &cloudParams.cloud2_height_min, 50.0f, 0.0f, 20000.0f, "%.0f m")) {
                    cloudParamsChanged = true;
                    layer2_preset = 0;
                }
                if (ImGui::DragFloat("Max Height##L2", &cloudParams.cloud2_height_max, 50.0f, 0.0f, 20000.0f, "%.0f m")) {
                    cloudParamsChanged = true;
                    layer2_preset = 0;
                }
                
                ImGui::Unindent();
            }
            
            ImGui::Spacing();
            
            // Camera altitude info
            if (ImGui::TreeNode("Camera Altitude Info")) {
                ImGui::TextColored(ImVec4(0.9f, 0.8f, 0.5f, 1.0f), "Camera Y: %.0f m", cloudParams.altitude);
                
                if (cloudParams.altitude >= cloudParams.cloud_height_min && 
                    cloudParams.altitude <= cloudParams.cloud_height_max) {
                    ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "Camera is INSIDE Layer 1!");
                }
              
                
                ImGui::TreePop();
            }
        }
        UIWidgets::EndSection();

        if (cloudParamsChanged) {
            world.setNishitaParams(cloudParams);
            changed = true;
        }
    }

    // ═══════════════════════════════════════════════════════════
    // Apply Changes
    // ═══════════════════════════════════════════════════════════
    if (changed) {
        ctx.renderer.resetCPUAccumulation();
        if (ctx.optix_gpu_ptr) {
            ctx.optix_gpu_ptr->setWorld(world.getGPUData());
            ctx.optix_gpu_ptr->resetAccumulation();
        }
    }
}


void SceneUI::drawRenderSettingsPanel(UIContext& ctx, float screen_y)
{
    // Dinamik yükseklik hesabı
    extern bool show_animation_panel; 
    // show_scene_log is member
    
    bool bottom_visible = show_animation_panel || show_scene_log;
    float bottom_margin = bottom_visible ? (200.0f + 24.0f) : 24.0f; // Panel(200) + StatusBar(24)

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

    // Remove NoResize flag
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus;

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
                        
                        ImGui::SameLine();
                        UIWidgets::HelpMarker("Select the acceleration structure backend for CPU rendering.\nEmbree is faster for complex scenes but requires rebuilding.");
                        
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
                        
                        ImGui::DragInt("Max Samples (Preview)", &ctx.render_settings.max_samples, 1, 1, 16384);
                        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Limit for viewport accumulation. Set high for continuous refinement.");

                        ImGui::Checkbox("Viewport Denoising", &ctx.render_settings.use_denoiser);
                        if (ctx.render_settings.use_denoiser) {
                            ImGui::Indent();
                            ImGui::SliderFloat("Blend", &ctx.render_settings.denoiser_blend_factor, 0.0f, 1.0f);
                            ImGui::Unindent();
                        }
                        ImGui::EndTabItem();
                    }

                    // --- FINAL RENDER TAB ---
                    if (ImGui::BeginTabItem("Final Render")) {
                        ImGui::TextDisabled("Output Settings");
                        ImGui::Separator();
                        
                        // Resolution Controls
                        ImGui::Text("Resolution:");
                        
                        const char* res_items[] = { "1280x720 (HD)", "1920x1080 (FHD)", "2560x1440 (2K)", "3840x2160 (4K)", "Custom" };
                        static int current_res_item = 1; // Default 1080p
                        
                        if (ImGui::Combo("Preset", &current_res_item, res_items, IM_ARRAYSIZE(res_items))) {
                            if (current_res_item == 0) { ctx.render_settings.final_render_width = 1280; ctx.render_settings.final_render_height = 720; }
                            if (current_res_item == 1) { ctx.render_settings.final_render_width = 1920; ctx.render_settings.final_render_height = 1080; }
                            if (current_res_item == 2) { ctx.render_settings.final_render_width = 2560; ctx.render_settings.final_render_height = 1440; }
                            if (current_res_item == 3) { ctx.render_settings.final_render_width = 3840; ctx.render_settings.final_render_height = 2160; }
                        }

                        ImGui::PushItemWidth(80);
                        if (ImGui::InputInt("W", &ctx.render_settings.final_render_width)) current_res_item = 4;
                        ImGui::SameLine(); 
                        if (ImGui::InputInt("H", &ctx.render_settings.final_render_height)) current_res_item = 4;
                        ImGui::PopItemWidth();
                        
                        ImGui::TextDisabled("(Viewport size is unaffected until render starts)");

                        ImGui::Separator();
                        
                        ImGui::Text("Quality:");
                        ImGui::DragInt("Target Samples", &ctx.render_settings.final_render_samples, 16, 16, 65536);
                        ImGui::Checkbox("Render Denoising", &ctx.render_settings.render_use_denoiser);
                        
                        ImGui::Spacing();
                        ImGui::Separator();
                        if (ImGui::Button("Start Final Render (F12)", ImVec2(-1, 30))) {
                             extern bool show_render_window;
                             show_render_window = true;
                             ctx.render_settings.is_final_render_mode = true; // IMPORTANT
                             ctx.start_render = true;
                        }

                        ImGui::Spacing();
                        ImGui::Separator();
                        ImGui::Text("Animation:");
                        
                        ImGui::PushItemWidth(70);
                        ImGui::InputInt("Start##anim", &ctx.render_settings.animation_start_frame);
                        ImGui::SameLine();
                        ImGui::InputInt("End##anim",   &ctx.render_settings.animation_end_frame);
                        ImGui::SameLine();
                        ImGui::InputInt("FPS##anim",   &ctx.render_settings.animation_fps);
                        ImGui::PopItemWidth();
                        
                        if (ImGui::Button("Render Animation Sequence", ImVec2(-1, 30))) {
                             ctx.render_settings.start_animation_render = true;
                        }
                        ImGui::EndTabItem();
                    }
                    
                    ImGui::EndTabBar();
                }

                // 3. Global Settings (Path Tracing)
                ImGui::Spacing();
                if (ImGui::TreeNodeEx("Light Paths & Optimization", ImGuiTreeNodeFlags_DefaultOpen)) {
                    ImGui::DragInt("Max Bounces", &ctx.render_settings.max_bounces, 1, 1, 32);
                    
                    ImGui::Checkbox("Adaptive Sampling", &ctx.render_settings.use_adaptive_sampling);
                    if (ctx.render_settings.use_adaptive_sampling) {
                        ImGui::Indent();
                        ImGui::DragFloat("Noise Threshold", &ctx.render_settings.variance_threshold, 0.001f, 0.0001f, 1.0f, "%.4f");
                        ImGui::DragInt("Min Samples", &ctx.render_settings.min_samples, 1, 1, 64);
                        ImGui::Unindent();
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
                ImGui::Separator();
                drawLightsContent(ctx);
                ImGui::EndTabItem();
            }

            // -------------------------------------------------------------
            // TAB: CAMERA (Camera params, PostFX)
            // -------------------------------------------------------------
            if (ImGui::BeginTabItem("Camera")) {
                drawCameraContent(ctx);
                ImGui::Spacing();
                drawToneMapContent(ctx);
                ImGui::EndTabItem();
            }

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
                ImGui::EndTabItem();
            }

            // -------------------------------------------------------------
            // TAB: SYSTEM (App settings)
            // -------------------------------------------------------------
            if (ImGui::BeginTabItem("System")) {
                drawThemeSelector();
                drawResolutionPanel();
                
                ImGui::Spacing();
                ImGui::Separator();
                ImGui::Checkbox("Show Log Window", &show_scene_log);
                extern bool show_animation_panel;
                ImGui::Checkbox("Show Timeline Panel", &show_animation_panel);
                
                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }
    }
    ImGui::End();
}

// Main Menu Bar implementation moved to separate file: scene_ui_menu.hpp check end of file

#include "scene_ui_menu.hpp"

void SceneUI::draw(UIContext& ctx)
{
    // 1. Draw Main Menu Bar (Top)
    drawMainMenuBar(ctx);

    ImGuiIO& io = ImGui::GetIO();
    float screen_y = io.DisplaySize.y;
    float screen_x = io.DisplaySize.x;
    
    // =========================================================================
    // INPUT HANDLING - Delete Selected Object/Light (Del or X key)
    // =========================================================================
    if (!io.WantCaptureKeyboard && ctx.selection.hasSelection()) {
        bool delete_pressed = ImGui::IsKeyPressed(ImGuiKey_Delete) || ImGui::IsKeyPressed(ImGuiKey_X);
        
        if (delete_pressed) {
            bool deleted = false;
            
            // Handle Light deletion
            if (ctx.selection.selected.type == SelectableType::Light && ctx.selection.selected.light) {
                auto light_to_delete = ctx.selection.selected.light;
                auto& lights = ctx.scene.lights;
                auto it = std::find(lights.begin(), lights.end(), light_to_delete);
                if (it != lights.end()) {
                    history.record(std::make_unique<DeleteLightCommand>(light_to_delete));
                    lights.erase(it);
                    deleted = true;
                    
                    // Update GPU
                    if (ctx.optix_gpu_ptr) {
                        ctx.optix_gpu_ptr->setLightParams(ctx.scene.lights);
                        ctx.optix_gpu_ptr->resetAccumulation();
                    }
                    ctx.renderer.resetCPUAccumulation();
                    
                    SCENE_LOG_INFO("Deleted Light");
                }
            }
            // Handle Object deletion
            else if (ctx.selection.selected.type == SelectableType::Object && ctx.selection.selected.object) {
                std::string deleted_name = ctx.selection.selected.name;
                
                // Find and remove the object(s) from scene
                auto& objects = ctx.scene.world.objects;
                size_t removed_count = 0;
                
                objects.erase(
                    std::remove_if(objects.begin(), objects.end(),
                        [&deleted_name, &removed_count](const std::shared_ptr<Hittable>& obj) {
                            auto tri = std::dynamic_pointer_cast<Triangle>(obj);
                            if (tri && tri->nodeName == deleted_name) {
                                removed_count++;
                                return true;
                            }
                            return false;
                        }),
                    objects.end());
                
                if (removed_count > 0) {
                    deleted = true;
                    
                    // Track deletion in project data
                    auto& proj = g_ProjectManager.getProjectData();
                    
                    // Find which imported model this object belongs to
                    for (auto& model : proj.imported_models) {
                        for (const auto& inst : model.objects) {
                            if (inst.node_name == deleted_name) {
                                // Add to deleted list if not already there
                                if (std::find(model.deleted_objects.begin(), model.deleted_objects.end(), deleted_name) 
                                    == model.deleted_objects.end()) {
                                    model.deleted_objects.push_back(deleted_name);
                                }
                                break;
                            }
                        }
                    }
                    
                    // Remove from procedural objects if applicable
                    auto& procs = proj.procedural_objects;
                    procs.erase(
                        std::remove_if(procs.begin(), procs.end(),
                            [&deleted_name](const ProceduralObjectData& p) {
                                return p.display_name == deleted_name;
                            }),
                        procs.end());
                    
                    // Mark project as modified
                    g_ProjectManager.markModified();
                    
                    // Rebuild caches and acceleration structures
                    rebuildMeshCache(ctx.scene.world.objects);
                    ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
                    ctx.renderer.resetCPUAccumulation();
                    if (ctx.optix_gpu_ptr) {
                        ctx.renderer.rebuildOptiXGeometry(ctx.scene, ctx.optix_gpu_ptr);
                        ctx.optix_gpu_ptr->resetAccumulation();
                    }
                    
                    SCENE_LOG_INFO("Deleted: " + deleted_name + " (" + std::to_string(removed_count) + " triangles)");
                }
            }
            
            if (deleted) {
                ctx.selection.clearSelection();
            }
        }
    }
    
    // Style adjustments for panels
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 4.0f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.1f, 0.1f, 0.13f, panel_alpha));

    // 2. Draw Side Panel (LEFT)
    float layout_left_offset = 0.0f;
    if (showSidePanel) {
        drawRenderSettingsPanel(ctx, screen_y);
        layout_left_offset = side_panel_width; // Updated by resize in drawRenderSettingsPanel
    }

    // ---------------------------------------------------------
    // STATUS BAR (Bottom Strip)
    // ---------------------------------------------------------
    float status_bar_height = 24.0f;
    
    ImGui::SetNextWindowPos(ImVec2(0, screen_y - status_bar_height));
    ImGui::SetNextWindowSize(ImVec2(screen_x, status_bar_height));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.08f, 0.08f, 0.08f, 1.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(4, 2));
    
    if (ImGui::Begin("StatusBar", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoBringToFrontOnFocus)) {
        
        // Toggle Buttons
        extern bool show_animation_panel;
        
        // Timeline Toggle
        bool anim_active = show_animation_panel;
        if (UIWidgets::StateButton(anim_active ? "[Timeline]" : "Timeline", anim_active, 
            ImVec4(0.3f, 0.6f, 1.0f, 1.0f), ImVec4(0.2f, 0.2f, 0.2f, 1.0f), ImVec2(80, 20))) {
            show_animation_panel = !show_animation_panel;
            // Auto switch tab if opening
            if (show_animation_panel) show_scene_log = false; // Optional: Single mode? Or just focus tab logic implies we might want logic.
            // Let's keep them separate but maybe auto-hide log if timeline opens? 
            // User requested "Minimize to Icon", so they toggle validity.
        }
        
        ImGui::SameLine();
        
        // Console Toggle
        bool log_active = show_scene_log;
        if (UIWidgets::StateButton(log_active ? "[Console]" : "Console", log_active, 
            ImVec4(0.3f, 0.6f, 1.0f, 1.0f), ImVec4(0.2f, 0.2f, 0.2f, 1.0f), ImVec2(80, 20))) {
            show_scene_log = !show_scene_log;
            if (show_scene_log) show_animation_panel = false; // Toggle behavior (one at a time preferable for bottom panel)
        }
        
        ImGui::SameLine();
        ImGui::TextDisabled("|");
        ImGui::SameLine();
        
        // Status Text
        if (ctx.scene.initialized) {
            ImGui::Text("Scene: %d Objects, %d Lights", (int)ctx.scene.world.objects.size(), (int)ctx.scene.lights.size());
        } else {
            ImGui::Text("Ready");
        }
        
        // Right side: Progress
        if (rendering_in_progress) {
             float p = ctx.render_settings.render_progress * 100.0f;
             std::string prog = "Rendering: " + std::to_string((int)p) + "%";
             float w = ImGui::CalcTextSize(prog.c_str()).x;
             ImGui::SameLine(screen_x - w - 20);
             ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "%s", prog.c_str());
        }

    }
    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(2);

    // ---------------------------------------------------------
    // BOTTOM PANEL (Timeline / Console)
    // ---------------------------------------------------------
    // Render ONLY if visible
    extern bool show_animation_panel;
    bool show_bottom = (show_animation_panel || show_scene_log);
    float bottom_height = 120.0f;
    
    // Safety: If both are false, show_bottom is false, no panel drawn.
    // If one is true, panel drawn above status bar.
    
    if (show_bottom) {
        ImGui::SetNextWindowPos(ImVec2(layout_left_offset, screen_y - bottom_height - status_bar_height), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(screen_x - layout_left_offset, bottom_height), ImGuiCond_Always);
        
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.12f, 0.12f, 0.15f, 1.0f)); // Slightly opaque
        if (ImGui::Begin("BottomPanel", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse)) {
            
            if (ImGui::BeginTabBar("BottomTabs")) {
                
                // Tab 1: Timeline
                // Only show tab if enabled (or always show logic? "Toggle" usually implies visibility)
                // If we want "One Bottom Panel" that has tabs, and buttons toggle the PANEL, then we show tabs.
                // But the user buttons were "Timeline" and "Console".
                // If I click "Timeline", I expect Timeline tab to be active.
                
                // Logic: If show_animation_panel is TRUE, we show the tab.
                // NOTE: We forced single-toggle in the status bar (clicking one closes other).
                // So effectively this is a context switcher.
                
                if (show_animation_panel) {
                     if (ImGui::BeginTabItem("Timeline", &show_animation_panel)) { 
                        drawTimelineContent(ctx);
                        ImGui::EndTabItem();
                    }
                }
                
                if (show_scene_log) {
                    if (ImGui::BeginTabItem("Console", &show_scene_log)) {
                        drawLogPanelEmbedded();
                        ImGui::EndTabItem();
                    }
                }
                
                ImGui::EndTabBar();
            }
        }
        ImGui::End();
        ImGui::PopStyleColor();
    }
    
    // Grid Drawing removed (Moved to Renderer for true 3D depth)


    // Light Gizmos & Selection (Drawn here to handle selection priority)
    bool gizmo_hit = false;
    if (ctx.scene.camera && ctx.selection.show_gizmo) {
        ImDrawList* draw_list = ImGui::GetBackgroundDrawList();
        ImGuiIO& io = ImGui::GetIO();
        Camera& cam = *ctx.scene.camera;
        
        // Camera Projection Setup
        Vec3 cam_forward = (cam.lookat - cam.lookfrom).normalize();
        Vec3 cam_right = cam_forward.cross(cam.vup).normalize();
        Vec3 cam_up = cam_right.cross(cam_forward).normalize();
        float fov_rad = cam.vfov * 3.14159265359f / 180.0f;
        float tan_half_fov = tanf(fov_rad * 0.5f);
        float aspect = io.DisplaySize.x / io.DisplaySize.y; 

        auto Project = [&](const Vec3& p) -> ImVec2 {
            Vec3 to_point = p - cam.lookfrom;
            float depth = to_point.dot(cam_forward);
            if (depth <= 0.1f) return ImVec2(-10000, -10000); 

            float local_x = to_point.dot(cam_right);
            float local_y = to_point.dot(cam_up);
            float half_height = depth * tan_half_fov;
            float half_width = half_height * aspect;
            
            return ImVec2(
                ((local_x / half_width) * 0.5f + 0.5f) * io.DisplaySize.x,
                (0.5f - (local_y / half_height) * 0.5f) * io.DisplaySize.y
            );
        };
        
        auto IsOnScreen = [](const ImVec2& v) { return v.x > -5000; };

        for (auto& light : ctx.scene.lights) {
            bool is_selected = (ctx.selection.selected.type == SelectableType::Light && ctx.selection.selected.light == light);
            ImU32 col = IM_COL32(255, 255, 100, 180);
            if (is_selected) col = IM_COL32(255, 100, 50, 255); 

            Vec3 pos = light->position;
            ImVec2 center = Project(pos);
            bool visible = IsOnScreen(center);
            
            if (visible) {
                 // Mouse Interaction (Priority Selection)
                 // Radius increased to 20px for easier selection
                 float d = sqrtf(powf(io.MousePos.x - center.x, 2) + powf(io.MousePos.y - center.y, 2));
                 
                 // Note: We check WantCaptureMouse usually to avoid clicking through UI windows, 
                 // but since Gizmos are overlays, we want to click them. 
                 // Typically if hovering a button, WantCaptureMouse is true.
                 // We only want to select if NOT hovering other interactive ImGui elements.
                 
                 if (d < 20.0f && ImGui::IsMouseClicked(0) && !ImGuizmo::IsOver()) {
                      ctx.selection.selectLight(light);
                      gizmo_hit = true;
                 }
                 
                 // Draw Name if Selected
                 if (is_selected) {
                     std::string label = light->nodeName.empty() ? "Light" : light->nodeName;
                     draw_list->AddText(ImVec2(center.x + 12, center.y - 12), col, label.c_str());
                 }
            }

            if (light->type() == LightType::Point) {
                // Point: Smaller Diamond + Emissive Core
                float r = 0.2f; // Smaller radius
                Vec3 pts[6] = {
                    pos + Vec3(0, r, 0), pos + Vec3(0, -r, 0),
                    pos + Vec3(r, 0, 0), pos + Vec3(-r, 0, 0),
                    pos + Vec3(0, 0, r), pos + Vec3(0, 0, -r)
                };
                ImVec2 s_pts[6];
                for(int i=0; i<6; ++i) s_pts[i] = Project(pts[i]);

                if (visible) {
                    draw_list->AddCircleFilled(center, 4.0f, IM_COL32(255, 255, 200, 200)); // Emissive Core
                    // Wireframe
                    draw_list->AddLine(s_pts[2], s_pts[4], col); draw_list->AddLine(s_pts[4], s_pts[3], col);
                    draw_list->AddLine(s_pts[3], s_pts[5], col); draw_list->AddLine(s_pts[5], s_pts[2], col);
                    draw_list->AddLine(s_pts[0], s_pts[2], col); draw_list->AddLine(s_pts[0], s_pts[3], col);
                    draw_list->AddLine(s_pts[0], s_pts[4], col); draw_list->AddLine(s_pts[0], s_pts[5], col);
                    draw_list->AddLine(s_pts[1], s_pts[2], col); draw_list->AddLine(s_pts[1], s_pts[3], col);
                    draw_list->AddLine(s_pts[1], s_pts[4], col); draw_list->AddLine(s_pts[1], s_pts[5], col);
                }
            }
            else if (light->type() == LightType::Directional) {
                // Directional: Sun symbol + Arrow
                if (visible) {
                    draw_list->AddCircle(center, 8.0f, col, 0, 2.0f); // Sun Body
                    // Rays
                    for(int i=0; i<8; ++i) {
                         float angle = i * (6.28f/8.0f);
                         ImVec2 dir(cosf(angle), sinf(angle));
                         draw_list->AddLine(
                            ImVec2(center.x + dir.x*12, center.y + dir.y*12),
                            ImVec2(center.x + dir.x*18, center.y + dir.y*18),
                            col
                         );
                    }
                    // Direction Arrow
                    auto dl = std::dynamic_pointer_cast<DirectionalLight>(light);
                    if (dl) {
                         Vec3 end3d = pos + dl->direction.normalize() * 3.0f; // Long arrow
                         ImVec2 end = Project(end3d);
                         if (IsOnScreen(end)) {
                            draw_list->AddLine(center, end, col, 2.0f);
                            draw_list->AddCircleFilled(end, 3.0f, col); // Arrow tip
                         }
                    }
                }
            }
            // Keep Area/Spot simple or similar to before, strictly projection for now
            else if (light->type() == LightType::Area && visible) {
                 // Area: Rectangle Wireframe
                 auto al = std::dynamic_pointer_cast<AreaLight>(light);
                 if (al) {
                     Vec3 u = al->getU();
                     Vec3 v = al->getV();
                     // 4 corners
                     Vec3 c1 = pos;
                     Vec3 c2 = pos + u;
                     Vec3 c3 = pos + u + v;
                     Vec3 c4 = pos + v;
                     
                     ImVec2 sc1 = Project(c1);
                     ImVec2 sc2 = Project(c2);
                     ImVec2 sc3 = Project(c3);
                     ImVec2 sc4 = Project(c4);
                     
                     draw_list->AddLine(sc1, sc2, col);
                     draw_list->AddLine(sc2, sc3, col);
                     draw_list->AddLine(sc3, sc4, col);
                     draw_list->AddLine(sc4, sc1, col);
                     // Diagonal for visibility
                     draw_list->AddLine(sc1, sc3, col, 1.0f);
                 } else {
                     draw_list->AddText(center, col, "[#]");
                 }
            }
            else if (light->type() == LightType::Spot && visible) {
                 // Spot: Cone Wireframe
                 auto sl = std::dynamic_pointer_cast<SpotLight>(light);
                 if (sl) {
                     Vec3 dir = sl->direction.normalize();
                     float len = 3.0f; // Cone length
                     float radius = len * tanf(sl->getAngleDegrees() * 3.14159f / 360.0f); // Half angle in radians
                     
                     // Base center
                     Vec3 base_center = pos + dir * len;
                     
                     // Create basis for base circle
                     Vec3 right = (std::abs(dir.y) > 0.9f) ? Vec3(1,0,0) : dir.cross(Vec3(0,1,0)).normalize();
                     Vec3 up = right.cross(dir).normalize();
                     
                     // Draw circle base + lines to tip
                     const int segs = 12;
                     ImVec2 last_p;
                     for (int i=0; i<=segs; ++i) {
                         float ang = i * (6.28f / segs);
                         Vec3 p = base_center + right * (cosf(ang) * radius) + up * (sinf(ang) * radius);
                         ImVec2 sp = Project(p);
                         
                         // Line to tip
                         if (i < segs && IsOnScreen(sp)) draw_list->AddLine(center, sp, col);
                         
                         // Circle edge
                         if (i > 0 && IsOnScreen(sp) && IsOnScreen(last_p)) draw_list->AddLine(last_p, sp, col);
                         last_p = sp;
                     }
                 } else {
                     draw_list->AddText(center, col, "\\/");
                 }
            }
        }
    }

    // Draw Camera Gizmos (for multi-camera scenes)
    drawCameraGizmos(ctx);

    // Draw Selection Bounding Box and Transform Gizmo (works on ALL tabs now)
    if (ctx.selection.hasSelection() && ctx.selection.show_gizmo && ctx.scene.camera) {
        drawSelectionBoundingBox(ctx);
        drawTransformGizmo(ctx);
    }

    // Scene Interaction (Picking) - works on ALL tabs
    if (ctx.scene.initialized && !gizmo_hit) {
        handleMouseSelection(ctx);
    }

    ImGui::PopStyleColor();
    ImGui::PopStyleVar();

    // Help / Controls Window
    if (show_controls_window) {
        ImGui::SetNextWindowSize(ImVec2(500, 600), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Controls & Help", &show_controls_window)) {
            drawControlsContent();
        }
        ImGui::End();
    }
    
    // Lazy BVH / Scene Update (Fixes CPU Ghosting after Delete)
    if (is_bvh_dirty) {
         // SCENE_LOG_INFO("Updating Scene BVH (Dirty Flag)...");
         ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
         ctx.renderer.resetCPUAccumulation();
         is_bvh_dirty = false;
    }
    
    // F12 Global Shortcut for Render
    if (ImGui::IsKeyPressed(ImGuiKey_F12)) {
         extern bool show_render_window;
         show_render_window = !show_render_window;
         if (show_render_window) ctx.start_render = true;
    }
    
    // Ctrl+Z / Ctrl+Y - Undo/Redo
    if (ImGui::IsKeyPressed(ImGuiKey_Z) && ImGui::GetIO().KeyCtrl && !ImGui::GetIO().KeyShift) {
        if (history.canUndo()) {
            history.undo(ctx);
            rebuildMeshCache(ctx.scene.world.objects);  // Refresh UI cache
            mesh_cache_valid = false;
            ctx.selection.updatePositionFromSelection(); // Sync Gizmo center
            ctx.selection.selected.has_cached_aabb = false; // Force bounding box rebuild
        }
    }
    if ((ImGui::IsKeyPressed(ImGuiKey_Y) && ImGui::GetIO().KeyCtrl) ||
        (ImGui::IsKeyPressed(ImGuiKey_Z) && ImGui::GetIO().KeyCtrl && ImGui::GetIO().KeyShift)) {
        if (history.canRedo()) {
            history.redo(ctx);
            rebuildMeshCache(ctx.scene.world.objects);  // Refresh UI cache
            mesh_cache_valid = false;
            ctx.selection.updatePositionFromSelection(); // Sync Gizmo center
            ctx.selection.selected.has_cached_aabb = false; // Force bounding box rebuild
        }
    }
    
    // Draw Final Render Window
    extern void DrawRenderWindow(UIContext& ctx);
    DrawRenderWindow(ctx);
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
     ImGui::BulletText("Save Image: Ctrl + S"); 
     ImGui::BulletText("Undo: Ctrl + Z");
     ImGui::BulletText("Redo: Ctrl + Y or Ctrl + Shift + Z");
     ImGui::BulletText("Delete Object: Delete or X");
     ImGui::BulletText("Duplicate Object: Shift + D");
     ImGui::BulletText("Toggle Render Window: F12");
     
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
     
     if (ImGui::CollapsingHeader("Camera Panel")) {
         ImGui::BulletText("FOV: Field of View (Zoom level).");
         ImGui::BulletText("Aperture: Controls Depth of Field (Blur amount).");
         ImGui::BulletText("Focus Dist: Distance to the sharpest point.");
         ImGui::BulletText("Reset Camera: Restores the initial view.");
     }
     
     if (ImGui::CollapsingHeader("Lights Panel")) {
         ImGui::BulletText("Lists all light sources in the scene.");
         ImGui::BulletText("Allows adjusting Intensity (Brightness) and Color.");
         ImGui::BulletText("Supports Point, Directional, Area, and Spot lights.");
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

// ═══════════════════════════════════════════════════════════════════════════════
// SCENE HIERARCHY PANEL (Outliner)
// ═══════════════════════════════════════════════════════════════════════════════
void SceneUI::drawSceneHierarchy(UIContext& ctx) {
    // Window creation logic removed - now embedded in tabs
    
    // Check if embedded in another window, if not create child
    // Since we are moving to tab, we don't need window creation here

    
    SceneSelection& sel = ctx.selection;
    
    // ─────────────────────────────────────────────────────────────────────────
    // DELETE LOGIC (Keyboard Shortcut)
    // ─────────────────────────────────────────────────────────────────────────
    // Only process when viewport has focus (not UI panels)
    if ((ImGui::IsKeyPressed(ImGuiKey_Delete) || ImGui::IsKeyPressed(ImGuiKey_X)) && 
        sel.hasSelection() && !ImGui::GetIO().WantCaptureKeyboard) {
        bool deleted = false;
        if (sel.selected.type == SelectableType::Object && sel.selected.object) {
            auto& objs = ctx.scene.world.objects;
            // Remove all triangles with the same node name (Group deletion)
             std::string targetName = sel.selected.object->nodeName;
             
             // STEP 1: Collect objects to delete (for undo)
             std::vector<std::shared_ptr<Triangle>> deleted_triangles;
             for (const auto& h : objs) {
                 auto t = std::dynamic_pointer_cast<Triangle>(h);
                 if (t && t->nodeName == targetName) {
                     deleted_triangles.push_back(t);
                 }
             }
             
             // STEP 2: Remove from scene
             auto new_end = std::remove_if(objs.begin(), objs.end(), [&](const std::shared_ptr<Hittable>& h){
                 auto t = std::dynamic_pointer_cast<Triangle>(h);
                 return t && t->nodeName == targetName;
             });
             
             if (new_end != objs.end()) {
                 size_t before_count = objs.size();
                 objs.erase(new_end, objs.end());
                 size_t after_count = objs.size();
                 deleted = true;
                 
                 // SCENE_LOG_INFO("Deleted " + std::to_string(before_count - after_count) + 
                 //               " triangles. Scene now has " + std::to_string(after_count) + " objects.");  // Verbose
                 
                 // STEP 3: Record undo command
                 auto command = std::make_unique<DeleteObjectCommand>(targetName, deleted_triangles);
                 history.record(std::move(command));
                 
                 // STEP 4: Rebuild GPU structures
                 // CRITICAL: Material index buffer MUST be regenerated
                 // When triangles are deleted, the index buffer shrinks
                 // If we don't rebuild, index buffer size != triangle count
                 // This causes material shifting and rendering artifacts
                 ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
                 ctx.renderer.resetCPUAccumulation();
                 if (ctx.optix_gpu_ptr) {
                    ctx.renderer.rebuildOptiXGeometry(ctx.scene, ctx.optix_gpu_ptr);
                 }
                 
                 // STEP 5: Refresh UI cache
                 rebuildMeshCache(ctx.scene.world.objects);
                 mesh_cache_valid = false;
                 
                 // STEP 6: Trigger render restart
                 ctx.start_render = true;
             }
        }
        else if (sel.selected.type == SelectableType::Light && sel.selected.light) {
             auto& lights = ctx.scene.lights;
             auto it = std::find(lights.begin(), lights.end(), sel.selected.light);
             if (it != lights.end()) {
                 lights.erase(it);
                 deleted = true;
                  // Update GPU
                 if (ctx.optix_gpu_ptr) {
                    ctx.optix_gpu_ptr->setLightParams(ctx.scene.lights);
                    ctx.optix_gpu_ptr->resetAccumulation();
                }
             }
        }
        
        if (deleted) {
            sel.clearSelection();
            // SCENE_LOG_INFO("Selection deleted.");  // Too verbose
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Toolbar
    // ─────────────────────────────────────────────────────────────────────────
    ImGui::TextColored(ImVec4(0.7f, 0.9f, 1.0f, 1.0f), "Transform Mode:");
    ImGui::SameLine();
    
    // Transform mode buttons
    bool is_translate = (sel.transform_mode == TransformMode::Translate);
    bool is_rotate = (sel.transform_mode == TransformMode::Rotate);
    bool is_scale = (sel.transform_mode == TransformMode::Scale);
    
    if (is_translate) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.9f, 1.0f));
    if (ImGui::Button("Move (W)")) sel.transform_mode = TransformMode::Translate;
    if (is_translate) ImGui::PopStyleColor();
    
    ImGui::SameLine();
    if (is_rotate) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.9f, 1.0f));
    if (ImGui::Button("Rotate (E)")) sel.transform_mode = TransformMode::Rotate;
    if (is_rotate) ImGui::PopStyleColor();
    
    ImGui::SameLine();
    if (is_scale) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.9f, 1.0f));
    if (ImGui::Button("Scale (R)")) sel.transform_mode = TransformMode::Scale;
    if (is_scale) ImGui::PopStyleColor();
    
    ImGui::Separator();
    
    // ─────────────────────────────────────────────────────────────────────────
    // Scene Tree
    // ─────────────────────────────────────────────────────────────────────────
    float available_h = ImGui::GetContentRegionAvail().y;
    // Split: 45% for Tree, Rest for Properties
    ImGui::BeginChild("HierarchyTree", ImVec2(0, available_h * 0.45f), true);
    
    // CAMERAS (Multi-camera support)
    if (!ctx.scene.cameras.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 0.8f, 1.0f, 1.0f));
        if (ImGui::TreeNode("Cameras")) {
            ImGui::PopStyleColor();
            
            for (size_t i = 0; i < ctx.scene.cameras.size(); i++) {
                ImGui::PushID((int)(1000 + i));  // Unique ID for each camera
                
                auto& cam = ctx.scene.cameras[i];
                if (!cam) { ImGui::PopID(); continue; }
                
                bool is_selected = (sel.selected.type == SelectableType::Camera && 
                                   sel.selected.camera == cam);
                bool is_active = (i == ctx.scene.active_camera_index);
                
                ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
                if (is_selected) flags |= ImGuiTreeNodeFlags_Selected;
                
                // Active camera indicator
                std::string label = is_active ? "[>] Camera #" + std::to_string(i) + " (Active)" 
                                               : "[O] Camera #" + std::to_string(i);
                
                ImVec4 color = is_active ? ImVec4(0.3f, 1.0f, 0.5f, 1.0f) : ImVec4(0.5f, 0.7f, 1.0f, 1.0f);
                ImGui::PushStyleColor(ImGuiCol_Text, color);
                ImGui::TreeNodeEx(label.c_str(), flags);
                ImGui::PopStyleColor();
                
                if (ImGui::IsItemClicked()) {
                    sel.selectCamera(cam);
                }
                
                // Double-click to set as active
                if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0)) {
                    ctx.scene.setActiveCamera(i);
                    if (ctx.optix_gpu_ptr && ctx.scene.camera) {
                        ctx.optix_gpu_ptr->setCameraParams(*ctx.scene.camera);
                        ctx.optix_gpu_ptr->resetAccumulation();
                    }
                    ctx.renderer.resetCPUAccumulation();
                    SCENE_LOG_INFO("Set Camera #" + std::to_string(i) + " as active");
                }
                
                ImGui::PopID();
            }
            ImGui::TreePop();
        } else {
            ImGui::PopStyleColor();
        }
    } else if (ctx.scene.camera) {
        // Fallback for single legacy camera
        bool is_selected = (sel.selected.type == SelectableType::Camera);
        ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
        if (is_selected) flags |= ImGuiTreeNodeFlags_Selected;
        
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 0.8f, 1.0f, 1.0f));
        ImGui::TreeNodeEx("[CAM] Camera", flags);
        ImGui::PopStyleColor();
        
        if (ImGui::IsItemClicked()) {
            sel.selectCamera(ctx.scene.camera);
        }
    }
    
    // LIGHTS
    if (!ctx.scene.lights.empty()) {
        if (ImGui::TreeNode("Lights")) {
            for (size_t i = 0; i < ctx.scene.lights.size(); i++) {
                ImGui::PushID((int)i);  // Unique ID for each light
                
                auto& light = ctx.scene.lights[i];
                bool is_selected = (sel.selected.type == SelectableType::Light && 
                                   sel.selected.light_index == (int)i);
                
                ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
                if (is_selected) flags |= ImGuiTreeNodeFlags_Selected;
                
                // Icon based on light type
                const char* icon = "[*]";  // Point light default
                ImVec4 color = ImVec4(1.0f, 0.9f, 0.4f, 1.0f);  // Yellow for lights
                
                std::string light_type = "Light";
                switch (light->type()) {
                    case LightType::Point: icon = "[*]"; light_type = "Point"; break;
                    case LightType::Directional: icon = "[>]"; light_type = "Directional"; color = ImVec4(1.0f, 0.7f, 0.3f, 1.0f); break;
                    case LightType::Spot: icon = "[V]"; light_type = "Spot"; break;
                    case LightType::Area: icon = "[#]"; light_type = "Area"; break;
                }
                
                std::string label = std::string(icon) + " " + light_type + " " + std::to_string(i + 1);
                
                ImGui::PushStyleColor(ImGuiCol_Text, color);
                ImGui::TreeNodeEx(label.c_str(), flags);
                ImGui::PopStyleColor();
                
                if (ImGui::IsItemClicked()) {
                    sel.selectLight(light, (int)i, label);
                }
                
                ImGui::PopID();  // End unique ID
            }
            ImGui::TreePop();
        }
    }
    // Check for scene changes to invalidate cache
    static size_t last_obj_count = 0;
    if (ctx.scene.world.objects.size() != last_obj_count) {
        mesh_cache_valid = false;
        last_obj_count = ctx.scene.world.objects.size();
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // OBJECTS LIST (HIERARCHY)
    // ─────────────────────────────────────────────────────────────────────────
    if (ImGui::TreeNode("Objects")) {
        
        // Ensure cache is valid
        if (!mesh_cache_valid) rebuildMeshCache(ctx.scene.world.objects);

        static ImGuiTextFilter filter;
        filter.Draw("Filter##objects");

        ImGuiListClipper clipper;
        clipper.Begin((int)mesh_ui_cache.size());
        
        while (clipper.Step()) {
            for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; i++) {
                if (i >= mesh_ui_cache.size()) break; 

                auto& kv = mesh_ui_cache[i];
                const std::string& name = kv.first;
                
                // Simple filter check
                if (filter.IsActive() && !filter.PassFilter(name.c_str())) continue;

                bool is_selected = (sel.selected.type == SelectableType::Object && 
                                    sel.selected.object && 
                                    sel.selected.object->nodeName == name);
                
                ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow;
                if (!is_selected) flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
                if (is_selected) flags |= ImGuiTreeNodeFlags_Selected | ImGuiTreeNodeFlags_DefaultOpen;

                ImGui::PushID(i);
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.8f, 0.8f, 0.85f, 1.0f));
                std::string displayName = name.empty() ? "Unnamed Object" : name;
                
                bool node_open = false;
                if (is_selected) {
                    node_open = ImGui::TreeNodeEx(displayName.c_str(), flags);
                } else {
                    ImGui::TreeNodeEx(displayName.c_str(), flags); // Leaf, no push
                }
                ImGui::PopStyleColor();

                if (ImGui::IsItemClicked()) {
                     if (!kv.second.empty()) {
                         auto& first_pair = kv.second[0]; 
                         sel.selectObject(first_pair.second, first_pair.first, name);
                     }
                }

                if (node_open && is_selected) {
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.7f, 0.7f, 0.7f, 1.0f));
                    ImGui::Indent();
                    
                    // --- In-Tree Properties ---
                    Vec3 pos = sel.selected.position;
                    // Position Control
                    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x);
                    if (ImGui::DragFloat3("##Pos", &pos.x, 0.1f)) {
                         Vec3 delta = pos - sel.selected.position;
                         sel.selected.position = pos;
                         if (!kv.second.empty()) {
                             auto tri = kv.second[0].second;
                             auto t_handle = tri->getTransformHandle();
                             if (t_handle) {
                                  t_handle->base.m[0][3] = pos.x;
                                  t_handle->base.m[1][3] = pos.y;
                                  t_handle->base.m[2][3] = pos.z;
                             }
                             // Sync Updates
                             ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
                             ctx.renderer.resetCPUAccumulation();
                             if (ctx.optix_gpu_ptr) {
                                ctx.optix_gpu_ptr->updateGeometry(ctx.scene.world.objects);
                                ctx.optix_gpu_ptr->resetAccumulation();
                             }
                             sel.selected.has_cached_aabb = false;
                         }
                    }
                    ImGui::PopItemWidth();

                    if (!kv.second.empty()) {
                        int matID = kv.second[0].second->getMaterialID();
                        ImGui::PushItemWidth(100);
                        if (ImGui::InputInt("Mat ID", &matID)) {
                             for(auto& pair : kv.second) pair.second->setMaterialID(matID);
                             if (ctx.optix_gpu_ptr) {
                                ctx.optix_gpu_ptr->updateGeometry(ctx.scene.world.objects);
                                ctx.optix_gpu_ptr->resetAccumulation();
                             }
                             ctx.renderer.resetCPUAccumulation();
                        }
                        ImGui::PopItemWidth();
                    }

                    ImGui::Unindent();
                    ImGui::PopStyleColor();
                    ImGui::TreePop(); 
                }
                ImGui::PopID();
            }
        }
        ImGui::TreePop();
    }
    // (Redundant internal property block removed)
    
    ImGui::EndChild();
    
    // ─────────────────────────────────────────────────────────────────────────
    // Selection Properties Panel (Bottom Half)
    // ─────────────────────────────────────────────────────────────────────────
    ImGui::BeginChild("PropertiesPanel", ImVec2(0, 0), true); // Fill remaining space

    if (sel.hasSelection()) {
        // Header with type and name
        const char* typeIcon = "[?]";
        ImVec4 typeColor = ImVec4(1, 1, 1, 1);
        
        switch (sel.selected.type) {
            case SelectableType::Camera: 
                typeIcon = "[CAM]"; 
                typeColor = ImVec4(0.4f, 0.8f, 1.0f, 1.0f);
                break;
            case SelectableType::Light: 
                typeIcon = "[*]"; 
                typeColor = ImVec4(1.0f, 0.9f, 0.4f, 1.0f);
                break;
            case SelectableType::Object: 
                typeIcon = "[M]"; 
                typeColor = ImVec4(0.7f, 0.8f, 0.9f, 1.0f);
                break;
            default: break;
        }
        
        ImGui::TextColored(typeColor, "%s %s", typeIcon, sel.selected.name.c_str());
        
        // Gizmo toggle and delete button on same line
        ImGui::Checkbox("Gizmo", &sel.show_gizmo);
        ImGui::SameLine(ImGui::GetWindowWidth() - 55);
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.2f, 0.2f, 1.0f));
        if (ImGui::SmallButton("Del")) {
            bool deleted = false;
            if (sel.selected.type == SelectableType::Object && sel.selected.object) {
                // Delete all triangles belonging to this mesh (by NodeName)
                std::string targetName = sel.selected.object->nodeName;
                auto& objs = ctx.scene.world.objects;
                
                auto new_end = std::remove_if(objs.begin(), objs.end(), 
                     [&](const std::shared_ptr<Hittable>& obj){
                         auto tri = std::dynamic_pointer_cast<Triangle>(obj);
                         return tri && tri->nodeName == targetName;
                     });
                     
                if (new_end != objs.end()) {
                    objs.erase(new_end, objs.end());
                    deleted = true;
                }
            }
            else if (sel.selected.type == SelectableType::Light && sel.selected.light) {
                // Delete Light with undo support
                auto light_to_delete = sel.selected.light;
                auto& lights = ctx.scene.lights;
                auto it = std::find(lights.begin(), lights.end(), light_to_delete);
                if (it != lights.end()) {
                    history.record(std::make_unique<DeleteLightCommand>(light_to_delete));
                    lights.erase(it);
                    deleted = true;
                }
            }
            
            if (deleted) {
                sel.clearSelection();
                invalidateCache();
                
                // SYNC RENDERERS
                ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
                ctx.renderer.resetCPUAccumulation();
                
                if (ctx.optix_gpu_ptr) {
                    ctx.renderer.rebuildOptiXGeometry(ctx.scene, ctx.optix_gpu_ptr);
                    ctx.optix_gpu_ptr->setLightParams(ctx.scene.lights);
                    ctx.optix_gpu_ptr->resetAccumulation();
                }
                is_bvh_dirty = false;
            } else {
                sel.clearSelection();
            }
        }
        ImGui::PopStyleColor();
        
        ImGui::Separator();
        
        // ═══════════════════════════════════════════════════════════════════
        // CAMERA PROPERTIES
        // ═══════════════════════════════════════════════════════════════════
        if (sel.selected.type == SelectableType::Camera && sel.selected.camera) {
            auto& cam = *sel.selected.camera;
            
            // Position
            Vec3 pos = cam.lookfrom;
            if (ImGui::DragFloat3("Position", &pos.x, 0.1f)) {
                Vec3 delta = pos - cam.lookfrom;
                cam.lookfrom = pos;
                cam.lookat = cam.lookat + delta;
                cam.update_camera_vectors();
                sel.selected.position = pos;
                
                if (ctx.optix_gpu_ptr && g_hasOptix) {
                    ctx.optix_gpu_ptr->setCameraParams(cam);
                    ctx.optix_gpu_ptr->resetAccumulation();
                }
            }
            
            // Target
            Vec3 target = cam.lookat;
            if (ImGui::DragFloat3("Target", &target.x, 0.1f)) {
                cam.lookat = target;
                cam.update_camera_vectors();
            }
            
            // FOV
            float fov = (float)cam.vfov;
            if (ImGui::SliderFloat("FOV", &fov, 10.0f, 120.0f)) {
                cam.vfov = fov;
                cam.fov = fov;
                cam.update_camera_vectors();
            }
            
            // Depth of Field
            if (ImGui::CollapsingHeader("Depth of Field")) {
                ImGui::SliderFloat("Aperture", &cam.aperture, 0.0f, 5.0f);
                ImGui::DragFloat("Focus Dist", &cam.focus_dist, 0.1f, 0.01f, 100.0f);
                cam.lens_radius = cam.aperture * 0.5f;
                ImGui::SliderInt("Blades", &cam.blade_count, 3, 12);
            }
            
            // Reset button
            if (ImGui::Button("Reset Camera", ImVec2(-1, 0))) {
                cam.reset();
                sel.selected.position = cam.lookfrom;
            }
        }
        
        // ═══════════════════════════════════════════════════════════════════
        // LIGHT PROPERTIES
        // ═══════════════════════════════════════════════════════════════════
        else if (sel.selected.type == SelectableType::Light && sel.selected.light) {
            auto& light = *sel.selected.light;
            bool light_changed = false;
            
            // Light type display
            const char* lightTypes[] = {"Point", "Directional", "Spot", "Area"};
            int typeIdx = (int)light.type();
            if (typeIdx >= 0 && typeIdx < 4) {
                ImGui::TextDisabled("Type: %s", lightTypes[typeIdx]);
            }
            
            // Position
            if (ImGui::DragFloat3("Position", &light.position.x, 0.1f)) {
                sel.selected.position = light.position;
                light_changed = true;
            }
            
            // Direction (for directional/spot)
            if (light.type() == LightType::Directional || light.type() == LightType::Spot) {
                if (ImGui::DragFloat3("Direction", &light.direction.x, 0.01f)) {
                    light_changed = true;
                }
            }
            
            // Color
            if (ImGui::ColorEdit3("Color", &light.color.x)) {
                light_changed = true;
            }
            
            // Intensity
            if (ImGui::DragFloat("Intensity", &light.intensity, 0.5f, 0.0f, 1000.0f)) {
                light_changed = true;
            }
            
            // Radius (Point/Directional)
            if (light.type() == LightType::Point || light.type() == LightType::Directional) {
                if (ImGui::DragFloat("Radius", &light.radius, 0.01f, 0.01f, 100.0f)) {
                    light_changed = true;
                }
            }

            // Spot Controls
            if (auto sl = dynamic_cast<SpotLight*>(&light)) {
                if (ImGui::DragFloat("Range", &sl->radius, 0.1f, 0.1f, 1000.0f)) light_changed = true;

                float angle = sl->getAngleDegrees();
                if (ImGui::DragFloat("Cone Angle", &angle, 0.5f, 1.0f, 89.0f)) {
                    sl->setAngleDegrees(angle);
                    light_changed = true;
                }
                float falloff = sl->getFalloff();
                if (ImGui::SliderFloat("Falloff", &falloff, 0.0f, 1.0f)) {
                    sl->setFalloff(falloff);
                    light_changed = true;
                }
            }
            // Area Controls
            else if (auto al = dynamic_cast<AreaLight*>(&light)) {
                 if (ImGui::DragFloat("Width", &al->width, 0.05f, 0.01f, 100.0f)) {
                     al->u = al->u.normalize() * al->width;
                     light_changed = true;
                 }
                 if (ImGui::DragFloat("Height", &al->height, 0.05f, 0.01f, 100.0f)) {
                     al->v = al->v.normalize() * al->height;
                     light_changed = true;
                 }
            }
            
            // Update GPU
            if (light_changed && ctx.optix_gpu_ptr && g_hasOptix) {
                ctx.optix_gpu_ptr->setLightParams(ctx.scene.lights);
                ctx.optix_gpu_ptr->resetAccumulation();
            }
        }
        
        // ═══════════════════════════════════════════════════════════════════
        // OBJECT/MESH PROPERTIES
        // ═══════════════════════════════════════════════════════════════════
        else if (sel.selected.type == SelectableType::Object && sel.selected.object) {
            // Show mesh info
            std::string meshName = sel.selected.object->nodeName;
            if (meshName.empty()) meshName = "Unnamed";
            
            ImGui::TextDisabled("Mesh: %s", meshName.c_str());
            
            // Count triangles with same name
            // FAST COUNT USING CACHE
            int tri_count = 0;
            if (mesh_cache_valid) {
                 auto it = mesh_cache.find(sel.selected.object->nodeName);
                 if (it != mesh_cache.end()) tri_count = (int)it->second.size();
            } else {
                 // Fallback or leave as 0
            }
            ImGui::TextDisabled("Triangles: %d", tri_count);
            
            // Position (read-only for now)
            Vec3 pos = sel.selected.position;
            ImGui::BeginDisabled();
            ImGui::DragFloat3("Position", &pos.x, 0.1f);
            ImGui::EndDisabled();
            
            // Material info
            auto mat = sel.selected.object->getMaterial();
            if (mat) {
                ImGui::TextDisabled("Material ID: %d", sel.selected.object->getMaterialID());
            }
        }
        
    } else {
        ImGui::TextDisabled("Select an item to view properties");
    }
    ImGui::EndChild(); // End PropertiesPanel

    
    
    
    // Window end removed as we are in a tab
    
    
    // Light Gizmos and Transform Gizmos moved to main draw() loop for all tabs
}

// ═══════════════════════════════════════════════════════════════════════════════
// SELECTION BOUNDING BOX DRAWING
// ═══════════════════════════════════════════════════════════════════════════════
void SceneUI::drawSelectionBoundingBox(UIContext& ctx) {
    SceneSelection& sel = ctx.selection;
    if (!sel.hasSelection() || !ctx.scene.camera) return;
    
    // Get bounding box corners based on selection type
    Vec3 bb_min, bb_max;
    bool has_bounds = false;
    
    if (sel.selected.type == SelectableType::Object && sel.selected.object) {
        
        // Use cached bounds if available
        if (sel.selected.has_cached_aabb) {
            bb_min = sel.selected.cached_aabb.min;
            bb_max = sel.selected.cached_aabb.max;
            has_bounds = true;
        } else {
            // NEEDED: Compute from all triangles with same nodeName
            std::string selectedName = sel.selected.object->nodeName;
            if (selectedName.empty()) selectedName = "Unnamed";

            bb_min = Vec3(1e10f, 1e10f, 1e10f);
            bb_max = Vec3(-1e10f, -1e10f, -1e10f);
            bool found_any = false;
            
            // OPTIMIZATION: Use mesh_cache instead of scanning all world objects
            if (!mesh_cache_valid) rebuildMeshCache(ctx.scene.world.objects);
            
            auto it = mesh_cache.find(selectedName);
            if (it != mesh_cache.end()) {
                 for (auto& pair : it->second) {
                    auto& tri = pair.second;
                    // No need for dynamic_cast check, cache stores Triangles
                    
                    // Get triangle vertices
                    Vec3 v0 = tri->getV0();
                    Vec3 v1 = tri->getV1();
                    Vec3 v2 = tri->getV2();
                    
                    // Use fminf/fmaxf to avoid Windows min/max macro conflict
                    bb_min.x = fminf(bb_min.x, fminf(v0.x, fminf(v1.x, v2.x)));
                    bb_min.y = fminf(bb_min.y, fminf(v0.y, fminf(v1.y, v2.y)));
                    bb_min.z = fminf(bb_min.z, fminf(v0.z, fminf(v1.z, v2.z)));
                    bb_max.x = fmaxf(bb_max.x, fmaxf(v0.x, fmaxf(v1.x, v2.x)));
                    bb_max.y = fmaxf(bb_max.y, fmaxf(v0.y, fmaxf(v1.y, v2.y)));
                    bb_max.z = fmaxf(bb_max.z, fmaxf(v0.z, fmaxf(v1.z, v2.z)));
                    found_any = true;
                 }
            }
            
            if (found_any) {
                has_bounds = true;
                sel.selected.cached_aabb.min = bb_min;
                sel.selected.cached_aabb.max = bb_max;
                sel.selected.has_cached_aabb = true;
            }
        }
    }
    else if (sel.selected.type == SelectableType::Light && sel.selected.light) {
        // Small box around light position
        Vec3 lightPos = sel.selected.light->position;
        float boxSize = 0.5f;
        bb_min = Vec3(lightPos.x - boxSize, lightPos.y - boxSize, lightPos.z - boxSize);
        bb_max = Vec3(lightPos.x + boxSize, lightPos.y + boxSize, lightPos.z + boxSize);
        has_bounds = true;
    }
    else if (sel.selected.type == SelectableType::Camera && sel.selected.camera) {
        // Small box around camera position
        Vec3 camPos = sel.selected.camera->lookfrom;
        float boxSize = 0.5f;
        bb_min = Vec3(camPos.x - boxSize, camPos.y - boxSize, camPos.z - boxSize);
        bb_max = Vec3(camPos.x + boxSize, camPos.y + boxSize, camPos.z + boxSize);
        has_bounds = true;
    }
    
    if (!has_bounds) return;
    
    // 8 corners of bounding box
    Vec3 corners[8] = {
        Vec3(bb_min.x, bb_min.y, bb_min.z),
        Vec3(bb_max.x, bb_min.y, bb_min.z),
        Vec3(bb_max.x, bb_max.y, bb_min.z),
        Vec3(bb_min.x, bb_max.y, bb_min.z),
        Vec3(bb_min.x, bb_min.y, bb_max.z),
        Vec3(bb_max.x, bb_min.y, bb_max.z),
        Vec3(bb_max.x, bb_max.y, bb_max.z),
        Vec3(bb_min.x, bb_max.y, bb_max.z),
    };
    
    // Project corners to screen space
    Camera& cam = *ctx.scene.camera;
    ImGuiIO& io = ImGui::GetIO();
    float screen_w = io.DisplaySize.x;
    float screen_h = io.DisplaySize.y;
    
    // Camera basis vectors
    Vec3 cam_forward = (cam.lookat - cam.lookfrom).normalize();
    Vec3 cam_right = cam_forward.cross(cam.vup).normalize();
    Vec3 cam_up = cam_right.cross(cam_forward).normalize();
    
    // FOV calculations
    float fov_rad = cam.vfov * 3.14159265359f / 180.0f;
    float tan_half_fov = tanf(fov_rad * 0.5f);
    
    ImVec2 screen_pts[8];
    bool all_visible = true;
    
    for (int i = 0; i < 8; i++) {
        // Vector from camera to corner
        Vec3 to_corner = corners[i] - cam.lookfrom;
        
        // Distance along camera forward axis
        float depth = to_corner.dot(cam_forward);
        
        // Check if behind camera
        if (depth <= 0.01f) {
            all_visible = false;
            break;
        }
        
        // Project onto camera's local X and Y axes
        float local_x = to_corner.dot(cam_right);
        float local_y = to_corner.dot(cam_up);
        
        // Perspective divide
        float half_height = depth * tan_half_fov;
        float half_width = half_height * aspect_ratio;
        
        // Normalized device coordinates (-1 to 1)
        float ndc_x = local_x / half_width;
        float ndc_y = local_y / half_height;
        
        // To screen coordinates
        screen_pts[i].x = (ndc_x * 0.5f + 0.5f) * screen_w;
        screen_pts[i].y = (0.5f - ndc_y * 0.5f) * screen_h;  // Y is inverted
    }
    
    if (!all_visible) return;
    
    // Draw wireframe using ImGui DrawList
    ImDrawList* draw_list = ImGui::GetForegroundDrawList();
    ImU32 color = IM_COL32(0, 255, 128, 255);  // Green
    float thickness = 2.0f;
    
    // 12 edges of the box
    // Bottom face
    draw_list->AddLine(screen_pts[0], screen_pts[1], color, thickness);
    draw_list->AddLine(screen_pts[1], screen_pts[2], color, thickness);
    draw_list->AddLine(screen_pts[2], screen_pts[3], color, thickness);
    draw_list->AddLine(screen_pts[3], screen_pts[0], color, thickness);
    
    // Top face
    draw_list->AddLine(screen_pts[4], screen_pts[5], color, thickness);
    draw_list->AddLine(screen_pts[5], screen_pts[6], color, thickness);
    draw_list->AddLine(screen_pts[6], screen_pts[7], color, thickness);
    draw_list->AddLine(screen_pts[7], screen_pts[4], color, thickness);
    
    // Vertical edges
    draw_list->AddLine(screen_pts[0], screen_pts[4], color, thickness);
    draw_list->AddLine(screen_pts[1], screen_pts[5], color, thickness);
    draw_list->AddLine(screen_pts[2], screen_pts[6], color, thickness);
    draw_list->AddLine(screen_pts[3], screen_pts[7], color, thickness);
}

// ═══════════════════════════════════════════════════════════════════════════════
// IMGUIZMO TRANSFORM GIZMO
// ═══════════════════════════════════════════════════════════════════════════════
void SceneUI::drawTransformGizmo(UIContext& ctx) {
    SceneSelection& sel = ctx.selection;
    if (!sel.hasSelection() || !sel.show_gizmo || !ctx.scene.camera) return;
    
    Camera& cam = *ctx.scene.camera;
    ImGuiIO& io = ImGui::GetIO();
    
    // Setup ImGuizmo
    ImGuizmo::BeginFrame();
    ImGuizmo::SetOrthographic(false);
    ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);
    
    // ─────────────────────────────────────────────────────────────────────────
    // Build View Matrix (LookAt)
    // ─────────────────────────────────────────────────────────────────────────
    Vec3 eye = cam.lookfrom;
    Vec3 target = cam.lookat;
    Vec3 up = cam.vup;
    
    Vec3 f = (target - eye).normalize();  // Forward
    Vec3 r = f.cross(up).normalize();     // Right
    Vec3 u = r.cross(f);                   // Up
    
    float viewMatrix[16] = {
        r.x,  u.x, -f.x, 0.0f,
        r.y,  u.y, -f.y, 0.0f,
        r.z,  u.z, -f.z, 0.0f,
        -r.dot(eye), -u.dot(eye), f.dot(eye), 1.0f
    };
    
    // ─────────────────────────────────────────────────────────────────────────
    // Build Projection Matrix (Perspective)
    // ─────────────────────────────────────────────────────────────────────────
    float fov_rad = cam.vfov * 3.14159265359f / 180.0f;
    float near_plane = 0.1f;
    float far_plane = 10000.0f;
    float tan_half_fov = tanf(fov_rad * 0.5f);
    
    float projMatrix[16] = {0};
    projMatrix[0] = 1.0f / (aspect_ratio * tan_half_fov);
    projMatrix[5] = 1.0f / tan_half_fov;
    projMatrix[10] = -(far_plane + near_plane) / (far_plane - near_plane);
    projMatrix[11] = -1.0f;
    projMatrix[14] = -(2.0f * far_plane * near_plane) / (far_plane - near_plane);
    
    auto Project = [&](Vec3 p) -> ImVec2 {
         float x=p.x, y=p.y, z=p.z;
         float vx = viewMatrix[0]*x + viewMatrix[4]*y + viewMatrix[8]*z + viewMatrix[12];
         float vy = viewMatrix[1]*x + viewMatrix[5]*y + viewMatrix[9]*z + viewMatrix[13];
         float vz = viewMatrix[2]*x + viewMatrix[6]*y + viewMatrix[10]*z + viewMatrix[14];
         float vw = viewMatrix[3]*x + viewMatrix[7]*y + viewMatrix[11]*z + viewMatrix[15];
         float cx = projMatrix[0]*vx + projMatrix[4]*vy + projMatrix[8]*vz + projMatrix[12]*vw;
         float cy = projMatrix[1]*vx + projMatrix[5]*vy + projMatrix[9]*vz + projMatrix[13]*vw;
         float cw = projMatrix[3]*vx + projMatrix[7]*vy + projMatrix[11]*vz + projMatrix[15]*vw;
         if (cw < 0.1f) return ImVec2(-10000, -10000);
         return ImVec2(((cx/cw)*0.5f+0.5f)*io.DisplaySize.x, (1.0f-((cy/cw)*0.5f+0.5f))*io.DisplaySize.y);
    };

    // ─────────────────────────────────────────────────────────────────────────
    // Get Object Matrix
    // ─────────────────────────────────────────────────────────────────────────
    float objectMatrix[16];
    Vec3 pos = sel.selected.position;

    // Fix AreaLight Pivot: Use Center instead of Corner
    if (sel.selected.type == SelectableType::Light) {
        if (auto al = std::dynamic_pointer_cast<AreaLight>(sel.selected.light)) {
             pos = al->position + al->u * 0.5f + al->v * 0.5f;
        }
    }
    
    // Initialize as identity with position
    Matrix4x4 startMat = Matrix4x4::identity();
    startMat.m[0][3] = pos.x;
    startMat.m[1][3] = pos.y;
    startMat.m[2][3] = pos.z;

    // Handle Light Rotation (Directional/Spot)
    if (sel.selected.type == SelectableType::Light && sel.selected.light) {
        Vec3 dir(0, 0, 0);
        bool hasDir = false;
        
        if (auto dl = std::dynamic_pointer_cast<DirectionalLight>(sel.selected.light)) {
             // DirectionalLight: getDirection returns vector TO Light (inverse of direction)
             // We want the light direction for visualization
             dir = dl->getDirection(Vec3(0)).normalize() * -1.0f; 
             hasDir = true;
        } 
        else if (auto sl = std::dynamic_pointer_cast<SpotLight>(sel.selected.light)) {
             // Manual Matrix for SpotLight to include Angle as Scale
             hasDir = false; 
             float angle = sl->getAngleDegrees();
             if (angle < 1.0f) angle = 1.0f;

             // Direction Basis
             Vec3 dirVec = sl->direction.normalize();
             Vec3 Z = -dirVec; 
             Vec3 Y_temp(0, 1, 0);
             if (abs(Vec3::dot(Z, Y_temp)) > 0.99f) Y_temp = Vec3(1, 0, 0);
             
             Vec3 X = Vec3::cross(Y_temp, Z).normalize();
             Vec3 Y = Vec3::cross(Z, X).normalize();
             
             // Scale X and Y by Angle for Gizmo Interaction
             Vec3 X_scaled = X * angle;
             Vec3 Y_scaled = Y * angle;
             
             // Scale Z by (1 + Falloff) for Falloff interaction
             float zScale = 1.0f + sl->getFalloff();
             Vec3 Z_scaled = Z * zScale;
             
             startMat.m[0][0] = X_scaled.x; startMat.m[0][1] = Y_scaled.x; startMat.m[0][2] = Z_scaled.x; 
             startMat.m[1][0] = X_scaled.y; startMat.m[1][1] = Y_scaled.y; startMat.m[1][2] = Z_scaled.y; 
             startMat.m[2][0] = X_scaled.z; startMat.m[2][1] = Y_scaled.z; startMat.m[2][2] = Z_scaled.z; 

             // -----------------------------------------------------
             // Visual Helper: Cone Draw
             // -----------------------------------------------------
             ImDrawList* dl = ImGui::GetBackgroundDrawList();
             Vec3 pos = sl->position;
             float h = 5.0f; // Visual height
             float r = tanf(angle * 3.14159f / 180.0f * 0.5f) * h; 
             
             ImVec2 pTip = Project(pos);
             Vec3 centerBase = pos + dirVec * h;
             
             ImU32 col = IM_COL32(255, 255, 0, 180);
             Vec3 prevP;
             bool first = true;
             
             for(int i=0; i<=24; ++i) {
                 float t = (float)i / 24.0f * 6.28318f;
                 Vec3 pBase = centerBase + (X * cosf(t) + Y * sinf(t)) * r;
                 ImVec2 pScreen = Project(pBase);
                 
                 if (!first) dl->AddLine(Project(prevP), pScreen, col, 2.0f);
                 
                 if (i % 6 == 0) dl->AddLine(pTip, pScreen, col, 1.0f);
                 
                 prevP = pBase;
                 first = false;
             }

             // Inner Cone (Falloff)
             float falloff = sl->getFalloff();
             if (falloff > 0.05f) {
                 float innerAngle = angle * (1.0f - falloff);
                 float rInner = tanf(innerAngle * 3.14159f / 180.0f * 0.5f) * h;
                 ImU32 colInner = IM_COL32(255, 160, 20, 120);
                 Vec3 prevIn;
                 bool firstIn = true;
                 for(int i=0; i<=24; i++) {
                     float t = (float)i / 24.0f * 6.28318f;
                     Vec3 pBase = centerBase + (X * cosf(t) + Y * sinf(t)) * rInner;
                     if (!firstIn && i%2==0) dl->AddLine(Project(prevIn), Project(pBase), colInner, 1.0f); // Dashed-ish effect
                     prevIn = pBase;
                     firstIn = false;
                 }
             }
        }
        else if (auto al = std::dynamic_pointer_cast<AreaLight>(sel.selected.light)) {
             hasDir = false; 
             // Use actual magnitude for Scale interaction
             Vec3 X = al->u;
             Vec3 Z = al->v;
             // Normalized Y (Normal)
             Vec3 Y = Vec3::cross(X, Z).normalize();
             
             startMat.m[0][0] = X.x; startMat.m[0][1] = Y.x; startMat.m[0][2] = Z.x; 
             startMat.m[1][0] = X.y; startMat.m[1][1] = Y.y; startMat.m[1][2] = Z.y; 
             startMat.m[2][0] = X.z; startMat.m[2][1] = Y.z; startMat.m[2][2] = Z.z; 
             
             // Visualization: Direction Arrow
             ImDrawList* dl = ImGui::GetBackgroundDrawList();
             Vec3 center = pos; 
             Vec3 normal = Y.normalize();
             float len = 3.0f;
             Vec3 pTip = center + normal * len;
             dl->AddLine(Project(center), Project(pTip), IM_COL32(255, 255, 0, 200), 2.0f);
             dl->AddCircleFilled(Project(pTip), 4.0f, IM_COL32(255, 255, 0, 255));
        }

        if (hasDir) {
            // Align Gizmo -Z with Light Direction
            Vec3 Z = -dir;
            Vec3 Y(0, 1, 0);
            if (abs(Vec3::dot(Z, Y)) > 0.99f) Y = Vec3(1, 0, 0); // Lock prevention
            Vec3 X = Vec3::cross(Y, Z).normalize();
            Y = Vec3::cross(Z, X).normalize();
            
            startMat.m[0][0] = X.x; startMat.m[0][1] = Y.x; startMat.m[0][2] = Z.x; 
            startMat.m[1][0] = X.y; startMat.m[1][1] = Y.y; startMat.m[1][2] = Z.y; 
            startMat.m[2][0] = X.z; startMat.m[2][1] = Y.z; startMat.m[2][2] = Z.z; 
        }
    }

    objectMatrix[0] = startMat.m[0][0]; objectMatrix[1] = startMat.m[1][0]; objectMatrix[2] = startMat.m[2][0]; objectMatrix[3] = startMat.m[3][0];
    objectMatrix[4] = startMat.m[0][1]; objectMatrix[5] = startMat.m[1][1]; objectMatrix[6] = startMat.m[2][1]; objectMatrix[7] = startMat.m[3][1];
    objectMatrix[8] = startMat.m[0][2]; objectMatrix[9] = startMat.m[1][2]; objectMatrix[10] = startMat.m[2][2]; objectMatrix[11] = startMat.m[3][2];
    objectMatrix[12] = startMat.m[0][3]; objectMatrix[13] = startMat.m[1][3]; objectMatrix[14] = startMat.m[2][3]; objectMatrix[15] = startMat.m[3][3];

    // If object has transform, use it
    if (sel.selected.type == SelectableType::Object && sel.selected.object) {
        auto transform = sel.selected.object->getTransformHandle();
        if (transform) {
            Matrix4x4 mat = transform->base;
            objectMatrix[0] = mat.m[0][0]; objectMatrix[1] = mat.m[1][0]; objectMatrix[2] = mat.m[2][0]; objectMatrix[3] = mat.m[3][0];
            objectMatrix[4] = mat.m[0][1]; objectMatrix[5] = mat.m[1][1]; objectMatrix[6] = mat.m[2][1]; objectMatrix[7] = mat.m[3][1];
            objectMatrix[8] = mat.m[0][2]; objectMatrix[9] = mat.m[1][2]; objectMatrix[10] = mat.m[2][2]; objectMatrix[11] = mat.m[3][2];
            objectMatrix[12] = mat.m[0][3]; objectMatrix[13] = mat.m[1][3]; objectMatrix[14] = mat.m[2][3]; objectMatrix[15] = mat.m[3][3];
        }
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Keyboard Shortcuts for Transform Mode
    // ─────────────────────────────────────────────────────────────────────────
    // Only process when viewport has focus (not UI panels)
    if (sel.hasSelection() && !ImGui::GetIO().WantCaptureKeyboard) {
        if (ImGui::IsKeyPressed(ImGuiKey_G)) {
            sel.transform_mode = TransformMode::Translate;
        }
        else if (ImGui::IsKeyPressed(ImGuiKey_R)) {
            sel.transform_mode = TransformMode::Rotate;
        }
        else if (ImGui::IsKeyPressed(ImGuiKey_S) && !ImGui::GetIO().KeyShift) {
            // S alone = Scale, Shift+S would trigger duplication so check
            sel.transform_mode = TransformMode::Scale;
        }
        else if (ImGui::IsKeyPressed(ImGuiKey_W)) {
            // Cycle through modes
            switch (sel.transform_mode) {
                case TransformMode::Translate: sel.transform_mode = TransformMode::Rotate; break;
                case TransformMode::Rotate: sel.transform_mode = TransformMode::Scale; break;
                case TransformMode::Scale: sel.transform_mode = TransformMode::Translate; break;
            }
        }
        
        // Shift + D = Duplicate Object
        if (ImGui::IsKeyPressed(ImGuiKey_D) && ImGui::GetIO().KeyShift) {
            if (sel.selected.type == SelectableType::Object && sel.selected.object) {
                std::string targetName = sel.selected.object->nodeName;
                if (targetName.empty()) targetName = "Unnamed";
                
                // Unique name generation
                std::string baseName = targetName;
                size_t lastUnderscore = baseName.rfind('_');
                if (lastUnderscore != std::string::npos) {
                    std::string suffix = baseName.substr(lastUnderscore + 1);
                    if (!suffix.empty() && std::all_of(suffix.begin(), suffix.end(), ::isdigit)) {
                        baseName = baseName.substr(0, lastUnderscore);
                    }
                }
                
                int counter = 1;
                std::string newName;
                bool nameExists = true;
                while (nameExists) {
                    newName = baseName + "_" + std::to_string(counter);
                    nameExists = false;
                    for (const auto& obj : ctx.scene.world.objects) {
                        auto tri = std::dynamic_pointer_cast<Triangle>(obj);
                        if (tri && tri->nodeName == newName) {
                            nameExists = true;
                            break;
                        }
                    }
                    counter++;
                }
                
                // Create duplicates
                std::vector<std::shared_ptr<Hittable>> newTriangles;
                std::shared_ptr<Triangle> firstNewTri = nullptr;
                auto it = mesh_cache.find(targetName);
                if (it != mesh_cache.end()) {
                    for (auto& pair : it->second) {
                        auto& oldTri = pair.second;
                        auto newTri = std::make_shared<Triangle>(*oldTri);
                        newTri->setNodeName(newName);
                        newTriangles.push_back(newTri);
                        if (!firstNewTri) firstNewTri = newTri;
                    }
                }
                
                // Add to scene
                if (!newTriangles.empty()) {
                    ctx.scene.world.objects.insert(ctx.scene.world.objects.end(), newTriangles.begin(), newTriangles.end());
                    sel.selectObject(firstNewTri, -1, newName);
                    rebuildMeshCache(ctx.scene.world.objects);
                    
                    // Record undo command
                    std::vector<std::shared_ptr<Triangle>> new_tri_vec;
                    for (auto& ht : newTriangles) {
                        auto tri = std::dynamic_pointer_cast<Triangle>(ht);
                        if (tri) new_tri_vec.push_back(tri);
                    }
                    auto command = std::make_unique<DuplicateObjectCommand>(targetName, newName, new_tri_vec);
                    history.record(std::move(command));
                    
                    // Full rebuild for duplication
                    ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
                    ctx.renderer.resetCPUAccumulation();
                    if (ctx.optix_gpu_ptr) {
                        ctx.renderer.rebuildOptiXGeometry(ctx.scene, ctx.optix_gpu_ptr);
                    }
                    is_bvh_dirty = false;
                    SCENE_LOG_INFO("Duplicated object: " + targetName + " -> " + newName);
                }
            }
        }
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Determine Gizmo Operation
    // ─────────────────────────────────────────────────────────────────────────
    ImGuizmo::OPERATION operation = ImGuizmo::TRANSLATE;
    switch (sel.transform_mode) {
        case TransformMode::Translate: operation = ImGuizmo::TRANSLATE; break;
        case TransformMode::Rotate: operation = ImGuizmo::ROTATE; break;
        case TransformMode::Scale: operation = ImGuizmo::SCALE; break;
    }
    
    ImGuizmo::MODE mode = (sel.transform_space == TransformSpace::Local) ? 
        ImGuizmo::LOCAL : ImGuizmo::WORLD;
    
    // ─────────────────────────────────────────────────────────────────────────
    // Shift + Drag Duplication Logic
    // ─────────────────────────────────────────────────────────────────────────
    static bool was_using_gizmo = false;
    static LightState drag_start_light_state;
    static std::shared_ptr<Light> drag_light = nullptr;
    bool is_using = ImGuizmo::IsUsing();

    if (is_using && !was_using_gizmo) {
        // Manipülasyon yeni başladı
        if (ImGui::GetIO().KeyShift && sel.selected.type == SelectableType::Object && sel.selected.object) {
            
            std::string targetName = sel.selected.object->nodeName;
            if (targetName.empty()) targetName = "Unnamed";
            
            // Unique name generation
            std::string baseName = targetName;
            size_t lastUnderscore = baseName.rfind('_');
            if (lastUnderscore != std::string::npos) {
                std::string suffix = baseName.substr(lastUnderscore + 1);
                if (!suffix.empty() && std::all_of(suffix.begin(), suffix.end(), ::isdigit)) {
                    baseName = baseName.substr(0, lastUnderscore);
                }
            }
            int copyNum = 1;
            std::string newName;
            do { newName = baseName + "_" + std::to_string(copyNum++); } while (mesh_cache.find(newName) != mesh_cache.end());

            if (!mesh_cache_valid) rebuildMeshCache(ctx.scene.world.objects);

            // Create Unique Transform
            std::shared_ptr<Transform> newTransform = std::make_shared<Transform>();
            if (sel.selected.object->getTransformHandle()) {
                *newTransform = *sel.selected.object->getTransformHandle();
            }

            // Duplicate Triangles
            std::vector<std::shared_ptr<Hittable>> newTriangles;
            std::shared_ptr<Triangle> firstNewTri = nullptr;
            auto it = mesh_cache.find(targetName);
            if (it != mesh_cache.end()) {
                for (auto& pair : it->second) {
                    auto& oldTri = pair.second;
                    auto newTri = std::make_shared<Triangle>(*oldTri); 
                    newTri->setTransformHandle(newTransform);
                    newTri->setNodeName(newName);
                    newTriangles.push_back(newTri);
                    if (!firstNewTri) firstNewTri = newTri;
                }
            }

            // Add to Scene
            if (!newTriangles.empty()) {
                ctx.scene.world.objects.insert(ctx.scene.world.objects.end(), newTriangles.begin(), newTriangles.end());
                sel.selectObject(firstNewTri, -1, newName); // Select New
                rebuildMeshCache(ctx.scene.world.objects);
                
                // CRITICAL: Duplication adds new triangles - requires full rebuild
                // Material index buffer must be regenerated (same as deletion)
                ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
                ctx.renderer.resetCPUAccumulation();
                if (ctx.optix_gpu_ptr) {
                    ctx.renderer.rebuildOptiXGeometry(ctx.scene, ctx.optix_gpu_ptr);
                }
                is_bvh_dirty = false;
                SCENE_LOG_INFO("Duplicated object: " + targetName + " -> " + newName);
            }
        }
        else if (ImGui::GetIO().KeyShift && sel.selected.type == SelectableType::Light && sel.selected.light) {
             std::shared_ptr<Light> newLight = nullptr;
             auto l = sel.selected.light;
             if (std::dynamic_pointer_cast<PointLight>(l)) newLight = std::make_shared<PointLight>(*(PointLight*)l.get());
             else if (std::dynamic_pointer_cast<DirectionalLight>(l)) newLight = std::make_shared<DirectionalLight>(*(DirectionalLight*)l.get());
             else if (std::dynamic_pointer_cast<SpotLight>(l)) newLight = std::make_shared<SpotLight>(*(SpotLight*)l.get());
             else if (std::dynamic_pointer_cast<AreaLight>(l)) newLight = std::make_shared<AreaLight>(*(AreaLight*)l.get());
             
             if (newLight) {
                 ctx.scene.lights.push_back(newLight);
                 history.record(std::make_unique<AddLightCommand>(newLight));
                 sel.selectLight(newLight);
                 if (ctx.optix_gpu_ptr) { ctx.optix_gpu_ptr->setLightParams(ctx.scene.lights); ctx.optix_gpu_ptr->resetAccumulation(); }
                 SCENE_LOG_INFO("Duplicated Light (Shift+Drag)");
             }
        }
        else if (sel.selected.type == SelectableType::Light && sel.selected.light && !ImGui::GetIO().KeyShift) {
             // START LIGHT TRANSFORM RECORDING
             drag_light = sel.selected.light;
             drag_start_light_state = LightState::capture(*drag_light);
        }
        else if (sel.selected.type == SelectableType::Object && sel.selected.object) {
             // START TRANSFORM RECORDING (Normal drag without Shift)
             auto transform = sel.selected.object->getTransformHandle();
             if (transform) {
                 drag_start_state.matrix = transform->base;
                 drag_object_name = sel.selected.object->nodeName;
             }
        }
    }
    
    // END DRAG (Release)
    if (!is_using && was_using_gizmo && sel.selected.type == SelectableType::Object && sel.selected.object) {
         // END TRANSFORM RECORDING
         auto t = sel.selected.object->getTransformHandle();
         if (t) {
             TransformState final_state;
             final_state.matrix = t->base;
             
             // Check delta
             bool changed = false;
             for(int i=0; i<4; ++i) 
                 for(int j=0; j<4; ++j) 
                     if (std::abs(final_state.matrix.m[i][j] - drag_start_state.matrix.m[i][j]) > 0.0001f) 
                         changed = true;
                          
             if (changed) {
                 history.record(std::make_unique<TransformCommand>(drag_object_name, drag_start_state, final_state));
             }
         }
    }
    
    // END DRAG for Light
    if (!is_using && was_using_gizmo && sel.selected.type == SelectableType::Light && drag_light) {
         LightState final_light_state = LightState::capture(*drag_light);
         
         // Check if position or other properties changed
         bool changed = (final_light_state.position - drag_start_light_state.position).length() > 0.0001f ||
                       (final_light_state.direction - drag_start_light_state.direction).length() > 0.0001f ||
                       std::abs(final_light_state.angle - drag_start_light_state.angle) > 0.0001f;
         
         if (changed) {
             history.record(std::make_unique<TransformLightCommand>(drag_light, drag_start_light_state, final_light_state));
         }
         drag_light = nullptr;
    }
    
    was_using_gizmo = is_using;

    // ─────────────────────────────────────────────────────────────────────────
    // Render and Manipulate Gizmo
    // ─────────────────────────────────────────────────────────────────────────
    bool manipulated = ImGuizmo::Manipulate(viewMatrix, projMatrix, operation, mode, objectMatrix);
    
    if (manipulated) {
        Vec3 newPos(objectMatrix[12], objectMatrix[13], objectMatrix[14]);
        sel.selected.position = newPos; // Update gizmo/bbox center

        if (sel.selected.type == SelectableType::Light && sel.selected.light) {
            sel.selected.light->position = newPos;
            Vec3 zAxis(objectMatrix[8], objectMatrix[9], objectMatrix[10]);
            Vec3 newDir = -zAxis.normalize(); // Gizmo -Z aligned
            
            if (auto dl = std::dynamic_pointer_cast<DirectionalLight>(sel.selected.light)) dl->setDirection(newDir);
            else if (auto sl = std::dynamic_pointer_cast<SpotLight>(sel.selected.light)) {
                 sl->direction = newDir;
                 
                 // Update Angle from Gizmo Scale
                 Vec3 right(objectMatrix[0], objectMatrix[1], objectMatrix[2]);
                 float angle = right.length();
                 
                 if (angle < 0.1f) angle = 0.1f;
                 if (angle > 179.0f) angle = 179.0f;
                 
                 sl->setAngleDegrees(angle);
                 
                 // Falloff Update (Z Scale represents 1.0 + Falloff)
                 Vec3 forward(objectMatrix[8], objectMatrix[9], objectMatrix[10]);
                 float sz = forward.length();
                 float newF = sz - 1.0f;
                 if (newF < 0.0f) newF = 0.0f;
                 if (newF > 1.0f) newF = 1.0f;
                 sl->setFalloff(newF);
            }
            else if (auto al = std::dynamic_pointer_cast<AreaLight>(sel.selected.light)) {
                 Vec3 right(objectMatrix[0], objectMatrix[1], objectMatrix[2]);
                 Vec3 forward(objectMatrix[8], objectMatrix[9], objectMatrix[10]);
                 
                 // Scale handling: Assign directly (Fixes infinite growth)
                 float sx = right.length();
                 float sz = forward.length();
                 
                 if (sx > 0.001f) al->width = sx;
                 if (sz > 0.001f) al->height = sz;

                 // Set vectors directly (they carry rotation and scale)
                 al->u = right;
                 al->v = forward;
                 
                 // Correct Position
                 al->position = newPos - (al->u * 0.5f) - (al->v * 0.5f);
            }
            
            if (ctx.optix_gpu_ptr) {
                ctx.optix_gpu_ptr->setLightParams(ctx.scene.lights);
                ctx.optix_gpu_ptr->resetAccumulation();
            }
        }
        else if (sel.selected.type == SelectableType::Camera && sel.selected.camera) {
            Vec3 delta = newPos - sel.selected.camera->lookfrom;
            sel.selected.camera->lookfrom = newPos;
            sel.selected.camera->lookat = sel.selected.camera->lookat + delta;
            sel.selected.camera->update_camera_vectors();
            if (ctx.optix_gpu_ptr) {
                ctx.optix_gpu_ptr->setCameraParams(*sel.selected.camera);
                ctx.optix_gpu_ptr->resetAccumulation();
            }
        }
        else if (sel.selected.type == SelectableType::Object && sel.selected.object) {
            Matrix4x4 newMat;
            newMat.m[0][0] = objectMatrix[0]; newMat.m[1][0] = objectMatrix[1]; newMat.m[2][0] = objectMatrix[2]; newMat.m[3][0] = objectMatrix[3];
            newMat.m[0][1] = objectMatrix[4]; newMat.m[1][1] = objectMatrix[5]; newMat.m[2][1] = objectMatrix[6]; newMat.m[3][1] = objectMatrix[7];
            newMat.m[0][2] = objectMatrix[8]; newMat.m[1][2] = objectMatrix[9]; newMat.m[2][2] = objectMatrix[10]; newMat.m[3][2] = objectMatrix[11];
            newMat.m[0][3] = objectMatrix[12]; newMat.m[1][3] = objectMatrix[13]; newMat.m[2][3] = objectMatrix[14]; newMat.m[3][3] = objectMatrix[15];

            std::string targetName = sel.selected.object->nodeName;
            if (targetName.empty()) targetName = "Unnamed";

            if (!mesh_cache_valid) rebuildMeshCache(ctx.scene.world.objects);

            auto it = mesh_cache.find(targetName);
            if (it != mesh_cache.end()) {
                for (auto& pair : it->second) {
                    auto& tri = pair.second;
                    auto t_handle = tri->getTransformHandle();
                    if (t_handle) t_handle->setBase(newMat);
                    tri->updateTransformedVertices();
                }
            }
            sel.selected.has_cached_aabb = false;

            // IMMEDIATE UPDATE - SYNCHRONIZE BOTH
            if (ctx.optix_gpu_ptr) {
                ctx.optix_gpu_ptr->updateGeometry(ctx.scene.world.objects);
                ctx.optix_gpu_ptr->resetAccumulation();
            }

            // Always update CPU BVH as well to keep them in sync
            ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
            ctx.renderer.resetCPUAccumulation();
            is_bvh_dirty = false;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CAMERA GIZMOS - Draw camera icons in viewport
// ═══════════════════════════════════════════════════════════════════════════════
void SceneUI::drawCameraGizmos(UIContext& ctx) {
    if (!ctx.scene.camera || ctx.scene.cameras.size() <= 1) return;
    
    Camera& activeCam = *ctx.scene.camera;
    ImDrawList* draw_list = ImGui::GetForegroundDrawList();
    ImGuiIO& io = ImGui::GetIO();
    float screen_w = io.DisplaySize.x;
    float screen_h = io.DisplaySize.y;
    
    // Camera basis vectors for projection
    Vec3 cam_forward = (activeCam.lookat - activeCam.lookfrom).normalize();
    Vec3 cam_right = cam_forward.cross(activeCam.vup).normalize();
    Vec3 cam_up = cam_right.cross(cam_forward).normalize();
    float tan_half_fov = tan(activeCam.vfov * 0.5f * M_PI / 180.0f);
    float aspect = screen_w / screen_h;
    
    // Lambda to project 3D point to screen
    auto Project = [&](const Vec3& world_pos, ImVec2& screen_pos) -> bool {
        Vec3 to_point = world_pos - activeCam.lookfrom;
        float depth = to_point.dot(cam_forward);
        if (depth < 0.1f) return false;  // Behind camera
        
        float local_x = to_point.dot(cam_right);
        float local_y = to_point.dot(cam_up);
        
        float half_height = depth * tan_half_fov;
        float half_width = half_height * aspect;
        
        float ndc_x = local_x / half_width;
        float ndc_y = local_y / half_height;
        
        if (fabs(ndc_x) > 1.2f || fabs(ndc_y) > 1.2f) return false;  // Outside frustum
        
        screen_pos.x = (ndc_x * 0.5f + 0.5f) * screen_w;
        screen_pos.y = (0.5f - ndc_y * 0.5f) * screen_h;
        return true;
    };
    
    // Draw each non-active camera
    for (size_t i = 0; i < ctx.scene.cameras.size(); ++i) {
        if (i == ctx.scene.active_camera_index) continue;  // Skip active camera
        
        auto& cam = ctx.scene.cameras[i];
        if (!cam) continue;
        
        ImVec2 screen_pos;
        if (!Project(cam->lookfrom, screen_pos)) continue;
        
        // Check if this camera is selected
        bool is_selected = (ctx.selection.hasSelection() && 
                           ctx.selection.selected.type == SelectableType::Camera &&
                           ctx.selection.selected.camera == cam);
        
        // Camera icon colors
        ImU32 body_color = is_selected ? IM_COL32(255, 200, 50, 255) : IM_COL32(100, 150, 255, 200);
        ImU32 lens_color = is_selected ? IM_COL32(255, 220, 100, 255) : IM_COL32(50, 100, 200, 200);
        ImU32 outline_color = is_selected ? IM_COL32(255, 255, 255, 255) : IM_COL32(200, 200, 200, 150);
        
        float size = 15.0f;  // Icon size
        
        // Draw camera body (rectangle)
        ImVec2 body_min(screen_pos.x - size, screen_pos.y - size * 0.6f);
        ImVec2 body_max(screen_pos.x + size * 0.5f, screen_pos.y + size * 0.6f);
        draw_list->AddRectFilled(body_min, body_max, body_color, 3.0f);
        draw_list->AddRect(body_min, body_max, outline_color, 3.0f, 0, 1.5f);
        
        // Draw lens (triangle/cone pointing in look direction)
        Vec3 look_dir = (cam->lookat - cam->lookfrom).normalize();
        ImVec2 look_screen;
        Vec3 lens_tip = cam->lookfrom + look_dir * 0.5f;
        
        // Simple lens representation (triangle pointing forward)
        ImVec2 lens_points[3] = {
            ImVec2(screen_pos.x + size * 0.5f, screen_pos.y - size * 0.4f),
            ImVec2(screen_pos.x + size * 0.5f, screen_pos.y + size * 0.4f),
            ImVec2(screen_pos.x + size * 1.2f, screen_pos.y)
        };
        draw_list->AddTriangleFilled(lens_points[0], lens_points[1], lens_points[2], lens_color);
        draw_list->AddTriangle(lens_points[0], lens_points[1], lens_points[2], outline_color, 1.5f);
        
        // Draw film reel circles (decoration)
        draw_list->AddCircleFilled(ImVec2(screen_pos.x - size * 0.5f, screen_pos.y - size * 0.9f), size * 0.35f, body_color);
        draw_list->AddCircle(ImVec2(screen_pos.x - size * 0.5f, screen_pos.y - size * 0.9f), size * 0.35f, outline_color, 0, 1.0f);
        draw_list->AddCircleFilled(ImVec2(screen_pos.x, screen_pos.y - size * 0.9f), size * 0.35f, body_color);
        draw_list->AddCircle(ImVec2(screen_pos.x, screen_pos.y - size * 0.9f), size * 0.35f, outline_color, 0, 1.0f);
        
        // Camera index label
        std::string label = "Cam " + std::to_string(i);
        ImVec2 text_pos(screen_pos.x - size, screen_pos.y + size * 0.9f);
        draw_list->AddText(text_pos, IM_COL32(255, 255, 255, 200), label.c_str());
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

void SceneUI::handleMouseSelection(UIContext& ctx) {
    // Only select if not interacting with UI or Gizmo
    if (ImGui::IsMouseClicked(0)) {
        
        // Debugging: Confirm click detection
        SCENE_LOG_INFO("Mouse Left Click Detected");

        // Ignore click if over UI elements or Gizmo
        if (ImGui::IsAnyItemHovered() || ImGuizmo::IsOver()) {
            SCENE_LOG_INFO("Selection Skipped: Mouse over UI or Gizmo");
            return;
        }

        int x, y;
        SDL_GetMouseState(&x, &y);
        
        float win_w = ImGui::GetIO().DisplaySize.x;
        float win_h = ImGui::GetIO().DisplaySize.y;
        
        float u = (float)x / win_w;
        float v = (float)y / win_h; 
        v = 1.0f - v; 

        if (ctx.scene.camera) {
            Ray r = ctx.scene.camera->get_ray(u, v);
            
            // Check for Light Selection first (Bounding Sphere Intersection)
            std::shared_ptr<Light> closest_light = nullptr;
            float closest_t = 1e9f;
            
            for (auto& light : ctx.scene.lights) {
                if (!light) continue;
                
                // Proxy Sphere at light position - smaller radius for precise selection
                Vec3 oc = r.origin - light->position;
                float radius = 0.2f;  // Reduced from 0.5 to not block nearby objects
                float a = r.direction.dot(r.direction);
                float half_b = oc.dot(r.direction);
                float c = oc.dot(oc) - radius*radius;
                float discriminant = half_b*half_b - a*c;
                
                if (discriminant > 0) {
                    float root = sqrt(discriminant);
                    float temp = (-half_b - root) / a;
                    if (temp < closest_t && temp > 0.001f) {
                        closest_t = temp;
                        closest_light = light;
                    }
                    temp = (-half_b + root) / a;
                    if (temp < closest_t && temp > 0.001f) {
                        closest_t = temp;
                        closest_light = light;
                    }
                }
            }
            
            // Check for Camera Selection (non-active cameras only)
            std::shared_ptr<Camera> closest_camera = nullptr;
            float closest_camera_t = closest_t;  // Must be closer than light
            
            for (size_t i = 0; i < ctx.scene.cameras.size(); ++i) {
                if (i == ctx.scene.active_camera_index) continue;  // Skip active camera
                
                auto& cam = ctx.scene.cameras[i];
                if (!cam) continue;
                
                // Camera selection sphere
                Vec3 oc = r.origin - cam->lookfrom;
                float radius = 0.4f;  // Reduced from 1.0 to not block nearby objects
                float a = r.direction.dot(r.direction);
                float half_b = oc.dot(r.direction);
                float c = oc.dot(oc) - radius * radius;
                float discriminant = half_b * half_b - a * c;
                
                if (discriminant > 0) {
                    float root = sqrt(discriminant);
                    float temp = (-half_b - root) / a;
                    if (temp < closest_camera_t && temp > 0.001f) {
                        closest_camera_t = temp;
                        closest_camera = cam;
                    }
                    temp = (-half_b + root) / a;
                    if (temp < closest_camera_t && temp > 0.001f) {
                        closest_camera_t = temp;
                        closest_camera = cam;
                    }
                }
            }
            
            // Perform Linear Selection (Bypasses BVH for accuracy and avoids rebuilds)
            
            HitRecord rec;
            bool hit = false;
            float closest_so_far = 1e9f;
            HitRecord temp_rec;

            for (const auto& obj : ctx.scene.world.objects) {
                if (obj->hit(r, 0.001f, closest_so_far, temp_rec)) {
                    hit = true;
                    closest_so_far = temp_rec.t;
                    rec = temp_rec;
                }
            }

            if (hit && rec.triangle && (rec.t < closest_t)) {
                // Log selection info
                // SCENE_LOG_INFO("Selected: " + (rec.triangle->nodeName.empty() ? "Object" : rec.triangle->nodeName));

                std::shared_ptr<Triangle> found_tri = nullptr;
                int index = -1;
                
                // Ensure cache is valid
                if (!mesh_cache_valid) rebuildMeshCache(ctx.scene.world.objects);
                
                bool found = false;
                for (auto& [name, list] : mesh_cache) {
                    for (auto& pair : list) {
                        if (pair.second.get() == rec.triangle) {
                            found_tri = pair.second;
                            index = pair.first;
                            found = true;
                            break;
                        }
                    }
                    if (found) break;
                }

                if (found_tri) {
                    ctx.selection.selectObject(found_tri, index, found_tri->nodeName);
                } else {
                     // Fallback if not in cache (should typically not happen)
                     SCENE_LOG_WARN("Selection: Object found but not in cache.");
                }
            } 
            else if (closest_camera && closest_camera_t < closest_t) {
                 // Camera is closer than light
                 ctx.selection.selectCamera(closest_camera);
                 SCENE_LOG_INFO("Selected Camera");
            }
            else if (closest_light) {
                 ctx.selection.selectLight(closest_light);
                 SCENE_LOG_INFO("Selected Light");
            }
            else {
                // Clicked on empty space
                ctx.selection.clearSelection();
            }
        }
    }
}

// Global flag for Render Window visibility
bool show_render_window = false;

void DrawRenderWindow(UIContext& ctx) {
    if (!show_render_window) return;

    ImGui::SetNextWindowSize(ImVec2(800, 600), ImGuiCond_FirstUseEver);
    
    // Auto-Stop Logic
    int current_samples = ctx.renderer.getCPUAccumulatedSamples();
    int target_samples = ctx.render_settings.final_render_samples;

    if (ctx.render_settings.is_final_render_mode && current_samples >= target_samples) {
        ctx.render_settings.is_final_render_mode = false; // Finish
        extern std::atomic<bool> rendering_stopped_cpu;
        rendering_stopped_cpu = true; 
    }

    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.08f, 0.08f, 0.08f, 1.0f)); // Opaque Background
    if (ImGui::Begin("Render Result", &show_render_window, ImGuiWindowFlags_NoCollapse)) {
        
        // Progress Info
        float progress = (float)current_samples / (float)target_samples;
        if (progress > 1.0f) progress = 1.0f;

        // Header
        ImGui::Text("Final Render Status");
        ImGui::SameLine();
        if (current_samples >= target_samples) {
            ImGui::TextColored(ImVec4(0,1,0,1), "[FINISHED]");
        } else if (ctx.render_settings.is_final_render_mode) {
             ImGui::TextColored(ImVec4(1,1,0,1), "[RENDERING...]");
        } else {
             ImGui::TextColored(ImVec4(0.7f,0.7f,0.7f,1), "[IDLE]");
        }

        // Progress Bar
        char buf[32];
        sprintf(buf, "%d / %d Samples", current_samples, target_samples);
        ImGui::ProgressBar(progress, ImVec2(-1, 0), buf);
        
        ImGui::Separator();

        // Control Toolbar
        if (ImGui::Button("Save Image")) {
             std::string filename = "Render_" + std::to_string(time(0)) + ".png";
             ctx.render_settings.save_image_requested = true;
        }
        
        ImGui::SameLine();
        
        if (ctx.render_settings.is_final_render_mode) {
            if (ImGui::Button("Stop / Cancel")) {
                ctx.render_settings.is_final_render_mode = false;
            }
        } else {
            if (ImGui::Button("Start Render (F12)")) {
                ctx.renderer.resetCPUAccumulation();
                if (ctx.optix_gpu_ptr) ctx.optix_gpu_ptr->resetAccumulation();
                ctx.render_settings.is_final_render_mode = true;
                ctx.start_render = true; 
            }
        }

        ImGui::SameLine();
        ImGui::TextDisabled("|");
        ImGui::SameLine();
        
        // ZOOM CONTROLS
        static float zoom = 1.0f;
        if (ImGui::Button("1:1")) zoom = 1.0f;
        ImGui::SameLine();
        if (ImGui::Button("Fit")) {
             extern int image_width, image_height;
             ImVec2 avail = ImGui::GetContentRegionAvail();
             if (image_width > 0 && image_height > 0) {
                 float rX = avail.x / image_width;
                 float rY = avail.y / image_height;
                 zoom = (rX < rY) ? rX : rY;
             }
        }
        ImGui::SameLine();
        ImGui::SetNextItemWidth(100);
        ImGui::SliderFloat("Zoom", &zoom, 0.1f, 5.0f, "%.1fx");

        ImGui::Separator();
        
        // Render Output Display (Scrollable & Zoomable)
        ImGui::BeginChild("RenderView", ImVec2(0, 0), true, ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_NoMove);
        
        extern SDL_Texture* raytrace_texture; // Global texture from Main.cpp
        extern int image_width, image_height;
        
        if (raytrace_texture && image_width > 0 && image_height > 0) {
             
             ImGuiIO& io = ImGui::GetIO();
             
             // Mouse Wheel Zoom
             if (ImGui::IsWindowHovered()) {
                 if (io.MouseWheel != 0.0f) {
                     float old_zoom = zoom;
                     zoom += io.MouseWheel * 0.1f * zoom; // Logarithmic-ish zoom
                     if (zoom < 0.1f) zoom = 0.1f;
                     if (zoom > 10.0f) zoom = 10.0f;
                     
                     // Center zoom (optional, simplified for now)
                 }
                 
                 // Middle Mouse Pan
                 if (ImGui::IsMouseDragging(ImGuiMouseButton_Middle)) {
                     ImVec2 delta = io.MouseDelta;
                     ImGui::SetScrollX(ImGui::GetScrollX() - delta.x);
                     ImGui::SetScrollY(ImGui::GetScrollY() - delta.y);
                 }
             }

             // Calculate Scaled Size
             float w = (float)image_width * zoom;
             float h = (float)image_height * zoom;
             
             // Center Image if smaller than window
             ImVec2 avail = ImGui::GetContentRegionAvail();
             float offX = (avail.x > w) ? (avail.x - w) * 0.5f : 0.0f;
             float offY = (avail.y > h) ? (avail.y - h) * 0.5f : 0.0f;
             
             if (offX > 0) ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offX);
             if (offY > 0) ImGui::SetCursorPosY(ImGui::GetCursorPosY() + offY);

             ImGui::Image((ImTextureID)raytrace_texture, ImVec2(w, h));
             
             // Tooltip for resolution
             if (ImGui::IsItemHovered()) {
                 ImGui::SetTooltip("Resolution: %dx%d | Zoom: %.1f%%", image_width, image_height, zoom * 100.0f);
             }

        } else {
             ImGui::TextColored(ImVec4(1,0,0,1), "Render Texture not available.");
        }
        ImGui::EndChild();
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
    

