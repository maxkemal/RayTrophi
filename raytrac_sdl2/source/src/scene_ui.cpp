#include "scene_ui.h"
#include "ui_modern.h"
#include "imgui.h"
#include "ImGuizmo.h"  // Transform gizmo
#include <string>
#include <memory>  // For std::make_unique
#include "KeyframeSystem.h"   // For keyframe animation
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
bool show_animation_panel = false; // Default closed as requested

// Pivot Mode State: 0=Median Point (Group), 1=Individual Origins
static int pivot_mode = 0; 

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




std::string openFileDialogW(const wchar_t* filter = L"All Files\0*.*\0", const std::string& initialDir = "", const std::string& defaultFilename = "") {
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

std::string saveFileDialogW(const wchar_t* filter = L"All Files\0*.*\0", const wchar_t* defExt = L"rts") {
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
    // Use the new modular TimelineWidget
    static TimelineWidget timeline;
    timeline.draw(ctx);
}

// ========== OLD TIMELINE CODE REMOVED - NOW IN TimelineWidget.cpp ==========
// The following code is kept temporarily for reference but is no longer active
#if 0
    // Static variables for timeline state
    static int currentFrame = 0;
    static bool expanded = true;
    static int selectedEntry = -1;
    static int firstFrame = 0;
    static bool frame_range_initialized = false;
    static bool is_playing = false;
    static auto last_frame_time = std::chrono::steady_clock::now();
    
    // Determine frame range from animation data or timeline
    static int start_frame = 0;     // Static to persist across frames
    static int end_frame = 250;     // Blender-like default 
  
    static int playback_frame = 0;
   
    // Determine frame range from animation data or use default
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

        // Initialize from animation data
        if (!frame_range_initialized) {
            start_frame = anim.startFrame;
            end_frame = anim.endFrame;
            playback_frame = start_frame;
            ctx.render_settings.animation_start_frame = start_frame;
            ctx.render_settings.animation_end_frame = end_frame;
            frame_range_initialized = true;
        }
    } else {
        // No animation data - use Blender-like default (0-250)
        if (!frame_range_initialized) {
            start_frame = 0;
            end_frame = 250;
            playback_frame = 0;
            ctx.render_settings.animation_start_frame = start_frame;
            ctx.render_settings.animation_end_frame = end_frame;
            frame_range_initialized = true;
        }
    }
    
    int total_frames = end_frame - start_frame + 1;
    bool is_rendering = rendering_in_progress;
    
    // --- TOP ROW: Frame Range & Settings ---
    ImGui::PushItemWidth(150);
    if (ImGui::SliderInt("Start", &start_frame, 0, 1000)) {
        ctx.render_settings.animation_start_frame = start_frame;
        if (playback_frame < start_frame) playback_frame = start_frame;
    }
    ImGui::SameLine();
    if (ImGui::SliderInt("End", &end_frame, 0, 1000)) {
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
    
    // --- MIDDLE ROW: Playback Controls ---
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
    
    // Show which object's keyframes are displayed
    bool has_selection = ctx.selection.hasSelection() && 
                        (ctx.selection.selected.type == SelectableType::Object ||
                         ctx.selection.selected.type == SelectableType::Light ||
                         ctx.selection.selected.type == SelectableType::Camera);
    
    if (has_selection) {
        std::string entity_name;
        std::string entity_type_str;
        
        if (ctx.selection.selected.type == SelectableType::Object) {
            auto& obj = ctx.selection.selected.object;
            entity_name = obj->nodeName.empty() ? 
                "Object_" + std::to_string(ctx.selection.selected.object_index) : 
                obj->nodeName;
            entity_type_str = "Object";
        } else if (ctx.selection.selected.type == SelectableType::Light) {
            entity_name = ctx.selection.selected.light->nodeName;
            entity_type_str = "Light";
        } else if (ctx.selection.selected.type == SelectableType::Camera) {
            entity_name = ctx.selection.selected.camera->nodeName;
            entity_type_str = "Camera";
        }
        
        int kf_count = 0;
        if (ctx.scene.timeline.tracks.find(entity_name) != ctx.scene.timeline.tracks.end()) {
            kf_count = ctx.scene.timeline.tracks[entity_name].keyframes.size();
        }
        
        ImGui::Text("Editing:");
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.4f, 1.0f), "%s", entity_name.c_str());
        ImGui::SameLine();
        ImGui::TextDisabled("(%s, %d keyframes)", entity_type_str.c_str(), kf_count);
    } else {
        ImGui::TextDisabled("No entity selected");
    }
    
    // Keyframe Insertion Button
    if (!has_selection) ImGui::BeginDisabled();
    
    if (UIWidgets::PrimaryButton("Insert Keyframe (I)", ImVec2(140, 24))) {
        if (has_selection) {
            // Open popup menu (Blender-style)
            ImGui::OpenPopup("KeyframePropertyPopup");
        }
    }
    
    if (!has_selection) ImGui::EndDisabled();
    
    ImGui::SameLine();
    
    
    // Popup Menu for Keyframe Property Selection (Blender-style)
    if (ImGui::BeginPopup("KeyframePropertyPopup")) {
        Keyframe kf(playback_frame);
        std::string entity_name;
        std::string props_desc;
        bool should_insert = false;
        
        // OBJECT: Show transform options
        if (ctx.selection.selected.type == SelectableType::Object) {
            auto& obj = ctx.selection.selected.object;
            entity_name = obj->nodeName.empty() ? 
                "Object_" + std::to_string(ctx.selection.selected.object_index) : 
                obj->nodeName;
            
            ImGui::TextDisabled("Transform:");
            ImGui::Separator();
            
            if (ImGui::MenuItem("Location")) {
                kf.has_transform = true;
                // ALWAYS capture ALL transform properties to preserve un-keyframed ones
                kf.transform.position = ctx.selection.selected.position;
                kf.transform.rotation = ctx.selection.selected.rotation;
                kf.transform.scale = ctx.selection.selected.scale;
                props_desc = "Location";
                should_insert = true;
            }
            if (ImGui::MenuItem("Rotation")) {
                kf.has_transform = true;
                // ALWAYS capture ALL transform properties
                kf.transform.position = ctx.selection.selected.position;
                kf.transform.rotation = ctx.selection.selected.rotation;
                kf.transform.scale = ctx.selection.selected.scale;
                props_desc = "Rotation";
                should_insert = true;
            }
            if (ImGui::MenuItem("Scale")) {
                kf.has_transform = true;
                // ALWAYS capture ALL transform properties
                kf.transform.position = ctx.selection.selected.position;
                kf.transform.rotation = ctx.selection.selected.rotation;
                kf.transform.scale = ctx.selection.selected.scale;
                props_desc = "Scale";
                should_insert = true;
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Location & Rotation")) {
                kf.has_transform = true;
                kf.transform.position = ctx.selection.selected.position;
                kf.transform.rotation = ctx.selection.selected.rotation;
                kf.transform.scale = ctx.selection.selected.scale;
                props_desc = "Location+Rotation";
                should_insert = true;
            }
            if (ImGui::MenuItem("Location & Rotation & Scale")) {
                kf.has_transform = true;
                kf.transform.position = ctx.selection.selected.position;
                kf.transform.rotation = ctx.selection.selected.rotation;
                kf.transform.scale = ctx.selection.selected.scale;
                props_desc = "Location+Rotation+Scale";
                should_insert = true;
            }
        }
        // LIGHT: Auto-capture all properties
        else if (ctx.selection.selected.type == SelectableType::Light) {
            auto& light = ctx.selection.selected.light;
            entity_name = light->nodeName;
            
            ImGui::TextDisabled("Light Properties:");
            ImGui::Separator();
            
            if (ImGui::MenuItem("All Light Properties")) {
                kf.has_light = true;
                kf.light.position = light->position;
                kf.light.color = light->color;
                kf.light.intensity = light->intensity;
                kf.light.direction = light->direction;
                props_desc = "Position+Color+Intensity+Direction";
                should_insert = true;
            }
        }
        // CAMERA: Auto-capture all properties
        else if (ctx.selection.selected.type == SelectableType::Camera) {
            auto& camera = ctx.selection.selected.camera;
            entity_name = camera->nodeName;
            
            ImGui::TextDisabled("Camera Properties:");
            ImGui::Separator();
            
            if (ImGui::MenuItem("All Camera Properties")) {
                kf.has_camera = true;
                kf.camera.position = camera->lookfrom;
                kf.camera.target = camera->lookat;
                kf.camera.fov = camera->vfov;
                kf.camera.focus_distance = camera->focus_dist;
                kf.camera.lens_radius = camera->lens_radius;
                props_desc = "Position+Target+FOV+DOF";
                should_insert = true;
            }
        }
        
        // Insert if user selected an option
        if (should_insert) {
            ctx.scene.timeline.insertKeyframe(entity_name, kf);
            
            // DEBUG: Log transform values for debugging
            if (kf.has_transform) {
                SCENE_LOG_INFO(">>> Transform Debug: Pos(" + 
                    std::to_string(kf.transform.position.x) + "," + 
                    std::to_string(kf.transform.position.y) + "," + 
                    std::to_string(kf.transform.position.z) + ") Rot(" +
                    std::to_string(kf.transform.rotation.x) + "," + 
                    std::to_string(kf.transform.rotation.y) + "," + 
                    std::to_string(kf.transform.rotation.z) + ") Scale(" +
                    std::to_string(kf.transform.scale.x) + "," + 
                    std::to_string(kf.transform.scale.y) + "," + 
                    std::to_string(kf.transform.scale.z) + ")");
            }
            
            SCENE_LOG_INFO("Keyframe [" + props_desc + "] inserted for '" + entity_name + 
                          "' at frame " + std::to_string(playback_frame));
            ImGui::CloseCurrentPopup();
        }
        
        ImGui::EndPopup();
    }
    
    // Delete Keyframe button
    bool has_keyframe_at_current = false;
    std::string entity_name_for_delete;
    
    if (has_selection) {
        // Get entity name based on type (avoid null pointer for Light/Camera)
        if (ctx.selection.selected.type == SelectableType::Object) {
            auto& obj = ctx.selection.selected.object;
            entity_name_for_delete = obj->nodeName.empty() ? 
                "Object_" + std::to_string(ctx.selection.selected.object_index) : 
                obj->nodeName;
        } else if (ctx.selection.selected.type == SelectableType::Light) {
            entity_name_for_delete = ctx.selection.selected.light->nodeName;
        } else if (ctx.selection.selected.type == SelectableType::Camera) {
            entity_name_for_delete = ctx.selection.selected.camera->nodeName;
        }
        
        auto& track = ctx.scene.timeline.getTrack(entity_name_for_delete);
        has_keyframe_at_current = (track.getKeyframeAt(playback_frame) != nullptr);
    }
    
    if (!has_keyframe_at_current) ImGui::BeginDisabled();
    if (UIWidgets::DangerButton("Delete Keyframe", ImVec2(120, 24))) {
        ctx.scene.timeline.removeKeyframe(entity_name_for_delete, playback_frame);
        SCENE_LOG_INFO("Keyframe deleted for '" + entity_name_for_delete + "' at frame " + 
                      std::to_string(playback_frame));
    }
    if (!has_keyframe_at_current) ImGui::EndDisabled();
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::SetTooltip(has_keyframe_at_current ? 
            "Delete keyframe at current frame" : 
            "No keyframe at current frame");
    }
    
    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();
    
    // Current Frame Display
    ImGui::AlignTextToFramePadding();
    ImGui::Text("Frame:");
    ImGui::SameLine();
    
    ImGui::PushItemWidth(80);
    if (ImGui::InputInt("##CurrentFrame", &playback_frame, 0, 0)) {
        playback_frame = std::clamp(playback_frame, start_frame, end_frame);
    }
    ImGui::PopItemWidth();
    
    // --- ALWAYS SHOW TIMELINE SCRUBBER (for navigation) ---
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("Timeline:");
    
    // Main scrubber slider - ALWAYS visible for frame navigation
    float slider_width = ImGui::GetContentRegionAvail().x - 10;
    ImVec2 slider_pos = ImGui::GetCursorScreenPos();
    
    ImGui::PushItemWidth(slider_width);
    if (ImGui::SliderInt("##Scrubber", &playback_frame, start_frame, end_frame, "Frame %d")) {
        // Scrubber moved - will be handled by frame change logic below
    }
    ImGui::PopItemWidth();
    
    // Draw keyframe markers on top of slider (for selected entity)
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    if (has_selection) {
        std::string selected_entity_name;
        
        if (ctx.selection.selected.type == SelectableType::Object) {
            auto& obj = ctx.selection.selected.object;
            selected_entity_name = obj->nodeName.empty() ? 
                "Object_" + std::to_string(ctx.selection.selected.object_index) : 
                obj->nodeName;
        } else if (ctx.selection.selected.type == SelectableType::Light) {
            selected_entity_name = ctx.selection.selected.light->nodeName;
        } else if (ctx.selection.selected.type == SelectableType::Camera) {
            selected_entity_name = ctx.selection.selected.camera->nodeName;
        }
        
        auto track_it = ctx.scene.timeline.tracks.find(selected_entity_name);
        if (track_it != ctx.scene.timeline.tracks.end()) {
            auto& track = track_it->second;
            
            for (auto& kf : track.keyframes) {
                if (kf.frame >= start_frame && kf.frame <= end_frame) {
                    float t = (float)(kf.frame - start_frame) / (float)(end_frame - start_frame);
                    float marker_x = slider_pos.x + t * slider_width;
                    float marker_y = slider_pos.y + 10;
                    
                    ImVec2 p1(marker_x, marker_y - 6);
                    ImVec2 p2(marker_x + 4, marker_y);
                    ImVec2 p3(marker_x, marker_y + 6);
                    ImVec2 p4(marker_x - 4, marker_y);
                    
                    ImU32 color;
                    if (kf.has_transform && kf.has_material) {
                        color = IM_COL32(255, 200, 50, 255);
                    } else if (kf.has_transform) {
                        color = IM_COL32(100, 200, 255, 255);
                    } else if (kf.has_light) {
                        color = IM_COL32(255, 150, 100, 255);
                    } else if (kf.has_camera) {
                        color = IM_COL32(150, 100, 255, 255);
                    } else {
                        color = IM_COL32(255, 100, 150, 255);
                    }
                    
                    draw_list->AddQuadFilled(p1, p2, p3, p4, color);
                    draw_list->AddQuad(p1, p2, p3, p4, IM_COL32(0, 0, 0, 255), 1.5f);
                }
            }
        }
    }
    
    // Keyframe legend
    ImGui::Spacing();
    ImGui::TextDisabled("Keyframes:");
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "◆");
    ImGui::SameLine();
    ImGui::Text("Transform");
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.4f, 1.0f), "◆");
    ImGui::SameLine();
    ImGui::Text("Light");
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.6f, 0.4f, 1.0f, 1.0f), "◆");
    ImGui::SameLine();
    ImGui::Text("Camera");
    
    // Show track count info
    size_t track_count = ctx.scene.timeline.tracks.size();
    if (track_count > 0) {
        ImGui::SameLine();
        ImGui::TextDisabled("| %zu tracks", track_count);
    }
    
    // Playback Logic Update
    static int last_evaluated_frame = -1;
    bool frame_changed = false;
    
    if (is_playing && !is_rendering) {
        auto now = std::chrono::steady_clock::now();
        float elapsed = std::chrono::duration<float>(now - last_frame_time).count();
        float frame_duration = 1.0f / (float)ctx.render_settings.animation_fps;
        
        if (elapsed >= frame_duration) {
            playback_frame++;
            if (playback_frame > end_frame) playback_frame = start_frame;
            last_frame_time = now;
            frame_changed = true;
        }
    }
    
    // Check if frame changed manually (scrubbing)
    if (playback_frame != last_evaluated_frame) {
        frame_changed = true;
        last_evaluated_frame = playback_frame;
    }
    
    // Apply keyframe transforms when frame changes
    if (frame_changed && !ctx.scene.timeline.tracks.empty()) {
        bool scene_modified = false;
        
        // Evaluate and apply keyframes for all objects
        for (auto& [obj_name, track] : ctx.scene.timeline.tracks) {
            if (track.keyframes.empty()) continue;
            
            // Evaluate keyframe at current frame
            Keyframe kf = track.evaluate(playback_frame);
            
            // DEBUG: Log what we're looking for
            static bool first_log = true;
            if (first_log) {
                SCENE_LOG_INFO(">>> Playback Debug: Looking for object '" + obj_name + "'");
                first_log = false;
            }
            
            // Find the object in the scene by name
            bool obj_found = false;
            for (size_t i = 0; i < ctx.scene.world.objects.size(); ++i) {
                auto tri = std::dynamic_pointer_cast<Triangle>(ctx.scene.world.objects[i]);
                if (!tri) continue;
                
                // DEBUG: Log each object we check
                if (first_log) {
                    SCENE_LOG_INFO(">>> Checking object: nodeName='" + tri->nodeName + "'");
                }
                
                if (tri->nodeName == obj_name) {
                    obj_found = true;
                    // Apply transform keyframe
                    if (kf.has_transform) {
                        // Get current transform matrix to preserve un-keyframed properties
                        Matrix4x4 current_transform = tri->getTransformMatrix();
                        
                        // Decompose or use keyframe values directly
                        Vec3 position = kf.transform.position;
                        Vec3 rotation = kf.transform.rotation;
                        Vec3 scale = kf.transform.scale;
                        
                        // Build transform matrix from keyframe
                        Matrix4x4 translation = Matrix4x4::translation(position);
                        
                        // Convert Euler angles to rotation matrix (deg to rad)
                        float rx = rotation.x * (3.14159265f / 180.0f);
                        float ry = rotation.y * (3.14159265f / 180.0f);
                        float rz = rotation.z * (3.14159265f / 180.0f);
                        
                        Matrix4x4 rot = Matrix4x4::rotationZ(rz) * 
                                       Matrix4x4::rotationY(ry) * 
                                       Matrix4x4::rotationX(rx);
                        
                        Matrix4x4 scl = Matrix4x4::scaling(scale);
                        
                        // Combine: T * R * S
                        Matrix4x4 final_transform = translation * rot * scl;
                        
                        // Apply to triangle
                        tri->set_transform(final_transform);
                        
                        // IMPORTANT: Update SelectableItem if this is the current selection
                        // This keeps gizmo and bounding box in sync
                        if (ctx.selection.hasSelection() && 
                            ctx.selection.selected.type == SelectableType::Object &&
                            ctx.selection.selected.object == tri) {
                            ctx.selection.selected.position = position;
                            ctx.selection.selected.rotation = rotation;
                            ctx.selection.selected.scale = scale;
                            ctx.selection.selected.has_cached_aabb = false;  // Invalidate cache
                        }
                        
                        scene_modified = true;
                    }
                    
                    // TODO: Apply material keyframe when material system is integrated
                    break;
                }
            }
        }
        
        // Trigger scene rebuild if modified
        if (scene_modified) {
            // For transform-only changes, we only need BVH rebuild
            // OptiX geometry rebuild is NOT needed (geometry unchanged, only transforms)
            ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
            ctx.renderer.resetCPUAccumulation();
            
            // NOTE: OptiX rebuild skipped for performance during keyframe playback
            // Geometry hasn't changed, only transforms. BVH handles this.
            // If we add material keyframes later, we'll need conditional OptiX rebuild
        }
    }
    
    // Store user state
    ctx.render_settings.animation_is_playing = is_playing;
    ctx.render_settings.animation_playback_frame = playback_frame;
    ctx.render_settings.animation_current_frame = playback_frame;
    ctx.scene.timeline.current_frame = playback_frame;
}
#endif

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
            
            // Mouse Sensitivity (for camera navigation)
            ImGui::Spacing();
            UIWidgets::SliderWithHelp("Mouse Sensitivity", &ctx.mouse_sensitivity, 0.01f, 5.0f,
                                       "Camera rotation/panning speed", "%.3f");
        } else {
            ImGui::TextDisabled("DOF disabled (Aperture = 0)");
            
            // Mouse Sensitivity visible even when DOF disabled
            ImGui::Spacing();
            UIWidgets::SliderWithHelp("Mouse Sensitivity", &ctx.mouse_sensitivity, 0.01f, 5.0f,
                                       "Camera rotation/panning speed", "%.3f");
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
            int current_type = (int)light->type();
            int new_type = current_type;

            if (ImGui::Combo("Type", &new_type, names, IM_ARRAYSIZE(names))) {
                if (new_type != current_type) {
                    // Prepare common data
                    Vec3 pos = light->position;
                    Vec3 col = light->color;
                    float inten = light->intensity;
                    std::string name = light->nodeName;
                    Vec3 dir = light->direction;
                    // Default direction if switching from Point (which might have 0,0,0 dir)
                    if (dir.length_squared() < 0.001f) dir = Vec3(0, -1, 0); 
                    
                    std::shared_ptr<Light> new_light = nullptr;

                    switch (new_type) {
                        case 0: // Point
                            new_light = std::make_shared<PointLight>(pos, col, inten);
                            new_light->radius = light->radius; 
                            break;
                        case 1: // Directional
                            new_light = std::make_shared<DirectionalLight>(dir, col, inten);
                            new_light->radius = light->radius; // Preserves soft shadow size if applicable
                            break;
                        case 2: // Spot
                            // Use reasonable defaults for Angle/Falloff
                            new_light = std::make_shared<SpotLight>(pos, dir, col, 45.0f, 10.0f);
                            new_light->intensity = inten;
                            break;
                        case 3: // Area
                            // Default 2x2 area facing +Z initially, will be overridden by logic potentially
                            new_light = std::make_shared<AreaLight>(pos, Vec3(2,0,0), Vec3(0,0,2), 2.0f, 2.0f, col);
                            // AreaLight constructor sig: pos, u, v, w, h, color
                            // Logic inside AreaLight usually sets internal u/v based on w/h.
                            new_light->intensity = inten;
                            break;
                    }

                    if (new_light) {
                        new_light->nodeName = name;
                        
                        // Update Scene List
                        ctx.scene.lights[i] = new_light;
                        
                        // Update Selection if necessary
                        if (ctx.selection.selected.type == SelectableType::Light && ctx.selection.selected.light == light) {
                             ctx.selection.selected.light = new_light;
                             // No need to reset index, it's the same i
                        }
                        
                        light = new_light; // Update local pointer for rest of this frame's UI
                        changed = true;
                    }
                }
            }

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
// drawWorldContent moved to scene_ui_world.cpp

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
    ImGui::PopStyleColor(); // Border
    ImGui::PopStyleVar();   // BorderSize
}

// Main Menu Bar implementation moved to separate file: scene_ui_menu.hpp check end of file

#include "scene_ui_menu.hpp"

void SceneUI::draw(UIContext& ctx)
{
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

    handleSceneInteraction(ctx, gizmo_hit);
    processDeferredSceneUpdates(ctx);
    drawAuxWindows(ctx);

    extern void DrawRenderWindow(UIContext & ctx);
    DrawRenderWindow(ctx);
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
    ImGui::PushStyleColor(ImGuiCol_WindowBg,
        ImVec4(0.08f, 0.08f, 0.08f, 1.0f));

    if (ImGui::Begin("StatusBar", nullptr,
        ImGuiWindowFlags_NoDecoration |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoBringToFrontOnFocus))
    {
        extern bool show_animation_panel;
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
            ImGui::Text("Scene: %d Objects, %d Lights",
                (int)ctx.scene.world.objects.size(),
                (int)ctx.scene.lights.size());
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

    ImGui::PopStyleColor();
    ImGui::PopStyleVar(2);

    // ---------------- BOTTOM PANEL (Resizable) ----------------
    extern bool show_animation_panel;
    bool show_bottom = (show_animation_panel || show_scene_log);
    if (!show_bottom) return;

    // Static height that persists across frames (user can resize)
    static float bottom_height = 280.0f;
    const float min_height = 100.0f;
    const float max_height = screen_y * 0.6f;  // Max 60% of screen
    const float resize_handle_height = 6.0f;
    
    // Clamp height to valid range
    bottom_height = std::clamp(bottom_height, min_height, max_height);

    // Calculate panel position
    float panel_top = screen_y - bottom_height - status_bar_height;
    
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
            bottom_height -= ImGui::GetIO().MouseDelta.y;
            bottom_height = std::clamp(bottom_height, min_height, max_height);
        }
    }
    ImGui::End();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();

    // --- MAIN BOTTOM PANEL ---
    ImGui::SetNextWindowPos(ImVec2(0, panel_top), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(screen_x, bottom_height), ImGuiCond_Always);

    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.12f, 0.12f, 0.15f, 1.0f));

    if (ImGui::Begin("BottomPanel", nullptr,
        ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoCollapse))
    {
        if (ImGui::BeginTabBar("BottomTabs")) {

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
bool SceneUI::drawOverlays(UIContext& ctx)
{
    bool gizmo_hit = false;

    if (ctx.scene.camera && ctx.selection.show_gizmo) {
        drawLightGizmos(ctx, gizmo_hit);
    }

    return gizmo_hit;
}
void SceneUI::drawLightGizmos(UIContext& ctx, bool& gizmo_hit)
{
    ImDrawList* draw_list = ImGui::GetBackgroundDrawList();
    ImGuiIO& io = ImGui::GetIO();
    Camera& cam = *ctx.scene.camera;

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

        float half_h = depth * tan_half_fov;
        float half_w = half_h * aspect;

        return ImVec2(
            ((local_x / half_w) * 0.5f + 0.5f) * io.DisplaySize.x,
            (0.5f - (local_y / half_h) * 0.5f) * io.DisplaySize.y
        );
        };

    auto IsOnScreen = [](const ImVec2& v) { return v.x > -5000; };

    for (auto& light : ctx.scene.lights) {

        bool selected =
            (ctx.selection.selected.type == SelectableType::Light &&
                ctx.selection.selected.light == light);

        ImU32 col = selected
            ? IM_COL32(255, 100, 50, 255)
            : IM_COL32(255, 255, 100, 180);

        Vec3 pos = light->position;
        ImVec2 center = Project(pos);
        bool visible = IsOnScreen(center);

        if (!visible) continue;

        // -------- PICKING --------
        float dx = io.MousePos.x - center.x;
        float dy = io.MousePos.y - center.y;
        float d = sqrtf(dx * dx + dy * dy);

        if (d < 20.0f && ImGui::IsMouseClicked(0) && !ImGuizmo::IsOver()) {
            ctx.selection.selectLight(light);
            gizmo_hit = true;
        }

        if (selected) {
            std::string label = light->nodeName.empty() ? "Light" : light->nodeName;
            draw_list->AddText(ImVec2(center.x + 12, center.y - 12), col, label.c_str());
        }

        // ================== DRAW BY TYPE ==================

        // ---- POINT (Diamond) ----
        if (light->type() == LightType::Point) {
            float r = 0.2f;
            Vec3 pts[6] = {
                pos + Vec3(0, r, 0), pos + Vec3(0, -r, 0),
                pos + Vec3(r, 0, 0), pos + Vec3(-r, 0, 0),
                pos + Vec3(0, 0, r), pos + Vec3(0, 0, -r)
            };

            ImVec2 s[6];
            for (int i = 0; i < 6; ++i) s[i] = Project(pts[i]);

            draw_list->AddCircleFilled(center, 4.0f,
                IM_COL32(255, 255, 200, 200));

            draw_list->AddLine(s[2], s[4], col); draw_list->AddLine(s[4], s[3], col);
            draw_list->AddLine(s[3], s[5], col); draw_list->AddLine(s[5], s[2], col);
            draw_list->AddLine(s[0], s[2], col); draw_list->AddLine(s[0], s[3], col);
            draw_list->AddLine(s[0], s[4], col); draw_list->AddLine(s[0], s[5], col);
            draw_list->AddLine(s[1], s[2], col); draw_list->AddLine(s[1], s[3], col);
            draw_list->AddLine(s[1], s[4], col); draw_list->AddLine(s[1], s[5], col);
        }

        // ---- DIRECTIONAL (Sun + Arrow) ----
        else if (light->type() == LightType::Directional) {
            draw_list->AddCircle(center, 8.0f, col, 0, 2.0f);

            for (int i = 0; i < 8; ++i) {
                float a = i * (6.28f / 8.0f);
                ImVec2 dir(cosf(a), sinf(a));
                draw_list->AddLine(
                    ImVec2(center.x + dir.x * 12, center.y + dir.y * 12),
                    ImVec2(center.x + dir.x * 18, center.y + dir.y * 18),
                    col);
            }

            auto dl = std::dynamic_pointer_cast<DirectionalLight>(light);
            if (dl) {
                Vec3 end3d = pos + dl->direction.normalize() * 3.0f;
                ImVec2 end = Project(end3d);
                if (IsOnScreen(end)) {
                    draw_list->AddLine(center, end, col, 2.0f);
                    draw_list->AddCircleFilled(end, 3.0f, col);
                }
            }
        }

        // ---- AREA (Rectangle) ----
        else if (light->type() == LightType::Area) {
            auto al = std::dynamic_pointer_cast<AreaLight>(light);
            if (!al) continue;

            Vec3 u = al->getU();
            Vec3 v = al->getV();

            ImVec2 c1 = Project(pos);
            ImVec2 c2 = Project(pos + u);
            ImVec2 c3 = Project(pos + u + v);
            ImVec2 c4 = Project(pos + v);

            draw_list->AddLine(c1, c2, col);
            draw_list->AddLine(c2, c3, col);
            draw_list->AddLine(c3, c4, col);
            draw_list->AddLine(c4, c1, col);
            draw_list->AddLine(c1, c3, col, 1.0f);
        }

        // ---- SPOT (Cone) ----
        else if (light->type() == LightType::Spot) {
            auto sl = std::dynamic_pointer_cast<SpotLight>(light);
            if (!sl) continue;

            Vec3 dir = sl->direction.normalize();
            float len = 3.0f;
            float radius = len * tanf(sl->getAngleDegrees() * 3.14159f / 360.0f);

            Vec3 base = pos + dir * len;
            Vec3 right = (fabs(dir.y) > 0.9f) ? Vec3(1, 0, 0)
                : dir.cross(Vec3(0, 1, 0)).normalize();
            Vec3 up = right.cross(dir).normalize();

            const int segs = 12;
            ImVec2 last;
            for (int i = 0; i <= segs; ++i) {
                float a = i * (6.28f / segs);
                Vec3 p = base + right * (cosf(a) * radius)
                    + up * (sinf(a) * radius);

                ImVec2 sp = Project(p);
                if (i > 0 && IsOnScreen(sp) && IsOnScreen(last))
                    draw_list->AddLine(last, sp, col);
                if (i < segs && IsOnScreen(sp))
                    draw_list->AddLine(center, sp, col);

                last = sp;
            }
        }
    }
}


bool SceneUI::deleteSelectedLight(UIContext& ctx)
{
    auto light = ctx.selection.selected.light;
    if (!light) return false;

    auto& lights = ctx.scene.lights;
    auto it = std::find(lights.begin(), lights.end(), light);
    if (it == lights.end()) return false;

    history.record(std::make_unique<DeleteLightCommand>(light));
    lights.erase(it);

    // GPU + CPU reset
    if (ctx.optix_gpu_ptr) {
        ctx.optix_gpu_ptr->setLightParams(ctx.scene.lights);
        ctx.optix_gpu_ptr->resetAccumulation();
    }
    ctx.renderer.resetCPUAccumulation();

    SCENE_LOG_INFO("Deleted Light");
    return true;
}
bool SceneUI::deleteSelectedObject(UIContext& ctx)
{
    std::string deleted_name = ctx.selection.selected.name;
    if (deleted_name.empty()) return false;

    auto& objects = ctx.scene.world.objects;
    size_t removed_count = 0;

    objects.erase(
        std::remove_if(objects.begin(), objects.end(),
            [&deleted_name, &removed_count](const std::shared_ptr<Hittable>& obj)
            {
                auto tri = std::dynamic_pointer_cast<Triangle>(obj);
                if (tri && tri->nodeName == deleted_name) {
                    removed_count++;
                    return true;
                }
                return false;
            }),
        objects.end()
    );

    if (removed_count == 0) return false;

    // --- Project data bookkeeping ---
    auto& proj = g_ProjectManager.getProjectData();

    for (auto& model : proj.imported_models) {
        for (const auto& inst : model.objects) {
            if (inst.node_name == deleted_name) {
                if (std::find(model.deleted_objects.begin(),
                    model.deleted_objects.end(),
                    deleted_name) == model.deleted_objects.end()) {
                    model.deleted_objects.push_back(deleted_name);
                }
                break;
            }
        }
    }

    // Remove procedural objects
    auto& procs = proj.procedural_objects;
    procs.erase(
        std::remove_if(procs.begin(), procs.end(),
            [&deleted_name](const ProceduralObjectData& p) {
                return p.display_name == deleted_name;
            }),
        procs.end()
    );

    g_ProjectManager.markModified();

    // --- Rebuild caches & acceleration ---
    rebuildMeshCache(ctx.scene.world.objects);
    ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
    ctx.renderer.resetCPUAccumulation();

    if (ctx.optix_gpu_ptr) {
        ctx.renderer.rebuildOptiXGeometry(ctx.scene, ctx.optix_gpu_ptr);
        ctx.optix_gpu_ptr->resetAccumulation();
    }

    SCENE_LOG_INFO(
        "Deleted: " + deleted_name + " (" +
        std::to_string(removed_count) + " triangles)"
    );

    return true;
}

void SceneUI::handleDeleteShortcut(UIContext& ctx)
{
    if (!ImGui::IsKeyPressed(ImGuiKey_Delete) &&
        !ImGui::IsKeyPressed(ImGuiKey_X)) return;

    bool deleted = false;

    if (ctx.selection.selected.type == SelectableType::Light) {
        deleted = deleteSelectedLight(ctx);
    }
    else if (ctx.selection.selected.type == SelectableType::Object) {
        deleted = deleteSelectedObject(ctx);
    }

    if (deleted) {
        ctx.selection.clearSelection();
    }
}
void SceneUI::drawSelectionGizmos(UIContext& ctx)
{
    if (ctx.selection.hasSelection() && ctx.selection.show_gizmo && ctx.scene.camera) {
        drawSelectionBoundingBox(ctx);
        drawTransformGizmo(ctx);
    }
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
     
     if (ImGui::CollapsingHeader("World Panel")) {
         ImGui::BulletText("Sky Model: Switch between Solid Color, HDRI, or “Raytrophi Spectral Sky.");
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
        triggerDelete(ctx);
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
    
    // Pivot Point Selection
    ImGui::SameLine();
    ImGui::SetNextItemWidth(140);
    const char* pivot_opts[] = { "Median Point", "Individual Origins" };
    if (ImGui::Combo("##Pivot", &pivot_mode, pivot_opts, IM_ARRAYSIZE(pivot_opts))) {
        // Mode switch logic if needed (state is just reading pivot_mode)
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Rotation/Scale Pivot: Group Center vs Individual Objects");

    ImGui::Separator();
    
    // ─────────────────────────────────────────────────────────────────────────
    // Scene Tree
    // ─────────────────────────────────────────────────────────────────────────
    float available_h = ImGui::GetContentRegionAvail().y;
    // Split: 35% for Tree, Rest for Material Editor
    ImGui::BeginChild("HierarchyTree", ImVec2(0, available_h * 0.35f), true);
    
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
            // SELECT ALL / SELECT NONE buttons
            if (ImGui::Button("Select All##lights")) {
                ctx.selection.clearSelection();
                for (size_t i = 0; i < ctx.scene.lights.size(); ++i) {
                    auto& light = ctx.scene.lights[i];
                    SelectableItem item;
                    item.type = SelectableType::Light;
                    item.light = light;
                    item.light_index = (int)i;
                    item.name = "Light_" + std::to_string(i);
                    ctx.selection.addToSelection(item);
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("Select None##lights")) {
                ctx.selection.clearSelection();
            }
            ImGui::SameLine();
            ImGui::Text("(%d lights)", (int)ctx.scene.lights.size());
            
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

        // SELECT ALL / SELECT NONE buttons
        if (ImGui::Button("Select All##obj")) {
            ctx.selection.clearSelection();
            for (auto& [name, triangles] : mesh_cache) {
                if (triangles.empty()) continue;
                
                // Check if all triangles share same transform (skip procedurals)
                auto firstHandle = triangles[0].second->getTransformHandle();
                bool all_same = true;
                for (size_t i = 1; i < triangles.size() && all_same; ++i) {
                    auto h = triangles[i].second->getTransformHandle();
                    if (h.get() != firstHandle.get()) all_same = false;
                }
                
                if (all_same) {
                    SelectableItem item;
                    item.type = SelectableType::Object;
                    item.object = triangles[0].second;
                    item.object_index = triangles[0].first;
                    item.name = name;
                    ctx.selection.addToSelection(item);
                }
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Select None##obj")) {
            ctx.selection.clearSelection();
        }
        ImGui::SameLine();
        ImGui::Text("(%d objects)", (int)mesh_ui_cache.size());

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
    
    ImGui::EndChild(); // End HierarchyTree
    
    // ─────────────────────────────────────────────────────────────────────────
    // Selection Properties (Compact - Camera/Light only)
    // ─────────────────────────────────────────────────────────────────────────
    if (sel.hasSelection()) {
        if (ImGui::CollapsingHeader("Selection Properties", ImGuiTreeNodeFlags_DefaultOpen)) {
            
            // Header with type and name + delete button
            const char* typeIcon = "[?]";
            ImVec4 typeColor = ImVec4(1, 1, 1, 1);
            switch (sel.selected.type) {
                case SelectableType::Camera: typeIcon = "[CAM]"; typeColor = ImVec4(0.4f, 0.8f, 1.0f, 1.0f); break;
                case SelectableType::Light: typeIcon = "[*]"; typeColor = ImVec4(1.0f, 0.9f, 0.4f, 1.0f); break;
                case SelectableType::Object: typeIcon = "[M]"; typeColor = ImVec4(0.7f, 0.8f, 0.9f, 1.0f); break;
                default: break;
            }
            ImGui::TextColored(typeColor, "%s %s", typeIcon, sel.selected.name.c_str());
            
            ImGui::Checkbox("Gizmo", &sel.show_gizmo);
            ImGui::SameLine(ImGui::GetWindowWidth() - 55);
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.2f, 0.2f, 1.0f));
            if (ImGui::SmallButton("Del")) {
                bool deleted = false;
                if (sel.selected.type == SelectableType::Object && sel.selected.object) {
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
            
            // CAMERA PROPERTIES
            if (sel.selected.type == SelectableType::Camera && sel.selected.camera) {
                auto& cam = *sel.selected.camera;
                ImGui::Separator();
                
                Vec3 pos = cam.lookfrom;
                if (ImGui::DragFloat3("Position##cam", &pos.x, 0.1f)) {
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
                
                Vec3 target = cam.lookat;
                if (ImGui::DragFloat3("Target", &target.x, 0.1f)) {
                    cam.lookat = target;
                    cam.update_camera_vectors();
                }
                
                float fov = (float)cam.vfov;
                if (ImGui::SliderFloat("FOV", &fov, 10.0f, 120.0f)) {
                    cam.vfov = fov;
                    cam.fov = fov;
                    cam.update_camera_vectors();
                }
                
                // DOF Settings
                ImGui::SliderFloat("Aperture", &cam.aperture, 0.0f, 5.0f);
                
                // Focus Distance with Pick Focus button
                ImGui::DragFloat("Focus Dist", &cam.focus_dist, 0.1f, 0.01f, 100.0f);
                // Pick Focus mode - sets focus to clicked object distance
                if (is_picking_focus) {
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.4f, 0.1f, 1.0f));
                }
                if (ImGui::Button(is_picking_focus ? "Picking..." : "Pick")) {
                    is_picking_focus = !is_picking_focus;
                }
                if (is_picking_focus) {
                    ImGui::PopStyleColor();
                }
                if (ImGui::IsItemHovered()) {
                     ImGui::SetTooltip("Click on an object in viewport to set focus distance (ignores selection)");
                }
                

                
                cam.lens_radius = cam.aperture * 0.5f;
                
                ImGui::SliderInt("Blades", &cam.blade_count, 3, 12);
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Aperture blade count (affects bokeh shape)");
                
                // Mouse Sensitivity
                ImGui::Spacing();
                if (ImGui::SliderFloat("Mouse Sensitivity", &ctx.mouse_sensitivity, 0.01f, 5.0f, "%.3f")) {
                    // Value updated directly via reference
                }
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Camera rotation/panning speed");
                
                // Set as Active Camera button
                bool is_active = (ctx.scene.camera.get() == &cam);
                if (!is_active) {
                    if (ImGui::Button("Set as Active Camera")) {
                        // Find camera index
                        for (size_t i = 0; i < ctx.scene.cameras.size(); ++i) {
                            if (ctx.scene.cameras[i].get() == &cam) {
                                ctx.scene.setActiveCamera(i);
                                if (ctx.optix_gpu_ptr) {
                                    ctx.optix_gpu_ptr->setCameraParams(*ctx.scene.camera);
                                    ctx.optix_gpu_ptr->resetAccumulation();
                                }
                                ctx.renderer.resetCPUAccumulation();
                                break;
                            }
                        }
                    }
                    ImGui::SameLine();
                } else {
                    ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.5f, 1.0f), "[Active]");
                    ImGui::SameLine();
                }
                
                // Reset Camera button
                if (ImGui::Button("Reset Camera")) {
                    cam.reset();
                    sel.selected.position = cam.lookfrom;
                    if (ctx.optix_gpu_ptr && g_hasOptix) {
                        ctx.optix_gpu_ptr->setCameraParams(cam);
                        ctx.optix_gpu_ptr->resetAccumulation();
                    }
                    ctx.renderer.resetCPUAccumulation();
                    ctx.start_render = true;
                }
            }
            
            // LIGHT PROPERTIES
            else if (sel.selected.type == SelectableType::Light && sel.selected.light) {
                auto& light = *sel.selected.light;
                bool light_changed = false;
                ImGui::Separator();
                
                const char* lightTypes[] = {"Point", "Directional", "Spot", "Area"};
                int typeIdx = (int)light.type();
                if (typeIdx >= 0 && typeIdx < 4) ImGui::TextDisabled("Type: %s", lightTypes[typeIdx]);
                
                if (ImGui::DragFloat3("Position##light", &light.position.x, 0.1f)) {
                    sel.selected.position = light.position;
                    light_changed = true;
                }
                
                if (light.type() == LightType::Directional || light.type() == LightType::Spot) {
                    if (ImGui::DragFloat3("Direction", &light.direction.x, 0.01f)) light_changed = true;
                }
                
                if (ImGui::ColorEdit3("Color", &light.color.x)) light_changed = true;
                if (ImGui::DragFloat("Intensity", &light.intensity, 0.5f, 0.0f, 1000.0f)) light_changed = true;
                
                if (light.type() == LightType::Point || light.type() == LightType::Directional) {
                    if (ImGui::DragFloat("Radius", &light.radius, 0.01f, 0.01f, 100.0f)) light_changed = true;
                }
                
                if (auto sl = dynamic_cast<SpotLight*>(&light)) {
                    float angle = sl->getAngleDegrees();
                    if (ImGui::DragFloat("Cone Angle", &angle, 0.5f, 1.0f, 89.0f)) {
                        sl->setAngleDegrees(angle);
                        light_changed = true;
                    }
                }
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
                
                if (light_changed && ctx.optix_gpu_ptr && g_hasOptix) {
                    ctx.optix_gpu_ptr->setLightParams(ctx.scene.lights);
                    ctx.optix_gpu_ptr->resetAccumulation();
                }
            }
        }
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // MATERIAL EDITOR (Takes remaining space)
    // ─────────────────────────────────────────────────────────────────────────
    ImGui::Separator();
    if (ImGui::CollapsingHeader("Material Editor", ImGuiTreeNodeFlags_DefaultOpen)) {
        drawMaterialPanel(ctx);
    }
    
    // Light Gizmos and Transform Gizmos moved to main draw() loop for all tabs
}

// ═══════════════════════════════════════════════════════════════════════════════
// MATERIAL & TEXTURE EDITOR PANEL
// ═══════════════════════════════════════════════════════════════════════════════
void SceneUI::drawMaterialPanel(UIContext& ctx) {
    SceneSelection& sel = ctx.selection;

    // Only show for selected objects
    if (sel.selected.type != SelectableType::Object || !sel.selected.object) {
        ImGui::TextDisabled("Select an object to edit materials");
        return;
    }

    std::string obj_name = sel.selected.name;
    if (obj_name.empty()) {
        ImGui::TextDisabled("Unnamed Object");
        return;
    }

    // Ensure mesh cache is valid to find all triangles of this object
    if (!mesh_cache_valid) {
        rebuildMeshCache(ctx.scene.world.objects);
    }

    auto cache_it = mesh_cache.find(obj_name);
    if (cache_it == mesh_cache.end()) {
        ImGui::TextColored(ImVec4(1, 0, 0, 1), "Object mesh data not found in cache.");
        return;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 1. SCAN FOR USED MATERIALS (SLOTS)
    // ─────────────────────────────────────────────────────────────────────────
    // We need to know which unique material IDs are used by this object's triangles.
    // We'll store them in a list preserving order if possible (or just sorted/unique).
    std::vector<uint16_t> used_material_ids;
    
    // Simple scan: Iterate all triangles in the cache for this object
    // Note: For very high-poly objects, this might be slow every frame. 
    // Optimization: Cache this list in SceneSelection or similar if needed.
    for (const auto& pair : cache_it->second) {
        std::shared_ptr<Triangle> tri = pair.second;
        if (tri) {
            uint16_t mid = tri->getMaterialID();
            bool found = false;
            for (uint16_t existing_id : used_material_ids) {
                if (existing_id == mid) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                used_material_ids.push_back(mid);
            }
        }
    }

    // Sort for consistent display order
    // std::sort(used_material_ids.begin(), used_material_ids.end()); 
    // (Optional: Sorting might re-order slots unexpectedly if ids change, but keeps it stable)

    if (used_material_ids.empty()) {
        ImGui::TextDisabled("No geometry/materials found.");
        return;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 2. SLOT SELECTION UI
    // ─────────────────────────────────────────────────────────────────────────
    static int active_slot_index = 0;
    
    // Safety: Reset slot index if out of bounds (e.g. material removed or selection changed)
    if (active_slot_index >= (int)used_material_ids.size()) {
        active_slot_index = 0;
    }
    
    // Reset slot index when selection changes (rudimentary check by name change)
    static std::string last_selected_obj_name = "";
    if (last_selected_obj_name != obj_name) {
        active_slot_index = 0;
        last_selected_obj_name = obj_name;
    }

    ImGui::TextColored(ImVec4(0.9f, 0.7f, 0.3f, 1.0f), "Material Slots");
    
    // Draw the Slot List
    if (ImGui::BeginListBox("##MaterialSlots", ImVec2(-FLT_MIN, 5 * ImGui::GetTextLineHeightWithSpacing()))) {
        for (int i = 0; i < (int)used_material_ids.size(); i++) {
            uint16_t mat_id = used_material_ids[i];
            Material* mat = MaterialManager::getInstance().getMaterial(mat_id);
            
            std::string slot_label;
            if (mat) {
                slot_label = std::string("Slot ") + std::to_string(i) + ": " + mat->materialName;
            } else {
                 slot_label = std::string("Slot ") + std::to_string(i) + ": [Null]";
            }

            const bool is_selected = (active_slot_index == i);
            if (ImGui::Selectable(slot_label.c_str(), is_selected)) {
                active_slot_index = i;
            }

            // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
            if (is_selected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndListBox();
    }

    // Current pointer to the active material
    uint16_t active_mat_id = used_material_ids[active_slot_index];
    Material* active_mat_ptr = MaterialManager::getInstance().getMaterial(active_mat_id);
    std::string current_mat_name = active_mat_ptr ? active_mat_ptr->materialName : "None";

    // ─────────────────────────────────────────────────────────────────────────
    // 3. ASSIGN MATERIAL TO ACTIVE SLOT
    // ─────────────────────────────────────────────────────────────────────────
    ImGui::Separator();
    ImGui::Text("Active Slot Assignment:");
    
    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 110); // Space for +New buttons (adjusted for +S and +V)
    if (ImGui::BeginCombo("##SlotAssignment", current_mat_name.c_str())) {
        auto& mgr = MaterialManager::getInstance();
        const auto& all_materials = mgr.getAllMaterials();
        
        for (size_t i = 0; i < all_materials.size(); i++) {
            if (!all_materials[i]) continue;
            
            bool is_selected = ((uint16_t)i == active_mat_id);
            std::string label = all_materials[i]->materialName;
            if (label.empty()) label = "Mat #" + std::to_string(i);

            if (ImGui::Selectable(label.c_str(), is_selected)) {
                if ((uint16_t)i != active_mat_id) {
                    // REPLACE MATERIAL LOGIC:
                    // Find all triangles that WERE using 'active_mat_id' (the old material for this slot)
                    // and update them to use 'i' (the new material).
                    // This preserves other slots.
                    
                    int count_replaced = 0;
                    for (auto& pair : cache_it->second) {
                         if (pair.second->getMaterialID() == active_mat_id) {
                             pair.second->setMaterialID((uint16_t)i);
                             count_replaced++;
                         }
                    }
                    
                    SCENE_LOG_INFO("Replaced material in Slot " + std::to_string(active_slot_index) + 
                                   " (ID: " + std::to_string(active_mat_id) + " -> " + std::to_string(i) + 
                                   "). Triangles updated: " + std::to_string(count_replaced));

                    // Trigger Rebuilds
                    ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
                    ctx.renderer.resetCPUAccumulation();
                    if (ctx.optix_gpu_ptr) {
                        ctx.renderer.rebuildOptiXGeometry(ctx.scene, ctx.optix_gpu_ptr);
                        ctx.optix_gpu_ptr->resetAccumulation();
                    }
                    g_ProjectManager.markModified();
                }
            }
            if (is_selected) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    ImGui::PopItemWidth();

    // ─────────────────────────────────────────────────────────────────────────
    // SHORTCUTS (New Surface / Volume) - affecting Active Slot
    // ─────────────────────────────────────────────────────────────────────────
    ImGui::SameLine();
    if (ImGui::Button("+S", ImVec2(40,0))) {
         ImGui::OpenPopup("NewSurfPopup");
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Create New Surface Material");

    ImGui::SameLine();
    if (ImGui::Button("+V", ImVec2(40,0))) {
         ImGui::OpenPopup("NewVolPopup");
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Create New Volumetric Material");

    if (ImGui::BeginPopup("NewSurfPopup")) {
        ImGui::Text("Create New Surface");
        if (ImGui::Selectable("Create & Assign")) {
            auto& mgr = MaterialManager::getInstance();
            auto new_mat = std::make_shared<PrincipledBSDF>(Vec3(0.8f), 0.5f, 0.0f);
            std::string name = "Surface_" + std::to_string(mgr.getMaterialCount());
            new_mat->materialName = name;
            uint16_t new_id = mgr.addMaterial(name, new_mat);
            
            // Assign to current slot triangles
            for (auto& pair : cache_it->second) {
                if (pair.second->getMaterialID() == active_mat_id) {
                    pair.second->setMaterialID(new_id);
                }
            }
            // Trigger updates
            ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
            ctx.renderer.resetCPUAccumulation();
            if (ctx.optix_gpu_ptr) {
                 ctx.renderer.rebuildOptiXGeometry(ctx.scene, ctx.optix_gpu_ptr);
                 ctx.optix_gpu_ptr->resetAccumulation();
            }
            g_ProjectManager.markModified();
        }
        ImGui::EndPopup();
    }

    if (ImGui::BeginPopup("NewVolPopup")) {
        ImGui::Text("Create New Volumetric");
        if (ImGui::Selectable("Create & Assign")) {
            auto& mgr = MaterialManager::getInstance();
            auto perlin = std::make_shared<Perlin>();
            auto new_mat = std::make_shared<Volumetric>(
                Vec3(0.8f),     // albedo
                1.0,            // density
                0.1,            // absorption
                0.5,            // scattering
                Vec3(0.0f),     // emission
                perlin          // noise
            );
            std::string name = "Volume_" + std::to_string(mgr.getMaterialCount());
            new_mat->materialName = name;
            uint16_t new_id = mgr.addMaterial(name, new_mat);
            
            // Assign to current slot triangles
            for (auto& pair : cache_it->second) {
                if (pair.second->getMaterialID() == active_mat_id) {
                    pair.second->setMaterialID(new_id);
                }
            }
            // Trigger updates
            ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
            ctx.renderer.resetCPUAccumulation();
            if (ctx.optix_gpu_ptr) {
                 ctx.renderer.rebuildOptiXGeometry(ctx.scene, ctx.optix_gpu_ptr);
                 ctx.optix_gpu_ptr->resetAccumulation();
            }
            g_ProjectManager.markModified();
        }
        ImGui::EndPopup();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // MATERIAL EDITOR (Context-Aware for Active Slot)
    // ─────────────────────────────────────────────────────────────────────────
    // Re-fetch active material in case it changed
    active_mat_id = used_material_ids[active_slot_index];
    // NOTE: If we just replaced the material, the used_material_ids list is somewhat stale 
    // until next frame, but since we updated the triangles, the 'active_mat_id' locally 
    // typically refers to the old ID unless found again. 
    // Ideally we should update the 'used_material_ids' vector immediately, but for now 
    // we can re-query the triangles or just wait for next frame UI refresh. 
    // To be safe, let's just use the pointer from the manager for the *assigned* ID.
    // However, we just changed the ID in the triangles. The 'used_material_ids' array 
    // still holds the OLD ID at index 'active_slot_index'.
    // We should fix this visual glitch by simply returning if we made a heavy change, 
    // or by updating the local array.
    
    // Let's rely on the frame refresh for perfect consistency, but try to show 
    // the potentially new material if we can.
    
    active_mat_ptr = MaterialManager::getInstance().getMaterial(active_mat_id);
    if (!active_mat_ptr) return;

    // Check material type
    Volumetric* vol_mat = dynamic_cast<Volumetric*>(active_mat_ptr);
    PrincipledBSDF* pbsdf = dynamic_cast<PrincipledBSDF*>(active_mat_ptr);

    ImGui::Separator();
    
    if (vol_mat) {
        ImGui::TextColored(ImVec4(0.8f, 0.6f, 1.0f, 1.0f), "[Volumetric Properties]");
        
        bool changed = false;
        
        // --- Reuse Volumetric UI Logic ---
        Vec3 alb = vol_mat->getAlbedo();
        if (ImGui::ColorEdit3("Scattering Color", &alb.x)) { vol_mat->setAlbedo(alb); changed = true; }
        
        float dens = (float)vol_mat->getDensity();
        if (ImGui::SliderFloat("Density", &dens, 0.0f, 10.0f)) { vol_mat->setDensity(dens); changed = true; }
        
        float scatt = (float)vol_mat->getScattering();
        if (ImGui::SliderFloat("Scattering", &scatt, 0.0f, 5.0f)) { vol_mat->setScattering(scatt); changed = true; }
        
        float abs = (float)vol_mat->getAbsorption();
        if (ImGui::SliderFloat("Absorption", &abs, 0.0f, 2.0f)) { vol_mat->setAbsorption(abs); changed = true; }
        
        float g = (float)vol_mat->getG();
        if (ImGui::SliderFloat("Anisotropy", &g, -0.99f, 0.99f)) { vol_mat->setG(g); changed = true; }

        // Emission Color with Strength Control                
        Vec3 emis = vol_mat->getEmissionColor();
        // Emissive Strength Logic (Infer max component as strength)
        float max_e = emis.x;
        if (emis.y > max_e) max_e = emis.y;
        if (emis.z > max_e) max_e = emis.z;
        float strength = (max_e > 1.0f) ? max_e : 1.0f;
        if (max_e < 0.001f) strength = 1.0f; // Default for black

        Vec3 normalized_emis = (max_e > 0.001f) ? emis * (1.0f / strength) : emis;

        if (ImGui::ColorEdit3("Emission Color", &normalized_emis.x)) {
             vol_mat->setEmissionColor(normalized_emis * strength);
             changed = true;
        }
        if (ImGui::DragFloat("Emission Strength", &strength, 0.1f, 0.0f, 1000.0f)) {
             vol_mat->setEmissionColor(normalized_emis * strength);
             changed = true;
        }
        
        float n_scale = vol_mat->getNoiseScale();
        if (ImGui::DragFloat("Noise Scale", &n_scale, 0.01f, 0.01f, 100.0f)) { 
            vol_mat->setNoiseScale(n_scale); 
            changed = true; 
        }

        float void_t = vol_mat->getVoidThreshold();
        if (ImGui::SliderFloat("Void Threshold", &void_t, 0.0f, 1.0f)) {
             vol_mat->setVoidThreshold(void_t);
             changed = true;
        }
        UIWidgets::HelpMarker("Controls empty space amount (Higher = More Voids)");

        // Ray Marching Quality Settings
        float step_size = vol_mat->getStepSize();
        if (ImGui::DragFloat("Step Size (Quality)", &step_size, 0.001f, 0.001f, 1.0f, "%.4f")) {
             vol_mat->setStepSize(step_size);
             changed = true;
        }
        UIWidgets::HelpMarker("Smaller values = Higher Quality (Slower)");

        int max_steps = vol_mat->getMaxSteps();
        if (ImGui::DragInt("Max Steps", &max_steps, 1, 1, 1000)) {
             vol_mat->setMaxSteps(max_steps);
             changed = true;
        }

        // ═══════════════════════════════════════════════════════════════════
        // MULTI-SCATTERING CONTROLS (NEW)
        // ═══════════════════════════════════════════════════════════════════
        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1.0f), "Multi-Scattering");
        
        float multi_scatter = vol_mat->getMultiScatter();
        if (ImGui::SliderFloat("Multi-Scatter##vol", &multi_scatter, 0.0f, 1.0f)) {
            vol_mat->setMultiScatter(multi_scatter);
            changed = true;
        }
        UIWidgets::HelpMarker("Controls multi-scattering brightness (0=single scatter, 1=full multi-scatter)");
        
        float g_back = vol_mat->getGBack();
        if (ImGui::SliderFloat("Backward G##vol", &g_back, -0.99f, 0.0f)) {
            vol_mat->setGBack(g_back);
            changed = true;
        }
        UIWidgets::HelpMarker("Backward scattering anisotropy for silver lining effect");
        
        float lobe_mix = vol_mat->getLobeMix();
        if (ImGui::SliderFloat("Lobe Mix##vol", &lobe_mix, 0.0f, 1.0f)) {
            vol_mat->setLobeMix(lobe_mix);
            changed = true;
        }
        UIWidgets::HelpMarker("Forward/Backward lobe blend (1.0=all forward, 0.0=all backward)");
        
        int light_steps = vol_mat->getLightSteps();
        if (ImGui::SliderInt("Shadow Steps##vol", &light_steps, 0, 8)) {
            vol_mat->setLightSteps(light_steps);
            changed = true;
        }
        UIWidgets::HelpMarker("Light march steps for self-shadowing (0=disabled, 4-8 recommended)");
        
        float shadow_str = vol_mat->getShadowStrength();
        if (ImGui::SliderFloat("Shadow Strength##vol", &shadow_str, 0.0f, 1.0f)) {
            vol_mat->setShadowStrength(shadow_str);
            changed = true;
        }
        UIWidgets::HelpMarker("Self-shadow intensity (0=no shadows, 1=full shadows)");

        if (changed) {
            // OPTIMIZED: Material property change - No geometry rebuild needed!
            // Only reset accumulation and update GPU material buffers
            ctx.renderer.resetCPUAccumulation();
            if (ctx.optix_gpu_ptr) {
                ctx.renderer.updateOptiXMaterialsOnly(ctx.scene, ctx.optix_gpu_ptr);
                ctx.optix_gpu_ptr->resetAccumulation();
            }
        }

    } else if (pbsdf) {
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.6f, 1.0f), "[Surface Properties]");
        
        bool changed = false;
        bool texture_changed = false;

        // ─────────────────────────────────────────────────────────────────────
        // HELPER LAMBDAS (Re-introduced for Texture Editing)
        // ─────────────────────────────────────────────────────────────────────
        
        // 1. Sync GPU Material
        auto SyncGpuMaterial = [](PrincipledBSDF* mat) {
            if (!mat->gpuMaterial) mat->gpuMaterial = std::make_shared<GpuMaterial>();
            
            Vec3 alb = mat->albedoProperty.color;
            mat->gpuMaterial->albedo = make_float3((float)alb.x, (float)alb.y, (float)alb.z);
            mat->gpuMaterial->roughness = (float)mat->roughnessProperty.color.x;
            mat->gpuMaterial->metallic = (float)mat->metallicProperty.intensity;
            
            Vec3 em = mat->emissionProperty.color;
            float emStr = mat->emissionProperty.intensity;
            mat->gpuMaterial->emission = make_float3((float)em.x * emStr, (float)em.y * emStr, (float)em.z * emStr);
            
            mat->gpuMaterial->ior = mat->ior;
            mat->gpuMaterial->transmission = mat->transmission;
            mat->gpuMaterial->opacity = mat->opacityProperty.alpha;
        };

        // 2. Update Texture Bundle for a Triangle
        auto UpdateTriangleTextureBundle = [&](std::shared_ptr<Triangle> target_tri, PrincipledBSDF* mat) {
            OptixGeometryData::TextureBundle bundle = {};
            
            auto SetupTex = [&](std::shared_ptr<Texture>& tex, cudaTextureObject_t& out_tex, int& out_has) {
                 if (tex && tex->is_loaded()) {
                     tex->upload_to_gpu();
                     out_tex = tex->get_cuda_texture();
                     out_has = 1;
                 } else {
                     out_tex = 0;
                     out_has = 0;
                 }
            };

            SetupTex(mat->albedoProperty.texture, bundle.albedo_tex, bundle.has_albedo_tex);
            SetupTex(mat->normalProperty.texture, bundle.normal_tex, bundle.has_normal_tex);
            SetupTex(mat->roughnessProperty.texture, bundle.roughness_tex, bundle.has_roughness_tex);
            SetupTex(mat->metallicProperty.texture, bundle.metallic_tex, bundle.has_metallic_tex);
            SetupTex(mat->emissionProperty.texture, bundle.emission_tex, bundle.has_emission_tex);
            SetupTex(mat->opacityProperty.texture, bundle.opacity_tex, bundle.has_opacity_tex);

            target_tri->textureBundle = bundle;
        };

        // 3. Trigger Update
        auto TriggerMaterialUpdate = [&](bool needs_texture_update) {
            SyncGpuMaterial(pbsdf);
            
            if (needs_texture_update) {
                 // Update all triangles using this material ID
                 // We don't have the scan readily available unless we re-scan or pass used_material_ids
                 // But we can iterate all objects effectively.
                 for (auto& obj : ctx.scene.world.objects) {
                     auto t = std::dynamic_pointer_cast<Triangle>(obj);
                     if (t && t->getMaterialID() == active_mat_id) {
                         UpdateTriangleTextureBundle(t, pbsdf);
                     }
                 }
            }

            ctx.renderer.resetCPUAccumulation();
            if (ctx.optix_gpu_ptr) {
                // OPTIMIZED: Material property change - use fast update path
                // Full rebuild only needed for texture changes (needs_texture_update=true)
                if (needs_texture_update) {
                    ctx.renderer.rebuildOptiXGeometry(ctx.scene, ctx.optix_gpu_ptr);
                } else {
                    ctx.renderer.updateOptiXMaterialsOnly(ctx.scene, ctx.optix_gpu_ptr);
                }
                ctx.optix_gpu_ptr->resetAccumulation();
            }
        };

        // 4. Load Texture Dialog
        auto LoadTextureFromDialog = [&](TextureType type, const std::string& initialDir = "", const std::string& defaultFile = "") -> std::shared_ptr<Texture> {
            
            std::string path = openFileDialogW(L"Image Files\0*.png;*.jpg;*.jpeg;*.tga;*.bmp;*.exr;*.hdr\0", initialDir, defaultFile);
            if (path.empty()) return nullptr;
            
            auto tex = std::make_shared<Texture>(path, type);
            if (tex && tex->is_loaded()) {
                tex->upload_to_gpu();
                SCENE_LOG_INFO("Loaded texture: " + path);
                return tex;
            }
            SCENE_LOG_WARN("Failed to load texture: " + path);
            return nullptr;
        };

        // ─── PROPERTIES UI ───────────────────────────────────────────────────
        
        // Base Color (Albedo)
        Vec3 albedo = pbsdf->albedoProperty.color;
        float albedo_arr[3] = { (float)albedo.x, (float)albedo.y, (float)albedo.z };
        if (ImGui::ColorEdit3("Base Color", albedo_arr)) {
            pbsdf->albedoProperty.color = Vec3(albedo_arr[0], albedo_arr[1], albedo_arr[2]);
            changed = true;
        }
        
        // Metallic
        float metallic = pbsdf->metallicProperty.intensity;
        if (ImGui::SliderFloat("Metallic", &metallic, 0.0f, 1.0f, "%.3f")) {
            pbsdf->metallicProperty.intensity = metallic;
            changed = true;
        }
        
        // Roughness
        float roughness = (float)pbsdf->roughnessProperty.color.x;
        if (ImGui::SliderFloat("Roughness", &roughness, 0.0f, 1.0f, "%.3f")) {
            pbsdf->roughnessProperty.color = Vec3(roughness);
            changed = true;
        }

        // IOR
        float ior = pbsdf->ior;
        if (ImGui::SliderFloat("IOR", &ior, 1.0f, 3.0f, "%.3f")) {
            pbsdf->ior = ior;
            changed = true;
        }

        // Transmission
        float transmission = pbsdf->transmission;
        if (ImGui::SliderFloat("Transmission", &transmission, 0.0f, 1.0f, "%.3f")) {
            pbsdf->setTransmission(transmission, pbsdf->ior);
            changed = true;
        }

        // Specular
        float specular = pbsdf->specularProperty.intensity;
        if (ImGui::SliderFloat("Specular", &specular, 0.0f, 1.0f, "%.3f")) {
            pbsdf->specularProperty.intensity = specular;
            changed = true;
        }
        // Opacity
        float opacity = pbsdf->opacityProperty.alpha;
        if (ImGui::SliderFloat("Opacity", &opacity, 0.0f, 1.0f, "%.3f")) {
            pbsdf->opacityProperty.alpha = opacity;
            changed = true;
        }

        
        // Emission
        Vec3 emission = pbsdf->emissionProperty.color;
        float emission_arr[3] = { (float)emission.x, (float)emission.y, (float)emission.z };
        if (ImGui::ColorEdit3("Emission", emission_arr)) {
            pbsdf->emissionProperty.color = Vec3(emission_arr[0], emission_arr[1], emission_arr[2]);
            changed = true;
        }
        
        float emissionStr = pbsdf->emissionProperty.intensity;
        if (ImGui::DragFloat("Emission Strength", &emissionStr, 0.1f, 0.0f, 1000.0f)) {
            pbsdf->emissionProperty.intensity = emissionStr;
            changed = true;
        }

        // ─── TEXTURES UI ─────────────────────────────────────────────────────
        if (ImGui::TreeNodeEx("Texture Maps", ImGuiTreeNodeFlags_DefaultOpen)) {
            
            // Texture Clipboard (Static to persist across frames/slots)
            static std::shared_ptr<Texture> texture_clipboard = nullptr;

            auto DrawTextureSlot = [&](const char* label, std::shared_ptr<Texture>& tex_ref, TextureType type) {
                ImGui::PushID(label);
                
                ImGui::Text("%s:", label);
                ImGui::SameLine(100);
                
                if (tex_ref && tex_ref->is_loaded()) {
                    // Show texture info
                    std::string name = tex_ref->name;
                    std::string shortName = name;
                    size_t lastSlash = name.find_last_of("/\\");
                    if (lastSlash != std::string::npos) shortName = name.substr(lastSlash + 1);
                    if (shortName.length() > 20) shortName = "..." + shortName.substr(shortName.length() - 17);

                    ImGui::TextColored(ImVec4(0.5f, 0.9f, 0.5f, 1.0f), "%s", shortName.c_str());
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s\n(%dx%d)", name.c_str(), tex_ref->width, tex_ref->height);

                    ImGui::SameLine();
                    ImGui::TextDisabled("(%dx%d)", tex_ref->width, tex_ref->height);
                    
                    // Clear button
                    ImGui::SameLine();
                    if (ImGui::SmallButton("X##Clear")) {
                        tex_ref = nullptr;
                        texture_changed = true;
                    }
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Remove Texture");
                    
                    // Copy Button
                    ImGui::SameLine();
                    if (ImGui::Button("C##Copy")) {
                        texture_clipboard = tex_ref;
                    }
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Copy Texture to Clipboard");

                } else {
                    ImGui::TextDisabled("[None]");
                    
                    // Paste Button (Only if clipboard has content)
                    if (texture_clipboard) {
                        ImGui::SameLine();
                        if (ImGui::Button("P##Paste")) {
                            tex_ref = texture_clipboard;
                            texture_changed = true;
                        }
                        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Paste Texture from Clipboard");
                    }
                }
                
                // Load button
                ImGui::SameLine();
                if (ImGui::SmallButton("Load...")) {
                    std::string initialDir = "";
                    std::string defaultFile = "";
                    
                    if (tex_ref && !tex_ref->name.empty()) {
                         std::string fullPath = tex_ref->name;
                         size_t lastSlash = fullPath.find_last_of("/\\");
                         if (lastSlash != std::string::npos) {
                             initialDir = fullPath.substr(0, lastSlash);
                             defaultFile = fullPath.substr(lastSlash + 1);
                         }
                    } else if (texture_clipboard && !texture_clipboard->name.empty()) {
                        // Optional: Start from clipboard's location if slot is empty
                         std::string fullPath = texture_clipboard->name;
                         size_t lastSlash = fullPath.find_last_of("/\\");
                         if (lastSlash != std::string::npos) {
                             initialDir = fullPath.substr(0, lastSlash);
                         }
                    }

                    auto new_tex = LoadTextureFromDialog(type, initialDir, defaultFile);
                    if (new_tex) {
                        tex_ref = new_tex;
                        texture_changed = true;
                    }
                }
                ImGui::PopID();
            };

            DrawTextureSlot("Albedo", pbsdf->albedoProperty.texture, TextureType::Albedo);
            DrawTextureSlot("Normal", pbsdf->normalProperty.texture, TextureType::Normal);
            DrawTextureSlot("Roughness", pbsdf->roughnessProperty.texture, TextureType::Roughness);
            DrawTextureSlot("Metallic", pbsdf->metallicProperty.texture, TextureType::Metallic);
            DrawTextureSlot("Emission", pbsdf->emissionProperty.texture, TextureType::Emission);
            DrawTextureSlot("Opacity", pbsdf->opacityProperty.texture, TextureType::Opacity);
            
            ImGui::TreePop();
        }

        if (changed || texture_changed) {
             TriggerMaterialUpdate(texture_changed);
             g_ProjectManager.markModified();
        }
    } else {
        ImGui::TextDisabled("Unknown material type.");
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
// SELECTION BOUNDING BOX DRAWING (Multi-selection support)
// ═══════════════════════════════════════════════════════════════════════════════
void SceneUI::drawSelectionBoundingBox(UIContext& ctx) {
    SceneSelection& sel = ctx.selection;
    if (!sel.hasSelection() || !ctx.scene.camera) return;
    
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
    
    // Helper lambda to draw a bounding box
    auto DrawBoundingBox = [&](Vec3 bb_min, Vec3 bb_max, ImU32 color, float thickness) {
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
        
        ImVec2 screen_pts[8];
        bool all_visible = true;
        
        for (int i = 0; i < 8; i++) {
            Vec3 to_corner = corners[i] - cam.lookfrom;
            float depth = to_corner.dot(cam_forward);
            
            if (depth <= 0.01f) {
                all_visible = false;
                break;
            }
            
            float local_x = to_corner.dot(cam_right);
            float local_y = to_corner.dot(cam_up);
            
            float half_height = depth * tan_half_fov;
            float half_width = half_height * aspect_ratio;
            
            float ndc_x = local_x / half_width;
            float ndc_y = local_y / half_height;
            
            screen_pts[i].x = (ndc_x * 0.5f + 0.5f) * screen_w;
            screen_pts[i].y = (0.5f - ndc_y * 0.5f) * screen_h;
        }
        
        if (!all_visible) return;
        
        ImDrawList* draw_list = ImGui::GetBackgroundDrawList();  // Draw behind UI panels
        
        draw_list->AddLine(screen_pts[0], screen_pts[1], color, thickness);
        draw_list->AddLine(screen_pts[1], screen_pts[2], color, thickness);
        draw_list->AddLine(screen_pts[2], screen_pts[3], color, thickness);
        draw_list->AddLine(screen_pts[3], screen_pts[0], color, thickness);
        
        draw_list->AddLine(screen_pts[4], screen_pts[5], color, thickness);
        draw_list->AddLine(screen_pts[5], screen_pts[6], color, thickness);
        draw_list->AddLine(screen_pts[6], screen_pts[7], color, thickness);
        draw_list->AddLine(screen_pts[7], screen_pts[4], color, thickness);
        
        draw_list->AddLine(screen_pts[0], screen_pts[4], color, thickness);
        draw_list->AddLine(screen_pts[1], screen_pts[5], color, thickness);
        draw_list->AddLine(screen_pts[2], screen_pts[6], color, thickness);
        draw_list->AddLine(screen_pts[3], screen_pts[7], color, thickness);
    };
    
    // Draw bounding box for each selected item (multi-selection support)
    for (size_t idx = 0; idx < sel.multi_selection.size(); ++idx) {
        auto& item = sel.multi_selection[idx];
        
        // Primary selection (last one) gets a brighter color
        bool is_primary = (idx == sel.multi_selection.size() - 1);
        ImU32 color = is_primary ? IM_COL32(0, 255, 128, 255) : IM_COL32(0, 200, 100, 180);
        float thickness = is_primary ? 2.0f : 1.5f;
        
        Vec3 bb_min, bb_max;
        bool has_bounds = false;
        
        if (item.type == SelectableType::Object && item.object) {
            std::string selectedName = item.object->nodeName;
            if (selectedName.empty()) selectedName = "Unnamed";
            
            bb_min = Vec3(1e10f, 1e10f, 1e10f);
            bb_max = Vec3(-1e10f, -1e10f, -1e10f);
            bool found_any = false;
            
            if (!mesh_cache_valid) rebuildMeshCache(ctx.scene.world.objects);
            
            auto it = mesh_cache.find(selectedName);
            if (it != mesh_cache.end()) {
                for (auto& pair : it->second) {
                    auto& tri = pair.second;
                    Vec3 v0 = tri->getV0();
                    Vec3 v1 = tri->getV1();
                    Vec3 v2 = tri->getV2();
                    
                    bb_min.x = fminf(bb_min.x, fminf(v0.x, fminf(v1.x, v2.x)));
                    bb_min.y = fminf(bb_min.y, fminf(v0.y, fminf(v1.y, v2.y)));
                    bb_min.z = fminf(bb_min.z, fminf(v0.z, fminf(v1.z, v2.z)));
                    bb_max.x = fmaxf(bb_max.x, fmaxf(v0.x, fmaxf(v1.x, v2.x)));
                    bb_max.y = fmaxf(bb_max.y, fmaxf(v0.y, fmaxf(v1.y, v2.y)));
                    bb_max.z = fmaxf(bb_max.z, fmaxf(v0.z, fmaxf(v1.z, v2.z)));
                    found_any = true;
                }
            }
            
            if (found_any) has_bounds = true;
        }
        else if (item.type == SelectableType::Light && item.light) {
            Vec3 lightPos = item.light->position;
            float boxSize = 0.5f;
            bb_min = Vec3(lightPos.x - boxSize, lightPos.y - boxSize, lightPos.z - boxSize);
            bb_max = Vec3(lightPos.x + boxSize, lightPos.y + boxSize, lightPos.z + boxSize);
            has_bounds = true;
            color = is_primary ? IM_COL32(255, 255, 100, 255) : IM_COL32(200, 200, 80, 180);
        }
        else if (item.type == SelectableType::Camera && item.camera) {
            Vec3 camPos = item.camera->lookfrom;
            float boxSize = 0.5f;
            bb_min = Vec3(camPos.x - boxSize, camPos.y - boxSize, camPos.z - boxSize);
            bb_max = Vec3(camPos.x + boxSize, camPos.y + boxSize, camPos.z + boxSize);
            has_bounds = true;
            color = is_primary ? IM_COL32(100, 200, 255, 255) : IM_COL32(80, 160, 200, 180);
        }
        
        if (has_bounds) {
            DrawBoundingBox(bb_min, bb_max, color, thickness);
        }
    }
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
    // If object has transform, use it - BUT ONLY if not a Mixed Group
    // Mixed groups use the Centroid pivot (startMat) to avoid fighting/resets
    bool is_mixed_group = false;
    if (sel.multi_selection.size() > 1) {
        is_mixed_group = true;
    } else if (sel.selected.type == SelectableType::Object && sel.selected.object) {
        std::string name = sel.selected.object->nodeName; 
        if (name.empty()) name = "Unnamed";
        auto it = mesh_cache.find(name);
        if (it != mesh_cache.end()) {
             Transform* firstT = nullptr;
             for(auto& p : it->second) {
                 auto th = p.second->getTransformHandle().get();
                 if (!firstT) firstT = th;
                 else if (th != firstT) {
                     is_mixed_group = true;
                     break;
                 }
             }
        }
    }

    static Matrix4x4 mixed_gizmo_matrix;
    static bool was_using_mixed = false;
    
    // Check global drag state
    bool is_using_gizmo_now = ImGuizmo::IsUsing();

    if (is_mixed_group) {
        // PERISISTENT GIZMO STATE for Mixed Groups
        // Prevents Gizmo from snapping back to Identity Rotation every frame which causes explosion
        
        if (!is_using_gizmo_now) {
             // Not dragging: Reset to Centroid Position + Identity Rotation
             // (Or we could average rotations, but Identity is safer for a group pivot)
             mixed_gizmo_matrix = Matrix4x4::identity();
             mixed_gizmo_matrix.m[0][3] = sel.selected.position.x;
             mixed_gizmo_matrix.m[1][3] = sel.selected.position.y;
             mixed_gizmo_matrix.m[2][3] = sel.selected.position.z;
        }
        
        // Use the persistent matrix
        objectMatrix[0] = mixed_gizmo_matrix.m[0][0]; objectMatrix[1] = mixed_gizmo_matrix.m[1][0]; objectMatrix[2] = mixed_gizmo_matrix.m[2][0]; objectMatrix[3] = mixed_gizmo_matrix.m[3][0];
        objectMatrix[4] = mixed_gizmo_matrix.m[0][1]; objectMatrix[5] = mixed_gizmo_matrix.m[1][1]; objectMatrix[6] = mixed_gizmo_matrix.m[2][1]; objectMatrix[7] = mixed_gizmo_matrix.m[3][1];
        objectMatrix[8] = mixed_gizmo_matrix.m[0][2]; objectMatrix[9] = mixed_gizmo_matrix.m[1][2]; objectMatrix[10] = mixed_gizmo_matrix.m[2][2]; objectMatrix[11] = mixed_gizmo_matrix.m[3][2];
        objectMatrix[12] = mixed_gizmo_matrix.m[0][3]; objectMatrix[13] = mixed_gizmo_matrix.m[1][3]; objectMatrix[14] = mixed_gizmo_matrix.m[2][3]; objectMatrix[15] = mixed_gizmo_matrix.m[3][3];
    }
    else if (sel.selected.type == SelectableType::Object && sel.selected.object) {
        // Single Object (or Homogeneous Group) - Lock to Object Transform
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
             // BUILD OPTIMIZATION MAP: Name (View) -> Find Triangle
    // Using string_view avoids allocating a copy of the name for every map node (Huge RAM/Time saver for 700k objs)
    std::unordered_map<std::string_view, std::shared_ptr<Triangle>> scene_obj_map;
    scene_obj_map.reserve(ctx.scene.world.objects.size());
    
    for (const auto& obj : ctx.scene.world.objects) {
        auto tri = std::dynamic_pointer_cast<Triangle>(obj);
        if (tri) {
            // Safe because tri->nodeName (std::string) lives in the scene which outlives this function map
            scene_obj_map[tri->nodeName] = tri;
        }
    }        
                bool nameExists = true;
                while (nameExists) {
                    newName = baseName + "_" + std::to_string(counter);
                    nameExists = scene_obj_map.count(newName) > 0;
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
    
    // Restriction Removed: Fallback logic now handles Rot/Scale for mixed groups

    
    ImGuizmo::MODE mode = (sel.transform_space == TransformSpace::Local) ? 
        ImGuizmo::LOCAL : ImGuizmo::WORLD;
    
    // ─────────────────────────────────────────────────────────────────────────
    // Shift + Drag Duplication Logic + IDLE PREVIEW
    // ─────────────────────────────────────────────────────────────────────────
    static bool was_using_gizmo = false;
    static LightState drag_start_light_state;
    static std::shared_ptr<Light> drag_light = nullptr;
    bool is_using = ImGuizmo::IsUsing();
    
    // IDLE PREVIEW: Track when mouse stops moving during drag
    static ImVec2 last_mouse_pos = ImVec2(0, 0);
    static float idle_time = 0.0f;
    static bool preview_updated = false;
    const float IDLE_THRESHOLD = 0.3f;  // 0.3 seconds before preview update
    
   
    if (is_using && is_bvh_dirty) {
        ImVec2 current_mouse = io.MousePos;
        float mouse_delta = sqrtf(powf(current_mouse.x - last_mouse_pos.x, 2) + 
                                  powf(current_mouse.y - last_mouse_pos.y, 2));
        
        if (mouse_delta < 1.0f) {  // Mouse essentially stationary
            idle_time += io.DeltaTime;
            
            // If idle for threshold and not yet updated, do preview update
            if (idle_time >= IDLE_THRESHOLD && !preview_updated) {
                // SCENE_LOG_INFO("[GIZMO] Idle preview - updating BVH");
                if (ctx.optix_gpu_ptr) {
                    ctx.optix_gpu_ptr->updateGeometry(ctx.scene.world.objects);
                    ctx.optix_gpu_ptr->setLightParams(ctx.scene.lights);
                    ctx.optix_gpu_ptr->resetAccumulation();
                }
                ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
                ctx.renderer.resetCPUAccumulation();
                preview_updated = true;
                // Note: Don't set is_bvh_dirty = false, so final update still happens
            }
        } else {
            // Mouse moved - reset idle tracking
            idle_time = 0.0f;
            preview_updated = false;  // Allow another preview after next pause
        }
        last_mouse_pos = current_mouse;
    } else {
        // Not using gizmo - reset tracking
        idle_time = 0.0f;
        preview_updated = false;
        last_mouse_pos = io.MousePos;
    }

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
    
    // NOTE: was_using_gizmo update moved to END of function (after is_bvh_dirty is set)

    // ─────────────────────────────────────────────────────────────────────────
    // Render and Manipulate Gizmo
    // ─────────────────────────────────────────────────────────────────────────
    // Save old position BEFORE manipulation for delta calculation (multi-selection)
    // Save old position & MATRIX BEFORE manipulation for delta calculation
    Vec3 oldGizmoPos(objectMatrix[12], objectMatrix[13], objectMatrix[14]);
    
    Matrix4x4 oldMat;
    oldMat.m[0][0] = objectMatrix[0]; oldMat.m[1][0] = objectMatrix[1]; oldMat.m[2][0] = objectMatrix[2]; oldMat.m[3][0] = objectMatrix[3];
    oldMat.m[0][1] = objectMatrix[4]; oldMat.m[1][1] = objectMatrix[5]; oldMat.m[2][1] = objectMatrix[6]; oldMat.m[3][1] = objectMatrix[7];
    oldMat.m[0][2] = objectMatrix[8]; oldMat.m[1][2] = objectMatrix[9]; oldMat.m[2][2] = objectMatrix[10]; oldMat.m[3][2] = objectMatrix[11];
    oldMat.m[0][3] = objectMatrix[12]; oldMat.m[1][3] = objectMatrix[13]; oldMat.m[2][3] = objectMatrix[14]; oldMat.m[3][3] = objectMatrix[15];
    
    bool manipulated = ImGuizmo::Manipulate(viewMatrix, projMatrix, operation, mode, objectMatrix);

    if (manipulated && is_mixed_group) {
        // Update persistent matrix for next frame interaction
        mixed_gizmo_matrix.m[0][0] = objectMatrix[0]; mixed_gizmo_matrix.m[1][0] = objectMatrix[1]; mixed_gizmo_matrix.m[2][0] = objectMatrix[2]; mixed_gizmo_matrix.m[3][0] = objectMatrix[3];
        mixed_gizmo_matrix.m[0][1] = objectMatrix[4]; mixed_gizmo_matrix.m[1][1] = objectMatrix[5]; mixed_gizmo_matrix.m[2][1] = objectMatrix[6]; mixed_gizmo_matrix.m[3][1] = objectMatrix[7];
        mixed_gizmo_matrix.m[0][2] = objectMatrix[8]; mixed_gizmo_matrix.m[1][2] = objectMatrix[9]; mixed_gizmo_matrix.m[2][2] = objectMatrix[10]; mixed_gizmo_matrix.m[3][2] = objectMatrix[11];
        mixed_gizmo_matrix.m[0][3] = objectMatrix[12]; mixed_gizmo_matrix.m[1][3] = objectMatrix[13]; mixed_gizmo_matrix.m[2][3] = objectMatrix[14]; mixed_gizmo_matrix.m[3][3] = objectMatrix[15];
    }
    
    if (manipulated) {
        Vec3 newPos(objectMatrix[12], objectMatrix[13], objectMatrix[14]);
        Vec3 deltaPos = newPos - oldGizmoPos;  // Calculate delta from BEFORE manipulation
        sel.selected.position = newPos; // Update gizmo/bbox center
        
        // CRITICAL FIX: Update rotation and scale from object's transform matrix
        // This ensures keyframes capture correct rotation/scale values
        if (sel.selected.type == SelectableType::Object && sel.selected.object) {
            auto transformHandle = sel.selected.object->getTransformHandle();
            if (transformHandle) {
                Matrix4x4 objTransform = transformHandle->getFinal();
                
                // Extract rotation (Euler angles in degrees)
                // Assuming rotation order: Z * Y * X
                float sy = sqrtf(objTransform.m[0][0] * objTransform.m[0][0] + objTransform.m[1][0] * objTransform.m[1][0]);
                bool singular = sy < 1e-6f;
                
                if (!singular) {
                    sel.selected.rotation.x = atan2f(objTransform.m[2][1], objTransform.m[2][2]) * (180.0f / 3.14159265f);
                    sel.selected.rotation.y = atan2f(-objTransform.m[2][0], sy) * (180.0f / 3.14159265f);
                    sel.selected.rotation.z = atan2f(objTransform.m[1][0], objTransform.m[0][0]) * (180.0f / 3.14159265f);
                } else {
                    sel.selected.rotation.x = atan2f(-objTransform.m[1][2], objTransform.m[1][1]) * (180.0f / 3.14159265f);
                    sel.selected.rotation.y = atan2f(-objTransform.m[2][0], sy) * (180.0f / 3.14159265f);
                    sel.selected.rotation.z = 0.0f;
                }
                
                // Extract scale
                sel.selected.scale.x = sqrtf(objTransform.m[0][0]*objTransform.m[0][0] + 
                                             objTransform.m[1][0]*objTransform.m[1][0] + 
                                             objTransform.m[2][0]*objTransform.m[2][0]);
                sel.selected.scale.y = sqrtf(objTransform.m[0][1]*objTransform.m[0][1] + 
                                             objTransform.m[1][1]*objTransform.m[1][1] + 
                                             objTransform.m[2][1]*objTransform.m[2][1]);
                sel.selected.scale.z = sqrtf(objTransform.m[0][2]*objTransform.m[0][2] + 
                                             objTransform.m[1][2]*objTransform.m[1][2] + 
                                             objTransform.m[2][2]*objTransform.m[2][2]);
            }
        }

        // Check if this is multi-selection (handle mixed types: lights + objects together)
        bool is_multi_select = sel.multi_selection.size() > 1;
        
        if (is_multi_select) {
            // MULTI-SELECTION: Apply delta to ALL selected items (mixed types)
            float deltaMagnitude = sqrtf(deltaPos.x*deltaPos.x + deltaPos.y*deltaPos.y + deltaPos.z*deltaPos.z);
            
            // For Rotation/Scale, deltaPos might be zero, so we check operation too
            if (deltaMagnitude >= 0.0001f || operation != ImGuizmo::TRANSLATE) {
                if (!mesh_cache_valid) rebuildMeshCache(ctx.scene.world.objects);
                
                // Calculate Delta Matrix for Rotation/Scale
                Matrix4x4 newMat;
                newMat.m[0][0] = objectMatrix[0]; newMat.m[1][0] = objectMatrix[1]; newMat.m[2][0] = objectMatrix[2]; newMat.m[3][0] = objectMatrix[3];
                newMat.m[0][1] = objectMatrix[4]; newMat.m[1][1] = objectMatrix[5]; newMat.m[2][1] = objectMatrix[6]; newMat.m[3][1] = objectMatrix[7];
                newMat.m[0][2] = objectMatrix[8]; newMat.m[1][2] = objectMatrix[9]; newMat.m[2][2] = objectMatrix[10]; newMat.m[3][2] = objectMatrix[11];
                newMat.m[0][3] = objectMatrix[12]; newMat.m[1][3] = objectMatrix[13]; newMat.m[2][3] = objectMatrix[14]; newMat.m[3][3] = objectMatrix[15];
                
                Matrix4x4 deltaMat = newMat * oldMat.inverse();
                
                // Decompose
                Vec3 deltaTranslation(deltaMat.m[0][3], deltaMat.m[1][3], deltaMat.m[2][3]);
                Matrix4x4 deltaRotScale = deltaMat;
                deltaRotScale.m[0][3] = 0; deltaRotScale.m[1][3] = 0; deltaRotScale.m[2][3] = 0;

                for (auto& item : sel.multi_selection) {
                    if (item.type == SelectableType::Object && item.object) {
                        std::string targetName = item.object->nodeName;
                        if (targetName.empty()) targetName = "Unnamed";
                        
                        auto it = mesh_cache.find(targetName);
                        if (it != mesh_cache.end() && !it->second.empty()) {
                            
                            // Apply to unique transforms in this name-group
                            std::unordered_set<Transform*> processed_transforms;
                            for (auto& pair : it->second) {
                                auto tri = pair.second;
                                auto th = tri->getTransformHandle();
                                
                                if (th && processed_transforms.find(th.get()) == processed_transforms.end()) {
                                    if (pivot_mode == 1) { 
                                        // Individual Origins
                                        Vec3 pos(th->base.m[0][3], th->base.m[1][3], th->base.m[2][3]);
                                        th->base.m[0][3] = 0; th->base.m[1][3] = 0; th->base.m[2][3] = 0;
                                        th->setBase(deltaRotScale * th->base);
                                        th->base.m[0][3] = pos.x + deltaTranslation.x;
                                        th->base.m[1][3] = pos.y + deltaTranslation.y;
                                        th->base.m[2][3] = pos.z + deltaTranslation.z;
                                    } else {
                                        // Median Point
                                        th->setBase(deltaMat * th->base); 
                                    }
                                    processed_transforms.insert(th.get());
                                }
                                tri->updateTransformedVertices();
                            }
                        }
                        item.has_cached_aabb = false;
                    }
                    else if (item.type == SelectableType::Light && item.light) {
                        item.light->position = item.light->position + deltaPos;
                    }
                    else if (item.type == SelectableType::Camera && item.camera) {
                        item.camera->lookfrom = item.camera->lookfrom + deltaPos;
                        item.camera->lookat = item.camera->lookat + deltaPos;
                        item.camera->update_camera_vectors();
                    }
                } // End of multi_selection loop
                
                sel.selected.has_cached_aabb = false;
                
                // DEFERRED UPDATE: Only mark dirty during drag
                // Lights don't need BVH but geometry does
                is_bvh_dirty = true;
            }
        }
        else if (sel.selected.type == SelectableType::Light && sel.selected.light) {
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
            // SINGLE SELECTION or Rotate/Scale operations
            // (Multi-select TRANSLATE is handled above)
            
            Matrix4x4 newMat;
            newMat.m[0][0] = objectMatrix[0]; newMat.m[1][0] = objectMatrix[1]; newMat.m[2][0] = objectMatrix[2]; newMat.m[3][0] = objectMatrix[3];
            newMat.m[0][1] = objectMatrix[4]; newMat.m[1][1] = objectMatrix[5]; newMat.m[2][1] = objectMatrix[6]; newMat.m[3][1] = objectMatrix[7];
            newMat.m[0][2] = objectMatrix[8]; newMat.m[1][2] = objectMatrix[9]; newMat.m[2][2] = objectMatrix[10]; newMat.m[3][2] = objectMatrix[11];
            newMat.m[0][3] = objectMatrix[12]; newMat.m[1][3] = objectMatrix[13]; newMat.m[2][3] = objectMatrix[14]; newMat.m[3][3] = objectMatrix[15];

            float deltaMagnitude = sqrtf(deltaPos.x*deltaPos.x + deltaPos.y*deltaPos.y + deltaPos.z*deltaPos.z);
            
            // Only apply transform if there's significant movement or it's rotate/scale
            if (deltaMagnitude >= 0.0001f || operation != ImGuizmo::TRANSLATE) {
                if (!mesh_cache_valid) rebuildMeshCache(ctx.scene.world.objects);

                std::string targetName = sel.selected.object->nodeName;
                if (targetName.empty()) targetName = "Unnamed";
                
                auto it = mesh_cache.find(targetName);
                if (it != mesh_cache.end() && !it->second.empty()) {
                    auto& firstTri = it->second[0].second;
                    auto t_handle = firstTri->getTransformHandle();
                    
                    // Safety check for mixed transforms
                    bool all_same_transform = true;
                    for (size_t i = 1; i < it->second.size() && all_same_transform; ++i) {
                        auto h = it->second[i].second->getTransformHandle();
                        if (h.get() != t_handle.get()) all_same_transform = false;
                    }
                    
                    if (all_same_transform && t_handle) {
                        // Apply full matrix from gizmo (supports translate, rotate, scale)
                        t_handle->setBase(newMat);
                        
                        for (auto& pair : it->second) {
                            pair.second->updateTransformedVertices();
                        }
                    } else {
                        // Fallback: Mixed transforms, apply delta MATRIX to each unique transform (Supports all ops)
                        Matrix4x4 newMat;
                        newMat.m[0][0] = objectMatrix[0]; newMat.m[1][0] = objectMatrix[1]; newMat.m[2][0] = objectMatrix[2]; newMat.m[3][0] = objectMatrix[3];
                        newMat.m[0][1] = objectMatrix[4]; newMat.m[1][1] = objectMatrix[5]; newMat.m[2][1] = objectMatrix[6]; newMat.m[3][1] = objectMatrix[7];
                        newMat.m[0][2] = objectMatrix[8]; newMat.m[1][2] = objectMatrix[9]; newMat.m[2][2] = objectMatrix[10]; newMat.m[3][2] = objectMatrix[11];
                        newMat.m[0][3] = objectMatrix[12]; newMat.m[1][3] = objectMatrix[13]; newMat.m[2][3] = objectMatrix[14]; newMat.m[3][3] = objectMatrix[15];
                        
                        Matrix4x4 deltaMat = newMat * oldMat.inverse();

                        // Decompose for Individual Origins logic
                        Vec3 deltaTranslation(deltaMat.m[0][3], deltaMat.m[1][3], deltaMat.m[2][3]);
                        Matrix4x4 deltaRotScale = deltaMat;
                        deltaRotScale.m[0][3] = 0; deltaRotScale.m[1][3] = 0; deltaRotScale.m[2][3] = 0;

                        std::unordered_set<Transform*> processed_transforms;
                        for (auto& pair : it->second) {
                            auto tri = pair.second;
                            auto th = tri->getTransformHandle();
                            if (th && processed_transforms.find(th.get()) == processed_transforms.end()) {
                                if (pivot_mode == 1) { 
                                     // Individual Origins
                                    Vec3 pos(th->base.m[0][3], th->base.m[1][3], th->base.m[2][3]);
                                    th->base.m[0][3] = 0; th->base.m[1][3] = 0; th->base.m[2][3] = 0;
                                    th->setBase(deltaRotScale * th->base);
                                    th->base.m[0][3] = pos.x + deltaTranslation.x;
                                    th->base.m[1][3] = pos.y + deltaTranslation.y;
                                    th->base.m[2][3] = pos.z + deltaTranslation.z;
                                } else {
                                     // Median Point
                                    th->setBase(deltaMat * th->base);
                                } 
                                processed_transforms.insert(th.get());
                            }
                            tri->updateTransformedVertices();
                        }
                    }
                }
                
                sel.selected.has_cached_aabb = false;

                // DEFERRED UPDATE: Only mark dirty, don't rebuild during drag
                // BVH rebuild will happen when gizmo is released (see below)
                is_bvh_dirty = true;
            }
        }
    }
    
    // DEFERRED BVH UPDATE: Rebuild when gizmo drag ends (not during)
    // This check is at the END so is_bvh_dirty has been set above
    if (!is_using && was_using_gizmo && is_bvh_dirty) {
        SCENE_LOG_INFO("[GIZMO] Released - Triggering BVH update");
        if (ctx.optix_gpu_ptr) {
            ctx.optix_gpu_ptr->updateGeometry(ctx.scene.world.objects);
            ctx.optix_gpu_ptr->setLightParams(ctx.scene.lights);
            if (ctx.scene.camera) ctx.optix_gpu_ptr->setCameraParams(*ctx.scene.camera);
            ctx.optix_gpu_ptr->resetAccumulation();
        }
        ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
        ctx.renderer.resetCPUAccumulation();
        is_bvh_dirty = false;
    }
    
    // Update gizmo state tracking at the END of the function
    was_using_gizmo = is_using;
}

// ═══════════════════════════════════════════════════════════════════════════════
// CAMERA GIZMOS - Draw camera icons in viewport
// ═══════════════════════════════════════════════════════════════════════════════
void SceneUI::drawCameraGizmos(UIContext& ctx) {
    if (!ctx.scene.camera || ctx.scene.cameras.size() <= 1) return;
    
    Camera& activeCam = *ctx.scene.camera;
    ImDrawList* draw_list = ImGui::GetBackgroundDrawList();  // Changed from ForegroundDrawList to render behind UI panels
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

        // Ignore click if over UI elements (Window/Panel) or Gizmo
        // WantCaptureMouse is true if mouse is over any ImGui window or interacting with it
        if (ImGui::GetIO().WantCaptureMouse || ImGuizmo::IsOver()) {
             // Exception: If we are dragging a gizmo, we might be 'over' it but not 'using' it yet?
             // Actually ImGuizmo::IsOver() handles gizmo hover.
             // But if we are just over an empty window background, WantCaptureMouse is true.
            return;
        }

        // Check if Ctrl is held for multi-selection
        bool ctrl_held = ImGui::GetIO().KeyCtrl;

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
            int closest_light_index = -1;
            
            for (size_t i = 0; i < ctx.scene.lights.size(); ++i) {
                auto& light = ctx.scene.lights[i];
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
                        closest_light_index = (int)i;
                    }
                    temp = (-half_b + root) / a;
                    if (temp < closest_t && temp > 0.001f) {
                        closest_t = temp;
                        closest_light = light;
                        closest_light_index = (int)i;
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

            // ═══════════════════════════════════════════════════════════
            // APPLY PICK FOCUS (Using Found Hits)
            // ═══════════════════════════════════════════════════════════
            if (is_picking_focus) {
                 float min_dist = 1e9f;
                 bool found_hit = false;

                 // Check Object Hit
                 if (hit && rec.t < min_dist) { 
                     min_dist = rec.t; 
                     found_hit = true; 
                 }
                 // Check Light Hit
                 if (closest_light && closest_t < min_dist) { 
                     min_dist = closest_t; 
                     found_hit = true; 
                 }
                 // Check Camera Hit
                 if (closest_camera && closest_camera_t < min_dist) { 
                     min_dist = closest_camera_t; 
                     found_hit = true; 
                 }

                 if (found_hit) {
                     ctx.scene.camera->focus_dist = min_dist;
                     ctx.scene.camera->update_camera_vectors();
                     if (ctx.optix_gpu_ptr) {
                         ctx.optix_gpu_ptr->setCameraParams(*ctx.scene.camera);
                         ctx.optix_gpu_ptr->resetAccumulation();
                     }
                     ctx.renderer.resetCPUAccumulation();
                     SCENE_LOG_INFO(std::string("Pick Focus set to: ") + std::to_string(min_dist) + "m");
                 }

                 is_picking_focus = false;
                 ctx.start_render = true;
                 return;
            }

            if (hit && rec.triangle && (rec.t < closest_t)) {
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
                    if (ctrl_held) {
                        // Multi-selection: Toggle object in selection list
                        SelectableItem item;
                        item.type = SelectableType::Object;
                        item.object = found_tri;
                        item.object_index = index;
                        item.name = found_tri->nodeName;
                        
                        if (ctx.selection.isSelected(item)) {
                            ctx.selection.removeFromSelection(item);
                            SCENE_LOG_INFO("Multi-select: Removed '" + item.name + "' (Total: " + std::to_string(ctx.selection.multi_selection.size()) + ")");
                        } else {
                            ctx.selection.addToSelection(item);
                            SCENE_LOG_INFO("Multi-select: Added '" + item.name + "' (Total: " + std::to_string(ctx.selection.multi_selection.size()) + ")");
                        }
                    } else {
                        // Single selection: Replace selection
                        ctx.selection.selectObject(found_tri, index, found_tri->nodeName);
                    }
                } else {
                     SCENE_LOG_WARN("Selection: Object found but not in cache.");
                }
            } 
            else if (closest_camera && closest_camera_t < closest_t) {
                 // Camera is closer than light
                 if (ctrl_held) {
                     SelectableItem item;
                     item.type = SelectableType::Camera;
                     item.camera = closest_camera;
                     item.name = "Camera";
                     
                     if (ctx.selection.isSelected(item)) {
                         ctx.selection.removeFromSelection(item);
                     } else {
                         ctx.selection.addToSelection(item);
                     }
                 } else {
                     ctx.selection.selectCamera(closest_camera);
                 }
            }
            else if (closest_light) {
                 if (ctrl_held) {
                     SelectableItem item;
                     item.type = SelectableType::Light;
                     item.light = closest_light;
                     item.light_index = closest_light_index;
                     item.name = "Light";
                     
                     if (ctx.selection.isSelected(item)) {
                         ctx.selection.removeFromSelection(item);
                     } else {
                         ctx.selection.addToSelection(item);
                     }
                 } else {
                     ctx.selection.selectLight(closest_light);
                 }
            }
            else {
                // Clicked on empty space - clear selection only if Ctrl is not held
                if (!ctrl_held) {
                    ctx.selection.clearSelection();
                }
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
    
    // Enable Resize and Collapse/Maximize (removed NoCollapse)
    ImGuiWindowFlags win_flags = ImGuiWindowFlags_None; 
    
    if (ImGui::Begin("Render Result", &show_render_window, win_flags)) {
        
        static bool show_sidebar = true;

        // Progress Info
        float progress = (float)current_samples / (float)target_samples;
        if (progress > 1.0f) progress = 1.0f;

        // Header and Toolbar
        // ---------------------------------------------------------
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

        ImGui::SameLine();
        ImGui::TextDisabled("|");
        ImGui::SameLine();

        // Toolbar Buttons
        if (ImGui::Button("Save Image")) {
             std::string filename = "Render_" + std::to_string(time(0)) + ".png";
             ctx.render_settings.save_image_requested = true;
        }
        
        ImGui::SameLine();
        if (ctx.render_settings.is_final_render_mode) {
            if (UIWidgets::DangerButton("Stop", ImVec2(60, 0))) {
                ctx.render_settings.is_final_render_mode = false;
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

                 ImGui::Image((ImTextureID)raytrace_texture, ImVec2(w, h));
                 
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
    

// ============================================================================
// Delete Operation (Shared by Menu and Key Shortcut)
// ============================================================================
// OPTIMIZED VERSION - O(n) instead of O(n²)
void SceneUI::triggerDelete(UIContext& ctx) {
    if (!ctx.selection.hasSelection()) return;

    // Collect all items to delete (supports multi-selection)
    std::vector<SelectableItem> items_to_delete = ctx.selection.multi_selection;
    
    // Build mesh cache once if needed
    if (!mesh_cache_valid) rebuildMeshCache(ctx.scene.world.objects);
    
    // OPTIMIZATION: Collect ALL triangles to delete into a single set for O(1) lookup
    std::unordered_set<Triangle*> triangles_to_delete;
    std::vector<std::string> deleted_names;
    std::vector<std::pair<std::string, std::vector<std::shared_ptr<Triangle>>>> undo_data;
    
    // First pass: Collect all triangles to delete
    for (const auto& item : items_to_delete) {
        if (item.type == SelectableType::Object && item.object) {
            std::string deleted_name = item.name;
            
            auto cache_it = mesh_cache.find(deleted_name);
            if (cache_it != mesh_cache.end()) {
                std::vector<std::shared_ptr<Triangle>> tris_for_undo;
                for (auto& pair : cache_it->second) {
                    triangles_to_delete.insert(pair.second.get());
                    tris_for_undo.push_back(pair.second);
                }
                
                if (!tris_for_undo.empty()) {
                    deleted_names.push_back(deleted_name);
                    undo_data.push_back({deleted_name, std::move(tris_for_undo)});
                }
            }
        }
    }
    
    // OPTIMIZATION: Single remove_if pass for ALL objects - O(n) instead of O(n²)
    if (!triangles_to_delete.empty()) {
        auto& objs = ctx.scene.world.objects;
        objs.erase(
            std::remove_if(objs.begin(), objs.end(), [&](const std::shared_ptr<Hittable>& h){
                auto t = std::dynamic_pointer_cast<Triangle>(h);
                return t && triangles_to_delete.count(t.get()) > 0;
            }),
            objs.end()
        );
    }
    
    // Track deletions in ProjectManager (batch update)
    auto& proj_data = g_ProjectManager.getProjectData();
    for (const auto& deleted_name : deleted_names) {
        // Check Imported Models
        bool found = false;
        for (auto& model : proj_data.imported_models) {
            std::string prefix = std::to_string(model.id) + "_";
            if (deleted_name.find(prefix) == 0) {
                model.deleted_objects.push_back(deleted_name);
                found = true;
                break;
            }
            for (const auto& obj_inst : model.objects) {
                if (obj_inst.node_name == deleted_name) {
                    model.deleted_objects.push_back(deleted_name);
                    found = true;
                    break;
                }
            }
            if (found) break;
        }
        
        // Check Procedural Objects
        if (!found) {
            auto it = std::remove_if(proj_data.procedural_objects.begin(), proj_data.procedural_objects.end(),
                [&](const ProceduralObjectData& p) { return p.display_name == deleted_name; });
            proj_data.procedural_objects.erase(it, proj_data.procedural_objects.end());
        }
    }
    
    // Record undo commands
    for (auto& [name, tris] : undo_data) {
        history.record(std::make_unique<DeleteObjectCommand>(name, tris));
    }
    
    // Handle light deletions
    int deleted_lights = 0;
    for (const auto& item : items_to_delete) {
        if (item.type == SelectableType::Light && item.light) {
            auto& lights = ctx.scene.lights;
            auto it = std::find(lights.begin(), lights.end(), item.light);
            if (it != lights.end()) {
                history.record(std::make_unique<DeleteLightCommand>(item.light));
                lights.erase(it);
                deleted_lights++;
            }
        }
    }
    
    // Only rebuild once after all deletions are done
    int deleted_objects = static_cast<int>(deleted_names.size());
    if (deleted_objects > 0 || deleted_lights > 0) {
        ctx.selection.clearSelection();
        g_ProjectManager.markModified();
        
        // Force UI cache update (rebuild sets mesh_cache_valid = true internally)
        rebuildMeshCache(ctx.scene.world.objects);
        // mesh_cache_valid is now TRUE after rebuild - don't set to false!
        
        // Single rebuild for all deleted objects
        if (deleted_objects > 0) {
            ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
            ctx.renderer.resetCPUAccumulation();
            if (ctx.optix_gpu_ptr) {
                 ctx.renderer.rebuildOptiXGeometry(ctx.scene, ctx.optix_gpu_ptr);
                 ctx.optix_gpu_ptr->resetAccumulation();
            }
        }
        
        // Update lights if any were deleted
        if (deleted_lights > 0 && ctx.optix_gpu_ptr) {
            ctx.optix_gpu_ptr->setLightParams(ctx.scene.lights);
            ctx.optix_gpu_ptr->resetAccumulation();
        }
        
        ctx.start_render = true;
        
        // [VERBOSE] SCENE_LOG_INFO("Deleted " + std::to_string(deleted_objects) + " objects, " + 
        //                std::to_string(deleted_lights) + " lights");
    }
}

// ============================================================================
// MARQUEE (BOX) SELECTION
// ============================================================================
void SceneUI::drawMarqueeRect() {
    if (!is_marquee_selecting) return;
    
    ImDrawList* draw_list = ImGui::GetForegroundDrawList();
    
    // Normalize rectangle (handle dragging in any direction)
    float x1 = fminf(marquee_start.x, marquee_end.x);
    float y1 = fminf(marquee_start.y, marquee_end.y);
    float x2 = fmaxf(marquee_start.x, marquee_end.x);
    float y2 = fmaxf(marquee_start.y, marquee_end.y);
    
    // Draw filled rect with transparency
    draw_list->AddRectFilled(ImVec2(x1, y1), ImVec2(x2, y2), IM_COL32(100, 150, 255, 40));
    // Draw border
    draw_list->AddRect(ImVec2(x1, y1), ImVec2(x2, y2), IM_COL32(100, 150, 255, 200), 0.0f, 0, 2.0f);
}

void SceneUI::handleMarqueeSelection(UIContext& ctx) {
    ImGuiIO& io = ImGui::GetIO();
    
    // Only handle when not interacting with UI windows and not using gizmo
    // WantCaptureMouse is true when mouse is over an interactive UI element (button, slider, etc.)
    // This is less restrictive than IsAnyItemHovered which blocks even when hovering inactive areas
    if (io.WantCaptureMouse || ImGuizmo::IsOver() || ImGuizmo::IsUsing()) {
        return;
    }
    
    // Start marquee on right mouse button down (or B key + left click for Blender style)
    bool start_marquee = ImGui::IsMouseClicked(ImGuiMouseButton_Right) && !io.KeyCtrl && !io.KeyShift;
    
    if (start_marquee && !is_marquee_selecting) {
        is_marquee_selecting = true;
        marquee_start = io.MousePos;
        marquee_end = io.MousePos;
    }
    
    // Update marquee while dragging
    if (is_marquee_selecting && ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
        marquee_end = io.MousePos;
    }
    
    // Complete marquee on mouse release
    if (is_marquee_selecting && ImGui::IsMouseReleased(ImGuiMouseButton_Right)) {
        is_marquee_selecting = false;
        
        // Normalize rectangle
        float x1 = fminf(marquee_start.x, marquee_end.x);
        float y1 = fminf(marquee_start.y, marquee_end.y);
        float x2 = fmaxf(marquee_start.x, marquee_end.x);
        float y2 = fmaxf(marquee_start.y, marquee_end.y);
        
        // Minimum size to prevent accidental selections
        if ((x2 - x1) < 10 || (y2 - y1) < 10) {
            return;
        }
        
        // Clear current selection if Ctrl is not held
        if (!io.KeyCtrl) {
            ctx.selection.clearSelection();
        }
        
        if (!ctx.scene.camera) return;
        
        Camera& cam = *ctx.scene.camera;
        float screen_w = io.DisplaySize.x;
        float screen_h = io.DisplaySize.y;
        
        // Camera basis vectors for projection
        Vec3 cam_forward = (cam.lookat - cam.lookfrom).normalize();
        Vec3 cam_right = cam_forward.cross(cam.vup).normalize();
        Vec3 cam_up = cam_right.cross(cam_forward).normalize();
        float fov_rad = cam.vfov * 3.14159265359f / 180.0f;
        float tan_half_fov = tanf(fov_rad * 0.5f);
        
        // Lambda to project 3D point to screen
        auto ProjectToScreen = [&](const Vec3& p) -> ImVec2 {
            Vec3 to_point = p - cam.lookfrom;
            float depth = to_point.dot(cam_forward);
            if (depth <= 0.01f) return ImVec2(-10000, -10000);
            
            float local_x = to_point.dot(cam_right);
            float local_y = to_point.dot(cam_up);
            
            float half_height = depth * tan_half_fov;
            float half_width = half_height * aspect_ratio;
            
            float ndc_x = local_x / half_width;
            float ndc_y = local_y / half_height;
            
            return ImVec2(
                (ndc_x * 0.5f + 0.5f) * screen_w,
                (0.5f - ndc_y * 0.5f) * screen_h
            );
        };
        
        // Check which objects are inside the marquee
        if (!mesh_cache_valid) rebuildMeshCache(ctx.scene.world.objects);
        
        int skipped_procedural = 0;
        
        for (auto& [name, triangles] : mesh_cache) {
            if (triangles.empty()) continue;
            
            // IMPORTANT: Check if all triangles share the same TransformHandle
            // Procedural objects may have separate transforms per triangle
            auto firstHandle = triangles[0].second->getTransformHandle();
            bool all_same_transform = true;
            
            for (size_t i = 1; i < triangles.size() && all_same_transform; ++i) {
                auto handle = triangles[i].second->getTransformHandle();
                if (handle.get() != firstHandle.get()) {
                    all_same_transform = false;
                }
            }
            
            if (!all_same_transform) {
                // This object has mixed transforms - skip it
                // (would break if selected because transform would only affect some triangles)
                skipped_procedural++;
                continue;
            }
            
            // Calculate bounding box center for quick check
            Vec3 bb_min(1e10f, 1e10f, 1e10f);
            Vec3 bb_max(-1e10f, -1e10f, -1e10f);
            
            for (auto& pair : triangles) {
                auto& tri = pair.second;
                Vec3 v0 = tri->getV0();
                Vec3 v1 = tri->getV1();
                Vec3 v2 = tri->getV2();
                
                bb_min.x = fminf(bb_min.x, fminf(v0.x, fminf(v1.x, v2.x)));
                bb_min.y = fminf(bb_min.y, fminf(v0.y, fminf(v1.y, v2.y)));
                bb_min.z = fminf(bb_min.z, fminf(v0.z, fminf(v1.z, v2.z)));
                bb_max.x = fmaxf(bb_max.x, fmaxf(v0.x, fmaxf(v1.x, v2.x)));
                bb_max.y = fmaxf(bb_max.y, fmaxf(v0.y, fmaxf(v1.y, v2.y)));
                bb_max.z = fmaxf(bb_max.z, fmaxf(v0.z, fmaxf(v1.z, v2.z)));
            }
            
            Vec3 center = (bb_min + bb_max) * 0.5f;
            ImVec2 screenPos = ProjectToScreen(center);
            
            // Check if center is inside marquee
            if (screenPos.x >= x1 && screenPos.x <= x2 && screenPos.y >= y1 && screenPos.y <= y2) {
                SelectableItem item;
                item.type = SelectableType::Object;
                item.object = triangles[0].second;
                item.object_index = triangles[0].first;
                item.name = name;
                
                if (!ctx.selection.isSelected(item)) {
                    ctx.selection.addToSelection(item);
                }
            }
        }
        
        if (skipped_procedural > 0) {
            SCENE_LOG_WARN("Skipped " + std::to_string(skipped_procedural) + " objects with mixed transforms (use Ctrl+Click)");
        }
        
        // Also check lights
        for (size_t i = 0; i < ctx.scene.lights.size(); ++i) {
            auto& light = ctx.scene.lights[i];
            if (!light) continue;
            
            ImVec2 screenPos = ProjectToScreen(light->position);
            
            if (screenPos.x >= x1 && screenPos.x <= x2 && screenPos.y >= y1 && screenPos.y <= y2) {
                SelectableItem item;
                item.type = SelectableType::Light;
                item.light = light;
                item.light_index = (int)i;
                item.name = "Light_" + std::to_string(i);
                
                if (!ctx.selection.isSelected(item)) {
                    ctx.selection.addToSelection(item);
                }
            }
        }
        
        if (ctx.selection.multi_selection.size() > 0) {
            // [VERBOSE] SCENE_LOG_INFO("Marquee selected " + std::to_string(ctx.selection.multi_selection.size()) + " items");
        }
    }
    
    // Draw the marquee rectangle while selecting
    drawMarqueeRect();
}
