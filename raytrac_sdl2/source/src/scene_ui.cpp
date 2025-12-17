#include "scene_ui.h"
#include "ui_modern.h"
#include "imgui.h"
#include <string>
#include "scene_data.h"
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
bool show_animation_panel = true; // Default closed as requested

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

// Timeline Panel - Blender tarzı ayrı panel (en altta)
void SceneUI::drawTimelinePanel(UIContext& ctx, float screen_y)
{
    static int start_frame = 0;
    static int end_frame = 100;
    static bool frame_range_initialized = false;
    static bool is_playing = false;
    static int playback_frame = 0;
    static auto last_frame_time = std::chrono::steady_clock::now();
    
    // Timeline panelini en altta göster
    float timeline_height = 140.0f;
    ImGui::SetNextWindowPos(ImVec2(0, screen_y - timeline_height), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(ImGui::GetIO().DisplaySize.x, timeline_height), ImGuiCond_Always);
    
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | 
                             ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse;
   
    ImGui::Begin("##Timeline", nullptr, flags);
  
    
    if (!ctx.scene.animationDataList.empty()) {
        auto& anim = ctx.scene.animationDataList[0];
        
        // Detect if we switched to a different animation (or first load) by checking source limits
        static int cached_source_start = -1;
        static int cached_source_end = -1;

        // If the source (file) limits differ from what we have cached, it means a new file/anim was loaded.
        // We should reset the UI selection to the full range of this new animation.
        if (anim.startFrame != cached_source_start || anim.endFrame != cached_source_end) {
            frame_range_initialized = false;
            cached_source_start = anim.startFrame;
            cached_source_end = anim.endFrame;
        }

        // Initialize or Reset defaults
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
        
        ImGui::SameLine();
        ImGui::Text("|");
        ImGui::SameLine();
        
        ImGui::PushItemWidth(80);
        ImGui::SliderInt("FPS", &ctx.render_settings.animation_fps, 1, 60);
        ImGui::PopItemWidth();
        
        ImGui::SameLine();
        ImGui::TextDisabled("(%d frames)", total_frames);
        
        // --- ORTA SATIR: Playback Controls (Blender tarzı) ---
        ImGui::Spacing();
        
        // Play/Pause/Stop butonları
        // if (is_rendering) ImGui::BeginDisabled(); // Removed to allow stopping during playback
        
        if (ImGui::Button(is_playing ? "||" : "|>", ImVec2(30, 25))) {
            is_playing = !is_playing;
            if (is_playing) {
                last_frame_time = std::chrono::steady_clock::now();
            }
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(is_playing ? "Pause" : "Play");
        }
        
        ImGui::SameLine();
        if (ImGui::Button("[]", ImVec2(30, 25))) {
            is_playing = false;
            playback_frame = start_frame;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Stop");
        }
        
        // if (is_rendering) ImGui::EndDisabled();
        
        ImGui::SameLine();
        ImGui::Text("|");
        ImGui::SameLine();
        
        // Current frame display
        ImGui::Text("Frame:");
        ImGui::SameLine();
        ImGui::PushItemWidth(60);
        if (ImGui::InputInt("##CurrentFrame", &playback_frame, 0, 0)) {
            playback_frame = std::clamp(playback_frame, start_frame, end_frame);
        }
        ImGui::PopItemWidth();
        
        
        // Playback logic - sadece state güncelleme
        if (is_playing && !is_rendering) {
            auto now = std::chrono::steady_clock::now();
            float elapsed = std::chrono::duration<float>(now - last_frame_time).count();
            float frame_duration = 1.0f / ctx.render_settings.animation_fps;
            
            if (elapsed >= frame_duration) {
                playback_frame++;
                if (playback_frame > end_frame) {
                    playback_frame = start_frame; // Reset to start
                    is_playing = false; // Stop playback
                }
                last_frame_time = now;
            }
        }
        
        // Playback state'i RenderSettings'e yaz (Main.cpp okuyacak)
        ctx.render_settings.animation_is_playing = is_playing;
        ctx.render_settings.animation_playback_frame = playback_frame;
        
        ImGui::SameLine();
        ImGui::Dummy(ImVec2(20, 0));
        ImGui::SameLine();
        
        // Output folder (Auto-generated)
        ImGui::Text("Output: Auto-generated/render_animation");

        
        // --- TIMELINE SCRUBBER BAR (Blender tarzı) ---
        ImGui::Spacing();
        
        // Render progress veya playback progress
        int current_display_frame = is_rendering ? ctx.render_settings.animation_current_frame : playback_frame;
        float scrubber_progress = (float)(current_display_frame - start_frame) / (float)total_frames;
        
       
        ImGui::PushItemWidth(-1);
        if (ImGui::SliderInt("##Scrubber", &current_display_frame, start_frame, end_frame, "")) {
            if (!is_rendering) {
                playback_frame = current_display_frame;
            }
        }
        ImGui::PopItemWidth();      
        
        // Scrubber üzerinde frame numarası ve durum göster
        ImVec2 scrubber_min = ImGui::GetItemRectMin();
        ImVec2 scrubber_max = ImGui::GetItemRectMax();
        ImVec2 scrubber_size = ImVec2(scrubber_max.x - scrubber_min.x, scrubber_max.y - scrubber_min.y);
        
        char frame_label[64];
        if (is_rendering) {
            snprintf(frame_label, sizeof(frame_label), "Rendering: %d / %d", current_display_frame, end_frame);
        } else if (is_playing) {
            snprintf(frame_label, sizeof(frame_label), "Playing: %d", current_display_frame);
        } else {
            snprintf(frame_label, sizeof(frame_label), "Frame: %d", current_display_frame);
        }
        
        ImVec2 text_size = ImGui::CalcTextSize(frame_label);
        ImVec2 text_pos = ImVec2(
            scrubber_min.x + (scrubber_size.x - text_size.x) * 0.5f,
            scrubber_min.y + (scrubber_size.y - text_size.y) * 0.5f
        );
        
        ImGui::GetWindowDrawList()->AddText(text_pos, IM_COL32(255, 255, 255, 200), frame_label);
        
        // --- ALT SATIR: Render Butonları ---
        ImGui::Spacing();
        
        if (is_rendering) ImGui::BeginDisabled();
        
        if (ImGui::Button("Render Animation", ImVec2(130, 22))) {
            ctx.render_settings.start_animation_render = true;
            ctx.start_render = true;
            is_playing = false; // Playback'i durdur
            SCENE_LOG_INFO("Starting animation render...");
        }
        
        if (is_rendering) ImGui::EndDisabled();
        
        ImGui::SameLine();
        
        if (!is_rendering) ImGui::BeginDisabled();
        
        if (ImGui::Button("Stop Render", ImVec2(90, 22))) {
            rendering_stopped_cpu = true;
            rendering_stopped_gpu = true;
            SCENE_LOG_WARN("Stop requested by user.");
        }
        
        if (!is_rendering) ImGui::EndDisabled();
        
        // Frame sayısı bilgisi
        if (!ctx.render_settings.animation_output_folder.empty()) {
            ImGui::SameLine();
            ImGui::Dummy(ImVec2(10, 0));
            ImGui::SameLine();
            
            // Render edilmiş frame sayısını say
            int rendered_count = 0;
            for (int f = start_frame; f <= end_frame; f++) {
                char check_path[512];
                snprintf(check_path, sizeof(check_path), "%s/frame_%04d.png", 
                        ctx.render_settings.animation_output_folder.c_str(), f);
                if (std::filesystem::exists(check_path)) {
                    rendered_count++;
                }
            }
            
            if (rendered_count > 0) {
                ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.5f, 1.0f), 
                    "%d/%d frames rendered", rendered_count, total_frames);
            } else {
                ImGui::TextDisabled("No frames rendered yet");
            }
        }
        
    } else {
        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.3f, 1.0f), "No animation data loaded.");
        frame_range_initialized = false;
    }
    
    ImGui::End();
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
    if (UIWidgets::BeginSection("Interface / Theme", ImVec4(0.5f, 0.7f, 1.0f, 1.0f))) {
        
        auto& themeManager = ThemeManager::instance();
        auto themeNames = themeManager.getAllThemeNames();
        int currentThemeIdx = themeManager.currentIndex();

        if (ImGui::Combo("Select Theme", &currentThemeIdx, 
                         themeNames.data(), static_cast<int>(themeNames.size()))) {
            themeManager.setTheme(currentThemeIdx);
            themeManager.applyCurrentTheme(panel_alpha);
        }

        // Panel Transparency   
        if (ImGui::SliderFloat("Panel Transparency", &panel_alpha, 0.1f, 1.0f, "%.2f")) {
            ImGuiStyle& style = ImGui::GetStyle();
            style.Colors[ImGuiCol_WindowBg].w = panel_alpha;
        }

        UIWidgets::EndSection();
    }
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
        bool aperture_changed = UIWidgets::SliderWithHelp("Aperture", &aperture, 0.0f, 5.0f,
                                                           "Lens aperture size - affects blur amount");
        bool focus_changed = UIWidgets::DragFloatWithHelp("Focus Distance", &focus_dist, 0.05f, 0.01f, 100.0f,
                                                           "Distance to sharp focus plane");

        if (aperture_changed || focus_changed) {
            ctx.scene.camera->lens_radius = aperture * 0.5f;
            ctx.scene.camera->update_camera_vectors();
        }

        ImGui::SliderInt("Blade Count", &ctx.scene.camera->blade_count, 3, 12);
        UIWidgets::HelpMarker("Number of aperture blades - affects bokeh shape");
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

            ImGui::DragFloat3("Position", &light->position.x, 0.1f);

            if (light->type() == LightType::Directional || light->type() == LightType::Spot)
                ImGui::DragFloat3("Direction", &light->direction.x, 0.1f);

            ImGui::ColorEdit3("Color", &light->color.x);
            ImGui::DragFloat("Intensity", &light->intensity, 0.1f, 0, 1000.0f);

            if (light->type() == LightType::Point ||
                light->type() == LightType::Area ||
                light->type() == LightType::Directional)
                ImGui::DragFloat("Radius", &light->radius, 0.01f, 0.01f, 100.0f);

            ImGui::TreePop();
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
                        cloudParams.cloud_coverage = 0.6f;
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
                        cloudParams.cloud_coverage = 0.7f;
                        cloudParams.cloud_density = 4.0f;
                        cloudParamsChanged = true;
                        break;
                    case 6: // Fog/Low Clouds
                        cloudParams.cloud_height_min = 0.0f;
                        cloudParams.cloud_height_max = 200.0f;
                        cloudParams.cloud_scale = 0.3f;
                        cloudParams.cloud_coverage = 0.8f;
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
    // Dinamik yükseklik hesabı - Timeline açıksa yer aç
    extern bool show_animation_panel;
    float bottom_margin = 20.0f; 
    if (show_animation_panel && ctx.scene.initialized) {
        bottom_margin = 160.0f; // Timeline height (140) + padding
    }

    // Maksimum yükseklik constraint'i ekle
    ImGui::SetNextWindowSizeConstraints(
        ImVec2(300, 200),                 // Min size
        ImVec2(FLT_MAX, screen_y - bottom_margin) // Max height
    );

    ImGui::SetNextWindowSize(ImVec2(400, screen_y - 20), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(1040, 20), ImGuiCond_FirstUseEver);
    ImGui::Begin("Properties", nullptr);

    if (ImGui::BeginTabBar("MainPropertiesTabs"))
    {
        if (ImGui::BeginTabItem("System")) {
            drawThemeSelector();
            drawResolutionPanel();

            if (UIWidgets::BeginSection("Panels", ImVec4(0.6f, 0.4f, 0.7f, 1.0f))) {
                extern bool show_animation_panel; // Defined at file scope
                ImGui::Checkbox("Show Animation Pnl", &show_animation_panel);
                UIWidgets::HelpMarker("Show/Hide the bottom timeline panel");
                UIWidgets::EndSection();
            }

            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Render")) {
    UIWidgets::ColoredHeader("Model & Scene", ImVec4(1.0f, 0.8f, 0.6f, 1.0f));
    bool disabled = scene_loading.load();
    if (disabled)
        ImGui::BeginDisabled();

    // Model yüklenmiş mi?
    bool loaded = (ctx.scene.initialized);
    const char* label = loaded ? "Loaded" : "Load Model";

    // Modern StateButton kullan
    if (UIWidgets::StateButton(label, loaded, 
                                ImVec4(0.3f, 1.0f, 0.3f, 1.0f),  // Aktif: yeşil
                                ImVec4(1.0f, 0.3f, 0.3f, 1.0f),  // Pasif: kırmızı
                                ImVec2(100, 0)))
    {
#ifdef _WIN32
        std::string file = openFileDialogW(
            L"3D Files\0*.gltf;*.glb;*.fbx\0All Files\0*.*\0"
        );

        if (!file.empty()) {
            // Render işlemleri devam ediyorsa durdur
            rendering_stopped_cpu = true;
            rendering_stopped_gpu = true;
            
            scene_loading = true;
            scene_loading_done = false;
            active_model_path = file;

            std::thread loader_thread([this, file, &ctx]() {
                // Wait for render thread to fully stop (max 2 seconds)
                int wait_count = 0;
                while (rendering_in_progress && wait_count < 20) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    wait_count++;
                }

                SCENE_LOG_INFO("Starting async scene load...");
                ctx.scene.clear();
                ctx.renderer.create_scene(ctx.scene, ctx.optix_gpu_ptr, file);
                if (ctx.scene.camera)
                    ctx.scene.camera->update_camera_vectors();
                ctx.render_settings.start_animation_render = false;

                SCENE_LOG_INFO("Scene loaded successfully.");
                ctx.start_render = true;
                scene_loading = false;
                scene_loading_done = true;
            });

            loader_thread.detach();
        }
#endif
    }

    if (disabled)
        ImGui::EndDisabled();

    ImGui::SameLine();
    ImGui::TextWrapped("Model: %s", active_model_path.c_str());
    
    UIWidgets::Divider();
    ImGui::PushItemWidth(180);
    UIWidgets::ColoredHeader("Render Engine", ImVec4(1.0f, 0.8f, 0.6f, 1.0f));


    if (ctx.scene.initialized) {
        // --- GPU Section ---
        ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1), "GPU (OptiX)");
        bool prev_use_optix = ctx.render_settings.use_optix;
        if (!g_hasOptix) {
            ImGui::BeginDisabled();
            ImGui::Checkbox("Use OptiX", &ctx.render_settings.use_optix);
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "(No RTX GPU detected)");
            ImGui::EndDisabled();
        }
        else if (ImGui::Checkbox("Use OptiX", &ctx.render_settings.use_optix)) {
            if (ctx.render_settings.use_optix != prev_use_optix) {
                SCENE_LOG_INFO(ctx.render_settings.use_optix ? "OptiX enabled" : "OptiX disabled");
            }
        }
        ImGui::SameLine();
        UIWidgets::HelpMarker("Enables GPU acceleration via NVIDIA OptiX. Requires an RTX-class GPU.");

        // --- Denoiser Section ---
        ImGui::Separator();
        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.8f, 1.0f, 0.6f, 1), "Denoiser");
        bool prev_use_denoiser = ctx.render_settings.use_denoiser;
        if (ImGui::Checkbox("Use Denoiser", &ctx.render_settings.use_denoiser)) {
            if (ctx.render_settings.use_denoiser != prev_use_denoiser) {
                SCENE_LOG_INFO(ctx.render_settings.use_denoiser ? "Denoiser enabled" : "Denoiser disabled");
            }
        }
        ImGui::SameLine();
        UIWidgets::HelpMarker("Applies denoising to reduce noise after rendering. Based on Intel OIDN.");
        if (ctx.render_settings.use_denoiser) {
            ImGui::SliderFloat("Denoiser Blend", &ctx.render_settings.denoiser_blend_factor, 0.0f, 1.0f, "%.2f");
            ImGui::SameLine();
            UIWidgets::HelpMarker("Blends the denoised result with the original. 1 = fully denoised, 0 = original image.");
        }
        // --- CPU Section ---
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.6f, 1), "CPU (BVH)");
        const char* bvh_options[] = { "Embree", "RT_BVH (RayTrophi)" };
        int current_bvh = ctx.render_settings.UI_use_embree ? 0 : 1;

        bool hovered = false;

        ImGui::PushID("BVHCombo");
        if (ImGui::BeginCombo("##BVHType", bvh_options[current_bvh])) {
            for (int n = 0; n < IM_ARRAYSIZE(bvh_options); n++) {
                bool is_selected = (current_bvh == n);
                if (ImGui::Selectable(bvh_options[n], is_selected)) {
                    current_bvh = n;
                    ctx.render_settings.UI_use_embree = (current_bvh == 0);
                    ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
                }
                if (is_selected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        hovered = ImGui::IsItemHovered();
        ImGui::PopID();

        // Warning badge for RT_BVH
        if (!ctx.render_settings.UI_use_embree) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(1.0f, 0.15f, 0.15f, 1.0f),
                "Experimental — Heavy Performance Cost");
        }

        // Tooltip
        if (hovered) {
            ImGui::BeginTooltip();
            if (ctx.render_settings.UI_use_embree) {
                ImGui::TextUnformatted(
                    "Intel Embree BVH backend\n"
                    "Stable, production-proven acceleration structure with "
                    "advanced CPU optimizations.\n"
                    "RayTrophi BVH (RT_BVH)\n"
                    "Experimental high-performance BVH design still under active development.\n"
                    "Currently lacks several optimizations (SIMD, wide nodes, memory compaction).\n"
                    "Rebuilds and traversal can be significantly slower on complex scenes.\n"
                    "Best suited for internal testing and research iterations."
                );
            }
            else {
                ImGui::TextUnformatted(
                    "RayTrophi BVH (RT_BVH)\n"
                    "Experimental high-performance BVH design still under active development.\n"
                    "Currently lacks several optimizations (SIMD, wide nodes, memory compaction).\n"
                    "Rebuilds and traversal can be significantly slower on complex scenes.\n"
                    "Best suited for internal testing and research iterations."
                );
            }
            ImGui::EndTooltip();
        }


        // ============ QUALITY PRESET ============
        ImGui::Separator();
        ImGui::TextColored(ImVec4(1.0f, 0.9f, 0.4f, 1), "Quality Preset");
        
        const char* preset_names[] = { "Preview", "Production", "Cinematic" };
        int current_preset = static_cast<int>(ctx.render_settings.quality_preset);
        
        static bool show_resolution_popup = false;
        static int suggested_width = 0;
        static int suggested_height = 0;
        static std::string preset_name_for_popup;
        
        ImGui::PushItemWidth(180);
        if (ImGui::Combo("##QualityPreset", &current_preset, preset_names, IM_ARRAYSIZE(preset_names))) {
            ctx.render_settings.quality_preset = static_cast<QualityPreset>(current_preset);
            
            // Reset render time tracking
            ctx.render_settings.render_elapsed_seconds = 0.0f;
            ctx.render_settings.render_estimated_remaining = 0.0f;
            ctx.render_settings.avg_sample_time_ms = 0.0f;
            
            // Apply preset settings
            switch (ctx.render_settings.quality_preset) {
                case QualityPreset::Preview:
                    ctx.render_settings.max_samples = 32;
                    ctx.render_settings.max_bounces = 4;
                    ctx.render_settings.use_denoiser = true;
                    ctx.render_settings.denoiser_blend_factor = 0.9f;
                    suggested_width = 1280; suggested_height = 720;
                    preset_name_for_popup = "Preview (720p)";
                    SCENE_LOG_INFO("Quality Preset: Preview (32 samples, 4 bounces)");
                    break;
                    
                case QualityPreset::Production:
                    ctx.render_settings.max_samples = 256;
                    ctx.render_settings.max_bounces = 8;
                    ctx.render_settings.use_denoiser = true;
                    ctx.render_settings.denoiser_blend_factor = 0.7f;
                    suggested_width = 1920; suggested_height = 1080;
                    preset_name_for_popup = "Production (1080p)";
                    SCENE_LOG_INFO("Quality Preset: Production (256 samples, 8 bounces)");
                    break;
                    
                case QualityPreset::Cinematic:
                    ctx.render_settings.max_samples = 1024;
                    ctx.render_settings.max_bounces = 16;
                    ctx.render_settings.use_denoiser = true;
                    ctx.render_settings.denoiser_blend_factor = 0.5f;
                    suggested_width = 2560; suggested_height = 1440;
                    preset_name_for_popup = "Cinematic (2K)";
                    SCENE_LOG_INFO("Quality Preset: Cinematic (1024 samples, 16 bounces)");
                    break;
            }
            
            // Check if resolution change is suggested
            if (image_width != suggested_width || image_height != suggested_height) {
                show_resolution_popup = true;
            }
        }
        ImGui::PopItemWidth();
        
        ImGui::SameLine();
        UIWidgets::HelpMarker(
            "Preview: 720p, 32 samples, 4 bounces - Fast positioning\n"
            "Production: 1080p, 256 samples, 8 bounces - Balanced\n"
            "Cinematic: 2K, 1024 samples, 16 bounces - Final quality"
        );
        
       
        
        if (ImGui::BeginPopupModal("Change Resolution?", &show_resolution_popup, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("Recommended resolution for %s:", preset_name_for_popup.c_str());
            ImGui::Text("%dx%d", suggested_width, suggested_height);
            ImGui::Separator();
            ImGui::Text("Current: %dx%d", image_width, image_height);
            ImGui::Spacing();
            
            if (ImGui::Button("Apply Resolution", ImVec2(140, 0))) {
                pending_width = suggested_width;
                pending_height = suggested_height;
                pending_aspect_ratio = (float)suggested_width / suggested_height;
                pending_resolution_change = true;
                
                // Sync resolution panel static variables
                new_width = suggested_width;
                new_height = suggested_height;
                aspect_w = 16;  // All presets use 16:9
                aspect_h = 9;
                
                // Update preset_index to match
                if (suggested_width == 1280 && suggested_height == 720) {
                    preset_index = 1;  // HD 720p
                } else if (suggested_width == 1920 && suggested_height == 1080) {
                    preset_index = 2;  // Full HD 1080p
                } else if (suggested_width == 2560 && suggested_height == 1440) {
                    preset_index = 3;  // 1440p
                } else {
                    preset_index = 0;  // Custom
                }
                
                // Allow rendering to restart after resolution change
                ctx.render_settings.is_render_paused = false;
                ctx.start_render = true;
                
                show_resolution_popup = false;
                ImGui::CloseCurrentPopup();
                SCENE_LOG_INFO("Resolution changed to " + std::to_string(suggested_width) + "x" + std::to_string(suggested_height));
            }
            ImGui::SameLine();
            if (ImGui::Button("Keep Current", ImVec2(140, 0))) {
                show_resolution_popup = false;
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }

        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.6f, 1.0f, 0.8f, 1), "Sampling Settings");

        // Use Adaptive Sampling Checkbox
        if (ImGui::Checkbox("Use Adaptive Sampling", &ctx.render_settings.use_adaptive_sampling)) {
             // Log...
        }
        UIWidgets::HelpMarker("Enable intelligent sampling that focuses more rays on noisy areas.");

        if (ctx.render_settings.use_adaptive_sampling) {
            ImGui::Indent(); 
            
            // Min Samples
            ImGui::DragInt("Min Samples", &ctx.render_settings.min_samples, 1.0f, 1, 1024);
            UIWidgets::HelpMarker("Minimum number of samples per pixel before noise variance is evaluated.");

            // Max Samples
            ImGui::DragInt("Max Samples", &ctx.render_settings.max_samples,1.0f, 1, 4096);
            UIWidgets::HelpMarker("Maximum target samples per pixel for noisy areas.");

            // Variance Threshold
            ImGui::SliderFloat("Variance Threshold", &ctx.render_settings.variance_threshold, 0.001f, 1.0f, "%.5f");
            UIWidgets::HelpMarker("Noise tolerance. Lower values mean better quality but longer render times.");
            
            ImGui::Unindent();
        } 
        else {
            // Samples Per Pixel (Sadece Adaptive KAPALIYSA Görünür)
            ImGui::DragInt("Samples Per Pixel (Fixed)", &ctx.render_settings.samples_per_pixel, 1.0f, 1, 64);
            UIWidgets::HelpMarker("Number of samples calculated per frame. Use this for consistent performance.");
        }

        // Max Bounces
        ImGui::Separator();
        if (ImGui::DragInt("Max Bounce", &ctx.render_settings.max_bounces, 1.0f, 1, 30)) {
           // SCENE_LOG_INFO("Max Bounce changed to " + std::to_string(ctx.render_settings.max_bounces));
        }
        UIWidgets::HelpMarker("Maximum number of ray bounces. Higher values improve indirect lighting, reflections, and refractions but increase render time.");

        // Environment
        ImGui::Separator();
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.6f, 1), "Environment");
        if (ImGui::ColorEdit3("Background Color", &ctx.scene.background_color.x)) {
            SCENE_LOG_INFO("Background Color changed to RGB(" +
                std::to_string(int(ctx.scene.background_color.x * 255)) + ", " +
                std::to_string(int(ctx.scene.background_color.y * 255)) + ", " +
                std::to_string(int(ctx.scene.background_color.z * 255)) + ")");
        }
        // --- START/STOP BUTTONS + PROGRESS ---
        ImGui::Separator();
        
        // Progress bar with color gradient
        float progress = ctx.render_settings.render_progress;
        int current = ctx.render_settings.render_current_samples;
        int target = ctx.render_settings.render_target_samples;
        bool is_active = ctx.render_settings.is_rendering_active;
        
        // Progress bar color: red -> yellow -> green
        ImVec4 progress_color;
        if (progress < 0.5f) {
            progress_color = ImVec4(1.0f, progress * 2.0f, 0.0f, 1.0f);  // Red to Yellow
        } else {
            progress_color = ImVec4(1.0f - (progress - 0.5f) * 2.0f, 1.0f, 0.0f, 1.0f);  // Yellow to Green
        }
        
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, progress_color);
        char overlay[64];
        snprintf(overlay, sizeof(overlay), "Sample %d / %d (%.1f%%)", current, target, progress * 100.0f);
        ImGui::ProgressBar(progress, ImVec2(-1, 20), overlay);
        ImGui::PopStyleColor();
        
        // Render time display
        float elapsed = ctx.render_settings.render_elapsed_seconds;
        float remaining = ctx.render_settings.render_estimated_remaining;
        
        if (is_active || elapsed > 0.0f) {
            int elapsed_min = (int)(elapsed / 60.0f);
            int elapsed_sec = (int)elapsed % 60;
            int remaining_min = (int)(remaining / 60.0f);
            int remaining_sec = (int)remaining % 60;
            
            ImGui::TextColored(ImVec4(0.7f, 0.9f, 1.0f, 1.0f), 
                "Elapsed: %02d:%02d  |  Remaining: %02d:%02d", 
                elapsed_min, elapsed_sec, remaining_min, remaining_sec);
        }
        
        ImGui::Spacing();
        
        bool is_paused = ctx.render_settings.is_render_paused;
        bool accumulation_done = (progress >= 1.0f);
        bool is_stopped = is_paused && (current == 0);  // Stopped = paused with no progress
        
        // Start/Pause Toggle Button
        const char* start_pause_label;
        ImVec4 button_color;
        
        if (!is_active && !is_paused) {
            // Not rendering - show "Start Render"
            start_pause_label = "Start Render";
            button_color = ImVec4(0.2f, 0.7f, 0.3f, 1.0f);  // Green
        } else if (is_stopped) {
            // Stopped (after Stop button) - show "Start Render"
            start_pause_label = "Start Render";
            button_color = ImVec4(0.2f, 0.7f, 0.3f, 1.0f);  // Green
        } else if (is_paused) {
            // Paused mid-render - show "Resume"
            start_pause_label = "Resume";
            button_color = ImVec4(0.2f, 0.6f, 0.9f, 1.0f);  // Blue
        } else if (is_active && !accumulation_done) {
            // Rendering - show "Pause"
            start_pause_label = "Pause";
            button_color = ImVec4(0.9f, 0.7f, 0.2f, 1.0f);  // Orange
        } else {
            // Completed - show "Restart"
            start_pause_label = "Restart";
            button_color = ImVec4(0.2f, 0.7f, 0.3f, 1.0f);  // Green
        }
        
        ImGui::PushStyleColor(ImGuiCol_Button, button_color);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(button_color.x * 1.2f, button_color.y * 1.2f, button_color.z * 1.2f, 1.0f));
        
        if (ImGui::Button(start_pause_label, ImVec2(150, 0))) {
            if (!is_active && !is_paused) {
                // Start new render
                ctx.start_render = true;
                ctx.render_settings.is_render_paused = false;
                SCENE_LOG_INFO("Start Render clicked");
            } else if (is_stopped) {
                // Start from stopped state
                ctx.start_render = true;
                ctx.render_settings.is_render_paused = false;
                SCENE_LOG_INFO("Start Render from stopped state");
            } else if (is_paused) {
                // Resume mid-render
                ctx.render_settings.is_render_paused = false;
                ctx.start_render = true;  // Trigger render to continue
                SCENE_LOG_INFO("Render resumed");
            } else if (is_active && !accumulation_done) {
                // Pause
                ctx.render_settings.is_render_paused = true;
                SCENE_LOG_INFO("Render paused");
            } else {
                // Restart (after completion)
                ctx.start_render = true;
                ctx.render_settings.is_render_paused = false;
                // Reset accumulation
                rendering_stopped_gpu = true;
                rendering_stopped_cpu = true;
                SCENE_LOG_INFO("Render restarted");
            }
        }
        ImGui::PopStyleColor(2);
        
        // Stop Button - always enabled when rendering or paused
        ImGui::SameLine();
        bool can_stop = is_active || is_paused;
        
        if (!can_stop) {
            ImGui::BeginDisabled();
        }
        
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.2f, 0.2f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1.0f, 0.3f, 0.3f, 1.0f));
        
        if (ImGui::Button("Stop", ImVec2(150, 0))) {
            // Set stop flags
            rendering_stopped_gpu = true;
            rendering_stopped_cpu = true;
            
            // Reset render state
            ctx.render_settings.is_rendering_active = false;
            ctx.render_settings.is_render_paused = true;  // Keep paused to prevent auto-restart
            ctx.render_settings.render_current_samples = 0;
            ctx.render_settings.render_progress = 0.0f;
            
            // Reset render time
            ctx.render_settings.render_elapsed_seconds = 0.0f;
            ctx.render_settings.render_estimated_remaining = 0.0f;
            ctx.render_settings.avg_sample_time_ms = 0.0f;
            
            // Reset accumulation buffers
            ctx.renderer.resetCPUAccumulation();
            if (ctx.optix_gpu_ptr) {
                ctx.optix_gpu_ptr->resetAccumulation();
            }
            
            SCENE_LOG_WARN("Render stopped and reset");
        }
        
        ImGui::PopStyleColor(2);
        
        if (!can_stop) {
            ImGui::EndDisabled();
        }

    }
    else {
        ImGui::BeginDisabled();
        ImGui::Checkbox("Use OptiX", &ctx.render_settings.use_optix);
        ImGui::Checkbox("Use Denoiser", &ctx.render_settings.use_denoiser);
        ImGui::DragInt("Min Samples", &ctx.render_settings.min_samples);
        ImGui::DragInt("Max Samples", &ctx.render_settings.max_samples);
        ImGui::SliderFloat("Variance Threshold", &ctx.render_settings.variance_threshold, 0.001f, 1.0f, "%.5f");
        ImGui::DragInt("Max Bounce", &ctx.render_settings.max_bounces, 1.0f, 1, 32);
        ImGui::ColorEdit3("Background Color", &ctx.scene.background_color.x);
        ImGui::Button("Start Render", ImVec2(150, 0));
        ImGui::EndDisabled();
    }



    // Animation settings timeline paneline taşındı
    ImGui::Spacing();

    ImGui::Separator();
    ImGui::TextColored(ImVec4(0.6f, 1.0f, 0.8f, 1), "Save Image");
    if (ImGui::Button("Save Image As...", ImVec2(310, 0))) {
        ctx.render_settings.save_image_requested = true;
    }
    // Show current render time text   
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::Spacing();
    drawLogPanelEmbedded(); // *** burada çağırıyoruz ***

            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("World")) {
            drawWorldContent(ctx);
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Camera")) {
            drawCameraContent(ctx);
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Lights")) {
            drawLightsContent(ctx);
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Post-FX")) {
            drawToneMapContent(ctx);
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Controls")) {
             drawControlsContent();
             ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }
    ImGui::End();
    ClampWindowToDisplay();
}

void SceneUI::draw(UIContext& ctx)
{

    ImGuiIO& io = ImGui::GetIO();
    float screen_y = io.DisplaySize.y;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, panel_alpha));

    drawRenderSettingsPanel(ctx, screen_y);
    
    // Timeline panelini en altta göster (Blender tarzı)
    // Timeline panelini en altta göster (Blender tarzı)
    extern bool show_animation_panel;
    if (ctx.scene.initialized && show_animation_panel) {
        drawTimelinePanel(ctx, screen_y);
    }
    
    ImGui::PopStyleColor();
    ImGui::PopStyleVar();
}

void SceneUI::drawControlsContent()
{
     UIWidgets::ColoredHeader("Camera Controls", ImVec4(1.0f, 0.9f, 0.4f, 1.0f));
     UIWidgets::Divider();
     
     ImGui::BulletText("Rotate: Middle Mouse Drag");
     ImGui::BulletText("Pan: Shift + Middle Mouse Drag");
     ImGui::BulletText("Zoom: Mouse Wheel OR Ctrl + Middle Mouse Drag");
     ImGui::BulletText("Move Forward/Back: W / S");
     ImGui::BulletText("Move Left/Right: A / D");
     ImGui::BulletText("Move Up/Down: Q / E");
     
     ImGui::Spacing();
     UIWidgets::ColoredHeader("Shortcuts", ImVec4(0.6f, 0.8f, 1.0f, 1.0f));
     UIWidgets::Divider();
     ImGui::BulletText("Save Image: Ctrl + S"); 
     
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
