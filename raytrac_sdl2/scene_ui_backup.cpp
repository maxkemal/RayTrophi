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
        
        // İlk kez açıldığında dosyadan gelen değerleri kullan
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
        if (is_rendering) ImGui::BeginDisabled();
        
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
        
        if (is_rendering) ImGui::EndDisabled();
        
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
                    playback_frame = start_frame; // Loop
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
        
        // Output folder
        ImGui::Text("Output:");
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "%s", 
            ctx.render_settings.animation_output_folder.empty() ? "Not selected" : ctx.render_settings.animation_output_folder.c_str());
        ImGui::SameLine();
        
        if (ImGui::Button("Browse", ImVec2(70, 0))) {
#ifdef _WIN32
            wchar_t folder[MAX_PATH] = L"";
            BROWSEINFOW bi = { 0 };
            bi.lpszTitle = L"Select Output Folder for Animation Frames";
            bi.ulFlags = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE;
            bi.hwndOwner = GetActiveWindow();
            
            LPITEMIDLIST pidl = SHBrowseForFolderW(&bi);
            if (pidl != nullptr) {
                if (SHGetPathFromIDListW(pidl, folder)) {
                    int size_needed = WideCharToMultiByte(CP_UTF8, 0, folder, -1, nullptr, 0, nullptr, nullptr);
                    std::string utf8_path(size_needed, 0);
                    WideCharToMultiByte(CP_UTF8, 0, folder, -1, utf8_path.data(), size_needed, nullptr, nullptr);
                    utf8_path.resize(size_needed - 1);
                    ctx.render_settings.animation_output_folder = utf8_path;
                    SCENE_LOG_INFO("Animation output folder: " + ctx.render_settings.animation_output_folder);
                }
                CoTaskMemFree(pidl);
            }
#endif
        }
        
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
            if (!ctx.render_settings.animation_output_folder.empty()) {
                ctx.render_settings.start_animation_render = true;
                ctx.start_render = true;
                is_playing = false; // Playback'i durdur
                SCENE_LOG_INFO("Starting animation render...");
            } else {
                SCENE_LOG_ERROR("Please select an output folder first!");
            }
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
        ImGui::BeginChild("scroll_log", ImVec2(0, 150), true);

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
            UIWidgets::SliderWithHelp("Sensitivity", &ctx.mouse_sensitivity, 0.01f, 0.5f,
                                       "Mouse movement sensitivity", "%.3f");
        }
        UIWidgets::EndSection();
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
void SceneUI::drawRenderSettingsPanel(UIContext& ctx, float screen_y)
{
    ImGui::SetNextWindowSize(ImVec2(400, screen_y - 20), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(1040, screen_y - 420), ImGuiCond_FirstUseEver);
    ImGui::Begin("Properties", nullptr);

    if (ImGui::BeginTabBar("MainPropertiesTabs"))
    {
        if (ImGui::BeginTabItem("System")) {
            drawThemeSelector();
            drawResolutionPanel();
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
            scene_loading = true;
            scene_loading_done = false;
            active_model_path = file;

            std::thread loader_thread([this, file, &ctx]() {
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



        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.6f, 1.0f, 0.8f, 1), "Adaptive Sampling");

        // Min Samples
        if (ImGui::DragInt("Min Samples", &ctx.render_settings.min_samples, 1.0f, 1, 1024)) {
           // SCENE_LOG_INFO("Min Samples changed to " + std::to_string(ctx.render_settings.min_samples));
        }
        UIWidgets::HelpMarker("Minimum number of samples per pixel before noise variance is evaluated.");

        // Max Samples
        if (ImGui::DragInt("Max Samples", &ctx.render_settings.max_samples, 1.0f, 1, 2048)) {
            //SCENE_LOG_INFO("Max Samples changed to " + std::to_string(ctx.render_settings.max_samples));
        }
        UIWidgets::HelpMarker("Maximum number of samples per pixel. Higher values allow cleaner results but take longer.");

        // Variance Threshold
        if (ImGui::SliderFloat("Variance Threshold", &ctx.render_settings.variance_threshold, 0.001f, 1.0f, "%.5f")) {
           // SCENE_LOG_INFO("Variance Threshold changed to " + std::to_string(ctx.render_settings.variance_threshold));
        }
        UIWidgets::HelpMarker("Pixels with variance below this threshold will stop sampling early. Lower = cleaner but slower.");

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
        // -- - START BUTTON-- -
        ImGui::Separator();
        if (rendering_in_progress) {
            ImGui::BeginDisabled(); // Butonu pasif yap
        }
        if (ImGui::Button("Start Render", ImVec2(150, 0))) {
            ctx.start_render = true;
        }
        if (rendering_in_progress) {
            ImGui::EndDisabled();
        }
        // --- STOP BUTTON ---
        ImGui::SameLine();

        if (!rendering_in_progress)
            ImGui::BeginDisabled();               // Render durduysa Stop pasif

        if (ImGui::Button("Stop", ImVec2(150, 0))) {
            rendering_stopped_gpu = true;        // Render durdur sinyali
            rendering_stopped_cpu = true;
        }

        if (!rendering_in_progress)
            ImGui::EndDisabled();

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




    // Show current render time text
    ImGui::Text("Last Render Time: %.4f sec", last_render_time_ms);
    // Clamp to [0, 1] range for progress bar
    float normalized = std::fmin(last_render_time_ms / 1000.0f, 1.0f); // 1 saniyeyi 100%
    // Label inside progress bar
    char label1[64];
    snprintf(label1, sizeof(label), "%.4f sec", last_render_time_ms);
    // Bar visualization
    ImGui::ProgressBar(normalized, ImVec2(-1, 25), label1);

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
    if (ctx.scene.initialized) {
        drawTimelinePanel(ctx, screen_y);
    }
    
    ImGui::PopStyleColor();
    ImGui::PopStyleVar();
}




