#include "scene_ui.h"
#include "imgui.h"
#include <string>
#include "scene_data.h"
#include <windows.h>
#include <commdlg.h>
#include <string>

std::string openFileDialog(const char* filter = "Tüm Dosyalar\0*.*\0") {
    char filename[MAX_PATH] = "";
    OPENFILENAMEA ofn;
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL; // SDL kullanıyorsan pencere handle'ı da olabilir
    ofn.lpstrFilter = filter;
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;
    ofn.lpstrTitle = "Bir dosya seçin";

    if (GetOpenFileNameA(&ofn)) {
        return std::string(filename);
    }
    return ""; // iptal edildi
}
std::string saveFileDialog(const char* filter = "PNG Dosyası\0*.png\0") {
    char filename[MAX_PATH] = "";
    OPENFILENAMEA ofn;
    ZeroMemory(&ofn, sizeof(ofn));

    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL; // SDL window handle verilebilir
    ofn.lpstrFilter = filter;
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.Flags = OFN_OVERWRITEPROMPT; // 👈 önemli: üzerine yaz uyarısı verir
    ofn.lpstrTitle = "Render Görüntüsünü Kaydet";

    if (GetSaveFileNameA(&ofn)) {
        return std::string(filename);
    }
    return ""; // iptal edildi
}


bool SaveSurface(SDL_Surface* surface, const char* file_path) {
    SDL_Surface* surface_to_save = SDL_ConvertSurfaceFormat(surface, SDL_PIXELFORMAT_RGB24, 0);
    /* int imgFlags = IMG_INIT_PNG;
     if (!(IMG_Init(imgFlags) & imgFlags)) {
         SDL_Log("SDL_image could not initialize! SDL_image Error: %s\n", IMG_GetError());
         SDL_Quit();
         return 1;
     }*/

    if (surface_to_save == NULL) {
        SDL_Log("Couldn't convert surface: %s", SDL_GetError());
        return false;
    }

    int result = IMG_SavePNG(surface_to_save, file_path);
    SDL_FreeSurface(surface_to_save);

    if (result != 0) {
        SDL_Log("Failed to save image: %s", IMG_GetError());
        return false;
    }

    return true;
}
static std::string active_model_path = "Henüz dosya seçilmedi.";
void SceneUI::drawHistogramPanel(UIContext& ctx) {
    ImGuiIO& io = ImGui::GetIO();
    float screen_y = io.DisplaySize.y;

    static std::vector<float> r_data(256, 0.0f);
    static std::vector<float> g_data(256, 0.0f);
    static std::vector<float> b_data(256, 0.0f);
    static std::vector<float> luma_data(256, 0.0f);

    std::fill(r_data.begin(), r_data.end(), 0.0f);
    std::fill(g_data.begin(), g_data.end(), 0.0f);
    std::fill(b_data.begin(), b_data.end(), 0.0f);
    std::fill(luma_data.begin(), luma_data.end(), 0.0f);

    int total_pixels = 0;

    if (ctx.surface) {
        SDL_LockSurface(ctx.surface);
        uint8_t* pixels = static_cast<uint8_t*>(ctx.surface->pixels);
        int pitch = ctx.surface->pitch;

        for (int y = 0; y < ctx.surface->h; ++y) {
            for (int x = 0; x < ctx.surface->w; ++x) {
                uint8_t* pixel = pixels + y * pitch + x * 4;
                uint8_t r = pixel[2];
                uint8_t g = pixel[1];
                uint8_t b = pixel[0];

                r_data[r]++;
                g_data[g]++;
                b_data[b]++;

                float luma = 0.2126f * r + 0.7152f * g + 0.0722f * b;
                int l_idx = std::fmin(255, static_cast<int>(luma));
                luma_data[l_idx]++;

                total_pixels++;
            }
        }

        SDL_UnlockSurface(ctx.surface);
    }

    if (total_pixels > 0) {
        for (int i = 0; i < 256; ++i) {
            r_data[i] /= total_pixels;
            g_data[i] /= total_pixels;
            b_data[i] /= total_pixels;
            luma_data[i] /= total_pixels;
        }
    }

    ImGui::SetNextWindowSize(ImVec2(500, 240), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(1000, screen_y - 540), ImGuiCond_FirstUseEver);
    ImGui::Begin("Histogram");

    ImGui::Text("Hover to inspect. RGB + Luminance");

    ImVec2 canvas_size(460, 140);
    ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    draw_list->AddRect(canvas_pos, ImVec2(canvas_pos.x + canvas_size.x, canvas_pos.y + canvas_size.y), IM_COL32(100, 100, 100, 255));

    float bin_width = canvas_size.x / 256.0f;

    for (int i = 0; i < 256; ++i) {
        float x = canvas_pos.x + i * bin_width;
        float base_y = canvas_pos.y + canvas_size.y;

        float rh = r_data[i] * canvas_size.y;
        float gh = g_data[i] * canvas_size.y;
        float bh = b_data[i] * canvas_size.y;
        float lh = luma_data[i] * canvas_size.y;

        draw_list->AddLine(ImVec2(x, base_y), ImVec2(x, base_y - rh), IM_COL32(255, 0, 0, 180));
        draw_list->AddLine(ImVec2(x, base_y), ImVec2(x, base_y - gh), IM_COL32(0, 255, 0, 180));
        draw_list->AddLine(ImVec2(x, base_y), ImVec2(x, base_y - bh), IM_COL32(0, 120, 255, 180));
        draw_list->AddLine(ImVec2(x, base_y), ImVec2(x, base_y - lh), IM_COL32(220, 220, 220, 80));
    }

    ImGui::InvisibleButton("histogram_hover", canvas_size);
    if (ImGui::IsItemHovered()) {
        ImVec2 mouse = ImGui::GetMousePos();
        int hover_index = static_cast<int>((mouse.x - canvas_pos.x) / bin_width);
        hover_index = std::clamp(hover_index, 0, 255);

        ImGui::BeginTooltip();
        ImGui::Text("Bin: %d", hover_index);
        ImGui::Text("R: %.4f", r_data[hover_index]);
        ImGui::Text("G: %.4f", g_data[hover_index]);
        ImGui::Text("B: %.4f", b_data[hover_index]);
        ImGui::Text("Luminance: %.4f", luma_data[hover_index]);
        ImGui::EndTooltip();
    }

    ImGui::Dummy(canvas_size);
    ImGui::End();
}

void SceneUI::drawLogConsole() {
    static ImGuiTextBuffer log_buffer;
    static bool scroll_to_bottom = true;

    ImGui::SetNextWindowSize(ImVec2(500, 200), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(20, 40), ImGuiCond_FirstUseEver);
    ImGui::Begin("Console Log");

    if (ImGui::Button("Clear")) log_buffer.clear();
    ImGui::SameLine();
    if (ImGui::Button("Add Log")) {
        log_buffer.appendf("New render pass started...\n");
        scroll_to_bottom = true;
    }

    ImGui::Separator();
    ImGui::BeginChild("LogScrollRegion", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);
    ImGui::TextUnformatted(log_buffer.begin());
    if (scroll_to_bottom)
        ImGui::SetScrollHereY(1.0f);
    scroll_to_bottom = false;
    ImGui::EndChild();

    ImGui::End();
}

void SceneUI::drawToneMapPanel(UIContext& ctx) {
    ImGuiIO& io = ImGui::GetIO();
    float screen_y = io.DisplaySize.y;

    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.08f, 0.09f, 0.12f, 0.9f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 10.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(6, 6));

    ImGui::SetNextWindowSize(ImVec2(340, 260), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(700, screen_y - 280), ImGuiCond_FirstUseEver);

    ImGui::Begin(" Tone Mapping", nullptr);

    ImGui::TextColored(ImVec4(1.0f, 0.65f, 0.6f, 1), "Post-Processing Controls");
    ImGui::Separator();

    ImGui::SliderFloat(" Gamma", &ctx.color_processor.params.global_gamma, 0.5f, 3.0f, "%.2f");
    ImGui::SliderFloat(" Exposure", &ctx.color_processor.params.global_exposure, 0.1f, 5.0f, "%.2f");
    ImGui::SliderFloat(" Saturation", &ctx.color_processor.params.saturation, 0.0f, 2.0f, "%.2f");
    ImGui::SliderFloat(" Temperature (K)", &ctx.color_processor.params.color_temperature, 1000.0f, 10000.0f, "%.0f");

    // Tonemap tipi seçimi
    const char* tone_names[] = { "AGX", "ACES", "Uncharted", "Filmic", "None" };
    int selected_tone = static_cast<int>(ctx.color_processor.params.tone_mapping_type);
    if (ImGui::Combo("Tonemapping", &selected_tone, tone_names, IM_ARRAYSIZE(tone_names))) {
        ctx.color_processor.params.tone_mapping_type = static_cast<ToneMappingType>(selected_tone);
    }
    ImGui::Separator();
    ImGui::Text("Effects");

    ImGui::Checkbox(" Vignette", &ctx.color_processor.params.enable_vignette);

    if (ctx.color_processor.params.enable_vignette) {
        ImGui::SliderFloat(" Vignette Strength", &ctx.color_processor.params.vignette_strength, 0.0f, 2.0f, "%.2f");
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    if (ImGui::Button(" Apply Tonemap", ImVec2(150, 0))) ctx.apply_tonemap = true;
    ImGui::SameLine();
    if (ImGui::Button(" Reset", ImVec2(150, 0))) ctx.reset_tonemap = true;

    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor();
    ImGui::End();
}



void SceneUI::draw(UIContext& ctx) {
    ImGuiIO& io = ImGui::GetIO();
    float screen_y = io.DisplaySize.y;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 10.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(6, 6));

    ImGui::SetNextWindowSize(ImVec2(340, 260), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(20, screen_y - 300), ImGuiCond_FirstUseEver);

    ImGui::Begin("Camera", nullptr);

    if (ctx.scene.camera) {
        static bool targetLock = true;

        Vec3 pos = ctx.scene.camera->lookfrom;
        Vec3 target = ctx.scene.camera->lookat;
        float fov = static_cast<float>(ctx.scene.camera->vfov);

        ImGui::Text("Position and Target");
        ImGui::PushItemWidth(300);

        bool pos_changed = ImGui::DragFloat3("Position", &pos.x, 0.01f);

        bool target_changed = false;
        if (!targetLock)
            target_changed = ImGui::DragFloat3("Target", &target.x, 0.01f);
        else
            ImGui::BeginDisabled(), ImGui::DragFloat3("Target", &target.x, 0.1f), ImGui::EndDisabled();

        ImGui::PopItemWidth();
        ImGui::Checkbox("Lock Target", &targetLock);

        ImGui::Text("View Parameters");
        if (ImGui::SliderFloat("FOV", &fov, 10.0f, 120.0f)) {
            ctx.scene.camera->vfov = fov;
            ctx.scene.camera->fov = fov;
            ctx.scene.camera->update_camera_vectors();
        }
        ImGui::Separator();
        ImGui::Text("Mouse Control");
        ImGui::Checkbox("Enable Mouse Look", &ctx.mouse_control_enabled);
        if (ctx.mouse_control_enabled)
            ImGui::SliderFloat("Sensitivity", &ctx.mouse_sensitivity, 0.001f, 0.2f, "%.3f");

        // Apply position changes
        if (pos_changed) {
            if (targetLock)
                ctx.scene.camera->moveToTargetLocked(pos);
            else {
                ctx.scene.camera->lookfrom = pos;
                ctx.scene.camera->origin = pos;
                ctx.scene.camera->update_camera_vectors();
            }
        }

        // Apply target changes
        if (target_changed) {
            ctx.scene.camera->lookat = target;
            ctx.scene.camera->update_camera_vectors();
        }
    }

    ImGui::End();

    // === LIGHTS PANEL ===
    ImGui::SetNextWindowSize(ImVec2(340, 260), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(340, screen_y - 280), ImGuiCond_FirstUseEver);
    ImGui::Begin("Lights", nullptr);
    ImGui::TextColored(ImVec4(0.6f, 0.9f, 1.0f, 1), "Scene Lights");
    for (size_t i = 0; i < ctx.scene.lights.size(); ++i) {
        auto light = ctx.scene.lights[i];
        if (!light) continue;

        std::string label = "Light #" + std::to_string(i);
        if (ImGui::TreeNode(label.c_str())) {
            ImGui::DragFloat3("Position", &light->position.x, 0.1f);
            ImGui::DragFloat3("Direction", &light->direction.x, 0.1f);
            ImGui::ColorEdit3("Color", &light->color.x);
            ImGui::DragFloat("Intensity", &light->intensity, 0.1f, 0.0f, 1000.0f);
            ImGui::DragFloat("Radius", &light->radius, 0.01f, 0.0f, 10.0f);
            ImGui::TreePop();
        }
    }
    ImGui::End();
	// TON MAP PANEL
    drawToneMapPanel(ctx);
	
    // === RENDER SETTINGS PANEL ===
    ImGui::SetNextWindowSize(ImVec2(360, 360), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(1040, screen_y - 280), ImGuiCond_FirstUseEver);
    ImGui::Begin("Render Settings", nullptr);

    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.6f, 1), "Model & Scene");

    if (ImGui::Button("Load Model", ImVec2(150, 0))) {
#ifdef _WIN32
        std::string file = openFileDialog("3D Files\0*.gltf\0");
        if (!file.empty()) {
            active_model_path = file;
          
            ctx.scene.clear();
            ctx.renderer.create_scene(ctx.scene, ctx.optix_gpu_ptr, file);            
                ctx.scene.camera->update_camera_vectors(); 
                render_settings.start_animation_render = false;
				ctx.start_render = false;				
        }

#endif
    }
    ImGui::SameLine();
    ImGui::TextWrapped("Model: %s", active_model_path.c_str());

    ImGui::Separator();

    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.6f, 1), "Render Engine");
    ImGui::Checkbox("Use OptiX", &render_settings.use_optix);

    ImGui::SliderInt("Samples Per Pixel", &render_settings.samples_per_pixel, 4, 4096);
    ImGui::SliderInt("Samples Per Pass", &render_settings.samples_per_pass, 4, 4096);
    ImGui::SliderInt("Max Bounces", &render_settings.max_bounces, 1, 64);
    
    ImGui::Separator();
    ImGui::TextColored(ImVec4(0.6f, 1.0f, 0.8f, 1), "Adaptive Sampling");

    ImGui::Checkbox("Use Adaptive", &render_settings.use_adaptive_sampling);
    if (render_settings.use_adaptive_sampling) {
        ImGui::SliderInt("Min Samples", &render_settings.min_samples, 4, 1024);
        ImGui::SliderInt("Max Samples", &render_settings.max_samples, 4, 2048);
        ImGui::SliderFloat("Variance Threshold", &render_settings.variance_threshold, 0.00001f, 0.01f, "%.5f");
    }

   

    ImGui::Separator();
    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.6f, 1), "Environment");
    ImGui::ColorEdit3("Background Color", &ctx.scene.background_color.x);

    ImGui::Separator();

    if (ImGui::Button("Start Render", ImVec2(150, 0))) ctx.start_render = true;
    ImGui::SameLine();
    if (ImGui::Button("Stop", ImVec2(150, 0))) ctx.start_render = false;
    ImGui::Separator();
    ImGui::TextColored(ImVec4(0.7f, 0.9f, 1.0f, 1), "Animation");

    ImGui::SliderFloat("Duration (sec)", &render_settings.animation_duration, 1.0f, 60.0f);
    ImGui::SliderInt("FPS", &render_settings.animation_fps, 1, 60);
    ImGui::Separator();
    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.6f, 1), "Animation Rendering");

    if (ImGui::Button("Start CPU Animation Render", ImVec2(310, 0))) {
        render_settings.start_animation_render = true;
    }
    ImGui::Spacing();
    if (ImGui::Button("Save Image As...", ImVec2(310, 0))) {
        std::string path = saveFileDialog("PNG Files\0*.png\0");
        if (!path.empty()) {
            SaveSurface(ctx.surface, path.c_str());
            SDL_Log("Image saved to: %s", path.c_str());
        }
    }

    ImGui::End();

    ImGui::PopStyleVar(2);
}

