#include "scene_ui.h"
#include "imgui.h"
#include <string>
#include "scene_data.h"
#include <windows.h>
#include <commdlg.h>
#include <string>
static void HelpMarker(const char* desc) {
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(450.0f);
        ImGui::TextUnformatted(desc);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}


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
        HelpMarker("Field of View in degrees.\nWider FOV captures more of the scene but increases distortion.");

        ImGui::Text("Depth of Field");

        float& aperture = ctx.scene.camera->aperture;
        float& focus_dist = ctx.scene.camera->focus_dist;

        bool aperture_changed = ImGui::SliderFloat("Aperture", &aperture, 0.0f, 5.0f, "%.2f");
        HelpMarker("Controls the size of the lens opening.\nHigher values produce stronger bokeh (background blur).");
        bool focus_changed = ImGui::DragFloat("Focus Distance", &focus_dist, 0.05f, 0.01f, 100.0f);
        HelpMarker("Sets the distance from the camera where objects appear sharp.\nObjects in front or behind will blur based on aperture.");
        if (aperture_changed || focus_changed) {
            ctx.scene.camera->lens_radius = aperture * 0.5f;
            ctx.scene.camera->update_camera_vectors();
        }
        ImGui::Text("Bokeh Settings");
        ImGui::SliderInt("Blade Count", &ctx.scene.camera->blade_count, 3, 12);
        HelpMarker("Number of aperture blades.\nAffects the shape of the bokeh highlights (e.g. triangle, pentagon).");
        ImGui::Separator();
        ImGui::Text("Mouse Control");
        ImGui::Checkbox("Enable Mouse Look", &ctx.mouse_control_enabled);
        HelpMarker("Enables first-person camera control with mouse movement.");
        if (ctx.mouse_control_enabled)
            ImGui::SliderFloat("Sensitivity", &ctx.mouse_sensitivity, 0.01f, 0.5f, "%.3f");
        HelpMarker("Controls how fast the camera rotates with mouse movement.");
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

    ImGui::SetNextWindowSize(ImVec2(340, 260), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(340, screen_y - 280), ImGuiCond_FirstUseEver);
    ImGui::Begin("Lights", nullptr);
    ImGui::TextColored(ImVec4(0.6f, 0.9f, 1.0f, 1), "Scene Lights");

    for (size_t i = 0; i < ctx.scene.lights.size(); ++i) {
        auto light = ctx.scene.lights[i];
        if (!light) continue;

        std::string label = "Light #" + std::to_string(i);
        if (ImGui::TreeNode(label.c_str())) {
            const char* lightTypeNames[] = { "Point", "Directional", "Spot", "Area" };
            int index = static_cast<int>(light->type());
            if (index >= 0 && index < 4)
                ImGui::Text("Type: %s", lightTypeNames[index]);
            else
                ImGui::Text("Type: Unknown (%d)", index);

            HelpMarker("Type of the light source.\nAffects how the light interacts with geometry.");

            ImGui::DragFloat3("Position", &light->position.x, 0.1f);
            HelpMarker("Position of the light in world space.");

            if (light->type() == LightType::Directional || light->type() == LightType::Spot) {
                ImGui::DragFloat3("Direction", &light->direction.x, 0.1f);
                HelpMarker("Direction the light is pointing.");
            }

            ImGui::ColorEdit3("Color", &light->color.x);
            HelpMarker("Base color of the emitted light.");

            ImGui::DragFloat("Intensity", &light->intensity, 0.1f, 0.0f, 1000.0f);
            HelpMarker("Brightness of the light.");

            if (light->type() == LightType::Point || light->type() == LightType::Area|| light->type() == LightType::Directional) {
                ImGui::DragFloat("Radius", &light->radius, 0.01f, 0.01f, 100.0f);
                HelpMarker("Physical size of the light.");
            }

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
    static int new_width = image_width;
	float new_aspect_ratio = aspect_ratio;
    static int new_height = image_height;
    static int aspect_w = 16;
    static int aspect_h = 9;

    ImGui::Text("Render Resolution");
    ImGui::InputInt("Width", &new_width);
    ImGui::InputInt("Height", &new_height);
    ImGui::InputInt("Aspect W", &aspect_w);
    ImGui::SameLine();
    ImGui::InputInt("Aspect H", &aspect_h);

    if (ImGui::Button("Apply")) {
        if (aspect_h != 0) {
            pending_aspect_ratio = static_cast<float>(aspect_w) / static_cast<float>(aspect_h);
        }
        else {
            pending_aspect_ratio = 1.0f; // Güvenli varsayılan
        }
        pending_width = new_width;
        pending_height = new_height;
		aspect_ratio = pending_aspect_ratio;
        pending_resolution_change = true;
    }

    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.6f, 1), "Model & Scene");

    if (ImGui::Button("Load Model", ImVec2(150, 0))) {
#ifdef _WIN32
        std::string file = openFileDialogW(
            L"3D Files\0*.gltf;*.glb;*.fbx;*.obj;*.dae;*.3ds;*.blend;*.ply;*.stl\0All Files\0*.*\0"
        );


        if (!file.empty()) {
            active_model_path = file;
			use_embree = ctx.render_settings.UI_use_embree;
            ctx.scene.clear();
            ctx.renderer.create_scene(ctx.scene, ctx.optix_gpu_ptr, file);            
                ctx.scene.camera->update_camera_vectors(); 
                ctx.render_settings.start_animation_render = false;
				ctx.start_render = false;				
        }

#endif
    }
    ImGui::SameLine();
    ImGui::TextWrapped("Model: %s", active_model_path.c_str());
    ImGui::Separator();
    ImGui::PushItemWidth(180);
    ImGui::Separator();
    ImGui::PushItemWidth(180);
    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.6f, 1), "Render Engine");

    // --- GPU Section ---
    ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1), "GPU (OptiX)");
    if (!g_hasOptix) ctx.render_settings.use_optix = false;
    ImGui::BeginDisabled(!g_hasOptix);
    ImGui::Checkbox("Use OptiX", &ctx.render_settings.use_optix);
    ImGui::SameLine();
    HelpMarker("Enables GPU acceleration via NVIDIA OptiX. Requires an RTX-class GPU.");
    ImGui::EndDisabled();

    // --- CPU Section ---
    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.6f, 1), "CPU (BVH)");
    const char* bvh_options[] = { "Embree", "In-house" };
    static int current_bvh = ctx.render_settings.UI_use_embree ? 0 : 1;
    if (ImGui::Combo("BVH Type", &current_bvh, bvh_options, IM_ARRAYSIZE(bvh_options))) {
        ctx.render_settings.UI_use_embree = (current_bvh == 0);
        ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
    }
    ImGui::SameLine();
    HelpMarker("Select which BVH structure to use for acceleration. Embree = highly optimized, In-house = custom implementation.");

    // --- Denoiser Section ---
    ImGui::TextColored(ImVec4(0.8f, 1.0f, 0.6f, 1), "Denoiser");
    ImGui::Checkbox("Use Denoiser", &ctx.render_settings.use_denoiser);
    ImGui::SameLine();
    HelpMarker("Applies denoising to reduce noise after rendering. Based on OIDN.");
    if (ctx.render_settings.use_denoiser) {
        ImGui::SliderFloat("Denoiser Blend", &ctx.render_settings.denoiser_blend_factor, 0.0f, 1.0f, "%.2f");
        ImGui::SameLine();
        HelpMarker("Blends the denoised result with the original. 1 = fully denoised, 0 = original image.");
    }

    ImGui::Separator();

    ImGui::TextColored(ImVec4(0.6f, 1.0f, 0.8f, 1), "Adaptive Sampling");

    // Min Samples
    ImGui::DragInt("Min Samples", &ctx.render_settings.min_samples, 1.0f, 1, 1024);
    ImGui::SameLine(); HelpMarker("Minimum number of samples per pixel before noise variance is evaluated.");

    // Max Samples
    ImGui::DragInt("Max Samples", &ctx.render_settings.max_samples, 1.0f, 1, 2048);
    ImGui::SameLine(); HelpMarker("Maximum number of samples per pixel. Higher values allow cleaner results but take longer.");

    // Variance Threshold
    ImGui::SliderFloat("Variance Threshold", &ctx.render_settings.variance_threshold, 0.001f, 1.0f, "%.5f");
    ImGui::SameLine(); HelpMarker("Pixels with variance below this threshold will stop sampling early. Lower = cleaner but slower.");
	// Max Bounces
    ImGui::Separator();
    ImGui::DragInt("Max Bounce", &ctx.render_settings.max_bounces, 1.0f, 1, 32);
    ImGui::SameLine(); HelpMarker("Maximum number of ray bounces. Higher values improve indirect lighting, reflections, and refractions but increase render time.");

    ImGui::Separator();
    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.6f, 1), "Environment");
    ImGui::ColorEdit3("Background Color", &ctx.scene.background_color.x);

    ImGui::Separator();
    if (ImGui::Button("Start Render", ImVec2(150, 0))) ctx.start_render = true;
    ImGui::SameLine();
    if (ImGui::Button("Stop", ImVec2(150, 0))) ctx.start_render = false;
   
    ImGui::Separator();
    // Show current render time text
    ImGui::Text("Last Render Time: %.4f sec", last_render_time_ms);
    // Clamp to [0, 1] range for progress bar
    float normalized = std::fmin(last_render_time_ms / 1000.0f, 1.0f);  // 1 saniyeyi 100%
    // Label inside progress bar
    char label[64];
    snprintf(label, sizeof(label), "%.4f sec", last_render_time_ms);
    // Bar visualization
    ImGui::ProgressBar(normalized, ImVec2(-1, 25), label);

    ImGui::Separator();
    ImGui::TextColored(ImVec4(0.7f, 0.9f, 1.0f, 1), "Animation");

    ImGui::SliderFloat("Duration (sec)", &ctx.render_settings.animation_duration, 0.1f, 60.0f);
    ImGui::SameLine(); HelpMarker("Length of the animation in seconds.");

    ImGui::SliderInt("FPS", &ctx.render_settings.animation_fps, 1, 60);
    ImGui::SameLine(); HelpMarker("Frames per second for animation rendering.");

    ImGui::Separator();
    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.6f, 1), "Animation Rendering");

    if (ImGui::Button("Start CPU Animation Render", ImVec2(310, 0))) {
        ctx.render_settings.start_animation_render = true;
		ctx.start_render = true;  // Start render immediately
    }
    ImGui::Spacing();
    if (ImGui::Button("Save Image As...", ImVec2(310, 0))) {
        ctx.render_settings.save_image_requested = true;
    }

    ImGui::PopItemWidth();
    ImGui::End();

    ImGui::PopStyleVar(2);
}

