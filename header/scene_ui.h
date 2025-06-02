#pragma once

#include "renderer.h" // SceneData'nın header'ı
#include "OptixWrapper.h"
#include "ColorProcessingParams.h"

struct UIContext {
    SceneData& scene;
    Renderer& renderer;
    OptixWrapper* optix_gpu_ptr;
    ColorProcessor& color_processor;
    RenderSettings& render_settings;
    SDL_Surface* surface;
    SDL_Surface* original_surface;
    SDL_Window* window;
    int& sample_count;
    int& sample_per_pass;
    float& animation_duration;
    float& animation_fps;
    bool& start_render;
    std::string& active_model_path;
    bool& apply_tonemap;
    bool& reset_tonemap;

    // Yeni eklenen alanlar:
    bool& mouse_control_enabled;
    float& mouse_sensitivity;
};



class SceneUI {
public:

    // Scene verisini ImGui üzerinden düzenler

     void drawHistogramPanel(UIContext& ctx);

     void drawLogConsole();
    bool camera_initialized = false;
    Vec3 camera_initial_pos;
    Vec3 camera_initial_target;
    float camera_initial_fov;


     void drawToneMapPanel(UIContext& ctx);

     void draw(UIContext& ctx);
};
