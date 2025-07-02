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
   

    int& sample_count;                // dynamic counter (not in settings)
    bool& start_render;              // trigger flag
    std::string& active_model_path;

    bool& apply_tonemap;
    bool& reset_tonemap;

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
