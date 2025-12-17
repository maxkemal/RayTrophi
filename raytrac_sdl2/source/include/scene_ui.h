#pragma once

#include "renderer.h" // SceneData'nın header'ı
#include "OptixWrapper.h"
#include "ColorProcessingParams.h"
#include "ui_modern.h"  // Modern UI sistemi
#include <fstream>


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

     void drawHistogramPanel(UIContext& ctx);      
     void drawLogPanelEmbedded();
     void drawThemeSelector();
     void drawResolutionPanel();
     void drawToneMapContent(UIContext& ctx);
     void drawCameraContent(UIContext& ctx);
     void drawLightsContent(UIContext& ctx);
     void drawRenderSettingsPanel(UIContext& ctx, float screen_y);
     static void ClampWindowToDisplay();
     void drawAnimationSettings(UIContext& ctx);  // Deprecated - timeline panel kullanılıyor
     void drawTimelinePanel(UIContext& ctx, float screen_y);  // Yeni timeline panel (Blender tarzı)
     void drawControlsContent(); // New method for controls/help tab
     void drawWorldContent(UIContext& ctx);
     void draw(UIContext& ctx);
     float panel_alpha = 0.25f; // varsayılan
private:
    bool showResolutionPanel = true; // class üyesi
    bool camera_initialized = false;
    Vec3 camera_initial_pos;
    Vec3 camera_initial_target;
    float camera_initial_fov;
   
    std::atomic<bool> scene_loading = false;
    std::atomic<bool> scene_loading_done = false;
    bool show_scene_log = true;
    float last_applied_width = 0.0f;
    float last_applied_height = 0.0f;
    int last_applied_aspect_w = 0;
    int last_applied_aspect_h = 0;
    // Scene verisini ImGui üzerinden düzenler
    float copyToastTimer = 0.0f;   // süre sayacı
    bool copyToastActive = false;  // toast açık mı
    std::string logTitle = "Scene Log";
    float titleResetTime = 0.0f;
    bool titleChanged = false;

};
