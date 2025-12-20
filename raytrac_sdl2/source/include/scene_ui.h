#pragma once

#include "renderer.h" // SceneData'nın header'ı
#include "OptixWrapper.h"
#include "ColorProcessingParams.h"
#include "ui_modern.h"  // Modern UI sistemi
#include "SceneSelection.h"  // Scene selection system
#include "SceneHistory.h"  // Undo/Redo system
#include "SceneCommand.h"  // TransformState struct
#include <fstream>
#include <map>
#include <vector>


struct UIContext {
    SceneData& scene;
    Renderer& renderer;
    OptixWrapper* optix_gpu_ptr;
    ColorProcessor& color_processor;
    RenderSettings& render_settings;
    SceneSelection& selection;  // Scene selection manager
   

    int& sample_count;                // dynamic counter (not in settings)
    bool& start_render;              // trigger flag
    std::string& active_model_path;

    bool& apply_tonemap;
    bool& reset_tonemap;

    bool& mouse_control_enabled;
    float& mouse_sensitivity;
    struct SDL_Texture* ray_texture; // Forward decl or void* if header dependency issues, but SDL.h is included in Renderer.h
    
    // Note: SDL_Texture* requires SDL.h. Renderer.h includes it.
    // If SceneUI.h doesn't include SDL, we need forward declaration or include.
    // Renderer.h is included at top (line 3). So SDL.h is available.

	
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
     void drawTimelineContent(UIContext& ctx); // Embedded timeline logic
     void drawMainMenuBar(UIContext& ctx); // New Main Menu Bar
     void drawControlsContent(); // New method for controls/help tab
     void drawWorldContent(UIContext& ctx);
     void drawSceneHierarchy(UIContext& ctx);  // Scene hierarchy / outliner panel
     void drawSelectionBoundingBox(UIContext& ctx);  // Draw bounding box for selected object
     void drawTransformGizmo(UIContext& ctx);  // ImGuizmo transform gizmo   
     void drawCameraGizmos(UIContext& ctx);    // Draw camera icons in viewport
     void draw(UIContext& ctx);
     void handleMouseSelection(UIContext& ctx); // Publicly accessible for Main loop call
     void triggerDelete(UIContext& ctx); // Trigger delete operation (for Menu and Key)
     void invalidateCache() { mesh_cache_valid = false; }
     
     // Project System Helpers
     void updateProjectFromScene(UIContext& ctx);  // Sync scene state to project data
     void addProceduralPlane(UIContext& ctx);      // Add a procedural plane mesh
     void addProceduralCube(UIContext& ctx);       // Add a procedural cube mesh
     float panel_alpha = 0.5f; // varsayılan
     
     // Scene loading state (public for Main.cpp popup access)
     std::atomic<bool> scene_loading{false};
     std::atomic<bool> scene_loading_done{false};
     std::atomic<int> scene_loading_progress{0};  // 0-100
     std::string scene_loading_stage = "";        // Current stage description
     
private:
    bool showSidePanel = true;
    bool show_controls_window = false; // Controls/Help window visibility
    bool showResolutionPanel = true; // class üyesi
    bool camera_initialized = false;
    Vec3 camera_initial_pos;
    Vec3 camera_initial_target;
    float camera_initial_fov;
   
    bool show_scene_log = false; // Default closed
    float side_panel_width = 360.0f; // Resizable Left Panel width
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

    // Mesh Selection Optimization Cache
    // Groups triangles by nodeName for O(1) access during transform
    // Stores pair of {original_index, triangle}
    // Groups triangles by nodeName for O(1) access during transform
    // Stores pair of {original_index, triangle}
    std::map<std::string, std::vector<std::pair<int, std::shared_ptr<class Triangle>>>> mesh_cache;
    // Sequential cache for ImGui Clipper (Visualization)
    std::vector<std::pair<std::string, std::vector<std::pair<int, std::shared_ptr<class Triangle>>>>> mesh_ui_cache;


    bool mesh_cache_valid = false;
    void rebuildMeshCache(const std::vector<std::shared_ptr<class Hittable>>& objects);    
    // Interaction State
    bool is_dragging = false; // Tracks if a gizmo manipulation is in progress
    bool is_bvh_dirty = false; // Flag for lazy BVH updates
    bool focus_scene_edit_tab = false; // Auto-focus Scene Edit tab after model load
    
    // Undo/Redo System
    SceneHistory history;  // Command history for undo/redo
    
    // Transform Undo/Redo State
    // We store the initial state when drag starts
    struct TransformState drag_start_state;
    std::string drag_object_name;

    // Marquee (Box) Selection State
    bool is_marquee_selecting = false;
    ImVec2 marquee_start;
    ImVec2 marquee_end;
    void handleMarqueeSelection(UIContext& ctx);  // Box selection implementation
    void drawMarqueeRect();  // Draw the selection rectangle
   
};
