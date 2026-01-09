#pragma once

// Forward Declarations
class Renderer;
struct SceneData;
class OptixWrapper;
class ColorProcessor;
class SceneSelection;
struct RenderSettings;
class Texture;
struct InstanceGroup;

#include "ui_modern.h"  // Modern UI sistemi
#include "SceneHistory.h"  // Undo/Redo system
#include "SceneCommand.h"  // TransformState struct
#include "TimelineWidget.h"  // Timeline animation widget
#include "CameraPresets.h"   // Camera body, lens, ISO, shutter presets
#include "TimelineWidget.h"  // Timeline animation widget
#include "CameraPresets.h"   // Camera body, lens, ISO, shutter presets
#include "TerrainNodesV2.h" // Terrain node graph V2 system
#include "scene_ui_nodeeditor.hpp" // Terrain node editor UI
#include <fstream>
#include <map>
#include <set> // For lazy CPU sync
#include <atomic> // For thread-safe flags
#include "Vec3.h" // For bbox_cache
// ═══════════════════════════════════════════════════════════════════════════════
// SCENE UI - HEADER
// ═══════════════════════════════════════════════════════════════════════════════
// This header defines the main SceneUI class and the UIContext structure.
//
// MODULE LOCATIONS (Implementations):
//   - scene_ui_camera.cpp    : Camera settings (drawCameraContent)
//   - scene_ui_materials.cpp : Material editor (drawMaterialPanel)
//   - scene_ui_hierarchy.cpp : Scene tree (drawSceneHierarchy)
//   - scene_ui_lights.cpp    : Lights panel (drawLightsContent)
//   - scene_ui_gizmos.cpp    : 3D gizmos & bounding boxes
//   - scene_ui_viewport.cpp  : Overlays (Focus/Zoom/Exposure/Dolly)
//   - scene_ui_selection.cpp : Selection logic & Marquee
//   - scene_ui_world.cpp     : World environment settings
// ═══════════════════════════════════════════════════════════════════════════════


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

    // Animation Preview Buffer (Thread-Safe)
    // Moved to end to preserve initializer list compatibility
    std::vector<uint32_t> animation_preview_buffer;
    std::atomic<bool> animation_preview_ready{false};
    std::mutex animation_preview_mutex;
    int animation_preview_width = 0;
    int animation_preview_height = 0;
    SDL_Texture* animation_preview_texture = nullptr;
    bool show_animation_preview = false;
    bool is_animation_mode = false; // NEW: For Unified Render Window
    struct SDL_Texture* ray_texture; // Forward decl or void* if header dependency issues, but SDL.h is included in Renderer.h
    
    // Note: SDL_Texture* requires SDL.h. Renderer.h includes it.
    // If SceneUI.h doesn't include SDL, we need forward declaration or include.
    // Renderer.h is included at top (line 3). So SDL.h is available.

	
};

class SceneUI {
public:   
    // UI State
    int pivot_mode = 0; // 0=Median Point (Group), 1=Individual Origins
    bool show_animation_panel = true; // Default open
    bool show_foliage_tab = false;    // Default closed (User preference)
    bool show_water_tab = true;       // DEFAULT OPEN
    bool show_terrain_tab = true;     // DEFAULT OPEN
    bool show_system_tab = false;     // Default closed
    bool show_terrain_graph = false;  // Terrain node editor panel
    bool show_anim_graph = false;     // Animation node editor panel
    std::string tab_to_focus = "";    // For auto-focusing tabs upon activation

    // Static Helpers (Shared across modules)
    static std::string openFileDialogW(const wchar_t* filter = L"All Files\0*.*\0", const std::string& initialDir = "", const std::string& defaultFilename = "");
    static std::string saveFileDialogW(const wchar_t* filter = L"All Files\0*.*\0", const wchar_t* defExt = L"rts");
    static void syncInstancesToScene(UIContext& ctx, InstanceGroup& group, bool clear_only);
    
    // ═══════════════════════════════════════════════════════════
    // GLOBAL UI WIDGETS (Themable)
    // ═══════════════════════════════════════════════════════════
    struct LCDTheme {
        ImU32 litColor = IM_COL32(200, 200, 200, 255);    // Active segment color
        ImU32 offColor = IM_COL32(40, 45, 50, 255);       // Inactive segment color
        ImU32 bgColor = IM_COL32(20, 20, 20, 255);        // Background/border color
        ImU32 textValColor = IM_COL32(180, 230, 255, 255);// Value text color
        bool isRetroGreen = false;                        // Preset flag
    };
    static LCDTheme currentTheme;
    
    enum class UISliderStyle { Modern, RetroLCD };
    static UISliderStyle globalSliderStyle; // Definition

    // Global LCD Slider Widget (Direct Call)
    static bool DrawLCDSlider(const char* id, const char* label, float* value, float min, float max, 
                              const char* format, bool keyed = false, 
                              std::function<void()> onKeyframeClick = nullptr, int segments = 16);

    // SMART SLIDER WRAPPER (Switchable Style)
    // Uses globalSliderStyle to determine whether to draw LCD or Standard slider
    static bool DrawSmartFloat(const char* id, const char* label, float* value, float min, float max, 
                               const char* format = "%.2f", bool keyed = false, 
                               std::function<void()> onKeyframeClick = nullptr, int segments = 16);


    void drawHistogramPanel(UIContext& ctx);      
    void drawLogPanelEmbedded();
    void drawThemeSelector();
     void drawResolutionPanel(UIContext& ctx);

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
     void drawMaterialPanel(UIContext& ctx);   // Material/Texture editor for selected object
     void drawSelectionBoundingBox(UIContext& ctx);  // Draw bounding box for selected object
     void drawTransformGizmo(UIContext& ctx);  // ImGuizmo transform gizmo   
     void drawCameraGizmos(UIContext& ctx);    // Draw camera icons in viewport
     void drawViewportControls(UIContext& ctx); // Blender-style viewport overlay (top-right)
     void drawRenderWindow(UIContext& ctx); // Converted from global to member function
     void draw(UIContext& ctx);
     void handleMouseSelection(UIContext& ctx); // Publicly accessible for Main loop call
     void triggerDelete(UIContext& ctx); // Trigger delete operation (for Menu and Key)
     void processAnimations(UIContext& ctx); // Apply keyframe data to scene objects
  
     void invalidateCache() { mesh_cache_valid = false; }
     void rebuildMeshCache(const std::vector<std::shared_ptr<class Hittable>>& objects);
     void updateBBoxCache(const std::string& objectName);  // Update bounding box for specific object after transform
     
     // Viewport Messages (HUD)
     struct ViewportMessage {
         std::string text;
         float time_remaining;
         ImVec4 color;
     };
     std::vector<ViewportMessage> active_messages;
     void addViewportMessage(const std::string& text, float duration = 2.0f, ImVec4 color = ImVec4(1,1,1,1));
     void clearViewportMessages(); // Force clear all messages
     void drawViewportMessages(UIContext& ctx, float left_offset);
     
     // Background Save State (0=Idle, 1=Saving, 2=Done, 3=Error)
     // Background Save State (0=Idle, 1=Saving, 2=Done, 3=Error)
     std::atomic<int> bg_save_state{0};

     // Exit Confirmation
     bool show_exit_confirmation = false;
     enum class PendingAction {
        None,
        Exit,
        NewProject,
        OpenProject
    };

    PendingAction pending_action = PendingAction::None;

    void tryNew(UIContext& ctx);
    void tryOpen(UIContext& ctx);
    void performNewProject(UIContext& ctx);
    void performOpenProject(UIContext& ctx);
    
    // Existing functions...
    void tryExit();
     void drawExitConfirmation(UIContext& ctx);

     // Interaction Flags
     bool is_picking_focus = false; // Flag for "Pick Focus" mode (hit distance only)
     bool hud_captured_mouse = false; // Flag set when HUD elements capture mouse clicks
     
     // Sun Sync Logic (Global)
     bool sync_sun_with_light = true;
     bool world_params_changed_this_frame = false;
     void processSunSync(UIContext& ctx);
     
     // Project System Helpers
     void updateProjectFromScene(UIContext& ctx);  // Sync scene state to project data
     void addProceduralPlane(UIContext& ctx);      // Add a procedural plane mesh
     void addProceduralCube(UIContext& ctx);       // Add a procedural cube mesh
     float panel_alpha = 0.9f; // Default transparency (0.9 = mostly opaque)
     
     // Scene loading state (public for Main.cpp popup access)
     std::atomic<bool> scene_loading{false};
     std::atomic<bool> scene_loading_done{false};
     std::atomic<int> scene_loading_progress{0};  // 0-100
     std::string scene_loading_stage = "";        // Current stage description
     
     // Viewport Display Settings (Blender-style overlay)
     struct ViewportDisplaySettings {
         int shading_mode = 1;  // 0=Solid, 1=Material, 2=Rendered
         bool show_gizmos = true;
         
         // Camera HUD (viewport lens controls)
         bool show_camera_hud = true;      // Master toggle for camera HUD
         bool show_focus_ring = true;      // Focus ring (DOF control)
         bool show_zoom_ring = true;       // Zoom ring (FOV control)
         
         // === PRO CAMERA FEATURES ===
         // Histogram
         bool show_histogram = false;      // RGB/Luma histogram overlay
         int histogram_mode = 0;           // 0=RGB, 1=Luma, 2=Parade
         float histogram_opacity = 0.7f;   // Histogram transparency
         
         // Focus Peaking
         bool show_focus_peaking = false;  // Edge detection overlay for sharp areas
         int focus_peaking_color = 0;      // 0=Red, 1=Yellow, 2=Green, 3=Blue, 4=White
         float focus_peaking_threshold = 0.15f; // Edge detection sensitivity
         
         // Zebra Stripes
         bool show_zebra = false;          // Overexposure warning stripes
         float zebra_threshold = 0.95f;    // Brightness threshold (0-1)
         
         // Multi-Point AF
         bool show_af_points = false;      // AF point grid overlay
         int af_mode = 0;                  // 0=Single, 1=Zone9, 2=Zone21, 3=Wide, 4=CenterWeighted
         int af_selected_point = 4;        // Selected point index (center=4 for 3x3)
         int focus_mode = 1;               // 0=MF, 1=AF-S, 2=AF-C (Default: AF-S)
     };
     ViewportDisplaySettings viewport_settings;
     
     // Viewport Guide Settings (Safe areas, letterbox, grids)
     struct GuideSettings {
         bool show_safe_areas = false;
         int safe_area_type = 0;
         float title_safe_percent = 0.80f;
         float action_safe_percent = 0.90f;
         bool show_letterbox = false;
         int aspect_ratio_index = 0;
         float letterbox_opacity = 0.7f;
         bool show_grid = false;
         int grid_type = 0;
         bool show_center = false;
     };
     GuideSettings guide_settings;
     
     // Scatter Brush Settings (Foliage/Instance painting)
     struct ScatterBrushSettings {
         bool enabled = false;           // Scatter brush mode active
         int active_group_id = -1;       // Currently selected instance group
         float brush_radius = 2.0f;      // Brush size in world units
         float brush_strength = 1.0f;    // Density multiplier
         int brush_mode = 0;             // 0=Add, 1=Remove, 2=Adjust
         bool show_brush_preview = true; // Show brush circle in viewport
         std::string target_surface_name; // Which mesh to paint on (empty = any)
     };
     ScatterBrushSettings scatter_brush;
     
     // Scatter Brush UI
     void drawScatterBrushPanel(UIContext& ctx);
     void handleScatterBrush(UIContext& ctx);  // Viewport brush interaction
     void drawBrushPreview(UIContext& ctx);    // Draw brush circle in viewport
    
    // Terrain Brush Settings
    struct TerrainBrushSettings {
        bool enabled = false;
        int active_terrain_id = -1;
        float radius = 5.0f;
        float strength = 0.5f;
        int mode = 0; // 0=Raise, 1=Lower, 2=Flatten, 3=Smooth, 4=Stamp
        bool show_preview = true;
        
        // Flatten Params
        float flatten_target = 0.0f;
        bool use_fixed_height = false;
        
        // Stamp Params
        std::shared_ptr<class Texture> stamp_texture;
        float stamp_rotation = 0.0f; // Degrees
        
        // Paint Params
        int paint_channel = 0; // 0=R(Layer0), 1=G(Layer1), 2=B(Layer2), 3=A(Layer3)
    };
    TerrainBrushSettings terrain_brush;

    // Terrain Foliage Brush Settings (Paint to add/remove foliage)
    struct FoliageBrushSettings {
        bool enabled = false;           // Foliage brush mode active
        std::string active_layer_name;  // Which foliage layer to paint (by name)
        float radius = 5.0f;            // Brush size in world units
        int density = 3;                // Instances per stroke
        int mode = 0;                   // 0=Add, 1=Remove
        bool show_preview = true;       // Show brush circle in viewport
    };
    FoliageBrushSettings foliage_brush;

    // Water, River & Terrain UI
    void drawWaterPanel(UIContext& ctx);
    void drawRiverPanel(UIContext& ctx);       // Bezier spline river editor
    void drawRiverGizmos(UIContext& ctx, bool& gizmo_hit);  // River spline visualization
    void drawTerrainPanel(UIContext& ctx);
    void handleTerrainBrush(UIContext& ctx);
    void handleTerrainFoliageBrush(UIContext& ctx);  // Foliage paint brush
    void tickProgressiveVertexSync(); // Called each frame to process a chunk
    void updateAutofocus(UIContext& ctx);  // Run autofocus logic (raycast center)rlay (focal, FOV, aperture)
private:
    // --- UI Structure ---
    void drawPanels(UIContext& ctx);
    void drawStatusAndBottom(UIContext& ctx, float screen_x, float screen_y, float left_offset);
    void drawAuxWindows(UIContext& ctx);

    // --- Input / Editor ---
    void handleEditorShortcuts(UIContext& ctx);
    bool deleteSelectedLight(UIContext& ctx);
    bool deleteSelectedObject(UIContext& ctx);
    void handleDeleteShortcut(UIContext& ctx);

    // --- Overlays & Gizmos ---
    bool drawOverlays(UIContext& ctx);
    void drawLightGizmos(UIContext& ctx, bool& gizmo_hit);
    void drawSelectionGizmos(UIContext& ctx);
    void drawFocusIndicator(UIContext& ctx);  // Split-prism focus aid
    void drawZoomRing(UIContext& ctx);        // FOV zoom control
    void drawExposureInfo(UIContext& ctx);    // Cinema camera exposure bar
    void drawDollyArc(UIContext& ctx);        // Camera dolly track control (disabled)
    
    // === PRO CAMERA HUD ===
    void drawHistogramOverlay(UIContext& ctx);     // RGB/Luma histogram
    void drawFocusPeakingOverlay(UIContext& ctx);  // Sharp edge highlighting
    void drawZebraOverlay(UIContext& ctx);         // Overexposure warning
    void drawAFPointsOverlay(UIContext& ctx);      // Multi-point autofocus grid
    void drawProCameraPanel(UIContext& ctx);       // Settings panel for pro features
     void drawLensInfoHUD(UIContext& ctx);  // Lens info top-left
    

    // --- Scene Interaction ---
    void handleSceneInteraction(UIContext& ctx, bool gizmo_hit);

    // --- Deferred / Maintenance ---
    void processDeferredSceneUpdates(UIContext& ctx);
    bool showSidePanel = true;
    bool show_controls_window = false; // Controls/Help window visibility
    bool showResolutionPanel = true; // class üyesi
    bool camera_initialized = false;
    Vec3 camera_initial_pos;
    Vec3 camera_initial_target;
    float camera_initial_fov;
   
    bool show_scene_log = false; // Default closed
    float side_panel_width = 360.0f; // Resizable Left Panel width
    float bottom_panel_height = 100.0f; // Default to minimum height
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
    
    // Bounding Box Cache - avoids recalculating bounds every frame (HUGE perf win for large objects)
    // Key: nodeName, Value: {bb_min, bb_max}
    std::map<std::string, std::pair<Vec3, Vec3>> bbox_cache;
    
    // Material Slots Cache - avoids scanning all triangles every frame to get unique material IDs
    // Key: nodeName, Value: list of unique material IDs used by that object
    std::map<std::string, std::vector<uint16_t>> material_slots_cache;

    bool mesh_cache_valid = false;
    
    // Lazy CPU Vertex Sync - objects that need CPU update before picking
    // In TLAS mode, we skip CPU vertex update on gizmo release for instant response.
    // Instead, we mark objects as "needing sync" and only update when picking is attempted.
    std::set<std::string> objects_needing_cpu_sync;
    void ensureCPUSyncForPicking(); // Called before mouse picking to sync pending objects
   
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
    
   
    
    // Terrain Node Graph (V2 System)
    TerrainNodesV2::TerrainNodeGraphV2 terrainNodeGraph;
    TerrainNodesV2::TerrainNodeEditorUI terrainNodeEditorUI;
    
public:
    // Timeline Widget
    class TimelineWidget timeline;  // Timeline animation widget
    // Terrain Node Graph accessors for serialization
    TerrainNodesV2::TerrainNodeGraphV2& getTerrainNodeGraph() { return terrainNodeGraph; }
    const TerrainNodesV2::TerrainNodeGraphV2& getTerrainNodeGraph() const { return terrainNodeGraph; }
};
