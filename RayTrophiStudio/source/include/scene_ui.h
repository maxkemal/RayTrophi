/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          scene_ui.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once

#include "globals.h"

// Forward Declarations
class Renderer;
struct SceneData;
class OptixWrapper;
namespace Backend { class IBackend; }
class ColorProcessor;
class SceneSelection;
struct RenderSettings;
class Texture;
struct HitRecord;
struct InstanceGroup;
struct WaterSurface;
class VDBVolume;
class GasVolume;
class Hittable;

#include "ui_modern.h"  // Modern UI sistemi
#include "SceneHistory.h"  // Undo/Redo system
#include "SceneCommand.h"  // TransformState struct
#include "TimelineWidget.h"  // Timeline animation widget
#include "CameraPresets.h"   // Camera body, lens, ISO, shutter presets
#include "Hair/HairUI.h"     // Hair/Fur editing panel

#include "TerrainNodesV2.h" // Terrain node graph V2 system
#include "scene_ui_nodeeditor.hpp" // Terrain node editor UI
#include <fstream>
#include <map>
#include <set> // For lazy CPU sync
#include <atomic> // For thread-safe flags
#include <mutex>
#include <future>
#include <string>
#include <memory>
#include <vector>
#include <array>
#include <unordered_map>
#include <tuple>
#include "Vec3.h" // For bbox_cache
#include "MeshEdit/HalfEdgeMesh.h" // Edit mode half-edge core
#include "instancegroup.h"
#include "Backend/IBackend.h"
#include "Backend/OptixBackend.h"
#include "AssetRegistry.h"
#include "AABB.h"
#include "Paint/PaintModeState.h"
#include "Paint/MeshPaintAdapter.h"
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
    Backend::IBackend* backend_ptr;
    ColorProcessor& color_processor;
    RenderSettings& render_settings;
    SceneSelection& selection;  // Scene selection manager
   

    int& sample_count;                // dynamic counter (not in settings)
    bool& start_render;              // trigger flag
    std::string& active_model_path;

    bool& apply_tonemap;
    bool& reset_tonemap;

    bool& mouse_control_enabled;
    
    // Animation Preview Buffer (Thread-Safe)
    // Moved to end to preserve initializer list compatibility
    std::vector<uint32_t> animation_preview_buffer;
    std::atomic<bool> animation_preview_ready{false};
    std::mutex animation_preview_mutex;
    int animation_preview_width = 0;
    int animation_preview_height = 0;
    SDL_Texture* animation_preview_texture = nullptr;
    bool show_animation_preview = false;
    // Particle viewport display mode: 0=Solid (Vulkan billboards), 1=Debug (ImGui
    // overlay dots, drawn on top), 2=Render (path-traced; previews as Solid for now).
    int particle_display_mode = 0;
    bool is_animation_mode = false; // NEW: For Unified Render Window
    struct SDL_Texture* ray_texture; // Forward decl or void* if header dependency issues, but SDL.h is included in Renderer.h
    
    // Note: SDL_Texture* requires SDL.h. Renderer.h includes it.
    // If SceneUI.h doesn't include SDL, we need forward declaration or include.
    // Renderer.h is included at top (line 3). So SDL.h is available.

	
};

namespace MeshModifiers { struct ModifierStack; }
bool hasEnabledSubdivisionPreview(const MeshModifiers::ModifierStack& stack);
std::size_t computeSubdivisionPreviewSignature(UIContext& ctx, const std::string& objectName);

class SceneUI {
public:   
    ~SceneUI();
    // UI State
    int pivot_mode = 0; // 0=Median Point (Group), 1=Individual Origins
    bool pivot_edit_mode = false; // When true, gizmo moves object pivot without moving geometry
    int active_properties_tab = 0; // NEW: Vertical side tab index
    bool show_animation_panel = true; // Default open
    bool show_foliage_tab = false;    // Default closed (User preference)
    bool show_water_tab = true;       // DEFAULT OPEN
    bool show_scatter_tab = true;     // Scatter panel (legacy) - visible in main panel
    bool show_terrain_tab = true;     // DEFAULT OPEN
    bool show_system_tab = true;     // Default closed
    bool show_terrain_graph = false;  // Terrain node editor panel
    bool show_anim_graph = false;     // Animation node editor panel
    bool show_volumetric_tab = true;  // Unified Volumetrics tab (VDB + Gas)
    bool show_forcefield_tab = true;   // Simulation tab (Default open)
    bool show_world_tab = true;        // World & Sky tab (Default open)
    bool show_stylize_tab = true;      // Stylize art-direction layer
    bool show_hair_tab = true;         // Hair & Fur tab (Default open)
    bool show_modifiers_tab = true;    // Modifiers & Sculpt tab (Default open)
    bool show_paint_tab = true;        // Paint Mode tab (Default open)
    bool focus_properties_panel_next_frame = false;
    int active_water_subtab = 0;       // 0=Water, 1=River
    int selected_water_surface_id = -1;
    int vdb_import_orientation_preset = 0; // 0=Auto, 1=Standard, 2=Y-up to Z-forward (-90 X)
    std::string tab_to_focus = "";    // For auto-focusing tabs upon activation

    // Hair/Fur System
    Hair::HairUI hairUI;               // Hair editing panel
    // Persisted open/closed state for terrain subsections
    bool terrain_layer_open[4] = { true, true, true, true };
    bool foliage_section_open = true;

    // Static Helpers (Shared across modules)
    static std::string openFileDialogW(const wchar_t* filter = L"All Files\0*.*\0", const std::string& initialDir = "", const std::string& defaultFilename = "");
    static std::string saveFileDialogW(const wchar_t* filter = L"All Files\0*.*\0", const wchar_t* defExt = L"rts");
    static std::string selectFolderDialogW(const wchar_t* title = L"Select Folder");
    
    // Shared Volumetric UI
    static bool drawVolumeShaderUI(UIContext& ctx, std::shared_ptr<class VolumeShader> shader, class VDBVolume* vdb = nullptr, class GasVolume* gas = nullptr);
    struct VDBShaderDefaults {
        float density_multiplier = 2.0f;
        float scattering_coefficient = 1.5f;
        float absorption_coefficient = 0.05f;
        float step_size = 0.15f;
        int max_steps = 64;
        int shadow_steps = 8;
    };
    static VDBShaderDefaults estimateVDBShaderDefaults(const class VDBVolume& vdb);
    static void applyEstimatedVDBShaderDefaults(class VDBVolume& vdb);
    static bool shouldApplySpecialVDBOrientation(const std::string& source_hint);
    static void applyVDBImportOrientation(class VDBVolume& vdb, int orientation_preset, const std::string& source_hint = "");

    static void syncInstancesToScene(UIContext& ctx, InstanceGroup& group, bool clear_only);
    static void syncVDBVolumesToGPU(UIContext& ctx);

    void appendInstancesToScene(UIContext& ctx, InstanceGroup& group, size_t start_index);
    
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
     void drawRenderInspectorContent(UIContext& ctx);
     static void ClampWindowToDisplay();    
     void drawAnimationSettings(UIContext& ctx);  // Deprecated - timeline panel kullanılıyor
     void drawTimelinePanel(UIContext& ctx, float screen_y);  // Yeni timeline panel (Blender tarzı)
     void drawTimelineContent(UIContext& ctx); // Embedded timeline logic
     void drawMainMenuBar(UIContext& ctx); // New Main Menu Bar
     void drawControlsContent(); // New method for controls/help tab
     void drawWorldContent(UIContext& ctx);
     void drawStylizePanel(UIContext& ctx);
     void drawSceneHierarchy(UIContext& ctx);  // Scene hierarchy / outliner panel
     void drawModifiersPanel(UIContext& ctx);  // Modifiers & Sculpting panel
     void drawSculptPanel(UIContext& ctx);      // Dedicated sculpt panel
     void activateEditWorkspace(UIContext& ctx);
     void activateSculptWorkspace(UIContext& ctx);
     void drawPaintPanel(UIContext& ctx);      // Lightweight paint workflow panel
     void drawTerrainPaintPanel(UIContext& ctx, class TerrainObject* terrain);
     void drawMeshPaintPanel(UIContext& ctx, const std::shared_ptr<class Triangle>& meshTriangle);
     void drawPaintBrushDock(UIContext& ctx);
     void drawPaintBrushControls(UIContext& ctx, const std::shared_ptr<class Triangle>& meshTriangle, bool rightDockOnly = false);
     void drawSculptBrushControls(UIContext& ctx, const std::shared_ptr<class Triangle>& meshTriangle, bool rightDockOnly = false);
     void drawEditToolControls(UIContext& ctx, const std::shared_ptr<class Triangle>& meshTriangle, bool rightDockOnly = false);
     void drawPaintLayerPanel(UIContext& ctx, Paint::MeshPaintAdapter* adapter);
     void drawPaintChannelTextureSlots(UIContext& ctx, Paint::MeshPaintAdapter* adapter);
     void drawMaterialPanel(UIContext& ctx);   // Material/Texture editor for selected object
     // Flat/proxy: a dense mesh stored as a single TriangleMesh (no facades) cannot be repainted
     // by the per-facade setMaterialID loop (only its one representative face would change). If
     // `name` is such a direct mesh, bulk-rewrite its SoA materialID buffer (oldID->newID; pass
     // oldID==INVALID to repaint all) AND remap the CPU Embree BVH in place (no full rebuild), so
     // CPU render reflects the change immediately. Returns true if handled as a direct mesh.
     bool repaintDirectMeshMaterial(UIContext& ctx, const std::string& name, uint16_t oldID, uint16_t newID);
     void drawPrincipledBSDFEditor(class PrincipledBSDF* pbsdf, uint16_t mat_id, UIContext& ctx); // Reusable editor widget
     void drawEditableMeshOverlay(UIContext& ctx); // Viewport edit overlay for selected mesh
     // GPU edit overlay: pushes editable-mesh wireframe/vertex/face data to the
     // raster viewport backend. Returns true when the GPU path is active this
     // frame (caller then skips the ImGui fallback drawing).
     bool syncGpuEditMeshOverlay(UIContext& ctx, const std::string& objectName,
                                 bool drawVertices, bool drawEdges, bool drawFaces);
     void releaseGpuEditMeshOverlay(); // clear backend buffers + disable the GPU overlay
     bool ensureEditableMeshCache(UIContext& ctx, const std::string& objectName);
     // Scatter sculpt/edit vertex edits from the editable cache back into a flat (direct) SoA
     // mesh (editable vertex id == SoA vertex id). Updates P_orig/P and recomputes N_orig/N for
     // the moved verts + their one-ring. No-op (returns false) when the cache is facade-backed.
     bool syncFlatSculptVerticesToSoA(const std::vector<size_t>& movedVertexIds);
     bool ensureSculptControlGraph(UIContext& ctx, const std::string& objectName);
     void invalidateSculptControlGraph(const std::string& objectName = std::string());
     bool ensureSculptPBVH(UIContext& ctx, const std::string& objectName);
     void invalidateSculptPBVH(const std::string& objectName = std::string());
     bool refineSculptHitWithPBVH(const Ray& ray, const std::string& objectName,
                                  HitRecord& hit, bool didHit);
     bool handleMeshElementSelection(UIContext& ctx, const ImVec2& mousePos);
     Vec3 getSelectedMeshElementWorldPosition(UIContext& ctx, bool* valid = nullptr);
     bool applySelectedMeshElementTranslation(UIContext& ctx, const Vec3& worldDelta);
     bool applySelectedMeshElementTransform(UIContext& ctx, const Matrix4x4& worldTransform);
     struct MeshShadingSettings;
     MeshShadingSettings& ensureMeshShadingSettings(const std::string& objectName);
     bool applyMeshShadingSettings(UIContext& ctx, const std::string& objectName, bool queueGpuSync = true);
     // Sculpt entry: repair only ZERO/non-finite shading normals (the dense-mesh black-shading
     // bug) WITHOUT rewriting valid ones — preserves the object's current flat/smooth look.
     bool repairSculptEntryShadingNormals(UIContext& ctx, const std::string& objectName, bool queueGpuSync = true);
     bool refreshEditableDisplayMeshFromBase(UIContext& ctx, const std::string& objectName, bool queueGpuSync = true, bool rebuildEditableCache = true);
     void beginInteractiveSubdivisionPreview(const std::string& objectName);
     void endInteractiveSubdivisionPreview(UIContext& ctx, const std::string& objectName, bool rebuildFull = true);
     bool isInteractiveSubdivisionPreviewActiveForObject(const std::string& objectName) const;
     void clearEditableMeshSelection();
     void resetMeshEditState(UIContext& ctx);
     void syncMeshEditState(UIContext& ctx);
     // Publishes the Edit-Mesh vertex selection (world space) to ForceFieldUI so the
     // Bodies panel can pin selected vertices on a cloth/soft body. Called per frame.
     void publishEditPinSelection(UIContext& ctx);
     bool ensureMeshEditLayer(UIContext& ctx, const std::string& objectName);
     void refreshMeshEditLayerEditedState(UIContext& ctx);
     void captureMeshEditLayerState(UIContext& ctx, const std::string& objectName, std::vector<MeshEditTriangleState>& outStates);
            // overlay grid removed; raster grid in backends handles depth-tested grid
     void applyMeshEditTriangleStates(UIContext& ctx, const std::vector<MeshEditTriangleState>& states, bool queueGpuSync = true);
     void setMeshEditLayerEnabled(UIContext& ctx, bool enabled);
     void applyMeshEditLayer(UIContext& ctx);
     void discardMeshEditLayer(UIContext& ctx);
     void tryRestoreSerializedMeshEditLayer(UIContext& ctx);
     void queueMeshEditGpuSync(const std::string& objectName);
     void processPendingMeshEditGpuSync(UIContext& ctx);
     bool mergeSelectedVerticesToCenter(UIContext& ctx);
     bool weldSelectedVerticesByDistance(UIContext& ctx, float distance);
     bool addFaceFromSelectedVertices(UIContext& ctx);
     bool deleteSelectedMeshFaces(UIContext& ctx);
     bool extrudeSelectedMeshFaces(UIContext& ctx, float distance);
     bool insetSelectedMeshFaces(UIContext& ctx, float amount);
     bool loopCutSelectedEdges(UIContext& ctx, float t);
     bool dissolveSelectedEdges(UIContext& ctx);
     bool dissolveSelectedVertices(UIContext& ctx);
     // Mesh-edit ops + selection helpers (restored: these working-tree declarations were
     // lost when a concurrent edit clobbered scene_ui.h; definitions live in the .cpp).
     void applyCreaseToSelectedEdges(UIContext& ctx, float weight);
     float getSelectedEdgesAverageCrease(UIContext& ctx) const;
     bool applyShadingToSelectedFaces(UIContext& ctx, bool flat, bool autoSmooth);
     bool flipSelectedMeshNormals(UIContext& ctx);
     bool recalculateMeshNormals(UIContext& ctx, bool outside);
     bool ensureEditableHalfEdge();
     void selectAllObjects(UIContext& ctx);
     void invertObjectSelection(UIContext& ctx);
     void selectAllMeshElements(UIContext& ctx);
     void invertMeshSelection(UIContext& ctx);
     void performObjectMarqueeSelection(UIContext& ctx, float x1, float y1, float x2, float y2);
     void performObjectLassoSelection(UIContext& ctx, const std::vector<ImVec2>& points);
     void performMeshElementMarqueeSelection(UIContext& ctx, float x1, float y1, float x2, float y2);
     void performMeshElementLassoSelection(UIContext& ctx, const std::vector<ImVec2>& points);
     void drawLassoOutline();
     size_t telemetry_candidate_vertices = 0; // HUD: verts gathered by the last sculpt dab
     size_t telemetry_pcie_upload_bytes = 0;  // HUD: bytes uploaded to GPU last mesh sync
     void drawSelectionBoundingBox(UIContext& ctx);  // Draw bounding box for selected object
     void drawTransformGizmo(UIContext& ctx);  // ImGuizmo transform gizmo   
     void drawCameraGizmos(UIContext& ctx);    // Draw camera icons in viewport
     void drawViewportControls(UIContext& ctx); // Blender-style viewport overlay (top-right)
     void drawRenderWindow(UIContext& ctx); // Converted from global to member function
     void draw(UIContext& ctx);
     void handleMouseSelection(UIContext& ctx); // Publicly accessible for Main loop call
     void triggerDelete(UIContext& ctx); // Trigger delete operation (for Menu and Key)
     void triggerDuplicate(UIContext& ctx); // Trigger duplicate operation (for Menu and Key)
     void processAnimations(UIContext& ctx); // Apply keyframe data to scene objects
  
     void invalidateCache();
     void rebuildTriToIndex(const std::vector<std::shared_ptr<class Hittable>>& objects);
     void rebuildMeshCache(const std::vector<std::shared_ptr<class Hittable>>& objects);
     void syncAllTransformedVertices(struct SceneData& scene);
     void updateBBoxCache(const std::string& objectName);  // Update bounding box for specific object after transform
     void moveObjectPivot(UIContext& ctx, const std::string& objectName, const Vec3& worldDelta);
     void recenterObjectPivotToBoundsCenter(UIContext& ctx, const std::string& objectName);
     
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
    void resetMaterialUI(); // Reset material editor state
    void addProceduralPlane(UIContext& ctx);      // Add a procedural plane mesh
    void addProceduralCube(UIContext& ctx);       // Add a procedural cube mesh
    void addProceduralSphere(UIContext& ctx);     // Add a procedural UV sphere mesh
    void addProceduralCylinder(UIContext& ctx);   // Add a procedural cylinder mesh
    void addProceduralRock(UIContext& ctx);       // Add a procedural rock mesh (quad-based)
    void addProceduralBrickWall(UIContext& ctx);  // Add a procedural brick wall mesh (quad-based)
    void addProceduralTorus(UIContext& ctx);      // Add a procedural torus mesh (quad-based)
    void addProceduralStaircase(UIContext& ctx);  // Add a procedural staircase mesh (quad-based)
    void drawProceduralGeneratorWindow(UIContext& ctx);

    // ── Destruction: convex Voronoi pre-fracture (Faz 1, geometry only) ───────
    // Split the given mesh node into convex shard meshes. The original mesh is
    // parked (kept alive, removed from the scene) so re-fracture reuses it and
    // unfractureMesh restores it exactly. No physics yet (Faz 2 makes shards
    // rigid bodies). `pattern`: 0 = uniform, 1 = impact-clustered.
    void fractureSelectedMesh(UIContext& ctx, const std::string& node,
                              int site_count, uint32_t seed, int pattern);
    void unfractureMesh(UIContext& ctx, const std::string& node);
    void fractureRefreshScene_(UIContext& ctx);  // rebuild caches/BVH/backends after shard edits
    bool isMeshFractured(const std::string& node) const {
        return fracture_shard_nodes_.find(node) != fracture_shard_nodes_.end();
    }
    // Fracture authoring state (panel controls).
    int   fracture_site_count = 15;
    int   fracture_seed = 1337;
    int   fracture_pattern = 0;     // 0 = Uniform, 1 = Impact-clustered
    float fracture_preview_gap = 0.02f;  // shrink shards toward their centroid so the
                                         // cuts are visible without physics (Faz 1)
    float fracture_break_threshold = 5.0f;  // impact impulse (kg·m/s) to shatter (Faz 2)
    // Parked source meshes (alive, out of the scene) + emitted shard node names,
    // keyed by the fractured source node. UI-side for Faz 1 (serialize = Faz 5).
    std::map<std::string, std::vector<std::shared_ptr<Hittable>>> fracture_parked_originals_;
    std::map<std::string, std::vector<std::string>> fracture_shard_nodes_;
    
    // Procedural generator window state
    bool show_procedural_generator = false;
    int procedural_generator_type = 0; // 0 = Rock, 1 = Brick Wall, 2 = Torus, 3 = Staircase
    
    // Rock parameters
    char rock_name[64] = "Rock";
    int rock_resolution = 16;
    float rock_radius = 1.0f;
    float rock_noise_scale = 1.5f;
    float rock_noise_strength = 0.3f;
    int rock_noise_octaves = 4;
    float rock_scale_x = 1.0f;
    float rock_scale_y = 1.0f;
    float rock_scale_z = 1.0f;
    float rock_flatness = 0.0f; // 0.0 = none
    int rock_seed = 1337;
    int rock_material_selection = 0;

    // Brick wall parameters
    char brick_name[64] = "Brick_Wall";
    int brick_rows = 5;
    int brick_cols = 8;
    float brick_width = 1.0f;
    float brick_height = 0.3f;
    float brick_depth = 0.5f;
    float brick_mortar_gap = 0.05f;
    float brick_tilt_variation = 0.02f;
    float brick_pos_variation = 0.01f;
    int brick_seed = 42;
    int brick_material_selection = 0;

    // Torus parameters
    char torus_name[64] = "Torus";
    float torus_major_radius = 1.0f;
    float torus_minor_radius = 0.3f;
    int torus_radial_segments = 32;
    int torus_tubular_segments = 16;
    int torus_material_selection = 0;

    // Staircase parameters
    char stairs_name[64] = "Staircase";
    int stairs_steps = 10;
    float stairs_step_width = 2.0f;
    float stairs_step_depth = 0.3f;
    float stairs_step_height = 0.2f;
    bool stairs_solid = true;
    int stairs_material_selection = 0;

    float panel_alpha = 0.9f; // Default transparency (0.9 = mostly opaque)
    void setSceneLoadingStage(const std::string& stage);
    std::string getSceneLoadingStage() const;
    
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

     // Scene loading state (public for Main.cpp popup access)
     std::atomic<bool> scene_loading{false};
     std::atomic<bool> scene_loading_done{false};
     std::atomic<int> scene_loading_progress{0};  // 0-100
     std::string scene_loading_stage = "";        // Current stage description
     mutable std::mutex scene_loading_stage_mutex;
     
     // Viewport Display Settings (Blender-style overlay)
     struct ViewportDisplaySettings {
         int shading_mode = 0;  // 0=Solid, 1=Material, 2=Rendered, 3=Matcap
            int matcap_preset = 0; // 0..9 allowed: 0=Solid clay, 1=User texture, 2..9=procedural presets
         bool show_gizmos = true;
         // Per-object silhouette outline drawn around the selected mesh. The
         // CPU raster path is cheap, but boundary pixels emit thousands of
         // ImGui rects per frame and that serializes with the raytrace on the
         // GPU queue — costs ~half the framerate on dense selections.
         bool show_selection_outline = true;

         // Camera HUD (viewport lens controls)
         bool show_camera_hud = false;      // Master toggle for camera HUD
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

     struct MeshOverlaySettings {
         bool enabled = false;
         bool edit_mode = false;
         bool show_vertices = false;
         bool xray_mode = false;          // GPU overlay draws through the mesh (Blender-style x-ray)
         bool proportional_edit = false;
         float proportional_radius = 0.75f;
         float proportional_falloff = 0.65f;
         int proportional_falloff_type = 0; // 0=Smooth,1=Linear,2=Sharp,3=Sphere,4=Root,5=Custom
         // Custom falloff curve LUT (y in [0,1] sampled over t in [0,1]); used when
         // proportional_falloff_type == 5. Empty until the user edits the curve.
         std::vector<float> custom_falloff_lut;
         float edge_thickness = 1.15f;
         float vertex_radius = 2.75f;
         int max_overlay_triangles = 12000;
         int max_vertex_markers = 1200;
         int selection_tool = 0;        // 0 = Box/marquee select, 1 = Lasso select
         bool show_normals = false;     // draw per-face normal lines in the overlay
         float normals_length = 0.25f;  // length of the drawn normal lines (world units)
     };
     MeshOverlaySettings mesh_overlay_settings;
     struct MeshShadingSettings {
         bool flat_shading = false;
         bool auto_smooth = true;
         float auto_smooth_angle_degrees = 60.0f;
     };
     std::unordered_map<std::string, MeshShadingSettings> mesh_shading_settings_by_object;
     struct CachedMeshOverlayEdgeSource {
         std::shared_ptr<class Triangle> triangle;
         int a = 0;
         int b = 0;
     };
     struct CachedMeshOverlayVertexSource {
         std::shared_ptr<class Triangle> triangle;
         int index = 0;
     };
     struct MeshOverlayCache {
         std::string object_name;
         size_t source_triangle_count = 0;
         std::vector<CachedMeshOverlayEdgeSource> edges;
         std::vector<CachedMeshOverlayVertexSource> vertices;
     };
     MeshOverlayCache mesh_overlay_cache;
     // SoA migration (Phase 1): refs/faces store an INDEX into the cache's owning
     // source_triangles vector instead of a per-element shared_ptr<Triangle>. On a dense
     // mesh this turns 6M (refs) + 2M (faces) scattered shared_ptr copies — each an atomic
     // refcount bump during the build — into a single owning vector, and roughly halves the
     // cache memory. Use EditableMeshCache::refTri()/faceTri() (raw Triangle*) or
     // refTriShared()/faceTriShared() (shared_ptr, when ownership must be handed on).
     struct EditableVertexRef {
         int triangle_index = -1;
         int corner = 0;
     };
     // Lightweight non-owning view over a slice of a CSR data array (P-CSR migration).
     // Replaces the per-vertex std::vector that EditableVertex.refs / vertex_neighbors used
     // to hold — millions of tiny heap allocations on a dense mesh — with two flat arrays
     // (offsets + data). Supports range-for, [] and empty()/size() so call sites read the
     // same as the old vector.
     template <typename T>
     struct CacheSpan {
         const T* b = nullptr;
         const T* e = nullptr;
         const T* begin() const { return b; }
         const T* end() const { return e; }
         bool empty() const { return b == e; }
         std::size_t size() const { return static_cast<std::size_t>(e - b); }
         const T& operator[](std::size_t i) const { return b[i]; }
     };
     struct EditableVertex {
         Vec3 local_position;
         bool is_boundary = false;
         // View into EditableMeshCache::vertex_ref_data (P-CSR). Behaves like the old
         // std::vector<EditableVertexRef> for call sites (range-for / [] / empty()/size()).
         CacheSpan<EditableVertexRef> refs;
     };
     struct EditableEdge {
         int v0 = -1;
         int v1 = -1;
     };
     struct EditableFace {
         int triangle_index = -1;
         int v0 = -1;
         int v1 = -1;
         int v2 = -1;
     };
     struct EditablePolygonFace {
         std::vector<int> vertex_ids;
         std::vector<int> triangle_ids;
     };
     struct EditableMeshSelection {
         int active_vertex_id = -1;
         int active_edge_id = -1;
         int active_face_id = -1;
         std::vector<int> vertex_ids;
         std::vector<int> edge_ids;
         std::vector<int> face_ids;
     };
     struct EditableSpatialCellKey {
         int x = 0;
         int y = 0;
         int z = 0;

         bool operator==(const EditableSpatialCellKey& other) const {
             return x == other.x && y == other.y && z == other.z;
         }
     };
     struct EditableSpatialCellKeyHasher {
         std::size_t operator()(const EditableSpatialCellKey& key) const {
             std::size_t hx = std::hash<int>{}(key.x);
             std::size_t hy = std::hash<int>{}(key.y);
             std::size_t hz = std::hash<int>{}(key.z);
             return hx ^ (hy << 1) ^ (hz << 2);
         }
     };
     struct EditableMeshCache {
         std::string object_name;
         // Monotonic rebuild stamp (0 = empty cache). Bumped every time the
         // cache is rebuilt so consumers (GPU edit overlay) can detect
         // topology changes without fingerprinting counts.
         uint64_t revision = 0;
         size_t source_triangle_count = 0;
         Matrix4x4 source_object_transform = Matrix4x4::identity();
         bool shade_flat = false;
         bool auto_smooth = true;
         float auto_smooth_angle_degrees = 60.0f;
         // Owning list of the source triangles this cache was welded from. EditableVertexRef
         // and EditableFace index into it (Phase 1 SoA migration). Kept alive here so raw
         // Triangle* views stay valid for the cache's lifetime.
         std::vector<std::shared_ptr<class Triangle>> source_triangles;
         std::vector<EditableVertex> vertices;
         std::vector<Vec3> vertex_positions;      // Aligned flat position buffer (1:1 with vertices)
         std::vector<uint8_t> vertex_is_boundary; // Fast per-vertex boundary flags (1:1 with vertices)
         std::vector<EditableEdge> edges;
         std::vector<EditableEdge> polygon_edges;
         std::vector<EditableFace> faces;
         std::vector<EditablePolygonFace> polygon_faces;
         // P-CSR backing storage. The per-vertex incident-triangle refs and neighbour lists
         // live in these two flat arrays (built once, never resized after); EditableVertex.refs
         // and vertex_neighbors[v] are lightweight CacheSpans into them, replacing the old
         // millions of tiny per-vertex std::vectors. Spans stay valid for the cache's lifetime.
         std::vector<EditableVertexRef> vertex_ref_data;
         std::vector<int> vertex_neighbor_data;
         std::vector<CacheSpan<int>> vertex_neighbors; // 1:1 with vertices; views into vertex_neighbor_data
         // Per-face source-mesh index (parallel to source_triangles / faces). Replaces the old
         // triangle_to_mesh_index hash map: a Triangle* resolves to its face index via
         // Triangle::editable_index, then face_to_mesh_index[faceIdx] gives the GPU/source
         // buffer slot. {v0,v1,v2} likewise comes from faces[faceIdx] (replaces the old
         // triangle_vertex_ids map). Both eliminate millions of serial hash inserts on entry.
         std::vector<int> face_to_mesh_index;
         float spatial_cell_size = 0.0f;
          struct SpatialBucket {
              EditableSpatialCellKey key;
              int start_index = 0;
              int count = 0;
          };
          std::vector<SpatialBucket> spatial_buckets;
          std::vector<int> vertex_spatial_indices;
         std::vector<uint32_t> vertex_mark_stamps;
         uint32_t vertex_mark_generation = 1;
         std::vector<uint32_t> triangle_mark_stamps;
         uint32_t triangle_mark_generation = 1;
         EditableMeshSelection selection;
         // Half-edge topology built from vertices + polygon_faces on every
         // cache rebuild. Vertex ids and polygon-face ids match this cache
         // 1:1. half_edge_valid == false => build or validation failed
         // (operators must keep using the legacy triangle-soup paths).
         MeshEdit::HalfEdgeMesh half_edge;
         MeshEdit::HalfEdgeBuildResult half_edge_build;
         bool half_edge_valid = false;
         // True when the cache was built in the cheaper sculpt-only layout (skips some
         // edit-mode topology). Lets ensureEditableMeshCache avoid a needless rebuild.
         bool built_minimal_for_sculpt = false;
         // Non-owning back-pointer to the flat (direct) SoA mesh this cache was built from, or
         // null for a per-face-Triangle (facade) cache. When set, sculpt/edit write-back targets
         // the mesh's DNA SoA directly instead of the empty source_triangles facade list. The
         // mesh is owned by direct_mesh_nodes/world.
         class TriangleMesh* flat_source_mesh = nullptr;
         // Flat cache only: editable vertex -> SoA vertex ids (CSR). The editable cache welds
         // coincident positions into ONE editable vertex (so a UV seam's split SoA copies move
         // together — without this they tear apart under a normal-direction brush), so one
         // editable vertex maps to one OR MORE SoA vertices. Write-back (syncFlatSculptVerticesToSoA)
         // scatters the edited position/normal to every SoA copy. Empty for a facade cache.
         std::vector<int> flat_soa_offsets;      // size = vertices.size() + 1
         std::vector<uint32_t> flat_soa_data;    // SoA vertex ids, grouped per editable vertex

         // --- Polygon (quad/ngon) grouping for LOOP NORMALS ---
         // Source triangles grouped by Triangle::faceIndex so sculpt/shading can compute ONE
         // normal per polygon (the quad) instead of per split triangle — the latter exposes the
         // triangulation diagonal as a facet/fold. Built lazily on the first normal recompute
         // (O(triangles), once per cache lifetime); empty/identity when the mesh carries no
         // faceIndex (faceIndex < 0), in which case shading falls back to per-triangle normals.
         std::vector<int> polygon_tri_off;   // CSR offsets, size = numPolygons + 1
         std::vector<int> polygon_tri_data;  // source-triangle indices, grouped by polygon
         std::vector<int> tri_to_polygon;    // per source-triangle -> dense polygon slot (or -1)
         bool polygon_grouping_built = false;
         bool has_polygon_grouping = false;  // true once a real quad/ngon grouping exists
         void buildPolygonGrouping();        // groups source_triangles by Triangle::faceIndex

         // --- index -> triangle accessors (Phase 1 SoA migration) ---
         class Triangle* triangleAt(int idx) const {
             return (idx >= 0 && idx < static_cast<int>(source_triangles.size()))
                 ? source_triangles[static_cast<size_t>(idx)].get() : nullptr;
         }
         const std::shared_ptr<class Triangle>& triangleSharedAt(int idx) const {
             static const std::shared_ptr<class Triangle> kNull;
             return (idx >= 0 && idx < static_cast<int>(source_triangles.size()))
                 ? source_triangles[static_cast<size_t>(idx)] : kNull;
         }
         class Triangle* refTri(const EditableVertexRef& r) const { return triangleAt(r.triangle_index); }
         const std::shared_ptr<class Triangle>& refTriShared(const EditableVertexRef& r) const { return triangleSharedAt(r.triangle_index); }
         class Triangle* faceTri(const EditableFace& f) const { return triangleAt(f.triangle_index); }
         const std::shared_ptr<class Triangle>& faceTriShared(const EditableFace& f) const { return triangleSharedAt(f.triangle_index); }
     };
     struct SculptControlNode {
         Vec3 local_position;
         Vec3 local_normal;
         float area_weight = 1.0f;
         bool is_boundary = false;
         std::vector<int> neighbor_ids;
         std::vector<int> source_vertex_ids;
         std::vector<float> source_weights;
     };
     struct SculptControlGraph {
         std::string object_name;
         size_t source_triangle_count = 0;
         Matrix4x4 source_object_transform = Matrix4x4::identity();
         float avg_edge_length = 0.0f;
         float spatial_cell_size = 0.0f;
         bool uses_spatial_leaf_nodes = false;
         size_t last_candidate_node_count = 0;
         size_t last_candidate_vertex_count = 0;
         size_t last_touched_leaf_count = 0;
         std::vector<SculptControlNode> nodes;
         std::vector<int> source_vertex_to_node_id;
         std::unordered_map<EditableSpatialCellKey, std::vector<int>, EditableSpatialCellKeyHasher> node_spatial_buckets;
     };
     struct SculptPBVHNode {
         AABB local_bounds;
         int parent_id = -1;
         int left_child_id = -1;
         int right_child_id = -1;
         int depth = 0;
         bool is_leaf = false;
         bool is_boundary = false;
         std::vector<int> vertex_ids;
     };
     struct SculptPBVH {
         std::string object_name;
         size_t source_triangle_count = 0;
         Matrix4x4 source_object_transform = Matrix4x4::identity();
         float avg_edge_length = 0.0f;
         int root_node_id = -1;
         int max_depth = 0;
         int leaf_vertex_limit = 64;
         size_t leaf_count = 0;
         size_t last_candidate_node_count = 0;
         size_t last_candidate_vertex_count = 0;
         std::vector<SculptPBVHNode> nodes;
         std::vector<int> source_vertex_to_leaf_id;
     };
     EditableMeshCache editable_mesh_cache;
     uint64_t editable_mesh_cache_revision_counter = 0;

     // GPU edit-mesh overlay sync (raster viewport wireframe/vertex/face pass).
     // Tracks what was last uploaded to the viewport backend so per-frame work
     // is limited to a params push; buffers re-upload only on real changes.
     struct GpuEditOverlaySync {
         bool active = false;            // backend currently holds overlay buffers
         bool geometry_dirty = true;     // vertex positions need re-upload
         bool drawn_this_frame = false;  // set by sync, cleared per frame in drawSelectionGizmos
         uint64_t cache_revision = 0;    // EditableMeshCache::revision last uploaded
         uint64_t selection_hash = 0;    // selection ids + soft-select params fingerprint
     };
     GpuEditOverlaySync gpu_edit_overlay_sync;

     SculptControlGraph sculpt_control_graph;
     SculptPBVH sculpt_pbvh;
     std::vector<Vec3> sculpt_updated_local_positions;
     std::vector<size_t> sculpt_dirty_mesh_cache_indices;
     struct MeshEditLayer {
         bool active = false;
         bool enabled = true;
         std::string object_name;
         std::vector<MeshEditTriangleState> base_states;
         std::vector<MeshEditTriangleState> edited_states;
     };
     MeshEditLayer mesh_edit_layer;
     struct PendingSerializedMeshEditLayer {
         bool has_data = false;
         bool enabled = true;
         std::string object_name;
         std::vector<std::array<Vec3, 3>> base_positions;
         std::vector<std::array<Vec3, 3>> edited_positions;
     };
     PendingSerializedMeshEditLayer pending_serialized_mesh_edit_layer;
     std::string active_mesh_edit_object_name;
     const class Triangle* active_mesh_edit_object_ptr = nullptr;
     float mesh_face_extrude_distance = 0.2f;
     float mesh_face_inset_amount = 0.25f;
     float mesh_vertex_weld_distance = 0.05f;
     float mesh_loop_cut_position = 0.5f;
     std::string modifier_panel_exit_object; // Object for which user manually exited edit mode
     bool mesh_edit_gpu_sync_pending = false;
     std::string mesh_edit_gpu_sync_object_name;
     ImVec2 mesh_overlay_view_min = ImVec2(0.0f, 0.0f);
     ImVec2 mesh_overlay_view_max = ImVec2(0.0f, 0.0f);
     bool mesh_overlay_view_valid = false;
     
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
         
         // Lazy Update Logic
         std::vector<InstanceTransform> pending_instances; // Instances waiting to be committed
         int pending_group_id = -1;      // ID of the group getting pending instances
     };
     ScatterBrushSettings scatter_brush;
     
     // Scatter Brush UI
     void drawScatterBrushPanel(UIContext& ctx);
     void handleScatterBrush(UIContext& ctx);  // Viewport brush interaction
     void drawBrushPreview(UIContext& ctx);    // Draw brush circle in viewport
     void drawSculptBrushViewportPreview(UIContext& ctx, const HitRecord& hit, bool ghost = false); // Alpha grid + rings for sculpt
     // Backend-independent ImGui overlay that tints masked vertices on the
     // shaded surface. Works in Solid AND Rendered viewport modes and during a
     // stroke (the GPU edit overlay runs only in Edit mode / Solid shading).
     void drawSculptMaskViewportOverlay(UIContext& ctx);
    
    // Terrain Brush Settings
    struct TerrainBrushSettings {
        bool enabled = false;
        int active_terrain_id = -1;
        float radius = 5.0f;
        float strength = 0.5f;
        float curve = 2.0f;
        int mode = 0; // 0=Raise, 1=Lower, 2=Flatten, 3=Smooth, 4=Stamp
        bool show_preview = true;
        
        // Flatten Params
        float flatten_target = 0.0f;
        bool use_fixed_height = false;
        
        // Stamp Params
        std::shared_ptr<class Texture> stamp_texture;
        float stamp_rotation = 0.0f; // Degrees
        Paint::StrokeMethod stroke_method = Paint::StrokeMethod::Space; // Stroke behavior
        
        // Paint Params
        int paint_channel = 0; // 0=R(Layer0), 1=G(Layer1), 2=B(Layer2), 3=A(Layer3)
    };
    TerrainBrushSettings terrain_brush;
    bool terrain_sculpt_proxy_active = false;
    Paint::PaintModeState paint_mode_state;
    enum class SculptBrushTool : int {
        Grab = 0,
        Inflate,
        Smooth,
        Flatten,
        Draw,
        Layer,
        Pinch,
        Clay,
        ClayStrips,
        Crease,
        Scrape,
        // Mask is not a deformer: it paints a per-vertex protection weight
        // (0 = free, 1 = frozen) that scales down every other brush's effect.
        // It moves no geometry, so the sculpt commit/BVH-refit tail is skipped.
        Mask,
        DrawSharp,   // crisp ridge/crease — Draw with a tightened falloff
        Nudge,       // tangential push along the stroke direction
        Blob,        // spherical swell (inflate + lateral gather)
        Fill,        // directional flatten: fill valleys (Ctrl = deepen)
        SnakeHook,   // drag the surface along the stroke into a hook/tentacle
        ElasticDeform, // Grab-family: soft, wide, volume-preserving pull
        Stamp        // imprint an alpha/mask texture along the surface normal
                     // (shares the Draw deform; the imprint shape comes from the
                     // brush footprint, i.e. the loaded alpha texture). With the
                     // Anchored stroke mode it presses at a fixed centre while the
                     // drag drives depth + rotation (non-accumulating single imprint).
    };
    // Grab-family tools share the drag-based solve path (primed candidate set +
    // start positions) rather than the per-dab worldDelta switch. Snake Hook is
    // included so its pulled region follows the cursor for the whole stroke
    // (like Grab) instead of dropping out of the per-frame candidate sphere.
    static inline bool isGrabFamilyTool(SculptBrushTool t) {
        return t == SculptBrushTool::Grab || t == SculptBrushTool::ElasticDeform ||
               t == SculptBrushTool::SnakeHook;
    }
     struct SculptModeState {
        bool enabled = false;
        bool compact_ui = true;
        std::string active_target_name;
        SculptBrushTool tool = SculptBrushTool::Draw;
        Paint::BrushSettings brush;
        float normal_strength = 0.35f;
        bool front_faces_only = false;
        bool accumulate_live = true;
        bool use_screen_space_radius = true;
        float screen_radius_px = 72.0f;
        bool mirror_x = false;
        bool mirror_y = false;
        bool mirror_z = false;
        // radial_symmetry: repeat each stroke as N evenly-spaced rotations about
        // an object-local axis (through the object origin). count >= 2 to apply.
        bool radial_symmetry = false;
        int radial_count = 6;     // number of copies including the primary
        int radial_axis = 1;      // 0 = X, 1 = Y, 2 = Z (object-local)
        // --- Dynamic "wet clay" brush (deposit wet → settle → flow → dry) ---
        bool wet_clay_enabled = false;
        float wet_clay_wetness = 0.7f;   // wetness injected at each deposit (0..1)
        float wet_clay_dry_rate = 0.5f;  // wetness lost per second (higher = sets faster)
        float wet_clay_settle = 0.6f;    // settle/relax rate while wet (0..1, surface tension)
        float wet_clay_flow = 0.0f;      // gravity flow rate while wet (0..1; 0 = off)
        float wet_clay_yield = 0.15f;    // min slope (rel. edge len) before clay flows
        bool wet_clay_water_only = false; // paint wetness only, no geometry deposit (re-wet)
        // Phase 4 — heterogeneous density: when on, a spatial noise field varies the local
        // flow mobility, so the mud creeps down at different rates (marbled flow) instead
        // of one uniform sheet. Off = homogeneous (uniform mobility).
        bool wet_clay_hetero = false;
        float wet_clay_hetero_scale = 2.0f; // density-noise frequency (higher = finer)
        // Cohesion/viscosity: how strongly the FLOWING region stays bonded. High = smooth
        // viscous tongue (putty); low = the flow keeps its roughness and breaks into chunks
        // (mud), especially with heterogeneous density on.
        float wet_clay_cohesion = 0.6f;
        // Per-dab surface relaxation: a light ISOTROPIC Laplacian on the touched verts
        // after each additive deposit. Sculpting the LINEAR (triangulated) evaluated mesh
        // makes brushed quads non-planar, and a quad rendered as two fixed-diagonal
        // triangles folds visibly along that diagonal ("triangulated-quad facets"). This
        // pass pulls each touched vertex a fraction toward its neighbour centroid, which
        // removes that sub-quad fold while the brush re-deposits the broad shape each dab —
        // so it gives a smooth quad surface WITHOUT paying for a multires limit surface on
        // dense meshes. Skipped for Grab/Smooth/Mask (those drive their own coherent solve).
        bool surface_relax_enabled = true;
        float surface_relax_strength = 0.4f; // 0 = off, 1 = aggressive smoothing
        // Reserved for the disabled experimental GPU sculpt path.
        bool use_gpu = false;
        SculptModeState() { brush.radius = 0.3f; brush.strength = 1.0f; brush.falloff = 0.75f; }
    };
    SculptModeState sculpt_mode_state;
    struct SculptStrokeState {
        bool active = false;
        bool changed = false;
        std::string object_name;
        Vec3 start_world_hit;
        bool has_last_world_hit = false;
        Vec3 last_world_hit;
        Vec3 last_world_hit_normal;
        Vec3 stroke_normal;
        
        // Gelişmiş Stroke Metodları (Line, Curve)
        bool is_line_drawing = false;
        Vec3 line_end_world;
        bool execute_line_stroke = false;
        
        bool is_curve_drawing = false;
        bool is_curve_bending = false;
        Vec3 curve_end_world;
        Vec3 curve_control_world;
        bool execute_curve_stroke = false;
        // Distance-gated dab scheduling (additive brushes only). last_dab_world is
        // the world position of the last emitted dab; new dabs are placed only every
        // `spacing` units travelled, with the remainder carried here. This stops the
        // same spot getting hammered every frame on slow / high-tess meshes (the
        // "circles piling up" artifact) while keeping even spacing at any speed.
        Vec3 last_dab_world;
        Vec3 last_dab_normal;
        bool has_last_dab = false;
        std::unordered_map<int, Vec3> grab_start_local_positions;
        std::unordered_map<int, float> grab_weights_by_vertex;
        // Grab mirror: per-symmetry-combo (mirrorBits 1..7) FIXED candidate set,
        // captured on the stroke's first frame. Driving the mirror from this fixed
        // set (instead of a per-frame PBVH re-gather) keeps it drift-free exactly
        // like the main grab — otherwise pulled mirror verts drop out of the
        // start-centred query sphere mid-stroke and freeze into spikes/dents.
        std::unordered_map<int, std::vector<int>> grab_mirror_candidate_sets;
        // Grab radial: per-radial-copy (k = 1..count-1) FIXED candidate set,
        // captured on the stroke's first frame — same drift-free approach as the
        // mirror grab, but the start hit/drag are rotated about the radial axis.
        std::unordered_map<int, std::vector<int>> grab_radial_candidate_sets;
        // Anchored stamp (Blender-style): the brush centre, normal and tangent frame
        // are FROZEN at the stroke's first frame; the cursor drag then drives the
        // imprint DEPTH (drag length) and ROTATION (drag angle in that frame) instead
        // of moving the centre. grab_start_local_positions holds the frozen footprint
        // verts' rest positions; every frame re-writes pos = rest + anchorNormal *
        // (depth * footprintWeight) absolutely, so the imprint never accumulates and
        // live-resizes/rotates as you drag. anchored_primed gates the one-time capture.
        bool anchored_primed = false;
        Vec3 anchor_world;
        Vec3 anchor_normal;
        Vec3 anchor_tangent;
        Vec3 anchor_bitangent;
        std::vector<float> layer_accum;
        std::vector<float> clay_layer_accum;
        std::vector<float> clay_strips_layer_accum;
        std::unordered_map<const class Triangle*, MeshEditTriangleState> before_triangle_states;
        std::unordered_map<const class Triangle*, std::shared_ptr<class Triangle>> touched_triangles;
        // FLAT (direct SoA) sculpt undo: facade pointers above stay empty on a flat mesh, so capture
        // each touched SoA vertex's local rest pos+normal the FIRST time it is written this stroke
        // (in syncFlatSculptVerticesToSoA). At stroke end this seeds a FlatSculptEditCommand.
        // Key = SoA vertex id; value = {P_orig, N_orig} BEFORE the stroke's first touch of it.
        std::unordered_map<uint32_t, std::pair<Vec3, Vec3>> flat_before_soa;
    };
    SculptStrokeState sculpt_stroke_state;

    // Idle cache for the sculpt brush viewport preview. The preview re-raycasts the scene BVH
    // and the sculpt PBVH every frame to place the cursor; when the cursor and camera are
    // parked and no stroke is running, the result can't change, so we reuse the last hit and
    // skip both raycasts (was a constant idle CPU draw with the brush stationary).
    int sculpt_preview_cache_mx = 0;
    int sculpt_preview_cache_my = 0;
    bool sculpt_preview_cache_valid = false;
    bool sculpt_preview_cache_did_hit = false;
    Vec3 sculpt_preview_cache_point;   // HitRecord is forward-declared here, so cache the
    Vec3 sculpt_preview_cache_normal;  // two fields the preview actually reads, not the record.
    Vec3 sculpt_preview_cache_cam_from;
    Vec3 sculpt_preview_cache_cam_at;

    // Per-vertex sculpt mask. values[vertexId] in [0,1]: 0 = fully sculptable,
    // 1 = fully protected. Sized 1:1 with editable_mesh_cache.vertices and
    // rebuilt whenever the cache revision changes (topology edits invalidate it).
    struct SculptMaskState {
        std::string object_name;
        uint64_t cache_revision = 0;     // editable_mesh_cache.revision this mask was sized against
        std::vector<float> values;       // per-vertex protection weight, 0..1
        bool has_any = false;            // fast skip when nothing is masked
        float paint_strength = 0.5f;     // mask brush deposit scale per stroke
        bool show_overlay = true;        // tint masked verts in the viewport overlay
        uint64_t version = 0;            // bumped on any mask change; folds into overlay fingerprint
        void clear() { object_name.clear(); cache_revision = 0; values.clear(); has_any = false; ++version; }
    };
    SculptMaskState sculpt_mask_state;
    // Mask buffer operations. operation: 0=Clear, 1=Invert, 2=Fill, 3=Smooth, 4=Sharpen.
    void applySculptMaskOperation(int operation);
    // --- Dynamic wet-clay field ---------------------------------------------------
    // Per-vertex wetness, sized 1:1 with editable_mesh_cache.vertices and reset on cache
    // revision change. Brushes inject wetness into the verts they deposit into;
    // stepWetClayField() (run once per frame in sculpt mode) settles + flows the wet
    // region, decays the wetness, and evicts dried verts so cost is bounded to the
    // still-active set.
    struct SculptWetClayState {
        std::string object_name;
        uint64_t cache_revision = 0;
        std::vector<float> wetness;      // per-vertex, 0 = dry/locked, 1 = freshly wet
        std::vector<int> active_list;    // compact list of verts with wetness > 0 (deduped)
        // Flow operates on the NORMAL-protrusion above the surface the vertex had when it
        // first got wet (the "wall"): anchor = that rest position, normal = that surface
        // normal. material m = (pos-anchor)·normal. Flow transports m downhill and moves the
        // vertex along its normal, so a blob slides DOWN a vertical wall (not just levels Y).
        std::vector<Vec3> flow_anchor;
        std::vector<Vec3> flow_normal;
        bool has_any = false;
        void clear() { object_name.clear(); cache_revision = 0; wetness.clear(); active_list.clear(); flow_anchor.clear(); flow_normal.clear(); has_any = false; }
    };
    SculptWetClayState sculpt_wet_clay_state;
    // Mark the given editable-cache vertex ids as freshly wet (from the deposit path).
    // anchors[k] is the PRE-deposit position of vertexIds[k] (the wall reference).
    void depositWetClay(const std::vector<int>& vertexIds, const std::vector<Vec3>& anchors, float wetnessInject,
                        const std::vector<float>* injectWeights = nullptr);
    // Per-frame evolution of the wet-clay field: settle + flow + dry + evict. No-op when
    // the active set is empty.
    void stepWetClayField(UIContext& ctx);
    enum class MeshWorkspaceMode : int {
        Edit = 0,
        Sculpt
    };
    MeshWorkspaceMode mesh_workspace_mode = MeshWorkspaceMode::Edit;
    struct PaintBrushPreset {
        std::string name;
        Paint::BrushSettings brush;
    };
    std::vector<PaintBrushPreset> paint_brush_presets;
    char paint_brush_preset_name[64] = "Custom Brush";
    bool paint_brush_presets_initialized = false;
    float paint_brush_dock_width = 50.0f;
    float paint_layer_list_height = 160.0f; // user-resizable layer list height
    void ensurePaintBrushPresets();

    std::vector<PaintBrushPreset> sculpt_brush_presets;
    char sculpt_brush_preset_name[64] = "Soft Sculpt";
    bool sculpt_brush_presets_initialized = false;
    void ensureSculptBrushPresets();

    // Terrain Foliage Brush Settings (Paint to add/remove foliage)
    struct FoliageBrushSettings {
        bool enabled = false;           // Foliage brush mode active
        int active_group_id = -1;       // Which foliage layer to paint (by ID)
        float radius = 5.0f;            // Brush size in world units
        int density = 3;                // Instances per stroke
        int mode = 0;                   // 0=Add, 1=Remove     
        bool show_preview = true;       // Show brush circle in viewport
        bool lazy_update = true;       // If true, waits for mouse release to spawn instances (Better for weak GPUs)
        
        // Lazy Update Logic
        std::vector<InstanceTransform> pending_instances; 
        int pending_group_id = -1;
    };
    FoliageBrushSettings foliage_brush;

    // Water, River & Terrain UI
    void drawWaterPanel(UIContext& ctx);
    bool drawWaterSurfaceMaterialEditor(UIContext& ctx, WaterSurface& surf, bool allow_delete = false);
    void drawRiverPanel(UIContext& ctx);       // Bezier spline river editor
    void drawRiverGizmos(UIContext& ctx, bool& gizmo_hit);  // River spline visualization
    void drawTerrainPanel(UIContext& ctx);
     // Central query: is ANY viewport tool/brush mode active?
     // Used by idle-tier system so every tool keeps the viewport alive.
     // Add new tools here — one place, all modes covered.
     bool isAnyViewportToolActive() const {
         return sculpt_stroke_state.active ||
                paint_mode_state.enabled ||
                terrain_brush.enabled ||
                foliage_brush.enabled ||
                scatter_brush.enabled ||
                hairUI.isPainting() ||
                (mesh_overlay_settings.enabled && mesh_overlay_settings.edit_mode) ||
                (sculpt_mode_state.enabled);
     }

     void handleTerrainBrush(UIContext& ctx);
     void handleTerrainFoliageBrush(UIContext& ctx);  // Foliage paint brush
     void handleHairBrush(UIContext& ctx);            // Hair paint brush
     // overrideHitPoint != nullptr => a single sub-step dab at that world point
     // (used internally for stroke interpolation on fast strokes; skips mouse
     // pick, stroke begin/end and re-substepping). Normal callers pass nullptr.
     void handleMeshSculpt(UIContext& ctx, const Vec3* overrideHitPoint = nullptr); // Mesh sculpt brush interaction
     void handleMeshPaint(UIContext& ctx);            // Mesh texture paint brush
     bool shouldShowPaintBrushDock() const;
     float getPaintBrushDockWidth() const;
     void drawHairBrushPreview(UIContext& ctx, const Vec3& hitPoint, const Vec3& hitNormal);       // Draw hair brush circle in viewport
    void tickProgressiveVertexSync(); // Called each frame to process a chunk
    void updateAutofocus(UIContext& ctx);  // Run autofocus logic (raycast center)
    
    // ═══════════════════════════════════════════════════════════════════════════
    // VOLUMETRIC UI (Unified VDB & Gas Simulation)
    // ═══════════════════════════════════════════════════════════════════════════
    void drawVolumetricPanel(UIContext& ctx);       // Unified list and properties
    void drawVDBVolumeProperties(UIContext& ctx, VDBVolume* vdb);  // Single VDB properties
    void importVDBVolume(UIContext& ctx);          // Import single VDB file
    void importVDBSequence(UIContext& ctx);        // Import VDB Sequence
    void drawVDBImportMenu(UIContext& ctx);        // Import menu items
    SceneHistory history;  // Command history for undo/redo
    // Fast lookup from Triangle pointer to its index in world.objects (for BVH Picking)
    std::unordered_map<const class Triangle*, int> tri_to_index;
    // Flat/proxy migration: dense meshes that live in world.objects as a single TriangleMesh
    // (no per-face facades) are registered here by node name + world.objects index, so the
    // facade-based UI (hierarchy listing, click selection) can treat them as first-class objects.
    // `rep` is a SINGLE representative facade (face 0) that serves as the UI handle: the
    // hierarchy/selection/gizmo/bbox/transform-sync machinery all read parentMesh->geometry for a
    // flat mesh and only need one facade as the handle, so one 136B facade replaces 12.6M of them.
    struct DirectMeshNode { std::shared_ptr<class TriangleMesh> mesh; int object_index = -1; std::shared_ptr<class Triangle> rep; };
    std::unordered_map<std::string, DirectMeshNode> direct_mesh_nodes;
    // Reverse lookup for viewport picking: a flat-mesh hit reports HitRecord.tri_mesh (no facade
    // pointer), so map the mesh pointer back to its representative facade to drive selection.
    std::unordered_map<const class TriangleMesh*, std::shared_ptr<class Triangle>> direct_mesh_rep_by_ptr;
    // Interaction State
    bool is_dragging = false; // Tracks if a gizmo manipulation is in progress
    bool is_bvh_dirty = false; // Flag for lazy BVH updates
    bool focus_scene_edit_tab = false; // Auto-focus Scene Edit tab after model load
    bool mesh_edit_optix_targeted_sync_enabled = true;
    bool interactive_subdiv_preview_active = false;
    std::string interactive_subdiv_preview_object_name;
    // Dirty-signature for the steady-state (non-drag) subdivision preview refresh in
    // syncMeshEditState. Lets the refresh fire ONLY when the subdivision params or base
    // mesh actually change (was an every-frame full re-evaluate + scene rebuild), and
    // when it fires it now also queues a GPU sync so Rendered-mode backends update too
    // (previously only the CPU BVH was flagged → Rendered object stayed stale until some
    // other op triggered a rebuild/refit).
    std::size_t subdiv_preview_refresh_signature = 0;
    std::string subdiv_preview_refresh_object_name;
    bool subdiv_preview_refresh_valid = false;
    // Subdivision signature the editable display mesh in mesh_cache was last BUILT with.
    // ensureEditableMeshCache used to decide "preview out of date" by comparing the cached
    // triangle count to base*4^levels — a guess that Catmull-Clark / quad-recovery never
    // matches, so it re-evaluated the display mesh EVERY frame after a subdivide (bumping the
    // cache revision, re-uploading the GPU edit overlay, raising g_bvh_rebuild_pending) and
    // pinned the viewport in a permanent re-render. Gate on this signature instead.
    std::size_t editable_subdiv_display_signature = 0;
    std::string editable_subdiv_display_signature_object;
    bool editable_subdiv_display_signature_valid = false;
    // picking fails because Triangle::hit() reads stale local-space positions.
    bool picking_vertices_synced = false;
private:
    // --- UI Structure ---
    void drawPanels(UIContext& ctx);
    void drawStatusAndBottom(UIContext& ctx, float screen_x, float screen_y, float left_offset);
     void drawAuxWindows(UIContext& ctx);
     void drawAssetBrowser(UIContext& ctx, bool embedded = false);
      void appendAssetToScene(UIContext& ctx, const std::filesystem::path& asset_path, const std::string& display_name);
      bool appendAnimationClipAssetToScene(UIContext& ctx, const AssetRecord& asset, const std::string& display_name);
      bool raycastViewportPlacement(UIContext& ctx, const ImVec2& screen_pos, Vec3& hit_point, Vec3& hit_normal) const;
      bool raycastViewportHit(UIContext& ctx, const ImVec2& screen_pos, HitRecord& hit_record) const;
      void drawAssetDragGhost(UIContext& ctx, const std::string& asset_name, const Vec3& hit_point, const Vec3& bounds_min, const Vec3& bounds_max) const;
     bool ensureSelectedAssetPreviewTexture(UIContext& ctx, const std::filesystem::path& preview_path, int& width, int& height);
     bool ensureAssetBrowserThumbnailTexture(UIContext& ctx, const std::filesystem::path& preview_path, SDL_Texture*& out_texture, int& width, int& height);
     void releaseSelectedAssetPreviewTexture();
     void releaseAssetBrowserThumbnailTextures();
     void startAsyncAssetLibraryRefresh(const std::filesystem::path& root_path, const std::string& status_text = "Scanning assets...");
     void pollAsyncAssetLibraryRefresh();
     struct AssetSmartFolderPreset {
         std::string name;
         std::string search;
         std::string tag_filter;
         std::string folder_relative_dir;
         bool favorites_only = false;
         bool only_3d = true;
     };

    // --- Input / Editor ---
    void handleEditorShortcuts(UIContext& ctx);
    bool deleteSelectedLight(UIContext& ctx);
    void handleDeleteShortcut(UIContext& ctx);

    // --- Overlays & Gizmos ---
    bool drawOverlays(UIContext& ctx);
    void drawLightGizmos(UIContext& ctx, bool& gizmo_hit);
    void drawForceFieldGizmos(UIContext& ctx, bool& gizmo_hit);
    void drawParticleDebugOverlay(UIContext& ctx);
    // Build camera-facing billboards from all visible particle systems and upload
    // them to the Vulkan viewport backend for real (depth-tested) solid-mode render.
    void uploadParticleBillboards(UIContext& ctx);
    void drawSelectionGizmos(UIContext& ctx);
    void drawOverlayGrid(UIContext& ctx);
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
    void validateSelectionAgainstScene(UIContext& ctx);
    bool showSidePanel = true;
     bool show_controls_window = false; // Controls/Help window visibility
     bool show_asset_browser = false;
     AssetRegistry asset_registry;
     std::vector<std::filesystem::path> asset_library_paths;
     int active_asset_library_index = 0;
     std::unordered_map<std::string, std::pair<Vec3, Vec3>> asset_drag_bbox_cache;
     std::string asset_browser_search;
     std::string asset_browser_tag_filter;
     std::string asset_smart_folder_name;
     int asset_browser_view_mode = 1; // 0=Tiles, 1=Compact, 2=Details
     float asset_browser_thumbnail_size = 170.0f;
     bool asset_browser_only_3d = true;
     bool asset_browser_favorites_only = false;
     std::vector<AssetSmartFolderPreset> asset_smart_folders;
     int active_asset_smart_folder_index = -1;
     float asset_browser_folder_width = 260.0f;
     float asset_browser_details_height = 250.0f;
     std::string selected_asset_relative_dir;
     std::string selected_asset_folder_relative_dir;
     std::string selected_asset_tags_edit;
     std::string selected_asset_tags_edit_target;
     std::filesystem::path selected_asset_preview_texture_path;
     SDL_Texture* selected_asset_preview_texture = nullptr;
     int selected_asset_preview_texture_width = 0;
     int selected_asset_preview_texture_height = 0;
     struct AssetThumbnailCacheEntry {
         SDL_Texture* texture = nullptr;
         int width = 0;
         int height = 0;
         uint64_t last_used = 0;
     };
     std::unordered_map<std::string, AssetThumbnailCacheEntry> asset_browser_thumbnail_cache;
     uint64_t asset_browser_thumbnail_use_counter = 0;
     std::future<std::pair<AssetRegistry, bool>> asset_registry_refresh_future;
     std::filesystem::path pending_asset_library_root;
     std::string asset_library_refresh_status;
     bool asset_library_refresh_in_progress = false;
    // Layer thumbnail cache for Photoshop-style layer panel
    struct LayerThumbEntry {
        SDL_Texture* texture = nullptr;
        uint32_t layer_id = 0;
        uint64_t content_hash = 0; // simple hash to detect changes
        void release() { if (texture) { SDL_DestroyTexture(texture); texture = nullptr; } }
    };
    std::vector<LayerThumbEntry> layer_thumb_cache;
    void releaseLayerThumbnails();
    SDL_Texture* getOrCreateLayerThumbnail(UIContext& ctx, Paint::PaintLayerData* layer, Paint::PaintChannel channel);

    bool showResolutionPanel = true; // class üyesi
    bool camera_initialized = false;
    Vec3 camera_initial_pos;
    Vec3 camera_initial_target;
    float camera_initial_fov;
   
    bool show_scene_log = false; // Default closed
    bool focus_bottom_panel_next_frame = false;
    bool pending_project_ui_restore = false;
    // --- Modern dockable panel layout (ImGui docking) ---
    bool docking_enabled = true;        // Master toggle: dockable panels vs. legacy pinned layout
    bool docking_layout_dirty = false;   // Rebuild the default DockBuilder layout next frame
    unsigned int dockspace_id = 0;      // Cached ID of the dockspace
    unsigned int dock_bottom_id = 0;     // Cached ID of the bottom dock node
    void dockToBottom(const char* window_name);
    void drawDockSpaceHost(UIContext& ctx); // Hosts the DockSpace + default layout, feeds viewport rect back to legacy offsets
    // --- Detachable (tear-off) Properties sub-tabs ---
    // Side-effect-free editor tabs can be popped into their own dockable window so two
    // panels (e.g. Render Settings + Stylize) are usable at once. Indexed by active_properties_tab.
    bool properties_tab_popped_[16] = {};
    ImVec2 properties_pop_spawn_pos_[16] = {};          // where a freshly torn-off window first appears (cursor)
    bool properties_pop_spawn_pending_[16] = {};        // true only for THIS-session button pops; serialized restores use the imgui.ini position
    void drawPoppedTabContent(UIContext& ctx, int tab); // renders one popped tab's content
    void drawPoppedPropertyWindows(UIContext& ctx);     // hosts all currently popped tabs as windows
    void drawHairTabContent(UIContext& ctx);            // hair tab body (shared by main panel + popped window)
    float side_panel_width = 360.0f; // Resizable Left Panel width
    float bottom_panel_height = 100.0f; // Default height
    float preferred_bottom_panel_height = 100.0f; // Persist desired height; avoid shrinking permanently during minimized/small viewport frames
    float hierarchy_panel_height = 250.0f; // New: Resizable hierarchy list height
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

    std::string serialize();  // Serialize ImGui layout and SceneUI state
    void deserialize(const std::string& data); // Deserialize and apply UI state

    std::map<std::string, std::vector<std::pair<int, std::shared_ptr<class Triangle>>>> mesh_cache;
  
    size_t cached_scene_triangle_count = 0;
    std::unordered_map<std::string, size_t> cached_triangle_count_by_object;

    // Accurate geometry breakdown for the debug HUD. cached_scene_triangle_count above counts
    // mesh_cache ENTRIES (a flat mesh = 1 representative facade), so it badly undercounts flat
    // scenes; this is the real per-triangle accounting (facade vs flat vs skinned vs animated),
    // recomputed only when the geometry generation or object count changes (not per frame).
    struct SceneGeometryStats {
        uint64_t generation = ~0ull;
        size_t object_count = ~0ull;
        size_t facade_tris = 0;    // per-face Triangle facades (parentMesh-backed or standalone)
        size_t flat_tris = 0;      // SoA TriangleMesh-as-Hittable faces
        size_t skinned_tris = 0;   // facade + flat tris carrying skin weights
        size_t animated_tris = 0;  // tris whose node has a timeline transform track
        size_t flat_meshes = 0;    // number of flat TriangleMesh objects
        size_t facade_nodes = 0;   // distinct node names still emitting facades
    };
    SceneGeometryStats scene_geometry_stats;
    void refreshSceneGeometryStats(class SceneData& scene);
    // Sequential cache for ImGui Clipper (Visualization)
    std::vector<std::pair<std::string, std::vector<std::pair<int, std::shared_ptr<class Triangle>>>>> mesh_ui_cache;
    
    // Bounding Box Cache - avoids recalculating bounds every frame (HUGE perf win for large objects)
    // Key: nodeName, Value: {bb_min, bb_max}
    std::map<std::string, std::pair<Vec3, Vec3>> bbox_cache;

    // Memoized LIVE world-AABB of a selected physics body, keyed by node. A body's
    // verts move every sim step but are STATIC when stopped, yet the gizmo/outline
    // re-walked them every frame — pinning the idle UI at ~6% CPU. Cached against
    // SceneData::bodyGeomVersion() (bumped on any body write-back/reset) so a stopped
    // body recomputes only once. Tuple: {geom_version, bb_min, bb_max}.
    std::unordered_map<std::string, std::tuple<uint64_t, Vec3, Vec3>> body_aabb_memo_;
    // Live world-AABB of a body's source mesh, memoized against bodyGeomVersion().
    // Returns false if the node has no cached mesh yet. Recomputes while dragging.
    bool bodyWorldAABB(UIContext& ctx, const std::string& node, Vec3& out_min, Vec3& out_max);

    // Selection-outline hull candidate cache: local-space extremal points per object.
    // Populated lazily on first selection; invalidated when mesh geometry changes.
    // Per-frame outline only projects these (~64 pts) instead of all triangle vertices.
    std::map<std::string, std::vector<Vec3>> hull_candidate_cache;

    // Selection-outline skinned-pose cache. Key: object name. Value: hash of the
    // bone-matrix buffer the last time we ran apply_skinning() for the selection
    // raster. If the hash matches we reuse the cached vertices[i].position the
    // triangles already hold — skipping a full CPU skinning pass per frame.
    std::map<std::string, uint64_t> selection_skin_pose_hash;

    // Selection-outline drawcall cache. When camera + transform + pose + screen
    // + style are unchanged frame-to-frame, replay the prior frame's boundary
    // pixels straight into ImGui instead of re-running projection, raster,
    // flood-fill and per-boundary BVH occlusion. Cuts the gizmo's main-thread
    // cost to a memcpy-ish replay on held-camera frames.
    struct SelectionOutlineFrameCache {
        uint64_t hash = 0;
        // Run-length encoded: a row of consecutive same-colour boundary
        // pixels is emitted as one wide rect instead of N per-pixel rects.
        // ImGui's vertex buffer (and the GPU draw it produces) shrinks ~5-10x
        // on typical silhouettes — that was the real per-frame bottleneck.
        struct Run { float sx, sy; uint16_t len; uint32_t col; };
        std::vector<Run> runs;
        float thickness = 0.0f;
        int scale = 1;
    };
    std::map<std::string, SelectionOutlineFrameCache> selection_outline_frame_cache;


    // Material Slots Cache - avoids scanning all triangles every frame to get unique material IDs
    // Key: nodeName, Value: list of unique material IDs used by that object
    std::map<std::string, std::vector<uint16_t>> material_slots_cache;

    bool mesh_cache_valid = false;
    size_t last_scene_obj_count = 0; // Tracking for cache invalidation
    size_t last_scene_light_count = 0;
    size_t last_scene_camera_count = 0;
    size_t last_scene_vdb_count = 0;
    size_t last_scene_gas_count = 0;
    size_t last_scene_forcefield_count = 0;
    bool selection_validation_pending = false;
    
    // Lazy CPU Vertex Sync - objects that need CPU update before picking
    // In TLAS mode, we skip CPU vertex update on gizmo release for instant response.
    // Instead, we mark objects as "needing sync" and only update when picking is attempted.
    std::set<std::string> objects_needing_cpu_sync;
    std::unordered_map<std::string, Matrix4x4> last_synced_transforms;
    void ensureCPUSyncForPicking(UIContext& ctx); // Called before mouse picking to sync pending objects

    // Tracks whether a full CPU vertex sync has been performed since the last
    // scene load / cache invalidation. Without this, Solid/Matcap viewport
  
    // Undo/Redo System
   
    
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

    // Lasso selection + mesh-edit UI state (restored: lost when a concurrent edit
    // clobbered scene_ui.h).
    bool is_lasso_selecting = false;
    std::vector<ImVec2> lasso_points;
    int mesh_cc_bake_levels = 2;          // Catmull-Clark "Apply" subdivision levels
    float mesh_edge_crease_value = 1.0f;  // crease weight written by the Crease slider
    
   
    
    // Texture Safety
    std::vector<std::shared_ptr<class Texture>> texture_graveyard;
    void manageTextureGraveyard();

    struct UvPreviewCacheEntry {
        std::array<Vec2, 3> uvs;
    };
    bool uv_workflow_cache_dirty = true;
    std::string uv_workflow_cached_object_name;
    uint16_t uv_workflow_cached_material_id = 0xFFFF;
    int uv_workflow_cached_uv_set = -1;
    size_t uv_workflow_cached_object_triangle_count = 0;
    int uv_workflow_cached_max_uv_sets = 1;
    std::vector<std::shared_ptr<class Triangle>> uv_workflow_cached_triangles;
    std::vector<UvPreviewCacheEntry> uv_workflow_preview_entries;
    
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

float getMainMenuReservedHeight();


