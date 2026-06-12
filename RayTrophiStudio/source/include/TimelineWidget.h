/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          TimelineWidget.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once
#include "imgui.h"
#include "imgui_internal.h"
#include "KeyframeSystem.h"
#include <string>
#include <vector>
#include <map>

// Forward declarations
struct UIContext;
struct SceneData;

// Track group types for organizing tracks
enum class TrackGroup {
    Objects = 0,
    Lights,
    Cameras,
    World,
    Terrain,  // For terrain morphing animation
    Water,    // For water wave parameter animation
    Gas       // For gas/smoke emitter animation
};


// Timeline editor display mode (combo in the toolbar).
// Both modes share playhead/zoom/pan/track-selection state, which is why the
// graph editor lives inside TimelineWidget instead of a separate bottom tab.
enum class TimelineEditorMode {
    DopeSheet = 0,   // classic keyframe diamonds per track
    GraphEditor = 1  // per-channel value curves (transform channels)
};

// Keyframe insert type for separate L/R/S options
enum class KeyframeInsertType {
    Location,
    Rotation,
    Scale,
    LocRot,
    All
};

// Channel type for sub-tracks
enum class ChannelType {
    None = 0,       // Main entity track
    Location,       // Position sub-channel (Compound)
    Rotation,       // Rotation sub-channel (Compound)
    Scale,          // Scale sub-channel (Compound)
    Material,       // Material sub-channel
    
    // Detailed axis channels
    LocationX, LocationY, LocationZ,
    RotationX, RotationY, RotationZ,
    ScaleX, ScaleY, ScaleZ,

    // Light sub-channels
    LightPosX, LightPosY, LightPosZ,
    LightColR, LightColG, LightColB,
    LightIntensity,
    LightDirX, LightDirY, LightDirZ,

    // Camera sub-channels
    CamPosX, CamPosY, CamPosZ,
    CamTgtX, CamTgtY, CamTgtZ,
    CamFOV,
    CamFocusDist,
    CamLensRad,

    // Material sub-channels
    MatAlbedoR, MatAlbedoG, MatAlbedoB,
    MatOpacity,
    MatRoughness,
    MatMetallic,
    MatClearcoat,
    MatTransmission,
    MatIOR,
    MatEmissionR, MatEmissionG, MatEmissionB,
    MatNormalStrength,
    MatEmissionStrength
};

// Visual representation of a track in the timeline
struct TimelineTrack {
    std::string name;           // Display name
    std::string entity_name;    // Reference to timeline track key
    std::string parent_entity;  // Parent entity name (for sub-tracks)
    TrackGroup group;
    ChannelType channel = ChannelType::None;  // None = main track, L/R/S = sub-track
    ImU32 color;
    bool expanded = false;      // For main tracks with sub-channels
    bool visible = true;
    bool is_sub_track = false;  // True for L/R/S sub-tracks
    int depth = 0;              // Indentation level (0=main, 1=sub)
    
    // Keyframe info cached for this track
    std::vector<int> keyframe_frames;
};

// Blender-style custom timeline widget
class TimelineWidget {
public:
    TimelineWidget() = default;
    
    // Main draw function - call from scene_ui
    void draw(UIContext& ctx);
    
    // Public state (for external access)
    int getCurrentFrame() const { return current_frame; }
    void setCurrentFrame(int frame) { current_frame = frame; }
    TimelineEditorMode getEditorMode() const { return editor_mode; }
    void setEditorMode(TimelineEditorMode mode) {
        editor_mode = mode;
        if (editor_mode == TimelineEditorMode::GraphEditor) {
            graph_fit_pending = true;
        }
    }
    // Timeline frame range — the single source of truth for how long the scene runs.
    // Physics disk-bake uses this (NOT the sequence-render range) so a bake always
    // covers the whole timeline regardless of render output settings.
    int getStartFrame() const { return start_frame; }
    int getEndFrame() const { return end_frame; }
    bool isPlaying() const { return is_playing; }
    std::string selected_track;

    // Force the next handleSelectionSync to re-apply selected_track from the live
    // viewport selection, even if the selection itself didn't change. Used when a
    // panel (e.g. World) temporarily hijacks selected_track and then releases it,
    // so keying objects works again without forcing the user to reselect.
    void invalidateSelectionSync() { last_selection_.clear(); selection_sync_force_ = true; }

    // Reset timeline state for new projects
    void reset() {
        current_frame = 0;
        is_playing = false;
        tracks.clear();
        tracks_dirty = true;
        zoom = 1.0f;
        pan_offset = 0.0f;
        selected_track = "";
        selected_keyframe_frame = -1;
        is_dragging_keyframe = false;
        last_selection_.clear();
        selection_sync_force_ = false;
        lastSyncedAnimCount = 0;  // re-sync animation data after project reload
        // Graph editor state
        editor_mode = TimelineEditorMode::DopeSheet;
        for (int i = 0; i < CURVE_CHANNEL_COUNT; ++i) graph_channel_visible[i] = true;
        graph_value_center = 0.0f;
        graph_pixels_per_unit = 40.0f;
        graph_fit_pending = true;
        graph_sel_channel = -1;
        graph_sel_frame = -1;
        graph_drag_mode = 0;
        anim_reapply_requested_ = false;
        graph_groups_expanded.clear();
    }

private:
    // ===== DRAWING FUNCTIONS =====
    void drawPlaybackControls(UIContext& ctx);
    void drawSelectedAnimGraphInspector(UIContext& ctx);
    void drawTrackList(UIContext& ctx, float list_width, float canvas_height);
    void drawTimelineCanvas(UIContext& ctx, float canvas_width, float canvas_height);
    void drawFrameNumbers(ImDrawList* draw_list, ImVec2 canvas_pos, float canvas_width);
    void drawKeyframeDiamond(ImDrawList* draw_list, float x, float y, ImU32 color, bool selected);
    void drawCurrentFrameIndicator(ImDrawList* draw_list, ImVec2 canvas_pos, float canvas_height);
    
    // ===== INPUT HANDLING =====
    void handleZoomPan(ImVec2 canvas_pos, ImVec2 canvas_size);
    void handleKeyframeInteraction(UIContext& ctx, ImVec2 canvas_pos, ImVec2 canvas_size);
    void handleScrubbing(ImVec2 canvas_pos, float canvas_width);

    // ===== GRAPH EDITOR =====
    void drawGraphChannelList(UIContext& ctx, float list_width);
    void drawGraphCanvas(UIContext& ctx, float canvas_width, float canvas_height);
    void fitGraphView(UIContext& ctx, float canvas_height);
    float valueToPixelY(float value, float canvas_height) const;
    float pixelYToValue(float y, float canvas_height) const;
    
    // ===== DATA MANAGEMENT =====
    void rebuildTrackList(UIContext& ctx);
    void handleSelectionSync(UIContext& ctx);  // Lightweight selection sync (runs every frame)
    void syncFromAnimationData(UIContext& ctx);  // Import animation keyframes (runs once)
    void insertKeyframeForTrack(UIContext& ctx, const std::string& track_name, int frame);
    void insertKeyframeType(UIContext& ctx, const std::string& track_name, int frame, KeyframeInsertType type);
    void deleteKeyframe(UIContext& ctx, const std::string& track_name, int frame);
    void moveKeyframe(UIContext& ctx, const std::string& track_name, int old_frame, int new_frame);
    void duplicateKeyframe(UIContext& ctx, const std::string& track_name, int src_frame, int dst_frame);
    int frameToPixelX(int frame, float canvas_width) const;
    int pixelXToFrame(float x, float canvas_width) const;
    
    // ===== STATE =====
    // Frame range
    int start_frame = 0;
    int end_frame = 250;
    int current_frame = 0;
    
    // Playback
    bool is_playing = false;
    // Loop playback. Default OFF: playback stops at end_frame. When the timeline
    // drives a simulation, looping forced a full re-bake + sim-cache wipe every
    // wrap (memory thrash), so looping is opt-in via the timeline toolbar.
    bool loop_enabled = false;
    float last_frame_time = 0.0f;
    
    // View controls
    float zoom = 1.0f;          // 1.0 = normal, >1 = zoomed in
    float pan_offset = 0.0f;    // Horizontal scroll offset in frames
    
    // Layout
    float track_height = 24.0f;
    float legend_width = 180.0f;
    float header_height = 20.0f;  // Reduced for compact timeline
    
    // Track data
    std::vector<TimelineTrack> tracks;
    bool tracks_dirty = true;
    // Selection-sync state (was a function-static; promoted to a member so a panel
    // releasing a hijacked selected_track can force a re-sync, and reset() clears it).
    std::string last_selection_;
    bool selection_sync_force_ = false;
    size_t lastSyncedAnimCount = 0;  // how many AnimationData entries have been synced
    
    // Selection & Interaction

    int selected_keyframe_frame = -1;
    bool is_dragging_keyframe = false;
    int drag_start_frame = -1;
    int context_menu_frame = 0;  // Frame for right-click context menu

    // ===== GRAPH EDITOR STATE =====
    TimelineEditorMode editor_mode = TimelineEditorMode::DopeSheet;
    bool graph_channel_visible[32] = {
        true, true, true, true, true, true, true, true, true, true,
        true, true, true, true, true, true, true, true, true, true,
        true, true, true, true, true, true, true, true, true, true,
        true, true
    };
    float graph_value_center = 0.0f;      // channel value at the vertical centre of the plot area
    float graph_pixels_per_unit = 40.0f;  // vertical zoom
    bool graph_fit_pending = true;        // auto-fit on first open / on request (F)
    int graph_sel_channel = -1;           // selected curve key: channel index (CURVE_*)
    int graph_sel_frame = -1;             // selected curve key: frame
    int graph_drag_mode = 0;              // 0=none 1=key 2=in-handle 3=out-handle
    // Graph edits (value/handle drags, interp changes) happen at an unchanged
    // current_frame, so draw()'s "frame changed?" gate would skip re-applying
    // keyframes to the scene; this flag forces one re-apply pass.
    bool anim_reapply_requested_ = false;
    
    // Group expansion state
    bool group_objects_expanded = true;
    bool group_lights_expanded = true;
    bool group_cameras_expanded = true;
    bool group_world_expanded = true;
    bool group_terrain_expanded = true;
    bool group_water_expanded = true;   // NEW: Water track group
    bool group_gas_expanded = true;     // NEW: Gas/Emitter track group
    std::map<std::string, bool> graph_groups_expanded;
    
    // Colors
    static constexpr ImU32 COLOR_TRANSFORM = IM_COL32(100, 150, 255, 255);  // Blue
    static constexpr ImU32 COLOR_MATERIAL = IM_COL32(100, 255, 150, 255);   // Green
    static constexpr ImU32 COLOR_LIGHT = IM_COL32(255, 200, 100, 255);      // Orange
    static constexpr ImU32 COLOR_CAMERA = IM_COL32(200, 100, 255, 255);     // Purple
    static constexpr ImU32 COLOR_WORLD = IM_COL32(255, 150, 200, 255);      // Pink
    static constexpr ImU32 COLOR_TERRAIN = IM_COL32(139, 90, 43, 255);      // Brown (terrain)
    static constexpr ImU32 COLOR_WATER = IM_COL32(64, 164, 223, 255);       // Ocean Blue (water)
    static constexpr ImU32 COLOR_GAS = IM_COL32(255, 140, 60, 255);         // Fire Orange (gas/flame)
    static constexpr ImU32 COLOR_SELECTED = IM_COL32(255, 255, 255, 255);   // White
    static constexpr ImU32 COLOR_GRID = IM_COL32(60, 60, 60, 255);
    static constexpr ImU32 COLOR_CURRENT_FRAME = IM_COL32(255, 80, 80, 200);
};

