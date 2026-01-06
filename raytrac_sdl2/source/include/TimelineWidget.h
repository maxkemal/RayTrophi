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
    Terrain  // For terrain morphing animation
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
    ScaleX, ScaleY, ScaleZ
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
    bool isPlaying() const { return is_playing; }
    std::string selected_track;
private:
    // ===== DRAWING FUNCTIONS =====
    void drawPlaybackControls(UIContext& ctx);
    void drawTrackList(UIContext& ctx, float list_width, float canvas_height);
    void drawTimelineCanvas(UIContext& ctx, float canvas_width, float canvas_height);
    void drawFrameNumbers(ImDrawList* draw_list, ImVec2 canvas_pos, float canvas_width);
    void drawKeyframeDiamond(ImDrawList* draw_list, float x, float y, ImU32 color, bool selected);
    void drawCurrentFrameIndicator(ImDrawList* draw_list, ImVec2 canvas_pos, float canvas_height);
    
    // ===== INPUT HANDLING =====
    void handleZoomPan(ImVec2 canvas_pos, ImVec2 canvas_size);
    void handleKeyframeInteraction(UIContext& ctx, ImVec2 canvas_pos, ImVec2 canvas_size);
    void handleScrubbing(ImVec2 canvas_pos, float canvas_width);
    
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
    
    // Selection & Interaction
 
    int selected_keyframe_frame = -1;
    bool is_dragging_keyframe = false;
    int drag_start_frame = -1;
    int context_menu_frame = 0;  // Frame for right-click context menu
    
    // Group expansion state
    bool group_objects_expanded = true;
    bool group_lights_expanded = true;
    bool group_cameras_expanded = true;
    bool group_world_expanded = true;
    bool group_terrain_expanded = true;
    
    // Colors
    static constexpr ImU32 COLOR_TRANSFORM = IM_COL32(100, 150, 255, 255);  // Blue
    static constexpr ImU32 COLOR_MATERIAL = IM_COL32(100, 255, 150, 255);   // Green
    static constexpr ImU32 COLOR_LIGHT = IM_COL32(255, 200, 100, 255);      // Orange
    static constexpr ImU32 COLOR_CAMERA = IM_COL32(200, 100, 255, 255);     // Purple
    static constexpr ImU32 COLOR_WORLD = IM_COL32(255, 150, 200, 255);      // Pink
    static constexpr ImU32 COLOR_TERRAIN = IM_COL32(139, 90, 43, 255);      // Brown (terrain)
    static constexpr ImU32 COLOR_SELECTED = IM_COL32(255, 255, 255, 255);   // White
    static constexpr ImU32 COLOR_GRID = IM_COL32(60, 60, 60, 255);
    static constexpr ImU32 COLOR_CURRENT_FRAME = IM_COL32(255, 80, 80, 200);
};
