#include "SequencerAdapter.h"
#include "imgui.h"
#include "imgui_internal.h"  // For ImRect
#include <algorithm>

// Color scheme for different track types
static const unsigned int TRACK_COLOR_TRANSFORM = IM_COL32(100, 150, 255, 255);  // Blue
static const unsigned int TRACK_COLOR_MATERIAL = IM_COL32(100, 255, 150, 255);   // Green
static const unsigned int TRACK_COLOR_LIGHT = IM_COL32(255, 255, 100, 255);      // Yellow
static const unsigned int TRACK_COLOR_CAMERA = IM_COL32(200, 100, 255, 255);     // Purple
static const unsigned int TRACK_COLOR_WORLD = IM_COL32(255, 150, 100, 255);      // Orange

SequencerAdapter::SequencerAdapter(TimelineManager& timeline, SceneData& scene)
    : timeline(timeline), scene(scene) {
    rebuildTracks();
}

void SequencerAdapter::rebuildTracks() {
    items.clear();
    
    // Iterate through all tracks in timeline
    for (auto& [entity_name, track] : timeline.tracks) {
        if (track.keyframes.empty()) continue;
        
        // Calculate frame range from keyframes
        int min_frame = track.keyframes.front().frame;
        int max_frame = track.keyframes.back().frame;
        
        // Check what types of keyframes exist
        bool has_transform = false;
        bool has_material = false;
        bool has_light = false;
        bool has_camera = false;
        
        for (const auto& kf : track.keyframes) {
            has_transform |= kf.has_transform;
            has_material |= kf.has_material;
            has_light |= kf.has_light;
            has_camera |= kf.has_camera;
        }
        
        // Create tracks for each animated property
        if (has_transform) {
            SequencerItem item;
            item.name = entity_name + " (Transform)";
            item.entity_name = entity_name;
            item.type = TrackType::Object_Transform;
            item.start_frame = min_frame;
            item.end_frame = max_frame;
            item.color = TRACK_COLOR_TRANSFORM;
            items.push_back(item);
        }
        
        if (has_material) {
            SequencerItem item;
            item.name = entity_name + " (Material)";
            item.entity_name = entity_name;
            item.type = TrackType::Object_Material;
            item.start_frame = min_frame;
            item.end_frame = max_frame;
            item.color = TRACK_COLOR_MATERIAL;
            items.push_back(item);
        }
        
        if (has_light) {
            SequencerItem item;
            item.name = entity_name + " (Light)";
            item.entity_name = entity_name;
            item.type = TrackType::Light;
            item.start_frame = min_frame;
            item.end_frame = max_frame;
            item.color = TRACK_COLOR_LIGHT;
            items.push_back(item);
        }
        
        if (has_camera) {
            SequencerItem item;
            item.name = entity_name + " (Camera)";
            item.entity_name = entity_name;
            item.type = TrackType::Camera;
            item.start_frame = min_frame;
            item.end_frame = max_frame;
            item.color = TRACK_COLOR_CAMERA;
            items.push_back(item);
        }
    }
    
    // Update global frame range
    if (!items.empty()) {
        frame_min = items[0].start_frame;
        frame_max = items[0].end_frame;
        
        for (const auto& item : items) {
            frame_min = std::min(frame_min, item.start_frame);
            frame_max = std::max(frame_max, item.end_frame);
        }
    }
}

void SequencerAdapter::Get(int index, int** start, int** end, int* type, unsigned int* color) {
    if (index < 0 || index >= static_cast<int>(items.size())) return;
    
    SequencerItem& item = items[index];
    if (start) *start = &item.start_frame;
    if (end) *end = &item.end_frame;
    if (type) *type = static_cast<int>(item.type);
    if (color) *color = item.color;
}

const char* SequencerAdapter::GetItemLabel(int index) const {
    if (index < 0 || index >= static_cast<int>(items.size())) return "";
    return items[index].name.c_str();
}

void SequencerAdapter::Add(int type) {
    // Not implemented yet - would create new track
}

void SequencerAdapter::Del(int index) {
    if (index < 0 || index >= static_cast<int>(items.size())) return;
    
    // Remove all keyframes for this track
    const std::string& entity_name = items[index].entity_name;
    auto it = timeline.tracks.find(entity_name);
    if (it != timeline.tracks.end()) {
        it->second.keyframes.clear();
    }
    
    items.erase(items.begin() + index);
}

void SequencerAdapter::Duplicate(int index) {
    // Not implemented yet - would duplicate track
}

size_t SequencerAdapter::GetCustomHeight(int index) {
    // Default height
    return 0;
}

void SequencerAdapter::CustomDraw(int index, ImDrawList* draw_list, const ImRect& rc, 
                                  const ImRect& legendRect, const ImRect& clippingRect, 
                                  const ImRect& legendClippingRect) {
    if (index < 0 || index >= static_cast<int>(items.size())) return;
    
    const SequencerItem& item = items[index];
    
    // Get the track from timeline
    auto it = timeline.tracks.find(item.entity_name);
    if (it == timeline.tracks.end()) return;
    
    const ObjectAnimationTrack& track = it->second;
    
    // Draw keyframe diamonds
    const float diamond_size = 6.0f;
    const float rc_height = rc.Max.y - rc.Min.y;
    const float center_y = rc.Min.y + rc_height * 0.5f;
    
    // Calculate pixels per frame
    const float rc_width = rc.Max.x - rc.Min.x;
    const int frame_range = frame_max - frame_min;
    const float pixels_per_frame = (frame_range > 0) ? rc_width / frame_range : 1.0f;
    
    // Draw each keyframe as a diamond
    for (const auto& kf : track.keyframes) {
        // Check if this keyframe matches the track type
        bool should_draw = false;
        switch (item.type) {
            case TrackType::Object_Transform: should_draw = kf.has_transform; break;
            case TrackType::Object_Material: should_draw = kf.has_material; break;
            case TrackType::Light: should_draw = kf.has_light; break;
            case TrackType::Camera: should_draw = kf.has_camera; break;
            case TrackType::World: should_draw = kf.has_world; break;
        }
        
        if (!should_draw) continue;
        
        // Calculate screen position
        const float x = rc.Min.x + (kf.frame - frame_min) * pixels_per_frame;
        
        // Check clipping
        if (x < clippingRect.Min.x - diamond_size || x > clippingRect.Max.x + diamond_size) {
            continue;
        }
        
        // Draw diamond (4 points)
        ImVec2 points[4] = {
            ImVec2(x, center_y - diamond_size),      // Top
            ImVec2(x + diamond_size, center_y),      // Right
            ImVec2(x, center_y + diamond_size),      // Bottom
            ImVec2(x - diamond_size, center_y)       // Left
        };
        
        // Filled diamond
        draw_list->AddConvexPolyFilled(points, 4, item.color);
        
        // Diamond outline
        draw_list->AddPolyline(points, 4, IM_COL32(0, 0, 0, 255), ImDrawFlags_Closed, 1.5f);
    }
}
