#include "TimelineWidget.h"
#include "scene_ui.h"
#include "scene_data.h"
#include "Triangle.h"
#include "Light.h"
#include "Camera.h"
#include <chrono>
#include <algorithm>
#include <set>

// ============================================================================
// MAIN DRAW FUNCTION
// ============================================================================
void TimelineWidget::draw(UIContext& ctx) {
    // Rebuild tracks if needed
    if (tracks_dirty) {
        rebuildTrackList(ctx);
        tracks_dirty = false;
    }
    
    // Sync imported animation data
    syncFromAnimationData(ctx);
    
    // Get available region
    ImVec2 region = ImGui::GetContentRegionAvail();
    float total_height = region.y;
    float canvas_height = total_height - header_height - 30.0f; // Reserve space for controls
    
    // --- PLAYBACK CONTROLS ---
    drawPlaybackControls(ctx);
    
    ImGui::Separator();
    
    // --- MAIN TIMELINE AREA ---
    // Split into track list (left) and canvas (right)
    ImGui::BeginChild("TimelineArea", ImVec2(0, canvas_height), false, ImGuiWindowFlags_NoScrollbar);
    
    // Left panel: Track list
    ImGui::BeginChild("TrackList", ImVec2(legend_width, canvas_height), true, ImGuiWindowFlags_NoScrollbar);
    drawTrackList(ctx, legend_width, canvas_height);
    ImGui::EndChild();
    
    ImGui::SameLine();
    
    // Right panel: Timeline canvas
    ImGui::BeginChild("TimelineCanvas", ImVec2(0, canvas_height), true, 
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoMove);
    drawTimelineCanvas(ctx, region.x - legend_width - 20, canvas_height);
    ImGui::EndChild();
    
    ImGui::EndChild();
    
    // --- PLAYBACK UPDATE ---
    if (is_playing) {
        auto now = std::chrono::steady_clock::now();
        static auto last_time = now;
        float elapsed = std::chrono::duration<float>(now - last_time).count();
        float frame_duration = 1.0f / ctx.render_settings.animation_fps;
        
        if (elapsed >= frame_duration) {
            current_frame++;
            if (current_frame > end_frame) current_frame = start_frame;
            last_time = now;
        }
    }
    
    // Sync to render settings
    ctx.render_settings.animation_current_frame = current_frame;
    ctx.render_settings.animation_playback_frame = current_frame;
    ctx.scene.timeline.current_frame = current_frame;
}

// ============================================================================
// PLAYBACK CONTROLS + TOOLBAR
// ============================================================================
void TimelineWidget::drawPlaybackControls(UIContext& ctx) {
    // --- TOOLBAR BUTTONS ---
    bool has_selection = !selected_track.empty();
    bool has_keyframe_selected = has_selection && selected_keyframe_frame >= 0;
    
    // Keyframe buttons
    ImGui::BeginDisabled(!has_selection);
    if (ImGui::Button("+K", ImVec2(30, 0))) {
        insertKeyframeForTrack(ctx, selected_track, current_frame);
        tracks_dirty = true;
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Insert Keyframe (I)");
    ImGui::EndDisabled();
    
    ImGui::SameLine();
    
    ImGui::BeginDisabled(!has_keyframe_selected);
    if (ImGui::Button("-K", ImVec2(30, 0))) {
        deleteKeyframe(ctx, selected_track, selected_keyframe_frame);
        selected_keyframe_frame = -1;
        tracks_dirty = true;
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Delete Keyframe (X)");
    
    ImGui::SameLine();
    
    if (ImGui::Button("Dup", ImVec2(35, 0))) {
        duplicateKeyframe(ctx, selected_track, selected_keyframe_frame, selected_keyframe_frame + 10);
        tracks_dirty = true;
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Duplicate Keyframe (+10 frames)");
    ImGui::EndDisabled();
    
    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();
    
    // --- HELP BUTTON ---
    if (ImGui::Button("?", ImVec2(25, 0))) {
        ImGui::OpenPopup("TimelineHelpPopup");
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Help - Keyboard Shortcuts");
    
    // Help Popup
    if (ImGui::BeginPopup("TimelineHelpPopup")) {
        ImGui::TextColored(ImVec4(1, 0.8f, 0.2f, 1), "Timeline Keyboard Shortcuts");
        ImGui::Separator();
        
        ImGui::TextDisabled("KEYFRAME:");
        ImGui::BulletText("I - Insert keyframe (all L+R+S)");
        ImGui::BulletText("X / Delete - Delete selected keyframe");
        ImGui::BulletText("Left-click - Select keyframe");
        ImGui::BulletText("Drag - Move keyframe");
        ImGui::BulletText("Right-click - Context menu");
        
        ImGui::Separator();
        ImGui::TextDisabled("PER-CHANNEL (Right-click menu):");
        ImGui::BulletText("Location (L) - Position only");
        ImGui::BulletText("Rotation (R) - Rotation only");
        ImGui::BulletText("Scale (S) - Scale only");
        ImGui::BulletText("Expand track - See L/R/S rows");
        
        ImGui::Separator();
        ImGui::TextDisabled("NAVIGATION:");
        ImGui::BulletText("Mouse Wheel - Zoom in/out");
        ImGui::BulletText("Middle Mouse Drag - Pan timeline");
        ImGui::BulletText("Click header - Scrub to frame");
        ImGui::BulletText("Home - Go to start frame");
        ImGui::BulletText("End - Go to end frame");
        
        ImGui::Separator();
        ImGui::TextDisabled("PLAYBACK:");
        ImGui::BulletText("Space - Play/Pause");
        
        ImGui::EndPopup();
    }
    
    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();
    // Frame range
    ImGui::PushItemWidth(80);
    ImGui::InputInt("Start", &start_frame, 0, 0);
    ImGui::SameLine();
    ImGui::InputInt("End", &end_frame, 0, 0);
    ImGui::SameLine();
    ImGui::InputInt("Frame", &current_frame, 0, 0);
    ImGui::PopItemWidth();
    
    current_frame = std::clamp(current_frame, start_frame, end_frame);
    
    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();
    
    // Play/Pause
    if (ImGui::Button(is_playing ? "||" : "|>", ImVec2(30, 0))) {
        is_playing = !is_playing;
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip(is_playing ? "Pause" : "Play");
    
    ImGui::SameLine();
    
    // Stop
    if (ImGui::Button("[]", ImVec2(30, 0))) {
        is_playing = false;
        current_frame = start_frame;
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Stop");
    
    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();
    
    // FPS
    ImGui::PushItemWidth(60);
    ImGui::SliderInt("FPS", &ctx.render_settings.animation_fps, 1, 60);
    ImGui::PopItemWidth();
    
    ImGui::SameLine();
    ImGui::TextDisabled("| Zoom: %.1fx", zoom);
    
    // Show selected track info
    if (!selected_track.empty()) {
        ImGui::SameLine();
        ImGui::TextDisabled("| Track: %s", selected_track.c_str());
    }
}

// ============================================================================
// TRACK LIST (LEFT PANEL) - Simplified with proper TreePop
// ============================================================================
void TimelineWidget::drawTrackList(UIContext& ctx, float list_width, float canvas_height) {
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    
    // Group: Objects
    if (ImGui::TreeNodeEx("Objects", ImGuiTreeNodeFlags_DefaultOpen)) {
        for (size_t i = 0; i < tracks.size(); i++) {
            auto& track = tracks[i];
            if (track.group != TrackGroup::Objects) continue;
            if (track.is_sub_track) continue;  // Skip sub-tracks, handled in main track loop
            
            bool is_selected = (track.entity_name == selected_track);
            
            // Expandable tree node for object
            ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth;
            if (is_selected) flags |= ImGuiTreeNodeFlags_Selected;
            if (track.expanded) flags |= ImGuiTreeNodeFlags_DefaultOpen;
            
            bool node_open = ImGui::TreeNodeEx(track.name.c_str(), flags);
            track.expanded = node_open;
            
            if (ImGui::IsItemClicked(0) && !ImGui::IsItemToggledOpen()) {
                selected_track = track.entity_name;
            }
            
            // Color bar
            ImVec2 item_min = ImGui::GetItemRectMin();
            draw_list->AddRectFilled(
                ImVec2(item_min.x - 12, item_min.y + 2),
                ImVec2(item_min.x - 4, item_min.y + 16),
                track.color, 2.0f);
            
            if (node_open) {
                // Draw sub-tracks (L/R/S)
                for (size_t j = i + 1; j < tracks.size() && j < i + 4; j++) {
                    auto& sub = tracks[j];
                    if (sub.parent_entity != track.entity_name) break;
                    if (!sub.is_sub_track) break;
                    
                    // Create channel-specific track name for selection
                    std::string channel_track = track.entity_name;
                    if (sub.channel == ChannelType::Location) channel_track += ".Location";
                    else if (sub.channel == ChannelType::Rotation) channel_track += ".Rotation";
                    else if (sub.channel == ChannelType::Scale) channel_track += ".Scale";
                    
                    bool sub_sel = (selected_track == channel_track);
                    
                    ImGui::Indent(10);
                    if (ImGui::Selectable(sub.name.c_str(), sub_sel, 0, ImVec2(list_width - 60, 18))) {
                        selected_track = channel_track;  // Set channel-specific selection
                    }
                    
                    ImVec2 sub_pos = ImGui::GetItemRectMin();
                    draw_list->AddRectFilled(
                        ImVec2(sub_pos.x - 8, sub_pos.y + 2),
                        ImVec2(sub_pos.x - 2, sub_pos.y + 16),
                        sub.color, 2.0f);
                    ImGui::Unindent(10);
                }
                ImGui::TreePop();
            }
        }
        ImGui::TreePop();
    }
    
    // Group: Lights
    if (ImGui::TreeNodeEx("Lights", ImGuiTreeNodeFlags_DefaultOpen)) {
        for (auto& track : tracks) {
            if (track.group != TrackGroup::Lights) continue;
            
            bool is_selected = (track.entity_name == selected_track);
            if (ImGui::Selectable(track.name.c_str(), is_selected, 0, ImVec2(list_width - 20, track_height - 4))) {
                selected_track = track.entity_name;
            }
            
            ImVec2 item_min = ImGui::GetItemRectMin();
            draw_list->AddRectFilled(
                ImVec2(item_min.x - 12, item_min.y + 2),
                ImVec2(item_min.x - 4, item_min.y + 16),
                track.color, 2.0f);
        }
        ImGui::TreePop();
    }
    
    // Group: Cameras
    if (ImGui::TreeNodeEx("Cameras", ImGuiTreeNodeFlags_DefaultOpen)) {
        for (auto& track : tracks) {
            if (track.group != TrackGroup::Cameras) continue;
            
            bool is_selected = (track.entity_name == selected_track);
            if (ImGui::Selectable(track.name.c_str(), is_selected, 0, ImVec2(list_width - 20, track_height - 4))) {
                selected_track = track.entity_name;
            }
            
            ImVec2 item_min = ImGui::GetItemRectMin();
            draw_list->AddRectFilled(
                ImVec2(item_min.x - 12, item_min.y + 2),
                ImVec2(item_min.x - 4, item_min.y + 16),
                track.color, 2.0f);
        }
        ImGui::TreePop();
    }
    
    // Group: World
    if (ImGui::TreeNodeEx("World", ImGuiTreeNodeFlags_DefaultOpen)) {
        for (auto& track : tracks) {
            if (track.group != TrackGroup::World) continue;
            
            bool is_selected = (track.entity_name == selected_track);
            if (ImGui::Selectable(track.name.c_str(), is_selected, 0, ImVec2(list_width - 20, track_height - 4))) {
                selected_track = track.entity_name;
            }
            
            ImVec2 item_min = ImGui::GetItemRectMin();
            draw_list->AddRectFilled(
                ImVec2(item_min.x - 12, item_min.y + 2),
                ImVec2(item_min.x - 4, item_min.y + 16),
                track.color, 2.0f);
        }
        ImGui::TreePop();
    }
}

// ============================================================================
// TIMELINE CANVAS (RIGHT PANEL)
// ============================================================================
void TimelineWidget::drawTimelineCanvas(UIContext& ctx, float canvas_width, float canvas_height) {
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
    ImVec2 canvas_size(canvas_width, canvas_height);
    
    // Background
    draw_list->AddRectFilled(canvas_pos, 
        ImVec2(canvas_pos.x + canvas_width, canvas_pos.y + canvas_height),
        IM_COL32(30, 30, 35, 255));
    
    // Handle input
    handleZoomPan(canvas_pos, canvas_size);
    handleScrubbing(canvas_pos, canvas_width);
    
    // Draw frame numbers at top
    drawFrameNumbers(draw_list, canvas_pos, canvas_width);
    
    // Draw grid lines
    int frame_step = std::max(1, (int)(10 / zoom));
    for (int f = start_frame; f <= end_frame; f += frame_step) {
        int px = frameToPixelX(f, canvas_width);
        if (px >= 0 && px <= canvas_width) {
            draw_list->AddLine(
                ImVec2(canvas_pos.x + px, canvas_pos.y + header_height),
                ImVec2(canvas_pos.x + px, canvas_pos.y + canvas_height),
                COLOR_GRID, 1.0f);
        }
    }
    
    // Draw keyframes for each track
    float y_offset = header_height;
    bool any_keyframe_hovered = false;
    int hovered_keyframe_frame = -1;
    std::string hovered_track;
    
    for (auto& track : tracks) {
        // Skip sub-tracks in the main loop - they're drawn with parent
        if (track.is_sub_track) {
            if (track.expanded) y_offset += track_height;  // Only if parent expanded
            continue;
        }
        
        // Find keyframes for this track
        auto it = ctx.scene.timeline.tracks.find(track.entity_name);
        if (it != ctx.scene.timeline.tracks.end()) {
            for (auto& kf : it->second.keyframes) {
                int px = frameToPixelX(kf.frame, canvas_width);
                if (px >= -10 && px <= canvas_width + 10) {
                    float x = canvas_pos.x + px;
                    float base_y = canvas_pos.y + y_offset + track_height / 2;
                    
                    // For transform keyframes, draw per-channel diamonds
                    if (kf.has_transform && track.group == TrackGroup::Objects) {
                        // If parent is expanded, draw on sub-track rows
                        if (track.expanded) {
                            float sub_row_height = 20.0f;
                            
                            // Location row
                            if (kf.transform.has_position) {
                                float y_loc = base_y + sub_row_height;
                                std::string loc_track = track.entity_name + ".Location";
                                bool is_sel = (loc_track == selected_track && kf.frame == selected_keyframe_frame);
                                bool is_hov = ImGui::IsMouseHoveringRect(
                                    ImVec2(x - 5, y_loc - 5), ImVec2(x + 5, y_loc + 5));
                                
                                drawKeyframeDiamond(draw_list, x, y_loc,
                                    is_hov ? IM_COL32(255, 200, 200, 255) : IM_COL32(255, 100, 100, 255), is_sel);
                                
                                if (ImGui::IsMouseClicked(0) && is_hov) {
                                    selected_track = loc_track;
                                    selected_keyframe_frame = kf.frame;
                                    is_dragging_keyframe = true;
                                    drag_start_frame = kf.frame;
                                }
                            }
                            
                            // Rotation row
                            if (kf.transform.has_rotation) {
                                float y_rot = base_y + sub_row_height * 2;
                                std::string rot_track = track.entity_name + ".Rotation";
                                bool is_sel = (rot_track == selected_track && kf.frame == selected_keyframe_frame);
                                bool is_hov = ImGui::IsMouseHoveringRect(
                                    ImVec2(x - 5, y_rot - 5), ImVec2(x + 5, y_rot + 5));
                                
                                drawKeyframeDiamond(draw_list, x, y_rot,
                                    is_hov ? IM_COL32(200, 255, 200, 255) : IM_COL32(100, 255, 100, 255), is_sel);
                                
                                if (ImGui::IsMouseClicked(0) && is_hov) {
                                    selected_track = rot_track;
                                    selected_keyframe_frame = kf.frame;
                                    is_dragging_keyframe = true;
                                    drag_start_frame = kf.frame;
                                }
                            }
                            
                            // Scale row
                            if (kf.transform.has_scale) {
                                float y_scl = base_y + sub_row_height * 3;
                                std::string scl_track = track.entity_name + ".Scale";
                                bool is_sel = (scl_track == selected_track && kf.frame == selected_keyframe_frame);
                                bool is_hov = ImGui::IsMouseHoveringRect(
                                    ImVec2(x - 5, y_scl - 5), ImVec2(x + 5, y_scl + 5));
                                
                                drawKeyframeDiamond(draw_list, x, y_scl,
                                    is_hov ? IM_COL32(200, 200, 255, 255) : IM_COL32(100, 100, 255, 255), is_sel);
                                
                                if (ImGui::IsMouseClicked(0) && is_hov) {
                                    selected_track = scl_track;
                                    selected_keyframe_frame = kf.frame;
                                    is_dragging_keyframe = true;
                                    drag_start_frame = kf.frame;
                                }
                            }
                        } else {
                            // Collapsed: draw combined diamond on main row
                            bool is_selected = (track.entity_name == selected_track && kf.frame == selected_keyframe_frame);
                            bool is_hov = ImGui::IsMouseHoveringRect(
                                ImVec2(x - 6, base_y - 6), ImVec2(x + 6, base_y + 6));
                            
                            if (is_hov) {
                                any_keyframe_hovered = true;
                                hovered_keyframe_frame = kf.frame;
                                hovered_track = track.entity_name;
                            }
                            
                            drawKeyframeDiamond(draw_list, x, base_y,
                                is_hov ? IM_COL32(255, 255, 200, 255) : track.color, is_selected);
                            
                            if (ImGui::IsMouseClicked(0) && is_hov) {
                                selected_track = track.entity_name;
                                selected_keyframe_frame = kf.frame;
                                is_dragging_keyframe = true;
                                drag_start_frame = kf.frame;
                            }
                        }
                    } else {
                        // Non-transform keyframes (light, camera, world)
                        float y = base_y;
                        bool is_selected = (track.entity_name == selected_track && kf.frame == selected_keyframe_frame);
                        bool is_hov = ImGui::IsMouseHoveringRect(
                            ImVec2(x - 6, y - 6), ImVec2(x + 6, y + 6));
                        
                        if (is_hov) {
                            any_keyframe_hovered = true;
                            hovered_keyframe_frame = kf.frame;
                            hovered_track = track.entity_name;
                        }
                        
                        drawKeyframeDiamond(draw_list, x, y,
                            is_hov ? IM_COL32(255, 255, 200, 255) : track.color, is_selected);
                        
                        if (ImGui::IsMouseClicked(0) && is_hov) {
                            selected_track = track.entity_name;
                            selected_keyframe_frame = kf.frame;
                            is_dragging_keyframe = true;
                            drag_start_frame = kf.frame;
                        }
                    }
                }
            }
        }
        
        // Add height for main track + sub-tracks if expanded
        y_offset += track_height;
        if (track.group == TrackGroup::Objects && track.expanded) {
            y_offset += track_height * 3;  // L/R/S sub-rows
        }
    }
    
    // Handle keyframe dragging
    if (is_dragging_keyframe && ImGui::IsMouseDown(0)) {
        ImGuiIO& io = ImGui::GetIO();
        if (io.MousePos.x >= canvas_pos.x && io.MousePos.x <= canvas_pos.x + canvas_width) {
            int new_frame = pixelXToFrame(io.MousePos.x - canvas_pos.x, canvas_width);
            new_frame = std::clamp(new_frame, start_frame, end_frame);
            
            // Visual feedback - draw where keyframe would move
            if (new_frame != selected_keyframe_frame) {
                float new_x = canvas_pos.x + frameToPixelX(new_frame, canvas_width);
                draw_list->AddLine(
                    ImVec2(new_x, canvas_pos.y + header_height),
                    ImVec2(new_x, canvas_pos.y + canvas_height),
                    IM_COL32(255, 200, 100, 150), 2.0f);
            }
        }
    }
    
    // End keyframe drag
    if (is_dragging_keyframe && ImGui::IsMouseReleased(0)) {
        ImGuiIO& io = ImGui::GetIO();
        int new_frame = pixelXToFrame(io.MousePos.x - canvas_pos.x, canvas_width);
        new_frame = std::clamp(new_frame, start_frame, end_frame);
        
        if (new_frame != drag_start_frame && !selected_track.empty()) {
            // Move keyframe to new position
            moveKeyframe(ctx, selected_track, drag_start_frame, new_frame);
            selected_keyframe_frame = new_frame;
            tracks_dirty = true;
        }
        is_dragging_keyframe = false;
    }
    
    // Draw current frame indicator
    drawCurrentFrameIndicator(draw_list, canvas_pos, canvas_height);
    
    // Invisible button to capture mouse events
    ImGui::SetCursorScreenPos(canvas_pos);
    ImGui::InvisibleButton("##TimelineCanvas", canvas_size);
    
    // Right-click on empty area for insert
    if (ImGui::IsItemClicked(1) && !any_keyframe_hovered) {
        context_menu_frame = pixelXToFrame(ImGui::GetIO().MousePos.x - canvas_pos.x, canvas_width);
        ImGui::OpenPopup("TimelineContextMenu");
    }
    
    // --- CONTEXT MENUS ---
    // Keyframe context menu (right-click on keyframe)
    if (ImGui::BeginPopup("KeyframeContextMenu")) {
        ImGui::TextDisabled("Keyframe @ Frame %d", selected_keyframe_frame);
        ImGui::TextDisabled("Track: %s", selected_track.c_str());
        ImGui::Separator();
        
        // Show keyframe info
        auto it = ctx.scene.timeline.tracks.find(selected_track);
        if (it != ctx.scene.timeline.tracks.end()) {
            for (auto& kf : it->second.keyframes) {
                if (kf.frame == selected_keyframe_frame) {
                    ImGui::TextDisabled("Contains:");
                    if (kf.has_transform) ImGui::BulletText("Transform (L+R+S)");
                    if (kf.has_material) ImGui::BulletText("Material");
                    if (kf.has_light) ImGui::BulletText("Light");
                    if (kf.has_camera) ImGui::BulletText("Camera");
                    if (kf.has_world) ImGui::BulletText("World");
                    ImGui::Separator();
                    break;
                }
            }
        }
        
        if (ImGui::MenuItem("Delete Keyframe", "X")) {
            deleteKeyframe(ctx, selected_track, selected_keyframe_frame);
            selected_keyframe_frame = -1;
            tracks_dirty = true;
        }
        if (ImGui::MenuItem("Duplicate Keyframe")) {
            duplicateKeyframe(ctx, selected_track, selected_keyframe_frame, selected_keyframe_frame + 10);
            tracks_dirty = true;
        }
        ImGui::EndPopup();
    }
    
    // Timeline context menu (right-click on empty area)
    if (ImGui::BeginPopup("TimelineContextMenu")) {
        ImGui::TextDisabled("Frame %d", context_menu_frame);
        if (!selected_track.empty()) {
            ImGui::TextDisabled("Track: %s", selected_track.c_str());
        }
        ImGui::Separator();
        
        // Insert sub-menu with separate L/R/S options
        if (ImGui::BeginMenu("Insert Keyframe", !selected_track.empty())) {
            if (ImGui::MenuItem("Location (L)", "Shift+L")) {
                current_frame = context_menu_frame;
                insertKeyframeType(ctx, selected_track, context_menu_frame, KeyframeInsertType::Location);
                tracks_dirty = true;
            }
            if (ImGui::MenuItem("Rotation (R)", "Shift+R")) {
                current_frame = context_menu_frame;
                insertKeyframeType(ctx, selected_track, context_menu_frame, KeyframeInsertType::Rotation);
                tracks_dirty = true;
            }
            if (ImGui::MenuItem("Scale (S)", "Shift+S")) {
                current_frame = context_menu_frame;
                insertKeyframeType(ctx, selected_track, context_menu_frame, KeyframeInsertType::Scale);
                tracks_dirty = true;
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Location + Rotation", nullptr)) {
                current_frame = context_menu_frame;
                insertKeyframeType(ctx, selected_track, context_menu_frame, KeyframeInsertType::LocRot);
                tracks_dirty = true;
            }
            if (ImGui::MenuItem("All (Location + Rotation + Scale)", "I")) {
                current_frame = context_menu_frame;
                insertKeyframeForTrack(ctx, selected_track, context_menu_frame);
                tracks_dirty = true;
            }
            ImGui::EndMenu();
        }
        
        ImGui::Separator();
        if (ImGui::MenuItem("Go to Frame")) {
            current_frame = context_menu_frame;
        }
        ImGui::EndPopup();
    }
}

// ============================================================================
// FRAME NUMBERS AT TOP
// ============================================================================
void TimelineWidget::drawFrameNumbers(ImDrawList* draw_list, ImVec2 canvas_pos, float canvas_width) {
    int frame_step = std::max(1, (int)(20 / zoom));
    
    for (int f = start_frame; f <= end_frame; f += frame_step) {
        int px = frameToPixelX(f, canvas_width);
        if (px >= 0 && px <= canvas_width) {
            char buf[16];
            snprintf(buf, sizeof(buf), "%d", f);
            draw_list->AddText(
                ImVec2(canvas_pos.x + px + 2, canvas_pos.y + 4),
                IM_COL32(180, 180, 180, 255), buf);
            
            // Tick mark
            draw_list->AddLine(
                ImVec2(canvas_pos.x + px, canvas_pos.y + header_height - 6),
                ImVec2(canvas_pos.x + px, canvas_pos.y + header_height),
                IM_COL32(100, 100, 100, 255), 1.0f);
        }
    }
}

// ============================================================================
// KEYFRAME DIAMOND
// ============================================================================
void TimelineWidget::drawKeyframeDiamond(ImDrawList* draw_list, float x, float y, ImU32 color, bool selected) {
    const float size = 5.0f;
    
    ImVec2 points[4] = {
        ImVec2(x, y - size),      // Top
        ImVec2(x + size, y),      // Right
        ImVec2(x, y + size),      // Bottom
        ImVec2(x - size, y)       // Left
    };
    
    draw_list->AddConvexPolyFilled(points, 4, color);
    draw_list->AddPolyline(points, 4, selected ? COLOR_SELECTED : IM_COL32(0, 0, 0, 255), 
        ImDrawFlags_Closed, selected ? 2.0f : 1.0f);
}

// ============================================================================
// CURRENT FRAME INDICATOR
// ============================================================================
void TimelineWidget::drawCurrentFrameIndicator(ImDrawList* draw_list, ImVec2 canvas_pos, float canvas_height) {
    int px = frameToPixelX(current_frame, canvas_pos.x);
    float x = canvas_pos.x + frameToPixelX(current_frame, ImGui::GetContentRegionAvail().x);
    
    // Vertical line
    draw_list->AddLine(
        ImVec2(x, canvas_pos.y),
        ImVec2(x, canvas_pos.y + canvas_height),
        COLOR_CURRENT_FRAME, 2.0f);
    
    // Triangle at top
    draw_list->AddTriangleFilled(
        ImVec2(x - 6, canvas_pos.y),
        ImVec2(x + 6, canvas_pos.y),
        ImVec2(x, canvas_pos.y + 10),
        COLOR_CURRENT_FRAME);
}

// ============================================================================
// ZOOM/PAN HANDLING
// ============================================================================
void TimelineWidget::handleZoomPan(ImVec2 canvas_pos, ImVec2 canvas_size) {
    ImGuiIO& io = ImGui::GetIO();
    
    // Check if mouse is over canvas
    bool hovered = ImGui::IsMouseHoveringRect(canvas_pos, 
        ImVec2(canvas_pos.x + canvas_size.x, canvas_pos.y + canvas_size.y));
    
    if (hovered) {
        // Zoom with mouse wheel
        if (io.MouseWheel != 0) {
            float zoom_factor = (io.MouseWheel > 0) ? 1.1f : 0.9f;
            zoom = std::clamp(zoom * zoom_factor, 0.1f, 10.0f);
        }
        
        // Pan with middle mouse button
        if (io.MouseDown[2]) {
            pan_offset -= io.MouseDelta.x / (zoom * 10.0f);
            pan_offset = std::clamp(pan_offset, 0.0f, (float)(end_frame - start_frame));
        }
    }
}

// ============================================================================
// SCRUBBING (CLICK TO SET FRAME)
// ============================================================================
void TimelineWidget::handleScrubbing(ImVec2 canvas_pos, float canvas_width) {
    ImGuiIO& io = ImGui::GetIO();
    
    // Left click in header area to scrub
    if (io.MouseDown[0]) {
        ImVec2 mouse = io.MousePos;
        if (mouse.x >= canvas_pos.x && mouse.x <= canvas_pos.x + canvas_width &&
            mouse.y >= canvas_pos.y && mouse.y <= canvas_pos.y + header_height) {
            current_frame = pixelXToFrame(mouse.x - canvas_pos.x, canvas_width);
            current_frame = std::clamp(current_frame, start_frame, end_frame);
        }
    }
}

// ============================================================================
// FRAME <-> PIXEL CONVERSION
// ============================================================================
int TimelineWidget::frameToPixelX(int frame, float canvas_width) const {
    float frame_range = end_frame - start_frame;
    if (frame_range <= 0) return 0;
    
    float t = (frame - start_frame - pan_offset) / frame_range;
    return (int)(t * canvas_width * zoom);
}

int TimelineWidget::pixelXToFrame(float x, float canvas_width) const {
    float frame_range = end_frame - start_frame;
    if (canvas_width <= 0 || zoom <= 0) return start_frame;
    
    float t = x / (canvas_width * zoom);
    return start_frame + (int)(t * frame_range + pan_offset);
}

// ============================================================================
// REBUILD TRACK LIST - Shows entities with keyframes + currently selected entity
// ============================================================================
void TimelineWidget::rebuildTrackList(UIContext& ctx) {
    tracks.clear();
    
    // Get currently selected entity name from viewport
    std::string selected_entity;
    TrackGroup selected_group = TrackGroup::Objects;
    
    if (ctx.selection.hasSelection()) {
        if (ctx.selection.selected.type == SelectableType::Object && ctx.selection.selected.object) {
            selected_entity = ctx.selection.selected.object->nodeName;
            if (selected_entity.empty()) {
                selected_entity = "Object_" + std::to_string(ctx.selection.selected.object_index);
            }
            selected_group = TrackGroup::Objects;
        } else if (ctx.selection.selected.type == SelectableType::Light && ctx.selection.selected.light) {
            selected_entity = ctx.selection.selected.light->nodeName;
            selected_group = TrackGroup::Lights;
        } else if (ctx.selection.selected.type == SelectableType::Camera && ctx.selection.selected.camera) {
            selected_entity = ctx.selection.selected.camera->nodeName;
            selected_group = TrackGroup::Cameras;
        }
    }
    
    std::set<std::string> added_entities;
    
    // Helper lambda to add Object track with L/R/S sub-tracks
    auto addObjectWithChannels = [&](const std::string& entity_name, const std::string& display_name, 
                                      ImU32 color, bool is_selected, const std::vector<int>& keyframes) {
        // Main object track
        TimelineTrack main_track;
        main_track.entity_name = entity_name;
        main_track.name = display_name;
        main_track.group = TrackGroup::Objects;
        main_track.channel = ChannelType::None;
        main_track.color = color;
        main_track.expanded = is_selected || !keyframes.empty();  // Auto-expand if selected or has keyframes
        main_track.is_sub_track = false;
        main_track.depth = 0;
        main_track.keyframe_frames = keyframes;
        tracks.push_back(main_track);
        
        // Add L/R/S sub-tracks (only if main track is for objects with transforms)
        // Location
        TimelineTrack loc_track;
        loc_track.entity_name = entity_name;  // Same entity for keyframe lookup
        loc_track.parent_entity = entity_name;
        loc_track.name = "Location";
        loc_track.group = TrackGroup::Objects;
        loc_track.channel = ChannelType::Location;
        loc_track.color = IM_COL32(255, 100, 100, 255);  // Red
        loc_track.is_sub_track = true;
        loc_track.depth = 1;
        loc_track.keyframe_frames = keyframes;
        tracks.push_back(loc_track);
        
        // Rotation
        TimelineTrack rot_track;
        rot_track.entity_name = entity_name;
        rot_track.parent_entity = entity_name;
        rot_track.name = "Rotation";
        rot_track.group = TrackGroup::Objects;
        rot_track.channel = ChannelType::Rotation;
        rot_track.color = IM_COL32(100, 255, 100, 255);  // Green
        rot_track.is_sub_track = true;
        rot_track.depth = 1;
        rot_track.keyframe_frames = keyframes;
        tracks.push_back(rot_track);
        
        // Scale
        TimelineTrack scl_track;
        scl_track.entity_name = entity_name;
        scl_track.parent_entity = entity_name;
        scl_track.name = "Scale";
        scl_track.group = TrackGroup::Objects;
        scl_track.channel = ChannelType::Scale;
        scl_track.color = IM_COL32(100, 100, 255, 255);  // Blue
        scl_track.is_sub_track = true;
        scl_track.depth = 1;
        scl_track.keyframe_frames = keyframes;
        tracks.push_back(scl_track);
    };
    
    // --- ADD TRACKS FROM TIMELINE (entities with keyframes) ---
    for (auto& [entity_name, track] : ctx.scene.timeline.tracks) {
        if (track.keyframes.empty()) continue;
        
        // Determine group based on keyframe types
        bool has_transform = false, has_material = false, has_light = false, has_camera = false, has_world = false;
        std::vector<int> keyframes;
        
        for (auto& kf : track.keyframes) {
            has_transform |= kf.has_transform;
            has_material |= kf.has_material;
            has_light |= kf.has_light;
            has_camera |= kf.has_camera;
            has_world |= kf.has_world;
            keyframes.push_back(kf.frame);
        }
        
        if (has_world) {
            TimelineTrack t;
            t.entity_name = entity_name;
            t.name = entity_name;
            t.group = TrackGroup::World;
            t.color = COLOR_WORLD;
            t.keyframe_frames = keyframes;
            tracks.push_back(t);
        } else if (has_camera) {
            TimelineTrack t;
            t.entity_name = entity_name;
            t.name = entity_name;
            t.group = TrackGroup::Cameras;
            t.color = COLOR_CAMERA;
            t.keyframe_frames = keyframes;
            tracks.push_back(t);
        } else if (has_light) {
            TimelineTrack t;
            t.entity_name = entity_name;
            t.name = entity_name;
            t.group = TrackGroup::Lights;
            t.color = COLOR_LIGHT;
            t.keyframe_frames = keyframes;
            tracks.push_back(t);
        } else if (has_transform || has_material) {
            // Object with L/R/S sub-tracks
            bool is_selected = (entity_name == selected_entity);
            ImU32 color = has_material ? COLOR_MATERIAL : COLOR_TRANSFORM;
            addObjectWithChannels(entity_name, entity_name, color, is_selected, keyframes);
        }
        
        added_entities.insert(entity_name);
    }
    
    // --- ADD CURRENTLY SELECTED ENTITY (if not already added) ---
    if (!selected_entity.empty() && added_entities.find(selected_entity) == added_entities.end()) {
        if (selected_group == TrackGroup::Objects) {
            // Object with L/R/S sub-tracks
            addObjectWithChannels(selected_entity, selected_entity + " (selected)", COLOR_TRANSFORM, true, {});
        } else {
            TimelineTrack t;
            t.entity_name = selected_entity;
            t.name = selected_entity + " (selected)";
            t.group = selected_group;
            
            switch (selected_group) {
                case TrackGroup::Lights: t.color = COLOR_LIGHT; break;
                case TrackGroup::Cameras: t.color = COLOR_CAMERA; break;
                case TrackGroup::World: t.color = COLOR_WORLD; break;
                default: t.color = COLOR_TRANSFORM; break;
            }
            
            tracks.push_back(t);
        }
    }
    
    // --- ADD WORLD TRACK (always present) ---
    if (added_entities.find("World") == added_entities.end()) {
        TimelineTrack t;
        t.entity_name = "World";
        t.name = "World";
        t.group = TrackGroup::World;
        t.color = COLOR_WORLD;
        tracks.push_back(t);
    }
}

// ============================================================================
// SYNC FROM IMPORTED ANIMATION DATA + KEYBOARD SHORTCUTS
// ============================================================================
void TimelineWidget::syncFromAnimationData(UIContext& ctx) {
    // Update frame range from animation data if available
    if (!ctx.scene.animationDataList.empty()) {
        auto& anim = ctx.scene.animationDataList[0];
        if (anim.startFrame != start_frame || anim.endFrame != end_frame) {
            start_frame = anim.startFrame;
            end_frame = anim.endFrame;
        }
    }
    
    // --- SYNC SELECTION FROM VIEWPORT ---
    // This ensures I key works on the viewport selected object
    std::string viewport_selection;
    if (ctx.selection.hasSelection()) {
        if (ctx.selection.selected.type == SelectableType::Object && ctx.selection.selected.object) {
            viewport_selection = ctx.selection.selected.object->nodeName;
            if (viewport_selection.empty()) {
                viewport_selection = "Object_" + std::to_string(ctx.selection.selected.object_index);
            }
        } else if (ctx.selection.selected.type == SelectableType::Light && ctx.selection.selected.light) {
            viewport_selection = ctx.selection.selected.light->nodeName;
        } else if (ctx.selection.selected.type == SelectableType::Camera && ctx.selection.selected.camera) {
            viewport_selection = ctx.selection.selected.camera->nodeName;
        }
    }
    
    // Auto-select the viewport selection in timeline  
    if (!viewport_selection.empty()) {
        selected_track = viewport_selection;
    }
    
    // --- KEYBOARD SHORTCUTS ---
    ImGuiIO& io = ImGui::GetIO();
    
    // I key - Insert keyframe for selected track (viewport selection)
    if (ImGui::IsKeyPressed(ImGuiKey_I) && !io.WantTextInput) {
        if (!selected_track.empty()) {
            insertKeyframeForTrack(ctx, selected_track, current_frame);
            tracks_dirty = true;
        }
    }
    
    // Delete/X key - Delete selected keyframe
    if ((ImGui::IsKeyPressed(ImGuiKey_Delete) || ImGui::IsKeyPressed(ImGuiKey_X)) && !io.WantTextInput) {
        if (!selected_track.empty() && selected_keyframe_frame >= 0) {
            deleteKeyframe(ctx, selected_track, selected_keyframe_frame);
            selected_keyframe_frame = -1;
            tracks_dirty = true;
        }
    }
    
    // Mark tracks dirty when selection changes
    static std::string last_selection;
    if (viewport_selection != last_selection) {
        last_selection = viewport_selection;
        tracks_dirty = true;
    }
    
    // Periodic rebuild to catch scene changes
    static int rebuild_counter = 0;
    if (++rebuild_counter > 60) {
        rebuild_counter = 0;
        tracks_dirty = true;
    }
}

// ============================================================================
// INSERT KEYFRAME FOR TRACK - Uses selection data like existing code
// ============================================================================
void TimelineWidget::insertKeyframeForTrack(UIContext& ctx, const std::string& track_name, int frame) {
    if (!ctx.selection.hasSelection()) return;
    
    Keyframe kf(frame);
    std::string entity_name;
    bool has_data = false;
    
    // Use selection directly - same as existing scene_ui.cpp code
    if (ctx.selection.selected.type == SelectableType::Object && ctx.selection.selected.object) {
        auto& obj = ctx.selection.selected.object;
        entity_name = obj->nodeName.empty() ? 
            "Object_" + std::to_string(ctx.selection.selected.object_index) : 
            obj->nodeName;
        
        if (entity_name == track_name) {
            kf.has_transform = true;
            // Transform stored in SelectableItem, not in object!
            kf.transform.position = ctx.selection.selected.position;
            kf.transform.rotation = ctx.selection.selected.rotation;
            kf.transform.scale = ctx.selection.selected.scale;
            // Set all channel flags (full keyframe)
            kf.transform.has_position = true;
            kf.transform.has_rotation = true;
            kf.transform.has_scale = true;
            has_data = true;
        }
    } 
    else if (ctx.selection.selected.type == SelectableType::Light && ctx.selection.selected.light) {
        auto& light = ctx.selection.selected.light;
        entity_name = light->nodeName;
        
        if (entity_name == track_name) {
            kf.has_light = true;
            kf.light.position = light->position;
            kf.light.color = light->color;
            kf.light.intensity = light->intensity;
            has_data = true;
        }
    }
    else if (ctx.selection.selected.type == SelectableType::Camera && ctx.selection.selected.camera) {
        auto& cam = ctx.selection.selected.camera;
        entity_name = cam->nodeName;
        
        if (entity_name == track_name) {
            kf.has_camera = true;
            kf.camera.position = cam->lookfrom;
            kf.camera.target = cam->lookat;
            kf.camera.fov = cam->vfov;
            kf.camera.focus_distance = cam->focus_dist;
            kf.camera.lens_radius = cam->lens_radius;
            has_data = true;
        }
    }
    else if (track_name == "World") {
        kf.has_world = true;
        kf.world.background_color = ctx.scene.background_color;
        entity_name = "World";
        has_data = true;
    }
    
    // Insert keyframe
    if (has_data && !entity_name.empty()) {
        ctx.scene.timeline.insertKeyframe(entity_name, kf);
    }
}

// ============================================================================
// DELETE KEYFRAME
// ============================================================================
void TimelineWidget::deleteKeyframe(UIContext& ctx, const std::string& track_name, int frame) {
    auto it = ctx.scene.timeline.tracks.find(track_name);
    if (it != ctx.scene.timeline.tracks.end()) {
        auto& keyframes = it->second.keyframes;
        keyframes.erase(
            std::remove_if(keyframes.begin(), keyframes.end(),
                [frame](const Keyframe& kf) { return kf.frame == frame; }),
            keyframes.end());
    }
}

// ============================================================================
// MOVE KEYFRAME
// ============================================================================
void TimelineWidget::moveKeyframe(UIContext& ctx, const std::string& track_name, int old_frame, int new_frame) {
    auto it = ctx.scene.timeline.tracks.find(track_name);
    if (it != ctx.scene.timeline.tracks.end()) {
        for (auto& kf : it->second.keyframes) {
            if (kf.frame == old_frame) {
                kf.frame = new_frame;
                break;
            }
        }
        // Re-sort keyframes by frame
        std::sort(it->second.keyframes.begin(), it->second.keyframes.end(),
            [](const Keyframe& a, const Keyframe& b) { return a.frame < b.frame; });
    }
}

// ============================================================================
// DUPLICATE KEYFRAME
// ============================================================================
void TimelineWidget::duplicateKeyframe(UIContext& ctx, const std::string& track_name, int src_frame, int dst_frame) {
    auto it = ctx.scene.timeline.tracks.find(track_name);
    if (it != ctx.scene.timeline.tracks.end()) {
        for (auto& kf : it->second.keyframes) {
            if (kf.frame == src_frame) {
                Keyframe new_kf = kf;  // Copy all data
                new_kf.frame = dst_frame;
                it->second.keyframes.push_back(new_kf);
                
                // Re-sort keyframes by frame
                std::sort(it->second.keyframes.begin(), it->second.keyframes.end(),
                    [](const Keyframe& a, const Keyframe& b) { return a.frame < b.frame; });
                break;
            }
        }
    }
}

// ============================================================================
// INSERT KEYFRAME TYPE (Separate L/R/S options with per-channel flags)
// ============================================================================
void TimelineWidget::insertKeyframeType(UIContext& ctx, const std::string& track_name, int frame, KeyframeInsertType type) {
    if (!ctx.selection.hasSelection()) return;
    
    // Only works for objects (transform data)
    if (ctx.selection.selected.type != SelectableType::Object || !ctx.selection.selected.object) {
        // For non-objects, fall back to full keyframe
        insertKeyframeForTrack(ctx, track_name, frame);
        return;
    }
    
    auto& obj = ctx.selection.selected.object;
    std::string entity_name = obj->nodeName.empty() ? 
        "Object_" + std::to_string(ctx.selection.selected.object_index) : 
        obj->nodeName;
    
    if (entity_name != track_name) return;
    
    // Get current transform from selection
    Vec3 pos = ctx.selection.selected.position;
    Vec3 rot = ctx.selection.selected.rotation;
    Vec3 scl = ctx.selection.selected.scale;
    
    Keyframe kf(frame);
    kf.has_transform = true;
    
    // ALWAYS store current transform values for ALL channels
    // This prevents values from resetting to defaults
    kf.transform.position = pos;
    kf.transform.rotation = rot;
    kf.transform.scale = scl;
    
    // Initialize flags from existing keyframe if available
    kf.transform.has_position = false;
    kf.transform.has_rotation = false;
    kf.transform.has_scale = false;
    
    // Check if there's an existing keyframe at this frame - merge flags
    auto it = ctx.scene.timeline.tracks.find(entity_name);
    if (it != ctx.scene.timeline.tracks.end()) {
        for (auto& existing : it->second.keyframes) {
            if (existing.frame == frame && existing.has_transform) {
                // Preserve existing keyed flags
                kf.transform.has_position = existing.transform.has_position;
                kf.transform.has_rotation = existing.transform.has_rotation;
                kf.transform.has_scale = existing.transform.has_scale;
                // Remove old keyframe, will be replaced
                deleteKeyframe(ctx, entity_name, frame);
                break;
            }
        }
    }
    
    // Set flags based on insert type (ADD to existing flags, don't replace)
    switch (type) {
        case KeyframeInsertType::Location:
            kf.transform.has_position = true;
            break;
        case KeyframeInsertType::Rotation:
            kf.transform.has_rotation = true;
            break;
        case KeyframeInsertType::Scale:
            kf.transform.has_scale = true;
            break;
        case KeyframeInsertType::LocRot:
            kf.transform.has_position = true;
            kf.transform.has_rotation = true;
            break;
        case KeyframeInsertType::All:
        default:
            kf.transform.has_position = true;
            kf.transform.has_rotation = true;
            kf.transform.has_scale = true;
            break;
    }
    
    ctx.scene.timeline.insertKeyframe(entity_name, kf);
}
