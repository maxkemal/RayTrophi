#include "TimelineWidget.h"
#include "scene_ui.h"
#include "scene_data.h"
#include "Triangle.h"
#include "HittableInstance.h"
#include "Light.h"
#include "Camera.h"
#include <chrono>
#include <algorithm>
#include <set>
#include <map>
#include <unordered_map>
#include <cmath>
#include <limits>
#include <cstring>
#include "SceneSelection.h"
#include "renderer.h"
#include "TerrainManager.h"
#include "WaterSystem.h"  // For water keyframe animation
#include "GasVolume.h"    // For gas/emitter keyframe animation
#include "globals.h"
#include "MaterialManager.h"
#include "PrincipledBSDF.h"
#include "Matrix4x4.h"
#include "Backend/IViewportBackend.h"
#include "world.h"
#include "OptixWrapper.h"  // For direct instance transform updates
#include "ui_modern.h"

#include <thread>

extern bool g_timeline_selection_sync_pending;

// External rendering state for animation render sync
extern std::atomic<bool> rendering_in_progress;
extern std::atomic<bool> rendering_stopped_cpu;
extern std::atomic<bool> rendering_stopped_gpu;
extern std::atomic<bool> rendering_paused;
extern std::unique_ptr<Backend::IBackend> g_backend;
extern std::unique_ptr<Backend::IViewportBackend> g_viewport_backend;
namespace {
void drainTimelineMutationBackends(UIContext& ctx) {
    int wait_count = 0;
    while (rendering_in_progress.load() && wait_count < 200) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        ++wait_count;
    }

    Backend::IBackend* renderBackend = g_backend
        ? g_backend.get()
        : ((ctx.backend_ptr && dynamic_cast<Backend::IViewportBackend*>(ctx.backend_ptr) == nullptr)
            ? ctx.backend_ptr
            : nullptr);
    Backend::IViewportBackend* viewportBackend = g_viewport_backend
        ? g_viewport_backend.get()
        : dynamic_cast<Backend::IViewportBackend*>(ctx.backend_ptr);

    if (renderBackend) {
        renderBackend->waitForCompletion();
    }
    if (viewportBackend && static_cast<Backend::IBackend*>(viewportBackend) != renderBackend) {
        viewportBackend->waitForCompletion();
    }
}
}

// Helper to parse entity name and channel from track name
enum class GraphTrackType {
    Transform,
    Light,
    Camera,
    Material
};

static GraphTrackType getGraphTrackType(UIContext& ctx, const std::string& selected_track) {
    size_t dot_pos = selected_track.find('.');
    std::string entity_name = (dot_pos == std::string::npos) ? selected_track : selected_track.substr(0, dot_pos);
    std::string suffix = (dot_pos == std::string::npos) ? "" : selected_track.substr(dot_pos + 1);

    auto track_it = ctx.scene.timeline.tracks.find(entity_name);
    if (track_it != ctx.scene.timeline.tracks.end() && !track_it->second.keyframes.empty()) {
        const auto& kf = track_it->second.keyframes[0];
        if (kf.has_light) return GraphTrackType::Light;
        if (kf.has_camera) return GraphTrackType::Camera;
    }
    if (suffix == "Material" || suffix.rfind("Material.", 0) == 0 || suffix.rfind("Mat.", 0) == 0) {
        return GraphTrackType::Material;
    }
    return GraphTrackType::Transform;
}

// Set a curve key's interpolation so BOTH segments adjacent to the key reflect it.
// A segment's Constant/Linear/Bezier mode is taken from its LEFT key's meta, so:
//   - the OUTGOING segment (key -> next) is governed by THIS key's meta, and
//   - the INCOMING segment (prev -> key) is governed by the PREVIOUS keyed key's meta.
// Editing both makes the graph-editor interp control produce a visible change for any
// selected key (including the last one, which has no outgoing segment). This is the
// user-chosen "both adjacent segments" convention.
static void applyKeyInterpBothSides(ObjectAnimationTrack& track, GraphTrackType track_type,
                                    int channel, int frame, KeyInterp mode) {
    auto curveMeta = [&](Keyframe& kf) -> ChannelKeyMeta* {
        switch (track_type) {
            case GraphTrackType::Light:    return (channel < CURVE_LIGHT_CHANNEL_COUNT) ? &kf.light.curve[channel]    : nullptr;
            case GraphTrackType::Camera:   return (channel < CURVE_CAM_CHANNEL_COUNT)   ? &kf.camera.curve[channel]   : nullptr;
            case GraphTrackType::Material: return (channel < CURVE_MAT_CHANNEL_COUNT)   ? &kf.material.curve[channel] : nullptr;
            default:                       return (channel < CURVE_CHANNEL_COUNT)       ? &kf.transform.curve[channel]: nullptr;
        }
    };
    auto channelKeyed = [&](const Keyframe& kf) -> bool {
        switch (track_type) {
            case GraphTrackType::Light:    return kf.has_light     && kf.light.channelKeyed(channel);
            case GraphTrackType::Camera:   return kf.has_camera    && kf.camera.channelKeyed(channel);
            case GraphTrackType::Material: return kf.has_material  && kf.material.channelKeyed(channel);
            default:                       return kf.has_transform && kf.transform.channelKeyed(channel);
        }
    };

    const bool bez = (mode == KeyInterp::Bezier);

    // Outgoing segment: the selected key's own meta.
    if (Keyframe* sel = track.getKeyframeAt(frame)) {
        if (ChannelKeyMeta* m = curveMeta(*sel)) { m->interp = mode; m->auto_tangent = bez; }
    }

    // Incoming segment: the previous keyed key for this channel governs it.
    Keyframe* prev = nullptr;
    for (auto& kf : track.keyframes) {
        if (kf.frame < frame && channelKeyed(kf) && (!prev || kf.frame > prev->frame)) prev = &kf;
    }
    if (prev) {
        if (ChannelKeyMeta* m = curveMeta(*prev)) { m->interp = mode; m->auto_tangent = bez; }
    }

    track.refreshAutoTangents();
}

// Returns { "Cube_1", ChannelType::Location } if input is "Cube_1.Location"
static std::pair<std::string, ChannelType> parseTrackName(const SceneData& scene, const std::string& track_name) {
    size_t dot_pos = track_name.find('.');
    if (dot_pos == std::string::npos) {
        return { track_name, ChannelType::None };
    }
    
    std::string entity = track_name.substr(0, dot_pos);
    std::string suffix = track_name.substr(dot_pos + 1);
    
    bool is_light = false;
    bool is_camera = false;
    
    auto track_it = scene.timeline.tracks.find(entity);
    if (track_it != scene.timeline.tracks.end() && !track_it->second.keyframes.empty()) {
        const auto& kf = track_it->second.keyframes[0];
        if (kf.has_light) is_light = true;
        else if (kf.has_camera) is_camera = true;
    }
    
    if (!is_light && !is_camera) {
        for (const auto& l : scene.lights) {
            if (l && l->nodeName == entity) { is_light = true; break; }
        }
    }
    if (!is_light && !is_camera) {
        for (const auto& c : scene.cameras) {
            if (c && c->nodeName == entity) { is_camera = true; break; }
        }
    }
    
    ChannelType channel = ChannelType::None;
    if (suffix == "Location") channel = ChannelType::Location;
    else if (suffix == "Location.X" || suffix == "X") {
        if (is_light) channel = ChannelType::LightPosX;
        else if (is_camera) channel = ChannelType::CamPosX;
        else channel = ChannelType::LocationX;
    }
    else if (suffix == "Location.Y" || suffix == "Y") {
        if (is_light) channel = ChannelType::LightPosY;
        else if (is_camera) channel = ChannelType::CamPosY;
        else channel = ChannelType::LocationY;
    }
    else if (suffix == "Location.Z" || suffix == "Z") {
        if (is_light) channel = ChannelType::LightPosZ;
        else if (is_camera) channel = ChannelType::CamPosZ;
        else channel = ChannelType::LocationZ;
    }
    
    else if (suffix == "Rotation") channel = ChannelType::Rotation;
    else if (suffix == "Rotation.X" || suffix == "X") {
        if (is_light) channel = ChannelType::LightDirX;
        else channel = ChannelType::RotationX;
    }
    else if (suffix == "Rotation.Y") {
        if (is_light) channel = ChannelType::LightDirY;
        else channel = ChannelType::RotationY;
    }
    else if (suffix == "Rotation.Z") {
        if (is_light) channel = ChannelType::LightDirZ;
        else channel = ChannelType::RotationZ;
    }
    
    else if (suffix == "Scale") channel = ChannelType::Scale;
    else if (suffix == "Scale.X") channel = ChannelType::ScaleX;
    else if (suffix == "Scale.Y") channel = ChannelType::ScaleY;
    else if (suffix == "Scale.Z") channel = ChannelType::ScaleZ;
    
    else if (suffix == "Material") channel = ChannelType::Material;
    
    // Light/Camera/World specifics
    else if (suffix == "Position") {
        channel = ChannelType::Location; 
    }
    else if (suffix == "Color") {
        channel = ChannelType::Material; 
    }
    else if (suffix == "Intensity") {
        channel = ChannelType::LightIntensity; 
    }
    else if (suffix == "Direction") {
        channel = ChannelType::Rotation; 
    }
    else if (suffix == "Target") {
        channel = ChannelType::Rotation; 
    }
    else if (suffix == "FOV") {
        channel = ChannelType::CamFOV; 
    }

    // Light/Camera detailed suffixes
    else if (suffix == "Position.X") {
        if (is_camera) channel = ChannelType::CamPosX;
        else channel = ChannelType::LightPosX;
    }
    else if (suffix == "Position.Y") {
        if (is_camera) channel = ChannelType::CamPosY;
        else channel = ChannelType::LightPosY;
    }
    else if (suffix == "Position.Z") {
        if (is_camera) channel = ChannelType::CamPosZ;
        else channel = ChannelType::LightPosZ;
    }
    else if (suffix == "Color.R") channel = ChannelType::LightColR;
    else if (suffix == "Color.G") channel = ChannelType::LightColG;
    else if (suffix == "Color.B") channel = ChannelType::LightColB;
    else if (suffix == "Direction.X") channel = ChannelType::LightDirX;
    else if (suffix == "Direction.Y") channel = ChannelType::LightDirY;
    else if (suffix == "Direction.Z") channel = ChannelType::LightDirZ;
    
    else if (suffix == "Target.X") channel = ChannelType::CamTgtX;
    else if (suffix == "Target.Y") channel = ChannelType::CamTgtY;
    else if (suffix == "Target.Z") channel = ChannelType::CamTgtZ;
    else if (suffix == "FocusDistance") channel = ChannelType::CamFocusDist;
    else if (suffix == "LensRadius") channel = ChannelType::CamLensRad;

    // Material detailed suffixes
    else if (suffix == "Material.Albedo.R") channel = ChannelType::MatAlbedoR;
    else if (suffix == "Material.Albedo.G") channel = ChannelType::MatAlbedoG;
    else if (suffix == "Material.Albedo.B") channel = ChannelType::MatAlbedoB;
    else if (suffix == "Material.Opacity") channel = ChannelType::MatOpacity;
    else if (suffix == "Material.Roughness") channel = ChannelType::MatRoughness;
    else if (suffix == "Material.Metallic") channel = ChannelType::MatMetallic;
    else if (suffix == "Material.Clearcoat") channel = ChannelType::MatClearcoat;
    else if (suffix == "Material.Transmission") channel = ChannelType::MatTransmission;
    else if (suffix == "Material.IOR") channel = ChannelType::MatIOR;
    else if (suffix == "Material.Emission.R") channel = ChannelType::MatEmissionR;
    else if (suffix == "Material.Emission.G") channel = ChannelType::MatEmissionG;
    else if (suffix == "Material.Emission.B") channel = ChannelType::MatEmissionB;
    else if (suffix == "Material.NormalStrength") channel = ChannelType::MatNormalStrength;
    else if (suffix == "Material.EmissionStrength") channel = ChannelType::MatEmissionStrength;
    
    if (channel == ChannelType::None) {
        return { track_name, ChannelType::None };
    }
    
    return { entity, channel };
}

static const char* channelSubtrackSuffix(int ch) {
    static const char* suffixes[CURVE_CHANNEL_COUNT] = {
        "Location.X", "Location.Y", "Location.Z",
        "Rotation.X", "Rotation.Y", "Rotation.Z",
        "Scale.X", "Scale.Y", "Scale.Z"
    };
    return (ch >= 0 && ch < CURVE_CHANNEL_COUNT) ? suffixes[ch] : "";
}

static const char* channelSubtrackSuffix(GraphTrackType type, int ch) {
    if (type == GraphTrackType::Light) {
        static const char* suffixes[CURVE_LIGHT_CHANNEL_COUNT] = {
            "Position.X", "Position.Y", "Position.Z",
            "Color.R", "Color.G", "Color.B",
            "Intensity",
            "Direction.X", "Direction.Y", "Direction.Z"
        };
        return (ch >= 0 && ch < CURVE_LIGHT_CHANNEL_COUNT) ? suffixes[ch] : "";
    } else if (type == GraphTrackType::Camera) {
        static const char* suffixes[CURVE_CAM_CHANNEL_COUNT] = {
            "Position.X", "Position.Y", "Position.Z",
            "Target.X", "Target.Y", "Target.Z",
            "FOV",
            "FocusDistance",
            "LensRadius"
        };
        return (ch >= 0 && ch < CURVE_CAM_CHANNEL_COUNT) ? suffixes[ch] : "";
    } else if (type == GraphTrackType::Material) {
        static const char* suffixes[CURVE_MAT_CHANNEL_COUNT] = {
            "Material.Albedo.R", "Material.Albedo.G", "Material.Albedo.B",
            "Material.Opacity",
            "Material.Roughness",
            "Material.Metallic",
            "Material.Clearcoat",
            "Material.Transmission",
            "Material.IOR",
            "Material.Emission.R", "Material.Emission.G", "Material.Emission.B",
            "Material.NormalStrength",
            "Material.EmissionStrength"
        };
        return (ch >= 0 && ch < CURVE_MAT_CHANNEL_COUNT) ? suffixes[ch] : "";
    } else {
        return channelSubtrackSuffix(ch);
    }
}

static SceneData::ImportedModelContext* findImportedModelContextByName(SceneData& scene, const std::string& name) {
    for (auto& mctx : scene.importedModelContexts) {
        if (mctx.importName == name) return &mctx;
    }
    return nullptr;
}

static std::string resolveCharacterTrackName(SceneData& scene, const std::string& selectionName) {
    if (selectionName.empty()) return selectionName;
    struct SelectionTrackResolveCache {
        const SceneData* scene = nullptr;
        size_t imported_model_count = 0;
        size_t total_member_count = 0;
        std::unordered_map<std::string, std::string> node_to_track;
    };

    static SelectionTrackResolveCache cache;

    size_t total_member_count = 0;
    for (const auto& mctx : scene.importedModelContexts) {
        total_member_count += mctx.members.size();
    }

    const bool rebuild_cache =
        cache.scene != &scene ||
        cache.imported_model_count != scene.importedModelContexts.size() ||
        cache.total_member_count != total_member_count;

    if (rebuild_cache) {
        cache.scene = &scene;
        cache.imported_model_count = scene.importedModelContexts.size();
        cache.total_member_count = total_member_count;
        cache.node_to_track.clear();

        for (const auto& mctx : scene.importedModelContexts) {
            cache.node_to_track[mctx.importName] = mctx.importName;

            // Only remap to importName for skeletal models (bone-weighted meshes).
            // Rigid node animation keeps its own unique node name as the track key.
            if (!mctx.hasAnimation || mctx.weightedBoneCount == 0) continue;

            for (const auto& member : mctx.members) {
                auto tri = std::dynamic_pointer_cast<Triangle>(member);
                if (!tri) continue;

                const std::string& node_name = tri->getNodeName();
                if (!node_name.empty()) {
                    cache.node_to_track.emplace(node_name, mctx.importName);
                }
            }
        }
    }

    auto it = cache.node_to_track.find(selectionName);
    if (it != cache.node_to_track.end()) {
        return it->second;
    }

    return selectionName;
}

static void addAnimGraphTrackTree(std::vector<TimelineTrack>& tracks,
    const std::string& entityName,
    const std::vector<Keyframe>& keyframes) {
    std::vector<int> allFrames;
    std::vector<int> stateFrames;
    std::vector<int> clipFrames;
    std::vector<int> paramFrames;
    std::vector<int> triggerFrames;
    std::set<std::string> stateLabels;
    std::set<std::string> clipLabels;
    std::set<std::string> paramLabels;

    for (const auto& kf : keyframes) {
        if (!kf.has_anim_graph) continue;
        allFrames.push_back(kf.frame);

        if (!kf.anim_graph.force_state.empty()) {
            stateFrames.push_back(kf.frame);
            stateLabels.insert(kf.anim_graph.force_state);
        }
        if (!kf.anim_graph.clip_overrides.empty() || !kf.anim_graph.clip_speed_overrides.empty()) {
            clipFrames.push_back(kf.frame);
            for (const auto& [_, clipName] : kf.anim_graph.clip_overrides) {
                if (!clipName.empty()) clipLabels.insert(clipName);
            }
        }
        if (!kf.anim_graph.float_params.empty() || !kf.anim_graph.bool_params.empty() || !kf.anim_graph.int_params.empty()) {
            paramFrames.push_back(kf.frame);
            for (const auto& [name, _] : kf.anim_graph.float_params) paramLabels.insert(name);
            for (const auto& [name, _] : kf.anim_graph.bool_params) paramLabels.insert(name);
            for (const auto& [name, _] : kf.anim_graph.int_params) paramLabels.insert(name);
        }
        if (!kf.anim_graph.triggers.empty()) {
            triggerFrames.push_back(kf.frame);
            for (const auto& trigger : kf.anim_graph.triggers) paramLabels.insert(trigger);
        }
    }

    TimelineTrack main;
    main.entity_name = entityName;
    main.name = entityName;
    main.group = TrackGroup::Objects;
    main.color = IM_COL32(220, 120, 255, 255);
    main.expanded = true;
    main.keyframe_frames = allFrames;
    tracks.push_back(main);

    TimelineTrack animRoot;
    animRoot.entity_name = entityName;
    animRoot.parent_entity = entityName;
    animRoot.name = "AnimGraph";
    animRoot.group = TrackGroup::Objects;
    animRoot.color = IM_COL32(220, 120, 255, 255);
    animRoot.expanded = true;
    animRoot.is_sub_track = true;
    animRoot.depth = 1;
    animRoot.keyframe_frames = allFrames;
    tracks.push_back(animRoot);

    TimelineTrack stateTrack;
    stateTrack.entity_name = entityName;
    stateTrack.parent_entity = entityName;
    stateTrack.name = stateLabels.empty() ? "States" : ("States: " + *stateLabels.begin());
    stateTrack.group = TrackGroup::Objects;
    stateTrack.color = IM_COL32(255, 210, 120, 255);
    stateTrack.is_sub_track = true;
    stateTrack.depth = 2;
    stateTrack.keyframe_frames = stateFrames;
    tracks.push_back(stateTrack);

    TimelineTrack clipTrack;
    clipTrack.entity_name = entityName;
    clipTrack.parent_entity = entityName;
    clipTrack.name = clipLabels.empty() ? "Clips" : ("Clips: " + *clipLabels.begin());
    clipTrack.group = TrackGroup::Objects;
    clipTrack.color = IM_COL32(120, 220, 255, 255);
    clipTrack.is_sub_track = true;
    clipTrack.depth = 2;
    clipTrack.keyframe_frames = clipFrames;
    tracks.push_back(clipTrack);

    TimelineTrack paramTrack;
    paramTrack.entity_name = entityName;
    paramTrack.parent_entity = entityName;
    paramTrack.name = paramLabels.empty() ? "Parameters" : ("Parameters: " + *paramLabels.begin());
    paramTrack.group = TrackGroup::Objects;
    paramTrack.color = IM_COL32(120, 255, 170, 255);
    paramTrack.is_sub_track = true;
    paramTrack.depth = 2;
    paramTrack.keyframe_frames = paramFrames;
    tracks.push_back(paramTrack);

    TimelineTrack triggerTrack;
    triggerTrack.entity_name = entityName;
    triggerTrack.parent_entity = entityName;
    triggerTrack.name = "Triggers";
    triggerTrack.group = TrackGroup::Objects;
    triggerTrack.color = IM_COL32(255, 140, 160, 255);
    triggerTrack.is_sub_track = true;
    triggerTrack.depth = 2;
    triggerTrack.keyframe_frames = triggerFrames;
    tracks.push_back(triggerTrack);
}

// ============================================================================
// MAIN DRAW FUNCTION
// ============================================================================
void TimelineWidget::draw(UIContext& ctx) {
    UIWidgets::PushControlSurfaceStyle(ImVec4(0.46f, 0.86f, 0.92f, 1.0f));
    // ANIMATION RENDER SYNC: When animation render is active, 
    // timeline should FOLLOW the render frame, not control it
    if (ctx.is_animation_mode && rendering_in_progress) {
        // Read current frame FROM render_settings (set by Renderer thread
        // during render, or by the Main.cpp pre-render pin during the
        // deferred-launch countdown). is_animation_mode covers both the
        // countdown window AND the active worker run, so the widget
        // never overwrites the pinned/worker frame with stale UI state.
        current_frame = ctx.render_settings.animation_current_frame;
        // Disable playback during render
        is_playing = false;
    }
    
    // Sync new AnimationData entries every frame — cheap (count compare + early return)
    syncFromAnimationData(ctx);
    
    // PERFORMANCE: Handle selection change with minimal overhead
    handleSelectionSync(ctx);

    static size_t last_scene_track_count = std::numeric_limits<size_t>::max();
    if (ctx.scene.timeline.tracks.size() != last_scene_track_count) {
        tracks_dirty = true;
        last_scene_track_count = ctx.scene.timeline.tracks.size();
    }
    
    // Rebuild tracks if needed
    if (tracks_dirty) {
        rebuildTrackList(ctx);
        tracks_dirty = false;
    }
    
    // Get available region
    ImVec2 region = ImGui::GetContentRegionAvail();
    float total_height = region.y;
    
    // --- PLAYBACK CONTROLS ---
    drawPlaybackControls(ctx);
    drawSelectedAnimGraphInspector(ctx);
    
    ImGui::Separator();
    
    // Calculate remaining height dynamically so we don't need hardcoded layout offsets
    float canvas_height = ImGui::GetContentRegionAvail().y;
    
    // --- MAIN TIMELINE AREA ---
    // Split into track list (left) and canvas (right)
    ImGui::BeginChild("TimelineArea", ImVec2(0, canvas_height), false, ImGuiWindowFlags_NoScrollbar);
    
    // Left panel: Track list (dope sheet) or channel list (graph editor)
    ImGui::BeginChild("TrackList", ImVec2(legend_width, canvas_height), true, ImGuiWindowFlags_NoScrollbar);
    if (editor_mode == TimelineEditorMode::GraphEditor)
        drawGraphChannelList(ctx, legend_width);
    else
        drawTrackList(ctx, legend_width, canvas_height);
    ImGui::EndChild();

    ImGui::SameLine();

    // Splitting resizer
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.4f, 0.4f, 0.4f, 0.5f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.3f, 0.3f, 0.3f));
    ImGui::Button("##splitter", ImVec2(4.0f, canvas_height));
    ImGui::PopStyleColor(3);
    if (ImGui::IsItemActive()) {
        legend_width += ImGui::GetIO().MouseDelta.x;
        if (legend_width < 100.0f) legend_width = 100.0f;
        if (legend_width > 600.0f) legend_width = 600.0f;
    }
    if (ImGui::IsItemHovered() || ImGui::IsItemActive()) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
    }

    ImGui::SameLine();

    // Right panel: Timeline canvas (keyframe diamonds) or curve canvas
    ImGui::BeginChild("TimelineCanvas", ImVec2(0, canvas_height), true,
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoMove);
    float avail_width = ImGui::GetContentRegionAvail().x;
    if (editor_mode == TimelineEditorMode::GraphEditor)
        drawGraphCanvas(ctx, avail_width, canvas_height);
    else
        drawTimelineCanvas(ctx, avail_width, canvas_height);
    ImGui::EndChild();
    
    ImGui::EndChild();
    
    // --- PLAYBACK UPDATE ---
    // --- PLAYBACK TIMER ---
    // last_time must be reset every time playback starts to prevent accumulated
    // dead-time from causing a burst of skipped frames on the first tick.
    static bool was_playing = false;
    static std::chrono::steady_clock::time_point last_time = std::chrono::steady_clock::now();
    if (is_playing && !was_playing) {
        // Playback just started - reset timer
        last_time = std::chrono::steady_clock::now();
    }
    was_playing = is_playing;

    if (is_playing) {
        auto now = std::chrono::steady_clock::now();
        float elapsed = std::chrono::duration<float>(now - last_time).count();
        float frame_duration = 1.0f / ctx.render_settings.animation_fps;

        if (elapsed >= frame_duration) {
            // Advance exactly as many frames as have elapsed (fixes slow-motion
            // when render time > frame_duration, e.g. Vulkan 50ms vs 33ms@30fps)
            int frames_to_advance = static_cast<int>(elapsed / frame_duration);
            // Cap to 1 frame to guarantee correct per-frame animation (prevents
            // skipping frames on a slow render, which looks like fast forwarding)
            if (frames_to_advance > 1) frames_to_advance = 1;

            current_frame += frames_to_advance;
            int range = end_frame - start_frame + 1;
            if (range > 0 && current_frame > end_frame) {
                if (loop_enabled) {
                    // Wrap back to start (only when the user opted into looping).
                    current_frame = start_frame + (current_frame - start_frame) % range;
                } else {
                    // Default: stop at the last frame. Avoids the per-loop sim
                    // re-bake / cache wipe that thrashes memory.
                    current_frame = end_frame;
                    is_playing = false;
                }
            }

            // Carry over the fractional remainder so timing stays accurate
            last_time += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<float>(frames_to_advance * frame_duration));
        }
    }
    
    // [RENDER-LOCK RACE FIX] During an active sequence render, the worker
    // thread owns ALL keyframe evaluation + transform application + backend
    // material updates — see Renderer::render_Animation. If we re-apply
    // keyframes on the main thread here we get concurrent writes to the
    // same Transform objects, HittableInstance::setTransform, and backend
    // resetAccumulation/updateBackendMaterial calls — manifesting as a
    // silent NVIDIA driver hang inside traceRays once the TLAS is touched
    // mid-trace (no AV, no exception; classic symptom of mutating an AS
    // while the GPU still references it). Setting g_optix_rebuild_pending
    // from here would also race with the worker. Skip the worker-owned
    // writes below while the render thread is authoritative; the worker
    // keeps render_settings.animation_current_frame updated for the UI.
    // [GUARD WINDOW FIX] Use is_animation_mode (set by Main.cpp BEFORE the
    // worker thread is detached) rather than animation_render_locked (set
    // by the worker thread AFTER it starts running inside
    // Renderer::render_Animation — tens of ms later for a detached thread).
    // Otherwise the first 1–2 rendered frames pick up the user's pre-render
    // scrub pose because this widget's draw() writes its own current_frame
    // back into scene.timeline.current_frame in the gap window.
    // A viewport-driven sequence save (g_seq_save_active) advances the timeline
    // itself (Main.cpp state machine via setCurrentFrame) and needs keyframes to
    // keep being applied here, so it is NOT a worker-owned timeline.
    extern bool g_seq_save_active;
    const bool render_owns_timeline =
        ctx.is_animation_mode && rendering_in_progress.load() && !g_seq_save_active;

    if (!render_owns_timeline) {
        // Sync to render settings (only when worker isn't authoritative —
        // otherwise these writes race with Renderer::render_Animation).
        ctx.render_settings.animation_is_playing = is_playing;
        ctx.render_settings.animation_current_frame = current_frame;
        ctx.render_settings.animation_playback_frame = current_frame;
        ctx.scene.timeline.current_frame = current_frame;
    } else {
        // Pull worker-set frame so UI scrub indicator tracks the render.
        current_frame = ctx.render_settings.animation_current_frame;
        UIWidgets::PopControlSurfaceStyle();
        return;
    }

    // --- APPLY TERRAIN ANIMATIONS ---
    // PERFORMANCE: Skip if no timeline tracks exist at all
    if (!ctx.scene.timeline.tracks.empty()) {
        static int last_terrain_update_frame = -1;
        if (current_frame != last_terrain_update_frame) {
            // Only iterate if we might have terrain keyframes
            for (auto& [track_name, track] : ctx.scene.timeline.tracks) {
                // PERFORMANCE: Skip empty tracks immediately
                if (track.keyframes.empty()) continue;
                
                // Quick check: does first keyframe have terrain? (common case optimization)
                bool has_terrain_kf = track.keyframes[0].has_terrain;
                if (!has_terrain_kf) {
                    // Check remaining only if first doesn't have it
                    for (size_t i = 1; i < track.keyframes.size(); ++i) {
                        if (track.keyframes[i].has_terrain) {
                            has_terrain_kf = true;
                            break;
                        }
                    }
                }
                
                if (has_terrain_kf) {
                    auto& terrains = TerrainManager::getInstance().getTerrains();
                    for (auto& terrain : terrains) {
                        if (terrain.name == track_name) {
                            TerrainManager::getInstance().updateFromTrack(&terrain, track, current_frame);
                            ctx.renderer.resetCPUAccumulation();
                            g_bvh_rebuild_pending = true;
                            g_optix_rebuild_pending = true;
                            if (ctx.backend_ptr) ctx.backend_ptr->resetAccumulation();
                            break;
                        }
                    }
                }
            }
            last_terrain_update_frame = current_frame;
        }
    }
    
    // --- APPLY WATER KEYFRAME ANIMATIONS ---
    // Similar to terrain, update water parameters when frame changes
    {
        static int last_water_update_frame = -1;
        if (current_frame != last_water_update_frame) {
            bool has_any_water_kf = false;
            for (auto& [track_name, track] : ctx.scene.timeline.tracks) {
                if (track.keyframes.empty()) continue;
                
                // Check if track has water keyframes
                bool has_water_kf = false;
                for (const auto& kf : track.keyframes) {
                    if (kf.has_water) {
                        has_water_kf = true;
                        break;
                    }
                }
                
                if (has_water_kf && track_name.find("Water_") == 0) {
                    has_any_water_kf = true;
                    // Extract water surface ID from track name (e.g., "Water_1" -> 1)
                    int water_id = -1;
                    try {
                        water_id = std::stoi(track_name.substr(6));
                    } catch (...) {
                        continue;
                    }
                    
                    WaterSurface* surf = WaterManager::getInstance().getWaterSurface(water_id);
                    if (surf) {
                        WaterManager::getInstance().updateFromTrack(surf, track, current_frame);
                        if (surf->material_id != MaterialManager::INVALID_MATERIAL_ID) {
                            WaterManager::getInstance().syncSurfaceMaterial(surf);
                            ctx.renderer.updateBackendMaterial(ctx.scene, surf->material_id);
                        }
                        ctx.renderer.resetCPUAccumulation();
                        g_bvh_rebuild_pending = true;
                        g_optix_rebuild_pending = true;
                        if (ctx.backend_ptr) ctx.backend_ptr->resetAccumulation();
                    }
                }
            }

            bool timeline_preview_water = false;
            if (WaterManager::getInstance().getPreviewTimeMode() == WaterPreviewTimeMode::Timeline) {
                for (const auto& surf : WaterManager::getInstance().getWaterSurfaces()) {
                    if (surf.params.use_fft_ocean || surf.params.use_geometric_waves ||
                        surf.animate_mesh || surf.params.wave_strength > 0.0001f) {
                        timeline_preview_water = true;
                        break;
                    }
                }
            }
            
            // After applying all water keyframes, run actual water simulation update
            // This updates FFT ocean and geometric wave mesh animation
            if (has_any_water_kf || timeline_preview_water) {
                float fps = static_cast<float>(ctx.render_settings.animation_fps > 0 ? ctx.render_settings.animation_fps : 24);
                float frame_time = static_cast<float>(current_frame) / fps;
                WaterUpdateResult waterUpdate = WaterManager::getInstance().update(frame_time);
                if (waterUpdate.mesh_changed) {
                    // Water mesh changed - need geometry update
                    g_bvh_rebuild_pending = true;
                    g_optix_rebuild_pending = true;
                }
                if (waterUpdate.requiresAccumulationReset() && ctx.backend_ptr) {
                    ctx.backend_ptr->resetAccumulation();
                }
            }
            
            last_water_update_frame = current_frame;
        }
    }
    
    // --- APPLY OBJECT/LIGHT/CAMERA/WORLD ANIMATIONS ---
    // PERFORMANCE: Skip entire animation loop if no tracks exist OR no keyframes exist
    // Count total keyframes FIRST to avoid entering expensive loops unnecessarily
    size_t total_keyframe_count = 0;
    for (const auto& [name, track] : ctx.scene.timeline.tracks) {
        total_keyframe_count += track.keyframes.size();
        if (total_keyframe_count > 0) break; // Early exit if any found
    }
    
    if (total_keyframe_count > 0) {
        static int last_anim_update_frame = -1;
        // anim_reapply_requested_: graph-editor edits change curve shape at an
        // unchanged current_frame, so force one re-apply pass.
        if (current_frame != last_anim_update_frame || anim_reapply_requested_) {
            bool needs_bvh_update = false;
            bool needs_light_update = false;
            bool needs_camera_update = false;
            std::map<std::string, Matrix4x4> pending_object_transforms;
            
            // PERFORMANCE: Build object cache ONCE before the track loop, not per-track
            struct CachedObject {
                std::shared_ptr<Triangle> triangle;               // Representative triangle (always set)
                std::shared_ptr<HittableInstance> instance;       // Set only if wrapped in HittableInstance
            };
            static std::map<std::string, CachedObject> object_cache;
            static bool cache_valid = false;
            static size_t last_object_count = 0;
            
            // Invalidate cache if object count changed
            if (ctx.scene.world.objects.size() != last_object_count) {
                cache_valid = false;
                last_object_count = ctx.scene.world.objects.size();
            }
            
            // Build cache once (not per track!)
            if (!cache_valid) {
                object_cache.clear();
                for (auto& obj : ctx.scene.world.objects) {
                    auto tri = std::dynamic_pointer_cast<Triangle>(obj);
                    if (tri && !tri->getNodeName().empty()) {
                        object_cache[tri->getNodeName()] = { tri, nullptr };
                    } else {
                        // Also check HittableInstance objects (e.g., default scene cube)
                        auto inst = std::dynamic_pointer_cast<HittableInstance>(obj);
                        if (inst && inst->source_triangles && !inst->source_triangles->empty()) {
                            auto& first_tri = inst->source_triangles->at(0);
                            if (first_tri && !first_tri->getNodeName().empty()) {
                                object_cache[first_tri->getNodeName()] = { first_tri, inst };
                            } else if (!inst->node_name.empty()) {
                                object_cache[inst->node_name] = { first_tri, inst };
                            }
                        }
                    }
                }
                cache_valid = true;
            }
            
            for (auto& [track_name, track] : ctx.scene.timeline.tracks) {
                // PERFORMANCE: Skip empty tracks immediately
                if (track.keyframes.empty()) continue;
                
                // Quick terrain check (first keyframe optimization)
                bool is_terrain = track.keyframes[0].has_terrain;
                if (is_terrain) continue;
            
            // Evaluate animation at current frame
            Keyframe evaluated = track.evaluate(current_frame);
            
            // Apply transform to objects
            // PERFORMANCE: Skip CPU vertex update - GPU will apply transform via TLAS
            if (evaluated.has_transform) {
                // O(1) lookup using pre-built cache
                auto it = object_cache.find(track_name);
                if (it != object_cache.end()) {
                    auto& cached = it->second;
                    Transform* th = cached.triangle ? cached.triangle->getTransformPtr() : nullptr;
                    if (th) {
                        // Build transform from evaluated keyframe
                        Matrix4x4 new_transform = Matrix4x4::fromTRS(
                            evaluated.transform.position,
                            evaluated.transform.rotation,
                            evaluated.transform.scale
                        );
                        th->setPivotMatrix(new_transform);
                        
                        // Also update HittableInstance transform if this object is wrapped
                        if (cached.instance) {
                            cached.instance->setTransform(th->base);
                        }
                        pending_object_transforms[track_name] = new_transform;
                        
                        // GPU mode: Skip CPU vertex update - TLAS handles transforms
                        // CPU mode: Need to update world-space vertices for raytracing
                        bool using_gpu = (ctx.render_settings.use_optix || ctx.render_settings.use_vulkan) && ctx.backend_ptr;
                        if (!using_gpu) {
                            // CPU rendering - update world-space vertices
                            if (cached.instance && cached.instance->source_triangles) {
                                // Keep instanced source geometry local; CPU BVH should move the wrapper transform.
                                if (!cached.instance->syncTransformFromSourceTriangles()) {
                                    for (auto& src_tri : *cached.instance->source_triangles) {
                                        if (src_tri) src_tri->updateTransformedVertices();
                                    }
                                }
                            } else {
                                // Raw triangles in scene: update all with same nodeName
                                for (auto& obj : ctx.scene.world.objects) {
                                    auto mesh_tri = std::dynamic_pointer_cast<Triangle>(obj);
                                    if (mesh_tri && mesh_tri->getNodeName() == track_name) {
                                        mesh_tri->updateTransformedVertices();
                                    }
                                }
                            }
                        }
                        
                        needs_bvh_update = true;
                    }
                }
            }
            
            // Apply light keyframes
            if (evaluated.has_light) {
                bool found = false;
                for (auto& light : ctx.scene.lights) {
                    if (light->nodeName == track_name) {
                        if (evaluated.light.has_position) light->position = evaluated.light.position;
                        if (evaluated.light.has_color) light->color = evaluated.light.color;
                        if (evaluated.light.has_intensity) light->intensity = evaluated.light.intensity;
                        if (evaluated.light.has_direction) light->direction = evaluated.light.direction;
                        needs_light_update = true;
                        found = true;
                        
                        // DEBUG: Confirmed application
                        // SCENE_LOG_INFO("Anim applied to light: " + track_name); 
                        break;
                    }
                }
                if (!found) {
                     // SMART RECOVERY: Try to match by index if name fails
                     // 1. Calculate which "Light Track" number this is
                     int light_track_index = 0;
                     for(auto& [t_name, t_track] : ctx.scene.timeline.tracks) {
                         if (t_name == track_name) break;
                         // Check if this is a light track
                         bool acts_on_light = false;
                         if (!t_track.keyframes.empty() && t_track.keyframes[0].has_light) acts_on_light = true;
                         if(acts_on_light) light_track_index++;
                     }

                     // 2. Try to find corresponding light
                     if (light_track_index < (int)ctx.scene.lights.size()) {
                         auto fallback_light = ctx.scene.lights[light_track_index];
                         
                         // Apply animation to fallback
                         if (evaluated.light.has_position) fallback_light->position = evaluated.light.position;
                         if (evaluated.light.has_color) fallback_light->color = evaluated.light.color;
                         if (evaluated.light.has_intensity) fallback_light->intensity = evaluated.light.intensity;
                         if (evaluated.light.has_direction) fallback_light->direction = evaluated.light.direction;
                         needs_light_update = true;

                         // Log and Auto-Fix Name
                         static bool logged_recovery = false;
                         if (!logged_recovery) {
                             SCENE_LOG_WARN("Recovered Anim Link: '" + track_name + "' -> '" + fallback_light->nodeName + "'");
                             logged_recovery = true;
                         }
                         
                         // Permanently fix name if it looks generic
                         if (fallback_light->nodeName.find("Light_") == 0) {
                             fallback_light->nodeName = track_name;
                         }
                     } else {
                         // Real failure
                         // SCENE_LOG_WARN("Anim failed to find light: " + track_name);
                     }
                }
            }
            
            // Apply camera keyframes
            if (evaluated.has_camera) {
                for (auto& cam : ctx.scene.cameras) {
                    if (cam->nodeName == track_name) {
                        if (evaluated.camera.has_position) cam->lookfrom = evaluated.camera.position;
                        if (evaluated.camera.has_target) cam->lookat = evaluated.camera.target;
                        if (evaluated.camera.has_fov) cam->vfov = evaluated.camera.fov;
                        if (evaluated.camera.has_aperture) cam->aperture = evaluated.camera.lens_radius;
                        if (evaluated.camera.has_focus) cam->focus_dist = evaluated.camera.focus_distance;
                        cam->update_camera_vectors();
                        needs_camera_update = true;
                        break;
                    }
                }
            }
            
            // Apply world keyframes
            if (evaluated.has_world) {
                NishitaSkyParams nishita = ctx.renderer.world.getNishitaParams();
                AtmosphereAdvanced advanced = ctx.renderer.world.getAdvancedParams();
                WeatherParams weather = ctx.renderer.world.getWeatherParams();
                const NishitaSkyParams originalNishita = nishita;
                const AtmosphereAdvanced originalAdvanced = advanced;
                const WeatherParams originalWeather = weather;
                const Vec3 originalBackgroundColor = ctx.renderer.world.getColor();
                const float originalBackgroundStrength = ctx.renderer.world.getColorIntensity();
                const float originalHDRIRotation = ctx.renderer.world.getHDRIRotation();
                bool changed = false;
                bool advancedChanged = false;
                bool environmentChanged = false;
                bool weatherChanged = false;
                bool sunAnglesChanged = false;
                
                if (evaluated.world.has_sun_elevation) { nishita.sun_elevation = evaluated.world.sun_elevation; changed = true; sunAnglesChanged = true; }
                if (evaluated.world.has_sun_azimuth) { nishita.sun_azimuth = evaluated.world.sun_azimuth; changed = true; sunAnglesChanged = true; }
                if (evaluated.world.has_sun_intensity) { nishita.sun_intensity = evaluated.world.sun_intensity; changed = true; }
                if (evaluated.world.has_sun_size) { nishita.sun_size = evaluated.world.sun_size; changed = true; }
                if (evaluated.world.has_atmosphere_intensity) { nishita.atmosphere_intensity = evaluated.world.atmosphere_intensity; changed = true; }
                if (evaluated.world.has_air_density) { nishita.air_density = evaluated.world.air_density; changed = true; }
                if (evaluated.world.has_dust_density) { nishita.dust_density = evaluated.world.dust_density; changed = true; }
                if (evaluated.world.has_ozone_density) { nishita.ozone_density = evaluated.world.ozone_density; changed = true; }
                if (evaluated.world.has_humidity) { nishita.humidity = evaluated.world.humidity; changed = true; }
                if (evaluated.world.has_temperature) { nishita.temperature = evaluated.world.temperature; changed = true; }
                if (evaluated.world.has_ozone_absorption_scale) { nishita.ozone_absorption_scale = evaluated.world.ozone_absorption_scale; changed = true; }
                if (evaluated.world.has_altitude) { nishita.altitude = evaluated.world.altitude; changed = true; }
                if (evaluated.world.has_mie_anisotropy) { nishita.mie_anisotropy = evaluated.world.mie_anisotropy; changed = true; }
                if (evaluated.world.has_cloud_density) { nishita.cloud_density = evaluated.world.cloud_density; changed = true; }
                if (evaluated.world.has_cloud_coverage) { nishita.cloud_coverage = evaluated.world.cloud_coverage; changed = true; }
                if (evaluated.world.has_cloud_scale) { nishita.cloud_scale = evaluated.world.cloud_scale; changed = true; }
                if (evaluated.world.has_cloud_offset) { 
                    nishita.cloud_offset_x = evaluated.world.cloud_offset_x; 
                    nishita.cloud_offset_z = evaluated.world.cloud_offset_z; 
                    changed = true; 
                }
                if (evaluated.world.has_cloud_lighting) {
                    nishita.cloud_light_steps = evaluated.world.cloud_light_steps;
                    nishita.cloud_shadow_strength = evaluated.world.cloud_shadow_strength;
                    nishita.cloud_ambient_strength = evaluated.world.cloud_ambient_strength;
                    nishita.cloud_silver_intensity = evaluated.world.cloud_silver_intensity;
                    nishita.cloud_absorption = evaluated.world.cloud_absorption;
                    nishita.cloud_anisotropy = evaluated.world.cloud_anisotropy;
                    nishita.cloud_anisotropy_back = evaluated.world.cloud_anisotropy_back;
                    nishita.cloud_lobe_mix = evaluated.world.cloud_lobe_mix;
                    nishita.cloud_emissive_intensity = evaluated.world.cloud_emissive_intensity;
                    nishita.cloud_emissive_color = make_float3(
                        evaluated.world.cloud_emissive_color.x,
                        evaluated.world.cloud_emissive_color.y,
                        evaluated.world.cloud_emissive_color.z);
                    changed = true;
                }
                if (evaluated.world.has_cloud_layer2) {
                    nishita.cloud_layer2_enabled = evaluated.world.cloud_layer2_enabled;
                    changed = true;
                }
                if (evaluated.world.has_cloud_layer2_params) {
                    nishita.cloud2_coverage = evaluated.world.cloud2_coverage;
                    nishita.cloud2_density = evaluated.world.cloud2_density;
                    nishita.cloud2_scale = evaluated.world.cloud2_scale;
                    changed = true;
                }
                if (evaluated.world.has_cloud_layer2_heights) {
                    nishita.cloud2_height_min = evaluated.world.cloud2_height_min;
                    nishita.cloud2_height_max = evaluated.world.cloud2_height_max;
                    changed = true;
                }
                if (evaluated.world.has_fog) { nishita.fog_enabled = evaluated.world.fog_enabled; changed = true; }
                if (evaluated.world.has_fog_params) {
                    nishita.fog_density = evaluated.world.fog_density;
                    nishita.fog_height = evaluated.world.fog_height;
                    nishita.fog_falloff = evaluated.world.fog_falloff;
                    nishita.fog_distance = evaluated.world.fog_distance;
                    nishita.fog_color = make_float3(
                        evaluated.world.fog_color.x,
                        evaluated.world.fog_color.y,
                        evaluated.world.fog_color.z);
                    nishita.fog_sun_scatter = evaluated.world.fog_sun_scatter;
                    changed = true;
                }
                if (evaluated.world.has_godrays) { nishita.godrays_enabled = evaluated.world.godrays_enabled; changed = true; }
                if (evaluated.world.has_godrays_params) {
                    nishita.godrays_intensity = evaluated.world.godrays_intensity;
                    nishita.godrays_density = evaluated.world.godrays_density;
                    nishita.godrays_samples = evaluated.world.godrays_samples;
                    changed = true;
                }

                if (evaluated.world.has_multi_scatter) {
                    advanced.multi_scatter_enabled = evaluated.world.multi_scatter_enabled;
                    advanced.multi_scatter_factor = evaluated.world.multi_scatter_factor;
                    advancedChanged = true;
                }
                if (evaluated.world.has_aerial_perspective) {
                    advanced.aerial_perspective = evaluated.world.aerial_perspective;
                    advancedChanged = true;
                }
                if (evaluated.world.has_aerial_params) {
                    advanced.aerial_density = evaluated.world.aerial_density;
                    advanced.aerial_min_distance = evaluated.world.aerial_min_distance;
                    advanced.aerial_max_distance = evaluated.world.aerial_max_distance;
                    advancedChanged = true;
                }
                if (evaluated.world.has_overlay) {
                    advanced.env_overlay_enabled = evaluated.world.env_overlay_enabled;
                    advancedChanged = true;
                }
                if (evaluated.world.has_overlay_params) {
                    advanced.env_overlay_intensity = evaluated.world.env_overlay_intensity;
                    advanced.env_overlay_rotation = evaluated.world.env_overlay_rotation;
                    advanced.env_overlay_blend_mode = evaluated.world.env_overlay_blend_mode;
                    advancedChanged = true;
                }
                
                if (changed) {
                    if (sunAnglesChanged) {
                        constexpr float kPi = 3.14159265358979323846f;
                        const float elevationRad = nishita.sun_elevation * kPi / 180.0f;
                        const float azimuthRad = nishita.sun_azimuth * kPi / 180.0f;
                        nishita.sun_direction = make_float3(
                            cosf(elevationRad) * sinf(azimuthRad),
                            sinf(elevationRad),
                            cosf(elevationRad) * cosf(azimuthRad));
                    }
                }

                if (evaluated.world.has_weather_params) {
                    weather.enabled = evaluated.world.weather_enabled;
                    weather.type = evaluated.world.weather_type;
                    weather.intensity = evaluated.world.weather_intensity;
                    weather.density = evaluated.world.weather_density;
                    weather.wind_direction = make_float3(
                        evaluated.world.weather_wind_direction.x,
                        evaluated.world.weather_wind_direction.y,
                        evaluated.world.weather_wind_direction.z);
                    weather.wind_speed = evaluated.world.weather_wind_speed;
                    weather.precipitation_scale = evaluated.world.weather_precipitation_scale;
                    weather.visibility = evaluated.world.weather_visibility;
                    weather.surface_wetness_output = evaluated.world.weather_surface_wetness;
                    weather.surface_accumulation_output = evaluated.world.weather_surface_accumulation;
                    weather.surface_settling_output = evaluated.world.weather_surface_settling;
                    weather.surface_height_output = evaluated.world.weather_surface_height;
                    weather.visual_mode = evaluated.world.weather_visual_mode;
                    weather.surface_response_enabled = evaluated.world.weather_surface_response_enabled;
                    weatherChanged = true;
                }

                // Background color (not in Nishita, stored in scene)
                if (evaluated.world.has_background_color) {
                    ctx.scene.background_color = evaluated.world.background_color;
                    ctx.renderer.world.setColor(evaluated.world.background_color);
                    environmentChanged = true;
                }
                // Background strength - Color mode intensity
                if (evaluated.world.has_background_strength) {
                    ctx.renderer.world.setColorIntensity(evaluated.world.background_strength);
                    environmentChanged = true;
                }
                // HDRI rotation
                if (evaluated.world.has_hdri_rotation) {
                    ctx.renderer.world.setHDRIRotation(evaluated.world.hdri_rotation);
                    environmentChanged = true;
                }

                changed = changed && (std::memcmp(&nishita, &originalNishita, sizeof(NishitaSkyParams)) != 0);
                advancedChanged = advancedChanged && (std::memcmp(&advanced, &originalAdvanced, sizeof(AtmosphereAdvanced)) != 0);
                weatherChanged = weatherChanged && (std::memcmp(&weather, &originalWeather, sizeof(WeatherParams)) != 0);
                environmentChanged =
                    environmentChanged &&
                    (ctx.renderer.world.getColor().x != originalBackgroundColor.x ||
                     ctx.renderer.world.getColor().y != originalBackgroundColor.y ||
                     ctx.renderer.world.getColor().z != originalBackgroundColor.z ||
                     ctx.renderer.world.getColorIntensity() != originalBackgroundStrength ||
                     ctx.renderer.world.getHDRIRotation() != originalHDRIRotation);

                if (changed) {
                    ctx.renderer.world.setNishitaParams(nishita);
                }
                if (advancedChanged) {
                    ctx.renderer.world.setAdvancedParams(advanced);
                }
                if (weatherChanged) {
                    ctx.renderer.world.setWeatherParams(weather);
                }

                if (changed || advancedChanged || environmentChanged || weatherChanged) {
                    extern bool g_world_dirty;
                    extern bool g_gas_volumes_dirty;
                    g_world_dirty = true;
                    const bool volumeLightingChanged = changed || advancedChanged || weatherChanged;
                    if (volumeLightingChanged) {
                        g_gas_volumes_dirty = true;
                    }
                    if (ctx.backend_ptr) {
                        ctx.backend_ptr->resetAccumulation();
                    } else {
                        ctx.renderer.resetCPUAccumulation();
                    }
                }
            }
            
            // Apply material keyframes
            if (evaluated.has_material && evaluated.material.material_id != MaterialManager::INVALID_MATERIAL_ID) {
                auto mat = MaterialManager::getInstance().getMaterial(evaluated.material.material_id);
                if (mat && mat->gpuMaterial) {
                    evaluated.material.applyTo(*mat->gpuMaterial);

                    // Sync CPU-side PrincipledBSDF properties so updateBackendMaterials'
                    // capturePBRMaterialSnapshot reads keyframed values, not stale CPU data.
                    if (auto* pbsdf = dynamic_cast<PrincipledBSDF*>(mat)) {
                        pbsdf->albedoProperty.color      = evaluated.material.albedo;
                        pbsdf->roughnessProperty.color   = Vec3(evaluated.material.roughness);
                        pbsdf->metallicProperty.intensity = evaluated.material.metallic;
                        pbsdf->emissionProperty.color    = evaluated.material.emission;
                        pbsdf->ior                       = evaluated.material.ior;
                        pbsdf->transmission              = evaluated.material.transmission;
                        pbsdf->opacityProperty.alpha     = evaluated.material.opacity;
                    }

                    // Push only the animated material slot; full material sync is too
                    // expensive for keyframe playback in dense scenes.
                    ctx.renderer.updateBackendMaterial(ctx.scene, evaluated.material.material_id);
                    ctx.renderer.resetCPUAccumulation();
                    if (ctx.backend_ptr) {
                        ctx.backend_ptr->resetAccumulation();
                    }
                }
            }
        }
        
        // Trigger updates
        if (needs_bvh_update) {
            
            // GPU backend is active when backend_ptr exists and TLAS is built.
            // NOTE: use_optix is FALSE when Vulkan is the active renderer, so we
            // must NOT gate on use_optix here - both renderers use the same call.
            
            if (ctx.backend_ptr) {
                // For the common "one keyed object in a crowded scene" case, avoid
                // scanning every scene object each frame. RT backends rebuild/refit
                // TLAS per targeted call, so keep the full batch path when several
                // objects moved in the same frame.
                // Full-scene instance sync scales with total scene size, so keep
                // small multi-object timeline edits on the targeted path and only
                // fall back to the batch sync once enough objects changed that one
                // TLAS rebuild per object becomes more expensive.
                constexpr size_t kTargetedTransformUpdateLimit = 8;
                const bool use_targeted_transform_update =
                    !pending_object_transforms.empty() &&
                    (!ctx.backend_ptr->isUsingTLAS() ||
                     pending_object_transforms.size() <= kTargetedTransformUpdateLimit);

                if (use_targeted_transform_update) {
                    for (const auto& [node_name, transform] : pending_object_transforms) {
                        ctx.backend_ptr->updateObjectTransform(node_name, transform);
                    }
                } else {
                    // Batch path: one scan and one TLAS update for many animated objects.
                    ctx.backend_ptr->updateInstanceTransforms(ctx.scene.world.objects);
                }
            }
            // CPU picking must not use stale timeline poses, but rebuilding/refitting
            // the CPU BVH every playback frame is too expensive. Mark selection as
            // dirty so the next click refreshes CPU-space vertices and bypasses the
            // stale BVH once.
            g_timeline_selection_sync_pending = true;
            
            ctx.renderer.resetCPUAccumulation();
            if (ctx.backend_ptr) ctx.backend_ptr->resetAccumulation();
        }
        
        if (needs_light_update && ctx.backend_ptr) {
            ctx.backend_ptr->setLights(ctx.scene.lights);
            ctx.backend_ptr->resetAccumulation();
            ctx.renderer.resetCPUAccumulation();
        }
        
        if (needs_camera_update && ctx.backend_ptr && ctx.scene.camera) {
            ctx.backend_ptr->syncCamera(*ctx.scene.camera);
            ctx.backend_ptr->resetAccumulation();
            ctx.renderer.resetCPUAccumulation();
        }
        
            last_anim_update_frame = current_frame;
            anim_reapply_requested_ = false;
        }
    } // end if (!ctx.scene.timeline.tracks.empty())
    UIWidgets::PopControlSurfaceStyle();
}

void TimelineWidget::drawSelectedAnimGraphInspector(UIContext& ctx) {
    if (selected_track.empty() || selected_keyframe_frame < 0) return;

    auto it = ctx.scene.timeline.tracks.find(selected_track);
    if (it == ctx.scene.timeline.tracks.end()) return;

    Keyframe* selectedKey = nullptr;
    for (auto& kf : it->second.keyframes) {
        if (kf.frame == selected_keyframe_frame) {
            selectedKey = &kf;
            break;
        }
    }
    if (!selectedKey || !selectedKey->has_anim_graph) return;

    if (!ImGui::CollapsingHeader("AnimGraph Inspector", ImGuiTreeNodeFlags_DefaultOpen)) return;

    ImGui::TextDisabled("Track: %s", selected_track.c_str());
    ImGui::SameLine();
    ImGui::TextDisabled("| Frame: %d", selected_keyframe_frame);

    if (!selectedKey->anim_graph.force_state.empty()) {
        ImGui::BulletText("Force State: %s", selectedKey->anim_graph.force_state.c_str());
    }
    for (const auto& [nodeId, clipName] : selectedKey->anim_graph.clip_overrides) {
        ImGui::BulletText("Clip Node %u -> %s", nodeId, clipName.c_str());
    }
    for (const auto& [nodeId, clipSpeed] : selectedKey->anim_graph.clip_speed_overrides) {
        ImGui::BulletText("Clip Speed %u = %.3f", nodeId, clipSpeed);
    }
    for (const auto& [name, value] : selectedKey->anim_graph.float_params) {
        ImGui::BulletText("%s = %.3f", name.c_str(), value);
    }
    for (const auto& [name, value] : selectedKey->anim_graph.bool_params) {
        ImGui::BulletText("%s = %s", name.c_str(), value ? "true" : "false");
    }
    for (const auto& [name, value] : selectedKey->anim_graph.int_params) {
        ImGui::BulletText("%s = %d", name.c_str(), value);
    }
    for (const auto& trigger : selectedKey->anim_graph.triggers) {
        ImGui::BulletText("%s [trigger]", trigger.c_str());
    }
}


// ============================================================================
// PLAYBACK CONTROLS + TOOLBAR
// ============================================================================
void TimelineWidget::drawPlaybackControls(UIContext& ctx) {
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4.0f, 2.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4.0f, 0.0f));

    // Check if animation render is active - disable most controls
    bool render_locked = ctx.render_settings.animation_render_locked && rendering_in_progress;
    
    // Show render status indicator when locked - but INCLUDE STOP BUTTON!
    if (render_locked) {
        // Show PAUSED status if paused
        if (rendering_paused.load()) {
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "PAUSED:");
        } else {
            ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f), "RENDERING:");
        }
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.3f, 1.0f), "Frame %d / %d", 
            ctx.render_settings.animation_current_frame,
            ctx.render_settings.animation_end_frame);
        ImGui::SameLine();
        
        // PAUSE/RESUME BUTTON
        if (rendering_paused.load()) {
            if (UIWidgets::IconActionButton("TimelineResume", UIWidgets::IconType::Play, "Resume", true,
                                            ImVec4(0.42f, 0.90f, 0.52f, 1.0f), ImVec2(74.0f, 22.0f),
                                            "Resume animation render")) {
                rendering_paused = false;
                SCENE_LOG_INFO("Animation render resumed from Timeline.");
            }
        } else {
            if (UIWidgets::IconActionButton("TimelinePauseRender", UIWidgets::IconType::Pause, "Pause", false,
                                            ImVec4(0.95f, 0.78f, 0.30f, 1.0f), ImVec2(70.0f, 22.0f),
                                            "Pause animation render")) {
                rendering_paused = true;
                SCENE_LOG_INFO("Animation render paused from Timeline.");
            }
        }
        
        ImGui::SameLine();
        
        // STOP BUTTON - Always accessible during render!
        if (UIWidgets::IconActionButton("TimelineStopRender", UIWidgets::IconType::Stop, "Stop", false,
                                        ImVec4(1.0f, 0.42f, 0.42f, 1.0f), ImVec2(66.0f, 22.0f),
                                        "Stop animation render")) {
            rendering_stopped_cpu = true;
            rendering_stopped_gpu = true;
            SCENE_LOG_WARN("Animation render stopped from Timeline.");
        }
        
        ImGui::SameLine();
        ImGui::TextDisabled("| P=Pause  ESC=Stop");
        
        ImGui::PopStyleVar(2);
        return;  // Skip all other controls during render
    }
    
    // --- TOOLBAR BUTTONS ---
    bool has_selection = !selected_track.empty();
    bool has_keyframe_selected = has_selection && selected_keyframe_frame >= 0;

    // Keyframe buttons
    if (UIWidgets::IconActionButton("TimelineAddKey", UIWidgets::IconType::AddKey, "Key",
                                    false, ImVec4(0.42f, 0.86f, 1.0f, 1.0f), ImVec2(60.0f, 22.0f),
                                    "Insert keyframe", has_selection)) {
        insertKeyframeForTrack(ctx, selected_track, current_frame);
        tracks_dirty = true;
    }
    
    ImGui::SameLine();
    
    if (UIWidgets::IconActionButton("TimelineRemoveKey", UIWidgets::IconType::RemoveKey, "Delete",
                                    false, ImVec4(1.0f, 0.46f, 0.46f, 1.0f), ImVec2(72.0f, 22.0f),
                                    "Delete selected keyframe", has_keyframe_selected)) {
        deleteKeyframe(ctx, selected_track, selected_keyframe_frame);
        selected_keyframe_frame = -1;
        tracks_dirty = true;
    }
    
    ImGui::SameLine();
    
    if (UIWidgets::IconActionButton("TimelineDuplicateKey", UIWidgets::IconType::Duplicate, "Duplicate",
                                    false, ImVec4(0.86f, 0.68f, 1.0f, 1.0f), ImVec2(90.0f, 22.0f),
                                    "Duplicate keyframe (+10 frames)", has_keyframe_selected)) {
        duplicateKeyframe(ctx, selected_track, selected_keyframe_frame, selected_keyframe_frame + 10);
        tracks_dirty = true;
    }
    
    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();
    
    // --- HELP BUTTON ---
    if (UIWidgets::IconActionButton("TimelineHelp", UIWidgets::IconType::Help, "",
                                    false, ImVec4(0.72f, 0.80f, 0.92f, 1.0f), ImVec2(28.0f, 22.0f),
                                    "Timeline shortcuts and help")) {
        ImGui::OpenPopup("TimelineHelpPopup");
    }
    
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
    if (UIWidgets::IconActionButton("TimelinePlayPause",
                                    is_playing ? UIWidgets::IconType::Pause : UIWidgets::IconType::Play,
                                    "",
                                    is_playing,
                                    ImVec4(0.46f, 0.86f, 0.92f, 1.0f),
                                    ImVec2(28.0f, 22.0f),
                                    is_playing ? "Pause" : "Play")) {
        is_playing = !is_playing;
        // Pressing Play at the last frame (loop off) restarts from the start —
        // the backward jump replays from the sim cache, it does not re-bake.
        if (is_playing && !loop_enabled && current_frame >= end_frame)
            current_frame = start_frame;
    }
    
    ImGui::SameLine();
    
    // Stop
    if (UIWidgets::IconActionButton("TimelineStop", UIWidgets::IconType::Stop, "",
                                    false, ImVec4(1.0f, 0.46f, 0.46f, 1.0f), ImVec2(28.0f, 22.0f),
                                    "Stop")) {
        if (ctx.scene.anySimulationRuntimeEnabled()) {
            drainTimelineMutationBackends(ctx);
            ctx.scene.requestSimulationTimelineRenderResync();
        }
        is_playing = false;
        current_frame = start_frame;
        // Force animation re-evaluation at start_frame and backend update.
        // We do NOT call updateInstanceTransforms here directly because keyframes
        // haven't been evaluated at start_frame yet in this draw call.
        // Invalidating last_anim_update_frame causes the main animation loop
        // (just above, in draw()) to re-evaluate ALL tracks at start_frame
        // and then call updateInstanceTransforms with the correct positions.
        // The extern linkage lets us reach the static local defined in draw().
        // We reset via a workaround: last_anim_update_frame is static in draw(),
        // so use the ctx.scene.timeline.current_frame sentinel trick:
        ctx.scene.timeline.current_frame = -1; // force frame mismatch next draw
        ctx.renderer.resetCPUAccumulation();
        if (ctx.backend_ptr) {
            ctx.backend_ptr->resetAccumulation();
        }
        ctx.start_render = true;
    }

    // Loop toggle. Default OFF — looping a simulation re-bakes from frame 0 and
    // wipes the sim frame cache on every wrap (memory thrash), so it is opt-in.
    ImGui::SameLine();
    {
        const bool loop_on = loop_enabled;
        if (loop_on) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.46f, 0.86f, 0.92f, 1.0f));
        if (ImGui::Button(loop_on ? "Loop: On" : "Loop: Off", ImVec2(62.0f, 22.0f)))
            loop_enabled = !loop_enabled;
        if (loop_on) ImGui::PopStyleColor();
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Loop playback.\nOff: stop at the last frame (recommended for simulations —\nlooping replays from the sim cache instead of re-baking).");
    }

    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();

    // FPS
    ImGui::PushItemWidth(60);
    ImGui::SliderInt("FPS", &ctx.render_settings.animation_fps, 1, 60);
    ImGui::PopItemWidth();
    
    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();
    
    // === TIMELINE QUALITY PRESET (NEW!) ===
    ImGui::PushItemWidth(80);
    const char* quality_names[] = { "Draft", "Low", "Medium", "High" };
    int quality_idx = static_cast<int>(ctx.render_settings.timeline_quality_preset);
    if (ImGui::Combo("##Quality", &quality_idx, quality_names, IM_ARRAYSIZE(quality_names))) {
        ctx.render_settings.timeline_quality_preset = static_cast<TimelineQualityPreset>(quality_idx);
        // Update sample count from preset
        ctx.render_settings.animation_samples_per_frame = getTimelineSamplesFromPreset(ctx.render_settings.timeline_quality_preset);
    }
    ImGui::PopItemWidth();
    if (ImGui::IsItemHovered()) {
        const char* quality_tooltips[] = {
            "Draft: 1 sample - Fastest, for quick scrubbing",
            "Low: 4 samples - Basic preview quality",
            "Medium: 16 samples - Balanced quality/speed",
            "High: 64 samples - High quality preview"
        };
        ImGui::SetTooltip("%s", quality_tooltips[quality_idx]);
    }
    
    ImGui::SameLine();
    
    // === TIMELINE DENOISER TOGGLE (NEW!) ===
    ImGui::Checkbox("Denoise", &ctx.render_settings.timeline_use_denoiser);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Apply AI denoiser during timeline playback\n(May reduce framerate but improves quality)");
    
    ImGui::SameLine();
    ImGui::TextDisabled("| Zoom: %.1fx", zoom);
    
    // Show selected track info
    if (!selected_track.empty()) {
        ImGui::SameLine();
        ImGui::TextDisabled("| Track: %s", selected_track.c_str());
    }

    ImGui::PopStyleVar(2);
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
                // Generic recursive-ish drawing for Objects using depth
                for (size_t j = i + 1; j < tracks.size(); ++j) {
                    auto& sub = tracks[j];
                    if (sub.depth <= 0) break; // Finished this object block
                    
                    // Determine if visible based on parent expansion
                    // Simplified: if depth > 1, check if prev track (parent) was expanded
                    // Tracks are ordered perfectly.
                    // If sub.depth == 2, check if current Depth 1 parent is expanded.
                    // To do this right in a flat loop, we need to track state.
                    // BUT for now, let's just assume depth 1 is always expanded? No.
                    // We need to support expanding depth 1 tracks (Location/Rotation/Scale).
                    // So we must draw them as TreeNodes too if they have children.
                    
                    bool parent_expanded = true;
                    // Look back for parent
                    for (int k = (int)j - 1; k >= (int)i; --k) {
                        if (tracks[k].depth < sub.depth) {
                            parent_expanded = tracks[k].expanded;
                            break;
                        }
                    }
                    if (!parent_expanded) continue;

                    std::string full_track_name = sub.entity_name;
                    if (sub.depth == 1) full_track_name += "." + sub.name;
                    else if (sub.depth == 2) {
                         // Reconstruct name properly or rely on sub.name?
                         // sub.name is "X".
                         // We need "Location.X".
                         // Find parent name again
                         std::string parent_suffix = "";
                         for (int k = (int)j - 1; k >= (int)i; --k) {
                             if (tracks[k].depth == 1) {
                                 parent_suffix = tracks[k].name;
                                 break;
                             }
                         }
                         full_track_name += "." + parent_suffix + "." + sub.name;
                    }
                    
                    bool is_sel = (selected_track == full_track_name);
                    
                    ImGui::Indent(sub.depth * 10.0f);
                    
                    // If track has children (depth 1 Loc/Rot/Scale), use TreeNode
                    // Check if next track is depth 2
                    bool has_children = (j + 1 < tracks.size() && tracks[j+1].depth > sub.depth);
                    
                    if (has_children) {
                        ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth;
                         if (is_sel) flags |= ImGuiTreeNodeFlags_Selected;
                         if (sub.expanded) flags |= ImGuiTreeNodeFlags_DefaultOpen;
                         
                         bool open = ImGui::TreeNodeEx(sub.name.c_str(), flags);
                         if (ImGui::IsItemClicked()) selected_track = full_track_name;
                         sub.expanded = open; // Update expanded state in vector (might be reset on rebuild)
                         if (open) ImGui::TreePop(); // Pop immediately because we handle hierarchy via linear loop + indent
                    } else {
                        if (ImGui::Selectable(sub.name.c_str(), is_sel, 0, ImVec2(list_width - 60 - sub.depth*10, 18))) {
                            selected_track = full_track_name;
                        }
                    }

                    ImVec2 sub_pos = ImGui::GetItemRectMin();
                    draw_list->AddRectFilled(
                        ImVec2(sub_pos.x - 8, sub_pos.y + 2),
                        ImVec2(sub_pos.x - 2, sub_pos.y + 16),
                        sub.color, 2.0f);
                    ImGui::Unindent(sub.depth * 10.0f);
                }
                ImGui::TreePop();
            }
        }
        ImGui::TreePop();
    }
    
    // Group: Lights (with sub-tracks like Objects)
    if (ImGui::TreeNodeEx("Lights", ImGuiTreeNodeFlags_DefaultOpen)) {
        for (size_t t = 0; t < tracks.size(); ++t) {
            auto& track = tracks[t];
            if (track.group != TrackGroup::Lights || track.is_sub_track) continue;
            
            bool is_selected = (track.entity_name == selected_track);
            
            // Main track as TreeNode
            ImGuiTreeNodeFlags node_flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth;
            if (is_selected) node_flags |= ImGuiTreeNodeFlags_Selected;
            if (track.expanded) node_flags |= ImGuiTreeNodeFlags_DefaultOpen;
            
            bool node_open = ImGui::TreeNodeEx(track.name.c_str(), node_flags);
            
            if (ImGui::IsItemClicked()) {
                selected_track = track.entity_name;
            }
            
            // Color indicator
            ImVec2 item_min = ImGui::GetItemRectMin();
            draw_list->AddRectFilled(
                ImVec2(item_min.x - 12, item_min.y + 2),
                ImVec2(item_min.x - 4, item_min.y + 16),
                track.color, 2.0f);
            
            if (node_open) {
                // Find and show sub-tracks for this light
                for (size_t s = t + 1; s < tracks.size(); ++s) {
                    auto& sub = tracks[s];
                    if (sub.group != TrackGroup::Lights || !sub.is_sub_track) break;
                    if (sub.parent_entity != track.entity_name) break;
                    
                    std::string channel_track = track.entity_name + "." + sub.name;
                    bool sub_sel = (selected_track == channel_track);
                    
                    ImGui::Indent(10);
                    if (ImGui::Selectable(sub.name.c_str(), sub_sel, 0, ImVec2(list_width - 60, 18))) {
                        selected_track = channel_track;
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
    
    // Group: Cameras (with sub-tracks like Objects)
    if (ImGui::TreeNodeEx("Cameras", ImGuiTreeNodeFlags_DefaultOpen)) {
        for (size_t t = 0; t < tracks.size(); ++t) {
            auto& track = tracks[t];
            if (track.group != TrackGroup::Cameras || track.is_sub_track) continue;
            
            bool is_selected = (track.entity_name == selected_track);
            
            // Main track as TreeNode
            ImGuiTreeNodeFlags node_flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth;
            if (is_selected) node_flags |= ImGuiTreeNodeFlags_Selected;
            if (track.expanded) node_flags |= ImGuiTreeNodeFlags_DefaultOpen;
            
            bool node_open = ImGui::TreeNodeEx(track.name.c_str(), node_flags);
            
            if (ImGui::IsItemClicked()) {
                selected_track = track.entity_name;
            }
            
            // Color indicator
            ImVec2 item_min = ImGui::GetItemRectMin();
            draw_list->AddRectFilled(
                ImVec2(item_min.x - 12, item_min.y + 2),
                ImVec2(item_min.x - 4, item_min.y + 16),
                track.color, 2.0f);
            
            if (node_open) {
                // Find and show sub-tracks for this camera
                for (size_t s = t + 1; s < tracks.size(); ++s) {
                    auto& sub = tracks[s];
                    if (sub.group != TrackGroup::Cameras || !sub.is_sub_track) break;
                    if (sub.parent_entity != track.entity_name) break;
                    
                    std::string channel_track = track.entity_name + "." + sub.name;
                    bool sub_sel = (selected_track == channel_track);
                    
                    ImGui::Indent(10);
                    if (ImGui::Selectable(sub.name.c_str(), sub_sel, 0, ImVec2(list_width - 60, 18))) {
                        selected_track = channel_track;
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
    
    // Group: Terrain (morphing animation)
    if (ImGui::TreeNodeEx("Terrain", ImGuiTreeNodeFlags_DefaultOpen)) {
        for (auto& track : tracks) {
            if (track.group != TrackGroup::Terrain) continue;
            
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
    
    // Group: Water (wave animation)
    if (ImGui::TreeNodeEx("Water", ImGuiTreeNodeFlags_DefaultOpen)) {
        for (auto& track : tracks) {
            if (track.group != TrackGroup::Water) continue;
            
            bool is_selected = (track.entity_name == selected_track);
            if (ImGui::Selectable(track.name.c_str(), is_selected, 0, ImVec2(list_width - 20, track_height - 4))) {
                selected_track = track.entity_name;
            }
            
            ImVec2 item_min = ImGui::GetItemRectMin();
            draw_list->AddRectFilled(
                ImVec2(item_min.x - 12, item_min.y + 2),
                ImVec2(item_min.x - 4, item_min.y + 16),
                COLOR_WATER, 2.0f);
        }
        ImGui::TreePop();
    }
    
    // Group: Gas (emitter animation)
    if (ImGui::TreeNodeEx("Gas/Emitters", ImGuiTreeNodeFlags_DefaultOpen)) {
        for (auto& track : tracks) {
            if (track.group != TrackGroup::Gas) continue;
            
            bool is_selected = (track.entity_name == selected_track);
            if (ImGui::Selectable(track.name.c_str(), is_selected, 0, ImVec2(list_width - 20, track_height - 4))) {
                selected_track = track.entity_name;
            }
            
            ImVec2 item_min = ImGui::GetItemRectMin();
            draw_list->AddRectFilled(
                ImVec2(item_min.x - 12, item_min.y + 2),
                ImVec2(item_min.x - 4, item_min.y + 16),
                COLOR_GAS, 2.0f);
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
    
  
    // Helper to determine if a track is visible based on parent expansion
    auto isTrackVisible = [&](size_t index) -> bool {
        if (tracks[index].depth == 0) return true;
        // Search backwards for parent
        size_t d = tracks[index].depth;
        for (int i = (int)index - 1; i >= 0; --i) {
            if (tracks[i].depth < d) {
                if (!tracks[i].expanded) return false;
                d = tracks[i].depth;
                if (d == 0) return true;
            }
        }
        return true;
    };

    for (size_t i = 0; i < tracks.size(); ++i) {
        auto& track = tracks[i];
        if (!isTrackVisible(i)) continue;
        
        auto it = ctx.scene.timeline.tracks.find(track.entity_name);
        if (it != ctx.scene.timeline.tracks.end()) {
            for (auto& kf : it->second.keyframes) {
                int px = frameToPixelX(kf.frame, canvas_width);
                if (px >= -10 && px <= canvas_width + 10) {
                    float x = canvas_pos.x + px;
                    float base_y = canvas_pos.y + y_offset + track_height / 2;
                    
                    bool show_diamond = false;
                    ImU32 col = track.color;
                    
                    // Check flags based on channel
                    if (track.channel == ChannelType::None) show_diamond = true; // Main track always shows
                    else if (kf.has_transform) {
                        if (track.channel == ChannelType::Location) show_diamond = kf.transform.has_position;
                        else if (track.channel == ChannelType::LocationX) show_diamond = kf.transform.has_pos_x;
                        else if (track.channel == ChannelType::LocationY) show_diamond = kf.transform.has_pos_y;
                        else if (track.channel == ChannelType::LocationZ) show_diamond = kf.transform.has_pos_z;
                        
                        else if (track.channel == ChannelType::Rotation) show_diamond = kf.transform.has_rotation;
                        else if (track.channel == ChannelType::RotationX) show_diamond = kf.transform.has_rot_x;
                        else if (track.channel == ChannelType::RotationY) show_diamond = kf.transform.has_rot_y;
                        else if (track.channel == ChannelType::RotationZ) show_diamond = kf.transform.has_rot_z;
                        
                        else if (track.channel == ChannelType::Scale) show_diamond = kf.transform.has_scale;
                        else if (track.channel == ChannelType::ScaleX) show_diamond = kf.transform.has_scl_x;
                        else if (track.channel == ChannelType::ScaleY) show_diamond = kf.transform.has_scl_y;
                        else if (track.channel == ChannelType::ScaleZ) show_diamond = kf.transform.has_scl_z;
                    }
                    else if (kf.has_material && track.channel == ChannelType::Material) show_diamond = true;
                    else if (kf.has_anim_graph && track.channel == ChannelType::None) show_diamond = true;
                    else if (kf.has_anim_graph && track.is_sub_track) {
                        if (track.name == "AnimGraph") show_diamond = true;
                        else if (track.name.find("States") == 0) show_diamond = !kf.anim_graph.force_state.empty();
                        else if (track.name.find("Clips") == 0) show_diamond = !kf.anim_graph.clip_overrides.empty() || !kf.anim_graph.clip_speed_overrides.empty();
                        else if (track.name.find("Parameters") == 0) show_diamond = !kf.anim_graph.float_params.empty() || !kf.anim_graph.bool_params.empty() || !kf.anim_graph.int_params.empty();
                        else if (track.name == "Triggers") show_diamond = !kf.anim_graph.triggers.empty();
                    }
                    // Terrain keyframes (for morphing animation)
                    else if (kf.has_terrain && track.group == TrackGroup::Terrain) show_diamond = true;
                    // Water keyframes (for wave parameter animation)
                    else if (kf.has_water && track.entity_name.find("Water_") == 0) show_diamond = true;

                    if (show_diamond) {
                         // Build specific track ID for selection
                         std::string track_id = track.entity_name;
                         if (track.channel != ChannelType::None) track_id += "." + track.name; 
                         // Note: Sub-sub tracks logic: Entity.Location.X ?
                         // parseTrackName handles "Entity.Suffix". 
                         // "Location" track name is "Location". "X" track name is "X".
                         // Current parseTrackName assumes "Entity.Suffix".
                         // I need to adjust selected_track naming to be robust.
                         // Let's rely on drawTrackList to set selected_track accurately.
                         
                         // The Loop above uses standard selected_track comparison
                         // I need to reconstruct the full name to match selected_track?
                         // Getting full path from hierarchy is hard here.
                         // But wait, my tracks have unique names? No, "X" is repeated.
                         // I should store `full_path` in TimelineTrack!
                         // For now, I will construct it:
                         std::string full_track_name = track.entity_name;
                         if (track.channel != ChannelType::None) {
                             // This is imperfect for X/Y/Z unless I store it.
                             // Hack: If channel is X, assume Parent.X ?
                             // I will update parseTrackName to handle "Location.X".
                             // OR simpler: `tracks` logic in drawTrackList determines the name.
                             // I will assume `selected_track` matches the name I generate here.
                             if (track.depth == 1) full_track_name += "." + track.name;
                             if (track.depth == 2) {
                                  // Find parent name
                                  // Hacky look-back
                                  // Let's assume standard names:
                                  if (track.channel == ChannelType::LocationX) full_track_name += ".Location.X";
                                  else if (track.channel == ChannelType::LocationY) full_track_name += ".Location.Y";
                                  else if (track.channel == ChannelType::LocationZ) full_track_name += ".Location.Z";
                                  else if (track.channel == ChannelType::RotationX) full_track_name += ".Rotation.X";
                                  else if (track.channel == ChannelType::RotationY) full_track_name += ".Rotation.Y";
                                  else if (track.channel == ChannelType::RotationZ) full_track_name += ".Rotation.Z";
                                  else if (track.channel == ChannelType::ScaleX) full_track_name += ".Scale.X";
                                  else if (track.channel == ChannelType::ScaleY) full_track_name += ".Scale.Y";
                                  else if (track.channel == ChannelType::ScaleZ) full_track_name += ".Scale.Z";
                             }
                         }

                        bool is_sel = (full_track_name == selected_track && kf.frame == selected_keyframe_frame);
                        // Guard hover check with ImGui::IsWindowHovered() to prevent clicks/hover behind overlapping windows
                        bool is_hov = ImGui::IsWindowHovered() && ImGui::IsMouseHoveringRect(ImVec2(x - 5, base_y - 5), ImVec2(x + 5, base_y + 5));
                        
                        drawKeyframeDiamond(draw_list, x, base_y, is_hov ? IM_COL32_WHITE : track.color, is_sel);
                        
                        if (is_hov) {
                            any_keyframe_hovered = true;
                            
                            // Left click to select and drag keyframe
                            if (ImGui::IsMouseClicked(0)) {
                                selected_track = full_track_name;
                                selected_keyframe_frame = kf.frame;
                                is_dragging_keyframe = true;
                                drag_start_frame = kf.frame;
                            }
                            
                            // Right click to open context menu for keyframe
                            if (ImGui::IsMouseClicked(1)) {
                                selected_track = full_track_name;
                                selected_keyframe_frame = kf.frame;
                                ImGui::OpenPopup("KeyframeContextMenu");
                            }
                        }
                    }
                }
            }
        }
        y_offset += track_height;
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
    
    // Process playhead scrubbing after checking keyframe drag/clicks to prevent overlap
    handleScrubbing(canvas_pos, canvas_width);
    
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
        ImGui::TextWrapped("AnimGraph keys apply runtime graph parameters, triggers, and force-state data on this frame.");
        ImGui::TextDisabled("Tip: Right click empty space on an imported character track to add an AnimGraph keyframe.");
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
                    if (kf.has_anim_graph) ImGui::BulletText("AnimGraph");
                    ImGui::Separator();

                    if (kf.has_anim_graph) {
                        ImGui::TextDisabled("AnimGraph Payload");
                        for (const auto& [name, value] : kf.anim_graph.float_params) {
                            ImGui::BulletText("%s = %.3f", name.c_str(), value);
                        }
                        for (const auto& [name, value] : kf.anim_graph.bool_params) {
                            ImGui::BulletText("%s = %s", name.c_str(), value ? "true" : "false");
                        }
                        for (const auto& [name, value] : kf.anim_graph.int_params) {
                            ImGui::BulletText("%s = %d", name.c_str(), value);
                        }
                        for (const auto& [nodeId, clipName] : kf.anim_graph.clip_overrides) {
                            ImGui::BulletText("Clip Node %u -> %s", nodeId, clipName.c_str());
                        }
                        for (const auto& [nodeId, clipSpeed] : kf.anim_graph.clip_speed_overrides) {
                            ImGui::BulletText("Clip Speed %u = %.3f", nodeId, clipSpeed);
                        }
                        for (const auto& trigger : kf.anim_graph.triggers) {
                            ImGui::BulletText("%s [trigger]", trigger.c_str());
                        }
                        if (!kf.anim_graph.force_state.empty()) {
                            ImGui::BulletText("Force State -> %s", kf.anim_graph.force_state.c_str());
                        }
                        ImGui::Separator();

                        if (ImGui::TreeNodeEx("Edit AnimGraph", ImGuiTreeNodeFlags_DefaultOpen)) {
                            ImGui::TextDisabled("This editor modifies Timeline payload only. It does not change the asset graph.");
                            for (auto itFloat = kf.anim_graph.float_params.begin(); itFloat != kf.anim_graph.float_params.end(); ) {
                                ImGui::PushID(itFloat->first.c_str());
                                ImGui::DragFloat(itFloat->first.c_str(), &itFloat->second, 0.01f);
                                ImGui::SameLine();
                                if (ImGui::SmallButton("X")) itFloat = kf.anim_graph.float_params.erase(itFloat);
                                else ++itFloat;
                                ImGui::PopID();
                            }

                            for (auto itBool = kf.anim_graph.bool_params.begin(); itBool != kf.anim_graph.bool_params.end(); ) {
                                ImGui::PushID(itBool->first.c_str());
                                ImGui::Checkbox(itBool->first.c_str(), &itBool->second);
                                ImGui::SameLine();
                                if (ImGui::SmallButton("X")) itBool = kf.anim_graph.bool_params.erase(itBool);
                                else ++itBool;
                                ImGui::PopID();
                            }

                            for (auto itInt = kf.anim_graph.int_params.begin(); itInt != kf.anim_graph.int_params.end(); ) {
                                ImGui::PushID(itInt->first.c_str());
                                ImGui::InputInt(itInt->first.c_str(), &itInt->second);
                                ImGui::SameLine();
                                if (ImGui::SmallButton("X")) itInt = kf.anim_graph.int_params.erase(itInt);
                                else ++itInt;
                                ImGui::PopID();
                            }

                            for (auto itClip = kf.anim_graph.clip_overrides.begin(); itClip != kf.anim_graph.clip_overrides.end(); ) {
                                ImGui::PushID(static_cast<int>(itClip->first));
                                char clipBuf[128] = {};
                                strncpy(clipBuf, itClip->second.c_str(), sizeof(clipBuf) - 1);
                                std::string label = "Clip Node " + std::to_string(itClip->first);
                                if (ImGui::InputText(label.c_str(), clipBuf, sizeof(clipBuf))) {
                                    itClip->second = clipBuf;
                                }
                                ImGui::SameLine();
                                if (ImGui::SmallButton("X")) itClip = kf.anim_graph.clip_overrides.erase(itClip);
                                else ++itClip;
                                ImGui::PopID();
                            }

                            for (auto itSpeed = kf.anim_graph.clip_speed_overrides.begin(); itSpeed != kf.anim_graph.clip_speed_overrides.end(); ) {
                                ImGui::PushID(static_cast<int>(itSpeed->first) + 100000);
                                std::string label = "Clip Speed " + std::to_string(itSpeed->first);
                                ImGui::DragFloat(label.c_str(), &itSpeed->second, 0.01f, -10.0f, 10.0f);
                                ImGui::SameLine();
                                if (ImGui::SmallButton("X")) itSpeed = kf.anim_graph.clip_speed_overrides.erase(itSpeed);
                                else ++itSpeed;
                                ImGui::PopID();
                            }

                            static char newAgName[64] = "";
                            static float newAgFloat = 0.0f;
                            static int newAgInt = 0;
                            static bool newAgBool = false;
                            static int newAgType = 0;
                            static int newClipNodeId = 0;
                            static char newClipName[128] = "";
                            static float newClipSpeed = 1.0f;
                            const char* agTypes[] = { "Float", "Bool", "Int", "Trigger", "Clip Override", "Clip Speed" };
                            ImGui::InputText("Param Name", newAgName, sizeof(newAgName));
                            ImGui::Combo("Param Type", &newAgType, agTypes, IM_ARRAYSIZE(agTypes));
                            if (newAgType == 0) ImGui::DragFloat("Float Value", &newAgFloat, 0.01f);
                            if (newAgType == 1) ImGui::Checkbox("Bool Value", &newAgBool);
                            if (newAgType == 2) ImGui::InputInt("Int Value", &newAgInt);
                            if (newAgType == 4 || newAgType == 5) ImGui::InputInt("Clip Node Id", &newClipNodeId);
                            if (newAgType == 4) ImGui::InputText("Clip Name", newClipName, sizeof(newClipName));
                            if (newAgType == 5) ImGui::DragFloat("Clip Speed", &newClipSpeed, 0.01f, -10.0f, 10.0f);
                            bool canAddAnimPayload =
                                (newAgType <= 3 && strlen(newAgName) > 0) ||
                                (newAgType == 4 && newClipNodeId > 0 && strlen(newClipName) > 0) ||
                                (newAgType == 5 && newClipNodeId > 0);
                            if (ImGui::Button("Add Anim Param") && canAddAnimPayload) {
                                if (newAgType == 0) kf.anim_graph.float_params[newAgName] = newAgFloat;
                                else if (newAgType == 1) kf.anim_graph.bool_params[newAgName] = newAgBool;
                                else if (newAgType == 2) kf.anim_graph.int_params[newAgName] = newAgInt;
                                else if (newAgType == 3) kf.anim_graph.triggers.push_back(newAgName);
                                else if (newAgType == 4 && newClipNodeId > 0 && strlen(newClipName) > 0) kf.anim_graph.clip_overrides[(uint32_t)newClipNodeId] = newClipName;
                                else if (newAgType == 5 && newClipNodeId > 0) kf.anim_graph.clip_speed_overrides[(uint32_t)newClipNodeId] = newClipSpeed;
                                newAgName[0] = '\0';
                                newClipName[0] = '\0';
                            }

                            for (size_t trigIdx = 0; trigIdx < kf.anim_graph.triggers.size(); ) {
                                ImGui::PushID(static_cast<int>(trigIdx));
                                ImGui::Text("%s [trigger]", kf.anim_graph.triggers[trigIdx].c_str());
                                ImGui::SameLine();
                                if (ImGui::SmallButton("X")) {
                                    kf.anim_graph.triggers.erase(kf.anim_graph.triggers.begin() + trigIdx);
                                } else {
                                    ++trigIdx;
                                }
                                ImGui::PopID();
                            }

                            char forceStateBuf[64] = {};
                            strncpy(forceStateBuf, kf.anim_graph.force_state.c_str(), sizeof(forceStateBuf) - 1);
                            if (ImGui::InputText("Force State", forceStateBuf, sizeof(forceStateBuf))) {
                                kf.anim_graph.force_state = forceStateBuf;
                            }

                            ImGui::TreePop();
                        }
                    }
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
        ImGui::TextDisabled("Timeline Insert");
        ImGui::TextWrapped("If an imported character is selected, you can add an AnimGraph key here. During playback it is applied to that character's runtime graph.");
        ImGui::Separator();
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
            if (findImportedModelContextByName(ctx.scene, selected_track) && ImGui::MenuItem("AnimGraph Keyframe")) {
                current_frame = context_menu_frame;
                Keyframe kf(context_menu_frame);
                kf.has_anim_graph = true;
                ctx.scene.timeline.insertKeyframe(selected_track, kf);
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
    
    // Check if mouse is over canvas and the window itself is hovered (not covered/blocked by other windows)
    bool hovered = ImGui::IsWindowHovered() && ImGui::IsMouseHoveringRect(canvas_pos, 
        ImVec2(canvas_pos.x + canvas_size.x, canvas_pos.y + canvas_size.y));
    
    if (hovered) {
        // Zoom with mouse wheel
        if (io.MouseWheel != 0) {
            float zoom_factor = (io.MouseWheel > 0) ? 1.1f : 0.9f;
            zoom = std::clamp(zoom * zoom_factor, 0.1f, 10.0f);
        }
    }
    
    // Pan with middle mouse button (allow drag continuation outside)
    static bool is_panning = false;
    if (ImGui::IsMouseClicked(2)) {
        if (hovered) {
            is_panning = true;
        }
    }
    if (!io.MouseDown[2]) {
        is_panning = false;
    }
    
    if (is_panning) {
        pan_offset -= io.MouseDelta.x / (zoom * 10.0f);
        pan_offset = std::clamp(pan_offset, 0.0f, (float)(end_frame - start_frame));
    }
}

// ============================================================================
// SCRUBBING (CLICK TO SET FRAME)
// ============================================================================
void TimelineWidget::handleScrubbing(ImVec2 canvas_pos, float canvas_width) {
    // RENDER LOCK: Don't allow scrubbing during animation render
    if (rendering_in_progress) {
        return;
    }
    
    ImGuiIO& io = ImGui::GetIO();
    
    // BUGFIX: Don't scrub if ImGui is capturing mouse (e.g., dropdown/popup open)
    if (ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel)) {
        return;
    }
    
    // Avoid scrubbing if we are actively dragging keyframes or curve handles
    if (is_dragging_keyframe || graph_drag_mode != 0) {
        return;
    }
    
    static bool is_scrubbing = false;
    
    if (ImGui::IsMouseClicked(0)) {
        ImVec2 mouse = io.MousePos;
        float playhead_x = canvas_pos.x + frameToPixelX(current_frame, canvas_width);
        
        bool in_header = (mouse.y >= canvas_pos.y && mouse.y <= canvas_pos.y + header_height);
        // Grab within 8 pixels of playhead (anywhere vertically)
        bool near_playhead = (std::abs(mouse.x - playhead_x) <= 8.0f);
        // Alt + click anywhere in canvas
        bool alt_click = io.KeyAlt;
        
        if (ImGui::IsWindowHovered() &&
            mouse.x >= canvas_pos.x && mouse.x <= canvas_pos.x + canvas_width &&
            mouse.y >= canvas_pos.y && mouse.y <= canvas_pos.y + ImGui::GetWindowHeight()) {
            
            if (in_header || near_playhead || alt_click) {
                is_scrubbing = true;
            }
        }
    }
    
    if (!io.MouseDown[0]) {
        is_scrubbing = false;
    }
    
    if (is_scrubbing) {
        ImVec2 mouse = io.MousePos;
        current_frame = pixelXToFrame(mouse.x - canvas_pos.x, canvas_width);
        current_frame = std::clamp(current_frame, start_frame, end_frame);
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
            selected_entity = ctx.selection.selected.object->getNodeName();
            if (selected_entity.empty()) {
                selected_entity = "Object_" + std::to_string(ctx.selection.selected.object_index);
            }
            selected_entity = resolveCharacterTrackName(ctx.scene, selected_entity);
            selected_group = TrackGroup::Objects;
        } else if (ctx.selection.selected.type == SelectableType::Light && ctx.selection.selected.light) {
            selected_entity = ctx.selection.selected.light->nodeName;
            selected_group = TrackGroup::Lights;
        } else if (ctx.selection.selected.type == SelectableType::Camera && ctx.selection.selected.camera) {
            selected_entity = ctx.selection.selected.camera->nodeName;
            selected_group = TrackGroup::Cameras;
        }
    }

    const bool focus_selected_entity = !selected_entity.empty();
    
    std::set<std::string> added_entities;
    
    // --- OPTIMIZATION: Skip building valid_entities from ALL objects ---
    // The old code iterated through ALL 10M+ objects just to validate deleted entities.
    // This caused 2+ second freezes when tracks_dirty was set.
    // 
    // NEW APPROACH: We don't validate entity existence. If an entity was deleted,
    // its track will remain in the timeline until the user deletes the keyframes.
    // This is acceptable trade-off for massive performance improvement.
    //
    // For lights and cameras, we still iterate (small lists - typically <100 items):
    std::set<std::string> valid_entities;
    
    // Add all lights (very small list - typically 1-20 lights)
    for (const auto& light : ctx.scene.lights) {
        if (!light->nodeName.empty()) {
            valid_entities.insert(light->nodeName);
        }
    }
    
    // Add all cameras (very small list - typically 1-5 cameras)
    for (const auto& cam : ctx.scene.cameras) {
        if (!cam->nodeName.empty()) {
            valid_entities.insert(cam->nodeName);
        }
    }
    
    // World is always valid
    valid_entities.insert("World");
    
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
        
        // Helper to add a sub-track
        auto addSub = [&](std::string name, ChannelType type, int depth, ImU32 col) {
            TimelineTrack sub;
            sub.entity_name = entity_name;
            sub.parent_entity = entity_name;
            sub.name = name;
            sub.group = TrackGroup::Objects;
            sub.channel = type;
            sub.color = col;
            sub.is_sub_track = true;
            sub.depth = depth;
            sub.keyframe_frames = keyframes;
            tracks.push_back(sub);
        };

        // Location Group
        addSub("Location", ChannelType::Location, 1, IM_COL32(255, 100, 100, 255));
        addSub("X", ChannelType::LocationX, 2, IM_COL32(255, 80, 80, 255));
        addSub("Y", ChannelType::LocationY, 2, IM_COL32(80, 255, 80, 255));
        addSub("Z", ChannelType::LocationZ, 2, IM_COL32(80, 80, 255, 255));

        // Rotation Group
        addSub("Rotation", ChannelType::Rotation, 1, IM_COL32(100, 255, 100, 255));
        addSub("X", ChannelType::RotationX, 2, IM_COL32(255, 80, 80, 255));
        addSub("Y", ChannelType::RotationY, 2, IM_COL32(80, 255, 80, 255));
        addSub("Z", ChannelType::RotationZ, 2, IM_COL32(80, 80, 255, 255));

        // Scale Group
        addSub("Scale", ChannelType::Scale, 1, IM_COL32(100, 100, 255, 255));
        addSub("X", ChannelType::ScaleX, 2, IM_COL32(255, 80, 80, 255));
        addSub("Y", ChannelType::ScaleY, 2, IM_COL32(80, 255, 80, 255));
        addSub("Z", ChannelType::ScaleZ, 2, IM_COL32(80, 80, 255, 255));
        
        // Material
        addSub("Material", ChannelType::Material, 1, IM_COL32(255, 180, 50, 255));
    };
    
    auto appendTimelineEntityTracks = [&](const std::string& entity_name, const auto& track) {
        if (track.keyframes.empty()) return;
        
        // OPTIMIZATION: Only validate lights/cameras/world (small lists).
        // We skipped object validation because iterating 10M+ objects is too slow.
        // Object tracks are always shown - user must manually delete orphan tracks.
        // Only skip lights/cameras that don't exist anymore (valid_entities only has these)
        bool is_light_or_camera = (valid_entities.find(entity_name) != valid_entities.end());
        bool is_world = (entity_name == "World");
        
        // For non-World, non-light/camera entities: assume valid (these are objects)
        // We don't check objects against valid_entities because that was the slow part
        
        // Determine group based on keyframe types
        bool has_transform = false, has_material = false, has_light = false, has_camera = false, has_world = false, has_terrain = false, has_emitter = false, has_anim_graph = false;
        std::vector<int> keyframes;
        
        for (auto& kf : track.keyframes) {
            has_transform |= kf.has_transform;
            has_material |= kf.has_material;
            has_light |= kf.has_light;
            has_camera |= kf.has_camera;
            has_world |= kf.has_world;
            has_terrain |= kf.has_terrain;
            has_emitter |= kf.has_emitter;
            has_anim_graph |= kf.has_anim_graph;
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
            // Camera with sub-tracks (Position, Target, FOV, Focus, Aperture)
            TimelineTrack main_track;
            main_track.entity_name = entity_name;
            main_track.name = entity_name;
            main_track.group = TrackGroup::Cameras;
            main_track.channel = ChannelType::None;
            main_track.color = COLOR_CAMERA;
            main_track.expanded = true;
            main_track.is_sub_track = false;
            main_track.depth = 0;
            main_track.keyframe_frames = keyframes;
            tracks.push_back(main_track);
            
            // Position sub-track
            TimelineTrack pos_track;
            pos_track.entity_name = entity_name;
            pos_track.parent_entity = entity_name;
            pos_track.name = "Position";
            pos_track.group = TrackGroup::Cameras;
            pos_track.channel = ChannelType::Location;
            pos_track.color = IM_COL32(255, 100, 100, 255);  // Red
            pos_track.is_sub_track = true;
            pos_track.depth = 1;
            pos_track.keyframe_frames = keyframes;
            tracks.push_back(pos_track);
            
            // Target sub-track
            TimelineTrack target_track;
            target_track.entity_name = entity_name;
            target_track.parent_entity = entity_name;
            target_track.name = "Target";
            target_track.group = TrackGroup::Cameras;
            target_track.channel = ChannelType::Rotation;  // Reuse for target
            target_track.color = IM_COL32(100, 255, 100, 255);  // Green
            target_track.is_sub_track = true;
            target_track.depth = 1;
            target_track.keyframe_frames = keyframes;
            tracks.push_back(target_track);
            
            // FOV sub-track
            TimelineTrack fov_track;
            fov_track.entity_name = entity_name;
            fov_track.parent_entity = entity_name;
            fov_track.name = "FOV";
            fov_track.group = TrackGroup::Cameras;
            fov_track.channel = ChannelType::Scale;  // Reuse for FOV
            fov_track.color = IM_COL32(100, 100, 255, 255);  // Blue
            fov_track.is_sub_track = true;
            fov_track.depth = 1;
            fov_track.keyframe_frames = keyframes;
            tracks.push_back(fov_track);
            
            // Focus/Aperture sub-track
            TimelineTrack focus_track;
            focus_track.entity_name = entity_name;
            focus_track.parent_entity = entity_name;
            focus_track.name = "Focus/DOF";
            focus_track.group = TrackGroup::Cameras;
            focus_track.channel = ChannelType::Material;  // Reuse for DOF
            focus_track.color = IM_COL32(200, 150, 255, 255);  // Purple
            focus_track.is_sub_track = true;
            focus_track.depth = 1;
            focus_track.keyframe_frames = keyframes;
            tracks.push_back(focus_track);
            
        } else if (has_light) {
            // Light with sub-tracks (Position, Color, Intensity, Direction)
            TimelineTrack main_track;
            main_track.entity_name = entity_name;
            main_track.name = entity_name;
            main_track.group = TrackGroup::Lights;
            main_track.channel = ChannelType::None;
            main_track.color = COLOR_LIGHT;
            main_track.expanded = true;
            main_track.is_sub_track = false;
            main_track.depth = 0;
            main_track.keyframe_frames = keyframes;
            tracks.push_back(main_track);
            
            // Position sub-track
            TimelineTrack pos_track;
            pos_track.entity_name = entity_name;
            pos_track.parent_entity = entity_name;
            pos_track.name = "Position";
            pos_track.group = TrackGroup::Lights;
            pos_track.channel = ChannelType::Location;
            pos_track.color = IM_COL32(255, 100, 100, 255);  // Red
            pos_track.is_sub_track = true;
            pos_track.depth = 1;
            pos_track.keyframe_frames = keyframes;
            tracks.push_back(pos_track);
            
            // Color sub-track
            TimelineTrack color_track;
            color_track.entity_name = entity_name;
            color_track.parent_entity = entity_name;
            color_track.name = "Color";
            color_track.group = TrackGroup::Lights;
            color_track.channel = ChannelType::Material;  // Reuse for color
            color_track.color = IM_COL32(255, 200, 100, 255);  // Orange
            color_track.is_sub_track = true;
            color_track.depth = 1;
            color_track.keyframe_frames = keyframes;
            tracks.push_back(color_track);
            
            // Intensity sub-track
            TimelineTrack intensity_track;
            intensity_track.entity_name = entity_name;
            intensity_track.parent_entity = entity_name;
            intensity_track.name = "Intensity";
            intensity_track.group = TrackGroup::Lights;
            intensity_track.channel = ChannelType::Scale;  // Reuse for intensity
            intensity_track.color = IM_COL32(255, 255, 100, 255);  // Yellow
            intensity_track.is_sub_track = true;
            intensity_track.depth = 1;
            intensity_track.keyframe_frames = keyframes;
            tracks.push_back(intensity_track);
            
            // Direction sub-track
            TimelineTrack dir_track;
            dir_track.entity_name = entity_name;
            dir_track.parent_entity = entity_name;
            dir_track.name = "Direction";
            dir_track.group = TrackGroup::Lights;
            dir_track.channel = ChannelType::Rotation;  // Reuse for direction
            dir_track.color = IM_COL32(100, 255, 100, 255);  // Green
            dir_track.is_sub_track = true;
            dir_track.depth = 1;
            dir_track.keyframe_frames = keyframes;
            tracks.push_back(dir_track);
            
        } else if (has_terrain) {
            // Terrain track (morphing animation)
            TimelineTrack t;
            t.entity_name = entity_name;
            t.name = entity_name;
            t.group = TrackGroup::Terrain;
            t.color = COLOR_TERRAIN;
            t.keyframe_frames = keyframes;
            tracks.push_back(t);
        } else if (entity_name.find("Water_") == 0) {
            // Water surface track (wave parameter animation)
            TimelineTrack t;
            t.entity_name = entity_name;
            t.name = entity_name;  // e.g. "Water_1"
            t.group = TrackGroup::Water;
            t.color = COLOR_WATER;
            t.keyframe_frames = keyframes;
            tracks.push_back(t);
        } else {
            // Check if it belongs to a Gas Volume (Volume itself OR its Emitters)
            bool is_gas_related = has_emitter;
            if (!is_gas_related) {
                // Check if it's explicitly a Gas Volume or Emitter by name
                if (entity_name.find("Gas") != std::string::npos || entity_name.find("Emitter") != std::string::npos) {
                    is_gas_related = true;
                } else {
                    for (auto& gas : ctx.scene.gas_volumes) {
                        if (entity_name == gas->getName() || entity_name.find(gas->getName() + "_") == 0) {
                            is_gas_related = true;
                            break;
                        }
                    }
                }
            }

            if (is_gas_related) {
                TimelineTrack t;
                t.entity_name = entity_name;
                t.name = entity_name;
                t.group = TrackGroup::Gas;
                t.color = COLOR_GAS;
                t.keyframe_frames = keyframes;
                t.expanded = true;
                tracks.push_back(t);
            } else if (has_transform || has_material) {
                // Object with L/R/S sub-tracks
                bool is_selected = (entity_name == selected_entity);
                ImU32 color = has_material ? COLOR_MATERIAL : COLOR_TRANSFORM;
                addObjectWithChannels(entity_name, entity_name, color, is_selected, keyframes);
            } else if (has_anim_graph) {
                addAnimGraphTrackTree(tracks, entity_name, track.keyframes);
            }
        }
        
        added_entities.insert(entity_name);
    };

    // --- ADD TRACKS FROM TIMELINE (entities with keyframes) ---
    if (focus_selected_entity) {
        auto selected_it = ctx.scene.timeline.tracks.find(selected_entity);
        if (selected_it != ctx.scene.timeline.tracks.end()) {
            appendTimelineEntityTracks(selected_it->first, selected_it->second);
        }
    } else {
        for (auto& [entity_name, track] : ctx.scene.timeline.tracks) {
            appendTimelineEntityTracks(entity_name, track);
        }
    }
    
    // --- ADD CURRENTLY SELECTED ENTITY (if not already added) ---
    // NOTE: Skip World here - it's always added below to ensure it always exists
    if (!selected_entity.empty() && selected_entity != "World" && added_entities.find(selected_entity) == added_entities.end()) {
        if (selected_group == TrackGroup::Objects) {
            if (findImportedModelContextByName(ctx.scene, selected_entity)) {
                addAnimGraphTrackTree(tracks, selected_entity, {});
            } else {
                // Object with L/R/S sub-tracks
                addObjectWithChannels(selected_entity, selected_entity + " (selected)", COLOR_TRANSFORM, true, {});
            }
        } else if (selected_group == TrackGroup::Terrain) {
            // Terrain track (for keying)
            TimelineTrack t;
            t.entity_name = selected_entity;
            t.name = selected_entity + " (selected)";
            t.group = TrackGroup::Terrain;
            t.color = COLOR_TERRAIN;
            tracks.push_back(t);
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
// LIGHTWEIGHT SELECTION SYNC - Runs every frame with minimal overhead
// ============================================================================
void TimelineWidget::handleSelectionSync(UIContext& ctx) {
    // PERFORMANCE: This function is called every frame, so keep it minimal
    
    // Get current viewport selection name
    std::string viewport_selection;
    if (ctx.selection.hasSelection()) {
        if (ctx.selection.selected.type == SelectableType::Object && ctx.selection.selected.object) {
            viewport_selection = ctx.selection.selected.object->getNodeName();
            if (viewport_selection.empty()) {
                viewport_selection = "Object_" + std::to_string(ctx.selection.selected.object_index);
            }
            viewport_selection = resolveCharacterTrackName(ctx.scene, viewport_selection);
        } else if (ctx.selection.selected.type == SelectableType::Light && ctx.selection.selected.light) {
            viewport_selection = ctx.selection.selected.light->nodeName;
        } else if (ctx.selection.selected.type == SelectableType::Camera && ctx.selection.selected.camera) {
            viewport_selection = ctx.selection.selected.camera->nodeName;
        }
    }
    
    // PERFORMANCE: Only update and mark dirty if selection actually changed.
    // selection_sync_force_ lets a panel that hijacked selected_track (e.g. World)
    // hand it back to the live selection on release, even when the selection itself
    // didn't change (otherwise re-selecting the same object wouldn't restore it and
    // keying objects would stay broken for the whole session).
    if (viewport_selection != last_selection_ || selection_sync_force_) {
        last_selection_ = viewport_selection;
        selection_sync_force_ = false;

        // Update selected track
        if (!viewport_selection.empty()) {
            selected_track = viewport_selection;
        }

        // PERFORMANCE: Only mark dirty when selection changes, not every frame
        tracks_dirty = true;
    }
    
    // --- KEYBOARD SHORTCUTS (lightweight, runs every frame) ---
    bool timeline_focused = ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows);
    ImGuiIO& io = ImGui::GetIO();
    
    // I key - Insert keyframe (global shortcut)
    if (ImGui::IsKeyPressed(ImGuiKey_I) && !io.WantTextInput) {
        if (!selected_track.empty()) {
            insertKeyframeForTrack(ctx, selected_track, current_frame);
            tracks_dirty = true;
        }
    }
    
    // Delete/X key - Delete selected keyframe (timeline focused only)
    if (timeline_focused && (ImGui::IsKeyPressed(ImGuiKey_Delete) || ImGui::IsKeyPressed(ImGuiKey_X)) && !io.WantTextInput) {
        if (!selected_track.empty() && selected_keyframe_frame >= 0) {
            deleteKeyframe(ctx, selected_track, selected_keyframe_frame);
            selected_keyframe_frame = -1;
            tracks_dirty = true;
        }
    }
}

// ============================================================================
// SYNC FROM IMPORTED ANIMATION DATA - Called ONCE on first frame
// ============================================================================
void TimelineWidget::syncFromAnimationData(UIContext& ctx) {
    // Sync any AnimationData entries not yet converted to timeline keyframes.
    // Uses a member counter so reset() clears it on project reload.
    const size_t currentCount = ctx.scene.animationDataList.size();
    if (currentCount <= lastSyncedAnimCount) return;

    for (size_t animIdx = lastSyncedAnimCount; animIdx < currentCount; ++animIdx) {
        const auto& anim = ctx.scene.animationDataList[animIdx];
        if (!anim) continue;
        {
            double tps = anim->ticksPerSecond > 0 ? anim->ticksPerSecond : 24.0;
            
            // Process Position Keys
            for (const auto& [nodeName, keys] : anim->positionKeys) {
                for (const auto& key : keys) {
                    int frame = static_cast<int>(std::round(key.mTime / tps * 24.0));
                    
                    // Get or create keyframe at this frame
                    auto& track = ctx.scene.timeline.tracks[nodeName];
                    Keyframe* existing = track.getKeyframeAt(frame);
                    if (existing) {
                        existing->transform.position = Vec3(key.mValue.x, key.mValue.y, key.mValue.z);
                        existing->transform.has_position = true;
                        existing->transform.has_pos_x = true; existing->transform.has_pos_y = true; existing->transform.has_pos_z = true;
                        existing->has_transform = true;
                    } else {
                        Keyframe kf(frame);
                        kf.has_transform = true;
                        kf.transform.position = Vec3(key.mValue.x, key.mValue.y, key.mValue.z);
                        kf.transform.has_position = true;
                        kf.transform.has_pos_x = true; kf.transform.has_pos_y = true; kf.transform.has_pos_z = true;
                        kf.transform.has_rotation = false;
                        kf.transform.has_scale = false;
                        track.addKeyframe(kf);
                    }
                }
            }
            
            // Process Rotation Keys (Quaternion -> Euler)
            for (const auto& [nodeName, keys] : anim->rotationKeys) {
                for (const auto& key : keys) {
                    int frame = static_cast<int>(std::round(key.mTime / tps * 24.0));
                    
                    // Convert quaternion to Euler angles (degrees)
                    float qx = key.mValue.x, qy = key.mValue.y, qz = key.mValue.z, qw = key.mValue.w;
                    
                    float sinr_cosp = 2.0f * (qw * qx + qy * qz);
                    float cosr_cosp = 1.0f - 2.0f * (qx * qx + qy * qy);
                    float rx = std::atan2(sinr_cosp, cosr_cosp);
                    
                    float sinp = 2.0f * (qw * qy - qz * qx);
                    float ry = (std::abs(sinp) >= 1.0f) ? std::copysign(3.14159f / 2.0f, sinp) : std::asin(sinp);
                    
                    float siny_cosp = 2.0f * (qw * qz + qx * qy);
                    float cosy_cosp = 1.0f - 2.0f * (qy * qy + qz * qz);
                    float rz = std::atan2(siny_cosp, cosy_cosp);
                    
                    const float rad2deg = 180.0f / 3.14159265f;
                    Vec3 euler(rx * rad2deg, ry * rad2deg, rz * rad2deg);
                    
                    auto& track = ctx.scene.timeline.tracks[nodeName];
                    Keyframe* existing = track.getKeyframeAt(frame);
                    if (existing) {
                        existing->transform.rotation = euler;
                        existing->transform.has_rotation = true;
                        existing->transform.has_rot_x = true; existing->transform.has_rot_y = true; existing->transform.has_rot_z = true;
                        existing->has_transform = true;
                    } else {
                        Keyframe kf(frame);
                        kf.has_transform = true;
                        kf.transform.rotation = euler;
                        kf.transform.has_rotation = true;
                        kf.transform.has_rot_x = true; kf.transform.has_rot_y = true; kf.transform.has_rot_z = true;
                        kf.transform.has_position = false;
                        kf.transform.has_scale = false;
                        track.addKeyframe(kf);
                    }
                }
            }
            
            // Process Scale Keys
            for (const auto& [nodeName, keys] : anim->scalingKeys) {
                for (const auto& key : keys) {
                    int frame = static_cast<int>(std::round(key.mTime / tps * 24.0));
                    
                    auto& track = ctx.scene.timeline.tracks[nodeName];
                    Keyframe* existing = track.getKeyframeAt(frame);
                    if (existing) {
                        existing->transform.scale = Vec3(key.mValue.x, key.mValue.y, key.mValue.z);
                        existing->transform.has_scale = true;
                        existing->transform.has_scl_x = true; existing->transform.has_scl_y = true; existing->transform.has_scl_z = true;
                        existing->has_transform = true;
                    } else {
                        Keyframe kf(frame);
                        kf.has_transform = true;
                        kf.transform.scale = Vec3(key.mValue.x, key.mValue.y, key.mValue.z);
                        kf.transform.has_scale = true;
                        kf.transform.has_scl_x = true; kf.transform.has_scl_y = true; kf.transform.has_scl_z = true;
                        kf.transform.has_position = false;
                        kf.transform.has_rotation = false;
                        track.addKeyframe(kf);
                    }
                }
            }
            
            // Expand frame range to cover this animation (keep user edits if already wider)
            if (anim->endFrame > 0) {
                if (start_frame == 0 && end_frame == 250) {
                    start_frame = anim->startFrame;
                    end_frame = std::max(anim->endFrame, 1);
                } else {
                    start_frame = std::min(start_frame, anim->startFrame);
                    end_frame   = std::max(end_frame,   anim->endFrame);
                }
            }
        }
    }

    lastSyncedAnimCount = currentCount;
    tracks_dirty = true;  // Rebuild track list to show new keyframes

    // NOTE: Selection sync and keyboard shortcuts are now handled by handleSelectionSync()
    // which is called every frame from draw().
}

// INSERT KEYFRAME FOR TRACK - Uses selection data like existing code
// ============================================================================
void TimelineWidget::insertKeyframeForTrack(UIContext& ctx, const std::string& track_name, int frame) {
    // Parse the track name to get Entity and Channel
    // Uses the static parseTrackName helper we moved to the top
    auto [entity_name, channel] = parseTrackName(ctx.scene, track_name);

    if (findImportedModelContextByName(ctx.scene, entity_name)) {
        Keyframe kf(frame);
        kf.has_anim_graph = true;
        ctx.scene.timeline.insertKeyframe(entity_name, kf);
        return;
    }

    if (!ctx.selection.hasSelection()) return;
    
    // Validate entity name against selection (to ensure we are keying what is selected)
    // Actually, we should allow keying ANY entity if it matches the track?
    // But the data source is the SELECTION. So we can only key the selected object.
    
    // Check if the track entity matches the selected object
    bool match = false;
    if (ctx.selection.selected.type == SelectableType::Object && ctx.selection.selected.object) {
        std::string sel_name = ctx.selection.selected.object->getNodeName();
        if (sel_name.empty()) sel_name = "Object_" + std::to_string(ctx.selection.selected.object_index);
        
        if (sel_name == entity_name) match = true;
    }
    // ... (Light/Camera checks omitted for brevity, logic follows same pattern) ...
    // For now, let's assume if entity_name matches, we proceed using Selection Data.
    
    Keyframe kf(frame);
    bool has_data = false;
    
    if (ctx.selection.selected.type == SelectableType::Object && ctx.selection.selected.object) {
        auto& obj = ctx.selection.selected.object;
        std::string sel_name = obj->getNodeName().empty() ? "Object_" + std::to_string(ctx.selection.selected.object_index) : obj->getNodeName();
        
        if (sel_name == entity_name) {
            kf.has_transform = true;
            kf.transform.position = ctx.selection.selected.position;
            kf.transform.rotation = ctx.selection.selected.rotation;
            kf.transform.scale = ctx.selection.selected.scale;
            
            // Set flags based on CHANNEL
            if (channel == ChannelType::None) {
                // Key Everything
                kf.transform.has_position = true;
                kf.transform.has_pos_x = true; kf.transform.has_pos_y = true; kf.transform.has_pos_z = true;
                
                kf.transform.has_rotation = true;
                kf.transform.has_rot_x = true; kf.transform.has_rot_y = true; kf.transform.has_rot_z = true;
                
                kf.transform.has_scale = true;
                kf.transform.has_scl_x = true; kf.transform.has_scl_y = true; kf.transform.has_scl_z = true;
            }
            else if (channel == ChannelType::Location) {
                kf.transform.has_position = true;
                kf.transform.has_pos_x = true; kf.transform.has_pos_y = true; kf.transform.has_pos_z = true;
            }
            else if (channel == ChannelType::LocationX) { kf.transform.has_position = true; kf.transform.has_pos_x = true; }
            else if (channel == ChannelType::LocationY) { kf.transform.has_position = true; kf.transform.has_pos_y = true; }
            else if (channel == ChannelType::LocationZ) { kf.transform.has_position = true; kf.transform.has_pos_z = true; }
            
            else if (channel == ChannelType::Rotation) {
                kf.transform.has_rotation = true;
                kf.transform.has_rot_x = true; kf.transform.has_rot_y = true; kf.transform.has_rot_z = true;
            }
            else if (channel == ChannelType::RotationX) { kf.transform.has_rotation = true; kf.transform.has_rot_x = true; }
            else if (channel == ChannelType::RotationY) { kf.transform.has_rotation = true; kf.transform.has_rot_y = true; }
            else if (channel == ChannelType::RotationZ) { kf.transform.has_rotation = true; kf.transform.has_rot_z = true; }
            
            else if (channel == ChannelType::Scale) {
                kf.transform.has_scale = true;
                kf.transform.has_scl_x = true; kf.transform.has_scl_y = true; kf.transform.has_scl_z = true;
            }
            else if (channel == ChannelType::ScaleX) { kf.transform.has_scale = true; kf.transform.has_scl_x = true; }
            else if (channel == ChannelType::ScaleY) { kf.transform.has_scale = true; kf.transform.has_scl_y = true; }
            else if (channel == ChannelType::ScaleZ) { kf.transform.has_scale = true; kf.transform.has_scl_z = true; }
            
            else if (channel == ChannelType::Material) {
                 kf.has_material = true;
                 // Need to fetch material from object?
                 // The current code didn't look like it keyed material for objects easily?
                 // Existing code didn't handle material keying in insertKeyframeForTrack for Object. 
                 // I will skip for now to match valid code.
            }
            
            has_data = true;
            
            // CRITICAL FIX: Ensure object has a permanent name if we are keying it
            // Otherwise SceneSerializer might skip it or it relies on unstable indices
            if (obj->getNodeName().empty()) {
                obj->setNodeName(sel_name);
            }
        }
    }
    else if (ctx.selection.selected.type == SelectableType::Light && ctx.selection.selected.light) {
        auto& light = ctx.selection.selected.light;
        std::string sel_name = light->nodeName;
        
        if (sel_name == entity_name) {
            Keyframe kf(frame);
            kf.has_light = true;
            
            // Capture Light Data
            kf.light.position = light->position;
            kf.light.has_position = true;
            
            kf.light.color = light->color;
            kf.light.has_color = true;
            
            kf.light.intensity = light->intensity;
            kf.light.has_intensity = true;
            
            // Direction (for Spot/Directional)
            // We need to cast to check type or just capture if available
            LightType type = light->type();
            if (type == LightType::Directional) {
                if (auto l = std::dynamic_pointer_cast<DirectionalLight>(light)) {
                    kf.light.direction = l->direction;
                    kf.light.has_direction = true;
                }
            } else if (type == LightType::Spot) {
                 if (auto l = std::dynamic_pointer_cast<SpotLight>(light)) {
                    kf.light.direction = l->direction;
                    kf.light.has_direction = true;
                }
            }

            // Set specific channel flags if needed
            if (channel == ChannelType::None) {
                // Key All Light Props
            } 
            else if (channel == ChannelType::Location) kf.light.has_position = true;
            else if (channel == ChannelType::Material) kf.light.has_color = true; // Color mapped to Material channel in tracks
            else if (channel == ChannelType::Scale) kf.light.has_intensity = true; // Intensity mapped to Scale
            else if (channel == ChannelType::Rotation) kf.light.has_direction = true;

            // Save to Timeline
            // Note: We don't overwrite the whole keyframe, insertKeyframe merges it
            ctx.scene.timeline.insertKeyframe(entity_name, kf);
            return; // Done
        }
    }
    else if (ctx.selection.selected.type == SelectableType::Camera && ctx.selection.selected.camera) {
        auto& cam = ctx.selection.selected.camera;
        std::string sel_name = cam->nodeName;
        
        if (sel_name == entity_name) {
            Keyframe kf(frame);
            kf.has_camera = true;

            // Capture Camera Data — values are always captured; the has_* flags
            // below decide which channels actually participate in evaluation.
            kf.camera.position = cam->lookfrom;
            kf.camera.target = cam->lookat;
            kf.camera.fov = cam->vfov;
            kf.camera.focus_distance = cam->focus_dist;
            kf.camera.lens_radius = cam->aperture;

            // Default keyframe channels by selector:
            //   None     → "Key All Camera Props" but EXCLUDE DOF (focus/aperture)
            //              so adding a routine camera key doesn't suddenly animate
            //              depth-of-field on top of pose/fov. Users who want to
            //              animate DOF should pick the Material channel explicitly.
            //   Location → position only
            //   Rotation → target (lookAt) only
            //   Scale    → fov only
            //   Material → DOF only (focus + aperture)
            if (channel == ChannelType::None) {
                kf.camera.has_position = true;
                kf.camera.has_target = true;
                kf.camera.has_fov = true;
                kf.camera.has_focus = false;
                kf.camera.has_aperture = false;
            }
            else if (channel == ChannelType::Location) {
                kf.camera.has_position = true;
                kf.camera.has_target = false;
                kf.camera.has_fov = false;
                kf.camera.has_focus = false;
                kf.camera.has_aperture = false;
            }
            else if (channel == ChannelType::Rotation) {
                kf.camera.has_position = false;
                kf.camera.has_target = true;
                kf.camera.has_fov = false;
                kf.camera.has_focus = false;
                kf.camera.has_aperture = false;
            }
            else if (channel == ChannelType::Scale) {
                kf.camera.has_position = false;
                kf.camera.has_target = false;
                kf.camera.has_fov = true;
                kf.camera.has_focus = false;
                kf.camera.has_aperture = false;
            }
            else if (channel == ChannelType::Material) {
                kf.camera.has_position = false;
                kf.camera.has_target = false;
                kf.camera.has_fov = false;
                kf.camera.has_focus = true;
                kf.camera.has_aperture = true;
            }
            
            ctx.scene.timeline.insertKeyframe(entity_name, kf);
            return; // Done
        }
    }

    // --- GAS / EMITTER KEYFRAME SUPPORT ---
    for (auto& gas : ctx.scene.gas_volumes) {
        auto& emitters = gas->getSimulator().getEmitters();
        for (int i = 0; i < (int)emitters.size(); ++i) {
            auto& e = emitters[i];
            std::string expected_name = gas->getName() + "_" + e.name + "_" + std::to_string(e.uid);
            if (expected_name == entity_name) {
                Keyframe kf(frame);
                kf.has_emitter = true;
                kf.emitter.fuel_rate = e.fuel_rate; kf.emitter.has_fuel_rate = true;
                kf.emitter.density_rate = e.density_rate; kf.emitter.has_density_rate = true;
                kf.emitter.temperature = e.temperature; kf.emitter.has_temperature = true;
                kf.emitter.position = e.position; kf.emitter.has_position = true;
                kf.emitter.velocity = e.velocity; kf.emitter.has_velocity = true;
                kf.emitter.size = e.size; kf.emitter.has_size = true;
                kf.emitter.radius = e.radius; kf.emitter.has_radius = true;
                kf.emitter.enabled = e.enabled; kf.emitter.has_enabled = true;

                ctx.scene.timeline.insertKeyframe(entity_name, kf);
                return;
            }
        }
    }
    
    // --- TERRAIN KEYFRAME SUPPORT ---
    // Check if this track is a terrain track
    auto& terrains = TerrainManager::getInstance().getTerrains();
    for (auto& terrain : terrains) {
        if (terrain.name == entity_name) {
            // Found matching terrain - capture keyframe using existing TerrainManager logic
            auto& track = ctx.scene.timeline.tracks[entity_name];
            TerrainManager::getInstance().captureKeyframeToTrack(&terrain, track, frame);
            return; // Done
        }
    }
    
    if (has_data && !entity_name.empty()) {
        ctx.scene.timeline.insertKeyframe(entity_name, kf);
    }
}

// ============================================================================
// DELETE KEYFRAME HELPERS
// ============================================================================

// ============================================================================
// DELETE KEYFRAME
// ============================================================================
static void clearKeyframeChannel(Keyframe& kf, ChannelType channel) {
    if (kf.has_transform) {
        if (channel == ChannelType::Location) { kf.transform.has_position = false; kf.transform.has_pos_x = kf.transform.has_pos_y = kf.transform.has_pos_z = false; }
        else if (channel == ChannelType::Rotation) { kf.transform.has_rotation = false; kf.transform.has_rot_x = kf.transform.has_rot_y = kf.transform.has_rot_z = false; }
        else if (channel == ChannelType::Scale) { kf.transform.has_scale = false; kf.transform.has_scl_x = kf.transform.has_scl_y = kf.transform.has_scl_z = false; }
        else if (channel == ChannelType::LocationX) kf.transform.has_pos_x = false;
        else if (channel == ChannelType::LocationY) kf.transform.has_pos_y = false;
        else if (channel == ChannelType::LocationZ) kf.transform.has_pos_z = false;
        else if (channel == ChannelType::RotationX) kf.transform.has_rot_x = false;
        else if (channel == ChannelType::RotationY) kf.transform.has_rot_y = false;
        else if (channel == ChannelType::RotationZ) kf.transform.has_rot_z = false;
        else if (channel == ChannelType::ScaleX) kf.transform.has_scl_x = false;
        else if (channel == ChannelType::ScaleY) kf.transform.has_scl_y = false;
        else if (channel == ChannelType::ScaleZ) kf.transform.has_scl_z = false;
        kf.transform.refreshCompoundFlags();
    }
    if (kf.has_light) {
        if (channel == ChannelType::Location) { kf.light.has_position = false; kf.light.has_pos_x = kf.light.has_pos_y = kf.light.has_pos_z = false; }
        else if (channel == ChannelType::Material) { kf.light.has_color = false; kf.light.has_col_r = kf.light.has_col_g = kf.light.has_col_b = false; }
        else if (channel == ChannelType::LightIntensity) { kf.light.has_intensity = kf.light.has_int = false; }
        else if (channel == ChannelType::Rotation) { kf.light.has_direction = false; kf.light.has_dir_x = kf.light.has_dir_y = kf.light.has_dir_z = false; }
        else if (channel == ChannelType::LightPosX) kf.light.has_pos_x = false;
        else if (channel == ChannelType::LightPosY) kf.light.has_pos_y = false;
        else if (channel == ChannelType::LightPosZ) kf.light.has_pos_z = false;
        else if (channel == ChannelType::LightColR) kf.light.has_col_r = false;
        else if (channel == ChannelType::LightColG) kf.light.has_col_g = false;
        else if (channel == ChannelType::LightColB) kf.light.has_col_b = false;
        else if (channel == ChannelType::LightDirX) kf.light.has_dir_x = false;
        else if (channel == ChannelType::LightDirY) kf.light.has_dir_y = false;
        else if (channel == ChannelType::LightDirZ) kf.light.has_dir_z = false;
        kf.light.refreshCompoundFlags();
    }
    if (kf.has_camera) {
        if (channel == ChannelType::Location) { kf.camera.has_position = false; kf.camera.has_pos_x = kf.camera.has_pos_y = kf.camera.has_pos_z = false; }
        else if (channel == ChannelType::Rotation) { kf.camera.has_target = false; kf.camera.has_tgt_x = kf.camera.has_tgt_y = kf.camera.has_tgt_z = false; }
        else if (channel == ChannelType::CamFOV) { kf.camera.has_fov = kf.camera.has_fv = false; }
        else if (channel == ChannelType::Material) { kf.camera.has_focus = kf.camera.has_aperture = kf.camera.has_foc_dist = kf.camera.has_lens_rad = false; }
        else if (channel == ChannelType::CamPosX) kf.camera.has_pos_x = false;
        else if (channel == ChannelType::CamPosY) kf.camera.has_pos_y = false;
        else if (channel == ChannelType::CamPosZ) kf.camera.has_pos_z = false;
        else if (channel == ChannelType::CamTgtX) kf.camera.has_tgt_x = false;
        else if (channel == ChannelType::CamTgtY) kf.camera.has_tgt_y = false;
        else if (channel == ChannelType::CamTgtZ) kf.camera.has_tgt_z = false;
        else if (channel == ChannelType::CamFocusDist) kf.camera.has_foc_dist = false;
        else if (channel == ChannelType::CamLensRad) kf.camera.has_lens_rad = false;
        kf.camera.refreshCompoundFlags();
    }
    if (kf.has_material) {
        if (channel == ChannelType::Material) { kf.material.clearAllChannels(); }
        else if (channel == ChannelType::MatAlbedoR) kf.material.has_alb_r = false;
        else if (channel == ChannelType::MatAlbedoG) kf.material.has_alb_g = false;
        else if (channel == ChannelType::MatAlbedoB) kf.material.has_alb_b = false;
        else if (channel == ChannelType::MatOpacity) kf.material.has_opac = false;
        else if (channel == ChannelType::MatRoughness) kf.material.has_rough = false;
        else if (channel == ChannelType::MatMetallic) kf.material.has_metal = false;
        else if (channel == ChannelType::MatClearcoat) kf.material.has_clear = false;
        else if (channel == ChannelType::MatTransmission) kf.material.has_transm = false;
        else if (channel == ChannelType::MatIOR) kf.material.has_ior_val = false;
        else if (channel == ChannelType::MatEmissionR) kf.material.has_emis_r = false;
        else if (channel == ChannelType::MatEmissionG) kf.material.has_emis_g = false;
        else if (channel == ChannelType::MatEmissionB) kf.material.has_emis_b = false;
        else if (channel == ChannelType::MatNormalStrength) kf.material.has_norm_str = false;
        else if (channel == ChannelType::MatEmissionStrength) kf.material.has_emis_str = false;
        kf.material.refreshCompoundFlags();
    }
}

void TimelineWidget::deleteKeyframe(UIContext& ctx, const std::string& track_name, int frame) {
    auto [entity_name, channel] = parseTrackName(ctx.scene, track_name);
    
    auto it = ctx.scene.timeline.tracks.find(entity_name);
    if (it != ctx.scene.timeline.tracks.end()) {
        auto& keyframes = it->second.keyframes;
        
        // If whole track (Channel::None), remove entire keyframe
        if (channel == ChannelType::None) {
            keyframes.erase(
                std::remove_if(keyframes.begin(), keyframes.end(),
                    [frame](const Keyframe& kf) { return kf.frame == frame; }),
                keyframes.end());
            return;
        }
        
        // Otherwise, clear specific flags
        for (auto it_kf = keyframes.begin(); it_kf != keyframes.end(); ) {
            if (it_kf->frame == frame) {
                clearKeyframeChannel(*it_kf, channel);

                // Update has_* flags if they no longer have any active keys
                if (it_kf->has_transform && !it_kf->transform.has_position && !it_kf->transform.has_rotation && !it_kf->transform.has_scale) it_kf->has_transform = false;
                if (it_kf->has_light && !it_kf->light.has_position && !it_kf->light.has_color && !it_kf->light.has_intensity && !it_kf->light.has_direction) it_kf->has_light = false;
                if (it_kf->has_camera && !it_kf->camera.has_position && !it_kf->camera.has_target && !it_kf->camera.has_fov && !it_kf->camera.has_focus && !it_kf->camera.has_aperture) it_kf->has_camera = false;
                if (it_kf->has_material && !it_kf->material.has_albedo && !it_kf->material.has_opacity && !it_kf->material.has_roughness && !it_kf->material.has_metallic && !it_kf->material.has_clearcoat && !it_kf->material.has_transmission && !it_kf->material.has_ior && !it_kf->material.has_emission && !it_kf->material.has_normal) it_kf->has_material = false;

                // Check if keyframe is now completely empty
                bool has_any_data = it_kf->has_transform || it_kf->has_material || it_kf->has_light || it_kf->has_camera || it_kf->has_world || it_kf->has_terrain;
                
                if (!has_any_data) {
                    // Remove mostly empty keyframe
                    it_kf = keyframes.erase(it_kf);
                    continue; 
                }
            }
            ++it_kf;
        }
    }
}

// ============================================================================
// MOVE KEYFRAME
// ============================================================================
void TimelineWidget::moveKeyframe(UIContext& ctx, const std::string& track_name, int old_frame, int new_frame) {
    if (old_frame == new_frame) return;
    auto [entity_name, channel] = parseTrackName(ctx.scene, track_name);

    auto it = ctx.scene.timeline.tracks.find(entity_name);
    if (it != ctx.scene.timeline.tracks.end()) {
        auto& keyframes = it->second.keyframes;
        
        // Find source keyframe
        Keyframe* src_kf = nullptr;
        for (auto& kf : keyframes) {
            if (kf.frame == old_frame) {
                src_kf = &kf;
                break;
            }
        }
        
        if (!src_kf) return;
        
        // If moving entire row (None), just change frame (and merge if exists?)
        if (channel == ChannelType::None) {
             // Check if target frame exists
             auto it_dst = std::find_if(keyframes.begin(), keyframes.end(), [new_frame](const Keyframe& k) { return k.frame == new_frame; });
             if (it_dst != keyframes.end()) {
                  keyframes.erase(it_dst);
             }
             // RESTARTING SEARCH TO BE SAFE
             for (auto& kf : keyframes) { 
                  if (kf.frame == old_frame) {
                      kf.frame = new_frame;
                      break;
                  }
             }
        } 
        else {
             // SPLIT MOVE: Move ONLY the specific channel data to new frame
             // 1. Check if dest frame exists
             Keyframe* dst_kf = nullptr;
             for (auto& kf : keyframes) { if (kf.frame == new_frame) { dst_kf = &kf; break; } }
             
             if (!dst_kf) {
                 // Create new keyframe at dest
                 Keyframe new_k(new_frame);
                 new_k.has_transform = src_kf->has_transform; // Init type flags
                 new_k.has_light = src_kf->has_light;
                 new_k.has_camera = src_kf->has_camera;
                 new_k.has_material = src_kf->has_material;
                 new_k.has_world = src_kf->has_world;
                 new_k.has_terrain = src_kf->has_terrain;
                 keyframes.push_back(new_k);
                 dst_kf = &keyframes.back();
                 // Pointers invalidated after push_back! Re-find src
                 for (auto& kf : keyframes) { if (kf.frame == old_frame) { src_kf = &kf; break; } }
             }
             
             // 2. Transfer data from src_kf to dst_kf based on channel
             if (src_kf->has_transform) {
                 if (channel == ChannelType::Location) {
                     dst_kf->transform.position = src_kf->transform.position;
                     dst_kf->transform.has_position = src_kf->transform.has_position;
                     dst_kf->transform.has_pos_x = src_kf->transform.has_pos_x;
                     dst_kf->transform.has_pos_y = src_kf->transform.has_pos_y;
                     dst_kf->transform.has_pos_z = src_kf->transform.has_pos_z;
                     for (int c = CURVE_POS_X; c <= CURVE_POS_Z; ++c) dst_kf->transform.curve[c] = src_kf->transform.curve[c];
                     dst_kf->has_transform = true;
                 } else if (channel == ChannelType::Rotation) {
                     dst_kf->transform.rotation = src_kf->transform.rotation;
                     dst_kf->transform.has_rotation = src_kf->transform.has_rotation;
                     dst_kf->transform.has_rot_x = src_kf->transform.has_rot_x;
                     dst_kf->transform.has_rot_y = src_kf->transform.has_rot_y;
                     dst_kf->transform.has_rot_z = src_kf->transform.has_rot_z;
                     for (int c = CURVE_ROT_X; c <= CURVE_ROT_Z; ++c) dst_kf->transform.curve[c] = src_kf->transform.curve[c];
                     dst_kf->has_transform = true;
                 } else if (channel == ChannelType::Scale) {
                     dst_kf->transform.scale = src_kf->transform.scale;
                     dst_kf->transform.has_scale = src_kf->transform.has_scale;
                     dst_kf->transform.has_scl_x = src_kf->transform.has_scl_x;
                     dst_kf->transform.has_scl_y = src_kf->transform.has_scl_y;
                     dst_kf->transform.has_scl_z = src_kf->transform.has_scl_z;
                     for (int c = CURVE_SCL_X; c <= CURVE_SCL_Z; ++c) dst_kf->transform.curve[c] = src_kf->transform.curve[c];
                     dst_kf->has_transform = true;
                 } else if (channel == ChannelType::LocationX) {
                     dst_kf->transform.position.x = src_kf->transform.position.x;
                     dst_kf->transform.has_pos_x = src_kf->transform.has_pos_x;
                     dst_kf->transform.curve[CURVE_POS_X] = src_kf->transform.curve[CURVE_POS_X];
                     dst_kf->has_transform = true;
                 } else if (channel == ChannelType::LocationY) {
                     dst_kf->transform.position.y = src_kf->transform.position.y;
                     dst_kf->transform.has_pos_y = src_kf->transform.has_pos_y;
                     dst_kf->transform.curve[CURVE_POS_Y] = src_kf->transform.curve[CURVE_POS_Y];
                     dst_kf->has_transform = true;
                 } else if (channel == ChannelType::LocationZ) {
                     dst_kf->transform.position.z = src_kf->transform.position.z;
                     dst_kf->transform.has_pos_z = src_kf->transform.has_pos_z;
                     dst_kf->transform.curve[CURVE_POS_Z] = src_kf->transform.curve[CURVE_POS_Z];
                     dst_kf->has_transform = true;
                 } else if (channel == ChannelType::RotationX) {
                     dst_kf->transform.rotation.x = src_kf->transform.rotation.x;
                     dst_kf->transform.has_rot_x = src_kf->transform.has_rot_x;
                     dst_kf->transform.curve[CURVE_ROT_X] = src_kf->transform.curve[CURVE_ROT_X];
                     dst_kf->has_transform = true;
                 } else if (channel == ChannelType::RotationY) {
                     dst_kf->transform.rotation.y = src_kf->transform.rotation.y;
                     dst_kf->transform.has_rot_y = src_kf->transform.has_rot_y;
                     dst_kf->transform.curve[CURVE_ROT_Y] = src_kf->transform.curve[CURVE_ROT_Y];
                     dst_kf->has_transform = true;
                 } else if (channel == ChannelType::RotationZ) {
                     dst_kf->transform.rotation.z = src_kf->transform.rotation.z;
                     dst_kf->transform.has_rot_z = src_kf->transform.has_rot_z;
                     dst_kf->transform.curve[CURVE_ROT_Z] = src_kf->transform.curve[CURVE_ROT_Z];
                     dst_kf->has_transform = true;
                 } else if (channel == ChannelType::ScaleX) {
                     dst_kf->transform.scale.x = src_kf->transform.scale.x;
                     dst_kf->transform.has_scl_x = src_kf->transform.has_scl_x;
                     dst_kf->transform.curve[CURVE_SCL_X] = src_kf->transform.curve[CURVE_SCL_X];
                     dst_kf->has_transform = true;
                 } else if (channel == ChannelType::ScaleY) {
                     dst_kf->transform.scale.y = src_kf->transform.scale.y;
                     dst_kf->transform.has_scl_y = src_kf->transform.has_scl_y;
                     dst_kf->transform.curve[CURVE_SCL_Y] = src_kf->transform.curve[CURVE_SCL_Y];
                     dst_kf->has_transform = true;
                 } else if (channel == ChannelType::ScaleZ) {
                     dst_kf->transform.scale.z = src_kf->transform.scale.z;
                     dst_kf->transform.has_scl_z = src_kf->transform.has_scl_z;
                     dst_kf->transform.curve[CURVE_SCL_Z] = src_kf->transform.curve[CURVE_SCL_Z];
                     dst_kf->has_transform = true;
                 }
             }

             if (src_kf->has_light) {
                 if (channel == ChannelType::Location) {
                     dst_kf->light.position = src_kf->light.position;
                     dst_kf->light.has_position = src_kf->light.has_position;
                     dst_kf->light.has_pos_x = src_kf->light.has_pos_x;
                     dst_kf->light.has_pos_y = src_kf->light.has_pos_y;
                     dst_kf->light.has_pos_z = src_kf->light.has_pos_z;
                     for (int c = CURVE_LIGHT_POS_X; c <= CURVE_LIGHT_POS_Z; ++c) dst_kf->light.curve[c] = src_kf->light.curve[c];
                     dst_kf->has_light = true;
                 } else if (channel == ChannelType::Material) {
                     dst_kf->light.color = src_kf->light.color;
                     dst_kf->light.has_color = src_kf->light.has_color;
                     dst_kf->light.has_col_r = src_kf->light.has_col_r;
                     dst_kf->light.has_col_g = src_kf->light.has_col_g;
                     dst_kf->light.has_col_b = src_kf->light.has_col_b;
                     for (int c = CURVE_LIGHT_COLOR_R; c <= CURVE_LIGHT_COLOR_B; ++c) dst_kf->light.curve[c] = src_kf->light.curve[c];
                     dst_kf->has_light = true;
                 } else if (channel == ChannelType::LightIntensity) {
                     dst_kf->light.intensity = src_kf->light.intensity;
                     dst_kf->light.has_intensity = src_kf->light.has_intensity;
                     dst_kf->light.has_int = src_kf->light.has_int;
                     dst_kf->light.curve[CURVE_LIGHT_INTENSITY] = src_kf->light.curve[CURVE_LIGHT_INTENSITY];
                     dst_kf->has_light = true;
                 } else if (channel == ChannelType::Rotation) {
                     dst_kf->light.direction = src_kf->light.direction;
                     dst_kf->light.has_direction = src_kf->light.has_direction;
                     dst_kf->light.has_dir_x = src_kf->light.has_dir_x;
                     dst_kf->light.has_dir_y = src_kf->light.has_dir_y;
                     dst_kf->light.has_dir_z = src_kf->light.has_dir_z;
                     for (int c = CURVE_LIGHT_DIR_X; c <= CURVE_LIGHT_DIR_Z; ++c) dst_kf->light.curve[c] = src_kf->light.curve[c];
                     dst_kf->has_light = true;
                 } else if (channel == ChannelType::LightPosX) {
                     dst_kf->light.position.x = src_kf->light.position.x;
                     dst_kf->light.has_pos_x = src_kf->light.has_pos_x;
                     dst_kf->light.curve[CURVE_LIGHT_POS_X] = src_kf->light.curve[CURVE_LIGHT_POS_X];
                     dst_kf->has_light = true;
                 } else if (channel == ChannelType::LightPosY) {
                     dst_kf->light.position.y = src_kf->light.position.y;
                     dst_kf->light.has_pos_y = src_kf->light.has_pos_y;
                     dst_kf->light.curve[CURVE_LIGHT_POS_Y] = src_kf->light.curve[CURVE_LIGHT_POS_Y];
                     dst_kf->has_light = true;
                 } else if (channel == ChannelType::LightPosZ) {
                     dst_kf->light.position.z = src_kf->light.position.z;
                     dst_kf->light.has_pos_z = src_kf->light.has_pos_z;
                     dst_kf->light.curve[CURVE_LIGHT_POS_Z] = src_kf->light.curve[CURVE_LIGHT_POS_Z];
                     dst_kf->has_light = true;
                 } else if (channel == ChannelType::LightColR) {
                     dst_kf->light.color.x = src_kf->light.color.x;
                     dst_kf->light.has_col_r = src_kf->light.has_col_r;
                     dst_kf->light.curve[CURVE_LIGHT_COLOR_R] = src_kf->light.curve[CURVE_LIGHT_COLOR_R];
                     dst_kf->has_light = true;
                 } else if (channel == ChannelType::LightColG) {
                     dst_kf->light.color.y = src_kf->light.color.y;
                     dst_kf->light.has_col_g = src_kf->light.has_col_g;
                     dst_kf->light.curve[CURVE_LIGHT_COLOR_G] = src_kf->light.curve[CURVE_LIGHT_COLOR_G];
                     dst_kf->has_light = true;
                 } else if (channel == ChannelType::LightColB) {
                     dst_kf->light.color.z = src_kf->light.color.z;
                     dst_kf->light.has_col_b = src_kf->light.has_col_b;
                     dst_kf->light.curve[CURVE_LIGHT_COLOR_B] = src_kf->light.curve[CURVE_LIGHT_COLOR_B];
                     dst_kf->has_light = true;
                 } else if (channel == ChannelType::LightDirX) {
                     dst_kf->light.direction.x = src_kf->light.direction.x;
                     dst_kf->light.has_dir_x = src_kf->light.has_dir_x;
                     dst_kf->light.curve[CURVE_LIGHT_DIR_X] = src_kf->light.curve[CURVE_LIGHT_DIR_X];
                     dst_kf->has_light = true;
                 } else if (channel == ChannelType::LightDirY) {
                     dst_kf->light.direction.y = src_kf->light.direction.y;
                     dst_kf->light.has_dir_y = src_kf->light.has_dir_y;
                     dst_kf->light.curve[CURVE_LIGHT_DIR_Y] = src_kf->light.curve[CURVE_LIGHT_DIR_Y];
                     dst_kf->has_light = true;
                 } else if (channel == ChannelType::LightDirZ) {
                     dst_kf->light.direction.z = src_kf->light.direction.z;
                     dst_kf->light.has_dir_z = src_kf->light.has_dir_z;
                     dst_kf->light.curve[CURVE_LIGHT_DIR_Z] = src_kf->light.curve[CURVE_LIGHT_DIR_Z];
                     dst_kf->has_light = true;
                 }
             }

             if (src_kf->has_camera) {
                 if (channel == ChannelType::Location) {
                     dst_kf->camera.position = src_kf->camera.position;
                     dst_kf->camera.has_position = src_kf->camera.has_position;
                     dst_kf->camera.has_pos_x = src_kf->camera.has_pos_x;
                     dst_kf->camera.has_pos_y = src_kf->camera.has_pos_y;
                     dst_kf->camera.has_pos_z = src_kf->camera.has_pos_z;
                     for (int c = CURVE_CAM_POS_X; c <= CURVE_CAM_POS_Z; ++c) dst_kf->camera.curve[c] = src_kf->camera.curve[c];
                     dst_kf->has_camera = true;
                 } else if (channel == ChannelType::Rotation) {
                     dst_kf->camera.target = src_kf->camera.target;
                     dst_kf->camera.has_target = src_kf->camera.has_target;
                     dst_kf->camera.has_tgt_x = src_kf->camera.has_tgt_x;
                     dst_kf->camera.has_tgt_y = src_kf->camera.has_tgt_y;
                     dst_kf->camera.has_tgt_z = src_kf->camera.has_tgt_z;
                     for (int c = CURVE_CAM_TGT_X; c <= CURVE_CAM_TGT_Z; ++c) dst_kf->camera.curve[c] = src_kf->camera.curve[c];
                     dst_kf->has_camera = true;
                 } else if (channel == ChannelType::CamFOV) {
                     dst_kf->camera.fov = src_kf->camera.fov;
                     dst_kf->camera.has_fov = src_kf->camera.has_fov;
                     dst_kf->camera.has_fv = src_kf->camera.has_fv;
                     dst_kf->camera.curve[CURVE_CAM_FOV] = src_kf->camera.curve[CURVE_CAM_FOV];
                     dst_kf->has_camera = true;
                 } else if (channel == ChannelType::Material) {
                     dst_kf->camera.focus_distance = src_kf->camera.focus_distance;
                     dst_kf->camera.lens_radius = src_kf->camera.lens_radius;
                     dst_kf->camera.has_focus = src_kf->camera.has_focus;
                     dst_kf->camera.has_aperture = src_kf->camera.has_aperture;
                     dst_kf->camera.has_foc_dist = src_kf->camera.has_foc_dist;
                     dst_kf->camera.has_lens_rad = src_kf->camera.has_lens_rad;
                     dst_kf->camera.curve[CURVE_CAM_FOCUS_DIST] = src_kf->camera.curve[CURVE_CAM_FOCUS_DIST];
                     dst_kf->camera.curve[CURVE_CAM_LENS_RAD] = src_kf->camera.curve[CURVE_CAM_LENS_RAD];
                     dst_kf->has_camera = true;
                 } else if (channel == ChannelType::CamPosX) {
                     dst_kf->camera.position.x = src_kf->camera.position.x;
                     dst_kf->camera.has_pos_x = src_kf->camera.has_pos_x;
                     dst_kf->camera.curve[CURVE_CAM_POS_X] = src_kf->camera.curve[CURVE_CAM_POS_X];
                     dst_kf->has_camera = true;
                 } else if (channel == ChannelType::CamPosY) {
                     dst_kf->camera.position.y = src_kf->camera.position.y;
                     dst_kf->camera.has_pos_y = src_kf->camera.has_pos_y;
                     dst_kf->camera.curve[CURVE_CAM_POS_Y] = src_kf->camera.curve[CURVE_CAM_POS_Y];
                     dst_kf->has_camera = true;
                 } else if (channel == ChannelType::CamPosZ) {
                     dst_kf->camera.position.z = src_kf->camera.position.z;
                     dst_kf->camera.has_pos_z = src_kf->camera.has_pos_z;
                     dst_kf->camera.curve[CURVE_CAM_POS_Z] = src_kf->camera.curve[CURVE_CAM_POS_Z];
                     dst_kf->has_camera = true;
                 } else if (channel == ChannelType::CamTgtX) {
                     dst_kf->camera.target.x = src_kf->camera.target.x;
                     dst_kf->camera.has_tgt_x = src_kf->camera.has_tgt_x;
                     dst_kf->camera.curve[CURVE_CAM_TGT_X] = src_kf->camera.curve[CURVE_CAM_TGT_X];
                     dst_kf->has_camera = true;
                 } else if (channel == ChannelType::CamTgtY) {
                     dst_kf->camera.target.y = src_kf->camera.target.y;
                     dst_kf->camera.has_tgt_y = src_kf->camera.has_tgt_y;
                     dst_kf->camera.curve[CURVE_CAM_TGT_Y] = src_kf->camera.curve[CURVE_CAM_TGT_Y];
                     dst_kf->has_camera = true;
                 } else if (channel == ChannelType::CamTgtZ) {
                     dst_kf->camera.target.z = src_kf->camera.target.z;
                     dst_kf->camera.has_tgt_z = src_kf->camera.has_tgt_z;
                     dst_kf->camera.curve[CURVE_CAM_TGT_Z] = src_kf->camera.curve[CURVE_CAM_TGT_Z];
                     dst_kf->has_camera = true;
                 } else if (channel == ChannelType::CamFocusDist) {
                     dst_kf->camera.focus_distance = src_kf->camera.focus_distance;
                     dst_kf->camera.has_foc_dist = src_kf->camera.has_foc_dist;
                     dst_kf->camera.curve[CURVE_CAM_FOCUS_DIST] = src_kf->camera.curve[CURVE_CAM_FOCUS_DIST];
                     dst_kf->has_camera = true;
                 } else if (channel == ChannelType::CamLensRad) {
                     dst_kf->camera.lens_radius = src_kf->camera.lens_radius;
                     dst_kf->camera.has_lens_rad = src_kf->camera.has_lens_rad;
                     dst_kf->camera.curve[CURVE_CAM_LENS_RAD] = src_kf->camera.curve[CURVE_CAM_LENS_RAD];
                     dst_kf->has_camera = true;
                 }
             }

             if (src_kf->has_material) {
                 if (channel == ChannelType::Material) {
                     dst_kf->material = src_kf->material;
                     dst_kf->has_material = true;
                 } else if (channel == ChannelType::MatAlbedoR) {
                     dst_kf->material.albedo.x = src_kf->material.albedo.x;
                     dst_kf->material.has_alb_r = src_kf->material.has_alb_r;
                     dst_kf->material.curve[CURVE_MAT_ALBEDO_R] = src_kf->material.curve[CURVE_MAT_ALBEDO_R];
                     dst_kf->has_material = true;
                 } else if (channel == ChannelType::MatAlbedoG) {
                     dst_kf->material.albedo.y = src_kf->material.albedo.y;
                     dst_kf->material.has_alb_g = src_kf->material.has_alb_g;
                     dst_kf->material.curve[CURVE_MAT_ALBEDO_G] = src_kf->material.curve[CURVE_MAT_ALBEDO_G];
                     dst_kf->has_material = true;
                 } else if (channel == ChannelType::MatAlbedoB) {
                     dst_kf->material.albedo.z = src_kf->material.albedo.z;
                     dst_kf->material.has_alb_b = src_kf->material.has_alb_b;
                     dst_kf->material.curve[CURVE_MAT_ALBEDO_B] = src_kf->material.curve[CURVE_MAT_ALBEDO_B];
                     dst_kf->has_material = true;
                 } else if (channel == ChannelType::MatOpacity) {
                     dst_kf->material.opacity = src_kf->material.opacity;
                     dst_kf->material.has_opac = src_kf->material.has_opac;
                     dst_kf->material.curve[CURVE_MAT_OPACITY] = src_kf->material.curve[CURVE_MAT_OPACITY];
                     dst_kf->has_material = true;
                 } else if (channel == ChannelType::MatRoughness) {
                     dst_kf->material.roughness = src_kf->material.roughness;
                     dst_kf->material.has_rough = src_kf->material.has_rough;
                     dst_kf->material.curve[CURVE_MAT_ROUGHNESS] = src_kf->material.curve[CURVE_MAT_ROUGHNESS];
                     dst_kf->has_material = true;
                 } else if (channel == ChannelType::MatMetallic) {
                     dst_kf->material.metallic = src_kf->material.metallic;
                     dst_kf->material.has_metal = src_kf->material.has_metal;
                     dst_kf->material.curve[CURVE_MAT_METALLIC] = src_kf->material.curve[CURVE_MAT_METALLIC];
                     dst_kf->has_material = true;
                 } else if (channel == ChannelType::MatClearcoat) {
                     dst_kf->material.clearcoat = src_kf->material.clearcoat;
                     dst_kf->material.has_clear = src_kf->material.has_clear;
                     dst_kf->material.curve[CURVE_MAT_CLEARCOAT] = src_kf->material.curve[CURVE_MAT_CLEARCOAT];
                     dst_kf->has_material = true;
                 } else if (channel == ChannelType::MatTransmission) {
                     dst_kf->material.transmission = src_kf->material.transmission;
                     dst_kf->material.has_transm = src_kf->material.has_transm;
                     dst_kf->material.curve[CURVE_MAT_TRANSMISSION] = src_kf->material.curve[CURVE_MAT_TRANSMISSION];
                     dst_kf->has_material = true;
                 } else if (channel == ChannelType::MatIOR) {
                     dst_kf->material.ior = src_kf->material.ior;
                     dst_kf->material.has_ior_val = src_kf->material.has_ior_val;
                     dst_kf->material.curve[CURVE_MAT_IOR] = src_kf->material.curve[CURVE_MAT_IOR];
                     dst_kf->has_material = true;
                 } else if (channel == ChannelType::MatEmissionR) {
                     dst_kf->material.emission.x = src_kf->material.emission.x;
                     dst_kf->material.has_emis_r = src_kf->material.has_emis_r;
                     dst_kf->material.curve[CURVE_MAT_EMISSION_R] = src_kf->material.curve[CURVE_MAT_EMISSION_R];
                     dst_kf->has_material = true;
                 } else if (channel == ChannelType::MatEmissionG) {
                     dst_kf->material.emission.y = src_kf->material.emission.y;
                     dst_kf->material.has_emis_g = src_kf->material.has_emis_g;
                     dst_kf->material.curve[CURVE_MAT_EMISSION_G] = src_kf->material.curve[CURVE_MAT_EMISSION_G];
                     dst_kf->has_material = true;
                 } else if (channel == ChannelType::MatEmissionB) {
                     dst_kf->material.emission.z = src_kf->material.emission.z;
                     dst_kf->material.has_emis_b = src_kf->material.has_emis_b;
                     dst_kf->material.curve[CURVE_MAT_EMISSION_B] = src_kf->material.curve[CURVE_MAT_EMISSION_B];
                     dst_kf->has_material = true;
                 } else if (channel == ChannelType::MatNormalStrength) {
                     dst_kf->material.normal_strength = src_kf->material.normal_strength;
                     dst_kf->material.has_norm_str = src_kf->material.has_norm_str;
                     dst_kf->material.curve[CURVE_MAT_NORMAL_STRENGTH] = src_kf->material.curve[CURVE_MAT_NORMAL_STRENGTH];
                     dst_kf->has_material = true;
                 } else if (channel == ChannelType::MatEmissionStrength) {
                     dst_kf->material.emission_strength = src_kf->material.emission_strength;
                     dst_kf->material.has_emis_str = src_kf->material.has_emis_str;
                     dst_kf->material.curve[CURVE_MAT_EMISSION_STRENGTH] = src_kf->material.curve[CURVE_MAT_EMISSION_STRENGTH];
                     dst_kf->has_material = true;
                 }
             }

             // 3. Clear channel on source
             clearKeyframeChannel(*src_kf, channel);
             
             // 4. Clean up empty source keyframe
             bool src_empty = 
                !(src_kf->has_transform && (src_kf->transform.has_position || src_kf->transform.has_rotation || src_kf->transform.has_scale)) &&
                !(src_kf->has_light && (src_kf->light.has_position || src_kf->light.has_color || src_kf->light.has_intensity || src_kf->light.has_direction)) &&
                !(src_kf->has_camera && (src_kf->camera.has_position || src_kf->camera.has_target || src_kf->camera.has_fov || src_kf->camera.has_focus || src_kf->camera.has_aperture)) &&
                !(src_kf->has_material && (src_kf->material.has_albedo || src_kf->material.has_opacity || src_kf->material.has_roughness || src_kf->material.has_metallic || src_kf->material.has_clearcoat || src_kf->material.has_transmission || src_kf->material.has_ior || src_kf->material.has_emission || src_kf->material.has_normal)) &&
                !src_kf->has_world && !src_kf->has_terrain;
                
             if (src_empty) {
                  keyframes.erase(
                    std::remove_if(keyframes.begin(), keyframes.end(),
                        [old_frame](const Keyframe& kf) { return kf.frame == old_frame; }),
                    keyframes.end());
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
    auto [entity_name, channel] = parseTrackName(ctx.scene, track_name);
    
    auto it = ctx.scene.timeline.tracks.find(entity_name);
    if (it != ctx.scene.timeline.tracks.end()) {
        auto& keyframes = it->second.keyframes;
        
        Keyframe* src_kf = nullptr;
        for (auto& kf : keyframes) { if (kf.frame == src_frame) { src_kf = &kf; break; } }
        
        if (!src_kf) return;
        
        Keyframe* dst_kf = nullptr;
        for (auto& kf : keyframes) { if (kf.frame == dst_frame) { dst_kf = &kf; break; } }
        
        if (channel == ChannelType::None) {
            // Full duplicate
             if (!dst_kf) {
                 Keyframe new_k = *src_kf;
                 new_k.frame = dst_frame;
                 keyframes.push_back(new_k);
             } else {
                 // Overwrite logic or merge? Overwrite for now
                 *dst_kf = *src_kf;
                 dst_kf->frame = dst_frame;
             }
        } else {
            // Partial duplicate
            if (!dst_kf) {
                 Keyframe new_k(dst_frame);
                 keyframes.push_back(new_k);
                 dst_kf = &keyframes.back();
                 // Re-acquire src pointer
                 for (auto& kf : keyframes) { if (kf.frame == src_frame) { src_kf = &kf; break; } }
            }
            
            // Allow merging types (e.g. adding Rot to a Pos keyframe)
            dst_kf->has_transform |= src_kf->has_transform;
            dst_kf->has_light |= src_kf->has_light;
            
             if (channel == ChannelType::Location) {
                 dst_kf->transform.position = src_kf->transform.position;
                 dst_kf->transform.has_position = src_kf->transform.has_position;
             } else if (channel == ChannelType::Rotation) {
                 dst_kf->transform.rotation = src_kf->transform.rotation;
                 dst_kf->transform.has_rotation = src_kf->transform.has_rotation;
             } else if (channel == ChannelType::Scale) {
                 dst_kf->transform.scale = src_kf->transform.scale;
                 dst_kf->transform.has_scale = src_kf->transform.has_scale;
             }
        }
        
        std::sort(it->second.keyframes.begin(), it->second.keyframes.end(),
            [](const Keyframe& a, const Keyframe& b) { return a.frame < b.frame; });
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
    std::string entity_name = obj->getNodeName().empty() ? 
        "Object_" + std::to_string(ctx.selection.selected.object_index) : 
        obj->getNodeName();
    
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
            kf.transform.has_pos_x = true;
            kf.transform.has_pos_y = true;
            kf.transform.has_pos_z = true;
            break;
        case KeyframeInsertType::Rotation:
            kf.transform.has_rotation = true;
            kf.transform.has_rot_x = true;
            kf.transform.has_rot_y = true;
            kf.transform.has_rot_z = true;
            break;
        case KeyframeInsertType::Scale:
            kf.transform.has_scale = true;
            kf.transform.has_scl_x = true;
            kf.transform.has_scl_y = true;
            kf.transform.has_scl_z = true;
            break;
        case KeyframeInsertType::LocRot:
            kf.transform.has_position = true;
            kf.transform.has_pos_x = true;
            kf.transform.has_pos_y = true;
            kf.transform.has_pos_z = true;
            kf.transform.has_rotation = true;
            kf.transform.has_rot_x = true;
            kf.transform.has_rot_y = true;
            kf.transform.has_rot_z = true;
            break;
        case KeyframeInsertType::All:
        default:
            kf.transform.has_position = true;
            kf.transform.has_pos_x = true;
            kf.transform.has_pos_y = true;
            kf.transform.has_pos_z = true;
            kf.transform.has_rotation = true;
            kf.transform.has_rot_x = true;
            kf.transform.has_rot_y = true;
            kf.transform.has_rot_z = true;
            kf.transform.has_scale = true;
            kf.transform.has_scl_x = true;
            kf.transform.has_scl_y = true;
            kf.transform.has_scl_z = true;
            break;
    }
    
    ctx.scene.timeline.insertKeyframe(entity_name, kf);
}

// ============================================================================
// GRAPH EDITOR — Value-to-pixel and pixel-to-value helpers
// ============================================================================
float TimelineWidget::valueToPixelY(float value, float canvas_height) const {
    // graph_value_center sits at the vertical centre of the canvas.
    // graph_pixels_per_unit controls vertical zoom.
    // Result is in canvas-local coordinates (0 at top, canvas_height at bottom).
    return canvas_height * 0.5f - (value - graph_value_center) * graph_pixels_per_unit;
}

float TimelineWidget::pixelYToValue(float y, float canvas_height) const {
    return graph_value_center - (y - canvas_height * 0.5f) / graph_pixels_per_unit;
}

// ============================================================================
// GRAPH EDITOR — Auto-fit the vertical view to the data range
// ============================================================================
void TimelineWidget::fitGraphView(UIContext& ctx, float canvas_height) {
    if (selected_track.empty()) return;
    auto [entity_name, chan] = parseTrackName(ctx.scene, selected_track);
    auto it = ctx.scene.timeline.tracks.find(entity_name);
    if (it == ctx.scene.timeline.tracks.end()) return;

    GraphTrackType track_type = getGraphTrackType(ctx, selected_track);
    float vmin = 1e18f, vmax = -1e18f;
    int count = 0;
    for (const auto& kf : it->second.keyframes) {
        if (track_type == GraphTrackType::Light) {
            if (!kf.has_light) continue;
            for (int ch = 0; ch < CURVE_LIGHT_CHANNEL_COUNT; ++ch) {
                if (!graph_channel_visible[ch]) continue;
                if (!kf.light.channelKeyed(ch)) continue;
                float v = kf.light.channelValue(ch);
                vmin = std::min(vmin, v);
                vmax = std::max(vmax, v);
                ++count;
            }
        } else if (track_type == GraphTrackType::Camera) {
            if (!kf.has_camera) continue;
            for (int ch = 0; ch < CURVE_CAM_CHANNEL_COUNT; ++ch) {
                if (!graph_channel_visible[ch]) continue;
                if (!kf.camera.channelKeyed(ch)) continue;
                float v = kf.camera.channelValue(ch);
                vmin = std::min(vmin, v);
                vmax = std::max(vmax, v);
                ++count;
            }
        } else if (track_type == GraphTrackType::Material) {
            if (!kf.has_material) continue;
            for (int ch = 0; ch < CURVE_MAT_CHANNEL_COUNT; ++ch) {
                if (!graph_channel_visible[ch]) continue;
                if (!kf.material.channelKeyed(ch)) continue;
                float v = kf.material.channelValue(ch);
                vmin = std::min(vmin, v);
                vmax = std::max(vmax, v);
                ++count;
            }
        } else {
            if (!kf.has_transform) continue;
            for (int ch = 0; ch < CURVE_CHANNEL_COUNT; ++ch) {
                if (!graph_channel_visible[ch]) continue;
                if (!kf.transform.channelKeyed(ch)) continue;
                float v = kf.transform.channelValue(ch);
                vmin = std::min(vmin, v);
                vmax = std::max(vmax, v);
                ++count;
            }
        }
    }
    if (count == 0) { graph_value_center = 0.0f; graph_pixels_per_unit = 40.0f; return; }

    float range = vmax - vmin;
    if (range < 0.01f) range = 2.0f; // avoid degenerate zoom
    graph_value_center = (vmin + vmax) * 0.5f;
    // Leave 15% padding on top and bottom
    graph_pixels_per_unit = (canvas_height * 0.7f) / range;
    graph_pixels_per_unit = std::clamp(graph_pixels_per_unit, 0.5f, 2000.0f);
}

// ============================================================================
// GRAPH EDITOR — Left panel: per-channel toggles
// ============================================================================
void TimelineWidget::drawGraphChannelList(UIContext& ctx, float list_width) {
    // Track name label
    if (!selected_track.empty()) {
        ImGui::TextColored(ImVec4(0.42f, 0.86f, 0.92f, 1.0f), "%s", selected_track.c_str());
    } else {
        ImGui::TextDisabled("(select a track)");
    }
    ImGui::Separator();

    GraphTrackType track_type = getGraphTrackType(ctx, selected_track);

    struct GraphChannelDef {
        int channel_index;
        const char* name;
        ImU32 color;
    };

    struct GraphGroupDef {
        const char* name;
        std::vector<GraphChannelDef> channels;
    };

    std::vector<GraphGroupDef> groups;
    if (track_type == GraphTrackType::Light) {
        groups = {
            { "Position", {
                { CURVE_LIGHT_POS_X, "Pos X", IM_COL32(255, 60, 60, 255) },
                { CURVE_LIGHT_POS_Y, "Pos Y", IM_COL32(80, 210, 80, 255) },
                { CURVE_LIGHT_POS_Z, "Pos Z", IM_COL32(60, 120, 255, 255) }
            }},
            { "Color", {
                { CURVE_LIGHT_COLOR_R, "Color R", IM_COL32(255, 60, 60, 255) },
                { CURVE_LIGHT_COLOR_G, "Color G", IM_COL32(80, 210, 80, 255) },
                { CURVE_LIGHT_COLOR_B, "Color B", IM_COL32(60, 120, 255, 255) }
            }},
            { "Intensity", {
                { CURVE_LIGHT_INTENSITY, "Intensity", IM_COL32(255, 180, 60, 255) }
            }},
            { "Direction", {
                { CURVE_LIGHT_DIR_X, "Dir X", IM_COL32(255, 60, 60, 255) },
                { CURVE_LIGHT_DIR_Y, "Dir Y", IM_COL32(80, 210, 80, 255) },
                { CURVE_LIGHT_DIR_Z, "Dir Z", IM_COL32(60, 120, 255, 255) }
            }}
        };
    } else if (track_type == GraphTrackType::Camera) {
        groups = {
            { "Position", {
                { CURVE_CAM_POS_X, "Pos X", IM_COL32(255, 60, 60, 255) },
                { CURVE_CAM_POS_Y, "Pos Y", IM_COL32(80, 210, 80, 255) },
                { CURVE_CAM_POS_Z, "Pos Z", IM_COL32(60, 120, 255, 255) }
            }},
            { "Target", {
                { CURVE_CAM_TGT_X, "Target X", IM_COL32(255, 60, 60, 255) },
                { CURVE_CAM_TGT_Y, "Target Y", IM_COL32(80, 210, 80, 255) },
                { CURVE_CAM_TGT_Z, "Target Z", IM_COL32(60, 120, 255, 255) }
            }},
            { "Settings", {
                { CURVE_CAM_FOV, "FOV", IM_COL32(60, 120, 255, 255) },
                { CURVE_CAM_FOCUS_DIST, "Focus Dist", IM_COL32(200, 150, 255, 255) },
                { CURVE_CAM_LENS_RAD, "Lens Rad", IM_COL32(255, 120, 200, 255) }
            }}
        };
    } else if (track_type == GraphTrackType::Material) {
        groups = {
            { "Albedo & Opacity", {
                { CURVE_MAT_ALBEDO_R, "Albedo R", IM_COL32(255, 60, 60, 255) },
                { CURVE_MAT_ALBEDO_G, "Albedo G", IM_COL32(80, 210, 80, 255) },
                { CURVE_MAT_ALBEDO_B, "Albedo B", IM_COL32(60, 120, 255, 255) },
                { CURVE_MAT_OPACITY, "Opacity", IM_COL32(200, 200, 200, 255) }
            }},
            { "PBR Core", {
                { CURVE_MAT_ROUGHNESS, "Roughness", IM_COL32(180, 255, 60, 255) },
                { CURVE_MAT_METALLIC, "Metallic", IM_COL32(160, 160, 160, 255) },
                { CURVE_MAT_CLEARCOAT, "Clearcoat", IM_COL32(60, 220, 255, 255) },
                { CURVE_MAT_TRANSMISSION, "Transmission", IM_COL32(60, 120, 255, 255) },
                { CURVE_MAT_IOR, "IOR", IM_COL32(100, 200, 200, 255) }
            }},
            { "Emission", {
                { CURVE_MAT_EMISSION_R, "Emission R", IM_COL32(255, 100, 100, 255) },
                { CURVE_MAT_EMISSION_G, "Emission G", IM_COL32(100, 255, 100, 255) },
                { CURVE_MAT_EMISSION_B, "Emission B", IM_COL32(100, 100, 255, 255) },
                { CURVE_MAT_EMISSION_STRENGTH, "Emission Strength", IM_COL32(255, 220, 60, 255) }
            }},
            { "Other Settings", {
                { CURVE_MAT_NORMAL_STRENGTH, "Normal Str", IM_COL32(255, 150, 100, 255) }
            }}
        };
    } else {
        groups = {
            { "Location", {
                { CURVE_POS_X, "X", IM_COL32(255, 60, 60, 255) },
                { CURVE_POS_Y, "Y", IM_COL32(80, 210, 80, 255) },
                { CURVE_POS_Z, "Z", IM_COL32(60, 120, 255, 255) }
            }},
            { "Rotation", {
                { CURVE_ROT_X, "X", IM_COL32(255, 100, 100, 255) },
                { CURVE_ROT_Y, "Y", IM_COL32(100, 230, 100, 255) },
                { CURVE_ROT_Z, "Z", IM_COL32(100, 160, 255, 255) }
            }},
            { "Scale", {
                { CURVE_SCL_X, "X", IM_COL32(255, 180, 60, 255) },
                { CURVE_SCL_Y, "Y", IM_COL32(180, 255, 60, 255) },
                { CURVE_SCL_Z, "Z", IM_COL32(60, 220, 255, 255) }
            }}
        };
    }

    for (size_t g = 0; g < groups.size(); ++g) {
        ImGui::PushID(g);
        auto& group = groups[g];
        
        bool group_vis = false;
        for (auto& ch_def : group.channels) {
            if (graph_channel_visible[ch_def.channel_index]) {
                group_vis = true;
                break;
            }
        }
        
        std::string group_key = std::string(group.name) + "##" + std::to_string((int)track_type);
        if (graph_groups_expanded.find(group_key) == graph_groups_expanded.end()) {
            graph_groups_expanded[group_key] = true;
        }
        bool& expanded = graph_groups_expanded[group_key];

        // 1. Draw bulk checkbox on the left first (avoiding overlap flags)
        bool bulk_toggle = group_vis;
        if (ImGui::Checkbox("##bulk", &bulk_toggle)) {
            for (auto& ch_def : group.channels) {
                graph_channel_visible[ch_def.channel_index] = bulk_toggle;
            }
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Toggle visibility of all channels in this group");
        }
        
        ImGui::SameLine();

        // 2. Draw collapsible node header
        ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_FramePadding;
        if (expanded) flags |= ImGuiTreeNodeFlags_DefaultOpen;
        
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(2.0f, 2.0f));
        bool node_open = ImGui::TreeNodeEx(group.name, flags);
        ImGui::PopStyleVar();
        
        expanded = node_open;
        if (node_open) {
            for (auto& ch_def : group.channels) {
                int ch = ch_def.channel_index;
                ImGui::PushID(ch);
                ImGui::Indent(16.0f);
                ImVec4 col = ImGui::ColorConvertU32ToFloat4(ch_def.color);
                ImGui::PushStyleColor(ImGuiCol_Text, col);
                bool vis = graph_channel_visible[ch];
                if (ImGui::Checkbox(ch_def.name, &vis)) {
                    graph_channel_visible[ch] = vis;
                }
                ImGui::PopStyleColor();
                ImGui::Unindent(16.0f);
                ImGui::PopID();
            }
            ImGui::TreePop();
        }
        ImGui::PopID();
    }

    ImGui::Separator();

    // Fit button
    if (ImGui::Button("Fit (F)", ImVec2(-1, 0))) {
        graph_fit_pending = true;
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Auto-fit curves to view");

    // Interpolation mode for selected key
    if (graph_sel_channel >= 0 && graph_sel_frame >= 0 && !selected_track.empty()) {
        auto [ent, _ch] = parseTrackName(ctx.scene, selected_track);
        auto tit = ctx.scene.timeline.tracks.find(ent);
        if (tit != ctx.scene.timeline.tracks.end()) {
            Keyframe* kf = tit->second.getKeyframeAt(graph_sel_frame);
            if (kf) {
                ChannelKeyMeta* m = nullptr;
                const char* chName = "?";
                if (track_type == GraphTrackType::Light && kf->has_light && graph_sel_channel < CURVE_LIGHT_CHANNEL_COUNT) {
                    m = &kf->light.curve[graph_sel_channel];
                    chName = LightKeyframe::channelName(graph_sel_channel);
                } else if (track_type == GraphTrackType::Camera && kf->has_camera && graph_sel_channel < CURVE_CAM_CHANNEL_COUNT) {
                    m = &kf->camera.curve[graph_sel_channel];
                    chName = CameraKeyframe::channelName(graph_sel_channel);
                } else if (track_type == GraphTrackType::Material && kf->has_material && graph_sel_channel < CURVE_MAT_CHANNEL_COUNT) {
                    m = &kf->material.curve[graph_sel_channel];
                    chName = MaterialKeyframe::channelName(graph_sel_channel);
                } else if (track_type == GraphTrackType::Transform && kf->has_transform && graph_sel_channel < CURVE_CHANNEL_COUNT) {
                    m = &kf->transform.curve[graph_sel_channel];
                    chName = TransformKeyframe::channelName(graph_sel_channel);
                }

                if (m) {
                    ImGui::Separator();
                    ImGui::TextDisabled("Key: %s", chName);
                    ImGui::TextDisabled("Key Interp:");
                    int interp_idx = static_cast<int>(m->interp);
                    const char* interpNames[] = { "Constant", "Linear", "Bezier" };
                    ImGui::PushItemWidth(-1);
                    if (ImGui::Combo("##KeyInterp", &interp_idx, interpNames, 3)) {
                        // Apply to both adjacent segments so the change is always visible.
                        applyKeyInterpBothSides(tit->second, track_type, graph_sel_channel,
                                                graph_sel_frame, static_cast<KeyInterp>(interp_idx));
                        anim_reapply_requested_ = true;
                    }
                    ImGui::PopItemWidth();

                    if (m->interp == KeyInterp::Bezier) {
                        if (ImGui::Checkbox("Auto tangent", &m->auto_tangent)) {
                            if (m->auto_tangent) {
                                tit->second.refreshAutoTangents();
                                anim_reapply_requested_ = true;
                            }
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// GRAPH EDITOR — Right panel: the curve canvas
// ============================================================================
void TimelineWidget::drawGraphCanvas(UIContext& ctx, float canvas_width, float canvas_height) {
    if (canvas_width <= 0 || canvas_height <= 0) return;

    // --- Auto-fit on first open / when requested ---
    if (graph_fit_pending) {
        fitGraphView(ctx, canvas_height);
        graph_fit_pending = false;
    }

    // 'F' key = auto-fit
    if (ImGui::IsWindowFocused() && ImGui::IsKeyPressed(ImGuiKey_F) && !ImGui::GetIO().WantTextInput) {
        fitGraphView(ctx, canvas_height);
    }

    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
    ImVec2 canvas_end = ImVec2(canvas_pos.x + canvas_width, canvas_pos.y + canvas_height);

    // Background
    dl->AddRectFilled(canvas_pos, canvas_end, IM_COL32(28, 28, 32, 255));

    // --- Vertical zoom / pan with mouse wheel ---
    if (ImGui::IsWindowHovered()) {
        ImGuiIO& io = ImGui::GetIO();
        if (io.MouseWheel != 0.0f) {
            if (io.KeyCtrl) {
                // Ctrl+Wheel = vertical zoom
                float factor = (io.MouseWheel > 0) ? 1.15f : (1.0f / 1.15f);
                graph_pixels_per_unit *= factor;
                graph_pixels_per_unit = std::clamp(graph_pixels_per_unit, 0.5f, 5000.0f);
            } else if (io.KeyShift) {
                // Shift+Wheel = vertical pan
                float pan_amount = 30.0f / graph_pixels_per_unit;
                graph_value_center += io.MouseWheel * pan_amount;
            } else {
                // Plain wheel = horizontal zoom (same as dope sheet)
                zoom *= (io.MouseWheel > 0) ? 1.1f : (1.0f / 1.1f);
                zoom = std::clamp(zoom, 0.1f, 50.0f);
            }
        }
        // Middle-mouse drag = pan
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Middle)) {
            ImVec2 delta = io.MouseDelta;
            float frame_range = (float)(end_frame - start_frame);
            if (canvas_width * zoom > 0)
                pan_offset -= delta.x / (canvas_width * zoom) * frame_range;
            if (graph_pixels_per_unit > 0)
                graph_value_center += delta.y / graph_pixels_per_unit;
        }
    }

    // --- Draw horizontal grid lines (value axis) ---
    {
        // Determine a good spacing for value grid lines
        float visible_range = canvas_height / graph_pixels_per_unit;
        float step = 1.0f;
        if (visible_range > 50.0f) step = 10.0f;
        else if (visible_range > 20.0f) step = 5.0f;
        else if (visible_range > 5.0f) step = 1.0f;
        else if (visible_range > 1.0f) step = 0.5f;
        else step = 0.1f;

        float v_bottom = pixelYToValue(canvas_height, canvas_height);
        float v_top = pixelYToValue(0, canvas_height);
        float v_start = std::floor(std::min(v_bottom, v_top) / step) * step;
        float v_end = std::ceil(std::max(v_bottom, v_top) / step) * step;

        for (float v = v_start; v <= v_end; v += step) {
            float py = canvas_pos.y + valueToPixelY(v, canvas_height);
            if (py < canvas_pos.y || py > canvas_end.y) continue;
            bool is_zero = std::abs(v) < step * 0.01f;
            ImU32 lineCol = is_zero ? IM_COL32(100, 100, 100, 180) : IM_COL32(50, 50, 55, 120);
            dl->AddLine(ImVec2(canvas_pos.x, py), ImVec2(canvas_end.x, py), lineCol);
            char buf[32];
            snprintf(buf, sizeof(buf), "%.1f", v);
            dl->AddText(ImVec2(canvas_pos.x + 4, py - 8), IM_COL32(140, 140, 140, 200), buf);
        }
    }

    // --- Draw vertical grid lines (frame axis, reusing dope-sheet logic) ---
    {
        int frame_range = end_frame - start_frame;
        if (frame_range > 0) {
            int step = 1;
            float px_per_frame = canvas_width * zoom / frame_range;
            if (px_per_frame < 4) step = 50;
            else if (px_per_frame < 10) step = 20;
            else if (px_per_frame < 20) step = 10;
            else if (px_per_frame < 40) step = 5;

            int f_start = (start_frame / step) * step;
            for (int f = f_start; f <= end_frame; f += step) {
                int px = frameToPixelX(f, canvas_width);
                if (px < 0 || px > (int)canvas_width) continue;
                float xp = canvas_pos.x + px;
                dl->AddLine(ImVec2(xp, canvas_pos.y), ImVec2(xp, canvas_end.y), IM_COL32(50, 50, 55, 100));
                char buf[16]; snprintf(buf, sizeof(buf), "%d", f);
                dl->AddText(ImVec2(xp + 2, canvas_pos.y + 2), IM_COL32(120, 120, 120, 200), buf);
            }
        }
    }

    // --- Resolve track data ---
    auto [entity_name, _chan] = parseTrackName(ctx.scene, selected_track);
    auto track_it = ctx.scene.timeline.tracks.find(entity_name);

    GraphTrackType track_type = getGraphTrackType(ctx, selected_track);
    int total_channels = CURVE_CHANNEL_COUNT;
    if (track_type == GraphTrackType::Light) total_channels = CURVE_LIGHT_CHANNEL_COUNT;
    else if (track_type == GraphTrackType::Camera) total_channels = CURVE_CAM_CHANNEL_COUNT;
    else if (track_type == GraphTrackType::Material) total_channels = CURVE_MAT_CHANNEL_COUNT;

    auto getChannelColor = [&](GraphTrackType type, int ch) -> ImU32 {
        if (type == GraphTrackType::Light) {
            static const ImU32 colors[CURVE_LIGHT_CHANNEL_COUNT] = {
                IM_COL32(255, 60, 60, 255), IM_COL32(80, 210, 80, 255), IM_COL32(60, 120, 255, 255),
                IM_COL32(255, 60, 60, 255), IM_COL32(80, 210, 80, 255), IM_COL32(60, 120, 255, 255),
                IM_COL32(255, 180, 60, 255),
                IM_COL32(255, 60, 60, 255), IM_COL32(80, 210, 80, 255), IM_COL32(60, 120, 255, 255)
            };
            return colors[ch];
        } else if (type == GraphTrackType::Camera) {
            static const ImU32 colors[CURVE_CAM_CHANNEL_COUNT] = {
                IM_COL32(255, 60, 60, 255), IM_COL32(80, 210, 80, 255), IM_COL32(60, 120, 255, 255),
                IM_COL32(255, 60, 60, 255), IM_COL32(80, 210, 80, 255), IM_COL32(60, 120, 255, 255),
                IM_COL32(60, 120, 255, 255), IM_COL32(200, 150, 255, 255), IM_COL32(255, 120, 200, 255)
            };
            return colors[ch];
        } else if (type == GraphTrackType::Material) {
            static const ImU32 colors[CURVE_MAT_CHANNEL_COUNT] = {
                IM_COL32(255, 60, 60, 255), IM_COL32(80, 210, 80, 255), IM_COL32(60, 120, 255, 255),
                IM_COL32(200, 200, 200, 255),
                IM_COL32(180, 255, 60, 255), IM_COL32(160, 160, 160, 255), IM_COL32(60, 220, 255, 255), IM_COL32(60, 120, 255, 255), IM_COL32(100, 200, 200, 255),
                IM_COL32(255, 100, 100, 255), IM_COL32(100, 255, 100, 255), IM_COL32(100, 100, 255, 255),
                IM_COL32(255, 150, 100, 255),
                IM_COL32(255, 220, 60, 255)
            };
            return colors[ch];
        } else {
            static const ImU32 colors[CURVE_CHANNEL_COUNT] = {
                IM_COL32(255, 60,  60,  255), IM_COL32(80,  210, 80,  255), IM_COL32(60,  120, 255, 255),
                IM_COL32(255, 100, 100, 255), IM_COL32(100, 230, 100, 255), IM_COL32(100, 160, 255, 255),
                IM_COL32(255, 180, 60,  255), IM_COL32(180, 255, 60,  255), IM_COL32(60,  220, 255, 255)
            };
            return colors[ch];
        }
    };

    auto channelKeyed = [&](const Keyframe& kf, int ch) -> bool {
        if (track_type == GraphTrackType::Light) return kf.has_light && kf.light.channelKeyed(ch);
        if (track_type == GraphTrackType::Camera) return kf.has_camera && kf.camera.channelKeyed(ch);
        if (track_type == GraphTrackType::Material) return kf.has_material && kf.material.channelKeyed(ch);
        return kf.has_transform && kf.transform.channelKeyed(ch);
    };

    auto channelValue = [&](const Keyframe& kf, int ch) -> float {
        if (track_type == GraphTrackType::Light) return kf.light.channelValue(ch);
        if (track_type == GraphTrackType::Camera) return kf.camera.channelValue(ch);
        if (track_type == GraphTrackType::Material) return kf.material.channelValue(ch);
        return kf.transform.channelValue(ch);
    };

    auto getCurveMeta = [&](const Keyframe& kf, int ch) -> const ChannelKeyMeta& {
        if (track_type == GraphTrackType::Light) return kf.light.curve[ch];
        if (track_type == GraphTrackType::Camera) return kf.camera.curve[ch];
        if (track_type == GraphTrackType::Material) return kf.material.curve[ch];
        return kf.transform.curve[ch];
    };

    auto getCurveMetaMutable = [&](Keyframe& kf, int ch) -> ChannelKeyMeta& {
        if (track_type == GraphTrackType::Light) return kf.light.curve[ch];
        if (track_type == GraphTrackType::Camera) return kf.camera.curve[ch];
        if (track_type == GraphTrackType::Material) return kf.material.curve[ch];
        return kf.transform.curve[ch];
    };

    auto setChannelVal = [&](Keyframe& kf, int ch, float v) {
        if (track_type == GraphTrackType::Light) kf.light.setChannelValue(ch, v);
        else if (track_type == GraphTrackType::Camera) kf.camera.setChannelValue(ch, v);
        else if (track_type == GraphTrackType::Material) kf.material.setChannelValue(ch, v);
        else kf.transform.setChannelValue(ch, v);
    };

    auto setChannelKey = [&](Keyframe& kf, int ch, bool keyed) {
        if (track_type == GraphTrackType::Light) kf.light.setChannelKeyed(ch, keyed);
        else if (track_type == GraphTrackType::Camera) kf.camera.setChannelKeyed(ch, keyed);
        else if (track_type == GraphTrackType::Material) kf.material.setChannelKeyed(ch, keyed);
        else kf.transform.setChannelKeyed(ch, keyed);
    };

    auto getChannelName = [&](int ch) -> const char* {
        if (track_type == GraphTrackType::Light) return LightKeyframe::channelName(ch);
        if (track_type == GraphTrackType::Camera) return CameraKeyframe::channelName(ch);
        if (track_type == GraphTrackType::Material) return MaterialKeyframe::channelName(ch);
        return TransformKeyframe::channelName(ch);
    };

    if (track_it != ctx.scene.timeline.tracks.end()) {
        auto& track = track_it->second;
        const auto& keyframes = track.keyframes;

        // --- Draw curves ---
        for (int ch = 0; ch < total_channels; ++ch) {
            if (!graph_channel_visible[ch]) continue;
            ImU32 curveCol = getChannelColor(track_type, ch);
            // Dimmer colour when not the selected channel
            if (graph_sel_channel >= 0 && ch != graph_sel_channel)
                curveCol = (curveCol & 0x00FFFFFF) | (0x60 << 24); // semi-transparent

            // Collect keyed frames for this channel
            struct ChKey { int frame; float value; const ChannelKeyMeta* meta; };
            std::vector<ChKey> keys;
            for (const auto& kf : keyframes) {
                if (!channelKeyed(kf, ch)) continue;
                keys.push_back({ kf.frame, channelValue(kf, ch), &getCurveMeta(kf, ch) });
            }
            if (keys.empty()) continue;

            // Draw curve segments between consecutive keys
            for (size_t i = 0; i + 1 < keys.size(); ++i) {
                const auto& k0 = keys[i];
                const auto& k1 = keys[i + 1];
                int px0 = frameToPixelX(k0.frame, canvas_width);
                int px1 = frameToPixelX(k1.frame, canvas_width);
                // Subdivide into small line segments for bezier curves
                int steps = std::max(2, (px1 - px0) / 3);
                steps = std::min(steps, 200); // cap to prevent slowdown
                ImVec2 prev;
                for (int s = 0; s <= steps; ++s) {
                    float t = (float)s / (float)steps;
                    float frame_f = k0.frame + (k1.frame - k0.frame) * t;
                    float val = evalCurveSegment(
                        (float)k0.frame, k0.value, *k0.meta,
                        (float)k1.frame, k1.value, *k1.meta,
                        frame_f);
                    float px = canvas_pos.x + frameToPixelX((int)std::round(frame_f), canvas_width);
                    // More precise X: interpolate linearly in pixel space
                    px = canvas_pos.x + px0 + (px1 - px0) * t;
                    float py = canvas_pos.y + valueToPixelY(val, canvas_height);
                    ImVec2 cur(px, py);
                    if (s > 0) {
                        dl->AddLine(prev, cur, curveCol, 2.0f);
                    }
                    prev = cur;
                }
            }

            // Extrapolation: flat before first key and after last key
            if (!keys.empty()) {
                // Before first
                int px_first = frameToPixelX(keys.front().frame, canvas_width);
                if (px_first > 0) {
                    float py = canvas_pos.y + valueToPixelY(keys.front().value, canvas_height);
                    dl->AddLine(ImVec2(canvas_pos.x, py), ImVec2(canvas_pos.x + px_first, py),
                                (curveCol & 0x00FFFFFF) | (0x40 << 24), 1.0f);
                }
                // After last
                int px_last = frameToPixelX(keys.back().frame, canvas_width);
                if (px_last < (int)canvas_width) {
                    float py = canvas_pos.y + valueToPixelY(keys.back().value, canvas_height);
                    dl->AddLine(ImVec2(canvas_pos.x + px_last, py), ImVec2(canvas_end.x, py),
                                (curveCol & 0x00FFFFFF) | (0x40 << 24), 1.0f);
                }
            }

            // --- Draw key dots + Bezier handles ---
            for (size_t ki = 0; ki < keys.size(); ++ki) {
                const auto& k = keys[ki];
                float kx = canvas_pos.x + frameToPixelX(k.frame, canvas_width);
                float ky = canvas_pos.y + valueToPixelY(k.value, canvas_height);
                bool is_selected = (ch == graph_sel_channel && k.frame == graph_sel_frame);

                // Bezier handles (only for bezier interp and when selected or hovered)
                if (k.meta->interp == KeyInterp::Bezier || (ki > 0 && keys[ki-1].meta->interp == KeyInterp::Bezier)) {
                    // In-handle (from previous key)
                    float in_hx = kx + k.meta->in_dx * (canvas_width * zoom / (float)(end_frame - start_frame));
                    float in_hy = ky - k.meta->in_dy * graph_pixels_per_unit;
                    dl->AddLine(ImVec2(kx, ky), ImVec2(in_hx, in_hy), IM_COL32(160, 160, 160, 140), 1.0f);
                    dl->AddCircleFilled(ImVec2(in_hx, in_hy), is_selected ? 4.0f : 3.0f, IM_COL32(200, 200, 200, 180));

                    // Out-handle
                    if (k.meta->interp == KeyInterp::Bezier) {
                        float out_hx = kx + k.meta->out_dx * (canvas_width * zoom / (float)(end_frame - start_frame));
                        float out_hy = ky - k.meta->out_dy * graph_pixels_per_unit;
                        dl->AddLine(ImVec2(kx, ky), ImVec2(out_hx, out_hy), IM_COL32(160, 160, 160, 140), 1.0f);
                        dl->AddCircleFilled(ImVec2(out_hx, out_hy), is_selected ? 4.0f : 3.0f, IM_COL32(200, 200, 200, 180));
                    }
                }

                // Key diamond/dot
                float dot_r = is_selected ? 6.0f : 4.5f;
                ImU32 dotCol = is_selected ? IM_COL32(255, 255, 255, 255) : curveCol;
                dl->AddCircleFilled(ImVec2(kx, ky), dot_r, dotCol);
                if (is_selected) {
                    dl->AddCircle(ImVec2(kx, ky), dot_r + 2.0f, IM_COL32(255, 255, 100, 200), 0, 2.0f);
                }
            }
        }

        // --- Key interaction (click to select, drag to edit value/handles) ---
        ImGuiIO& io = ImGui::GetIO();
        ImVec2 mouse = io.MousePos;
        bool in_canvas = mouse.x >= canvas_pos.x && mouse.x <= canvas_end.x &&
                         mouse.y >= canvas_pos.y && mouse.y <= canvas_end.y;

        if (in_canvas && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !io.KeyCtrl) {
            bool handle_clicked = false;
            // 1. Check if we clicked on the selected key's handles
            if (graph_sel_channel >= 0 && graph_sel_frame >= 0) {
                Keyframe* kf = track.getKeyframeAt(graph_sel_frame);
                bool has_data = (track_type == GraphTrackType::Light && kf && kf->has_light) ||
                                (track_type == GraphTrackType::Camera && kf && kf->has_camera) ||
                                (track_type == GraphTrackType::Material && kf && kf->has_material) ||
                                (track_type == GraphTrackType::Transform && kf && kf->has_transform);
                if (kf && has_data) {
                    ChannelKeyMeta& m = getCurveMetaMutable(*kf, graph_sel_channel);
                    float kx = canvas_pos.x + frameToPixelX(graph_sel_frame, canvas_width);
                    float ky = canvas_pos.y + valueToPixelY(channelValue(*kf, graph_sel_channel), canvas_height);
                    float scale_x = (canvas_width * zoom / (float)(end_frame - start_frame));

                    if (scale_x > 0.0f) {
                        // In-handle
                        float in_hx = kx + m.in_dx * scale_x;
                        float in_hy = ky - m.in_dy * graph_pixels_per_unit;
                        float dist_in = std::hypot(mouse.x - in_hx, mouse.y - in_hy);

                        // Out-handle
                        float out_hx = kx + m.out_dx * scale_x;
                        float out_hy = ky - m.out_dy * graph_pixels_per_unit;
                        float dist_out = std::hypot(mouse.x - out_hx, mouse.y - out_hy);

                        if (dist_in < 8.0f) {
                            graph_drag_mode = 2; // in-handle
                            handle_clicked = true;
                        } else if (dist_out < 8.0f && m.interp == KeyInterp::Bezier) {
                            graph_drag_mode = 3; // out-handle
                            handle_clicked = true;
                        }
                    }
                }
            }

            // 2. If no handle was clicked, check closest key dot
            if (!handle_clicked) {
                float best_dist = 15.0f; // max click distance in pixels
                int best_ch = -1, best_frame = -1;
                for (int ch = 0; ch < total_channels; ++ch) {
                    if (!graph_channel_visible[ch]) continue;
                    for (const auto& kf : keyframes) {
                        if (!channelKeyed(kf, ch)) continue;
                        float kx = canvas_pos.x + frameToPixelX(kf.frame, canvas_width);
                        float ky = canvas_pos.y + valueToPixelY(channelValue(kf, ch), canvas_height);
                        float dist = std::hypot(mouse.x - kx, mouse.y - ky);
                        if (dist < best_dist) {
                            best_dist = dist;
                            best_ch = ch;
                            best_frame = kf.frame;
                        }
                    }
                }
                if (best_ch >= 0) {
                    graph_sel_channel = best_ch;
                    graph_sel_frame = best_frame;
                    drag_start_frame = best_frame; // Remember starting frame for horizontal drag
                    graph_drag_mode = 1; // key drag
                } else {
                    graph_sel_channel = -1;
                    graph_sel_frame = -1;
                    graph_drag_mode = 0;
                }
            }
        }

        // Drag selected key value (Y) AND frame (X) — both applied LIVE so the key
        // tracks the cursor in real time. Horizontal used to defer its moveKeyframe to
        // release, which made the X move feel unpredictable next to the live Y move.
        if (graph_drag_mode == 1 && ImGui::IsMouseDragging(ImGuiMouseButton_Left) &&
            graph_sel_channel >= 0 && graph_sel_frame >= 0) {
            Keyframe* kf = track.getKeyframeAt(graph_sel_frame);
            bool has_data = (track_type == GraphTrackType::Light && kf && kf->has_light) ||
                            (track_type == GraphTrackType::Camera && kf && kf->has_camera) ||
                            (track_type == GraphTrackType::Material && kf && kf->has_material) ||
                            (track_type == GraphTrackType::Transform && kf && kf->has_transform);
            if (kf && has_data) {
                // 1. Apply value (Y) at the current frame first, so the move below carries it.
                float new_val = pixelYToValue(mouse.y - canvas_pos.y, canvas_height);
                setChannelVal(*kf, graph_sel_channel, new_val);

                // 2. Commit the horizontal frame move (X) live — one hop per integer-frame
                //    crossing (cheap; only fires when the frame actually changes). NOTE:
                //    moveKeyframe() mutates the keyframe vector, so `kf` is dangling after it.
                int new_frame = pixelXToFrame(mouse.x - canvas_pos.x, canvas_width);
                new_frame = std::clamp(new_frame, start_frame, end_frame);
                if (new_frame != graph_sel_frame && track_it != ctx.scene.timeline.tracks.end()) {
                    std::string specific_track = entity_name + "." + channelSubtrackSuffix(track_type, graph_sel_channel);
                    moveKeyframe(ctx, specific_track, graph_sel_frame, new_frame);
                    graph_sel_frame = new_frame;
                    drag_start_frame = new_frame; // keeps the release-time move a no-op
                    tracks_dirty = true;
                }
                track.refreshAutoTangents(); // operates on the live track (kf may be stale here)
                anim_reapply_requested_ = true;
            }
        }

        // Handle dragging curve tangents
        if ((graph_drag_mode == 2 || graph_drag_mode == 3) && ImGui::IsMouseDragging(ImGuiMouseButton_Left) &&
            graph_sel_channel >= 0 && graph_sel_frame >= 0) {
            Keyframe* kf = track.getKeyframeAt(graph_sel_frame);
            bool has_data = (track_type == GraphTrackType::Light && kf && kf->has_light) ||
                            (track_type == GraphTrackType::Camera && kf && kf->has_camera) ||
                            (track_type == GraphTrackType::Material && kf && kf->has_material) ||
                            (track_type == GraphTrackType::Transform && kf && kf->has_transform);
            if (kf && has_data) {
                ChannelKeyMeta& m = getCurveMetaMutable(*kf, graph_sel_channel);
                float kx = canvas_pos.x + frameToPixelX(graph_sel_frame, canvas_width);
                float ky = canvas_pos.y + valueToPixelY(channelValue(*kf, graph_sel_channel), canvas_height);
                float scale_x = (canvas_width * zoom / (float)(end_frame - start_frame));

                if (scale_x > 0.0f) {
                    float delta_x = (mouse.x - kx) / scale_x;
                    float delta_y = (ky - mouse.y) / graph_pixels_per_unit;

                    if (graph_drag_mode == 2) {
                        m.auto_tangent = false;
                        m.in_dx = std::min(delta_x, 0.0f);
                        m.in_dy = delta_y;
                        float slope = (m.in_dx != 0.0f) ? (m.in_dy / m.in_dx) : 0.0f;
                        m.out_dy = slope * m.out_dx; // Align out-handle
                    } else if (graph_drag_mode == 3) {
                        m.auto_tangent = false;
                        m.out_dx = std::max(delta_x, 0.0f);
                        m.out_dy = delta_y;
                        float slope = (m.out_dx != 0.0f) ? (m.out_dy / m.out_dx) : 0.0f;
                        m.in_dy = slope * m.in_dx; // Align in-handle
                    }
                    anim_reapply_requested_ = true;
                }
            }
        }

        if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
            if (graph_drag_mode == 1 && graph_sel_channel >= 0 && graph_sel_frame >= 0 && track_it != ctx.scene.timeline.tracks.end()) {
                int new_frame = pixelXToFrame(mouse.x - canvas_pos.x, canvas_width);
                new_frame = std::clamp(new_frame, start_frame, end_frame);
                if (new_frame != drag_start_frame) {
                    std::string specific_track = entity_name + "." + channelSubtrackSuffix(track_type, graph_sel_channel);
                    moveKeyframe(ctx, specific_track, drag_start_frame, new_frame);
                    graph_sel_frame = new_frame;
                    tracks_dirty = true;
                    anim_reapply_requested_ = true;
                }
            }
            graph_drag_mode = 0;
        }

        // Keyboard delete shortcut for selected curve key
        if (ImGui::IsWindowFocused() && (ImGui::IsKeyPressed(ImGuiKey_Delete) || ImGui::IsKeyPressed(ImGuiKey_X)) &&
            !ImGui::GetIO().WantTextInput && graph_sel_channel >= 0 && graph_sel_frame >= 0) {
            Keyframe* kf = track.getKeyframeAt(graph_sel_frame);
            bool has_data = (track_type == GraphTrackType::Light && kf && kf->has_light && graph_sel_channel < CURVE_LIGHT_CHANNEL_COUNT) ||
                            (track_type == GraphTrackType::Camera && kf && kf->has_camera && graph_sel_channel < CURVE_CAM_CHANNEL_COUNT) ||
                            (track_type == GraphTrackType::Material && kf && kf->has_material && graph_sel_channel < CURVE_MAT_CHANNEL_COUNT) ||
                            (track_type == GraphTrackType::Transform && kf && kf->has_transform && graph_sel_channel < CURVE_CHANNEL_COUNT);
            if (kf && has_data) {
                setChannelKey(*kf, graph_sel_channel, false);
                if (track_type == GraphTrackType::Light) {
                    kf->light.refreshCompoundFlags();
                    bool any_keyed = false;
                    for (int c = 0; c < CURVE_LIGHT_CHANNEL_COUNT; ++c)
                        if (kf->light.channelKeyed(c)) { any_keyed = true; break; }
                    if (!any_keyed) kf->has_light = false;
                } else if (track_type == GraphTrackType::Camera) {
                    kf->camera.refreshCompoundFlags();
                    bool any_keyed = false;
                    for (int c = 0; c < CURVE_CAM_CHANNEL_COUNT; ++c)
                        if (kf->camera.channelKeyed(c)) { any_keyed = true; break; }
                    if (!any_keyed) kf->has_camera = false;
                } else if (track_type == GraphTrackType::Material) {
                    kf->material.refreshCompoundFlags();
                    bool any_keyed = false;
                    for (int c = 0; c < CURVE_MAT_CHANNEL_COUNT; ++c)
                        if (kf->material.channelKeyed(c)) { any_keyed = true; break; }
                    if (!any_keyed) kf->has_material = false;
                } else {
                    kf->transform.refreshCompoundFlags();
                    bool any_keyed = false;
                    for (int c = 0; c < CURVE_CHANNEL_COUNT; ++c)
                        if (kf->transform.channelKeyed(c)) { any_keyed = true; break; }
                    if (!any_keyed) kf->has_transform = false;
                }
                graph_sel_channel = -1;
                graph_sel_frame = -1;
                tracks_dirty = true;
                anim_reapply_requested_ = true;
            }
        }

        // Right-click context menu for key operations
        if (in_canvas && ImGui::IsMouseClicked(ImGuiMouseButton_Right) && graph_sel_channel >= 0 && graph_sel_frame >= 0) {
            ImGui::OpenPopup("GraphKeyContextMenu");
        }
        if (ImGui::BeginPopup("GraphKeyContextMenu")) {
            Keyframe* kf = track.getKeyframeAt(graph_sel_frame);
            bool has_data = (track_type == GraphTrackType::Light && kf && kf->has_light && graph_sel_channel >= 0 && graph_sel_channel < CURVE_LIGHT_CHANNEL_COUNT) ||
                            (track_type == GraphTrackType::Camera && kf && kf->has_camera && graph_sel_channel >= 0 && graph_sel_channel < CURVE_CAM_CHANNEL_COUNT) ||
                            (track_type == GraphTrackType::Material && kf && kf->has_material && graph_sel_channel >= 0 && graph_sel_channel < CURVE_MAT_CHANNEL_COUNT) ||
                            (track_type == GraphTrackType::Transform && kf && kf->has_transform && graph_sel_channel >= 0 && graph_sel_channel < CURVE_CHANNEL_COUNT);
            if (kf && has_data) {
                ChannelKeyMeta& m = getCurveMetaMutable(*kf, graph_sel_channel);
                ImGui::TextDisabled("Key: %s @ %d", getChannelName(graph_sel_channel), graph_sel_frame);
                ImGui::Separator();

                if (ImGui::MenuItem("Constant", nullptr, m.interp == KeyInterp::Constant)) {
                    applyKeyInterpBothSides(track, track_type, graph_sel_channel, graph_sel_frame, KeyInterp::Constant);
                    anim_reapply_requested_ = true;
                }
                if (ImGui::MenuItem("Linear", nullptr, m.interp == KeyInterp::Linear)) {
                    applyKeyInterpBothSides(track, track_type, graph_sel_channel, graph_sel_frame, KeyInterp::Linear);
                    anim_reapply_requested_ = true;
                }
                if (ImGui::MenuItem("Bezier", nullptr, m.interp == KeyInterp::Bezier)) {
                    applyKeyInterpBothSides(track, track_type, graph_sel_channel, graph_sel_frame, KeyInterp::Bezier);
                    anim_reapply_requested_ = true;
                }
                ImGui::Separator();
                if (ImGui::MenuItem("Delete Key")) {
                    setChannelKey(*kf, graph_sel_channel, false);
                    if (track_type == GraphTrackType::Light) {
                        kf->light.refreshCompoundFlags();
                        bool any_keyed = false;
                        for (int c = 0; c < CURVE_LIGHT_CHANNEL_COUNT; ++c)
                            if (kf->light.channelKeyed(c)) { any_keyed = true; break; }
                        if (!any_keyed) kf->has_light = false;
                    } else if (track_type == GraphTrackType::Camera) {
                        kf->camera.refreshCompoundFlags();
                        bool any_keyed = false;
                        for (int c = 0; c < CURVE_CAM_CHANNEL_COUNT; ++c)
                            if (kf->camera.channelKeyed(c)) { any_keyed = true; break; }
                        if (!any_keyed) kf->has_camera = false;
                    } else if (track_type == GraphTrackType::Material) {
                        kf->material.refreshCompoundFlags();
                        bool any_keyed = false;
                        for (int c = 0; c < CURVE_MAT_CHANNEL_COUNT; ++c)
                            if (kf->material.channelKeyed(c)) { any_keyed = true; break; }
                        if (!any_keyed) kf->has_material = false;
                    } else {
                        kf->transform.refreshCompoundFlags();
                        bool any_keyed = false;
                        for (int c = 0; c < CURVE_CHANNEL_COUNT; ++c)
                            if (kf->transform.channelKeyed(c)) { any_keyed = true; break; }
                        if (!any_keyed) kf->has_transform = false;
                    }
                    graph_sel_channel = -1;
                    graph_sel_frame = -1;
                    tracks_dirty = true;
                    anim_reapply_requested_ = true;
                }
            }
            ImGui::EndPopup();
        }
    }

    // --- Current frame indicator (vertical red line, same as dope sheet) ---
    {
        int px = frameToPixelX(current_frame, canvas_width);
        float x = canvas_pos.x + px;
        if (x >= canvas_pos.x && x <= canvas_end.x) {
            dl->AddLine(ImVec2(x, canvas_pos.y), ImVec2(x, canvas_end.y), COLOR_CURRENT_FRAME, 2.0f);
            // Triangle indicator at top
            dl->AddTriangleFilled(
                ImVec2(x - 5, canvas_pos.y),
                ImVec2(x + 5, canvas_pos.y),
                ImVec2(x, canvas_pos.y + 8),
                COLOR_CURRENT_FRAME);
        }
    }

    // --- Scrubbing ---
    handleScrubbing(canvas_pos, canvas_width);

    // --- Tooltip showing channel value at current frame ---
    if (ImGui::IsWindowHovered() && !selected_track.empty() && track_it != ctx.scene.timeline.tracks.end()) {
        ImVec2 mouse = ImGui::GetIO().MousePos;
        bool in_canvas = mouse.x >= canvas_pos.x && mouse.x <= canvas_end.x &&
                         mouse.y >= canvas_pos.y && mouse.y <= canvas_end.y;
        if (in_canvas) {
            int hover_frame = pixelXToFrame(mouse.x - canvas_pos.x, canvas_width);
            float hover_val = pixelYToValue(mouse.y - canvas_pos.y, canvas_height);
            ImGui::BeginTooltip();
            ImGui::Text("Frame: %d  Value: %.2f", hover_frame, hover_val);
            ImGui::EndTooltip();
        }
    }
}
