/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          Api/RtApi.h
* Author:        Kemal Demirtas
* Date:          July 2026
* License:       MIT
* =========================================================================
*
* RtApi — the single stable facade every scripting/automation entry point
* goes through (embedded Python "rt" module, ImGui console, headless CLI,
* future IPC/JSON listener). See docs/API_SCRIPTING_ROADMAP.md.
*
* Rules (must hold for every function added here):
*  - Objects are addressed by nodeName string, never by pointer.
*  - Scene mutations run on the main thread only; cross-thread callers use
*    enqueue() and the main loop drains once per frame.
*  - Mutations go through SceneCommand + SceneHistory so they are undoable.
*  - No exceptions across this boundary: mutations return Result.
*  - Keep this header light: forward declarations only, no heavy includes.
*/
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>
#include "Matrix4x4.h"
#include "Vec3.h"

struct UIContext;
class SceneHistory;

namespace rtapi {

// ---------------------------------------------------------------------------
// Versioning — semantic; bump minor on additions, major on breaking changes.
// ---------------------------------------------------------------------------
struct Version {
    int major = 0;
    int minor = 5;
    int patch = 0;
};
Version version();

// ---------------------------------------------------------------------------
// Result — the error model of the whole API surface (no exceptions).
// ---------------------------------------------------------------------------
struct Result {
    bool ok = false;
    std::string error;

    static Result success() { return { true, {} }; }
    static Result fail(std::string msg) { return { false, std::move(msg) }; }
    explicit operator bool() const { return ok; }
};

// ---------------------------------------------------------------------------
// Lifecycle — called once from Main.cpp after UIContext / SceneUI exist.
// ---------------------------------------------------------------------------
void bind(UIContext* ctx, SceneHistory* history);
void unbind();
bool isBound();

// ---------------------------------------------------------------------------
// Main-thread dispatch.
// enqueue() may be called from any thread; the callback runs inside the next
// drainMainThreadQueue(), which Main.cpp calls once per frame (skipped while
// a scene load is in progress).
// ---------------------------------------------------------------------------
void enqueue(std::function<void(UIContext&)> fn);
void drainMainThreadQueue();

// ---------------------------------------------------------------------------
// Events (Faz 3b). Callbacks fire on the main thread from drainMainThreadQueue,
// which runs once per frame with scene loads gated out. on_frame_change fires
// whenever the timeline frame differs from the previous drain; on_scene_load
// fires once after a project/scene finishes loading. Each add* returns an id
// used to unsubscribe. Exceptions thrown by a callback are caught by the
// runtime layer and never cross this boundary.
// ---------------------------------------------------------------------------
int addFrameChangeCallback(std::function<void(int)> fn);   // arg: new frame
int addSceneLoadCallback(std::function<void()> fn);
void removeFrameChangeCallback(int id);
void removeSceneLoadCallback(int id);
void clearEventCallbacks();

// Internal notification hooks. notifySceneLoaded() is called by the facade's
// own openProject and by Main.cpp once a UI-driven load completes.
void notifySceneLoaded();

// ---------------------------------------------------------------------------
// Scene queries (main thread only). The API surface is flat-only: every scene
// node is a flat SoA TriangleMesh (skinned, terrain, water, procedural,
// imported alike) — there is no Triangle-facade path in this API.
// ---------------------------------------------------------------------------
struct ObjectInfo {
    std::string name;
    size_t triangle_count = 0;
    size_t vertex_count = 0;
};

std::vector<std::string> listObjects();
bool objectExists(const std::string& name);
Result getObjectInfo(const std::string& name, ObjectInfo& out);

// ---------------------------------------------------------------------------
// Transform (undoable via TransformCommand, which is flat-aware).
// Flat-only: reads/writes the TriangleMesh's own transform handle.
// ---------------------------------------------------------------------------
Result getObjectTransform(const std::string& name, Matrix4x4& out);
Result setObjectTransform(const std::string& name, const Matrix4x4& matrix);

// ---------------------------------------------------------------------------
// Object lifecycle (undoable). Delete hides + marks pending-delete (physical
// purge happens at save time, same as the UI path); undo restores fully.
// duplicateObject reuses the UI's whole duplicate flow (deep SoA copy +
// incremental backend clone); the clone becomes the current selection and its
// generated name is returned. importModel runs the ProjectManager import
// synchronously on the main thread (blocks until loaded).
// ---------------------------------------------------------------------------
Result deleteObject(const std::string& name);
Result duplicateObject(const std::string& name, std::string& out_new_name);
Result importModel(const std::string& filepath);
Result addPrimitive(const std::string& type, const std::string& name, float size, std::string& out_new_name);

// ---------------------------------------------------------------------------
// Material parameters (undoable). An object may use more than one material;
// setters update every distinct Principled BSDF material assigned to that
// flat mesh. Shared materials retain their normal shared-material semantics.
// Supported color parameters: base_color, emission.
// Supported scalar parameters: roughness, metallic, specular,
// emission_strength, transmission, ior, opacity.
// ---------------------------------------------------------------------------
struct MaterialParamValue {
    bool is_color = false;
    float scalar = 0.0f;
    Vec3 color;
};

Result getMaterialParam(const std::string& object_name, const std::string& param,
                        MaterialParamValue& out);
Result setMaterialParam(const std::string& object_name, const std::string& param,
                        float value);
Result setMaterialParam(const std::string& object_name, const std::string& param,
                        const Vec3& value);

// ---------------------------------------------------------------------------
// Material assets (Faz 5.5a, implemented in RtApiMaterial.cpp). The functions
// above edit parameters through an OBJECT; these manage the assets themselves,
// which is what lets a script build a look from scratch. Materials are addressed
// by name — MaterialManager keeps Material::materialName and its registry key
// identical, so the name is a stable handle for the asset's lifetime.
//
// createMaterial may adjust the requested name to keep it unique; the name that
// was actually registered comes back in out_name. assignMaterial replaces the
// object's WHOLE material assignment (every slot), which is the useful primitive
// for scripted look-building. Not undoable — bulk authoring, like mesh writes.
//
// Texture slots: base_color | roughness | metallic | normal | emission |
// opacity | specular | transmission | height. Principled BSDF only.
// ---------------------------------------------------------------------------
struct MaterialInfo {
    uint16_t id = 0;
    std::string name;
    std::string type;   // "principled" | "volumetric" | "other"
};

std::vector<MaterialInfo> listMaterials();
Result getMaterial(const std::string& name, MaterialInfo& out);
Result createMaterial(const std::string& type, const std::string& requested_name,
                      std::string& out_name);
std::vector<std::string> objectMaterials(const std::string& object_name);
Result assignMaterial(const std::string& object_name, const std::string& material_name);
Result setMaterialTexture(const std::string& material_name, const std::string& slot,
                          const std::string& filepath);
Result clearMaterialTexture(const std::string& material_name, const std::string& slot);
std::vector<std::string> materialTextureSlots(const std::string& material_name);

// ---------------------------------------------------------------------------
// Selection (Faz 5.5a). Most editor operations are selection-driven, so this is
// what lets a script drive those paths instead of duplicating them. NOT undoable
// — selection is treated like viewport navigation, the same exception the camera
// API takes. `index` is the object's index in scene.world.objects, or the light
// index; `primary` marks the item the gizmo follows.
// ---------------------------------------------------------------------------
struct SelectionItem {
    std::string type;   // "object"|"light"|"camera"|"vdb_volume"|"force_field"|...
    std::string name;
    int index = -1;
    bool primary = false;
};

std::vector<SelectionItem> listSelection();
Result selectObject(const std::string& name, bool additive = false);
Result deselectObject(const std::string& name);
Result selectLight(int index, bool additive = false);
Result selectAllObjects(int& out_count);
Result clearSelection();

// ---------------------------------------------------------------------------
// Lights (undoable via Add/Delete/TransformLightCommand). Lights are indexed
// into scene.lights; index is stable until a light is added/removed.
// type strings: "point" | "directional" | "spot" | "area".
// addLight uses the same defaults as the UI menu; for "directional" the
// position parameter is stored but the direction keeps its default until a
// dedicated parameter API lands (Faz 1c).
// ---------------------------------------------------------------------------
struct LightInfo {
    int index = -1;
    std::string name;
    std::string type;
    Vec3 position;
    Vec3 direction;                 // directional / spot only
    Vec3 color = Vec3(1.0f);
    float intensity = 1.0f;
    float radius = 0.0f;            // soft-shadow / source radius
    float spot_angle = 0.0f;        // spot only, degrees
    float spot_falloff = 0.0f;      // spot only
    float width = 0.0f;             // area only
    float height = 0.0f;            // area only
    bool visible = true;
};

std::vector<LightInfo> listLights();
Result getLight(int index, LightInfo& out);
Result addLight(const std::string& type, const Vec3& position, std::string& out_name);
Result deleteLight(int index);
Result setLightPosition(int index, const Vec3& position);

// Geometric edits below reuse LightState + TransformLightCommand (the viewport
// gizmo's own path); appearance edits use a sibling command. Both are undoable.
// setLightParam accepts: radius | spot_angle | spot_falloff | width | height |
// intensity (the last one forwards to setLightIntensity).
Result setLightDirection(int index, const Vec3& direction);
Result setLightParam(int index, const std::string& param, float value);
Result setLightColor(int index, const Vec3& color);
Result setLightIntensity(int index, float intensity);
Result setLightVisible(int index, bool visible);
Result renameLight(int index, const std::string& name);

// ---------------------------------------------------------------------------
// Mesh data (Faz 3a). Positions/normals are exposed in the mesh's local/bind
// space (P_orig/N_orig), matching the flat-node transform invariant that the
// rest of the API already relies on (P = transform->getFinal() * P_orig).
// A write re-bakes world P/N from the edited local buffers before returning,
// so a later gizmo/transform edit composes with the script instead of
// silently overwriting it. UVs have no local/world split and are written
// in place. Buffers returned by the getters point directly at engine memory
// (zero-copy) and are only valid for the duration of the current script call;
// any further scene mutation may relocate them. Not undoable — bulk vertex
// edits are treated like sculpt strokes, not single commands.
// ---------------------------------------------------------------------------
struct MeshBufferView {
    float* data = nullptr;   // interleaved, `components` floats per vertex
    size_t vertex_count = 0;
    int components = 0;      // 3 for positions/normals, 2 for uvs
};

Result getMeshPositions(const std::string& name, MeshBufferView& out);
Result getMeshNormals(const std::string& name, MeshBufferView& out);
Result getMeshUVs(const std::string& name, MeshBufferView& out);

// `vertex_count` must match the object's current vertex count. Triggers the
// same CPU BVH / Vulkan / OptiX rebuild path as other scene-geometry edits.
Result setMeshPositions(const std::string& name, const float* data, size_t vertex_count);
Result setMeshNormals(const std::string& name, const float* data, size_t vertex_count);
Result setMeshUVs(const std::string& name, const float* data, size_t vertex_count);

// Re-derives N_orig from current P_orig topology (area-weighted, weld-aware),
// then re-bakes world N. Useful after setMeshPositions() when the caller does
// not want to compute normals itself.
Result recomputeMeshNormals(const std::string& name);

// ---------------------------------------------------------------------------
// Camera (Faz 5.1a). Operates on the scene's active camera. position=lookfrom,
// target=lookat, fov in vertical degrees, aperture drives depth-of-field.
// Not undoable (camera is treated like viewport navigation); each setter marks
// the camera dirty, resets accumulation, and requests a render.
// ---------------------------------------------------------------------------
struct CameraState {
    Vec3 position;
    Vec3 target;
    Vec3 up;
    float fov = 45.0f;
    float focus_distance = 10.0f;
    float aperture = 0.0f;
};

Result getCamera(CameraState& out);
Result setCameraPosition(const Vec3& position);
Result setCameraTarget(const Vec3& target);
Result setCameraFov(float fov);
Result setCameraFocusDistance(float focus_distance);
Result setCameraAperture(float aperture);

// ---------------------------------------------------------------------------
// World / environment (Faz 5.1c). Narrow surface: background color plus the
// Nishita sky's sun. Elevation/azimuth setters recompute the sun direction.
// Clouds/fog/weather are intentionally out of scope for now. Not undoable;
// each setter marks the world dirty, resets accumulation, and requests a render.
// ---------------------------------------------------------------------------
struct WorldState {
    std::string mode;                  // "solid" | "hdri" | "nishita"
    Vec3 background_color;             // only visible in "solid" mode
    float sun_elevation = 15.0f;       // degrees
    float sun_azimuth = 0.0f;          // degrees
    float sun_intensity = 1.0f;
    float atmosphere_intensity = 10.0f;
    float sun_size = 0.545f;           // degrees
};

Result getWorld(WorldState& out);
// Sky model: "solid" (background_color), "hdri" (environment map), "nishita"
// (procedural Raytrophi spectral sky). background_color only shows in "solid".
Result setWorldMode(const std::string& mode);
Result setWorldBackgroundColor(const Vec3& color);
Result setWorldSunElevation(float degrees);
Result setWorldSunAzimuth(float degrees);
Result setWorldSunIntensity(float intensity);
Result setWorldAtmosphereIntensity(float intensity);
Result setWorldSunSize(float degrees);

// ---------------------------------------------------------------------------
// Post-processing (Faz 5.1d). Exposure, tonemapping, color adjustment,
// vignette, and stylize settings. CRITICAL RULE: Post-processing changes
// MUST NEVER call resetAccumulation (accumulation is preserved).
// ---------------------------------------------------------------------------
struct PostState {
    float exposure = 1.0f;
    float gamma = 2.2f;
    float saturation = 1.0f;
    float color_temperature = 6500.0f;
    std::string tone_mapping = "agx";  // "agx" | "aces" | "uncharted" | "filmic" | "none"
    bool vignette_enabled = true;
    float vignette_strength = 0.0f;
    bool stylize_enabled = false;
    float stylize_strength = 0.75f;
};

Result getPost(PostState& out);
Result setPostExposure(float exposure);
Result setPostGamma(float gamma);
Result setPostSaturation(float saturation);
Result setPostColorTemperature(float temp_k);
Result setPostToneMapping(const std::string& type);
Result setPostVignetteEnabled(bool enabled);
Result setPostVignetteStrength(float strength);
Result setPostStylizeEnabled(bool enabled);
Result setPostStylizeStrength(float strength);

// ---------------------------------------------------------------------------
// Undo / redo (thin wrappers over SceneHistory).
// ---------------------------------------------------------------------------
Result undo();
Result redo();
std::string undoDescription();
std::string redoDescription();

// ---------------------------------------------------------------------------
// Render control.
// ---------------------------------------------------------------------------
Result requestRender();       // arms the start_render trigger flag
Result resetAccumulation();   // active backend if any, else CPU accumulation

// Targeted final-render job. The call is asynchronous because the canonical
// progressive renderer advances in Main.cpp's frame loop. Poll renderStatus()
// from Python/UI, or use the CLI which waits and exits automatically.
enum class RenderJobState {
    Idle,
    Rendering,
    Completed,
    Failed,
    Cancelled
};

struct RenderJobInfo {
    RenderJobState state = RenderJobState::Idle;
    std::string output_path;
    std::string error;
    int current_samples = 0;
    int target_samples = 0;
    float progress = 0.0f;
};

Result renderFrame(const std::string& output_path, int spp);
RenderJobInfo renderStatus();
Result cancelRender();

// Multi-frame sequence render (CLI --frames / Python rt.render.start_sequence).
// Frames [start_frame, end_frame] inclusive are rendered with spp samples each
// and saved to output_dir/frame_NNNN.png using the same g_seq_save_active state
// machine as the UI viewport-driven sequence export. The call is asynchronous;
// the main loop drives frame accumulation and file writes, then sets quit on CLI.
struct SequenceJobInfo {
    bool active = false;
    int current_frame = 0;
    int start_frame = 0;
    int end_frame = 0;
    float frame_progress = 0.0f;  // 0..1 progress of the current frame
    float total_progress = 0.0f;  // 0..1 overall sequence progress
    std::string output_dir;
    std::string error;
};
Result renderSequence(const std::string& output_dir, int spp,
                      int start_frame, int end_frame);
SequenceJobInfo sequenceStatus();
Result cancelSequence();

// Main-loop bridge: output is saved only after the normal render + denoise +
// tonemap/stylize display pipeline has produced the final SDL surface.
bool renderOutputPending();
std::string renderOutputPath();
void completeRenderOutput(bool ok, const std::string& error = {});

// ---------------------------------------------------------------------------
// Project and timeline. Project loading is synchronous and clears selection +
// undo history because commands from the previous scene must never survive.
// Passing an empty path to saveProject() saves to the current project path.
// setFrame() schedules normal TimelineWidget evaluation in the same UI frame.
// ---------------------------------------------------------------------------
std::string currentProjectPath();
Result saveProject(const std::string& filepath = {});
Result openProject(const std::string& filepath);
int currentFrame();
Result setFrame(int frame);

// ---------------------------------------------------------------------------
// Keyframes (Faz 3c). Inserts a transform key on the object's timeline track at
// `frame`. channel is "location" | "rotation" | "scale"; value is that channel's
// full Vec3 (rotation in Euler degrees). An existing key at the same frame keeps
// its other channels — only the named channel is (re)written. Not undoable;
// treated like other bulk-authoring API writes. Schedules a re-render.
// ---------------------------------------------------------------------------
Result insertKeyframe(const std::string& object_name, const std::string& channel,
                      int frame, const Vec3& value);
Result removeKeyframe(const std::string& object_name, int frame);
std::vector<int> listKeyframes(const std::string& object_name);

// ---------------------------------------------------------------------------
// Skeletal animation playback (Faz 5.6c, implemented in RtApiAnim.cpp).
//
// ⚠️ SCOPE — this is deliberately the PLAYBACK + PARAMETER half only. Animation
// node-graph TOPOLOGY (add/link/remove nodes, state machines, blend spaces) is
// NOT exposed, because the layer underneath it is still moving:
//   * three playback paths coexist (node graph, Ozz runtime, AnimationController)
//     selected by useAnimGraph / preferOzzRuntime, and preferOzzRuntime is true;
//   * the Ozz runtime is a declared future migration and today a stub
//     (buildStubAnimationSet / IntegrationState::StubReady), so the pose and
//     skeleton model can still change underneath a frozen API;
//   * AnimationNodeGraph is NOT a NodeSystem::GraphBase — it owns its own node
//     and link vectors, so rt.nodes cannot address it without either an
//     invasive refactor or a second node-scripting dialect.
// What IS exposed here keeps its meaning whichever runtime wins: characters,
// clips, transport (play/stop/pause/time/speed/loop) and graph parameters.
// Revisit topology once the runtime question is settled and AnimationNodeGraph
// shares the common graph base.
//
// Characters are addressed by import name (ImportedModelContext::importName).
// `layer` indexes the controller's animation layers (0..3); layer 0 is the base
// layer. Not undoable — playback is transport state, like the timeline frame.
// ---------------------------------------------------------------------------
struct AnimCharacterInfo {
    std::string name;                 // import name, the handle for every call
    bool has_animation = false;
    int clip_count = 0;
    int bone_count = 0;               // weighted bones
    bool uses_graph = false;          // node graph instead of the clip controller
    std::string graph_asset_key;
    bool graph_follows_timeline = false;
    bool root_motion = false;
    std::string root_motion_bone;     // empty = auto detect
    bool visible = true;
};

struct AnimClipInfo {
    std::string name;
    float duration_seconds = 0.0f;
    float ticks_per_second = 24.0f;
    bool loop = true;
    int start_frame = 0;
    int end_frame = 0;
};

struct AnimPlaybackInfo {
    std::string clip;
    bool playing = false;
    bool paused = false;
    bool blending = false;
    float time = 0.0f;                // seconds into the clip
    float normalized_time = 0.0f;     // 0..1
    int layer = 0;
};

// Characters this module can actually DRIVE — imports that own an AnimationController.
// A static mesh import (cube, plane) also has an ImportedModelContext but no controller, and
// listing those made the obvious loop over characters() fail on scenes with no animation.
// getAnimCharacter() below is deliberately unfiltered: a name asked for by hand still answers.
std::vector<AnimCharacterInfo> listAnimCharacters();
Result getAnimCharacter(const std::string& character, AnimCharacterInfo& out);
Result listAnimClips(const std::string& character, std::vector<AnimClipInfo>& out);

Result playAnimClip(const std::string& character, const std::string& clip,
                    float blend_seconds = 0.3f, int layer = 0);
Result stopAnimation(const std::string& character, float blend_out_seconds = 0.3f,
                     int layer = 0);
Result setAnimPaused(const std::string& character, bool paused);
Result setAnimTime(const std::string& character, float seconds, int layer = 0);
Result setAnimSpeed(const std::string& character, float speed, int layer = 0);
Result setAnimLoop(const std::string& character, bool loop, int layer = 0);
Result getAnimPlayback(const std::string& character, int layer, AnimPlaybackInfo& out);

// Graph parameters drive the character's animation node graph (its Parameter
// nodes and state-machine transition conditions). They are only meaningful for
// a character whose uses_graph is true; setting them on a clip-driven character
// is reported rather than silently ignored.
Result setAnimGraphFloat(const std::string& character, const std::string& name, float value);
Result setAnimGraphBool(const std::string& character, const std::string& name, bool value);
Result triggerAnimGraphParam(const std::string& character, const std::string& name);
Result getAnimGraphPlayback(const std::string& character, AnimPlaybackInfo& out);

// ---------------------------------------------------------------------------
// Node graphs (Faz 3d). Builds material / geometry node graphs through the
// shared NodeRegistry (typeId -> factory). graph_type is "material" (addressed
// by material name) or "geometry" (addressed by object nodeName); the named
// graph must already exist. This is graph-construction only: nodes and links
// are created and the graph is marked dirty, but evaluation/apply still runs
// through the normal editor path (open the node editor's Live toggle, or a
// future rt.nodes.evaluate). Pins are addressed by node id + slot index.
// ---------------------------------------------------------------------------
struct NodeTypeDesc {
    std::string type_id;
    std::string category;
    std::string display_name;
    std::string description;
};

struct NodeDesc {
    unsigned int id = 0;
    std::string type_id;
    std::string display_name;
    int input_count = 0;
    int output_count = 0;
};

// Graph lifecycle (Faz 5.5b). A material graph is keyed by material name and is
// seeded from that material the way the node editor seeds it; a geometry graph
// is keyed by object nodeName and starts empty. Terrain graphs are owned by the
// TerrainObject and are created by applyTerrainPreset instead.
std::vector<std::string> listNodeGraphs(const std::string& graph_type);
Result createNodeGraph(const std::string& graph_type, const std::string& graph_name);
Result removeNodeGraph(const std::string& graph_type, const std::string& graph_name);

// Applies a graph to what it drives — the step that closes the scripted
// authoring loop (Faz 5.5c). This is NOT terrain's async evaluate: for a
// material it is the editor's own Apply, i.e. fold the constant chains into the
// material, publish the volume branch, compile the spatially-varying chains
// into the per-pixel program, force the one-time geometry pass a Pointiness /
// Attribute read needs, then re-upload material + program and reset
// accumulation. Pushing the material to a device cannot substitute for the
// compile. Geometry graphs run the Geo-DAG apply (swaps the object's mesh);
// terrain graphs keep their async contract and stay with evaluateTerrain.
//
// `warnings` mirrors the editor's diagnostics panel — a successful apply can
// still report that slots shade per-pixel, or that an Attribute name found no
// free slot and compiled to 0.
struct NodeGraphApplyInfo {
    bool ok = false;
    std::vector<std::string> warnings;
    std::vector<std::string> errors;
};

Result applyNodeGraph(const std::string& graph_type, const std::string& graph_name,
                      NodeGraphApplyInfo& out);

std::vector<NodeTypeDesc> listNodeTypes();
Result addNode(const std::string& graph_type, const std::string& graph_name,
               const std::string& type_id, unsigned int& out_node_id);
Result removeNode(const std::string& graph_type, const std::string& graph_name,
                  unsigned int node_id);
Result linkNodes(const std::string& graph_type, const std::string& graph_name,
                 unsigned int from_node, int from_output, unsigned int to_node,
                 int to_input, unsigned int& out_link_id);
Result listNodes(const std::string& graph_type, const std::string& graph_name,
                 std::vector<NodeDesc>& out);

// ---------------------------------------------------------------------------
// Node parameters (Faz 5.1b). A node's scriptable parameters are its input
// pins' default values — the value used when the pin is unconnected. Pins are
// addressed by input-slot index (0-based), matching listNodeParams() order.
// setNodeParam coerces the provided components to the pin's declared data type;
// a linked input keeps its connection and ignores its default during evaluation
// until unlinked. Setting a parameter marks the node (and everything downstream)
// dirty. Not undoable — treated like other bulk graph authoring. Evaluation
// still runs through the normal editor path (same scope note as Faz 3d).
// ---------------------------------------------------------------------------
struct NodeParamValue {
    enum class Kind { None, Float, Int, Bool, Vector2, Vector3, Vector4, String };
    Kind kind = Kind::None;
    float floats[4] = { 0.0f, 0.0f, 0.0f, 0.0f };  // scalar in [0]; vectors fill in order
    int int_value = 0;
    bool bool_value = false;
    std::string string_value;
};

struct NodeParamInfo {
    int index = 0;
    std::string name;
    std::string data_type;   // "float"|"int"|"bool"|"vector2/3/4"|"color"|"string"|...
    bool connected = false;  // a linked input ignores its default during evaluation
    NodeParamValue value;
};

Result listNodeParams(const std::string& graph_type, const std::string& graph_name,
                      unsigned int node_id, std::vector<NodeParamInfo>& out);
Result getNodeParam(const std::string& graph_type, const std::string& graph_name,
                    unsigned int node_id, int pin_index, NodeParamValue& out);
Result setNodeParam(const std::string& graph_type, const std::string& graph_name,
                    unsigned int node_id, int pin_index, const NodeParamValue& value);

struct NodePropertyInfo {
    std::string name;        // dotted serialized path, e.g. "amplitude" or "params.iterations"
    std::string data_type;   // float|int|bool|string
    NodeParamValue value;
};

Result listNodeProperties(const std::string& graph_type, const std::string& graph_name,
                          unsigned int node_id, std::vector<NodePropertyInfo>& out);
Result getNodeProperty(const std::string& graph_type, const std::string& graph_name,
                       unsigned int node_id, const std::string& property, NodeParamValue& out);
Result setNodeProperty(const std::string& graph_type, const std::string& graph_name,
                       unsigned int node_id, const std::string& property,
                       const NodeParamValue& value);

// ---------------------------------------------------------------------------
// Mesh Modifiers (Faz 5.2b). Allows adding, querying, updating, removing,
// and applying mesh modifiers (e.g., Catmull-Clark / Simple Subdivision).
// ---------------------------------------------------------------------------
struct ModifierInfo {
    int index = 0;
    std::string name;
    std::string type;       // "catmull_clark", "simple", "smooth"
    bool enabled = true;
    int levels = 1;         // Viewport level
    int render_levels = 2;  // Render level
    float smooth_angle = 0.5f;
};

Result getModifierStack(const std::string& object_name, std::vector<ModifierInfo>& out_stack);
Result addModifier(const std::string& object_name, const std::string& type, const std::string& name,
                   int levels, int render_levels, ModifierInfo& out_mod);
Result removeModifier(const std::string& object_name, int index);
Result updateModifier(const std::string& object_name, int index,
                       const std::string* new_name, const bool* enabled,
                       const int* levels, const int* render_levels,
                       const float* smooth_angle);
Result applyModifier(const std::string& object_name, int index = 0);

// ---------------------------------------------------------------------------
// Scatter & Foliage System (Faz 5.2c). Allows managing scatter layers/groups,
// adding/removing source meshes, configuring density/slope/height rules,
// procedural surface filling, and manual instance positioning.
// ---------------------------------------------------------------------------
struct ScatterSourceInfo {
    std::string name;
    float weight = 1.0f;
    float scale_min = 0.8f;
    float scale_max = 1.2f;
    float rotation_random_y = 360.0f;
    bool align_to_normal = true;
};

struct ScatterGroupInfo {
    int id = -1;
    std::string name;
    std::string target_type; // "mesh" or "terrain"
    std::string target_node_name;
    size_t instance_count = 0;
    size_t triangle_count = 0;
    std::vector<ScatterSourceInfo> sources;
};

Result listScatterGroups(std::vector<ScatterGroupInfo>& out_groups);
Result createScatterGroup(const std::string& name, const std::string& target_node_name,
                           const std::string& target_type, ScatterGroupInfo& out_info);
Result deleteScatterGroup(const std::string& group_id_or_name);
Result clearScatterGroup(const std::string& group_id_or_name);
Result addScatterSource(const std::string& group_id_or_name, const std::string& mesh_name,
                        float weight = 1.0f, float scale_min = 0.8f, float scale_max = 1.2f,
                        float rotation_y = 360.0f, bool align_to_normal = true);
Result removeScatterSource(const std::string& group_id_or_name, int source_index = 0);
Result setScatterGroupSettings(const std::string& group_id_or_name,
                                const int* target_count, const int* seed,
                                const float* min_distance, const float* slope_max,
                                const float* height_min, const float* height_max,
                                const std::string* density_mask, const std::string* scale_mask);
Result fillScatterGroup(const std::string& group_id_or_name, int& out_spawned);
Result addScatterInstance(const std::string& group_id_or_name, Vec3 pos, Vec3 rot, Vec3 scale, int source_index = 0);

// ---------------------------------------------------------------------------
// Physics Engine (Faz 5.3a). Rigid Body, Soft Body, and Cloth simulation.
// ---------------------------------------------------------------------------
struct PhysicsBodyInfo {
    std::string object_name;
    std::string kind;        // "rigid", "soft", "cloth"
    std::string motion_type; // "dynamic", "static", "kinematic"
    std::string shape;       // "box", "sphere", "capsule", "mesh"
    bool enabled = true;
    float mass = 1.0f;
    float friction = 0.5f;
    float restitution = 0.2f;
    float linear_damping = 0.05f;
    float angular_damping = 0.05f;
    float gravity_scale = 1.0f;
    float soft_stiffness = 0.8f;
    float soft_pressure = 0.0f;
    float soft_damping = 0.05f;
};

Result getPhysicsBody(const std::string& object_name, PhysicsBodyInfo& out_info);
Result addPhysicsBody(const std::string& object_name, const std::string& kind,
                      const std::string& motion_type, const std::string& shape,
                      float mass, PhysicsBodyInfo& out_info);
Result removePhysicsBody(const std::string& object_name);
Result updatePhysicsBody(const std::string& object_name,
                        const std::string* kind = nullptr, const std::string* motion_type = nullptr,
                        const std::string* shape = nullptr, const bool* enabled = nullptr,
                        const float* mass = nullptr, const float* friction = nullptr, const float* restitution = nullptr,
                        const float* linear_damping = nullptr, const float* angular_damping = nullptr,
                        const float* gravity_scale = nullptr, const float* soft_stiffness = nullptr,
                        const float* soft_pressure = nullptr, const float* soft_damping = nullptr);
Result resetPhysicsSimulation();
Result stepPhysicsSimulation(float dt = 0.0166667f);
Result setPhysicsGravity(Vec3 gravity);
Result getPhysicsGravity(Vec3& out_gravity);

struct FractureGroupInfo {
    std::string group;
    int shard_count = 0;
    int broken_count = 0;
    float base_break_impulse = 0.0f;
    float effective_break_impulse = 0.0f;
    bool integrity_weakening = true;
    float integrity_exponent = 1.5f;
    float minimum_threshold_scale = 0.15f;
    float mean_integrity = 1.0f;
    float minimum_integrity = 1.0f;
    float remaining_support_ratio = 1.0f;
};

Result makePhysicsFractureGroup(const std::string& group,
                                const std::vector<std::string>& shard_objects,
                                float break_impulse,
                                bool integrity_weakening,
                                float integrity_exponent,
                                float minimum_threshold_scale,
                                FractureGroupInfo& out_info);
Result getPhysicsFractureGroup(const std::string& group,
                               FractureGroupInfo& out_info);
Result breakPhysicsFractureGroup(const std::string& group, float strength);
Result applyPhysicsFractureImpulse(const std::string& group, Vec3 point,
                                   Vec3 direction, float impulse,
                                   bool& out_triggered);

struct StructuralImpulseInfo {
    uint64_t queued = 0;
    uint64_t consumed = 0;
    uint64_t affected_groups = 0;
    uint64_t fractured_groups = 0;
    float last_peak_pressure_kpa = 0.0f;
    float last_max_impulse = 0.0f;
};
Result emitGasPressurePulse(const std::string& domain, Vec3 center,
                            float radius, float peak_pressure_kpa,
                            float duration_seconds, float coupling,
                            uint64_t& out_sequence);
Result getStructuralImpulseInfo(StructuralImpulseInfo& out_info);

struct AshDebrisInfo {
    bool enabled = true;
    uint64_t max_particles = 4096;
    float particles_per_kg = 120.0f;
    float near_distance = 12.0f;
    float far_lod_scale = 0.25f;
    float lifetime_seconds = 5.0f;
    uint64_t alive_particles = 0;
    uint64_t events = 0;
    uint64_t requested_particles = 0;
    uint64_t spawned_particles = 0;
    uint64_t lod_reduced_particles = 0;
    uint64_t budget_rejected_particles = 0;
    float accepted_mass_kg = 0.0f;
};
Result configureAshDebris(bool enabled, uint64_t max_particles,
                          float particles_per_kg, float near_distance,
                          float far_lod_scale, float lifetime_seconds);
Result emitAshDebris(Vec3 center, Vec3 velocity, float mass_kg,
                     float camera_distance, uint32_t seed,
                     uint64_t& out_spawned);
Result getAshDebrisInfo(AshDebrisInfo& out_info);

// ---------------------------------------------------------------------------
// Force fields (Faz 5.6a, implemented in RtApiForceField.cpp). One field drives
// every simulation family at once — gas, particles, cloth, rigid bodies and the
// APIC liquid — through Physics::ForceFieldManager, so this is the highest-
// leverage sim surface a script can reach.
//
// Fields are addressed by ID or by name (`id_or_name`); the manager assigns the
// ID and keeps it stable for the field's lifetime, while names are only uniqued
// on create, so prefer the ID when a script holds a handle across edits.
//
// The write surface is deliberately ONE function: read with getForceField(),
// modify the struct, write it back with updateForceField(). A field carries ~35
// parameters and per-parameter setters would have meant ~35 entry points on
// every layer (facade + binding + IPC dispatch + capability). The bindings turn
// this into a kwargs patch, so from a script it still reads as a partial edit.
//
// NOT undoable — matches the force-field panel, which mutates the manager
// directly. Every mutation runs the panel's own post-edit step: invalidate the
// rigid-body cache and reset CPU + backend accumulation. Skipping that leaves a
// stale simulation cache and a stale image, so a scripted edit would appear to
// do nothing until the next unrelated scene change.
//
// evaluateForceFields() is a read-only probe of the combined field at a point —
// the same evaluation the solvers run. It is what lets a script verify a field
// does what it intends without stepping a simulation.
// ---------------------------------------------------------------------------
struct ForceFieldInfo {
    int id = -1;
    std::string name;
    // wind | gravity | attractor | repeller | vortex | turbulence | curlnoise |
    // drag | magnetic | directionalnoise | thermal
    std::string type = "wind";
    std::string shape = "sphere";     // infinite | sphere | box | cylinder | cone
    // none | linear | smooth | sphere | inverse_square | exponential | custom
    std::string falloff = "smooth";
    bool enabled = true;
    bool visible = true;              // viewport gizmo only, not a force switch

    Vec3 position;
    Vec3 rotation;                    // euler degrees
    Vec3 scale = Vec3(1.0f, 1.0f, 1.0f);
    Vec3 direction = Vec3(0.0f, -1.0f, 0.0f);   // wind / gravity
    Vec3 axis = Vec3(0.0f, 1.0f, 0.0f);         // vortex

    float strength = 1.0f;
    float falloff_radius = 5.0f;      // outer radius where the force reaches 0
    float inner_radius = 0.0f;        // inner radius at full strength

    bool use_noise = false;
    int noise_octaves = 4;
    int noise_seed = 42;
    float noise_frequency = 0.5f;
    float noise_lacunarity = 2.0f;
    float noise_persistence = 0.5f;
    float noise_amplitude = 1.0f;
    float noise_speed = 0.1f;

    float inward_force = 0.0f;        // vortex: spiral pull toward the centre
    float upward_force = 0.0f;        // vortex: lift along the axis (tornado)
    float linear_drag = 0.1f;         // drag: F = -drag * v
    float quadratic_drag = 0.0f;      // drag: F = -drag * v^2

    // thermal: Kelvin added on top of the local ambient at the core, attenuated
    // by the falloff. A Thermal field exerts no force at all — it drives Material
    // State Field surface heating (ignition, char, incandescence) only.
    float thermal_delta_kelvin = 600.0f;

    // Wind -> APIC liquid coupling. With the drag model on, `strength` is read
    // as the TARGET surface speed (m/s) rather than an acceleration, and only
    // the horizontal band just below the free surface is pushed.
    bool fluid_surface_drag = true;
    float fluid_drag_coupling = 4.0f;
    float fluid_surface_depth = 0.5f;
    float fluid_curl_detail = 0.0f;

    float start_frame = 0.0f;
    float end_frame = -1.0f;          // -1 = never stops
    float phase = 0.0f;

    bool affects_gas = true;
    bool affects_particles = true;
    bool affects_cloth = true;
    bool affects_rigidbody = true;
    bool affects_fluid = true;
};

std::vector<std::string> forceFieldTypes();
std::vector<ForceFieldInfo> listForceFields();
Result getForceField(const std::string& id_or_name, ForceFieldInfo& out);
// Seeds the per-type defaults the panel's "add field" menu uses (noise on for
// turbulence/curl, a cylinder + spiral for vortex, linear drag for drag), so a
// scripted field behaves like one created from the UI.
Result createForceField(const std::string& type, const std::string& requested_name,
                        ForceFieldInfo& out);
Result removeForceField(const std::string& id_or_name);
Result updateForceField(const std::string& id_or_name, const ForceFieldInfo& info);
Result evaluateForceFields(Vec3 world_position, float time, Vec3 velocity, Vec3& out_force);

// ---------------------------------------------------------------------------
// Particle systems (Faz 5.6b, implemented in RtApiParticle.cpp). Emitters,
// solver settings and live statistics for the discrete particle runtime.
//
// ⚠️ Particle COLLIDERS and grid domains are NOT here: they hang off the same
// ParticleSimulationSystem and are already scripted as `simulation collider` /
// fluid domain (RtApiFluid.cpp). Adding a second spelling of them would mean
// two facades mutating one runtime.
//
// Emitters are addressed by INDEX or by name; the index is the position in the
// runtime's emitter list and shifts when an earlier emitter is removed, exactly
// like the panel's list. Names are not uniqued by the runtime, so a name lookup
// returns the first match.
//
// Same one-write-function shape as force fields: read with getParticleEmitter(),
// modify, write back with updateParticleEmitter(); the bindings expose it as a
// kwargs/JSON patch. Not undoable (the panel edits the runtime directly).
//
// ★`burst_count` is one-shot but must NOT be zeroed to "consume" it — the
// runtime tracks that separately so the burst survives serialization and
// replays on rewind. A script that clears burst_count kills the effect on disk.
// ---------------------------------------------------------------------------
struct ParticleEmitterInfo {
    int index = -1;
    std::string name = "Particle Emitter";
    std::string source_mode = "point";  // point | object_origin | force_field_origin
    std::string spawn_mode = "center";  // center | object_aabb_surface | mesh_surface
    std::string source_name;            // object / force field the emitter binds to
    bool enabled = true;

    Vec3 point = Vec3(0.0f, 1.0f, 0.0f);
    Vec3 local_offset;
    Vec3 direction = Vec3(0.0f, 1.0f, 0.0f);
    float surface_offset = 0.02f;

    float rate_per_second = 32.0f;
    int burst_count = 0;                // one-shot; see the note above
    float speed = 2.0f;
    float spread = 0.35f;
    float lifetime_seconds = 4.0f;
    float mass = 1.0f;

    // Visual attributes evolve linearly from birth to death.
    float start_size = 0.06f;
    float end_size = 0.02f;
    float size_jitter = 0.0f;
    float start_opacity = 1.0f;
    float end_opacity = 0.0f;
    Vec3 start_color = Vec3(1.0f, 0.85f, 0.5f);
    Vec3 end_color = Vec3(1.0f, 0.25f, 0.08f);
    float angular_velocity = 0.0f;
    float angular_jitter = 0.0f;
    unsigned int seed = 1;

    // Object binding. When set, `point`/`local_offset`/`direction` are
    // parent-LOCAL and the emitter rides the object's full transform —
    // including motion produced by rigid-body physics.
    std::string parent_object;
    std::string velocity_space = "local"; // local|world
    float inherit_velocity = 1.0f;

    // Per-emitter particle -> gas deposit. Off by default, in which case the
    // system-wide rates apply. Fuel is what lets a flying particle IGNITE the
    // gas it passes through.
    bool  override_grid_deposit = false;
    float grid_density_deposit = 0.0f;
    float grid_temperature_deposit = 0.0f;
    float grid_fuel_deposit = 0.0f;
};

// Solver settings are per particle SYSTEM, not per emitter.
struct ParticlePhysicsInfo {
    std::string mode = "spark";        // spark | granular | fluid | gas
    std::string quality = "realtime";  // realtime | preview | offline
    float particle_radius = 0.04f;
    bool self_collision_enabled = false;
    int solver_iterations = 1;
    int max_neighbors_per_particle = 32;
    float viscosity = 0.0f;
    float cohesion = 0.0f;
    float pressure_stiffness = 0.0f;
    float rest_density = 1000.0f;
    float buoyancy = 0.0f;
    float gravity_scale = 1.0f;
    float vorticity = 0.0f;
    // Particle -> gas grid deposit, per second and per particle. This is what
    // makes debris CARRY fire and smoke instead of being a decorative overlay;
    // fuel deposit additionally needs the domain's Fuel channel + fire enabled.
    float grid_density_deposit = 0.0f;
    float grid_temperature_deposit = 0.0f;
    float grid_fuel_deposit = 0.0f;
    bool grid_deposit_fade_with_age = true;
};

// Counts are read live from the runtime, so they are correct the instant an
// emitter is added. The timings are per-step measurements and stay zero until
// the simulation has actually stepped.
struct ParticleStatsInfo {
    int alive_count = 0;
    int capacity = 0;
    int emitter_count = 0;
    int collider_count = 0;
    int domain_count = 0;
    float total_ms = 0.0f;
    float emit_ms = 0.0f;
    float integrate_ms = 0.0f;
    float self_collision_ms = 0.0f;
    float grid_domain_ms = 0.0f;
};

std::vector<ParticleEmitterInfo> listParticleEmitters();
Result getParticleEmitter(const std::string& index_or_name, ParticleEmitterInfo& out);
Result addParticleEmitter(const ParticleEmitterInfo& info, ParticleEmitterInfo& out);
Result removeParticleEmitter(const std::string& index_or_name);
Result updateParticleEmitter(const std::string& index_or_name, const ParticleEmitterInfo& info);
Result clearParticleEmitters();

// Particle SYSTEMS, not particles. Every other simulation call is scoped to the
// active system; these two are the only way to see and remove what lives in the
// others (a UI preset always creates its own).
struct ParticleSystemInfo {
    int index = -1;
    uint32_t id = 0;
    std::string name;
    bool active = false;
    int domain_count = 0;
    int flow_source_count = 0;
    int emitter_count = 0;
    int collider_count = 0;
};
Result listParticleSystems(std::vector<ParticleSystemInfo>& out);
Result clearParticleSystems();

Result getParticlePhysics(ParticlePhysicsInfo& out);
Result updateParticlePhysics(const ParticlePhysicsInfo& info);
Result getParticleStats(ParticleStatsInfo& out);

// Direct particle authoring / control. spawnParticle bypasses the emitters and
// injects one particle; stepParticleSimulation advances only the particle
// system (fluid/gas domains have their own step), and clearParticles drops the
// live particles while keeping emitters and settings.
Result spawnParticle(Vec3 position, Vec3 velocity, float lifetime_seconds, float mass,
                     float size, int& out_index);
Result clearParticles();
Result stepParticleSimulation(float dt = 0.0166667f);

// ---------------------------------------------------------------------------
// Fluid Simulation Engine (Faz 5.3b). APIC liquid & grid domain simulation.
// ---------------------------------------------------------------------------
struct FluidDomainInfo {
    uint32_t id = 0;
    std::string name;
    std::string type;        // "fluid" (liquid) or "gas" (smoke/fire)
    Vec3 domain_min;
    Vec3 domain_max;
    float voxel_size = 0.05f;
    size_t particle_count = 0;
    std::string render_mode; // "volume", "surface", "particles"
    std::string backend;     // "cpu", "gpu", "vulkan", "cpu_sparse"
    std::string boundary;    // "closed", "open", "periodic"
    std::string preset;      // "water", "oil", "mud", "honey", "lava", "sand", "custom"
    float viscosity = 0.0f;
    bool enabled = true;
    bool visible = true;
};

struct GasDomainSettings {
    std::string quality_profile = "preview"; // interactive, preview, final, cinema, custom
    uint32_t resource_budget_mb = 1024;
    bool enforce_resource_budget = true;
    bool use_sparse_tiles = true;
    bool render_to_nanovdb = true;
    bool fire_enabled = false;
    float ignition_temperature = 0.3f;
    float burn_rate = 1.5f;
    float heat_release = 2.0f;
    float smoke_generation = 0.6f;
    float flame_dissipation = 3.0f;
    float fire_max_temperature = 10.0f;
    float buoyancy_heat = 1.0f;
    float buoyancy_density = 0.08f;
    float vorticity = 0.35f;
    float fire_expansion = 0.0f;
    float turbulence_strength = 0.0f;
    float turbulence_scale = 1.2f;
    int turbulence_octaves = 3;
    float turbulence_lacunarity = 2.0f;
    float turbulence_persistence = 0.5f;
    float turbulence_speed = 0.5f;
};

struct CombustibleFluidSettings {
    std::string chemistry_preset = "inert";
    bool enabled = false;
    bool auto_ignite = false;
    float ignition_temperature = 0.8f;
    float evaporation_rate = 0.35f;
    float surface_fuel_capacity = 4.0f;
    float heat_release = 2.0f;
    float smoke_yield = 0.45f;
    float surface_cooling = 0.35f;
};

struct SimulationFlowSourceInfo {
    std::string name;
    std::string domain;
    std::string source_mode = "point"; // point|object_bounds|mesh_surface
    std::string source_object;
    bool enabled = true;
    // Object binding. When set, `position`/`velocity` are parent-LOCAL.
    std::string parent_object;
    std::string velocity_space = "local"; // local|world
    float inherit_velocity = 1.0f;
    Vec3 position = Vec3(0.0f, 1.0f, 0.0f);
    Vec3 velocity = Vec3(0.0f, 1.0f, 0.0f);
    float radius = 0.35f;
    float velocity_coupling = 8.0f;
    float density = 1.0f;
    float temperature = 0.0f;
    float fuel = 0.0f;
    float falloff = 1.0f;
    float fluid_particles_per_second = 1000.0f;
    float fluid_velocity_spread = 0.15f;
    bool fluid_emit_along_normal = false;
    bool use_time_limit = false;
    float start_time = 0.0f;
    float end_time = 5.0f;
    bool use_particle_limit = false;
    int max_emitted_particles = 100000;
};

struct SimulationColliderInfo {
    std::string name;
    std::string source_mode = "plane"; // plane|sphere|capsule|aabb|obb|mesh_sdf|convex|mesh_bvh
    std::string source_object;
    bool enabled = true;
    bool fluid_collision_enabled = true;
    float plane_y = 0.0f;
    Vec3 sphere_center = Vec3(0.0f, 1.0f, 0.0f);
    float sphere_radius = 1.0f;
    Vec3 capsule_start = Vec3(0.0f);
    Vec3 capsule_end = Vec3(0.0f, 2.0f, 0.0f);
    float capsule_radius = 0.5f;
    Vec3 bounds_min = Vec3(-1.0f);
    Vec3 bounds_max = Vec3(1.0f);
    float friction = 0.0f;
    float restitution = 0.35f;
    float thickness = 0.0f;
    int sdf_resolution_mode = 1; // 0=32^3, 1=64^3, 2=128^3
    bool sdf_ready = false;      // read-only asynchronous cook status
    int sdf_resolution = 0;      // read-only cooked cubic resolution
    bool gas_interaction_enabled = false;
    float gas_density_rate = 0.0f;
    float gas_temperature_rate = 0.0f;
    float gas_fuel_rate = 0.0f;
    float gas_flame_rate = 0.0f;
    bool gas_ignite_on_contact = false;
    // ── Material State Field (thermo-chemistry) ──────────────────────────────
    // What this object is made of. Drives pyrolysis/ignition/charring from real
    // physical constants; see MaterialStateField.h. This is how "the crate in
    // the room catches fire from the burning floor" is authored.
    std::string msf_substance = "Wood (Oak)";
    bool  msf_override_ignition = false;
    float msf_ignition_kelvin = 573.0f;
    float msf_burn_rate_scale = 1.0f;
    float msf_fuel_capacity_scale = 1.0f;
    int   msf_mask_resolution = 128;
    bool  msf_generate_char_mask = true;
    bool  msf_auto_transfer = false;
    std::string msf_transfer_domain;
    float msf_transfer_rate_kg_s = 0.10f;
    float msf_transfer_min_mass_kg = 0.01f;
    float msf_transfer_particles_per_kg = 2048.0f;
    uint32_t msf_transfer_max_batch_particles = 256u;
    Vec3 msf_transfer_velocity = Vec3(0.0f, -0.1f, 0.0f);
    bool msf_melt_flow_enabled = true;
    float msf_melt_height_loss = 0.85f;
    float msf_melt_spread = 1.50f;
    bool msf_melt_sdf_refresh = false; // legacy; explicit force rebuild is authoritative
    uint32_t msf_melt_sdf_revision_interval = 4u;
    float msf_melt_sdf_change_threshold = 0.025f;
};

// Per-domain volume appearance. Separated from GasDomainSettings because it is
// a LOOK, not a solver setting — and because a fire domain rendered with the
// default smoke preset has no blackbody emission at all, so it simulates fire
// correctly and shows flat grey. Scripts need to be able to fix that.
struct GasShaderSettings {
    std::string preset = "fire";   // fire|smoke  (applied first, then overrides)
    float density_multiplier = 1.0f;
    float density_cutoff = 0.01f;
    float blackbody_intensity = 5.0f;
    float temperature_min = 800.0f;   // Kelvin
    float temperature_max = 1900.0f;  // Kelvin
    float scattering_coefficient = 0.15f;
    float absorption_coefficient = 0.5f;
};

Result getGasShaderSettings(const std::string& domain_id_or_name,
                            GasShaderSettings& out_settings);
Result updateGasShaderSettings(const std::string& domain_id_or_name,
                               const GasShaderSettings& settings);
// Names in the built-in substance library, for scripts and UI pickers.
Result listMaterialSubstances(std::vector<std::string>& out_names);

struct MaterialFieldInfo {
    std::string object_key;
    std::string substance;
    uint64_t topology_generation = 0;
    uint64_t content_generation = 0;
    uint32_t element_count = 0;
    int mask_resolution = 0;
    bool centers_dirty = false;
    float mean_integrity = 1.0f;
    float minimum_integrity = 1.0f;
    float mass_loss = 0.0f;
    float initial_mass = 0.0f;
    float solid_mass = 0.0f;
    float pyrolyzed_mass = 0.0f;
    float molten_reservoir_mass = 0.0f;
    float transferred_mass = 0.0f;
    float mass_conservation_error = 0.0f;
    std::vector<std::string> semantics;
};

// Read-only Phase-0 field-contract view used by scripts and diagnostics.
Result listMaterialFields(std::vector<MaterialFieldInfo>& out_fields);

struct MoltenMassTransferInfo {
    uint64_t queued = 0, completed = 0;
    uint64_t deferred_no_domain = 0, deferred_no_capacity = 0;
    float requested_mass = 0.0f, transferred_mass = 0.0f;
    uint64_t spawned_particles = 0;
    std::string last_object, last_domain, last_substance;
    float last_temperature_kelvin = 0.0f;
    float last_combustible_fraction = 0.0f;
    uint64_t live_tagged_particles = 0;
    float mean_remaining_mass_fraction = 0.0f;
};
Result queueMoltenMassTransfer(const std::string& object_key,
                               const std::string& preferred_domain,
                               float mass_kg, float particles_per_kg,
                               Vec3 velocity, uint64_t& out_sequence);
Result getMoltenMassTransferInfo(MoltenMassTransferInfo& out);

Result createFluidDomain(const std::string& name, Vec3 domain_min, Vec3 domain_max,
                         float voxel_size, const std::string& type, FluidDomainInfo& out_info);
Result removeFluidDomain(const std::string& domain_id_or_name);
Result getFluidDomain(const std::string& domain_id_or_name, FluidDomainInfo& out_info);
// Enumerate every simulation grid domain (liquid AND gas) across all particle
// systems. Without this a script can only address domains whose names it
// authored itself, so it can never clean up what a UI preset left behind.
Result listFluidDomains(std::vector<FluidDomainInfo>& out_domains);
Result seedFluidParticles(const std::string& domain_id_or_name, Vec3 seed_min, Vec3 seed_max,
                           int particles_per_cell = 4, bool replace = true);
Result clearFluidParticles(const std::string& domain_id_or_name);
Result updateFluidDomain(const std::string& domain_id_or_name,
                         const Vec3* domain_min = nullptr, const Vec3* domain_max = nullptr,
                         const float* voxel_size = nullptr, const std::string* render_mode = nullptr,
                         const std::string* backend = nullptr, const std::string* boundary = nullptr,
                         const std::string* preset = nullptr, const float* viscosity = nullptr,
                         const bool* enabled = nullptr, const bool* visible = nullptr);
Result getGasDomainSettings(const std::string& domain_id_or_name, GasDomainSettings& out_settings);
Result updateGasDomainSettings(const std::string& domain_id_or_name, const GasDomainSettings& settings);
Result getCombustibleFluidSettings(const std::string& domain_id_or_name,
                                   CombustibleFluidSettings& out_settings);
Result updateCombustibleFluidSettings(const std::string& domain_id_or_name,
                                      const CombustibleFluidSettings& settings);
// One timeline key on a flow source. Only the channels whose has_* flag is set
// are written, so two calls can key different channels on the same frame — the
// same independent-channel model the panel's diamond buttons author.
struct SimulationFlowSourceKey {
    int frame = 0;
    bool has_enabled = false;           bool  enabled = true;
    bool has_position = false;          Vec3  position;
    bool has_velocity = false;          Vec3  velocity;
    bool has_radius = false;            float radius = 0.35f;
    bool has_density = false;           float density = 1.0f;
    bool has_temperature = false;       float temperature = 0.0f;
    bool has_fuel = false;              float fuel = 0.0f;
    bool has_falloff = false;           float falloff = 1.0f;
    bool has_velocity_coupling = false; float velocity_coupling = 8.0f;
    bool has_flow_rate = false;         float flow_rate = 1000.0f;
};
Result keySimulationFlowSource(const std::string& name, const SimulationFlowSourceKey& key);
Result clearSimulationFlowSourceKey(const std::string& name, int frame);

struct ParticleEmitterKey {
    int frame = 0;
    bool has_enabled = false;   bool  enabled = true;
    bool has_rate = false;      float rate_per_second = 32.0f;
    bool has_speed = false;     float speed = 2.0f;
    bool has_spread = false;    float spread = 0.35f;
    bool has_point = false;     Vec3  point;
    bool has_direction = false; Vec3  direction;
};
Result keyParticleEmitter(const std::string& name, const ParticleEmitterKey& key);
Result clearParticleEmitterKey(const std::string& name, int frame);

Result listSimulationFlowSources(std::vector<SimulationFlowSourceInfo>& out_sources);
Result getSimulationFlowSource(const std::string& name, SimulationFlowSourceInfo& out_source);
Result createSimulationFlowSource(const SimulationFlowSourceInfo& source,
                                  SimulationFlowSourceInfo& out_source);
Result updateSimulationFlowSource(const std::string& name,
                                  const SimulationFlowSourceInfo& source);
Result removeSimulationFlowSource(const std::string& name);
Result listSimulationColliders(std::vector<SimulationColliderInfo>& out_colliders);
Result getSimulationCollider(const std::string& name, SimulationColliderInfo& out_collider);
Result createSimulationCollider(const SimulationColliderInfo& collider,
                                SimulationColliderInfo& out_collider);
Result updateSimulationCollider(const std::string& name,
                                const SimulationColliderInfo& collider);
Result removeSimulationCollider(const std::string& name);
Result rebuildSimulationColliderSDF(const std::string& name);
Result resetFluidSimulation();
Result stepFluidSimulation(float dt = 0.0166667f);

// ---------------------------------------------------------------------------
// Terrain System (Faz 5.3c). The public handle is the terrain name; persistent
// manager ids remain diagnostic metadata and are never required by scripts.
// Creation/removal use the TerrainManager's normal mesh registration path and
// schedule the same renderer/acceleration-structure rebuilds as UI mutations.
// ---------------------------------------------------------------------------
struct TerrainInfo {
    int id = -1;
    std::string name;
    int width = 0;
    int height = 0;
    float size = 0.0f;
    float height_scale = 0.0f;
    bool has_node_graph = false;
    bool dirty = false;
};

Result listTerrains(std::vector<TerrainInfo>& out_terrains);
Result getTerrain(const std::string& terrain_name, TerrainInfo& out_info);
Result createTerrain(const std::string& requested_name, int resolution, float size,
                     float height_scale, TerrainInfo& out_info);
Result importTerrainHeightmap(const std::string& filepath, const std::string& requested_name,
                              float size, float height_scale, int max_resolution,
                              TerrainInfo& out_info);
Result removeTerrain(const std::string& terrain_name);
Result exportTerrainHeightmap(const std::string& terrain_name, const std::string& filepath);

struct TerrainEvaluationInfo {
    std::string terrain_name;
    std::string state;       // "idle"|"running"|"completed"|"cancelled"|"failed"
    float progress = 0.0f;
    unsigned int current_node_id = 0;
    std::string error;
};

Result evaluateTerrain(const std::string& terrain_name, TerrainEvaluationInfo& out_info);
Result getTerrainEvaluationStatus(const std::string& terrain_name, TerrainEvaluationInfo& out_info);
Result cancelTerrainEvaluation(const std::string& terrain_name);

struct TerrainErosionSettings {
    std::string type = "hydraulic";  // hydraulic|thermal|fluvial|wind
    std::string backend = "auto";    // auto|gpu|cpu
    int iterations = 0;              // 0 keeps the solver's type default
    unsigned int seed = 1337;
    float strength = 0.2f;           // wind
    float direction = 45.0f;         // wind degrees
    float talus_angle = 0.5f;        // thermal
    float amount = 0.3f;             // thermal erosion amount
    bool undo = true;
};

Result erodeTerrain(const std::string& terrain_name, const TerrainErosionSettings& settings);
Result applyTerrainPreset(const std::string& terrain_name, const std::string& preset,
                          bool replace_graph = false);
Result calculateTerrainFlow(const std::string& terrain_name);
Result sampleTerrainHeight(const std::string& terrain_name, float world_x, float world_z,
                           float& out_height);

struct TerrainRiverCarveSettings {
    std::string mode = "natural"; // simple|natural
    float depth_multiplier = 1.0f;
    float smoothness = 0.5f;
    bool post_erosion = false;
    int post_erosion_iterations = 12;
    float noise_strength = 0.3f;
    bool deep_pools = true;
    bool riffles = true;
    bool asymmetric_banks = true;
    bool point_bars = true;
    bool undo = true;
};

struct TerrainRiverInfo {
    int id = -1;
    std::string name;
    int control_point_count = 0;
    bool follow_terrain = true;
};

Result listTerrainRivers(std::vector<TerrainRiverInfo>& out_rivers);
Result carveTerrainRiver(const std::string& terrain_name, const std::string& river_name,
                         const TerrainRiverCarveSettings& settings);

// ---------------------------------------------------------------------------
// Hair & Groom System (Faz 5.4a). Interactive mouse strokes remain an artist
// workflow; this facade exposes deterministic groom creation and styling.
// ---------------------------------------------------------------------------
struct HairSettings {
    uint32_t guide_count = 1000;
    uint32_t children_per_guide = 4;
    uint32_t points_per_strand = 8;
    float length = 0.1f;
    float length_variation = 0.2f;
    float root_radius = 0.001f;
    float tip_radius = 0.0001f;
    float clumpiness = 0.5f;
    float child_radius = 0.01f;
    float curl_frequency = 0.0f;
    float curl_radius = 0.01f;
    float wave_frequency = 0.0f;
    float wave_amplitude = 0.0f;
    float frizz = 0.0f;
    float roughness = 0.0f;
    float gravity = 0.0f;
    float force_influence = 1.0f;
    bool use_dynamics = false;
    float physics_damping = 0.95f;
    float physics_stiffness = 0.1f;
    float physics_mass = 1.0f;
    bool use_tangent_shading = true;
    bool use_bspline = true;
    uint32_t subdivisions = 2;
};

struct HairGroomInfo {
    std::string name;
    std::string bound_mesh;
    size_t guide_count = 0;
    size_t child_count = 0;
    size_t point_count = 0;
    std::string material;
    bool visible = true;
    bool dirty = false;
    HairSettings settings;
};

Result listHairGrooms(std::vector<HairGroomInfo>& out_grooms);
Result getHairGroom(const std::string& groom_name, HairGroomInfo& out_info);
Result createHairGroom(const std::string& mesh_name, const std::string& requested_name,
                       const HairSettings& settings, HairGroomInfo& out_info);
Result removeHairGroom(const std::string& groom_name);
Result renameHairGroom(const std::string& groom_name, const std::string& new_name,
                       HairGroomInfo& out_info);
Result updateHairGroom(const std::string& groom_name, const HairSettings& settings,
                       const bool* visible = nullptr);
Result restyleHairGroom(const std::string& groom_name);
Result listHairPresets(std::vector<std::string>& out_presets);
Result applyHairPreset(const std::string& groom_name, const std::string& preset);
Result trimHairGroom(const std::string& groom_name, float length_factor);
Result growHairGroom(const std::string& groom_name, float length_factor);
Result combHairGroom(const std::string& groom_name, Vec3 world_direction,
                     float strength = 0.5f, float root_stiffness = 0.75f);
Result smoothHairGroom(const std::string& groom_name, float strength = 0.5f,
                       int iterations = 2);
Result resetHairSimulation(const std::string& groom_name);
Result bakeHairGroom(const std::string& groom_name);

// ---------------------------------------------------------------------------
// Mesh Paint Automation (Faz 5.4b). Mouse strokes remain interactive; this
// surface exposes deterministic layer-stack and full-channel operations.
// ---------------------------------------------------------------------------
struct PaintLayerInfo {
    int index = -1;
    uint32_t id = 0;
    std::string name;
    bool visible = true;
    bool locked = false;
    float opacity = 1.0f;
    std::string blend_mode;
    std::vector<std::string> channels;
};

struct PaintTargetInfo {
    std::string object_name;
    uint16_t material_id = 0;
    int resolution = 0;
    std::vector<std::string> channels;
    std::vector<PaintLayerInfo> layers;
};

Result getPaintTarget(const std::string& object_name, int material_id,
                      PaintTargetInfo& out_info);
Result ensurePaintTarget(const std::string& object_name, int material_id, int resolution,
                         PaintTargetInfo& out_info);
Result addPaintLayer(const std::string& object_name, int material_id,
                     const std::string& name, int insert_at, PaintLayerInfo& out_info);
Result removePaintLayer(const std::string& object_name, int material_id, int layer_index);
Result updatePaintLayer(const std::string& object_name, int material_id, int layer_index,
                        const std::string* name = nullptr, const bool* visible = nullptr,
                        const bool* locked = nullptr, const float* opacity = nullptr,
                        const std::string* blend_mode = nullptr);
Result fillPaintLayer(const std::string& object_name, int material_id, int layer_index,
                      const std::string& channel, Vec3 color);
Result clearPaintLayerChannel(const std::string& object_name, int material_id,
                              int layer_index, const std::string& channel);
Result duplicatePaintLayer(const std::string& object_name, int material_id, int layer_index,
                           PaintLayerInfo& out_info);
Result movePaintLayer(const std::string& object_name, int material_id,
                      int from_index, int to_index);
Result mergePaintLayerDown(const std::string& object_name, int material_id, int layer_index);
Result flattenPaintLayers(const std::string& object_name, int material_id);
Result bakePaintHeightToNormal(const std::string& object_name, int material_id,
                              float strength = 4.0f, bool clear_height = false);
Result importPaintChannel(const std::string& object_name, int material_id, int layer_index,
                          const std::string& channel, const std::string& filepath);
Result exportPaintChannel(const std::string& object_name, int material_id, int layer_index,
                          const std::string& channel, const std::string& filepath);
Result listPaintMaskPresets(std::vector<std::string>& out_presets);
Result applyPaintMaskPreset(const std::string& object_name, int material_id, int layer_index,
                            const std::string& preset, float strength = 1.0f,
                            unsigned int seed = 1337);

// ---------------------------------------------------------------------------
// Deterministic Sculpt Automation (Faz 5.4c). Points and directions are world
// space; viewport picking and mouse-event replay are intentionally excluded.
// ---------------------------------------------------------------------------
struct SculptInfo {
    std::string object_name;
    size_t vertex_count = 0;
    bool has_mask = false;
    float mask_min = 0.0f;
    float mask_max = 0.0f;
};

struct SculptStrokeSettings {
    std::string tool = "inflate"; // draw|inflate|flatten|smooth|stamp|noise
    std::vector<Vec3> points;
    float radius = 0.25f;
    float strength = 0.05f;
    float falloff = 0.75f;
    Vec3 direction = Vec3(0.0f, 1.0f, 0.0f); // flatten/stamp direction
    unsigned int seed = 1337;
    bool use_mask = true;
    bool undo = true;
};

Result getSculptInfo(const std::string& object_name, SculptInfo& out_info);
Result applySculptStroke(const std::string& object_name,
                         const SculptStrokeSettings& settings);
Result paintSculptMask(const std::string& object_name, const std::vector<Vec3>& points,
                       float radius, float value, float strength = 1.0f,
                       bool undo = true);
Result applySculptMaskOperation(const std::string& object_name,
                                const std::string& operation,
                                unsigned int seed = 1337, bool undo = true);

// ---------------------------------------------------------------------------
// Scripting. Execution stays on the main thread; Python exceptions are caught
// by the runtime layer and returned through the facade's normal Result model.
// ---------------------------------------------------------------------------
Result runScriptFile(const std::string& filepath);

} // namespace rtapi
