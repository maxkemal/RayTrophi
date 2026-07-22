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
};

std::vector<LightInfo> listLights();
Result addLight(const std::string& type, const Vec3& position, std::string& out_name);
Result deleteLight(int index);
Result setLightPosition(int index, const Vec3& position);

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

Result createFluidDomain(const std::string& name, Vec3 domain_min, Vec3 domain_max,
                         float voxel_size, const std::string& type, FluidDomainInfo& out_info);
Result removeFluidDomain(const std::string& domain_id_or_name);
Result getFluidDomain(const std::string& domain_id_or_name, FluidDomainInfo& out_info);
Result seedFluidParticles(const std::string& domain_id_or_name, Vec3 seed_min, Vec3 seed_max,
                           int particles_per_cell = 4, bool replace = true);
Result clearFluidParticles(const std::string& domain_id_or_name);
Result updateFluidDomain(const std::string& domain_id_or_name,
                         const Vec3* domain_min = nullptr, const Vec3* domain_max = nullptr,
                         const float* voxel_size = nullptr, const std::string* render_mode = nullptr,
                         const std::string* backend = nullptr, const std::string* boundary = nullptr,
                         const std::string* preset = nullptr, const float* viscosity = nullptr,
                         const bool* enabled = nullptr, const bool* visible = nullptr);
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
