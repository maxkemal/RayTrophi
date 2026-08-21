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
namespace NodeSystem { namespace Sim { class SimulationNodeGraph; } }

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
// The bound context, for in-process callers that must run an IPC handler
// directly (the rt.agent script bindings reuse the agent.* dispatch instead of
// reimplementing it, so script and IPC cannot answer differently). Null when
// unbound. Not an IPC-visible value: nothing here crosses the wire.
UIContext* boundContext();

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

struct TemplateOpenInfo {
    std::string template_id;
    std::string state;
    std::string code;
    bool opened = false;
    bool ui_state_applied = false;
    std::vector<std::string> errors;
    std::vector<std::string> warnings;
};
Result openTemplate(const std::string& id, const std::string& conflict_policy,
                    TemplateOpenInfo& out);
Result saveUserTemplate(const std::string& display_name,
                        const std::string& description,
                        const std::string& category);
Result deleteUserTemplate(const std::string& template_id);

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

// ★★★ getObjectTransform above returns the AUTHORED pose, and no solver
// writes there. Worse than that, and it cost a wrong fix first: the rigid
// solver does not write a transform AT ALL. RigidBodySystem::step bakes the
// world delta D = B(t)*inv(B0) into the MESH VERTICES and deliberately leaves
// the transform handle untouched (moving an imported object's transform
// corrupted it in the renderer). So `Transform::current` is empty too, and
// composing final = current * base changes nothing.
//
// Measured 2026-08-19: physics.set_gravity + physics.add_body + 240 x
// physics.step, over BOTH channels, and every transform reader still said
// y = 50.0 - successfully, with entirely plausible numbers. The obvious
// reading is "the solver is broken"; the truth was "the pose you are asking
// for is not stored in a pose".
//
// This reader composes the rigid body's `last_rigid_delta` (the pose-tracking
// authority) onto the spawn transform. `out_simulated`, when given, says
// whether a solver actually contributed - false means the value is merely the
// authored pose, which is exactly the distinction a validation case must not
// have to guess. Free fall, buoyancy draft, restitution: none of them are
// observable without this.
//
// Soft bodies and cloth own vertices rather than a pose; for those this
// reports the authored transform with out_simulated = false.
Result getObjectWorldTransform(const std::string& name, Matrix4x4& out,
                               bool* out_simulated = nullptr);

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
// Supported color parameters: base_color, emission,
//   resin_color, resin_dirt_color, dust_color_a, dust_color_b.
// Supported scalar parameters: roughness, metallic, specular,
// emission_strength, transmission, ior, opacity,
//   bubble_ior, bubble_film,
//   resin_density, resin_roughness, resin_inclusion, resin_dirt,
//   resin_inclusion_scale, resin_shard, resin_shard_hue.
// Booleans ride the scalar (>0.5 = true): is_bubble, resin_object_space.
// Enums ride the scalar (rounded): dust_style (0..3), shard_shape (0..1).
//
// Names match the .rtp serializer keys (MaterialNodesV2.h) so a save file and
// a script speak the same vocabulary. NOTE resin_density is the field the
// shaders call transmission_density — it is the resin COAT thickness on the
// resin path and the interior depth on the glass path.
//
// The thin-shell (is_bubble) and resin parameters also drive the fluid
// ISOSURFACE when a material is bound to a grid domain's fluid surface, so
// these are what a script needs to exercise volume_closesthit.rchit's
// thin-shell and resin branches.
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
// World thermal ambient (docs/dev/SIMULATION_NODE_OBJECT_MODEL.md section 7
// item 1). Distinct from WorldState above: that one is the render sky, this
// is the ambient condition every uncoupled substance relaxes toward. Mirrors
// WorldThermalState (MaterialStateField.h) field for field so a script sees
// exactly what the solver reads -- this is the first scripting surface it has
// ever had.
// ---------------------------------------------------------------------------
struct WorldThermalInfo {
    float ambient_kelvin = 293.0f;
    float kelvin_per_unit = 350.0f;
    float convection_coefficient = 1.0f;
    float oxygen_availability = 1.0f;
};
Result getWorldThermal(WorldThermalInfo& out);
// Every field optional: nullptr leaves it unchanged. Same partial-update
// shape as updateFluidDomain, so a caller (or a World node's opt-in fields)
// can touch one field without first reading the other three.
Result setWorldThermal(const float* ambient_kelvin = nullptr,
                       const float* kelvin_per_unit = nullptr,
                       const float* convection_coefficient = nullptr,
                       const float* oxygen_availability = nullptr);

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
// Vulkan volume instrumentation (render.volume_stats / render.volume_counters).
//
// ★ These counters existed for a long time but reached only the Volume
// Performance panel, so diagnosing a volume cost or a missing-surface report
// meant asking a human to read numbers off a panel and type them back. That is
// the one manual step that stops a render regression from being reproducible.
//
// The counters are OPT-IN: GPU atomics cost a little, so leaving them on
// contaminates frame-time measurements. Enable, render some frames, read, then
// disable. Counters accumulate until reset.
//
// ★★ Zeroes need care. `volume_rays == 0` means no ray entered a volume at all
// (wrong camera, wrong frame, or the domain is not in the TLAS) — it does NOT
// mean the volume is cheap. And `solid_probe_runs == 0` with rays > 0 means the
// gate suppressed every probe, which is a different failure from a probe that
// ran and found nothing (`hits == 0`). Report both, never just their sum.
struct VolumeInstrumentationInfo {
    bool     available = false;   // false = no Vulkan backend active
    bool     enabled = false;
    uint32_t volume_rays = 0;
    uint32_t density_samples = 0;
    uint32_t shadow_density_samples = 0;
    uint32_t empty_segments_skipped = 0;
    uint32_t topology_segments_skipped = 0;
    uint32_t majorant_segments_skipped = 0;
    uint32_t majorant_queries = 0;
    uint32_t majorant_available_queries = 0;
    uint32_t extinction_terminations = 0;
    uint32_t step_budget_exhausted = 0;
    uint32_t completed_intervals = 0;
    uint32_t temporal_accepted = 0;
    uint32_t temporal_rejected = 0;
    uint32_t solid_probe_runs = 0;
    uint32_t solid_probe_hits = 0;
    uint32_t gas_handoffs = 0;
    uint32_t layered_handoffs = 0;
    uint32_t arbiter_rejects = 0;
    uint32_t teleports = 0;
    uint32_t arbiter_candidates = 0;
    uint32_t arbiter_gate_open = 0;
    uint32_t arbiter_no_box = 0;
    uint32_t arbiter_empty_range = 0;
    uint32_t arbiter_no_crossing = 0;
};
VolumeInstrumentationInfo volumeStats();
Result setVolumeInstrumentation(bool enabled);  // also zeroes the counters

// ---------------------------------------------------------------------------
// Viewport measurement.
//
// ★★★ An agent that can only SAVE a render is guessing; one that can read the
// render's DATA is measuring. Both halves of the 2026-08-16 black-band round
// were lost to this gap: every counter had to be copied out of the panel by
// hand, and one whole round was spent on a dump taken from a frame with no fire
// in it — a condition the agent could not check because it could neither drive
// the viewport nor query it.
//
// ★★ Everything here stays inside the IPC data-model rule: names, ids and
// VALUES only. Driving is a command, counters are a value struct, pixels are a
// buffer. No handle, no engine object and no core access crosses the boundary.
// See docs/dev/IPC_SECURITY_PERFORMANCE.md, "Data model boundary".
struct ViewportStatusInfo {
    bool available = false;      // false = no backend bound yet
    std::string backend;         // "vulkan", "optix", "cpu"
    int  width = 0;
    int  height = 0;
    int  samples = 0;            // accumulated samples so far
    bool accumulation_complete = false;
    float ms_per_sample = 0.0f;
    // ★ Distinguishes "converged, so it stopped" from "never started". Reading
    // zero counters while this is false is NOT a measurement of a cheap scene;
    // it is the absence of a measurement, which is the trap volume_rays == 0
    // set on 2026-08-16.
    bool rendering_active = false;
    bool capture_enabled = false;   // is the probe buffer being filled?
    bool frame_available = false;   // has a frame been captured yet?
    // ★ WHICH viewport produced the numbers above. A probe taken in Solid mode
    // and one taken in Rendered mode are different measurements of different
    // images; reporting the counters without this field invites comparing them.
    std::string shading;            // "solid" | "material" | "rendered" | "matcap"
};
ViewportStatusInfo viewportStatus();

// ---------------------------------------------------------------------------
// Viewport shading mode.
//
// ★★★ Panel-only until 2026-08-19, and it broke agent chains in the middle: an
// agent could drive the scene, force frames and measure pixels, but could not
// say which viewport it was measuring, so "switch to Rendered and check" always
// ended in "please click the button" — exactly the manual step CLAUDE.md rule 1
// forbids. Names cross the boundary, never the panel's integer.
struct ViewportShadingInfo {
    std::string mode;                    // current mode, as a name
    int  matcap_preset = 0;              // 0..9, meaningful in matcap mode
    // ★ false = this machine has no raster viewport (no Vulkan), so "rendered"
    // is the ONLY reachable mode. Without this a caller reads a rejected
    // set_shading as a bug in its own request.
    bool interactive_available = false;
};
ViewportShadingInfo viewportShading();

// `mode`: solid | material | rendered | matcap. "preview" is accepted as an
// alias for material because the panel button reads Preview.
// `matcap_preset`: 0..9, or -1 to leave it alone.
// Resets accumulation exactly like the panel buttons do — otherwise the next
// probe would measure the frame from the mode you just left.
Result setViewportShading(const std::string& mode, int matcap_preset = -1);

// Per-frame capture of the displayed frame costs a copy, so it is opt-in and
// off by default. Enabling it does not change what is rendered.
Result setViewportCapture(bool enabled);

// Force the viewport to accumulate N frames synchronously.
// Used by agents to artificially advance the viewport without waiting for UI loops.
//
// ★★★ This ACCUMULATES but does not PUBLISH. The capture buffer is filled by the
// display loop (Main.cpp), from the same SDL surface the viewport presents; this
// call renders without a surface on purpose. Over IPC that is invisible, because
// each request returns to the loop and the loop publishes before the next one
// arrives. Inside a SINGLE script it is not: the script holds the main thread, so
// no frame is ever published and frame_available stays false however many frames
// were rendered. Measured 2026-08-19.
//   script:  capture(true); render_frames(8); status() -> frame_available FALSE
//   IPC:     the same three calls -> frame_available TRUE
// A script that needs a measured frame has to hand control back - split it across
// separate script.run_file calls, or drive the sequence over IPC.
// This is the producer-vs-consumer split CLAUDE.md warns about: the producer is
// the display loop, the consumer is the probe, and they are different loops.
struct ViewportRenderResult {
    bool success = false;
    std::string error;
    int samples_rendered = 0;
    bool converged = false;
    float ms_per_frame = 0.0f;
};
ViewportRenderResult renderViewportFrames(int count);

// Rectangle in pixels. All zero = whole frame.
struct ViewportProbeRegion {
    int x = 0, y = 0, width = 0, height = 0;
};

// Numeric description of the captured frame. This is the half that turns "it
// looks black" into a threshold an automated check can fail on.
struct ViewportProbeInfo {
    bool available = false;      // false = capture disabled or no frame yet
    int  width = 0;              // of the probed REGION, not the frame
    int  height = 0;
    uint32_t pixels = 0;
    float mean_luminance = 0.0f;
    float min_luminance = 0.0f;
    float max_luminance = 0.0f;
    // Fraction of pixels at or below the black threshold. The black band was a
    // step change in exactly this number.
    float black_fraction = 0.0f;
    // ★ A separate class of failure nobody was watching for: a NaN reads as
    // neither black nor lit, and averages hide it completely.
    float nan_fraction = 0.0f;
    uint32_t histogram[8] = {0};  // luminance buckets, 0..1 clamped
};
ViewportProbeInfo probeViewportFrame(const ViewportProbeRegion& region,
                                     float black_threshold = 0.001f);

// Captures the current viewport frame and returns it as a Base64 encoded JPEG string.
std::string getViewportScreenshotAsBase64();

// Engine-side hook. Called by the display loop with the SAME frame the viewport
// shows, so a probe measures what the user sees rather than a re-render. Copies
// under a lock and only when capture is enabled; a no-op otherwise.
// `pixels` is tightly packed RGBA8 unless `pitch_bytes` says otherwise.
void publishViewportFrame(const void* pixels, int width, int height,
                          int pitch_bytes);

// ★ Cheap gate for the display loop. The publish call itself is a no-op when
// capture is off, but the CALLER still has to convert the surface to RGBA8 to
// make the call — a per-frame allocation charged to every user who never asked
// for a probe. Test this first and skip the conversion entirely.
bool viewportCaptureEnabled();

// ---------------------------------------------------------------------------
// Simulation node graph (Faz N0/N1/N2).
// docs/dev/NODE_SIMULATION_ARCHITECTURE_PLAN.md, BOLUM D.
//
// ★★ The graph DRIVES the existing solvers; it is not a simulation core and it
// owns no state. Evaluating it produces COMMANDS — an intent the caller can
// inspect without anything being applied. That separation is what lets a script
// verify a graph's meaning without running a simulation.
struct SimCommandInfo {
    std::string kind;        // "bind_domain", "set_parameter", "couple"
    // "domain" | "object" | "world" — which storage this command targets.
    // Only set_parameter reads this today: a World command carries no target
    // name (there is exactly one world), so the apply layer needs this to
    // tell "domain command with an empty target" (an error) apart from
    // "world command, which has none by design".
    std::string scope = "domain";
    std::string target;      // domain/object NAME — identity, never a handle
    std::string key;
    float       value = 0.0f;
    std::string text;
    uint32_t    source_node = 0;
};
// ★ A node asking for a restart REPORTS it; nothing acts on it. Discarding a
// running simulation is the user's decision, and doing it silently is the
// failure shape this codebase keeps rediscovering.
struct SimRestartRequest {
    uint32_t    node_id = 0;
    std::string reason;
};
struct SimGraphEvaluation {
    bool evaluated = false;
    // ★★ Why a graph produced nothing. An empty command list on a graph that was
    // never found must not read as "this graph declares no commands" — that is
    // the "a default is not a measurement" shape. `evaluated` false plus this
    // string says the reading was not taken.
    std::string error;
    std::vector<SimCommandInfo>   commands;        // topological order IS meaningful
    std::vector<SimRestartRequest> restart_requests;
};
// One opt-in parameter on a Solver / Domain Settings node.
//
// ★★★ `in_use` false does NOT mean "zero". It means this graph has no opinion
// about the parameter and will not write it. Collapsing the two would make an
// untouched dial overwrite an authored value with a number nobody chose.
struct SimNodeField {
    std::string key;          // the same name readParameter/writeParameter use
    bool        in_use = false;
    float       value = 0.0f;
    // Applying this field would discard the accumulated simulation state.
    // Reported per field so the prompt can name the one dial responsible.
    bool        requires_restart = false;
};

struct SimNodeInfo {
    uint32_t    id = 0;
    // The implicit node naming this graph's owner. It cannot be retargeted, so a
    // panel should draw it pinned rather than offering an editable name field.
    bool        is_owner_node = false;
    std::string type_id;
    std::string display_name;
    bool        enabled = true;
    int         input_count = 0;
    int         output_count = 0;
    std::string domain;      // DomainRef nodes
    std::string channel;     // Field nodes
    std::string source;      // "grid" | "attribute"
    // Field Inspect only. ★ `stats_available` false means the value could NOT be
    // measured; it is not the same as a field that measured zero.
    bool     has_stats = false;
    bool     stats_available = false;
    uint32_t particle_count = 0;   // elements measured
    // ★★ The backing array can be LONGER than the particle count — a granular
    // array outlived six removed particles on 2026-08-17. `array_in_sync` false
    // means the solver's arrays disagree with each other; the statistics above
    // still describe the live particles only.
    uint32_t array_size = 0;
    bool     array_in_sync = true;
    // ★★★ Surface Inspect only. False means the reading is the host mirror as of
    // the last readback — or, before any readback, the INITIALISATION values.
    // Measured 2026-08-17: fuel_remaining = -1 across a whole crate is the
    // "not seeded yet" sentinel, not a fuel level.
    bool     host_fresh = true;
    // ★ double: substance_tag is a uint32 identity and a float rounds it into a
    // DIFFERENT identity past 2^24 (measured: ...163 read back as ...160).
    double   min_value = 0.0;
    double   max_value = 0.0;
    double   mean_value = 0.0;
    // Cache node only. ★★★ `cache_stale` true means a cache EXISTS and was built
    // from a DIFFERENT authored config — it still serves frames, and they
    // describe a scene that no longer exists. Nothing else tells it apart from a
    // healthy cache, which is how stale physics reaches a render.
    bool     has_cache_status = false;
    bool     cache_valid = false;
    bool     cache_baking = false;
    bool     cache_stale = false;
    uint32_t cache_ram_frames = 0;
    // Solver / Domain Settings nodes. Empty for every other node type.
    // ★ Published so a script can DISCOVER which parameters a node offers
    // instead of reading the source — the same reason the attribute naming
    // layer exists.
    std::vector<SimNodeField> fields;
};

// ── Editor view state (rt.editor) ───────────────────────────────────────────
//
// ★★★ Not a hole in the "panels are not scripted" rule — a correction to what
// that rule was protecting. Drawing is exempt because a draw call has no meaning
// outside a frame's draw context, and `rt.ui` (register your own panel, emit
// widgets) stays in-process for exactly that reason. But which editor is OPEN is
// not a draw call, it is a VALUE, and leaving it unreadable made agents
// structurally blind to this repository's most expensive failure class: the
// panel disagreeing with the core. `Volume` as a default made the panel a liar;
// the gas shader reader answered from a field the writer never touched. Neither
// is visible to a caller that can only read the solver.
//
// ★★ What is deliberately NOT here: driving widgets ("click the button labelled
// X"). That would make labels load-bearing, break on every restyle, and restore
// the UI as an authority. If a button does something, that something needs its
// own API — which is rule 1, not an exception to it.
struct EditorState {
    // "none" | "dope_sheet" | "graph_editor" | "console" | "assets" |
    // "simulation" | "geometry" | "material" | "terrain" | "anim_graph"
    std::string bottom_editor;
    // "simulation" | "geometry" | "material" | "terrain" | "animation"
    std::string node_editor_domain;
    bool        node_editor_open = false;   // the Nodes window itself is showing
    // ★★★ EVERY bottom editor currently showing, not just the first one found.
    // `bottom_editor` above names one, and a reader that only ever names one
    // cannot report the failure it is most likely to be asked about: two panels
    // open at once because an exclusivity rule was routed around. A reader that
    // structurally cannot see a defect is the same trap as the gas shader reader
    // that answered from a field its writer never touched.
    std::vector<std::string> open_editors;
    // Which SCOPED simulation graph the Nodes canvas is on: "object" | "domain"
    // | "world", plus the owner's name (empty for world, and empty for the
    // other two when nothing has been picked yet). Reported even when the Nodes
    // window is closed — it is the selection, not a property of the window.
    std::string sim_graph_scope;
    std::string sim_graph_owner;
};
EditorState editorState();
Result setBottomEditor(const std::string& name);
Result setNodeEditorDomain(const std::string& name);
// ★ Selecting a scope does not require a graph to exist there: the panel draws
// an explicit empty state, which is how a user reaches graph creation at all.
Result setSimGraphScope(const std::string& scope, const std::string& owner);

// ── rt.perf: where the time went, as values ─────────────────────────────────
//
// ★★★ A timing that only reaches the Scene Log is not a measurement this project
// can use. The single maintainer cannot sit in front of every build, and an
// agent driving the app over IPC cannot read the log at all — so the previous
// mesh profiler (MeshProfileTimer.h, macro compiled out to `((void)0)`) produced
// exactly zero readable numbers. Sections are values now.
//
// ★ Section names are stable strings written by whoever performs the work, so a
// caller reads a phase by name rather than by guessing at ordering:
//   terrain.graph.evaluate / .height / .aux_outputs / .finalize_mesh
//   terrain.mesh_fill / .create / .update / .publish_fields
//   terrain.splat_resize
//   Renderer::rebuildBVH(...) / Renderer::rebuildBackendGeometry(GPU)
//
// ★★ Reading does NOT go through the frame-loop queue (see PerfProfile.h): the
// most useful moment to ask what a build is spending its time on is while the
// UI thread is busy, and an enqueued read would wait behind that exact work.
struct PerfSection {
    std::string name;
    double   last_ms = 0.0;
    double   total_ms = 0.0;
    double   max_ms = 0.0;
    uint64_t count = 0;
    double   last_rss_delta_mb = 0.0;   // working-set delta across the scope
    double   rss_after_mb = 0.0;
    uint64_t seq = 0;                   // monotonic write order, newest highest
};
// Newest write first.
std::vector<PerfSection> perfSections();
bool perfSection(const std::string& name, PerfSection& out);
Result perfReset();
// Mirror completed sections into the Scene Log as well. Off by default.
Result perfSetLogging(bool enabled);
bool perfLogging();

void initSimulationNodes();   // register types + install the attribute resolver

// ── Scoped simulation graphs ────────────────────────────────────────────────
//
// Decision record: docs/dev/SIMULATION_NODE_OBJECT_MODEL.md, section 8 steps 1-2.
//
// ★★★ A graph belongs to a NAMED scene entity: an object, a domain, or the
// world. There is no "the" simulation graph any more, and — deliberately — no
// optional scope argument. "Use the active domain" is exactly the silent
// assumption this repository keeps paying for: the call succeeds, configures a
// different entity, and no later reading contradicts it.
//
// scope is "object" | "domain" | "world"; owner is the entity's name, and is
// ignored for "world" because there is only one.
struct SimGraphRef {
    std::string scope;
    std::string owner;
    uint32_t    node_count = 0;
    // The implicit node naming this graph's owner. 0 means the scope has none —
    // World, whose thermal state has no scripting surface yet (section 7).
    uint32_t    owner_node = 0;
    // ★★★ The named entity is GONE, but the graph is still here. Removing a
    // domain drops its graph at that one call site; objects are deleted through
    // several paths (command history, UI, project load), so instead of claiming
    // to have hooked them all this is MEASURED and reported. A stranded graph
    // still draws and still accepts edits while driving nothing — the fracture
    // UI state shape — and a flag nobody can read is how that stays invisible.
    bool        owner_missing = false;
};
std::vector<SimGraphRef> simGraphList();
// Idempotent: creating an existing graph succeeds and changes nothing. Fails
// when the named entity does not exist, so a typo cannot produce a live graph
// that drives nothing.
Result simGraphCreate(const std::string& scope, const std::string& owner);
Result simGraphDelete(const std::string& scope, const std::string& owner);

// ★★★ The editor panel draws THIS object; it does not keep a copy. A panel that
// mirrors state the core owns is how the fracture UI cache outlived a scene
// change and how the panel came to disagree with the solver — so the drawing
// surface is given the original, never a snapshot.
//
// ★★ Returns null when no graph exists for the scope. The panel must say so
// rather than fall back to another graph: showing one owner's nodes under
// another's name is the panel-lies failure this layer exists to end.
// Forward-declared to keep solver/node headers out of the API header (D.4).
NodeSystem::Sim::SimulationNodeGraph* simulationGraph(const std::string& scope,
                                                      const std::string& owner);
// Discoverability, which is the whole point of the naming layer: until now the
// only way to learn an attribute existed was to read the solver source.
std::vector<std::string> simListAttributes(const std::string& domain);
// Same, for per-object surface (MSF) attributes: temperature, char, melt,
// moisture, fuel_remaining, mass_loss. N5.
std::vector<std::string> simListSurfaceAttributes(const std::string& object);
// ★ clear() empties the canvas but re-seeds the owner node: a scoped graph is
// never ownerless, because a node authored on an ownerless canvas would name
// nothing.
Result simGraphClear(const std::string& scope, const std::string& owner);
Result simGraphAddNode(const std::string& scope, const std::string& owner,
                       const std::string& type_id, uint32_t& out_id);
Result simGraphSetNodeText(const std::string& scope, const std::string& owner,
                           uint32_t node_id, const std::string& key,
                           const std::string& value);
Result simGraphSetNodeValue(const std::string& scope, const std::string& owner,
                            uint32_t node_id, const std::string& key, float value);
Result simGraphConnect(const std::string& scope, const std::string& owner,
                       uint32_t from_node, int from_pin,
                       uint32_t to_node, int to_pin);
SimGraphEvaluation simGraphEvaluate(const std::string& scope, const std::string& owner);
// ★★★ Returns a Result, not just the vector. An empty list from a graph that
// was never found is indistinguishable from a graph that genuinely has no
// nodes — "a default is not a measurement", and here the default would tell a
// caller its typo'd owner name was fine. Measured 2026-08-18: the test that
// asserted an unknown owner is refused caught exactly this.
Result simGraphNodes(const std::string& scope, const std::string& owner,
                     std::vector<SimNodeInfo>& out_nodes);

// N3 — applying the graph as an OVERRIDE layer.
//
// ★★★ Overrides are reversible by construction: the authored value is captured
// before the first write of a key and restored by simGraphClearOverrides().
// A graph never mutates authored data (plan B.5) — solver configuration is
// runtime state and has to stay resettable to frame 0.
struct SimApplyResult {
    bool     ok = false;
    uint32_t applied = 0;
    uint32_t overrides_held = 0;     // keys whose authored value we are holding
    // Parameters that need a simulation restart, when the caller did not allow
    // one. ★ Refused and reported, never applied quietly: a graph edit must not
    // discard a running simulation on its own.
    std::vector<std::string> refused;
    std::vector<std::string> failed;
};
SimApplyResult simGraphApply(const std::string& scope, const std::string& owner,
                             bool allow_restart);
// ★★★ The override layer is deliberately NOT scoped. It is keyed by the entity
// and parameter actually written, so it already spans graphs — and restoring
// "only this graph's" overrides is not a thing the solver can do: two graphs
// that wrote the same key hold one authored value between them. Scoping the
// restore would leave whichever graph cleared second unable to put anything
// back, silently stranding an authored value.
Result   simGraphClearOverrides();   // restores every captured authored value
uint32_t simGraphOverrideCount();

// N4 — couplings, and the reason this phase exists.
//
// ★★★ The graph DECLARES couplings; stepGridDomains decides the order they
// actually run in. Both are reported here so they can be compared. A graph that
// showed a chosen order while the solver ran a different one would look like
// control and be a lie — and "producer ≠ consumer" is already one of this
// repository's recurring failure classes.
struct SimCouplingEntry {
    std::string coupling;        // "fluid_to_gas", "gas_to_fluid_ignition", ...
    std::string producer;        // actual entries only: which system wrote
    std::string consumer;        // actual entries only: which system read
    std::string source_domain;
    std::string target_domain;
    bool        active = false;
    uint32_t    source_node = 0; // declared entries only
    // Which graph declared it (declared entries only). A coupling joins two
    // domains, so the report spans every scope and has to say where each
    // declaration came from.
    std::string scope;
    std::string owner;
};
struct SimCouplingReport {
    std::vector<SimCouplingEntry> declared;   // graph order
    std::vector<SimCouplingEntry> actual;     // solver execution order
    // ★ false means the solver was never asked — no particle system exists. An
    // empty `actual` with traced == true means "stepped, and nothing coupled",
    // which is a measurement. Without this flag the two are indistinguishable.
    bool traced = false;
    bool order_matches = true;
    std::vector<std::string> declared_not_running;
    std::vector<std::string> running_not_declared;
};
SimCouplingReport simGraphCouplings();

// N6 — the bake, and the state a script needs to reason about it.
//
// ★★ "Bake" in this application is not a hidden background job: playing the
// timeline caches each frame in RAM, and bakeSimulation() writes a deterministic
// disk cache for a frame range. Both already exist; what was missing was any way
// to SEE the result from outside.
//
// ★★★ Three states look identical from the outside and are not: nothing baked
// yet, a bake running, and a bake INVALIDATED because the authored config
// changed. The last is the one that silently makes a render use stale physics,
// and it is invisible without the signature.
struct SimCacheStatus {
    bool     valid = false;          // a disk bake is bound and usable
    bool     baking = false;         // a cooperative bake is running right now
    std::string cache_dir;           // empty when the bake is RAM-only
    uint32_t ram_frames = 0;         // frames held in the timeline scrub cache
    int      first_frame = 0;
    int      last_frame = 0;
    bool     has_range = false;      // false when ram_frames == 0
    uint64_t config_signature = 0;   // authored config hash the cache was built from
};
Result simCacheStatus(SimCacheStatus& out);
// Blocking on purpose: baking is an explicit action, and the caller asked for
// it. The interactive UI uses the cooperative begin/tick path instead.
Result simBake(const std::string& cache_dir, int start_frame, int end_frame,
               float fps);
Result simClearCache();

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

struct FoliageAssetInfo {
    std::string name;
    std::string category;
    std::string relative_path;
};

Result listFoliageAssets(std::vector<FoliageAssetInfo>& out_assets);
Result addLibraryScatterSource(const std::string& group_id_or_name, const std::string& relative_path);

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
// ★★★★ Advances the solver AND the playhead, and claims the timeline for
// the caller until the user takes it back.
//
// It used to advance only the solver. The playhead stayed put, so the frame
// loop saw a rigid state that disagreed with the displayed frame and corrected
// it - resetting the runtime to the rest pose and erasing every scripted step,
// while this call kept returning success. Measured over IPC 2026-08-19:
// 0.29 m of real motion (exactly gravity, read inside one batch), 100% of it
// gone one call later. A caller did real work, was told it worked, measured
// nothing, and had every reason to blame the solver.
//
// The timeline is still the USER's: scrubbing, play or stop drop the claim on
// the spot. What the caller gets is not ownership, it is NOTICE - see
// getSimControlState.
Result stepPhysicsSimulation(float dt = 0.0166667f);

// Who last moved the solvers, and an epoch that changes whenever anything
// does. Read it before and after a measurement: if it moved, the measurement
// is void. Without it a reverted pose and an unmoved body read identically.
struct SimControlStateInfo {
    unsigned long long epoch = 0;
    std::string driver = "none";   // none | user | playback | script
    bool script_driving = false;
    double script_sim_seconds = 0.0;
    int frame = 0;
    bool playing = false;
};
Result getSimControlState(SimControlStateInfo& out);
Result setPhysicsGravity(Vec3 gravity);
Result getPhysicsGravity(Vec3& out_gravity);

// What a cut actually produced. `shard_objects` is what you hand to
// makePhysicsFractureGroup, and `shard_clusters` is the parallel cluster index
// per shard — same order, same length, so the two cannot drift.
struct FractureResultInfo {
    std::string object;
    std::vector<std::string> shard_objects;
    std::vector<int> shard_clusters;
    int cluster_count = 0;
    // Sites the generator settled on, which is NOT always what was asked for:
    // candidates outside the hull are rejected. Reported so a test can tell
    // "the pattern starved" apart from "the cut failed".
    int site_count = 0;
};

// Cut `node` into Voronoi shards. The object is parked (kept alive, pulled out
// of the scene) and the shards take its place, exactly as the panel does — this
// drives the same generator, so there is one parking/naming/erasing path.
// `pattern`: 0 uniform, 1 impact-clustered, 2 thermal (burn-guided).
Result fractureObject(const std::string& node, int site_count, uint32_t seed,
                      int pattern, int cluster_count, bool exact_surface,
                      float preview_gap, FractureResultInfo& out_info);
// Drop the shards and restore the parked original.
Result unfractureObject(const std::string& node);
// The group names + members "Make Breakable" would create for this object, so a
// script registers the same clusters the panel would.
Result fractureClusterGroups(const std::string& node,
                             std::vector<std::string>& out_groups,
                             std::vector<std::vector<std::string>>& out_members);

struct FractureGroupInfo {
    std::string group;
    int shard_count = 0;
    int broken_count = 0;
    // ★ The authored resistance, in METRES PER SECOND: the velocity change this
    // group absorbs before it comes apart. Mass-free, so one value reads the
    // same on a plank and on a tower leg.
    float base_break_velocity = 0.0f;
    // Summed shard mass [kg]. The two impulse figures below are this times the
    // velocity above, which is the only reason they can be compared against an
    // incoming impulse at all.
    float group_mass_kg = 0.0f;
    float base_break_impulse = 0.0f;      // N.s, = velocity * mass
    float effective_break_impulse = 0.0f; // N.s, after thermal weakening
    bool integrity_weakening = true;
    float integrity_exponent = 1.5f;
    float minimum_threshold_scale = 0.15f;
    float mean_integrity = 1.0f;
    float minimum_integrity = 1.0f;
    float remaining_support_ratio = 1.0f;
    // World centre of the group's shards. Exposed so a test can tell "survived
    // because it was out of range" apart from "survived because it held" — a
    // localisation check that cannot measure distance is really a radius check
    // wearing a strength check's clothes.
    Vec3 world_center = Vec3(0.0f, 0.0f, 0.0f);
    // World AABB size of the group. The pressure bridge derives its impulse from
    // the projection of exactly this box, so when a blast lands harder on a far
    // cluster than a near one, this is the number that explains it — a finely
    // shattered region has smaller boxes and therefore catches less of the front.
    // Without it that observation can only be guessed at from one aggregate.
    Vec3 world_extent = Vec3(0.0f, 0.0f, 0.0f);
    // ★ Where the integrity numbers above came from, so a test can tell a
    // per-cluster reading apart from the whole-object average silently
    // substituted when the cluster's region held no elements. Several clusters
    // reporting the identical mean is the visible symptom of that fallback and
    // reads exactly like uniform damage without these two fields.
    bool integrity_regional = false;
    int integrity_sampled_elements = 0;
};

Result makePhysicsFractureGroup(const std::string& group,
                                const std::vector<std::string>& shard_objects,
                                float break_velocity,   // m/s, times mass = N.s
                                bool integrity_weakening,
                                float integrity_exponent,
                                float minimum_threshold_scale,
                                FractureGroupInfo& out_info,
                                // Node the shards were cut from — the one that
                                // carries the MSF field. Required whenever the
                                // group name differs from the object name (i.e.
                                // any structural cluster), or the group finds no
                                // damage and never thermally weakens.
                                const std::string& source_object = "");
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
    // ★ Now real newton-seconds: pressure x projected area x duration x coupling.
    // Fracture thresholds authored against the old area-free expression are not
    // comparable with these numbers.
    float last_max_impulse = 0.0f;
    float last_projected_area_m2 = 0.0f;
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
    // Debris mass held back by a full particle budget. Budget limits VISUAL
    // detail; it is never allowed to destroy mass, so anything unrepresented
    // waits here for the next event instead of vanishing.
    float reservoir_mass_kg = 0.0f;
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
    // ★★★ particle_count is only a MEASUREMENT when this is true.
    //
    // Measured 2026-08-16: fluid.get reported particle_count = 0 for a domain
    // that fluid.list_domains reported as 9963 at the same instant. get resolves
    // through the ACTIVE particle system only, list walks every system; when the
    // lookup missed, get fell back to the legacy FluidObject mirror, which
    // deliberately owns no stepped particle copy and is therefore always empty.
    // Zero read as "the domain is empty" instead of "I could not measure it".
    //
    // A script watching a burn would have concluded there was nothing left to
    // burn — a plausible, completely wrong observation, which is the worst kind.
    // Callers must check this before acting on particle_count.
    bool live_state = false;
    std::string render_mode; // "volume", "surface", "particles"
    std::string backend;     // "cpu", "gpu", "vulkan", "cpu_sparse"
    std::string boundary;    // "closed", "open", "periodic"
    // "water","oil","mud","honey","lava","chocolate" (liquid) |
    // "sand","wet_sand","gravel","cohesive_soil" (granular) | "custom".
    // Accepted on write as well as reported on read, "custom" included, so a
    // get -> set round trip cannot fail on a value this API produced.
    std::string preset;
    // Kinematic viscosity in m²/s. Renamed from the old unitless `viscosity`
    // together with the solver field it mirrors, so a script written against the
    // old 0..200 dial fails loudly on the missing key instead of quietly asking
    // for lava when it meant honey.
    float kinematic_viscosity = 0.0f;
    int   viscosity_sweeps = 8;
    float viscosity_wall_slip = 1.0f;  // 0 = no-slip, 1 = free-slip
    bool  granular_enabled = false;
    float granular_friction_angle_degrees = 35.0f;
    float granular_cohesion = 0.0f;
    float granular_dilatancy_degrees = 5.0f;
    float granular_young_modulus = 2.0e5f;
    float granular_poisson_ratio = 0.25f;
    float granular_tensile_cutoff = 0.0f;
    float granular_hardening = 0.0f;
    float granular_fracture_strain = 0.04f;
    float granular_damage_rate = 6.0f;
    float granular_healing_rate = 0.0f;
    bool  granular_rebonding = false;
    int   granular_max_solver_substeps = 32;
    // Thermal/burn softening of the skeleton. 0 K disables it.
    float granular_softening_temperature = 0.0f;
    float granular_softening_range = 40.0f;
    float granular_residual_strength = 0.05f;
    float granular_tack_peak = 1.0f;
    float granular_thermal_conductivity = 0.0f;
    // Material shading the SurfaceSDF isosurface; empty = built-in dielectric.
    std::string surface_material;
    // Explicit material shading splat geometry. Empty means scene default for
    // Built-in Icosphere or preserved per-face materials for a scene object.
    std::string splat_material;
    // Zero-level-set displacement in simulation voxels. This is the canonical
    // geometric fullness control; optical volume density must never be used to
    // grow or shrink a SurfaceSDF.
    float surface_offset_voxels = 0.65f;
    // Procedural porosity on the isosurface (fermented dough / aerated batter /
    // pumice). Bubbles are cut out of the FIELD before the surface is found, so
    // the pore rims are real geometry with real normals — not an alpha cutout.
    // 0 = off. Bubble size is in world units, so it survives resolution changes.
    float pore_amount = 0.0f;
    float pore_scale  = 0.05f;
    float pore_detail = 0.5f;
    // Coordinate space every isosurface pattern is addressed in.
    // 0 = Material, 1 = Domain, 2 = World. See setFluidSurface's coord_space.
    int   coord_space = 0;
    // ── Material coordinate (UVW) diagnostics ───────────────────────────────
    // Read-only. There is nothing to author here — the coordinate is carried by
    // the particles automatically — but it MUST be observable, or the only way
    // to tell a working anchor from a silent fallback to world space is to look
    // at a render and judge.
    //
    // ★ uvw_drift is the measurement that separates those two. It is the mean
    // |uvw - position| over the particles: the average distance each parcel of
    // liquid has travelled since birth, in world units.
    //   ~0            -> nothing has moved (or the coordinate is being reseeded
    //                    every frame, which is the failure this catches)
    //   grows with the pour -> the coordinate is being carried, as intended
    // A test asserts that it RISES while liquid falls. Asserting it is merely
    // non-zero would pass on a coordinate that was seeded once and then frozen.
    // ── Substance breakdown ─────────────────────────────────────────────────
    // Which substances are actually present in this domain right now, and how
    // much of each. One entry per distinct tag found on the live particles.
    //
    // ★ Reported by NAME wherever a flow source in this domain claims the tag,
    // because a hash is not something a caller can act on. Unresolvable tags
    // (liquid poured by a source that has since been renamed or deleted, or
    // handed over by a mass transfer) keep their number and an empty name —
    // "present but unnamed" is a real and different state from "absent", and
    // collapsing the two would hide exactly the case where identity outlived
    // its source.
    struct SubstanceCount {
        std::string name;        // empty when the tag resolves to nothing
        uint32_t    tag = 0u;    // 0 = untagged liquid (domain material)
        uint64_t    particles = 0;
    };
    std::vector<SubstanceCount> substances;
    // Substance -> material bindings authored on this domain. material_id -1
    // means "the built-in dielectric FOR THAT SUBSTANCE" — a real choice, not
    // an unset value.
    struct SubstanceMaterialBinding {
        std::string substance;
        int         material_id = -1;
        std::string material;    // resolved name, empty when id is -1 or stale
        std::string representation;
        // What this substance is ACTUALLY drawn as, with "inherit" already
        // resolved against the domain's own mode: "splat" or "sdf".
        //
        // ★★★ REPORTED BECAUSE "inherit" IS NOT AN ANSWER. Two knobs decide one
        // question here — the domain default and this override — and reading
        // back "inherit" tells a script only that the question was delegated,
        // not what came out of it. A test asserting on `representation` passes
        // happily while the substance is drawn the other way. The panel shows
        // the same resolution, so the script and the picture can be compared.
        // Read-only: routing is still authored through `representation`.
        std::string effective_representation; // "inherit", "splat", "sdf"
        // ★ REPORTED AS AUTHORED, including the -1 sentinel: a reader has to be
        // able to tell "inherits the domain" from "was explicitly set to the
        // same number the domain happens to have". Resolving it here would make
        // the two indistinguishable and a test could not prove the override was
        // ever written.
        float kinematic_viscosity = -1.0f;  // < 0 = inherit domain
        float miscibility = 1.0f;
        // "liquid" or "solid" — a state of MATTER, reported separately from
        // `representation` above because they are separate axes: a solid can be
        // drawn as splat spheres or reconstructed into the isosurface.
        std::string phase;
    };
    std::vector<SubstanceMaterialBinding> substance_materials;
    bool  uvw_available = false;   // the domain published a coordinate field
    int   uvw_dim[3] = { 0, 0, 0 };
    // World placement of that grid, as the SHADER will index it. Reported so a
    // script can check producer and consumer agree without a render.
    // ★ Worth reporting because the failure it catches is silent: a grid indexed
    // over the wrong region still shades, it just smears the pattern along the
    // flow — which looks like a quality limit rather than like a wiring bug.
    float uvw_origin[3] = { 0.0f, 0.0f, 0.0f };
    float uvw_voxel = 0.0f;
    // Solver steps between resets of one coordinate generation. Two run half a
    // period apart and are blended, which is what bounds the stretch.
    int   uvw_refresh_period = 240;
    float uvw_drift = 0.0f;        // mean |uvw - position|, world units
    uint64_t uvw_particles = 0;    // particles the drift was averaged over
    // ── Solid-phase measurement (last simulated step) ───────────────────────
    // Parcels carrying a substance the domain declares SOLID, and the cells
    // they actually blocked.
    //
    // ★★★ BOTH, because the pair is the diagnosis and either alone lies. Zero
    // parcels means the binding or the emitter never took. Parcels with zero
    // cells means the phase DID land and the chunk is simply thinner than the
    // voxel size can express — the fix is resolution, not the binding, and no
    // amount of re-authoring the material would ever reveal that. This is the
    // number a script asserts on; the rendered picture cannot distinguish the
    // two states at all.
    uint64_t solid_phase_particles = 0;
    uint64_t solid_phase_cells = 0;
    // ── Sealed pressure pockets (last simulated step) ───────────────────
    // Fluid regions the projection found with no pressure reference — every
    // face fluid-fluid or closed — and the cells they held. Such a region makes
    // the Poisson block singular, and a singular block does not fail loudly: the
    // pressure simply grows with the ITERATION COUNT and hurls the particles at
    // its boundary. Liquid trapped under a collider is the usual way one forms.
    //
    // ★★ A SCRIPT CAN ASSERT ON THIS AND A PICTURE CANNOT. On screen a sealed
    // pocket looks like "the solver is unstable"; only this counter says which
    // frame it appeared and how big it was. Non-zero is not automatically a bug
    // (a genuinely enclosed volume is legal, and the solver now handles it), but
    // it is always the first number to read when contact goes violent.
    uint64_t sealed_pockets = 0;
    uint64_t sealed_pocket_cells = 0;
    // False when the scan did not run at all (GPU pressure path, or the
    // non-free-surface solver). Assert on THIS before asserting on the counts,
    // or a test that never exercised the scan will report a clean bill of health.
    bool sealed_pockets_measured = false;
    // Fluid cells with liquid on all six sides (see active_fluid_cells for the
    // total). Near zero while particles are plentiful = the liquid is thinner
    // than one voxel everywhere, so there is no pressure field and no splash;
    // raise the emission rate or lower the voxel size. A script can assert on
    // this ratio; a rendered frame cannot be told apart from a viscous liquid.
    uint64_t interior_fluid_cells = 0;
    uint64_t reseed_added_particles = 0;
    uint64_t reseed_removed_particles = 0;
    uint64_t granular_yielded_particles = 0;
    uint64_t granular_detached_particles = 0;
    uint64_t granular_invalid_particles = 0;
    uint64_t granular_sleeping_particles = 0;
    uint64_t granular_damaged_particles = 0;
    uint64_t granular_damage_over_10_particles = 0;
    uint64_t granular_damage_over_50_particles = 0;
    uint64_t granular_damage_over_90_particles = 0;
    float granular_max_yield = 0.0f;
    float granular_max_plastic_increment = 0.0f;
    float granular_max_accumulated_plastic = 0.0f;
    float granular_mean_accumulated_plastic = 0.0f;
    float granular_max_fracture_history = 0.0f;
    float granular_mean_fracture_history = 0.0f;
    float granular_max_damage = 0.0f;
    float granular_mean_damage = 0.0f;
    float granular_requested_young_modulus = 0.0f;
    float granular_effective_young_modulus = 0.0f;
    int granular_required_substeps = 1;
    int granular_solver_substeps = 1;
    bool granular_stiffness_capped = false;
    // Which limit sized the subcycle, and whether the shader had to clamp the
    // deformation-gradient increment because it was not granted.
    int granular_wave_substeps = 1;
    int granular_strain_substeps = 1;
    float granular_strain_rate = 0.0f;
    uint64_t granular_strain_limited_particles = 0;
    uint64_t granular_compaction_capped_particles = 0;
    // Melt readout: how far the weakest particle has softened, and how many
    // have softened at all. below_load is judged on the SOFTENED stiffness.
    float granular_min_softening = 1.0f;
    uint64_t granular_softened_particles = 0;
    // Material validity: the load this domain puts on its own bottom layer and
    // the stiffness the small-strain model needs to carry it.
    float granular_overburden_pressure = 0.0f;
    float granular_young_modulus_for_load = 0.0f;
    bool granular_stiffness_below_load = false;
    // Domain-wide solid-phase coupling: master switch, and how full of solid
    // parcels a cell must be to block flow (fraction of the seed density).
    bool  solid_phase_enabled = true;
    float solid_phase_fill = 0.25f;
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
    // Combustion -> structure. See SimulationGridDomainDesc for why the
    // pressure scale is a CALIBRATION knob and not a physical constant.
    bool  structural_coupling_enabled = false;
    float structural_pressure_scale = 400.0f;
    float structural_min_intensity = 0.05f;
    float structural_event_interval = 0.25f;
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
    // Substance this source pours; empty = untagged (domain material).
    std::string fluid_substance;
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
    // ★ Read the header comment on MaterialMassBudgetSummary before treating
    // mass_conservation_error as a pass/fail gate. It is now measured on the RAW
    // field, so it can be non-zero — that is the whole point. It used to be
    // derived from the four clamped masses above, which made it structurally
    // incapable of reporting anything but 0.0.
    float mass_conservation_error = 0.0f;
    // The two independent ways the budget can break, reported separately so a
    // failure names itself: overflow = one kilogram spent by two processes,
    // negative = a sink ran backwards.
    float mass_budget_overflow = 0.0f;
    float mass_negative = 0.0f;
    uint32_t mass_invalid_elements = 0u;
    std::vector<std::string> semantics;
};

// Read-only Phase-0 field-contract view used by scripts and diagnostics.
Result listMaterialFields(std::vector<MaterialFieldInfo>& out_fields);

struct MoltenMassTransferInfo {
    uint64_t queued = 0, completed = 0;
    uint64_t deferred_no_domain = 0, deferred_no_capacity = 0;
    // Requests that ended without transferring and without being retried, and
    // requests thrown away because the simulation was reset or a cached frame
    // restored. Both used to be invisible.
    uint64_t dropped = 0, discarded_on_reset = 0;
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
// ★★★ seed_min/seed_max are POINTERS, and null means "derive from the domain".
//
// Both channels used to substitute the same hardcoded box, (-0.5, 1.0, -0.5) to
// (0.5, 1.5, 0.5), when the caller omitted the region. That box is not derived
// from anything: any domain that does not happen to contain it seeds nothing.
// Combined with the empty-overlap silence this produced a call that succeeded,
// created no particles, and left the caller to blame the solver (measured
// 2026-08-19).
//
// Null now fills the bottom half of the domain, inset by one voxel, which is
// what "seed this tank" means and is correct for every domain rather than for
// one. Passing a region explicitly is unchanged, and a region that does not
// overlap the domain is now REFUSED with both boxes in the message.
Result seedFluidParticles(const std::string& domain_id_or_name,
                           const Vec3* seed_min, const Vec3* seed_max,
                           int particles_per_cell = 4, bool replace = true,
                           bool persistent = false);
Result clearFluidParticles(const std::string& domain_id_or_name,
                           bool clear_seed_recipe = false);
// Assign by material NAME so scripts never depend on unstable registry ids.
// Empty clears the override (scene default / source-object face materials).
Result setFluidSplatMaterial(const std::string& domain_id_or_name,
                             const std::string& material_name);
Result updateFluidDomain(const std::string& domain_id_or_name,
                         const Vec3* domain_min = nullptr, const Vec3* domain_max = nullptr,
                         const float* voxel_size = nullptr, const std::string* render_mode = nullptr,
                         const std::string* backend = nullptr, const std::string* boundary = nullptr,
                         const std::string* preset = nullptr,
                         const float* kinematic_viscosity = nullptr,
                         const int* viscosity_sweeps = nullptr,
                         const float* viscosity_wall_slip = nullptr,
                         // Scene material shading the SurfaceSDF isosurface, BY NAME.
                         // Empty string clears it back to the built-in dielectric.
                         // A name and not an index: material ids shift as the scene
                         // is edited, so an id in a script is a number nobody can
                         // check, and pointing at the wrong material is silent.
                         const std::string* surface_material = nullptr,
                         // Geometric SurfaceSDF dilation in simulation voxels.
                         // Bounded to [-0.75, 1.25]; constant reconstruction cost.
                         const float* surface_offset_voxels = nullptr,
                         // Procedural porosity — see FluidDomainInfo::pore_*.
                         // On the DOMAIN because a pore is a property of the
                         // SUBSTANCE, and because the gas/liquid handoff arbiter
                         // evaluates the same field: the two must agree term for
                         // term or smoke is clipped against a surface the shader
                         // never draws.
                         const float* pore_amount = nullptr,
                         const float* pore_scale = nullptr,
                         const float* pore_detail = nullptr,
                         // Coordinate space every isosurface pattern is
                         // addressed in — textures, resin interior, porosity and
                         // the opacity mask, all through one anchor.
                         // 0 = Material (carried by the liquid, default and
                         //     identical to World until something moves),
                         // 1 = Domain (carried by the container),
                         // 2 = World (fixed to the scene).
                         // Out-of-range values are clamped, not rejected: this
                         // is a look control and refusing a script mid-sequence
                         // over a typo'd enum is worse than snapping it.
                         const int* coord_space = nullptr,
                         // Solver steps between resets of one material-coordinate
                         // generation. Clamped to >= 2: below that the two
                         // generations reset on the same step and stop being two.
                         const int* uvw_refresh_period = nullptr,
                         // Solid-phase coupling for this domain.
                         //   solid_phase       — master switch. Off leaves every
                         //     substance's authored phase intact and simply
                         //     stamps nothing, so it is the one control that
                         //     answers "is the solid causing this?" without
                         //     destroying the setup being tested.
                         //   solid_phase_fill  — cell fill fraction (of the seed
                         //     density) required to block. Too low dams a
                         //     channel with one stray parcel; too high blocks
                         //     nothing while the panel still says Solid.
                         const bool* solid_phase = nullptr,
                         const float* solid_phase_fill = nullptr,
                         const bool* enabled = nullptr, const bool* visible = nullptr,
                         const bool* granular_enabled = nullptr,
                         const float* granular_friction_angle_degrees = nullptr,
                         const float* granular_cohesion = nullptr,
                         const float* granular_dilatancy_degrees = nullptr,
                         const float* granular_young_modulus = nullptr,
                         const float* granular_poisson_ratio = nullptr,
                         const float* granular_tensile_cutoff = nullptr,
                         const float* granular_hardening = nullptr,
                         const float* granular_fracture_strain = nullptr,
                         const float* granular_damage_rate = nullptr,
                         const float* granular_healing_rate = nullptr,
                         const bool* granular_rebonding = nullptr,
                         const int* granular_max_solver_substeps = nullptr,
                         const float* granular_softening_temperature = nullptr,
                         const float* granular_softening_range = nullptr,
                         const float* granular_residual_strength = nullptr,
                         const float* granular_tack_peak = nullptr,
                         const float* granular_thermal_conductivity = nullptr);
// Bind a material to a SUBSTANCE within one fluid domain. An empty
// `material_name` clears the binding; the literal "dielectric" binds the
// built-in refractive liquid for that substance specifically, which is how one
// substance can be a full Principled BSDF while another stays plain water in
// the same body.
//
// ★ Keyed by substance NAME rather than by emitter: at a surface point where two
// streams have met there is no "which emitter", but there is a mixture of
// substances — so this is the only key the shading question can be answered
// with, and two emitters pouring the same substance are one body.
//
// ★★ Also carries the substance's PHYSICS, because a substance is one authored
// thing. Splitting "what it looks like" from "how it flows" across two calls is
// what let the panel imply that mixing was a shading choice. Both are optional:
// a null pointer leaves the current value, it does not reset it.
//   kinematic_viscosity — ABSOLUTE m^2/s. Negative means inherit the domain's.
//   miscibility         — 1 fully miscible (soft gradient), 0 immiscible
//                         (sharp front). The pair uses the MINIMUM of its two.
//   phase               — "liquid" (default) or "solid". A solid substance's
//                         parcels are rasterized into the grid's solid mask
//                         every step, so the liquid flows around them and
//                         clings to them.
//
// ★★★ `phase` IS NOT `representation`. Representation says how a parcel is
// DRAWN; phase says what it IS. They are kept apart so a scene's flow can
// never change because somebody switched a display mode — the same rule the
// substance viscosity gather is built on.
//
// ★ A solid here is an OBSTACLE with mass and velocity, not a rigid body: the
// parcels have no cohesion, so a pile spreads under load. Dragged cohesive
// clusters belong to Jolt rather than to a second rigid solver in the fluid
// step, and saying so is more useful to a caller than a number that quietly
// under-delivers.
Result setFluidSubstanceMaterial(const std::string& domain_id_or_name,
                                 const std::string& substance,
                                 const std::string& material_name,
                                 const std::string* representation = nullptr,
                                 const float* kinematic_viscosity = nullptr,
                                 const float* miscibility = nullptr,
                                 const std::string* phase = nullptr);

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
    // FIELD grid: heights and every analysis product live here.
    int width = 0;
    int height = 0;
    float size = 0.0f;
    float height_scale = 0.0f;
    // MESH grid: the vertex/triangle resolution, which is a separate decision.
    // mesh_resolution == 0 means "follow the field" (the historical behaviour);
    // mesh_width/mesh_height report the grid that was actually built, so a
    // caller never has to re-derive the 0 case.
    int mesh_resolution = 0;
    int mesh_width = 0;
    int mesh_height = 0;
    // PAINT grid: the splat map and macro color map resolution.
    int paint_resolution = 0;
    int paint_width = 0;
    int paint_height = 0;
    bool has_surface_semantic = false;
    bool has_node_graph = false;
    bool dirty = false;
};

// ★ Vertex-grid resolution, independent of the field. 0 restores "same as the
// field". Values above the field resolution are clamped down to it: a mesh
// denser than the data it samples invents detail it cannot have.
Result setTerrainMeshResolution(const std::string& terrain_name, int mesh_resolution,
                                TerrainInfo& out_info);

// ★ Paint-grid resolution, independent of the field. 0 restores "same as the
// field". This dictates the size of splatMap and macroColorMap.
Result setTerrainPaintResolution(const std::string& terrain_name, int paint_resolution,
                                 TerrainInfo& out_info);

Result listTerrains(std::vector<TerrainInfo>& out_terrains);
Result getTerrain(const std::string& terrain_name, TerrainInfo& out_info);
// `resolution` is the FIELD grid; `mesh_resolution` is the vertex grid (0 =
// follow the field). Both are creation parameters because decimating after the
// fact still pays one full-resolution acceleration-structure build.
Result createTerrain(const std::string& requested_name, int resolution, float size,
                     float height_scale, int mesh_resolution, TerrainInfo& out_info);
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
                          bool replace_graph = false, bool add_satmap = false);
struct TerrainSatMapPresetInfo {
    std::string id;
    std::string label;
    std::string category;
    std::string description;
    int version = 1;
    int layer_count = 0;
};
Result listTerrainSatMapPresets(std::vector<TerrainSatMapPresetInfo>& out_presets);
Result applyTerrainSatMapPreset(const std::string& terrain_name, const std::string& preset_id,
                                std::vector<std::string>& out_warnings);
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
