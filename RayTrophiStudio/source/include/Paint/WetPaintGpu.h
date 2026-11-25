#pragma once

// GPU wet paint over many surfaces (docs/dev/PLAYER_GAME_PAINT_ROOM.md step 3,
// section 5.3). The model is MeshPaintAdapter::noteWetDab / tickWetPaint carried
// to shaders/paint_wet.comp; this class owns the atlas, feeds it and publishes
// the result into each surface's base colour texture.
//
// Sources of paint, applied on the device every tick:
//   * host dabs (queueDab: IPC paint.wet_dab, game rules, hands/feet later);
//   * particle collision events, read straight from the particle systems'
//     device buffers on the SAME compute context (no download). A system on
//     the CPU reference hands its host events over instead (counted).
//
// Output: the composite (original base colour + paint layer) of every touched
// tile is copied device-to-device into the surface's base colour VkImage when
// the render backend shares the compute VkDevice; otherwise it is downloaded
// into the texture's host pixels and uploaded the normal way (counted, slower).
//
// Paint is session state: it is not saved with the project (a play room starts
// clean) and the CPU-side pixels of the bound texture are NOT kept in sync on
// the device-copy path. A CPU paint operation on the same target (Studio brush,
// paint.fill_layer...) composites from its own layers and overwrites it.

#include <cstdint>
#include <memory>
#include <array>
#include <string>
#include <unordered_map>
#include <vector>

#include "SimulationCompute.h"

struct SceneData;
class Texture;
namespace Backend { class IBackend; }

namespace Paint {

// The BrushSettings wet subset (IPaintSurfaceAdapter.h), same names and
// defaults; tickWetPaint's derived terms are formed from these on upload.
// Exception: lifetime is 12 s here (2.5 s on the brush) so splats stay wet long
// enough for the runoff to be seen.
struct WetPaintSurfaceSettings {
    float wetness = 0.65f;
    float lifetime_seconds = 12.0f;
    float diffusion = 0.24f;
    float runoff = 0.35f;
    float absorption = 0.18f;
    float drip_head = 0.45f;
    float falloff = 0.65f;
    float paint_load = 0.85f;
    float deposit_rate = 0.65f;
    float pickup_rate = 0.35f;
    // brush.radius of the CPU model (pixels at 1024 reference resolution):
    // scales the transport trail length exactly like tickWetPaint does.
    float brush_radius = 5.0f;
    bool oil = false;
    // Optional outputs reuse PaintTextureSet Roughness / Mask(height) / Normal / Opacity.
    bool roughness_channel = false;
    bool height_normal_channel = false;
    // Owns opacity on solid geometry: paint coverage reveals it; reset hides it.
    bool reveal_opacity = false;
    float wet_roughness = 0.12f;
    float dry_roughness = 0.30f;
    float height_contribution = 0.35f;
    float normal_strength = 0.5f;
    // Hidden picture (2026-11-24): an image laid over the surface's UV page.
    // Transparent pixels show the wall; dark/unsaturated opaque pixels
    // (outlines) are always visible; saturated pixels are colourable regions,
    // hidden until paint of a matching HUE (within picture_tolerance degrees,
    // mixed paint counts) lands on them - then that texel shows the picture,
    // permanently until the room resets. Empty path = off.
    std::string reveal_picture;
    float picture_tolerance = 30.0f;   // degrees of hue, 5..90
    float picture_hint = 0.0f;         // 0..1 faint preview of hidden regions
    // Opacity of the "always visible" picture pixels (outlines, white/grey)
    // while NOT revealed; 0 = the whole picture is invisible. Any paint reveals
    // them (no hue test).
    float picture_outline = 1.0f;
    // reveal_picture == "@base": the surface's OWN base texture is the hidden
    // picture and the visible wall is this flat sRGB colour.
    float picture_blank[3] = {1.0f, 1.0f, 1.0f};
};

// How particle contacts become paint.
struct WetPaintSplatSettings {
    bool consume_particles = true;
    // Empty: every particle system that records collision events.
    std::string particle_system;
    float radius = 0.12f;          // world metres
    float amount = 0.8f;           // the dab strength of one contact (0..1)
    float min_speed = 0.0f;        // m/s; slower contacts leave nothing
    float facing_min = 0.0f;       // dot(texel normal, contact normal) floor
    uint32_t max_events_per_tick = 4096u;
};

struct WetPaintDab {
    float position[3] = {0.0f, 0.0f, 0.0f};
    float radius = 0.1f;           // world metres
    float color[3] = {1.0f, 0.2f, 0.2f};   // display-referred (sRGB) 0..1
    float amount = 0.8f;           // strength 0..1
    float normal[3] = {0.0f, 1.0f, 0.0f};
    bool facing = false;           // only texels facing `normal`
};

struct WetPaintSurfaceInfo {
    std::string object;
    int material_id = 0;
    int resolution = 0;
    uint32_t tiles = 0;
    uint32_t valid_texels = 0;     // covered by a triangle (gutter excluded)
    uint32_t gutter_texels = 0;
    uint32_t triangles = 0;
    uint32_t active_tiles = 0;     // last tick
    uint32_t ticked_tiles = 0;     // last tick (active + neighbours)
    uint32_t stamped_texels = 0;   // last tick
    uint32_t sources_dropped = 0;  // last tick: > 64 sources in one tile
    std::string texture;
    // Canvas over a tiled base (docs/dev/WET_PAINT_CANVAS_UV.md): chosen by the
    // material itself (a base colour texture under the canvas and a non-identity
    // UV transform). base_texture is that texture's name, "" in composite mode.
    bool canvas_overlay = false;
    std::string base_texture;
    WetPaintSurfaceSettings settings;
};

struct WetPaintStats {
    bool device_ready = false;
    std::string status = "idle";
    uint64_t ticks = 0;
    float last_dt = 0.0f;
    uint32_t surfaces = 0;
    uint32_t tiles = 0;
    uint64_t dabs_applied = 0;
    uint64_t device_event_dispatches = 0;   // stamp dispatches on particle buffers
    uint64_t host_events_applied = 0;       // CPU-reference contacts sent as dabs
    uint32_t active_tiles = 0;              // last tick, all surfaces
    uint32_t ticked_tiles = 0;
    uint32_t stamped_texels = 0;
    uint64_t sources_dropped = 0;
    uint64_t pbr_dispatches = 0;
    uint64_t pbr_device_copies = 0;
    uint64_t pbr_buffer_bytes = 0;     // additional GPU atlas storage, not total VRAM
    uint64_t pbr_texture_bytes = 0;    // allocated channel texture pixels (RGBA8)
    uint64_t publish_device_copies = 0;     // dirty rectangles copied device-to-device
    uint64_t publish_host_downloads = 0;    // dirty rectangles that went through the host
    uint64_t publish_bytes_downloaded = 0;
    uint64_t device_rebuilds = 0;
    uint64_t geometry_uploads = 0;
    uint64_t geometry_rebakes = 0;          // surfaces re-baked after their object moved/scaled
    std::string publish_mode = "none";      // device_copy | host_download | none
    // simulation: the simulation context (a GPU particle system keeps it on
    // Vulkan; contacts stamped from its device buffers). own: wet paint's own
    // Vulkan context on the same device (no device particle events there).
    std::string compute_context = "none";
    double last_tick_ms = 0.0;              // host wall time of tick() incl. stats readback
    uint64_t idle_ticks = 0;                // ticks that dispatched nothing (nothing wet or arriving)
    uint64_t coverage_cached = 0;           // coverage() answered from cache (paint unchanged)
    uint32_t event_sources = 0;             // last tick: particle systems whose contacts were stamped
    uint64_t coverage_in_tick = 0;          // coverage measured inside tick()'s own readback
    uint64_t stats_delayed = 0;             // dispatching ticks whose stats were read one tick later
    uint64_t stats_readbacks_lost = 0;      // delayed stats that never arrived (dirty rectangles lost)
};

struct WetPaintCoverage {
    struct Surface {
        std::string object;
        int material_id = 0;
        uint32_t covered_texels = 0;
        uint32_t valid_texels = 0;
        uint32_t wet_texels = 0;
        uint32_t roughness_texels = 0; // output differs from original channel
        uint32_t normal_texels = 0;
        float ratio = 0.0f;
        float mean_color[3] = {0.0f, 0.0f, 0.0f};   // sRGB 0..1 of covered texels
        // Hidden picture progress (reveal_picture): colourable texels and how
        // many of them are revealed. 0/0 when the surface has no picture.
        uint32_t picture_texels = 0;
        uint32_t picture_revealed_texels = 0;
        float picture_ratio = 0.0f;
    };
    std::vector<Surface> surfaces;
    uint64_t covered_texels = 0;
    uint64_t valid_texels = 0;
    float ratio = 0.0f;
};

class WetPaintGpu {
public:
    WetPaintGpu();
    ~WetPaintGpu();
    WetPaintGpu(const WetPaintGpu&) = delete;
    WetPaintGpu& operator=(const WetPaintGpu&) = delete;

    // Makes (object, material_id) a paint surface. Ensures the mesh paint
    // target (base colour texture bound to the material, like paint.ensure_target)
    // and bakes the surface's texels once. material_id < 0: the object's first.
    bool addSurface(SceneData& scene, const std::string& object, int material_id,
                    int resolution, std::string& error);
    // With a scene, the object's material gets its original textures back
    // (the paint textures addSurface bound are released). Without one the
    // paint textures stay bound - only for callers that rebuild the scene.
    bool removeSurface(const std::string& object, int material_id, SceneData* scene = nullptr);
    // Drops every surface and releases the device atlas (project change).
    void clear();

    bool setSurfaceSettings(const std::string& object, int material_id,
                            const WetPaintSurfaceSettings& settings);
    void setSplatSettings(const WetPaintSplatSettings& settings);
    const WetPaintSplatSettings& splatSettings() const { return splat_; }

    void queueDab(const WetPaintDab& dab);
    // Clears all paint (layer, wetness) back to the original base colour.
    bool resetPaint();
    void checkGeometryNextTick() { geometry_check_clock_ = 0.5f; }

    // Stamp + tick on the simulation compute context. Called once per frame
    // by the shared simulation clock (Studio and Player).
    bool tick(SceneData& scene, float dt);
    // Copies the dirty rectangles into the bound textures of each backend.
    // Returns kWetPaintPublished* flags (0 = nothing changed). With
    // kWetPaintPublishedHost the texture's host pixels changed and the CALLER
    // runs the brush's per-material sync for hostPublishedMaterials()
    // (Renderer::updateBackendMaterial: that one slot + the dirty region, both
    // backends) - g_materials_dirty alone is not enough in Studio, where that
    // flag is serviced only inside a full scene sync.
    // GPU-only callers pass their same-device target in its usual slot.
    // Unpublished rectangles remain queued for a later compatible target.
    int publish(Backend::IBackend* render_backend, Backend::IBackend* viewport_backend,
                bool allow_host_copy = true);
    void* publicationDevice() const;
    const std::vector<uint16_t>& hostPublishedMaterials() const { return host_materials_; }

    // Painted share of every surface. Exact by default: a dispatch + synchronize
    // + download (the device queue is shared with rendering, so that wait
    // includes the frame in flight). allow_stale (periodic readers: room goal,
    // game session): while the paint is changing, answer the last measurement
    // and let the next tick() measure inside its own readback - no extra wait.
    // Unchanged paint is answered from cache either way.
    bool coverage(SceneData& scene, WetPaintCoverage& out, bool allow_stale = false);

    bool empty() const { return surfaces_.empty(); }
    std::vector<WetPaintSurfaceInfo> surfaces() const;
    // Returns the clean base colour texture that should be written on project
    // save for `material_id` (the real source texture when one exists, or an
    // un-blanked / un-painted copy of the surface's original base pixels so
    // saving while "@base" or wet paint is active never white-washes the wall).
    std::shared_ptr<Texture> cleanBaseTextureForSave(uint16_t material_id) const;
    const WetPaintStats& stats() const { return stats_; }
    RayTrophiSim::SimulationComputeContext* fallbackCompute(void* activeDevice) const {
        return activeDevice == own_compute_device_ ? own_compute_.get() : nullptr;
    }

private:
    struct Surface;
    struct Device;

    Surface* find(const std::string& object, int material_id);
    bool rebuildDevice(SceneData& scene);
    bool patchGeometry();
    void uploadSurfaceTable();
    bool ensureCompute(SceneData& scene);
    bool prepareChannels(SceneData& scene);
    bool bakeGeometry(SceneData& scene, Surface& s, std::string& error);
    void setCanvasOverlay(Surface& s, bool overlay);
    void updateCanvasModes();
    static std::array<double, 24> geometryKey(SceneData& scene, const std::string& object);
    bool channels_dirty_ = false;
    float geometry_check_clock_ = 0.0f;

    std::vector<std::unique_ptr<Surface>> surfaces_;
    std::unique_ptr<Device> device_;
    std::unique_ptr<RayTrophiSim::SimulationComputeContext> own_compute_;
    void* own_compute_device_ = nullptr;
    std::vector<WetPaintDab> pending_dabs_;
    WetPaintSplatSettings splat_;
    WetPaintStats stats_;
    std::vector<uint16_t> host_materials_;   // last publish, host path
    bool reset_requested_ = false;
    bool layout_dirty_ = true;
    bool settings_dirty_ = true;
    // Contact sources alive at the last tick (key: the particle system). A
    // system with no live particle now or then has no contact pending.
    std::unordered_map<const void*, bool> source_alive_;
    uint64_t paint_serial_ = 0;          // ticks that dispatched: the paint may have changed
    uint64_t coverage_serial_ = ~0ull;   // paint_serial_ of coverage_cache_
    WetPaintCoverage coverage_cache_;
    bool coverage_requested_ = false;    // a stale answer was given: tick() measures
    void readCoverage(const std::vector<uint32_t>& st, WetPaintCoverage& out) const;
    void applyStats(const std::vector<uint32_t>& st, bool force, bool measured_coverage, uint64_t serial);
};

inline constexpr int kWetPaintPublishedDevice = 1;   // device-to-device copy done
inline constexpr int kWetPaintPublishedHost = 2;     // host pixels changed: material sync needed

// Process-wide instance (one play room / one scene at a time).
WetPaintGpu& wetPaintGpu();

} // namespace Paint
