#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rtapi {
struct Result;

// GPU wet paint (docs/dev/PLAYER_GAME_PAINT_ROOM.md step 3/4). A domain header
// so the shared RtApi.h does not change. Values only; the atlas lives in
// Paint::WetPaintGpu.

struct WetPaintSurfaceSettingsInfo {
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
    float brush_radius = 5.0f;   // CPU brush.radius (px at 1024): transport trail scale
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
    // Hidden picture: image path laid over the UV page (empty = off); paint of
    // matching hue (within picture_tolerance degrees, 5..90) reveals it.
    std::string reveal_picture;
    float picture_tolerance = 30.0f;
    float picture_hint = 0.0f;     // 0..1
    float picture_outline = 1.0f;  // 0..1 opacity of always-visible picture pixels until revealed
    float picture_blank[3] = {1.0f, 1.0f, 1.0f};  // sRGB wall colour, used when reveal_picture == "@base"
};

Result validateWetPaintSurfaceSettings(const WetPaintSurfaceSettingsInfo& settings);

struct WetPaintSurfaceRuntimeInfo {
    std::string object;
    int material_id = 0;
    int resolution = 0;
    uint32_t tiles = 0;
    uint32_t valid_texels = 0;
    uint32_t gutter_texels = 0;
    uint32_t triangles = 0;
    uint32_t active_tiles = 0;
    uint32_t ticked_tiles = 0;
    uint32_t stamped_texels = 0;
    uint32_t sources_dropped = 0;
    std::string texture;
    // Canvas over a tiled base (docs/dev/WET_PAINT_CANVAS_UV.md): true when the
    // material keeps a UV transform for its base colour texture (base_texture)
    // and the canvas is composited over it at the raw UV.
    bool canvas_overlay = false;
    std::string base_texture;
    WetPaintSurfaceSettingsInfo settings;
};

struct WetPaintSplatInfo {
    bool consume_particles = true;
    std::string particle_system;   // empty: every recording system
    float radius = 0.12f;
    float amount = 0.8f;
    float min_speed = 0.0f;
    float facing_min = 0.0f;
    uint32_t max_events_per_tick = 4096u;
};

struct WetPaintStatusInfo {
    bool device_ready = false;
    std::string status;
    uint64_t ticks = 0;
    float last_dt = 0.0f;
    uint32_t surfaces = 0;
    uint32_t tiles = 0;
    uint64_t dabs_applied = 0;
    uint64_t device_event_dispatches = 0;
    uint64_t host_events_applied = 0;
    uint32_t active_tiles = 0;
    uint32_t ticked_tiles = 0;
    uint32_t stamped_texels = 0;
    uint64_t sources_dropped = 0;
    uint64_t pbr_dispatches = 0;
    uint64_t pbr_device_copies = 0;
    uint64_t pbr_buffer_bytes = 0;     // additional GPU atlas storage, not total VRAM
    uint64_t pbr_texture_bytes = 0;    // allocated channel texture pixels (RGBA8)
    uint64_t publish_device_copies = 0;
    uint64_t publish_host_downloads = 0;
    uint64_t publish_bytes_downloaded = 0;
    uint64_t device_rebuilds = 0;
    uint64_t geometry_uploads = 0;
    uint64_t geometry_rebakes = 0;
    std::string publish_mode;
    std::string compute_context;   // simulation | own | none
    double last_tick_ms = 0.0;
    uint64_t idle_ticks = 0;
    uint64_t coverage_cached = 0;
    uint32_t event_sources = 0;
    uint64_t coverage_in_tick = 0;
    uint64_t stats_delayed = 0;
    uint64_t stats_readbacks_lost = 0;
    std::vector<WetPaintSurfaceRuntimeInfo> surface_list;
    WetPaintSplatInfo splat;
};

struct WetPaintDabInfo {
    float position[3] = {0.0f, 0.0f, 0.0f};
    float radius = 0.1f;
    float color[3] = {1.0f, 0.2f, 0.2f};   // sRGB 0..1
    float amount = 0.8f;
    float normal[3] = {0.0f, 1.0f, 0.0f};
    bool facing = false;
};

struct WetPaintCoverageInfo {
    struct Surface {
        std::string object;
        int material_id = 0;
        uint32_t covered_texels = 0;
        uint32_t valid_texels = 0;
        uint32_t wet_texels = 0;
        uint32_t roughness_texels = 0; // output differs from original channel
        uint32_t normal_texels = 0;
        uint32_t picture_texels = 0;           // colourable picture texels (0 = no picture)
        uint32_t picture_revealed_texels = 0;
        float picture_ratio = 0.0f;
        float ratio = 0.0f;
        float mean_color[3] = {0.0f, 0.0f, 0.0f};
    };
    std::vector<Surface> surfaces;
    uint64_t covered_texels = 0;
    uint64_t valid_texels = 0;
    float ratio = 0.0f;
};

// material_id < 0: the object's first material. resolution 32..4096.
Result addWetPaintSurface(const std::string& object, int material_id, int resolution,
                          WetPaintSurfaceRuntimeInfo& out);
Result removeWetPaintSurface(const std::string& object, int material_id);
Result clearWetPaintSurfaces();
Result resizeWetPaintSurfaces(int resolution);
Result getWetPaintStatus(WetPaintStatusInfo& out);
Result getWetPaintSurfaceSettings(const std::string& object, int material_id,
                                  WetPaintSurfaceSettingsInfo& out);
Result setWetPaintSurfaceSettings(const std::string& object, int material_id,
                                  const WetPaintSurfaceSettingsInfo& settings);
Result getWetPaintSplat(WetPaintSplatInfo& out);
Result setWetPaintSplat(const WetPaintSplatInfo& settings);
// Queued; applied on the device at the next tick.
Result wetPaintDab(const WetPaintDabInfo& dab);
Result resetWetPaint();
// GPU reduction + a small download: a measurement call.
Result getWetPaintCoverage(WetPaintCoverageInfo& out);

// JSON values are owned by the shared demo core; both bindings use this entry.
Result wetPaintDemoCommand(const std::string& operation, const std::string& config_json,
                           std::string& output_json);

// The project's game design (Paint/GameDesign.h): design_json is exactly what
// the next project save writes under "game_design" ("null" when the scene has
// no paint surface and no demo). pending: a just-opened project's document
// waits for the next simulation clock tick. status: last apply report.
struct GameDesignInfo {
    std::string design_json;
    bool pending = false;
    std::string status;
};
Result getGameDesign(GameDesignInfo& out);

} // namespace rtapi
