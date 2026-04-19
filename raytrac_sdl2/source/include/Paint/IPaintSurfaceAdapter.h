#pragma once

#include <memory>
#include <string>
#include "Vec3.h"
#include "Paint/PaintLayer.h"
#include "Paint/PaintSurfaceTarget.h"

class Texture;

namespace Paint {

enum class BrushAlphaPreset : unsigned char {
    SoftRound = 0,
    HardRound,
    Noise,
    Scratch,
    Cloud
};

enum class BrushPaintMode : unsigned char {
    Normal = 0,
    Mix,
    Smudge,
    Wet
};

enum class BrushTool : unsigned char {
    Paint = 0,
    Erase,
    Soften,
    Stamp,
    Fill,
    Clone,
    Spray
};

enum class StampPlacementMode : unsigned char {
    Single = 0,
    Continuous
};

enum class PaintTextureTintMode : unsigned char {
    Multiply = 0,
    Recolor,
    Overlay
};

struct BrushSettings {
    float radius = 5.0f;
    float strength = 0.5f;
    float falloff = 0.65f;
    float spacing = 0.15f;
    float flow = 1.0f;
    BrushTool tool = BrushTool::Paint;
    BrushAlphaPreset alpha_preset = BrushAlphaPreset::SoftRound;
    BrushPaintMode paint_mode = BrushPaintMode::Normal;
    float alpha_scale = 1.0f;
    float alpha_rotation_degrees = 0.0f;
    bool follow_stroke_angle = false;
    StampPlacementMode stamp_mode = StampPlacementMode::Single;
    bool stamp_random_rotation = false;
    float stamp_scale_jitter = 0.0f;
    float scatter_jitter = 0.0f;
    int spray_particles = 12;
    float spray_spread = 0.65f;
    float spray_droplet_size = 0.25f;
    float spray_size_jitter = 0.25f;
    float spray_opacity_jitter = 0.2f;
    bool mirror_x = false;
    bool mirror_y = false;
    bool mirror_z = false;
    float mix_amount = 0.45f;
    float smudge_strength = 0.75f;
    float wetness = 0.65f;
    float paint_load = 0.85f;
    float pickup_rate = 0.35f;
    float deposit_rate = 0.65f;
    bool write_height_mask = false;
    float height_contribution = 0.35f;
    std::shared_ptr<Texture> alpha_texture;
    std::string alpha_texture_path;
    bool use_imported_alpha = false;
    std::shared_ptr<Texture> paint_texture;
    std::string paint_texture_path;
    bool use_paint_texture = false;
    Vec3 color = Vec3(1.0f, 1.0f, 1.0f);
    float paint_texture_tint_strength = 0.6f;
    PaintTextureTintMode paint_texture_tint_mode = PaintTextureTintMode::Recolor;
    bool show_preview = true;
};

struct PaintStrokeContext {
    int layer_index = 0;
    float dt = 1.0f / 60.0f;
};

struct PaintAdapterCapabilities {
    bool supports_layers = false;
    bool supports_texture_set = false;
    bool supports_vertex_color = false;
    bool supports_material_channels = false;
    bool supports_terrain_channels = false;
};

class IPaintSurfaceAdapter {
public:
    virtual ~IPaintSurfaceAdapter() = default;

    virtual SurfaceType getSurfaceType() const = 0;
    virtual PaintSurfaceTarget getTarget() const = 0;
    virtual PaintAdapterCapabilities getCapabilities() const = 0;
    virtual bool isValid() const = 0;

    virtual int getLayerCount() const = 0;
    virtual std::string getLayerName(int index) const = 0;

    virtual bool beginStroke(const PaintStrokeContext& ctx) = 0;
    virtual bool applyDab(const Vec3& world_hit_point, const BrushSettings& brush, const PaintStrokeContext& ctx) = 0;
    virtual void endStroke() = 0;

    // Drop any persistent layer-stack state this adapter keeps in its scene.
    // Called when the user exits paint mode so stale data does not resurrect
    // on re-entry (e.g. after Fill + Add Layer). Default no-op.
    virtual void releaseLayerStackFromScene() {}
};

using PaintSurfaceAdapterPtr = std::shared_ptr<IPaintSurfaceAdapter>;

} // namespace Paint
