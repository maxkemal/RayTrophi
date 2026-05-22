#pragma once

#include "Vec3.h"

#include <array>
#include <string>
#include <vector>

namespace Stylize {

enum class StrokeDirectionMode {
    SurfaceNormal = 0,
    Vertical,
    Horizontal,
    Diagonal,
    CrossHatch
};

enum class StyleProfileId {
    PainterlyOil = 0,
    Gouache,
    InkWash,
    GraphicToon,
    ClayMaquette,
    DreamySunset
};

enum class OutlineLineType {
    Ink = 0,
    OilPaint,
    Pencil,
    DryBrush,
    Pressure
};

enum class OutlineColorMode {
    PaletteShadow = 0,
    CustomColor,
    MaterialTint,
    WarmPaint,
    CoolPencil
};

enum class SkyStylePreset {
    PainterlyClouds = 0,
    CartoonCel,
    SunsetBands,
    InkWash,
    ClearGradient
};

struct StylizedSkySettings {
    bool enabled = true;
    SkyStylePreset style = SkyStylePreset::PainterlyClouds;
    Vec3 horizon_color = Vec3(0.74f, 0.64f, 0.48f);
    Vec3 zenith_color = Vec3(0.30f, 0.45f, 0.72f);
    Vec3 sun_glow_color = Vec3(1.0f, 0.72f, 0.38f);
    float gradient_strength = 0.85f;
    float cloud_brush_scale = 1.8f;
    float cloud_brush_strength = 0.55f;
    float wind_smear = 0.25f;
    float horizon_haze = 0.35f;
    float sun_disc_scale = 7.0f;
    float cloud_roundness = 0.72f;
};

struct PainterlyMaterialSettings {
    bool enabled = true;
    StrokeDirectionMode stroke_direction = StrokeDirectionMode::SurfaceNormal;
    float brush_strength = 0.45f;
    float brush_scale = 1.0f;
    float pigment_thickness = 0.35f;
    float dry_brush = 0.2f;
    bool wet_oil_model = false;
    float oil_body = 0.62f;
    float paint_load = 0.85f;
    float pickup_rate = 0.18f;
    float deposit_rate = 0.82f;
    float bristle_buildup = 0.58f;
    float surface_adherence = 0.68f;
    float depth_scale_response = 0.45f;
    float edge_respect = 0.72f;
    float palette_influence = 0.6f;
    float material_color_preservation = 0.65f;
    float color_simplification = 0.35f;
    float roughness_bias = 0.2f;
    float normal_softening = 0.15f;
};

struct StylizedOutlineSettings {
    bool enabled = true;
    OutlineLineType line_type = OutlineLineType::Ink;
    OutlineColorMode color_mode = OutlineColorMode::PaletteShadow;
    Vec3 custom_color = Vec3(0.05f, 0.045f, 0.04f);
    float strength = 0.35f;
    float width = 1.25f;
    float depth_sensitivity = 0.45f;
    float normal_sensitivity = 0.55f;
    float taper = 0.35f;
    float break_up = 0.18f;
    float color_bleed = 0.25f;
    float distance_thinning = 0.55f;
    float detail_protection = 0.45f;
};

struct StylizedWorldAdapterSettings {
    float terrain_stroke_blend = 0.35f;
    float foliage_cluster_simplification = 0.3f;
    float foliage_palette_variance = 0.2f;
    float volume_grain = 0.15f;
    float force_field_motion_response = 0.25f;
};

struct StyleProfile {
    StyleProfileId id = StyleProfileId::PainterlyOil;
    const char* name = "Painterly Oil";
    Vec3 palette_shadow = Vec3(0.20f, 0.16f, 0.13f);
    Vec3 palette_mid = Vec3(0.62f, 0.48f, 0.34f);
    Vec3 palette_highlight = Vec3(1.0f, 0.82f, 0.52f);
    float global_strength = 0.75f;
    float temporal_coherence = 0.6f;
    StylizedSkySettings sky;
    PainterlyMaterialSettings material;
    StylizedOutlineSettings outline;
    StylizedWorldAdapterSettings world_adapters;
};

class StylizeModeState {
public:
    bool enabled = false;
    StyleProfileId active_profile = StyleProfileId::PainterlyOil;
    StyleProfile profile;

    StylizeModeState();

    void applyPreset(StyleProfileId id);
    void setGlobalStrength(float strength);
    const char* activeProfileName() const;

    static const std::vector<StyleProfile>& presets();
};

} // namespace Stylize
