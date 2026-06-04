#include "Stylize/StylizeModeState.h"

#include <algorithm>

namespace Stylize {
namespace {

StyleProfile makePainterlyOil() {
    StyleProfile p;
    p.id = StyleProfileId::PainterlyOil;
    p.name = "Painterly Oil";
    p.palette_shadow = Vec3(0.18f, 0.13f, 0.10f);
    p.palette_mid = Vec3(0.60f, 0.45f, 0.30f);
    p.palette_highlight = Vec3(1.0f, 0.80f, 0.48f);
    p.global_strength = 0.78f;
    p.temporal_coherence = 0.72f;
    p.sky.style = SkyStylePreset::PainterlyClouds;
    p.sky.cloud_brush_scale = 2.1f;
    p.sky.cloud_brush_strength = 0.68f;
    p.sky.sun_disc_scale = 8.0f;
    p.sky.cloud_roundness = 0.70f;
    p.material.brush_strength = 0.64f;
    p.material.brush_scale = 0.85f;
    p.material.pigment_thickness = 0.50f;
    p.material.dry_brush = 0.18f;
    p.material.wet_oil_model = true;
    p.material.oil_body = 0.62f;
    p.material.paint_load = 0.93f;
    p.material.pickup_rate = 0.12f;
    p.material.deposit_rate = 0.88f;
    p.material.bristle_buildup = 0.72f;
    p.material.surface_adherence = 0.78f;
    p.material.depth_scale_response = 0.55f;
    p.material.edge_respect = 0.82f;
    p.material.palette_influence = 0.55f;
    p.material.material_color_preservation = 0.72f;
    p.material.color_simplification = 0.28f;
    p.outline.line_type = OutlineLineType::OilPaint;
    p.outline.color_mode = OutlineColorMode::WarmPaint;
    p.outline.strength = 0.30f;
    p.outline.width = 1.35f;
    p.outline.taper = 0.42f;
    p.outline.break_up = 0.24f;
    p.outline.color_bleed = 0.44f;
    p.outline.distance_thinning = 0.62f;
    p.outline.detail_protection = 0.58f;
    p.world_adapters.terrain_stroke_blend = 0.52f;
    p.world_adapters.force_field_motion_response = 0.34f;
    return p;
}

StyleProfile makeGouache() {
    StyleProfile p;
    p.id = StyleProfileId::Gouache;
    p.name = "Gouache";
    p.palette_shadow = Vec3(0.16f, 0.20f, 0.22f);
    p.palette_mid = Vec3(0.48f, 0.56f, 0.50f);
    p.palette_highlight = Vec3(0.94f, 0.88f, 0.68f);
    p.global_strength = 0.68f;
    p.temporal_coherence = 0.65f;
    p.sky.style = SkyStylePreset::PainterlyClouds;
    p.sky.gradient_strength = 0.72f;
    p.sky.cloud_brush_strength = 0.46f;
    p.sky.sun_disc_scale = 6.0f;
    p.sky.cloud_roundness = 0.82f;
    p.material.brush_strength = 0.42f;
    p.material.brush_scale = 0.75f;
    p.material.pigment_thickness = 0.30f;
    p.material.dry_brush = 0.12f;
    p.material.wet_oil_model = true;
    p.material.oil_body = 0.38f;
    p.material.paint_load = 0.72f;
    p.material.pickup_rate = 0.18f;
    p.material.deposit_rate = 0.64f;
    p.material.bristle_buildup = 0.36f;
    p.material.surface_adherence = 0.66f;
    p.material.depth_scale_response = 0.42f;
    p.material.edge_respect = 0.76f;
    p.material.palette_influence = 0.45f;
    p.material.material_color_preservation = 0.80f;
    p.material.color_simplification = 0.34f;
    p.outline.line_type = OutlineLineType::DryBrush;
    p.outline.color_mode = OutlineColorMode::MaterialTint;
    p.outline.strength = 0.20f;
    p.outline.taper = 0.50f;
    p.outline.break_up = 0.28f;
    p.outline.color_bleed = 0.34f;
    p.outline.distance_thinning = 0.60f;
    p.outline.detail_protection = 0.56f;
    p.world_adapters.foliage_cluster_simplification = 0.42f;
    return p;
}

StyleProfile makeInkWash() {
    StyleProfile p;
    p.id = StyleProfileId::InkWash;
    p.name = "Ink + Wash";
    p.palette_shadow = Vec3(0.05f, 0.06f, 0.06f);
    p.palette_mid = Vec3(0.42f, 0.45f, 0.43f);
    p.palette_highlight = Vec3(0.90f, 0.88f, 0.78f);
    p.global_strength = 0.74f;
    p.temporal_coherence = 0.58f;
    p.sky.style = SkyStylePreset::InkWash;
    p.sky.horizon_color = Vec3(0.70f, 0.72f, 0.66f);
    p.sky.zenith_color = Vec3(0.42f, 0.50f, 0.56f);
    p.sky.cloud_brush_strength = 0.34f;
    p.sky.sun_disc_scale = 5.5f;
    p.sky.cloud_roundness = 0.56f;
    p.material.brush_strength = 0.38f;
    p.material.brush_scale = 0.65f;
    p.material.dry_brush = 0.08f;
    p.material.palette_influence = 0.42f;
    p.material.material_color_preservation = 0.74f;
    p.material.color_simplification = 0.44f;
    p.material.normal_softening = 0.36f;
    p.material.surface_adherence = 0.72f;
    p.material.edge_respect = 0.88f;
    p.outline.line_type = OutlineLineType::Ink;
    p.outline.color_mode = OutlineColorMode::PaletteShadow;
    p.outline.strength = 0.72f;
    p.outline.width = 1.6f;
    p.outline.taper = 0.28f;
    p.outline.break_up = 0.10f;
    p.outline.distance_thinning = 0.48f;
    p.outline.detail_protection = 0.38f;
    p.world_adapters.volume_grain = 0.34f;
    return p;
}

StyleProfile makeGraphicToon() {
    StyleProfile p;
    p.id = StyleProfileId::GraphicToon;
    p.name = "Graphic Toon";
    p.palette_shadow = Vec3(0.10f, 0.12f, 0.18f);
    p.palette_mid = Vec3(0.36f, 0.52f, 0.72f);
    p.palette_highlight = Vec3(0.98f, 0.88f, 0.36f);
    p.global_strength = 0.82f;
    p.temporal_coherence = 0.9f;
    p.sky.style = SkyStylePreset::CartoonCel;
    p.sky.horizon_color = Vec3(0.62f, 0.84f, 1.0f);
    p.sky.zenith_color = Vec3(0.18f, 0.48f, 0.94f);
    p.sky.sun_glow_color = Vec3(1.0f, 0.84f, 0.24f);
    p.sky.cloud_brush_strength = 0.86f;
    p.sky.sun_disc_scale = 12.0f;
    p.sky.cloud_roundness = 0.92f;
    p.material.brush_strength = 0.18f;
    p.material.palette_influence = 0.75f;
    p.material.material_color_preservation = 0.52f;
    p.material.color_simplification = 0.78f;
    p.material.normal_softening = 0.24f;
    p.outline.line_type = OutlineLineType::Pressure;
    p.outline.color_mode = OutlineColorMode::PaletteShadow;
    p.outline.strength = 0.86f;
    p.outline.width = 1.75f;
    p.outline.taper = 0.18f;
    p.outline.break_up = 0.02f;
    p.outline.distance_thinning = 0.36f;
    p.outline.detail_protection = 0.28f;
    p.world_adapters.foliage_cluster_simplification = 0.62f;
    return p;
}

StyleProfile makeClayMaquette() {
    StyleProfile p;
    p.id = StyleProfileId::ClayMaquette;
    p.name = "Clay / Maquette";
    p.palette_shadow = Vec3(0.28f, 0.24f, 0.21f);
    p.palette_mid = Vec3(0.62f, 0.54f, 0.45f);
    p.palette_highlight = Vec3(0.92f, 0.82f, 0.66f);
    p.global_strength = 0.64f;
    p.temporal_coherence = 0.82f;
    p.sky.style = SkyStylePreset::ClearGradient;
    p.sky.enabled = false;
    p.material.brush_strength = 0.22f;
    p.material.pigment_thickness = 0.18f;
    p.material.palette_influence = 0.28f;
    p.material.material_color_preservation = 0.88f;
    p.material.roughness_bias = 0.55f;
    p.material.normal_softening = 0.42f;
    p.outline.line_type = OutlineLineType::Pencil;
    p.outline.color_mode = OutlineColorMode::CoolPencil;
    p.outline.strength = 0.18f;
    p.outline.taper = 0.58f;
    p.outline.break_up = 0.32f;
    p.outline.distance_thinning = 0.68f;
    p.outline.detail_protection = 0.64f;
    p.world_adapters.terrain_stroke_blend = 0.18f;
    return p;
}

StyleProfile makeDreamySunset() {
    StyleProfile p;
    p.id = StyleProfileId::DreamySunset;
    p.name = "Dreamy Sunset";
    p.palette_shadow = Vec3(0.18f, 0.12f, 0.26f);
    p.palette_mid = Vec3(0.78f, 0.40f, 0.34f);
    p.palette_highlight = Vec3(1.0f, 0.72f, 0.34f);
    p.global_strength = 0.7f;
    p.temporal_coherence = 0.68f;
    p.sky.style = SkyStylePreset::SunsetBands;
    p.sky.horizon_color = Vec3(0.98f, 0.48f, 0.30f);
    p.sky.zenith_color = Vec3(0.26f, 0.20f, 0.52f);
    p.sky.sun_glow_color = Vec3(1.0f, 0.62f, 0.26f);
    p.sky.cloud_brush_strength = 0.70f;
    p.sky.sun_disc_scale = 10.0f;
    p.sky.cloud_roundness = 0.78f;
    p.sky.horizon_haze = 0.62f;
    p.sky.wind_smear = 0.42f;
    p.material.brush_strength = 0.38f;
    p.material.brush_scale = 0.9f;
    p.material.dry_brush = 0.18f;
    p.material.palette_influence = 0.62f;
    p.material.material_color_preservation = 0.68f;
    p.material.color_simplification = 0.30f;
    p.outline.line_type = OutlineLineType::OilPaint;
    p.outline.color_mode = OutlineColorMode::WarmPaint;
    p.outline.strength = 0.26f;
    p.outline.taper = 0.48f;
    p.outline.break_up = 0.22f;
    p.outline.color_bleed = 0.46f;
    p.outline.distance_thinning = 0.64f;
    p.outline.detail_protection = 0.58f;
    p.world_adapters.volume_grain = 0.26f;
    p.world_adapters.force_field_motion_response = 0.42f;
    return p;
}

} // namespace

StylizeModeState::StylizeModeState() {
    applyPreset(active_profile);
}

void StylizeModeState::applyPreset(StyleProfileId id) {
    active_profile = id;
    for (const StyleProfile& preset : presets()) {
        if (preset.id == id) {
            profile = preset;
            return;
        }
    }
    profile = presets().front();
    active_profile = profile.id;
}

void StylizeModeState::setGlobalStrength(float strength) {
    profile.global_strength = std::clamp(strength, 0.0f, 1.0f);
}

const char* StylizeModeState::activeProfileName() const {
    return profile.name;
}

const std::vector<StyleProfile>& StylizeModeState::presets() {
    static const std::vector<StyleProfile> kPresets = {
        makePainterlyOil(),
        makeGouache(),
        makeInkWash(),
        makeGraphicToon(),
        makeClayMaquette(),
        makeDreamySunset()
    };
    return kPresets;
}

} // namespace Stylize
