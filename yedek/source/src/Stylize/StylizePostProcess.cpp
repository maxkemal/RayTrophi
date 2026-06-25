#include "Stylize/StylizePostProcess.h"
#include "Stylize/StylizeCore.h"

//
// Thin host wrapper over the shared StylizeCore math. All the actual stylize
// logic lives in StylizeCore.h so the CPU path and the CUDA stylize kernel
// share ONE implementation (no drift). This file only converts the project's
// Vec3 / StyleProfile / StylizeAOVSample types into the POD core mirrors.
//

namespace Stylize {
namespace {

inline StylizeCore::SV3 toSV3(const Vec3& v) { return StylizeCore::mk(v.x, v.y, v.z); }
inline Vec3 fromSV3(const StylizeCore::SV3& v) { return Vec3(v.x, v.y, v.z); }

StylizeCore::StyleProfileCore toCoreProfile(const StyleProfile& p) {
    StylizeCore::StyleProfileCore c;
    c.palette_shadow = toSV3(p.palette_shadow);
    c.palette_mid = toSV3(p.palette_mid);
    c.palette_highlight = toSV3(p.palette_highlight);
    c.global_strength = p.global_strength;
    c.temporal_coherence = p.temporal_coherence;

    c.sky.enabled = p.sky.enabled ? 1 : 0;
    c.sky.style = static_cast<int>(p.sky.style);
    c.sky.horizon_color = toSV3(p.sky.horizon_color);
    c.sky.zenith_color = toSV3(p.sky.zenith_color);
    c.sky.sun_glow_color = toSV3(p.sky.sun_glow_color);
    c.sky.gradient_strength = p.sky.gradient_strength;
    c.sky.cloud_brush_scale = p.sky.cloud_brush_scale;
    c.sky.cloud_brush_strength = p.sky.cloud_brush_strength;
    c.sky.wind_smear = p.sky.wind_smear;
    c.sky.horizon_haze = p.sky.horizon_haze;
    c.sky.sun_disc_scale = p.sky.sun_disc_scale;
    c.sky.cloud_roundness = p.sky.cloud_roundness;

    c.material.enabled = p.material.enabled ? 1 : 0;
    c.material.stroke_direction = static_cast<int>(p.material.stroke_direction);
    c.material.brush_strength = p.material.brush_strength;
    c.material.brush_scale = p.material.brush_scale;
    c.material.pigment_thickness = p.material.pigment_thickness;
    c.material.dry_brush = p.material.dry_brush;
    c.material.wet_oil_model = p.material.wet_oil_model ? 1 : 0;
    c.material.oil_body = p.material.oil_body;
    c.material.paint_load = p.material.paint_load;
    c.material.pickup_rate = p.material.pickup_rate;
    c.material.deposit_rate = p.material.deposit_rate;
    c.material.bristle_buildup = p.material.bristle_buildup;
    c.material.surface_adherence = p.material.surface_adherence;
    c.material.depth_scale_response = p.material.depth_scale_response;
    c.material.edge_respect = p.material.edge_respect;
    c.material.palette_influence = p.material.palette_influence;
    c.material.material_color_preservation = p.material.material_color_preservation;
    c.material.color_simplification = p.material.color_simplification;
    c.material.roughness_bias = p.material.roughness_bias;
    c.material.normal_softening = p.material.normal_softening;

    c.outline.enabled = p.outline.enabled ? 1 : 0;
    c.outline.line_type = static_cast<int>(p.outline.line_type);
    c.outline.color_mode = static_cast<int>(p.outline.color_mode);
    c.outline.custom_color = toSV3(p.outline.custom_color);
    c.outline.strength = p.outline.strength;
    c.outline.width = p.outline.width;
    c.outline.depth_sensitivity = p.outline.depth_sensitivity;
    c.outline.normal_sensitivity = p.outline.normal_sensitivity;
    c.outline.taper = p.outline.taper;
    c.outline.break_up = p.outline.break_up;
    c.outline.color_bleed = p.outline.color_bleed;
    c.outline.distance_thinning = p.outline.distance_thinning;
    c.outline.detail_protection = p.outline.detail_protection;
    return c;
}

StylizeCore::StylizeAOVCore toCoreAOV(const StylizeAOVSample& a) {
    StylizeCore::StylizeAOVCore c;
    c.albedo = toSV3(a.albedo);
    c.normal = toSV3(a.normal);
    c.world_position = toSV3(a.world_position);
    c.view_dir = toSV3(a.view_dir);
    c.sun_dir = toSV3(a.sun_dir);
    c.screen_u = a.screen_u;
    c.screen_v = a.screen_v;
    c.sun_size_degrees = a.sun_size_degrees;
    c.sun_elevation_degrees = a.sun_elevation_degrees;
    c.nishita_clouds_enabled = a.nishita_clouds_enabled ? 1 : 0;
    c.nishita_cloud_coverage = a.nishita_cloud_coverage;
    c.nishita_cloud_density = a.nishita_cloud_density;
    c.nishita_cloud_scale = a.nishita_cloud_scale;
    c.nishita_cloud_offset_x = a.nishita_cloud_offset_x;
    c.nishita_cloud_offset_z = a.nishita_cloud_offset_z;
    c.nishita_cloud_seed = a.nishita_cloud_seed;
    c.depth = a.depth;
    c.edge = a.edge;
    c.pixel_scale = a.pixel_scale;
    c.material_id = a.material_id;
    c.valid = a.valid ? 1 : 0;
    c.hit = a.hit ? 1 : 0;
    return c;
}

} // namespace

Vec3 applyPostProcess(const Vec3& input_color,
                      int x,
                      int y,
                      int frame_index,
                      const StylizeModeState& state) {
    return applyPostProcess(input_color, StylizeAOVSample{}, x, y, frame_index, state);
}

Vec3 applyPostProcess(const Vec3& input_color,
                      const StylizeAOVSample& aov,
                      int x,
                      int y,
                      int frame_index,
                      const StylizeModeState& state) {
    if (!state.enabled) {
        return input_color;
    }
    const StylizeCore::StyleProfileCore profile = toCoreProfile(state.profile);
    const StylizeCore::StylizeAOVCore core_aov = toCoreAOV(aov);
    const StylizeCore::SV3 out =
        StylizeCore::applyPostProcess(toSV3(input_color), core_aov, x, y, frame_index, profile);
    return fromSV3(out);
}

StylizeCore::StyleProfileCore makeCoreProfile(const StyleProfile& profile) {
    return toCoreProfile(profile);
}

} // namespace Stylize
