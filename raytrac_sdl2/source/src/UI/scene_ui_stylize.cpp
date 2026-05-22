#include "scene_ui.h"
#include "Renderer.h"
#include "Stylize/StylizeModeState.h"
#include "imgui.h"

#include <algorithm>

// Defined in Main.cpp — one-shot request to re-run the display post (honoring the tonemap
// on/off setting) without re-rendering. Must be declared at global scope: an `extern`
// inside the anonymous namespace below would acquire internal linkage and never resolve
// to the real global definition.
extern bool stylize_redisplay;

namespace {

bool drawVec3Color(const char* label, Vec3& color) {
    float values[3] = {
        color.x,
        color.y,
        color.z
    };
    if (!ImGui::ColorEdit3(label, values)) {
        return false;
    }
    color = Vec3(values[0], values[1], values[2]);
    return true;
}

bool drawStrength(const char* id, const char* label, float& value) {
    return SceneUI::DrawSmartFloat(id, label, &value, 0.0f, 1.0f, "%.2f", false, nullptr, 16);
}

const char* strokeDirectionLabel(Stylize::StrokeDirectionMode mode) {
    switch (mode) {
        case Stylize::StrokeDirectionMode::SurfaceNormal: return "Surface Normal";
        case Stylize::StrokeDirectionMode::Vertical: return "Vertical";
        case Stylize::StrokeDirectionMode::Horizontal: return "Horizontal";
        case Stylize::StrokeDirectionMode::Diagonal: return "Diagonal";
        case Stylize::StrokeDirectionMode::CrossHatch: return "Cross Hatch";
    }
    return "Surface Normal";
}

const char* outlineLineTypeLabel(Stylize::OutlineLineType type) {
    switch (type) {
        case Stylize::OutlineLineType::Ink: return "Ink";
        case Stylize::OutlineLineType::OilPaint: return "Oil Paint";
        case Stylize::OutlineLineType::Pencil: return "Pencil";
        case Stylize::OutlineLineType::DryBrush: return "Dry Brush";
        case Stylize::OutlineLineType::Pressure: return "Pressure";
    }
    return "Ink";
}

const char* outlineColorModeLabel(Stylize::OutlineColorMode mode) {
    switch (mode) {
        case Stylize::OutlineColorMode::PaletteShadow: return "Palette Shadow";
        case Stylize::OutlineColorMode::CustomColor: return "Custom Color";
        case Stylize::OutlineColorMode::MaterialTint: return "Material Tint";
        case Stylize::OutlineColorMode::WarmPaint: return "Warm Paint";
        case Stylize::OutlineColorMode::CoolPencil: return "Cool Pencil";
    }
    return "Palette Shadow";
}

const char* skyStyleLabel(Stylize::SkyStylePreset style) {
    switch (style) {
        case Stylize::SkyStylePreset::PainterlyClouds: return "Painterly Clouds";
        case Stylize::SkyStylePreset::CartoonCel: return "Cartoon Cel";
        case Stylize::SkyStylePreset::SunsetBands: return "Sunset Bands";
        case Stylize::SkyStylePreset::InkWash: return "Ink Wash";
        case Stylize::SkyStylePreset::ClearGradient: return "Clear Gradient";
    }
    return "Painterly Clouds";
}

void applySkyStyleDefaults(Stylize::StylizedSkySettings& sky, Stylize::SkyStylePreset style) {
    sky.style = style;
    sky.enabled = true;
    switch (style) {
        case Stylize::SkyStylePreset::CartoonCel:
            sky.horizon_color = Vec3(0.62f, 0.84f, 1.0f);
            sky.zenith_color = Vec3(0.18f, 0.48f, 0.94f);
            sky.sun_glow_color = Vec3(1.0f, 0.84f, 0.24f);
            sky.gradient_strength = 0.96f;
            sky.cloud_brush_scale = 2.4f;
            sky.cloud_brush_strength = 0.86f;
            sky.wind_smear = 0.05f;
            sky.horizon_haze = 0.24f;
            sky.sun_disc_scale = 12.0f;
            sky.cloud_roundness = 0.92f;
            break;
        case Stylize::SkyStylePreset::SunsetBands:
            sky.horizon_color = Vec3(0.98f, 0.48f, 0.30f);
            sky.zenith_color = Vec3(0.26f, 0.20f, 0.52f);
            sky.sun_glow_color = Vec3(1.0f, 0.62f, 0.26f);
            sky.gradient_strength = 0.94f;
            sky.cloud_brush_scale = 2.0f;
            sky.cloud_brush_strength = 0.70f;
            sky.wind_smear = 0.32f;
            sky.horizon_haze = 0.68f;
            sky.sun_disc_scale = 10.0f;
            sky.cloud_roundness = 0.78f;
            break;
        case Stylize::SkyStylePreset::InkWash:
            sky.horizon_color = Vec3(0.70f, 0.72f, 0.66f);
            sky.zenith_color = Vec3(0.42f, 0.50f, 0.56f);
            sky.sun_glow_color = Vec3(0.86f, 0.78f, 0.58f);
            sky.gradient_strength = 0.78f;
            sky.cloud_brush_scale = 1.4f;
            sky.cloud_brush_strength = 0.38f;
            sky.wind_smear = 0.18f;
            sky.horizon_haze = 0.42f;
            sky.sun_disc_scale = 5.5f;
            sky.cloud_roundness = 0.56f;
            break;
        case Stylize::SkyStylePreset::ClearGradient:
            sky.horizon_color = Vec3(0.76f, 0.88f, 1.0f);
            sky.zenith_color = Vec3(0.28f, 0.52f, 0.90f);
            sky.sun_glow_color = Vec3(1.0f, 0.84f, 0.48f);
            sky.gradient_strength = 0.86f;
            sky.cloud_brush_strength = 0.0f;
            sky.wind_smear = 0.0f;
            sky.horizon_haze = 0.26f;
            sky.sun_disc_scale = 6.5f;
            sky.cloud_roundness = 0.72f;
            break;
        case Stylize::SkyStylePreset::PainterlyClouds:
        default:
            sky.horizon_color = Vec3(0.74f, 0.64f, 0.48f);
            sky.zenith_color = Vec3(0.30f, 0.45f, 0.72f);
            sky.sun_glow_color = Vec3(1.0f, 0.72f, 0.38f);
            sky.gradient_strength = 0.85f;
            sky.cloud_brush_scale = 2.1f;
            sky.cloud_brush_strength = 0.68f;
            sky.wind_smear = 0.25f;
            sky.horizon_haze = 0.35f;
            sky.sun_disc_scale = 8.0f;
            sky.cloud_roundness = 0.70f;
            break;
    }
}

void markStylizeChanged(UIContext& ctx) {
    (void)ctx;
    // Stylize is a screen-space post pass — it does not change the path-traced render or
    // its AOVs. Re-apply the post on the EXISTING converged image instead of resetting
    // accumulation (which would clear the render + AOV buffers and make the layer
    // flicker/disappear until the next render). Use stylize_redisplay rather than
    // apply_tonemap so the rebuild HONORS the user's tonemap on/off setting — apply_tonemap
    // would force tonemapping even when the user has it disabled.
    stylize_redisplay = true;
}

} // namespace

void SceneUI::drawStylizePanel(UIContext& ctx) {
    Stylize::StylizeModeState& state = ctx.renderer.stylizeMode;
    Stylize::StyleProfile& profile = state.profile;

    UIWidgets::ColoredHeader("Stylize Mode", ImVec4(0.88f, 0.70f, 0.42f, 1.0f));
    UIWidgets::Divider();

    bool changed = false;
    changed |= ImGui::Checkbox("Enable Stylize Layer", &state.enabled);

    const auto& presets = Stylize::StylizeModeState::presets();
    int current_index = 0;
    for (int i = 0; i < static_cast<int>(presets.size()); ++i) {
        if (presets[i].id == state.active_profile) {
            current_index = i;
            break;
        }
    }

    if (ImGui::BeginCombo("Style Profile", presets[current_index].name)) {
        for (int i = 0; i < static_cast<int>(presets.size()); ++i) {
            const bool selected = (i == current_index);
            if (ImGui::Selectable(presets[i].name, selected)) {
                state.applyPreset(presets[i].id);
                changed = true;
            }
            if (selected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }

    changed |= drawStrength("stylize_global", "Global", profile.global_strength);
    changed |= drawStrength("stylize_temporal", "Temporal", profile.temporal_coherence);

    if (UIWidgets::BeginSection("Palette", ImVec4(0.92f, 0.68f, 0.40f, 1.0f), true)) {
        changed |= drawVec3Color("Shadow", profile.palette_shadow);
        changed |= drawVec3Color("Mid", profile.palette_mid);
        changed |= drawVec3Color("Highlight", profile.palette_highlight);
        UIWidgets::EndSection();
    }

    if (UIWidgets::BeginSection("Painterly Sky", ImVec4(0.42f, 0.78f, 1.0f, 1.0f), true)) {
        changed |= ImGui::Checkbox("Sky Layer", &profile.sky.enabled);
        int sky_style = static_cast<int>(profile.sky.style);
        if (ImGui::BeginCombo("Sky Style", skyStyleLabel(profile.sky.style))) {
            for (int i = 0; i <= static_cast<int>(Stylize::SkyStylePreset::ClearGradient); ++i) {
                auto style = static_cast<Stylize::SkyStylePreset>(i);
                const bool selected = (sky_style == i);
                if (ImGui::Selectable(skyStyleLabel(style), selected)) {
                    applySkyStyleDefaults(profile.sky, style);
                    changed = true;
                }
                if (selected) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        changed |= drawVec3Color("Horizon", profile.sky.horizon_color);
        changed |= drawVec3Color("Zenith", profile.sky.zenith_color);
        changed |= drawVec3Color("Sun Glow", profile.sky.sun_glow_color);
        changed |= drawStrength("sky_grad", "Gradient", profile.sky.gradient_strength);
        changed |= SceneUI::DrawSmartFloat("sky_cloud_scale", "Cloud Scale", &profile.sky.cloud_brush_scale, 0.1f, 8.0f, "%.2f", false, nullptr, 16);
        changed |= drawStrength("sky_cloud_strength", "Cloud Brush", profile.sky.cloud_brush_strength);
        changed |= SceneUI::DrawSmartFloat("sky_sun_disc", "Sun Disc", &profile.sky.sun_disc_scale, 0.5f, 24.0f, "%.1f", false, nullptr, 16);
        changed |= drawStrength("sky_cloud_round", "Cloud Round", profile.sky.cloud_roundness);
        changed |= drawStrength("sky_wind_smear", "Wind Smear", profile.sky.wind_smear);
        changed |= drawStrength("sky_haze", "Horizon Haze", profile.sky.horizon_haze);
        UIWidgets::EndSection();
    }

    if (UIWidgets::BeginSection("Painterly Materials", ImVec4(1.0f, 0.58f, 0.44f, 1.0f), true)) {
        changed |= ImGui::Checkbox("Material Layer", &profile.material.enabled);
        int stroke_mode = static_cast<int>(profile.material.stroke_direction);
        if (ImGui::BeginCombo("Stroke Direction", strokeDirectionLabel(profile.material.stroke_direction))) {
            for (int i = 0; i <= static_cast<int>(Stylize::StrokeDirectionMode::CrossHatch); ++i) {
                auto mode = static_cast<Stylize::StrokeDirectionMode>(i);
                const bool selected = (stroke_mode == i);
                if (ImGui::Selectable(strokeDirectionLabel(mode), selected)) {
                    profile.material.stroke_direction = mode;
                    changed = true;
                }
                if (selected) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        changed |= drawStrength("mat_brush", "Brush", profile.material.brush_strength);
        changed |= SceneUI::DrawSmartFloat("mat_brush_scale", "Scale", &profile.material.brush_scale, 0.1f, 12.0f, "%.2f", false, nullptr, 16);
        changed |= drawStrength("mat_pigment", "Pigment", profile.material.pigment_thickness);
        changed |= drawStrength("mat_dry", "Dry Brush", profile.material.dry_brush);
        changed |= ImGui::Checkbox("Wet Oil Model", &profile.material.wet_oil_model);
        if (profile.material.wet_oil_model) {
            changed |= drawStrength("mat_oil_body", "Body", profile.material.oil_body);
            changed |= drawStrength("mat_oil_load", "Load", profile.material.paint_load);
            changed |= drawStrength("mat_oil_pickup", "Pickup", profile.material.pickup_rate);
            changed |= drawStrength("mat_oil_deposit", "Deposit", profile.material.deposit_rate);
            changed |= drawStrength("mat_oil_buildup", "Buildup", profile.material.bristle_buildup);
        }
        changed |= drawStrength("mat_surface_adherence", "Surface Lock", profile.material.surface_adherence);
        changed |= drawStrength("mat_depth_scale", "Depth Scale", profile.material.depth_scale_response);
        changed |= drawStrength("mat_edge_respect", "Edge Respect", profile.material.edge_respect);
        changed |= drawStrength("mat_palette_influence", "Palette", profile.material.palette_influence);
        changed |= drawStrength("mat_material_color", "Material Color", profile.material.material_color_preservation);
        changed |= drawStrength("mat_simplify", "Simplify", profile.material.color_simplification);
        changed |= drawStrength("mat_rough_bias", "Rough Bias", profile.material.roughness_bias);
        changed |= drawStrength("mat_normal_soft", "Normal Soft", profile.material.normal_softening);
        UIWidgets::EndSection();
    }

    if (UIWidgets::BeginSection("Outlines", ImVec4(0.72f, 0.72f, 0.78f, 1.0f), true)) {
        changed |= ImGui::Checkbox("Outline Layer", &profile.outline.enabled);
        int line_type = static_cast<int>(profile.outline.line_type);
        if (ImGui::BeginCombo("Line Type", outlineLineTypeLabel(profile.outline.line_type))) {
            for (int i = 0; i <= static_cast<int>(Stylize::OutlineLineType::Pressure); ++i) {
                auto type = static_cast<Stylize::OutlineLineType>(i);
                const bool selected = (line_type == i);
                if (ImGui::Selectable(outlineLineTypeLabel(type), selected)) {
                    profile.outline.line_type = type;
                    changed = true;
                }
                if (selected) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        int color_mode = static_cast<int>(profile.outline.color_mode);
        if (ImGui::BeginCombo("Line Color", outlineColorModeLabel(profile.outline.color_mode))) {
            for (int i = 0; i <= static_cast<int>(Stylize::OutlineColorMode::CoolPencil); ++i) {
                auto mode = static_cast<Stylize::OutlineColorMode>(i);
                const bool selected = (color_mode == i);
                if (ImGui::Selectable(outlineColorModeLabel(mode), selected)) {
                    profile.outline.color_mode = mode;
                    changed = true;
                }
                if (selected) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        if (profile.outline.color_mode == Stylize::OutlineColorMode::CustomColor) {
            changed |= drawVec3Color("Custom Ink", profile.outline.custom_color);
        }
        changed |= SceneUI::DrawSmartFloat("out_strength", "Strength", &profile.outline.strength, 0.0f, 16.0f, "%.2f", false, nullptr, 16);
        changed |= SceneUI::DrawSmartFloat("out_width", "Width", &profile.outline.width, 0.1f, 24.0f, "%.2f", false, nullptr, 16);
        changed |= drawStrength("out_depth", "Depth", profile.outline.depth_sensitivity);
        changed |= drawStrength("out_normal", "Normal", profile.outline.normal_sensitivity);
        changed |= drawStrength("out_taper", "Taper", profile.outline.taper);
        changed |= drawStrength("out_breakup", "Breakup", profile.outline.break_up);
        changed |= drawStrength("out_bleed", "Color Bleed", profile.outline.color_bleed);
        changed |= drawStrength("out_distance_thin", "Distance Thin", profile.outline.distance_thinning);
        changed |= drawStrength("out_detail_protect", "Detail Protect", profile.outline.detail_protection);
        UIWidgets::EndSection();
    }

    if (UIWidgets::BeginSection("World Adapters", ImVec4(0.58f, 0.92f, 0.58f, 1.0f), true)) {
        changed |= drawStrength("world_terrain", "Terrain Stroke", profile.world_adapters.terrain_stroke_blend);
        changed |= drawStrength("world_foliage_cluster", "Foliage Cluster", profile.world_adapters.foliage_cluster_simplification);
        changed |= drawStrength("world_foliage_palette", "Foliage Var", profile.world_adapters.foliage_palette_variance);
        changed |= drawStrength("world_volume_grain", "Volume Grain", profile.world_adapters.volume_grain);
        changed |= drawStrength("world_force_motion", "Force Motion", profile.world_adapters.force_field_motion_response);
        UIWidgets::EndSection();
    }

    if (changed) {
        profile.global_strength = std::clamp(profile.global_strength, 0.0f, 1.0f);
        profile.temporal_coherence = std::clamp(profile.temporal_coherence, 0.0f, 1.0f);
        markStylizeChanged(ctx);
    }
}
