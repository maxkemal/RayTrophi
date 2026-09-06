#pragma once

#include "Api/RtApi.h"
#include "globals.h"
#include "imgui.h"
#include "scene_ui.h"

#include <algorithm>

// Realtime PBR kademe raporu -- Render Inspector > Sampling & Quality, Lighting
// Mode'un hemen altinda. Butun mutasyonlar Python/IPC'nin kullandigi AYNI rtapi
// operasyonlarindan gecer; okunanlar da ayni degerlerdir.
//
// ★★ Buradaki satirlarin cogu AYAR DEGIL, shader'a giden degerin RAPORU.
//    Panel ile cekirdek ayrisirsa bu deponun en pahali hata sinifi burada
//    gorunur hale gelir -- o yuzden rapor bir yerde durmali.
//
// ★ 2026-09-03: viewport'un sag ustundeki "Q" overlay'i kaldirildi. Ayni
//    icerigi ikinci kez cizen, surekli yer kaplayan bir yuzeydi; ve Render
//    Settings'teki kopya `#if 0` icinde oldugu icin ASLINDA tek canli yuzeydi.
//    Baslik kapali acilir: rapor lazim olunca bakilir, surekli degil.
inline void DrawRealtimeQualitySettingsSection(UIContext& ctx, int shadingMode) {
    if (!ImGui::CollapsingHeader("Realtime PBR Quality##RenderSettings"))
        return;

    rtapi::ViewportQualityInfo quality = rtapi::viewportQuality();
    static const char* qualityNames[] = {
        "Auto", "Performance", "Balanced", "Quality", "Full (no proxy)"};
    static const char* qualityKeys[] = {
        "auto", "performance", "balanced", "quality", "full"};
    int qualityIndex = std::clamp(static_cast<int>(
        ctx.render_settings.raster_viewport_quality_preset), 0, 4);
    ImGui::SetNextItemWidth(-1.0f);
    if (ImGui::Combo("Quality##RealtimeRenderSettings", &qualityIndex,
                     qualityNames, IM_ARRAYSIZE(qualityNames))) {
        rtapi::setViewportQuality(qualityKeys[qualityIndex]);
        quality = rtapi::viewportQuality();
    }

    int lightingIndex =
        ctx.render_settings.material_preview_lighting_preset ==
                MaterialPreviewLightingPreset::Scene ? 0 : 1;
    static const char* lightingNames[] = {"Scene", "3 Point"};
    if (shadingMode != 1) ImGui::BeginDisabled();
    ImGui::SetNextItemWidth(-1.0f);
    if (ImGui::Combo("Lighting##RealtimeRenderSettings", &lightingIndex,
                     lightingNames, IM_ARRAYSIZE(lightingNames))) {
        rtapi::setViewportPreviewLighting(
            lightingIndex == 0 ? "scene" : "three_point");
    }
    if (shadingMode != 1) ImGui::EndDisabled();

    const rtapi::ViewportPreviewLightingInfo lighting =
        rtapi::viewportPreviewLighting();
    ImGui::TextDisabled("PBR shader");
    ImGui::SameLine(150.0f);
    ImGui::TextUnformatted(quality.scene_pbr_shader.c_str());
    ImGui::TextDisabled("Opaque / Graph");
    ImGui::SameLine(150.0f);
    ImGui::Text("%s / %s", quality.opaque_core_parity.c_str(),
                quality.material_graph_surface.c_str());
    ImGui::TextDisabled("Clearcoat / SSS");
    ImGui::SameLine(150.0f);
    ImGui::Text("%s / %s", quality.clearcoat.c_str(),
                quality.subsurface.c_str());
    ImGui::TextDisabled("Translucency");
    ImGui::SameLine(150.0f);
    ImGui::TextUnformatted(quality.translucency.c_str());
    ImGui::TextDisabled("Anisotropy");
    ImGui::SameLine(150.0f);
    ImGui::TextUnformatted(quality.surface_anisotropy.c_str());
    ImGui::TextDisabled("Transparency");
    ImGui::SameLine(150.0f);
    ImGui::TextUnformatted(quality.transparency.c_str());
    ImGui::TextDisabled("Transmission");
    ImGui::SameLine(150.0f);
    ImGui::TextUnformatted(quality.transmission.c_str());
    ImGui::TextDisabled("Resin interior");
    ImGui::SameLine(150.0f);
    ImGui::TextUnformatted(quality.resin_interior.c_str());
    ImGui::TextDisabled("VDB / gas");
    ImGui::SameLine(150.0f);
    ImGui::TextUnformatted(quality.volumes.c_str());
    ImGui::TextDisabled("Shadows");
    ImGui::SameLine(150.0f);
    ImGui::Text("%d px, %d PCF, %d/%d lights",
                quality.shadow_tile_resolution, quality.shadow_pcf_samples,
                lighting.shadowed_light_count, quality.shadow_light_budget);
    ImGui::TextDisabled("Shadow atlas");
    ImGui::SameLine(150.0f);
    ImGui::Text("%d px, capacity %d tiles",
                quality.shadow_atlas_resolution, quality.shadow_tile_capacity);
    ImGui::TextDisabled("Directional CSM");
    ImGui::SameLine(150.0f);
    ImGui::Text("%d cascades", quality.directional_shadow_cascades);

    // ★ Bu uc uyari yalnizca kaldirilan overlay'de vardi. Hepsi "istedigin sey
    //   TAM olarak olmadi" diyor -- sessizce dusen bir yetenek, bu depoda
    //   raporlanmadigi surece hic fark edilmez.
    if (lighting.uses_scene_lights &&
        lighting.scene_light_total > lighting.shadowed_light_count) {
        ImGui::TextColored(ImVec4(0.96f, 0.74f, 0.34f, 1.0f),
            "%d visible light(s) continue without shadow",
            lighting.scene_light_total - lighting.shadowed_light_count);
    }
    if (lighting.world_sun_shadow)
        ImGui::TextDisabled("Physical Sky sun reserves one atlas tile.");
    if (lighting.world_ibl_fallback)
        ImGui::TextColored(ImVec4(0.96f, 0.74f, 0.34f, 1.0f),
                           "HDRI IBL: bounded fallback");

    if (shadingMode != 1)
        ImGui::TextDisabled("Lighting is editable in Material viewport shading.");
    if (quality.transmission != "screen_space_thickness" ||
        quality.resin_interior != "rt_aligned") {
        ImGui::TextColored(ImVec4(0.96f, 0.74f, 0.34f, 1.0f),
                           "Transmission/resin use bounded raster approximations.");
    }
    if (quality.sdf_surface == "shared_nanovdb_depth_pbr")
        ImGui::TextDisabled("SurfaceSDF: shared RT field + realtime depth/PBR");
    if (quality.volumes == "shared_vdb_dense_single_scatter")
        ImGui::TextDisabled("Volumes: shared RT fields + bounded single scatter");
}
