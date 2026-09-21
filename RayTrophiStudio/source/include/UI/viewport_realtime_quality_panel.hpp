#pragma once

#include "Api/RtApi.h"
#include "UI/ViewportCutoutUI.h"
#include "UI/rayfusion_status_panel.hpp"
#include "globals.h"
#include "imgui.h"
#include "scene_ui.h"

#include <algorithm>

// Realtime PBR kademe raporu -- Render Inspector > Sampling & Quality, Lighting
// Mode'un hemen altinda. Butun mutasyonlar Python/IPC'nin kullandigi AYNI rtapi
// operasyonlarindan gecer; okunanlar da ayni degerlerdir.
//
// ★★ Panel üç katmana bölünmüştür:
//   1. Kullanıcı ayarları   — her zaman görünür, en üstte.
//   2. Shader Report        — CollapsingHeader (varsayılan kapalı):
//      shader'a giden değerlerin raporu. Ayar değil; panelle çekirdek
//      ayrışırsa burada görünür hale gelir.
//   3. RayFusion development — geliştirici tanılama (TreeNode, kapalı).
//
// ★ 2026-09-03: viewport'un sağ üstündeki "Q" overlay'i kaldırıldı. Aynı
//   içeriği ikinci kez çizen, sürekli yer kaplayan bir yüzeydi; ve Render
//   Settings'teki kopya `#if 0` içinde olduğu için ASLINDA tek canlı yüzeydi.
//   Başlık kapalı açılır: rapor lazım olunca bakılır, sürekli değil.
inline void DrawRealtimeQualitySettingsSection(UIContext& ctx, int shadingMode) {
    if (!ImGui::CollapsingHeader("Realtime PBR Quality##RenderSettings"))
        return;

    rtapi::ViewportQualityInfo quality = rtapi::viewportQuality();

    // ── 1. Kullanıcı ayarları ────────────────────────────────────────────────
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
    if (shadingMode != 1)
        ImGui::TextDisabled("Lighting is editable in Material viewport shading.");

    DrawViewportAutomaticCutout();

    // ── 2. Shader Report (kapalı) ────────────────────────────────────────────
    // Bu satırların tümü AYAR DEĞİL, shader'a giden değerlerin raporudur.
    // Panelle çekirdek ayrışırsa burada görünür hale gelir -- o yüzden rapor
    // bir yerde durmalı, ama kullanıcı ayarlarının önünde değil.
    if (ImGui::CollapsingHeader("Shader Report##RealtimeRenderSettings")) {
        const rtapi::ViewportPreviewLightingInfo lighting =
            rtapi::viewportPreviewLighting();

        // ── PBR özellik matrisi ──────────────────────────────────────────────
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

        // ── Gölge bütçesi ────────────────────────────────────────────────────
        ImGui::Separator();
        ImGui::TextDisabled("Shadows");
        ImGui::SameLine(150.0f);
        ImGui::Text("%d px, %d PCF, %d/%d lights",
                    quality.shadow_tile_resolution, quality.shadow_pcf_samples,
                    lighting.shadowed_light_count, quality.shadow_light_budget);
        ImGui::TextDisabled("Volume shadows");
        ImGui::SameLine(150.0f);
        ImGui::Text("%d px, %d layers, %d steps",
                    quality.volume_shadow_tile_resolution,
                    quality.volume_shadow_depth_layers, quality.volume_shadow_steps);
        ImGui::TextDisabled("Shadow atlas");
        ImGui::SameLine(150.0f);
        ImGui::Text("%d px, capacity %d tiles",
                    quality.shadow_atlas_resolution, quality.shadow_tile_capacity);
        ImGui::TextDisabled("Directional CSM");
        ImGui::SameLine(150.0f);
        ImGui::Text("%d cascades", quality.directional_shadow_cascades);

        // ── Uyarılar ─────────────────────────────────────────────────────────
        // ★ Bu üç uyarı yalnızca kaldırılan overlay'de vardı. Hepsi "istediğin
        //   şey TAM olarak olmadı" diyor -- sessizce düşen bir yetenek, bu
        //   depoda raporlanmadığı sürece hiç fark edilmez.
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
        if (quality.transmission != "screen_space_thickness" ||
            quality.resin_interior != "rt_aligned") {
            ImGui::TextColored(ImVec4(0.96f, 0.74f, 0.34f, 1.0f),
                               "Transmission/resin use bounded raster approximations.");
        }
        if (quality.sdf_surface == "shared_nanovdb_depth_pbr")
            ImGui::TextDisabled("SurfaceSDF: shared RT field + realtime depth/PBR");
        if (quality.volumes == "shared_vdb_dense_single_scatter")
            ImGui::TextDisabled("Volumes: shared RT fields + bounded single scatter");

        // ── Geometri gönderimi (telemetri) ───────────────────────────────────
        // ★★★★ 2026-09-08'e kadar realtime viewport'ta GPU culling HİÇ açılmadı
        //   ve bunun ekranda TEK bir belirtisi yoktu: sahne doğru çizilir,
        //   yalnızca yavaştır. 45,9M üçgen kare başına gönderiliyordu, LOD
        //   proxy'leri o sayının 30 bininde kalıyordu. Bir arızanın görsel ipucu
        //   yoksa geriye tek yol kalır: sayıyı ekrana yazmak.
        //
        // ★ Anahtarın kendisi bilerek panele KONMADI. A/B için
        //   viewport.set_raster_gpu_instancing (IPC/Python) var.
        const rtapi::ViewportFrameTelemetryInfo frame = rtapi::viewportFrameTelemetry();
        if (frame.available) {
            ImGui::Separator();
            ImGui::Text("Submitted: %.2fM tris in %u draws",
                        static_cast<double>(frame.visible_triangles) / 1.0e6,
                        static_cast<unsigned>(frame.draw_calls));
            if (frame.gpu_culling) {
                ImGui::TextDisabled("GPU culling on - %u full / %u proxy instances",
                                    static_cast<unsigned>(frame.full_instances),
                                    static_cast<unsigned>(frame.proxy_instances));
            } else {
                // Bu kombinasyon bir ARIZADIR ve ekranda doğru görünür.
                ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.35f, 1.0f),
                    "GPU culling OFF - every instance is drawn, wherever the camera looks");
                if (frame.cull_mesh_count == 0)
                    ImGui::TextDisabled("cull_mesh_count 0: the instance layout never built.");
            }
            // ★★★ Derinlik ön geçişi: UYGULANAN değer. A/B için
            //   viewport.set_raster_depth_prepass var.
            if (frame.depth_prepass) {
                ImGui::TextDisabled("Depth prepass on - each pixel shades once");
            } else {
                ImGui::TextDisabled("Depth prepass OFF - every overlapping layer "
                                    "reruns the scene shader");
            }
            if (frame.scatter_triangle_target > 0 &&
                frame.visible_triangles > frame.scatter_triangle_target)
                ImGui::TextColored(ImVec4(0.96f, 0.74f, 0.34f, 1.0f),
                    "%.0f%% over the scatter LOD target",
                    100.0 * static_cast<double>(frame.visible_triangles) /
                        static_cast<double>(frame.scatter_triangle_target) - 100.0);
        }
    } // Shader Report

    // ── 3. RayFusion development (geliştirici tanılama, kapalı) ─────────────
    DrawRayFusionDevelopmentStatus();
}
