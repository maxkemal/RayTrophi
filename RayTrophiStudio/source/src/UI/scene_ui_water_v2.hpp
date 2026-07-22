#pragma once

#include "scene_ui.h"
#include "imgui.h"
#include "scene_data.h"
#include "WaterSystem.h"
#include "MeshModifiers.h"
#include "TriangleMesh.h"
#include "ProjectManager.h"
#include "Backend/IViewportBackend.h"
#include "Backend/VulkanBackend.h"
#include <algorithm>

extern std::unique_ptr<Backend::IBackend> g_backend;
extern std::unique_ptr<Backend::IViewportBackend> g_viewport_backend;
extern bool g_viewport_raster_rebuild_pending;
extern bool g_vulkan_rebuild_pending;
extern bool g_materials_dirty;
extern bool g_geometry_dirty;
extern std::atomic<uint64_t> g_scene_geometry_generation;
extern std::atomic<bool> g_needs_optix_sync;
extern bool g_mesh_cache_dirty;
extern bool g_cpu_bvh_refit_pending;

namespace {

Backend::IBackend* waterV2Backend(UIContext& ctx) {
    if (g_backend) return g_backend.get();
    if (ctx.backend_ptr && dynamic_cast<Backend::IViewportBackend*>(ctx.backend_ptr) == nullptr) {
        return ctx.backend_ptr;
    }
    return nullptr;
}

bool beginWaterV2Section(const char* title, const ImVec4& color, bool open = true) {
    ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(color.x * 0.24f, color.y * 0.24f, color.z * 0.24f, 0.92f));
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(color.x * 0.34f, color.y * 0.34f, color.z * 0.34f, 0.98f));
    const bool result = ImGui::CollapsingHeader(title,
        (open ? ImGuiTreeNodeFlags_DefaultOpen : 0) | ImGuiTreeNodeFlags_SpanAvailWidth);
    ImGui::PopStyleColor(2);
    return result;
}

void syncWaterV2Material(UIContext& ctx, WaterSurface& surface) {
    WaterManager::getInstance().syncSurfaceMaterial(&surface);
    g_materials_dirty = true;
    ctx.renderer.updateBackendMaterial(ctx.scene, surface.material_id);
    ctx.renderer.resetCPUAccumulation();
    if (Backend::IBackend* backend = waterV2Backend(ctx)) backend->resetAccumulation();
    if (g_viewport_backend && g_viewport_backend.get() != waterV2Backend(ctx)) {
        g_viewport_backend->resetAccumulation();
    }
}

void rebuildWaterV2Geometry(UIContext& ctx) {
    ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
    ctx.renderer.resetCPUAccumulation();
    if (Backend::IBackend* backend = waterV2Backend(ctx)) {
        ctx.renderer.rebuildBackendGeometry(ctx.scene);
        backend->resetAccumulation();
    }
    if (g_viewport_backend) {
        g_viewport_raster_rebuild_pending = true;
        g_viewport_backend->resetAccumulation();
    }
    if (dynamic_cast<Backend::VulkanBackendAdapter*>(waterV2Backend(ctx))) g_vulkan_rebuild_pending = true;
    g_geometry_dirty = true;
    g_scene_geometry_generation.fetch_add(1, std::memory_order_release);
    g_needs_optix_sync.store(true, std::memory_order_release);
    g_mesh_cache_dirty = true;
}

const char* waterV2ProfileName(const WaterSurface& surface) {
    if (surface.type == WaterSurface::Type::River) return "River / Hydrology";
    if (surface.type == WaterSurface::Type::Lake) return "Lake / Inland";
    return "Ocean / Open Water";
}

bool drawWaterV2Editor(UIContext& ctx, WaterSurface& surface, bool allowDelete) {
    bool manualChanged = false;
    bool presetApplied = false;
    bool geometryChanged = false;

    ImGui::TextDisabled("Water V3 / Dedicated BSDF");
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.25f, 0.82f, 1.0f, 1.0f), "%s", waterV2ProfileName(surface));

    const char* names[4] = { "Custom", nullptr, nullptr, nullptr };
    WaterWaveParams::WaterPreset values[4] = { WaterWaveParams::WaterPreset::Custom };
    int count = 1;
    if (surface.type == WaterSurface::Type::River) {
        names[count] = "Natural River";
        values[count++] = WaterWaveParams::WaterPreset::River;
    } else if (surface.type == WaterSurface::Type::Lake) {
        names[count] = "Still Lake"; values[count++] = WaterWaveParams::WaterPreset::Lake;
        names[count] = "Clear Pool"; values[count++] = WaterWaveParams::WaterPreset::Pool;
        names[count] = "Murky Pond"; values[count++] = WaterWaveParams::WaterPreset::Pond;
    } else {
        names[count] = "Calm Ocean"; values[count++] = WaterWaveParams::WaterPreset::CalmOcean;
        names[count] = "Storm Ocean"; values[count++] = WaterWaveParams::WaterPreset::StormyOcean;
        names[count] = "Tropical Ocean"; values[count++] = WaterWaveParams::WaterPreset::TropicalOcean;
    }
    int selected = 0;
    for (int i = 1; i < count; ++i) if (values[i] == surface.params.current_preset) selected = i;
    ImGui::SetNextItemWidth(-1.0f);
    if (ImGui::Combo("Preset", &selected, names, count)) {
        surface.params.applyPreset(values[selected]);
        presetApplied = true;
    }

    if (beginWaterV2Section("Optical Body", ImVec4(0.0f, 0.55f, 0.80f, 1.0f))) {
        manualChanged |= ImGui::ColorEdit3("Shallow Water", &surface.params.shallow_color.x);
        manualChanged |= ImGui::ColorEdit3("Deep Water", &surface.params.deep_color.x);
        manualChanged |= SceneUI::DrawSmartFloat("wv2_ior", "IOR", &surface.params.ior, 1.0f, 1.6f, "%.3f", false, nullptr, 16);
        manualChanged |= SceneUI::DrawSmartFloat("wv2_rough", "Unresolved Roughness", &surface.params.roughness, 0.0f, 0.60f, "%.3f", false, nullptr, 16);
        manualChanged |= SceneUI::DrawSmartFloat("wv2_depth", "Optical Depth", &surface.params.depth_max, 0.25f, 100.0f, "%.2f m", false, nullptr, 16);
        manualChanged |= SceneUI::DrawSmartFloat("wv2_absorb", "Absorption", &surface.params.absorption_density, 0.0f, 3.0f, "%.3f", false, nullptr, 16);
    }

    if (beginWaterV2Section(surface.type == WaterSurface::Type::River ? "Flow Response" : "Wave Spectrum",
                            ImVec4(0.18f, 0.62f, 1.0f, 1.0f))) {
        if (surface.type == WaterSurface::Type::River) {
            manualChanged |= SceneUI::DrawSmartFloat("wv2_flow", "Flow Speed Multiplier", &surface.params.wave_speed, 0.0f, 4.0f, "%.2f", false, nullptr, 16);
            manualChanged |= SceneUI::DrawSmartFloat("wv2_turb", "Turbulence Response", &surface.params.wave_strength, 0.0f, 0.5f, "%.3f", false, nullptr, 16);
            manualChanged |= SceneUI::DrawSmartFloat("wv2_scale", "Flow Feature Scale", &surface.params.wave_frequency, 0.05f, 6.0f, "%.2f", false, nullptr, 16);

            bool hasHydrology = false;
            if (surface.flatMesh && surface.flatMesh->geometry) {
                hasHydrology = surface.flatMesh->geometry->has_attribute("river_flow_direction") &&
                               surface.flatMesh->geometry->has_attribute("water_depth") &&
                               surface.flatMesh->geometry->has_attribute("river_flow_speed") &&
                               surface.flatMesh->geometry->has_attribute("river_discharge") &&
                               surface.flatMesh->geometry->has_attribute("river_froude") &&
                               surface.flatMesh->geometry->has_attribute("river_foam_potential");
            }
            ImGui::TextColored(hasHydrology ? ImVec4(0.35f, 1.0f, 0.48f, 1.0f)
                                            : ImVec4(1.0f, 0.55f, 0.20f, 1.0f),
                               hasHydrology ? "Hydrology stream: Bound" : "Hydrology stream: Missing");
            ImGui::TextDisabled("6 analytic bands; depth, discharge, Froude and foam are sampled per vertex.");
        } else {
            manualChanged |= SceneUI::DrawSmartFloat("wv2_speed", "Wave Speed", &surface.params.wave_speed, 0.0f, 4.0f, "%.2f", false, nullptr, 16);
            manualChanged |= SceneUI::DrawSmartFloat("wv2_strength", "Wave Height Response", &surface.params.wave_strength, 0.0f, 2.0f, "%.3f", false, nullptr, 16);
            manualChanged |= SceneUI::DrawSmartFloat("wv2_frequency", "Wave Frequency", &surface.params.wave_frequency, 0.02f, 8.0f, "%.2f", false, nullptr, 16);
            float degrees = surface.params.fft_wind_direction * 180.0f / 3.14159265f;
            if (SceneUI::DrawSmartFloat("wv2_direction", "Travel Direction", &degrees, 0.0f, 360.0f, "%.0f deg", false, nullptr, 16)) {
                surface.params.fft_wind_direction = degrees * 3.14159265f / 180.0f;
                manualChanged = true;
            }
        }
    }

    if (beginWaterV2Section("Wave Deform Modifier", ImVec4(0.34f, 0.72f, 1.0f, 1.0f), false)) {
        const bool wasEnabled = surface.params.use_geometric_waves;
        if (ImGui::Checkbox("Enable Geometry Deform", &surface.params.use_geometric_waves)) {
            geometryChanged = true;
        }
        ImGui::TextDisabled("Topology stays fixed; Vulkan RT uses flat-mesh BLAS refit.");
        ImGui::TextDisabled("Animation follows the Water panel Time Source.");

        if (surface.params.use_geometric_waves) {
            static const char* spectrumNames[] = {
                "Perlin", "FBM", "Ridge", "Voronoi", "Billow", "Gerstner", "Tessendorf Simple"
            };
            int spectrum = static_cast<int>(surface.params.geo_noise_type);
            if (ImGui::Combo("Spectrum", &spectrum, spectrumNames, IM_ARRAYSIZE(spectrumNames))) {
                surface.params.geo_noise_type = static_cast<WaterWaveParams::NoiseType>(spectrum);
                geometryChanged = true;
            }
            geometryChanged |= SceneUI::DrawSmartFloat("wv3_geo_height", "Height", &surface.params.geo_wave_height,
                                                        0.0f, 20.0f, "%.3f m", false, nullptr, 16);
            geometryChanged |= SceneUI::DrawSmartFloat("wv3_geo_scale", "Wavelength", &surface.params.geo_wave_scale,
                                                        0.1f, 500.0f, "%.2f m", false, nullptr, 16);
            geometryChanged |= SceneUI::DrawSmartFloat("wv3_geo_chop", "Choppiness", &surface.params.geo_wave_choppiness,
                                                        0.0f, 3.0f, "%.2f", false, nullptr, 16);
            geometryChanged |= SceneUI::DrawSmartFloat("wv3_geo_speed", "Speed", &surface.params.geo_wave_speed,
                                                        0.0f, 5.0f, "%.2f", false, nullptr, 16);
            if (ImGui::TreeNodeEx("Spectrum Detail", ImGuiTreeNodeFlags_SpanAvailWidth)) {
                geometryChanged |= ImGui::SliderInt("Octaves", &surface.params.geo_octaves, 1, 8);
                geometryChanged |= SceneUI::DrawSmartFloat("wv3_geo_persist", "Persistence", &surface.params.geo_persistence,
                                                            0.05f, 0.95f, "%.2f", false, nullptr, 16);
                geometryChanged |= SceneUI::DrawSmartFloat("wv3_geo_lacun", "Lacunarity", &surface.params.geo_lacunarity,
                                                            1.1f, 4.0f, "%.2f", false, nullptr, 16);
                geometryChanged |= SceneUI::DrawSmartFloat("wv3_geo_align", "Alignment", &surface.params.geo_alignment,
                                                            0.0f, 1.0f, "%.2f", false, nullptr, 16);
                geometryChanged |= SceneUI::DrawSmartFloat("wv3_geo_dir", "Direction", &surface.params.geo_swell_direction,
                                                            0.0f, 360.0f, "%.1f deg", false, nullptr, 16);
                geometryChanged |= ImGui::Checkbox("Smooth Normals", &surface.params.geo_smooth_normals);
                ImGui::TreePop();
            }
        }

        if (wasEnabled && !surface.params.use_geometric_waves) {
            surface.animate_mesh = false;
        }
    }

    if (beginWaterV2Section("Fine Surface", ImVec4(0.95f, 0.58f, 0.12f, 1.0f), false)) {
        manualChanged |= SceneUI::DrawSmartFloat("wv2_micro", "Ripple Strength", &surface.params.micro_detail_strength, 0.0f, 0.25f, "%.3f", false, nullptr, 16);
        manualChanged |= SceneUI::DrawSmartFloat("wv2_micro_scale", "Ripple Scale", &surface.params.micro_detail_scale, 0.5f, 80.0f, "%.2f", false, nullptr, 16);
        manualChanged |= SceneUI::DrawSmartFloat("wv2_micro_speed", "Ripple Drift", &surface.params.micro_anim_speed, 0.0f, 2.0f, "%.3f", false, nullptr, 16);
    }

    if (beginWaterV2Section("Whitewater", ImVec4(0.88f, 0.92f, 0.96f, 1.0f), false)) {
        manualChanged |= SceneUI::DrawSmartFloat("wv2_foam", "Hydraulic Foam Gain", &surface.params.foam_level, 0.0f, 1.0f, "%.2f", false, nullptr, 16);
        manualChanged |= SceneUI::DrawSmartFloat("wv2_bank_foam", "Bank Foam", &surface.params.shore_foam_intensity, 0.0f, 1.0f, "%.2f", false, nullptr, 16);
        manualChanged |= SceneUI::DrawSmartFloat("wv2_foam_break", "Breakup Scale", &surface.params.foam_noise_scale, 0.2f, 30.0f, "%.2f", false, nullptr, 16);
        manualChanged |= SceneUI::DrawSmartFloat("wv2_foam_threshold", "Crest Threshold", &surface.params.foam_threshold, 0.0f, 1.0f, "%.2f", false, nullptr, 16);
    }

    if (beginWaterV2Section("Caustics", ImVec4(0.20f, 0.84f, 0.68f, 1.0f), false)) {
        manualChanged |= SceneUI::DrawSmartFloat("wv2_caustic", "Intensity", &surface.params.caustic_intensity, 0.0f, 2.0f, "%.2f", false, nullptr, 16);
        manualChanged |= SceneUI::DrawSmartFloat("wv2_caustic_scale", "Pattern Scale", &surface.params.caustic_scale, 0.1f, 20.0f, "%.2f", false, nullptr, 16);
        manualChanged |= SceneUI::DrawSmartFloat("wv2_caustic_speed", "Drift Speed", &surface.params.caustic_speed, 0.0f, 5.0f, "%.2f", false, nullptr, 16);
    }

    if (manualChanged || presetApplied) {
        if (manualChanged) surface.params.current_preset = WaterWaveParams::WaterPreset::Custom;
        syncWaterV2Material(ctx, surface);
        ProjectManager::getInstance().markModified();
    }

    if (geometryChanged) {
        auto& waterManager = WaterManager::getInstance();
        if (surface.params.use_geometric_waves) {
            if (surface.original_positions.empty()) waterManager.cacheOriginalPositions(&surface);
            surface.animate_mesh = true;
            // Native Vulkan refit is the production path. Do not reactivate the
            // legacy CUDA -> CPU download bridge from this modifier UI.
            surface.use_gpu_animation = false;
            waterManager.updateAnimatedWaterMesh(&surface, surface.animation_time);
        } else {
            waterManager.restoreAnimatedWaterMesh(&surface);
        }
        g_cpu_bvh_refit_pending = true;
        ctx.renderer.resetCPUAccumulation();
        if (Backend::IBackend* backend = waterV2Backend(ctx)) backend->resetAccumulation();
        ProjectManager::getInstance().markModified();
    }

    if (allowDelete) {
        ImGui::Spacing();
        ImGui::Separator();
        if (UIWidgets::DangerButton("Delete Water Surface", ImVec2(UIWidgets::GetInspectorActionWidth(), 0.0f))) {
            const bool ownsSceneMesh = surface.owns_scene_mesh;
            if (!ownsSceneMesh) {
                WaterManager::getInstance().restoreAnimatedWaterMesh(&surface);
                auto stackIt = ctx.scene.mesh_modifiers.find(surface.name);
                if (stackIt != ctx.scene.mesh_modifiers.end()) {
                    auto& modifiers = stackIt->second.modifiers;
                    modifiers.erase(
                        std::remove_if(modifiers.begin(), modifiers.end(),
                            [](const MeshModifiers::ModifierData& modifier) {
                                return modifier.type == MeshModifiers::ModifierType::WaterSurface;
                            }),
                        modifiers.end());
                }
            }
            WaterManager::getInstance().removeWaterSurface(ctx.scene, surface.id);
            if (ownsSceneMesh) rebuildWaterV2Geometry(ctx);
            ProjectManager::getInstance().markModified();
            return true;
        }
    }
    return false;
}

} // namespace

bool SceneUI::drawWaterSurfaceMaterialEditor(UIContext& ctx, WaterSurface& surface, bool allow_delete) {
    ImGui::PushID(surface.id);
    const bool removed = drawWaterV2Editor(ctx, surface, allow_delete);
    ImGui::PopID();
    return removed;
}

void SceneUI::drawWaterPanel(UIContext& ctx) {
    ImGui::TextColored(ImVec4(0.25f, 0.78f, 1.0f, 1.0f), "WATER V3 / VULKAN RT");
    ImGui::TextDisabled("Hydrology mesh + material shading + optional topology-stable Wave Deform.");
    ImGui::Separator();

    // Reuse the renderer's reset-free AOV visualizer for the first Water V3
    // diagnostics. These modes inspect the actual RT path, not a second preview
    // shader, so look-dev and final rendering cannot silently diverge.
    static const char* diagnosticNames[] = {
        "Beauty", "Resolved Normal", "Depth", "Transmission", "Absorption"
    };
    static const int diagnosticViews[] = { 0, 10, 12, 7, 8 };
    int diagnosticIndex = 0;
    for (int i = 1; i < IM_ARRAYSIZE(diagnosticViews); ++i) {
        if (ctx.render_settings.debug_view == diagnosticViews[i]) diagnosticIndex = i;
    }
    ImGui::SetNextItemWidth(-1.0f);
    if (ImGui::Combo("Water Diagnostic", &diagnosticIndex,
                     diagnosticNames, IM_ARRAYSIZE(diagnosticNames))) {
        ctx.render_settings.debug_view = diagnosticViews[diagnosticIndex];
    }

    static float planeSize = 20.0f;
    SceneUI::DrawSmartFloat("wv2_plane_size", "Ocean Plane Size", &planeSize, 2.0f, 1000.0f, "%.1f m", false, nullptr, 16);
    if (UIWidgets::PrimaryButton("Add Ocean Surface", ImVec2(UIWidgets::GetInspectorActionWidth(), 30.0f))) {
        WaterSurface* created = WaterManager::getInstance().createWaterPlane(ctx.scene, Vec3(0.0f), planeSize, 3.0f);
        if (created) {
            created->params.applyPreset(WaterWaveParams::WaterPreset::CalmOcean);
            WaterManager::getInstance().syncSurfaceMaterial(created);
        }
        rebuildWaterV2Geometry(ctx);
        ProjectManager::getInstance().markModified();
    }

    auto& surfaces = WaterManager::getInstance().getWaterSurfaces();
    if (surfaces.empty()) {
        ImGui::Spacing();
        ImGui::TextDisabled("No water surfaces in the scene.");
        return;
    }

    if (!WaterManager::getInstance().getWaterSurface(selected_water_surface_id)) {
        selected_water_surface_id = surfaces.front().id;
    }
    ImGui::Spacing();
    if (ImGui::BeginListBox("##water_v2_surfaces", ImVec2(-1.0f, 92.0f))) {
        for (const WaterSurface& surface : surfaces) {
            const bool selected = selected_water_surface_id == surface.id;
            if (ImGui::Selectable(surface.name.c_str(), selected)) {
                selected_water_surface_id = surface.id;
                selectManagedMesh(ctx, surface.flatMesh);
            }
            if (selected) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndListBox();
    }

    const char* timeModes[] = { "Realtime", "Timeline", "Static" };
    int timeMode = static_cast<int>(WaterManager::getInstance().getPreviewTimeMode());
    if (ImGui::Combo("Time Source", &timeMode, timeModes, IM_ARRAYSIZE(timeModes))) {
        WaterManager::getInstance().setPreviewTimeMode(static_cast<WaterPreviewTimeMode>(timeMode));
        ctx.renderer.resetCPUAccumulation();
        if (Backend::IBackend* backend = waterV2Backend(ctx)) backend->resetAccumulation();
        ProjectManager::getInstance().markModified();
    }

    if (WaterSurface* selected = WaterManager::getInstance().getWaterSurface(selected_water_surface_id)) {
        ImGui::Separator();
        if (drawWaterSurfaceMaterialEditor(ctx, *selected, true)) selected_water_surface_id = -1;
    }
}
