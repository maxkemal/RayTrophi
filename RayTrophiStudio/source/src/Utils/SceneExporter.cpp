#include "SceneExporter.h"
#include "GltfDirectWriter.h"
#include "scene_data.h"
#include "Triangle.h"
#include "TriangleMesh.h"   // flat SoA scatter sources in computeExportEstimate
#include "MaterialManager.h"
#include "PrincipledBSDF.h"
#include "Texture.h"
#include "InstanceManager.h"
#include "HittableInstance.h"
#include "globals.h"
#include "TerrainManager.h"
#include "imgui.h"
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <thread>
#include <future>
#include <iostream>
#include <filesystem>

// Light Headers
#include "Light.h"
#include "PointLight.h"
#include "DirectionalLight.h"
#include "SpotLight.h"
#include "json.hpp"

// STB Image Write for embedded textures
// #define STB_IMAGE_WRITE_IMPLEMENTATION // Already defined in Main.cpp
#include "stb_image_write.h"
#include <ui_modern.h>
#include <fstream>
#include <cstdint>
#include <cmath>

struct TerrainBakeResult {
    std::shared_ptr<Material> material;
    std::shared_ptr<Texture> baseColor;
    std::shared_ptr<Texture> normal;
    std::shared_ptr<Texture> roughness;
    std::shared_ptr<Texture> metallic;
};

using json = nlohmann::json;

namespace {
uint8_t toByte(float value) {
    return static_cast<uint8_t>(std::clamp(value, 0.0f, 1.0f) * 255.0f + 0.5f);
}

uint8_t linearToSrgbByte(float value) {
    value = std::clamp(value, 0.0f, 1.0f);
    const float srgb = (value <= 0.0031308f)
        ? (value * 12.92f)
        : (1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f);
    return static_cast<uint8_t>(std::clamp(srgb, 0.0f, 1.0f) * 255.0f + 0.5f);
}

Vec3 sampleTerrainMaskRGBA(const TerrainObject& terrain, float u, float v, float& outA) {
    if (!terrain.splatMap) {
        outA = 0.0f;
        return Vec3(1.0f, 0.0f, 0.0f);
    }
    Vec3 rgb = terrain.splatMap->get_color_bilinear(u, v);
    outA = terrain.splatMap->get_alpha_bilinear(u, v);
    return rgb;
}

std::shared_ptr<Texture> makeBakedTexture(const std::string& name,
                                          int width,
                                          int height,
                                          TextureType type,
                                          const std::vector<unsigned char>& rgba) {
    return std::make_shared<Texture>(width, height, 4, rgba, type, name);
}

TerrainBakeResult bakeTerrainMaterialForExport(const TerrainObject& terrain,
                                               uint16_t materialId,
                                               int resolution) {
    TerrainBakeResult result;
    if (!terrain.splatMap || terrain.layers.empty()) {
        return result;
    }

    // Keep splat resolution aligned with the terrain grid before sampling so
    // height-derived layer masks land on the same texel space during export.
    TerrainManager::getInstance().resizeSplatMap(const_cast<TerrainObject*>(&terrain));

    const int bakeResolution = (std::max)(128, resolution);
    std::vector<unsigned char> baseColorPixels;
    std::vector<unsigned char> normalPixels;
    std::vector<unsigned char> roughnessPixels;
    std::vector<unsigned char> metallicPixels;
    baseColorPixels.resize(static_cast<size_t>(bakeResolution) * bakeResolution * 4);
    normalPixels.resize(static_cast<size_t>(bakeResolution) * bakeResolution * 4);
    roughnessPixels.resize(static_cast<size_t>(bakeResolution) * bakeResolution * 4);
    metallicPixels.resize(static_cast<size_t>(bakeResolution) * bakeResolution * 4);

    for (int y = 0; y < bakeResolution; ++y) {
        for (int x = 0; x < bakeResolution; ++x) {
            const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(bakeResolution);
            const float v = 1.0f - (static_cast<float>(y) + 0.5f) / static_cast<float>(bakeResolution);

            float maskA = 0.0f;
            Vec3 maskRgb = sampleTerrainMaskRGBA(terrain, u, v, maskA);
            float weights[4] = {
                static_cast<float>(maskRgb.x),
                static_cast<float>(maskRgb.y),
                static_cast<float>(maskRgb.z),
                maskA
            };

            Vec3 blendedAlbedo(0.0f);
            Vec3 blendedNormal(0.0f, 0.0f, 1.0f);
            float blendedRoughness = 0.5f;
            float blendedMetallic = 0.0f;
            float totalWeight = 0.0f;
            bool hasNormal = false;

            for (int i = 0; i < 4 && i < static_cast<int>(terrain.layers.size()); ++i) {
                const float weight = weights[i];
                if (weight < 0.001f || !terrain.layers[i]) {
                    continue;
                }

                auto* layer = dynamic_cast<PrincipledBSDF*>(terrain.layers[i].get());
                if (!layer) {
                    continue;
                }

                const float scale = (i < static_cast<int>(terrain.layer_uv_scales.size()))
                    ? terrain.layer_uv_scales[i]
                    : 1.0f;
                Vec2 layerUv(u * scale, v * scale);

                blendedAlbedo = blendedAlbedo + layer->getPropertyValue(layer->albedoProperty, layerUv) * weight;
                blendedRoughness += layer->getPropertyValue(layer->roughnessProperty, layerUv).y * weight;
                blendedMetallic += layer->getPropertyValue(layer->metallicProperty, layerUv).z * weight;

                if (layer->has_normal_map()) {
                    Vec3 ns = layer->get_normal_from_map(layerUv.u, layerUv.v) * 2.0 - Vec3(1.0);
                    ns.x = -ns.x;
                    ns.y = -ns.y;
                    blendedNormal = blendedNormal + ns * weight;
                    hasNormal = true;
                }

                totalWeight += weight;
            }

            if (totalWeight > 0.0f) {
                const float invWeight = 1.0f / totalWeight;
                blendedAlbedo = blendedAlbedo * invWeight;
                blendedRoughness *= invWeight;
                blendedMetallic *= invWeight;
                if (hasNormal) {
                    blendedNormal = blendedNormal * invWeight;
                }
            } else {
                blendedAlbedo = Vec3(0.5f, 0.5f, 0.5f);
                blendedRoughness = 0.5f;
                blendedMetallic = 0.0f;
                blendedNormal = Vec3(0.0f, 0.0f, 1.0f);
            }

            if (hasNormal) {
                blendedNormal = blendedNormal.normalize();
            }

            if (terrain.surfaceSemanticMap && terrain.surfaceSemanticMap->is_loaded()) {
                const Vec3 semanticRgb = terrain.surfaceSemanticMap->get_color_bilinear(u, v);
                const float hardness = terrain.surfaceSemanticMap->get_alpha_bilinear(u, v);
                const float wetness = std::clamp((std::max)(
                    static_cast<float>(semanticRgb.x), static_cast<float>(semanticRgb.y)), 0.0f, 1.0f);
                const float ice = std::clamp(static_cast<float>(semanticRgb.z), 0.0f, 1.0f);
                blendedAlbedo = blendedAlbedo * (1.0f - wetness * 0.28f);
                blendedRoughness = blendedRoughness * (1.0f - wetness * 0.65f) +
                    0.16f * wetness * 0.65f;
                const float iceLuma = static_cast<float>(blendedAlbedo.x * 0.2126 +
                    blendedAlbedo.y * 0.7152 + blendedAlbedo.z * 0.0722);
                const Vec3 iceColor = Vec3(0.70f, 0.82f, 0.88f) * (std::max)(iceLuma, 0.35f);
                blendedAlbedo = blendedAlbedo * (1.0f - ice * 0.55f) + iceColor * (ice * 0.55f);
                blendedRoughness = blendedRoughness * (1.0f - ice * 0.65f) + 0.12f * ice * 0.65f;
                blendedRoughness = std::clamp(blendedRoughness + hardness * 0.035f, 0.0f, 1.0f);
            }

            const size_t idx = (static_cast<size_t>(y) * bakeResolution + x) * 4;
            // Albedo textures are stored as sRGB bytes for export. The sampled
            // layer colors above are already in linear space.
            baseColorPixels[idx + 0] = linearToSrgbByte(static_cast<float>(blendedAlbedo.x));
            baseColorPixels[idx + 1] = linearToSrgbByte(static_cast<float>(blendedAlbedo.y));
            baseColorPixels[idx + 2] = linearToSrgbByte(static_cast<float>(blendedAlbedo.z));
            baseColorPixels[idx + 3] = 255;

            normalPixels[idx + 0] = toByte(static_cast<float>(blendedNormal.x * 0.5 + 0.5));
            normalPixels[idx + 1] = toByte(static_cast<float>(blendedNormal.y * 0.5 + 0.5));
            normalPixels[idx + 2] = toByte(static_cast<float>(blendedNormal.z * 0.5 + 0.5));
            normalPixels[idx + 3] = 255;

            const uint8_t roughByte = toByte(blendedRoughness);
            roughnessPixels[idx + 0] = roughByte;
            roughnessPixels[idx + 1] = roughByte;
            roughnessPixels[idx + 2] = roughByte;
            roughnessPixels[idx + 3] = 255;

            const uint8_t metalByte = toByte(blendedMetallic);
            metallicPixels[idx + 0] = metalByte;
            metallicPixels[idx + 1] = metalByte;
            metallicPixels[idx + 2] = metalByte;
            metallicPixels[idx + 3] = 255;
        }
    }

    const std::string baseName = "TerrainBake_" + terrain.name + "_" + std::to_string(materialId);
    result.baseColor = makeBakedTexture(baseName + "_BaseColor", bakeResolution, bakeResolution, TextureType::Albedo, baseColorPixels);
    result.normal = makeBakedTexture(baseName + "_Normal", bakeResolution, bakeResolution, TextureType::Normal, normalPixels);
    result.roughness = makeBakedTexture(baseName + "_Roughness", bakeResolution, bakeResolution, TextureType::Roughness, roughnessPixels);
    result.metallic = makeBakedTexture(baseName + "_Metallic", bakeResolution, bakeResolution, TextureType::Metallic, metallicPixels);

    auto bakedMaterial = std::make_shared<PrincipledBSDF>();
    bakedMaterial->materialName = baseName + "_Material";
    bakedMaterial->albedoProperty.texture = result.baseColor;
    bakedMaterial->albedoProperty.intensity = 1.0f;
    bakedMaterial->normalProperty.texture = result.normal;
    bakedMaterial->normalProperty.intensity = 1.0f;
    bakedMaterial->roughnessProperty.texture = result.roughness;
    bakedMaterial->roughnessProperty.intensity = 1.0f;
    bakedMaterial->metallicProperty.texture = result.metallic;
    bakedMaterial->metallicProperty.intensity = 1.0f;
    result.material = bakedMaterial;

    return result;
}

} // namespace

bool SceneExporter::drawExportPopup(SceneData& scene) {
    if (!show_export_popup) {
        popup_open_last_frame = false;
        return false;
    }

    // Recompute the size estimate once per popup-open, not every frame - the
    // scan is O(scene object count) and this function redraws every frame
    // while the popup is up.
    if (!popup_open_last_frame) {
        export_estimate = computeExportEstimate(scene, settings);
        confirm_large_export = false;
    }
    popup_open_last_frame = true;

    // Heuristic thresholds for the "this will be slow/RAM-heavy" warning below.
    // Deliberately conservative - see computeExportEstimate's own comment.
    constexpr double kWarnPeakMb = 2048.0;
    constexpr size_t kWarnTriangles = 3'000'000;
    constexpr size_t kWarnUncollapsedInstances = 50'000;

    bool trigger_export = false;

    if (ImGui::Begin("Export Settings", &show_export_popup, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("Configuration");
        ImGui::Separator();

        ImGui::Checkbox("Geometry", &settings.export_geometry);
        ImGui::Checkbox("Materials", &settings.export_materials);

        // Added Cameras and Lights options
        ImGui::Checkbox("Cameras", &settings.export_cameras);
        ImGui::Checkbox("Lights", &settings.export_lights);

        ImGui::Separator();
        ImGui::Checkbox("Selected Objects Only", &settings.export_selected_only);
        UIWidgets::HelpMarker("Export only the currently selected objects.");

        if (export_estimate.instance_count > 0) {
            ImGui::Separator();
            ImGui::Text("Scattered / Instanced Objects");
            ImGui::Checkbox("Export instances as shared references (recommended)", &settings.use_gpu_instancing_extension);
            UIWidgets::HelpMarker(
                "Scattered/foliage objects that share one source mesh are written ONCE and referenced "
                "by every instance (EXT_mesh_gpu_instancing) instead of getting a full node entry each. "
                "Turn off only if the target tool can't read that extension - file size will then scale "
                "with instance count instead of unique source count.");
            ImGui::Text("  %zu instance(s), %zu unique source mesh(es), %zu source triangle(s)",
                export_estimate.instance_count, export_estimate.unique_instance_sources,
                export_estimate.instance_triangle_count);
            UIWidgets::HelpMarker(
                "Counted from InstanceManager - the same place the writer reads - not from "
                "world.objects. On the Vulkan path scatter never enters world.objects, so a "
                "count taken from there reads 0 for a scene that exports thousands.");
        }

        if (TerrainManager::getInstance().hasActiveTerrain()) {
            ImGui::Separator();
            ImGui::Text("Terrain");
            ImGui::Checkbox("Bake Terrain Layer Materials", &settings.bake_terrain_materials);
            if (settings.bake_terrain_materials) {
                ImGui::SliderInt("Terrain Bake Resolution", &settings.terrain_bake_resolution, 256, 4096);
                UIWidgets::HelpMarker("Bakes splat-blended terrain layers into export textures and embeds them into the GLB.");
            } else {
                ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.2f, 1.0f),
                    "Layered terrain materials will not be preserved in GLB without baking.");
            }
        }

        ImGui::Separator();
        ImGui::Text("Animation Support");
        ImGui::Checkbox("Export Skeleton (Bones)", &settings.export_skinning);
        ImGui::Checkbox("Export Animation Clips", &settings.export_animations);
        if (settings.export_animations && !settings.export_skinning) {
            ImGui::TextColored(ImVec4(1,1,0,1), "Warning: Animations usually require Skeleton!");
        }

        ImGui::Separator();
        ImGui::Text("Format");
        static int format_idx = 0; // 0: binary, 1: text
        if (settings.binary_mode) format_idx = 0; else format_idx = 1;
        
        if (ImGui::Combo("Type", &format_idx, "GLTF Binary (.glb)\0GLTF Text (.gltf)\0")) {
            settings.binary_mode = (format_idx == 0);
        }

        ImGui::Separator();
        ImGui::Text("Estimated export size");
        const size_t totalTriangles = export_estimate.triangle_count + export_estimate.instance_triangle_count;
        ImGui::Text("  %zu object(s), %zu triangle(s)", export_estimate.object_count, totalTriangles);
        ImGui::Text("  Rough peak memory (this export only, not the final file): ~%.0f MB", export_estimate.estimated_peak_mb);
        UIWidgets::HelpMarker(
            "A deliberately conservative estimate, not a guarantee. Ignores selection/geometry "
            "checkboxes above; assumes everything currently in the scene gets exported.");

        const bool isLarge = export_estimate.estimated_peak_mb > kWarnPeakMb
            || totalTriangles > kWarnTriangles
            || (!settings.use_gpu_instancing_extension && export_estimate.instance_count > kWarnUncollapsedInstances);

        if (isLarge) {
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f),
                "Large/crowded scene - this export may take a while and use significant RAM.");
            ImGui::Checkbox("I understand, export anyway", &confirm_large_export);
        } else {
            confirm_large_export = true;
        }

        ImGui::Separator();
        ImGui::BeginDisabled(!confirm_large_export);
        if (ImGui::Button("Select File & Export", ImVec2(-1, 0))) {
            trigger_export = true;
            show_export_popup = false; // Close popup
        }
        ImGui::EndDisabled();

        ImGui::End();
    }
    return trigger_export;
}


SceneExporter::ExportEstimate SceneExporter::computeExportEstimate(SceneData& scene, const ExportSettings& settings) {
    ExportEstimate est;
    est.computed = true;

    std::unordered_set<const void*> uniqueSources;
    for (const auto& obj : scene.world.objects) {
        if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
            est.object_count++;
            est.triangle_count++;
            est.legacy_triangle_count++;
        }
        else if (auto mesh = std::dynamic_pointer_cast<TriangleMesh>(obj)) {
            if (!mesh->geometry) continue;
            est.object_count++;
            est.triangle_count += mesh->geometry->indices.size() / 3;
        }
        else if (auto inst = std::dynamic_pointer_cast<HittableInstance>(obj)) {
            if (!inst->source_triangles || inst->source_triangles->empty()) continue;
            est.object_count++;
            est.instance_count++;
            if (uniqueSources.insert(inst->source_triangles.get()).second) {
                est.unique_instance_sources++;
                est.instance_triangle_count += inst->source_triangles->size();
                // Legacy facade source: the writer builds this one in RAM.
                est.materialised_instance_triangles += inst->source_triangles->size();
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // ★ Scatter / foliage read from InstanceManager - the SAME source
    // GltfWriter::collectScatter() reads, and NOT a duplicate of the loop above.
    //
    // On the Vulkan backends SceneUI::syncInstancesToScene() returns early and
    // never expands scatter into world.objects, so the loop above sees nothing.
    // The writer was fixed for this on 2026-09-04; this estimate was not, and a
    // panel that reports "0 instances" for an export that then writes a thousand
    // of them is the failure mode nobody files: the number looks plausible.
    //
    // Transform convention and the transient-group exclusion are mirrored from
    // collectScatter so the two counts cannot drift apart silently.
    // ─────────────────────────────────────────────────────────────────────────
    for (const InstanceGroup& group : InstanceManager::getInstance().getGroups()) {
        if (group.transient) continue;   // runtime particle bridge, never exported
        if (group.instances.empty() || group.sources.empty()) continue;

        for (size_t si = 0; si < group.sources.size(); ++si) {
            const ScatterSource& source = group.sources[si];

            size_t placements = 0;
            for (const auto& inst : group.instances) {
                int idx = inst.source_index;
                if (idx < 0 || idx >= (int)group.sources.size()) idx = 0;
                if ((size_t)idx == si) placements++;
            }
            if (placements == 0) continue;

            if (!source.flat_meshes.empty()) {
                for (const auto& mesh : source.flat_meshes) {
                    if (!mesh || !mesh->geometry || mesh->geometry->indices.empty()) continue;
                    // One node per (source mesh) x (placement), exactly as the
                    // writer emits it - a multi-mesh source multiplies both.
                    est.object_count += placements;
                    est.instance_count += placements;
                    if (uniqueSources.insert(mesh.get()).second) {
                        est.unique_instance_sources++;
                        // Flat SoA: streamed, so it does NOT feed
                        // materialised_instance_triangles.
                        est.instance_triangle_count += mesh->geometry->indices.size() / 3;
                    }
                }
            } else if (source.centered_triangles_ptr && !source.centered_triangles_ptr->empty()) {
                est.object_count += placements;
                est.instance_count += placements;
                if (uniqueSources.insert(source.centered_triangles_ptr.get()).second) {
                    est.unique_instance_sources++;
                    est.instance_triangle_count += source.centered_triangles_ptr->size();
                    est.materialised_instance_triangles += source.centered_triangles_ptr->size();
                }
            }
        }
    }

    // Peak RAM is no longer a function of triangle count for flat SoA meshes:
    // the writer streams their existing position/normal/uv arrays straight to
    // disk (see GltfDirectWriter.h), so they cost ZERO extra heap. Only the
    // legacy Triangle facade and LEGACY-FACADE scatter sources are materialised,
    // at roughly 44 bytes per triangle of interleaved vertex + index data,
    // counted once per unique source. A flat SoA scatter source streams like any
    // other flat mesh, so it is excluded here even though it is instanced.
    //
    // est.triangle_count still drives the popup's "this is a big export"
    // warning, because file size and write time do scale with it.
    const double materialisedTriangles =
        static_cast<double>(est.legacy_triangle_count + est.materialised_instance_triangles);
    est.estimated_peak_mb = (materialisedTriangles * 44.0) / (1024.0 * 1024.0);

    return est;
}

bool SceneExporter::exportScene(const std::string& filepath, SceneData& scene, const ExportSettings& settings,
                     const std::vector<std::shared_ptr<Hittable>>& selected_objects) {
    if (filepath.empty()) {
        SCENE_LOG_ERROR("Export failed: Filepath is empty.");
        return false;
    }

    SCENE_LOG_INFO("Starting glTF export to: " + filepath);
    is_exporting = true;
    current_export_status = "Baking terrain materials...";

    // Terrain layers are flattened to ordinary textures first; from the writer's
    // point of view they are just material overrides keyed by material id.
    std::map<uint16_t, std::shared_ptr<Material>> exportMaterialOverrides;
    if (settings.bake_terrain_materials) {
        auto& terrains = TerrainManager::getInstance().getTerrains();
        for (const auto& terrain : terrains) {
            if (terrain.material_id == MaterialManager::INVALID_MATERIAL_ID) continue;
            if (!terrain.splatMap || terrain.layers.empty()) continue;

            TerrainBakeResult bakeResult = bakeTerrainMaterialForExport(
                terrain, terrain.material_id, settings.terrain_bake_resolution);
            if (bakeResult.material) {
                exportMaterialOverrides[terrain.material_id] = bakeResult.material;
            }
        }
    }

    current_export_status = "Writing glTF...";

    // The scene is written straight to disk from the geometry that already
    // exists in memory - no intermediate aiScene, no per-triangle facade, no
    // per-face heap allocation. See GltfDirectWriter.h for why the previous
    // Assimp round trip cost minutes and ~15 GB on a single-giant-mesh scene.
    std::string error;
    rtgltf::WriteStats stats;
    const bool ok = rtgltf::writeScene(filepath, scene, settings, selected_objects,
                                       exportMaterialOverrides, stats, error);
    last_stats = stats;
    last_error = ok ? std::string() : error;

    auto mb = [](uint64_t bytes) {
        return std::to_string((double)bytes / (1024.0 * 1024.0)).substr(0, 8);
    };

    SCENE_LOG_INFO("[Export] " + std::to_string(stats.mesh_count) + " meshes, "
        + std::to_string(stats.primitive_count) + " primitives, "
        + std::to_string(stats.triangle_count) + " triangles, "
        + std::to_string(stats.vertex_count) + " vertices, "
        + std::to_string(stats.instance_count) + " instances in "
        + std::to_string(stats.instanced_group_count) + " EXT_mesh_gpu_instancing node(s), "
        + std::to_string(stats.material_count) + " materials, "
        + std::to_string(stats.image_count) + " images.");
    SCENE_LOG_INFO("[Export] file=" + mb(stats.file_bytes) + " MB, writer peak heap="
        + mb((uint64_t)(stats.peak_writer_mb * 1024.0 * 1024.0)) + " MB, total="
        + std::to_string(stats.seconds_total) + "s (collect=" + std::to_string(stats.seconds_collect)
        + " materials=" + std::to_string(stats.seconds_materials)
        + " plan=" + std::to_string(stats.seconds_plan)
        + " write=" + std::to_string(stats.seconds_write) + ")");

    if (!ok) {
        SCENE_LOG_ERROR("[Export] FAILED: " + error);
        current_export_status = "Export failed: " + error;
        is_exporting = false;
        return false;
    }

    SCENE_LOG_INFO("Scene exported successfully to: " + filepath);
    current_export_status = "Done";
    is_exporting = false;
    return true;
}
