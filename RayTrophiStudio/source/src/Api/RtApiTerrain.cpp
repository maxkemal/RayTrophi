/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          Api/RtApiTerrain.cpp
 * Author:        Kemal Demirtas
 * License:       MIT
 * =========================================================================
 */

#include "RtApiInternal.h"
#include "TerrainManager.h"
#include "TerrainNodesV2.h"
#include "TerrainSystem.h"
#include "TerrainSatMapPresetLibrary.h"
// Layer slot binding resolves material names and marks the project dirty.
#include "MaterialManager.h"
#include "ProjectManager.h"
#include "Texture.h"
#include "RiverSpline.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <unordered_map>

namespace rtapi {
namespace {

struct TerrainEvaluationRecord {
    std::string state = "idle";
    float progress = 0.0f;
    unsigned int current_node_id = 0;
    std::string error;
    bool cancel_requested = false;
};

std::unordered_map<std::string, TerrainEvaluationRecord> g_terrain_evaluations;

struct TerrainSnapshot {
    Heightmap heightmap;
    std::vector<float> hardness;
    std::vector<float> flow;
    std::vector<float> erosion;
    std::vector<float> original_height;
};

TerrainSnapshot captureTerrainSnapshot(const TerrainObject& terrain) {
    return {terrain.heightmap, terrain.hardnessMap, terrain.flowMap,
            terrain.erosionMapRGBA, terrain.original_heightmap_data};
}

class TerrainSnapshotCommand final : public SceneCommand {
public:
    TerrainSnapshotCommand(std::string terrain_name, std::string description,
                           TerrainSnapshot before, TerrainSnapshot after)
        : terrain_name_(std::move(terrain_name)), description_(std::move(description)),
          before_(std::move(before)), after_(std::move(after)) {}
    void execute(UIContext& ctx) override { apply(ctx, after_); }
    void undo(UIContext& ctx) override { apply(ctx, before_); }
    Type getType() const override { return Type::Heavy; }
    std::string getDescription() const override { return description_; }
    bool isHeavyGeometry() const override { return true; }
private:
    void apply(UIContext& ctx, const TerrainSnapshot& snapshot) {
        TerrainObject* terrain = TerrainManager::getInstance().getTerrainByName(terrain_name_);
        if (!terrain) return;
        terrain->heightmap = snapshot.heightmap;
        terrain->hardnessMap = snapshot.hardness;
        terrain->flowMap = snapshot.flow;
        terrain->erosionMapRGBA = snapshot.erosion;
        terrain->original_heightmap_data = snapshot.original_height;
        TerrainManager::getInstance().updateTerrainMesh(terrain, false);
        ui.mesh_cache_valid = false;
        scheduleSceneMutationRebuilds(ctx, true);
    }
    std::string terrain_name_;
    std::string description_;
    TerrainSnapshot before_;
    TerrainSnapshot after_;
};

TerrainInfo terrainInfo(const TerrainObject& terrain) {
    TerrainInfo info;
    info.id = terrain.id;
    info.name = terrain.name;
    info.width = terrain.heightmap.width;
    info.height = terrain.heightmap.height;
    info.size = terrain.heightmap.scale_xz;
    info.height_scale = terrain.heightmap.scale_y;
    info.mesh_resolution = terrain.mesh_resolution;
    info.mesh_width = terrain.meshGridWidth();
    info.mesh_height = terrain.meshGridHeight();
    info.paint_resolution = terrain.paint_resolution;
    info.paint_width = terrain.paintGridWidth();
    info.paint_height = terrain.paintGridHeight();
    info.has_surface_semantic = terrain.surfaceSemanticMap &&
        terrain.surfaceSemanticMap->is_loaded();
    info.has_node_graph = static_cast<bool>(terrain.nodeGraph);
    info.dirty = terrain.dirty_mesh;
    return info;
}

std::string uniqueTerrainName(const std::string& requested) {
    const std::string base = requested.empty() ? "Terrain" : requested;
    std::string candidate = base;
    int suffix = 1;
    auto& manager = TerrainManager::getInstance();
    while (manager.getTerrainByName(candidate) || objectExists(candidate)) {
        char buffer[32];
        std::snprintf(buffer, sizeof(buffer), ".%03d", suffix++);
        candidate = base + buffer;
    }
    return candidate;
}

TerrainEvaluationInfo evaluationInfo(const std::string& name,
                                     const TerrainEvaluationRecord& record) {
    TerrainEvaluationInfo info;
    info.terrain_name = name;
    info.state = record.state;
    info.progress = record.progress;
    info.current_node_id = record.current_node_id;
    info.error = record.error;
    return info;
}

std::string lowerCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

} // namespace

Result listTerrains(std::vector<TerrainInfo>& out_terrains) {
    if (!g_ctx) return notBound();
    out_terrains.clear();
    const auto& terrains = TerrainManager::getInstance().getTerrains();
    out_terrains.reserve(terrains.size());
    for (const auto& terrain : terrains) out_terrains.push_back(terrainInfo(terrain));
    return Result::success();
}

Result getTerrain(const std::string& terrain_name, TerrainInfo& out_info) {
    if (!g_ctx) return notBound();
    if (terrain_name.empty()) return Result::fail("terrain name must not be empty");
    TerrainObject* terrain = TerrainManager::getInstance().getTerrainByName(terrain_name);
    if (!terrain) return Result::fail("terrain not found: " + terrain_name);
    out_info = terrainInfo(*terrain);
    return Result::success();
}

Result listTerrainLayers(const std::string& terrain_name,
                         std::vector<TerrainLayerInfo>& out_layers) {
    out_layers.clear();
    if (!g_ctx) return notBound();
    TerrainObject* terrain = TerrainManager::getInstance().getTerrainByName(terrain_name);
    if (!terrain) return Result::fail("terrain not found: " + terrain_name);

    // Measure the semantic map once so every overlay slot can report whether
    // its drive channel actually carries anything. A slot bound to a channel
    // the graph never fills renders nothing and says nothing about it.
    bool semanticMeasured = false;
    constexpr int kSem = TerrainObject::kSemanticLayerSlots;
    float channelMin[kSem] = {0.0f, 0.0f, 0.0f, 0.0f};
    float channelMax[kSem] = {0.0f, 0.0f, 0.0f, 0.0f};
    float channelMean[kSem] = {0.0f, 0.0f, 0.0f, 0.0f};
    float channelCoverage[kSem] = {0.0f, 0.0f, 0.0f, 0.0f};
    bool channelConstant[kSem] = {false, false, false, false};
    if (terrain->surfaceSemanticMap && terrain->surfaceSemanticMap->is_loaded()) {
        const auto& pixels = terrain->surfaceSemanticMap->pixels;
        if (!pixels.empty()) {
            semanticMeasured = true;
            double sum[kSem] = {0, 0, 0, 0};
            size_t covered[kSem] = {0, 0, 0, 0};
            uint8_t peak[kSem] = {0, 0, 0, 0};
            uint8_t trough[kSem] = {255, 255, 255, 255};
            for (const auto& p : pixels) {
                const uint8_t ch[kSem] = {p.r, p.g, p.b, p.a};
                for (int c = 0; c < kSem; ++c) {
                    sum[c] += ch[c];
                    if (ch[c] > peak[c]) peak[c] = ch[c];
                    if (ch[c] < trough[c]) trough[c] = ch[c];
                    if (ch[c] > 0) ++covered[c];
                }
            }
            const double count = static_cast<double>(pixels.size());
            for (int c = 0; c < kSem; ++c) {
                channelMin[c] = trough[c] / 255.0f;
                channelMax[c] = peak[c] / 255.0f;
                channelMean[c] = static_cast<float>(sum[c] / count / 255.0);
                channelCoverage[c] = static_cast<float>(covered[c] / count);
                // A flat fill selects nothing. This is how an unwired Surface
                // Composer Hardness input presents: a constant 0.45 that
                // reports full coverage and washes the bound material evenly
                // over the whole terrain.
                channelConstant[c] = peak[c] == trough[c];
            }
        }
    }

    for (int slot = 0; slot < TerrainObject::kMaxLayerSlots; ++slot) {
        TerrainLayerInfo info;
        info.slot = slot;
        info.channel = TerrainObject::layerSlotName(slot);
        info.semantic_overlay = TerrainObject::isSemanticLayerSlot(slot);
        if (info.semantic_overlay) {
            const int c = slot - TerrainObject::kSplatLayerSlots;
            info.channel_measured = semanticMeasured;
            info.channel_min = channelMin[c];
            info.channel_max = channelMax[c];
            info.channel_mean = channelMean[c];
            info.channel_coverage = channelCoverage[c];
            info.channel_constant = semanticMeasured && channelConstant[c];
        }
        const size_t index = static_cast<size_t>(slot);
        if (index < terrain->layers.size() && terrain->layers[index]) {
            info.bound = true;
            info.material = terrain->layers[index]->materialName;
        }
        if (index < terrain->layer_uv_scales.size())
            info.uv_scale = terrain->layer_uv_scales[index];
        if (index < terrain->layer_overlay_strength.size())
            info.overlay_strength = terrain->layer_overlay_strength[index];
        if (index < terrain->layer_overlay_ignore_cover.size())
            info.overlay_ignore_cover = terrain->layer_overlay_ignore_cover[index] != 0;
        out_layers.push_back(std::move(info));
    }
    return Result::success();
}

Result setTerrainLayer(const std::string& terrain_name, int slot,
                       const std::string* material,
                       const float* uv_scale,
                       const float* overlay_strength,
                       const bool* overlay_ignore_cover) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (slot < 0 || slot >= TerrainObject::kMaxLayerSlots)
        return Result::fail("slot must be in the range [0, " +
                            std::to_string(TerrainObject::kMaxLayerSlots - 1) + "]");
    TerrainObject* terrain = TerrainManager::getInstance().getTerrainByName(terrain_name);
    if (!terrain) return Result::fail("terrain not found: " + terrain_name);

    // A terrain created before the overlay slots existed still carries four.
    if (terrain->layers.size() < TerrainObject::kMaxLayerSlots)
        terrain->layers.resize(TerrainObject::kMaxLayerSlots, nullptr);
    if (terrain->layer_uv_scales.size() < TerrainObject::kMaxLayerSlots)
        terrain->layer_uv_scales.resize(TerrainObject::kMaxLayerSlots, 50.0f);
    if (terrain->layer_overlay_strength.size() < TerrainObject::kMaxLayerSlots)
        terrain->layer_overlay_strength.resize(TerrainObject::kMaxLayerSlots, 1.0f);
    if (terrain->layer_overlay_ignore_cover.size() < TerrainObject::kMaxLayerSlots)
        terrain->layer_overlay_ignore_cover.resize(TerrainObject::kMaxLayerSlots, 0);

    const size_t index = static_cast<size_t>(slot);
    const bool semantic = TerrainObject::isSemanticLayerSlot(slot);

    if (overlay_strength) {
        // Refused rather than ignored: the splat slots are normalized against
        // each other, so an independent strength there has no meaning and
        // silently accepting it would report a setting that does nothing.
        if (!semantic)
            return Result::fail("overlay_strength applies to semantic slots 4-7 only; slot " +
                                std::to_string(slot) + " (" +
                                TerrainObject::layerSlotName(slot) +
                                ") is normalized against the other splat slots");
        if (!std::isfinite(*overlay_strength) || *overlay_strength < 0.0f || *overlay_strength > 1.0f)
            return Result::fail("overlay_strength must be in the range [0, 1]");
        terrain->layer_overlay_strength[index] = *overlay_strength;
    }

    if (overlay_ignore_cover) {
        if (!semantic)
            return Result::fail("overlay_ignore_cover applies to semantic slots 4-7 only; slot " +
                                std::to_string(slot) + " (" +
                                TerrainObject::layerSlotName(slot) + ") is not an overlay");
        // Slot 6 is Ice, which is a cover in its own right and is never
        // buried. Accepting the flag there would report a setting that
        // changes nothing.
        if (slot == TerrainObject::kSplatLayerSlots + 2)
            return Result::fail("slot 6 (Ice) is a cover and is never buried by snow, "
                                "so overlay_ignore_cover has no meaning there");
        terrain->layer_overlay_ignore_cover[index] = *overlay_ignore_cover ? 1 : 0;
    }

    if (uv_scale) {
        if (!std::isfinite(*uv_scale) || *uv_scale <= 0.0f)
            return Result::fail("uv_scale must be finite and positive");
        terrain->layer_uv_scales[index] = *uv_scale;
    }

    if (material) {
        if (material->empty()) {
            // Clearing a semantic slot restores the built-in shading for that
            // channel; clearing a splat slot drops it from the partition.
            terrain->layers[index] = nullptr;
        } else {
            auto& manager = MaterialManager::getInstance();
            if (!manager.hasMaterial(*material))
                return Result::fail("material not found: " + *material);
            auto shared = manager.getMaterialShared(manager.getMaterialID(*material));
            if (!shared) return Result::fail("material could not be resolved: " + *material);
            terrain->layers[index] = shared;
        }
    }

    // Re-upload the terrain layer buffer; a slot binding is invisible until
    // the backends see it, and the panel path does exactly this.
    g_ctx->renderer.resetCPUAccumulation();
    g_ctx->renderer.updateBackendMaterials(g_ctx->scene);
    ProjectManager::getInstance().markModified();
    return Result::success();
}

Result createTerrain(const std::string& requested_name, int resolution, float size,
                     float height_scale, int mesh_resolution, TerrainInfo& out_info) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (resolution < 64 || resolution > 4096)
        return Result::fail("resolution must be in the range [64, 4096]");
    if (!std::isfinite(size) || size <= 0.0f)
        return Result::fail("size must be finite and positive");
    if (!std::isfinite(height_scale) || height_scale <= 0.0f)
        return Result::fail("height_scale must be finite and positive");

    if (mesh_resolution != 0 && (mesh_resolution < 2 || mesh_resolution > resolution))
        return Result::fail("mesh_resolution must be 0 (follow the field) or in the range [2, "
                            + std::to_string(resolution) + "]");

    const std::string name = uniqueTerrainName(requested_name);
    TerrainObject* terrain = TerrainManager::getInstance().createTerrain(g_ctx->scene, resolution,
                                                                        size, height_scale,
                                                                        mesh_resolution);
    if (!terrain) return Result::fail("failed to create terrain: " + name);

    terrain->name = name;
    // ★ No second updateTerrainMesh() here. It used to exist only to re-apply
    // height_scale after the fact, which meant every scripted terrain built its
    // vertex+normal grid TWICE -- 156 ms of pure waste at 4096^2, and it scales
    // with the square of the resolution. createTerrain() now takes the scale.
    if (terrain->flatMesh) terrain->flatMesh->nodeName = name;

    ui.terrain_brush.active_terrain_id = terrain->id;
    ui.mesh_cache_valid = false;
    scheduleSceneMutationRebuilds(*g_ctx, true);
    out_info = terrainInfo(*terrain);
    return Result::success();
}

Result importTerrainHeightmap(const std::string& filepath, const std::string& requested_name,
                              float size, float height_scale, int max_resolution,
                              TerrainInfo& out_info) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (filepath.empty()) return Result::fail("heightmap filepath must not be empty");
    if (!std::isfinite(size) || size <= 0.0f || !std::isfinite(height_scale) || height_scale <= 0.0f)
        return Result::fail("size and height_scale must be finite and positive");
    if (max_resolution < 64 || max_resolution > 16384)
        return Result::fail("max_resolution must be in the range [64, 16384]");
    const std::string name = uniqueTerrainName(requested_name.empty() ? "TerrainImported" : requested_name);
    TerrainObject* terrain = TerrainManager::getInstance().createTerrainFromHeightmap(
        g_ctx->scene, filepath, size, height_scale, max_resolution);
    if (!terrain) return Result::fail("failed to import terrain heightmap: " + filepath);
    terrain->name = name;
    if (terrain->flatMesh) terrain->flatMesh->nodeName = name;
    ui.terrain_brush.active_terrain_id = terrain->id;
    ui.mesh_cache_valid = false;
    scheduleSceneMutationRebuilds(*g_ctx, true);
    out_info = terrainInfo(*terrain);
    return Result::success();
}

Result setTerrainMeshResolution(const std::string& terrain_name, int mesh_resolution,
                                TerrainInfo& out_info) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (terrain_name.empty()) return Result::fail("terrain name must not be empty");
    if (mesh_resolution != 0 && (mesh_resolution < 2 || mesh_resolution > 16384))
        return Result::fail("mesh_resolution must be 0 (follow the field) or in the range [2, 16384]");

    TerrainObject* terrain = TerrainManager::getInstance().getTerrainByName(terrain_name);
    if (!terrain) return Result::fail("terrain not found: " + terrain_name);
    if (terrain->nodeGraph && terrain->nodeGraph->isEvaluatingAsync())
        return Result::fail("terrain evaluation is still running: " + terrain_name);

    // ★ Report the clamp instead of applying it silently. Asking for a mesh
    // denser than the field is a real mistake -- it would interpolate detail the
    // data does not contain -- and a silent clamp would let the caller keep
    // believing it got what it asked for.
    const int field = (std::min)(terrain->heightmap.width, terrain->heightmap.height);
    if (mesh_resolution > field)
        return Result::fail("mesh_resolution " + std::to_string(mesh_resolution) +
                            " exceeds the field resolution " + std::to_string(field) +
                            "; a mesh denser than the field invents detail it cannot have");

    terrain->mesh_resolution = mesh_resolution;
    // Vertex count changes, so this is a topology change: the mesh has to be
    // rebuilt and re-registered, not updated in place.
    TerrainManager::getInstance().rebuildTerrainMesh(g_ctx->scene, terrain);
    ui.mesh_cache_valid = false;
    scheduleSceneMutationRebuilds(*g_ctx, true);
    out_info = terrainInfo(*terrain);
    return Result::success();
}

Result setTerrainPaintResolution(const std::string& terrain_name, int paint_resolution,
                                 TerrainInfo& out_info) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (terrain_name.empty()) return Result::fail("terrain name must not be empty");
    if (paint_resolution != 0 && (paint_resolution < 64 || paint_resolution > 16384))
        return Result::fail("paint_resolution must be 0 (follow the field) or in the range [64, 16384]");

    TerrainObject* terrain = TerrainManager::getInstance().getTerrainByName(terrain_name);
    if (!terrain) return Result::fail("terrain not found: " + terrain_name);
    if (terrain->nodeGraph && terrain->nodeGraph->isEvaluatingAsync())
        return Result::fail("terrain evaluation is still running: " + terrain_name);

    terrain->paint_resolution = paint_resolution;
    TerrainManager::getInstance().resizePaintMaps(terrain);

    // Unlike mesh resolution, paint resolution doesn't change the BVH topology,
    // but the API pattern is to schedule a full mutation rebuild to ensure
    // textures and accumulators are fully refreshed.
    ui.mesh_cache_valid = false;
    scheduleSceneMutationRebuilds(*g_ctx, true);

    out_info = terrainInfo(*terrain);
    return Result::success();
}

Result removeTerrain(const std::string& terrain_name) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (terrain_name.empty()) return Result::fail("terrain name must not be empty");

    TerrainObject* terrain = TerrainManager::getInstance().getTerrainByName(terrain_name);
    if (!terrain) return Result::fail("terrain not found: " + terrain_name);
    if (terrain->nodeGraph && terrain->nodeGraph->isEvaluatingAsync())
        return Result::fail("terrain evaluation is still running: " + terrain_name);
    const int id = terrain->id;
    TerrainManager::getInstance().removeTerrain(g_ctx->scene, id);
    if (ui.terrain_brush.active_terrain_id == id) ui.terrain_brush.active_terrain_id = -1;
    ui.mesh_cache_valid = false;
    scheduleSceneMutationRebuilds(*g_ctx, true);
    g_terrain_evaluations.erase(terrain_name);
    return Result::success();
}

Result exportTerrainHeightmap(const std::string& terrain_name, const std::string& filepath) {
    if (!g_ctx) return notBound();
    if (filepath.empty()) return Result::fail("heightmap filepath must not be empty");
    TerrainObject* terrain = TerrainManager::getInstance().getTerrainByName(terrain_name);
    if (!terrain) return Result::fail("terrain not found: " + terrain_name);
    TerrainManager::getInstance().exportHeightmap(terrain, filepath);
    return Result::success();
}

Result evaluateTerrain(const std::string& terrain_name, TerrainEvaluationInfo& out_info) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    TerrainObject* terrain = TerrainManager::getInstance().getTerrainByName(terrain_name);
    if (!terrain) return Result::fail("terrain not found: " + terrain_name);
    if (!terrain->nodeGraph) return Result::fail("terrain has no node graph: " + terrain_name);
    if (terrain->nodeGraph->isEvaluatingAsync())
        return Result::fail("terrain evaluation is already running: " + terrain_name);

    TerrainEvaluationRecord& record = g_terrain_evaluations[terrain_name];
    record = {};
    record.state = "running";
    terrain->nodeGraph->evaluateTerrainAsync(terrain, g_ctx->scene);
    record.progress = terrain->nodeGraph->asyncEvalProgress();
    record.current_node_id = terrain->nodeGraph->currentAsyncNodeId();
    out_info = evaluationInfo(terrain_name, record);
    return Result::success();
}

Result getTerrainEvaluationStatus(const std::string& terrain_name,
                                  TerrainEvaluationInfo& out_info) {
    if (!g_ctx) return notBound();
    TerrainObject* terrain = TerrainManager::getInstance().getTerrainByName(terrain_name);
    if (!terrain) return Result::fail("terrain not found: " + terrain_name);

    auto found = g_terrain_evaluations.find(terrain_name);
    if (found == g_terrain_evaluations.end()) {
        TerrainEvaluationRecord idle;
        out_info = evaluationInfo(terrain_name, idle);
        return Result::success();
    }
    if (terrain->nodeGraph && terrain->nodeGraph->isEvaluatingAsync()) {
        found->second.progress = terrain->nodeGraph->asyncEvalProgress();
        found->second.current_node_id = terrain->nodeGraph->currentAsyncNodeId();
    }
    out_info = evaluationInfo(terrain_name, found->second);
    return Result::success();
}

Result cancelTerrainEvaluation(const std::string& terrain_name) {
    if (!g_ctx) return notBound();
    TerrainObject* terrain = TerrainManager::getInstance().getTerrainByName(terrain_name);
    if (!terrain) return Result::fail("terrain not found: " + terrain_name);
    if (!terrain->nodeGraph || !terrain->nodeGraph->isEvaluatingAsync())
        return Result::fail("terrain evaluation is not running: " + terrain_name);
    if (!terrain->nodeGraph->activeEvalContext)
        return Result::fail("terrain evaluation cannot be cancelled yet: " + terrain_name);

    terrain->nodeGraph->activeEvalContext->requestCancel();
    g_terrain_evaluations[terrain_name].cancel_requested = true;
    return Result::success();
}

void pollTerrainEvaluations() {
    if (!g_ctx || g_terrain_evaluations.empty()) return;
    for (auto& [name, record] : g_terrain_evaluations) {
        if (record.state != "running") continue;
        TerrainObject* terrain = TerrainManager::getInstance().getTerrainByName(name);
        if (!terrain || !terrain->nodeGraph) {
            record.state = "failed";
            record.error = "terrain disappeared during evaluation";
            continue;
        }

        auto& graph = *terrain->nodeGraph;
        record.progress = graph.asyncEvalProgress();
        record.current_node_id = graph.currentAsyncNodeId();
        const bool updated = graph.pollEvaluateAsync();
        if (graph.isEvaluatingAsync()) continue;

        record.current_node_id = 0;
        record.error = graph.lastAsyncEvaluationError();
        if (record.cancel_requested || graph.lastAsyncEvaluationCancelled()) {
            record.state = "cancelled";
        } else if (!record.error.empty()) {
            record.state = "failed";
        } else {
            record.state = "completed";
            record.progress = 1.0f;
        }

        if (updated) {
            g_ctx->renderer.resetCPUAccumulation();
            g_ctx->renderer.updateBackendMaterials(g_ctx->scene);
            ui.mesh_cache_valid = false;
            scheduleSceneMutationRebuilds(*g_ctx, true);
        }
    }
}

// Only fields the caller actually set are applied; the sentinels keep a script
// from silently resetting twenty solver parameters it never mentioned.
static void applyFluvialCycleSettings(HydraulicErosionParams& params,
                                      const TerrainErosionSettings& s) {
    auto setInt = [](int& dst, int src, int lo, int hi) {
        if (src >= 0) dst = std::clamp(src, lo, hi);
    };
    auto setFloat = [](float& dst, float src, float lo, float hi) {
        if (src >= 0.0f && std::isfinite(src)) dst = std::clamp(src, lo, hi);
    };
    if (s.fluvial_cycle >= 0) params.fluvialCycle = s.fluvial_cycle != 0;
    // Preset first, explicit budgets second: "high quality but only 32
    // transport steps" has to be expressible in one call.
    const std::string quality = lowerCopy(s.fluvial_quality);
    if (quality == "draft") applyFluvialQuality(params, FluvialQuality::Draft);
    else if (quality == "balanced") applyFluvialQuality(params, FluvialQuality::Balanced);
    else if (quality == "high") applyFluvialQuality(params, FluvialQuality::High);
    if (s.mass_wasting >= 0) params.massWasting = s.mass_wasting != 0;
    setInt(params.fluvialIterations, s.fluvial_iterations, 0, 512);
    setInt(params.sedimentRouteSteps, s.sediment_route_steps, 0, 4096);
    setInt(params.avulsionInterval, s.avulsion_interval, 0, 4096);
    setInt(params.alluviumSteps, s.alluvium_steps, 0, 64);
    setInt(params.drainageRefreshInterval, s.drainage_refresh_interval, 1, 64);
    setInt(params.drainageFillPasses, s.drainage_fill_passes, 8, 4096);
    setInt(params.drainageAccumulatePasses, s.drainage_accumulate_passes, 8, 4096);
    setInt(params.drainageCoarsestSize, s.drainage_coarsest_size, 32, 512);
    setInt(params.massWastingSteps, s.mass_wasting_steps, 0, 64);
    setFloat(params.fluvialTimeStep, s.fluvial_time_step, 0.0f, 16.0f);
    setFloat(params.rainRate, s.rain_rate, 1.0e-4f, 100.0f);
    setFloat(params.orographicRain, s.orographic_rain, 0.0f, 1.0f);
    setFloat(params.incisionK, s.incision_k, 0.0f, 20000.0f);
    setFloat(params.streamPowerM, s.stream_power_m, 0.0f, 2.0f);
    setFloat(params.streamPowerN, s.stream_power_n, 0.1f, 4.0f);
    setFloat(params.transportK, s.transport_k, 0.0f, 20000.0f);
    setFloat(params.sedimentCover, s.sediment_cover, 0.0f, 1.0f);
    setFloat(params.settlingVelocity, s.settling_velocity, 0.0f, 100.0f);
    setFloat(params.reposeAngleDegrees, s.repose_angle_degrees, 1.0f, 80.0f);
    setFloat(params.alluviumSlopeDegrees, s.alluvium_slope_degrees, 0.1f, 20.0f);
    setFloat(params.alluviumRate, s.alluvium_rate, 0.0f, 1.0f);
    setFloat(params.alluviumConsolidation, s.alluvium_consolidation, 0.0f, 1.0f);
    setFloat(params.massWastingRate, s.mass_wasting_rate, 0.0f, 1.0f);
    setFloat(params.hillslopeDiffusion, s.hillslope_diffusion, 0.0f, 100.0f);
    setFloat(params.incisionSafety, s.incision_safety, 0.0f, 0.95f);
    setFloat(params.depositionSafety, s.deposition_safety, 0.0f, 0.95f);
    setFloat(params.maxStepMeters, s.max_step_meters, 0.0f, 1000.0f);
    setFloat(params.lakeEpsilonMeters, s.lake_epsilon_meters, 0.0f, 100.0f);
    setFloat(params.fluvialHeadwaterAreaKm2, s.headwater_area_km2, 1.0e-6f, 100.0f);
    // Bearings are legitimately negative, so this one has its own sentinel.
    if (s.rain_wind_degrees > -999.0f && std::isfinite(s.rain_wind_degrees))
        params.rainWindDegrees = s.rain_wind_degrees;
}

Result getTerrainErosionStats(TerrainErosionStats& out_stats) {
    if (!g_ctx) return notBound();
    const HydraulicErosionStats& s = TerrainManager::getInstance().lastErosionStats();
    out_stats.eroded = s.eroded;
    out_stats.deposited = s.deposited;
    out_stats.exported = s.exported;
    out_stats.carried = s.carried;
    out_stats.mass_error = s.massError;
    out_stats.mass_error_fraction = s.massErrorFraction;
    out_stats.lake_cells = s.lakeCells;
    out_stats.lake_area_fraction = s.lakeAreaFraction;
    out_stats.max_drainage_area_km2 = s.maxDrainageAreaKm2;
    out_stats.max_drainage_area_fraction = s.maxDrainageAreaFraction;
    out_stats.deep_lake_cells = s.deepLakeCells;
    out_stats.deep_lake_area_fraction = s.deepLakeAreaFraction;
    out_stats.deepest_lake_meters = s.deepestLakeMeters;
    out_stats.deposited_cells = s.depositedCells;
    out_stats.deposited_area_fraction = s.depositedAreaFraction;
    out_stats.deepest_deposit_meters = s.deepestDepositMeters;
    out_stats.mean_deposit_meters = s.meanDepositMeters;
    out_stats.drainage_density = s.drainageDensity;
    out_stats.cycle_iterations = s.cycleIterations;
    out_stats.gpu_path = s.gpuPath;
    return Result::success();
}

Result erodeTerrain(const std::string& terrain_name, const TerrainErosionSettings& settings) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    TerrainObject* terrain = TerrainManager::getInstance().getTerrainByName(terrain_name);
    if (!terrain) return Result::fail("terrain not found: " + terrain_name);
    if (terrain->nodeGraph && terrain->nodeGraph->isEvaluatingAsync())
        return Result::fail("terrain evaluation is still running: " + terrain_name);

    const std::string type = lowerCopy(settings.type);
    const std::string backend = lowerCopy(settings.backend);
    if (backend != "auto" && backend != "gpu" && backend != "cpu")
        return Result::fail("backend must be auto, gpu, or cpu");
    if (settings.iterations < 0)
        return Result::fail("iterations must be non-negative");
    if (type != "hydraulic" && type != "thermal" && type != "fluvial" && type != "wind")
        return Result::fail("unknown erosion type '" + settings.type +
                            "' (expected hydraulic|thermal|fluvial|wind)");
    const std::string quality_check = lowerCopy(settings.fluvial_quality);
    if (!quality_check.empty() && quality_check != "draft" &&
        quality_check != "balanced" && quality_check != "high")
        return Result::fail("fluvial_quality must be draft, balanced or high");
    const bool use_gpu = backend != "cpu";
    auto& manager = TerrainManager::getInstance();
    TerrainSnapshot before;
    if (settings.undo && g_history) before = captureTerrainSnapshot(*terrain);

    if (type == "hydraulic" || type == "fluvial") {
        HydraulicErosionParams params;
        params.seed = settings.seed;
        if (type == "fluvial") {
            params.iterations = settings.iterations > 0 ? settings.iterations : 250000;
            params.dropletLifetime = 384;
            params.inertia = 0.25f;
            params.sedimentCapacity = 1.5f;
            params.erodeSpeed = 0.12f;
            params.depositSpeed = 0.20f;
            params.evaporateSpeed = 0.001f;
            params.erosionRadius = 4;
            params.minSlope = 0.003f;
            if (use_gpu) manager.fluvialErosionGPU(terrain, params);
            else manager.fluvialErosion(terrain, params);
        } else {
            if (settings.iterations > 0) params.iterations = settings.iterations;
            applyFluvialCycleSettings(params, settings);
            if (use_gpu) manager.hydraulicErosionGPU(terrain, params);
            else manager.hydraulicErosion(terrain, params);
        }
    } else if (type == "thermal") {
        if (!std::isfinite(settings.talus_angle) || settings.talus_angle < 0.0f ||
            !std::isfinite(settings.amount) || settings.amount < 0.0f)
            return Result::fail("thermal talus_angle and amount must be finite and non-negative");
        ThermalErosionParams params;
        if (settings.iterations > 0) params.iterations = settings.iterations;
        params.talusAngle = settings.talus_angle;
        params.erosionAmount = settings.amount;
        if (use_gpu) manager.thermalErosionGPU(terrain, params);
        else manager.thermalErosion(terrain, params);
    } else if (type == "wind") {
        if (!std::isfinite(settings.strength) || settings.strength < 0.0f ||
            !std::isfinite(settings.direction))
            return Result::fail("wind strength must be non-negative and direction must be finite");
        const int iterations = settings.iterations > 0 ? settings.iterations : 10;
        if (use_gpu) manager.windErosionGPU(terrain, settings.strength, settings.direction, iterations);
        else manager.windErosion(terrain, settings.strength, settings.direction, iterations);
    }

    manager.updateTerrainMesh(terrain, false);
    ui.mesh_cache_valid = false;
    scheduleSceneMutationRebuilds(*g_ctx, true);
    if (settings.undo && g_history) {
        g_history->record(std::make_unique<TerrainSnapshotCommand>(
            terrain_name, "Terrain " + type + " erosion", std::move(before),
            captureTerrainSnapshot(*terrain)));
    }
    return Result::success();
}

Result applyTerrainPreset(const std::string& terrain_name, const std::string& preset,
                          std::vector<std::string>& out_wiring_faults,
                          bool replace_graph, bool add_satmap) {
    out_wiring_faults.clear();
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    TerrainObject* terrain = TerrainManager::getInstance().getTerrainByName(terrain_name);
    if (!terrain) return Result::fail("terrain not found: " + terrain_name);
    if (terrain->nodeGraph && terrain->nodeGraph->isEvaluatingAsync())
        return Result::fail("terrain evaluation is still running: " + terrain_name);

    const std::string kind = lowerCopy(preset);
    if (!terrain->nodeGraph || replace_graph) {
        terrain->nodeGraph = std::make_shared<TerrainNodesV2::TerrainNodeGraphV2>();
    }
    auto& graph = *terrain->nodeGraph;
    std::string satMapPreset = "Temperate";
    if (kind == "default") {
        if (!replace_graph && graph.nodeCount() != 0)
            return Result::fail("default preset requires replace_graph=True for a non-empty graph");
        graph.createDefaultGraph(terrain);
    } else if (kind == "snowy_mountain_valley") {
        if (!replace_graph && graph.nodeCount() != 0)
            return Result::fail("snowy_mountain_valley requires replace_graph=True for a non-empty graph");
        graph.createSnowyMountainValleyGraph(terrain);
        satMapPreset = "Alpine";
    } else if (kind == "snow_layer") {
        if (graph.nodeCount() == 0) graph.createDefaultGraph(terrain);
        if (!graph.addSnowLayerSetup()) return Result::fail("failed to add snow layer setup");
        satMapPreset = "Alpine";
    } else if (kind == "river_network") {
        if (graph.nodeCount() == 0) graph.createDefaultGraph(terrain);
        if (!graph.addRiverNetworkSetup()) return Result::fail("failed to add river network setup");
    } else if (kind == "biome_temperate") {
        if (graph.nodeCount() == 0) graph.createDefaultGraph(terrain);
        if (!graph.addBiomeFieldsSetup(TerrainNodesV2::BiomeClimatePreset::TemperateMixed)) return Result::fail("failed to add biome setup");
    } else if (kind == "biome_lush") {
        if (graph.nodeCount() == 0) graph.createDefaultGraph(terrain);
        if (!graph.addBiomeFieldsSetup(TerrainNodesV2::BiomeClimatePreset::LushValleys)) return Result::fail("failed to add biome setup");
        satMapPreset = "Tropical";
    } else if (kind == "biome_alpine") {
        if (graph.nodeCount() == 0) graph.createDefaultGraph(terrain);
        if (!graph.addBiomeFieldsSetup(TerrainNodesV2::BiomeClimatePreset::AlpineTundra)) return Result::fail("failed to add biome setup");
        satMapPreset = "Alpine";
    } else if (kind == "biome_arid") {
        if (graph.nodeCount() == 0) graph.createDefaultGraph(terrain);
        if (!graph.addBiomeFieldsSetup(TerrainNodesV2::BiomeClimatePreset::AridHighlands)) return Result::fail("failed to add biome setup");
        satMapPreset = "Desert";
    } else if (kind == "biome_boreal") {
        if (graph.nodeCount() == 0) graph.createDefaultGraph(terrain);
        if (!graph.addBiomeFieldsSetup(TerrainNodesV2::BiomeClimatePreset::BorealMountains)) return Result::fail("failed to add biome setup");
        satMapPreset = "Boreal";
    } else if (kind == "biome_foliage") {
        if (graph.nodeCount() == 0) graph.createDefaultGraph(terrain);
        if (!graph.addBiomeFoliageSetup()) return Result::fail("failed to add biome foliage setup");
    } else if (kind == "geology_foundation") {
        if (graph.nodeCount() == 0) graph.createDefaultGraph(terrain);
        if (!graph.addGeologyFoundationSetup()) return Result::fail("failed to add geology setup");
    } else {
        return Result::fail("unknown terrain preset '" + preset +
                            "' (expected default|snow_layer|snowy_mountain_valley|river_network|biome_boreal|biome_alpine|biome_foliage|geology_foundation|...)");
    }
    // Report what the setup could not wire. addLink answers a refused
    // connection with 0, and a setup that ignores that answer produces a
    // graph with holes whose only symptom is a plausible-looking result.
    for (const auto& fault : graph.lastSetupWiringFaults()) {
        out_wiring_faults.push_back(
            fault.fromNode + "." + fault.fromPin + " -> " +
            fault.toNode + "." + fault.toPin + ": " + fault.reason);
    }
    if (add_satmap && !graph.addSatMapSetup(satMapPreset))
        return Result::fail("terrain preset was applied but SatMap setup could not find a connected Height Output");
    graph.markAllDirty();
    return Result::success();
}

Result listTerrainSatMapPresets(std::vector<TerrainSatMapPresetInfo>& out_presets) {
    out_presets.clear();
    auto& library = TerrainNodesV2::SatMapPresetLibrary::instance();
    std::string error;
    if (!library.reload(&error)) return Result::fail(error);
    for (const auto& preset : library.presets()) {
        out_presets.push_back({preset.id, preset.label, preset.category,
                               preset.description, preset.version,
                               static_cast<int>(preset.layers.size())});
    }
    return Result::success();
}

Result applyTerrainSatMapPreset(const std::string& terrain_name, const std::string& preset_id,
                                std::vector<std::string>& out_warnings) {
    out_warnings.clear();
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    TerrainObject* terrain = TerrainManager::getInstance().getTerrainByName(terrain_name);
    if (!terrain) return Result::fail("terrain not found: " + terrain_name);
    if (terrain->nodeGraph && terrain->nodeGraph->isEvaluatingAsync())
        return Result::fail("terrain evaluation is still running: " + terrain_name);
    if (!terrain->nodeGraph) {
        terrain->nodeGraph = std::make_shared<TerrainNodesV2::TerrainNodeGraphV2>();
        terrain->nodeGraph->createDefaultGraph(terrain);
    }
    std::string error;
    if (!terrain->nodeGraph->applySatMapPresetRecipe(
            preset_id, &error, &out_warnings)) return Result::fail(error);
    return Result::success();
}

Result getTerrainFlowAuthority(const std::string& terrain_name,
                               TerrainFlowAuthority& out_authority) {
    out_authority = TerrainFlowAuthority{};
    if (!g_ctx) return notBound();
    TerrainObject* terrain = TerrainManager::getInstance().getTerrainByName(terrain_name);
    if (!terrain) return Result::fail("terrain not found: " + terrain_name);
    if (!terrain->nodeGraph) return Result::success();

    const TerrainNodesV2::FlowMaskNode* flow = nullptr;
    const TerrainNodesV2::RiverNetworkNode* riverNetwork = nullptr;
    for (const auto& node : terrain->nodeGraph->nodes) {
        if (auto* f = dynamic_cast<const TerrainNodesV2::FlowMaskNode*>(node.get())) {
            flow = f;
        }
        if (auto* r = dynamic_cast<const TerrainNodesV2::RiverNetworkNode*>(node.get()))
            riverNetwork = r;
    }

    if (flow) {
        out_authority.has_flow_node = true;
        out_authority.evaluated = flow->lastEvaluated;
        out_authority.discharge_measured = flow->lastDischargeMeasured;
        out_authority.erosion_unwired = flow->lastErosionUnwired;
        if (!flow->lastEvaluated)             out_authority.source = "not_evaluated";
        else if (flow->lastDischargeMeasured) out_authority.source = "measured";
        else if (flow->lastErosionUnwired)    out_authority.source = "derived_erosion_unwired";
        else                                  out_authority.source = "derived";
    }

    if (riverNetwork && riverNetwork->inputs.size() >= 2) {
        out_authority.has_river_network = true;
        const NodeSystem::Pin* areaSource =
            terrain->nodeGraph->getInputSource(riverNetwork->inputs[0].id);
        const NodeSystem::Pin* directionSource =
            terrain->nodeGraph->getInputSource(riverNetwork->inputs[1].id);
        out_authority.river_area_physical = areaSource &&
            areaSource->imageUnit == NodeSystem::ImageUnit::SquareMeters;
        out_authority.river_direction_channels = directionSource
            ? directionSource->imageChannels : 0;
        out_authority.river_lake_mask_connected = riverNetwork->inputs.size() > 3 &&
            terrain->nodeGraph->getInputSource(riverNetwork->inputs[3].id) != nullptr;
        out_authority.river_lake_spill_connected = riverNetwork->inputs.size() > 4 &&
            terrain->nodeGraph->getInputSource(riverNetwork->inputs[4].id) != nullptr;
        const NodeSystem::NodeBase* areaOwner = areaSource
            ? terrain->nodeGraph->getPinOwner(areaSource->id) : nullptr;
        const NodeSystem::NodeBase* directionOwner = directionSource
            ? terrain->nodeGraph->getPinOwner(directionSource->id) : nullptr;
        if (!areaOwner || !directionOwner) {
            out_authority.river_network_source = "unwired";
        // There is no "hydraulic" arm any more: Hydraulic Erosion stopped
        // publishing Drainage Area and Flow Direction when the compact port
        // contract moved both to Watershed Analysis, so it can no longer own
        // this pair. Feeding the network from Hydraulic's remaining Flow pin
        // (sediment transport, not discharge) now reports "mixed", which is
        // the honest answer -- an arm that can never be true would tell a
        // reader that case is still detectable.
        } else if (dynamic_cast<const TerrainNodesV2::WatershedAnalysisNode*>(areaOwner) &&
                   dynamic_cast<const TerrainNodesV2::WatershedAnalysisNode*>(directionOwner)) {
            out_authority.river_network_source = "watershed";
        } else {
            out_authority.river_network_source = "mixed";
        }
    }
    return Result::success();
}

Result getTerrainSlopeAreaFit(const std::string& terrain_name,
                              TerrainSlopeAreaFit& out_fit) {
    out_fit = TerrainSlopeAreaFit{};
    if (!g_ctx) return notBound();
    TerrainObject* terrain = TerrainManager::getInstance().getTerrainByName(terrain_name);
    if (!terrain) return Result::fail("terrain not found: " + terrain_name);

    const Heightmap& hm = terrain->heightmap;
    const int w = hm.width, h = hm.height;
    const std::vector<float>& flow = terrain->flowMap;
    if (w < 8 || h < 8 || static_cast<int>(hm.data.size()) < w * h ||
        static_cast<int>(flow.size()) < w * h) {
        return Result::success(); // status stays "no_flow_field"
    }

    const float cellSize = (std::max)(hm.scale_xz / static_cast<float>((std::max)(w, h)), 1e-4f);
    const float heightScale = (std::max)(std::abs(hm.scale_y), 1e-4f);
    const float cellArea = cellSize * cellSize;

    float peak = 0.0f;
    for (int i = 0; i < w * h; ++i) peak = (std::max)(peak, flow[static_cast<size_t>(i)]);
    out_fit.flow_peak = peak;
    // The flow field is filled by calculate_flow, not by evaluating the graph.
    // Skip that call and this array is allocated, correctly sized and entirely
    // zero -- so it sails past the size guard above and every cell falls below
    // the channel threshold. The old answer was "too_few_channels", which
    // reads as a measured fact about the terrain rather than as a missing
    // input, and it cost a false regression report.
    if (!(peak > 0.0f)) {
        out_fit.status = "flow_field_empty";
        return Result::success();
    }
    // Same channel definition as calculate_flow, so the two measurements are
    // talking about the same cells.
    const float channelThreshold = (std::max)(peak * 0.01f, 2.0f);

    // Log-spaced bins over drainage area. The relation is a power law, so the
    // fit is done on the BINNED MEDIAN slope per bin rather than raw cells:
    // hillslope noise and single steep pixels would otherwise dominate a
    // least-squares line that is supposed to describe channels.
    constexpr int kBins = 24;
    std::vector<std::vector<float>> binSlopes(kBins);
    std::vector<double> binLogArea(kBins, 0.0);
    std::vector<int> binCount(kBins, 0);
    const double logAreaMin = std::log(static_cast<double>(channelThreshold) * cellArea);
    const double logAreaMax = std::log((std::max)(static_cast<double>(peak), channelThreshold + 1.0) * cellArea);
    const double logSpan = (std::max)(logAreaMax - logAreaMin, 1e-6);

    for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {
            const size_t idx = static_cast<size_t>(y) * w + x;
            if (flow[idx] < channelThreshold) continue;
            ++out_fit.channel_cells;

            const float dzdx = (hm.data[idx + 1] - hm.data[idx - 1]) * heightScale / (2.0f * cellSize);
            const float dzdy = (hm.data[idx + w] - hm.data[idx - w]) * heightScale / (2.0f * cellSize);
            const float slope = std::sqrt(dzdx * dzdx + dzdy * dzdy);
            if (!(slope > 1e-5f)) continue; // log(0) is not a data point

            const double logArea = std::log(static_cast<double>(flow[idx]) * cellArea);
            int bin = static_cast<int>((logArea - logAreaMin) / logSpan * kBins);
            bin = (std::min)((std::max)(bin, 0), kBins - 1);
            binSlopes[static_cast<size_t>(bin)].push_back(slope);
            binLogArea[static_cast<size_t>(bin)] += logArea;
            ++binCount[static_cast<size_t>(bin)];
        }
    }

    std::vector<double> xs, ys;
    for (int b = 0; b < kBins; ++b) {
        auto& samples = binSlopes[static_cast<size_t>(b)];
        if (samples.size() < 16) continue; // a bin of three cells is not a point
        std::nth_element(samples.begin(), samples.begin() + samples.size() / 2, samples.end());
        const double medianSlope = samples[samples.size() / 2];
        xs.push_back(binLogArea[static_cast<size_t>(b)] / binCount[static_cast<size_t>(b)]);
        ys.push_back(std::log(medianSlope));
    }
    out_fit.bin_count = static_cast<int>(xs.size());
    if (out_fit.bin_count < 4) {
        out_fit.status = "too_few_channels";
        return Result::success();
    }

    double sx = 0.0, sy = 0.0;
    for (size_t i = 0; i < xs.size(); ++i) { sx += xs[i]; sy += ys[i]; }
    const double mx = sx / xs.size(), my = sy / ys.size();
    double sxy = 0.0, sxx = 0.0, syy = 0.0;
    for (size_t i = 0; i < xs.size(); ++i) {
        const double dx = xs[i] - mx, dy = ys[i] - my;
        sxy += dx * dy; sxx += dx * dx; syy += dy * dy;
    }
    if (sxx < 1e-12 || syy < 1e-12) {
        out_fit.status = "too_few_channels";
        return Result::success();
    }
    const double slope = sxy / sxx;
    out_fit.measured = true;
    out_fit.concavity_index = static_cast<float>(-slope); // S ~ A^-theta
    out_fit.intercept = static_cast<float>(my - slope * mx);
    out_fit.r_squared = static_cast<float>((sxy * sxy) / (sxx * syy));
    out_fit.status = "ok";
    return Result::success();
}

Result calculateTerrainFlow(const std::string& terrain_name, TerrainFlowStats& out_stats) {
    out_stats = TerrainFlowStats{};
    if (!g_ctx) return notBound();
    TerrainObject* terrain = TerrainManager::getInstance().getTerrainByName(terrain_name);
    if (!terrain) return Result::fail("terrain not found: " + terrain_name);
    if (terrain->nodeGraph && terrain->nodeGraph->isEvaluatingAsync())
        return Result::fail("terrain evaluation is still running: " + terrain_name);
    TerrainManager::getInstance().calculateFlowMap(terrain);

    const int w = terrain->heightmap.width;
    const int h = terrain->heightmap.height;
    const std::vector<float>& flow = terrain->flowMap;
    if (w < 3 || h < 3 || static_cast<int>(flow.size()) < w * h) return Result::success();

    out_stats.width = w;
    out_stats.height = h;
    double sum = 0.0;
    float peak = 0.0f;
    for (int i = 0; i < w * h; ++i) {
        sum += flow[static_cast<size_t>(i)];
        peak = (std::max)(peak, flow[static_cast<size_t>(i)]);
    }
    out_stats.max_accumulation = peak;
    out_stats.mean_accumulation = static_cast<float>(sum / (static_cast<double>(w) * h));

    // A channel is a cell carrying a meaningful share of the peak discharge.
    // 1% of peak is well above the per-cell rainfall of 1.0 on any terrain
    // large enough to have a network at all.
    const float channelThreshold = (std::max)(peak * 0.01f, 2.0f);
    static const int dx8[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    static const int dy8[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float here = flow[static_cast<size_t>(y) * w + x];
            if (here < channelThreshold) continue;
            ++out_stats.channel_cells;
            const bool onBorder = (x == 0 || y == 0 || x == w - 1 || y == h - 1);
            bool downstreamExists = false;
            for (int d = 0; d < 8 && !downstreamExists; ++d) {
                const int nx = x + dx8[d], ny = y + dy8[d];
                if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
                if (flow[static_cast<size_t>(ny) * w + nx] > here) downstreamExists = true;
            }
            if (downstreamExists) continue;
            if (onBorder) ++out_stats.border_terminations;
            else ++out_stats.inland_terminations;
        }
    }
    if (out_stats.channel_cells > 0) {
        out_stats.inland_termination_ratio =
            static_cast<float>(out_stats.inland_terminations) /
            static_cast<float>(out_stats.channel_cells);
    }
    return Result::success();
}

Result sampleTerrainHeight(const std::string& terrain_name, float world_x, float world_z,
                           float& out_height) {
    if (!g_ctx) return notBound();
    if (!std::isfinite(world_x) || !std::isfinite(world_z))
        return Result::fail("sample coordinates must be finite");
    TerrainObject* terrain = TerrainManager::getInstance().getTerrainByName(terrain_name);
    if (!terrain) return Result::fail("terrain not found: " + terrain_name);
    const Heightmap& hm = terrain->heightmap;
    if (hm.data.empty() || hm.width < 2 || hm.height < 2 || hm.scale_xz <= 0.0f)
        return Result::fail("terrain heightmap is empty: " + terrain_name);

    Vec3 local(world_x, 0.0f, world_z);
    if (terrain->transform) {
        local = terrain->transform->getFinal().inverse()
                    .multiplyVector(Vec4(world_x, 0.0f, world_z, 1.0f)).xyz();
    }
    if (local.x < 0.0f || local.z < 0.0f || local.x > hm.scale_xz || local.z > hm.scale_xz)
        return Result::fail("sample point lies outside terrain bounds");
    const float gx = local.x / hm.scale_xz * static_cast<float>(hm.width - 1);
    const float gz = local.z / hm.scale_xz * static_cast<float>(hm.height - 1);
    const int x0 = static_cast<int>(std::floor(gx));
    const int z0 = static_cast<int>(std::floor(gz));
    const int x1 = (std::min)(x0 + 1, hm.width - 1);
    const int z1 = (std::min)(z0 + 1, hm.height - 1);
    const float fx = gx - static_cast<float>(x0);
    const float fz = gz - static_cast<float>(z0);
    const float h0 = hm.data[z0 * hm.width + x0] * (1.0f - fx) + hm.data[z0 * hm.width + x1] * fx;
    const float h1 = hm.data[z1 * hm.width + x0] * (1.0f - fx) + hm.data[z1 * hm.width + x1] * fx;
    const float local_y = (h0 * (1.0f - fz) + h1 * fz) * hm.scale_y;
    out_height = terrain->transform
        ? terrain->transform->getFinal().multiplyVector(Vec4(local.x, local_y, local.z, 1.0f)).y
        : local_y;
    return Result::success();
}

namespace {

// World XZ -> the terrain's own local frame, the space TerrainManager's brushes
// index the field in. Returns false when the point is off the tile: a dab there
// is not "no change", it is a miss, and reporting a miss as a change is how a
// broken coordinate mapping passes a test.
bool terrainLocalFromWorld(const TerrainObject& terrain, float world_x, float world_z,
                           Vec3& out_local) {
    out_local = Vec3(world_x, 0.0f, world_z);
    if (terrain.transform) {
        out_local = terrain.transform->getFinal().inverse()
                        .multiplyVector(Vec4(world_x, 0.0f, world_z, 1.0f)).xyz();
    }
    const float size = terrain.heightmap.scale_xz;
    return size > 0.0f && out_local.x >= 0.0f && out_local.z >= 0.0f &&
           out_local.x <= size && out_local.z <= size;
}

void terrainFieldBounds(const TerrainObject& terrain, float& out_min, float& out_max) {
    out_min = 0.0f;
    out_max = 0.0f;
    if (terrain.heightmap.data.empty()) return;
    const auto bounds = std::minmax_element(terrain.heightmap.data.begin(),
                                            terrain.heightmap.data.end());
    out_min = *bounds.first;
    out_max = *bounds.second;
}

// Splat paint undo. TerrainSnapshot deliberately carries the height field and
// its derived maps, NOT the splat texture, so reusing it here would have made
// `undo` a parameter that reports success and restores nothing.
class TerrainSplatSnapshotCommand final : public SceneCommand {
public:
    TerrainSplatSnapshotCommand(std::string terrain_name, std::vector<CompactVec4> before,
                                std::vector<CompactVec4> after)
        : terrain_name_(std::move(terrain_name)), before_(std::move(before)),
          after_(std::move(after)) {}
    void execute(UIContext& ctx) override { apply(ctx, after_); }
    void undo(UIContext& ctx) override { apply(ctx, before_); }
    Type getType() const override { return Type::Generic; }
    std::string getDescription() const override { return "Terrain splat paint"; }
private:
    void apply(UIContext& ctx, const std::vector<CompactVec4>& pixels) {
        TerrainObject* terrain = TerrainManager::getInstance().getTerrainByName(terrain_name_);
        if (!terrain || !terrain->splatMap) return;
        if (terrain->splatMap->pixels.size() != pixels.size()) return; // resolution changed
        terrain->splatMap->pixels = pixels;
        terrain->splatMap->updateGPU();
        ctx.renderer.resetCPUAccumulation();
        ctx.renderer.updateBackendMaterials(ctx.scene);
    }
    std::string terrain_name_;
    std::vector<CompactVec4> before_;
    std::vector<CompactVec4> after_;
};

int terrainSculptModeFromName(const std::string& mode) {
    if (mode == "raise")   return 0;
    if (mode == "lower")   return 1;
    if (mode == "flatten") return 2;
    if (mode == "smooth")  return 3;
    if (mode == "stamp")   return 4;
    return -1;
}

// Mean weight of one splat channel over the whole map. The splat map is the
// paint's only durable record, so this is what a paint assertion reads.
float terrainSplatCoverage(const TerrainObject& terrain, int channel) {
    if (!terrain.splatMap || !terrain.splatMap->is_loaded()) return 0.0f;
    if (channel < 0 || channel > 3) return 0.0f;
    // Texture::pixels is one CompactVec4 (RGBA bytes) PER TEXEL, not a flat byte
    // array - indexing it as 4 bytes per texel reads a quarter of the map and
    // calls it coverage.
    const std::vector<CompactVec4>& pixels = terrain.splatMap->pixels;
    if (pixels.empty()) return 0.0f;
    double sum = 0.0;
    for (const CompactVec4& texel : pixels) {
        const uint8_t weight = (channel == 0) ? texel.r
                             : (channel == 1) ? texel.g
                             : (channel == 2) ? texel.b
                                              : texel.a;
        sum += static_cast<double>(weight);
    }
    return static_cast<float>(sum / static_cast<double>(pixels.size()) / 255.0);
}

} // namespace

Result sculptTerrain(const std::string& terrain_name,
                     const std::vector<TerrainBrushDab>& dabs,
                     const TerrainSculptSettings& settings,
                     TerrainBrushResult& out_result) {
    out_result = TerrainBrushResult{};
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (dabs.empty()) return Result::fail("a sculpt stroke needs at least one dab");
    TerrainObject* terrain = TerrainManager::getInstance().getTerrainByName(terrain_name);
    if (!terrain) return Result::fail("terrain not found: " + terrain_name);
    if (terrain->nodeGraph && terrain->nodeGraph->isEvaluatingAsync())
        return Result::fail("terrain evaluation is still running: " + terrain_name);
    if (terrain->heightmap.data.empty() || terrain->heightmap.width < 2)
        return Result::fail("terrain heightmap is empty: " + terrain_name);

    const int mode = terrainSculptModeFromName(lowerCopy(settings.mode));
    if (mode < 0)
        return Result::fail("unknown sculpt mode '" + settings.mode +
                            "' (expected raise|lower|flatten|smooth|stamp)");
    if (!std::isfinite(settings.radius) || settings.radius <= 0.0f)
        return Result::fail("radius must be finite and positive");
    if (!std::isfinite(settings.strength))
        return Result::fail("strength must be finite");
    if (!std::isfinite(settings.dt) || settings.dt <= 0.0f)
        return Result::fail("dt must be finite and positive");
    if (!std::isfinite(settings.curve) || settings.curve < 0.25f || settings.curve > 4.0f)
        return Result::fail("curve must be in the range [0.25, 4]");

    std::shared_ptr<Texture> stamp;
    if (mode == 4) {
        if (settings.stamp_texture_path.empty())
            return Result::fail("stamp mode needs stamp_texture_path");
        stamp = std::make_shared<Texture>(settings.stamp_texture_path, TextureType::Albedo);
        if (!stamp->is_loaded())
            return Result::fail("stamp texture could not be loaded: " + settings.stamp_texture_path);
    }

    // Resolve every dab BEFORE mutating anything: a stroke that runs half off
    // the tile would otherwise report success for the part that landed.
    std::vector<Vec3> worldPoints;
    worldPoints.reserve(dabs.size());
    for (size_t i = 0; i < dabs.size(); ++i) {
        if (!std::isfinite(dabs[i].world_x) || !std::isfinite(dabs[i].world_z))
            return Result::fail("dab coordinates must be finite");
        Vec3 local;
        if (!terrainLocalFromWorld(*terrain, dabs[i].world_x, dabs[i].world_z, local))
            return Result::fail("dab " + std::to_string(i) + " lies outside terrain bounds");
        worldPoints.push_back(Vec3(dabs[i].world_x, 0.0f, dabs[i].world_z));
    }

    (void)sampleTerrainHeight(terrain_name, dabs[0].world_x, dabs[0].world_z,
                              out_result.height_before);
    terrainFieldBounds(*terrain, out_result.field_min_before, out_result.field_max_before);

    // Flatten samples its target from the surface under the first dab, matching
    // the panel's click behaviour, unless the caller pinned an altitude.
    float target = settings.flatten_target;
    if (mode == 2 && !settings.use_fixed_height) target = out_result.height_before;

    auto& manager = TerrainManager::getInstance();
    TerrainSnapshot before;
    if (settings.undo && g_history) before = captureTerrainSnapshot(*terrain);

    for (const Vec3& world : worldPoints) {
        // TerrainManager::sculpt maps world -> local itself; the local
        // resolution above was the bounds check, not the argument.
        manager.sculpt(terrain, world, mode, settings.radius, settings.strength,
                       settings.dt, settings.curve, target, stamp,
                       settings.stamp_rotation, false);
    }

    manager.updateTerrainMesh(terrain, false);
    ui.mesh_cache_valid = false;
    scheduleSceneMutationRebuilds(*g_ctx, true);

    out_result.dab_count = static_cast<int>(dabs.size());
    (void)sampleTerrainHeight(terrain_name, dabs[0].world_x, dabs[0].world_z,
                              out_result.height_after);
    terrainFieldBounds(*terrain, out_result.field_min_after, out_result.field_max_after);

    if (settings.undo && g_history) {
        g_history->record(std::make_unique<TerrainSnapshotCommand>(
            terrain_name, "Terrain sculpt (" + lowerCopy(settings.mode) + ")",
            std::move(before), captureTerrainSnapshot(*terrain)));
    }
    ProjectManager::getInstance().markModified();
    return Result::success();
}

Result paintTerrainSplat(const std::string& terrain_name,
                         const std::vector<TerrainBrushDab>& dabs,
                         const TerrainSplatPaintSettings& settings,
                         TerrainSplatPaintResult& out_result) {
    out_result = TerrainSplatPaintResult{};
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (dabs.empty()) return Result::fail("a paint stroke needs at least one dab");
    TerrainObject* terrain = TerrainManager::getInstance().getTerrainByName(terrain_name);
    if (!terrain) return Result::fail("terrain not found: " + terrain_name);
    if (terrain->nodeGraph && terrain->nodeGraph->isEvaluatingAsync())
        return Result::fail("terrain evaluation is still running: " + terrain_name);
    if (settings.channel < 0 || settings.channel > 3)
        return Result::fail("channel must be in the range [0, 3]");
    if (!std::isfinite(settings.radius) || settings.radius <= 0.0f)
        return Result::fail("radius must be finite and positive");
    if (!std::isfinite(settings.strength))
        return Result::fail("strength must be finite");
    if (!std::isfinite(settings.dt) || settings.dt <= 0.0f)
        return Result::fail("dt must be finite and positive");
    // Refused, not auto-created: an uninitialised terrain has no layer stack to
    // paint into, and silently making one would hide the missing setup step.
    if (!terrain->splatMap || !terrain->splatMap->is_loaded() || terrain->layers.empty())
        return Result::fail("terrain has no splat map / layers; initialise its layers first: " +
                            terrain_name);

    std::vector<Vec3> worldPoints;
    worldPoints.reserve(dabs.size());
    for (size_t i = 0; i < dabs.size(); ++i) {
        if (!std::isfinite(dabs[i].world_x) || !std::isfinite(dabs[i].world_z))
            return Result::fail("dab coordinates must be finite");
        Vec3 local;
        if (!terrainLocalFromWorld(*terrain, dabs[i].world_x, dabs[i].world_z, local))
            return Result::fail("dab " + std::to_string(i) + " lies outside terrain bounds");
        worldPoints.push_back(Vec3(dabs[i].world_x, 0.0f, dabs[i].world_z));
    }

    out_result.channel = settings.channel;
    out_result.coverage_before = terrainSplatCoverage(*terrain, settings.channel);

    const bool record_undo = settings.undo && g_history;
    std::vector<CompactVec4> splat_before;
    if (record_undo) splat_before = terrain->splatMap->pixels;

    auto& manager = TerrainManager::getInstance();
    for (const Vec3& world : worldPoints) {
        manager.paintSplatMap(terrain, world, settings.channel, settings.radius,
                              settings.strength, settings.dt);
    }

    // The splat map is a texture: without this refresh the paint exists only in
    // the CPU buffer and no backend shows it (the panel path does the same).
    g_ctx->renderer.resetCPUAccumulation();
    g_ctx->renderer.updateBackendMaterials(g_ctx->scene);
    scheduleSceneMutationRebuilds(*g_ctx, false);

    out_result.dab_count = static_cast<int>(dabs.size());
    out_result.coverage_after = terrainSplatCoverage(*terrain, settings.channel);
    if (record_undo) {
        g_history->record(std::make_unique<TerrainSplatSnapshotCommand>(
            terrain_name, std::move(splat_before), terrain->splatMap->pixels));
    }
    ProjectManager::getInstance().markModified();
    return Result::success();
}

Result carveTerrainRiver(const std::string& terrain_name, const std::string& river_name,
                         const TerrainRiverCarveSettings& settings) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    TerrainObject* terrain = TerrainManager::getInstance().getTerrainByName(terrain_name);
    if (!terrain) return Result::fail("terrain not found: " + terrain_name);
    if (terrain->nodeGraph && terrain->nodeGraph->isEvaluatingAsync())
        return Result::fail("terrain evaluation is still running: " + terrain_name);
    if (!std::isfinite(settings.depth_multiplier) || settings.depth_multiplier <= 0.0f ||
        !std::isfinite(settings.smoothness) || settings.smoothness < 0.0f || settings.smoothness > 1.0f)
        return Result::fail("depth_multiplier must be positive and smoothness must be in [0, 1]");

    RiverSpline* river = nullptr;
    for (auto& candidate : RiverManager::getInstance().getRivers()) {
        if (candidate.name == river_name) { river = &candidate; break; }
    }
    if (!river) return Result::fail("river not found: " + river_name);
    if (river->spline.pointCount() < 2) return Result::fail("river requires at least two control points");
    TerrainSnapshot before;
    if (settings.undo && g_history) before = captureTerrainSnapshot(*terrain);

    const int samples = (std::max)(2, river->lengthSubdivisions * 3);
    std::vector<Vec3> points;
    std::vector<float> widths;
    std::vector<float> depths;
    points.reserve(samples + 1);
    widths.reserve(samples + 1);
    depths.reserve(samples + 1);
    for (int i = 0; i <= samples; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(samples);
        points.push_back(river->samplePosition(t));
        widths.push_back((std::max)(0.02f, river->sampleWidth(t)));
        depths.push_back((std::max)(0.0f, river->sampleDepth(t) * settings.depth_multiplier));
    }

    auto& manager = TerrainManager::getInstance();
    const std::string mode = lowerCopy(settings.mode);
    if (mode == "simple") {
        manager.carveRiverBed(terrain->id, points, widths, depths, settings.smoothness, g_ctx->scene);
    } else if (mode == "natural") {
        TerrainManager::NaturalCarveParams params;
        params.noiseStrength = settings.noise_strength;
        params.enableDeepPools = settings.deep_pools;
        params.enableRiffles = settings.riffles;
        params.enableAsymmetry = settings.asymmetric_banks;
        params.enablePointBars = settings.point_bars;
        manager.carveRiverBedNatural(terrain->id, points, widths, depths, settings.smoothness,
                                     params, g_ctx->scene);
    } else {
        return Result::fail("river carve mode must be simple or natural");
    }
    if (settings.post_erosion) {
        ThermalErosionParams params;
        params.iterations = (std::max)(1, settings.post_erosion_iterations);
        params.talusAngle = 0.3f;
        params.erosionAmount = 0.4f;
        manager.thermalErosion(terrain, params);
        manager.updateTerrainMesh(terrain, false);
    }
    river->needsRebuild = true;
    RiverManager::getInstance().generateMesh(river, g_ctx->scene);
    ui.mesh_cache_valid = false;
    scheduleSceneMutationRebuilds(*g_ctx, true);
    if (settings.undo && g_history) {
        g_history->record(std::make_unique<TerrainSnapshotCommand>(
            terrain_name, "Carve river " + river_name, std::move(before),
            captureTerrainSnapshot(*terrain)));
    }
    return Result::success();
}

Result listTerrainRivers(std::vector<TerrainRiverInfo>& out_rivers) {
    if (!g_ctx) return notBound();
    out_rivers.clear();
    for (const auto& river : RiverManager::getInstance().getRivers()) {
        TerrainRiverInfo info;
        info.id = river.id;
        info.name = river.name;
        info.control_point_count = static_cast<int>(river.controlPointCount());
        info.follow_terrain = river.followTerrain;
        out_rivers.push_back(std::move(info));
    }
    return Result::success();
}


namespace {

// Mean relief inside square windows of `w` cells, using a robust 3/97 spread
// so a single spike does not decide what a whole window is worth.
float windowRelief(const std::vector<float>& z, int side, int w) {
    if (w < 2 || side < w) return 0.0f;
    const int tiles = side / w;
    double total = 0.0;
    int counted = 0;
    std::vector<float> scratch(static_cast<size_t>(w) * w);
    for (int ty = 0; ty < tiles; ++ty) {
        for (int tx = 0; tx < tiles; ++tx) {
            size_t n = 0;
            for (int y = 0; y < w; ++y) {
                const int row = ty * w + y;
                for (int x = 0; x < w; ++x)
                    scratch[n++] = z[static_cast<size_t>(row) * side + (tx * w + x)];
            }
            const size_t lo = static_cast<size_t>(n * 0.03);
            const size_t hi = n - 1 - lo;
            std::nth_element(scratch.begin(), scratch.begin() + lo, scratch.begin() + n);
            const float low = scratch[lo];
            std::nth_element(scratch.begin(), scratch.begin() + hi, scratch.begin() + n);
            total += static_cast<double>(scratch[hi] - low);
            ++counted;
        }
    }
    return counted > 0 ? static_cast<float>(total / counted) : 0.0f;
}

float quantileOf(std::vector<float> values, float q) {
    if (values.empty()) return 0.0f;
    const size_t k = (std::min)(values.size() - 1, static_cast<size_t>(
        (values.size() - 1) * static_cast<double>(q)));
    std::nth_element(values.begin(), values.begin() + k, values.end());
    return values[k];
}

} // namespace

Result getTerrainLandformStats(const std::string& terrain_name,
                               TerrainLandformStats& out_stats) {
    out_stats = TerrainLandformStats{};
    if (!g_ctx) return notBound();
    TerrainObject* terrain = TerrainManager::getInstance().getTerrainByName(terrain_name);
    if (!terrain) return Result::fail("terrain not found: " + terrain_name);

    const Heightmap& hm = terrain->heightmap;
    const int w = hm.width, h = hm.height;
    if (w < 16 || h < 16 || static_cast<int>(hm.data.size()) < w * h)
        return Result::success();  // status stays "no_heightfield"

    const float sizeMeters = (std::max)(hm.scale_xz, 1.0f);
    const float heightScale = (std::max)(std::abs(hm.scale_y), 1.0e-4f);
    const int side = (std::min)(w, h);
    const float cell = sizeMeters / static_cast<float>((std::max)(w, h) - 1);

    out_stats.width = w;
    out_stats.height = h;
    out_stats.size_meters = sizeMeters;
    out_stats.cell_meters = cell;

    // Metres, once, here. Everything below is in world units - reporting a
    // "relief" in normalised heightmap units would be a number that changes
    // meaning with the terrain's vertical scale while looking like a length.
    std::vector<float> z(static_cast<size_t>(side) * side, 0.0f);
    for (int y = 0; y < side; ++y)
        for (int x = 0; x < side; ++x)
            z[static_cast<size_t>(y) * side + x] =
                hm.data[static_cast<size_t>(y) * w + x] * heightScale;

    const float lowTail = quantileOf(z, 0.005f);
    const float highTail = quantileOf(z, 0.995f);
    const float relief = highTail - lowTail;
    out_stats.relief_meters = relief;
    out_stats.measured = true;
    if (!(relief > 1.0e-3f)) {
        out_stats.status = "flat";
        return Result::success();
    }
    out_stats.status = "ok";

    // Slope, central differences in metres per metre.
    std::vector<float> slope;
    slope.reserve(static_cast<size_t>(side - 2) * (side - 2));
    for (int y = 1; y < side - 1; ++y) {
        for (int x = 1; x < side - 1; ++x) {
            const size_t i = static_cast<size_t>(y) * side + x;
            const float dzdx = (z[i + 1] - z[i - 1]) / (2.0f * cell);
            const float dzdy = (z[i + side] - z[i - side]) / (2.0f * cell);
            slope.push_back(std::atan(std::sqrt(dzdx * dzdx + dzdy * dzdy)) *
                            57.29577951308232f);
        }
    }
    size_t under3 = 0, under8 = 0;
    for (float s : slope) { if (s < 3.0f) ++under3; if (s < 8.0f) ++under8; }
    const double slopeCount = slope.empty() ? 1.0 : static_cast<double>(slope.size());
    out_stats.flat_fraction = static_cast<float>(static_cast<double>(under3) / slopeCount);
    out_stats.gentle_fraction = static_cast<float>(static_cast<double>(under8) / slopeCount);
    out_stats.median_slope_deg = quantileOf(slope, 0.5f);
    out_stats.p95_slope_deg = quantileOf(slope, 0.95f);
    size_t overForty = 0;
    for (float s : slope) if (s > 40.0f) ++overForty;
    out_stats.cliff_fraction =
        static_cast<float>(static_cast<double>(overForty) / slopeCount);

    // Roughness against slope. Deviation from the 3x3 mean is the smallest
    // honest measure of cell-scale texture; anything wider starts measuring
    // the landform instead of the surface.
    const float gentleCut = quantileOf(slope, 0.20f);
    const float steepCut = quantileOf(slope, 0.80f);
    double steepSum = 0.0, gentleSum = 0.0;
    size_t steepCount = 0, gentleCount = 0;
    for (int y = 1; y < side - 1; ++y) {
        for (int x = 1; x < side - 1; ++x) {
            float mean = 0.0f;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx)
                    mean += z[static_cast<size_t>(y + dy) * side + (x + dx)];
            const float deviation = std::abs(
                z[static_cast<size_t>(y) * side + x] - mean / 9.0f);
            const float s = slope[static_cast<size_t>(y - 1) * (side - 2) + (x - 1)];
            if (s >= steepCut) { steepSum += deviation; ++steepCount; }
            else if (s <= gentleCut) { gentleSum += deviation; ++gentleCount; }
        }
    }
    if (steepCount > 0)
        out_stats.steep_roughness_meters = static_cast<float>(steepSum / steepCount);
    if (gentleCount > 0)
        out_stats.gentle_roughness_meters = static_cast<float>(gentleSum / gentleCount);
    if (out_stats.gentle_roughness_meters > 1.0e-6f)
        out_stats.roughness_slope_ratio =
            out_stats.steep_roughness_meters / out_stats.gentle_roughness_meters;

    // Relief-vs-window ladder. Two things come out of it: where the curve
    // saturates (the broadest landform present) and its slope below that
    // point (the realised Hurst exponent).
    // Starts at 2 cells, not 8. The octaves between the grid and 8 cells are
    // exactly where a surface-detail pass lives, so a ladder that begins above
    // them cannot see whether that pass agrees with the landform under it.
    std::vector<float> windowMeters, windowRelieves;
    for (int wnd = 2; wnd <= side / 2; wnd *= 2) {
        windowMeters.push_back(static_cast<float>(wnd) * cell);
        windowRelieves.push_back(windowRelief(z, side, wnd));
    }
    out_stats.relief_window_meters = windowMeters;
    out_stats.relief_window_relief = windowRelieves;

    // Correlation length by the growth knee: the first window past which
    // doubling the window buys less than 15% more relief. Above the widest
    // landform the field is uncorrelated, so relief keeps creeping up with
    // sample count but stops climbing in steps - that change of regime is the
    // landform width, and it is the only feature of the curve that does not
    // depend on how the relief happened to be normalised.
    float landformScale = windowMeters.empty() ? sizeMeters : windowMeters.back();
    bool saturated = false;
    for (size_t i = 0; i + 1 < windowRelieves.size(); ++i) {
        if (!(windowRelieves[i] > 0.0f)) continue;
        if (windowRelieves[i + 1] / windowRelieves[i] >= 1.15f) continue;
        landformScale = windowMeters[i];
        saturated = true;
        break;
    }
    out_stats.landform_scale_meters = landformScale;
    out_stats.landform_scale_saturated = saturated;
    if (windowRelieves.size() >= 2) {
        const float previous = windowRelieves[windowRelieves.size() - 2];
        if (previous > 0.0f)
            out_stats.broad_growth = windowRelieves.back() / previous;
    }

    // The well-sampled band: below the landform scale, and no wider than an
    // eighth of the tile so at least 64 windows go into every point. The tile
    // limit is not fussiness - the top of any ladder rolls off because the tile
    // ends, and that roll-off is not a property of the terrain.
    const float fitCeiling = (std::min)(landformScale, sizeMeters * 0.125f);
    double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
    int n = 0;
    for (size_t i = 0; i < windowRelieves.size(); ++i) {
        if (windowMeters[i] > fitCeiling + 1.0e-3f ||
            !(windowRelieves[i] > 0.0f)) continue;
        const double lx = std::log(static_cast<double>(windowMeters[i]));
        const double ly = std::log(static_cast<double>(windowRelieves[i]));
        sx += lx; sy += ly; sxx += lx * lx; sxy += lx * ly; ++n;
    }
    out_stats.hurst_sample_count = n;
    if (n >= 2) {
        const double denom = n * sxx - sx * sx;
        if (std::abs(denom) > 1.0e-12)
            out_stats.realised_hurst = static_cast<float>((n * sxy - sx * sy) / denom);
    }

    // Macro-micro coherence: how far the worst octave departs from that law.
    if (n >= 3 && out_stats.realised_hurst > 0.0f) {
        float worst = 0.0f, worstAt = 0.0f, worstSigned = 0.0f;
        for (size_t i = 0; i + 1 < windowRelieves.size(); ++i) {
            if (windowMeters[i + 1] > fitCeiling + 1.0e-3f) break;
            if (!(windowRelieves[i] > 0.0f)) continue;
            // log2 of the growth over one doubling IS the local exponent.
            const float local = std::log2(windowRelieves[i + 1] / windowRelieves[i]);
            const float deviation = local - out_stats.realised_hurst;
            if (std::abs(deviation) > worst) {
                worst = std::abs(deviation);
                worstSigned = deviation;
                worstAt = windowMeters[i];
            }
        }
        out_stats.spectrum_kink = worst;
        out_stats.spectrum_kink_signed = worstSigned;
        out_stats.spectrum_kink_meters = worstAt;
    }

    // Local relief over an 8x8 grid of tiles.
    const int tile = side / 8;
    if (tile >= 4) {
        std::vector<float> tileRelief;
        std::vector<float> scratch(static_cast<size_t>(tile) * tile);
        for (int ty = 0; ty < 8; ++ty) {
            for (int tx = 0; tx < 8; ++tx) {
                size_t k = 0;
                for (int y = 0; y < tile; ++y)
                    for (int x = 0; x < tile; ++x)
                        scratch[k++] = z[static_cast<size_t>(ty * tile + y) * side +
                                         (tx * tile + x)];
                tileRelief.push_back(quantileOf(scratch, 0.95f) - quantileOf(scratch, 0.05f));
            }
        }
        out_stats.local_relief_p10 = quantileOf(tileRelief, 0.10f);
        out_stats.local_relief_p90 = quantileOf(tileRelief, 0.90f);
        out_stats.local_relief_ratio = out_stats.local_relief_p90 /
            (std::max)(out_stats.local_relief_p10, 1.0e-4f);
    }

    // Hypsometry over the same robust band the relief used, so one outlier
    // pixel cannot move the bands the areas are counted in.
    size_t low = 0, mid = 0;
    double sum = 0.0;
    for (float value : z) {
        const float t = (value - lowTail) / relief;
        if (t < 0.20f) ++low;
        if (t >= 0.40f && t < 0.60f) ++mid;
        sum += static_cast<double>(t);
    }
    const double total = static_cast<double>(z.size());
    out_stats.lowland_fraction = static_cast<float>(static_cast<double>(low) / total);
    out_stats.midland_fraction = static_cast<float>(static_cast<double>(mid) / total);
    out_stats.hypsometric_integral = static_cast<float>(sum / total);
    return Result::success();
}

} // namespace rtapi
