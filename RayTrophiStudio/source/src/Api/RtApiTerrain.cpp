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

Result createTerrain(const std::string& requested_name, int resolution, float size,
                     float height_scale, TerrainInfo& out_info) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (resolution < 64 || resolution > 4096)
        return Result::fail("resolution must be in the range [64, 4096]");
    if (!std::isfinite(size) || size <= 0.0f)
        return Result::fail("size must be finite and positive");
    if (!std::isfinite(height_scale) || height_scale <= 0.0f)
        return Result::fail("height_scale must be finite and positive");

    const std::string name = uniqueTerrainName(requested_name);
    TerrainObject* terrain = TerrainManager::getInstance().createTerrain(g_ctx->scene, resolution, size);
    if (!terrain) return Result::fail("failed to create terrain: " + name);

    terrain->name = name;
    terrain->heightmap.scale_y = height_scale;
    if (terrain->flatMesh) terrain->flatMesh->nodeName = name;
    TerrainManager::getInstance().updateTerrainMesh(terrain, false);

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
                          bool replace_graph) {
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
    if (kind == "default") {
        if (!replace_graph && graph.nodeCount() != 0)
            return Result::fail("default preset requires replace_graph=True for a non-empty graph");
        graph.createDefaultGraph(terrain);
    } else if (kind == "snowy_mountain_valley") {
        if (!replace_graph && graph.nodeCount() != 0)
            return Result::fail("snowy_mountain_valley requires replace_graph=True for a non-empty graph");
        graph.createSnowyMountainValleyGraph(terrain);
    } else if (kind == "snow_layer") {
        if (graph.nodeCount() == 0) graph.createDefaultGraph(terrain);
        if (!graph.addSnowLayerSetup()) return Result::fail("failed to add snow layer setup");
    } else if (kind == "river_network") {
        if (graph.nodeCount() == 0) graph.createDefaultGraph(terrain);
        if (!graph.addRiverNetworkSetup()) return Result::fail("failed to add river network setup");
    } else {
        return Result::fail("unknown terrain preset '" + preset +
                            "' (expected default|snow_layer|snowy_mountain_valley|river_network)");
    }
    graph.markAllDirty();
    return Result::success();
}

Result calculateTerrainFlow(const std::string& terrain_name) {
    if (!g_ctx) return notBound();
    TerrainObject* terrain = TerrainManager::getInstance().getTerrainByName(terrain_name);
    if (!terrain) return Result::fail("terrain not found: " + terrain_name);
    if (terrain->nodeGraph && terrain->nodeGraph->isEvaluatingAsync())
        return Result::fail("terrain evaluation is still running: " + terrain_name);
    TerrainManager::getInstance().calculateFlowMap(terrain);
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

} // namespace rtapi
