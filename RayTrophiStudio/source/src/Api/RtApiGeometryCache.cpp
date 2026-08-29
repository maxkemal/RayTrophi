#include "RtApiInternal.h"

#include "Animation/GeometryCache.h"
#include "GeometryNodesV2.h"
#include "ProjectManager.h"
#include "TriangleMesh.h"

#include <algorithm>
#include <optional>
#include <sstream>

namespace rtapi {
namespace {

std::shared_ptr<TriangleMesh> findFlatMesh(const std::string& name) {
    if (!g_ctx) return {};
    for (const auto& object : g_ctx->scene.world.objects) {
        auto mesh = std::dynamic_pointer_cast<TriangleMesh>(object);
        if (mesh && mesh->nodeName == name) return mesh;
    }
    return {};
}

uint64_t graphSignature(const GeometryNodesV2::GeometryNodeGraphV2* graph) {
    if (!graph) return 0;
    nlohmann::json state;
    GeometryNodesV2::serializeGeometryGraph(*graph, state, nullptr);
    const std::string text = state.dump();
    uint64_t hash = 1469598103934665603ull;
    for (const unsigned char value : text) {
        hash ^= static_cast<uint64_t>(value);
        hash *= 1099511628211ull;
    }
    return hash;
}

void fillInfo(const Animation::GeometryCacheClip& clip, const TriangleMesh* mesh,
              uint64_t currentSourceSignature, GeometryCacheInfo& out) {
    out = {};
    out.object_name = clip.object_name;
    out.start_frame = clip.start_frame;
    out.end_frame = clip.end_frame;
    out.frame_step = clip.frame_step;
    out.vertex_count = clip.vertex_count;
    out.sample_count = clip.samples.size();
    out.memory_bytes = Animation::geometryCacheMemoryBytes(clip);
    out.enabled = clip.enabled;
    out.topology_valid = mesh && Animation::geometryTopologyHash(*mesh) == clip.topology_hash;
    out.source_stale = clip.source_signature != 0 && currentSourceSignature != 0 &&
        clip.source_signature != currentSourceSignature;
}

void publishGeometryCacheMutation() {
    scheduleSceneMutationRebuilds(*g_ctx, true);
    g_ctx->start_render = true;
    ProjectManager::getInstance().markModified();
}

} // namespace

Result bakeGeometryCache(const std::string& objectName, int startFrame,
                         int endFrame, int frameStep, GeometryCacheInfo& out) {
    out = {};
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (objectName.empty()) return Result::fail("object_name must not be empty");
    if (startFrame > endFrame) return Result::fail("start_frame must be <= end_frame");
    if (frameStep < 1 || frameStep > 1000) return Result::fail("frame_step must be between 1 and 1000");
    const int64_t sampleCount = 1 +
        (static_cast<int64_t>(endFrame) - static_cast<int64_t>(startFrame)) / frameStep;
    if (sampleCount > 10000) return Result::fail("geometry cache bake exceeds 10000 samples");
    if (!findFlatMesh(objectName)) return Result::fail("flat mesh not found: " + objectName);

    GeometryNodesV2::GeometryNodeGraphV2* graph = nullptr;
    const auto graphIt = g_ctx->scene.geometry_node_graphs.find(objectName);
    if (graphIt != g_ctx->scene.geometry_node_graphs.end() && graphIt->second)
        graph = graphIt->second.get();

    const int restoreFrame = g_ctx->scene.timeline.current_frame;
    const bool restoreLivePreview = graph ? graph->liveCurvePreview : false;
    if (graph) graph->liveCurvePreview = false;
    std::optional<Animation::GeometryCacheClip> previousClip;
    const auto previousIt = g_ctx->scene.geometry_caches.find(objectName);
    if (previousIt != g_ctx->scene.geometry_caches.end()) {
        previousClip = previousIt->second;
        g_ctx->scene.geometry_caches.erase(previousIt);
    }

    Animation::GeometryCacheClip clip;
    clip.object_name = objectName;
    clip.start_frame = startFrame;
    clip.end_frame = endFrame;
    clip.frame_step = frameStep;
    clip.source_signature = graphSignature(graph);
    clip.enabled = true;
    std::string failure;

    for (int frame = startFrame; frame <= endFrame; frame += frameStep) {
        g_ctx->scene.timeline.current_frame = frame;
        ui.processAnimations(*g_ctx);
        if (graph && !ui.evaluateGeometryGraph(*g_ctx, objectName, *graph, true, true)) {
            failure = "geometry graph evaluation failed at frame " + std::to_string(frame);
            break;
        }
        const auto mesh = findFlatMesh(objectName);
        if (!mesh) { failure = "target mesh disappeared at frame " + std::to_string(frame); break; }
        const uint64_t topology = Animation::geometryTopologyHash(*mesh);
        if (clip.samples.empty()) {
            clip.vertex_count = mesh->num_vertices();
            clip.topology_hash = topology;
        } else if (mesh->num_vertices() != clip.vertex_count || topology != clip.topology_hash) {
            failure = "topology changed at frame " + std::to_string(frame) +
                "; fixed-topology cache bake was cancelled";
            break;
        }
        Animation::GeometryCacheSample sample;
        if (!Animation::captureGeometryCacheSample(*mesh, frame, sample, &failure)) break;
        clip.samples.push_back(std::move(sample));
        if (frame > endFrame - frameStep) break; // overflow-safe final iteration
    }

    if (graph) graph->liveCurvePreview = restoreLivePreview;
    g_ctx->scene.timeline.current_frame = restoreFrame;
    ui.processAnimations(*g_ctx);

    if (!failure.empty() || clip.samples.empty()) {
        if (previousClip) g_ctx->scene.geometry_caches[objectName] = std::move(*previousClip);
        if (graph) ui.evaluateGeometryGraph(*g_ctx, objectName, *graph, true, true);
        const auto restoredIt = g_ctx->scene.geometry_caches.find(objectName);
        if (const auto mesh = findFlatMesh(objectName);
            mesh && restoredIt != g_ctx->scene.geometry_caches.end()) {
            Animation::applyGeometryCacheFrame(restoredIt->second, *mesh, restoreFrame);
        }
        scheduleSceneMutationRebuilds(*g_ctx, true);
        return Result::fail(failure.empty() ? "geometry cache bake produced no samples" : failure);
    }

    g_ctx->scene.geometry_caches[objectName] = std::move(clip);
    const auto mesh = findFlatMesh(objectName);
    if (mesh) Animation::applyGeometryCacheFrame(
        g_ctx->scene.geometry_caches[objectName], *mesh, restoreFrame);
    fillInfo(g_ctx->scene.geometry_caches[objectName], mesh.get(), graphSignature(graph), out);
    publishGeometryCacheMutation();
    return Result::success();
}

Result getGeometryCacheInfo(const std::string& objectName, GeometryCacheInfo& out) {
    out = {};
    if (!g_ctx) return notBound();
    const auto it = g_ctx->scene.geometry_caches.find(objectName);
    if (it == g_ctx->scene.geometry_caches.end())
        return Result::fail("geometry cache not found: " + objectName);
    GeometryNodesV2::GeometryNodeGraphV2* graph = nullptr;
    const auto graphIt = g_ctx->scene.geometry_node_graphs.find(objectName);
    if (graphIt != g_ctx->scene.geometry_node_graphs.end() && graphIt->second)
        graph = graphIt->second.get();
    const auto mesh = findFlatMesh(objectName);
    fillInfo(it->second, mesh.get(), graphSignature(graph), out);
    return Result::success();
}

Result setGeometryCacheEnabled(const std::string& objectName, bool enabled) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    const auto it = g_ctx->scene.geometry_caches.find(objectName);
    if (it == g_ctx->scene.geometry_caches.end())
        return Result::fail("geometry cache not found: " + objectName);
    const bool previousEnabled = it->second.enabled;
    it->second.enabled = enabled;
    const auto graphIt = g_ctx->scene.geometry_node_graphs.find(objectName);
    if (!enabled && graphIt != g_ctx->scene.geometry_node_graphs.end() && graphIt->second) {
        if (!ui.evaluateGeometryGraph(*g_ctx, objectName, *graphIt->second, true, true)) {
            it->second.enabled = previousEnabled;
            return Result::fail("could not restore live geometry graph output");
        }
    } else if (enabled) {
        const auto mesh = findFlatMesh(objectName);
        if (!mesh) {
            it->second.enabled = previousEnabled;
            return Result::fail("flat mesh not found: " + objectName);
        }
        const auto applied = Animation::applyGeometryCacheFrame(
            it->second, *mesh, g_ctx->scene.timeline.current_frame);
        if (!applied.changed) {
            it->second.enabled = previousEnabled;
            return Result::fail(applied.error);
        }
    }
    publishGeometryCacheMutation();
    return Result::success();
}

Result clearGeometryCache(const std::string& objectName) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    const auto cacheIt = g_ctx->scene.geometry_caches.find(objectName);
    if (cacheIt == g_ctx->scene.geometry_caches.end())
        return Result::fail("geometry cache not found: " + objectName);
    Animation::GeometryCacheClip removed = std::move(cacheIt->second);
    g_ctx->scene.geometry_caches.erase(cacheIt);
    const auto graphIt = g_ctx->scene.geometry_node_graphs.find(objectName);
    if (graphIt != g_ctx->scene.geometry_node_graphs.end() && graphIt->second &&
        !ui.evaluateGeometryGraph(*g_ctx, objectName, *graphIt->second, true, true)) {
        g_ctx->scene.geometry_caches[objectName] = std::move(removed);
        if (const auto mesh = findFlatMesh(objectName))
            Animation::applyGeometryCacheFrame(g_ctx->scene.geometry_caches[objectName],
                                               *mesh, g_ctx->scene.timeline.current_frame);
        return Result::fail("could not restore live geometry graph output");
    }
    publishGeometryCacheMutation();
    return Result::success();
}

Result geometryCacheSelfTest(std::string& outDetails) {
    return Animation::runGeometryCacheSelfTest(&outDetails)
        ? Result::success() : Result::fail("geometry cache self-test failed: " + outDetails);
}

} // namespace rtapi
