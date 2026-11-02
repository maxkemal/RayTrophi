#include "Fluid/ColliderGpuQuery.h"

#include "Fluid/MatterGrainColliderBvh.h"
#include "SurfaceMeshCache.h"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace RayTrophiSim::Fluid {
namespace {

constexpr uint64_t kEvictAfterQueries = 240;

struct QueryConstants {
    float row[3][4];
    int32_t dims[4];
    int32_t base[4];
};
static_assert(sizeof(QueryConstants) == 80, "sim_collider_bvh_query push ABI");

ComputeBufferHandle makeBuffer(SimulationComputeContext& compute, const char* name,
                               std::size_t bytes) {
    ComputeBufferDesc desc;
    desc.debug_name = name;
    desc.size_bytes = std::max<std::size_t>(bytes, 16);
    desc.usage = ComputeBufferUsage::Storage | ComputeBufferUsage::Upload |
                 ComputeBufferUsage::Download | ComputeBufferUsage::ReadWrite;
    return compute.createBuffer(desc);
}

// A handle from a backend that was since rebuilt keeps its nonzero id but no
// longer names a buffer: every dispatch with it then fails. The other device
// stages test the same way (FluidGpuOccupancy).
bool alive(SimulationComputeContext& compute, const ComputeBufferHandle& handle) {
    return handle.valid() && handle.backend == compute.backendType() &&
        compute.getBufferSize(handle) != 0;
}

} // namespace

bool ColliderGpuQuery::ensureMesh(SimulationComputeContext& compute, const void* key,
                                  uint64_t revision,
                                  const std::vector<SurfaceMeshTriangle>& triangles,
                                  Mesh*& mesh, std::string& error) {
    auto found = meshes_.find(key);
    if (found != meshes_.end() && found->second.revision == revision &&
        found->second.triangles == triangles.size() &&
        alive(compute, found->second.nodes) && alive(compute, found->second.vertices)) {
        mesh = &found->second;
        return true;
    }
    // The grain BVH builder rejects degenerate faces; a scanned or decimated
    // mesh has some, and they cannot be the nearest surface anyway.
    std::vector<SurfaceMeshTriangle> usable;
    usable.reserve(triangles.size());
    for (const auto& t : triangles) {
        const float area = (t.p1 - t.p0).cross(t.p2 - t.p0).length();
        if (std::isfinite(area) && area >= 1e-10f) {
            usable.push_back(t);
        }
    }
    if (usable.empty()) {
        error = "collider GPU query: no usable triangles";
        return false;
    }
    MatterGrainColliderBvh bvh;
    if (!buildMatterGrainColliderBvh(usable, bvh, error)) {
        error = "collider GPU query: " + error;
        return false;
    }
    if (found != meshes_.end()) {
        compute.destroyBuffer(found->second.nodes);
        compute.destroyBuffer(found->second.vertices);
        meshes_.erase(found);
    }
    Mesh entry;
    const std::size_t node_bytes = bvh.nodes.size() * sizeof(MatterGrainBvhNode);
    const std::size_t vertex_bytes = bvh.vertices.size() * sizeof(Vec3);
    const std::size_t limit = compute.caps().max_storage_buffer_bytes;
    if (limit && std::max(node_bytes, vertex_bytes) > limit) {
        error = "collider GPU query: mesh of " + std::to_string(usable.size()) +
            " triangles exceeds the device storage-buffer limit";
        return false;
    }
    entry.nodes = makeBuffer(compute, "ColliderQueryNodes", node_bytes);
    entry.vertices = makeBuffer(compute, "ColliderQueryVertices", vertex_bytes);
    if (!entry.nodes.valid() || !entry.vertices.valid()) {
        compute.destroyBuffer(entry.nodes);
        compute.destroyBuffer(entry.vertices);
        error = "collider GPU query: buffer allocation failed";
        return false;
    }
    compute.beginTransferBatch();
    bool ok = compute.uploadBuffer(entry.nodes, bvh.nodes.data(), node_bytes);
    ok = compute.uploadBuffer(entry.vertices, bvh.vertices.data(), vertex_bytes) && ok;
    if (!compute.endTransferBatch() || !ok) {
        compute.destroyBuffer(entry.nodes);
        compute.destroyBuffer(entry.vertices);
        error = "collider GPU query: BVH upload failed";
        return false;
    }
    entry.revision = revision;
    entry.triangles = triangles.size();
    mesh = &(meshes_[key] = entry);
    return true;
}

bool ColliderGpuQuery::query(SimulationComputeContext& compute, const void* key,
                             uint64_t revision,
                             const std::vector<SurfaceMeshTriangle>& triangles,
                             const ColliderQueryLattice& lattice, std::vector<float>& out,
                             std::string& error) {
    const auto start = std::chrono::steady_clock::now();
    error.clear();
    if (!compute.supportsDispatch() ||
        compute.backendType() != ComputeBackendType::VulkanCompute) {
        error = "collider GPU query needs Vulkan compute";
        return false;
    }
    const int64_t count = static_cast<int64_t>(lattice.dims[0]) * lattice.dims[1] *
        lattice.dims[2];
    if (lattice.dims[0] <= 0 || lattice.dims[1] <= 0 || lattice.dims[2] <= 0 ||
        count > 0x7fffffff) {
        error = "collider GPU query: invalid lattice";
        return false;
    }
    Mesh* mesh = nullptr;
    if (!ensureMesh(compute, key, revision, triangles, mesh, error)) {
        return false;
    }
    mesh->last_use = ++use_counter_;
    const std::size_t bytes = static_cast<std::size_t>(count) * sizeof(float);
    if (!alive(compute, output_) || output_bytes_ < bytes) {
        compute.destroyBuffer(output_);
        output_ = makeBuffer(compute, "ColliderQueryOutput", bytes);
        output_bytes_ = output_.valid() ? bytes : 0;
        if (!output_.valid()) {
            error = "collider GPU query: output allocation failed";
            return false;
        }
    }
    QueryConstants constants{};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 4; ++c) {
            constants.row[r][c] = lattice.m[r][c];
        }
        constants.dims[r] = lattice.dims[r];
        constants.base[r] = lattice.base[r];
    }
    constants.dims[3] = lattice.signed_distance ? 1 : 0;
    constants.base[3] = static_cast<int32_t>(count);
    const ComputeBufferHandle buffers[] = {mesh->nodes, mesh->vertices, output_};
    ComputeDispatch cmd;
    cmd.kernel = "sim_collider_bvh_query";
    cmd.buffers = buffers;
    cmd.buffer_count = 3;
    cmd.constants = &constants;
    cmd.constants_size = sizeof(constants);
    const uint64_t groups = (static_cast<uint64_t>(count) + 255u) / 256u;
    cmd.groups.groups_x = static_cast<uint32_t>(std::min<uint64_t>(groups, 65535u));
    cmd.groups.groups_y = static_cast<uint32_t>((groups + cmd.groups.groups_x - 1) /
                                                cmd.groups.groups_x);
    out.resize(static_cast<std::size_t>(count));
    // The download batch records its copy after the dispatch in the same
    // submission (as the P2G readback does).
    const bool dispatched = compute.dispatch(cmd);
    bool downloaded = false;
    bool flushed = false;
    if (dispatched) {
        compute.beginTransferBatch();
        downloaded = compute.downloadBuffer(output_, out.data(), bytes);
        flushed = compute.endTransferBatch();
    }
    const bool ok = dispatched && downloaded && flushed;
    // Meshes no query has touched for a while (a removed or replaced collider).
    for (auto it = meshes_.begin(); it != meshes_.end();) {
        if (use_counter_ - it->second.last_use > kEvictAfterQueries) {
            compute.destroyBuffer(it->second.nodes);
            compute.destroyBuffer(it->second.vertices);
            it = meshes_.erase(it);
        } else {
            ++it;
        }
    }
    last_ms_ = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    if (!ok) {
        // Which step failed, and the sizes, so the next report is actionable.
        error = std::string("collider GPU query ") +
            (!dispatched ? "dispatch failed" : !downloaded ? "download failed" : "flush failed") +
            " (" + std::to_string(count) + " points, " + std::to_string(mesh->triangles) +
            " triangles, dims " + std::to_string(lattice.dims[0]) + "x" +
            std::to_string(lattice.dims[1]) + "x" + std::to_string(lattice.dims[2]) + ")";
        // Drop the mesh: a stale or half-uploaded BVH must not be reused.
        compute.destroyBuffer(mesh->nodes);
        compute.destroyBuffer(mesh->vertices);
        meshes_.erase(key);
    }
    return ok;
}

void ColliderGpuQuery::release(SimulationComputeContext* compute) {
    if (compute) {
        for (auto& entry : meshes_) {
            compute->destroyBuffer(entry.second.nodes);
            compute->destroyBuffer(entry.second.vertices);
        }
        compute->destroyBuffer(output_);
    }
    meshes_.clear();
    output_ = {};
    output_bytes_ = 0;
}

} // namespace RayTrophiSim::Fluid
