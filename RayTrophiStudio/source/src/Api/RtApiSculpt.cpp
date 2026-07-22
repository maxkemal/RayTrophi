/* RayTrophi Studio - deterministic flat-mesh sculpt scripting (Faz 5.4c) */

#include "RtApiInternal.h"
#include "SceneHistory.h"
#include "GeometryNodesV2.h"
#include "ProjectManager.h"
#include "TriangleMesh.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace rtapi {
namespace {

TriangleMesh* findSculptMesh(const std::string& name) {
    if (!g_ctx) return nullptr;
    for (const auto& object : g_ctx->scene.world.objects) {
        if (auto mesh = std::dynamic_pointer_cast<TriangleMesh>(object))
            if (mesh->nodeName == name && mesh->geometry) return mesh.get();
    }
    return nullptr;
}

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

float brushWeight(float distance, float radius, float falloff) {
    if (distance >= radius) return 0.0f;
    float t = 1.0f - distance / radius;
    const float smooth = t * t * (3.0f - 2.0f * t);
    return t + (smooth - t) * std::clamp(falloff, 0.0f, 1.0f);
}

struct WeldKey {
    int64_t x = 0, y = 0, z = 0;
    bool operator==(const WeldKey& rhs) const { return x == rhs.x && y == rhs.y && z == rhs.z; }
};
struct WeldHash {
    size_t operator()(const WeldKey& k) const {
        size_t h = std::hash<int64_t>{}(k.x);
        h ^= std::hash<int64_t>{}(k.y) + 0x9e3779b9u + (h << 6u) + (h >> 2u);
        h ^= std::hash<int64_t>{}(k.z) + 0x9e3779b9u + (h << 6u) + (h >> 2u);
        return h;
    }
};
WeldKey weldKey(const Vec3& p) {
    constexpr double scale = 1000000.0;
    return {static_cast<int64_t>(std::llround(p.x * scale)),
            static_cast<int64_t>(std::llround(p.y * scale)),
            static_cast<int64_t>(std::llround(p.z * scale))};
}

struct WeldTopology {
    std::vector<int> vertex_group;
    std::vector<std::vector<uint32_t>> members;
    std::vector<std::vector<int>> neighbors;
};

WeldTopology buildTopology(const DNA::GeometryDetail& geom, const Vec3* positions) {
    WeldTopology topo;
    const size_t count = geom.get_vertex_count();
    topo.vertex_group.resize(count, -1);
    std::unordered_map<WeldKey, int, WeldHash> groups;
    groups.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        const WeldKey key = weldKey(positions[i]);
        auto [it, inserted] = groups.emplace(key, static_cast<int>(topo.members.size()));
        if (inserted) topo.members.emplace_back();
        topo.vertex_group[i] = it->second;
        topo.members[it->second].push_back(static_cast<uint32_t>(i));
    }
    std::vector<std::unordered_set<int>> links(topo.members.size());
    const auto& indices = geom.indices;
    for (size_t i = 0; i + 2 < indices.size(); i += 3) {
        const uint32_t a = indices[i], b = indices[i + 1], c = indices[i + 2];
        if (a >= count || b >= count || c >= count) continue;
        const int ga = topo.vertex_group[a], gb = topo.vertex_group[b], gc = topo.vertex_group[c];
        if (ga != gb) { links[ga].insert(gb); links[gb].insert(ga); }
        if (gb != gc) { links[gb].insert(gc); links[gc].insert(gb); }
        if (gc != ga) { links[gc].insert(ga); links[ga].insert(gc); }
    }
    topo.neighbors.resize(links.size());
    for (size_t i = 0; i < links.size(); ++i)
        topo.neighbors[i].assign(links[i].begin(), links[i].end());
    return topo;
}

class SculptMaskCommand final : public SceneCommand {
public:
    SculptMaskCommand(std::string name, std::vector<float> before, std::vector<float> after)
        : name_(std::move(name)), before_(std::move(before)), after_(std::move(after)) {}
    void execute(UIContext& ctx) override { apply(ctx, after_); }
    void undo(UIContext& ctx) override { apply(ctx, before_); }
    Type getType() const override { return Type::Transform; }
    std::string getDescription() const override { return "Sculpt Mask " + name_; }
private:
    std::string name_;
    std::vector<float> before_, after_;
    void apply(UIContext& ctx, const std::vector<float>& values) {
        TriangleMesh* mesh = nullptr;
        for (const auto& object : ctx.scene.world.objects)
            if (auto tm = std::dynamic_pointer_cast<TriangleMesh>(object); tm && tm->nodeName == name_) { mesh = tm.get(); break; }
        if (!mesh || !mesh->geometry || mesh->geometry->get_vertex_count() != values.size()) return;
        float* mask = GeometryNodesV2::ensureFloatAttribute(*mesh->geometry, "sculpt_mask");
        if (!mask) return;
        std::copy(values.begin(), values.end(), mask);
        ProjectManager::getInstance().markModified();
        ctx.renderer.resetCPUAccumulation();
    }
};

Result prepare(const std::string& object_name, TriangleMesh*& mesh, Vec3*& positions,
               Vec3*& normals, float*& mask) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    mesh = findSculptMesh(object_name);
    if (!mesh) return Result::fail("sculpt requires a flat mesh object: " + object_name);
    positions = mesh->geometry->get_attribute_data_mut<Vec3>("P_orig");
    normals = mesh->geometry->get_attribute_data_mut<Vec3>("N_orig");
    if (!positions || !normals) return Result::fail("sculpt target has no P_orig/N_orig buffers: " + object_name);
    mask = GeometryNodesV2::ensureFloatAttribute(*mesh->geometry, "sculpt_mask");
    if (!mask) return Result::fail("failed to create sculpt_mask attribute");
    return Result::success();
}

void finishGeometry(TriangleMesh& mesh) {
    GeometryNodesV2::recomputeOrigNormals(*mesh.geometry);
    GeometryNodesV2::rebakeFromOrig(mesh);
    scheduleSceneMutationRebuilds(*g_ctx, true);
    ProjectManager::getInstance().markModified();
}

} // namespace

Result getSculptInfo(const std::string& object_name, SculptInfo& out_info) {
    if (!g_ctx) return notBound();
    TriangleMesh* mesh = findSculptMesh(object_name);
    if (!mesh) return Result::fail("sculpt requires a flat mesh object: " + object_name);
    out_info = {}; out_info.object_name = object_name;
    out_info.vertex_count = mesh->geometry->get_vertex_count();
    const float* mask = mesh->geometry->get_attribute_data<float>("sculpt_mask");
    out_info.has_mask = mask != nullptr;
    if (mask && out_info.vertex_count) {
        out_info.mask_min = 1.0f; out_info.mask_max = 0.0f;
        for (size_t i = 0; i < out_info.vertex_count; ++i) {
            out_info.mask_min = std::min(out_info.mask_min, mask[i]);
            out_info.mask_max = std::max(out_info.mask_max, mask[i]);
        }
    }
    return Result::success();
}

Result applySculptStroke(const std::string& object_name, const SculptStrokeSettings& settings) {
    if (settings.points.empty()) return Result::fail("sculpt stroke requires at least one world-space point");
    if (settings.radius <= 0.0f) return Result::fail("sculpt radius must be greater than zero");
    if (settings.falloff < 0.0f || settings.falloff > 1.0f)
        return Result::fail("sculpt falloff must be between 0 and 1");
    const std::string tool = lower(settings.tool);
    if (tool != "draw" && tool != "inflate" && tool != "flatten" &&
        tool != "smooth" && tool != "stamp" && tool != "noise")
        return Result::fail("unknown sculpt tool: " + settings.tool);

    TriangleMesh* mesh = nullptr; Vec3* positions = nullptr; Vec3* normals = nullptr; float* mask = nullptr;
    Result r = prepare(object_name, mesh, positions, normals, mask); if (!r.ok) return r;
    const size_t count = mesh->geometry->get_vertex_count();
    std::vector<Vec3> before_pos(positions, positions + count);
    std::vector<Vec3> before_nrm(normals, normals + count);
    const Matrix4x4 world = mesh->transform ? mesh->transform->getFinal() : Matrix4x4::identity();
    const Matrix4x4 world_to_local = world.inverse();
    const Matrix4x4 normal_matrix = world.inverse().transpose();
    Vec3 direction = settings.direction;
    if (direction.length() > 1e-8f) direction = direction / direction.length();
    else direction = Vec3(0.0f, 1.0f, 0.0f);

    if (tool == "smooth") {
        WeldTopology topo = buildTopology(*mesh->geometry, positions);
        for (const Vec3& center : settings.points) {
            std::vector<Vec3> group_pos(topo.members.size());
            for (size_t g = 0; g < topo.members.size(); ++g) group_pos[g] = positions[topo.members[g][0]];
            std::vector<Vec3> next = group_pos;
            for (size_t g = 0; g < topo.members.size(); ++g) {
                const Vec3 world_pos = world.transform_point(group_pos[g]);
                const float weight = brushWeight((world_pos - center).length(), settings.radius, settings.falloff);
                if (weight <= 0.0f || topo.neighbors[g].empty()) continue;
                Vec3 average(0.0f); for (int n : topo.neighbors[g]) average = average + group_pos[n];
                average = average / static_cast<float>(topo.neighbors[g].size());
                const float protection = settings.use_mask ? mask[topo.members[g][0]] : 0.0f;
                next[g] = Vec3::mix(group_pos[g], average,
                                    std::clamp(std::abs(settings.strength) * weight * (1.0f - protection), 0.0f, 1.0f));
            }
            for (size_t g = 0; g < topo.members.size(); ++g)
                for (uint32_t vertex : topo.members[g]) positions[vertex] = next[g];
        }
    } else {
        for (const Vec3& center : settings.points) {
            for (size_t i = 0; i < count; ++i) {
                const Vec3 world_pos = world.transform_point(positions[i]);
                float weight = brushWeight((world_pos - center).length(), settings.radius, settings.falloff);
                if (weight <= 0.0f) continue;
                if (settings.use_mask) weight *= 1.0f - std::clamp(mask[i], 0.0f, 1.0f);
                Vec3 world_delta(0.0f);
                Vec3 world_normal = normal_matrix.transform_vector(normals[i]);
                if (world_normal.length() > 1e-8f) world_normal = world_normal / world_normal.length();
                else world_normal = direction;
                if (tool == "flatten") {
                    const float signed_distance = Vec3::dot(world_pos - center, direction);
                    world_delta = direction * (-signed_distance * std::clamp(std::abs(settings.strength), 0.0f, 1.0f) * weight);
                } else {
                    float amount = settings.strength * weight;
                    if (tool == "noise") {
                        const WeldKey key = weldKey(before_pos[i]);
                        uint64_t h = static_cast<uint64_t>(key.x) * 73856093ull ^ static_cast<uint64_t>(key.y) * 19349663ull ^
                                     static_cast<uint64_t>(key.z) * 83492791ull ^ settings.seed;
                        h ^= h >> 13u; h *= 1274126177ull; h ^= h >> 16u;
                        amount *= (static_cast<float>(h & 0xffffu) / 32767.5f - 1.0f);
                    }
                    world_delta = (tool == "stamp" ? direction : world_normal) * amount;
                }
                positions[i] = positions[i] + world_to_local.transform_vector(world_delta);
            }
        }
    }

    finishGeometry(*mesh);
    positions = mesh->geometry->get_attribute_data_mut<Vec3>("P_orig");
    Vec3* after_normals = mesh->geometry->get_attribute_data_mut<Vec3>("N_orig");
    if (!positions || !after_normals) return Result::fail("sculpt buffers were lost while rebuilding geometry");
    std::vector<FlatSculptVertexState> states;
    states.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        if ((positions[i] - before_pos[i]).length_squared() > 1e-20f ||
            (after_normals[i] - before_nrm[i]).length_squared() > 1e-20f) {
            states.push_back({static_cast<uint32_t>(i), before_pos[i], before_nrm[i],
                              positions[i], after_normals[i]});
        }
    }
    if (states.empty()) return Result::fail("sculpt stroke did not affect any vertices");
    if (settings.undo && g_history)
        g_history->record(std::make_unique<FlatSculptEditCommand>(object_name, std::move(states)));
    return Result::success();
}

Result paintSculptMask(const std::string& object_name, const std::vector<Vec3>& points,
                       float radius, float value, float strength, bool undo) {
    if (points.empty()) return Result::fail("sculpt mask stroke requires at least one point");
    if (radius <= 0.0f) return Result::fail("sculpt mask radius must be greater than zero");
    if (value < 0.0f || value > 1.0f || strength < 0.0f || strength > 1.0f)
        return Result::fail("sculpt mask value/strength must be between 0 and 1");
    TriangleMesh* mesh = nullptr; Vec3* positions = nullptr; Vec3* normals = nullptr; float* mask = nullptr;
    Result r = prepare(object_name, mesh, positions, normals, mask); if (!r.ok) return r;
    const size_t count = mesh->geometry->get_vertex_count();
    std::vector<float> before(mask, mask + count);
    const Matrix4x4 world = mesh->transform ? mesh->transform->getFinal() : Matrix4x4::identity();
    for (const Vec3& center : points) for (size_t i = 0; i < count; ++i) {
        const float weight = brushWeight((world.transform_point(positions[i]) - center).length(), radius, 0.75f) * strength;
        if (weight > 0.0f) mask[i] += (value - mask[i]) * weight;
    }
    std::vector<float> after(mask, mask + count);
    if (undo && g_history) g_history->record(std::make_unique<SculptMaskCommand>(object_name, before, after));
    ProjectManager::getInstance().markModified();
    g_ctx->renderer.resetCPUAccumulation();
    return Result::success();
}

Result applySculptMaskOperation(const std::string& object_name, const std::string& operation,
                                unsigned int seed, bool undo) {
    TriangleMesh* mesh = nullptr; Vec3* positions = nullptr; Vec3* normals = nullptr; float* mask = nullptr;
    Result r = prepare(object_name, mesh, positions, normals, mask); if (!r.ok) return r;
    const size_t count = mesh->geometry->get_vertex_count();
    std::vector<float> before(mask, mask + count);
    const std::string op = lower(operation);
    if (op == "clear") std::fill(mask, mask + count, 0.0f);
    else if (op == "fill") std::fill(mask, mask + count, 1.0f);
    else if (op == "invert") for (size_t i = 0; i < count; ++i) mask[i] = 1.0f - mask[i];
    else if (op == "noise") for (size_t i = 0; i < count; ++i) {
        const WeldKey key = weldKey(positions[i]);
        uint64_t h = static_cast<uint64_t>(key.x) * 73856093ull ^ static_cast<uint64_t>(key.y) * 19349663ull ^
                     static_cast<uint64_t>(key.z) * 83492791ull ^ seed;
        h ^= h >> 13u; h *= 1274126177ull; h ^= h >> 16u;
        mask[i] = static_cast<float>(h & 0xffffu) / 65535.0f;
    } else return Result::fail("unknown sculpt mask operation: " + operation);
    std::vector<float> after(mask, mask + count);
    if (undo && g_history) g_history->record(std::make_unique<SculptMaskCommand>(object_name, before, after));
    ProjectManager::getInstance().markModified();
    g_ctx->renderer.resetCPUAccumulation();
    return Result::success();
}

} // namespace rtapi
