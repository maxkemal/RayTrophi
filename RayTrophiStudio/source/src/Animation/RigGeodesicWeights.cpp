#include "Animation/RigGeodesicWeights.h"
#include "Animation/RigPosePreview.h"
#include "Animation/SkinWeightContract.h"
#include "DNA/GeometryDetail.h"
#include "TriangleMesh.h"
#include "scene_data.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace RigAuthoring {
namespace {
constexpr size_t kMaxVoxels = 16000000;
constexpr size_t kMaxVertices = 2000000;
// Distance growth stops at this fraction of the rig height; a bone farther than that through
// the body would never make a vertex's strongest four anyway.
constexpr float kReach = .5f;
constexpr int kInfluences = 4;
constexpr float kInfinity = std::numeric_limits<float>::infinity();

enum VoxelState : uint8_t { Empty = 0, Surface = 1, Exterior = 2, Interior = 3 };

struct Grid {
    int nx = 0, ny = 0, nz = 0;
    Vec3 origin;
    float h = 1;
    std::vector<uint8_t> state;

    size_t index(int x, int y, int z) const {
        return (size_t(z) * size_t(ny) + size_t(y)) * size_t(nx) + size_t(x);
    }
    bool solid(size_t i) const { return state[i] == Surface || state[i] == Interior; }
    void cell(const Vec3 &p, int &x, int &y, int &z) const {
        x = std::clamp(int(std::floor((p.x - origin.x) / h)), 0, nx - 1);
        y = std::clamp(int(std::floor((p.y - origin.y) / h)), 0, ny - 1);
        z = std::clamp(int(std::floor((p.z - origin.z) / h)), 0, nz - 1);
    }
    size_t cellIndex(const Vec3 &p) const {
        int x, y, z;
        cell(p, x, y, z);
        return index(x, y, z);
    }
    Vec3 center(size_t i) const {
        const size_t x = i % size_t(nx), y = (i / size_t(nx)) % size_t(ny),
                     z = i / (size_t(nx) * size_t(ny));
        return origin + Vec3((float(x) + .5f) * h, (float(y) + .5f) * h, (float(z) + .5f) * h);
    }
};

struct Bone {
    int skin = -1;
    std::string name;
    std::vector<std::pair<Vec3, Vec3>> segments;
};

struct Part {
    std::shared_ptr<TriangleMesh> mesh;
    const DNA::GeometryDetail *geometry = nullptr;
    const Vec3 *positions = nullptr;
    size_t vertices = 0;
};

Vec3 origin(const Matrix4x4 &m) {
    return Vec3(m.m[0][3], m.m[1][3], m.m[2][3]);
}

float segmentDistance(const Vec3 &p, const Vec3 &a, const Vec3 &b) {
    const auto ab = b - a;
    const float length = ab.length_squared();
    const float t = length > 0 ? std::clamp(Vec3::dot(p - a, ab) / length, 0.f, 1.f) : 0.f;
    return (p - (a + ab * t)).length();
}

void rasterizeTriangle(Grid &grid, const Vec3 &a, const Vec3 &b, const Vec3 &c) {
    const float longest = (std::max)({(b - a).length(), (c - a).length(), (c - b).length()});
    const int steps = std::clamp(int(std::ceil(longest / (.5f * grid.h))), 1, 512);
    for (int i = 0; i <= steps; ++i) {
        for (int j = 0; j <= steps - i; ++j) {
            const float u = float(i) / steps, v = float(j) / steps;
            grid.state[grid.cellIndex(a + (b - a) * u + (c - a) * v)] = Surface;
        }
    }
}

// Everything reachable from the padded border without crossing the surface is outside; the
// remaining empty voxels are interior. An open mesh leaks and keeps only its shell, which still
// gives through-the-surface distances.
size_t floodExterior(Grid &grid) {
    std::vector<size_t> stack;
    auto seed = [&](int x, int y, int z) {
        const size_t i = grid.index(x, y, z);
        if (grid.state[i] == Empty) {
            grid.state[i] = Exterior;
            stack.push_back(i);
        }
    };
    for (int z = 0; z < grid.nz; ++z)
        for (int y = 0; y < grid.ny; ++y)
            for (int x = 0; x < grid.nx; ++x)
                if (x == 0 || y == 0 || z == 0 || x == grid.nx - 1 || y == grid.ny - 1 ||
                    z == grid.nz - 1)
                    seed(x, y, z);
    while (!stack.empty()) {
        const size_t i = stack.back();
        stack.pop_back();
        const int x = int(i % size_t(grid.nx)), y = int((i / size_t(grid.nx)) % size_t(grid.ny)),
                  z = int(i / (size_t(grid.nx) * size_t(grid.ny)));
        if (x > 0) seed(x - 1, y, z);
        if (x + 1 < grid.nx) seed(x + 1, y, z);
        if (y > 0) seed(x, y - 1, z);
        if (y + 1 < grid.ny) seed(x, y + 1, z);
        if (z > 0) seed(x, y, z - 1);
        if (z + 1 < grid.nz) seed(x, y, z + 1);
    }
    size_t interior = 0;
    for (auto &value : grid.state)
        if (value == Empty) {
            value = Interior;
            ++interior;
        }
    return interior;
}

struct GeodesicResult {
    std::vector<std::vector<std::pair<int, float>>> rows; // Per global vertex.
    size_t surfaceVoxels = 0, interiorVoxels = 0, fallback = 0, unseeded = 0;
    Grid grid;
};

bool computeWeights(const std::vector<Part> &parts, const std::vector<Bone> &bones, float height,
                    const GeodesicWeightSettings &settings, GeodesicResult &out, std::string &error) {
    const float inf = kInfinity;
    Vec3 lo(inf, inf, inf), hi(-inf, -inf, -inf);
    auto grow = [&](const Vec3 &p) {
        lo = Vec3((std::min)(lo.x, p.x), (std::min)(lo.y, p.y), (std::min)(lo.z, p.z));
        hi = Vec3((std::max)(hi.x, p.x), (std::max)(hi.y, p.y), (std::max)(hi.z, p.z));
    };
    size_t vertexTotal = 0;
    for (const auto &part : parts) {
        for (size_t v = 0; v < part.vertices; ++v)
            grow(part.positions[v]);
        vertexTotal += part.vertices;
    }
    for (const auto &bone : bones)
        for (const auto &segment : bone.segments) {
            grow(segment.first);
            grow(segment.second);
        }
    const auto extent = hi - lo;
    const float largest = (std::max)({extent.x, extent.y, extent.z});
    if (!std::isfinite(largest) || largest <= 1e-6f) {
        error = "rig_geodesic_invalid_bounds";
        return false;
    }
    auto &grid = out.grid;
    grid.h = largest / float(settings.resolution);
    grid.origin = lo - Vec3(2 * grid.h, 2 * grid.h, 2 * grid.h);
    grid.nx = int(std::ceil(extent.x / grid.h)) + 5;
    grid.ny = int(std::ceil(extent.y / grid.h)) + 5;
    grid.nz = int(std::ceil(extent.z / grid.h)) + 5;
    const size_t voxels = size_t(grid.nx) * size_t(grid.ny) * size_t(grid.nz);
    if (voxels > kMaxVoxels) {
        error = "rig_geodesic_limit";
        return false;
    }
    grid.state.assign(voxels, Empty);
    for (const auto &part : parts) {
        const auto &indices = part.geometry->indices;
        for (size_t f = 0; f + 2 < indices.size(); f += 3) {
            const auto a = indices[f], b = indices[f + 1], c = indices[f + 2];
            if (a >= part.vertices || b >= part.vertices || c >= part.vertices) {
                error = "rig_bind_invalid_indices";
                return false;
            }
            rasterizeTriangle(grid, part.positions[a], part.positions[b], part.positions[c]);
        }
        for (size_t v = 0; v < part.vertices; ++v)
            grid.state[grid.cellIndex(part.positions[v])] = Surface;
    }
    for (const auto value : grid.state)
        out.surfaceVoxels += value == Surface ? 1 : 0;
    out.interiorVoxels = floodExterior(grid);
    std::vector<uint32_t> solids;
    for (size_t i = 0; i < voxels; ++i)
        if (grid.solid(i))
            solids.push_back(static_cast<uint32_t>(i));

    // Every vertex keeps its four nearest bones by distance through the solid.
    std::vector<size_t> vertexCell(vertexTotal);
    std::vector<int> bestBone(vertexTotal * kInfluences, -1);
    std::vector<float> bestDistance(vertexTotal * kInfluences, inf);
    {
        size_t offset = 0;
        for (const auto &part : parts) {
            for (size_t v = 0; v < part.vertices; ++v)
                vertexCell[offset + v] = grid.cellIndex(part.positions[v]);
            offset += part.vertices;
        }
    }
    std::unordered_map<size_t, std::vector<size_t>> cellVertices;
    for (size_t v = 0; v < vertexTotal; ++v)
        cellVertices[vertexCell[v]].push_back(v);

    struct Step {
        int dx, dy, dz;
        float cost;
    };
    std::vector<Step> steps;
    for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx)
                if (dx || dy || dz)
                    steps.push_back({dx, dy, dz,
                                     grid.h * std::sqrt(float(dx * dx + dy * dy + dz * dz))});

    const float reach = kReach * height;
    std::vector<float> distance(voxels, inf);
    std::vector<uint32_t> touched;
    using Entry = std::pair<float, uint32_t>;
    for (size_t b = 0; b < bones.size(); ++b) {
        std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> queue;
        auto relax = [&](size_t cell, float d) {
            if (d < distance[cell]) {
                if (distance[cell] == inf)
                    touched.push_back(static_cast<uint32_t>(cell));
                distance[cell] = d;
                queue.push({d, static_cast<uint32_t>(cell)});
            }
        };
        for (const auto &segment : bones[b].segments) {
            const float length = (segment.second - segment.first).length();
            const int samples = (std::max)(1, int(std::ceil(length / (.5f * grid.h))));
            for (int s = 0; s <= samples; ++s) {
                const auto p = segment.first + (segment.second - segment.first) * (float(s) / samples);
                const size_t cell = grid.cellIndex(p);
                if (grid.solid(cell))
                    relax(cell, 0.f);
            }
        }
        if (queue.empty()) {
            // The bone lies outside the mesh: start from the solid voxel nearest to it.
            ++out.unseeded;
            float bestValue = inf;
            uint32_t bestCell = 0;
            for (const auto cell : solids) {
                const auto c = grid.center(cell);
                for (const auto &segment : bones[b].segments) {
                    const float d = segmentDistance(c, segment.first, segment.second);
                    if (d < bestValue) {
                        bestValue = d;
                        bestCell = cell;
                    }
                }
            }
            if (bestValue < inf)
                relax(bestCell, bestValue);
        }
        while (!queue.empty()) {
            const auto [d, cell] = queue.top();
            queue.pop();
            if (d > distance[cell] || d > reach)
                continue;
            const int x = int(cell % uint32_t(grid.nx)),
                      y = int((cell / uint32_t(grid.nx)) % uint32_t(grid.ny)),
                      z = int(cell / (uint32_t(grid.nx) * uint32_t(grid.ny)));
            for (const auto &step : steps) {
                const int nx = x + step.dx, ny = y + step.dy, nz = z + step.dz;
                if (nx < 0 || ny < 0 || nz < 0 || nx >= grid.nx || ny >= grid.ny || nz >= grid.nz)
                    continue;
                const size_t next = grid.index(nx, ny, nz);
                if (grid.solid(next))
                    relax(next, d + step.cost);
            }
        }
        for (const auto cell : touched) {
            const float d = distance[cell];
            const auto found = cellVertices.find(cell);
            if (found != cellVertices.end() && d <= reach) {
                for (const size_t v : found->second) {
                    float *dist = &bestDistance[v * kInfluences];
                    int *bone = &bestBone[v * kInfluences];
                    int worst = 0;
                    for (int k = 1; k < kInfluences; ++k)
                        if (dist[k] > dist[worst])
                            worst = k;
                    if (d < dist[worst]) {
                        dist[worst] = d;
                        bone[worst] = static_cast<int>(b);
                    }
                }
            }
            distance[cell] = inf;
        }
        touched.clear();
    }

    out.rows.assign(vertexTotal, {});
    size_t offset = 0;
    const float minimumDistance = .5f * grid.h;
    for (const auto &part : parts) {
        for (size_t v = 0; v < part.vertices; ++v) {
            const size_t global = offset + v;
            VertexInfluences row;
            for (int k = 0; k < kInfluences; ++k) {
                const int bone = bestBone[global * kInfluences + k];
                if (bone < 0)
                    continue;
                const float d = (std::max)(bestDistance[global * kInfluences + k], minimumDistance);
                row.emplace_back(bones[size_t(bone)].skin, std::pow(d, -settings.falloff));
            }
            if (row.empty()) {
                // Unreached through the solid (a detached part): nearest bone in straight line.
                ++out.fallback;
                float nearest = inf;
                int skin = bones.front().skin;
                for (const auto &bone : bones)
                    for (const auto &segment : bone.segments) {
                        const float d = segmentDistance(part.positions[v], segment.first,
                                                        segment.second);
                        if (d < nearest) {
                            nearest = d;
                            skin = bone.skin;
                        }
                    }
                row.emplace_back(skin, 1.f);
            }
            canonicalizeInfluences(row);
            out.rows[global] = std::move(row);
        }
        offset += part.vertices;
    }
    return true;
}

bool build(const SceneData &scene, const std::string &character,
           const GeodesicWeightSettings &settings, bool checkRevision, uint64_t expectedRevision,
           EnvelopeWeightState *state, nlohmann::json &report, std::string &error) {
    report = nullptr;
    error.clear();
    if (!validGeodesicWeightSettings(settings)) {
        error = "rig_geodesic_invalid_settings";
        return false;
    }
    const SceneData::ImportedModelContext *model = nullptr;
    for (const auto &candidate : scene.importedModelContexts) {
        if (candidate.importName == character) {
            if (model) {
                error = "ambiguous_character";
                return false;
            }
            model = &candidate;
        }
    }
    if (!model) {
        error = "unknown_character";
        return false;
    }
    if (checkRevision && model->rigRevision != expectedRevision) {
        error = "rig_edit_stale_revision";
        return false;
    }
    if (model->rigBoundMeshes.empty()) {
        error = "rig_envelope_requires_authored_binding";
        return false;
    }
    if (model->rigRevision == (std::numeric_limits<uint64_t>::max)()) {
        error = "rig_revision_overflow";
        return false;
    }
    std::vector<PreviewJoint> joints;
    if (!sampleRigPose(model->nodeHierarchy, nullptr, 0, joints, error))
        return false;
    if (joints.empty()) {
        error = "rig_envelope_invalid_height";
        return false;
    }
    float minimum = origin(joints.front().world).y, maximum = minimum;
    for (const auto &joint : joints) {
        minimum = (std::min)(minimum, origin(joint.world).y);
        maximum = (std::max)(maximum, origin(joint.world).y);
    }
    const float height = maximum - minimum;
    if (!std::isfinite(height) || height <= 1e-6f) {
        error = "rig_envelope_invalid_height";
        return false;
    }
    // Same segments as the envelope weights: each joint's parent owns the segment to it, and the
    // root role does not deform.
    std::unordered_set<std::string> nonDeforming;
    for (const auto &role : model->rigAnatomy.roles)
        if (role.role == "root")
            nonDeforming.insert(role.bone);
    std::vector<Bone> bones;
    std::unordered_map<std::string, size_t> boneSlot;
    for (size_t i = 0; i < joints.size(); ++i) {
        const int parent = model->nodeHierarchy.nodes[i].parent;
        if (parent < 0 || nonDeforming.count(joints[size_t(parent)].name))
            continue;
        const auto &name = joints[size_t(parent)].name;
        const auto index = scene.boneData.boneNameToIndex.find(name);
        if (index == scene.boneData.boneNameToIndex.end() ||
            index->second > static_cast<unsigned int>((std::numeric_limits<int>::max)())) {
            error = "rig_bind_invalid_indices";
            return false;
        }
        auto slot = boneSlot.find(name);
        if (slot == boneSlot.end()) {
            slot = boneSlot.emplace(name, bones.size()).first;
            bones.push_back({static_cast<int>(index->second), name, {}});
        }
        bones[slot->second].segments.emplace_back(origin(joints[size_t(parent)].world),
                                                  origin(joints[i].world));
    }
    if (bones.empty()) {
        error = "rig_bind_requires_segments";
        return false;
    }
    std::vector<Part> parts;
    size_t vertexCount = 0;
    for (const auto &meshName : model->rigBoundMeshes) {
        Part part;
        for (const auto &object : scene.world.objects) {
            auto candidate = std::dynamic_pointer_cast<TriangleMesh>(object);
            if (candidate && candidate->nodeName == meshName) {
                if (part.mesh) {
                    error = "ambiguous_mesh_name";
                    return false;
                }
                part.mesh = std::move(candidate);
            }
        }
        if (!part.mesh || !part.mesh->geometry) {
            error = "rig_envelope_missing_bound_mesh";
            return false;
        }
        part.geometry = part.mesh->geometry.get();
        part.vertices = part.mesh->num_vertices();
        part.positions = part.geometry->get_positions_orig();
        if (!part.positions || part.geometry->get_positions_orig_count() < part.vertices) {
            error = "rig_envelope_missing_bind_positions";
            return false;
        }
        vertexCount += part.vertices;
        if (vertexCount > kMaxVertices) {
            error = "rig_geodesic_limit";
            return false;
        }
        parts.push_back(std::move(part));
    }

    GeodesicResult result;
    if (!computeWeights(parts, bones, height, settings, result, error))
        return false;

    std::unordered_map<int, std::pair<size_t, size_t>> counts;
    for (const auto &row : result.rows)
        for (const auto &influence : row) {
            auto &count = counts[influence.first];
            ++count.first;
            count.second += influence.second > .05f ? 1 : 0;
        }
    auto influenceCounts = nlohmann::json::array();
    for (const auto &bone : bones) {
        const auto found = counts.find(bone.skin);
        influenceCounts.push_back(
            {{"bone", bone.name},
             {"nonzero_vertices", found == counts.end() ? 0 : found->second.first},
             {"significant_vertices", found == counts.end() ? 0 : found->second.second}});
    }
    report = {{"character", character},
              {"rig_revision", model->rigRevision},
              {"algorithm", "geodesic_voxel_v1"},
              {"resolution", settings.resolution},
              {"falloff", settings.falloff},
              {"voxel_size", result.grid.h},
              {"grid", {result.grid.nx, result.grid.ny, result.grid.nz}},
              {"surface_voxels", result.surfaceVoxels},
              {"interior_voxels", result.interiorVoxels},
              {"closed_volume", result.interiorVoxels > 0},
              {"vertex_count", vertexCount},
              {"part_count", parts.size()},
              {"bone_count", bones.size()},
              {"fallback_vertices", result.fallback},
              {"bones_outside_mesh", result.unseeded},
              {"influence_counts", std::move(influenceCounts)}};
    if (!state)
        return true;

    *state = {};
    size_t offset = 0;
    for (const auto &part : parts) {
        auto geometry = std::make_shared<DNA::GeometryDetail>(*part.geometry);
        geometry->skin_weights.assign(part.vertices, {});
        for (size_t v = 0; v < part.vertices; ++v)
            geometry->skin_weights[v] = std::move(result.rows[offset + v]);
        geometry->last_skinned_pose_hash = 0;
        offset += part.vertices;
        state->parts.push_back({part.mesh, std::move(geometry)});
    }
    state->character = character;
    state->algorithm = "geodesic_voxel_v1";
    state->settings = {model->rigEnvelopeTorsoRadius, model->rigEnvelopeLimbRadius,
                       model->rigEnvelopeExtremityRadius, model->rigEnvelopeFalloff};
    state->profiles = model->rigEnvelopeProfiles;
    state->revision = model->rigRevision + 1;
    return true;
}
} // namespace

bool validGeodesicWeightSettings(const GeodesicWeightSettings &settings) {
    return settings.resolution >= 32 && settings.resolution <= 256 &&
           std::isfinite(settings.falloff) && settings.falloff >= .5f && settings.falloff <= 8.f;
}

bool previewGeodesicWeights(const SceneData &scene, const std::string &character,
                            const GeodesicWeightSettings &settings, nlohmann::json &report,
                            std::string &error) {
    return build(scene, character, settings, false, 0, nullptr, report, error);
}

bool stageGeodesicWeights(const SceneData &scene, const std::string &character,
                          const GeodesicWeightSettings &settings, uint64_t expectedRevision,
                          EnvelopeWeightState &state, nlohmann::json &report, std::string &error) {
    return build(scene, character, settings, true, expectedRevision, &state, report, error);
}
} // namespace RigAuthoring
