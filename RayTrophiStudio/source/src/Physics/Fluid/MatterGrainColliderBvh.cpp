#include "Fluid/MatterGrainColliderBvh.h"
#include "Fluid/MatterGrain.h"
#include "SurfaceMeshCache.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <functional>
#include <numeric>
#include <map>

namespace RayTrophiSim::Fluid {
namespace {
float component(const Vec3& v, int axis) {
    return axis == 0 ? v.x : axis == 1 ? v.y : v.z;
}
bool finite(const Vec3& v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

std::vector<uint32_t> surfacePatches(const std::vector<SurfaceMeshTriangle>& triangles) {
    std::vector<uint32_t> parents(triangles.size());
    std::vector<std::vector<uint32_t>> members(triangles.size());
    std::iota(parents.begin(), parents.end(), 0u);
    for (uint32_t face = 0; face < triangles.size(); ++face) {
        members[face].push_back(face);
    }
    auto root = [&](uint32_t index) {
        while (parents[index] != index) {
            parents[index] = parents[parents[index]];
            index = parents[index];
        }
        return index;
    };
    std::map<std::array<uint32_t, 3>, uint32_t> vertices;
    std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> edges;
    for (uint32_t face = 0; face < triangles.size(); ++face) {
        const auto& t = triangles[face];
        uint32_t ids[3];
        const Vec3 points[] = {t.p0, t.p1, t.p2};
        for (int corner = 0; corner < 3; ++corner) {
            std::array<uint32_t, 3> key{};
            for (int axis = 0; axis < 3; ++axis) {
                // Normalize signed zero so coincident flat vertices join.
                const float value = component(points[corner], axis) == 0.0f
                    ? 0.0f : component(points[corner], axis);
                std::memcpy(&key[axis], &value, sizeof(float));
            }
            const auto inserted = vertices.emplace(key, static_cast<uint32_t>(vertices.size()));
            ids[corner] = inserted.first->second;
        }
        Vec3 normal = (t.p1 - t.p0).cross(t.p2 - t.p0);
        normal = normal / normal.length();
        for (int edge = 0; edge < 3; ++edge) {
            const auto a = ids[edge], b = ids[(edge + 1) % 3];
            auto& adjacent = edges[{std::min(a, b), std::max(a, b)}];
            for (const auto other : adjacent) {
                const auto& o = triangles[other];
                Vec3 on = (o.p1 - o.p0).cross(o.p2 - o.p0);
                on = on / on.length();
                const float alignment = normal.x * on.x + normal.y * on.y + normal.z * on.z;
                if (std::abs(alignment) >= .99999f) {
                    const auto ra = root(face), rb = root(other);
                    if (ra == rb) {
                        continue;
                    }
                    // Union by size: the smaller group joins the larger and is
                    // the one tested. Joining by index re-tested a whole flat
                    // ground plane on every merge, O(faces^2) once colliders
                    // were no longer capped at 4096 faces.
                    const bool ra_larger = members[ra].size() > members[rb].size() ||
                        (members[ra].size() == members[rb].size() && ra < rb);
                    const auto low = ra_larger ? ra : rb, high = ra_larger ? rb : ra;
                    const auto& seed = triangles[low];
                    Vec3 base = (seed.p1 - seed.p0).cross(seed.p2 - seed.p0);
                    base = base / base.length();
                    bool planar = true;
                    // Test the entire joining group against the canonical plane;
                    // pairwise normal similarity alone chains curved surfaces.
                    for (const auto member : members[high]) {
                        const auto& joined = triangles[member];
                        for (const auto& point : {joined.p0, joined.p1, joined.p2}) {
                            const auto delta = point - seed.p0;
                            const auto distance = base.x * delta.x + base.y * delta.y +
                                base.z * delta.z;
                            if (std::abs(distance) > 1e-5f) {
                                planar = false;
                                break;
                            }
                        }
                        if (!planar) {
                            break;
                        }
                    }
                    if (planar) {
                        parents[high] = low;
                        members[low].insert(members[low].end(), members[high].begin(),
                            members[high].end());
                        members[high].clear();
                    }
                }
            }
            adjacent.push_back(face);
        }
    }
    for (uint32_t face = 0; face < parents.size(); ++face) {
        parents[face] = root(face);
    }
    return parents;
}

// ★ Which faces bound a SOLID (closed, consistently wound) shell, and which way
// is out. The grain contact is otherwise two-sided: n = (p - q) / d flips as a
// crushed grain's centre crosses the face, the grain is pulled into the body
// and shot out of it (2026-11-07: keyframed sphere driven into a sand bed,
// 12-24 m/s "jet", collider overlap 0.99999). A shell is closed when every
// position-welded edge is used by exactly two of its faces in opposite
// directions; its winding is outward when the signed volume is positive.
// Open sheets (ground planes, thin walls) stay two-sided.
std::vector<uint32_t> closedShellFlags(const std::vector<SurfaceMeshTriangle>& triangles) {
    const std::size_t count = triangles.size();
    std::vector<uint32_t> flags(count, 0u);
    if (count < 4) return flags;
    // ★ Weld with a TOLERANCE, not by float bits: a UV sphere's seam vertices
    // come from sin/cos(0) and sin/cos(2 pi) and differ by a few ULP, so an
    // exact weld left every seam edge single-used and the closed sphere read
    // as open (closed_collider_faces = 0, 2026-11-07 live). Tolerance 1e-5 of
    // the collider extent; neighbouring cells are searched so a vertex pair
    // straddling a cell boundary still joins.
    Vec3 low = triangles[0].p0, high = triangles[0].p0;
    for (const auto& t : triangles) {
        low = Vec3::min(low, Vec3::min(t.p0, Vec3::min(t.p1, t.p2)));
        high = Vec3::max(high, Vec3::max(t.p0, Vec3::max(t.p1, t.p2)));
    }
    const double tolerance = std::max(1e-7, 1e-5 * double((high - low).length()));
    std::map<std::array<int64_t, 3>, std::vector<uint32_t>> cells;
    std::vector<Vec3> welded;
    auto weld = [&](const Vec3& p) -> uint32_t {
        std::array<int64_t, 3> key{};
        for (int axis = 0; axis < 3; ++axis) {
            key[axis] = static_cast<int64_t>(std::floor(double(component(p, axis)) / tolerance));
        }
        for (int dz = -1; dz <= 1; ++dz) for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
            const auto found = cells.find({key[0] + dx, key[1] + dy, key[2] + dz});
            if (found == cells.end()) continue;
            for (const uint32_t id : found->second) {
                if (double((welded[id] - p).length()) <= tolerance) return id;
            }
        }
        const auto id = static_cast<uint32_t>(welded.size());
        welded.push_back(p);
        cells[key].push_back(id);
        return id;
    };
    std::vector<std::array<uint32_t, 3>> ids(count);
    for (std::size_t face = 0; face < count; ++face) {
        const auto& t = triangles[face];
        ids[face] = {weld(t.p0), weld(t.p1), weld(t.p2)};
    }
    // Directed edge uses: +1 for a->b with a < b, -1 for the reverse.
    struct EdgeUse { uint32_t face; int direction; };
    std::map<std::pair<uint32_t, uint32_t>, std::vector<EdgeUse>> edges;
    for (uint32_t face = 0; face < count; ++face) {
        // A face the weld collapsed (two corners joined) has no area: its two
        // remaining edges would count the same edge twice and break the shell.
        if (ids[face][0] == ids[face][1] || ids[face][1] == ids[face][2] ||
            ids[face][2] == ids[face][0]) continue;
        for (int edge = 0; edge < 3; ++edge) {
            const uint32_t a = ids[face][edge], b = ids[face][(edge + 1) % 3];
            edges[{std::min(a, b), std::max(a, b)}].push_back({face, a < b ? 1 : -1});
        }
    }
    std::vector<uint32_t> parent(count);
    std::iota(parent.begin(), parent.end(), 0u);
    const std::function<uint32_t(uint32_t)> root = [&](uint32_t i) {
        while (parent[i] != i) { parent[i] = parent[parent[i]]; i = parent[i]; }
        return i;
    };
    for (const auto& [key, uses] : edges) {
        for (std::size_t u = 1; u < uses.size(); ++u) {
            const uint32_t ra = root(uses[0].face), rb = root(uses[u].face);
            if (ra != rb) parent[rb] = ra;
        }
    }
    std::vector<uint8_t> broken(count, 0u);
    for (const auto& [key, uses] : edges) {
        const bool manifold = uses.size() == 2 && uses[0].direction != uses[1].direction;
        if (!manifold) {
            for (const auto& use : uses) broken[root(use.face)] = 1u;
        }
    }
    std::vector<double> volume(count, 0.0);
    for (uint32_t face = 0; face < count; ++face) {
        const auto& t = triangles[face];
        volume[root(face)] += double(t.p0.x) * (double(t.p1.y) * t.p2.z - double(t.p1.z) * t.p2.y) -
            double(t.p0.y) * (double(t.p1.x) * t.p2.z - double(t.p1.z) * t.p2.x) +
            double(t.p0.z) * (double(t.p1.x) * t.p2.y - double(t.p1.y) * t.p2.x);
    }
    for (uint32_t face = 0; face < count; ++face) {
        const uint32_t r = root(face);
        if (broken[r] || volume[r] == 0.0 || !std::isfinite(volume[r])) continue;
        flags[face] = kMatterGrainPatchClosed | (volume[r] < 0.0 ? kMatterGrainPatchInward : 0u);
    }
    return flags;
}
} // namespace

uint64_t matterGrainColliderFingerprint(const std::vector<SurfaceMeshTriangle>& triangles,
                                        const std::vector<Vec3>* velocities) {
    uint64_t hash = 14695981039346656037ull;
    const auto mix = [&](const Vec3& p) {
        for (int axis = 0; axis < 3; ++axis) {
            const float value = component(p, axis);
            uint32_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            for (int byte = 0; byte < 4; ++byte) {
                hash = (hash ^ ((bits >> (8 * byte)) & 255u)) * 1099511628211ull;
            }
        }
    };
    for (const auto& t : triangles) {
        for (const auto& p : {t.p0, t.p1, t.p2}) {
            mix(p);
        }
    }
    if (velocities) {
        for (const auto& v : *velocities) {
            mix(v);
        }
    }
    return hash;
}

bool buildMatterGrainColliderBvh(const std::vector<SurfaceMeshTriangle>& triangles,
                                MatterGrainColliderBvh& result, std::string& error,
                                const std::vector<Vec3>* velocities, float sweep_seconds) {
    if (triangles.size() > kMatterGrainMaxColliderFaces) {
        error = "grain collider BVH exceeds the 30-bit contact patch id";
        return false;
    }
    if (velocities && velocities->size() != 3 * triangles.size()) {
        error = "grain collider BVH: one velocity per triangle vertex expected";
        return false;
    }
    MatterGrainColliderBvh candidate;
    if (triangles.empty()) {
        result = std::move(candidate);
        error.clear();
        return true;
    }
    std::vector<Vec3> lower, upper, centers;
    for (const auto& t : triangles) {
        const auto cross = (t.p1 - t.p0).cross(t.p2 - t.p0);
        const auto area = cross.length();
        if (!finite(t.p0) || !finite(t.p1) || !finite(t.p2) ||
            !finite(cross) || !std::isfinite(area) || area < 1e-10f) {
            error = "grain collider BVH received nonfinite/degenerate geometry";
            return false;
        }
        Vec3 lo = Vec3::min(t.p0, Vec3::min(t.p1, t.p2));
        Vec3 hi = Vec3::max(t.p0, Vec3::max(t.p1, t.p2));
        if (velocities) {
            // The triangle sweeps from end - v * sweep to end during the frame.
            const std::size_t face = lower.size();
            const Vec3 points[] = {t.p0, t.p1, t.p2};
            for (int corner = 0; corner < 3; ++corner) {
                const Vec3& v = (*velocities)[3 * face + corner];
                if (!finite(v)) {
                    error = "grain collider BVH received a nonfinite vertex velocity";
                    return false;
                }
                const Vec3 start = points[corner] - v * sweep_seconds;
                lo = Vec3::min(lo, start);
                hi = Vec3::max(hi, start);
            }
        }
        lower.push_back(lo);
        upper.push_back(hi);
        centers.push_back(lo * .5f + hi * .5f);
    }
    std::vector<uint32_t> order(triangles.size());
    std::iota(order.begin(), order.end(), 0u);
    std::function<uint32_t(std::size_t, std::size_t)> build =
        [&](std::size_t start, std::size_t end) -> uint32_t {
        const auto index = static_cast<uint32_t>(candidate.nodes.size());
        candidate.nodes.emplace_back();
        Vec3 lo = lower[order[start]], hi = upper[order[start]];
        for (std::size_t i = start + 1; i < end; ++i) {
            lo = Vec3::min(lo, lower[order[i]]);
            hi = Vec3::max(hi, upper[order[i]]);
        }
        MatterGrainBvhNode node;
        for (int axis = 0; axis < 3; ++axis) {
            node.low[axis] = component(lo, axis);
            node.high[axis] = component(hi, axis);
        }
        if (end - start <= 4) {
            node.first = 0x80000000u | static_cast<uint32_t>(start);
            node.second = static_cast<uint32_t>(end - start);
        } else {
            const Vec3 size = hi - lo;
            const int axis = size.y > size.x ? (size.z > size.y ? 2 : 1) :
                (size.z > size.x ? 2 : 0);
            const auto middle = start + (end - start) / 2;
            std::nth_element(order.begin() + start, order.begin() + middle, order.begin() + end,
                [&](uint32_t a, uint32_t b) {
                    const auto ca = component(centers[a], axis);
                    const auto cb = component(centers[b], axis);
                    return ca == cb ? a < b : ca < cb;
                });
            node.first = build(start, middle);
            node.second = build(middle, end);
        }
        candidate.nodes[index] = node;
        return index;
    };
    build(0, triangles.size());
    const auto patches = surfacePatches(triangles);
    const auto shells = closedShellFlags(triangles);
    for (const auto face : order) {
        const auto& t = triangles[face];
        candidate.vertices.insert(candidate.vertices.end(), {t.p0, t.p1, t.p2});
        candidate.source_faces.push_back(face);
        // Patch id in bits 0-29 (kMatterGrainMaxColliderFaces), shell flags above.
        candidate.surface_patches.push_back(patches[face] | shells[face]);
        if (shells[face] & kMatterGrainPatchClosed) ++candidate.closed_faces;
        if (velocities) {
            candidate.velocities.insert(candidate.velocities.end(), {(*velocities)[3 * face],
                (*velocities)[3 * face + 1], (*velocities)[3 * face + 2]});
        }
    }
    result = std::move(candidate);
    error.clear();
    return true;
}

} // namespace RayTrophiSim::Fluid
