#include "scene_ui.h"

#include "Vec3SIMD.h"
#include <omp.h>

#include "Backend/OptixBackend.h"
#include "Backend/IViewportBackend.h"
#include "Backend/VulkanBackend.h"
#include "Viewport/ViewportSceneSync.h"
#include "Triangle.h"
#include "MeshProfileTimer.h"
#include "HittableInstance.h"
#include "ImGuizmo.h"
#include "imgui.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <execution>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <SceneSelection.h>
#include "Paint/IPaintSurfaceAdapter.h"

extern std::unique_ptr<Backend::IViewportBackend> g_viewport_backend;
extern std::unique_ptr<Backend::IBackend> g_backend;

// Forward declarations from scene_ui_modifiers.cpp
// Slice 4: live device-resident GPU Catmull-Clark refit. Returns true when the node is
// driven by a device-resident CC mesh (cage excluded from RT) — the caller must then skip
// the host BLAS update and NOT raise a full rebuild; the edit was routed to a GPU stencil
// re-apply + BLAS refit instead. Returns false for non-CC / OptiX / CPU / Solid (host path).
bool driveLiveDeviceResidentCC(const std::string& nodeName,
                               const std::vector<std::shared_ptr<Triangle>>& cage,
                               const MeshModifiers::ModifierStack& stack);
float uiSampleBrushMask(const Paint::BrushSettings& brush, float nx, float ny);
float uiSampleBrushFootprintWeight(const Paint::BrushSettings& brush, float nx, float ny);
float uiBrushFootprintBoundScale(const Paint::BrushSettings& brush);
float uiBrushFootprintDistNorm(const Paint::BrushSettings& brush, float nx, float ny);

namespace {
// Grab-family (Grab/Elastic/SnakeHook) cursor-follow ratio. FIXED on purpose — the
// brush "Strength" slider no longer drives the grab, so a fast flick always produces
// the SAME controlled displacement (cursorDelta * weight * this). Decoupling from the
// slider keeps the per-sub-step triangle rotation bounded regardless of how hard the
// user cranked Strength, so the topology-safety guard stops rejecting boundary verts
// on fast strokes (the "fast movement drops/freezes vertices" artifact). 0.5 = the
// mesh trails the cursor at half speed: enough control, never a teleport.
constexpr float kSculptGrabFollowStrength = 0.5f;

Backend::IBackend* getMeshOverlayRenderBackend(UIContext& ctx) {
    if (g_backend) {
        return g_backend.get();
    }
    if (ctx.backend_ptr && dynamic_cast<Backend::IViewportBackend*>(ctx.backend_ptr) == nullptr) {
        return ctx.backend_ptr;
    }
    return nullptr;
}

Backend::IViewportBackend* getMeshOverlayViewportBackend(UIContext& ctx) {
    if (g_viewport_backend) {
        return g_viewport_backend.get();
    }
    return dynamic_cast<Backend::IViewportBackend*>(ctx.backend_ptr);
}

std::string getMeshOverlayNodeName(const std::shared_ptr<Hittable>& obj) {
    if (!obj) {
        return {};
    }
    if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
        return tri->getNodeName();
    }
    if (auto inst = std::dynamic_pointer_cast<HittableInstance>(obj)) {
        return inst->node_name;
    }
    return {};
}

struct QuantizedVertexKey {
    int x = 0;
    int y = 0;
    int z = 0;

    bool operator==(const QuantizedVertexKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct QuantizedVertexKeyHasher {
    std::size_t operator()(const QuantizedVertexKey& key) const {
        std::size_t hx = std::hash<int>{}(key.x);
        std::size_t hy = std::hash<int>{}(key.y);
        std::size_t hz = std::hash<int>{}(key.z);
        return hx ^ (hy << 1) ^ (hz << 2);
    }
};

struct EdgeKey {
    QuantizedVertexKey a;
    QuantizedVertexKey b;

    bool operator==(const EdgeKey& other) const {
        return a == other.a && b == other.b;
    }
};

struct EdgeKeyHasher {
    std::size_t operator()(const EdgeKey& key) const {
        QuantizedVertexKeyHasher hasher;
        return hasher(key.a) ^ (hasher(key.b) << 1);
    }
};

QuantizedVertexKey quantizeVertex(const Vec3& p) {
    constexpr float scale = 10000.0f;
    return QuantizedVertexKey{
        static_cast<int>(std::lround(p.x * scale)),
        static_cast<int>(std::lround(p.y * scale)),
        static_cast<int>(std::lround(p.z * scale))
    };
}

QuantizedVertexKey quantizeTopologyVertex(const Vec3& p) {
    constexpr float scale = 1000000.0f;
    return QuantizedVertexKey{
        static_cast<int>(std::lround(p.x * scale)),
        static_cast<int>(std::lround(p.y * scale)),
        static_cast<int>(std::lround(p.z * scale))
    };
}

bool lessVertexKey(const QuantizedVertexKey& lhs, const QuantizedVertexKey& rhs) {
    if (lhs.x != rhs.x) return lhs.x < rhs.x;
    if (lhs.y != rhs.y) return lhs.y < rhs.y;
    return lhs.z < rhs.z;
}

EdgeKey makeSortedEdgeKey(const Vec3& p0, const Vec3& p1) {
    QuantizedVertexKey a = quantizeVertex(p0);
    QuantizedVertexKey b = quantizeVertex(p1);
    if (lessVertexKey(b, a)) {
        std::swap(a, b);
    }
    return EdgeKey{a, b};
}

bool projectPointToScreen(const Camera& cam, const ImVec2& displaySize, const Vec3& point, ImVec2& out, bool isOrtho = false) {
    if (displaySize.x <= 1.0f || displaySize.y <= 1.0f) {
        return false;
    }

    const Vec3 camForward = (cam.lookat - cam.lookfrom).normalize();
    const Vec3 camRight = camForward.cross(cam.vup).normalize();
    const Vec3 camUp = camRight.cross(camForward).normalize();
    const Vec3 toPoint = point - cam.lookfrom;
    const float depth = toPoint.dot(camForward);
    if (!isOrtho && depth <= 0.01f) {
        return false;
    }

    // Use cam.aspect_ratio to match selection logic and Camera::get_ray's frustum exactly.
    const float aspect = cam.aspect_ratio;
    const float tanHalfFov = std::tan(cam.vfov * 3.14159265359f / 180.0f * 0.5f);
    if (std::fabs(aspect) <= 1e-6f || std::fabs(tanHalfFov) <= 1e-6f) {
        return false;
    }

    const float localX = toPoint.dot(camRight);
    const float localY = toPoint.dot(camUp);

    float halfH, halfW;
    if (isOrtho) {
        halfH = cam.ortho_height * 0.5f;
        halfW = halfH * aspect;
    } else {
        halfH = depth * tanHalfFov;
        halfW = halfH * aspect;
    }

    if (std::fabs(halfW) <= 1e-6f || std::fabs(halfH) <= 1e-6f) {
        return false;
    }

    out.x = ((localX / halfW) * 0.5f + 0.5f) * displaySize.x;
    out.y = (0.5f - (localY / halfH) * 0.5f) * displaySize.y;
    return true;
}

float distancePointToSegmentSq(const ImVec2& p, const ImVec2& a, const ImVec2& b) {
    const float abx = b.x - a.x;
    const float aby = b.y - a.y;
    const float apx = p.x - a.x;
    const float apy = p.y - a.y;
    const float abLenSq = abx * abx + aby * aby;
    if (abLenSq <= 1e-6f) {
        return apx * apx + apy * apy;
    }

    const float t = (std::max)(0.0f, (std::min)(1.0f, (apx * abx + apy * aby) / abLenSq));
    const float dx = p.x - (a.x + abx * t);
    const float dy = p.y - (a.y + aby * t);
    return dx * dx + dy * dy;
}

float signedArea2D(const ImVec2& a, const ImVec2& b, const ImVec2& c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

bool pointInTriangle2D(const ImVec2& p, const ImVec2& a, const ImVec2& b, const ImVec2& c) {
    const float d1 = signedArea2D(p, a, b);
    const float d2 = signedArea2D(p, b, c);
    const float d3 = signedArea2D(p, c, a);
    const bool hasNeg = (d1 < 0.0f) || (d2 < 0.0f) || (d3 < 0.0f);
    const bool hasPos = (d1 > 0.0f) || (d2 > 0.0f) || (d3 > 0.0f);
    return !(hasNeg && hasPos);
}

Matrix4x4 getEditableObjectTransform(const SceneUI::EditableMeshCache& cache) {
    return cache.source_object_transform;
}

bool isEditableVertexIdValid(const SceneUI::EditableMeshCache& cache, int vertexId) {
    return vertexId >= 0 && vertexId < static_cast<int>(cache.vertices.size());
}

// Resolve a Triangle* to its face index in the active cache in O(1) via the transient
// Triangle::editable_index, validated against source_triangles so a stale index (left over
// from a previous cache / another object) can never alias the wrong face. Replaces the old
// triangle_vertex_ids / triangle_to_mesh_index hash lookups: faces[idx] gives {v0,v1,v2} and
// face_to_mesh_index[idx] gives the source/GPU buffer slot. Returns -1 if not in this cache.
inline int editableFaceIndexOf(const SceneUI::EditableMeshCache& cache, const Triangle* tri) {
    if (!tri) {
        return -1;
    }
    const int idx = tri->editable_index;
    return (idx >= 0 && idx < static_cast<int>(cache.source_triangles.size()) &&
            cache.source_triangles[static_cast<size_t>(idx)].get() == tri)
        ? idx : -1;
}

std::vector<int> getEditablePolygonVertexIds(const SceneUI::EditableMeshCache& cache, int polygonFaceId) {
    if (polygonFaceId >= 0 && polygonFaceId < static_cast<int>(cache.polygon_faces.size())) {
        return cache.polygon_faces[polygonFaceId].vertex_ids;
    }
    if (polygonFaceId >= 0 && polygonFaceId < static_cast<int>(cache.faces.size())) {
        const auto& face = cache.faces[polygonFaceId];
        return { face.v0, face.v1, face.v2 };
    }
    return {};
}

const SceneUI::EditableEdge* getEditableSelectableEdge(const SceneUI::EditableMeshCache& cache, int edgeId) {
    if (edgeId >= 0 && edgeId < static_cast<int>(cache.polygon_edges.size())) {
        return &cache.polygon_edges[edgeId];
    }
    if (edgeId >= 0 && edgeId < static_cast<int>(cache.edges.size())) {
        return &cache.edges[edgeId];
    }
    return nullptr;
}

// Gather the cage-edge endpoint position pairs a crease op should touch, from the current
// sub-element selection. Crease lives on EDGES (the Catmull-Clark stencil reads only edge
// sharpness — corner/vertex behaviour falls out of how many sharp edges meet), so face and
// vertex selections are mapped to their implied edges, Blender-style:
//   • selected edges    -> those edges
//   • selected faces    -> every boundary edge of the face
//   • selected vertices -> every edge whose BOTH endpoints are selected (vertex-loop creasing)
// The union is taken so Combined mode and any single mode all work without a mode switch.
std::vector<std::pair<Vec3, Vec3>> collectSelectedCreaseEdgePositions(
        const SceneUI::EditableMeshCache& cache) {
    std::vector<std::pair<Vec3, Vec3>> pairs;
    const auto& verts = cache.vertices;
    auto vtxValid = [&](int v) { return v >= 0 && v < static_cast<int>(verts.size()); };
    auto addPair = [&](int a, int b) {
        if (a != b && vtxValid(a) && vtxValid(b)) {
            pairs.emplace_back(verts[a].local_position, verts[b].local_position);
        }
    };

    const auto& sel = cache.selection;

    for (const int edgeId : sel.edge_ids) {
        if (const SceneUI::EditableEdge* edge = getEditableSelectableEdge(cache, edgeId)) {
            addPair(edge->v0, edge->v1);
        }
    }
    for (const int faceId : sel.face_ids) {
        const std::vector<int> vids = getEditablePolygonVertexIds(cache, faceId);
        for (size_t i = 0; i < vids.size(); ++i) {
            addPair(vids[i], vids[(i + 1) % vids.size()]);
        }
    }
    if (sel.vertex_ids.size() >= 2) {
        const std::unordered_set<int> selVerts(sel.vertex_ids.begin(), sel.vertex_ids.end());
        const auto& edgeList = !cache.polygon_edges.empty() ? cache.polygon_edges : cache.edges;
        for (const auto& edge : edgeList) {
            if (selVerts.count(edge.v0) && selVerts.count(edge.v1)) {
                addPair(edge.v0, edge.v1);
            }
        }
    }
    return pairs;
}

unsigned long long makeEditablePackedEdgeKey(int a, int b) {
    if (b < a) {
        std::swap(a, b);
    }
    return (static_cast<unsigned long long>(static_cast<unsigned int>(a)) << 32ull) |
           static_cast<unsigned long long>(static_cast<unsigned int>(b));
}

// The cache's half-edge mesh is rebuilt only on topology changes; vertex
// moves since the last rebuild live in cache.vertices only. Ids are 1:1, so
// every half-edge operator MUST refresh positions before mutating, or the
// rebuilt triangle soup snaps the mesh back to its last-rebuild state.
void syncHalfEdgePositionsFromCache(MeshEdit::HalfEdgeMesh& heMesh,
                                    const SceneUI::EditableMeshCache& cache) {
    const size_t count = (std::min)(heMesh.vertices.size(), cache.vertices.size());
    for (size_t i = 0; i < count; ++i) {
        heMesh.vertices[i].position = cache.vertices[i].local_position;
    }
}

int findEditablePolygonEdgeId(
    const SceneUI::EditableMeshCache& cache,
    const std::unordered_map<unsigned long long, int>& edgeIdByKey,
    int a,
    int b) {
    const auto it = edgeIdByKey.find(makeEditablePackedEdgeKey(a, b));
    if (it == edgeIdByKey.end()) {
        return -1;
    }
    const int edgeId = it->second;
    return (edgeId >= 0 && edgeId < static_cast<int>(cache.polygon_edges.size())) ? edgeId : -1;
}

std::vector<int> collectEditableEdgeLoop(
    const SceneUI::EditableMeshCache& cache,
    int startEdgeId) {
    if (startEdgeId < 0 || startEdgeId >= static_cast<int>(cache.polygon_edges.size())) {
        return {};
    }

    std::vector<std::vector<int>> vertexToEdges(cache.vertices.size());
    for (size_t edgeId = 0; edgeId < cache.polygon_edges.size(); ++edgeId) {
        const auto& edge = cache.polygon_edges[edgeId];
        if (!isEditableVertexIdValid(cache, edge.v0) || !isEditableVertexIdValid(cache, edge.v1)) {
            continue;
        }
        vertexToEdges[edge.v0].push_back(static_cast<int>(edgeId));
        vertexToEdges[edge.v1].push_back(static_cast<int>(edgeId));
    }

    auto getOtherVertex = [&](int edgeId, int vertexId) -> int {
        const auto* edge = getEditableSelectableEdge(cache, edgeId);
        if (!edge) {
            return -1;
        }
        if (edge->v0 == vertexId) {
            return edge->v1;
        }
        if (edge->v1 == vertexId) {
            return edge->v0;
        }
        return -1;
    };

    auto normalizedEdgeDirection = [&](int fromVertexId, int toVertexId) -> Vec3 {
        if (!isEditableVertexIdValid(cache, fromVertexId) || !isEditableVertexIdValid(cache, toVertexId)) {
            return Vec3(0.0f, 0.0f, 0.0f);
        }
        const Vec3 delta =
            cache.vertices[toVertexId].local_position - cache.vertices[fromVertexId].local_position;
        const float lenSq = delta.length_squared();
        if (!std::isfinite(lenSq) || lenSq <= 1e-10f) {
            return Vec3(0.0f, 0.0f, 0.0f);
        }
        return delta / std::sqrt(lenSq);
    };

    std::vector<int> loopEdgeIds = { startEdgeId };
    std::unordered_set<int> visitedEdges = { startEdgeId };
    const auto& startEdge = cache.polygon_edges[startEdgeId];

    auto walkFromVertex = [&](int previousVertexId, int currentVertexId) {
        int currentEdgeId = startEdgeId;
        Vec3 incomingDirection = normalizedEdgeDirection(previousVertexId, currentVertexId);
        while (isEditableVertexIdValid(cache, currentVertexId)) {
            int bestEdgeId = -1;
            int bestNextVertexId = -1;
            float bestScore = 0.45f;

            for (const int candidateEdgeId : vertexToEdges[currentVertexId]) {
                if (candidateEdgeId == currentEdgeId) {
                    continue;
                }

                const int nextVertexId = getOtherVertex(candidateEdgeId, currentVertexId);
                if (nextVertexId < 0 || nextVertexId == previousVertexId) {
                    continue;
                }

                const Vec3 candidateDirection = normalizedEdgeDirection(currentVertexId, nextVertexId);
                const float score = incomingDirection.dot(candidateDirection);
                if (score > bestScore) {
                    bestScore = score;
                    bestEdgeId = candidateEdgeId;
                    bestNextVertexId = nextVertexId;
                }
            }

            if (bestEdgeId < 0 || bestNextVertexId < 0 || !visitedEdges.insert(bestEdgeId).second) {
                break;
            }

            loopEdgeIds.push_back(bestEdgeId);
            previousVertexId = currentVertexId;
            currentVertexId = bestNextVertexId;
            currentEdgeId = bestEdgeId;
            incomingDirection = normalizedEdgeDirection(previousVertexId, currentVertexId);
        }
    };

    walkFromVertex(startEdge.v0, startEdge.v1);
    walkFromVertex(startEdge.v1, startEdge.v0);

    std::sort(loopEdgeIds.begin(), loopEdgeIds.end());
    loopEdgeIds.erase(std::unique(loopEdgeIds.begin(), loopEdgeIds.end()), loopEdgeIds.end());
    return loopEdgeIds;
}

std::vector<int> collectEditableEdgeRing(
    const SceneUI::EditableMeshCache& cache,
    int startEdgeId) {
    if (startEdgeId < 0 || startEdgeId >= static_cast<int>(cache.polygon_edges.size())) {
        return {};
    }

    std::unordered_map<unsigned long long, int> edgeIdByKey;
    edgeIdByKey.reserve(cache.polygon_edges.size() * 2 + 1);
    for (size_t edgeId = 0; edgeId < cache.polygon_edges.size(); ++edgeId) {
        const auto& edge = cache.polygon_edges[edgeId];
        edgeIdByKey[makeEditablePackedEdgeKey(edge.v0, edge.v1)] = static_cast<int>(edgeId);
    }

    std::vector<std::vector<int>> edgeToFaces(cache.polygon_edges.size());
    for (size_t faceId = 0; faceId < cache.polygon_faces.size(); ++faceId) {
        const auto& face = cache.polygon_faces[faceId];
        if (face.vertex_ids.size() < 3) {
            continue;
        }
        for (size_t i = 0; i < face.vertex_ids.size(); ++i) {
            const int edgeId = findEditablePolygonEdgeId(
                cache,
                edgeIdByKey,
                face.vertex_ids[i],
                face.vertex_ids[(i + 1) % face.vertex_ids.size()]);
            if (edgeId >= 0) {
                edgeToFaces[edgeId].push_back(static_cast<int>(faceId));
            }
        }
    }

    auto findOppositeQuadEdge = [&](int faceId, int edgeId) -> int {
        if (faceId < 0 || faceId >= static_cast<int>(cache.polygon_faces.size())) {
            return -1;
        }
        const auto& face = cache.polygon_faces[faceId];
        if (face.vertex_ids.size() != 4) {
            return -1;
        }

        const auto* edge = getEditableSelectableEdge(cache, edgeId);
        if (!edge) {
            return -1;
        }

        for (size_t i = 0; i < face.vertex_ids.size(); ++i) {
            const int a = face.vertex_ids[i];
            const int b = face.vertex_ids[(i + 1) % face.vertex_ids.size()];
            if ((a == edge->v0 && b == edge->v1) || (a == edge->v1 && b == edge->v0)) {
                return findEditablePolygonEdgeId(
                    cache,
                    edgeIdByKey,
                    face.vertex_ids[(i + 2) % face.vertex_ids.size()],
                    face.vertex_ids[(i + 3) % face.vertex_ids.size()]);
            }
        }
        return -1;
    };

    std::vector<int> ringEdgeIds = { startEdgeId };
    std::unordered_set<int> visitedEdges = { startEdgeId };

    auto walkFromFace = [&](int startFaceId) {
        int currentFaceId = startFaceId;
        int currentEdgeId = startEdgeId;
        while (currentFaceId >= 0) {
            const int oppositeEdgeId = findOppositeQuadEdge(currentFaceId, currentEdgeId);
            if (oppositeEdgeId < 0 || !visitedEdges.insert(oppositeEdgeId).second) {
                break;
            }
            ringEdgeIds.push_back(oppositeEdgeId);

            int nextFaceId = -1;
            for (const int candidateFaceId : edgeToFaces[oppositeEdgeId]) {
                if (candidateFaceId != currentFaceId) {
                    nextFaceId = candidateFaceId;
                    break;
                }
            }

            currentEdgeId = oppositeEdgeId;
            currentFaceId = nextFaceId;
        }
    };

    if (startEdgeId >= 0 && startEdgeId < static_cast<int>(edgeToFaces.size())) {
        for (const int faceId : edgeToFaces[startEdgeId]) {
            walkFromFace(faceId);
        }
    }

    std::sort(ringEdgeIds.begin(), ringEdgeIds.end());
    ringEdgeIds.erase(std::unique(ringEdgeIds.begin(), ringEdgeIds.end()), ringEdgeIds.end());
    return ringEdgeIds;
}

void drawEditableScreenPolygonFill(ImDrawList* drawList, const std::vector<ImVec2>& screenVertices, ImU32 color) {
    if (!drawList || screenVertices.size() < 3) {
        return;
    }
    drawList->AddConvexPolyFilled(screenVertices.data(), static_cast<int>(screenVertices.size()), color);
}

Vec3 computeEditableFaceNormal(const SceneUI::EditableMeshCache& cache, const std::vector<int>& vertexIds) {
    if (vertexIds.size() < 3) {
        return Vec3(0.0f, 0.0f, 1.0f);
    }

    Vec3 normal(0.0f, 0.0f, 0.0f);
    for (size_t i = 0; i < vertexIds.size(); ++i) {
        const int currentId = vertexIds[i];
        const int nextId = vertexIds[(i + 1) % vertexIds.size()];
        if (!isEditableVertexIdValid(cache, currentId) || !isEditableVertexIdValid(cache, nextId)) {
            return Vec3(0.0f, 0.0f, 1.0f);
        }

        const Vec3& current = cache.vertices[currentId].local_position;
        const Vec3& next = cache.vertices[nextId].local_position;
        normal.x += (current.y - next.y) * (current.z + next.z);
        normal.y += (current.z - next.z) * (current.x + next.x);
        normal.z += (current.x - next.x) * (current.y + next.y);
    }
    const float lenSq = normal.length_squared();
    if (!std::isfinite(lenSq) || lenSq <= 1e-10f) {
        return Vec3(0.0f, 0.0f, 1.0f);
    }
    return normal / std::sqrt(lenSq);
}

Vec3 computeEditableReferenceNormal(
    const SceneUI::EditableMeshCache& cache,
    const std::vector<int>& triangleIds) {
    Vec3 accumulatedNormal(0.0f, 0.0f, 0.0f);
    int sampleCount = 0;
    for (const int triangleId : triangleIds) {
        if (triangleId < 0 || triangleId >= static_cast<int>(cache.faces.size())) {
            continue;
        }
        const auto& face = cache.faces[triangleId];
        if (!isEditableVertexIdValid(cache, face.v0) ||
            !isEditableVertexIdValid(cache, face.v1) ||
            !isEditableVertexIdValid(cache, face.v2)) {
            continue;
        }

        const Vec3 triangleNormal = computeEditableFaceNormal(cache, { face.v0, face.v1, face.v2 });
        if (!std::isfinite(triangleNormal.length_squared()) || triangleNormal.length_squared() <= 1e-10f) {
            continue;
        }
        accumulatedNormal += triangleNormal;
        ++sampleCount;
    }

    if (sampleCount <= 0) {
        return Vec3(0.0f, 0.0f, 1.0f);
    }

    const float lenSq = accumulatedNormal.length_squared();
    if (!std::isfinite(lenSq) || lenSq <= 1e-10f) {
        return Vec3(0.0f, 0.0f, 1.0f);
    }
    return accumulatedNormal / std::sqrt(lenSq);
}

std::vector<int> sortEditablePolygonVertices(
    const SceneUI::EditableMeshCache& cache,
    const std::vector<int>& unsortedVertexIds) {
    std::vector<int> ordered = unsortedVertexIds;
    if (ordered.size() <= 3) {
        return ordered;
    }

    Vec3 center(0.0f, 0.0f, 0.0f);
    for (const int vertexId : ordered) {
        if (!isEditableVertexIdValid(cache, vertexId)) {
            return {};
        }
        center += cache.vertices[vertexId].local_position;
    }
    center /= static_cast<float>(ordered.size());

    const Vec3 normal = computeEditableFaceNormal(cache, ordered);
    Vec3 axisX = cache.vertices[ordered[0]].local_position - center;
    if (axisX.length_squared() <= 1e-10f) {
        axisX = Vec3(1.0f, 0.0f, 0.0f);
    } else {
        axisX = axisX.normalize();
    }
    Vec3 axisY = normal.cross(axisX);
    if (axisY.length_squared() <= 1e-10f) {
        axisY = Vec3(0.0f, 1.0f, 0.0f);
    } else {
        axisY = axisY.normalize();
    }

    std::sort(ordered.begin(), ordered.end(),
        [&](int lhs, int rhs) {
            const Vec3 lhsOffset = cache.vertices[lhs].local_position - center;
            const Vec3 rhsOffset = cache.vertices[rhs].local_position - center;
            const float lhsAngle = std::atan2(lhsOffset.dot(axisY), lhsOffset.dot(axisX));
            const float rhsAngle = std::atan2(rhsOffset.dot(axisY), rhsOffset.dot(axisX));
            return lhsAngle < rhsAngle;
        });

    return ordered;
}

bool buildOrderedQuadFromTriangles(
    const SceneUI::EditableFace& faceA,
    const SceneUI::EditableFace& faceB,
    std::vector<int>& outVertexIds) {
    outVertexIds.clear();

    const int triA[3] = { faceA.v0, faceA.v1, faceA.v2 };
    const int triB[3] = { faceB.v0, faceB.v1, faceB.v2 };

    auto makeEdgeKey = [](int a, int b) {
        if (b < a) {
            std::swap(a, b);
        }
        return (static_cast<unsigned long long>(static_cast<unsigned int>(a)) << 32ull) |
               static_cast<unsigned long long>(static_cast<unsigned int>(b));
    };

    std::unordered_map<unsigned long long, int> edgeCounts;
    edgeCounts.reserve(6);
    std::vector<std::pair<int, int>> orientedEdges;
    orientedEdges.reserve(6);

    auto appendTriangleEdges = [&](const int tri[3]) {
        for (int i = 0; i < 3; ++i) {
            const int a = tri[i];
            const int b = tri[(i + 1) % 3];
            orientedEdges.emplace_back(a, b);
            edgeCounts[makeEdgeKey(a, b)] += 1;
        }
    };

    appendTriangleEdges(triA);
    appendTriangleEdges(triB);

    std::vector<std::pair<int, int>> boundaryEdges;
    boundaryEdges.reserve(4);
    for (const auto& edge : orientedEdges) {
        if (edgeCounts[makeEdgeKey(edge.first, edge.second)] == 1) {
            boundaryEdges.push_back(edge);
        }
    }
    if (boundaryEdges.size() != 4) {
        return false;
    }

    std::unordered_map<int, std::vector<int>> adjacency;
    adjacency.reserve(4);
    for (const auto& edge : boundaryEdges) {
        adjacency[edge.first].push_back(edge.second);
        adjacency[edge.second].push_back(edge.first);
    }

    int startVertex = boundaryEdges.front().first;
    for (const auto& [vertexId, neighbors] : adjacency) {
        if (neighbors.size() != 2) {
            return false;
        }
        startVertex = (std::min)(startVertex, vertexId);
    }

    outVertexIds.push_back(startVertex);
    int previousVertex = std::numeric_limits<int>::min();
    int currentVertex = startVertex;
    for (int step = 0; step < 3; ++step) {
        const auto adjacencyIt = adjacency.find(currentVertex);
        if (adjacencyIt == adjacency.end() || adjacencyIt->second.size() != 2) {
            return false;
        }

        int nextVertex = adjacencyIt->second[0];
        if (nextVertex == previousVertex) {
            nextVertex = adjacencyIt->second[1];
        }

        if (nextVertex == previousVertex || nextVertex == startVertex) {
            return false;
        }

        outVertexIds.push_back(nextVertex);
        previousVertex = currentVertex;
        currentVertex = nextVertex;
    }

    const auto lastAdjacencyIt = adjacency.find(currentVertex);
    if (lastAdjacencyIt == adjacency.end() || lastAdjacencyIt->second.size() != 2) {
        return false;
    }

    const bool closesLoop =
        (lastAdjacencyIt->second[0] == startVertex && lastAdjacencyIt->second[1] == previousVertex) ||
        (lastAdjacencyIt->second[1] == startVertex && lastAdjacencyIt->second[0] == previousVertex);
    if (!closesLoop) {
        return false;
    }

    return outVertexIds.size() == 4;
}

bool canMergeEditableTrianglesToQuad(
    const SceneUI::EditableMeshCache& cache,
    const SceneUI::EditableFace& faceA,
    const SceneUI::EditableFace& faceB,
    std::vector<int>& outVertexIds) {
    std::unordered_set<int> uniqueIds = { faceA.v0, faceA.v1, faceA.v2, faceB.v0, faceB.v1, faceB.v2 };
    if (uniqueIds.size() != 4) {
        return false;
    }

    std::vector<int> sharedIds;
    const int faceAVertices[3] = { faceA.v0, faceA.v1, faceA.v2 };
    const int faceBVertices[3] = { faceB.v0, faceB.v1, faceB.v2 };
    for (const int vertexA : faceAVertices) {
        for (const int vertexB : faceBVertices) {
            if (vertexA == vertexB) {
                sharedIds.push_back(vertexA);
            }
        }
    }
    if (sharedIds.size() != 2) {
        return false;
    }

    for (const int vertexId : uniqueIds) {
        if (!isEditableVertexIdValid(cache, vertexId)) {
            return false;
        }
    }

    const Vec3 normalA = computeEditableFaceNormal(cache, { faceA.v0, faceA.v1, faceA.v2 });
    const Vec3 normalB = computeEditableFaceNormal(cache, { faceB.v0, faceB.v1, faceB.v2 });
    if (normalA.dot(normalB) < 0.94f) {
        return false;
    }

    const float sharedEdgeLength =
        (cache.vertices[sharedIds[0]].local_position - cache.vertices[sharedIds[1]].local_position).length();
    if (!std::isfinite(sharedEdgeLength) || sharedEdgeLength <= 1e-6f) {
        return false;
    }

    auto isSharedEdgeLongestForFace = [&](const SceneUI::EditableFace& face) {
        const int faceVertices[3] = { face.v0, face.v1, face.v2 };
        float maxEdgeLength = 0.0f;
        for (int edgeIndex = 0; edgeIndex < 3; ++edgeIndex) {
            const int a = faceVertices[edgeIndex];
            const int b = faceVertices[(edgeIndex + 1) % 3];
            if (!isEditableVertexIdValid(cache, a) || !isEditableVertexIdValid(cache, b)) {
                return false;
            }
            const float edgeLength =
                (cache.vertices[a].local_position - cache.vertices[b].local_position).length();
            if (!std::isfinite(edgeLength)) {
                return false;
            }
            maxEdgeLength = (std::max)(maxEdgeLength, edgeLength);
        }

        // Recover quads only when the shared edge behaves like the triangulation diagonal.
        return sharedEdgeLength + 1e-5f >= maxEdgeLength;
    };
    if (!isSharedEdgeLongestForFace(faceA) || !isSharedEdgeLongestForFace(faceB)) {
        return false;
    }

    float averageEdgeLength = 0.0f;
    int edgeLengthSampleCount = 0;
    auto accumulateFaceEdgeLengths = [&](const SceneUI::EditableFace& face) {
        const int faceVertices[3] = { face.v0, face.v1, face.v2 };
        for (int edgeIndex = 0; edgeIndex < 3; ++edgeIndex) {
            const int a = faceVertices[edgeIndex];
            const int b = faceVertices[(edgeIndex + 1) % 3];
            if (!isEditableVertexIdValid(cache, a) || !isEditableVertexIdValid(cache, b)) {
                continue;
            }
            const float edgeLength =
                (cache.vertices[a].local_position - cache.vertices[b].local_position).length();
            if (!std::isfinite(edgeLength) || edgeLength <= 1e-6f) {
                continue;
            }
            averageEdgeLength += edgeLength;
            ++edgeLengthSampleCount;
        }
    };
    accumulateFaceEdgeLengths(faceA);
    accumulateFaceEdgeLengths(faceB);
    if (edgeLengthSampleCount > 0) {
        averageEdgeLength /= static_cast<float>(edgeLengthSampleCount);
    } else {
        averageEdgeLength = sharedEdgeLength;
    }

    const float maxPlaneDistance = (std::max)(averageEdgeLength * 0.085f, 1e-3f);
    const Vec3 planeOrigin = cache.vertices[faceA.v0].local_position;
    for (const int vertexId : uniqueIds) {
        const float planeDistance = std::fabs((cache.vertices[vertexId].local_position - planeOrigin).dot(normalA));
        if (!std::isfinite(planeDistance) || planeDistance > maxPlaneDistance) {
            return false;
        }
    }

    if (!buildOrderedQuadFromTriangles(faceA, faceB, outVertexIds)) {
        return false;
    }
    return outVertexIds.size() == 4;
}

void addEditablePolygonFaceFromTriangles(
    SceneUI::EditableMeshCache& cache,
    const std::vector<int>& vertexIds,
    const std::vector<int>& triangleIds) {
    if (vertexIds.size() < 3 || triangleIds.empty()) {
        return;
    }

    SceneUI::EditablePolygonFace polygonFace;
    polygonFace.vertex_ids = vertexIds;
    const Vec3 polygonNormal = computeEditableFaceNormal(cache, polygonFace.vertex_ids);
    const Vec3 referenceNormal = computeEditableReferenceNormal(cache, triangleIds);
    if (polygonNormal.dot(referenceNormal) < 0.0f) {
        std::reverse(polygonFace.vertex_ids.begin(), polygonFace.vertex_ids.end());
    }
    polygonFace.triangle_ids = triangleIds;
    cache.polygon_faces.push_back(std::move(polygonFace));
}

std::shared_ptr<Triangle> cloneTriangleForEdit(const std::shared_ptr<Triangle>& source) {
    if (!source) {
        return nullptr;
    }
    return std::make_shared<Triangle>(*source);
}

std::vector<std::shared_ptr<Triangle>> cloneTriangleVectorForEdit(
    const std::vector<std::shared_ptr<Triangle>>& triangles) {
    std::vector<std::shared_ptr<Triangle>> clones;
    clones.reserve(triangles.size());
    for (const auto& triangle : triangles) {
        if (auto clone = cloneTriangleForEdit(triangle)) {
            clones.push_back(clone);
        }
    }
    return clones;
}

std::array<Vec2, 4> buildExtrudedQuadUVs(
    const Vec3& a,
    const Vec3& b,
    const Vec3& c,
    const Vec3& d) {
    const float edgeLength = (b - a).length();
    const float height = (d - a).length();
    const float safeWidth = (std::max)(edgeLength, 1e-4f);
    const float safeHeight = (std::max)(height, 1e-4f);
    return {
        Vec2(0.0f, 0.0f),
        Vec2(safeWidth, 0.0f),
        Vec2(safeWidth, safeHeight),
        Vec2(0.0f, safeHeight)
    };
}

std::vector<Vec2> buildPolygonPlanarUVs(const std::vector<Vec3>& vertices, const Vec3& normal) {
    std::vector<Vec2> uvs(vertices.size(), Vec2(0.0f, 0.0f));
    if (vertices.size() < 3) {
        return uvs;
    }

    Vec3 center(0.0f, 0.0f, 0.0f);
    for (const Vec3& vertex : vertices) {
        center += vertex;
    }
    center /= static_cast<float>(vertices.size());

    Vec3 axisX = vertices[0] - center;
    if (axisX.length_squared() <= 1e-10f) {
        axisX = Vec3(1.0f, 0.0f, 0.0f);
    } else {
        axisX = axisX.normalize();
    }
    Vec3 axisY = normal.cross(axisX);
    if (axisY.length_squared() <= 1e-10f) {
        axisY = Vec3(0.0f, 1.0f, 0.0f);
    } else {
        axisY = axisY.normalize();
    }

    float minU = (std::numeric_limits<float>::max)();
    float minV = (std::numeric_limits<float>::max)();
    float maxU = -(std::numeric_limits<float>::max)();
    float maxV = -(std::numeric_limits<float>::max)();
    std::vector<std::pair<float, float>> projected;
    projected.reserve(vertices.size());
    for (const Vec3& vertex : vertices) {
        const Vec3 offset = vertex - center;
        const float u = offset.dot(axisX);
        const float v = offset.dot(axisY);
        projected.emplace_back(u, v);
        minU = (std::min)(minU, u);
        minV = (std::min)(minV, v);
        maxU = (std::max)(maxU, u);
        maxV = (std::max)(maxV, v);
    }

    const float spanU = (std::max)(maxU - minU, 1e-4f);
    const float spanV = (std::max)(maxV - minV, 1e-4f);
    for (size_t i = 0; i < projected.size(); ++i) {
        uvs[i] = Vec2((projected[i].first - minU) / spanU, (projected[i].second - minV) / spanV);
    }
    return uvs;
}

std::shared_ptr<Triangle> resolveEditablePolygonTemplateTriangle(
    const SceneUI::EditableMeshCache& cache,
    const std::vector<std::shared_ptr<Triangle>>& fallbackMesh,
    int faceId) {
    if (faceId >= 0 && faceId < static_cast<int>(cache.polygon_faces.size())) {
        const auto& polygonFace = cache.polygon_faces[faceId];
        for (const int triangleId : polygonFace.triangle_ids) {
            if (triangleId >= 0 &&
                triangleId < static_cast<int>(cache.faces.size()) &&
                cache.faceTri(cache.faces[triangleId])) {
                return cache.faceTriShared(cache.faces[triangleId]);
            }
        }
    }
    return !fallbackMesh.empty() ? fallbackMesh.front() : nullptr;
}

void appendTriangulatedEditablePolygon(
    const SceneUI::EditableMeshCache& cache,
    const std::vector<int>& vertexIds,
    const std::shared_ptr<Triangle>& templateTriangle,
    std::vector<std::shared_ptr<Triangle>>& outMesh) {
    if (!templateTriangle || vertexIds.size() < 3) {
        return;
    }

    std::vector<Vec3> vertices;
    vertices.reserve(vertexIds.size());
    for (const int vertexId : vertexIds) {
        if (!isEditableVertexIdValid(cache, vertexId)) {
            return;
        }
        vertices.push_back(cache.vertices[vertexId].local_position);
    }

    const Vec3 faceNormal = computeEditableFaceNormal(cache, vertexIds);
    const std::vector<Vec2> faceUVs = buildPolygonPlanarUVs(vertices, faceNormal);
    for (size_t i = 1; i + 1 < vertices.size(); ++i) {
        auto tri = cloneTriangleForEdit(templateTriangle);
        if (!tri) {
            continue;
        }
        tri->setOriginalVertexPosition(0, vertices[0]);
        tri->setOriginalVertexPosition(1, vertices[i]);
        tri->setOriginalVertexPosition(2, vertices[i + 1]);
        tri->setOriginalVertexNormal(0, faceNormal);
        tri->setOriginalVertexNormal(1, faceNormal);
        tri->setOriginalVertexNormal(2, faceNormal);
        tri->set_normals(faceNormal, faceNormal, faceNormal);
        tri->setUVCoordinates(faceUVs[0], faceUVs[i], faceUVs[i + 1]);
        tri->markAABBDirty();
        tri->updateTransformedVertices();
        outMesh.push_back(tri);
    }
}

// Rebuild the whole triangle soup from a (mutated) half-edge mesh. Face ids
// below originalFaceCount map 1:1 to cache polygon faces; faces created by
// operators resolve their template (material inheritance) transitively
// through faceSourceMap. Same planar-UV / flat-normal convention as
// appendTriangulatedEditablePolygon, but positions come from the half-edge
// mesh so operator-created vertices are included.
std::vector<std::shared_ptr<Triangle>> rebuildTriangleSoupFromHalfEdge(
    const MeshEdit::HalfEdgeMesh& heMesh,
    const SceneUI::EditableMeshCache& cache,
    const std::vector<std::shared_ptr<Triangle>>& fallbackMesh,
    MeshEdit::HEIndex originalFaceCount,
    const std::unordered_map<MeshEdit::HEIndex, MeshEdit::HEIndex>& faceSourceMap) {
    auto resolveSourceFaceId = [&](MeshEdit::HEIndex f) -> int {
        const size_t guard = faceSourceMap.size() + 1;
        for (size_t step = 0; step < guard && f >= originalFaceCount; ++step) {
            const auto it = faceSourceMap.find(f);
            if (it == faceSourceMap.end()) {
                break;
            }
            f = it->second;
        }
        return (f >= 0 && f < originalFaceCount) ? static_cast<int>(f) : -1;
    };

    std::vector<std::shared_ptr<Triangle>> outMesh;
    outMesh.reserve(heMesh.liveFaceCount() * 2);
    std::vector<MeshEdit::HEIndex> faceVertexIds;
    for (MeshEdit::HEIndex f = 0; f < static_cast<MeshEdit::HEIndex>(heMesh.faces.size()); ++f) {
        if (heMesh.faces[f].removed) {
            continue;
        }
        heMesh.collectFaceVertices(f, faceVertexIds);
        if (faceVertexIds.size() < 3) {
            continue;
        }
        std::shared_ptr<Triangle> templateTriangle = resolveEditablePolygonTemplateTriangle(
            cache, fallbackMesh, resolveSourceFaceId(f));
        if (!templateTriangle) {
            continue;
        }

        std::vector<Vec3> faceVertices;
        faceVertices.reserve(faceVertexIds.size());
        for (const MeshEdit::HEIndex v : faceVertexIds) {
            faceVertices.push_back(heMesh.vertices[v].position);
        }
        Vec3 faceNormal = heMesh.faceNormal(f);
        if (faceNormal.length_squared() <= 1e-10f) {
            faceNormal = Vec3(0.0f, 1.0f, 0.0f);
        }
        const std::vector<Vec2> faceUVs = buildPolygonPlanarUVs(faceVertices, faceNormal);
        for (size_t i = 1; i + 1 < faceVertices.size(); ++i) {
            auto newTriangle = cloneTriangleForEdit(templateTriangle);
            if (!newTriangle) {
                continue;
            }
            newTriangle->setOriginalVertexPosition(0, faceVertices[0]);
            newTriangle->setOriginalVertexPosition(1, faceVertices[i]);
            newTriangle->setOriginalVertexPosition(2, faceVertices[i + 1]);
            newTriangle->setOriginalVertexNormal(0, faceNormal);
            newTriangle->setOriginalVertexNormal(1, faceNormal);
            newTriangle->setOriginalVertexNormal(2, faceNormal);
            newTriangle->set_normals(faceNormal, faceNormal, faceNormal);
            newTriangle->setUVCoordinates(faceUVs[0], faceUVs[i], faceUVs[i + 1]);
            newTriangle->markAABBDirty();
            newTriangle->updateTransformedVertices();
            outMesh.push_back(newTriangle);
        }
    }
    return outMesh;
}

bool buildMergedEditablePolygonBoundary(
    const std::vector<std::vector<int>>& inputFaces,
    std::vector<int>& outVertexIds) {
    outVertexIds.clear();
    if (inputFaces.size() < 2) {
        return false;
    }

    std::unordered_map<unsigned long long, int> edgeCounts;
    std::vector<std::pair<int, int>> orientedEdges;
    edgeCounts.reserve(inputFaces.size() * 4);
    orientedEdges.reserve(inputFaces.size() * 4);

    for (const auto& face : inputFaces) {
        if (face.size() < 3) {
            return false;
        }
        for (size_t i = 0; i < face.size(); ++i) {
            const int a = face[i];
            const int b = face[(i + 1) % face.size()];
            orientedEdges.emplace_back(a, b);
            edgeCounts[makeEditablePackedEdgeKey(a, b)] += 1;
        }
    }

    std::vector<std::pair<int, int>> boundaryEdges;
    boundaryEdges.reserve(orientedEdges.size());
    for (const auto& edge : orientedEdges) {
        if (edgeCounts[makeEditablePackedEdgeKey(edge.first, edge.second)] == 1) {
            boundaryEdges.push_back(edge);
        }
    }
    if (boundaryEdges.size() < 3) {
        return false;
    }

    std::unordered_map<int, std::vector<int>> adjacency;
    for (const auto& edge : boundaryEdges) {
        adjacency[edge.first].push_back(edge.second);
        adjacency[edge.second].push_back(edge.first);
    }

    for (const auto& [vertexId, neighbors] : adjacency) {
        (void)vertexId;
        if (neighbors.size() != 2) {
            return false;
        }
    }

    int startVertex = adjacency.begin()->first;
    for (const auto& [vertexId, neighbors] : adjacency) {
        (void)neighbors;
        startVertex = (std::min)(startVertex, vertexId);
    }

    outVertexIds.push_back(startVertex);
    int previousVertex = std::numeric_limits<int>::min();
    int currentVertex = startVertex;
    while (true) {
        const auto adjacencyIt = adjacency.find(currentVertex);
        if (adjacencyIt == adjacency.end() || adjacencyIt->second.size() != 2) {
            return false;
        }

        int nextVertex = adjacencyIt->second[0];
        if (nextVertex == previousVertex) {
            nextVertex = adjacencyIt->second[1];
        }

        if (nextVertex == previousVertex) {
            return false;
        }
        if (nextVertex == startVertex) {
            break;
        }
        if (std::find(outVertexIds.begin(), outVertexIds.end(), nextVertex) != outVertexIds.end()) {
            return false;
        }

        outVertexIds.push_back(nextVertex);
        previousVertex = currentVertex;
        currentVertex = nextVertex;
    }

    return outVertexIds.size() >= 3;
}
} // Temporarily close anonymous namespace to expose global functions

bool hasEnabledSubdivisionPreview(const MeshModifiers::ModifierStack& stack) {
    for (const auto& mod : stack.modifiers) {
        if (!mod.enabled) {
            continue;
        }
        if (mod.type == MeshModifiers::ModifierType::FlatSubdivision ||
            mod.type == MeshModifiers::ModifierType::SmoothSubdivision ||
            mod.type == MeshModifiers::ModifierType::CatmullClark) {
            return true;
        }
    }
    return false;
}

MeshModifiers::ModifierStack buildSubdivisionPreviewEvaluationStack(
    const MeshModifiers::ModifierStack& stack,
    bool interactivePreview) {
    // Blender-style Viewport vs Render level: the Rendered viewport (path-trace) shows the
    // high RENDER level; Solid/Matcap/edit shows the cheaper VIEWPORT level. Bake the mode-
    // appropriate level into the preview copy's `levels` (which evaluate() consumes).
    g_solid_viewport_active;
    const bool renderMode = !g_solid_viewport_active;

    MeshModifiers::ModifierStack previewStack = stack;
    bool keptSubdivisionStage = false;
    for (auto& mod : previewStack.modifiers) {
        if (!mod.enabled) {
            continue;
        }
        if (mod.type != MeshModifiers::ModifierType::FlatSubdivision &&
            mod.type != MeshModifiers::ModifierType::SmoothSubdivision &&
            mod.type != MeshModifiers::ModifierType::CatmullClark) {
            continue;
        }

        int effective = renderMode ? mod.renderLevels : mod.levels;

        if (interactivePreview) {
            // During an interactive cage drag keep only the first subdivision stage, clamped
            // to level 1, so edits stay responsive regardless of the authored levels.
            if (!keptSubdivisionStage) {
                effective = std::clamp(effective, 0, 1);
                mod.levels = effective;
                mod.enabled = effective > 0;
                keptSubdivisionStage = mod.enabled;
            } else {
                mod.enabled = false;
            }
        } else {
            mod.levels = effective;
        }
    }

    return previewStack;
}

bool isSubdivisionPreviewActive(UIContext& ctx, const std::string& objectName) {
    if (objectName.empty()) {
        return false;
    }
    const auto stackIt = ctx.scene.mesh_modifiers.find(objectName);
    if (stackIt == ctx.scene.mesh_modifiers.end()) {
        return false;
    }
    return hasEnabledSubdivisionPreview(stackIt->second) &&
        ctx.scene.base_mesh_cache.find(objectName) != ctx.scene.base_mesh_cache.end() &&
        !ctx.scene.base_mesh_cache[objectName].empty();
}

// Fingerprint of everything the steady-state subdivision preview depends on: the
// subdivision modifier params and the base (cage) mesh vertex positions. When this
// changes the display mesh must be re-evaluated AND re-synced to all backends; when it
// is stable we can skip the (expensive) every-frame rebuild entirely.
std::size_t computeSubdivisionPreviewSignature(UIContext& ctx, const std::string& objectName) {
    std::size_t h = std::hash<std::string>{}(objectName);
    auto mix = [&h](std::size_t v) {
        h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    };

    // Viewport vs Render mode selects which subdivision level the preview evaluates, so a
    // Solid<->Rendered switch must re-evaluate the display mesh.
   
    mix(g_solid_viewport_active ? 1u : 2u);

    const auto stackIt = ctx.scene.mesh_modifiers.find(objectName);
    if (stackIt != ctx.scene.mesh_modifiers.end()) {
        for (const auto& mod : stackIt->second.modifiers) {
            if (mod.type != MeshModifiers::ModifierType::FlatSubdivision &&
                mod.type != MeshModifiers::ModifierType::SmoothSubdivision &&
                mod.type != MeshModifiers::ModifierType::CatmullClark) {
                continue;
            }
            mix(mod.enabled ? 1u : 0u);
            mix(static_cast<std::size_t>(static_cast<int>(mod.type)));
            mix(static_cast<std::size_t>(mod.levels));
            mix(static_cast<std::size_t>(mod.renderLevels));
            mix(std::hash<float>{}(mod.smoothAngle));
        }
    }

    const auto baseIt = ctx.scene.base_mesh_cache.find(objectName);
    if (baseIt != ctx.scene.base_mesh_cache.end()) {
        mix(baseIt->second.size());
        for (const auto& tri : baseIt->second) {
            if (!tri) {
                continue;
            }
            for (int c = 0; c < 3; ++c) {
                const Vec3& p = tri->getOriginalVertexPosition(c);
                mix(std::hash<float>{}(p.x));
                mix(std::hash<float>{}(p.y));
                mix(std::hash<float>{}(p.z));
            }
        }
    }

    return h;
}

std::vector<std::shared_ptr<Triangle>> evaluateDisplayMeshFromBase(
    const std::vector<std::shared_ptr<Triangle>>& baseMesh,
    const MeshModifiers::ModifierStack& stack) {
    if (stack.modifiers.empty()) {
        return cloneTriangleVectorForEdit(baseMesh);
    }
    return stack.evaluate(baseMesh);
}

namespace { // Reopen anonymous namespace

float saturateFloat(float value) {
    if (!std::isfinite(value)) {
        return 0.0f;
    }
    return (std::max)(0.0f, (std::min)(1.0f, value));
}

bool isFiniteVec3(const Vec3& value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

float sanitizeFiniteFloat(float value, float fallback, float minValue, float maxValue) {
    if (!std::isfinite(value)) {
        return fallback;
    }
    return std::clamp(value, minValue, maxValue);
}

float computeSafeClayStrength(float brushStrength, float brushFlow) {
    const float safeStrength = sanitizeFiniteFloat(brushStrength, 1.0f, 0.0f, 10.0f);
    const float safeFlow = sanitizeFiniteFloat(brushFlow, 1.0f, 0.05f, 4.0f);
    const float combined = safeStrength * safeFlow;
    if (combined <= 1.0f) {
        return combined;
    }

    // Keep the first unit of strength fully responsive, then compress harder
    // so high clay strengths add mass without producing deep inversion artifacts.
    const float overshoot = combined - 1.0f;
    return 1.0f + std::sqrt(overshoot) * 0.42f;
}

float computeStrokeAdvanceFactor(float strokeDistance, float strokeSpacing) {
    const float safeDistance = sanitizeFiniteFloat(strokeDistance, 0.0f, 0.0f, 100000.0f);
    const float safeSpacing = sanitizeFiniteFloat(strokeSpacing, 0.01f, 0.0001f, 100000.0f);
    const float normalizedAdvance = safeDistance / safeSpacing;
    if (normalizedAdvance <= 0.35f) {
        return 0.5f;
    }
    if (normalizedAdvance <= 1.0f) {
        return 0.5f + (normalizedAdvance - 0.35f) * (0.5f / 0.65f);
    }
    return std::clamp(1.0f + (normalizedAdvance - 1.0f) * 0.45f, 1.0f, 2.4f);
}

float computeClayRadiusCompensation(float radiusWorld) {
    const float safeRadius = sanitizeFiniteFloat(radiusWorld, 0.3f, 0.0001f, 100000.0f);
    constexpr float kReferenceRadius = 0.3f;
    const float normalizedRadius = safeRadius / kReferenceRadius;
    if (normalizedRadius <= 1.0f) {
        return 1.0f;
    }
    // Full inverse: radius controls area, not height.
    // Large brushes affect more vertices but deposit the same height.
    return std::clamp(kReferenceRadius / safeRadius, 0.12f, 1.0f);
}

float computeLargeBrushSurfaceFactor(float radiusWorld) {
    const float safeRadius = sanitizeFiniteFloat(radiusWorld, 0.3f, 0.0001f, 100000.0f);
    // 1.0 world units and below: no extra cleanup pressure.
    // Past that, gradually increase smoothing/anti-pit behavior for broad brushes.
    return std::clamp((safeRadius - 1.0f) / 3.0f, 0.0f, 1.0f);
}

float computeLargeBrushProjectionBlend(float radiusWorld) {
    const float safeRadius = sanitizeFiniteFloat(radiusWorld, 0.3f, 0.0001f, 100000.0f);
    return std::clamp((safeRadius - 1.2f) / 2.8f, 0.0f, 1.0f);
}

float computeHybridBrushDistance(
    const Vec3& toPoint,
    const Vec3& hitNormalWorld,
    float planarDistance,
    float radiusWorld) {
    // Capsule along the hit normal: vertex normalde 0.5*R bandı içindeyken planar
    // mesafe (silindir gibi) kullanılır; bu band dışına çıkanlarda height fazlası
    // mesafeye eklenip karekök ile birleşir. Bu, küçük fırçaların sonsuz silindir
    // gibi davranıp ince duvarın arka yüzeyini ya da fırça ekseninde uzaktaki
    // katmanları yutmasını engeller.
    const float heightAlong = toPoint.dot(hitNormalWorld);
    const float halfLength = radiusWorld * 0.5f;
    const float heightOver = (std::max)(0.0f, std::abs(heightAlong) - halfLength);
    return std::sqrt(planarDistance * planarDistance + heightOver * heightOver);
}

float computeFrontFaceBrushPenalty(
    float heightFromPlane,
    float radiusWorld,
    bool frontFacesOnly) {
    if (!frontFacesOnly) {
        return 1.0f;
    }

    const float safeRadius = sanitizeFiniteFloat(radiusWorld, 0.3f, 0.0001f, 100000.0f);
    // Yumuşak kesim bandını biraz daha geniş "tam kabul" tut: yüksek vertex
    // yoğunluğunda yüzey normalleri dalgalanan vertexler eskiden bu banda
    // düşüp yutuluyordu. Capsule height-clamp (computeHybridBrushDistance)
    // arka yüzü zaten sıkı kapatıyor; burada false-negative'leri azaltıyoruz.
    const float softRejectStart = -safeRadius * 0.22f;
    const float softRejectEnd = -safeRadius * (0.50f + 0.20f * computeLargeBrushProjectionBlend(radiusWorld));
    if (heightFromPlane >= softRejectStart) {
        return 1.0f;
    }
    if (heightFromPlane <= softRejectEnd) {
        return 0.0f;
    }

    const float t = saturateFloat((heightFromPlane - softRejectEnd) / (softRejectStart - softRejectEnd));
    return t * t * (3.0f - 2.0f * t);
}

float computeRepeatedStrokeDamping(
    float strokeDistance,
    float radiusWorld,
    float dt,
    bool strokeHasAccumulated) {
    if (!strokeHasAccumulated) {
        return 1.0f;
    }

    const float safeRadius = sanitizeFiniteFloat(radiusWorld, 0.3f, 0.0001f, 100000.0f);
    const float safeDistance = sanitizeFiniteFloat(strokeDistance, 0.0f, 0.0f, 100000.0f);
    const float safeDt = sanitizeFiniteFloat(dt, 1.0f / 60.0f, 1.0f / 240.0f, 0.25f);

    const float movementWindow = (std::max)(safeRadius * 0.14f, 1e-4f);
    const float movementRatio = saturateFloat(safeDistance / movementWindow);
    const float movementRecovery = movementRatio * movementRatio * (3.0f - 2.0f * movementRatio);

    const float dtT = saturateFloat((safeDt - (1.0f / 90.0f)) / ((1.0f / 18.0f) - (1.0f / 90.0f)));
    // Yavaş hareketli stroke'larda bile vertex ağırlığı 0.2'ye inip epsilon kapısına
    // takılıyordu; yüksek vertex yoğunluğunda fırça altında "atlanan" vertexlerin
    // başlıca nedeni buydu. Tabanı 0.55-0.75 aralığına yükseltiyoruz.
    const float lowMotionDamping = std::lerp(0.55f, 0.75f, 1.0f - dtT);
    return std::lerp(lowMotionDamping, 1.0f, movementRecovery);
}

Vec3 sanitizeVec3(const Vec3& value, const Vec3& fallback) {
    return isFiniteVec3(value) ? value : fallback;
}

Vec3 resolveEditableSnapshotLocalPosition(
    const SceneUI::EditableMeshCache& cache,
    const std::vector<Vec3>& snapshotLocalPositions,
    size_t vertexId) {
    if (vertexId >= cache.vertex_positions.size()) {
        return Vec3(0.0f, 0.0f, 0.0f);
    }
    if (vertexId < snapshotLocalPositions.size() && isFiniteVec3(snapshotLocalPositions[vertexId])) {
        return snapshotLocalPositions[vertexId];
    }
    return cache.vertex_positions[vertexId];
}

Vec3 computeBoundarySafeSmoothDelta(
    const SceneUI::EditableMeshCache& cache,
    const std::vector<Vec3>& snapshotLocalPositions,
    size_t vertexId,
    float brushStrength,
    float dt,
    float weight,
    float localRadius) {
    if (vertexId >= cache.vertex_positions.size()) {
        return Vec3(0.0f, 0.0f, 0.0f);
    }

    const auto& neighbors = cache.vertex_neighbors[vertexId];
    if (neighbors.empty()) {
        return Vec3(0.0f, 0.0f, 0.0f);
    }

    const bool isBoundary = cache.vertex_is_boundary[vertexId] != 0;
    Vec3 average(0.0f, 0.0f, 0.0f);
    int neighborCount = 0;
    for (const int neighborId : neighbors) {
        if (neighborId < 0 || neighborId >= static_cast<int>(cache.vertex_positions.size())) {
            continue;
        }
        const bool neighborIsBoundary = cache.vertex_is_boundary[static_cast<size_t>(neighborId)] != 0;
        if (isBoundary && !neighborIsBoundary) {
            continue;
        }
        average += resolveEditableSnapshotLocalPosition(cache, snapshotLocalPositions, static_cast<size_t>(neighborId));
        ++neighborCount;
    }

    if (neighborCount == 0) {
        return Vec3(0.0f, 0.0f, 0.0f);
    }

    average /= static_cast<float>(neighborCount);
    float smoothFactor = brushStrength * dt * 10.0f * weight;
    if (isBoundary) {
        smoothFactor *= 0.35f;
    }

    const Vec3 currentLocal = resolveEditableSnapshotLocalPosition(cache, snapshotLocalPositions, vertexId);
    Vec3 smoothDelta = sanitizeVec3(
        (average - currentLocal) * smoothFactor,
        Vec3(0.0f, 0.0f, 0.0f));
    const float deltaLen = smoothDelta.length();
    const float maxDelta = (std::max)(localRadius * (isBoundary ? 0.03f : 0.08f), 1e-5f);
    if (std::isfinite(deltaLen) && deltaLen > maxDelta) {
        smoothDelta *= (maxDelta / deltaLen);
    }
    return smoothDelta;
}
Vec3 safeNormalizeVec3(const Vec3& value, const Vec3& fallback) {
    if (!isFiniteVec3(value)) {
        return fallback;
    }
    const float lenSq = value.length_squared();
    if (!std::isfinite(lenSq) || lenSq <= 1e-12f) {
        return fallback;
    }
    const Vec3 normalized = value / std::sqrt(lenSq);
    return isFiniteVec3(normalized) ? normalized : fallback;
}

Vec3 computeStableSculptHitNormal(const HitRecord& hit, const Vec3& fallbackNormal) {
    const Vec3 fallback = safeNormalizeVec3(fallbackNormal, Vec3(0.0f, 1.0f, 0.0f));
    if (hit.interpolated_normal.length_squared() > 1e-8f) {
        return safeNormalizeVec3(hit.interpolated_normal, fallback);
    }
    if (!hit.triangle) {
        return fallback;
    }

    const Vec3 e1 = hit.triangle->vertices[1].position - hit.triangle->vertices[0].position;
    const Vec3 e2 = hit.triangle->vertices[2].position - hit.triangle->vertices[0].position;
    Vec3 geometricNormal = Vec3::cross(e1, e2);
    const float geometricLenSq = geometricNormal.length_squared();
    if (!std::isfinite(geometricLenSq) || geometricLenSq <= 1e-12f) {
        return fallback;
    }
    geometricNormal = geometricNormal / std::sqrt(geometricLenSq);

    return geometricNormal;
}

Vec3 computeOrientationPreservingFaceNormal(const Triangle& triangle, const Vec3& fallbackNormal) {
    Vec3 faceNormal = Vec3::cross(
        triangle.vertices[1].original - triangle.vertices[0].original,
        triangle.vertices[2].original - triangle.vertices[0].original);
    const float faceLenSq = faceNormal.length_squared();
    if (!std::isfinite(faceLenSq) || faceLenSq <= 1e-12f) {
        return safeNormalizeVec3(fallbackNormal, Vec3(0.0f, 1.0f, 0.0f));
    }
    faceNormal = faceNormal / std::sqrt(faceLenSq);

    Vec3 referenceNormal(0.0f, 0.0f, 0.0f);
    for (int corner = 0; corner < 3; ++corner) {
        if (triangle.vertices[corner].originalNormal.length_squared() > 1e-8f) {
            referenceNormal += triangle.vertices[corner].originalNormal;
        } else if (triangle.vertices[corner].normal.length_squared() > 1e-8f) {
            referenceNormal += triangle.vertices[corner].normal;
        }
    }

    if (referenceNormal.length_squared() > 1e-8f &&
        faceNormal.dot(referenceNormal) < 0.0f) {
        faceNormal = -faceNormal;
    }
    return faceNormal;
}

// --- Sculpt mask helpers -------------------------------------------------
// The mask is a per-vertex protection weight (0 = free, 1 = frozen) sized 1:1
// with the editable mesh cache. It survives across strokes but is rebuilt when
// the cache revision changes (any topology edit), since vertex ids may shift.
void ensureSculptMaskSized(SceneUI::SculptMaskState& mask,
                           const SceneUI::EditableMeshCache& cache,
                           const std::string& objectName) {
    const size_t vertexCount = cache.vertices.size();
    const bool stale = mask.object_name != objectName ||
                       mask.cache_revision != cache.revision ||
                       mask.values.size() != vertexCount;
    if (stale) {
        mask.object_name = objectName;
        mask.cache_revision = cache.revision;
        mask.values.assign(vertexCount, 0.0f);
        mask.has_any = false;
    }
}

// Wet-clay field is sized 1:1 with the editable cache, exactly like the mask, and reset
// whenever the cache revision changes (topology edits shift vertex ids).
void ensureSculptWetClaySized(SceneUI::SculptWetClayState& wet,
                              const SceneUI::EditableMeshCache& cache,
                              const std::string& objectName) {
    const size_t vertexCount = cache.vertices.size();
    const bool stale = wet.object_name != objectName ||
                       wet.cache_revision != cache.revision ||
                       wet.wetness.size() != vertexCount;
    if (stale) {
        wet.object_name = objectName;
        wet.cache_revision = cache.revision;
        wet.wetness.assign(vertexCount, 0.0f);
        wet.flow_anchor.assign(vertexCount, Vec3(0.0f, 0.0f, 0.0f));
        wet.flow_normal.assign(vertexCount, Vec3(0.0f, 1.0f, 0.0f));
        wet.active_list.clear();
        wet.has_any = false;
    }
}

// Smooth (averaged) shading normal of an editable-cache vertex, read from its incident
// triangle corner normals. Used to anchor a wet vertex's flow direction at wetting time.
inline Vec3 editableVertexNormal(const SceneUI::EditableMeshCache& cache, size_t vid) {
    if (vid >= cache.vertices.size()) {
        return Vec3(0.0f, 1.0f, 0.0f);
    }
    Vec3 nsum(0.0f, 0.0f, 0.0f);
    for (const auto& ref : cache.vertices[vid].refs) {
        if (const Triangle* tri = cache.refTri(ref)) {
            nsum += tri->getOriginalVertexNormal(ref.corner);
        }
    }
    return safeNormalizeVec3(nsum, Vec3(0.0f, 1.0f, 0.0f));
}

// Per-vertex flow mobility for wet-clay gravity flow. Homogeneous = 1.0 everywhere
// (uniform sheet). Heterogeneous = a spatially-coherent noise in ~[0.2,1] derived from
// the LOCAL position, so neighbouring regions creep downhill at different rates (the
// "different densities inside the mud" look). Sum-of-sines value noise: cheap,
// deterministic, no storage — the same vertex always gets the same mobility.
inline float wetClayFlowMobility(const Vec3& localPos, bool hetero, float scale) {
    if (!hetero) {
        return 1.0f;
    }
    const float s = (std::max)(scale, 1e-3f);
    const float n = std::sin(localPos.x * s * 1.00f + 1.3f)
                  + std::sin(localPos.y * s * 1.30f + 2.1f)
                  + std::sin(localPos.z * s * 1.90f + 3.7f);
    const float t = saturateFloat(0.5f + n * (1.0f / 6.0f)); // [-3,3] -> [0,1]
    return 0.2f + 0.8f * t;
}

// Returns the brush attenuation for a vertex: 1 = full effect, 0 = fully masked.
inline float sculptMaskFactor(const SceneUI::SculptMaskState& mask, size_t vertexId) {
    if (!mask.has_any || vertexId >= mask.values.size()) {
        return 1.0f;
    }
    return 1.0f - std::clamp(mask.values[vertexId], 0.0f, 1.0f);
}

// Rotate an object-local point/vector about a local axis (0=X,1=Y,2=Z) through
// the object origin. Used for radial symmetry — works for both positions and
// direction vectors since the rotation is about the origin.
inline Vec3 rotateLocalAroundAxis(const Vec3& p, int axis, float angleRad) {
    const float c = std::cos(angleRad);
    const float s = std::sin(angleRad);
    switch (axis) {
    case 0: return Vec3(p.x, c * p.y - s * p.z, s * p.y + c * p.z); // X
    case 2: return Vec3(c * p.x - s * p.y, s * p.x + c * p.y, p.z); // Z
    case 1:
    default: return Vec3(c * p.x + s * p.z, p.y, -s * p.x + c * p.z); // Y
    }
}

void applyClayPolishPass(
    const SceneUI::EditableMeshCache& cache,
    std::vector<Vec3>& updatedLocalPositions,
    const std::vector<int>& touchedVertexIds,
    const Matrix4x4& transform,
    const Matrix4x4& inverseTransform,
    const Vec3& strokeNormalWorld,
    float brushStrength,
    float dt,
    float localRadius,
    float polishStrength) {
    if (touchedVertexIds.empty() || updatedLocalPositions.empty()) {
        return;
    }

    const Vec3 strokeNormal = safeNormalizeVec3(strokeNormalWorld, Vec3(0.0f, 1.0f, 0.0f));
    const float safePolish = sanitizeFiniteFloat(polishStrength, 0.0f, 0.0f, 1.0f);
    if (safePolish <= 1e-5f) {
        return;
    }

    const std::vector<Vec3> snapshot = updatedLocalPositions;
    const float maxDelta = (std::max)(localRadius * 0.035f, 1e-5f);

    auto processVertex = [&](const int vertexIdInt) {
        if (vertexIdInt < 0 || vertexIdInt >= static_cast<int>(cache.vertex_positions.size())) {
            return;
        }
        const size_t vertexId = static_cast<size_t>(vertexIdInt);
        const auto& neighbors = cache.vertex_neighbors[vertexId];
        if (neighbors.empty()) {
            return;
        }

        const bool isBoundary = cache.vertex_is_boundary[vertexId] != 0;
        Vec3 average(0.0f, 0.0f, 0.0f);
        int neighborCount = 0;
        for (const int neighborId : neighbors) {
            if (neighborId < 0 || neighborId >= static_cast<int>(cache.vertex_positions.size())) {
                continue;
            }
            // Include ALL neighbors for averaging — boundary vertices need
            // interior neighbors to stay anchored and avoid folding.
            average += snapshot[static_cast<size_t>(neighborId)];
            ++neighborCount;
        }
        if (neighborCount == 0) {
            return;
        }

        average /= static_cast<float>(neighborCount);
        const Vec3 currentLocal = snapshot[vertexId];
        const Vec3 currentWorld = sanitizeVec3(transform.transform_point(currentLocal), currentLocal);
        const Vec3 averageWorld = sanitizeVec3(transform.transform_point(average), average);
        Vec3 relaxWorld = averageWorld - currentWorld;
        relaxWorld = relaxWorld - strokeNormal * relaxWorld.dot(strokeNormal);
        // Boundary vertices need stronger polish — their asymmetric topology
        // makes them prone to folding/inverting under clay deposits.
        relaxWorld *= safePolish * (isBoundary ? 0.8f : 1.0f);

        Vec3 relaxLocal = sanitizeVec3(inverseTransform.transform_vector(relaxWorld), Vec3(0.0f, 0.0f, 0.0f));
        const float deltaLen = relaxLocal.length();
        if (std::isfinite(deltaLen) && deltaLen > maxDelta) {
            relaxLocal *= (maxDelta / deltaLen);
        }

        updatedLocalPositions[vertexId] = sanitizeVec3(currentLocal + relaxLocal, currentLocal);
    };

    constexpr size_t kPolishParallelThreshold = 128u;
    if (touchedVertexIds.size() >= kPolishParallelThreshold) {
        std::for_each(std::execution::par_unseq, touchedVertexIds.begin(), touchedVertexIds.end(), processVertex);
    } else {
        std::for_each(touchedVertexIds.begin(), touchedVertexIds.end(), processVertex);
    }
}

// Removes the high-frequency height oscillation that discrete dab centers leave behind
// — the "stair-step / corduroy" ribbing perpendicular to the stroke that only shows up
// once the mesh is dense enough to resolve the dab pitch (coarse meshes low-pass it away
// geometrically, which is why it appears on high-tess meshes). A plain Laplacian toward
// the neighbour average attenuates HIGH curvature (the ribble) far more than the broad
// sculpted form, so a light pass flattens the ridges while leaving the intended shape.
//
// Unlike applyClayPolishPass (which deliberately STRIPS the stroke-normal component to
// preserve deposited height, and therefore cannot touch the ribble at all), this KEEPS
// the full vector — the ridge lives in the normal direction. Kept light + clamped so the
// broad form is barely affected. Local-space Laplacian (no world round-trip) for speed.
//
// ANISOTROPY (localStrokeTangent + tangentBias): additive brushes (Draw/Inflate) ribble
// isotropically (the dab-sum), so they pass a zero tangent = plain Laplacian. Clay instead
// TERRACES: each dab flattens its footprint to an instantaneous reference plane, so as the
// brush steps along its path it leaves stair-steps ALONG the stroke tangent. A plain
// Laplacian strong enough to erase those steps would also flatten clay's broad cross-section
// (defeating the brush). So when a tangent is given we weight each neighbour by how aligned
// its edge is with the stroke tangent: along-stroke neighbours (the terrace steps) smooth
// hard, across-stroke neighbours (the intended cross-section profile) are left alone. This
// is the same effect ClayStrips gets for free from its tine drag.
void applyDabRibbleSmoothPass(
    const SceneUI::EditableMeshCache& cache,
    std::vector<Vec3>& updatedLocalPositions,
    const std::vector<int>& touchedVertexIds,
    float localRadius,
    float strength,
    const Vec3& localStrokeTangent = Vec3(0.0f, 0.0f, 0.0f),
    float tangentBias = 0.0f,
    float maxDeltaFactor = 0.05f) {
    if (touchedVertexIds.empty() || updatedLocalPositions.empty()) {
        return;
    }
    const float safeStrength = sanitizeFiniteFloat(strength, 0.0f, 0.0f, 1.0f);
    if (safeStrength <= 1e-5f) {
        return;
    }

    const std::vector<Vec3> snapshot = updatedLocalPositions;
    const float maxDelta = (std::max)(localRadius * maxDeltaFactor, 1e-6f);
    const float safeBias = saturateFloat(tangentBias);
    const float tangentLenSq = localStrokeTangent.length_squared();
    const bool useTangent = (safeBias > 1e-4f) && (tangentLenSq > 1e-12f);
    const Vec3 tangentDir = useTangent
        ? localStrokeTangent * (1.0f / std::sqrt(tangentLenSq))
        : Vec3(0.0f, 0.0f, 0.0f);

    auto processVertex = [&](const int vertexIdInt) {
        if (vertexIdInt < 0 || vertexIdInt >= static_cast<int>(cache.vertex_positions.size())) {
            return;
        }
        const size_t vertexId = static_cast<size_t>(vertexIdInt);
        // Boundary verts are left alone — pulling them toward interior neighbours would
        // shrink the silhouette; the ribble there is masked by the open edge anyway.
        if (cache.vertex_is_boundary[vertexId] != 0) {
            return;
        }
        const auto& neighbors = cache.vertex_neighbors[vertexId];
        if (neighbors.empty()) {
            return;
        }
        const Vec3 currentLocal = snapshot[vertexId];
        Vec3 weightedSum(0.0f, 0.0f, 0.0f);
        float weightTotal = 0.0f;
        for (const int neighborId : neighbors) {
            if (neighborId < 0 || neighborId >= static_cast<int>(cache.vertex_positions.size())) {
                continue;
            }
            const Vec3 neighborLocal = snapshot[static_cast<size_t>(neighborId)];
            float w = 1.0f;
            if (useTangent) {
                const Vec3 edge = neighborLocal - currentLocal;
                const float el = edge.length();
                // Edges along the stroke tangent carry the terrace steps -> full weight;
                // edges across the stroke carry the cross-section -> down to (1-bias).
                const float align = (el > 1e-8f)
                    ? std::abs(edge.dot(tangentDir) / el)
                    : 0.0f;
                w = (1.0f - safeBias) + safeBias * align;
            }
            weightedSum += neighborLocal * w;
            weightTotal += w;
        }
        if (weightTotal <= 1e-8f) {
            return;
        }
        const Vec3 average = weightedSum / weightTotal;
        Vec3 relaxLocal = (average - currentLocal) * safeStrength;
        const float deltaLen = relaxLocal.length();
        if (std::isfinite(deltaLen) && deltaLen > maxDelta) {
            relaxLocal *= (maxDelta / deltaLen);
        }
        updatedLocalPositions[vertexId] = sanitizeVec3(currentLocal + relaxLocal, currentLocal);
    };

    constexpr size_t kRibbleParallelThreshold = 128u;
    if (touchedVertexIds.size() >= kRibbleParallelThreshold) {
        std::for_each(std::execution::par_unseq, touchedVertexIds.begin(), touchedVertexIds.end(), processVertex);
    } else {
        std::for_each(touchedVertexIds.begin(), touchedVertexIds.end(), processVertex);
    }
}

// Maps how finely the mesh is tessellated under the brush (verts across the radius) to a
// 0..1 ribbing risk. Dab ribble is invisible until the mesh can resolve the dab pitch, so
// the anti-ribble pass stays OFF on coarse meshes (no needless smoothing/cost) and ramps
// in as triangle density under the brush climbs.
float computeDabRibbleRisk(float localRadius, float avgEdgeLength) {
    const float safeRadius = sanitizeFiniteFloat(localRadius, 0.1f, 1e-5f, 1e6f);
    const float safeEdge = sanitizeFiniteFloat(avgEdgeLength, 0.05f, 1e-6f, 1e6f);
    const float vertsAcross = safeRadius / safeEdge;
    // Below ~6 verts across the radius the dab pitch isn't resolved; full risk by ~22.
    return std::clamp((vertsAcross - 6.0f) / 16.0f, 0.0f, 1.0f);
}

void applyClayAntiPitPass(
    const SceneUI::EditableMeshCache& cache,
    std::vector<Vec3>& updatedLocalPositions,
    const std::vector<int>& touchedVertexIds,
    const Matrix4x4& transform,
    const Matrix4x4& inverseTransform,
    const Vec3& strokeNormalWorld,
    float localRadius,
    float liftStrength,
    float flattenStrength) {
    if (touchedVertexIds.empty() || updatedLocalPositions.empty()) {
        return;
    }

    const Vec3 strokeNormal = safeNormalizeVec3(strokeNormalWorld, Vec3(0.0f, 1.0f, 0.0f));
    const float safeLift = sanitizeFiniteFloat(liftStrength, 0.0f, 0.0f, 1.0f);
    const float safeFlatten = sanitizeFiniteFloat(flattenStrength, 0.0f, 0.0f, 1.0f);
    if (safeLift <= 1e-5f && safeFlatten <= 1e-5f) {
        return;
    }

    const std::vector<Vec3> snapshot = updatedLocalPositions;
    const float maxLift = (std::max)(localRadius * 0.045f, 1e-5f);
    const float maxFlatten = (std::max)(localRadius * 0.025f, 1e-5f);

    auto processVertex = [&](const int vertexIdInt) {
        if (vertexIdInt < 0 || vertexIdInt >= static_cast<int>(cache.vertex_positions.size())) {
            return;
        }

        const size_t vertexId = static_cast<size_t>(vertexIdInt);
        const auto& neighbors = cache.vertex_neighbors[vertexId];
        if (neighbors.empty()) {
            return;
        }

        Vec3 averageLocal(0.0f, 0.0f, 0.0f);
        int neighborCount = 0;
        for (const int neighborId : neighbors) {
            if (neighborId < 0 || neighborId >= static_cast<int>(cache.vertex_positions.size())) {
                continue;
            }
            averageLocal += snapshot[static_cast<size_t>(neighborId)];
            ++neighborCount;
        }
        if (neighborCount == 0) {
            return;
        }
        averageLocal /= static_cast<float>(neighborCount);

        const Vec3 currentLocal = snapshot[vertexId];
        const Vec3 currentWorld = sanitizeVec3(transform.transform_point(currentLocal), currentLocal);
        const Vec3 averageWorld = sanitizeVec3(transform.transform_point(averageLocal), averageLocal);
        const Vec3 toAverageWorld = averageWorld - currentWorld;
        const float signedDepth = toAverageWorld.dot(strokeNormal);

        Vec3 correctedWorld(0.0f, 0.0f, 0.0f);
        if (signedDepth > 0.0f) {
            Vec3 liftWorld = strokeNormal * signedDepth * safeLift;
            const float liftLen = liftWorld.length();
            if (std::isfinite(liftLen) && liftLen > maxLift) {
                liftWorld *= (maxLift / liftLen);
            }
            correctedWorld += liftWorld;
        }

        Vec3 planarFlattenWorld = toAverageWorld - strokeNormal * toAverageWorld.dot(strokeNormal);
        planarFlattenWorld *= safeFlatten;
        const float flattenLen = planarFlattenWorld.length();
        if (std::isfinite(flattenLen) && flattenLen > maxFlatten) {
            planarFlattenWorld *= (maxFlatten / flattenLen);
        }
        correctedWorld += planarFlattenWorld;

        const Vec3 correctedLocal = sanitizeVec3(
            inverseTransform.transform_vector(correctedWorld),
            Vec3(0.0f, 0.0f, 0.0f));
        updatedLocalPositions[vertexId] = sanitizeVec3(currentLocal + correctedLocal, currentLocal);
    };

    constexpr size_t kAntiPitParallelThreshold = 128u;
    if (touchedVertexIds.size() >= kAntiPitParallelThreshold) {
        std::for_each(std::execution::par_unseq, touchedVertexIds.begin(), touchedVertexIds.end(), processVertex);
    } else {
        std::for_each(touchedVertexIds.begin(), touchedVertexIds.end(), processVertex);
    }
}





// Active custom falloff LUT, pointed at MeshOverlaySettings::custom_falloff_lut by
// the UI thread before each sculpt / soft-select solve. Read-only during the
// parallel solves (the vector is never resized while a solve runs), so a plain
// file-static pointer is safe and avoids threading a LUT arg through ~12 call sites.
static const std::vector<float>* g_activeFalloffLut = nullptr;
void setActiveFalloffLut(const std::vector<float>* lut) { g_activeFalloffLut = lut; }

float applyFalloffCurve(float t, int falloffType) {
    t = saturateFloat(t);
    switch (falloffType) {
    case 1: // Linear
        return t;
    case 2: // Sharp
        return t * t;
    case 3: // Sphere
        return std::sqrt((std::max)(0.0f, 1.0f - (1.0f - t) * (1.0f - t)));
    case 4: // Root
        return std::sqrt(t);
    case 5: { // Custom LUT (linear interpolation between samples)
        if (g_activeFalloffLut && g_activeFalloffLut->size() >= 2) {
            const std::vector<float>& lut = *g_activeFalloffLut;
            const float ft = t * static_cast<float>(lut.size() - 1);
            const int i0 = static_cast<int>(ft);
            const int i1 = (std::min)(i0 + 1, static_cast<int>(lut.size()) - 1);
            const float frac = ft - static_cast<float>(i0);
            return saturateFloat(lut[i0] * (1.0f - frac) + lut[i1] * frac);
        }
        return t * t * (3.0f - 2.0f * t); // fall back to Smooth until edited
    }
    case 0: // Smooth
    default:
        return t * t * (3.0f - 2.0f * t);
    }
}

float computeSoftSelectionWeight(float distance, float radius, float falloff, int falloffType) {
    if (radius <= 1e-6f) {
        return 0.0f;
    }
    const float t = applyFalloffCurve(1.0f - (distance / radius), falloffType);
    const float exponent = 0.35f + (1.0f - saturateFloat(falloff)) * 3.65f;
    return std::pow(t, exponent);
}

std::vector<float> buildSoftSelectionWeights(
    const SceneUI::EditableMeshCache& cache,
    const SceneUI::MeshOverlaySettings& settings,
    const std::unordered_set<int>& targetVertices) {
    std::vector<float> weights(cache.vertices.size(), 0.0f);
    if (cache.vertices.empty() || targetVertices.empty()) {
        return weights;
    }

    if (!settings.proportional_edit) {
        for (const int targetId : targetVertices) {
            if (targetId >= 0 && targetId < static_cast<int>(weights.size())) {
                weights[targetId] = 1.0f;
            }
        }
        return weights;
    }

    const float radius = (std::max)(0.001f, settings.proportional_radius);
    for (size_t i = 0; i < cache.vertices.size(); ++i) {
        float bestWeight = 0.0f;
        for (const int targetId : targetVertices) {
            if (targetId < 0 || targetId >= static_cast<int>(cache.vertices.size())) {
                continue;
            }

            const float distance = (cache.vertices[i].local_position -
                                    cache.vertices[targetId].local_position).length();
            bestWeight = (std::max)(
                bestWeight,
                computeSoftSelectionWeight(
                    distance, radius, settings.proportional_falloff, settings.proportional_falloff_type));
        }
        weights[i] = bestWeight;
    }
    return weights;
}

ImU32 weightToColor(float weight, int alphaBase = 220) {
    const float t = saturateFloat(weight);
    const int r = static_cast<int>(70.0f + 185.0f * t);
    const int g = static_cast<int>(220.0f - 45.0f * t);
    const int b = static_cast<int>(255.0f - 190.0f * t);
    const int a = static_cast<int>((std::max)(20.0f, alphaBase * t));
    return IM_COL32(r, g, b, a);
}

ImU32 scaleColorAlpha(ImU32 color, float alphaScale) {
    const float clampedScale = saturateFloat(alphaScale);
    const int a = static_cast<int>(static_cast<float>((color >> IM_COL32_A_SHIFT) & 0xFF) * clampedScale);
    return (color & 0x00FFFFFFu) | (static_cast<ImU32>(a) << IM_COL32_A_SHIFT);
}

float computeOverlayVisibility(const Camera& cam, const Vec3& p0, const Vec3& p1, const Vec3& p2) {
    const Vec3 normal = (p1 - p0).cross(p2 - p0);
    const float normalLen = normal.length();
    if (normalLen <= 1e-6f) {
        return 1.0f;
    }

    const Vec3 faceNormal = normal / normalLen;
    const Vec3 toCamera = (cam.lookfrom - ((p0 + p1 + p2) / 3.0f)).normalize();
    const float facing = faceNormal.dot(toCamera);
    return (facing >= 0.0f) ? 1.0f : 0.28f;
}

bool containsSelectionId(const std::vector<int>& ids, int id) {
    return std::find(ids.begin(), ids.end(), id) != ids.end();
}

void toggleSelectionId(std::vector<int>& ids, int id) {
    auto it = std::find(ids.begin(), ids.end(), id);
    if (it == ids.end()) {
        ids.push_back(id);
    } else {
        ids.erase(it);
    }
}

void replaceSelectionId(std::vector<int>& ids, int id) {
    ids.clear();
    if (id >= 0) {
        ids.push_back(id);
    }
}

float estimateBrushWorldRadius(const Camera& cam, const ImVec2& displaySize, const Vec3& worldPoint, float radiusPx) {
    if (displaySize.y <= 1.0f) {
        return 0.0f;
    }
    const Vec3 camForward = (cam.lookat - cam.lookfrom).normalize();
    const float depth = (worldPoint - cam.lookfrom).dot(camForward);
    if (depth <= 0.01f) {
        return 0.0f;
    }
    const float tanHalfFov = std::tan(cam.vfov * 3.14159265359f / 180.0f * 0.5f);
    return (2.0f * depth * tanHalfFov) * (radiusPx / displaySize.y);
}

float estimateSculptWorldRadius(const Triangle& tri, float radiusPx) {
    const Vec3 v0 = tri.getV0();
    const Vec3 v1 = tri.getV1();
    const Vec3 v2 = tri.getV2();
    const float avgEdge =
        ((v1 - v0).length() + (v2 - v1).length() + (v0 - v2).length()) / 3.0f;
    const float baseline = 96.0f;
    return (std::max)(0.001f, avgEdge * (radiusPx / baseline));
}

bool raycastEditableObjectTriangles(
    const std::vector<std::pair<int, std::shared_ptr<Triangle>>>& meshEntries,
    const Ray& ray,
    HitRecord& outHit) {
    bool hitAnything = false;
    float closest = 1e30f;
    HitRecord bestHit;
    for (const auto& entry : meshEntries) {
        if (!entry.second) {
            continue;
        }
        HitRecord rec;
        if (entry.second->hit(ray, 0.001f, closest, rec)) {
            closest = rec.t;
            bestHit = rec;
            bestHit.triangle = entry.second.get();
            hitAnything = true;
        }
    }
    if (hitAnything) {
        outHit = bestHit;
    }
    return hitAnything;
}

// Closest ray hit against the live sculpt PBVH, in object-LOCAL space.
//
// Why this exists: the CPU picking BVH refit is deferred to stroke end while a GPU
// viewport backend is active (a per-dab global Embree refit is O(N) — see
// processPendingMeshEditGpuSync), so mid-stroke raycastViewportHit reports the
// PRE-deformation surface. When a brush pushes a region far out, the picking ray keeps
// hitting the stale (original) surface and the brush "loses" the bulge. The PBVH, by
// contrast, has its leaf bounds refit every dab (refreshSculptPBVHLeavesAndAncestors),
// so a local-space traversal of it always sees the current surface for cheap
// (O(log N + local triangles)) — no global rebuild required.
//
// Local space: PBVH node bounds and Triangle::getOriginalVertexPosition() are both in
// object-local space and both kept current per dab, so we transform the ray to local,
// traverse, and hand the local hit back to the caller to map into world.
bool raycastSculptPBVHLocal(
    const SceneUI::SculptPBVH& pbvh,
    const SceneUI::EditableMeshCache& cache,
    const Vec3& o,
    const Vec3& d,
    Vec3& outLocalPoint,
    Vec3& outLocalNormal,
    Triangle*& outTriangle) {
    if (pbvh.root_node_id < 0 || pbvh.nodes.empty() || cache.vertices.empty()) {
        return false;
    }

    const Vec3 invD(
        std::fabs(d.x) > 1e-12f ? 1.0f / d.x : 0.0f,
        std::fabs(d.y) > 1e-12f ? 1.0f / d.y : 0.0f,
        std::fabs(d.z) > 1e-12f ? 1.0f / d.z : 0.0f);

    auto rayHitsAABB = [&](const AABB& box, float tmax) -> bool {
        float t0 = 0.0f, t1 = tmax;
        for (int a = 0; a < 3; ++a) {
            if (std::fabs(d[a]) < 1e-12f) {
                if (o[a] < box.min[a] || o[a] > box.max[a]) {
                    return false;
                }
                continue;
            }
            float tNear = (box.min[a] - o[a]) * invD[a];
            float tFar = (box.max[a] - o[a]) * invD[a];
            if (tNear > tFar) std::swap(tNear, tFar);
            if (tNear > t0) t0 = tNear;
            if (tFar < t1) t1 = tFar;
            if (t1 < t0) return false;
        }
        return true;
    };

    auto rayHitsTriangle = [&](const Vec3& a, const Vec3& b, const Vec3& c,
                               float& tOut, Vec3& nOut) -> bool {
        const Vec3 e1 = b - a;
        const Vec3 e2 = c - a;
        const Vec3 p = Vec3::cross(d, e2);
        const float det = e1.dot(p);
        if (std::fabs(det) < 1e-15f) return false;
        const float inv = 1.0f / det;
        const Vec3 tv = o - a;
        const float u = tv.dot(p) * inv;
        if (u < -1e-5f || u > 1.0f + 1e-5f) return false;
        const Vec3 q = Vec3::cross(tv, e1);
        const float v = d.dot(q) * inv;
        if (v < -1e-5f || u + v > 1.0f + 1e-5f) return false;
        const float t = e2.dot(q) * inv;
        if (t <= 1e-6f) return false;
        tOut = t;
        nOut = Vec3::cross(e1, e2);
        return true;
    };

    float closest = 1e30f;
    bool found = false;
    Vec3 bestNormal;
    Triangle* bestTriangle = nullptr;

    std::vector<int> stack;
    stack.reserve(64);
    stack.push_back(pbvh.root_node_id);
    while (!stack.empty()) {
        const int nodeId = stack.back();
        stack.pop_back();
        if (nodeId < 0 || nodeId >= static_cast<int>(pbvh.nodes.size())) {
            continue;
        }
        const auto& node = pbvh.nodes[static_cast<size_t>(nodeId)];
        if (!rayHitsAABB(node.local_bounds, closest)) {
            continue;
        }
        if (!node.is_leaf) {
            if (node.left_child_id >= 0) stack.push_back(node.left_child_id);
            if (node.right_child_id >= 0) stack.push_back(node.right_child_id);
            continue;
        }
        for (const int vid : node.vertex_ids) {
            if (vid < 0 || static_cast<size_t>(vid) >= cache.vertices.size()) {
                continue;
            }
            for (const auto& ref : cache.vertices[static_cast<size_t>(vid)].refs) {
                Triangle* tri = cache.refTri(ref);
                if (!tri) continue;
                float t;
                Vec3 n;
                if (rayHitsTriangle(
                        tri->getOriginalVertexPosition(0),
                        tri->getOriginalVertexPosition(1),
                        tri->getOriginalVertexPosition(2),
                        t, n) &&
                    t < closest) {
                    closest = t;
                    bestNormal = n;
                    bestTriangle = tri;
                    found = true;
                }
            }
        }
    }

    if (!found) {
        return false;
    }
    outLocalPoint = o + d * closest;
    outLocalNormal = bestNormal;
    outTriangle = bestTriangle;
    return true;
}

bool intersectRayPlane(const Ray& ray, const Vec3& planePoint, const Vec3& planeNormal, Vec3& outPoint) {
    if (!isFiniteVec3(ray.origin) || !isFiniteVec3(ray.direction) || !isFiniteVec3(planePoint)) {
        return false;
    }
    const Vec3 n = safeNormalizeVec3(planeNormal, Vec3(0, 1, 0));
    const float denom = ray.direction.dot(n);
    if (!std::isfinite(denom) || std::fabs(denom) <= 1e-6f) {
        return false;
    }
    const float t = (planePoint - ray.origin).dot(n) / denom;
    if (!std::isfinite(t) || t <= 0.0f) {
        return false;
    }
    outPoint = ray.origin + ray.direction * t;
    return isFiniteVec3(outPoint);
}

float computeTerrainLikeBrushWeight(float normalizedDistance, float falloff) {
    const float n = saturateFloat(normalizedDistance);
    const float inner = saturateFloat(1.0f - falloff);
    if (n <= inner) {
        return 1.0f;
    }
    const float outerT = saturateFloat((n - inner) / (std::max)(0.001f, 1.0f - inner));
    return 1.0f - (outerT * outerT * (3.0f - 2.0f * outerT));
}

SceneUI::EditableSpatialCellKey makeEditableSpatialCellKey(const Vec3& localPos, float cellSize) {
    const float safeCellSize = (std::max)(cellSize, 1e-5f);
    return SceneUI::EditableSpatialCellKey{
        static_cast<int>(std::floor(localPos.x / safeCellSize)),
        static_cast<int>(std::floor(localPos.y / safeCellSize)),
        static_cast<int>(std::floor(localPos.z / safeCellSize))
    };
}

float estimateEditableLocalBrushRadius(const Matrix4x4& inverseTransform, float radiusWorld) {
    const float safeRadiusWorld = sanitizeFiniteFloat(radiusWorld, 0.1f, 0.0001f, 1000.0f);
    const float sx = inverseTransform.transform_vector(Vec3(1.0f, 0.0f, 0.0f)).length();
    const float sy = inverseTransform.transform_vector(Vec3(0.0f, 1.0f, 0.0f)).length();
    const float sz = inverseTransform.transform_vector(Vec3(0.0f, 0.0f, 1.0f)).length();
    const float maxScale = (std::max)(sx, (std::max)(sy, sz));
    return safeRadiusWorld * (std::max)(maxScale, 1e-4f);
}

std::vector<int> collectEditableVertexCandidates(
    const SceneUI::EditableMeshCache& cache,
    const Vec3& localCenter,
    float localRadius) {
    std::vector<int> candidates;
    if (cache.vertices.empty()) {
        return candidates;
    }

    const float safeRadius = sanitizeFiniteFloat(localRadius, 0.1f, 0.0001f, 100000.0f);
    const float cellSize = sanitizeFiniteFloat(cache.spatial_cell_size, 0.0f, 0.0f, 100000.0f);
    if (cellSize <= 1e-6f || cache.vertex_spatial_buckets.empty()) {
        candidates.reserve(cache.vertices.size());
        for (size_t vertexId = 0; vertexId < cache.vertices.size(); ++vertexId) {
            candidates.push_back(static_cast<int>(vertexId));
        }
        return candidates;
    }

    const SceneUI::EditableSpatialCellKey centerKey = makeEditableSpatialCellKey(localCenter, cellSize);
    const int cellRadius = (std::max)(1, static_cast<int>(std::ceil(safeRadius / cellSize)));
    // Cap cell iteration to avoid O(N^3) loop when brush is much larger than cells.
    // If we'd iterate more than ~50^3 = 125K cells, just do a linear scan instead.
    if (cellRadius > 50) {
        const float radiusSq = safeRadius * safeRadius;
        candidates.reserve(cache.vertices.size() / 4 + 1);
        for (size_t vertexId = 0; vertexId < cache.vertices.size(); ++vertexId) {
            const Vec3 delta = cache.vertices[vertexId].local_position - localCenter;
            const float distSq = delta.length_squared();
            if (std::isfinite(distSq) && distSq <= radiusSq) {
                candidates.push_back(static_cast<int>(vertexId));
            }
        }
        return candidates;
    }
    const float radiusSq = safeRadius * safeRadius;
    candidates.reserve((cellRadius * 2 + 1) * (cellRadius * 2 + 1) * 4);

    for (int z = centerKey.z - cellRadius; z <= centerKey.z + cellRadius; ++z) {
        for (int y = centerKey.y - cellRadius; y <= centerKey.y + cellRadius; ++y) {
            for (int x = centerKey.x - cellRadius; x <= centerKey.x + cellRadius; ++x) {
                const auto bucketIt = cache.vertex_spatial_buckets.find(SceneUI::EditableSpatialCellKey{ x, y, z });
                if (bucketIt == cache.vertex_spatial_buckets.end()) {
                    continue;
                }
                for (const int vertexId : bucketIt->second) {
                    if (vertexId < 0 || vertexId >= static_cast<int>(cache.vertices.size())) {
                        continue;
                    }
                    const Vec3 delta = cache.vertices[vertexId].local_position - localCenter;
                    const float distSq = delta.length_squared();
                    if (std::isfinite(distSq) && distSq <= radiusSq) {
                        candidates.push_back(vertexId);
                    }
                }
            }
        }
    }

    if (candidates.empty()) {
        // Spatial grid found no vertices in brush radius — return empty.
        // The old fallback returned ALL vertices (catastrophic for 1M+ meshes).
    }

    return candidates;
}

std::vector<int> collectSculptControlNodeCandidates(
    const SceneUI::SculptControlGraph& graph,
    const Vec3& localCenter,
    float localRadius) {
    std::vector<int> candidates;
    if (graph.nodes.empty()) {
        return candidates;
    }

    const float safeRadius = sanitizeFiniteFloat(localRadius, 0.1f, 0.0001f, 100000.0f);
    const float cellSize = sanitizeFiniteFloat(graph.spatial_cell_size, 0.0f, 0.0f, 100000.0f);
    // A leaf's member vertices can sit up to half the cell diagonal away from the
    // leaf center. Pad the candidate radius so leaves that straddle the brush border
    // are still picked; otherwise triangles split across a leaf boundary lose
    // vertices on one side (one-corner-moves spikes for Grab, boundary gaps for Draw
    // on dense meshes).
    const float candidatePad = cellSize * 0.866f;
    const float candidateRadius = safeRadius + candidatePad;
    const float candidateRadiusSq = candidateRadius * candidateRadius;
    if (cellSize <= 1e-6f || graph.node_spatial_buckets.empty()) {
        candidates.reserve(graph.nodes.size());
        for (size_t nodeId = 0; nodeId < graph.nodes.size(); ++nodeId) {
            const Vec3 delta = graph.nodes[nodeId].local_position - localCenter;
            const float distSq = delta.length_squared();
            if (std::isfinite(distSq) && distSq <= candidateRadiusSq) {
                candidates.push_back(static_cast<int>(nodeId));
            }
        }
        return candidates;
    }

    const SceneUI::EditableSpatialCellKey centerKey = makeEditableSpatialCellKey(localCenter, cellSize);
    const int cellRadius = (std::max)(1, static_cast<int>(std::ceil(candidateRadius / cellSize)));
    if (cellRadius > 50) {
        candidates.reserve(graph.nodes.size() / 4 + 1);
        for (size_t nodeId = 0; nodeId < graph.nodes.size(); ++nodeId) {
            const Vec3 delta = graph.nodes[nodeId].local_position - localCenter;
            const float distSq = delta.length_squared();
            if (std::isfinite(distSq) && distSq <= candidateRadiusSq) {
                candidates.push_back(static_cast<int>(nodeId));
            }
        }
        return candidates;
    }

    candidates.reserve((cellRadius * 2 + 1) * (cellRadius * 2 + 1) * 4);
    for (int z = centerKey.z - cellRadius; z <= centerKey.z + cellRadius; ++z) {
        for (int y = centerKey.y - cellRadius; y <= centerKey.y + cellRadius; ++y) {
            for (int x = centerKey.x - cellRadius; x <= centerKey.x + cellRadius; ++x) {
                const auto bucketIt = graph.node_spatial_buckets.find(SceneUI::EditableSpatialCellKey{ x, y, z });
                if (bucketIt == graph.node_spatial_buckets.end()) {
                    continue;
                }
                for (const int nodeId : bucketIt->second) {
                    if (nodeId < 0 || nodeId >= static_cast<int>(graph.nodes.size())) {
                        continue;
                    }
                    const Vec3 delta = graph.nodes[nodeId].local_position - localCenter;
                    const float distSq = delta.length_squared();
                    if (std::isfinite(distSq) && distSq <= candidateRadiusSq) {
                        candidates.push_back(nodeId);
                    }
                }
            }
        }
    }

    return candidates;
}

std::vector<int> collectSourceVerticesForSculptNodes(
    const SceneUI::SculptControlGraph& graph,
    const std::vector<int>& nodeIds) {
    std::vector<int> sourceVertexIds;
    if (nodeIds.empty()) {
        return sourceVertexIds;
    }

    size_t reserveCount = 0;
    for (const int nodeId : nodeIds) {
        if (nodeId >= 0 && nodeId < static_cast<int>(graph.nodes.size())) {
            reserveCount += graph.nodes[nodeId].source_vertex_ids.size();
        }
    }
    sourceVertexIds.reserve(reserveCount);

    for (const int nodeId : nodeIds) {
        if (nodeId < 0 || nodeId >= static_cast<int>(graph.nodes.size())) {
            continue;
        }
        const auto& node = graph.nodes[nodeId];
        sourceVertexIds.insert(
            sourceVertexIds.end(),
            node.source_vertex_ids.begin(),
            node.source_vertex_ids.end());
    }

    std::sort(sourceVertexIds.begin(), sourceVertexIds.end());
    sourceVertexIds.erase(
        std::unique(sourceVertexIds.begin(), sourceVertexIds.end()),
        sourceVertexIds.end());
    return sourceVertexIds;
}

void beginSculptStroke(
    SceneUI::SculptStrokeState& strokeState,
    SceneUI::SculptBrushTool activeTool,
    const std::string& objectName,
    const HitRecord& hit,
    size_t editableVertexCount) {
    strokeState = SceneUI::SculptStrokeState{};
    strokeState.active = true;
    strokeState.object_name = objectName;
    strokeState.start_world_hit = hit.point;
    strokeState.last_world_hit = hit.point;
    strokeState.stroke_normal = hit.normal.length_squared() > 1e-8f
        ? hit.normal.normalize()
        : Vec3(0.0f, 1.0f, 0.0f);

    if (activeTool == SceneUI::SculptBrushTool::Grab) {
        strokeState.grab_start_local_positions.reserve(512);
        strokeState.grab_weights_by_vertex.reserve(512);
    }

    strokeState.before_triangle_states.reserve(512);
    strokeState.touched_triangles.reserve(512);

    if (activeTool == SceneUI::SculptBrushTool::Layer) {
        strokeState.layer_accum.assign(editableVertexCount, 0.0f);
    }
    if (activeTool == SceneUI::SculptBrushTool::Clay) {
        strokeState.clay_layer_accum.assign(editableVertexCount, 0.0f);
    }
    if (activeTool == SceneUI::SculptBrushTool::ClayStrips) {
        strokeState.clay_strips_layer_accum.assign(editableVertexCount, 0.0f);
    }
}

struct SculptCandidateSet {
    std::vector<int> node_ids;
    std::vector<int> source_vertex_ids;
};

SculptCandidateSet resolveSculptCandidates(
    const SceneUI::SculptControlGraph& graph,
    const Vec3& localCenter,
    float localRadius) {
    SculptCandidateSet result;
    result.node_ids = collectSculptControlNodeCandidates(graph, localCenter, localRadius);
    result.source_vertex_ids = collectSourceVerticesForSculptNodes(graph, result.node_ids);
    return result;
}

// Grab-family radial weight. Grab uses the user falloff curve (tight, rigid).
// Elastic uses a soft, wide shoulder (sqrt of smoothstep) so the whole region
// drags together like soft rubber — a broad, smooth pull instead of a focused one.
inline float grabFamilyWeight(float planarDistOverRadius, int falloffType, bool elastic) {
    const float t = saturateFloat(1.0f - saturateFloat(planarDistOverRadius));
    if (elastic) {
        const float s = t * t * (3.0f - 2.0f * t); // smoothstep
        return std::sqrt(s);                        // wider shoulder, 0 at the rim
    }
    return applyFalloffCurve(t, falloffType);
}

void primeGrabStrokeWeights(
    SceneUI::SculptStrokeState& strokeState,
    const SceneUI::EditableMeshCache& editableMeshCache,
    const Matrix4x4& transform,
    const std::vector<int>& candidateVertexIds,
    const Vec3& brushCenter,
    const Vec3& hitNormalWorld,
    float radiusWorld,
    bool frontFacesOnly,
    int falloffType,
    bool elastic) {
    if (!strokeState.grab_weights_by_vertex.empty()) {
        return;
    }

    strokeState.grab_start_local_positions.reserve(candidateVertexIds.size());
    strokeState.grab_weights_by_vertex.reserve(candidateVertexIds.size());
    for (const int vertexIdInt : candidateVertexIds) {
        if (vertexIdInt < 0) {
            continue;
        }
        const size_t vertexId = static_cast<size_t>(vertexIdInt);
        if (vertexId >= editableMeshCache.vertices.size()) {
            continue;
        }

        const Vec3 startLocalPos = editableMeshCache.vertices[vertexId].local_position;
        strokeState.grab_start_local_positions.emplace(vertexIdInt, startLocalPos);
        const Vec3 startWorldPos = sanitizeVec3(
            transform.transform_point(startLocalPos),
            brushCenter);
        const Vec3 toVertex = startWorldPos - brushCenter;
        const float heightFromPlane = toVertex.dot(hitNormalWorld);
        const Vec3 planarOffset = toVertex - hitNormalWorld * heightFromPlane;
        const float planarDistance = planarOffset.length();
        if (!std::isfinite(heightFromPlane) || !std::isfinite(planarDistance) || planarDistance > radiusWorld) {
            continue;
        }
        if (frontFacesOnly && heightFromPlane < -(radiusWorld * 0.12f)) {
            continue;
        }

        const float weight = grabFamilyWeight(planarDistance / radiusWorld, falloffType, elastic);
        strokeState.grab_weights_by_vertex[vertexIdInt] = weight;
    }
}
float getSculptNodeVertexInfluence(
    const SceneUI::SculptControlNode& node,
    int sourceVertexId) {
    for (size_t localIndex = 0; localIndex < node.source_vertex_ids.size(); ++localIndex) {
        if (node.source_vertex_ids[localIndex] == sourceVertexId) {
            if (localIndex < node.source_weights.size() && std::isfinite(node.source_weights[localIndex])) {
                return node.source_weights[localIndex];
            }
            return 1.0f;
        }
    }
    return 1.0f;
}
bool applyGrabSculptCandidate(
    const SceneUI::SculptStrokeState& strokeState,
    float brushStrength,
    const Vec3& localGrabDelta,
    int vertexIdInt,
    Vec3& inOutLocalPosition) {
    if (vertexIdInt < 0) {
        return false;
    }

    const auto weightIt = strokeState.grab_weights_by_vertex.find(vertexIdInt);
    const auto startIt = strokeState.grab_start_local_positions.find(vertexIdInt);
    const float weight = weightIt != strokeState.grab_weights_by_vertex.end()
        ? weightIt->second
        : 0.0f;
    if (weight <= 1e-5f || startIt == strokeState.grab_start_local_positions.end()) {
        return false;
    }

    const Vec3& startLocalPos = startIt->second;
    // Grab-family follow ratio is FIXED (kSculptGrabFollowStrength), decoupled from the
    // brush Strength slider on purpose: displacement = cursorDelta * weight * follow.
    // A constant follow keeps the per-sub-step displacement (and thus triangle rotation)
    // bounded no matter how fast the cursor moves or how high Strength was set, so the
    // topology-safety guard stops dropping/freezing boundary verts on fast strokes.
    (void)brushStrength; // slider intentionally ignored for grab
    const float grabStrength = kSculptGrabFollowStrength;
    Vec3 localDelta = localGrabDelta * weight * grabStrength;
    const float localDeltaLenSq = localDelta.length_squared();
    if (!std::isfinite(localDeltaLenSq) || localDeltaLenSq <= 1e-14f) {
        return false;
    }

    const float localDeltaLen = std::sqrt(localDeltaLenSq);
    const float baseGrabDeltaLen = localGrabDelta.length();
    const float maxLocalDelta = (std::max)(baseGrabDeltaLen * 3.0f, 0.05f);
    if (std::isfinite(localDeltaLen) && localDeltaLen > maxLocalDelta) {
        localDelta *= (maxLocalDelta / localDeltaLen);
    }

    inOutLocalPosition = sanitizeVec3(startLocalPos + localDelta, startLocalPos);
    return true;
}

struct SculptBrushSample {
    Vec3 local_position;
    Vec3 world_position;
    Vec3 planar_offset;
    float height_from_plane = 0.0f;
    float planar_distance = 0.0f;
    float weight = 0.0f;
    float clay_drag_factor = 1.0f;
};

struct SIMDSculptParams {
    Vec3 center;
    Vec3 prevCenter;
    Vec3 normal;
    Vec3 tangent;
    Vec3 bitangent;
    float radius;
    float cullRadius = 0.0f; // geometric inclusion radius (≥ radius for shaped brushes); 0 ⇒ use radius
    float falloff;
    float strokeDistance;
    float strokeSpacing;
    float dt;
    bool strokeChanged;
    bool frontFacesOnly;
    int falloffType;
    SceneUI::SculptBrushTool activeTool;
    bool useMask;
    Paint::BrushSettings brushSettings;
};

struct EvaluatedSculptVertex {
    int id;
    Vec3 worldPos;
    float height;
    float planarDist;
    float weight;
    float clayDragFactor;
};

inline bool canUseFastSIMDSculpt(const Paint::BrushSettings& brush) {
    return !brush.use_imported_alpha && 
           (brush.alpha_preset == Paint::BrushAlphaPreset::SoftRound || 
            brush.alpha_preset == Paint::BrushAlphaPreset::HardRound) &&
           brush.shape == Paint::BrushShape::Circle;
}

std::vector<EvaluatedSculptVertex> evaluateActiveSculptVerticesSIMD(
    const std::vector<int>& candidateIds,
    const SceneUI::EditableMeshCache& cache,
    const Matrix4x4& transform,
    const SIMDSculptParams& params) {
    
    std::vector<EvaluatedSculptVertex> activeVerts;
    if (candidateIds.empty() || cache.vertex_positions.empty()) {
        return activeVerts;
    }
    
    const size_t numCandidates = candidateIds.size();
    const size_t sculptVertexCount = cache.vertex_positions.size();
    activeVerts.reserve(numCandidates);

    // Setup SIMD constants for matrix
    __m256 m00 = _mm256_set1_ps(transform.m[0][0]);
    __m256 m01 = _mm256_set1_ps(transform.m[0][1]);
    __m256 m02 = _mm256_set1_ps(transform.m[0][2]);
    __m256 m03 = _mm256_set1_ps(transform.m[0][3]);

    __m256 m10 = _mm256_set1_ps(transform.m[1][0]);
    __m256 m11 = _mm256_set1_ps(transform.m[1][1]);
    __m256 m12 = _mm256_set1_ps(transform.m[1][2]);
    __m256 m13 = _mm256_set1_ps(transform.m[1][3]);

    __m256 m20 = _mm256_set1_ps(transform.m[2][0]);
    __m256 m21 = _mm256_set1_ps(transform.m[2][1]);
    __m256 m22 = _mm256_set1_ps(transform.m[2][2]);
    __m256 m23 = _mm256_set1_ps(transform.m[2][3]);

    // Segment direction for capsule sweep
    Vec3 segDir = params.center - params.prevCenter;
    float segLenSq = segDir.dot(segDir);
    __m256 segDir_x = _mm256_set1_ps(segDir.x);
    __m256 segDir_y = _mm256_set1_ps(segDir.y);
    __m256 segDir_z = _mm256_set1_ps(segDir.z);
    __m256 segLenSq_vec = _mm256_set1_ps(segLenSq);

    __m256 prevCenter_x = _mm256_set1_ps(params.prevCenter.x);
    __m256 prevCenter_y = _mm256_set1_ps(params.prevCenter.y);
    __m256 prevCenter_z = _mm256_set1_ps(params.prevCenter.z);

    __m256 center_x = _mm256_set1_ps(params.center.x);
    __m256 center_y = _mm256_set1_ps(params.center.y);
    __m256 center_z = _mm256_set1_ps(params.center.z);

    __m256 hn_x = _mm256_set1_ps(params.normal.x);
    __m256 hn_y = _mm256_set1_ps(params.normal.y);
    __m256 hn_z = _mm256_set1_ps(params.normal.z);

    __m256 radius_vec = _mm256_set1_ps(params.radius);
    // Geometric inclusion radius — widened for non-circular/aspect shapes so their corners
    // (e.g. a square's, at sqrt(2)·radius) survive the cull and reach the per-vertex footprint
    // test in the scalar tail. Falloff/UV normalization still use the nominal params.radius.
    __m256 cull_vec = _mm256_set1_ps(params.cullRadius > params.radius ? params.cullRadius : params.radius);

    // Repeated stroke damping factor
    float damp_factor = computeRepeatedStrokeDamping(
        params.strokeDistance, params.radius, params.dt, params.strokeChanged);
    __m256 damp_vec = _mm256_set1_ps(damp_factor);

    bool fastSIMD = canUseFastSIMDSculpt(params.brushSettings) && params.useMask;

    // Pull constant calculations outside the loop
    __m256 inner_weight = _mm256_set1_ps(std::clamp(1.0f - params.falloff, 0.0f, 1.0f));
    __m256 denom_weight = _mm256_set1_ps(std::max(0.001f, 1.0f - std::clamp(1.0f - params.falloff, 0.0f, 1.0f)));

    __m256 softRejectStart = _mm256_setzero_ps();
    __m256 softRejectEnd = _mm256_setzero_ps();
    __m256 denom_penalty = _mm256_set1_ps(1.0f);
    if (params.frontFacesOnly) {
        float safeRadius = std::max(params.radius, 0.0001f);
        float blend_val = std::clamp((params.radius - 1.2f) / 2.8f, 0.0f, 1.0f);
        softRejectStart = _mm256_set1_ps(safeRadius * -0.22f);
        softRejectEnd = _mm256_set1_ps(safeRadius * -(0.50f + 0.20f * blend_val));
        denom_penalty = _mm256_set1_ps(std::max(1e-6f, (safeRadius * -0.22f) - (safeRadius * -(0.50f + 0.20f * blend_val))));
    }

    __m256 tangent_x = _mm256_set1_ps(params.tangent.x);
    __m256 tangent_y = _mm256_set1_ps(params.tangent.y);
    __m256 tangent_z = _mm256_set1_ps(params.tangent.z);

    const size_t numBlocks = (numCandidates + 7) / 8;

    if (numCandidates < 800) {
        // --- 1. SEQUENTIAL PATH (NO THREAD OVERHEAD) ---
        for (int blockIdx = 0; blockIdx < (int)numBlocks; ++blockIdx) {
            size_t startIdx = blockIdx * 8;
            size_t endIdx = (std::min)(startIdx + 8, numCandidates);
            size_t count = endIdx - startIdx;

            alignas(32) int vids[8];
            if (startIdx + 8 <= numCandidates) {
                // Hızlı yol: 8 indeksi doğrudan hizasız olarak yükle ve SIMD ile sınırla
                __m256i vids_vec = _mm256_loadu_si256((const __m256i*)&candidateIds[startIdx]);
                __m256i max_val = _mm256_set1_epi32(static_cast<int>(sculptVertexCount - 1));
                vids_vec = _mm256_max_epi32(_mm256_setzero_si256(), _mm256_min_epi32(vids_vec, max_val));
                _mm256_store_si256((__m256i*)vids, vids_vec);
            } else {
                // Sınır bloğu: Güvenli dolgu yap
                int fallbackVid = (numCandidates > 0) ? candidateIds[0] : 0;
                if (fallbackVid < 0 || fallbackVid >= (int)sculptVertexCount) fallbackVid = 0;
                for (size_t k = 0; k < 8; ++k) {
                    if (startIdx + k < numCandidates) {
                        int candidateVid = candidateIds[startIdx + k];
                        vids[k] = (candidateVid >= 0 && static_cast<size_t>(candidateVid) < sculptVertexCount) ? candidateVid : fallbackVid;
                    } else {
                        vids[k] = fallbackVid;
                    }
                }
            }

            // İndeksleri yükle ve x3 offset'leri hesapla
            __m256i vids_vec = _mm256_load_si256((const __m256i*)vids);
            __m256i vids_x2 = _mm256_slli_epi32(vids_vec, 1);
            __m256i vidx = _mm256_add_epi32(vids_x2, vids_vec);
            __m256i vidy = _mm256_add_epi32(vidx, _mm256_set1_epi32(1));
            __m256i vidz = _mm256_add_epi32(vidx, _mm256_set1_epi32(2));

            // Donanımsal Gather
            const float* base_ptr = (const float*)cache.vertex_positions.data();
            __m256 lx = _mm256_i32gather_ps(base_ptr, vidx, 4);
            __m256 ly = _mm256_i32gather_ps(base_ptr, vidy, 4);
            __m256 lz = _mm256_i32gather_ps(base_ptr, vidz, 4);

            __m256 wx = _mm256_add_ps(_mm256_mul_ps(m00, lx), _mm256_add_ps(_mm256_mul_ps(m01, ly), _mm256_add_ps(_mm256_mul_ps(m02, lz), m03)));
            __m256 wy = _mm256_add_ps(_mm256_mul_ps(m10, lx), _mm256_add_ps(_mm256_mul_ps(m11, ly), _mm256_add_ps(_mm256_mul_ps(m12, lz), m13)));
            __m256 wz = _mm256_add_ps(_mm256_mul_ps(m20, lx), _mm256_add_ps(_mm256_mul_ps(m21, ly), _mm256_add_ps(_mm256_mul_ps(m22, lz), m23)));

            __m256 closest_x, closest_y, closest_z;
            if (segLenSq > 1e-10f) {
                __m256 dx = _mm256_sub_ps(wx, prevCenter_x);
                __m256 dy = _mm256_sub_ps(wy, prevCenter_y);
                __m256 dz = _mm256_sub_ps(wz, prevCenter_z);

                __m256 t = _mm256_add_ps(_mm256_mul_ps(dx, segDir_x), _mm256_add_ps(_mm256_mul_ps(dy, segDir_y), _mm256_mul_ps(dz, segDir_z)));
                t = _mm256_div_ps(t, segLenSq_vec);
                t = _mm256_max_ps(_mm256_setzero_ps(), _mm256_min_ps(_mm256_set1_ps(1.0f), t));

                closest_x = _mm256_add_ps(_mm256_mul_ps(segDir_x, t), prevCenter_x);
                closest_y = _mm256_add_ps(_mm256_mul_ps(segDir_y, t), prevCenter_y);
                closest_z = _mm256_add_ps(_mm256_mul_ps(segDir_z, t), prevCenter_z);
            } else {
                closest_x = center_x;
                closest_y = center_y;
                closest_z = center_z;
            }

            __m256 to_x = _mm256_sub_ps(wx, closest_x);
            __m256 to_y = _mm256_sub_ps(wy, closest_y);
            __m256 to_z = _mm256_sub_ps(wz, closest_z);

            __m256 height_from_plane = _mm256_add_ps(_mm256_mul_ps(to_x, hn_x), _mm256_add_ps(_mm256_mul_ps(to_y, hn_y), _mm256_mul_ps(to_z, hn_z)));

            __m256 pox = _mm256_sub_ps(to_x, _mm256_mul_ps(hn_x, height_from_plane));
            __m256 poy = _mm256_sub_ps(to_y, _mm256_mul_ps(hn_y, height_from_plane));
            __m256 poz = _mm256_sub_ps(to_z, _mm256_mul_ps(hn_z, height_from_plane));

            __m256 planarDistanceSq = _mm256_add_ps(_mm256_mul_ps(pox, pox), _mm256_add_ps(_mm256_mul_ps(poy, poy), _mm256_mul_ps(poz, poz)));
            __m256 planarDistance = _mm256_sqrt_ps(planarDistanceSq);

            __m256 abs_height = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), height_from_plane);
            __m256 heightOver = _mm256_sub_ps(abs_height, _mm256_set1_ps(params.radius * 0.5f));
            heightOver = _mm256_max_ps(_mm256_setzero_ps(), heightOver);
            __m256 planar_distance = _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(heightOver, heightOver), planarDistanceSq));

            __m256 in_radius_mask = _mm256_cmp_ps(planar_distance, cull_vec, _CMP_LE_OQ);

            __m256 penalty = _mm256_set1_ps(1.0f);
            if (params.frontFacesOnly) {
                __m256 t_penalty = _mm256_div_ps(_mm256_sub_ps(height_from_plane, softRejectEnd), denom_penalty);
                t_penalty = _mm256_max_ps(_mm256_setzero_ps(), _mm256_min_ps(_mm256_set1_ps(1.0f), t_penalty));

                __m256 term1_penalty = _mm256_sub_ps(_mm256_set1_ps(3.0f), _mm256_mul_ps(_mm256_set1_ps(2.0f), t_penalty));
                penalty = _mm256_mul_ps(t_penalty, _mm256_mul_ps(t_penalty, term1_penalty));

                __m256 gt_start = _mm256_cmp_ps(height_from_plane, softRejectStart, _CMP_GE_OQ);
                __m256 lt_end = _mm256_cmp_ps(height_from_plane, softRejectEnd, _CMP_LE_OQ);
                penalty = _mm256_blendv_ps(penalty, _mm256_set1_ps(1.0f), gt_start);
                penalty = _mm256_blendv_ps(penalty, _mm256_setzero_ps(), lt_end);
            }

            __m256 normalizedDistance = _mm256_div_ps(planar_distance, radius_vec);
            __m256 n_weight = _mm256_max_ps(_mm256_setzero_ps(), _mm256_min_ps(_mm256_set1_ps(1.0f), normalizedDistance));

            __m256 outerT = _mm256_div_ps(_mm256_sub_ps(n_weight, inner_weight), denom_weight);
            outerT = _mm256_max_ps(_mm256_setzero_ps(), _mm256_min_ps(_mm256_set1_ps(1.0f), outerT));

            __m256 term1_weight = _mm256_sub_ps(_mm256_set1_ps(3.0f), _mm256_mul_ps(_mm256_set1_ps(2.0f), outerT));
            __m256 weight_base = _mm256_sub_ps(_mm256_set1_ps(1.0f), _mm256_mul_ps(outerT, _mm256_mul_ps(outerT, term1_weight)));

            __m256 le_inner = _mm256_cmp_ps(n_weight, inner_weight, _CMP_LE_OQ);
            weight_base = _mm256_blendv_ps(weight_base, _mm256_set1_ps(1.0f), le_inner);

            __m256 t_falloff = _mm256_sub_ps(_mm256_set1_ps(1.0f), n_weight);
            t_falloff = _mm256_max_ps(_mm256_setzero_ps(), _mm256_min_ps(_mm256_set1_ps(1.0f), t_falloff));

            __m256 falloff_val;
            switch (params.falloffType) {
                case 1: falloff_val = t_falloff; break;
                case 2: falloff_val = _mm256_mul_ps(t_falloff, t_falloff); break;
                case 3:
                    {
                        __m256 one_minus_t = _mm256_sub_ps(_mm256_set1_ps(1.0f), t_falloff);
                        __m256 term = _mm256_sub_ps(_mm256_set1_ps(1.0f), _mm256_mul_ps(one_minus_t, one_minus_t));
                        falloff_val = _mm256_sqrt_ps(_mm256_max_ps(_mm256_setzero_ps(), term));
                    }
                    break;
                case 4: falloff_val = _mm256_sqrt_ps(t_falloff); break;
                case 5:
                case 0:
                default:
                    {
                        __m256 term = _mm256_sub_ps(_mm256_set1_ps(3.0f), _mm256_mul_ps(_mm256_set1_ps(2.0f), t_falloff));
                        falloff_val = _mm256_mul_ps(t_falloff, _mm256_mul_ps(t_falloff, term));
                    }
                    break;
            }

            __m256 weight = _mm256_mul_ps(weight_base, falloff_val);
            weight = _mm256_mul_ps(weight, penalty);
            weight = _mm256_mul_ps(weight, damp_vec);

            if (fastSIMD) {
                __m256 radial = _mm256_sub_ps(_mm256_set1_ps(1.0f), normalizedDistance);
                radial = _mm256_max_ps(_mm256_setzero_ps(), _mm256_min_ps(_mm256_set1_ps(1.0f), radial));

                __m256 mask_val;
                if (params.brushSettings.alpha_preset == Paint::BrushAlphaPreset::HardRound) {
                    __m256 gt = _mm256_cmp_ps(radial, _mm256_set1_ps(0.2f), _CMP_GT_OQ);
                    mask_val = _mm256_blendv_ps(_mm256_setzero_ps(), _mm256_set1_ps(1.0f), gt);
                } else {
                    mask_val = radial;
                }
                weight = _mm256_mul_ps(weight, mask_val);
            }

            weight = _mm256_blendv_ps(_mm256_setzero_ps(), weight, in_radius_mask);

            // Aktif şerit (active lane) kontrolü ve erken atlama
            __m256 active_mask = _mm256_cmp_ps(weight, _mm256_set1_ps(1e-5f), _CMP_GT_OQ);
            int moveMask = _mm256_movemask_ps(active_mask);
            int activeBits = moveMask & ((1 << count) - 1);
            if (activeBits == 0) {
                continue; // Blokta fırça etki alanına giren aktif vertex yok, doğrudan atla
            }

            __m256 clay_drag_factor = _mm256_set1_ps(1.0f);
            if ((params.activeTool == SceneUI::SculptBrushTool::Clay || params.activeTool == SceneUI::SculptBrushTool::ClayStrips) &&
                params.strokeDistance > params.strokeSpacing * 0.15f) {
                __m256 tangentNorm = _mm256_add_ps(_mm256_mul_ps(pox, tangent_x), _mm256_add_ps(_mm256_mul_ps(poy, tangent_y), _mm256_mul_ps(poz, tangent_z)));
                tangentNorm = _mm256_div_ps(tangentNorm, radius_vec);

                clay_drag_factor = _mm256_add_ps(_mm256_mul_ps(tangentNorm, _mm256_set1_ps(1.1f)), _mm256_set1_ps(0.5f));
                clay_drag_factor = _mm256_max_ps(_mm256_setzero_ps(), _mm256_min_ps(_mm256_set1_ps(1.6f), clay_drag_factor));
            }

            alignas(32) float w_res[8];
            alignas(32) float h_res[8];
            alignas(32) float pd_res[8];
            alignas(32) float cdf_res[8];
            alignas(32) float wx_res[8];
            alignas(32) float wy_res[8];
            alignas(32) float wz_res[8];
            alignas(32) float pox_res[8];
            alignas(32) float poy_res[8];
            alignas(32) float poz_res[8];
            alignas(32) float pen_res[8];

            _mm256_store_ps(w_res, weight);
            _mm256_store_ps(h_res, height_from_plane);
            _mm256_store_ps(pd_res, planar_distance);
            _mm256_store_ps(cdf_res, clay_drag_factor);
            _mm256_store_ps(wx_res, wx);
            _mm256_store_ps(wy_res, wy);
            _mm256_store_ps(wz_res, wz);
            _mm256_store_ps(pox_res, pox);
            _mm256_store_ps(poy_res, poy);
            _mm256_store_ps(poz_res, poz);
            _mm256_store_ps(pen_res, penalty);

            for (size_t k = 0; k < count; ++k) {
                if (!(activeBits & (1 << k))) {
                    continue;
                }
                int vid = vids[k];

                float final_w = w_res[k];
                if (params.useMask && !fastSIMD) {
                    float nxAlpha = (pox_res[k] * params.tangent.x + poy_res[k] * params.tangent.y + poz_res[k] * params.tangent.z) / params.radius;
                    float nyAlpha = (pox_res[k] * params.bitangent.x + poy_res[k] * params.bitangent.y + poz_res[k] * params.bitangent.z) / params.radius;
                    // Shaped/aspect/alpha brush: REPLACE the circular radial profile
                    // (weight_base × falloff_val, baked into w_res) with the footprint weight so
                    // the deform matches the Alpha Preview. Front-face penalty + repeated-stroke
                    // damping are preserved (penalty × damp_factor). Circle + round preset takes
                    // the fastSIMD path above and never reaches here, so it is unchanged.
                    final_w = pen_res[k] * damp_factor *
                              uiSampleBrushFootprintWeight(params.brushSettings, nxAlpha, nyAlpha);
                }

                if (std::isfinite(final_w) && final_w > 1e-5f &&
                    std::isfinite(pd_res[k]) && std::isfinite(h_res[k])) {
                    EvaluatedSculptVertex ev;
                    ev.id = vid;
                    ev.worldPos = Vec3(wx_res[k], wy_res[k], wz_res[k]);
                    ev.height = h_res[k];
                    ev.planarDist = pd_res[k];
                    ev.weight = final_w;
                    ev.clayDragFactor = cdf_res[k];
                    activeVerts.push_back(ev);
                }
            }
        }
    } else {
        // --- 2. PARALLEL LOCK-FREE PATH (ZERO LOCK CONTENTION) ---
        std::vector<EvaluatedSculptVertex> results(numCandidates);
        std::vector<uint8_t> activeFlags(numCandidates, 0);

        #pragma omp parallel for
        for (int blockIdx = 0; blockIdx < (int)numBlocks; ++blockIdx) {
            size_t startIdx = blockIdx * 8;
            size_t endIdx = (std::min)(startIdx + 8, numCandidates);
            size_t count = endIdx - startIdx;

            alignas(32) int vids[8];
            if (startIdx + 8 <= numCandidates) {
                // Hızlı yol: 8 indeksi doğrudan hizasız olarak yükle ve SIMD ile sınırla
                __m256i vids_vec = _mm256_loadu_si256((const __m256i*)&candidateIds[startIdx]);
                __m256i max_val = _mm256_set1_epi32(static_cast<int>(sculptVertexCount - 1));
                vids_vec = _mm256_max_epi32(_mm256_setzero_si256(), _mm256_min_epi32(vids_vec, max_val));
                _mm256_store_si256((__m256i*)vids, vids_vec);
            } else {
                // Sınır bloğu: Güvenli dolgu yap
                int fallbackVid = (numCandidates > 0) ? candidateIds[0] : 0;
                if (fallbackVid < 0 || fallbackVid >= (int)sculptVertexCount) fallbackVid = 0;
                for (size_t k = 0; k < 8; ++k) {
                    if (startIdx + k < numCandidates) {
                        int candidateVid = candidateIds[startIdx + k];
                        vids[k] = (candidateVid >= 0 && static_cast<size_t>(candidateVid) < sculptVertexCount) ? candidateVid : fallbackVid;
                    } else {
                        vids[k] = fallbackVid;
                    }
                }
            }

            // İndeksleri yükle ve x3 offset'leri hesapla
            __m256i vids_vec = _mm256_load_si256((const __m256i*)vids);
            __m256i vids_x2 = _mm256_slli_epi32(vids_vec, 1);
            __m256i vidx = _mm256_add_epi32(vids_x2, vids_vec);
            __m256i vidy = _mm256_add_epi32(vidx, _mm256_set1_epi32(1));
            __m256i vidz = _mm256_add_epi32(vidx, _mm256_set1_epi32(2));

            // Donanımsal Gather
            const float* base_ptr = (const float*)cache.vertex_positions.data();
            __m256 lx = _mm256_i32gather_ps(base_ptr, vidx, 4);
            __m256 ly = _mm256_i32gather_ps(base_ptr, vidy, 4);
            __m256 lz = _mm256_i32gather_ps(base_ptr, vidz, 4);

            __m256 wx = _mm256_add_ps(_mm256_mul_ps(m00, lx), _mm256_add_ps(_mm256_mul_ps(m01, ly), _mm256_add_ps(_mm256_mul_ps(m02, lz), m03)));
            __m256 wy = _mm256_add_ps(_mm256_mul_ps(m10, lx), _mm256_add_ps(_mm256_mul_ps(m11, ly), _mm256_add_ps(_mm256_mul_ps(m12, lz), m13)));
            __m256 wz = _mm256_add_ps(_mm256_mul_ps(m20, lx), _mm256_add_ps(_mm256_mul_ps(m21, ly), _mm256_add_ps(_mm256_mul_ps(m22, lz), m23)));

            __m256 closest_x, closest_y, closest_z;
            if (segLenSq > 1e-10f) {
                __m256 dx = _mm256_sub_ps(wx, prevCenter_x);
                __m256 dy = _mm256_sub_ps(wy, prevCenter_y);
                __m256 dz = _mm256_sub_ps(wz, prevCenter_z);

                __m256 t = _mm256_add_ps(_mm256_mul_ps(dx, segDir_x), _mm256_add_ps(_mm256_mul_ps(dy, segDir_y), _mm256_mul_ps(dz, segDir_z)));
                t = _mm256_div_ps(t, segLenSq_vec);
                t = _mm256_max_ps(_mm256_setzero_ps(), _mm256_min_ps(_mm256_set1_ps(1.0f), t));

                closest_x = _mm256_add_ps(_mm256_mul_ps(segDir_x, t), prevCenter_x);
                closest_y = _mm256_add_ps(_mm256_mul_ps(segDir_y, t), prevCenter_y);
                closest_z = _mm256_add_ps(_mm256_mul_ps(segDir_z, t), prevCenter_z);
            } else {
                closest_x = center_x;
                closest_y = center_y;
                closest_z = center_z;
            }

            __m256 to_x = _mm256_sub_ps(wx, closest_x);
            __m256 to_y = _mm256_sub_ps(wy, closest_y);
            __m256 to_z = _mm256_sub_ps(wz, closest_z);

            __m256 height_from_plane = _mm256_add_ps(_mm256_mul_ps(to_x, hn_x), _mm256_add_ps(_mm256_mul_ps(to_y, hn_y), _mm256_mul_ps(to_z, hn_z)));

            __m256 pox = _mm256_sub_ps(to_x, _mm256_mul_ps(hn_x, height_from_plane));
            __m256 poy = _mm256_sub_ps(to_y, _mm256_mul_ps(hn_y, height_from_plane));
            __m256 poz = _mm256_sub_ps(to_z, _mm256_mul_ps(hn_z, height_from_plane));

            __m256 planarDistanceSq = _mm256_add_ps(_mm256_mul_ps(pox, pox), _mm256_add_ps(_mm256_mul_ps(poy, poy), _mm256_mul_ps(poz, poz)));
            __m256 planarDistance = _mm256_sqrt_ps(planarDistanceSq);

            __m256 abs_height = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), height_from_plane);
            __m256 heightOver = _mm256_sub_ps(abs_height, _mm256_set1_ps(params.radius * 0.5f));
            heightOver = _mm256_max_ps(_mm256_setzero_ps(), heightOver);
            __m256 planar_distance = _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(heightOver, heightOver), planarDistanceSq));

            __m256 in_radius_mask = _mm256_cmp_ps(planar_distance, cull_vec, _CMP_LE_OQ);

            __m256 penalty = _mm256_set1_ps(1.0f);
            if (params.frontFacesOnly) {
                __m256 t_penalty = _mm256_div_ps(_mm256_sub_ps(height_from_plane, softRejectEnd), denom_penalty);
                t_penalty = _mm256_max_ps(_mm256_setzero_ps(), _mm256_min_ps(_mm256_set1_ps(1.0f), t_penalty));

                __m256 term1_penalty = _mm256_sub_ps(_mm256_set1_ps(3.0f), _mm256_mul_ps(_mm256_set1_ps(2.0f), t_penalty));
                penalty = _mm256_mul_ps(t_penalty, _mm256_mul_ps(t_penalty, term1_penalty));

                __m256 gt_start = _mm256_cmp_ps(height_from_plane, softRejectStart, _CMP_GE_OQ);
                __m256 lt_end = _mm256_cmp_ps(height_from_plane, softRejectEnd, _CMP_LE_OQ);
                penalty = _mm256_blendv_ps(penalty, _mm256_set1_ps(1.0f), gt_start);
                penalty = _mm256_blendv_ps(penalty, _mm256_setzero_ps(), lt_end);
            }

            __m256 normalizedDistance = _mm256_div_ps(planar_distance, radius_vec);
            __m256 n_weight = _mm256_max_ps(_mm256_setzero_ps(), _mm256_min_ps(_mm256_set1_ps(1.0f), normalizedDistance));

            __m256 outerT = _mm256_div_ps(_mm256_sub_ps(n_weight, inner_weight), denom_weight);
            outerT = _mm256_max_ps(_mm256_setzero_ps(), _mm256_min_ps(_mm256_set1_ps(1.0f), outerT));

            __m256 term1_weight = _mm256_sub_ps(_mm256_set1_ps(3.0f), _mm256_mul_ps(_mm256_set1_ps(2.0f), outerT));
            __m256 weight_base = _mm256_sub_ps(_mm256_set1_ps(1.0f), _mm256_mul_ps(outerT, _mm256_mul_ps(outerT, term1_weight)));

            __m256 le_inner = _mm256_cmp_ps(n_weight, inner_weight, _CMP_LE_OQ);
            weight_base = _mm256_blendv_ps(weight_base, _mm256_set1_ps(1.0f), le_inner);

            __m256 t_falloff = _mm256_sub_ps(_mm256_set1_ps(1.0f), n_weight);
            t_falloff = _mm256_max_ps(_mm256_setzero_ps(), _mm256_min_ps(_mm256_set1_ps(1.0f), t_falloff));

            __m256 falloff_val;
            switch (params.falloffType) {
                case 1: falloff_val = t_falloff; break;
                case 2: falloff_val = _mm256_mul_ps(t_falloff, t_falloff); break;
                case 3:
                    {
                        __m256 one_minus_t = _mm256_sub_ps(_mm256_set1_ps(1.0f), t_falloff);
                        __m256 term = _mm256_sub_ps(_mm256_set1_ps(1.0f), _mm256_mul_ps(one_minus_t, one_minus_t));
                        falloff_val = _mm256_sqrt_ps(_mm256_max_ps(_mm256_setzero_ps(), term));
                    }
                    break;
                case 4: falloff_val = _mm256_sqrt_ps(t_falloff); break;
                case 5:
                case 0:
                default:
                    {
                        __m256 term = _mm256_sub_ps(_mm256_set1_ps(3.0f), _mm256_mul_ps(_mm256_set1_ps(2.0f), t_falloff));
                        falloff_val = _mm256_mul_ps(t_falloff, _mm256_mul_ps(t_falloff, term));
                    }
                    break;
            }

            __m256 weight = _mm256_mul_ps(weight_base, falloff_val);
            weight = _mm256_mul_ps(weight, penalty);
            weight = _mm256_mul_ps(weight, damp_vec);

            if (fastSIMD) {
                __m256 radial = _mm256_sub_ps(_mm256_set1_ps(1.0f), normalizedDistance);
                radial = _mm256_max_ps(_mm256_setzero_ps(), _mm256_min_ps(_mm256_set1_ps(1.0f), radial));

                __m256 mask_val;
                if (params.brushSettings.alpha_preset == Paint::BrushAlphaPreset::HardRound) {
                    __m256 gt = _mm256_cmp_ps(radial, _mm256_set1_ps(0.2f), _CMP_GT_OQ);
                    mask_val = _mm256_blendv_ps(_mm256_setzero_ps(), _mm256_set1_ps(1.0f), gt);
                } else {
                    mask_val = radial;
                }
                weight = _mm256_mul_ps(weight, mask_val);
            }

            weight = _mm256_blendv_ps(_mm256_setzero_ps(), weight, in_radius_mask);

            // Aktif şerit (active lane) kontrolü ve erken atlama
            __m256 active_mask = _mm256_cmp_ps(weight, _mm256_set1_ps(1e-5f), _CMP_GT_OQ);
            int moveMask = _mm256_movemask_ps(active_mask);
            int activeBits = moveMask & ((1 << count) - 1);
            if (activeBits == 0) {
                continue; // Blokta fırça etki alanına giren aktif vertex yok, doğrudan atla
            }

            __m256 clay_drag_factor = _mm256_set1_ps(1.0f);
            if ((params.activeTool == SceneUI::SculptBrushTool::Clay || params.activeTool == SceneUI::SculptBrushTool::ClayStrips) &&
                params.strokeDistance > params.strokeSpacing * 0.15f) {
                __m256 tangentNorm = _mm256_add_ps(_mm256_mul_ps(pox, tangent_x), _mm256_add_ps(_mm256_mul_ps(poy, tangent_y), _mm256_mul_ps(poz, tangent_z)));
                tangentNorm = _mm256_div_ps(tangentNorm, radius_vec);

                clay_drag_factor = _mm256_add_ps(_mm256_mul_ps(tangentNorm, _mm256_set1_ps(1.1f)), _mm256_set1_ps(0.5f));
                clay_drag_factor = _mm256_max_ps(_mm256_setzero_ps(), _mm256_min_ps(_mm256_set1_ps(1.6f), clay_drag_factor));
            }

            alignas(32) float w_res[8];
            alignas(32) float h_res[8];
            alignas(32) float pd_res[8];
            alignas(32) float cdf_res[8];
            alignas(32) float wx_res[8];
            alignas(32) float wy_res[8];
            alignas(32) float wz_res[8];
            alignas(32) float pox_res[8];
            alignas(32) float poy_res[8];
            alignas(32) float poz_res[8];
            alignas(32) float pen_res[8];

            _mm256_store_ps(w_res, weight);
            _mm256_store_ps(h_res, height_from_plane);
            _mm256_store_ps(pd_res, planar_distance);
            _mm256_store_ps(cdf_res, clay_drag_factor);
            _mm256_store_ps(wx_res, wx);
            _mm256_store_ps(wy_res, wy);
            _mm256_store_ps(wz_res, wz);
            _mm256_store_ps(pox_res, pox);
            _mm256_store_ps(poy_res, poy);
            _mm256_store_ps(poz_res, poz);
            _mm256_store_ps(pen_res, penalty);

            for (size_t k = 0; k < count; ++k) {
                if (!(activeBits & (1 << k))) {
                    continue;
                }
                int vid = vids[k];

                float final_w = w_res[k];
                if (params.useMask && !fastSIMD) {
                    float nxAlpha = (pox_res[k] * params.tangent.x + poy_res[k] * params.tangent.y + poz_res[k] * params.tangent.z) / params.radius;
                    float nyAlpha = (pox_res[k] * params.bitangent.x + poy_res[k] * params.bitangent.y + poz_res[k] * params.bitangent.z) / params.radius;
                    // Shaped/aspect/alpha brush: REPLACE the circular radial profile
                    // (weight_base × falloff_val, baked into w_res) with the footprint weight so
                    // the deform matches the Alpha Preview. Front-face penalty + repeated-stroke
                    // damping are preserved (penalty × damp_factor). Circle + round preset takes
                    // the fastSIMD path above and never reaches here, so it is unchanged.
                    final_w = pen_res[k] * damp_factor *
                              uiSampleBrushFootprintWeight(params.brushSettings, nxAlpha, nyAlpha);
                }

                if (std::isfinite(final_w) && final_w > 1e-5f &&
                    std::isfinite(pd_res[k]) && std::isfinite(h_res[k])) {
                    size_t destIdx = startIdx + k;
                    results[destIdx].id = vid;
                    results[destIdx].worldPos = Vec3(wx_res[k], wy_res[k], wz_res[k]);
                    results[destIdx].height = h_res[k];
                    results[destIdx].planarDist = pd_res[k];
                    results[destIdx].weight = final_w;
                    results[destIdx].clayDragFactor = cdf_res[k];
                    activeFlags[destIdx] = 1;
                }
            }
        }

        for (size_t i = 0; i < numCandidates; ++i) {
            if (activeFlags[i]) {
                activeVerts.push_back(results[i]);
            }
        }
    }

    return activeVerts;
}

bool computeSculptBrushSample(
    const SceneUI::EditableVertex& vertex,
    const Matrix4x4& transform,
    const Vec3& planePoint,
    const Vec3& hitNormalWorld,
    const Vec3& tangentWorld,
    const Vec3& bitangentWorld,
    const Paint::BrushSettings& brushSettings,
    int proportionalFalloffType,
    SceneUI::SculptBrushTool activeTool,
    float falloff,
    float brushStrength,
    float radiusWorld,
    float strokeDistance,
    float strokeSpacing,
    float dt,
    bool strokeHasAccumulated,
    bool frontFacesOnly,
    SculptBrushSample& outSample) {
    outSample = SculptBrushSample{};
    outSample.local_position = vertex.local_position;
    outSample.world_position = sanitizeVec3(
        transform.transform_point(vertex.local_position),
        Vec3(0.0f, 0.0f, 0.0f));

    const Vec3 toVertex = outSample.world_position - planePoint;
    outSample.height_from_plane = toVertex.dot(hitNormalWorld);
    outSample.planar_offset = toVertex - hitNormalWorld * outSample.height_from_plane;
    const float planarDistance = outSample.planar_offset.length();
    outSample.planar_distance = computeHybridBrushDistance(
        toVertex,
        hitNormalWorld,
        planarDistance,
        radiusWorld);

    if (!std::isfinite(outSample.height_from_plane) ||
        !std::isfinite(outSample.planar_distance) ||
        outSample.planar_distance > radiusWorld) {
        return false;
    }
    const float frontFacePenalty = computeFrontFaceBrushPenalty(
        outSample.height_from_plane,
        radiusWorld,
        frontFacesOnly);
    if (frontFacePenalty <= 1e-5f) {
        return false;
    }

    float weight = computeTerrainLikeBrushWeight(outSample.planar_distance / radiusWorld, falloff);
    weight *= applyFalloffCurve(
        1.0f - saturateFloat(outSample.planar_distance / radiusWorld),
        proportionalFalloffType);
    weight *= frontFacePenalty;
    weight *= computeRepeatedStrokeDamping(
        strokeDistance,
        radiusWorld,
        dt,
        strokeHasAccumulated);

    if ((activeTool == SceneUI::SculptBrushTool::Clay ||
         activeTool == SceneUI::SculptBrushTool::ClayStrips) &&
        strokeDistance > strokeSpacing * 0.15f) {
        const float tangentNorm = outSample.planar_offset.dot(tangentWorld) / radiusWorld;
        outSample.clay_drag_factor = std::clamp(0.5f + tangentNorm * 1.1f, 0.0f, 1.6f);
    }

    const float nxAlpha = outSample.planar_offset.dot(tangentWorld) / radiusWorld;
    const float nyAlpha = outSample.planar_offset.dot(bitangentWorld) / radiusWorld;
    weight *= uiSampleBrushMask(brushSettings, nxAlpha, nyAlpha);
    if (!std::isfinite(weight) || weight <= 1e-5f) {
        return false;
    }

    outSample.weight = weight;
    return true;
}

bool computeSculptBrushSampleAtWorldPoint(
    const Vec3& localPosition,
    const Vec3& worldPosition,
    const Vec3& planePoint,
    const Vec3& prevPlanePoint,
    const Vec3& hitNormalWorld,
    const Vec3& tangentWorld,
    const Vec3& bitangentWorld,
    const Paint::BrushSettings& brushSettings,
    int proportionalFalloffType,
    SceneUI::SculptBrushTool activeTool,
    float falloff,
    float brushStrength,
    float radiusWorld,
    float strokeDistance,
    float strokeSpacing,
    float dt,
    bool strokeHasAccumulated,
    bool frontFacesOnly,
    SculptBrushSample& outSample) {
    outSample = SculptBrushSample{};
    outSample.local_position = localPosition;
    outSample.world_position = worldPosition;

    // Capsule sweep: nokta yerine [prevPlanePoint -> planePoint] segmentine olan
    // mesafeyi kullanıyoruz. Bu, fare hızlı hareket ettiğinde ardışık dab merkezleri
    // arasında oluşan boşluğu tek pass ile kapatır — ekstra dab maliyeti yok,
    // vertex başına yalnızca +1 dot product, +1 clamp.
    const Vec3 segDir = planePoint - prevPlanePoint;
    const float segLenSq = segDir.dot(segDir);
    Vec3 closestPlanePoint;
    if (segLenSq > 1e-10f) {
        const float t = std::clamp(
            (worldPosition - prevPlanePoint).dot(segDir) / segLenSq,
            0.0f,
            1.0f);
        closestPlanePoint = prevPlanePoint + segDir * t;
    } else {
        closestPlanePoint = planePoint;
    }
    const Vec3 toPoint = worldPosition - closestPlanePoint;
    outSample.height_from_plane = toPoint.dot(hitNormalWorld);
    outSample.planar_offset = toPoint - hitNormalWorld * outSample.height_from_plane;
    const float planarDistance = outSample.planar_offset.length();
    outSample.planar_distance = computeHybridBrushDistance(
        toPoint,
        hitNormalWorld,
        planarDistance,
        radiusWorld);

    if (!std::isfinite(outSample.height_from_plane) ||
        !std::isfinite(outSample.planar_distance) ||
        outSample.planar_distance > radiusWorld) {
        return false;
    }
    const float frontFacePenalty = computeFrontFaceBrushPenalty(
        outSample.height_from_plane,
        radiusWorld,
        frontFacesOnly);
    if (frontFacePenalty <= 1e-5f) {
        return false;
    }

    float weight = computeTerrainLikeBrushWeight(outSample.planar_distance / radiusWorld, falloff);
    weight *= applyFalloffCurve(
        1.0f - saturateFloat(outSample.planar_distance / radiusWorld),
        proportionalFalloffType);
    weight *= frontFacePenalty;
    weight *= computeRepeatedStrokeDamping(
        strokeDistance,
        radiusWorld,
        dt,
        strokeHasAccumulated);

    if ((activeTool == SceneUI::SculptBrushTool::Clay ||
         activeTool == SceneUI::SculptBrushTool::ClayStrips) &&
        strokeDistance > strokeSpacing * 0.15f) {
        const float tangentNorm = outSample.planar_offset.dot(tangentWorld) / radiusWorld;
        outSample.clay_drag_factor = std::clamp(0.5f + tangentNorm * 1.1f, 0.0f, 1.6f);
    }

    const float nxAlpha = outSample.planar_offset.dot(tangentWorld) / radiusWorld;
    const float nyAlpha = outSample.planar_offset.dot(bitangentWorld) / radiusWorld;
    weight *= uiSampleBrushMask(brushSettings, nxAlpha, nyAlpha);
    if (!std::isfinite(weight) || weight <= 1e-5f) {
        return false;
    }

    outSample.weight = weight;
    return true;
}

std::vector<int> collectTouchedSculptLeafIds(
    const SceneUI::SculptControlGraph& graph,
    const std::vector<int>& sourceVertexIds) {
    std::vector<int> touchedLeafIds;
    touchedLeafIds.reserve(sourceVertexIds.size());
    for (const int vertexIdInt : sourceVertexIds) {
        if (vertexIdInt < 0 || vertexIdInt >= static_cast<int>(graph.source_vertex_to_node_id.size())) {
            continue;
        }
        const int nodeId = graph.source_vertex_to_node_id[static_cast<size_t>(vertexIdInt)];
        if (nodeId >= 0) {
            touchedLeafIds.push_back(nodeId);
        }
    }
    std::sort(touchedLeafIds.begin(), touchedLeafIds.end());
    touchedLeafIds.erase(std::unique(touchedLeafIds.begin(), touchedLeafIds.end()), touchedLeafIds.end());
    return touchedLeafIds;
}

std::vector<int> collectTouchedSculptPBVHLeafIds(
    const SceneUI::SculptPBVH& pbvh,
    const std::vector<int>& sourceVertexIds) {
    std::vector<int> touchedLeafIds;
    touchedLeafIds.reserve(sourceVertexIds.size());
    for (const int vertexIdInt : sourceVertexIds) {
        if (vertexIdInt < 0 || vertexIdInt >= static_cast<int>(pbvh.source_vertex_to_leaf_id.size())) {
            continue;
        }
        const int leafId = pbvh.source_vertex_to_leaf_id[static_cast<size_t>(vertexIdInt)];
        if (leafId >= 0) {
            touchedLeafIds.push_back(leafId);
        }
    }
    std::sort(touchedLeafIds.begin(), touchedLeafIds.end());
    touchedLeafIds.erase(std::unique(touchedLeafIds.begin(), touchedLeafIds.end()), touchedLeafIds.end());
    return touchedLeafIds;
}

std::vector<int> collectSculptPBVHCandidateVertices(
    SceneUI::SculptPBVH& pbvh,
    const SceneUI::EditableMeshCache& editableMeshCache,
    const Vec3& localCenter,
    float localRadius);

std::vector<int> collectSculptCandidateVerticesWithPBVHFallback(
    SceneUI::SculptPBVH& pbvh,
    const SceneUI::EditableMeshCache& editableMeshCache,
    const Vec3& localCenter,
    float localRadius) {
    std::vector<int> candidateVertexIds = collectSculptPBVHCandidateVertices(
        pbvh,
        editableMeshCache,
        localCenter,
        localRadius);
    if (!candidateVertexIds.empty()) {
        return candidateVertexIds;
    }
    return collectEditableVertexCandidates(
        editableMeshCache,
        localCenter,
        localRadius);
}


bool applyNonGrabSculptCandidate(
    SceneUI::SculptBrushTool activeTool,
    SceneUI::EditableMeshCache& editableMeshCache,
    const SceneUI::SculptControlGraph& sculptControlGraph,
    SceneUI::SculptStrokeState& strokeState,
    const std::vector<Vec3>& snapshotLocalPositions,
    const Matrix4x4& transform,
    const Matrix4x4& inverseTransform,
    const Vec3& planePoint,
    const Vec3& claySamplePoint,
    const Vec3& hitNormalWorld,
    const Vec3& strokeTangentWorld,
    const Vec3& strokeBitangentWorld,
    const Vec3& strokeStepWorld,
    float brushStrength,
    float clayBrushStrength,
    float clayRadiusCompensation,
    float normalStrength,
    float directionSign,
    float radiusWorld,
    float dt,
    float localRadius,
    float strokeAdvanceFactor,
    int vertexIdInt,
    const SculptBrushSample& sample,
    Vec3& inOutLocalPosition) {
    if (vertexIdInt < 0) {
        return false;
    }

    const size_t vertexId = static_cast<size_t>(vertexIdInt);
    if (vertexId >= editableMeshCache.vertices.size()) {
        return false;
    }

    auto& vertex = editableMeshCache.vertices[vertexId];
    const bool isBoundary = (vertexId < editableMeshCache.vertex_is_boundary.size()) ? (editableMeshCache.vertex_is_boundary[vertexId] != 0) : false;
    const bool useSpatialNodeSolve = sculptControlGraph.uses_spatial_leaf_nodes;
    const int nodeId = (useSpatialNodeSolve && vertexId < sculptControlGraph.source_vertex_to_node_id.size())
        ? sculptControlGraph.source_vertex_to_node_id[vertexId]
        : -1;
    const SceneUI::SculptControlNode* node = nullptr;
    if (useSpatialNodeSolve &&
        nodeId >= 0 &&
        nodeId < static_cast<int>(sculptControlGraph.nodes.size())) {
        node = &sculptControlGraph.nodes[static_cast<size_t>(nodeId)];
    }
    const Vec3 nodeWorldPosition = (useSpatialNodeSolve && node)
        ? sanitizeVec3(transform.transform_point(node->local_position), sample.world_position)
        : sample.world_position;
    const float nodeVertexInfluence = (useSpatialNodeSolve && node)
        ? std::clamp(getSculptNodeVertexInfluence(*node, vertexIdInt), 0.15f, 1.0f)
        : 1.0f;
    const float leafWeightScale = (useSpatialNodeSolve && node)
        ? (1.0f / std::sqrt((std::max)(1.0f, node->area_weight)))
        : 1.0f;
    const float largeBrushSurfaceFactor = computeLargeBrushSurfaceFactor(radiusWorld);
    const Vec3 currentLocalPosition = resolveEditableSnapshotLocalPosition(
        editableMeshCache,
        snapshotLocalPositions,
        vertexId);
    Vec3 worldDelta(0.0f, 0.0f, 0.0f);

    switch (activeTool) {
    case SceneUI::SculptBrushTool::Inflate:
        worldDelta = hitNormalWorld * (radiusWorld * 0.22f * brushStrength * dt * sample.weight * (1.0f + normalStrength) * directionSign);
        break;
    // Stamp shares the Draw deform: it presses the surface along the (vertex/hit
    // blended) normal, scaled by sample.weight — which already carries the loaded
    // alpha/mask texture from the brush footprint. So a Stamp imprint == a Draw
    // dab whose footprint IS the mask texture. The Anchored stroke mode (handled
    // upstream) is what turns it into a single fixed-centre press + live resize.
    case SceneUI::SculptBrushTool::Stamp:
    case SceneUI::SculptBrushTool::Draw: {
        Vec3 vertexNormal = hitNormalWorld;
        if (!vertex.refs.empty() && editableMeshCache.refTri(vertex.refs[0])) {
            const auto& ref = vertex.refs[0];
            const Vec3 n = editableMeshCache.refTri(ref)->vertices[ref.corner].normal;
            if (n.length_squared() > 1e-8f) {
                vertexNormal = safeNormalizeVec3(n, hitNormalWorld);
            }
        }
        const float centerBias = std::clamp(sample.weight, 0.0f, 1.0f);
        const float drawStrokeNormalMix =
            std::clamp(0.68f + 0.22f * largeBrushSurfaceFactor + 0.08f * centerBias, 0.0f, 0.94f);
        const Vec3 drawDirection = safeNormalizeVec3(
            vertexNormal * (1.0f - drawStrokeNormalMix) + hitNormalWorld * drawStrokeNormalMix,
            hitNormalWorld);
        worldDelta = drawDirection * (radiusWorld * 0.22f * brushStrength * dt * sample.weight * (1.0f + normalStrength) * directionSign);
        break;
    }
    case SceneUI::SculptBrushTool::Layer: {
        if (vertexId >= strokeState.layer_accum.size()) {
            return false;
        }
        float& layerAccum = strokeState.layer_accum[vertexId];
        const float previousAccum = layerAccum;
        const float targetLayer = radiusWorld * 0.22f * clayRadiusCompensation * sample.weight * (0.9f + 0.35f * normalStrength);
        const float deposit = radiusWorld * 0.09f * clayRadiusCompensation * clayBrushStrength * dt * sample.weight * directionSign * leafWeightScale;
        if (directionSign > 0.0f) {
            layerAccum = std::min(layerAccum + std::max(0.0f, deposit), targetLayer);
        } else {
            layerAccum = std::max(layerAccum + std::min(0.0f, deposit), -targetLayer);
        }
        worldDelta = hitNormalWorld * (layerAccum - previousAccum);
        break;
    }
    case SceneUI::SculptBrushTool::Clay: {
        Vec3 vertexNormal = hitNormalWorld;
        if (!vertex.refs.empty() && editableMeshCache.refTri(vertex.refs[0])) {
            const auto& ref = vertex.refs[0];
            const Vec3 n = editableMeshCache.refTri(ref)->vertices[ref.corner].normal;
            if (n.length_squared() > 1e-8f) {
                vertexNormal = safeNormalizeVec3(n, hitNormalWorld);
            }
        }
        if (vertexId >= strokeState.clay_layer_accum.size()) {
            return false;
        }
        const float clayHeightRef = radiusWorld * clayRadiusCompensation;
        const Vec3 clayReferenceWorldPosition =
            (sculptControlGraph.uses_spatial_leaf_nodes && node)
                ? sanitizeVec3(
                    sample.world_position + (nodeWorldPosition - sample.world_position) * (0.35f * nodeVertexInfluence),
                    sample.world_position)
                : sample.world_position;
        // Clay'i tamamen brush strength'e duyarlı hale getir: saturated clayAccum
        // ve settle terimleri yerleşim referansına bağlı olduğu için eskiden
        // strength düşürülse bile alt sınır azalmıyordu. clayHeightRef'i strength
        // ile ölçeklediğimizde tüm türev terimler (target, deposit, accum cap,
        // settle clamp) orantılı olarak küçülür.
        const float clayStrengthScale = sanitizeFiniteFloat(clayBrushStrength, 1.0f, 0.0f, 5.0f);
        const float clayHeightRefScaled = clayHeightRef * clayStrengthScale;
        const float signedDistance = (clayReferenceWorldPosition - claySamplePoint).dot(hitNormalWorld);
        const float targetHeight =
            directionSign * clayHeightRefScaled * 0.16f * (0.8f + 0.5f * normalStrength) * sample.clay_drag_factor;
        const float heightError = targetHeight - signedDistance;
        const float absTarget = (std::abs)(targetHeight);
        const float fillNeed = (absTarget > 1e-5f)
            ? std::clamp(((directionSign > 0.0f) ? heightError : -heightError) / absTarget, 0.0f, 1.0f)
            : 0.0f;
        const float deposit =
            clayHeightRefScaled * 0.075f * clayBrushStrength * dt * sample.weight * strokeAdvanceFactor *
            (0.8f + 0.5f * normalStrength) * directionSign * fillNeed * sample.clay_drag_factor * leafWeightScale *
            (1.0f - 0.22f * largeBrushSurfaceFactor);
        float& clayAccum = strokeState.clay_layer_accum[vertexId];
        clayAccum = std::clamp(
            clayAccum + deposit,
            -clayHeightRefScaled * (0.45f - 0.08f * largeBrushSurfaceFactor),
            clayHeightRefScaled * (0.45f - 0.08f * largeBrushSurfaceFactor));
        const float settle =
            std::clamp(
                heightError * sample.weight * (0.28f + 0.32f * sample.weight),
                -clayHeightRefScaled * (0.05f - 0.012f * largeBrushSurfaceFactor),
                clayHeightRefScaled * (0.05f - 0.012f * largeBrushSurfaceFactor));
        const float layerDelta = settle * (0.7f + 0.12f * largeBrushSurfaceFactor) +
            deposit * (0.55f - 0.10f * largeBrushSurfaceFactor) +
            clayAccum * (0.12f - 0.03f * largeBrushSurfaceFactor) * sample.clay_drag_factor;
        const float buildup = (deposit * 0.14f + (std::max)(0.0f, heightError) * 0.10f * fillNeed) *
            (1.0f - 0.45f * largeBrushSurfaceFactor);
        if (isBoundary) {
            worldDelta = hitNormalWorld * (layerDelta + buildup) * (0.6f - 0.12f * largeBrushSurfaceFactor);
        } else {
            const float vertexNormalMix = 1.0f - 0.35f * largeBrushSurfaceFactor;
            worldDelta = hitNormalWorld * layerDelta + vertexNormal * (buildup * vertexNormalMix);
        }
        break;
    }
    case SceneUI::SculptBrushTool::ClayStrips: {
        if (vertexId >= strokeState.clay_strips_layer_accum.size()) {
            return false;
        }
        const Vec3 clayReferenceWorldPosition =
            (sculptControlGraph.uses_spatial_leaf_nodes && node)
                ? sanitizeVec3(
                    sample.world_position + (nodeWorldPosition - sample.world_position) * (0.28f * nodeVertexInfluence),
                    sample.world_position)
                : sample.world_position;
        const Vec3 clayPlanarOffset = clayReferenceWorldPosition - claySamplePoint -
            hitNormalWorld * (clayReferenceWorldPosition - claySamplePoint).dot(hitNormalWorld);
        const float stripCoord = clayPlanarOffset.dot(strokeBitangentWorld) / radiusWorld;
        const float tineWave = 0.5f + 0.5f * std::cos(stripCoord * 22.0f);
        const float tineProfile = std::pow((std::max)(0.0f, tineWave), 1.6f);
        const float rakePattern = 0.45f + 0.55f * tineProfile;
        // ClayStrips: stripsHeightRef'i de strength ile ölçekle (Clay ile aynı gerekçe).
        const float stripsStrengthScale = sanitizeFiniteFloat(clayBrushStrength, 1.0f, 0.0f, 5.0f);
        const float stripsHeightRef = radiusWorld * clayRadiusCompensation * stripsStrengthScale;
        const float signedDistance = (clayReferenceWorldPosition - claySamplePoint).dot(hitNormalWorld);
        const float stripTargetHeight =
            directionSign * stripsHeightRef * 0.14f * rakePattern * (0.9f + 0.35f * normalStrength) * sample.clay_drag_factor;
        const float stripHeightError = stripTargetHeight - signedDistance;
        const float absStripTarget = (std::abs)(stripTargetHeight);
        const float stripFillNeed = (absStripTarget > 1e-5f)
            ? std::clamp(((directionSign > 0.0f) ? stripHeightError : -stripHeightError) / absStripTarget, 0.0f, 1.0f)
            : 0.0f;
        const float deposit =
            stripsHeightRef * 0.11f * clayBrushStrength * dt * sample.weight * strokeAdvanceFactor * rakePattern *
            (0.95f + 0.45f * normalStrength) * directionSign * stripFillNeed * sample.clay_drag_factor * leafWeightScale *
            (1.0f - 0.18f * largeBrushSurfaceFactor);
        float& clayStripsAccum = strokeState.clay_strips_layer_accum[vertexId];
        clayStripsAccum = std::clamp(
            clayStripsAccum + deposit,
            -stripsHeightRef * (0.6f - 0.10f * largeBrushSurfaceFactor),
            stripsHeightRef * (0.6f - 0.10f * largeBrushSurfaceFactor));
        const float settle =
            std::clamp(
                stripHeightError * sample.weight * (0.24f + 0.28f * sample.weight),
                -stripsHeightRef * (0.05f - 0.012f * largeBrushSurfaceFactor),
                stripsHeightRef * (0.05f - 0.012f * largeBrushSurfaceFactor));
        const float layerDelta = settle * (0.62f + 0.10f * largeBrushSurfaceFactor) +
            deposit * (0.62f - 0.10f * largeBrushSurfaceFactor) +
            clayStripsAccum * (0.18f - 0.05f * largeBrushSurfaceFactor);
        const float tineDrag =
            radiusWorld * 0.05f * clayBrushStrength * dt * sample.weight * strokeAdvanceFactor *
            (0.35f + 0.65f * tineProfile) * (1.0f - 0.25f * largeBrushSurfaceFactor);
        if (isBoundary) {
            worldDelta = hitNormalWorld * layerDelta * (0.6f - 0.10f * largeBrushSurfaceFactor);
        } else {
            worldDelta = hitNormalWorld * layerDelta + strokeTangentWorld * tineDrag;
        }
        break;
    }
    case SceneUI::SculptBrushTool::Pinch: {
        const Vec3 toCenter = planePoint - sample.world_position;
        const Vec3 lateral = toCenter - hitNormalWorld * toCenter.dot(hitNormalWorld);
        const float lateralLen = lateral.length();
        if (lateralLen > 1e-8f) {
            worldDelta = (lateral / lateralLen) * (radiusWorld * 0.3f * brushStrength * dt * sample.weight);
        }
        break;
    }
    case SceneUI::SculptBrushTool::Flatten: {
        const float signedDistance = (sample.world_position - planePoint).dot(hitNormalWorld);
        worldDelta = hitNormalWorld * (-signedDistance * brushStrength * dt * 12.0f * sample.weight);
        break;
    }
    case SceneUI::SculptBrushTool::Scrape: {
        const float signedDistance = (sample.world_position - planePoint).dot(hitNormalWorld);
        if (signedDistance > 0.0f) {
            worldDelta = hitNormalWorld * (-signedDistance * brushStrength * dt * 14.0f * sample.weight);
        }
        break;
    }
    case SceneUI::SculptBrushTool::Crease: {
        const Vec3 toCenter = planePoint - sample.world_position;
        const Vec3 lateral = toCenter - hitNormalWorld * toCenter.dot(hitNormalWorld);
        const float lateralLen = lateral.length();
        const Vec3 pinchDelta = lateralLen > 1e-8f
            ? (lateral / lateralLen) * (radiusWorld * 0.38f * brushStrength * dt * sample.weight)
            : Vec3(0.0f, 0.0f, 0.0f);
        const Vec3 cutDelta = hitNormalWorld * (-radiusWorld * 0.12f * brushStrength * dt * sample.weight * (1.0f + normalStrength));
        worldDelta = pinchDelta + cutDelta;
        break;
    }
    case SceneUI::SculptBrushTool::DrawSharp: {
        // Crisp ridge/crease: square the weight so the falloff edge is much
        // tighter than Draw, and push along the vertex normal. Ctrl digs inward.
        Vec3 vertexNormal = hitNormalWorld;
        if (!vertex.refs.empty() && editableMeshCache.refTri(vertex.refs[0])) {
            const Vec3 n = editableMeshCache.refTri(vertex.refs[0])->vertices[vertex.refs[0].corner].normal;
            if (n.length_squared() > 1e-8f) {
                vertexNormal = safeNormalizeVec3(n, hitNormalWorld);
            }
        }
        const float sharpW = sample.weight * sample.weight;
        worldDelta = vertexNormal * (radiusWorld * 0.26f * brushStrength * dt * sharpW *
            (1.0f + normalStrength) * directionSign);
        break;
    }
    case SceneUI::SculptBrushTool::Nudge: {
        // Move the surface by the actual stroke step this frame (speed-aware),
        // weighted by falloff. No normal component, so it slides along the surface.
        worldDelta = strokeStepWorld * (sample.weight * brushStrength * directionSign);
        break;
    }
    case SceneUI::SculptBrushTool::Blob: {
        // Spherical swell: inflate along the normal + gather laterally toward the
        // brush center for a rounded bulge. Ctrl deflates/contracts.
        const Vec3 toCenter = planePoint - sample.world_position;
        const Vec3 lateral = toCenter - hitNormalWorld * toCenter.dot(hitNormalWorld);
        const float lateralLen = lateral.length();
        Vec3 gather(0.0f, 0.0f, 0.0f);
        if (lateralLen > 1e-8f) {
            gather = (lateral / lateralLen) * (radiusWorld * 0.10f * brushStrength * dt *
                sample.weight * directionSign);
        }
        worldDelta = hitNormalWorld * (radiusWorld * 0.26f * brushStrength * dt * sample.weight *
            (1.0f + normalStrength) * directionSign) + gather;
        break;
    }
    case SceneUI::SculptBrushTool::Fill: {
        // Directional flatten. dirSign > 0: only raise verts BELOW the brush plane
        // (fill valleys). Ctrl (dirSign < 0): only push down verts above (deepen).
        const float signedDistance = (sample.world_position - planePoint).dot(hitNormalWorld);
        if ((directionSign > 0.0f && signedDistance < 0.0f) ||
            (directionSign < 0.0f && signedDistance > 0.0f)) {
            worldDelta = hitNormalWorld * (-signedDistance * brushStrength * dt * 12.0f * sample.weight);
        }
        break;
    }
    case SceneUI::SculptBrushTool::SnakeHook: {
        // Drag the surface by the actual stroke motion (so the region follows the
        // cursor and trails into a hook), and pinch laterally toward the center so
        // the pulled tip narrows. Pinch scales with stroke speed (0 when still).
        const float stepLen = strokeStepWorld.length();
        const Vec3 toCenter = planePoint - sample.world_position;
        const Vec3 lateral = toCenter - hitNormalWorld * toCenter.dot(hitNormalWorld);
        const float lateralLen = lateral.length();
        Vec3 pinch(0.0f, 0.0f, 0.0f);
        if (lateralLen > 1e-8f) {
            pinch = (lateral / lateralLen) * (stepLen * 0.45f * brushStrength * sample.weight);
        }
        worldDelta = strokeStepWorld * (sample.weight * brushStrength) + pinch;
        break;
    }
    case SceneUI::SculptBrushTool::Smooth: {
        const Vec3 smoothDeltaLocal = computeBoundarySafeSmoothDelta(
            editableMeshCache,
            snapshotLocalPositions,
            vertexId,
            brushStrength,
            dt,
            sample.weight,
            localRadius);
        const float smoothLenSq = smoothDeltaLocal.length_squared();
        if (!std::isfinite(smoothLenSq) || smoothLenSq <= 1e-14f) {
            return false;
        }
        inOutLocalPosition = sanitizeVec3(currentLocalPosition + smoothDeltaLocal, currentLocalPosition);
        return true;
    }
    case SceneUI::SculptBrushTool::Grab:
    case SceneUI::SculptBrushTool::ElasticDeform:
        return false;
    default:
        return false;
    }

    worldDelta = sanitizeVec3(worldDelta, Vec3(0.0f, 0.0f, 0.0f));
    Vec3 localDelta = sanitizeVec3(inverseTransform.transform_vector(worldDelta), Vec3(0.0f, 0.0f, 0.0f));
    const float localDeltaLenSq = localDelta.length_squared();
    if (!std::isfinite(localDeltaLenSq) || localDeltaLenSq <= 1e-14f) {
        return false;
    }

    // Subdivide ile yoğunlaşmış meshlerde dab başına yer değiştirme tek bir tiny
    // üçgenin kenarını birkaç katı geçebiliyor; bu durumda
    // isEditableVertexTopologySafe vertex'i geri çeviriyor ve "anchor + komşular
    // hareket etmiş" desenli flip/çukur oluşuyor. Yerel ortalama kenar uzunluğunun
    // küçük bir kesri kadar adım sınırı koyuyoruz — vertex hâlâ doğru yönde
    // hareket eder ama her dab güvenli aralıkta kalır. Coarse meshlerde clamp
    // pratikte tetiklenmez; yalnızca yoğun bölgelerde devreye girer.
    const float meshAvgEdgeLocal = sanitizeFiniteFloat(
        sculptControlGraph.avg_edge_length, 0.05f, 1e-6f, 1000000.0f);
    float maxLocalStep = (std::max)(meshAvgEdgeLocal * 0.4f, localRadius * 1e-4f);
    // Nudge / Snake Hook drag the surface along with the cursor: the tight
    // anti-flip clamp would make pulled verts lag behind the brush and drop out
    // of the candidate sphere (the pull "stalls" at the radius edge). Let them
    // take a much larger per-frame step so the region keeps up with the stroke.
    if (activeTool == SceneUI::SculptBrushTool::SnakeHook ||
        activeTool == SceneUI::SculptBrushTool::Nudge) {
        maxLocalStep = (std::max)(maxLocalStep, localRadius * 0.9f);
    }
    const float localDeltaLen = std::sqrt(localDeltaLenSq);
    if (std::isfinite(localDeltaLen) && localDeltaLen > maxLocalStep) {
        localDelta *= (maxLocalStep / localDeltaLen);
    }

    inOutLocalPosition = sanitizeVec3(currentLocalPosition + localDelta, currentLocalPosition);
    return true;
}

void rebuildSculptControlGraphBuckets(SceneUI::SculptControlGraph& graph) {
    graph.node_spatial_buckets.clear();
    graph.node_spatial_buckets.reserve(graph.nodes.size());
    for (size_t nodeId = 0; nodeId < graph.nodes.size(); ++nodeId) {
        const SceneUI::EditableSpatialCellKey key = makeEditableSpatialCellKey(
            graph.nodes[nodeId].local_position,
            graph.spatial_cell_size);
        graph.node_spatial_buckets[key].push_back(static_cast<int>(nodeId));
    }
}

void refreshSculptControlGraphNodeData(
    SceneUI::SculptControlGraph& graph,
    const SceneUI::EditableMeshCache& editableMeshCache) {
    for (size_t nodeId = 0; nodeId < graph.nodes.size(); ++nodeId) {
        auto& node = graph.nodes[nodeId];
        Vec3 accumulatedPosition(0.0f, 0.0f, 0.0f);
        Vec3 accumulatedNormal(0.0f, 0.0f, 0.0f);
        float accumulatedWeight = 0.0f;
        bool isBoundary = false;

        for (size_t localIndex = 0; localIndex < node.source_vertex_ids.size(); ++localIndex) {
            const int vertexIdInt = node.source_vertex_ids[localIndex];
            const size_t vertexId = static_cast<size_t>(vertexIdInt);
            if (vertexIdInt < 0 || vertexId >= editableMeshCache.vertices.size()) {
                continue;
            }

            const auto& vertex = editableMeshCache.vertices[vertexId];
            const float weight = (localIndex < node.source_weights.size() && std::isfinite(node.source_weights[localIndex]))
                ? node.source_weights[localIndex]
                : 1.0f;
            accumulatedPosition += vertex.local_position * weight;

            Vec3 vertexNormal(0.0f, 1.0f, 0.0f);
            if (!vertex.refs.empty() && editableMeshCache.refTri(vertex.refs[0])) {
                const Vec3 normal = editableMeshCache.refTri(vertex.refs[0])->getOriginalVertexNormal(vertex.refs[0].corner);
                if (normal.length_squared() > 1e-8f) {
                    vertexNormal = normal.normalize();
                }
            }
            accumulatedNormal += vertexNormal * weight;
            accumulatedWeight += weight;
            isBoundary = isBoundary || vertex.is_boundary;
        }

        if (accumulatedWeight > 1e-8f) {
            node.local_position = accumulatedPosition / accumulatedWeight;
        }
        if (accumulatedNormal.length_squared() > 1e-8f) {
            node.local_normal = accumulatedNormal.normalize();
        } else {
            node.local_normal = Vec3(0.0f, 1.0f, 0.0f);
        }
        node.area_weight = (std::max)(1.0f, accumulatedWeight);
        node.is_boundary = isBoundary;
    }

    rebuildSculptControlGraphBuckets(graph);
}

void buildVertexSculptControlGraph(
    SceneUI::SculptControlGraph& graph,
    const SceneUI::EditableMeshCache& editableMeshCache) {
    graph.uses_spatial_leaf_nodes = false;
    graph.source_vertex_to_node_id.resize(editableMeshCache.vertices.size(), -1);
    graph.nodes.resize(editableMeshCache.vertices.size());
    for (size_t vertexId = 0; vertexId < editableMeshCache.vertices.size(); ++vertexId) {
        graph.source_vertex_to_node_id[vertexId] = static_cast<int>(vertexId);
        auto& node = graph.nodes[vertexId];
        const auto nodeNeighbors = editableMeshCache.vertex_neighbors[vertexId];
        node.neighbor_ids.assign(nodeNeighbors.begin(), nodeNeighbors.end());
        node.source_vertex_ids = { static_cast<int>(vertexId) };
        node.source_weights = { 1.0f };
    }
    refreshSculptControlGraphNodeData(graph, editableMeshCache);
}

void buildSpatialLeafSculptControlGraph(
    SceneUI::SculptControlGraph& graph,
    const SceneUI::EditableMeshCache& editableMeshCache) {
    graph.uses_spatial_leaf_nodes = true;
    graph.source_vertex_to_node_id.assign(editableMeshCache.vertices.size(), -1);
    graph.nodes.clear();
    graph.nodes.reserve(editableMeshCache.vertex_spatial_buckets.size());

    for (const auto& bucketEntry : editableMeshCache.vertex_spatial_buckets) {
        const auto& sourceVertexIds = bucketEntry.second;
        if (sourceVertexIds.empty()) {
            continue;
        }

        const int nodeId = static_cast<int>(graph.nodes.size());
        graph.nodes.push_back(SceneUI::SculptControlNode{});
        auto& node = graph.nodes.back();
        node.source_vertex_ids = sourceVertexIds;
        node.source_weights.assign(sourceVertexIds.size(), 1.0f);

        Vec3 center(0.0f, 0.0f, 0.0f);
        int validCount = 0;
        for (const int vertexIdInt : sourceVertexIds) {
            if (vertexIdInt < 0 || vertexIdInt >= static_cast<int>(editableMeshCache.vertices.size())) {
                continue;
            }
            center += editableMeshCache.vertices[static_cast<size_t>(vertexIdInt)].local_position;
            ++validCount;
        }
        if (validCount > 0) {
            center /= static_cast<float>(validCount);
        }

        for (size_t localIndex = 0; localIndex < sourceVertexIds.size(); ++localIndex) {
            const int vertexIdInt = sourceVertexIds[localIndex];
            if (vertexIdInt < 0 || vertexIdInt >= static_cast<int>(editableMeshCache.vertices.size())) {
                continue;
            }
            const Vec3 offset = editableMeshCache.vertices[static_cast<size_t>(vertexIdInt)].local_position - center;
            const float distance = offset.length();
            node.source_weights[localIndex] = 1.0f / (1.0f + distance);
        }
        for (const int vertexIdInt : sourceVertexIds) {
            if (vertexIdInt >= 0 && vertexIdInt < static_cast<int>(graph.source_vertex_to_node_id.size())) {
                graph.source_vertex_to_node_id[static_cast<size_t>(vertexIdInt)] = nodeId;
            }
        }
    }

    std::vector<std::unordered_set<int>> neighborSets(graph.nodes.size());
    for (size_t vertexId = 0; vertexId < editableMeshCache.vertex_neighbors.size(); ++vertexId) {
        const int nodeId = (vertexId < graph.source_vertex_to_node_id.size())
            ? graph.source_vertex_to_node_id[vertexId]
            : -1;
        if (nodeId < 0) {
            continue;
        }
        for (const int neighborVertexId : editableMeshCache.vertex_neighbors[vertexId]) {
            if (neighborVertexId < 0 || neighborVertexId >= static_cast<int>(graph.source_vertex_to_node_id.size())) {
                continue;
            }
            const int neighborNodeId = graph.source_vertex_to_node_id[static_cast<size_t>(neighborVertexId)];
            if (neighborNodeId >= 0 && neighborNodeId != nodeId) {
                neighborSets[static_cast<size_t>(nodeId)].insert(neighborNodeId);
            }
        }
    }
    for (size_t nodeId = 0; nodeId < graph.nodes.size(); ++nodeId) {
        graph.nodes[nodeId].neighbor_ids.assign(
            neighborSets[nodeId].begin(),
            neighborSets[nodeId].end());
    }

    refreshSculptControlGraphNodeData(graph, editableMeshCache);
}

float computeEditableAverageEdgeLength(const SceneUI::EditableMeshCache& editableMeshCache) {
    float avgEdgeLength = 0.0f;
    int avgEdgeSamples = 0;
    for (const auto& edge : editableMeshCache.edges) {
        if (edge.v0 < 0 || edge.v1 < 0 ||
            edge.v0 >= static_cast<int>(editableMeshCache.vertices.size()) ||
            edge.v1 >= static_cast<int>(editableMeshCache.vertices.size())) {
            continue;
        }
        const float edgeLength = (
            editableMeshCache.vertices[edge.v1].local_position -
            editableMeshCache.vertices[edge.v0].local_position).length();
        if (!std::isfinite(edgeLength) || edgeLength <= 1e-6f) {
            continue;
        }
        avgEdgeLength += edgeLength;
        ++avgEdgeSamples;
    }

    if (avgEdgeSamples > 0) {
        avgEdgeLength /= static_cast<float>(avgEdgeSamples);
    }
    return avgEdgeLength;
}

AABB computeEditableVertexBounds(
    const SceneUI::EditableMeshCache& editableMeshCache,
    const std::vector<int>& vertexIds) {
    AABB bounds;
    bool hasVertex = false;
    for (const int vertexIdInt : vertexIds) {
        if (vertexIdInt < 0 || vertexIdInt >= static_cast<int>(editableMeshCache.vertices.size())) {
            continue;
        }
        const Vec3 p = editableMeshCache.vertices[static_cast<size_t>(vertexIdInt)].local_position;
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
            continue;
        }
        if (!hasVertex) {
            bounds = AABB(p, p);
            hasVertex = true;
        } else {
            bounds.min.x = (std::min)(bounds.min.x, p.x);
            bounds.min.y = (std::min)(bounds.min.y, p.y);
            bounds.min.z = (std::min)(bounds.min.z, p.z);
            bounds.max.x = (std::max)(bounds.max.x, p.x);
            bounds.max.y = (std::max)(bounds.max.y, p.y);
            bounds.max.z = (std::max)(bounds.max.z, p.z);
        }
    }
    if (!hasVertex) {
        bounds = AABB(Vec3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, 0.0f));
    }
    return bounds;
}

bool pbvhNodeIntersectsSphere(const SceneUI::SculptPBVHNode& node, const Vec3& center, float radius) {
    const float safeRadius = sanitizeFiniteFloat(radius, 0.1f, 0.0001f, 100000.0f);
    float distSq = 0.0f;
    for (int axis = 0; axis < 3; ++axis) {
        const float v = center[axis];
        if (v < node.local_bounds.min[axis]) {
            const float d = node.local_bounds.min[axis] - v;
            distSq += d * d;
        } else if (v > node.local_bounds.max[axis]) {
            const float d = v - node.local_bounds.max[axis];
            distSq += d * d;
        }
    }
    return distSq <= safeRadius * safeRadius;
}

int buildSculptPBVHRecursive(
    SceneUI::SculptPBVH& pbvh,
    const SceneUI::EditableMeshCache& editableMeshCache,
    std::vector<int>& vertexIds,
    int parentNodeId,
    int depth) {
    const int nodeId = static_cast<int>(pbvh.nodes.size());
    pbvh.nodes.push_back(SceneUI::SculptPBVHNode{});
    auto& node = pbvh.nodes[static_cast<size_t>(nodeId)];
    node.parent_id = parentNodeId;
    node.depth = depth;
    node.local_bounds = computeEditableVertexBounds(editableMeshCache, vertexIds);
    pbvh.max_depth = (std::max)(pbvh.max_depth, depth);

    if (vertexIds.size() <= static_cast<size_t>(pbvh.leaf_vertex_limit)) {
        node.is_leaf = true;
        node.vertex_ids = vertexIds;
        node.is_boundary = false;
        for (const int vertexIdInt : node.vertex_ids) {
            if (vertexIdInt >= 0 && vertexIdInt < static_cast<int>(editableMeshCache.vertices.size())) {
                pbvh.source_vertex_to_leaf_id[static_cast<size_t>(vertexIdInt)] = nodeId;
                node.is_boundary = node.is_boundary || editableMeshCache.vertices[static_cast<size_t>(vertexIdInt)].is_boundary;
            }
        }
        ++pbvh.leaf_count;
        return nodeId;
    }

    const Vec3 diag = node.local_bounds.diagonal();
    int splitAxis = 0;
    if (diag.y > diag.x && diag.y >= diag.z) {
        splitAxis = 1;
    } else if (diag.z > diag.x && diag.z >= diag.y) {
        splitAxis = 2;
    }

    std::sort(vertexIds.begin(), vertexIds.end(), [&](int lhs, int rhs) {
        const Vec3& a = editableMeshCache.vertices[static_cast<size_t>(lhs)].local_position;
        const Vec3& b = editableMeshCache.vertices[static_cast<size_t>(rhs)].local_position;
        if (a[splitAxis] == b[splitAxis]) {
            return lhs < rhs;
        }
        return a[splitAxis] < b[splitAxis];
    });

    const size_t mid = vertexIds.size() / 2;
    std::vector<int> leftVertexIds(vertexIds.begin(), vertexIds.begin() + mid);
    std::vector<int> rightVertexIds(vertexIds.begin() + mid, vertexIds.end());
    if (leftVertexIds.empty() || rightVertexIds.empty()) {
        node.is_leaf = true;
        node.vertex_ids = vertexIds;
        node.is_boundary = false;
        for (const int vertexIdInt : node.vertex_ids) {
            if (vertexIdInt >= 0 && vertexIdInt < static_cast<int>(editableMeshCache.vertices.size())) {
                pbvh.source_vertex_to_leaf_id[static_cast<size_t>(vertexIdInt)] = nodeId;
                node.is_boundary = node.is_boundary || editableMeshCache.vertices[static_cast<size_t>(vertexIdInt)].is_boundary;
            }
        }
        ++pbvh.leaf_count;
        return nodeId;
    }

    const int leftChildId = buildSculptPBVHRecursive(pbvh, editableMeshCache, leftVertexIds, nodeId, depth + 1);
    const int rightChildId = buildSculptPBVHRecursive(pbvh, editableMeshCache, rightVertexIds, nodeId, depth + 1);
    auto& parentNode = pbvh.nodes[static_cast<size_t>(nodeId)];
    parentNode.left_child_id = leftChildId;
    parentNode.right_child_id = rightChildId;
    parentNode.is_leaf = false;
    parentNode.vertex_ids.clear();
    parentNode.local_bounds = surrounding_box(
        pbvh.nodes[static_cast<size_t>(leftChildId)].local_bounds,
        pbvh.nodes[static_cast<size_t>(rightChildId)].local_bounds);
    parentNode.is_boundary =
        pbvh.nodes[static_cast<size_t>(leftChildId)].is_boundary ||
        pbvh.nodes[static_cast<size_t>(rightChildId)].is_boundary;
    return nodeId;
}

AABB refreshSculptPBVHNodeBounds(
    SceneUI::SculptPBVH& pbvh,
    const SceneUI::EditableMeshCache& editableMeshCache,
    int nodeId) {
    auto& node = pbvh.nodes[static_cast<size_t>(nodeId)];
    if (node.is_leaf) {
        node.local_bounds = computeEditableVertexBounds(editableMeshCache, node.vertex_ids);
        node.is_boundary = false;
        for (const int vertexIdInt : node.vertex_ids) {
            if (vertexIdInt >= 0 && vertexIdInt < static_cast<int>(editableMeshCache.vertices.size())) {
                node.is_boundary = node.is_boundary || editableMeshCache.vertices[static_cast<size_t>(vertexIdInt)].is_boundary;
            }
        }
        return node.local_bounds;
    }

    const AABB leftBounds = refreshSculptPBVHNodeBounds(pbvh, editableMeshCache, node.left_child_id);
    const AABB rightBounds = refreshSculptPBVHNodeBounds(pbvh, editableMeshCache, node.right_child_id);
    node.local_bounds = surrounding_box(leftBounds, rightBounds);
    node.is_boundary =
        pbvh.nodes[static_cast<size_t>(node.left_child_id)].is_boundary ||
        pbvh.nodes[static_cast<size_t>(node.right_child_id)].is_boundary;
    return node.local_bounds;
}

void refreshSculptPBVHLeavesAndAncestors(
    SceneUI::SculptPBVH& pbvh,
    const SceneUI::EditableMeshCache& editableMeshCache,
    const std::vector<int>& touchedLeafIds) {
    if (touchedLeafIds.empty() || pbvh.nodes.empty()) {
        return;
    }

    std::vector<uint8_t> visited(pbvh.nodes.size(), 0u);
    std::vector<int> dirtyChain;
    dirtyChain.reserve(touchedLeafIds.size() * 8);

    for (const int leafId : touchedLeafIds) {
        int nodeId = leafId;
        while (nodeId >= 0 &&
               nodeId < static_cast<int>(pbvh.nodes.size()) &&
               visited[static_cast<size_t>(nodeId)] == 0u) {
            visited[static_cast<size_t>(nodeId)] = 1u;
            dirtyChain.push_back(nodeId);
            nodeId = pbvh.nodes[static_cast<size_t>(nodeId)].parent_id;
        }
    }

    for (auto it = dirtyChain.rbegin(); it != dirtyChain.rend(); ++it) {
        const int nodeId = *it;
        if (nodeId < 0 || nodeId >= static_cast<int>(pbvh.nodes.size())) {
            continue;
        }

        auto& node = pbvh.nodes[static_cast<size_t>(nodeId)];
        if (node.is_leaf) {
            node.local_bounds = computeEditableVertexBounds(editableMeshCache, node.vertex_ids);
            node.is_boundary = false;
            for (const int vertexIdInt : node.vertex_ids) {
                if (vertexIdInt >= 0 && vertexIdInt < static_cast<int>(editableMeshCache.vertices.size())) {
                    node.is_boundary = node.is_boundary || editableMeshCache.vertices[static_cast<size_t>(vertexIdInt)].is_boundary;
                }
            }
            continue;
        }

        const bool hasLeft = node.left_child_id >= 0 && node.left_child_id < static_cast<int>(pbvh.nodes.size());
        const bool hasRight = node.right_child_id >= 0 && node.right_child_id < static_cast<int>(pbvh.nodes.size());
        if (hasLeft && hasRight) {
            node.local_bounds = surrounding_box(
                pbvh.nodes[static_cast<size_t>(node.left_child_id)].local_bounds,
                pbvh.nodes[static_cast<size_t>(node.right_child_id)].local_bounds);
            node.is_boundary =
                pbvh.nodes[static_cast<size_t>(node.left_child_id)].is_boundary ||
                pbvh.nodes[static_cast<size_t>(node.right_child_id)].is_boundary;
        }
    }
}

void buildSculptPBVH(
    SceneUI::SculptPBVH& pbvh,
    const SceneUI::EditableMeshCache& editableMeshCache) {
    pbvh.nodes.clear();
    pbvh.root_node_id = -1;
    pbvh.max_depth = 0;
    pbvh.leaf_count = 0;
    pbvh.source_vertex_to_leaf_id.assign(editableMeshCache.vertices.size(), -1);
    if (editableMeshCache.vertices.empty()) {
        return;
    }

    pbvh.nodes.reserve(editableMeshCache.vertices.size() * 2);
    std::vector<int> rootVertexIds(editableMeshCache.vertices.size());
    std::iota(rootVertexIds.begin(), rootVertexIds.end(), 0);
    pbvh.root_node_id = buildSculptPBVHRecursive(pbvh, editableMeshCache, rootVertexIds, -1, 0);
}

void refreshSculptPBVH(
    SceneUI::SculptPBVH& pbvh,
    const SceneUI::EditableMeshCache& editableMeshCache) {
    if (pbvh.root_node_id < 0 || pbvh.nodes.empty()) {
        return;
    }
    refreshSculptPBVHNodeBounds(pbvh, editableMeshCache, pbvh.root_node_id);
}

std::vector<int> collectSculptPBVHCandidateVertices(
    SceneUI::SculptPBVH& pbvh,
    const SceneUI::EditableMeshCache& editableMeshCache,
    const Vec3& localCenter,
    float localRadius) {
    std::vector<int> candidateVertexIds;
    pbvh.last_candidate_node_count = 0;
    pbvh.last_candidate_vertex_count = 0;
    if (pbvh.root_node_id < 0 || pbvh.nodes.empty()) {
        return candidateVertexIds;
    }

    std::vector<int> stack = { pbvh.root_node_id };
    while (!stack.empty()) {
        const int nodeId = stack.back();
        stack.pop_back();
        if (nodeId < 0 || nodeId >= static_cast<int>(pbvh.nodes.size())) {
            continue;
        }

        const auto& node = pbvh.nodes[static_cast<size_t>(nodeId)];
        if (!pbvhNodeIntersectsSphere(node, localCenter, localRadius)) {
            continue;
        }

        ++pbvh.last_candidate_node_count;
        if (node.is_leaf) {
            candidateVertexIds.insert(
                candidateVertexIds.end(),
                node.vertex_ids.begin(),
                node.vertex_ids.end());
            continue;
        }

        if (node.left_child_id >= 0) {
            stack.push_back(node.left_child_id);
        }
        if (node.right_child_id >= 0) {
            stack.push_back(node.right_child_id);
        }
    }

    std::sort(candidateVertexIds.begin(), candidateVertexIds.end());
    candidateVertexIds.erase(
        std::unique(candidateVertexIds.begin(), candidateVertexIds.end()),
        candidateVertexIds.end());
    pbvh.last_candidate_vertex_count = candidateVertexIds.size();
    return candidateVertexIds;
}

template <typename Fn>
void forEachEditableCandidate(const std::vector<int>& candidateVertexIds, Fn&& fn) {
    constexpr size_t kEditableParallelThreshold = 96u;
    if (candidateVertexIds.size() >= kEditableParallelThreshold) {
        std::for_each(
            std::execution::par_unseq,
            candidateVertexIds.begin(),
            candidateVertexIds.end(),
            std::forward<Fn>(fn));
        return;
    }

    std::for_each(
        candidateVertexIds.begin(),
        candidateVertexIds.end(),
        std::forward<Fn>(fn));
}

bool tryMarkEditableTriangleTouched(
    SceneUI::EditableMeshCache& editableMeshCache,
    const Triangle* triangle);

std::vector<size_t> collectAffectedEditableVertexIds(
    SceneUI::EditableMeshCache& editableMeshCache,
    const std::vector<std::shared_ptr<Triangle>>& touchedTriangles) {
    std::vector<size_t> affectedVertexIds;
    if (touchedTriangles.empty() || editableMeshCache.vertices.empty()) {
        return affectedVertexIds;
    }

    if (editableMeshCache.vertex_mark_stamps.size() != editableMeshCache.vertices.size()) {
        editableMeshCache.vertex_mark_stamps.assign(editableMeshCache.vertices.size(), 0u);
        editableMeshCache.vertex_mark_generation = 1u;
    }
    if (editableMeshCache.vertex_mark_generation == std::numeric_limits<uint32_t>::max()) {
        std::fill(
            editableMeshCache.vertex_mark_stamps.begin(),
            editableMeshCache.vertex_mark_stamps.end(),
            0u);
        editableMeshCache.vertex_mark_generation = 1u;
    }

    const uint32_t currentGeneration = editableMeshCache.vertex_mark_generation++;
    affectedVertexIds.reserve(touchedTriangles.size() * 2);

    for (const auto& tri : touchedTriangles) {
        if (!tri) {
            continue;
        }

        const int faceIdx = editableFaceIndexOf(editableMeshCache, tri.get());
        if (faceIdx < 0) {
            continue;
        }
        const SceneUI::EditableFace& face = editableMeshCache.faces[static_cast<size_t>(faceIdx)];
        const int faceVertexIds[3] = { face.v0, face.v1, face.v2 };

        for (const int vertexIdInt : faceVertexIds) {
            if (vertexIdInt < 0) {
                continue;
            }

            const size_t vertexId = static_cast<size_t>(vertexIdInt);
            if (vertexId >= editableMeshCache.vertices.size() ||
                editableMeshCache.vertex_mark_stamps[vertexId] == currentGeneration) {
                continue;
            }

            editableMeshCache.vertex_mark_stamps[vertexId] = currentGeneration;
            affectedVertexIds.push_back(vertexId);
        }
    }

    return affectedVertexIds;
}

std::vector<int> expandEditableCandidateVerticesByTopology(
    SceneUI::EditableMeshCache& editableMeshCache,
    const std::vector<int>& seedVertexIds,
    int ringCount) {
    std::vector<int> expandedVertexIds;
    if (seedVertexIds.empty() || ringCount <= 0 || editableMeshCache.vertices.empty()) {
        return expandedVertexIds;
    }

    if (editableMeshCache.vertex_mark_stamps.size() != editableMeshCache.vertices.size()) {
        editableMeshCache.vertex_mark_stamps.assign(editableMeshCache.vertices.size(), 0u);
        editableMeshCache.vertex_mark_generation = 1u;
    }
    if (editableMeshCache.vertex_mark_generation == std::numeric_limits<uint32_t>::max()) {
        std::fill(
            editableMeshCache.vertex_mark_stamps.begin(),
            editableMeshCache.vertex_mark_stamps.end(),
            0u);
        editableMeshCache.vertex_mark_generation = 1u;
    }

    const uint32_t currentGeneration = editableMeshCache.vertex_mark_generation++;
    std::vector<int> frontier;
    frontier.reserve(seedVertexIds.size());
    expandedVertexIds.reserve(seedVertexIds.size() * (ringCount + 1));

    auto tryMarkVertex = [&](int vertexIdInt, std::vector<int>* optionalFrontier) {
        if (vertexIdInt < 0) {
            return;
        }
        const size_t vertexId = static_cast<size_t>(vertexIdInt);
        if (vertexId >= editableMeshCache.vertices.size() ||
            editableMeshCache.vertex_mark_stamps[vertexId] == currentGeneration) {
            return;
        }
        editableMeshCache.vertex_mark_stamps[vertexId] = currentGeneration;
        expandedVertexIds.push_back(vertexIdInt);
        if (optionalFrontier) {
            optionalFrontier->push_back(vertexIdInt);
        }
    };

    for (const int vertexIdInt : seedVertexIds) {
        tryMarkVertex(vertexIdInt, &frontier);
    }

    for (int ring = 0; ring < ringCount && !frontier.empty(); ++ring) {
        std::vector<int> nextFrontier;
        for (const int vertexIdInt : frontier) {
            if (vertexIdInt < 0 ||
                vertexIdInt >= static_cast<int>(editableMeshCache.vertex_neighbors.size())) {
                continue;
            }
            for (const int neighborVertexId : editableMeshCache.vertex_neighbors[static_cast<size_t>(vertexIdInt)]) {
                tryMarkVertex(neighborVertexId, &nextFrontier);
            }
        }
        frontier = std::move(nextFrontier);
    }

    return expandedVertexIds;
}

std::vector<size_t> collectAffectedEditableVertexIdsFromPBVHLeaves(
    SceneUI::EditableMeshCache& editableMeshCache,
    const SceneUI::SculptPBVH& pbvh,
    const std::vector<int>& touchedLeafIds) {
    std::vector<size_t> affectedVertexIds;
    if (touchedLeafIds.empty() || editableMeshCache.vertices.empty() || pbvh.nodes.empty()) {
        return affectedVertexIds;
    }

    if (editableMeshCache.vertex_mark_stamps.size() != editableMeshCache.vertices.size()) {
        editableMeshCache.vertex_mark_stamps.assign(editableMeshCache.vertices.size(), 0u);
        editableMeshCache.vertex_mark_generation = 1u;
    }
    if (editableMeshCache.vertex_mark_generation == std::numeric_limits<uint32_t>::max()) {
        std::fill(
            editableMeshCache.vertex_mark_stamps.begin(),
            editableMeshCache.vertex_mark_stamps.end(),
            0u);
        editableMeshCache.vertex_mark_generation = 1u;
    }

    const uint32_t currentGeneration = editableMeshCache.vertex_mark_generation++;
    affectedVertexIds.reserve(touchedLeafIds.size() * 64);

    auto tryMarkVertex = [&](int vertexIdInt) {
        if (vertexIdInt < 0) {
            return;
        }
        const size_t vertexId = static_cast<size_t>(vertexIdInt);
        if (vertexId >= editableMeshCache.vertices.size() ||
            editableMeshCache.vertex_mark_stamps[vertexId] == currentGeneration) {
            return;
        }
        editableMeshCache.vertex_mark_stamps[vertexId] = currentGeneration;
        affectedVertexIds.push_back(vertexId);
    };

    for (const int leafId : touchedLeafIds) {
        if (leafId < 0 || leafId >= static_cast<int>(pbvh.nodes.size())) {
            continue;
        }
        const auto& leaf = pbvh.nodes[static_cast<size_t>(leafId)];
        for (const int vertexIdInt : leaf.vertex_ids) {
            tryMarkVertex(vertexIdInt);
            if (vertexIdInt >= 0 &&
                vertexIdInt < static_cast<int>(editableMeshCache.vertex_neighbors.size())) {
                for (const int neighborVertexId : editableMeshCache.vertex_neighbors[static_cast<size_t>(vertexIdInt)]) {
                    tryMarkVertex(neighborVertexId);
                }
            }
        }
    }

    return affectedVertexIds;
}

std::vector<int> convertAffectedVertexIdsToIntList(const std::vector<size_t>& affectedVertexIds) {
    std::vector<int> result;
    result.reserve(affectedVertexIds.size());
    for (const size_t vertexId : affectedVertexIds) {
        if (vertexId <= static_cast<size_t>((std::numeric_limits<int>::max)())) {
            result.push_back(static_cast<int>(vertexId));
        }
    }
    return result;
}

void expandTouchedTrianglesFromAffectedVertices(
    SceneUI::EditableMeshCache& editableMeshCache,
    const std::vector<size_t>& affectedVertexIds,
    std::vector<std::shared_ptr<Triangle>>& touchedTriangles) {
    for (const size_t vertexId : affectedVertexIds) {
        if (vertexId >= editableMeshCache.vertices.size()) {
            continue;
        }
        for (const auto& ref : editableMeshCache.vertices[vertexId].refs) {
            Triangle* refTri = editableMeshCache.refTri(ref);
            if (!refTri) {
                continue;
            }
            if (tryMarkEditableTriangleTouched(editableMeshCache, refTri)) {
                touchedTriangles.push_back(editableMeshCache.refTriShared(ref));
            }
        }
    }
}

Vec3 resolveEditableProposedLocalPosition(
    const SceneUI::EditableMeshCache& editableMeshCache,
    const std::vector<Vec3>& proposedPositions,
    const std::vector<int>& updatedVertexIdsSorted,
    int vertexIdInt) {
    if (vertexIdInt < 0) {
        return Vec3(0.0f, 0.0f, 0.0f);
    }
    const size_t vertexId = static_cast<size_t>(vertexIdInt);
    if (vertexId >= editableMeshCache.vertices.size()) {
        return Vec3(0.0f, 0.0f, 0.0f);
    }
    if (std::binary_search(updatedVertexIdsSorted.begin(), updatedVertexIdsSorted.end(), vertexIdInt) &&
        vertexId < proposedPositions.size()) {
        return proposedPositions[vertexId];
    }
    return editableMeshCache.vertices[vertexId].local_position;
}

bool isEditableVertexTopologySafe(
    const SceneUI::EditableMeshCache& editableMeshCache,
    int vertexIdInt,
    const std::vector<Vec3>& proposedPositions,
    const std::vector<int>& updatedVertexIdsSorted) {
    if (vertexIdInt < 0) {
        return false;
    }
    const size_t vertexId = static_cast<size_t>(vertexIdInt);
    if (vertexId >= editableMeshCache.vertices.size()) {
        return false;
    }

    const auto& refs = editableMeshCache.vertices[vertexId].refs;
    for (const auto& ref : refs) {
        // ref.triangle_index IS the face index (refs index into source_triangles 1:1 with
        // faces), so the triangle's vertex ids come straight from faces[] — no hash lookup.
        if (ref.triangle_index < 0 ||
            ref.triangle_index >= static_cast<int>(editableMeshCache.faces.size())) {
            continue;
        }
        const SceneUI::EditableFace& face = editableMeshCache.faces[static_cast<size_t>(ref.triangle_index)];
        const int triVertexIds[3] = { face.v0, face.v1, face.v2 };
        const Vec3 oldP0 = editableMeshCache.vertices[static_cast<size_t>(triVertexIds[0])].local_position;
        const Vec3 oldP1 = editableMeshCache.vertices[static_cast<size_t>(triVertexIds[1])].local_position;
        const Vec3 oldP2 = editableMeshCache.vertices[static_cast<size_t>(triVertexIds[2])].local_position;
        const Vec3 newP0 = resolveEditableProposedLocalPosition(editableMeshCache, proposedPositions, updatedVertexIdsSorted, triVertexIds[0]);
        const Vec3 newP1 = resolveEditableProposedLocalPosition(editableMeshCache, proposedPositions, updatedVertexIdsSorted, triVertexIds[1]);
        const Vec3 newP2 = resolveEditableProposedLocalPosition(editableMeshCache, proposedPositions, updatedVertexIdsSorted, triVertexIds[2]);

        const Vec3 oldNormal = Vec3::cross(oldP1 - oldP0, oldP2 - oldP0);
        const Vec3 newNormal = Vec3::cross(newP1 - newP0, newP2 - newP0);
        const float oldLenSq = oldNormal.length_squared();
        const float newLenSq = newNormal.length_squared();

        const float oldEdge01LenSq = (oldP1 - oldP0).length_squared();
        const float oldEdge12LenSq = (oldP2 - oldP1).length_squared();
        const float oldEdge20LenSq = (oldP0 - oldP2).length_squared();
        const float newEdge01LenSq = (newP1 - newP0).length_squared();
        const float newEdge12LenSq = (newP2 - newP1).length_squared();
        const float newEdge20LenSq = (newP0 - newP2).length_squared();

        // Absolute floors must stay BELOW the mesh's actual triangle scale or they reject
        // every move on dense meshes: at ~8M tris the per-triangle area² (newLenSq) is itself
        // ~1e-12, so a 1e-12 floor froze Draw/Clay entirely (Grab/Stamp skip this guard, which
        // is why they were unaffected). The relative thresholds (shrunk to <0.25%/1% of the
        // ORIGINAL edge/area) already detect a genuine collapse/flip at any density; the
        // absolute term only needs to reject literal zero, so push it far below float noise.
        if (!std::isfinite(newEdge01LenSq) || !std::isfinite(newEdge12LenSq) || !std::isfinite(newEdge20LenSq)) {
            return false;
        }
        if (newEdge01LenSq <= (std::max)(1e-24f, oldEdge01LenSq * 0.0025f) ||
            newEdge12LenSq <= (std::max)(1e-24f, oldEdge12LenSq * 0.0025f) ||
            newEdge20LenSq <= (std::max)(1e-24f, oldEdge20LenSq * 0.0025f)) {
            return false;
        }

        if (!std::isfinite(newLenSq) || newLenSq <= (std::max)(1e-24f, oldLenSq * 0.01f)) {
            return false;
        }

        if (oldLenSq > 1e-24f) {
            const float orientDot = oldNormal.dot(newNormal);
            if (!std::isfinite(orientDot) || orientDot <= oldLenSq * 0.04f) {
                return false;
            }
        }
    }

    return true;
}

bool resolveEditableTopologySafePosition(
    const SceneUI::EditableMeshCache& editableMeshCache,
    int vertexIdInt,
    std::vector<Vec3>& proposedPositions,
    const std::vector<int>& updatedVertexIdsSorted,
    Vec3& outLocalPosition) {
    if (vertexIdInt < 0) {
        return false;
    }

    const size_t vertexId = static_cast<size_t>(vertexIdInt);
    if (vertexId >= editableMeshCache.vertices.size() || vertexId >= proposedPositions.size()) {
        return false;
    }

    const Vec3 originalLocalPosition = editableMeshCache.vertices[vertexId].local_position;
    const Vec3 targetLocalPosition = proposedPositions[vertexId];
    const Vec3 totalDelta = targetLocalPosition - originalLocalPosition;
    const float totalDeltaLenSq = totalDelta.length_squared();
    if (!std::isfinite(totalDeltaLenSq) || totalDeltaLenSq <= 1e-14f) {
        outLocalPosition = originalLocalPosition;
        proposedPositions[vertexId] = originalLocalPosition;
        return false;
    }

    if (isEditableVertexTopologySafe(
            editableMeshCache,
            vertexIdInt,
            proposedPositions,
            updatedVertexIdsSorted)) {
        outLocalPosition = targetLocalPosition;
        return true;
    }

    float blend = 0.5f;
    for (int iteration = 0; iteration < 8; ++iteration) {
        const Vec3 backedOffPosition = sanitizeVec3(
            originalLocalPosition + totalDelta * blend,
            originalLocalPosition);
        proposedPositions[vertexId] = backedOffPosition;
        if (isEditableVertexTopologySafe(
                editableMeshCache,
                vertexIdInt,
                proposedPositions,
                updatedVertexIdsSorted)) {
            outLocalPosition = backedOffPosition;
            return true;
        }
        blend *= 0.5f;
    }

    proposedPositions[vertexId] = originalLocalPosition;
    outLocalPosition = originalLocalPosition;
    return false;
}

// Interior angle (radians) at a triangle corner, from its CURRENT local positions.
// Used to angle-weight the smooth-normal average so the result is independent of how a
// polygon was triangulated: the two triangles of a quad contribute in proportion to the
// angle each subtends at the shared vertex, which cancels the diagonal bias that pure
// equal-weight (unit) face-normal averaging produces (the "the diagonal shades like a
// ridge / quad looks like 2 separate triangle normals" sculpt artifact).
float editableTriangleCornerAngle(const Triangle& tri, int corner) {
    const Vec3& p = tri.vertices[corner].original;
    const Vec3& a = tri.vertices[(corner + 1) % 3].original;
    const Vec3& b = tri.vertices[(corner + 2) % 3].original;
    const Vec3 e1 = a - p;
    const Vec3 e2 = b - p;
    const float l1 = e1.length();
    const float l2 = e2.length();
    if (l1 < 1e-8f || l2 < 1e-8f) {
        return 0.0f;
    }
    const float c = std::clamp(e1.dot(e2) / (l1 * l2), -1.0f, 1.0f);
    return std::acos(c);
}

// Oriented, UN-normalized face normal (|n| ~ 2*area), oriented to match the triangle's stored
// normals. Summing these over a polygon's triangles yields the area-weighted polygon normal —
// the single "loop normal" both triangles of a quad should share so the diagonal disappears.
Vec3 editableTriangleOrientedCross(const Triangle& tri) {
    Vec3 c = Vec3::cross(
        tri.vertices[1].original - tri.vertices[0].original,
        tri.vertices[2].original - tri.vertices[0].original);
    Vec3 ref(0.0f, 0.0f, 0.0f);
    for (int k = 0; k < 3; ++k) {
        if (tri.vertices[k].originalNormal.length_squared() > 1e-8f) ref += tri.vertices[k].originalNormal;
        else if (tri.vertices[k].normal.length_squared() > 1e-8f)    ref += tri.vertices[k].normal;
    }
    if (ref.length_squared() > 1e-8f && c.dot(ref) < 0.0f) c = -c;
    return c;
}

void recomputeEditableSmoothNormals(
    SceneUI::EditableMeshCache& editableMeshCache,
    const std::vector<size_t>& affectedVertexIds) {
    if (affectedVertexIds.empty()) {
        return;
    }

    const float clampedAngle = std::clamp(editableMeshCache.auto_smooth_angle_degrees, 1.0f, 180.0f);
    const float autoSmoothThreshold = std::cos(clampedAngle * 3.14159265359f / 180.0f);

    // Phase 1 — snapshot ONE face normal per triangle adjacent to the affected verts,
    // READ-ONLY. The previous version recomputed face normals inside the per-vertex
    // par_unseq WRITE loop, and computeOrientationPreservingFaceNormal reads all three
    // of a triangle's originalNormals as its flip reference — the very field that
    // neighbouring vertices were concurrently overwriting via setOriginalVertexNormal.
    // That data race flipped some face normals to the wrong hemisphere (worst on large/
    // dense affected sets that cross the parallel threshold = the "big brush = scattered
    // black, small brush = fine" artifact), and a manual full re-shade
    // (applyShadingSettingsToTriangles, which snapshots first) is exactly what "fixed"
    // it. Snapshot first here too. One normal per triangle (reference = corner 0, like
    // applyShadingSettingsToTriangles) keeps the flip decision consistent regardless of
    // which incident vertex consumes it. Serial: it is only cross products, and removing
    // the read/write overlap is what matters.
    // Ensure the polygon (quad/ngon) grouping exists — built once per cache lifetime. With it, a
    // vertex's shading normal is averaged from POLYGON (loop) normals (one per quad) instead of
    // the two split-triangle normals, so the triangulation diagonal stops reading as a facet/fold.
    if (!editableMeshCache.polygon_grouping_built) {
        editableMeshCache.buildPolygonGrouping();
    }
    const bool usePolygons = editableMeshCache.has_polygon_grouping;

    auto polygonSlotForRef = [&](const SceneUI::EditableVertexRef& ref) -> int {
        if (!usePolygons) return -1;
        const int triIdx = ref.triangle_index;
        return (triIdx >= 0 && triIdx < static_cast<int>(editableMeshCache.tri_to_polygon.size()))
            ? editableMeshCache.tri_to_polygon[triIdx] : -1;
    };

    std::unordered_map<const Triangle*, Vec3> faceNormals;   // per split-triangle (fallback + orient ref)
    std::unordered_map<int, Vec3> polygonNormals;            // dense polygon slot -> area-weighted unit normal
    faceNormals.reserve(affectedVertexIds.size() * 6 + 1);
    for (const size_t vertexId : affectedVertexIds) {
        if (vertexId >= editableMeshCache.vertices.size()) {
            continue;
        }
        for (const auto& ref : editableMeshCache.vertices[vertexId].refs) {
            const Triangle* triKey = editableMeshCache.refTri(ref);
            if (!triKey) {
                continue;
            }
            if (faceNormals.find(triKey) == faceNormals.end()) {
                faceNormals.emplace(
                    triKey,
                    computeOrientationPreservingFaceNormal(
                        *triKey,
                        triKey->getOriginalVertexNormal(0)));
            }
            const int slot = polygonSlotForRef(ref);
            if (slot >= 0 && polygonNormals.find(slot) == polygonNormals.end()) {
                Vec3 acc(0.0f, 0.0f, 0.0f);
                for (int p = editableMeshCache.polygon_tri_off[slot];
                     p < editableMeshCache.polygon_tri_off[slot + 1]; ++p) {
                    const Triangle* pt = editableMeshCache.triangleAt(editableMeshCache.polygon_tri_data[p]);
                    if (pt) acc += editableTriangleOrientedCross(*pt);
                }
                const float l = acc.length();
                polygonNormals.emplace(slot, l > 1e-12f ? acc / l : faceNormals[triKey]);
            }
        }
    }

    // The shading face normal for a vertex ref: the polygon (loop) normal when the triangle
    // belongs to a multi-triangle polygon, else the plain triangle normal.
    auto normalForRef = [&](const SceneUI::EditableVertexRef& ref, const Triangle* tri) -> Vec3 {
        const int slot = polygonSlotForRef(ref);
        if (slot >= 0) {
            const auto it = polygonNormals.find(slot);
            if (it != polygonNormals.end()) return it->second;
        }
        const auto fit = faceNormals.find(tri);
        return fit != faceNormals.end() ? fit->second : tri->getOriginalVertexNormal(0);
    };

    // Phase 2 — average per affected vertex and write. par_unseq-safe: it reads only the
    // immutable face-normal snapshot and writes only this vertex's own corners.
    auto recomputeVertexNormal = [&](const size_t vertexId) {
        if (vertexId >= editableMeshCache.vertices.size()) {
            return;
        }
        const auto& refs = editableMeshCache.vertices[vertexId].refs;
        if (refs.empty()) {
            return;
        }
        for (const auto& ref : refs) {
            Triangle* selfTri = editableMeshCache.refTri(ref);
            if (!selfTri) {
                continue;
            }
            const Vec3 selfFaceNormal = normalForRef(ref, selfTri);

            // Angle-weight each incident face by the interior angle it subtends at THIS vertex
            // (neighborRef.corner is where this vertex sits in that triangle). With loop normals
            // the contribution is the POLYGON normal, and a diagonal vertex (in both triangles of
            // its quad) sums the two corner angles = the quad's interior angle, so the quad counts
            // once at the right weight regardless of the triangulation.
            Vec3 accumulated(0.0f, 0.0f, 0.0f);
            if (editableMeshCache.shade_flat) {
                // Per-corner the polygon normal → the whole quad flat-shades as one facet.
                accumulated = selfFaceNormal;
            } else if (editableMeshCache.auto_smooth) {
                for (const auto& neighborRef : refs) {
                    const Triangle* neighborTri = editableMeshCache.refTri(neighborRef);
                    if (!neighborTri) {
                        continue;
                    }
                    const Vec3 nN = normalForRef(neighborRef, neighborTri);
                    if (selfFaceNormal.dot(nN) >= autoSmoothThreshold) {
                        accumulated += nN * editableTriangleCornerAngle(*neighborTri, neighborRef.corner);
                    }
                }
            } else {
                for (const auto& neighborRef : refs) {
                    const Triangle* neighborTri = editableMeshCache.refTri(neighborRef);
                    if (!neighborTri) {
                        continue;
                    }
                    const Vec3 nN = normalForRef(neighborRef, neighborTri);
                    accumulated += nN * editableTriangleCornerAngle(*neighborTri, neighborRef.corner);
                }
            }

            const float len = accumulated.length();
            const Vec3 shadingNormal = len > 1e-8f ? accumulated / len : selfFaceNormal;
            selfTri->setOriginalVertexNormal(ref.corner, shadingNormal);
        }
    };

    constexpr size_t kSmoothNormalParallelThreshold = 128u;
    if (affectedVertexIds.size() >= kSmoothNormalParallelThreshold) {
        std::for_each(
            std::execution::par_unseq,
            affectedVertexIds.begin(),
            affectedVertexIds.end(),
            recomputeVertexNormal);
    } else {
        std::for_each(
            affectedVertexIds.begin(),
            affectedVertexIds.end(),
            recomputeVertexNormal);
    }
}

void applyShadingSettingsToTriangles(
    const std::vector<std::shared_ptr<Triangle>>& triangles,
    bool flatShading,
    bool autoSmooth,
    float autoSmoothAngleDegrees) {
    if (triangles.empty()) {
        return;
    }
    MESH_PROFILE_SCOPE("applyShadingSettingsToTriangles (" + std::to_string(triangles.size()) + " tris)");

    struct TriangleCornerRef {
        std::shared_ptr<Triangle> triangle;
        int corner = 0;
        Vec3 faceNormal;
    };

    // Per-polygon (loop) normals: group triangles by Triangle::faceIndex so BOTH triangles of a
    // quad share ONE area-weighted normal (the triangulation diagonal stops showing). faceIndex
    // < 0 = the triangle is its own polygon (per-triangle fallback for non-subdivided meshes).
    std::unordered_map<int, Vec3> polygonNormalByFace;
    polygonNormalByFace.reserve(triangles.size());
    for (const auto& tri : triangles) {
        if (!tri || tri->getFaceIndex() < 0) {
            continue;
        }
        polygonNormalByFace[tri->getFaceIndex()] += editableTriangleOrientedCross(*tri);
    }
    for (auto& kv : polygonNormalByFace) {
        const float l = kv.second.length();
        if (l > 1e-12f) kv.second = kv.second / l;
    }

    std::unordered_map<QuantizedVertexKey, std::vector<TriangleCornerRef>, QuantizedVertexKeyHasher> refsByVertex;
    refsByVertex.reserve(triangles.size() * 2 + 1);

    for (const auto& tri : triangles) {
        if (!tri) {
            continue;
        }

        const Vec3 flatNormal = computeOrientationPreservingFaceNormal(
            *tri,
            tri->getOriginalVertexNormal(0));
        // Shade by the polygon normal when this triangle belongs to a quad/ngon, else its own.
        Vec3 faceNormal = flatNormal;
        if (tri->getFaceIndex() >= 0) {
            const auto it = polygonNormalByFace.find(tri->getFaceIndex());
            if (it != polygonNormalByFace.end() && it->second.length_squared() > 1e-8f) {
                faceNormal = it->second;
            }
        }

        for (int corner = 0; corner < 3; ++corner) {
            refsByVertex[quantizeTopologyVertex(tri->getOriginalVertexPosition(corner))].push_back(
                TriangleCornerRef{ tri, corner, faceNormal });
        }
    }

    const float clampedAngle = std::clamp(autoSmoothAngleDegrees, 1.0f, 180.0f);
    const float autoSmoothThreshold = std::cos(clampedAngle * 3.14159265359f / 180.0f);

    for (auto& bucket : refsByVertex) {
        auto& refs = bucket.second;
        for (size_t i = 0; i < refs.size(); ++i) {
            // Angle-weighted average (Thürmer-Wüthrich): each incident face contributes in
            // proportion to the interior angle it subtends at this vertex, so the result is
            // independent of how a polygon was triangulated (a quad shades as one polygon
            // instead of two facets along its diagonal). Equal weighting biased the normal
            // toward whichever side of the diagonal had more triangles.
            Vec3 accumulated(0.0f, 0.0f, 0.0f);
            if (flatShading) {
                accumulated = refs[i].faceNormal;
            } else if (autoSmooth) {
                for (size_t j = 0; j < refs.size(); ++j) {
                    if (refs[i].faceNormal.dot(refs[j].faceNormal) >= autoSmoothThreshold) {
                        accumulated += refs[j].faceNormal * editableTriangleCornerAngle(*refs[j].triangle, refs[j].corner);
                    }
                }
            } else {
                for (const auto& ref : refs) {
                    accumulated += ref.faceNormal * editableTriangleCornerAngle(*ref.triangle, ref.corner);
                }
            }

            const float len = accumulated.length();
            const Vec3 shadingNormal = len > 1e-8f ? accumulated / len : refs[i].faceNormal;
            refs[i].triangle->setOriginalVertexNormal(refs[i].corner, shadingNormal);
        }
    }

    for (const auto& tri : triangles) {
        if (!tri) {
            continue;
        }
        tri->set_normals(
            tri->getOriginalVertexNormal(0),
            tri->getOriginalVertexNormal(1),
            tri->getOriginalVertexNormal(2));
        tri->updateTransformedVertices();
    }
}

// Sculpt entry needs NON-DEGENERATE shading normals (a dense / subdivided mesh can
// enter with per-vertex normals that collapsed to zero during modifier averaging →
// black shading + the brush disc falling back to world-up). The old entry fix re-ran
// applyShadingSettingsToTriangles, but that REWRITES every normal from the per-object
// MeshShadingSettings — which defaults to auto-smooth — so it silently flipped flat /
// custom-shaded meshes to smooth on every sculpt entry (user complaint: "stays in
// whatever shading mode it already has"). This repair touches ONLY broken normals:
// any valid normal (i.e. the object's real shading, flat or smooth) is left exactly as
// it is, while zero / non-finite corners are filled from the average of the VALID
// normals sharing that position (orientation-preserving face normal if none are valid).
// Returns true if it changed anything.
bool repairDegenerateShadingNormals(
    const std::vector<std::shared_ptr<Triangle>>& triangles) {
    if (triangles.empty()) {
        return false;
    }
    auto isDegenerate = [](const Vec3& n) {
        const float lenSq = n.length_squared();
        return !std::isfinite(lenSq) || lenSq <= 1e-12f;
    };

    // Fast out: if every normal is valid, leave the object's shading completely alone.
    bool anyDegenerate = false;
    for (const auto& tri : triangles) {
        if (!tri) {
            continue;
        }
        for (int corner = 0; corner < 3 && !anyDegenerate; ++corner) {
            if (isDegenerate(tri->getOriginalVertexNormal(corner))) {
                anyDegenerate = true;
            }
        }
        if (anyDegenerate) {
            break;
        }
    }
    if (!anyDegenerate) {
        return false;
    }

    struct CornerRef {
        std::shared_ptr<Triangle> triangle;
        int corner = 0;
    };
    std::unordered_map<QuantizedVertexKey, std::vector<CornerRef>, QuantizedVertexKeyHasher> byVertex;
    std::unordered_map<QuantizedVertexKey, Vec3, QuantizedVertexKeyHasher> validNormalSum;
    byVertex.reserve(triangles.size() * 2 + 1);

    for (const auto& tri : triangles) {
        if (!tri) {
            continue;
        }
        for (int corner = 0; corner < 3; ++corner) {
            const QuantizedVertexKey key = quantizeTopologyVertex(tri->getOriginalVertexPosition(corner));
            byVertex[key].push_back(CornerRef{ tri, corner });
            const Vec3 n = tri->getOriginalVertexNormal(corner);
            if (!isDegenerate(n)) {
                validNormalSum[key] += n;
            }
        }
    }

    for (auto& bucket : byVertex) {
        const auto sumIt = validNormalSum.find(bucket.first);
        const Vec3 fill = sumIt != validNormalSum.end() ? sumIt->second : Vec3(0.0f, 0.0f, 0.0f);
        const float fillLen = fill.length();
        for (const auto& ref : bucket.second) {
            if (!isDegenerate(ref.triangle->getOriginalVertexNormal(ref.corner))) {
                continue; // keep valid normals → preserves the object's existing shading
            }
            Vec3 repaired;
            if (fillLen > 1e-8f) {
                repaired = fill / fillLen; // smooth-blend from the valid normals here
            } else {
                repaired = computeOrientationPreservingFaceNormal(
                    *ref.triangle, ref.triangle->getOriginalVertexNormal(ref.corner));
            }
            ref.triangle->setOriginalVertexNormal(ref.corner, repaired);
        }
    }

    for (const auto& tri : triangles) {
        if (!tri) {
            continue;
        }
        tri->set_normals(
            tri->getOriginalVertexNormal(0),
            tri->getOriginalVertexNormal(1),
            tri->getOriginalVertexNormal(2));
        tri->updateTransformedVertices();
    }
    return true;
}

bool tryMarkEditableTriangleTouched(
    SceneUI::EditableMeshCache& editableMeshCache,
    const Triangle* triangle) {
    if (!triangle) {
        return false;
    }

    const int faceIdx = editableFaceIndexOf(editableMeshCache, triangle);
    if (faceIdx < 0) {
        return false;
    }

    if (editableMeshCache.triangle_mark_stamps.size() != editableMeshCache.faces.size()) {
        editableMeshCache.triangle_mark_stamps.assign(editableMeshCache.faces.size(), 0u);
        editableMeshCache.triangle_mark_generation = 1u;
    }
    if (editableMeshCache.triangle_mark_generation == std::numeric_limits<uint32_t>::max()) {
        std::fill(
            editableMeshCache.triangle_mark_stamps.begin(),
            editableMeshCache.triangle_mark_stamps.end(),
            0u);
        editableMeshCache.triangle_mark_generation = 1u;
    }

    const size_t triangleIndex = static_cast<size_t>(faceIdx);
    const uint32_t currentGeneration = editableMeshCache.triangle_mark_generation;
    if (triangleIndex >= editableMeshCache.triangle_mark_stamps.size()) {
        return false;
    }
    if (editableMeshCache.triangle_mark_stamps[triangleIndex] == currentGeneration) {
        return false;
    }

    editableMeshCache.triangle_mark_stamps[triangleIndex] = currentGeneration;
    return true;
}

void beginEditableTriangleTouchPass(SceneUI::EditableMeshCache& editableMeshCache) {
    if (editableMeshCache.triangle_mark_stamps.size() != editableMeshCache.faces.size()) {
        editableMeshCache.triangle_mark_stamps.assign(editableMeshCache.faces.size(), 0u);
        editableMeshCache.triangle_mark_generation = 1u;
    }
    if (editableMeshCache.triangle_mark_generation == std::numeric_limits<uint32_t>::max()) {
        std::fill(
            editableMeshCache.triangle_mark_stamps.begin(),
            editableMeshCache.triangle_mark_stamps.end(),
            0u);
        editableMeshCache.triangle_mark_generation = 1u;
    } else {
        ++editableMeshCache.triangle_mark_generation;
    }
}

// EXPERIMENTAL (2026-06-13): incremental Vulkan RT BLAS refit for editable/sculpt
// meshes. Re-enabled after making the refit ORDER-SAFE: the solo-mesh BLAS now records
// its build-order triangle pointers (m_soloBlasBuildTriangles) so updateMeshBLASPartial
// uploads vertex positions into the exact BLAS slots — previously the editable mesh_cache
// order could differ from the scene-graph build order and scatter positions (corruption).
// Space/TLAS-transform handling already matched (local-space BLAS + object TLAS transform).
// If this corrupts transformed/imported edit meshes in Vulkan RT Rendered mode, set back
// to false and rebuild — the full-rebuild path is the safe fallback (refit returns false
// on any identity/count mismatch).
constexpr bool kEnableVulkanInteractiveMeshRtUpdates = true;

} // namespace

// Defined OUTSIDE the anonymous namespace above: it is a member of SceneUI::EditableMeshCache,
// which a member-of-an-unnamed-namespace definition would be ill-formed for. Groups source
// triangles by Triangle::faceIndex so sculpt/shading get one normal per polygon (loop normals).
void SceneUI::EditableMeshCache::buildPolygonGrouping() {
    polygon_grouping_built = true;
    polygon_tri_off.clear();
    polygon_tri_data.clear();
    tri_to_polygon.assign(source_triangles.size(), -1);
    has_polygon_grouping = false;

    // Group source triangles by Triangle::faceIndex (the source polygon id). Sparse face ids
    // are remapped to dense slots. faceIndex < 0 = no polygon (per-triangle fallback).
    std::unordered_map<int, int> faceToSlot;
    faceToSlot.reserve(source_triangles.size());
    std::vector<std::vector<int>> slots;
    for (size_t t = 0; t < source_triangles.size(); ++t) {
        const Triangle* tri = source_triangles[t].get();
        if (!tri || tri->getFaceIndex() < 0) continue;
        auto it = faceToSlot.find(tri->getFaceIndex());
        int slot;
        if (it == faceToSlot.end()) {
            slot = static_cast<int>(slots.size());
            faceToSlot.emplace(tri->getFaceIndex(), slot);
            slots.emplace_back();
        } else {
            slot = it->second;
        }
        slots[static_cast<size_t>(slot)].push_back(static_cast<int>(t));
        tri_to_polygon[t] = slot;
    }

    // A grouping only helps when some polygon spans >1 triangle (a quad/ngon). If every face id
    // is unique it would just reproduce per-triangle normals — leave it off (cheap fallback).
    bool anyMulti = false;
    for (const auto& s : slots) { if (s.size() > 1) { anyMulti = true; break; } }

    if (!anyMulti) {
        std::fill(tri_to_polygon.begin(), tri_to_polygon.end(), -1);
        return;
    }

    polygon_tri_off.reserve(slots.size() + 1);
    polygon_tri_off.push_back(0);
    for (const auto& s : slots) {
        polygon_tri_data.insert(polygon_tri_data.end(), s.begin(), s.end());
        polygon_tri_off.push_back(static_cast<int>(polygon_tri_data.size()));
    }
    has_polygon_grouping = true;
}

bool SceneUI::refineSculptHitWithPBVH(const Ray& ray, const std::string& objectName,
                                      HitRecord& hit, bool didHit) {
    if (sculpt_pbvh.object_name != objectName || sculpt_pbvh.nodes.empty()) {
        return didHit;
    }
    const Matrix4x4 objXf = getEditableObjectTransform(editable_mesh_cache);
    const Matrix4x4 invXf = objXf.inverse();
    const Vec3 localO = invXf.transform_point(ray.origin);
    const Vec3 localD = invXf.transform_vector(ray.direction);
    Vec3 localPt, localN;
    Triangle* pbvhTri = nullptr;
    if (!raycastSculptPBVHLocal(sculpt_pbvh, editable_mesh_cache, localO, localD,
                                localPt, localN, pbvhTri)) {
        return didHit;
    }
    const Vec3 worldPt = objXf.transform_point(localPt);
    const float pbvhDist = (worldPt - ray.origin).dot(ray.direction);
    const float sceneDist = didHit ? (hit.point - ray.origin).dot(ray.direction) : 1e30f;
    if (pbvhDist <= 0.0f || pbvhDist >= sceneDist - 1e-5f) {
        return didHit;
    }
    hit.point = worldPt;
    hit.triangle = pbvhTri;
    const Vec3 worldN = invXf.transpose().transform_vector(localN);
    hit.interpolated_normal = Vec3(0.0f, 0.0f, 0.0f);
    hit.normal = computeStableSculptHitNormal(hit, safeNormalizeVec3(worldN, hit.normal));
    hit.interpolated_normal = hit.normal;
    return true;
}

bool SceneUI::isInteractiveSubdivisionPreviewActiveForObject(const std::string& objectName) const {
    return interactive_subdiv_preview_active &&
        !objectName.empty() &&
        interactive_subdiv_preview_object_name == objectName;
}

void SceneUI::beginInteractiveSubdivisionPreview(const std::string& objectName) {
    if (objectName.empty()) {
        return;
    }
    interactive_subdiv_preview_active = true;
    interactive_subdiv_preview_object_name = objectName;
}

void SceneUI::endInteractiveSubdivisionPreview(UIContext& ctx, const std::string& objectName, bool rebuildFull) {
    const bool wasActive = isInteractiveSubdivisionPreviewActiveForObject(objectName);
    if (wasActive) {
        interactive_subdiv_preview_active = false;
        interactive_subdiv_preview_object_name.clear();
    }

    if (wasActive && rebuildFull && !objectName.empty()) {
        refreshEditableDisplayMeshFromBase(ctx, objectName, true);
    }
}

void SceneUI::clearEditableMeshSelection() {
    editable_mesh_cache.selection = EditableMeshSelection{};
}

void SceneUI::resetMeshEditState(UIContext& ctx) {
    const std::string previewObjectName = active_mesh_edit_object_name;
    if (!previewObjectName.empty()) {
        interactive_subdiv_preview_active = false;
        interactive_subdiv_preview_object_name.clear();
        refreshEditableDisplayMeshFromBase(ctx, previewObjectName, false);
    }
    mesh_overlay_settings.enabled = false;
    mesh_overlay_settings.edit_mode = false;
    sculpt_mode_state.enabled = false;
    sculpt_mode_state.active_target_name.clear();
    terrain_sculpt_proxy_active = false;
    mesh_workspace_mode = MeshWorkspaceMode::Edit;
    ctx.selection.mesh_element_mode = MeshElementSelectMode::Object;
    clearEditableMeshSelection();
    mesh_edit_layer = MeshEditLayer{};
    pending_serialized_mesh_edit_layer = PendingSerializedMeshEditLayer{};
    active_mesh_edit_object_name.clear();
    active_mesh_edit_object_ptr = nullptr;
    mesh_edit_gpu_sync_pending = false;
    mesh_edit_gpu_sync_object_name.clear();
    mesh_edit_optix_targeted_sync_enabled = true;
    interactive_subdiv_preview_active = false;
    interactive_subdiv_preview_object_name.clear();
    subdiv_preview_refresh_valid = false;
    subdiv_preview_refresh_object_name.clear();
    sculpt_stroke_state = SculptStrokeState{};

    // Flush a deferred (lazy) sculpt CPU-BVH refit on the way out of edit/sculpt mode.
    // During GPU-viewport sculpting the whole-mesh refit is postponed (g_cpu_bvh_stale)
    // so brush release never freezes; the pick fallback normally consumes it, but if the
    // user simply leaves sculpt without picking, settle the debt here so the picking /
    // CPU-render BVH is current for whatever they do next.
    {
        extern bool g_cpu_bvh_refit_pending;
        extern bool g_cpu_bvh_stale;
        if (g_cpu_bvh_stale) {
            g_cpu_bvh_stale = false;
            g_cpu_bvh_refit_pending = true;
        }
    }
}

void SceneUI::syncMeshEditState(UIContext& ctx) {
    const bool sculptWorkspaceActive =
        mesh_workspace_mode == MeshWorkspaceMode::Sculpt &&
        sculpt_mode_state.enabled;

    std::string currentObjectName;
    const Triangle* currentObjectPtr = nullptr;
    if (ctx.selection.selected.type == SelectableType::Object && ctx.selection.selected.object) {
        currentObjectName = ctx.selection.selected.object->getNodeName();
        currentObjectPtr = ctx.selection.selected.object.get();
    }

    if (!mesh_overlay_settings.enabled && !sculptWorkspaceActive) {
        if (!active_mesh_edit_object_name.empty() || mesh_overlay_settings.edit_mode ||
            ctx.selection.mesh_element_mode != MeshElementSelectMode::Object) {
            resetMeshEditState(ctx);
        }
        return;
    }

    if (sculptWorkspaceActive) {
        if (!currentObjectName.empty()) {
            active_mesh_edit_object_name = currentObjectName;
            active_mesh_edit_object_ptr = currentObjectPtr;
            sculpt_mode_state.active_target_name = currentObjectName;
        } else if (!sculpt_mode_state.active_target_name.empty()) {
            active_mesh_edit_object_name = sculpt_mode_state.active_target_name;
            active_mesh_edit_object_ptr = nullptr;
        }
        return;
    }

    if (mesh_overlay_settings.edit_mode &&
        mesh_workspace_mode == MeshWorkspaceMode::Edit &&
        ctx.selection.mesh_element_mode == MeshElementSelectMode::Object) {
        ctx.selection.mesh_element_mode = MeshElementSelectMode::Vertex;
        clearEditableMeshSelection();
    }

    if (active_mesh_edit_object_name.empty()) {
        if (currentObjectName.empty()) {
            if (mesh_overlay_settings.edit_mode ||
                ctx.selection.mesh_element_mode != MeshElementSelectMode::Object) {
                resetMeshEditState(ctx);
            }
            return;
        }

        active_mesh_edit_object_name = currentObjectName;
        active_mesh_edit_object_ptr = currentObjectPtr;
        return;
    }

    if (mesh_overlay_settings.edit_mode &&
        mesh_workspace_mode == MeshWorkspaceMode::Edit &&
        isSubdivisionPreviewActive(ctx, active_mesh_edit_object_name)) {
        // Interactive vertex drag drives its own live preview from
        // applySelectedMeshElementTranslation / endInteractiveSubdivisionPreview, so skip
        // here while a drag is in flight. Outside a drag, re-evaluate ONLY when the
        // subdivision params or cage mesh actually changed — and when they do, queue a GPU
        // sync so Rendered-mode backends (Vulkan RT / OptiX) refresh too. Previously this
        // ran every frame with queueGpuSync=false, which kept the CPU BVH current but left
        // the GPU object drawn in its old (un-subdivided) state until some other op forced
        // a backend rebuild/refit.
        if (!isInteractiveSubdivisionPreviewActiveForObject(active_mesh_edit_object_name)) {
            const std::size_t signature =
                computeSubdivisionPreviewSignature(ctx, active_mesh_edit_object_name);
            if (!subdiv_preview_refresh_valid ||
                subdiv_preview_refresh_object_name != active_mesh_edit_object_name ||
                subdiv_preview_refresh_signature != signature) {
                subdiv_preview_refresh_valid = true;
                subdiv_preview_refresh_object_name = active_mesh_edit_object_name;
                subdiv_preview_refresh_signature = signature;
                refreshEditableDisplayMeshFromBase(ctx, active_mesh_edit_object_name, true);
            }
        }
    }

    auto activeCacheIt = mesh_cache.find(active_mesh_edit_object_name);
    const bool activeObjectStillValid =
        activeCacheIt != mesh_cache.end() && !activeCacheIt->second.empty();
    if (!activeObjectStillValid) {
        resetMeshEditState(ctx);
        return;
    }

    const bool selectedMatchesActive =
        ctx.selection.selected.type == SelectableType::Object &&
        ctx.selection.selected.object &&
        ctx.selection.selected.object->getNodeName() == active_mesh_edit_object_name;
    if (!selectedMatchesActive && !activeCacheIt->second.empty() && activeCacheIt->second.front().second) {
        ctx.selection.selectObject(activeCacheIt->second.front().second, -1, active_mesh_edit_object_name);
        currentObjectName = active_mesh_edit_object_name;
        currentObjectPtr = activeCacheIt->second.front().second.get();
    }

    if (currentObjectName == active_mesh_edit_object_name) {
        active_mesh_edit_object_ptr = currentObjectPtr;
    }
}

void SceneUI::captureMeshEditLayerState(UIContext& ctx, const std::string& objectName, std::vector<MeshEditTriangleState>& outStates) {
    outStates.clear();
    const auto stackIt = ctx.scene.mesh_modifiers.find(objectName);
    const bool useBaseMesh =
        stackIt != ctx.scene.mesh_modifiers.end() &&
        hasEnabledSubdivisionPreview(stackIt->second) &&
        ctx.scene.base_mesh_cache.find(objectName) != ctx.scene.base_mesh_cache.end();

    if (useBaseMesh) {
        const auto baseIt = ctx.scene.base_mesh_cache.find(objectName);
        if (baseIt == ctx.scene.base_mesh_cache.end()) {
            return;
        }

        outStates.reserve(baseIt->second.size());
        for (const auto& tri : baseIt->second) {
            if (!tri) {
                continue;
            }
            MeshEditTriangleState state;
            state.triangle = tri;
            for (int corner = 0; corner < 3; ++corner) {
                state.positions[corner] = tri->getOriginalVertexPosition(corner);
            }
            outStates.push_back(state);
        }
        return;
    }

    auto meshIt = mesh_cache.find(objectName);
    if (meshIt == mesh_cache.end()) {
        return;
    }

    outStates.reserve(meshIt->second.size());
    for (const auto& entry : meshIt->second) {
        if (!entry.second) {
            continue;
        }
        MeshEditTriangleState state;
        state.triangle = entry.second;
        for (int corner = 0; corner < 3; ++corner) {
            state.positions[corner] = entry.second->getOriginalVertexPosition(corner);
        }
        outStates.push_back(state);
    }
}

bool SceneUI::ensureMeshEditLayer(UIContext& ctx, const std::string& objectName) {
    if (objectName.empty()) {
        return false;
    }
    if (!mesh_cache_valid) {
        return false;
    }
    if (mesh_edit_layer.active && mesh_edit_layer.object_name == objectName) {
        return true;
    }

    mesh_edit_layer = MeshEditLayer{};
    mesh_edit_layer.active = true;
    mesh_edit_layer.enabled = true;
    mesh_edit_layer.object_name = objectName;
    captureMeshEditLayerState(ctx, objectName, mesh_edit_layer.base_states);
    mesh_edit_layer.edited_states = mesh_edit_layer.base_states;
    return !mesh_edit_layer.base_states.empty();
}

void SceneUI::refreshMeshEditLayerEditedState(UIContext& ctx) {
    if (!mesh_edit_layer.active || mesh_edit_layer.object_name.empty()) {
        return;
    }
    captureMeshEditLayerState(ctx, mesh_edit_layer.object_name, mesh_edit_layer.edited_states);
}

SceneUI::MeshShadingSettings& SceneUI::ensureMeshShadingSettings(const std::string& objectName) {
    auto [it, inserted] = mesh_shading_settings_by_object.try_emplace(objectName);
    if (inserted) {
        it->second = MeshShadingSettings{};
    }
    return it->second;
}

bool SceneUI::applyMeshShadingSettings(UIContext& ctx, const std::string& objectName, bool queueGpuSync) {
    if (objectName.empty()) {
        return false;
    }
    if (!mesh_cache_valid) {
        rebuildMeshCache(ctx.scene.world.objects);
    }

    auto meshIt = mesh_cache.find(objectName);
    if (meshIt == mesh_cache.end() || meshIt->second.empty()) {
        return false;
    }

    MeshShadingSettings& shading = ensureMeshShadingSettings(objectName);

    std::vector<std::shared_ptr<Triangle>> displayTriangles;
    displayTriangles.reserve(meshIt->second.size());
    std::unordered_set<const Triangle*> seenTriangles;
    seenTriangles.reserve(meshIt->second.size());
    for (const auto& entry : meshIt->second) {
        if (entry.second && seenTriangles.insert(entry.second.get()).second) {
            displayTriangles.push_back(entry.second);
        }
    }
    applyShadingSettingsToTriangles(
        displayTriangles,
        shading.flat_shading,
        shading.auto_smooth,
        shading.auto_smooth_angle_degrees);

    auto baseIt = ctx.scene.base_mesh_cache.find(objectName);
    if (baseIt != ctx.scene.base_mesh_cache.end() && !baseIt->second.empty()) {
        applyShadingSettingsToTriangles(
            baseIt->second,
            shading.flat_shading,
            shading.auto_smooth,
            shading.auto_smooth_angle_degrees);
    }

    if (editable_mesh_cache.object_name == objectName) {
        editable_mesh_cache.shade_flat = shading.flat_shading;
        editable_mesh_cache.auto_smooth = shading.auto_smooth;
        editable_mesh_cache.auto_smooth_angle_degrees = shading.auto_smooth_angle_degrees;
    }
    updateBBoxCache(objectName);
    objects_needing_cpu_sync.erase(objectName);

    if (queueGpuSync && (ctx.backend_ptr != nullptr || g_backend != nullptr || g_viewport_backend != nullptr)) {
        queueMeshEditGpuSync(objectName);
    }

    extern bool g_bvh_rebuild_pending;
    g_bvh_rebuild_pending = true;
    ctx.renderer.resetCPUAccumulation();
    ctx.start_render = true;
    return true;
}

bool SceneUI::repairSculptEntryShadingNormals(UIContext& ctx, const std::string& objectName, bool queueGpuSync) {
    if (objectName.empty()) {
        return false;
    }
    if (!mesh_cache_valid) {
        rebuildMeshCache(ctx.scene.world.objects);
    }

    auto meshIt = mesh_cache.find(objectName);
    if (meshIt == mesh_cache.end() || meshIt->second.empty()) {
        return false;
    }

    std::vector<std::shared_ptr<Triangle>> displayTriangles;
    displayTriangles.reserve(meshIt->second.size());
    std::unordered_set<const Triangle*> seenTriangles;
    seenTriangles.reserve(meshIt->second.size());
    for (const auto& entry : meshIt->second) {
        if (entry.second && seenTriangles.insert(entry.second.get()).second) {
            displayTriangles.push_back(entry.second);
        }
    }

    // Repair only degenerate normals (preserves the object's existing flat/smooth
    // shading), mirroring the GPU/BVH refresh applyMeshShadingSettings does.
    bool changed = repairDegenerateShadingNormals(displayTriangles);

    auto baseIt = ctx.scene.base_mesh_cache.find(objectName);
    if (baseIt != ctx.scene.base_mesh_cache.end() && !baseIt->second.empty()) {
        changed = repairDegenerateShadingNormals(baseIt->second) || changed;
    }

    updateBBoxCache(objectName);
    objects_needing_cpu_sync.erase(objectName);

    if (!changed) {
        return false; // mesh was already clean — nothing flipped, nothing to re-upload
    }

    if (queueGpuSync && (ctx.backend_ptr != nullptr || g_backend != nullptr || g_viewport_backend != nullptr)) {
        queueMeshEditGpuSync(objectName);
    }

    extern bool g_bvh_rebuild_pending;
    g_bvh_rebuild_pending = true;
    ctx.renderer.resetCPUAccumulation();
    ctx.start_render = true;
    return true;
}

void SceneUI::depositWetClay(const std::vector<int>& vertexIds,
                            const std::vector<Vec3>& anchors,
                            float wetnessInject,
                            const std::vector<float>* injectWeights) {
    if (vertexIds.empty()) {
        return;
    }
    const float inject = std::clamp(wetnessInject, 0.0f, 1.0f);
    if (inject <= 0.0f) {
        return;
    }
    ensureSculptWetClaySized(sculpt_wet_clay_state, editable_mesh_cache, sculpt_mode_state.active_target_name);
    auto& wet = sculpt_wet_clay_state;
    const bool haveAnchors = anchors.size() == vertexIds.size();
    // Optional per-vertex feather (1 in the brush core → 0 at the rim). Without it the
    // footprint injects a FLAT wetness, so the brush boundary is a hard wet/dry cliff:
    // fully-mobile clay sits right next to a frozen ring, and at high fluidity (Water,
    // flow≈1) the flow piles material against that ring into a sharp lower ridge. The
    // feather turns the boundary into a gradient so the mobility tapers off smoothly.
    const bool haveWeights = injectWeights && injectWeights->size() == vertexIds.size();
    for (size_t k = 0; k < vertexIds.size(); ++k) {
        const int vid = vertexIds[k];
        if (vid < 0 || static_cast<size_t>(vid) >= wet.wetness.size()) {
            continue;
        }
        const size_t vIndex = static_cast<size_t>(vid);
        const float vInject = haveWeights
            ? inject * std::clamp((*injectWeights)[k], 0.0f, 1.0f)
            : inject;
        // Below the evict threshold the vertex would dry out the very next frame, so
        // adding it just churns the active set — skip it and leave the rim un-wetted.
        if (vInject < 0.02f) {
            continue;
        }
        float& w = wet.wetness[vIndex];
        if (w <= 1e-4f) {
            // Newly wet → joins the active set; anchor the flow at the PRE-deposit surface
            // (the wall) and capture its normal so the deposited protrusion can flow down it.
            wet.active_list.push_back(vid);
            wet.flow_anchor[vIndex] = haveAnchors ? anchors[k] : editable_mesh_cache.vertex_positions[vIndex];
            wet.flow_normal[vIndex] = editableVertexNormal(editable_mesh_cache, vIndex);
        }
        w = (std::max)(w, vInject);
    }
    wet.has_any = !wet.active_list.empty();
}

void SceneUI::stepWetClayField(UIContext& ctx) {
    auto& wet = sculpt_wet_clay_state;
    const bool sculptActive =
        sculpt_mode_state.enabled &&
        mesh_workspace_mode == MeshWorkspaceMode::Sculpt &&
        !sculpt_mode_state.active_target_name.empty();
    if (!sculptActive || !wet.has_any || wet.active_list.empty()) {
        return;
    }
    const std::string& objectName = sculpt_mode_state.active_target_name;
    EditableMeshCache& cache = editable_mesh_cache;
    if (cache.object_name != objectName || cache.vertices.empty()) {
        return;
    }
    ensureSculptWetClaySized(wet, cache, objectName);
    if (wet.active_list.empty()) {
        wet.has_any = false;
        return;
    }

    const ImGuiIO& io = ImGui::GetIO();
    const float dt = sanitizeFiniteFloat(io.DeltaTime, 1.0f / 60.0f, 1.0f / 240.0f, 1.0f / 60.0f);
    const float settle = std::clamp(sculpt_mode_state.wet_clay_settle, 0.0f, 1.0f);
    const float flow = std::clamp(sculpt_mode_state.wet_clay_flow, 0.0f, 1.0f);
    const float cohesion = std::clamp(sculpt_mode_state.wet_clay_cohesion, 0.0f, 1.0f);
    const float dryRate = (std::max)(0.0f, sculpt_mode_state.wet_clay_dry_rate);
    const float dryFactor = std::exp(-dryRate * dt);
    constexpr float kEvictThreshold = 0.02f;
    const size_t vertexCount = cache.vertices.size();

    // Object transform (reused for the world-space triangle update at the end). For the
    // gravity flow we need world-down expressed in LOCAL space: a vertex's height along
    // gravity is ~ localPos·localUp (the constant translation cancels in neighbour
    // differences, which is all the flow uses).
    const Matrix4x4 transform = getEditableObjectTransform(cache);
    const Matrix4x4 inverseTransform = transform.inverse();
    const Vec3 localUp = (flow > 0.0f)
        ? safeNormalizeVec3(inverseTransform.transform_vector(Vec3(0.0f, 1.0f, 0.0f)), Vec3(0.0f, 1.0f, 0.0f))
        : Vec3(0.0f, 1.0f, 0.0f);
    // Yield: clay only flows where the local height drop exceeds this (so it holds a
    // shape on gentle slopes like real clay instead of running off like water).
    const float avgEdgeLen = (std::max)(sculpt_control_graph.avg_edge_length, 0.0f);
    const float yieldDrop = (std::max)(0.0f, sculpt_mode_state.wet_clay_yield) * avgEdgeLen;
    const bool hetero = sculpt_mode_state.wet_clay_hetero;
    const float heteroScale = sculpt_mode_state.wet_clay_hetero_scale;

    // Phase 4 — advance the flow FRONT downhill. Each frame, wet the dry direct neighbours
    // that sit BELOW an active wet vertex by more than the yield drop, so the mud actually
    // creeps DOWN the slope (and pools at the bottom) instead of only leveling in place.
    // Bounded: it only spreads downhill, stops where the slope falls below yield, and every
    // wetted vertex dries on its own — one ring per frame keeps the advance controlled.
    if (flow > 0.0f && yieldDrop >= 0.0f) {
        std::vector<int> frontAdds;
        const size_t activeCountBefore = wet.active_list.size();
        for (size_t a = 0; a < activeCountBefore; ++a) {
            const int vid = wet.active_list[a];
            if (vid < 0 || static_cast<size_t>(vid) >= vertexCount) {
                continue;
            }
            const float wi = wet.wetness[static_cast<size_t>(vid)];
            if (wi <= 0.03f) {
                continue; // too dry to keep feeding the front
            }
            const float hi = cache.vertex_positions[static_cast<size_t>(vid)].dot(localUp);
            for (const int nb : cache.vertex_neighbors[static_cast<size_t>(vid)]) {
                if (nb < 0 || static_cast<size_t>(nb) >= vertexCount) {
                    continue;
                }
                if (wet.wetness[static_cast<size_t>(nb)] > 1e-4f) {
                    continue; // already wet/active (also stops double-adds within this pass)
                }
                const float hj = cache.vertex_positions[static_cast<size_t>(nb)].dot(localUp);
                if (hi - hj > yieldDrop) {
                    // Carry MOST of the source wetness into the downhill path so the front
                    // stays wet enough to reach the bottom of a tall/steep surface instead
                    // of drying out partway and piling at the wet/dry boundary. Drying +
                    // the yield gate (flats stop it) still bound the spread. Anchor the new
                    // front vertex at its CURRENT surface (no protrusion yet) so material
                    // flowing into it from above builds a fresh bulge there.
                    const size_t nbI = static_cast<size_t>(nb);
                    wet.wetness[nbI] = (std::max)(wet.wetness[nbI], wi * 0.9f);
                    wet.flow_anchor[nbI] = cache.vertex_positions[nbI];
                    wet.flow_normal[nbI] = editableVertexNormal(cache, nbI);
                    frontAdds.push_back(nb);
                }
            }
        }
        for (const int nb : frontAdds) {
            wet.active_list.push_back(nb);
        }
    }

    // Jacobi snapshot of the ACTIVE verts only (bounded by the wet set, never O(N)). An
    // active neighbour is read from this pre-step snapshot, an inactive one from the
    // committed buffer — so the settle is order-independent and doesn't drift.
    std::unordered_map<int, Vec3> oldPos;
    oldPos.reserve(wet.active_list.size() * 2 + 1);
    for (const int vid : wet.active_list) {
        if (vid >= 0 && static_cast<size_t>(vid) < vertexCount) {
            oldPos.emplace(vid, cache.vertex_positions[static_cast<size_t>(vid)]);
        }
    }

    std::unordered_set<const Triangle*> touchedSet;
    std::vector<std::shared_ptr<Triangle>> touchedTriangles;
    std::vector<size_t> movedVertexIds;
    std::vector<int> survivors;
    movedVertexIds.reserve(wet.active_list.size());
    survivors.reserve(wet.active_list.size());

    for (const int vid : wet.active_list) {
        if (vid < 0 || static_cast<size_t>(vid) >= vertexCount) {
            continue;
        }
        const size_t vIndex = static_cast<size_t>(vid);
        float w = wet.wetness[vIndex];
        if (w <= 0.0f) {
            continue;
        }

        const auto& neighbors = cache.vertex_neighbors[vIndex];
        if (!neighbors.empty()) {
            const auto curIt = oldPos.find(vid);
            const Vec3 cur = curIt != oldPos.end() ? curIt->second : cache.vertex_positions[vIndex];
            const float hCur = cur.dot(localUp);
            const Vec3 nI = wet.flow_normal[vIndex];
            const Vec3 anchorI = wet.flow_anchor[vIndex];
            const float mI = (std::max)(0.0f, (cur - anchorI).dot(nI)); // protrusion above the wall
            const float mobSelf = (flow > 0.0f) ? wetClayFlowMobility(cur, hetero, heteroScale) : 1.0f;

            Vec3 avg(0.0f, 0.0f, 0.0f);
            int n = 0;
            // Per-EDGE symmetric flux. flowRate is small so the TOTAL a vertex sheds to ALL
            // its lower neighbours (rate * degree * m) stays well under its own material m —
            // this is conservative (each edge moves the same amount both ways) and CANNOT
            // create mass. (The runaway spikes were the old version handing EACH lower
            // neighbour the FULL donor material, multiplying it by the neighbour count.)
            const float flowRate = flow * dt * 2.0f;
            float inflow = 0.0f, outflow = 0.0f;
            // Geometric slope-limiter ceiling: the highest this vertex may sit ALONG ITS OWN
            // NORMAL while keeping the slope to EVERY neighbour (wet or dry) under ~kFlowMaxSlope:1.
            // The dry downhill neighbour at the flow front pins this low, so material driven down
            // the wall cannot pile into a vertical cliff/ridge there. (4.4's ceiling only looked at
            // the tallest WET neighbour, which is uphill — it never restrained the front, so the
            // directional pile-up against the dry boundary survived.) Built from neighbour
            // positions, so it is independent of wetness/anchors.
            constexpr float kFlowMaxSlope = 1.5f;
            float slopeCeilN = 1e30f;
            for (const int nb : neighbors) {
                if (nb < 0 || static_cast<size_t>(nb) >= vertexCount) {
                    continue;
                }
                const size_t nbI = static_cast<size_t>(nb);
                const auto it = oldPos.find(nb);
                const Vec3 pj = (it != oldPos.end()) ? it->second : cache.vertex_positions[nbI];
                avg += pj;
                ++n;
                if (flow > 0.0f) {
                    // Slope ceiling from this neighbour: its height along nI plus the lateral
                    // (in-plane) gap times the max slope. min over all neighbours = binding limit.
                    const Vec3 d = pj - cur;
                    const float along = d.dot(nI);
                    const float inPlane = (d - nI * along).length();
                    slopeCeilN = (std::min)(slopeCeilN, pj.dot(nI) + kFlowMaxSlope * inPlane);
                    // Flow transports the NORMAL-protrusion (the deposited material), not the
                    // gravity-height — so it works on a vertical wall, where the bump sticks out
                    // along the normal and has no Y-height variation. Only WET neighbours exchange
                    // material; the front-expansion wets the downhill path ahead so the blob
                    // slides down it. Each edge is scaled by the HIGHER (donor) endpoint's
                    // wetness + mobility + available material → dry/empty clay stops flowing.
                    if (wet.wetness[nbI] > 1e-4f) {
                        const float mJ = (std::max)(0.0f, (pj - wet.flow_anchor[nbI]).dot(wet.flow_normal[nbI]));
                        const float dY = pj.dot(localUp) - hCur; // > 0 = neighbour is higher
                        if (dY > yieldDrop) { // higher neighbour drains its material DOWN into us
                            inflow += flowRate * wet.wetness[nbI] * wetClayFlowMobility(pj, hetero, heteroScale) * mJ;
                        } else if (-dY > yieldDrop) { // we drain our material down into a lower neighbour
                            outflow += flowRate * w * mobSelf * mI;
                        }
                    }
                }
            }
            if (n > 0) {
                avg /= static_cast<float>(n);
                const bool flowing = (flow > 0.0f && (inflow > 0.0f || outflow > 0.0f));
                // Phase 1 — settle (surface tension): full-vector Laplacian toward the
                // neighbour average. Coefficient < 0.5 (stable) and scaled by wetness, so
                // settling fades as the clay dries. Phase 4 cohesion: a FLOWING vertex gets
                // extra smoothing scaled by cohesion, so high cohesion keeps the moving clay
                // a smooth bonded tongue (putty) while low cohesion lets it stay rough and
                // break into chunks (mud), especially with heterogeneous density on.
                const float settleCoef = settle + (flowing ? cohesion * flow : 0.0f);
                const float lambdaSettle = std::clamp(w * settleCoef * dt * 8.0f, 0.0f, 0.5f);
                Vec3 newPos = cur + (avg - cur) * lambdaSettle;
                // Phase 4 — advective gravity flow along the NORMAL: shift this vertex's
                // protrusion by the net (down-in − down-out) flux so the blob descends as a
                // BODY and pools below. Clamp the RESULT to [0, mMax] so material can never go
                // negative or grow into a runaway spike, then move along the normal by the
                // actual delta.
                if (flowing) {
                    const float mMax = (std::max)(avgEdgeLen * 4.0f, 1e-4f);
                    const float targetM = std::clamp(mI + (inflow - outflow), 0.0f, mMax);
                    // Scale the protrusion step by this vertex's own wetness so a drying vertex
                    // tapers its motion to a stop instead of getting one last kick the instant
                    // before it locks. That final kick — fired against neighbours that have
                    // ALREADY dried and frozen — was what flung the thin ribbons/spikes off the
                    // wall (the moving vertex stretched the incident triangles into splinters).
                    newPos += nI * ((targetM - mI) * w);
                    // Geometric slope limiter: if the moved vertex now sits above the
                    // neighbour-derived ceiling along its normal, pull it back down. This is what
                    // stops the directional pile-up — at the flow front the dry downhill neighbour
                    // pins the ceiling, so the descending material spreads down the slope instead
                    // of stacking into the sharp line against the dry boundary. Lets the tongue
                    // descend as a 45°-ish ramp; the residual is smoothed by settle.
                    const float hN = newPos.dot(nI);
                    if (hN > slopeCeilN) {
                        newPos -= nI * (hN - slopeCeilN);
                    }
                }
                // Stretch guard (kills the remaining sharp ribbons): a vertex must not pull away
                // from its neighbour centroid by more than a small multiple of the local edge
                // length. When surrounding verts dry and freeze while this one keeps flowing the
                // incident triangles stretch into thin spikes; clamping the deviation from the
                // centroid keeps the surface coherent (C0) and lets the whole tongue descend as a
                // body rather than spitting out splinters. Cheap, runs only for flowing clay.
                if (flow > 0.0f && avgEdgeLen > 0.0f) {
                    const float maxDev = avgEdgeLen * 2.0f;
                    const Vec3 devFromAvg = newPos - avg;
                    const float devLen = devFromAvg.length();
                    if (devLen > maxDev && devLen > 1e-6f) {
                        newPos = avg + devFromAvg * (maxDev / devLen);
                    }
                }
                if (isFiniteVec3(newPos) &&
                    (newPos - cache.vertex_positions[vIndex]).length_squared() > 1e-16f) {
                    const float maskFactor = sculptMaskFactor(sculpt_mask_state, vIndex);
                    if (maskFactor < 0.999f) {
                        newPos = cache.vertex_positions[vIndex] +
                            (newPos - cache.vertex_positions[vIndex]) * maskFactor;
                    }
                    cache.vertices[vIndex].local_position = newPos;
                    cache.vertex_positions[vIndex] = newPos;
                    if (vIndex < sculpt_updated_local_positions.size()) {
                        sculpt_updated_local_positions[vIndex] = newPos;
                    }
                    movedVertexIds.push_back(vIndex);
                    for (const auto& ref : cache.vertices[vIndex].refs) {
                        Triangle* refTri = cache.refTri(ref);
                        if (!refTri) {
                            continue;
                        }
                        refTri->setOriginalVertexPosition(ref.corner, newPos);
                        refTri->markAABBDirty();
                        if (touchedSet.insert(refTri).second) {
                            touchedTriangles.push_back(cache.refTriShared(ref));
                        }
                    }
                }
            }
        }

        w *= dryFactor;
        if (w < kEvictThreshold) {
            wet.wetness[vIndex] = 0.0f; // dried → locks, leaves the active set
        } else {
            wet.wetness[vIndex] = w;
            survivors.push_back(vid);
        }
    }

    wet.active_list.swap(survivors);
    wet.has_any = !wet.active_list.empty();

    if (touchedTriangles.empty()) {
        return; // only drying bookkeeping happened this frame — nothing to re-sync
    }

    recomputeEditableSmoothNormals(cache, movedVertexIds);

    {
        std::vector<int> movedInts;
        movedInts.reserve(movedVertexIds.size());
        for (const size_t v : movedVertexIds) {
            movedInts.push_back(static_cast<int>(v));
        }
        const std::vector<int> touchedLeafIds =
            collectTouchedSculptPBVHLeafIds(sculpt_pbvh, movedInts);
        if (!touchedLeafIds.empty()) {
            refreshSculptPBVHLeavesAndAncestors(sculpt_pbvh, cache, touchedLeafIds);
        }
    }

    const Matrix4x4 normalTransform = inverseTransform.transpose();
    for (const auto& tri : touchedTriangles) {
        for (int i = 0; i < 3; ++i) {
            tri->vertices[i].position = transform.transform_point(tri->vertices[i].original);
            tri->vertices[i].normal = safeNormalizeVec3(
                normalTransform.transform_vector(tri->vertices[i].originalNormal),
                tri->vertices[i].normal);
        }
        tri->markAABBDirty();
        tri->update_bounding_box();
    }

    const bool hasGpu =
        (ctx.backend_ptr != nullptr || g_backend != nullptr || g_viewport_backend != nullptr);
    if (!hasGpu) {
        sculpt_dirty_mesh_cache_indices.clear();
    }
    sculpt_dirty_mesh_cache_indices.reserve(
        sculpt_dirty_mesh_cache_indices.size() + touchedTriangles.size());
    for (const auto& tri : touchedTriangles) {
        const int faceIdx = editableFaceIndexOf(cache, tri.get());
        if (faceIdx >= 0) {
            sculpt_dirty_mesh_cache_indices.push_back(cache.face_to_mesh_index[static_cast<size_t>(faceIdx)]);
        }
    }
    objects_needing_cpu_sync.erase(objectName);

    extern bool g_cpu_bvh_refit_pending;
    if (!ctx.backend_ptr) {
        g_cpu_bvh_refit_pending = true;
    }
    if (hasGpu) {
        queueMeshEditGpuSync(objectName);
    }
    ctx.renderer.resetCPUAccumulation();
    ctx.start_render = true;
}

void SceneUI::applyCreaseToSelectedEdges(UIContext& ctx, float weight) {
    const std::string objectName = active_mesh_edit_object_name;
    if (objectName.empty()) {
        return;
    }
    if (editable_mesh_cache.object_name != objectName) {
        return;
    }
    // Crease lives on cage edges. Edge / face / vertex selections all map to a set of
    // edges (see collectSelectedCreaseEdgePositions) so Shift+E-style authoring works
    // from any sub-element mode.
    const auto creaseEdges = collectSelectedCreaseEdgePositions(editable_mesh_cache);
    if (creaseEdges.empty()) {
        return;
    }

    // Crease lives in the object's modifier stack (created on demand so it is ready
    // even if the Catmull-Clark modifier is added afterward).
    MeshModifiers::ModifierStack& stack = ctx.scene.mesh_modifiers[objectName];
    for (const auto& edgePair : creaseEdges) {
        stack.setEdgeCrease(edgePair.first, edgePair.second, weight);
    }

    // Re-evaluate the subdivision preview so the new crease is visible immediately.
    // (No-op + returns false if no subdivision preview is enabled on this object; the
    //  crease is still stored and will take effect once a Smooth Subdivision is added.)
    refreshEditableDisplayMeshFromBase(ctx, objectName, ctx.backend_ptr != nullptr);
    ProjectManager::getInstance().markModified();
}

float SceneUI::getSelectedEdgesAverageCrease(UIContext& ctx) const {
    const std::string& objectName = active_mesh_edit_object_name;
    if (objectName.empty() || editable_mesh_cache.object_name != objectName) {
        return -1.0f;
    }
    const auto creaseEdges = collectSelectedCreaseEdgePositions(editable_mesh_cache);
    if (creaseEdges.empty()) {
        return -1.0f;
    }
    const auto stackIt = ctx.scene.mesh_modifiers.find(objectName);
    if (stackIt == ctx.scene.mesh_modifiers.end()) {
        return 0.0f;
    }
    float sum = 0.0f;
    int count = 0;
    for (const auto& edgePair : creaseEdges) {
        sum += stackIt->second.getEdgeCrease(edgePair.first, edgePair.second);
        ++count;
    }
    return count > 0 ? sum / static_cast<float>(count) : -1.0f;
}

bool SceneUI::refreshEditableDisplayMeshFromBase(UIContext& ctx, const std::string& objectName, bool queueGpuSync, bool rebuildEditableCache) {
    if (objectName.empty()) {
        return false;
    }

    const auto baseIt = ctx.scene.base_mesh_cache.find(objectName);
    if (baseIt == ctx.scene.base_mesh_cache.end() || baseIt->second.empty()) {
        return false;
    }

    const auto stackIt = ctx.scene.mesh_modifiers.find(objectName);
    const MeshModifiers::ModifierStack stack =
        (stackIt != ctx.scene.mesh_modifiers.end()) ? stackIt->second : MeshModifiers::ModifierStack{};
    if (!hasEnabledSubdivisionPreview(stack)) {
        return false;
    }

    const MeshModifiers::ModifierStack evalStack =
        buildSubdivisionPreviewEvaluationStack(
            stack,
            isInteractiveSubdivisionPreviewActiveForObject(objectName));
    const std::vector<std::shared_ptr<Triangle>> displayMesh = evaluateDisplayMeshFromBase(baseIt->second, evalStack);
    auto& objects = ctx.scene.world.objects;
    objects.erase(
        std::remove_if(objects.begin(), objects.end(), [&](const auto& obj) {
            return getMeshOverlayNodeName(obj) == objectName;
        }),
        objects.end());

    for (const auto& tri : displayMesh) {
        if (tri) {
            objects.push_back(tri);
        }
    }

    if (!displayMesh.empty() &&
        ((ctx.selection.selected.type == SelectableType::Object &&
          ctx.selection.selected.object &&
          getMeshOverlayNodeName(ctx.selection.selected.object) == objectName) ||
         active_mesh_edit_object_name == objectName)) {
        ctx.selection.selectObject(displayMesh.front(), -1, objectName);
    }
    if (active_mesh_edit_object_name == objectName) {
        active_mesh_edit_object_ptr = displayMesh.empty() ? nullptr : displayMesh.front().get();
    }

    // CRITICAL: rebuildMeshCache() UNCONDITIONALLY does `editable_mesh_cache = EditableMeshCache{}`
    // (it is generic and assumes topology may have changed). But this function runs ONLY for a
    // subdivision-preview DISPLAY refresh, where the editable cache is the CAGE — and the cage is
    // NOT changed by a display refresh. Letting the wipe through is exactly what made live-CC
    // editing feel like a "detached second object": the first cage move triggered a refresh →
    // rebuildMeshCache wiped the cage cache + its sub-element selection → the next move saw
    // selV=0/targets=0 ("can't move", selection lost, crease on wrong edge). The editable cache
    // owns its own source_triangles (the same shared_ptrs as base_mesh_cache, independent of
    // world.objects / mesh_cache), so it stays fully valid across rebuildMeshCache. Preserve it
    // across the wipe and restore it.
    EditableMeshCache preservedEditable;
    const bool preserveEditable = !editable_mesh_cache.object_name.empty() &&
        editable_mesh_cache.object_name == objectName &&
        !editable_mesh_cache.vertices.empty();
    if (preserveEditable) {
        preservedEditable = std::move(editable_mesh_cache);
    }
    rebuildMeshCache(ctx.scene.world.objects);
    if (preserveEditable) {
        editable_mesh_cache = std::move(preservedEditable);
        // The cage source triangles may have been re-posed without the cache copies being
        // updated (undo/programmatic edit via applyMeshEditTriangleStates). A direct cage drag
        // already updated these in place, so this is a cheap O(V) no-op there; it keeps the
        // overlay correct after undo/release. Selection (vertex/edge/face ids) is unchanged —
        // the whole point of preserving instead of rebuilding (a rebuild reshuffles the
        // position-sorted vertex ids and would lose/scramble the selection).
        for (size_t v = 0; v < editable_mesh_cache.vertices.size(); ++v) {
            EditableVertex& vert = editable_mesh_cache.vertices[v];
            if (vert.refs.empty()) {
                continue;
            }
            const EditableVertexRef& ref = vert.refs[0];
            const Triangle* refTri = editable_mesh_cache.refTri(ref);
            if (!refTri) {
                continue;
            }
            const Vec3 srcPos = refTri->getOriginalVertexPosition(ref.corner);
            vert.local_position = srcPos;
            if (v < editable_mesh_cache.vertex_positions.size()) {
                editable_mesh_cache.vertex_positions[v] = srcPos;
            }
        }
    }
    applyMeshShadingSettings(ctx, objectName, false);
    (void)rebuildEditableCache;
    updateBBoxCache(objectName);
    objects_needing_cpu_sync.erase(objectName);

    extern bool g_bvh_rebuild_pending;
    // For a live device-resident CC node the dense subdivision mesh lives on the GPU and is
    // EXCLUDED from the RT gather — the dense host copy we just re-evaluated into world.objects
    // only feeds the CPU picking / CPU-reference BVH, neither of which runs mid-stroke in a
    // Rendered + Vulkan session. A full scene rebuild here (the dense mesh is regenerated as
    // fresh Triangle objects every cage change, so a cheap Embree refit can't track it) fired
    // on EVERY edit/sculpt dab. Defer it: each cage edit re-arms the countdown, so a single
    // correct full rebuild fires ~30 idle frames after editing stops (same pattern the
    // selection/gizmo edit paths already use). The GPU side stays live via the device refit.
    bool deferCpuBvhRebuild = false;
    if (auto* vkRender = dynamic_cast<Backend::VulkanBackendAdapter*>(g_backend.get())) {
        if (vkRender->hasDeviceResidentCCNode(objectName)) {
            extern int g_bvh_rebuild_deferred_frames;
            g_bvh_rebuild_deferred_frames = std::max(g_bvh_rebuild_deferred_frames, 30);
            deferCpuBvhRebuild = true;
        }
    }
    if (!deferCpuBvhRebuild) {
        g_bvh_rebuild_pending = true;
    }

    if (queueGpuSync && (ctx.backend_ptr != nullptr || g_backend != nullptr || g_viewport_backend != nullptr)) {
        queueMeshEditGpuSync(objectName);
    }

    ProjectManager::getInstance().markModified();
    ctx.renderer.resetCPUAccumulation();
    ctx.start_render = true;
    return true;
}

void SceneUI::applyMeshEditTriangleStates(UIContext& ctx, const std::vector<MeshEditTriangleState>& states, bool queueGpuSync) {
    if (states.empty()) {
        return;
    }

    std::string objectName;
    for (const auto& state : states) {
        if (!state.triangle) {
            continue;
        }
        objectName = state.triangle->getNodeName();
        for (int corner = 0; corner < 3; ++corner) {
            state.triangle->setOriginalVertexPosition(corner, state.positions[corner]);
        }
        state.triangle->markAABBDirty();
    }

    if (!objectName.empty()) {
        if (!refreshEditableDisplayMeshFromBase(ctx, objectName, queueGpuSync)) {
            applyMeshShadingSettings(ctx, objectName, queueGpuSync);
        }
    }
}

void SceneUI::setMeshEditLayerEnabled(UIContext& ctx, bool enabled) {
    if (!mesh_edit_layer.active || mesh_edit_layer.object_name.empty()) {
        return;
    }
    mesh_edit_layer.enabled = enabled;
    applyMeshEditTriangleStates(ctx, enabled ? mesh_edit_layer.edited_states : mesh_edit_layer.base_states);
}

void SceneUI::applyMeshEditLayer(UIContext& ctx) {
    if (!mesh_edit_layer.active || mesh_edit_layer.object_name.empty()) {
        return;
    }

    if (!mesh_edit_layer.enabled) {
        applyMeshEditTriangleStates(ctx, mesh_edit_layer.base_states);
    } else {
        refreshMeshEditLayerEditedState(ctx);
    }

    mesh_edit_layer = MeshEditLayer{};
    pending_serialized_mesh_edit_layer = PendingSerializedMeshEditLayer{};
    ProjectManager::getInstance().markModified();
    ctx.start_render = true;
}

void SceneUI::discardMeshEditLayer(UIContext& ctx) {
    if (!mesh_edit_layer.active || mesh_edit_layer.object_name.empty()) {
        return;
    }
    applyMeshEditTriangleStates(ctx, mesh_edit_layer.base_states);
    mesh_edit_layer = MeshEditLayer{};
    pending_serialized_mesh_edit_layer = PendingSerializedMeshEditLayer{};
    ProjectManager::getInstance().markModified();
}

void SceneUI::tryRestoreSerializedMeshEditLayer(UIContext& ctx) {
    if (!pending_serialized_mesh_edit_layer.has_data || pending_serialized_mesh_edit_layer.object_name.empty()) {
        return;
    }
    if (!mesh_cache_valid) {
        return;
    }

    auto meshIt = mesh_cache.find(pending_serialized_mesh_edit_layer.object_name);
    if (meshIt == mesh_cache.end()) {
        return;
    }

    const auto& entries = meshIt->second;
    if (entries.size() != pending_serialized_mesh_edit_layer.base_positions.size() ||
        entries.size() != pending_serialized_mesh_edit_layer.edited_positions.size()) {
        pending_serialized_mesh_edit_layer = PendingSerializedMeshEditLayer{};
        return;
    }

    mesh_edit_layer = MeshEditLayer{};
    mesh_edit_layer.active = true;
    mesh_edit_layer.enabled = pending_serialized_mesh_edit_layer.enabled;
    mesh_edit_layer.object_name = pending_serialized_mesh_edit_layer.object_name;
    mesh_edit_layer.base_states.reserve(entries.size());
    mesh_edit_layer.edited_states.reserve(entries.size());

    for (size_t i = 0; i < entries.size(); ++i) {
        if (!entries[i].second) {
            continue;
        }
        MeshEditTriangleState baseState;
        baseState.triangle = entries[i].second;
        baseState.positions = pending_serialized_mesh_edit_layer.base_positions[i];
        mesh_edit_layer.base_states.push_back(baseState);

        MeshEditTriangleState editedState;
        editedState.triangle = entries[i].second;
        editedState.positions = pending_serialized_mesh_edit_layer.edited_positions[i];
        mesh_edit_layer.edited_states.push_back(editedState);
    }

    applyMeshEditTriangleStates(ctx, mesh_edit_layer.enabled ? mesh_edit_layer.edited_states : mesh_edit_layer.base_states);
    pending_serialized_mesh_edit_layer = PendingSerializedMeshEditLayer{};
}

bool SceneUI::ensureEditableMeshCache(UIContext& ctx, const std::string& objectName) {
    if (objectName.empty()) {
        editable_mesh_cache = EditableMeshCache{};
        invalidateSculptControlGraph();
        invalidateSculptPBVH();
        return false;
    }

    if (!mesh_cache_valid) {
        rebuildMeshCache(ctx.scene.world.objects);
    }

    const auto modifierIt = ctx.scene.mesh_modifiers.find(objectName);
    const MeshModifiers::ModifierStack* modifierStack =
        (modifierIt != ctx.scene.mesh_modifiers.end()) ? &modifierIt->second : nullptr;
    auto cacheIt = mesh_cache.find(objectName);
    if (modifierStack &&
        hasEnabledSubdivisionPreview(*modifierStack) &&
        ctx.scene.base_mesh_cache.find(objectName) != ctx.scene.base_mesh_cache.end()) {
        // Re-evaluate the display mesh ONLY when the subdivision actually changed. The old
        // test compared the cached triangle count to base*4^levels, but Catmull-Clark /
        // quad-recovery subdivision does not produce exactly base*4^levels triangles, so the
        // count never matched and this re-evaluated EVERY frame after a subdivide — bumping the
        // cache revision, re-uploading the GPU edit overlay (m_interactiveViewport.dirty), and
        // raising g_bvh_rebuild_pending, which pinned the viewport in a permanent re-render
        // (constant CPU until a Rendered round-trip released the overlay). The signature folds
        // the modifier params + cage vertex positions, so it is stable in steady state and
        // changes every frame only during an actual cage drag (where re-eval IS wanted).
        const std::size_t subdivSignature = computeSubdivisionPreviewSignature(ctx, objectName);
        const bool signatureCurrent =
            editable_subdiv_display_signature_valid &&
            editable_subdiv_display_signature_object == objectName &&
            editable_subdiv_display_signature == subdivSignature;
        const bool previewOutOfDate =
            cacheIt == mesh_cache.end() ||
            cacheIt->second.empty() ||
            !signatureCurrent;
        if (previewOutOfDate) {
            // Don't rebuild the editable cache mid-drag (keeps vertex ids + selection stable).
            refreshEditableDisplayMeshFromBase(ctx, objectName, false,
                !isInteractiveSubdivisionPreviewActiveForObject(objectName));
            cacheIt = mesh_cache.find(objectName);
            editable_subdiv_display_signature = subdivSignature;
            editable_subdiv_display_signature_object = objectName;
            editable_subdiv_display_signature_valid = true;
        }
    }
    if (cacheIt == mesh_cache.end() || cacheIt->second.empty()) {
        editable_mesh_cache = EditableMeshCache{};
        invalidateSculptControlGraph(objectName);
        invalidateSculptPBVH(objectName);
        return false;
    }

    const auto& meshEntries = cacheIt->second;
    const bool useControlCage =
        modifierStack &&
        hasEnabledSubdivisionPreview(*modifierStack) &&
        ctx.scene.base_mesh_cache.find(objectName) != ctx.scene.base_mesh_cache.end() &&
        !ctx.scene.base_mesh_cache[objectName].empty();
    const std::vector<std::shared_ptr<Triangle>>* editableSourceTriangles =
        useControlCage ? &ctx.scene.base_mesh_cache[objectName] : nullptr;
    const size_t triangleCount = editableSourceTriangles ? editableSourceTriangles->size() : meshEntries.size();
    Matrix4x4 currentObjectTransform = Matrix4x4::identity();
    if (!meshEntries.empty() && meshEntries[0].second) {
        currentObjectTransform = meshEntries[0].second->getTransformMatrix();
    } else if (editableSourceTriangles && !editableSourceTriangles->empty() && (*editableSourceTriangles)[0]) {
        currentObjectTransform = (*editableSourceTriangles)[0]->getTransformMatrix();
    }
    // Sculpt only needs the welded vertices / faces / refs / neighbors / spatial buckets,
    // not the Edit-mode topology (polygon faces & edges, half-edge, edge→triangle map).
    // Building those on a 2M-tri mesh costs ~10s, so skip them when entering sculpt. If the
    // cache was built minimal and the user now wants full Edit topology, force a rebuild.
    const bool buildForSculpt = (mesh_workspace_mode == MeshWorkspaceMode::Sculpt);

    // Edit-mode quad topology (polygon faces/edges + edge→triangle pairing for quad
    // recovery) is built with a PARALLEL SORT below (the old serial edge→triangle map
    // was ~26s of the 29.7s at 8M), so quad editing stays available even on dense meshes.
    // This threshold is now only a safety valve for pathological densities (32M+ in edit
    // mode) where even the parallel build is too costly; above it we fall back to the
    // per-triangle edges/faces (every consumer already handles empty polygon_faces/
    // polygon_edges — the same path the sculpt build takes).
    constexpr size_t kEditPolygonTopologyMaxTris = 4000000;
    
    if (!buildForSculpt && triangleCount > kEditPolygonTopologyMaxTris) {
        addViewportMessage("Mesh too dense for Edit Mode. Exiting to Object mode.", 3.0f, ImVec4(1.0f, 0.5f, 0.0f, 1.0f));
        mesh_overlay_settings.edit_mode = false;
        ctx.selection.mesh_element_mode = MeshElementSelectMode::Object;
        editable_mesh_cache = EditableMeshCache{};
        return false;
    }
    
    const bool buildEditTopology = !buildForSculpt && triangleCount <= kEditPolygonTopologyMaxTris;

    const bool needsRebuild =
        editable_mesh_cache.object_name != objectName ||
        editable_mesh_cache.source_triangle_count != triangleCount ||
        !(editable_mesh_cache.source_object_transform == currentObjectTransform) ||
        (editable_mesh_cache.built_minimal_for_sculpt && !buildForSculpt);

    const MeshShadingSettings& shading = ensureMeshShadingSettings(objectName);

    if (!needsRebuild) {
        editable_mesh_cache.shade_flat = shading.flat_shading;
        editable_mesh_cache.auto_smooth = shading.auto_smooth;
        editable_mesh_cache.auto_smooth_angle_degrees = shading.auto_smooth_angle_degrees;
        return true;
    }

    MESH_PROFILE_SCOPE(std::string("ensureEditableMeshCache.build[") +
        (buildForSculpt ? "sculpt" : "edit") + "] (" + std::to_string(triangleCount) + " tris)");

    editable_mesh_cache = EditableMeshCache{};
    invalidateSculptControlGraph(objectName);
    invalidateSculptPBVH(objectName);
    editable_mesh_cache.object_name = objectName;
    editable_mesh_cache.revision = ++editable_mesh_cache_revision_counter;
    editable_mesh_cache.source_triangle_count = triangleCount;
    editable_mesh_cache.source_object_transform = currentObjectTransform;
    editable_mesh_cache.shade_flat = shading.flat_shading;
    editable_mesh_cache.auto_smooth = shading.auto_smooth;
    editable_mesh_cache.auto_smooth_angle_degrees = shading.auto_smooth_angle_degrees;
    editable_mesh_cache.built_minimal_for_sculpt = buildForSculpt;


    // === Parallel vertex weld (sort-based dedup) ===
    // Flatten the source triangles into one list, then dedup welded vertices WITHOUT a
    // serial growing hash map: quantize every corner in parallel, sort an index permutation
    // by key in parallel, and assign vertex ids in one cheap serial scan. Vertex ids end up
    // in sorted-key order rather than first-seen order — safe, because editable-cache vertex
    // ids are runtime-only (rebuilt each session; the sculpt control graph derives from them
    // 1:1; nothing persists or cross-references them).
    std::vector<std::shared_ptr<Triangle>> weldTris;
    // Source-mesh index per welded triangle (parallel to weldTris). Becomes
    // editable_mesh_cache.face_to_mesh_index — the flat replacement for triangle_to_mesh_index.
    std::vector<int> srcMeshIndex;
    {
        const size_t srcCount = editableSourceTriangles ? editableSourceTriangles->size() : meshEntries.size();
        weldTris.reserve(srcCount);
        srcMeshIndex.reserve(srcCount);
        if (editableSourceTriangles) {
            for (size_t i = 0; i < editableSourceTriangles->size(); ++i) {
                if ((*editableSourceTriangles)[i]) {
                    weldTris.push_back((*editableSourceTriangles)[i]);
                    srcMeshIndex.push_back(static_cast<int>(i));
                }
            }
        } else {
            for (size_t i = 0; i < meshEntries.size(); ++i) {
                if (meshEntries[i].second) {
                    weldTris.push_back(meshEntries[i].second);
                    srcMeshIndex.push_back(static_cast<int>(i));
                }
            }
        }
    }
    const size_t weldCornerCount = weldTris.size() * 3;

    // Pass 1 (parallel): quantize every corner position into a flat key array.
    std::vector<QuantizedVertexKey> cornerKeys(weldCornerCount);
    {
        std::vector<size_t> triIota(weldTris.size());
        std::iota(triIota.begin(), triIota.end(), size_t{ 0 });
        std::for_each(std::execution::par, triIota.begin(), triIota.end(),
            [&](size_t t) {
                for (int c = 0; c < 3; ++c) {
                    cornerKeys[3 * t + c] =
                        quantizeTopologyVertex(weldTris[t]->getOriginalVertexPosition(c));
                }
            });
    }

    // Pass 2: dedup keys → vertex ids (parallel sort of an index permutation, then a cheap
    // serial scan that assigns ids in sorted order — no per-corner hashing).
    std::vector<int> cornerVertexId(weldCornerCount, -1);
    {
        std::vector<uint32_t> order(weldCornerCount);
        std::iota(order.begin(), order.end(), uint32_t{ 0 });
        std::sort(std::execution::par, order.begin(), order.end(),
            [&](uint32_t a, uint32_t b) {
                const QuantizedVertexKey& ka = cornerKeys[a];
                const QuantizedVertexKey& kb = cornerKeys[b];
                if (ka.x != kb.x) return ka.x < kb.x;
                if (ka.y != kb.y) return ka.y < kb.y;
                return ka.z < kb.z;
            });

        editable_mesh_cache.vertices.reserve(weldCornerCount / 4 + 1);
        for (size_t k = 0; k < weldCornerCount; ++k) {
            const uint32_t idx = order[k];
            if (k == 0 || !(cornerKeys[idx] == cornerKeys[order[k - 1]])) {
                EditableVertex vertex;
                vertex.local_position =
                    weldTris[idx / 3]->getOriginalVertexPosition(static_cast<int>(idx % 3));
                editable_mesh_cache.vertices.push_back(vertex);
            }
            cornerVertexId[idx] = static_cast<int>(editable_mesh_cache.vertices.size()) - 1;
        }
    }

    // Take ownership of the welded source triangles once; refs/faces index into this
    // vector (Phase 1 SoA migration) instead of each holding a shared_ptr copy.
    editable_mesh_cache.source_triangles = std::move(weldTris);
    editable_mesh_cache.face_to_mesh_index = std::move(srcMeshIndex);
    const size_t sourceTriCount = editable_mesh_cache.source_triangles.size();
    const size_t vertexCount = editable_mesh_cache.vertices.size();

    // Pass 3a (P-CSR): offset table (LOCAL) for the per-vertex incident-triangle refs.
    // Counting scan into offsets[v+1], then prefix-sum, then point each vertex's refs span
    // into the flat vertex_ref_data — replaces the old per-vertex std::vector (millions of
    // tiny heap allocs) with one flat array + lightweight spans.
    std::vector<int> refOffsets(vertexCount + 1, 0);
    for (size_t k = 0; k < weldCornerCount; ++k) {
        const int vid = cornerVertexId[k];
        if (vid >= 0) ++refOffsets[static_cast<size_t>(vid) + 1];
    }
    for (size_t v = 0; v < vertexCount; ++v) {
        refOffsets[v + 1] += refOffsets[v];
    }
    editable_mesh_cache.vertex_ref_data.resize(static_cast<size_t>(refOffsets[vertexCount]));
    {
        const EditableVertexRef* base = editable_mesh_cache.vertex_ref_data.data();
        for (size_t v = 0; v < vertexCount; ++v) {
            editable_mesh_cache.vertices[v].refs = { base + refOffsets[v], base + refOffsets[v + 1] };
        }
    }

    // Pass 3b (serial): faces + editable_index stamp + fill the refs data via a per-vertex
    // write cursor (seeded from the offsets). {v0,v1,v2} now comes straight from faces[idx].
    std::vector<int> refCursor = refOffsets; // V+1 (last unused)
    editable_mesh_cache.faces.reserve(sourceTriCount);
    for (size_t t = 0; t < sourceTriCount; ++t) {
        Triangle* tri = editable_mesh_cache.source_triangles[t].get();
        const int triIndex = static_cast<int>(t);
        tri->editable_index = triIndex;
        const int v[3] = { cornerVertexId[3 * t + 0], cornerVertexId[3 * t + 1], cornerVertexId[3 * t + 2] };
        for (int c = 0; c < 3; ++c) {
            if (v[c] < 0) continue;
            editable_mesh_cache.vertex_ref_data[static_cast<size_t>(refCursor[v[c]]++)] =
                EditableVertexRef{ triIndex, c };
        }

        EditableFace face;
        face.triangle_index = triIndex;
        face.v0 = v[0];
        face.v1 = v[1];
        face.v2 = v[2];
        editable_mesh_cache.faces.push_back(face);
    }


    // Pass 3c: unique undirected edges + per-edge face count, derived from a PARALLEL SORT
    // of all directed edges (replaces the old per-corner unordered_set/unordered_map
    // insertions — same proven pattern as the vertex weld above). edgeFaceCount[e] == 1
    // marks a boundary edge; vertex_neighbors is filled straight from this unique list.
    std::vector<int> edgeFaceCount; // parallel to editable_mesh_cache.edges
    {
        const size_t F = editable_mesh_cache.faces.size();
        std::vector<unsigned long long> ekeys(F * 3);
        {
            std::vector<size_t> fIota(F);
            std::iota(fIota.begin(), fIota.end(), size_t{ 0 });
            std::for_each(std::execution::par, fIota.begin(), fIota.end(),
                [&](size_t f) {
                    const EditableFace& face = editable_mesh_cache.faces[f];
                    const int vv[3] = { face.v0, face.v1, face.v2 };
                    for (int i = 0; i < 3; ++i) {
                        int a = vv[i];
                        int b = vv[(i + 1) % 3];
                        if (a < 0 || b < 0) { ekeys[3 * f + i] = ~0ull; continue; }
                        if (b < a) std::swap(a, b);
                        ekeys[3 * f + i] =
                            (static_cast<unsigned long long>(static_cast<unsigned int>(a)) << 32ull) |
                            static_cast<unsigned long long>(static_cast<unsigned int>(b));
                    }
                });
        }
        std::sort(std::execution::par, ekeys.begin(), ekeys.end());

        editable_mesh_cache.edges.reserve(F * 3 / 2 + 1);
        edgeFaceCount.reserve(F * 3 / 2 + 1);
        for (size_t k = 0; k < ekeys.size();) {
            const unsigned long long key = ekeys[k];
            if (key == ~0ull) break; // degenerate sentinels sort to the very end
            size_t k2 = k + 1;
            while (k2 < ekeys.size() && ekeys[k2] == key) ++k2;
            const int a = static_cast<int>(static_cast<unsigned int>(key >> 32));
            const int b = static_cast<int>(static_cast<unsigned int>(key & 0xffffffffull));
            editable_mesh_cache.edges.push_back(EditableEdge{ a, b });
            edgeFaceCount.push_back(static_cast<int>(k2 - k));
            k = k2;
        }
    }

    // Edit-only: quad recovery (n-gon faces) + polygon edge list. Sculpt never touches
    // polygon_faces / polygon_edges, so skip this entire pass in the sculpt build.
    if (buildEditTopology) {
    editable_mesh_cache.polygon_faces.clear();
    editable_mesh_cache.polygon_faces.reserve(editable_mesh_cache.faces.size());
    std::vector<bool> triangleFaceConsumed(editable_mesh_cache.faces.size(), false);

    // Edge→triangle pairing via a PARALLEL SORT of (undirected-edge-key, triangle-id)
    // pairs — replaces the old serial unordered_map that dominated the dense edit build
    // (~26s at 8M). A manifold interior edge owns exactly two triangles → a quad-merge
    // candidate. Greedy consumption is sorted-by-edge (deterministic) instead of hash
    // order; any valid pairing is fine (the merge test still gates each quad).
    {
        const size_t F = editable_mesh_cache.faces.size();
        struct EdgeTri { unsigned long long key; int tri; };
        std::vector<EdgeTri> edgeTris(F * 3);
        {
            std::vector<size_t> fIota(F);
            std::iota(fIota.begin(), fIota.end(), size_t{ 0 });
            std::for_each(std::execution::par, fIota.begin(), fIota.end(),
                [&](size_t f) {
                    const EditableFace& face = editable_mesh_cache.faces[f];
                    const int vv[3] = { face.v0, face.v1, face.v2 };
                    for (int i = 0; i < 3; ++i) {
                        int a = vv[i];
                        int b = vv[(i + 1) % 3];
                        unsigned long long key = ~0ull;
                        if (a >= 0 && b >= 0) {
                            if (b < a) std::swap(a, b);
                            key = (static_cast<unsigned long long>(static_cast<unsigned int>(a)) << 32ull) |
                                  static_cast<unsigned long long>(static_cast<unsigned int>(b));
                        }
                        edgeTris[3 * f + i] = EdgeTri{ key, static_cast<int>(f) };
                    }
                });
        }
        std::sort(std::execution::par, edgeTris.begin(), edgeTris.end(),
            [](const EdgeTri& x, const EdgeTri& y) { return x.key < y.key; });

        for (size_t k = 0; k < edgeTris.size();) {
            const unsigned long long key = edgeTris[k].key;
            if (key == ~0ull) break; // degenerate sentinels sort to the very end
            size_t k2 = k + 1;
            while (k2 < edgeTris.size() && edgeTris[k2].key == key) ++k2;
            if (k2 - k == 2) {
                const int faceAId = edgeTris[k].tri;
                const int faceBId = edgeTris[k + 1].tri;
                if (faceAId >= 0 && faceBId >= 0 &&
                    faceAId < static_cast<int>(editable_mesh_cache.faces.size()) &&
                    faceBId < static_cast<int>(editable_mesh_cache.faces.size()) &&
                    !triangleFaceConsumed[faceAId] && !triangleFaceConsumed[faceBId]) {
                    std::vector<int> mergedVertexIds;
                    if (canMergeEditableTrianglesToQuad(
                            editable_mesh_cache,
                            editable_mesh_cache.faces[faceAId],
                            editable_mesh_cache.faces[faceBId],
                            mergedVertexIds)) {
                        addEditablePolygonFaceFromTriangles(editable_mesh_cache, mergedVertexIds, { faceAId, faceBId });
                        triangleFaceConsumed[faceAId] = true;
                        triangleFaceConsumed[faceBId] = true;
                    }
                }
            }
            k = k2;
        }
    }

    for (size_t faceId = 0; faceId < editable_mesh_cache.faces.size(); ++faceId) {
        if (triangleFaceConsumed[faceId]) {
            continue;
        }
        const auto& face = editable_mesh_cache.faces[faceId];
        addEditablePolygonFaceFromTriangles(
            editable_mesh_cache,
            { face.v0, face.v1, face.v2 },
            { static_cast<int>(faceId) });
    }

    editable_mesh_cache.polygon_edges.clear();
    editable_mesh_cache.polygon_edges.reserve(editable_mesh_cache.edges.size());
    std::unordered_set<unsigned long long> polygonEdgeLookup;
    polygonEdgeLookup.reserve(editable_mesh_cache.edges.size());
    for (const auto& polygonFace : editable_mesh_cache.polygon_faces) {
        if (polygonFace.vertex_ids.size() < 2) {
            continue;
        }

        for (size_t edgeIndex = 0; edgeIndex < polygonFace.vertex_ids.size(); ++edgeIndex) {
            int a = polygonFace.vertex_ids[edgeIndex];
            int b = polygonFace.vertex_ids[(edgeIndex + 1) % polygonFace.vertex_ids.size()];
            if (a < 0 || b < 0) {
                continue;
            }
            if (b < a) {
                std::swap(a, b);
            }

            const unsigned long long packed =
                (static_cast<unsigned long long>(static_cast<unsigned int>(a)) << 32ull) |
                static_cast<unsigned long long>(static_cast<unsigned int>(b));
            if (polygonEdgeLookup.insert(packed).second) {
                editable_mesh_cache.polygon_edges.push_back(EditableEdge{ a, b });
            }
        }
    }
    } // if (buildEditTopology) — quad recovery + polygon edges

    // TEMP diagnostic: ground-truth what the edit build actually produced, so we can tell
    // a stale binary / wrong path from a genuine no-quad mesh. Remove once ① is confirmed.
    {
        size_t quadCount = 0, ngonCount = 0, triPolyCount = 0;
        for (const auto& pf : editable_mesh_cache.polygon_faces) {
            const size_t n = pf.vertex_ids.size();
            if (n == 4) ++quadCount; else if (n == 3) ++triPolyCount; else if (n > 4) ++ngonCount;
        }
        char qbuf[512];
        std::snprintf(qbuf, sizeof(qbuf),
            "[QUADDIAG] obj='%s' useControlCage=%d buildEditTopology=%d triCount=%zu threshold=%zu "
            "faces=%zu polyFaces=%zu (quads=%zu tris=%zu ngons=%zu)",
            objectName.c_str(), useControlCage ? 1 : 0, buildEditTopology ? 1 : 0,
            triangleCount, kEditPolygonTopologyMaxTris,
            editable_mesh_cache.faces.size(), editable_mesh_cache.polygon_faces.size(),
            quadCount, triPolyCount, ngonCount);
        SCENE_LOG_INFO(qbuf);
    }

    float avgEdgeLength = 0.0f;
    int avgEdgeSamples = 0;
    for (const auto& edge : editable_mesh_cache.edges) {
        if (edge.v0 < 0 || edge.v1 < 0 ||
            edge.v0 >= static_cast<int>(editable_mesh_cache.vertices.size()) ||
            edge.v1 >= static_cast<int>(editable_mesh_cache.vertices.size())) {
            continue;
        }
        const float edgeLength = (editable_mesh_cache.vertices[edge.v1].local_position -
                                  editable_mesh_cache.vertices[edge.v0].local_position).length();
        if (!std::isfinite(edgeLength) || edgeLength <= 1e-6f) {
            continue;
        }
        avgEdgeLength += edgeLength;
        ++avgEdgeSamples;
    }

    if (avgEdgeSamples > 0) {
        avgEdgeLength /= static_cast<float>(avgEdgeSamples);
    }
    editable_mesh_cache.spatial_cell_size = (std::max)(avgEdgeLength * 2.5f, 1e-4f);
    editable_mesh_cache.vertex_spatial_buckets.clear();
    editable_mesh_cache.vertex_spatial_buckets.reserve(editable_mesh_cache.vertices.size());

    // Build the per-vertex neighbour list as CSR (P-CSR): count pass (degree per vertex,
    // also flagging boundary verts) -> prefix sum -> fill via cursor -> point each vertex's
    // neighbour span into the flat data. Replaces the old std::vector<std::vector<int>>
    // vertex_neighbors (one heap vector per vertex).
    std::vector<int> nbrOffsets(vertexCount + 1, 0);
    const auto edgeVertsValid = [&](const EditableEdge& edge) {
        return edge.v0 >= 0 && edge.v1 >= 0 &&
               edge.v0 < static_cast<int>(vertexCount) && edge.v1 < static_cast<int>(vertexCount);
    };
    for (size_t e = 0; e < editable_mesh_cache.edges.size(); ++e) {
        const EditableEdge& edge = editable_mesh_cache.edges[e];
        if (!edgeVertsValid(edge)) continue;
        ++nbrOffsets[static_cast<size_t>(edge.v0) + 1];
        ++nbrOffsets[static_cast<size_t>(edge.v1) + 1];
        // Boundary = an undirected edge touched by a single face (edgeFaceCount parallels
        // the edges array, built together by the sort pass above).
        if (e < edgeFaceCount.size() && edgeFaceCount[e] <= 1) {
            editable_mesh_cache.vertices[edge.v0].is_boundary = true;
            editable_mesh_cache.vertices[edge.v1].is_boundary = true;
        }
    }
    for (size_t v = 0; v < vertexCount; ++v) {
        nbrOffsets[v + 1] += nbrOffsets[v];
    }
    editable_mesh_cache.vertex_neighbor_data.resize(static_cast<size_t>(nbrOffsets[vertexCount]));
    std::vector<int> nbrCursor = nbrOffsets;
    for (size_t e = 0; e < editable_mesh_cache.edges.size(); ++e) {
        const EditableEdge& edge = editable_mesh_cache.edges[e];
        if (!edgeVertsValid(edge)) continue;
        editable_mesh_cache.vertex_neighbor_data[static_cast<size_t>(nbrCursor[edge.v0]++)] = edge.v1;
        editable_mesh_cache.vertex_neighbor_data[static_cast<size_t>(nbrCursor[edge.v1]++)] = edge.v0;
    }
    editable_mesh_cache.vertex_neighbors.resize(vertexCount);
    {
        const int* nbase = editable_mesh_cache.vertex_neighbor_data.data();
        for (size_t v = 0; v < vertexCount; ++v) {
            editable_mesh_cache.vertex_neighbors[v] = { nbase + nbrOffsets[v], nbase + nbrOffsets[v + 1] };
        }
    }
    for (size_t vertexId = 0; vertexId < editable_mesh_cache.vertices.size(); ++vertexId) {
        const EditableSpatialCellKey key = makeEditableSpatialCellKey(
            editable_mesh_cache.vertices[vertexId].local_position,
            editable_mesh_cache.spatial_cell_size);
        editable_mesh_cache.vertex_spatial_buckets[key].push_back(static_cast<int>(vertexId));
    }

    // Triangle* -> source-mesh index (for partial raster sync) is now the flat
    // face_to_mesh_index, built in the gather above and keyed by Triangle::editable_index —
    // no per-triangle hash map. (See editableFaceIndexOf.)
    editable_mesh_cache.vertex_mark_stamps.assign(editable_mesh_cache.vertices.size(), 0u);
    editable_mesh_cache.vertex_mark_generation = 1u;
    // Stamps are now indexed by face index (Triangle::editable_index), so size by the face
    // count rather than the source-mesh count.
    editable_mesh_cache.triangle_mark_stamps.assign(editable_mesh_cache.faces.size(), 0u);
    editable_mesh_cache.triangle_mark_generation = 1u;

    // Half-edge topology is now built LAZILY (see ensureEditableHalfEdge). Only a few
    // Edit-mode operators (loop cut, etc.) need it, and buildFromPolygons costs ~3s on a
    // 2M-tri mesh, so building it on every cache rebuild stalled both Edit and Sculpt
    // entry. half_edge_valid stays false here (cache was reset above); the operators call
    // ensureEditableHalfEdge() before use.

    // Populate SoA arrays inside rebuilding cache
    editable_mesh_cache.vertex_positions.resize(editable_mesh_cache.vertices.size());
    editable_mesh_cache.vertex_is_boundary.resize(editable_mesh_cache.vertices.size());
    for (size_t i = 0; i < editable_mesh_cache.vertices.size(); ++i) {
        editable_mesh_cache.vertex_positions[i] = editable_mesh_cache.vertices[i].local_position;
        editable_mesh_cache.vertex_is_boundary[i] = editable_mesh_cache.vertices[i].is_boundary ? 1 : 0;
    }

    // TEMP memory breakdown (remove with the MESHPROF timers): how much of the working set is
    // the per-face Triangle soup vs this editable cache, so the P-render effort can target the
    // real bulk of the ~12GB-at-8M footprint.
    {
        const auto& c = editable_mesh_cache;
        auto mb = [](size_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0); };
        const size_t soupBytes = c.source_triangles.size() * sizeof(Triangle);
        size_t cacheBytes =
            c.source_triangles.capacity() * sizeof(std::shared_ptr<Triangle>) +
            c.vertices.capacity() * sizeof(SceneUI::EditableVertex) +
            c.vertex_positions.capacity() * sizeof(Vec3) +
            c.vertex_is_boundary.capacity() +
            c.edges.capacity() * sizeof(SceneUI::EditableEdge) +
            c.polygon_edges.capacity() * sizeof(SceneUI::EditableEdge) +
            c.faces.capacity() * sizeof(SceneUI::EditableFace) +
            c.vertex_ref_data.capacity() * sizeof(SceneUI::EditableVertexRef) +
            c.vertex_neighbor_data.capacity() * sizeof(int) +
            c.vertex_neighbors.capacity() * sizeof(SceneUI::CacheSpan<int>) +
            c.face_to_mesh_index.capacity() * sizeof(int) +
            c.vertex_mark_stamps.capacity() * sizeof(uint32_t) +
            c.triangle_mark_stamps.capacity() * sizeof(uint32_t);
        size_t bucketBytes = c.vertex_spatial_buckets.size() *
            (sizeof(SceneUI::EditableSpatialCellKey) + sizeof(std::vector<int>) + 48);
        for (const auto& kv : c.vertex_spatial_buckets) bucketBytes += kv.second.capacity() * sizeof(int);

        // Total process working set, and the "unaccounted" remainder = everything
        // that is NOT the editable cache's soup+structures (Embree CPU BVH, undo
        // history, GPU staging buffers, the pre-subdivision base cage, ImGui/app).
        // This tells us whether eliminating the soup (G3) actually targets the bulk
        // of the footprint or whether the real bloat lives elsewhere.
        // base_mesh_cache holds each node's modifier BASE mesh. Normally tiny (the
        // pre-subdivision cage), but an Apply bakes the subdivided result back into
        // base — prime suspect for a retained DUPLICATE full-res soup. Report its
        // size, and whether THIS object's base shares the same Triangle objects as
        // the editable soup (shared = no extra RAM; separate = a real duplicate).
        size_t baseTris = 0, baseBytes = 0;
        for (const auto& kv : ctx.scene.base_mesh_cache) {
            baseTris += kv.second.size();
            baseBytes += kv.second.capacity() * sizeof(std::shared_ptr<Triangle>) +
                         kv.second.size() * sizeof(Triangle);
        }
        const char* baseShare = "n/a";
        auto bmcIt = ctx.scene.base_mesh_cache.find(objectName);
        if (bmcIt != ctx.scene.base_mesh_cache.end() && !bmcIt->second.empty() && !c.source_triangles.empty())
            baseShare = (bmcIt->second.front().get() == c.source_triangles.front().get())
                ? "SHARED-with-soup" : "SEPARATE-COPY";

        const size_t totalRSS = meshprof_detail::workingSetBytes();
        const size_t accounted = soupBytes + cacheBytes + bucketBytes;
        const double unaccountedMB = (totalRSS > accounted) ? mb(totalRSS - accounted) : 0.0;
        char buf[512];
        std::snprintf(buf, sizeof(buf),
            "[MESHMEM] %zu tris: Triangle soup=%.0f MB, editable cache=%.0f MB (spatial buckets %.0f MB), "
            "sizeof(Triangle)=%zu B | base_mesh_cache=%zu tris %.0f MB (%s) | process RSS=%.0f MB, "
            "unaccounted=%.0f MB (Embree BVH/undo/GPU/app)",
            c.source_triangles.size(), mb(soupBytes), mb(cacheBytes + bucketBytes), mb(bucketBytes),
            sizeof(Triangle), baseTris, mb(baseBytes), baseShare, mb(totalRSS), unaccountedMB);
        SCENE_LOG_INFO(buf);
    }

    return !editable_mesh_cache.vertices.empty();
}

bool SceneUI::ensureEditableHalfEdge() {
    if (editable_mesh_cache.half_edge_valid) {
        return true;
    }
    // Half-edge is derived from polygon_faces, which the sculpt build skips. Edit-mode
    // operators (the only callers) run on a full cache, so this is normally satisfied; if
    // the cache happens to be the minimal sculpt layout, there is nothing to build from.
    if (editable_mesh_cache.built_minimal_for_sculpt ||
        editable_mesh_cache.polygon_faces.empty()) {
        return false;
    }

    std::vector<Vec3> hePositions;
    hePositions.reserve(editable_mesh_cache.vertices.size());
    for (const auto& vertex : editable_mesh_cache.vertices) {
        hePositions.push_back(vertex.local_position);
    }
    std::vector<std::vector<int>> hePolygons;
    hePolygons.reserve(editable_mesh_cache.polygon_faces.size());
    for (const auto& polygonFace : editable_mesh_cache.polygon_faces) {
        hePolygons.push_back(polygonFace.vertex_ids);
    }
    editable_mesh_cache.half_edge_valid = editable_mesh_cache.half_edge.buildFromPolygons(
        hePositions, hePolygons, &editable_mesh_cache.half_edge_build);
    return editable_mesh_cache.half_edge_valid;
}

void SceneUI::invalidateSculptControlGraph(const std::string& objectName) {
    if (objectName.empty() || sculpt_control_graph.object_name == objectName) {
        sculpt_control_graph = SculptControlGraph{};
    }
}

void SceneUI::invalidateSculptPBVH(const std::string& objectName) {
    if (objectName.empty() || sculpt_pbvh.object_name == objectName) {
        sculpt_pbvh = SculptPBVH{};
    }
}

bool SceneUI::ensureSculptControlGraph(UIContext& ctx, const std::string& objectName) {
    if (!ensureEditableMeshCache(ctx, objectName)) {
        invalidateSculptControlGraph(objectName);
        return false;
    }

    // Keep the legacy vertex graph active for every sculpt brush until the
    // PBVH replacement is ready. The spatial leaf prototype produced visible
    // blocking on curved surfaces without delivering a meaningful speedup.
    constexpr bool shouldUseSpatialLeafNodes = false;
    const bool needsRebuild =
        sculpt_control_graph.object_name != editable_mesh_cache.object_name ||
        sculpt_control_graph.source_triangle_count != editable_mesh_cache.source_triangle_count ||
        !(sculpt_control_graph.source_object_transform == editable_mesh_cache.source_object_transform) ||
        sculpt_control_graph.source_vertex_to_node_id.size() != editable_mesh_cache.vertices.size() ||
        sculpt_control_graph.uses_spatial_leaf_nodes != shouldUseSpatialLeafNodes;
    if (!needsRebuild) {
        return !sculpt_control_graph.nodes.empty();
    }

    sculpt_control_graph = SculptControlGraph{};
    sculpt_control_graph.object_name = editable_mesh_cache.object_name;
    sculpt_control_graph.source_triangle_count = editable_mesh_cache.source_triangle_count;
    sculpt_control_graph.source_object_transform = editable_mesh_cache.source_object_transform;

    const float avgEdgeLength = computeEditableAverageEdgeLength(editable_mesh_cache);
    sculpt_control_graph.avg_edge_length = avgEdgeLength;
    sculpt_control_graph.spatial_cell_size = sanitizeFiniteFloat(
        editable_mesh_cache.spatial_cell_size > 1e-6f ? editable_mesh_cache.spatial_cell_size : avgEdgeLength * 2.5f,
        1e-4f,
        1e-4f,
        100000.0f);

    buildVertexSculptControlGraph(sculpt_control_graph, editable_mesh_cache);

    return !sculpt_control_graph.nodes.empty();
}

bool SceneUI::ensureSculptPBVH(UIContext& ctx, const std::string& objectName) {
    if (!ensureEditableMeshCache(ctx, objectName)) {
        invalidateSculptPBVH(objectName);
        return false;
    }

    const bool needsRebuild =
        sculpt_pbvh.object_name != editable_mesh_cache.object_name ||
        sculpt_pbvh.source_triangle_count != editable_mesh_cache.source_triangle_count ||
        !(sculpt_pbvh.source_object_transform == editable_mesh_cache.source_object_transform) ||
        sculpt_pbvh.source_vertex_to_leaf_id.size() != editable_mesh_cache.vertices.size();
    if (!needsRebuild) {
        return !sculpt_pbvh.nodes.empty();
    }

    sculpt_pbvh = SculptPBVH{};
    sculpt_pbvh.object_name = editable_mesh_cache.object_name;
    sculpt_pbvh.source_triangle_count = editable_mesh_cache.source_triangle_count;
    sculpt_pbvh.source_object_transform = editable_mesh_cache.source_object_transform;
    sculpt_pbvh.avg_edge_length = computeEditableAverageEdgeLength(editable_mesh_cache);

    const float safeAvgEdgeLength = sanitizeFiniteFloat(
        sculpt_pbvh.avg_edge_length,
        0.05f,
        0.0001f,
        100000.0f);
    const size_t vertexCount = editable_mesh_cache.vertices.size();
    if (vertexCount >= 200000) {
        sculpt_pbvh.leaf_vertex_limit = 32;
    } else if (vertexCount >= 50000) {
        sculpt_pbvh.leaf_vertex_limit = 48;
    } else {
        sculpt_pbvh.leaf_vertex_limit = 64;
    }
    if (safeAvgEdgeLength > 1.0f) {
        sculpt_pbvh.leaf_vertex_limit = (std::min)(sculpt_pbvh.leaf_vertex_limit + 16, 128);
    }

    buildSculptPBVH(sculpt_pbvh, editable_mesh_cache);
    return !sculpt_pbvh.nodes.empty();
}

Vec3 SceneUI::getSelectedMeshElementWorldPosition(UIContext& ctx, bool* valid) {
    bool isValid = false;
    Vec3 result(0.0f, 0.0f, 0.0f);

    const std::string objectName =
        (!active_mesh_edit_object_name.empty() ? active_mesh_edit_object_name :
            (ctx.selection.selected.type == SelectableType::Object && ctx.selection.selected.object
                ? ctx.selection.selected.object->getNodeName()
                : std::string{}));
    if (!objectName.empty() && ensureEditableMeshCache(ctx, objectName)) {
        const Matrix4x4 transform = getEditableObjectTransform(editable_mesh_cache);
        Vec3 accumulated(0.0f, 0.0f, 0.0f);
        int count = 0;

        if (!editable_mesh_cache.selection.vertex_ids.empty()) {
            for (const int vertexId : editable_mesh_cache.selection.vertex_ids) {
                if (vertexId >= 0 && vertexId < static_cast<int>(editable_mesh_cache.vertices.size())) {
                    accumulated += transform.transform_point(editable_mesh_cache.vertices[vertexId].local_position);
                    ++count;
                }
            }
        } else if (!editable_mesh_cache.selection.edge_ids.empty()) {
            for (const int edgeId : editable_mesh_cache.selection.edge_ids) {
                const auto* edge = getEditableSelectableEdge(editable_mesh_cache, edgeId);
                if (!edge) {
                    continue;
                }
                if (edge->v0 >= 0 && edge->v1 >= 0 &&
                    edge->v0 < static_cast<int>(editable_mesh_cache.vertices.size()) &&
                    edge->v1 < static_cast<int>(editable_mesh_cache.vertices.size())) {
                    const Vec3 p0 = transform.transform_point(editable_mesh_cache.vertices[edge->v0].local_position);
                    const Vec3 p1 = transform.transform_point(editable_mesh_cache.vertices[edge->v1].local_position);
                    accumulated += (p0 + p1) * 0.5f;
                    ++count;
                }
            }
        } else if (!editable_mesh_cache.selection.face_ids.empty()) {
            for (const int faceId : editable_mesh_cache.selection.face_ids) {
                const std::vector<int> vertexIds = getEditablePolygonVertexIds(editable_mesh_cache, faceId);
                if (vertexIds.size() < 3) {
                    continue;
                }
                Vec3 faceCenter(0.0f, 0.0f, 0.0f);
                int validVertexCount = 0;
                for (const int vertexId : vertexIds) {
                    if (!isEditableVertexIdValid(editable_mesh_cache, vertexId)) {
                        continue;
                    }
                    faceCenter += transform.transform_point(editable_mesh_cache.vertices[vertexId].local_position);
                    ++validVertexCount;
                }
                if (validVertexCount > 0) {
                    accumulated += faceCenter / static_cast<float>(validVertexCount);
                    ++count;
                }
            }
        }

        if (count > 0) {
            result = accumulated / static_cast<float>(count);
            isValid = true;
        }
    }

    if (valid) {
        *valid = isValid;
    }
    return result;
}

bool SceneUI::handleMeshElementSelection(UIContext& ctx, const ImVec2& mousePos) {
    if (!mesh_overlay_settings.enabled || !ctx.scene.camera) {
        return false;
    }
    if (ctx.selection.mesh_element_mode == MeshElementSelectMode::Object) {
        return false;
    }

    std::string objectName = active_mesh_edit_object_name;
    if (objectName.empty() &&
        ctx.selection.selected.type == SelectableType::Object &&
        ctx.selection.selected.object) {
        objectName = ctx.selection.selected.object->getNodeName();
    }
    if (objectName.empty()) {
        return false;
    }

    if (!ensureEditableMeshCache(ctx, objectName)) {
        return false;
    }

    const Matrix4x4 transform = getEditableObjectTransform(editable_mesh_cache);
    const ImVec2 displaySize = ImGui::GetIO().DisplaySize;

    EditableMeshSelection pickedSelection;
    bool handled = false;

    const MeshElementSelectMode requestedMode = ctx.selection.mesh_element_mode;
    const bool combinedMode = (requestedMode == MeshElementSelectMode::Combined);
    const bool tryVertex = combinedMode || requestedMode == MeshElementSelectMode::Vertex;
    const bool tryEdge   = combinedMode || requestedMode == MeshElementSelectMode::Edge;
    const bool tryFace   = combinedMode || requestedMode == MeshElementSelectMode::Face;

    // Each component type picks its own best candidate independently; in Combined
    // mode they are then resolved by priority (nearest vertex wins over edge,
    // edge over face), matching Blender's combined-select feel.
    const bool isOrtho = ctx.scene.camera->orthographic && viewport_settings.shading_mode != 2;
    int vertexCandidate = -1;
    int edgeCandidate = -1;
    int faceCandidate = -1;

    if (tryVertex) {
        const float maxDistanceSq = 14.0f * 14.0f;
        float bestDistanceSq = maxDistanceSq;
        for (size_t i = 0; i < editable_mesh_cache.vertices.size(); ++i) {
            ImVec2 screen;
            if (!projectPointToScreen(*ctx.scene.camera, displaySize,
                                      transform.transform_point(editable_mesh_cache.vertices[i].local_position), screen, isOrtho)) {
                continue;
            }

            const float dx = screen.x - mousePos.x;
            const float dy = screen.y - mousePos.y;
            const float distanceSq = dx * dx + dy * dy;
            if (distanceSq <= bestDistanceSq) {
                bestDistanceSq = distanceSq;
                vertexCandidate = static_cast<int>(i);
            }
        }
    }
    if (tryEdge) {
        const float maxDistanceSq = 12.0f * 12.0f;
        float bestDistanceSq = maxDistanceSq;
        const auto& selectableEdges =
            !editable_mesh_cache.polygon_edges.empty() ? editable_mesh_cache.polygon_edges : editable_mesh_cache.edges;
        for (size_t i = 0; i < selectableEdges.size(); ++i) {
            const auto& edge = selectableEdges[i];
            if (edge.v0 < 0 || edge.v1 < 0 ||
                edge.v0 >= static_cast<int>(editable_mesh_cache.vertices.size()) ||
                edge.v1 >= static_cast<int>(editable_mesh_cache.vertices.size())) {
                continue;
            }

            ImVec2 s0, s1;
            if (!projectPointToScreen(*ctx.scene.camera, displaySize,
                                      transform.transform_point(editable_mesh_cache.vertices[edge.v0].local_position), s0, isOrtho) ||
                !projectPointToScreen(*ctx.scene.camera, displaySize,
                                      transform.transform_point(editable_mesh_cache.vertices[edge.v1].local_position), s1, isOrtho)) {
                continue;
            }

            const float distanceSq = distancePointToSegmentSq(mousePos, s0, s1);
            if (distanceSq <= bestDistanceSq) {
                bestDistanceSq = distanceSq;
                edgeCandidate = static_cast<int>(i);
            }
        }
    }
    if (tryFace) {
        float bestDepth = (std::numeric_limits<float>::max)();
        const size_t polygonFaceCount =
            editable_mesh_cache.polygon_faces.empty()
                ? editable_mesh_cache.faces.size()
                : editable_mesh_cache.polygon_faces.size();
        for (size_t i = 0; i < polygonFaceCount; ++i) {
            const std::vector<int> vertexIds =
                getEditablePolygonVertexIds(editable_mesh_cache, static_cast<int>(i));
            if (vertexIds.size() < 3) {
                continue;
            }

            std::vector<Vec3> worldVertices;
            std::vector<ImVec2> screenVertices;
            worldVertices.reserve(vertexIds.size());
            screenVertices.reserve(vertexIds.size());
            bool projectionFailed = false;
            for (const int vertexId : vertexIds) {
                if (!isEditableVertexIdValid(editable_mesh_cache, vertexId)) {
                    projectionFailed = true;
                    break;
                }
                const Vec3 worldPosition =
                    transform.transform_point(editable_mesh_cache.vertices[vertexId].local_position);
                ImVec2 screenPosition;
                if (!projectPointToScreen(*ctx.scene.camera, displaySize, worldPosition, screenPosition, isOrtho)) {
                    projectionFailed = true;
                    break;
                }
                worldVertices.push_back(worldPosition);
                screenVertices.push_back(screenPosition);
            }
            if (projectionFailed) {
                continue;
            }

            bool containsPoint = false;
            for (size_t triIndex = 1; triIndex + 1 < screenVertices.size(); ++triIndex) {
                if (pointInTriangle2D(mousePos, screenVertices[0], screenVertices[triIndex], screenVertices[triIndex + 1])) {
                    containsPoint = true;
                    break;
                }
            }
            if (!containsPoint) {
                continue;
            }

            float depth = 0.0f;
            for (const Vec3& worldVertex : worldVertices) {
                depth += (worldVertex - ctx.scene.camera->lookfrom).length();
            }
            depth /= static_cast<float>(worldVertices.size());
            if (depth < bestDepth) {
                bestDepth = depth;
                faceCandidate = static_cast<int>(i);
            }
        }
    }

    // Resolve which component type the click actually lands on. Priority for
    // Combined mode: vertex > edge > face (a near vertex/edge should win over
    // the face it sits on). In a single-type mode only that candidate exists.
    MeshElementSelectMode resolvedMode = MeshElementSelectMode::Object;
    if (vertexCandidate >= 0) {
        resolvedMode = MeshElementSelectMode::Vertex;
        pickedSelection.active_vertex_id = vertexCandidate;
        handled = true;
    } else if (edgeCandidate >= 0) {
        resolvedMode = MeshElementSelectMode::Edge;
        pickedSelection.active_edge_id = edgeCandidate;
        handled = true;
    } else if (faceCandidate >= 0) {
        resolvedMode = MeshElementSelectMode::Face;
        pickedSelection.active_face_id = faceCandidate;
        handled = true;
    }

    if (handled) {
        const bool appendSelection = ImGui::GetIO().KeyCtrl;
        const bool ringSelection = ImGui::GetIO().KeyAlt && ImGui::GetIO().KeyShift;
        const bool loopSelection = ImGui::GetIO().KeyAlt && !ImGui::GetIO().KeyShift;
        EditableMeshSelection& selection = editable_mesh_cache.selection;

        if (resolvedMode == MeshElementSelectMode::Vertex && pickedSelection.active_vertex_id >= 0) {
            selection.active_edge_id = -1;
            selection.active_face_id = -1;
            selection.edge_ids.clear();
            selection.face_ids.clear();
            selection.active_vertex_id = pickedSelection.active_vertex_id;
            if (appendSelection) {
                toggleSelectionId(selection.vertex_ids, pickedSelection.active_vertex_id);
                if (!containsSelectionId(selection.vertex_ids, pickedSelection.active_vertex_id)) {
                    selection.active_vertex_id = selection.vertex_ids.empty() ? -1 : selection.vertex_ids.back();
                }
            } else {
                replaceSelectionId(selection.vertex_ids, pickedSelection.active_vertex_id);
            }
        } else if (resolvedMode == MeshElementSelectMode::Edge && pickedSelection.active_edge_id >= 0) {
            selection.active_vertex_id = -1;
            selection.active_face_id = -1;
            selection.vertex_ids.clear();
            selection.face_ids.clear();
            selection.active_edge_id = pickedSelection.active_edge_id;
            std::vector<int> pickedEdgeIds =
                ringSelection
                    ? collectEditableEdgeRing(editable_mesh_cache, pickedSelection.active_edge_id)
                    : (loopSelection
                    ? collectEditableEdgeLoop(editable_mesh_cache, pickedSelection.active_edge_id)
                    : std::vector<int>{ pickedSelection.active_edge_id });
            if (pickedEdgeIds.empty()) {
                pickedEdgeIds = { pickedSelection.active_edge_id };
            }
            if (appendSelection) {
                for (const int edgeId : pickedEdgeIds) {
                    toggleSelectionId(selection.edge_ids, edgeId);
                }
                if (!containsSelectionId(selection.edge_ids, pickedSelection.active_edge_id)) {
                    selection.active_edge_id = selection.edge_ids.empty() ? -1 : selection.edge_ids.back();
                }
            } else {
                selection.edge_ids = pickedEdgeIds;
            }
        } else if (resolvedMode == MeshElementSelectMode::Face && pickedSelection.active_face_id >= 0) {
            selection.active_vertex_id = -1;
            selection.active_edge_id = -1;
            selection.vertex_ids.clear();
            selection.edge_ids.clear();
            selection.active_face_id = pickedSelection.active_face_id;
            if (appendSelection) {
                toggleSelectionId(selection.face_ids, pickedSelection.active_face_id);
                if (!containsSelectionId(selection.face_ids, pickedSelection.active_face_id)) {
                    selection.active_face_id = selection.face_ids.empty() ? -1 : selection.face_ids.back();
                }
            } else {
                replaceSelectionId(selection.face_ids, pickedSelection.active_face_id);
            }
        }

        bool hasPosition = false;
        const Vec3 worldPos = getSelectedMeshElementWorldPosition(ctx, &hasPosition);
        if (hasPosition) {
            ctx.selection.selected.position = worldPos;
        }
        return true;
    }

    if (!ImGui::GetIO().KeyCtrl) {
        clearEditableMeshSelection();
    }
    return false;
}

bool SceneUI::applySelectedMeshElementTranslation(UIContext& ctx, const Vec3& worldDelta) {
    if (worldDelta.length_squared() < 1e-12f) {
        return false;
    }
    const std::string objectName =
        (!active_mesh_edit_object_name.empty() ? active_mesh_edit_object_name :
            (ctx.selection.selected.type == SelectableType::Object && ctx.selection.selected.object
                ? ctx.selection.selected.object->getNodeName()
                : std::string{}));
    if (objectName.empty()) {
        return false;
    }

    if (!ensureEditableMeshCache(ctx, objectName)) {
        return false;
    }
    const auto modifierIt = ctx.scene.mesh_modifiers.find(objectName);
    const bool preserveModifierPreview =
        modifierIt != ctx.scene.mesh_modifiers.end() &&
        hasEnabledSubdivisionPreview(modifierIt->second);
    if (!mesh_edit_layer.active || mesh_edit_layer.object_name != objectName) {
        ensureMeshEditLayer(ctx, objectName);
    }

    const Matrix4x4 transform = getEditableObjectTransform(editable_mesh_cache);
    const Matrix4x4 inverseTransform = transform.inverse();
    const Vec3 localDelta = inverseTransform.transform_vector(worldDelta);

    std::vector<int> targetVertices;
    if (!editable_mesh_cache.selection.vertex_ids.empty()) {
        targetVertices.insert(targetVertices.end(),
            editable_mesh_cache.selection.vertex_ids.begin(),
            editable_mesh_cache.selection.vertex_ids.end());
    } else if (!editable_mesh_cache.selection.edge_ids.empty()) {
        for (const int edgeId : editable_mesh_cache.selection.edge_ids) {
            const auto* edge = getEditableSelectableEdge(editable_mesh_cache, edgeId);
            if (!edge) {
                continue;
            }
            targetVertices.push_back(edge->v0);
            targetVertices.push_back(edge->v1);
        }
    } else if (!editable_mesh_cache.selection.face_ids.empty()) {
        for (const int faceId : editable_mesh_cache.selection.face_ids) {
            const std::vector<int> vertexIds = getEditablePolygonVertexIds(editable_mesh_cache, faceId);
            targetVertices.insert(targetVertices.end(), vertexIds.begin(), vertexIds.end());
        }
    }

    if (targetVertices.empty()) {
        return false;
    }
    if (mesh_edit_layer.active && !mesh_edit_layer.enabled) {
        return false;
    }

    std::vector<std::shared_ptr<Triangle>> touchedTriangles;
    touchedTriangles.reserve(targetVertices.size() * 2);
    beginEditableTriangleTouchPass(editable_mesh_cache);

    std::unordered_set<int> uniqueTargets(targetVertices.begin(), targetVertices.end());
    // Combined mode is plain selection — no proportional falloff during edits
    // (keeps the move local and matches the suppressed soft-select heat).
    MeshOverlaySettings softSettings = mesh_overlay_settings;
    if (ctx.selection.mesh_element_mode == MeshElementSelectMode::Combined) {
        softSettings.proportional_edit = false;
    }
    // Apply the local delta to one vertex (weight-scaled) and touch its triangles.
    auto applyTranslationToVertex = [&](size_t vertexId, float weight) {
        if (weight <= 1e-5f) {
            return;
        }
        EditableVertex& vertex = editable_mesh_cache.vertices[vertexId];
        vertex.local_position = vertex.local_position + (localDelta * weight);
        editable_mesh_cache.vertex_positions[vertexId] = vertex.local_position;
        for (const auto& ref : vertex.refs) {
            Triangle* refTri = editable_mesh_cache.refTri(ref);
            if (!refTri) {
                continue;
            }
            refTri->setOriginalVertexPosition(ref.corner, vertex.local_position);
            refTri->markAABBDirty();
            if (tryMarkEditableTriangleTouched(editable_mesh_cache, refTri)) {
                touchedTriangles.push_back(editable_mesh_cache.refTriShared(ref));
            }
        }
    };

    // PERF: proportional editing needs a falloff weight for every vertex, so it must scan
    // the whole mesh. A plain move (the common case) only touches the selected vertices —
    // iterate just those, O(selection) instead of O(N). On a multi-million-vertex mesh the
    // full scan + the N-float weight allocation was the dominant per-drag cost.
    if (softSettings.proportional_edit) {
        const std::vector<float> weights = buildSoftSelectionWeights(
            editable_mesh_cache, softSettings, uniqueTargets);
        for (size_t vertexId = 0; vertexId < editable_mesh_cache.vertices.size(); ++vertexId) {
            applyTranslationToVertex(vertexId, weights[vertexId]);
        }
    } else {
        for (const int targetId : uniqueTargets) {
            if (targetId >= 0 && static_cast<size_t>(targetId) < editable_mesh_cache.vertices.size()) {
                applyTranslationToVertex(static_cast<size_t>(targetId), 1.0f);
            }
        }
    }

    // Recalculate per-vertex smooth normals for affected vertices after element transform.
    {
        const std::vector<size_t> affectedVertexIds =
            collectAffectedEditableVertexIds(editable_mesh_cache, touchedTriangles);
        recomputeEditableSmoothNormals(editable_mesh_cache, affectedVertexIds);
    }

    for (const auto& tri : touchedTriangles) {
        tri->updateTransformedVertices();
    }

    // A position-only move does not change topology, so the overlay cache (which stores
    // triangle+corner SOURCES and re-projects live positions every frame) stays valid —
    // do NOT invalidate it here or the CPU fallback rebuilds its edge/vertex hash maps
    // (O(N)) every drag frame. It is rebuilt automatically when object/triangle count
    // changes (see drawMeshEditOverlay needsRebuild).
    updateBBoxCache(objectName);

    if (preserveModifierPreview) {
        // During an interactive cage drag, refresh ONLY the displayed subdivision surface;
        // do NOT rebuild the editable cache (it reshuffles position-sorted vertex ids and
        // reshuffles/loses the selection mid-drag → "can't move", crease lands on the wrong
        // edge). The cage positions are already updated in place above; the full rebuild
        // happens once on release (endInteractiveSubdivisionPreview).
        const bool rebuildCache = !isInteractiveSubdivisionPreviewActiveForObject(objectName);
        refreshEditableDisplayMeshFromBase(ctx, objectName, ctx.backend_ptr && sculpt_mode_state.accumulate_live, rebuildCache);
        bool hasPosition = false;
        const Vec3 worldPos = getSelectedMeshElementWorldPosition(ctx, &hasPosition);
        if (hasPosition) {
            ctx.selection.selected.position = worldPos;
        }
        return true;
    }

    // If this object was pending lazy CPU sync (e.g. after a Vulkan-mode scale),
    // flush ALL triangles' CPU vertices now – not just the touched ones – so the
    // BLAS update and any subsequent picking/BVH work see a fully consistent mesh.
    bool didFullCpuFlush = false;
    if (objects_needing_cpu_sync.count(objectName)) {
        auto cacheIt = mesh_cache.find(objectName);
        if (cacheIt != mesh_cache.end()) {
            for (auto& pair : cacheIt->second) {
                if (pair.second && tryMarkEditableTriangleTouched(editable_mesh_cache, pair.second.get())) {
                    pair.second->updateTransformedVertices();
                }
            }
        }
        objects_needing_cpu_sync.erase(objectName);
        didFullCpuFlush = true;
    }

    // PERF: hand the touched-triangle set to processPendingMeshEditGpuSync so it can
    // do a partial GPU upload (patchRasterMeshTriangles / updateMeshBLASPartial) instead
    // of re-extracting & re-uploading the whole mesh every drag frame — the same fast
    // path sculpt uses. Skip (leave empty → full-upload fallback) when we just flushed
    // the full mesh above, since those extra triangles aren't in touchedTriangles.
    sculpt_dirty_mesh_cache_indices.clear();
    if (!didFullCpuFlush) {
        sculpt_dirty_mesh_cache_indices.reserve(touchedTriangles.size());
        for (const auto& tri : touchedTriangles) {
            const int faceIdx = editableFaceIndexOf(editable_mesh_cache, tri.get());
            if (faceIdx >= 0) {
                sculpt_dirty_mesh_cache_indices.push_back(editable_mesh_cache.face_to_mesh_index[static_cast<size_t>(faceIdx)]);
            }
        }
    }

    if (!ctx.backend_ptr) {
        extern bool g_cpu_bvh_refit_pending;
        g_cpu_bvh_refit_pending = true;
    }

    if ((ctx.backend_ptr != nullptr || g_backend != nullptr || g_viewport_backend != nullptr) && sculpt_mode_state.accumulate_live) {
        queueMeshEditGpuSync(objectName);
    }

    ctx.renderer.resetCPUAccumulation();
    ctx.start_render = true;
    bool hasPosition = false;
    const Vec3 worldPos = getSelectedMeshElementWorldPosition(ctx, &hasPosition);
    if (hasPosition) {
        ctx.selection.selected.position = worldPos;
    }
    return true;
}

bool SceneUI::applySelectedMeshElementTransform(UIContext& ctx, const Matrix4x4& worldTransform) {
    if (worldTransform.isIdentity()) {
        return false;
    }

    const std::string objectName =
        (!active_mesh_edit_object_name.empty() ? active_mesh_edit_object_name :
            (ctx.selection.selected.type == SelectableType::Object && ctx.selection.selected.object
                ? ctx.selection.selected.object->getNodeName()
                : std::string{}));
    if (objectName.empty()) {
        return false;
    }

    if (!ensureEditableMeshCache(ctx, objectName)) {
        return false;
    }
    const auto modifierIt = ctx.scene.mesh_modifiers.find(objectName);
    const bool preserveModifierPreview =
        modifierIt != ctx.scene.mesh_modifiers.end() &&
        hasEnabledSubdivisionPreview(modifierIt->second);
    if (!mesh_edit_layer.active || mesh_edit_layer.object_name != objectName) {
        ensureMeshEditLayer(ctx, objectName);
    }
    if (mesh_edit_layer.active && !mesh_edit_layer.enabled) {
        return false;
    }

    const Matrix4x4 transform = getEditableObjectTransform(editable_mesh_cache);
    const Matrix4x4 inverseTransform = transform.inverse();

    std::vector<int> targetVertices;
    if (!editable_mesh_cache.selection.vertex_ids.empty()) {
        targetVertices.insert(targetVertices.end(),
            editable_mesh_cache.selection.vertex_ids.begin(),
            editable_mesh_cache.selection.vertex_ids.end());
    } else if (!editable_mesh_cache.selection.edge_ids.empty()) {
        for (const int edgeId : editable_mesh_cache.selection.edge_ids) {
            const auto* edge = getEditableSelectableEdge(editable_mesh_cache, edgeId);
            if (!edge) {
                continue;
            }
            targetVertices.push_back(edge->v0);
            targetVertices.push_back(edge->v1);
        }
    } else if (!editable_mesh_cache.selection.face_ids.empty()) {
        for (const int faceId : editable_mesh_cache.selection.face_ids) {
            const std::vector<int> vertexIds = getEditablePolygonVertexIds(editable_mesh_cache, faceId);
            targetVertices.insert(targetVertices.end(), vertexIds.begin(), vertexIds.end());
        }
    }

    if (targetVertices.empty()) {
        return false;
    }

    std::vector<std::shared_ptr<Triangle>> touchedTriangles;
    touchedTriangles.reserve(targetVertices.size() * 2);
    beginEditableTriangleTouchPass(editable_mesh_cache);

    std::unordered_set<int> uniqueTargets(targetVertices.begin(), targetVertices.end());
    // Combined mode is plain selection — no proportional falloff during edits
    // (keeps the move local and matches the suppressed soft-select heat).
    MeshOverlaySettings softSettings = mesh_overlay_settings;
    if (ctx.selection.mesh_element_mode == MeshElementSelectMode::Combined) {
        softSettings.proportional_edit = false;
    }
    bool changed = false;
    // Transform one vertex about the object's pivot (weight-scaled blend) and touch its
    // triangles. Returns whether the vertex actually moved.
    auto applyTransformToVertex = [&](size_t vertexId, float weight) {
        if (weight <= 1e-5f) {
            return;
        }
        EditableVertex& vertex = editable_mesh_cache.vertices[vertexId];
        const Vec3 currentLocal = vertex.local_position;
        const Vec3 currentWorld = sanitizeVec3(transform.transform_point(currentLocal), currentLocal);
        const Vec3 transformedWorld = sanitizeVec3(worldTransform.transform_point(currentWorld), currentWorld);
        const Vec3 blendedWorld = sanitizeVec3(
            currentWorld + (transformedWorld - currentWorld) * weight,
            currentWorld);
        const Vec3 localDelta = sanitizeVec3(
            inverseTransform.transform_vector(blendedWorld - currentWorld),
            Vec3(0.0f, 0.0f, 0.0f));
        const Vec3 updatedLocal = sanitizeVec3(currentLocal + localDelta, currentLocal);

        if ((updatedLocal - currentLocal).length_squared() <= 1e-12f) {
            return;
        }
        changed = true;
        vertex.local_position = updatedLocal;
        editable_mesh_cache.vertex_positions[vertexId] = updatedLocal;

        for (const auto& ref : vertex.refs) {
            Triangle* refTri = editable_mesh_cache.refTri(ref);
            if (!refTri) {
                continue;
            }
            refTri->setOriginalVertexPosition(ref.corner, updatedLocal);
            refTri->markAABBDirty();
            if (tryMarkEditableTriangleTouched(editable_mesh_cache, refTri)) {
                touchedTriangles.push_back(editable_mesh_cache.refTriShared(ref));
            }
        }
    };

    // PERF: see applySelectedMeshElementTranslation — only proportional editing needs the
    // O(N) whole-mesh scan; a plain element transform touches just the selection.
    if (softSettings.proportional_edit) {
        const std::vector<float> weights = buildSoftSelectionWeights(
            editable_mesh_cache, softSettings, uniqueTargets);
        for (size_t vertexId = 0; vertexId < editable_mesh_cache.vertices.size(); ++vertexId) {
            applyTransformToVertex(vertexId, weights[vertexId]);
        }
    } else {
        for (const int targetId : uniqueTargets) {
            if (targetId >= 0 && static_cast<size_t>(targetId) < editable_mesh_cache.vertices.size()) {
                applyTransformToVertex(static_cast<size_t>(targetId), 1.0f);
            }
        }
    }

    if (!changed) {
        return false;
    }

    {
        const std::vector<size_t> affectedVertexIds =
            collectAffectedEditableVertexIds(editable_mesh_cache, touchedTriangles);
        recomputeEditableSmoothNormals(editable_mesh_cache, affectedVertexIds);
    }

    for (const auto& tri : touchedTriangles) {
        tri->updateTransformedVertices();
    }

    // Position-only move keeps topology — leave the overlay cache valid (see the note in
    // applySelectedMeshElementTranslation); invalidating it forces an O(N) CPU rebuild.
    updateBBoxCache(objectName);

    if (preserveModifierPreview) {
        refreshEditableDisplayMeshFromBase(ctx, objectName, ctx.backend_ptr != nullptr);
        bool hasPosition = false;
        const Vec3 worldPos = getSelectedMeshElementWorldPosition(ctx, &hasPosition);
        if (hasPosition) {
            ctx.selection.selected.position = worldPos;
        }
        return true;
    }

    bool didFullCpuFlush = false;
    if (objects_needing_cpu_sync.count(objectName)) {
        auto cacheIt = mesh_cache.find(objectName);
        if (cacheIt != mesh_cache.end()) {
            for (auto& pair : cacheIt->second) {
                if (pair.second && tryMarkEditableTriangleTouched(editable_mesh_cache, pair.second.get())) {
                    pair.second->updateTransformedVertices();
                }
            }
        }
        objects_needing_cpu_sync.erase(objectName);
        didFullCpuFlush = true;
    }

    // PERF: partial GPU upload of only the touched triangles (see translation path).
    sculpt_dirty_mesh_cache_indices.clear();
    if (!didFullCpuFlush) {
        sculpt_dirty_mesh_cache_indices.reserve(touchedTriangles.size());
        for (const auto& tri : touchedTriangles) {
            const int faceIdx = editableFaceIndexOf(editable_mesh_cache, tri.get());
            if (faceIdx >= 0) {
                sculpt_dirty_mesh_cache_indices.push_back(editable_mesh_cache.face_to_mesh_index[static_cast<size_t>(faceIdx)]);
            }
        }
    }

    if (!ctx.backend_ptr) {
        extern bool g_cpu_bvh_refit_pending;
        g_cpu_bvh_refit_pending = true;
    }

    if (ctx.backend_ptr != nullptr || g_backend != nullptr || g_viewport_backend != nullptr) {
        queueMeshEditGpuSync(objectName);
    }

    ctx.renderer.resetCPUAccumulation();
    ctx.start_render = true;
    bool hasPosition = false;
    const Vec3 worldPos = getSelectedMeshElementWorldPosition(ctx, &hasPosition);
    if (hasPosition) {
        ctx.selection.selected.position = worldPos;
    }
    return true;
}

bool SceneUI::extrudeSelectedMeshFaces(UIContext& ctx, float distance) {
    if (std::fabs(distance) <= 1e-6f) {
        return false;
    }

    const std::string objectName =
        (!active_mesh_edit_object_name.empty() ? active_mesh_edit_object_name :
            (ctx.selection.selected.type == SelectableType::Object && ctx.selection.selected.object
                ? ctx.selection.selected.object->getNodeName()
                : std::string{}));
    if (objectName.empty()) {
        return false;
    }

    if (!ensureEditableMeshCache(ctx, objectName)) {
        return false;
    }
    if (editable_mesh_cache.selection.face_ids.empty()) {
        return false;
    }

    std::vector<std::shared_ptr<Triangle>> currentDisplayMesh;
    {
        auto meshIt = mesh_cache.find(objectName);
        if (meshIt == mesh_cache.end() || meshIt->second.empty()) {
            return false;
        }

        std::unordered_set<const Triangle*> seenTriangles;
        seenTriangles.reserve(meshIt->second.size());
        currentDisplayMesh.reserve(meshIt->second.size());
        for (const auto& entry : meshIt->second) {
            if (entry.second && seenTriangles.insert(entry.second.get()).second) {
                currentDisplayMesh.push_back(entry.second);
            }
        }
    }
    if (currentDisplayMesh.empty()) {
        return false;
    }

    std::vector<std::shared_ptr<Triangle>> currentBaseMesh;
    {
        auto baseIt = ctx.scene.base_mesh_cache.find(objectName);
        if (baseIt != ctx.scene.base_mesh_cache.end() && !baseIt->second.empty()) {
            currentBaseMesh = cloneTriangleVectorForEdit(baseIt->second);
        } else {
            currentBaseMesh = cloneTriangleVectorForEdit(currentDisplayMesh);
        }
    }

    const auto modifierIt = ctx.scene.mesh_modifiers.find(objectName);
    const MeshModifiers::ModifierStack beforeModifierStack =
        (modifierIt != ctx.scene.mesh_modifiers.end()) ? modifierIt->second : MeshModifiers::ModifierStack{};
    const bool preserveModifierPreview = hasEnabledSubdivisionPreview(beforeModifierStack);

    const std::vector<std::shared_ptr<Triangle>> beforeDisplayMesh = cloneTriangleVectorForEdit(currentDisplayMesh);

    std::vector<int> selectedFaceIds = editable_mesh_cache.selection.face_ids;
    std::sort(selectedFaceIds.begin(), selectedFaceIds.end());
    selectedFaceIds.erase(std::unique(selectedFaceIds.begin(), selectedFaceIds.end()), selectedFaceIds.end());

    // The extruded faces' original triangles become the bottom cap INSIDE the
    // new volume — they must not survive (the old behaviour left them in,
    // producing hidden internal geometry). Skip them while cloning the rest.
    std::unordered_set<const Triangle*> extrudedCapTriangles;
    for (const int faceId : selectedFaceIds) {
        if (faceId >= 0 && faceId < static_cast<int>(editable_mesh_cache.polygon_faces.size())) {
            for (const int triangleId : editable_mesh_cache.polygon_faces[faceId].triangle_ids) {
                if (triangleId >= 0 &&
                    triangleId < static_cast<int>(editable_mesh_cache.faces.size()) &&
                    editable_mesh_cache.faceTri(editable_mesh_cache.faces[triangleId])) {
                    extrudedCapTriangles.insert(editable_mesh_cache.faceTri(editable_mesh_cache.faces[triangleId]));
                }
            }
        } else if (faceId >= 0 && faceId < static_cast<int>(editable_mesh_cache.faces.size()) &&
                   editable_mesh_cache.faceTri(editable_mesh_cache.faces[faceId])) {
            extrudedCapTriangles.insert(editable_mesh_cache.faceTri(editable_mesh_cache.faces[faceId]));
        }
    }
    // The editable cache (and so extrudedCapTriangles) points at the same
    // source triangles the cache was built from: base_mesh_cache when a
    // subdivision preview is active, the live display mesh otherwise.
    const std::vector<std::shared_ptr<Triangle>>* extrudeSourceMesh = &currentDisplayMesh;
    if (preserveModifierPreview) {
        auto baseSourceIt = ctx.scene.base_mesh_cache.find(objectName);
        if (baseSourceIt != ctx.scene.base_mesh_cache.end() && !baseSourceIt->second.empty()) {
            extrudeSourceMesh = &baseSourceIt->second;
        }
    }
    std::vector<std::shared_ptr<Triangle>> extrudedMesh;
    extrudedMesh.reserve(extrudeSourceMesh->size());
    for (const auto& sourceTriangle : *extrudeSourceMesh) {
        if (!sourceTriangle || extrudedCapTriangles.count(sourceTriangle.get()) > 0) {
            continue;
        }
        if (auto clonedTriangle = cloneTriangleForEdit(sourceTriangle)) {
            extrudedMesh.push_back(clonedTriangle);
        }
    }

    struct ExtrudedFaceSelectionTarget {
        Vec3 center = Vec3(0.0f, 0.0f, 0.0f);
        Vec3 normal = Vec3(0.0f, 1.0f, 0.0f);
        int vertex_count = 0;
    };

    std::vector<ExtrudedFaceSelectionTarget> extrudedSelectionTargets;
    extrudedSelectionTargets.reserve(selectedFaceIds.size());

    int generatedTriangleCount = 0;
    for (const int faceId : selectedFaceIds) {
        const std::vector<int> vertexIds = getEditablePolygonVertexIds(editable_mesh_cache, faceId);
        if (vertexIds.size() < 3) {
            continue;
        }

        std::vector<Vec3> bottomVertices;
        bottomVertices.reserve(vertexIds.size());
        bool invalidFace = false;
        for (const int vertexId : vertexIds) {
            if (!isEditableVertexIdValid(editable_mesh_cache, vertexId)) {
                invalidFace = true;
                break;
            }
            bottomVertices.push_back(editable_mesh_cache.vertices[vertexId].local_position);
        }
        if (invalidFace) {
            continue;
        }

        Vec3 faceNormal = computeEditableFaceNormal(editable_mesh_cache, vertexIds);
        if (const SceneUI::EditablePolygonFace* polygonFace =
                (faceId >= 0 && faceId < static_cast<int>(editable_mesh_cache.polygon_faces.size()))
                    ? &editable_mesh_cache.polygon_faces[faceId]
                    : nullptr) {
            const Vec3 referenceNormal =
                computeEditableReferenceNormal(editable_mesh_cache, polygonFace->triangle_ids);
            if (faceNormal.dot(referenceNormal) < 0.0f) {
                faceNormal = faceNormal * -1.0f;
            }
        }
        const Vec3 extrudeVector = faceNormal * distance;
        std::vector<Vec3> topVertices;
        topVertices.reserve(bottomVertices.size());
        for (const Vec3& vertex : bottomVertices) {
            topVertices.push_back(vertex + extrudeVector);
        }

        ExtrudedFaceSelectionTarget selectionTarget;
        selectionTarget.vertex_count = static_cast<int>(topVertices.size());
        selectionTarget.normal = faceNormal;
        for (const Vec3& vertex : topVertices) {
            selectionTarget.center += vertex;
        }
        selectionTarget.center /= static_cast<float>((std::max)(1, selectionTarget.vertex_count));
        extrudedSelectionTargets.push_back(selectionTarget);

        const SceneUI::EditablePolygonFace* polygonFace =
            (faceId >= 0 && faceId < static_cast<int>(editable_mesh_cache.polygon_faces.size()))
                ? &editable_mesh_cache.polygon_faces[faceId]
                : nullptr;
        std::shared_ptr<Triangle> templateTriangle;
        if (polygonFace) {
            for (const int triangleId : polygonFace->triangle_ids) {
                if (triangleId >= 0 &&
                    triangleId < static_cast<int>(editable_mesh_cache.faces.size()) &&
                    editable_mesh_cache.faceTri(editable_mesh_cache.faces[triangleId])) {
                    templateTriangle = editable_mesh_cache.faceTriShared(editable_mesh_cache.faces[triangleId]);
                    break;
                }
            }
        }
        if (!templateTriangle) {
            templateTriangle = preserveModifierPreview ? currentBaseMesh.front() : currentDisplayMesh.front();
        }
        if (!templateTriangle) {
            continue;
        }

        const std::vector<Vec2> topUVs = buildPolygonPlanarUVs(topVertices, faceNormal);
        for (size_t i = 1; i + 1 < topVertices.size(); ++i) {
            auto topTriangle = cloneTriangleForEdit(templateTriangle);
            if (!topTriangle) {
                continue;
            }
            topTriangle->setOriginalVertexPosition(0, topVertices[0]);
            topTriangle->setOriginalVertexPosition(1, topVertices[i]);
            topTriangle->setOriginalVertexPosition(2, topVertices[i + 1]);
            topTriangle->setOriginalVertexNormal(0, faceNormal);
            topTriangle->setOriginalVertexNormal(1, faceNormal);
            topTriangle->setOriginalVertexNormal(2, faceNormal);
            topTriangle->set_normals(faceNormal, faceNormal, faceNormal);
            topTriangle->setUVCoordinates(topUVs[0], topUVs[i], topUVs[i + 1]);
            topTriangle->markAABBDirty();
            topTriangle->updateTransformedVertices();
            extrudedMesh.push_back(topTriangle);
            ++generatedTriangleCount;
        }

        for (size_t i = 0; i < bottomVertices.size(); ++i) {
            const size_t next = (i + 1) % bottomVertices.size();
            const Vec3& v0 = bottomVertices[i];
            const Vec3& v1 = bottomVertices[next];
            const Vec3& v2 = topVertices[next];
            const Vec3& v3 = topVertices[i];
            Vec3 sideNormal = (v1 - v0).cross(v3 - v0);
            if (sideNormal.length_squared() <= 1e-10f) {
                sideNormal = faceNormal;
            } else {
                sideNormal = sideNormal.normalize();
            }

            const auto quadUVs = buildExtrudedQuadUVs(v0, v1, v2, v3);
            auto sideTriangleA = cloneTriangleForEdit(templateTriangle);
            auto sideTriangleB = cloneTriangleForEdit(templateTriangle);
            if (!sideTriangleA || !sideTriangleB) {
                continue;
            }

            sideTriangleA->setOriginalVertexPosition(0, v0);
            sideTriangleA->setOriginalVertexPosition(1, v1);
            sideTriangleA->setOriginalVertexPosition(2, v2);
            sideTriangleA->setOriginalVertexNormal(0, sideNormal);
            sideTriangleA->setOriginalVertexNormal(1, sideNormal);
            sideTriangleA->setOriginalVertexNormal(2, sideNormal);
            sideTriangleA->set_normals(sideNormal, sideNormal, sideNormal);
            sideTriangleA->setUVCoordinates(quadUVs[0], quadUVs[1], quadUVs[2]);
            sideTriangleA->markAABBDirty();
            sideTriangleA->updateTransformedVertices();

            sideTriangleB->setOriginalVertexPosition(0, v0);
            sideTriangleB->setOriginalVertexPosition(1, v2);
            sideTriangleB->setOriginalVertexPosition(2, v3);
            sideTriangleB->setOriginalVertexNormal(0, sideNormal);
            sideTriangleB->setOriginalVertexNormal(1, sideNormal);
            sideTriangleB->setOriginalVertexNormal(2, sideNormal);
            sideTriangleB->set_normals(sideNormal, sideNormal, sideNormal);
            sideTriangleB->setUVCoordinates(quadUVs[0], quadUVs[2], quadUVs[3]);
            sideTriangleB->markAABBDirty();
            sideTriangleB->updateTransformedVertices();

            extrudedMesh.push_back(sideTriangleA);
            extrudedMesh.push_back(sideTriangleB);
            generatedTriangleCount += 2;
        }
    }

    if (generatedTriangleCount <= 0) {
        return false;
    }

    for (size_t triangleIndex = 0; triangleIndex < extrudedMesh.size(); ++triangleIndex) {
        if (extrudedMesh[triangleIndex]) {
            extrudedMesh[triangleIndex]->setFaceIndex(static_cast<int>(triangleIndex));
        }
    }

    const std::vector<std::shared_ptr<Triangle>> afterDisplayMesh = evaluateDisplayMeshFromBase(extrudedMesh, beforeModifierStack);
    const std::vector<std::shared_ptr<Triangle>> afterBaseMesh = cloneTriangleVectorForEdit(extrudedMesh);
    const MeshModifiers::ModifierStack afterModifierStack = beforeModifierStack;

    auto command = std::make_unique<ReplaceMeshGeometryCommand>(
        objectName,
        beforeDisplayMesh,
        afterDisplayMesh,
        currentBaseMesh,
        afterBaseMesh,
        beforeModifierStack,
        afterModifierStack);
    command->execute(ctx);
    history.record(std::move(command));
    rebuildMeshCache(ctx.scene.world.objects);
    applyMeshShadingSettings(ctx, objectName, true);

    editable_mesh_cache = EditableMeshCache{};
    mesh_overlay_cache = MeshOverlayCache{};
    mesh_edit_layer = MeshEditLayer{};
    pending_serialized_mesh_edit_layer = PendingSerializedMeshEditLayer{};
    clearEditableMeshSelection();
    active_mesh_edit_object_name = objectName;

    auto refreshedMeshIt = mesh_cache.find(objectName);
    if (refreshedMeshIt != mesh_cache.end() && !refreshedMeshIt->second.empty() && refreshedMeshIt->second.front().second) {
        ctx.selection.selectObject(refreshedMeshIt->second.front().second, -1, objectName);
        active_mesh_edit_object_ptr = refreshedMeshIt->second.front().second.get();
    } else {
        active_mesh_edit_object_ptr = nullptr;
    }

    ensureMeshEditLayer(ctx, objectName);
    if (ensureEditableMeshCache(ctx, objectName) && !extrudedSelectionTargets.empty()) {
        clearEditableMeshSelection();
        std::unordered_set<int> claimedFaceIds;
        for (const auto& target : extrudedSelectionTargets) {
            int bestFaceId = -1;
            float bestScore = std::numeric_limits<float>::max();

            for (size_t polygonFaceId = 0; polygonFaceId < editable_mesh_cache.polygon_faces.size(); ++polygonFaceId) {
                const int faceId = static_cast<int>(polygonFaceId);
                if (claimedFaceIds.count(faceId) > 0) {
                    continue;
                }

                const std::vector<int> vertexIds = getEditablePolygonVertexIds(editable_mesh_cache, faceId);
                if (static_cast<int>(vertexIds.size()) != target.vertex_count || vertexIds.size() < 3) {
                    continue;
                }

                Vec3 faceCenter(0.0f, 0.0f, 0.0f);
                bool invalidFace = false;
                for (const int vertexId : vertexIds) {
                    if (!isEditableVertexIdValid(editable_mesh_cache, vertexId)) {
                        invalidFace = true;
                        break;
                    }
                    faceCenter += editable_mesh_cache.vertices[vertexId].local_position;
                }
                if (invalidFace) {
                    continue;
                }
                faceCenter /= static_cast<float>(vertexIds.size());

                const Vec3 faceNormal = computeEditableFaceNormal(editable_mesh_cache, vertexIds);
                const float normalAlignment = faceNormal.dot(target.normal);
                if (!std::isfinite(normalAlignment) || normalAlignment < 0.9f) {
                    continue;
                }

                const float centerDistanceSq = (faceCenter - target.center).length_squared();
                const float score = centerDistanceSq + (1.0f - normalAlignment) * 0.05f;
                if (score < bestScore) {
                    bestScore = score;
                    bestFaceId = faceId;
                }
            }

            if (bestFaceId >= 0) {
                claimedFaceIds.insert(bestFaceId);
                editable_mesh_cache.selection.face_ids.push_back(bestFaceId);
            }
        }

        if (!editable_mesh_cache.selection.face_ids.empty()) {
            editable_mesh_cache.selection.active_face_id = editable_mesh_cache.selection.face_ids.back();
            // Keep Combined mode; only single-type modes snap to the result type.
            if (ctx.selection.mesh_element_mode != MeshElementSelectMode::Combined) {
                ctx.selection.mesh_element_mode = MeshElementSelectMode::Face;
            }
        }
    }
    addViewportMessage("Extrude Face baked current mesh", 2.2f, ImVec4(0.36f, 0.84f, 1.0f, 1.0f));
    return true;
}

bool SceneUI::loopCutSelectedEdges(UIContext& ctx, float t) {
    const float cutT = std::clamp(t, 0.01f, 0.99f);

    const std::string objectName =
        (!active_mesh_edit_object_name.empty() ? active_mesh_edit_object_name :
            (ctx.selection.selected.type == SelectableType::Object && ctx.selection.selected.object
                ? ctx.selection.selected.object->getNodeName()
                : std::string{}));
    if (objectName.empty()) {
        return false;
    }

    if (!ensureEditableMeshCache(ctx, objectName)) {
        return false;
    }

    std::vector<int> ringEdgeIds = editable_mesh_cache.selection.edge_ids;
    std::sort(ringEdgeIds.begin(), ringEdgeIds.end());
    ringEdgeIds.erase(std::unique(ringEdgeIds.begin(), ringEdgeIds.end()), ringEdgeIds.end());
    if (ringEdgeIds.empty() && editable_mesh_cache.selection.active_edge_id >= 0) {
        ringEdgeIds = collectEditableEdgeRing(editable_mesh_cache, editable_mesh_cache.selection.active_edge_id);
        if (ringEdgeIds.empty()) {
            ringEdgeIds.push_back(editable_mesh_cache.selection.active_edge_id);
        }
    }
    if (ringEdgeIds.empty()) {
        return false;
    }

    std::vector<std::shared_ptr<Triangle>> currentDisplayMesh;
    {
        auto meshIt = mesh_cache.find(objectName);
        if (meshIt == mesh_cache.end() || meshIt->second.empty()) {
            return false;
        }

        std::unordered_set<const Triangle*> seenTriangles;
        seenTriangles.reserve(meshIt->second.size());
        currentDisplayMesh.reserve(meshIt->second.size());
        for (const auto& entry : meshIt->second) {
            if (entry.second && seenTriangles.insert(entry.second.get()).second) {
                currentDisplayMesh.push_back(entry.second);
            }
        }
    }
    if (currentDisplayMesh.empty()) {
        return false;
    }

    std::vector<std::shared_ptr<Triangle>> currentBaseMesh;
    {
        auto baseIt = ctx.scene.base_mesh_cache.find(objectName);
        if (baseIt != ctx.scene.base_mesh_cache.end() && !baseIt->second.empty()) {
            currentBaseMesh = cloneTriangleVectorForEdit(baseIt->second);
        } else {
            currentBaseMesh = cloneTriangleVectorForEdit(currentDisplayMesh);
        }
    }

    const auto modifierIt = ctx.scene.mesh_modifiers.find(objectName);
    const MeshModifiers::ModifierStack beforeModifierStack =
        (modifierIt != ctx.scene.mesh_modifiers.end()) ? modifierIt->second : MeshModifiers::ModifierStack{};
    const bool preserveModifierPreview = hasEnabledSubdivisionPreview(beforeModifierStack);
    const std::vector<std::shared_ptr<Triangle>> beforeDisplayMesh = cloneTriangleVectorForEdit(currentDisplayMesh);

    // ---- half-edge loop cut ----------------------------------------------
    // The cut is pure topology on the half-edge core: walk the quad ring
    // from each selected edge, split the ring edges at cutT and connect the
    // new vertices across each quad. The triangle soup is then rebuilt from
    // the cut topology with the same planar-UV / flat-normal convention the
    // legacy rebuild used.
    ensureEditableHalfEdge(); // lazy: half-edge is no longer built at cache-build time
    if (!editable_mesh_cache.half_edge_valid ||
        editable_mesh_cache.half_edge_build.skipped_polygons > 0) {
        addViewportMessage("Loop cut: mesh topology unavailable",
                           2.4f, ImVec4(1.0f, 0.62f, 0.3f, 1.0f));
        return false;
    }

    struct LoopCutSelectionTarget {
        Vec3 center = Vec3(0.0f, 0.0f, 0.0f);
        Vec3 direction = Vec3(1.0f, 0.0f, 0.0f);
    };

    MeshEdit::HalfEdgeMesh heMesh = editable_mesh_cache.half_edge; // mutate a copy
    syncHalfEdgePositionsFromCache(heMesh, editable_mesh_cache); // pick up vertex edits
    const MeshEdit::HEIndex originalFaceCount =
        static_cast<MeshEdit::HEIndex>(heMesh.faces.size());

    std::vector<LoopCutSelectionTarget> cutSelectionTargets;
    std::unordered_map<MeshEdit::HEIndex, MeshEdit::HEIndex> faceSourceMap;
    int cutRingCount = 0;
    for (const int edgeId : ringEdgeIds) {
        if (edgeId < 0 || edgeId >= static_cast<int>(editable_mesh_cache.polygon_edges.size())) {
            continue;
        }
        const auto& cacheEdge = editable_mesh_cache.polygon_edges[edgeId];
        // Cache vertex ids are 1:1 with half-edge vertex ids. findEdge fails
        // naturally for edges already consumed (split) by a previous ring.
        const MeshEdit::HEIndex heEdge = heMesh.findEdge(cacheEdge.v0, cacheEdge.v1);
        if (heEdge == MeshEdit::kHEInvalid) {
            continue;
        }
        MeshEdit::HalfEdgeMesh::LoopCutResult cut;
        if (!heMesh.loopCut(heEdge, cutT, &cut)) {
            continue;
        }
        ++cutRingCount;
        for (const auto& split : cut.face_splits) {
            faceSourceMap[split.second] = split.first;
        }
        for (const MeshEdit::HEIndex cutEdge : cut.new_edges) {
            const MeshEdit::HEIndex he = heMesh.edges[cutEdge].half_edge;
            const Vec3 p0 = heMesh.vertices[heMesh.half_edges[he].origin].position;
            const Vec3 p1 = heMesh.vertices[heMesh.headVertex(he)].position;
            LoopCutSelectionTarget target;
            target.center = (p0 + p1) * 0.5f;
            const Vec3 direction = p1 - p0;
            const float dirLenSq = direction.length_squared();
            target.direction =
                (std::isfinite(dirLenSq) && dirLenSq > 1e-10f)
                    ? direction / std::sqrt(dirLenSq)
                    : Vec3(1.0f, 0.0f, 0.0f);
            cutSelectionTargets.push_back(target);
        }
    }
    if (cutRingCount <= 0) {
        return false;
    }

    std::vector<std::shared_ptr<Triangle>> cutMesh = rebuildTriangleSoupFromHalfEdge(
        heMesh,
        editable_mesh_cache,
        preserveModifierPreview ? currentBaseMesh : currentDisplayMesh,
        originalFaceCount,
        faceSourceMap);
    if (cutMesh.empty()) {
        return false;
    }

    for (size_t triangleIndex = 0; triangleIndex < cutMesh.size(); ++triangleIndex) {
        if (cutMesh[triangleIndex]) {
            cutMesh[triangleIndex]->setFaceIndex(static_cast<int>(triangleIndex));
        }
    }

    const std::vector<std::shared_ptr<Triangle>> afterDisplayMesh = evaluateDisplayMeshFromBase(cutMesh, beforeModifierStack);
    const std::vector<std::shared_ptr<Triangle>> afterBaseMesh = cloneTriangleVectorForEdit(cutMesh);
    const MeshModifiers::ModifierStack afterModifierStack = beforeModifierStack;

    auto command = std::make_unique<ReplaceMeshGeometryCommand>(
        objectName,
        beforeDisplayMesh,
        afterDisplayMesh,
        currentBaseMesh,
        afterBaseMesh,
        beforeModifierStack,
        afterModifierStack);
    command->execute(ctx);
    history.record(std::move(command));
    rebuildMeshCache(ctx.scene.world.objects);
    applyMeshShadingSettings(ctx, objectName, true);

    editable_mesh_cache = EditableMeshCache{};
    mesh_overlay_cache = MeshOverlayCache{};
    mesh_edit_layer = MeshEditLayer{};
    pending_serialized_mesh_edit_layer = PendingSerializedMeshEditLayer{};
    clearEditableMeshSelection();
    active_mesh_edit_object_name = objectName;

    auto refreshedMeshIt = mesh_cache.find(objectName);
    if (refreshedMeshIt != mesh_cache.end() && !refreshedMeshIt->second.empty() && refreshedMeshIt->second.front().second) {
        ctx.selection.selectObject(refreshedMeshIt->second.front().second, -1, objectName);
        active_mesh_edit_object_ptr = refreshedMeshIt->second.front().second.get();
    } else {
        active_mesh_edit_object_ptr = nullptr;
    }

    ensureMeshEditLayer(ctx, objectName);
    if (ensureEditableMeshCache(ctx, objectName) && !cutSelectionTargets.empty()) {
        std::unordered_set<int> claimedEdgeIds;
        std::vector<int> selectedNewEdges;
        for (const auto& target : cutSelectionTargets) {
            int bestEdgeId = -1;
            float bestScore = std::numeric_limits<float>::max();
            for (size_t edgeId = 0; edgeId < editable_mesh_cache.polygon_edges.size(); ++edgeId) {
                const auto& edge = editable_mesh_cache.polygon_edges[edgeId];
                if (!isEditableVertexIdValid(editable_mesh_cache, edge.v0) ||
                    !isEditableVertexIdValid(editable_mesh_cache, edge.v1) ||
                    claimedEdgeIds.count(static_cast<int>(edgeId)) > 0) {
                    continue;
                }

                const Vec3 p0 = editable_mesh_cache.vertices[edge.v0].local_position;
                const Vec3 p1 = editable_mesh_cache.vertices[edge.v1].local_position;
                const Vec3 center = (p0 + p1) * 0.5f;
                Vec3 direction = p1 - p0;
                const float dirLenSq = direction.length_squared();
                if (!std::isfinite(dirLenSq) || dirLenSq <= 1e-10f) {
                    continue;
                }
                direction /= std::sqrt(dirLenSq);

                const float centerError = (center - target.center).length_squared();
                const float alignmentPenalty = 1.0f - std::fabs(direction.dot(target.direction));
                const float score = centerError + alignmentPenalty * 0.02f;
                if (score < bestScore) {
                    bestScore = score;
                    bestEdgeId = static_cast<int>(edgeId);
                }
            }

            if (bestEdgeId >= 0) {
                claimedEdgeIds.insert(bestEdgeId);
                selectedNewEdges.push_back(bestEdgeId);
            }
        }

        std::sort(selectedNewEdges.begin(), selectedNewEdges.end());
        selectedNewEdges.erase(std::unique(selectedNewEdges.begin(), selectedNewEdges.end()), selectedNewEdges.end());
        editable_mesh_cache.selection.edge_ids = selectedNewEdges;
        editable_mesh_cache.selection.active_edge_id = selectedNewEdges.empty() ? -1 : selectedNewEdges.front();
        editable_mesh_cache.selection.vertex_ids.clear();
        editable_mesh_cache.selection.face_ids.clear();
        editable_mesh_cache.selection.active_vertex_id = -1;
        editable_mesh_cache.selection.active_face_id = -1;
        // Keep Combined mode; only single-type modes snap to the result type.
        if (ctx.selection.mesh_element_mode != MeshElementSelectMode::Combined) {
            ctx.selection.mesh_element_mode = MeshElementSelectMode::Edge;
        }
    }

    addViewportMessage("Loop cut applied", 2.0f, ImVec4(0.34f, 0.84f, 1.0f, 1.0f));
    return true;
}

bool SceneUI::insetSelectedMeshFaces(UIContext& ctx, float amount) {
    const float insetT = std::clamp(amount, 0.02f, 0.98f);

    const std::string objectName =
        (!active_mesh_edit_object_name.empty() ? active_mesh_edit_object_name :
            (ctx.selection.selected.type == SelectableType::Object && ctx.selection.selected.object
                ? ctx.selection.selected.object->getNodeName()
                : std::string{}));
    if (objectName.empty() || !ensureEditableMeshCache(ctx, objectName)) {
        return false;
    }

    std::vector<int> selectedFaceIds = editable_mesh_cache.selection.face_ids;
    std::sort(selectedFaceIds.begin(), selectedFaceIds.end());
    selectedFaceIds.erase(std::unique(selectedFaceIds.begin(), selectedFaceIds.end()), selectedFaceIds.end());
    if (selectedFaceIds.empty() && editable_mesh_cache.selection.active_face_id >= 0) {
        selectedFaceIds.push_back(editable_mesh_cache.selection.active_face_id);
    }
    if (selectedFaceIds.empty()) {
        return false;
    }

    std::vector<std::shared_ptr<Triangle>> currentDisplayMesh;
    {
        auto meshIt = mesh_cache.find(objectName);
        if (meshIt == mesh_cache.end() || meshIt->second.empty()) {
            return false;
        }
        std::unordered_set<const Triangle*> seenTriangles;
        currentDisplayMesh.reserve(meshIt->second.size());
        for (const auto& entry : meshIt->second) {
            if (entry.second && seenTriangles.insert(entry.second.get()).second) {
                currentDisplayMesh.push_back(entry.second);
            }
        }
    }
    if (currentDisplayMesh.empty()) {
        return false;
    }

    std::vector<std::shared_ptr<Triangle>> currentBaseMesh;
    {
        auto baseIt = ctx.scene.base_mesh_cache.find(objectName);
        currentBaseMesh = (baseIt != ctx.scene.base_mesh_cache.end() && !baseIt->second.empty())
            ? cloneTriangleVectorForEdit(baseIt->second)
            : cloneTriangleVectorForEdit(currentDisplayMesh);
    }
    const auto modifierIt = ctx.scene.mesh_modifiers.find(objectName);
    const MeshModifiers::ModifierStack beforeModifierStack =
        (modifierIt != ctx.scene.mesh_modifiers.end()) ? modifierIt->second : MeshModifiers::ModifierStack{};
    const bool preserveModifierPreview = hasEnabledSubdivisionPreview(beforeModifierStack);
    const std::vector<std::shared_ptr<Triangle>> beforeDisplayMesh = cloneTriangleVectorForEdit(currentDisplayMesh);

    ensureEditableHalfEdge(); // lazy: half-edge is no longer built at cache-build time
    if (!editable_mesh_cache.half_edge_valid ||
        editable_mesh_cache.half_edge_build.skipped_polygons > 0) {
        addViewportMessage("Inset: mesh topology unavailable",
                           2.4f, ImVec4(1.0f, 0.62f, 0.3f, 1.0f));
        return false;
    }

    MeshEdit::HalfEdgeMesh heMesh = editable_mesh_cache.half_edge;
    syncHalfEdgePositionsFromCache(heMesh, editable_mesh_cache);
    const MeshEdit::HEIndex originalFaceCount =
        static_cast<MeshEdit::HEIndex>(heMesh.faces.size());

    std::unordered_map<MeshEdit::HEIndex, MeshEdit::HEIndex> faceSourceMap;
    std::vector<MeshEdit::HEIndex> sideFaces;
    int insetCount = 0;
    for (const int faceId : selectedFaceIds) {
        if (faceId < 0 || faceId >= originalFaceCount || heMesh.faces[faceId].removed) {
            continue;
        }
        if (heMesh.insetFace(faceId, insetT, &sideFaces) == MeshEdit::kHEInvalid) {
            continue;
        }
        for (const MeshEdit::HEIndex sideFace : sideFaces) {
            faceSourceMap[sideFace] = faceId;
        }
        ++insetCount;
    }
    if (insetCount <= 0) {
        return false;
    }

    std::vector<std::shared_ptr<Triangle>> insetMesh = rebuildTriangleSoupFromHalfEdge(
        heMesh,
        editable_mesh_cache,
        preserveModifierPreview ? currentBaseMesh : currentDisplayMesh,
        originalFaceCount,
        faceSourceMap);
    if (insetMesh.empty()) {
        return false;
    }
    for (size_t triangleIndex = 0; triangleIndex < insetMesh.size(); ++triangleIndex) {
        if (insetMesh[triangleIndex]) {
            insetMesh[triangleIndex]->setFaceIndex(static_cast<int>(triangleIndex));
        }
    }

    const std::vector<std::shared_ptr<Triangle>> afterDisplayMesh = evaluateDisplayMeshFromBase(insetMesh, beforeModifierStack);
    const std::vector<std::shared_ptr<Triangle>> afterBaseMesh = cloneTriangleVectorForEdit(insetMesh);
    const MeshModifiers::ModifierStack afterModifierStack = beforeModifierStack;
    auto command = std::make_unique<ReplaceMeshGeometryCommand>(
        objectName, beforeDisplayMesh, afterDisplayMesh, currentBaseMesh, afterBaseMesh,
        beforeModifierStack, afterModifierStack);
    command->execute(ctx);
    history.record(std::move(command));
    rebuildMeshCache(ctx.scene.world.objects);
    applyMeshShadingSettings(ctx, objectName, true);

    editable_mesh_cache = EditableMeshCache{};
    mesh_overlay_cache = MeshOverlayCache{};
    mesh_edit_layer = MeshEditLayer{};
    pending_serialized_mesh_edit_layer = PendingSerializedMeshEditLayer{};
    clearEditableMeshSelection();
    active_mesh_edit_object_name = objectName;
    ensureMeshEditLayer(ctx, objectName);
    ensureEditableMeshCache(ctx, objectName);
    addViewportMessage("Inset faces applied", 2.0f, ImVec4(0.34f, 0.84f, 1.0f, 1.0f));
    return true;
}

bool SceneUI::dissolveSelectedEdges(UIContext& ctx) {
    const std::string objectName =
        (!active_mesh_edit_object_name.empty() ? active_mesh_edit_object_name :
            (ctx.selection.selected.type == SelectableType::Object && ctx.selection.selected.object
                ? ctx.selection.selected.object->getNodeName()
                : std::string{}));
    if (objectName.empty() || !ensureEditableMeshCache(ctx, objectName)) {
        return false;
    }

    std::vector<int> selectedEdgeIds = editable_mesh_cache.selection.edge_ids;
    std::sort(selectedEdgeIds.begin(), selectedEdgeIds.end());
    selectedEdgeIds.erase(std::unique(selectedEdgeIds.begin(), selectedEdgeIds.end()), selectedEdgeIds.end());
    if (selectedEdgeIds.empty()) {
        return false;
    }

    std::vector<std::shared_ptr<Triangle>> currentDisplayMesh;
    {
        auto meshIt = mesh_cache.find(objectName);
        if (meshIt == mesh_cache.end() || meshIt->second.empty()) {
            return false;
        }
        std::unordered_set<const Triangle*> seenTriangles;
        currentDisplayMesh.reserve(meshIt->second.size());
        for (const auto& entry : meshIt->second) {
            if (entry.second && seenTriangles.insert(entry.second.get()).second) {
                currentDisplayMesh.push_back(entry.second);
            }
        }
    }
    if (currentDisplayMesh.empty()) {
        return false;
    }

    std::vector<std::shared_ptr<Triangle>> currentBaseMesh;
    {
        auto baseIt = ctx.scene.base_mesh_cache.find(objectName);
        currentBaseMesh = (baseIt != ctx.scene.base_mesh_cache.end() && !baseIt->second.empty())
            ? cloneTriangleVectorForEdit(baseIt->second)
            : cloneTriangleVectorForEdit(currentDisplayMesh);
    }
    const auto modifierIt = ctx.scene.mesh_modifiers.find(objectName);
    const MeshModifiers::ModifierStack beforeModifierStack =
        (modifierIt != ctx.scene.mesh_modifiers.end()) ? modifierIt->second : MeshModifiers::ModifierStack{};
    const bool preserveModifierPreview = hasEnabledSubdivisionPreview(beforeModifierStack);
    const std::vector<std::shared_ptr<Triangle>> beforeDisplayMesh = cloneTriangleVectorForEdit(currentDisplayMesh);

    // ---- half-edge dissolve ------------------------------------------------
    // Pure topology: dissolveEdge merges the two adjacent faces in the
    // half-edge core. Chains of selected edges merge transitively; bridges
    // and multi-shared edges are rejected by the core. No vertices are
    // created or moved, so surviving face loops still use cache vertex ids
    // and the soup rebuild reads live cache positions directly.
    ensureEditableHalfEdge(); // lazy: half-edge is no longer built at cache-build time
    if (!editable_mesh_cache.half_edge_valid ||
        editable_mesh_cache.half_edge_build.skipped_polygons > 0) {
        addViewportMessage("Dissolve: mesh topology unavailable",
                           2.4f, ImVec4(1.0f, 0.62f, 0.3f, 1.0f));
        return false;
    }

    MeshEdit::HalfEdgeMesh heMesh = editable_mesh_cache.half_edge;
    int dissolvedCount = 0;
    for (const int edgeId : selectedEdgeIds) {
        if (edgeId < 0 || edgeId >= static_cast<int>(editable_mesh_cache.polygon_edges.size())) {
            continue;
        }
        const auto& cacheEdge = editable_mesh_cache.polygon_edges[edgeId];
        const MeshEdit::HEIndex heEdge = heMesh.findEdge(cacheEdge.v0, cacheEdge.v1);
        if (heEdge == MeshEdit::kHEInvalid) {
            continue;
        }
        if (heMesh.dissolveEdge(heEdge) != MeshEdit::kHEInvalid) {
            ++dissolvedCount;
        }
    }
    if (dissolvedCount <= 0) {
        return false;
    }

    // Dissolve only removes elements, so every surviving face id still maps
    // 1:1 to its cache polygon face (template/material lookup stays direct).
    std::vector<std::shared_ptr<Triangle>> dissolvedMesh;
    std::vector<MeshEdit::HEIndex> faceVertexIds;
    std::vector<int> polygonVertexIds;
    for (MeshEdit::HEIndex f = 0; f < static_cast<MeshEdit::HEIndex>(heMesh.faces.size()); ++f) {
        if (heMesh.faces[f].removed) {
            continue;
        }
        heMesh.collectFaceVertices(f, faceVertexIds);
        polygonVertexIds.assign(faceVertexIds.begin(), faceVertexIds.end());
        appendTriangulatedEditablePolygon(
            editable_mesh_cache,
            polygonVertexIds,
            resolveEditablePolygonTemplateTriangle(
                editable_mesh_cache, currentDisplayMesh, static_cast<int>(f)),
            dissolvedMesh);
    }
    if (dissolvedMesh.empty()) {
        return false;
    }
    for (size_t triangleIndex = 0; triangleIndex < dissolvedMesh.size(); ++triangleIndex) {
        if (dissolvedMesh[triangleIndex]) {
            dissolvedMesh[triangleIndex]->setFaceIndex(static_cast<int>(triangleIndex));
        }
    }

    const std::vector<std::shared_ptr<Triangle>> afterDisplayMesh = evaluateDisplayMeshFromBase(dissolvedMesh, beforeModifierStack);
    const std::vector<std::shared_ptr<Triangle>> afterBaseMesh = cloneTriangleVectorForEdit(dissolvedMesh);
    const MeshModifiers::ModifierStack afterModifierStack = beforeModifierStack;
    auto command = std::make_unique<ReplaceMeshGeometryCommand>(
        objectName, beforeDisplayMesh, afterDisplayMesh, currentBaseMesh, afterBaseMesh,
        beforeModifierStack, afterModifierStack);
    command->execute(ctx);
    history.record(std::move(command));
    rebuildMeshCache(ctx.scene.world.objects);
    applyMeshShadingSettings(ctx, objectName, true);

    editable_mesh_cache = EditableMeshCache{};
    mesh_overlay_cache = MeshOverlayCache{};
    mesh_edit_layer = MeshEditLayer{};
    pending_serialized_mesh_edit_layer = PendingSerializedMeshEditLayer{};
    clearEditableMeshSelection();
    active_mesh_edit_object_name = objectName;
    ensureMeshEditLayer(ctx, objectName);
    ensureEditableMeshCache(ctx, objectName);
    addViewportMessage("Dissolved selected edges", 2.0f, ImVec4(0.34f, 0.84f, 1.0f, 1.0f));
    return true;
}

bool SceneUI::dissolveSelectedVertices(UIContext& ctx) {
    const std::string objectName =
        (!active_mesh_edit_object_name.empty() ? active_mesh_edit_object_name :
            (ctx.selection.selected.type == SelectableType::Object && ctx.selection.selected.object
                ? ctx.selection.selected.object->getNodeName()
                : std::string{}));
    if (objectName.empty() || !ensureEditableMeshCache(ctx, objectName)) {
        return false;
    }

    std::vector<int> selectedVertexIds = editable_mesh_cache.selection.vertex_ids;
    std::sort(selectedVertexIds.begin(), selectedVertexIds.end());
    selectedVertexIds.erase(std::unique(selectedVertexIds.begin(), selectedVertexIds.end()), selectedVertexIds.end());
    if (selectedVertexIds.empty()) {
        return false;
    }
    std::unordered_set<int> selectedVertexSet(selectedVertexIds.begin(), selectedVertexIds.end());

    std::vector<std::shared_ptr<Triangle>> currentDisplayMesh;
    {
        auto meshIt = mesh_cache.find(objectName);
        if (meshIt == mesh_cache.end() || meshIt->second.empty()) {
            return false;
        }
        std::unordered_set<const Triangle*> seenTriangles;
        currentDisplayMesh.reserve(meshIt->second.size());
        for (const auto& entry : meshIt->second) {
            if (entry.second && seenTriangles.insert(entry.second.get()).second) {
                currentDisplayMesh.push_back(entry.second);
            }
        }
    }
    if (currentDisplayMesh.empty()) {
        return false;
    }

    std::vector<std::shared_ptr<Triangle>> currentBaseMesh;
    {
        auto baseIt = ctx.scene.base_mesh_cache.find(objectName);
        currentBaseMesh = (baseIt != ctx.scene.base_mesh_cache.end() && !baseIt->second.empty())
            ? cloneTriangleVectorForEdit(baseIt->second)
            : cloneTriangleVectorForEdit(currentDisplayMesh);
    }
    const auto modifierIt = ctx.scene.mesh_modifiers.find(objectName);
    const MeshModifiers::ModifierStack beforeModifierStack =
        (modifierIt != ctx.scene.mesh_modifiers.end()) ? modifierIt->second : MeshModifiers::ModifierStack{};
    const bool preserveModifierPreview = hasEnabledSubdivisionPreview(beforeModifierStack);
    const std::vector<std::shared_ptr<Triangle>> beforeDisplayMesh = cloneTriangleVectorForEdit(currentDisplayMesh);

    bool changed = false;
    std::vector<std::pair<std::vector<int>, std::shared_ptr<Triangle>>> outputPolygons;
    outputPolygons.reserve(editable_mesh_cache.polygon_faces.size());
    for (size_t faceId = 0; faceId < editable_mesh_cache.polygon_faces.size(); ++faceId) {
        std::vector<int> reducedVertexIds;
        const auto& faceVertexIds = editable_mesh_cache.polygon_faces[faceId].vertex_ids;
        reducedVertexIds.reserve(faceVertexIds.size());
        for (const int vertexId : faceVertexIds) {
            if (selectedVertexSet.count(vertexId) == 0) {
                if (reducedVertexIds.empty() || reducedVertexIds.back() != vertexId) {
                    reducedVertexIds.push_back(vertexId);
                }
            } else {
                changed = true;
            }
        }
        if (reducedVertexIds.size() >= 2 && reducedVertexIds.front() == reducedVertexIds.back()) {
            reducedVertexIds.pop_back();
        }
        if (reducedVertexIds.size() < 3) {
            continue;
        }
        outputPolygons.emplace_back(
            reducedVertexIds,
            resolveEditablePolygonTemplateTriangle(editable_mesh_cache, currentDisplayMesh, static_cast<int>(faceId)));
    }

    if (!changed) {
        return false;
    }

    std::vector<std::shared_ptr<Triangle>> dissolvedMesh;
    for (const auto& polygon : outputPolygons) {
        appendTriangulatedEditablePolygon(editable_mesh_cache, polygon.first, polygon.second, dissolvedMesh);
    }
    if (dissolvedMesh.empty()) {
        return false;
    }
    for (size_t triangleIndex = 0; triangleIndex < dissolvedMesh.size(); ++triangleIndex) {
        if (dissolvedMesh[triangleIndex]) {
            dissolvedMesh[triangleIndex]->setFaceIndex(static_cast<int>(triangleIndex));
        }
    }

    const std::vector<std::shared_ptr<Triangle>> afterDisplayMesh = evaluateDisplayMeshFromBase(dissolvedMesh, beforeModifierStack);
    const std::vector<std::shared_ptr<Triangle>> afterBaseMesh = cloneTriangleVectorForEdit(dissolvedMesh);
    const MeshModifiers::ModifierStack afterModifierStack = beforeModifierStack;
    auto command = std::make_unique<ReplaceMeshGeometryCommand>(
        objectName, beforeDisplayMesh, afterDisplayMesh, currentBaseMesh, afterBaseMesh,
        beforeModifierStack, afterModifierStack);
    command->execute(ctx);
    history.record(std::move(command));
    rebuildMeshCache(ctx.scene.world.objects);
    applyMeshShadingSettings(ctx, objectName, true);

    editable_mesh_cache = EditableMeshCache{};
    mesh_overlay_cache = MeshOverlayCache{};
    mesh_edit_layer = MeshEditLayer{};
    pending_serialized_mesh_edit_layer = PendingSerializedMeshEditLayer{};
    clearEditableMeshSelection();
    active_mesh_edit_object_name = objectName;
    ensureMeshEditLayer(ctx, objectName);
    ensureEditableMeshCache(ctx, objectName);
    addViewportMessage("Dissolved selected vertices", 2.0f, ImVec4(0.34f, 0.84f, 1.0f, 1.0f));
    return true;
}

bool SceneUI::deleteSelectedMeshFaces(UIContext& ctx) {
    const std::string objectName =
        (!active_mesh_edit_object_name.empty() ? active_mesh_edit_object_name :
            (ctx.selection.selected.type == SelectableType::Object && ctx.selection.selected.object
                ? ctx.selection.selected.object->getNodeName()
                : std::string{}));
    if (objectName.empty()) {
        return false;
    }

    if (!ensureEditableMeshCache(ctx, objectName)) {
        return false;
    }
    if (editable_mesh_cache.selection.face_ids.empty()) {
        return false;
    }

    std::vector<std::shared_ptr<Triangle>> currentDisplayMesh;
    {
        auto meshIt = mesh_cache.find(objectName);
        if (meshIt == mesh_cache.end() || meshIt->second.empty()) {
            return false;
        }

        std::unordered_set<const Triangle*> seenTriangles;
        seenTriangles.reserve(meshIt->second.size());
        currentDisplayMesh.reserve(meshIt->second.size());
        for (const auto& entry : meshIt->second) {
            if (entry.second && seenTriangles.insert(entry.second.get()).second) {
                currentDisplayMesh.push_back(entry.second);
            }
        }
    }
    if (currentDisplayMesh.empty()) {
        return false;
    }

    std::vector<std::shared_ptr<Triangle>> currentBaseMesh;
    {
        auto baseIt = ctx.scene.base_mesh_cache.find(objectName);
        if (baseIt != ctx.scene.base_mesh_cache.end() && !baseIt->second.empty()) {
            currentBaseMesh = cloneTriangleVectorForEdit(baseIt->second);
        } else {
            currentBaseMesh = cloneTriangleVectorForEdit(currentDisplayMesh);
        }
    }

    const auto modifierIt = ctx.scene.mesh_modifiers.find(objectName);
    const MeshModifiers::ModifierStack beforeModifierStack =
        (modifierIt != ctx.scene.mesh_modifiers.end()) ? modifierIt->second : MeshModifiers::ModifierStack{};
    const bool preserveModifierPreview = hasEnabledSubdivisionPreview(beforeModifierStack);

    std::vector<int> selectedFaceIds = editable_mesh_cache.selection.face_ids;
    std::sort(selectedFaceIds.begin(), selectedFaceIds.end());
    selectedFaceIds.erase(std::unique(selectedFaceIds.begin(), selectedFaceIds.end()), selectedFaceIds.end());

    std::unordered_set<int> triangleIdsToDelete;
    for (const int faceId : selectedFaceIds) {
        if (faceId >= 0 && faceId < static_cast<int>(editable_mesh_cache.polygon_faces.size())) {
            for (const int triangleId : editable_mesh_cache.polygon_faces[faceId].triangle_ids) {
                if (triangleId >= 0) {
                    triangleIdsToDelete.insert(triangleId);
                }
            }
        } else if (faceId >= 0 && faceId < static_cast<int>(editable_mesh_cache.faces.size())) {
            triangleIdsToDelete.insert(faceId);
        }
    }

    if (triangleIdsToDelete.empty()) {
        return false;
    }

    std::vector<std::shared_ptr<Triangle>> remainingMesh;
    const auto& sourceTopologyMesh = preserveModifierPreview ? currentBaseMesh : currentDisplayMesh;
    remainingMesh.reserve(sourceTopologyMesh.size());
    for (size_t triangleIndex = 0; triangleIndex < sourceTopologyMesh.size(); ++triangleIndex) {
        if (triangleIdsToDelete.count(static_cast<int>(triangleIndex)) == 0 && sourceTopologyMesh[triangleIndex]) {
            remainingMesh.push_back(cloneTriangleForEdit(sourceTopologyMesh[triangleIndex]));
        }
    }

    if (remainingMesh.size() == sourceTopologyMesh.size()) {
        return false;
    }

    for (size_t triangleIndex = 0; triangleIndex < remainingMesh.size(); ++triangleIndex) {
        if (remainingMesh[triangleIndex]) {
            remainingMesh[triangleIndex]->setFaceIndex(static_cast<int>(triangleIndex));
        }
    }

    const std::vector<std::shared_ptr<Triangle>> beforeDisplayMesh = cloneTriangleVectorForEdit(currentDisplayMesh);
    const std::vector<std::shared_ptr<Triangle>> afterDisplayMesh = evaluateDisplayMeshFromBase(remainingMesh, beforeModifierStack);
    const std::vector<std::shared_ptr<Triangle>> afterBaseMesh = cloneTriangleVectorForEdit(remainingMesh);
    const MeshModifiers::ModifierStack afterModifierStack = beforeModifierStack;

    auto command = std::make_unique<ReplaceMeshGeometryCommand>(
        objectName,
        beforeDisplayMesh,
        afterDisplayMesh,
        currentBaseMesh,
        afterBaseMesh,
        beforeModifierStack,
        afterModifierStack);
    command->execute(ctx);
    history.record(std::move(command));
    rebuildMeshCache(ctx.scene.world.objects);
    applyMeshShadingSettings(ctx, objectName, true);

    editable_mesh_cache = EditableMeshCache{};
    mesh_overlay_cache = MeshOverlayCache{};
    mesh_edit_layer = MeshEditLayer{};
    pending_serialized_mesh_edit_layer = PendingSerializedMeshEditLayer{};
    clearEditableMeshSelection();
    active_mesh_edit_object_name = objectName;

    auto refreshedMeshIt = mesh_cache.find(objectName);
    if (refreshedMeshIt != mesh_cache.end() && !refreshedMeshIt->second.empty() && refreshedMeshIt->second.front().second) {
        ctx.selection.selectObject(refreshedMeshIt->second.front().second, -1, objectName);
        active_mesh_edit_object_ptr = refreshedMeshIt->second.front().second.get();
    } else {
        active_mesh_edit_object_ptr = nullptr;
    }

    ensureMeshEditLayer(ctx, objectName);
    if (ensureEditableMeshCache(ctx, objectName) &&
        ctx.selection.mesh_element_mode != MeshElementSelectMode::Combined) {
        ctx.selection.mesh_element_mode = MeshElementSelectMode::Face;
    }
    addViewportMessage("Deleted selected faces", 2.0f, ImVec4(0.96f, 0.54f, 0.34f, 1.0f));
    return true;
}

bool SceneUI::flipSelectedMeshNormals(UIContext& ctx) {
    const std::string objectName =
        (!active_mesh_edit_object_name.empty() ? active_mesh_edit_object_name :
            (ctx.selection.selected.type == SelectableType::Object && ctx.selection.selected.object
                ? ctx.selection.selected.object->getNodeName()
                : std::string{}));
    if (objectName.empty()) {
        return false;
    }

    if (!ensureEditableMeshCache(ctx, objectName)) {
        return false;
    }

    std::vector<std::shared_ptr<Triangle>> currentDisplayMesh;
    {
        auto meshIt = mesh_cache.find(objectName);
        if (meshIt == mesh_cache.end() || meshIt->second.empty()) {
            return false;
        }

        std::unordered_set<const Triangle*> seenTriangles;
        seenTriangles.reserve(meshIt->second.size());
        currentDisplayMesh.reserve(meshIt->second.size());
        for (const auto& entry : meshIt->second) {
            if (entry.second && seenTriangles.insert(entry.second.get()).second) {
                currentDisplayMesh.push_back(entry.second);
            }
        }
    }
    if (currentDisplayMesh.empty()) {
        return false;
    }

    std::vector<std::shared_ptr<Triangle>> currentBaseMesh;
    {
        auto baseIt = ctx.scene.base_mesh_cache.find(objectName);
        if (baseIt != ctx.scene.base_mesh_cache.end() && !baseIt->second.empty()) {
            currentBaseMesh = cloneTriangleVectorForEdit(baseIt->second);
        } else {
            currentBaseMesh = cloneTriangleVectorForEdit(currentDisplayMesh);
        }
    }

    const auto modifierIt = ctx.scene.mesh_modifiers.find(objectName);
    const MeshModifiers::ModifierStack beforeModifierStack =
        (modifierIt != ctx.scene.mesh_modifiers.end()) ? modifierIt->second : MeshModifiers::ModifierStack{};

    // Identify which triangles to flip
    std::unordered_set<int> baseTriangleIndicesToFlip;

    const bool hasFaceSel = !editable_mesh_cache.selection.face_ids.empty();
    const bool hasVertexSel = !editable_mesh_cache.selection.vertex_ids.empty();
    const bool hasEdgeSel = !editable_mesh_cache.selection.edge_ids.empty();

    if (mesh_overlay_settings.edit_mode && (hasFaceSel || hasVertexSel || hasEdgeSel)) {
        if (hasFaceSel) {
            for (const int faceId : editable_mesh_cache.selection.face_ids) {
                if (faceId >= 0 && faceId < static_cast<int>(editable_mesh_cache.polygon_faces.size())) {
                    for (const int triangleId : editable_mesh_cache.polygon_faces[faceId].triangle_ids) {
                        if (triangleId >= 0) baseTriangleIndicesToFlip.insert(triangleId);
                    }
                } else if (faceId >= 0 && faceId < static_cast<int>(editable_mesh_cache.faces.size())) {
                    baseTriangleIndicesToFlip.insert(faceId);
                }
            }
        } else if (hasVertexSel) {
            std::unordered_set<int> selectedVerts(editable_mesh_cache.selection.vertex_ids.begin(), editable_mesh_cache.selection.vertex_ids.end());
            for (int faceId = 0; faceId < static_cast<int>(editable_mesh_cache.faces.size()); ++faceId) {
                const auto& face = editable_mesh_cache.faces[faceId];
                if (selectedVerts.count(face.v0) || selectedVerts.count(face.v1) || selectedVerts.count(face.v2)) {
                    baseTriangleIndicesToFlip.insert(faceId);
                }
            }
        } else if (hasEdgeSel) {
            std::unordered_set<int> selectedEdges(editable_mesh_cache.selection.edge_ids.begin(), editable_mesh_cache.selection.edge_ids.end());
            for (int edgeId : selectedEdges) {
                const auto* edge = getEditableSelectableEdge(editable_mesh_cache, edgeId);
                if (!edge) continue;
                for (int faceId = 0; faceId < static_cast<int>(editable_mesh_cache.faces.size()); ++faceId) {
                    const auto& face = editable_mesh_cache.faces[faceId];
                    bool sharesV0 = (face.v0 == edge->v0 || face.v1 == edge->v0 || face.v2 == edge->v0);
                    bool sharesV1 = (face.v0 == edge->v1 || face.v1 == edge->v1 || face.v2 == edge->v1);
                    if (sharesV0 && sharesV1) {
                        baseTriangleIndicesToFlip.insert(faceId);
                    }
                }
            }
        }
    } else {
        // Object mode or nothing selected: flip the whole mesh
        for (size_t i = 0; i < currentBaseMesh.size(); ++i) {
            baseTriangleIndicesToFlip.insert(static_cast<int>(i));
        }
    }

    if (baseTriangleIndicesToFlip.empty()) {
        return false;
    }

    std::vector<std::shared_ptr<Triangle>> modifiedBaseMesh = cloneTriangleVectorForEdit(currentBaseMesh);
    
    auto flipTri = [](const std::shared_ptr<Triangle>& tri) {
        if (!tri) return;
        std::swap(tri->vertices[0].position, tri->vertices[2].position);
        std::swap(tri->vertices[0].original, tri->vertices[2].original);
        std::swap(tri->vertices[0].normal, tri->vertices[2].normal);
        std::swap(tri->vertices[0].originalNormal, tri->vertices[2].originalNormal);
        
        std::swap(tri->t0, tri->t2);
        for (auto& uv_set : tri->uv_sets) {
            std::swap(uv_set[0], uv_set[2]);
        }
        
        for (int i = 0; i < 3; ++i) {
            tri->vertices[i].normal = -tri->vertices[i].normal;
            tri->vertices[i].originalNormal = -tri->vertices[i].originalNormal;
        }
        tri->markAABBDirty();
    };

    for (int idx : baseTriangleIndicesToFlip) {
        if (idx >= 0 && idx < static_cast<int>(modifiedBaseMesh.size())) {
            flipTri(modifiedBaseMesh[idx]);
        }
    }

    const std::vector<std::shared_ptr<Triangle>> beforeDisplayMesh = cloneTriangleVectorForEdit(currentDisplayMesh);
    const std::vector<std::shared_ptr<Triangle>> afterDisplayMesh = evaluateDisplayMeshFromBase(modifiedBaseMesh, beforeModifierStack);
    const std::vector<std::shared_ptr<Triangle>> afterBaseMesh = cloneTriangleVectorForEdit(modifiedBaseMesh);
    const MeshModifiers::ModifierStack afterModifierStack = beforeModifierStack;

    auto command = std::make_unique<ReplaceMeshGeometryCommand>(
        objectName,
        beforeDisplayMesh,
        afterDisplayMesh,
        currentBaseMesh,
        afterBaseMesh,
        beforeModifierStack,
        afterModifierStack);
    command->execute(ctx);
    history.record(std::move(command));
    rebuildMeshCache(ctx.scene.world.objects);
    applyMeshShadingSettings(ctx, objectName, true);

    editable_mesh_cache = EditableMeshCache{};
    mesh_overlay_cache = MeshOverlayCache{};
    mesh_edit_layer = MeshEditLayer{};
    pending_serialized_mesh_edit_layer = PendingSerializedMeshEditLayer{};
    clearEditableMeshSelection();
    active_mesh_edit_object_name = objectName;

    ProjectManager::getInstance().markModified();
    addViewportMessage("Flipped Normals", 2.0f, ImVec4(0.38f, 0.82f, 1.0f, 1.0f));
    return true;
}

bool SceneUI::recalculateMeshNormals(UIContext& ctx, bool outside) {
    const std::string objectName =
        (!active_mesh_edit_object_name.empty() ? active_mesh_edit_object_name :
            (ctx.selection.selected.type == SelectableType::Object && ctx.selection.selected.object
                ? ctx.selection.selected.object->getNodeName()
                : std::string{}));
    if (objectName.empty()) {
        return false;
    }

    if (!ensureEditableMeshCache(ctx, objectName)) {
        return false;
    }

    std::vector<std::shared_ptr<Triangle>> currentDisplayMesh;
    {
        auto meshIt = mesh_cache.find(objectName);
        if (meshIt == mesh_cache.end() || meshIt->second.empty()) {
            return false;
        }

        std::unordered_set<const Triangle*> seenTriangles;
        seenTriangles.reserve(meshIt->second.size());
        currentDisplayMesh.reserve(meshIt->second.size());
        for (const auto& entry : meshIt->second) {
            if (entry.second && seenTriangles.insert(entry.second.get()).second) {
                currentDisplayMesh.push_back(entry.second);
            }
        }
    }
    if (currentDisplayMesh.empty()) {
        return false;
    }

    std::vector<std::shared_ptr<Triangle>> currentBaseMesh;
    {
        auto baseIt = ctx.scene.base_mesh_cache.find(objectName);
        if (baseIt != ctx.scene.base_mesh_cache.end() && !baseIt->second.empty()) {
            currentBaseMesh = cloneTriangleVectorForEdit(baseIt->second);
        } else {
            currentBaseMesh = cloneTriangleVectorForEdit(currentDisplayMesh);
        }
    }

    Vec3 meshCenter(0.0f, 0.0f, 0.0f);
    float vertexCount = 0.0f;
    for (const auto& tri : currentBaseMesh) {
        if (!tri) continue;
        for (int i = 0; i < 3; ++i) {
            meshCenter += tri->getOriginalVertexPosition(i);
            vertexCount += 1.0f;
        }
    }
    if (vertexCount > 0.0f) {
        meshCenter = meshCenter / vertexCount;
    }

    const auto modifierIt = ctx.scene.mesh_modifiers.find(objectName);
    const MeshModifiers::ModifierStack beforeModifierStack =
        (modifierIt != ctx.scene.mesh_modifiers.end()) ? modifierIt->second : MeshModifiers::ModifierStack{};

    std::unordered_set<int> baseTriangleIndicesToRecalculate;
    const bool hasFaceSel = !editable_mesh_cache.selection.face_ids.empty();
    const bool hasVertexSel = !editable_mesh_cache.selection.vertex_ids.empty();
    const bool hasEdgeSel = !editable_mesh_cache.selection.edge_ids.empty();

    if (mesh_overlay_settings.edit_mode && (hasFaceSel || hasVertexSel || hasEdgeSel)) {
        if (hasFaceSel) {
            for (const int faceId : editable_mesh_cache.selection.face_ids) {
                if (faceId >= 0 && faceId < static_cast<int>(editable_mesh_cache.polygon_faces.size())) {
                    for (const int triangleId : editable_mesh_cache.polygon_faces[faceId].triangle_ids) {
                        if (triangleId >= 0) baseTriangleIndicesToRecalculate.insert(triangleId);
                    }
                } else if (faceId >= 0 && faceId < static_cast<int>(editable_mesh_cache.faces.size())) {
                    baseTriangleIndicesToRecalculate.insert(faceId);
                }
            }
        } else if (hasVertexSel) {
            std::unordered_set<int> selectedVerts(editable_mesh_cache.selection.vertex_ids.begin(), editable_mesh_cache.selection.vertex_ids.end());
            for (int faceId = 0; faceId < static_cast<int>(editable_mesh_cache.faces.size()); ++faceId) {
                const auto& face = editable_mesh_cache.faces[faceId];
                if (selectedVerts.count(face.v0) || selectedVerts.count(face.v1) || selectedVerts.count(face.v2)) {
                    baseTriangleIndicesToRecalculate.insert(faceId);
                }
            }
        } else if (hasEdgeSel) {
            std::unordered_set<int> selectedEdges(editable_mesh_cache.selection.edge_ids.begin(), editable_mesh_cache.selection.edge_ids.end());
            for (int edgeId : selectedEdges) {
                const auto* edge = getEditableSelectableEdge(editable_mesh_cache, edgeId);
                if (!edge) continue;
                for (int faceId = 0; faceId < static_cast<int>(editable_mesh_cache.faces.size()); ++faceId) {
                    const auto& face = editable_mesh_cache.faces[faceId];
                    bool sharesV0 = (face.v0 == edge->v0 || face.v1 == edge->v0 || face.v2 == edge->v0);
                    bool sharesV1 = (face.v0 == edge->v1 || face.v1 == edge->v1 || face.v2 == edge->v1);
                    if (sharesV0 && sharesV1) {
                        baseTriangleIndicesToRecalculate.insert(faceId);
                    }
                }
            }
        }
    } else {
        for (size_t i = 0; i < currentBaseMesh.size(); ++i) {
            baseTriangleIndicesToRecalculate.insert(static_cast<int>(i));
        }
    }

    if (baseTriangleIndicesToRecalculate.empty()) {
        return false;
    }

    std::vector<std::shared_ptr<Triangle>> modifiedBaseMesh = cloneTriangleVectorForEdit(currentBaseMesh);

    auto flipTri = [](const std::shared_ptr<Triangle>& tri) {
        if (!tri) return;
        std::swap(tri->vertices[0].position, tri->vertices[2].position);
        std::swap(tri->vertices[0].original, tri->vertices[2].original);
        std::swap(tri->vertices[0].normal, tri->vertices[2].normal);
        std::swap(tri->vertices[0].originalNormal, tri->vertices[2].originalNormal);
        
        std::swap(tri->t0, tri->t2);
        for (auto& uv_set : tri->uv_sets) {
            std::swap(uv_set[0], uv_set[2]);
        }
        
        for (int i = 0; i < 3; ++i) {
            tri->vertices[i].normal = -tri->vertices[i].normal;
            tri->vertices[i].originalNormal = -tri->vertices[i].originalNormal;
        }
        tri->markAABBDirty();
    };

    bool anyModified = false;
    for (int idx : baseTriangleIndicesToRecalculate) {
        if (idx >= 0 && idx < static_cast<int>(modifiedBaseMesh.size())) {
            const auto& tri = modifiedBaseMesh[idx];
            if (!tri) continue;

            Vec3 faceCenter = (tri->getOriginalVertexPosition(0) +
                               tri->getOriginalVertexPosition(1) +
                               tri->getOriginalVertexPosition(2)) / 3.0f;

            Vec3 faceNormal = Vec3::cross(
                tri->getOriginalVertexPosition(1) - tri->getOriginalVertexPosition(0),
                tri->getOriginalVertexPosition(2) - tri->getOriginalVertexPosition(0));
            float faceNormalLen = faceNormal.length();
            if (faceNormalLen > 1e-6f) {
                faceNormal = faceNormal / faceNormalLen;
            }

            Vec3 dir = faceCenter - meshCenter;
            float dotVal = faceNormal.dot(dir);
            bool shouldFlip = outside ? (dotVal < 0.0f) : (dotVal > 0.0f);

            if (shouldFlip) {
                flipTri(tri);
                anyModified = true;
            }
        }
    }

    if (!anyModified) {
        return false;
    }

    const std::vector<std::shared_ptr<Triangle>> beforeDisplayMesh = cloneTriangleVectorForEdit(currentDisplayMesh);
    const std::vector<std::shared_ptr<Triangle>> afterDisplayMesh = evaluateDisplayMeshFromBase(modifiedBaseMesh, beforeModifierStack);
    const std::vector<std::shared_ptr<Triangle>> afterBaseMesh = cloneTriangleVectorForEdit(modifiedBaseMesh);
    const MeshModifiers::ModifierStack afterModifierStack = beforeModifierStack;

    auto command = std::make_unique<ReplaceMeshGeometryCommand>(
        objectName,
        beforeDisplayMesh,
        afterDisplayMesh,
        currentBaseMesh,
        afterBaseMesh,
        beforeModifierStack,
        afterModifierStack);
    command->execute(ctx);
    history.record(std::move(command));
    rebuildMeshCache(ctx.scene.world.objects);
    applyMeshShadingSettings(ctx, objectName, true);

    editable_mesh_cache = EditableMeshCache{};
    mesh_overlay_cache = MeshOverlayCache{};
    mesh_edit_layer = MeshEditLayer{};
    pending_serialized_mesh_edit_layer = PendingSerializedMeshEditLayer{};
    clearEditableMeshSelection();
    active_mesh_edit_object_name = objectName;

    ProjectManager::getInstance().markModified();
    addViewportMessage(outside ? "Recalculated Outside" : "Recalculated Inside", 2.0f, ImVec4(0.38f, 0.82f, 1.0f, 1.0f));
    return true;
}

bool SceneUI::applyShadingToSelectedFaces(UIContext& ctx, bool flat, bool autoSmooth) {
    const std::string objectName = active_mesh_edit_object_name;
    if (objectName.empty()) return false;
    if (!ensureEditableMeshCache(ctx, objectName)) return false;

    // Collect indices of selected triangles
    std::unordered_set<int> selectedTriangleIds;
    const bool hasFaceSel = !editable_mesh_cache.selection.face_ids.empty();

    if (hasFaceSel) {
        for (const int faceId : editable_mesh_cache.selection.face_ids) {
            if (faceId >= 0 && faceId < static_cast<int>(editable_mesh_cache.polygon_faces.size())) {
                for (const int triangleId : editable_mesh_cache.polygon_faces[faceId].triangle_ids) {
                    if (triangleId >= 0) selectedTriangleIds.insert(triangleId);
                }
            } else if (faceId >= 0 && faceId < static_cast<int>(editable_mesh_cache.faces.size())) {
                selectedTriangleIds.insert(faceId);
            }
        }
    } else {
        // Shading in Edit Mode only applies to selected faces, not vertices or edges
        return false;
    }

    if (selectedTriangleIds.empty()) {
        return false;
    }

    // Clone current base mesh and display mesh
    std::vector<std::shared_ptr<Triangle>> currentDisplayMesh;
    {
        auto meshIt = mesh_cache.find(objectName);
        if (meshIt == mesh_cache.end() || meshIt->second.empty()) return false;
        std::unordered_set<const Triangle*> seenTriangles;
        for (const auto& entry : meshIt->second) {
            if (entry.second && seenTriangles.insert(entry.second.get()).second) {
                currentDisplayMesh.push_back(entry.second);
            }
        }
    }

    std::vector<std::shared_ptr<Triangle>> currentBaseMesh;
    {
        auto baseIt = ctx.scene.base_mesh_cache.find(objectName);
        if (baseIt != ctx.scene.base_mesh_cache.end() && !baseIt->second.empty()) {
            currentBaseMesh = cloneTriangleVectorForEdit(baseIt->second);
        } else {
            currentBaseMesh = cloneTriangleVectorForEdit(currentDisplayMesh);
        }
    }

    const auto modifierIt = ctx.scene.mesh_modifiers.find(objectName);
    const MeshModifiers::ModifierStack beforeModifierStack =
        (modifierIt != ctx.scene.mesh_modifiers.end()) ? modifierIt->second : MeshModifiers::ModifierStack{};

    // Prepare modified base mesh
    std::vector<std::shared_ptr<Triangle>> modifiedBaseMesh = cloneTriangleVectorForEdit(currentBaseMesh);

    // Apply shading to base mesh selected triangles
    std::vector<std::shared_ptr<Triangle>> targetBaseTris;
    for (int idx : selectedTriangleIds) {
        if (idx >= 0 && idx < static_cast<int>(modifiedBaseMesh.size())) {
            targetBaseTris.push_back(modifiedBaseMesh[idx]);
        }
    }
    float angle = ensureMeshShadingSettings(objectName).auto_smooth_angle_degrees;
    applyShadingSettingsToTriangles(targetBaseTris, flat, autoSmooth, angle);

    const std::vector<std::shared_ptr<Triangle>> beforeDisplayMesh = cloneTriangleVectorForEdit(currentDisplayMesh);
    const std::vector<std::shared_ptr<Triangle>> afterDisplayMesh = evaluateDisplayMeshFromBase(modifiedBaseMesh, beforeModifierStack);
    const std::vector<std::shared_ptr<Triangle>> afterBaseMesh = cloneTriangleVectorForEdit(modifiedBaseMesh);
    const MeshModifiers::ModifierStack afterModifierStack = beforeModifierStack;

    auto command = std::make_unique<ReplaceMeshGeometryCommand>(
        objectName,
        beforeDisplayMesh,
        afterDisplayMesh,
        currentBaseMesh,
        afterBaseMesh,
        beforeModifierStack,
        afterModifierStack);
    command->execute(ctx);
    history.record(std::move(command));

    rebuildMeshCache(ctx.scene.world.objects);
    
    // Clear cache to force rebuild on draw
    editable_mesh_cache = EditableMeshCache{};
    mesh_overlay_cache = MeshOverlayCache{};
    mesh_edit_layer = MeshEditLayer{};
    pending_serialized_mesh_edit_layer = PendingSerializedMeshEditLayer{};
    clearEditableMeshSelection();
    active_mesh_edit_object_name = objectName;

    ProjectManager::getInstance().markModified();
    addViewportMessage(flat ? "Faceted Shading Applied" : "Smooth Shading Applied", 2.0f, ImVec4(0.38f, 0.82f, 1.0f, 1.0f));
    return true;
}

bool SceneUI::addFaceFromSelectedVertices(UIContext& ctx) {
    const std::string objectName =
        (!active_mesh_edit_object_name.empty() ? active_mesh_edit_object_name :
            (ctx.selection.selected.type == SelectableType::Object && ctx.selection.selected.object
                ? ctx.selection.selected.object->getNodeName()
                : std::string{}));
    if (objectName.empty()) {
        return false;
    }

    if (!ensureEditableMeshCache(ctx, objectName)) {
        return false;
    }

    // Gather face corners from the vertex selection, and — when too few
    // vertices are selected — fall back to the endpoints of the edge
    // selection so a face can be built directly from picked edges too.
    std::vector<int> selectedVertexIds = editable_mesh_cache.selection.vertex_ids;
    if (selectedVertexIds.size() < 3 && !editable_mesh_cache.selection.edge_ids.empty()) {
        for (const int edgeId : editable_mesh_cache.selection.edge_ids) {
            const auto* edge = getEditableSelectableEdge(editable_mesh_cache, edgeId);
            if (!edge) {
                continue;
            }
            selectedVertexIds.push_back(edge->v0);
            selectedVertexIds.push_back(edge->v1);
        }
    }
    std::sort(selectedVertexIds.begin(), selectedVertexIds.end());
    selectedVertexIds.erase(std::unique(selectedVertexIds.begin(), selectedVertexIds.end()), selectedVertexIds.end());
    // 3 = triangle, 4 = quad, 5+ = n-gon fan-triangulated. Below 3 there is no face.
    if (selectedVertexIds.size() < 3) {
        return false;
    }

    std::vector<int> orderedVertexIds = sortEditablePolygonVertices(editable_mesh_cache, selectedVertexIds);
    if (orderedVertexIds.size() != selectedVertexIds.size()) {
        return false;
    }

    std::vector<int> canonicalSelection = orderedVertexIds;
    std::sort(canonicalSelection.begin(), canonicalSelection.end());
    for (size_t polygonFaceId = 0; polygonFaceId < editable_mesh_cache.polygon_faces.size(); ++polygonFaceId) {
        std::vector<int> existingFace = editable_mesh_cache.polygon_faces[polygonFaceId].vertex_ids;
        if (existingFace.size() != canonicalSelection.size()) {
            continue;
        }
        std::sort(existingFace.begin(), existingFace.end());
        if (existingFace == canonicalSelection) {
            editable_mesh_cache.selection.face_ids = { static_cast<int>(polygonFaceId) };
            editable_mesh_cache.selection.active_face_id = static_cast<int>(polygonFaceId);
            editable_mesh_cache.selection.vertex_ids.clear();
            editable_mesh_cache.selection.active_vertex_id = -1;
            // Keep Combined mode; only single-type modes snap to Face focus.
            if (ctx.selection.mesh_element_mode != MeshElementSelectMode::Combined) {
                ctx.selection.mesh_element_mode = MeshElementSelectMode::Face;
            }
            return false;
        }
    }

    std::vector<std::shared_ptr<Triangle>> currentDisplayMesh;
    {
        auto meshIt = mesh_cache.find(objectName);
        if (meshIt == mesh_cache.end() || meshIt->second.empty()) {
            return false;
        }

        std::unordered_set<const Triangle*> seenTriangles;
        seenTriangles.reserve(meshIt->second.size());
        currentDisplayMesh.reserve(meshIt->second.size());
        for (const auto& entry : meshIt->second) {
            if (entry.second && seenTriangles.insert(entry.second.get()).second) {
                currentDisplayMesh.push_back(entry.second);
            }
        }
    }
    if (currentDisplayMesh.empty()) {
        return false;
    }

    std::vector<std::shared_ptr<Triangle>> currentBaseMesh;
    {
        auto baseIt = ctx.scene.base_mesh_cache.find(objectName);
        if (baseIt != ctx.scene.base_mesh_cache.end() && !baseIt->second.empty()) {
            currentBaseMesh = cloneTriangleVectorForEdit(baseIt->second);
        } else {
            currentBaseMesh = cloneTriangleVectorForEdit(currentDisplayMesh);
        }
    }

    const auto modifierIt = ctx.scene.mesh_modifiers.find(objectName);
    const MeshModifiers::ModifierStack beforeModifierStack =
        (modifierIt != ctx.scene.mesh_modifiers.end()) ? modifierIt->second : MeshModifiers::ModifierStack{};
    const bool preserveModifierPreview = hasEnabledSubdivisionPreview(beforeModifierStack);

    // Winding fix: a freshly built face has no inherent front side, so the
    // angle-sort above can wind it either way. Derive a reference orientation
    // from the faces already touching these vertices and flip the order if the
    // candidate normal opposes it; this keeps the new face consistent with the
    // surrounding surface instead of pointing inward. When the points are
    // isolated (no neighbours) fall back to facing the camera.
    {
        Vec3 referenceNormal(0.0f, 0.0f, 0.0f);
        std::unordered_set<const Triangle*> seenAdjacent;
        for (const int vertexId : orderedVertexIds) {
            if (!isEditableVertexIdValid(editable_mesh_cache, vertexId)) {
                continue;
            }
            for (const auto& ref : editable_mesh_cache.vertices[vertexId].refs) {
                const Triangle* tri = editable_mesh_cache.refTri(ref);
                if (!tri || !seenAdjacent.insert(tri).second) {
                    continue;
                }
                const Vec3 a = tri->getOriginalVertexPosition(0);
                const Vec3 b = tri->getOriginalVertexPosition(1);
                const Vec3 c = tri->getOriginalVertexPosition(2);
                Vec3 n = (b - a).cross(c - a);
                const float lenSq = n.length_squared();
                if (std::isfinite(lenSq) && lenSq > 1e-12f) {
                    referenceNormal += n / std::sqrt(lenSq);
                }
            }
        }

        if (referenceNormal.length_squared() <= 1e-10f && ctx.scene.camera) {
            // No adjacent faces — orient toward the camera (in local space).
            const Matrix4x4 faceTransform = getEditableObjectTransform(editable_mesh_cache);
            const Vec3 cameraLocal = faceTransform.inverse().transform_point(ctx.scene.camera->lookfrom);
            Vec3 centroid(0.0f, 0.0f, 0.0f);
            for (const int vertexId : orderedVertexIds) {
                centroid += editable_mesh_cache.vertices[vertexId].local_position;
            }
            centroid /= static_cast<float>(orderedVertexIds.size());
            referenceNormal = cameraLocal - centroid;
        }

        if (referenceNormal.length_squared() > 1e-10f) {
            const Vec3 candidateNormal = computeEditableFaceNormal(editable_mesh_cache, orderedVertexIds);
            if (candidateNormal.dot(referenceNormal) < 0.0f) {
                std::reverse(orderedVertexIds.begin(), orderedVertexIds.end());
            }
        }
    }

    std::vector<Vec3> faceVertices;
    faceVertices.reserve(orderedVertexIds.size());
    for (const int vertexId : orderedVertexIds) {
        if (!isEditableVertexIdValid(editable_mesh_cache, vertexId)) {
            return false;
        }
        faceVertices.push_back(editable_mesh_cache.vertices[vertexId].local_position);
    }

    const Vec3 faceNormal = computeEditableFaceNormal(editable_mesh_cache, orderedVertexIds);
    std::shared_ptr<Triangle> templateTriangle = preserveModifierPreview ? currentBaseMesh.front() : currentDisplayMesh.front();
    for (const int vertexId : orderedVertexIds) {
        for (const auto& ref : editable_mesh_cache.vertices[vertexId].refs) {
            if (editable_mesh_cache.refTri(ref)) {
                templateTriangle = editable_mesh_cache.refTriShared(ref);
                break;
            }
        }
        if (templateTriangle) {
            break;
        }
    }
    if (!templateTriangle) {
        return false;
    }

    std::vector<std::shared_ptr<Triangle>> updatedMesh = preserveModifierPreview
        ? cloneTriangleVectorForEdit(currentBaseMesh)
        : cloneTriangleVectorForEdit(currentDisplayMesh);
    const std::vector<Vec2> faceUVs = buildPolygonPlanarUVs(faceVertices, faceNormal);
    const size_t firstNewTriangleIndex = updatedMesh.size();

    auto addFaceTriangle = [&](int i0, int i1, int i2) {
        auto newTriangle = cloneTriangleForEdit(templateTriangle);
        if (!newTriangle) {
            return;
        }
        newTriangle->setOriginalVertexPosition(0, faceVertices[static_cast<size_t>(i0)]);
        newTriangle->setOriginalVertexPosition(1, faceVertices[static_cast<size_t>(i1)]);
        newTriangle->setOriginalVertexPosition(2, faceVertices[static_cast<size_t>(i2)]);
        newTriangle->setOriginalVertexNormal(0, faceNormal);
        newTriangle->setOriginalVertexNormal(1, faceNormal);
        newTriangle->setOriginalVertexNormal(2, faceNormal);
        newTriangle->set_normals(faceNormal, faceNormal, faceNormal);
        newTriangle->setUVCoordinates(
            faceUVs[static_cast<size_t>(i0)],
            faceUVs[static_cast<size_t>(i1)],
            faceUVs[static_cast<size_t>(i2)]);
        newTriangle->markAABBDirty();
        newTriangle->updateTransformedVertices();
        updatedMesh.push_back(newTriangle);
    };

    // Fan-triangulate the ordered polygon: triangle (1 tri), quad (2 tris),
    // n-gon (n-2 tris). The shared vertex 0 anchors the fan.
    for (size_t i = 1; i + 1 < faceVertices.size(); ++i) {
        addFaceTriangle(0, static_cast<int>(i), static_cast<int>(i + 1));
    }

    if (updatedMesh.size() == firstNewTriangleIndex) {
        return false;
    }

    for (size_t triangleIndex = 0; triangleIndex < updatedMesh.size(); ++triangleIndex) {
        if (updatedMesh[triangleIndex]) {
            updatedMesh[triangleIndex]->setFaceIndex(static_cast<int>(triangleIndex));
        }
    }

    const std::vector<std::shared_ptr<Triangle>> beforeDisplayMesh = cloneTriangleVectorForEdit(currentDisplayMesh);
    const std::vector<std::shared_ptr<Triangle>> afterDisplayMesh = evaluateDisplayMeshFromBase(updatedMesh, beforeModifierStack);
    const std::vector<std::shared_ptr<Triangle>> afterBaseMesh = cloneTriangleVectorForEdit(updatedMesh);
    const MeshModifiers::ModifierStack afterModifierStack = beforeModifierStack;

    auto command = std::make_unique<ReplaceMeshGeometryCommand>(
        objectName,
        beforeDisplayMesh,
        afterDisplayMesh,
        currentBaseMesh,
        afterBaseMesh,
        beforeModifierStack,
        afterModifierStack);
    command->execute(ctx);
    history.record(std::move(command));
    rebuildMeshCache(ctx.scene.world.objects);
    applyMeshShadingSettings(ctx, objectName, true);

    editable_mesh_cache = EditableMeshCache{};
    mesh_overlay_cache = MeshOverlayCache{};
    mesh_edit_layer = MeshEditLayer{};
    pending_serialized_mesh_edit_layer = PendingSerializedMeshEditLayer{};
    clearEditableMeshSelection();
    active_mesh_edit_object_name = objectName;

    auto refreshedMeshIt = mesh_cache.find(objectName);
    if (refreshedMeshIt != mesh_cache.end() && !refreshedMeshIt->second.empty() && refreshedMeshIt->second.front().second) {
        ctx.selection.selectObject(refreshedMeshIt->second.front().second, -1, objectName);
        active_mesh_edit_object_ptr = refreshedMeshIt->second.front().second.get();
    } else {
        active_mesh_edit_object_ptr = nullptr;
    }

    ensureMeshEditLayer(ctx, objectName);
    if (ensureEditableMeshCache(ctx, objectName)) {
        int bestFaceId = -1;
        float bestScore = std::numeric_limits<float>::max();
        Vec3 targetCenter(0.0f, 0.0f, 0.0f);
        for (const Vec3& vertex : faceVertices) {
            targetCenter += vertex;
        }
        targetCenter /= static_cast<float>(faceVertices.size());

        for (size_t polygonFaceId = 0; polygonFaceId < editable_mesh_cache.polygon_faces.size(); ++polygonFaceId) {
            const std::vector<int> vertexIds = getEditablePolygonVertexIds(editable_mesh_cache, static_cast<int>(polygonFaceId));
            if (vertexIds.size() != orderedVertexIds.size()) {
                continue;
            }

            std::vector<int> sortedFace = vertexIds;
            std::sort(sortedFace.begin(), sortedFace.end());
            if (sortedFace != canonicalSelection) {
                continue;
            }

            Vec3 faceCenter(0.0f, 0.0f, 0.0f);
            for (const int vertexId : vertexIds) {
                faceCenter += editable_mesh_cache.vertices[vertexId].local_position;
            }
            faceCenter /= static_cast<float>(vertexIds.size());

            const Vec3 currentFaceNormal = computeEditableFaceNormal(editable_mesh_cache, vertexIds);
            const float score = (faceCenter - targetCenter).length_squared() + (1.0f - currentFaceNormal.dot(faceNormal)) * 0.05f;
            if (score < bestScore) {
                bestScore = score;
                bestFaceId = static_cast<int>(polygonFaceId);
            }
        }

        if (bestFaceId >= 0) {
            editable_mesh_cache.selection.face_ids = { bestFaceId };
            editable_mesh_cache.selection.active_face_id = bestFaceId;
            editable_mesh_cache.selection.vertex_ids.clear();
            editable_mesh_cache.selection.active_vertex_id = -1;
            // Keep Combined mode; only single-type modes snap to Face focus.
            if (ctx.selection.mesh_element_mode != MeshElementSelectMode::Combined) {
                ctx.selection.mesh_element_mode = MeshElementSelectMode::Face;
            }
        }
    }

    addViewportMessage("Added face from selected vertices", 2.0f, ImVec4(0.36f, 0.92f, 0.68f, 1.0f));
    return true;
}

bool SceneUI::mergeSelectedVerticesToCenter(UIContext& ctx) {
    const std::string objectName =
        (!active_mesh_edit_object_name.empty() ? active_mesh_edit_object_name :
            (ctx.selection.selected.type == SelectableType::Object && ctx.selection.selected.object
                ? ctx.selection.selected.object->getNodeName()
                : std::string{}));
    if (objectName.empty()) {
        return false;
    }

    if (!ensureEditableMeshCache(ctx, objectName)) {
        return false;
    }

    std::vector<int> selectedVertexIds = editable_mesh_cache.selection.vertex_ids;
    std::sort(selectedVertexIds.begin(), selectedVertexIds.end());
    selectedVertexIds.erase(std::unique(selectedVertexIds.begin(), selectedVertexIds.end()), selectedVertexIds.end());
    if (selectedVertexIds.size() < 2) {
        return false;
    }

    Vec3 mergeCenter(0.0f, 0.0f, 0.0f);
    for (const int vertexId : selectedVertexIds) {
        if (!isEditableVertexIdValid(editable_mesh_cache, vertexId)) {
            return false;
        }
        mergeCenter += editable_mesh_cache.vertices[vertexId].local_position;
    }
    mergeCenter /= static_cast<float>(selectedVertexIds.size());

    std::unordered_set<QuantizedVertexKey, QuantizedVertexKeyHasher> selectedKeys;
    selectedKeys.reserve(selectedVertexIds.size());
    for (const int vertexId : selectedVertexIds) {
        selectedKeys.insert(quantizeTopologyVertex(editable_mesh_cache.vertices[vertexId].local_position));
    }

    std::vector<std::shared_ptr<Triangle>> currentDisplayMesh;
    {
        auto meshIt = mesh_cache.find(objectName);
        if (meshIt == mesh_cache.end() || meshIt->second.empty()) {
            return false;
        }

        std::unordered_set<const Triangle*> seenTriangles;
        seenTriangles.reserve(meshIt->second.size());
        currentDisplayMesh.reserve(meshIt->second.size());
        for (const auto& entry : meshIt->second) {
            if (entry.second && seenTriangles.insert(entry.second.get()).second) {
                currentDisplayMesh.push_back(entry.second);
            }
        }
    }
    if (currentDisplayMesh.empty()) {
        return false;
    }

    std::vector<std::shared_ptr<Triangle>> currentBaseMesh;
    {
        auto baseIt = ctx.scene.base_mesh_cache.find(objectName);
        if (baseIt != ctx.scene.base_mesh_cache.end() && !baseIt->second.empty()) {
            currentBaseMesh = cloneTriangleVectorForEdit(baseIt->second);
        } else {
            currentBaseMesh = cloneTriangleVectorForEdit(currentDisplayMesh);
        }
    }

    const auto modifierIt = ctx.scene.mesh_modifiers.find(objectName);
    const MeshModifiers::ModifierStack beforeModifierStack =
        (modifierIt != ctx.scene.mesh_modifiers.end()) ? modifierIt->second : MeshModifiers::ModifierStack{};
    const bool preserveModifierPreview = hasEnabledSubdivisionPreview(beforeModifierStack);

    std::vector<std::shared_ptr<Triangle>> mergedMesh;
    const auto& sourceTopologyMesh = preserveModifierPreview ? currentBaseMesh : currentDisplayMesh;
    mergedMesh.reserve(sourceTopologyMesh.size());
    bool changed = false;
    for (const auto& sourceTriangle : sourceTopologyMesh) {
        auto tri = cloneTriangleForEdit(sourceTriangle);
        if (!tri) {
            continue;
        }

        for (int corner = 0; corner < 3; ++corner) {
            if (selectedKeys.count(quantizeTopologyVertex(tri->getOriginalVertexPosition(corner))) > 0) {
                tri->setOriginalVertexPosition(corner, mergeCenter);
                changed = true;
            }
        }

        const Vec3 p0 = tri->getOriginalVertexPosition(0);
        const Vec3 p1 = tri->getOriginalVertexPosition(1);
        const Vec3 p2 = tri->getOriginalVertexPosition(2);
        const float areaSq = Vec3::cross(p1 - p0, p2 - p0).length_squared();
        if (areaSq <= 1e-12f) {
            continue;
        }

        tri->markAABBDirty();
        mergedMesh.push_back(tri);
    }

    if (!changed) {
        return false;
    }

    for (size_t triangleIndex = 0; triangleIndex < mergedMesh.size(); ++triangleIndex) {
        if (mergedMesh[triangleIndex]) {
            mergedMesh[triangleIndex]->setFaceIndex(static_cast<int>(triangleIndex));
        }
    }

    const std::vector<std::shared_ptr<Triangle>> beforeDisplayMesh = cloneTriangleVectorForEdit(currentDisplayMesh);
    const std::vector<std::shared_ptr<Triangle>> afterDisplayMesh = evaluateDisplayMeshFromBase(mergedMesh, beforeModifierStack);
    const std::vector<std::shared_ptr<Triangle>> afterBaseMesh = cloneTriangleVectorForEdit(mergedMesh);
    const MeshModifiers::ModifierStack afterModifierStack = beforeModifierStack;

    auto command = std::make_unique<ReplaceMeshGeometryCommand>(
        objectName,
        beforeDisplayMesh,
        afterDisplayMesh,
        currentBaseMesh,
        afterBaseMesh,
        beforeModifierStack,
        afterModifierStack);
    command->execute(ctx);
    history.record(std::move(command));
    rebuildMeshCache(ctx.scene.world.objects);
    applyMeshShadingSettings(ctx, objectName, true);

    editable_mesh_cache = EditableMeshCache{};
    mesh_overlay_cache = MeshOverlayCache{};
    mesh_edit_layer = MeshEditLayer{};
    pending_serialized_mesh_edit_layer = PendingSerializedMeshEditLayer{};
    clearEditableMeshSelection();
    active_mesh_edit_object_name = objectName;

    auto refreshedMeshIt = mesh_cache.find(objectName);
    if (refreshedMeshIt != mesh_cache.end() && !refreshedMeshIt->second.empty() && refreshedMeshIt->second.front().second) {
        ctx.selection.selectObject(refreshedMeshIt->second.front().second, -1, objectName);
        active_mesh_edit_object_ptr = refreshedMeshIt->second.front().second.get();
    } else {
        active_mesh_edit_object_ptr = nullptr;
    }

    ensureMeshEditLayer(ctx, objectName);
    if (ensureEditableMeshCache(ctx, objectName)) {
        int bestVertexId = -1;
        float bestDistanceSq = std::numeric_limits<float>::max();
        for (size_t vertexId = 0; vertexId < editable_mesh_cache.vertices.size(); ++vertexId) {
            const float distanceSq = (editable_mesh_cache.vertices[vertexId].local_position - mergeCenter).length_squared();
            if (distanceSq < bestDistanceSq) {
                bestDistanceSq = distanceSq;
                bestVertexId = static_cast<int>(vertexId);
            }
        }
        if (bestVertexId >= 0) {
            editable_mesh_cache.selection.vertex_ids = { bestVertexId };
            editable_mesh_cache.selection.active_vertex_id = bestVertexId;
            // Keep Combined mode; only single-type modes snap to the result type.
            if (ctx.selection.mesh_element_mode != MeshElementSelectMode::Combined) {
                ctx.selection.mesh_element_mode = MeshElementSelectMode::Vertex;
            }
        }
    }

    addViewportMessage("Merged selected vertices", 2.0f, ImVec4(0.34f, 0.84f, 1.0f, 1.0f));
    return true;
}

bool SceneUI::weldSelectedVerticesByDistance(UIContext& ctx, float distance) {
    if (distance <= 0.0f) {
        return false;
    }

    const std::string objectName =
        (!active_mesh_edit_object_name.empty() ? active_mesh_edit_object_name :
            (ctx.selection.selected.type == SelectableType::Object && ctx.selection.selected.object
                ? ctx.selection.selected.object->getNodeName()
                : std::string{}));
    if (objectName.empty()) {
        return false;
    }

    if (!ensureEditableMeshCache(ctx, objectName)) {
        return false;
    }

    std::vector<int> selectedVertexIds = editable_mesh_cache.selection.vertex_ids;
    std::sort(selectedVertexIds.begin(), selectedVertexIds.end());
    selectedVertexIds.erase(std::unique(selectedVertexIds.begin(), selectedVertexIds.end()), selectedVertexIds.end());
    if (selectedVertexIds.size() < 2) {
        return false;
    }

    for (const int vertexId : selectedVertexIds) {
        if (!isEditableVertexIdValid(editable_mesh_cache, vertexId)) {
            return false;
        }
    }

    const float weldDistanceSq = distance * distance;
    std::vector<int> parent(selectedVertexIds.size());
    std::iota(parent.begin(), parent.end(), 0);

    const auto findRoot = [&](auto&& self, int index) -> int {
        if (parent[index] == index) {
            return index;
        }
        parent[index] = self(self, parent[index]);
        return parent[index];
    };

    const auto unite = [&](int a, int b) {
        const int rootA = findRoot(findRoot, a);
        const int rootB = findRoot(findRoot, b);
        if (rootA != rootB) {
            parent[rootB] = rootA;
        }
    };

    for (size_t i = 0; i < selectedVertexIds.size(); ++i) {
        const Vec3 pi = editable_mesh_cache.vertices[selectedVertexIds[i]].local_position;
        for (size_t j = i + 1; j < selectedVertexIds.size(); ++j) {
            const Vec3 pj = editable_mesh_cache.vertices[selectedVertexIds[j]].local_position;
            if ((pj - pi).length_squared() <= weldDistanceSq) {
                unite(static_cast<int>(i), static_cast<int>(j));
            }
        }
    }

    std::unordered_map<int, std::vector<int>> clusterIndices;
    clusterIndices.reserve(selectedVertexIds.size());
    for (size_t i = 0; i < selectedVertexIds.size(); ++i) {
        clusterIndices[findRoot(findRoot, static_cast<int>(i))].push_back(static_cast<int>(i));
    }

    std::unordered_map<QuantizedVertexKey, Vec3, QuantizedVertexKeyHasher> weldedPositions;
    weldedPositions.reserve(selectedVertexIds.size());
    size_t weldedClusterCount = 0;
    for (const auto& [rootIndex, memberIndices] : clusterIndices) {
        (void)rootIndex;
        if (memberIndices.size() < 2) {
            continue;
        }

        Vec3 clusterCenter(0.0f, 0.0f, 0.0f);
        for (const int memberIndex : memberIndices) {
            clusterCenter += editable_mesh_cache.vertices[selectedVertexIds[memberIndex]].local_position;
        }
        clusterCenter /= static_cast<float>(memberIndices.size());

        for (const int memberIndex : memberIndices) {
            const int vertexId = selectedVertexIds[memberIndex];
            weldedPositions[quantizeTopologyVertex(editable_mesh_cache.vertices[vertexId].local_position)] = clusterCenter;
        }
        ++weldedClusterCount;
    }

    if (weldedClusterCount == 0 || weldedPositions.empty()) {
        return false;
    }

    std::vector<std::shared_ptr<Triangle>> currentDisplayMesh;
    {
        auto meshIt = mesh_cache.find(objectName);
        if (meshIt == mesh_cache.end() || meshIt->second.empty()) {
            return false;
        }

        std::unordered_set<const Triangle*> seenTriangles;
        seenTriangles.reserve(meshIt->second.size());
        currentDisplayMesh.reserve(meshIt->second.size());
        for (const auto& entry : meshIt->second) {
            if (entry.second && seenTriangles.insert(entry.second.get()).second) {
                currentDisplayMesh.push_back(entry.second);
            }
        }
    }
    if (currentDisplayMesh.empty()) {
        return false;
    }

    std::vector<std::shared_ptr<Triangle>> currentBaseMesh;
    {
        auto baseIt = ctx.scene.base_mesh_cache.find(objectName);
        if (baseIt != ctx.scene.base_mesh_cache.end() && !baseIt->second.empty()) {
            currentBaseMesh = cloneTriangleVectorForEdit(baseIt->second);
        } else {
            currentBaseMesh = cloneTriangleVectorForEdit(currentDisplayMesh);
        }
    }

    const auto modifierIt = ctx.scene.mesh_modifiers.find(objectName);
    const MeshModifiers::ModifierStack beforeModifierStack =
        (modifierIt != ctx.scene.mesh_modifiers.end()) ? modifierIt->second : MeshModifiers::ModifierStack{};
    const bool preserveModifierPreview = hasEnabledSubdivisionPreview(beforeModifierStack);

    std::vector<std::shared_ptr<Triangle>> weldedMesh;
    const auto& sourceTopologyMesh = preserveModifierPreview ? currentBaseMesh : currentDisplayMesh;
    weldedMesh.reserve(sourceTopologyMesh.size());
    bool changed = false;
    for (const auto& sourceTriangle : sourceTopologyMesh) {
        auto tri = cloneTriangleForEdit(sourceTriangle);
        if (!tri) {
            continue;
        }

        for (int corner = 0; corner < 3; ++corner) {
            const auto weldIt = weldedPositions.find(quantizeTopologyVertex(tri->getOriginalVertexPosition(corner)));
            if (weldIt != weldedPositions.end()) {
                tri->setOriginalVertexPosition(corner, weldIt->second);
                changed = true;
            }
        }

        const Vec3 p0 = tri->getOriginalVertexPosition(0);
        const Vec3 p1 = tri->getOriginalVertexPosition(1);
        const Vec3 p2 = tri->getOriginalVertexPosition(2);
        const float areaSq = Vec3::cross(p1 - p0, p2 - p0).length_squared();
        if (areaSq <= 1e-12f) {
            continue;
        }

        tri->markAABBDirty();
        weldedMesh.push_back(tri);
    }

    if (!changed) {
        return false;
    }

    for (size_t triangleIndex = 0; triangleIndex < weldedMesh.size(); ++triangleIndex) {
        if (weldedMesh[triangleIndex]) {
            weldedMesh[triangleIndex]->setFaceIndex(static_cast<int>(triangleIndex));
        }
    }

    const std::vector<std::shared_ptr<Triangle>> beforeDisplayMesh = cloneTriangleVectorForEdit(currentDisplayMesh);
    const std::vector<std::shared_ptr<Triangle>> afterDisplayMesh = evaluateDisplayMeshFromBase(weldedMesh, beforeModifierStack);
    const std::vector<std::shared_ptr<Triangle>> afterBaseMesh = cloneTriangleVectorForEdit(weldedMesh);
    const MeshModifiers::ModifierStack afterModifierStack = beforeModifierStack;

    auto command = std::make_unique<ReplaceMeshGeometryCommand>(
        objectName,
        beforeDisplayMesh,
        afterDisplayMesh,
        currentBaseMesh,
        afterBaseMesh,
        beforeModifierStack,
        afterModifierStack);
    command->execute(ctx);
    history.record(std::move(command));
    rebuildMeshCache(ctx.scene.world.objects);
    applyMeshShadingSettings(ctx, objectName, true);

    editable_mesh_cache = EditableMeshCache{};
    mesh_overlay_cache = MeshOverlayCache{};
    mesh_edit_layer = MeshEditLayer{};
    pending_serialized_mesh_edit_layer = PendingSerializedMeshEditLayer{};
    clearEditableMeshSelection();
    active_mesh_edit_object_name = objectName;

    auto refreshedMeshIt = mesh_cache.find(objectName);
    if (refreshedMeshIt != mesh_cache.end() && !refreshedMeshIt->second.empty() && refreshedMeshIt->second.front().second) {
        ctx.selection.selectObject(refreshedMeshIt->second.front().second, -1, objectName);
        active_mesh_edit_object_ptr = refreshedMeshIt->second.front().second.get();
    } else {
        active_mesh_edit_object_ptr = nullptr;
    }

    ensureMeshEditLayer(ctx, objectName);
    if (ensureEditableMeshCache(ctx, objectName)) {
        std::vector<int> reselection;
        reselection.reserve(weldedPositions.size());
        for (const auto& [key, weldedPosition] : weldedPositions) {
            (void)key;
            int bestVertexId = -1;
            float bestDistanceSq = std::numeric_limits<float>::max();
            for (size_t vertexId = 0; vertexId < editable_mesh_cache.vertices.size(); ++vertexId) {
                const float distanceSq =
                    (editable_mesh_cache.vertices[vertexId].local_position - weldedPosition).length_squared();
                if (distanceSq < bestDistanceSq) {
                    bestDistanceSq = distanceSq;
                    bestVertexId = static_cast<int>(vertexId);
                }
            }
            if (bestVertexId >= 0) {
                reselection.push_back(bestVertexId);
            }
        }

        std::sort(reselection.begin(), reselection.end());
        reselection.erase(std::unique(reselection.begin(), reselection.end()), reselection.end());
        editable_mesh_cache.selection.vertex_ids = reselection;
        editable_mesh_cache.selection.active_vertex_id = reselection.empty() ? -1 : reselection.front();
        editable_mesh_cache.selection.edge_ids.clear();
        editable_mesh_cache.selection.face_ids.clear();
        editable_mesh_cache.selection.active_edge_id = -1;
        editable_mesh_cache.selection.active_face_id = -1;
        // Keep Combined mode; only single-type modes snap to the result type.
        if (ctx.selection.mesh_element_mode != MeshElementSelectMode::Combined) {
            ctx.selection.mesh_element_mode = MeshElementSelectMode::Vertex;
        }
    }

    addViewportMessage("Welded selected vertices", 2.0f, ImVec4(0.34f, 0.84f, 1.0f, 1.0f));
    return true;
}

void SceneUI::queueMeshEditGpuSync(const std::string& objectName) {
    mesh_edit_gpu_sync_pending = true;
    mesh_edit_gpu_sync_object_name = objectName;
    // Every mesh-mutating path funnels through here, so this is also the
    // single choke point that invalidates the GPU edit-overlay positions.
    gpu_edit_overlay_sync.geometry_dirty = true;
}

void SceneUI::handleMeshSculpt(UIContext& ctx, const Vec3* overrideHitPoint) {
    if (terrain_sculpt_proxy_active) {
        return;
    }
    if (ctx.selection.selected.type == SelectableType::Object && ctx.selection.selected.object) {
        auto tri = std::dynamic_pointer_cast<Triangle>(ctx.selection.selected.object);
        if (tri && tri->terrain_id != -1) {
            return;
        }
    }

    ImGuiIO& io = ImGui::GetIO();
    const bool sculptActive =
        sculpt_mode_state.enabled &&
        mesh_workspace_mode == MeshWorkspaceMode::Sculpt &&
        mesh_overlay_settings.edit_mode &&
        !sculpt_mode_state.active_target_name.empty();
    const bool liveAccumulationEnabled = sculpt_mode_state.accumulate_live;
    const SculptBrushTool activeTool = io.KeyShift ? SculptBrushTool::Smooth : sculpt_mode_state.tool;

    auto finishStroke = [&]() {
        if (!sculpt_stroke_state.active) {
            return;
        }

        if (sculpt_stroke_state.changed && !sculpt_stroke_state.object_name.empty()) {
            const bool activeViewportInSolidMode =
                ctx.backend_ptr &&
                dynamic_cast<Backend::IViewportBackend*>(ctx.backend_ptr) != nullptr &&
                ctx.backend_ptr->getViewportMode() != Backend::ViewportMode::Rendered;

            if (activeViewportInSolidMode && !sculpt_stroke_state.touched_triangles.empty()) {
                std::vector<std::shared_ptr<Triangle>> touchedTriangles;
                touchedTriangles.reserve(sculpt_stroke_state.touched_triangles.size());
                for (const auto& [trianglePtr, triangle] : sculpt_stroke_state.touched_triangles) {
                    (void)trianglePtr;
                    if (triangle) {
                        touchedTriangles.push_back(triangle);
                    }
                }
                if (!touchedTriangles.empty()) {
                    const std::vector<size_t> affectedVertexIds =
                        collectAffectedEditableVertexIds(editable_mesh_cache, touchedTriangles);
                    recomputeEditableSmoothNormals(editable_mesh_cache, affectedVertexIds);

                    const Matrix4x4 transform = getEditableObjectTransform(editable_mesh_cache);
                    const Matrix4x4 inverseTransform = transform.inverse();
                    const Matrix4x4 normalTransform = inverseTransform.transpose();
                    for (const auto& tri : touchedTriangles) {
                        for (int i = 0; i < 3; ++i) {
                            tri->vertices[i].position = transform.transform_point(tri->vertices[i].original);
                            tri->vertices[i].normal = safeNormalizeVec3(
                                normalTransform.transform_vector(tri->vertices[i].originalNormal),
                                tri->vertices[i].normal);
                        }
                        tri->markAABBDirty();
                        tri->update_bounding_box();
                    }
                }
            }

            // Keep the non-destructive edit layer's "edited" snapshot current so the
            // mute/unmute toggle and serialization reflect this stroke. Recapturing the
            // WHOLE mesh (refreshMeshEditLayerEditedState) is O(N) and was the freeze felt
            // on brush release over a multi-million-triangle object. edited_states is stored
            // in the same order as triangle_to_mesh_index, so update only the stroke's
            // touched entries in place; fall back to the full recapture only if the layout
            // no longer matches (e.g. a subdivision-preview source mesh).
            if (mesh_edit_layer.active &&
                mesh_edit_layer.object_name == sculpt_stroke_state.object_name &&
                !mesh_edit_layer.edited_states.empty()) {
                bool layoutMatches = true;
                for (const auto& [trianglePtr, triangle] : sculpt_stroke_state.touched_triangles) {
                    (void)trianglePtr;
                    if (!triangle) {
                        continue;
                    }
                    const int faceIdx = editableFaceIndexOf(editable_mesh_cache, triangle.get());
                    if (faceIdx < 0) {
                        layoutMatches = false;
                        break;
                    }
                    const size_t meshIdx = static_cast<size_t>(
                        editable_mesh_cache.face_to_mesh_index[static_cast<size_t>(faceIdx)]);
                    if (meshIdx >= mesh_edit_layer.edited_states.size() ||
                        mesh_edit_layer.edited_states[meshIdx].triangle.get() != triangle.get()) {
                        layoutMatches = false;
                        break;
                    }
                    auto& state = mesh_edit_layer.edited_states[meshIdx];
                    for (int corner = 0; corner < 3; ++corner) {
                        state.positions[corner] = triangle->getOriginalVertexPosition(corner);
                    }
                }
                if (!layoutMatches) {
                    refreshMeshEditLayerEditedState(ctx);
                }
            } else {
                refreshMeshEditLayerEditedState(ctx);
            }

            std::vector<MeshEditTriangleState> endStates;
            endStates.reserve(sculpt_stroke_state.before_triangle_states.size());
            for (const auto& [trianglePtr, beforeState] : sculpt_stroke_state.before_triangle_states) {
                (void)trianglePtr;
                if (!beforeState.triangle) {
                    continue;
                }
                MeshEditTriangleState endState;
                endState.triangle = beforeState.triangle;
                for (int corner = 0; corner < 3; ++corner) {
                    endState.positions[corner] = beforeState.triangle->getOriginalVertexPosition(corner);
                }
                endStates.push_back(endState);
            }
            if (!sculpt_stroke_state.before_triangle_states.empty() && !endStates.empty()) {
                std::vector<MeshEditTriangleState> beforeStates;
                beforeStates.reserve(sculpt_stroke_state.before_triangle_states.size());
                for (const auto& [trianglePtr, beforeState] : sculpt_stroke_state.before_triangle_states) {
                    (void)trianglePtr;
                    beforeStates.push_back(beforeState);
                }
                history.record(std::make_unique<MeshEditCommand>(
                    sculpt_stroke_state.object_name,
                    std::move(beforeStates),
                    endStates));
            }

            // Mark scene geometry dirty even when the solid-mode raster sync path
            // handled the live viewport update. The Rendered/Vulkan RT transition
            // checks these flags/generation to decide whether a full RT geometry
            // sync is needed after leaving Solid mode.
            extern bool g_geometry_dirty;
            extern std::atomic<uint64_t> g_scene_geometry_generation;
            g_geometry_dirty = true;
            g_scene_geometry_generation.fetch_add(1, std::memory_order_release);

            // Expand the cached bbox by the stroke's touched verts (O(touched)) rather than
            // rescanning the whole mesh — updateBBoxCache is O(N) and was the other half of
            // the release freeze on a multi-million-triangle object. A brush stroke only
            // grows the bound in practice; a slightly loose bbox is harmless for selection /
            // picking and is recomputed exactly on the next mesh-cache rebuild. Fall back to
            // the full scan when no cached bound exists yet.
            {
                auto bboxIt = bbox_cache.find(sculpt_stroke_state.object_name);
                if (bboxIt != bbox_cache.end() && !sculpt_stroke_state.touched_triangles.empty()) {
                    Vec3 bb_min = bboxIt->second.first;
                    Vec3 bb_max = bboxIt->second.second;
                    for (const auto& [trianglePtr, triangle] : sculpt_stroke_state.touched_triangles) {
                        (void)trianglePtr;
                        if (!triangle) {
                            continue;
                        }
                        for (int c = 0; c < 3; ++c) {
                            const Vec3 v = triangle->getOriginalVertexPosition(c);
                            bb_min.x = fminf(bb_min.x, v.x);
                            bb_min.y = fminf(bb_min.y, v.y);
                            bb_min.z = fminf(bb_min.z, v.z);
                            bb_max.x = fmaxf(bb_max.x, v.x);
                            bb_max.y = fmaxf(bb_max.y, v.y);
                            bb_max.z = fmaxf(bb_max.z, v.z);
                        }
                    }
                    bboxIt->second = { bb_min, bb_max };
                    hull_candidate_cache.erase(sculpt_stroke_state.object_name);
                } else {
                    updateBBoxCache(sculpt_stroke_state.object_name);
                }
            }

            // Rebuild the dirty-triangle set for the stroke-end sync from the FULL stroke's
            // touched triangles. The per-dab sync clears sculpt_dirty_mesh_cache_indices after
            // each dab, so by the time finishStroke queues its sync the set is empty — and
            // processPendingMeshEditGpuSync treats an empty dirty set as "unknown change" and
            // falls back to a FULL geometry rebuild (collect-all triangles + Vulkan/BLAS full
            // rebuild). That full rebuild is the freeze felt on brush release, and when strokes
            // arrive quickly it bleeds into the next stroke as an intermittent mid-stroke
            // stutter. Repopulating from the stroke's touched triangles keeps the release on
            // the cheap PARTIAL upload path and also carries finishStroke's final solid-mode
            // normal recompute to the GPU.
            sculpt_dirty_mesh_cache_indices.clear();
            sculpt_dirty_mesh_cache_indices.reserve(sculpt_stroke_state.touched_triangles.size());
            for (const auto& [trianglePtr, triangle] : sculpt_stroke_state.touched_triangles) {
                (void)trianglePtr;
                if (!triangle) {
                    continue;
                }
                const int faceIdx = editableFaceIndexOf(editable_mesh_cache, triangle.get());
                if (faceIdx >= 0) {
                    sculpt_dirty_mesh_cache_indices.push_back(editable_mesh_cache.face_to_mesh_index[static_cast<size_t>(faceIdx)]);
                }
            }

            if (ctx.backend_ptr != nullptr || g_backend != nullptr || g_viewport_backend != nullptr) {
                queueMeshEditGpuSync(sculpt_stroke_state.object_name);
            }
            // CPU picking/CPU-render BVH at stroke end. The CPU BVH is consulted only by
            // picking (and that tries GPU pick first) and the CPU reference render — never
            // mid-sculpt — so a synchronous whole-mesh Embree refit on release would be a
            // pure freeze for a multi-million-triangle mesh. Make it LAZY: for GPU-backed
            // viewports just mark the BVH stale and let the pick fallback / CPU render
            // promote it to a refit on demand (see g_cpu_bvh_stale). The CPU-render-as-
            // viewport path (no backend) IS the thing being drawn, so refit it right away.
            {
                extern bool g_cpu_bvh_refit_pending;
                extern bool g_cpu_bvh_stale;
                if (!ctx.backend_ptr) {
                    g_cpu_bvh_refit_pending = true;
                } else {
                    g_cpu_bvh_stale = true;
                }
            }

            ProjectManager::getInstance().markModified();
        }

        sculpt_stroke_state = SculptStrokeState{};
    };

    if (!sculptActive) {
        finishStroke();
        return;
    }

    if (!ctx.scene.camera || io.WantCaptureMouse || ImGuizmo::IsOver() || ImGuizmo::IsUsing()) {
        finishStroke();
        return;
    }

    const std::string objectName = sculpt_mode_state.active_target_name;
    if (!ensureEditableMeshCache(ctx, objectName)) {
        finishStroke();
        return;
    }
    if (!ensureSculptControlGraph(ctx, objectName)) {
        finishStroke();
        return;
    }
    if (!ensureSculptPBVH(ctx, objectName)) {
        finishStroke();
        return;
    }
    if (!mesh_edit_layer.active || mesh_edit_layer.object_name != objectName) {
        ensureMeshEditLayer(ctx, objectName);
    }
    if (!mesh_edit_layer.active || !mesh_edit_layer.enabled) {
        finishStroke();
        return;
    }

    // Idle early-out: with no stroke running and the left button neither held nor just
    // clicked this frame, there is nothing to sculpt — so skip the per-frame CPU picking
    // sync + scene-BVH raycast + PBVH refine below. Those ran EVERY frame even with the
    // cursor parked off the object (the mouse-state check used to come only AFTER them),
    // which was a constant idle CPU draw. The brush PREVIEW does its own cached raycast, so
    // the cursor ring is unaffected. (active==false here ⇒ no stroke to finish.)
    if (!overrideHitPoint &&
        !sculpt_stroke_state.active &&
        !ImGui::IsMouseDown(ImGuiMouseButton_Left) &&
        !ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        return;
    }

    auto meshEntryIt = mesh_cache.find(objectName);
    const bool objectHadPendingCpuSync = objects_needing_cpu_sync.count(objectName) > 0;
    if (!sculpt_stroke_state.active &&
        meshEntryIt != mesh_cache.end() &&
        !meshEntryIt->second.empty() &&
        (!picking_vertices_synced || objectHadPendingCpuSync)) {
        for (auto& entry : meshEntryIt->second) {
            if (entry.second) {
                entry.second->updateTransformedVertices();
            }
        }
        objects_needing_cpu_sync.erase(objectName);
    }

    ensureCPUSyncForPicking(ctx);
    const ImVec2 displaySize = ImGui::GetIO().DisplaySize;
    const float u = io.MousePos.x / (std::max)(1.0f, displaySize.x);
    const float v = 1.0f - (io.MousePos.y / (std::max)(1.0f, displaySize.y));
    const Ray ray = ctx.scene.camera->get_ray(u, v);

    HitRecord hit;
    bool rawDidHit = false;

    // Use the scene BVH as the hot path for sculpt picking. The previous behavior
    // linearly scanned every triangle of the selected object each mouse update,
    // which becomes catastrophic on dense meshes (e.g. 500k+ tris). Keep the
    // direct-object scan only as a narrow fallback when the selected object was
    // just CPU-synced and the scene BVH may still be one frame behind.
    rawDidHit = raycastViewportHit(ctx, io.MousePos, hit) &&
        hit.triangle &&
        hit.triangle->getNodeName() == objectName;

    if (!rawDidHit &&
        objectHadPendingCpuSync &&
        meshEntryIt != mesh_cache.end() &&
        !meshEntryIt->second.empty()) {
        rawDidHit = raycastEditableObjectTriangles(meshEntryIt->second, ray, hit);
    }
    bool didHit = rawDidHit;
    if (didHit) {
        hit.normal = computeStableSculptHitNormal(hit, hit.normal);
        hit.interpolated_normal = hit.normal;
    }

    // Refine the surface hit against the LIVE sculpt PBVH (leaf bounds refit per dab) so
    // the brush keeps catching geometry the deferred scene-BVH refit has not caught up to
    // (e.g. a region pushed far out mid-stroke). Only on the real call — sub-steps reuse
    // the corrected segment endpoints.
    if (!overrideHitPoint) {
        didHit = refineSculptHitWithPBVH(ray, objectName, hit, didHit);
    }

    // Sub-step dab: override the brush center with the interpolated segment point.
    // The mouse-picked normal is a good-enough surface normal over the short span.
    if (overrideHitPoint && didHit) {
        hit.point = sanitizeVec3(*overrideHitPoint, hit.point);
    }

    // Stroke begin / end and mouse polling only happen on the real (non-substep)
    // call. Sub-step dabs run mid-stroke with the mouse already held.
    if (!overrideHitPoint) {
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && didHit) {
            beginSculptStroke(
                sculpt_stroke_state,
                activeTool,
                objectName,
                hit,
                editable_mesh_cache.vertices.size());
            // Force a full cache->buffer resync on the stroke's first solve frame. The
            // persistent sculpt_updated_local_positions may be stale from a prior stroke,
            // an undo, or an external cache edit; clearing it triggers the size-mismatch
            // rebuild in the solve block exactly once (keeps capacity, no realloc churn).
            sculpt_updated_local_positions.clear();
        }

        if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            finishStroke();
            return;
        }
    }

    if (sculpt_stroke_state.active &&
        sculpt_stroke_state.object_name == objectName &&
        isGrabFamilyTool(activeTool) &&
        !didHit) {
        const Vec3 fallbackNormal = sculpt_stroke_state.stroke_normal.length_squared() > 1e-8f
            ? sculpt_stroke_state.stroke_normal.normalize()
            : Vec3(0.0f, 1.0f, 0.0f);
        Vec3 planeHit;
        if (intersectRayPlane(ray, sculpt_stroke_state.start_world_hit, fallbackNormal, planeHit)) {
            hit.point = planeHit;
            hit.normal = fallbackNormal;
            didHit = true;
        }
    }

    if (!sculpt_stroke_state.active || !didHit || sculpt_stroke_state.object_name != objectName) {
        return;
    }

    const Matrix4x4 transform = getEditableObjectTransform(editable_mesh_cache);
    const Matrix4x4 inverseTransform = transform.inverse();
    const Vec3 fallbackWorldNormal(0.0f, 1.0f, 0.0f);
    const Vec3 strokeNormal = sanitizeVec3(sculpt_stroke_state.stroke_normal, fallbackWorldNormal);
    const Vec3 hitNormal = sanitizeVec3(hit.normal, fallbackWorldNormal);
    const Vec3 hitPoint = sanitizeVec3(hit.point, Vec3(0.0f, 0.0f, 0.0f));
    const Vec3 hitNormalWorld = strokeNormal.length_squared() > 1e-8f
        ? safeNormalizeVec3(strokeNormal, fallbackWorldNormal)
        : safeNormalizeVec3(hitNormal, fallbackWorldNormal);
    // Sculpt can use either object/world units or a screen-space brush. Keep the
    // preview and the actual vertex selection on the same resolved world radius.
    float radiusWorld = sanitizeFiniteFloat(sculpt_mode_state.brush.radius, 0.3f, 0.0001f, 1000.0f);
    if (sculpt_mode_state.use_screen_space_radius && ctx.scene.camera) {
        const float radiusPx = sanitizeFiniteFloat(sculpt_mode_state.screen_radius_px, 72.0f, 1.0f, 1000.0f);
        const float estimatedRadius = estimateBrushWorldRadius(
            *ctx.scene.camera,
            ImGui::GetIO().DisplaySize,
            hitPoint,
            radiusPx);
        if (std::isfinite(estimatedRadius) && estimatedRadius > 1e-5f) {
            radiusWorld = std::clamp(estimatedRadius, 0.0001f, 1000.0f);
        }
    }

    // Anchored stamp runs an ABSOLUTE coherent solve from a frozen anchor every frame
    // (like grab), so it must NOT go through the additive distance-gate below — that
    // gate skips frames where the cursor barely moves, which would freeze the live
    // depth/rotation drag. Computed here (before the scheduler); the frozen frame and
    // drag-derived depth/rotation are filled in further down once the tangent frame exists.
    const bool anchoredActive =
        sculpt_mode_state.stroke_anchored && activeTool == SculptBrushTool::Stamp;

    // --- Stroke dab scheduling (sub-stepping + distance gating) ---
    // Two distinct problems, two paths, both driven from the real (non-substep)
    // call. Each emitted sub-dab recurses with overrideHitPoint and advances
    // last_world_hit so the chain stays seamless.
    //
    //  * Grab-family (Grab/Elastic/SnakeHook): an absolute, idempotent solve
    //    (startLocal + drag*weight). It MUST run every frame to keep following
    //    the cursor, and re-applying it at the same spot does nothing — so there
    //    is no pile-up. Only sub-step to seed dynamic verts along fast segments.
    //
    //  * Additive brushes (DrawSharp/Clay/Inflate/Fill/Nudge/...): every dab ADDS
    //    displacement, so applying one per frame hammers the same spot when the
    //    cursor moves slowly or the frame rate drops on a high-tess mesh — that's
    //    the "circles piling up / block-block" artifact. Gate dabs by DISTANCE:
    //    place one only every `spacing` units travelled, carry the remainder, and
    //    emit nothing on frames that didn't travel a full spacing. Same mechanism
    //    also interpolates fast strokes (multiple dabs per segment).
    if (!overrideHitPoint && sculpt_stroke_state.active && sculpt_stroke_state.changed && didHit) {
        const Vec3 curHit = hitPoint;
        const float spacing = (std::max)(radiusWorld * 0.25f, 1e-4f);

        // Smooth/relax is NOT additive — it pulls each vertex toward its local
        // neighbour average, so it converges instead of piling up. The distance-gate
        // (below) was meant for stamp brushes that hammer one spot when the cursor is
        // slow/stationary; applying it to Smooth meant holding the brush still over a
        // bumpy area did NOTHING until you moved a full `spacing`, so it felt like the
        // smoothing only "landed" on release. Route Smooth through the continuous
        // (grab-style) path: apply every frame, and sub-step along fast segments.
        const bool continuousTool =
            isGrabFamilyTool(activeTool) || activeTool == SculptBrushTool::Smooth ||
            anchoredActive;
        if (continuousTool) {
            const bool absoluteGrab = activeTool == SculptBrushTool::Grab ||
                                      activeTool == SculptBrushTool::ElasticDeform ||
                                      anchoredActive;
            if (absoluteGrab) {
                // Grab/Elastic are an ABSOLUTE coherent solve (start_local + weight*delta)
                // and the per-vertex topology guard is now BYPASSED for grab-family in the
                // commit loop (the coherent field can't spike a triangle the way an
                // additive brush can). Sub-stepping only ever existed to keep each commit
                // under that guard's per-step rotation limit; with no guard to satisfy the
                // intermediate steps are pure redundant work — one final absolute commit
                // from the frozen start reproduces the exact same positions. Commit once
                // per frame, which also drops up to 32x the commit-loop cost on dense grab
                // strokes. The single-dab path below recomputes the plane hit from the real
                // cursor ray, so the final position is unchanged. (fall through)
            } else {
                // SnakeHook (dynamic re-prime — sweeps in new geometry) and Smooth (relax
                // toward neighbour average — must run every frame, not distance-gated) still
                // sub-step along fast segments so a quick stroke doesn't skip travel.
                const Vec3 segStart = sanitizeVec3(sculpt_stroke_state.last_world_hit, curHit);
                const Vec3 segEnd = curHit;
                const float meshSeg = (segEnd - segStart).length();
                const int substeps = std::clamp(
                    static_cast<int>(std::ceil(meshSeg / spacing)), 1, 24);
                if (substeps > 1) {
                    for (int i = 1; i <= substeps; ++i) {
                        const Vec3 sub = segStart + (segEnd - segStart) *
                            (static_cast<float>(i) / static_cast<float>(substeps));
                        handleMeshSculpt(ctx, &sub);
                    }
                    return;
                }
            }
            // Grab/Elastic + single-step SnakeHook/Smooth: one dab this frame — fall through.
        } else {
            // Anchor on the last emitted dab (carries the sub-spacing remainder);
            // on the very first gated frame fall back to last_world_hit (the click).
            const Vec3 lastDab = sculpt_stroke_state.has_last_dab
                ? sanitizeVec3(sculpt_stroke_state.last_dab_world, curHit)
                : sanitizeVec3(sculpt_stroke_state.last_world_hit, curHit);
            const Vec3 seg = curHit - lastDab;
            const float segDist = seg.length();
            if (segDist < spacing) {
                // Not enough travel for a new dab — skip this frame entirely so
                // the spot isn't hammered. last_dab_world / last_world_hit unchanged.
                return;
            }
            const Vec3 dir = seg / segDist;
            // Per-frame work budget. Every dab re-gathers its footprint and re-runs the
            // commit + relax passes, so the frame cost is (dabs * footprintVerts). A fast
            // stroke drives the dab count toward the cap while a large/dense radius drives
            // the footprint up — the product is the "exponential" blow-up the user hit.
            // Cap the dab count so the per-frame vertex work stays under a budget: because
            // consecutive dabs are only spacing=radius*0.25 apart they overlap ~75%, so on
            // a large footprint the surplus dabs are near-duplicates and dropping them
            // barely changes coverage. telemetry_candidate_vertices is the previous dab's
            // measured footprint (0 on the stroke's first frame → full 24 dabs, then it
            // self-corrects). Typical brushes (small footprint) keep all 24 dabs untouched.
            const size_t lastFootprint = (std::max)(telemetry_candidate_vertices, size_t{1});
            constexpr size_t kSculptPerFrameDabVertexBudget = 250000;
            const int budgetDabs = static_cast<int>(std::clamp<size_t>(
                kSculptPerFrameDabVertexBudget / lastFootprint, size_t{1}, size_t{24}));
            const int idealDabs = static_cast<int>(std::floor(segDist / spacing));
            const int dabCount = std::clamp(idealDabs, 1, budgetDabs);
            if (dabCount >= idealDabs) {
                // Full sampling: fixed spacing, carry the sub-spacing remainder so the
                // next frame accumulates the tail (no pile-up at the same spot).
                for (int i = 1; i <= dabCount; ++i) {
                    const Vec3 sub = lastDab + dir * (spacing * static_cast<float>(i));
                    handleMeshSculpt(ctx, &sub);
                }
                sculpt_stroke_state.last_dab_world =
                    lastDab + dir * (spacing * static_cast<float>(dabCount));
            } else {
                // Budget-capped: SPAN the allowed dabs across the whole travelled segment
                // rather than clustering them at the start and dropping the tail — keeps
                // the stroke continuous (just lighter) so the brush doesn't visibly lag a
                // fast cursor. The large footprint that triggered the cap overlaps enough
                // to cover the wider gaps; the whole segment is consumed this frame.
                const float step = segDist / static_cast<float>(dabCount);
                for (int i = 1; i <= dabCount; ++i) {
                    const Vec3 sub = lastDab + dir * (step * static_cast<float>(i));
                    handleMeshSculpt(ctx, &sub);
                }
                sculpt_stroke_state.last_dab_world = curHit;
            }
            sculpt_stroke_state.has_last_dab = true;
            return;
        }
    }

    const float brushStrength = sanitizeFiniteFloat(sculpt_mode_state.brush.strength, 1.0f, 0.0f, 10.0f);
    // Per-dab timestep. Additive brushes are DISTANCE-gated (one dab per `spacing`
    // travelled) yet every deposit formula scales by this dt, so the per-dab amount
    // MUST stay frame-rate independent — otherwise the deposit-per-unit-length tracks
    // FPS. The old 0.25s cap let a single slow frame on a dense mesh deposit up to
    // ~15x a 60fps dab (and each sub-dab emitted that frame re-read the same large
    // io.DeltaTime), dumping saturated, discrete "circular block" craters whose
    // falloff edges all clipped past the visible threshold. Cap at the 60fps reference
    // the strength sliders are tuned against: high-fps frames (dt < 1/60) are untouched,
    // only the dense-mesh low-fps overshoot is removed. Grab-family tools use an
    // absolute (dt-free) solve, so they are unaffected.
    const float dt = sanitizeFiniteFloat(io.DeltaTime, 1.0f / 60.0f, 1.0f / 240.0f, 1.0f / 60.0f);
    const float normalStrength = sanitizeFiniteFloat(sculpt_mode_state.normal_strength, 0.35f, 0.0f, 8.0f);
    // Brush falloff, dome-corrected for large radii. computeTerrainLikeBrushWeight keeps
    // weight=1 across a central PLATEAU (n < 1-falloff), which is a fixed FRACTION of the
    // radius — so a wide brush became a huge flat mesa with a narrow, steep rim (the
    // "short side wall" at large diameter). Raising the effective falloff toward 1 as the
    // brush grows dissolves that plateau (inner = 1-falloff -> 0), so wide brushes fall off
    // smoothly from the center = a dome. Small brushes (radius <= 1 world unit) are
    // untouched; the dome ramps in over radius 1..4 via the existing large-brush factor.
    // This falloff VALUE only drives the plateau weight (grab uses falloffType, not this),
    // so the change is scoped to the additive/relax footprint shape. User-chosen behavior.
    const float falloffRaw = saturateFloat(sculpt_mode_state.brush.falloff);
    const float falloff = std::lerp(falloffRaw, 1.0f, computeLargeBrushSurfaceFactor(radiusWorld));
    const float directionSign = io.KeyCtrl ? -1.0f : 1.0f;
    const float brushFlow = sanitizeFiniteFloat(sculpt_mode_state.brush.flow, 1.0f, 0.05f, 4.0f);
    const float clayBrushStrength = computeSafeClayStrength(brushStrength, brushFlow);
    const float clayRadiusCompensation = computeClayRadiusCompensation(radiusWorld);

    // Tangent frame for alpha-mask / shape UV sampling (nx/ny in brush-local [-1,1] space).
    // VIEW-STABLE: derived from the camera right/up projected onto the surface tangent plane,
    // NOT from normal×worldUp. The old frame spun with the surface normal, so a directional
    // shape or alpha mask rotated arbitrarily as you moved across a curved surface (invisible
    // for a circle, wrong for Rectangle/Capsule/Flat and any alpha texture). Projecting the
    // camera axes keeps the footprint screen-stable; alpha_rotation_degrees still offsets it.
    Vec3 tangentWorld;
    Vec3 bitangentWorld;
    {
        Vec3 camRight(1.0f, 0.0f, 0.0f);
        Vec3 camUp(0.0f, 1.0f, 0.0f);
        if (ctx.scene.camera) {
            const Vec3 fwdRaw = ctx.scene.camera->lookat - ctx.scene.camera->lookfrom;
            const Vec3 fwd = fwdRaw.length_squared() > 1e-12f
                ? fwdRaw.normalize() : Vec3(0.0f, 0.0f, -1.0f);
            Vec3 r = fwd.cross(ctx.scene.camera->vup);
            if (r.length_squared() < 1e-12f) r = Vec3(1.0f, 0.0f, 0.0f);
            camRight = r.normalize();
            camUp = camRight.cross(fwd).normalize();
        }
        // Project the camera right onto the tangent plane; fall back to camera up at grazing
        // angles, then to the old normal-cross method if both are parallel to the normal.
        Vec3 t = camRight - hitNormalWorld * camRight.dot(hitNormalWorld);
        if (t.length_squared() < 1e-8f) {
            t = camUp - hitNormalWorld * camUp.dot(hitNormalWorld);
        }
        if (t.length_squared() < 1e-8f) {
            t = std::abs(hitNormalWorld.y) < 0.95f
                ? hitNormalWorld.cross(Vec3(0.0f, 1.0f, 0.0f))
                : hitNormalWorld.cross(Vec3(1.0f, 0.0f, 0.0f));
        }
        tangentWorld = safeNormalizeVec3(t, Vec3(1.0f, 0.0f, 0.0f));
        bitangentWorld = safeNormalizeVec3(hitNormalWorld.cross(tangentWorld), Vec3(0.0f, 0.0f, 1.0f));
        if (!isFiniteVec3(tangentWorld) || !isFiniteVec3(bitangentWorld)) {
            tangentWorld = Vec3(1.0f, 0.0f, 0.0f);
            bitangentWorld = Vec3(0.0f, 0.0f, 1.0f);
        }
    }

    Vec3 planeHit = hitPoint;
    if (isGrabFamilyTool(activeTool)) {
        const bool absoluteGrab = activeTool == SculptBrushTool::Grab ||
                                  activeTool == SculptBrushTool::ElasticDeform;
        if (absoluteGrab && overrideHitPoint) {
            // Sub-step call: the dab scheduler already interpolated this target ON the
            // camera-facing plane, so consume it directly. This is what makes the
            // absolute grab commit in small, topology-guard-safe increments instead of
            // one over-rotated jump (see the dab-scheduling comment above).
            planeHit = sanitizeVec3(*overrideHitPoint, hitPoint);
        } else {
            const Vec3 startWorldHit = sanitizeVec3(sculpt_stroke_state.start_world_hit, hitPoint);
            // Grab drag direction lives in the VIEW plane (perpendicular to the camera
            // forward axis), like Blender/ZBrush — NOT the surface tangent plane. Using
            // the surface normal as the drag plane normal locked motion to the tangent
            // and silently dropped the screen-up component whenever the normal leaned
            // toward screen-up or the surface was viewed at a grazing angle ("dragging
            // up doesn't move part of it" — the Y/Z mismatch). A camera-facing plane
            // maps screen movement to world movement 1:1 and keeps the ray well-
            // conditioned (never near-parallel to the plane).
            // NOTE: the brush FOOTPRINT (which verts are picked) still uses the surface
            // normal in primeGrabStrokeWeights — only the motion direction changes here.
            Vec3 grabPlaneNormal = hitNormalWorld;
            if (ctx.scene.camera) {
                const Vec3 camForward = ctx.scene.camera->lookat - ctx.scene.camera->lookfrom;
                if (camForward.length_squared() > 1e-10f) {
                    grabPlaneNormal = safeNormalizeVec3(camForward, hitNormalWorld);
                }
            }
            if (!intersectRayPlane(ray, startWorldHit, grabPlaneNormal, planeHit)) {
                planeHit = hitPoint;
            }
        }
    }
    // For Grab: use raw 3D delta so the mesh follows the mouse in all directions.
    // Stripping the normal component prevented movement when viewing at grazing angles.
    const Vec3 worldDragDelta = sanitizeVec3(planeHit - sanitizeVec3(sculpt_stroke_state.start_world_hit, planeHit), Vec3(0.0f, 0.0f, 0.0f));
    const Vec3 localGrabDelta = sanitizeVec3(
        inverseTransform.transform_point(planeHit) -
        inverseTransform.transform_point(sanitizeVec3(sculpt_stroke_state.start_world_hit, planeHit)),
        Vec3(0.0f, 0.0f, 0.0f));
    const Vec3 lastStrokeHit = sanitizeVec3(sculpt_stroke_state.last_world_hit, hitPoint);
    // Nudge / Snake Hook derive their motion from the CAMERA-FACING plane (screen
    // motion), not the raw surface hit. As the mesh deforms under the brush the
    // ray↔surface intersection drifts, so a surface-based step gives an
    // inconsistent drag axis and the pull stalls at the radius edge — the exact
    // axis-inconsistency we already fixed for Grab. Projecting the cursor ray onto
    // a camera-forward plane anchored at the previous step keeps the axis stable.
    Vec3 snakePlaneHit = hitPoint;
    Vec3 strokeStepWorld = sanitizeVec3(hitPoint - lastStrokeHit, Vec3(0.0f, 0.0f, 0.0f));
    // During sub-stepping the segment is already interpolated, so the step is
    // simply sub - lastSub (above) — skip the camera-plane projection.
    if (activeTool == SculptBrushTool::Nudge && !overrideHitPoint && ctx.scene.camera) {
        const Vec3 camFwd = safeNormalizeVec3(
            ctx.scene.camera->lookat - ctx.scene.camera->lookfrom, hitNormalWorld);
        Vec3 projHit;
        if (intersectRayPlane(ray, lastStrokeHit, camFwd, projHit)) {
            snakePlaneHit = projHit;
            strokeStepWorld = sanitizeVec3(projHit - lastStrokeHit, Vec3(0.0f, 0.0f, 0.0f));
        }
    }
    const Vec3 projectedStrokeStep = sanitizeVec3(
        strokeStepWorld - hitNormalWorld * strokeStepWorld.dot(hitNormalWorld),
        Vec3(0.0f, 0.0f, 0.0f));
    const Vec3 strokeTangentWorld = projectedStrokeStep.length_squared() > 1e-10f
        ? safeNormalizeVec3(projectedStrokeStep, tangentWorld)
        : tangentWorld;
    const Vec3 strokeBitangentWorld = safeNormalizeVec3(
        hitNormalWorld.cross(strokeTangentWorld),
        bitangentWorld);
    const bool clayLikeTool =
        activeTool == SculptBrushTool::Layer ||
        activeTool == SculptBrushTool::Clay ||
        activeTool == SculptBrushTool::ClayStrips;

    // --- Anchored stamp (Blender-style) -------------------------------------------
    // When active the brush centre/normal/tangent frame are FROZEN at the first frame
    // and the cursor drag drives the imprint DEPTH (drag length on the camera-facing
    // plane through the anchor) and ROTATION (drag angle in the frozen tangent frame),
    // instead of moving the centre. The deform is written absolutely from each frozen
    // footprint vertex's rest position every frame (grab-style), so it never piles up
    // and live-resizes/rotates as you drag (verts whose footprint weight falls to 0
    // return to rest automatically). Radius stays the brush-radius slider value.
    // (anchoredActive is declared earlier, before the dab scheduler.)
    float anchoredDepth = 0.0f;
    float anchoredRotationDeg = 0.0f;
    if (anchoredActive) {
        if (!sculpt_stroke_state.anchored_primed) {
            sculpt_stroke_state.anchor_world = sanitizeVec3(sculpt_stroke_state.start_world_hit, hitPoint);
            sculpt_stroke_state.anchor_normal = hitNormalWorld;
            sculpt_stroke_state.anchor_tangent = tangentWorld;
            sculpt_stroke_state.anchor_bitangent = bitangentWorld;
        }
        const Vec3 aC = sculpt_stroke_state.anchor_world;
        const Vec3 aN = sculpt_stroke_state.anchor_normal;
        // Project the cursor ray onto a camera-facing plane through the anchor so the
        // drag maps 1:1 to screen motion (same well-conditioned plane the grab drag
        // uses), then read depth = |drag| and rotation = angle in the frozen frame.
        Vec3 dragPlaneNormal = aN;
        if (ctx.scene.camera) {
            const Vec3 camF = ctx.scene.camera->lookat - ctx.scene.camera->lookfrom;
            if (camF.length_squared() > 1e-10f) {
                dragPlaneNormal = safeNormalizeVec3(camF, aN);
            }
        }
        Vec3 dragHit;
        if (!intersectRayPlane(ray, aC, dragPlaneNormal, dragHit)) {
            dragHit = hitPoint;
        }
        const Vec3 drag = sanitizeVec3(dragHit - aC, Vec3(0.0f, 0.0f, 0.0f));
        anchoredDepth = drag.length();
        const float du = drag.dot(sculpt_stroke_state.anchor_tangent);
        const float dv = drag.dot(sculpt_stroke_state.anchor_bitangent);
        if (du * du + dv * dv > 1e-10f) {
            anchoredRotationDeg = std::atan2(dv, du) * (180.0f / 3.14159265358979f);
        }
    }
    const float strokeDistance = projectedStrokeStep.length();
    const float strokeSpacing = radiusWorld * (std::max)(0.05f, sculpt_mode_state.brush.spacing);
    const float strokeAdvanceFactor = clayLikeTool
        ? computeStrokeAdvanceFactor(strokeDistance, strokeSpacing)
        : 1.0f;
    if (clayLikeTool && sculpt_stroke_state.changed &&
        strokeDistance < strokeSpacing * 0.35f) {
        return;
    }
    // Clay sample point: bias toward current hit (0.75) so the height reference
    // tracks the brush front, not the midpoint. This prevents the "stamp behind"
    // effect where the settle plane lags the brush and creates discrete mounds.
    const Vec3 claySamplePoint = (clayLikeTool && sculpt_stroke_state.changed)
        ? sanitizeVec3(lastStrokeHit + (hitPoint - lastStrokeHit) * 0.75f, hitPoint)
        : hitPoint;
    const float localRadius = estimateEditableLocalBrushRadius(inverseTransform, radiusWorld);
    const Vec3 localHitPoint = sanitizeVec3(inverseTransform.transform_point(hitPoint), Vec3(0.0f, 0.0f, 0.0f));
    const Vec3 localClaySamplePoint = sanitizeVec3(inverseTransform.transform_point(claySamplePoint), localHitPoint);
    const Vec3 localGrabStartPoint = sanitizeVec3(
        inverseTransform.transform_point(sanitizeVec3(sculpt_stroke_state.start_world_hit, hitPoint)),
        localHitPoint);
    const Vec3 localPrevHitPoint = sanitizeVec3(inverseTransform.transform_point(lastStrokeHit), localHitPoint);

    // Advanced brush footprint (shape / aspect / roundness / rotation / alpha). The SIMD
    // evaluator reads these off params.brushSettings; follow_stroke_angle adds the live stroke
    // direction (measured in the surface tangent frame) to the rotation so directional shapes
    // trail the cursor like in mesh paint. boundScale widens the gather + cull so non-circular
    // footprints aren't clipped to the radius circle. Circle@aspect1 ⇒ boundScale 1 (unchanged).
    Paint::BrushSettings effectiveBrush = sculpt_mode_state.brush;
    if (effectiveBrush.follow_stroke_angle) {
        const float su = strokeStepWorld.dot(tangentWorld);
        const float sv = strokeStepWorld.dot(bitangentWorld);
        if (su * su + sv * sv > 1e-12f) {
            effectiveBrush.alpha_rotation_degrees +=
                std::atan2(sv, su) * (180.0f / 3.14159265358979f);
        }
    }
    const float brushFootprintBoundScale = uiBrushFootprintBoundScale(effectiveBrush);
    const float brushCullRadiusWorld = radiusWorld * brushFootprintBoundScale;

    // Candidate gathering now prefers PBVH traversal for broad-phase culling,
    // but the actual brush solve remains vertex-based for consistent quality.
    // Capsule sweep: non-Grab tool'larda toplama küresini [prevHit, currentHit]
    // segmentini kuşatacak şekilde genişletip merkeziyetlendiriyoruz. Yarıçap
    // kapsüller (3R)'a kadar büyüyor — fare hızlı hareket ettiğinde aradaki
    // vertexleri tek pass ile yakalamamızı sağlar.
    std::vector<int> sculptCandidateVertexIdsStorage;
    Vec3 candidateCenterLocal;
    float candidateCollectionRadius;
    if (anchoredActive) {
        // Anchored: gather once around the FROZEN anchor at the slider radius; the set
        // is primed and frozen below, then driven absolutely for the whole stroke.
        candidateCenterLocal = localGrabStartPoint;
        candidateCollectionRadius = localRadius;
    } else if (activeTool == SculptBrushTool::SnakeHook) {
        // Snake Hook gathers along the CURRENT brush position + a capsule sweep of
        // this frame's motion, so fast strokes keep feeding new verts into the set.
        candidateCenterLocal = (localHitPoint + localPrevHitPoint) * 0.5f;
        const float halfSeg = (localHitPoint - localPrevHitPoint).length() * 0.5f;
        candidateCollectionRadius = localRadius + (std::min)(halfSeg, localRadius * 2.0f);
    } else if (isGrabFamilyTool(activeTool)) {
        candidateCenterLocal = localGrabStartPoint;
        candidateCollectionRadius = localRadius;
    } else {
        // Widen the additive gather so shaped/elongated footprints reach their corners.
        const float gatherLocalRadius = localRadius * brushFootprintBoundScale;
        candidateCenterLocal = (localClaySamplePoint + localPrevHitPoint) * 0.5f;
        const float halfSeg = (localClaySamplePoint - localPrevHitPoint).length() * 0.5f;
        const float cappedHalf = (std::min)(halfSeg, gatherLocalRadius * 2.0f);
        candidateCollectionRadius = gatherLocalRadius + cappedHalf;
    }
    // Grab/Elastic drive the whole stroke from the FIXED primed weight set (rebuilt
    // from grab_weights_by_vertex below), so once primed the spatial gather result is
    // discarded — skip it to avoid one PBVH query per sub-step on fast strokes (the
    // grab dab scheduler can emit up to 32 sub-steps/frame). SnakeHook still re-gathers
    // every frame to eat new geometry, and the prime frame itself still needs it.
    const bool absoluteGrabPrimed =
        ((activeTool == SculptBrushTool::Grab || activeTool == SculptBrushTool::ElasticDeform) &&
         !sculpt_stroke_state.grab_weights_by_vertex.empty()) ||
        (anchoredActive && sculpt_stroke_state.anchored_primed);
    if (!absoluteGrabPrimed) {
        sculptCandidateVertexIdsStorage = collectSculptCandidateVerticesWithPBVHFallback(
            sculpt_pbvh,
            editable_mesh_cache,
            candidateCenterLocal,
            candidateCollectionRadius);
    }
    if (activeTool == SculptBrushTool::Draw || activeTool == SculptBrushTool::Inflate ||
        activeTool == SculptBrushTool::Stamp) {
        const float largeBrushFactor = computeLargeBrushProjectionBlend(radiusWorld);
        if (largeBrushFactor > 0.08f) {
            const int expandRings = largeBrushFactor > 0.55f ? 2 : 1;
            sculptCandidateVertexIdsStorage = expandEditableCandidateVerticesByTopology(
                editable_mesh_cache,
                sculptCandidateVertexIdsStorage,
                expandRings);
        }
    }
    // Grab family seeds its footprint from a single spatial PBVH query. Because
    // grab freezes that set for the whole stroke (the spike/dent fix), any vertex
    // the sphere query misses stays missed — unlike the per-frame brushes, which
    // re-gather and self-heal. Missed interior verts leave the boundary triangles
    // between moved/unmoved verts stretched, which inverts their winding (the
    // black back-faces). Grow the seed by topology (connected one-ring rings) so
    // every connected vertex in the footprint is captured; primeGrabStrokeWeights'
    // planarDistance <= radiusWorld test still clips the actual footprint to the
    // brush radius, so this only recovers in-radius verts the sphere query dropped.
    if (isGrabFamilyTool(activeTool)) {
        // Grab/Elastic only consume this set on the prime frame (afterwards the set
        // is rebuilt from the frozen weights), so grow only then. SnakeHook re-reads
        // it every frame to eat new geometry, so it always benefits from the grow.
        const bool needsTopologyGrow = activeTool == SculptBrushTool::SnakeHook ||
                                       sculpt_stroke_state.grab_weights_by_vertex.empty();
        if (needsTopologyGrow) {
            sculptCandidateVertexIdsStorage = expandEditableCandidateVerticesByTopology(
                editable_mesh_cache,
                sculptCandidateVertexIdsStorage,
                2);
        }
    }
    telemetry_candidate_vertices = sculptCandidateVertexIdsStorage.size();
    if (isGrabFamilyTool(activeTool)) {
        if (activeTool == SculptBrushTool::SnakeHook) {
            // DYNAMIC prime: every frame add NEW verts under the (swept) brush to the
            // set so the hook keeps eating geometry and fast strokes don't skip verts.
            // The set is never cleared mid-stroke. Each new vert's start position is
            // offset back by the drag already applied (start = pos - localGrabDelta*
            // grabStrength*w) so the absolute grab solve leaves it in place on the
            // frame it joins (no jump), then it trails the cursor like the rest.
            // Grab-family follow ratio is FIXED (kSculptGrabFollowStrength), decoupled from the
    // brush Strength slider: displacement = cursorDelta * weight * follow. A constant
    // follow keeps per-sub-step displacement bounded on fast strokes so the topology
    // guard stops dropping/freezing verts; speed and slider no longer cause teleports.
    const float grabStrength = kSculptGrabFollowStrength;
            for (const int vid : sculptCandidateVertexIdsStorage) {
                if (vid < 0 || static_cast<size_t>(vid) >= editable_mesh_cache.vertices.size()) {
                    continue;
                }
                if (sculpt_stroke_state.grab_weights_by_vertex.count(vid)) {
                    continue; // already in the set
                }
                const Vec3 localPos = editable_mesh_cache.vertices[vid].local_position;
                const Vec3 worldPos = sanitizeVec3(transform.transform_point(localPos), hitPoint);
                const Vec3 toV = worldPos - hitPoint;
                const float h = toV.dot(hitNormalWorld);
                const float planarDist = (toV - hitNormalWorld * h).length();
                if (!std::isfinite(planarDist) || planarDist > radiusWorld) {
                    continue;
                }
                if (sculpt_mode_state.front_faces_only && h < -(radiusWorld * 0.12f)) {
                    continue;
                }
                const float w = grabFamilyWeight(
                    planarDist / radiusWorld, mesh_overlay_settings.proportional_falloff_type, false);
                if (w <= 1e-5f) {
                    continue;
                }
                sculpt_stroke_state.grab_weights_by_vertex[vid] = w;
                sculpt_stroke_state.grab_start_local_positions[vid] =
                    sanitizeVec3(localPos - localGrabDelta * (grabStrength * w), localPos);
            }
        }
        // Prime the grab set once, from the PBVH-gathered candidates at the stroke's
        // first frame.
        else if (sculpt_stroke_state.grab_weights_by_vertex.empty()) {
            primeGrabStrokeWeights(
                sculpt_stroke_state,
                editable_mesh_cache,
                transform,
                sculptCandidateVertexIdsStorage,
                sanitizeVec3(sculpt_stroke_state.start_world_hit, hitPoint),
                hitNormalWorld,
                radiusWorld,
                sculpt_mode_state.front_faces_only,
                mesh_overlay_settings.proportional_falloff_type,
                activeTool == SculptBrushTool::ElasticDeform);
            // (Grab honours the protection mask at commit time via the maskFactor
            // lerp in the commit loop, not by reweighting the primed set here.)
        }
        // Drive the whole stroke from the FIXED primed set, not a per-frame PBVH
        // re-gather. Grab recomputes each vertex as start_local + delta, so it does
        // not need fresh candidates — and as the pull deforms the mesh the PBVH leaf
        // bounds drift away from the start-centred query sphere, dropping primed
        // vertices mid-stroke. Those frozen verts become spikes/dents while their
        // still-gathered neighbours keep following the cursor (worse on dense meshes
        // / large pulls). Iterating the primed set guarantees every grabbed vertex
        // tracks the cursor for the entire stroke.
        sculptCandidateVertexIdsStorage.clear();
        sculptCandidateVertexIdsStorage.reserve(sculpt_stroke_state.grab_weights_by_vertex.size());
        for (const auto& [vid, weight] : sculpt_stroke_state.grab_weights_by_vertex) {
            sculptCandidateVertexIdsStorage.push_back(vid);
        }
        std::sort(sculptCandidateVertexIdsStorage.begin(), sculptCandidateVertexIdsStorage.end());
    }
    // Anchored: prime the frozen footprint set once (rest positions of every gathered
    // vertex), then drive the whole stroke from that fixed set — like grab, the centre
    // never moves so a per-frame re-gather would only drop verts as the imprint pushes
    // the surface out. Each frozen vert is rewritten absolutely from its rest position
    // every frame in the apply branch below, so the imprint resizes/returns cleanly.
    if (anchoredActive) {
        if (!sculpt_stroke_state.anchored_primed) {
            for (const int vid : sculptCandidateVertexIdsStorage) {
                if (vid < 0 || static_cast<size_t>(vid) >= editable_mesh_cache.vertices.size()) {
                    continue;
                }
                sculpt_stroke_state.grab_start_local_positions.try_emplace(
                    vid, editable_mesh_cache.vertices[vid].local_position);
            }
            sculpt_stroke_state.anchored_primed = true;
        }
        sculptCandidateVertexIdsStorage.clear();
        sculptCandidateVertexIdsStorage.reserve(sculpt_stroke_state.grab_start_local_positions.size());
        for (const auto& [vid, p] : sculpt_stroke_state.grab_start_local_positions) {
            (void)p;
            sculptCandidateVertexIdsStorage.push_back(vid);
        }
        std::sort(sculptCandidateVertexIdsStorage.begin(), sculptCandidateVertexIdsStorage.end());
    }
    const std::vector<int>& sculptCandidateVertexIds = sculptCandidateVertexIdsStorage;
    sculpt_control_graph.last_candidate_node_count = sculpt_pbvh.last_candidate_node_count;
    sculpt_control_graph.last_candidate_vertex_count = sculptCandidateVertexIds.size();

    // Bind the custom falloff LUT for this solve (no-op unless type == Custom and
    // the user has edited the curve). All brush/mask falloff reads see it.
    setActiveFalloffLut(mesh_overlay_settings.custom_falloff_lut.empty()
        ? nullptr : &mesh_overlay_settings.custom_falloff_lut);

    // Keep the protection mask matched to the active object + cache revision so
    // gating (and the Mask brush itself) read a buffer that aligns with the
    // current vertex-id space. Switching object or editing topology resets it.
    ensureSculptMaskSized(sculpt_mask_state, editable_mesh_cache, objectName);

    // --- Mask brush: paint the per-vertex protection weight ----------------
    // The mask moves no geometry, so it bypasses the whole commit / topology /
    // PBVH-refit / GPU-geometry-sync tail. It only updates the mask buffer and
    // bumps its version so the edit overlay re-uploads the tint flags.
    if (activeTool == SculptBrushTool::Mask) {
        const float maskFlow = sanitizeFiniteFloat(sculpt_mask_state.paint_strength, 0.5f, 0.0f, 1.0f);
        // Paint the mask around one brush center (local + world + surface normal).
        // Reused for the primary hit and every mirror plane so symmetry works.
        auto paintMaskAt = [&](const Vec3& localCenter, const Vec3& worldCenter,
                               const Vec3& worldNormal) -> bool {
            const std::vector<int> cands = collectSculptCandidateVerticesWithPBVHFallback(
                sculpt_pbvh, editable_mesh_cache, localCenter, localRadius);
            bool changedLocal = false;
            for (const int vid : cands) {
                if (vid < 0 || static_cast<size_t>(vid) >= editable_mesh_cache.vertex_positions.size()) {
                    continue;
                }
                const size_t vmid = static_cast<size_t>(vid);
                const Vec3 worldPos = sanitizeVec3(
                    transform.transform_point(editable_mesh_cache.vertex_positions[vmid]),
                    Vec3(0.0f, 0.0f, 0.0f));
                const Vec3 toVertex = worldPos - worldCenter;
                const float h = toVertex.dot(worldNormal);
                const Vec3 planarOffset = toVertex - worldNormal * h;
                const float planarDist = computeHybridBrushDistance(
                    toVertex, worldNormal, planarOffset.length(), radiusWorld);
                if (!std::isfinite(planarDist) || planarDist > radiusWorld) {
                    continue;
                }
                const float frontFacePenalty = computeFrontFaceBrushPenalty(
                    h, radiusWorld, sculpt_mode_state.front_faces_only);
                if (frontFacePenalty <= 1e-5f) {
                    continue;
                }
                float w = computeTerrainLikeBrushWeight(planarDist / radiusWorld, falloff);
                w *= applyFalloffCurve(
                    1.0f - saturateFloat(planarDist / radiusWorld),
                    mesh_overlay_settings.proportional_falloff_type);
                w *= frontFacePenalty;
                if (!std::isfinite(w) || w <= 1e-5f) {
                    continue;
                }
                // directionSign < 0 (Ctrl) erases the mask.
                // Per-dab deposit. Mask dabs are DISTANCE-gated (one per ~0.25·radius of
                // travel) — NOT one per frame — so scaling by dt was wrong: it tied each
                // dab's amount to framerate (~0.1·flow at 60fps, even less higher), so even
                // paint_strength=1 needed many passes to fill. Use a framerate-independent
                // per-dab amount that reaches full mask in ~1-2 passes at full strength
                // while still building a soft gradient at lower strengths.
                const float deposit = w * maskFlow * 0.5f * directionSign;
                float& mv = sculpt_mask_state.values[vmid];
                const float nv = std::clamp(mv + deposit, 0.0f, 1.0f);
                if (nv != mv) {
                    mv = nv;
                    changedLocal = true;
                }
            }
            return changedLocal;
        };

        bool maskChanged = paintMaskAt(localHitPoint, hitPoint, hitNormalWorld);

        // Mirror planes: paint the mirrored center(s) too, mirroring the surface
        // normal in object-local space exactly like the deformer mirror passes.
        if (sculpt_mode_state.mirror_x || sculpt_mode_state.mirror_y || sculpt_mode_state.mirror_z) {
            const Matrix4x4 normalMtx = inverseTransform.transpose();
            for (int mirrorBits = 1; mirrorBits < 8; ++mirrorBits) {
                const bool do_mx = (mirrorBits & 1) && sculpt_mode_state.mirror_x;
                const bool do_my = (mirrorBits & 2) && sculpt_mode_state.mirror_y;
                const bool do_mz = (mirrorBits & 4) && sculpt_mode_state.mirror_z;
                if (!do_mx && !do_my && !do_mz) {
                    continue;
                }
                Vec3 mirLocal = localHitPoint;
                if (do_mx) mirLocal.x = -mirLocal.x;
                if (do_my) mirLocal.y = -mirLocal.y;
                if (do_mz) mirLocal.z = -mirLocal.z;
                const Vec3 mirWorldCenter = sanitizeVec3(transform.transform_point(mirLocal), hitPoint);
                Vec3 localNormal = sanitizeVec3(inverseTransform.transform_vector(hitNormalWorld), hitNormalWorld);
                const float lnLen = localNormal.length();
                if (lnLen > 1e-8f) localNormal = localNormal / lnLen;
                if (do_mx) localNormal.x = -localNormal.x;
                if (do_my) localNormal.y = -localNormal.y;
                if (do_mz) localNormal.z = -localNormal.z;
                Vec3 mirWorldNormal = sanitizeVec3(normalMtx.transform_vector(localNormal), hitNormalWorld);
                const float mwnLen = mirWorldNormal.length();
                mirWorldNormal = mwnLen > 1e-8f ? mirWorldNormal / mwnLen : hitNormalWorld;
                if (paintMaskAt(mirLocal, mirWorldCenter, mirWorldNormal)) {
                    maskChanged = true;
                }
            }
        }

        // Radial symmetry: paint each rotated copy of the brush center too.
        if (sculpt_mode_state.radial_symmetry && sculpt_mode_state.radial_count >= 2) {
            const Matrix4x4 normalMtx = inverseTransform.transpose();
            const int radialAxis = std::clamp(sculpt_mode_state.radial_axis, 0, 2);
            const int radialCount = std::clamp(sculpt_mode_state.radial_count, 2, 64);
            const Vec3 localHitNormal = safeNormalizeVec3(
                inverseTransform.transform_vector(hitNormalWorld), hitNormalWorld);
            for (int k = 1; k < radialCount; ++k) {
                const float angle = (6.28318530718f * static_cast<float>(k)) /
                                    static_cast<float>(radialCount);
                const Vec3 radLocal = rotateLocalAroundAxis(localHitPoint, radialAxis, angle);
                const Vec3 radWorldCenter = sanitizeVec3(transform.transform_point(radLocal), hitPoint);
                const Vec3 radLocalNormal = rotateLocalAroundAxis(localHitNormal, radialAxis, angle);
                const Vec3 radWorldNormal = safeNormalizeVec3(
                    normalMtx.transform_vector(radLocalNormal), hitNormalWorld);
                if (paintMaskAt(radLocal, radWorldCenter, radWorldNormal)) {
                    maskChanged = true;
                }
            }
        }

        if (maskChanged) {
            sculpt_mask_state.has_any = true;
            ++sculpt_mask_state.version;
            // The protection mask only affects the raster edit overlay (which
            // re-syncs every frame off mask version), so no path-trace reset here.
        }
        sculpt_stroke_state.changed = true;
        sculpt_stroke_state.last_world_hit = hitPoint;
        return;
    }

    std::vector<int> strokeTouchedVertexIds = sculptCandidateVertexIds;
    strokeTouchedVertexIds.reserve(sculptCandidateVertexIds.size() * 4 + 32);

    // Per-vertex wetness feather: 1 in the brush core, smoothly ramping to 0 across the
    // outer band of the footprint. Injecting a FLAT wetness makes the brush boundary a hard
    // wet/dry cliff — fully-mobile clay against a frozen ring — and at high fluidity the flow
    // piles material into a sharp ridge along it (worst with Water, flow≈1, on a vertical
    // wall). Feathering the injected wetness turns that boundary into a gradient so the clay's
    // mobility tapers off instead of stopping at a wall. Uses planar (cylinder) distance from
    // the brush hit, matching the footprint-clip test the candidate gather already uses.
    // Brush centers (primary hit + each enabled mirror plane) in WORLD space, used by the
    // wetness feather. A deposited vertex can belong to ANY of these footprints (the mirror
    // side included), so the feather must be measured against the NEAREST center. Measuring
    // only the primary hit zeroed the feather for mirror-side verts (planarDist ≫ radius →
    // feather 0 → vInject < threshold → skipped), which silently stopped the mirror side from
    // ever getting wet — i.e. mirror "didn't support" the wet system.
    std::vector<std::pair<Vec3, Vec3>> brushWetCenters;
    brushWetCenters.emplace_back(hitPoint, hitNormalWorld);
    if (sculpt_mode_state.mirror_x || sculpt_mode_state.mirror_y || sculpt_mode_state.mirror_z) {
        const Matrix4x4 normalMtxW = inverseTransform.transpose();
        for (int mirrorBits = 1; mirrorBits < 8; ++mirrorBits) {
            const bool do_mx = (mirrorBits & 1) && sculpt_mode_state.mirror_x;
            const bool do_my = (mirrorBits & 2) && sculpt_mode_state.mirror_y;
            const bool do_mz = (mirrorBits & 4) && sculpt_mode_state.mirror_z;
            if (!do_mx && !do_my && !do_mz) {
                continue;
            }
            Vec3 localHit = sanitizeVec3(inverseTransform.transform_point(hitPoint), Vec3(0.0f, 0.0f, 0.0f));
            if (do_mx) localHit.x = -localHit.x;
            if (do_my) localHit.y = -localHit.y;
            if (do_mz) localHit.z = -localHit.z;
            const Vec3 mCenter = sanitizeVec3(transform.transform_point(localHit), hitPoint);
            Vec3 localNormal = sanitizeVec3(inverseTransform.transform_vector(hitNormalWorld), hitNormalWorld);
            const float lnLen = localNormal.length();
            if (lnLen > 1e-8f) localNormal = localNormal / lnLen;
            if (do_mx) localNormal.x = -localNormal.x;
            if (do_my) localNormal.y = -localNormal.y;
            if (do_mz) localNormal.z = -localNormal.z;
            Vec3 mNormal = sanitizeVec3(normalMtxW.transform_vector(localNormal), hitNormalWorld);
            const float mnLen = mNormal.length();
            mNormal = mnLen > 1e-8f ? mNormal / mnLen : hitNormalWorld;
            brushWetCenters.emplace_back(mCenter, mNormal);
        }
    }
    auto wetClayFeatherWeight = [&](int vid) -> float {
        if (vid < 0 || static_cast<size_t>(vid) >= editable_mesh_cache.vertices.size() ||
            radiusWorld <= 1e-6f) {
            return 1.0f;
        }
        const Vec3 worldPos = sanitizeVec3(
            transform.transform_point(editable_mesh_cache.vertices[static_cast<size_t>(vid)].local_position),
            hitPoint);
        // Max feather over all footprints → a vertex in ANY footprint's core stays fully wet,
        // only the rims feather. Fixes the mirror side being skipped.
        float best = 0.0f;
        for (const auto& c : brushWetCenters) {
            const Vec3 toV = worldPos - c.first;
            const float h = toV.dot(c.second);
            const float planarDist = (toV - c.second * h).length();
            if (!std::isfinite(planarDist)) {
                continue;
            }
            const float t = saturateFloat(planarDist / radiusWorld);
            const float s = saturateFloat((t - 0.55f) / 0.45f);
            best = (std::max)(best, 1.0f - s * s * (3.0f - 2.0f * s));
        }
        return best;
    };

    // Phase 3 — "Water" brush: paint WETNESS onto the footprint without depositing any
    // geometry, re-softening dried clay so settle/flow act on it again (like adding water
    // in pottery). Short-circuit before the deposit so no clay is added; per-frame
    // stepWetClayField does the rest. Keep the stroke bookkeeping current so the dab
    // scheduler keeps advancing along the stroke.
    if (sculpt_mode_state.wet_clay_enabled && sculpt_mode_state.wet_clay_water_only) {
        // Paint feathered wetness around ONE brush center (local + world + surface normal).
        // Factored out so the primary hit AND every mirror plane share it — exactly like the
        // Mask brush's paintMaskAt. (Water used to return before the mirror passes, so it never
        // mirrored; this restores symmetry for the wet field.) No geometry deposit → empty
        // anchors → settle re-softens. Feather so the rim doesn't leave a frozen cliff for the
        // (highly mobile) water to ridge against.
        auto wetPaintAt = [&](const Vec3& localCenter, const Vec3& worldCenter,
                              const Vec3& worldNormal) {
            const std::vector<int> cands = collectSculptCandidateVerticesWithPBVHFallback(
                sculpt_pbvh, editable_mesh_cache, localCenter, localRadius);
            if (cands.empty()) {
                return;
            }
            std::vector<int> ids;
            std::vector<float> feather;
            ids.reserve(cands.size());
            feather.reserve(cands.size());
            for (const int vid : cands) {
                if (vid < 0 || static_cast<size_t>(vid) >= editable_mesh_cache.vertex_positions.size()) {
                    continue;
                }
                const Vec3 worldPos = sanitizeVec3(
                    transform.transform_point(editable_mesh_cache.vertex_positions[static_cast<size_t>(vid)]),
                    worldCenter);
                const Vec3 toV = worldPos - worldCenter;
                const float h = toV.dot(worldNormal);
                const float planarDist = (toV - worldNormal * h).length();
                if (!std::isfinite(planarDist) || planarDist > radiusWorld) {
                    continue;
                }
                if (sculpt_mode_state.front_faces_only && h < -(radiusWorld * 0.12f)) {
                    continue;
                }
                const float t = saturateFloat(radiusWorld > 1e-6f ? planarDist / radiusWorld : 0.0f);
                const float s = saturateFloat((t - 0.55f) / 0.45f);
                ids.push_back(vid);
                feather.push_back(1.0f - s * s * (3.0f - 2.0f * s));
            }
            if (!ids.empty()) {
                depositWetClay(ids, {}, sculpt_mode_state.wet_clay_wetness, &feather);
            }
        };

        wetPaintAt(localHitPoint, hitPoint, hitNormalWorld);

        // Mirror planes: re-wet the mirrored center(s) too, mirroring the surface normal in
        // object-local space exactly like the deformer/Mask mirror passes.
        if (sculpt_mode_state.mirror_x || sculpt_mode_state.mirror_y || sculpt_mode_state.mirror_z) {
            const Matrix4x4 normalMtx = inverseTransform.transpose();
            for (int mirrorBits = 1; mirrorBits < 8; ++mirrorBits) {
                const bool do_mx = (mirrorBits & 1) && sculpt_mode_state.mirror_x;
                const bool do_my = (mirrorBits & 2) && sculpt_mode_state.mirror_y;
                const bool do_mz = (mirrorBits & 4) && sculpt_mode_state.mirror_z;
                if (!do_mx && !do_my && !do_mz) {
                    continue;
                }
                Vec3 localHit = sanitizeVec3(inverseTransform.transform_point(hitPoint), Vec3(0.0f, 0.0f, 0.0f));
                if (do_mx) localHit.x = -localHit.x;
                if (do_my) localHit.y = -localHit.y;
                if (do_mz) localHit.z = -localHit.z;
                const Vec3 mWorldCenter = sanitizeVec3(transform.transform_point(localHit), hitPoint);
                Vec3 localNormal = sanitizeVec3(inverseTransform.transform_vector(hitNormalWorld), hitNormalWorld);
                const float lnLen = localNormal.length();
                if (lnLen > 1e-8f) localNormal = localNormal / lnLen;
                if (do_mx) localNormal.x = -localNormal.x;
                if (do_my) localNormal.y = -localNormal.y;
                if (do_mz) localNormal.z = -localNormal.z;
                Vec3 mWorldNormal = sanitizeVec3(normalMtx.transform_vector(localNormal), hitNormalWorld);
                const float mwnLen = mWorldNormal.length();
                mWorldNormal = mwnLen > 1e-8f ? mWorldNormal / mwnLen : hitNormalWorld;
                wetPaintAt(localHit, mWorldCenter, mWorldNormal);
            }
        }

        sculpt_stroke_state.changed = true;
        sculpt_stroke_state.last_world_hit = hitPoint;
        return;
    }

    std::vector<std::shared_ptr<Triangle>> touchedTriangles;
    touchedTriangles.reserve(sculptCandidateVertexIds.size() * 2 + 32);
    beginEditableTriangleTouchPass(editable_mesh_cache);

    bool changed = false;
    // PERF: sculpt_updated_local_positions is a PERSISTENT member kept in lockstep
    // with the cache — it equals cache.local_position for every vertex NOT touched
    // this frame (the commit loop writes both the cache and this buffer for touched
    // verts, untouched verts never change, and the post-commit reset below restores
    // any polish-only writes). So a full O(N) sync is needed only when the buffer is
    // stale: size mismatch, or the stroke's first solve frame (the .clear() at
    // stroke begin forces that, which also covers undo / external cache edits between
    // strokes). Per drag-frame we only re-seed the candidate verts from the cache.
    // This removes the two full-mesh Vec3 copies that previously ran on EVERY
    // mouse-move frame — the dominant per-frame cost on dense meshes (O(N) -> O(touched)).
    //
    // The frame-start "snapshot" the brush solve reads is the cache itself: the cache
    // is not mutated until the commit loop, so during the read phase
    // cache.local_position == frame-start. Passing kEmptySnapshot makes
    // resolveEditableSnapshotLocalPosition fall back to the cache — bit-identical to
    // the old per-frame snapshot copy, zero allocation. (Direct reads below read the
    // cache for the same reason.)
    static const std::vector<Vec3> kEmptySnapshot;
    const size_t sculptVertexCount = editable_mesh_cache.vertex_positions.size();
    if (sculpt_updated_local_positions.size() != sculptVertexCount) {
        sculpt_updated_local_positions.resize(sculptVertexCount);
        for (size_t i = 0; i < sculptVertexCount; ++i) {
            sculpt_updated_local_positions[i] = editable_mesh_cache.vertex_positions[i];
        }
    }
    for (const int vid : sculptCandidateVertexIds) {
        if (vid >= 0 && static_cast<size_t>(vid) < sculptVertexCount) {
            sculpt_updated_local_positions[static_cast<size_t>(vid)] =
                editable_mesh_cache.vertex_positions[static_cast<size_t>(vid)];
        }
    }

    const Vec3 planePoint = hitPoint;
    // Capsule sweep'in segment ucu: bu frame'in dab merkezi (planePoint) ile bir
    // önceki dab merkezi arasındaki boşluğu kapatıyoruz. Stroke ilk frame'inde
    // last_world_hit == hitPoint olduğu için segment dejenere — eski davranış.
    const Vec3 prevPlanePoint = sanitizeVec3(sculpt_stroke_state.last_world_hit, hitPoint);
    std::atomic<bool> basePassChanged{ false };
    if (anchoredActive) {
        // Absolute stamp: pos = rest + anchorNormal * (depth * footprintWeight). The
        // footprint weight carries the loaded alpha/mask texture (rotated by the drag),
        // and writing from the FROZEN rest position every frame makes the imprint
        // resize/rotate live and never accumulate. radiusWorld normalises the (u,v) the
        // mask is sampled at, measured in the frozen anchor tangent frame.
        Paint::BrushSettings stampBrush = effectiveBrush;
        stampBrush.alpha_rotation_degrees =
            effectiveBrush.alpha_rotation_degrees + anchoredRotationDeg;
        const Vec3 aC = sculpt_stroke_state.anchor_world;
        const Vec3 aN = sculpt_stroke_state.anchor_normal;
        const Vec3 aT = sculpt_stroke_state.anchor_tangent;
        const Vec3 aB = sculpt_stroke_state.anchor_bitangent;
        const float invR = (radiusWorld > 1e-6f) ? (1.0f / radiusWorld) : 0.0f;
        // Depth = world drag length, scaled by the Strength slider so it has a lever.
        const float depthSigned = anchoredDepth * brushStrength * directionSign;
        forEachEditableCandidate(sculptCandidateVertexIds, [&](int vertexIdInt) {
            if (vertexIdInt < 0) {
                return;
            }
            const size_t vertexId = static_cast<size_t>(vertexIdInt);
            const auto startIt = sculpt_stroke_state.grab_start_local_positions.find(vertexIdInt);
            if (startIt == sculpt_stroke_state.grab_start_local_positions.end()) {
                return;
            }
            const Vec3& startLocal = startIt->second;
            const Vec3 startWorld = sanitizeVec3(transform.transform_point(startLocal), aC);
            const Vec3 toV = startWorld - aC;
            const float u = toV.dot(aT) * invR;
            const float v = toV.dot(aB) * invR;
            const float weight = uiSampleBrushFootprintWeight(stampBrush, u, v);
            const Vec3 worldOffset = aN * (depthSigned * weight);
            const Vec3 localOffset = sanitizeVec3(
                inverseTransform.transform_vector(worldOffset), Vec3(0.0f, 0.0f, 0.0f));
            sculpt_updated_local_positions[vertexId] =
                sanitizeVec3(startLocal + localOffset, startLocal);
            basePassChanged.store(true, std::memory_order_relaxed);
        });
    } else if (isGrabFamilyTool(activeTool)) {
        forEachEditableCandidate(sculptCandidateVertexIds, [&](int vertexIdInt) {
            if (vertexIdInt < 0) {
                return;
            }
            const size_t vertexId = static_cast<size_t>(vertexIdInt);
            if (applyGrabSculptCandidate(
                    sculpt_stroke_state,
                    brushStrength,
                    localGrabDelta,
                    vertexIdInt,
                    sculpt_updated_local_positions[vertexId])) {
                // Snake Hook narrows the dragged region into a hook tip: pull each
                // vertex laterally toward the current brush center (perpendicular to
                // the surface normal), scaled by its grab weight. Grab/Elastic skip this.
                if (activeTool == SculptBrushTool::SnakeHook) {
                    const Vec3 vp = sculpt_updated_local_positions[vertexId];
                    const Vec3 vpWorld = sanitizeVec3(transform.transform_point(vp), vp);
                    const Vec3 toCenter = planeHit - vpWorld;
                    const Vec3 lateral = toCenter - hitNormalWorld * toCenter.dot(hitNormalWorld);
                    const float lateralLen = lateral.length();
                    if (lateralLen > 1e-6f) {
                        const auto wIt = sculpt_stroke_state.grab_weights_by_vertex.find(vertexIdInt);
                        const float wgt = (wIt != sculpt_stroke_state.grab_weights_by_vertex.end())
                            ? wIt->second : 0.0f;
                        // Stronger lateral pull toward the tip + a lift along the normal so
                        // the dragged region curls into a proper hook instead of reading as
                        // a plain grab. Inverse-weighted so the rim (low weight) pinches more
                        // than the center, narrowing the tail.
                        const float tipBias = 0.22f + 0.18f * (1.0f - wgt);
                        const Vec3 pinchWorld = (lateral / lateralLen) * (lateralLen * tipBias * wgt) +
                            hitNormalWorld * (lateralLen * 0.10f * wgt);
                        const Vec3 pinchLocal = sanitizeVec3(
                            inverseTransform.transform_vector(pinchWorld), Vec3(0.0f, 0.0f, 0.0f));
                        sculpt_updated_local_positions[vertexId] =
                            sanitizeVec3(vp + pinchLocal, vp);
                    }
                }
                basePassChanged.store(true, std::memory_order_relaxed);
            }
        });
    } else {
        SIMDSculptParams params;
        params.center = planePoint;
        params.prevCenter = prevPlanePoint;
        params.normal = hitNormalWorld;
        params.tangent = tangentWorld;
        params.bitangent = bitangentWorld;
        params.radius = radiusWorld;
        params.cullRadius = brushCullRadiusWorld;
        params.falloff = falloff;
        params.strokeDistance = strokeDistance;
        params.strokeSpacing = strokeSpacing;
        params.dt = dt;
        params.strokeChanged = sculpt_stroke_state.changed;
        params.frontFacesOnly = sculpt_mode_state.front_faces_only;
        params.falloffType = mesh_overlay_settings.proportional_falloff_type;
        params.activeTool = activeTool;
        params.useMask = true;
        params.brushSettings = effectiveBrush;

        std::vector<EvaluatedSculptVertex> activeVerts = evaluateActiveSculptVerticesSIMD(
            sculptCandidateVertexIds,
            editable_mesh_cache,
            transform,
            params);

        std::vector<int> activeIndices(activeVerts.size());
        for (size_t i = 0; i < activeVerts.size(); ++i) {
            activeIndices[i] = static_cast<int>(i);
        }

        forEachEditableCandidate(activeIndices, [&](int index) {
            const auto& ev = activeVerts[index];
            const size_t vertexId = static_cast<size_t>(ev.id);

            SculptBrushSample sample;
            sample.local_position = editable_mesh_cache.vertex_positions[vertexId];
            sample.world_position = ev.worldPos;
            sample.height_from_plane = ev.height;
            sample.planar_distance = ev.planarDist;
            sample.weight = ev.weight;
            sample.clay_drag_factor = ev.clayDragFactor;

            const Vec3 segDir = planePoint - prevPlanePoint;
            const float segLenSq = segDir.dot(segDir);
            Vec3 closestPlanePoint;
            if (segLenSq > 1e-10f) {
                const float t = std::clamp((ev.worldPos - prevPlanePoint).dot(segDir) / segLenSq, 0.0f, 1.0f);
                closestPlanePoint = prevPlanePoint + segDir * t;
            } else {
                closestPlanePoint = planePoint;
            }
            sample.planar_offset = ev.worldPos - closestPlanePoint - hitNormalWorld * ev.height;

            if (applyNonGrabSculptCandidate(
                    activeTool,
                    editable_mesh_cache,
                    sculpt_control_graph,
                    sculpt_stroke_state,
                    kEmptySnapshot,
                    transform,
                    inverseTransform,
                    planePoint,
                    claySamplePoint,
                    hitNormalWorld,
                    strokeTangentWorld,
                    strokeBitangentWorld,
                    strokeStepWorld,
                    brushStrength,
                    clayBrushStrength,
                    clayRadiusCompensation,
                    normalStrength,
                    directionSign,
                    radiusWorld,
                    dt,
                    localRadius,
                    strokeAdvanceFactor,
                    ev.id,
                    sample,
                    sculpt_updated_local_positions[vertexId])) {
                basePassChanged.store(true, std::memory_order_relaxed);
            }
        });
    }
    changed = changed || basePassChanged.load(std::memory_order_relaxed);

    // Keep the old GPU sculpt experiment behind a hard-off gate for now.
    // The current implementation still computes brush behavior on CPU and then
    // adds full-buffer upload/dispatch/readback overhead, which makes it a poor
    // fit for the interactive sculpt path.
    constexpr bool kEnableExperimentalGpuSculpt = false;
    if (kEnableExperimentalGpuSculpt && sculpt_mode_state.use_gpu) {
        auto* vkAdapter = dynamic_cast<Backend::VulkanBackendAdapter*>(getMeshOverlayRenderBackend(ctx));
        if (vkAdapter) {
            VulkanRT::VulkanDevice* dev = vkAdapter->getVulkanDevice();
            if (dev && dev->isInitialized() && dev->m_sculptPipeline != VK_NULL_HANDLE) {
                const size_t vertexCount = editable_mesh_cache.vertices.size();
                if (vertexCount > 0) {
                    std::vector<float> posData(vertexCount * 3);
                    std::vector<float> nrmData(vertexCount * 3);
                    std::vector<float> wData(vertexCount);

                    for (size_t i = 0; i < vertexCount; ++i) {
                        const Vec3 lp = editable_mesh_cache.vertices[i].local_position;
                        posData[i * 3 + 0] = lp.x;
                        posData[i * 3 + 1] = lp.y;
                        posData[i * 3 + 2] = lp.z;
                        // simple normal fallback: use first ref's vertex normal if available
                        Vec3 n(0.0f, 1.0f, 0.0f);
                        if (!editable_mesh_cache.vertices[i].refs.empty() && editable_mesh_cache.refTri(editable_mesh_cache.vertices[i].refs[0])) {
                            const auto& ref = editable_mesh_cache.vertices[i].refs[0];
                            n = editable_mesh_cache.refTri(ref)->vertices[ref.corner].normal;
                        }
                        nrmData[i * 3 + 0] = n.x;
                        nrmData[i * 3 + 1] = n.y;
                        nrmData[i * 3 + 2] = n.z;
                        // weight = 1.0 inside localRadius, 0 otherwise (simple test weight)
                        const Vec3 worldPos = sanitizeVec3(transform.transform_point(editable_mesh_cache.vertices[i].local_position), Vec3(0,0,0));
                        const float dist = (worldPos - planePoint).length();
                        wData[i] = (dist <= radiusWorld) ? 1.0f : 0.0f;
                    }

                    // Create GPU buffers and upload
                    VulkanRT::BufferCreateInfo bci{};
                    bci.size = static_cast<uint64_t>(posData.size() * sizeof(float));
                    bci.usage = VulkanRT::BufferUsage::STORAGE | VulkanRT::BufferUsage::TRANSFER_DST | VulkanRT::BufferUsage::TRANSFER_SRC;
                    bci.location = VulkanRT::MemoryLocation::GPU_ONLY;
                    VulkanRT::BufferHandle posBuf = dev->createBuffer(bci);
                    dev->uploadBuffer(posBuf, posData.data(), bci.size, 0);

                    bci.size = static_cast<uint64_t>(nrmData.size() * sizeof(float));
                    VulkanRT::BufferHandle nrmBuf = dev->createBuffer(bci);
                    dev->uploadBuffer(nrmBuf, nrmData.data(), bci.size, 0);

                    bci.size = static_cast<uint64_t>(wData.size() * sizeof(float));
                    VulkanRT::BufferHandle wBuf = dev->createBuffer(bci);
                    dev->uploadBuffer(wBuf, wData.data(), bci.size, 0);

                    // Push constants struct: center.xyz, radius, strength, dt
                    struct PC { float cx, cy, cz, radius, strength, dt; } pc;
                    pc.cx = localHitPoint.x; pc.cy = localHitPoint.y; pc.cz = localHitPoint.z;
                    pc.radius = localRadius;
                    pc.strength = brushStrength * directionSign;
                    pc.dt = dt;

                    dev->dispatchSculpt(posBuf, nrmBuf, wBuf, static_cast<uint32_t>(vertexCount), &pc, sizeof(pc));

                    // Read back positions
                    std::vector<float> newPos(posData.size());
                    dev->downloadBuffer(posBuf, newPos.data(), static_cast<uint64_t>(newPos.size() * sizeof(float)), 0);

                    // Destroy temporary buffers
                    dev->destroyBuffer(posBuf);
                    dev->destroyBuffer(nrmBuf);
                    dev->destroyBuffer(wBuf);

                    // Apply results into sculpt_updated_local_positions for touched vertices
                    for (size_t i = 0; i < vertexCount; ++i) {
                        sculpt_updated_local_positions[i] = Vec3(newPos[i * 3 + 0], newPos[i * 3 + 1], newPos[i * 3 + 2]);
                    }

                    changed = true;
                    sculpt_stroke_state.changed = true;
                }
            }
        }
    }

    // Mirror passes: repeat the sculpt in object-local mirrored space for each enabled axis.
    // (Anchored stamp is a single fixed-centre imprint — mirror/radial are skipped for it.)
    if (!anchoredActive &&
        (sculpt_mode_state.mirror_x || sculpt_mode_state.mirror_y || sculpt_mode_state.mirror_z)) {
        // Pre-compute normal matrix (M^-1)^T to convert local normals to world.
        const Matrix4x4 normalMtx = inverseTransform.transpose();

        for (int mirrorBits = 1; mirrorBits < 8; ++mirrorBits) {
            const bool do_mx = (mirrorBits & 1) && sculpt_mode_state.mirror_x;
            const bool do_my = (mirrorBits & 2) && sculpt_mode_state.mirror_y;
            const bool do_mz = (mirrorBits & 4) && sculpt_mode_state.mirror_z;
            if (!do_mx && !do_my && !do_mz) {
                continue;
            }

            // Mirror the brush hit center in local space, then back to world.
            Vec3 localHit = sanitizeVec3(inverseTransform.transform_point(hitPoint), Vec3(0.0f, 0.0f, 0.0f));
            if (do_mx) localHit.x = -localHit.x;
            if (do_my) localHit.y = -localHit.y;
            if (do_mz) localHit.z = -localHit.z;
            const Vec3 mirWorldCenter = sanitizeVec3(transform.transform_point(localHit), hitPoint);
            const Vec3 mirLocalCenter = localHit;

            // Capsule sweep'in mirror karşılığı için bir önceki dab merkezini de
            // aynı eksenlerde aynalıyoruz; segment [mirPrevCenter -> mirWorldCenter]
            // olur. Stroke ilk frame'inde lastStrokeHit == hitPoint olduğu için
            // segment dejenere — eski davranış korunur.
            Vec3 localMirPrevHit = sanitizeVec3(
                inverseTransform.transform_point(lastStrokeHit),
                Vec3(0.0f, 0.0f, 0.0f));
            if (do_mx) localMirPrevHit.x = -localMirPrevHit.x;
            if (do_my) localMirPrevHit.y = -localMirPrevHit.y;
            if (do_mz) localMirPrevHit.z = -localMirPrevHit.z;
            const Vec3 mirWorldPrevCenter = sanitizeVec3(
                transform.transform_point(localMirPrevHit),
                mirWorldCenter);

            // Mirror the stroke normal in local space.
            Vec3 localNormal = sanitizeVec3(inverseTransform.transform_vector(hitNormalWorld), hitNormalWorld);
            const float lnLen = localNormal.length();
            if (lnLen > 1e-8f) localNormal = localNormal / lnLen;
            if (do_mx) localNormal.x = -localNormal.x;
            if (do_my) localNormal.y = -localNormal.y;
            if (do_mz) localNormal.z = -localNormal.z;
            Vec3 mirWorldNormal = sanitizeVec3(normalMtx.transform_vector(localNormal), hitNormalWorld);
            const float mwnLen = mirWorldNormal.length();
            mirWorldNormal = mwnLen > 1e-8f ? mirWorldNormal / mwnLen : hitNormalWorld;

            // Mirror the brush tangent frame the SAME way (local axis flip) so directional
            // shapes / alpha masks become a true mirror image. Because the vertex offset,
            // tangent and bitangent are all mirrored consistently, a mirror vertex's (nx,ny)
            // equals its primary counterpart's → the SAME brushSettings yield the mirror image,
            // no rotation flip needed. This is what lets the mirror SIMD pass apply the shape
            // for free (just turn its footprint mask on below).
            auto mirrorBrushAxis = [&](const Vec3& worldVec) -> Vec3 {
                Vec3 lv = sanitizeVec3(inverseTransform.transform_vector(worldVec), worldVec);
                const float l = lv.length();
                if (l > 1e-8f) lv = lv / l;
                if (do_mx) lv.x = -lv.x;
                if (do_my) lv.y = -lv.y;
                if (do_mz) lv.z = -lv.z;
                Vec3 wv = sanitizeVec3(normalMtx.transform_vector(lv), worldVec);
                const float wl = wv.length();
                return wl > 1e-8f ? wv / wl : worldVec;
            };
            const Vec3 mirBrushTangent = mirrorBrushAxis(tangentWorld);
            const Vec3 mirBrushBitangent = mirrorBrushAxis(bitangentWorld);

            // Mirror drag delta for Grab (flip each axis in local, then back to world).
            Vec3 mirWorldDrag = worldDragDelta;
            if (isGrabFamilyTool(activeTool)) {
                Vec3 localDrag = sanitizeVec3(inverseTransform.transform_vector(worldDragDelta), Vec3(0.0f, 0.0f, 0.0f));
                if (do_mx) localDrag.x = -localDrag.x;
                if (do_my) localDrag.y = -localDrag.y;
                if (do_mz) localDrag.z = -localDrag.z;
                mirWorldDrag = sanitizeVec3(transform.transform_vector(localDrag), Vec3(0.0f, 0.0f, 0.0f));
            }

            // Grab: mirror start hit and compute weights on-the-fly.
            if (isGrabFamilyTool(activeTool)) {
                Vec3 localStartHit = sanitizeVec3(
                    inverseTransform.transform_point(sanitizeVec3(sculpt_stroke_state.start_world_hit, hitPoint)),
                    Vec3(0.0f, 0.0f, 0.0f));
                if (do_mx) localStartHit.x = -localStartHit.x;
                if (do_my) localStartHit.y = -localStartHit.y;
                if (do_mz) localStartHit.z = -localStartHit.z;
                const Vec3 mirWorldStartHit = sanitizeVec3(transform.transform_point(localStartHit), hitPoint);
                // Drive the mirror from a FIXED per-combo candidate set captured on the
                // stroke's first frame — same drift-free approach as the main grab. A
                // per-frame PBVH re-gather at the start-centred sphere would drop pulled
                // mirror verts mid-stroke (they freeze into spikes/dents). Also capture
                // each mirror vert's stroke-start local position: grab_start_local_positions
                // is primed (primeGrabStrokeWeights) only around the MAIN hit, so mirror
                // verts on the opposite side of the symmetry plane were never in it and the
                // solve below early-returned → the mirror moved nothing. On frame 1 the cache
                // is still at the start pose (commit runs later), so this records the correct
                // grab origin. Done serially, BEFORE the parallel solve, so the shared maps
                // are never written from worker threads.
                auto& mirrorSet = sculpt_stroke_state.grab_mirror_candidate_sets[mirrorBits];
                if (mirrorSet.empty()) {
                    mirrorSet = collectSculptCandidateVerticesWithPBVHFallback(
                        sculpt_pbvh,
                        editable_mesh_cache,
                        localStartHit,
                        localRadius);
                    // Same topology grow as the main grab seed: close the spatial
                    // sphere query's gaps so the frozen mirror set is complete (no
                    // missed verts → no inverted boundary triangles on the mirror side).
                    mirrorSet = expandEditableCandidateVerticesByTopology(
                        editable_mesh_cache, mirrorSet, 2);
                    for (const int vid : mirrorSet) {
                        if (vid >= 0 && static_cast<size_t>(vid) < editable_mesh_cache.vertex_positions.size()) {
                            sculpt_stroke_state.grab_start_local_positions.try_emplace(
                                vid, editable_mesh_cache.vertex_positions[vid]);
                        }
                    }
                }
                const std::vector<int>& mirrorGrabCandidateVertexIds = mirrorSet;
                // Re-seed the persistent buffer for this combo's mirror candidates each frame.
                for (const int vid : mirrorGrabCandidateVertexIds) {
                    if (vid >= 0 && static_cast<size_t>(vid) < editable_mesh_cache.vertex_positions.size()) {
                        sculpt_updated_local_positions[vid] = editable_mesh_cache.vertex_positions[vid];
                    }
                }
                strokeTouchedVertexIds.insert(
                    strokeTouchedVertexIds.end(),
                    mirrorGrabCandidateVertexIds.begin(),
                    mirrorGrabCandidateVertexIds.end());

                std::atomic<bool> mirrorGrabChanged{ false };
                forEachEditableCandidate(mirrorGrabCandidateVertexIds, [&](int vertexIdInt) {
                    if (vertexIdInt < 0) {
                        return;
                    }
                    const size_t vertexId = static_cast<size_t>(vertexIdInt);
                    auto startIt = sculpt_stroke_state.grab_start_local_positions.find(vertexIdInt);
                    if (startIt == sculpt_stroke_state.grab_start_local_positions.end()) {
                        return;
                    }
                    const Vec3& startLocalPos = startIt->second;
                    const Vec3 startWorldPos = sanitizeVec3(
                        transform.transform_point(startLocalPos),
                        mirWorldStartHit);
                    const Vec3 toVertex = startWorldPos - mirWorldStartHit;
                    const float h = toVertex.dot(mirWorldNormal);
                    const float planarDist = (toVertex - mirWorldNormal * h).length();
                    if (!std::isfinite(h) || !std::isfinite(planarDist) || planarDist > radiusWorld) {
                        return;
                    }
                    if (sculpt_mode_state.front_faces_only && h < -(radiusWorld * 0.12f)) {
                        return;
                    }
                    const float w = grabFamilyWeight(
                        planarDist / radiusWorld,
                        mesh_overlay_settings.proportional_falloff_type,
                        activeTool == SculptBrushTool::ElasticDeform);
                    if (w <= 1e-5f) {
                        return;
                    }
                    // Grab-family follow ratio is FIXED (kSculptGrabFollowStrength), decoupled from the
    // brush Strength slider: displacement = cursorDelta * weight * follow. A constant
    // follow keeps per-sub-step displacement bounded on fast strokes so the topology
    // guard stops dropping/freezing verts; speed and slider no longer cause teleports.
    const float grabStrength = kSculptGrabFollowStrength;
                    Vec3 wd = mirWorldDrag * w * grabStrength;
                    const float wdLen = wd.length();
                    if (!std::isfinite(wdLen)) {
                        return;
                    }
                    if (wdLen > radiusWorld * 3.0f) {
                        wd = wd * (radiusWorld * 3.0f / wdLen);
                    }
                    const Vec3 ld = sanitizeVec3(inverseTransform.transform_vector(wd), Vec3(0.0f, 0.0f, 0.0f));
                    const float ldLenSq = ld.length_squared();
                    if (!std::isfinite(ldLenSq) || ldLenSq <= 1e-14f) {
                        return;
                    }
                    sculpt_updated_local_positions[vertexId] = sanitizeVec3(
                        startLocalPos + ld,
                        startLocalPos);
                    mirrorGrabChanged.store(true, std::memory_order_relaxed);
                });
                changed = changed || mirrorGrabChanged.load(std::memory_order_relaxed);
                continue; // next mirror combination
            }

            // Non-Grab tools mirror pass.
            const Vec3 mirrorClaySampleCenter = (clayLikeTool && sculpt_stroke_state.changed)
                ? [&]() {
                    Vec3 localPrevHit = sanitizeVec3(inverseTransform.transform_point(lastStrokeHit), Vec3(0.0f, 0.0f, 0.0f));
                    if (do_mx) localPrevHit.x = -localPrevHit.x;
                    if (do_my) localPrevHit.y = -localPrevHit.y;
                    if (do_mz) localPrevHit.z = -localPrevHit.z;
                    return sanitizeVec3(localPrevHit + (mirLocalCenter - localPrevHit) * 0.75f, mirLocalCenter);
                }()
                : mirLocalCenter;
            const std::vector<int> mirrorCandidateVertexIds = collectSculptCandidateVerticesWithPBVHFallback(
                sculpt_pbvh,
                editable_mesh_cache,
                mirrorClaySampleCenter,
                localRadius * brushFootprintBoundScale);
            // Re-seed persistent buffer for mirror candidates from the cache (frame-start).
            for (const int vid : mirrorCandidateVertexIds) {
                if (vid >= 0 && static_cast<size_t>(vid) < editable_mesh_cache.vertex_positions.size()) {
                    sculpt_updated_local_positions[static_cast<size_t>(vid)] =
                        editable_mesh_cache.vertex_positions[static_cast<size_t>(vid)];
                }
            }
            strokeTouchedVertexIds.insert(
                strokeTouchedVertexIds.end(),
                mirrorCandidateVertexIds.begin(),
                mirrorCandidateVertexIds.end());
            std::atomic<bool> mirrorPassChanged{ false };
            SIMDSculptParams params;
            params.center = mirWorldCenter;
            params.prevCenter = mirWorldPrevCenter;
            params.normal = mirWorldNormal;
            params.tangent = mirBrushTangent;
            params.bitangent = mirBrushBitangent;
            params.radius = radiusWorld;
            params.cullRadius = brushCullRadiusWorld;
            params.falloff = falloff;
            params.strokeDistance = strokeDistance;
            params.strokeSpacing = strokeSpacing;
            params.dt = dt;
            params.strokeChanged = sculpt_stroke_state.changed;
            params.frontFacesOnly = sculpt_mode_state.front_faces_only;
            params.falloffType = mesh_overlay_settings.proportional_falloff_type;
            params.activeTool = activeTool;
            // Apply the brush footprint (shape/aspect/roundness/rotation/alpha) on the mirror
            // too. Free: the mirror SIMD pass already runs; this just enables the scalar-tail
            // mask. The mirrored tangent frame above makes it a correct mirror image.
            params.useMask = true;
            params.brushSettings = effectiveBrush;

            std::vector<EvaluatedSculptVertex> activeMirrorVerts = evaluateActiveSculptVerticesSIMD(
                mirrorCandidateVertexIds,
                editable_mesh_cache,
                transform,
                params);

            std::vector<int> activeIndices(activeMirrorVerts.size());
            for (size_t i = 0; i < activeMirrorVerts.size(); ++i) {
                activeIndices[i] = static_cast<int>(i);
            }

            forEachEditableCandidate(activeIndices, [&](int index) {
                const auto& ev = activeMirrorVerts[index];
                const size_t vertexId = static_cast<size_t>(ev.id);
                EditableVertex& vertex = editable_mesh_cache.vertices[vertexId];
                const Vec3 snapshotLocalPosition = resolveEditableSnapshotLocalPosition(
                    editable_mesh_cache,
                    kEmptySnapshot,
                    vertexId);
                const Vec3 worldPos = ev.worldPos;
                const float h = ev.height;
                const float planarDist = ev.planarDist;
                const float w = ev.weight;
                const float mirDragFactor = ev.clayDragFactor;

                const Vec3 mirSegDir = mirWorldCenter - mirWorldPrevCenter;
                const float mirSegLenSq = mirSegDir.dot(mirSegDir);
                Vec3 mirClosestCenter;
                if (mirSegLenSq > 1e-10f) {
                    const float mirT = std::clamp((worldPos - mirWorldPrevCenter).dot(mirSegDir) / mirSegLenSq, 0.0f, 1.0f);
                    mirClosestCenter = mirWorldPrevCenter + mirSegDir * mirT;
                } else {
                    mirClosestCenter = mirWorldCenter;
                }
                const Vec3 toVertex = worldPos - mirClosestCenter;
                const Vec3 planarOffset = toVertex - mirWorldNormal * h;

                Vec3 wd(0.0f, 0.0f, 0.0f);
                switch (activeTool) {
                case SculptBrushTool::Inflate:
                    wd = mirWorldNormal * (radiusWorld * 0.22f * brushStrength * dt * w * (1.0f + normalStrength) * directionSign);
                    break;
                case SculptBrushTool::Stamp:
                case SculptBrushTool::Draw: {
                    Vec3 vn = mirWorldNormal;
                    if (!vertex.refs.empty() && editable_mesh_cache.refTri(vertex.refs[0])) {
                        const Vec3 n = editable_mesh_cache.refTri(vertex.refs[0])->vertices[vertex.refs[0].corner].normal;
                        if (n.length_squared() > 1e-8f) vn = safeNormalizeVec3(n, mirWorldNormal);
                    }
                    const float centerBias = std::clamp(w, 0.0f, 1.0f);
                    const float drawStrokeNormalMix =
                        std::clamp(0.68f + 0.22f * computeLargeBrushSurfaceFactor(radiusWorld) + 0.08f * centerBias, 0.0f, 0.94f);
                    const Vec3 drawDirection = safeNormalizeVec3(
                        vn * (1.0f - drawStrokeNormalMix) + mirWorldNormal * drawStrokeNormalMix,
                        mirWorldNormal);
                    wd = drawDirection * (radiusWorld * 0.22f * brushStrength * dt * w * (1.0f + normalStrength) * directionSign);
                    break;
                }
                case SculptBrushTool::Layer: {
                    if (vertexId >= sculpt_stroke_state.layer_accum.size()) {
                        return;
                    }
                    float& layerAccum = sculpt_stroke_state.layer_accum[vertexId];
                    const float previousAccum = layerAccum;
                    const float targetLayer = radiusWorld * 0.22f * clayRadiusCompensation * w * (0.9f + 0.35f * normalStrength);
                    const float deposit = radiusWorld * 0.09f * clayRadiusCompensation * clayBrushStrength * dt * w * directionSign;
                    if (directionSign > 0.0f) {
                        layerAccum = std::min(layerAccum + std::max(0.0f, deposit), targetLayer);
                    } else {
                        layerAccum = std::max(layerAccum + std::min(0.0f, deposit), -targetLayer);
                    }
                    wd = mirWorldNormal * (layerAccum - previousAccum);
                    break;
                }
                case SculptBrushTool::Clay: {
                    Vec3 vn = mirWorldNormal;
                    if (!vertex.refs.empty() && editable_mesh_cache.refTri(vertex.refs[0])) {
                        const Vec3 n = editable_mesh_cache.refTri(vertex.refs[0])->vertices[vertex.refs[0].corner].normal;
                        if (n.length_squared() > 1e-8f) vn = safeNormalizeVec3(n, mirWorldNormal);
                    }
                    Vec3 mirClaySampleCenter = mirWorldCenter;
                    if (clayLikeTool && sculpt_stroke_state.changed) {
                        Vec3 localPrevHit = sanitizeVec3(inverseTransform.transform_point(lastStrokeHit), Vec3(0.0f, 0.0f, 0.0f));
                        if (do_mx) localPrevHit.x = -localPrevHit.x;
                        if (do_my) localPrevHit.y = -localPrevHit.y;
                        if (do_mz) localPrevHit.z = -localPrevHit.z;
                        const Vec3 mirPrevCenter = sanitizeVec3(transform.transform_point(localPrevHit), mirWorldCenter);
                        mirClaySampleCenter = sanitizeVec3(mirPrevCenter + (mirWorldCenter - mirPrevCenter) * 0.75f, mirWorldCenter);
                    }
                    const float mirClayHeightRef = radiusWorld * clayRadiusCompensation;
                    const float sd = (worldPos - mirClaySampleCenter).dot(mirWorldNormal);
                    const float targetHeight =
                        directionSign * mirClayHeightRef * 0.16f * (0.8f + 0.5f * normalStrength) * mirDragFactor;
                    if (vertexId >= sculpt_stroke_state.clay_layer_accum.size()) {
                        return;
                    }
                    const float heightError = targetHeight - sd;
                    const float absTarget = (std::abs)(targetHeight);
                    const float fillNeed = (absTarget > 1e-5f)
                        ? std::clamp(((directionSign > 0.0f) ? heightError : -heightError) / absTarget, 0.0f, 1.0f)
                        : 0.0f;
                    const float deposit =
                        mirClayHeightRef * 0.075f * clayBrushStrength * dt * w * strokeAdvanceFactor *
                        (0.8f + 0.5f * normalStrength) * directionSign * fillNeed * mirDragFactor;
                    float& clayAccum = sculpt_stroke_state.clay_layer_accum[vertexId];
                    clayAccum = std::clamp(clayAccum + deposit, -mirClayHeightRef * 0.45f, mirClayHeightRef * 0.45f);
                    const float settle =
                        std::clamp(heightError * w * (0.28f + 0.32f * w), -mirClayHeightRef * 0.05f, mirClayHeightRef * 0.05f);
                    const float layerDelta = settle * 0.7f + deposit * 0.55f + clayAccum * 0.12f * mirDragFactor;
                    const float buildup = deposit * 0.14f + (std::max)(0.0f, heightError) * 0.10f * fillNeed;
                    if (vertex.is_boundary) {
                        wd = mirWorldNormal * (layerDelta + buildup) * 0.6f;
                    } else {
                        wd = mirWorldNormal * layerDelta + vn * buildup;
                    }
                    break;
                }
                case SculptBrushTool::ClayStrips: {
                    Vec3 mirClaySampleCenter = mirWorldCenter;
                    if (clayLikeTool && sculpt_stroke_state.changed) {
                        Vec3 localPrevHit = sanitizeVec3(inverseTransform.transform_point(lastStrokeHit), Vec3(0.0f, 0.0f, 0.0f));
                        if (do_mx) localPrevHit.x = -localPrevHit.x;
                        if (do_my) localPrevHit.y = -localPrevHit.y;
                        if (do_mz) localPrevHit.z = -localPrevHit.z;
                        const Vec3 mirPrevCenter = sanitizeVec3(transform.transform_point(localPrevHit), mirWorldCenter);
                        mirClaySampleCenter = sanitizeVec3(mirPrevCenter + (mirWorldCenter - mirPrevCenter) * 0.75f, mirWorldCenter);
                    }
                    const float sd = (worldPos - mirClaySampleCenter).dot(mirWorldNormal);
                    const Vec3 mirStrokeTangent = strokeTangentWorld;
                    const Vec3 mirStrokeBitangent = safeNormalizeVec3(
                        mirWorldNormal.cross(mirStrokeTangent),
                        strokeBitangentWorld);
                    const Vec3 mirClayPlanarOffset = worldPos - mirClaySampleCenter -
                        mirWorldNormal * (worldPos - mirClaySampleCenter).dot(mirWorldNormal);
                    const float stripCoord = mirClayPlanarOffset.dot(mirStrokeBitangent) / radiusWorld;
                    const float tineWave = 0.5f + 0.5f * std::cos(stripCoord * 22.0f);
                    const float tineProfile = std::pow((std::max)(0.0f, tineWave), 1.6f);
                    const float rakePattern = 0.45f + 0.55f * tineProfile;
                    if (vertexId >= sculpt_stroke_state.clay_strips_layer_accum.size()) {
                        return;
                    }
                    const float mirStripsHeightRef = radiusWorld * clayRadiusCompensation;
                    const float stripTargetHeight =
                        directionSign * mirStripsHeightRef * 0.14f * rakePattern * (0.9f + 0.35f * normalStrength) * mirDragFactor;
                    const float stripHeightError = stripTargetHeight - sd;
                    const float absStripTarget = (std::abs)(stripTargetHeight);
                    const float stripFillNeed = (absStripTarget > 1e-5f)
                        ? std::clamp(((directionSign > 0.0f) ? stripHeightError : -stripHeightError) / absStripTarget, 0.0f, 1.0f)
                        : 0.0f;
                    const float deposit =
                        mirStripsHeightRef * 0.11f * clayBrushStrength * dt * w * strokeAdvanceFactor * rakePattern *
                        (0.95f + 0.45f * normalStrength) * directionSign * stripFillNeed * mirDragFactor;
                    float& clayStripsAccum = sculpt_stroke_state.clay_strips_layer_accum[vertexId];
                    clayStripsAccum = std::clamp(clayStripsAccum + deposit, -mirStripsHeightRef * 0.6f, mirStripsHeightRef * 0.6f);
                    const float settle =
                        std::clamp(stripHeightError * w * (0.24f + 0.28f * w), -mirStripsHeightRef * 0.05f, mirStripsHeightRef * 0.05f);
                    const float layerDelta = settle * 0.62f + deposit * 0.62f + clayStripsAccum * 0.18f;
                    const float tineDrag =
                        radiusWorld * 0.05f * clayBrushStrength * dt * w * strokeAdvanceFactor *
                        (0.35f + 0.65f * tineProfile);
                    if (vertex.is_boundary) {
                        wd = mirWorldNormal * layerDelta * 0.6f;
                    } else {
                        wd = mirWorldNormal * layerDelta + mirStrokeTangent * tineDrag;
                    }
                    break;
                }
                case SculptBrushTool::Pinch: {
                    const Vec3 toCenter = mirWorldCenter - worldPos;
                    const Vec3 lateral = toCenter - mirWorldNormal * toCenter.dot(mirWorldNormal);
                    const float lateralLen = lateral.length();
                    if (lateralLen > 1e-8f) {
                        wd = (lateral / lateralLen) * (radiusWorld * 0.3f * brushStrength * dt * w);
                    }
                    break;
                }
                case SculptBrushTool::Flatten: {
                    const float sd = (worldPos - mirWorldCenter).dot(mirWorldNormal);
                    wd = mirWorldNormal * (-sd * brushStrength * dt * 12.0f * w);
                    break;
                }
                case SculptBrushTool::Scrape: {
                    const float sd = (worldPos - mirWorldCenter).dot(mirWorldNormal);
                    if (sd > 0.0f) {
                        wd = mirWorldNormal * (-sd * brushStrength * dt * 14.0f * w);
                    }
                    break;
                }
                case SculptBrushTool::Crease: {
                    const Vec3 toCenter = mirWorldCenter - worldPos;
                    const Vec3 lateral = toCenter - mirWorldNormal * toCenter.dot(mirWorldNormal);
                    const float lateralLen = lateral.length();
                    const Vec3 pinchDelta = lateralLen > 1e-8f
                        ? (lateral / lateralLen) * (radiusWorld * 0.38f * brushStrength * dt * w)
                        : Vec3(0.0f, 0.0f, 0.0f);
                    const Vec3 cutDelta = mirWorldNormal * (-radiusWorld * 0.12f * brushStrength * dt * w * (1.0f + normalStrength));
                    wd = pinchDelta + cutDelta;
                    break;
                }
                case SculptBrushTool::DrawSharp: {
                    Vec3 vnSharp = mirWorldNormal;
                    if (!vertex.refs.empty() && editable_mesh_cache.refTri(vertex.refs[0])) {
                        const Vec3 n = editable_mesh_cache.refTri(vertex.refs[0])->vertices[vertex.refs[0].corner].normal;
                        if (n.length_squared() > 1e-8f) vnSharp = safeNormalizeVec3(n, mirWorldNormal);
                    }
                    const float sharpW = w * w;
                    wd = vnSharp * (radiusWorld * 0.26f * brushStrength * dt * sharpW * (1.0f + normalStrength) * directionSign);
                    break;
                }
                case SculptBrushTool::Nudge: {
                    wd = (mirWorldCenter - mirWorldPrevCenter) * (w * brushStrength * directionSign);
                    break;
                }
                case SculptBrushTool::Blob: {
                    const Vec3 toCenter = mirWorldCenter - worldPos;
                    const Vec3 lateral = toCenter - mirWorldNormal * toCenter.dot(mirWorldNormal);
                    const float lateralLen = lateral.length();
                    Vec3 gather(0.0f, 0.0f, 0.0f);
                    if (lateralLen > 1e-8f) {
                        gather = (lateral / lateralLen) * (radiusWorld * 0.10f * brushStrength * dt * w * directionSign);
                    }
                    wd = mirWorldNormal * (radiusWorld * 0.26f * brushStrength * dt * w * (1.0f + normalStrength) * directionSign) + gather;
                    break;
                }
                case SculptBrushTool::Fill: {
                    const float sd = (worldPos - mirWorldCenter).dot(mirWorldNormal);
                    if ((directionSign > 0.0f && sd < 0.0f) || (directionSign < 0.0f && sd > 0.0f)) {
                        wd = mirWorldNormal * (-sd * brushStrength * dt * 12.0f * w);
                    }
                    break;
                }
                case SculptBrushTool::SnakeHook: {
                    const Vec3 mirStep = mirWorldCenter - mirWorldPrevCenter;
                    const float stepLen = mirStep.length();
                    const Vec3 toCenter = mirWorldCenter - worldPos;
                    const Vec3 lateral = toCenter - mirWorldNormal * toCenter.dot(mirWorldNormal);
                    const float lateralLen = lateral.length();
                    Vec3 pinch(0.0f, 0.0f, 0.0f);
                    if (lateralLen > 1e-8f) {
                        pinch = (lateral / lateralLen) * (stepLen * 0.45f * brushStrength * w);
                    }
                    wd = mirStep * (w * brushStrength) + pinch;
                    break;
                }
                case SculptBrushTool::Smooth: {
                    const Vec3 sld = computeBoundarySafeSmoothDelta(
                        editable_mesh_cache,
                        kEmptySnapshot,
                        vertexId,
                        brushStrength,
                        dt,
                        w,
                        localRadius);
                    const float sldLenSq = sld.length_squared();
                    if (std::isfinite(sldLenSq) && sldLenSq > 1e-14f) {
                        sculpt_updated_local_positions[vertexId] = sanitizeVec3(
                            snapshotLocalPosition + sld,
                            snapshotLocalPosition);
                        mirrorPassChanged.store(true, std::memory_order_relaxed);
                    }
                    return;
                }
                default:
                    return;
                }

                wd = sanitizeVec3(wd, Vec3(0.0f, 0.0f, 0.0f));

                Vec3 ld = sanitizeVec3(inverseTransform.transform_vector(wd), Vec3(0.0f, 0.0f, 0.0f));
                const float ldLenSq = ld.length_squared();
                if (!std::isfinite(ldLenSq) || ldLenSq <= 1e-14f) {
                    return;
                }
                // Mirror non-grab pass: aynı dab-step clamp'ı (yoğun meshde flip
                // önler).
                const float mirMeshAvgEdge = sanitizeFiniteFloat(
                    sculpt_control_graph.avg_edge_length, 0.05f, 1e-6f, 1000000.0f);
                const float mirMaxLocalStep = (std::max)(mirMeshAvgEdge * 0.4f, localRadius * 1e-4f);
                const float ldLen = std::sqrt(ldLenSq);
                if (std::isfinite(ldLen) && ldLen > mirMaxLocalStep) {
                    ld *= (mirMaxLocalStep / ldLen);
                }
                sculpt_updated_local_positions[vertexId] = sanitizeVec3(
                    snapshotLocalPosition + ld,
                    snapshotLocalPosition);
                mirrorPassChanged.store(true, std::memory_order_relaxed);
            });
            changed = changed || mirrorPassChanged.load(std::memory_order_relaxed);
        }
    }

    // Radial symmetry: repeat the stroke as N-1 extra rotated copies about an
    // object-local axis through the origin. Non-grab tools reuse the primary
    // solve (computeSculptBrushSampleAtWorldPoint + applyNonGrabSculptCandidate)
    // with a rotated brush center/normal — no per-tool switch duplication. Grab
    // gets a rotated copy of the mirror-grab path (primed per-copy candidate set
    // + rotated start hit / drag).
    if (!anchoredActive && sculpt_mode_state.radial_symmetry && sculpt_mode_state.radial_count >= 2) {
        const Matrix4x4 radialNormalMtx = inverseTransform.transpose();
        const int radialAxis = std::clamp(sculpt_mode_state.radial_axis, 0, 2);
        const int radialCount = std::clamp(sculpt_mode_state.radial_count, 2, 64);
        const Vec3 localHitNormal = safeNormalizeVec3(
            inverseTransform.transform_vector(hitNormalWorld), hitNormalWorld);
        for (int k = 1; k < radialCount; ++k) {
            const float angle = (6.28318530718f * static_cast<float>(k)) /
                                static_cast<float>(radialCount);
            const Vec3 radLocalNormal = rotateLocalAroundAxis(localHitNormal, radialAxis, angle);
            const Vec3 radWorldNormal = safeNormalizeVec3(
                radialNormalMtx.transform_vector(radLocalNormal), hitNormalWorld);

            // Grab: rotate the start hit + drag, drive a fixed primed candidate set
            // (drift-free, mirrors the mirror-grab path exactly).
            if (isGrabFamilyTool(activeTool)) {
                Vec3 localDrag = sanitizeVec3(inverseTransform.transform_vector(worldDragDelta), Vec3(0.0f, 0.0f, 0.0f));
                localDrag = rotateLocalAroundAxis(localDrag, radialAxis, angle);
                const Vec3 radWorldDrag = sanitizeVec3(transform.transform_vector(localDrag), Vec3(0.0f, 0.0f, 0.0f));
                Vec3 localStartHit = sanitizeVec3(
                    inverseTransform.transform_point(sanitizeVec3(sculpt_stroke_state.start_world_hit, hitPoint)),
                    Vec3(0.0f, 0.0f, 0.0f));
                localStartHit = rotateLocalAroundAxis(localStartHit, radialAxis, angle);
                const Vec3 radWorldStartHit = sanitizeVec3(transform.transform_point(localStartHit), hitPoint);

                auto& radialSet = sculpt_stroke_state.grab_radial_candidate_sets[k];
                if (radialSet.empty()) {
                    radialSet = collectSculptCandidateVerticesWithPBVHFallback(
                        sculpt_pbvh, editable_mesh_cache, localStartHit, localRadius);
                    // Topology grow (same as main + mirror grab seeds): complete the
                    // frozen radial set so no missed verts invert boundary triangles.
                    radialSet = expandEditableCandidateVerticesByTopology(
                        editable_mesh_cache, radialSet, 2);
                    for (const int vid : radialSet) {
                        if (vid >= 0 && static_cast<size_t>(vid) < editable_mesh_cache.vertices.size()) {
                            sculpt_stroke_state.grab_start_local_positions.try_emplace(
                                vid, editable_mesh_cache.vertices[vid].local_position);
                        }
                    }
                }
                for (const int vid : radialSet) {
                    if (vid >= 0 && static_cast<size_t>(vid) < editable_mesh_cache.vertices.size()) {
                        sculpt_updated_local_positions[vid] = editable_mesh_cache.vertices[vid].local_position;
                    }
                }
                strokeTouchedVertexIds.insert(
                    strokeTouchedVertexIds.end(), radialSet.begin(), radialSet.end());

                std::atomic<bool> radialGrabChanged{ false };
                forEachEditableCandidate(radialSet, [&](int vertexIdInt) {
                    if (vertexIdInt < 0) {
                        return;
                    }
                    const size_t vertexId = static_cast<size_t>(vertexIdInt);
                    auto startIt = sculpt_stroke_state.grab_start_local_positions.find(vertexIdInt);
                    if (startIt == sculpt_stroke_state.grab_start_local_positions.end()) {
                        return;
                    }
                    const Vec3& startLocalPos = startIt->second;
                    const Vec3 startWorldPos = sanitizeVec3(
                        transform.transform_point(startLocalPos), radWorldStartHit);
                    const Vec3 toVertex = startWorldPos - radWorldStartHit;
                    const float h = toVertex.dot(radWorldNormal);
                    const float planarDist = (toVertex - radWorldNormal * h).length();
                    if (!std::isfinite(h) || !std::isfinite(planarDist) || planarDist > radiusWorld) {
                        return;
                    }
                    if (sculpt_mode_state.front_faces_only && h < -(radiusWorld * 0.12f)) {
                        return;
                    }
                    const float w = grabFamilyWeight(
                        planarDist / radiusWorld,
                        mesh_overlay_settings.proportional_falloff_type,
                        activeTool == SculptBrushTool::ElasticDeform);
                    if (w <= 1e-5f) {
                        return;
                    }
                    // Grab-family follow ratio is FIXED (kSculptGrabFollowStrength), decoupled from the
    // brush Strength slider: displacement = cursorDelta * weight * follow. A constant
    // follow keeps per-sub-step displacement bounded on fast strokes so the topology
    // guard stops dropping/freezing verts; speed and slider no longer cause teleports.
    const float grabStrength = kSculptGrabFollowStrength;
                    Vec3 wd = radWorldDrag * w * grabStrength;
                    const float wdLen = wd.length();
                    if (!std::isfinite(wdLen)) {
                        return;
                    }
                    if (wdLen > radiusWorld * 3.0f) {
                        wd = wd * (radiusWorld * 3.0f / wdLen);
                    }
                    const Vec3 ld = sanitizeVec3(inverseTransform.transform_vector(wd), Vec3(0.0f, 0.0f, 0.0f));
                    const float ldLenSq = ld.length_squared();
                    if (!std::isfinite(ldLenSq) || ldLenSq <= 1e-14f) {
                        return;
                    }
                    sculpt_updated_local_positions[vertexId] = sanitizeVec3(
                        startLocalPos + ld, startLocalPos);
                    radialGrabChanged.store(true, std::memory_order_relaxed);
                });
                changed = changed || radialGrabChanged.load(std::memory_order_relaxed);
                continue;
            }

            const Vec3 radLocalCenter = rotateLocalAroundAxis(localHitPoint, radialAxis, angle);
            const Vec3 radLocalPrev = rotateLocalAroundAxis(localPrevHitPoint, radialAxis, angle);
            const Vec3 radWorldCenter = sanitizeVec3(transform.transform_point(radLocalCenter), hitPoint);
            const Vec3 radWorldPrev = sanitizeVec3(transform.transform_point(radLocalPrev), radWorldCenter);
            const Vec3 radTangent = std::abs(radWorldNormal.y) < 0.95f
                ? safeNormalizeVec3(radWorldNormal.cross(Vec3(0, 1, 0)), Vec3(1, 0, 0))
                : safeNormalizeVec3(radWorldNormal.cross(Vec3(1, 0, 0)), Vec3(0, 0, 1));
            const Vec3 radBitangent = safeNormalizeVec3(radWorldNormal.cross(radTangent), Vec3(0, 0, 1));
            const Vec3 radClaySample = (clayLikeTool && sculpt_stroke_state.changed)
                ? sanitizeVec3(radWorldPrev + (radWorldCenter - radWorldPrev) * 0.75f, radWorldCenter)
                : radWorldCenter;

            const std::vector<int> radialCandidates = collectSculptCandidateVerticesWithPBVHFallback(
                sculpt_pbvh, editable_mesh_cache, radLocalCenter, localRadius * brushFootprintBoundScale);
            for (const int vid : radialCandidates) {
                if (vid >= 0 && static_cast<size_t>(vid) < editable_mesh_cache.vertex_positions.size()) {
                    sculpt_updated_local_positions[static_cast<size_t>(vid)] =
                        editable_mesh_cache.vertex_positions[static_cast<size_t>(vid)];
                }
            }
            strokeTouchedVertexIds.insert(
                strokeTouchedVertexIds.end(),
                radialCandidates.begin(),
                radialCandidates.end());

            std::atomic<bool> radialPassChanged{ false };
            SIMDSculptParams params;
            params.center = radWorldCenter;
            params.prevCenter = radWorldPrev;
            params.normal = radWorldNormal;
            params.tangent = radTangent;
            params.bitangent = radBitangent;
            params.radius = radiusWorld;
            params.cullRadius = brushCullRadiusWorld;
            params.falloff = falloff;
            params.strokeDistance = strokeDistance;
            params.strokeSpacing = strokeSpacing;
            params.dt = dt;
            params.strokeChanged = sculpt_stroke_state.changed;
            params.frontFacesOnly = sculpt_mode_state.front_faces_only;
            params.falloffType = mesh_overlay_settings.proportional_falloff_type;
            params.activeTool = activeTool;
            params.useMask = true;
            params.brushSettings = effectiveBrush;

            std::vector<EvaluatedSculptVertex> activeRadVerts = evaluateActiveSculptVerticesSIMD(
                radialCandidates,
                editable_mesh_cache,
                transform,
                params);

            std::vector<int> activeIndices(activeRadVerts.size());
            for (size_t i = 0; i < activeRadVerts.size(); ++i) {
                activeIndices[i] = static_cast<int>(i);
            }

            forEachEditableCandidate(activeIndices, [&](int index) {
                const auto& ev = activeRadVerts[index];
                const size_t vertexId = static_cast<size_t>(ev.id);

                SculptBrushSample sample;
                sample.local_position = editable_mesh_cache.vertex_positions[vertexId];
                sample.world_position = ev.worldPos;
                sample.height_from_plane = ev.height;
                sample.planar_distance = ev.planarDist;
                sample.weight = ev.weight;
                sample.clay_drag_factor = ev.clayDragFactor;

                const Vec3 segDir = radWorldCenter - radWorldPrev;
                const float segLenSq = segDir.dot(segDir);
                Vec3 closestPlanePoint;
                if (segLenSq > 1e-10f) {
                    const float t = std::clamp((ev.worldPos - radWorldPrev).dot(segDir) / segLenSq, 0.0f, 1.0f);
                    closestPlanePoint = radWorldPrev + segDir * t;
                } else {
                    closestPlanePoint = radWorldCenter;
                }
                sample.planar_offset = ev.worldPos - closestPlanePoint - radWorldNormal * ev.height;

                if (applyNonGrabSculptCandidate(
                        activeTool,
                        editable_mesh_cache,
                        sculpt_control_graph,
                        sculpt_stroke_state,
                        kEmptySnapshot,
                        transform,
                        inverseTransform,
                        radWorldCenter,
                        radClaySample,
                        radWorldNormal,
                        radTangent,
                        radBitangent,
                        (radWorldCenter - radWorldPrev),
                        brushStrength,
                        clayBrushStrength,
                        clayRadiusCompensation,
                        normalStrength,
                        directionSign,
                        radiusWorld,
                        dt,
                        localRadius,
                        strokeAdvanceFactor,
                        ev.id,
                        sample,
                        sculpt_updated_local_positions[vertexId])) {
                    radialPassChanged.store(true, std::memory_order_relaxed);
                }
            });
            changed = changed || radialPassChanged.load(std::memory_order_relaxed);
        }
    }

    if (!changed) {
        sculpt_stroke_state.last_world_hit = (isGrabFamilyTool(activeTool)) ? planeHit
            : ((activeTool == SculptBrushTool::Nudge) ? snakePlaneHit : hit.point);
        return;
    }

    std::sort(strokeTouchedVertexIds.begin(), strokeTouchedVertexIds.end());
    strokeTouchedVertexIds.erase(
        std::unique(strokeTouchedVertexIds.begin(), strokeTouchedVertexIds.end()),
        strokeTouchedVertexIds.end());
    const std::vector<int> touchedPBVHLeafIds =
        collectTouchedSculptPBVHLeafIds(sculpt_pbvh, strokeTouchedVertexIds);
    sculpt_control_graph.last_touched_leaf_count = touchedPBVHLeafIds.size();
    std::vector<size_t> pbvhExpandedAffectedVertexIds =
        collectAffectedEditableVertexIdsFromPBVHLeaves(
            editable_mesh_cache,
            sculpt_pbvh,
            touchedPBVHLeafIds);
    const std::vector<int> expandedStrokeVertexIds =
        convertAffectedVertexIdsToIntList(pbvhExpandedAffectedVertexIds);

    // The PBVH-leaf affected set (expandedStrokeVertexIds) can reach FAR outside the
    // brush on SPARSE meshes — a leaf's spatial bounds are large when verts are few,
    // so it sweeps in vertices well beyond the brush radius. The clay/draw polish,
    // anti-pit and ribble passes below relax EVERY vertex they're handed toward its
    // neighbour average with no radius weighting, so handing them the raw leaf set
    // drags those far verts toward the stroke ("the brush pulls in vertices outside
    // its circle"). Clip the cleanup set to the brush radius: out-of-radius verts are
    // still read as FIXED neighbour anchors for the in-radius relaxation, they're just
    // never moved themselves. Cheap O(expanded) pre-filter, done once.
    std::vector<int> cleanupVertexIds;
    cleanupVertexIds.reserve(expandedStrokeVertexIds.size());
    {
        const float cleanupRadiusSq = localRadius * localRadius;
        for (const int vid : expandedStrokeVertexIds) {
            if (vid < 0 || vid >= static_cast<int>(editable_mesh_cache.vertex_positions.size())) {
                continue;
            }
            const Vec3 d = editable_mesh_cache.vertex_positions[static_cast<size_t>(vid)] - localHitPoint;
            if (d.length_squared() <= cleanupRadiusSq) {
                cleanupVertexIds.push_back(vid);
            }
        }
    }

    if (activeTool == SculptBrushTool::Clay || activeTool == SculptBrushTool::ClayStrips) {
        const float largeBrushSurfaceFactor = computeLargeBrushSurfaceFactor(radiusWorld);
        const std::vector<int>& polishVertexIds =
            !cleanupVertexIds.empty() ? cleanupVertexIds : strokeTouchedVertexIds;
        const float clayPolishStrength = (activeTool == SculptBrushTool::Clay ? 0.16f : 0.11f) +
            (activeTool == SculptBrushTool::Clay ? 0.10f : 0.08f) * largeBrushSurfaceFactor;
        applyClayPolishPass(
            editable_mesh_cache,
            sculpt_updated_local_positions,
            polishVertexIds,
            transform,
            inverseTransform,
            hitNormalWorld,
            brushStrength,
            dt,
            localRadius,
            clayPolishStrength);
        applyClayAntiPitPass(
            editable_mesh_cache,
            sculpt_updated_local_positions,
            polishVertexIds,
            transform,
            inverseTransform,
            hitNormalWorld,
            localRadius,
            (activeTool == SculptBrushTool::Clay ? 0.42f : 0.28f) +
                (activeTool == SculptBrushTool::Clay ? 0.24f : 0.18f) * largeBrushSurfaceFactor,
            (activeTool == SculptBrushTool::Clay ? 0.12f : 0.18f) +
                0.10f * largeBrushSurfaceFactor);
    } else if (activeTool == SculptBrushTool::Draw && !cleanupVertexIds.empty()) {
        const float largeBrushSurfaceFactor = computeLargeBrushSurfaceFactor(radiusWorld);
        const size_t touchedCount = cleanupVertexIds.size();
        const float densityFactor = std::clamp(
            (static_cast<float>(touchedCount) - 24.0f) / 72.0f, 0.0f, 1.0f);
        const float cleanupBoost = (std::max)(largeBrushSurfaceFactor, densityFactor);
        if (cleanupBoost > 0.04f) {
            applyClayPolishPass(
                editable_mesh_cache,
                sculpt_updated_local_positions,
                cleanupVertexIds,
                transform,
                inverseTransform,
                hitNormalWorld,
                brushStrength,
                dt,
                localRadius,
                0.04f + 0.10f * cleanupBoost);
            applyClayAntiPitPass(
                editable_mesh_cache,
                sculpt_updated_local_positions,
                cleanupVertexIds,
                transform,
                inverseTransform,
                hitNormalWorld,
                localRadius,
                0.24f + 0.16f * cleanupBoost,
                0.08f + 0.08f * cleanupBoost);
        }
    }

    // Dab ribble removal — the clay/draw polish above only smooths TANGENTIALLY (it strips
    // the stroke-normal component to keep deposited height), so it leaves the cross-stroke
    // "stair-step" ridges that discrete dab centers stamp into dense meshes. This pass adds
    // a NORMAL-inclusive Laplacian that attenuates the high-curvature ribble while the broad
    // form (low curvature) survives, gated to only run once the mesh is dense enough to
    // resolve the dab pitch (coarse meshes pay nothing).
    //
    // ClayStrips is deliberately EXCLUDED: it doesn't terrace (its tine drag already smears
    // the plates together) and a generic smoothing would only risk softening its rake tines —
    // user-confirmed it already looks correct, so leave it untouched.
    const bool ribbleProneTool =
        activeTool == SculptBrushTool::Draw ||
        activeTool == SculptBrushTool::Inflate ||
        activeTool == SculptBrushTool::Clay ||
        activeTool == SculptBrushTool::Layer;
    if (ribbleProneTool) {
        const float ribbleRisk = computeDabRibbleRisk(localRadius, sculpt_control_graph.avg_edge_length);
        if (ribbleRisk > 0.0f) {
            const std::vector<int>& ribbleVertexIds =
                !cleanupVertexIds.empty() ? cleanupVertexIds : strokeTouchedVertexIds;
            // Clay terraces ALONG the stroke (per-dab flatten-to-reference), so it smooths
            // hard but ONLY along the stroke tangent (tangentBias high) — that erases the
            // stair-steps without flattening the broad cross-section the brush is building.
            // A larger clamp lets a step actually collapse in one pass. Draw/Inflate ribble
            // isotropically (dab-sum), so they stay a plain light Laplacian.
            const bool clayLike = activeTool == SculptBrushTool::Clay;
            const Vec3 localStrokeTangent = clayLike
                ? sanitizeVec3(inverseTransform.transform_vector(strokeTangentWorld), Vec3(0.0f, 0.0f, 0.0f))
                : Vec3(0.0f, 0.0f, 0.0f);
            const float ribbleStrength = (clayLike ? 0.55f : 0.20f) * ribbleRisk;
            const float ribbleTangentBias = clayLike ? 0.85f : 0.0f;
            const float ribbleMaxDeltaFactor = clayLike ? 0.12f : 0.05f;
            applyDabRibbleSmoothPass(
                editable_mesh_cache,
                sculpt_updated_local_positions,
                ribbleVertexIds,
                localRadius,
                ribbleStrength,
                localStrokeTangent,
                ribbleTangentBias,
                ribbleMaxDeltaFactor);
        }
    }

    // Surface relaxation (anti-fold) — the real cure for the "brushed quads fold along their
    // triangulation diagonal" artifact. We sculpt the LINEAR evaluated mesh, so an additive
    // brush makes each touched quad non-planar and the fixed 2-triangle diagonal shows as a
    // crease (worst at low/moderate density; multires would hide it but costs on dense meshes).
    // A light ISOTROPIC Laplacian (pull each touched vert a fraction toward its neighbour
    // centroid) removes that sub-quad fold while the brush re-deposits the broad form every dab,
    // so the shape survives. Unlike the ribble pass this is NOT density-gated (the fold is
    // visible even on coarse meshes) and isotropic (tangentBias 0). Skipped for Grab/Smooth/Mask
    // (own coherent solve) and ClayStrips (its rake tines are user-confirmed; a generic smooth
    // would soften them).
    if (sculpt_mode_state.surface_relax_enabled &&
        sculpt_mode_state.surface_relax_strength > 1e-4f &&
        !isGrabFamilyTool(activeTool) &&
        activeTool != SculptBrushTool::Smooth &&
        activeTool != SculptBrushTool::Mask &&
        activeTool != SculptBrushTool::ClayStrips) {
        const std::vector<int>& relaxVertexIds =
            !cleanupVertexIds.empty() ? cleanupVertexIds : strokeTouchedVertexIds;
        applyDabRibbleSmoothPass(
            editable_mesh_cache,
            sculpt_updated_local_positions,
            relaxVertexIds,
            localRadius,
            sculpt_mode_state.surface_relax_strength,
            Vec3(0.0f, 0.0f, 0.0f), // isotropic — remove the fold in every direction
            0.0f,
            0.18f);                 // generous relax-delta clamp: a deep fold can collapse in one dab
    }

    // Wet-clay deposit: when enabled, the verts this additive stroke commits become
    // freshly WET, so stepWetClayField keeps settling them after the dab (the dynamic
    // "soft clay that sets" behaviour). Grab/Smooth/Mask drive their own coherent solves
    // and aren't a clay deposit, so they never inject wetness.
    const bool wetClayActive =
        sculpt_mode_state.wet_clay_enabled &&
        !isGrabFamilyTool(activeTool) &&
        activeTool != SculptBrushTool::Smooth &&
        activeTool != SculptBrushTool::Mask;
    std::vector<int> wetDepositIds;
    std::vector<Vec3> wetDepositAnchors; // pre-deposit position (wall reference) per id
    if (wetClayActive) {
        wetDepositIds.reserve(strokeTouchedVertexIds.size());
        wetDepositAnchors.reserve(strokeTouchedVertexIds.size());
    }

    for (const int vertexIdInt : strokeTouchedVertexIds) {
        if (vertexIdInt < 0) {
            continue;
        }
        const size_t vertexId = static_cast<size_t>(vertexIdInt);
        if (vertexId >= editable_mesh_cache.vertices.size()) {
            continue;
        }
        EditableVertex& vertex = editable_mesh_cache.vertices[vertexId];
        if ((sculpt_updated_local_positions[vertexId] - editable_mesh_cache.vertex_positions[vertexId]).length_squared() <= 1e-14f) {
            continue;
        }
        Vec3 safeLocalPosition = editable_mesh_cache.vertex_positions[vertexId];
        if (isGrabFamilyTool(activeTool) || anchoredActive) {
            // Grab/Elastic/SnakeHook and Anchored stamp all move a COHERENT field
            // (start_local + weight*offset): every footprint vertex shares the same
            // solve offset, only scaled by a smooth falloff weight, so neighbours
            // travel together (anchored shares the normal*depth offset). The per-vertex
            // topology guard below is built for ADDITIVE (normal-push) brushes where a
            // single vertex can spike a triangle; applying it to grab is actively
            // harmful — on dense meshes / fast strokes its binary backoff FREEZES
            // scattered centre verts at their old position while their moved neighbours
            // pull away, producing the thin spikes from the brush centre. Commit the
            // coherent field directly (Blender-style); reject only non-finite results so
            // a NaN can never leak into the mesh.
            const Vec3 proposed = sculpt_updated_local_positions[vertexId];
            if (!isFiniteVec3(proposed)) {
                continue;
            }
            safeLocalPosition = proposed;
        } else if (!resolveEditableTopologySafePosition(
                editable_mesh_cache,
                vertexIdInt,
                sculpt_updated_local_positions,
                strokeTouchedVertexIds,
                safeLocalPosition)) {
            continue;
        }
        // Protection mask — single choke point. Pull this frame's motion back
        // toward the pre-move position by the mask factor (1 = free, 0 = frozen).
        // Because every brush path (Grab, Clay + clay-polish/anti-pit, Smooth,
        // and all mirror passes) writes into sculpt_updated_local_positions and
        // commits through this loop, masking here covers them uniformly — no
        // per-brush gating needed.
        const float maskFactor = sculptMaskFactor(sculpt_mask_state, vertexId);
        if (maskFactor < 0.999f) {
            safeLocalPosition = editable_mesh_cache.vertex_positions[vertexId] +
                (safeLocalPosition - editable_mesh_cache.vertex_positions[vertexId]) * maskFactor;
        }
        const Vec3 preDepositPos = editable_mesh_cache.vertex_positions[vertexId];
        vertex.local_position = safeLocalPosition;
        editable_mesh_cache.vertex_positions[vertexId] = safeLocalPosition;
        sculpt_updated_local_positions[vertexId] = safeLocalPosition;
        if (wetClayActive) {
            wetDepositIds.push_back(vertexIdInt);
            wetDepositAnchors.push_back(preDepositPos); // wall reference, before this dab pushed out
        }
        for (const auto& ref : vertex.refs) {
            Triangle* refTri = editable_mesh_cache.refTri(ref);
            if (!refTri) {
                continue;
            }
            sculpt_stroke_state.before_triangle_states.try_emplace(
                refTri,
                MeshEditTriangleState{
                    editable_mesh_cache.refTriShared(ref),
                    {
                        refTri->getOriginalVertexPosition(0),
                        refTri->getOriginalVertexPosition(1),
                        refTri->getOriginalVertexPosition(2)
                    }
                });
            refTri->setOriginalVertexPosition(ref.corner, vertex.local_position);
            refTri->markAABBDirty();
            if (tryMarkEditableTriangleTouched(editable_mesh_cache, refTri)) {
                touchedTriangles.push_back(editable_mesh_cache.refTriShared(ref));
            }
        }
    }
    if (wetClayActive && !wetDepositIds.empty()) {
        // Same boundary feather as the Water brush so the deposited clay's wetness fades at
        // the footprint rim instead of cutting off into a frozen ring (the flow ridges
        // against that cliff at high fluidity).
        std::vector<float> depositFeather;
        depositFeather.reserve(wetDepositIds.size());
        for (const int vid : wetDepositIds) {
            depositFeather.push_back(wetClayFeatherWeight(vid));
        }
        depositWetClay(wetDepositIds, wetDepositAnchors, sculpt_mode_state.wet_clay_wetness, &depositFeather);
    }
    // Restore the cache-mirror invariant for verts the clay polish / anti-pit passes
    // wrote but the commit loop did NOT push to the cache: those passes operate on
    // expandedStrokeVertexIds (a superset of strokeTouchedVertexIds), while commit
    // only iterates strokeTouchedVertexIds. Without this, the persistent buffer would
    // carry stale positions into the next frame's neighbor reads. O(expanded), cheap;
    // committed verts already match the cache so resetting them here is a no-op.
    for (const int vid : expandedStrokeVertexIds) {
        if (vid >= 0 && static_cast<size_t>(vid) < editable_mesh_cache.vertex_positions.size()) {
            sculpt_updated_local_positions[static_cast<size_t>(vid)] =
                editable_mesh_cache.vertex_positions[static_cast<size_t>(vid)];
        }
    }
    std::vector<size_t> affectedVertexIds = pbvhExpandedAffectedVertexIds;
    if (affectedVertexIds.empty()) {
        affectedVertexIds = collectAffectedEditableVertexIds(editable_mesh_cache, touchedTriangles);
    }
    expandTouchedTrianglesFromAffectedVertices(
        editable_mesh_cache,
        affectedVertexIds,
        touchedTriangles);
    recomputeEditableSmoothNormals(editable_mesh_cache, affectedVertexIds);
    refreshSculptPBVHLeavesAndAncestors(
        sculpt_pbvh,
        editable_mesh_cache,
        touchedPBVHLeafIds);

    // Batch-apply transform to all touched triangles using precomputed matrices
    // instead of per-triangle getTransformMatrix() + inverse().transpose().
    {
        const Matrix4x4 finalTransform = transform;
        const Matrix4x4 normalTransform = inverseTransform.transpose();
        for (const auto& tri : touchedTriangles) {
            sculpt_stroke_state.touched_triangles.try_emplace(tri.get(), tri);
            for (int i = 0; i < 3; ++i) {
                tri->vertices[i].position = finalTransform.transform_point(tri->vertices[i].original);
                tri->vertices[i].normal = safeNormalizeVec3(
                    normalTransform.transform_vector(tri->vertices[i].originalNormal),
                    tri->vertices[i].normal);
            }
            tri->markAABBDirty();
            tri->update_bounding_box();
        }
    }

    // Track dirty triangle indices for partial raster sync (avoids full-mesh extraction).
    // ACCUMULATE across every dab/sub-step in this frame: a fast stroke fires several
    // interpolated sub-step dabs per frame, but only ONE processPendingMeshEditGpuSync
    // runs per frame and consumes (then clears) this set. If each dab cleared it, only
    // the LAST dab's triangles would reach the GPU; earlier dabs' triangles stay stale
    // and render as stretched/degenerate (until stroke end re-syncs the whole set). The
    // consumer dedups + clears; here we only append. With no GPU backend nothing consumes
    // it, so clear to avoid unbounded growth (the CPU BVH refit path is used instead).
    const bool willGpuSyncDirty =
        (ctx.backend_ptr != nullptr || g_backend != nullptr || g_viewport_backend != nullptr);
    if (!willGpuSyncDirty) {
        sculpt_dirty_mesh_cache_indices.clear();
    }
    sculpt_dirty_mesh_cache_indices.reserve(
        sculpt_dirty_mesh_cache_indices.size() + touchedTriangles.size());
    for (const auto& tri : touchedTriangles) {
        const int faceIdx = editableFaceIndexOf(editable_mesh_cache, tri.get());
        if (faceIdx >= 0) {
            sculpt_dirty_mesh_cache_indices.push_back(editable_mesh_cache.face_to_mesh_index[static_cast<size_t>(faceIdx)]);
        }
    }

    // Don't invalidate overlay cache during sculpt — triangle data is updated in-place
    // so the overlay will read correct positions from the same Triangle pointers.
    objects_needing_cpu_sync.erase(objectName);

    if (!ctx.backend_ptr) {
        extern bool g_cpu_bvh_refit_pending;
        g_cpu_bvh_refit_pending = true;
    }

    if (liveAccumulationEnabled) {
        if (ctx.backend_ptr != nullptr || g_backend != nullptr || g_viewport_backend != nullptr) {
            queueMeshEditGpuSync(objectName);
        }
        ctx.renderer.resetCPUAccumulation();
        ctx.start_render = true;
    }
    sculpt_stroke_state.changed = true;
    sculpt_stroke_state.last_world_hit = (isGrabFamilyTool(activeTool)) ? planeHit : hit.point;
}

void SceneUI::processPendingMeshEditGpuSync(UIContext& ctx) {
    if (!mesh_edit_gpu_sync_pending || (!ctx.backend_ptr && !g_backend && !g_viewport_backend)) {
        return;
    }

    const std::string objectName = mesh_edit_gpu_sync_object_name;
    mesh_edit_gpu_sync_pending = false;
    mesh_edit_gpu_sync_object_name.clear();

    // The dirty set is accumulated across this frame's dabs (see the sculpt commit), so it
    // may hold duplicates from triangles touched by several sub-step dabs — collapse them
    // before the partial upload so each slot is patched once.
    if (!sculpt_dirty_mesh_cache_indices.empty()) {
        std::sort(sculpt_dirty_mesh_cache_indices.begin(), sculpt_dirty_mesh_cache_indices.end());
        sculpt_dirty_mesh_cache_indices.erase(
            std::unique(sculpt_dirty_mesh_cache_indices.begin(), sculpt_dirty_mesh_cache_indices.end()),
            sculpt_dirty_mesh_cache_indices.end());
    }

    extern bool g_bvh_rebuild_pending;
    extern bool g_optix_rebuild_pending;
    extern bool g_vulkan_rebuild_pending;
    extern bool g_viewport_raster_rebuild_pending;
    extern bool g_cpu_bvh_refit_pending;

    auto* renderVkBackend = dynamic_cast<Backend::VulkanBackendAdapter*>(getMeshOverlayRenderBackend(ctx));
    auto* activeViewportRasterBackend = dynamic_cast<Backend::IViewportBackend*>(ctx.backend_ptr);
    auto* viewportVkBackend = getMeshOverlayViewportBackend(ctx);
    const bool renderBackendIsVulkan = (renderVkBackend != nullptr);
    const bool activeViewportIsRaster = (activeViewportRasterBackend != nullptr);
    const bool activeViewportInSolidMode = activeViewportIsRaster &&
        ctx.backend_ptr &&
        ctx.backend_ptr->getViewportMode() != Backend::ViewportMode::Rendered;

    auto syncMeshRasterViewport = [&](Backend::IViewportBackend* vkBackend) -> bool {
        if (!vkBackend) {
            return false;
        }

        auto meshCacheIt = mesh_cache.find(objectName);
        if (meshCacheIt != mesh_cache.end() && !sculpt_dirty_mesh_cache_indices.empty()) {
            if (vkBackend->patchRasterMeshTriangles(objectName, sculpt_dirty_mesh_cache_indices, meshCacheIt->second)) {
                vkBackend->resetAccumulation();
                return true;
            }
        }

        const auto triangles = Viewport::collectMeshTrianglesForObject(mesh_cache, objectName);
        if (!triangles.empty() && vkBackend->updateRasterMeshFromTriangles(objectName, triangles)) {
            vkBackend->resetAccumulation();
            return true;
        }

        vkBackend->buildRasterGeometry(ctx.scene.world.objects);
        vkBackend->resetAccumulation();

        if (meshCacheIt != mesh_cache.end() && !sculpt_dirty_mesh_cache_indices.empty()) {
            if (vkBackend->patchRasterMeshTriangles(objectName, sculpt_dirty_mesh_cache_indices, meshCacheIt->second)) {
                vkBackend->resetAccumulation();
                return true;
            }
        }
        if (!triangles.empty() && vkBackend->updateRasterMeshFromTriangles(objectName, triangles)) {
            vkBackend->resetAccumulation();
            return true;
        }

        return false;
    };

    bool rasterViewportUpdated = false;
    if (activeViewportRasterBackend) {
        rasterViewportUpdated = syncMeshRasterViewport(activeViewportRasterBackend);
    }
    if (viewportVkBackend && viewportVkBackend != activeViewportRasterBackend) {
        if (syncMeshRasterViewport(viewportVkBackend)) {
            rasterViewportUpdated = true;
        }
    }
    if (!rasterViewportUpdated && (activeViewportRasterBackend || viewportVkBackend)) {
        g_viewport_raster_rebuild_pending = true;
    }

    // CPU picking/CPU-render BVH refit. refitBVH does an Embree RTC_BUILD_QUALITY_REFIT
    // (self-escalates to a full rebuild on real topology change), but it still walks the
    // whole scene + rebuilds the particle BVH — O(N) on a multi-million-triangle mesh.
    // With a GPU viewport backend the CPU BVH is consulted ONLY for picking and CPU
    // reference render, neither of which runs mid-stroke, so a per-dab refit is wasted
    // work. Defer it to stroke end (see finishStroke) in that case. Only the
    // CPU-render-as-viewport path (no backend) needs the live per-dab refit because there
    // the CPU BVH IS the thing being drawn.
    if (!ctx.backend_ptr) {
        g_cpu_bvh_refit_pending = true;
    }

    // In Solid/Matcap sculpt we only need the interactive raster viewport to stay live.
    // Syncing the inactive render backend here (OptiX / Vulkan RT / CPU render backend)
    // duplicates geometry work every dab and can make Solid mode slower than Rendered.
    // We already mark geometry dirty at stroke end, so Rendered mode can catch up when
    // the user exits sculpt or switches viewport mode.
    if (activeViewportInSolidMode) {
        if (!sculpt_dirty_mesh_cache_indices.empty()) {
            size_t minIdx = std::numeric_limits<size_t>::max();
            size_t maxIdx = 0;
            for (size_t idx : sculpt_dirty_mesh_cache_indices) {
                if (idx < minIdx) minIdx = idx;
                if (idx > maxIdx) maxIdx = idx;
            }
            size_t numSlots = maxIdx - minIdx + 1;
          //  telemetry_touched_triangles = sculpt_dirty_mesh_cache_indices.size();
          //  telemetry_pcie_upload_bytes = numSlots * 72;
          //  telemetry_partial_upload_active = true;
        } else {
          //  telemetry_touched_triangles = 0;
          ////  telemetry_pcie_upload_bytes = 0;
          //  telemetry_partial_upload_active = false;
        }

        sculpt_dirty_mesh_cache_indices.clear();
        ctx.renderer.resetCPUAccumulation();
        ctx.start_render = true;
        return;
    }

    if (renderBackendIsVulkan) {
        // Device-resident CC node: the dense subdivision mesh lives on the GPU and the cage
        // is EXCLUDED from the RT gather, so a host BLAS partial update can't apply here — it
        // would fall through to a full scene rebuild on every sculpt dab / vertex move. Route
        // the edit to the GPU stencil re-apply + BLAS refit instead (no Apply, no rebuild).
        // sculpt/edit author the control cage (base_mesh_cache) when a subdivision modifier is
        // active, so re-evaluating the device CC from that cage reflects the edit live.
        {
            auto stackIt = ctx.scene.mesh_modifiers.find(objectName);
            auto cageIt = ctx.scene.base_mesh_cache.find(objectName);
            if (stackIt != ctx.scene.mesh_modifiers.end() &&
                cageIt != ctx.scene.base_mesh_cache.end() && !cageIt->second.empty() &&
                driveLiveDeviceResidentCC(objectName, cageIt->second, stackIt->second)) {
                sculpt_dirty_mesh_cache_indices.clear();
                ctx.renderer.resetCPUAccumulation();
                ctx.start_render = true;
                return;
            }
        }

        const bool allowVulkanIncrementalMeshEdit = kEnableVulkanInteractiveMeshRtUpdates;

        auto meshCacheIt = mesh_cache.find(objectName);
        if (allowVulkanIncrementalMeshEdit &&
            meshCacheIt != mesh_cache.end() &&
            !sculpt_dirty_mesh_cache_indices.empty()) {

            int64_t bytesUploaded = renderVkBackend->updateMeshBLASPartial(
                objectName,
                sculpt_dirty_mesh_cache_indices,
                meshCacheIt->second);

            if (bytesUploaded >= 0) {
               // telemetry_touched_triangles = sculpt_dirty_mesh_cache_indices.size();
               // telemetry_pcie_upload_bytes = static_cast<size_t>(bytesUploaded);
               // telemetry_partial_upload_active = true;

                sculpt_dirty_mesh_cache_indices.clear();
                ctx.renderer.resetCPUAccumulation();
                ctx.start_render = true;
                return;
            }
        }

        // Fallback: full rebuild/sync
        const auto triangles = Viewport::collectMeshTrianglesForObject(mesh_cache, objectName);
       // telemetry_touched_triangles = triangles.size();
       // telemetry_pcie_upload_bytes = triangles.size() * 72; // positions (36 bytes) + normals (36 bytes)
       // telemetry_partial_upload_active = false;

        if (allowVulkanIncrementalMeshEdit &&
            !triangles.empty() &&
            renderVkBackend->updateMeshBLASPartial(objectName, triangles)) {
            sculpt_dirty_mesh_cache_indices.clear();
            ctx.renderer.resetCPUAccumulation();
            ctx.start_render = true;
            return;
        }

        if (allowVulkanIncrementalMeshEdit) {
            const Viewport::MeshEditSyncRequest request{ objectName, activeViewportInSolidMode, false };
            const auto syncResult = Viewport::syncInteractiveMeshEdit(renderVkBackend, request, triangles);
            if (syncResult == Viewport::InteractiveMeshSyncResult::UpdatedIncrementally) {
                sculpt_dirty_mesh_cache_indices.clear();
                ctx.renderer.resetCPUAccumulation();
                ctx.start_render = true;
                return;
            }
        }
        g_vulkan_rebuild_pending = true;
    }
    else if (dynamic_cast<Backend::OptixBackend*>(getMeshOverlayRenderBackend(ctx)) != nullptr) {
        bool handled = false;
        if (mesh_edit_optix_targeted_sync_enabled) {
            try {
                if (auto* optixBackend = dynamic_cast<Backend::OptixBackend*>(g_backend.get())) {
                    auto meshIt = mesh_cache.find(objectName);
                    if (meshIt != mesh_cache.end() && !meshIt->second.empty()) {
                        std::vector<std::shared_ptr<Triangle>> triangles;
                        triangles.reserve(meshIt->second.size());
                        for (const auto& entry : meshIt->second) {
                            if (entry.second) {
                                triangles.push_back(entry.second);
                            }
                        }
                        if (!triangles.empty()) {
                            if (auto* wrapper = optixBackend->getOptixWrapper()) {
                                handled = wrapper->updateMeshBLASFromTriangles(objectName, triangles);
                            }
                        }
                    }
                }
            } catch (const std::exception&) {
                mesh_edit_optix_targeted_sync_enabled = false;
                handled = false;
            }
        }
        if (!handled) {
            g_optix_rebuild_pending = true;
        }
    } else {
        Backend::IBackend* renderBackend = getMeshOverlayRenderBackend(ctx);
        if (renderBackend) {
            renderBackend->updateGeometry(ctx.scene.world.objects);
            renderBackend->resetAccumulation();
        }
    }

    sculpt_dirty_mesh_cache_indices.clear();
    ctx.renderer.resetCPUAccumulation();
    ctx.start_render = true;
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU EDIT-MESH OVERLAY SYNC
// Pushes editable-mesh wireframe/vertex/face data into the raster viewport
// backend (Vulkan) so the per-frame overlay cost on the CPU collapses to a
// params push. Buffers are split by update frequency:
//   topology (edge/face index buffers)  -> on EditableMeshCache rebuild
//   geometry (local positions)          -> on vertex edits (queueMeshEditGpuSync)
//   flags + selection index buffers     -> on selection / soft-select changes
// ─────────────────────────────────────────────────────────────────────────────
namespace {
void appendEditOverlayFanIndices(std::vector<uint32_t>& out, const std::vector<int>& vertexIds, int vertexCount) {
    if (vertexIds.size() < 3) {
        return;
    }
    for (size_t i = 1; i + 1 < vertexIds.size(); ++i) {
        const int a = vertexIds[0];
        const int b = vertexIds[i];
        const int c = vertexIds[i + 1];
        if (a < 0 || b < 0 || c < 0 || a >= vertexCount || b >= vertexCount || c >= vertexCount) {
            continue;
        }
        out.push_back(static_cast<uint32_t>(a));
        out.push_back(static_cast<uint32_t>(b));
        out.push_back(static_cast<uint32_t>(c));
    }
}
} // namespace

void SceneUI::releaseGpuEditMeshOverlay() {
    if (!gpu_edit_overlay_sync.active) {
        return;
    }
    gpu_edit_overlay_sync = GpuEditOverlaySync{};
    if (auto* vpb = dynamic_cast<Backend::VulkanBackendAdapter*>(g_viewport_backend.get())) {
        vpb->clearEditMeshOverlay();
    }
}

void SceneUI::applySculptMaskOperation(int operation) {
    ensureSculptMaskSized(sculpt_mask_state, editable_mesh_cache, sculpt_mode_state.active_target_name);
    SculptMaskState& mask = sculpt_mask_state;
    const size_t n = mask.values.size();
    if (n == 0) {
        return;
    }
    switch (operation) {
    case 0: // Clear
        std::fill(mask.values.begin(), mask.values.end(), 0.0f);
        break;
    case 2: // Fill
        std::fill(mask.values.begin(), mask.values.end(), 1.0f);
        break;
    case 1: // Invert
        for (float& v : mask.values) {
            v = 1.0f - std::clamp(v, 0.0f, 1.0f);
        }
        break;
    case 3: { // Smooth: one Laplacian pass over the cached vertex neighbours.
        std::vector<float> out = mask.values;
        const auto& neighbors = editable_mesh_cache.vertex_neighbors;
        const size_t cap = (std::min)(n, neighbors.size());
        for (size_t i = 0; i < cap; ++i) {
            const auto& nb = neighbors[i];
            if (nb.empty()) {
                continue;
            }
            float sum = mask.values[i];
            int count = 1;
            for (int j : nb) {
                if (j >= 0 && static_cast<size_t>(j) < n) {
                    sum += mask.values[static_cast<size_t>(j)];
                    ++count;
                }
            }
            out[i] = sum / static_cast<float>(count);
        }
        mask.values.swap(out);
        break;
    }
    case 4: // Sharpen: push values away from 0.5 toward 0/1.
        for (float& v : mask.values) {
            const float c = std::clamp(v, 0.0f, 1.0f);
            v = std::clamp(c + (c - 0.5f) * 0.5f, 0.0f, 1.0f);
        }
        break;
    default:
        return;
    }
    mask.has_any = false;
    for (float v : mask.values) {
        if (v > 0.003f) {
            mask.has_any = true;
            break;
        }
    }
    ++mask.version;
}

bool SceneUI::syncGpuEditMeshOverlay(UIContext& ctx, const std::string& objectName,
                                     bool drawVertices, bool drawEdges, bool drawFaces) {
    (void)ctx;
    // GPU overlay lives in the raster interactive viewport pass; Rendered
    // mode (shading_mode == 2) keeps the ImGui fallback.
    if (viewport_settings.shading_mode == 2) {
        releaseGpuEditMeshOverlay();
        return false;
    }
    auto* vpb = dynamic_cast<Backend::VulkanBackendAdapter*>(g_viewport_backend.get());
    if (!vpb) {
        return false;
    }
    if (editable_mesh_cache.object_name.empty() ||
        editable_mesh_cache.object_name != objectName ||
        editable_mesh_cache.vertices.empty()) {
        releaseGpuEditMeshOverlay();
        return false;
    }

    auto& sync = gpu_edit_overlay_sync;
    const int vertexCount = static_cast<int>(editable_mesh_cache.vertices.size());
    const bool topologyChanged = !sync.active || sync.cache_revision != editable_mesh_cache.revision;

    if (topologyChanged) {
        // Polygon edges hide triangulation diagonals when available.
        const auto& edgeList = editable_mesh_cache.polygon_edges.empty()
            ? editable_mesh_cache.edges
            : editable_mesh_cache.polygon_edges;
        std::vector<uint32_t> edgeIndices;
        edgeIndices.reserve(edgeList.size() * 2);
        for (const auto& edge : edgeList) {
            if (edge.v0 < 0 || edge.v1 < 0 || edge.v0 >= vertexCount || edge.v1 >= vertexCount) {
                continue;
            }
            edgeIndices.push_back(static_cast<uint32_t>(edge.v0));
            edgeIndices.push_back(static_cast<uint32_t>(edge.v1));
        }

        const size_t faceCount = editable_mesh_cache.polygon_faces.empty()
            ? editable_mesh_cache.faces.size()
            : editable_mesh_cache.polygon_faces.size();
        std::vector<uint32_t> faceIndices;
        faceIndices.reserve(faceCount * 3);
        for (size_t faceId = 0; faceId < faceCount; ++faceId) {
            appendEditOverlayFanIndices(
                faceIndices,
                getEditablePolygonVertexIds(editable_mesh_cache, static_cast<int>(faceId)),
                vertexCount);
        }

        vpb->uploadEditMeshOverlayTopology(edgeIndices, faceIndices);
        sync.cache_revision = editable_mesh_cache.revision;
        sync.geometry_dirty = true;
        sync.selection_hash = ~0ull; // force flag/selection re-upload for the new vertex set
    }

    if (sync.geometry_dirty) {
        std::vector<float> positions;
        positions.reserve(editable_mesh_cache.vertices.size() * 3);
        for (const auto& vertex : editable_mesh_cache.vertices) {
            positions.push_back(vertex.local_position.x);
            positions.push_back(vertex.local_position.y);
            positions.push_back(vertex.local_position.z);
        }
        vpb->uploadEditMeshOverlayGeometry(positions, static_cast<uint32_t>(vertexCount));
        sync.geometry_dirty = false;
    }

    // Fingerprint selection ids + soft-select parameters; rebuild the small
    // flag/selection buffers only when it changes.
    // Combined is a plain multi-element selection mode — suppress the soft-select
    // falloff heat (low-weight neighbours read as a blue blob); only the picked
    // element's own selection colour should show there.
    const bool softActive =
        mesh_overlay_settings.edit_mode && mesh_overlay_settings.proportional_edit &&
        ctx.selection.mesh_element_mode != MeshElementSelectMode::Combined;
    uint64_t selectionHash = 1469598103934665603ull; // FNV-1a basis
    auto mixHash = [&selectionHash](uint64_t value) {
        selectionHash ^= value;
        selectionHash *= 1099511628211ull;
    };
    for (const int id : editable_mesh_cache.selection.vertex_ids) mixHash(static_cast<uint64_t>(id) + 0x100000ull);
    for (const int id : editable_mesh_cache.selection.edge_ids)   mixHash(static_cast<uint64_t>(id) + 0x200000ull);
    for (const int id : editable_mesh_cache.selection.face_ids)   mixHash(static_cast<uint64_t>(id) + 0x300000ull);
    if (softActive) {
        mixHash(0x400000ull);
        mixHash(static_cast<uint64_t>(std::hash<float>{}(mesh_overlay_settings.proportional_radius)));
        mixHash(static_cast<uint64_t>(std::hash<float>{}(mesh_overlay_settings.proportional_falloff)));
        mixHash(static_cast<uint64_t>(mesh_overlay_settings.proportional_falloff_type));
    }
    // Fold the sculpt protection mask in so painting / clearing it re-uploads the
    // overlay flags. version bumps on every mask edit; show_overlay toggles the tint.
    const bool maskOverlayActive =
        sculpt_mode_state.enabled && sculpt_mask_state.show_overlay &&
        sculpt_mask_state.has_any && sculpt_mask_state.object_name == objectName &&
        sculpt_mask_state.values.size() == static_cast<size_t>(vertexCount);
    if (maskOverlayActive) {
        mixHash(0x500000ull);
        mixHash(sculpt_mask_state.version);
    }

    if (selectionHash != sync.selection_hash) {
        std::vector<uint32_t> flags(static_cast<size_t>(vertexCount), 0u);
        std::unordered_set<int> softTargets;
        for (const int id : editable_mesh_cache.selection.vertex_ids) {
            if (isEditableVertexIdValid(editable_mesh_cache, id)) {
                flags[id] |= 1u;
                softTargets.insert(id);
            }
        }

        std::vector<uint32_t> selEdgeIndices;
        selEdgeIndices.reserve(editable_mesh_cache.selection.edge_ids.size() * 2);
        for (const int edgeId : editable_mesh_cache.selection.edge_ids) {
            const auto* edge = getEditableSelectableEdge(editable_mesh_cache, edgeId);
            if (!edge || edge->v0 < 0 || edge->v1 < 0 ||
                edge->v0 >= vertexCount || edge->v1 >= vertexCount) {
                continue;
            }
            selEdgeIndices.push_back(static_cast<uint32_t>(edge->v0));
            selEdgeIndices.push_back(static_cast<uint32_t>(edge->v1));
            softTargets.insert(edge->v0);
            softTargets.insert(edge->v1);
        }

        std::vector<uint32_t> selFaceIndices;
        for (const int faceId : editable_mesh_cache.selection.face_ids) {
            const std::vector<int> vertexIds = getEditablePolygonVertexIds(editable_mesh_cache, faceId);
            appendEditOverlayFanIndices(selFaceIndices, vertexIds, vertexCount);
            for (const int id : vertexIds) {
                if (isEditableVertexIdValid(editable_mesh_cache, id)) {
                    softTargets.insert(id);
                }
            }
        }

        if (softActive && !softTargets.empty()) {
            const std::vector<float> softWeights = buildSoftSelectionWeights(
                editable_mesh_cache, mesh_overlay_settings, softTargets);
            const size_t weightCount = (std::min)(softWeights.size(), flags.size());
            for (size_t i = 0; i < weightCount; ++i) {
                const float w = softWeights[i];
                if (w > 0.0f) {
                    const uint32_t q = static_cast<uint32_t>(
                        (std::min)(1.0f, (std::max)(0.0f, w)) * 255.0f + 0.5f);
                    flags[i] |= (q << 8);
                }
            }
        }

        // Sculpt protection mask: pack the per-vertex weight into bits 16..23 so
        // the overlay shaders can tint frozen regions.
        if (maskOverlayActive) {
            const size_t maskCount = (std::min)(sculpt_mask_state.values.size(), flags.size());
            for (size_t i = 0; i < maskCount; ++i) {
                const float m = sculpt_mask_state.values[i];
                if (m > 0.003f) {
                    const uint32_t q = static_cast<uint32_t>(
                        (std::min)(1.0f, (std::max)(0.0f, m)) * 255.0f + 0.5f);
                    flags[i] |= (q << 16);
                }
            }
        }

        vpb->uploadEditMeshOverlayFlags(flags);
        vpb->uploadEditMeshOverlaySelectionIndices(selEdgeIndices, selFaceIndices);
        sync.selection_hash = selectionHash;
    }

    Backend::EditMeshOverlayParams params;
    params.enabled = true;
    params.drawEdges = drawEdges;
    params.drawPoints = drawVertices;
    params.drawFaces = drawFaces;
    params.softHighlight = softActive;
    params.xray = mesh_overlay_settings.xray_mode;
    params.model = getEditableObjectTransform(editable_mesh_cache);
    params.pointRadiusPx = (std::max)(2.0f, mesh_overlay_settings.vertex_radius + 1.25f);
    vpb->setEditMeshOverlayParams(params);

    sync.active = true;
    sync.drawn_this_frame = true;
    return true;
}

void SceneUI::drawEditableMeshOverlay(UIContext& ctx) {
    if (!mesh_overlay_settings.enabled || !ctx.scene.camera) {
        return;
    }

    // Skip the wireframe/vertex overlay for the WHOLE sculpt workspace, not just
    // while a stroke is active. The brush preview circle + mask tint are the only
    // feedback sculpt needs, and rebuilding/drawing edges & points for 1M+ triangle
    // meshes EVERY frame (even when merely hovering / examining the mesh) is the
    // per-frame "brush cost" on dense meshes. Gating it on the whole mode removes
    // that idle overhead; the mask overlay and brush preview have their own passes.
    if (sculpt_mode_state.enabled &&
        mesh_workspace_mode == MeshWorkspaceMode::Sculpt) {
        // Drop any GPU overlay carried over from edit mode so a stale wireframe
        // doesn't keep rendering now that we skip the per-frame sync below.
        releaseGpuEditMeshOverlay();
        return;
    }

    std::string objectName = active_mesh_edit_object_name;
    if (objectName.empty() &&
        ctx.selection.selected.type == SelectableType::Object &&
        ctx.selection.selected.object) {
        objectName = ctx.selection.selected.object->getNodeName();
    }
    if (objectName.empty()) {
        return;
    }

    if (!mesh_cache_valid) {
        rebuildMeshCache(ctx.scene.world.objects);
    }

    auto cacheIt = mesh_cache.find(objectName);
    if (cacheIt == mesh_cache.end() || cacheIt->second.empty()) {
        return;
    }

    const auto& meshEntries = cacheIt->second;
    const size_t triangleCount = meshEntries.size();
    ensureEditableMeshCache(ctx, objectName);
    const bool onCagePreview = isSubdivisionPreviewActive(ctx, objectName);

    const ImGuiIO& io = ImGui::GetIO();
    const bool editContextMenuAllowed =
        mesh_workspace_mode == MeshWorkspaceMode::Edit &&
        mesh_overlay_settings.edit_mode &&
        ctx.selection.mesh_element_mode != MeshElementSelectMode::Object &&
        !io.WantCaptureMouse &&
        !ImGuizmo::IsOver() &&
        !ImGuizmo::IsUsing();
    if (editContextMenuAllowed && ImGui::IsMouseReleased(ImGuiMouseButton_Right)) {
        ImVec2 delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Right);
        float dragDistSq = delta.x * delta.x + delta.y * delta.y;
        if (dragDistSq < 25.0f) {
            ImGui::OpenPopup("EditMeshContextMenu");
        }
    }
    bool contextMenuMutatedTopology = false;
    if (ImGui::BeginPopup("EditMeshContextMenu")) {
        const MeshElementSelectMode meshMode = ctx.selection.mesh_element_mode;
        const size_t selectedVertexCount = editable_mesh_cache.selection.vertex_ids.size();
        const size_t selectedFaceCount = editable_mesh_cache.selection.face_ids.size();

        const size_t selectedEdgeCountForMenu = editable_mesh_cache.selection.edge_ids.size();
        // Add Face accepts either 3+ picked vertices or 2+ picked edges (the
        // edge endpoints become the face corners).
        const bool canBuildFace = selectedVertexCount >= 3 || selectedEdgeCountForMenu >= 2;

        if (meshMode == MeshElementSelectMode::Vertex) {
            if (selectedVertexCount >= 3 &&
                ImGui::MenuItem("Add Face")) {
                contextMenuMutatedTopology = addFaceFromSelectedVertices(ctx);
                ImGui::CloseCurrentPopup();
            }
            if (selectedVertexCount >= 2 &&
                ImGui::MenuItem("Merge To Center")) {
                contextMenuMutatedTopology = mergeSelectedVerticesToCenter(ctx) || contextMenuMutatedTopology;
                ImGui::CloseCurrentPopup();
            }
            if (selectedVertexCount >= 1 &&
                ImGui::MenuItem("Dissolve Vertex")) {
                contextMenuMutatedTopology = dissolveSelectedVertices(ctx) || contextMenuMutatedTopology;
                ImGui::CloseCurrentPopup();
            }
            if (selectedVertexCount >= 2 &&
                ImGui::MenuItem("Weld by Distance")) {
                contextMenuMutatedTopology =
                    weldSelectedVerticesByDistance(ctx, mesh_vertex_weld_distance) || contextMenuMutatedTopology;
                ImGui::CloseCurrentPopup();
            }
            if (selectedVertexCount == 0) {
                ImGui::TextDisabled("Select vertex");
            } else {
                ImGui::Separator();
                ImGui::TextDisabled("Weld distance: %.4f", mesh_vertex_weld_distance);
            }
        } else if (meshMode == MeshElementSelectMode::Face) {
            if (selectedFaceCount > 0 &&
                ImGui::MenuItem("Extrude Face")) {
                contextMenuMutatedTopology =
                    extrudeSelectedMeshFaces(ctx, mesh_face_extrude_distance) || contextMenuMutatedTopology;
                ImGui::CloseCurrentPopup();
            }
            if (selectedFaceCount > 0 &&
                ImGui::MenuItem("Inset Face")) {
                contextMenuMutatedTopology =
                    insetSelectedMeshFaces(ctx, mesh_face_inset_amount) || contextMenuMutatedTopology;
                ImGui::CloseCurrentPopup();
            }
            if (selectedFaceCount > 0 &&
                ImGui::MenuItem("Delete Face")) {
                contextMenuMutatedTopology = deleteSelectedMeshFaces(ctx) || contextMenuMutatedTopology;
                ImGui::CloseCurrentPopup();
            }
            if (selectedFaceCount == 0) {
                ImGui::TextDisabled("Select face");
            } else {
                ImGui::Separator();
                ImGui::TextDisabled("Extrude distance: %.3f", mesh_face_extrude_distance);
                ImGui::TextDisabled("Inset amount: %.2f", mesh_face_inset_amount);
            }
        } else if (meshMode == MeshElementSelectMode::Edge) {
            const size_t selectedEdgeCount = editable_mesh_cache.selection.edge_ids.size();
            if (selectedEdgeCount >= 2 &&
                ImGui::MenuItem("Add Face")) {
                contextMenuMutatedTopology = addFaceFromSelectedVertices(ctx);
                ImGui::CloseCurrentPopup();
            }
            if (selectedEdgeCount > 0 &&
                ImGui::MenuItem("Loop Cut")) {
                contextMenuMutatedTopology =
                    loopCutSelectedEdges(ctx, mesh_loop_cut_position) || contextMenuMutatedTopology;
                ImGui::CloseCurrentPopup();
            }
            if (selectedEdgeCount > 0 &&
                ImGui::MenuItem("Dissolve Edge")) {
                contextMenuMutatedTopology =
                    dissolveSelectedEdges(ctx) || contextMenuMutatedTopology;
                ImGui::CloseCurrentPopup();
            }
            if (selectedEdgeCount == 0) {
                ImGui::TextDisabled("Select edge");
            } else {
                ImGui::Separator();
                ImGui::TextDisabled("Alt+Click: Loop");
                ImGui::TextDisabled("Shift+Alt+Click: Ring");
                ImGui::TextDisabled("Cut position: %.2f", mesh_loop_cut_position);
            }
        } else if (meshMode == MeshElementSelectMode::Combined) {
            // Combined mode mixes element types — surface whichever tools the
            // current selection supports.
            if (canBuildFace && ImGui::MenuItem("Add Face")) {
                contextMenuMutatedTopology = addFaceFromSelectedVertices(ctx);
                ImGui::CloseCurrentPopup();
            }
            if (selectedVertexCount >= 2 && ImGui::MenuItem("Merge To Center")) {
                contextMenuMutatedTopology = mergeSelectedVerticesToCenter(ctx) || contextMenuMutatedTopology;
                ImGui::CloseCurrentPopup();
            }
            if (selectedFaceCount > 0 && ImGui::MenuItem("Extrude Face")) {
                contextMenuMutatedTopology =
                    extrudeSelectedMeshFaces(ctx, mesh_face_extrude_distance) || contextMenuMutatedTopology;
                ImGui::CloseCurrentPopup();
            }
            if (selectedFaceCount > 0 && ImGui::MenuItem("Inset Face")) {
                contextMenuMutatedTopology =
                    insetSelectedMeshFaces(ctx, mesh_face_inset_amount) || contextMenuMutatedTopology;
                ImGui::CloseCurrentPopup();
            }
            if (selectedFaceCount > 0 && ImGui::MenuItem("Delete Face")) {
                contextMenuMutatedTopology = deleteSelectedMeshFaces(ctx) || contextMenuMutatedTopology;
                ImGui::CloseCurrentPopup();
            }
            if (editable_mesh_cache.selection.edge_ids.size() > 0 && ImGui::MenuItem("Dissolve Edge")) {
                contextMenuMutatedTopology = dissolveSelectedEdges(ctx) || contextMenuMutatedTopology;
                ImGui::CloseCurrentPopup();
            }
            if (selectedVertexCount == 0 && selectedFaceCount == 0 &&
                editable_mesh_cache.selection.edge_ids.empty()) {
                ImGui::TextDisabled("Select an element");
            }
        }

        ImGui::Separator();

        if (ImGui::BeginMenu("Shading")) {
            const bool hasFaces = !editable_mesh_cache.selection.face_ids.empty();
            if (ImGui::MenuItem("Shade Flat", nullptr, false, hasFaces)) {
                applyShadingToSelectedFaces(ctx, true, false);
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("Shade Smooth", nullptr, false, hasFaces)) {
                applyShadingToSelectedFaces(ctx, false, false);
                ImGui::CloseCurrentPopup();
            }
            bool autoSmoothActive = ensureMeshShadingSettings(active_mesh_edit_object_name).auto_smooth;
            if (ImGui::MenuItem("Shade Auto Smooth", nullptr, autoSmoothActive, hasFaces)) {
                applyShadingToSelectedFaces(ctx, false, true);
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Normals")) {
            if (ImGui::MenuItem("Flip")) {
                flipSelectedMeshNormals(ctx);
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("Recalculate Outside")) {
                recalculateMeshNormals(ctx, true);
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::MenuItem("Recalculate Inside")) {
                recalculateMeshNormals(ctx, false);
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndMenu();
        }

        ImGui::EndPopup();
    }

    if (contextMenuMutatedTopology) {
        return;
    }

    const MeshElementSelectMode meshMode = ctx.selection.mesh_element_mode;
    const bool isCombinedMode = (meshMode == MeshElementSelectMode::Combined);
    const bool drawVertices =
        isCombinedMode ||
        meshMode == MeshElementSelectMode::Vertex ||
        (meshMode == MeshElementSelectMode::Object && mesh_overlay_settings.show_vertices);
    // Wireframe draws in every select mode — face mode needs edge outlines to
    // read face boundaries (the dim fill alone is nearly invisible on a shaded
    // surface). The old ImGui-only path skipped edges in face mode for CPU
    // cost; the GPU overlay has no such constraint.
    const bool drawEdges = true;
    const bool drawFaces = isCombinedMode || (meshMode == MeshElementSelectMode::Face);

    // GPU overlay path: the raster viewport backend renders wireframe, vertex
    // markers, face fills and selection highlights with real depth testing.
    // Everything below the sync stays as the Rendered-mode ImGui fallback.
    const bool gpuOverlayActive =
        syncGpuEditMeshOverlay(ctx, objectName, drawVertices, drawEdges, drawFaces);

    // Rendered (path-traced) viewport: skip the ImGui overlay — wireframe and
    // soft-select heat over a pathtrace read as noise, and editing feedback
    // lives in the raster modes on this machine. The ImGui path below remains
    // only as the fallback for machines without the Vulkan raster viewport.
    if (!gpuOverlayActive && viewport_settings.shading_mode == 2 &&
        dynamic_cast<Backend::VulkanBackendAdapter*>(g_viewport_backend.get()) != nullptr) {
        return;
    }

    const bool needsRebuild =
        !gpuOverlayActive &&
        (mesh_overlay_cache.object_name != objectName ||
         mesh_overlay_cache.source_triangle_count != triangleCount);

    if (needsRebuild) {
        mesh_overlay_cache = MeshOverlayCache{};
        mesh_overlay_cache.object_name = objectName;
        mesh_overlay_cache.source_triangle_count = triangleCount;

        const int maxTriangles = (std::max)(1, mesh_overlay_settings.max_overlay_triangles);
        const int triangleStep = triangleCount > static_cast<size_t>(maxTriangles)
            ? static_cast<int>(std::ceil(static_cast<float>(triangleCount) / static_cast<float>(maxTriangles)))
            : 1;

        std::unordered_map<EdgeKey, CachedMeshOverlayEdgeSource, EdgeKeyHasher> uniqueEdges;
        uniqueEdges.reserve((triangleCount / triangleStep) * 2 + 1);

        std::unordered_map<QuantizedVertexKey, CachedMeshOverlayVertexSource, QuantizedVertexKeyHasher> uniqueVertices;
        uniqueVertices.reserve((triangleCount / triangleStep) * 2 + 1);

        for (size_t i = 0; i < triangleCount; i += static_cast<size_t>(triangleStep)) {
            const std::shared_ptr<Triangle>& tri = meshEntries[i].second;
            if (!tri) {
                continue;
            }

            const Matrix4x4 transform = tri->getTransformMatrix();
            const Vec3 v0 = transform * tri->getOriginalVertexPosition(0);
            const Vec3 v1 = transform * tri->getOriginalVertexPosition(1);
            const Vec3 v2 = transform * tri->getOriginalVertexPosition(2);

            uniqueEdges.try_emplace(makeSortedEdgeKey(v0, v1), CachedMeshOverlayEdgeSource{tri, 0, 1});
            uniqueEdges.try_emplace(makeSortedEdgeKey(v1, v2), CachedMeshOverlayEdgeSource{tri, 1, 2});
            uniqueEdges.try_emplace(makeSortedEdgeKey(v2, v0), CachedMeshOverlayEdgeSource{tri, 2, 0});

            uniqueVertices.try_emplace(quantizeVertex(v0), CachedMeshOverlayVertexSource{tri, 0});
            uniqueVertices.try_emplace(quantizeVertex(v1), CachedMeshOverlayVertexSource{tri, 1});
            uniqueVertices.try_emplace(quantizeVertex(v2), CachedMeshOverlayVertexSource{tri, 2});
        }

        mesh_overlay_cache.edges.reserve(uniqueEdges.size());
        for (const auto& [key, edge] : uniqueEdges) {
            (void)key;
            mesh_overlay_cache.edges.push_back(edge);
        }

        mesh_overlay_cache.vertices.reserve(uniqueVertices.size());
        for (const auto& [key, vertex] : uniqueVertices) {
            (void)key;
            mesh_overlay_cache.vertices.push_back(vertex);
        }
    }

    const bool hasEditableTopologyForObject =
        !editable_mesh_cache.object_name.empty() &&
        editable_mesh_cache.object_name == objectName;
    if (mesh_overlay_cache.edges.empty() &&
        mesh_overlay_cache.vertices.empty() &&
        (!hasEditableTopologyForObject || editable_mesh_cache.vertices.empty())) {
        return;
    }

    ImDrawList* drawList = ImGui::GetBackgroundDrawList();
    const ImVec2 displaySize = ImGui::GetIO().DisplaySize;
    const bool isOrtho = ctx.scene.camera->orthographic && viewport_settings.shading_mode != 2;

    const ImU32 edgeColor = onCagePreview
        ? ((meshMode == MeshElementSelectMode::Edge)
            ? IM_COL32(214, 224, 236, 255)
            : IM_COL32(198, 210, 222, 242))
        : ((meshMode == MeshElementSelectMode::Edge)
            ? IM_COL32(176, 186, 196, 245)
            : IM_COL32(156, 166, 176, 226));
    const ImU32 vertexColor = IM_COL32(158, 166, 174, 228);
    const ImU32 vertexOutline = IM_COL32(18, 22, 26, 220);
    const ImU32 faceColor = IM_COL32(120, 132, 144, 34);
    const ImU32 selectedVertexColor = IM_COL32(72, 186, 214, 255);
    const ImU32 selectedEdgeColor = IM_COL32(72, 186, 214, 255);
    const ImU32 selectedFaceColor = IM_COL32(72, 186, 214, 122);
    const ImU32 softRadiusColor = IM_COL32(72, 186, 214, 70);
    const float vertexRadius = (std::max)(1.0f, mesh_overlay_settings.vertex_radius);
    const Matrix4x4 editableTransform =
        hasEditableTopologyForObject ? getEditableObjectTransform(editable_mesh_cache) : Matrix4x4::identity();
    std::vector<ImVec2> editableProjectedVertices;
    std::vector<uint8_t> editableProjectedValid;
    bool editableProjectionCacheReady = false;
    auto ensureEditableProjectionCache = [&]() -> bool {
        if (!hasEditableTopologyForObject || editable_mesh_cache.vertices.empty()) {
            return false;
        }
        if (editableProjectionCacheReady) {
            return true;
        }

        editableProjectedVertices.resize(editable_mesh_cache.vertices.size());
        editableProjectedValid.assign(editable_mesh_cache.vertices.size(), 0u);
        for (size_t i = 0; i < editable_mesh_cache.vertices.size(); ++i) {
            ImVec2 screenPosition;
            if (projectPointToScreen(
                    *ctx.scene.camera,
                    displaySize,
                    editableTransform.transform_point(editable_mesh_cache.vertices[i].local_position),
                    screenPosition,
                    isOrtho)) {
                editableProjectedVertices[i] = screenPosition;
                editableProjectedValid[i] = 1u;
            }
        }
        editableProjectionCacheReady = true;
        return true;
    };
    auto tryGetEditableProjectedVertex = [&](int vertexId, ImVec2& out) -> bool {
        if (vertexId < 0 || vertexId >= static_cast<int>(editable_mesh_cache.vertices.size())) {
            return false;
        }
        if (!ensureEditableProjectionCache()) {
            return false;
        }
        if (editableProjectedValid[vertexId] == 0u) {
            return false;
        }
        out = editableProjectedVertices[vertexId];
        return true;
    };

    if (!gpuOverlayActive && drawFaces) {
        const size_t overlayFaceCount =
            (!editable_mesh_cache.object_name.empty() && editable_mesh_cache.object_name == objectName &&
             !editable_mesh_cache.polygon_faces.empty())
                ? editable_mesh_cache.polygon_faces.size()
                : meshEntries.size();
        const int maxFaces = (std::max)(1, mesh_overlay_settings.max_overlay_triangles / 6);
        const size_t faceStep = overlayFaceCount > static_cast<size_t>(maxFaces)
            ? static_cast<size_t>(std::ceil(static_cast<float>(overlayFaceCount) / static_cast<float>(maxFaces)))
            : static_cast<size_t>(1);

        if (!editable_mesh_cache.object_name.empty() &&
            editable_mesh_cache.object_name == objectName &&
            !editable_mesh_cache.polygon_faces.empty()) {
            ensureEditableProjectionCache();
            for (size_t polygonFaceId = 0; polygonFaceId < editable_mesh_cache.polygon_faces.size(); polygonFaceId += faceStep) {
                const std::vector<int> vertexIds =
                    getEditablePolygonVertexIds(editable_mesh_cache, static_cast<int>(polygonFaceId));
                if (vertexIds.size() < 3) {
                    continue;
                }

                std::vector<ImVec2> screenVertices;
                screenVertices.reserve(vertexIds.size());
                bool projectionFailed = false;
                for (const int vertexId : vertexIds) {
                    ImVec2 screenPosition;
                    if (!tryGetEditableProjectedVertex(vertexId, screenPosition)) {
                        projectionFailed = true;
                        break;
                    }
                    screenVertices.push_back(screenPosition);
                }
                if (projectionFailed) {
                    continue;
                }

                const Vec3 world0 = editableTransform.transform_point(editable_mesh_cache.vertices[vertexIds[0]].local_position);
                const Vec3 world1 = editableTransform.transform_point(editable_mesh_cache.vertices[vertexIds[1]].local_position);
                const Vec3 world2 = editableTransform.transform_point(editable_mesh_cache.vertices[vertexIds[2]].local_position);
                const float visibility = computeOverlayVisibility(*ctx.scene.camera, world0, world1, world2);
                drawEditableScreenPolygonFill(
                    drawList,
                    screenVertices,
                    scaleColorAlpha(faceColor, visibility));
            }
        } else {
            for (size_t i = 0; i < meshEntries.size(); i += faceStep) {
                const std::shared_ptr<Triangle>& tri = meshEntries[i].second;
                if (!tri) {
                    continue;
                }

                const Matrix4x4 transform = tri->getTransformMatrix();
                const Vec3 v0 = transform * tri->getOriginalVertexPosition(0);
                const Vec3 v1 = transform * tri->getOriginalVertexPosition(1);
                const Vec3 v2 = transform * tri->getOriginalVertexPosition(2);
                const float visibility = computeOverlayVisibility(*ctx.scene.camera, v0, v1, v2);
                ImVec2 s0, s1, s2;
                if (!projectPointToScreen(*ctx.scene.camera, displaySize, v0, s0, isOrtho) ||
                    !projectPointToScreen(*ctx.scene.camera, displaySize, v1, s1, isOrtho) ||
                    !projectPointToScreen(*ctx.scene.camera, displaySize, v2, s2, isOrtho)) {
                    continue;
                }
                drawList->AddTriangleFilled(s0, s1, s2, scaleColorAlpha(faceColor, visibility));
            }
        }
    }

    if (!gpuOverlayActive && drawEdges) {
        const bool usePolygonEdges =
            (meshMode == MeshElementSelectMode::Face ||
             meshMode == MeshElementSelectMode::Edge ||
             meshMode == MeshElementSelectMode::Vertex) &&
            !editable_mesh_cache.object_name.empty() &&
            editable_mesh_cache.object_name == objectName &&
            !editable_mesh_cache.polygon_faces.empty();
        if (usePolygonEdges) {
            ensureEditableProjectionCache();
            const auto& weightedEdges =
                !editable_mesh_cache.polygon_edges.empty()
                    ? editable_mesh_cache.polygon_edges
                    : editable_mesh_cache.edges;
            const int maxOverlayEdges = (std::max)(1, mesh_overlay_settings.max_overlay_triangles * 2);
            const size_t edgeStep = weightedEdges.size() > static_cast<size_t>(maxOverlayEdges)
                ? static_cast<size_t>(std::ceil(static_cast<float>(weightedEdges.size()) / static_cast<float>(maxOverlayEdges)))
                : static_cast<size_t>(1);

            for (size_t edgeIndex = 0; edgeIndex < weightedEdges.size(); edgeIndex += edgeStep) {
                const auto& edge = weightedEdges[edgeIndex];
                if (edge.v0 < 0 || edge.v1 < 0 ||
                    edge.v0 >= static_cast<int>(editable_mesh_cache.vertices.size()) ||
                    edge.v1 >= static_cast<int>(editable_mesh_cache.vertices.size())) {
                    continue;
                }

                ImVec2 a, b;
                if (!tryGetEditableProjectedVertex(edge.v0, a) ||
                    !tryGetEditableProjectedVertex(edge.v1, b)) {
                    continue;
                }

                drawList->AddLine(a, b, edgeColor, mesh_overlay_settings.edge_thickness);
            }
        } else {
            const int maxOverlayEdges = (std::max)(1, mesh_overlay_settings.max_overlay_triangles * 2);
            const size_t edgeStep = mesh_overlay_cache.edges.size() > static_cast<size_t>(maxOverlayEdges)
                ? static_cast<size_t>(std::ceil(static_cast<float>(mesh_overlay_cache.edges.size()) / static_cast<float>(maxOverlayEdges)))
                : static_cast<size_t>(1);
            for (size_t edgeIndex = 0; edgeIndex < mesh_overlay_cache.edges.size(); edgeIndex += edgeStep) {
                const auto& edge = mesh_overlay_cache.edges[edgeIndex];
                if (!edge.triangle) {
                    continue;
                }

                const Matrix4x4 transform = edge.triangle->getTransformMatrix();
                const Vec3 a = transform * edge.triangle->getOriginalVertexPosition(edge.a);
                const Vec3 b = transform * edge.triangle->getOriginalVertexPosition(edge.b);
                const Vec3 c = transform * edge.triangle->getOriginalVertexPosition(3 - edge.a - edge.b);
                const float visibility = computeOverlayVisibility(*ctx.scene.camera, a, b, c);
                ImVec2 sa, sb;
                if (!projectPointToScreen(*ctx.scene.camera, displaySize, a, sa, isOrtho) ||
                    !projectPointToScreen(*ctx.scene.camera, displaySize, b, sb, isOrtho)) {
                    continue;
                }

                drawList->AddLine(sa, sb, scaleColorAlpha(edgeColor, visibility), mesh_overlay_settings.edge_thickness);
            }
        }
    }

    if (!gpuOverlayActive && drawVertices) {
        const int maxVertexMarkers = (std::max)(1, mesh_overlay_settings.max_vertex_markers);
        if (hasEditableTopologyForObject && !editable_mesh_cache.vertices.empty()) {
            ensureEditableProjectionCache();
            const size_t vertexStep = editable_mesh_cache.vertices.size() > static_cast<size_t>(maxVertexMarkers)
                ? static_cast<size_t>(std::ceil(static_cast<float>(editable_mesh_cache.vertices.size()) / static_cast<float>(maxVertexMarkers)))
                : static_cast<size_t>(1);

            for (size_t i = 0; i < editable_mesh_cache.vertices.size(); i += vertexStep) {
                const auto& vertex = editable_mesh_cache.vertices[i];
                if (vertex.refs.empty()) {
                    continue;
                }

                ImVec2 screen;
                if (!tryGetEditableProjectedVertex(static_cast<int>(i), screen)) {
                    continue;
                }

                drawList->AddCircleFilled(screen, vertexRadius + 1.0f, vertexOutline, 8);
                drawList->AddCircleFilled(screen, vertexRadius, vertexColor, 8);
            }
        } else {
            const size_t vertexStep = mesh_overlay_cache.vertices.size() > static_cast<size_t>(maxVertexMarkers)
                ? static_cast<size_t>(std::ceil(static_cast<float>(mesh_overlay_cache.vertices.size()) / static_cast<float>(maxVertexMarkers)))
                : static_cast<size_t>(1);

            for (size_t i = 0; i < mesh_overlay_cache.vertices.size(); i += vertexStep) {
                const auto& vertex = mesh_overlay_cache.vertices[i];
                if (!vertex.triangle) {
                    continue;
                }

                const Matrix4x4 transform = vertex.triangle->getTransformMatrix();
                const Vec3 p = transform * vertex.triangle->getOriginalVertexPosition(vertex.index);
                const Vec3 p0 = transform * vertex.triangle->getOriginalVertexPosition(0);
                const Vec3 p1 = transform * vertex.triangle->getOriginalVertexPosition(1);
                const Vec3 p2 = transform * vertex.triangle->getOriginalVertexPosition(2);
                const float visibility = computeOverlayVisibility(*ctx.scene.camera, p0, p1, p2);
                ImVec2 screen;
                if (!projectPointToScreen(*ctx.scene.camera, displaySize, p, screen, isOrtho)) {
                    continue;
                }

                drawList->AddCircleFilled(screen, vertexRadius + 1.0f, scaleColorAlpha(vertexOutline, visibility), 8);
                drawList->AddCircleFilled(screen, vertexRadius, scaleColorAlpha(vertexColor, visibility), 8);
            }
        }
    }

    if (!editable_mesh_cache.object_name.empty() &&
        editable_mesh_cache.object_name == objectName) {
        // GPU overlay already renders selection highlights + soft-select heat;
        // only the proportional radius circle further below stays on ImGui.
        if (!gpuOverlayActive) {
        ensureEditableProjectionCache();
        std::unordered_set<int> softTargets;
        for (const int vertexId : editable_mesh_cache.selection.vertex_ids) {
            softTargets.insert(vertexId);
        }
        for (const int edgeId : editable_mesh_cache.selection.edge_ids) {
            const auto* edge = getEditableSelectableEdge(editable_mesh_cache, edgeId);
            if (!edge) {
                continue;
            }
            softTargets.insert(edge->v0);
            softTargets.insert(edge->v1);
        }
        for (const int faceId : editable_mesh_cache.selection.face_ids) {
            const std::vector<int> vertexIds = getEditablePolygonVertexIds(editable_mesh_cache, faceId);
            softTargets.insert(vertexIds.begin(), vertexIds.end());
        }
        const std::vector<float> softWeights = buildSoftSelectionWeights(
            editable_mesh_cache, mesh_overlay_settings, softTargets);

        for (const int faceId : editable_mesh_cache.selection.face_ids) {
            const std::vector<int> vertexIds = getEditablePolygonVertexIds(editable_mesh_cache, faceId);
            if (vertexIds.size() < 3) {
                continue;
            }
            std::vector<ImVec2> screenVertices;
            screenVertices.reserve(vertexIds.size());
            bool projectionFailed = false;
            for (const int vertexId : vertexIds) {
                ImVec2 screenVertex;
                if (!tryGetEditableProjectedVertex(vertexId, screenVertex)) {
                    projectionFailed = true;
                    break;
                }
                screenVertices.push_back(screenVertex);
            }
            if (projectionFailed) {
                continue;
            }

            drawEditableScreenPolygonFill(drawList, screenVertices, selectedFaceColor);
            for (size_t edgeIndex = 0; edgeIndex < screenVertices.size(); ++edgeIndex) {
                const ImVec2 a = screenVertices[edgeIndex];
                const ImVec2 b = screenVertices[(edgeIndex + 1) % screenVertices.size()];
                drawList->AddLine(a, b, selectedEdgeColor, mesh_overlay_settings.edge_thickness + 0.75f);
            }
        }

        for (const int edgeId : editable_mesh_cache.selection.edge_ids) {
            const auto* edge = getEditableSelectableEdge(editable_mesh_cache, edgeId);
            if (!edge) {
                continue;
            }
            if (edge->v0 >= 0 && edge->v1 >= 0 &&
                edge->v0 < static_cast<int>(editable_mesh_cache.vertices.size()) &&
                edge->v1 < static_cast<int>(editable_mesh_cache.vertices.size())) {
                ImVec2 s0, s1;
                if (tryGetEditableProjectedVertex(edge->v0, s0) &&
                    tryGetEditableProjectedVertex(edge->v1, s1)) {
                    drawList->AddLine(s0, s1, selectedEdgeColor, mesh_overlay_settings.edge_thickness + 1.35f);
                }
            }
        }

        for (const int vertexId : editable_mesh_cache.selection.vertex_ids) {
            if (vertexId < 0 || vertexId >= static_cast<int>(editable_mesh_cache.vertices.size())) {
                continue;
            }
            ImVec2 screen;
            if (tryGetEditableProjectedVertex(vertexId, screen)) {
                drawList->AddCircleFilled(screen, vertexRadius + 2.0f, vertexOutline, 10);
                drawList->AddCircleFilled(screen, vertexRadius + 0.8f, selectedVertexColor, 10);
            }
        }

        if (mesh_overlay_settings.edit_mode &&
            mesh_overlay_settings.proportional_edit &&
            !softTargets.empty()) {
            if (ctx.selection.mesh_element_mode == MeshElementSelectMode::Face) {
                const size_t polygonFaceCount =
                    editable_mesh_cache.polygon_faces.empty()
                        ? editable_mesh_cache.faces.size()
                        : editable_mesh_cache.polygon_faces.size();
                for (size_t polygonFaceId = 0; polygonFaceId < polygonFaceCount; ++polygonFaceId) {
                    const std::vector<int> vertexIds =
                        getEditablePolygonVertexIds(editable_mesh_cache, static_cast<int>(polygonFaceId));
                    if (vertexIds.size() < 3) {
                        continue;
                    }

                    float faceWeight = 0.0f;
                    int faceWeightSamples = 0;
                    for (const int vertexId : vertexIds) {
                        if (vertexId >= 0 && vertexId < static_cast<int>(softWeights.size())) {
                            faceWeight += softWeights[vertexId];
                            ++faceWeightSamples;
                        }
                    }
                    if (faceWeightSamples == 0) {
                        continue;
                    }
                    faceWeight /= static_cast<float>(faceWeightSamples);
                    if (faceWeight <= 0.01f) {
                        continue;
                    }

                    std::vector<ImVec2> screenVertices;
                    screenVertices.reserve(vertexIds.size());
                    bool projectionFailed = false;
                    for (const int vertexId : vertexIds) {
                        ImVec2 screenVertex;
                        if (!tryGetEditableProjectedVertex(vertexId, screenVertex)) {
                            projectionFailed = true;
                            break;
                        }
                        screenVertices.push_back(screenVertex);
                    }
                    if (projectionFailed) {
                        continue;
                    }

                    drawEditableScreenPolygonFill(drawList, screenVertices, weightToColor(faceWeight, 110));
                }
            } else if (ctx.selection.mesh_element_mode == MeshElementSelectMode::Edge) {
                const auto& weightedEdges =
                    !editable_mesh_cache.polygon_edges.empty()
                        ? editable_mesh_cache.polygon_edges
                        : editable_mesh_cache.edges;
                for (const auto& edge : weightedEdges) {
                    if (edge.v0 < 0 || edge.v1 < 0 ||
                        edge.v0 >= static_cast<int>(softWeights.size()) ||
                        edge.v1 >= static_cast<int>(softWeights.size())) {
                        continue;
                    }

                    const float edgeWeight = (softWeights[edge.v0] + softWeights[edge.v1]) * 0.5f;
                    if (edgeWeight <= 0.01f) {
                        continue;
                    }

                    ImVec2 s0, s1;
                    if (!tryGetEditableProjectedVertex(edge.v0, s0) ||
                        !tryGetEditableProjectedVertex(edge.v1, s1)) {
                        continue;
                    }

                    drawList->AddLine(s0, s1, weightToColor(edgeWeight, 230), mesh_overlay_settings.edge_thickness + edgeWeight * 1.6f);
                }
            } else if (ctx.selection.mesh_element_mode == MeshElementSelectMode::Vertex) {
                for (size_t i = 0; i < editable_mesh_cache.vertices.size(); ++i) {
                    const float weight = softWeights[i];
                    if (weight <= 0.01f) {
                        continue;
                    }

                    ImVec2 screen;
                    if (!tryGetEditableProjectedVertex(static_cast<int>(i), screen)) {
                        continue;
                    }

                    drawList->AddCircleFilled(screen, vertexRadius + weight * 2.0f, weightToColor(weight, 235), 10);
                }
            }
        }

        } // !gpuOverlayActive (ImGui selection/soft-select fallback)

        if (mesh_overlay_settings.show_normals) {
            ensureEditableProjectionCache();
            for (const int faceId : editable_mesh_cache.selection.face_ids) {
                const std::vector<int> vertexIds = getEditablePolygonVertexIds(editable_mesh_cache, faceId);
                if (vertexIds.size() < 3) {
                    continue;
                }

                Vec3 localCenter(0.0f, 0.0f, 0.0f);
                for (int vId : vertexIds) {
                    if (vId >= 0 && vId < static_cast<int>(editable_mesh_cache.vertices.size())) {
                        localCenter += editable_mesh_cache.vertices[vId].local_position;
                    }
                }
                localCenter = localCenter / static_cast<float>(vertexIds.size());

                if (vertexIds[0] >= 0 && vertexIds[1] >= 0 && vertexIds[2] >= 0 &&
                    vertexIds[0] < static_cast<int>(editable_mesh_cache.vertices.size()) &&
                    vertexIds[1] < static_cast<int>(editable_mesh_cache.vertices.size()) &&
                    vertexIds[2] < static_cast<int>(editable_mesh_cache.vertices.size())) {

                    Vec3 localNormal = Vec3::cross(
                        editable_mesh_cache.vertices[vertexIds[1]].local_position - editable_mesh_cache.vertices[vertexIds[0]].local_position,
                        editable_mesh_cache.vertices[vertexIds[2]].local_position - editable_mesh_cache.vertices[vertexIds[0]].local_position);
                    float localNormalLen = localNormal.length();
                    if (localNormalLen > 1e-6f) {
                        localNormal = localNormal / localNormalLen;
                    }

                    // Align face normal direction with average vertex normal
                    Vec3 referenceNormal(0.0f, 0.0f, 0.0f);
                    for (int vId : vertexIds) {
                        if (vId >= 0 && vId < static_cast<int>(editable_mesh_cache.vertices.size())) {
                            for (const auto& ref : editable_mesh_cache.vertices[vId].refs) {
                                if (const Triangle* rt = editable_mesh_cache.refTri(ref)) {
                                    referenceNormal += rt->getOriginalVertexNormal(ref.corner);
                                    break;
                                }
                            }
                        }
                    }
                    if (referenceNormal.length_squared() > 1e-8f && localNormal.dot(referenceNormal) < 0.0f) {
                        localNormal = -localNormal;
                    }

                    Vec3 worldCenter = editableTransform.transform_point(localCenter);
                    Vec3 worldNormal = editableTransform.transform_vector(localNormal).normalize();
                    Vec3 worldEnd = worldCenter + worldNormal * mesh_overlay_settings.normals_length;

                    ImVec2 screenStart, screenEnd;
                    if (projectPointToScreen(*ctx.scene.camera, displaySize, worldCenter, screenStart, isOrtho) &&
                        projectPointToScreen(*ctx.scene.camera, displaySize, worldEnd, screenEnd, isOrtho)) {
                        ImU32 normalColor = IM_COL32(0, 255, 255, 255); // Neon Cyan/Blue
                        drawList->AddLine(screenStart, screenEnd, normalColor, 1.5f);
                        drawList->AddCircleFilled(screenEnd, 2.0f, normalColor, 6);
                    }
                }
            }
        }

        if (mesh_overlay_settings.edit_mode && mesh_overlay_settings.proportional_edit) {
            bool hasCenter = false;
            const Vec3 centerWorld = getSelectedMeshElementWorldPosition(ctx, &hasCenter);
            if (hasCenter) {
                ImVec2 centerScreen;
                if (projectPointToScreen(*ctx.scene.camera, displaySize, centerWorld, centerScreen, isOrtho)) {
                    // Approximate projected proportional radius using a point offset in local/world X.
                    Vec3 radiusWorld = centerWorld + Vec3(mesh_overlay_settings.proportional_radius, 0.0f, 0.0f);
                    ImVec2 radiusScreen;
                    if (projectPointToScreen(*ctx.scene.camera, displaySize, radiusWorld, radiusScreen, isOrtho)) {
                        const float radiusPx = (std::max)(8.0f, std::fabs(radiusScreen.x - centerScreen.x));
                        drawList->AddCircle(centerScreen, radiusPx, softRadiusColor, 48, 1.2f);
                    }
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SCULPT BRUSH VIEWPORT PREVIEW
// Draws an alpha-mask grid on the mesh surface + outer ring + inner falloff ring,
// matching the visual language of the mesh-paint brush preview.
// ─────────────────────────────────────────────────────────────────────────────
void SceneUI::drawSculptBrushViewportPreview(UIContext& ctx, const HitRecord& hit, bool ghost)
{
    if (!ctx.scene.camera) return;

    const Paint::BrushSettings& brush = sculpt_mode_state.brush;
    float world_radius = (std::max)(0.0001f, brush.radius);
    if (sculpt_mode_state.use_screen_space_radius && ctx.scene.camera) {
        const float radiusPx = sanitizeFiniteFloat(sculpt_mode_state.screen_radius_px, 72.0f, 1.0f, 1000.0f);
        const float estimatedRadius = estimateBrushWorldRadius(
            *ctx.scene.camera,
            ImGui::GetIO().DisplaySize,
            hit.point,
            radiusPx);
        if (std::isfinite(estimatedRadius) && estimatedRadius > 1e-5f) {
            world_radius = estimatedRadius;
        }
    }

    // Build a VIEW-STABLE tangent frame (camera right/up projected onto the surface tangent
    // plane), matching handleMeshSculpt so the cursor outline orients exactly as the deform.
    const Vec3 normal = hit.normal.length_squared() > 1e-8f ? hit.normal.normalize() : Vec3(0, 1, 0);
    Vec3 tangent;
    Vec3 bitangent;
    {
        Vec3 camRight(1.0f, 0.0f, 0.0f);
        Vec3 camUp(0.0f, 1.0f, 0.0f);
        const Vec3 fwdRaw = ctx.scene.camera->lookat - ctx.scene.camera->lookfrom;
        const Vec3 fwd = fwdRaw.length_squared() > 1e-12f ? fwdRaw.normalize() : Vec3(0.0f, 0.0f, -1.0f);
        Vec3 r = fwd.cross(ctx.scene.camera->vup);
        if (r.length_squared() < 1e-12f) r = Vec3(1.0f, 0.0f, 0.0f);
        camRight = r.normalize();
        camUp = camRight.cross(fwd).normalize();
        Vec3 t = camRight - normal * camRight.dot(normal);
        if (t.length_squared() < 1e-8f) t = camUp - normal * camUp.dot(normal);
        if (t.length_squared() < 1e-8f) {
            t = std::abs(normal.y) < 0.95f ? normal.cross(Vec3(0, 1, 0)) : normal.cross(Vec3(1, 0, 0));
        }
        tangent = t.normalize();
        bitangent = normal.cross(tangent).normalize();
    }

    ImGuiIO& io = ImGui::GetIO();
    Camera& cam = *ctx.scene.camera;
    const float win_w = io.DisplaySize.x;
    const float win_h = io.DisplaySize.y;

    const bool isOrtho = cam.orthographic && viewport_settings.shading_mode != 2;
    auto project = [&](const Vec3& p) -> ImVec2 {
        const Vec3 cam_forward = (cam.lookat - cam.lookfrom).normalize();
        const Vec3 cam_right   = cam_forward.cross(cam.vup).normalize();
        const Vec3 cam_up      = cam_right.cross(cam_forward).normalize();
        const float fov_rad    = cam.vfov * 3.14159f / 180.0f;
        const Vec3 to_p = p - cam.lookfrom;
        const float depth = to_p.dot(cam_forward);
        if (!isOrtho && depth <= 0.1f) return ImVec2(-1000.0f, -1000.0f);
        
        float half_h, half_w;
        if (isOrtho) {
            half_h = cam.ortho_height * 0.5f;
            half_w = half_h * cam.aspect_ratio;
        } else {
            half_h = depth * tanf(fov_rad * 0.5f);
            half_w = half_h * cam.aspect_ratio;
        }
        
        if (std::fabs(half_w) <= 1e-6f || std::fabs(half_h) <= 1e-6f) {
            return ImVec2(-1000.0f, -1000.0f);
        }

        const float lx = to_p.dot(cam_right);
        const float ly = to_p.dot(cam_up);
        return ImVec2(
            (lx / half_w * 0.5f + 0.5f) * win_w,
            (0.5f - ly / half_h * 0.5f) * win_h);
    };

    ImDrawList* dl = ImGui::GetForegroundDrawList();
    constexpr int segments = 40;

    // Approximate screen radius for adaptive grid density
    const ImVec2 center_screen = project(hit.point + normal * 0.002f);
    const ImVec2 radius_screen = project(hit.point + tangent * world_radius + normal * 0.002f);
    if (center_screen.x < -900.0f) return;
    const float approx_screen_radius = (radius_screen.x < -900.0f)
        ? 32.0f
        : std::sqrt(
            (radius_screen.x - center_screen.x) * (radius_screen.x - center_screen.x) +
            (radius_screen.y - center_screen.y) * (radius_screen.y - center_screen.y));

    // Footprint bound: the preview grid + rings must reach the shape's corners (a square at
    // sqrt(2), aspect-elongated further), otherwise the display clips back to a circle — which
    // is exactly the "brush display stays circular" the user saw. boundScale = 1 for Circle.
    const float previewBoundScale = uiBrushFootprintBoundScale(brush);

    // --- Alpha grid (skip for ghost pass to keep it lightweight) ---
    if (!ghost) {
        const int grid = std::clamp(static_cast<int>(approx_screen_radius * 1.1f), 24, 80);
        const float cell_span = (2.0f * previewBoundScale) / static_cast<float>(grid);

        for (int gy = 0; gy < grid; ++gy) {
            for (int gx = 0; gx < grid; ++gx) {
                // Span [-boundScale, boundScale] so elongated/cornered footprints are covered.
                const float nx = (((static_cast<float>(gx) + 0.5f) / static_cast<float>(grid)) * 2.0f - 1.0f) * previewBoundScale;
                const float ny = (((static_cast<float>(gy) + 0.5f) / static_cast<float>(grid)) * 2.0f - 1.0f) * previewBoundScale;

                // Exact deform weight: shape cutoff + shape-edge falloff × alpha. Matches the
                // SIMD evaluator and the Alpha Preview pixel-for-pixel.
                const float alpha = uiSampleBrushFootprintWeight(brush, nx, ny);
                if (alpha <= 0.025f) continue;

                const float local_x = nx * world_radius;
                const float local_y = ny * world_radius;
                const float half_cell = world_radius * cell_span * 0.42f;
                const Vec3 cell_center = hit.point + tangent * local_x - bitangent * local_y + normal * 0.002f;
                const Vec3 right_vec = tangent   * half_cell;
                const Vec3 up_vec    = bitangent * half_cell;
                const ImVec2 p0 = project(cell_center - right_vec - up_vec);
                const ImVec2 p1 = project(cell_center + right_vec - up_vec);
                const ImVec2 p2 = project(cell_center + right_vec + up_vec);
                const ImVec2 p3 = project(cell_center - right_vec + up_vec);
                if (p0.x < -900.0f || p1.x < -900.0f || p2.x < -900.0f || p3.x < -900.0f) continue;
                const int cell_a = static_cast<int>(alpha * 140.0f);
                const ImVec2 pts[4] = { p0, p1, p2, p3 };
                dl->AddConvexPolyFilled(pts, 4, IM_COL32(255, 170, 64, cell_a));
            }
        }
    }

    // --- Rings: trace the actual shape boundary (outer = dist_norm 1, inner = falloff edge) ---
    // so the outline matches the deformed footprint instead of always drawing a circle. Same
    // brush-local→world mapping as the grid (tangent·u − bitangent·v). Circle is unchanged
    // (dist_norm == r), so circular brushes look exactly as before.
    auto draw_shape_ring = [&](float targetDist, ImU32 color, float thickness) {
        ImVec2 prev;
        bool has_prev = false;
        for (int i = 0; i <= segments; ++i) {
            const float angle = (static_cast<float>(i) / static_cast<float>(segments)) * 6.2831853f;
            const float du = std::cos(angle), dv = std::sin(angle);
            // Binary-search the radius along this ray where the footprint distance hits target
            // (dist_norm is monotonic in r for these convex shapes).
            float lo = 0.0f, hi = previewBoundScale * 1.25f + 0.05f;
            for (int it = 0; it < 22; ++it) {
                const float mid = 0.5f * (lo + hi);
                if (uiBrushFootprintDistNorm(brush, mid * du, mid * dv) < targetDist) lo = mid; else hi = mid;
            }
            const float r = 0.5f * (lo + hi);
            const Vec3 offset = tangent * (r * du * world_radius) - bitangent * (r * dv * world_radius);
            const ImVec2 p = project(hit.point + offset + normal * 0.002f);
            if (p.x > -900.0f) {
                if (has_prev) dl->AddLine(prev, p, color, thickness);
                prev = p;
                has_prev = true;
            } else {
                has_prev = false;
            }
        }
    };

    const float inner_target = std::clamp(1.0f - brush.falloff, 0.15f, 0.95f);
    const ImU32 outer_color = ghost ? IM_COL32(255, 170, 64, 100) : IM_COL32(255, 170, 64, 230);
    const ImU32 inner_color = ghost ? IM_COL32(255, 170, 64,  60) : IM_COL32(255, 170, 64, 120);
    draw_shape_ring(1.0f,         outer_color, ghost ? 1.2f : 2.0f);
    draw_shape_ring(inner_target, inner_color, 1.0f);

    // Center dot
    if (!ghost) {
        dl->AddCircleFilled(center_screen, 3.5f, IM_COL32(255, 170, 64, 230));
    }
}

void SceneUI::drawSculptMaskViewportOverlay(UIContext& ctx) {
    if (!ctx.scene.camera) {
        return;
    }
    if (!sculpt_mode_state.enabled || mesh_workspace_mode != MeshWorkspaceMode::Sculpt) {
        return;
    }
    if (!sculpt_mask_state.show_overlay || !sculpt_mask_state.has_any) {
        return;
    }
    const std::vector<EditableVertex>& verts = editable_mesh_cache.vertices;
    const size_t n = verts.size();
    if (n == 0 || sculpt_mask_state.values.size() != n) {
        return;
    }
    if (editable_mesh_cache.object_name != sculpt_mode_state.active_target_name) {
        return;
    }

    const Matrix4x4 transform = getEditableObjectTransform(editable_mesh_cache);
    Camera& cam = *ctx.scene.camera;
    const ImGuiIO& io = ImGui::GetIO();
    const float win_w = io.DisplaySize.x;
    const float win_h = io.DisplaySize.y;
    const Vec3 cam_forward = (cam.lookat - cam.lookfrom).normalize();
    const Vec3 cam_right = cam_forward.cross(cam.vup).normalize();
    const Vec3 cam_up = cam_right.cross(cam_forward).normalize();
    const float fov_rad = cam.vfov * 3.14159265f / 180.0f;
    const bool isOrtho = cam.orthographic && viewport_settings.shading_mode != 2;
    auto project = [&](const Vec3& p) -> ImVec2 {
        const Vec3 to_p = p - cam.lookfrom;
        const float depth = to_p.dot(cam_forward);
        if (!isOrtho && depth <= 0.1f) {
            return ImVec2(-10000.0f, -10000.0f);
        }
        
        float half_h, half_w;
        if (isOrtho) {
            half_h = cam.ortho_height * 0.5f;
            half_w = half_h * cam.aspect_ratio;
        } else {
            half_h = depth * tanf(fov_rad * 0.5f);
            half_w = half_h * cam.aspect_ratio;
        }
        
        if (std::fabs(half_w) <= 1e-6f || std::fabs(half_h) <= 1e-6f) {
            return ImVec2(-10000.0f, -10000.0f);
        }

        const float lx = to_p.dot(cam_right);
        const float ly = to_p.dot(cam_up);
        return ImVec2(
            (lx / half_w * 0.5f + 0.5f) * win_w,
            (0.5f - ly / half_h * 0.5f) * win_h);
    };

    const Vec3 camPos = cam.lookfrom;
    auto vertexScreen = [&](size_t vid, ImVec2& out) -> bool {
        const Vec3 world = transform.transform_point(verts[vid].local_position);
        const ImVec2 sp = project(world);
        out = sp;
        return sp.x > -9000.0f &&
               sp.x >= -64.0f && sp.y >= -64.0f &&
               sp.x <= win_w + 64.0f && sp.y <= win_h + 64.0f;
    };

    // Render masked triangles as a per-vertex-coloured fill so the falloff reads
    // as a continuous gradient on the surface (vertices with mask 0 contribute a
    // transparent corner). ImGui's low-level PrimVtx interpolates the corner
    // colours for free. Works in every viewport mode (it's an ImGui overlay).
    // Use the BACKGROUND draw list (not foreground): the 3D texture is blitted
    // first, then ImGui draws background list → windows → foreground, so the
    // background list paints over the 3D viewport but stays BEHIND the panels
    // (the mask tint lives on the surface, it must not bleed over the UI).
    ImDrawList* dl = ImGui::GetBackgroundDrawList();
    const ImVec2 uvWhite = ImGui::GetFontTexUvWhitePixel();
    const std::vector<EditableFace>& faces = editable_mesh_cache.faces;
    constexpr int kMaxTris = 120000;
    int drawnTris = 0;
    auto maskColor = [](float m) -> ImU32 {
        const float c = std::clamp(m, 0.0f, 1.0f);
        // Transparent where unmasked → soft gradient edge; opaque blue at full mask.
        const int a = static_cast<int>(c * c * 165.0f);
        return IM_COL32(70, 150, 240, a);
    };
    for (const EditableFace& f : faces) {
        const int i0 = f.v0, i1 = f.v1, i2 = f.v2;
        if (i0 < 0 || i1 < 0 || i2 < 0 ||
            static_cast<size_t>(i0) >= n || static_cast<size_t>(i1) >= n || static_cast<size_t>(i2) >= n) {
            continue;
        }
        const float m0 = sculpt_mask_state.values[i0];
        const float m1 = sculpt_mask_state.values[i1];
        const float m2 = sculpt_mask_state.values[i2];
        if (m0 <= 0.02f && m1 <= 0.02f && m2 <= 0.02f) {
            continue;
        }
        // Back-face cull using a vertex shading normal (winding-independent, so a
        // reversed-winding mesh can't make the whole tint vanish). No depth buffer
        // on the ImGui overlay, so this keeps far-side tint from bleeding through.
        if (!verts[i0].refs.empty() && editable_mesh_cache.refTri(verts[i0].refs[0])) {
            const Vec3 vnrm = editable_mesh_cache.refTri(verts[i0].refs[0])->vertices[verts[i0].refs[0].corner].normal;
            if (vnrm.length_squared() > 1e-8f) {
                const Vec3 w0 = transform.transform_point(verts[i0].local_position);
                if (vnrm.dot(camPos - w0) < 0.0f) {
                    continue;
                }
            }
        }
        ImVec2 p0, p1, p2;
        if (!vertexScreen(static_cast<size_t>(i0), p0) ||
            !vertexScreen(static_cast<size_t>(i1), p1) ||
            !vertexScreen(static_cast<size_t>(i2), p2)) {
            continue;
        }
        dl->PrimReserve(3, 3);
        dl->PrimVtx(p0, uvWhite, maskColor(m0));
        dl->PrimVtx(p1, uvWhite, maskColor(m1));
        dl->PrimVtx(p2, uvWhite, maskColor(m2));
        if (++drawnTris >= kMaxTris) {
            break;
        }
    }
}

void SceneUI::drawLassoOutline() {
    if (!is_lasso_selecting || lasso_points.size() < 2) return;

    ImDrawList* draw_list = ImGui::GetForegroundDrawList();
    ImU32 outlineColor = IM_COL32(100, 150, 255, 200);
    ImU32 fillColor = IM_COL32(100, 150, 255, 40);

    for (size_t i = 1; i < lasso_points.size(); ++i) {
        draw_list->AddLine(lasso_points[i - 1], lasso_points[i], outlineColor, 2.0f);
    }
    if (lasso_points.size() >= 3) {
        draw_list->AddLine(lasso_points.back(), lasso_points.front(), outlineColor, 2.0f);
        draw_list->AddConvexPolyFilled(lasso_points.data(), static_cast<int>(lasso_points.size()), fillColor);
    }
}

void SceneUI::performMeshElementMarqueeSelection(UIContext& ctx, float x1, float y1, float x2, float y2) {
    std::string objectName = active_mesh_edit_object_name;
    if (objectName.empty() && ctx.selection.selected.type == SelectableType::Object && ctx.selection.selected.object) {
        objectName = ctx.selection.selected.object->getNodeName();
    }
    if (objectName.empty()) return;
    if (!ensureEditableMeshCache(ctx, objectName)) return;

    const Matrix4x4 transform = getEditableObjectTransform(editable_mesh_cache);
    const ImVec2 displaySize = ImGui::GetIO().DisplaySize;
    const bool isOrtho = ctx.scene.camera->orthographic && viewport_settings.shading_mode != 2;

    const MeshElementSelectMode requestedMode = ctx.selection.mesh_element_mode;
    const bool combinedMode = (requestedMode == MeshElementSelectMode::Combined);
    const bool tryVertex = combinedMode || requestedMode == MeshElementSelectMode::Vertex;
    const bool tryEdge   = combinedMode || requestedMode == MeshElementSelectMode::Edge;
    const bool tryFace   = combinedMode || requestedMode == MeshElementSelectMode::Face;

    EditableMeshSelection& selection = editable_mesh_cache.selection;

    auto updateSelection = [&](std::vector<int>& selectionList, const std::vector<int>& newSelectedElements) {
        if (ImGui::GetIO().KeyShift) {
            std::unordered_set<int> uniqueSel(selectionList.begin(), selectionList.end());
            for (int id : newSelectedElements) {
                if (uniqueSel.insert(id).second) {
                    selectionList.push_back(id);
                }
            }
        } else if (ImGui::GetIO().KeyCtrl) {
            std::unordered_set<int> uniqueToRemove(newSelectedElements.begin(), newSelectedElements.end());
            std::vector<int> kept;
            for (int id : selectionList) {
                if (uniqueToRemove.find(id) == uniqueToRemove.end()) {
                    kept.push_back(id);
                }
            }
            selectionList = kept;
        } else {
            selectionList = newSelectedElements;
        }
    };

    if (tryVertex) {
        std::vector<int> inRegion;
        for (size_t i = 0; i < editable_mesh_cache.vertices.size(); ++i) {
            ImVec2 screen;
            if (!projectPointToScreen(*ctx.scene.camera, displaySize,
                                      transform.transform_point(editable_mesh_cache.vertices[i].local_position), screen, isOrtho)) {
                continue;
            }
            if (screen.x >= x1 && screen.x <= x2 && screen.y >= y1 && screen.y <= y2) {
                inRegion.push_back(static_cast<int>(i));
            }
        }
        updateSelection(selection.vertex_ids, inRegion);
        selection.active_vertex_id = selection.vertex_ids.empty() ? -1 : selection.vertex_ids.back();
        if (!combinedMode) {
            selection.edge_ids.clear();
            selection.face_ids.clear();
            selection.active_edge_id = -1;
            selection.active_face_id = -1;
        }
    }

    if (tryEdge) {
        std::vector<int> inRegion;
        const auto& selectableEdges = !editable_mesh_cache.polygon_edges.empty() ? editable_mesh_cache.polygon_edges : editable_mesh_cache.edges;
        for (size_t i = 0; i < selectableEdges.size(); ++i) {
            const auto& edge = selectableEdges[i];
            if (edge.v0 < 0 || edge.v1 < 0 ||
                edge.v0 >= static_cast<int>(editable_mesh_cache.vertices.size()) ||
                edge.v1 >= static_cast<int>(editable_mesh_cache.vertices.size())) {
                continue;
            }
            ImVec2 s0, s1;
            if (!projectPointToScreen(*ctx.scene.camera, displaySize,
                                      transform.transform_point(editable_mesh_cache.vertices[edge.v0].local_position), s0, isOrtho) ||
                !projectPointToScreen(*ctx.scene.camera, displaySize,
                                      transform.transform_point(editable_mesh_cache.vertices[edge.v1].local_position), s1, isOrtho)) {
                continue;
            }
            if (s0.x >= x1 && s0.x <= x2 && s0.y >= y1 && s0.y <= y2 &&
                s1.x >= x1 && s1.x <= x2 && s1.y >= y1 && s1.y <= y2) {
                inRegion.push_back(static_cast<int>(i));
            }
        }
        updateSelection(selection.edge_ids, inRegion);
        selection.active_edge_id = selection.edge_ids.empty() ? -1 : selection.edge_ids.back();
        if (!combinedMode) {
            selection.vertex_ids.clear();
            selection.face_ids.clear();
            selection.active_vertex_id = -1;
            selection.active_face_id = -1;
        }
    }

    if (tryFace) {
        std::vector<int> inRegion;
        const size_t polygonFaceCount = editable_mesh_cache.polygon_faces.empty() ? editable_mesh_cache.faces.size() : editable_mesh_cache.polygon_faces.size();
        for (size_t i = 0; i < polygonFaceCount; ++i) {
            const std::vector<int> vertexIds = getEditablePolygonVertexIds(editable_mesh_cache, static_cast<int>(i));
            if (vertexIds.size() < 3) continue;

            bool allInside = true;
            for (const int vertexId : vertexIds) {
                if (!isEditableVertexIdValid(editable_mesh_cache, vertexId)) {
                    allInside = false;
                    break;
                }
                ImVec2 screen;
                if (!projectPointToScreen(*ctx.scene.camera, displaySize,
                                          transform.transform_point(editable_mesh_cache.vertices[vertexId].local_position), screen, isOrtho)) {
                    allInside = false;
                    break;
                }
                if (!(screen.x >= x1 && screen.x <= x2 && screen.y >= y1 && screen.y <= y2)) {
                    allInside = false;
                    break;
                }
            }
            if (allInside) {
                inRegion.push_back(static_cast<int>(i));
            }
        }
        updateSelection(selection.face_ids, inRegion);
        selection.active_face_id = selection.face_ids.empty() ? -1 : selection.face_ids.back();
        if (!combinedMode) {
            selection.vertex_ids.clear();
            selection.edge_ids.clear();
            selection.active_vertex_id = -1;
            selection.active_edge_id = -1;
        }
    }

    bool hasPosition = false;
    const Vec3 worldPos = getSelectedMeshElementWorldPosition(ctx, &hasPosition);
    if (hasPosition) {
        ctx.selection.selected.position = worldPos;
    }
}

void SceneUI::performMeshElementLassoSelection(UIContext& ctx, const std::vector<ImVec2>& points) {
    if (points.size() < 3) return;

    std::string objectName = active_mesh_edit_object_name;
    if (objectName.empty() && ctx.selection.selected.type == SelectableType::Object && ctx.selection.selected.object) {
        objectName = ctx.selection.selected.object->getNodeName();
    }
    if (objectName.empty()) return;
    if (!ensureEditableMeshCache(ctx, objectName)) return;

    const Matrix4x4 transform = getEditableObjectTransform(editable_mesh_cache);
    const ImVec2 displaySize = ImGui::GetIO().DisplaySize;
    const bool isOrtho = ctx.scene.camera->orthographic && viewport_settings.shading_mode != 2;

    const MeshElementSelectMode requestedMode = ctx.selection.mesh_element_mode;
    const bool combinedMode = (requestedMode == MeshElementSelectMode::Combined);
    const bool tryVertex = combinedMode || requestedMode == MeshElementSelectMode::Vertex;
    const bool tryEdge   = combinedMode || requestedMode == MeshElementSelectMode::Edge;
    const bool tryFace   = combinedMode || requestedMode == MeshElementSelectMode::Face;

    EditableMeshSelection& selection = editable_mesh_cache.selection;

    auto updateSelection = [&](std::vector<int>& selectionList, const std::vector<int>& newSelectedElements) {
        if (ImGui::GetIO().KeyShift) {
            std::unordered_set<int> uniqueSel(selectionList.begin(), selectionList.end());
            for (int id : newSelectedElements) {
                if (uniqueSel.insert(id).second) {
                    selectionList.push_back(id);
                }
            }
        } else if (ImGui::GetIO().KeyCtrl) {
            std::unordered_set<int> uniqueToRemove(newSelectedElements.begin(), newSelectedElements.end());
            std::vector<int> kept;
            for (int id : selectionList) {
                if (uniqueToRemove.find(id) == uniqueToRemove.end()) {
                    kept.push_back(id);
                }
            }
            selectionList = kept;
        } else {
            selectionList = newSelectedElements;
        }
    };

    auto isPointInPolygon = [](const ImVec2& p, const std::vector<ImVec2>& polygon) -> bool {
        bool inside = false;
        for (size_t i = 0, j = polygon.size() - 1; i < polygon.size(); j = i++) {
            if (((polygon[i].y > p.y) != (polygon[j].y > p.y)) &&
                (p.x < (polygon[j].x - polygon[i].x) * (p.y - polygon[i].y) / (polygon[j].y - polygon[i].y) + polygon[i].x)) {
                inside = !inside;
            }
        }
        return inside;
    };

    if (tryVertex) {
        std::vector<int> inRegion;
        for (size_t i = 0; i < editable_mesh_cache.vertices.size(); ++i) {
            ImVec2 screen;
            if (!projectPointToScreen(*ctx.scene.camera, displaySize,
                                      transform.transform_point(editable_mesh_cache.vertices[i].local_position), screen, isOrtho)) {
                continue;
            }
            if (isPointInPolygon(screen, points)) {
                inRegion.push_back(static_cast<int>(i));
            }
        }
        updateSelection(selection.vertex_ids, inRegion);
        selection.active_vertex_id = selection.vertex_ids.empty() ? -1 : selection.vertex_ids.back();
        if (!combinedMode) {
            selection.edge_ids.clear();
            selection.face_ids.clear();
            selection.active_edge_id = -1;
            selection.active_face_id = -1;
        }
    }

    if (tryEdge) {
        std::vector<int> inRegion;
        const auto& selectableEdges = !editable_mesh_cache.polygon_edges.empty() ? editable_mesh_cache.polygon_edges : editable_mesh_cache.edges;
        for (size_t i = 0; i < selectableEdges.size(); ++i) {
            const auto& edge = selectableEdges[i];
            if (edge.v0 < 0 || edge.v1 < 0 ||
                edge.v0 >= static_cast<int>(editable_mesh_cache.vertices.size()) ||
                edge.v1 >= static_cast<int>(editable_mesh_cache.vertices.size())) {
                continue;
            }
            ImVec2 s0, s1;
            if (!projectPointToScreen(*ctx.scene.camera, displaySize,
                                      transform.transform_point(editable_mesh_cache.vertices[edge.v0].local_position), s0, isOrtho) ||
                !projectPointToScreen(*ctx.scene.camera, displaySize,
                                      transform.transform_point(editable_mesh_cache.vertices[edge.v1].local_position), s1, isOrtho)) {
                continue;
            }
            if (isPointInPolygon(s0, points) && isPointInPolygon(s1, points)) {
                inRegion.push_back(static_cast<int>(i));
            }
        }
        updateSelection(selection.edge_ids, inRegion);
        selection.active_edge_id = selection.edge_ids.empty() ? -1 : selection.edge_ids.back();
        if (!combinedMode) {
            selection.vertex_ids.clear();
            selection.face_ids.clear();
            selection.active_vertex_id = -1;
            selection.active_face_id = -1;
        }
    }

    if (tryFace) {
        std::vector<int> inRegion;
        const size_t polygonFaceCount = editable_mesh_cache.polygon_faces.empty() ? editable_mesh_cache.faces.size() : editable_mesh_cache.polygon_faces.size();
        for (size_t i = 0; i < polygonFaceCount; ++i) {
            const std::vector<int> vertexIds = getEditablePolygonVertexIds(editable_mesh_cache, static_cast<int>(i));
            if (vertexIds.size() < 3) continue;

            bool allInside = true;
            for (const int vertexId : vertexIds) {
                if (!isEditableVertexIdValid(editable_mesh_cache, vertexId)) {
                    allInside = false;
                    break;
                }
                ImVec2 screen;
                if (!projectPointToScreen(*ctx.scene.camera, displaySize,
                                          transform.transform_point(editable_mesh_cache.vertices[vertexId].local_position), screen, isOrtho)) {
                    allInside = false;
                    break;
                }
                if (!isPointInPolygon(screen, points)) {
                    allInside = false;
                    break;
                }
            }
            if (allInside) {
                inRegion.push_back(static_cast<int>(i));
            }
        }
        updateSelection(selection.face_ids, inRegion);
        selection.active_face_id = selection.face_ids.empty() ? -1 : selection.face_ids.back();
        if (!combinedMode) {
            selection.vertex_ids.clear();
            selection.edge_ids.clear();
            selection.active_vertex_id = -1;
            selection.active_edge_id = -1;
        }
    }

    bool hasPosition = false;
    const Vec3 worldPos = getSelectedMeshElementWorldPosition(ctx, &hasPosition);
    if (hasPosition) {
        ctx.selection.selected.position = worldPos;
    }
}

void SceneUI::selectAllObjects(UIContext& ctx) {
    ctx.selection.clearSelection();

    if (!mesh_cache_valid) rebuildMeshCache(ctx.scene.world.objects);

    for (auto& [name, triangles] : mesh_cache) {
        if (triangles.empty()) continue;
        SelectableItem item;
        item.type = SelectableType::Object;
        item.object = triangles[0].second;
        item.object_index = triangles[0].first;
        item.name = name;
        ctx.selection.addToSelection(item);
    }

    for (size_t i = 0; i < ctx.scene.lights.size(); ++i) {
        auto& light = ctx.scene.lights[i];
        if (!light) continue;
        SelectableItem item;
        item.type = SelectableType::Light;
        item.light = light;
        item.light_index = (int)i;
        item.name = "Light_" + std::to_string(i);
        ctx.selection.addToSelection(item);
    }
}

void SceneUI::invertObjectSelection(UIContext& ctx) {
    if (!mesh_cache_valid) rebuildMeshCache(ctx.scene.world.objects);

    std::vector<SelectableItem> allItems;
    for (auto& [name, triangles] : mesh_cache) {
        if (triangles.empty()) continue;
        SelectableItem item;
        item.type = SelectableType::Object;
        item.object = triangles[0].second;
        item.object_index = triangles[0].first;
        item.name = name;
        allItems.push_back(item);
    }
    for (size_t i = 0; i < ctx.scene.lights.size(); ++i) {
        auto& light = ctx.scene.lights[i];
        if (!light) continue;
        SelectableItem item;
        item.type = SelectableType::Light;
        item.light = light;
        item.light_index = (int)i;
        item.name = "Light_" + std::to_string(i);
        allItems.push_back(item);
    }

    std::vector<SelectableItem> currentSelection = ctx.selection.multi_selection;
    ctx.selection.clearSelection();

    for (const auto& item : allItems) {
        bool wasSelected = false;
        for (const auto& cur : currentSelection) {
            if (cur.type == item.type) {
                if (cur.type == SelectableType::Object && cur.name == item.name) {
                    wasSelected = true;
                    break;
                } else if (cur.type == SelectableType::Light && cur.light == item.light) {
                    wasSelected = true;
                    break;
                }
            }
        }
        if (!wasSelected) {
            ctx.selection.addToSelection(item);
        }
    }
}

void SceneUI::selectAllMeshElements(UIContext& ctx) {
    std::string objectName = active_mesh_edit_object_name;
    if (objectName.empty() && ctx.selection.selected.type == SelectableType::Object && ctx.selection.selected.object) {
        objectName = ctx.selection.selected.object->getNodeName();
    }
    if (objectName.empty()) return;
    if (!ensureEditableMeshCache(ctx, objectName)) return;

    EditableMeshSelection& sel = editable_mesh_cache.selection;
    const MeshElementSelectMode mode = ctx.selection.mesh_element_mode;

    if (mode == MeshElementSelectMode::Vertex) {
        sel.vertex_ids.clear();
        sel.vertex_ids.reserve(editable_mesh_cache.vertices.size());
        for (size_t i = 0; i < editable_mesh_cache.vertices.size(); ++i) {
            sel.vertex_ids.push_back(static_cast<int>(i));
        }
        sel.active_vertex_id = sel.vertex_ids.empty() ? -1 : sel.vertex_ids.back();
    } else if (mode == MeshElementSelectMode::Edge) {
        sel.edge_ids.clear();
        const auto& selectableEdges = !editable_mesh_cache.polygon_edges.empty() ? editable_mesh_cache.polygon_edges : editable_mesh_cache.edges;
        sel.edge_ids.reserve(selectableEdges.size());
        for (size_t i = 0; i < selectableEdges.size(); ++i) {
            sel.edge_ids.push_back(static_cast<int>(i));
        }
        sel.active_edge_id = sel.edge_ids.empty() ? -1 : sel.edge_ids.back();
    } else if (mode == MeshElementSelectMode::Face) {
        sel.face_ids.clear();
        const size_t polygonFaceCount = editable_mesh_cache.polygon_faces.empty() ? editable_mesh_cache.faces.size() : editable_mesh_cache.polygon_faces.size();
        sel.face_ids.reserve(polygonFaceCount);
        for (size_t i = 0; i < polygonFaceCount; ++i) {
            sel.face_ids.push_back(static_cast<int>(i));
        }
        sel.active_face_id = sel.face_ids.empty() ? -1 : sel.face_ids.back();
    } else if (mode == MeshElementSelectMode::Combined) {
        sel.vertex_ids.clear();
        sel.vertex_ids.reserve(editable_mesh_cache.vertices.size());
        for (size_t i = 0; i < editable_mesh_cache.vertices.size(); ++i) {
            sel.vertex_ids.push_back(static_cast<int>(i));
        }
        sel.active_vertex_id = sel.vertex_ids.empty() ? -1 : sel.vertex_ids.back();

        sel.edge_ids.clear();
        const auto& selectableEdges = !editable_mesh_cache.polygon_edges.empty() ? editable_mesh_cache.polygon_edges : editable_mesh_cache.edges;
        sel.edge_ids.reserve(selectableEdges.size());
        for (size_t i = 0; i < selectableEdges.size(); ++i) {
            sel.edge_ids.push_back(static_cast<int>(i));
        }
        sel.active_edge_id = sel.edge_ids.empty() ? -1 : sel.edge_ids.back();

        sel.face_ids.clear();
        const size_t polygonFaceCount = editable_mesh_cache.polygon_faces.empty() ? editable_mesh_cache.faces.size() : editable_mesh_cache.polygon_faces.size();
        sel.face_ids.reserve(polygonFaceCount);
        for (size_t i = 0; i < polygonFaceCount; ++i) {
            sel.face_ids.push_back(static_cast<int>(i));
        }
        sel.active_face_id = sel.face_ids.empty() ? -1 : sel.face_ids.back();
    }

    bool hasPosition = false;
    const Vec3 worldPos = getSelectedMeshElementWorldPosition(ctx, &hasPosition);
    if (hasPosition) {
        ctx.selection.selected.position = worldPos;
    }
}

void SceneUI::invertMeshSelection(UIContext& ctx) {
    std::string objectName = active_mesh_edit_object_name;
    if (objectName.empty() && ctx.selection.selected.type == SelectableType::Object && ctx.selection.selected.object) {
        objectName = ctx.selection.selected.object->getNodeName();
    }
    if (objectName.empty()) return;
    if (!ensureEditableMeshCache(ctx, objectName)) return;

    EditableMeshSelection& sel = editable_mesh_cache.selection;
    const MeshElementSelectMode mode = ctx.selection.mesh_element_mode;

    if (mode == MeshElementSelectMode::Vertex) {
        std::unordered_set<int> curSel(sel.vertex_ids.begin(), sel.vertex_ids.end());
        sel.vertex_ids.clear();
        for (size_t i = 0; i < editable_mesh_cache.vertices.size(); ++i) {
            if (curSel.find(static_cast<int>(i)) == curSel.end()) {
                sel.vertex_ids.push_back(static_cast<int>(i));
            }
        }
        sel.active_vertex_id = sel.vertex_ids.empty() ? -1 : sel.vertex_ids.back();
    } else if (mode == MeshElementSelectMode::Edge) {
        std::unordered_set<int> curSel(sel.edge_ids.begin(), sel.edge_ids.end());
        sel.edge_ids.clear();
        const auto& selectableEdges = !editable_mesh_cache.polygon_edges.empty() ? editable_mesh_cache.polygon_edges : editable_mesh_cache.edges;
        for (size_t i = 0; i < selectableEdges.size(); ++i) {
            if (curSel.find(static_cast<int>(i)) == curSel.end()) {
                sel.edge_ids.push_back(static_cast<int>(i));
            }
        }
        sel.active_edge_id = sel.edge_ids.empty() ? -1 : sel.edge_ids.back();
    } else if (mode == MeshElementSelectMode::Face) {
        std::unordered_set<int> curSel(sel.face_ids.begin(), sel.face_ids.end());
        sel.face_ids.clear();
        const size_t polygonFaceCount = editable_mesh_cache.polygon_faces.empty() ? editable_mesh_cache.faces.size() : editable_mesh_cache.polygon_faces.size();
        for (size_t i = 0; i < polygonFaceCount; ++i) {
            if (curSel.find(static_cast<int>(i)) == curSel.end()) {
                sel.face_ids.push_back(static_cast<int>(i));
            }
        }
        sel.active_face_id = sel.face_ids.empty() ? -1 : sel.face_ids.back();
    } else if (mode == MeshElementSelectMode::Combined) {
        {
            std::unordered_set<int> curSel(sel.vertex_ids.begin(), sel.vertex_ids.end());
            sel.vertex_ids.clear();
            for (size_t i = 0; i < editable_mesh_cache.vertices.size(); ++i) {
                if (curSel.find(static_cast<int>(i)) == curSel.end()) {
                    sel.vertex_ids.push_back(static_cast<int>(i));
                }
            }
            sel.active_vertex_id = sel.vertex_ids.empty() ? -1 : sel.vertex_ids.back();
        }
        {
            std::unordered_set<int> curSel(sel.edge_ids.begin(), sel.edge_ids.end());
            sel.edge_ids.clear();
            const auto& selectableEdges = !editable_mesh_cache.polygon_edges.empty() ? editable_mesh_cache.polygon_edges : editable_mesh_cache.edges;
            for (size_t i = 0; i < selectableEdges.size(); ++i) {
                if (curSel.find(static_cast<int>(i)) == curSel.end()) {
                    sel.edge_ids.push_back(static_cast<int>(i));
                }
            }
            sel.active_edge_id = sel.edge_ids.empty() ? -1 : sel.edge_ids.back();
        }
        {
            std::unordered_set<int> curSel(sel.face_ids.begin(), sel.face_ids.end());
            sel.face_ids.clear();
            const size_t polygonFaceCount = editable_mesh_cache.polygon_faces.empty() ? editable_mesh_cache.faces.size() : editable_mesh_cache.polygon_faces.size();
            for (size_t i = 0; i < polygonFaceCount; ++i) {
                if (curSel.find(static_cast<int>(i)) == curSel.end()) {
                    sel.face_ids.push_back(static_cast<int>(i));
                }
            }
            sel.active_face_id = sel.face_ids.empty() ? -1 : sel.face_ids.back();
        }
    }

    bool hasPosition = false;
    const Vec3 worldPos = getSelectedMeshElementWorldPosition(ctx, &hasPosition);
    if (hasPosition) {
        ctx.selection.selected.position = worldPos;
    }
}

