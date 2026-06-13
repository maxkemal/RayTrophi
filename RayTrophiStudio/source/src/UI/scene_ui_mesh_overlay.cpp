#include "scene_ui.h"

#include "Backend/OptixBackend.h"
#include "Backend/IViewportBackend.h"
#include "Backend/VulkanBackend.h"
#include "Viewport/ViewportSceneSync.h"
#include "Triangle.h"
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
float uiSampleBrushMask(const Paint::BrushSettings& brush, float nx, float ny);

namespace {
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

bool projectPointToScreen(const Camera& cam, const ImVec2& displaySize, const Vec3& point, ImVec2& out) {
    if (displaySize.x <= 1.0f || displaySize.y <= 1.0f) {
        return false;
    }

    const Vec3 camForward = (cam.lookat - cam.lookfrom).normalize();
    const Vec3 camRight = camForward.cross(cam.vup).normalize();
    const Vec3 camUp = camRight.cross(camForward).normalize();
    const Vec3 toPoint = point - cam.lookfrom;
    const float depth = toPoint.dot(camForward);
    if (depth <= 0.01f) {
        return false;
    }

    // Use render resolution aspect ratio to match the GPU projection matrix.
    // SDL_RenderCopy stretches the render texture to fill the window, so the
    // final pixel coords still use displaySize — but the horizontal FOV must
    // be derived from the render resolution (image_width/image_height), not
    // the window size, to stay in sync with the Vulkan perspective matrix.
    const float aspect = (image_height > 0)
        ? (static_cast<float>(image_width) / static_cast<float>(image_height))
        : (displaySize.x / displaySize.y);
    const float tanHalfFov = std::tan(cam.vfov * 3.14159265359f / 180.0f * 0.5f);
    if (std::fabs(aspect) <= 1e-6f || std::fabs(tanHalfFov) <= 1e-6f) {
        return false;
    }

    const float localX = toPoint.dot(camRight);
    const float localY = toPoint.dot(camUp);
    const float halfH = depth * tanHalfFov;
    const float halfW = halfH * aspect;
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
                cache.faces[triangleId].triangle) {
                return cache.faces[triangleId].triangle;
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

bool hasEnabledSubdivisionPreview(const MeshModifiers::ModifierStack& stack) {
    for (const auto& mod : stack.modifiers) {
        if (!mod.enabled) {
            continue;
        }
        if (mod.type == MeshModifiers::ModifierType::FlatSubdivision ||
            mod.type == MeshModifiers::ModifierType::SmoothSubdivision) {
            return true;
        }
    }
    return false;
}

MeshModifiers::ModifierStack buildSubdivisionPreviewEvaluationStack(
    const MeshModifiers::ModifierStack& stack,
    bool interactivePreview) {
    if (!interactivePreview) {
        return stack;
    }

    MeshModifiers::ModifierStack previewStack = stack;
    bool keptSubdivisionStage = false;
    for (auto& mod : previewStack.modifiers) {
        if (!mod.enabled) {
            continue;
        }
        if (mod.type != MeshModifiers::ModifierType::FlatSubdivision &&
            mod.type != MeshModifiers::ModifierType::SmoothSubdivision) {
            continue;
        }

        if (!keptSubdivisionStage) {
            mod.levels = std::clamp(mod.levels, 0, 1);
            mod.enabled = mod.levels > 0;
            keptSubdivisionStage = mod.enabled;
        } else {
            mod.enabled = false;
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

std::vector<std::shared_ptr<Triangle>> evaluateDisplayMeshFromBase(
    const std::vector<std::shared_ptr<Triangle>>& baseMesh,
    const MeshModifiers::ModifierStack& stack) {
    if (stack.modifiers.empty()) {
        return cloneTriangleVectorForEdit(baseMesh);
    }
    return stack.evaluate(baseMesh);
}

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
    if (vertexId >= cache.vertices.size()) {
        return Vec3(0.0f, 0.0f, 0.0f);
    }
    if (vertexId < snapshotLocalPositions.size() && isFiniteVec3(snapshotLocalPositions[vertexId])) {
        return snapshotLocalPositions[vertexId];
    }
    return cache.vertices[vertexId].local_position;
}

Vec3 computeBoundarySafeSmoothDelta(
    const SceneUI::EditableMeshCache& cache,
    const std::vector<Vec3>& snapshotLocalPositions,
    size_t vertexId,
    float brushStrength,
    float dt,
    float weight,
    float localRadius) {
    if (vertexId >= cache.vertices.size()) {
        return Vec3(0.0f, 0.0f, 0.0f);
    }

    const auto& vertex = cache.vertices[vertexId];
    const auto& neighbors = cache.vertex_neighbors[vertexId];
    if (neighbors.empty()) {
        return Vec3(0.0f, 0.0f, 0.0f);
    }

    Vec3 average(0.0f, 0.0f, 0.0f);
    int neighborCount = 0;
    for (const int neighborId : neighbors) {
        if (neighborId < 0 || neighborId >= static_cast<int>(cache.vertices.size())) {
            continue;
        }
        const auto& neighbor = cache.vertices[static_cast<size_t>(neighborId)];
        if (vertex.is_boundary && !neighbor.is_boundary) {
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
    if (vertex.is_boundary) {
        smoothFactor *= 0.35f;
    }

    const Vec3 currentLocal = resolveEditableSnapshotLocalPosition(cache, snapshotLocalPositions, vertexId);
    Vec3 smoothDelta = sanitizeVec3(
        (average - currentLocal) * smoothFactor,
        Vec3(0.0f, 0.0f, 0.0f));
    const float deltaLen = smoothDelta.length();
    const float maxDelta = (std::max)(localRadius * (vertex.is_boundary ? 0.03f : 0.08f), 1e-5f);
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

    std::vector<Vec3> snapshot = updatedLocalPositions;
    const float maxDelta = (std::max)(localRadius * 0.035f, 1e-5f);

    for (const int vertexIdInt : touchedVertexIds) {
        if (vertexIdInt < 0 || vertexIdInt >= static_cast<int>(cache.vertices.size())) {
            continue;
        }
        const size_t vertexId = static_cast<size_t>(vertexIdInt);
        const auto& neighbors = cache.vertex_neighbors[vertexId];
        if (neighbors.empty()) {
            continue;
        }

        const auto& vertex = cache.vertices[vertexId];
        Vec3 average(0.0f, 0.0f, 0.0f);
        int neighborCount = 0;
        for (const int neighborId : neighbors) {
            if (neighborId < 0 || neighborId >= static_cast<int>(cache.vertices.size())) {
                continue;
            }
            // Include ALL neighbors for averaging — boundary vertices need
            // interior neighbors to stay anchored and avoid folding.
            average += snapshot[static_cast<size_t>(neighborId)];
            ++neighborCount;
        }
        if (neighborCount == 0) {
            continue;
        }

        average /= static_cast<float>(neighborCount);
        const Vec3 currentLocal = snapshot[vertexId];
        const Vec3 currentWorld = sanitizeVec3(transform.transform_point(currentLocal), currentLocal);
        const Vec3 averageWorld = sanitizeVec3(transform.transform_point(average), average);
        Vec3 relaxWorld = averageWorld - currentWorld;
        relaxWorld = relaxWorld - strokeNormal * relaxWorld.dot(strokeNormal);
        // Boundary vertices need stronger polish — their asymmetric topology
        // makes them prone to folding/inverting under clay deposits.
        relaxWorld *= safePolish * (vertex.is_boundary ? 0.8f : 1.0f);

        Vec3 relaxLocal = sanitizeVec3(inverseTransform.transform_vector(relaxWorld), Vec3(0.0f, 0.0f, 0.0f));
        const float deltaLen = relaxLocal.length();
        if (std::isfinite(deltaLen) && deltaLen > maxDelta) {
            relaxLocal *= (maxDelta / deltaLen);
        }

        updatedLocalPositions[vertexId] = sanitizeVec3(currentLocal + relaxLocal, currentLocal);
    }
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

    for (const int vertexIdInt : touchedVertexIds) {
        if (vertexIdInt < 0 || vertexIdInt >= static_cast<int>(cache.vertices.size())) {
            continue;
        }

        const size_t vertexId = static_cast<size_t>(vertexIdInt);
        const auto& neighbors = cache.vertex_neighbors[vertexId];
        if (neighbors.empty()) {
            continue;
        }

        Vec3 averageLocal(0.0f, 0.0f, 0.0f);
        int neighborCount = 0;
        for (const int neighborId : neighbors) {
            if (neighborId < 0 || neighborId >= static_cast<int>(cache.vertices.size())) {
                continue;
            }
            averageLocal += snapshot[static_cast<size_t>(neighborId)];
            ++neighborCount;
        }
        if (neighborCount == 0) {
            continue;
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
    }
}





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

void primeGrabStrokeWeights(
    SceneUI::SculptStrokeState& strokeState,
    const SceneUI::EditableMeshCache& editableMeshCache,
    const Matrix4x4& transform,
    const std::vector<int>& candidateVertexIds,
    const Vec3& brushCenter,
    const Vec3& hitNormalWorld,
    float radiusWorld,
    bool frontFacesOnly,
    int falloffType) {
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

        const float weight = applyFalloffCurve(
            1.0f - saturateFloat(planarDistance / radiusWorld),
            falloffType);
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
    const float grabStrength = std::clamp(brushStrength * 0.5f, 0.0f, 5.0f);
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
    case SceneUI::SculptBrushTool::Draw: {
        Vec3 vertexNormal = hitNormalWorld;
        if (!vertex.refs.empty() && vertex.refs[0].triangle) {
            const auto& ref = vertex.refs[0];
            const Vec3 n = ref.triangle->vertices[ref.corner].normal;
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
        if (!vertex.refs.empty() && vertex.refs[0].triangle) {
            const auto& ref = vertex.refs[0];
            const Vec3 n = ref.triangle->vertices[ref.corner].normal;
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
        if (vertex.is_boundary) {
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
        if (vertex.is_boundary) {
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
    const float maxLocalStep = (std::max)(meshAvgEdgeLocal * 0.4f, localRadius * 1e-4f);
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
            if (!vertex.refs.empty() && vertex.refs[0].triangle) {
                const Vec3 normal = vertex.refs[0].triangle->getOriginalVertexNormal(vertex.refs[0].corner);
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
        node.neighbor_ids = editableMeshCache.vertex_neighbors[vertexId];
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

        const auto found = editableMeshCache.triangle_vertex_ids.find(tri.get());
        if (found == editableMeshCache.triangle_vertex_ids.end()) {
            continue;
        }

        for (const int vertexIdInt : found->second) {
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
            if (!ref.triangle) {
                continue;
            }
            if (tryMarkEditableTriangleTouched(editableMeshCache, ref.triangle.get())) {
                touchedTriangles.push_back(ref.triangle);
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
        if (!ref.triangle) {
            continue;
        }

        const auto triFound = editableMeshCache.triangle_vertex_ids.find(ref.triangle.get());
        if (triFound == editableMeshCache.triangle_vertex_ids.end()) {
            continue;
        }

        const auto& triVertexIds = triFound->second;
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

        if (!std::isfinite(newEdge01LenSq) || !std::isfinite(newEdge12LenSq) || !std::isfinite(newEdge20LenSq)) {
            return false;
        }
        if (newEdge01LenSq <= (std::max)(1e-12f, oldEdge01LenSq * 0.0025f) ||
            newEdge12LenSq <= (std::max)(1e-12f, oldEdge12LenSq * 0.0025f) ||
            newEdge20LenSq <= (std::max)(1e-12f, oldEdge20LenSq * 0.0025f)) {
            return false;
        }

        if (!std::isfinite(newLenSq) || newLenSq <= (std::max)(1e-12f, oldLenSq * 0.01f)) {
            return false;
        }

        if (oldLenSq > 1e-12f) {
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

void recomputeEditableSmoothNormals(
    SceneUI::EditableMeshCache& editableMeshCache,
    const std::vector<size_t>& affectedVertexIds) {
    if (affectedVertexIds.empty()) {
        return;
    }

    const float clampedAngle = std::clamp(editableMeshCache.auto_smooth_angle_degrees, 1.0f, 180.0f);
    const float autoSmoothThreshold = std::cos(clampedAngle * 3.14159265359f / 180.0f);

    auto recomputeVertexNormal = [&](const size_t vertexId) {
        if (vertexId >= editableMeshCache.vertices.size()) {
            return;
        }

        const auto& refs = editableMeshCache.vertices[vertexId].refs;
        if (refs.empty()) {
            return;
        }

        thread_local std::vector<Vec3> faceNormals;
        faceNormals.clear();
        faceNormals.reserve(refs.size());
        for (const auto& ref : refs) {
            if (!ref.triangle) {
                faceNormals.push_back(Vec3(0.0f, 1.0f, 0.0f));
                continue;
            }

            faceNormals.push_back(computeOrientationPreservingFaceNormal(
                *ref.triangle,
                ref.triangle->getOriginalVertexNormal(ref.corner)));
        }

        for (size_t refIndex = 0; refIndex < refs.size(); ++refIndex) {
            const auto& ref = refs[refIndex];
            if (!ref.triangle) {
                continue;
            }

            Vec3 accumulated(0.0f, 0.0f, 0.0f);
            if (editableMeshCache.shade_flat) {
                accumulated = faceNormals[refIndex];
            } else if (editableMeshCache.auto_smooth) {
                for (size_t neighborIndex = 0; neighborIndex < refs.size(); ++neighborIndex) {
                    if (faceNormals[refIndex].dot(faceNormals[neighborIndex]) >= autoSmoothThreshold) {
                        accumulated += faceNormals[neighborIndex];
                    }
                }
            } else {
                for (const Vec3& normal : faceNormals) {
                    accumulated += normal;
                }
            }

            const float len = accumulated.length();
            const Vec3 shadingNormal = len > 1e-8f ? accumulated / len : faceNormals[refIndex];
            ref.triangle->setOriginalVertexNormal(ref.corner, shadingNormal);
        }
    };

    for (const size_t vertexId : affectedVertexIds) {
        recomputeVertexNormal(vertexId);
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

    struct TriangleCornerRef {
        std::shared_ptr<Triangle> triangle;
        int corner = 0;
        Vec3 faceNormal;
    };

    std::unordered_map<QuantizedVertexKey, std::vector<TriangleCornerRef>, QuantizedVertexKeyHasher> refsByVertex;
    refsByVertex.reserve(triangles.size() * 2 + 1);

    for (const auto& tri : triangles) {
        if (!tri) {
            continue;
        }

        const Vec3 flatNormal = computeOrientationPreservingFaceNormal(
            *tri,
            tri->getOriginalVertexNormal(0));

        for (int corner = 0; corner < 3; ++corner) {
            refsByVertex[quantizeTopologyVertex(tri->getOriginalVertexPosition(corner))].push_back(
                TriangleCornerRef{ tri, corner, flatNormal });
        }
    }

    const float clampedAngle = std::clamp(autoSmoothAngleDegrees, 1.0f, 180.0f);
    const float autoSmoothThreshold = std::cos(clampedAngle * 3.14159265359f / 180.0f);

    for (auto& bucket : refsByVertex) {
        auto& refs = bucket.second;
        for (size_t i = 0; i < refs.size(); ++i) {
            Vec3 accumulated(0.0f, 0.0f, 0.0f);
            if (flatShading) {
                accumulated = refs[i].faceNormal;
            } else if (autoSmooth) {
                for (size_t j = 0; j < refs.size(); ++j) {
                    if (refs[i].faceNormal.dot(refs[j].faceNormal) >= autoSmoothThreshold) {
                        accumulated += refs[j].faceNormal;
                    }
                }
            } else {
                for (const auto& ref : refs) {
                    accumulated += ref.faceNormal;
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

bool tryMarkEditableTriangleTouched(
    SceneUI::EditableMeshCache& editableMeshCache,
    const Triangle* triangle) {
    if (!triangle) {
        return false;
    }

    const auto found = editableMeshCache.triangle_to_mesh_index.find(triangle);
    if (found == editableMeshCache.triangle_to_mesh_index.end()) {
        return false;
    }

    if (editableMeshCache.triangle_mark_stamps.size() != editableMeshCache.triangle_to_mesh_index.size()) {
        editableMeshCache.triangle_mark_stamps.assign(editableMeshCache.triangle_to_mesh_index.size(), 0u);
        editableMeshCache.triangle_mark_generation = 1u;
    }
    if (editableMeshCache.triangle_mark_generation == std::numeric_limits<uint32_t>::max()) {
        std::fill(
            editableMeshCache.triangle_mark_stamps.begin(),
            editableMeshCache.triangle_mark_stamps.end(),
            0u);
        editableMeshCache.triangle_mark_generation = 1u;
    }

    const size_t triangleIndex = found->second;
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
    if (editableMeshCache.triangle_mark_stamps.size() != editableMeshCache.triangle_to_mesh_index.size()) {
        editableMeshCache.triangle_mark_stamps.assign(editableMeshCache.triangle_to_mesh_index.size(), 0u);
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
    sculpt_stroke_state = SculptStrokeState{};
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
        refreshEditableDisplayMeshFromBase(ctx, active_mesh_edit_object_name, false);
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

bool SceneUI::refreshEditableDisplayMeshFromBase(UIContext& ctx, const std::string& objectName, bool queueGpuSync) {
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
    const EditableMeshSelection preservedSelection =
        (editable_mesh_cache.object_name == objectName) ? editable_mesh_cache.selection : EditableMeshSelection{};
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

    rebuildMeshCache(ctx.scene.world.objects);
    applyMeshShadingSettings(ctx, objectName, false);
    if (editable_mesh_cache.object_name == objectName) {
        editable_mesh_cache.object_name.clear();
        ensureEditableMeshCache(ctx, objectName);
        editable_mesh_cache.selection = preservedSelection;
    }
    updateBBoxCache(objectName);
    objects_needing_cpu_sync.erase(objectName);

    extern bool g_bvh_rebuild_pending;
    g_bvh_rebuild_pending = true;

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
        const MeshModifiers::ModifierStack evalStack =
            buildSubdivisionPreviewEvaluationStack(
                *modifierStack,
                isInteractiveSubdivisionPreviewActiveForObject(objectName));
        size_t expectedTriangleCount = ctx.scene.base_mesh_cache[objectName].size();
        for (const auto& mod : evalStack.modifiers) {
            if (!mod.enabled) {
                continue;
            }
            if (mod.type != MeshModifiers::ModifierType::FlatSubdivision &&
                mod.type != MeshModifiers::ModifierType::SmoothSubdivision) {
                continue;
            }
            for (int level = 0; level < mod.levels; ++level) {
                if (expectedTriangleCount > ((std::numeric_limits<size_t>::max)() / 4u)) {
                    expectedTriangleCount = (std::numeric_limits<size_t>::max)();
                    break;
                }
                expectedTriangleCount *= 4u;
            }
        }

        const bool previewOutOfDate =
            cacheIt == mesh_cache.end() ||
            cacheIt->second.empty() ||
            cacheIt->second.size() != expectedTriangleCount;
        if (previewOutOfDate) {
            refreshEditableDisplayMeshFromBase(ctx, objectName, false);
            cacheIt = mesh_cache.find(objectName);
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
    const bool needsRebuild =
        editable_mesh_cache.object_name != objectName ||
        editable_mesh_cache.source_triangle_count != triangleCount ||
        !(editable_mesh_cache.source_object_transform == currentObjectTransform);

    const MeshShadingSettings& shading = ensureMeshShadingSettings(objectName);

    if (!needsRebuild) {
        editable_mesh_cache.shade_flat = shading.flat_shading;
        editable_mesh_cache.auto_smooth = shading.auto_smooth;
        editable_mesh_cache.auto_smooth_angle_degrees = shading.auto_smooth_angle_degrees;
        return true;
    }

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

    std::unordered_map<QuantizedVertexKey, int, QuantizedVertexKeyHasher> vertexLookup;
    vertexLookup.reserve(triangleCount * 2 + 1);

    std::unordered_set<unsigned long long> edgeLookup;
    edgeLookup.reserve(triangleCount * 2 + 1);
    std::unordered_map<unsigned long long, int> edgeFaceCounts;
    edgeFaceCounts.reserve(triangleCount * 2 + 1);
    std::unordered_map<unsigned long long, std::vector<int>> edgeToTriangleIds;
    edgeToTriangleIds.reserve(triangleCount * 2 + 1);

    if (editableSourceTriangles) {
        for (const auto& tri : *editableSourceTriangles) {
            if (!tri) {
                continue;
            }

            int faceVertexIds[3] = { -1, -1, -1 };
            for (int corner = 0; corner < 3; ++corner) {
                const Vec3 localPos = tri->getOriginalVertexPosition(corner);
                const QuantizedVertexKey key = quantizeTopologyVertex(localPos);
                auto found = vertexLookup.find(key);
                int vertexId = -1;
                if (found == vertexLookup.end()) {
                    vertexId = static_cast<int>(editable_mesh_cache.vertices.size());
                    vertexLookup.emplace(key, vertexId);
                    EditableVertex vertex;
                    vertex.local_position = localPos;
                    editable_mesh_cache.vertices.push_back(vertex);
                } else {
                    vertexId = found->second;
                }

                faceVertexIds[corner] = vertexId;
                editable_mesh_cache.vertices[vertexId].refs.push_back(EditableVertexRef{ tri, corner });
            }

            EditableFace face;
            face.triangle = tri;
            face.v0 = faceVertexIds[0];
            face.v1 = faceVertexIds[1];
            face.v2 = faceVertexIds[2];
            const int triangleFaceId = static_cast<int>(editable_mesh_cache.faces.size());
            editable_mesh_cache.faces.push_back(face);
            editable_mesh_cache.triangle_vertex_ids[tri.get()] = {
                faceVertexIds[0],
                faceVertexIds[1],
                faceVertexIds[2]
            };

            const int edgePairs[3][2] = {
                { face.v0, face.v1 },
                { face.v1, face.v2 },
                { face.v2, face.v0 }
            };
            for (int i = 0; i < 3; ++i) {
                int a = edgePairs[i][0];
                int b = edgePairs[i][1];
                if (a < 0 || b < 0) {
                    continue;
                }
                if (b < a) {
                    std::swap(a, b);
                }

                const unsigned long long packed =
                    (static_cast<unsigned long long>(static_cast<unsigned int>(a)) << 32ull) |
                    static_cast<unsigned long long>(static_cast<unsigned int>(b));
                edgeFaceCounts[packed] += 1;
                edgeToTriangleIds[packed].push_back(triangleFaceId);
                if (edgeLookup.insert(packed).second) {
                    editable_mesh_cache.edges.push_back(EditableEdge{ a, b });
                }
            }
        }
    } else {
        for (const auto& entry : meshEntries) {
            const std::shared_ptr<Triangle>& tri = entry.second;
            if (!tri) {
                continue;
            }

            int faceVertexIds[3] = { -1, -1, -1 };
            for (int corner = 0; corner < 3; ++corner) {
                const Vec3 localPos = tri->getOriginalVertexPosition(corner);
                const QuantizedVertexKey key = quantizeTopologyVertex(localPos);
                auto found = vertexLookup.find(key);
                int vertexId = -1;
                if (found == vertexLookup.end()) {
                    vertexId = static_cast<int>(editable_mesh_cache.vertices.size());
                    vertexLookup.emplace(key, vertexId);
                    EditableVertex vertex;
                    vertex.local_position = localPos;
                    editable_mesh_cache.vertices.push_back(vertex);
                } else {
                    vertexId = found->second;
                }

                faceVertexIds[corner] = vertexId;
                editable_mesh_cache.vertices[vertexId].refs.push_back(EditableVertexRef{ tri, corner });
            }

            EditableFace face;
            face.triangle = tri;
            face.v0 = faceVertexIds[0];
            face.v1 = faceVertexIds[1];
            face.v2 = faceVertexIds[2];
            const int triangleFaceId = static_cast<int>(editable_mesh_cache.faces.size());
            editable_mesh_cache.faces.push_back(face);
            editable_mesh_cache.triangle_vertex_ids[tri.get()] = {
                faceVertexIds[0],
                faceVertexIds[1],
                faceVertexIds[2]
            };

            const int edgePairs[3][2] = {
                { face.v0, face.v1 },
                { face.v1, face.v2 },
                { face.v2, face.v0 }
            };
            for (int i = 0; i < 3; ++i) {
                int a = edgePairs[i][0];
                int b = edgePairs[i][1];
                if (a < 0 || b < 0) {
                    continue;
                }
                if (b < a) {
                    std::swap(a, b);
                }

                const unsigned long long packed =
                    (static_cast<unsigned long long>(static_cast<unsigned int>(a)) << 32ull) |
                    static_cast<unsigned long long>(static_cast<unsigned int>(b));
                edgeFaceCounts[packed] += 1;
                edgeToTriangleIds[packed].push_back(triangleFaceId);
                if (edgeLookup.insert(packed).second) {
                    editable_mesh_cache.edges.push_back(EditableEdge{ a, b });
                }
            }
        }
    }

    editable_mesh_cache.polygon_faces.clear();
    editable_mesh_cache.polygon_faces.reserve(editable_mesh_cache.faces.size());
    std::vector<bool> triangleFaceConsumed(editable_mesh_cache.faces.size(), false);
    for (const auto& edgeEntry : edgeToTriangleIds) {
        const auto& touchingTriangles = edgeEntry.second;
        if (touchingTriangles.size() != 2) {
            continue;
        }

        const int faceAId = touchingTriangles[0];
        const int faceBId = touchingTriangles[1];
        if (faceAId < 0 || faceBId < 0 ||
            faceAId >= static_cast<int>(editable_mesh_cache.faces.size()) ||
            faceBId >= static_cast<int>(editable_mesh_cache.faces.size()) ||
            triangleFaceConsumed[faceAId] || triangleFaceConsumed[faceBId]) {
            continue;
        }

        std::vector<int> mergedVertexIds;
        if (!canMergeEditableTrianglesToQuad(
                editable_mesh_cache,
                editable_mesh_cache.faces[faceAId],
                editable_mesh_cache.faces[faceBId],
                mergedVertexIds)) {
            continue;
        }

        addEditablePolygonFaceFromTriangles(editable_mesh_cache, mergedVertexIds, { faceAId, faceBId });
        triangleFaceConsumed[faceAId] = true;
        triangleFaceConsumed[faceBId] = true;
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
    editable_mesh_cache.vertex_neighbors.clear();
    editable_mesh_cache.vertex_neighbors.resize(editable_mesh_cache.vertices.size());
    editable_mesh_cache.vertex_spatial_buckets.clear();
    editable_mesh_cache.vertex_spatial_buckets.reserve(editable_mesh_cache.vertices.size());
    for (const auto& edge : editable_mesh_cache.edges) {
        if (edge.v0 < 0 || edge.v1 < 0 ||
            edge.v0 >= static_cast<int>(editable_mesh_cache.vertices.size()) ||
            edge.v1 >= static_cast<int>(editable_mesh_cache.vertices.size())) {
            continue;
        }
        editable_mesh_cache.vertex_neighbors[edge.v0].push_back(edge.v1);
        editable_mesh_cache.vertex_neighbors[edge.v1].push_back(edge.v0);

        int a = edge.v0;
        int b = edge.v1;
        if (b < a) {
            std::swap(a, b);
        }
        const unsigned long long packed =
            (static_cast<unsigned long long>(static_cast<unsigned int>(a)) << 32ull) |
            static_cast<unsigned long long>(static_cast<unsigned int>(b));
        const auto countIt = edgeFaceCounts.find(packed);
        if (countIt != edgeFaceCounts.end() && countIt->second <= 1) {
            editable_mesh_cache.vertices[edge.v0].is_boundary = true;
            editable_mesh_cache.vertices[edge.v1].is_boundary = true;
        }
    }
    for (size_t vertexId = 0; vertexId < editable_mesh_cache.vertices.size(); ++vertexId) {
        const EditableSpatialCellKey key = makeEditableSpatialCellKey(
            editable_mesh_cache.vertices[vertexId].local_position,
            editable_mesh_cache.spatial_cell_size);
        editable_mesh_cache.vertex_spatial_buckets[key].push_back(static_cast<int>(vertexId));
    }

    // Build reverse lookup: Triangle* → index in mesh_cache vector (for partial raster sync)
    editable_mesh_cache.triangle_to_mesh_index.clear();
    editable_mesh_cache.triangle_to_mesh_index.reserve(editableSourceTriangles ? editableSourceTriangles->size() : meshEntries.size());
    if (editableSourceTriangles) {
        for (size_t i = 0; i < editableSourceTriangles->size(); ++i) {
            if ((*editableSourceTriangles)[i]) {
                editable_mesh_cache.triangle_to_mesh_index[(*editableSourceTriangles)[i].get()] = i;
            }
        }
    } else {
        for (size_t i = 0; i < meshEntries.size(); ++i) {
            if (meshEntries[i].second) {
                editable_mesh_cache.triangle_to_mesh_index[meshEntries[i].second.get()] = i;
            }
        }
    }
    editable_mesh_cache.vertex_mark_stamps.assign(editable_mesh_cache.vertices.size(), 0u);
    editable_mesh_cache.vertex_mark_generation = 1u;
    editable_mesh_cache.triangle_mark_stamps.assign(editableSourceTriangles ? editableSourceTriangles->size() : meshEntries.size(), 0u);
    editable_mesh_cache.triangle_mark_generation = 1u;

    // Half-edge topology over the welded vertices + polygon faces. Vertex ids
    // and polygon-face ids stay 1:1 with this cache, so operators can move
    // between the two representations without remapping.
    {
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
    }

    return !editable_mesh_cache.vertices.empty();
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
    int vertexCandidate = -1;
    int edgeCandidate = -1;
    int faceCandidate = -1;

    if (tryVertex) {
        const float maxDistanceSq = 14.0f * 14.0f;
        float bestDistanceSq = maxDistanceSq;
        for (size_t i = 0; i < editable_mesh_cache.vertices.size(); ++i) {
            ImVec2 screen;
            if (!projectPointToScreen(*ctx.scene.camera, displaySize,
                                      transform.transform_point(editable_mesh_cache.vertices[i].local_position), screen)) {
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
                                      transform.transform_point(editable_mesh_cache.vertices[edge.v0].local_position), s0) ||
                !projectPointToScreen(*ctx.scene.camera, displaySize,
                                      transform.transform_point(editable_mesh_cache.vertices[edge.v1].local_position), s1)) {
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
                if (!projectPointToScreen(*ctx.scene.camera, displaySize, worldPosition, screenPosition)) {
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
    const std::vector<float> weights = buildSoftSelectionWeights(
        editable_mesh_cache, softSettings, uniqueTargets);

    for (size_t vertexId = 0; vertexId < editable_mesh_cache.vertices.size(); ++vertexId) {
        const float weight = weights[vertexId];
        if (weight <= 1e-5f) {
            continue;
        }

        EditableVertex& vertex = editable_mesh_cache.vertices[vertexId];
        vertex.local_position = vertex.local_position + (localDelta * weight);

        for (const auto& ref : vertex.refs) {
            if (!ref.triangle) {
                continue;
            }

            ref.triangle->setOriginalVertexPosition(ref.corner, vertex.local_position);
            ref.triangle->markAABBDirty();
            if (tryMarkEditableTriangleTouched(editable_mesh_cache, ref.triangle.get())) {
                touchedTriangles.push_back(ref.triangle);
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

    mesh_overlay_cache = MeshOverlayCache{};
    updateBBoxCache(objectName);

    if (preserveModifierPreview) {
        refreshEditableDisplayMeshFromBase(ctx, objectName, ctx.backend_ptr && sculpt_mode_state.accumulate_live);
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
    const std::vector<float> weights = buildSoftSelectionWeights(
        editable_mesh_cache, softSettings, uniqueTargets);

    bool changed = false;
    for (size_t vertexId = 0; vertexId < editable_mesh_cache.vertices.size(); ++vertexId) {
        const float weight = weights[vertexId];
        if (weight <= 1e-5f) {
            continue;
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
            continue;
        }
        changed = true;
        vertex.local_position = updatedLocal;

        for (const auto& ref : vertex.refs) {
            if (!ref.triangle) {
                continue;
            }

            ref.triangle->setOriginalVertexPosition(ref.corner, updatedLocal);
            ref.triangle->markAABBDirty();
            if (tryMarkEditableTriangleTouched(editable_mesh_cache, ref.triangle.get())) {
                touchedTriangles.push_back(ref.triangle);
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

    mesh_overlay_cache = MeshOverlayCache{};
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
                    editable_mesh_cache.faces[triangleId].triangle) {
                    extrudedCapTriangles.insert(editable_mesh_cache.faces[triangleId].triangle.get());
                }
            }
        } else if (faceId >= 0 && faceId < static_cast<int>(editable_mesh_cache.faces.size()) &&
                   editable_mesh_cache.faces[faceId].triangle) {
            extrudedCapTriangles.insert(editable_mesh_cache.faces[faceId].triangle.get());
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
                    editable_mesh_cache.faces[triangleId].triangle) {
                    templateTriangle = editable_mesh_cache.faces[triangleId].triangle;
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
                const Triangle* tri = ref.triangle.get();
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
            if (ref.triangle) {
                templateTriangle = ref.triangle;
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

void SceneUI::handleMeshSculpt(UIContext& ctx) {
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

            refreshMeshEditLayerEditedState(ctx);

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

            updateBBoxCache(sculpt_stroke_state.object_name);
            if (ctx.backend_ptr != nullptr || g_backend != nullptr || g_viewport_backend != nullptr) {
                queueMeshEditGpuSync(sculpt_stroke_state.object_name);
            }
            if (!ctx.backend_ptr) {
                extern bool g_cpu_bvh_refit_pending;
                g_cpu_bvh_refit_pending = true;
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

    if (sculpt_stroke_state.active &&
        sculpt_stroke_state.object_name == objectName &&
        activeTool == SculptBrushTool::Grab &&
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
    const float brushStrength = sanitizeFiniteFloat(sculpt_mode_state.brush.strength, 1.0f, 0.0f, 1.0f);
    const float dt = sanitizeFiniteFloat(io.DeltaTime, 1.0f / 60.0f, 1.0f / 240.0f, 0.25f);
    const float normalStrength = sanitizeFiniteFloat(sculpt_mode_state.normal_strength, 0.35f, 0.0f, 8.0f);
    const float falloff = saturateFloat(sculpt_mode_state.brush.falloff);
    const float directionSign = io.KeyCtrl ? -1.0f : 1.0f;
    const float brushFlow = sanitizeFiniteFloat(sculpt_mode_state.brush.flow, 1.0f, 0.05f, 4.0f);
    const float clayBrushStrength = computeSafeClayStrength(brushStrength, brushFlow);
    const float clayRadiusCompensation = computeClayRadiusCompensation(radiusWorld);

    // Tangent frame for alpha mask UV sampling (nx/ny in brush-local [-1,1] space)
    Vec3 tangentWorld = std::abs(hitNormalWorld.y) < 0.95f
        ? safeNormalizeVec3(hitNormalWorld.cross(Vec3(0, 1, 0)), Vec3(1, 0, 0))
        : safeNormalizeVec3(hitNormalWorld.cross(Vec3(1, 0, 0)), Vec3(0, 0, 1));
    Vec3 bitangentWorld = safeNormalizeVec3(hitNormalWorld.cross(tangentWorld), Vec3(0, 0, 1));
    if (!isFiniteVec3(tangentWorld) || !isFiniteVec3(bitangentWorld)) {
        tangentWorld = Vec3(1, 0, 0);
        bitangentWorld = Vec3(0, 0, 1);
    }

    Vec3 planeHit = hitPoint;
    if (activeTool == SculptBrushTool::Grab) {
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
    // For Grab: use raw 3D delta so the mesh follows the mouse in all directions.
    // Stripping the normal component prevented movement when viewing at grazing angles.
    const Vec3 worldDragDelta = sanitizeVec3(planeHit - sanitizeVec3(sculpt_stroke_state.start_world_hit, planeHit), Vec3(0.0f, 0.0f, 0.0f));
    const Vec3 localGrabDelta = sanitizeVec3(
        inverseTransform.transform_point(planeHit) -
        inverseTransform.transform_point(sanitizeVec3(sculpt_stroke_state.start_world_hit, planeHit)),
        Vec3(0.0f, 0.0f, 0.0f));
    const Vec3 lastStrokeHit = sanitizeVec3(sculpt_stroke_state.last_world_hit, hitPoint);
    const Vec3 strokeStepWorld = sanitizeVec3(hitPoint - lastStrokeHit, Vec3(0.0f, 0.0f, 0.0f));
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
    // Candidate gathering now prefers PBVH traversal for broad-phase culling,
    // but the actual brush solve remains vertex-based for consistent quality.
    // Capsule sweep: non-Grab tool'larda toplama küresini [prevHit, currentHit]
    // segmentini kuşatacak şekilde genişletip merkeziyetlendiriyoruz. Yarıçap
    // kapsüller (3R)'a kadar büyüyor — fare hızlı hareket ettiğinde aradaki
    // vertexleri tek pass ile yakalamamızı sağlar.
    std::vector<int> sculptCandidateVertexIdsStorage;
    Vec3 candidateCenterLocal;
    float candidateCollectionRadius;
    if (activeTool == SculptBrushTool::Grab) {
        candidateCenterLocal = localGrabStartPoint;
        candidateCollectionRadius = localRadius;
    } else {
        candidateCenterLocal = (localClaySamplePoint + localPrevHitPoint) * 0.5f;
        const float halfSeg = (localClaySamplePoint - localPrevHitPoint).length() * 0.5f;
        const float cappedHalf = (std::min)(halfSeg, localRadius * 2.0f);
        candidateCollectionRadius = localRadius + cappedHalf;
    }
    sculptCandidateVertexIdsStorage = collectSculptCandidateVerticesWithPBVHFallback(
        sculpt_pbvh,
        editable_mesh_cache,
        candidateCenterLocal,
        candidateCollectionRadius);
    if (activeTool == SculptBrushTool::Draw || activeTool == SculptBrushTool::Inflate) {
        const float largeBrushFactor = computeLargeBrushProjectionBlend(radiusWorld);
        if (largeBrushFactor > 0.08f) {
            const int expandRings = largeBrushFactor > 0.55f ? 2 : 1;
            sculptCandidateVertexIdsStorage = expandEditableCandidateVerticesByTopology(
                editable_mesh_cache,
                sculptCandidateVertexIdsStorage,
                expandRings);
        }
    }
    if (activeTool == SculptBrushTool::Grab) {
        // Prime the grab set once, from the PBVH-gathered candidates at the stroke's
        // first frame.
        if (sculpt_stroke_state.grab_weights_by_vertex.empty()) {
            primeGrabStrokeWeights(
                sculpt_stroke_state,
                editable_mesh_cache,
                transform,
                sculptCandidateVertexIdsStorage,
                sanitizeVec3(sculpt_stroke_state.start_world_hit, hitPoint),
                hitNormalWorld,
                radiusWorld,
                sculpt_mode_state.front_faces_only,
                mesh_overlay_settings.proportional_falloff_type);
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
    const std::vector<int>& sculptCandidateVertexIds = sculptCandidateVertexIdsStorage;
    sculpt_control_graph.last_candidate_node_count = sculpt_pbvh.last_candidate_node_count;
    sculpt_control_graph.last_candidate_vertex_count = sculptCandidateVertexIds.size();
    std::vector<int> strokeTouchedVertexIds = sculptCandidateVertexIds;
    strokeTouchedVertexIds.reserve(sculptCandidateVertexIds.size() * 4 + 32);

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
    const size_t sculptVertexCount = editable_mesh_cache.vertices.size();
    if (sculpt_updated_local_positions.size() != sculptVertexCount) {
        sculpt_updated_local_positions.resize(sculptVertexCount);
        for (size_t i = 0; i < sculptVertexCount; ++i) {
            sculpt_updated_local_positions[i] = editable_mesh_cache.vertices[i].local_position;
        }
    }
    for (const int vid : sculptCandidateVertexIds) {
        if (vid >= 0 && static_cast<size_t>(vid) < sculptVertexCount) {
            sculpt_updated_local_positions[static_cast<size_t>(vid)] =
                editable_mesh_cache.vertices[static_cast<size_t>(vid)].local_position;
        }
    }

    const Vec3 planePoint = hitPoint;
    // Capsule sweep'in segment ucu: bu frame'in dab merkezi (planePoint) ile bir
    // önceki dab merkezi arasındaki boşluğu kapatıyoruz. Stroke ilk frame'inde
    // last_world_hit == hitPoint olduğu için segment dejenere — eski davranış.
    const Vec3 prevPlanePoint = sanitizeVec3(sculpt_stroke_state.last_world_hit, hitPoint);
    std::atomic<bool> basePassChanged{ false };
    forEachEditableCandidate(sculptCandidateVertexIds, [&](int vertexIdInt) {
        if (vertexIdInt < 0) {
            return;
        }
        const size_t vertexId = static_cast<size_t>(vertexIdInt);
        EditableVertex& vertex = editable_mesh_cache.vertices[vertexId];
        if (activeTool == SculptBrushTool::Grab) {
            if (applyGrabSculptCandidate(
                    sculpt_stroke_state,
                    brushStrength,
                    localGrabDelta,
                    vertexIdInt,
                    sculpt_updated_local_positions[vertexId])) {
                basePassChanged.store(true, std::memory_order_relaxed);
            }
            return;
        }

        const Vec3 snapshotLocalPosition = editable_mesh_cache.vertices[vertexId].local_position;
        const Vec3 snapshotWorldPosition = sanitizeVec3(
            transform.transform_point(snapshotLocalPosition),
            Vec3(0.0f, 0.0f, 0.0f));

        SculptBrushSample sample;
        if (!computeSculptBrushSampleAtWorldPoint(
                snapshotLocalPosition,
                snapshotWorldPosition,
                planePoint,
                prevPlanePoint,
                hitNormalWorld,
                tangentWorld,
                bitangentWorld,
                sculpt_mode_state.brush,
                mesh_overlay_settings.proportional_falloff_type,
                activeTool,
                falloff,
                brushStrength,
                radiusWorld,
                strokeDistance,
                strokeSpacing,
                dt,
                sculpt_stroke_state.changed,
                sculpt_mode_state.front_faces_only,
                sample)) {
            return;
        }

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
                brushStrength,
                clayBrushStrength,
                clayRadiusCompensation,
                normalStrength,
                directionSign,
                radiusWorld,
                dt,
                localRadius,
                strokeAdvanceFactor,
                vertexIdInt,
                sample,
                sculpt_updated_local_positions[vertexId])) {
            basePassChanged.store(true, std::memory_order_relaxed);
        }
    });
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
                        if (!editable_mesh_cache.vertices[i].refs.empty() && editable_mesh_cache.vertices[i].refs[0].triangle) {
                            const auto& ref = editable_mesh_cache.vertices[i].refs[0];
                            n = ref.triangle->vertices[ref.corner].normal;
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
    if (sculpt_mode_state.mirror_x || sculpt_mode_state.mirror_y || sculpt_mode_state.mirror_z) {
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

            // Mirror drag delta for Grab (flip each axis in local, then back to world).
            Vec3 mirWorldDrag = worldDragDelta;
            if (activeTool == SculptBrushTool::Grab) {
                Vec3 localDrag = sanitizeVec3(inverseTransform.transform_vector(worldDragDelta), Vec3(0.0f, 0.0f, 0.0f));
                if (do_mx) localDrag.x = -localDrag.x;
                if (do_my) localDrag.y = -localDrag.y;
                if (do_mz) localDrag.z = -localDrag.z;
                mirWorldDrag = sanitizeVec3(transform.transform_vector(localDrag), Vec3(0.0f, 0.0f, 0.0f));
            }

            // Grab: mirror start hit and compute weights on-the-fly.
            if (activeTool == SculptBrushTool::Grab) {
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
                    for (const int vid : mirrorSet) {
                        if (vid >= 0 && static_cast<size_t>(vid) < editable_mesh_cache.vertices.size()) {
                            sculpt_stroke_state.grab_start_local_positions.try_emplace(
                                vid, editable_mesh_cache.vertices[vid].local_position);
                        }
                    }
                }
                const std::vector<int>& mirrorGrabCandidateVertexIds = mirrorSet;
                // Re-seed the persistent buffer for this combo's mirror candidates each frame.
                for (const int vid : mirrorGrabCandidateVertexIds) {
                    if (vid >= 0 && static_cast<size_t>(vid) < editable_mesh_cache.vertices.size()) {
                        sculpt_updated_local_positions[vid] = editable_mesh_cache.vertices[vid].local_position;
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
                    const float w = applyFalloffCurve(
                        1.0f - saturateFloat(planarDist / radiusWorld),
                        mesh_overlay_settings.proportional_falloff_type);
                    if (w <= 1e-5f) {
                        return;
                    }
                    const float grabStrength = std::clamp(brushStrength * 0.5f, 0.0f, 5.0f);
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
                localRadius);
            // Re-seed persistent buffer for mirror candidates from the cache (frame-start).
            for (const int vid : mirrorCandidateVertexIds) {
                if (vid >= 0 && static_cast<size_t>(vid) < editable_mesh_cache.vertices.size()) {
                    sculpt_updated_local_positions[static_cast<size_t>(vid)] =
                        editable_mesh_cache.vertices[static_cast<size_t>(vid)].local_position;
                }
            }
            strokeTouchedVertexIds.insert(
                strokeTouchedVertexIds.end(),
                mirrorCandidateVertexIds.begin(),
                mirrorCandidateVertexIds.end());
            std::atomic<bool> mirrorPassChanged{ false };
            forEachEditableCandidate(mirrorCandidateVertexIds, [&](int vertexIdInt) {
                if (vertexIdInt < 0) {
                    return;
                }
                const size_t vertexId = static_cast<size_t>(vertexIdInt);
                EditableVertex& vertex = editable_mesh_cache.vertices[vertexId];
                const Vec3 snapshotLocalPosition = resolveEditableSnapshotLocalPosition(
                    editable_mesh_cache,
                    kEmptySnapshot,
                    vertexId);
                const Vec3 worldPos = sanitizeVec3(
                    transform.transform_point(snapshotLocalPosition),
                    mirWorldCenter);
                // Capsule sweep mirror eşdeğeri: segment [mirWorldPrevCenter -> mirWorldCenter].
                const Vec3 mirSegDir = mirWorldCenter - mirWorldPrevCenter;
                const float mirSegLenSq = mirSegDir.dot(mirSegDir);
                Vec3 mirClosestCenter;
                if (mirSegLenSq > 1e-10f) {
                    const float mirT = std::clamp(
                        (worldPos - mirWorldPrevCenter).dot(mirSegDir) / mirSegLenSq,
                        0.0f,
                        1.0f);
                    mirClosestCenter = mirWorldPrevCenter + mirSegDir * mirT;
                } else {
                    mirClosestCenter = mirWorldCenter;
                }
                const Vec3 toVertex = worldPos - mirClosestCenter;
                const float h = toVertex.dot(mirWorldNormal);
                const Vec3 planarOffset = toVertex - mirWorldNormal * h;
                const float planarDist = computeHybridBrushDistance(
                    toVertex,
                    mirWorldNormal,
                    planarOffset.length(),
                    radiusWorld);
                if (!std::isfinite(h) || !std::isfinite(planarDist) || planarDist > radiusWorld) {
                    return;
                }
                const float frontFacePenalty = computeFrontFaceBrushPenalty(
                    h,
                    radiusWorld,
                    sculpt_mode_state.front_faces_only);
                if (frontFacePenalty <= 1e-5f) {
                    return;
                }
                float w = computeTerrainLikeBrushWeight(planarDist / radiusWorld, falloff);
                w *= applyFalloffCurve(1.0f - saturateFloat(planarDist / radiusWorld), mesh_overlay_settings.proportional_falloff_type);
                w *= frontFacePenalty;
                w *= computeRepeatedStrokeDamping(
                    strokeDistance,
                    radiusWorld,
                    dt,
                    sculpt_stroke_state.changed);
                if (!std::isfinite(w) || w <= 1e-5f) {
                    return;
                }

                // Directional drag factor for mirror pass Clay/ClayStrips
                float mirDragFactor = 1.0f;
                if ((activeTool == SculptBrushTool::Clay || activeTool == SculptBrushTool::ClayStrips)
                    && strokeDistance > strokeSpacing * 0.15f) {
                    const float tangentNorm = planarOffset.dot(strokeTangentWorld) / radiusWorld;
                    mirDragFactor = std::clamp(0.5f + tangentNorm * 1.1f, 0.0f, 1.6f);
                }

                Vec3 wd(0.0f, 0.0f, 0.0f);
                switch (activeTool) {
                case SculptBrushTool::Inflate:
                    wd = mirWorldNormal * (radiusWorld * 0.22f * brushStrength * dt * w * (1.0f + normalStrength) * directionSign);
                    break;
                case SculptBrushTool::Draw: {
                    Vec3 vn = mirWorldNormal;
                    if (!vertex.refs.empty() && vertex.refs[0].triangle) {
                        const Vec3 n = vertex.refs[0].triangle->vertices[vertex.refs[0].corner].normal;
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
                    if (!vertex.refs.empty() && vertex.refs[0].triangle) {
                        const Vec3 n = vertex.refs[0].triangle->vertices[vertex.refs[0].corner].normal;
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

    if (!changed) {
        sculpt_stroke_state.last_world_hit = (activeTool == SculptBrushTool::Grab) ? planeHit : hit.point;
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

    if (activeTool == SculptBrushTool::Clay || activeTool == SculptBrushTool::ClayStrips) {
        const float largeBrushSurfaceFactor = computeLargeBrushSurfaceFactor(radiusWorld);
        const std::vector<int>& polishVertexIds =
            !expandedStrokeVertexIds.empty() ? expandedStrokeVertexIds : strokeTouchedVertexIds;
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
    } else if (activeTool == SculptBrushTool::Draw && !expandedStrokeVertexIds.empty()) {
        const float largeBrushSurfaceFactor = computeLargeBrushSurfaceFactor(radiusWorld);
        const size_t touchedCount = expandedStrokeVertexIds.size();
        const float densityFactor = std::clamp(
            (static_cast<float>(touchedCount) - 24.0f) / 72.0f, 0.0f, 1.0f);
        const float cleanupBoost = (std::max)(largeBrushSurfaceFactor, densityFactor);
        if (cleanupBoost > 0.04f) {
            applyClayPolishPass(
                editable_mesh_cache,
                sculpt_updated_local_positions,
                expandedStrokeVertexIds,
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
                expandedStrokeVertexIds,
                transform,
                inverseTransform,
                hitNormalWorld,
                localRadius,
                0.24f + 0.16f * cleanupBoost,
                0.08f + 0.08f * cleanupBoost);
        }
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
        if ((sculpt_updated_local_positions[vertexId] - vertex.local_position).length_squared() <= 1e-14f) {
            continue;
        }
        Vec3 safeLocalPosition = vertex.local_position;
        if (!resolveEditableTopologySafePosition(
                editable_mesh_cache,
                vertexIdInt,
                sculpt_updated_local_positions,
                strokeTouchedVertexIds,
                safeLocalPosition)) {
            continue;
        }
        vertex.local_position = safeLocalPosition;
        sculpt_updated_local_positions[vertexId] = safeLocalPosition;
        for (const auto& ref : vertex.refs) {
            if (!ref.triangle) {
                continue;
            }
            sculpt_stroke_state.before_triangle_states.try_emplace(
                ref.triangle.get(),
                MeshEditTriangleState{
                    ref.triangle,
                    {
                        ref.triangle->getOriginalVertexPosition(0),
                        ref.triangle->getOriginalVertexPosition(1),
                        ref.triangle->getOriginalVertexPosition(2)
                    }
                });
            ref.triangle->setOriginalVertexPosition(ref.corner, vertex.local_position);
            ref.triangle->markAABBDirty();
            if (tryMarkEditableTriangleTouched(editable_mesh_cache, ref.triangle.get())) {
                touchedTriangles.push_back(ref.triangle);
            }
        }
    }
    // Restore the cache-mirror invariant for verts the clay polish / anti-pit passes
    // wrote but the commit loop did NOT push to the cache: those passes operate on
    // expandedStrokeVertexIds (a superset of strokeTouchedVertexIds), while commit
    // only iterates strokeTouchedVertexIds. Without this, the persistent buffer would
    // carry stale positions into the next frame's neighbor reads. O(expanded), cheap;
    // committed verts already match the cache so resetting them here is a no-op.
    for (const int vid : expandedStrokeVertexIds) {
        if (vid >= 0 && static_cast<size_t>(vid) < editable_mesh_cache.vertices.size()) {
            sculpt_updated_local_positions[static_cast<size_t>(vid)] =
                editable_mesh_cache.vertices[static_cast<size_t>(vid)].local_position;
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

    // Track dirty triangle indices for partial raster sync (avoids full-mesh extraction)
    sculpt_dirty_mesh_cache_indices.clear();
    sculpt_dirty_mesh_cache_indices.reserve(touchedTriangles.size());
    for (const auto& tri : touchedTriangles) {
        auto it = editable_mesh_cache.triangle_to_mesh_index.find(tri.get());
        if (it != editable_mesh_cache.triangle_to_mesh_index.end()) {
            sculpt_dirty_mesh_cache_indices.push_back(it->second);
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
    sculpt_stroke_state.last_world_hit = (activeTool == SculptBrushTool::Grab) ? planeHit : hit.point;
}

void SceneUI::processPendingMeshEditGpuSync(UIContext& ctx) {
    if (!mesh_edit_gpu_sync_pending || (!ctx.backend_ptr && !g_backend && !g_viewport_backend)) {
        return;
    }

    const std::string objectName = mesh_edit_gpu_sync_object_name;
    mesh_edit_gpu_sync_pending = false;
    mesh_edit_gpu_sync_object_name.clear();

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

    // CPU picking/CPU-render BVH only needs a fast REFIT for a position-stable mesh
    // edit (sculpt dab / vertex drag) — true in both Solid and Rendered modes. refitBVH
    // does an Embree RTC_BUILD_QUALITY_REFIT and self-escalates to a full rebuild if the
    // triangle topology actually changed, so this stays correct for edit-mode topology
    // ops too. Rendered mode previously forced a per-dab async full-scene rebuild here
    // (+ "Rebuilding BVH..." HUD spam) even though the GPU side already refits.
    g_cpu_bvh_refit_pending = true;

    // In Solid/Matcap sculpt we only need the interactive raster viewport to stay live.
    // Syncing the inactive render backend here (OptiX / Vulkan RT / CPU render backend)
    // duplicates geometry work every dab and can make Solid mode slower than Rendered.
    // We already mark geometry dirty at stroke end, so Rendered mode can catch up when
    // the user exits sculpt or switches viewport mode.
    if (activeViewportInSolidMode) {
        sculpt_dirty_mesh_cache_indices.clear();
        ctx.renderer.resetCPUAccumulation();
        ctx.start_render = true;
        return;
    }

    if (renderBackendIsVulkan) {
        const auto triangles = Viewport::collectMeshTrianglesForObject(mesh_cache, objectName);
        // Keep Vulkan RT mesh edits on the robust full-rebuild path for now.
        // Backend switching proves the rebuild path is correct, while incremental
        // RT updates still corrupt transformed editable meshes.
        const bool allowVulkanIncrementalMeshEdit = kEnableVulkanInteractiveMeshRtUpdates;

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

    // Skip expensive wireframe overlay during active sculpt stroke —
    // the brush preview circle is sufficient visual feedback and drawing
    // edges/vertices for 1M+ triangle meshes kills frame rate.
    if (sculpt_stroke_state.active && sculpt_mode_state.enabled &&
        mesh_workspace_mode == MeshWorkspaceMode::Sculpt) {
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
    if (editContextMenuAllowed && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
        ImGui::OpenPopup("EditMeshContextMenu");
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
                    screenPosition)) {
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
                if (!projectPointToScreen(*ctx.scene.camera, displaySize, v0, s0) ||
                    !projectPointToScreen(*ctx.scene.camera, displaySize, v1, s1) ||
                    !projectPointToScreen(*ctx.scene.camera, displaySize, v2, s2)) {
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
                if (!projectPointToScreen(*ctx.scene.camera, displaySize, a, sa) ||
                    !projectPointToScreen(*ctx.scene.camera, displaySize, b, sb)) {
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
                if (!projectPointToScreen(*ctx.scene.camera, displaySize, p, screen)) {
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

        if (mesh_overlay_settings.edit_mode && mesh_overlay_settings.proportional_edit) {
            bool hasCenter = false;
            const Vec3 centerWorld = getSelectedMeshElementWorldPosition(ctx, &hasCenter);
            if (hasCenter) {
                ImVec2 centerScreen;
                if (projectPointToScreen(*ctx.scene.camera, displaySize, centerWorld, centerScreen)) {
                    // Approximate projected proportional radius using a point offset in local/world X.
                    Vec3 radiusWorld = centerWorld + Vec3(mesh_overlay_settings.proportional_radius, 0.0f, 0.0f);
                    ImVec2 radiusScreen;
                    if (projectPointToScreen(*ctx.scene.camera, displaySize, radiusWorld, radiusScreen)) {
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

    // Build tangent frame from the hit normal
    const Vec3 normal = hit.normal.length_squared() > 1e-8f ? hit.normal.normalize() : Vec3(0, 1, 0);
    Vec3 tangent = std::abs(normal.y) < 0.95f
        ? normal.cross(Vec3(0, 1, 0)).normalize()
        : normal.cross(Vec3(1, 0, 0)).normalize();
    Vec3 bitangent = normal.cross(tangent).normalize();

    ImGuiIO& io = ImGui::GetIO();
    Camera& cam = *ctx.scene.camera;
    const float win_w = io.DisplaySize.x;
    const float win_h = io.DisplaySize.y;

    auto project = [&](const Vec3& p) -> ImVec2 {
        const Vec3 cam_forward = (cam.lookat - cam.lookfrom).normalize();
        const Vec3 cam_right   = cam_forward.cross(cam.vup).normalize();
        const Vec3 cam_up      = cam_right.cross(cam_forward).normalize();
        const float fov_rad    = cam.vfov * 3.14159f / 180.0f;
        const Vec3 to_p = p - cam.lookfrom;
        const float depth = to_p.dot(cam_forward);
        if (depth <= 0.1f) return ImVec2(-1000.0f, -1000.0f);
        const float half_h = depth * tanf(fov_rad * 0.5f);
        const float half_w = half_h * (win_w / (std::max)(1.0f, win_h));
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

    // --- Alpha grid (skip for ghost pass to keep it lightweight) ---
    if (!ghost) {
        const int grid = std::clamp(static_cast<int>(approx_screen_radius * 1.1f), 24, 80);
        const float cell_span = 2.0f / static_cast<float>(grid);
        const float falloff_val = std::clamp(brush.falloff, 0.0f, 1.0f);
        const float inner = std::clamp(1.0f - falloff_val, 0.0f, 1.0f);

        for (int gy = 0; gy < grid; ++gy) {
            for (int gx = 0; gx < grid; ++gx) {
                const float nx = ((static_cast<float>(gx) + 0.5f) / static_cast<float>(grid)) * 2.0f - 1.0f;
                const float ny = ((static_cast<float>(gy) + 0.5f) / static_cast<float>(grid)) * 2.0f - 1.0f;
                const float rr = std::sqrt(nx * nx + ny * ny);
                if (rr > 1.0f) continue;

                float base = 1.0f;
                if (rr > inner) {
                    const float t = std::clamp((rr - inner) / (std::max)(0.001f, 1.0f - inner), 0.0f, 1.0f);
                    base = 1.0f - (t * t * (3.0f - 2.0f * t));
                }

                const float mask_alpha = uiSampleBrushMask(brush, nx, ny);
                const float alpha = base * mask_alpha;
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

    // --- Rings (outer + inner falloff) ---
    auto draw_ring = [&](float radius, ImU32 color, float thickness) {
        ImVec2 prev;
        bool has_prev = false;
        for (int i = 0; i <= segments; ++i) {
            const float angle = (static_cast<float>(i) / static_cast<float>(segments)) * 6.2831853f;
            const Vec3 offset = tangent * std::cos(angle) * radius + bitangent * std::sin(angle) * radius;
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

    const float inner_radius = world_radius * std::clamp(1.0f - brush.falloff, 0.15f, 0.95f);
    const ImU32 outer_color = ghost ? IM_COL32(255, 170, 64, 100) : IM_COL32(255, 170, 64, 230);
    const ImU32 inner_color = ghost ? IM_COL32(255, 170, 64,  60) : IM_COL32(255, 170, 64, 120);
    draw_ring(world_radius,  outer_color, ghost ? 1.2f : 2.0f);
    draw_ring(inner_radius,  inner_color, 1.0f);

    // Center dot
    if (!ghost) {
        dl->AddCircleFilled(center_screen, 3.5f, IM_COL32(255, 170, 64, 230));
    }
}
