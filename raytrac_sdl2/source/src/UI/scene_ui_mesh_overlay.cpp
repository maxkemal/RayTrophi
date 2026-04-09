#include "scene_ui.h"

#include "Backend/OptixBackend.h"
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

    const float aspect = displaySize.x / displaySize.y;
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
    return 1.0f + std::sqrt(overshoot) * 0.55f;
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
    return std::clamp(1.0f / std::sqrt(normalizedRadius), 0.42f, 1.0f);
}
Vec3 sanitizeVec3(const Vec3& value, const Vec3& fallback) {
    return isFiniteVec3(value) ? value : fallback;
}
Vec3 computeBoundarySafeSmoothDelta(
    const SceneUI::EditableMeshCache& cache,
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
        average += neighbor.local_position;
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

    Vec3 smoothDelta = sanitizeVec3(
        (average - vertex.local_position) * smoothFactor,
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
            const auto& neighbor = cache.vertices[static_cast<size_t>(neighborId)];
            if (vertex.is_boundary && !neighbor.is_boundary) {
                continue;
            }
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
        relaxWorld *= safePolish * (vertex.is_boundary ? 0.35f : 1.0f);

        Vec3 relaxLocal = sanitizeVec3(inverseTransform.transform_vector(relaxWorld), Vec3(0.0f, 0.0f, 0.0f));
        const float deltaLen = relaxLocal.length();
        if (std::isfinite(deltaLen) && deltaLen > maxDelta) {
            relaxLocal *= (maxDelta / deltaLen);
        }

        updatedLocalPositions[vertexId] = sanitizeVec3(currentLocal + relaxLocal, currentLocal);
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

template <typename Fn>
void forEachEditableCandidate(const std::vector<int>& candidateVertexIds, Fn&& fn) {
    if (candidateVertexIds.size() >= 384u) {
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

void recomputeEditableSmoothNormals(
    SceneUI::EditableMeshCache& editableMeshCache,
    const std::vector<size_t>& affectedVertexIds) {
    if (affectedVertexIds.empty()) {
        return;
    }

    auto recomputeVertexNormal = [&](const size_t vertexId) {
        if (vertexId >= editableMeshCache.vertices.size()) {
            return;
        }

        const auto& refs = editableMeshCache.vertices[vertexId].refs;
        if (refs.empty()) {
            return;
        }

        std::vector<Vec3> faceNormals;
        faceNormals.reserve(refs.size());
        for (const auto& ref : refs) {
            if (!ref.triangle) {
                faceNormals.push_back(Vec3(0.0f, 1.0f, 0.0f));
                continue;
            }

            const Vec3 faceNormal = Vec3::cross(
                ref.triangle->vertices[1].original - ref.triangle->vertices[0].original,
                ref.triangle->vertices[2].original - ref.triangle->vertices[0].original);
            const float len = faceNormal.length();
            faceNormals.push_back(len > 1e-8f ? faceNormal / len : Vec3(0.0f, 1.0f, 0.0f));
        }

        const float clampedAngle = std::clamp(editableMeshCache.auto_smooth_angle_degrees, 1.0f, 180.0f);
        const float autoSmoothThreshold = std::cos(clampedAngle * 3.14159265359f / 180.0f);

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

    if (affectedVertexIds.size() >= 384u) {
        std::for_each(
            std::execution::par_unseq,
            affectedVertexIds.begin(),
            affectedVertexIds.end(),
            recomputeVertexNormal);
        return;
    }

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

        const Vec3 flatNormalRaw = Vec3::cross(
            tri->getOriginalVertexPosition(1) - tri->getOriginalVertexPosition(0),
            tri->getOriginalVertexPosition(2) - tri->getOriginalVertexPosition(0));
        const float flatLen = flatNormalRaw.length();
        const Vec3 flatNormal = flatLen > 1e-8f ? flatNormalRaw / flatLen : Vec3(0.0f, 1.0f, 0.0f);

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

// Full Vulkan RT rebuilds are currently the only reliable path for editable meshes.
// Incremental RT mesh updates remain disabled until BLAS/TLAS transform ownership is
// made consistent for source triangles, instances, and transformed edit-mode meshes.
constexpr bool kEnableVulkanInteractiveMeshRtUpdates = false;

} // namespace

void SceneUI::clearEditableMeshSelection() {
    editable_mesh_cache.selection = EditableMeshSelection{};
}

void SceneUI::resetMeshEditState(UIContext& ctx) {
    const std::string previewObjectName = active_mesh_edit_object_name;
    if (!previewObjectName.empty()) {
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

    if (queueGpuSync && ctx.backend_ptr) {
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

    const std::vector<std::shared_ptr<Triangle>> displayMesh = evaluateDisplayMeshFromBase(baseIt->second, stack);
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

    if (queueGpuSync && ctx.backend_ptr) {
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
        size_t expectedTriangleCount = ctx.scene.base_mesh_cache[objectName].size();
        for (const auto& mod : modifierStack->modifiers) {
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
    editable_mesh_cache.object_name = objectName;
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

    return !editable_mesh_cache.vertices.empty();
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

    if (ctx.selection.mesh_element_mode == MeshElementSelectMode::Vertex) {
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
                pickedSelection.active_vertex_id = static_cast<int>(i);
                handled = true;
            }
        }
    } else if (ctx.selection.mesh_element_mode == MeshElementSelectMode::Edge) {
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
                pickedSelection.active_edge_id = static_cast<int>(i);
                handled = true;
            }
        }
    } else if (ctx.selection.mesh_element_mode == MeshElementSelectMode::Face) {
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
                pickedSelection.active_face_id = static_cast<int>(i);
                handled = true;
            }
        }
    }

    if (handled) {
        const bool appendSelection = ImGui::GetIO().KeyCtrl;
        const bool ringSelection = ImGui::GetIO().KeyAlt && ImGui::GetIO().KeyShift;
        const bool loopSelection = ImGui::GetIO().KeyAlt && !ImGui::GetIO().KeyShift;
        EditableMeshSelection& selection = editable_mesh_cache.selection;

        if (ctx.selection.mesh_element_mode == MeshElementSelectMode::Vertex && pickedSelection.active_vertex_id >= 0) {
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
        } else if (ctx.selection.mesh_element_mode == MeshElementSelectMode::Edge && pickedSelection.active_edge_id >= 0) {
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
        } else if (ctx.selection.mesh_element_mode == MeshElementSelectMode::Face && pickedSelection.active_face_id >= 0) {
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
    const std::vector<float> weights = buildSoftSelectionWeights(
        editable_mesh_cache, mesh_overlay_settings, uniqueTargets);

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

    if (ctx.backend_ptr && sculpt_mode_state.accumulate_live) {
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
    const std::vector<float> weights = buildSoftSelectionWeights(
        editable_mesh_cache, mesh_overlay_settings, uniqueTargets);

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

    if (ctx.backend_ptr) {
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
    std::vector<std::shared_ptr<Triangle>> extrudedMesh = preserveModifierPreview
        ? cloneTriangleVectorForEdit(currentBaseMesh)
        : cloneTriangleVectorForEdit(currentDisplayMesh);
    if (extrudedMesh.empty()) {
        return false;
    }

    std::vector<int> selectedFaceIds = editable_mesh_cache.selection.face_ids;
    std::sort(selectedFaceIds.begin(), selectedFaceIds.end());
    selectedFaceIds.erase(std::unique(selectedFaceIds.begin(), selectedFaceIds.end()), selectedFaceIds.end());

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
            ctx.selection.mesh_element_mode = MeshElementSelectMode::Face;
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

    std::unordered_set<int> selectedEdgeSet(ringEdgeIds.begin(), ringEdgeIds.end());
    std::unordered_map<unsigned long long, int> edgeIdByKey;
    edgeIdByKey.reserve(editable_mesh_cache.polygon_edges.size() * 2 + 1);
    for (size_t edgeId = 0; edgeId < editable_mesh_cache.polygon_edges.size(); ++edgeId) {
        const auto& edge = editable_mesh_cache.polygon_edges[edgeId];
        edgeIdByKey[makeEditablePackedEdgeKey(edge.v0, edge.v1)] = static_cast<int>(edgeId);
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

    struct PolygonBuildInput {
        std::vector<Vec3> vertices;
        std::shared_ptr<Triangle> template_triangle;
    };
    struct LoopCutSelectionTarget {
        Vec3 center = Vec3(0.0f, 0.0f, 0.0f);
        Vec3 direction = Vec3(1.0f, 0.0f, 0.0f);
    };

    std::vector<PolygonBuildInput> polygonsToBuild;
    polygonsToBuild.reserve(editable_mesh_cache.polygon_faces.size() * 2 + editable_mesh_cache.faces.size());
    std::vector<LoopCutSelectionTarget> cutSelectionTargets;

    auto resolveTemplateTriangle = [&](int faceId) -> std::shared_ptr<Triangle> {
        if (faceId >= 0 && faceId < static_cast<int>(editable_mesh_cache.polygon_faces.size())) {
            const auto& polygonFace = editable_mesh_cache.polygon_faces[faceId];
            for (const int triangleId : polygonFace.triangle_ids) {
                if (triangleId >= 0 &&
                    triangleId < static_cast<int>(editable_mesh_cache.faces.size()) &&
                    editable_mesh_cache.faces[triangleId].triangle) {
                    return editable_mesh_cache.faces[triangleId].triangle;
                }
            }
        }
        return preserveModifierPreview
            ? (!currentBaseMesh.empty() ? currentBaseMesh.front() : nullptr)
            : (!currentDisplayMesh.empty() ? currentDisplayMesh.front() : nullptr);
    };

    int splitFaceCount = 0;
    const size_t polygonFaceCount =
        editable_mesh_cache.polygon_faces.empty()
            ? editable_mesh_cache.faces.size()
            : editable_mesh_cache.polygon_faces.size();
    for (size_t faceId = 0; faceId < polygonFaceCount; ++faceId) {
        const std::vector<int> vertexIds = getEditablePolygonVertexIds(editable_mesh_cache, static_cast<int>(faceId));
        if (vertexIds.size() < 3) {
            continue;
        }

        std::vector<Vec3> faceVertices;
        faceVertices.reserve(vertexIds.size());
        bool invalidFace = false;
        for (const int vertexId : vertexIds) {
            if (!isEditableVertexIdValid(editable_mesh_cache, vertexId)) {
                invalidFace = true;
                break;
            }
            faceVertices.push_back(editable_mesh_cache.vertices[vertexId].local_position);
        }
        if (invalidFace) {
            continue;
        }

        std::shared_ptr<Triangle> templateTriangle = resolveTemplateTriangle(static_cast<int>(faceId));
        if (!templateTriangle) {
            continue;
        }

        bool splitThisFace = false;
        int selectedEdgeIndex = -1;
        if (vertexIds.size() == 4) {
            std::vector<int> faceEdgeIds;
            faceEdgeIds.reserve(4);
            for (size_t i = 0; i < 4; ++i) {
                faceEdgeIds.push_back(findEditablePolygonEdgeId(
                    editable_mesh_cache,
                    edgeIdByKey,
                    vertexIds[i],
                    vertexIds[(i + 1) % 4]));
            }

            std::vector<int> selectedEdgeIndices;
            for (int i = 0; i < 4; ++i) {
                if (faceEdgeIds[i] >= 0 && selectedEdgeSet.count(faceEdgeIds[i]) > 0) {
                    selectedEdgeIndices.push_back(i);
                }
            }

            if (selectedEdgeIndices.size() == 2 &&
                ((selectedEdgeIndices[0] + 2) % 4 == selectedEdgeIndices[1] ||
                 (selectedEdgeIndices[1] + 2) % 4 == selectedEdgeIndices[0])) {
                splitThisFace = true;
                selectedEdgeIndex = selectedEdgeIndices[0];
            }
        }

        if (!splitThisFace) {
            polygonsToBuild.push_back(PolygonBuildInput{ faceVertices, templateTriangle });
            continue;
        }

        auto computeDirectedCutPoint = [&](int startVertexId, int endVertexId) {
            const Vec3& startPosition = editable_mesh_cache.vertices[startVertexId].local_position;
            const Vec3& endPosition = editable_mesh_cache.vertices[endVertexId].local_position;
            const bool followsCanonicalDirection = startVertexId <= endVertexId;
            const float directedT = followsCanonicalDirection ? cutT : (1.0f - cutT);
            return startPosition + (endPosition - startPosition) * directedT;
        };

        const int a = selectedEdgeIndex;
        const int b = (a + 1) % 4;
        const int c = (a + 2) % 4;
        const int d = (a + 3) % 4;
        const Vec3 cutPointBC = computeDirectedCutPoint(vertexIds[b], vertexIds[c]);
        const Vec3 cutPointDA = computeDirectedCutPoint(vertexIds[d], vertexIds[a]);

        polygonsToBuild.push_back(PolygonBuildInput{
            { faceVertices[a], faceVertices[b], cutPointBC, cutPointDA },
            templateTriangle
        });
        polygonsToBuild.push_back(PolygonBuildInput{
            { cutPointDA, cutPointBC, faceVertices[c], faceVertices[d] },
            templateTriangle
        });

        LoopCutSelectionTarget target;
        target.center = (cutPointBC + cutPointDA) * 0.5f;
        const Vec3 direction = cutPointBC - cutPointDA;
        const float dirLenSq = direction.length_squared();
        target.direction =
            (std::isfinite(dirLenSq) && dirLenSq > 1e-10f)
                ? direction / std::sqrt(dirLenSq)
                : Vec3(1.0f, 0.0f, 0.0f);
        cutSelectionTargets.push_back(target);
        ++splitFaceCount;
    }

    if (splitFaceCount <= 0 || polygonsToBuild.empty()) {
        return false;
    }

    auto computePolygonNormalFromVertices = [](const std::vector<Vec3>& vertices) {
        Vec3 normal(0.0f, 0.0f, 0.0f);
        for (size_t i = 0; i < vertices.size(); ++i) {
            const Vec3& current = vertices[i];
            const Vec3& next = vertices[(i + 1) % vertices.size()];
            normal.x += (current.y - next.y) * (current.z + next.z);
            normal.y += (current.z - next.z) * (current.x + next.x);
            normal.z += (current.x - next.x) * (current.y + next.y);
        }
        const float lenSq = normal.length_squared();
        if (!std::isfinite(lenSq) || lenSq <= 1e-10f) {
            return Vec3(0.0f, 1.0f, 0.0f);
        }
        return normal / std::sqrt(lenSq);
    };

    std::vector<std::shared_ptr<Triangle>> cutMesh;
    cutMesh.reserve(beforeDisplayMesh.size() + polygonsToBuild.size() * 2);
    for (const auto& polygon : polygonsToBuild) {
        if (!polygon.template_triangle || polygon.vertices.size() < 3) {
            continue;
        }

        const Vec3 faceNormal = computePolygonNormalFromVertices(polygon.vertices);
        const std::vector<Vec2> faceUVs = buildPolygonPlanarUVs(polygon.vertices, faceNormal);
        for (size_t i = 1; i + 1 < polygon.vertices.size(); ++i) {
            auto newTriangle = cloneTriangleForEdit(polygon.template_triangle);
            if (!newTriangle) {
                continue;
            }

            newTriangle->setOriginalVertexPosition(0, polygon.vertices[0]);
            newTriangle->setOriginalVertexPosition(1, polygon.vertices[i]);
            newTriangle->setOriginalVertexPosition(2, polygon.vertices[i + 1]);
            newTriangle->setOriginalVertexNormal(0, faceNormal);
            newTriangle->setOriginalVertexNormal(1, faceNormal);
            newTriangle->setOriginalVertexNormal(2, faceNormal);
            newTriangle->set_normals(faceNormal, faceNormal, faceNormal);
            newTriangle->setUVCoordinates(faceUVs[0], faceUVs[i], faceUVs[i + 1]);
            newTriangle->markAABBDirty();
            newTriangle->updateTransformedVertices();
            cutMesh.push_back(newTriangle);
        }
    }

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
        ctx.selection.mesh_element_mode = MeshElementSelectMode::Edge;
    }

    addViewportMessage("Loop cut applied", 2.0f, ImVec4(0.34f, 0.84f, 1.0f, 1.0f));
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

    std::unordered_map<unsigned long long, int> edgeIdByKey;
    edgeIdByKey.reserve(editable_mesh_cache.polygon_edges.size() * 2 + 1);
    for (size_t edgeId = 0; edgeId < editable_mesh_cache.polygon_edges.size(); ++edgeId) {
        const auto& edge = editable_mesh_cache.polygon_edges[edgeId];
        edgeIdByKey[makeEditablePackedEdgeKey(edge.v0, edge.v1)] = static_cast<int>(edgeId);
    }

    std::vector<std::vector<int>> edgeToFaces(editable_mesh_cache.polygon_edges.size());
    for (size_t faceId = 0; faceId < editable_mesh_cache.polygon_faces.size(); ++faceId) {
        const auto& face = editable_mesh_cache.polygon_faces[faceId];
        for (size_t i = 0; i < face.vertex_ids.size(); ++i) {
            const int edgeId = findEditablePolygonEdgeId(
                editable_mesh_cache,
                edgeIdByKey,
                face.vertex_ids[i],
                face.vertex_ids[(i + 1) % face.vertex_ids.size()]);
            if (edgeId >= 0) {
                edgeToFaces[edgeId].push_back(static_cast<int>(faceId));
            }
        }
    }

    std::vector<bool> consumedFaces(editable_mesh_cache.polygon_faces.size(), false);
    std::vector<std::pair<std::vector<int>, std::shared_ptr<Triangle>>> outputPolygons;
    bool changed = false;

    for (const int edgeId : selectedEdgeIds) {
        if (edgeId < 0 || edgeId >= static_cast<int>(edgeToFaces.size())) {
            continue;
        }
        const auto& touchingFaces = edgeToFaces[edgeId];
        if (touchingFaces.size() != 2) {
            continue;
        }
        const int faceA = touchingFaces[0];
        const int faceB = touchingFaces[1];
        if (faceA < 0 || faceB < 0 ||
            faceA >= static_cast<int>(editable_mesh_cache.polygon_faces.size()) ||
            faceB >= static_cast<int>(editable_mesh_cache.polygon_faces.size()) ||
            consumedFaces[faceA] || consumedFaces[faceB]) {
            continue;
        }

        std::vector<int> mergedVertexIds;
        if (!buildMergedEditablePolygonBoundary(
                { editable_mesh_cache.polygon_faces[faceA].vertex_ids,
                  editable_mesh_cache.polygon_faces[faceB].vertex_ids },
                mergedVertexIds)) {
            continue;
        }

        const Vec3 mergedNormal = computeEditableFaceNormal(editable_mesh_cache, mergedVertexIds);
        const Vec3 referenceNormal =
            computeEditableReferenceNormal(editable_mesh_cache, editable_mesh_cache.polygon_faces[faceA].triangle_ids);
        if (mergedNormal.dot(referenceNormal) < 0.0f) {
            std::reverse(mergedVertexIds.begin(), mergedVertexIds.end());
        }

        outputPolygons.emplace_back(
            mergedVertexIds,
            resolveEditablePolygonTemplateTriangle(editable_mesh_cache, currentDisplayMesh, faceA));
        consumedFaces[faceA] = true;
        consumedFaces[faceB] = true;
        changed = true;
    }

    for (size_t faceId = 0; faceId < editable_mesh_cache.polygon_faces.size(); ++faceId) {
        if (consumedFaces[faceId]) {
            continue;
        }
        outputPolygons.emplace_back(
            editable_mesh_cache.polygon_faces[faceId].vertex_ids,
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
    if (ensureEditableMeshCache(ctx, objectName)) {
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

    std::vector<int> selectedVertexIds = editable_mesh_cache.selection.vertex_ids;
    std::sort(selectedVertexIds.begin(), selectedVertexIds.end());
    selectedVertexIds.erase(std::unique(selectedVertexIds.begin(), selectedVertexIds.end()), selectedVertexIds.end());
    if (selectedVertexIds.size() < 3 || selectedVertexIds.size() > 4) {
        return false;
    }

    const std::vector<int> orderedVertexIds = sortEditablePolygonVertices(editable_mesh_cache, selectedVertexIds);
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
            ctx.selection.mesh_element_mode = MeshElementSelectMode::Face;
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

    addFaceTriangle(0, 1, 2);
    if (faceVertices.size() == 4) {
        addFaceTriangle(0, 2, 3);
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
            ctx.selection.mesh_element_mode = MeshElementSelectMode::Face;
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
            ctx.selection.mesh_element_mode = MeshElementSelectMode::Vertex;
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
        ctx.selection.mesh_element_mode = MeshElementSelectMode::Vertex;
    }

    addViewportMessage("Welded selected vertices", 2.0f, ImVec4(0.34f, 0.84f, 1.0f, 1.0f));
    return true;
}

void SceneUI::queueMeshEditGpuSync(const std::string& objectName) {
    mesh_edit_gpu_sync_pending = true;
    mesh_edit_gpu_sync_object_name = objectName;
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
    const SculptBrushTool activeTool = io.KeyShift ? SculptBrushTool::Smooth : sculpt_mode_state.tool;

    auto finishStroke = [&]() {
        if (!sculpt_stroke_state.active) {
            return;
        }

        if (sculpt_stroke_state.changed && !sculpt_stroke_state.object_name.empty()) {
            refreshMeshEditLayerEditedState(ctx);

            std::vector<MeshEditTriangleState> endStates;
            captureMeshEditLayerState(ctx, sculpt_stroke_state.object_name, endStates);
            if (!sculpt_stroke_state.before_states.empty() && !endStates.empty()) {
                history.record(std::make_unique<MeshEditCommand>(
                    sculpt_stroke_state.object_name,
                    sculpt_stroke_state.before_states,
                    endStates));
            }

            if (ctx.backend_ptr) {
                queueMeshEditGpuSync(sculpt_stroke_state.object_name);
            } else {
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
    if (!mesh_edit_layer.active || mesh_edit_layer.object_name != objectName) {
        ensureMeshEditLayer(ctx, objectName);
    }
    if (!mesh_edit_layer.active || !mesh_edit_layer.enabled) {
        finishStroke();
        return;
    }

    auto meshEntryIt = mesh_cache.find(objectName);
    if (!sculpt_stroke_state.active &&
        meshEntryIt != mesh_cache.end() &&
        !meshEntryIt->second.empty() &&
        (!picking_vertices_synced || objects_needing_cpu_sync.count(objectName) > 0)) {
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
    if (meshEntryIt != mesh_cache.end() && !meshEntryIt->second.empty()) {
        // Sculpt targets a single editable object, so prefer a direct triangle scan
        // over the object's current CPU triangles. This avoids stale global BVH hits
        // immediately after scale/rotate transforms where triangle positions have
        // already been synced but the acceleration structure is still catching up.
        rawDidHit = raycastEditableObjectTriangles(meshEntryIt->second, ray, hit);
    }
    if (!rawDidHit) {
        rawDidHit = raycastViewportHit(ctx, io.MousePos, hit) &&
            hit.triangle &&
            hit.triangle->getNodeName() == objectName;
    }
    bool didHit = rawDidHit;

    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && didHit) {
        sculpt_stroke_state = SculptStrokeState{};
        sculpt_stroke_state.active = true;
        sculpt_stroke_state.object_name = objectName;
        sculpt_stroke_state.start_world_hit = hit.point;
        sculpt_stroke_state.last_world_hit = hit.point;
        sculpt_stroke_state.stroke_normal = hit.normal.length_squared() > 1e-8f ? hit.normal.normalize() : Vec3(0, 1, 0);
        captureMeshEditLayerState(ctx, objectName, sculpt_stroke_state.before_states);
        sculpt_stroke_state.start_local_positions.reserve(editable_mesh_cache.vertices.size());
        for (const auto& vertex : editable_mesh_cache.vertices) {
            sculpt_stroke_state.start_local_positions.push_back(vertex.local_position);
        }
        sculpt_stroke_state.layer_accum.assign(editable_mesh_cache.vertices.size(), 0.0f);
        sculpt_stroke_state.clay_layer_accum.assign(editable_mesh_cache.vertices.size(), 0.0f);
        sculpt_stroke_state.clay_strips_layer_accum.assign(editable_mesh_cache.vertices.size(), 0.0f);
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
    // Use brush.radius directly as world-space distance, matching the preview circle
    // which also uses it as world units (same convention as scatter_brush.brush_radius).
    // The old pixel→world conversions created a mismatch: the on-screen circle showed
    // one radius while the actual vertex selection used a completely different radius.
    const float radiusWorld = sanitizeFiniteFloat(sculpt_mode_state.brush.radius, 0.3f, 0.0001f, 1000.0f);
    const float brushStrength = sanitizeFiniteFloat(sculpt_mode_state.brush.strength, 1.0f, 0.0f, 100.0f);
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
        if (!intersectRayPlane(ray, startWorldHit, hitNormalWorld, planeHit)) {
            planeHit = hitPoint;
        }
    }
    // For Grab: use raw 3D delta so the mesh follows the mouse in all directions.
    // Stripping the normal component prevented movement when viewing at grazing angles.
    const Vec3 worldDragDelta = sanitizeVec3(planeHit - sanitizeVec3(sculpt_stroke_state.start_world_hit, planeHit), Vec3(0.0f, 0.0f, 0.0f));
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
    const Vec3 claySamplePoint = (clayLikeTool && sculpt_stroke_state.changed)
        ? sanitizeVec3(lastStrokeHit + (hitPoint - lastStrokeHit) * 0.5f, hitPoint)
        : hitPoint;
    const float localRadius = estimateEditableLocalBrushRadius(inverseTransform, radiusWorld);
    const Vec3 localHitPoint = sanitizeVec3(inverseTransform.transform_point(hitPoint), Vec3(0.0f, 0.0f, 0.0f));
    const Vec3 localClaySamplePoint = sanitizeVec3(inverseTransform.transform_point(claySamplePoint), localHitPoint);
    const Vec3 localGrabStartPoint = sanitizeVec3(
        inverseTransform.transform_point(sanitizeVec3(sculpt_stroke_state.start_world_hit, hitPoint)),
        localHitPoint);
    const std::vector<int> sculptCandidateVertexIds = collectEditableVertexCandidates(
        editable_mesh_cache,
        activeTool == SculptBrushTool::Grab ? localGrabStartPoint : localClaySamplePoint,
        localRadius);
    std::vector<int> strokeTouchedVertexIds = sculptCandidateVertexIds;
    strokeTouchedVertexIds.reserve(sculptCandidateVertexIds.size() * 4 + 32);

    std::vector<std::shared_ptr<Triangle>> touchedTriangles;
    touchedTriangles.reserve(sculptCandidateVertexIds.size() * 2 + 32);
    beginEditableTriangleTouchPass(editable_mesh_cache);

    bool changed = false;
    // Use persistent buffer to avoid per-frame allocation for large meshes
    sculpt_updated_local_positions.resize(editable_mesh_cache.vertices.size());
    for (const int vid : sculptCandidateVertexIds) {
        if (vid >= 0 && static_cast<size_t>(vid) < editable_mesh_cache.vertices.size()) {
            sculpt_updated_local_positions[vid] = editable_mesh_cache.vertices[vid].local_position;
        }
    }

    if (activeTool == SculptBrushTool::Grab &&
        sculpt_stroke_state.grab_weights.size() != editable_mesh_cache.vertices.size()) {
        sculpt_stroke_state.grab_weights.assign(editable_mesh_cache.vertices.size(), 0.0f);
        const Vec3 brushCenter = sanitizeVec3(sculpt_stroke_state.start_world_hit, hitPoint);
        forEachEditableCandidate(sculptCandidateVertexIds, [&](int vertexIdInt) {
            if (vertexIdInt < 0) {
                return;
            }
            const size_t vertexId = static_cast<size_t>(vertexIdInt);
            const Vec3 startWorldPos = sanitizeVec3(
                transform.transform_point(sculpt_stroke_state.start_local_positions[vertexId]),
                brushCenter);
            const Vec3 toVertex = startWorldPos - brushCenter;
            const float heightFromPlane = toVertex.dot(hitNormalWorld);
            const Vec3 planarOffset = toVertex - hitNormalWorld * heightFromPlane;
            const float planarDistance = planarOffset.length();
            if (!std::isfinite(heightFromPlane) || !std::isfinite(planarDistance) || planarDistance > radiusWorld) {
                return;
            }
            if (sculpt_mode_state.front_faces_only && heightFromPlane < -(radiusWorld * 0.12f)) {
                return;
            }
            // Single smooth falloff for grab so the whole brush area moves together.
            // Double-multiplying two falloff functions squared the weights, creating
            // a sharp central spike where only center vertices received meaningful weight.
            const float weight = applyFalloffCurve(
                1.0f - saturateFloat(planarDistance / radiusWorld),
                mesh_overlay_settings.proportional_falloff_type);
            sculpt_stroke_state.grab_weights[vertexId] = weight;
        });
    }

    const Vec3 planePoint = hitPoint;
    std::atomic<bool> basePassChanged{ false };
    forEachEditableCandidate(sculptCandidateVertexIds, [&](int vertexIdInt) {
        if (vertexIdInt < 0) {
            return;
        }
        const size_t vertexId = static_cast<size_t>(vertexIdInt);
        EditableVertex& vertex = editable_mesh_cache.vertices[vertexId];
        if (activeTool == SculptBrushTool::Grab) {
            const float weight = vertexId < sculpt_stroke_state.grab_weights.size()
                ? sculpt_stroke_state.grab_weights[vertexId]
                : 0.0f;
            if (weight <= 1e-5f || vertexId >= sculpt_stroke_state.start_local_positions.size()) {
                return;
            }
            // Remap strength: 1.0 = mesh tracks mouse 1:1; use 0.5 scaling so the
            // user-facing [0,10] range doesn't cause runaway deltas at high values.
            const float grabStrength = std::clamp(brushStrength * 0.5f, 0.0f, 5.0f);
            Vec3 worldDelta = worldDragDelta * weight * grabStrength;
            // Clamp world delta to radiusWorld to prevent vertices flying off on
            // large drags or when the camera is far from the surface.
            const float maxWorldDelta = radiusWorld * 3.0f;
            const float worldDeltaLen = worldDelta.length();
            if (!std::isfinite(worldDeltaLen)) {
                return;
            }
            if (worldDeltaLen > maxWorldDelta) {
                worldDelta = worldDelta * (maxWorldDelta / worldDeltaLen);
            }
            const Vec3 localDelta = sanitizeVec3(inverseTransform.transform_vector(worldDelta), Vec3(0.0f, 0.0f, 0.0f));
            const float localDeltaLenSq = localDelta.length_squared();
            if (!std::isfinite(localDeltaLenSq) || localDeltaLenSq <= 1e-14f) {
                return;
            }
            sculpt_updated_local_positions[vertexId] = sanitizeVec3(
                sculpt_stroke_state.start_local_positions[vertexId] + localDelta,
                sculpt_stroke_state.start_local_positions[vertexId]);
            basePassChanged.store(true, std::memory_order_relaxed);
            return;
        }

        const Vec3 worldPos = sanitizeVec3(transform.transform_point(vertex.local_position), Vec3(0.0f, 0.0f, 0.0f));
        const Vec3 toVertex = worldPos - planePoint;
        const float heightFromPlane = toVertex.dot(hitNormalWorld);
        const Vec3 planarOffset = toVertex - hitNormalWorld * heightFromPlane;
        const float planarDistance = planarOffset.length();
        if (!std::isfinite(heightFromPlane) || !std::isfinite(planarDistance) || planarDistance > radiusWorld) {
            return;
        }

        if (sculpt_mode_state.front_faces_only && heightFromPlane < -(radiusWorld * 0.12f)) {
            return;
        }

        float weight = computeTerrainLikeBrushWeight(
            planarDistance / radiusWorld,
            falloff);
        weight *= applyFalloffCurve(
            1.0f - saturateFloat(planarDistance / radiusWorld),
            mesh_overlay_settings.proportional_falloff_type);

        // Alpha mask modulates weight (same coordinate convention as mesh paint)
        const float nx_alpha = planarOffset.dot(tangentWorld) / radiusWorld;
        const float ny_alpha = planarOffset.dot(bitangentWorld) / radiusWorld;
        weight *= uiSampleBrushMask(sculpt_mode_state.brush, nx_alpha, ny_alpha);
        if (!std::isfinite(weight)) {
            return;
        }

        if (weight <= 1e-5f) {
            return;
        }

        Vec3 worldDelta(0.0f, 0.0f, 0.0f);
        switch (activeTool) {
        case SculptBrushTool::Inflate:
            worldDelta = hitNormalWorld * (radiusWorld * 0.22f * brushStrength * dt * weight * (1.0f + normalStrength) * directionSign);
            break;
        case SculptBrushTool::Draw: {
            // Push each vertex along its own world-space normal (per-vertex direction).
            // Unlike Inflate which uses the stroke hit normal, Draw respects surface curvature.
            Vec3 vertexNormal = hitNormalWorld;
            if (!vertex.refs.empty() && vertex.refs[0].triangle) {
                const auto& ref = vertex.refs[0];
                const Vec3 n = ref.triangle->vertices[ref.corner].normal;
                if (n.length_squared() > 1e-8f) {
                    vertexNormal = safeNormalizeVec3(n, hitNormalWorld);
                }
            }
            worldDelta = vertexNormal * (radiusWorld * 0.22f * brushStrength * dt * weight * (1.0f + normalStrength) * directionSign);
            break;
        }
        case SculptBrushTool::Layer: {
            if (vertexId >= sculpt_stroke_state.layer_accum.size()) {
                return;
            }
            float& layerAccum = sculpt_stroke_state.layer_accum[vertexId];
            const float previousAccum = layerAccum;
            const float targetLayer = radiusWorld * 0.22f * clayRadiusCompensation * weight * (0.9f + 0.35f * normalStrength);
            const float deposit = radiusWorld * 0.09f * clayRadiusCompensation * clayBrushStrength * dt * weight * directionSign;
            if (directionSign > 0.0f) {
                layerAccum = std::min(layerAccum + std::max(0.0f, deposit), targetLayer);
            } else {
                layerAccum = std::max(layerAccum + std::min(0.0f, deposit), -targetLayer);
            }
            worldDelta = hitNormalWorld * (layerAccum - previousAccum);
            break;
        }
        case SculptBrushTool::Clay: {
            Vec3 vertexNormal = hitNormalWorld;
            if (!vertex.refs.empty() && vertex.refs[0].triangle) {
                const auto& ref = vertex.refs[0];
                const Vec3 n = ref.triangle->vertices[ref.corner].normal;
                if (n.length_squared() > 1e-8f) {
                    vertexNormal = safeNormalizeVec3(n, hitNormalWorld);
                }
            }
            if (vertexId >= sculpt_stroke_state.clay_layer_accum.size()) {
                return;
            }
            const float deposit =
                radiusWorld * 0.075f * clayRadiusCompensation * clayBrushStrength * dt * weight * strokeAdvanceFactor *
                (0.8f + 0.5f * normalStrength) * directionSign;
            float& clayAccum = sculpt_stroke_state.clay_layer_accum[vertexId];
            clayAccum = std::clamp(clayAccum + deposit, -radiusWorld * 0.45f, radiusWorld * 0.45f);
            const float signedDistance = (worldPos - claySamplePoint).dot(hitNormalWorld);
            const float targetHeight =
                directionSign * radiusWorld * 0.16f * clayRadiusCompensation * weight * (0.8f + 0.5f * normalStrength);
            const float heightError = targetHeight - signedDistance;
            const float settle =
                std::clamp(heightError * (0.28f + 0.32f * weight), -radiusWorld * 0.05f, radiusWorld * 0.05f);
            const float layerDelta = settle * 0.7f + deposit * 0.55f + clayAccum * 0.16f;
            const float buildup = deposit * 0.18f + (std::max)(0.0f, heightError) * 0.16f;
            worldDelta = hitNormalWorld * layerDelta + vertexNormal * buildup;
            break;
        }
        case SculptBrushTool::ClayStrips: {
            if (vertexId >= sculpt_stroke_state.clay_strips_layer_accum.size()) {
                return;
            }
            const Vec3 clayPlanarOffset = worldPos - claySamplePoint -
                hitNormalWorld * (worldPos - claySamplePoint).dot(hitNormalWorld);
            const float stripCoord = clayPlanarOffset.dot(strokeBitangentWorld) / radiusWorld;
            const float tineWave = 0.5f + 0.5f * std::cos(stripCoord * 22.0f);
            const float tineProfile = std::pow((std::max)(0.0f, tineWave), 1.6f);
            const float rakePattern = 0.45f + 0.55f * tineProfile;
            const float deposit =
                radiusWorld * 0.11f * clayRadiusCompensation * clayBrushStrength * dt * weight * strokeAdvanceFactor * rakePattern *
                (0.95f + 0.45f * normalStrength) * directionSign;
            float& clayStripsAccum = sculpt_stroke_state.clay_strips_layer_accum[vertexId];
            clayStripsAccum = std::clamp(clayStripsAccum + deposit, -radiusWorld * 0.6f, radiusWorld * 0.6f);
            const float signedDistance = (worldPos - claySamplePoint).dot(hitNormalWorld);
            const float stripTargetHeight =
                directionSign * radiusWorld * 0.14f * clayRadiusCompensation * weight * rakePattern * (0.9f + 0.35f * normalStrength);
            const float stripHeightError = stripTargetHeight - signedDistance;
            const float settle =
                std::clamp(stripHeightError * (0.24f + 0.28f * weight), -radiusWorld * 0.05f, radiusWorld * 0.05f);
            const float layerDelta = settle * 0.62f + deposit * 0.62f + clayStripsAccum * 0.18f;
            const float tineDrag =
                radiusWorld * 0.05f * clayBrushStrength * dt * weight * strokeAdvanceFactor *
                (0.35f + 0.65f * tineProfile);
            worldDelta = hitNormalWorld * layerDelta + strokeTangentWorld * tineDrag;
            break;
        }
        case SculptBrushTool::Pinch: {
            // Pull vertices laterally toward the brush center (tightening effect).
            const Vec3 toCenter = planePoint - worldPos;
            const Vec3 lateral = toCenter - hitNormalWorld * toCenter.dot(hitNormalWorld);
            const float lateralLen = lateral.length();
            if (lateralLen > 1e-8f) {
                worldDelta = (lateral / lateralLen) * (radiusWorld * 0.3f * brushStrength * dt * weight);
            }
            break;
        }
        case SculptBrushTool::Flatten: {
            const float signedDistance = (worldPos - planePoint).dot(hitNormalWorld);
            worldDelta = hitNormalWorld * (-signedDistance * brushStrength * dt * 12.0f * weight);
            break;
        }
        case SculptBrushTool::Scrape: {
            const float signedDistance = (worldPos - planePoint).dot(hitNormalWorld);
            if (signedDistance > 0.0f) {
                worldDelta = hitNormalWorld * (-signedDistance * brushStrength * dt * 14.0f * weight);
            }
            break;
        }
        case SculptBrushTool::Crease: {
            const Vec3 toCenter = planePoint - worldPos;
            const Vec3 lateral = toCenter - hitNormalWorld * toCenter.dot(hitNormalWorld);
            const float lateralLen = lateral.length();
            const Vec3 pinchDelta = lateralLen > 1e-8f
                ? (lateral / lateralLen) * (radiusWorld * 0.38f * brushStrength * dt * weight)
                : Vec3(0.0f, 0.0f, 0.0f);
            const Vec3 cutDelta = hitNormalWorld * (-radiusWorld * 0.12f * brushStrength * dt * weight * (1.0f + normalStrength));
            worldDelta = pinchDelta + cutDelta;
            break;
        }
        case SculptBrushTool::Smooth: {
            const Vec3 smoothDeltaLocal = computeBoundarySafeSmoothDelta(
                editable_mesh_cache,
                vertexId,
                brushStrength,
                dt,
                weight,
                localRadius);
            const float smoothLenSq = smoothDeltaLocal.length_squared();
            if (std::isfinite(smoothLenSq) && smoothLenSq > 1e-14f) {
                sculpt_updated_local_positions[vertexId] = sanitizeVec3(vertex.local_position + smoothDeltaLocal, vertex.local_position);
                basePassChanged.store(true, std::memory_order_relaxed);
            }
            return;
        }
        case SculptBrushTool::Grab:
            return;
        }

        worldDelta = sanitizeVec3(worldDelta, Vec3(0.0f, 0.0f, 0.0f));

        const Vec3 localDelta = sanitizeVec3(inverseTransform.transform_vector(worldDelta), Vec3(0.0f, 0.0f, 0.0f));
        const float localDeltaLenSq = localDelta.length_squared();
        if (!std::isfinite(localDeltaLenSq) || localDeltaLenSq <= 1e-14f) {
            return;
        }
        sculpt_updated_local_positions[vertexId] = sanitizeVec3(vertex.local_position + localDelta, vertex.local_position);
        basePassChanged.store(true, std::memory_order_relaxed);
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
                const std::vector<int> mirrorGrabCandidateVertexIds = collectEditableVertexCandidates(
                    editable_mesh_cache,
                    localStartHit,
                    localRadius);
                // Initialize persistent buffer for mirror grab candidates
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
                    if (vertexId >= sculpt_stroke_state.start_local_positions.size()) {
                        return;
                    }
                    const Vec3 startWorldPos = sanitizeVec3(
                        transform.transform_point(sculpt_stroke_state.start_local_positions[vertexId]),
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
                        sculpt_stroke_state.start_local_positions[vertexId] + ld,
                        sculpt_stroke_state.start_local_positions[vertexId]);
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
                    return sanitizeVec3(localPrevHit + (mirLocalCenter - localPrevHit) * 0.5f, mirLocalCenter);
                }()
                : mirLocalCenter;
            const std::vector<int> mirrorCandidateVertexIds = collectEditableVertexCandidates(
                editable_mesh_cache,
                mirrorClaySampleCenter,
                localRadius);
            // Initialize persistent buffer for mirror candidates
            for (const int vid : mirrorCandidateVertexIds) {
                if (vid >= 0 && static_cast<size_t>(vid) < editable_mesh_cache.vertices.size()) {
                    sculpt_updated_local_positions[vid] = editable_mesh_cache.vertices[vid].local_position;
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
                const Vec3 worldPos = sanitizeVec3(transform.transform_point(vertex.local_position), mirWorldCenter);
                const Vec3 toVertex = worldPos - mirWorldCenter;
                const float h = toVertex.dot(mirWorldNormal);
                const Vec3 planarOffset = toVertex - mirWorldNormal * h;
                const float planarDist = planarOffset.length();
                if (!std::isfinite(h) || !std::isfinite(planarDist) || planarDist > radiusWorld) {
                    return;
                }
                if (sculpt_mode_state.front_faces_only && h < -(radiusWorld * 0.12f)) {
                    return;
                }
                float w = computeTerrainLikeBrushWeight(planarDist / radiusWorld, falloff);
                w *= applyFalloffCurve(1.0f - saturateFloat(planarDist / radiusWorld), mesh_overlay_settings.proportional_falloff_type);
                if (!std::isfinite(w) || w <= 1e-5f) {
                    return;
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
                    wd = vn * (radiusWorld * 0.22f * brushStrength * dt * w * (1.0f + normalStrength) * directionSign);
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
                        mirClaySampleCenter = sanitizeVec3(mirPrevCenter + (mirWorldCenter - mirPrevCenter) * 0.5f, mirWorldCenter);
                    }
                    const float sd = (worldPos - mirClaySampleCenter).dot(mirWorldNormal);
                    const float targetHeight =
                        directionSign * radiusWorld * 0.16f * clayRadiusCompensation * w * (0.8f + 0.5f * normalStrength);
                    if (vertexId >= sculpt_stroke_state.clay_layer_accum.size()) {
                        return;
                    }
                    const float deposit =
                        radiusWorld * 0.075f * clayRadiusCompensation * clayBrushStrength * dt * w * strokeAdvanceFactor *
                        (0.8f + 0.5f * normalStrength) * directionSign;
                    float& clayAccum = sculpt_stroke_state.clay_layer_accum[vertexId];
                    clayAccum = std::clamp(clayAccum + deposit, -radiusWorld * 0.45f, radiusWorld * 0.45f);
                    const float heightError = targetHeight - sd;
                    const float settle =
                        std::clamp(heightError * (0.28f + 0.32f * w), -radiusWorld * 0.05f, radiusWorld * 0.05f);
                    const float layerDelta = settle * 0.7f + deposit * 0.55f + clayAccum * 0.16f;
                    const float buildup = deposit * 0.18f + (std::max)(0.0f, heightError) * 0.16f;
                    wd = mirWorldNormal * layerDelta + vn * buildup;
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
                        mirClaySampleCenter = sanitizeVec3(mirPrevCenter + (mirWorldCenter - mirPrevCenter) * 0.5f, mirWorldCenter);
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
                    const float deposit =
                        radiusWorld * 0.11f * clayRadiusCompensation * clayBrushStrength * dt * w * strokeAdvanceFactor * rakePattern *
                        (0.95f + 0.45f * normalStrength) * directionSign;
                    float& clayStripsAccum = sculpt_stroke_state.clay_strips_layer_accum[vertexId];
                    clayStripsAccum = std::clamp(clayStripsAccum + deposit, -radiusWorld * 0.6f, radiusWorld * 0.6f);
                    const float stripTargetHeight =
                        directionSign * radiusWorld * 0.14f * clayRadiusCompensation * w * rakePattern * (0.9f + 0.35f * normalStrength);
                    const float stripHeightError = stripTargetHeight - sd;
                    const float settle =
                        std::clamp(stripHeightError * (0.24f + 0.28f * w), -radiusWorld * 0.05f, radiusWorld * 0.05f);
                    const float layerDelta = settle * 0.62f + deposit * 0.62f + clayStripsAccum * 0.18f;
                    const float tineDrag =
                        radiusWorld * 0.05f * clayBrushStrength * dt * w * strokeAdvanceFactor *
                        (0.35f + 0.65f * tineProfile);
                    wd = mirWorldNormal * layerDelta + mirStrokeTangent * tineDrag;
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
                        vertexId,
                        brushStrength,
                        dt,
                        w,
                        localRadius);
                    const float sldLenSq = sld.length_squared();
                    if (std::isfinite(sldLenSq) && sldLenSq > 1e-14f) {
                        sculpt_updated_local_positions[vertexId] = sanitizeVec3(vertex.local_position + sld, vertex.local_position);
                        mirrorPassChanged.store(true, std::memory_order_relaxed);
                    }
                    return;
                }
                default:
                    return;
                }

                wd = sanitizeVec3(wd, Vec3(0.0f, 0.0f, 0.0f));

                const Vec3 ld = sanitizeVec3(inverseTransform.transform_vector(wd), Vec3(0.0f, 0.0f, 0.0f));
                const float ldLenSq = ld.length_squared();
                if (!std::isfinite(ldLenSq) || ldLenSq <= 1e-14f) {
                    return;
                }
                sculpt_updated_local_positions[vertexId] = sanitizeVec3(vertex.local_position + ld, vertex.local_position);
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

    if (activeTool == SculptBrushTool::Clay || activeTool == SculptBrushTool::ClayStrips) {
        const float clayPolishStrength = activeTool == SculptBrushTool::Clay ? 0.16f : 0.11f;
        applyClayPolishPass(
            editable_mesh_cache,
            sculpt_updated_local_positions,
            strokeTouchedVertexIds,
            transform,
            inverseTransform,
            hitNormalWorld,
            brushStrength,
            dt,
            localRadius,
            clayPolishStrength);
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
        vertex.local_position = sculpt_updated_local_positions[vertexId];
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

    // Recalculate per-vertex smooth normals for all vertices touched by this sculpt dab.
    {
        const std::vector<size_t> affectedVertexIds =
            collectAffectedEditableVertexIds(editable_mesh_cache, touchedTriangles);
        recomputeEditableSmoothNormals(editable_mesh_cache, affectedVertexIds);
    }

    // Batch-apply transform to all touched triangles using precomputed matrices
    // instead of per-triangle getTransformMatrix() + inverse().transpose().
    {
        const Matrix4x4 finalTransform = transform;
        const Matrix4x4 normalTransform = inverseTransform.transpose();
        for (const auto& tri : touchedTriangles) {
            for (int i = 0; i < 3; ++i) {
                tri->vertices[i].position = finalTransform.transform_point(tri->vertices[i].original);
                tri->vertices[i].normal = normalTransform.transform_vector(tri->vertices[i].originalNormal).normalize();
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
    updateBBoxCache(objectName);
    objects_needing_cpu_sync.erase(objectName);

    if (!ctx.backend_ptr) {
        extern bool g_cpu_bvh_refit_pending;
        g_cpu_bvh_refit_pending = true;
    }

    if (ctx.backend_ptr) {
        queueMeshEditGpuSync(objectName);
    }

    ctx.renderer.resetCPUAccumulation();
    ctx.start_render = true;
    sculpt_stroke_state.changed = true;
    sculpt_stroke_state.last_world_hit = (activeTool == SculptBrushTool::Grab) ? planeHit : hit.point;
}

void SceneUI::processPendingMeshEditGpuSync(UIContext& ctx) {
    if (!mesh_edit_gpu_sync_pending || (!ctx.backend_ptr && !g_backend)) {
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

    // Solid/Matcap viewport should stay responsive and mostly rely on the raster patch path.
    // Keep CPU-side picking/CPU render in sync, but prefer the quieter refit path there.
    if (activeViewportInSolidMode) {
        g_cpu_bvh_refit_pending = true;
    } else {
        g_bvh_rebuild_pending = true;
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

        if (meshMode == MeshElementSelectMode::Vertex) {
            if ((selectedVertexCount == 3 || selectedVertexCount == 4) &&
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
                ImGui::MenuItem("Delete Face")) {
                contextMenuMutatedTopology = deleteSelectedMeshFaces(ctx) || contextMenuMutatedTopology;
                ImGui::CloseCurrentPopup();
            }
            if (selectedFaceCount == 0) {
                ImGui::TextDisabled("Select face");
            } else {
                ImGui::Separator();
                ImGui::TextDisabled("Extrude distance: %.3f", mesh_face_extrude_distance);
            }
        } else if (meshMode == MeshElementSelectMode::Edge) {
            const size_t selectedEdgeCount = editable_mesh_cache.selection.edge_ids.size();
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
        }

        ImGui::EndPopup();
    }

    if (contextMenuMutatedTopology) {
        return;
    }

    const bool needsRebuild =
        mesh_overlay_cache.object_name != objectName ||
        mesh_overlay_cache.source_triangle_count != triangleCount;

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
    const MeshElementSelectMode meshMode = ctx.selection.mesh_element_mode;
    const bool drawVertices =
        meshMode == MeshElementSelectMode::Vertex ||
        (meshMode == MeshElementSelectMode::Object && mesh_overlay_settings.show_vertices);
    const bool drawEdges =
        onCagePreview ||
        meshMode == MeshElementSelectMode::Object ||
        meshMode == MeshElementSelectMode::Vertex ||
        meshMode == MeshElementSelectMode::Edge;
    const bool drawFaces = (meshMode == MeshElementSelectMode::Face);

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

    if (drawFaces) {
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
            const Matrix4x4 transform = getEditableObjectTransform(editable_mesh_cache);
            for (size_t polygonFaceId = 0; polygonFaceId < editable_mesh_cache.polygon_faces.size(); polygonFaceId += faceStep) {
                const std::vector<int> vertexIds =
                    getEditablePolygonVertexIds(editable_mesh_cache, static_cast<int>(polygonFaceId));
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

                const float visibility =
                    (worldVertices.size() >= 3)
                        ? computeOverlayVisibility(*ctx.scene.camera, worldVertices[0], worldVertices[1], worldVertices[2])
                        : 1.0f;
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

    if (drawEdges) {
        const bool usePolygonEdges =
            (meshMode == MeshElementSelectMode::Face ||
             meshMode == MeshElementSelectMode::Edge ||
             meshMode == MeshElementSelectMode::Vertex) &&
            !editable_mesh_cache.object_name.empty() &&
            editable_mesh_cache.object_name == objectName &&
            !editable_mesh_cache.polygon_faces.empty();
        if (usePolygonEdges) {
            const Matrix4x4 transform = getEditableObjectTransform(editable_mesh_cache);
            for (const auto& polygonFace : editable_mesh_cache.polygon_faces) {
                if (polygonFace.vertex_ids.size() < 2) {
                    continue;
                }

                std::vector<Vec3> worldVertices;
                std::vector<ImVec2> screenVertices;
                worldVertices.reserve(polygonFace.vertex_ids.size());
                screenVertices.reserve(polygonFace.vertex_ids.size());
                bool projectionFailed = false;
                for (const int vertexId : polygonFace.vertex_ids) {
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
                if (projectionFailed || screenVertices.size() < 2) {
                    continue;
                }

                const float visibility =
                    (worldVertices.size() >= 3)
                        ? computeOverlayVisibility(*ctx.scene.camera, worldVertices[0], worldVertices[1], worldVertices[2])
                        : 1.0f;
                const ImU32 color = scaleColorAlpha(edgeColor, visibility);
                for (size_t edgeIndex = 0; edgeIndex < screenVertices.size(); ++edgeIndex) {
                    const ImVec2 a = screenVertices[edgeIndex];
                    const ImVec2 b = screenVertices[(edgeIndex + 1) % screenVertices.size()];
                    drawList->AddLine(a, b, color, mesh_overlay_settings.edge_thickness);
                }
            }
        } else {
            for (const auto& edge : mesh_overlay_cache.edges) {
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

    if (drawVertices) {
        const int maxVertexMarkers = (std::max)(1, mesh_overlay_settings.max_vertex_markers);
        if (hasEditableTopologyForObject && !editable_mesh_cache.vertices.empty()) {
            const Matrix4x4 transform = getEditableObjectTransform(editable_mesh_cache);
            const size_t vertexStep = editable_mesh_cache.vertices.size() > static_cast<size_t>(maxVertexMarkers)
                ? static_cast<size_t>(std::ceil(static_cast<float>(editable_mesh_cache.vertices.size()) / static_cast<float>(maxVertexMarkers)))
                : static_cast<size_t>(1);

            for (size_t i = 0; i < editable_mesh_cache.vertices.size(); i += vertexStep) {
                const auto& vertex = editable_mesh_cache.vertices[i];
                if (vertex.refs.empty() || !vertex.refs[0].triangle) {
                    continue;
                }

                const Vec3 p = transform.transform_point(vertex.local_position);
                const auto& tri = vertex.refs[0].triangle;
                const Vec3 p0 = transform.transform_point(tri->getOriginalVertexPosition(0));
                const Vec3 p1 = transform.transform_point(tri->getOriginalVertexPosition(1));
                const Vec3 p2 = transform.transform_point(tri->getOriginalVertexPosition(2));
                const float visibility = computeOverlayVisibility(*ctx.scene.camera, p0, p1, p2);
                ImVec2 screen;
                if (!projectPointToScreen(*ctx.scene.camera, displaySize, p, screen)) {
                    continue;
                }

                drawList->AddCircleFilled(screen, vertexRadius + 1.0f, scaleColorAlpha(vertexOutline, visibility), 8);
                drawList->AddCircleFilled(screen, vertexRadius, scaleColorAlpha(vertexColor, visibility), 8);
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
        const Matrix4x4 transform = getEditableObjectTransform(editable_mesh_cache);
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
                if (!isEditableVertexIdValid(editable_mesh_cache, vertexId)) {
                    projectionFailed = true;
                    break;
                }
                ImVec2 screenVertex;
                if (!projectPointToScreen(
                        *ctx.scene.camera,
                        displaySize,
                        transform.transform_point(editable_mesh_cache.vertices[vertexId].local_position),
                        screenVertex)) {
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
                if (projectPointToScreen(*ctx.scene.camera, displaySize, transform.transform_point(editable_mesh_cache.vertices[edge->v0].local_position), s0) &&
                    projectPointToScreen(*ctx.scene.camera, displaySize, transform.transform_point(editable_mesh_cache.vertices[edge->v1].local_position), s1)) {
                    drawList->AddLine(s0, s1, selectedEdgeColor, mesh_overlay_settings.edge_thickness + 1.35f);
                }
            }
        }

        for (const int vertexId : editable_mesh_cache.selection.vertex_ids) {
            if (vertexId < 0 || vertexId >= static_cast<int>(editable_mesh_cache.vertices.size())) {
                continue;
            }
            ImVec2 screen;
            if (projectPointToScreen(*ctx.scene.camera, displaySize,
                                     transform.transform_point(editable_mesh_cache.vertices[vertexId].local_position), screen)) {
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
                        if (!isEditableVertexIdValid(editable_mesh_cache, vertexId)) {
                            projectionFailed = true;
                            break;
                        }
                        ImVec2 screenVertex;
                        if (!projectPointToScreen(
                                *ctx.scene.camera,
                                displaySize,
                                transform.transform_point(editable_mesh_cache.vertices[vertexId].local_position),
                                screenVertex)) {
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
                    if (!projectPointToScreen(*ctx.scene.camera, displaySize, transform.transform_point(editable_mesh_cache.vertices[edge.v0].local_position), s0) ||
                        !projectPointToScreen(*ctx.scene.camera, displaySize, transform.transform_point(editable_mesh_cache.vertices[edge.v1].local_position), s1)) {
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
                    if (!projectPointToScreen(*ctx.scene.camera, displaySize,
                                              transform.transform_point(editable_mesh_cache.vertices[i].local_position), screen)) {
                        continue;
                    }

                    drawList->AddCircleFilled(screen, vertexRadius + weight * 2.0f, weightToColor(weight, 235), 10);
                }
            }
        }

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
    const float world_radius = (std::max)(0.0001f, brush.radius);

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
