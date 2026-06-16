#include "MeshModifiers.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <execution>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace MeshModifiers {

    namespace {

        struct QuantizedPositionKey {
            int x = 0;
            int y = 0;
            int z = 0;

            bool operator==(const QuantizedPositionKey& other) const {
                return x == other.x && y == other.y && z == other.z;
            }
        };

        struct QuantizedPositionKeyHasher {
            std::size_t operator()(const QuantizedPositionKey& key) const {
                std::size_t hx = std::hash<int>{}(key.x);
                std::size_t hy = std::hash<int>{}(key.y);
                std::size_t hz = std::hash<int>{}(key.z);
                return hx ^ (hy << 1) ^ (hz << 2);
            }
        };

        struct VertexData {
            Vec3 accumulatedPos{0.0f, 0.0f, 0.0f};
            int count = 0;
            Vec3 computedNormal{0.0f, 0.0f, 0.0f};
        };

        QuantizedPositionKey quantizePosition(const Vec3& v, float epsilon) {
            const float safeEpsilon = (std::max)(epsilon, 1e-6f);
            return QuantizedPositionKey{
                static_cast<int>(std::lround(v.x / safeEpsilon)),
                static_cast<int>(std::lround(v.y / safeEpsilon)),
                static_cast<int>(std::lround(v.z / safeEpsilon))
            };
        }

        template <typename Fn>
        void forEachMeshIndex(size_t count, Fn&& fn) {
            std::vector<size_t> indices(count);
            std::iota(indices.begin(), indices.end(), static_cast<size_t>(0));

            if (count >= 2048u) {
                std::for_each(
                    std::execution::par_unseq,
                    indices.begin(),
                    indices.end(),
                    std::forward<Fn>(fn));
                return;
            }

            std::for_each(indices.begin(), indices.end(), std::forward<Fn>(fn));
        }

    } // namespace

    std::vector<std::shared_ptr<Triangle>> SubdivideSubD(const std::vector<std::shared_ptr<Triangle>>& inputMesh, int levels) {
        std::vector<std::shared_ptr<Triangle>> currentMesh = inputMesh;

        for (int level = 0; level < levels; ++level) {
            std::vector<std::shared_ptr<Triangle>> nextMesh;
            nextMesh.resize(currentMesh.size() * 4);

            forEachMeshIndex(currentMesh.size(), [&](size_t triIndex) {
                const auto& tri = currentMesh[triIndex];
                if (tri) {
                    // Subdivision must operate in the mesh's local/original space.
                    // Using transformed vertex positions here bakes object scale/rotation
                    // into the generated topology and then reapplies the same transform
                    // through the shared transform handle, causing doubled overlay/sculpt
                    // extents after scaling.
                    Vec3 v0 = tri->getOriginalVertexPosition(0);
                    Vec3 v1 = tri->getOriginalVertexPosition(1);
                    Vec3 v2 = tri->getOriginalVertexPosition(2);

                    Vec3 n0 = tri->getOriginalVertexNormal(0);
                    Vec3 n1 = tri->getOriginalVertexNormal(1);
                    Vec3 n2 = tri->getOriginalVertexNormal(2);

                    auto [uv0, uv1, uv2] = tri->getUVCoordinates();
                    
                    uint16_t matID = tri->getMaterialID();
                    auto transform = tri->getTransformHandle();
                    std::string nodeName = tri->getNodeName();

                    // Midpoints
                    Vec3 m01 = (v0 + v1) * 0.5f;
                    Vec3 m12 = (v1 + v2) * 0.5f;
                    Vec3 m20 = (v2 + v0) * 0.5f;

                    // Normals (Slerp-like for exactness, but simple lerp + normalize is fine here)
                    Vec3 nm01 = (n0 + n1).normalize();
                    Vec3 nm12 = (n1 + n2).normalize();
                    Vec3 nm20 = (n2 + n0).normalize();

                    // UVs
                    Vec2 uvm01 = (uv0 + uv1) * 0.5f;
                    Vec2 uvm12 = (uv1 + uv2) * 0.5f;
                    Vec2 uvm20 = (uv2 + uv0) * 0.5f;

                    // Create 4 new triangles
                    std::shared_ptr<Triangle> t1 = std::make_shared<Triangle>(
                        v0, m01, m20, n0, nm01, nm20, uv0, uvm01, uvm20, matID);
                    std::shared_ptr<Triangle> t2 = std::make_shared<Triangle>(
                        m01, v1, m12, nm01, n1, nm12, uvm01, uv1, uvm12, matID);
                    std::shared_ptr<Triangle> t3 = std::make_shared<Triangle>(
                        m12, v2, m20, nm12, n2, nm20, uvm12, uv2, uvm20, matID);
                    std::shared_ptr<Triangle> t4 = std::make_shared<Triangle>(
                        m01, m12, m20, nm01, nm12, nm20, uvm01, uvm12, uvm20, matID);

                    t1->setTransformHandle(transform);
                    t2->setTransformHandle(transform);
                    t3->setTransformHandle(transform);
                    t4->setTransformHandle(transform);

                    t1->setNodeName(nodeName);
                    t2->setNodeName(nodeName);
                    t3->setNodeName(nodeName);
                    t4->setNodeName(nodeName);

                    t1->update_bounding_box();
                    t2->update_bounding_box();
                    t3->update_bounding_box();
                    t4->update_bounding_box();

                    const size_t baseIndex = triIndex * 4;
                    nextMesh[baseIndex + 0] = std::move(t1);
                    nextMesh[baseIndex + 1] = std::move(t2);
                    nextMesh[baseIndex + 2] = std::move(t3);
                    nextMesh[baseIndex + 3] = std::move(t4);
                }
            });
            currentMesh = std::move(nextMesh);
        }

        return currentMesh;
    }

    std::vector<std::shared_ptr<Triangle>> SmoothSubD(const std::vector<std::shared_ptr<Triangle>>& inputMesh, int levels, float smoothAngle) {
        // 1. Linearly subdivide first to get the topology density
        auto currentMesh = SubdivideSubD(inputMesh, levels);

        // We do a simple Laplacian smoothing iteration
        // To do this, we need to build an adjacency or vertex map
        // Due to lack of half-edge data structure, we'll use a position-based point cloud smoothing.
        //
        // The weld epsilon must stay BELOW the mesh's vertex spacing. A fixed 1e-4 (0.1mm)
        // worked at low subdivision but, on dense meshes (~700k tris) where the edge length
        // drops under 1e-4, DISTINCT — sometimes opposite-facing — vertices collided into the
        // same position bucket. Their face normals then cancelled in the averaging pass and
        // produced ~zero normals (black shading in sculpt, brush falling back to world-up).
        // Genuinely shared vertices here are bit-identical (the same midpoint / corner computed
        // by SubdivideSubD), so a fraction of the shortest edge welds them with huge margin
        // while never merging neighbours. One O(N) min-edge pass — negligible next to the
        // subdivision/Laplacian passes already running. Clamped so behaviour is unchanged at
        // low density (cap 1e-4) and float-safe at extreme density (floor 1e-6).
        float minEdgeSq = std::numeric_limits<float>::max();
        for (const auto& tri : currentMesh) {
            if (!tri) continue;
            const Vec3 v0 = tri->getOriginalVertexPosition(0);
            const Vec3 v1 = tri->getOriginalVertexPosition(1);
            const Vec3 v2 = tri->getOriginalVertexPosition(2);
            const float e0 = (v1 - v0).length_squared();
            const float e1 = (v2 - v1).length_squared();
            const float e2 = (v0 - v2).length_squared();
            if (e0 > 1e-20f) minEdgeSq = (std::min)(minEdgeSq, e0);
            if (e1 > 1e-20f) minEdgeSq = (std::min)(minEdgeSq, e1);
            if (e2 > 1e-20f) minEdgeSq = (std::min)(minEdgeSq, e2);
        }
        const float minEdge = (minEdgeSq < std::numeric_limits<float>::max())
            ? std::sqrt(minEdgeSq) : 1e-4f;
        const float epsilon = std::clamp(minEdge * 0.25f, 1e-6f, 1e-4f);

        // Smooth iterations
        int smoothIterations = 1; // 1 iter for basic smoothing
        for (int iter = 0; iter < smoothIterations; ++iter) {
            std::unordered_map<QuantizedPositionKey, VertexData, QuantizedPositionKeyHasher> vertexMap;
            vertexMap.reserve(currentMesh.size() * 2);

            // 1. Accumulate positions from neighboring triangles (Laplacian approximation)
            for (const auto& tri : currentMesh) {
                if (!tri) continue;
                for (int i = 0; i < 3; ++i) {
                    const Vec3 pos = tri->getOriginalVertexPosition(i);
                    const QuantizedPositionKey key = quantizePosition(pos, epsilon);
                    
                    // Add vertices of THIS triangle to the neighbors of the current vertex
                    for(int j=0; j<3; ++j) {
                        if (i == j) continue;
                        vertexMap[key].accumulatedPos =
                            vertexMap[key].accumulatedPos + tri->getOriginalVertexPosition(j);
                        vertexMap[key].count++;
                    }
                }
            }

            // 2. Apply smoothed positions
            forEachMeshIndex(currentMesh.size(), [&](size_t triIndex) {
                auto& tri = currentMesh[triIndex];
                if (tri) {
                    for (int i = 0; i < 3; ++i) {
                        const Vec3 pos = tri->getOriginalVertexPosition(i);
                        const QuantizedPositionKey key = quantizePosition(pos, epsilon);
                        
                        const auto found = vertexMap.find(key);
                        if (found != vertexMap.end() && found->second.count > 0) {
                            const Vec3 avgPos = found->second.accumulatedPos * (1.0f / (float)found->second.count);
                            // Interpolate between original and average based on smoothAngle (0.0 to 1.0)
                            const Vec3 newPos = pos * (1.0f - smoothAngle) + avgPos * smoothAngle;
                            tri->setOriginalVertexPosition(i, newPos);
                            tri->setVertexPosition(i, newPos);
                        }
                    }
                }
            });

            // 3. Recompute Normals for smoothed geometry
            std::unordered_map<QuantizedPositionKey, VertexData, QuantizedPositionKeyHasher> normalMap;
            normalMap.reserve(currentMesh.size() * 2);
            // Face normals to vertex normals
            for (auto& tri : currentMesh) {
                if (!tri) continue;
                const Vec3 v0 = tri->getOriginalVertexPosition(0);
                const Vec3 v1 = tri->getOriginalVertexPosition(1);
                const Vec3 v2 = tri->getOriginalVertexPosition(2);
                
                const Vec3 faceNormal = ((v1 - v0).cross(v2 - v0)).normalize();
                
                for(int i=0; i<3; ++i) {
                    const Vec3 pos = tri->getOriginalVertexPosition(i);
                    const QuantizedPositionKey key = quantizePosition(pos, epsilon);
                    normalMap[key].computedNormal = (normalMap[key].computedNormal + faceNormal);
                }
            }
            
            // Apply smoothed normals
            forEachMeshIndex(currentMesh.size(), [&](size_t triIndex) {
               auto& tri = currentMesh[triIndex];
               if (tri) {
                   for (int i = 0; i < 3; ++i) {
                       const Vec3 pos = tri->getOriginalVertexPosition(i);
                       const QuantizedPositionKey key = quantizePosition(pos, epsilon);
                       const auto found = normalMap.find(key);
                       if (found != normalMap.end()) {
                           const Vec3 smoothNormal = found->second.computedNormal.normalize();
                           tri->setOriginalVertexNormal(i, smoothNormal);
                           tri->setVertexNormal(i, smoothNormal);
                       }
                   }
                   tri->updateTransformedVertices();
                   tri->update_bounding_box();
               }
            });
        }

        return currentMesh;
    }

    std::vector<std::shared_ptr<Triangle>> CatmullClarkSubD(
            const std::vector<std::shared_ptr<Triangle>>& inputMesh,
            int levels,
            const EdgeCreaseFn& creaseLookup) {
        if (inputMesh.empty() || levels <= 0) {
            return inputMesh;
        }

        // All triangles of one object share the transform handle + node name.
        std::shared_ptr<Transform> sharedTransform;
        std::string nodeName;
        for (const auto& tri : inputMesh) {
            if (tri) {
                sharedTransform = tri->getTransformHandle();
                nodeName = tri->getNodeName();
                break;
            }
        }

        struct EdgeKey {
            int a = 0;
            int b = 0;
            bool operator==(const EdgeKey& o) const { return a == o.a && b == o.b; }
        };
        struct EdgeKeyHasher {
            std::size_t operator()(const EdgeKey& k) const {
                return (static_cast<std::size_t>(static_cast<uint32_t>(k.a)) << 32) ^
                       static_cast<std::size_t>(static_cast<uint32_t>(k.b));
            }
        };
        auto makeEdge = [](int u, int v) -> EdgeKey {
            return (u < v) ? EdgeKey{ u, v } : EdgeKey{ v, u };
        };

        // ---- Reconstruct an indexed polygon mesh from the triangle soup ----
        // Adaptive weld epsilon: stay below vertex spacing so distinct vertices never
        // merge (same reasoning as the SmoothSubD fix).
        float minEdgeSq = std::numeric_limits<float>::max();
        for (const auto& tri : inputMesh) {
            if (!tri) continue;
            const Vec3 a = tri->getOriginalVertexPosition(0);
            const Vec3 b = tri->getOriginalVertexPosition(1);
            const Vec3 c = tri->getOriginalVertexPosition(2);
            const float e0 = (b - a).length_squared();
            const float e1 = (c - b).length_squared();
            const float e2 = (a - c).length_squared();
            if (e0 > 1e-20f) minEdgeSq = (std::min)(minEdgeSq, e0);
            if (e1 > 1e-20f) minEdgeSq = (std::min)(minEdgeSq, e1);
            if (e2 > 1e-20f) minEdgeSq = (std::min)(minEdgeSq, e2);
        }
        const float weldEdge = (minEdgeSq < std::numeric_limits<float>::max())
            ? std::sqrt(minEdgeSq) : 1e-3f;
        const float weldEps = std::clamp(weldEdge * 0.25f, 1e-6f, 1e-3f);

        std::vector<Vec3> P;                    // welded positions (local space)
        std::vector<std::vector<int>> F;        // face -> vertex ids (CCW)
        std::vector<std::vector<Vec2>> FUV;     // face -> per-corner uvs
        std::vector<uint16_t> FM;               // face -> material id
        std::unordered_map<EdgeKey, float, EdgeKeyHasher> edgeCrease;

        std::unordered_map<QuantizedPositionKey, int, QuantizedPositionKeyHasher> vlookup;
        vlookup.reserve(inputMesh.size() * 2 + 1);
        auto weld = [&](const Vec3& p) -> int {
            const QuantizedPositionKey key = quantizePosition(p, weldEps);
            auto it = vlookup.find(key);
            if (it != vlookup.end()) return it->second;
            const int id = static_cast<int>(P.size());
            vlookup.emplace(key, id);
            P.push_back(p);
            return id;
        };

        // Build welded triangle faces.
        struct TriF { int v[3]; Vec2 uv[3]; uint16_t mat; };
        std::vector<TriF> tris;
        tris.reserve(inputMesh.size());
        for (const auto& tri : inputMesh) {
            if (!tri) continue;
            auto [uv0, uv1, uv2] = tri->getUVCoordinates();
            TriF tf;
            tf.v[0] = weld(tri->getOriginalVertexPosition(0));
            tf.v[1] = weld(tri->getOriginalVertexPosition(1));
            tf.v[2] = weld(tri->getOriginalVertexPosition(2));
            if (tf.v[0] == tf.v[1] || tf.v[1] == tf.v[2] || tf.v[2] == tf.v[0]) continue;
            tf.uv[0] = uv0; tf.uv[1] = uv1; tf.uv[2] = uv2;
            tf.mat = tri->getMaterialID();
            tris.push_back(tf);
        }
        if (tris.empty()) {
            return inputMesh;
        }

        // ---- Quad recovery -------------------------------------------------------
        // CC on a pure triangle soup is correct but NOT symmetric: a triangulated quad
        // (e.g. every face of a cube) keeps its diagonal as a real edge, raising vertex
        // valence unevenly → lopsided smoothing and slight dents. Merge triangle pairs
        // that share an edge back into a quad when the pair is near-coplanar, forms a
        // convex quad, and has matching UVs across the shared edge (so seams survive).
        // The shared edge that qualifies is the original quad DIAGONAL: a cube's real
        // edges join perpendicular faces (rejected by the coplanarity test), while the
        // diagonal joins two coplanar halves of the same face. Anything not safely
        // mergeable stays a triangle — CC handles a mixed quad/tri mesh fine.
        std::unordered_map<EdgeKey, std::vector<int>, EdgeKeyHasher> e2t;
        e2t.reserve(tris.size() * 3);
        for (int t = 0; t < static_cast<int>(tris.size()); ++t)
            for (int i = 0; i < 3; ++i)
                e2t[makeEdge(tris[t].v[i], tris[t].v[(i + 1) % 3])].push_back(t);

        auto triNormal = [&](const TriF& t) {
            return (P[t.v[1]] - P[t.v[0]]).cross(P[t.v[2]] - P[t.v[0]]);
        };

        std::vector<char> consumed(tris.size(), 0);
        for (int t = 0; t < static_cast<int>(tris.size()); ++t) {
            if (consumed[t]) continue;
            const Vec3 n1 = triNormal(tris[t]);
            const float l1 = n1.length();
            bool merged = false;

            for (int i = 0; i < 3 && !merged; ++i) {
                const int x = tris[t].v[i];
                const int y = tris[t].v[(i + 1) % 3];
                auto it = e2t.find(makeEdge(x, y));
                if (it == e2t.end()) continue;

                for (int t2 : it->second) {
                    if (t2 == t || consumed[t2]) continue;
                    const Vec3 n2 = triNormal(tris[t2]);
                    const float l2 = n2.length();
                    if (l1 < 1e-12f || l2 < 1e-12f) continue;
                    if ((n1 * (1.0f / l1)).dot(n2 * (1.0f / l2)) < 0.90f) continue;  // near-coplanar

                    // Corner layout of t: c1 = vertex off the shared edge.
                    int ia = -1;
                    for (int a = 0; a < 3; ++a)
                        if (tris[t].v[a] != x && tris[t].v[a] != y) { ia = a; break; }
                    if (ia < 0) continue;
                    const int ib = (ia + 1) % 3;
                    const int ic = (ia + 2) % 3;
                    // Third vertex of t2.
                    int c2 = -1;
                    for (int a = 0; a < 3; ++a)
                        if (tris[t2].v[a] != x && tris[t2].v[a] != y) { c2 = a; break; }
                    if (c2 < 0) continue;

                    // UV seam guard: shared-edge UVs must match between both triangles.
                    auto uvOf = [&](const TriF& tf, int vi) -> Vec2 {
                        for (int a = 0; a < 3; ++a) if (tf.v[a] == vi) return tf.uv[a];
                        return Vec2(0.0f, 0.0f);
                    };
                    if ((uvOf(tris[t], x) - uvOf(tris[t2], x)).lengthSquared() > 1e-8f) continue;
                    if ((uvOf(tris[t], y) - uvOf(tris[t2], y)).lengthSquared() > 1e-8f) continue;

                    // Quad CCW = [c1, e0, c2, e1] (insert t2's apex across the shared edge).
                    const int q0 = tris[t].v[ia];
                    const int q1 = tris[t].v[ib];
                    const int q2 = tris[t2].v[c2];
                    const int q3 = tris[t].v[ic];
                    const int quad[4] = { q0, q1, q2, q3 };

                    const Vec3 nf = n1 * (1.0f / l1);
                    bool convex = true;
                    for (int k = 0; k < 4; ++k) {
                        const Vec3 a = P[quad[k]];
                        const Vec3 b = P[quad[(k + 1) % 4]];
                        const Vec3 c = P[quad[(k + 2) % 4]];
                        if (((b - a).cross(c - b)).dot(nf) <= 0.0f) { convex = false; break; }
                    }
                    if (!convex) continue;

                    F.push_back({ q0, q1, q2, q3 });
                    FUV.push_back({ tris[t].uv[ia], tris[t].uv[ib], tris[t2].uv[c2], tris[t].uv[ic] });
                    FM.push_back(tris[t].mat);
                    consumed[t] = 1;
                    consumed[t2] = 1;
                    merged = true;
                    break;
                }
            }

            if (!merged) {
                F.push_back({ tris[t].v[0], tris[t].v[1], tris[t].v[2] });
                FUV.push_back({ tris[t].uv[0], tris[t].uv[1], tris[t].uv[2] });
                FM.push_back(tris[t].mat);
                consumed[t] = 1;
            }
        }
        if (F.empty()) {
            return inputMesh;
        }

        // Seed authored creases over the cage edges (boundaries auto-detect per level).
        if (creaseLookup) {
            std::unordered_set<EdgeKey, EdgeKeyHasher> seen;
            for (const auto& vs : F) {
                const int k = static_cast<int>(vs.size());
                for (int i = 0; i < k; ++i) {
                    const EdgeKey e = makeEdge(vs[i], vs[(i + 1) % k]);
                    if (!seen.insert(e).second) continue;
                    const float s = creaseLookup(P[e.a], P[e.b]);
                    if (s > 0.0f) edgeCrease[e] = std::clamp(s, 0.0f, 1.0f);
                }
            }
        }

        // ---- Catmull-Clark refinement ----
        for (int level = 0; level < levels; ++level) {
            const int V = static_cast<int>(P.size());
            const int Fn = static_cast<int>(F.size());

            std::vector<Vec3> facePt(Fn);
            for (int f = 0; f < Fn; ++f) {
                Vec3 c(0.0f, 0.0f, 0.0f);
                for (int vi : F[f]) c += P[vi];
                facePt[f] = c * (1.0f / static_cast<float>(F[f].size()));
            }

            struct EdgeInfo { int faces[2] = { -1, -1 }; int nf = 0; };
            std::unordered_map<EdgeKey, EdgeInfo, EdgeKeyHasher> edges;
            edges.reserve(Fn * 3);
            for (int f = 0; f < Fn; ++f) {
                const auto& vs = F[f];
                const int k = static_cast<int>(vs.size());
                for (int i = 0; i < k; ++i) {
                    EdgeInfo& info = edges[makeEdge(vs[i], vs[(i + 1) % k])];
                    if (info.nf < 2) info.faces[info.nf] = f;
                    ++info.nf;
                }
            }

            std::vector<Vec3> vFaceSum(V, Vec3(0.0f, 0.0f, 0.0f));
            std::vector<int> vFaceCnt(V, 0);
            for (int f = 0; f < Fn; ++f)
                for (int vi : F[f]) { vFaceSum[vi] += facePt[f]; ++vFaceCnt[vi]; }

            std::vector<Vec3> vEdgeMidSum(V, Vec3(0.0f, 0.0f, 0.0f));
            std::vector<int> vEdgeCnt(V, 0);
            std::vector<int> vSharpCnt(V, 0);
            std::vector<std::array<int, 2>> vSharpNbr(V, std::array<int, 2>{ -1, -1 });

            std::unordered_map<EdgeKey, Vec3, EdgeKeyHasher> edgePt;
            edgePt.reserve(edges.size());
            for (const auto& kv : edges) {
                const EdgeKey& e = kv.first;
                const EdgeInfo& info = kv.second;
                const Vec3 pa = P[e.a];
                const Vec3 pb = P[e.b];
                const Vec3 mid = (pa + pb) * 0.5f;

                float sigma = 0.0f;
                if (info.nf < 2) {
                    sigma = 1.0f;                            // boundary = infinitely sharp
                } else {
                    auto it = edgeCrease.find(e);
                    if (it != edgeCrease.end()) sigma = std::clamp(it->second, 0.0f, 1.0f);
                }

                Vec3 smoothEP = mid;
                if (info.nf >= 2)
                    smoothEP = (pa + pb + facePt[info.faces[0]] + facePt[info.faces[1]]) * 0.25f;
                edgePt[e] = smoothEP * (1.0f - sigma) + mid * sigma;

                vEdgeMidSum[e.a] += mid; ++vEdgeCnt[e.a];
                vEdgeMidSum[e.b] += mid; ++vEdgeCnt[e.b];

                if (sigma > 0.0f) {
                    if (vSharpCnt[e.a] < 2) vSharpNbr[e.a][vSharpCnt[e.a]] = e.b;
                    ++vSharpCnt[e.a];
                    if (vSharpCnt[e.b] < 2) vSharpNbr[e.b][vSharpCnt[e.b]] = e.a;
                    ++vSharpCnt[e.b];
                }
            }

            std::vector<Vec3> newVertPt(V);
            for (int v = 0; v < V; ++v) {
                if (vFaceCnt[v] == 0 || vSharpCnt[v] >= 3) {
                    newVertPt[v] = P[v];                          // corner / isolated → fixed
                } else if (vSharpCnt[v] == 2) {
                    const Vec3 n1 = P[vSharpNbr[v][0]];
                    const Vec3 n2 = P[vSharpNbr[v][1]];
                    newVertPt[v] = (P[v] * 6.0f + n1 + n2) * (1.0f / 8.0f);   // crease/boundary rule
                } else {
                    const float n = static_cast<float>(vFaceCnt[v]);
                    const Vec3 Favg = vFaceSum[v] * (1.0f / n);
                    const Vec3 Ravg = (vEdgeCnt[v] > 0)
                        ? vEdgeMidSum[v] * (1.0f / static_cast<float>(vEdgeCnt[v]))
                        : P[v];
                    newVertPt[v] = (Favg + Ravg * 2.0f + P[v] * (n - 3.0f)) * (1.0f / n);
                }
            }

            // Next-level position layout: [vertex pts | edge pts | face pts]
            std::vector<Vec3> NP;
            NP.reserve(V + edges.size() + Fn);
            for (int v = 0; v < V; ++v) NP.push_back(newVertPt[v]);
            std::unordered_map<EdgeKey, int, EdgeKeyHasher> edgeIndex;
            edgeIndex.reserve(edges.size());
            for (const auto& kv : edges) {
                edgeIndex[kv.first] = static_cast<int>(NP.size());
                NP.push_back(edgePt[kv.first]);
            }
            const int faceBase = static_cast<int>(NP.size());
            for (int f = 0; f < Fn; ++f) NP.push_back(facePt[f]);

            std::vector<std::vector<int>> NF;
            std::vector<std::vector<Vec2>> NFUV;
            std::vector<uint16_t> NFM;
            NF.reserve(Fn * 4);
            NFUV.reserve(Fn * 4);
            NFM.reserve(Fn * 4);
            for (int f = 0; f < Fn; ++f) {
                const auto& vs = F[f];
                const auto& uvs = FUV[f];
                const int k = static_cast<int>(vs.size());
                Vec2 cUV(0.0f, 0.0f);
                for (const auto& u : uvs) cUV = cUV + u;
                cUV = cUV * (1.0f / static_cast<float>(k));
                const int centerIdx = faceBase + f;
                // Each face of k sides -> k quads: (vertexPt, edgePt(next), facePt, edgePt(prev)).
                for (int i = 0; i < k; ++i) {
                    const int vCur = vs[i];
                    const int vNext = vs[(i + 1) % k];
                    const int vPrev = vs[(i - 1 + k) % k];
                    const int eNextIdx = edgeIndex[makeEdge(vCur, vNext)];
                    const int ePrevIdx = edgeIndex[makeEdge(vPrev, vCur)];
                    NF.push_back({ vCur, eNextIdx, centerIdx, ePrevIdx });
                    // UVs averaged per-face (seam-safe: uses only this face's corners).
                    const Vec2 uvNext = (uvs[i] + uvs[(i + 1) % k]) * 0.5f;
                    const Vec2 uvPrev = (uvs[(i - 1 + k) % k] + uvs[i]) * 0.5f;
                    NFUV.push_back({ uvs[i], uvNext, cUV, uvPrev });
                    NFM.push_back(FM[f]);
                }
            }

            // Propagate authored creases to child edges (boundaries re-detect next level).
            std::unordered_map<EdgeKey, float, EdgeKeyHasher> newCrease;
            for (const auto& kv : edges) {
                if (kv.second.nf < 2) continue;
                auto it = edgeCrease.find(kv.first);
                if (it == edgeCrease.end() || it->second <= 0.0f) continue;
                const int ep = edgeIndex[kv.first];
                newCrease[makeEdge(kv.first.a, ep)] = it->second;
                newCrease[makeEdge(ep, kv.first.b)] = it->second;
            }

            P.swap(NP);
            F.swap(NF);
            FUV.swap(NFUV);
            FM.swap(NFM);
            edgeCrease.swap(newCrease);
        }

        // ---- Smooth vertex normals from final topology (Newell, area-weighted) ----
        std::vector<Vec3> VN(P.size(), Vec3(0.0f, 0.0f, 0.0f));
        for (const auto& vs : F) {
            const int k = static_cast<int>(vs.size());
            Vec3 nrm(0.0f, 0.0f, 0.0f);
            for (int i = 0; i < k; ++i) {
                const Vec3& cur = P[vs[i]];
                const Vec3& nx = P[vs[(i + 1) % k]];
                nrm.x += (cur.y - nx.y) * (cur.z + nx.z);
                nrm.y += (cur.z - nx.z) * (cur.x + nx.x);
                nrm.z += (cur.x - nx.x) * (cur.y + nx.y);
            }
            for (int vi : vs) VN[vi] += nrm;
        }
        for (auto& n : VN) {
            const float L = n.length();
            n = (L > 1e-12f) ? n * (1.0f / L) : Vec3(0.0f, 1.0f, 0.0f);
        }

        // ---- Triangulate (post-CC faces are quads) -> Triangle objects ----
        std::vector<std::shared_ptr<Triangle>> out;
        out.reserve(F.size() * 2);
        auto emitTri = [&](int a, int b, int c,
                           const Vec2& ua, const Vec2& ub, const Vec2& uc,
                           uint16_t mat) {
            auto t = std::make_shared<Triangle>(
                P[a], P[b], P[c], VN[a], VN[b], VN[c], ua, ub, uc, mat);
            t->setTransformHandle(sharedTransform);
            t->setNodeName(nodeName);
            t->updateTransformedVertices();
            t->update_bounding_box();
            out.push_back(std::move(t));
        };
        for (size_t f = 0; f < F.size(); ++f) {
            const auto& vs = F[f];
            const auto& uvs = FUV[f];
            const uint16_t mat = FM[f];
            if (vs.size() == 4) {
                emitTri(vs[0], vs[1], vs[2], uvs[0], uvs[1], uvs[2], mat);
                emitTri(vs[0], vs[2], vs[3], uvs[0], uvs[2], uvs[3], mat);
            } else if (vs.size() == 3) {
                emitTri(vs[0], vs[1], vs[2], uvs[0], uvs[1], uvs[2], mat);
            } else {
                for (size_t i = 1; i + 1 < vs.size(); ++i)
                    emitTri(vs[0], vs[static_cast<int>(i)], vs[static_cast<int>(i) + 1],
                            uvs[0], uvs[static_cast<int>(i)], uvs[static_cast<int>(i) + 1], mat);
            }
        }
        return out;
    }

    void ModifierData::serialize(nlohmann::json& j) const {
        j["name"] = name;
        j["type"] = static_cast<int>(type);
        j["enabled"] = enabled;
        j["levels"] = levels;
        j["smoothAngle"] = smoothAngle;
    }

    void ModifierData::deserialize(const nlohmann::json& j) {
        if (j.contains("name")) name = j["name"].get<std::string>();
        if (j.contains("type")) type = static_cast<ModifierType>(j["type"].get<int>());
        if (j.contains("enabled")) enabled = j["enabled"].get<bool>();
        if (j.contains("levels")) levels = j["levels"].get<int>();
        if (j.contains("smoothAngle")) smoothAngle = j["smoothAngle"].get<float>();
    }

    std::array<int, 6> ModifierStack::makeCreaseKey(const Vec3& a, const Vec3& b) {
        constexpr float s = 100000.0f;   // 0.01mm grid; matches cage vertex precision
        std::array<int, 3> qa{
            static_cast<int>(std::lround(a.x * s)),
            static_cast<int>(std::lround(a.y * s)),
            static_cast<int>(std::lround(a.z * s)) };
        std::array<int, 3> qb{
            static_cast<int>(std::lround(b.x * s)),
            static_cast<int>(std::lround(b.y * s)),
            static_cast<int>(std::lround(b.z * s)) };
        // Order endpoints deterministically so (a,b) and (b,a) collapse to one key.
        if (std::lexicographical_compare(qb.begin(), qb.end(), qa.begin(), qa.end())) {
            std::swap(qa, qb);
        }
        return { qa[0], qa[1], qa[2], qb[0], qb[1], qb[2] };
    }

    void ModifierStack::setEdgeCrease(const Vec3& a, const Vec3& b, float weight) {
        const auto key = makeCreaseKey(a, b);
        if (weight <= 0.0f) {
            edgeCreases.erase(key);
        } else {
            edgeCreases[key] = (std::min)(weight, 1.0f);
        }
    }

    float ModifierStack::getEdgeCrease(const Vec3& a, const Vec3& b) const {
        const auto it = edgeCreases.find(makeCreaseKey(a, b));
        return (it != edgeCreases.end()) ? it->second : 0.0f;
    }

    std::vector<std::shared_ptr<Triangle>> ModifierStack::evaluate(const std::vector<std::shared_ptr<Triangle>>& baseMesh) const {
        std::vector<std::shared_ptr<Triangle>> currentMesh = baseMesh;

        // Apply modifiers sequentially
        for (const auto& mod : modifiers) {
            if (!mod.enabled) continue;

            if (mod.type == ModifierType::FlatSubdivision) {
                currentMesh = SubdivideSubD(currentMesh, mod.levels);
            } else if (mod.type == ModifierType::SmoothSubdivision) {
                // LIVE "Smooth Subdivision" uses the lightweight linear+Laplacian path so
                // that interactive cage editing (gizmo move/scale of vertices/edges/faces)
                // stays stable. True Catmull-Clark is offered as a destructive bake instead
                // (see SceneUI bake button → CatmullClarkSubD), which avoids the per-frame
                // re-evaluation fragility of editing while a CC preview is live.
                currentMesh = SmoothSubD(currentMesh, mod.levels, mod.smoothAngle);
            }
        }

        return currentMesh;
    }

    void ModifierStack::serialize(nlohmann::json& j) const {
        j["modifiers"] = nlohmann::json::array();
        for (const auto& mod : modifiers) {
            nlohmann::json modJson;
            mod.serialize(modJson);
            j["modifiers"].push_back(modJson);
        }
        j["edgeCreases"] = nlohmann::json::array();
        for (const auto& [key, weight] : edgeCreases) {
            nlohmann::json e;
            e["k"] = { key[0], key[1], key[2], key[3], key[4], key[5] };
            e["w"] = weight;
            j["edgeCreases"].push_back(e);
        }
    }

    void ModifierStack::deserialize(const nlohmann::json& j) {
        modifiers.clear();
        if (j.contains("modifiers") && j["modifiers"].is_array()) {
            for (const auto& modJson : j["modifiers"]) {
                ModifierData mod;
                mod.deserialize(modJson);
                modifiers.push_back(mod);
            }
        }
        edgeCreases.clear();
        if (j.contains("edgeCreases") && j["edgeCreases"].is_array()) {
            for (const auto& e : j["edgeCreases"]) {
                if (!e.contains("k") || !e["k"].is_array() || e["k"].size() != 6) continue;
                std::array<int, 6> key{};
                for (int i = 0; i < 6; ++i) key[i] = e["k"][i].get<int>();
                edgeCreases[key] = e.contains("w") ? e["w"].get<float>() : 1.0f;
            }
        }
    }

}
