#include "MeshModifiers.h"
#include "TriangleProxyConverter.h"
#include "GeometryNodesV2.h"      // Faz 8a Geo-DAG node wrappers around this file's algorithms
#include "NodeSystem/NodeRegistry.h"
#include "SimulationCompute.h"   // shared Vulkan compute backend (GPU subdivision)
#include "globals.h"             // g_gpu_subdivide_enabled, kGpuSubdivideMinTris, SCENE_LOG_INFO
#include <algorithm>
#include <array>
#include <chrono>                // [CCPERF] stage timing
#include <cmath>
#include <cstdint>
#include <cstring>
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

        QuantizedPositionKey quantizePosition(const Vec3& v, float epsilon) {
            const float safeEpsilon = (std::max)(epsilon, 1e-6f);
            return QuantizedPositionKey{
                static_cast<int>(std::lround(v.x / safeEpsilon)),
                static_cast<int>(std::lround(v.y / safeEpsilon)),
                static_cast<int>(std::lround(v.z / safeEpsilon))
            };
        }

    } // namespace

    namespace {

        // Lightweight, heap-free vertex/triangle used for the intermediate subdivision
        // levels. The old path rebuilt the whole std::vector<std::shared_ptr<Triangle>>
        // at EVERY level (millions of make_shared + atomic refcount + per-triangle string
        // copies, with the 4x / 16x intermediate triangles allocated only to be thrown
        // away). We now do all the midpoint math on this contiguous POD soup and pay the
        // Triangle allocation exactly ONCE, at the final density. Same output topology.
        struct PodVert {
            Vec3 pos;
            Vec3 nrm;
            Vec2 uv;
        };
        struct PodTri {
            PodVert v[3];
            uint16_t mat = 0;
        };

        // All triangles of one object share the transform handle + node name; capture
        // them once instead of copying per generated triangle.
        struct PodMeshMeta {
            std::shared_ptr<Transform> transform;
            std::string nodeName;
        };

        std::vector<PodTri> trianglesToPod(
                const std::vector<std::shared_ptr<Triangle>>& mesh, PodMeshMeta& meta) {
            std::vector<PodTri> pod;
            pod.reserve(mesh.size());
            bool metaCaptured = false;
            for (const auto& tri : mesh) {
                // Subdivision must operate in the mesh's local/original space (using the
                // transformed positions would bake object scale/rotation into the topology
                // and then re-apply the same transform through the shared handle).
                if (!tri) continue;
                if (!metaCaptured) {
                    meta.transform = tri->getTransformHandle();
                    meta.nodeName = tri->getNodeName();
                    metaCaptured = true;
                }
                auto [uv0, uv1, uv2] = tri->getUVCoordinates();
                PodTri t;
                t.v[0] = { tri->getOriginalVertexPosition(0), tri->getOriginalVertexNormal(0), uv0 };
                t.v[1] = { tri->getOriginalVertexPosition(1), tri->getOriginalVertexNormal(1), uv1 };
                t.v[2] = { tri->getOriginalVertexPosition(2), tri->getOriginalVertexNormal(2), uv2 };
                t.mat = tri->getMaterialID();
                pod.push_back(t);
            }
            return pod;
        }

        // One level of linear (1->4 midpoint) subdivision on the POD soup. Per-triangle
        // independent midpoints — bit-identical to the old shared_ptr<Triangle> path.
        void subdividePodOnce(std::vector<PodTri>& mesh) {
            const int n = static_cast<int>(mesh.size());
            std::vector<PodTri> next(static_cast<size_t>(n) * 4);
            #pragma omp parallel for num_threads(get_omp_threads_limit()) schedule(static) if(n >= 2048)
            for (int ti = 0; ti < n; ++ti) {
                const PodTri& tri = mesh[static_cast<size_t>(ti)];
                const Vec3& v0 = tri.v[0].pos; const Vec3& v1 = tri.v[1].pos; const Vec3& v2 = tri.v[2].pos;
                const Vec3& n0 = tri.v[0].nrm; const Vec3& n1 = tri.v[1].nrm; const Vec3& n2 = tri.v[2].nrm;
                const Vec2& uv0 = tri.v[0].uv; const Vec2& uv1 = tri.v[1].uv; const Vec2& uv2 = tri.v[2].uv;

                const Vec3 m01 = (v0 + v1) * 0.5f;
                const Vec3 m12 = (v1 + v2) * 0.5f;
                const Vec3 m20 = (v2 + v0) * 0.5f;
                const Vec3 nm01 = (n0 + n1).normalize();
                const Vec3 nm12 = (n1 + n2).normalize();
                const Vec3 nm20 = (n2 + n0).normalize();
                const Vec2 uvm01 = (uv0 + uv1) * 0.5f;
                const Vec2 uvm12 = (uv1 + uv2) * 0.5f;
                const Vec2 uvm20 = (uv2 + uv0) * 0.5f;

                const size_t base = static_cast<size_t>(ti) * 4;
                next[base + 0] = PodTri{ { { v0, n0, uv0 }, { m01, nm01, uvm01 }, { m20, nm20, uvm20 } }, tri.mat };
                next[base + 1] = PodTri{ { { m01, nm01, uvm01 }, { v1, n1, uv1 }, { m12, nm12, uvm12 } }, tri.mat };
                next[base + 2] = PodTri{ { { m12, nm12, uvm12 }, { v2, n2, uv2 }, { m20, nm20, uvm20 } }, tri.mat };
                next[base + 3] = PodTri{ { { m01, nm01, uvm01 }, { m12, nm12, uvm12 }, { m20, nm20, uvm20 } }, tri.mat };
            }
            mesh.swap(next);
        }

        // Materialize the POD soup into Triangle objects — the ONLY place we pay the
        // shared_ptr<Triangle> allocation, instead of at every subdivision level.
        std::vector<std::shared_ptr<Triangle>> podToTriangles(
                const std::vector<PodTri>& pod, const PodMeshMeta& meta, bool transformVertices) {
            const int n = static_cast<int>(pod.size());
            std::vector<std::shared_ptr<Triangle>> out(static_cast<size_t>(n));

            auto sharedGeom = std::make_shared<OriginalMeshGeometry>();
            sharedGeom->positions.resize(static_cast<size_t>(n) * 3);
            sharedGeom->normals.resize(static_cast<size_t>(n) * 3);

            #pragma omp parallel for num_threads(get_omp_threads_limit()) schedule(static) if(n >= 2048)
            for (int i = 0; i < n; ++i) {
                const PodTri& t = pod[static_cast<size_t>(i)];

                const size_t baseIdx = static_cast<size_t>(i) * 3;
                sharedGeom->positions[baseIdx + 0] = t.v[0].pos;
                sharedGeom->positions[baseIdx + 1] = t.v[1].pos;
                sharedGeom->positions[baseIdx + 2] = t.v[2].pos;
                sharedGeom->normals[baseIdx + 0] = t.v[0].nrm;
                sharedGeom->normals[baseIdx + 1] = t.v[1].nrm;
                sharedGeom->normals[baseIdx + 2] = t.v[2].nrm;

                auto tri = std::make_shared<Triangle>(
                    t.v[0].pos, t.v[1].pos, t.v[2].pos,
                    t.v[0].nrm, t.v[1].nrm, t.v[2].nrm,
                    t.v[0].uv,  t.v[1].uv,  t.v[2].uv, t.mat);
                
                tri->setTransformHandle(meta.transform);
                tri->setNodeName(meta.nodeName);
                
                tri->setAssimpVertexIndices(
                    static_cast<unsigned int>(baseIdx + 0),
                    static_cast<unsigned int>(baseIdx + 1),
                    static_cast<unsigned int>(baseIdx + 2)
                );
                tri->setOriginalGeometry(sharedGeom);

                // Flat path matches the old SubdivideSubD (bbox only); the Smooth path
                // mutated positions post-construction so it also refreshed transformed verts.
                if (transformVertices) tri->updateTransformedVertices();
                tri->update_bounding_box();
                out[static_cast<size_t>(i)] = std::move(tri);
            }
            return out;
        }

        // GPU layout mirroring subdivide_linear.comp (std430 scalar floats, 112 bytes).
        struct GpuVert { float px, py, pz, nx, ny, nz, u, v; };
        struct GpuTri  { GpuVert a, b, c; uint32_t mat; uint32_t p0, p1, p2; };
        static_assert(sizeof(GpuVert) == 32, "GpuVert must match the GLSL std430 layout");
        static_assert(sizeof(GpuTri)  == 112, "GpuTri must match the GLSL std430 layout");

        GpuVert packVert(const PodVert& v) {
            return GpuVert{ v.pos.x, v.pos.y, v.pos.z,
                            v.nrm.x, v.nrm.y, v.nrm.z,
                            v.uv.x,  v.uv.y };
        }
        PodVert unpackVert(const GpuVert& g) {
            return PodVert{ Vec3(g.px, g.py, g.pz),
                            Vec3(g.nx, g.ny, g.nz),
                            Vec2(g.u, g.v) };
        }

        // GPU port of the per-level subdivide loop (linear 1->4). Drives the
        // subdivide_linear kernel once per level via ping-pong buffers, then downloads
        // the final soup back into `mesh`. Returns false (leaving `mesh` untouched) when
        // the compute backend is unavailable or any GPU step fails — caller then runs the
        // CPU loop. Output is numerically equivalent to subdividePodOnce (GPU float
        // rounding may differ in the last bits).
        bool subdivideLinearGpu(std::vector<PodTri>& mesh, int levels) {
            using namespace RayTrophiSim;
            std::lock_guard<std::recursive_mutex> computeLock(sharedMeshComputeMutex());
            ISimulationComputeBackend* backend = acquireSharedMeshComputeBackend();
            if (!backend || mesh.empty() || levels <= 0) return false;

            const size_t inCount = mesh.size();
            size_t finalCount = inCount;
            for (int i = 0; i < levels; ++i) finalCount *= 4;
            const size_t bytes = finalCount * sizeof(GpuTri);

            ComputeBufferDesc desc;
            desc.debug_name = "subdivide_linear";
            desc.size_bytes = bytes;
            ComputeBufferHandle bufA = backend->createBuffer(desc);
            ComputeBufferHandle bufB = backend->createBuffer(desc);
            auto cleanup = [&]() {
                if (bufA.valid()) backend->destroyBuffer(bufA);
                if (bufB.valid()) backend->destroyBuffer(bufB);
            };
            if (!bufA.valid() || !bufB.valid()) { cleanup(); return false; }

            // Pack input triangles and upload to A.
            std::vector<GpuTri> host(inCount);
            for (size_t i = 0; i < inCount; ++i) {
                const PodTri& t = mesh[i];
                host[i] = GpuTri{ packVert(t.v[0]), packVert(t.v[1]), packVert(t.v[2]),
                                  static_cast<uint32_t>(t.mat), 0u, 0u, 0u };
            }
            if (!backend->uploadBuffer(bufA, host.data(), inCount * sizeof(GpuTri))) {
                cleanup();
                return false;
            }

            ComputeBufferHandle src = bufA, dst = bufB;
            uint32_t curCount = static_cast<uint32_t>(inCount);
            for (int level = 0; level < levels; ++level) {
                struct { uint32_t inCount; uint32_t pad0, pad1, pad2; } pc{ curCount, 0u, 0u, 0u };
                ComputeBufferHandle bufs[2] = { src, dst };
                ComputeDispatch cmd;
                cmd.kernel = "subdivide_linear";
                cmd.groups.groups_x = (curCount + 63u) / 64u;
                cmd.buffers = bufs;
                cmd.buffer_count = 2;
                cmd.constants = &pc;
                cmd.constants_size = sizeof(pc);
                if (!backend->dispatch(cmd)) { cleanup(); return false; }
                curCount *= 4u;
                std::swap(src, dst);   // latest result now lives in `src`
            }
            backend->synchronize();

            std::vector<GpuTri> out(finalCount);
            if (!backend->downloadBuffer(src, out.data(), finalCount * sizeof(GpuTri))) {
                cleanup();
                return false;
            }

            mesh.resize(finalCount);
            for (size_t i = 0; i < finalCount; ++i) {
                mesh[i] = PodTri{ { unpackVert(out[i].a), unpackVert(out[i].b), unpackVert(out[i].c) },
                                  static_cast<uint16_t>(out[i].mat) };
            }
            cleanup();
            return true;
        }

        // Flat/proxy: materialize the POD soup straight into ONE shared TriangleMesh (SoA) instead
        // of per-face Triangle facades — the same materialize+soup elimination CC gets, for Simple
        // (linear) subdivision. Emitted UN-WELDED (3N, indices 0..3N-1) so the linear subdivider's
        // exact per-corner normals are preserved (no weld = no risk of merging a flat-shaded mesh's
        // hard edges). P_orig/N_orig are local rest; P/N are world-baked via the shared transform —
        // matching convertFromRawArraysToMesh so every consumer (Embree/Vulkan/sculpt) reads it the
        // same way. The editable sculpt cache welds by position itself, so 3N is fine for editing.
        std::shared_ptr<TriangleMesh> podToFlatMesh(const std::vector<PodTri>& pod, const PodMeshMeta& meta) {
            const size_t n = pod.size();
            if (n == 0) return nullptr;
            auto tm = std::make_shared<TriangleMesh>();
            tm->transform = meta.transform;
            tm->nodeName = meta.nodeName;
            const size_t vCount = n * 3;
            tm->geometry->resize_vertices(vCount);
            tm->geometry->add_attribute<Vec3>("P");
            tm->geometry->add_attribute<Vec3>("N");
            tm->geometry->add_attribute<Vec3>("P_orig");
            tm->geometry->add_attribute<Vec3>("N_orig");
            tm->geometry->add_attribute<Vec2>("uv");
            tm->geometry->add_attribute<uint16_t>("materialID");
            Vec3*     P  = tm->geometry->get_attribute_data_mut<Vec3>("P");
            Vec3*     N  = tm->geometry->get_attribute_data_mut<Vec3>("N");
            Vec3*     Po = tm->geometry->get_attribute_data_mut<Vec3>("P_orig");
            Vec3*     No = tm->geometry->get_attribute_data_mut<Vec3>("N_orig");
            Vec2*     UV = tm->geometry->get_attribute_data_mut<Vec2>("uv");
            uint16_t* M  = tm->geometry->get_attribute_data_mut<uint16_t>("materialID");
            tm->geometry->indices.resize(vCount);
            Matrix4x4 finalT = Matrix4x4::identity(), normalT = Matrix4x4::identity();
            if (meta.transform) { finalT = meta.transform->getFinal(); normalT = meta.transform->getNormalTransform(); }
            #pragma omp parallel for num_threads(get_omp_threads_limit()) schedule(static) if(n >= 2048)
            for (int i = 0; i < (int)n; ++i) {
                const PodTri& t = pod[(size_t)i];
                for (int c = 0; c < 3; ++c) {
                    const size_t v = (size_t)i * 3 + (size_t)c;
                    const Vec3 lp = t.v[c].pos;
                    Vec3 ln = t.v[c].nrm;
                    const float nl = ln.length();
                    ln = (nl > 1e-6f) ? (ln / nl) : Vec3(0.0f, 1.0f, 0.0f);
                    if (Po) Po[v] = lp;
                    if (No) No[v] = ln;
                    if (UV) UV[v] = t.v[c].uv;
                    if (M)  M[v]  = t.mat;
                    const Vec3 wp = finalT.transform_point(lp);
                    Vec3 wn = normalT.transform_vector(ln);
                    const float wl = wn.length();
                    if (P) P[v] = wp;
                    if (N) N[v] = (wl > 1e-6f) ? (wn / wl) : Vec3(0.0f, 1.0f, 0.0f);
                    tm->geometry->indices[v] = (uint32_t)v;
                }
            }
            return tm;
        }

        // Simple (linear / FlatSubdivision) subdivide that returns ONE flat TriangleMesh. Mirrors
        // SubdivideSubD's POD pipeline (GPU when available, else CPU) then materializes flat.
        std::shared_ptr<TriangleMesh> subdivideSimpleToFlatMesh(
                const std::vector<std::shared_ptr<Triangle>>& inputMesh, int levels) {
            PodMeshMeta meta;
            std::vector<PodTri> mesh = trianglesToPod(inputMesh, meta);
            if (mesh.empty()) return nullptr;
            if (levels > 0) {
                size_t finalCount = mesh.size();
                for (int i = 0; i < levels; ++i) finalCount *= 4;
                const bool gpuOk = g_gpu_subdivide_enabled
                                && finalCount >= kGpuSubdivideMinTris
                                && subdivideLinearGpu(mesh, levels);
                if (!gpuOk) {
                    for (int level = 0; level < levels; ++level) subdividePodOnce(mesh);
                }
            }
            return podToFlatMesh(mesh, meta);
        }

    } // namespace

    std::shared_ptr<TriangleMesh> facadesToFlatMesh(const std::vector<std::shared_ptr<Triangle>>& facades) {
        if (facades.empty()) return nullptr;
        PodMeshMeta meta;
        std::vector<PodTri> pod = trianglesToPod(facades, meta);
        if (pod.empty()) return nullptr;
        auto mesh = podToFlatMesh(pod, meta);
        if (!mesh || !mesh->geometry) return mesh;

        // Preserve legacy/project-loaded rig weights while removing the facades.
        // The flat result is unwelded (3 vertices per face), so every corner maps
        // exactly to its source influence list.
        bool hasSkinWeights = false;
        for (const auto& triangle : facades) {
            if (triangle && triangle->hasAnySkinWeights()) {
                hasSkinWeights = true;
                break;
            }
        }
        if (hasSkinWeights) {
            const size_t vertexCount = facades.size() * 3;
            mesh->geometry->skin_weights.resize(vertexCount);
            for (size_t face = 0; face < facades.size(); ++face) {
                const auto& triangle = facades[face];
                if (!triangle) continue;
                for (int corner = 0; corner < 3; ++corner) {
                    mesh->geometry->skin_weights[face * 3 + static_cast<size_t>(corner)] =
                        triangle->getSkinBoneWeights(corner);
                }
            }

            // Skinned P/N stay mesh-local; the object Transform is applied once by
            // TriangleMesh/TLAS. podToFlatMesh world-bakes static P/N, so restore bind data.
            const Vec3* bindPositions = mesh->geometry->get_positions_orig();
            const Vec3* bindNormals = mesh->geometry->get_normals_orig();
            Vec3* positions = mesh->geometry->get_positions_mut();
            Vec3* normals = mesh->geometry->get_normals_mut();
            if (bindPositions && positions) {
                std::memcpy(positions, bindPositions, vertexCount * sizeof(Vec3));
            }
            if (bindNormals && normals) {
                std::memcpy(normals, bindNormals, vertexCount * sizeof(Vec3));
            }
        }
        return mesh;
    }

    std::vector<std::shared_ptr<Triangle>> SubdivideSubD(const std::vector<std::shared_ptr<Triangle>>& inputMesh, int levels) {
        if (levels <= 0) return inputMesh;

        PodMeshMeta meta;
        std::vector<PodTri> mesh = trianglesToPod(inputMesh, meta);
        if (mesh.empty()) return inputMesh;

        // Try the GPU path for large enough meshes (below the threshold the upload/
        // download round-trip loses to the CPU arithmetic). Falls back to the CPU loop
        // when the backend is unavailable or any GPU step fails.
        size_t finalCount = mesh.size();
        for (int i = 0; i < levels; ++i) finalCount *= 4;
        const bool gpuOk = g_gpu_subdivide_enabled
                        && finalCount >= kGpuSubdivideMinTris
                        && subdivideLinearGpu(mesh, levels);
        if (!gpuOk) {
            for (int level = 0; level < levels; ++level) {
                subdividePodOnce(mesh);
            }
        }

        return podToTriangles(mesh, meta, /*transformVertices=*/false);
    }

    std::vector<std::shared_ptr<Triangle>> SmoothSubD(const std::vector<std::shared_ptr<Triangle>>& inputMesh, int levels, float smoothAngle) {
        // 1. Linearly subdivide the POD soup to reach the topology density (no Triangle
        //    objects materialized for the intermediate levels).
        PodMeshMeta meta;
        std::vector<PodTri> mesh = trianglesToPod(inputMesh, meta);
        if (mesh.empty()) return inputMesh;
        for (int level = 0; level < levels; ++level) {
            subdividePodOnce(mesh);
        }
        const size_t triCount = mesh.size();

        // 2. Adaptive weld epsilon — must stay BELOW the mesh's vertex spacing. A fixed
        //    1e-4 (0.1mm) worked at low subdivision but on dense meshes (~700k tris) where
        //    the edge length drops under 1e-4, DISTINCT (sometimes opposite-facing) vertices
        //    collided into one bucket; their face normals then cancelled and produced ~zero
        //    normals (black shading in sculpt, brush falling back to world-up). Genuinely
        //    shared vertices here are bit-identical, so a fraction of the shortest edge welds
        //    them with huge margin while never merging neighbours. Clamped: unchanged at low
        //    density (cap 1e-4), float-safe at extreme density (floor 1e-6).
        float minEdgeSq = std::numeric_limits<float>::max();
        for (const auto& tri : mesh) {
            const float e0 = (tri.v[1].pos - tri.v[0].pos).length_squared();
            const float e1 = (tri.v[2].pos - tri.v[1].pos).length_squared();
            const float e2 = (tri.v[0].pos - tri.v[2].pos).length_squared();
            if (e0 > 1e-20f) minEdgeSq = (std::min)(minEdgeSq, e0);
            if (e1 > 1e-20f) minEdgeSq = (std::min)(minEdgeSq, e1);
            if (e2 > 1e-20f) minEdgeSq = (std::min)(minEdgeSq, e2);
        }
        const float minEdge = (minEdgeSq < std::numeric_limits<float>::max())
            ? std::sqrt(minEdgeSq) : 1e-4f;
        const float epsilon = std::clamp(minEdge * 0.25f, 1e-6f, 1e-4f);

        // 3. Weld every triangle corner to a shared vertex id ONCE. The old path re-hashed
        //    each corner on four separate passes (accumulate / apply / normal-accumulate /
        //    normal-apply); now each corner is quantize-hashed a single time and all the
        //    Laplacian + normal work runs on integer indices.
        std::unordered_map<QuantizedPositionKey, int, QuantizedPositionKeyHasher> vlookup;
        vlookup.reserve(triCount * 2);
        std::vector<int> corner2vid(triCount * 3);
        std::vector<Vec3> vpos;
        vpos.reserve(triCount);
        for (size_t t = 0; t < triCount; ++t) {
            for (int i = 0; i < 3; ++i) {
                const Vec3& p = mesh[t].v[i].pos;
                const QuantizedPositionKey key = quantizePosition(p, epsilon);
                auto it = vlookup.find(key);
                int id;
                if (it != vlookup.end()) {
                    id = it->second;
                } else {
                    id = static_cast<int>(vpos.size());
                    vlookup.emplace(key, id);
                    vpos.push_back(p);
                }
                corner2vid[t * 3 + i] = id;
            }
        }
        const int V = static_cast<int>(vpos.size());

        // 4. Laplacian position pass (1 iteration — same neighbour weighting as before:
        //    every incident triangle-corner contributes with multiplicity).
        std::vector<Vec3> acc(V, Vec3(0.0f, 0.0f, 0.0f));
        std::vector<int> cnt(V, 0);
        for (size_t t = 0; t < triCount; ++t) {
            for (int i = 0; i < 3; ++i) {
                const int vid = corner2vid[t * 3 + i];
                for (int j = 0; j < 3; ++j) {
                    if (i == j) continue;
                    acc[vid] += mesh[t].v[j].pos;
                    ++cnt[vid];
                }
            }
        }
        std::vector<Vec3> smoothed(V);
        for (int v = 0; v < V; ++v) {
            if (cnt[v] > 0) {
                const Vec3 avg = acc[v] * (1.0f / static_cast<float>(cnt[v]));
                // Interpolate between original and average based on smoothAngle (0..1).
                smoothed[v] = vpos[v] * (1.0f - smoothAngle) + avg * smoothAngle;
            } else {
                smoothed[v] = vpos[v];
            }
        }
        for (size_t t = 0; t < triCount; ++t)
            for (int i = 0; i < 3; ++i)
                mesh[t].v[i].pos = smoothed[corner2vid[t * 3 + i]];

        // 5. Recompute smooth vertex normals from the smoothed geometry (welded verts moved
        //    identically, so the corner->vid map is still valid).
        std::vector<Vec3> nrmAcc(V, Vec3(0.0f, 0.0f, 0.0f));
        for (size_t t = 0; t < triCount; ++t) {
            const Vec3 faceNormal =
                ((mesh[t].v[1].pos - mesh[t].v[0].pos).cross(mesh[t].v[2].pos - mesh[t].v[0].pos)).normalize();
            for (int i = 0; i < 3; ++i)
                nrmAcc[corner2vid[t * 3 + i]] += faceNormal;
        }
        for (auto& n : nrmAcc) n = n.normalize();
        for (size_t t = 0; t < triCount; ++t)
            for (int i = 0; i < 3; ++i)
                mesh[t].v[i].nrm = nrmAcc[corner2vid[t * 3 + i]];

        // 6. Materialize Triangle objects once (refresh transformed verts: positions moved).
        return podToTriangles(mesh, meta, /*transformVertices=*/true);
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
        auto sharedGeom = std::make_shared<OriginalMeshGeometry>();
        sharedGeom->positions = P;
        sharedGeom->normals = VN;

        std::vector<std::shared_ptr<Triangle>> out;
        out.reserve(F.size() * 2);
        auto emitTri = [&](int a, int b, int c,
                           const Vec2& ua, const Vec2& ub, const Vec2& uc,
                           uint16_t mat) {
            auto t = std::make_shared<Triangle>(
                P[a], P[b], P[c], VN[a], VN[b], VN[c], ua, ub, uc, mat);
            t->setTransformHandle(sharedTransform);
            t->setNodeName(nodeName);
            t->setOriginalGeometry(sharedGeom);
            t->setAssimpVertexIndices(
                static_cast<unsigned int>(a),
                static_cast<unsigned int>(b),
                static_cast<unsigned int>(c)
            );
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

    // ===================================================================================
    //  Catmull-Clark stencil engine (OpenSubdiv-style prepass + refine)
    //  Factors the CatmullClarkSubD position math above into position-INDEPENDENT sparse
    //  stencils so a cage edit only re-applies them (CPU or GPU), never rebuilds topology.
    //  Numerically identical to CatmullClarkSubD (same formulas, stored as weights).
    // ===================================================================================
    CCSubdivPlan buildCCSubdivPlan(
            const std::vector<std::shared_ptr<Triangle>>& inputMesh,
            int levels,
            const EdgeCreaseFn& creaseLookup) {
        CCSubdivPlan plan;
        if (inputMesh.empty() || levels <= 0) return plan;

        // [CCPLAN] TEMP sub-stage timing — at 10M+ output buildCCSubdivPlan dominates
        // (super-linear); pinpoint prepass-weld vs quad-recovery vs per-level stencil.
        const auto _pt0 = std::chrono::high_resolution_clock::now();
        auto _ptNow = []() { return std::chrono::high_resolution_clock::now(); };
        std::chrono::high_resolution_clock::time_point _pt1{}, _pt2{}, _pt3{};

        struct EdgeKey {
            int a = 0, b = 0;
            bool operator==(const EdgeKey& o) const { return a == o.a && b == o.b; }
        };
        struct EdgeKeyHasher {
            std::size_t operator()(const EdgeKey& k) const {
                return (static_cast<std::size_t>(static_cast<uint32_t>(k.a)) << 32) ^
                       static_cast<std::size_t>(static_cast<uint32_t>(k.b));
            }
        };
        auto makeEdge = [](int u, int v) -> EdgeKey { return (u < v) ? EdgeKey{ u, v } : EdgeKey{ v, u }; };

        // ---- Prepass: weld + quad recovery + crease seed (mirrors CatmullClarkSubD) ----
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

        std::vector<Vec3> P;
        std::vector<std::vector<int>> F;
        std::vector<std::vector<Vec2>> FUV;
        std::vector<uint16_t> FM;
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

        struct TriF { int v[3]; Vec2 uv[3]; uint16_t mat; };
        std::vector<TriF> tris;
        tris.reserve(inputMesh.size());
        plan.cornerToWelded.assign(inputMesh.size() * 3, -1);
        for (size_t ti = 0; ti < inputMesh.size(); ++ti) {
            const auto& tri = inputMesh[ti];
            if (!tri) continue;
            auto [uv0, uv1, uv2] = tri->getUVCoordinates();
            TriF tf;
            tf.v[0] = weld(tri->getOriginalVertexPosition(0));
            tf.v[1] = weld(tri->getOriginalVertexPosition(1));
            tf.v[2] = weld(tri->getOriginalVertexPosition(2));
            plan.cornerToWelded[ti * 3 + 0] = tf.v[0];
            plan.cornerToWelded[ti * 3 + 1] = tf.v[1];
            plan.cornerToWelded[ti * 3 + 2] = tf.v[2];
            if (tf.v[0] == tf.v[1] || tf.v[1] == tf.v[2] || tf.v[2] == tf.v[0]) continue;
            tf.uv[0] = uv0; tf.uv[1] = uv1; tf.uv[2] = uv2;
            tf.mat = tri->getMaterialID();
            tris.push_back(tf);
        }
        if (tris.empty()) return plan;
        _pt1 = _ptNow();   // prepass weld + tri build done

        // Quad recovery (identical criteria to CatmullClarkSubD).
        struct EdgeToTri { EdgeKey edge; int triIdx; };
        std::vector<EdgeToTri> e2tArray;
        e2tArray.reserve(tris.size() * 3);
        for (int t = 0; t < static_cast<int>(tris.size()); ++t) {
            for (int i = 0; i < 3; ++i) {
                e2tArray.push_back({ makeEdge(tris[t].v[i], tris[t].v[(i + 1) % 3]), t });
            }
        }
        std::sort(std::execution::par_unseq, e2tArray.begin(), e2tArray.end(), [](const EdgeToTri& a, const EdgeToTri& b) {
            if (a.edge.a != b.edge.a) return a.edge.a < b.edge.a;
            return a.edge.b < b.edge.b;
        });

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
                EdgeKey searchEdge = makeEdge(x, y);
                
                EdgeToTri searchObj = { searchEdge, 0 };
                auto range = std::equal_range(e2tArray.begin(), e2tArray.end(), searchObj, 
                    [](const EdgeToTri& a, const EdgeToTri& b) {
                        if (a.edge.a != b.edge.a) return a.edge.a < b.edge.a;
                        return a.edge.b < b.edge.b;
                    });

                for (auto it = range.first; it != range.second; ++it) {
                    int t2 = it->triIdx;
                    if (t2 == t || consumed[t2]) continue;
                    const Vec3 n2 = triNormal(tris[t2]);
                    const float l2 = n2.length();
                    if (l1 < 1e-12f || l2 < 1e-12f) continue;
                    if ((n1 * (1.0f / l1)).dot(n2 * (1.0f / l2)) < 0.90f) continue;
                    int ia = -1;
                    for (int a = 0; a < 3; ++a)
                        if (tris[t].v[a] != x && tris[t].v[a] != y) { ia = a; break; }
                    if (ia < 0) continue;
                    const int ib = (ia + 1) % 3;
                    const int ic = (ia + 2) % 3;
                    int c2 = -1;
                    for (int a = 0; a < 3; ++a)
                        if (tris[t2].v[a] != x && tris[t2].v[a] != y) { c2 = a; break; }
                    if (c2 < 0) continue;
                    auto uvOf = [&](const TriF& tf, int vi) -> Vec2 {
                        for (int a = 0; a < 3; ++a) if (tf.v[a] == vi) return tf.uv[a];
                        return Vec2(0.0f, 0.0f);
                    };
                    if ((uvOf(tris[t], x) - uvOf(tris[t2], x)).lengthSquared() > 1e-8f) continue;
                    if ((uvOf(tris[t], y) - uvOf(tris[t2], y)).lengthSquared() > 1e-8f) continue;
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
                    consumed[t] = 1; consumed[t2] = 1; merged = true;
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
        if (F.empty()) return plan;

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

        plan.cageVertCount = static_cast<int>(P.size());
        plan.cageP0 = P;
        _pt2 = _ptNow();   // quad recovery + crease seed done

        // ---- Per-level: build sparse stencil + advance topology ----
        int curV = static_cast<int>(P.size());   // current-level vertex count (positions are virtual here)
        for (int level = 0; level < levels; ++level) {
            const int V = curV;
            const int Fn = static_cast<int>(F.size());

            struct EdgeToFace { EdgeKey edge; int f; };
            std::vector<EdgeToFace> e2fArray;
            size_t numEdgesTotal = 0;
            for (int f = 0; f < Fn; ++f) numEdgesTotal += F[f].size();
            e2fArray.reserve(numEdgesTotal);
            for (int f = 0; f < Fn; ++f) {
                const auto& vs = F[f];
                const int k = static_cast<int>(vs.size());
                for (int i = 0; i < k; ++i) {
                    e2fArray.push_back({ makeEdge(vs[i], vs[(i + 1) % k]), f });
                }
            }

            std::sort(std::execution::par_unseq, e2fArray.begin(), e2fArray.end(), [](const EdgeToFace& a, const EdgeToFace& b) {
                if (a.edge.a != b.edge.a) return a.edge.a < b.edge.a;
                return a.edge.b < b.edge.b;
            });

            struct EdgeInfo { int faces[2] = { -1, -1 }; int nf = 0; };
            std::vector<std::pair<EdgeKey, EdgeInfo>> uniqueEdges;
            uniqueEdges.reserve(Fn * 2);
            for (size_t i = 0; i < e2fArray.size(); ) {
                EdgeKey currentEdge = e2fArray[i].edge;
                EdgeInfo info;
                while (i < e2fArray.size() && e2fArray[i].edge == currentEdge) {
                    if (info.nf < 2) info.faces[info.nf] = e2fArray[i].f;
                    info.nf++;
                    i++;
                }
                uniqueEdges.push_back({currentEdge, info});
            }

            // Stable edge ordering + per-edge sharpness + edge index in the NP layout.
            std::vector<EdgeKey> edgeList; edgeList.reserve(uniqueEdges.size());
            std::vector<float> edgeSig;    edgeSig.reserve(uniqueEdges.size());
            std::vector<std::array<int, 2>> edgeFaces; edgeFaces.reserve(uniqueEdges.size());
            
            for (size_t i = 0; i < uniqueEdges.size(); ++i) {
                const auto& kv = uniqueEdges[i];
                edgeList.push_back(kv.first);
                edgeFaces.push_back({ kv.second.faces[0], kv.second.faces[1] });
                float sigma = 0.0f;
                if (kv.second.nf < 2) sigma = 1.0f;
                else { auto it = edgeCrease.find(kv.first); if (it != edgeCrease.end()) sigma = std::clamp(it->second, 0.0f, 1.0f); }
                edgeSig.push_back(sigma);
            }
            const int E = static_cast<int>(edgeList.size());
            const int faceBase = V + E;
            
            auto getEdgeIdx = [&](EdgeKey e) {
                auto it = std::lower_bound(uniqueEdges.begin(), uniqueEdges.end(), e, [](const auto& item, const EdgeKey& val) {
                    if (item.first.a != val.a) return item.first.a < val.a;
                    return item.first.b < val.b;
                });
                return V + static_cast<int>(std::distance(uniqueEdges.begin(), it));
            };

            // Per-vertex incidence (faces + edge neighbours) and crease counts.
            std::vector<int> vFaceCnt(V, 0), vEdgeCnt(V, 0), vSharpCnt(V, 0);
            std::vector<std::array<int, 2>> vSharpNbr(V, std::array<int, 2>{ -1, -1 });
            std::vector<std::vector<int>> incidentFaces(V);
            std::vector<std::vector<int>> incidentNbr(V);
            for (int f = 0; f < Fn; ++f)
                for (int vi : F[f]) { ++vFaceCnt[vi]; incidentFaces[vi].push_back(f); }
            for (int i = 0; i < E; ++i) {
                const EdgeKey& e = edgeList[i];
                ++vEdgeCnt[e.a]; incidentNbr[e.a].push_back(e.b);
                ++vEdgeCnt[e.b]; incidentNbr[e.b].push_back(e.a);
                if (edgeSig[i] > 0.0f) {
                    if (vSharpCnt[e.a] < 2) vSharpNbr[e.a][vSharpCnt[e.a]] = e.b;
                    ++vSharpCnt[e.a];
                    if (vSharpCnt[e.b] < 2) vSharpNbr[e.b][vSharpCnt[e.b]] = e.a;
                    ++vSharpCnt[e.b];
                }
            }

            // Accumulate weights per output vertex into ONE flat scratch (CSR), not a per-row
            // std::vector. Each output row is built independently; duplicate input ids inside a
            // row are merged after the fill. This evolved from a per-row std::map<int,float>
            // (red-black-tree node per weight — the original single-threaded cost + multi-GB heap
            // spike) → a per-row std::vector<pair> (still ONE heap allocation PER output vertex:
            // ~12.6M tiny allocs at the final level, ~700MB of vector overhead, and the dominant
            // remaining allocation churn) → this flat scratch: an exact pre-count sizes a single
            // contiguous buffer, three passes fill disjoint slices in parallel, then each slice is
            // sorted/merged in place. Output is numerically equivalent (ascending idx, summed
            // weights; FP summation order may differ negligibly, same as the existing GPU path).
            CCSubdivPlan::StencilLevel sl;
            sl.inCount = V;
            sl.outCount = V + E + Fn;
            const int outCount = sl.outCount;

            // Single source of truth for a row's (idx, weight) entries. The count pass runs it with
            // a counting sink, the fill pass with a writing sink — they can never drift out of sync.
            auto genRow = [&](int o, auto&& emit) {
                if (o < V) {
                    const int v = o;                                 // vertex point: newVertPt classes
                    if (vFaceCnt[v] == 0 || vSharpCnt[v] >= 3) {
                        emit(v, 1.0f);                               // corner / isolated → fixed
                    } else if (vSharpCnt[v] == 2) {
                        emit(v, 6.0f / 8.0f);
                        emit(vSharpNbr[v][0], 1.0f / 8.0f);
                        emit(vSharpNbr[v][1], 1.0f / 8.0f);
                    } else {
                        const float n = static_cast<float>(vFaceCnt[v]);
                        emit(v, (n - 3.0f) / n);                     // P[v] term
                        for (int f : incidentFaces[v]) {             // Q/n
                            const int k = static_cast<int>(F[f].size());
                            const float wf = 1.0f / (n * n * static_cast<float>(k));
                            for (int vi : F[f]) emit(vi, wf);
                        }
                        const int ec = vEdgeCnt[v];                  // 2R/n
                        if (ec > 0) {
                            const float we = 1.0f / (n * static_cast<float>(ec));
                            for (int w : incidentNbr[v]) { emit(v, we); emit(w, we); }
                        }
                    }
                } else if (o < V + E) {
                    const int i = o - V;                             // edge point: (1-σ)smooth + σ mid
                    const EdgeKey& e = edgeList[i];
                    const float sigma = edgeSig[i];
                    const int f0 = edgeFaces[i][0], f1 = edgeFaces[i][1];
                    if (f1 < 0) {                                    // boundary (nf < 2)
                        emit(e.a, 0.5f); emit(e.b, 0.5f);
                    } else {
                        const float sm = 1.0f - sigma;
                        emit(e.a, sm * 0.25f + sigma * 0.5f);
                        emit(e.b, sm * 0.25f + sigma * 0.5f);
                        { const int k0 = static_cast<int>(F[f0].size()); const float w = sm * 0.25f / static_cast<float>(k0); for (int vi : F[f0]) emit(vi, w); }
                        { const int k1 = static_cast<int>(F[f1].size()); const float w = sm * 0.25f / static_cast<float>(k1); for (int vi : F[f1]) emit(vi, w); }
                    }
                } else {
                    const int f = o - (V + E);                       // face point: centroid
                    const int k = static_cast<int>(F[f].size());
                    const float w = 1.0f / static_cast<float>(k);
                    for (int vi : F[f]) emit(vi, w);
                }
            };

            // Pass 1: exact pre-dedup entry count per row → flat-scratch offsets (one allocation).
            std::vector<int> rawLen(outCount, 0);
            #pragma omp parallel for num_threads(get_omp_threads_limit()) schedule(dynamic, 512) if(outCount >= 4096)
            for (int o = 0; o < outCount; ++o) {
                int c = 0;
                genRow(o, [&](int, float) { ++c; });
                rawLen[o] = c;
            }
            std::vector<int64_t> rawOff(static_cast<size_t>(outCount) + 1, 0);
            for (int o = 0; o < outCount; ++o) rawOff[o + 1] = rawOff[o] + rawLen[o];
            std::vector<std::pair<int, float>> scratch(static_cast<size_t>(rawOff[outCount]));

            // Pass 2: fill each row's disjoint slice (parallel, no contention).
            #pragma omp parallel for num_threads(get_omp_threads_limit()) schedule(dynamic, 512) if(outCount >= 4096)
            for (int o = 0; o < outCount; ++o) {
                int64_t p = rawOff[o];
                genRow(o, [&](int idx, float w) { scratch[static_cast<size_t>(p++)] = { idx, w }; });
            }

            // Pass 3: sort + merge duplicate input ids within each slice, in place. rowLen = merged.
            std::vector<int> rowLen(outCount, 0);
            #pragma omp parallel for num_threads(get_omp_threads_limit()) schedule(dynamic, 512) if(outCount >= 4096)
            for (int o = 0; o < outCount; ++o) {
                const int64_t s = rawOff[o], e = rawOff[o + 1];
                if (e <= s) continue;
                std::sort(scratch.begin() + s, scratch.begin() + e,
                          [](const std::pair<int, float>& a, const std::pair<int, float>& b) { return a.first < b.first; });
                int64_t wpos = s;
                for (int64_t rd = s; rd < e; ) {
                    const int key = scratch[static_cast<size_t>(rd)].first;
                    float acc = 0.0f;
                    int64_t rr = rd;
                    while (rr < e && scratch[static_cast<size_t>(rr)].first == key) { acc += scratch[static_cast<size_t>(rr)].second; ++rr; }
                    scratch[static_cast<size_t>(wpos++)] = { key, acc };
                    rd = rr;
                }
                rowLen[o] = static_cast<int>(wpos - s);
            }

            // Prefix-sum merged lengths → CSR offsets, then compact each slice into the output.
            sl.off.assign(sl.outCount + 1, 0);
            for (int o = 0; o < outCount; ++o) sl.off[o + 1] = sl.off[o] + rowLen[o];
            sl.idx.resize(sl.off[sl.outCount]);
            sl.w.resize(sl.off[sl.outCount]);
            #pragma omp parallel for num_threads(get_omp_threads_limit()) schedule(dynamic, 512) if(outCount >= 4096)
            for (int o = 0; o < outCount; ++o) {
                const int64_t src = rawOff[o];
                int p = sl.off[o];
                const int len = rowLen[o];
                for (int j = 0; j < len; ++j) {
                    const auto& kv = scratch[static_cast<size_t>(src + j)];
                    sl.idx[p] = kv.first; sl.w[p] = kv.second; ++p;
                }
            }
            plan.levels.push_back(std::move(sl));

            // Advance topology (child quads), identical to CatmullClarkSubD.
            std::vector<std::vector<int>> NF;
            std::vector<std::vector<Vec2>> NFUV;
            std::vector<uint16_t> NFM;
            NF.reserve(Fn * 4); NFUV.reserve(Fn * 4); NFM.reserve(Fn * 4);
            for (int f = 0; f < Fn; ++f) {
                const auto& vs = F[f];
                const auto& uvs = FUV[f];
                const int k = static_cast<int>(vs.size());
                Vec2 cUV(0.0f, 0.0f);
                for (const auto& u : uvs) cUV = cUV + u;
                cUV = cUV * (1.0f / static_cast<float>(k));
                const int centerIdx = faceBase + f;
                for (int i = 0; i < k; ++i) {
                    const int vCur = vs[i];
                    const int vNext = vs[(i + 1) % k];
                    const int vPrev = vs[(i - 1 + k) % k];
                    const int eNextIdx = getEdgeIdx(makeEdge(vCur, vNext));
                    const int ePrevIdx = getEdgeIdx(makeEdge(vPrev, vCur));
                    NF.push_back({ vCur, eNextIdx, centerIdx, ePrevIdx });
                    const Vec2 uvNext = (uvs[i] + uvs[(i + 1) % k]) * 0.5f;
                    const Vec2 uvPrev = (uvs[(i - 1 + k) % k] + uvs[i]) * 0.5f;
                    NFUV.push_back({ uvs[i], uvNext, cUV, uvPrev });
                    NFM.push_back(FM[f]);
                }
            }
            std::unordered_map<EdgeKey, float, EdgeKeyHasher> newCrease;
            for (int i = 0; i < E; ++i) {
                if (edgeFaces[i][1] < 0) continue;
                auto it = edgeCrease.find(edgeList[i]);
                if (it == edgeCrease.end() || it->second <= 0.0f) continue;
                const int ep = getEdgeIdx(edgeList[i]);
                newCrease[makeEdge(edgeList[i].a, ep)] = it->second;
                newCrease[makeEdge(ep, edgeList[i].b)] = it->second;
            }

            curV = plan.levels.back().outCount;   // next-level vertex count
            F.swap(NF); FUV.swap(NFUV); FM.swap(NFM); edgeCrease.swap(newCrease);
        }

        plan.finalVertCount = curV;
        _pt3 = _ptNow();   // per-level stencil build done

        // ---- Final triangulated geometry + face CSR + vertex→face CSR ----
        int currentFace = 0;  // source polygon id of the triangles being emitted (for loop normals)
        auto emitTri = [&](int a, int b, int c, const Vec2& ua, const Vec2& ub, const Vec2& uc, uint16_t mat) {
            plan.triIndices.push_back(static_cast<uint32_t>(a));
            plan.triIndices.push_back(static_cast<uint32_t>(b));
            plan.triIndices.push_back(static_cast<uint32_t>(c));
            plan.triUV.push_back(ua); plan.triUV.push_back(ub); plan.triUV.push_back(uc);
            plan.triMat.push_back(mat);
            plan.triFace.push_back(currentFace);
        };
        plan.faceVertOff.push_back(0);
        for (size_t f = 0; f < F.size(); ++f) {
            currentFace = static_cast<int>(f);
            const auto& vs = F[f];
            const auto& uvs = FUV[f];
            const uint16_t mat = FM[f];
            for (int vi : vs) plan.faceVertIdx.push_back(vi);
            plan.faceVertOff.push_back(static_cast<int>(plan.faceVertIdx.size()));
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

        // vertex → incident final FACE CSR.
        const int numFaces = static_cast<int>(plan.faceVertOff.size()) - 1;
        std::vector<int> vcnt(plan.finalVertCount, 0);
        for (int f = 0; f < numFaces; ++f)
            for (int p = plan.faceVertOff[f]; p < plan.faceVertOff[f + 1]; ++p)
                ++vcnt[plan.faceVertIdx[p]];
        plan.vfaceOff.assign(plan.finalVertCount + 1, 0);
        for (int v = 0; v < plan.finalVertCount; ++v) plan.vfaceOff[v + 1] = plan.vfaceOff[v] + vcnt[v];
        plan.vfaceIdx.resize(plan.vfaceOff[plan.finalVertCount]);
        std::vector<int> vcur(plan.vfaceOff.begin(), plan.vfaceOff.end() - 1);
        for (int f = 0; f < numFaces; ++f)
            for (int p = plan.faceVertOff[f]; p < plan.faceVertOff[f + 1]; ++p)
                plan.vfaceIdx[vcur[plan.faceVertIdx[p]]++] = f;

        {
            const auto _pt4 = _ptNow();
            auto ms = [](const std::chrono::high_resolution_clock::time_point& a,
                         const std::chrono::high_resolution_clock::time_point& b) {
                return std::chrono::duration<double, std::milli>(b - a).count();
            };
            char buf[224];
            std::snprintf(buf, sizeof(buf),
                "[CCPLAN] finalV=%d faces=%d | prepass=%.0f ms  quadRec=%.0f ms  stencil=%.0f ms  triangulate=%.0f ms",
                plan.finalVertCount, numFaces,
                ms(_pt0, _pt1), ms(_pt1, _pt2), ms(_pt2, _pt3), ms(_pt3, _pt4));
            SCENE_LOG_INFO(buf);
        }
        return plan;
    }

    namespace {

        // ---- GPU Catmull-Clark refine (Phase 3b) ----
        // Mirrors evaluateCCPositions on the shared Vulkan compute backend: per-level
        // sparse stencil apply (cc_stencil_apply), then Newell face normals
        // (cc_face_normals) gathered per vertex atomic-free (cc_vertex_normals). Drop-in
        // like the linear path — uploads the plan's cached CSR + the (edited) cage,
        // downloads the final positions + smooth normals. Returns false (caller runs the
        // CPU path) when the backend is unavailable or any GPU step fails. Output is
        // numerically equivalent to the CPU loop (GPU float rounding may differ).
        bool evaluateCCPositionsGpu(const CCSubdivPlan& plan,
                                    const std::vector<Vec3>& cagePositions,
                                    std::vector<Vec3>& outPositions,
                                    std::vector<Vec3>& outNormals) {
            using namespace RayTrophiSim;
            if (plan.levels.empty()) return false;            // nothing to refine on GPU
            std::lock_guard<std::recursive_mutex> computeLock(sharedMeshComputeMutex());
            ISimulationComputeBackend* backend = acquireSharedMeshComputeBackend();
            if (!backend) return false;

            const int finalV = plan.finalVertCount;
            const int numFaces = static_cast<int>(plan.faceVertOff.size()) - 1;
            if (finalV <= 0 || numFaces <= 0) return false;

            std::vector<ComputeBufferHandle> owned;
            auto makeBuf = [&](size_t bytes, const char* name) -> ComputeBufferHandle {
                ComputeBufferDesc d; d.debug_name = name; d.size_bytes = bytes;
                ComputeBufferHandle h = backend->createBuffer(d);
                if (h.valid()) owned.push_back(h);
                return h;
            };
            auto cleanup = [&]() { for (auto& h : owned) backend->destroyBuffer(h); };

            // Ping-pong position buffers, sized to the largest (final) level.
            const size_t posBytes = static_cast<size_t>(finalV) * 3 * sizeof(float);
            ComputeBufferHandle posA = makeBuf(posBytes, "cc_pos_a");
            ComputeBufferHandle posB = makeBuf(posBytes, "cc_pos_b");
            if (!posA.valid() || !posB.valid()) { cleanup(); return false; }

            // Upload the (possibly edited) cage positions into A.
            {
                std::vector<float> flat(static_cast<size_t>(plan.cageVertCount) * 3);
                for (int i = 0; i < plan.cageVertCount; ++i) {
                    flat[i * 3 + 0] = cagePositions[i].x;
                    flat[i * 3 + 1] = cagePositions[i].y;
                    flat[i * 3 + 2] = cagePositions[i].z;
                }
                if (!backend->uploadBuffer(posA, flat.data(), flat.size() * sizeof(float))) { cleanup(); return false; }
            }

            // Upload every level's CSR stencil (kept alive until after synchronize()).
            struct LevelBufs { ComputeBufferHandle off, idx, w; uint32_t outCount; };
            std::vector<LevelBufs> lb;
            lb.reserve(plan.levels.size());
            for (const auto& sl : plan.levels) {
                LevelBufs b;
                b.outCount = static_cast<uint32_t>(sl.outCount);
                b.off = makeBuf(sl.off.size() * sizeof(int),   "cc_off");
                b.idx = makeBuf(sl.idx.size() * sizeof(int),   "cc_idx");
                b.w   = makeBuf(sl.w.size()   * sizeof(float), "cc_w");
                if (!b.off.valid() || !b.idx.valid() || !b.w.valid()) { cleanup(); return false; }
                if (!backend->uploadBuffer(b.off, sl.off.data(), sl.off.size() * sizeof(int)) ||
                    !backend->uploadBuffer(b.idx, sl.idx.data(), sl.idx.size() * sizeof(int)) ||
                    !backend->uploadBuffer(b.w,   sl.w.data(),   sl.w.size()   * sizeof(float))) { cleanup(); return false; }
                lb.push_back(b);
            }

            // Per-level sparse stencil apply (ping-pong; final positions end up in `src`).
            ComputeBufferHandle src = posA, dst = posB;
            for (const auto& b : lb) {
                struct { uint32_t outCount, p0, p1, p2; } pc{ b.outCount, 0u, 0u, 0u };
                ComputeBufferHandle bufs[5] = { src, dst, b.off, b.idx, b.w };
                ComputeDispatch cmd;
                cmd.kernel = "cc_stencil_apply";
                cmd.groups.groups_x = (b.outCount + 63u) / 64u;
                cmd.buffers = bufs; cmd.buffer_count = 5;
                cmd.constants = &pc; cmd.constants_size = sizeof(pc);
                if (!backend->dispatch(cmd)) { cleanup(); return false; }
                std::swap(src, dst);
            }

            // Newell normal per final face.
            ComputeBufferHandle faceOff = makeBuf(plan.faceVertOff.size() * sizeof(int),    "cc_face_off");
            ComputeBufferHandle faceIdx = makeBuf(plan.faceVertIdx.size() * sizeof(int),    "cc_face_idx");
            ComputeBufferHandle faceN   = makeBuf(static_cast<size_t>(numFaces) * 3 * sizeof(float), "cc_face_n");
            if (!faceOff.valid() || !faceIdx.valid() || !faceN.valid()) { cleanup(); return false; }
            if (!backend->uploadBuffer(faceOff, plan.faceVertOff.data(), plan.faceVertOff.size() * sizeof(int)) ||
                !backend->uploadBuffer(faceIdx, plan.faceVertIdx.data(), plan.faceVertIdx.size() * sizeof(int))) { cleanup(); return false; }
            {
                struct { uint32_t numFaces, p0, p1, p2; } pc{ static_cast<uint32_t>(numFaces), 0u, 0u, 0u };
                ComputeBufferHandle bufs[4] = { src, faceOff, faceIdx, faceN };
                ComputeDispatch cmd;
                cmd.kernel = "cc_face_normals";
                cmd.groups.groups_x = (static_cast<uint32_t>(numFaces) + 63u) / 64u;
                cmd.buffers = bufs; cmd.buffer_count = 4;
                cmd.constants = &pc; cmd.constants_size = sizeof(pc);
                if (!backend->dispatch(cmd)) { cleanup(); return false; }
            }

            // Atomic-free per-vertex normal gather.
            ComputeBufferHandle vfOff = makeBuf(plan.vfaceOff.size() * sizeof(int), "cc_vf_off");
            ComputeBufferHandle vfIdx = makeBuf(plan.vfaceIdx.size() * sizeof(int), "cc_vf_idx");
            ComputeBufferHandle outN  = makeBuf(posBytes, "cc_out_n");
            if (!vfOff.valid() || !vfIdx.valid() || !outN.valid()) { cleanup(); return false; }
            if (!backend->uploadBuffer(vfOff, plan.vfaceOff.data(), plan.vfaceOff.size() * sizeof(int)) ||
                !backend->uploadBuffer(vfIdx, plan.vfaceIdx.data(), plan.vfaceIdx.size() * sizeof(int))) { cleanup(); return false; }
            {
                struct { uint32_t vertCount, p0, p1, p2; } pc{ static_cast<uint32_t>(finalV), 0u, 0u, 0u };
                ComputeBufferHandle bufs[4] = { faceN, vfOff, vfIdx, outN };
                ComputeDispatch cmd;
                cmd.kernel = "cc_vertex_normals";
                cmd.groups.groups_x = (static_cast<uint32_t>(finalV) + 63u) / 64u;
                cmd.buffers = bufs; cmd.buffer_count = 4;
                cmd.constants = &pc; cmd.constants_size = sizeof(pc);
                if (!backend->dispatch(cmd)) { cleanup(); return false; }
            }

            backend->synchronize();

            // Download final positions + normals and unflatten.
            std::vector<float> posOut(static_cast<size_t>(finalV) * 3);
            std::vector<float> nrmOut(static_cast<size_t>(finalV) * 3);
            if (!backend->downloadBuffer(src,  posOut.data(), posOut.size() * sizeof(float)) ||
                !backend->downloadBuffer(outN, nrmOut.data(), nrmOut.size() * sizeof(float))) { cleanup(); return false; }

            outPositions.resize(finalV);
            outNormals.resize(finalV);
            for (int i = 0; i < finalV; ++i) {
                outPositions[i] = Vec3(posOut[i * 3 + 0], posOut[i * 3 + 1], posOut[i * 3 + 2]);
                outNormals[i]   = Vec3(nrmOut[i * 3 + 0], nrmOut[i * 3 + 1], nrmOut[i * 3 + 2]);
            }
            cleanup();
            return true;
        }

    } // namespace

    void evaluateCCPositions(
            const CCSubdivPlan& plan,
            const std::vector<Vec3>& cagePositions,
            std::vector<Vec3>& outPositions,
            std::vector<Vec3>& outNormals) {
        outPositions.clear();
        outNormals.clear();
        if (!plan.valid() || static_cast<int>(cagePositions.size()) != plan.cageVertCount) return;

        // GPU refine (Phase 3b) for large enough meshes; drops to the CPU path on any
        // failure (backend unavailable, alloc/upload/dispatch error) with identical output.
        if (g_gpu_subdivide_enabled
                && plan.finalVertCount >= static_cast<int>(kGpuSubdivideMinTris)
                && evaluateCCPositionsGpu(plan, cagePositions, outPositions, outNormals)) {
            return;
        }
        outPositions.clear();
        outNormals.clear();

        // ---- Position refine: ping-pong sparse stencil apply per level (CPU path) ----
        std::vector<Vec3> cur = cagePositions;
        for (const auto& sl : plan.levels) {
            std::vector<Vec3> next(sl.outCount);
            const int n = sl.outCount;
            #pragma omp parallel for num_threads(get_omp_threads_limit()) schedule(static) if(n >= 4096)
            for (int o = 0; o < n; ++o) {
                Vec3 acc(0.0f, 0.0f, 0.0f);
                for (int k = sl.off[o]; k < sl.off[o + 1]; ++k)
                    acc += cur[sl.idx[k]] * sl.w[k];
                next[o] = acc;
            }
            cur.swap(next);
        }
        outPositions.swap(cur);

        // ---- Normals: Newell per FINAL FACE (quad), gathered per vertex (matches ref) ----
        const int numFaces = static_cast<int>(plan.faceVertOff.size()) - 1;
        std::vector<Vec3> faceN(numFaces);
        #pragma omp parallel for num_threads(get_omp_threads_limit()) schedule(static) if(numFaces >= 4096)
        for (int f = 0; f < numFaces; ++f) {
            Vec3 nrm(0.0f, 0.0f, 0.0f);
            const int beg = plan.faceVertOff[f], end = plan.faceVertOff[f + 1];
            const int k = end - beg;
            for (int i = 0; i < k; ++i) {
                const Vec3& cu = outPositions[plan.faceVertIdx[beg + i]];
                const Vec3& nx = outPositions[plan.faceVertIdx[beg + (i + 1) % k]];
                nrm.x += (cu.y - nx.y) * (cu.z + nx.z);
                nrm.y += (cu.z - nx.z) * (cu.x + nx.x);
                nrm.z += (cu.x - nx.x) * (cu.y + nx.y);
            }
            faceN[f] = nrm;
        }
        outNormals.assign(plan.finalVertCount, Vec3(0.0f, 0.0f, 0.0f));
        #pragma omp parallel for num_threads(get_omp_threads_limit()) schedule(static) if(plan.finalVertCount >= 4096)
        for (int v = 0; v < plan.finalVertCount; ++v) {
            Vec3 acc(0.0f, 0.0f, 0.0f);
            for (int k = plan.vfaceOff[v]; k < plan.vfaceOff[v + 1]; ++k)
                acc += faceN[plan.vfaceIdx[k]];
            const float L = acc.length();
            outNormals[v] = (L > 1e-12f) ? acc * (1.0f / L) : Vec3(0.0f, 1.0f, 0.0f);
        }
    }

    // Phase 3d: evaluate the plan on the GPU and expand into a device-resident BLAS-layout
    // buffer (no host download). The stencil + normal sequence mirrors evaluateCCPositionsGpu
    // (kept duplicated rather than refactored so the proven download path stays untouched);
    // the difference is the final step keeps positions/normals on the device and runs the
    // cc_expand_blas kernel into a persistent AccelInput buffer that the caller owns.
    bool evaluateCCToDeviceGeometry(
            const CCSubdivPlan& plan,
            const std::vector<Vec3>& cagePositions,
            CCDeviceGeometry& out) {
        using namespace RayTrophiSim;
        out = {};
        if (plan.levels.empty()) return false;
        if (static_cast<int>(cagePositions.size()) != plan.cageVertCount) return false;
        std::lock_guard<std::recursive_mutex> computeLock(sharedMeshComputeMutex());
        ISimulationComputeBackend* backend = acquireSharedMeshComputeBackend();
        if (!backend) return false;

        const int finalV   = plan.finalVertCount;
        const int numFaces = static_cast<int>(plan.faceVertOff.size()) - 1;
        const int triCount = static_cast<int>(plan.triMat.size());
        if (finalV <= 0 || numFaces <= 0 || triCount <= 0) return false;
        if (static_cast<int>(plan.triIndices.size()) != triCount * 3 ||
            static_cast<int>(plan.triUV.size())      != triCount * 3) return false;

        std::vector<ComputeBufferHandle> scratch;   // everything except the kept output buffer
        auto makeBuf = [&](size_t bytes, const char* name, bool accelInput = false) -> ComputeBufferHandle {
            ComputeBufferDesc d; d.debug_name = name; d.size_bytes = bytes;
            if (accelInput) d.usage = ComputeBufferUsage::Storage | ComputeBufferUsage::AccelInput;
            ComputeBufferHandle h = backend->createBuffer(d);
            return h;
        };
        auto track = [&](ComputeBufferHandle h) { if (h.valid()) scratch.push_back(h); return h; };
        auto cleanupScratch = [&]() { for (auto& h : scratch) backend->destroyBuffer(h); };

        // --- Position refine (ping-pong stencil apply), identical to the download path ---
        const size_t posBytes = static_cast<size_t>(finalV) * 3 * sizeof(float);
        ComputeBufferHandle posA = track(makeBuf(posBytes, "ccd_pos_a"));
        ComputeBufferHandle posB = track(makeBuf(posBytes, "ccd_pos_b"));
        if (!posA.valid() || !posB.valid()) { cleanupScratch(); return false; }
        {
            std::vector<float> flat(static_cast<size_t>(plan.cageVertCount) * 3);
            for (int i = 0; i < plan.cageVertCount; ++i) {
                flat[i * 3 + 0] = cagePositions[i].x;
                flat[i * 3 + 1] = cagePositions[i].y;
                flat[i * 3 + 2] = cagePositions[i].z;
            }
            if (!backend->uploadBuffer(posA, flat.data(), flat.size() * sizeof(float))) { cleanupScratch(); return false; }
        }
        struct LevelBufs { ComputeBufferHandle off, idx, w; uint32_t outCount; };
        std::vector<LevelBufs> lb; lb.reserve(plan.levels.size());
        for (const auto& sl : plan.levels) {
            LevelBufs b; b.outCount = static_cast<uint32_t>(sl.outCount);
            b.off = track(makeBuf(sl.off.size() * sizeof(int),   "ccd_off"));
            b.idx = track(makeBuf(sl.idx.size() * sizeof(int),   "ccd_idx"));
            b.w   = track(makeBuf(sl.w.size()   * sizeof(float), "ccd_w"));
            if (!b.off.valid() || !b.idx.valid() || !b.w.valid()) { cleanupScratch(); return false; }
            if (!backend->uploadBuffer(b.off, sl.off.data(), sl.off.size() * sizeof(int)) ||
                !backend->uploadBuffer(b.idx, sl.idx.data(), sl.idx.size() * sizeof(int)) ||
                !backend->uploadBuffer(b.w,   sl.w.data(),   sl.w.size()   * sizeof(float))) { cleanupScratch(); return false; }
            lb.push_back(b);
        }
        ComputeBufferHandle src = posA, dst = posB;
        for (const auto& b : lb) {
            struct { uint32_t outCount, p0, p1, p2; } pcStencil{ b.outCount, 0u, 0u, 0u };
            ComputeBufferHandle bufs[5] = { src, dst, b.off, b.idx, b.w };
            ComputeDispatch cmd; cmd.kernel = "cc_stencil_apply";
            cmd.groups.groups_x = (b.outCount + 63u) / 64u;
            cmd.buffers = bufs; cmd.buffer_count = 5;
            cmd.constants = &pcStencil; cmd.constants_size = sizeof(pcStencil);
            if (!backend->dispatch(cmd)) { cleanupScratch(); return false; }
            std::swap(src, dst);   // final positions in `src`
        }

        // --- Normals (face Newell -> atomic-free vertex gather), as in the download path ---
        ComputeBufferHandle faceOff = track(makeBuf(plan.faceVertOff.size() * sizeof(int), "ccd_face_off"));
        ComputeBufferHandle faceIdx = track(makeBuf(plan.faceVertIdx.size() * sizeof(int), "ccd_face_idx"));
        ComputeBufferHandle faceN   = track(makeBuf(static_cast<size_t>(numFaces) * 3 * sizeof(float), "ccd_face_n"));
        if (!faceOff.valid() || !faceIdx.valid() || !faceN.valid()) { cleanupScratch(); return false; }
        if (!backend->uploadBuffer(faceOff, plan.faceVertOff.data(), plan.faceVertOff.size() * sizeof(int)) ||
            !backend->uploadBuffer(faceIdx, plan.faceVertIdx.data(), plan.faceVertIdx.size() * sizeof(int))) { cleanupScratch(); return false; }
        {
            struct { uint32_t numFaces, p0, p1, p2; } pcF{ static_cast<uint32_t>(numFaces), 0u, 0u, 0u };
            ComputeBufferHandle bufs[4] = { src, faceOff, faceIdx, faceN };
            ComputeDispatch cmd; cmd.kernel = "cc_face_normals";
            cmd.groups.groups_x = (static_cast<uint32_t>(numFaces) + 63u) / 64u;
            cmd.buffers = bufs; cmd.buffer_count = 4;
            cmd.constants = &pcF; cmd.constants_size = sizeof(pcF);
            if (!backend->dispatch(cmd)) { cleanupScratch(); return false; }
        }
        ComputeBufferHandle vfOff = track(makeBuf(plan.vfaceOff.size() * sizeof(int), "ccd_vf_off"));
        ComputeBufferHandle vfIdx = track(makeBuf(plan.vfaceIdx.size() * sizeof(int), "ccd_vf_idx"));
        ComputeBufferHandle nrm   = track(makeBuf(posBytes, "ccd_nrm"));
        if (!vfOff.valid() || !vfIdx.valid() || !nrm.valid()) { cleanupScratch(); return false; }
        if (!backend->uploadBuffer(vfOff, plan.vfaceOff.data(), plan.vfaceOff.size() * sizeof(int)) ||
            !backend->uploadBuffer(vfIdx, plan.vfaceIdx.data(), plan.vfaceIdx.size() * sizeof(int))) { cleanupScratch(); return false; }
        {
            struct { uint32_t vertCount, p0, p1, p2; } pcV{ static_cast<uint32_t>(finalV), 0u, 0u, 0u };
            ComputeBufferHandle bufs[4] = { faceN, vfOff, vfIdx, nrm };
            ComputeDispatch cmd; cmd.kernel = "cc_vertex_normals";
            cmd.groups.groups_x = (static_cast<uint32_t>(finalV) + 63u) / 64u;
            cmd.buffers = bufs; cmd.buffer_count = 4;
            cmd.constants = &pcV; cmd.constants_size = sizeof(pcV);
            if (!backend->dispatch(cmd)) { cleanupScratch(); return false; }
        }

        // --- Expand into the non-indexed combined BLAS-layout buffer (kept on device) ---
        ComputeBufferHandle triIdxBuf = track(makeBuf(plan.triIndices.size() * sizeof(uint32_t), "ccd_tri_idx"));
        ComputeBufferHandle triUVBuf  = track(makeBuf(plan.triUV.size() * sizeof(float) * 2,     "ccd_tri_uv"));
        ComputeBufferHandle triMatBuf = track(makeBuf(static_cast<size_t>(triCount) * sizeof(uint32_t), "ccd_tri_mat"));
        if (!triIdxBuf.valid() || !triUVBuf.valid() || !triMatBuf.valid()) { cleanupScratch(); return false; }
        {
            // triUV is Vec2 (two contiguous floats) and triMat is uint16_t -> widen to uint32.
            std::vector<float> uvFlat(plan.triUV.size() * 2);
            for (size_t i = 0; i < plan.triUV.size(); ++i) { uvFlat[i * 2] = plan.triUV[i].x; uvFlat[i * 2 + 1] = plan.triUV[i].y; }
            std::vector<uint32_t> matWide(static_cast<size_t>(triCount));
            for (int i = 0; i < triCount; ++i) matWide[i] = static_cast<uint32_t>(plan.triMat[i]);
            if (!backend->uploadBuffer(triIdxBuf, plan.triIndices.data(), plan.triIndices.size() * sizeof(uint32_t)) ||
                !backend->uploadBuffer(triUVBuf,  uvFlat.data(),  uvFlat.size()  * sizeof(float)) ||
                !backend->uploadBuffer(triMatBuf, matWide.data(), matWide.size() * sizeof(uint32_t))) { cleanupScratch(); return false; }
        }
        // Combined buffer = triCount * (9 pos + 9 norm + 6 uv + 1 mat) words * 4 bytes = triCount*100.
        const size_t combinedBytes = static_cast<size_t>(triCount) * 100;
        ComputeBufferHandle combined = makeBuf(combinedBytes, "ccd_blas_geom", /*accelInput=*/true);
        if (!combined.valid()) { cleanupScratch(); return false; }
        {
            struct { uint32_t triCount, p0, p1, p2; } pcE{ static_cast<uint32_t>(triCount), 0u, 0u, 0u };
            ComputeBufferHandle bufs[6] = { src, nrm, triIdxBuf, triUVBuf, triMatBuf, combined };
            ComputeDispatch cmd; cmd.kernel = "cc_expand_blas";
            cmd.groups.groups_x = (static_cast<uint32_t>(triCount) + 63u) / 64u;
            cmd.buffers = bufs; cmd.buffer_count = 6;
            cmd.constants = &pcE; cmd.constants_size = sizeof(pcE);
            if (!backend->dispatch(cmd)) { backend->destroyBuffer(combined); cleanupScratch(); return false; }
        }

        backend->synchronize();

        const uint64_t addr = backend->bufferDeviceAddress(combined);
        cleanupScratch();
        if (addr == 0) { backend->destroyBuffer(combined); return false; }

        out.bufferId      = combined.id;
        out.deviceAddress = addr;
        out.vertexCount   = static_cast<uint32_t>(triCount) * 3u;
        out.triCount      = static_cast<uint32_t>(triCount);
        return true;
    }

    void releaseCCDeviceGeometry(CCDeviceGeometry& geo) {
        if (geo.bufferId == 0) { geo = {}; return; }
        using namespace RayTrophiSim;
        std::lock_guard<std::recursive_mutex> computeLock(sharedMeshComputeMutex());
        ISimulationComputeBackend* backend = acquireSharedMeshComputeBackend();
        if (backend) {
            ComputeBufferHandle h; h.id = geo.bufferId; h.backend = ComputeBackendType::VulkanCompute;
            backend->destroyBuffer(h);
        }
        geo = {};
    }

    std::vector<std::shared_ptr<Triangle>> catmullClarkSubDStencil(
            const std::vector<std::shared_ptr<Triangle>>& inputMesh,
            int levels,
            const EdgeCreaseFn& creaseLookup,
            std::shared_ptr<TriangleMesh>* outMesh) {
        if (inputMesh.empty() || levels <= 0) return inputMesh;

        std::shared_ptr<Transform> sharedTransform;
        std::string nodeName;
        for (const auto& tri : inputMesh) {
            if (tri) { sharedTransform = tri->getTransformHandle(); nodeName = tri->getNodeName(); break; }
        }

        // [CCPERF] TEMP stage timing — locate where the 2M+ subdivide/apply seconds go
        // (plan stencil build vs position/normal eval vs Triangle materialization).
        const auto _ccT0 = std::chrono::high_resolution_clock::now();
        CCSubdivPlan plan = buildCCSubdivPlan(inputMesh, levels, creaseLookup);
        if (!plan.valid()) return inputMesh;
        const auto _ccT1 = std::chrono::high_resolution_clock::now();

        std::vector<Vec3> pos, nrm;
        evaluateCCPositions(plan, plan.cageP0, pos, nrm);
        if (pos.empty()) return inputMesh;
        const auto _ccT2 = std::chrono::high_resolution_clock::now();

        const int nTris = static_cast<int>(plan.triMat.size());
        std::vector<std::shared_ptr<Triangle>> out;
        
        // Faz 1 (B): emit lightweight facade triangles over a shared SoA mesh instead of fat
        // standalone Triangles. One-line revert to convertFromRawArrays if a regression shows.
        TriangleProxyConverter::convertFromRawArraysToMesh(
            pos,
            nrm,
            plan.triIndices,
            plan.triUV,
            plan.triMat,
            plan.triFace,
            sharedTransform,
            nodeName,
            out,
            outMesh);   // flip: when set, emits one TriangleMesh and skips facade materialization

        {
            const auto _ccT3 = std::chrono::high_resolution_clock::now();
            auto ms = [](const std::chrono::high_resolution_clock::time_point& a,
                         const std::chrono::high_resolution_clock::time_point& b) {
                return std::chrono::duration<double, std::milli>(b - a).count();
            };
            char buf[192];
            std::snprintf(buf, sizeof(buf),
                "[CCPERF] lvl=%d tris=%d | buildPlan=%.0f ms  eval=%.0f ms  materialize=%.0f ms  total=%.0f ms",
                levels, nTris, ms(_ccT0, _ccT1), ms(_ccT1, _ccT2), ms(_ccT2, _ccT3), ms(_ccT0, _ccT3));
            SCENE_LOG_INFO(buf);
        }
        
        return out;
    }

    void ModifierData::serialize(nlohmann::json& j) const {
        j["name"] = name;
        j["type"] = static_cast<int>(type);
        j["enabled"] = enabled;
        j["levels"] = levels;
        j["renderLevels"] = renderLevels;
        j["smoothAngle"] = smoothAngle;
        j["bevelWidth"] = bevelWidth;
        j["bevelAngle"] = bevelAngle;
    }

    void ModifierData::deserialize(const nlohmann::json& j) {
        if (j.contains("name")) name = j["name"].get<std::string>();
        if (j.contains("type")) type = static_cast<ModifierType>(j["type"].get<int>());
        if (j.contains("enabled")) enabled = j["enabled"].get<bool>();
        if (j.contains("levels")) levels = j["levels"].get<int>();
        // Backward compat: pre-Viewport/Render files default render level to the viewport level.
        renderLevels = j.contains("renderLevels") ? j["renderLevels"].get<int>() : levels;
        if (j.contains("smoothAngle")) smoothAngle = j["smoothAngle"].get<float>();
        if (j.contains("bevelWidth")) bevelWidth = j["bevelWidth"].get<float>();
        if (j.contains("bevelAngle")) bevelAngle = j["bevelAngle"].get<float>();
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

    std::vector<std::shared_ptr<Triangle>> ModifierStack::evaluate(const std::vector<std::shared_ptr<Triangle>>& baseMesh, bool forRender, std::shared_ptr<TriangleMesh>* outMesh) const {
        // Flat/proxy migration flip: a single enabled Catmull-Clark modifier can emit ONE shared
        // TriangleMesh (no per-face facades) — the materialize+soup win. Restricted to the
        // single-CC case (the common "tek subdivide" stack) so multi-modifier chaining, which
        // consumes facades between stages, keeps the proven facade path.
        if (outMesh) {
            *outMesh = nullptr;
            int enabledGeometryCount = 0;
            bool hasEnabledWater = false;
            const ModifierData* only = nullptr;
            for (const auto& m : modifiers) {
                if (!m.enabled) continue;
                if (m.type == ModifierType::WaterSurface) {
                    hasEnabledWater = true;
                    continue;
                }
                ++enabledGeometryCount;
                only = &m;
            }
            if (enabledGeometryCount == 0 && hasEnabledWater && !baseMesh.empty()) {
                if (baseMesh.front()->parentMesh && baseMesh.front()->parentMesh->geometry) {
                    const auto& source = baseMesh.front()->parentMesh;
                    auto clone = std::make_shared<TriangleMesh>();
                    clone->geometry = std::make_shared<DNA::GeometryDetail>(*source->geometry);
                    clone->nodeName = source->nodeName;
                    clone->transform = source->transform;
                    clone->terrain_id = source->terrain_id;
                    clone->pointiness = source->pointiness;
                    clone->material_attribs = source->material_attribs;
                    *outMesh = std::move(clone);
                } else {
                    *outMesh = facadesToFlatMesh(baseMesh);
                }
                if (*outMesh) return {};
            }
            if (enabledGeometryCount == 1 && only && only->type == ModifierType::CatmullClark) {
                const int lvl = forRender ? only->renderLevels : only->levels;
                EdgeCreaseFn creaseFn;
                if (!edgeCreases.empty()) {
                    creaseFn = [this](const Vec3& a, const Vec3& b) { return getEdgeCrease(a, b); };
                }
                catmullClarkSubDStencil(baseMesh, lvl, creaseFn, outMesh);
                return {}; // facades intentionally not produced; caller uses *outMesh
            }
            // Simple (linear / FlatSubdivision) — the "Simple" subdivision mode — also emits ONE
            // flat TriangleMesh (no facades), matching the Catmull-Clark fast path above.
            if (enabledGeometryCount == 1 && only && only->type == ModifierType::FlatSubdivision) {
                const int lvl = forRender ? only->renderLevels : only->levels;
                *outMesh = subdivideSimpleToFlatMesh(baseMesh, lvl);
                if (*outMesh) {
                    return {}; // emitted one flat TriangleMesh; caller uses *outMesh
                }
                // else: conversion unavailable — fall through to the facade path below
            }
        }

        std::vector<std::shared_ptr<Triangle>> currentMesh = baseMesh;
        bool hasEnabledWater = false;
        bool hasEnabledSimpleSubdivision = false;

        // Apply modifiers sequentially
        for (const auto& mod : modifiers) {
            if (!mod.enabled) continue;

            if (mod.type == ModifierType::WaterSurface) {
                hasEnabledWater = true;
                continue;
            }

            // Blender-style Viewport vs Render subdivision level.
            const int lvl = forRender ? mod.renderLevels : mod.levels;

            if (mod.type == ModifierType::FlatSubdivision) {
                hasEnabledSimpleSubdivision = true;
                currentMesh = SubdivideSubD(currentMesh, lvl);
            } else if (mod.type == ModifierType::SmoothSubdivision) {
                // LIVE "Smooth Subdivision" uses the lightweight linear+Laplacian path so
                // that interactive cage editing (gizmo move/scale of vertices/edges/faces)
                // stays stable. True Catmull-Clark is the separate CatmullClark modifier.
                currentMesh = SmoothSubD(currentMesh, lvl, mod.smoothAngle);
            } else if (mod.type == ModifierType::CatmullClark) {
                // LIVE true Catmull-Clark via the stencil engine. The cage stays the
                // editable base; a position-only edit only re-applies the stencils (the
                // interactive-drag preview clamps to level 1, so per-frame eval is cheap).
                EdgeCreaseFn creaseFn;
                if (!edgeCreases.empty()) {
                    creaseFn = [this](const Vec3& a, const Vec3& b) { return getEdgeCrease(a, b); };
                }
                currentMesh = catmullClarkSubDStencil(currentMesh, lvl, creaseFn);
            }
            // WaterSurface deliberately does not alter topology here. The UI/runtime
            // binds WaterManager to this evaluated output and applies its animated
            // position-only deformation after the subdivision stack has completed.
            // ModifierType::Bevel deliberately has NO branch: the stack modifier was
            // retired same-day (2026-07-04, compounded over its own output on every
            // evaluate) — a leftover entry in an old save is inert. Bevel lives in the
            // Geo-DAG BevelNode / Edit Mode instead.
        }

        // Water needs one stable indexed SoA mesh for per-frame BLAS refit, and
        // Simple subdivision is itself a flat-output modifier. Materialize any
        // facade-producing intermediate only once after the stack has evaluated.
        if (outMesh && (hasEnabledWater || hasEnabledSimpleSubdivision) && !currentMesh.empty()) {
            *outMesh = currentMesh.front()->parentMesh
                ? currentMesh.front()->parentMesh
                : facadesToFlatMesh(currentMesh);
            if (*outMesh) return {};
        }

        // [PERF FIX] The convertToProxyMesh call is a no-op because its return value (TriangleMesh)
        // is discarded, yet it performs an extremely expensive single-threaded vertex weld over
        // millions of triangles. Removing it saves 6.4 seconds and gigabytes of RAM in ModifierStack::evaluate.
        /*
        if (!modifiers.empty()) {
            TriangleProxyConverter::convertToProxyMesh(currentMesh);
        }
        */

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

} // namespace MeshModifiers

// ============================================================================
// GEO-DAG (Faz 8a) NODE SELF-REGISTRATION
// ============================================================================
// GeometryNodesV2 nodes live in their own namespace (they're generic — not
// tied to MeshModifiers specifically), but their only current implementation
// depends on this file's catmullClarkSubDStencil, so this is a convenient,
// already-compiled home for their NodeRegistry registration without adding a
// new .cpp / touching the MSBuild project file (this project's .vcxproj lists
// .cpp files explicitly; headers don't need an entry, but a new .cpp would).
// See NodeSystem/NodeRegistry.h and TerrainNodesV2.cpp's identical pattern.

// ============================================================================
// SCATTER INSTANCES NODE — compute() (Faz 8b)
// ============================================================================
// Lives here (not in GeometryNodesV2.h) because it needs InstanceManager, which
// the header deliberately doesn't include (it's included by scene_data.h).
#include "InstanceManager.h"
#include <random>

NodeSystem::PinValue GeometryNodesV2::ScatterInstancesNode::compute(int /*outputIndex*/, NodeSystem::EvaluationContext& ctx) {
    NodeSystem::GeometryValue inMesh = getGeometryInput(0, ctx);
    if (!inMesh || !inMesh->geometry || inMesh->geometry->indices.empty()) {
        ctx.addError(id, "Scatter Instances: no input geometry");
        return NodeSystem::PinValue{};
    }
    // The surface itself is untouched — every failure below reports an error but
    // still passes the geometry through, so a scatter problem never destroys the
    // mesh result of the chain (errors are surfaced in the viewport regardless).
    auto passThrough = [&]() { return NodeSystem::GeometryValue(inMesh); };

    auto* gctx = getGeometryContext(ctx);
    if (!gctx) {
        ctx.addError(id, "Scatter Instances: no geometry context");
        return passThrough();
    }
    if (sourceObject[0] == '\0') {
        ctx.addError(id, "Scatter Instances: no source object selected");
        return passThrough();
    }
    // Gather ALL of the source's facades — a multi-material import owns several
    // sibling TriangleMesh entries under one nodeName; resolving just one of them
    // (the old resolveObjectMesh path) scattered instances with missing triangles
    // and a single wrong material.
    std::vector<std::shared_ptr<Triangle>> srcTris;
    if (gctx->gatherObjectFacades) {
        srcTris = gctx->gatherObjectFacades(sourceObject);
    } else if (gctx->resolveObjectMesh) {
        // Fallback for hosts that only wire the single-mesh resolver.
        if (std::shared_ptr<TriangleMesh> srcMesh = gctx->resolveObjectMesh(sourceObject)) {
            if (srcMesh->geometry && !srcMesh->geometry->indices.empty()) {
                const size_t nTris = srcMesh->num_triangles();
                srcTris.reserve(nTris);
                for (size_t f = 0; f < nTris; ++f) {
                    srcTris.push_back(std::make_shared<Triangle>(srcMesh, static_cast<uint32_t>(f)));
                }
            }
        }
    }
    if (srcTris.empty()) {
        ctx.addError(id, std::string("Scatter Instances: source '") + sourceObject + "' not found or has no geometry");
        return passThrough();
    }

    // Stable group identity: generated once, then serialized with the node — so a
    // re-Evaluate (this session or after a project reload) REPLACES the group
    // instead of stacking a duplicate.
    if (groupName[0] == '\0') {
        snprintf(groupName, sizeof(groupName), "GeoDAG %s #%u", gctx->objectName.c_str(), id);
    }

    // ---- Sample the input surface (world space) -----------------------------
    const auto& geom = *inMesh->geometry;
    const Vec3* Po = geom.get_attribute_data<Vec3>("P_orig");
    const auto& idx = geom.indices;
    const size_t triCount = idx.size() / 3;
    if (!Po || triCount == 0) {
        ctx.addError(id, "Scatter Instances: input has no P_orig channel");
        return passThrough();
    }
    const Matrix4x4 fT = inMesh->transform ? inMesh->transform->getFinal() : Matrix4x4();

    const float* densityMaskAttr = nullptr;
    if (useDensityMask) {
        densityMaskAttr = geom.get_attribute_data<float>(densityMaskAttribute);
        if (!densityMaskAttr) {
            lastDensityMaskState = 0;
            ctx.addError(id, std::string("Scatter Instances: density mask attribute '") + densityMaskAttribute
                + "' not found — add a Mask node upstream (after Subdivide)");
            return passThrough();
        }
        lastDensityMaskState = 1;
    } else {
        lastDensityMaskState = -1;
    }
    const float* scaleMaskAttr = nullptr;
    if (useScaleMask) {
        scaleMaskAttr = geom.get_attribute_data<float>(scaleMaskAttribute);
        if (!scaleMaskAttr) {
            lastScaleMaskState = 0;
            ctx.addError(id, std::string("Scatter Instances: scale mask attribute '") + scaleMaskAttribute
                + "' not found — add a Mask node upstream (after Subdivide)");
            return passThrough();
        }
        lastScaleMaskState = 1;
    } else {
        lastScaleMaskState = -1;
    }

    // Area-weighted cumulative distribution over WORLD-space triangle areas.
    std::vector<double> cdf(triCount, 0.0);
    double totalArea = 0.0;
    std::vector<Vec3> worldV(geom.get_vertex_count());
    for (size_t v = 0; v < worldV.size(); ++v) worldV[v] = fT.transform_point(Po[v]);
    for (size_t t = 0; t < triCount; ++t) {
        const Vec3& A = worldV[idx[t * 3 + 0]];
        const Vec3& B = worldV[idx[t * 3 + 1]];
        const Vec3& C = worldV[idx[t * 3 + 2]];
        const Vec3 cr = Vec3::cross(B - A, C - A);
        totalArea += 0.5 * std::sqrt(static_cast<double>(cr.x) * cr.x + static_cast<double>(cr.y) * cr.y + static_cast<double>(cr.z) * cr.z);
        cdf[t] = totalArea;
    }
    if (totalArea <= 0.0) {
        ctx.addError(id, "Scatter Instances: input surface has zero area");
        return passThrough();
    }

    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
    std::uniform_real_distribution<double> distArea(0.0, totalArea);

    struct Sample { Vec3 pos; Vec3 normal; float scaleMask; };
    std::vector<Sample> samples;
    samples.reserve(static_cast<size_t>((std::max)(count, 0)));

    // Min-distance rejection: spatial-hash grid (cell = minDistance, same scheme as
    // WeldNode) instead of an O(N^2) scan against every placed sample — matters once
    // Count reaches the thousands. Cell membership only needs the CANDIDATE's own
    // cell + 26 neighbors checked, since minDistance == cell size.
    const bool useMinDist = minDistance > 0.0f;
    const float invCell = useMinDist ? 1.0f / minDistance : 0.0f;
    const float minDist2 = minDistance * minDistance;
    auto cellKey = [](int x, int y, int z) -> uint64_t {
        return (static_cast<uint64_t>(static_cast<uint32_t>(x) & 0x1FFFFF) << 42)
             | (static_cast<uint64_t>(static_cast<uint32_t>(y) & 0x1FFFFF) << 21)
             |  static_cast<uint64_t>(static_cast<uint32_t>(z) & 0x1FFFFF);
    };
    std::unordered_map<uint64_t, std::vector<uint32_t>> placedCells;

    const int maxAttempts = (std::max)(count, 0) * 20;
    for (int attempt = 0; attempt < maxAttempts && static_cast<int>(samples.size()) < count; ++attempt) {
        const double r = distArea(rng);
        const size_t t = static_cast<size_t>(std::lower_bound(cdf.begin(), cdf.end(), r) - cdf.begin());
        if (t >= triCount) continue;
        const uint32_t ia = idx[t * 3 + 0], ib = idx[t * 3 + 1], ic = idx[t * 3 + 2];

        // Uniform barycentric sample (sqrt trick).
        const float su = std::sqrt(dist01(rng));
        const float r2 = dist01(rng);
        const float b0 = 1.0f - su, b1 = su * (1.0f - r2), b2 = su * r2;

        if (densityMaskAttr) {
            const float dval = b0 * densityMaskAttr[ia] + b1 * densityMaskAttr[ib] + b2 * densityMaskAttr[ic];
            if (dist01(rng) > dval) continue;   // rejection sampling — mask = density
        }
        float sval = 1.0f;
        if (scaleMaskAttr) {
            sval = b0 * scaleMaskAttr[ia] + b1 * scaleMaskAttr[ib] + b2 * scaleMaskAttr[ic];
        }

        const Vec3& A = worldV[ia];
        const Vec3& B = worldV[ib];
        const Vec3& C = worldV[ic];
        const Vec3 pos = A * b0 + B * b1 + C * b2;

        if (useMinDist) {
            const int cx = static_cast<int>(std::floor(pos.x * invCell));
            const int cy = static_cast<int>(std::floor(pos.y * invCell));
            const int cz = static_cast<int>(std::floor(pos.z * invCell));
            bool tooClose = false;
            for (int dx = -1; dx <= 1 && !tooClose; ++dx)
            for (int dy = -1; dy <= 1 && !tooClose; ++dy)
            for (int dz = -1; dz <= 1 && !tooClose; ++dz) {
                auto cit = placedCells.find(cellKey(cx + dx, cy + dy, cz + dz));
                if (cit == placedCells.end()) continue;
                for (const uint32_t si : cit->second) {
                    const Vec3 diff = pos - samples[si].pos;
                    if (diff.x * diff.x + diff.y * diff.y + diff.z * diff.z < minDist2) {
                        tooClose = true;
                        break;
                    }
                }
            }
            if (tooClose) continue;
            placedCells[cellKey(cx, cy, cz)].push_back(static_cast<uint32_t>(samples.size()));
        }

        Sample s;
        s.pos = pos;
        const Vec3 fn = Vec3::cross(B - A, C - A);
        const float len = std::sqrt(fn.x * fn.x + fn.y * fn.y + fn.z * fn.z);
        s.normal = (len > 1e-12f) ? Vec3(fn.x / len, fn.y / len, fn.z / len) : Vec3(0.0f, 1.0f, 0.0f);
        s.scaleMask = sval;
        samples.push_back(s);
    }

    // ---- Create/replace the InstanceGroup -----------------------------------
    auto& im = InstanceManager::getInstance();
    if (InstanceGroup* existing = im.findGroupByName(groupName)) {
        im.deleteGroup(existing->id);
    }
    if (samples.empty() || count <= 0) {
        // Nothing to place (mask rejected everything / count 0) — group removed,
        // surface passes through untouched.
        lastPlaced = 0;
        gctx->instancesDirty = true;
        return passThrough();
    }

    const int gid = im.createGroup(groupName, sourceObject, {});
    InstanceGroup* group = im.getGroup(gid);
    if (!group) {
        ctx.addError(id, "Scatter Instances: failed to create instance group");
        return passThrough();
    }

    // Modern multi-source path (same contract ParticleRenderBridge uses: name +
    // facades + computeCenter; BVH/centered triangles are built on demand).
    {
        ScatterSource src;
        src.name = sourceObject;
        src.triangles = std::move(srcTris);
        src.weight = 1.0f;
        src.computeCenter();
        group->sources.push_back(std::move(src));
    }

    group->target_type = InstanceGroup::TargetType::MESH;
    group->target_node_name = gctx->objectName;
    auto& bs = group->brush_settings;
    bs.use_global_settings = true;   // sources[0].settings untouched — globals below win
    bs.seed = seed;
    bs.target_count = count;
    bs.scale_min = scaleMin;
    bs.scale_max = scaleMax;
    bs.rotation_random_y = yawRandom;
    bs.rotation_random_xz = tiltRandom;
    bs.align_to_normal = alignToNormal;
    bs.normal_influence = normalInfluence;
    bs.y_offset_min = 0.0f;
    bs.y_offset_max = 0.0f;

    for (const Sample& s : samples) {
        InstanceTransform t = group->generateRandomTransform(s.pos, s.normal);
        if (scaleMaskAttr && scaleMaskInfluence > 0.0f) {
            // Independent scale role: instances at low-scale-mask spots come out smaller
            // (small plants near a patch edge, full-size in the core) regardless of what
            // (if anything) is gating density. Influence 0 disables the effect.
            const float f = 1.0f - scaleMaskInfluence * (1.0f - s.scaleMask);
            t.scale = t.scale * f;
        }
        group->addInstance(t);
    }
    group->gpu_dirty = true;

    lastPlaced = static_cast<int>(samples.size());
    gctx->instancesDirty = true;
    return passThrough();
}

// ============================================================================
// REMESH (VOXEL) NODE — compute() (Faz 8a)
// ============================================================================
// Blender-style voxel remesh: input surface → narrow-band SDF → marching cubes.
// Lives here (not in GeometryNodesV2.h) because the marching-cubes tables are
// ~4KB of static data that shouldn't ride into every TU including the header —
// same declared-in-header/defined-here pattern as ScatterInstancesNode above.
//
// SDF construction is Bridson's classic makelevelset3 scheme:
//   1. UNSIGNED distance: every triangle stamps the cells inside its inflated
//      AABB with point-to-triangle distance (narrow band only).
//   2. SIGN: per-(y,z) column ray parity — each triangle records where it
//      crosses each column's X ray; a cell behind an odd number of crossings
//      is inside. Robust for closed surfaces; open boundaries have no defined
//      inside (documented node limitation, same as Blender's voxel remesher).
//   3. Marching cubes (Paul Bourke tables — FoamSurface.cpp keeps its own
//      deliberately self-contained copy of the same public-domain tables) with
//      an edge-keyed vertex map, so the output comes out fully WELDED — no
//      post-weld pass needed, and Laplacian smoothing has real adjacency.

namespace {
namespace remeshmc {

static const int kEdgeTable[256] = {
0x0,0x109,0x203,0x30a,0x406,0x50f,0x605,0x70c,0x80c,0x905,0xa0f,0xb06,0xc0a,0xd03,0xe09,0xf00,
0x190,0x99,0x393,0x29a,0x596,0x49f,0x795,0x69c,0x99c,0x895,0xb9f,0xa96,0xd9a,0xc93,0xf99,0xe90,
0x230,0x339,0x33,0x13a,0x636,0x73f,0x435,0x53c,0xa3c,0xb35,0x83f,0x936,0xe3a,0xf33,0xc39,0xd30,
0x3a0,0x2a9,0x1a3,0xaa,0x7a6,0x6af,0x5a5,0x4ac,0xbac,0xaa5,0x9af,0x8a6,0xfaa,0xea3,0xda9,0xca0,
0x460,0x569,0x663,0x76a,0x66,0x16f,0x265,0x36c,0xc6c,0xd65,0xe6f,0xf66,0x86a,0x963,0xa69,0xb60,
0x5f0,0x4f9,0x7f3,0x6fa,0x1f6,0xff,0x3f5,0x2fc,0xdfc,0xcf5,0xfff,0xef6,0x9fa,0x8f3,0xbf9,0xaf0,
0x650,0x759,0x453,0x55a,0x256,0x35f,0x55,0x15c,0xe5c,0xf55,0xc5f,0xd56,0xa5a,0xb53,0x859,0x950,
0x7c0,0x6c9,0x5c3,0x4ca,0x3c6,0x2cf,0x1c5,0xcc,0xfcc,0xec5,0xdcf,0xcc6,0xbca,0xac3,0x9c9,0x8c0,
0x8c0,0x9c9,0xac3,0xbca,0xcc6,0xdcf,0xec5,0xfcc,0xcc,0x1c5,0x2cf,0x3c6,0x4ca,0x5c3,0x6c9,0x7c0,
0x950,0x859,0xb53,0xa5a,0xd56,0xc5f,0xf55,0xe5c,0x15c,0x55,0x35f,0x256,0x55a,0x453,0x759,0x650,
0xaf0,0xbf9,0x8f3,0x9fa,0xef6,0xfff,0xcf5,0xdfc,0x2fc,0x3f5,0xff,0x1f6,0x6fa,0x7f3,0x4f9,0x5f0,
0xb60,0xa69,0x963,0x86a,0xf66,0xe6f,0xd65,0xc6c,0x36c,0x265,0x16f,0x66,0x76a,0x663,0x569,0x460,
0xca0,0xda9,0xea3,0xfaa,0x8a6,0x9af,0xaa5,0xbac,0x4ac,0x5a5,0x6af,0x7a6,0xaa,0x1a3,0x2a9,0x3a0,
0xd30,0xc39,0xf33,0xe3a,0x936,0x83f,0xb35,0xa3c,0x53c,0x435,0x73f,0x636,0x13a,0x33,0x339,0x230,
0xe90,0xf99,0xc93,0xd9a,0xa96,0xb9f,0x895,0x99c,0x69c,0x795,0x49f,0x596,0x29a,0x393,0x99,0x190,
0xf00,0xe09,0xd03,0xc0a,0xb06,0xa0f,0x905,0x80c,0x70c,0x605,0x50f,0x406,0x30a,0x203,0x109,0x0 };

static const int kTriTable[256][16] = {
{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,8,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,1,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{1,8,3,9,8,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{1,2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,8,3,1,2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{9,2,10,0,2,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{2,8,3,2,10,8,10,9,8,-1,-1,-1,-1,-1,-1,-1},
{3,11,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,11,2,8,11,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{1,9,0,2,3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{1,11,2,1,9,11,9,8,11,-1,-1,-1,-1,-1,-1,-1},
{3,10,1,11,10,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,10,1,0,8,10,8,11,10,-1,-1,-1,-1,-1,-1,-1},
{3,9,0,3,11,9,11,10,9,-1,-1,-1,-1,-1,-1,-1},
{9,8,10,10,8,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{4,7,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{4,3,0,7,3,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,1,9,8,4,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{4,1,9,4,7,1,7,3,1,-1,-1,-1,-1,-1,-1,-1},
{1,2,10,8,4,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{3,4,7,3,0,4,1,2,10,-1,-1,-1,-1,-1,-1,-1},
{9,2,10,9,0,2,8,4,7,-1,-1,-1,-1,-1,-1,-1},
{2,10,9,2,9,7,2,7,3,7,9,4,-1,-1,-1,-1},
{8,4,7,3,11,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{11,4,7,11,2,4,2,0,4,-1,-1,-1,-1,-1,-1,-1},
{9,0,1,8,4,7,2,3,11,-1,-1,-1,-1,-1,-1,-1},
{4,7,11,9,4,11,9,11,2,9,2,1,-1,-1,-1,-1},
{3,10,1,3,11,10,7,8,4,-1,-1,-1,-1,-1,-1,-1},
{1,11,10,1,4,11,1,0,4,7,11,4,-1,-1,-1,-1},
{4,7,8,9,0,11,9,11,10,11,0,3,-1,-1,-1,-1},
{4,7,11,4,11,9,9,11,10,-1,-1,-1,-1,-1,-1,-1},
{9,5,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{9,5,4,0,8,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,5,4,1,5,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{8,5,4,8,3,5,3,1,5,-1,-1,-1,-1,-1,-1,-1},
{1,2,10,9,5,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{3,0,8,1,2,10,4,9,5,-1,-1,-1,-1,-1,-1,-1},
{5,2,10,5,4,2,4,0,2,-1,-1,-1,-1,-1,-1,-1},
{2,10,5,3,2,5,3,5,4,3,4,8,-1,-1,-1,-1},
{9,5,4,2,3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,11,2,0,8,11,4,9,5,-1,-1,-1,-1,-1,-1,-1},
{0,5,4,0,1,5,2,3,11,-1,-1,-1,-1,-1,-1,-1},
{2,1,5,2,5,8,2,8,11,4,8,5,-1,-1,-1,-1},
{10,3,11,10,1,3,9,5,4,-1,-1,-1,-1,-1,-1,-1},
{4,9,5,0,8,1,8,10,1,8,11,10,-1,-1,-1,-1},
{5,4,0,5,0,11,5,11,10,11,0,3,-1,-1,-1,-1},
{5,4,8,5,8,10,10,8,11,-1,-1,-1,-1,-1,-1,-1},
{9,7,8,5,7,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{9,3,0,9,5,3,5,7,3,-1,-1,-1,-1,-1,-1,-1},
{0,7,8,0,1,7,1,5,7,-1,-1,-1,-1,-1,-1,-1},
{1,5,3,3,5,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{9,7,8,9,5,7,10,1,2,-1,-1,-1,-1,-1,-1,-1},
{10,1,2,9,5,0,5,3,0,5,7,3,-1,-1,-1,-1},
{8,0,2,8,2,5,8,5,7,10,5,2,-1,-1,-1,-1},
{2,10,5,2,5,3,3,5,7,-1,-1,-1,-1,-1,-1,-1},
{7,9,5,7,8,9,3,11,2,-1,-1,-1,-1,-1,-1,-1},
{9,5,7,9,7,2,9,2,0,2,7,11,-1,-1,-1,-1},
{2,3,11,0,1,8,1,7,8,1,5,7,-1,-1,-1,-1},
{11,2,1,11,1,7,7,1,5,-1,-1,-1,-1,-1,-1,-1},
{9,5,8,8,5,7,10,1,3,10,3,11,-1,-1,-1,-1},
{5,7,0,5,0,9,7,11,0,1,0,10,11,10,0,-1},
{11,10,0,11,0,3,10,5,0,8,0,7,5,7,0,-1},
{11,10,5,7,11,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{10,6,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,8,3,5,10,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{9,0,1,5,10,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{1,8,3,1,9,8,5,10,6,-1,-1,-1,-1,-1,-1,-1},
{1,6,5,2,6,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{1,6,5,1,2,6,3,0,8,-1,-1,-1,-1,-1,-1,-1},
{9,6,5,9,0,6,0,2,6,-1,-1,-1,-1,-1,-1,-1},
{5,9,8,5,8,2,5,2,6,3,2,8,-1,-1,-1,-1},
{2,3,11,10,6,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{11,0,8,11,2,0,10,6,5,-1,-1,-1,-1,-1,-1,-1},
{0,1,9,2,3,11,5,10,6,-1,-1,-1,-1,-1,-1,-1},
{5,10,6,1,9,2,9,11,2,9,8,11,-1,-1,-1,-1},
{6,3,11,6,5,3,5,1,3,-1,-1,-1,-1,-1,-1,-1},
{0,8,11,0,11,5,0,5,1,5,11,6,-1,-1,-1,-1},
{3,11,6,0,3,6,0,6,5,0,5,9,-1,-1,-1,-1},
{6,5,9,6,9,11,11,9,8,-1,-1,-1,-1,-1,-1,-1},
{5,10,6,4,7,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{4,3,0,4,7,3,6,5,10,-1,-1,-1,-1,-1,-1,-1},
{1,9,0,5,10,6,8,4,7,-1,-1,-1,-1,-1,-1,-1},
{10,6,5,1,9,7,1,7,3,7,9,4,-1,-1,-1,-1},
{6,1,2,6,5,1,4,7,8,-1,-1,-1,-1,-1,-1,-1},
{1,2,5,5,2,6,3,0,4,3,4,7,-1,-1,-1,-1},
{8,4,7,9,0,5,0,6,5,0,2,6,-1,-1,-1,-1},
{7,3,9,7,9,4,3,2,9,5,9,6,2,6,9,-1},
{3,11,2,7,8,4,10,6,5,-1,-1,-1,-1,-1,-1,-1},
{5,10,6,4,7,2,4,2,0,2,7,11,-1,-1,-1,-1},
{0,1,9,4,7,8,2,3,11,5,10,6,-1,-1,-1,-1},
{9,2,1,9,11,2,9,4,11,7,11,4,5,10,6,-1},
{8,4,7,3,11,5,3,5,1,5,11,6,-1,-1,-1,-1},
{5,1,11,5,11,6,1,0,11,7,11,4,0,4,11,-1},
{0,5,9,0,6,5,0,3,6,11,6,3,8,4,7,-1},
{6,5,9,6,9,11,4,7,9,7,11,9,-1,-1,-1,-1},
{10,4,9,6,4,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{4,10,6,4,9,10,0,8,3,-1,-1,-1,-1,-1,-1,-1},
{10,0,1,10,6,0,6,4,0,-1,-1,-1,-1,-1,-1,-1},
{8,3,1,8,1,6,8,6,4,6,1,10,-1,-1,-1,-1},
{1,4,9,1,2,4,2,6,4,-1,-1,-1,-1,-1,-1,-1},
{3,0,8,1,2,9,2,4,9,2,6,4,-1,-1,-1,-1},
{0,2,4,4,2,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{8,3,2,8,2,4,4,2,6,-1,-1,-1,-1,-1,-1,-1},
{10,4,9,10,6,4,11,2,3,-1,-1,-1,-1,-1,-1,-1},
{0,8,2,2,8,11,4,9,10,4,10,6,-1,-1,-1,-1},
{3,11,2,0,1,6,0,6,4,6,1,10,-1,-1,-1,-1},
{6,4,1,6,1,10,4,8,1,2,1,11,8,11,1,-1},
{9,6,4,9,3,6,9,1,3,11,6,3,-1,-1,-1,-1},
{8,11,1,8,1,0,11,6,1,9,1,4,6,4,1,-1},
{3,11,6,3,6,0,0,6,4,-1,-1,-1,-1,-1,-1,-1},
{6,4,8,11,6,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{7,10,6,7,8,10,8,9,10,-1,-1,-1,-1,-1,-1,-1},
{0,7,3,0,10,7,0,9,10,6,7,10,-1,-1,-1,-1},
{10,6,7,1,10,7,1,7,8,1,8,0,-1,-1,-1,-1},
{10,6,7,10,7,1,1,7,3,-1,-1,-1,-1,-1,-1,-1},
{1,2,6,1,6,8,1,8,9,8,6,7,-1,-1,-1,-1},
{2,6,9,2,9,1,6,7,9,0,9,3,7,3,9,-1},
{7,8,0,7,0,6,6,0,2,-1,-1,-1,-1,-1,-1,-1},
{7,3,2,6,7,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{2,3,11,10,6,8,10,8,9,8,6,7,-1,-1,-1,-1},
{2,0,7,2,7,11,0,9,7,6,7,10,9,10,7,-1},
{1,8,0,1,7,8,1,10,7,6,7,10,2,3,11,-1},
{11,2,1,11,1,7,10,6,1,6,7,1,-1,-1,-1,-1},
{8,9,6,8,6,7,9,1,6,11,6,3,1,3,6,-1},
{0,9,1,11,6,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{7,8,0,7,0,6,3,11,0,11,6,0,-1,-1,-1,-1},
{7,11,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{7,6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{3,0,8,11,7,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,1,9,11,7,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{8,1,9,8,3,1,11,7,6,-1,-1,-1,-1,-1,-1,-1},
{10,1,2,6,11,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{1,2,10,3,0,8,6,11,7,-1,-1,-1,-1,-1,-1,-1},
{2,9,0,2,10,9,6,11,7,-1,-1,-1,-1,-1,-1,-1},
{6,11,7,2,10,3,10,8,3,10,9,8,-1,-1,-1,-1},
{7,2,3,6,2,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{7,0,8,7,6,0,6,2,0,-1,-1,-1,-1,-1,-1,-1},
{2,7,6,2,3,7,0,1,9,-1,-1,-1,-1,-1,-1,-1},
{1,6,2,1,8,6,1,9,8,8,7,6,-1,-1,-1,-1},
{10,7,6,10,1,7,1,3,7,-1,-1,-1,-1,-1,-1,-1},
{10,7,6,1,7,10,1,8,7,1,0,8,-1,-1,-1,-1},
{0,3,7,0,7,10,0,10,9,6,10,7,-1,-1,-1,-1},
{7,6,10,7,10,8,8,10,9,-1,-1,-1,-1,-1,-1,-1},
{6,8,4,11,8,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{3,6,11,3,0,6,0,4,6,-1,-1,-1,-1,-1,-1,-1},
{8,6,11,8,4,6,9,0,1,-1,-1,-1,-1,-1,-1,-1},
{9,4,6,9,6,3,9,3,1,11,3,6,-1,-1,-1,-1},
{6,8,4,6,11,8,2,10,1,-1,-1,-1,-1,-1,-1,-1},
{1,2,10,3,0,11,0,6,11,0,4,6,-1,-1,-1,-1},
{4,11,8,4,6,11,0,2,9,2,10,9,-1,-1,-1,-1},
{10,9,3,10,3,2,9,4,3,11,3,6,4,6,3,-1},
{8,2,3,8,4,2,4,6,2,-1,-1,-1,-1,-1,-1,-1},
{0,4,2,4,6,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{1,9,0,2,3,4,2,4,6,4,3,8,-1,-1,-1,-1},
{1,9,4,1,4,2,2,4,6,-1,-1,-1,-1,-1,-1,-1},
{8,1,3,8,6,1,8,4,6,6,10,1,-1,-1,-1,-1},
{10,1,0,10,0,6,6,0,4,-1,-1,-1,-1,-1,-1,-1},
{4,6,3,4,3,8,6,10,3,0,3,9,10,9,3,-1},
{10,9,4,6,10,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{4,9,5,7,6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,8,3,4,9,5,11,7,6,-1,-1,-1,-1,-1,-1,-1},
{5,0,1,5,4,0,7,6,11,-1,-1,-1,-1,-1,-1,-1},
{11,7,6,8,3,4,3,5,4,3,1,5,-1,-1,-1,-1},
{9,5,4,10,1,2,7,6,11,-1,-1,-1,-1,-1,-1,-1},
{6,11,7,1,2,10,0,8,3,4,9,5,-1,-1,-1,-1},
{7,6,11,5,4,10,4,2,10,4,0,2,-1,-1,-1,-1},
{3,4,8,3,5,4,3,2,5,10,5,2,11,7,6,-1},
{7,2,3,7,6,2,5,4,9,-1,-1,-1,-1,-1,-1,-1},
{9,5,4,0,8,6,0,6,2,6,8,7,-1,-1,-1,-1},
{3,6,2,3,7,6,1,5,0,5,4,0,-1,-1,-1,-1},
{6,2,8,6,8,7,2,1,8,4,8,5,1,5,8,-1},
{9,5,4,10,1,6,1,7,6,1,3,7,-1,-1,-1,-1},
{1,6,10,1,7,6,1,0,7,8,7,0,9,5,4,-1},
{4,0,10,4,10,5,0,3,10,6,10,7,3,7,10,-1},
{7,6,10,7,10,8,5,4,10,4,8,10,-1,-1,-1,-1},
{6,9,5,6,11,9,11,8,9,-1,-1,-1,-1,-1,-1,-1},
{3,6,11,0,6,3,0,5,6,0,9,5,-1,-1,-1,-1},
{0,11,8,0,5,11,0,1,5,5,6,11,-1,-1,-1,-1},
{6,11,3,6,3,5,5,3,1,-1,-1,-1,-1,-1,-1,-1},
{1,2,10,9,5,11,9,11,8,11,5,6,-1,-1,-1,-1},
{0,11,3,0,6,11,0,9,6,5,6,9,1,2,10,-1},
{11,8,5,11,5,6,8,0,5,10,5,2,0,2,5,-1},
{6,11,3,6,3,5,2,10,3,10,5,3,-1,-1,-1,-1},
{5,8,9,5,2,8,5,6,2,3,8,2,-1,-1,-1,-1},
{9,5,6,9,6,0,0,6,2,-1,-1,-1,-1,-1,-1,-1},
{1,5,8,1,8,0,5,6,8,3,8,2,6,2,8,-1},
{1,5,6,2,1,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{1,3,6,1,6,10,3,8,6,5,6,9,8,9,6,-1},
{10,1,0,10,0,6,9,5,0,5,6,0,-1,-1,-1,-1},
{0,3,8,5,6,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{10,5,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{11,5,10,7,5,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{11,5,10,11,7,5,8,3,0,-1,-1,-1,-1,-1,-1,-1},
{5,11,7,5,10,11,1,9,0,-1,-1,-1,-1,-1,-1,-1},
{10,7,5,10,11,7,9,8,1,8,3,1,-1,-1,-1,-1},
{11,1,2,11,7,1,7,5,1,-1,-1,-1,-1,-1,-1,-1},
{0,8,3,1,2,7,1,7,5,7,2,11,-1,-1,-1,-1},
{9,7,5,9,2,7,9,0,2,2,11,7,-1,-1,-1,-1},
{7,5,2,7,2,11,5,9,2,3,2,8,9,8,2,-1},
{2,5,10,2,3,5,3,7,5,-1,-1,-1,-1,-1,-1,-1},
{8,2,0,8,5,2,8,7,5,10,2,5,-1,-1,-1,-1},
{9,0,1,5,10,3,5,3,7,3,10,2,-1,-1,-1,-1},
{9,8,2,9,2,1,8,7,2,10,2,5,7,5,2,-1},
{1,3,5,3,7,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,8,7,0,7,1,1,7,5,-1,-1,-1,-1,-1,-1,-1},
{9,0,3,9,3,5,5,3,7,-1,-1,-1,-1,-1,-1,-1},
{9,8,7,5,9,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{5,8,4,5,10,8,10,11,8,-1,-1,-1,-1,-1,-1,-1},
{5,0,4,5,11,0,5,10,11,11,3,0,-1,-1,-1,-1},
{0,1,9,8,4,10,8,10,11,10,4,5,-1,-1,-1,-1},
{10,11,4,10,4,5,11,3,4,9,4,1,3,1,4,-1},
{2,5,1,2,8,5,2,11,8,4,5,8,-1,-1,-1,-1},
{0,4,11,0,11,3,4,5,11,2,11,1,5,1,11,-1},
{0,2,5,0,5,9,2,11,5,4,5,8,11,8,5,-1},
{9,4,5,2,11,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{2,5,10,3,5,2,3,4,5,3,8,4,-1,-1,-1,-1},
{5,10,2,5,2,4,4,2,0,-1,-1,-1,-1,-1,-1,-1},
{3,10,2,3,5,10,3,8,5,4,5,8,0,1,9,-1},
{5,10,2,5,2,4,1,9,2,9,4,2,-1,-1,-1,-1},
{8,4,5,8,5,3,3,5,1,-1,-1,-1,-1,-1,-1,-1},
{0,4,5,1,0,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{8,4,5,8,5,3,9,0,5,0,3,5,-1,-1,-1,-1},
{9,4,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{4,11,7,4,9,11,9,10,11,-1,-1,-1,-1,-1,-1,-1},
{0,8,3,4,9,7,9,11,7,9,10,11,-1,-1,-1,-1},
{1,10,11,1,11,4,1,4,0,7,4,11,-1,-1,-1,-1},
{3,1,4,3,4,8,1,10,4,7,4,11,10,11,4,-1},
{4,11,7,9,11,4,9,2,11,9,1,2,-1,-1,-1,-1},
{9,7,4,9,11,7,9,1,11,2,11,1,0,8,3,-1},
{11,7,4,11,4,2,2,4,0,-1,-1,-1,-1,-1,-1,-1},
{11,7,4,11,4,2,8,3,4,3,2,4,-1,-1,-1,-1},
{2,9,10,2,7,9,2,3,7,7,4,9,-1,-1,-1,-1},
{9,10,7,9,7,4,10,2,7,8,7,0,2,0,7,-1},
{3,7,10,3,10,2,7,4,10,1,10,0,4,0,10,-1},
{1,10,2,8,7,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{4,9,1,4,1,7,7,1,3,-1,-1,-1,-1,-1,-1,-1},
{4,9,1,4,1,7,0,8,1,8,7,1,-1,-1,-1,-1},
{4,0,3,7,4,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{4,8,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{9,10,8,10,11,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{3,0,9,3,9,11,11,9,10,-1,-1,-1,-1,-1,-1,-1},
{0,1,10,0,10,8,8,10,11,-1,-1,-1,-1,-1,-1,-1},
{3,1,10,11,3,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{1,2,11,1,11,9,9,11,8,-1,-1,-1,-1,-1,-1,-1},
{3,0,9,3,9,11,1,2,9,2,11,9,-1,-1,-1,-1},
{0,2,11,8,0,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{3,2,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{2,3,8,2,8,10,10,8,9,-1,-1,-1,-1,-1,-1,-1},
{9,10,2,0,9,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{2,3,8,2,8,10,0,1,8,1,10,8,-1,-1,-1,-1},
{1,10,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{1,3,8,9,1,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,9,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,3,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1} };

// Corner offsets (Bourke ordering) and the two corners each of the 12 edges joins.
static const int kCorner[8][3] = {
    {0,0,0},{1,0,0},{1,0,1},{0,0,1},{0,1,0},{1,1,0},{1,1,1},{0,1,1} };
static const int kEdgeCorner[12][2] = {
    {0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},{0,4},{1,5},{2,6},{3,7} };

// Ericson's closest-point-on-triangle (same routine ParticleSimulation.cpp keeps
// as its own TU-local copy for collider voxelization — small enough to duplicate
// rather than promote a physics TU's internal helper into a shared header).
inline Vec3 closestPointOnTri(const Vec3& p, const Vec3& a, const Vec3& b, const Vec3& c) {
    const Vec3 ab = b - a, ac = c - a, ap = p - a;
    const float d1 = Vec3::dot(ab, ap), d2 = Vec3::dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a;
    const Vec3 bp = p - b;
    const float d3 = Vec3::dot(ab, bp), d4 = Vec3::dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b;
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) return a + ab * (d1 / (d1 - d3));
    const Vec3 cp = p - c;
    const float d5 = Vec3::dot(ab, cp), d6 = Vec3::dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c;
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) return a + ac * (d2 / (d2 - d6));
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + (c - b) * w;
    }
    const float denom = 1.0f / (va + vb + vc);
    return a + ab * (vb * denom) + ac * (vc * denom);
}

} // namespace remeshmc
} // namespace

NodeSystem::PinValue GeometryNodesV2::RemeshNode::compute(int /*outputIndex*/, NodeSystem::EvaluationContext& ctx) {
    using namespace remeshmc;

    NodeSystem::GeometryValue inMesh = getGeometryInput(0, ctx);
    if (!inMesh || !inMesh->geometry || inMesh->geometry->indices.empty()) {
        ctx.addError(id, "Remesh: no input geometry");
        return NodeSystem::PinValue{};
    }
    const auto& sg = *inMesh->geometry;
    const Vec3* Po = sg.get_attribute_data<Vec3>("P_orig");
    const size_t vc = sg.get_vertex_count();
    const auto& srcIdx = sg.indices;
    const size_t triCount = srcIdx.size() / 3;
    if (!Po || vc == 0 || triCount == 0) {
        ctx.addError(id, "Remesh: input has no P_orig channel");
        return NodeSystem::PinValue{};
    }
    const uint16_t* srcMat = sg.get_material_ids();

    // ---- Grid setup (LOCAL space — output shares the input's pivot) ----------
    Vec3 bmin = Po[0], bmax = Po[0];
    for (size_t v = 1; v < vc; ++v) {
        bmin.x = (std::min)(bmin.x, Po[v].x); bmax.x = (std::max)(bmax.x, Po[v].x);
        bmin.y = (std::min)(bmin.y, Po[v].y); bmax.y = (std::max)(bmax.y, Po[v].y);
        bmin.z = (std::min)(bmin.z, Po[v].z); bmax.z = (std::max)(bmax.z, Po[v].z);
    }
    const Vec3 ext = bmax - bmin;
    const float maxSide = (std::max)(ext.x, (std::max)(ext.y, ext.z));
    if (maxSide <= 1e-8f) {
        ctx.addError(id, "Remesh: degenerate input bounds");
        return NodeSystem::PinValue{};
    }
    float h = relativeSize ? (std::max)(voxelSize, 0.002f) * maxSide : voxelSize;
    if (h <= 1e-8f) {
        ctx.addError(id, "Remesh: voxel size must be > 0");
        return NodeSystem::PinValue{};
    }

    // Memory guard: dist+closestTri are 8 bytes/cell — cap the grid and coarsen
    // the voxel size to fit instead of silently allocating gigabytes. 20M cells
    // (~270^3) is ~160MB peak, released before the mesh is materialized.
    const int pad = 3;
    int nx = 0, ny = 0, nz = 0;
    auto computeDims = [&]() {
        nx = static_cast<int>(std::ceil(ext.x / h)) + 2 * pad + 1;
        ny = static_cast<int>(std::ceil(ext.y / h)) + 2 * pad + 1;
        nz = static_cast<int>(std::ceil(ext.z / h)) + 2 * pad + 1;
    };
    computeDims();
    const size_t kMaxCells = 20'000'000;
    while (static_cast<size_t>(nx) * ny * nz > kMaxCells) {
        h *= 1.26f;   // ~2x cell count reduction per step
        computeDims();
    }
    const size_t total = static_cast<size_t>(nx) * ny * nz;
    const Vec3 origin = bmin - Vec3(pad * h, pad * h, pad * h);
    lastGridX = nx; lastGridY = ny; lastGridZ = nz;

    auto cellIndex = [nx, ny](int i, int j, int k) -> size_t {
        return static_cast<size_t>(i) + static_cast<size_t>(j) * nx
             + static_cast<size_t>(k) * nx * static_cast<size_t>(ny);
    };
    auto cellCenter = [&](int i, int j, int k) -> Vec3 {
        return origin + Vec3((i + 0.5f) * h, (j + 0.5f) * h, (k + 0.5f) * h);
    };

    // ---- 1) Unsigned narrow-band distance (squared, sqrt'ed at the end) ------
    const float band = 2.5f * h;
    std::vector<float> dist2(total, band * band);
    std::vector<int32_t> closestTri;
    if (transferMaterial) closestTri.assign(total, -1);

    // Cell-center c >= lo  <=>  i >= (lo - origin)/h - 0.5 ; symmetric for hi.
    auto rangeLo = [&](float lo, float o) { return (std::max)(0, static_cast<int>(std::floor((lo - o) / h - 0.5f))); };
    auto rangeHi = [&](float hi, float o, int n) { return (std::min)(n - 1, static_cast<int>(std::ceil((hi - o) / h - 0.5f))); };

    for (size_t t = 0; t < triCount; ++t) {
        const Vec3& A = Po[srcIdx[t * 3 + 0]];
        const Vec3& B = Po[srcIdx[t * 3 + 1]];
        const Vec3& C = Po[srcIdx[t * 3 + 2]];
        const float lox = (std::min)(A.x, (std::min)(B.x, C.x)) - band;
        const float hix = (std::max)(A.x, (std::max)(B.x, C.x)) + band;
        const float loy = (std::min)(A.y, (std::min)(B.y, C.y)) - band;
        const float hiy = (std::max)(A.y, (std::max)(B.y, C.y)) + band;
        const float loz = (std::min)(A.z, (std::min)(B.z, C.z)) - band;
        const float hiz = (std::max)(A.z, (std::max)(B.z, C.z)) + band;
        const int i0 = rangeLo(lox, origin.x), i1 = rangeHi(hix, origin.x, nx);
        const int j0 = rangeLo(loy, origin.y), j1 = rangeHi(hiy, origin.y, ny);
        const int k0 = rangeLo(loz, origin.z), k1 = rangeHi(hiz, origin.z, nz);
        for (int k = k0; k <= k1; ++k)
        for (int j = j0; j <= j1; ++j)
        for (int i = i0; i <= i1; ++i) {
            const Vec3 p = cellCenter(i, j, k);
            const Vec3 q = closestPointOnTri(p, A, B, C);
            const Vec3 d = p - q;
            const float dd = d.x * d.x + d.y * d.y + d.z * d.z;
            const size_t ci = cellIndex(i, j, k);
            if (dd < dist2[ci]) {
                dist2[ci] = dd;
                if (transferMaterial) closestTri[ci] = static_cast<int32_t>(t);
            }
        }
    }

    // ---- 2) Sign via per-(y,z)-column X-ray parity (makelevelset3) -----------
    // Column sample point is nudged off the exact cell-center lattice so a ray
    // grazing a shared triangle edge/vertex doesn't get counted by both (or
    // neither) incident triangle — the classic parity tie-break, bought with a
    // sub-1e-4-voxel bias instead of exact predicates.
    const float jog = 1.3e-4f * h;
    std::vector<std::vector<float>> columns(static_cast<size_t>(ny) * nz);
    for (size_t t = 0; t < triCount; ++t) {
        const Vec3& A = Po[srcIdx[t * 3 + 0]];
        const Vec3& B = Po[srcIdx[t * 3 + 1]];
        const Vec3& C = Po[srcIdx[t * 3 + 2]];
        const float denom = (B.y - A.y) * (C.z - A.z) - (B.z - A.z) * (C.y - A.y);
        if (std::fabs(denom) < 1e-14f) continue;   // edge-on to X; a closed mesh's other faces cover it
        const float loy = (std::min)(A.y, (std::min)(B.y, C.y));
        const float hiy = (std::max)(A.y, (std::max)(B.y, C.y));
        const float loz = (std::min)(A.z, (std::min)(B.z, C.z));
        const float hiz = (std::max)(A.z, (std::max)(B.z, C.z));
        const int j0 = rangeLo(loy, origin.y), j1 = rangeHi(hiy, origin.y, ny);
        const int k0 = rangeLo(loz, origin.z), k1 = rangeHi(hiz, origin.z, nz);
        const float inv = 1.0f / denom;
        for (int k = k0; k <= k1; ++k)
        for (int j = j0; j <= j1; ++j) {
            const float cy = origin.y + (j + 0.5f) * h + jog;
            const float cz = origin.z + (k + 0.5f) * h + jog;
            const float beta  = ((cy - A.y) * (C.z - A.z) - (cz - A.z) * (C.y - A.y)) * inv;
            const float gamma = ((B.y - A.y) * (cz - A.z) - (B.z - A.z) * (cy - A.y)) * inv;
            const float alpha = 1.0f - beta - gamma;
            if (alpha < 0.0f || beta < 0.0f || gamma < 0.0f) continue;
            columns[static_cast<size_t>(j) + static_cast<size_t>(k) * ny]
                .push_back(alpha * A.x + beta * B.x + gamma * C.x);
        }
    }
    std::vector<float> sdf(total);
    for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j) {
        auto& xs = columns[static_cast<size_t>(j) + static_cast<size_t>(k) * ny];
        std::sort(xs.begin(), xs.end());
        size_t next = 0;
        bool inside = false;
        for (int i = 0; i < nx; ++i) {
            const float cx = origin.x + (i + 0.5f) * h;
            while (next < xs.size() && xs[next] < cx) { inside = !inside; ++next; }
            const size_t ci = cellIndex(i, j, k);
            const float d = std::sqrt(dist2[ci]);
            sdf[ci] = inside ? -d : d;
        }
        xs.clear();
        xs.shrink_to_fit();
    }
    dist2.clear();
    dist2.shrink_to_fit();
    columns.clear();
    columns.shrink_to_fit();

    // ---- 3) Marching cubes with an edge-keyed vertex map (welded output) -----
    // An MC vertex lives on a lattice edge shared by up to 4 cubes; keying it by
    // (lower corner cell, axis) makes every cube reuse the same index — the
    // output is born welded, no coincident-position pass needed.
    std::vector<Vec3> outP;
    std::vector<uint32_t> outIdx;
    std::unordered_map<uint64_t, uint32_t> edgeVerts;
    outP.reserve(4096);
    outIdx.reserve(8192);
    edgeVerts.reserve(8192);

    auto val = [&](int i, int j, int k) -> float { return sdf[cellIndex(i, j, k)]; };
    const float iso = 0.0f;
    for (int k = 0; k + 1 < nz; ++k)
    for (int j = 0; j + 1 < ny; ++j)
    for (int i = 0; i + 1 < nx; ++i) {
        float cv[8];
        int cubeindex = 0;
        for (int c = 0; c < 8; ++c) {
            cv[c] = val(i + kCorner[c][0], j + kCorner[c][1], k + kCorner[c][2]);
            if (cv[c] < iso) cubeindex |= (1 << c);
        }
        const int edges = kEdgeTable[cubeindex];
        if (edges == 0) continue;

        uint32_t ev[12];
        for (int e = 0; e < 12; ++e) {
            if (!(edges & (1 << e))) continue;
            const int a = kEdgeCorner[e][0], b = kEdgeCorner[e][1];
            const int ai = i + kCorner[a][0], aj = j + kCorner[a][1], ak = k + kCorner[a][2];
            const int bi = i + kCorner[b][0], bj = j + kCorner[b][1], bk = k + kCorner[b][2];
            // Canonical key: the edge's lower corner + its axis (0=x, 1=y, 2=z).
            const int li = (std::min)(ai, bi), lj = (std::min)(aj, bj), lk = (std::min)(ak, bk);
            const int axis = (ai != bi) ? 0 : (aj != bj) ? 1 : 2;
            const uint64_t key = (static_cast<uint64_t>(cellIndex(li, lj, lk)) << 2) | static_cast<uint64_t>(axis);
            auto it = edgeVerts.find(key);
            if (it != edgeVerts.end()) { ev[e] = it->second; continue; }
            const float va = cv[a], vb = cv[b];
            float tt = (std::fabs(vb - va) > 1e-12f) ? (iso - va) / (vb - va) : 0.5f;
            if (tt < 0.0f) tt = 0.0f; else if (tt > 1.0f) tt = 1.0f;
            const Vec3 pa = cellCenter(ai, aj, ak);
            const Vec3 pb = cellCenter(bi, bj, bk);
            const uint32_t nvi = static_cast<uint32_t>(outP.size());
            outP.push_back(pa + (pb - pa) * tt);
            edgeVerts.emplace(key, nvi);
            ev[e] = nvi;
        }

        const int* tri = kTriTable[cubeindex];
        for (int t = 0; tri[t] != -1; t += 3) {
            // Same winding FoamSurface uses: with phi negative INSIDE, e0,e2,e1
            // fronts the outward direction.
            outIdx.push_back(ev[tri[t]]);
            outIdx.push_back(ev[tri[t + 2]]);
            outIdx.push_back(ev[tri[t + 1]]);
        }
    }
    if (outIdx.empty()) {
        ctx.addError(id, "Remesh: produced no surface - is the input a closed mesh? (Open sheets have no defined inside; try a smaller Voxel Size.)");
        return NodeSystem::PinValue{};
    }
    lastTrisOut = static_cast<int>(outIdx.size() / 3);

    // ---- 4) Optional Laplacian post-smooth (real adjacency — output is welded)
    if (smoothIterations > 0 && smoothFactor > 0.0f) {
        const size_t nv = outP.size();
        std::vector<Vec3> acc(nv);
        std::vector<uint32_t> cnt(nv);
        for (int it = 0; it < smoothIterations; ++it) {
            std::fill(acc.begin(), acc.end(), Vec3(0.0f, 0.0f, 0.0f));
            std::fill(cnt.begin(), cnt.end(), 0u);
            for (size_t e = 0; e + 2 < outIdx.size(); e += 3) {
                const uint32_t a = outIdx[e], b = outIdx[e + 1], c = outIdx[e + 2];
                acc[a] = acc[a] + outP[b] + outP[c]; cnt[a] += 2;
                acc[b] = acc[b] + outP[c] + outP[a]; cnt[b] += 2;
                acc[c] = acc[c] + outP[a] + outP[b]; cnt[c] += 2;
            }
            for (size_t v = 0; v < nv; ++v) {
                if (cnt[v] == 0) continue;
                const Vec3 avg = acc[v] * (1.0f / static_cast<float>(cnt[v]));
                outP[v] = outP[v] + (avg - outP[v]) * smoothFactor;
            }
        }
    }

    // ---- 5) Materialize -------------------------------------------------------
    // Smooth shading keeps the welded topology (area-weighted vertex normals via
    // the shared helper). Flat shading needs per-face normals, and this pipeline
    // stores normals per-vertex — so the flat path expands to soup (3 verts/tri),
    // exactly how every other flat-shaded mesh in the app represents hard faces.
    auto out = std::make_shared<TriangleMesh>();
    out->transform = std::make_shared<Transform>();
    if (inMesh->transform) *out->transform = *inMesh->transform;
    out->geometry = std::make_shared<DNA::GeometryDetail>();
    auto& dg = *out->geometry;
    dg.add_attribute<Vec3>("P");
    dg.add_attribute<Vec3>("N");
    dg.add_attribute<Vec3>("P_orig");
    dg.add_attribute<Vec3>("N_orig");
    if (sg.has_attribute("uv")) dg.add_attribute<Vec2>("uv");     // zero-filled: a remesh has no UVs
    if (srcMat) dg.add_attribute<uint16_t>("materialID");

    // Nearest input triangle for a surface point: the containing cell is inside
    // the stamped band (|phi| < ~h there), so its closestTri is set; fall back to
    // a 3x3x3 neighborhood scan, then material 0.
    auto materialAt = [&](const Vec3& p) -> uint16_t {
        if (!transferMaterial || !srcMat || closestTri.empty()) return 0;
        const int i = (std::max)(0, (std::min)(nx - 1, static_cast<int>((p.x - origin.x) / h)));
        const int j = (std::max)(0, (std::min)(ny - 1, static_cast<int>((p.y - origin.y) / h)));
        const int k = (std::max)(0, (std::min)(nz - 1, static_cast<int>((p.z - origin.z) / h)));
        int32_t tri = closestTri[cellIndex(i, j, k)];
        for (int dk = -1; dk <= 1 && tri < 0; ++dk)
        for (int dj = -1; dj <= 1 && tri < 0; ++dj)
        for (int di = -1; di <= 1 && tri < 0; ++di) {
            const int ii = i + di, jj = j + dj, kk = k + dk;
            if (ii < 0 || jj < 0 || kk < 0 || ii >= nx || jj >= ny || kk >= nz) continue;
            tri = closestTri[cellIndex(ii, jj, kk)];
        }
        return (tri >= 0) ? srcMat[srcIdx[static_cast<size_t>(tri) * 3]] : static_cast<uint16_t>(0);
    };

    if (smoothShading) {
        const size_t nv = outP.size();
        dg.resize_vertices(nv);
        Vec3* dPo = dg.get_attribute_data_mut<Vec3>("P_orig");
        uint16_t* dMat = dg.get_attribute_data_mut<uint16_t>("materialID");
        for (size_t v = 0; v < nv; ++v) {
            dPo[v] = outP[v];
            if (dMat) dMat[v] = materialAt(outP[v]);
        }
        dg.indices.assign(outIdx.begin(), outIdx.end());
        recomputeOrigNormals(dg);
    } else {
        const size_t nt = outIdx.size() / 3;
        dg.resize_vertices(nt * 3);
        Vec3* dPo = dg.get_attribute_data_mut<Vec3>("P_orig");
        Vec3* dNo = dg.get_attribute_data_mut<Vec3>("N_orig");
        uint16_t* dMat = dg.get_attribute_data_mut<uint16_t>("materialID");
        dg.indices.resize(nt * 3);
        for (size_t t = 0; t < nt; ++t) {
            const Vec3& a = outP[outIdx[t * 3 + 0]];
            const Vec3& b = outP[outIdx[t * 3 + 1]];
            const Vec3& c = outP[outIdx[t * 3 + 2]];
            const Vec3 fn = Vec3::cross(b - a, c - a);
            const float len = std::sqrt(fn.x * fn.x + fn.y * fn.y + fn.z * fn.z);
            const Vec3 n = (len > 1e-12f) ? fn * (1.0f / len) : Vec3(0.0f, 1.0f, 0.0f);
            const uint16_t m = materialAt(a);
            for (int c3 = 0; c3 < 3; ++c3) {
                const size_t dst = t * 3 + c3;
                dPo[dst] = (c3 == 0) ? a : (c3 == 1) ? b : c;
                if (dNo) dNo[dst] = n;
                if (dMat) dMat[dst] = m;
                dg.indices[dst] = static_cast<uint32_t>(dst);
            }
        }
    }

    rebakeFromOrig(*out);
    return NodeSystem::GeometryValue(out);
}

namespace {
    NodeSystem::AutoRegisterNode<GeometryNodesV2::BaseMeshNode>      reg_GeoBaseMesh("GeoV2.BaseMesh");
    NodeSystem::AutoRegisterNode<GeometryNodesV2::SubdivideCCNode>   reg_GeoSubdivideCC("GeoV2.SubdivideCC");
    NodeSystem::AutoRegisterNode<GeometryNodesV2::TransformNode>     reg_GeoTransform("GeoV2.Transform");
    NodeSystem::AutoRegisterNode<GeometryNodesV2::OutputNode>        reg_GeoOutput("GeoV2.Output");
    NodeSystem::AutoRegisterNode<GeometryNodesV2::MirrorNode>        reg_GeoMirror("GeoV2.Mirror");
    NodeSystem::AutoRegisterNode<GeometryNodesV2::NoiseDisplaceNode> reg_GeoNoiseDisplace("GeoV2.NoiseDisplace");
    NodeSystem::AutoRegisterNode<GeometryNodesV2::MergeNode>         reg_GeoMerge("GeoV2.Merge");
    NodeSystem::AutoRegisterNode<GeometryNodesV2::ObjectSourceNode>  reg_GeoObjectSource("GeoV2.ObjectSource");
    NodeSystem::AutoRegisterNode<GeometryNodesV2::WeldNode>          reg_GeoWeld("GeoV2.Weld");
    NodeSystem::AutoRegisterNode<GeometryNodesV2::MaskByHeightNode>  reg_GeoMaskByHeight("GeoV2.MaskByHeight");
    NodeSystem::AutoRegisterNode<GeometryNodesV2::MaskBySlopeNode>   reg_GeoMaskBySlope("GeoV2.MaskBySlope");
    NodeSystem::AutoRegisterNode<GeometryNodesV2::MaskNoiseNode>     reg_GeoMaskNoise("GeoV2.MaskNoise");
    NodeSystem::AutoRegisterNode<GeometryNodesV2::MaskRemapNode>     reg_GeoMaskRemap("GeoV2.MaskRemap");
    NodeSystem::AutoRegisterNode<GeometryNodesV2::MaskMathNode>      reg_GeoMaskMath("GeoV2.MaskMath");
    NodeSystem::AutoRegisterNode<GeometryNodesV2::ScatterInstancesNode> reg_GeoScatterInstances("GeoV2.ScatterInstances");
    NodeSystem::AutoRegisterNode<GeometryNodesV2::ArrayNode>         reg_GeoArray("GeoV2.Array");
    NodeSystem::AutoRegisterNode<GeometryNodesV2::ExtrudeNode>       reg_GeoExtrude("GeoV2.Extrude");
    NodeSystem::AutoRegisterNode<GeometryNodesV2::InsetNode>         reg_GeoInset("GeoV2.Inset");
    NodeSystem::AutoRegisterNode<GeometryNodesV2::BevelNode>         reg_GeoBevel("GeoV2.Bevel");
    NodeSystem::AutoRegisterNode<GeometryNodesV2::RemeshNode>        reg_GeoRemesh("GeoV2.Remesh");
}
