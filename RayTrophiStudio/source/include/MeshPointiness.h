#pragma once

// =============================================================================
// MeshPointiness — per-vertex convexity/concavity ("pointiness") attribute
// =============================================================================
// The Geometry node's Pointiness output. Same construction as Blender/Cycles'
// ATTR_STD_POINTINESS so the values are directly comparable:
//
//   1. WELD coincident positions. This is not optional here: a flat mesh out of
//      facadesToFlatMesh is an UNWELDED triangle soup, so without the weld every
//      vertex's 1-ring is its own face alone and the whole mesh reads "flat".
//   2. For every welded vertex, accumulate the normalized direction to its 1-ring
//      neighbours. Cycles de-duplicates edges through an edge map; we scatter per
//      DIRECTED edge instead — identical on a closed mesh (the neighbour count cancels
//      in the normalize), a slightly different weighting only on boundary seams, and
//      O(V) memory instead of an O(6F) adjacency table.
//   3. raw = angle(vertex normal, mean neighbour direction) / PI.
//      A neighbourhood that sits BELOW the tangent plane (a convex ridge/spike)
//      gives a negative dot -> angle > 90deg -> raw > 0.5; a crevice gives < 0.5;
//      a flat patch has the mean direction perpendicular to N -> exactly 0.5.
//   4. One blur pass over the 1-ring (Cycles' "approximate 2-ring") to kill the
//      per-vertex noise that a raw curvature estimate has on dense meshes.
//   5. Copy the welded value back onto every duplicate.
//
// The SAME function feeds the CPU render (per-vertex cache on TriangleMesh,
// barycentric-interpolated at the hit) and the Vulkan RT upload (per-vertex GPU
// buffer, barycentric-interpolated in closesthit) — so CPU stays a bit-comparable
// oracle for the GPU path instead of the two drifting apart.
//
// Cost is paid ONLY when some live material graph actually reads Pointiness
// (anyMaterialUsesPointiness()); scenes that never touch it pay nothing.

#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>

#include "Vec3.h"
#include "TriangleMesh.h"
#include "Transform.h"       // objectOrigin() needs the full type (TriangleMesh.h only forward-declares it)
#include "MaterialManager.h"
#include "PrincipledBSDF.h"

namespace MeshAttr {

    /// Value of a perfectly flat surface — also the fallback everywhere the
    /// attribute is missing (no mesh, no cache, degenerate face).
    inline constexpr float kPointinessFlat = 0.5f;

    namespace detail {

        inline uint32_t pcgMix(uint32_t v) {
            v = v * 747796405u + 2891336453u;
            const uint32_t word = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
            return (word >> 22u) ^ word;
        }

        inline uint32_t floatBits(float f) {
            if (f == 0.0f) f = 0.0f;          // -0.0 and +0.0 must weld together
            uint32_t u;
            std::memcpy(&u, &f, sizeof(u));
            return u;
        }

        // Avalanche-chained hash (never a plain XOR of the coordinate words — that
        // makes (x,y,z) and (-x,-y,-z) collide on a mirrored mesh).
        inline uint32_t posHash(float x, float y, float z) {
            uint32_t h = pcgMix(floatBits(x));
            h = pcgMix(h ^ floatBits(y));
            h = pcgMix(h ^ floatBits(z));
            return h;
        }

        inline Vec3 safeDir(const Vec3& d) {
            const float l = std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
            // Deliberately NOT Vec3::normalize(): its 1e-3 length gate zeroes short
            // edges, and a dense mesh is mostly short edges.
            if (l < 1e-20f) return Vec3(0.0f, 0.0f, 0.0f);
            const float inv = 1.0f / l;
            return Vec3(d.x * inv, d.y * inv, d.z * inv);
        }

    } // namespace detail

    /**
     * @brief Per-vertex pointiness for a triangle mesh.
     *
     * @param P          3*vertexCount floats (x,y,z). Any space — the measure is an
     *                   angle, so translation/rotation/uniform scale don't change it.
     * @param N          3*vertexCount vertex normals, or null (face normals are used).
     * @param indices    Corner -> vertex. Null means an implicit soup (0,1,2,3,...).
     * @param out        vertexCount values; 0.5 flat, >0.5 convex, <0.5 concave.
     */
    inline void computePointiness(const float* P, const float* N, size_t vertexCount,
                                  const uint32_t* indices, size_t indexCount,
                                  std::vector<float>& out) {
        out.assign(vertexCount, kPointinessFlat);
        if (!P || vertexCount == 0) return;

        const size_t triCount = indices ? (indexCount / 3) : (vertexCount / 3);
        if (triCount == 0) return;

        auto corner = [&](size_t c) -> uint32_t {
            return indices ? indices[c] : static_cast<uint32_t>(c);
        };
        auto pos = [&](uint32_t v) -> Vec3 {
            return Vec3(P[v * 3 + 0], P[v * 3 + 1], P[v * 3 + 2]);
        };

        // ── 1) Weld by exact position into a COMPACT id space. Split UV/material copies
        //       carry bit-identical coordinates, so exact equality is the right key — no
        //       quantization. Compacting (rather than picking a representative vertex id)
        //       keeps every array below sized by unique verts, which on an unwelded soup
        //       is a third of the corners.
        std::vector<uint32_t> compact(vertexCount);
        std::vector<uint32_t> firstVert;   // compact id -> one source vertex (for P lookup)
        {
            // Load factor <= ~0.67. A 2x table would cost half a gigabyte on a 36M-corner
            // soup — the probe chains at 0.67 are far cheaper than that allocation.
            size_t cap = 16;
            while (cap < vertexCount + vertexCount / 2) cap <<= 1;
            const uint32_t mask = static_cast<uint32_t>(cap - 1);
            std::vector<uint32_t> table(cap, 0);   // slot -> (compact id + 1); 0 = empty
            firstVert.reserve(vertexCount / 2 + 1);

            for (size_t v = 0; v < vertexCount; ++v) {
                const float x = P[v * 3 + 0], y = P[v * 3 + 1], z = P[v * 3 + 2];
                uint32_t slot = detail::posHash(x, y, z) & mask;
                uint32_t found = UINT32_MAX;
                while (table[slot] != 0) {
                    const uint32_t cid = table[slot] - 1;
                    const uint32_t o = firstVert[cid];
                    if (P[o * 3 + 0] == x && P[o * 3 + 1] == y && P[o * 3 + 2] == z) {
                        found = cid;
                        break;
                    }
                    slot = (slot + 1) & mask;
                }
                if (found == UINT32_MAX) {
                    found = static_cast<uint32_t>(firstVert.size());
                    firstVert.push_back(static_cast<uint32_t>(v));
                    table[slot] = found + 1;
                }
                compact[v] = found;
            }
        }
        const size_t uCount = firstVert.size();

        // ── 2) Welded vertex normals: duplicates' normals sum into the welded vertex.
        //       With no N channel, area-weighted face normals stand in.
        std::vector<Vec3> vN(uCount, Vec3(0.0f, 0.0f, 0.0f));
        if (N) {
            for (size_t v = 0; v < vertexCount; ++v) {
                vN[compact[v]] += Vec3(N[v * 3 + 0], N[v * 3 + 1], N[v * 3 + 2]);
            }
        } else {
            for (size_t t = 0; t < triCount; ++t) {
                const uint32_t a = compact[corner(t * 3 + 0)];
                const uint32_t b = compact[corner(t * 3 + 1)];
                const uint32_t c = compact[corner(t * 3 + 2)];
                const Vec3 fn = Vec3::cross(pos(firstVert[b]) - pos(firstVert[a]),
                                            pos(firstVert[c]) - pos(firstVert[a]));
                vN[a] += fn; vN[b] += fn; vN[c] += fn;
            }
        }

        // ── 3) 1-ring edge accumulation, scattered straight off the face list.
        //       Cycles walks a de-duplicated edge map here; we accumulate per DIRECTED
        //       edge instead, which weights a shared (interior) edge twice and a boundary
        //       edge once. On a closed mesh the result is identical — the count cancels in
        //       the normalize below — and on a boundary it only nudges the seam vertices.
        //       That trade buys O(V) memory instead of an O(6F) adjacency table, which on
        //       a multi-million-triangle mesh is the difference between ~100MB and ~1GB.
        std::vector<Vec3> edgeAccum(uCount, Vec3(0.0f, 0.0f, 0.0f));
        std::vector<uint32_t> ringCount(uCount, 0);
        for (size_t t = 0; t < triCount; ++t) {
            for (int k = 0; k < 3; ++k) {
                const uint32_t a = compact[corner(t * 3 + k)];
                const uint32_t b = compact[corner(t * 3 + (k + 1) % 3)];
                if (a == b) continue;                       // degenerate edge
                const Vec3 pa = pos(firstVert[a]);
                const Vec3 pb = pos(firstVert[b]);
                const Vec3 d = detail::safeDir(pb - pa);
                edgeAccum[a] += d;
                edgeAccum[b] += -d;
                ++ringCount[a];
                ++ringCount[b];
            }
        }

        // ── 4) raw = angle(N, mean neighbour direction) / PI.
        std::vector<float> raw(uCount, kPointinessFlat);
        const int uCount_i = static_cast<int>(uCount);
        #pragma omp parallel for schedule(static)
        for (int ui = 0; ui < uCount_i; ++ui) {
            if (ringCount[ui] == 0) continue;
            const Vec3 nrm = detail::safeDir(vN[ui]);
            const Vec3 dir = detail::safeDir(edgeAccum[ui]);  // the 1/count cancels here
            const float d = std::clamp(Vec3::dot(nrm, dir), -1.0f, 1.0f);
            raw[ui] = std::acos(d) * (1.0f / 3.14159265358979f);
        }

        // ── 5) Blur over the 1-ring (Cycles' 2-ring approximation) — same directed-edge
        //       scatter, so a raw curvature estimate doesn't read as noise on dense meshes.
        std::vector<float> blur(raw);
        std::vector<uint32_t> blurCount(uCount, 0);
        for (size_t t = 0; t < triCount; ++t) {
            for (int k = 0; k < 3; ++k) {
                const uint32_t a = compact[corner(t * 3 + k)];
                const uint32_t b = compact[corner(t * 3 + (k + 1) % 3)];
                if (a == b) continue;
                blur[a] += raw[b];
                blur[b] += raw[a];
                ++blurCount[a];
                ++blurCount[b];
            }
        }

        // ── 6) Scatter the welded values back onto every source vertex.
        const int vCount_i = static_cast<int>(vertexCount);
        #pragma omp parallel for schedule(static)
        for (int vi = 0; vi < vCount_i; ++vi) {
            const uint32_t c = compact[vi];
            out[vi] = blur[c] / static_cast<float>(blurCount[c] + 1);
        }
    }

    /// Rebuild a mesh's per-vertex cache from its live SoA. Cheap no-op for an
    /// empty mesh; otherwise always recomputes (callers gate on "is it needed").
    inline void computeMeshPointiness(TriangleMesh& mesh) {
        if (!mesh.geometry) { mesh.pointiness.clear(); return; }
        const DNA::GeometryDetail& g = *mesh.geometry;
        const size_t vCount = g.get_vertex_count();
        const Vec3* P = g.get_positions();
        if (vCount == 0 || !P || g.indices.empty()) { mesh.pointiness.clear(); return; }
        const Vec3* N = g.get_normals();

        computePointiness(reinterpret_cast<const float*>(P),
                          reinterpret_cast<const float*>(N),
                          vCount, g.indices.data(), g.indices.size(),
                          mesh.pointiness);
    }

    /// Does any live PrincipledBSDF's compiled program read the Geometry node's
    /// Pointiness output? Gates the whole precompute.
    inline bool anyMaterialUsesPointiness() {
        for (const auto& mat : MaterialManager::getInstance().getAllMaterials()) {
            if (!mat || mat->type() != MaterialType::PrincipledBSDF) continue;
            const auto* pbsdf = static_cast<const PrincipledBSDF*>(mat.get());
            const auto& prog = pbsdf->proceduralProgram;
            if (prog && prog->active && prog->usesPointiness) return true;
        }
        return false;
    }

    // ---- Attribute node: named per-vertex channels ------------------------------
    // HitRecord::mat_attrib is a plain float[4] because Hittable.h sits below
    // MaterialProgram.h in the include graph and cannot name the constant. Pin them here,
    // where both headers are visible, so widening kMatAttribSlots can never silently
    // overrun the HitRecord array.
    static_assert(MaterialNodesV2::kMatAttribSlots == 4,
                  "HitRecord::mat_attrib[4] and the GLSL MP_ATTRIB_SLOTS must be widened together");

    /// Does any live material graph read an Attribute node? Gates the per-vertex block
    /// exactly like pointiness: no reader => no mesh cache, no GPU upload, and the hot-path
    /// sampler collapses to a null check.
    inline bool anyMaterialUsesAttributes() {
        for (const auto& mat : MaterialManager::getInstance().getAllMaterials()) {
            if (!mat || mat->type() != MaterialType::PrincipledBSDF) continue;
            const auto* pbsdf = static_cast<const PrincipledBSDF*>(mat.get());
            const auto& prog = pbsdf->proceduralProgram;
            if (prog && prog->active && prog->usesAttributes) return true;
        }
        return false;
    }

    /// Gather the interned attribute channels out of a GeometryDetail's custom map into the
    /// interleaved per-vertex block the hit path and the GPU upload both read. A slot whose
    /// name this mesh does not carry is left at 0 (unpainted) — meshes are free to have
    /// different channels; the SLOT numbering is scene-wide, the DATA is not. `out` comes
    /// back EMPTY when the mesh carries none of them, so it costs nothing instead of
    /// vCount*K zeros.
    ///
    /// Split from the TriangleMesh overload because the Vulkan indexed upload holds the mesh
    /// as `const TriangleMesh*` and must be able to build the block into its own scratch.
    inline void computeMaterialAttributes(const DNA::GeometryDetail& g, std::vector<float>& out) {
        out.clear();
        const auto& slots = MaterialNodesV2::materialAttributeSlots();
        const size_t vCount = g.get_vertex_count();
        if (slots.empty() || vCount == 0) return;

        constexpr int K = MaterialNodesV2::kMatAttribSlots;

        const float* src[K] = { nullptr, nullptr, nullptr, nullptr };
        bool any = false;
        for (size_t s = 0; s < slots.size() && s < static_cast<size_t>(K); ++s) {
            // The Attribute node's name field is free text, so a user can type a CORE channel
            // name ("P", "N", "uv"). GeometryDetail would happily hand those back reinterpreted
            // as float* — i.e. we would read the x/y/z of a Vec3 buffer as three separate
            // "vertices" and walk off the end of a shorter one. Named channels are the custom
            // map only; a core name is simply not an attribute here.
            if (DNA::core_attr_index(slots[s]) >= 0) continue;
            src[s] = g.get_attribute_data<float>(slots[s]);
            if (src[s]) any = true;
        }
        if (!any) return;

        out.assign(vCount * static_cast<size_t>(K), 0.0f);
        for (int s = 0; s < K; ++s) {
            if (!src[s]) continue;
            for (size_t v = 0; v < vCount; ++v) {
                out[v * static_cast<size_t>(K) + static_cast<size_t>(s)] = src[s][v];
            }
        }
    }

    /// Rebuild a mesh's own block (the CPU hit path's cache).
    inline void computeMeshMaterialAttributes(TriangleMesh& mesh) {
        if (!mesh.geometry) { mesh.material_attribs.clear(); return; }
        computeMaterialAttributes(*mesh.geometry, mesh.material_attribs);
    }

    /// Shading point in OBJECT space (HitRecord::object_position) for the procedural
    /// "Object Space" toggle: the barycentric blend of the face's BIND-POSE positions
    /// (P_orig), which is the space the Vulkan BLAS vertices are uploaded in — closesthit
    /// gets the same quantity by interpolating its object-space verts. Returns false when
    /// the mesh has no bind pose to offer (procedural / transformless geometry), and the
    /// caller then leaves the world point in place: for those objects world IS object space,
    /// because their BLAS carries world verts under an identity instance transform.
    inline bool sampleObjectPosition(const TriangleMesh* mesh, uint32_t face,
                                     float w, float u, float v, Vec3& out) {
        if (!mesh || !mesh->geometry || !mesh->transform) return false;
        const DNA::GeometryDetail& g = *mesh->geometry;
        const Vec3* Po = g.get_positions_orig();
        if (!Po) return false;
        const auto& idx = g.indices;
        const size_t base = static_cast<size_t>(face) * 3u;
        if (base + 2u >= idx.size()) return false;
        const uint32_t i0 = idx[base + 0], i1 = idx[base + 1], i2 = idx[base + 2];
        const size_t vc = g.get_vertex_count();
        if (i0 >= vc || i1 >= vc || i2 >= vc) return false;
        out = Po[i0] * w + Po[i1] * u + Po[i2] * v;
        return true;
    }

    /// Barycentric fetch of all slots at a hit (w,u,v over the face's three corners).
    /// Writes zeros when the mesh has no block — i.e. whenever nothing reads attributes.
    inline void sampleMaterialAttributes(const TriangleMesh* mesh, uint32_t face,
                                         float w, float u, float v, float* out) {
        constexpr int K = MaterialNodesV2::kMatAttribSlots;
        for (int s = 0; s < K; ++s) out[s] = 0.0f;
        if (!mesh || mesh->material_attribs.empty() || !mesh->geometry) return;

        const auto& idx = mesh->geometry->indices;
        const size_t base = static_cast<size_t>(face) * 3u;
        if (base + 2u >= idx.size()) return;
        const uint32_t i0 = idx[base + 0], i1 = idx[base + 1], i2 = idx[base + 2];

        const std::vector<float>& A = mesh->material_attribs;
        const size_t o0 = static_cast<size_t>(i0) * K;
        const size_t o1 = static_cast<size_t>(i1) * K;
        const size_t o2 = static_cast<size_t>(i2) * K;
        if (o0 + K > A.size() || o1 + K > A.size() || o2 + K > A.size()) return;

        for (int s = 0; s < K; ++s) {
            out[s] = A[o0 + s] * w + A[o1 + s] * u + A[o2 + s] * v;
        }
    }

    /// Object Info: the world-space origin of the object a hit belongs to, for
    /// HitRecord::object_origin.
    ///
    /// The one rule that matters here is that this must return the translation of the
    /// SAME matrix the Vulkan TLAS instance was built from, because closesthit reads its
    /// copy of it as gl_ObjectToWorldEXT[3].xyz and both sides then hash the result. The
    /// Vulkan uploader takes Transform::getFinal() (VulkanBackend's solo-group and direct-
    /// mesh paths), which is what this returns — and it falls back to the origin for a
    /// null handle, matching the identity instance transform that world-baked geometry
    /// gets on the GPU. Change one side and the two silently disagree.
    ///
    /// Called per hit, from every render thread. getFinal() lazily recomputes and writes a
    /// cache when the Transform is dirty, which would be a data race here — it is safe only
    /// because a dirty transform cannot survive into a render pass: the BVH bakes world
    /// vertices through getFinal() at build time, so a transform still dirty when rays fly
    /// would already mean stale geometry. (TriangleMesh::hit has leaned on the same
    /// invariant for as long as it has called getMatrix() in its hit path.)
    inline Vec3 objectOrigin(const Transform* xf) {
        return xf ? xf->getFinal().getTranslation() : Vec3(0.0f, 0.0f, 0.0f);
    }

    /// Barycentric fetch at a hit (w,u,v over the face's three corners). Returns the
    /// flat value whenever the mesh has no cache — i.e. whenever nothing reads it.
    inline float samplePointiness(const TriangleMesh* mesh, uint32_t face,
                                  float w, float u, float v) {
        if (!mesh || mesh->pointiness.empty() || !mesh->geometry) return kPointinessFlat;
        const auto& idx = mesh->geometry->indices;
        const size_t base = static_cast<size_t>(face) * 3u;
        if (base + 2u >= idx.size()) return kPointinessFlat;
        const uint32_t i0 = idx[base + 0], i1 = idx[base + 1], i2 = idx[base + 2];
        const std::vector<float>& pt = mesh->pointiness;
        if (i0 >= pt.size() || i1 >= pt.size() || i2 >= pt.size()) return kPointinessFlat;
        return pt[i0] * w + pt[i1] * u + pt[i2] * v;
    }

} // namespace MeshAttr
