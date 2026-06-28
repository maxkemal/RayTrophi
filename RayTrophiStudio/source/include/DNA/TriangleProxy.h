#pragma once

#include "Vec3.h"
#include "Vec2.h"
#include "TriangleMesh.h"
#include "DNA/GeometryDetail.h"
#include <cstdint>
#include <string>
#include <functional>

namespace DNA {

    /**
     * @brief Lightweight by-value handle to a single face of a TriangleMesh's SoA geometry.
     *
     * Faz 1 of the DNA migration: replaces the per-face, heap-allocated `Triangle` facade
     * (~200 B object + shared_ptr control block + heap fragmentation, multiplied by every
     * triangle in the scene — the 12.5M / 25 GB CC-apply problem) with a 16-byte value that
     * stores only (mesh, faceIndex). Every geometric query reads directly from the mesh's
     * flat DNA::GeometryDetail buffers via the enum-slot fast path.
     *
     * Non-owning: a proxy is valid only while its referenced TriangleMesh is alive. The
     * (mesh, faceIndex) pair is a STABLE identity (unlike a transient Triangle*), so it can
     * back hit records and selection/edit lookups without per-face heap objects.
     *
     * This mirrors the read surface of the legacy Triangle facade so consumers can migrate
     * call-by-call. It deliberately does NOT inherit Hittable: traversal is index-based
     * (TriangleMesh::hit / the BVH iterate faces), not virtual-per-triangle.
     */
    struct TriangleProxy {
        TriangleMesh* mesh = nullptr;
        uint32_t      faceIndex = 0;

        TriangleProxy() = default;
        TriangleProxy(TriangleMesh* m, uint32_t f) noexcept : mesh(m), faceIndex(f) {}

        inline bool valid() const noexcept {
            return mesh != nullptr && mesh->geometry != nullptr;
        }

        // Global vertex index of a triangle corner (0..2) within the mesh's flat buffers.
        inline uint32_t vertexIndex(int corner) const noexcept {
            return mesh->geometry->indices[faceIndex * 3 + corner];
        }

        // ---- Geometry reads (current/world state) ----------------------------------------
        inline Vec3 position(int corner) const {
            const Vec3* P = mesh->geometry->get_positions();
            return P ? P[vertexIndex(corner)] : Vec3();
        }
        inline Vec3 normal(int corner) const {
            const Vec3* N = mesh->geometry->get_normals();
            return N ? N[vertexIndex(corner)] : Vec3();
        }
        inline Vec2 uv(int corner) const {
            const Vec2* uvs = mesh->geometry->get_uvs();
            return uvs ? uvs[vertexIndex(corner)] : Vec2();
        }

        // ---- Bind/original-pose reads ----------------------------------------------------
        inline Vec3 originalPosition(int corner) const {
            const Vec3* P = mesh->geometry->get_positions_orig();
            return P ? P[vertexIndex(corner)] : position(corner);
        }
        inline Vec3 originalNormal(int corner) const {
            const Vec3* N = mesh->geometry->get_normals_orig();
            return N ? N[vertexIndex(corner)] : normal(corner);
        }

        // ---- Writes (in-place into the SoA buffers) --------------------------------------
        inline void setPosition(int corner, const Vec3& p) {
            Vec3* P = mesh->geometry->get_positions_mut();
            if (P) P[vertexIndex(corner)] = p;
        }
        inline void setNormal(int corner, const Vec3& n) {
            Vec3* N = mesh->geometry->get_normals_mut();
            if (N) N[vertexIndex(corner)] = n;
        }

        // ---- Material / identity / metadata ----------------------------------------------
        inline uint16_t materialID() const {
            const uint16_t* m = mesh->geometry->get_material_ids();
            return m ? m[vertexIndex(0)] : static_cast<uint16_t>(0xFFFF);
        }
        inline const std::string& nodeName() const noexcept {
            return mesh->nodeName;
        }
        inline Transform* transformPtr() const noexcept {
            return mesh->transform.get();
        }

        // ---- AABB in local mesh space ----------------------------------------------------
        inline AABB bounds() const {
            const Vec3 v0 = position(0), v1 = position(1), v2 = position(2);
            return AABB(Vec3::min(Vec3::min(v0, v1), v2),
                        Vec3::max(Vec3::max(v0, v1), v2));
        }

        // (mesh, faceIndex) is the stable identity.
        inline bool operator==(const TriangleProxy& o) const noexcept {
            return mesh == o.mesh && faceIndex == o.faceIndex;
        }
        inline bool operator!=(const TriangleProxy& o) const noexcept {
            return !(*this == o);
        }
    };

} // namespace DNA

// Hash so a TriangleProxy can key the selection/edit identity maps that currently key on
// Triangle* (tri_to_index). Migrating those maps to TriangleProxy is the final Faz 1 slice.
namespace std {
    template <>
    struct hash<DNA::TriangleProxy> {
        size_t operator()(const DNA::TriangleProxy& p) const noexcept {
            return std::hash<void*>{}(p.mesh) ^ (std::hash<uint32_t>{}(p.faceIndex) << 1);
        }
    };
}
