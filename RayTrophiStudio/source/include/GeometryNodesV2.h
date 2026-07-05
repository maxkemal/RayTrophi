/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          GeometryNodesV2.h
* Author:        Kemal Demirtas
* Date:          July 2026
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once

/**
 * @file GeometryNodesV2.h
 * @brief Faz 8a — first slice of the unified Geo-DAG: a typed Geometry socket
 * (std::shared_ptr<TriangleMesh>, see NodeCore.h's GeometryValue) flowing
 * through NodeSystem::NodeBase nodes, mirroring TerrainNodesV2's pattern
 * (GeometryContext/GeometryNodeBase are the Geo-DAG's TerrainContext/
 * TerrainNodeBase). This is purely ADDITIVE — the existing MeshModifiers::
 * ModifierStack UI/panel is untouched and keeps working exactly as before;
 * these nodes are a new, optional way to build the same kind of geometry
 * chain, wired into NodeEditorUIV2/NodeRegistry instead of the linear stack
 * panel.
 *
 * Current node set:
 *   - BaseMeshNode:      source node, outputs GeometryContext::baseMesh as-is.
 *   - ObjectSourceNode:  source node, another scene object's mesh by name
 *     (via GeometryContext::resolveObjectMesh) — second input for Merge.
 *   - SubdivideCCNode:   wraps MeshModifiers::catmullClarkSubDStencil (the
 *     exact function TerrainNodesV2's/the ModifierStack's live CC modifier
 *     already uses), so the algorithm is proven/shared, not reimplemented.
 *   - TransformNode:     pivot translate/rotate/scale.
 *   - MirrorNode:        local-axis reflection + winding flip, optional
 *     merge-with-original (Blender Mirror modifier behaviour).
 *   - NoiseDisplaceNode: displacement along normals with 8 selectable noise
 *     types (FBM/Perlin/Simplex/Turbulence/Ridge/Billow/Voronoi/Crackle),
 *     reusing Physics::Noise (CurlNoise) — same samplers force fields use.
 *   - MergeNode:         join two geometry streams (B re-expressed in A's
 *     pivot space), per-vertex materialIDs preserved.
 *   - WeldNode:          merge vertices by distance (close Mirror seams,
 *     fuse Merge results) — spatial-hash clustering.
 *   - MaskByHeightNode / MaskBySlopeNode / MaskNoiseNode (Faz 8b): write a
 *     per-vertex float "Field" attribute (default "mask") from local height /
 *     surface steepness / procedural noise; consumed by mask-aware nodes
 *     (Noise Displace / Scatter Instances). Fields ride the Geometry wire as
 *     GeometryDetail named attributes — no separate socket type (the Faz 8
 *     plan's Houdini model).
 *   - ScatterInstancesNode (Faz 8b): mask-weighted surface scattering of
 *     another scene object through the existing InstanceManager/foliage
 *     pipeline (stable group name = re-Evaluate replaces, never duplicates).
 *   - ArrayNode:         Blender Array modifier (Object Offset style) — Count copies,
 *     each placed by compounding the same per-step translate/rotate/scale onto the
 *     previous one (reuses Merge's appendMeshInto relPoint/relNormal mechanism).
 *   - BevelNode:         angle-limited edge bevel (Blender Bevel modifier, Limit=Angle)
 *     with Segments + Flat/Round profile — selectBevelEdgesByAngle + bevelEdgesPolygons
 *     over the same half-edge bridge. The SAME core drives the Edit Mode selected-edge
 *     bevel tool (SceneUI::bevelSelectedEdges); the short-lived ModifierStack Bevel was
 *     retired (it compounded over its own output on every stack evaluate).
 *   - ExtrudeNode / InsetNode: per-face topology ops built on MeshEdit::HalfEdgeMesh —
 *     the SAME half-edge core the facade-based Edit Mode's Extrude/Inset already use.
 *     buildHalfEdgeFromFlatMesh welds the flat soup by position (buildCoincidentRemap)
 *     into a connected half-edge mesh, extrudeFace/insetFace run per face (Blender's
 *     "Individual Faces" semantics, not Region), rebuildFlatMeshFromHalfEdge fan-
 *     triangulates the result back via the same Triangle-facade + facadesToFlatMesh
 *     path SubdivideCC's Flat mode uses.
 *   - OutputNode:        the graph's single result sink.
 *
 * More node types (Bevel, Boolean, ...) are meant to be
 * added the same way TerrainNodesV2 grew to 36 node types: one class + one
 * NodeType enum entry + one NodeRegistry self-registration line, without
 * touching the core graph/eval machinery.
 */

#include "NodeSystem/NodeCore.h"
#include "NodeSystem/Node.h"
#include "NodeSystem/Graph.h"
#include "NodeSystem/EvaluationContext.h"
#include "NodeSystem/NodeRegistry.h"   // deserializeGeometryGraph creates nodes by typeId
#include "TriangleMesh.h"
#include "Triangle.h"
#include "MeshModifiers.h"
#include "MeshEdit/HalfEdgeMesh.h"   // Extrude/Inset — same half-edge core the facade Edit Mode uses
#include "Transform.h"
#include "CurlNoise.h"   // Physics::Noise::fbm3D — same host noise force fields/emitters use
#include <memory>
#include <vector>
#include <functional>
#include <utility>
#include <cmath>
#include <unordered_map>
#include <string>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <limits>

namespace GeometryNodesV2 {

    // ============================================================================
    // GEOMETRY EVALUATION CONTEXT
    // ============================================================================

    /**
     * @brief Domain context for a Geo-DAG evaluation pass — set once by the
     * caller before pulling the graph's terminal node(s), mirroring
     * TerrainNodesV2::TerrainContext.
     */
    struct GeometryContext {
        std::shared_ptr<TriangleMesh> baseMesh;  ///< Source geometry for BaseMeshNode

        /// Resolve another scene object's flat TriangleMesh by nodeName (for
        /// ObjectSourceNode). Set by evaluateGeometryGraph from direct_mesh_nodes;
        /// null when the caller doesn't support cross-object lookups. Returns the
        /// object's CURRENT live mesh (or the pristine snapshot for the graph's own
        /// bound object, to avoid compounding on the previous Evaluate's output).
        std::function<std::shared_ptr<TriangleMesh>(const std::string&)> resolveObjectMesh;

        /// Gather per-face Triangle facades for EVERY scene object sharing a nodeName —
        /// unlike resolveObjectMesh (single flat mesh), this handles multi-material
        /// imports, where one nodeName owns SEVERAL sibling TriangleMesh entries (and
        /// legacy facade-based objects). ScatterInstancesNode uses this for its SOURCE:
        /// resolving only one sibling produced instances with missing triangles and a
        /// single (wrong) material. Set by evaluateGeometryGraph from world.objects.
        std::function<std::vector<std::shared_ptr<Triangle>>(const std::string&)> gatherObjectFacades;

        /// nodeName of the object this graph is bound to (set by evaluateGeometryGraph).
        /// Used by ScatterInstancesNode for stable group naming + target binding.
        std::string objectName;

        /// Set by nodes whose side effect touches InstanceManager (ScatterInstancesNode):
        /// tells evaluateGeometryGraph to run InstanceManager::rebuildSceneObjects after
        /// the geometry result is applied, so new instances enter world.objects before
        /// the BVH/backend rebuilds.
        bool instancesDirty = false;
    };

    /// Scene-object name list for ObjectSourceNode's picker combo. Reassigned every
    /// frame by the Geometry Graph window block (captures that frame's scene by
    /// reference, same lifetime pattern as NodeEditorUIV2::onDrawBackgroundMenu — only
    /// ever invoked synchronously within the same frame's properties-panel draw).
    /// Invoked lazily, ONLY while the combo popup is actually open, so the cost is
    /// zero until the user opens the picker — and because it re-reads the scene on
    /// every open, freshly added/deleted objects show up with no polling/tracking.
    inline std::function<std::vector<std::string>()> g_sceneObjectListProvider;

    // ============================================================================
    // NODE TYPES ENUM
    // ============================================================================

    enum class NodeType {
        BaseMesh,
        SubdivideCC,
        Transform,
        Output,
        Mirror,
        NoiseDisplace,
        Merge,
        ObjectSource,
        Weld,
        MaskByHeight,
        MaskBySlope,
        MaskNoise,
        ScatterInstances,
        MaskRemap,
        MaskMath,
        Array,
        Extrude,
        Inset,
        Bevel,
        Remesh,
    };

    // ============================================================================
    // SHARED MESH HELPERS (used by the nodes below — all pure, input untouched)
    // ============================================================================

    /// Deep-copy a mesh the way TransformNode does: distinct GeometryDetail + Transform
    /// instances so the node's output never aliases (and never mutates) its input.
    /// The copy's delta stack is FLATTENED into base: if the source carried deltas
    /// (sculpt/undo history), a plain copy would route this node's subsequent writes
    /// (mask attributes, displaced P_orig, ...) into the active clone — which the NEXT
    /// deepCopyMesh down the chain silently discards (copy-construction only carries
    /// base + deltas). Flattening is visually a no-op and makes writes stick.
    inline std::shared_ptr<TriangleMesh> deepCopyMesh(const NodeSystem::GeometryValue& in) {
        auto out = std::make_shared<TriangleMesh>();
        out->geometry = std::make_shared<DNA::GeometryDetail>(*in->geometry);
        out->geometry->flatten_deltas();
        out->transform = std::make_shared<Transform>();
        if (in->transform) *out->transform = *in->transform;
        return out;
    }

    /// Re-derive the active/world P/N buffers from local P_orig/N_orig using the mesh's
    /// current pivot — the rebakeMesh() invariant (P = transform->getFinal() * P_orig)
    /// every node that writes P_orig must restore before returning its output.
    inline void rebakeFromOrig(TriangleMesh& mesh) {
        if (!mesh.geometry || !mesh.transform) return;
        auto& geom = *mesh.geometry;
        const int vc = static_cast<int>(geom.get_vertex_count());
        const Vec3* Po = geom.get_attribute_data<Vec3>("P_orig");
        const Vec3* No = geom.get_attribute_data<Vec3>("N_orig");
        Vec3* P = geom.get_positions_mut();
        Vec3* N = geom.get_normals_mut();
        const Matrix4x4 fT = mesh.transform->getFinal();
        const Matrix4x4 fN = mesh.transform->getNormalTransform();
        if (P && Po) {
            for (int v = 0; v < vc; ++v) {
                P[v] = fT.transform_point(Po[v]);
                if (N && No) N[v] = fN.transform_vector(No[v]).normalize();
            }
        }
    }

    /**
     * @brief Cluster vertices that sit at the EXACT same position (bit-identical
     * floats): remap[v] = lowest vertex index sharing v's position. UV-seam splits and
     * flat-subdivide "soup" duplicates are exact copies of the same float values, so
     * bit equality is both sufficient and collision-safe (no distance threshold to
     * mis-tune). Any operation that moves vertices (displacement) or averages normals
     * MUST treat such a cluster as one logical vertex, or the surface tears/shades
     * apart along seam borders.
     */
    /// Bit-identical position hash (FNV-1a over the raw float bits, -0.0f canonicalized
    /// to 0.0f so it hashes with +0.0f). Exported (not a local lambda) so other position-
    /// keyed lookups outside this header — e.g. the sculpt-mask-to-attribute bridge in
    /// scene_ui_mesh_overlay.cpp, which maps EditableMeshCache's welded local_position
    /// vertices onto a flat mesh's (possibly seam-duplicated) P_orig — can reuse the same
    /// bucketing scheme instead of re-implementing it.
    inline uint64_t positionBitsKey(const Vec3& p) {
        const float f[3] = { p.x == 0.0f ? 0.0f : p.x,
                             p.y == 0.0f ? 0.0f : p.y,
                             p.z == 0.0f ? 0.0f : p.z };
        uint32_t b[3];
        std::memcpy(b, f, sizeof(b));
        uint64_t h = 1469598103934665603ULL;               // FNV-1a
        for (int i = 0; i < 3; ++i) { h ^= b[i]; h *= 1099511628211ULL; }
        return h;
    }

    inline std::vector<uint32_t> buildCoincidentRemap(const Vec3* P, size_t vc) {
        std::vector<uint32_t> remap(vc);
        std::unordered_map<uint64_t, std::vector<uint32_t>> buckets;
        buckets.reserve(vc);
        for (size_t v = 0; v < vc; ++v) {
            auto& bucket = buckets[positionBitsKey(P[v])];
            uint32_t canon = static_cast<uint32_t>(v);
            for (const uint32_t cand : bucket) {
                if (P[cand].x == P[v].x && P[cand].y == P[v].y && P[cand].z == P[v].z) {
                    canon = cand;
                    break;
                }
            }
            remap[v] = canon;
            if (canon == v) bucket.push_back(static_cast<uint32_t>(v));
        }
        return remap;
    }

    /// Build a position -> value lookup (bucketed by positionBitsKey, exact-equality
    /// verified within a bucket) usable to transfer a per-vertex scalar from ONE mesh's
    /// vertex set onto ANOTHER's, when both are known to share the same local-space
    /// positions at matching vertices (welded cache -> seam-duplicated flat mesh, or vice
    /// versa). Positions with no match in `lookup` return `defaultValue`.
    class PositionValueLookup {
    public:
        PositionValueLookup(const Vec3* positions, const float* values, size_t count) {
            buckets_.reserve(count);
            for (size_t i = 0; i < count; ++i) {
                buckets_[positionBitsKey(positions[i])].push_back({ positions[i], values[i] });
            }
        }
        float sample(const Vec3& p, float defaultValue = 0.0f) const {
            auto it = buckets_.find(positionBitsKey(p));
            if (it == buckets_.end()) return defaultValue;
            for (const auto& entry : it->second) {
                if (entry.first.x == p.x && entry.first.y == p.y && entry.first.z == p.z) return entry.second;
            }
            return defaultValue;
        }
    private:
        std::unordered_map<uint64_t, std::vector<std::pair<Vec3, float>>> buckets_;
    };

    /// Rebuild N_orig from topology: area-weighted face-normal accumulation, WELD-AWARE —
    /// contributions are gathered per coincident-position cluster (buildCoincidentRemap),
    /// not per raw vertex index, so UV-seam splits and soup duplicates end up with one
    /// shared smooth normal instead of one-sided normals that crack the shading along
    /// seams. Manual normalization instead of Vec3::normalize because accumulated normals
    /// on tiny triangles can fall under normalize()'s 1e-6 length-squared gate and would
    /// get zeroed (black shading).
    inline void recomputeOrigNormals(DNA::GeometryDetail& geom) {
        const size_t vc = geom.get_vertex_count();
        const Vec3* Po = geom.get_attribute_data<Vec3>("P_orig");
        Vec3* No = geom.get_attribute_data_mut<Vec3>("N_orig");
        if (!Po || !No || vc == 0) return;

        const std::vector<uint32_t> remap = buildCoincidentRemap(Po, vc);
        std::vector<Vec3> acc(vc, Vec3(0.0f, 0.0f, 0.0f));
        const auto& idx = geom.indices;
        for (size_t i = 0; i + 2 < idx.size(); i += 3) {
            const uint32_t a = idx[i], b = idx[i + 1], c = idx[i + 2];
            if (a >= vc || b >= vc || c >= vc) continue;
            const Vec3 fn = Vec3::cross(Po[b] - Po[a], Po[c] - Po[a]);  // length ∝ 2*area
            acc[remap[a]] = acc[remap[a]] + fn;
            acc[remap[b]] = acc[remap[b]] + fn;
            acc[remap[c]] = acc[remap[c]] + fn;
        }
        for (size_t v = 0; v < vc; ++v) {
            const Vec3& s = acc[remap[v]];
            const float len = std::sqrt(s.x * s.x + s.y * s.y + s.z * s.z);
            if (len > 1e-20f) {
                No[v] = Vec3(s.x / len, s.y / len, s.z / len);
            } else {
                No[v] = Vec3(0.0f, 1.0f, 0.0f);
            }
        }
    }

    /// Get-or-create a per-vertex float attribute (a "Field" in Faz 8b terms — fields
    /// are NOT a separate socket type, they ride the Geometry wire as GeometryDetail
    /// named attributes, exactly the Houdini model the Faz 8 plan chose).
    inline float* ensureFloatAttribute(DNA::GeometryDetail& geom, const std::string& name) {
        if (name.empty()) return nullptr;
        if (!geom.has_attribute(name)) geom.add_attribute<float>(name);
        return geom.get_attribute_data_mut<float>(name);
    }

    /**
     * @brief Append src's geometry into dst (vertices + offset indices), optionally
     * re-expressing src's local-space P_orig/N_orig through relPoint/relNormal (used by
     * Merge to bring another object's local space into dst's pivot space). dst must be a
     * deep copy the caller owns. P/N tails are left stale — callers rebakeFromOrig() after.
     */
    inline bool appendMeshInto(TriangleMesh& dst, const TriangleMesh& src,
                               const Matrix4x4* relPoint, const Matrix4x4* relNormal) {
        if (!dst.geometry || !src.geometry) return false;
        auto& dg = *dst.geometry;
        const auto& sg = *src.geometry;
        const size_t vcA = dg.get_vertex_count();
        const size_t vcB = sg.get_vertex_count();
        if (vcB == 0 || sg.indices.empty()) return false;

        // Core channels src carries but dst doesn't must exist BEFORE the resize so their
        // buffers get sized with everything else (add_attribute allocates zero-filled).
        if (sg.has_attribute("P") && !dg.has_attribute("P")) dg.add_attribute<Vec3>("P");
        if (sg.has_attribute("N") && !dg.has_attribute("N")) dg.add_attribute<Vec3>("N");
        if (sg.has_attribute("P_orig") && !dg.has_attribute("P_orig")) dg.add_attribute<Vec3>("P_orig");
        if (sg.has_attribute("N_orig") && !dg.has_attribute("N_orig")) dg.add_attribute<Vec3>("N_orig");
        if (sg.has_attribute("uv") && !dg.has_attribute("uv")) dg.add_attribute<Vec2>("uv");
        if (sg.has_attribute("materialID") && !dg.has_attribute("materialID")) dg.add_attribute<uint16_t>("materialID");

        dg.resize_vertices(vcA + vcB);

        // P_orig / N_orig — the authoritative local channels (P/N rebaked by the caller).
        {
            const Vec3* sPo = sg.get_attribute_data<Vec3>("P_orig");
            Vec3* dPo = dg.get_attribute_data_mut<Vec3>("P_orig");
            if (sPo && dPo) {
                for (size_t v = 0; v < vcB; ++v) {
                    dPo[vcA + v] = relPoint ? relPoint->transform_point(sPo[v]) : sPo[v];
                }
            }
            const Vec3* sNo = sg.get_attribute_data<Vec3>("N_orig");
            Vec3* dNo = dg.get_attribute_data_mut<Vec3>("N_orig");
            if (sNo && dNo) {
                for (size_t v = 0; v < vcB; ++v) {
                    dNo[vcA + v] = relNormal ? relNormal->transform_vector(sNo[v]).normalize() : sNo[v];
                }
            }
        }
        {
            const Vec2* sUv = sg.get_attribute_data<Vec2>("uv");
            Vec2* dUv = dg.get_attribute_data_mut<Vec2>("uv");
            if (sUv && dUv) for (size_t v = 0; v < vcB; ++v) dUv[vcA + v] = sUv[v];
        }
        {
            const uint16_t* sMat = sg.get_material_ids();
            uint16_t* dMat = dg.get_attribute_data_mut<uint16_t>("materialID");
            if (sMat && dMat) for (size_t v = 0; v < vcB; ++v) dMat[vcA + v] = sMat[v];
        }

        dg.indices.reserve(dg.indices.size() + sg.indices.size());
        for (const uint32_t i : sg.indices) {
            dg.indices.push_back(static_cast<uint32_t>(vcA + i));
        }

        if (!dg.skin_weights.empty() || !sg.skin_weights.empty()) {
            dg.skin_weights.resize(vcA + vcB);
            for (size_t v = 0; v < vcB && v < sg.skin_weights.size(); ++v) {
                dg.skin_weights[vcA + v] = sg.skin_weights[v];
            }
        }
        return true;
    }

    /**
     * @brief Project a planar (or near-planar) polygon's vertices into a local 2D frame
     * centered on its centroid and normalized to 0..1 — used to give topology-changing
     * ops (Extrude/Inset) SOME uv instead of leaving new faces unwrapped. Same formula
     * the facade-based Edit Mode uses for Inset/Extrude/Bridge/Dissolve
     * (scene_ui_mesh_overlay.cpp's buildPolygonPlanarUVs), duplicated here rather than
     * shared across a UI<->header include boundary — this header has no UI dependency.
     */
    inline std::vector<Vec2> buildFacePlanarUVs(const std::vector<Vec3>& vertices, const Vec3& normal) {
        std::vector<Vec2> uvs(vertices.size(), Vec2(0.0f, 0.0f));
        if (vertices.size() < 3) return uvs;

        Vec3 center(0.0f, 0.0f, 0.0f);
        for (const Vec3& v : vertices) center = center + v;
        center = center * (1.0f / static_cast<float>(vertices.size()));

        Vec3 axisX = vertices[0] - center;
        axisX = (axisX.length_squared() > 1e-10f) ? axisX.normalize() : Vec3(1.0f, 0.0f, 0.0f);
        Vec3 axisY = normal.cross(axisX);
        axisY = (axisY.length_squared() > 1e-10f) ? axisY.normalize() : Vec3(0.0f, 1.0f, 0.0f);

        float minU = (std::numeric_limits<float>::max)(), minV = (std::numeric_limits<float>::max)();
        float maxU = -(std::numeric_limits<float>::max)(), maxV = -(std::numeric_limits<float>::max)();
        std::vector<std::pair<float, float>> proj;
        proj.reserve(vertices.size());
        for (const Vec3& v : vertices) {
            const Vec3 d = v - center;
            const float u = d.dot(axisX);
            const float w = d.dot(axisY);
            proj.emplace_back(u, w);
            minU = (std::min)(minU, u); maxU = (std::max)(maxU, u);
            minV = (std::min)(minV, w); maxV = (std::max)(maxV, w);
        }
        const float spanU = (std::max)(maxU - minU, 1e-4f);
        const float spanV = (std::max)(maxV - minV, 1e-4f);
        for (size_t i = 0; i < proj.size(); ++i) {
            uvs[i] = Vec2((proj[i].first - minU) / spanU, (proj[i].second - minV) / spanV);
        }
        return uvs;
    }

    /**
     * @brief Bridge from the flat SoA TriangleMesh's per-face-corner triangle soup to
     * MeshEdit::HalfEdgeMesh's connected (welded) topology — the same MeshEdit core the
     * facade-based Edit Mode already uses for Inset/Extrude/loop-cut/dissolve, now made
     * usable from a Geo-DAG node instead of the interactive editor. The flat soup often
     * duplicates a position across triangle corners (UV seams, per-face-soup imports),
     * so raw indices alone don't describe real face adjacency — buildCoincidentRemap
     * (the same bit-identical clustering NoiseDisplace/recomputeOrigNormals rely on)
     * welds by position first. Degenerate triangles (a coincident-clustered pair of
     * corners) are dropped up front, BEFORE calling buildFromPolygons, so its internal
     * face ids stay 1:1 with faceToTriangle (buildFromPolygons's own skip counter would
     * otherwise desync a naive index mapping).
     */
    struct FlatHalfEdgeBridge {
        MeshEdit::HalfEdgeMesh heMesh;
        MeshEdit::HalfEdgeBuildResult buildResult;
        std::vector<uint32_t> faceToTriangle;  ///< heMesh face id -> original flat-mesh triangle index
        std::vector<int32_t> sourceToWelded;   ///< source flat-mesh vertex -> heMesh (welded) vertex id
                                               ///< (lets nodes transfer per-vertex attributes, e.g. masks)
    };

    inline bool buildHalfEdgeFromFlatMesh(const TriangleMesh& mesh, FlatHalfEdgeBridge& bridge) {
        if (!mesh.geometry) return false;
        const auto& geom = *mesh.geometry;
        const Vec3* Po = geom.get_attribute_data<Vec3>("P_orig");
        const size_t vc = geom.get_vertex_count();
        if (!Po || vc == 0 || geom.indices.size() < 3) return false;

        const std::vector<uint32_t> remap = buildCoincidentRemap(Po, vc);
        std::vector<int32_t> weldedOf(vc, -1);
        std::vector<Vec3> weldedPositions;
        weldedPositions.reserve(vc);
        for (size_t v = 0; v < vc; ++v) {
            if (remap[v] == v) {
                weldedOf[v] = static_cast<int32_t>(weldedPositions.size());
                weldedPositions.push_back(Po[v]);
            }
        }
        bridge.sourceToWelded.resize(vc);
        for (size_t v = 0; v < vc; ++v) {
            bridge.sourceToWelded[v] = weldedOf[remap[v]];
        }

        std::vector<std::vector<int>> polygons;
        const size_t nTris = geom.indices.size() / 3;
        polygons.reserve(nTris);
        bridge.faceToTriangle.clear();
        bridge.faceToTriangle.reserve(nTris);
        for (size_t f = 0; f < nTris; ++f) {
            const uint32_t a = remap[geom.indices[f * 3 + 0]];
            const uint32_t b = remap[geom.indices[f * 3 + 1]];
            const uint32_t c = remap[geom.indices[f * 3 + 2]];
            if (a == b || b == c || a == c) continue;  // degenerate after weld — drop, keep ids aligned
            polygons.push_back({ weldedOf[a], weldedOf[b], weldedOf[c] });
            bridge.faceToTriangle.push_back(static_cast<uint32_t>(f));
        }
        const bool built = bridge.heMesh.buildFromPolygons(weldedPositions, polygons, &bridge.buildResult);
        return built && bridge.buildResult.skipped_polygons == 0;
    }

    /// Transfer a per-vertex float Field (mask) from the source flat mesh onto the
    /// bridge's WELDED vertex set, averaging over each coincident cluster — the same
    /// cluster-averaging rule NoiseDisplace uses (seam duplicates carry different
    /// per-side values; averaging keeps the result consistent per logical vertex).
    /// Returns false when the attribute doesn't exist on the source mesh.
    inline bool buildWeldedVertexMask(const TriangleMesh& mesh, const FlatHalfEdgeBridge& bridge,
                                      const char* attrName, std::vector<float>& outMask) {
        const float* attr = mesh.geometry ? mesh.geometry->get_attribute_data<float>(attrName) : nullptr;
        if (!attr) return false;
        const size_t vc = mesh.geometry->get_vertex_count();
        if (bridge.sourceToWelded.size() < vc) return false;
        outMask.assign(bridge.heMesh.vertices.size(), 0.0f);
        std::vector<uint32_t> cnt(bridge.heMesh.vertices.size(), 0u);
        for (size_t v = 0; v < vc; ++v) {
            const int32_t w = bridge.sourceToWelded[v];
            if (w < 0 || static_cast<size_t>(w) >= outMask.size()) continue;
            outMask[w] += attr[v];
            cnt[w] += 1u;
        }
        for (size_t w = 0; w < outMask.size(); ++w) {
            if (cnt[w]) outMask[w] /= static_cast<float>(cnt[w]);
        }
        return true;
    }

    /// Average a welded-vertex mask over one half-edge face's vertices — the per-FACE
    /// mask value gating/scaling per-face ops (Extrude/Inset). Valid after quad merge
    /// too: dissolveEdge keeps vertex ids stable, only face membership changes.
    inline float faceMaskAverage(const MeshEdit::HalfEdgeMesh& he, MeshEdit::HEIndex f,
                                 const std::vector<float>& weldedMask,
                                 std::vector<MeshEdit::HEIndex>& scratchVerts) {
        he.collectFaceVertices(f, scratchVerts);
        if (scratchVerts.empty()) return 0.0f;
        float sum = 0.0f;
        for (const MeshEdit::HEIndex v : scratchVerts) {
            if (v >= 0 && static_cast<size_t>(v) < weldedMask.size()) sum += weldedMask[v];
        }
        return sum / static_cast<float>(scratchVerts.size());
    }

    /**
     * @brief Inverse of buildHalfEdgeFromFlatMesh: fan-triangulate every surviving
     * half-edge face back into a flat TriangleMesh via Triangle facades + the same
     * MeshModifiers::facadesToFlatMesh conversion SubdivideCC's Flat mode already uses.
     * Every new face gets flat-shaded normals + a planar UV (buildFacePlanarUVs) — the
     * facade Edit Mode's Inset/Extrude do the same for the same reason: post-topology-
     * change faces have no natural smooth-shading or unwrap information to inherit.
     * materialID is inherited per-face by walking faceSourceMap back to an ORIGINAL
     * (pre-op) face id (extrudeFace/insetFace keep the top/center face's id; new side
     * faces are recorded in faceSourceMap by the caller), then faceToTriangle resolves
     * that to the source flat mesh's triangle to read its materialID.
     */
    inline std::shared_ptr<TriangleMesh> rebuildFlatMeshFromHalfEdge(
        const MeshEdit::HalfEdgeMesh& heMesh,
        const TriangleMesh& sourceMesh,
        const std::vector<uint32_t>& faceToTriangle,
        MeshEdit::HEIndex originalFaceCount,
        const std::unordered_map<MeshEdit::HEIndex, MeshEdit::HEIndex>& faceSourceMap) {

        auto resolveSourceTriangle = [&](MeshEdit::HEIndex f) -> int {
            const size_t guard = faceSourceMap.size() + 1;
            for (size_t step = 0; step < guard && f >= originalFaceCount; ++step) {
                const auto it = faceSourceMap.find(f);
                if (it == faceSourceMap.end()) break;
                f = it->second;
            }
            return (f >= 0 && f < originalFaceCount && static_cast<size_t>(f) < faceToTriangle.size())
                ? static_cast<int>(faceToTriangle[f]) : -1;
        };

        const auto& sg = *sourceMesh.geometry;
        const uint16_t* srcMat = sg.get_material_ids();

        std::vector<std::shared_ptr<Triangle>> facades;
        facades.reserve(heMesh.liveFaceCount() * 2);
        std::vector<MeshEdit::HEIndex> faceVerts;
        for (MeshEdit::HEIndex f = 0; f < static_cast<MeshEdit::HEIndex>(heMesh.faces.size()); ++f) {
            if (heMesh.faces[f].removed) continue;
            heMesh.collectFaceVertices(f, faceVerts);
            if (faceVerts.size() < 3) continue;

            std::vector<Vec3> pos;
            pos.reserve(faceVerts.size());
            for (const MeshEdit::HEIndex v : faceVerts) pos.push_back(heMesh.vertices[v].position);
            Vec3 n = heMesh.faceNormal(f);
            if (n.length_squared() <= 1e-10f) n = Vec3(0.0f, 1.0f, 0.0f);
            const std::vector<Vec2> uvs = buildFacePlanarUVs(pos, n);

            uint16_t matID = 0;
            const int srcTri = resolveSourceTriangle(f);
            if (srcMat && srcTri >= 0) matID = srcMat[sg.indices[static_cast<size_t>(srcTri) * 3]];

            for (size_t i = 1; i + 1 < pos.size(); ++i) {
                facades.push_back(std::make_shared<Triangle>(
                    pos[0], pos[i], pos[i + 1],
                    n, n, n,
                    uvs[0], uvs[i], uvs[i + 1],
                    matID));
            }
        }
        if (facades.empty()) return nullptr;
        auto out = MeshModifiers::facadesToFlatMesh(facades);
        // Standalone facades carry no parentMesh, so facadesToFlatMesh cannot recover the
        // object's transform (it bakes P with identity). Re-attach the source transform and
        // rebake P/N from P_orig/N_orig so the world-baked channels match the pivot again.
        if (out && !out->transform && sourceMesh.transform) {
            out->transform = std::make_shared<Transform>(*sourceMesh.transform);
            rebakeFromOrig(*out);
        }
        return out;
    }

    /**
     * @brief Re-derive quad (or larger source n-gon) faces that the triangulated flat
     * soup lost, so per-face ops (Extrude/Inset) treat "one quad" as one face instead of
     * creasing/splitting along the invisible triangulation diagonal — the facade Edit
     * Mode gets this for free from EditableMeshCache's recorded polygon_faces; the flat
     * SoA has no such record, only individual triangles, so it has to be RE-DETECTED.
     *
     * Pairwise ONLY: each triangle merges with AT MOST one neighbor (the first coplanar
     * un-merged twin found across its 3 edges), never chained further. This matters
     * because a fully flat, coplanar surface built from MANY quads (e.g. a subdivided
     * ground plane) would otherwise collapse into ONE giant face if merging kept
     * absorbing every coplanar neighbor transitively — pairwise merging only removes the
     * ONE diagonal edge each source quad's triangulation added, leaving real quad-to-quad
     * boundaries alone. Reuses dissolveEdge (existing Euler op, same one the manual
     * Dissolve Edge tool uses) — it already refuses bridges/non-manifold cases on its
     * own, so a bad candidate is simply skipped rather than force-merged.
     */
    inline int mergeCoplanarTrianglePairs(MeshEdit::HalfEdgeMesh& he, float minNormalDot = 0.999f) {
        int merged = 0;
        std::vector<bool> done(he.faces.size(), false);
        std::vector<MeshEdit::HEIndex> faceHalfEdges;
        for (MeshEdit::HEIndex f = 0; f < static_cast<MeshEdit::HEIndex>(he.faces.size()); ++f) {
            if (he.faces[f].removed || done[f]) continue;
            if (he.faceVertexCount(f) != 3) { done[f] = true; continue; }
            const Vec3 nf = he.faceNormal(f);
            if (nf.length_squared() <= 1e-10f) { done[f] = true; continue; }

            he.collectFaceHalfEdges(f, faceHalfEdges);
            MeshEdit::HEIndex bestEdge = MeshEdit::kHEInvalid;
            MeshEdit::HEIndex bestPartner = MeshEdit::kHEInvalid;
            for (const MeshEdit::HEIndex heIdx : faceHalfEdges) {
                const MeshEdit::HEIndex twin = he.half_edges[heIdx].twin;
                if (he.isBoundaryHalfEdge(twin)) continue;
                const MeshEdit::HEIndex g = he.half_edges[twin].face;
                if (g == MeshEdit::kHEInvalid || g == f || done[g] || he.faces[g].removed) continue;
                if (he.faceVertexCount(g) != 3) continue;
                const Vec3 ng = he.faceNormal(g);
                if (ng.length_squared() <= 1e-10f) continue;
                if (nf.dot(ng) < minNormalDot) continue;
                bestEdge = he.half_edges[heIdx].edge;
                bestPartner = g;
                break;
            }
            done[f] = true;
            if (bestEdge == MeshEdit::kHEInvalid) continue;
            done[bestPartner] = true;  // claimed even if dissolveEdge below rejects it
            if (he.dissolveEdge(bestEdge) != MeshEdit::kHEInvalid) ++merged;
        }
        return merged;
    }

    // ============================================================================
    // BEVEL CORE (edge chamfer/round over a half-edge mesh)
    // ============================================================================

    /// One output polygon of a bevel pass. `sourceFace` is the half-edge face the
    /// polygon inherits material/template attributes from (a shrunk face inherits
    /// itself; a chamfer band and a vertex patch inherit an adjacent source face).
    struct BevelPolygon {
        std::vector<Vec3> pts;          ///< Ordered loop, outward winding already fixed
        Vec3 normal{ 0.0f, 1.0f, 0.0f };///< Representative flat normal
        /// Optional per-point smooth normals (same size/order as pts when present).
        /// Round-profile strip bands fill these with faceA->faceB blended normals so
        /// the bevel shades as one continuous curve — with flat per-band normals the
        /// facet crease between bands sits exactly where the original sharp edge was,
        /// visually reading as "the edge is still there" and hiding the bevel.
        std::vector<Vec3> ptNormals;
        MeshEdit::HEIndex sourceFace = MeshEdit::kHEInvalid;
    };

    /// Flag every interior manifold edge whose dihedral angle (between face normals)
    /// is >= angleThresholdDeg — Blender Bevel's Limit Method = Angle. Boundary and
    /// non-manifold edges never qualify; coplanar edges (quad-merge leftovers, ngon
    /// fan diagonals) naturally fall below any sane threshold. Returns the count.
    inline size_t selectBevelEdgesByAngle(const MeshEdit::HalfEdgeMesh& he,
                                          float angleThresholdDeg,
                                          std::vector<bool>& outFlags) {
        const size_t nEdges = he.edges.size();
        outFlags.assign(nEdges, false);
        const float cosThresh = std::cos(angleThresholdDeg * 3.14159265358979f / 180.0f);
        size_t count = 0;
        for (MeshEdit::HEIndex e = 0; e < static_cast<MeshEdit::HEIndex>(nEdges); ++e) {
            if (he.edges[e].removed) continue;
            const MeshEdit::HEIndex h0 = he.edges[e].half_edge;
            const MeshEdit::HEIndex h1 = he.half_edges[h0].twin;
            if (he.isBoundaryHalfEdge(h0) || he.isBoundaryHalfEdge(h1)) continue;
            const MeshEdit::HEIndex fA = he.half_edges[h0].face;
            const MeshEdit::HEIndex fB = he.half_edges[h1].face;
            if (fA == MeshEdit::kHEInvalid || fB == MeshEdit::kHEInvalid) continue;
            if (he.faceNormal(fA).dot(he.faceNormal(fB)) <= cosThresh) {
                outFlags[e] = true;
                ++count;
            }
        }
        return count;
    }

    /**
     * @brief Edge bevel over an EXPLICIT edge set: every flagged interior edge is
     * replaced by a chamfer strip of `segments` bands (roundProfile sweeps the bands
     * along a circular arc around the original edge — Blender's default profile 0.5;
     * flat keeps them linear = classic multi-cut chamfer), with n-gon patches closing
     * the vertex gaps. Construction is a full REBUILD (not incremental Euler surgery):
     *
     *  1. Per face, each beveled edge's line is shifted inward (in the face plane,
     *     perpendicular to the edge) by `width`; each face corner is re-derived as the
     *     intersection of its two (possibly shifted) edge lines. Corner displacement is
     *     CLAMPED to half the shorter adjacent edge — the guard that keeps a second
     *     bevel pass (or too-large width on small faces) from exploding the mesh
     *     instead of just flattening out (Blender's Clamp Overlap equivalent).
     *  2. Per beveled edge, the two rails plus the swept arc between them become
     *     `segments` quad bands.
     *  3. Per vertex touching >= 1 beveled edge, the shrunk face corners AND the strip
     *     end-arc points around it (in circulation order) close the gap as one n-gon.
     *
     * Winding of every emitted polygon is fixed by a Newell-normal check against the
     * adjacent source-face normals (robust regardless of circulation direction).
     * Callers select edges first (selectBevelEdgesByAngle, or an edit-mode selection)
     * and materialize the returned polygons themselves (facades for the Geo-DAG node,
     * template-cloned Triangles for edit mode). Boundary edges/vertices are skipped.
     */
    inline std::vector<BevelPolygon> bevelEdgesPolygons(
        const MeshEdit::HalfEdgeMesh& he,
        const std::vector<bool>& bevelEdge,
        float width,
        int segments,
        bool roundProfile) {

        std::vector<BevelPolygon> out;
        segments = (std::max)(1, (std::min)(segments, 16));
        const size_t nFaces = he.faces.size();
        const size_t nHalf = he.half_edges.size();
        const size_t nEdges = he.edges.size();
        if (width <= 0.0f || nFaces == 0 || nHalf == 0 || bevelEdge.size() < nEdges) return out;

        // Sanitize the selection: boundary/removed edges can't be beveled (only one
        // rail exists). Cleared HERE — not just skipped later — because the corner
        // shift (step 2) also reads these flags, and a shifted-but-stripless edge
        // would open a hole along the border.
        std::vector<bool> flags = bevelEdge;
        size_t bevelCount = 0;
        for (MeshEdit::HEIndex e = 0; e < static_cast<MeshEdit::HEIndex>(nEdges); ++e) {
            if (!flags[e]) continue;
            if (he.edges[e].removed || he.isBoundaryEdge(e)) {
                flags[e] = false;
                continue;
            }
            ++bevelCount;
        }
        if (bevelCount == 0) return out;

        // Cached face normals (Newell over the source topology).
        std::vector<Vec3> fN(nFaces, Vec3(0.0f, 1.0f, 0.0f));
        for (MeshEdit::HEIndex f = 0; f < static_cast<MeshEdit::HEIndex>(nFaces); ++f) {
            if (!he.faces[f].removed) fN[f] = he.faceNormal(f);
        }

        // 2. New corner per interior half-edge: intersection of its face's two adjacent
        // (possibly inward-shifted) edge lines at the half-edge's ORIGIN vertex, with
        // the displacement clamped (see doc above).
        auto shiftedLine = [&](MeshEdit::HEIndex h, Vec3& a, Vec3& d) {
            const MeshEdit::HEIndex hn = he.half_edges[h].next;
            a = he.vertices[he.half_edges[h].origin].position;
            const Vec3 b = he.vertices[he.half_edges[hn].origin].position;
            d = b - a;
            if (flags[he.half_edges[h].edge]) {
                const float len = std::sqrt(d.length_squared());
                if (len > 1e-12f) {
                    const Vec3 dir = d * (1.0f / len);
                    Vec3 inward = fN[he.half_edges[h].face].cross(dir);
                    const float il = std::sqrt(inward.length_squared());
                    if (il > 1e-12f) a = a + inward * (width / il);
                }
            }
        };
        std::vector<Vec3> corner(nHalf);
        for (MeshEdit::HEIndex h = 0; h < static_cast<MeshEdit::HEIndex>(nHalf); ++h) {
            if (he.half_edges[h].removed) continue;
            const Vec3 vpos = he.vertices[he.half_edges[h].origin].position;
            corner[h] = vpos;
            if (he.half_edges[h].face == MeshEdit::kHEInvalid) continue;  // boundary halves keep v
            const MeshEdit::HEIndex hp = he.half_edges[h].prev;
            const bool bevPrev = flags[he.half_edges[hp].edge];
            const bool bevThis = flags[he.half_edges[h].edge];
            if (!bevPrev && !bevThis) continue;  // untouched corner
            Vec3 a1, d1, a2, d2;
            shiftedLine(hp, a1, d1);
            shiftedLine(h, a2, d2);
            const Vec3 n = Vec3::cross(d1, d2);
            const float n2 = n.length_squared();
            if (n2 > 1e-12f * d1.length_squared() * d2.length_squared()) {
                // In-plane line-line intersection: point = a2 + d2 * s.
                // From a1 + t*d1 = a2 + s*d2, crossing both sides with d1:
                // s = [(a2-a1) x d1] . (d1 x d2) / |d1 x d2|^2. Getting (a1-a2) here
                // (the first shipped version) mirrors every corner to the WRONG side of
                // its vertex along the edge — user-visible as each face looking rotated
                // in place and every chamfer strip twisting.
                const float s = Vec3::cross(a2 - a1, d1).dot(n) / n2;
                corner[h] = a2 + d2 * s;
            } else {
                // Near-parallel corner (straight vertex on a beveled edge chain): both
                // shifted lines are (nearly) the same line, so the corner is v moved by
                // the AVERAGE applied shift — summing both would double the offset on a
                // collinear chain where the two shifts are identical.
                Vec3 shiftSum(0.0f, 0.0f, 0.0f);
                int shiftCount = 0;
                if (bevPrev) { shiftSum = shiftSum + (a1 - he.vertices[he.half_edges[hp].origin].position); ++shiftCount; }
                if (bevThis) { shiftSum = shiftSum + (a2 - vpos); ++shiftCount; }
                corner[h] = (shiftCount > 0) ? vpos + shiftSum * (1.0f / shiftCount) : vpos;
            }
            // Clamp Overlap: cap the corner's displacement at half the shorter adjacent
            // edge, so opposite ends of a short edge can never cross — a second bevel
            // pass over the (narrow) strips of a first one degenerates gracefully to a
            // flat limit instead of exploding the mesh (user-reported: double bevel in
            // the stack made the object vanish — ill-conditioned intersections threw
            // corners far outside the model).
            const Vec3 off = corner[h] - vpos;
            const float off2 = off.length_squared();
            const float maxOff = 0.5f * std::sqrt((std::min)(d1.length_squared(), d2.length_squared()));
            if (off2 > maxOff * maxOff && off2 > 1e-20f) {
                corner[h] = vpos + off * (maxOff / std::sqrt(off2));
            }
        }

        auto newellNormal = [](const std::vector<Vec3>& pts) -> Vec3 {
            Vec3 n(0.0f, 0.0f, 0.0f);
            for (size_t i = 0; i < pts.size(); ++i) {
                const Vec3& a = pts[i];
                const Vec3& b = pts[(i + 1) % pts.size()];
                n.x += (a.y - b.y) * (a.z + b.z);
                n.y += (a.z - b.z) * (a.x + b.x);
                n.z += (a.x - b.x) * (a.y + b.y);
            }
            return n;
        };
        auto emitPolygon = [&](const std::vector<Vec3>& pts, const Vec3& n, MeshEdit::HEIndex srcFace) {
            BevelPolygon poly;
            poly.pts = pts;
            poly.normal = n;
            poly.sourceFace = srcFace;
            out.push_back(std::move(poly));
        };

        // 3. Shrunk face polygons.
        std::vector<MeshEdit::HEIndex> faceHEs;
        std::vector<Vec3> pts;
        for (MeshEdit::HEIndex f = 0; f < static_cast<MeshEdit::HEIndex>(nFaces); ++f) {
            if (he.faces[f].removed) continue;
            he.collectFaceHalfEdges(f, faceHEs);
            if (faceHEs.size() < 3) continue;
            pts.clear();
            for (const MeshEdit::HEIndex h : faceHEs) pts.push_back(corner[h]);
            emitPolygon(pts, fN[f], f);
        }

        // 4. Chamfer strips — `segments` bands swept between the two rails. End arcs
        // are cached per edge because the vertex patches (step 5) must reuse the EXACT
        // same points, or the strip and the patch crack apart.
        auto rotateAroundAxis = [](const Vec3& v, const Vec3& axis, float ang) -> Vec3 {
            const float c = std::cos(ang);
            const float s = std::sin(ang);
            return v * c + Vec3::cross(axis, v) * s + axis * (axis.dot(v) * (1.0f - c));
        };
        // arc from pFrom to pTo around the edge LINE (origin `lineO`, unit dir `lineD`):
        // roundProfile sweeps radially (circular profile), flat lerps.
        auto buildArc = [&](const Vec3& pFrom, const Vec3& pTo,
                            const Vec3& lineO, const Vec3& lineD,
                            std::vector<Vec3>& arc) {
            arc.assign(static_cast<size_t>(segments) + 1, pFrom);
            arc[static_cast<size_t>(segments)] = pTo;
            if (segments == 1) return;
            const Vec3 cF = lineO + lineD * lineD.dot(pFrom - lineO);
            const Vec3 cT = lineO + lineD * lineD.dot(pTo - lineO);
            const Vec3 rF = pFrom - cF;
            const Vec3 rT = pTo - cT;
            const float lenF = std::sqrt(rF.length_squared());
            const float lenT = std::sqrt(rT.length_squared());
            bool round = roundProfile && lenF > 1e-9f && lenT > 1e-9f;
            float signedAng = 0.0f;
            Vec3 uF(0.0f, 0.0f, 0.0f);
            if (round) {
                uF = rF * (1.0f / lenF);
                const Vec3 uT = rT * (1.0f / lenT);
                const float d = (std::max)(-1.0f, (std::min)(1.0f, uF.dot(uT)));
                const float ang = std::acos(d);
                const float sgn = Vec3::cross(uF, uT).dot(lineD);
                signedAng = (sgn >= 0.0f) ? ang : -ang;
                if (std::fabs(signedAng) < 1e-4f) round = false;  // rails parallel — lerp
            }
            for (int k = 1; k < segments; ++k) {
                const float t = static_cast<float>(k) / static_cast<float>(segments);
                if (round) {
                    const Vec3 c = cF + (cT - cF) * t;
                    const float r = lenF + (lenT - lenF) * t;
                    arc[static_cast<size_t>(k)] = c + rotateAroundAxis(uF, lineD, signedAng * t) * r;
                } else {
                    arc[static_cast<size_t>(k)] = pFrom + (pTo - pFrom) * t;
                }
            }
        };
        // Per beveled edge: [0] = arc at v0 (origin of the canonical half), [1] = arc
        // at v1; both ordered from the face-A rail (canonical half's face) to face B.
        std::unordered_map<MeshEdit::HEIndex, std::array<std::vector<Vec3>, 2>> endArcs;
        endArcs.reserve(bevelCount);
        for (MeshEdit::HEIndex e = 0; e < static_cast<MeshEdit::HEIndex>(nEdges); ++e) {
            if (!flags[e]) continue;
            const MeshEdit::HEIndex h0 = he.edges[e].half_edge;
            const MeshEdit::HEIndex h1 = he.half_edges[h0].twin;
            const MeshEdit::HEIndex fA = he.half_edges[h0].face;
            const MeshEdit::HEIndex fB = he.half_edges[h1].face;
            // Boundary edges can't be beveled (only one rail exists). Guarded here, not
            // just in the angle selector — edit mode passes an explicit selection that
            // may legitimately include boundary edges.
            if (fA == MeshEdit::kHEInvalid || fB == MeshEdit::kHEInvalid) continue;
            const Vec3 v0 = he.vertices[he.half_edges[h0].origin].position;
            const Vec3 v1 = he.vertices[he.half_edges[h1].origin].position;
            Vec3 axis = v1 - v0;
            const float axLen = std::sqrt(axis.length_squared());
            if (axLen <= 1e-12f) continue;
            axis = axis * (1.0f / axLen);

            auto& arcs = endArcs[e];
            buildArc(corner[h0], corner[he.half_edges[h1].next], v0, axis, arcs[0]);          // at v0: A -> B
            buildArc(corner[he.half_edges[h0].next], corner[h1], v0, axis, arcs[1]);          // at v1: A -> B

            // Winding probe on the first band, applied to every band of this strip.
            pts.clear();
            pts.push_back(arcs[0][0]);
            pts.push_back(arcs[1][0]);
            pts.push_back(arcs[1][1]);
            pts.push_back(arcs[0][1]);
            const bool flip = newellNormal(pts).dot(fN[fA] + fN[fB]) < 0.0f;

            // Round profile: per-ROW smooth normals blended from face A's normal to
            // face B's across the strip, so the whole bevel shades as one continuous
            // curve (Blender bevel + Shade Smooth look). Flat per-band normals put a
            // facet crease exactly where the original sharp edge was — geometrically
            // beveled but visually still reading as the old hard edge. Flat profile
            // keeps flat bands (a faceted multi-cut chamfer is the POINT of that mode).
            auto rowNormal = [&](int k) -> Vec3 {
                const float t = static_cast<float>(k) / static_cast<float>(segments);
                Vec3 n = fN[fA] * (1.0f - t) + fN[fB] * t;
                const float l = std::sqrt(n.length_squared());
                return (l > 1e-12f) ? n * (1.0f / l) : fN[fA];
            };
            for (int k = 0; k < segments; ++k) {
                pts.clear();
                std::vector<Vec3> ptNs;
                if (!flip) {
                    pts.push_back(arcs[0][static_cast<size_t>(k)]);
                    pts.push_back(arcs[1][static_cast<size_t>(k)]);
                    pts.push_back(arcs[1][static_cast<size_t>(k) + 1]);
                    pts.push_back(arcs[0][static_cast<size_t>(k) + 1]);
                    if (roundProfile) {
                        ptNs = { rowNormal(k), rowNormal(k), rowNormal(k + 1), rowNormal(k + 1) };
                    }
                } else {
                    pts.push_back(arcs[0][static_cast<size_t>(k) + 1]);
                    pts.push_back(arcs[1][static_cast<size_t>(k) + 1]);
                    pts.push_back(arcs[1][static_cast<size_t>(k)]);
                    pts.push_back(arcs[0][static_cast<size_t>(k)]);
                    if (roundProfile) {
                        ptNs = { rowNormal(k + 1), rowNormal(k + 1), rowNormal(k), rowNormal(k) };
                    }
                }
                Vec3 bn = newellNormal(pts);
                const float bl = std::sqrt(bn.length_squared());
                if (bl <= 1e-12f) continue;
                emitPolygon(pts, bn * (1.0f / bl), fA);
                if (!ptNs.empty()) out.back().ptNormals = std::move(ptNs);
            }
        }

        // 5. Vertex patches — close the gap around each vertex touching a beveled
        // edge: shrunk face corners plus the strips' end-arc interior points, in
        // circulation order.
        std::vector<MeshEdit::HEIndex> outgoing;
        for (MeshEdit::HEIndex v = 0; v < static_cast<MeshEdit::HEIndex>(he.vertices.size()); ++v) {
            if (he.vertices[v].removed || he.vertices[v].half_edge == MeshEdit::kHEInvalid) continue;
            if (he.isBoundaryVertex(v)) continue;  // v1: open borders keep their corner
            he.collectVertexOutgoing(v, outgoing);
            bool touched = false;
            for (const MeshEdit::HEIndex h : outgoing) {
                if (flags[he.half_edges[h].edge]) { touched = true; break; }
            }
            if (!touched) continue;

            pts.clear();
            Vec3 avgN(0.0f, 0.0f, 0.0f);
            MeshEdit::HEIndex srcFace = MeshEdit::kHEInvalid;
            auto pushPoint = [&](const Vec3& p) {
                if (!pts.empty() && (p - pts.back()).length_squared() <= 1e-20f) return;
                pts.push_back(p);
            };
            for (const MeshEdit::HEIndex h : outgoing) {
                if (he.half_edges[h].face != MeshEdit::kHEInvalid) {
                    pushPoint(corner[h]);
                    avgN = avgN + fN[he.half_edges[h].face];
                    if (srcFace == MeshEdit::kHEInvalid) srcFace = he.half_edges[h].face;
                }
                // Interior end-arc points of the beveled edge leaving v along h — they
                // sit BETWEEN this face's corner and the next face's corner in the ring.
                const MeshEdit::HEIndex e = he.half_edges[h].edge;
                if (segments > 1 && flags[e]) {
                    auto arcIt = endArcs.find(e);
                    if (arcIt != endArcs.end()) {
                        const bool atV0 = (he.edges[e].half_edge == h);  // canonical half leaves v
                        const std::vector<Vec3>& arc = arcIt->second[atV0 ? 0 : 1];
                        if (atV0) {
                            // arc runs A->B; circulation here also runs A->B.
                            for (int k = 1; k < segments; ++k) pushPoint(arc[static_cast<size_t>(k)]);
                        } else {
                            // At the v1 end the circulation direction is reversed.
                            for (int k = segments - 1; k >= 1; --k) pushPoint(arc[static_cast<size_t>(k)]);
                        }
                    }
                }
            }
            while (pts.size() >= 2 && (pts.front() - pts.back()).length_squared() <= 1e-20f) {
                pts.pop_back();  // wrap-around duplicate
            }
            if (pts.size() < 3) continue;

            Vec3 pn = newellNormal(pts);
            const float pl = std::sqrt(pn.length_squared());
            if (pl <= 1e-12f) continue;
            pn = pn * (1.0f / pl);
            if (pn.dot(avgN) < 0.0f) {
                std::reverse(pts.begin(), pts.end());
                pn = -pn;
            }
            emitPolygon(pts, pn, srcFace);
        }

        return out;
    }

    /// Fan-triangulate bevel polygons into standalone Triangle facades — the Geo-DAG
    /// node's materialization (flat-shaded + planar UVs, materialID via faceToTriangle).
    /// Edit mode materializes the same polygons differently (template-cloned Triangles).
    inline std::vector<std::shared_ptr<Triangle>> bevelPolygonsToFacades(
        const std::vector<BevelPolygon>& polys,
        const TriangleMesh& sourceMesh,
        const std::vector<uint32_t>& faceToTriangle) {
        std::vector<std::shared_ptr<Triangle>> out;
        if (!sourceMesh.geometry) return out;
        const auto& sg = *sourceMesh.geometry;
        const uint16_t* srcMat = sg.get_material_ids();
        auto materialOfFace = [&](MeshEdit::HEIndex f) -> uint16_t {
            if (f >= 0 && static_cast<size_t>(f) < faceToTriangle.size() && srcMat) {
                const size_t tri = faceToTriangle[f];
                if (tri * 3 < sg.indices.size()) return srcMat[sg.indices[tri * 3]];
            }
            return 0;
        };
        out.reserve(polys.size() * 2);
        for (const BevelPolygon& poly : polys) {
            const std::vector<Vec3>& pts = poly.pts;
            if (pts.size() < 3) continue;
            const bool smooth = poly.ptNormals.size() == pts.size();
            const uint16_t matID = materialOfFace(poly.sourceFace);
            const std::vector<Vec2> uvs = buildFacePlanarUVs(pts, poly.normal);
            for (size_t i = 1; i + 1 < pts.size(); ++i) {
                const Vec3 c = Vec3::cross(pts[i] - pts[0], pts[i + 1] - pts[0]);
                if (c.length_squared() <= 1e-20f) continue;  // collinear sliver — skip
                const Vec3 n0 = smooth ? poly.ptNormals[0] : poly.normal;
                const Vec3 n1 = smooth ? poly.ptNormals[i] : poly.normal;
                const Vec3 n2 = smooth ? poly.ptNormals[i + 1] : poly.normal;
                out.push_back(std::make_shared<Triangle>(
                    pts[0], pts[i], pts[i + 1],
                    n0, n1, n2,
                    uvs[0], uvs[i], uvs[i + 1], matID));
            }
        }
        return out;
    }

    // ============================================================================
    // GEOMETRY NODE BASE
    // ============================================================================

    class GeometryNodeBase : public NodeSystem::NodeBase {
    public:
        NodeType geometryNodeType;

        /// Parameter persistence for project save/load (see serializeGeometryGraph /
        /// deserializeGeometryGraph below). Base impl is empty — nodes with parameters
        /// override BOTH. Unknown keys are ignored and missing keys keep the
        /// constructor defaults, so adding a parameter later never breaks old files.
        virtual void serializeParams(nlohmann::json& j) const { (void)j; }
        virtual void deserializeParams(const nlohmann::json& j) { (void)j; }

        GeometryContext* getGeometryContext(NodeSystem::EvaluationContext& ctx) {
            return ctx.getDomainContext<GeometryContext>();
        }

        /// Pull a Geometry value from an input pin (empty shared_ptr if
        /// unconnected or the upstream node produced nothing).
        NodeSystem::GeometryValue getGeometryInput(int inputIndex, NodeSystem::EvaluationContext& ctx) {
            NodeSystem::PinValue val = getInputValue(inputIndex, ctx);
            NodeSystem::GeometryValue mesh;
            NodeSystem::tryGetGeometry(val, mesh);
            return mesh;
        }
    };

    // ============================================================================
    // SOURCE NODE: BASE MESH
    // ============================================================================

    /**
     * @brief Passes GeometryContext::baseMesh through unchanged. Zero-copy —
     * the DAG shares the same shared_ptr<TriangleMesh> the caller supplied.
     */
    class BaseMeshNode : public GeometryNodeBase {
    public:
        BaseMeshNode() {
            name = "Base Mesh";
            geometryNodeType = NodeType::BaseMesh;

            outputs.push_back(NodeSystem::Pin::createOutput("Geometry", NodeSystem::DataType::Geometry));

            metadata.displayName = "Base Mesh";
            metadata.category = "Input";
            metadata.description = "The object's current base mesh — source geometry for this graph.";
            metadata.headerColor = IM_COL32(76, 175, 80, 255);
            headerColor = ImVec4(0.3f, 0.68f, 0.31f, 1.0f);
        }

        std::string getTypeId() const override { return "GeoV2.BaseMesh"; }

        NodeSystem::PinValue compute(int /*outputIndex*/, NodeSystem::EvaluationContext& ctx) override {
            auto* gctx = getGeometryContext(ctx);
            if (!gctx || !gctx->baseMesh) {
                ctx.addError(id, "Base Mesh: no base mesh in GeometryContext");
                return NodeSystem::PinValue{};
            }
            return NodeSystem::GeometryValue(gctx->baseMesh);
        }
    };

    // ============================================================================
    // SUBDIVIDE (CATMULL-CLARK) NODE
    // ============================================================================

    /// Mirrors MeshModifiers::ModifierType's two live subdivision modes (Flat/CatmullClark —
    /// SmoothSubdivision is excluded, the header marks it "superseded, kept for reference").
    enum class SubdivideMode { Flat, CatmullClark };

    /**
     * @brief Subdivision node with the same two modes as the ModifierStack's Subdivide
     * modifier (MeshModifiers::ModifierType::FlatSubdivision / CatmullClark), wrapping the
     * exact same functions: MeshModifiers::SubdivideSubD + facadesToFlatMesh for Flat,
     * MeshModifiers::catmullClarkSubDStencil for Catmull-Clark — same algorithms, same
     * numerical results, exposed as a DAG node instead of a linear stack entry.
     *
     * The cage input still has to be materialized as per-face Triangle facades (both
     * underlying functions take vector<shared_ptr<Triangle>>) — this is the one conversion
     * point in the node, on the INPUT side only. The output side is already flat/zero-copy.
     */
    class SubdivideCCNode : public GeometryNodeBase {
    public:
        int levels = 1;  ///< Blender-style subdivision level
        SubdivideMode mode = SubdivideMode::CatmullClark;

        SubdivideCCNode() {
            name = "Subdivide";
            geometryNodeType = NodeType::SubdivideCC;

            inputs.push_back(NodeSystem::Pin::createInput("Geometry", NodeSystem::DataType::Geometry));
            outputs.push_back(NodeSystem::Pin::createOutput("Geometry", NodeSystem::DataType::Geometry));

            metadata.displayName = "Subdivide";
            metadata.category = "Geometry";
            metadata.description = "Flat or true Catmull-Clark subdivision — same modes as the ModifierStack's Subdivide modifier.";
            metadata.headerColor = IM_COL32(76, 175, 80, 255);
            headerColor = ImVec4(0.3f, 0.68f, 0.31f, 1.0f);
        }

        std::string getTypeId() const override { return "GeoV2.SubdivideCC"; }

        void serializeParams(nlohmann::json& j) const override {
            j["levels"] = levels;
            j["mode"] = (mode == SubdivideMode::Flat) ? 0 : 1;
        }
        void deserializeParams(const nlohmann::json& j) override {
            levels = j.value("levels", 1);
            mode = (j.value("mode", 1) == 0) ? SubdivideMode::Flat : SubdivideMode::CatmullClark;
        }

        void drawContent() override {
            ImGui::SetNextItemWidth(120);
            int modeIdx = (mode == SubdivideMode::Flat) ? 0 : 1;
            const char* modeNames[] = { "Flat", "Catmull-Clark" };
            if (ImGui::Combo("Mode", &modeIdx, modeNames, IM_ARRAYSIZE(modeNames))) {
                mode = (modeIdx == 0) ? SubdivideMode::Flat : SubdivideMode::CatmullClark;
            }
            ImGui::SetNextItemWidth(120);
            ImGui::SliderInt("Levels", &levels, 0, 5);
        }

        NodeSystem::PinValue compute(int /*outputIndex*/, NodeSystem::EvaluationContext& ctx) override {
            NodeSystem::GeometryValue inMesh = getGeometryInput(0, ctx);
            if (!inMesh || !inMesh->geometry || inMesh->geometry->indices.empty()) {
                ctx.addError(id, "Subdivide: no input geometry");
                return NodeSystem::PinValue{};
            }

            if (levels <= 0) {
                return NodeSystem::GeometryValue(inMesh);  // pass-through, no facade materialization
            }

            const size_t nTris = inMesh->num_triangles();
            std::vector<std::shared_ptr<Triangle>> cage;
            cage.reserve(nTris);
            for (size_t f = 0; f < nTris; ++f) {
                cage.push_back(std::make_shared<Triangle>(inMesh, static_cast<uint32_t>(f)));
            }

            std::shared_ptr<TriangleMesh> outMesh;
            if (mode == SubdivideMode::Flat) {
                std::vector<std::shared_ptr<Triangle>> flatFacades = MeshModifiers::SubdivideSubD(cage, levels);
                outMesh = MeshModifiers::facadesToFlatMesh(flatFacades);
            } else {
                MeshModifiers::catmullClarkSubDStencil(cage, levels, MeshModifiers::EdgeCreaseFn{}, &outMesh);
            }

            if (!outMesh) {
                ctx.addError(id, "Subdivide: evaluation failed");
                return NodeSystem::PinValue{};
            }
            return NodeSystem::GeometryValue(outMesh);
        }
    };

    // ============================================================================
    // TRANSFORM NODE (Translate / Rotate / Scale)
    // ============================================================================

    /**
     * @brief Offsets/rotates/scales the mesh's PIVOT (Transform::position/rotation/scale),
     * not its vertex data — the same thing dragging the move/rotate/scale gizmo on any other
     * object does. P_orig (local bind geometry) is left untouched; only the derived active
     * P/N cache is re-baked from the NEW transform for immediate visual correctness, mirroring
     * appendAssetToScene's rebakeMesh() invariant (P = transform->getFinal() applied to P_orig).
     *
     * An earlier version baked the offset straight into P_orig instead, which moved the mesh
     * visually but left Transform::position unchanged — SceneSelection::updatePositionFromSelection()
     * (called every time the object is (re)selected, e.g. right after Evaluate) decomposes
     * transform->getPivotMatrix() to place the gizmo, so the pivot manipulator stayed stuck at
     * the OLD position even though the mesh itself had moved. Moving the actual pivot instead
     * fixes this at the root: the gizmo now naturally follows, with no special-case code needed
     * in evaluateGeometryGraph.
     */
    class TransformNode : public GeometryNodeBase {
    public:
        Vec3 position{ 0.0f, 0.0f, 0.0f };
        Vec3 rotation{ 0.0f, 0.0f, 0.0f };  ///< Euler degrees, ZYX order (matches Transform::getPivotMatrix)
        Vec3 scale{ 1.0f, 1.0f, 1.0f };

        TransformNode() {
            name = "Transform";
            geometryNodeType = NodeType::Transform;

            inputs.push_back(NodeSystem::Pin::createInput("Geometry", NodeSystem::DataType::Geometry));
            outputs.push_back(NodeSystem::Pin::createOutput("Geometry", NodeSystem::DataType::Geometry));

            metadata.displayName = "Transform";
            metadata.category = "Geometry";
            metadata.description = "Translate / rotate / scale all vertices in local space.";
            metadata.headerColor = IM_COL32(76, 175, 80, 255);
            headerColor = ImVec4(0.3f, 0.68f, 0.31f, 1.0f);
        }

        std::string getTypeId() const override { return "GeoV2.Transform"; }

        void serializeParams(nlohmann::json& j) const override {
            j["position"] = { position.x, position.y, position.z };
            j["rotation"] = { rotation.x, rotation.y, rotation.z };
            j["scale"]    = { scale.x, scale.y, scale.z };
        }
        void deserializeParams(const nlohmann::json& j) override {
            auto readVec3 = [&j](const char* key, Vec3 def) -> Vec3 {
                if (!j.contains(key) || !j[key].is_array() || j[key].size() < 3) return def;
                return Vec3(j[key][0].get<float>(), j[key][1].get<float>(), j[key][2].get<float>());
            };
            position = readVec3("position", Vec3(0.0f, 0.0f, 0.0f));
            rotation = readVec3("rotation", Vec3(0.0f, 0.0f, 0.0f));
            scale    = readVec3("scale", Vec3(1.0f, 1.0f, 1.0f));
        }

        void drawContent() override {
            ImGui::SetNextItemWidth(180);
            ImGui::DragFloat3("Position", &position.x, 0.01f);
            ImGui::SetNextItemWidth(180);
            ImGui::DragFloat3("Rotation", &rotation.x, 0.5f);
            ImGui::SetNextItemWidth(180);
            ImGui::DragFloat3("Scale", &scale.x, 0.01f, 0.0001f, 1000.0f);
        }

        NodeSystem::PinValue compute(int /*outputIndex*/, NodeSystem::EvaluationContext& ctx) override {
            NodeSystem::GeometryValue inMesh = getGeometryInput(0, ctx);
            if (!inMesh || !inMesh->geometry || inMesh->geometry->indices.empty()) {
                ctx.addError(id, "Transform: no input geometry");
                return NodeSystem::PinValue{};
            }
            const bool isIdentity =
                position.x == 0.0f && position.y == 0.0f && position.z == 0.0f &&
                rotation.x == 0.0f && rotation.y == 0.0f && rotation.z == 0.0f &&
                scale.x == 1.0f && scale.y == 1.0f && scale.z == 1.0f;
            if (isIdentity) {
                return NodeSystem::GeometryValue(inMesh);  // no-op passthrough
            }

            auto outMesh = std::make_shared<TriangleMesh>();
            // P_orig is untouched, so the geometry buffer can be deep-copied as-is (no vertex
            // writes below) — kept as a copy rather than a shared pointer so this node's output
            // is a distinct TriangleMesh instance, consistent with every other node in this graph.
            outMesh->geometry = std::make_shared<DNA::GeometryDetail>(*inMesh->geometry);
            outMesh->transform = std::make_shared<Transform>();
            if (inMesh->transform) *outMesh->transform = *inMesh->transform;

            outMesh->transform->position = outMesh->transform->position + position;
            outMesh->transform->rotation = outMesh->transform->rotation + rotation;
            outMesh->transform->scale = Vec3(
                outMesh->transform->scale.x * scale.x,
                outMesh->transform->scale.y * scale.y,
                outMesh->transform->scale.z * scale.z);
            // position/rotation/scale only feed `base` through composeBaseMatrix() when
            // explicitly recomposed — getFinal() otherwise keeps returning the stale cached
            // matrix from before this node's offset was applied.
            outMesh->transform->updateMatrix();

            // Re-derive the active/world buffer from the (unchanged) local base using the NEW
            // pivot — the same invariant rebakeMesh() enforces elsewhere for flat meshes.
            auto& geom = *outMesh->geometry;
            const int vc = static_cast<int>(geom.get_vertex_count());
            Vec3* Po = geom.get_attribute_data_mut<Vec3>("P_orig");
            Vec3* No = geom.get_attribute_data_mut<Vec3>("N_orig");
            Vec3* P = geom.get_positions_mut();
            Vec3* N = geom.get_normals_mut();
            Matrix4x4 fT = outMesh->transform->getFinal();
            Matrix4x4 fN = outMesh->transform->getNormalTransform();
            if (P && Po) {
                for (int v = 0; v < vc; ++v) {
                    P[v] = fT.transform_point(Po[v]);
                    if (N && No) N[v] = fN.transform_vector(No[v]).normalize();
                }
            }

            return NodeSystem::GeometryValue(outMesh);
        }
    };

    // ============================================================================
    // SINK NODE: OUTPUT
    // ============================================================================

    /**
     * @brief Explicit single output/sink for a Geo-DAG — evaluation always resolves the
     * result from THIS node (exactly one allowed per graph, enforced in the UI same as
     * BaseMesh), instead of guessing among whichever nodes happen to have no outgoing
     * links. Without this, a Base Mesh feeding several parallel, unconnected branches
     * (e.g. two different Subdivide nodes off the same source) made "which branch wins"
     * an implementation detail (node array order) rather than something the user controls.
     * Purely a sink — no output pins of its own, so it's always "terminal".
     */
    class OutputNode : public GeometryNodeBase {
    public:
        OutputNode() {
            name = "Output";
            geometryNodeType = NodeType::Output;

            inputs.push_back(NodeSystem::Pin::createInput("Geometry", NodeSystem::DataType::Geometry));

            metadata.displayName = "Output";
            metadata.category = "Output";
            metadata.description = "The graph's single result — evaluation always reads this node's input.";
            metadata.headerColor = IM_COL32(220, 120, 60, 255);
            headerColor = ImVec4(0.86f, 0.47f, 0.24f, 1.0f);
        }

        std::string getTypeId() const override { return "GeoV2.Output"; }

        NodeSystem::PinValue compute(int /*outputIndex*/, NodeSystem::EvaluationContext& ctx) override {
            return NodeSystem::GeometryValue(getGeometryInput(0, ctx));
        }
    };

    // ============================================================================
    // SOURCE NODE: OBJECT SOURCE (another scene object as input)
    // ============================================================================

    /**
     * @brief Brings ANOTHER scene object's flat mesh into this graph as a second
     * geometry source (e.g. to feed Merge). Resolved by nodeName through
     * GeometryContext::resolveObjectMesh at evaluation time — the node itself never
     * touches the scene. Output is the live mesh shared_ptr (zero-copy, same contract
     * as BaseMeshNode); downstream nodes deep-copy before mutating.
     */
    class ObjectSourceNode : public GeometryNodeBase {
    public:
        char objectName[128] = "";

        ObjectSourceNode() {
            name = "Object Source";
            geometryNodeType = NodeType::ObjectSource;

            outputs.push_back(NodeSystem::Pin::createOutput("Geometry", NodeSystem::DataType::Geometry));

            metadata.displayName = "Object Source";
            metadata.category = "Input";
            metadata.description = "Another scene object's mesh (by name) — e.g. a second input for Merge.";
            metadata.headerColor = IM_COL32(76, 175, 80, 255);
            headerColor = ImVec4(0.3f, 0.68f, 0.31f, 1.0f);
        }

        std::string getTypeId() const override { return "GeoV2.ObjectSource"; }

        void serializeParams(nlohmann::json& j) const override {
            j["object"] = std::string(objectName);
        }
        void deserializeParams(const nlohmann::json& j) override {
            const std::string name = j.value("object", std::string());
            snprintf(objectName, sizeof(objectName), "%s", name.c_str());
        }

        void drawContent() override {
            ImGui::SetNextItemWidth(160);
            if (g_sceneObjectListProvider) {
                // Picker combo — the scene list is fetched fresh each time the popup is
                // open, so newly added/removed objects appear without any tracking.
                const char* preview = objectName[0] ? objectName : "<select object>";
                if (ImGui::BeginCombo("Object", preview)) {
                    const std::vector<std::string> names = g_sceneObjectListProvider();
                    for (const auto& n : names) {
                        const bool selected = (n == objectName);
                        if (ImGui::Selectable(n.c_str(), selected)) {
                            snprintf(objectName, sizeof(objectName), "%s", n.c_str());
                        }
                        if (selected) ImGui::SetItemDefaultFocus();
                    }
                    if (names.empty()) ImGui::TextDisabled("(no flat mesh objects)");
                    ImGui::EndCombo();
                }
            } else {
                // No provider wired (e.g. some future host embeds this graph without a
                // scene) — fall back to manual name entry.
                ImGui::InputText("Object", objectName, sizeof(objectName));
            }
        }

        NodeSystem::PinValue compute(int /*outputIndex*/, NodeSystem::EvaluationContext& ctx) override {
            auto* gctx = getGeometryContext(ctx);
            if (!gctx || !gctx->resolveObjectMesh) {
                ctx.addError(id, "Object Source: no object resolver in GeometryContext");
                return NodeSystem::PinValue{};
            }
            if (objectName[0] == '\0') {
                ctx.addError(id, "Object Source: no object name set");
                return NodeSystem::PinValue{};
            }
            std::shared_ptr<TriangleMesh> mesh = gctx->resolveObjectMesh(objectName);
            if (!mesh || !mesh->geometry || mesh->geometry->indices.empty()) {
                ctx.addError(id, std::string("Object Source: '") + objectName + "' not found or not a flat mesh");
                return NodeSystem::PinValue{};
            }
            return NodeSystem::GeometryValue(mesh);
        }
    };

    // ============================================================================
    // MIRROR NODE
    // ============================================================================

    /**
     * @brief Blender-style Mirror: reflect the mesh across its pivot on one local axis.
     * Reflection happens on P_orig/N_orig (local space, so it is always "across the
     * object's own pivot plane" regardless of world placement), triangle winding is
     * flipped to keep faces outward (reflection determinant is -1), and by default the
     * mirrored half is merged with the original (mergeWithOriginal), matching the
     * modeling workflow the Mirror modifier exists for.
     */
    class MirrorNode : public GeometryNodeBase {
    public:
        int axis = 0;                    ///< 0=X, 1=Y, 2=Z (local)
        bool mergeWithOriginal = true;

        MirrorNode() {
            name = "Mirror";
            geometryNodeType = NodeType::Mirror;

            inputs.push_back(NodeSystem::Pin::createInput("Geometry", NodeSystem::DataType::Geometry));
            outputs.push_back(NodeSystem::Pin::createOutput("Geometry", NodeSystem::DataType::Geometry));

            metadata.displayName = "Mirror";
            metadata.category = "Geometry";
            metadata.description = "Reflect across the pivot on a local axis, optionally merged with the original half.";
            metadata.headerColor = IM_COL32(76, 175, 80, 255);
            headerColor = ImVec4(0.3f, 0.68f, 0.31f, 1.0f);
        }

        std::string getTypeId() const override { return "GeoV2.Mirror"; }

        void serializeParams(nlohmann::json& j) const override {
            j["axis"] = axis;
            j["merge_with_original"] = mergeWithOriginal;
        }
        void deserializeParams(const nlohmann::json& j) override {
            axis = j.value("axis", 0);
            mergeWithOriginal = j.value("merge_with_original", true);
        }

        void drawContent() override {
            ImGui::SetNextItemWidth(120);
            const char* axisNames[] = { "X", "Y", "Z" };
            ImGui::Combo("Axis", &axis, axisNames, IM_ARRAYSIZE(axisNames));
            ImGui::Checkbox("Merge With Original", &mergeWithOriginal);
        }

        NodeSystem::PinValue compute(int /*outputIndex*/, NodeSystem::EvaluationContext& ctx) override {
            NodeSystem::GeometryValue inMesh = getGeometryInput(0, ctx);
            if (!inMesh || !inMesh->geometry || inMesh->geometry->indices.empty()) {
                ctx.addError(id, "Mirror: no input geometry");
                return NodeSystem::PinValue{};
            }

            auto mirrored = deepCopyMesh(inMesh);
            auto& geom = *mirrored->geometry;
            const size_t vc = geom.get_vertex_count();
            Vec3* Po = geom.get_attribute_data_mut<Vec3>("P_orig");
            Vec3* No = geom.get_attribute_data_mut<Vec3>("N_orig");
            if (!Po) {
                ctx.addError(id, "Mirror: input has no P_orig channel");
                return NodeSystem::PinValue{};
            }
            const int a = (axis < 0 || axis > 2) ? 0 : axis;
            for (size_t v = 0; v < vc; ++v) {
                (&Po[v].x)[a] = -(&Po[v].x)[a];
                if (No) (&No[v].x)[a] = -(&No[v].x)[a];
            }
            // Reflection flips handedness — swap two indices per triangle to restore
            // outward-facing winding.
            auto& idx = geom.indices;
            for (size_t i = 0; i + 2 < idx.size(); i += 3) {
                std::swap(idx[i + 1], idx[i + 2]);
            }

            if (!mergeWithOriginal) {
                rebakeFromOrig(*mirrored);
                return NodeSystem::GeometryValue(mirrored);
            }

            auto out = deepCopyMesh(inMesh);
            if (!appendMeshInto(*out, *mirrored, nullptr, nullptr)) {
                ctx.addError(id, "Mirror: merge failed");
                return NodeSystem::PinValue{};
            }
            rebakeFromOrig(*out);
            return NodeSystem::GeometryValue(out);
        }
    };

    // ============================================================================
    // NOISE DISPLACE NODE
    // ============================================================================

    /// Which Physics::Noise sampler NoiseDisplaceNode uses. Order == UI combo order.
    /// FBM/Perlin/Simplex are signed (~[-1,1], in/out displacement); Turbulence/Ridge/
    /// Billow/Voronoi are (mostly) one-sided — Midlevel recenters them when needed.
    enum class NoiseKind {
        FBM,             // fbm3D — layered Perlin, the general-purpose default
        Perlin,          // perlin3D — single octave, soft
        Simplex,         // simplex3D — single octave, faster/less axis-aligned
        Turbulence,      // turbulence3D — |FBM|, billowy smoke-like creases
        Ridge,           // ridgeNoise3D — sharp mountain ridges
        Billow,          // billowNoise3D — puffy cloud-like lumps
        Voronoi,         // voronoi3D (F1) — cellular bumps
        VoronoiCrackle,  // voronoiCrackle (F2-F1) — cracked/leather pattern
    };

    /// Shared sampler for every NoiseKind — used by NoiseDisplaceNode and MaskNoiseNode.
    inline float sampleNoiseKind(NoiseKind kind, const Vec3& p, int octaves, float frequency, int seed) {
        using namespace Physics::Noise;
        switch (kind) {
            case NoiseKind::FBM:            return fbm3D(p, octaves, frequency, 2.0f, 0.5f, seed);
            case NoiseKind::Perlin:         return perlin3D(p * frequency, seed);
            case NoiseKind::Simplex:        return simplex3D(p * frequency, seed);
            case NoiseKind::Turbulence:     return turbulence3D(p, octaves, frequency, 2.0f, 0.5f, seed);
            case NoiseKind::Ridge:          return ridgeNoise3D(p, octaves, frequency, 2.0f, 1.0f, seed);
            case NoiseKind::Billow:         return billowNoise3D(p, octaves, frequency, 2.0f, 0.5f, seed);
            case NoiseKind::Voronoi:        return voronoi3D(p * frequency, seed);
            case NoiseKind::VoronoiCrackle: return voronoiCrackle(p * frequency, seed);
        }
        return 0.0f;
    }

    /// True for kinds whose output is signed (~[-1,1]) and needs a *0.5+0.5 remap to
    /// land in 0..1 mask space; the remaining kinds are (mostly) one-sided already.
    inline bool isSignedNoiseKind(NoiseKind kind) {
        return kind == NoiseKind::FBM || kind == NoiseKind::Perlin || kind == NoiseKind::Simplex;
    }

    /**
     * @brief Displace vertices along their normals by 3D procedural noise — the classic
     * "surface detail / rock / crumple" operator. Reuses the host noise library in
     * CurlNoise.h (Physics::Noise — the exact same samplers force fields and emitters
     * use), selectable via NoiseKind. Sampled at each vertex's LOCAL position (P_orig)
     * so the pattern sticks to the mesh instead of swimming when the object moves.
     * Displacement is (noise - midlevel) * strength, Blender-Displace-style: midlevel 0
     * keeps signed noises symmetric, ~0.5 recenters the one-sided types (Turbulence/
     * Voronoi/...) so they push both in and out. Normals are recomputed from the
     * displaced topology afterwards (area-weighted), then P/N rebaked.
     */
    class NoiseDisplaceNode : public GeometryNodeBase {
    public:
        NoiseKind noiseType = NoiseKind::FBM;
        float strength = 0.1f;    ///< Displacement amplitude (local units, signed)
        float frequency = 1.0f;   ///< Noise base frequency ("scale")
        float midlevel = 0.0f;    ///< Noise value that maps to zero displacement
        int   octaves = 4;        ///< Fractal types only (FBM/Turbulence/Ridge/Billow)
        int   seed = 0;
        bool  useMask = false;            ///< Scale displacement by a Field attribute
        char  maskAttribute[64] = "mask"; ///< Per-vertex float attribute name (see Mask nodes)
        int   lastMaskState = -1;         ///< Live diagnostic: -1 off, 0 missing, 1 found (last eval)

        float sampleNoise(const Vec3& p) const {
            return sampleNoiseKind(noiseType, p, octaves, frequency, seed);
        }

        NoiseDisplaceNode() {
            name = "Noise Displace";
            geometryNodeType = NodeType::NoiseDisplace;

            inputs.push_back(NodeSystem::Pin::createInput("Geometry", NodeSystem::DataType::Geometry));
            outputs.push_back(NodeSystem::Pin::createOutput("Geometry", NodeSystem::DataType::Geometry));

            metadata.displayName = "Noise Displace";
            metadata.category = "Geometry";
            metadata.description = "Procedural noise displacement along vertex normals — 8 selectable noise types (surface detail / rock / crumple).";
            metadata.headerColor = IM_COL32(76, 175, 80, 255);
            headerColor = ImVec4(0.3f, 0.68f, 0.31f, 1.0f);
        }

        std::string getTypeId() const override { return "GeoV2.NoiseDisplace"; }

        void serializeParams(nlohmann::json& j) const override {
            j["noise_type"] = static_cast<int>(noiseType);
            j["strength"] = strength;
            j["frequency"] = frequency;
            j["midlevel"] = midlevel;
            j["octaves"] = octaves;
            j["seed"] = seed;
            j["use_mask"] = useMask;
            j["mask_attribute"] = std::string(maskAttribute);
        }
        void deserializeParams(const nlohmann::json& j) override {
            const int t = j.value("noise_type", 0);
            noiseType = (t >= 0 && t <= static_cast<int>(NoiseKind::VoronoiCrackle))
                ? static_cast<NoiseKind>(t) : NoiseKind::FBM;
            strength = j.value("strength", 0.1f);
            frequency = j.value("frequency", 1.0f);
            midlevel = j.value("midlevel", 0.0f);
            octaves = j.value("octaves", 4);
            seed = j.value("seed", 0);
            useMask = j.value("use_mask", false);
            const std::string ma = j.value("mask_attribute", std::string("mask"));
            snprintf(maskAttribute, sizeof(maskAttribute), "%s", ma.c_str());
        }

        void drawContent() override {
            const char* typeNames[] = {
                "FBM", "Perlin", "Simplex", "Turbulence",
                "Ridge", "Billow", "Voronoi", "Voronoi Crackle"
            };
            int typeIdx = static_cast<int>(noiseType);
            ImGui::SetNextItemWidth(140);
            if (ImGui::Combo("Type", &typeIdx, typeNames, IM_ARRAYSIZE(typeNames))) {
                noiseType = static_cast<NoiseKind>(typeIdx);
            }
            ImGui::SetNextItemWidth(140);
            ImGui::DragFloat("Strength", &strength, 0.005f, -100.0f, 100.0f);
            ImGui::SetNextItemWidth(140);
            ImGui::DragFloat("Frequency", &frequency, 0.01f, 0.001f, 1000.0f);
            ImGui::SetNextItemWidth(140);
            ImGui::DragFloat("Midlevel", &midlevel, 0.01f, -2.0f, 2.0f);
            const bool fractal = noiseType == NoiseKind::FBM || noiseType == NoiseKind::Turbulence
                              || noiseType == NoiseKind::Ridge || noiseType == NoiseKind::Billow;
            if (fractal) {
                ImGui::SetNextItemWidth(140);
                ImGui::SliderInt("Octaves", &octaves, 1, 8);
            }
            ImGui::SetNextItemWidth(140);
            ImGui::DragInt("Seed", &seed, 1.0f);
            ImGui::Checkbox("Use Mask", &useMask);
            if (useMask) {
                ImGui::SetNextItemWidth(140);
                ImGui::InputText("Mask Attr", maskAttribute, sizeof(maskAttribute));
                if (lastMaskState == 1) {
                    ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.6f, 1.0f), "Last eval: mask found");
                } else if (lastMaskState == 0) {
                    ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.3f, 1.0f), "Last eval: mask NOT found");
                    ImGui::TextWrapped("Add a Mask node upstream, AFTER Subdivide (Subdivide drops attributes).");
                }
            }
        }

        NodeSystem::PinValue compute(int /*outputIndex*/, NodeSystem::EvaluationContext& ctx) override {
            NodeSystem::GeometryValue inMesh = getGeometryInput(0, ctx);
            if (!inMesh || !inMesh->geometry || inMesh->geometry->indices.empty()) {
                ctx.addError(id, "Noise Displace: no input geometry");
                return NodeSystem::PinValue{};
            }
            if (strength == 0.0f) {
                return NodeSystem::GeometryValue(inMesh);  // no-op passthrough
            }

            auto out = deepCopyMesh(inMesh);
            auto& geom = *out->geometry;
            const size_t vc = geom.get_vertex_count();
            Vec3* Po = geom.get_attribute_data_mut<Vec3>("P_orig");
            const Vec3* No = geom.get_attribute_data<Vec3>("N_orig");
            if (!Po || !No) {
                ctx.addError(id, "Noise Displace: input has no P_orig/N_orig channels");
                return NodeSystem::PinValue{};
            }

            // Displacement DIRECTION must be shared across coincident duplicates (UV-seam
            // splits, flat-subdivide soup verts): they sample the same noise value (same
            // P_orig) but carry DIFFERENT per-side normals, so displacing each along its
            // own normal pulls the copies apart and tears the surface open along seam
            // borders (user-reported on Flat subdivide; CC was immune only because its
            // output is fully welded). Average N_orig over each coincident cluster and
            // displace every member along that one direction — clusters stay coincident.
            const std::vector<uint32_t> remap = buildCoincidentRemap(Po, vc);
            std::vector<Vec3> dirSum(vc, Vec3(0.0f, 0.0f, 0.0f));
            for (size_t v = 0; v < vc; ++v) {
                dirSum[remap[v]] = dirSum[remap[v]] + No[v];
            }

            // Optional Field mask (Faz 8b): displacement scaled by a per-vertex float
            // attribute (see MaskByHeight/MaskBySlope). Cluster-AVERAGED for the same
            // reason directions are — a slope mask differs per seam duplicate (each
            // side has its own normal), and per-duplicate magnitudes would re-open the
            // seam tear the direction averaging just fixed.
            const float* maskAttr = nullptr;
            std::vector<float> maskSum;
            std::vector<uint32_t> maskCnt;
            if (useMask) {
                maskAttr = geom.get_attribute_data<float>(maskAttribute);
                if (!maskAttr) {
                    // HARD error, not a silent unmasked fallback — an ignored mask is
                    // visually indistinguishable from "the mask doesn't work", which is
                    // exactly the confusion it caused. evaluateGeometryGraph surfaces
                    // this message in the viewport.
                    lastMaskState = 0;
                    ctx.addError(id, std::string("Noise Displace: mask attribute '") + maskAttribute
                        + "' not found — add a Mask node upstream (AFTER Subdivide: Subdivide rebuilds the mesh and drops attributes)");
                    return NodeSystem::PinValue{};
                }
                lastMaskState = 1;
                maskSum.assign(vc, 0.0f);
                maskCnt.assign(vc, 0u);
                for (size_t v = 0; v < vc; ++v) {
                    maskSum[remap[v]] += maskAttr[v];
                    maskCnt[remap[v]] += 1u;
                }
            } else {
                lastMaskState = -1;
            }

            for (size_t v = 0; v < vc; ++v) {
                const Vec3& s = dirSum[remap[v]];
                const float len = std::sqrt(s.x * s.x + s.y * s.y + s.z * s.z);
                if (len <= 1e-20f) continue;  // opposing normals cancelled — leave vertex put
                float d = (sampleNoise(Po[v]) - midlevel) * strength / len;
                if (maskAttr) {
                    const uint32_t canon = remap[v];
                    const float mval = maskCnt[canon] ? (maskSum[canon] / static_cast<float>(maskCnt[canon])) : 1.0f;
                    if (mval <= 0.0f) continue;
                    d *= mval;
                }
                Po[v] = Po[v] + s * d;
            }
            recomputeOrigNormals(geom);
            rebakeFromOrig(*out);
            return NodeSystem::GeometryValue(out);
        }
    };

    // ============================================================================
    // MERGE (JOIN) NODE
    // ============================================================================

    /**
     * @brief Join two geometry streams into one mesh (Blender's Join Geometry). Input B
     * is re-expressed in A's pivot space (rel = inverse(A.final) * B.final) so two
     * objects with different world placements land where they visually are, then
     * appended vertex/index-wise. Per-vertex materialIDs are preserved from both inputs,
     * so a multi-material result renders correctly. Output inherits A's transform.
     */
    class MergeNode : public GeometryNodeBase {
    public:
        MergeNode() {
            name = "Merge";
            geometryNodeType = NodeType::Merge;

            inputs.push_back(NodeSystem::Pin::createInput("Geometry A", NodeSystem::DataType::Geometry));
            inputs.push_back(NodeSystem::Pin::createInput("Geometry B", NodeSystem::DataType::Geometry));
            outputs.push_back(NodeSystem::Pin::createOutput("Geometry", NodeSystem::DataType::Geometry));

            metadata.displayName = "Merge (Join)";
            metadata.category = "Geometry";
            metadata.description = "Join two geometry inputs into one mesh (B is placed relative to A's pivot).";
            metadata.headerColor = IM_COL32(76, 175, 80, 255);
            headerColor = ImVec4(0.3f, 0.68f, 0.31f, 1.0f);
        }

        std::string getTypeId() const override { return "GeoV2.Merge"; }

        NodeSystem::PinValue compute(int /*outputIndex*/, NodeSystem::EvaluationContext& ctx) override {
            NodeSystem::GeometryValue inA = getGeometryInput(0, ctx);
            NodeSystem::GeometryValue inB = getGeometryInput(1, ctx);
            const bool okA = inA && inA->geometry && !inA->geometry->indices.empty();
            const bool okB = inB && inB->geometry && !inB->geometry->indices.empty();
            if (!okA && !okB) {
                ctx.addError(id, "Merge: no input geometry");
                return NodeSystem::PinValue{};
            }
            if (okA != okB) {
                return NodeSystem::GeometryValue(okA ? inA : inB);  // single input: passthrough
            }

            auto out = deepCopyMesh(inA);

            // Bring B's local space into A's: p_Alocal = inv(A.final) * B.final * p_Blocal.
            Matrix4x4 relP, relN;
            const Matrix4x4* relPPtr = nullptr;
            const Matrix4x4* relNPtr = nullptr;
            if (inA->transform && inB->transform) {
                relP = inA->transform->getFinal().inverse() * inB->transform->getFinal();
                relN = relP.inverse().transpose();
                relPPtr = &relP;
                relNPtr = &relN;
            }

            if (!appendMeshInto(*out, *inB, relPPtr, relNPtr)) {
                ctx.addError(id, "Merge: append failed");
                return NodeSystem::PinValue{};
            }
            rebakeFromOrig(*out);
            return NodeSystem::GeometryValue(out);
        }
    };

    // ============================================================================
    // WELD (MERGE BY DISTANCE) NODE
    // ============================================================================

    /**
     * @brief Merge vertices closer than a distance threshold (Blender's Merge by
     * Distance). Deliberately a SEPARATE node rather than an auto-weld inside
     * Merge/Mirror — joining and welding are different intents (Blender splits them the
     * same way), and keeping them orthogonal lets Mirror→Weld close the seam only when
     * the user wants a continuous surface. Spatial-hash grid (cell = threshold), 27
     * neighbor cells checked so near-boundary pairs are never missed. First vertex in a
     * cluster wins (keeps its uv/materialID); degenerate triangles produced by the
     * collapse are dropped. Optionally recomputes normals from the welded topology —
     * ON by default since welding usually means "make this one continuous surface"
     * (turn OFF to keep the inputs' original hard-edge shading).
     */
    class WeldNode : public GeometryNodeBase {
    public:
        float distance = 0.001f;      ///< Merge threshold (local units)
        bool recomputeNormalsAfter = true;

        WeldNode() {
            name = "Weld";
            geometryNodeType = NodeType::Weld;

            inputs.push_back(NodeSystem::Pin::createInput("Geometry", NodeSystem::DataType::Geometry));
            outputs.push_back(NodeSystem::Pin::createOutput("Geometry", NodeSystem::DataType::Geometry));

            metadata.displayName = "Weld (Merge by Distance)";
            metadata.category = "Geometry";
            metadata.description = "Merge vertices closer than a threshold — e.g. close a Mirror seam or fuse a Merge result.";
            metadata.headerColor = IM_COL32(76, 175, 80, 255);
            headerColor = ImVec4(0.3f, 0.68f, 0.31f, 1.0f);
        }

        std::string getTypeId() const override { return "GeoV2.Weld"; }

        void serializeParams(nlohmann::json& j) const override {
            j["distance"] = distance;
            j["recompute_normals"] = recomputeNormalsAfter;
        }
        void deserializeParams(const nlohmann::json& j) override {
            distance = j.value("distance", 0.001f);
            recomputeNormalsAfter = j.value("recompute_normals", true);
        }

        void drawContent() override {
            ImGui::SetNextItemWidth(140);
            ImGui::DragFloat("Distance", &distance, 0.0001f, 0.0f, 10.0f, "%.5f");
            ImGui::Checkbox("Recompute Normals", &recomputeNormalsAfter);
        }

        NodeSystem::PinValue compute(int /*outputIndex*/, NodeSystem::EvaluationContext& ctx) override {
            NodeSystem::GeometryValue inMesh = getGeometryInput(0, ctx);
            if (!inMesh || !inMesh->geometry || inMesh->geometry->indices.empty()) {
                ctx.addError(id, "Weld: no input geometry");
                return NodeSystem::PinValue{};
            }
            if (distance <= 0.0f) {
                return NodeSystem::GeometryValue(inMesh);  // no-op passthrough
            }

            const auto& sg = *inMesh->geometry;
            const size_t vc = sg.get_vertex_count();
            const Vec3* Po = sg.get_attribute_data<Vec3>("P_orig");
            if (!Po || vc == 0) {
                ctx.addError(id, "Weld: input has no P_orig channel");
                return NodeSystem::PinValue{};
            }

            // 1. Cluster: remap[v] = canonical (lowest-index) vertex within `distance`.
            // Cell size == threshold, so any pair within the threshold is at most one
            // cell apart on each axis — the 3x3x3 neighborhood scan is exhaustive.
            const float inv = 1.0f / distance;
            const float d2 = distance * distance;
            auto cellKey = [](int x, int y, int z) -> uint64_t {
                return (static_cast<uint64_t>(static_cast<uint32_t>(x) & 0x1FFFFF) << 42)
                     | (static_cast<uint64_t>(static_cast<uint32_t>(y) & 0x1FFFFF) << 21)
                     |  static_cast<uint64_t>(static_cast<uint32_t>(z) & 0x1FFFFF);
            };
            std::unordered_map<uint64_t, std::vector<uint32_t>> cells;
            cells.reserve(vc);
            std::vector<uint32_t> remap(vc);
            for (size_t v = 0; v < vc; ++v) {
                const int ix = static_cast<int>(std::floor(Po[v].x * inv));
                const int iy = static_cast<int>(std::floor(Po[v].y * inv));
                const int iz = static_cast<int>(std::floor(Po[v].z * inv));
                uint32_t canon = static_cast<uint32_t>(v);
                for (int dx = -1; dx <= 1 && canon == v; ++dx)
                for (int dy = -1; dy <= 1 && canon == v; ++dy)
                for (int dz = -1; dz <= 1 && canon == v; ++dz) {
                    auto it = cells.find(cellKey(ix + dx, iy + dy, iz + dz));
                    if (it == cells.end()) continue;
                    for (const uint32_t cand : it->second) {
                        const Vec3 diff = Po[v] - Po[cand];
                        if (diff.x * diff.x + diff.y * diff.y + diff.z * diff.z <= d2) {
                            canon = cand;
                            break;
                        }
                    }
                }
                remap[v] = canon;
                if (canon == v) {
                    cells[cellKey(ix, iy, iz)].push_back(static_cast<uint32_t>(v));
                }
            }

            // 2. Compact surviving vertices into a fresh GeometryDetail (only core
            // channels — custom attributes have no enumeration API yet, acceptable for
            // this slice since import/DAG meshes only carry core channels).
            std::vector<uint32_t> newIndex(vc, 0xFFFFFFFFu);
            uint32_t nv = 0;
            for (size_t v = 0; v < vc; ++v) {
                if (remap[v] == v) newIndex[v] = nv++;
            }
            if (nv == vc) {
                return NodeSystem::GeometryValue(inMesh);  // nothing within threshold
            }

            auto out = std::make_shared<TriangleMesh>();
            out->transform = std::make_shared<Transform>();
            if (inMesh->transform) *out->transform = *inMesh->transform;
            out->geometry = std::make_shared<DNA::GeometryDetail>();
            auto& dg = *out->geometry;
            if (sg.has_attribute("P")) dg.add_attribute<Vec3>("P");
            if (sg.has_attribute("N")) dg.add_attribute<Vec3>("N");
            if (sg.has_attribute("P_orig")) dg.add_attribute<Vec3>("P_orig");
            if (sg.has_attribute("N_orig")) dg.add_attribute<Vec3>("N_orig");
            if (sg.has_attribute("uv")) dg.add_attribute<Vec2>("uv");
            if (sg.has_attribute("materialID")) dg.add_attribute<uint16_t>("materialID");
            dg.resize_vertices(nv);

            {
                const Vec3* sP = sg.get_positions();
                const Vec3* sN = sg.get_normals();
                const Vec3* sNo = sg.get_attribute_data<Vec3>("N_orig");
                const Vec2* sUv = sg.get_uvs();
                const uint16_t* sMat = sg.get_material_ids();
                Vec3* dP = dg.get_positions_mut();
                Vec3* dN = dg.get_normals_mut();
                Vec3* dPo = dg.get_attribute_data_mut<Vec3>("P_orig");
                Vec3* dNo = dg.get_attribute_data_mut<Vec3>("N_orig");
                Vec2* dUv = dg.get_uvs_mut();
                uint16_t* dMat = dg.get_attribute_data_mut<uint16_t>("materialID");
                for (size_t v = 0; v < vc; ++v) {
                    if (remap[v] != v) continue;
                    const uint32_t dst = newIndex[v];
                    if (dP && sP) dP[dst] = sP[v];
                    if (dN && sN) dN[dst] = sN[v];
                    if (dPo) dPo[dst] = Po[v];
                    if (dNo && sNo) dNo[dst] = sNo[v];
                    if (dUv && sUv) dUv[dst] = sUv[v];
                    if (dMat && sMat) dMat[dst] = sMat[v];
                }
            }

            // 3. Remap indices, dropping triangles the collapse degenerated.
            dg.indices.reserve(sg.indices.size());
            for (size_t i = 0; i + 2 < sg.indices.size(); i += 3) {
                const uint32_t a = newIndex[remap[sg.indices[i]]];
                const uint32_t b = newIndex[remap[sg.indices[i + 1]]];
                const uint32_t c = newIndex[remap[sg.indices[i + 2]]];
                if (a == b || b == c || a == c) continue;
                dg.indices.push_back(a);
                dg.indices.push_back(b);
                dg.indices.push_back(c);
            }
            if (dg.indices.empty()) {
                ctx.addError(id, "Weld: threshold collapsed the whole mesh");
                return NodeSystem::PinValue{};
            }

            if (recomputeNormalsAfter) {
                recomputeOrigNormals(dg);
            }
            rebakeFromOrig(*out);
            return NodeSystem::GeometryValue(out);
        }
    };

    // ============================================================================
    // ARRAY NODE
    // ============================================================================

    /**
     * @brief Blender's Array modifier, "Object Offset"-style: Count copies of the input,
     * each placed by compounding the SAME per-step transform (translate + rotate + scale,
     * all local-space) onto the previous one — so a non-zero rotation naturally spirals
     * the copies around the pivot instead of just marching in a straight line (a plain
     * per-step *i* translate can't reproduce that, this needs the matrix to compound).
     * Reuses the Merge node's exact mechanism: appendMeshInto's relPoint/relNormal matrix
     * pair (originally built for Merge's "bring B into A's pivot space"), applied here
     * count-1 times against the SAME original input instead of a second geometry stream.
     * Copy 0 is the untouched original; copies 1..count-1 are the transformed appends.
     */
    class ArrayNode : public GeometryNodeBase {
    public:
        int count = 3;                       ///< Total instances, including the original (>=1)
        Vec3 offset{ 1.0f, 0.0f, 0.0f };      ///< Per-step translate (local units)
        Vec3 rotation{ 0.0f, 0.0f, 0.0f };    ///< Per-step rotate, Euler degrees (ZYX, matches Transform)
        Vec3 scale{ 1.0f, 1.0f, 1.0f };       ///< Per-step scale multiplier (compounds every step)

        ArrayNode() {
            name = "Array";
            geometryNodeType = NodeType::Array;

            inputs.push_back(NodeSystem::Pin::createInput("Geometry", NodeSystem::DataType::Geometry));
            outputs.push_back(NodeSystem::Pin::createOutput("Geometry", NodeSystem::DataType::Geometry));

            metadata.displayName = "Array";
            metadata.category = "Geometry";
            metadata.description = "Repeat the input Count times, compounding a per-step translate/rotate/scale (Blender Array modifier, Object Offset style).";
            metadata.headerColor = IM_COL32(76, 175, 80, 255);
            headerColor = ImVec4(0.3f, 0.68f, 0.31f, 1.0f);
        }

        std::string getTypeId() const override { return "GeoV2.Array"; }

        void serializeParams(nlohmann::json& j) const override {
            j["count"] = count;
            j["offset"] = { offset.x, offset.y, offset.z };
            j["rotation"] = { rotation.x, rotation.y, rotation.z };
            j["scale"] = { scale.x, scale.y, scale.z };
        }
        void deserializeParams(const nlohmann::json& j) override {
            auto readVec3 = [&j](const char* key, Vec3 def) -> Vec3 {
                if (!j.contains(key) || !j[key].is_array() || j[key].size() < 3) return def;
                return Vec3(j[key][0].get<float>(), j[key][1].get<float>(), j[key][2].get<float>());
            };
            count = j.value("count", 3);
            offset = readVec3("offset", Vec3(1.0f, 0.0f, 0.0f));
            rotation = readVec3("rotation", Vec3(0.0f, 0.0f, 0.0f));
            scale = readVec3("scale", Vec3(1.0f, 1.0f, 1.0f));
        }

        void drawContent() override {
            ImGui::SetNextItemWidth(120);
            ImGui::SliderInt("Count", &count, 1, 64);
            ImGui::SetNextItemWidth(180);
            ImGui::DragFloat3("Offset", &offset.x, 0.01f);
            ImGui::SetNextItemWidth(180);
            ImGui::DragFloat3("Rotation", &rotation.x, 0.5f);
            ImGui::SetNextItemWidth(180);
            ImGui::DragFloat3("Scale", &scale.x, 0.01f, 0.0001f, 1000.0f);
        }

        NodeSystem::PinValue compute(int /*outputIndex*/, NodeSystem::EvaluationContext& ctx) override {
            NodeSystem::GeometryValue inMesh = getGeometryInput(0, ctx);
            if (!inMesh || !inMesh->geometry || inMesh->geometry->indices.empty()) {
                ctx.addError(id, "Array: no input geometry");
                return NodeSystem::PinValue{};
            }
            if (count <= 1) {
                return NodeSystem::GeometryValue(inMesh);  // no-op passthrough
            }

            // Per-step relative matrix — same T*R*S composition as Transform::getPivotMatrix,
            // built directly (no Transform object needed, this never touches a mesh's pivot).
            const float deg2rad = 3.14159265358979f / 180.0f;
            const Matrix4x4 T = Matrix4x4::translation(offset);
            const Matrix4x4 Rx = Matrix4x4::rotationX(rotation.x * deg2rad);
            const Matrix4x4 Ry = Matrix4x4::rotationY(rotation.y * deg2rad);
            const Matrix4x4 Rz = Matrix4x4::rotationZ(rotation.z * deg2rad);
            const Matrix4x4 R = Rz * Ry * Rx;
            const Matrix4x4 S = Matrix4x4::scaling(scale);
            const Matrix4x4 step = T * R * S;

            auto out = deepCopyMesh(inMesh);  // copy 0: the untouched original
            Matrix4x4 accumP = Matrix4x4::identity();
            for (int i = 1; i < count; ++i) {
                accumP = step * accumP;  // compounds: i=1 -> step, i=2 -> step^2, ...
                const Matrix4x4 accumN = accumP.inverse().transpose();
                if (!appendMeshInto(*out, *inMesh, &accumP, &accumN)) {
                    ctx.addError(id, "Array: append failed");
                    return NodeSystem::PinValue{};
                }
            }
            rebakeFromOrig(*out);
            return NodeSystem::GeometryValue(out);
        }
    };

    // ============================================================================
    // EXTRUDE NODE (per-face, half-edge backed)
    // ============================================================================

    /**
     * @brief Extrude EVERY face individually along its own normal (Blender's "Extrude
     * Individual Faces", not "Extrude Region" — each face gets its own separate prism;
     * two adjacent extruded faces pull apart at their old shared edge instead of staying
     * welded). Built on MeshEdit::HalfEdgeMesh::extrudeFace, the exact Euler operator the
     * facade-based Edit Mode's Extrude already uses — this node only adds the flat-SoA
     * bridge (weld -> half-edge -> extrude every face -> fan-triangulate back), no new
     * topology algorithm. A whole-mesh individual-faces extrude is also a legitimate
     * stylized-geometry effect on its own (faceted "shatter/spike" look); per-face
     * selection (a Field-masked subset) is a natural follow-up, not done here.
     */
    class ExtrudeNode : public GeometryNodeBase {
    public:
        float distance = 0.2f;   ///< Offset along each face's own normal (local units)
        bool  mergeQuads = true; ///< Re-detect quad faces (mergeCoplanarTrianglePairs) before extruding
        bool  useMask = false;            ///< Gate faces by a per-vertex Field attribute
        char  maskAttribute[64] = "mask"; ///< Field name (see Mask nodes)
        float maskThreshold = 0.5f;       ///< Face extruded when its avg mask >= threshold
        bool  maskScalesDistance = false; ///< Also multiply distance by the face's mask value
        int   lastMaskState = -1;         ///< Diagnostic: -1 off, 0 missing, 1 found (last eval)

        ExtrudeNode() {
            name = "Extrude";
            geometryNodeType = NodeType::Extrude;

            inputs.push_back(NodeSystem::Pin::createInput("Geometry", NodeSystem::DataType::Geometry));
            outputs.push_back(NodeSystem::Pin::createOutput("Geometry", NodeSystem::DataType::Geometry));

            metadata.displayName = "Extrude";
            metadata.category = "Geometry";
            metadata.description = "Extrude every face individually along its own normal (half-edge backed).";
            metadata.headerColor = IM_COL32(76, 175, 80, 255);
            headerColor = ImVec4(0.3f, 0.68f, 0.31f, 1.0f);
        }

        std::string getTypeId() const override { return "GeoV2.Extrude"; }

        void serializeParams(nlohmann::json& j) const override {
            j["distance"] = distance;
            j["merge_quads"] = mergeQuads;
            j["use_mask"] = useMask;
            j["mask_attribute"] = std::string(maskAttribute);
            j["mask_threshold"] = maskThreshold;
            j["mask_scales_distance"] = maskScalesDistance;
        }
        void deserializeParams(const nlohmann::json& j) override {
            distance = j.value("distance", 0.2f);
            mergeQuads = j.value("merge_quads", true);
            useMask = j.value("use_mask", false);
            const std::string ma = j.value("mask_attribute", std::string("mask"));
            snprintf(maskAttribute, sizeof(maskAttribute), "%s", ma.c_str());
            maskThreshold = j.value("mask_threshold", 0.5f);
            maskScalesDistance = j.value("mask_scales_distance", false);
        }

        void drawContent() override {
            ImGui::SetNextItemWidth(140);
            ImGui::DragFloat("Distance", &distance, 0.005f, -100.0f, 100.0f);
            ImGui::Checkbox("Detect Quads", &mergeQuads);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Merge coplanar triangle pairs back into quads before extruding,\nso the diagonal from triangulation doesn't crease each source quad.\nOff = extrude every raw triangle individually.");
            }
            ImGui::Checkbox("Use Mask", &useMask);
            if (useMask) {
                ImGui::SetNextItemWidth(140);
                ImGui::InputText("Mask Attr", maskAttribute, sizeof(maskAttribute));
                ImGui::SetNextItemWidth(140);
                ImGui::SliderFloat("Threshold", &maskThreshold, 0.0f, 1.0f);
                ImGui::Checkbox("Mask Scales Distance", &maskScalesDistance);
                if (lastMaskState == 1) {
                    ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.6f, 1.0f), "Last eval: mask found");
                } else if (lastMaskState == 0) {
                    ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.3f, 1.0f), "Last eval: mask NOT found");
                    ImGui::TextWrapped("Add a Mask node upstream (directly before this node —\nExtrude/Inset/Subdivide outputs drop attributes).");
                }
            }
        }

        NodeSystem::PinValue compute(int /*outputIndex*/, NodeSystem::EvaluationContext& ctx) override {
            NodeSystem::GeometryValue inMesh = getGeometryInput(0, ctx);
            if (!inMesh || !inMesh->geometry || inMesh->geometry->indices.empty()) {
                ctx.addError(id, "Extrude: no input geometry");
                return NodeSystem::PinValue{};
            }
            if (distance == 0.0f) {
                return NodeSystem::GeometryValue(inMesh);  // no-op passthrough
            }

            FlatHalfEdgeBridge bridge;
            if (!buildHalfEdgeFromFlatMesh(*inMesh, bridge)) {
                ctx.addError(id, "Extrude: mesh topology unavailable (empty or fully degenerate)");
                return NodeSystem::PinValue{};
            }
            auto& he = bridge.heMesh;
            if (mergeQuads) mergeCoplanarTrianglePairs(he);
            const MeshEdit::HEIndex originalFaceCount = static_cast<MeshEdit::HEIndex>(he.faces.size());

            // Optional Field gate (same hard-error contract as NoiseDisplace: a missing
            // mask is an error surfaced in the viewport, never a silent unmasked run).
            std::vector<float> weldedMask;
            if (useMask) {
                if (!buildWeldedVertexMask(*inMesh, bridge, maskAttribute, weldedMask)) {
                    lastMaskState = 0;
                    ctx.addError(id, std::string("Extrude: mask attribute '") + maskAttribute
                        + "' not found — add a Mask node directly upstream (topology-changing nodes drop attributes)");
                    return NodeSystem::PinValue{};
                }
                lastMaskState = 1;
            } else {
                lastMaskState = -1;
            }

            std::unordered_map<MeshEdit::HEIndex, MeshEdit::HEIndex> faceSourceMap;
            std::vector<MeshEdit::HEIndex> sideFaces;
            std::vector<MeshEdit::HEIndex> maskScratch;
            int extrudedCount = 0;
            for (MeshEdit::HEIndex f = 0; f < originalFaceCount; ++f) {
                if (he.faces[f].removed) continue;
                float faceDistance = distance;
                if (useMask) {
                    const float m = faceMaskAverage(he, f, weldedMask, maskScratch);
                    if (m < maskThreshold) continue;
                    if (maskScalesDistance) faceDistance *= m;
                }
                const Vec3 n = he.faceNormal(f);
                if (n.length_squared() <= 1e-10f) continue;
                sideFaces.clear();
                if (he.extrudeFace(f, n * faceDistance, &sideFaces) == MeshEdit::kHEInvalid) continue;
                for (const MeshEdit::HEIndex sf : sideFaces) faceSourceMap[sf] = f;
                ++extrudedCount;
            }
            if (extrudedCount <= 0) {
                ctx.addError(id, useMask ? "Extrude: mask gated out every face (threshold too high?)"
                                         : "Extrude: no faces could be extruded");
                return NodeSystem::PinValue{};
            }

            auto outMesh = rebuildFlatMeshFromHalfEdge(he, *inMesh, bridge.faceToTriangle, originalFaceCount, faceSourceMap);
            if (!outMesh) {
                ctx.addError(id, "Extrude: rebuild failed");
                return NodeSystem::PinValue{};
            }
            return NodeSystem::GeometryValue(outMesh);
        }
    };

    // ============================================================================
    // INSET NODE (per-face, half-edge backed)
    // ============================================================================

    /**
     * @brief Inset EVERY face individually (Blender's default multi-face Inset — each
     * face shrinks toward its own centroid, producing a ring of new quads per face).
     * Built on MeshEdit::HalfEdgeMesh::insetFace, the exact Euler operator the
     * facade-based Edit Mode's Inset already uses (see scene_ui_mesh_overlay.cpp's
     * insetSelectedFaces) — same bridge/rebuild pair as ExtrudeNode.
     */
    class InsetNode : public GeometryNodeBase {
    public:
        float amount = 0.2f;     ///< 0 = no change, 1 = collapse to centroid
        bool  mergeQuads = true; ///< Re-detect quad faces (mergeCoplanarTrianglePairs) before insetting
        bool  useMask = false;            ///< Gate faces by a per-vertex Field attribute
        char  maskAttribute[64] = "mask"; ///< Field name (see Mask nodes)
        float maskThreshold = 0.5f;       ///< Face inset when its avg mask >= threshold
        bool  maskScalesAmount = false;   ///< Also multiply amount by the face's mask value
        int   lastMaskState = -1;         ///< Diagnostic: -1 off, 0 missing, 1 found (last eval)

        InsetNode() {
            name = "Inset";
            geometryNodeType = NodeType::Inset;

            inputs.push_back(NodeSystem::Pin::createInput("Geometry", NodeSystem::DataType::Geometry));
            outputs.push_back(NodeSystem::Pin::createOutput("Geometry", NodeSystem::DataType::Geometry));

            metadata.displayName = "Inset";
            metadata.category = "Geometry";
            metadata.description = "Inset every face individually toward its own centroid (half-edge backed).";
            metadata.headerColor = IM_COL32(76, 175, 80, 255);
            headerColor = ImVec4(0.3f, 0.68f, 0.31f, 1.0f);
        }

        std::string getTypeId() const override { return "GeoV2.Inset"; }

        void serializeParams(nlohmann::json& j) const override {
            j["amount"] = amount;
            j["merge_quads"] = mergeQuads;
            j["use_mask"] = useMask;
            j["mask_attribute"] = std::string(maskAttribute);
            j["mask_threshold"] = maskThreshold;
            j["mask_scales_amount"] = maskScalesAmount;
        }
        void deserializeParams(const nlohmann::json& j) override {
            amount = j.value("amount", 0.2f);
            mergeQuads = j.value("merge_quads", true);
            useMask = j.value("use_mask", false);
            const std::string ma = j.value("mask_attribute", std::string("mask"));
            snprintf(maskAttribute, sizeof(maskAttribute), "%s", ma.c_str());
            maskThreshold = j.value("mask_threshold", 0.5f);
            maskScalesAmount = j.value("mask_scales_amount", false);
        }

        void drawContent() override {
            ImGui::SetNextItemWidth(140);
            ImGui::SliderFloat("Amount", &amount, 0.0f, 1.0f);
            ImGui::Checkbox("Detect Quads", &mergeQuads);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Merge coplanar triangle pairs back into quads before insetting,\nso each source quad gets ONE inset ring instead of insetting\nits two triangulation halves separately. Off = per-triangle inset.");
            }
            ImGui::Checkbox("Use Mask", &useMask);
            if (useMask) {
                ImGui::SetNextItemWidth(140);
                ImGui::InputText("Mask Attr", maskAttribute, sizeof(maskAttribute));
                ImGui::SetNextItemWidth(140);
                ImGui::SliderFloat("Threshold", &maskThreshold, 0.0f, 1.0f);
                ImGui::Checkbox("Mask Scales Amount", &maskScalesAmount);
                if (lastMaskState == 1) {
                    ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.6f, 1.0f), "Last eval: mask found");
                } else if (lastMaskState == 0) {
                    ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.3f, 1.0f), "Last eval: mask NOT found");
                    ImGui::TextWrapped("Add a Mask node upstream (directly before this node —\nExtrude/Inset/Subdivide outputs drop attributes).");
                }
            }
        }

        NodeSystem::PinValue compute(int /*outputIndex*/, NodeSystem::EvaluationContext& ctx) override {
            NodeSystem::GeometryValue inMesh = getGeometryInput(0, ctx);
            if (!inMesh || !inMesh->geometry || inMesh->geometry->indices.empty()) {
                ctx.addError(id, "Inset: no input geometry");
                return NodeSystem::PinValue{};
            }
            if (amount <= 0.0f) {
                return NodeSystem::GeometryValue(inMesh);  // no-op passthrough
            }

            FlatHalfEdgeBridge bridge;
            if (!buildHalfEdgeFromFlatMesh(*inMesh, bridge)) {
                ctx.addError(id, "Inset: mesh topology unavailable (empty or fully degenerate)");
                return NodeSystem::PinValue{};
            }
            auto& he = bridge.heMesh;
            if (mergeQuads) mergeCoplanarTrianglePairs(he);
            const MeshEdit::HEIndex originalFaceCount = static_cast<MeshEdit::HEIndex>(he.faces.size());

            std::vector<float> weldedMask;
            if (useMask) {
                if (!buildWeldedVertexMask(*inMesh, bridge, maskAttribute, weldedMask)) {
                    lastMaskState = 0;
                    ctx.addError(id, std::string("Inset: mask attribute '") + maskAttribute
                        + "' not found — add a Mask node directly upstream (topology-changing nodes drop attributes)");
                    return NodeSystem::PinValue{};
                }
                lastMaskState = 1;
            } else {
                lastMaskState = -1;
            }

            std::unordered_map<MeshEdit::HEIndex, MeshEdit::HEIndex> faceSourceMap;
            std::vector<MeshEdit::HEIndex> sideFaces;
            std::vector<MeshEdit::HEIndex> maskScratch;
            int insetCount = 0;
            for (MeshEdit::HEIndex f = 0; f < originalFaceCount; ++f) {
                if (he.faces[f].removed) continue;
                float faceAmount = amount;
                if (useMask) {
                    const float m = faceMaskAverage(he, f, weldedMask, maskScratch);
                    if (m < maskThreshold) continue;
                    if (maskScalesAmount) faceAmount *= m;
                }
                if (faceAmount <= 0.0f) continue;
                sideFaces.clear();
                if (he.insetFace(f, faceAmount, &sideFaces) == MeshEdit::kHEInvalid) continue;
                for (const MeshEdit::HEIndex sf : sideFaces) faceSourceMap[sf] = f;
                ++insetCount;
            }
            if (insetCount <= 0) {
                ctx.addError(id, useMask ? "Inset: mask gated out every face (threshold too high?)"
                                         : "Inset: no faces could be inset");
                return NodeSystem::PinValue{};
            }

            auto outMesh = rebuildFlatMeshFromHalfEdge(he, *inMesh, bridge.faceToTriangle, originalFaceCount, faceSourceMap);
            if (!outMesh) {
                ctx.addError(id, "Inset: rebuild failed");
                return NodeSystem::PinValue{};
            }
            return NodeSystem::GeometryValue(outMesh);
        }
    };

    // ============================================================================
    // BEVEL NODE (angle-limited edge chamfer, half-edge backed)
    // ============================================================================

    /**
     * @brief Blender's Bevel modifier with Limit Method = Angle: every interior edge
     * sharper than the angle threshold becomes a chamfer strip of Segments bands
     * (Round profile sweeps them on a circular arc), with n-gon patches closing the
     * corners. All geometry work is in bevelEdgesPolygons (see its doc for the
     * construction + limits); this node adds the flat-SoA bridge (with quad
     * re-detection so a cube imports as 12 sharp edges, not 12 + 6 invisible
     * diagonals), the angle-based edge selection, and an optional weld-aware
     * smooth-shading pass — even a 1-segment chamfer reads as "rounded" once normals
     * blend across the strips. The Edit Mode selected-edge bevel tool shares the same
     * core with an explicit selection instead of the angle rule.
     */
    class BevelNode : public GeometryNodeBase {
    public:
        float width = 0.05f;      ///< Chamfer offset per face (local units)
        float angleDeg = 30.0f;   ///< Bevel edges with dihedral angle >= this (degrees)
        int   segments = 1;       ///< Bands per strip (1 = chamfer, more = rounded)
        bool  roundProfile = true;    ///< Sweep bands on a circular arc (vs flat multi-cut)
        bool  smoothShading = false;  ///< Recompute weld-aware smooth normals on the result

        BevelNode() {
            name = "Bevel";
            geometryNodeType = NodeType::Bevel;

            inputs.push_back(NodeSystem::Pin::createInput("Geometry", NodeSystem::DataType::Geometry));
            outputs.push_back(NodeSystem::Pin::createOutput("Geometry", NodeSystem::DataType::Geometry));

            metadata.displayName = "Bevel";
            metadata.category = "Geometry";
            metadata.description = "Chamfer sharp edges (angle-limited, 1 segment) — Blender Bevel modifier semantics.";
            metadata.headerColor = IM_COL32(76, 175, 80, 255);
            headerColor = ImVec4(0.3f, 0.68f, 0.31f, 1.0f);
        }

        std::string getTypeId() const override { return "GeoV2.Bevel"; }

        void serializeParams(nlohmann::json& j) const override {
            j["width"] = width;
            j["angle"] = angleDeg;
            j["segments"] = segments;
            j["round_profile"] = roundProfile;
            j["smooth_shading"] = smoothShading;
        }
        void deserializeParams(const nlohmann::json& j) override {
            width = j.value("width", 0.05f);
            angleDeg = j.value("angle", 30.0f);
            segments = j.value("segments", 1);
            roundProfile = j.value("round_profile", true);
            smoothShading = j.value("smooth_shading", false);
        }

        void drawContent() override {
            ImGui::SetNextItemWidth(140);
            ImGui::DragFloat("Width", &width, 0.002f, 0.0f, 100.0f, "%.4f");
            ImGui::SetNextItemWidth(140);
            ImGui::SliderFloat("Angle", &angleDeg, 0.0f, 180.0f, "%.0f deg");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Only edges whose faces meet at an angle >= this are beveled\n(Blender Bevel modifier, Limit Method: Angle).");
            }
            ImGui::SetNextItemWidth(140);
            ImGui::SliderInt("Segments", &segments, 1, 8);
            if (segments > 1) {
                ImGui::SetNextItemWidth(140);
                int profileIdx = roundProfile ? 1 : 0;
                const char* profileNames[] = { "Flat", "Round" };
                if (ImGui::Combo("Profile", &profileIdx, profileNames, IM_ARRAYSIZE(profileNames))) {
                    roundProfile = (profileIdx == 1);
                }
            }
            ImGui::Checkbox("Smooth Shading", &smoothShading);
        }

        NodeSystem::PinValue compute(int /*outputIndex*/, NodeSystem::EvaluationContext& ctx) override {
            NodeSystem::GeometryValue inMesh = getGeometryInput(0, ctx);
            if (!inMesh || !inMesh->geometry || inMesh->geometry->indices.empty()) {
                ctx.addError(id, "Bevel: no input geometry");
                return NodeSystem::PinValue{};
            }
            if (width <= 0.0f) {
                return NodeSystem::GeometryValue(inMesh);  // no-op passthrough
            }

            FlatHalfEdgeBridge bridge;
            if (!buildHalfEdgeFromFlatMesh(*inMesh, bridge)) {
                ctx.addError(id, "Bevel: mesh topology unavailable (empty or fully degenerate)");
                return NodeSystem::PinValue{};
            }
            // Always merge quads for Bevel: a quad's triangulation diagonal is coplanar
            // (never beveled), but leaving it in makes the corner shrink open zero-width
            // slivers along the diagonal — merging removes the problem at the source.
            mergeCoplanarTrianglePairs(bridge.heMesh);

            std::vector<bool> edgeFlags;
            if (selectBevelEdgesByAngle(bridge.heMesh, angleDeg, edgeFlags) == 0) {
                // No edge above the angle threshold — legitimate pass-through, not an error.
                return NodeSystem::GeometryValue(inMesh);
            }
            std::vector<BevelPolygon> polys =
                bevelEdgesPolygons(bridge.heMesh, edgeFlags, width, segments, roundProfile);
            std::vector<std::shared_ptr<Triangle>> facades =
                bevelPolygonsToFacades(polys, *inMesh, bridge.faceToTriangle);
            if (facades.empty()) {
                return NodeSystem::GeometryValue(inMesh);
            }

            auto outMesh = MeshModifiers::facadesToFlatMesh(facades);
            if (!outMesh) {
                ctx.addError(id, "Bevel: rebuild failed");
                return NodeSystem::PinValue{};
            }
            if (!outMesh->transform && inMesh->transform) {
                outMesh->transform = std::make_shared<Transform>(*inMesh->transform);
            }
            if (smoothShading) {
                recomputeOrigNormals(*outMesh->geometry);
            }
            rebakeFromOrig(*outMesh);
            return NodeSystem::GeometryValue(outMesh);
        }
    };

    // ============================================================================
    // FIELD NODES (Faz 8b first slice) — masks as named per-vertex float attributes
    // ============================================================================
    // A "Field" is a per-vertex float (0..1) stored as a GeometryDetail custom
    // attribute (default name "mask") that rides the Geometry wire — no separate
    // socket type. Producers below WRITE the attribute; consumers (Noise Displace's
    // "Use Mask") READ it. Custom attributes survive deep copies and TransformNode,
    // but NOT SubdivideCC (facade path) — so put mask nodes AFTER Subdivide in the
    // chain. Field nodes use the blue header the Faz 8 plan assigned to Field wires.

    /**
     * @brief Write a mask from vertex height (local P_orig.y): smoothstep ramp from
     * Min to Max. In "Relative" mode (default) Min/Max are 0..1 fractions of the
     * mesh's own local Y range — works on any mesh without knowing its units.
     */
    class MaskByHeightNode : public GeometryNodeBase {
    public:
        char attributeName[64] = "mask";
        float minV = 0.3f;   ///< Ramp start (fraction of bbox Y when relative, else local units)
        float maxV = 0.7f;   ///< Ramp end
        bool relativeToBBox = true;
        bool invert = false;
        float lastMin = -1.0f, lastMean = -1.0f, lastMax = -1.0f;  ///< Live diagnostic (last eval)

        MaskByHeightNode() {
            name = "Mask by Height";
            geometryNodeType = NodeType::MaskByHeight;

            inputs.push_back(NodeSystem::Pin::createInput("Geometry", NodeSystem::DataType::Geometry));
            outputs.push_back(NodeSystem::Pin::createOutput("Geometry", NodeSystem::DataType::Geometry));

            metadata.displayName = "Mask by Height";
            metadata.category = "Field";
            metadata.description = "Per-vertex mask attribute from local height — feeds mask-aware nodes like Noise Displace.";
            metadata.headerColor = IM_COL32(66, 133, 244, 255);
            headerColor = ImVec4(0.26f, 0.52f, 0.96f, 1.0f);
        }

        std::string getTypeId() const override { return "GeoV2.MaskByHeight"; }

        void serializeParams(nlohmann::json& j) const override {
            j["attribute"] = std::string(attributeName);
            j["min"] = minV;
            j["max"] = maxV;
            j["relative"] = relativeToBBox;
            j["invert"] = invert;
        }
        void deserializeParams(const nlohmann::json& j) override {
            const std::string a = j.value("attribute", std::string("mask"));
            snprintf(attributeName, sizeof(attributeName), "%s", a.c_str());
            minV = j.value("min", 0.3f);
            maxV = j.value("max", 0.7f);
            relativeToBBox = j.value("relative", true);
            invert = j.value("invert", false);
        }

        void drawContent() override {
            ImGui::SetNextItemWidth(140);
            ImGui::InputText("Attribute", attributeName, sizeof(attributeName));
            ImGui::SetNextItemWidth(140);
            ImGui::DragFloat("Min", &minV, 0.005f);
            ImGui::SetNextItemWidth(140);
            ImGui::DragFloat("Max", &maxV, 0.005f);
            ImGui::Checkbox("Relative (0-1 of height)", &relativeToBBox);
            ImGui::Checkbox("Invert", &invert);
            if (lastMin >= 0.0f) {
                ImGui::TextDisabled("Last eval: min %.2f  mean %.2f  max %.2f", lastMin, lastMean, lastMax);
                if (lastMax - lastMin < 0.01f) {
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.7f, 0.3f, 1.0f));
                    ImGui::TextWrapped("Mask is (nearly) uniform. A flat mesh has no height variation - displace it first, or use Mask by Noise instead.");
                    ImGui::PopStyleColor();
                }
            }
        }

        NodeSystem::PinValue compute(int /*outputIndex*/, NodeSystem::EvaluationContext& ctx) override {
            NodeSystem::GeometryValue inMesh = getGeometryInput(0, ctx);
            if (!inMesh || !inMesh->geometry || inMesh->geometry->indices.empty()) {
                ctx.addError(id, "Mask by Height: no input geometry");
                return NodeSystem::PinValue{};
            }
            auto out = deepCopyMesh(inMesh);
            auto& geom = *out->geometry;
            const size_t vc = geom.get_vertex_count();
            const Vec3* Po = geom.get_attribute_data<Vec3>("P_orig");
            float* m = ensureFloatAttribute(geom, attributeName);
            if (!Po || !m) {
                ctx.addError(id, "Mask by Height: missing P_orig or invalid attribute name");
                return NodeSystem::PinValue{};
            }

            float lo = minV, hi = maxV;
            if (relativeToBBox) {
                float yMin = Po[0].y, yMax = Po[0].y;
                for (size_t v = 1; v < vc; ++v) {
                    yMin = (std::min)(yMin, Po[v].y);
                    yMax = (std::max)(yMax, Po[v].y);
                }
                const float range = yMax - yMin;
                lo = yMin + minV * range;
                hi = yMin + maxV * range;
            }
            const float span = hi - lo;
            for (size_t v = 0; v < vc; ++v) {
                float s;
                if (span <= 1e-12f) {
                    s = (Po[v].y >= hi) ? 1.0f : 0.0f;
                } else {
                    float t = (Po[v].y - lo) / span;
                    t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
                    s = t * t * (3.0f - 2.0f * t);  // smoothstep
                }
                m[v] = invert ? 1.0f - s : s;
            }

            lastMin = 1.0f; lastMax = 0.0f; lastMean = 0.0f;
            for (size_t v = 0; v < vc; ++v) {
                lastMin = (std::min)(lastMin, m[v]);
                lastMax = (std::max)(lastMax, m[v]);
                lastMean += m[v];
            }
            lastMean /= static_cast<float>(vc);

            return NodeSystem::GeometryValue(out);
        }
    };

    /**
     * @brief Write a mask from surface slope: the angle between N_orig and local up
     * (+Y). 0° = flat horizontal, 90° = vertical wall, 180° = facing straight down.
     * Smoothstep ramp from Min Angle to Max Angle (steeper = more masked); Invert
     * flips it (flat = masked). The classic driver for "scatter on flat ground only" /
     * "rock detail on cliffs only".
     */
    class MaskBySlopeNode : public GeometryNodeBase {
    public:
        char attributeName[64] = "mask";
        float minAngle = 20.0f;  ///< Degrees — ramp start
        float maxAngle = 60.0f;  ///< Degrees — ramp end
        bool invert = false;
        float lastMin = -1.0f, lastMean = -1.0f, lastMax = -1.0f;  ///< Live diagnostic (last eval)

        MaskBySlopeNode() {
            name = "Mask by Slope";
            geometryNodeType = NodeType::MaskBySlope;

            inputs.push_back(NodeSystem::Pin::createInput("Geometry", NodeSystem::DataType::Geometry));
            outputs.push_back(NodeSystem::Pin::createOutput("Geometry", NodeSystem::DataType::Geometry));

            metadata.displayName = "Mask by Slope";
            metadata.category = "Field";
            metadata.description = "Per-vertex mask attribute from surface steepness (angle vs. local up).";
            metadata.headerColor = IM_COL32(66, 133, 244, 255);
            headerColor = ImVec4(0.26f, 0.52f, 0.96f, 1.0f);
        }

        std::string getTypeId() const override { return "GeoV2.MaskBySlope"; }

        void serializeParams(nlohmann::json& j) const override {
            j["attribute"] = std::string(attributeName);
            j["min_angle"] = minAngle;
            j["max_angle"] = maxAngle;
            j["invert"] = invert;
        }
        void deserializeParams(const nlohmann::json& j) override {
            const std::string a = j.value("attribute", std::string("mask"));
            snprintf(attributeName, sizeof(attributeName), "%s", a.c_str());
            minAngle = j.value("min_angle", 20.0f);
            maxAngle = j.value("max_angle", 60.0f);
            invert = j.value("invert", false);
        }

        void drawContent() override {
            ImGui::SetNextItemWidth(140);
            ImGui::InputText("Attribute", attributeName, sizeof(attributeName));
            ImGui::SetNextItemWidth(140);
            ImGui::SliderFloat("Min Angle", &minAngle, 0.0f, 180.0f, "%.1f deg");
            ImGui::SetNextItemWidth(140);
            ImGui::SliderFloat("Max Angle", &maxAngle, 0.0f, 180.0f, "%.1f deg");
            ImGui::Checkbox("Invert", &invert);
            if (lastMin >= 0.0f) {
                ImGui::TextDisabled("Last eval: min %.2f  mean %.2f  max %.2f", lastMin, lastMean, lastMax);
                if (lastMax - lastMin < 0.01f) {
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.7f, 0.3f, 1.0f));
                    ImGui::TextWrapped("Mask is (nearly) uniform. A flat mesh has one slope everywhere - displace it first, or use Mask by Noise instead.");
                    ImGui::PopStyleColor();
                }
            }
        }

        NodeSystem::PinValue compute(int /*outputIndex*/, NodeSystem::EvaluationContext& ctx) override {
            NodeSystem::GeometryValue inMesh = getGeometryInput(0, ctx);
            if (!inMesh || !inMesh->geometry || inMesh->geometry->indices.empty()) {
                ctx.addError(id, "Mask by Slope: no input geometry");
                return NodeSystem::PinValue{};
            }
            auto out = deepCopyMesh(inMesh);
            auto& geom = *out->geometry;
            const size_t vc = geom.get_vertex_count();
            const Vec3* No = geom.get_attribute_data<Vec3>("N_orig");
            float* m = ensureFloatAttribute(geom, attributeName);
            if (!No || !m) {
                ctx.addError(id, "Mask by Slope: missing N_orig or invalid attribute name");
                return NodeSystem::PinValue{};
            }

            const float lo = (std::min)(minAngle, maxAngle);
            const float hi = (std::max)(minAngle, maxAngle);
            const float span = hi - lo;
            constexpr float kRad2Deg = 57.2957795f;
            for (size_t v = 0; v < vc; ++v) {
                float ny = No[v].y;
                ny = ny < -1.0f ? -1.0f : (ny > 1.0f ? 1.0f : ny);
                const float angle = std::acos(ny) * kRad2Deg;
                float s;
                if (span <= 1e-6f) {
                    s = (angle >= hi) ? 1.0f : 0.0f;
                } else {
                    float t = (angle - lo) / span;
                    t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
                    s = t * t * (3.0f - 2.0f * t);
                }
                m[v] = invert ? 1.0f - s : s;
            }

            lastMin = 1.0f; lastMax = 0.0f; lastMean = 0.0f;
            for (size_t v = 0; v < vc; ++v) {
                lastMin = (std::min)(lastMin, m[v]);
                lastMax = (std::max)(lastMax, m[v]);
                lastMean += m[v];
            }
            lastMean /= static_cast<float>(vc);

            return NodeSystem::GeometryValue(out);
        }
    };

    /**
     * @brief Write a mask from procedural 3D noise sampled at each vertex's LOCAL
     * position — works on ANY geometry, including a perfectly flat plane where
     * Height/Slope masks are inherently uniform (no variation to key off). Noise is
     * normalized to 0..1, then a smoothstep contrast window (Min..Max) shapes it:
     * a narrow window gives crisp islands, a wide one soft gradients.
     */
    class MaskNoiseNode : public GeometryNodeBase {
    public:
        char attributeName[64] = "mask";
        NoiseKind noiseType = NoiseKind::FBM;
        float frequency = 1.0f;
        int   octaves = 4;
        int   seed = 0;
        float levelLo = 0.4f;   ///< Contrast window start (on 0..1 normalized noise)
        float levelHi = 0.6f;   ///< Contrast window end
        bool  invert = false;
        float lastMin = -1.0f, lastMean = -1.0f, lastMax = -1.0f;  ///< Live diagnostic (last eval)

        MaskNoiseNode() {
            name = "Mask by Noise";
            geometryNodeType = NodeType::MaskNoise;

            inputs.push_back(NodeSystem::Pin::createInput("Geometry", NodeSystem::DataType::Geometry));
            outputs.push_back(NodeSystem::Pin::createOutput("Geometry", NodeSystem::DataType::Geometry));

            metadata.displayName = "Mask by Noise";
            metadata.category = "Field";
            metadata.description = "Per-vertex mask from procedural noise — works on any shape, including flat planes.";
            metadata.headerColor = IM_COL32(66, 133, 244, 255);
            headerColor = ImVec4(0.26f, 0.52f, 0.96f, 1.0f);
        }

        std::string getTypeId() const override { return "GeoV2.MaskNoise"; }

        void serializeParams(nlohmann::json& j) const override {
            j["attribute"] = std::string(attributeName);
            j["noise_type"] = static_cast<int>(noiseType);
            j["frequency"] = frequency;
            j["octaves"] = octaves;
            j["seed"] = seed;
            j["level_lo"] = levelLo;
            j["level_hi"] = levelHi;
            j["invert"] = invert;
        }
        void deserializeParams(const nlohmann::json& j) override {
            const std::string a = j.value("attribute", std::string("mask"));
            snprintf(attributeName, sizeof(attributeName), "%s", a.c_str());
            const int t = j.value("noise_type", 0);
            noiseType = (t >= 0 && t <= static_cast<int>(NoiseKind::VoronoiCrackle))
                ? static_cast<NoiseKind>(t) : NoiseKind::FBM;
            frequency = j.value("frequency", 1.0f);
            octaves = j.value("octaves", 4);
            seed = j.value("seed", 0);
            levelLo = j.value("level_lo", 0.4f);
            levelHi = j.value("level_hi", 0.6f);
            invert = j.value("invert", false);
        }

        void drawContent() override {
            ImGui::SetNextItemWidth(140);
            ImGui::InputText("Attribute", attributeName, sizeof(attributeName));
            const char* typeNames[] = {
                "FBM", "Perlin", "Simplex", "Turbulence",
                "Ridge", "Billow", "Voronoi", "Voronoi Crackle"
            };
            int typeIdx = static_cast<int>(noiseType);
            ImGui::SetNextItemWidth(140);
            if (ImGui::Combo("Type", &typeIdx, typeNames, IM_ARRAYSIZE(typeNames))) {
                noiseType = static_cast<NoiseKind>(typeIdx);
            }
            ImGui::SetNextItemWidth(140);
            ImGui::DragFloat("Frequency", &frequency, 0.01f, 0.001f, 1000.0f);
            const bool fractal = noiseType == NoiseKind::FBM || noiseType == NoiseKind::Turbulence
                              || noiseType == NoiseKind::Ridge || noiseType == NoiseKind::Billow;
            if (fractal) {
                ImGui::SetNextItemWidth(140);
                ImGui::SliderInt("Octaves", &octaves, 1, 8);
            }
            ImGui::SetNextItemWidth(140);
            ImGui::DragInt("Seed", &seed, 1.0f);
            ImGui::SetNextItemWidth(140);
            ImGui::SliderFloat("Min", &levelLo, 0.0f, 1.0f);
            ImGui::SetNextItemWidth(140);
            ImGui::SliderFloat("Max", &levelHi, 0.0f, 1.0f);
            ImGui::Checkbox("Invert", &invert);
            if (lastMin >= 0.0f) {
                ImGui::TextDisabled("Last eval: min %.2f  mean %.2f  max %.2f", lastMin, lastMean, lastMax);
            }
        }

        NodeSystem::PinValue compute(int /*outputIndex*/, NodeSystem::EvaluationContext& ctx) override {
            NodeSystem::GeometryValue inMesh = getGeometryInput(0, ctx);
            if (!inMesh || !inMesh->geometry || inMesh->geometry->indices.empty()) {
                ctx.addError(id, "Mask by Noise: no input geometry");
                return NodeSystem::PinValue{};
            }
            auto out = deepCopyMesh(inMesh);
            auto& geom = *out->geometry;
            const size_t vc = geom.get_vertex_count();
            const Vec3* Po = geom.get_attribute_data<Vec3>("P_orig");
            float* m = ensureFloatAttribute(geom, attributeName);
            if (!Po || !m) {
                ctx.addError(id, "Mask by Noise: missing P_orig or invalid attribute name");
                return NodeSystem::PinValue{};
            }

            const float lo = (std::min)(levelLo, levelHi);
            const float hi = (std::max)(levelLo, levelHi);
            const float span = hi - lo;
            for (size_t v = 0; v < vc; ++v) {
                float n = sampleNoiseKind(noiseType, Po[v], octaves, frequency, seed);
                if (isSignedNoiseKind(noiseType)) n = n * 0.5f + 0.5f;
                n = n < 0.0f ? 0.0f : (n > 1.0f ? 1.0f : n);
                float s;
                if (span <= 1e-6f) {
                    s = (n >= hi) ? 1.0f : 0.0f;
                } else {
                    float t = (n - lo) / span;
                    t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
                    s = t * t * (3.0f - 2.0f * t);
                }
                m[v] = invert ? 1.0f - s : s;
            }

            lastMin = 1.0f; lastMax = 0.0f; lastMean = 0.0f;
            for (size_t v = 0; v < vc; ++v) {
                lastMin = (std::min)(lastMin, m[v]);
                lastMax = (std::max)(lastMax, m[v]);
                lastMean += m[v];
            }
            lastMean /= static_cast<float>(vc);

            return NodeSystem::GeometryValue(out);
        }
    };

    // ============================================================================
    // MASK REMAP NODE (Faz 8b) — contrast/gamma/invert shaping for any Field
    // ============================================================================

    /**
     * @brief Reshape an existing Field attribute's values in place: gamma curve +
     * contrast window (smoothstep Min..Max, same control as MaskByHeight/MaskNoise) +
     * invert. Decouples "how a mask is produced" (Height/Slope/Noise/painted) from
     * "how its falloff feels" — chain this after ANY mask producer instead of each
     * producer growing its own curve controls. A dedicated curve-widget version (reusing
     * the sculpt falloff-curve editor) is a natural follow-up; this covers the common
     * cases (steeper/softer transition, push toward 0/1) with two sliders.
     */
    class MaskRemapNode : public GeometryNodeBase {
    public:
        char attributeName[64] = "mask";
        float gamma = 1.0f;      ///< <1 brightens (grows the masked area), >1 darkens/shrinks it
        float levelLo = 0.0f;    ///< Contrast window start (post-gamma)
        float levelHi = 1.0f;    ///< Contrast window end
        bool  invert = false;
        float lastMin = -1.0f, lastMean = -1.0f, lastMax = -1.0f;

        MaskRemapNode() {
            name = "Mask Remap";
            geometryNodeType = NodeType::MaskRemap;

            inputs.push_back(NodeSystem::Pin::createInput("Geometry", NodeSystem::DataType::Geometry));
            outputs.push_back(NodeSystem::Pin::createOutput("Geometry", NodeSystem::DataType::Geometry));

            metadata.displayName = "Mask Remap";
            metadata.category = "Field";
            metadata.description = "Reshape a Field's falloff: gamma + contrast window + invert. Chain after any mask producer.";
            metadata.headerColor = IM_COL32(66, 133, 244, 255);
            headerColor = ImVec4(0.26f, 0.52f, 0.96f, 1.0f);
        }

        std::string getTypeId() const override { return "GeoV2.MaskRemap"; }

        void serializeParams(nlohmann::json& j) const override {
            j["attribute"] = std::string(attributeName);
            j["gamma"] = gamma;
            j["level_lo"] = levelLo;
            j["level_hi"] = levelHi;
            j["invert"] = invert;
        }
        void deserializeParams(const nlohmann::json& j) override {
            const std::string a = j.value("attribute", std::string("mask"));
            snprintf(attributeName, sizeof(attributeName), "%s", a.c_str());
            gamma = j.value("gamma", 1.0f);
            levelLo = j.value("level_lo", 0.0f);
            levelHi = j.value("level_hi", 1.0f);
            invert = j.value("invert", false);
        }

        void drawContent() override {
            ImGui::SetNextItemWidth(140);
            ImGui::InputText("Attribute", attributeName, sizeof(attributeName));
            ImGui::SetNextItemWidth(140);
            ImGui::DragFloat("Gamma", &gamma, 0.01f, 0.01f, 10.0f);
            ImGui::SetNextItemWidth(140);
            ImGui::SliderFloat("Min", &levelLo, 0.0f, 1.0f);
            ImGui::SetNextItemWidth(140);
            ImGui::SliderFloat("Max", &levelHi, 0.0f, 1.0f);
            ImGui::Checkbox("Invert", &invert);
            if (lastMin >= 0.0f) {
                ImGui::TextDisabled("Last eval: min %.2f  mean %.2f  max %.2f", lastMin, lastMean, lastMax);
            }
        }

        NodeSystem::PinValue compute(int /*outputIndex*/, NodeSystem::EvaluationContext& ctx) override {
            NodeSystem::GeometryValue inMesh = getGeometryInput(0, ctx);
            if (!inMesh || !inMesh->geometry || inMesh->geometry->indices.empty()) {
                ctx.addError(id, "Mask Remap: no input geometry");
                return NodeSystem::PinValue{};
            }
            auto out = deepCopyMesh(inMesh);
            auto& geom = *out->geometry;
            float* m = geom.get_attribute_data_mut<float>(attributeName);
            if (!m) {
                ctx.addError(id, std::string("Mask Remap: attribute '") + attributeName + "' not found — add a Mask node upstream");
                return NodeSystem::PinValue{};
            }
            const size_t vc = geom.get_vertex_count();
            const float lo = (std::min)(levelLo, levelHi);
            const float hi = (std::max)(levelLo, levelHi);
            const float span = hi - lo;
            const float g = (gamma > 0.0001f) ? gamma : 0.0001f;
            for (size_t v = 0; v < vc; ++v) {
                float x = m[v] < 0.0f ? 0.0f : (m[v] > 1.0f ? 1.0f : m[v]);
                x = std::pow(x, g);
                float s;
                if (span <= 1e-6f) {
                    s = (x >= hi) ? 1.0f : 0.0f;
                } else {
                    float t = (x - lo) / span;
                    t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
                    s = t * t * (3.0f - 2.0f * t);
                }
                m[v] = invert ? 1.0f - s : s;
            }

            lastMin = 1.0f; lastMax = 0.0f; lastMean = 0.0f;
            for (size_t v = 0; v < vc; ++v) {
                lastMin = (std::min)(lastMin, m[v]);
                lastMax = (std::max)(lastMax, m[v]);
                lastMean += m[v];
            }
            if (vc > 0) lastMean /= static_cast<float>(vc);

            return NodeSystem::GeometryValue(out);
        }
    };

    // ============================================================================
    // MASK MATH NODE (Faz 8b) — combine two Fields
    // ============================================================================

    enum class MaskMathOp { Add, Multiply, Min, Max, Subtract, Average };

    /**
     * @brief Combine two Field attributes (e.g. Height mask AND Slope mask) into an
     * output attribute via a per-vertex binary op. Multiply is the common "intersect"
     * case (both conditions must hold); Max is "union"; Add/Average blend; Subtract
     * carves B out of A. Result is clamped to 0..1 (Add/Subtract can overshoot).
     */
    class MaskMathNode : public GeometryNodeBase {
    public:
        char attributeA[64] = "mask";
        char attributeB[64] = "mask";
        char outputAttribute[64] = "mask";
        MaskMathOp op = MaskMathOp::Multiply;

        MaskMathNode() {
            name = "Mask Math";
            geometryNodeType = NodeType::MaskMath;

            inputs.push_back(NodeSystem::Pin::createInput("Geometry", NodeSystem::DataType::Geometry));
            outputs.push_back(NodeSystem::Pin::createOutput("Geometry", NodeSystem::DataType::Geometry));

            metadata.displayName = "Mask Math";
            metadata.category = "Field";
            metadata.description = "Combine two Field attributes (Multiply = intersect, Max = union, ...) into an output attribute.";
            metadata.headerColor = IM_COL32(66, 133, 244, 255);
            headerColor = ImVec4(0.26f, 0.52f, 0.96f, 1.0f);
        }

        std::string getTypeId() const override { return "GeoV2.MaskMath"; }

        void serializeParams(nlohmann::json& j) const override {
            j["attribute_a"] = std::string(attributeA);
            j["attribute_b"] = std::string(attributeB);
            j["output_attribute"] = std::string(outputAttribute);
            j["op"] = static_cast<int>(op);
        }
        void deserializeParams(const nlohmann::json& j) override {
            const std::string a = j.value("attribute_a", std::string("mask"));
            snprintf(attributeA, sizeof(attributeA), "%s", a.c_str());
            const std::string b = j.value("attribute_b", std::string("mask"));
            snprintf(attributeB, sizeof(attributeB), "%s", b.c_str());
            const std::string o = j.value("output_attribute", std::string("mask"));
            snprintf(outputAttribute, sizeof(outputAttribute), "%s", o.c_str());
            const int opv = j.value("op", 1);
            op = (opv >= 0 && opv <= static_cast<int>(MaskMathOp::Average)) ? static_cast<MaskMathOp>(opv) : MaskMathOp::Multiply;
        }

        void drawContent() override {
            ImGui::SetNextItemWidth(140);
            ImGui::InputText("Attribute A", attributeA, sizeof(attributeA));
            ImGui::SetNextItemWidth(140);
            ImGui::InputText("Attribute B", attributeB, sizeof(attributeB));
            const char* opNames[] = { "Add", "Multiply", "Min", "Max", "Subtract (A-B)", "Average" };
            int opIdx = static_cast<int>(op);
            ImGui::SetNextItemWidth(160);
            if (ImGui::Combo("Operation", &opIdx, opNames, IM_ARRAYSIZE(opNames))) {
                op = static_cast<MaskMathOp>(opIdx);
            }
            ImGui::SetNextItemWidth(140);
            ImGui::InputText("Output Attr", outputAttribute, sizeof(outputAttribute));
        }

        NodeSystem::PinValue compute(int /*outputIndex*/, NodeSystem::EvaluationContext& ctx) override {
            NodeSystem::GeometryValue inMesh = getGeometryInput(0, ctx);
            if (!inMesh || !inMesh->geometry || inMesh->geometry->indices.empty()) {
                ctx.addError(id, "Mask Math: no input geometry");
                return NodeSystem::PinValue{};
            }
            auto out = deepCopyMesh(inMesh);
            auto& geom = *out->geometry;
            const float* a = geom.get_attribute_data<float>(attributeA);
            const float* b = geom.get_attribute_data<float>(attributeB);
            if (!a || !b) {
                ctx.addError(id, std::string("Mask Math: attribute '") + (!a ? attributeA : attributeB) + "' not found — add a Mask node upstream");
                return NodeSystem::PinValue{};
            }
            const size_t vc = geom.get_vertex_count();
            // Snapshot A/B before writing the output — needed when Output aliases A or B
            // (the common case: "combine into the same mask"), since get_attribute_data_mut
            // could otherwise return the SAME buffer as `a`/`b` and corrupt the read as we write.
            std::vector<float> av(a, a + vc), bv(b, b + vc);
            float* outAttr = ensureFloatAttribute(geom, outputAttribute);
            if (!outAttr) {
                ctx.addError(id, "Mask Math: invalid output attribute name");
                return NodeSystem::PinValue{};
            }
            for (size_t v = 0; v < vc; ++v) {
                float r;
                switch (op) {
                    case MaskMathOp::Add:      r = av[v] + bv[v]; break;
                    case MaskMathOp::Multiply: r = av[v] * bv[v]; break;
                    case MaskMathOp::Min:      r = (std::min)(av[v], bv[v]); break;
                    case MaskMathOp::Max:      r = (std::max)(av[v], bv[v]); break;
                    case MaskMathOp::Subtract: r = av[v] - bv[v]; break;
                    case MaskMathOp::Average:  r = (av[v] + bv[v]) * 0.5f; break;
                    default:                   r = av[v]; break;
                }
                outAttr[v] = r < 0.0f ? 0.0f : (r > 1.0f ? 1.0f : r);
            }
            return NodeSystem::GeometryValue(out);
        }
    };

    // ============================================================================
    // SCATTER INSTANCES NODE (Faz 8b) — mask-driven surface scattering
    // ============================================================================

    /**
     * @brief Scatter instances of another scene object over the input surface —
     * the Geo-DAG face of the existing foliage/instancing system. Samples the input
     * geometry area-uniformly (optionally weighted by a Field mask: mask 0 = never,
     * 1 = full density — the payoff of MaskByHeight/Slope/Noise), then creates or
     * REPLACES one InstanceGroup in InstanceManager (identified by a stable,
     * serialized group name), so re-Evaluating updates the scatter instead of
     * stacking duplicates. Renders through the exact same instancing pipeline the
     * scatter brush / foliage system already uses.
     *
     * Geometry-wise this node is a pure PASS-THROUGH (the surface is unchanged);
     * the instances are a side effect applied by evaluateGeometryGraph via
     * GeometryContext::instancesDirty. compute() is defined in MeshModifiers.cpp —
     * it needs InstanceManager, which this header deliberately doesn't include.
     */
    class ScatterInstancesNode : public GeometryNodeBase {
    public:
        char  sourceObject[128] = "";   ///< Scene object to instance (picker combo)
        char  groupName[128] = "";      ///< Stable InstanceGroup identity — auto-generated once, serialized
        int   count = 500;              ///< Target instance count
        int   seed = 1234;              ///< Deterministic placement seed
        float minDistance = 0.0f;       ///< 0 = off; otherwise reject samples closer than this to an existing one
        float scaleMin = 0.8f;
        float scaleMax = 1.2f;
        float yawRandom = 360.0f;       ///< Random Y rotation (degrees)
        float tiltRandom = 5.0f;        ///< Random XZ tilt (degrees)
        bool  alignToNormal = true;
        float normalInfluence = 1.0f;   ///< 0 = world up, 1 = full surface normal
        // Faz 8b Field bridge: DENSITY and SCALE read independent named attributes
        // (Blender vertex-group-per-slot style) instead of one attribute driving both —
        // e.g. gate density from a paint mask while size falls off from a separate
        // "edge distance" attribute.
        bool  useDensityMask = false;
        char  densityMaskAttribute[64] = "mask";
        bool  useScaleMask = false;
        char  scaleMaskAttribute[64] = "mask";
        float scaleMaskInfluence = 1.0f; ///< 0 = scale mask ignored; 1 = scale fully follows it
        int   lastPlaced = -1;          ///< Live diagnostic: instances placed on last eval
        int   lastDensityMaskState = -1; ///< -1 off, 0 missing, 1 found (last eval)
        int   lastScaleMaskState = -1;   ///< -1 off, 0 missing, 1 found (last eval)

        ScatterInstancesNode() {
            name = "Scatter Instances";
            geometryNodeType = NodeType::ScatterInstances;

            inputs.push_back(NodeSystem::Pin::createInput("Geometry", NodeSystem::DataType::Geometry));
            outputs.push_back(NodeSystem::Pin::createOutput("Geometry", NodeSystem::DataType::Geometry));

            metadata.displayName = "Scatter Instances";
            metadata.category = "Instancing";
            metadata.description = "Scatter another object over this surface (optionally mask-weighted) via the foliage instancing pipeline.";
            metadata.headerColor = IM_COL32(255, 193, 7, 255);   // amber — Instances family
            headerColor = ImVec4(1.0f, 0.76f, 0.03f, 1.0f);
        }

        std::string getTypeId() const override { return "GeoV2.ScatterInstances"; }

        void serializeParams(nlohmann::json& j) const override {
            j["source"] = std::string(sourceObject);
            j["group_name"] = std::string(groupName);
            j["count"] = count;
            j["seed"] = seed;
            j["min_distance"] = minDistance;
            j["scale_min"] = scaleMin;
            j["scale_max"] = scaleMax;
            j["yaw_random"] = yawRandom;
            j["tilt_random"] = tiltRandom;
            j["align_to_normal"] = alignToNormal;
            j["normal_influence"] = normalInfluence;
            j["use_density_mask"] = useDensityMask;
            j["density_mask_attribute"] = std::string(densityMaskAttribute);
            j["use_scale_mask"] = useScaleMask;
            j["scale_mask_attribute"] = std::string(scaleMaskAttribute);
            j["scale_mask_influence"] = scaleMaskInfluence;
        }
        void deserializeParams(const nlohmann::json& j) override {
            const std::string src = j.value("source", std::string());
            snprintf(sourceObject, sizeof(sourceObject), "%s", src.c_str());
            const std::string grp = j.value("group_name", std::string());
            snprintf(groupName, sizeof(groupName), "%s", grp.c_str());
            count = j.value("count", 500);
            seed = j.value("seed", 1234);
            minDistance = j.value("min_distance", 0.0f);
            scaleMin = j.value("scale_min", 0.8f);
            scaleMax = j.value("scale_max", 1.2f);
            yawRandom = j.value("yaw_random", 360.0f);
            tiltRandom = j.value("tilt_random", 5.0f);
            alignToNormal = j.value("align_to_normal", true);
            normalInfluence = j.value("normal_influence", 1.0f);
            // Legacy single-attribute save (pre Round 9 split): both roles inherit it.
            const bool legacyUseMask = j.value("use_mask", false);
            const std::string legacyAttr = j.value("mask_attribute", std::string("mask"));
            const float legacyInfluence = j.value("mask_scale_influence", 0.0f);
            useDensityMask = j.value("use_density_mask", legacyUseMask);
            const std::string da = j.value("density_mask_attribute", legacyAttr);
            snprintf(densityMaskAttribute, sizeof(densityMaskAttribute), "%s", da.c_str());
            useScaleMask = j.value("use_scale_mask", legacyUseMask && legacyInfluence > 0.0f);
            const std::string sa = j.value("scale_mask_attribute", legacyAttr);
            snprintf(scaleMaskAttribute, sizeof(scaleMaskAttribute), "%s", sa.c_str());
            scaleMaskInfluence = j.value("scale_mask_influence", legacyInfluence > 0.0f ? legacyInfluence : 1.0f);
        }

        void drawContent() override {
            ImGui::SetNextItemWidth(160);
            if (g_sceneObjectListProvider) {
                const char* preview = sourceObject[0] ? sourceObject : "<select object>";
                if (ImGui::BeginCombo("Source", preview)) {
                    const std::vector<std::string> names = g_sceneObjectListProvider();
                    for (const auto& n : names) {
                        const bool selected = (n == sourceObject);
                        if (ImGui::Selectable(n.c_str(), selected)) {
                            snprintf(sourceObject, sizeof(sourceObject), "%s", n.c_str());
                        }
                        if (selected) ImGui::SetItemDefaultFocus();
                    }
                    if (names.empty()) ImGui::TextDisabled("(no flat mesh objects)");
                    ImGui::EndCombo();
                }
            } else {
                ImGui::InputText("Source", sourceObject, sizeof(sourceObject));
            }
            ImGui::SetNextItemWidth(140);
            ImGui::DragInt("Count", &count, 1.0f, 0, 1000000);
            ImGui::SetNextItemWidth(140);
            ImGui::DragInt("Seed", &seed, 1.0f);
            ImGui::SetNextItemWidth(140);
            ImGui::DragFloat("Min Distance", &minDistance, 0.01f, 0.0f, 1000.0f);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("0 = off. Otherwise rejects a candidate if it lands closer than this\nto an already-placed instance (same idea as the scatter brush's own\nMin Distance) - reduces clumping at high Count.");
            }
            ImGui::SetNextItemWidth(140);
            ImGui::DragFloat("Scale Min", &scaleMin, 0.01f, 0.001f, 100.0f);
            ImGui::SetNextItemWidth(140);
            ImGui::DragFloat("Scale Max", &scaleMax, 0.01f, 0.001f, 100.0f);
            ImGui::SetNextItemWidth(140);
            ImGui::SliderFloat("Yaw Random", &yawRandom, 0.0f, 360.0f, "%.0f deg");
            ImGui::SetNextItemWidth(140);
            ImGui::SliderFloat("Tilt Random", &tiltRandom, 0.0f, 45.0f, "%.1f deg");
            ImGui::Checkbox("Align To Normal", &alignToNormal);
            if (alignToNormal) {
                ImGui::SetNextItemWidth(140);
                ImGui::SliderFloat("Normal Influence", &normalInfluence, 0.0f, 1.0f);
            }
            // Faz 8b Field bridge: density and scale read INDEPENDENT named attributes
            // (Blender vertex-group-per-slot style) instead of one attribute driving both.
            ImGui::Checkbox("Use Density Mask", &useDensityMask);
            if (useDensityMask) {
                ImGui::SetNextItemWidth(140);
                ImGui::InputText("Density Attr", densityMaskAttribute, sizeof(densityMaskAttribute));
                if (lastDensityMaskState == 1) {
                    ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.6f, 1.0f), "Last eval: found");
                } else if (lastDensityMaskState == 0) {
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.45f, 0.3f, 1.0f));
                    ImGui::TextWrapped("Last eval: NOT found - add a Mask node upstream (after Subdivide).");
                    ImGui::PopStyleColor();
                }
            }
            ImGui::Checkbox("Use Scale Mask", &useScaleMask);
            if (useScaleMask) {
                ImGui::SetNextItemWidth(140);
                ImGui::InputText("Scale Attr", scaleMaskAttribute, sizeof(scaleMaskAttribute));
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Can be a different attribute than Density Mask\n(e.g. size falls off near a patch edge while density\nstays gated by a separate paint mask).");
                }
                ImGui::SetNextItemWidth(140);
                ImGui::SliderFloat("Mask > Scale", &scaleMaskInfluence, 0.0f, 1.0f);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("0 = scale mask ignored, 1 = instance scale fully follows it\n(small plants at mask edges, full-size in the core).");
                }
                if (lastScaleMaskState == 1) {
                    ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.6f, 1.0f), "Last eval: found");
                } else if (lastScaleMaskState == 0) {
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.45f, 0.3f, 1.0f));
                    ImGui::TextWrapped("Last eval: NOT found - add a Mask node upstream (after Subdivide).");
                    ImGui::PopStyleColor();
                }
            }
            if (lastPlaced >= 0) {
                ImGui::TextDisabled("Last eval: %d instances placed", lastPlaced);
            }
        }

        // Defined in MeshModifiers.cpp (needs InstanceManager).
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
    };

    // ============================================================================
    // REMESH NODE (voxel remesh)
    // ============================================================================

    /**
     * @brief Blender-style VOXEL remesh: the input surface is converted to a
     * narrow-band signed distance field (unsigned distance stamped per triangle,
     * sign from per-column ray-parity — Bridson's makelevelset3 scheme) and the
     * iso-0 surface is re-extracted with marching cubes into a fully welded,
     * uniform, manifold triangulation. Rebuilds topology from scratch: fixes
     * self-intersections / soup / boolean junk, at the cost of UVs and custom
     * attributes (masks) — same trade Blender's voxel remesh makes. Material is
     * carried over per-vertex from the nearest input triangle (single-material
     * regions stay intact; boundaries re-snap to the voxel resolution).
     *
     * Works best on CLOSED surfaces — the ray-parity sign is undefined across
     * open boundaries (an open plane has no "inside"); thin open sheets may
     * vanish or thicken, exactly like Blender's voxel remesher.
     */
    class RemeshNode : public GeometryNodeBase {
    public:
        float voxelSize = 0.02f;        ///< Relative mode: fraction of the largest bbox side. Absolute: local units.
        bool  relativeSize = true;      ///< ON: voxelSize is a 0..1 fraction of the mesh's own size (unit-independent)
        int   smoothIterations = 0;     ///< Laplacian post-smooth passes on the remeshed surface (0 = off)
        float smoothFactor = 0.5f;      ///< Per-pass blend toward the neighbor average
        bool  smoothShading = true;     ///< ON: area-weighted vertex normals; OFF: faceted (flat) normals
        bool  transferMaterial = true;  ///< Copy materialID from the nearest input triangle
        // Live diagnostics (last eval)
        int   lastGridX = 0, lastGridY = 0, lastGridZ = 0;
        int   lastTrisOut = -1;

        RemeshNode() {
            name = "Remesh";
            geometryNodeType = NodeType::Remesh;

            inputs.push_back(NodeSystem::Pin::createInput("Geometry", NodeSystem::DataType::Geometry));
            outputs.push_back(NodeSystem::Pin::createOutput("Geometry", NodeSystem::DataType::Geometry));

            metadata.displayName = "Remesh (Voxel)";
            metadata.category = "Geometry";
            metadata.description = "Rebuild the surface as a uniform, welded, manifold mesh via an SDF + marching cubes. Loses UVs/masks.";
            metadata.headerColor = IM_COL32(76, 175, 80, 255);
            headerColor = ImVec4(0.3f, 0.68f, 0.31f, 1.0f);
        }

        std::string getTypeId() const override { return "GeoV2.Remesh"; }

        void serializeParams(nlohmann::json& j) const override {
            j["voxel_size"] = voxelSize;
            j["relative_size"] = relativeSize;
            j["smooth_iterations"] = smoothIterations;
            j["smooth_factor"] = smoothFactor;
            j["smooth_shading"] = smoothShading;
            j["transfer_material"] = transferMaterial;
        }
        void deserializeParams(const nlohmann::json& j) override {
            voxelSize = j.value("voxel_size", 0.02f);
            relativeSize = j.value("relative_size", true);
            smoothIterations = j.value("smooth_iterations", 0);
            smoothFactor = j.value("smooth_factor", 0.5f);
            smoothShading = j.value("smooth_shading", true);
            transferMaterial = j.value("transfer_material", true);
        }

        void drawContent() override {
            ImGui::SetNextItemWidth(140);
            ImGui::DragFloat("Voxel Size", &voxelSize, relativeSize ? 0.001f : 0.01f,
                             relativeSize ? 0.002f : 0.0001f, relativeSize ? 0.25f : 1000.0f, "%.4f");
            ImGui::Checkbox("Relative", &relativeSize);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("ON: Voxel Size is a fraction of the mesh's largest bounding-box side\n(0.02 = ~50 voxels across, works at any scene scale).\nOFF: Voxel Size is in local units.");
            }
            ImGui::SetNextItemWidth(140);
            ImGui::SliderInt("Smooth Iters", &smoothIterations, 0, 25);
            if (smoothIterations > 0) {
                ImGui::SetNextItemWidth(140);
                ImGui::SliderFloat("Smooth Factor", &smoothFactor, 0.05f, 1.0f);
            }
            ImGui::Checkbox("Smooth Shading", &smoothShading);
            ImGui::Checkbox("Transfer Material", &transferMaterial);
            if (lastTrisOut >= 0) {
                ImGui::TextDisabled("Last eval: %dx%dx%d grid, %d tris out",
                                    lastGridX, lastGridY, lastGridZ, lastTrisOut);
            }
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.75f, 0.75f, 0.55f, 1.0f));
            ImGui::TextWrapped("Rebuilds topology from scratch: UVs and mask attributes are lost - put Mask nodes AFTER Remesh. Best on closed surfaces.");
            ImGui::PopStyleColor();
        }

        // Defined in MeshModifiers.cpp (marching-cubes tables live there, not in
        // this header — same declared-here/defined-there pattern as ScatterInstances).
        NodeSystem::PinValue compute(int outputIndex, NodeSystem::EvaluationContext& ctx) override;
    };

    // ============================================================================
    // GEOMETRY GRAPH V2
    // ============================================================================

    class GeometryNodeGraphV2 : public NodeSystem::GraphBase {
    public:
        // Snapshot of the object's PRISTINE mesh, captured once when this graph is first
        // created/bound (see the "Geometry Graph" window block in scene_ui.cpp). Evaluation
        // always reads from here, never from whatever is currently live in the scene —
        // otherwise each Evaluate would compound on top of the previous Evaluate's OUTPUT
        // (e.g. re-subdividing an already-subdivided mesh) instead of always starting fresh
        // from the original import/edit. The original TriangleMesh isn't mutated by any node
        // (SubdivideCC/Translate both produce new TriangleMesh instances), so this shared_ptr
        // keeps it alive even after the scene's copy gets replaced by an evaluated result.
        std::shared_ptr<TriangleMesh> originalBaseMesh;

        NodeSystem::NodeBase* addGeometryNode(NodeType type, float x = 0, float y = 0) {
            NodeSystem::NodeBase* node = nullptr;
            switch (type) {
                case NodeType::BaseMesh:      node = addNode<BaseMeshNode>(); break;
                case NodeType::SubdivideCC:   node = addNode<SubdivideCCNode>(); break;
                case NodeType::Transform:     node = addNode<TransformNode>(); break;
                case NodeType::Output:        node = addNode<OutputNode>(); break;
                case NodeType::Mirror:        node = addNode<MirrorNode>(); break;
                case NodeType::NoiseDisplace: node = addNode<NoiseDisplaceNode>(); break;
                case NodeType::Merge:         node = addNode<MergeNode>(); break;
                case NodeType::ObjectSource:  node = addNode<ObjectSourceNode>(); break;
                case NodeType::Weld:          node = addNode<WeldNode>(); break;
                case NodeType::MaskByHeight:  node = addNode<MaskByHeightNode>(); break;
                case NodeType::MaskBySlope:   node = addNode<MaskBySlopeNode>(); break;
                case NodeType::MaskNoise:     node = addNode<MaskNoiseNode>(); break;
                case NodeType::ScatterInstances: node = addNode<ScatterInstancesNode>(); break;
                case NodeType::MaskRemap:     node = addNode<MaskRemapNode>(); break;
                case NodeType::MaskMath:      node = addNode<MaskMathNode>(); break;
                case NodeType::Array:         node = addNode<ArrayNode>(); break;
                case NodeType::Extrude:       node = addNode<ExtrudeNode>(); break;
                case NodeType::Inset:         node = addNode<InsetNode>(); break;
                case NodeType::Bevel:         node = addNode<BevelNode>(); break;
                case NodeType::Remesh:        node = addNode<RemeshNode>(); break;
            }
            if (node) {
                node->x = x;
                node->y = y;
            }
            return node;
        }
    };

    // ============================================================================
    // GRAPH SERIALIZATION (project save/load)
    // ============================================================================
    // Format notes:
    //  - Nodes are stored by typeId and re-created through NodeRegistry::create() on
    //    load (the registry's first real call site — exactly the "create by string id
    //    from JSON" use case it was built for). Unknown typeIds are skipped, so a
    //    project saved by a newer build with extra node types still loads.
    //  - Links are stored as (nodeId, pin INDEX) pairs, NOT pin ids — registerNode()
    //    reassigns pin ids on load, but a node type's pin ORDER is part of its
    //    definition and stable.
    //  - originalBaseMesh (the pristine pre-graph snapshot) goes into the project's
    //    binary sidecar (same offset+size convention as PaintLayerStack::serialize).
    //    Without it, the first Evaluate after load would snapshot the scene's saved
    //    mesh — which IS the previous session's evaluated RESULT — and compound the
    //    graph onto its own output. `bin` may be null (JSON-only callers): the graph
    //    structure still round-trips, only the snapshot is skipped.
    //  - Active P/N are serialized (not base + delta stack): deltas are flattened into
    //    plain buffers on save, matching what the user last saw.

    inline void serializeGeometryGraph(const GeometryNodeGraphV2& graph, nlohmann::json& j, std::ostream* bin) {
        nlohmann::json jNodes = nlohmann::json::array();
        for (const auto& n : graph.nodes) {
            nlohmann::json jn;
            jn["type_id"] = n->getTypeId();
            jn["id"] = n->id;
            jn["x"] = n->x;
            jn["y"] = n->y;
            if (const auto* gn = dynamic_cast<const GeometryNodeBase*>(n.get())) {
                nlohmann::json params = nlohmann::json::object();
                gn->serializeParams(params);
                if (!params.empty()) jn["params"] = params;
            }
            jNodes.push_back(std::move(jn));
        }
        j["nodes"] = std::move(jNodes);

        auto pinRef = [&graph](uint32_t pinId, bool output) -> std::pair<uint32_t, int> {
            for (const auto& n : graph.nodes) {
                const auto& pins = output ? n->outputs : n->inputs;
                for (size_t i = 0; i < pins.size(); ++i) {
                    if (pins[i].id == pinId) return { n->id, static_cast<int>(i) };
                }
            }
            return { 0u, -1 };
        };
        nlohmann::json jLinks = nlohmann::json::array();
        for (const auto& l : graph.links) {
            const auto [fromNode, fromIdx] = pinRef(l.startPinId, true);
            const auto [toNode, toIdx] = pinRef(l.endPinId, false);
            if (fromIdx < 0 || toIdx < 0) continue;  // dangling link — don't persist
            nlohmann::json jl;
            jl["from_node"] = fromNode;
            jl["from_out"] = fromIdx;
            jl["to_node"] = toNode;
            jl["to_in"] = toIdx;
            jLinks.push_back(std::move(jl));
        }
        j["links"] = std::move(jLinks);

        nlohmann::json jGroups = nlohmann::json::array();
        for (const auto& g : graph.groups) {
            nlohmann::json jg;
            jg["name"] = g.name;
            jg["comment"] = g.comment;
            jg["pos"] = { g.position.x, g.position.y };
            jg["size"] = { g.size.x, g.size.y };
            jg["color"] = static_cast<uint32_t>(g.color);
            jg["collapsed"] = g.collapsed;
            jg["locked"] = g.locked;
            jg["node_ids"] = g.nodeIds;
            jGroups.push_back(std::move(jg));
        }
        if (!jGroups.empty()) j["groups"] = std::move(jGroups);

        if (bin && graph.originalBaseMesh && graph.originalBaseMesh->geometry) {
            const auto& geo = *graph.originalBaseMesh->geometry;
            const uint64_t vc = geo.get_vertex_count();
            const uint64_t ic = geo.indices.size();
            const Vec3* P = geo.get_positions();
            const Vec3* N = geo.get_normals();
            const Vec3* Po = geo.get_positions_orig();
            const Vec3* No = geo.get_normals_orig();
            const Vec2* uv = geo.get_uvs();
            const uint16_t* mat = geo.get_material_ids();
            if (vc > 0 && ic > 0) {
                const uint32_t flags = (P ? 1u : 0u) | (N ? 2u : 0u) | (Po ? 4u : 0u)
                                     | (No ? 8u : 0u) | (uv ? 16u : 0u) | (mat ? 32u : 0u);
                const int64_t offset = static_cast<int64_t>(bin->tellp());
                if (P)   bin->write(reinterpret_cast<const char*>(P),   static_cast<std::streamsize>(vc * sizeof(Vec3)));
                if (N)   bin->write(reinterpret_cast<const char*>(N),   static_cast<std::streamsize>(vc * sizeof(Vec3)));
                if (Po)  bin->write(reinterpret_cast<const char*>(Po),  static_cast<std::streamsize>(vc * sizeof(Vec3)));
                if (No)  bin->write(reinterpret_cast<const char*>(No),  static_cast<std::streamsize>(vc * sizeof(Vec3)));
                if (uv)  bin->write(reinterpret_cast<const char*>(uv),  static_cast<std::streamsize>(vc * sizeof(Vec2)));
                if (mat) bin->write(reinterpret_cast<const char*>(mat), static_cast<std::streamsize>(vc * sizeof(uint16_t)));
                bin->write(reinterpret_cast<const char*>(geo.indices.data()), static_cast<std::streamsize>(ic * sizeof(uint32_t)));

                nlohmann::json jm;
                jm["offset"] = offset;
                jm["vc"] = vc;
                jm["ic"] = ic;
                jm["flags"] = flags;
                if (graph.originalBaseMesh->transform) {
                    const auto& t = *graph.originalBaseMesh->transform;
                    jm["t_pos"] = { t.position.x, t.position.y, t.position.z };
                    jm["t_rot"] = { t.rotation.x, t.rotation.y, t.rotation.z };
                    jm["t_scl"] = { t.scale.x, t.scale.y, t.scale.z };
                }
                j["base_mesh"] = std::move(jm);
            }
        }
    }

    inline std::shared_ptr<GeometryNodeGraphV2> deserializeGeometryGraph(const nlohmann::json& j, std::istream* bin) {
        auto graph = std::make_shared<GeometryNodeGraphV2>();
        std::unordered_map<uint32_t, NodeSystem::NodeBase*> byOldId;

        if (j.contains("nodes") && j["nodes"].is_array()) {
            for (const auto& jn : j["nodes"]) {
                const std::string typeId = jn.value("type_id", std::string());
                auto node = NodeSystem::NodeRegistry::instance().create(typeId);
                if (!node) continue;  // unknown/renamed node type — keep loading the rest
                NodeSystem::NodeBase* raw = graph->registerNode(std::move(node));
                raw->x = jn.value("x", 0.0f);
                raw->y = jn.value("y", 0.0f);
                if (jn.contains("params")) {
                    if (auto* gn = dynamic_cast<GeometryNodeBase*>(raw)) {
                        gn->deserializeParams(jn["params"]);
                    }
                }
                byOldId[jn.value("id", 0u)] = raw;
            }
        }

        if (j.contains("links") && j["links"].is_array()) {
            for (const auto& jl : j["links"]) {
                auto fromIt = byOldId.find(jl.value("from_node", 0u));
                auto toIt = byOldId.find(jl.value("to_node", 0u));
                if (fromIt == byOldId.end() || toIt == byOldId.end()) continue;
                const int oi = jl.value("from_out", -1);
                const int ii = jl.value("to_in", -1);
                if (oi < 0 || oi >= static_cast<int>(fromIt->second->outputs.size())) continue;
                if (ii < 0 || ii >= static_cast<int>(toIt->second->inputs.size())) continue;
                graph->addLink(fromIt->second->outputs[oi].id, toIt->second->inputs[ii].id);
            }
        }

        if (j.contains("groups") && j["groups"].is_array()) {
            for (const auto& jg : j["groups"]) {
                NodeSystem::NodeGroup grp;
                grp.id = graph->nextGroupId++;
                grp.name = jg.value("name", std::string("Group"));
                grp.comment = jg.value("comment", std::string());
                if (jg.contains("pos") && jg["pos"].is_array() && jg["pos"].size() >= 2) {
                    grp.position = ImVec2(jg["pos"][0].get<float>(), jg["pos"][1].get<float>());
                }
                if (jg.contains("size") && jg["size"].is_array() && jg["size"].size() >= 2) {
                    grp.size = ImVec2(jg["size"][0].get<float>(), jg["size"][1].get<float>());
                }
                grp.color = static_cast<ImU32>(jg.value("color", static_cast<uint32_t>(IM_COL32(80, 80, 100, 100))));
                grp.collapsed = jg.value("collapsed", false);
                grp.locked = jg.value("locked", false);
                if (jg.contains("node_ids") && jg["node_ids"].is_array()) {
                    for (const auto& oldId : jg["node_ids"]) {
                        auto it = byOldId.find(oldId.get<uint32_t>());
                        if (it != byOldId.end()) grp.nodeIds.push_back(it->second->id);
                    }
                }
                graph->groups.push_back(std::move(grp));
            }
        }

        if (bin && j.contains("base_mesh")) {
            const auto& jm = j["base_mesh"];
            const int64_t offset = jm.value("offset", int64_t(-1));
            const uint64_t vc = jm.value("vc", uint64_t(0));
            const uint64_t ic = jm.value("ic", uint64_t(0));
            const uint32_t flags = jm.value("flags", 0u);
            if (offset >= 0 && vc > 0 && ic > 0) {
                auto mesh = std::make_shared<TriangleMesh>();
                mesh->geometry = std::make_shared<DNA::GeometryDetail>();
                auto& geo = *mesh->geometry;
                if (flags & 1u)  geo.add_attribute<Vec3>("P");
                if (flags & 2u)  geo.add_attribute<Vec3>("N");
                if (flags & 4u)  geo.add_attribute<Vec3>("P_orig");
                if (flags & 8u)  geo.add_attribute<Vec3>("N_orig");
                if (flags & 16u) geo.add_attribute<Vec2>("uv");
                if (flags & 32u) geo.add_attribute<uint16_t>("materialID");
                geo.resize_vertices(vc);

                bin->clear();
                bin->seekg(offset);
                bool ok = true;
                auto readInto = [&](void* dst, size_t bytes) {
                    if (!ok || !dst) { ok = false; return; }
                    bin->read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(bytes));
                    if (static_cast<size_t>(bin->gcount()) != bytes) ok = false;
                };
                if (flags & 1u)  readInto(geo.get_positions_mut(), vc * sizeof(Vec3));
                if (flags & 2u)  readInto(geo.get_normals_mut(), vc * sizeof(Vec3));
                if (flags & 4u)  readInto(geo.get_attribute_data_mut<Vec3>("P_orig"), vc * sizeof(Vec3));
                if (flags & 8u)  readInto(geo.get_attribute_data_mut<Vec3>("N_orig"), vc * sizeof(Vec3));
                if (flags & 16u) readInto(geo.get_uvs_mut(), vc * sizeof(Vec2));
                if (flags & 32u) readInto(geo.get_attribute_data_mut<uint16_t>("materialID"), vc * sizeof(uint16_t));
                if (ok) {
                    geo.indices.resize(ic);
                    readInto(geo.indices.data(), ic * sizeof(uint32_t));
                }

                if (ok) {
                    mesh->transform = std::make_shared<Transform>();
                    auto readVec3 = [&jm](const char* key, Vec3 def) -> Vec3 {
                        if (!jm.contains(key) || !jm[key].is_array() || jm[key].size() < 3) return def;
                        return Vec3(jm[key][0].get<float>(), jm[key][1].get<float>(), jm[key][2].get<float>());
                    };
                    mesh->transform->position = readVec3("t_pos", Vec3(0.0f, 0.0f, 0.0f));
                    mesh->transform->rotation = readVec3("t_rot", Vec3(0.0f, 0.0f, 0.0f));
                    mesh->transform->scale = readVec3("t_scl", Vec3(1.0f, 1.0f, 1.0f));
                    mesh->transform->updateMatrix();
                    graph->originalBaseMesh = mesh;
                }
                // !ok → snapshot dropped; graph still loads, first Evaluate re-snapshots
                // (with the known compounding caveat — better than failing the whole load).
            }
        }

        return graph->nodes.empty() ? nullptr : graph;
    }

} // namespace GeometryNodesV2
