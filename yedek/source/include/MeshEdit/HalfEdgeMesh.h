/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          MeshEdit/HalfEdgeMesh.h
* Author:        Kemal Demirtas
* Date:          June 2026
* =========================================================================
*
* Half-edge mesh core for edit mode.
*
* Index-based (no pointers) so the structure is trivially copyable,
* serialisable and cache-friendly. Boundary is represented with EXPLICIT
* boundary half-edges (face == kHEInvalid) linked into closed boundary
* loops, so every half-edge always has a valid twin and vertex/face
* circulation never needs special cases.
*
* Non-manifold input does not abort the build: edges used by more than two
* face sides (or two sides with the same winding) are flagged in the build
* result, the extra sides are left unpaired and become boundary. The
* resulting structure always satisfies validate(); "manifold" is a quality
* flag, not a structural requirement.
*
* Depends only on Vec3 and the standard library — no scene/UI types.
*/
#pragma once

#include "Vec3.h"

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace MeshEdit {

using HEIndex = int32_t;
constexpr HEIndex kHEInvalid = -1;

// Removal ops (collapse/dissolve) tombstone elements instead of erasing so
// indices stay stable mid-operation; call compact() to drop the dead slots.
struct HEVertex {
    Vec3 position;
    HEIndex half_edge = kHEInvalid; // any OUTGOING half-edge (kHEInvalid = isolated)
    bool removed = false;
};

struct HEHalfEdge {
    HEIndex origin = kHEInvalid; // vertex this half-edge starts at
    HEIndex twin = kHEInvalid;   // opposite half-edge (always valid after build)
    HEIndex next = kHEInvalid;   // next around the face / boundary loop
    HEIndex prev = kHEInvalid;   // previous around the face / boundary loop
    HEIndex face = kHEInvalid;   // kHEInvalid => boundary half-edge
    HEIndex edge = kHEInvalid;   // parent undirected edge
    bool removed = false;
};

struct HEEdge {
    HEIndex half_edge = kHEInvalid; // one of the two halves
    bool removed = false;
};

struct HEFace {
    HEIndex half_edge = kHEInvalid; // any half-edge of the face cycle
    bool removed = false;
};

struct HalfEdgeBuildResult {
    bool ok = false;
    bool manifold = true;            // false if any edge had >2 sides or same-winding sides
    size_t skipped_polygons = 0;     // degenerate input polygons dropped (dup verts, <3 verts, bad ids)
    size_t non_manifold_edges = 0;   // undirected edges that could not be paired cleanly
    size_t non_manifold_vertices = 0;// "bowtie" vertices owning more than one disjoint fan
    size_t boundary_loops = 0;
    std::string message;             // first validate error when ok == false
};

class HalfEdgeMesh {
public:
    std::vector<HEVertex> vertices;
    std::vector<HEHalfEdge> half_edges;
    std::vector<HEEdge> edges;
    std::vector<HEFace> faces;

    void clear();

    // Build from welded vertex positions + per-face vertex id loops (CCW).
    // Polygons may be any size >= 3. Returns false only on structural
    // failure; non-manifold input still builds (see result flags).
    bool buildFromPolygons(const std::vector<Vec3>& positions,
                           const std::vector<std::vector<int>>& polygons,
                           HalfEdgeBuildResult* result = nullptr);

    // ---- basic accessors -------------------------------------------------
    HEIndex headVertex(HEIndex he) const { return half_edges[half_edges[he].twin].origin; }
    bool isBoundaryHalfEdge(HEIndex he) const { return half_edges[he].face == kHEInvalid; }
    bool isBoundaryEdge(HEIndex e) const;
    bool isBoundaryVertex(HEIndex v) const;
    int vertexValence(HEIndex v) const; // number of incident edges
    int faceVertexCount(HEIndex f) const;
    size_t liveVertexCount() const;
    size_t liveEdgeCount() const;
    size_t liveFaceCount() const;
    Vec3 faceNormal(HEIndex f) const;   // Newell, normalised (zero for degenerate)
    Vec3 faceCentroid(HEIndex f) const;

    // ---- circulators / queries (append into out, which is cleared) -------
    void collectVertexOutgoing(HEIndex v, std::vector<HEIndex>& out) const;
    void collectVertexNeighbors(HEIndex v, std::vector<HEIndex>& out) const;
    void collectVertexFaces(HEIndex v, std::vector<HEIndex>& out) const;
    void collectFaceHalfEdges(HEIndex f, std::vector<HEIndex>& out) const;
    void collectFaceVertices(HEIndex f, std::vector<HEIndex>& out) const;

    // Directed half-edge from -> to, or kHEInvalid.
    HEIndex findHalfEdge(HEIndex from, HEIndex to) const;
    // Undirected edge between v0 and v1, or kHEInvalid.
    HEIndex findEdge(HEIndex v0, HEIndex v1) const;

    // Quad edge-loop walk from edge e (both directions). Continues across
    // interior valence-4 vertices, stops at boundary / poles / when the
    // loop closes. out_edges is ordered along the loop and contains e.
    // Returns true if the loop is closed (cyclic).
    bool collectEdgeLoop(HEIndex e, std::vector<HEIndex>& out_edges) const;

    // Quad edge-RING walk (parallel edges crossing a strip of quads — the
    // path a loop cut follows). out_halves are consistently oriented (all
    // origins on the same side of the strip); out_forward marks halves from
    // the forward walk (false = backward walk, origin on the opposite side).
    // Returns true if the ring is closed (cyclic).
    bool collectEdgeRing(HEIndex e,
                         std::vector<HEIndex>& out_halves,
                         std::vector<uint8_t>& out_forward) const;

    // ---- Euler operators -------------------------------------------------
    // Insert a vertex on edge e at parameter t (0..1 from the edge's stored
    // half origin). Adjacent faces gain one vertex; boundary handled
    // uniformly. Returns the new vertex id.
    HEIndex splitEdge(HEIndex e, float t);

    // Connect two non-adjacent vertices of face f with a new edge, splitting
    // the polygon in two. Returns the new edge id, kHEInvalid on failure.
    HEIndex splitFace(HEIndex f, HEIndex va, HEIndex vb);

    // Flip an interior edge shared by two triangles. Fails (returns false)
    // on boundary edges, non-triangle faces, or if the flipped edge would
    // duplicate an existing one.
    bool flipEdge(HEIndex e);

    // Collapse edge e, merging its head into its origin at lerp parameter t.
    // Guarded by the link condition (shared one-ring neighbours must be the
    // triangle apexes of the adjacent faces) and the boundary-pinch rule.
    // Triangle faces degenerate to digons and are cleaned up; tombstones the
    // removed vertex/edge/faces. Returns the surviving vertex, kHEInvalid on
    // rejection.
    HEIndex collapseEdge(HEIndex e, float t = 0.5f);

    // Remove edge e and merge its two faces into one polygon. Fails on
    // boundary edges, bridges (same face on both sides) and face pairs that
    // share more than this one edge. Returns the surviving face id.
    HEIndex dissolveEdge(HEIndex e);

    // Extrude a single face along offset: the face keeps its id and moves to
    // n new vertices; n side quads connect it to the old ring. Original
    // outer-edge gluing is untouched. Optionally reports the side faces.
    // Returns f (the top face) or kHEInvalid.
    HEIndex extrudeFace(HEIndex f, const Vec3& offset,
                        std::vector<HEIndex>* out_side_faces = nullptr);

    // Inset: same topology as extrudeFace (ring of quads + centre face), with
    // the new vertices pulled toward the centroid by t (0 = no change,
    // 1 = collapse to centroid). Returns f.
    HEIndex insetFace(HEIndex f, float t,
                      std::vector<HEIndex>* out_side_faces = nullptr);

    struct LoopCutResult {
        bool closed = false;
        std::vector<HEIndex> new_vertices; // one per ring edge, in ring order
        std::vector<HEIndex> new_edges;    // the cut edges connecting them
        // (source face, new face) per splitFace — lets callers inherit
        // per-face attributes (material/UV template) across the cut.
        std::vector<std::pair<HEIndex, HEIndex>> face_splits;
    };
    // Loop cut: walk the quad edge ring through e, split every ring edge at
    // parameter t (measured consistently from one side of the strip) and
    // connect the new vertices across each quad. Returns false if the ring
    // is empty (e.g. immediately blocked by non-quad faces).
    bool loopCut(HEIndex e, float t, LoopCutResult* out = nullptr);

    // Drop tombstoned elements and re-index densely. Remap tables map old
    // index -> new index (kHEInvalid for removed slots).
    struct CompactRemap {
        std::vector<HEIndex> vertices;
        std::vector<HEIndex> half_edges;
        std::vector<HEIndex> edges;
        std::vector<HEIndex> faces;
    };
    void compact(CompactRemap* remap = nullptr);

    // ---- output ------------------------------------------------------------
    // Fan-triangulate every face. out_face_ids (optional) maps each output
    // triangle back to its source face.
    void triangulate(std::vector<std::array<HEIndex, 3>>& out_triangles,
                     std::vector<HEIndex>* out_face_ids = nullptr) const;

    // ---- integrity ---------------------------------------------------------
    // Full structural check (twin involution, next/prev inverse, face cycle
    // closure and coverage, edge pairing, boundary loop closure, vertex
    // anchors). Returns false and writes the first failure to error.
    bool validate(std::string* error = nullptr) const;

private:
    // Rotate one outgoing half-edge of a vertex to the next outgoing one.
    HEIndex rotateOutgoing(HEIndex he) const { return half_edges[half_edges[he].twin].next; }
    size_t countNonManifoldVertices() const;
    size_t countBoundaryLoops() const;
};

// Self-contained correctness suite (cube, grid plane, loop walk, Euler ops,
// non-manifold input). Appends a human-readable report; returns overall pass.
bool runHalfEdgeSelfTest(std::string& report);

} // namespace MeshEdit
