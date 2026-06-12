/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          MeshEdit/HalfEdgeMesh.cpp
* Author:        Kemal Demirtas
* Date:          June 2026
* =========================================================================
*/
#include "MeshEdit/HalfEdgeMesh.h"

#include <algorithm>
#include <cstdio>
#include <unordered_map>
#include <unordered_set>

namespace MeshEdit {

namespace {

uint64_t packUndirected(HEIndex a, HEIndex b) {
    if (b < a) {
        std::swap(a, b);
    }
    return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32ull) |
           static_cast<uint64_t>(static_cast<uint32_t>(b));
}

} // namespace

void HalfEdgeMesh::clear() {
    vertices.clear();
    half_edges.clear();
    edges.clear();
    faces.clear();
}

bool HalfEdgeMesh::buildFromPolygons(const std::vector<Vec3>& positions,
                                     const std::vector<std::vector<int>>& polygons,
                                     HalfEdgeBuildResult* result) {
    clear();
    HalfEdgeBuildResult local;
    HalfEdgeBuildResult& res = result ? *result : local;
    res = HalfEdgeBuildResult{};

    vertices.resize(positions.size());
    for (size_t i = 0; i < positions.size(); ++i) {
        vertices[i].position = positions[i];
    }

    // Pass 1: faces + interior half-edges (twin/edge unresolved).
    std::vector<int> cleaned;
    std::unordered_set<int> dupCheck;
    for (const auto& poly : polygons) {
        cleaned.clear();
        bool bad = false;
        for (int id : poly) {
            if (id < 0 || id >= static_cast<int>(vertices.size())) {
                bad = true;
                break;
            }
            // Drop consecutive duplicates (incl. wrap-around).
            if (!cleaned.empty() && cleaned.back() == id) {
                continue;
            }
            cleaned.push_back(id);
        }
        if (!bad && cleaned.size() >= 2 && cleaned.front() == cleaned.back()) {
            cleaned.pop_back();
        }
        if (!bad && cleaned.size() >= 3) {
            dupCheck.clear();
            for (int id : cleaned) {
                if (!dupCheck.insert(id).second) {
                    bad = true; // non-consecutive repeat => degenerate polygon
                    break;
                }
            }
        }
        if (bad || cleaned.size() < 3) {
            ++res.skipped_polygons;
            continue;
        }

        const HEIndex faceId = static_cast<HEIndex>(faces.size());
        const HEIndex base = static_cast<HEIndex>(half_edges.size());
        const int n = static_cast<int>(cleaned.size());
        for (int i = 0; i < n; ++i) {
            HEHalfEdge he;
            he.origin = static_cast<HEIndex>(cleaned[i]);
            he.face = faceId;
            he.next = base + static_cast<HEIndex>((i + 1) % n);
            he.prev = base + static_cast<HEIndex>((i + n - 1) % n);
            half_edges.push_back(he);
            vertices[cleaned[i]].half_edge = base + static_cast<HEIndex>(i);
        }
        HEFace face;
        face.half_edge = base;
        faces.push_back(face);
    }

    // Pass 2: pair twins per undirected edge. Opposite-direction sides pair
    // greedily; same-direction or surplus sides stay unpaired (-> boundary)
    // and flag the edge as non-manifold.
    std::unordered_map<uint64_t, std::vector<HEIndex>> sideBuckets;
    sideBuckets.reserve(half_edges.size());
    for (HEIndex he = 0; he < static_cast<HEIndex>(half_edges.size()); ++he) {
        const HEIndex headV = half_edges[half_edges[he].next].origin;
        sideBuckets[packUndirected(half_edges[he].origin, headV)].push_back(he);
    }

    std::vector<HEIndex> dirA;
    std::vector<HEIndex> dirB;
    for (const auto& bucket : sideBuckets) {
        dirA.clear();
        dirB.clear();
        for (HEIndex he : bucket.second) {
            const HEIndex headV = half_edges[half_edges[he].next].origin;
            if (half_edges[he].origin < headV) {
                dirA.push_back(he);
            } else {
                dirB.push_back(he);
            }
        }
        if (bucket.second.size() > 2 || dirA.size() > 1 || dirB.size() > 1) {
            res.manifold = false;
            ++res.non_manifold_edges;
        }
        const size_t pairCount = (std::min)(dirA.size(), dirB.size());
        for (size_t i = 0; i < pairCount; ++i) {
            const HEIndex ha = dirA[i];
            const HEIndex hb = dirB[i];
            const HEIndex e = static_cast<HEIndex>(edges.size());
            half_edges[ha].twin = hb;
            half_edges[hb].twin = ha;
            half_edges[ha].edge = e;
            half_edges[hb].edge = e;
            HEEdge edge;
            edge.half_edge = ha;
            edges.push_back(edge);
        }
    }

    // Pass 3: explicit boundary half-edges for every unpaired side.
    const HEIndex interiorCount = static_cast<HEIndex>(half_edges.size());
    for (HEIndex he = 0; he < interiorCount; ++he) {
        if (half_edges[he].twin != kHEInvalid) {
            continue;
        }
        const HEIndex b = static_cast<HEIndex>(half_edges.size());
        const HEIndex e = static_cast<HEIndex>(edges.size());
        HEHalfEdge boundary;
        boundary.origin = half_edges[half_edges[he].next].origin; // head of he
        boundary.twin = he;
        boundary.face = kHEInvalid;
        boundary.edge = e;
        half_edges.push_back(boundary);
        half_edges[he].twin = b;
        half_edges[he].edge = e;
        HEEdge edge;
        edge.half_edge = he;
        edges.push_back(edge);
    }

    // Pass 4: link boundary half-edges into loops by rotating around the
    // head vertex through the interior fan until the next boundary side.
    const size_t guard = half_edges.size() + 1;
    for (HEIndex b = interiorCount; b < static_cast<HEIndex>(half_edges.size()); ++b) {
        const HEIndex w = half_edges[half_edges[b].twin].origin; // head of b
        HEIndex h = half_edges[b].twin; // interior, outgoing from w
        bool linked = false;
        for (size_t step = 0; step < guard; ++step) {
            const HEIndex incoming = half_edges[h].prev; // interior, points to w
            const HEIndex t = half_edges[incoming].twin; // outgoing from w
            if (half_edges[t].face == kHEInvalid) {
                half_edges[b].next = t;
                half_edges[t].prev = b;
                linked = true;
                break;
            }
            h = t;
        }
        if (!linked) {
            res.ok = false;
            res.message = "boundary linking failed (corrupt fan around vertex " +
                          std::to_string(w) + ")";
            return false;
        }
    }

    res.boundary_loops = countBoundaryLoops();
    res.non_manifold_vertices = countNonManifoldVertices();
    if (res.non_manifold_vertices > 0) {
        res.manifold = false;
    }

    std::string err;
    res.ok = validate(&err);
    if (!res.ok) {
        res.message = err;
    }
    return res.ok;
}

bool HalfEdgeMesh::isBoundaryEdge(HEIndex e) const {
    const HEIndex he = edges[e].half_edge;
    return isBoundaryHalfEdge(he) || isBoundaryHalfEdge(half_edges[he].twin);
}

bool HalfEdgeMesh::isBoundaryVertex(HEIndex v) const {
    const HEIndex start = vertices[v].half_edge;
    if (start == kHEInvalid) {
        return false;
    }
    HEIndex he = start;
    const size_t guard = half_edges.size() + 1;
    for (size_t step = 0; step < guard; ++step) {
        if (isBoundaryHalfEdge(he) || isBoundaryHalfEdge(half_edges[he].twin)) {
            return true;
        }
        he = rotateOutgoing(he);
        if (he == start) {
            return false;
        }
    }
    return false;
}

int HalfEdgeMesh::vertexValence(HEIndex v) const {
    const HEIndex start = vertices[v].half_edge;
    if (start == kHEInvalid) {
        return 0;
    }
    int valence = 0;
    HEIndex he = start;
    const size_t guard = half_edges.size() + 1;
    for (size_t step = 0; step < guard; ++step) {
        ++valence;
        he = rotateOutgoing(he);
        if (he == start) {
            break;
        }
    }
    return valence;
}

int HalfEdgeMesh::faceVertexCount(HEIndex f) const {
    const HEIndex start = faces[f].half_edge;
    int count = 0;
    HEIndex he = start;
    const size_t guard = half_edges.size() + 1;
    for (size_t step = 0; step < guard; ++step) {
        ++count;
        he = half_edges[he].next;
        if (he == start) {
            break;
        }
    }
    return count;
}

void HalfEdgeMesh::collectVertexOutgoing(HEIndex v, std::vector<HEIndex>& out) const {
    out.clear();
    const HEIndex start = vertices[v].half_edge;
    if (start == kHEInvalid) {
        return;
    }
    HEIndex he = start;
    const size_t guard = half_edges.size() + 1;
    for (size_t step = 0; step < guard; ++step) {
        out.push_back(he);
        he = rotateOutgoing(he);
        if (he == start) {
            break;
        }
    }
}

void HalfEdgeMesh::collectVertexNeighbors(HEIndex v, std::vector<HEIndex>& out) const {
    collectVertexOutgoing(v, out);
    for (HEIndex& he : out) {
        he = headVertex(he);
    }
}

void HalfEdgeMesh::collectVertexFaces(HEIndex v, std::vector<HEIndex>& out) const {
    collectVertexOutgoing(v, out);
    size_t kept = 0;
    for (size_t i = 0; i < out.size(); ++i) {
        const HEIndex f = half_edges[out[i]].face;
        if (f != kHEInvalid) {
            out[kept++] = f;
        }
    }
    out.resize(kept);
}

void HalfEdgeMesh::collectFaceHalfEdges(HEIndex f, std::vector<HEIndex>& out) const {
    out.clear();
    const HEIndex start = faces[f].half_edge;
    HEIndex he = start;
    const size_t guard = half_edges.size() + 1;
    for (size_t step = 0; step < guard; ++step) {
        out.push_back(he);
        he = half_edges[he].next;
        if (he == start) {
            break;
        }
    }
}

void HalfEdgeMesh::collectFaceVertices(HEIndex f, std::vector<HEIndex>& out) const {
    collectFaceHalfEdges(f, out);
    for (HEIndex& he : out) {
        he = half_edges[he].origin;
    }
}

HEIndex HalfEdgeMesh::findHalfEdge(HEIndex from, HEIndex to) const {
    const HEIndex start = vertices[from].half_edge;
    if (start == kHEInvalid) {
        return kHEInvalid;
    }
    HEIndex he = start;
    const size_t guard = half_edges.size() + 1;
    for (size_t step = 0; step < guard; ++step) {
        if (headVertex(he) == to) {
            return he;
        }
        he = rotateOutgoing(he);
        if (he == start) {
            break;
        }
    }
    return kHEInvalid;
}

HEIndex HalfEdgeMesh::findEdge(HEIndex v0, HEIndex v1) const {
    const HEIndex he = findHalfEdge(v0, v1);
    return (he != kHEInvalid) ? half_edges[he].edge : kHEInvalid;
}

bool HalfEdgeMesh::collectEdgeLoop(HEIndex e, std::vector<HEIndex>& out_edges) const {
    out_edges.clear();
    out_edges.push_back(e);

    // Walk one direction; returns true when the loop closed on itself.
    auto walk = [&](HEIndex startHe, std::vector<HEIndex>& acc) -> bool {
        HEIndex he = startHe;
        const size_t guard = edges.size() + 1;
        for (size_t step = 0; step < guard; ++step) {
            const HEIndex w = headVertex(he);
            if (isBoundaryVertex(w) || vertexValence(w) != 4) {
                return false;
            }
            // Opposite outgoing half-edge in the valence-4 fan: rotate twice
            // from the back-pointing half (twin of he).
            HEIndex cont = rotateOutgoing(rotateOutgoing(half_edges[he].twin));
            const HEIndex contEdge = half_edges[cont].edge;
            if (contEdge == e) {
                return true; // closed loop
            }
            acc.push_back(contEdge);
            he = cont;
        }
        return false;
    };

    const HEIndex h0 = edges[e].half_edge;
    if (walk(h0, out_edges)) {
        return true;
    }
    std::vector<HEIndex> backward;
    walk(half_edges[h0].twin, backward);
    std::reverse(backward.begin(), backward.end());
    out_edges.insert(out_edges.begin(), backward.begin(), backward.end());
    return false;
}

HEIndex HalfEdgeMesh::splitEdge(HEIndex e, float t) {
    const HEIndex h = edges[e].half_edge;  // a -> b
    const HEIndex ht = half_edges[h].twin; // b -> a
    const HEIndex a = half_edges[h].origin;
    const HEIndex b = half_edges[ht].origin;

    const HEIndex w = static_cast<HEIndex>(vertices.size());
    HEVertex vert;
    vert.position = vertices[a].position * (1.0f - t) + vertices[b].position * t;
    vertices.push_back(vert);

    const HEIndex hn = half_edges[h].next;
    const HEIndex htn = half_edges[ht].next;

    const HEIndex h2 = static_cast<HEIndex>(half_edges.size());      // w -> b, face of h
    const HEIndex ht2 = static_cast<HEIndex>(half_edges.size()) + 1; // w -> a, face of ht
    const HEIndex e2 = static_cast<HEIndex>(edges.size());           // (w, b)

    HEHalfEdge newH2;
    newH2.origin = w;
    newH2.twin = ht;
    newH2.next = hn;
    newH2.prev = h;
    newH2.face = half_edges[h].face;
    newH2.edge = e2;
    half_edges.push_back(newH2);

    HEHalfEdge newHt2;
    newHt2.origin = w;
    newHt2.twin = h;
    newHt2.next = htn;
    newHt2.prev = ht;
    newHt2.face = half_edges[ht].face;
    newHt2.edge = e; // (a, w) keeps the original edge id
    half_edges.push_back(newHt2);

    half_edges[h].next = h2;
    half_edges[hn].prev = h2;
    half_edges[ht].next = ht2;
    half_edges[htn].prev = ht2;
    half_edges[h].twin = ht2;
    half_edges[ht].twin = h2;
    half_edges[ht].edge = e2;

    edges[e].half_edge = h;
    HEEdge newEdge;
    newEdge.half_edge = h2;
    edges.push_back(newEdge);

    vertices[w].half_edge = h2;
    return w;
}

HEIndex HalfEdgeMesh::splitFace(HEIndex f, HEIndex va, HEIndex vb) {
    if (va == vb) {
        return kHEInvalid;
    }
    HEIndex ha = kHEInvalid;
    HEIndex hb = kHEInvalid;
    {
        const HEIndex start = faces[f].half_edge;
        HEIndex he = start;
        const size_t guard = half_edges.size() + 1;
        for (size_t step = 0; step < guard; ++step) {
            if (half_edges[he].origin == va) ha = he;
            if (half_edges[he].origin == vb) hb = he;
            he = half_edges[he].next;
            if (he == start) {
                break;
            }
        }
    }
    if (ha == kHEInvalid || hb == kHEInvalid) {
        return kHEInvalid;
    }
    // Adjacent corners would produce a two-sided face.
    if (half_edges[ha].next == hb || half_edges[hb].next == ha) {
        return kHEInvalid;
    }

    const HEIndex pa = half_edges[ha].prev;
    const HEIndex pb = half_edges[hb].prev;

    const HEIndex d1 = static_cast<HEIndex>(half_edges.size());     // va -> vb
    const HEIndex d2 = static_cast<HEIndex>(half_edges.size()) + 1; // vb -> va
    const HEIndex e = static_cast<HEIndex>(edges.size());
    const HEIndex g = static_cast<HEIndex>(faces.size());

    HEHalfEdge newD1;
    newD1.origin = va;
    newD1.twin = d2;
    newD1.next = hb;
    newD1.prev = pa;
    newD1.face = g;
    newD1.edge = e;
    half_edges.push_back(newD1);

    HEHalfEdge newD2;
    newD2.origin = vb;
    newD2.twin = d1;
    newD2.next = ha;
    newD2.prev = pb;
    newD2.face = f;
    newD2.edge = e;
    half_edges.push_back(newD2);

    half_edges[pa].next = d1;
    half_edges[hb].prev = d1;
    half_edges[pb].next = d2;
    half_edges[ha].prev = d2;

    HEEdge newEdge;
    newEdge.half_edge = d1;
    edges.push_back(newEdge);

    // Cycle [ha .. pb, d2] keeps face f; cycle [hb .. pa, d1] becomes face g.
    faces[f].half_edge = ha;
    HEFace newFace;
    newFace.half_edge = hb;
    faces.push_back(newFace);
    {
        HEIndex he = hb;
        const size_t guard = half_edges.size() + 1;
        for (size_t step = 0; step < guard; ++step) {
            half_edges[he].face = g;
            he = half_edges[he].next;
            if (he == hb) {
                break;
            }
        }
    }
    return e;
}

bool HalfEdgeMesh::flipEdge(HEIndex e) {
    const HEIndex h = edges[e].half_edge;
    const HEIndex ht = half_edges[h].twin;
    const HEIndex f = half_edges[h].face;
    const HEIndex g = half_edges[ht].face;
    if (f == kHEInvalid || g == kHEInvalid) {
        return false;
    }
    if (faceVertexCount(f) != 3 || faceVertexCount(g) != 3) {
        return false;
    }

    // F = (h: a->b, hn: b->c, hp: c->a)   G = (ht: b->a, gn: a->d, gp: d->b)
    const HEIndex hn = half_edges[h].next;
    const HEIndex hp = half_edges[h].prev;
    const HEIndex gn = half_edges[ht].next;
    const HEIndex gp = half_edges[ht].prev;
    const HEIndex a = half_edges[h].origin;
    const HEIndex b = half_edges[ht].origin;
    const HEIndex c = half_edges[hp].origin;
    const HEIndex d = half_edges[gp].origin;
    if (findEdge(c, d) != kHEInvalid) {
        return false; // flip would duplicate an existing edge
    }

    // After flip: F' = (h: d->c, hp: c->a, gn: a->d)  G' = (ht: c->d, gp: d->b, hn: b->c)
    half_edges[h].origin = d;
    half_edges[ht].origin = c;

    half_edges[h].next = hp;
    half_edges[hp].next = gn;
    half_edges[gn].next = h;
    half_edges[h].prev = gn;
    half_edges[hp].prev = h;
    half_edges[gn].prev = hp;
    half_edges[gn].face = f;

    half_edges[ht].next = gp;
    half_edges[gp].next = hn;
    half_edges[hn].next = ht;
    half_edges[ht].prev = hn;
    half_edges[gp].prev = ht;
    half_edges[hn].prev = gp;
    half_edges[hn].face = g;

    faces[f].half_edge = h;
    faces[g].half_edge = ht;

    // a / b may have anchored on the re-originated halves.
    if (vertices[a].half_edge == h) vertices[a].half_edge = gn;
    if (vertices[b].half_edge == ht) vertices[b].half_edge = hn;
    return true;
}

size_t HalfEdgeMesh::liveVertexCount() const {
    size_t n = 0;
    for (const HEVertex& v : vertices) n += v.removed ? 0 : 1;
    return n;
}

size_t HalfEdgeMesh::liveEdgeCount() const {
    size_t n = 0;
    for (const HEEdge& e : edges) n += e.removed ? 0 : 1;
    return n;
}

size_t HalfEdgeMesh::liveFaceCount() const {
    size_t n = 0;
    for (const HEFace& f : faces) n += f.removed ? 0 : 1;
    return n;
}

Vec3 HalfEdgeMesh::faceNormal(HEIndex f) const {
    // Newell's method — robust for non-planar / concave polygons.
    Vec3 n(0.0f, 0.0f, 0.0f);
    const HEIndex start = faces[f].half_edge;
    HEIndex he = start;
    const size_t guard = half_edges.size() + 1;
    for (size_t step = 0; step < guard; ++step) {
        const Vec3& p = vertices[half_edges[he].origin].position;
        const Vec3& q = vertices[headVertex(he)].position;
        n.x += (p.y - q.y) * (p.z + q.z);
        n.y += (p.z - q.z) * (p.x + q.x);
        n.z += (p.x - q.x) * (p.y + q.y);
        he = half_edges[he].next;
        if (he == start) {
            break;
        }
    }
    const float len = n.length();
    return (len > 1e-12f) ? n / len : Vec3(0.0f, 0.0f, 0.0f);
}

Vec3 HalfEdgeMesh::faceCentroid(HEIndex f) const {
    Vec3 c(0.0f, 0.0f, 0.0f);
    int count = 0;
    const HEIndex start = faces[f].half_edge;
    HEIndex he = start;
    const size_t guard = half_edges.size() + 1;
    for (size_t step = 0; step < guard; ++step) {
        c += vertices[half_edges[he].origin].position;
        ++count;
        he = half_edges[he].next;
        if (he == start) {
            break;
        }
    }
    return (count > 0) ? c / static_cast<float>(count) : c;
}

HEIndex HalfEdgeMesh::collapseEdge(HEIndex e, float t) {
    const HEIndex h = edges[e].half_edge;  // a -> b
    const HEIndex ht = half_edges[h].twin; // b -> a
    const HEIndex a = half_edges[h].origin;
    const HEIndex b = half_edges[ht].origin;
    if (a == b) {
        return kHEInvalid;
    }
    const HEIndex f = half_edges[h].face;
    const HEIndex g = half_edges[ht].face;
    if (f != kHEInvalid && f == g) {
        return kHEInvalid; // bridge edge inside one face
    }
    // Boundary pinch: collapsing an interior edge between two boundary
    // vertices would fuse two boundary loops through one vertex.
    if (!isBoundaryEdge(e) && isBoundaryVertex(a) && isBoundaryVertex(b)) {
        return kHEInvalid;
    }
    const int nf = (f != kHEInvalid) ? faceVertexCount(f) : 0;
    const int ng = (g != kHEInvalid) ? faceVertexCount(g) : 0;
    const HEIndex apexF = (nf == 3) ? half_edges[half_edges[h].prev].origin : kHEInvalid;
    const HEIndex apexG = (ng == 3) ? half_edges[half_edges[ht].prev].origin : kHEInvalid;
    if (apexF != kHEInvalid && apexF == apexG) {
        return kHEInvalid; // two triangles glued along two edges
    }
    // Degenerate triangle whose two non-collapse sides are the same edge.
    if (nf == 3 && half_edges[half_edges[h].next].twin == half_edges[h].prev) {
        return kHEInvalid;
    }
    if (ng == 3 && half_edges[half_edges[ht].next].twin == half_edges[ht].prev) {
        return kHEInvalid;
    }
    // Link condition: shared one-ring neighbours of a and b must be exactly
    // the triangle apexes of the adjacent faces, else collapse pinches the
    // surface into a non-manifold configuration.
    {
        std::vector<HEIndex> na;
        std::vector<HEIndex> nb;
        collectVertexNeighbors(a, na);
        collectVertexNeighbors(b, nb);
        std::unordered_set<HEIndex> nbSet(nb.begin(), nb.end());
        for (HEIndex v : na) {
            if (!nbSet.count(v)) {
                continue;
            }
            if (v != apexF && v != apexG) {
                return kHEInvalid;
            }
        }
    }

    vertices[a].position = vertices[a].position * (1.0f - t) + vertices[b].position * t;

    // Re-origin every half-edge leaving b. Linear scan instead of circulation
    // so bowtie (non-manifold) vertices cannot leave stale origins behind.
    for (HEIndex i = 0; i < static_cast<HEIndex>(half_edges.size()); ++i) {
        if (!half_edges[i].removed && half_edges[i].origin == b) {
            half_edges[i].origin = a;
        }
    }

    const HEIndex hn = half_edges[h].next;
    const HEIndex hp = half_edges[h].prev;
    const HEIndex gn = half_edges[ht].next;

    // Splice h and ht out of their cycles (face or boundary loop alike).
    // Sequential splices stay correct even when h and ht are adjacent in the
    // same boundary loop.
    half_edges[hp].next = hn;
    half_edges[hn].prev = hp;
    half_edges[half_edges[ht].prev].next = half_edges[ht].next;
    half_edges[half_edges[ht].next].prev = half_edges[ht].prev;
    if (f != kHEInvalid) {
        faces[f].half_edge = hn;
    }
    if (g != kHEInvalid) {
        faces[g].half_edge = gn;
    }

    half_edges[h].removed = true;
    half_edges[ht].removed = true;
    edges[e].removed = true;
    vertices[b].removed = true;
    vertices[a].half_edge = (hn != ht) ? hn : gn; // hn origin was b -> now a

    // A triangle face shrinks to a digon: remove it and weld its two edges
    // into one by twinning the outer halves.
    auto cleanupDigon = [&](HEIndex face) {
        if (face == kHEInvalid) {
            return;
        }
        const HEIndex x = faces[face].half_edge;
        const HEIndex y = half_edges[x].next;
        if (half_edges[y].next != x || y == x) {
            return; // not a digon
        }
        const HEIndex tx = half_edges[x].twin;
        const HEIndex ty = half_edges[y].twin;
        half_edges[tx].twin = ty;
        half_edges[ty].twin = tx;
        const HEIndex keepEdge = half_edges[tx].edge;
        const HEIndex dropEdge = half_edges[ty].edge;
        half_edges[ty].edge = keepEdge;
        edges[keepEdge].half_edge = tx;
        edges[dropEdge].removed = true;
        half_edges[x].removed = true;
        half_edges[y].removed = true;
        faces[face].removed = true;
        // ty leaves head(y) = origin(x); tx leaves head(x) = origin(y).
        if (vertices[half_edges[x].origin].half_edge == x) {
            vertices[half_edges[x].origin].half_edge = ty;
        }
        if (vertices[half_edges[y].origin].half_edge == y) {
            vertices[half_edges[y].origin].half_edge = tx;
        }
    };
    if (nf == 3) {
        cleanupDigon(f);
    }
    if (ng == 3) {
        cleanupDigon(g);
    }

    // Safety net: anchors may have landed on a half-edge removed above
    // (digon cleanup, wire-edge collapse). Rare path -> linear rescue scan.
    auto rescueAnchor = [&](HEIndex v) {
        const HEIndex anchor = vertices[v].half_edge;
        if (anchor == kHEInvalid || !half_edges[anchor].removed) {
            return;
        }
        vertices[v].half_edge = kHEInvalid;
        for (HEIndex i = 0; i < static_cast<HEIndex>(half_edges.size()); ++i) {
            if (!half_edges[i].removed && half_edges[i].origin == v) {
                vertices[v].half_edge = i;
                break;
            }
        }
    };
    rescueAnchor(a);
    if (apexF != kHEInvalid) rescueAnchor(apexF);
    if (apexG != kHEInvalid) rescueAnchor(apexG);
    return a;
}

HEIndex HalfEdgeMesh::dissolveEdge(HEIndex e) {
    const HEIndex h = edges[e].half_edge;
    const HEIndex ht = half_edges[h].twin;
    const HEIndex f = half_edges[h].face;
    const HEIndex g = half_edges[ht].face;
    if (f == kHEInvalid || g == kHEInvalid || f == g) {
        return kHEInvalid;
    }
    // Faces sharing more than this edge would merge into a pinched polygon
    // that visits the surviving shared edge from both sides.
    {
        const HEIndex start = faces[f].half_edge;
        HEIndex he = start;
        int sharedEdges = 0;
        const size_t guard = half_edges.size() + 1;
        for (size_t step = 0; step < guard; ++step) {
            if (half_edges[half_edges[he].twin].face == g) {
                ++sharedEdges;
            }
            he = half_edges[he].next;
            if (he == start) {
                break;
            }
        }
        if (sharedEdges != 1) {
            return kHEInvalid;
        }
    }

    const HEIndex hp = half_edges[h].prev;
    const HEIndex hn = half_edges[h].next;
    const HEIndex gp = half_edges[ht].prev;
    const HEIndex gn = half_edges[ht].next;

    half_edges[hp].next = gn;
    half_edges[gn].prev = hp;
    half_edges[gp].next = hn;
    half_edges[hn].prev = gp;
    faces[f].half_edge = hn;

    // Absorb g's cycle into f.
    {
        HEIndex he = hn;
        const size_t guard = half_edges.size() + 1;
        for (size_t step = 0; step < guard; ++step) {
            half_edges[he].face = f;
            he = half_edges[he].next;
            if (he == hn) {
                break;
            }
        }
    }

    const HEIndex a = half_edges[h].origin;
    const HEIndex b = half_edges[ht].origin;
    if (vertices[a].half_edge == h) {
        vertices[a].half_edge = half_edges[hp].twin; // head(hp) == a
    }
    if (vertices[b].half_edge == ht) {
        vertices[b].half_edge = half_edges[gp].twin; // head(gp) == b
    }

    half_edges[h].removed = true;
    half_edges[ht].removed = true;
    edges[e].removed = true;
    faces[g].removed = true;
    return f;
}

HEIndex HalfEdgeMesh::extrudeFace(HEIndex f, const Vec3& offset,
                                  std::vector<HEIndex>* out_side_faces) {
    std::vector<HEIndex> ring; // original halves h_i: v_i -> v_{i+1}
    collectFaceHalfEdges(f, ring);
    const int n = static_cast<int>(ring.size());
    if (n < 3) {
        return kHEInvalid;
    }
    if (out_side_faces) {
        out_side_faces->clear();
    }

    const HEIndex vBase = static_cast<HEIndex>(vertices.size());   // w_i
    const HEIndex heBase = static_cast<HEIndex>(half_edges.size());
    const HEIndex eBase = static_cast<HEIndex>(edges.size());
    const HEIndex fBase = static_cast<HEIndex>(faces.size());
    // Half-edge layout per i: top_i = heBase+4i, tts_i = +1 (twin of top, in
    // side quad), a_i = +2 (v_i -> w_i), b_i = +3 (w_i -> v_i).
    auto topAt = [&](int i) { return heBase + 4 * ((i + n) % n); };
    auto ttsAt = [&](int i) { return heBase + 4 * ((i + n) % n) + 1; };
    auto upAt = [&](int i) { return heBase + 4 * ((i + n) % n) + 2; };
    auto downAt = [&](int i) { return heBase + 4 * ((i + n) % n) + 3; };

    vertices.reserve(vertices.size() + n);
    half_edges.reserve(half_edges.size() + 4 * n);
    edges.reserve(edges.size() + 2 * n);
    faces.reserve(faces.size() + n);

    for (int i = 0; i < n; ++i) {
        const HEIndex vi = half_edges[ring[i]].origin;
        HEVertex w;
        w.position = vertices[vi].position + offset;
        w.half_edge = topAt(i);
        vertices.push_back(w);
    }
    for (int i = 0; i < n; ++i) {
        const HEIndex hi = ring[i];
        const HEIndex vi = half_edges[hi].origin;
        const HEIndex quad = fBase + i;
        const HEIndex topEdge = eBase + 2 * i;      // w_i  - w_{i+1}
        const HEIndex upEdge = eBase + 2 * i + 1;   // v_i  - w_i

        HEHalfEdge top;        // w_i -> w_{i+1}, stays in f
        top.origin = vBase + i;
        top.twin = ttsAt(i);
        top.next = topAt(i + 1);
        top.prev = topAt(i - 1);
        top.face = f;
        top.edge = topEdge;
        half_edges.push_back(top);

        HEHalfEdge tts;        // w_{i+1} -> w_i, side quad i
        tts.origin = vBase + ((i + 1) % n);
        tts.twin = topAt(i);
        tts.next = downAt(i);
        tts.prev = upAt(i + 1);
        tts.face = quad;
        tts.edge = topEdge;
        half_edges.push_back(tts);

        HEHalfEdge up;         // v_i -> w_i, side quad i-1
        up.origin = vi;
        up.twin = downAt(i);
        up.next = ttsAt(i - 1);
        up.prev = ring[(i + n - 1) % n];
        up.face = fBase + ((i + n - 1) % n);
        up.edge = upEdge;
        half_edges.push_back(up);

        HEHalfEdge down;       // w_i -> v_i, side quad i
        down.origin = vBase + i;
        down.twin = upAt(i);
        down.next = hi;
        down.prev = ttsAt(i);
        down.face = quad;
        down.edge = upEdge;
        half_edges.push_back(down);

        HEEdge te;
        te.half_edge = topAt(i);
        edges.push_back(te);
        HEEdge ue;
        ue.half_edge = upAt(i);
        edges.push_back(ue);

        HEFace q;
        q.half_edge = hi;
        faces.push_back(q);
        if (out_side_faces) {
            out_side_faces->push_back(quad);
        }
    }
    // Original boundary halves of f move into their side quads; their
    // origin/twin/edge stay untouched so outer gluing is preserved.
    for (int i = 0; i < n; ++i) {
        const HEIndex hi = ring[i];
        half_edges[hi].face = fBase + i;
        half_edges[hi].next = upAt(i + 1);
        half_edges[hi].prev = downAt(i);
    }
    faces[f].half_edge = topAt(0);
    return f;
}

HEIndex HalfEdgeMesh::insetFace(HEIndex f, float t,
                                std::vector<HEIndex>* out_side_faces) {
    const Vec3 centroid = faceCentroid(f);
    if (extrudeFace(f, Vec3(0.0f, 0.0f, 0.0f), out_side_faces) == kHEInvalid) {
        return kHEInvalid;
    }
    std::vector<HEIndex> topVerts;
    collectFaceVertices(f, topVerts);
    for (HEIndex v : topVerts) {
        vertices[v].position = vertices[v].position * (1.0f - t) + centroid * t;
    }
    return f;
}

bool HalfEdgeMesh::collectEdgeRing(HEIndex e,
                                   std::vector<HEIndex>& out_halves,
                                   std::vector<uint8_t>& out_forward) const {
    out_halves.clear();
    out_forward.clear();

    // Step across a quad: opposite edge of face(h), oriented to keep the
    // origin on the same side of the strip.
    auto step = [&](HEIndex h) -> HEIndex {
        const HEIndex f = half_edges[h].face;
        if (f == kHEInvalid || faceVertexCount(f) != 4) {
            return kHEInvalid;
        }
        return half_edges[half_edges[half_edges[h].next].next].twin;
    };

    const HEIndex h0 = edges[e].half_edge;
    out_halves.push_back(h0);
    out_forward.push_back(1);

    HEIndex h = h0;
    const size_t guard = edges.size() + 1;
    for (size_t i = 0; i < guard; ++i) {
        h = step(h);
        if (h == kHEInvalid) {
            break;
        }
        if (h == h0) {
            return true; // closed ring
        }
        out_halves.push_back(h);
        out_forward.push_back(1);
    }

    // Open: extend backward from the twin side (origins on the other side).
    std::vector<HEIndex> back;
    h = half_edges[h0].twin;
    for (size_t i = 0; i < guard; ++i) {
        h = step(h);
        if (h == kHEInvalid || h == half_edges[h0].twin) {
            break;
        }
        back.push_back(h);
    }
    out_halves.insert(out_halves.begin(), back.rbegin(), back.rend());
    out_forward.insert(out_forward.begin(), back.size(), 0);
    return false;
}

bool HalfEdgeMesh::loopCut(HEIndex e, float t, LoopCutResult* out) {
    std::vector<HEIndex> ringHalves;
    std::vector<uint8_t> forward;
    const bool closed = collectEdgeRing(e, ringHalves, forward);
    const size_t k = ringHalves.size();
    if (k < 2) {
        return false; // nothing to cut across (also: no mutation on failure)
    }

    LoopCutResult local;
    LoopCutResult& res = out ? *out : local;
    res = LoopCutResult{};
    res.closed = closed;

    // The faces between consecutive ring halves, captured BEFORE splitting
    // (splitEdge keeps them valid; splitFace later subdivides them). Forward
    // halves cross their own face; backward halves cross their twin's face.
    std::vector<HEIndex> stripFaces(k, kHEInvalid);
    for (size_t i = 0; i < k; ++i) {
        const bool last = (i + 1 == k);
        if (last && !closed) {
            break;
        }
        const HEIndex he = ringHalves[i];
        stripFaces[i] = forward[i] ? half_edges[he].face
                                   : half_edges[half_edges[he].twin].face;
    }

    // Split every ring edge. t is measured from the forward-walk origin side;
    // backward halves and reversed edge anchors mirror the parameter.
    res.new_vertices.reserve(k);
    for (size_t i = 0; i < k; ++i) {
        const HEIndex he = ringHalves[i];
        const HEIndex eid = half_edges[he].edge;
        float param = forward[i] ? t : (1.0f - t);
        if (edges[eid].half_edge != he) {
            param = 1.0f - param;
        }
        res.new_vertices.push_back(splitEdge(eid, param));
    }

    // Connect consecutive new vertices across each strip face.
    for (size_t i = 0; i < k; ++i) {
        if (stripFaces[i] == kHEInvalid) {
            continue;
        }
        const HEIndex va = res.new_vertices[i];
        const HEIndex vb = res.new_vertices[(i + 1) % k];
        const HEIndex cut = splitFace(stripFaces[i], va, vb);
        if (cut != kHEInvalid) {
            res.new_edges.push_back(cut);
            res.face_splits.emplace_back(stripFaces[i],
                                         static_cast<HEIndex>(faces.size()) - 1);
        }
    }
    return !res.new_edges.empty();
}

void HalfEdgeMesh::compact(CompactRemap* remap) {
    CompactRemap local;
    CompactRemap& map = remap ? *remap : local;

    auto buildRemap = [](auto& elements, std::vector<HEIndex>& table) {
        table.assign(elements.size(), kHEInvalid);
        HEIndex next = 0;
        for (size_t i = 0; i < elements.size(); ++i) {
            if (!elements[i].removed) {
                table[i] = next++;
            }
        }
    };
    buildRemap(vertices, map.vertices);
    buildRemap(half_edges, map.half_edges);
    buildRemap(edges, map.edges);
    buildRemap(faces, map.faces);

    auto applyRemap = [](HEIndex idx, const std::vector<HEIndex>& table) {
        return (idx == kHEInvalid) ? kHEInvalid : table[idx];
    };

    auto compactInto = [](auto& elements, const std::vector<HEIndex>& table) {
        size_t write = 0;
        for (size_t i = 0; i < elements.size(); ++i) {
            if (table[i] != kHEInvalid) {
                elements[write++] = elements[i];
            }
        }
        elements.resize(write);
    };
    compactInto(vertices, map.vertices);
    compactInto(half_edges, map.half_edges);
    compactInto(edges, map.edges);
    compactInto(faces, map.faces);

    for (HEVertex& v : vertices) {
        v.half_edge = applyRemap(v.half_edge, map.half_edges);
    }
    for (HEHalfEdge& h : half_edges) {
        h.origin = applyRemap(h.origin, map.vertices);
        h.twin = applyRemap(h.twin, map.half_edges);
        h.next = applyRemap(h.next, map.half_edges);
        h.prev = applyRemap(h.prev, map.half_edges);
        h.face = applyRemap(h.face, map.faces);
        h.edge = applyRemap(h.edge, map.edges);
    }
    for (HEEdge& e : edges) {
        e.half_edge = applyRemap(e.half_edge, map.half_edges);
    }
    for (HEFace& f : faces) {
        f.half_edge = applyRemap(f.half_edge, map.half_edges);
    }
}

void HalfEdgeMesh::triangulate(std::vector<std::array<HEIndex, 3>>& out_triangles,
                               std::vector<HEIndex>* out_face_ids) const {
    out_triangles.clear();
    if (out_face_ids) {
        out_face_ids->clear();
    }
    std::vector<HEIndex> faceVerts;
    for (HEIndex f = 0; f < static_cast<HEIndex>(faces.size()); ++f) {
        if (faces[f].removed) {
            continue;
        }
        collectFaceVertices(f, faceVerts);
        for (size_t i = 1; i + 1 < faceVerts.size(); ++i) {
            out_triangles.push_back({ faceVerts[0], faceVerts[i], faceVerts[i + 1] });
            if (out_face_ids) {
                out_face_ids->push_back(f);
            }
        }
    }
}

bool HalfEdgeMesh::validate(std::string* error) const {
    auto fail = [&](const std::string& msg) {
        if (error) {
            *error = msg;
        }
        return false;
    };

    const HEIndex heCount = static_cast<HEIndex>(half_edges.size());
    const HEIndex vCount = static_cast<HEIndex>(vertices.size());
    const HEIndex eCount = static_cast<HEIndex>(edges.size());
    const HEIndex fCount = static_cast<HEIndex>(faces.size());

    std::vector<int> edgeUse(edges.size(), 0);

    for (HEIndex he = 0; he < heCount; ++he) {
        const HEHalfEdge& h = half_edges[he];
        if (h.removed) {
            continue;
        }
        if (h.origin < 0 || h.origin >= vCount) return fail("half-edge " + std::to_string(he) + ": origin out of range");
        if (h.twin < 0 || h.twin >= heCount) return fail("half-edge " + std::to_string(he) + ": twin out of range");
        if (h.next < 0 || h.next >= heCount) return fail("half-edge " + std::to_string(he) + ": next out of range");
        if (h.prev < 0 || h.prev >= heCount) return fail("half-edge " + std::to_string(he) + ": prev out of range");
        if (h.edge < 0 || h.edge >= eCount) return fail("half-edge " + std::to_string(he) + ": edge out of range");
        if (h.face != kHEInvalid && (h.face < 0 || h.face >= fCount)) return fail("half-edge " + std::to_string(he) + ": face out of range");
        if (vertices[h.origin].removed) return fail("half-edge " + std::to_string(he) + ": origin is removed");
        if (half_edges[h.twin].removed) return fail("half-edge " + std::to_string(he) + ": twin is removed");
        if (half_edges[h.next].removed) return fail("half-edge " + std::to_string(he) + ": next is removed");
        if (half_edges[h.prev].removed) return fail("half-edge " + std::to_string(he) + ": prev is removed");
        if (edges[h.edge].removed) return fail("half-edge " + std::to_string(he) + ": edge is removed");
        if (h.face != kHEInvalid && faces[h.face].removed) return fail("half-edge " + std::to_string(he) + ": face is removed");
        if (h.twin == he) return fail("half-edge " + std::to_string(he) + ": twin is self");
        if (half_edges[h.twin].twin != he) return fail("half-edge " + std::to_string(he) + ": twin not involutive");
        if (half_edges[h.next].prev != he) return fail("half-edge " + std::to_string(he) + ": next/prev mismatch");
        if (half_edges[h.next].face != h.face) return fail("half-edge " + std::to_string(he) + ": face differs along cycle");
        if (half_edges[h.next].origin != half_edges[h.twin].origin) return fail("half-edge " + std::to_string(he) + ": head inconsistent (next.origin != twin.origin)");
        if (half_edges[h.twin].edge != h.edge) return fail("half-edge " + std::to_string(he) + ": twin has different edge");
        ++edgeUse[h.edge];
    }

    for (HEIndex e = 0; e < eCount; ++e) {
        if (edges[e].removed) {
            if (edgeUse[e] != 0) return fail("edge " + std::to_string(e) + ": removed but referenced by live halves");
            continue;
        }
        if (edgeUse[e] != 2) return fail("edge " + std::to_string(e) + ": expected 2 halves, found " + std::to_string(edgeUse[e]));
        const HEIndex he = edges[e].half_edge;
        if (he < 0 || he >= heCount) return fail("edge " + std::to_string(e) + ": half_edge out of range");
        if (half_edges[he].removed) return fail("edge " + std::to_string(e) + ": anchor half-edge removed");
        if (half_edges[he].edge != e) return fail("edge " + std::to_string(e) + ": half_edge points to different edge");
    }

    // Face cycles: close, length >= 3, and together cover every interior half-edge.
    std::vector<uint8_t> seen(half_edges.size(), 0);
    size_t interiorVisited = 0;
    for (HEIndex f = 0; f < fCount; ++f) {
        if (faces[f].removed) {
            continue;
        }
        const HEIndex start = faces[f].half_edge;
        if (start < 0 || start >= heCount) return fail("face " + std::to_string(f) + ": half_edge out of range");
        if (half_edges[start].removed) return fail("face " + std::to_string(f) + ": anchor half-edge removed");
        if (half_edges[start].face != f) return fail("face " + std::to_string(f) + ": anchor half-edge belongs to another face");
        HEIndex he = start;
        int length = 0;
        const size_t guard = half_edges.size() + 1;
        bool closed = false;
        for (size_t step = 0; step < guard; ++step) {
            if (seen[he]) return fail("face " + std::to_string(f) + ": half-edge " + std::to_string(he) + " visited twice");
            seen[he] = 1;
            ++interiorVisited;
            ++length;
            he = half_edges[he].next;
            if (he == start) {
                closed = true;
                break;
            }
        }
        if (!closed) return fail("face " + std::to_string(f) + ": cycle does not close");
        if (length < 3) return fail("face " + std::to_string(f) + ": fewer than 3 sides");
    }

    // Boundary loops: every boundary half-edge sits in exactly one closed loop.
    for (HEIndex he = 0; he < heCount; ++he) {
        if (half_edges[he].removed || half_edges[he].face != kHEInvalid || seen[he]) {
            continue;
        }
        HEIndex cur = he;
        const size_t guard = half_edges.size() + 1;
        bool closed = false;
        for (size_t step = 0; step < guard; ++step) {
            if (seen[cur]) return fail("boundary half-edge " + std::to_string(cur) + " visited twice");
            seen[cur] = 1;
            cur = half_edges[cur].next;
            if (cur == he) {
                closed = true;
                break;
            }
        }
        if (!closed) return fail("boundary loop through half-edge " + std::to_string(he) + " does not close");
    }

    size_t totalSeen = 0;
    for (uint8_t s : seen) {
        totalSeen += s;
    }
    size_t liveHalfEdges = 0;
    for (const HEHalfEdge& h : half_edges) {
        liveHalfEdges += h.removed ? 0 : 1;
    }
    if (totalSeen != liveHalfEdges) {
        return fail("half-edge coverage mismatch: " + std::to_string(totalSeen) + " of " + std::to_string(liveHalfEdges));
    }
    (void)interiorVisited;

    // Vertex anchors + circulation closure.
    for (HEIndex v = 0; v < vCount; ++v) {
        if (vertices[v].removed) {
            continue;
        }
        const HEIndex start = vertices[v].half_edge;
        if (start == kHEInvalid) {
            continue;
        }
        if (start < 0 || start >= heCount) return fail("vertex " + std::to_string(v) + ": half_edge out of range");
        if (half_edges[start].removed) return fail("vertex " + std::to_string(v) + ": anchor half-edge removed");
        if (half_edges[start].origin != v) return fail("vertex " + std::to_string(v) + ": anchor half-edge has different origin");
        HEIndex he = start;
        const size_t guard = half_edges.size() + 1;
        bool closed = false;
        for (size_t step = 0; step < guard; ++step) {
            if (half_edges[he].origin != v) return fail("vertex " + std::to_string(v) + ": circulation reached foreign half-edge");
            he = rotateOutgoing(he);
            if (he == start) {
                closed = true;
                break;
            }
        }
        if (!closed) return fail("vertex " + std::to_string(v) + ": circulation does not close");
    }

    return true;
}

size_t HalfEdgeMesh::countNonManifoldVertices() const {
    // A vertex is non-manifold ("bowtie") when circulation from its anchor
    // does not reach every outgoing half-edge.
    std::vector<int> outgoingTotal(vertices.size(), 0);
    for (const HEHalfEdge& h : half_edges) {
        if (!h.removed && h.origin >= 0 && h.origin < static_cast<HEIndex>(vertices.size())) {
            ++outgoingTotal[h.origin];
        }
    }
    size_t count = 0;
    for (HEIndex v = 0; v < static_cast<HEIndex>(vertices.size()); ++v) {
        if (vertices[v].removed || vertices[v].half_edge == kHEInvalid) {
            continue;
        }
        if (vertexValence(v) != outgoingTotal[v]) {
            ++count;
        }
    }
    return count;
}

size_t HalfEdgeMesh::countBoundaryLoops() const {
    std::vector<uint8_t> seen(half_edges.size(), 0);
    size_t loops = 0;
    const size_t guard = half_edges.size() + 1;
    for (HEIndex he = 0; he < static_cast<HEIndex>(half_edges.size()); ++he) {
        if (half_edges[he].removed || half_edges[he].face != kHEInvalid || seen[he]) {
            continue;
        }
        ++loops;
        HEIndex cur = he;
        for (size_t step = 0; step < guard; ++step) {
            seen[cur] = 1;
            cur = half_edges[cur].next;
            if (cur == he || cur == kHEInvalid) {
                break;
            }
        }
    }
    return loops;
}

// ===========================================================================
// Self-test
// ===========================================================================

namespace {

struct SelfTest {
    std::string& report;
    bool ok = true;

    explicit SelfTest(std::string& r) : report(r) {}

    void check(bool cond, const std::string& what) {
        report += (cond ? "[PASS] " : "[FAIL] ") + what + "\n";
        if (!cond) {
            ok = false;
        }
    }
};

std::vector<Vec3> cubePositions() {
    return {
        Vec3(-1, -1, -1), Vec3(1, -1, -1), Vec3(1, 1, -1), Vec3(-1, 1, -1),
        Vec3(-1, -1, 1),  Vec3(1, -1, 1),  Vec3(1, 1, 1),  Vec3(-1, 1, 1)
    };
}

std::vector<std::vector<int>> cubeQuads() {
    return {
        { 0, 3, 2, 1 }, // -z
        { 4, 5, 6, 7 }, // +z
        { 0, 1, 5, 4 }, // -y
        { 2, 3, 7, 6 }, // +y
        { 1, 2, 6, 5 }, // +x
        { 0, 4, 7, 3 }  // -x
    };
}

// (cols+1) x (rows+1) vertex grid of quads in the XZ plane.
void gridPlane(int cols, int rows,
               std::vector<Vec3>& positions,
               std::vector<std::vector<int>>& polys) {
    positions.clear();
    polys.clear();
    for (int z = 0; z <= rows; ++z) {
        for (int x = 0; x <= cols; ++x) {
            positions.push_back(Vec3(static_cast<float>(x), 0.0f, static_cast<float>(z)));
        }
    }
    const int stride = cols + 1;
    for (int z = 0; z < rows; ++z) {
        for (int x = 0; x < cols; ++x) {
            const int v0 = z * stride + x;
            polys.push_back({ v0, v0 + 1, v0 + 1 + stride, v0 + stride });
        }
    }
}

} // namespace

bool runHalfEdgeSelfTest(std::string& report) {
    SelfTest t(report);
    std::string err;

    // --- single triangle ---------------------------------------------------
    {
        HalfEdgeMesh m;
        HalfEdgeBuildResult r;
        const bool built = m.buildFromPolygons(
            { Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0) }, { { 0, 1, 2 } }, &r);
        t.check(built && r.ok, "triangle: build");
        t.check(m.vertices.size() == 3 && m.edges.size() == 3 &&
                m.faces.size() == 1 && m.half_edges.size() == 6,
                "triangle: counts V3 E3 F1 HE6");
        t.check(r.boundary_loops == 1, "triangle: one boundary loop");
        t.check(m.isBoundaryVertex(0) && m.isBoundaryEdge(0), "triangle: boundary flags");
        t.check(r.manifold, "triangle: manifold");
    }

    // --- quad cube (closed) --------------------------------------------------
    {
        HalfEdgeMesh m;
        HalfEdgeBuildResult r;
        const bool built = m.buildFromPolygons(cubePositions(), cubeQuads(), &r);
        t.check(built && r.ok, "cube: build");
        t.check(m.vertices.size() == 8 && m.edges.size() == 12 &&
                m.faces.size() == 6 && m.half_edges.size() == 24,
                "cube: counts V8 E12 F6 HE24");
        t.check(r.boundary_loops == 0 && r.manifold, "cube: closed manifold");
        bool valence3 = true;
        for (HEIndex v = 0; v < 8; ++v) {
            valence3 = valence3 && (m.vertexValence(v) == 3);
        }
        t.check(valence3, "cube: every vertex valence 3");
        std::vector<HEIndex> faceVerts;
        m.collectFaceVertices(0, faceVerts);
        t.check(faceVerts.size() == 4, "cube: face is a quad");
        std::vector<std::array<HEIndex, 3>> tris;
        m.triangulate(tris);
        t.check(tris.size() == 12, "cube: triangulates to 12 tris");

        // splitEdge: two adjacent quads become pentagons.
        const HEIndex newV = m.splitEdge(0, 0.5f);
        t.check(newV == 8 && m.vertices.size() == 9 &&
                m.edges.size() == 13 && m.half_edges.size() == 26,
                "cube: splitEdge counts");
        t.check(m.validate(&err), "cube: validate after splitEdge (" + err + ")");
        t.check(m.vertexValence(newV) == 2, "cube: split vertex valence 2");
        int penta = 0;
        for (HEIndex f = 0; f < static_cast<HEIndex>(m.faces.size()); ++f) {
            if (m.faceVertexCount(f) == 5) {
                ++penta;
            }
        }
        t.check(penta == 2, "cube: splitEdge made 2 pentagons");

        // splitFace: cut one pentagon from the new vertex to a non-adjacent corner.
        HEIndex pentaFace = kHEInvalid;
        for (HEIndex f = 0; f < static_cast<HEIndex>(m.faces.size()); ++f) {
            if (m.faceVertexCount(f) == 5) {
                pentaFace = f;
                break;
            }
        }
        std::vector<HEIndex> pv;
        m.collectFaceVertices(pentaFace, pv);
        size_t newVAt = 0;
        for (size_t i = 0; i < pv.size(); ++i) {
            if (pv[i] == newV) {
                newVAt = i;
            }
        }
        const HEIndex across = pv[(newVAt + 2) % pv.size()];
        const HEIndex cutEdge = m.splitFace(pentaFace, newV, across);
        t.check(cutEdge != kHEInvalid && m.faces.size() == 7 && m.edges.size() == 14,
                "cube: splitFace counts");
        t.check(m.validate(&err), "cube: validate after splitFace (" + err + ")");
        t.check(m.findEdge(newV, across) == cutEdge, "cube: splitFace edge findable");
        t.check(m.splitFace(pentaFace, newV, across) == kHEInvalid,
                "cube: splitFace rejects now-adjacent pair");
    }

    // --- grid plane: boundary loop + edge-loop walk -------------------------
    {
        std::vector<Vec3> positions;
        std::vector<std::vector<int>> polys;
        gridPlane(3, 3, positions, polys);
        HalfEdgeMesh m;
        HalfEdgeBuildResult r;
        const bool built = m.buildFromPolygons(positions, polys, &r);
        t.check(built && r.ok, "grid: build");
        t.check(m.vertices.size() == 16 && m.faces.size() == 9 && m.edges.size() == 24,
                "grid: counts V16 F9 E24");
        t.check(r.boundary_loops == 1, "grid: single boundary loop");
        t.check(m.vertexValence(5) == 4 && !m.isBoundaryVertex(5),
                "grid: interior vertex valence 4");

        // Horizontal edge in the middle row: loop should span the row (3 edges, open).
        const HEIndex rowEdge = m.findEdge(5, 6);
        t.check(rowEdge != kHEInvalid, "grid: middle row edge exists");
        std::vector<HEIndex> loop;
        const bool closed = m.collectEdgeLoop(rowEdge, loop);
        t.check(!closed && loop.size() == 3, "grid: row edge loop = 3 edges, open");
    }

    // --- two triangles: flipEdge --------------------------------------------
    {
        HalfEdgeMesh m;
        HalfEdgeBuildResult r;
        // Quad (0,1,2,3) as two triangles sharing diagonal 0-2.
        const bool built = m.buildFromPolygons(
            { Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(1, 1, 0), Vec3(0, 1, 0) },
            { { 0, 1, 2 }, { 0, 2, 3 } }, &r);
        t.check(built && r.ok, "flip: build");
        const HEIndex diag = m.findEdge(0, 2);
        t.check(diag != kHEInvalid, "flip: diagonal exists");
        t.check(m.flipEdge(diag), "flip: flipEdge succeeds");
        t.check(m.validate(&err), "flip: validate after flip (" + err + ")");
        t.check(m.findEdge(1, 3) != kHEInvalid && m.findEdge(0, 2) == kHEInvalid,
                "flip: diagonal now 1-3");
        t.check(!m.flipEdge(m.findEdge(0, 1)), "flip: boundary edge rejected");
        t.check(m.flipEdge(m.findEdge(1, 3)) && m.findEdge(0, 2) != kHEInvalid,
                "flip: flip back restores 0-2");
        t.check(m.validate(&err), "flip: validate after flip back (" + err + ")");
    }

    // --- tetrahedron: every flip must hit the duplicate-edge guard ------------
    {
        HalfEdgeMesh m;
        HalfEdgeBuildResult r;
        const bool built = m.buildFromPolygons(
            { Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1) },
            { { 0, 2, 1 }, { 0, 1, 3 }, { 1, 2, 3 }, { 0, 3, 2 } }, &r);
        t.check(built && r.ok && r.boundary_loops == 0 && m.edges.size() == 6,
                "tet: closed build, 6 edges");
        bool allRejected = true;
        for (HEIndex e = 0; e < static_cast<HEIndex>(m.edges.size()); ++e) {
            allRejected = allRejected && !m.flipEdge(e);
        }
        t.check(allRejected, "tet: duplicate-edge flips rejected");
        t.check(m.validate(&err), "tet: validate untouched (" + err + ")");
    }

    // --- non-manifold: three triangles sharing one edge ----------------------
    {
        HalfEdgeMesh m;
        HalfEdgeBuildResult r;
        const bool built = m.buildFromPolygons(
            { Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1), Vec3(0, -1, 0) },
            { { 0, 1, 2 }, { 0, 3, 1 }, { 0, 1, 4 } }, &r);
        t.check(built && r.ok, "non-manifold: build still structurally valid");
        t.check(!r.manifold && r.non_manifold_edges >= 1, "non-manifold: edge flagged");
        t.check(m.validate(&err), "non-manifold: validate (" + err + ")");
    }

    // --- degenerate input ------------------------------------------------------
    {
        HalfEdgeMesh m;
        HalfEdgeBuildResult r;
        const bool built = m.buildFromPolygons(
            { Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0) },
            { { 0, 1 }, { 0, 1, 1, 2 }, { 0, 1, 2, 1 }, { 0, 1, 99 } }, &r);
        t.check(built && r.ok, "degenerate: build");
        // {0,1} too short; {0,1,1,2} cleans to a valid triangle; {0,1,2,1}
        // repeats non-consecutively; {0,1,99} has a bad id.
        t.check(r.skipped_polygons == 3 && m.faces.size() == 1,
                "degenerate: 3 skipped, 1 kept");
    }

    // --- dissolveEdge ---------------------------------------------------------
    {
        HalfEdgeMesh m;
        m.buildFromPolygons(cubePositions(), cubeQuads(), nullptr);
        const HEIndex surviving = m.dissolveEdge(0);
        t.check(surviving != kHEInvalid, "dissolve: cube edge dissolves");
        t.check(m.liveFaceCount() == 5 && m.liveEdgeCount() == 11 && m.liveVertexCount() == 8,
                "dissolve: counts V8 E11 F5");
        t.check(m.faceVertexCount(surviving) == 6, "dissolve: merged face is a hexagon");
        t.check(m.validate(&err), "dissolve: validate (" + err + ")");

        HalfEdgeMesh::CompactRemap remap;
        m.compact(&remap);
        t.check(m.vertices.size() == 8 && m.edges.size() == 11 &&
                m.faces.size() == 5 && m.half_edges.size() == 22,
                "dissolve: compact shrinks arrays");
        t.check(m.validate(&err), "dissolve: validate after compact (" + err + ")");
    }
    {
        HalfEdgeMesh m;
        m.buildFromPolygons(
            { Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(1, 1, 0), Vec3(0, 1, 0) },
            { { 0, 1, 2 }, { 0, 2, 3 } }, nullptr);
        const HEIndex diag = m.findEdge(0, 2);
        t.check(m.dissolveEdge(diag) != kHEInvalid &&
                m.liveFaceCount() == 1 && m.liveEdgeCount() == 4,
                "dissolve: tri pair merges to quad");
        t.check(m.validate(&err), "dissolve: quad validate (" + err + ")");
        t.check(m.dissolveEdge(m.findEdge(0, 1)) == kHEInvalid,
                "dissolve: boundary edge rejected");
    }

    // --- collapseEdge ----------------------------------------------------------
    {
        std::vector<Vec3> positions;
        std::vector<std::vector<int>> polys;
        gridPlane(3, 3, positions, polys);
        HalfEdgeMesh m;
        m.buildFromPolygons(positions, polys, nullptr);
        const HEIndex e = m.findEdge(5, 6); // interior horizontal edge
        const HEIndex kept = m.collapseEdge(e, 0.5f);
        t.check(kept != kHEInvalid, "collapse: grid interior edge");
        t.check(m.liveVertexCount() == 15 && m.liveEdgeCount() == 23 && m.liveFaceCount() == 9,
                "collapse: counts V15 E23 F9");
        t.check(m.validate(&err), "collapse: validate (" + err + ")");
        int triCount = 0;
        for (HEIndex f = 0; f < static_cast<HEIndex>(m.faces.size()); ++f) {
            if (!m.faces[f].removed && m.faceVertexCount(f) == 3) {
                ++triCount;
            }
        }
        t.check(triCount == 2, "collapse: adjacent quads became triangles");
        m.compact(nullptr);
        t.check(m.vertices.size() == 15 && m.validate(&err),
                "collapse: compact + validate (" + err + ")");
    }
    {
        // Hexagon triangle fan: collapsing a spoke removes two digons.
        std::vector<Vec3> positions = { Vec3(0, 0, 0) };
        std::vector<std::vector<int>> polys;
        for (int i = 0; i < 6; ++i) {
            const float ang = static_cast<float>(i) * (M_PI / 3.0f);
            positions.push_back(Vec3(std::cos(ang), std::sin(ang), 0.0f));
            polys.push_back({ 0, 1 + i, 1 + (i + 1) % 6 });
        }
        HalfEdgeMesh m;
        HalfEdgeBuildResult r;
        m.buildFromPolygons(positions, polys, &r);
        t.check(r.ok && m.liveEdgeCount() == 12 && m.liveFaceCount() == 6,
                "collapse: fan build V7 E12 F6");
        const HEIndex spoke = m.findEdge(0, 1);
        t.check(m.collapseEdge(spoke, 0.0f) != kHEInvalid, "collapse: fan spoke");
        t.check(m.liveVertexCount() == 6 && m.liveEdgeCount() == 9 && m.liveFaceCount() == 4,
                "collapse: fan counts V6 E9 F4 (digons welded)");
        t.check(m.validate(&err), "collapse: fan validate (" + err + ")");
    }
    {
        // Boundary-pinch guard: interior edge between two boundary vertices.
        HalfEdgeMesh m;
        m.buildFromPolygons(
            { Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(1, 1, 0), Vec3(0, 1, 0) },
            { { 0, 1, 2 }, { 0, 2, 3 } }, nullptr);
        t.check(m.collapseEdge(m.findEdge(0, 2)) == kHEInvalid,
                "collapse: boundary pinch rejected");
    }

    // --- extrudeFace / insetFace ------------------------------------------------
    {
        HalfEdgeMesh m;
        m.buildFromPolygons(cubePositions(), cubeQuads(), nullptr);
        const HEIndex f = 1; // +z quad (4,5,6,7)
        const Vec3 before = m.faceCentroid(f);
        std::vector<HEIndex> sideFaces;
        t.check(m.extrudeFace(f, Vec3(0, 0, 1), &sideFaces) == f,
                "extrude: returns top face");
        t.check(m.liveVertexCount() == 12 && m.liveEdgeCount() == 20 && m.liveFaceCount() == 10,
                "extrude: counts V12 E20 F10");
        t.check(sideFaces.size() == 4 && m.faceVertexCount(f) == 4,
                "extrude: 4 side quads, top stays quad");
        const Vec3 after = m.faceCentroid(f);
        t.check(std::fabs((after - before).z - 1.0f) < 1e-5f,
                "extrude: top moved by offset");
        t.check(m.validate(&err), "extrude: validate (" + err + ")");

        // Inset the extruded top: same topology again, verts pulled inward.
        const Vec3 insetCentroid = m.faceCentroid(f);
        t.check(m.insetFace(f, 0.5f) == f, "inset: returns face");
        t.check((m.faceCentroid(f) - insetCentroid).length() < 1e-5f,
                "inset: centroid preserved");
        t.check(m.liveVertexCount() == 16 && m.liveFaceCount() == 14,
                "inset: counts V16 F14");
        t.check(m.validate(&err), "inset: validate (" + err + ")");
    }

    // --- loopCut ------------------------------------------------------------------
    {
        HalfEdgeMesh m;
        m.buildFromPolygons(cubePositions(), cubeQuads(), nullptr);
        HalfEdgeMesh::LoopCutResult cut;
        t.check(m.loopCut(0, 0.5f, &cut), "loopcut: cube ring cuts");
        t.check(cut.closed && cut.new_vertices.size() == 4 && cut.new_edges.size() == 4,
                "loopcut: closed ring, 4 verts + 4 edges");
        t.check(m.liveVertexCount() == 12 && m.liveEdgeCount() == 20 && m.liveFaceCount() == 10,
                "loopcut: counts V12 E20 F10");
        t.check(m.validate(&err), "loopcut: validate (" + err + ")");
        // The cut edges themselves form a closed edge loop.
        std::vector<HEIndex> loop;
        t.check(m.collectEdgeLoop(cut.new_edges[0], loop) && loop.size() == 4,
                "loopcut: cut edges form a closed loop of 4");
        bool splitsValid = cut.face_splits.size() == 4;
        for (const auto& split : cut.face_splits) {
            splitsValid = splitsValid && split.first < 6 && split.second >= 6;
        }
        t.check(splitsValid, "loopcut: face splits map new faces to originals");
    }
    {
        std::vector<Vec3> positions;
        std::vector<std::vector<int>> polys;
        gridPlane(3, 1, positions, polys); // 3-quad strip
        HalfEdgeMesh m;
        m.buildFromPolygons(positions, polys, nullptr);
        HalfEdgeMesh::LoopCutResult cut;
        const HEIndex crossEdge = m.findEdge(1, 5);
        t.check(crossEdge != kHEInvalid && m.loopCut(crossEdge, 0.25f, &cut),
                "loopcut: open strip cuts");
        t.check(!cut.closed && cut.new_vertices.size() == 4 && cut.new_edges.size() == 3,
                "loopcut: open ring, 4 verts + 3 edges");
        t.check(m.liveVertexCount() == 12 && m.liveEdgeCount() == 17 && m.liveFaceCount() == 6,
                "loopcut: strip counts V12 E17 F6");
        t.check(m.validate(&err), "loopcut: strip validate (" + err + ")");
        // t=0.25 must land at the same height on every cut vertex (no zigzag).
        bool sameParam = true;
        for (HEIndex v : cut.new_vertices) {
            sameParam = sameParam && std::fabs(m.vertices[v].position.z -
                                               m.vertices[cut.new_vertices[0]].position.z) < 1e-5f;
        }
        t.check(sameParam, "loopcut: consistent cut parameter across the strip");
    }

    report += t.ok ? "HalfEdge self-test: ALL PASS\n" : "HalfEdge self-test: FAILURES\n";
    return t.ok;
}

} // namespace MeshEdit
