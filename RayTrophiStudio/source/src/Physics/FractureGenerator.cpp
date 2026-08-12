/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          FractureGenerator.cpp
 * Author:        Kemal Demirtas
 * License:       MIT
 * =========================================================================
 *
 * Voronoi fracture (see FractureGenerator.h).
 *
 * Pipeline:
 *   1. Convex hull of the source vertices (incremental, orientation-by-reference
 *      so winding bugs can't flip a face inward). Used for site rejection, for
 *      the sliver threshold, and as the convex cutting body.
 *   2. Scatter `site_count` sites inside the hull (uniform, impact-clustered, or
 *      steered by recorded burn damage).
 *   3. For each site, clip by every bisector half-space against the other sites:
 *        - convex mode: clip the HULL polyhedron — always closed, but cavities
 *          and every surface UV are already gone before the first cut.
 *        - exact mode:  clip the SOURCE SOUP and seal each cut cross-section
 *          (chain cut edges into loops, bridge holes, ear clip). Keeps cavities,
 *          UVs and material IDs; falls back to the convex form for any cell
 *          whose cut cannot be sealed.
 *   4. Triangulate each cell into a shard; cull slivers.
 */

#include "FractureGenerator.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <random>
#include <unordered_set>
#include <utility>

namespace RayTrophiSim {
namespace {

constexpr float kEps = 1.0e-6f;

// Outward-oriented plane: a point x is INSIDE the half-space when n·x - d <= 0.
struct Plane {
    Vec3  n = Vec3(0.0f, 0.0f, 0.0f);
    float d = 0.0f;
    float signedDist(const Vec3& x) const { return n.dot(x) - d; }
};

// A convex polygon face (CCW around `normal`, which points outward).
struct Face {
    std::vector<Vec3> poly;
    Vec3 normal = Vec3(0.0f, 0.0f, 0.0f);
    bool interior = false;  // created by a cut (vs. an original hull face)
};

using Polyhedron = std::vector<Face>;

// ── Convex hull ──────────────────────────────────────────────────────────────

struct HullFace {
    int a, b, c;
    Vec3 n;     // outward normal
    float d;    // n·x = d on the face
};

// Orient a face so the reference interior point is inside (n·ref - d <= 0).
static HullFace makeOrientedFace(const std::vector<Vec3>& pts, int a, int b, int c,
                                 const Vec3& interior_ref) {
    HullFace f;
    f.a = a; f.b = b; f.c = c;
    Vec3 n = (pts[b] - pts[a]).cross(pts[c] - pts[a]);
    float len = n.length();
    if (len > kEps) n = n * (1.0f / len);
    float d = n.dot(pts[a]);
    if (n.dot(interior_ref) - d > 0.0f) {  // ref is outside → flip
        n = -n;
        d = -d;
        std::swap(f.b, f.c);
    }
    f.n = n;
    f.d = d;
    return f;
}

// Incremental convex hull. Returns false if the points are degenerate (coplanar
// / collinear → no 3D hull). Fills `out` with outward-oriented triangle faces.
static bool buildConvexHull(const std::vector<Vec3>& pts, std::vector<HullFace>& out) {
    out.clear();
    const int n = static_cast<int>(pts.size());
    if (n < 4) return false;

    // Seed tetrahedron from 4 well-separated, non-coplanar points.
    // p0/p1: the extreme pair along the widest spread axis.
    int p0 = 0, p1 = 0;
    {
        Vec3 mn = pts[0], mx = pts[0];
        std::array<int, 3> mni{0, 0, 0}, mxi{0, 0, 0};
        for (int i = 1; i < n; ++i) {
            for (int ax = 0; ax < 3; ++ax) {
                if (pts[i][ax] < mn[ax]) { mn[ax] = pts[i][ax]; mni[ax] = i; }
                if (pts[i][ax] > mx[ax]) { mx[ax] = pts[i][ax]; mxi[ax] = i; }
            }
        }
        float best = -1.0f;
        for (int ax = 0; ax < 3; ++ax) {
            float spread = (pts[mxi[ax]] - pts[mni[ax]]).length();
            if (spread > best) { best = spread; p0 = mni[ax]; p1 = mxi[ax]; }
        }
        if (best <= kEps) return false;  // all coincident
    }
    // p2: farthest from line p0-p1.
    int p2 = -1;
    {
        Vec3 dir = (pts[p1] - pts[p0]).normalize();
        float best = kEps;
        for (int i = 0; i < n; ++i) {
            Vec3 ap = pts[i] - pts[p0];
            float perp = (ap - dir * ap.dot(dir)).length();
            if (perp > best) { best = perp; p2 = i; }
        }
        if (p2 < 0) return false;  // collinear
    }
    // p3: farthest from plane p0-p1-p2.
    int p3 = -1;
    {
        Vec3 nrm = (pts[p1] - pts[p0]).cross(pts[p2] - pts[p0]).normalize();
        float best = kEps;
        for (int i = 0; i < n; ++i) {
            float dist = std::fabs((pts[i] - pts[p0]).dot(nrm));
            if (dist > best) { best = dist; p3 = i; }
        }
        if (p3 < 0) return false;  // coplanar
    }

    const Vec3 interior_ref = (pts[p0] + pts[p1] + pts[p2] + pts[p3]) * 0.25f;
    std::vector<HullFace> faces;
    faces.push_back(makeOrientedFace(pts, p0, p1, p2, interior_ref));
    faces.push_back(makeOrientedFace(pts, p0, p1, p3, interior_ref));
    faces.push_back(makeOrientedFace(pts, p0, p2, p3, interior_ref));
    faces.push_back(makeOrientedFace(pts, p1, p2, p3, interior_ref));

    auto edgeKey = [](int u, int v) -> long long {
        return (static_cast<long long>(u) << 32) | static_cast<unsigned int>(v);
    };

    for (int i = 0; i < n; ++i) {
        if (i == p0 || i == p1 || i == p2 || i == p3) continue;
        const Vec3& p = pts[i];
        // Find visible faces.
        std::vector<char> visible(faces.size(), 0);
        bool any_visible = false;
        for (size_t f = 0; f < faces.size(); ++f) {
            if (faces[f].n.dot(p) - faces[f].d > kEps) { visible[f] = 1; any_visible = true; }
        }
        if (!any_visible) continue;  // inside the current hull

        // Horizon = directed edges of visible faces whose reverse is NOT visible.
        std::unordered_set<long long> visibleEdges;
        for (size_t f = 0; f < faces.size(); ++f) {
            if (!visible[f]) continue;
            const HullFace& hf = faces[f];
            visibleEdges.insert(edgeKey(hf.a, hf.b));
            visibleEdges.insert(edgeKey(hf.b, hf.c));
            visibleEdges.insert(edgeKey(hf.c, hf.a));
        }
        std::vector<std::pair<int, int>> horizon;
        auto checkHorizon = [&](int u, int v) {
            if (visibleEdges.find(edgeKey(v, u)) == visibleEdges.end())
                horizon.emplace_back(u, v);
        };
        for (size_t f = 0; f < faces.size(); ++f) {
            if (!visible[f]) continue;
            const HullFace& hf = faces[f];
            checkHorizon(hf.a, hf.b);
            checkHorizon(hf.b, hf.c);
            checkHorizon(hf.c, hf.a);
        }

        // Remove visible faces.
        std::vector<HullFace> kept;
        kept.reserve(faces.size());
        for (size_t f = 0; f < faces.size(); ++f)
            if (!visible[f]) kept.push_back(faces[f]);
        // Stitch new faces from p to each horizon edge.
        for (const auto& e : horizon)
            kept.push_back(makeOrientedFace(pts, e.first, e.second, i, interior_ref));
        faces.swap(kept);
        if (faces.size() > static_cast<size_t>(8 * n + 16)) break;  // safety against runaway
    }

    out = std::move(faces);
    return out.size() >= 4;
}

// ── Convex polyhedron clipping ───────────────────────────────────────────────

// Sutherland-Hodgman clip of one polygon by `plane`, keeping the n·x-d <= 0 side.
// Intersection points (which lie on the plane) are appended to `cap_pts`.
static void clipPolygon(const std::vector<Vec3>& in, const Plane& plane,
                        std::vector<Vec3>& out, std::vector<Vec3>& cap_pts) {
    out.clear();
    const size_t m = in.size();
    if (m < 3) return;
    for (size_t i = 0; i < m; ++i) {
        const Vec3& A = in[i];
        const Vec3& B = in[(i + 1) % m];
        float dA = plane.signedDist(A);
        float dB = plane.signedDist(B);
        bool Ain = dA <= kEps;
        bool Bin = dB <= kEps;
        if (Ain) out.push_back(A);
        if (Ain != Bin) {
            float denom = dA - dB;
            float t = (std::fabs(denom) > 1e-20f) ? (dA / denom) : 0.0f;
            Vec3 I = A + (B - A) * t;
            out.push_back(I);
            cap_pts.push_back(I);
        }
    }
    if (out.size() < 3) out.clear();
}

// Order coplanar convex points CCW around `normal`.
static std::vector<Vec3> orderConvex(std::vector<Vec3> pts, const Vec3& normal) {
    if (pts.size() < 3) return {};
    Vec3 c(0.0f, 0.0f, 0.0f);
    for (const Vec3& p : pts) c += p;
    c = c * (1.0f / static_cast<float>(pts.size()));
    // Build an in-plane basis.
    Vec3 u = std::fabs(normal.x) > 0.9f ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
    u = (u - normal * u.dot(normal));
    float ul = u.length();
    if (ul < kEps) return {};
    u = u * (1.0f / ul);
    Vec3 v = normal.cross(u);
    std::sort(pts.begin(), pts.end(), [&](const Vec3& a, const Vec3& b) {
        Vec3 da = a - c, db = b - c;
        return std::atan2(da.dot(v), da.dot(u)) < std::atan2(db.dot(v), db.dot(u));
    });
    // Drop near-duplicate consecutive points.
    std::vector<Vec3> result;
    for (const Vec3& p : pts) {
        if (result.empty() || (p - result.back()).length() > 1e-5f) result.push_back(p);
    }
    if (result.size() >= 2 && (result.front() - result.back()).length() <= 1e-5f) result.pop_back();
    return result;
}

// Clip the polyhedron by a half-space; rebuild the cut cap. Returns false if the
// polyhedron is fully clipped away. `cap_interior` flags the new cap face.
static bool clipPolyhedron(Polyhedron& poly, const Plane& plane, bool cap_interior) {
    Polyhedron next;
    next.reserve(poly.size() + 1);
    std::vector<Vec3> cap_pts;
    std::vector<Vec3> clipped;
    for (const Face& f : poly) {
        clipPolygon(f.poly, plane, clipped, cap_pts);
        if (clipped.size() >= 3) {
            Face nf;
            nf.poly = clipped;
            nf.normal = f.normal;
            nf.interior = f.interior;
            next.push_back(std::move(nf));
        }
    }
    if (cap_pts.size() >= 3) {
        std::vector<Vec3> cap = orderConvex(cap_pts, plane.n);
        if (cap.size() >= 3) {
            Face cf;
            cf.poly = std::move(cap);
            cf.normal = plane.n;
            cf.interior = cap_interior;
            next.push_back(std::move(cf));
        }
    }
    poly.swap(next);
    return poly.size() >= 4;
}

// ── Exact surface clipping ───────────────────────────────────────────────────
//
// Same sites, same half-spaces, different thing being cut: the SOURCE TRIANGLE
// SOUP instead of its convex hull. What survives a cut is the original surface,
// so cavities, concave profiles and UVs come through untouched — the hull path
// destroys all three before the first cut is even made.
//
// The price is that a cut cross-section is no longer guaranteed convex, and may
// be several disjoint loops with holes in them (cut a mug across the handle).
// `orderConvex` cannot seal that. What follows is the machinery that can:
// segments -> welded loops -> nesting -> hole bridging -> ear clipping.
//
// ★ ONE INVARIANT MAKES ALL OF THIS TRACTABLE: every face stays CONVEX. Source
// triangles are convex, the intersection of a convex polygon with a half-space
// is convex, and the caps below are emitted as TRIANGLES rather than as n-gons.
// So no face can cross a plane more than twice, which is exactly the condition
// under which Sutherland-Hodgman is exact and each face contributes at most ONE
// cut segment. Emit a cap as a single n-gon and that invariant is gone.

struct Vert {
    Vec3 p;
    Vec2 uv;
};

struct AttrFace {
    std::vector<Vert> poly;
    Vec3 normal = Vec3(0.0f, 0.0f, 0.0f);
    bool interior = false;
    uint16_t material = 0;
};

static Vert lerpVert(const Vert& A, const Vert& B, float t) {
    Vert o;
    o.p = A.p + (B.p - A.p) * t;
    o.uv = A.uv + (B.uv - A.uv) * t;
    return o;
}

// Clip an attributed convex polygon by `plane`, keeping n·x - d <= 0. When the
// polygon straddles the plane the two crossing points are also reported as the
// cut segment, ordered exit -> entry so the segment runs along the polygon's own
// winding.
static void clipAttrPolygon(const std::vector<Vert>& in, const Plane& plane,
                            std::vector<Vert>& out,
                            Vec3& seg_from, Vec3& seg_to, bool& has_segment) {
    out.clear();
    has_segment = false;
    const size_t m = in.size();
    if (m < 3) return;
    bool have_exit = false, have_entry = false;
    Vec3 exit_p(0.0f), entry_p(0.0f);
    for (size_t i = 0; i < m; ++i) {
        const Vert& A = in[i];
        const Vert& B = in[(i + 1) % m];
        const float dA = plane.signedDist(A.p);
        const float dB = plane.signedDist(B.p);
        const bool Ain = dA <= kEps;
        const bool Bin = dB <= kEps;
        if (Ain) out.push_back(A);
        if (Ain != Bin) {
            const float denom = dA - dB;
            const float t = (std::fabs(denom) > 1e-20f) ? (dA / denom) : 0.0f;
            const Vert I = lerpVert(A, B, std::clamp(t, 0.0f, 1.0f));
            out.push_back(I);
            if (Ain) { exit_p = I.p; have_exit = true; }
            else     { entry_p = I.p; have_entry = true; }
        }
    }
    if (out.size() < 3) { out.clear(); return; }
    if (have_exit && have_entry) {
        seg_from = exit_p;
        seg_to = entry_p;
        has_segment = true;
    }
}

// Quantised position key. Two segments that meet at a shared corner must agree
// BIT FOR BIT to chain, and two independently computed intersections of the same
// edge with the same plane generally do not. Welding on a grid is what turns a
// pile of nearly-touching segments into closed loops; without it the loop walk
// dead-ends and the cap is dropped as unsealed.
struct WeldKey {
    int64_t x, y, z;
    bool operator<(const WeldKey& o) const {
        if (x != o.x) return x < o.x;
        if (y != o.y) return y < o.y;
        return z < o.z;
    }
};

static WeldKey weldKey(const Vec3& p, float cell) {
    const float inv = 1.0f / cell;
    return WeldKey{static_cast<int64_t>(std::llround(p.x * inv)),
                   static_cast<int64_t>(std::llround(p.y * inv)),
                   static_cast<int64_t>(std::llround(p.z * inv))};
}

// Chain cut segments into closed loops.
//
// ★ AN OPEN CHAIN IS CLOSED ANYWAY, and reporting it is the whole subtlety.
//
// The first version refused the entire cell the moment ONE chain failed to
// close. On a real asset that is a disaster: game meshes routinely have their
// hidden faces deleted, so a 7000-triangle water tower is open in a dozen small
// places, and a single one of them anywhere in the cut discarded a whole shard's
// worth of exact geometry. Measured on SM_Water_Tower: 31 of 35 cells thrown
// away, i.e. the feature did nothing at all on the first asset it met.
//
// A chain that does not close means the cut crossed a hole in the surface. The
// cross-section there is genuinely unknown, and joining its two loose ends is a
// guess — but it is a LOCAL guess, over the width of a hole that was already
// invisible in the source, and the alternative is to throw away the entire
// surface of that shard. So: close it, and count it, so nobody reads an
// approximated shard as an exact one. `out_approximated` is that count, not a
// failure flag.
static bool chainSegmentsToLoops(const std::vector<std::pair<Vec3, Vec3>>& segments,
                                 float weld_cell,
                                 std::vector<std::vector<Vec3>>& out_loops,
                                 bool& out_approximated) {
    out_loops.clear();
    out_approximated = false;
    if (segments.empty()) return true;

    std::map<WeldKey, std::vector<size_t>> outgoing;
    for (size_t i = 0; i < segments.size(); ++i)
        outgoing[weldKey(segments[i].first, weld_cell)].push_back(i);

    std::vector<bool> used(segments.size(), false);
    for (size_t start = 0; start < segments.size(); ++start) {
        if (used[start]) continue;
        std::vector<Vec3> loop;
        const WeldKey start_key = weldKey(segments[start].first, weld_cell);
        size_t current = start;
        bool closed = false;
        // Bounded by the segment count: a walk that revisits nothing cannot be
        // longer, and this stops a corrupt map from spinning forever.
        for (size_t guard = 0; guard <= segments.size(); ++guard) {
            used[current] = true;
            loop.push_back(segments[current].first);
            const WeldKey tail = weldKey(segments[current].second, weld_cell);
            if (tail < start_key || start_key < tail) {
                // Not back at the start yet: follow an unused segment leaving here.
                const auto it = outgoing.find(tail);
                if (it == outgoing.end()) break;
                size_t next = segments.size();
                for (size_t candidate : it->second)
                    if (!used[candidate]) { next = candidate; break; }
                if (next == segments.size()) break;
                current = next;
                continue;
            }
            closed = true;
            break;
        }
        if (!closed) out_approximated = true;   // loose ends joined below
        if (loop.size() >= 3) out_loops.push_back(std::move(loop));
    }
    // Every chain was a sliver (two-point dead ends and the like): there is no
    // cross-section to seal with, and inventing one would be fabrication rather
    // than approximation.
    return !out_loops.empty();
}

// ── Cap triangulation (concave, multi-loop, holes) ───────────────────────────

struct Loop2 {
    std::vector<Vec2> pts;     // projected into the cut plane's basis
    std::vector<Vec3> world;   // the exact 3D position of each pts[i]
    float area = 0.0f;         // signed, CCW positive
    bool  hole = false;
};

static float signedArea2(const std::vector<Vec2>& p) {
    double s = 0.0;
    for (size_t i = 0; i < p.size(); ++i) {
        const Vec2& a = p[i];
        const Vec2& b = p[(i + 1) % p.size()];
        s += static_cast<double>(a.x) * b.y - static_cast<double>(b.x) * a.y;
    }
    return static_cast<float>(s * 0.5);
}

static bool pointInLoop2(const std::vector<Vec2>& poly, const Vec2& q) {
    bool inside = false;
    for (size_t i = 0, j = poly.size() - 1; i < poly.size(); j = i++) {
        const Vec2& a = poly[i];
        const Vec2& b = poly[j];
        if ((a.y > q.y) != (b.y > q.y)) {
            const float t = (q.y - a.y) / ((b.y - a.y) != 0.0f ? (b.y - a.y) : 1e-20f);
            if (q.x < a.x + t * (b.x - a.x)) inside = !inside;
        }
    }
    return inside;
}

// Proper segment intersection (shared endpoints do not count).
static bool segmentsCross(const Vec2& p1, const Vec2& p2,
                          const Vec2& p3, const Vec2& p4) {
    auto orient = [](const Vec2& a, const Vec2& b, const Vec2& c) {
        const float v = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
        return (v > 1e-9f) ? 1 : (v < -1e-9f ? -1 : 0);
    };
    const int d1 = orient(p1, p2, p3), d2 = orient(p1, p2, p4);
    const int d3 = orient(p3, p4, p1), d4 = orient(p3, p4, p2);
    return d1 != 0 && d2 != 0 && d3 != 0 && d4 != 0 &&
           d1 != d2 && d3 != d4;
}

// Splice each hole into its outer loop with a bridge, producing one simple
// polygon that ear clipping can chew.
//
// The textbook method (Eberly: ray-cast from the hole's rightmost vertex, then
// disambiguate with reflex vertices) is what a general triangulator needs. A cut
// cross-section is not general: it is a handful of vertices, and the honest
// O(n·m·E) version — try candidate pairs nearest-first, take the first bridge
// that crosses no edge — is far easier to be sure is CORRECT. A wrong bridge
// makes a self-intersecting polygon and silently bad geometry; that trade is
// worth more here than the speed.
static void bridgeHoles(std::vector<Vec2>& outer, std::vector<Vec3>& outer_world,
                        const std::vector<const Loop2*>& holes) {
    for (const Loop2* hole : holes) {
        if (hole->pts.size() < 3) continue;
        struct Candidate { float d2; size_t oi, hi; };
        std::vector<Candidate> candidates;
        candidates.reserve(outer.size() * hole->pts.size());
        for (size_t oi = 0; oi < outer.size(); ++oi) {
            for (size_t hi = 0; hi < hole->pts.size(); ++hi) {
                const Vec2 d = outer[oi] - hole->pts[hi];
                candidates.push_back({d.x * d.x + d.y * d.y, oi, hi});
            }
        }
        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate& a, const Candidate& b) { return a.d2 < b.d2; });

        bool spliced = false;
        for (const Candidate& c : candidates) {
            const Vec2& A = outer[c.oi];
            const Vec2& B = hole->pts[c.hi];
            bool blocked = false;
            for (size_t i = 0; i < outer.size() && !blocked; ++i)
                blocked = segmentsCross(A, B, outer[i], outer[(i + 1) % outer.size()]);
            for (size_t i = 0; i < hole->pts.size() && !blocked; ++i)
                blocked = segmentsCross(A, B, hole->pts[i],
                                        hole->pts[(i + 1) % hole->pts.size()]);
            if (blocked) continue;

            // outer[0..oi] + hole[hi..hi] (full cycle) + outer[oi..end]
            std::vector<Vec2> merged;
            std::vector<Vec3> merged_world;
            merged.reserve(outer.size() + hole->pts.size() + 2);
            merged_world.reserve(merged.capacity());
            for (size_t i = 0; i <= c.oi; ++i) {
                merged.push_back(outer[i]);
                merged_world.push_back(outer_world[i]);
            }
            for (size_t k = 0; k <= hole->pts.size(); ++k) {
                const size_t i = (c.hi + k) % hole->pts.size();
                merged.push_back(hole->pts[i]);
                merged_world.push_back(hole->world[i]);
            }
            for (size_t i = c.oi; i < outer.size(); ++i) {
                merged.push_back(outer[i]);
                merged_world.push_back(outer_world[i]);
            }
            outer.swap(merged);
            outer_world.swap(merged_world);
            spliced = true;
            break;
        }
        // No visible bridge (degenerate cut): the hole is dropped rather than
        // splicing something that self-intersects. The cap is then slightly too
        // solid, which is a far smaller lie than inverted geometry.
        (void)spliced;
    }
}

// Ear clipping of one simple, CCW polygon.
static void earClip(const std::vector<Vec2>& pts, const std::vector<Vec3>& world,
                    std::vector<std::array<Vec3, 3>>& out_tris) {
    const size_t n = pts.size();
    if (n < 3) return;
    std::vector<size_t> idx(n);
    for (size_t i = 0; i < n; ++i) idx[i] = i;

    auto cross2 = [](const Vec2& a, const Vec2& b, const Vec2& c) {
        return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    };
    auto inTriangle = [&](const Vec2& a, const Vec2& b, const Vec2& c, const Vec2& q) {
        const float d1 = cross2(a, b, q), d2 = cross2(b, c, q), d3 = cross2(c, a, q);
        const bool neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
        const bool pos = (d1 > 0) || (d2 > 0) || (d3 > 0);
        return !(neg && pos);
    };

    size_t guard = 0;
    while (idx.size() > 3 && guard++ < n * n + 16) {
        bool clipped = false;
        for (size_t i = 0; i < idx.size(); ++i) {
            const size_t i0 = idx[(i + idx.size() - 1) % idx.size()];
            const size_t i1 = idx[i];
            const size_t i2 = idx[(i + 1) % idx.size()];
            if (cross2(pts[i0], pts[i1], pts[i2]) <= 0.0f) continue;  // reflex
            bool contains = false;
            for (size_t k = 0; k < idx.size() && !contains; ++k) {
                const size_t j = idx[k];
                if (j == i0 || j == i1 || j == i2) continue;
                contains = inTriangle(pts[i0], pts[i1], pts[i2], pts[j]);
            }
            if (contains) continue;
            out_tris.push_back({world[i0], world[i1], world[i2]});
            idx.erase(idx.begin() + static_cast<std::ptrdiff_t>(i));
            clipped = true;
            break;
        }
        // No ear found: the polygon is degenerate (collinear run or a bridge that
        // touched). Stop rather than loop; the fan below still closes it roughly.
        if (!clipped) break;
    }
    for (size_t k = 1; k + 1 < idx.size(); ++k)
        out_tris.push_back({world[idx[0]], world[idx[k]], world[idx[k + 1]]});
}

// Seal one cut: loops -> nesting -> bridged simple polygons -> triangles.
static void triangulateCap(const std::vector<std::vector<Vec3>>& loops,
                           const Vec3& n,
                           std::vector<std::array<Vec3, 3>>& out_tris) {
    if (loops.empty()) return;
    Vec3 u = std::fabs(n.x) > 0.9f ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
    u = u - n * u.dot(n);
    const float ul = u.length();
    if (ul < kEps) return;
    u = u * (1.0f / ul);
    const Vec3 v = n.cross(u);

    std::vector<Loop2> loops2;
    loops2.reserve(loops.size());
    for (const std::vector<Vec3>& loop : loops) {
        Loop2 l;
        l.world = loop;
        l.pts.reserve(loop.size());
        for (const Vec3& p : loop) l.pts.push_back(Vec2(p.dot(u), p.dot(v)));
        l.area = signedArea2(l.pts);
        if (std::fabs(l.area) < 1e-10f) continue;   // slivers seal nothing
        loops2.push_back(std::move(l));
    }
    if (loops2.empty()) return;

    // Even-odd nesting decides hole-ness. Orientation does NOT: the segments were
    // welded and chained, and a loop's winding after that says nothing reliable
    // about whether it is a rim or a cavity.
    for (size_t i = 0; i < loops2.size(); ++i) {
        int depth = 0;
        for (size_t j = 0; j < loops2.size(); ++j) {
            if (i == j) continue;
            if (pointInLoop2(loops2[j].pts, loops2[i].pts[0])) ++depth;
        }
        loops2[i].hole = (depth % 2) == 1;
    }

    for (size_t i = 0; i < loops2.size(); ++i) {
        if (loops2[i].hole) continue;
        std::vector<const Loop2*> holes;
        for (size_t j = 0; j < loops2.size(); ++j) {
            if (i == j || !loops2[j].hole) continue;
            if (pointInLoop2(loops2[i].pts, loops2[j].pts[0])) holes.push_back(&loops2[j]);
        }
        std::vector<Vec2> outer = loops2[i].pts;
        std::vector<Vec3> outer_world = loops2[i].world;
        // Ear clipping below assumes CCW; holes must run the other way so the
        // bridged polygon stays simple.
        if (loops2[i].area < 0.0f) {
            std::reverse(outer.begin(), outer.end());
            std::reverse(outer_world.begin(), outer_world.end());
        }
        std::vector<Loop2> oriented_holes;
        oriented_holes.reserve(holes.size());
        for (const Loop2* h : holes) {
            Loop2 copy = *h;
            if (copy.area > 0.0f) {
                std::reverse(copy.pts.begin(), copy.pts.end());
                std::reverse(copy.world.begin(), copy.world.end());
            }
            oriented_holes.push_back(std::move(copy));
        }
        std::vector<const Loop2*> hole_ptrs;
        hole_ptrs.reserve(oriented_holes.size());
        for (const Loop2& h : oriented_holes) hole_ptrs.push_back(&h);

        bridgeHoles(outer, outer_world, hole_ptrs);
        earClip(outer, outer_world, out_tris);
    }
}

enum class ExactCell {
    Ok,           // every cut sealed against real geometry
    Approximated, // a cut crossed a hole in the surface; its ends were joined
    Empty,        // the cell is outside the object entirely
    Unsealed      // no cross-section at all — nothing honest to build from
};

// Clip the attributed soup by every plane, sealing each cut as it goes.
static ExactCell buildExactCell(const std::vector<AttrFace>& source,
                                const std::vector<Plane>& planes,
                                float weld_cell,
                                std::vector<AttrFace>& out_faces) {
    out_faces = source;
    std::vector<AttrFace> next;
    std::vector<Vert> clipped;
    std::vector<std::pair<Vec3, Vec3>> segments;
    std::vector<std::vector<Vec3>> loops;
    std::vector<std::array<Vec3, 3>> cap_tris;

    // AABB of the faces still alive, maintained so a plane that cannot cut them
    // can be skipped outright. Without this every cell pays for every bisector
    // over the whole soup — and most bisectors of a many-site fracture pass
    // nowhere near any given cell, so the overwhelming majority of that work
    // clips nothing. This is the difference between an authoring action that
    // takes a moment and one that takes a minute on a dense mesh.
    auto boundsOf = [](const std::vector<AttrFace>& faces, Vec3& lo, Vec3& hi) {
        lo = Vec3(1e30f, 1e30f, 1e30f);
        hi = Vec3(-1e30f, -1e30f, -1e30f);
        for (const AttrFace& f : faces)
            for (const Vert& v : f.poly) {
                lo = Vec3(std::min(lo.x, v.p.x), std::min(lo.y, v.p.y), std::min(lo.z, v.p.z));
                hi = Vec3(std::max(hi.x, v.p.x), std::max(hi.y, v.p.y), std::max(hi.z, v.p.z));
            }
    };
    Vec3 lo(0.0f), hi(0.0f);
    boundsOf(out_faces, lo, hi);
    bool any_approximated = false;

    for (const Plane& plane : planes) {
        // Farthest corner of the AABB along the plane normal: if even that is
        // inside, nothing can be outside, so this plane is a no-op.
        const Vec3 far_corner(plane.n.x >= 0.0f ? hi.x : lo.x,
                              plane.n.y >= 0.0f ? hi.y : lo.y,
                              plane.n.z >= 0.0f ? hi.z : lo.z);
        if (plane.signedDist(far_corner) <= kEps) continue;

        next.clear();
        segments.clear();
        for (const AttrFace& f : out_faces) {
            Vec3 sa(0.0f), sb(0.0f);
            bool has_segment = false;
            clipAttrPolygon(f.poly, plane, clipped, sa, sb, has_segment);
            if (has_segment) segments.emplace_back(sa, sb);
            if (clipped.size() >= 3) {
                AttrFace nf;
                nf.poly = clipped;
                nf.normal = f.normal;
                nf.interior = f.interior;
                nf.material = f.material;
                next.push_back(std::move(nf));
            }
        }
        if (next.empty()) { out_faces.clear(); return ExactCell::Empty; }

        loops.clear();
        bool approximated = false;
        if (!chainSegmentsToLoops(segments, weld_cell, loops, approximated))
            return ExactCell::Unsealed;
        if (approximated) any_approximated = true;

        cap_tris.clear();
        triangulateCap(loops, plane.n, cap_tris);
        if (!segments.empty() && cap_tris.empty()) return ExactCell::Unsealed;

        for (const std::array<Vec3, 3>& t : cap_tris) {
            AttrFace cf;
            cf.normal = plane.n;
            cf.interior = true;
            // Cut faces have no source surface to inherit a UV from. They are
            // flagged interior so the caller can give them their own material —
            // a fresh break must not sample the char mask, because it is not
            // burnt. Leaving the UV at (0,0) and the material inherited would
            // paint the inside of the shard with whatever is at that texel.
            Vec3 fn = (t[1] - t[0]).cross(t[2] - t[0]);
            cf.poly = (fn.dot(plane.n) < 0.0f)
                ? std::vector<Vert>{{t[0], Vec2(0, 0)}, {t[2], Vec2(0, 0)}, {t[1], Vec2(0, 0)}}
                : std::vector<Vert>{{t[0], Vec2(0, 0)}, {t[1], Vec2(0, 0)}, {t[2], Vec2(0, 0)}};
            next.push_back(std::move(cf));
        }
        out_faces.swap(next);
        boundsOf(out_faces, lo, hi);
    }
    if (out_faces.size() < 4) return ExactCell::Empty;
    return any_approximated ? ExactCell::Approximated : ExactCell::Ok;
}

static void triangulateAttrInto(const std::vector<AttrFace>& faces,
                                FractureShard& shard) {
    for (const AttrFace& f : faces) {
        if (f.poly.size() < 3) continue;
        for (size_t k = 1; k + 1 < f.poly.size(); ++k) {
            Vert a = f.poly[0], b = f.poly[k], c = f.poly[k + 1];
            Vec3 fn = (b.p - a.p).cross(c.p - a.p);
            if (fn.dot(f.normal) < 0.0f) std::swap(b, c);
            FractureShardTri t;
            t.a = a.p; t.b = b.p; t.c = c.p;
            t.ua = a.uv; t.ub = b.uv; t.uc = c.uv;
            t.n = f.normal;
            t.interior = f.interior;
            t.material = f.material;
            shard.tris.push_back(t);
        }
    }
}

// ── Shard assembly ───────────────────────────────────────────────────────────

static void triangulateInto(const Polyhedron& poly, FractureShard& shard) {
    for (const Face& f : poly) {
        if (f.poly.size() < 3) continue;
        for (size_t k = 1; k + 1 < f.poly.size(); ++k) {
            const Vec3& a = f.poly[0];
            Vec3 b = f.poly[k];
            Vec3 c = f.poly[k + 1];
            // Winding consistent with the stored outward normal.
            Vec3 fn = (b - a).cross(c - a);
            if (fn.dot(f.normal) < 0.0f) std::swap(b, c);
            FractureShardTri t;
            t.a = a; t.b = b; t.c = c;
            t.n = f.normal;
            t.interior = f.interior;
            shard.tris.push_back(t);
        }
    }
}

// Volume + centroid via the divergence (tetra-to-origin) sum over the surface.
static void computeMassProps(FractureShard& shard) {
    double vol6 = 0.0;
    Vec3 acc(0.0f, 0.0f, 0.0f);
    for (const FractureShardTri& t : shard.tris) {
        float sv = t.a.dot(t.b.cross(t.c));  // 6× signed tetra volume
        vol6 += sv;
        acc += (t.a + t.b + t.c) * sv;
    }
    float vol = static_cast<float>(vol6) / 6.0f;
    shard.volume = std::fabs(vol);
    if (std::fabs(vol6) > 1e-12f)
        shard.centroid = acc * (1.0f / (4.0f * static_cast<float>(vol6)));
}

// ── Structural clustering ────────────────────────────────────────────────────
// Partition the surviving shards into `k` spatially contiguous groups, so a
// blast detaches the part of the object it actually hit rather than the whole
// object.
//
// ★ DETERMINISTIC BY CONSTRUCTION — no RNG anywhere. Farthest-point seeding
// picks the same k centres for the same shard set every time, and Lloyd
// relaxation is a fixed number of averaging passes. This is not fussiness: the
// cluster index decides which rigid bodies exist, so a run-to-run difference
// here would make a cached timeline replay a DIFFERENT shatter, which is the
// one thing the destruction cache contract cannot tolerate.
static void assignStructuralClusters(std::vector<FractureShard>& shards, int k) {
    const int count = static_cast<int>(shards.size());
    if (count <= 0) return;
    k = std::max(1, std::min(k, count));
    if (k == 1) {
        for (FractureShard& s : shards) s.cluster = 0;
        return;
    }

    // Farthest-point (k-center) seeding: start from shard 0, then repeatedly
    // take the shard furthest from every centre chosen so far. Spreads the
    // seeds over the object instead of clumping them the way random picks do.
    std::vector<Vec3> centres;
    centres.reserve(k);
    centres.push_back(shards[0].centroid);
    std::vector<float> nearest(count, std::numeric_limits<float>::max());
    for (int c = 1; c < k; ++c) {
        int best = -1;
        float best_distance = -1.0f;
        for (int i = 0; i < count; ++i) {
            const Vec3 d = shards[i].centroid - centres.back();
            nearest[i] = std::min(nearest[i], d.dot(d));
            if (nearest[i] > best_distance) { best_distance = nearest[i]; best = i; }
        }
        if (best < 0 || !(best_distance > 0.0f)) break;  // fewer distinct positions than k
        centres.push_back(shards[best].centroid);
    }

    // Lloyd relaxation. Eight passes is well past the point where the labelling
    // stops moving for the shard counts this generator produces.
    const int cluster_count = static_cast<int>(centres.size());
    std::vector<int> label(count, 0);
    for (int pass = 0; pass < 8; ++pass) {
        bool changed = false;
        for (int i = 0; i < count; ++i) {
            int best = 0;
            float best_distance = std::numeric_limits<float>::max();
            for (int c = 0; c < cluster_count; ++c) {
                const Vec3 d = shards[i].centroid - centres[c];
                const float distance = d.dot(d);
                if (distance < best_distance) { best_distance = distance; best = c; }
            }
            if (label[i] != best) { label[i] = best; changed = true; }
        }
        if (!changed) break;
        std::vector<Vec3> accum(cluster_count, Vec3(0.0f, 0.0f, 0.0f));
        // Weight by volume: a cluster's centre should follow its MASS, so a
        // shower of slivers cannot drag it away from the block it belongs to.
        std::vector<float> weight(cluster_count, 0.0f);
        for (int i = 0; i < count; ++i) {
            const float w = std::max(shards[i].volume, 1.0e-9f);
            accum[label[i]] += shards[i].centroid * w;
            weight[label[i]] += w;
        }
        for (int c = 0; c < cluster_count; ++c)
            if (weight[c] > 0.0f) centres[c] = accum[c] * (1.0f / weight[c]);
    }

    // Compact the labels so the emitted cluster indices are 0..n-1 with no gaps
    // (an empty cluster would otherwise become an empty fracture group).
    std::vector<int> remap(cluster_count, -1);
    int next = 0;
    for (int i = 0; i < count; ++i) {
        if (remap[label[i]] < 0) remap[label[i]] = next++;
        shards[i].cluster = remap[label[i]];
    }
}

} // namespace

bool generateFracture(const std::vector<FractureInputTri>& source,
                      const FractureParams& params,
                      std::vector<FractureShard>& out_shards,
                      FractureStats* out_stats) {
    FractureStats stats;
    if (out_stats) *out_stats = stats;
    out_shards.clear();
    if (source.empty()) return false;

    // Dedup source vertices (quantized) → hull input.
    std::vector<Vec3> pts;
    pts.reserve(source.size() * 3);
    {
        std::map<std::array<int64_t, 3>, int> seen;
        const double q = 1.0e5;  // 10 micron weld
        auto add = [&](const Vec3& p) {
            std::array<int64_t, 3> key{
                static_cast<int64_t>(std::llround(static_cast<double>(p.x) * q)),
                static_cast<int64_t>(std::llround(static_cast<double>(p.y) * q)),
                static_cast<int64_t>(std::llround(static_cast<double>(p.z) * q))};
            if (seen.emplace(key, 1).second) pts.push_back(p);
        };
        for (const FractureInputTri& t : source) { add(t.a); add(t.b); add(t.c); }
    }
    if (pts.size() < 4) return false;

    std::vector<HullFace> hull;
    if (!buildConvexHull(pts, hull)) return false;

    // Reference polyhedron = the hull as outward polygon faces (each a triangle).
    Polyhedron hull_poly;
    hull_poly.reserve(hull.size());
    for (const HullFace& hf : hull) {
        Face f;
        f.poly = {pts[hf.a], pts[hf.b], pts[hf.c]};
        f.normal = hf.n;
        f.interior = false;
        hull_poly.push_back(std::move(f));
    }

    // Hull AABB + a quick inside test (used for site rejection).
    Vec3 mn = pts[0], mx = pts[0];
    for (const Vec3& p : pts) {
        mn = Vec3(std::min(mn.x, p.x), std::min(mn.y, p.y), std::min(mn.z, p.z));
        mx = Vec3(std::max(mx.x, p.x), std::max(mx.y, p.y), std::max(mx.z, p.z));
    }
    auto insideHull = [&](const Vec3& p) {
        for (const HullFace& hf : hull)
            if (hf.n.dot(p) - hf.d > -1.0e-4f) return false;  // a touch inside the faces
        return true;
    };

    // Hull volume (for the sliver cull threshold).
    FractureShard hull_shard;
    triangulateInto(hull_poly, hull_shard);
    computeMassProps(hull_shard);
    const float min_volume = std::max(0.0f, params.min_shard_volume_ratio) * hull_shard.volume;

    // Scatter sites inside the hull — unless this is a replay, in which case the
    // sites were decided once, at authoring time, and are restored verbatim.
    // See FractureParams::explicit_sites for why nothing here may filter them.
    const int want = std::max(1, params.site_count);
    std::vector<Vec3> sites = params.explicit_sites;
    if (sites.empty()) {
        sites.reserve(want);
        std::mt19937 rng(params.seed ? params.seed : 1u);
        std::uniform_real_distribution<float> ux(mn.x, mx.x), uy(mn.y, mx.y), uz(mn.z, mx.z);
        std::normal_distribution<float> gauss(0.0f, 1.0f);
        const int max_attempts = want * 200 + 1000;

        // ── ThermalWeakened: a CDF over the damaged surface elements ──────────
        // ★ This is the payoff of the whole material-transformation chain. MSF
        // already knows, per surface texel, how much mass that patch has lost to
        // pyrolysis, melting and APIC transfer. Sampling seed positions from that
        // distribution makes the object break where it BURNT, which is the thing
        // a noise-seeded Voronoi can never do no matter how good the cutting is.
        //
        // ★ And it is why the sites have to be SAVED rather than re-derived: this
        // distribution is a snapshot of the damage field at the moment the artist
        // pressed the button, and reopening the project does not bring it back.
        //
        // Seeds are pulled slightly INWARD from the damaged surface point (surface
        // samples sit on the hull boundary and would be rejected by insideHull,
        // and a cell seeded exactly on the surface degenerates to a sliver).
        std::vector<float> damage_cdf;
        if (params.pattern == FracturePattern::ThermalWeakened) {
            damage_cdf.reserve(params.damage_samples.size());
            float running = 0.0f;
            for (const FractureDamageSample& sample : params.damage_samples) {
                running += std::max(sample.weight, 0.0f);
                damage_cdf.push_back(running);
            }
            // No damage yet: an unburnt object must still be fracturable, so fall
            // through to uniform rather than emitting nothing.
            if (damage_cdf.empty() || !(damage_cdf.back() > 0.0f)) damage_cdf.clear();
        }
        const Vec3 hull_centre = (mn + mx) * 0.5f;
        const float inward = std::max((mx - mn).length() * 0.06f, 1.0e-4f);
        std::uniform_real_distribution<float> upick(0.0f, 1.0f);

        for (int attempt = 0; attempt < max_attempts && static_cast<int>(sites.size()) < want; ++attempt) {
            Vec3 candidate;
            if (params.pattern == FracturePattern::ImpactClustered) {
                float r = std::max(1.0e-4f, params.impact_radius);
                candidate = params.impact_point + Vec3(gauss(rng), gauss(rng), gauss(rng)) * r;
            } else if (!damage_cdf.empty() &&
                       upick(rng) <= std::clamp(params.damage_bias, 0.0f, 1.0f)) {
                const float pick = upick(rng) * damage_cdf.back();
                const std::size_t index = static_cast<std::size_t>(
                    std::lower_bound(damage_cdf.begin(), damage_cdf.end(), pick) -
                    damage_cdf.begin());
                const FractureDamageSample& sample =
                    params.damage_samples[std::min(index, params.damage_samples.size() - 1u)];
                Vec3 toward_centre = hull_centre - sample.position;
                const float length = toward_centre.length();
                candidate = length > 1.0e-5f
                    ? sample.position + toward_centre * (inward / length)
                    : sample.position;
                // Jitter along the surface so several seeds on one hot patch do
                // not land on top of each other and cull each other as slivers.
                candidate += Vec3(gauss(rng), gauss(rng), gauss(rng)) * (inward * 0.5f);
            } else {
                candidate = Vec3(ux(rng), uy(rng), uz(rng));
            }
            if (insideHull(candidate)) sites.push_back(candidate);
        }
        // Impact-clustered may starve near a small hull; fall back to uniform fill.
        for (int attempt = 0; attempt < max_attempts && static_cast<int>(sites.size()) < want; ++attempt) {
            Vec3 candidate(ux(rng), uy(rng), uz(rng));
            if (insideHull(candidate)) sites.push_back(candidate);
        }
    }

    // Recorded before the degenerate bail-out below, so replaying a run that
    // produced one hull shard reproduces that too instead of re-scattering.
    stats.sites = sites;

    if (sites.size() < 2) {
        // Degenerate (tiny/thin hull): emit the whole hull as a single shard.
        if (hull_shard.volume > 0.0f) out_shards.push_back(std::move(hull_shard));
        if (out_stats) *out_stats = stats;
        return !out_shards.empty();
    }

    // The attributed source soup, built once and clipped per cell in exact mode.
    std::vector<AttrFace> exact_source;
    float weld_cell = 0.0f;
    if (params.exact_surface) {
        exact_source.reserve(source.size());
        for (const FractureInputTri& t : source) {
            Vec3 fn = (t.b - t.a).cross(t.c - t.a);
            const float len = fn.length();
            if (len <= kEps) continue;
            AttrFace f;
            f.normal = fn * (1.0f / len);
            f.material = t.material;
            f.poly = {{t.a, t.ua}, {t.b, t.ub}, {t.c, t.uc}};
            exact_source.push_back(std::move(f));
        }
        // ★ Weld tolerance scales with the OBJECT, not with the world. A fixed
        // 1e-5 m grid welds nothing on a 100 m building (its cut points land far
        // apart in float terms) and welds real detail away on a 1 cm bolt. What
        // matters is "small compared to this mesh", which is what the diagonal
        // measures.
        weld_cell = std::max((mx - mn).length() * 1.0e-5f, 1.0e-7f);
    }

    // Build each Voronoi cell = the source (or the hull) clipped by all bisectors.
    std::vector<Plane> bisectors;
    for (size_t i = 0; i < sites.size(); ++i) {
        bisectors.clear();
        for (size_t j = 0; j < sites.size(); ++j) {
            if (j == i) continue;
            Vec3 dir = sites[j] - sites[i];
            float L = dir.length();
            if (L < kEps) continue;
            dir = dir * (1.0f / L);
            Vec3 mid = (sites[i] + sites[j]) * 0.5f;
            Plane bisector;
            bisector.n = dir;             // outward = toward site j
            bisector.d = dir.dot(mid);    // keep n·x <= d → the site-i side
            bisectors.push_back(bisector);
        }

        FractureShard shard;
        if (params.exact_surface) {
            std::vector<AttrFace> cell;
            const ExactCell result =
                buildExactCell(exact_source, bisectors, weld_cell, cell);
            if (result == ExactCell::Empty) continue;
            if (result == ExactCell::Ok || result == ExactCell::Approximated) {
                triangulateAttrInto(cell, shard);
                if (result == ExactCell::Ok) ++stats.cells_exact;
                else ++stats.cells_approximated;
            } else {
                ++stats.cells_unsealed;
            }
            // Unsealed: the input was not closed here, so fall through to the
            // convex form of THIS CELL rather than emitting a shard with a hole
            // in it. Degrading one cell is a visible loss of detail; an unsealed
            // shard has no interior, so its volume, centroid and mass are all
            // wrong — and it is the mass that drives the physics.
        }
        if (shard.tris.empty()) {
            Polyhedron cell = hull_poly;
            bool alive = true;
            for (size_t b = 0; b < bisectors.size() && alive; ++b)
                alive = clipPolyhedron(cell, bisectors[b], /*cap_interior=*/true);
            if (!alive) continue;
            triangulateInto(cell, shard);
            ++stats.cells_hull;
        }
        if (shard.tris.size() < 4) continue;
        computeMassProps(shard);
        if (shard.volume <= min_volume) continue;
        out_shards.push_back(std::move(shard));
    }

    // Nothing survived (e.g. all sites coincident) → fall back to the whole hull.
    if (out_shards.empty() && hull_shard.volume > 0.0f)
        out_shards.push_back(std::move(hull_shard));

    // Partition into structural clusters LAST, on the shards that actually
    // survived the sliver cull — clustering the sites instead would leave
    // clusters that lost every one of their cells.
    assignStructuralClusters(out_shards, params.cluster_count);

    if (out_stats) *out_stats = stats;
    return !out_shards.empty();
}

} // namespace RayTrophiSim
