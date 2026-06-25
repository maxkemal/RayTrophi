/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          FractureGenerator.cpp
 * Author:        Kemal Demirtas
 * License:       MIT
 * =========================================================================
 *
 * Convex Voronoi fracture (see FractureGenerator.h).
 *
 * Pipeline:
 *   1. Convex hull of the source vertices (incremental, orientation-by-reference
 *      so winding bugs can't flip a face inward).
 *   2. Scatter `site_count` sites inside the hull (uniform or impact-clustered).
 *   3. For each site, clip the hull polyhedron by every bisector half-space
 *      against the other sites — the surviving convex polytope is that site's
 *      Voronoi cell ∩ hull.
 *   4. Triangulate each cell into a shard; cull slivers.
 */

#include "FractureGenerator.h"

#include <algorithm>
#include <array>
#include <cmath>
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

} // namespace

bool generateConvexFracture(const std::vector<FractureInputTri>& source,
                            const FractureParams& params,
                            std::vector<FractureShard>& out_shards) {
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

    // Scatter sites inside the hull.
    const int want = std::max(1, params.site_count);
    std::vector<Vec3> sites;
    sites.reserve(want);
    std::mt19937 rng(params.seed ? params.seed : 1u);
    std::uniform_real_distribution<float> ux(mn.x, mx.x), uy(mn.y, mx.y), uz(mn.z, mx.z);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    const int max_attempts = want * 200 + 1000;
    for (int attempt = 0; attempt < max_attempts && static_cast<int>(sites.size()) < want; ++attempt) {
        Vec3 candidate;
        if (params.pattern == FracturePattern::ImpactClustered) {
            float r = std::max(1.0e-4f, params.impact_radius);
            candidate = params.impact_point + Vec3(gauss(rng), gauss(rng), gauss(rng)) * r;
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
    if (sites.size() < 2) {
        // Degenerate (tiny/thin hull): emit the whole hull as a single shard.
        if (hull_shard.volume > 0.0f) out_shards.push_back(std::move(hull_shard));
        return !out_shards.empty();
    }

    // Build each Voronoi cell = hull clipped by all bisector half-spaces.
    for (size_t i = 0; i < sites.size(); ++i) {
        Polyhedron cell = hull_poly;
        bool alive = true;
        for (size_t j = 0; j < sites.size() && alive; ++j) {
            if (j == i) continue;
            Vec3 dir = sites[j] - sites[i];
            float L = dir.length();
            if (L < kEps) continue;
            dir = dir * (1.0f / L);
            Vec3 mid = (sites[i] + sites[j]) * 0.5f;
            Plane bisector;
            bisector.n = dir;             // outward = toward site j
            bisector.d = dir.dot(mid);    // keep n·x <= d → the site-i side
            alive = clipPolyhedron(cell, bisector, /*cap_interior=*/true);
        }
        if (!alive) continue;

        FractureShard shard;
        triangulateInto(cell, shard);
        if (shard.tris.size() < 4) continue;
        computeMassProps(shard);
        if (shard.volume <= min_volume) continue;
        out_shards.push_back(std::move(shard));
    }

    // Nothing survived (e.g. all sites coincident) → fall back to the whole hull.
    if (out_shards.empty() && hull_shard.volume > 0.0f)
        out_shards.push_back(std::move(hull_shard));

    return !out_shards.empty();
}

} // namespace RayTrophiSim
