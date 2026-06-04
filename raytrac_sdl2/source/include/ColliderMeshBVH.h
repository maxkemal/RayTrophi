/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          ColliderMeshBVH.h
* Purpose:       Lightweight, dependency-free median-split triangle BVH used to
*                bake collider signed-distance fields. Self-contained (no Embree/
*                OptiX) so it runs on the async SDF cook worker. Provides:
*                  - closestDistanceSquared(): nearest-point unsigned distance,
*                    accelerated with AABB pruning (replaces the old brute-force
*                    N^3 x triangle loop AND its >5000-triangle stride decimation).
*                  - countRayHits(): triangle crossings along a ray, for robust
*                    inside/outside sign via parity voting (replaces the fragile
*                    single-nearest-triangle normal dot that flipped on edges).
* =========================================================================
*/
#pragma once

#include "Vec3.h"

#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

class ColliderMeshBVH {
public:
    struct Triangle { Vec3 a, b, c; };

    void build(std::vector<Triangle> tris) {
        tris_ = std::move(tris);
        nodes_.clear();
        order_.clear();
        centroids_.clear();
        tri_min_.clear();
        tri_max_.clear();
        const int n = static_cast<int>(tris_.size());
        if (n == 0) return;

        order_.resize(n);
        centroids_.resize(n);
        tri_min_.resize(n);
        tri_max_.resize(n);
        for (int i = 0; i < n; ++i) {
            order_[i] = i;
            const Triangle& t = tris_[i];
            tri_min_[i] = Vec3::min(Vec3::min(t.a, t.b), t.c);
            tri_max_[i] = Vec3::max(Vec3::max(t.a, t.b), t.c);
            centroids_[i] = (t.a + t.b + t.c) * (1.0f / 3.0f);
        }
        nodes_.reserve(static_cast<size_t>(n) * 2);
        buildNode(0, n, 0);
    }

    bool empty() const { return tris_.empty(); }

    // Squared distance from p to the nearest point on the mesh; out_closest set.
    float closestDistanceSquared(const Vec3& p, Vec3& out_closest) const {
        out_closest = p;
        if (nodes_.empty()) return std::numeric_limits<float>::max();
        float best = std::numeric_limits<float>::max();
        int stack[96];
        int sp = 0;
        stack[sp++] = 0;
        while (sp > 0) {
            const Node& nd = nodes_[stack[--sp]];
            if (aabbDistSq(p, nd.bmin, nd.bmax) >= best) continue;
            if (nd.count > 0) { // leaf
                for (int k = 0; k < nd.count; ++k) {
                    const Triangle& t = tris_[order_[nd.start + k]];
                    Vec3 cl;
                    const float d2 = pointTriDistSq(p, t.a, t.b, t.c, cl);
                    if (d2 < best) { best = d2; out_closest = cl; }
                }
            } else if (sp + 2 <= 96) {
                stack[sp++] = nd.left;
                stack[sp++] = nd.right;
            }
        }
        return best;
    }

    // Count triangles crossed by the ray (origin + t*dir, t > eps). Two-sided
    // (no backface cull) so the parity is well defined for inside/outside tests.
    int countRayHits(const Vec3& origin, const Vec3& dir) const {
        if (nodes_.empty()) return 0;
        const Vec3 inv(safeInv(dir.x), safeInv(dir.y), safeInv(dir.z));
        int count = 0;
        int stack[96];
        int sp = 0;
        stack[sp++] = 0;
        while (sp > 0) {
            const Node& nd = nodes_[stack[--sp]];
            if (!rayAABB(origin, inv, nd.bmin, nd.bmax)) continue;
            if (nd.count > 0) {
                for (int k = 0; k < nd.count; ++k) {
                    const Triangle& t = tris_[order_[nd.start + k]];
                    float tt;
                    if (rayTri(origin, dir, t.a, t.b, t.c, tt) && tt > 1e-6f) ++count;
                }
            } else if (sp + 2 <= 96) {
                stack[sp++] = nd.left;
                stack[sp++] = nd.right;
            }
        }
        return count;
    }

private:
    struct Node {
        Vec3 bmin{1e30f, 1e30f, 1e30f};
        Vec3 bmax{-1e30f, -1e30f, -1e30f};
        int start = 0;
        int count = 0;     // > 0 → leaf range into order_; 0 → internal
        int left = -1;
        int right = -1;
    };

    std::vector<Triangle> tris_;
    std::vector<int> order_;
    std::vector<Vec3> centroids_, tri_min_, tri_max_;
    std::vector<Node> nodes_;

    static float axisVal(const Vec3& v, int axis) {
        return axis == 0 ? v.x : (axis == 1 ? v.y : v.z);
    }
    static float safeInv(float x) {
        return (std::fabs(x) > 1e-12f) ? (1.0f / x) : (x >= 0.0f ? 1e12f : -1e12f);
    }

    // Build a node over order_[start, start+count). Returns its index. Uses an
    // index (not a Node&) across the recursive push_backs so vector reallocation
    // can't dangle.
    int buildNode(int start, int count, int depth) {
        const int idx = static_cast<int>(nodes_.size());
        nodes_.push_back(Node{});

        Vec3 bmin(1e30f, 1e30f, 1e30f), bmax(-1e30f, -1e30f, -1e30f);
        Vec3 cmin(1e30f, 1e30f, 1e30f), cmax(-1e30f, -1e30f, -1e30f);
        for (int i = 0; i < count; ++i) {
            const int ti = order_[start + i];
            bmin = Vec3::min(bmin, tri_min_[ti]);
            bmax = Vec3::max(bmax, tri_max_[ti]);
            cmin = Vec3::min(cmin, centroids_[ti]);
            cmax = Vec3::max(cmax, centroids_[ti]);
        }

        if (count <= 4 || depth > 48) {
            nodes_[idx].bmin = bmin;
            nodes_[idx].bmax = bmax;
            nodes_[idx].start = start;
            nodes_[idx].count = count;
            return idx;
        }

        const Vec3 ext = cmax - cmin;
        const int axis = (ext.x >= ext.y && ext.x >= ext.z) ? 0 : (ext.y >= ext.z ? 1 : 2);
        const float split = axisVal(cmin, axis) + axisVal(ext, axis) * 0.5f;

        int mid = partitionByAxis(start, count, axis, split);
        if (mid == start || mid == start + count) mid = start + count / 2; // degenerate split → median

        const int l = buildNode(start, mid - start, depth + 1);
        const int r = buildNode(mid, start + count - mid, depth + 1);
        nodes_[idx].bmin = bmin;
        nodes_[idx].bmax = bmax;
        nodes_[idx].count = 0;
        nodes_[idx].left = l;
        nodes_[idx].right = r;
        return idx;
    }

    // In-place partition of order_[start, start+count) by centroid axis < split.
    int partitionByAxis(int start, int count, int axis, float split) {
        int i = start;
        int j = start + count - 1;
        while (i <= j) {
            while (i <= j && axisVal(centroids_[order_[i]], axis) < split) ++i;
            while (i <= j && axisVal(centroids_[order_[j]], axis) >= split) --j;
            if (i < j) std::swap(order_[i], order_[j]);
        }
        return i;
    }

    static float aabbDistSq(const Vec3& p, const Vec3& bmin, const Vec3& bmax) {
        float d = 0.0f;
        const float dx = p.x < bmin.x ? bmin.x - p.x : (p.x > bmax.x ? p.x - bmax.x : 0.0f);
        const float dy = p.y < bmin.y ? bmin.y - p.y : (p.y > bmax.y ? p.y - bmax.y : 0.0f);
        const float dz = p.z < bmin.z ? bmin.z - p.z : (p.z > bmax.z ? p.z - bmax.z : 0.0f);
        d = dx * dx + dy * dy + dz * dz;
        return d;
    }

    static bool rayAABB(const Vec3& o, const Vec3& inv, const Vec3& bmin, const Vec3& bmax) {
        float t1 = (bmin.x - o.x) * inv.x, t2 = (bmax.x - o.x) * inv.x;
        float tmin = std::min(t1, t2), tmax = std::max(t1, t2);
        t1 = (bmin.y - o.y) * inv.y; t2 = (bmax.y - o.y) * inv.y;
        tmin = std::max(tmin, std::min(t1, t2)); tmax = std::min(tmax, std::max(t1, t2));
        t1 = (bmin.z - o.z) * inv.z; t2 = (bmax.z - o.z) * inv.z;
        tmin = std::max(tmin, std::min(t1, t2)); tmax = std::min(tmax, std::max(t1, t2));
        return tmax >= std::max(tmin, 0.0f);
    }

    // Möller–Trumbore, two-sided. Returns hit + ray parameter t.
    static bool rayTri(const Vec3& o, const Vec3& d, const Vec3& a, const Vec3& b, const Vec3& c, float& t) {
        const Vec3 e1 = b - a, e2 = c - a;
        const Vec3 pv = Vec3::cross(d, e2);
        const float det = Vec3::dot(e1, pv);
        if (std::fabs(det) < 1e-12f) return false;
        const float invDet = 1.0f / det;
        const Vec3 tv = o - a;
        const float u = Vec3::dot(tv, pv) * invDet;
        if (u < 0.0f || u > 1.0f) return false;
        const Vec3 qv = Vec3::cross(tv, e1);
        const float v = Vec3::dot(d, qv) * invDet;
        if (v < 0.0f || u + v > 1.0f) return false;
        t = Vec3::dot(e2, qv) * invDet;
        return true;
    }

    // Closest point on triangle (Ericson, Real-Time Collision Detection §5.1.5);
    // returns squared distance and the closest point.
    static float pointTriDistSq(const Vec3& p, const Vec3& a, const Vec3& b, const Vec3& c, Vec3& out) {
        const Vec3 ab = b - a, ac = c - a, ap = p - a;
        const float d1 = Vec3::dot(ab, ap), d2 = Vec3::dot(ac, ap);
        if (d1 <= 0.0f && d2 <= 0.0f) { out = a; return (p - a).length_squared(); }
        const Vec3 bp = p - b;
        const float d3 = Vec3::dot(ab, bp), d4 = Vec3::dot(ac, bp);
        if (d3 >= 0.0f && d4 <= d3) { out = b; return (p - b).length_squared(); }
        const float vc = d1 * d4 - d3 * d2;
        if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
            const float v = d1 / (d1 - d3);
            out = a + ab * v; return (p - out).length_squared();
        }
        const Vec3 cp = p - c;
        const float d5 = Vec3::dot(ab, cp), d6 = Vec3::dot(ac, cp);
        if (d6 >= 0.0f && d5 <= d6) { out = c; return (p - c).length_squared(); }
        const float vb = d5 * d2 - d1 * d6;
        if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
            const float w = d2 / (d2 - d6);
            out = a + ac * w; return (p - out).length_squared();
        }
        const float va = d3 * d6 - d5 * d4;
        if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
            const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            out = b + (c - b) * w; return (p - out).length_squared();
        }
        const float denom = 1.0f / (va + vb + vc);
        const float v = vb * denom, w = vc * denom;
        out = a + ab * v + ac * w;
        return (p - out).length_squared();
    }
};
