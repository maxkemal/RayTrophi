// Liquid inflow / outflow regions. See MatterLiquidInflow.h.
#include "Fluid/MatterLiquidInflow.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace RayTrophiSim::Fluid {

namespace {

float dot3(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

float axisOf(const Vec3& v, int a) { return a == 0 ? v.x : (a == 1 ? v.y : v.z); }

uint64_t mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

float unit01(uint64_t x) {
    return static_cast<float>((mix64(x) >> 40) & 0xFFFFFFull) / 16777216.0f;
}

// Is some point p + axis * s with s in [0, sweep] inside the box?
bool sweptBoxContains(const LiquidFlowRegion& r, const Vec3& p) {
    float s_lo = 0.0f;
    float s_hi = std::max(0.0f, r.sweep);
    for (int a = 0; a < 3; ++a) {
        const float pa = axisOf(p, a);
        const float na = axisOf(r.axis, a);
        const float lo = axisOf(r.box_min, a);
        const float hi = axisOf(r.box_max, a);
        if (std::fabs(na) < 1.0e-7f) {
            if (pa < lo || pa > hi) return false;
            continue;
        }
        float t0 = (lo - pa) / na;
        float t1 = (hi - pa) / na;
        if (t0 > t1) std::swap(t0, t1);
        s_lo = std::max(s_lo, t0);
        s_hi = std::min(s_hi, t1);
        if (s_lo > s_hi) return false;
    }
    return true;
}

} // namespace

bool insideLiquidFlowRegion(const LiquidFlowRegion& r, const Vec3& p) {
    switch (r.shape) {
    case LiquidFlowRegion::Shape::Sphere: {
        const Vec3 d = p - r.center;
        return dot3(d, d) <= r.radius * r.radius;
    }
    case LiquidFlowRegion::Shape::Box:
        return sweptBoxContains(r, p);
    case LiquidFlowRegion::Shape::Disk:
    default: {
        // Upstream distance of p from the face plane, then its radial offset.
        const Vec3 d = p - r.center;
        const float s = -dot3(d, r.axis);
        if (s < 0.0f || s > r.sweep) return false;
        const Vec3 radial = d + r.axis * s;
        return dot3(radial, radial) <= r.radius * r.radius;
    }
    }
}

void liquidFlowRegionBounds(const LiquidFlowRegion& r, Vec3& lo, Vec3& hi) {
    switch (r.shape) {
    case LiquidFlowRegion::Shape::Sphere:
        lo = r.center - Vec3(r.radius);
        hi = r.center + Vec3(r.radius);
        return;
    case LiquidFlowRegion::Shape::Box: {
        const Vec3 back = r.axis * (-r.sweep);
        lo = Vec3::min(r.box_min, r.box_min + back);
        hi = Vec3::max(r.box_max, r.box_max + back);
        return;
    }
    case LiquidFlowRegion::Shape::Disk:
    default: {
        const Vec3 tail = r.center - r.axis * r.sweep;
        lo = Vec3::min(r.center, tail) - Vec3(r.radius);
        hi = Vec3::max(r.center, tail) + Vec3(r.radius);
        return;
    }
    }
}

float liquidFlowRegionArea(const LiquidFlowRegion& r) {
    constexpr float kPi = 3.14159265358979f;
    switch (r.shape) {
    case LiquidFlowRegion::Shape::Box: {
        const Vec3 e = r.box_max - r.box_min;
        return std::fabs(r.axis.x) * e.y * e.z + std::fabs(r.axis.y) * e.x * e.z +
               std::fabs(r.axis.z) * e.x * e.y;
    }
    case LiquidFlowRegion::Shape::Sphere:
    case LiquidFlowRegion::Shape::Disk:
    default:
        return kPi * r.radius * r.radius;
    }
}

void planLiquidInflow(const LiquidFlowRegion& region, const LiquidLattice& lattice,
                      int particles_per_cell, const FluidParticles& particles,
                      uint64_t seed, LiquidInflowPlan& out) {
    out.births.clear();
    out.held.clear();
    out.region_parcels = 0;
    const float h = lattice.h;
    const int ppc = std::max(1, particles_per_cell);
    if (!(h > 0.0f) || lattice.nx <= 0 || lattice.ny <= 0 || lattice.nz <= 0) return;

    Vec3 lo, hi;
    liquidFlowRegionBounds(region, lo, hi);
    const auto cellOf = [&](float v, float o, int n) {
        return std::clamp(static_cast<int>(std::floor((v - o) / h)), 0, n - 1);
    };
    const Vec3 grid_hi = lattice.origin + Vec3(static_cast<float>(lattice.nx),
        static_cast<float>(lattice.ny), static_cast<float>(lattice.nz)) * h;
    if (hi.x < lattice.origin.x || hi.y < lattice.origin.y || hi.z < lattice.origin.z ||
        lo.x >= grid_hi.x || lo.y >= grid_hi.y || lo.z >= grid_hi.z) {
        return;
    }
    const int i0 = cellOf(lo.x, lattice.origin.x, lattice.nx);
    const int i1 = cellOf(hi.x, lattice.origin.x, lattice.nx);
    const int j0 = cellOf(lo.y, lattice.origin.y, lattice.ny);
    const int j1 = cellOf(hi.y, lattice.origin.y, lattice.ny);
    const int k0 = cellOf(lo.z, lattice.origin.z, lattice.nz);
    const int k1 = cellOf(hi.z, lattice.origin.z, lattice.nz);
    const int rx = i1 - i0 + 1, ry = j1 - j0 + 1, rz = k1 - k0 + 1;
    std::vector<uint32_t> existing(static_cast<std::size_t>(rx) * ry * rz, 0u);
    const auto local = [&](int i, int j, int k) {
        return static_cast<std::size_t>(i - i0) +
               static_cast<std::size_t>(j - j0) * rx +
               static_cast<std::size_t>(k - k0) * rx * ry;
    };

    // Parcels of ANY substance occupy the region: a nozzle half full of
    // something else pours only into the rest.
    const std::size_t n = particles.position.size();
    for (std::size_t p = 0; p < n; ++p) {
        const Vec3& x = particles.position[p];
        if (x.x < lo.x || x.y < lo.y || x.z < lo.z || x.x > hi.x || x.y > hi.y || x.z > hi.z) {
            continue;
        }
        if (!insideLiquidFlowRegion(region, x)) continue;
        out.held.push_back(p);
        const int i = static_cast<int>(std::floor((x.x - lattice.origin.x) / h));
        const int j = static_cast<int>(std::floor((x.y - lattice.origin.y) / h));
        const int k = static_cast<int>(std::floor((x.z - lattice.origin.z) / h));
        if (i < i0 || i > i1 || j < j0 || j > j1 || k < k0 || k > k1) continue;
        ++existing[local(i, j, k)];
    }

    // Jittered strata: k^3 >= ppc sub-cells, ppc of them spread evenly.
    int k = 1;
    while (k * k * k < ppc) ++k;
    const int strata = k * k * k;
    std::vector<Vec3> inside;
    inside.reserve(static_cast<std::size_t>(ppc));
    for (int cz = k0; cz <= k1; ++cz) {
        for (int cy = j0; cy <= j1; ++cy) {
            for (int cx = i0; cx <= i1; ++cx) {
                inside.clear();
                const uint64_t cell_key = seed ^
                    (static_cast<uint64_t>(cx) * 0x9e3779b1ull) ^
                    (static_cast<uint64_t>(cy) * 0x85ebca77ull << 16) ^
                    (static_cast<uint64_t>(cz) * 0xc2b2ae3dull << 32);
                for (int s = 0; s < ppc; ++s) {
                    const int sub = static_cast<int>((static_cast<long long>(s) * strata) / ppc);
                    const int sx = sub % k;
                    const int sy = (sub / k) % k;
                    const int sz = sub / (k * k);
                    const uint64_t key = mix64(cell_key + static_cast<uint64_t>(s));
                    const Vec3 q = lattice.origin + Vec3(
                        (static_cast<float>(cx) + (static_cast<float>(sx) + unit01(key)) / k) * h,
                        (static_cast<float>(cy) + (static_cast<float>(sy) + unit01(key ^ 0x51ull)) / k) * h,
                        (static_cast<float>(cz) + (static_cast<float>(sz) + unit01(key ^ 0xa3ull)) / k) * h);
                    if (insideLiquidFlowRegion(region, q)) inside.push_back(q);
                }
                out.region_parcels += inside.size();
                const uint32_t have = existing[local(cx, cy, cz)];
                if (inside.size() <= have) continue;
                // Which samples stay empty rotates per step, so a partly full
                // cell is not always topped up at the same corner.
                const std::size_t need = inside.size() - have;
                const std::size_t start = static_cast<std::size_t>(mix64(cell_key) % inside.size());
                for (std::size_t m = 0; m < need; ++m) {
                    out.births.push_back(inside[(start + m) % inside.size()]);
                }
            }
        }
    }
}

void limitLiquidInflowBirths(LiquidInflowPlan& plan, double requested_parcels,
                            bool& initialized, double& fractional_parcel,
                            uint64_t seed) {
    if (!std::isfinite(requested_parcels) || requested_parcels <= 0.0) {
        plan.births.clear();
        return;
    }
    if (!initialized) {
        initialized = true;
        fractional_parcel = 0.0;
        return;
    }
    const double carry = std::isfinite(fractional_parcel)
        ? std::clamp(fractional_parcel, 0.0, 1.0) : 0.0;
    const double exact = requested_parcels + carry;
    const double whole = std::floor(exact);
    fractional_parcel = exact - whole;
    const std::size_t available = plan.births.size();
    if (whole >= static_cast<double>(available)) {
        return;
    }
    const auto allowed = static_cast<std::size_t>(whole);
    if (allowed == 0) {
        plan.births.clear();
        return;
    }
    const auto start = static_cast<std::size_t>(mix64(seed) % available);
    std::vector<Vec3> selected;
    selected.reserve(allowed);
    for (std::size_t i = 0; i < allowed; ++i) {
        const auto offset = static_cast<std::size_t>(
            static_cast<double>(i) * available / allowed);
        selected.push_back(plan.births[(start + offset) % available]);
    }
    plan.births = std::move(selected);
}

std::size_t markLiquidOutflow(const LiquidFlowRegion& region,
                              const FluidParticles& particles,
                              std::vector<uint8_t>& remove_mask) {
    const std::size_t n = particles.position.size();
    remove_mask.assign(n, 0u);
    Vec3 lo, hi;
    liquidFlowRegionBounds(region, lo, hi);
    std::size_t marked = 0;
    for (std::size_t p = 0; p < n; ++p) {
        const Vec3& x = particles.position[p];
        if (x.x < lo.x || x.y < lo.y || x.z < lo.z || x.x > hi.x || x.y > hi.y || x.z > hi.z) {
            continue;
        }
        if (!insideLiquidFlowRegion(region, x)) continue;
        remove_mask[p] = 1u;
        ++marked;
    }
    return marked;
}

} // namespace RayTrophiSim::Fluid
