// Standalone C++17 regression for the actual CPU scalar characteristic helper.
#include "GridAdvectionTrace.h"

#include <cassert>
#include <cmath>

namespace GridFluid = RayTrophiSim::GridFluid;

struct V {
    float x, y;
    V operator-(V b) const {
        return {x - b.x, y - b.y};
    }
    V operator*(float a) const {
        return {x * a, y * a};
    }
};

float distance(V a, V b) {
    return std::hypot(a.x - b.x, a.y - b.y);
}

int main() {
    const V p{1.0f, 0.0f};
    const auto rotation = [](V v) { return V{-v.y, v.x}; };
    const auto constant = [](V) { return V{2.0f, -1.0f}; };
    const float dt = 0.2f;
    const V transported = GridFluid::traceScalarMidpoint(p, dt, constant);
    assert(distance(transported, V{0.6f, 0.2f}) < 1e-6f);
    const V exact{std::cos(dt), -std::sin(dt)};
    const V euler = p - rotation(p) * dt;
    const V midpoint = GridFluid::traceScalarMidpoint(p, dt, rotation);
    assert(distance(midpoint, exact) < 0.1f * distance(euler, exact));
    const V reverse = GridFluid::traceScalarMidpoint(midpoint, -dt, rotation);
    assert(distance(reverse, p) < 0.001f);
    assert(distance(GridFluid::traceScalarMidpoint(p, 0.0f, rotation), p) == 0.0f);
    return 0;
}
