#pragma once

namespace RayTrophiSim {
namespace GridFluid {

// Signed dt: positive backtraces the material, negative is the reverse trace
// used by MacCormack. The midpoint uses the same frozen velocity field.
template <typename Vector, typename SampleVelocity>
Vector traceScalarMidpoint(const Vector& position, float dt, SampleVelocity sampleVelocity) {
    const Vector midpoint = position - sampleVelocity(position) * (0.5f * dt);
    return position - sampleVelocity(midpoint) * dt;
}

} // namespace GridFluid
} // namespace RayTrophiSim
