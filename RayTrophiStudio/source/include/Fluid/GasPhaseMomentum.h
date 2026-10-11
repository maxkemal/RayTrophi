#pragma once
#include "Vec3.h"
#include <cstddef>

namespace RayTrophiSim {
struct SimulationGridDomainDesc;
struct SimulationGridDomainState;
struct SimulationGridDomainComputeBuffers;
struct MaterialTemperatureScale;
class SimulationComputeContext;
namespace Fluid {
struct GasPhaseMomentumDeposit {
    float mass = 0;
    Vec3 momentum;
};
void queueGasPhaseMomentum(SimulationGridDomainState& gas, std::size_t cell,
                           float mass, const Vec3& momentum);
bool flushGasPhaseMomentum(SimulationGridDomainState& gas,
                           const SimulationGridDomainDesc& domain,
                           const MaterialTemperatureScale& scale,
                           float ambient_kelvin, float gas_ambient_normalized,
                           SimulationComputeContext* compute = nullptr,
                           SimulationGridDomainComputeBuffers* buffers = nullptr);
}
}
