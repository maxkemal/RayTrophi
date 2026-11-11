// Host-only preparation test: default CPU buffer backend; no dispatch or Studio.
// User compiles with the bond/particle core, SimulationCompute and Vec3.
#include "Fluid/DemBondGpu.h"
#include "Fluid/MatterGrain.h"
#include <cassert>

using namespace RayTrophiSim;
namespace F = RayTrophiSim::Fluid;
namespace B = F::DemBond;

int main() {
    F::FluidParticles p;
    for (float x : {0.0f, 0.04f, 0.08f}) {
        p.emit(Vec3(x, 0.0f, 0.0f), Vec3(0.0f), 263.0f, 0.0f, 0,
            nullptr, nullptr, 1.0f, F::MatterConstitutiveModel::Granular);
    }
    F::MatterGrainParams params;
    B::Material material{1.0e5, 0.25, 1.0, 1.0e4, 1.0e4};
    std::string error;
    uint64_t body = 0;
    assert(B::bindBody(p, params, p.particle_id, material, 1.25, 0.5, 0, body, error));
    assert(p.dem_bonds.edges.size() == 2);
    p.dem_bonds.edges[0].state.failure = B::Failure::Tension;
    p.dem_bonds.edges[0].state.fracture_dissipation_j = 0.001;
    SimulationComputeContext compute;
    F::MatterGrainGpuRuntime runtime;
    const auto snapshot = B::encode(p.dem_bonds);
    {
        F::DemBondGpuFrame frame(compute, runtime);
        assert(frame.prepare(p, params, error));
        assert(frame.active && frame.constrains_motion);
        assert(frame.extraDispatchesPerStep() == 3);
        // Three poses, only ONE edge row, CSR headers + its two incidences, status.
        assert(frame.upload_bytes == 3 * 16 + 96 + 6 * 4 + 4);
        assert(frame.download_bytes == 3 * 16 + 96 + 4);
        assert(B::encode(p.dem_bonds) == snapshot);
    }
    p.dem_bonds.edges[1].state.failure = B::Failure::Shear;
    p.dem_bonds.edges[1].state.fracture_dissipation_j = 0.002;
    const auto broken = B::encode(p.dem_bonds);
    {
        F::DemBondGpuFrame frame(compute, runtime);
        assert(frame.prepare(p, params, error));
        assert(frame.active && !frame.constrains_motion);
        assert(frame.extraDispatchesPerStep() == 1);
        assert(frame.maximum_dt == 1.0e30);
        assert(frame.upload_bytes == 3 * 16 + 4 + 4);
        assert(frame.download_bytes == 3 * 16 + 4);
        assert(frame.before(runtime, 0, 0.01f, error)); // no dispatch, even on CPU backend
        assert(B::encode(p.dem_bonds) == broken);
        assert(!runtime.bond_buffers[0].valid()); // upload/allocation is still deferred
    }
}
