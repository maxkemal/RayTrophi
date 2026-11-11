// Invoke from the user's Vulkan test fixture with an initialized compute context.
// Does not access the live scene or advance Studio's timeline.
#include "Fluid/DemBondGpu.h"
#include "Fluid/MatterGrain.h"
#include <cassert>
#include <cmath>

using namespace RayTrophiSim;
namespace F = RayTrophiSim::Fluid;
namespace B = F::DemBond;

static void testPartialRemap(SimulationComputeContext& compute) {
    F::FluidParticles p;
    for (float x : {0.0f, 0.04f, 0.08f}) {
        p.emit(Vec3(x, 0.0f, 0.0f), Vec3(0.0f), 263.0f, 0.0f, 0,
            nullptr, nullptr, 1.0f, F::MatterConstitutiveModel::Granular);
    }
    F::MatterGrainParams params;
    B::Material material{1.0e5, 0.25, 1.0, 1.0e9, 1.0e9};
    std::string error;
    uint64_t body = 0;
    assert(B::bindBody(p, params, p.particle_id, material, 1.25, 0.5, 0, body, error));
    p.dem_bonds.edges[0].state.failure = B::Failure::Tension;
    p.dem_bonds.edges[0].state.fracture_dissipation_j = 0.123456789;
    const auto history = B::encode(p.dem_bonds).at("edges").at(0);
    p.position[2].x += 0.0004f;
    F::MatterGrainGpuRuntime runtime;
    const auto buffer = [&](std::size_t bytes) {
        auto handle = compute.createBuffer({"partial_bond_test", bytes,
            ComputeBufferUsage::Storage | ComputeBufferUsage::Upload | ComputeBufferUsage::Download});
        assert(handle.valid());
        return handle;
    };
    runtime.positions = buffer(3 * sizeof(Vec3));
    runtime.velocities = buffer(3 * sizeof(Vec3));
    runtime.affines = buffer(3 * sizeof(F::AffineC));
    runtime.scratch = buffer(27 * sizeof(float));
    runtime.mass = buffer(6 * sizeof(float));
    const float masses[]{1.0f, 1.0f, 1.0f, params.radius_m, params.radius_m, params.radius_m};
    assert(compute.uploadBuffer(runtime.positions, p.position.data(), 3 * sizeof(Vec3)));
    assert(compute.uploadBuffer(runtime.velocities, p.velocity.data(), 3 * sizeof(Vec3)));
    assert(compute.uploadBuffer(runtime.affines, p.affine.data(), 3 * sizeof(F::AffineC)));
    assert(compute.uploadBuffer(runtime.mass, masses, sizeof(masses)));
    {
        F::DemBondGpuFrame frame(compute, runtime);
        assert(frame.prepare(p, params, error));
        const float dt = static_cast<float>(std::min(1.0e-5, frame.maximum_dt * 0.1));
        assert(frame.before(runtime, 0, dt, error));
        assert(frame.readback(error));
        assert(frame.publish(p, error));
        assert(B::encode(p.dem_bonds).at("edges").at(0) == history);
        assert(p.dem_bonds.edges[1].elastic_energy_j > 0.0);
        Vec3 velocity[3];
        assert(compute.downloadBuffer(runtime.velocities, velocity, sizeof(velocity)));
        assert(velocity[0].length_squared() == 0.0f);
        assert(velocity[1].x > 0.0f && velocity[2].x < 0.0f);
        assert(std::abs(velocity[1].x + velocity[2].x) < 1.0e-6);
    }
    F::releaseMatterGrainGpu(compute, runtime);
}

void testDemBondGpuReference(SimulationComputeContext& compute) {
    assert(compute.backendType() == ComputeBackendType::VulkanCompute);
    testPartialRemap(compute);
    for (bool fracture : {false, true}) {
        F::FluidParticles p;
        for (float x : {0.0f, 0.04f}) {
            p.emit(Vec3(x, 0.0f, 0.0f), Vec3(0.0f), 263.0f, 0.0f, 0,
                nullptr, nullptr, 1.0f, F::MatterConstitutiveModel::Granular);
        }
        F::MatterGrainParams params;
        B::Material material{1.0e5, 0.25, 1.0, fracture ? 1.0 : 1.0e9, 1.0e9};
        uint64_t body = 0;
        std::string error;
        assert(B::bindBody(p, params, p.particle_id, material, 1.25, 0.5, 0, body, error));
        p.position[1] += Vec3(0.0004f, 0.0002f, 0.0f);
        auto edge = p.dem_bonds.edges.front();
        B::Endpoint a{p.particle_id[0], {p.position[0].x, p.position[0].y, p.position[0].z}};
        B::Endpoint b{p.particle_id[1], {p.position[1].x, p.position[1].y, p.position[1].z}};
        B::Response expected;
        assert(B::evaluate(a, b, edge.material, edge.state, expected, error));
        F::MatterGrainGpuRuntime runtime;
        const auto buffer = [&](const char* name, std::size_t bytes) {
            auto handle = compute.createBuffer({name, bytes, ComputeBufferUsage::Storage |
                ComputeBufferUsage::Upload | ComputeBufferUsage::Download});
            assert(handle.valid());
            return handle;
        };
        runtime.positions = buffer("bond_test_positions", 2 * sizeof(Vec3));
        runtime.velocities = buffer("bond_test_velocities", 2 * sizeof(Vec3));
        runtime.affines = buffer("bond_test_affines", 2 * sizeof(F::AffineC));
        runtime.scratch = buffer("bond_test_scratch", 18 * sizeof(float));
        runtime.mass = buffer("bond_test_mass", 4 * sizeof(float));
        const float masses[]{1.0f, 1.0f, params.radius_m, params.radius_m};
        const float zero[18]{};
        assert(compute.uploadBuffer(runtime.positions, p.position.data(), 2 * sizeof(Vec3)));
        assert(compute.uploadBuffer(runtime.velocities, p.velocity.data(), 2 * sizeof(Vec3)));
        assert(compute.uploadBuffer(runtime.affines, p.affine.data(), 2 * sizeof(F::AffineC)));
        assert(compute.uploadBuffer(runtime.scratch, zero, sizeof(zero)));
        assert(compute.uploadBuffer(runtime.mass, masses, sizeof(masses)));
        {
            F::DemBondGpuFrame frame(compute, runtime);
            assert(frame.prepare(p, params, error));
            const float dt = static_cast<float>(std::min(1.0e-5, frame.maximum_dt * 0.1));
            assert(frame.before(runtime, 0, dt, error));
            float spins[18]{};
            spins[14] = spins[17] = 2.0f; // destination bank: both omega_z = 2 rad/s
            assert(compute.uploadBuffer(runtime.scratch, spins, sizeof(spins)));
            assert(frame.after(runtime, 0, dt, error));
            assert(frame.readback(error));
            Vec3 velocity[2];
            assert(compute.downloadBuffer(runtime.velocities, velocity, sizeof(velocity)));
            for (int axis = 0; axis < 3; ++axis) {
                assert(std::abs(velocity[0][axis] - dt * expected.first_force_n[axis]) < 1.0e-5);
                assert(std::abs(velocity[0][axis] + velocity[1][axis]) < 1.0e-6);
            }
            assert(frame.publish(p, error));
            const auto& pose = p.dem_bonds.nodes.at(p.particle_id[0]).orientation;
            assert(std::abs(pose[0] - std::cos(dt)) < 1.0e-6);
            assert(std::abs(pose[3] - std::sin(dt)) < 1.0e-6);
            const auto& actual = p.dem_bonds.edges.front();
            assert(actual.state.failure == edge.state.failure);
            const double reference_energy = expected.elastic_energy_j + expected.released_energy_j;
            const double device_energy = actual.elastic_energy_j + actual.state.fracture_dissipation_j;
            assert(std::abs(reference_energy - device_energy) < 1.0e-4 * (1.0 + reference_energy));
            if (fracture) {
                const auto history = B::encode(p.dem_bonds).at("edges");
                F::DemBondGpuFrame retired(compute, runtime);
                assert(retired.prepare(p, params, error));
                assert(retired.active && !retired.constrains_motion);
                assert(retired.extraDispatchesPerStep() == 1);
                assert(retired.upload_bytes == 40 && retired.download_bytes == 36);
                assert(retired.before(runtime, 0, dt, error));
                assert(retired.after(runtime, 0, dt, error));
                assert(retired.readback(error));
                assert(retired.publish(p, error));
                assert(B::encode(p.dem_bonds).at("edges") == history);
                const auto& turned = p.dem_bonds.nodes.at(p.particle_id[0]).orientation;
                assert(std::abs(turned[0] - std::cos(2.0 * dt)) < 1.0e-6);
                assert(std::abs(turned[3] - std::sin(2.0 * dt)) < 1.0e-6);
                Vec3 unchanged[2];
                assert(compute.downloadBuffer(runtime.velocities, unchanged, sizeof(unchanged)));
                for (int i = 0; i < 2; ++i) {
                    for (int axis = 0; axis < 3; ++axis) {
                        assert(unchanged[i][axis] == velocity[i][axis]);
                    }
                }
            }
        }
        F::releaseMatterGrainGpu(compute, runtime);
    }
}
