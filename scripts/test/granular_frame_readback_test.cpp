// User test target links the normal simulation core; no application/GPU launch.
#include "Fluid/GranularFrameReadback.h"
#include <stdexcept>
#include <type_traits>

using namespace RayTrophiSim;
namespace F = RayTrophiSim::Fluid;

static void require(bool ok, const char* message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

template<class Value>
Value pattern(int field) {
    if constexpr (std::is_same_v<Value, Vec3>) {
        return Vec3(float(field), float(field + 1), float(field + 2));
    } else {
        return static_cast<Value>(field);
    }
}

template<class Value>
bool same(const Value& a, const Value& b) {
    if constexpr (std::is_same_v<Value, Vec3>) {
        return a.x == b.x && a.y == b.y && a.z == b.z;
    } else {
        return a == b;
    }
}

int main() {
    SimulationComputeContext compute;
    F::Granular::GpuBuffers gpu;
    F::GranularFrameReadback input(2), observed(2);
    int field = 1;
    F::forEachGranularReadbackColumn(input, gpu, [&](auto& handle, auto& column) {
        using Value = typename std::decay_t<decltype(column)>::value_type;
        column.assign(2, pattern<Value>(field++));
        handle = compute.createBuffer({"granular_readback_test", column.size() * sizeof(Value),
            ComputeBufferUsage::Storage | ComputeBufferUsage::Upload | ComputeBufferUsage::Download});
        require(handle.valid(), "test allocation failed");
        require(compute.uploadBuffer(handle, column.data(), column.size() * sizeof(Value)),
            "test input upload failed");
    });
    require(field == 15, "readback field count changed: update expected contract");
    SimulationComputeContext::TransferStats transfer;
    compute.setTransferProbe(&transfer);
    compute.beginTransferBatch();
    bool ok = F::enqueueGranularSidecars(compute, gpu, observed, 2);
    ok = compute.endTransferBatch() && ok;
    require(ok, "sidecar readback failed");
    require(transfer.download_calls == 14 && transfer.batch_end_calls == 1 &&
        transfer.synchronize_calls == 0, "sidecar helper owned an extra submission boundary");
    compute.setTransferProbe(nullptr);
    F::FluidParticles particles;
    particles.emit(Vec3(1.0f), Vec3(2.0f), 293.0f, 0.0f, 17u,
        nullptr, nullptr, 0.01f, F::MatterConstitutiveModel::Fluid);
    particles.emit(Vec3(3.0f), Vec3(4.0f), 263.0f, 0.0f, 19u,
        nullptr, nullptr, 0.02f, F::MatterConstitutiveModel::Granular);
    particles.granular_damage[0] = 0.125f;
    const auto id = particles.particle_id[1];
    require(observed.canCommit(particles, 2), "valid destination refused");
    observed.commitParticle(particles, 1);
    require(particles.position[1].x == 3.0f && particles.velocity[1].x == 4.0f &&
        particles.particle_id[1] == id && particles.rest_mass_kg[1] == 0.02f &&
        particles.substance_tag[1] == 19u && particles.granular_damage[0] == 0.125f,
        "sidecar commit changed canonical transport, identity, mass or liquid state");
    field = 1;
    F::forEachGranularReadbackColumn(particles, gpu, [&](auto, const auto& column) {
        using Value = typename std::decay_t<decltype(column)>::value_type;
        require(same(column[1], pattern<Value>(field++)), "MPM sidecar field missing or misrouted");
    });
    particles.granular_damage.clear();
    require(!observed.canCommit(particles, 2), "undersized destination accepted");
    F::forEachGranularReadbackColumn(input, gpu, [&](auto handle, auto&) {
        compute.destroyBuffer(handle);
    });
}
