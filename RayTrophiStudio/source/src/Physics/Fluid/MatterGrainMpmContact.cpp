#include "Fluid/MatterGrainMpmContact.h"
#include "Fluid/MatterWetResponse.h"
#include "ParticleSimulation.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace RayTrophiSim::Fluid {
namespace {

struct Metadata {
    float mass;
    float radius;
    uint32_t index;
    uint32_t reserved = 0;
};
static_assert(sizeof(Metadata) == 16);

bool finite(const Vec3& value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

} // namespace

MatterGrainMpmContact::MatterGrainMpmContact(SimulationComputeContext& compute)
    : compute_(compute) {}

MatterGrainMpmContact::~MatterGrainMpmContact() {
    for (std::size_t i = 6; i < buffers_.size(); ++i) {
        if (buffers_[i].valid()) {
            compute_.destroyBuffer(buffers_[i]);
        }
    }
}

bool MatterGrainMpmContact::prepare(FluidParticles& continuum,
    SimulationGridDomainComputeBuffers& buffers, std::size_t grain_count,
    float grain_radius, float friction, float frame_dt, std::size_t budget_bytes,
    std::string& error) {
    std::vector<Metadata> metadata;
    float maximum_radius = 0.0f;
    for (std::size_t i = 0; i < continuum.size(); ++i) {
        if (continuum.constitutive_model[i] !=
            static_cast<uint8_t>(MatterConstitutiveModel::Granular)) {
            continue;
        }
        const float mass = continuum.rest_mass_kg[i] * continuum.mass_fraction[i] +
            continuum.pore_water_mass_kg[i];
        const float volume = matterParticleDryVolume(continuum, i);
        // Circumscribed rest-volume cell: a continuum support, not a DEM grain.
        const float radius = 0.866025404f * std::cbrt(volume);
        if (!std::isfinite(mass) || mass <= 0.0f || !std::isfinite(radius) ||
            radius <= 0.0f || !finite(continuum.velocity[i])) {
            error = "MPM/grain contact requires finite positive carrier mass and volume";
            return false;
        }
        indices_.push_back(static_cast<uint32_t>(i));
        masses_.push_back(mass);
        metadata.push_back({mass, radius, static_cast<uint32_t>(i)});
        maximum_radius = std::max(maximum_radius, radius);
        maximum_speed_ = std::max(maximum_speed_, continuum.velocity[i].length());
    }
    if (indices_.empty()) {
        return true;
    }
    const auto count = grain_count + indices_.size();
    if (!grain_count || count > std::numeric_limits<uint32_t>::max() - 255u ||
        !buffers.fluid_positions.valid() ||
        !buffers.fluid_velocities.valid() || !(frame_dt > 0.0f)) {
        error = "MPM/grain contact device view or carrier count is invalid";
        return false;
    }
    uint32_t buckets = 1;
    // Hash load factor is a performance choice, not a particle count ceiling.
    // Two head arrays must remain addressable by the shader's uint index.
    while (std::size_t(buckets) < count * 4 && buckets < (uint32_t{1} << 30)) {
        buckets *= 2;
    }
    const std::array<std::size_t, 7> sizes{
        metadata.size() * sizeof(Metadata), 2 * buckets * sizeof(uint32_t),
        count * sizeof(uint32_t), count * 4 * sizeof(float),
        count * 4 * sizeof(float), 8 * sizeof(uint32_t), count * sizeof(uint32_t)};
    for (const auto bytes : sizes) {
        working_set_bytes_ += bytes;
        if (compute_.caps().max_storage_buffer_bytes &&
            bytes > compute_.caps().max_storage_buffer_bytes) {
            error = "MPM/grain contact exceeds the device buffer limit";
            return false;
        }
    }
    if (budget_bytes && working_set_bytes_ >= budget_bytes) {
        error = "MPM/grain contact exceeds the remaining domain resource budget";
        return false;
    }
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        ComputeBufferDesc desc;
        desc.debug_name = "matter_grain_mpm_contact";
        desc.size_bytes = sizes[i];
        desc.usage = ComputeBufferUsage::Storage | ComputeBufferUsage::ReadWrite |
            ComputeBufferUsage::Upload | ComputeBufferUsage::Download;
        buffers_[6 + i] = compute_.createBuffer(desc);
        if (!buffers_[6 + i].valid()) {
            error = "MPM/grain contact allocation failed";
            return false;
        }
    }
    if (!compute_.uploadBuffer(buffers_[6], metadata.data(), sizes[0])) {
        error = "MPM/grain contact metadata upload failed";
        return false;
    }
    continuum_ = &continuum;
    buffers_[4] = buffers.fluid_positions;
    buffers_[5] = buffers.fluid_velocities;
    meta_ = {static_cast<uint32_t>(grain_count), static_cast<uint32_t>(indices_.size()),
             buckets, 0};
    params_ = {grain_radius, grain_radius + maximum_radius, friction, frame_dt};
    return true;
}

bool MatterGrainMpmContact::step(MatterGrainGpuRuntime& grains, uint32_t substep,
                               std::string& error) {
    if (!active()) {
        return true;
    }
    buffers_[0] = grains.positions;
    buffers_[1] = grains.velocities;
    buffers_[2] = grains.scratch;
    buffers_[3] = grains.mass;
    meta_[3] = substep & 1u;
    struct Constants {
        std::array<uint32_t, 4> meta;
        std::array<float, 4> params;
    } constants{meta_, params_};
    static_assert(sizeof(Constants) == 32);
    const uint32_t count = meta_[0] + meta_[1];
    const auto dispatch = [&](const char* kernel, uint32_t threads) {
        ComputeDispatch command;
        command.kernel = kernel;
        command.buffers = buffers_.data();
        command.buffer_count = static_cast<uint32_t>(buffers_.size());
        command.constants = &constants;
        command.constants_size = sizeof(constants);
        const uint32_t groups = (threads + 255u) / 256u;
        command.groups.groups_x = std::min(groups, 65535u);
        command.groups.groups_y = (groups + command.groups.groups_x - 1u) /
            command.groups.groups_x;
        ++dispatches_;
        if (!compute_.dispatch(command)) {
            error = std::string("MPM/grain contact dispatch failed: ") + kernel;
            return false;
        }
        return true;
    };
    return (substep != 0 || dispatch("sim_grain_mpm_init", count)) &&
        dispatch("sim_grain_mpm_clear", 2 * meta_[2]) &&
        dispatch("sim_grain_mpm_hash", count) &&
        dispatch("sim_grain_mpm_count", count) &&
        dispatch("sim_grain_mpm_gather", count) &&
        dispatch("sim_grain_mpm_apply", count);
}

bool MatterGrainMpmContact::publish(MatterGrainStepReport& report, std::string& error,
                                  bool apply_host_reaction) {
    if (!active()) {
        return true;
    }
    std::vector<std::array<float, 4>> impulses(meta_[0] + meta_[1]);
    std::array<uint32_t, 8> diagnostics{};
    compute_.beginTransferBatch();
    bool ok = compute_.downloadBuffer(buffers_[10], impulses.data(),
        impulses.size() * sizeof(impulses[0]));
    ok = compute_.downloadBuffer(buffers_[11], diagnostics.data(), sizeof(diagnostics)) && ok;
    ok = compute_.endTransferBatch() && ok;
    if (!ok || diagnostics[0] || diagnostics[1] != 2u) {
        error = "MPM/grain contact readback/revision failed; compile simulation shaders";
        return false;
    }
    std::array<double, 3> residual{};
    double contact_impulse = 0.0;
    for (std::size_t i = 0; i < impulses.size(); ++i) {
        const Vec3 impulse(impulses[i][0], impulses[i][1], impulses[i][2]);
        if (!finite(impulse)) {
            error = "MPM/grain contact produced a nonfinite impulse";
            return false;
        }
        for (int axis = 0; axis < 3; ++axis) {
            residual[axis] += impulses[i][axis];
        }
        if (i < meta_[0]) {
            contact_impulse += impulse.length();
        }
    }
    // Stage all updates before mutating the owner snapshot; no partial publication.
    if (apply_host_reaction) {
        std::vector<Vec3> velocities(indices_.size());
        for (std::size_t i = 0; i < indices_.size(); ++i) {
            const auto& impulse = impulses[meta_[0] + i];
            velocities[i] = continuum_->velocity[indices_[i]] +
                Vec3(impulse[0], impulse[1], impulse[2]) / masses_[i];
            if (!finite(velocities[i])) {
                error = "MPM/grain contact produced a nonfinite MPM velocity";
                return false;
            }
            const Vec3 actual = (velocities[i] - continuum_->velocity[indices_[i]]) * masses_[i];
            residual[0] += double(actual.x) - impulse[0];
            residual[1] += double(actual.y) - impulse[1];
            residual[2] += double(actual.z) - impulse[2];
        }
        for (std::size_t i = 0; i < indices_.size(); ++i) {
            continuum_->velocity[indices_[i]] = velocities[i];
        }
    }
    report.mpm_parcels = indices_.size();
    report.mpm_contact_events = uint64_t(diagnostics[2]) | (uint64_t(diagnostics[4]) << 32);
    report.mpm_contact_max_neighbours = diagnostics[3];
    report.mpm_contact_impulse = contact_impulse;
    report.mpm_contact_momentum_residual = std::sqrt(residual[0] * residual[0] +
        residual[1] * residual[1] + residual[2] * residual[2]);
    report.dispatches += dispatches_;
    report.working_set_bytes += working_set_bytes_;
    report.upload_bytes += indices_.size() * sizeof(Metadata);
    report.download_bytes += impulses.size() * sizeof(impulses[0]) + sizeof(diagnostics);
    ++report.transfer_batches;
    return true;
}

} // namespace RayTrophiSim::Fluid
