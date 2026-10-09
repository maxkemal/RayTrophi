#include "Fluid/MatterGrainFluidGpuCoupling.h"
#include "ParticleSimulation.h"
#include "Fluid/FluidPhysicalMass.h"
#include "Fluid/FluidDomainSubstance.h"
#include "MaterialStateField.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_map>

namespace RayTrophiSim::Fluid {
namespace {
uint32_t bits(float value) {
    uint32_t result;
    std::memcpy(&result, &value, sizeof(value));
    return result;
}
} // namespace

void releaseMatterGrainFluidGpuStorage(SimulationComputeContext& compute,
                                     MatterGrainFluidGpuStorage& storage) {
    for (auto& handle : storage.handles) {
        if (handle.valid()) {
            compute.destroyBuffer(handle);
        }
        handle = {};
    }
}

MatterGrainFluidGpuCoupling::MatterGrainFluidGpuCoupling(SimulationComputeContext& compute,
    MatterGrainFluidGpuStorage* storage) : compute_(compute), storage_(storage) {
    if (storage_) {
        buffers_ = storage_->handles;
        storage_->handles = {};
    }
}

MatterGrainFluidGpuCoupling::~MatterGrainFluidGpuCoupling() {
    for (std::size_t i = 3; i < buffers_.size(); ++i) {
        if ((i < 13 || (i > 15 && i < 20)) && buffers_[i].valid()) {
            if (storage_) {
                storage_->handles[i] = buffers_[i];
            } else {
                compute_.destroyBuffer(buffers_[i]);
            }
        }
    }
}

bool MatterGrainFluidGpuCoupling::prepare(const FluidParticles& continuum,
    const MatterGrainCouplingFrame& frame, SimulationGridDomainComputeBuffers& buffers,
    MatterGrainGpuRuntime& grains, std::size_t budget_bytes, std::string& error,
    const APICSolverParams& params, const MatterGrainMotion& motion) {
    using Row = std::array<uint32_t, 4>;
    const auto grain_count = frame.inputs.size();
    std::vector<Row> parcels;
    const auto* fallback = domainSubstance(params);
    for (std::size_t p = 0; p < continuum.size(); ++p) {
        if (continuum.constitutive_model[p] !=
            static_cast<uint8_t>(MatterConstitutiveModel::Fluid)) {
            continue;
        }
        const float mass = continuum.rest_mass_kg[p] * continuum.mass_fraction[p];
        if (!(mass > 0.0f)) {
            continue;
        }
        const auto tag = continuum.substance_tag[p];
        const float density = fluidParticleRestMassKg(tag, fallback, 1.0f, 1,
            MatterConstitutiveModel::Fluid, false);
        const auto* profile = resolveFluidSubstanceProfile(tag, fallback);
        const float viscosity = profile ? profile->liquid_kinematic_viscosity *
            profile->liquid_density : 1.0e-3f;
        if (!std::isfinite(mass) || !std::isfinite(density) || density <= 0.0f ||
            !std::isfinite(viscosity)) {
            error = "dynamic liquid coupling received invalid parcel material/mass";
            return false;
        }
        parcels.push_back({static_cast<uint32_t>(p), bits(mass), bits(density),
            bits(viscosity > 0.0f ? viscosity : 1.0e-3f)});
    }
    if (parcels.empty()) {
        for (std::size_t i = 3; i < buffers_.size(); ++i) {
            if ((i < 13 || (i > 15 && i < 20)) && buffers_[i].valid()) {
                compute_.destroyBuffer(buffers_[i]);
                buffers_[i] = {};
            }
        }
        return true;
    }
    const auto count = parcels.size() + grain_count;
    const auto index_limit = std::numeric_limits<uint32_t>::max() - 255u;
    if (count > index_limit / 4 || grain_count * 8 > index_limit ||
        !(frame.field.h > 0.0f) || !grain_count) {
        error = "dynamic liquid coupling layout exceeds the shader index width";
        return false;
    }
    uint32_t buckets = 1;
    while (std::size_t(buckets) < count * 4 && buckets < (uint32_t{1} << 30)) {
        buckets *= 2;
    }
    counts_ = {static_cast<uint32_t>(parcels.size()), buckets,
        static_cast<uint32_t>(grain_count), 0};
    const auto& field = frame.field;
    origin_h_ = {field.origin.x, field.origin.y, field.origin.z, field.h};
    material_ = {params.grain.radius_m, 4.188790205f * params.grain.radius_m *
        params.grain.radius_m * params.grain.radius_m, 0.35f, 0.0f};
    dimensions_ = {uint32_t(field.nx), uint32_t(field.ny), uint32_t(field.nz), 0};
    gravity_ = {params.gravity.x, params.gravity.y, params.gravity.z, 0.0f};
    const std::array<std::size_t, 21> sizes{
        0, 0, 0, parcels.size() * sizeof(Row), 2 * std::size_t(buckets) * sizeof(uint32_t),
        count * 2 * sizeof(uint32_t), grain_count * 8 * 2 * sizeof(uint32_t),
        count * 16, count * 16, grain_count * 16, grain_count * 16,
        parcels.size() * 16, grain_count * 16, 0, 0, 0,
        count * 16, grain_count * 16, grain_count * 16, grain_count * 16, 0};
    std::size_t retained_bytes = 0;
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        const auto size = sizes[i];
        bytes_ += size;
        retained_bytes += std::max(size, compute_.getBufferSize(buffers_[i]));
        if (compute_.caps().max_storage_buffer_bytes &&
            size > compute_.caps().max_storage_buffer_bytes) {
            error = "dynamic liquid coupling exceeds the device buffer capacity";
            return false;
        }
    }
    if (budget_bytes && bytes_ > budget_bytes) {
        error = "dynamic liquid coupling exceeds the authored domain resource budget";
        return false;
    }
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        if (!sizes[i]) {
            continue;
        }
        const auto current = compute_.getBufferSize(buffers_[i]);
        const bool trim = budget_bytes && retained_bytes > budget_bytes && current > sizes[i];
        if (!trim && current >= sizes[i]) {
            continue;
        }
        if (trim) {
            compute_.destroyBuffer(buffers_[i]);
            buffers_[i] = {};
            retained_bytes -= current - sizes[i];
        }
        ComputeBufferDesc desc;
        desc.debug_name = "matter_dynamic_liquid_coupling";
        desc.size_bytes = sizes[i];
        desc.usage = ComputeBufferUsage::Storage | ComputeBufferUsage::ReadWrite |
            ComputeBufferUsage::Upload | ComputeBufferUsage::Download;
        auto candidate = compute_.createBuffer(desc);
        if (!candidate.valid()) {
            error = "dynamic liquid coupling allocation failed";
            return false;
        }
        if (buffers_[i].valid()) {
            compute_.destroyBuffer(buffers_[i]);
        }
        buffers_[i] = candidate;
    }
    bytes_ = 0;
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        if (sizes[i]) {
            bytes_ += compute_.getBufferSize(buffers_[i]);
        }
    }
    std::vector<std::array<float, 4>> external(grain_count);
    for (std::size_t g = 0; g < motion.external_acceleration.size(); ++g) {
        const auto& a = motion.external_acceleration[g];
        external[g] = {a.x, a.y, a.z, 0.0f};
    }
    if (!compute_.uploadBuffer(buffers_[3], parcels.data(), sizes[3]) ||
        !compute_.uploadBuffer(buffers_[18], external.data(), sizes[18])) {
        error = "dynamic liquid coupling metadata upload failed";
        return false;
    }
    upload_bytes_ = sizes[3] + sizes[18];
    buffers_[0] = buffers.fluid_velocities;
    buffers_[1] = grains.coupling;
    buffers_[2] = grains.mass;
    buffers_[13] = buffers.fluid_positions;
    buffers_[14] = grains.positions;
    buffers_[15] = grains.scratch;
    buffers_[20] = grains.velocities;
    return true;
}

bool MatterGrainFluidGpuCoupling::dispatch(const char* kernel, uint32_t threads,
    uint32_t index, float dt, std::string& error) {
    if (!counts_[0]) {
        return true;
    }
    struct Constants {
        std::array<uint32_t, 4> counts;
        std::array<float, 4> step;
        std::array<float, 4> origin_h;
        std::array<float, 4> material;
        std::array<uint32_t, 4> dimensions;
        std::array<float, 4> gravity;
    } constants{counts_, {dt, 0.0f, 0.0f, 0.0f}, origin_h_, material_, dimensions_, gravity_};
    static_assert(sizeof(Constants) == 96);
    constants.counts[3] = index;
    ComputeDispatch command;
    command.kernel = kernel;
    command.buffers = buffers_.data();
    command.buffer_count = buffers_.size();
    command.constants = &constants;
    command.constants_size = sizeof(constants);
    const uint32_t groups = (threads + 255u) / 256u;
    command.groups.groups_x = std::min(groups, 65535u);
    command.groups.groups_y = (groups + command.groups.groups_x - 1u) /
        command.groups.groups_x;
    ++dispatches_;
    if (!compute_.dispatch(command)) {
        error = std::string("common-clock liquid coupling dispatch failed: ") + kernel;
        return false;
    }
    return true;
}

bool MatterGrainFluidGpuCoupling::refresh(uint32_t index, float dt, std::string& error) {
    const auto clear_threads = index == 0 ? 2 * counts_[1] : counts_[0] + counts_[2];
    return dispatch("sim_grain_fluid_clear", clear_threads, index, dt, error) &&
        dispatch("sim_grain_fluid_hash", counts_[0] + counts_[2], index, dt, error) &&
        dispatch("sim_grain_fluid_cells", counts_[0] + counts_[2], index, dt, error) &&
        dispatch("sim_grain_fluid_solid", counts_[2], index, dt, error) &&
        dispatch("sim_grain_fluid_refresh", counts_[2], index, dt, error);
}

bool MatterGrainFluidGpuCoupling::react(uint32_t index, float dt, std::string& error) {
    return dispatch("sim_grain_fluid_delta", counts_[2], index, dt, error) &&
        dispatch("sim_grain_fluid_reaction", counts_[2], index, dt, error) &&
        dispatch("sim_grain_fluid_apply", counts_[0], index, dt, error);
}

bool MatterGrainFluidGpuCoupling::publish(const std::vector<MatterGrainCouplingOutput>& drag,
    MatterGrainStepReport& report, std::string& error) {
    report.liquid_reaction_on_gpu = true;
    report.liquid_support_dynamic = counts_[0] > 0;
    if (!counts_[0]) {
        return true;
    }
    std::vector<std::array<float, 4>> impulses(counts_[0]);
    std::vector<std::array<float, 4>> gains(counts_[2]), metrics(counts_[2]);
    compute_.beginTransferBatch();
    bool ok = compute_.downloadBuffer(buffers_[11], impulses.data(), impulses.size() * 16);
    ok = compute_.downloadBuffer(buffers_[17], gains.data(), gains.size() * 16) && ok;
    ok = compute_.downloadBuffer(buffers_[19], metrics.data(), metrics.size() * 16) && ok;
    ok = compute_.endTransferBatch() && ok;
    if (!ok) {
        error = "dynamic liquid impulse publication failed";
        return false;
    }
    double reaction[3]{};
    for (const auto& impulse : impulses) {
        for (int axis = 0; axis < 3; ++axis) {
            if (!std::isfinite(impulse[axis])) {
                error = "common-clock liquid reaction is nonfinite";
                return false;
            }
            reaction[axis] += impulse[axis];
        }
    }
    report.drag_impulse = Vec3(0.0f);
    for (const auto& grain : drag) {
        report.drag_impulse = report.drag_impulse + grain.drag_impulse;
    }
    report.liquid_reaction = Vec3(float(reaction[0]), float(reaction[1]), float(reaction[2]));
    double total_gain[3]{};
    report.coupled_grains = 0;
    report.max_drag_coefficient = report.max_submerged_fraction = 0.0f;
    for (std::size_t g = 0; g < gains.size(); ++g) {
        for (int axis = 0; axis < 3; ++axis) {
            if (!std::isfinite(gains[g][axis])) {
                error = "dynamic liquid grain impulse is nonfinite";
                return false;
            }
            total_gain[axis] += gains[g][axis];
        }
        report.coupled_grains += metrics[g][2] > 0.0f ? 1 : 0;
        report.max_submerged_fraction = std::max(report.max_submerged_fraction, metrics[g][0]);
        report.max_drag_coefficient = std::max(report.max_drag_coefficient, metrics[g][1]);
    }
    const Vec3 gain{float(total_gain[0]), float(total_gain[1]), float(total_gain[2])};
    report.buoyancy_impulse = gain - report.drag_impulse;
    report.momentum_residual = std::sqrt(
        (reaction[0] + gain.x) * (reaction[0] + gain.x) +
        (reaction[1] + gain.y) * (reaction[1] + gain.y) +
        (reaction[2] + gain.z) * (reaction[2] + gain.z));
    report.dispatches += dispatches_;
    report.working_set_bytes += bytes_;
    report.upload_bytes += upload_bytes_;
    report.download_bytes += (impulses.size() + gains.size() + metrics.size()) * 16;
    ++report.transfer_batches;
    return true;
}

} // namespace RayTrophiSim::Fluid
