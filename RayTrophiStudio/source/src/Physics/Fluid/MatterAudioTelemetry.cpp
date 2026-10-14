#include "Fluid/MatterAudioTelemetry.h"
#include "Fluid/MatterSubstanceState.h"
#include "Fluid/FluidPhysicalMass.h"
#include "Fluid/MatterPhaseGrid.h"
#include "ParticleSimulation.h"
#include "MaterialStateField.h"
#include <algorithm>
#include <cmath>
#include <exception>

namespace RayTrophiSim::Fluid {
namespace {
bool finite(const Vec3& value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

MatterTransportOwner owner(const FluidParticles& particles, std::size_t i,
                          const APICSolverParams& params) {
    const auto model = i < particles.constitutive_model.size()
        ? static_cast<MatterConstitutiveModel>(particles.constitutive_model[i])
        : MatterConstitutiveModel::Auto;
    return substanceTransportOwner(i < particles.substance_tag.size()
        ? particles.substance_tag[i] : 0u, model, params);
}

struct Sum {
    double value = 0.0;
    double x = 0.0, y = 0.0, z = 0.0;
    std::uint64_t carriers = 0;
    bool valid = true;

    void add(double contribution, const Vec3& position) {
        if (!std::isfinite(contribution) || contribution < 0 || !finite(position)) {
            valid = false;
            return;
        }
        value += contribution;
        x += contribution * position.x;
        y += contribution * position.y;
        z += contribution * position.z;
        ++carriers;
    }

    DomainAudioMeasurement finish(double duration, bool physicalPower = true) const {
        DomainAudioMeasurement result;
        result.physical_power = physicalPower;
        if (!valid || !std::isfinite(value) || value > 1e9 || duration <= 0 ||
            !std::isfinite(duration)) {
            result.reason = "invalid_measurement";
            return result;
        }
        if (carriers == 0) {
            result.reason = "no_carriers";
            return result;
        }
        result.measured = true;
        result.value = value;
        // End-state power/rate samples are not integrated work over the step.
        result.energy_j = 0;
        if (!std::isfinite(result.energy_j) || result.energy_j > 1e9) {
            return {};
        }
        result.duration_seconds = duration;
        result.position_weight = value;
        result.carriers = carriers;
        result.reason = "available";
        if (value > 0) {
            result.position = Vec3(static_cast<float>(x / value), static_cast<float>(y / value),
                                   static_cast<float>(z / value));
            result.measured = finite(result.position);
        }
        return result;
    }
};
}

MatterAudioFrame::MatterAudioFrame(SimulationGridDomainState& state,
                                 const SimulationGridDomainDesc& domain, double duration)
    : state_(state), domain_(domain), duration_(duration) {
    const auto publication = std::max(state_.version, state_.audio_stats.publication_version) + 1;
    state_.audio_stats = {};
    state_.audio_stats.publication_version = publication;
    if (!domain_.fluid_params.grain.audio_telemetry_enabled) {
        state_.audio_stats.fluid_flow.reason = "telemetry_disabled";
        state_.audio_stats.gas_flow.reason = "telemetry_disabled";
        state_.audio_stats.mpm_deformation.reason = "telemetry_disabled";
        state_.audio_stats.mpm_fracture.reason = "telemetry_disabled";
        return;
    }
    const auto& particles = state_.particles;
    try {
        for (std::size_t i = 0; i < particles.size(); ++i) {
            if (owner(particles, i, domain_.fluid_params) == MatterTransportOwner::Mpm &&
                i < particles.particle_id.size() && i < particles.granular_damage.size()) {
                damage_[particles.particle_id[i]] = particles.granular_damage[i];
            }
        }
    } catch (const std::exception&) {
        damage_.clear();
        damage_valid_ = false;
    }
}

void MatterAudioFrame::publishLiquid() {
    if (!domain_.fluid_params.grain.audio_telemetry_enabled) {
        return;
    }
    if (state_.fluid_stats.mixed_step_held || !std::isfinite(duration_) || duration_ <= 0) {
        return;
    }
    Sum flow, deformation, fracture;
    fracture.valid = damage_valid_;
    double mpmMass = 0.0;
    const auto& particles = state_.particles;
    const auto* fallback = tryFindSubstance(domain_.fluid_params.default_substance);
    const double length = std::max(static_cast<double>(state_.grid.voxel_size), 1e-6);
    for (std::size_t i = 0; i < particles.size(); ++i) {
        const auto transport = owner(particles, i, domain_.fluid_params);
        if (transport != MatterTransportOwner::Fluid && transport != MatterTransportOwner::Mpm) {
            continue;
        }
        if (i >= particles.rest_mass_kg.size() || i >= particles.mass_fraction.size() ||
            i >= particles.velocity.size()) {
            flow.valid = deformation.valid = fracture.valid = false;
            continue;
        }
        const double mass = particles.rest_mass_kg[i] * particles.mass_fraction[i];
        if (!(mass > 0) || !std::isfinite(mass)) {
            continue;
        }
        const auto& position = particles.position[i];
        if (transport == MatterTransportOwner::Fluid) {
            const double speed2 = particles.velocity[i].dot(particles.velocity[i]);
            // Gross kinetic energy transported over one voxel scale. This is
            // a physical flow activity metric, not a claim of acoustic watts.
            flow.add(0.5 * mass * speed2 * std::sqrt(speed2) / length, position);
            continue;
        }
        const auto* substance = resolveFluidSubstanceProfile(
            i < particles.substance_tag.size() ? particles.substance_tag[i] : 0u, fallback);
        if (!substance || !(substance->density > 0) || i >= particles.affine.size() ||
            i >= particles.granular_stress_diag.size() ||
            i >= particles.granular_stress_shear.size()) {
            deformation.valid = fracture.valid = false;
            continue;
        }
        const auto& affine = particles.affine[i];
        const auto& diagonal = particles.granular_stress_diag[i];
        const auto& shear = particles.granular_stress_shear[i];
        if (!finite(affine.col0) || !finite(affine.col1) || !finite(affine.col2) ||
            !finite(diagonal) || !finite(shear) || i >= particles.granular_deformation_col0.size() ||
            i >= particles.granular_deformation_col1.size() ||
            i >= particles.granular_deformation_col2.size()) {
            deformation.valid = false;
            continue;
        }
        const double jacobian = particles.granular_deformation_col0[i].dot(
            particles.granular_deformation_col1[i].cross(particles.granular_deformation_col2[i]));
        if (!std::isfinite(jacobian) || jacobian <= 0) {
            deformation.valid = false;
            continue;
        }
        const double stressPower = diagonal.x * affine.col0.x + diagonal.y * affine.col1.y +
            diagonal.z * affine.col2.z + shear.x * (affine.col0.y + affine.col1.x) +
            shear.y * (affine.col0.z + affine.col2.x) +
            shear.z * (affine.col1.z + affine.col2.y);
        deformation.add(std::max(0.0, stressPower) * mass * jacobian / substance->density, position);
        const auto previous = i < particles.particle_id.size()
            ? damage_.find(particles.particle_id[i]) : damage_.end();
        const double before = previous != damage_.end() ? previous->second : 0;
        const double after = i < particles.granular_damage.size()
            ? particles.granular_damage[i] : before;
        if (!std::isfinite(before) || !std::isfinite(after) || after < 0 || after > 1) {
            fracture.valid = false;
        } else {
            fracture.add(mass * std::max(0.0, after - before) / duration_, position);
            mpmMass += mass;
        }
    }
    state_.audio_stats.fluid_flow = flow.finish(duration_);
    state_.audio_stats.mpm_deformation = deformation.finish(duration_);
    auto& broken = state_.audio_stats.mpm_fracture;
    broken = fracture.finish(duration_, false);
    // Mass-weighted damage fraction per second, not fracture energy.
    if (broken.measured && mpmMass > 0) {
        broken.value /= mpmMass;
    }
}

void MatterAudioFrame::publishGas(SimulationComputeContext* compute,
                                 SimulationGridDomainComputeBuffers* buffers) {
    if (!domain_.fluid_params.grain.audio_telemetry_enabled) {
        return;
    }
    auto& measurement = state_.audio_stats.gas_flow;
    if (!state_.gas_stats.stepped || !state_.gas_inventory_error.empty()) {
        return;
    }
    if (compute && buffers && state_.gas_inventory_revision > 0) {
        try {
            if (!measureGasAudioGpu(state_, *compute, *buffers, duration_, measurement)) {
                measurement.reason = "device_metric_unavailable";
            }
        } catch (const std::exception&) {
            measurement = {};
            measurement.reason = "device_metric_unavailable";
        }
        return;
    }
    const auto& grid = gasGrid(state_);
    Sum flow;
    if (grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0 || grid.voxel_size <= 0) {
        return;
    }
    const std::size_t cells = grid.density.size();
    if (state_.gas_phase_mass_kg.size() != cells || state_.gas_phase_foreign_kg.size() != cells) {
        measurement.reason = "physical_mass_unavailable";
        return;
    }
    for (std::size_t cell = 0; cell < cells; ++cell) {
        const int x = static_cast<int>(cell % grid.nx);
        const int y = static_cast<int>((cell / grid.nx) % grid.ny);
        const int z = static_cast<int>(cell / (grid.nx * grid.ny));
        const Vec3 position = grid.origin + Vec3(x + 0.5f, y + 0.5f, z + 0.5f) * grid.voxel_size;
        const auto velocity = grid.sampleVelocity(position);
        const double mass = state_.gas_phase_mass_kg[cell] + state_.gas_phase_foreign_kg[cell];
        if (!std::isfinite(mass) || mass < 0) {
            flow.valid = false;
            continue;
        }
        if (mass > 0) {
            const double speed2 = velocity.dot(velocity);
            flow.add(0.5 * mass * speed2 * std::sqrt(speed2) / grid.voxel_size, position);
        }
    }
    measurement = flow.finish(duration_);
}

} // namespace RayTrophiSim::Fluid
