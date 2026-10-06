#include "Fluid/MatterPoreExchange.h"
#include "Fluid/FluidParticles.h"
#include "Fluid/SubstanceTag.h"
#include "FluidGrid.h"
#include "MaterialStateField.h"
#include "SimulationCompute.h"
#include "MatterExchangeLedger.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>

namespace RayTrophiSim::Fluid {
namespace {

bool finite(const Vec3& value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

struct Inventory {
    double mass = 0.0;
    double free_water = 0.0;
    double pore = 0.0;
    double capacity = 0.0;
    double energy = 0.0;
    double momentum[3]{};
    double momentum_scale = 0.0;
};

Inventory inventory(const FluidParticles& particles, uint32_t water_tag, float cp) {
    Inventory result;
    for (std::size_t i = 0; i < particles.size(); ++i) {
        const double dry = particles.rest_mass_kg[i] * particles.mass_fraction[i];
        const double pore = particles.pore_water_mass_kg[i];
        const double total = dry + pore;
        result.mass += total;
        result.pore += pore;
        result.capacity += particles.pore_capacity_kg[i];
        result.energy += particles.pore_water_energy_j[i];
        if (particles.substance_tag[i] == water_tag) {
            result.free_water += dry;
            result.energy += dry * cp * particles.temperature[i];
        }
        const auto& v = particles.velocity[i];
        result.momentum[0] += total * v.x;
        result.momentum[1] += total * v.y;
        result.momentum[2] += total * v.z;
        result.momentum_scale += total * v.length();
    }
    return result;
}

} // namespace

uint64_t matterPoreSettingsHash(const MatterPoreParams& params) {
    uint64_t hash = 0xC6000001ull ^ (params.enabled ? 1u : 0u);
    hash = (hash ^ (params.wet_response_enabled ? 1u : 0u)) * 1099511628211ull;
    hash = (hash ^ (params.wet_appearance_enabled ? 1u : 0u)) * 1099511628211ull;
    const float values[] = {params.wet_friction_scale, params.wet_dilatancy_scale,
        params.capillary_cohesion_pa, params.pore_pressure_scale,
        params.wet_color_scale, params.wet_roughness_scale,
        params.wet_appearance_full_saturation, params.porosity,
        params.permeability_m2, params.viscosity_pa_s,
        params.gravity_m_s2, params.drainage_scale};
    for (const float value : values) {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        hash = (hash ^ bits) * 1099511628211ull;
    }
    return hash;
}

bool validateMatterPoreParams(const MatterPoreParams& params, std::string& error) {
    const auto range = [](float value, float low, float high) {
        return std::isfinite(value) && value >= low && value <= high;
    };
    if (!range(params.porosity, 0.01f, 0.8f) ||
        !range(params.permeability_m2, 0.0f, 1.0e-6f) ||
        !range(params.viscosity_pa_s, 1.0e-5f, 10.0f) ||
        !range(params.gravity_m_s2, 0.0f, 100.0f) ||
        !range(params.drainage_scale, 0.0f, 100.0f) ||
        !range(params.wet_friction_scale, 0.0f, 1.0f) ||
        !range(params.wet_dilatancy_scale, 0.0f, 1.0f) ||
        !range(params.capillary_cohesion_pa, 0.0f, 100000.0f) ||
        !range(params.pore_pressure_scale, 0.0f, 10.0f) ||
        !range(params.wet_color_scale, 0.05f, 1.0f) ||
        !range(params.wet_roughness_scale, 0.05f, 1.0f) ||
        !range(params.wet_appearance_full_saturation, 0.001f, 1.0f)) {
        error = "C5 requires finite porosity [0.01,0.8], permeability [0,1e-6] m2, "
            "viscosity [1e-5,10] Pa.s, gravity [0,100] m/s2, drainage_scale [0,100], "
            "wet friction/dilatancy [0,1], capillary cohesion [0,100000] Pa, "
            "pore pressure scale [0,10], wet color/roughness [0.05,1], "
            "full wet appearance saturation [0.001,1]";
        return false;
    }
    error.clear();
    return true;
}

bool needsMatterPoreTransport(const FluidParticles& particles, const MatterPoreParams&) {
    return std::any_of(particles.pore_water_mass_kg.begin(),
        particles.pore_water_mass_kg.end(), [](float value) { return value > 0.0f; });
}

void matterPoreTransportMasses(const FluidParticles& particles, bool legacy_granular,
                              std::vector<float>& rest, std::vector<float>& fraction) {
    rest = particles.rest_mass_kg;
    fraction = particles.mass_fraction;
    for (std::size_t i = 0; i < particles.size(); ++i) {
        const auto model = static_cast<MatterConstitutiveModel>(particles.constitutive_model[i]);
        if (model == MatterConstitutiveModel::Granular ||
            (model == MatterConstitutiveModel::Auto && legacy_granular)) {
            const float pore = i < particles.pore_water_mass_kg.size()
                ? particles.pore_water_mass_kg[i] : 0.0f;
            rest[i] = rest[i] * fraction[i] + pore;
            fraction[i] = 1.0f;
        }
    }
}

bool exchangeMatterPoresGpu(FluidParticles& particles, const FluidSim::FluidGrid& grid,
                           const MatterPoreParams& params, float dt,
                           std::size_t particle_limit, std::size_t budget_bytes,
                           SimulationComputeContext& compute, MatterPoreReport& report,
                           std::vector<MatterExchangeRecord>& events,
                           std::string& error) {
    report = {};
    events.clear();
    if (!params.enabled) {
        return true;
    }
    if (!validateMatterPoreParams(params, error)) {
        return false;
    }
    if (compute.backendType() != ComputeBackendType::VulkanCompute ||
        !compute.supportsDispatch() || !std::isfinite(dt) || dt <= 0.0f ||
        grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0 ||
        !std::isfinite(grid.voxel_size) || grid.voxel_size <= 0.0f) {
        error = "C5 requires Vulkan and a finite grid/time; no automatic CPU exchange";
        return false;
    }
    const auto count = particles.size();
    const auto cells = grid.getCellCount();
    if (!count || count > std::numeric_limits<uint32_t>::max() / 10u ||
        cells >= std::numeric_limits<uint32_t>::max() ||
        particles.rest_mass_kg.size() != count || particles.mass_fraction.size() != count ||
        particles.particle_id.size() != count ||
        particles.velocity.size() != count || particles.temperature.size() != count ||
        particles.substance_tag.size() != count || particles.constitutive_model.size() != count ||
        particles.pore_water_mass_kg.size() != count || particles.pore_capacity_kg.size() != count ||
        particles.pore_porosity.size() != count || particles.pore_water_energy_j.size() != count) {
        error = "C5 canonical sidecar cardinality is invalid";
        return false;
    }
    const auto working = (cells + 1) * 16 + count * 2048;
    if (budget_bytes && working > budget_bytes) {
        error = "C5 exchange working set exceeds the remaining domain resource budget";
        return false;
    }
    const auto* water = tryFindSubstance("Water");
    if (!water) {
        error = "C5 Water profile is unavailable";
        return false;
    }
    const auto water_tag = substanceTag("Water");
    auto candidate = particles;
    std::vector<uint32_t> models(count), birth_slots(count, 0), cell_of(count);
    std::vector<uint32_t> offsets(cells + 1, 0);
    const auto outside = std::numeric_limits<uint32_t>::max();
    std::size_t available_births = particle_limit > count ? particle_limit - count : 0;
    if (candidate.next_particle_id == 0 || available_births >=
        std::numeric_limits<uint64_t>::max() - candidate.next_particle_id) {
        error = "C5 drainage cannot reserve canonical particle identities";
        return false;
    }
    for (std::size_t i = 0; i < count; ++i) {
        const auto model = static_cast<MatterConstitutiveModel>(candidate.constitutive_model[i]);
        models[i] = static_cast<uint32_t>(model);
        const auto* profile = tryFindSubstanceByTag(candidate.substance_tag[i]);
        if (model == MatterConstitutiveModel::Auto || model == MatterConstitutiveModel::Elastic ||
            !finite(candidate.position[i]) || !finite(candidate.velocity[i]) ||
            !std::isfinite(candidate.mass_fraction[i]) || candidate.mass_fraction[i] < 0.0f ||
            candidate.mass_fraction[i] > 1.0f || !std::isfinite(candidate.rest_mass_kg[i]) ||
            candidate.rest_mass_kg[i] <= 0.0f || !std::isfinite(candidate.temperature[i]) ||
            candidate.temperature[i] <= 0.0f ||
            !std::isfinite(candidate.pore_water_mass_kg[i]) || candidate.pore_water_mass_kg[i] < 0.0f ||
            !std::isfinite(candidate.pore_water_energy_j[i]) || candidate.pore_water_energy_j[i] < 0.0f) {
            error = "C5 requires resolved mobile carriers, finite mass and Kelvin temperature";
            return false;
        }
        const bool porous = model == MatterConstitutiveModel::Granular && profile &&
            profile->default_constitutive_model == MatterConstitutiveModel::Granular;
        const float capacity = porous ? candidate.rest_mass_kg[i] * candidate.mass_fraction[i] /
            profile->density * params.porosity / (1.0f - params.porosity) * water->liquid_density : 0.0f;
        if (!std::isfinite(capacity) || capacity < candidate.pore_water_mass_kg[i] ||
            (candidate.pore_water_mass_kg[i] == 0.0f && candidate.pore_water_energy_j[i] != 0.0f)) {
            error = "C5 pore water exceeds authored capacity; drain before reducing capacity";
            return false;
        }
        candidate.pore_capacity_kg[i] = capacity;
        candidate.pore_porosity[i] = porous ? params.porosity : 0.0f;
        const auto& p = candidate.position[i];
        const double xyz[] = {(p.x - grid.origin.x) / grid.voxel_size,
            (p.y - grid.origin.y) / grid.voxel_size, (p.z - grid.origin.z) / grid.voxel_size};
        if (xyz[0] < 0 || xyz[1] < 0 || xyz[2] < 0 ||
            xyz[0] >= grid.nx || xyz[1] >= grid.ny || xyz[2] >= grid.nz) {
            cell_of[i] = outside;
            continue;
        }
        const auto cell = static_cast<uint32_t>(xyz[0]) + grid.nx *
            (static_cast<uint32_t>(xyz[1]) + grid.ny * static_cast<uint32_t>(xyz[2]));
        cell_of[i] = cell;
        ++offsets[cell + 1];
    }
    for (std::size_t cell = 0; cell < cells; ++cell) {
        if (offsets[cell + 1] > 1024u) {
            error = "C5 cell occupancy exceeds the 1024-carrier GPU execution safety limit";
            return false;
        }
        offsets[cell + 1] += offsets[cell];
    }
    std::vector<uint32_t> indices(offsets.back()), cursors = offsets;
    for (uint32_t i = 0; i < count; ++i) {
        if (cell_of[i] != outside) {
            indices[cursors[cell_of[i]]++] = i;
        }
    }
    // One output budget per cell. Refill a local Water parcel before reserving
    // a new identity; never redirect drainage to a different cell.
    const float drainage_parcel_mass = water->liquid_density * grid.voxel_size *
        grid.voxel_size * grid.voxel_size / 8.0f;
    std::vector<uint32_t> drainage_receiver(cells, outside);
    for (int pass = 0; pass < 2; ++pass) {
        for (std::size_t cell = 0; cell < cells; ++cell) {
            bool wet = false;
            bool porous = false;
            uint32_t receiver = outside;
            bool has_water = false;
            for (auto k = offsets[cell]; k < offsets[cell + 1]; ++k) {
                const auto i = indices[k];
                porous |= candidate.pore_capacity_kg[i] > 0.0f;
                wet |= candidate.pore_water_mass_kg[i] > 0.0f;
                if (models[i] == static_cast<uint32_t>(MatterConstitutiveModel::Fluid) &&
                    candidate.substance_tag[i] == water_tag &&
                    candidate.pore_water_mass_kg[i] == 0.0f) {
                    has_water |= candidate.mass_fraction[i] > 0.0f;
                    if (receiver == outside && candidate.rest_mass_kg[i] *
                        candidate.mass_fraction[i] < drainage_parcel_mass) {
                        receiver = i;
                    }
                }
            }
            if (!porous || (pass == 0 ? !wet : wet) || (!wet && !has_water)) {
                continue;
            }
            if (receiver == outside) {
                if (!available_births) {
                    continue;
                }
                --available_births;
            } else {
                drainage_receiver[cell] = receiver;
                birth_slots[receiver] = 2;
            }
            for (auto k = offsets[cell]; k < offsets[cell + 1]; ++k) {
                const auto i = indices[k];
                if (candidate.pore_capacity_kg[i] > 0.0f) {
                    birth_slots[i] = 1;
                }
            }
        }
    }
    const auto before = inventory(candidate, water_tag, water->specific_heat);
    std::vector<std::array<float, 10>> records(count);
    struct Buffers {
        SimulationComputeContext& compute;
        std::array<ComputeBufferHandle, 13> handles{};
        ~Buffers() {
            compute.synchronize();
            for (auto handle : handles) {
                if (handle.valid()) {
                    compute.destroyBuffer(handle);
                }
            }
        }
    } buffers{compute};
    const void* data[] = {offsets.data(), indices.data(), models.data(), candidate.substance_tag.data(),
        candidate.rest_mass_kg.data(), candidate.mass_fraction.data(), candidate.pore_water_mass_kg.data(),
        candidate.pore_capacity_kg.data(), candidate.pore_water_energy_j.data(), candidate.velocity.data(),
        candidate.temperature.data(), birth_slots.data(), records.data()};
    const std::size_t bytes[] = {offsets.size() * 4, indices.size() * 4, count * 4, count * 4,
        count * 4, count * 4, count * 4, count * 4, count * 4, count * sizeof(Vec3),
        count * 4, count * 4, count * sizeof(records[0])};
    for (std::size_t i = 0; i < buffers.handles.size(); ++i) {
        ComputeBufferDesc desc;
        desc.debug_name = "matter_pore_exchange";
        desc.size_bytes = std::max<std::size_t>(bytes[i], 4);
        desc.usage = ComputeBufferUsage::Storage | ComputeBufferUsage::Upload |
            ComputeBufferUsage::Download;
        buffers.handles[i] = compute.createBuffer(desc);
        if (!buffers.handles[i].valid() ||
            (bytes[i] && !compute.uploadBuffer(buffers.handles[i], data[i], bytes[i]))) {
            error = "C5 GPU allocation/upload failed";
            return false;
        }
    }
    struct Constants {
        uint32_t cells, water_tag;
        float dt, permeability, viscosity, gravity, drainage_scale, voxel, water_density, water_cp;
    } constants{static_cast<uint32_t>(cells), water_tag, dt, params.permeability_m2,
        params.viscosity_pa_s, params.gravity_m_s2, params.drainage_scale,
        grid.voxel_size, water->liquid_density, water->specific_heat};
    static_assert(sizeof(Constants) == 40);
    ComputeDispatch command;
    command.kernel = "sim_matter_pores";
    command.groups.groups_x = (constants.cells + 63u) / 64u;
    command.buffers = buffers.handles.data();
    command.buffer_count = buffers.handles.size();
    command.constants = &constants;
    command.constants_size = sizeof(constants);
    if (!compute.dispatch(command)) {
        error = "C5 Vulkan kernel dispatch failed";
        return false;
    }
    compute.beginTransferBatch();
    bool ok = compute.downloadBuffer(buffers.handles[5], candidate.mass_fraction.data(), bytes[5]);
    ok = compute.downloadBuffer(buffers.handles[6], candidate.pore_water_mass_kg.data(), bytes[6]) && ok;
    ok = compute.downloadBuffer(buffers.handles[8], candidate.pore_water_energy_j.data(), bytes[8]) && ok;
    ok = compute.downloadBuffer(buffers.handles[9], candidate.velocity.data(), bytes[9]) && ok;
    ok = compute.downloadBuffer(buffers.handles[12], records.data(), bytes[12]) && ok;
    ok = compute.endTransferBatch() && ok;
    if (!ok) {
        error = "C5 atomic GPU publication failed";
        return false;
    }
    for (std::size_t i = 0; i < count; ++i) {
        const auto& record = records[i];
        if (!std::all_of(record.begin(), record.end(), [](float value) { return std::isfinite(value); }) ||
            !finite(candidate.velocity[i]) || !std::isfinite(candidate.mass_fraction[i]) ||
            candidate.mass_fraction[i] < 0.0f || candidate.mass_fraction[i] > 1.0f ||
            !std::isfinite(candidate.pore_water_mass_kg[i]) || candidate.pore_water_mass_kg[i] < 0.0f ||
            candidate.pore_water_mass_kg[i] > candidate.pore_capacity_kg[i] * 1.00001f ||
            !std::isfinite(candidate.pore_water_energy_j[i]) || candidate.pore_water_energy_j[i] < 0.0f ||
            record[0] < 0.0f || record[5] < 0.0f) {
            error = "C5 GPU produced invalid mass/capacity/energy/velocity";
            return false;
        }
        report.absorbed_kg += record[0];
        report.absorbed_energy_j += record[1];
        report.absorbed_momentum = report.absorbed_momentum + Vec3(record[2], record[3], record[4]);
        report.drained_kg += record[5];
        if (!birth_slots[i] && candidate.pore_water_mass_kg[i] > 0.0f) {
            ++report.drainage_budget_blocked_carriers;
        }
        report.drained_energy_j += record[6];
        report.drained_momentum = report.drained_momentum + Vec3(record[7], record[8], record[9]);
        for (int transfer = 0; transfer < 2; ++transfer) {
            const int offset = transfer * 5;
            if (record[offset] <= 0.0f) {
                continue;
            }
            MatterExchangeRecord event;
            event.kind = transfer == 0 ? MatterExchangeKind::Absorption : MatterExchangeKind::Drainage;
            const auto carrier = "pore:" + std::to_string(candidate.particle_id[i]);
            event.source = transfer == 0 ? "free_water" : carrier;
            event.target = transfer == 0 ? carrier : "free_water";
            event.substance = "Water";
            event.source_mass_kg = event.target_mass_kg = record[offset];
            event.source_energy_j = event.target_energy_j = record[offset + 1];
            event.source_momentum_kg_m_s = event.target_momentum_kg_m_s =
                Vec3(record[offset + 2], record[offset + 3], record[offset + 4]);
            events.push_back(std::move(event));
        }
    }
    // GPU records are exchange results, not new per-carrier particles. Publish
    // their cell sum with mass-weighted momentum, temperature and birth position.
    for (std::size_t cell = 0; cell < cells; ++cell) {
        double mass = 0.0;
        double energy = 0.0;
        double momentum[3]{};
        double position[3]{};
        for (auto k = offsets[cell]; k < offsets[cell + 1]; ++k) {
            const auto i = indices[k];
            const auto& record = records[i];
            mass += record[5];
            energy += record[6];
            const auto& p = candidate.position[i];
            const float xyz[] = {p.x, p.y, p.z};
            for (int axis = 0; axis < 3; ++axis) {
                momentum[axis] += record[7 + axis];
                position[axis] += record[5] * static_cast<double>(xyz[axis]);
            }
        }
        if (mass <= 0.0) {
            continue;
        }
        const auto receiver = drainage_receiver[cell];
        Vec3 birth(0.0f);
        if (receiver != outside) {
            const double old_mass = candidate.rest_mass_kg[receiver] *
                candidate.mass_fraction[receiver];
            const auto& v = candidate.velocity[receiver];
            momentum[0] += old_mass * v.x;
            momentum[1] += old_mass * v.y;
            momentum[2] += old_mass * v.z;
            energy += old_mass * water->specific_heat * candidate.temperature[receiver];
            mass += old_mass;
        } else {
            birth = Vec3(static_cast<float>(position[0] / mass),
                static_cast<float>(position[1] / mass),
                static_cast<float>(position[2] / mass));
            birth.y = std::max(birth.y - grid.voxel_size * 0.5f,
                grid.origin.y + grid.voxel_size * 0.05f);
        }
        const Vec3 velocity(static_cast<float>(momentum[0] / mass),
            static_cast<float>(momentum[1] / mass), static_cast<float>(momentum[2] / mass));
        const float temperature = static_cast<float>(energy / (mass * water->specific_heat));
        if (!finite(birth) || !finite(velocity) || !std::isfinite(temperature) ||
            temperature <= 0.0f || !std::isfinite(static_cast<float>(mass))) {
            error = "C5 drainage batch has invalid state";
            return false;
        }
        if (receiver != outside) {
            candidate.rest_mass_kg[receiver] = static_cast<float>(mass);
            candidate.mass_fraction[receiver] = 1.0f;
            candidate.velocity[receiver] = velocity;
            candidate.temperature[receiver] = temperature;
            ++report.drainage_refills;
        } else {
            if (candidate.size() >= particle_limit) {
                error = "C5 drainage batch exceeds particle capacity";
                return false;
            }
            candidate.emit(birth, velocity, temperature, 0.0f, water_tag,
                nullptr, nullptr, static_cast<float>(mass), MatterConstitutiveModel::Fluid);
            ++report.drainage_births;
        }
    }
    const auto after = inventory(candidate, water_tag, water->specific_heat);
    report.mass_error_kg = after.mass - before.mass;
    double squared = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
        const auto difference = after.momentum[axis] - before.momentum[axis];
        squared += difference * difference;
    }
    report.momentum_error_kg_m_s = std::sqrt(squared);
    report.thermal_energy_error_j = after.energy - before.energy;
    if (std::abs(report.mass_error_kg) > 1e-5 * std::max(before.mass, 1.0) ||
        report.momentum_error_kg_m_s > 2e-5 * std::max(before.momentum_scale, 1.0) ||
        std::abs(report.thermal_energy_error_j) > 5e-5 * std::max(before.energy, 1.0)) {
        error = "C5 conservation gate rejected GPU exchange";
        return false;
    }
    report.free_water_kg = after.free_water;
    report.pore_water_kg = after.pore;
    report.capacity_kg = after.capacity;
    report.budget_bytes = working;
    report.measured = true;
    report.status = "Vulkan C5 cell exchange";
    particles = std::move(candidate);
    return true;
}

} // namespace RayTrophiSim::Fluid
