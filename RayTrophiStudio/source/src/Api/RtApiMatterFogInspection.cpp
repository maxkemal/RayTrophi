#include "RtApiInternal.h"
#include "Api/MatterPhaseOutputApi.h"
#include "Fluid/FluidFogDensity.h"
#include "Fluid/MatterFogFields.h"
#include "Fluid/MatterFogSnapshot.h"
#include "Fluid/MatterFogStreams.h"
#include "Fluid/MatterPhaseGrid.h"
#include "MaterialStateField.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <utility>

namespace rtapi {
namespace {
using namespace RayTrophiSim::Fluid;

const char* phaseName(MatterOutputPhase phase) {
    switch (phase) {
        case MatterOutputPhase::Solid: return "solid";
        case MatterOutputPhase::Liquid: return "liquid";
        default: return "unknown";
    }
}

nlohmann::json fieldStats(const MatterFogField& field) {
    double density_sum = 0.0;
    std::size_t active = 0;
    std::size_t temperature_samples = 0;
    float minimum = std::numeric_limits<float>::max();
    float maximum = 0.0f;
    for (std::size_t i = 0; i < field.density.size(); ++i) {
        const float density = field.density[i];
        density_sum += density;
        if (density <= 0.0f) {
            continue;
        }
        ++active;
        if (i < field.temperature_kelvin.size() && field.temperature_kelvin[i] > 0.0f) {
            const float kelvin = field.temperature_kelvin[i];
            minimum = std::min(minimum, kelvin);
            maximum = std::max(maximum, kelvin);
            ++temperature_samples;
        }
    }
    const bool gas = field.key.kind != FogFieldKind::Carrier;
    nlohmann::json result = {
        {"kind", field.key.kind == FogFieldKind::GasSource ? "gas_source" :
            gas ? "gas_aggregate" : "carrier"},
        {"producer_uid", field.key.producer_uid}, {"producer_name", field.producer_name},
        {"substance_tag", field.key.substance_tag},
        {"phase", gas ? "gas" : phaseName(field.key.phase)},
        {"carrier_count", field.carrier_count},
        {"active_cells", active}, {"density_sum", density_sum},
        {"temperature_samples", temperature_samples},
        {"temperature_min_kelvin", nullptr}, {"temperature_max_kelvin", nullptr}
    };
    if (temperature_samples > 0) {
        result["temperature_min_kelvin"] = minimum;
        result["temperature_max_kelvin"] = maximum;
    }
    if (!gas) {
        if (const auto* substance = RayTrophiSim::tryFindSubstanceByTag(
                field.key.substance_tag)) {
            result["substance"] = substance->name;
        }
    }
    return result;
}

Result inspectDomain(RayTrophiSim::ParticleSimulationSystem& runtime,
    std::size_t index, nlohmann::json& output) {
    MatterFogFields fields;
    std::string snapshot;
    std::string error;
    if (!prepareMatterFogSnapshot(runtime, index, g_ctx->scene.simulation_world.compute(),
            fields, snapshot, error)) {
        return Result::fail(error);
    }
    const auto& desc = runtime.gridDomains()[index];
    const auto& state = runtime.gridDomainStates()[index];
    const auto& grid = gasGrid(state);
    const auto plan = resolveFluidViews(desc,
        cachedDistinctViewKeys(state.particles, state.version));
    const float parcel_density = 1.0f /
        static_cast<float>(std::max(1, desc.fluid_params.particles_per_cell));
    // Compare the separate carrier fields to the existing aggregate deposition.
    // The reference is released before preparing the next scene inspection.
    std::vector<float> reference;
    std::vector<uint8_t> reference_mask(state.particles.size(), 0u);
    const Vec3 high = grid.origin + Vec3(
        static_cast<float>(grid.nx) + 1.0f, static_cast<float>(grid.ny) + 1.0f,
        static_cast<float>(grid.nz) + 1.0f) * grid.voxel_size;
    const Vec3 low = grid.origin - Vec3(grid.voxel_size);
    for (std::size_t i = 0; i < state.particles.size(); ++i) {
        const auto& position = state.particles.position[i];
        reference_mask[i] = position.x >= low.x && position.y >= low.y &&
            position.z >= low.z && position.x <= high.x && position.y <= high.y &&
            position.z <= high.z;
    }
    const FluidViewSelection selection{&plan, FluidView::Fog, &reference_mask};
    if (plan.anyLiveIn(FluidView::Fog)) {
        splatFogDensityWeighted(state.particles, grid.nx, grid.ny, grid.nz,
            grid.origin, grid.voxel_size, parcel_density, &selection, reference);
    }
    double maximum_error = 0.0;
    double maximum_density = 0.0;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        double sum = 0.0;
        for (const auto& field : fields.fields) {
            if (field.key.kind == FogFieldKind::Carrier) {
                sum += field.density[i];
            }
        }
        maximum_error = std::max(maximum_error, std::abs(sum - reference[i]));
        maximum_density = std::max(maximum_density, static_cast<double>(reference[i]));
    }
    nlohmann::json rows = nlohmann::json::array();
    bool connected = !fields.fields.empty();
    bool any_published = false;
    for (const auto& field : fields.fields) {
        auto row = fieldStats(field);
        row["published"] = false;
        row["vdb_id"] = -1;
        row["material_target"] = "matter:" + desc.name + ":" + fogFieldId(field.key);
        for (const auto& system : g_ctx->scene.particle_systems) {
            if (system.runtime.get() != &runtime || index >= system.matter_fog_streams.size()) {
                continue;
            }
            const auto found = system.matter_fog_streams[index].find(field.key);
            if (found != system.matter_fog_streams[index].end() && found->second.volume &&
                found->second.vdb_id >= 0) {
                const auto& stream = found->second;
                row["published"] = true;
                row["vdb_id"] = stream.vdb_id;
                row["visible"] = stream.volume->visible;
                row["name"] = stream.volume->name;
                any_published = true;
            }
        }
        if (row["active_cells"].get<std::size_t>() > 0 && !row["published"].get<bool>()) {
            connected = false;
        }
        rows.push_back(std::move(row));
    }
    output = {{"domain", desc.name}, {"state_version", state.version},
        {"render_connected", connected && any_published}, {"gas_snapshot", snapshot},
        {"grid", {grid.nx, grid.ny, grid.nz}},
        {"builder_peak_dense_bytes_bound", fields.peak_dense_bytes_bound},
        {"carrier_max_abs_density_error", maximum_error},
        {"gas_max_abs_density_error", fields.gas_max_abs_density_error},
        {"carrier_reference_max_density", maximum_density}, {"fields", std::move(rows)}};
    return Result::success();
}
} // namespace

Result inspectMatterFogFields(const std::string& domain, nlohmann::json& output) {
    if (!g_ctx) {
        return notBound();
    }
    try {
        // Existing read APIs resolve the scene's simulation runtime the same way.
        auto& runtime = scriptSimulationRuntime();
        const auto& domains = runtime.gridDomains();
        for (std::size_t index = 0; index < domains.size(); ++index) {
            if (domains[index].name == domain) {
                return inspectDomain(runtime, index, output);
            }
        }
        return Result::fail("domain not found: " + domain);
    } catch (const std::exception& error) {
        return Result::fail(error.what());
    }
}
} // namespace rtapi
