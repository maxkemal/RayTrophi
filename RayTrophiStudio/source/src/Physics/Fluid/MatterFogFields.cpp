#include "Fluid/MatterFogFields.h"
#include "Fluid/FluidFogDensity.h"
#include "Fluid/FluidParticles.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <new>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace RayTrophiSim::Fluid {

bool FogFieldKey::operator<(const FogFieldKey& other) const {
    return std::tie(kind, substance_tag, phase, producer_uid) <
        std::tie(other.kind, other.substance_tag, other.phase, other.producer_uid);
}

bool FogFieldKey::operator==(const FogFieldKey& other) const {
    return kind == other.kind && substance_tag == other.substance_tag && phase == other.phase &&
        producer_uid == other.producer_uid;
}

namespace {
bool finiteVector(const Vec3& value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

bool gridCells(const FogFieldGrid& grid, std::size_t& cells) {
    if (grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0 ||
        !std::isfinite(grid.voxel) || grid.voxel <= 0.0f || !finiteVector(grid.origin)) {
        return false;
    }
    cells = 1;
    for (const int size : {grid.nx, grid.ny, grid.nz}) {
        if (cells > std::numeric_limits<std::size_t>::max() / std::size_t(size)) {
            return false;
        }
        cells *= std::size_t(size);
    }
    return finiteVector(grid.origin + Vec3(float(grid.nx) + 0.5f,
        float(grid.ny) + 0.5f, float(grid.nz) + 0.5f) * grid.voxel);
}

bool charge(std::size_t count, std::size_t bytes, std::size_t budget, std::size_t& total) {
    if (total > budget || count > (budget - total) / bytes) {
        return false;
    }
    total += count * bytes;
    return true;
}

FogFieldKey carrierKey(const MatterFogFieldInput& input, std::size_t index) {
    const auto& particles = *input.particles;
    uint32_t tag = index < particles.substance_tag.size() ? particles.substance_tag[index] : 0;
    if (tag == 0) {
        tag = input.default_substance_tag;
    }
    return {FogFieldKind::Carrier, tag, particleOutputPhase(particles, index)};
}

bool inSplatSupport(const Vec3& position, const FogFieldGrid& grid) {
    const Vec3 low = grid.origin - Vec3(0.5f * grid.voxel);
    const Vec3 high = grid.origin + Vec3(float(grid.nx) + 0.5f,
        float(grid.ny) + 0.5f, float(grid.nz) + 0.5f) * grid.voxel;
    return position.x > low.x && position.y > low.y && position.z > low.z &&
        position.x < high.x && position.y < high.y && position.z < high.z;
}

bool prepare(const MatterFogFieldInput& input, MatterFogFields& result, std::string& error) {
    const auto refuse = [&](const char* message) {
        error = message;
        return false;
    };
    if (!input.particles || !input.plan || !std::isfinite(input.parcel_density) ||
        input.parcel_density < 0.0f || input.max_streams == 0) {
        return refuse("Fog field input requires particles, plan and valid density/stream budget");
    }
    const auto& particles = *input.particles;
    std::map<FogFieldKey, std::size_t> groups;
    for (std::size_t i = 0; i < particles.size(); ++i) {
        if (input.plan->viewForParticle(particles, i) != FluidView::Fog) {
            continue;
        }
        if (!finiteVector(particles.position[i]) ||
            (i < particles.mass_fraction.size() &&
             (!std::isfinite(particles.mass_fraction[i]) || particles.mass_fraction[i] < 0.0f))) {
            return refuse("Fog carrier positions and mass fractions must be finite and valid");
        }
        ++groups[carrierKey(input, i)];
        if (groups.size() > input.max_streams) {
            return refuse("Fog stream budget exceeded");
        }
    }
    const bool gas_enabled = input.gas && input.plan->phase_output.gas != LabelRoute::Hidden;
    const auto* sources = gas_enabled ? input.gas->sources : nullptr;
    const std::size_t gas_streams = !gas_enabled ? 0u :
        (sources && !sources->empty() ? sources->size() : 1u);
    if (groups.size() + gas_streams > input.max_streams) {
        return refuse("Fog stream budget exceeded");
    }
    std::size_t carrier_cells = 0;
    std::size_t gas_cells = 0;
    if (!groups.empty() && !gridCells(input.carrier_grid, carrier_cells)) {
        return refuse("Invalid carrier Fog grid");
    }
    if (gas_enabled) {
        const auto& gas = *input.gas;
        if (!gas.authoritative || !gridCells(gas.grid, gas_cells) || !gas.density ||
            gas.density->size() != gas_cells ||
            (gas.temperature && gas.temperature->size() != gas_cells)) {
            return refuse("Gas Fog requires an authoritative snapshot with matching field sizes");
        }
        if (gas.temperature_units != GasFogTemperatureUnits::Kelvin &&
            gas.temperature_units != GasFogTemperatureUnits::Normalized3000) {
            return refuse("Unknown gas temperature units");
        }
    }
    std::size_t bytes = 0;
    for (std::size_t stream = 0; stream < groups.size(); ++stream) {
        if (!charge(carrier_cells, 2 * sizeof(float), input.max_dense_bytes, bytes)) {
            return refuse("Fog dense field budget exceeded");
        }
    }
    const std::size_t gas_floats = gas_streams * 2 + (sources ? 4u : 0u);
    if (!charge(gas_cells, gas_floats * sizeof(float), input.max_dense_bytes, bytes) ||
        (!groups.empty() &&
         (!charge(carrier_cells, 2 * sizeof(float), input.max_dense_bytes, bytes) ||
          !charge(particles.size(), sizeof(uint8_t), input.max_dense_bytes, bytes)))) {
        return refuse("Fog dense field/scratch budget exceeded");
    }
    result.peak_dense_bytes_bound = bytes;
    std::vector<uint8_t> mask;
    if (!groups.empty()) {
        mask.resize(particles.size());
    }
    for (const auto& group : groups) {
        for (std::size_t i = 0; i < particles.size(); ++i) {
            mask[i] = carrierKey(input, i) == group.first &&
                inSplatSupport(particles.position[i], input.carrier_grid);
        }
        MatterFogField field;
        field.key = group.first;
        field.grid = input.carrier_grid;
        field.carrier_count = group.second;
        const FluidViewSelection selection{input.plan, FluidView::Fog, &mask};
        const auto& grid = field.grid;
        if (!splatFogDensityWeighted(particles, grid.nx, grid.ny, grid.nz, grid.origin,
                grid.voxel, input.parcel_density, &selection, field.density)) {
            field.density.assign(carrier_cells, 0.0f);
        }
        splatFogTemperatureKelvin(particles, grid.nx, grid.ny, grid.nz, grid.origin,
            grid.voxel, 0.0f, field.temperature_kelvin, &selection);
        result.fields.push_back(std::move(field));
    }
    if (gas_enabled) {
        MatterFogField field;
        field.key.kind = FogFieldKind::GasAggregate;
        field.grid = input.gas->grid;
        field.density = *input.gas->density;
        if (input.gas->temperature) {
            field.temperature_kelvin = *input.gas->temperature;
        }
        const float scale = input.gas->temperature_units == GasFogTemperatureUnits::Kelvin
            ? 1.0f : 3000.0f;
        for (std::size_t i = 0; i < gas_cells; ++i) {
            if (!std::isfinite(field.density[i]) || field.density[i] < 0.0f) {
                return refuse("Gas Fog density must be finite and nonnegative");
            }
            if (!field.temperature_kelvin.empty()) {
                float& temperature = field.temperature_kelvin[i];
                temperature *= scale;
                if (!std::isfinite(temperature) || temperature < 0.0f) {
                    return refuse("Gas Fog temperature must be finite and nonnegative");
                }
            }
        }
        if (!sources || sources->empty()) {
            result.fields.push_back(std::move(field));
        } else {
            // Reconcile a late/unattributed producer to UID 0. Source ownership
            // never takes a second copy of the complete gas density field.
            if (!sources->count(0)) {
                return refuse("Gas source partition requires an unattributed stream");
            }
            for (const auto& [uid, image] : *sources) {
                if (image.density.size() != gas_cells || !image.pending_density.empty()) {
                    return refuse("Gas source partition has invalid or uncommitted state");
                }
                for (float value : image.density) {
                    if (!std::isfinite(value) || value < 0.0f) {
                        return refuse("Gas source contribution must be finite and nonnegative");
                    }
                }
            }
            // Move the aggregate storage into the first component, then reuse
            // its temperature image for all streams (one physical temperature).
            const auto aggregate_density = std::move(field.density);
            const auto temperature = std::move(field.temperature_kelvin);
            std::vector<double> totals(gas_cells, 0.0);
            for (const auto& [uid, image] : *sources) {
                for (std::size_t cell = 0; cell < gas_cells; ++cell) {
                    totals[cell] += image.density[cell];
                }
            }
            for (const auto& [uid, image] : *sources) {
                MatterFogField component;
                component.key.kind = uid == 0 ? FogFieldKind::GasAggregate : FogFieldKind::GasSource;
                component.key.producer_uid = uid;
                component.producer_name = image.name;
                component.grid = field.grid;
                component.density.resize(gas_cells);
                component.temperature_kelvin = temperature;
                for (std::size_t cell = 0; cell < gas_cells; ++cell) {
                    const double sum = totals[cell];
                    const float target = aggregate_density[cell];
                    component.density[cell] = sum > target
                        ? float(image.density[cell] * target / sum)
                        : image.density[cell] + (uid == 0 ? float(target - sum) : 0.0f);
                }
                result.fields.push_back(std::move(component));
            }
            for (std::size_t cell = 0; cell < gas_cells; ++cell) {
                double sum = 0.0;
                for (const auto& component : result.fields) {
                    if (component.key.kind != FogFieldKind::Carrier) {
                        sum += component.density[cell];
                    }
                }
                result.gas_max_abs_density_error = std::max(result.gas_max_abs_density_error,
                    std::abs(sum - aggregate_density[cell]));
            }
        }
    }
    return true;
}
} // namespace

bool buildMatterFogFields(const MatterFogFieldInput& input,
                         MatterFogFields& output, std::string& error) {
    error.clear();
    MatterFogFields candidate;
    try {
        if (!prepare(input, candidate, error)) {
            return false;
        }
    } catch (const std::bad_alloc&) {
        error = "Fog field allocation failed";
        return false;
    } catch (const std::length_error&) {
        error = "Fog field allocation exceeds container capacity";
        return false;
    }
    output = std::move(candidate);
    return true;
}

} // namespace RayTrophiSim::Fluid
