#include "Fluid/MatterPhaseConfig.h"
#include "Fluid/FluidGridResourceBudget.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace RayTrophiSim::Fluid {
namespace {

bool finite(const Vec3& value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

float logicalPadding(const SimulationGridDomainDesc& domain) {
    return domain.source_mode == SimulationGridDomainSourceMode::Adaptive
        ? 0.0f : std::max(domain.padding, 0.0f);
}

bool validBox(const Vec3& lo, const Vec3& hi, float voxel) {
    const Vec3 extent = hi - lo;
    return finite(lo) && finite(hi) && finite(extent) &&
        extent.x > 0.0f && extent.y > 0.0f && extent.z > 0.0f &&
        std::isfinite(voxel) && voxel >= 1.0e-6f;
}

void fit(PhaseLayout& layout, float voxel, int cap) {
    const Vec3 extent = layout.requested_max - layout.origin;
    const auto dimension = [voxel, cap](float length) {
        const double count = std::ceil(static_cast<double>(length) / voxel);
        return static_cast<int>(std::clamp(count, 8.0, static_cast<double>(cap)));
    };
    layout.nx = dimension(extent.x);
    layout.ny = dimension(extent.y);
    layout.nz = dimension(extent.z);
    layout.voxel = std::max({voxel, extent.x / layout.nx,
                            extent.y / layout.ny, extent.z / layout.nz});
}

PhaseLayout makeLayout(const MatterPhaseSettings& setting, const Vec3& lo,
                       const Vec3& hi, int nx, int ny, int nz, float voxel, int cap) {
    PhaseLayout result;
    result.origin = lo;
    result.requested_max = hi;
    result.nx = std::max(nx, 8);
    result.ny = std::max(ny, 8);
    result.nz = std::max(nz, 8);
    result.voxel = voxel;
    if (setting.override_enabled) {
        result.origin = lo + setting.offset_min;
        result.requested_max = lo + setting.offset_max;
        fit(result, setting.voxel_size, cap);
    }
    return result;
}

std::size_t workingBytes(const PhaseLayouts& layouts) {
    return layouts.gas.cells() * kGasWorkingBytesPerCell +
        layouts.liquid.cells() * kLiquidWorkingBytesPerCell;
}

bool syncGrid(FluidSim::FluidGrid& grid, const PhaseLayout& layout,
              bool gas, bool sparse, bool force = false) {
    if (layout.cells() == 0) {
        if (grid.getCellCount() != 0) {
            grid = FluidSim::FluidGrid{};
            return true;
        }
        return false;
    }
    const bool layout_changed = force || grid.nx != layout.nx || grid.ny != layout.ny ||
        grid.nz != layout.nz || grid.voxel_size != layout.voxel ||
        grid.allocate_gas_channels != gas;
    const bool changed = layout_changed || grid.pressure.size() != layout.cells();
    grid.sparse_mode_enabled = sparse;
    grid.allocate_gas_channels = gas;
    if (changed) {
        // A disk replay has scalar fields but no solver scratch. Hydrating the
        // same layout must keep those fields; a real layout change resets them.
        auto density = !layout_changed ? std::move(grid.density) : std::vector<float>{};
        auto temperature = !layout_changed
            ? std::move(grid.temperature) : std::vector<float>{};
        auto fuel = !layout_changed ? std::move(grid.fuel) : std::vector<float>{};
        auto interaction = !layout_changed
            ? std::move(grid.interaction) : std::vector<float>{};
        grid.resize(layout.nx, layout.ny, layout.nz, layout.voxel, layout.origin);
        if (density.size() == layout.cells()) {
            grid.density = std::move(density);
        }
        if (gas && temperature.size() == layout.cells()) {
            grid.temperature = std::move(temperature);
        }
        if (gas && fuel.size() == layout.cells()) {
            grid.fuel = std::move(fuel);
        }
        if (gas && interaction.size() == layout.cells()) {
            grid.interaction = std::move(interaction);
        }
    } else {
        grid.origin = layout.origin;
    }
    return changed;
}

nlohmann::json vectorJson(const Vec3& value) {
    return {value.x, value.y, value.z};
}

Vec3 readVector(const nlohmann::json& value) {
    if (!value.is_array() || value.size() != 3) {
        throw std::runtime_error("phase bounds must contain three numbers");
    }
    for (const auto& component : value) {
        if (!component.is_number()) {
            throw std::runtime_error("phase bounds must contain three numbers");
        }
    }
    return Vec3(value[0].get<float>(), value[1].get<float>(), value[2].get<float>());
}

} // namespace

std::size_t PhaseLayout::cells() const {
    return static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
        static_cast<std::size_t>(nz);
}

Vec3 logicalGridOrigin(const SimulationGridDomainDesc& domain) {
    return Vec3::min(domain.bounds_min, domain.bounds_max) -
        Vec3(logicalPadding(domain));
}

bool parseGridPhase(const std::string& name, GridPhase& phase, std::string& error) {
    if (name == "gas") {
        phase = GridPhase::Gas;
        return true;
    }
    if (name == "liquid") {
        phase = GridPhase::Liquid;
        return true;
    }
    error = "phase must be gas or liquid (granular uses the liquid grid)";
    return false;
}

bool setPhaseGrid(SimulationGridDomainDesc& domain, GridPhase phase, bool inherit,
                  const Vec3& bounds_min, const Vec3& bounds_max, float voxel,
                  std::string& error) {
    const bool present = phase == GridPhase::Gas
        ? simulationDomainHasGas(domain.type) : simulationDomainHasLiquid(domain.type);
    if (!present) {
        error = "requested phase is absent from this domain";
        return false;
    }
    auto& destination = phase == GridPhase::Gas
        ? domain.gas_phase_grid : domain.liquid_phase_grid;
    if (inherit) {
        destination.override_enabled = false;
        return true;
    }
    const Vec3 origin = logicalGridOrigin(domain);
    if (!validBox(bounds_min, bounds_max, voxel) || !finite(origin) ||
        !finite(bounds_min - origin) || !finite(bounds_max - origin)) {
        error = "phase bounds must be finite and strictly ordered; voxel must be >= 1e-6 m";
        return false;
    }
    MatterPhaseSettings candidate;
    candidate.override_enabled = true;
    candidate.offset_min = bounds_min - origin;
    candidate.offset_max = bounds_max - origin;
    candidate.voxel_size = voxel;
    destination = candidate;
    return true;
}

PhaseLayouts resolvePhaseLayouts(const SimulationGridDomainDesc& domain,
                                const Vec3& logical_min, const Vec3& logical_max,
                                int nx, int ny, int nz, float voxel) {
    PhaseLayouts result;
    // Same knob as the main grid, no hidden ceiling (MatterDomainSynchronization).
    const int cap = std::max(domain.max_auto_resolution, 32);
    if (simulationDomainHasGas(domain.type)) {
        result.gas = makeLayout(domain.gas_phase_grid, logical_min, logical_max,
                               nx, ny, nz, voxel, cap);
    }
    if (simulationDomainHasLiquid(domain.type)) {
        result.liquid = makeLayout(domain.liquid_phase_grid, logical_min, logical_max,
                                  nx, ny, nz, voxel, cap);
    }
    const std::size_t budget = domain.enforce_resource_budget && domain.resource_budget_mb
        ? static_cast<std::size_t>(domain.resource_budget_mb) * 1024u * 1024u
        : std::numeric_limits<std::size_t>::max();
    while (workingBytes(result) > budget) {
        bool reducible = false;
        for (PhaseLayout* phase : {&result.gas, &result.liquid}) {
            if (phase->cells() == 0 || (phase->nx == 8 && phase->ny == 8 && phase->nz == 8)) {
                continue;
            }
            reducible = true;
            fit(*phase, phase->voxel * 1.1f, cap);
            phase->budget_clamped = true;
        }
        // Positive MB budgets always fit the 8^3 minimum working set.
        if (!reducible) {
            break;
        }
    }
    result.working_bytes = workingBytes(result);
    return result;
}

PhaseLayouts previewPhaseLayouts(const SimulationGridDomainDesc& domain) {
    const Vec3 lo = logicalGridOrigin(domain);
    const Vec3 hi = Vec3::max(domain.bounds_min, domain.bounds_max) +
        Vec3(logicalPadding(domain));
    const Vec3 extent = hi - lo;
    const int cap = std::max(domain.max_auto_resolution, 32);
    const float largest = std::max({extent.x, extent.y, extent.z, 0.001f});
    const float requested = domain.preserve_voxel_size_on_resize && domain.voxel_size > 1e-6f
        ? domain.voxel_size : largest / static_cast<float>(cap);
    PhaseLayout fallback;
    fallback.origin = lo;
    fallback.requested_max = hi;
    fit(fallback, requested, cap);
    // Legacy fallback derives a cubic covering voxel from the integer request.
    fallback.voxel = std::max({extent.x / fallback.nx, extent.y / fallback.ny,
                               extent.z / fallback.nz, 1e-6f});
    return resolvePhaseLayouts(domain, lo, hi, fallback.nx, fallback.ny,
                               fallback.nz, fallback.voxel);
}

void synchronizePhaseStorage(SimulationGridDomainState& state,
                             const SimulationGridDomainDesc& domain,
                             const PhaseLayouts& layouts) {
    const bool sparse = domain.use_sparse_tiles ||
        domain.backend == SimulationDomainBackend::CPU_SparseVDB;
    const bool gas = simulationDomainHasGas(domain.type);
    const auto& primary_layout = gas ? layouts.gas : layouts.liquid;
    const bool hydrate_primary = state.valid && state.channels == domain.channels &&
        state.type == domain.type && state.grid.nx == primary_layout.nx &&
        state.grid.ny == primary_layout.ny && state.grid.nz == primary_layout.nz &&
        state.grid.voxel_size == primary_layout.voxel &&
        state.grid.allocate_gas_channels == gas &&
        state.grid.pressure.size() != primary_layout.cells();
    const bool primary_changed = syncGrid(state.grid,
        gas ? layouts.gas : layouts.liquid, gas, sparse,
        !state.valid || state.channels != domain.channels || state.type != domain.type);
    const bool secondary_changed = syncGrid(state.matter_liquid_grid,
        domain.type == SimulationDomainType::Matter ? layouts.liquid : PhaseLayout{},
        false, sparse);
    if ((primary_changed && !hydrate_primary) || state.channels != domain.channels ||
        state.type != domain.type) {
        state.gas_phase_mass_kg.clear();
        state.gas_phase_energy_j.clear();
    }
    if (primary_changed || secondary_changed) {
        ++state.version;
    }
}

bool phaseStorageMatches(const SimulationGridDomainState& state,
                         const SimulationGridDomainDesc& domain) {
    if (state.type != domain.type || state.matter_liquid_active ||
        state.phase_config_hash != hashPhaseSettings(0, domain)) {
        return false;
    }
    const auto layouts = previewPhaseLayouts(domain);
    const auto matches = [](const FluidSim::FluidGrid& grid, const PhaseLayout& layout) {
        return grid.nx == layout.nx && grid.ny == layout.ny && grid.nz == layout.nz &&
            std::abs(grid.voxel_size - layout.voxel) <= 1e-6f &&
            grid.pressure.size() == layout.cells();
    };
    if (!matches(state.grid, simulationDomainHasGas(domain.type) ? layouts.gas : layouts.liquid)) {
        return false;
    }
    return domain.type != SimulationDomainType::Matter ||
        matches(state.matter_liquid_grid, layouts.liquid);
}

nlohmann::json phaseSettingsJson(const SimulationGridDomainDesc& domain) {
    const auto encode = [](const MatterPhaseSettings& setting) {
        return nlohmann::json{{"override", setting.override_enabled},
            {"offset_min", vectorJson(setting.offset_min)},
            {"offset_max", vectorJson(setting.offset_max)}, {"voxel", setting.voxel_size}};
    };
    return {{"version", 1}, {"gas", encode(domain.gas_phase_grid)},
            {"liquid", encode(domain.liquid_phase_grid)}};
}

void loadPhaseSettings(const nlohmann::json& object, SimulationGridDomainDesc& domain) {
    if (!object.contains("phase_grids")) {
        return;
    }
    const auto& grids = object.at("phase_grids");
    if (!grids.is_object() || grids.value("version", 0) != 1) {
        throw std::runtime_error("unsupported phase_grids schema");
    }
    const auto decode = [](const nlohmann::json& value) {
        MatterPhaseSettings setting;
        setting.override_enabled = value.at("override").get<bool>();
        setting.offset_min = readVector(value.at("offset_min"));
        setting.offset_max = readVector(value.at("offset_max"));
        setting.voxel_size = value.at("voxel").get<float>();
        if (!validBox(setting.offset_min, setting.offset_max, setting.voxel_size)) {
            throw std::runtime_error("invalid saved phase grid bounds/voxel");
        }
        return setting;
    };
    const auto gas = decode(grids.at("gas"));
    const auto liquid = decode(grids.at("liquid"));
    domain.gas_phase_grid = gas;
    domain.liquid_phase_grid = liquid;
}

nlohmann::json phaseGridInfo(const SimulationGridDomainDesc& domain,
                            const SimulationGridDomainState* state) {
    const auto layouts = previewPhaseLayouts(domain);
    const Vec3 logical = logicalGridOrigin(domain);
    const auto encode = [&](bool present, const MatterPhaseSettings& setting,
                            const PhaseLayout& layout, const FluidSim::FluidGrid* grid,
                            std::size_t bytes_per_cell) {
        nlohmann::json value{{"present", present}, {"inherit", !setting.override_enabled},
                            {"measured", grid != nullptr && present}};
        if (!present) {
            return value;
        }
        const Vec3 lo = setting.override_enabled ? logical + setting.offset_min : logical;
        const Vec3 hi = setting.override_enabled ? logical + setting.offset_max
            : Vec3::max(domain.bounds_min, domain.bounds_max) + Vec3(logicalPadding(domain));
        value["requested_bounds_min"] = vectorJson(lo);
        value["requested_bounds_max"] = vectorJson(hi);
        value["requested_voxel"] = setting.override_enabled
            ? setting.voxel_size : domain.voxel_size;
        value["bounds_min"] = vectorJson(grid ? grid->origin : layout.origin);
        value["bounds_max"] = vectorJson(grid ? gridBoundsMax(*grid)
            : layout.origin + Vec3(static_cast<float>(layout.nx),
                static_cast<float>(layout.ny), static_cast<float>(layout.nz)) * layout.voxel);
        value["voxel"] = grid ? grid->voxel_size : layout.voxel;
        value["resolution"] = grid ? nlohmann::json{grid->nx, grid->ny, grid->nz}
            : nlohmann::json{layout.nx, layout.ny, layout.nz};
        value["cells"] = grid ? grid->getCellCount() : layout.cells();
        value["working_bytes"] = value["cells"].get<std::size_t>() * bytes_per_cell;
        value["budget_clamped"] = layout.budget_clamped;
        return value;
    };
    const bool live = state && state->valid;
    nlohmann::json result{{"domain", domain.name},
        {"gas", encode(simulationDomainHasGas(domain.type), domain.gas_phase_grid,
            layouts.gas, live ? &gasGrid(*state) : nullptr, kGasWorkingBytesPerCell)},
        {"liquid", encode(simulationDomainHasLiquid(domain.type), domain.liquid_phase_grid,
            layouts.liquid, live ? &liquidGrid(*state) : nullptr, kLiquidWorkingBytesPerCell)},
        {"estimated_working_bytes", layouts.working_bytes},
        {"budget_mb", domain.resource_budget_mb},
        {"budget_enforced", domain.enforce_resource_budget && domain.resource_budget_mb > 0}};
    result["working_bytes"] = result["gas"].value("working_bytes", std::size_t(0)) +
        result["liquid"].value("working_bytes", std::size_t(0));
    return result;
}

uint64_t hashPhaseSettings(uint64_t hash, const SimulationGridDomainDesc& domain) {
    const auto mix = [](uint64_t value, uint64_t next) {
        return value ^ (next + 0x9e3779b97f4a7c15ull + (value << 6) + (value >> 2));
    };
    for (const auto* setting : {&domain.gas_phase_grid, &domain.liquid_phase_grid}) {
        hash = mix(hash, setting->override_enabled);
        if (!setting->override_enabled) {
            continue;
        }
        for (float value : {setting->offset_min.x, setting->offset_min.y,
                            setting->offset_min.z, setting->offset_max.x,
                            setting->offset_max.y, setting->offset_max.z, setting->voxel_size}) {
            uint32_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            hash = mix(hash, bits);
        }
    }
    return hash;
}

void adoptPresetPhaseGrids(SimulationGridDomainDesc& matter,
                           const SimulationGridDomainDesc& gas,
                           const SimulationGridDomainDesc& liquid) {
    auto candidate = matter;
    std::string error;
    if (!setPhaseGrid(candidate, GridPhase::Gas, false, gas.bounds_min,
                       gas.bounds_max, gas.voxel_size, error) ||
        !setPhaseGrid(candidate, GridPhase::Liquid, false, liquid.bounds_min,
                       liquid.bounds_max, liquid.voxel_size, error)) {
        throw std::runtime_error("invalid preset phase grids: " + error);
    }
    matter.gas_phase_grid = candidate.gas_phase_grid;
    matter.liquid_phase_grid = candidate.liquid_phase_grid;
}

} // namespace RayTrophiSim::Fluid
