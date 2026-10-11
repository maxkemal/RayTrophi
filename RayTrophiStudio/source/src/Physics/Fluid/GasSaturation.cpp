// Fog water: saturation adjustment of a gas cell's water, autoconversion to
// rain, rain fall and rain-out. See GasSaturation.h.
#include "Fluid/GasSaturation.h"

#include "Fluid/FluidDomainSubstance.h"
#include "Fluid/FluidPhysicalMass.h"
#include "Fluid/GasInventoryGpu.h"
#include "Fluid/MatterPhaseGrid.h"
#include "Fluid/SubstanceTag.h"
#include "MaterialStateField.h"
#include "MatterExchangeLedger.h"
#include "ParticleSimulation.h"
#include "SimulationCompute.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

namespace RayTrophiSim::Fluid {
namespace {

constexpr float kGasConstant = 8.314462f;
constexpr float kAtmosphere = 101325.0f;
constexpr float kAirMolar = 0.028965f;
constexpr uint32_t kRainCapacity = 4096;
constexpr uint32_t kMaxParcelsPerCell = 8;
constexpr float kMaxRainSpeed = 10.0f;     // m/s, cap of the terminal speed
constexpr uint32_t kMaxFallSubsteps = 256;

struct SaturationConstants {
    uint32_t cells;
    float cell_volume;
    float ambient;
    float gas_ambient;
    float kelvin_per_unit;
    float boiling;
    float latent;
    float molar;
    float specific_heat;
    float background;
    float quantum;
    uint32_t capacity;
    int nx;
    int ny;
    float origin_x;
    float origin_y;
    float origin_z;
    float voxel;
    float rain_min_x;
    float rain_min_y;
    float rain_min_z;
    float rain_max_x;
    float rain_max_y;
    float rain_max_z;
    float onset;
    float conversion;
    float ceiling;
    float padding;
};
static_assert(sizeof(SaturationConstants) == 112);

struct FallConstants {
    uint32_t nx;
    uint32_t ny;
    uint32_t nz;
    uint32_t stamp;
    uint32_t substeps;
    float dt;
    float voxel;
    float cell_volume;
};
static_assert(sizeof(FallConstants) == 32);

struct RainRecord {
    uint32_t cell;
    uint32_t parcels;
    float kelvin;
    uint32_t padding;
};
static_assert(sizeof(RainRecord) == 16);

// Same layout as GasInventoryDevice.cpp's summary constants.
struct SummaryConstants {
    uint32_t cells;
    uint32_t blocks;
    uint32_t phase;
    int nx;
    int ny;
    float voxel;
    float origin_x;
    float origin_y;
    float origin_z;
};
static_assert(sizeof(SummaryConstants) == 36);

float saturatedMass(float kelvin, float cell_volume, const CondensableVapour& v) {
    const float t = std::max(kelvin, 1.0f);
    const float a = v.latent_j_kg * v.molar_kg_mol / kGasConstant;
    const float psat = kAtmosphere *
        std::exp(std::clamp(a * (1.0f / v.boiling_kelvin - 1.0f / t), -80.0f, 30.0f));
    return psat * cell_volume * v.molar_kg_mol / (kGasConstant * t);
}

bool ensureBuffer(SimulationComputeContext& compute, ComputeBufferHandle& handle,
                  std::size_t bytes, const char* name) {
    if (!handle.valid()) {
        const auto usage = ComputeBufferUsage::Storage | ComputeBufferUsage::Upload |
                           ComputeBufferUsage::Download | ComputeBufferUsage::ReadWrite;
        handle = compute.createBuffer({name, bytes, usage});
        return handle.valid();
    }
    return compute.getBufferSize(handle) >= bytes || compute.resizeBuffer(handle, bytes);
}

struct Totals {
    double condensed = 0.0;
    double evaporated = 0.0;
    double autoconverted = 0.0;
    double rained = 0.0;          // rain-out parcels (the absorbed part comes separately)
    double rained_energy = 0.0;
    std::size_t parcels = 0;
};

// Rain-out parcels in the matter domain's own liquid phase: one liquid parcel
// of exact rest mass `quantum`, jittered inside the gas cell, at rest.
void birthRain(SimulationGridDomainState& state, const CondensableVapour& vapour,
               uint32_t cell, uint32_t parcels, float kelvin, float quantum,
               Totals& totals) {
    const auto& grid = gasGrid(state);
    const int x = static_cast<int>(cell % static_cast<uint32_t>(grid.nx));
    const int y = static_cast<int>((cell / static_cast<uint32_t>(grid.nx)) %
                                   static_cast<uint32_t>(grid.ny));
    const int z = static_cast<int>(cell / (static_cast<uint32_t>(grid.nx) *
                                           static_cast<uint32_t>(grid.ny)));
    for (uint32_t k = 0; k < parcels; ++k) {
        uint32_t h = cell * 2654435761u ^ (k + 1u) * 2246822519u ^
                     static_cast<uint32_t>(state.particles.size()) * 3266489917u;
        auto next = [&]() {
            h ^= h >> 15u; h *= 2654435761u; h ^= h >> 13u;
            return 0.15f + 0.7f * static_cast<float>(h & 0xffffu) / 65535.0f;
        };
        const float jx = next(), jy = next(), jz = next();
        const Vec3 position = grid.origin +
            Vec3(x + jx, y + jy, z + jz) * grid.voxel_size;
        state.particles.emit(position, Vec3(0.0f), kelvin, 0.0f, vapour.tag,
                             nullptr, nullptr, quantum, MatterConstitutiveModel::Fluid);
        totals.rained += quantum;
        totals.rained_energy += static_cast<double>(quantum) * vapour.specific_heat * kelvin;
        ++totals.parcels;
    }
}

void recordLedger(const SimulationGridDomainDesc& domain, const CondensableVapour& vapour,
                  const Totals& totals, float air_kelvin, uint64_t event_id,
                  MatterExchangeLedger& ledger) {
    const auto add = [&](MatterExchangeKind kind, const char* source, const char* target,
                         double mass, double energy, double latent) {
        if (!(mass > 0.0)) return;
        MatterExchangeRecord record;
        record.event_id = event_id++;
        record.kind = kind;
        record.source = domain.name + source;
        record.target = domain.name + target;
        record.substance = vapour.substance;
        record.source_mass_kg = record.target_mass_kg = mass;
        record.source_energy_j = record.target_energy_j = energy;
        // Released into (condensation) or drawn from (evaporation) the air.
        record.latent_required_j = record.latent_accounted_j = latent;
        ledger.record(std::move(record));
    };
    const double sensible = static_cast<double>(vapour.specific_heat) * air_kelvin;
    add(MatterExchangeKind::Condensation, ":vapour", ":fog", totals.condensed,
        totals.condensed * sensible, totals.condensed * vapour.latent_j_kg);
    add(MatterExchangeKind::Vaporization, ":fog", ":vapour", totals.evaporated,
        totals.evaporated * sensible, totals.evaporated * vapour.latent_j_kg);
    add(MatterExchangeKind::Autoconversion, ":fog", ":rain", totals.autoconverted,
        totals.autoconverted * sensible, 0.0);
    add(MatterExchangeKind::Precipitation, ":rain", "", totals.rained,
        totals.rained_energy, 0.0);
}

} // namespace

CondensableVapour resolveCondensableVapour(const SimulationGridDomainDesc& domain) {
    CondensableVapour out;
    const auto usable = [](const SubstanceProfile* p) {
        return p && !p->fluid_flammable && p->boiling_kelvin > 0.0f &&
               p->latent_heat_vaporization > 0.0f && p->vapor_molar_mass_kg_mol > 0.0f;
    };
    const SubstanceProfile* profile = domainSubstance(domain.fluid_params);
    bool is_default = usable(profile);
    if (!is_default) {
        profile = tryFindSubstance("Water");
        if (!usable(profile)) {
            return out;
        }
    }
    out.substance = profile->name;
    out.tag = is_default ? kSubstanceUntagged : substanceTag(profile->name);
    out.boiling_kelvin = profile->boiling_kelvin;
    out.latent_j_kg = profile->latent_heat_vaporization;
    out.molar_kg_mol = profile->vapor_molar_mass_kg_mol;
    out.specific_heat = profile->specific_heat > 0.0f ? profile->specific_heat : 1000.0f;
    return out;
}

float adjustCellSaturation(float& vapour_kg, float& water_kg, float& energy_j,
                           float& air_kelvin, float cell_volume,
                           const CondensableVapour& vapour, float background_kg) {
    float m = std::max(vapour_kg, 0.0f);
    float w = std::max(water_kg, 0.0f);
    const float t0 = std::max(air_kelvin, 1.0f);
    if (!(m > 0.0f || w > 0.0f) || !std::isfinite(t0)) {
        return 0.0f;
    }
    const float rho_air = kAtmosphere * kAirMolar / (kGasConstant * t0);
    const float capacity = rho_air * cell_volume * 1005.0f + (m + w) * vapour.specific_heat;
    const float heat_per_kg = vapour.latent_j_kg / capacity;
    const auto f = [&](float d) {
        return (m - d) + background_kg -
            saturatedMass(t0 + heat_per_kg * d, cell_volume, vapour);
    };
    const float excess = m + background_kg - saturatedMass(t0, cell_volume, vapour);
    float lo = 0.0f, hi = 0.0f;
    if (excess > 0.0f && m > 0.0f) {
        hi = m;
    } else if (excess < 0.0f && w > 0.0f) {
        lo = -w;
    }
    float d = 0.0f;
    if (hi > lo) {
        if (f(hi) >= 0.0f) {
            d = hi;
        } else if (f(lo) <= 0.0f) {
            d = lo;
        } else {
            for (int i = 0; i < 28; ++i) {
                const float mid = 0.5f * (lo + hi);
                if (f(mid) > 0.0f) lo = mid; else hi = mid;
            }
            d = 0.5f * (lo + hi);
        }
    }
    if (d != 0.0f) {
        vapour_kg = std::max(m - d, 0.0f);
        water_kg = std::max(w + d, 0.0f);
        energy_j = std::max(energy_j - d * vapour.latent_j_kg, 0.0f);
        air_kelvin = t0 + d * heat_per_kg;
    }
    return d;
}

float autoconvertCellFog(float& water_kg, float& energy_j, float& rain_kg,
                         float& rain_energy_j, float air_kelvin, float specific_heat,
                         float onset_kg, float conversion, float ceiling_kg) {
    const float w = std::max(water_kg, 0.0f);
    float converted = w > onset_kg ? (w - onset_kg) * conversion : 0.0f;
    if (w - converted > ceiling_kg) {
        converted = w - ceiling_kg;
    }
    if (!(converted > 0.0f)) {
        return 0.0f;
    }
    const float sensible = std::min(std::max(energy_j, 0.0f),
                                    converted * specific_heat * std::max(air_kelvin, 1.0f));
    energy_j = std::max(energy_j, 0.0f) - sensible;
    water_kg = w - converted;
    rain_kg = std::max(rain_kg, 0.0f) + converted;
    rain_energy_j = std::max(rain_energy_j, 0.0f) + sensible;
    return converted;
}

float rainTerminalSpeed(float rain_kg_m3) {
    if (!(rain_kg_m3 > 0.0f)) {
        return 0.0f;
    }
    return std::min(36.34f * std::pow(rain_kg_m3 * 1.0e-3f, 0.1364f), kMaxRainSpeed);
}

namespace {
uint32_t fallSubsteps(float dt, float voxel) {
    const float cells = kMaxRainSpeed * dt / voxel;
    return std::clamp(static_cast<uint32_t>(std::ceil(std::max(cells, 0.0f))), 1u,
                      kMaxFallSubsteps);
}
} // namespace

void fallRain(std::vector<float>& rain_kg, std::vector<float>& rain_energy_j,
              const std::vector<uint32_t>& landing, uint32_t stamp,
              const std::vector<uint8_t>& solid, int nx, int ny, int nz,
              float voxel, float dt) {
    const std::size_t cells = static_cast<std::size_t>(nx) * ny * nz;
    if (!(dt > 0.0f) || !(voxel > 0.0f) || rain_kg.size() != cells ||
        rain_energy_j.size() != cells) {
        return;
    }
    const bool use_landing = landing.size() == cells;
    const bool use_solid = solid.size() == cells;
    const uint32_t substeps = fallSubsteps(dt, voxel);
    const float step = dt / static_cast<float>(substeps);
    const float volume = voxel * voxel * voxel;
    for (int z = 0; z < nz; ++z) {
        for (int x = 0; x < nx; ++x) {
            const std::size_t base = static_cast<std::size_t>(x) +
                static_cast<std::size_t>(nx) * ny * z;
            for (uint32_t s = 0; s < substeps; ++s) {
                for (int y = 1; y < ny; ++y) {
                    const std::size_t cell = base + static_cast<std::size_t>(y) * nx;
                    const float r = rain_kg[cell];
                    if (!(r > 0.0f) || (use_landing && landing[cell] == stamp)) {
                        continue;
                    }
                    const std::size_t below = cell - static_cast<std::size_t>(nx);
                    if (use_solid && solid[below] != 0u) {
                        continue;
                    }
                    const float f = std::min(rainTerminalSpeed(r / volume) * step / voxel, 1.0f);
                    const float moved = r * f;
                    const float moved_energy = std::max(rain_energy_j[cell], 0.0f) * f;
                    rain_kg[cell] = r - moved;
                    rain_energy_j[cell] = std::max(rain_energy_j[cell] - moved_energy, 0.0f);
                    rain_kg[below] += moved;
                    rain_energy_j[below] += moved_energy;
                }
            }
        }
    }
}

bool adjustGasSaturation(const SimulationGridDomainDesc& domain,
                         SimulationGridDomainState& state,
                         const GasSaturationInputs& inputs,
                         SimulationComputeContext* compute,
                         SimulationGridDomainComputeBuffers* buffers,
                         MatterExchangeLedger& ledger) {
    auto& stats = state.gas_stats;
    stats.saturation_measured = false;
    stats.fog_condensed_kg = stats.fog_evaporated_kg = stats.fog_latent_to_air_j = 0.0;
    stats.fog_autoconverted_kg = 0.0;
    stats.rained_kg = stats.rain_absorbed_kg = 0.0;
    stats.rain_parcels = 0;
    // The surface exchange stamped this step's landing cells with the current
    // stamp; the fall below reads it, and the next step uses a new one.
    const uint32_t stamp = state.gas_rain_stamp;
    state.gas_rain_stamp = stamp + 1u == 0u ? 1u : stamp + 1u;
    const double absorbed = state.gas_rain_absorbed_pending_kg;
    state.gas_rain_absorbed_pending_kg = 0.0;
    const bool matter_thermal = domain.type == SimulationDomainType::Matter &&
        domain.fluid_params.thermal_liquid_enabled;
    if (!inputs.scale || !(inputs.scale->kelvin_per_unit > 0.0f) || !state.valid ||
        !simulationDomainHasGas(state.type) || !state.gas_inventory_error.empty() ||
        !(matter_thermal || state.gas_liquid_water_present)) {
        return true;
    }
    const CondensableVapour vapour = resolveCondensableVapour(domain);
    stats.condensable_substance = vapour.substance;
    if (!vapour.valid()) {
        return true;
    }
    const auto begin = std::chrono::steady_clock::now();
    auto& grid = gasGrid(state);
    const std::size_t cells = grid.getCellCount();
    const float volume = grid.voxel_size * grid.voxel_size * grid.voxel_size;
    if (cells == 0 || !(volume > 0.0f) || grid.temperature.size() != cells) {
        return true;
    }
    const float kpu = inputs.scale->kelvin_per_unit;
    const float background = std::clamp(inputs.humidity, 0.0f, 1.0f) *
        saturatedMass(inputs.world_ambient_kelvin, volume, vapour);
    // Microphysics in kg per cell (the domain authors kg/m^3).
    const auto& fp = domain.fluid_params;
    const float onset = std::max(fp.fog_autoconversion_kg_m3, 0.0f) * volume;
    const float ceiling = std::max(fp.fog_ceiling_kg_m3, 0.0f) * volume;
    const float dt = std::max(inputs.dt, 0.0f);
    const float conversion = -std::expm1(-std::max(fp.fog_autoconversion_rate, 0.0f) * dt);

    // Rain-out only into this domain's own liquid phase.
    float quantum = 0.0f;
    Vec3 rain_min(0.0f), rain_max(0.0f);
    if (domain.type == SimulationDomainType::Matter) {
        const auto& liquid = liquidGrid(state);
        if (liquid.nx > 0 && liquid.voxel_size > 0.0f) {
            quantum = fluidParticleRestMassKg(vapour.tag, domainSubstance(domain.fluid_params),
                liquid.voxel_size, domain.fluid_params.particles_per_cell,
                MatterConstitutiveModel::Fluid);
            rain_min = liquid.origin;
            rain_max = liquid.origin + Vec3(static_cast<float>(liquid.nx),
                static_cast<float>(liquid.ny), static_cast<float>(liquid.nz)) * liquid.voxel_size;
        }
    }

    Totals totals;
    float mean_air_kelvin = inputs.ambient_kelvin;
    auto owner = gasInventoryDevice(state);
    if (owner) {
        // ── GPU: resident inventory ───────────────────────────────────────
        if (!compute || !buffers || !owner->populated) {
            return true;
        }
        if (!flushGasInventoryDeposits(state) || !ensureGasInventoryStorage(state) ||
            !buffers->temperature.valid() || !buffers->gas_solid_mask.valid() ||
            !buffers->deviceHasCurrent(SimulationGridDomainComputeBuffers::GridField::Temperature)) {
            holdGasInventory(state, "gas saturation: device temperature/inventory unavailable; reset required");
            return false;
        }
        const auto pools = gasInventoryReadBuffers(state);
        const auto water = gasInventoryCondensateBuffer(state);
        const auto rain = gasInventoryRainBuffers(state);
        const auto landing = gasInventoryRainLanding(state);
        const auto scratch = gasInventoryScratchBuffers(state);
        if (!pools[0].valid() || !water.valid() || !rain[0].valid() || !rain[1].valid() ||
            !landing.valid() || !scratch[0].valid() || !scratch[1].valid() ||
            !scratch[2].valid() ||
            !ensureBuffer(*compute, owner->saturation_records, kRainCapacity * sizeof(RainRecord),
                          "GasSaturationRain") ||
            !ensureBuffer(*compute, owner->saturation_counter, 2 * sizeof(uint32_t),
                          "GasSaturationCounters") ||
            !ensureBuffer(*compute, owner->summary_partials,
                          ((cells + 255) / 256) * 8 * sizeof(double), "GasInventorySummaryPartials") ||
            !ensureBuffer(*compute, owner->summary_totals, 8 * sizeof(double),
                          "GasInventorySummary")) {
            holdGasInventory(state, "gas saturation buffer allocation failed; reset required");
            return false;
        }
        const uint32_t zeros[2] = {0u, 0u};
        compute->beginTransferBatch();
        bool ok = compute->uploadBuffer(owner->saturation_counter, zeros, sizeof(zeros));
        ok = compute->endTransferBatch() && ok;
        const SaturationConstants pc{
            static_cast<uint32_t>(cells), volume, inputs.ambient_kelvin,
            inputs.gas_ambient_normalized, kpu, vapour.boiling_kelvin, vapour.latent_j_kg,
            vapour.molar_kg_mol, vapour.specific_heat, background, quantum, kRainCapacity,
            grid.nx, grid.ny, grid.origin.x, grid.origin.y, grid.origin.z, grid.voxel_size,
            rain_min.x, rain_min.y, rain_min.z, rain_max.x, rain_max.y, rain_max.z,
            onset, conversion, ceiling, 0.0f};
        ComputeBufferHandle bindings[11] = {
            pools[0], pools[1], water, buffers->temperature, scratch[0], scratch[1],
            owner->saturation_records, owner->saturation_counter, rain[0], rain[1], scratch[2]};
        ComputeDispatch command;
        command.kernel = "sim_inventory_saturation";
        command.buffers = bindings;
        command.buffer_count = 11;
        command.constants = &pc;
        command.constants_size = sizeof(pc);
        command.groups = gasInventoryDispatch(cells);
        ok = ok && compute->dispatch(command);
        // The fall after the cell pass: rain formed this step starts falling.
        const uint32_t substeps = fallSubsteps(dt, grid.voxel_size);
        const FallConstants fall{static_cast<uint32_t>(grid.nx), static_cast<uint32_t>(grid.ny),
                                 static_cast<uint32_t>(grid.nz), stamp, substeps,
                                 dt / static_cast<float>(substeps), grid.voxel_size, volume};
        ComputeBufferHandle fall_bindings[4] = {rain[0], rain[1], landing,
                                                buffers->gas_solid_mask};
        ComputeDispatch fall_command;
        fall_command.kernel = "sim_inventory_rain_fall";
        fall_command.buffers = fall_bindings;
        fall_command.buffer_count = 4;
        fall_command.constants = &fall;
        fall_command.constants_size = sizeof(fall);
        fall_command.groups = gasInventoryDispatch(
            static_cast<std::size_t>(grid.nx) * static_cast<std::size_t>(grid.nz), 64);
        if (dt > 0.0f) {
            ok = ok && compute->dispatch(fall_command);
        }
        // Totals through the inventory summary reduction: condensed and
        // evaporated in one pass, autoconverted in a second.
        const uint32_t blocks = static_cast<uint32_t>((cells + 255) / 256);
        SummaryConstants sc{static_cast<uint32_t>(cells), blocks, 0, grid.nx, grid.ny,
                            grid.voxel_size, grid.origin.x, grid.origin.y, grid.origin.z};
        double sums[8]{};
        double converted[8]{};
        for (int pass = 0; pass < 2 && ok; ++pass) {
            ComputeBufferHandle summary[4] = {
                pass == 0 ? scratch[0] : scratch[2], pass == 0 ? scratch[1] : scratch[2],
                owner->summary_partials, owner->summary_totals};
            ComputeDispatch reduce;
            reduce.kernel = "sim_inventory_summary";
            reduce.buffers = summary;
            reduce.buffer_count = 4;
            reduce.constants = &sc;
            reduce.constants_size = sizeof(sc);
            sc.phase = 0;
            reduce.groups = gasInventoryDispatch(cells);
            ok = ok && compute->dispatch(reduce);
            sc.phase = 1;
            reduce.groups = {1, 1, 1};
            ok = ok && compute->dispatch(reduce);
            compute->beginTransferBatch();
            ok = ok && compute->downloadBuffer(owner->summary_totals,
                                               pass == 0 ? sums : converted, sizeof(sums));
            ok = compute->endTransferBatch() && ok;
        }
        uint32_t counters[2]{};
        compute->beginTransferBatch();
        ok = ok && compute->downloadBuffer(owner->saturation_counter, counters, sizeof(counters));
        ok = compute->endTransferBatch() && ok;
        const uint32_t written = std::min(counters[0], kRainCapacity);
        std::vector<RainRecord> records(written);
        if (ok && written > 0) {
            compute->beginTransferBatch();
            ok = compute->downloadBuffer(owner->saturation_records, records.data(),
                                         written * sizeof(RainRecord));
            ok = compute->endTransferBatch() && ok;
        }
        // The summary totals buffer now holds these sums, not the inventory.
        owner->summary_valid = false;
        if (!ok) {
            holdGasInventory(state, "gas saturation dispatch/readback failed; reset required");
            return false;
        }
        owner->query_download_bytes += 2 * sizeof(sums) + sizeof(counters) +
                                       written * sizeof(RainRecord);
        totals.condensed = sums[0];
        totals.evaporated = sums[1];
        totals.autoconverted = converted[0];
        state.gas_liquid_water_present = counters[1] != 0u;
        for (const auto& record : records) {
            if (record.cell < cells && record.parcels > 0 &&
                record.parcels <= kMaxParcelsPerCell) {
                birthRain(state, vapour, record.cell, record.parcels, record.kelvin,
                          quantum, totals);
            }
        }
        markGasInventoryDeviceChanged(state);
        buffers->markDeviceWrote(SimulationGridDomainComputeBuffers::GridField::Temperature);
    } else {
        // ── CPU reference ──────────────────────────────────────────────────
        if (state.gas_inventory_revision != 0 || state.gas_phase_mass_kg.size() != cells ||
            state.gas_phase_energy_j.size() != cells) {
            return true;   // nothing deposited yet
        }
        state.gas_phase_condensate_kg.resize(cells, 0.0f);
        state.gas_phase_rain_kg.resize(cells, 0.0f);
        state.gas_phase_rain_energy_j.resize(cells, 0.0f);
        double kelvin_sum = 0.0;
        std::size_t kelvin_cells = 0;
        for (std::size_t cell = 0; cell < cells; ++cell) {
            float& m = state.gas_phase_mass_kg[cell];
            float& w = state.gas_phase_condensate_kg[cell];
            float& e = state.gas_phase_energy_j[cell];
            float& r = state.gas_phase_rain_kg[cell];
            float& re = state.gas_phase_rain_energy_j[cell];
            if ((m > 0.0f || w > 0.0f) && std::isfinite(grid.temperature[cell])) {
                float air = std::max(inputs.ambient_kelvin +
                    (grid.temperature[cell] - inputs.gas_ambient_normalized) * kpu, 1.0f);
                const float d = adjustCellSaturation(m, w, e, air, volume, vapour, background);
                if (d != 0.0f) {
                    grid.temperature[cell] = inputs.gas_ambient_normalized +
                        (air - inputs.ambient_kelvin) / kpu;
                    if (d > 0.0f) totals.condensed += d; else totals.evaporated -= d;
                    kelvin_sum += air;
                    ++kelvin_cells;
                }
                totals.autoconverted += autoconvertCellFog(w, e, r, re, air,
                    vapour.specific_heat, onset, conversion, ceiling);
            }
            if (quantum > 0.0f && r >= quantum) {
                const int x = static_cast<int>(cell % grid.nx);
                const int y = static_cast<int>((cell / grid.nx) % grid.ny);
                const int z = static_cast<int>(cell / (std::size_t(grid.nx) * grid.ny));
                const Vec3 centre = grid.gridToWorld(x, y, z);
                if (centre.x > rain_min.x && centre.y > rain_min.y && centre.z > rain_min.z &&
                    centre.x < rain_max.x && centre.y < rain_max.y && centre.z < rain_max.z) {
                    const uint32_t parcels = std::min(
                        static_cast<uint32_t>(r / quantum), kMaxParcelsPerCell);
                    const float rained = parcels * quantum;
                    const float kelvin = std::max(std::max(re, 0.0f) /
                                                  (vapour.specific_heat * r), 1.0f);
                    re = std::max(re, 0.0f) * std::max(1.0f - rained / r, 0.0f);
                    r = std::max(r - rained, 0.0f);
                    birthRain(state, vapour, static_cast<uint32_t>(cell), parcels,
                              kelvin, quantum, totals);
                }
            }
            if (!(r > 0.0f)) {
                re = 0.0f;
            }
        }
        fallRain(state.gas_phase_rain_kg, state.gas_phase_rain_energy_j,
                 state.gas_rain_landing, stamp, grid.solid, grid.nx, grid.ny, grid.nz,
                 grid.voxel_size, dt);
        const auto positive = [](float value) { return value > 0.0f; };
        state.gas_liquid_water_present =
            std::any_of(state.gas_phase_condensate_kg.begin(),
                        state.gas_phase_condensate_kg.end(), positive) ||
            std::any_of(state.gas_phase_rain_kg.begin(), state.gas_phase_rain_kg.end(), positive);
        if (kelvin_cells > 0) {
            mean_air_kelvin = static_cast<float>(kelvin_sum / kelvin_cells);
        }
    }

    stats.saturation_measured = true;
    stats.fog_condensed_kg = totals.condensed;
    stats.fog_evaporated_kg = totals.evaporated;
    stats.fog_latent_to_air_j = (totals.condensed - totals.evaporated) * vapour.latent_j_kg;
    stats.fog_autoconverted_kg = totals.autoconverted;
    stats.rain_absorbed_kg = absorbed;
    stats.rained_kg = totals.rained + absorbed;
    stats.rain_parcels = totals.parcels;
    stats.saturation_ms = std::chrono::duration<float, std::milli>(
        std::chrono::steady_clock::now() - begin).count();
    if (totals.parcels > 0) {
        ++state.version;
    }
    recordLedger(domain, vapour, totals, mean_air_kelvin, inputs.event_id_base, ledger);
    return true;
}

} // namespace RayTrophiSim::Fluid
