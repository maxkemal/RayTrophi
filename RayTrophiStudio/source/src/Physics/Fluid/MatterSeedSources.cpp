#include "Fluid/MatterSeedSources.h"
#include "Fluid/FluidDomainSubstance.h"
#include "Fluid/FluidPhysicalMass.h"
#include "Fluid/MatterGrain.h"
#include "Fluid/MatterGrainBirth.h"
#include "Fluid/MatterPhaseGrid.h"
#include "Fluid/MatterSeedPlanning.h"
#include "Fluid/MatterSubstanceState.h"
#include "Fluid/MeshVolumeTest.h"
#include "Fluid/SubstanceTag.h"
#include "MaterialStateField.h"
#include "ParticleSimulation.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>

namespace RayTrophiSim::Fluid {
namespace {

bool finite(const Vec3 &value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

Vec3 vectorValue(const nlohmann::json &value) {
    if (!value.is_array() || value.size() != 3) {
        throw std::runtime_error("expected a three-number vector");
    }
    return Vec3(value.at(0).get<float>(), value.at(1).get<float>(), value.at(2).get<float>());
}

nlohmann::json vectorJson(const Vec3 &value) {
    return nlohmann::json::array({value.x, value.y, value.z});
}

bool inside(const Vec3 &point, const Vec3 &low, const Vec3 &high) {
    return point.x >= low.x && point.x < high.x && point.y >= low.y && point.y < high.y &&
           point.z >= low.z && point.z < high.z;
}

} // namespace

nlohmann::json matterSeedSourceToJson(const MatterSeedSource &source) {
    return {{"id", source.id},
            {"name", source.name},
            {"enabled", source.enabled},
            {"substance", source.substance},
            {"bounds_min", vectorJson(source.bounds_min)},
            {"bounds_max", vectorJson(source.bounds_max)},
            {"mesh_object", source.mesh_object},
            {"velocity", vectorJson(source.velocity)},
            {"temperature_mode", source.temperature_mode},
            {"temperature_kelvin", source.temperature_kelvin},
            {"grain_saturation", source.grain_saturation},
            {"bond_body", DemBond::seedJson(source.bond_body)}};
}

uint64_t matterSeedSourcesHash(const std::vector<MatterSeedSource> &sources) {
    if (sources.empty()) {
        return 0;
    }
    uint64_t hash = 14695981039346656037ull;
    for (const auto &source : sources) {
        for (unsigned char byte : matterSeedSourceToJson(source).dump()) {
            hash = (hash ^ byte) * 1099511628211ull;
        }
    }
    return hash;
}

uint64_t matterSeedSourceId(const nlohmann::json &value) {
    if ((value.is_number_unsigned() && value.get<uint64_t>() > 0) ||
        (value.type() == nlohmann::json::value_t::number_integer && value.get<int64_t>() > 0)) {
        return value.get<uint64_t>();
    }
    throw std::runtime_error("seed source ID must be a positive integer");
}

uint64_t matterSeedRecipeHash(const SimulationGridDomainDesc &domain) {
    uint64_t hash = matterSeedSourcesHash(domain.matter_seed_sources);
    const nlohmann::json settings{{"ppc", domain.fluid_seed_particles_per_cell},
                                  {"settle", domain.fluid_seed_settle},
                                  {"steps", domain.fluid_settle_max_steps},
                                  {"tolerance", domain.fluid_settle_displacement_m}};
    for (unsigned char byte : settings.dump()) {
        hash = (hash ^ byte) * 1099511628211ull;
    }
    return hash;
}

uint64_t matterSeedAuthoringSignature(const SimulationGridDomainDesc &domain) {
    if (domain.matter_seed_sources_enabled) {
        return domain.matter_seed_sources_applied_hash;
    }
    uint64_t hash = 14695981039346656037ull;
    const nlohmann::json settings{{"ppc", domain.fluid_seed_particles_per_cell},
                                  {"wetness", domain.fluid_seed_grain_saturation},
                                  {"fill", domain.fluid_fill_level},
                                  {"margin", domain.fluid_fill_wall_margin},
                                  {"settle", domain.fluid_seed_settle},
                                  {"steps", domain.fluid_settle_max_steps},
                                  {"tolerance", domain.fluid_settle_displacement_m}};
    for (unsigned char byte : settings.dump()) {
        hash = (hash ^ byte) * 1099511628211ull;
    }
    return hash;
}

bool parseMatterSeedSource(const nlohmann::json &patch, MatterSeedSource &source,
                           std::string &error, bool loading) {
    auto candidate = source;
    try {
        if (!patch.is_object()) {
            throw std::runtime_error("seed source settings must be an object");
        }
        for (auto item = patch.begin(); item != patch.end(); ++item) {
            const auto &key = item.key();
            const auto &value = item.value();
            if (key == "id" && loading) {
                candidate.id = matterSeedSourceId(value);
            } else if (key == "name") {
                candidate.name = value.get<std::string>();
            } else if (key == "enabled") {
                candidate.enabled = value.get<bool>();
            } else if (key == "substance") {
                candidate.substance = value.get<std::string>();
            } else if (key == "bounds_min") {
                candidate.bounds_min = vectorValue(value);
            } else if (key == "bounds_max") {
                candidate.bounds_max = vectorValue(value);
            } else if (key == "mesh_object") {
                candidate.mesh_object = value.get<std::string>();
            } else if (key == "velocity") {
                candidate.velocity = vectorValue(value);
            } else if (key == "temperature_mode") {
                candidate.temperature_mode = value.get<std::string>();
            } else if (key == "temperature_kelvin") {
                candidate.temperature_kelvin = value.get<float>();
            } else if (key == "grain_saturation") {
                candidate.grain_saturation = value.get<float>();
            } else if (key == "bond_body") {
                if (!DemBond::parseSeed(value, candidate.bond_body, error)) {
                    return false;
                }
            } else {
                throw std::runtime_error("unknown/read-only seed field: " + key);
            }
        }
        if (candidate.name.empty() || candidate.name.size() > 63 || !finite(candidate.bounds_min) ||
            !finite(candidate.bounds_max - candidate.bounds_min) || !finite(candidate.bounds_max) ||
            !finite(candidate.velocity) || candidate.bounds_min.x >= candidate.bounds_max.x ||
            candidate.bounds_min.y >= candidate.bounds_max.y ||
            candidate.bounds_min.z >= candidate.bounds_max.z) {
            throw std::runtime_error("seed source requires a name and finite positive-size bounds");
        }
        if (candidate.temperature_mode != "material" && candidate.temperature_mode != "domain" &&
            candidate.temperature_mode != "custom") {
            throw std::runtime_error("temperature_mode must be material|domain|custom");
        }
        if (!std::isfinite(candidate.temperature_kelvin) || candidate.temperature_kelvin < 1.0f ||
            candidate.temperature_kelvin > 3000.0f || !std::isfinite(candidate.grain_saturation) ||
            candidate.grain_saturation < 0.0f || candidate.grain_saturation > 1.0f) {
            throw std::runtime_error("seed Kelvin must be [1,3000] and saturation [0,1]");
        }
        if (!tryFindSubstance(candidate.substance)) {
            throw std::runtime_error("unknown seed substance: " + candidate.substance);
        }
    } catch (const std::exception &exception) {
        error = exception.what();
        return false;
    }
    source = std::move(candidate);
    error.clear();
    return true;
}

bool matterSeedSourcesUseObject(const SimulationGridDomainDesc &domain, const std::string &name) {
    return std::any_of(domain.matter_seed_sources.begin(), domain.matter_seed_sources.end(),
                       [&](const MatterSeedSource &source) {
                           return source.mesh_object == name && !name.empty();
                       });
}

bool validateMatterSeedSources(const SimulationGridDomainDesc &domain, std::string &error,
                               const std::vector<SimulationFlowSourceDesc> *flows,
                               int domain_index) {
    if (domain.matter_seed_sources_enabled &&
        (domain.fluid_seed_particles_per_cell < 1 || domain.fluid_seed_particles_per_cell > 64 ||
         domain.fluid_max_particles < 1)) {
        error = "seed recipe needs particles_per_cell in [1,64] and positive domain capacity";
        return false;
    }
    if (domain.matter_seed_sources_enabled &&
        (domain.type != SimulationDomainType::Matter || !simulationDomainHasLiquid(domain))) {
        error = "seed sources require an enabled Matter particle lane";
        return false;
    }
    std::set<uint64_t> ids;
    std::set<std::string> names, grains, mpm;
    for (const auto &source : domain.matter_seed_sources) {
        auto checked = source;
        if (!parseMatterSeedSource(nlohmann::json::object(), checked, error)) {
            return false;
        }
        if (source.id == 0 || !ids.insert(source.id).second || !names.insert(source.name).second) {
            error = "seed source IDs and names must be unique and IDs nonzero";
            return false;
        }
        if (!domain.matter_seed_sources_enabled || !source.enabled) {
            continue;
        }
        const auto *profile = tryFindSubstance(source.substance);
        if (source.bond_body.enabled &&
            (profile->default_constitutive_model != MatterConstitutiveModel::Granular ||
             profile->granular_transport != MatterGranularTransport::Dem ||
             domain.fluid_seed_settle)) {
            error = source.name + ": bonded bodies require a DEM material and Seed Settle off";
            return false;
        }
        if (source.grain_saturation > 0.0f &&
            (profile->default_constitutive_model != MatterConstitutiveModel::Granular ||
             profile->granular_transport != MatterGranularTransport::Dem ||
             !(profile->grain_water_capacity_fraction > 0.0f))) {
            error = source.name +
                    ": initial grain saturation requires a DEM material with water capacity";
            return false;
        }
        if (profile->default_constitutive_model == MatterConstitutiveModel::Granular &&
            profile->granular_transport == MatterGranularTransport::Dem) {
            grains.insert(profile->name);
        } else if (profile->default_constitutive_model == MatterConstitutiveModel::Granular ||
                   profile->default_constitutive_model == MatterConstitutiveModel::Elastic) {
            mpm.insert(profile->name);
        }
    }
    if (flows) {
        for (const auto &flow : *flows) {
            if (flow.domain_index != domain_index ||
                flow.phase == SimulationFlowSourceDesc::Phase::Gas) {
                continue;
            }
            const auto *profile = tryFindSubstance(flow.fluid_substance.empty()
                                                       ? domain.fluid_params.default_substance
                                                       : flow.fluid_substance);
            if (!profile) {
                error = "unknown emitter substance in the seed domain";
                return false;
            }
            if (profile->default_constitutive_model == MatterConstitutiveModel::Elastic ||
                (profile->default_constitutive_model == MatterConstitutiveModel::Granular &&
                 profile->granular_transport == MatterGranularTransport::Mpm)) {
                mpm.insert(profile->name);
            } else if (profile->default_constitutive_model == MatterConstitutiveModel::Granular) {
                grains.insert(profile->name);
            }
        }
    }
    if (grains.size() > 1 || mpm.size() > 1) {
        error = "one DEM and one MPM material are supported per domain; "
                "seed layers cannot silently share different constitutive parameters";
        return false;
    }
    if (!grains.empty()) {
        auto probe = domain.fluid_params.grain;
        probe.enabled = true;
        if (!validateMatterGrainDomain(domain, probe, error)) {
            return false;
        }
    }
    error.clear();
    return true;
}

float matterSeedBirthKelvin(const MatterSeedSource &source, const SimulationGridDomainDesc &domain,
                            float ambient_kelvin) {
    if (source.temperature_mode == "custom") {
        return source.temperature_kelvin;
    }
    if (source.temperature_mode == "domain") {
        return ambient_kelvin;
    }
    return substanceAmbientBirthKelvin(substanceTag(source.substance), domain.fluid_params,
                                       ambient_kelvin);
}

void writeMatterSeedSources(nlohmann::json &json, const SimulationGridDomainDesc &domain) {
    json["matter_seed_auto_capacity"] = domain.matter_seed_auto_capacity;
    json["matter_seed_memory_guard"] = domain.matter_seed_memory_guard;
    json["matter_seed_sources_enabled"] = domain.matter_seed_sources_enabled;
    json["matter_seed_sources"] = nlohmann::json::array();
    for (const auto &source : domain.matter_seed_sources) {
        json["matter_seed_sources"].push_back(matterSeedSourceToJson(source));
    }
}

void readMatterSeedSources(const nlohmann::json &json, SimulationGridDomainDesc &domain) {
    if (!json.contains("matter_seed_sources")) {
        return;
    }
    if (!json.at("matter_seed_sources").is_array()) {
        throw std::runtime_error("matter_seed_sources must be an array");
    }
    auto candidate = domain;
    candidate.matter_seed_auto_capacity = json.value("matter_seed_auto_capacity", true);
    candidate.matter_seed_memory_guard = json.value("matter_seed_memory_guard", true);
    candidate.matter_seed_sources_enabled =
        json.value("matter_seed_sources_enabled", !json.at("matter_seed_sources").empty());
    candidate.matter_seed_sources.clear();
    std::string error;
    for (const auto &value : json.at("matter_seed_sources")) {
        MatterSeedSource source;
        if (!parseMatterSeedSource(value, source, error, true)) {
            throw std::runtime_error(error);
        }
        candidate.matter_seed_sources.push_back(std::move(source));
    }
    if (!validateMatterSeedSources(candidate, error)) {
        throw std::runtime_error(error);
    }
    domain.matter_seed_sources = std::move(candidate.matter_seed_sources);
    domain.matter_seed_sources_enabled = candidate.matter_seed_sources_enabled;
    domain.matter_seed_auto_capacity = candidate.matter_seed_auto_capacity;
    domain.matter_seed_memory_guard = candidate.matter_seed_memory_guard;
    domain.matter_seed_sources_applied_hash =
        domain.matter_seed_sources_enabled && domain.fluid_reseed_on_reset
            ? matterSeedRecipeHash(domain)
            : 0;
}

bool applyMatterSeedSources(SimulationGridDomainDesc &domain, SimulationGridDomainState &state,
                            float ambient_kelvin, const MatterSeedMeshResolver &meshes,
                            const std::vector<SimulationFlowSourceDesc> *flows, int domain_index) {
    std::string error;
    const auto fail = [&](const std::string &message) {
        domain.fluid_pending_seed = false;
        state.seed_stats = SimulationSeedStats{};
        state.seed_stats.error = message;
        return false;
    };
    try {
        if (!domain.matter_seed_sources_enabled) {
            return fail("layered initial-source recipe is not active");
        }
        if (!validateMatterSeedSources(domain, error, flows, domain_index)) {
            return fail(error);
        }
        FluidParticles initial;
        const std::vector<SimulationFlowSourceDesc> empty_flows;
        const auto ownership =
            matterGrainOwnership(domain, flows ? *flows : empty_flows, domain_index, false);
        if (ownership.wanted && !ownership.enabled) {
            return fail(ownership.blockers.empty() ? "grain ownership is not ready"
                                                   : ownership.blockers.front());
        }
        auto target_grain = domain.fluid_params.grain;
        target_grain.enabled = ownership.enabled;
        const auto *liquid =
            ownership.liquid.empty() ? nullptr : tryFindSubstance(ownership.liquid);
        if (ownership.enabled) {
            const auto *grain = tryFindSubstance(ownership.substance);
            applyMatterGrainSubstance(target_grain, *grain, liquid, ownership.wet_grains);
        }
        const auto &grid = liquidGrid(state);
        if (!state.valid || grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0 ||
            !std::isfinite(grid.voxel_size) || grid.voxel_size <= 0.0f) {
            return fail("seed sources require a valid particle grid");
        }
        auto plan = planMatterSeeds(domain, meshes, grid, ambient_kelvin);
        if (!checkMatterSeedMemory(plan, error) && domain.matter_seed_memory_guard) {
            return fail(error);
        }
        const auto capacity = domain.matter_seed_auto_capacity
                                  ? plan.at("suggested_capacity").get<std::size_t>()
                                  : domain.fluid_max_particles;
        const int ppc = std::max(1, domain.fluid_seed_particles_per_cell);
        std::size_t requested = 0;
        std::vector<std::pair<Vec3, Vec3>> regions;
        for (const auto &source : domain.matter_seed_sources) {
            if (!source.enabled) {
                continue;
            }
            auto params = domain.fluid_params;
            params.default_substance = source.substance;
            params.grain = target_grain;
            const auto *profile = tryFindSubstance(source.substance);
            const float kelvin = matterSeedBirthKelvin(source, domain, ambient_kelvin);
            if (!std::isfinite(kelvin) || kelvin < 1.0f || kelvin > 3000.0f) {
                return fail(source.name + ": resolved birth Kelvin must be in [1,3000]");
            }
            const auto model = substanceBirthModel(substanceTag(source.substance), params, kelvin);
            const bool dem = model == MatterConstitutiveModel::Granular &&
                             profile->granular_transport == MatterGranularTransport::Dem;
            Vec3 low = source.bounds_min, high = source.bounds_max;
            std::shared_ptr<const MeshVolumeTest> mesh;
            std::unique_ptr<MeshVolumeCellMask> mask;
            if (!source.mesh_object.empty()) {
                mesh = meshes ? meshes(source.mesh_object) : nullptr;
                if (!mesh || mesh->empty()) {
                    return fail(source.name + ": mesh is missing or empty");
                }
                mask = std::make_unique<MeshVolumeCellMask>(*mesh, grid);
                if (mask->empty()) {
                    return fail(source.name + ": mesh has no inside in the domain");
                }
                low = mesh->boundsMin();
                high = mesh->boundsMax();
            }
            low = Vec3::max(low, grid.origin);
            high = Vec3::min(high, gridBoundsMax(grid));
            if (low.x >= high.x || low.y >= high.y || low.z >= high.z) {
                return fail(source.name + ": seed region does not overlap the domain");
            }
            const auto cells = countSeedBoxCells(grid, low, high, mask.get());
            for (const auto &region : regions) {
                if (std::max(low.x, region.first.x) < std::min(high.x, region.second.x) &&
                    std::max(low.y, region.first.y) < std::min(high.y, region.second.y) &&
                    std::max(low.z, region.first.z) < std::min(high.z, region.second.z)) {
                    return fail(source.name +
                                ": seed bounds overlap another layer; "
                                "interpenetrating initial materials are not supported");
                }
            }
            regions.emplace_back(low, high);
            if (cells == 0) {
                return fail(source.name + ": seed region has no cells in the domain");
            }
            const auto remaining = capacity - initial.size();
            const auto begin = initial.size();
            if (dem) {
                params.grain.enabled = true;
                applyMatterGrainSubstance(params.grain, *profile, liquid,
                                          liquid || source.grain_saturation > 0.0f);
                const auto result = seedMatterGrainBox(initial, grid, low, high, params, kelvin,
                                                       remaining, mask.get());
                if (result.truncated) {
                    return fail(source.name + ": DEM layer exceeds the remaining " +
                                std::to_string(remaining) + " particle slots (domain capacity " +
                                std::to_string(capacity) +
                                "). Increase capacity or reduce the layer volume. "
                                "Existing particles were preserved.");
                }
            } else {
                if (cells > remaining / static_cast<std::size_t>(ppc)) {
                    return fail(source.name + ": region budget requires " + std::to_string(cells) +
                                " cells x " + std::to_string(ppc) + " particles per voxel; only " +
                                std::to_string(remaining) +
                                " particle slots remain (domain capacity " +
                                std::to_string(capacity) +
                                "). Increase capacity, reduce layer volume, or lower "
                                "particles per voxel. Existing particles were preserved.");
                }
                seedBox(initial, grid, low, high, ppc,
                        static_cast<uint32_t>(source.id) * 2654435761u, remaining, kelvin,
                        mask.get());
                // Legacy seedBox fills whole boundary cells. Source regions use
                // exact bounds, preventing a shared boundary cell doubling a layer.
                for (std::size_t p = initial.size(); p > begin; --p) {
                    if (!inside(initial.position[p - 1], low, high)) {
                        initial.removeSwap(p - 1);
                    }
                }
            }
            if (initial.size() == begin) {
                return fail(source.name + ": seed produced no particles");
            }
            for (std::size_t p = begin; p < initial.size(); ++p) {
                initial.substance_tag[p] = substanceTag(source.substance);
                initial.velocity[p] = source.velocity;
                if (dem) {
                    initMatterGrainBirthWater(initial, p, source.grain_saturation, params.grain);
                }
            }
            initializeMatterBirthModels(initial, begin, params);
            if (dem) {
                ensureMatterGrainRestMasses(initial, params.grain, profile, false);
            }
            if (!DemBond::bindSeed(initial, begin, params.grain, source.bond_body,
                    domain.fluid_params.mixed_working_set_budget_bytes, error)) {
                return fail(source.name + ": " + error);
            }
            requested += initial.size() - begin;
        }
        ensureFluidParticleRestMasses(initial, domainSubstance(domain.fluid_params),
                                      grid.voxel_size, ppc, false);
        const auto applied_hash = matterSeedRecipeHash(domain);
        state.particles = std::move(initial);
        domain.fluid_max_particles = capacity;
        state.foam.clear();
        state.fluid_stats = APICSolverStats{};
        state.foam_stats = FoamStats{};
        state.thermal_stats = ThermalLiquidStats{};
        state.settle_stats = SimulationSeedSettleStats{};
        state.seed_stats = SimulationSeedStats{};
        state.seed_stats.ran = true;
        state.seed_stats.requested = requested;
        state.seed_stats.emitted = state.particles.size();
        state.seed_stats.top_y = -std::numeric_limits<float>::max();
        for (const auto &point : state.particles.position) {
            state.seed_stats.top_y = std::max(state.seed_stats.top_y, point.y);
        }
        state.seed_stats.region_top_y = state.seed_stats.top_y;
        if (state.particles.empty()) {
            state.seed_stats.top_y = 0.0f;
            state.seed_stats.region_top_y = 0.0f;
        }
        domain.fluid_params.grain = target_grain;
        domain.matter_seed_sources_applied_hash = applied_hash;
        state.settle_stats.pending = domain.fluid_seed_settle && !state.particles.empty();
        domain.fluid_params.particles_per_cell = ppc;
        domain.fluid_pending_seed = false;
        ++state.version;
        return true;
    } catch (const std::exception &exception) {
        return fail(std::string("seed generation failed: ") + exception.what());
    }
}

} // namespace RayTrophiSim::Fluid
