#include "ParticleSimulation.h"
#include "Fluid/DemBondNetwork.h"
#include "Fluid/MatterGrain.h"
#include "Fluid/MatterGrainCoupling.h"
#include <set>
#include <stdexcept>

namespace RayTrophiSim {
nlohmann::json ParticleSimulationSystem::demBondOperation(
    const std::string& name, const nlohmann::json& request) {
    namespace B = Fluid::DemBond;
    if (!request.is_object()) {
        throw std::runtime_error("DEM body request must be an object");
    }
    const std::string action = request.value("action", "inspect");
    std::set<std::string> allowed{"action"};
    if (action == "bind") {
        allowed.insert({"particle_ids", "all_current_grains", "material", "reach_ratio",
                        "section_ratio"});
    } else if (action == "remove") {
        allowed.insert("body");
    } else if (action != "inspect" && action != "snapshot") {
        throw std::runtime_error("DEM body action must be inspect, snapshot, bind or remove");
    }
    for (auto it = request.begin(); it != request.end(); ++it) {
        if (allowed.count(it.key()) == 0) {
            throw std::runtime_error("unknown DEM body field: " + it.key());
        }
    }
    for (std::size_t domain = 0; domain < grid_domains_.size(); ++domain) {
        const auto& desc = grid_domains_[domain];
        if (desc.name != name) {
            continue;
        }
        if (desc.type != SimulationDomainType::Matter || domain >= grid_domain_states_.size() ||
            !grid_domain_states_[domain].valid) {
            throw std::runtime_error("DEM body requires a synchronized Matter domain");
        }
        auto& particles = grid_domain_states_[domain].particles;
        if (action == "snapshot") {
            return B::encode(particles.dem_bonds);
        }
        if (action == "inspect") {
            return B::summary(particles.dem_bonds);
        }
        auto candidate = particles.dem_bonds;
        std::string error;
        uint64_t created = 0;
        if (action == "remove") {
            if (!request.contains("body") || !request["body"].is_number_unsigned() ||
                !B::removeBody(candidate, request["body"].get<uint64_t>(), error)) {
                throw std::runtime_error(error.empty() ? "body must be a positive uint64" : error);
            }
            created = request["body"].get<uint64_t>();
            particles.dem_bonds = std::move(candidate);
        } else {
            std::vector<std::size_t> liquid, grains, held;
            const auto& params = desc.fluid_params;
            if (!params.grain.enabled || !Fluid::partitionMatterGrainOwners(particles, params,
                    params.granular_enabled, params.grain.wet_grains,
                    liquid, grains, held, error)) {
                throw std::runtime_error(error.empty() ? "DEM transport is not enabled" : error);
            }
            std::set<uint64_t> grain_ids;
            for (std::size_t i : grains) {
                grain_ids.insert(particles.particle_id[i]);
            }
            std::vector<uint64_t> selected;
            if (request.contains("all_current_grains") &&
                !request["all_current_grains"].is_boolean()) {
                throw std::runtime_error("all_current_grains must be boolean");
            }
            const bool all = request.value("all_current_grains", false);
            if (all && request.contains("particle_ids")) {
                throw std::runtime_error("select particle_ids or all_current_grains, not both");
            }
            if (all) {
                selected.assign(grain_ids.begin(), grain_ids.end());
            } else {
                if (!request.contains("particle_ids") || !request["particle_ids"].is_array()) {
                    throw std::runtime_error("particle_ids array is required");
                }
                for (const auto& id : request["particle_ids"]) {
                    if (!id.is_number_unsigned() || grain_ids.count(id.get<uint64_t>()) == 0) {
                        throw std::runtime_error("every particle_id must select a current DEM grain");
                    }
                    selected.push_back(id.get<uint64_t>());
                }
            }
            const auto& m = request.at("material");
            if (!m.is_object() || m.size() != 4 || !m.contains("young_pa") ||
                !m.contains("poisson") || !m.contains("tensile_strength_pa") ||
                !m.contains("shear_strength_pa")) {
                throw std::runtime_error("material requires young_pa, poisson, "
                    "tensile_strength_pa and shear_strength_pa");
            }
            for (auto it = m.begin(); it != m.end(); ++it) {
                if (!it->is_number()) {
                    throw std::runtime_error("bond material values must be numbers");
                }
            }
            for (const char* key : {"reach_ratio", "section_ratio"}) {
                if (request.contains(key) && !request.at(key).is_number()) {
                    throw std::runtime_error(std::string(key) + " must be numeric");
                }
            }
            B::Material material{m.at("young_pa").get<double>(), m.at("poisson").get<double>(),
                1.0, m.at("tensile_strength_pa").get<double>(),
                m.at("shear_strength_pa").get<double>()};
            if (!B::bindBody(particles, params.grain, selected, material,
                    request.value("reach_ratio", 1.25), request.value("section_ratio", 0.5),
                    params.mixed_working_set_budget_bytes, created, error)) {
                throw std::runtime_error(error);
            }
        }
        auto result = B::summary(particles.dem_bonds);
        result["body"] = created;
        return result;
    }
    throw std::runtime_error("Matter domain not found: " + name);
}
} // namespace RayTrophiSim
