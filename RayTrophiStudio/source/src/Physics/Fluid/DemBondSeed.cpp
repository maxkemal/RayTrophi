#include "Fluid/DemBondSeed.h"
#include "Fluid/FluidParticles.h"
#include <cmath>
#include <stdexcept>

namespace RayTrophiSim::Fluid::DemBond {
nlohmann::json seedJson(const SeedSettings& s) {
    return {{"enabled", s.enabled}, {"young_pa", s.material.young_pa},
        {"poisson", s.material.poisson}, {"tensile_strength_pa", s.material.tensile_strength_pa},
        {"shear_strength_pa", s.material.shear_strength_pa},
        {"reach_ratio", s.reach_ratio}, {"section_ratio", s.section_ratio}};
}

bool parseSeed(const nlohmann::json& patch, SeedSettings& settings, std::string& error) {
    error.clear();
    try {
        auto fields = seedJson(settings);
        if (!patch.is_object()) {
            throw std::runtime_error("bond_body must be an object");
        }
        for (auto it = patch.begin(); it != patch.end(); ++it) {
            if (!fields.contains(it.key())) {
                throw std::runtime_error("unknown bond_body field: " + it.key());
            }
            if ((it.key() == "enabled" && !it->is_boolean()) ||
                (it.key() != "enabled" && !it->is_number())) {
                throw std::runtime_error("invalid bond_body field type: " + it.key());
            }
            fields[it.key()] = it.value();
        }
        SeedSettings next;
        next.enabled = fields.at("enabled").get<bool>();
        next.material = {fields.at("young_pa").get<double>(), fields.at("poisson").get<double>(),
            1.0, fields.at("tensile_strength_pa").get<double>(),
            fields.at("shear_strength_pa").get<double>()};
        next.reach_ratio = fields.at("reach_ratio").get<double>();
        next.section_ratio = fields.at("section_ratio").get<double>();
        if (!std::isfinite(next.reach_ratio) || next.reach_ratio < 1.0 || next.reach_ratio > 2.0 ||
            !std::isfinite(next.section_ratio) || next.section_ratio <= 0.0 ||
            next.section_ratio > 1.0 ||
            !gpuMaterialRepresentable(next.material) ||
            stableStep(next.material, 1.0, 1.0, 1.0, 1.0, 1.0, 1, error) <= 0.0) {
            throw std::runtime_error("invalid bond material, reach [1,2] or section ratio (0,1]");
        }
        settings = next;
        return true;
    } catch (const std::exception& e) {
        error = std::string("bond_body: ") + e.what();
        return false;
    }
}

bool bindSeed(FluidParticles& p, std::size_t begin, const MatterGrainParams& params,
              const SeedSettings& settings, uint64_t budget, std::string& error) {
    if (!settings.enabled) {
        return true;
    }
    if (begin > p.size() || p.particle_id.size() != p.size()) {
        error = "bond_body: invalid seed population";
        return false;
    }
    const std::vector<uint64_t> ids(p.particle_id.begin() + begin, p.particle_id.end());
    uint64_t body = 0;
    return bindBody(p, params, ids, settings.material, settings.reach_ratio,
                    settings.section_ratio, budget, body, error);
}
}
