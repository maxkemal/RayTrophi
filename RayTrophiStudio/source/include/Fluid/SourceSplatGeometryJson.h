#pragma once

#include "Fluid/SourceSplatGeometry.h"
#include "json.hpp"
#include <stdexcept>
#include <map>

namespace RayTrophiSim::Fluid {

inline void to_json(nlohmann::json& j, const SourceSplatGeometry& value) {
    j = {{"mode", value.mode}, {"objects", value.objects},
         {"source_materials", value.source_materials},
         {"near_distance", value.near_distance}, {"mesh_budget", value.mesh_budget}};
}

inline void from_json(const nlohmann::json& j, SourceSplatGeometry& value) {
    if (!j.is_object()) {
        throw std::invalid_argument("splat_geometry must be an object");
    }
    for (const auto& item : j.items()) {
        if (item.key() != "mode" && item.key() != "objects" &&
            item.key() != "source_materials" && item.key() != "near_distance" &&
            item.key() != "mesh_budget") {
            throw std::invalid_argument("unknown splat_geometry field: " + item.key());
        }
    }
    if (j.contains("mode")) {
        j.at("mode").get_to(value.mode);
    }
    if (j.contains("objects")) {
        j.at("objects").get_to(value.objects);
    }
    if (j.contains("source_materials")) {
        j.at("source_materials").get_to(value.source_materials);
    }
    if (j.contains("near_distance")) {
        j.at("near_distance").get_to(value.near_distance);
    }
    if (j.contains("mesh_budget")) {
        const auto& budget = j.at("mesh_budget");
        if (!budget.is_number_integer() || budget.get<double>() < 0.0 ||
            budget.get<double>() > 4294967295.0) {
            throw std::invalid_argument("mesh_budget must be an unsigned 32-bit integer");
        }
        budget.get_to(value.mesh_budget);
    }
}

inline SourceSplatGeometry readSourceSplatGeometry(const nlohmann::json& source) {
    SourceSplatGeometry result;
    if (source.contains("splat_geometry")) {
        source.at("splat_geometry").get_to(result);
    } else {
        result.mode = "domain";
    }
    return result;
}

inline std::map<std::string, SourceSplatGeometry> readSubstanceSplatGeometries(
    const nlohmann::json& domain) {
    std::map<std::string, SourceSplatGeometry> result;
    if (domain.contains("fluid_substance_splat_geometry")) {
        domain.at("fluid_substance_splat_geometry").get_to(result);
    }
    return result;
}

} // namespace RayTrophiSim::Fluid
