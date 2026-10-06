#include "RtMatterModels.h"
#include "Fluid/GranularReference.h"

#include <cmath>
#include <initializer_list>
#include <stdexcept>

namespace rtapi {
namespace {

using Json = nlohmann::json;

void keys(const Json& object, std::initializer_list<const char*> allowed) {
    if (!object.is_object()) {
        throw std::runtime_error("grain_reference_invalid_parameter: object required");
    }
    for (auto it = object.begin(); it != object.end(); ++it) {
        bool found = false;
        for (const char* key : allowed) {
            found = found || it.key() == key;
        }
        if (!found) {
            throw std::runtime_error("grain_reference_invalid_parameter: unknown key " + it.key());
        }
    }
}

double number(const Json& value) {
    if (!value.is_number() || !std::isfinite(value.get<double>())) {
        throw std::runtime_error("grain_reference_invalid_parameter: finite number required");
    }
    return value.get<double>();
}

float scalar(const Json& value) {
    const float result = static_cast<float>(number(value));
    if (!std::isfinite(result)) {
        throw std::runtime_error("grain_reference_invalid_parameter: float overflow");
    }
    return result;
}

Vec3 vector(const Json& value) {
    if (!value.is_array() || value.size() != 3) {
        throw std::runtime_error("grain_reference_invalid_parameter: vector needs three numbers");
    }
    return Vec3(scalar(value[0]), scalar(value[1]), scalar(value[2]));
}

Json jsonVector(const Vec3& value) {
    return Json::array({value.x, value.y, value.z});
}

} // namespace

nlohmann::json runGrainReferenceProbe(const nlohmann::json& params) {
    using namespace RayTrophiSim::Fluid::Granular;
    keys(params, {"bodies", "gravity", "plane", "duration_s", "maximum_dt_s",
        "sample_interval_s", "contact"});
    ReferenceConfig config;
    if (!params.contains("bodies") || !params["bodies"].is_array() ||
        params["bodies"].empty() || params["bodies"].size() > 64) {
        throw std::runtime_error("grain_reference_invalid_parameter: 1..64 bodies required");
    }
    for (const auto& item : params["bodies"]) {
        keys(item, {"id", "position", "velocity", "angular_velocity", "radius_m",
            "mass_kg", "saturation"});
        if (!item.contains("position") || !item.contains("radius_m") ||
            !item.contains("mass_kg")) {
            throw std::runtime_error(
                "grain_reference_invalid_parameter: position/radius_m/mass_kg required");
        }
        if (!item.contains("id") ||
            (!item["id"].is_number_unsigned() && !item["id"].is_number_integer()) ||
            number(item["id"]) <= 0.0 || number(item["id"]) > 9007199254740991.0) {
            throw std::runtime_error(
                "grain_reference_invalid_parameter: exact positive id required");
        }
        ContactBody body;
        body.id = item["id"].get<uint64_t>();
        body.position = vector(item.at("position"));
        body.radius_m = scalar(item.at("radius_m"));
        body.mass_kg = scalar(item.at("mass_kg"));
        if (item.contains("velocity")) {
            body.velocity = vector(item["velocity"]);
        }
        if (item.contains("angular_velocity")) {
            body.angular_velocity = vector(item["angular_velocity"]);
        }
        if (item.contains("saturation")) {
            body.saturation = scalar(item["saturation"]);
        }
        config.bodies.push_back(body);
    }
    if (params.contains("gravity")) {
        config.gravity = vector(params["gravity"]);
    }
    if (params.contains("duration_s")) {
        config.duration_s = number(params["duration_s"]);
    }
    if (params.contains("maximum_dt_s")) {
        config.maximum_dt_s = number(params["maximum_dt_s"]);
    }
    if (params.contains("sample_interval_s")) {
        config.sample_interval_s = number(params["sample_interval_s"]);
    }
    if (params.contains("plane")) {
        config.plane_enabled = !params["plane"].is_null();
        if (config.plane_enabled) {
            keys(params["plane"], {"normal", "offset_m"});
            if (!params["plane"].contains("normal") || !params["plane"].contains("offset_m")) {
                throw std::runtime_error(
                    "grain_reference_invalid_parameter: plane normal/offset_m required");
            }
            config.plane_normal = vector(params["plane"].at("normal"));
            config.plane_offset_m = scalar(params["plane"].at("offset_m"));
        }
    }
    if (params.contains("contact")) {
        const auto& patch = params["contact"];
        keys(patch, {"normal_stiffness_n_m", "normal_damping_n_s_m",
            "tangential_stiffness_n_m", "tangential_damping_n_s_m", "dry_friction",
            "saturated_friction", "rolling_friction"});
        const auto set = [&patch](const char* key, float& target) {
            if (patch.contains(key)) {
                target = scalar(patch[key]);
            }
        };
        set("normal_stiffness_n_m", config.contact.normal_stiffness_n_m);
        set("normal_damping_n_s_m", config.contact.normal_damping_n_s_m);
        set("tangential_stiffness_n_m", config.contact.tangential_stiffness_n_m);
        set("tangential_damping_n_s_m", config.contact.tangential_damping_n_s_m);
        set("dry_friction", config.contact.dry_friction);
        set("saturated_friction", config.contact.saturated_friction);
        set("rolling_friction", config.contact.rolling_friction);
    }
    ReferenceReport report;
    std::string error;
    if (!runGrainReference(config, report, error)) {
        throw std::runtime_error("grain_reference_rejected: " + error);
    }
    Json frames = Json::array();
    for (const auto& frame : report.frames) {
        Json bodies = Json::array();
        for (const auto& body : frame.bodies) {
            bodies.push_back({{"id", body.id}, {"position", jsonVector(body.position)},
                {"velocity", jsonVector(body.velocity)},
                {"angular_velocity", jsonVector(body.angular_velocity)},
                {"radius_m", body.radius_m}, {"mass_kg", body.mass_kg},
                {"saturation", body.saturation}});
        }
        frames.push_back({{"seconds", frame.seconds}, {"bodies", bodies},
            {"mass_kg", frame.mass_kg}, {"kinetic_energy_j", frame.kinetic_energy_j},
            {"momentum_kg_m_s", jsonVector(frame.momentum)},
            {"angular_momentum_kg_m2_s", jsonVector(frame.angular_momentum)}});
    }
    return {{"ok", true}, {"backend", "cpu_grain_reference"},
        {"scene_mutated", false}, {"production_dem_enabled", false},
        {"micro_steps", report.micro_steps}, {"candidate_pairs", report.candidate_pairs},
        {"maximum_micro_dt_s", report.maximum_micro_dt_s},
        {"contact_evaluations", report.contact_evaluations},
        {"maximum_overlap_ratio", report.maximum_overlap_ratio}, {"frames", frames}};
}

} // namespace rtapi
