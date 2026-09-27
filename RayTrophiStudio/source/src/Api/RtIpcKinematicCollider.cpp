#include "RtIpcKinematicCollider.h"

#include "Api/RtApi.h"

#include <algorithm>
#include <stdexcept>

using json = nlohmann::json;

namespace {

uint64_t requireInt(const json& params, const char* key) {
    if (!params.contains(key) ||
        (!params[key].is_number_unsigned() &&
         !params[key].is_number_integer())) {
        throw std::runtime_error(
            std::string("missing or invalid int param: ") + key);
    }
    const long long value = params[key].get<long long>();
    if (value <= 0) {
        throw std::runtime_error(
            std::string("invalid positive int param: ") + key);
    }
    return static_cast<uint64_t>(value);
}

std::string requireString(const json& params, const char* key) {
    if (!params.contains(key) || !params[key].is_string()) {
        throw std::runtime_error(
            std::string("missing or invalid string param: ") + key);
    }
    return params[key].get<std::string>();
}

float optionalFloat(const json& params, const char* key, float fallback) {
    if (!params.contains(key)) {
        return fallback;
    }
    if (!params[key].is_number()) {
        throw std::runtime_error(std::string("invalid number param: ") + key);
    }
    return params[key].get<float>();
}

Vec3 readVec3(const json& value, const char* key) {
    if (!value.is_array() || value.size() != 3) {
        throw std::runtime_error(std::string("invalid Vec3 param: ") + key);
    }
    return Vec3(
        value[0].get<float>(),
        value[1].get<float>(),
        value[2].get<float>());
}

json vec3ToJson(const Vec3& value) {
    return json::array({value.x, value.y, value.z});
}

json resultToJson(const rtapi::Result& result) {
    return result.ok ? json(true) : json{{"__error", result.error}};
}

json proxyToJson(const RayTrophiSim::KinematicProxyDesc& proxy) {
    return json{
        {"id", proxy.id},
        {"name", proxy.name},
        {"bone", proxy.bone},
        {"shape", RayTrophiSim::kinematicProxyShapeName(proxy.shape)},
        {"enabled", proxy.enabled},
        {"local_position", vec3ToJson(proxy.local_position)},
        {"local_rotation", vec3ToJson(proxy.local_rotation_degrees)},
        {"local_axis", vec3ToJson(proxy.local_axis)},
        {"radius", proxy.radius},
        {"half_length", proxy.half_length},
        {"half_extents", vec3ToJson(proxy.half_extents)}};
}

json setToJson(const RayTrophiSim::KinematicProxySet& set) {
    json proxies = json::array();
    for (const auto& proxy : set.proxies) {
        proxies.push_back(proxyToJson(proxy));
    }
    return json{
        {"id", set.id},
        {"name", set.name},
        {"target_character", set.target_character},
        {"target_node_id", set.target_node_id},
        {"enabled", set.enabled},
        {"viewport_visible", set.viewport_visible},
        {"consumer_mask", set.consumer_mask},
        {"friction", set.friction},
        {"restitution", set.restitution},
        {"thickness", set.thickness},
        {"revision", set.revision},
        {"proxies", std::move(proxies)}};
}

json sampleToJson(const RayTrophiSim::KinematicProxySample& sample) {
    json matrix = json::array();
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            matrix.push_back(sample.world_transform.m[row][column]);
        }
    }
    return json{
        {"set_id", sample.set_id},
        {"proxy_id", sample.proxy_id},
        {"set_name", sample.set_name},
        {"proxy_name", sample.proxy_name},
        {"target_character", sample.target_character},
        {"bone", sample.bone},
        {"shape", RayTrophiSim::kinematicProxyShapeName(sample.shape)},
        {"consumer_mask", sample.consumer_mask},
        {"friction", sample.friction},
        {"restitution", sample.restitution},
        {"resolved", sample.resolved},
        {"unresolved_reason", sample.unresolved_reason},
        {"world_transform", std::move(matrix)},
        {"center", vec3ToJson(sample.center)},
        {"capsule_start", vec3ToJson(sample.capsule_start)},
        {"capsule_end", vec3ToJson(sample.capsule_end)},
        {"radius", sample.radius},
        {"half_extents", vec3ToJson(sample.half_extents)},
        {"linear_velocity", vec3ToJson(sample.linear_velocity)},
        {"angular_velocity", vec3ToJson(sample.angular_velocity)},
        {"velocity_valid", sample.velocity_valid}};
}

void patchSet(const json& params, RayTrophiSim::KinematicProxySet& set) {
    if (params.contains("name")) {
        set.name = params["name"].get<std::string>();
    }
    if (params.contains("target_character")) {
        set.target_character = params["target_character"].get<std::string>();
    }
    if (params.contains("target_node_id")) {
        set.target_node_id = params["target_node_id"].get<std::string>();
    }
    if (params.contains("enabled")) {
        set.enabled = params["enabled"].get<bool>();
    }
    if (params.contains("viewport_visible")) {
        set.viewport_visible = params["viewport_visible"].get<bool>();
    }
    if (params.contains("consumer_mask")) {
        set.consumer_mask = params["consumer_mask"].get<uint32_t>();
    }
    if (params.contains("friction")) {
        set.friction = params["friction"].get<float>();
    }
    if (params.contains("restitution")) {
        set.restitution = params["restitution"].get<float>();
    }
    if (params.contains("thickness")) {
        set.thickness = params["thickness"].get<float>();
    }
}

void patchProxy(const json& params, RayTrophiSim::KinematicProxyDesc& proxy) {
    if (params.contains("name")) {
        proxy.name = params["name"].get<std::string>();
    }
    if (params.contains("bone")) {
        proxy.bone = params["bone"].get<std::string>();
    }
    if (params.contains("shape")) {
        const std::string shape = params["shape"].get<std::string>();
        if (!RayTrophiSim::parseKinematicProxyShape(shape, proxy.shape)) {
            throw std::runtime_error("shape must be sphere, capsule, or box");
        }
    }
    if (params.contains("enabled")) {
        proxy.enabled = params["enabled"].get<bool>();
    }
    if (params.contains("local_position")) {
        proxy.local_position = readVec3(
            params["local_position"], "local_position");
    }
    if (params.contains("local_rotation")) {
        proxy.local_rotation_degrees = readVec3(
            params["local_rotation"], "local_rotation");
    }
    if (params.contains("local_axis")) {
        proxy.local_axis = readVec3(params["local_axis"], "local_axis");
    }
    if (params.contains("radius")) {
        proxy.radius = params["radius"].get<float>();
    }
    if (params.contains("half_length")) {
        proxy.half_length = params["half_length"].get<float>();
    }
    if (params.contains("half_extents")) {
        proxy.half_extents = readVec3(
            params["half_extents"], "half_extents");
    }
}

} // namespace

bool dispatchKinematicColliderIpc(
    const std::string& method,
    const json& params,
    const RtIpcKinematicColliderEnqueue& enqueue,
    json& out_result) {
    if (method == "physics.collider.proxy_set.list") {
        out_result = enqueue([](UIContext&) {
            std::vector<RayTrophiSim::KinematicProxySet> sets;
            const rtapi::Result result = rtapi::listKinematicProxySets(sets);
            if (!result.ok) {
                return json{{"__error", result.error}};
            }
            json out = json::array();
            for (const auto& set : sets) {
                out.push_back(setToJson(set));
            }
            return out;
        });
        return true;
    }
    if (method == "physics.collider.proxy_set.get") {
        const uint64_t set_id = requireInt(params, "set_id");
        out_result = enqueue([set_id](UIContext&) {
            RayTrophiSim::KinematicProxySet set;
            const rtapi::Result result = rtapi::getKinematicProxySet(set_id, set);
            return result.ok
                ? setToJson(set)
                : json{{"__error", result.error}};
        });
        return true;
    }
    if (method == "physics.collider.proxy_set.create") {
        const json patch = params;
        requireString(params, "name");
        requireString(params, "target_character");
        out_result = enqueue([patch](UIContext&) {
            RayTrophiSim::KinematicProxySet set;
            patchSet(patch, set);
            RayTrophiSim::KinematicProxySet created;
            const rtapi::Result result =
                rtapi::createKinematicProxySet(set, created);
            return result.ok
                ? setToJson(created)
                : json{{"__error", result.error}};
        });
        return true;
    }
    if (method == "physics.collider.proxy_set.set") {
        const uint64_t set_id = requireInt(params, "set_id");
        const json patch = params;
        out_result = enqueue([set_id, patch](UIContext&) {
            RayTrophiSim::KinematicProxySet set;
            rtapi::Result result = rtapi::getKinematicProxySet(set_id, set);
            if (!result.ok) {
                return resultToJson(result);
            }
            patchSet(patch, set);
            return resultToJson(rtapi::updateKinematicProxySet(set_id, set));
        });
        return true;
    }
    if (method == "physics.collider.proxy_set.delete") {
        const uint64_t set_id = requireInt(params, "set_id");
        out_result = enqueue([set_id](UIContext&) {
            return resultToJson(rtapi::removeKinematicProxySet(set_id));
        });
        return true;
    }
    if (method == "physics.collider.proxy_set.auto_fit") {
        const uint64_t set_id = requireInt(params, "set_id");
        RayTrophiSim::KinematicAutoFitOptions options;
        if (params.contains("replace_existing")) {
            options.replace_existing = params["replace_existing"].get<bool>();
        }
        if (params.contains("weighted_bones_only")) {
            options.weighted_bones_only =
                params["weighted_bones_only"].get<bool>();
        }
        if (params.contains("include_detail_bones")) {
            options.include_detail_bones =
                params["include_detail_bones"].get<bool>();
        }
        options.radius_fraction = optionalFloat(
            params, "radius_fraction", options.radius_fraction);
        options.minimum_radius = optionalFloat(
            params, "minimum_radius", options.minimum_radius);
        options.maximum_radius = optionalFloat(
            params, "maximum_radius", options.maximum_radius);
        options.minimum_bone_length = optionalFloat(
            params, "minimum_bone_length", options.minimum_bone_length);
        if (params.contains("maximum_proxies")) {
            options.maximum_proxies = params["maximum_proxies"].get<uint32_t>();
        }
        out_result = enqueue([set_id, options](UIContext&) {
            uint32_t count = 0;
            const rtapi::Result result =
                rtapi::autoFitKinematicProxySet(set_id, options, count);
            return result.ok
                ? json{{"created_count", count}}
                : json{{"__error", result.error}};
        });
        return true;
    }
    if (method == "physics.collider.proxy.set") {
        const uint64_t set_id = requireInt(params, "set_id");
        const json patch = params;
        out_result = enqueue([set_id, patch](UIContext&) {
            RayTrophiSim::KinematicProxyDesc proxy;
            if (patch.contains("proxy_id")) {
                const uint64_t proxy_id = requireInt(patch, "proxy_id");
                RayTrophiSim::KinematicProxySet set;
                rtapi::Result result = rtapi::getKinematicProxySet(set_id, set);
                if (!result.ok) {
                    return resultToJson(result);
                }
                const auto it = std::find_if(
                    set.proxies.begin(), set.proxies.end(),
                    [proxy_id](const RayTrophiSim::KinematicProxyDesc& value) {
                        return value.id == proxy_id;
                    });
                if (it == set.proxies.end()) {
                    return json{{"__error", "unknown_proxy"}};
                }
                proxy = *it;
            }
            patchProxy(patch, proxy);
            RayTrophiSim::KinematicProxyDesc stored;
            const rtapi::Result result =
                rtapi::setKinematicProxy(set_id, proxy, stored);
            return result.ok
                ? proxyToJson(stored)
                : json{{"__error", result.error}};
        });
        return true;
    }
    if (method == "physics.collider.proxy.delete") {
        const uint64_t set_id = requireInt(params, "set_id");
        const uint64_t proxy_id = requireInt(params, "proxy_id");
        out_result = enqueue([set_id, proxy_id](UIContext&) {
            return resultToJson(
                rtapi::removeKinematicProxy(set_id, proxy_id));
        });
        return true;
    }
    if (method == "physics.collider.proxy_set.sample") {
        const uint64_t set_id = requireInt(params, "set_id");
        const float dt = optionalFloat(params, "dt", 1.0f / 24.0f);
        out_result = enqueue([set_id, dt](UIContext&) {
            std::vector<RayTrophiSim::KinematicProxySample> samples;
            const rtapi::Result result =
                rtapi::sampleKinematicProxySet(set_id, dt, samples);
            if (!result.ok) {
                return json{{"__error", result.error}};
            }
            json out = json::array();
            for (const auto& sample : samples) {
                out.push_back(sampleToJson(sample));
            }
            return out;
        });
        return true;
    }
    if (method == "physics.collider.proxy_set.solver_stamps") {
        out_result = enqueue([](UIContext&) {
            std::vector<rtapi::KinematicSolverStamps> systems;
            const rtapi::Result result =
                rtapi::getKinematicSolverStamps(systems);
            if (!result.ok) {
                return json{{"__error", result.error}};
            }
            json out = json::array();
            for (const auto& system : systems) {
                json stamps = json::array();
                for (const auto& stamp : system.stamps) {
                    stamps.push_back(json{
                        {"domain", stamp.domain},
                        {"consumer_mask", stamp.consumer_mask},
                        {"frame", stamp.frame},
                        {"set_id", stamp.set_id},
                        {"proxy_id", stamp.proxy_id},
                        {"proxy_name", stamp.proxy_name},
                        {"stamped_cells", stamp.stamped_cells},
                        {"linear_velocity", vec3ToJson(stamp.linear_velocity)},
                        {"angular_velocity", vec3ToJson(stamp.angular_velocity)},
                        {"velocity_valid", stamp.velocity_valid}});
                }
                out.push_back(json{
                    {"system_id", system.system_id},
                    {"system_name", system.system_name},
                    {"steps", system.steps},
                    {"stamps", std::move(stamps)}});
            }
            return out;
        });
        return true;
    }
    return false;
}
