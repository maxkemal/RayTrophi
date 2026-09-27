#include "KinematicColliderSerialization.h"

#include <exception>
#include <utility>
#include <vector>

namespace RayTrophiSim {
namespace {

nlohmann::json vec3ToJson(const Vec3& value) {
    return nlohmann::json::array({value.x, value.y, value.z});
}

bool vec3FromJson(const nlohmann::json& value,
                  Vec3& result,
                  std::string& error) {
    if (!value.is_array() || value.size() != 3 ||
        !value[0].is_number() || !value[1].is_number() ||
        !value[2].is_number()) {
        error = "invalid_proxy_vec3";
        return false;
    }
    result = Vec3(
        value[0].get<float>(),
        value[1].get<float>(),
        value[2].get<float>());
    return true;
}

nlohmann::json serializeProxy(const KinematicProxyDesc& proxy) {
    return {
        {"id", proxy.id},
        {"name", proxy.name},
        {"bone", proxy.bone},
        {"shape", kinematicProxyShapeName(proxy.shape)},
        {"enabled", proxy.enabled},
        {"local_position", vec3ToJson(proxy.local_position)},
        {"local_rotation", vec3ToJson(proxy.local_rotation_degrees)},
        {"local_axis", vec3ToJson(proxy.local_axis)},
        {"radius", proxy.radius},
        {"half_length", proxy.half_length},
        {"half_extents", vec3ToJson(proxy.half_extents)}};
}

bool deserializeProxy(const nlohmann::json& data,
                      KinematicProxyDesc& proxy,
                      std::string& error) {
    if (!data.is_object()) {
        error = "invalid_proxy_record";
        return false;
    }
    proxy.id = data.at("id").get<uint64_t>();
    proxy.name = data.value("name", std::string());
    proxy.bone = data.at("bone").get<std::string>();
    proxy.enabled = data.value("enabled", true);
    const std::string shape = data.value("shape", std::string("capsule"));
    if (!parseKinematicProxyShape(shape, proxy.shape)) {
        error = "invalid_proxy_shape";
        return false;
    }
    if (data.contains("local_position") &&
        !vec3FromJson(data["local_position"], proxy.local_position, error)) {
        return false;
    }
    if (data.contains("local_rotation") &&
        !vec3FromJson(
            data["local_rotation"], proxy.local_rotation_degrees, error)) {
        return false;
    }
    if (data.contains("local_axis") &&
        !vec3FromJson(data["local_axis"], proxy.local_axis, error)) {
        return false;
    }
    if (data.contains("half_extents") &&
        !vec3FromJson(data["half_extents"], proxy.half_extents, error)) {
        return false;
    }
    proxy.radius = data.value("radius", proxy.radius);
    proxy.half_length = data.value("half_length", proxy.half_length);
    return true;
}

} // namespace

nlohmann::json serializeKinematicColliders(
    const KinematicColliderRegistry& registry) {
    nlohmann::json sets = nlohmann::json::array();
    for (const KinematicProxySet& set : registry.sets()) {
        nlohmann::json proxies = nlohmann::json::array();
        for (const KinematicProxyDesc& proxy : set.proxies) {
            proxies.push_back(serializeProxy(proxy));
        }
        sets.push_back({
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
            {"proxies", std::move(proxies)}});
    }
    return {{"version", 1}, {"sets", std::move(sets)}};
}

bool deserializeKinematicColliders(
    const nlohmann::json& data,
    KinematicColliderRegistry& registry,
    std::string& error) {
    try {
        if (!data.is_object() || data.value("version", 0) != 1 ||
            !data.contains("sets") || !data["sets"].is_array()) {
            error = "invalid_kinematic_collider_schema";
            return false;
        }
        if (data["sets"].size() > kMaxKinematicProxySets) {
            error = "proxy_set_limit_reached";
            return false;
        }

        std::vector<KinematicProxySet> sets;
        sets.reserve(data["sets"].size());
        for (const nlohmann::json& item : data["sets"]) {
            if (!item.is_object() || !item.contains("proxies") ||
                !item["proxies"].is_array()) {
                error = "invalid_proxy_set_record";
                return false;
            }
            if (item["proxies"].size() > kMaxKinematicProxiesPerSet) {
                error = "proxy_limit_reached";
                return false;
            }
            KinematicProxySet set;
            set.id = item.at("id").get<uint64_t>();
            set.name = item.at("name").get<std::string>();
            set.target_character =
                item.at("target_character").get<std::string>();
            set.target_node_id =
                item.value("target_node_id", std::string());
            set.enabled = item.value("enabled", true);
            set.viewport_visible = item.value("viewport_visible", true);
            set.consumer_mask = item.value(
                "consumer_mask", static_cast<uint32_t>(KinematicConsumerAll));
            set.friction = item.value("friction", set.friction);
            set.restitution = item.value("restitution", set.restitution);
            set.thickness = item.value("thickness", set.thickness);
            set.revision = item.value("revision", static_cast<uint64_t>(1));
            set.proxies.reserve(item["proxies"].size());
            for (const nlohmann::json& proxy_data : item["proxies"]) {
                KinematicProxyDesc proxy;
                if (!deserializeProxy(proxy_data, proxy, error)) {
                    return false;
                }
                set.proxies.push_back(std::move(proxy));
            }
            sets.push_back(std::move(set));
        }
        return registry.restoreSets(sets, error);
    } catch (const std::exception&) {
        error = "invalid_kinematic_collider_value";
        return false;
    }
}

} // namespace RayTrophiSim
