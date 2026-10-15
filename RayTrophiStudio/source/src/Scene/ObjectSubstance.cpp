#include "ObjectSubstance.h"
#include "AcousticMaterial.h"
#include "scene_data.h"
#include "SubstanceLibrary.h"
#include "TriangleMesh.h"
#include <cmath>
#include <cstring>
#include <set>
#include <stdexcept>

namespace matter {
namespace {
using json = nlohmann::json;

void require(bool valid, const char* message) {
    if (!valid) {
        throw std::invalid_argument(message);
    }
}

float number(const json& data, const char* key, double fallback, double maximum) {
    const auto value = data.value(key, json(fallback));
    require(value.is_number() && std::isfinite(value.get<double>()) &&
            value.get<double>() >= 0 && value.get<double>() <= maximum,
            "Object substance override must be a finite number within its bounds");
    return value.get<float>();
}

bool objectCollider(const RayTrophiSim::ParticleColliderDesc& collider) {
    using Mode = RayTrophiSim::ParticleColliderSourceMode;
    return !collider.source_name.empty() &&
        (collider.source_mode == Mode::ObjectAABB || collider.source_mode == Mode::ObjectOBB ||
         collider.source_mode == Mode::ObjectMeshSDF ||
         collider.source_mode == Mode::ObjectConvexDecomp ||
         collider.source_mode == Mode::ObjectMeshBVH || collider.source_mode == Mode::Sphere ||
         collider.source_mode == Mode::Capsule);
}

json colliderBinding(const RayTrophiSim::ParticleColliderDesc& collider) {
    const auto& value = collider.msf_override;
    return {{"substance", collider.msf_substance}, {"overrides", {
        {"override_ignition", value.override_ignition}, {"ignition_kelvin", value.ignition_kelvin},
        {"burn_rate_scale", value.burn_rate_scale},
        {"fuel_capacity_scale", value.fuel_capacity_scale}}}};
}
}

json validateObjectSubstance(const json& binding) {
    require(binding.is_object(), "Object substance binding must be an object");
    for (auto item = binding.begin(); item != binding.end(); ++item) {
        require(item.key() == "substance" || item.key() == "overrides",
                "Unknown object substance binding field");
    }
    require(binding.contains("substance") && binding.at("substance").is_string(),
            "Object substance needs a substance name");
    const auto name = binding.at("substance").get<std::string>();
    require(RayTrophiSim::tryFindSubstance(name) != nullptr, "Unknown object substance");
    const auto values = binding.value("overrides", json::object());
    require(values.is_object(), "Object substance overrides must be an object");
    for (auto item = values.begin(); item != values.end(); ++item) {
        require(item.key() == "override_ignition" || item.key() == "ignition_kelvin" ||
                item.key() == "burn_rate_scale" || item.key() == "fuel_capacity_scale",
                "Unknown object substance override");
    }
    require(!values.contains("override_ignition") || values.at("override_ignition").is_boolean(),
            "override_ignition must be boolean");
    return {{"substance", name}, {"overrides", {
        {"override_ignition", values.value("override_ignition", false)},
        {"ignition_kelvin", number(values, "ignition_kelvin", 573, 100000)},
        {"burn_rate_scale", number(values, "burn_rate_scale", 1, 1000000)},
        {"fuel_capacity_scale", number(values, "fuel_capacity_scale", 1, 1000000)}}}};
}

json objectSubstanceState(const SceneData& scene, const std::string& object, bool knownFlatNode) {
    bool exists = knownFlatNode;
    if (!knownFlatNode) {
        for (const auto& entry : scene.world.objects) {
            const auto mesh = std::dynamic_pointer_cast<TriangleMesh>(entry);
            exists = exists || (mesh && !mesh->transient && mesh->nodeName == object);
        }
    }
    exists = exists && !scene.isEditorPendingDeleteObjectName(object);
    json result = {{"object", object}, {"exists", exists}, {"source", "unassigned"},
        {"binding", nullptr}, {"colliders", json::array()}, {"conflict", false}};
    json legacy;
    for (const auto& system : scene.particle_systems) {
        if (!system.runtime) {
            continue;
        }
        for (const auto& collider : system.runtime->colliders()) {
            if (!objectCollider(collider) || collider.source_name != object) {
                continue;
            }
            const auto binding = colliderBinding(collider);
            result["colliders"].push_back({{"system_id", system.id}, {"name", collider.name},
                {"binding", binding}});
            if (legacy.is_null()) {
                legacy = binding;
            } else if (legacy != binding) {
                result["conflict"] = true;
            }
        }
    }
    if (scene.object_substances.contains(object)) {
        result["source"] = "object";
        result["binding"] = scene.object_substances.at(object);
        result["conflict"] = false;
    } else if (!legacy.is_null()) {
        result["source"] = result.at("conflict").get<bool>() ? "conflict" : "legacy_collider";
        if (!result.at("conflict").get<bool>()) {
            result["binding"] = legacy;
        }
    }
    if (!result.at("binding").is_null()) {
        const auto name = result.at("binding").at("substance").get<std::string>();
        const auto* profile = RayTrophiSim::tryFindSubstance(name);
        result["resolved"] = profile != nullptr;
        if (profile) {
            result["profile_fields"] = RayTrophiSim::substanceFieldsToJson(*profile);
            result["acoustics"] = audio::acousticMaterialResponse(*profile);
        }
    } else {
        result["resolved"] = false;
    }
    return result;
}

void applyObjectSubstance(const SceneData& scene, RayTrophiSim::ParticleColliderDesc& collider) {
    if (!objectCollider(collider) || !scene.object_substances.contains(collider.source_name)) {
        return;
    }
    const auto& binding = scene.object_substances.at(collider.source_name);
    collider.msf_substance = binding.at("substance").get<std::string>();
    const auto& values = binding.at("overrides");
    collider.msf_override.override_ignition = values.at("override_ignition");
    collider.msf_override.ignition_kelvin = values.at("ignition_kelvin");
    collider.msf_override.burn_rate_scale = values.at("burn_rate_scale");
    collider.msf_override.fuel_capacity_scale = values.at("fuel_capacity_scale");
}

bool colliderMatchesObjectSubstance(const SceneData& scene,
                                   const RayTrophiSim::ParticleColliderDesc& collider) {
    if (!objectCollider(collider) || !scene.object_substances.contains(collider.source_name)) {
        return true;
    }
    const auto& binding = scene.object_substances.at(collider.source_name);
    const auto& values = binding.at("overrides");
    return collider.msf_substance == binding.at("substance").get<std::string>() &&
        collider.msf_override.override_ignition == values.at("override_ignition").get<bool>() &&
        collider.msf_override.ignition_kelvin == values.at("ignition_kelvin").get<float>() &&
        collider.msf_override.burn_rate_scale == values.at("burn_rate_scale").get<float>() &&
        collider.msf_override.fuel_capacity_scale == values.at("fuel_capacity_scale").get<float>();
}

void syncObjectSubstances(SceneData& scene) {
    if (scene.object_substances.empty()) {
        return;
    }
    for (auto& system : scene.particle_systems) {
        if (!system.runtime) {
            continue;
        }
        for (auto& collider : system.runtime->colliders()) {
            if (!colliderMatchesObjectSubstance(scene, collider)) {
                applyObjectSubstance(scene, collider);
                system.runtime->clearMaterialStateField(collider.source_name);
                scene.clearMaterialDamageHistory(collider.source_name);
            }
        }
    }
}

json saveObjectSubstances(const SceneData& scene) {
    auto result = json::object();
    for (auto item = scene.object_substances.begin(); item != scene.object_substances.end(); ++item) {
        if (!scene.isEditorPendingDeleteObjectName(item.key())) {
            result[item.key()] = item.value();
        }
    }
    return result;
}

void restoreObjectSubstances(SceneData& scene, const json& data) {
    require(data.is_object() && data.size() <= 65536,
            "object_substances must be an object with at most 65536 bindings");
    auto candidate = json::object();
    for (auto item = data.begin(); item != data.end(); ++item) {
        require(!item.key().empty() && item.key().size() <= 1024,
                "Object substance key must contain 1..1024 characters");
        candidate[item.key()] = validateObjectSubstance(item.value());
    }
    if (scene.object_substances != candidate) {
        ++scene.object_substance_revision;
    }
    scene.object_substances = std::move(candidate);
    syncObjectSubstances(scene);
}

json takeObjectSubstance(SceneData& scene, const std::string& object) {
    const auto saved = scene.object_substances.value(object, json());
    if (!saved.is_null()) {
        ++scene.object_substance_revision;
    }
    scene.object_substances.erase(object);
    return saved;
}

void putObjectSubstance(SceneData& scene, const std::string& object, const json& binding) {
    const auto before = scene.object_substances.value(object, json());
    if (binding.is_null()) {
        scene.object_substances.erase(object);
    } else {
        scene.object_substances[object] = validateObjectSubstance(binding);
    }
    if (before != scene.object_substances.value(object, json())) {
        ++scene.object_substance_revision;
    }
    syncObjectSubstances(scene);
}

void copyObjectSubstance(SceneData& scene, const std::string& source, const std::string& target) {
    // Legacy collider metadata is not implicitly promoted by a geometry duplicate.
    if (scene.object_substances.contains(source)) {
        putObjectSubstance(scene, target, scene.object_substances.at(source));
    }
}

std::uint64_t objectSubstanceFingerprint(const SceneData& scene) {
    std::uint64_t result = 14695981039346656037ULL;
    const auto word = [&](std::uint64_t value) {
        for (unsigned byte = 0; byte < 8; ++byte) {
            result = (result ^ ((value >> (byte * 8)) & 255)) * 1099511628211ULL;
        }
    };
    const auto text = [&](const std::string& value) {
        word(value.size());
        for (const unsigned char character : value) {
            result = (result ^ character) * 1099511628211ULL;
        }
    };
    const auto scalar = [&](float value) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        word(bits);
    };
    word(scene.object_substance_revision);
    word(RayTrophiSim::substanceLibraryRevision());
    word(scene.editor_pending_delete_object_names.size());
    for (const auto& object : scene.editor_pending_delete_object_names) {
        text(object);
    }
    for (const auto& system : scene.particle_systems) {
        if (!system.runtime) {
            continue;
        }
        word(system.id);
        for (const auto& collider : system.runtime->colliders()) {
            if (!objectCollider(collider)) {
                continue;
            }
            text(collider.name);
            text(collider.source_name);
            text(collider.msf_substance);
            word(collider.msf_override.override_ignition);
            scalar(collider.msf_override.ignition_kelvin);
            scalar(collider.msf_override.burn_rate_scale);
            scalar(collider.msf_override.fuel_capacity_scale);
        }
    }
    return result;
}
} // namespace matter
