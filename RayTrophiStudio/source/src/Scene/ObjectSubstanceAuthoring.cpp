#include "ObjectSubstance.h"
#include "scene_ui.h"
#include "scene_data.h"
#include "SceneHistory.h"
#include "ProjectManager.h"
#include "SubstanceLibrary.h"
#include "Api/RtApi.h"
#include "globals.h"
#include "Renderer.h"
#include "TriangleMesh.h"
#include "Backend/IBackend.h"
#include <set>
#include <stdexcept>

namespace matter {
namespace {
using json = nlohmann::json;

void invalidate(UIContext& ctx, const std::string& object) {
    for (auto& system : ctx.scene.particle_systems) {
        if (system.runtime) {
            system.runtime->clearMaterialStateField(object);
        }
    }
    ctx.scene.clearMaterialDamageHistory(object);
    ctx.scene.clearSimFrameCache();
    ctx.scene.requestSimulationTimelineRenderResync();
    ctx.renderer.resetCPUAccumulation();
    if (ctx.backend_ptr) {
        ctx.backend_ptr->resetAccumulation();
    }
    ctx.start_render = true;
    ProjectManager::getInstance().markModified();
}

json snapshot(const SceneData& scene, const std::string& object) {
    const auto state = objectSubstanceState(scene, object);
    return {{"authored", scene.object_substances.value(object, json())},
        {"colliders", state.at("colliders")}};
}

void restore(UIContext& ctx, const std::string& object, const json& state) {
    // Refuse an undo that would introduce a dangling library reference.
    const auto& authored = state.at("authored");
    if (!authored.is_null() && !RayTrophiSim::tryFindSubstance(
            authored.at("substance").get<std::string>())) {
        throw std::invalid_argument("Undo substance no longer exists in the library");
    }
    for (const auto& saved : state.at("colliders")) {
        if (!RayTrophiSim::tryFindSubstance(
                saved.at("binding").at("substance").get<std::string>())) {
            throw std::invalid_argument("Undo collider substance no longer exists in the library");
        }
    }
    if (authored.is_null()) {
        ctx.scene.object_substances.erase(object);
    } else {
        ctx.scene.object_substances[object] = authored;
    }
    for (auto& system : ctx.scene.particle_systems) {
        if (!system.runtime) {
            continue;
        }
        for (auto& collider : system.runtime->colliders()) {
            if (collider.source_name != object) {
                continue;
            }
            for (const auto& saved : state.at("colliders")) {
                if (saved.at("system_id") != system.id || saved.at("name") != collider.name) {
                    continue;
                }
                const auto& binding = saved.at("binding");
                collider.msf_substance = binding.at("substance").get<std::string>();
                const auto& values = binding.at("overrides");
                collider.msf_override.override_ignition = values.at("override_ignition");
                collider.msf_override.ignition_kelvin = values.at("ignition_kelvin");
                collider.msf_override.burn_rate_scale = values.at("burn_rate_scale");
                collider.msf_override.fuel_capacity_scale = values.at("fuel_capacity_scale");
            }
        }
    }
    syncObjectSubstances(ctx.scene);
    ++ctx.scene.object_substance_revision;
    invalidate(ctx, object);
}

class ObjectSubstanceCommand final : public SceneCommand {
public:
    ObjectSubstanceCommand(std::string object, json before, json after)
        : object_(std::move(object)), before_(std::move(before)), after_(std::move(after)),
          references_(retainObjectSubstanceHistory(json::array({before_, after_}))) {}

    void execute(UIContext& ctx) override {
        restore(ctx, object_, after_);
    }
    void undo(UIContext& ctx) override {
        restore(ctx, object_, before_);
    }
    Type getType() const override {
        return Type::Generic;
    }
    std::string getDescription() const override {
        return "Object substance: " + object_;
    }
private:
    std::string object_;
    json before_, after_;
    std::shared_ptr<void> references_;
};
}

json invokeObjectSubstance(UIContext& ctx, const std::string& operation, const json& params) {
    if (g_scene_loading_in_progress.load()) {
        throw std::runtime_error("Object substances are unavailable during project loading");
    }
    if (!params.is_object()) {
        throw std::invalid_argument("Object substance parameters must be an object");
    }
    const bool list = operation == "list_objects";
    const bool get = operation == "get_object";
    const bool set = operation == "set_object";
    const bool clear = operation == "clear_object";
    const bool adopt = operation == "adopt_object";
    if (!list && !get && !set && !clear && !adopt) {
        throw std::invalid_argument("Unknown object substance operation");
    }
    for (auto item = params.begin(); item != params.end(); ++item) {
        if ((list || item.key() != "object") && (!set || item.key() != "binding")) {
            throw std::invalid_argument("Unknown object substance parameter: " + item.key());
        }
    }
    if (list) {
        std::set<std::string> names;
        for (const auto& item : ctx.scene.world.objects) {
            const auto mesh = std::dynamic_pointer_cast<TriangleMesh>(item);
            if (mesh && !mesh->transient && !mesh->nodeName.empty() &&
                !ctx.scene.isEditorPendingDeleteObjectName(mesh->nodeName)) {
                names.insert(mesh->nodeName);
            }
        }
        for (auto item = ctx.scene.object_substances.begin();
             item != ctx.scene.object_substances.end(); ++item) {
            names.insert(item.key());
        }
        auto result = json::array();
        for (const auto& name : names) {
            result.push_back(objectSubstanceState(ctx.scene, name));
        }
        return result;
    }
    if (!params.contains("object") || !params.at("object").is_string() ||
        params.at("object").get<std::string>().empty() ||
        params.at("object").get<std::string>().size() > 1024) {
        throw std::invalid_argument("object must be a name with 1..1024 characters");
    }
    const auto object = params.at("object").get<std::string>();
    const auto state = objectSubstanceState(ctx.scene, object);
    if (get) {
        return state;
    }
    if (rtapi::renderOutputPending()) {
        throw std::runtime_error("Object substance authoring is locked by the render job");
    }
    if (!clear && !state.at("exists").get<bool>()) {
        throw std::invalid_argument("Object substance requires an existing flat mesh node");
    }
    json binding;
    if (set) {
        if (!params.contains("binding")) {
            throw std::invalid_argument("set_object requires a binding object");
        }
        binding = validateObjectSubstance(params.at("binding"));
    } else if (adopt) {
        if (state.at("source") != "legacy_collider") {
            throw std::invalid_argument("Adopt requires one unambiguous legacy collider material");
        }
        binding = validateObjectSubstance(state.at("binding"));
    }
    if (!clear && !ctx.scene.object_substances.contains(object) &&
        ctx.scene.object_substances.size() >= 65536) {
        throw std::invalid_argument("Object substance binding limit reached");
    }
    const auto before = snapshot(ctx.scene, object);
    if (clear) {
        ctx.scene.object_substances.erase(object);
    } else {
        ctx.scene.object_substances[object] = binding;
    }
    syncObjectSubstances(ctx.scene);
    const auto after = snapshot(ctx.scene, object);
    if (before != after) {
        ++ctx.scene.object_substance_revision;
        if (ctx.scene_ui_ptr) {
            ctx.scene_ui_ptr->history.record(
                std::make_unique<ObjectSubstanceCommand>(object, before, after));
        }
        invalidate(ctx, object);
    }
    return objectSubstanceState(ctx.scene, object);
}
} // namespace matter
