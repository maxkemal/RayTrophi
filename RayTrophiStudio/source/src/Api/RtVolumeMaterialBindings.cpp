#include "RtVolumeMaterialBindings.h"
#include "RtApiInternal.h"
#include "RtIpcMethodRegistry.h"
#include "VolumeMaterialAuthoring.h"
#include "VolumeMaterialService.h"
#include "VDBVolume.h"
#include "Fluid/MatterFogStreams.h"
#include "ProjectManager.h"
#include "Renderer.h"
#include "Backend/IBackend.h"

#include <pybind11/stl.h>
#include <functional>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
using json = nlohmann::json;

namespace {

struct Target {
    std::string id;
    std::string name;
    std::shared_ptr<VolumeShader> shader;
    std::function<void(std::shared_ptr<VolumeShader>)> assign;
    bool* visible = nullptr;
    std::function<void(bool)> set_visible;
};

std::vector<Target> targets(UIContext& ctx) {
    std::vector<Target> result;
    for (const auto& volume : ctx.scene.vdb_volumes) {
        if (!volume || volume->name == "__RayTrophi_Internal_SkyCloudVolume") {
            continue;
        }
        result.push_back({"vdb:" + std::to_string(volume->getVDBVolumeID()), volume->name,
            volume->getShader(), [volume](auto shader) { volume->setShader(shader); }});
    }
    auto& domains = ctx.scene.ensureParticleSimulationSystem().gridDomains();
    for (auto& domain : domains) {
        if (!RayTrophiSim::simulationDomainHasGas(domain.type)) {
            continue;
        }
        if (domain.type == RayTrophiSim::SimulationDomainType::Matter) {
            for (auto& [key, appearance] : domain.matter_fog_streams) {
                auto* slot = &appearance;
                const std::string name = domain.name + " / " +
                    RayTrophiSim::Fluid::fogFieldName(key);
                result.push_back({"matter:" + domain.name + ":" +
                    RayTrophiSim::Fluid::fogFieldId(key), name, slot->shader,
                    [slot](auto shader) { slot->shader = shader; }, &slot->visible,
                    [&ctx, slot](bool visible) {
                        slot->visible = visible;
                        ctx.scene.requestSimulationTimelineRenderResync();
                        g_geometry_dirty = true;
                        g_vulkan_rebuild_pending = true;
                        g_optix_rebuild_pending = true;
                        g_bvh_rebuild_pending = true;
                    }});
            }
            continue;
        }
        auto* slot = domain.type == RayTrophiSim::SimulationDomainType::Matter
            ? &domain.fluid_fog_shader : &domain.shader;
        result.push_back({"gas:" + domain.name, domain.name, *slot,
            [slot](auto shader) { *slot = shader; }});
    }
    return result;
}

json executeMaterial(UIContext& ctx, const std::string& operation, const std::string& id,
             const json& patch = json::object()) {
    if (operation == "list") {
        json rows = json::array();
        for (const auto& target : targets(ctx)) {
            rows.push_back({{"target", target.id}, {"name", target.name},
                            {"has_material", bool(target.shader)}});
        }
        return {{"targets", rows}, {"presets",
            {"physical_cloud", "blackbody_fire", "artistic_fire"}}};
    }
    std::vector<Target> matches;
    for (auto& target : targets(ctx)) {
        if (target.id == id || (id.rfind("gas:", 0) == 0 &&
                target.id == "matter:" + id.substr(4) + ":gas")) {
            matches.push_back(std::move(target));
        }
    }
    if (matches.size() != 1) {
        return {{"__error", "volume material target missing or ambiguous: " + id}};
    }
    auto& target = matches.front();
    if (operation == "get") {
        if (!target.shader) {
            return {{"__error", "target material has not been initialized"}};
        }
        auto result = volume_material::inspect(*target.shader);
        if (target.visible) {
            result["visible"] = *target.visible;
        }
        return result;
    }
    if (rtapi::renderJobActive()) {
        return {{"__error", "scene is locked by the final render job"}};
    }
    {
        VolumeShader next = target.shader ? *target.shader : VolumeShader{};
        std::string error;
        auto material_patch = patch;
        bool visibility = target.visible ? *target.visible : true;
        if (material_patch.contains("visible") && target.visible) {
            if (!material_patch["visible"].is_boolean()) {
                return {{"__error", "visible must be bool"}};
            }
            visibility = material_patch["visible"].get<bool>();
            material_patch.erase("visible");
        }
        if (!volume_material::configure(next, material_patch, error)) {
            return {{"__error", error}, {"code", "invalid_parameter"}};
        }
        if (target.shader) {
            *target.shader = std::move(next);
        } else {
            target.shader = std::make_shared<VolumeShader>(std::move(next));
        }
        target.assign(target.shader);
        if (target.set_visible) {
            target.set_visible(visibility);
        }
    }
    SceneUI::syncVDBVolumesToGPU(ctx);
    ctx.renderer.resetCPUAccumulation();
    if (ctx.backend_ptr) {
        ctx.backend_ptr->resetAccumulation();
    }
    ctx.start_render = true;
    ProjectManager::getInstance().markModified();
    return executeMaterial(ctx, "get", id);
}

py::object toPython(const json& result) {
    if (result.contains("__error")) {
        throw std::runtime_error(result["__error"].get<std::string>());
    }
    return py::module_::import("json").attr("loads")(result.dump());
}

const MethodParam params[] = {
    {"target", "string", true, "Exact target from volume.list_materials", nullptr, nullptr},
    {"settings", "object", true, "Atomic material overlay",
        nullptr, nullptr}
};
const MethodDescriptor listDescriptor = {
    "volume.list_materials", "volume", "List gas and VDB volume material targets", nullptr,
    "read", "SceneRead", false, "object", "volume|gas|vdb|material", "volume.get_material",
    nullptr, nullptr, nullptr, nullptr, nullptr, 0, true
};
const MethodDescriptor getDescriptor = {
    "volume.get_material", "volume", "Read canonical gas/VDB material settings", nullptr,
    "read", "SceneRead", false, "object", "volume|gas|vdb|material", "volume.set_material",
    "volume.list_materials", nullptr, nullptr, nullptr, params, 1, true
};
const MethodDescriptor setDescriptor = {
    "volume.set_material", "volume", "Configure gas/VDB physical optics and emission response",
    "Stable matter:<domain>:<field> targets persist independent Fog appearances; visible is "
    "an optional boolean on these targets. Atomic overlay. Unknown keys and invalid values "
    "are rejected. Presets preserve density "
    "scale and channel names. Vulkan RT random walk; RayFusion uses bounded approximation.",
    "write", "SceneWrite", false, "object", "volume|gas|vdb|material|scattering|emission",
    "volume.get_material", "volume.list_materials", nullptr, "volume.get_material",
    "render_accumulation", params, 2, true
};
const MethodRegistration listRegistration(listDescriptor);
const MethodRegistration getRegistration(getDescriptor);
const MethodRegistration setRegistration(setDescriptor);

} // namespace

nlohmann::json volume_material::execute(UIContext& ctx, const std::string& operation,
    const std::string& target, const nlohmann::json& patch) {
    return executeMaterial(ctx, operation, target, patch);
}

bool dispatchVolumeMaterialIpc(const std::string& method, const json& params,
    const RtIpcTemplateEnqueue& enqueue, json& out) {
    if (method != "volume.list_materials" && method != "volume.get_material" &&
        method != "volume.set_material") {
        return false;
    }
    const bool list = method == "volume.list_materials";
    const bool write = method == "volume.set_material";
    if (!params.is_object()) {
        throw std::invalid_argument("parameters must be an object");
    }
    for (auto it = params.begin(); it != params.end(); ++it) {
        if (list || (it.key() != "target" && (!write || it.key() != "settings"))) {
            throw std::invalid_argument("unknown volume parameter: " + it.key());
        }
    }
    const std::string id = list ? "" : params.at("target").get<std::string>();
    const json patch = write ? params.at("settings") : json::object();
    out = enqueue([list, write, id, patch](UIContext& ctx) {
        return volume_material::execute(ctx, list ? "list" : (write ? "set" : "get"), id, patch);
    });
    return true;
}

void registerVolumeMaterialPython(py::module_& module) {
    auto volume = module.def_submodule("volume", "Shared gas/VDB material authoring");
    volume.def("list_materials", [] {
        if (!rtapi::g_ctx) {
            throw std::runtime_error("rtapi is not bound");
        }
        return toPython(volume_material::execute(*rtapi::g_ctx, "list", ""));
    });
    volume.def("get_material", [](const std::string& target) {
        if (!rtapi::g_ctx) {
            throw std::runtime_error("rtapi is not bound");
        }
        return toPython(volume_material::execute(*rtapi::g_ctx, "get", target));
    }, py::arg("target"));
    volume.def("set_material", [](const std::string& target, const py::kwargs& kwargs) {
        if (!rtapi::g_ctx) {
            throw std::runtime_error("rtapi is not bound");
        }
        const auto text = py::module_::import("json").attr("dumps")(
            kwargs, py::arg("allow_nan") = false).cast<std::string>();
        return toPython(volume_material::execute(*rtapi::g_ctx, "set", target, json::parse(text)));
    }, py::arg("target"));
}
