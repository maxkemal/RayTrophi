#include "RtPythonKinematicCollider.h"

#include "Api/RtApi.h"
#include "RtPyCommon.h"

#include <pybind11/stl.h>

#include <algorithm>

namespace py = pybind11;

namespace {

py::dict proxyToDict(const RayTrophiSim::KinematicProxyDesc& proxy) {
    py::dict out;
    out["id"] = proxy.id;
    out["name"] = proxy.name;
    out["bone"] = proxy.bone;
    out["shape"] = RayTrophiSim::kinematicProxyShapeName(proxy.shape);
    out["enabled"] = proxy.enabled;
    out["local_position"] = vec3ToPython(proxy.local_position);
    out["local_rotation"] = vec3ToPython(proxy.local_rotation_degrees);
    out["local_axis"] = vec3ToPython(proxy.local_axis);
    out["radius"] = proxy.radius;
    out["half_length"] = proxy.half_length;
    out["half_extents"] = vec3ToPython(proxy.half_extents);
    return out;
}

py::dict setToDict(const RayTrophiSim::KinematicProxySet& set) {
    py::dict out;
    out["id"] = set.id;
    out["name"] = set.name;
    out["target_character"] = set.target_character;
    out["target_node_id"] = set.target_node_id;
    out["enabled"] = set.enabled;
    out["consumer_mask"] = set.consumer_mask;
    out["friction"] = set.friction;
    out["restitution"] = set.restitution;
    out["thickness"] = set.thickness;
    out["revision"] = set.revision;
    py::list proxies;
    for (const RayTrophiSim::KinematicProxyDesc& proxy : set.proxies) {
        proxies.append(proxyToDict(proxy));
    }
    out["proxies"] = proxies;
    return out;
}

py::dict sampleToDict(const RayTrophiSim::KinematicProxySample& sample) {
    py::dict out;
    out["set_id"] = sample.set_id;
    out["proxy_id"] = sample.proxy_id;
    out["set_name"] = sample.set_name;
    out["proxy_name"] = sample.proxy_name;
    out["target_character"] = sample.target_character;
    out["bone"] = sample.bone;
    out["shape"] = RayTrophiSim::kinematicProxyShapeName(sample.shape);
    out["resolved"] = sample.resolved;
    out["unresolved_reason"] = sample.unresolved_reason;
    out["center"] = vec3ToPython(sample.center);
    out["capsule_start"] = vec3ToPython(sample.capsule_start);
    out["capsule_end"] = vec3ToPython(sample.capsule_end);
    out["radius"] = sample.radius;
    out["half_extents"] = vec3ToPython(sample.half_extents);
    out["linear_velocity"] = vec3ToPython(sample.linear_velocity);
    out["angular_velocity"] = vec3ToPython(sample.angular_velocity);
    out["velocity_valid"] = sample.velocity_valid;
    py::list matrix;
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            matrix.append(sample.world_transform.m[row][column]);
        }
    }
    out["world_transform"] = matrix;
    return out;
}

void patchSet(const py::kwargs& kwargs,
              RayTrophiSim::KinematicProxySet& set) {
    if (kwargs.contains("name")) {
        set.name = py::cast<std::string>(kwargs["name"]);
    }
    if (kwargs.contains("target_character")) {
        set.target_character =
            py::cast<std::string>(kwargs["target_character"]);
    }
    if (kwargs.contains("target_node_id")) {
        set.target_node_id =
            py::cast<std::string>(kwargs["target_node_id"]);
    }
    if (kwargs.contains("enabled")) {
        set.enabled = py::cast<bool>(kwargs["enabled"]);
    }
    if (kwargs.contains("consumer_mask")) {
        set.consumer_mask = py::cast<uint32_t>(kwargs["consumer_mask"]);
    }
    if (kwargs.contains("friction")) {
        set.friction = py::cast<float>(kwargs["friction"]);
    }
    if (kwargs.contains("restitution")) {
        set.restitution = py::cast<float>(kwargs["restitution"]);
    }
    if (kwargs.contains("thickness")) {
        set.thickness = py::cast<float>(kwargs["thickness"]);
    }
}

void patchProxy(const py::kwargs& kwargs,
                RayTrophiSim::KinematicProxyDesc& proxy) {
    if (kwargs.contains("proxy_id")) {
        proxy.id = py::cast<uint64_t>(kwargs["proxy_id"]);
    }
    if (kwargs.contains("name")) {
        proxy.name = py::cast<std::string>(kwargs["name"]);
    }
    if (kwargs.contains("bone")) {
        proxy.bone = py::cast<std::string>(kwargs["bone"]);
    }
    if (kwargs.contains("shape")) {
        const std::string name = py::cast<std::string>(kwargs["shape"]);
        if (!RayTrophiSim::parseKinematicProxyShape(name, proxy.shape)) {
            throw py::value_error("shape must be sphere, capsule, or box");
        }
    }
    if (kwargs.contains("enabled")) {
        proxy.enabled = py::cast<bool>(kwargs["enabled"]);
    }
    if (kwargs.contains("local_position")) {
        proxy.local_position = vec3FromPython(kwargs["local_position"]);
    }
    if (kwargs.contains("local_rotation")) {
        proxy.local_rotation_degrees =
            vec3FromPython(kwargs["local_rotation"]);
    }
    if (kwargs.contains("local_axis")) {
        proxy.local_axis = vec3FromPython(kwargs["local_axis"]);
    }
    if (kwargs.contains("radius")) {
        proxy.radius = py::cast<float>(kwargs["radius"]);
    }
    if (kwargs.contains("half_length")) {
        proxy.half_length = py::cast<float>(kwargs["half_length"]);
    }
    if (kwargs.contains("half_extents")) {
        proxy.half_extents = vec3FromPython(kwargs["half_extents"]);
    }
}

} // namespace

namespace rtpy {

void registerKinematicColliderBindings(py::module_& collider_module) {
    py::module_ proxy_set = collider_module.def_submodule(
        "proxy_set",
        "Bone-attached solver-neutral kinematic collider proxy sets");

    proxy_set.def("list", [] {
        std::vector<RayTrophiSim::KinematicProxySet> sets;
        requireResult(rtapi::listKinematicProxySets(sets));
        py::list out;
        for (const auto& set : sets) {
            out.append(setToDict(set));
        }
        return out;
    });
    proxy_set.def("get", [](uint64_t set_id) {
        RayTrophiSim::KinematicProxySet set;
        requireResult(rtapi::getKinematicProxySet(set_id, set));
        return setToDict(set);
    }, py::arg("set_id"));
    proxy_set.def("create", [](const std::string& name,
                                const std::string& target_character,
                                const py::kwargs& kwargs) {
        RayTrophiSim::KinematicProxySet set;
        set.name = name;
        set.target_character = target_character;
        patchSet(kwargs, set);
        RayTrophiSim::KinematicProxySet created;
        requireResult(rtapi::createKinematicProxySet(set, created));
        return setToDict(created);
    }, py::arg("name"), py::arg("target_character"));
    proxy_set.def("set", [](uint64_t set_id, const py::kwargs& kwargs) {
        RayTrophiSim::KinematicProxySet set;
        requireResult(rtapi::getKinematicProxySet(set_id, set));
        patchSet(kwargs, set);
        requireResult(rtapi::updateKinematicProxySet(set_id, set));
    }, py::arg("set_id"));
    proxy_set.def("delete", [](uint64_t set_id) {
        requireResult(rtapi::removeKinematicProxySet(set_id));
    }, py::arg("set_id"));
    proxy_set.def("auto_fit", [](uint64_t set_id,
                                  const py::kwargs& kwargs) {
        RayTrophiSim::KinematicAutoFitOptions options;
        if (kwargs.contains("replace_existing")) {
            options.replace_existing =
                py::cast<bool>(kwargs["replace_existing"]);
        }
        if (kwargs.contains("weighted_bones_only")) {
            options.weighted_bones_only =
                py::cast<bool>(kwargs["weighted_bones_only"]);
        }
        if (kwargs.contains("radius_fraction")) {
            options.radius_fraction =
                py::cast<float>(kwargs["radius_fraction"]);
        }
        if (kwargs.contains("minimum_radius")) {
            options.minimum_radius =
                py::cast<float>(kwargs["minimum_radius"]);
        }
        if (kwargs.contains("maximum_radius")) {
            options.maximum_radius =
                py::cast<float>(kwargs["maximum_radius"]);
        }
        if (kwargs.contains("minimum_bone_length")) {
            options.minimum_bone_length =
                py::cast<float>(kwargs["minimum_bone_length"]);
        }
        if (kwargs.contains("maximum_proxies")) {
            options.maximum_proxies =
                py::cast<uint32_t>(kwargs["maximum_proxies"]);
        }
        uint32_t count = 0;
        requireResult(rtapi::autoFitKinematicProxySet(
            set_id, options, count));
        return count;
    }, py::arg("set_id"));
    proxy_set.def("set_proxy", [](uint64_t set_id,
                                   const std::string& bone,
                                   const py::kwargs& kwargs) {
        RayTrophiSim::KinematicProxyDesc proxy;
        proxy.bone = bone;
        if (kwargs.contains("proxy_id")) {
            const uint64_t proxy_id = py::cast<uint64_t>(kwargs["proxy_id"]);
            RayTrophiSim::KinematicProxySet set;
            requireResult(rtapi::getKinematicProxySet(set_id, set));
            const auto it = std::find_if(
                set.proxies.begin(), set.proxies.end(),
                [proxy_id](const RayTrophiSim::KinematicProxyDesc& value) {
                    return value.id == proxy_id;
                });
            if (it == set.proxies.end()) {
                throw py::key_error("unknown_proxy");
            }
            proxy = *it;
            proxy.bone = bone;
        }
        patchProxy(kwargs, proxy);
        RayTrophiSim::KinematicProxyDesc stored;
        requireResult(rtapi::setKinematicProxy(set_id, proxy, stored));
        return proxyToDict(stored);
    }, py::arg("set_id"), py::arg("bone"));
    proxy_set.def("remove_proxy", [](uint64_t set_id, uint64_t proxy_id) {
        requireResult(rtapi::removeKinematicProxy(set_id, proxy_id));
    }, py::arg("set_id"), py::arg("proxy_id"));
    proxy_set.def("sample", [](uint64_t set_id, float dt) {
        std::vector<RayTrophiSim::KinematicProxySample> samples;
        requireResult(rtapi::sampleKinematicProxySet(set_id, dt, samples));
        py::list out;
        for (const auto& sample : samples) {
            out.append(sampleToDict(sample));
        }
        return out;
    }, py::arg("set_id"), py::arg("dt") = 1.0f / 24.0f);
}

} // namespace rtpy
