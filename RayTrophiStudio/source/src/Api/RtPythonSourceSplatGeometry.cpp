#include "Fluid/SourceSplatGeometryJson.h"
#include "RtPythonSourceSplatGeometry.h"
#include "Api/SourceSplatGeometryApi.h"
#include "Api/MatterPhaseOutputApi.h"
#include "Fluid/MatterPhaseOutputJson.h"
#include <pybind11/stl.h>

namespace py = pybind11;

void writeSourceSplatGeometry(py::dict& out, const rtapi::SimulationFlowSourceInfo& source) {
    const auto& value = source.splat_geometry;
    py::dict geometry;
    geometry["mode"] = value.mode;
    geometry["objects"] = value.objects;
    geometry["source_materials"] = value.source_materials;
    geometry["near_distance"] = value.near_distance;
    geometry["mesh_budget"] = value.mesh_budget;
    out["splat_geometry"] = geometry;
}

void applySourceSplatGeometry(rtapi::SimulationFlowSourceInfo& source, const py::kwargs& kwargs) {
    if (!kwargs.contains("splat_geometry")) {
        return;
    }
    const py::dict fields = py::cast<py::dict>(kwargs["splat_geometry"]);
    nlohmann::json patch = nlohmann::json::object();
    for (const auto& item : fields) {
        const std::string key = py::cast<std::string>(item.first);
        if (key == "mode") {
            patch[key] = py::cast<std::string>(item.second);
        } else if (key == "objects") {
            patch[key] = py::cast<std::vector<std::string>>(item.second);
        } else if (key == "source_materials") {
            if (!py::isinstance<py::bool_>(item.second)) {
                throw py::value_error("source_materials must be bool");
            }
            patch[key] = py::cast<bool>(item.second);
        } else if (key == "near_distance") {
            if (py::isinstance<py::bool_>(item.second)) {
                throw py::value_error("near_distance must be numeric, not bool");
            }
            patch[key] = py::cast<double>(item.second);
        } else if (key == "mesh_budget") {
            if (!py::isinstance<py::int_>(item.second) || py::isinstance<py::bool_>(item.second)) {
                throw py::value_error("mesh_budget must be an integer");
            }
            patch[key] = py::cast<int64_t>(item.second);
        } else {
            throw py::value_error("unknown splat_geometry field: " + key);
        }
    }
    RayTrophiSim::Fluid::from_json(patch, source.splat_geometry);
}

void registerSourceSplatGeometry(py::module_& flow_source) {
    flow_source.def("set_splat_geometry", [](const std::string& name, const py::kwargs& kwargs) {
        rtapi::SimulationFlowSourceInfo source;
        auto status = rtapi::getSimulationFlowSource(name, source);
        if (!status.ok) {
            throw py::value_error(status.error);
        }
        py::kwargs wrapped;
        wrapped["splat_geometry"] = kwargs;
        applySourceSplatGeometry(source, wrapped);
        status = rtapi::setSourceSplatGeometry(name, source.splat_geometry);
        if (!status.ok) {
            throw py::value_error(status.error);
        }
    }, py::arg("name"), "Set this emitter's splat geometry without resetting simulation. "
       "Fields: mode, objects, source_materials, near_distance, mesh_budget. "
       "Identical validation to IPC flow_source.set_splat_geometry.");
}

void registerSubstanceSplatGeometry(py::module_& fluid) {
    fluid.def("inspect_fog_fields", [](const std::string& domain) {
        nlohmann::json result;
        const auto status = rtapi::inspectMatterFogFields(domain, result);
        if (!status.ok) {
            throw py::value_error(status.error);
        }
        return py::module_::import("json").attr("loads")(result.dump());
    }, py::arg("domain"), "Inspect separate Matter Fog fields without advancing physics.");
    fluid.def("get_phase_output", [](const std::string& domain) {
        RayTrophiSim::Fluid::MatterPhaseOutput value;
        const auto status = rtapi::getPhaseOutput(domain, value);
        if (!status.ok) {
            throw py::value_error(status.error);
        }
        return py::module_::import("json").attr("loads")(nlohmann::json(value).dump());
    }, py::arg("domain"));
    fluid.def("set_phase_output", [](const std::string& domain, const py::kwargs& kwargs) {
        RayTrophiSim::Fluid::MatterPhaseOutput value;
        auto status = rtapi::getPhaseOutput(domain, value);
        if (!status.ok) {
            throw py::value_error(status.error);
        }
        nlohmann::json patch = nlohmann::json::object();
        for (const auto& item : kwargs) {
            if (!py::isinstance<py::str>(item.second)) {
                throw py::value_error("phase output routes must be strings");
            }
            patch[py::cast<std::string>(item.first)] = py::cast<std::string>(item.second);
        }
        RayTrophiSim::Fluid::from_json(patch, value);
        status = rtapi::setPhaseOutput(domain, value);
        if (!status.ok) {
            throw py::value_error(status.error);
        }
    }, py::arg("domain"));
    fluid.def("get_substance_splat_geometry", [](const std::string& domain,
        const std::string& substance) {
        RayTrophiSim::Fluid::SourceSplatGeometry geometry;
        const auto status = rtapi::getSubstanceSplatGeometry(domain, substance, geometry);
        if (!status.ok) {
            throw py::value_error(status.error);
        }
        return py::module_::import("json").attr("loads")(nlohmann::json(geometry).dump());
    }, py::arg("domain"), py::arg("substance"));
    fluid.def("set_substance_splat_geometry", [](const std::string& domain,
        const std::string& substance, const py::kwargs& kwargs) {
        rtapi::SimulationFlowSourceInfo holder;
        auto status = rtapi::getSubstanceSplatGeometry(domain, substance, holder.splat_geometry);
        if (!status.ok) {
            throw py::value_error(status.error);
        }
        py::kwargs wrapped;
        wrapped["splat_geometry"] = kwargs;
        applySourceSplatGeometry(holder, wrapped);
        status = rtapi::setSubstanceSplatGeometry(domain, substance, holder.splat_geometry);
        if (!status.ok) {
            throw py::value_error(status.error);
        }
    }, py::arg("domain"), py::arg("substance"));
}
