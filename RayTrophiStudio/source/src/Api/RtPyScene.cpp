/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          Api/RtPyScene.cpp
* Author:        Kemal Demirtas
* Date:          July 2026
* License:       MIT
* =========================================================================
*
* Python bindings for the scene-core surface: rt.select (new), rt.material
* (asset management added, parameter get/set moved here) and rt.lights
* (parameters added, existing functions moved here).
*
* Split out of RtPython.cpp, which was already at its size budget. The `rt`
* module object is passed in from PYBIND11_EMBEDDED_MODULE — submodules can be
* defined from any translation unit as long as it happens during that one call.
*/

#include "RtPyCommon.h"

#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace rtpy {
namespace {

py::dict lightToDict(const rtapi::LightInfo& info) {
    py::dict item;
    item["index"] = info.index;
    item["name"] = info.name;
    item["type"] = info.type;
    item["position"] = vec3ToPython(info.position);
    item["direction"] = vec3ToPython(info.direction);
    item["color"] = vec3ToPython(info.color);
    item["intensity"] = info.intensity;
    item["radius"] = info.radius;
    item["spot_angle"] = info.spot_angle;
    item["spot_falloff"] = info.spot_falloff;
    item["width"] = info.width;
    item["height"] = info.height;
    item["visible"] = info.visible;
    return item;
}

py::dict materialToDict(const rtapi::MaterialInfo& info) {
    py::dict item;
    item["id"] = info.id;
    item["name"] = info.name;
    item["type"] = info.type;
    return item;
}

} // namespace

void registerSceneBindings(py::module_& module) {
    // -----------------------------------------------------------------------
    // rt.select — most editor operations are selection-driven. Not undoable.
    // -----------------------------------------------------------------------
    py::module_ select = module.def_submodule(
        "select", "Scene selection (drives the editor's selection-based operations)");

    select.def("list", [] {
        py::list result;
        for (const rtapi::SelectionItem& item : rtapi::listSelection()) {
            py::dict entry;
            entry["type"] = item.type;
            entry["name"] = item.name;
            entry["index"] = item.index;
            entry["primary"] = item.primary;
            result.append(std::move(entry));
        }
        return result;
    }, "Currently selected items; 'primary' marks the one the gizmo follows.");

    select.def("object", [](const std::string& name, bool additive) {
        requireResult(rtapi::selectObject(name, additive));
    }, py::arg("name"), py::arg("additive") = false);

    select.def("deselect", [](const std::string& name) {
        requireResult(rtapi::deselectObject(name));
    }, py::arg("name"));

    select.def("light", [](int index, bool additive) {
        requireResult(rtapi::selectLight(index, additive));
    }, py::arg("index"), py::arg("additive") = false);

    select.def("all", [] {
        int count = 0;
        requireResult(rtapi::selectAllObjects(count));
        return count;
    }, "Selects every non-deleted flat mesh; returns how many were selected.");

    select.def("clear", [] { requireResult(rtapi::clearSelection()); });

    // -----------------------------------------------------------------------
    // rt.material — get/set edit parameters through an OBJECT; the asset calls
    // manage the materials themselves.
    // -----------------------------------------------------------------------
    py::module_ materials = module.def_submodule(
        "material", "Material parameters and material asset management");

    materials.def("get", [](const std::string& object_name, const std::string& param) -> py::object {
        rtapi::MaterialParamValue value;
        requireResult(rtapi::getMaterialParam(object_name, param, value));
        if (value.is_color) return vec3ToPython(value.color);
        return py::float_(value.scalar);
    }, py::arg("object_name"), py::arg("param"));

    materials.def("set", [](const std::string& object_name, const std::string& param,
                            const py::handle& value) {
        if (PyFloat_Check(value.ptr()) || PyLong_Check(value.ptr())) {
            requireResult(rtapi::setMaterialParam(object_name, param, py::cast<float>(value)));
        } else {
            requireResult(rtapi::setMaterialParam(object_name, param, vec3FromPython(value)));
        }
    }, py::arg("object_name"), py::arg("param"), py::arg("value"));

    materials.def("list", [] {
        py::list result;
        for (const rtapi::MaterialInfo& info : rtapi::listMaterials())
            result.append(materialToDict(info));
        return result;
    }, "Every registered material asset.");

    materials.def("info", [](const std::string& name) {
        rtapi::MaterialInfo info;
        requireResult(rtapi::getMaterial(name, info));
        return materialToDict(info);
    }, py::arg("name"));

    materials.def("create", [](const std::string& type, const std::string& name) {
        std::string created;
        requireResult(rtapi::createMaterial(type, name, created));
        return created;
    }, py::arg("type") = std::string("principled"), py::arg("name") = std::string(),
       "Creates a material asset. Returns the registered name, which may differ "
       "from the requested one when that name was taken.");

    materials.def("of_object", [](const std::string& object_name) {
        return rtapi::objectMaterials(object_name);
    }, py::arg("object_name"), "Material names used by an object.");

    materials.def("assign", [](const std::string& object_name, const std::string& material_name) {
        requireResult(rtapi::assignMaterial(object_name, material_name));
    }, py::arg("object_name"), py::arg("material_name"),
       "Replaces the object's whole material assignment.");

    materials.def("set_texture", [](const std::string& material_name, const std::string& slot,
                                    const std::string& filepath) {
        requireResult(rtapi::setMaterialTexture(material_name, slot, filepath));
    }, py::arg("material_name"), py::arg("slot"), py::arg("filepath"),
       "Slots: base_color|roughness|metallic|normal|emission|opacity|specular|"
       "transmission|height (Principled BSDF only).");

    materials.def("clear_texture", [](const std::string& material_name, const std::string& slot) {
        requireResult(rtapi::clearMaterialTexture(material_name, slot));
    }, py::arg("material_name"), py::arg("slot"));

    materials.def("textures", [](const std::string& material_name) {
        return rtapi::materialTextureSlots(material_name);
    }, py::arg("material_name"), "Slot names that currently hold a texture.");

    // -----------------------------------------------------------------------
    // rt.lights — indexed into scene.lights; undoable.
    // -----------------------------------------------------------------------
    py::module_ lights = module.def_submodule("lights", "Scene light operations");

    lights.def("list", [] {
        py::list result;
        for (const rtapi::LightInfo& info : rtapi::listLights())
            result.append(lightToDict(info));
        return result;
    });

    lights.def("get", [](int index) {
        rtapi::LightInfo info;
        requireResult(rtapi::getLight(index, info));
        return lightToDict(info);
    }, py::arg("index"));

    lights.def("add", [](const std::string& type, const py::handle& position) {
        std::string name;
        requireResult(rtapi::addLight(type, vec3FromPython(position), name));
        return name;
    }, py::arg("type"), py::arg("position"));

    lights.def("delete", [](int index) { requireResult(rtapi::deleteLight(index)); },
               py::arg("index"));

    lights.def("set_position", [](int index, const py::handle& position) {
        requireResult(rtapi::setLightPosition(index, vec3FromPython(position)));
    }, py::arg("index"), py::arg("position"));

    lights.def("set_direction", [](int index, const py::handle& direction) {
        requireResult(rtapi::setLightDirection(index, vec3FromPython(direction)));
    }, py::arg("index"), py::arg("direction"),
       "Directional and spot lights only.");

    lights.def("set_color", [](int index, const py::handle& color) {
        requireResult(rtapi::setLightColor(index, vec3FromPython(color)));
    }, py::arg("index"), py::arg("color"));

    lights.def("set_intensity", [](int index, float intensity) {
        requireResult(rtapi::setLightIntensity(index, intensity));
    }, py::arg("index"), py::arg("intensity"));

    lights.def("set_visible", [](int index, bool visible) {
        requireResult(rtapi::setLightVisible(index, visible));
    }, py::arg("index"), py::arg("visible"));

    lights.def("rename", [](int index, const std::string& name) {
        requireResult(rtapi::renameLight(index, name));
    }, py::arg("index"), py::arg("name"));

    lights.def("set_param", [](int index, const std::string& param, float value) {
        requireResult(rtapi::setLightParam(index, param, value));
    }, py::arg("index"), py::arg("param"), py::arg("value"),
       "radius | spot_angle | spot_falloff | width | height | intensity");
}

} // namespace rtpy
