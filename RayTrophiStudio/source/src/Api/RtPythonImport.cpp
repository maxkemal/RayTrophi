#include "RtImportBindings.h"
#include "Api/RtApi.h"
#include <pybind11/pybind11.h>
#include <stdexcept>
void registerImportPython(pybind11::module_& scene) {
    scene.def("get_fbx_reader", &rtapi::getFbxReader, "Current session FBX reader.");
    scene.def("set_fbx_reader", [](const std::string& reader) {
        const auto result = rtapi::setFbxReader(reader);
        if (!result.ok) throw std::invalid_argument(result.error);
    }, pybind11::arg("reader"), "Select assimp or ufbx (static FBX only in increment 1). No fallback.");
}
