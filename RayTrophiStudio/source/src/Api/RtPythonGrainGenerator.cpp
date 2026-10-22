#include "RtPythonGrainGenerator.h"
#include "Api/GrainGeneratorApi.h"

#include <stdexcept>

namespace rtpy {

void registerGrainGeneratorBindings(pybind11::module_& scene) {
    scene.def("generate_grains", [](const pybind11::kwargs& kwargs) {
        const auto json_module = pybind11::module_::import("json");
        const auto input = nlohmann::json::parse(
            json_module.attr("dumps")(kwargs).cast<std::string>());
        nlohmann::json output;
        const auto result = rtapi::generateGrains(input, output);
        if (!result.ok) {
            throw std::runtime_error(result.error);
        }
        return json_module.attr("loads")(output.dump());
    }, "Generate editable flat rock, Voronoi, snow or crystal meshes. "
       "Same settings and errors as IPC scene.generate_grains; one undo step.");
}

} // namespace rtpy
