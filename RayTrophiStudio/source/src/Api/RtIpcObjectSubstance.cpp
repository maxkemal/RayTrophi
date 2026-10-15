#include "RtObjectSubstanceBindings.h"
#include "RtApiInternal.h"
#include "ObjectSubstance.h"
#include <stdexcept>

bool dispatchObjectSubstanceIpc(const std::string& method, const nlohmann::json& params,
                                const RtIpcTemplateEnqueue& enqueue, nlohmann::json& result) {
    if (method == "substance.list_objects") {
        result = enqueue([method, params](UIContext& ctx) {
            return matter::invokeObjectSubstance(ctx, method.substr(10), params);
        });
        return true;
    }
    if (method == "substance.get_object" || method == "substance.clear_object" ||
        method == "substance.adopt_object") {
        const auto object = params.at("object").get<std::string>();
        result = enqueue([method, params, object](UIContext& ctx) {
            auto request = params;
            request["object"] = object;
            return matter::invokeObjectSubstance(ctx, method.substr(10), request);
        });
        return true;
    }
    if (method == "substance.set_object") {
        const auto object = params.at("object").get<std::string>();
        const auto binding = params.at("binding");
        result = enqueue([params, object, binding](UIContext& ctx) {
            auto request = params;
            request["object"] = object;
            request["binding"] = binding;
            return matter::invokeObjectSubstance(ctx, "set_object", request);
        });
        return true;
    }
    return false;
}

namespace rtpy {
void registerObjectSubstanceBindings(pybind11::module_& substance) {
    namespace py = pybind11;
    using json = nlohmann::json;
    for (const char* operation : {"list_objects", "get_object", "set_object", "clear_object",
                                "adopt_object"}) {
        substance.def(operation, [name = std::string(operation)](const py::kwargs& kwargs) {
            if (!rtapi::g_ctx) {
                throw std::runtime_error("No active scene context");
            }
            const auto params = json::parse(py::cast<std::string>(
                py::module_::import("json").attr("dumps")(kwargs)));
            const auto result = matter::invokeObjectSubstance(*rtapi::g_ctx, name, params);
            return py::module_::import("json").attr("loads")(result.dump());
        });
    }
}
} // namespace rtpy
