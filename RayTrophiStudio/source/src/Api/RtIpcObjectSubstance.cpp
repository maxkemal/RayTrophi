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
