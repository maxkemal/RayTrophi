// SPDX-License-Identifier: MIT
#include "RtUvBindings.h"

bool dispatchUvIpc(const std::string& method, const nlohmann::json& params,
                   const RtIpcTemplateEnqueue& enqueue, nlohmann::json& result) {
    if (method == "uv.capabilities" || method == "uv.targets" ||
        method == "uv.info" || method == "uv.layout" || method == "uv.transform" ||
        method == "uv.normalize" || method == "uv.pack" || method == "uv.unwrap" ||
        method == "uv.open" || method == "uv.flip_u" || method == "uv.flip_v" ||
        method == "uv.swap_uv" || method == "uv.project_view") {
        result = enqueue([operation = method.substr(3), params](UIContext&) {
            return invokeUv(operation, params);
        });
        return true;
    }
    return false;
}
