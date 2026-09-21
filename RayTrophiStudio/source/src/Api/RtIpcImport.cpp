#include "RtImportBindings.h"
#include "Api/RtApi.h"
using json = nlohmann::json;
bool dispatchImportIpc(const std::string& method, const json& params,
                       const RtIpcTemplateEnqueue& enqueue, json& out) {
    if (method == "scene.get_fbx_reader") {
        out = enqueue([](UIContext&) { return json(rtapi::getFbxReader()); });
        return true;
    }
    if (method == "scene.set_fbx_reader") {
        if (!params.contains("reader") || !params["reader"].is_string()) {
            out = {{"__error", "reader must be a string: 'assimp' or 'ufbx'"}, {"code", "invalid_parameter"}};
            return true;
        }
        const std::string reader = params.at("reader").get<std::string>();
        out = enqueue([reader](UIContext&) {
            const auto result = rtapi::setFbxReader(reader);
            if (!result.ok) return json{{"__error", result.error}, {"code", "invalid_parameter"}};
            return json{{"ok", true}};
        });
        return true;
    }
    return false;
}
