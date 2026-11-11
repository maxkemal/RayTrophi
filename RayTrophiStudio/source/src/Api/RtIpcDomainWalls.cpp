#include "RtIpcDomainWalls.h"
#include "Api/DomainWallsApi.h"
#include <stdexcept>

namespace {
std::string requireString(const nlohmann::json& params, const char* key) {
    if (!params.is_object() || !params.contains(key) || !params.at(key).is_string()) {
        throw std::runtime_error(std::string(key) + " must be a string");
    }
    return params.at(key).get<std::string>();
}
} // namespace

bool dispatchDomainWallsIpc(const std::string& method, const nlohmann::json& params,
    const RtIpcParticleEnqueue& enqueue, nlohmann::json& output) {
    if (method == "fluid.get_wall_contact") {
        const auto domain = requireString(params, "domain");
        output = enqueue([domain](UIContext&) {
            nlohmann::json result;
            const auto status = rtapi::getDomainWallContact(domain, result);
            return status.ok ? result : nlohmann::json{{"__error", status.error}};
        });
        return true;
    }
    if (method == "fluid.set_wall_contact") {
        const auto domain = requireString(params, "domain");
        const auto patch = params.value("settings", nlohmann::json::object());
        output = enqueue([domain, patch](UIContext&) {
            nlohmann::json result;
            const auto status = rtapi::setDomainWallContact(domain, patch, result);
            return status.ok ? result : nlohmann::json{{"__error", status.error}};
        });
        return true;
    }
    return false;
}
