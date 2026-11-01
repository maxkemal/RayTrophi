#include "RtMatterLanes.h"
#include <exception>

bool dispatchMatterLanesIpc(const std::string& method, const nlohmann::json& params,
    const RtIpcTemplateEnqueue& enqueue, nlohmann::json& out) {
    if (method == "fluid.get_lanes") {
        const std::string domain = params.at("domain").get<std::string>();
        out = enqueue([domain](UIContext&) {
            try {
                return rtapi::getMatterDomainLanes(domain);
            } catch (const std::exception& error) {
                return nlohmann::json{{"__error", error.what()}};
            }
        });
        return true;
    }
    if (method == "fluid.set_lanes") {
        const std::string domain = params.at("domain").get<std::string>();
        const bool gas = params.at("gas_lane_enabled").get<bool>();
        const bool liquid = params.at("liquid_lane_enabled").get<bool>();
        out = enqueue([domain, gas, liquid](UIContext&) {
            try {
                return rtapi::setMatterDomainLanes(domain, gas, liquid);
            } catch (const std::exception& error) {
                return nlohmann::json{{"__error", error.what()}};
            }
        });
        return true;
    }
    return false;
}
