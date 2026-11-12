#include "RtDomainVisualBindings.h"
#include "Api/DomainVisualApi.h"

namespace {
uint64_t unsignedParameter(const nlohmann::json& value, const char* key) {
    if ((!value.is_number_integer() && !value.is_number_unsigned()) ||
        (!value.is_number_unsigned() && value.get<int64_t>() < 0)) {
        throw std::invalid_argument(std::string(key) + " must be a nonnegative integer");
    }
    return value.get<uint64_t>();
}
void keys(const nlohmann::json& params, std::initializer_list<const char*> allowed) {
    if (!params.is_object()) {
        throw std::invalid_argument("params must be an object");
    }
    for (auto it = params.begin(); it != params.end(); ++it) {
        bool recognized = false;
        for (const auto* name : allowed) {
            recognized |= it.key() == name;
        }
        if (!recognized) {
            throw std::invalid_argument("unknown parameter: " + it.key());
        }
    }
}
}

bool dispatchDomainVisualIpc(const std::string& method, const nlohmann::json& params,
    const RtIpcParticleEnqueue& enqueue, nlohmann::json& output) {
    if (method == "fluid.visual_samples") {
        output = enqueue([params](UIContext&) {
            try {
                keys(params, {"domain", "offset", "limit"});
                const auto domain = params.at("domain").get<std::string>();
                const auto offset = unsignedParameter(params.value("offset", nlohmann::json(0)),
                                                       "offset");
                const auto limit = unsignedParameter(params.value("limit", nlohmann::json(0)),
                                                      "limit");
                nlohmann::json result;
                const auto status = rtapi::getDomainVisualSamples(domain, result, offset, limit);
                return status.ok ? result : nlohmann::json{{"__error", status.error}};
            } catch (const std::exception& error) {
                return nlohmann::json{{"__error", error.what()}};
            }
        });
        return true;
    }
    if (method == "fluid.set_visual_samples") {
        output = enqueue([params](UIContext&) {
            try {
                keys(params, {"domain", "settings"});
                const auto domain = params.at("domain").get<std::string>();
                auto status = rtapi::setDomainVisualSamples(domain, params.at("settings"));
                nlohmann::json result;
                if (status.ok) {
                    status = rtapi::getDomainVisualSamples(domain, result);
                }
                return status.ok ? result : nlohmann::json{{"__error", status.error}};
            } catch (const std::exception& error) {
                return nlohmann::json{{"__error", error.what()}};
            }
        });
        return true;
    }
    if (method == "fluid.reset_visual_samples") {
        output = enqueue([params](UIContext&) {
            try {
                keys(params, {"domain"});
                const auto domain = params.at("domain").get<std::string>();
                auto status = rtapi::resetDomainVisualSamples(domain);
                nlohmann::json result;
                if (status.ok) {
                    status = rtapi::getDomainVisualSamples(domain, result);
                }
                return status.ok ? result : nlohmann::json{{"__error", status.error}};
            } catch (const std::exception& error) {
                return nlohmann::json{{"__error", error.what()}};
            }
        });
        return true;
    }
    return false;
}
