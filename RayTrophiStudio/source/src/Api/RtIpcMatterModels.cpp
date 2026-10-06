#include "RtMatterModels.h"

#include <exception>

bool dispatchMatterModelIpc(const std::string& method, const nlohmann::json& params,
                           const RtIpcTemplateEnqueue& enqueue, nlohmann::json& out) {
    if (method == "fluid.grain_settings") {
        const std::string domain = params.at("domain").get<std::string>();
        const bool valid = params.size() == 1;
        out = enqueue([domain, valid](UIContext&) {
            try {
                if (!valid) {
                    throw std::runtime_error("grain_settings accepts domain only");
                }
                return rtapi::matterGrainSettings(domain, nlohmann::json::object(), false);
            } catch (const std::exception& error) {
                return nlohmann::json{{"__error", error.what()}};
            }
        });
        return true;
    }
    if (method == "fluid.set_grain_settings") {
        const std::string domain = params.at("domain").get<std::string>();
        nlohmann::json patch = params;
        patch.erase("domain");
        out = enqueue([domain, patch](UIContext&) {
            try {
                return rtapi::matterGrainSettings(domain, patch, true);
            } catch (const std::exception& error) {
                return nlohmann::json{{"__error", error.what()}};
            }
        });
        return true;
    }
    if (method == "fluid.grain_reference") {
        out = enqueue([params](UIContext&) {
            try {
                return rtapi::runGrainReferenceProbe(params);
            } catch (const std::exception& error) {
                return nlohmann::json{{"__error", error.what()}};
            }
        });
        return true;
    }
    if (method == "fluid.set_pore_exchange") {
        const std::string domain = params.at("domain").get<std::string>();
        auto patch = params;
        patch.erase("domain");
        out = enqueue([domain, patch](UIContext&) {
            try {
                return rtapi::setMatterPoreExchange(domain, patch);
            } catch (const std::exception& error) {
                return nlohmann::json{{"__error", error.what()}};
            }
        });
        return true;
    }
    if (method == "fluid.matter_models") {
        const std::string domain = params.at("domain").get<std::string>();
        const bool include_transfer = params.value("include_transfer", false);
        out = enqueue([domain, include_transfer](UIContext&) {
            try {
                return rtapi::getMatterModels(domain, include_transfer);
            } catch (const std::exception& error) {
                return nlohmann::json{{"__error", error.what()}};
            }
        });
        return true;
    }
    return false;
}
