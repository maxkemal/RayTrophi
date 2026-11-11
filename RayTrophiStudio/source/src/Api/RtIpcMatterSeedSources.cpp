#include "RtIpcMatterSeedSources.h"
#include "Api/DomainCreationApi.h"
#include "Api/MatterSeedSourcesApi.h"
#include "Fluid/MatterSeedSources.h"

bool dispatchMatterSeedSourcesIpc(const std::string &method, const nlohmann::json &params,
                                  const RtIpcParticleEnqueue &enqueue, nlohmann::json &output) {
    if (method == "fluid.create_placed_domain") {
        const auto name = params.value("name", std::string());
        const auto source = params.value("source_object", std::string());
        output = enqueue([name, source](UIContext &) {
            nlohmann::json result;
            const auto status = rtapi::createPlacedMatterDomain(name, source, result);
            return status.ok ? result : nlohmann::json{{"__error", status.error}};
        });
        return true;
    }
    // nlohmann's get<> aborts the process on a wrong type in this build
    // (JSON_NOEXCEPTION): an agent passing domain = null killed the app with an
    // unsaved scene (2026-11-09). Check the type, answer with an error instead.
    std::string domain_param;
    const auto readDomain = [&]() {
        const auto found = params.find("domain");
        if (found == params.end() || !found->is_string()) {
            output = nlohmann::json{{"__error", "missing or invalid string param: domain"}};
            return false;
        }
        domain_param = found->get<std::string>();
        return true;
    };
    const auto invoke = [&](const std::string &operation, const std::string &domain, uint64_t id,
                            const nlohmann::json &settings) {
        output = enqueue([domain, operation, id, settings](UIContext &) {
            nlohmann::json result;
            const auto status = rtapi::matterSeedSources(domain, operation, id, settings, result);
            return status.ok ? result : nlohmann::json{{"__error", status.error}};
        });
        return true;
    };
    if (method == "fluid.get_seed_sources") {
        if (!readDomain()) return true;
        const auto domain = domain_param;
        return invoke("list", domain, 0, nlohmann::json::object());
    }
    if (method == "fluid.select_seed_source") {
        if (!readDomain()) return true;
        const auto domain = domain_param;
        const auto value = params.value("id", nlohmann::json(0));
        const uint64_t id = value.is_number_integer() && value == 0
                                ? 0
                                : RayTrophiSim::Fluid::matterSeedSourceId(value);
        return invoke("select", domain, id, nlohmann::json::object());
    }
    if (method == "fluid.add_seed_source") {
        if (!readDomain()) return true;
        const auto domain = domain_param;
        const auto settings = params.at("settings");
        return invoke("add", domain, 0, settings);
    }
    if (method == "fluid.update_seed_source") {
        if (!readDomain()) return true;
        const auto domain = domain_param;
        const uint64_t id = RayTrophiSim::Fluid::matterSeedSourceId(params.at("id"));
        const auto settings = params.at("settings");
        return invoke("update", domain, id, settings);
    }
    if (method == "fluid.remove_seed_source") {
        if (!readDomain()) return true;
        const auto domain = domain_param;
        const uint64_t id = RayTrophiSim::Fluid::matterSeedSourceId(params.at("id"));
        return invoke("remove", domain, id, nlohmann::json::object());
    }
    if (method == "fluid.apply_seed_sources") {
        if (!readDomain()) return true;
        const auto domain = domain_param;
        return invoke("apply", domain, 0, nlohmann::json::object());
    }
    if (method == "fluid.set_seed_source_settings") {
        if (!readDomain()) return true;
        const auto domain = domain_param;
        const auto settings = params.at("settings");
        return invoke("settings", domain, 0, settings);
    }
    return false;
}
