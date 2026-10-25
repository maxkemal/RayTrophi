#include "Fluid/SourceSplatGeometryJson.h"
#include "RtIpcSourceSplatGeometry.h"
#include "Api/RtApi.h"
#include "Api/SourceSplatGeometryApi.h"
#include "Api/MatterPhaseOutputApi.h"
#include "Fluid/MatterPhaseOutputJson.h"

bool dispatchSourceSplatGeometryIpc(const std::string& method, const nlohmann::json& params,
    const RtIpcParticleEnqueue& enqueue, nlohmann::json& output) {
    if (method == "fluid.inspect_fog_fields") {
        output = enqueue([params](UIContext&) {
            try {
                for (auto it = params.begin(); it != params.end(); ++it) {
                    if (it.key() != "domain") {
                        return nlohmann::json{{"__error", "unknown inspection parameter: " +
                            it.key()}};
                    }
                }
                nlohmann::json result;
                const auto status = rtapi::inspectMatterFogFields(
                    params.at("domain").get<std::string>(), result);
                return status.ok ? result : nlohmann::json{{"__error", status.error}};
            } catch (const std::exception& error) {
                return nlohmann::json{{"__error", error.what()}};
            }
        });
        return true;
    }
    if (method == "fluid.get_phase_output" || method == "fluid.set_phase_output") {
        output = enqueue([method, params](UIContext&) {
            try {
                const auto domain = params.at("domain").get<std::string>();
                RayTrophiSim::Fluid::MatterPhaseOutput value;
                auto status = rtapi::getPhaseOutput(domain, value);
                if (status.ok && method == "fluid.set_phase_output") {
                    auto patch = params;
                    patch.erase("domain");
                    RayTrophiSim::Fluid::from_json(patch, value);
                    status = rtapi::setPhaseOutput(domain, value);
                }
                return status.ok ? nlohmann::json(value) :
                    nlohmann::json{{"__error", status.error}};
            } catch (const std::exception& error) {
                return nlohmann::json{{"__error", error.what()}};
            }
        });
        return true;
    }
    if (method == "fluid.get_substance_splat_geometry") {
        output = enqueue([params](UIContext&) {
            RayTrophiSim::Fluid::SourceSplatGeometry geometry;
            const auto status = rtapi::getSubstanceSplatGeometry(
                params.at("domain").get<std::string>(),
                params.at("substance").get<std::string>(), geometry);
            return status.ok ? nlohmann::json(geometry) :
                nlohmann::json{{"__error", status.error}};
        });
        return true;
    }
    if (method == "fluid.set_substance_splat_geometry") {
        output = enqueue([method, params](UIContext&) {
            try {
                const std::string domain = params.at("domain").get<std::string>();
                const std::string substance = params.at("substance").get<std::string>();
                RayTrophiSim::Fluid::SourceSplatGeometry geometry;
                auto status = rtapi::getSubstanceSplatGeometry(domain, substance, geometry);
                if (status.ok) {
                    nlohmann::json patch = params;
                    patch.erase("domain");
                    patch.erase("substance");
                    RayTrophiSim::Fluid::from_json(patch, geometry);
                    status = rtapi::setSubstanceSplatGeometry(domain, substance, geometry);
                }
                return status.ok ? nlohmann::json(geometry) :
                    nlohmann::json{{"__error", status.error}};
            } catch (const std::exception& error) {
                return nlohmann::json{{"__error", error.what()}};
            }
        });
        return true;
    }
    if (method == "flow_source.set_splat_geometry") {
        output = enqueue([params](UIContext&) {
            try {
                const std::string name = params.at("name").get<std::string>();
                rtapi::SimulationFlowSourceInfo source;
                auto status = rtapi::getSimulationFlowSource(name, source);
                if (status.ok) {
                    nlohmann::json patch = params;
                    patch.erase("name");
                    RayTrophiSim::Fluid::from_json(patch, source.splat_geometry);
                    status = rtapi::setSourceSplatGeometry(name, source.splat_geometry);
                }
                return status.ok ? nlohmann::json{{"ok", true}} :
                    nlohmann::json{{"__error", status.error}};
            } catch (const std::exception& error) {
                return nlohmann::json{{"__error", error.what()}};
            }
        });
        return true;
    }
    return false;
}
