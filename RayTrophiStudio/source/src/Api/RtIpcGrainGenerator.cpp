#include "RtIpcGrainGenerator.h"
#include "Api/GrainGeneratorApi.h"

bool dispatchGrainGeneratorIpc(const std::string& method, const nlohmann::json& params,
    const RtIpcParticleEnqueue& enqueue, nlohmann::json& output) {
    if (method == "scene.generate_grains") {
        output = enqueue([params](UIContext&) {
            nlohmann::json result;
            const auto status = rtapi::generateGrains(params, result);
            return status.ok ? result : nlohmann::json{{"__error", status.error}};
        });
        return true;
    }
    return false;
}
