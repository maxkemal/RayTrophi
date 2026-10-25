#include "RtRayGpuTimingBindings.h"
#include "Api/RayGpuTimingApi.h"
#include <stdexcept>

namespace {
bool requireBool(const nlohmann::json& params, const char* key) {
    if (!params.is_object() || !params.contains(key) || !params.at(key).is_boolean()) {
        throw std::invalid_argument(std::string(key) + " must be a boolean");
    }
    return params.at(key).get<bool>();
}
}

bool dispatchRayGpuTimingIpc(const std::string& method, const nlohmann::json& params,
    const RtIpcParticleEnqueue& enqueue, nlohmann::json& output) {
    if (method == "render.gpu_timing_stats") {
        output = enqueue([params](UIContext&) {
            if (!params.empty()) {
                return nlohmann::json{{"__error", "GPU timing stats accepts no parameters"}};
            }
            return rtapi::rayGpuTimingCaptureStats();
        });
        return true;
    }
    if (method == "render.set_gpu_timing") {
        output = enqueue([params](UIContext&) {
            for (auto it = params.begin(); it != params.end(); ++it) {
                if (it.key() != "enabled") {
                    return nlohmann::json{{"__error", "unknown GPU timing parameter: " + it.key()}};
                }
            }
            bool enabled = false;
            try {
                enabled = requireBool(params, "enabled");
            } catch (const std::exception& error) {
                return nlohmann::json{{"__error", error.what()}};
            }
            const auto result = rtapi::setRayGpuTimingCapture(enabled);
            return result.ok ? rtapi::rayGpuTimingCaptureStats() :
                nlohmann::json{{"__error", result.error}};
        });
        return true;
    }
    return false;
}
