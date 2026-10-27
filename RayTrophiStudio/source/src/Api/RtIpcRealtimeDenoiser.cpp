#include "RtIpcRealtimeDenoiser.h"
#include "RtRealtimeDenoiserBindings.h"
#include "Api/RtApi.h"

#include <limits>
#include <stdexcept>

bool dispatchRealtimeDenoiserIpc(const std::string& method, const nlohmann::json& params,
    const RtIpcTemplateEnqueue& enqueue, nlohmann::json& result) {
    using json = nlohmann::json;
    if ((method == "render.realtime_denoiser_stats" ||
         method == "render.set_realtime_denoiser") && !params.is_object()) {
        throw std::invalid_argument("realtime denoiser parameters must be an object");
    }
    if (method == "render.realtime_denoiser_stats") {
        bool synchronize = true;
        for (auto it = params.begin(); it != params.end(); ++it) {
            if (it.key() != "synchronize" || !it.value().is_boolean()) {
                throw std::invalid_argument("only boolean synchronize is accepted");
            }
        }
        if (params.contains("synchronize")) {
            synchronize = params["synchronize"].get<bool>();
        }
        result = enqueue([synchronize](UIContext&) {
            return realtimeDenoiserDictionary<json>(synchronize);
        });
        return true;
    }
    if (method == "render.set_realtime_denoiser") {
        std::optional<std::string> mode;
        std::optional<int> iterations, history;
        std::optional<float> clip, still;
        std::optional<bool> debugHistory, sceneMotion;
        for (auto it = params.begin(); it != params.end(); ++it) {
            if (it.key() != "mode" && it.key() != "atrous_iterations" &&
                it.key() != "history_max" && it.key() != "firefly_clip" &&
                it.key() != "still_noise" && it.key() != "debug_history" &&
                it.key() != "scene_motion") {
                throw std::invalid_argument("unknown realtime denoiser field: " + it.key());
            }
        }
        if (params.contains("mode")) {
            if (!params["mode"].is_string()) {
                throw std::invalid_argument("mode must be a string");
            }
            mode = params["mode"].get<std::string>();
        }
        auto integer = [&](const char* key) {
            const auto& value = params[key];
            if (!value.is_number_integer() || value.get<double>() < 1 ||
                value.get<double>() > 64) {
                throw std::invalid_argument(std::string(key) + " must be an integer in [1, 64]");
            }
            return value.get<int>();
        };
        if (params.contains("atrous_iterations")) {
            iterations = integer("atrous_iterations");
        }
        if (params.contains("history_max")) {
            history = integer("history_max");
        }
        if (params.contains("firefly_clip")) {
            if (!params["firefly_clip"].is_number()) {
                throw std::invalid_argument("firefly_clip must be numeric");
            }
            clip = params["firefly_clip"].get<float>();
        }
        if (params.contains("still_noise")) {
            if (!params["still_noise"].is_number()) {
                throw std::invalid_argument("still_noise must be numeric (0 = off)");
            }
            still = params["still_noise"].get<float>();
        }
        if (params.contains("debug_history")) {
            if (!params["debug_history"].is_boolean()) {
                throw std::invalid_argument("debug_history must be a boolean");
            }
            debugHistory = params["debug_history"].get<bool>();
        }
        if (params.contains("scene_motion")) {
            if (!params["scene_motion"].is_boolean()) {
                throw std::invalid_argument("scene_motion must be a boolean");
            }
            sceneMotion = params["scene_motion"].get<bool>();
        }
        result = enqueue([=](UIContext&) {
            const auto update = rtapi::setRealtimeDenoiser(mode, iterations, history, clip, still,
                debugHistory, sceneMotion);
            if (!update.ok) {
                throw std::invalid_argument(update.error);
            }
            return realtimeDenoiserDictionary<json>(false);
        });
        return true;
    }
    return false;
}
