#include "RtIpcUpscale.h"
#include "RtUpscaleBindings.h"
#include "Api/RtApi.h"
#include <stdexcept>

bool dispatchUpscaleIpc(const std::string& method, const nlohmann::json& params,
    const RtIpcTemplateEnqueue& enqueue, nlohmann::json& result) {
    using json = nlohmann::json;
    if (method == "render.upscale_stats") {
        bool synchronize = true;
        if (!params.is_object()) {
            throw std::invalid_argument("upscale parameters must be an object");
        }
        for (auto it = params.begin(); it != params.end(); ++it) {
            if (it.key() != "synchronize" || !it.value().is_boolean()) {
                throw std::invalid_argument("only boolean synchronize is accepted");
            }
        }
        if (params.contains("synchronize")) {
            synchronize = params["synchronize"].get<bool>();
        }
        result = enqueue([synchronize](UIContext&) {
            return upscaleDictionary<json>(synchronize);
        });
        return true;
    }
    if (method == "render.set_upscale") {
        if (!params.is_object()) {
            throw std::invalid_argument("upscale parameters must be an object");
        }
        for (auto it = params.begin(); it != params.end(); ++it) {
            if (it.key() != "mode" && it.key() != "quality" && it.key() != "sharpness" &&
                it.key() != "rejection_reactive") {
                throw std::invalid_argument("unknown upscale parameter: " + it.key());
            }
        }
        std::optional<std::string> mode, quality;
        std::optional<float> sharpness;
        if (params.contains("mode")) {
            if (!params["mode"].is_string()) {
                throw std::invalid_argument("mode must be a string");
            }
            mode = params["mode"].get<std::string>();
        }
        if (params.contains("quality")) {
            if (!params["quality"].is_string()) {
                throw std::invalid_argument("quality must be a string");
            }
            quality = params["quality"].get<std::string>();
        }
        if (params.contains("sharpness")) {
            if (!params["sharpness"].is_number()) {
                throw std::invalid_argument("sharpness must be numeric");
            }
            sharpness = params["sharpness"].get<float>();
        }
        std::optional<bool> rejectionReactive;
        if (params.contains("rejection_reactive")) {
            if (!params["rejection_reactive"].is_boolean()) {
                throw std::invalid_argument("rejection_reactive must be a boolean");
            }
            rejectionReactive = params["rejection_reactive"].get<bool>();
        }
        result = enqueue([=](UIContext&) {
            const auto update = rtapi::setUpscale(mode, quality, sharpness, rejectionReactive);
            if (!update.ok) {
                throw std::invalid_argument(update.error);
            }
            return upscaleDictionary<json>(false);
        });
        return true;
    }
    return false;
}
