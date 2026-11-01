#include "RtSimulationTransport.h"
#include <exception>

bool dispatchSimulationTransportIpc(const std::string& method, const nlohmann::json& params,
    const RtIpcTemplateEnqueue& enqueue, nlohmann::json& result) {
    if (method != "timeline.play" && method != "timeline.pause" &&
        method != "timeline.transport" && method != "sim.set_mode" &&
        method != "sim.get_mode") {
        return false;
    }
    std::string mode;
    if (method == "sim.set_mode") {
        mode = params.at("mode").get<std::string>();
    }
    result = enqueue([method, mode](UIContext&) {
        try {
            if (method == "timeline.play" || method == "timeline.pause") {
                return rtapi::setTimelinePlaying(method == "timeline.play");
            }
            if (method == "sim.set_mode") {
                return rtapi::setSimulationMode(mode);
            }
            if (method == "timeline.transport" || method == "sim.get_mode") {
                return rtapi::timelineTransport();
            }
            return rtapi::timelineTransport();
        } catch (const std::exception& error) {
            return nlohmann::json{{"__error", error.what()}};
        }
    });
    return true;
}
