#include "RtIpcGame.h"
#include "Api/RtApi.h"
#include "Api/RtApiGame.h"

using json = nlohmann::json;

bool dispatchGameIpc(const std::string &method, const json &params,
                     const RtIpcPaintWetEnqueue &enqueue, json &output) {
    if (method == "game.session") {
        const auto operation = params.value("operation", std::string("status"));
        const auto config = params.value("config", json::object());
        output = enqueue([operation, config](UIContext &) {
            json result;
            const auto status = rtapi::gameSessionCommand(operation, config, result);
            return status.ok ? result : json{{"__error", status.error}};
        });
        return true;
    }
    if (method == "game.post_event") {
        const auto event = params.value("event", json::object());
        output = enqueue([event](UIContext &) {
            const auto status = rtapi::postGameEvent(event);
            return status.ok ? json{{"ok", true}} : json{{"__error", status.error}};
        });
        return true;
    }
    if (method == "game.rules") {
        output = enqueue([](UIContext &) {
            json result;
            const auto status = rtapi::getGameRules(result);
            return status.ok ? result : json{{"__error", status.error}};
        });
        return true;
    }
    return false;
}
