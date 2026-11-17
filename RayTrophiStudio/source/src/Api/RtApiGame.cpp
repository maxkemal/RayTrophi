#include "Api/RtApiGame.h"
#include "Game/GameRuntime.h"
#include "Game/GameSession.h"
#include "RtApiInternal.h"

namespace rtapi {
Result gameSessionCommand(const std::string &operation, const nlohmann::json &config,
                          nlohmann::json &output) {
    if (!g_ctx) {
        return notBound();
    }
    if (operation != "status" && renderJobActive()) {
        return Result::fail("scene is locked by the final render job");
    }
    try {
        if (!config.is_object() || (operation != "configure" && !config.empty())) {
            return Result::fail("session_config_only_allowed_for_configure");
        }
        std::string error;
        return Game::runtimeCommand(g_ctx->scene, operation, config, output, error)
                   ? Result::success()
                   : Result::fail(error);
    } catch (const std::exception &exception) {
        return Result::fail(std::string("invalid_session_request: ") + exception.what());
    }
}

Result postGameEvent(const nlohmann::json &event) {
    if (!g_ctx) {
        return notBound();
    }
    if (renderJobActive()) {
        return Result::fail("scene is locked by the final render job");
    }
    std::string error;
    return Game::postExternalEvent(event, error) ? Result::success() : Result::fail(error);
}

Result getGameRules(nlohmann::json &output) {
    if (!g_ctx) {
        return notBound();
    }
    output = Game::session().snapshot()["rules"];
    return Result::success();
}
} // namespace rtapi
