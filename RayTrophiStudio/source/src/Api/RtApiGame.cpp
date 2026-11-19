#include "Api/RtApiGame.h"
#include "Game/GameRuntime.h"
#include "Game/GameSession.h"
#include "RtApiInternal.h"
#include "Paint/BalloonPaintDemo.h"
#include "UI/SimulationTransport.h"
#include "globals.h"
#include "scene_ui.h"

#include <chrono>

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

namespace {
double steadySeconds() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

nlohmann::json playtestStatus() {
    const Game::PlaytestState &p = Game::playtest();
    nlohmann::json demo = {{"prepared", false}, {"running", false}};
    nlohmann::json st;
    std::string error;
    if (Paint::balloonPaintCommand(g_ctx->scene, "status", nlohmann::json::object(), st, error)) {
        demo["prepared"] = st.value("prepared", false);
        demo["running"] = st.value("running", false);
    }
    demo["started_by_playtest"] = p.started_demo;
    return {{"active", p.active},
            {"mode", g_sim_timeline_mode ? "timeline" : "live"},
            {"seconds", p.active ? steadySeconds() - p.started_seconds : 0.0},
            {"particles", p.active ? "billboards" : "scene"},
            {"background", p.active || !g_sim_timeline_mode ? "steps" : "sleeps"},
            {"session_configured", Game::session().configured()},
            {"demo", demo}};
}
} // namespace

Result gamePlaytest(const std::string &operation, nlohmann::json &output) {
    if (!g_ctx) {
        return notBound();
    }
    if (operation != "start" && operation != "stop" && operation != "status") {
        return Result::fail("operation must be start, stop or status");
    }
    Game::PlaytestState &p = Game::playtest();
    if (operation != "status" && (renderJobActive() || rendering_in_progress.load())) {
        return Result::fail("scene is locked by the final render job");
    }
    if (operation == "start" && !p.active) {
        p = Game::PlaytestState{};
        p.previous_timeline_mode = g_sim_timeline_mode;
        SimulationTransport::setTimelineMode(*g_ctx, false);
        // The Player starts the demo on project load when its design says so.
        nlohmann::json st;
        std::string error;
        if (Paint::balloonPaintCommand(g_ctx->scene, "status", nlohmann::json::object(), st, error) &&
            st.value("prepared", false) && !st.value("running", false) &&
            st.contains("config") && st["config"].value("autostart", false)) {
            nlohmann::json ignored;
            p.started_demo = Paint::balloonPaintCommand(g_ctx->scene, "start",
                                                        nlohmann::json::object(), ignored, error);
        }
        p.started_seconds = steadySeconds();
        p.active = true;
        g_ctx->start_render = true;
    } else if (operation == "stop" && p.active) {
        if (p.started_demo) {
            nlohmann::json ignored;
            std::string error;
            Paint::balloonPaintCommand(g_ctx->scene, "stop", nlohmann::json::object(), ignored, error);
        }
        SimulationTransport::setTimelineMode(*g_ctx, p.previous_timeline_mode);
        p = Game::PlaytestState{};
        g_ctx->start_render = true;
    }
    output = playtestStatus();
    return Result::success();
}
} // namespace rtapi
