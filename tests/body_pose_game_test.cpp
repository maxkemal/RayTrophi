// Standalone native contract tests. Source delivered; user builds/runs them.
// cl /std:c++17 /EHsc /UNDEBUG /I RayTrophiStudio/source/include
//   tests/body_pose_game_test.cpp source/src/Input/BodyPoseRuntime.cpp
//   source/src/Game/GameRules.cpp source/src/Game/GameSession.cpp
#include "Input/BodyPoseRuntime.h"
#include "Game/GameSession.h"
#include <cassert>
#include <chrono>
#include <limits>

using json = nlohmann::json;

json packet(double t, double wrist_y = 0.7, double ankle_y = 0.0) {
    json joints = json::object();
    for (const char *name : Input::kBodyJoints) {
        joints[name] = {0.0, 0.6, 1.5, 1.0};
    }
    joints["head"] = {0, 1.2, 1.5, 1};
    joints["l_wrist"] = {-0.2, wrist_y, 1.5, 1};
    joints["r_wrist"] = {0.2, wrist_y, 1.5, 1};
    joints["r_ankle"] = {0.1, ankle_y, 1.5, 1};
    return {{"t", t},
            {"persons", json::array({{{"id", 1}, {"confidence", 1.0}, {"joints", joints}}})}};
}

void clearEvents() {
    std::vector<Game::Event> discarded;
    Game::drainEvents(discarded);
}

int main() {
    clearEvents();
    Input::BodyPoseRuntime body;
    const auto now = Input::BodyPoseRuntime::Clock::now();
    std::string error;
    assert(body.set(packet(0), error, now));
    const auto before = body.snapshot(now);
    auto invalid = packet(0.1);
    invalid["persons"][0]["joints"]["head"][0] = std::numeric_limits<double>::infinity();
    assert(!body.set(invalid, error, now));
    assert(body.snapshot(now) == before); // validation is transactional
    assert(!body.set(packet(0), error, now));
    assert(body.set(packet(0.1, 1.5), error, now));
    assert(body.set(packet(0.2, 1.5), error, now));
    std::vector<Game::Event> events;
    Game::drainEvents(events);
    assert(events.size() == 1 && events[0].type == "gesture.arms_up");
    assert(body.set(packet(0.3, 0.7, 0.2), error, now));
    assert(body.set(packet(0.4, 0.7, 0.0), error, now));
    events.clear();
    Game::drainEvents(events);
    assert(events.size() == 1 && events[0].type == "gesture.stomp");
    assert(body.set({{"t", 9999}, {"persons", json::array()}}, error, now));
    assert(body.snapshot(now)["persons"].size() == 1); // sender jump does not expire
    const auto later = now + std::chrono::milliseconds(501);
    assert(body.snapshot(later)["persons"].empty());
    body.clear();
    assert(body.set(packet(0), error, later));

    Game::Rules rules;
    json config = json::array({{{"id", "group"},
                                {"event", "contact"},
                                {"min_persons", 2},
                                {"cooldown_seconds", 1},
                                {"action", {{"type", "paint.dab"}}}}});
    assert(rules.configure(config, error));
    std::vector<Game::RuleAction> actions;
    Game::Event event;
    event.type = "contact";
    event.target = "BalloonA";
    event.person = 1;
    rules.evaluate(event, "playing", 0.0, actions);
    assert(actions.empty());
    rules.evaluate(event, "playing", 0.1, actions);
    assert(actions.empty()); // repeated joint/person never counts as two children
    event.target = "BalloonB";
    event.person = 2;
    rules.evaluate(event, "playing", 0.2, actions);
    assert(actions.empty()); // distinct targets never cooperate
    event.target = "BalloonA";
    rules.evaluate(event, "playing", 0.3, actions);
    assert(actions.size() == 1);
    event.target = "BalloonB";
    event.person = 1;
    rules.evaluate(event, "playing", 0.4, actions);
    assert(actions.size() == 2); // another target's partial group survives A firing
    const auto valid_rules = rules.snapshot();
    config[0]["action"]["radius"] = -1;
    assert(!rules.configure(config, error));
    assert(rules.snapshot() == valid_rules);

    clearEvents();
    Game::Session session;
    json output;
    assert(session.command(
        "configure", {{"duration_seconds", 0.5}, {"finale_seconds", 0.25}, {"reset_seconds", 0.25}},
        output, error));
    assert(session.command("start", json::object(), output, error));
    assert(!session.command("start", json::object(), output, error));
    actions.clear();
    session.tick(0.25, {}, 0, actions);
    session.tick(0.25, {}, 0, actions);
    assert(session.snapshot()["state"] == "finale");
    session.tick(0.25, {}, 0, actions);
    assert(session.snapshot()["state"] == "resetting");
    session.tick(0.25, {}, 0, actions);
    assert(session.snapshot()["state"] == "attract");
    assert(session.snapshot()["completed_sessions"] == 1);
    assert(actions.size() == 1 && actions[0].action["type"] == "room.reset");
    assert(session.command("start", json::object(), output, error));
    session.tick(0.01, {}, 1, actions);
    assert(session.snapshot()["state"] == "finale");
    clearEvents();
    return 0;
}
