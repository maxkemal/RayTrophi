#include "Paint/BalloonBodyInteraction.h"
#include "Paint/BalloonPaintDemo.h"
#include "Input/BodyPoseRuntime.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>

namespace Paint {
namespace {
using json = nlohmann::json;
using Clock = Input::BodyPoseRuntime::Clock;
constexpr double kConfidence = 0.6;
constexpr double kFreshSeconds = 0.20;
constexpr double kTouchDelay = 0.24;
constexpr double kGripDelay = 0.12;
constexpr double kReleaseDelay = 0.15;
constexpr double kLossSeconds = 0.80;

struct PersonState {
    Clock::time_point sample{};
    Clock::time_point valid{};
    Clock::time_point pending_since{};
    Clock::time_point releasing_since{};
    int candidate = -1;
    bool grip = false;
    bool touch_left = false;
    bool latched = false;
    Vec3 latch_position;
    float latch_radius = 0.4f;
    Vec3 smooth;
    float span = 0.0f;
    std::string phase = "idle";
};
std::map<int, PersonState> people;
int active_person = -1;
std::uint64_t grabs = 0, moves = 0, releases = 0, taps = 0, losses = 0;
std::uint64_t last_packet = 0;
std::string last_error;

double age(Clock::time_point now, Clock::time_point then) {
    return std::chrono::duration<double>(now - then).count();
}

std::string owner(int id) {
    return "body:" + std::to_string(id) + ":both";
}

bool ownsHold() {
    return active_person >= 0 && heldBalloon() >= 0 &&
           balloonInteractionStatus().value("owner", std::string()) == owner(active_person);
}

void cancelActive() {
    if (ownsHold()) {
        cancelBalloonInteraction();
    }
    active_person = -1;
}

bool command(SceneData& scene, const char* operation, const json& input) {
    json output;
    std::string error;
    if (!balloonPaintCommand(scene, operation, input, output, error)) {
        last_error = error;
        return false;
    }
    const std::string operation_name(operation);
    if (operation_name == "pop") {
        return output.value("popped", -1) >= 0;
    }
    if (operation_name == "pointer") {
        const auto action = input.value("action", std::string());
        return action == "down" ? output.value("held", -1) >= 0 :
            (action == "up" ? output.value("popped", -1) >= 0 : output.value("moved", -1) >= 0);
    }
    return true;
}

Vec3 position(const std::array<double, 4>& point) {
    return Vec3(static_cast<float>(point[0]), static_cast<float>(point[1]),
                static_cast<float>(point[2]));
}

json coordinates(const Vec3& point) {
    return json::array({point.x, point.y, point.z});
}

void latch(PersonState& state, const BalloonInteractionTarget& target) {
    state.latched = true;
    state.latch_position = target.position;
    state.latch_radius = target.radius + 0.25f;
    state.candidate = -1;
    state.phase = "latched";
    state.releasing_since = {};
}
} // namespace

void resetBalloonBodyInteraction() {
    cancelActive();
    people.clear();
    grabs = moves = releases = taps = losses = 0;
    last_packet = Input::bodyPoses().packetCount();
    last_error.clear();
}

json balloonBodyInteractionStatus(bool enabled) {
    json actors = json::array();
    for (const auto& entry : people) {
        actors.push_back({{"person", entry.first}, {"phase", entry.second.phase},
                          {"candidate", entry.second.candidate}});
    }
    return {{"enabled", enabled}, {"active_person", active_person}, {"persons", actors},
            {"grabs", grabs}, {"moves", moves}, {"releases", releases}, {"taps", taps},
            {"tracking_cancels", losses}, {"error", last_error}};
}

bool balloonBodyControlCommand(json& config, const json& input, json& output,
                               std::string& error) {
    if (!input.is_object() || input.size() > 1 ||
        (!input.empty() && (!input.contains("enabled") || !input["enabled"].is_boolean()))) {
        error = "body controls accept only optional boolean enabled";
        return false;
    }
    if (!input.empty()) {
        if (!config.is_object() ||
            (input["enabled"] == true && !config.contains("visual_nodes"))) {
            error = "body controls require a prepared mesh balloon demo";
            return false;
        }
        const bool enabled = input["enabled"].get<bool>();
        if (config.value("body_controls", false) != enabled) {
            resetBalloonBodyInteraction();
            config["body_controls"] = enabled;
        }
    }
    output = balloonBodyInteractionStatus(config.is_object() &&
                                          config.value("body_controls", false));
    return true;
}

void tickBalloonBodyInteraction(SceneData& scene, bool enabled,
                               const std::vector<BalloonInteractionTarget>& targets) {
    if (!enabled) {
        if (!people.empty() || active_person >= 0) {
            resetBalloonBodyInteraction();
        }
        return;
    }
    auto& input = Input::bodyPoses();
    const auto now = Clock::now();
    if (active_person >= 0 && !ownsHold()) {
        // External pop/reset/mode switch ended the hold. Require a fresh gesture.
        people[active_person].latched = true;
        people[active_person].phase = "latched";
        active_person = -1;
    }
    if (active_person >= 0) {
        auto& state = people.at(active_person);
        if (age(now, input.receivedAt(active_person)) > kFreshSeconds) {
            state.phase = "tracking_lost";
            state.releasing_since = {};
        }
        if (age(now, state.valid) >= kLossSeconds) {
            const auto target = std::find_if(targets.begin(), targets.end(),
                [](const auto& value) { return value.id == heldBalloon(); });
            if (target != targets.end()) {
                state.latch_position = target->position;
                state.latch_radius = target->radius + 0.25f;
            }
            state.latched = true;
            state.phase = "tracking_lost";
            ++losses;
            cancelActive();
        }
    }
    // At most ten tracks. Expired pending candidates cannot accumulate across IDs.
    for (auto it = people.begin(); it != people.end();) {
        if (it->first != active_person && age(now, it->second.sample) > 2.0) {
            it = people.erase(it);
        } else {
            ++it;
        }
    }
    if (input.packetCount() == last_packet) {
        return; // Loss timers above still run; pose maps are copied only on new input.
    }
    last_packet = input.packetCount();
    for (const auto& entry : input.persons()) {
        const int id = entry.first;
        const auto& person = entry.second;
        const auto received = input.receivedAt(id);
        if (age(now, received) > kFreshSeconds) {
            continue;
        }
        auto& state = people[id];
        if (state.sample == received) {
            continue; // A retained pose is not a fresh hand observation.
        }
        const double delta = state.sample == Clock::time_point{} ? 0.033 :
                             std::clamp(age(received, state.sample), 0.001, 0.1);
        const bool gap = state.sample != Clock::time_point{} &&
                         age(received, state.sample) > kFreshSeconds;
        state.sample = received;
        const auto& l = person.joints.at("l_wrist");
        const auto& r = person.joints.at("r_wrist");
        const bool left_valid = person.confidence >= kConfidence && l[3] >= kConfidence;
        const bool right_valid = person.confidence >= kConfidence && r[3] >= kConfidence;
        const Vec3 left = position(l), right = position(r);
        const Vec3 middle = (left + right) * 0.5f;
        if (active_person == id) {
            if (!left_valid || !right_valid) {
                state.phase = "tracking_lost";
                state.releasing_since = {};
                continue; // Never release because a wrist disappeared.
            }
            state.valid = now;
            const auto target = std::find_if(targets.begin(), targets.end(),
                [](const auto& value) { return value.id == heldBalloon(); });
            if (target == targets.end()) {
                cancelActive();
                state.phase = "idle";
                continue;
            }
            state.latch_position = target->position;
            state.latch_radius = target->radius + 0.25f;
            const float release_span = std::max(state.span + 0.25f, state.span * 1.65f);
            if ((left - right).length() > release_span) {
                state.phase = "release_pending";
                if (gap || state.releasing_since == Clock::time_point{}) {
                    state.releasing_since = now;
                }
                if (age(now, state.releasing_since) >= kReleaseDelay) {
                    const auto released = *target;
                    if (command(scene, "pointer", {{"action", "up"}, {"owner", owner(id)}})) {
                        ++releases;
                    } else {
                        cancelActive();
                    }
                    active_person = -1;
                    latch(state, released);
                }
                continue; // Freeze the last carried position during release intent.
            }
            state.releasing_since = {};
            state.phase = "holding";
            const float alpha = static_cast<float>(1.0 - std::exp(-delta / 0.06));
            state.smooth = state.smooth + (middle - state.smooth) * alpha;
            if (command(scene, "pointer", {{"action", "move"}, {"owner", owner(id)},
                                            {"position", coordinates(state.smooth)}})) {
                ++moves;
            } else {
                cancelActive();
            }
            continue;
        }
        if (heldBalloon() >= 0) {
            state.candidate = -1;
            continue; // Mouse/another person owns the shared single hold.
        }
        if (state.latched) {
            if ((left_valid || right_valid) &&
                (!left_valid || (left - state.latch_position).length() > state.latch_radius) &&
                (!right_valid || (right - state.latch_position).length() > state.latch_radius)) {
                state.latched = false;
                state.phase = "idle";
            } else {
                continue;
            }
        }
        int candidate = -1;
        bool grip = false;
        float nearest = 1e9f;
        for (const auto& target : targets) {
            const float reach = target.radius + 0.10f;
            const Vec3 a = left - target.position, b = right - target.position;
            const float dl = a.length(), dr = b.length();
            const bool lc = left_valid && dl <= reach;
            const bool rc = right_valid && dr <= reach;
            const bool pair = lc && rc && (left - right).length() >= 0.15f &&
                Vec3::dot(a, b) < -0.2f * dl * dr;
            const bool single = lc != rc;
            if (!(pair || single)) {
                continue;
            }
            const float distance = pair ? (middle - target.position).length() : (lc ? dl : dr);
            if ((pair && !grip) || (pair == grip && distance < nearest)) {
                nearest = distance;
                candidate = target.id;
                grip = pair;
            }
        }
        // A quick touch may already have withdrawn or reached another object.
        // Preserve its edge unless the second hand encloses that SAME balloon.
        // Fresh wrist confidence is still required; missing data never fires it.
        if (candidate != state.candidate && !gap && state.candidate >= 0 && !state.grip &&
            state.phase == "touch_pending" &&
            (state.touch_left ? left_valid : right_valid)) {
            const auto touched = std::find_if(targets.begin(), targets.end(),
                [&](const auto& value) { return value.id == state.candidate; });
            if (touched != targets.end()) {
                if (age(now, state.pending_since) >= kTouchDelay &&
                    command(scene, "pop", {{"balloon", state.candidate}})) {
                    ++taps;
                    latch(state, *touched);
                }
                continue;
            }
        }
        if (candidate < 0) {
            state.candidate = -1;
            state.phase = "idle";
            continue;
        }
        if (gap || candidate != state.candidate || grip != state.grip) {
            state.pending_since = now;
            state.candidate = candidate;
            state.grip = grip;
            const auto selected = std::find_if(targets.begin(), targets.end(),
                [candidate](const auto& value) { return value.id == candidate; });
            state.touch_left = left_valid &&
                (left - selected->position).length() <= selected->radius + 0.10f;
        }
        state.phase = grip ? "grip_pending" : "touch_pending";
        if (age(now, state.pending_since) < (grip ? kGripDelay : kTouchDelay)) {
            continue;
        }
        const auto target = std::find_if(targets.begin(), targets.end(),
            [candidate](const auto& value) { return value.id == candidate; });
        if (grip) {
            if (command(scene, "pointer", {{"action", "down"}, {"balloon", candidate},
                    {"position", coordinates(middle)}, {"owner", owner(id)}}) &&
                balloonInteractionStatus().value("owner", std::string()) == owner(id)) {
                active_person = id;
                state.valid = now;
                state.smooth = middle;
                state.span = (left - right).length();
                state.latch_position = target->position;
                state.latch_radius = target->radius + 0.25f;
                state.phase = "holding";
                ++grabs;
            }
        } else if (command(scene, "pop", {{"balloon", candidate}})) {
            ++taps;
            latch(state, *target);
        }
    }
}
} // namespace Paint
