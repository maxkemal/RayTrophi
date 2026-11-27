#include "Paint/BalloonBodyInteraction.h"
#include "Paint/BalloonPaintDemo.h"
#include "Input/BodyPoseRuntime.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <vector>

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
constexpr double kMinSweepDt = 0.005;
constexpr double kMaxSweepDt = 0.20;
constexpr float kSwingSpeed = 1.2f;

struct PersonState {
    Clock::time_point sample{};
    Clock::time_point valid{};
    Clock::time_point pending_since{};
    Clock::time_point releasing_since{};
    Clock::time_point prev_time{};
    int candidate = -1;
    bool grip = false;
    bool touch_left = false;
    bool latched = false;
    bool has_prev_l = false;
    bool has_prev_r = false;
    Vec3 prev_l_wrist;
    Vec3 prev_r_wrist;
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

bool segmentIntersectsSphere(const Vec3& a, const Vec3& b, const Vec3& center, float radius) {
    if (radius < 0.0f) {
        return false;
    }
    const Vec3 ab = b - a;
    const float len_sq = ab.length_squared();
    float t = 0.0f;
    if (len_sq > 1e-10f) {
        t = std::clamp(Vec3::dot(center - a, ab) / len_sq, 0.0f, 1.0f);
    }
    const Vec3 closest = a + ab * t;
    return (closest - center).length_squared() <= radius * radius;
}

void latch(PersonState& state, const BalloonInteractionTarget& target) {
    state.latched = true;
    state.latch_position = target.position;
    state.latch_radius = target.radius + 0.25f;
    state.candidate = -1;
    state.grip = false;
    state.pending_since = {};
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
    std::vector<int> popped_this_tick;
    const auto already_popped = [&](int balloon_id) {
        return std::find(popped_this_tick.begin(), popped_this_tick.end(), balloon_id) !=
            popped_this_tick.end();
    };
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
        const Vec3 prev_l = state.prev_l_wrist;
        const Vec3 prev_r = state.prev_r_wrist;
        const bool had_prev_l = state.has_prev_l;
        const bool had_prev_r = state.has_prev_r;
        const double sweep_dt = state.prev_time == Clock::time_point{}
            ? 0.0
            : age(received, state.prev_time);
        state.prev_l_wrist = left;
        state.prev_r_wrist = right;
        state.has_prev_l = left_valid;
        state.has_prev_r = right_valid;
        state.prev_time = (left_valid || right_valid) ? received : Clock::time_point{};
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
            const float release_span = std::min(
                std::max(state.span + 0.25f, state.span * 1.45f), state.span + 0.55f);
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
        bool unlatched_this_tick = false;
        if (state.latched) {
            if ((left_valid || right_valid) &&
                (!left_valid || (left - state.latch_position).length() > state.latch_radius) &&
                (!right_valid || (right - state.latch_position).length() > state.latch_radius)) {
                state.latched = false;
                state.phase = "idle";
                unlatched_this_tick = true;
            } else {
                continue;
            }
        }
        const int held_id = heldBalloon();
        if (sweep_dt >= kMinSweepDt && sweep_dt <= kMaxSweepDt) {
            const auto find_swing_target = [&](const Vec3& prev_pos, const Vec3& curr_pos)
                -> const BalloonInteractionTarget* {
                const float speed =
                    (curr_pos - prev_pos).length() / static_cast<float>(sweep_dt);
                if (speed < kSwingSpeed) {
                    return nullptr;
                }
                const BalloonInteractionTarget* hit = nullptr;
                float best_dist_sq = 1e9f;
                for (const auto& target : targets) {
                    if (target.id == held_id || already_popped(target.id)) {
                        continue;
                    }
                    if (unlatched_this_tick &&
                        (target.position - state.latch_position).length_squared() < 1e-4f) {
                        continue;
                    }
                    const float reach = target.radius + 0.10f;
                    if (segmentIntersectsSphere(prev_pos, curr_pos, target.position, reach)) {
                        const float dist_sq = (target.position - prev_pos).length_squared();
                        if (dist_sq < best_dist_sq) {
                            best_dist_sq = dist_sq;
                            hit = &target;
                        }
                    }
                }
                return hit;
            };
            const BalloonInteractionTarget* swing_target = nullptr;
            if (had_prev_l && left_valid) {
                swing_target = find_swing_target(prev_l, left);
            }
            if (!swing_target && had_prev_r && right_valid) {
                swing_target = find_swing_target(prev_r, right);
            }
            if (swing_target && command(scene, "pop", {{"balloon", swing_target->id}})) {
                popped_this_tick.push_back(swing_target->id);
                ++taps;
                latch(state, *swing_target);
                continue;
            }
        }
        int candidate = -1;
        bool grip = false;
        float nearest = 1e9f;
        for (const auto& target : targets) {
            if (target.id == held_id || already_popped(target.id)) {
                continue;
            }
            const float reach = target.radius + 0.10f;
            const Vec3 a = left - target.position, b = right - target.position;
            const float dl = a.length(), dr = b.length();
            const bool lc = left_valid && dl <= reach;
            const bool rc = right_valid && dr <= reach;
            const bool pair = held_id < 0 && lc && rc && (left - right).length() >= 0.15f &&
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
        if (candidate < 0) {
            state.candidate = -1;
            state.grip = false;
            state.phase = "idle";
            state.pending_since = {};
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
            popped_this_tick.push_back(candidate);
            ++taps;
            latch(state, *target);
        }
    }
}
} // namespace Paint
