#include "Game/GameSession.h"
#include <algorithm>
#include <cmath>

namespace Game {
namespace {
bool bounded(const nlohmann::json &value, double low, double high) {
    return value.is_number() && std::isfinite(value.get<double>()) && value.get<double>() >= low &&
           value.get<double>() <= high;
}
} // namespace

Session &session() {
    static Session value;
    return value;
}

void Session::clear() {
    *this = Session{};
}

void Session::transition(const std::string &state) {
    state_ = state;
    state_time_ = 0.0;
    Event event;
    event.type = "session.state";
    event.target = state;
    event.time_s = time_;
    local_events_.push_back(std::move(event));
}

bool Session::command(const std::string &operation, const nlohmann::json &input,
                      nlohmann::json &output, std::string &error) {
    if (operation == "configure") {
        if (!input.is_object()) {
            error = "session_config_requires_object";
            return false;
        }
        auto candidate = input;
        const std::set<std::string> keys = {
            "duration_seconds", "finale_seconds", "reset_seconds", "coverage_goal",
            "auto_start",       "rules",          "zones",         "contact_targets"};
        for (const auto &item : candidate.items()) {
            if (!keys.count(item.key())) {
                error = "unknown_session_config_field: " + item.key();
                return false;
            }
        }
        const std::map<std::string, double> defaults = {{"duration_seconds", 180},
                                                        {"finale_seconds", 8},
                                                        {"reset_seconds", 3},
                                                        {"coverage_goal", 0.65}};
        for (const auto &entry : defaults) {
            if (!candidate.contains(entry.first)) {
                candidate[entry.first] = entry.second;
            }
            const bool ratio = entry.first == "coverage_goal";
            if (!bounded(candidate[entry.first], ratio ? 0.001 : 0.25, ratio ? 1 : 3600)) {
                error = "invalid_session_" + entry.first;
                return false;
            }
        }
        if (!candidate.contains("auto_start")) {
            candidate["auto_start"] = false;
        }
        if (!candidate["auto_start"].is_boolean()) {
            error = "session_auto_start_requires_boolean";
            return false;
        }
        if (!candidate.contains("rules")) {
            candidate["rules"] = nlohmann::json::array();
        }
        if (!candidate.contains("zones")) {
            candidate["zones"] = nlohmann::json::array();
        }
        if (!candidate.contains("contact_targets")) {
            candidate["contact_targets"] = nlohmann::json::array();
        }
        if (!candidate["contact_targets"].is_array() || candidate["contact_targets"].size() > 64) {
            error = "contact_targets_require_array_max_64";
            return false;
        }
        for (const auto &target : candidate["contact_targets"]) {
            if (!target.is_string() || target.get<std::string>().empty()) {
                error = "contact_target_requires_name";
                return false;
            }
        }
        Rules new_rules;
        if (!new_rules.configure(candidate["rules"], error)) {
            return false;
        }
        if (!candidate["zones"].is_array() || candidate["zones"].size() > 32) {
            error = "zones_require_array_max_32";
            return false;
        }
        std::set<std::string> names;
        for (const auto &zone : candidate["zones"]) {
            if (!zone.is_object() || !zone.contains("name") || !zone["name"].is_string() ||
                zone["name"].get<std::string>().empty() ||
                !names.insert(zone["name"].get<std::string>()).second) {
                error = "zone_requires_unique_name";
                return false;
            }
            for (const char *key : {"min", "max"}) {
                if (!zone.contains(key) || !zone[key].is_array() || zone[key].size() != 3) {
                    error = "zone_requires_min_max_vec3";
                    return false;
                }
                for (const auto &component : zone[key]) {
                    if (!bounded(component, -10000, 10000)) {
                        error = "invalid_zone_coordinate";
                        return false;
                    }
                }
            }
            for (int axis = 0; axis < 3; ++axis) {
                if (zone["min"][axis].get<double>() >= zone["max"][axis].get<double>()) {
                    error = "zone_min_must_be_less_than_max";
                    return false;
                }
            }
        }
        clear();
        config_ = std::move(candidate);
        rules_ = std::move(new_rules);
        configured_ = true;
    } else if (operation == "start") {
        if (!configured_ || state_ != "attract") {
            error = "session_start_requires_configured_attract";
            return false;
        }
        last_error_.clear();
        rules_.resetHistory();
        transition("playing");
    } else if (operation == "stop") {
        if (!configured_) {
            error = "session_not_configured";
            return false;
        }
        transition("resetting");
    } else if (operation == "reset") {
        if (!configured_) {
            error = "session_not_configured";
            return false;
        }
        transition("resetting");
    } else if (operation != "status") {
        error = "unknown_session_operation";
        return false;
    }
    output = snapshot();
    return true;
}

void Session::tick(double dt, const std::map<int, Input::BodyPerson> &persons, double coverage,
                   std::vector<RuleAction> &actions) {
    if (!configured_ || !std::isfinite(dt) || dt < 0) {
        return;
    }
    dt = std::min(dt, 0.25);
    person_count_ = persons.size();
    time_ += dt;
    state_time_ += dt;
    const double old_coverage = coverage_;
    if (std::isfinite(coverage)) {
        coverage_ = std::clamp(coverage, 0.0, 1.0);
    }
    if (state_ == "attract" && config_["auto_start"].get<bool>() && !persons.empty()) {
        rules_.resetHistory();
        transition("playing");
    }
    const double goal = config_["coverage_goal"].get<double>();
    if (state_ == "playing" && state_time_ >= config_["duration_seconds"].get<double>()) {
        Event timer;
        timer.type = "timer";
        timer.target = "playing";
        timer.time_s = time_;
        local_events_.push_back(timer);
    }
    if (state_ == "playing" && old_coverage < goal && coverage_ >= goal) {
        Event event;
        event.type = "coverage.threshold";
        event.value = static_cast<float>(coverage_);
        event.time_s = time_;
        local_events_.push_back(event);
    }
    // Evaluate the events in the state that generated them, then advance the
    // state machine below. The goal-crossing paint action still belongs to play.
    for (const auto &zone : config_["zones"]) {
        const auto name = zone["name"].get<std::string>();
        std::set<int> inside;
        for (const auto &person : persons) {
            const auto &pelvis = person.second.joints.at("pelvis");
            bool hit = person.second.confidence >= 0.5 && pelvis[3] >= 0.5;
            for (int axis = 0; axis < 3; ++axis) {
                hit = hit && pelvis[axis] >= zone["min"][axis].get<double>() &&
                      pelvis[axis] <= zone["max"][axis].get<double>();
            }
            if (hit) {
                inside.insert(person.first);
            }
        }
        const auto post = [&](const char *type, int id) {
            Event event;
            event.type = type;
            event.target = name;
            event.person = id;
            event.time_s = time_;
            if (persons.count(id)) {
                const auto &point = persons.at(id).joints.at("pelvis");
                for (int axis = 0; axis < 3; ++axis) {
                    event.position[axis] = static_cast<float>(point[axis]);
                }
            }
            local_events_.push_back(event);
        };
        for (int id : inside) {
            if (!zones_[name].count(id)) {
                post("zone.enter", id);
            }
        }
        for (int id : zones_[name]) {
            if (!inside.count(id)) {
                post("zone.exit", id);
            }
        }
        zones_[name] = std::move(inside);
    }
    std::vector<Event> events;
    drainEvents(events);
    events.insert(events.end(), local_events_.begin(), local_events_.end());
    local_events_.clear();
    for (const auto &event : events) {
        ++events_[event.type];
        if (event.type == "session.state" && event.target == "resetting") {
            // End of a session: the next child gets a clean room, balloons included.
            actions.push_back({{{"type", "room.reset"}}, event});
        }
        rules_.evaluate(event, state_, time_, actions);
    }
    if (state_ == "playing" &&
        (state_time_ >= config_["duration_seconds"].get<double>() || coverage_ >= goal)) {
        transition("finale");
    } else if (state_ == "finale" && state_time_ >= config_["finale_seconds"].get<double>()) {
        transition("resetting");
    } else if (state_ == "resetting" && state_time_ >= config_["reset_seconds"].get<double>()) {
        ++sessions_;
        transition("attract");
    }
}

void Session::actionResult(bool ok, const std::string &error) {
    if (ok) {
        ++action_ok_;
    } else {
        ++action_failed_;
        last_error_ = error;
    }
}

nlohmann::json Session::snapshot() const {
    return {{"configured", configured_},
            {"state", state_},
            {"time_s", time_},
            {"state_time_s", state_time_},
            {"coverage", coverage_},
            {"completed_sessions", sessions_},
            {"actions_ok", action_ok_},
            {"person_count", person_count_},
            {"actions_failed", action_failed_},
            {"last_error", last_error_},
            {"events", events_},
            {"rules", rules_.snapshot()},
            {"config", config_},
            {"posted_events", postedEvents()},
            {"dropped_events", droppedEvents()}};
}
} // namespace Game
