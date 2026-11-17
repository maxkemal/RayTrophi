#include "Game/GameRules.h"
#include <algorithm>
#include <cmath>
#include <set>

namespace Game {
namespace {
bool scalar(const nlohmann::json &value, double low, double high) {
    return value.is_number() && std::isfinite(value.get<double>()) && value.get<double>() >= low &&
           value.get<double>() <= high;
}
} // namespace

bool Rules::configure(const nlohmann::json &input, std::string &error) {
    if (!input.is_array() || input.size() > 128) {
        error = "rules_require_array_max_128";
        return false;
    }
    std::vector<Rule> parsed;
    std::set<std::string> ids;
    const std::set<std::string> events = {
        "gesture.stomp", "gesture.arms_up", "gesture.reach",      "contact", "zone.enter",
        "zone.exit",     "balloon.pop",     "coverage.threshold", "timer",   "session.state"};
    for (const auto &value : input) {
        if (!value.is_object() || !value.contains("id") || !value["id"].is_string() ||
            value["id"].get<std::string>().empty() || !value.contains("event") ||
            !value["event"].is_string() || !value.contains("action") ||
            !value["action"].is_object()) {
            error = "rule_requires_id_event_action";
            return false;
        }
        auto config = value;
        const auto id = config["id"].get<std::string>();
        if (id.size() > 128 || !events.count(config["event"].get<std::string>())) {
            error = "invalid_rule_id_or_event";
            return false;
        }
        const std::set<std::string> keys = {"id",        "event",       "action",
                                            "state",     "target",      "cooldown_seconds",
                                            "min_value", "min_persons", "group_window_seconds"};
        for (const auto &item : config.items()) {
            if (!keys.count(item.key())) {
                error = "unknown_rule_field: " + item.key();
                return false;
            }
        }
        if (!ids.insert(id).second) {
            error = "duplicate_rule_id";
            return false;
        }
        for (const auto &key : {"state", "target"}) {
            if (config.contains(key) && !config[key].is_string()) {
                error = std::string("invalid_rule_") + key;
                return false;
            }
        }
        const std::set<std::string> states = {"", "attract", "playing", "finale", "resetting"};
        if (!states.count(config.value("state", "")) || config.value("target", "").size() > 128) {
            error = "invalid_rule_state_or_target";
            return false;
        }
        for (const auto &key : {"cooldown_seconds", "min_value", "group_window_seconds"}) {
            const double fallback = std::string(key) == "group_window_seconds" ? 0.5 : 0.0;
            if (!config.contains(key)) {
                config[key] = fallback;
            }
            if (!scalar(config[key], 0.0, 3600.0)) {
                error = std::string("invalid_rule_") + key;
                return false;
            }
        }
        if (!config.contains("min_persons")) {
            config["min_persons"] = 1;
        }
        if (!config["min_persons"].is_number_integer() || !scalar(config["min_persons"], 1, 10)) {
            error = "invalid_rule_min_persons";
            return false;
        }
        auto &action = config["action"];
        if (!action.contains("type") || !action["type"].is_string()) {
            error = "action_requires_type";
            return false;
        }
        const auto type = action["type"].get<std::string>();
        std::set<std::string> action_keys = {"type"};
        if (type == "paint.dab") {
            action_keys.insert({"radius", "amount", "color", "position"});
        } else if (type == "balloon.burst") {
            action_keys.insert("count");
        } else if (type == "audio.play") {
            action_keys.insert("id");
        }
        for (const auto &item : action.items()) {
            if (!action_keys.count(item.key())) {
                error = "unknown_action_field: " + item.key();
                return false;
            }
        }
        if (type == "paint.dab") {
            if (!action.contains("radius")) {
                action["radius"] = 0.15;
            }
            if (!action.contains("amount")) {
                action["amount"] = 0.8;
            }
            if (!scalar(action["radius"], 0.001, 5) || !scalar(action["amount"], 0, 1)) {
                error = "invalid_paint_action_radius_or_amount";
                return false;
            }
            if (action.contains("color")) {
                if (!action["color"].is_array() || action["color"].size() != 3) {
                    error = "invalid_paint_action_color";
                    return false;
                }
                for (const auto &component : action["color"]) {
                    if (!scalar(component, 0, 1)) {
                        error = "invalid_paint_action_color";
                        return false;
                    }
                }
            }
            if (action.contains("position")) {
                if (!action["position"].is_array() || action["position"].size() != 3) {
                    error = "invalid_paint_action_position";
                    return false;
                }
                for (const auto &component : action["position"]) {
                    if (!scalar(component, -10000, 10000)) {
                        error = "invalid_paint_action_position";
                        return false;
                    }
                }
            }
        } else if (type == "balloon.burst") {
            if (!action.contains("count")) {
                action["count"] = 1;
            }
            if (!action["count"].is_number_integer() || !scalar(action["count"], 1, 3)) {
                error = "invalid_balloon_action_count";
                return false;
            }
        } else if (type == "audio.play") {
            if (!action.contains("id") || !action["id"].is_string() ||
                action["id"].get<std::string>().empty()) {
                error = "audio_action_requires_id";
                return false;
            }
        } else if (type != "paint.reset" && type != "room.reset") {
            error = "unsupported_rule_action: " + type;
            return false;
        }
        Rule rule;
        rule.config = std::move(config);
        parsed.push_back(std::move(rule));
    }
    rules_ = std::move(parsed);
    return true;
}

void Rules::evaluate(const Event &event, const std::string &state, double time,
                     std::vector<RuleAction> &output) {
    for (auto &rule : rules_) {
        const auto &c = rule.config;
        if (event.type != c["event"].get<std::string>() ||
            (!c.value("state", "").empty() && c.value("state", "") != state) ||
            (!c.value("target", "").empty() && c.value("target", "") != event.target) ||
            event.value < c["min_value"].get<double>()) {
            continue;
        }
        for (auto it = rule.people.begin(); it != rule.people.end();) {
            if (time - it->second > c["group_window_seconds"].get<double>()) {
                it = rule.people.erase(it);
            } else {
                ++it;
            }
        }
        if (event.person >= 0) {
            rule.people[{event.target, event.person}] = time;
        }
        const int needed = c["min_persons"].get<int>();
        const auto group_count =
            std::count_if(rule.people.begin(), rule.people.end(),
                          [&](const auto &entry) { return entry.first.first == event.target; });
        const auto cooldown_key = std::make_pair(event.target, needed > 1 ? -1 : event.person);
        for (auto it = rule.last_fire.begin(); it != rule.last_fire.end();) {
            if (time - it->second >= c["cooldown_seconds"].get<double>()) {
                it = rule.last_fire.erase(it);
            } else {
                ++it;
            }
        }
        if ((needed > 1 && group_count < needed) || rule.last_fire.count(cooldown_key)) {
            continue;
        }
        if (output.size() >= 256) {
            ++rule.dropped;
            continue;
        }
        output.push_back({c["action"], event});
        if (rule.last_fire.size() >= 1024) {
            rule.last_fire.erase(rule.last_fire.begin());
        }
        rule.last_fire[cooldown_key] = time;
        for (auto it = rule.people.begin(); it != rule.people.end();) {
            if (it->first.first == event.target) {
                it = rule.people.erase(it);
            } else {
                ++it;
            }
        }
        ++rule.fires;
    }
}

void Rules::resetHistory() {
    for (auto &rule : rules_) {
        rule.last_fire.clear();
        rule.people.clear();
    }
}

nlohmann::json Rules::snapshot() const {
    auto result = nlohmann::json::array();
    for (const auto &rule : rules_) {
        auto value = rule.config;
        value["fires"] = rule.fires;
        value["dropped"] = rule.dropped;
        result.push_back(std::move(value));
    }
    return result;
}
} // namespace Game
