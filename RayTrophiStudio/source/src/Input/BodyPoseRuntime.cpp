#include "Input/BodyPoseRuntime.h"
#include "Paint/HiddenGameObjects.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <utility>
#include <vector>

namespace Input {
namespace {

constexpr std::size_t kMaxPersons = 10;

bool finiteNumber(const nlohmann::json &value) {
    return value.is_number() && std::isfinite(value.get<double>());
}

bool parse(const nlohmann::json &packet, std::map<int, BodyPerson> &people, double &time,
           std::string &error) {
    if (!packet.is_object() || !packet.contains("t") || !finiteNumber(packet["t"]) ||
        packet["t"].get<double>() < 0.0 || !packet.contains("persons") ||
        !packet["persons"].is_array() || packet["persons"].size() > kMaxPersons) {
        error = "invalid_body_packet: finite t >= 0 and persons[0..10] required";
        return false;
    }
    time = packet["t"].get<double>();
    for (const auto &value : packet["persons"]) {
        if (!value.is_object() || !value.contains("id") || !value["id"].is_number_integer() ||
            value["id"].get<double>() < 0 ||
            value["id"].get<double>() > std::numeric_limits<int>::max() ||
            !value.contains("confidence") || !finiteNumber(value["confidence"]) ||
            value["confidence"].get<double>() < 0 || value["confidence"].get<double>() > 1 ||
            !value.contains("joints") || !value["joints"].is_object() ||
            value["joints"].size() != kBodyJoints.size()) {
            error = "invalid_body_person: id, confidence[0,1], 15 joints required";
            return false;
        }
        BodyPerson person;
        person.id = value["id"].get<int>();
        person.confidence = value["confidence"].get<double>();
        if (people.count(person.id)) {
            error = "duplicate_body_person_id";
            return false;
        }
        for (const char *name : kBodyJoints) {
            const auto &joints = value["joints"];
            if (!joints.contains(name) || !joints[name].is_array() || joints[name].size() != 4) {
                error = std::string("invalid_body_joint: ") + name;
                return false;
            }
            std::array<double, 4> joint{};
            for (std::size_t i = 0; i < joint.size(); ++i) {
                if (!finiteNumber(joints[name][i])) {
                    error = std::string("nonfinite_body_joint: ") + name;
                    return false;
                }
                joint[i] = joints[name][i].get<double>();
            }
            if (joint[3] < 0 || joint[3] > 1 || std::abs(joint[0]) > 10000 ||
                std::abs(joint[1]) > 10000 || std::abs(joint[2]) > 10000) {
                error = "body_joint_out_of_range";
                return false;
            }
            person.joints.emplace(name, joint);
        }
        people.emplace(person.id, std::move(person));
    }
    return true;
}

} // namespace

BodyPoseRuntime &bodyPoses() {
    static BodyPoseRuntime runtime;
    return runtime;
}

bool BodyPoseRuntime::set(const nlohmann::json &packet, std::string &error,
                          Clock::time_point received) {
    std::map<int, BodyPerson> parsed;
    double sender_t = 0.0;
    if (!parse(packet, parsed, sender_t, error)) {
        return false;
    }
    if (sender_t <= last_sender_t_) {
        error = "body_timestamp_not_increasing: clear before restarting sender clock";
        return false;
    }
    expire(received);
    if (parsed.size() > kMaxPersons) {
        error = "body_capacity_exceeded: wait for lost identities to expire or clear";
        return false;
    }
    // Bound retained identities too: omitted people have a 0.5 s grace, but if new
    // identities arrive rapidly, evict the oldest omitted tracks instead of dropping frames.
    std::vector<std::pair<Clock::time_point, int>> omitted;
    for (const auto &entry : tracks_) {
        if (parsed.count(entry.first) == 0) {
            omitted.emplace_back(entry.second.received, entry.first);
        }
    }
    if (parsed.size() + omitted.size() > kMaxPersons) {
        std::sort(omitted.begin(), omitted.end());
        const std::size_t evict_count = parsed.size() + omitted.size() - kMaxPersons;
        for (std::size_t i = 0; i < evict_count; ++i) {
            tracks_.erase(omitted[i].second);
            ++expired_;
        }
    }
    for (auto &entry : parsed) {
        const auto found = tracks_.find(entry.first);
        Track current;
        Paint::constrainHiddenBodyPerson(entry.second);
        current.person = std::move(entry.second);
        current.received = received;
        current.sender_t = sender_t;
        if (found != tracks_.end()) {
            current.arms_up = found->second.arms_up;
            current.reach = found->second.reach;
            current.last_stomp = found->second.last_stomp;
        }
        gestures(current, found == tracks_.end() ? nullptr : &found->second);
        tracks_[entry.first] = std::move(current);
    }
    last_sender_t_ = sender_t;
    ++packets_;
    return true;
}

void BodyPoseRuntime::gestures(Track &current, const Track *previous) {
    const auto &joints = current.person.joints;
    if (current.person.confidence < 0.5) {
        current.arms_up = current.reach = false;
        return;
    }
    const auto emit = [&](const char *type, const std::array<double, 4> &joint, double value) {
        Game::Event event;
        event.type = type;
        event.person = current.person.id;
        event.time_s = seconds_;
        event.value = static_cast<float>(value);
        for (int axis = 0; axis < 3; ++axis) {
            event.position[axis] = static_cast<float>(joint[axis]);
        }
        Game::postEvent(std::move(event));
        ++gestures_;
    };
    const auto &head = joints.at("head");
    const auto &pelvis = joints.at("pelvis");
    const auto &left = joints.at("l_wrist");
    const auto &right = joints.at("r_wrist");
    const bool hands_valid = head[3] >= 0.5 && left[3] >= 0.5 && right[3] >= 0.5;
    const double arms_margin = current.arms_up ? 0.02 : 0.10;
    const bool arms =
        hands_valid && left[1] > head[1] + arms_margin && right[1] > head[1] + arms_margin;
    if (arms && !current.arms_up) {
        emit("gesture.arms_up", head, 1.0);
    }
    current.arms_up = arms;
    const double reach_margin = current.reach ? 0.30 : 0.45;
    const bool reaching =
        pelvis[3] >= 0.5 && ((left[3] >= 0.5 && pelvis[2] - left[2] > reach_margin) ||
                             (right[3] >= 0.5 && pelvis[2] - right[2] > reach_margin));
    if (reaching && !current.reach) {
        const auto &hand = left[2] < right[2] ? left : right;
        emit("gesture.reach", hand, pelvis[2] - hand[2]);
    }
    current.reach = reaching;
    if (!previous || previous->person.confidence < 0.5) {
        return;
    }
    const double dt = current.sender_t - previous->sender_t;
    if (dt < 0.001 || dt > 0.2) {
        return;
    }
    for (const char *foot : {"l_ankle", "r_ankle"}) {
        const auto &now = joints.at(foot);
        const auto &old = previous->person.joints.at(foot);
        const double speed = (old[1] - now[1]) / dt;
        if (now[3] >= 0.5 && old[3] >= 0.5 && now[1] <= 0.08 && speed >= 1.2 &&
            current.sender_t - current.last_stomp >= 0.3) {
            emit("gesture.stomp", now, speed);
            current.last_stomp = current.sender_t;
        }
    }
}

void BodyPoseRuntime::expire(Clock::time_point now) {
    for (auto it = tracks_.begin(); it != tracks_.end();) {
        if (std::chrono::duration<double>(now - it->second.received).count() > 0.5) {
            it = tracks_.erase(it);
            ++expired_;
        } else {
            ++it;
        }
    }
}

void BodyPoseRuntime::tick(double dt) {
    if (std::isfinite(dt) && dt > 0.0) {
        seconds_ += std::min(dt, 0.25);
    }
    expire();
}

void BodyPoseRuntime::clear() {
    Paint::clearHiddenPlayerHistory();
    tracks_.clear();
    last_sender_t_ = -1.0;
}

const std::map<int, BodyPerson> BodyPoseRuntime::persons() const {
    std::map<int, BodyPerson> result;
    for (const auto &entry : tracks_) {
        result.emplace(entry.first, entry.second.person);
    }
    return result;
}

const BodyPerson* BodyPoseRuntime::findPerson(int id) const {
    const auto found = tracks_.find(id);
    return found == tracks_.end() ? nullptr : &found->second.person;
}

BodyPoseRuntime::Clock::time_point BodyPoseRuntime::receivedAt(int id) const {
    const auto found = tracks_.find(id);
    return found == tracks_.end() ? Clock::time_point{} : found->second.received;
}

nlohmann::json BodyPoseRuntime::snapshot(Clock::time_point now) {
    expire(now);
    nlohmann::json people = nlohmann::json::array();
    for (const auto &entry : tracks_) {
        const auto &person = entry.second.person;
        nlohmann::json proxies = nlohmann::json::array();
        for (const char *joint : {"l_wrist", "r_wrist", "l_ankle", "r_ankle", "pelvis"}) {
            const auto &point = person.joints.at(joint);
            if (point[3] >= 0.5 && person.confidence >= 0.5) {
                proxies.push_back({{"joint", joint},
                                   {"position", {point[0], point[1], point[2]}},
                                   {"radius", std::string(joint) == "pelvis" ? 0.18 : 0.09}});
            }
        }
        people.push_back({{"id", person.id},
                          {"confidence", person.confidence},
                          {"joints", person.joints},
                          {"proxies", proxies}});
    }
    return {{"t", last_sender_t_},   {"persons", people},   {"packets", packets_},
            {"gestures", gestures_}, {"expired", expired_}, {"time_s", seconds_}};
}

} // namespace Input
