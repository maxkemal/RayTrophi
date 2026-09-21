#include "Animation/RigIKChannels.h"
#include "Animation/RigSplineIK.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace RigAuthoring {
namespace {
bool time(double t) {
    return std::isfinite(t) && t >= 0 && t <= 1000000;
}
bool pose(const IKPose& p) {
    const auto& q = p.orientationWorld;
    const float n = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
    return validateSplineIKPose(p) && std::isfinite(n) && std::fabs(n - 1) < 1e-3f &&
           std::isfinite(p.target.length_squared()) && std::isfinite(p.pole.length_squared()) &&
           std::isfinite(p.blend) && p.blend >= 0 && p.blend <= 1;
}
nlohmann::json encode(const IKPose& p) {
    const auto& q = p.orientationWorld;
    nlohmann::json result = {{"enabled", p.enabled},
                             {"blend", p.blend},
                             {"orientation_enabled", p.orientationEnabled},
                             {"target_world", {p.target.x, p.target.y, p.target.z}},
                             {"pole_world", {p.pole.x, p.pole.y, p.pole.z}},
                             {"orientation_world", {q.w, q.x, q.y, q.z}}};
    if (!p.splineWorld.empty()) {
        result["spline_enabled"] = p.splineEnabled;
        result["spline_world"] = nlohmann::json::array();
        for (const auto& point : p.splineWorld) {
            result["spline_world"].push_back({point.x, point.y, point.z});
        }
    }
    return result;
}
IKPose decode(const nlohmann::json& j) {
    IKPose p;
    if (j.contains("spline_world") || j.contains("spline_enabled")) {
        p.splineEnabled = j.at("spline_enabled").get<bool>();
        const auto& points = j.at("spline_world");
        if (!points.is_array() || points.size() != 2) {
            throw std::runtime_error("spline");
        }
        for (const auto& point : points) {
            if (!point.is_array() || point.size() != 3) {
                throw std::runtime_error("spline");
            }
            p.splineWorld.emplace_back(point.at(0).get<float>(), point.at(1).get<float>(),
                                       point.at(2).get<float>());
        }
    }
    p.enabled = j.at("enabled").get<bool>();
    p.blend = j.at("blend").get<float>();
    p.orientationEnabled = j.at("orientation_enabled").get<bool>();
    auto vector = [](const nlohmann::json& v) {
        if (!v.is_array() || v.size() != 3)
            throw std::runtime_error("vector");
        return Vec3(v.at(0).get<float>(), v.at(1).get<float>(), v.at(2).get<float>());
    };
    p.target = vector(j.at("target_world"));
    p.pole = vector(j.at("pole_world"));
    const auto& q = j.at("orientation_world");
    if (!q.is_array() || q.size() != 4)
        throw std::runtime_error("quaternion");
    p.orientationWorld = Quaternion(q.at(0).get<float>(), q.at(1).get<float>(),
                                    q.at(2).get<float>(), q.at(3).get<float>());
    return p;
}
}
bool validateIKChannels(const IKChannels& channels, std::string& error) {
    error.clear();
    size_t count = 0;
    if (channels.size() > 256) {
        error = "rig_ik_channel_limit";
        return false;
    }
    for (const auto& entry : channels) {
        if (entry.first.empty() || entry.first.size() > 128) {
            error = "rig_ik_invalid_name";
            return false;
        }
        const auto& c = entry.second;
        count += c.keys.size() + c.contacts.size();
        if (count > 10000) {
            error = "rig_ik_channel_limit";
            return false;
        }
        double previous = -1;
        for (const auto& k : c.keys) {
            if (!time(k.seconds) || k.seconds <= previous || !pose(k.pose)) {
                error = "rig_ik_invalid_channel";
                return false;
            }
            previous = k.seconds;
        }
        previous = -1;
        for (const auto& k : c.contacts) {
            if (!time(k.start) || !time(k.end) || k.end <= k.start || k.start < previous ||
                !pose(k.pose) || !k.pose.enabled || k.pose.blend != 1) {
                error = "rig_ik_invalid_contact_interval";
                return false;
            }
            previous = k.end;
        }
    }
    return true;
}
bool validateIKChannelControls(const IKChannels& channels, const std::vector<IKControl>& controls,
                               std::string& error) {
    if (!validateIKChannels(channels, error)) {
        return false;
    }
    for (const auto& entry : channels) {
        const auto found = std::find_if(controls.begin(), controls.end(),
                                        [&](const auto& c) { return c.name == entry.first; });
        if (found == controls.end()) {
            error = "rig_ik_channel_unknown_control";
            return false;
        }
        if (found->solver == "aim") {
            if (!entry.second.contacts.empty()) {
                error = "rig_ik_contact_unsupported";
                return false;
            }
            for (const auto& key : entry.second.keys) {
                if (key.pose.orientationEnabled || key.pose.splineEnabled ||
                    !key.pose.splineWorld.empty()) {
                    error = "rig_ik_aim_incompatible_pose";
                    return false;
                }
            }
        }
        if (found->chain.empty()) {
            for (const auto& key : entry.second.keys) {
                if (key.pose.splineEnabled || !key.pose.splineWorld.empty()) {
                    error = "rig_ik_spline_requires_chain";
                    return false;
                }
            }
            for (const auto& contact : entry.second.contacts) {
                if (contact.pose.splineEnabled || !contact.pose.splineWorld.empty()) {
                    error = "rig_ik_spline_requires_chain";
                    return false;
                }
            }
        }
    }
    return true;
}
bool insertIKChannelKey(IKChannels& channels, const std::string& control, double seconds,
                        const IKPose& value, std::string& error) {
    auto staged = channels;
    auto& keys = staged[control].keys;
    auto key = std::lower_bound(keys.begin(), keys.end(), seconds,
                                [](const auto& k, double t) { return k.seconds < t; });
    auto p = value;
    p.contact = false;
    if (key != keys.end() && std::fabs(key->seconds - seconds) < 1e-8)
        key->pose = p;
    else
        keys.insert(key, {seconds, p});
    if (!validateIKChannels(staged, error))
        return false;
    channels = std::move(staged);
    return true;
}
bool removeIKChannelKey(IKChannels& channels, const std::string& control, double seconds,
                        std::string& error) {
    if (!time(seconds)) {
        error = "rig_pose_time_limit";
        return false;
    }
    auto staged = channels;
    const auto channel = staged.find(control);
    if (channel == staged.end()) {
        error = "rig_edit_no_change";
        return false;
    }
    auto& keys = channel->second.keys;
    const auto oldSize = keys.size();
    keys.erase(std::remove_if(keys.begin(), keys.end(), [&](const auto& key) {
                   return std::fabs(key.seconds - seconds) < 1e-8;
               }),
               keys.end());
    if (keys.size() == oldSize) {
        error = "rig_edit_no_change";
        return false;
    }
    if (keys.empty() && channel->second.contacts.empty()) {
        staged.erase(channel);
    }
    if (!validateIKChannels(staged, error)) {
        return false;
    }
    channels = std::move(staged);
    return true;
}
IKPoses sampleIKChannels(const IKChannels& channels, double seconds) {
    IKPoses result;
    for (const auto& entry : channels) {
        const auto& c = entry.second;
        auto upper = std::upper_bound(c.keys.begin(), c.keys.end(), seconds,
                                      [](double t, const auto& k) { return t < k.seconds; });
        if (upper != c.keys.begin()) {
            const auto& a = *(upper - 1);
            auto p = a.pose;
            p.contact = false;
            if (upper != c.keys.end()) {
                const auto& b = *upper;
                const float t = static_cast<float>((seconds - a.seconds) / (b.seconds - a.seconds));
                p.target = a.pose.target + (b.pose.target - a.pose.target) * t;
                p.pole = a.pose.pole + (b.pose.pole - a.pose.pole) * t;
                p.blend = a.pose.blend + (b.pose.blend - a.pose.blend) * t;
                p.enabled = p.blend > 0 && (a.pose.enabled || b.pose.enabled);
                p.orientationWorld =
                    Quaternion::slerp(a.pose.orientationWorld, b.pose.orientationWorld, t);
                p.orientationWorld.normalize();
                if (a.pose.splineWorld.size() == 2 && b.pose.splineWorld.size() == 2) {
                    for (size_t i = 0; i < 2; ++i)
                        p.splineWorld[i] = a.pose.splineWorld[i] +
                                           (b.pose.splineWorld[i] - a.pose.splineWorld[i]) * t;
                }
            }
            result[entry.first] = p;
        }
        for (const auto& k : c.contacts)
            if (seconds >= k.start && seconds < k.end) {
                auto p = k.pose;
                p.contact = true;
                result[entry.first] = p;
                break;
            }
    }
    return result;
}
nlohmann::json serializeIKChannels(const IKChannels& channels) {
    std::string error;
    if (!validateIKChannels(channels, error))
        throw std::runtime_error(error);
    auto rows = nlohmann::json::object();
    for (const auto& entry : channels) {
        auto keys = nlohmann::json::array(), contacts = nlohmann::json::array();
        for (const auto& k : entry.second.keys)
            keys.push_back({{"seconds", k.seconds}, {"pose", encode(k.pose)}});
        for (const auto& k : entry.second.contacts)
            contacts.push_back(
                {{"start_seconds", k.start}, {"end_seconds", k.end}, {"pose", encode(k.pose)}});
        rows[entry.first] = {{"keys", keys}, {"contacts", contacts}};
    }
    int version = 1;
    for (const auto& entry : channels) {
        for (const auto& k : entry.second.keys)
            if (!k.pose.splineWorld.empty())
                version = 2;
        for (const auto& k : entry.second.contacts)
            if (!k.pose.splineWorld.empty())
                version = 2;
    }
    return {{"version", version}, {"channels", rows}};
}
bool deserializeIKChannels(const nlohmann::json& j, IKChannels& output, std::string& error) {
    try {
        if (!j.is_object() ||
            (!j.at("version").is_number_integer() ||
             (j.at("version") != 1 && j.at("version") != 2)) ||
            !j.at("channels").is_object() || j.at("channels").size() > 256)
            throw std::runtime_error("format");
        IKChannels staged;
        size_t total = 0;
        for (const auto& entry : j.at("channels").items()) {
            const auto& keys = entry.value().at("keys");
            const auto& contacts = entry.value().at("contacts");
            if (!keys.is_array() || !contacts.is_array())
                throw std::runtime_error("format");
            total += keys.size() + contacts.size();
            if (total > 10000) {
                error = "rig_ik_channel_limit";
                return false;
            }
            auto& c = staged[entry.key()];
            for (const auto& k : keys)
                c.keys.push_back({k.at("seconds").get<double>(), decode(k.at("pose"))});
            for (const auto& k : contacts)
                c.contacts.push_back({k.at("start_seconds").get<double>(),
                                      k.at("end_seconds").get<double>(), decode(k.at("pose"))});
        }
        if (j.at("version") == 1) {
            for (const auto& entry : staged) {
                for (const auto& k : entry.second.keys)
                    if (!k.pose.splineWorld.empty())
                        throw std::runtime_error("version");
                for (const auto& k : entry.second.contacts)
                    if (!k.pose.splineWorld.empty())
                        throw std::runtime_error("version");
            }
        }
        if (!validateIKChannels(staged, error))
            return false;
        output = std::move(staged);
        return true;
    } catch (const std::exception&) {
        error = "rig_ik_invalid_channel";
        return false;
    }
}
}
