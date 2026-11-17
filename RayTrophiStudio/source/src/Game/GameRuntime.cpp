#include "Game/GameRuntime.h"
#include "Game/GameSession.h"
#include "Input/ExternalBodyRig.h"
#include "Paint/WetPaintGpu.h"
#include "Paint/BalloonPaintDemo.h"
#include "TriangleMesh.h"
#include "scene_data.h"

#include <algorithm>
#include <cmath>

namespace Game {
namespace {
double coverage_time = 0.0;
double last_coverage = 0.0;
double contact_time = 0.0;
std::vector<std::string> contact_targets;

bool execute(SceneData &scene, const RuleAction &command, std::string &error) {
    const auto &action = command.action;
    const auto type = action["type"].get<std::string>();
    auto &paint = Paint::wetPaintGpu();
    if (type == "paint.dab") {
        if (paint.empty()) {
            error = "paint_action_requires_wet_surfaces";
            return false;
        }
        Paint::WetPaintDab dab;
        static constexpr float colors[6][3] = {{1, 0.15f, 0.1f}, {0.1f, 0.35f, 1},
                                               {1, 0.8f, 0.1f},  {0.1f, 0.9f, 0.3f},
                                               {0.8f, 0.1f, 1},  {0.1f, 0.9f, 0.9f}};
        const int color = std::max(0, command.event.person) % 6;
        for (int axis = 0; axis < 3; ++axis) {
            dab.position[axis] = action.contains("position") ? action["position"][axis].get<float>()
                                                             : command.event.position[axis];
            dab.color[axis] =
                action.contains("color") ? action["color"][axis].get<float>() : colors[color][axis];
        }
        dab.radius = action["radius"].get<float>();
        dab.amount = action["amount"].get<float>();
        paint.queueDab(dab);
        return true;
    }
    if (type == "paint.reset") {
        last_coverage = 0.0;
        return paint.empty() || paint.resetPaint();
    }
    if (type == "room.reset") {
        // Paint cleared AND every popped balloon back (the balloon demo's reset;
        // without a prepared demo it only clears the paint).
        last_coverage = 0.0;
        nlohmann::json output;
        return Paint::balloonPaintCommand(scene, "reset", nlohmann::json::object(), output, error);
    }
    if (type == "balloon.burst") {
        nlohmann::json output;
        if (!Paint::balloonPaintCommand(scene, "status", nlohmann::json::object(), output, error)) {
            return false;
        }
        const auto serial_before = output.value("events_total", std::uint64_t{0});
        if (!Paint::balloonPaintCommand(scene, "burst", {{"count", action["count"]}}, output,
                                        error)) {
            return false;
        }
        if (output.value("events_total", std::uint64_t{0}) == serial_before) {
            error = "no_balloon_available";
            return false;
        }
        Event event = command.event;
        event.type = "balloon.pop";
        event.value = output["events"].back()["simultaneous"].get<float>();
        postEvent(event);
        return true;
    }
    if (type == "audio.play") {
        try {
            if (!scene.audio) {
                error = "audio_service_unavailable";
                return false;
            }
            const auto result = scene.audio->invoke("play", {{"id", action["id"]}});
            if (result.contains("__error") || result.contains("error")) {
                error = result.dump();
                return false;
            }
            return true;
        } catch (const std::exception &exception) {
            error = exception.what();
            return false;
        }
    }
    error = "unsupported_rule_action";
    return false;
}

void sampleContacts(SceneData &scene, const std::map<int, Input::BodyPerson> &people) {
    // Analytic joint spheres against named flat-mesh world AABBs. No Triangle facades.
    for (const auto &object : scene.world.objects) {
        const auto mesh = std::dynamic_pointer_cast<TriangleMesh>(object);
        if (!mesh || std::find(contact_targets.begin(), contact_targets.end(), mesh->nodeName) ==
                         contact_targets.end()) {
            continue;
        }
        AABB box;
        if (!mesh->bounding_box(0, 0, box)) {
            continue;
        }
        for (const auto &entry : people) {
            const auto &person = entry.second;
            if (person.confidence < 0.5) {
                continue;
            }
            for (const char *name : {"l_wrist", "r_wrist", "l_ankle", "r_ankle", "pelvis"}) {
                const auto &point = person.joints.at(name);
                if (point[3] < 0.5) {
                    continue;
                }
                const double radius = std::string(name) == "pelvis" ? 0.18 : 0.09;
                double squared = 0.0;
                for (int axis = 0; axis < 3; ++axis) {
                    const double closest =
                        std::clamp(point[axis], static_cast<double>(box.min[axis]),
                                   static_cast<double>(box.max[axis]));
                    squared += (point[axis] - closest) * (point[axis] - closest);
                }
                if (squared <= radius * radius) {
                    Event event;
                    event.type = "contact";
                    event.target = mesh->nodeName;
                    event.person = person.id;
                    event.time_s = Input::bodyPoses().seconds();
                    event.value = 1.0f;
                    for (int axis = 0; axis < 3; ++axis) {
                        event.position[axis] = static_cast<float>(point[axis]);
                    }
                    postEvent(event);
                    break; // one event per person/mesh, never one per joint
                }
            }
        }
    }
}
} // namespace

void resetRuntime() {
    Input::resetExternalBodyRigs();
    session().clear();
    Input::bodyPoses().clear();
    coverage_time = last_coverage = contact_time = 0.0;
    contact_targets.clear();
    std::vector<Event> discarded;
    drainEvents(discarded);
}

bool runtimeCommand(SceneData &scene, const std::string &operation, const nlohmann::json &config,
                    nlohmann::json &output, std::string &error) {
    std::vector<std::string> targets;
    if (operation == "configure" && config.contains("contact_targets") &&
        config["contact_targets"].is_array()) {
        for (const auto &value : config["contact_targets"]) {
            if (!value.is_string()) {
                error = "contact_target_requires_name";
                return false;
            }
            const auto name = value.get<std::string>();
            bool found = false;
            for (const auto &object : scene.world.objects) {
                const auto mesh = std::dynamic_pointer_cast<TriangleMesh>(object);
                found = found || (mesh && mesh->nodeName == name);
            }
            if (!found) {
                error = "unknown_flat_contact_target: " + name;
                return false;
            }
            targets.push_back(name);
        }
    }
    if (!session().command(operation, config, output, error)) {
        return false;
    }
    if (operation == "configure") {
        contact_targets = std::move(targets);
        coverage_time = last_coverage = contact_time = 0.0;
        std::vector<Event> discarded;
        drainEvents(discarded);
    }
    return true;
}

void tickRuntime(SceneData &scene, double dt) {
    Input::bodyPoses().tick(dt);
    Input::tickExternalBodyRigs(scene);
    if (!session().configured()) {
        return;
    }
    if (!std::isfinite(dt) || dt < 0) {
        return;
    }
    coverage_time += std::min(dt, 0.25);
    contact_time += std::min(dt, 0.25);
    if (coverage_time >= 0.5 && !Paint::wetPaintGpu().empty()) {
        coverage_time = 0.0;
        Paint::WetPaintCoverage coverage;
        if (Paint::wetPaintGpu().coverage(scene, coverage)) {
            last_coverage = coverage.ratio;
        }
    }
    const auto people = Input::bodyPoses().persons();
    if (contact_time >= 0.1) {
        contact_time = 0.0;
        sampleContacts(scene, people);
    }
    std::vector<RuleAction> actions;
    session().tick(dt, people, last_coverage, actions);
    for (const auto &action : actions) {
        std::string error;
        const bool ok = execute(scene, action, error);
        session().actionResult(ok, error.empty() && !ok ? "game_action_failed" : error);
    }
}

bool postExternalEvent(const nlohmann::json &value, std::string &error) {
    try {
        static const std::set<std::string> types = {
            "gesture.stomp", "gesture.arms_up", "gesture.reach",      "contact", "zone.enter",
            "zone.exit",     "balloon.pop",     "coverage.threshold", "timer",   "session.state"};
        if (!value.is_object() || !value.contains("type") || !value["type"].is_string() ||
            !types.count(value["type"].get<std::string>())) {
            error = "invalid_game_event_type";
            return false;
        }
        Event event;
        event.type = value["type"].get<std::string>();
        event.target = value.value("target", "");
        if (event.target.size() > 128 ||
            (value.contains("person") &&
             (!value["person"].is_number_integer() || value["person"].get<double>() < -1 ||
              value["person"].get<double>() > 2147483647.0))) {
            error = "invalid_game_event_target_or_person";
            return false;
        }
        event.person = value.value("person", -1);
        event.time_s = Input::bodyPoses().seconds();
        if (value.contains("value")) {
            if (!value["value"].is_number() || !std::isfinite(value["value"].get<double>()) ||
                std::abs(value["value"].get<double>()) > 1e9) {
                error = "invalid_game_event_value";
                return false;
            }
            event.value = value["value"].get<float>();
        }
        for (const char *key : {"position", "normal"}) {
            if (!value.contains(key)) {
                continue;
            }
            if (!value[key].is_array() || value[key].size() != 3) {
                error = "invalid_game_event_vec3";
                return false;
            }
            for (int axis = 0; axis < 3; ++axis) {
                if (!value[key][axis].is_number() ||
                    !std::isfinite(value[key][axis].get<double>()) ||
                    std::abs(value[key][axis].get<double>()) > 10000) {
                    error = "invalid_game_event_vec3";
                    return false;
                }
                (std::string(key) == "position" ? event.position : event.normal)[axis] =
                    value[key][axis].get<float>();
            }
        }
        postEvent(event);
        return true;
    } catch (const std::exception &) {
        error = "invalid_game_event";
        return false;
    }
}
} // namespace Game
