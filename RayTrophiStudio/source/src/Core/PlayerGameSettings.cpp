#include "Core/PlayerGameSettings.h"
#include "Core/RuntimeProjectHost.h"
#include "Api/RtApiPaintWet.h"
#include "UIContext.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <cctype>
#include <cstddef>
#include <utility>
#include <stdexcept>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace AppRuntime {
namespace {
using json = nlohmann::json;
namespace fs = std::filesystem;
constexpr const char *demo_fields[] = {
    "autostart",    "balloons",           "respawn_seconds",      "birth_seconds", "interval_min",
    "interval_max", "idle_reset_seconds", "finale_reset_seconds", "room_goal"};
struct Session {
    std::string project;
    std::uint64_t revision = std::numeric_limits<std::uint64_t>::max();
    bool ready = false;
    bool attempted_profile = false;
    json authored_demo;
    std::string error;
};
Session session;
json profiles = {{"version", 1}, {"games", json::object()}};
bool loaded = false;
std::string storage_error;
fs::path profile_path;
void require(bool condition, const std::string &message) {
    if (!condition) {
        throw std::invalid_argument(message);
    }
}
void check(const rtapi::Result &result) {
    require(result.ok, result.error);
}
json demo() {
    std::string text;
    check(rtapi::wetPaintDemoCommand("status", "{}", text));
    return json::parse(text);
}
std::string projectKey(const fs::path &path) {
    auto value = projectPathToUtf8(fs::weakly_canonical(path));
#ifdef _WIN32
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
#endif
    return value;
}
void loadProfiles() {
    if (loaded) {
        return;
    }
    loaded = true;
    profile_path = fs::absolute("player_game_settings.json");
    std::ifstream stream(profile_path, std::ios::binary);
    if (!stream) {
        return;
    }
    auto value = json::parse(stream, nullptr, false);
    if (value.is_object() && value.contains("version") && value["version"] == 1 &&
        value.contains("games") && value["games"].is_object()) {
        profiles = std::move(value);
    } else {
        storage_error = "Saved game settings are invalid; authored settings kept";
        session.error = storage_error;
    }
}
void writeProfiles(const json &value) {
    auto temporary = profile_path;
    temporary += ".tmp";
    {
        std::ofstream stream(temporary, std::ios::binary);
        stream << value.dump(2);
        require(static_cast<bool>(stream), "Could not write game settings");
    }
#ifdef _WIN32
    require(MoveFileExW(temporary.c_str(), profile_path.c_str(),
                        MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0,
            "Could not publish game settings");
#else
    fs::rename(temporary, profile_path);
#endif
    profiles = value;
    storage_error.clear();
}
int capacity(const json &config) {
    if (config.is_null()) {
        return 0;
    }
    const double particles = config.value("particles_per_balloon", 1800.0);
    const double size = config.value("size_max", 1.0);
    int maximum = std::min(24, static_cast<int>(131072.0 / std::max(1.0, particles * size * size)));
    if (config.contains("visual_nodes")) {
        maximum = std::min(maximum, static_cast<int>(config["visual_nodes"].size()));
    }
    return maximum;
}
bool ensureSession(UIContext &context) {
    loadProfiles();
    const auto host = runtimeStatus(context);
    const auto revision = host.value("scene_revision", std::uint64_t{0});
    const auto path = host.value("project_path", std::string());
    if (!host.value("scene_ready", false) || path.empty()) {
        session.ready = false;
        return false;
    }
    const auto key = projectKey(projectPathFromUtf8(path));
    if (key != session.project || revision != session.revision) {
        session = Session{};
        session.project = key;
        session.revision = revision;
        session.error = storage_error;
    }
    if (session.ready) {
        return true;
    }
    rtapi::GameDesignInfo design;
    check(rtapi::getGameDesign(design));
    if (design.pending) {
        return false;
    }
    const auto state = demo();
    rtapi::WetPaintStatusInfo paint;
    check(rtapi::getWetPaintStatus(paint));
    if (!state.value("prepared", false) && paint.surface_list.empty()) {
        return false;
    }
    if (state.value("prepared", false)) {
        session.authored_demo = state["config"];
    }
    session.ready = true;
    return true;
}
json values() {
    json result = json::object();
    const auto state = demo();
    if (state.value("prepared", false)) {
        const auto &config = state["config"];
        for (const auto *key : demo_fields) {
            if (config.contains(key)) {
                result[key] = config[key];
            }
        }
        result["respawn_seconds"] = config.value("respawn_seconds", 3.5);
        result["birth_seconds"] = config.value("birth_seconds", 0.75);
    }
    rtapi::WetPaintStatusInfo paint;
    check(rtapi::getWetPaintStatus(paint));
    bool first = true;
    for (const auto &surface : paint.surface_list) {
        const json row = {{"lifetime_seconds", surface.settings.lifetime_seconds},
                          {"diffusion", surface.settings.diffusion},
                          {"runoff", surface.settings.runoff},
                          {"resolution", surface.resolution}};
        for (const auto &field : row.items()) {
            if (first) {
                result[field.key()] = field.value();
            } else if (result[field.key()] != field.value()) {
                result[field.key()] = nullptr;
            }
        }
        first = false;
    }
    return result;
}
void scalar(const json &patch, const char *key, double minimum, double maximum) {
    if (!patch.contains(key)) {
        return;
    }
    const auto &value = patch[key];
    require(value.is_number() && std::isfinite(value.get<double>()) &&
                value.get<double>() >= minimum && value.get<double>() <= maximum,
            std::string(key) + " must be finite in " + std::to_string(minimum) + ".." +
                std::to_string(maximum));
}
void validate(const json &patch, const json &current) {
    require(patch.is_object(), "Game settings must be an object");
    for (const auto &field : patch.items()) {
        require(current.contains(field.key()), "Unsupported game setting: " + field.key());
    }
    if (patch.contains("autostart")) {
        require(patch["autostart"].is_boolean(), "autostart must be boolean");
    }
    if (patch.contains("balloons")) {
        require(patch["balloons"].is_number_integer() && patch["balloons"].get<double>() >= 3 &&
                    patch["balloons"].get<double>() <= capacity(session.authored_demo),
                "balloons exceeds the prepared scene capacity or particle budget");
    }
    if (patch.contains("resolution")) {
        const auto &size = patch["resolution"];
        require(size.is_number_integer() && (size == 512 || size == 1024 || size == 2048),
                "Personnel texture resolution must be 512, 1024 or 2048");
    }
    scalar(patch, "respawn_seconds", 0.25, 60);
    scalar(patch, "birth_seconds", 0, 3);
    scalar(patch, "interval_min", 2, 30);
    scalar(patch, "interval_max", 2, 30);
    scalar(patch, "lifetime_seconds", 0.05, 600);
    scalar(patch, "diffusion", 0, 2);
    scalar(patch, "runoff", 0, 2);
    scalar(patch, "idle_reset_seconds", 0, 3600);
    scalar(patch, "finale_reset_seconds", 0, 120);
    scalar(patch, "room_goal", 0, 1);
    auto merged = current;
    merged.update(patch);
    for (const auto &item : {std::pair<const char *, double>{"idle_reset_seconds", 10},
                             {"finale_reset_seconds", 3},
                             {"room_goal", 0.05}}) {
        if (merged.contains(item.first)) {
            const double value = merged[item.first];
            require(value == 0 || value >= item.second, std::string(item.first) +
                                                            ": use 0 to disable, or at least " +
                                                            std::to_string(item.second));
        }
    }
    if (merged.contains("interval_min")) {
        require(merged["interval_min"].get<double>() <= merged["interval_max"].get<double>(),
                "Minimum pop interval must not exceed maximum");
    }
}
json apply(const json &patch) {
    const auto current = values();
    validate(patch, current);
    auto state = demo();
    auto config = state.value("config", json::object());
    bool rebuild_demo = false;
    for (const auto *key : demo_fields) {
        if (patch.contains(key) && patch[key] != current[key]) {
            config[key] = patch[key];
            rebuild_demo = true;
        }
    }
    if (rebuild_demo && config.contains("visual_nodes")) {
        auto nodes = session.authored_demo["visual_nodes"];
        const auto count = config["balloons"].get<size_t>();
        nodes.erase(nodes.begin() + static_cast<std::ptrdiff_t>(count), nodes.end());
        config["visual_nodes"] = std::move(nodes);
    }
    rtapi::WetPaintStatusInfo paint;
    check(rtapi::getWetPaintStatus(paint));
    auto updated = paint.surface_list;
    for (auto &surface : updated) {
        if (patch.contains("lifetime_seconds")) {
            surface.settings.lifetime_seconds = patch["lifetime_seconds"];
        }
        if (patch.contains("diffusion")) {
            surface.settings.diffusion = patch["diffusion"];
        }
        if (patch.contains("runoff")) {
            surface.settings.runoff = patch["runoff"];
        }
        check(rtapi::validateWetPaintSurfaceSettings(surface.settings));
    }
    if (rebuild_demo) {
        // Registered surfaces are authoritative, including mixed texture sizes.
        config.erase("surfaces");
        config.erase("resolution");
        std::string output;
        check(rtapi::wetPaintDemoCommand("configure", config.dump(), output));
    }
    bool resized = false;
    if (patch.contains("resolution") && patch["resolution"] != current["resolution"]) {
        check(rtapi::resizeWetPaintSurfaces(patch["resolution"].get<int>()));
        resized = true;
    }
    for (const auto &surface : updated) {
        check(rtapi::setWetPaintSurfaceSettings(surface.object, surface.material_id,
                                                surface.settings));
    }
    return {{"ok", true}, {"balloons_reprepared", rebuild_demo}, {"wet_paint_cleared", resized}};
}
json preset(const std::string &name) {
    int count = 0;
    json result;
    if (name == "calm") {
        count = 6;
        result = {{"respawn_seconds", 6.0},   {"interval_min", 6.0}, {"interval_max", 9.0},
                  {"lifetime_seconds", 20.0}, {"diffusion", 0.12},   {"runoff", 0.15}};
    } else if (name == "balanced") {
        count = 12;
        result = {{"respawn_seconds", 3.5},   {"interval_min", 2.5}, {"interval_max", 5.0},
                  {"lifetime_seconds", 12.0}, {"diffusion", 0.24},   {"runoff", 0.35}};
    } else if (name == "lively") {
        count = 18;
        result = {{"respawn_seconds", 2.0},  {"interval_min", 2.0}, {"interval_max", 3.0},
                  {"lifetime_seconds", 8.0}, {"diffusion", 0.35},   {"runoff", 0.55}};
    } else {
        throw std::invalid_argument("Unknown preset: " + name);
    }
    const auto current = values();
    if (current.contains("balloons")) {
        result["balloons"] = std::min(count, capacity(session.authored_demo));
    }
    for (auto it = result.begin(); it != result.end();) {
        if (!current.contains(it.key())) {
            it = result.erase(it);
        } else {
            ++it;
        }
    }
    return result;
}
json status() {
    return {{"ready", session.ready},
            {"project", session.project},
            {"revision", session.revision},
            {"values", session.ready ? values() : json::object()},
            {"max_balloons", capacity(session.authored_demo)},
            {"saved", profiles["games"].contains(session.project)},
            {"error", session.error},
            {"storage", "player_game_settings.json"}};
}
} // namespace

void tickPlayerGameSettings(UIContext &context) {
    try {
        if (!ensureSession(context) || session.attempted_profile) {
            return;
        }
        session.attempted_profile = true;
        if (profiles["games"].contains(session.project)) {
            apply(profiles["games"][session.project]);
        }
    } catch (const std::exception &error) {
        session.error = error.what();
    }
}
void resetPlayerGameSettings() {
    session = Session{};
    profiles = {{"version", 1}, {"games", json::object()}};
    loaded = false;
    storage_error.clear();
}
rtapi::Result playerGameSettingsCommand(UIContext &context, const std::string &operation,
                                        const json &config, json &output) {
    try {
        require(config.is_object(), "Game settings config must be an object");
        ensureSession(context);
        if (operation == "game.settings.status") {
            require(config.empty(), "Status takes no settings");
            output = status();
            return rtapi::Result::success();
        }
        if (operation == "game.settings.clear_paint") {
            require(config.empty(), "Clear paint takes no settings");
            check(rtapi::resetWetPaint());
            output = {{"ok", true}};
            return rtapi::Result::success();
        }
        require(session.ready,
                "Wait for a game with paint surfaces or a balloon demo to finish loading");
        if (operation == "game.settings.preset") {
            require(config.size() == 1 && config.contains("name") && config["name"].is_string(),
                    "Preset requires only a name string");
            output = {{"patch", preset(config["name"].get<std::string>())}};
        } else if (operation == "game.settings.apply" || operation == "game.settings.save") {
            output = apply(config);
            session.attempted_profile = true;
            if (operation == "game.settings.save") {
                auto saved = profiles;
                auto snapshot = values();
                for (auto it = snapshot.begin(); it != snapshot.end();) {
                    if (it.value().is_null()) {
                        it = snapshot.erase(it);
                    } else {
                        ++it;
                    }
                }
                saved["games"][session.project] = snapshot;
                writeProfiles(saved);
                output["saved"] = true;
            }
        } else if (operation == "game.settings.restore") {
            require(config.empty(), "Restore takes no settings");
            auto saved = profiles;
            saved["games"].erase(session.project);
            writeProfiles(saved);
            output = {{"reload_project", context.active_model_path}, {"scheduled", true}};
        } else if (operation == "game.settings.renew_balloons") {
            require(config.empty(), "Renew balloons takes no settings");
            std::string text;
            check(rtapi::wetPaintDemoCommand("reset_balloons", "{}", text));
            output = json::parse(text);
        } else {
            return rtapi::Result::fail("Unknown game settings operation");
        }
        session.error.clear();
        return rtapi::Result::success();
    } catch (const std::exception &error) {
        session.error = error.what();
        return rtapi::Result::fail(error.what());
    }
}
} // namespace AppRuntime
