#include "Core/PlayerOperator.h"
#include "Core/Localization.h"
#include "Core/PlayerCamera.h"
#include "Core/PlayerGameSettings.h"
#include "Paint/HiddenGameObjects.h"
#include "Core/RuntimeProjectHost.h"
#include "Api/RtApiPlayerOperator.h"
#include "Api/RtApiBodyPose.h"
#include "Api/RtApiExternalBody.h"
#include "Api/RtApiGame.h"
#include "Api/RtApiPaintWet.h"
#include "Audio/AudioService.h"
#include "UIContext.h"
#include "scene_data.h"
#include "TriangleMesh.h"
#include "Input/ExternalBodyRigMath.h"
#include <SDL.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <future>
#include <set>
#include <tuple>
#include <utility>
#include <cctype>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <stdexcept>

extern SDL_Window *window;
namespace AppRuntime {
namespace {
using json = nlohmann::json;
namespace fs = std::filesystem;
json read(const fs::path &path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return json();
    }
    auto value = json::parse(stream, nullptr, false);
    return value.is_discarded() ? json() : value;
}
void require(bool valid, const std::string &error) {
    if (!valid) {
        throw std::invalid_argument(error);
    }
}
void keys(const json &config, std::initializer_list<const char *> names) {
    for (const auto &entry : config.items()) {
        require(std::any_of(names.begin(), names.end(),
                            [&](const char *name) { return entry.key() == name; }),
                "Unknown field: " + entry.key());
    }
}
std::string text(const json &value, const char *key) {
    require(value.contains(key) && value[key].is_string(), std::string(key) + " must be a string");
    return value[key].get<std::string>();
}
std::string pathKey(const fs::path &path) {
    std::error_code error;
    auto resolved = fs::weakly_canonical(path, error);
    auto result = projectPathToUtf8(error ? path.lexically_normal() : resolved);
#ifdef _WIN32
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
#endif
    return result;
}
struct ScanResult {
    json games = json::array();
    json warnings = json::array();
};
ScanResult scan(const json &roots) {
    ScanResult result;
    std::set<std::string> seen;
    const auto add = [&](const fs::path &path, const std::string &title, bool packaged) {
        std::error_code error;
        if (!fs::is_regular_file(path, error) || !seen.insert(pathKey(path)).second) {
            return;
        }
        auto thumbnail = path;
        thumbnail.replace_extension(".preview.png");
        result.games.push_back(
            {{"name", title.empty() ? projectPathToUtf8(path.stem()) : title},
             {"path", projectPathToUtf8(fs::absolute(path))},
             {"packaged", packaged},
             {"thumbnail",
              fs::is_regular_file(thumbnail, error) ? projectPathToUtf8(thumbnail) : ""}});
    };
    const auto packaged = read("player_scenes.json");
    if (packaged.is_object() && packaged.contains("scenes") && packaged["scenes"].is_array()) {
        for (const auto &entry : packaged["scenes"]) {
            if (entry.is_object() && entry.contains("path") && entry["path"].is_string()) {
                add(projectPathFromUtf8(entry["path"]), entry.value("name", ""), true);
            }
        }
    }
    for (const auto &root : roots) {
        std::error_code error;
        const auto directory = projectPathFromUtf8(root.get<std::string>());
        if (!fs::is_directory(directory, error)) {
            result.warnings.push_back("Folder unavailable: " + root.get<std::string>());
            continue;
        }
        fs::recursive_directory_iterator it(directory,
                                            fs::directory_options::skip_permission_denied, error);
        const fs::recursive_directory_iterator end;
        size_t visited = 0;
        while (it != end && visited++ < 50000 && result.games.size() < 4096) {
            const auto entry = *it;
            const auto name = projectPathToUtf8(entry.path().filename());
            if (entry.is_directory(error) &&
                (it.depth() >= 6 || name == ".git" || name == ".archive" ||
                 name == "node_modules" || name == ".venv" || entry.is_symlink(error))) {
                it.disable_recursion_pending();
            }
            auto extension = projectPathToUtf8(entry.path().extension());
            std::transform(extension.begin(), extension.end(), extension.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (extension == ".rtp") {
                add(entry.path(), "", false);
            }
            it.increment(error);
            if (error) {
                result.warnings.push_back("Some entries could not be read: " +
                                          root.get<std::string>());
                error.clear();
            }
        }
        if (visited >= 50000 || result.games.size() >= 4096) {
            result.warnings.push_back("Library scan limit reached; choose a narrower folder");
        }
    }
    std::sort(result.games.begin(), result.games.end(), [](const json &a, const json &b) {
        return a["name"].get<std::string>() < b["name"].get<std::string>();
    });
    return result;
}
bool calibrationReady(const json &camera) {
    const auto document = read(projectPathFromUtf8(camera.value("calibration", "")));
    return document.is_object() && document.contains("device_id") &&
           document["device_id"] == camera["device_id"] && document.contains("image_size") &&
           document["image_size"] == json::array({camera["width"], camera["height"]}) &&
           document.contains("points") && document["points"].is_array() &&
           document["points"].size() == 4;
}

void validateCamera(const json &candidate) {
    require(candidate["device_id"].is_string(), "device_id must be a string");
    for (const auto &range : {std::tuple<const char *, int, int>{"width", 160, 8192},
                              {"height", 120, 8192},
                              {"people", 1, 10}}) {
        const auto *key = std::get<0>(range);
        require(candidate[key].is_number_integer() && !candidate[key].is_boolean() &&
                    candidate[key].get<double>() >= std::get<1>(range) &&
                    candidate[key].get<double>() <= std::get<2>(range),
                std::string("Invalid ") + key);
    }
    for (const auto &range :
         {std::tuple<const char *, double, double>{"fps", 1, 240}, {"confidence", 0.1, 1}}) {
        const auto *key = std::get<0>(range);
        require(candidate[key].is_number() && std::isfinite(candidate[key].get<double>()) &&
                    candidate[key].get<double>() >= std::get<1>(range) &&
                    candidate[key].get<double>() <= std::get<2>(range),
                std::string("Invalid ") + key);
    }
    require(candidate["mirror"].is_boolean(), "mirror must be boolean");
    require(candidate.contains("calibration") && candidate["calibration"].is_string(),
            "calibration must be a path string");
}

rtapi::Result selectMonitor(int index) {
    require(index >= 0 && index < SDL_GetNumVideoDisplays(), "Display is unavailable");
    json previous;
    runtimeDisplayStatus(previous);
    auto result = setRuntimeDisplay({{"fullscreen", false}});
    if (!result.ok) {
        return result;
    }
    SDL_SetWindowPosition(window, SDL_WINDOWPOS_CENTERED_DISPLAY(index),
                          SDL_WINDOWPOS_CENTERED_DISPLAY(index));
    if (previous.value("fullscreen", false)) {
        result = setRuntimeDisplay({{"fullscreen", true}});
    }
    return result;
}

struct State {
    json preferences = {{"version", 1}, {"locale", "en"},
                        {"roots", json::array({"scenes"})},
                        {"camera",
                         {{"device_id", ""},
                          {"width", 1280},
                          {"height", 720},
                          {"fps", 30.0},
                          {"people", 4},
                          {"confidence", 0.5},
                          {"mirror", false},
                          {"calibration", ""}}},
                        {"appearance", "camera"},
                        {"avatar_profiles", json::object()},
                        {"avatar_assignments", json::object()},
                        {"display", json::object()},
                        {"audio", json::object()},
                        {"monitor", json::object()}};
    json launch_display = json::object();
    ScanResult library;
    std::future<ScanResult> scanning;
    PlayerCamera camera;
    std::string pending_open;
    std::string current_project;
    std::string error;
    bool display_restored = false;
    bool loaded = false;
    bool camera_was_active = false;
    bool camera_owns_card = false;
    std::string audio_project;
    UIContext *context = nullptr;
    fs::path settings = fs::absolute("player_operator.json");
    void load() {
        if (loaded) {
            return;
        }
        loaded = true;
        const auto defaults = preferences;
        try {
            const auto saved = read(settings);
            if (saved.is_object() && saved.contains("version") &&
                saved["version"].is_number_integer() && saved["version"].get<int>() == 1) {
                for (const char *name : {"roots", "appearance", "avatar_profiles",
                                         "avatar_assignments", "display", "audio", "monitor", "locale"}) {
                    if (saved.contains(name) && saved[name].type() == preferences[name].type()) {
                        preferences[name] = saved[name];
                    }
                }
                if (saved.contains("camera") && saved["camera"].is_object()) {
                    preferences["camera"].update(saved["camera"]);
                }
                validateCamera(preferences["camera"]);
                require(preferences["roots"].size() <= 16, "Too many saved library folders");
                for (const auto &root : preferences["roots"]) {
                    require(root.is_string(), "Saved library folder must be a path string");
                }
                require(preferences["appearance"] == "camera" ||
                            preferences["appearance"] == "avatar",
                        "Unknown saved player appearance");
            }
        } catch (const std::exception &exception) {
            preferences = defaults;
            error =
                std::string("Operator settings were invalid; defaults loaded: ") + exception.what();
        }
        std::string language_error;
        if (!Localization::setLocale(preferences.value("locale", std::string("en")), language_error)) {
            const auto unavailable = language_error;
            std::string ignored;
            Localization::setLocale("en", ignored);
            error = unavailable;
        }
        scanning =
            std::async(std::launch::async, [roots = preferences["roots"]] { return scan(roots); });
    }
    void save() {
        auto temporary = settings;
        temporary += ".tmp";
        {
            std::ofstream stream(temporary, std::ios::binary);
            stream << preferences.dump(2);
            require(static_cast<bool>(stream), "Could not save operator settings");
        }
        std::error_code code;
#ifdef _WIN32
        // Windows replace preserves the old settings when publication fails.
        require(MoveFileExW(temporary.c_str(), settings.c_str(),
                            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0,
                "Could not publish operator settings");
#else
        fs::rename(temporary, settings, code);
        require(!code, "Could not publish operator settings");
#endif
    }
};
std::unique_ptr<State> instance;
State &state() {
    if (!instance) {
        instance = std::make_unique<State>();
    }
    instance->load();
    return *instance;
}
void avatarVisible(UIContext &context, const std::string &character, bool visible) {
    for (auto &model : context.scene.importedModelContexts) {
        if (model.importName != character) {
            continue;
        }
        for (const auto &member : model.members) {
            const auto mesh = std::dynamic_pointer_cast<TriangleMesh>(member);
            if (mesh) {
                const auto result = rtapi::setObjectVisible(mesh->nodeName, visible);
                require(result.ok, result.error);
            }
        }
        model.visible = visible;
    }
}

json characters(State &s, UIContext &context) {
    json result = json::array();
    json bindings;
    rtapi::getExternalBodyPoses(bindings);
    for (const auto &character : rtapi::listAnimCharacters()) {
        const std::string key = s.current_project + "|" + character.name;
        auto mapping = s.preferences["avatar_profiles"].value(key, json::object());
        for (const auto &binding : bindings.value("bindings", json::array())) {
            if (binding.value("character", "") == character.name) {
                mapping = binding["config"];
            }
        }
        bool ready = false;
        std::string reason = "Needs a prepared body-to-rig mapping";
        if (!mapping.empty()) {
            for (const auto &model : context.scene.importedModelContexts) {
                if (model.importName != character.name) {
                    continue;
                }
                Input::ExternalRigMapping parsed;
                ready = Input::parseExternalRigMapping(mapping, model.nodeHierarchy,
                                                       model.rigAnatomy.controls, parsed, reason);
                break;
            }
        }
        result.push_back({{"name", character.name},
                          {"bones", character.bone_count},
                          {"ready", ready},
                          {"reason", ready ? "Prepared mapping" : reason},
                          {"visible", character.visible}});
    }
    return result;
}
} // namespace

void tickPlayerOperator() {
    auto &s = state();
    s.camera.tick();
    if (s.context && !s.context->active_model_path.empty()) {
        const auto project = pathKey(projectPathFromUtf8(s.context->active_model_path));
        if (!s.current_project.empty() && s.current_project != project &&
            s.camera.status().value("active", false)) {
            s.camera.command("stop", json::object());
            rtapi::clearBodyPoses();
            s.error = "Game changed; restart camera input for the new scene";
        }
        s.current_project = project;
    }
    if (s.context) {
        tickPlayerGameSettings(*s.context);
    }
    if (s.scanning.valid() &&
        s.scanning.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        try {
            s.library = s.scanning.get();
        } catch (const std::exception &error) {
            s.error = error.what();
        }
    }
    const bool active = s.camera.status().value("active", false);
    if (s.camera_was_active && !active) {
        rtapi::clearBodyPoses();
        if (s.camera_owns_card) {
            setRuntimeDisplay({{"video_card", false}});
            s.camera_owns_card = false;
        }
    }
    s.camera_was_active = active;
    if (!s.display_restored) {
        s.display_restored = true;
        try {
            const auto saved_monitor = s.preferences["monitor"].value("name", std::string());
            if (!saved_monitor.empty()) {
                int selected = -1;
                for (int index = 0; index < SDL_GetNumVideoDisplays(); ++index) {
                    const auto *name = SDL_GetDisplayName(index);
                    if (name && saved_monitor == name) {
                        selected = index;
                        break;
                    }
                }
                if (selected >= 0) {
                    const auto result = selectMonitor(selected);
                    require(result.ok, result.error);
                } else {
                    s.error = "Saved screen is unavailable; current screen kept";
                }
            }
            auto patch = s.preferences["display"];
            for (const auto &entry : s.launch_display.items()) {
                patch.erase(entry.key());
            }
            if (s.launch_display.contains("width") || s.launch_display.contains("height")) {
                patch.erase("fullscreen");
            }
            if (!patch.empty()) {
                const auto restored = setRuntimeDisplay(patch);
                require(restored.ok, restored.error);
            }
        } catch (const std::exception &error) {
            s.error = std::string("Saved display setting unavailable: ") + error.what();
        }
    }
    if (s.context && s.context->scene.audio && s.audio_project != s.current_project) {
        s.audio_project = s.current_project;
        if (!s.preferences["audio"].empty()) {
            try {
                auto patch = s.preferences["audio"];
                patch["bus"] = "Master";
                s.context->scene.audio->invoke("mixer_update", patch);
            } catch (const std::exception &error) {
                s.error = std::string("Saved room audio setting unavailable: ") + error.what();
            }
        }
    }
    if (!s.pending_open.empty()) {
        const auto path = std::exchange(s.pending_open, std::string());
        const auto loaded = rtapi::openProject(path);
        if (loaded.ok) {
            s.current_project = pathKey(projectPathFromUtf8(path));
        } else {
            s.error = loaded.error;
        }
    }
}
void setPlayerOperatorLaunchOverrides(const json &display) {
    state().launch_display = display;
}
void shutdownPlayerOperator() {
    resetPlayerGameSettings();
    instance.reset();
}
// UI preview is a borrowed SDL texture; the camera service remains its sole owner.
SDL_Texture *playerOperatorPreview(SDL_Renderer *renderer) {
    return state().camera.preview(renderer);
}

rtapi::Result playerOperatorCommand(UIContext &context, const std::string &operation,
                                    const json &config, json &output) {
    try {
        require(config.is_object(), "Operator config must be an object");
        auto &s = state();
        s.context = &context;
        if (!context.active_model_path.empty()) {
            s.current_project = pathKey(projectPathFromUtf8(context.active_model_path));
        }
        if (operation == "language.status") {
            keys(config, {});
            output = Localization::status();
            return rtapi::Result::success();
        }
        if (operation == "language.set") {
            keys(config, {"locale"});
            const auto code = text(config, "locale");
            const auto previous = Localization::locale();
            std::string language_error;
            if (!Localization::setLocale(code, language_error)) {
                return rtapi::Result::fail(language_error);
            }
            const auto saved = s.preferences["locale"];
            s.preferences["locale"] = code;
            try {
                s.save();
            } catch (...) {
                s.preferences["locale"] = saved;
                Localization::setLocale(previous, language_error);
                throw;
            }
            output = Localization::status();
            return rtapi::Result::success();
        }
        if (operation.rfind("game.hidden.", 0) == 0) {
            auto input = config;
            input["operation"] = operation.substr(12);
            std::string error;
            return Paint::hiddenGameObjectsCommand(context.scene, input, output, error)
                ? rtapi::Result::success() : rtapi::Result::fail(error);
        }
        if (operation.rfind("game.settings.", 0) == 0) {
            if (operation == "game.settings.restore" && s.camera.status().value("active", false)) {
                return rtapi::Result::fail("Stop camera input before restoring the authored game");
            }
            const auto result = playerGameSettingsCommand(context, operation, config, output);
            if (result.ok && output.contains("reload_project")) {
                s.pending_open = output["reload_project"].get<std::string>();
            }
            return result;
        }
        if (operation == "status") {
            keys(config, {});
            json display, poses;
            runtimeDisplayStatus(display);
            rtapi::getBodyPoses(poses);
            output = {{"preferences", s.preferences},
                      {"games", s.library.games},
                      {"scan_warnings", s.library.warnings},
                      {"scanning", s.scanning.valid()},
                      {"camera", s.camera.status()},
                      {"display", display},
                      {"body", poses},
                      {"characters", characters(s, context)},
                      {"project", s.current_project},
                      {"error", s.error}};
            json game_settings;
            const auto game_result = playerGameSettingsCommand(context, "game.settings.status",
                                                               json::object(), game_settings);
            output["game_settings"] = game_result.ok ? game_settings :
                json{{"ready", false}, {"error", game_result.error}};
            json hidden;
            std::string hidden_error;
            if (Paint::hiddenGameObjectsCommand(context.scene, json::object(), hidden, hidden_error)) {
                output["hidden_objects"] = std::move(hidden);
            }
            output["language"] = Localization::status();
            output["calibration_ready"] = calibrationReady(s.preferences["camera"]);
            output["monitors"] = json::array();
            for (int index = 0; index < SDL_GetNumVideoDisplays(); ++index) {
                const auto *name = SDL_GetDisplayName(index);
                output["monitors"].push_back(
                    {{"index", index},
                     {"name", name ? name : "Display"},
                     {"selected", SDL_GetWindowDisplayIndex(window) == index}});
            }
            if (context.scene.audio) {
                output["audio"] = context.scene.audio->invoke("mixer_get", json::object());
            }
        } else if (operation == "library.refresh" || operation == "library.add" ||
                   operation == "library.remove") {
            keys(config, operation == "library.refresh"
                             ? std::initializer_list<const char *>{}
                             : std::initializer_list<const char *>{"path"});
            require(!s.scanning.valid(), "Library scan is already running");
            auto roots = s.preferences["roots"];
            if (operation != "library.refresh") {
                const auto path = fs::absolute(projectPathFromUtf8(text(config, "path")));
                if (operation == "library.add") {
                    require(fs::is_directory(path), "Choose an existing game folder");
                    bool duplicate = false;
                    for (const auto &root : roots) {
                        duplicate |= pathKey(projectPathFromUtf8(root)) == pathKey(path);
                    }
                    require(!duplicate, "This folder is already in the library");
                    require(roots.size() < 16, "At most 16 library folders are supported");
                    roots.push_back(projectPathToUtf8(path));
                } else {
                    roots.erase(std::remove_if(roots.begin(), roots.end(),
                                               [&](const json &root) {
                                                   return pathKey(projectPathFromUtf8(root)) ==
                                                          pathKey(path);
                                               }),
                                roots.end());
                }
                s.preferences["roots"] = roots;
                s.save();
            }
            s.scanning = std::async(std::launch::async, [roots] { return scan(roots); });
        } else if (operation == "library.open") {
            keys(config, {"path"});
            const auto path = fs::absolute(projectPathFromUtf8(text(config, "path")));
            auto extension = projectPathToUtf8(path.extension());
            std::transform(extension.begin(), extension.end(), extension.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            require(fs::is_regular_file(path) && extension == ".rtp",
                    "Choose an existing .rtp project");
            require(!s.camera.status().value("active", false),
                    "Stop camera input before changing games");
            s.pending_open = projectPathToUtf8(path);
            output = {{"scheduled", true}, {"path", s.pending_open}};
        } else if (operation == "camera.refresh") {
            keys(config, {});
            const auto result = s.camera.command("refresh", s.preferences["camera"]);
            if (!result.ok) {
                return result;
            }
        } else if (operation == "camera.configure") {
            keys(config, {"device_id", "width", "height", "fps", "people", "confidence", "mirror"});
            require(!s.camera.status().value("active", false),
                    "Stop camera input before changing its settings");
            auto candidate = s.preferences["camera"];
            candidate.update(config);
            validateCamera(candidate);
            for (const auto *key : {"device_id", "width", "height"}) {
                if (candidate[key] != s.preferences["camera"][key]) {
                    candidate["calibration"] = "";
                }
            }
            s.preferences["camera"] = candidate;
            s.save();
        } else if (operation == "camera.start" || operation == "camera.preview") {
            keys(config, {});
            auto request = s.preferences["camera"];
            require(!request.value("device_id", "").empty(), "Select a connected camera first");
            const auto catalog = s.camera.status();
            bool supported = false;
            for (const auto &device : catalog["catalog"]["devices"]) {
                if (device["id"] != request["device_id"]) {
                    continue;
                }
                for (const auto &mode : device["modes"]) {
                    supported |=
                        mode["width"] == request["width"] && mode["height"] == request["height"] &&
                        std::abs(mode["fps"].get<double>() - request["fps"].get<double>()) < 0.01;
                }
            }
            require(supported, "Refresh the selected camera and choose one of its supported modes");
            const bool tracking = operation == "camera.start";
            require(!tracking || calibrationReady(request),
                    "Calibrate the selected camera and resolution before starting tracking");
            fs::path base = fs::absolute("perception");
            fs::path python = base / "python/python.exe";
            fs::path worker = base / "bridge/operator_camera.py";
            fs::path model = base / "models/pose_landmarker.task";
            if (!fs::is_regular_file(python)) {
                for (const auto &root : {fs::path("../../tools/perception_bridge"),
                                         fs::path("tools/perception_bridge")}) {
                    if (fs::is_regular_file(root / ".venv/Scripts/python.exe")) {
                        base = fs::absolute(root);
                        python = base / ".venv/Scripts/python.exe";
                        worker = base / "operator_camera.py";
                        model = base / "models/pose_landmarker.task";
                        break;
                    }
                }
            }
            if (tracking && s.preferences["appearance"] == "avatar") {
                const auto assignments =
                    s.preferences["avatar_assignments"].value(s.current_project, json::object());
                require(!assignments.empty(),
                        "Choose an avatar for this game before starting tracking");
                for (const auto &assignment : assignments.items()) {
                    const auto key = s.current_project + "|" + assignment.key();
                    require(s.preferences["avatar_profiles"].contains(key),
                            "Avatar mapping is missing");
                    const auto bound = rtapi::setExternalBodyPose(
                        {{"character", assignment.key()},
                         {"person_id", assignment.value()},
                         {"config", s.preferences["avatar_profiles"][key]}});
                    require(bound.ok, bound.error);
                    avatarVisible(context, assignment.key(), true);
                }
            }
            request["python"] = projectPathToUtf8(python);
            request["worker"] = projectPathToUtf8(worker);
            request["model"] = projectPathToUtf8(model);
            request["tracking"] = tracking;
            request["video_card"] = tracking && s.preferences["appearance"] == "camera";
            request["pipe"] = R"(\\.\pipe\RayTrophiPlayer)";
            const bool previous_card = playerVideoCardEnabled();
            if (tracking) {
                const auto display_result =
                    setRuntimeDisplay({{"video_card", request["video_card"]}});
                if (!display_result.ok) {
                    return display_result;
                }
            }
            const auto result = s.camera.command("start", request);
            if (!result.ok) {
                if (tracking) {
                    setRuntimeDisplay({{"video_card", previous_card}});
                }
                return result;
            }
            s.camera_was_active = true;
            s.camera_owns_card = request["video_card"].get<bool>();
        } else if (operation == "camera.stop") {
            keys(config, {});
            const auto result = s.camera.command("stop", config);
            if (!result.ok) {
                return result;
            }
        } else if (operation == "display.configure") {
            keys(config, {"fullscreen", "render_width", "render_height", "render_scale",
                          "pacing_hz", "hud"});
            const auto result = setRuntimeDisplay(config);
            if (!result.ok) {
                return result;
            }
            s.preferences["display"].update(config);
            s.save();
        } else if (operation == "display.monitor") {
            keys(config, {"index"});
            require(config.contains("index") && config["index"].is_number_integer(),
                    "index must be an integer");
            const int index = config["index"];
            const auto result = selectMonitor(index);
            if (!result.ok) {
                return result;
            }
            const auto *name = SDL_GetDisplayName(index);
            s.preferences["monitor"] = {{"name", name ? name : ""}, {"index", index}};
            s.save();
        } else if (operation == "calibration.save") {
            const auto camera_state = s.camera.status();
            require(!camera_state.value("active", false) ||
                        camera_state.value("state", "") == "preview",
                    "Stop tracking before saving a new floor calibration");
            keys(config,
                 {"points", "camera_yaw_degrees", "camera_pitch_degrees", "camera_roll_degrees"});
            auto document = config;
            const auto &camera = s.preferences["camera"];
            require(!camera.value("device_id", "").empty(), "Select a camera before calibration");
            require(document.contains("points") && document["points"].is_array() &&
                        document["points"].size() == 4,
                    "Choose four measured floor corners");
            for (const auto &point : document["points"]) {
                for (const auto *name : {"pixel", "room_xz"}) {
                    require(point.contains(name) && point[name].is_array() &&
                                point[name].size() == 2,
                            "Calibration points require pixel and room_xz pairs");
                    for (const auto &value : point[name]) {
                        require(value.is_number() && std::isfinite(value.get<double>()) &&
                                    std::abs(value.get<double>()) < 10000,
                                "Invalid calibration coordinate");
                    }
                }
                require(point["pixel"][0].get<double>() >= 0 &&
                            point["pixel"][0].get<double>() < camera["width"].get<int>() &&
                            point["pixel"][1].get<double>() >= 0 &&
                            point["pixel"][1].get<double>() < camera["height"].get<int>(),
                        "Floor corner is outside the camera image");
            }
            // Reject duplicate/collinear correspondences before publishing the document.
            for (const auto *name : {"pixel", "room_xz"}) {
                for (int omitted = 0; omitted < 4; ++omitted) {
                    std::vector<json> triangle;
                    for (int i = 0; i < 4; ++i) {
                        if (i != omitted) {
                            triangle.push_back(document["points"][i][name]);
                        }
                    }
                    const double ax = triangle[1][0].get<double>() - triangle[0][0].get<double>();
                    const double ay = triangle[1][1].get<double>() - triangle[0][1].get<double>();
                    const double bx = triangle[2][0].get<double>() - triangle[0][0].get<double>();
                    const double by = triangle[2][1].get<double>() - triangle[0][1].get<double>();
                    require(std::abs(ax * by - ay * bx) > 1e-6,
                            "Floor corners must be distinct and must not be collinear");
                }
            }
            for (const auto *name :
                 {"camera_yaw_degrees", "camera_pitch_degrees", "camera_roll_degrees"}) {
                if (!document.contains(name)) {
                    document[name] = 0.0;
                }
                require(document[name].is_number() && std::isfinite(document[name].get<double>()) &&
                            std::abs(document[name].get<double>()) <= 360,
                        "Invalid camera orientation");
            }
            document["image_size"] = {camera["width"], camera["height"]};
            document["device_id"] = camera["device_id"];
            const auto path = s.settings.parent_path() / "player_floor_calibration.json";
            {
                std::ofstream stream(path, std::ios::binary);
                stream << document.dump(2);
                require(static_cast<bool>(stream), "Could not save floor calibration");
            }
            s.preferences["camera"]["calibration"] = projectPathToUtf8(path);
            s.save();
            output = {{"ok", true}, {"path", projectPathToUtf8(path)}};
        } else if (operation == "audio.configure") {
            keys(config, {"gain", "muted"});
            require(context.scene.audio != nullptr, "Audio service unavailable");
            auto patch = config;
            patch["bus"] = "Master";
            output = context.scene.audio->invoke("mixer_update", patch);
            s.preferences["audio"].update(config);
            s.save();
        } else if (operation == "avatar.bind") {
            keys(config, {"character", "person_id", "mapping"});
            require(!s.camera.status().value("active", false),
                    "Stop camera input before changing player appearance");
            const auto character = text(config, "character");
            const auto key = s.current_project + "|" + character;
            json mapping = config.value("mapping", json::object());
            if (mapping.empty() && s.preferences["avatar_profiles"].contains(key)) {
                mapping = s.preferences["avatar_profiles"][key];
            }
            if (mapping.empty()) {
                json bindings;
                rtapi::getExternalBodyPoses(bindings);
                for (const auto &binding : bindings.value("bindings", json::array())) {
                    if (binding.value("character", "") == character) {
                        mapping = binding["config"];
                    }
                }
            }
            require(!mapping.empty(), "This character needs a prepared body-to-rig mapping");
            const auto result =
                rtapi::setExternalBodyPose({{"character", character},
                                            {"person_id", config.value("person_id", json(0))},
                                            {"config", mapping}});
            if (!result.ok) {
                return result;
            }
            avatarVisible(context, character, true);
            s.preferences["avatar_profiles"][key] = mapping;
            if (!s.preferences["avatar_assignments"].contains(s.current_project)) {
                s.preferences["avatar_assignments"][s.current_project] = json::object();
            }
            s.preferences["avatar_assignments"][s.current_project][character] =
                config.value("person_id", json(0));
            s.preferences["appearance"] = "avatar";
            setRuntimeDisplay({{"video_card", false}});
            s.save();
        } else if (operation == "appearance.camera") {
            keys(config, {});
            require(!s.camera.status().value("active", false),
                    "Stop camera input before changing player appearance");
            json bindings;
            rtapi::getExternalBodyPoses(bindings);
            for (const auto &binding : bindings.value("bindings", json::array())) {
                const auto result =
                    rtapi::clearExternalBodyPose({{"character", binding["character"]}});
                if (!result.ok) {
                    return result;
                }
                const auto character = binding["character"].get<std::string>();
                if (s.preferences["avatar_profiles"].contains(s.current_project + "|" +
                                                              character)) {
                    avatarVisible(context, character, false);
                }
            }
            s.preferences["appearance"] = "camera";
            setRuntimeDisplay({{"video_card", true}});
            s.save();
        } else if (operation == "game.start" || operation == "game.stop" ||
                   operation == "game.reset") {
            keys(config, {});
            const auto command = operation.substr(5);
            std::string demo;
            auto result = rtapi::wetPaintDemoCommand("status", "{}", demo);
            if (!result.ok) {
                return result;
            }
            const auto status = json::parse(demo, nullptr, false);
            if (status.is_object() && status.value("prepared", false)) {
                // Entering gameplay must preserve the scene author's input mode.
                // Demo start explicitly enables its random-pop scheduler, even
                // when the project was saved for mouse/body-driven play.
                const bool automatic =
                    status.value("config", json::object()).value("autostart", false);
                const auto demo_command = command == "start" && !automatic ? "stop" : command;
                result = rtapi::wetPaintDemoCommand(demo_command, "{}", demo);
                if (!result.ok) {
                    return result;
                }
                output = json::parse(demo, nullptr, false);
            } else {
                json session;
                result = rtapi::gameSessionCommand("status", json::object(), session);
                if (!result.ok) {
                    return result;
                }
                if (session.value("configured", false)) {
                    if (command != "start" || session.value("state", "") == "attract") {
                        result = rtapi::gameSessionCommand(command, json::object(), output);
                        if (!result.ok) {
                            return result;
                        }
                    }
                } else {
                    output = {{"ok", true}, {"mode", "scene_viewer"}};
                }
            }
        } else {
            return rtapi::Result::fail("Unknown operator operation: " + operation);
        }
        if (operation != "status") {
            s.error.clear();
        }
        if (output.is_null()) {
            output = {{"ok", true}};
        }
        return rtapi::Result::success();
    } catch (const std::exception &error) {
        if (instance) {
            instance->error = error.what();
        }
        return rtapi::Result::fail(error.what());
    }
}
} // namespace AppRuntime
