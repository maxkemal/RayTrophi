#include "HiddenGameObjectsInternal.h"
#include "Api/RtApi.h"
#include "Api/RtApiPaintWet.h"
#include "Paint/BalloonPaintDemo.h"
#include "Paint/BalloonInteraction.h"
#include "Paint/BalloonBodyInteraction.h"
#include "Paint/WetPaintGpu.h"
#include "Paint/HiddenPaintSafety.h"
#include "TriangleMesh.h"
#include "UIContext.h"
#include "scene_data.h"
#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>

namespace Paint::HiddenDetail {
State& state() {
    static State value;
    return value;
}
void require(bool ok, const std::string& error) {
    if (!ok) {
        throw std::invalid_argument(error);
    }
}
Vec3 vector3(const json& value) {
    require(value.is_array() && value.size() == 3, "Expected XYZ array");
    float result[3];
    for (int axis = 0; axis < 3; ++axis) {
        require(value[axis].is_number(), "XYZ must contain numbers");
        result[axis] = value[axis].get<float>();
        require(std::isfinite(result[axis]) && std::abs(result[axis]) <= 10000,
                "XYZ must be finite and within +/-10000 metres");
    }
    return Vec3(result[0], result[1], result[2]);
}
json vectorJson(const Vec3& value) {
    return json::array({value.x, value.y, value.z});
}
json entryJson(const Entry& entry) {
    json matrix = json::array();
    for (const auto& row : entry.base.m) {
        for (float value : row) {
            matrix.push_back(value);
        }
    }
    return {{"object", entry.object}, {"visibility", entry.visibility},
            {"randomize_on_reset", entry.randomize}, {"blocks_player", entry.blocks_player},
            {"blocks_balloon", entry.blocks_balloon}, {"yaw_min", entry.yaw_min},
            {"yaw_max", entry.yaw_max}, {"base_transform", matrix},
            {"paint_collider_mode", entry.collider}, {"sdf_resolution_mode", entry.sdf_resolution}};
}
Box objectBox(const std::string& name) {
    rtapi::ObjectInfo info;
    const auto result = rtapi::getObjectInfo(name, info);
    require(result.ok && info.has_bounds, "Object has no flat bounds: " + name);
    return {info.world_min, info.world_max};
}
bool overlaps(const Box& a, const Box& b, float margin) {
    return a.high.x + margin > b.low.x && a.low.x - margin < b.high.x &&
           a.high.y + margin > b.low.y && a.low.y - margin < b.high.y &&
           a.high.z + margin > b.low.z && a.low.z - margin < b.high.z;
}
void discover(SceneData& scene) {
    auto& s = state();
    if (s.scene != &scene) {
        s = State{};
        s.scene = &scene;
        clearHiddenPlayerHistory();
    }
    const auto surfaces = wetPaintGpu().surfaces();
    for (auto it = s.entries.begin(); it != s.entries.end();) {
        const bool paintable = std::any_of(surfaces.begin(), surfaces.end(), [&](const auto& surface) {
            return surface.object == it->first;
        });
        if (!scene.getFlatNodeMesh(it->first) || !paintable) {
            it = s.entries.erase(it);
        } else {
            ++it;
        }
    }
    const auto colliders = balloonPaintColliders(scene);
    for (const auto& surface : wetPaintGpu().surfaces()) {
        if (!surface.settings.reveal_opacity || s.entries.count(surface.object)) {
            continue;
        }
        auto* mesh = scene.getFlatNodeMesh(surface.object);
        if (!mesh || mesh->transient || mesh->hasSkinWeights()) {
            continue;
        }
        Entry entry;
        entry.object = surface.object;
        const auto pose = rtapi::getObjectTransform(entry.object, entry.base);
        if (!pose.ok) {
            continue;
        }
        entry.visibility = mesh->visible ? "paint" : "hidden";
        for (const auto& collider : colliders) {
            if (collider.value("object", std::string()) == entry.object &&
                collider.value("managed", false)) {
                entry.collider = collider.value("mode", std::string("none"));
                entry.sdf_resolution = collider.value("sdf_resolution_mode", 1);
                break;
            }
        }
        if (s.entries.size() < 64) {
            s.entries.emplace(entry.object, std::move(entry));
        }
    }
}
void preflightVisibility(SceneData& scene, const Entry& entry) {
    require(rtapi::renderStatus().state != rtapi::RenderJobState::Rendering,
            "Scene is locked by the final render job");
    const auto desired = entry.visibility == "hidden" ? std::string("none") : entry.collider;
    std::string current = "none";
    int resolution = entry.sdf_resolution;
    bool ready = true;
    for (const auto& collider : balloonPaintColliders(scene)) {
        if (collider.value("object", std::string()) == entry.object &&
            collider.value("managed", false)) {
            current = collider.value("mode", std::string("none"));
            resolution = collider.value("sdf_resolution_mode", 1);
            ready = collider.value("ready", false);
        }
    }
    if (current == desired && (desired == "none" || resolution == entry.sdf_resolution)) {
        return;
    }
    require(ready, "Wait for paint collider: " + entry.object);
    for (const char* name : {"BalloonPaintRed", "BalloonPaintBlue",
                             "BalloonPaintYellow", "BalloonPaintGreen"}) {
        require(std::any_of(scene.particle_systems.begin(), scene.particle_systems.end(),
                    [&](const auto& system) { return system.name == name && system.runtime; }),
                std::string("Droplet system missing: ") + name);
    }
}
void applyVisibility(SceneData& scene, const Entry& entry) {
    const auto* mesh = scene.getFlatNodeMesh(entry.object);
    if (mesh && mesh->visible != (entry.visibility != "hidden")) {
        const auto visible = rtapi::setObjectVisible(entry.object, entry.visibility != "hidden");
        require(visible.ok, visible.error);
    }
    for (const auto& surface : wetPaintGpu().surfaces()) {
        if (surface.object != entry.object) {
            continue;
        }
        auto settings = surface.settings;
        const bool reveal = entry.visibility != "shown";
        if (settings.reveal_opacity != reveal) {
            settings.reveal_opacity = reveal;
            require(wetPaintGpu().setSurfaceSettings(entry.object, surface.material_id, settings),
                    "Could not update reveal mode: " + entry.object);
        }
    }
    const auto desired = entry.visibility == "hidden" ? std::string("none") : entry.collider;
    std::string current = "none";
    int resolution = entry.sdf_resolution;
    for (const auto& collider : balloonPaintColliders(scene)) {
        if (collider.value("object", std::string()) == entry.object &&
            collider.value("managed", false)) {
            current = collider.value("mode", std::string("none"));
            resolution = collider.value("sdf_resolution_mode", 1);
            break;
        }
    }
    if (current != desired || (desired != "none" && resolution != entry.sdf_resolution)) {
        std::string error;
        require(setBalloonPaintCollider(scene, entry.object, desired,
                                        entry.sdf_resolution, error), error);
    }
}
} // namespace Paint::HiddenDetail

namespace Paint {
using namespace HiddenDetail;
namespace {
using namespace HiddenDetail;
void keys(const json& input, std::initializer_list<const char*> allowed) {
    require(input.is_object(), "Expected hidden object settings object");
    for (const auto& item : input.items()) {
        bool found = false;
        for (const char* key : allowed) {
            found = found || item.key() == key;
        }
        require(found, "Unknown hidden object field: " + item.key());
    }
}
Entry readEntry(SceneData& scene, const json& input) {
    keys(input, {"object", "visibility", "randomize_on_reset", "blocks_player", "blocks_balloon",
                 "yaw_min", "yaw_max", "base_transform", "paint_collider_mode",
                 "sdf_resolution_mode"});
    require(input.contains("object") && input["object"].is_string(), "Object name required");
    Entry entry;
    entry.object = input["object"].get<std::string>();
    auto* mesh = scene.getFlatNodeMesh(entry.object);
    require(!scene.isEditorPendingDeleteObjectName(entry.object), "Object is pending delete");
    require(mesh && !mesh->transient && !mesh->hasSkinWeights(),
            "Hidden object must be a static flat mesh: " + entry.object);
    const auto surfaces = wetPaintGpu().surfaces();
    require(std::any_of(surfaces.begin(), surfaces.end(), [&](const auto& surface) {
        return surface.object == entry.object;
    }), "Hidden object requires a wet paint surface: " + entry.object);
    const auto original = state().entries.find(entry.object);
    if (original != state().entries.end()) {
        entry = original->second;
    } else {
        const auto pose = rtapi::getObjectTransform(entry.object, entry.base);
        require(pose.ok, pose.error);
    }
    for (const char* key : {"randomize_on_reset", "blocks_player", "blocks_balloon"}) {
        require(!input.contains(key) || input[key].is_boolean(), std::string(key) + " must be bool");
    }
    entry.randomize = input.value("randomize_on_reset", entry.randomize);
    entry.blocks_player = input.value("blocks_player", entry.blocks_player);
    entry.blocks_balloon = input.value("blocks_balloon", entry.blocks_balloon);
    if (input.contains("visibility")) {
        require(input["visibility"].is_string(), "visibility must be paint|shown|hidden");
        entry.visibility = input["visibility"].get<std::string>();
    }
    require(entry.visibility == "paint" || entry.visibility == "shown" ||
                entry.visibility == "hidden", "visibility must be paint|shown|hidden");
    for (const char* key : {"yaw_min", "yaw_max"}) {
        require(!input.contains(key) || input[key].is_number(), std::string(key) + " must be numeric");
    }
    entry.yaw_min = input.value("yaw_min", entry.yaw_min);
    entry.yaw_max = input.value("yaw_max", entry.yaw_max);
    require(std::isfinite(entry.yaw_min) && std::isfinite(entry.yaw_max) &&
                entry.yaw_min >= -45 && entry.yaw_max <= 45 && entry.yaw_min <= entry.yaw_max,
            "Yaw range must be ordered within -45..45 degrees");
    if (input.contains("base_transform")) {
        const auto& matrix = input["base_transform"];
        require(matrix.is_array() && matrix.size() == 16, "base_transform needs 16 numbers");
        for (int i = 0; i < 16; ++i) {
            require(matrix[i].is_number(), "base_transform needs numeric values");
            entry.base.m[i / 4][i % 4] = matrix[i].get<float>();
            require(std::isfinite(entry.base.m[i / 4][i % 4]) &&
                        std::abs(entry.base.m[i / 4][i % 4]) <= 10000,
                    "base_transform must be finite within +/-10000");
        }
        require(std::abs(entry.base.m[3][0]) + std::abs(entry.base.m[3][1]) +
                    std::abs(entry.base.m[3][2]) < 1e-5f &&
                    std::abs(entry.base.m[3][3] - 1) < 1e-5f, "base_transform must be affine");
        const Vec3 a(entry.base.m[0][0], entry.base.m[1][0], entry.base.m[2][0]);
        const Vec3 b(entry.base.m[0][1], entry.base.m[1][1], entry.base.m[2][1]);
        const Vec3 c(entry.base.m[0][2], entry.base.m[1][2], entry.base.m[2][2]);
        const float determinant = Vec3::dot(a, b.cross(c));
        require(std::isfinite(determinant) && std::abs(determinant) > 1e-8f,
                "Singular base_transform");
    }
    if (input.contains("paint_collider_mode")) {
        require(input["paint_collider_mode"].is_string(), "Invalid paint collider mode");
        entry.collider = input["paint_collider_mode"].get<std::string>();
    }
    require(entry.collider == "none" || entry.collider == "aabb" || entry.collider == "obb" ||
                entry.collider == "mesh_sdf", "Invalid paint collider mode");
    if (input.contains("sdf_resolution_mode")) {
        require(input["sdf_resolution_mode"].is_number_integer(), "SDF mode must be integer");
        entry.sdf_resolution = input["sdf_resolution_mode"].get<int>();
    }
    require(entry.sdf_resolution >= 0 && entry.sdf_resolution <= 2, "SDF mode must be 0..2");
    return entry;
}
State readConfig(SceneData& scene, const json& input) {
    keys(input, {"version", "room", "objects"});
    require(!input.contains("version") ||
                (input["version"].is_number_integer() && input["version"] == 1), "Unsupported hidden object version");
    State result;
    result.scene = &scene;
    result.round = state().round;
    const auto room = input.value("room", json{{"mode", "auto"}});
    keys(room, {"mode", "min", "max", "placement", "wall_band", "camera_clearance"});
    require(!room.contains("placement") || room["placement"].is_string(),
            "Room placement must be walls_floor|free");
    result.placement = room.value("placement", std::string("walls_floor"));
    require(result.placement == "walls_floor" || result.placement == "free",
            "Room placement must be walls_floor|free");
    for (const char* key : {"wall_band", "camera_clearance"}) {
        require(!room.contains(key) || room[key].is_number(), std::string(key) + " must be numeric");
    }
    result.wall_band = room.value("wall_band", 0.65f);
    result.camera_clearance = room.value("camera_clearance", 3.0f);
    require(std::isfinite(result.wall_band) && result.wall_band >= 0.1f &&
                result.wall_band <= 2 && std::isfinite(result.camera_clearance) &&
                result.camera_clearance >= 0 && result.camera_clearance <= 10,
            "Wall band must be 0.1..2, camera clearance 0..10 metres");
    require(room.contains("mode") && room["mode"].is_string() &&
                (room["mode"] == "auto" || room["mode"] == "manual"), "Room mode must be auto|manual");
    result.manual_room = room["mode"] == "manual";
    if (result.manual_room) {
        require(room.contains("min") && room.contains("max"), "Manual room requires min/max");
        result.manual = {vector3(room["min"]), vector3(room["max"])};
        require(result.manual.low.x < result.manual.high.x &&
                    result.manual.low.y < result.manual.high.y &&
                    result.manual.low.z < result.manual.high.z, "Room min must precede max");
    }
    const auto objects = input.value("objects", json::array());
    require(objects.is_array() && objects.size() <= 64, "Hidden objects must be an array of at most 64");
    for (const auto& object : objects) {
        auto entry = readEntry(scene, object);
        require(result.entries.emplace(entry.object, entry).second, "Duplicate hidden object name");
    }
    return result;
}
json status(SceneData& scene) {
    discover(scene);
    json rows = json::array();
    int active = 0;
    for (const auto& item : state().entries) {
        auto row = entryJson(item.second);
        const auto* mesh = scene.getFlatNodeMesh(item.first);
        row["available"] = mesh != nullptr;
        row["visible"] = mesh && mesh->visible;
        if (mesh) {
            const auto box = objectBox(item.first);
            row["world_min"] = vectorJson(box.low);
            row["world_max"] = vectorJson(box.high);
        }
        active += mesh && item.second.visibility != "hidden";
        rows.push_back(std::move(row));
    }
    return {{"count", rows.size()}, {"active_count", active}, {"objects", rows},
            {"room", roomStatus(scene)}, {"round", state().round}, {"error", state().error}};
}
} // namespace

void clearHiddenGameObjects() {
    HiddenDetail::state() = HiddenDetail::State{};
    clearHiddenPlayerHistory();
}
json hiddenGameObjectsDesign() {
    auto* context = rtapi::boundContext();
    if (context) {
        discover(context->scene);
    }
    json rows = json::array();
    for (const auto& entry : state().entries) {
        rows.push_back(entryJson(entry.second));
    }
    json room = {{"mode", state().manual_room ? "manual" : "auto"}};
    room["placement"] = state().placement;
    room["wall_band"] = state().wall_band;
    room["camera_clearance"] = state().camera_clearance;
    if (state().manual_room) {
        room["min"] = vectorJson(state().manual.low);
        room["max"] = vectorJson(state().manual.high);
    }
    return {{"version", 1}, {"room", room}, {"objects", rows}};
}
bool validateHiddenGameObjects(SceneData& scene, const json& config, std::string& error) {
    try {
        const auto staged = readConfig(scene, config);
        for (const auto& entry : staged.entries) {
            preflightVisibility(scene, entry.second);
        }
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }
}
bool configureHiddenGameObjects(SceneData& scene, const json& config, std::string& error) {
    try {
        require(rtapi::renderStatus().state != rtapi::RenderJobState::Rendering,
                "Scene is locked by the final render job");
        State staged = readConfig(scene, config);
        for (const auto& entry : staged.entries) {
            preflightVisibility(scene, entry.second);
        }
        cancelBalloonInteraction();
        resetBalloonBodyInteraction();
        for (const auto& entry : staged.entries) {
            applyVisibility(scene, entry.second);
        }
        state() = std::move(staged);
        clearHiddenPlayerHistory();
        discover(scene);
        return true;
    } catch (const std::exception& exception) {
        error = state().error = exception.what();
        return false;
    }
}
bool hiddenObjectUsesInstancePose(const std::string& object) {
    return state().entries.count(object) != 0;
}
bool hiddenObjectBlocksBalloon(const std::string& object) {
    const auto found = state().entries.find(object);
    return found == state().entries.end() ||
           (found->second.visibility != "hidden" && found->second.blocks_balloon);
}
bool hiddenGameObjectsCommand(SceneData& scene, const json& input, json& output,
                              std::string& error) {
    try {
        require(input.is_object(), "Hidden operation requires an object");
        const auto operation = input.value("operation", std::string("status"));
        discover(scene);
        if (operation == "status") {
            keys(input, {"operation"});
        } else if (operation == "update") {
            auto patch = input;
            patch.erase("operation");
            const auto entry = readEntry(scene, patch);
            require(state().entries.count(entry.object) || state().entries.size() < 64,
                    "At most 64 hidden objects are supported");
            preflightVisibility(scene, entry);
            cancelBalloonInteraction();
            resetBalloonBodyInteraction();
            applyVisibility(scene, entry);
            state().entries[entry.object] = entry;
            clearHiddenPlayerHistory();
        } else if (operation == "bounds") {
            keys(input, {"operation", "mode", "min", "max", "placement", "wall_band",
                         "camera_clearance"});
            auto config = hiddenGameObjectsDesign();
            auto room = input;
            room.erase("operation");
            config["room"] = room;
            require(configureHiddenGameObjects(scene, config, error), error);
        } else if (operation == "configure") {
            keys(input, {"operation", "config"});
            require(input.contains("config"), "Missing hidden config");
            require(configureHiddenGameObjects(scene, input["config"], error), error);
        } else if (operation == "randomize") {
            keys(input, {"operation"});
            clearBalloonPaintInFlight(scene);
            require(resetHiddenGameObjects(scene, error), error);
            if (!wetPaintGpu().empty()) {
                wetPaintGpu().resetPaint();
            }
        } else if (operation == "paint_origin") {
            keys(input, {"operation", "position"});
            require(input.contains("position"), "Paint origin requires position");
            output = {{"allowed", HiddenPaintSpawnGuard(scene).allows(vector3(input["position"]))}};
            return true;
        } else if (operation == "player_anchor") {
            keys(input, {"operation", "owner", "position", "radius", "height"});
            require(input.contains("position") && input.contains("owner") &&
                        input["owner"].is_string(), "Anchor requires owner and position");
            const auto owner = input["owner"].get<std::string>();
            require(!owner.empty() && owner.size() <= 64, "Owner length must be 1..64");
            const auto desired = vector3(input["position"]);
            for (const char* key : {"radius", "height"}) {
                require(!input.contains(key) || input[key].is_number(), "Invalid capsule number");
            }
            const float radius = input.value("radius", 0.25f);
            const float height = input.value("height", 1.7f);
            require(std::isfinite(radius) && radius >= 0.05f && radius <= 2 &&
                        std::isfinite(height) && height >= 0.2f && height <= 5,
                    "Capsule radius/height out of range");
            output = {{"position", vectorJson(constrainHiddenPlayerAnchor(owner, desired,
                                                                            radius, height))}};
            return true;
        } else {
            throw std::invalid_argument("Unknown hidden object operation");
        }
        if (operation != "status") {
            state().error.clear();
        }
        output = status(scene);
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }
}
} // namespace Paint
