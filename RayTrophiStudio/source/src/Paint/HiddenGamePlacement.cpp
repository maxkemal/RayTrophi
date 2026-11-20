#include "HiddenGameObjectsInternal.h"
#include "Api/RtApi.h"
#include "UIContext.h"
#include "Paint/BalloonPaintDemo.h"
#include "Paint/BalloonInteraction.h"
#include "Paint/BalloonBodyInteraction.h"
#include "Paint/WetPaintGpu.h"
#include "Input/BodyPoseRuntime.h"
#include "TriangleMesh.h"
#include "scene_data.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <set>

namespace Paint::HiddenDetail {
bool roomNode(const std::string& name) {
    return name == "PaintBack" || name == "PaintLeft" || name == "PaintRight" || name == "PaintFloor" || name == "PaintCeiling" ||
           name.ends_with("Room_Wall") || name.ends_with("Room_Floor") ||
           name.ends_with("Room_Ceiling");
}
namespace {
Vec3 corner(const Box& box, int index) {
    return Vec3(index & 1 ? box.high.x : box.low.x, index & 2 ? box.high.y : box.low.y,
                index & 4 ? box.high.z : box.low.z);
}
void include(Box& box, const Vec3& point) {
    box.low = Vec3(std::min(box.low.x, point.x), std::min(box.low.y, point.y),
                   std::min(box.low.z, point.z));
    box.high = Vec3(std::max(box.high.x, point.x), std::max(box.high.y, point.y),
                    std::max(box.high.z, point.z));
}
Box transformed(const Box& local, const Matrix4x4& matrix) {
    const auto first = matrix.transform_point(corner(local, 0));
    Box result{first, first};
    for (int i = 1; i < 8; ++i) {
        include(result, matrix.transform_point(corner(local, i)));
    }
    return result;
}
bool inside(const Box& box, const Box& room, const std::vector<Plane>& planes) {
    constexpr float margin = 0.025f;
    if (box.low.x < room.low.x + margin || box.high.x > room.high.x - margin ||
        box.low.y < room.low.y + margin || box.high.y > room.high.y - margin ||
        box.low.z < room.low.z + margin || box.high.z > room.high.z - margin) {
        return false;
    }
    // Intersection of inward-facing room half-spaces is conservative, including
    // a curved shell: an AABB alone would let an object escape behind its arc.
    for (const auto& plane : planes) {
        for (int i = 0; i < 8; ++i) {
            if (Vec3::dot(corner(box, i) - plane.point, plane.inward) < margin) {
                return false;
            }
        }
    }
    return true;
}
} // namespace

bool roomGeometry(SceneData& scene, Box& room, std::vector<Plane>& planes,
                  std::set<std::string>* boundary_slabs) {
    bool found = false;
    std::set<std::string> names;
    for (const auto& object : scene.world.objects) {
        const auto mesh = std::dynamic_pointer_cast<TriangleMesh>(object);
        if (!mesh || !roomNode(mesh->nodeName) || !names.insert(mesh->nodeName).second) {
            continue;
        }
        const auto box = objectBox(mesh->nodeName);
        if (!found) {
            room = box;
            found = true;
        } else {
            include(room, box.low);
            include(room, box.high);
        }
    }
    if (state().manual_room) {
        room = state().manual;
    } else if (!found) {
        return false;
    }
    if (room.high.x - room.low.x <= 0.1f || room.high.y - room.low.y <= 0.1f ||
        room.high.z - room.low.z <= 0.1f) {
        return false;
    }
    // Imported room slabs often have generic names (e.g. Cube_1 / Cube_2).
    // Their AABB encloses the entire floor / ceiling, not a furnishing volume.
    // Start on the slab's TOP, not the wall canvas' lower edge: with a 3.3 cm
    // floor and 3 cm placement offset, every old candidate intersected it.
    const Box shell = room;
    const Vec3 room_size = shell.high - shell.low;
    std::set<std::string> slab_names;
    for (const auto& object : scene.world.objects) {
        const auto mesh = std::dynamic_pointer_cast<TriangleMesh>(object);
        if (!mesh || !mesh->visible || mesh->transient || mesh->hasSkinWeights() ||
            state().entries.count(mesh->nodeName)) {
            continue;
        }
        const auto box = objectBox(mesh->nodeName);
        const float overlap_x = std::max(0.0f, std::min(box.high.x, shell.high.x) -
                                              std::max(box.low.x, shell.low.x));
        const float overlap_z = std::max(0.0f, std::min(box.high.z, shell.high.z) -
                                              std::max(box.low.z, shell.low.z));
        if (overlap_x < room_size.x * 0.9f || overlap_z < room_size.z * 0.9f ||
            box.high.y - box.low.y > room_size.y * 0.15f) {
            continue;
        }
        const float band = room_size.y * 0.1f;
        if (box.low.y <= shell.low.y + 0.02f && box.high.y <= shell.low.y + band) {
            room.low.y = std::max(room.low.y, box.high.y);
            slab_names.insert(mesh->nodeName);
        } else if (box.high.y >= shell.high.y - 0.02f &&
                   box.low.y >= shell.high.y - band) {
            room.high.y = std::min(room.high.y, box.low.y);
            slab_names.insert(mesh->nodeName);
        }
    }
    if (boundary_slabs) {
        *boundary_slabs = std::move(slab_names);
    }
    if (room.high.y - room.low.y <= 0.1f) {
        return false;
    }
    const Vec3 reference = (room.low + room.high) * 0.5f;
    for (const auto& object : scene.world.objects) {
        const auto mesh = std::dynamic_pointer_cast<TriangleMesh>(object);
        if (!mesh || !mesh->geometry || !roomNode(mesh->nodeName)) {
            continue;
        }
        const auto& geometry = *mesh->geometry;
        const auto* positions = geometry.get_positions_orig();
        if (!positions) {
            continue;
        }
        Matrix4x4 world;
        const auto result = rtapi::getObjectWorldTransform(mesh->nodeName, world);
        require(result.ok, result.error);
        for (size_t i = 0; i + 2 < geometry.indices.size(); i += 3) {
            const auto a = geometry.indices[i], b = geometry.indices[i + 1];
            const auto c = geometry.indices[i + 2];
            if (a >= geometry.get_positions_orig_count() || b >= geometry.get_positions_orig_count() ||
                c >= geometry.get_positions_orig_count()) {
                continue;
            }
            const auto point = world.transform_point(positions[a]);
            Vec3 normal = (world.transform_point(positions[b]) - point).cross(
                          world.transform_point(positions[c]) - point);
            const float length = normal.length();
            if (length < 1e-8f) {
                continue;
            }
            normal = normal / length;
            const float side = Vec3::dot(reference - point, normal);
            if (std::abs(side) < 0.03f) {
                continue;
            }
            if (side < 0) {
                normal = normal * -1.0f;
            }
            planes.push_back({point, normal});
        }
    }
    return true;
}
json roomStatus(SceneData& scene) {
    Box box;
    std::vector<Plane> planes;
    std::set<std::string> slabs;
    const bool valid = roomGeometry(scene, box, planes, &slabs);
    return {{"mode", state().manual_room ? "manual" : "auto"}, {"valid", valid},
            {"min", valid ? vectorJson(box.low) : json()},
            {"max", valid ? vectorJson(box.high) : json()}, {"wall_planes", planes.size()},
            {"boundary_slabs", std::vector<std::string>(slabs.begin(), slabs.end())},
            {"placement", state().placement}, {"wall_band", state().wall_band},
            {"camera_clearance", state().camera_clearance}};
}
bool planHiddenPlacements(SceneData& scene,
                          std::vector<std::pair<std::string, Matrix4x4>>& placements,
                          std::string& error) {
    try {
        bool needed = false;
        for (const auto& entry : state().entries) {
            needed = needed || (entry.second.randomize && entry.second.visibility != "hidden");
        }
        if (!needed) {
            return true;
        }
        Box room;
        std::vector<Plane> planes;
        std::set<std::string> boundary_slabs;
        require(roomGeometry(scene, room, planes, &boundary_slabs),
                "Set valid room bounds before random placement");
        std::set<std::string> balloons;
        const auto demo = balloonPaintDesign();
        if (demo.is_object()) {
            for (const auto& node : demo.value("visual_nodes", json::array())) {
                balloons.insert(node.get<std::string>());
            }
        }
        std::vector<Box> occupied;
        std::set<std::string> seen;
        for (const auto& object : scene.world.objects) {
            const auto mesh = std::dynamic_pointer_cast<TriangleMesh>(object);
            if (!mesh || !mesh->visible || mesh->transient || mesh->hasSkinWeights() ||
                roomNode(mesh->nodeName) || boundary_slabs.count(mesh->nodeName) ||
                balloons.count(mesh->nodeName) ||
                !seen.insert(mesh->nodeName).second) {
                continue;
            }
            const auto managed = state().entries.find(mesh->nodeName);
            if (managed != state().entries.end() &&
                (managed->second.visibility == "hidden" || managed->second.randomize)) {
                continue;
            }
            occupied.push_back(objectBox(mesh->nodeName));
        }
        std::mt19937 random(static_cast<uint32_t>(417u + state().round));
        for (const auto& item : state().entries) {
            const auto& entry = item.second;
            if (!entry.randomize || entry.visibility == "hidden") {
                continue;
            }
            rtapi::ObjectInfo info;
            const auto read = rtapi::getObjectInfo(entry.object, info);
            require(read.ok && info.has_bounds, "Missing object bounds: " + entry.object);
            Matrix4x4 authored, world;
            require(rtapi::getObjectTransform(entry.object, authored).ok &&
                        rtapi::getObjectWorldTransform(entry.object, world).ok,
                    "Cannot resolve static pose: " + entry.object);
            for (int row = 0; row < 4; ++row) {
                for (int column = 0; column < 4; ++column) {
                    require(std::abs(authored.m[row][column] - world.m[row][column]) < 1e-4f,
                            "Random placement requires an unsimulated static pose: " + entry.object);
                }
            }
            const Box local{info.local_min, info.local_max};
            bool placed = false;
            for (int attempt = 0; attempt < 64 && !placed; ++attempt) {
                const float angle = std::uniform_real_distribution<float>(entry.yaw_min,
                    entry.yaw_max)(random) * 0.01745329252f;
                auto matrix = Matrix4x4::rotationY(angle) * entry.base;
                const auto origin_box = transformed(local, matrix);
                const auto extent = (origin_box.high - origin_box.low) * 0.5f;
                const auto low = room.low + extent + Vec3(0.03f);
                const auto high = room.high - extent - Vec3(0.03f);
                if (low.x > high.x || low.y > high.y || low.z > high.z) {
                    continue;
                }
                Vec3 center(std::uniform_real_distribution<float>(low.x, high.x)(random),
                                  std::uniform_real_distribution<float>(low.y, high.y)(random),
                                  std::uniform_real_distribution<float>(low.z, high.z)(random));
                if (state().placement == "walls_floor") {
                    center.y = low.y;
                    const int wall = std::uniform_int_distribution<int>(0, 3)(random);
                    const float band = std::uniform_real_distribution<float>(0,
                        state().wall_band)(random);
                    if (wall == 0) {
                        center.x = std::min(high.x, low.x + band);
                    } else if (wall == 1) {
                        center.x = std::max(low.x, high.x - band);
                    } else if (wall == 2) {
                        center.z = std::min(high.z, low.z + band);
                    } else {
                        center.z = std::max(low.z, high.z - band);
                    }
                }
                const auto offset = center - (origin_box.low + origin_box.high) * 0.5f;
                matrix.m[0][3] += offset.x;
                matrix.m[1][3] += offset.y;
                matrix.m[2][3] += offset.z;
                const auto box = transformed(local, matrix);
                if (!inside(box, room, planes) ||
                    std::any_of(occupied.begin(), occupied.end(), [&](const auto& other) {
                        return overlaps(box, other);
                    })) {
                    continue;
                }
                auto* context = rtapi::boundContext();
                if (context && context->scene.camera && state().camera_clearance > 0) {
                    rtapi::CameraState camera;
                    const auto read_camera = rtapi::getCamera(camera);
                    require(read_camera.ok, read_camera.error);
                    const Vec3 nearest(std::clamp(camera.position.x, box.low.x, box.high.x),
                                       std::clamp(camera.position.y, box.low.y, box.high.y),
                                       std::clamp(camera.position.z, box.low.z, box.high.z));
                    if ((nearest - camera.position).length() < state().camera_clearance) {
                        continue;
                    }
                }
                bool balloon_overlap = false;
                if (entry.blocks_balloon) {
                    for (const auto& name : balloons) {
                        if (scene.getFlatNodeMesh(name)) {
                            balloon_overlap = balloon_overlap || overlaps(box, objectBox(name));
                        }
                    }
                }
                bool player_overlap = false;
                if (entry.blocks_player) {
                    for (const auto& person : Input::bodyPoses().persons()) {
                        const auto& pelvis = person.second.joints.at("pelvis");
                        const auto& head = person.second.joints.at("head");
                        if (person.second.confidence < 0.5 || pelvis[3] < 0.5) {
                            continue;
                        }
                        const Vec3 base(static_cast<float>(pelvis[0]), 0,
                                        static_cast<float>(pelvis[2]));
                        const Box actor{base - Vec3(0.3f, 0, 0.3f),
                                        base + Vec3(0.3f, static_cast<float>(head[1]) + 0.2f, 0.3f)};
                        player_overlap = player_overlap || overlaps(box, actor);
                    }
                }
                if (player_overlap || balloon_overlap) {
                    continue;
                }
                placements.emplace_back(entry.object, matrix);
                occupied.push_back(box);
                placed = true;
            }
            require(placed, "No safe placement found within 64 attempts: " + entry.object);
        }
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        placements.clear();
        return false;
    }
}
} // namespace Paint::HiddenDetail

namespace Paint {
bool resetHiddenGameObjects(SceneData& scene, std::string& error) {
    using namespace HiddenDetail;
    discover(scene);
    if (rtapi::renderStatus().state == rtapi::RenderJobState::Rendering) {
        error = state().error = "Scene is locked by the final render job";
        return false;
    }
    if (state().entries.empty()) {
        return true;
    }
    std::vector<std::pair<std::string, Matrix4x4>> placements;
    if (!planHiddenPlacements(scene, placements, error)) {
        state().error = error;
        return false;
    }
    std::string waiting;
    if (!placements.empty() && !balloonPaintCollidersReady(scene, waiting)) {
        error = state().error = "Wait for paint colliders: " + waiting;
        return false;
    }
    std::vector<std::pair<std::string, Matrix4x4>> previous;
    try {
        for (const auto& placement : placements) {
            Matrix4x4 matrix;
            const auto read = rtapi::getObjectTransform(placement.first, matrix);
            require(read.ok, read.error);
            previous.emplace_back(placement.first, matrix);
        }
        waitHiddenResetWork(scene);
        cancelBalloonInteraction();
        resetBalloonBodyInteraction();
        for (const auto& placement : placements) {
            require(moveHiddenObject(scene, placement.first, placement.second, error), error);
        }
        scene.preserveScriptSimulationPreview();
        wetPaintGpu().checkGeometryNextTick();
        clearHiddenPlayerHistory();
        ++state().round;
        state().error.clear();
        return true;
    } catch (const std::exception& exception) {
        for (const auto& placement : previous) {
            std::string ignored;
            (void)moveHiddenObject(scene, placement.first, placement.second, ignored);
        }
        scene.preserveScriptSimulationPreview();
        wetPaintGpu().checkGeometryNextTick();
        error = state().error = exception.what();
        return false;
    }
}
} // namespace Paint
