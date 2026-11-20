#include "Paint/BalloonInteraction.h"
#include "Paint/HiddenGameObjects.h"

#include "Api/RtApi.h"
#include "Camera.h"
#include "Matrix4x4.h"
#include "Ray.h"
#include "TriangleMesh.h"
#include "scene_data.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace Paint {
namespace {
using json = nlohmann::json;
struct Face {
    Vec3 a, b, c;
    Vec3 low, high;
};
struct Hold {
    int id = -1;
    std::string owner;
    float depth = 1.0f;
    float radius = 0.4f;
    float x = 0.5f, y = 0.5f;
    Vec3 offset;
    bool world_input = false;
    std::vector<Face> obstacles;
};
Hold hold;

void require(bool condition, const char* error) {
    if (!condition) {
        throw std::invalid_argument(error);
    }
}

Ray screenRay(SceneData& scene, float x, float y) {
    require(scene.camera != nullptr, "pointer input requires a scene camera");
    const auto ray = scene.camera->get_ray(x, 1.0f - y);
    return Ray(ray.origin, ray.direction.normalize());
}

Vec3 pointAtDepth(SceneData& scene, float x, float y, float depth) {
    const auto centre = screenRay(scene, 0.5f, 0.5f);
    const auto ray = screenRay(scene, x, y);
    const float cosine = Vec3::dot(ray.direction, centre.direction);
    require(cosine > 1e-4f, "pointer ray does not face the camera plane");
    const float origin_offset = Vec3::dot(ray.origin - centre.origin, centre.direction);
    return ray.origin + ray.direction * ((depth - origin_offset) / cosine);
}

// Closest point on a non-degenerate triangle (all data comes from flat DNA).
Vec3 closestPoint(const Vec3& p, const Face& f) {
    const Vec3 ab = f.b - f.a, ac = f.c - f.a, ap = p - f.a;
    const float d1 = Vec3::dot(ab, ap), d2 = Vec3::dot(ac, ap);
    if (d1 <= 0 && d2 <= 0) {
        return f.a;
    }
    const Vec3 bp = p - f.b;
    const float d3 = Vec3::dot(ab, bp), d4 = Vec3::dot(ac, bp);
    if (d3 >= 0 && d4 <= d3) {
        return f.b;
    }
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) {
        return f.a + ab * (d1 / (d1 - d3));
    }
    const Vec3 cp = p - f.c;
    const float d5 = Vec3::dot(ab, cp), d6 = Vec3::dot(ac, cp);
    if (d6 >= 0 && d5 <= d6) {
        return f.c;
    }
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) {
        return f.a + ac * (d2 / (d2 - d6));
    }
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
        return f.b + (f.c - f.b) * ((d4 - d3) / (d4 - d3 + d5 - d6));
    }
    const float inverse = 1.0f / (va + vb + vc);
    return f.a + ab * (vb * inverse) + ac * (vc * inverse);
}

std::vector<Face> obstacleSnapshot(SceneData& scene,
                                  const std::vector<std::string>& excluded) {
    (void)hiddenGameObjectsDesign();
    std::vector<Face> result;
    for (const auto& object : scene.world.objects) {
        const auto mesh = std::dynamic_pointer_cast<TriangleMesh>(object);
        if (!mesh || !mesh->geometry || !mesh->visible || mesh->transient ||
            mesh->hasSkinWeights() || !hiddenObjectBlocksBalloon(mesh->nodeName) ||
            std::find(excluded.begin(), excluded.end(), mesh->nodeName) != excluded.end()) {
            continue;
        }
        // Managed reveal objects follow their balloon obstacle policy. Static room
        // transforms are captured once per hold, never rebaked per pointer move.
        const auto& geometry = *mesh->geometry;
        const Vec3* positions = geometry.get_positions_orig();
        if (!positions) {
            positions = geometry.get_positions();
        }
        if (!positions) {
            continue;
        }
        Matrix4x4 world = Matrix4x4::identity();
        const auto transform = rtapi::getObjectWorldTransform(mesh->nodeName, world);
        require(transform.ok, "could not resolve carry obstacle transform");
        for (size_t i = 0; i + 2 < geometry.indices.size(); i += 3) {
            const auto a = geometry.indices[i], b = geometry.indices[i + 1];
            const auto c = geometry.indices[i + 2];
            if (a >= geometry.get_vertex_count() || b >= geometry.get_vertex_count() ||
                c >= geometry.get_vertex_count()) {
                continue;
            }
            Face face{world.transform_point(positions[a]), world.transform_point(positions[b]),
                      world.transform_point(positions[c])};
            face.low = Vec3(std::min({face.a.x, face.b.x, face.c.x}),
                            std::min({face.a.y, face.b.y, face.c.y}),
                            std::min({face.a.z, face.b.z, face.c.z}));
            face.high = Vec3(std::max({face.a.x, face.b.x, face.c.x}),
                             std::max({face.a.y, face.b.y, face.c.y}),
                             std::max({face.a.z, face.b.z, face.c.z}));
            if ((face.b - face.a).cross(face.c - face.a).length() > 1e-8f) {
                result.push_back(face);
            }
        }
    }
    return result;
}

// Conservative advancement: distance to a triangle is 1-Lipschitz, so no step
// crosses a wall even for a large wheel/input jump. A bounded iteration count
// returns the last safe point. Contact planes permit tangential sliding.
Vec3 constrain(const Vec3& start, const Vec3& desired, const Hold& active) {
    Vec3 position = start;
    Vec3 destination = desired;
    const float radius = active.radius + 0.01f;
    for (int slide = 0; slide < 3; ++slide) {
        const Vec3 delta = destination - position;
        const float length = delta.length();
        if (length < 1e-5f) {
            break;
        }
        float t = 0.0f;
        Vec3 normal(0.0f);
        for (int iteration = 0; iteration < 32 && t < 1.0f; ++iteration) {
            const Vec3 at = position + delta * t;
            float clearance = length;
            Vec3 nearest_normal(0.0f);
            for (const auto& face : active.obstacles) {
                const Vec3 box_point(std::clamp(at.x, face.low.x, face.high.x),
                                     std::clamp(at.y, face.low.y, face.high.y),
                                     std::clamp(at.z, face.low.z, face.high.z));
                const float reach = std::max(0.0f, clearance + radius);
                const Vec3 box_delta = at - box_point;
                if (Vec3::dot(box_delta, box_delta) > reach * reach) {
                    continue;
                }
                const Vec3 separation = at - closestPoint(at, face);
                const float distance = separation.length();
                const Vec3 n = distance > 1e-7f ? separation / distance : Vec3(0.0f);
                if (distance > 1e-7f && distance <= radius + 0.002f &&
                    Vec3::dot(delta, n) >= 0.0f) {
                    continue;
                }
                if (distance - radius < clearance) {
                    clearance = distance - radius;
                    nearest_normal = n;
                }
            }
            if (clearance <= 0.001f) {
                normal = nearest_normal;
                break;
            }
            t = std::min(1.0f, t + clearance * 0.9f / length);
        }
        position = position + delta * t;
        if (t >= 1.0f || normal.length() < 0.5f) {
            break;
        }
        Vec3 remaining = destination - position;
        remaining = remaining - normal * std::min(0.0f, Vec3::dot(remaining, normal));
        destination = position + remaining;
    }
    return position;
}
} // namespace

int pickBalloonTarget(SceneData& scene, float x, float y,
                      const std::vector<BalloonInteractionTarget>& targets) {
    const auto ray = screenRay(scene, x, y);
    float nearest = std::numeric_limits<float>::max();
    int id = -1;
    for (const auto& target : targets) {
        const Vec3 offset = target.position - ray.origin;
        const float along = Vec3::dot(offset, ray.direction);
        const float miss = (offset - ray.direction * along).length();
        if (along > 0 && along < nearest && miss <= target.radius) {
            nearest = along;
            id = target.id;
        }
    }
    return id;
}

BalloonInteractionResult balloonPointerCommand(
    SceneData& scene, const json& input, const std::string& mode,
    const std::vector<BalloonInteractionTarget>& targets,
    const std::vector<std::string>& visual_nodes) {
    require(input.is_object(), "pointer config must be an object");
    for (const auto& item : input.items()) {
        require(item.key() == "action" || item.key() == "screen" || item.key() == "wheel" ||
                    item.key() == "owner" || item.key() == "position" || item.key() == "balloon",
                "unknown pointer field");
    }
    require(input.contains("action") && input["action"].is_string(),
            "pointer action must be down|move|up|wheel|cancel");
    const auto action = input["action"].get<std::string>();
    require(action == "down" || action == "move" || action == "up" || action == "wheel" ||
                action == "cancel", "unknown pointer action");
    require(!input.contains("owner") || input["owner"].is_string(),
            "pointer owner must be a string");
    const auto owner = input.value("owner", std::string("mouse"));
    require(!owner.empty() && owner.size() <= 64, "pointer owner length must be 1..64");
    float x = hold.x, y = hold.y;
    const bool world = input.contains("position");
    require(!world || !input.contains("screen"), "use position or screen, not both");
    Vec3 world_position(0.0f);
    if (world) {
        const auto& position = input["position"];
        require(position.is_array() && position.size() == 3, "position must be world [x,y,z]");
        float values[3];
        for (int axis = 0; axis < 3; ++axis) {
            require(position[axis].is_number(), "position must contain numbers");
            values[axis] = position[axis].get<float>();
            require(std::isfinite(values[axis]) && std::abs(values[axis]) <= 10000,
                    "position must be finite within +/-10000 metres");
        }
        world_position = Vec3(values[0], values[1], values[2]);
    }
    require(!input.contains("balloon") || (world && action == "down" &&
                input["balloon"].is_number_integer() && input["balloon"].get<double>() >= 0 &&
                input["balloon"].get<double>() <= std::numeric_limits<int>::max()),
            "balloon requires world down and a nonnegative integer id");
    if (input.contains("screen")) {
        const auto& point = input["screen"];
        require(point.is_array() && point.size() == 2 && point[0].is_number() &&
                    point[1].is_number(), "screen must be [x,y] in 0..1");
        x = point[0].get<float>();
        y = point[1].get<float>();
        require(std::isfinite(x) && std::isfinite(y) && x >= 0 && x <= 1 && y >= 0 && y <= 1,
                "screen must be finite and in 0..1");
    }
    require(action != "down" || world || input.contains("screen"),
            "pointer down requires screen or world position");
    require(!input.contains("wheel") || action == "wheel", "wheel is only valid for wheel action");
    float wheel = 0;
    if (action == "wheel") {
        require(!world && (!hold.world_input || owner != hold.owner),
                "wheel is only supported for screen holds");
        require(input.contains("wheel") && input["wheel"].is_number(), "wheel requires a number");
        wheel = input["wheel"].get<float>();
        require(std::isfinite(wheel) && std::abs(wheel) <= 100,
                "wheel must be finite in -100..100");
    }
    BalloonInteractionResult result;
    if (hold.id >= 0 && owner != hold.owner) {
        return result;
    }
    if (action == "cancel") {
        cancelBalloonInteraction();
        return result;
    }
    if (action == "down") {
        if (hold.id >= 0) {
            return result;
        }
        int id = -1;
        if (world) {
            float nearest = std::numeric_limits<float>::max();
            for (const auto& target : targets) {
                if (input.contains("balloon") && input["balloon"].get<int>() != target.id) {
                    continue;
                }
                const float distance = (target.position - world_position).length();
                if (distance <= target.radius + 0.15f && distance < nearest) {
                    nearest = distance;
                    id = target.id;
                }
            }
        } else {
            id = pickBalloonTarget(scene, x, y, targets);
        }
        if (id < 0) {
            return result;
        }
        if (mode == "pop" && !world) {
            result.pop = id;
            return result;
        }
        require((world || mode == "carry") && !visual_nodes.empty(),
                "carry requires mesh balloon visuals");
        const auto target = std::find_if(targets.begin(), targets.end(),
            [id](const auto& value) { return value.id == id; });
        Hold next;
        next.id = id;
        next.owner = owner;
        next.radius = target->carry_radius;
        next.world_input = world;
        next.x = x;
        next.y = y;
        if (world) {
            next.offset = target->position - world_position;
            next.depth = 0.0f;
        } else {
            const auto centre = screenRay(scene, 0.5f, 0.5f);
            next.depth = std::clamp(Vec3::dot(target->position - centre.origin, centre.direction),
                                    next.radius + 0.2f, 50.0f);
            next.offset = target->position - pointAtDepth(scene, x, y, next.depth);
        }
        next.obstacles = obstacleSnapshot(scene, visual_nodes);
        hold = std::move(next);
        return result;
    }
    if (hold.id < 0) {
        return result;
    }
    const auto target = std::find_if(targets.begin(), targets.end(),
        [](const auto& value) { return value.id == hold.id; });
    if (target == targets.end()) {
        cancelBalloonInteraction();
        return result;
    }
    if (action == "up" && !world &&
        (!input.contains("screen") || (!hold.world_input && x == hold.x && y == hold.y))) {
        result.pop = hold.id;
        cancelBalloonInteraction();
        return result;
    }
    require(world == hold.world_input, "move/release must use the hold's coordinate space");
    // Compute everything before mutating the active hold (invalid input is atomic).
    const float depth = std::clamp(hold.depth + wheel * 0.35f, hold.radius + 0.2f, 50.0f);
    const Vec3 desired = (world ? world_position : pointAtDepth(scene, x, y, depth)) + hold.offset;
    result.move = hold.id;
    result.position = constrain(target->position, desired, hold);
    hold.depth = world ? 0.0f : depth;
    if (!world && (result.position - desired).length() > 0.001f) {
        const auto centre = screenRay(scene, 0.5f, 0.5f);
        hold.depth = std::clamp(Vec3::dot(result.position - hold.offset - centre.origin,
                                         centre.direction), hold.radius + 0.2f, 50.0f);
    }
    hold.x = x;
    hold.y = y;
    if (action == "up") {
        result.pop = hold.id;
        cancelBalloonInteraction();
    }
    return result;
}

json balloonInteractionStatus() {
    return {{"held", hold.id}, {"owner", hold.owner}, {"depth", hold.depth}};
}

bool balloonIsHeld(int id) {
    return hold.id == id;
}

int heldBalloon() {
    return hold.id;
}

void cancelBalloonInteraction() {
    hold = Hold{};
}
} // namespace Paint
