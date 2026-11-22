#include "Paint/BalloonInteraction.h"
#include "Paint/BalloonRoomFront.h"
#include "Paint/HiddenGameObjects.h"
#include "Paint/WetPaintGpu.h"

#include "AABB.h"
#include "Api/RtApi.h"
#include "Camera.h"
#include "Matrix4x4.h"
#include "Ray.h"
#include "Transform.h"
#include "TriangleMesh.h"
#include "scene_data.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace Paint {
namespace {
using json = nlohmann::json;
struct Face {
    Vec3 a, b, c;
    Vec3 low, high;
    bool compresses = true;
};

Face obstacleFace(const Vec3& a, const Vec3& b, const Vec3& c) {
    return {a, b, c, Vec3::min(a, Vec3::min(b, c)), Vec3::max(a, Vec3::max(b, c))};
}
struct Hold {
    int id = -1;
    std::string owner;
    float depth = 1.0f;
    float radius = 0.4f;
    float x = 0.5f, y = 0.5f;
    Vec3 offset;
    bool world_input = false;
    Vec3 wall_normal = Vec3(0.0f);
    float compression = 0.0f;
    float visual_axial = 1.0f;
    float visual_lateral = 1.0f;
    float visual_offset = 0.0f;
    Vec3 visual_normal = Vec3(0.0f);
    float contact_support = 0.0f;
    float crush_seconds = 0.0f;
    bool crushing = false;
    std::vector<Face> obstacles;
    std::unordered_map<const TriangleMesh*, AABB> local_visual_bounds;
};
Hold hold;
int released_compressed_balloon = -1;

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
            const auto face = obstacleFace(world.transform_point(positions[a]),
                                           world.transform_point(positions[b]),
                                           world.transform_point(positions[c]));
            if ((face.b - face.a).cross(face.c - face.a).length() > 1e-8f) {
                result.push_back(face);
            }
        }
    }
    return result;
}

struct RoomSweep {
    float fraction = 1.0f;
    Vec3 normal = Vec3(0.0f);
};

// Work in ellipsoid space so each wall uses the balloon's extent along its
// actual contact direction. The room mesh stays in world space.
Vec3 roomScale(const Vec3& radii) {
    constexpr float skin = 0.002f;
    return Vec3(1.0f / (radii.x + skin), 1.0f / (radii.y + skin),
                1.0f / (radii.z + skin));
}

float roomBoxDistanceSquared(const Face& face, const Vec3& center, const Vec3& scale) {
    const Vec3 nearest(std::clamp(center.x, face.low.x, face.high.x),
                       std::clamp(center.y, face.low.y, face.high.y),
                       std::clamp(center.z, face.low.z, face.high.z));
    return ((center - nearest) * scale).length_squared();
}

Vec3 roomSeparation(const Face& face, const Vec3& center, const Vec3& scale) {
    const Face scaled{(face.a - center) * scale, (face.b - center) * scale,
                      (face.c - center) * scale, {}, {}};
    return -closestPoint(Vec3(0.0f), scaled);
}

// Distance to the scaled triangle is 1-Lipschitz. Stop at the last safe point
// even when the pointer jumps across the room in one event.
RoomSweep sweepRoom(const Vec3& position, const Vec3& delta, const Vec3& scale,
                    const std::vector<Face>& obstacles) {
    const Vec3 scaled_delta = delta * scale;
    const float length = scaled_delta.length();
    if (length < 1e-5f) {
        return {};
    }

    constexpr float contact_slop = 0.002f;
    float t = 0.0f;
    for (int iteration = 0; iteration < 32 && t < 1.0f; ++iteration) {
        const Vec3 at = position + delta * t;
        float clearance = length;
        Vec3 nearest_normal(0.0f);
        for (const auto& face : obstacles) {
            const float reach = clearance + 1.0f;
            if (roomBoxDistanceSquared(face, at, scale) > reach * reach) {
                continue;
            }
            const Vec3 separation = roomSeparation(face, at, scale);
            const float distance = separation.length();
            const Vec3 normal = distance > 1e-7f ? separation / distance : Vec3(0.0f);
            // The closest point defines a separating plane for the triangle.
            const float closing = -Vec3::dot(scaled_delta, normal) * (1.0f - t);
            if (distance > 1e-7f && closing <= std::max(0.0f, distance - 1.0f - contact_slop)) {
                continue;
            }
            if (distance - 1.0f < clearance) {
                clearance = distance - 1.0f;
                nearest_normal = normal;
            }
        }
        if (clearance <= contact_slop) {
            return {t, (nearest_normal * scale).normalize()};
        }
        t = std::min(1.0f, t + clearance * 0.9f / length);
    }
    return {t, Vec3(0.0f)};
}

Vec3 roomPressure(const Vec3& center, const Vec3& scale, const Vec3& blocked,
                  const std::vector<Face>& obstacles) {
    constexpr float contact_band = 1.025f;
    std::array<Vec3, 3> normals;
    size_t count = 0;
    Vec3 pressure(0.0f);
    for (const auto& face : obstacles) {
        if (!face.compresses ||
            roomBoxDistanceSquared(face, center, scale) > contact_band * contact_band) {
            continue;
        }
        const Vec3 separation = roomSeparation(face, center, scale);
        const float distance_squared = separation.length_squared();
        if (distance_squared <= 1e-12f || distance_squared > contact_band * contact_band) {
            continue;
        }
        const Vec3 normal = (separation * scale).normalize();
        const float push = -Vec3::dot(blocked, normal);
        if (push <= 0.0f) {
            continue;
        }
        // Tessellation must not give one wall more influence at an edge.
        const auto end = normals.begin() + count;
        if (std::any_of(normals.begin(), end, [&](const Vec3& other) {
                return Vec3::dot(normal, other) >= 0.94f;
            })) {
            continue;
        }
        normals[count++] = normal;
        pressure = pressure + normal * push;
        if (count == normals.size()) {
            break;
        }
    }
    return pressure;
}

float sphereContact(const Vec3& offset, const Vec3& delta, float radius) {
    const float approach = -Vec3::dot(offset, delta);
    if (approach <= 0.0f) {
        return 1.0f;
    }
    const float gap = offset.length_squared() - radius * radius;
    if (gap <= 0.0f) {
        return 0.0f;
    }
    const float discriminant = approach * approach - delta.length_squared() * gap;
    if (discriminant <= 0.0f) {
        return 1.0f;
    }
    return std::min(1.0f, gap / (approach + std::sqrt(discriminant)));
}

struct CarryBody {
    Vec3 center;
    Vec3 radii;
    Vec3 displacement = Vec3(0.0f);
    float pressure_radius = 0.0f;
    bool advancing = false;
    bool hit_room = false;
};

void scaleContactAxes(Matrix4x4& matrix, const Vec3& normal, float axial, float lateral) {
    for (int column = 0; column < 3; ++column) {
        const Vec3 axis(matrix.m[0][column], matrix.m[1][column], matrix.m[2][column]);
        const Vec3 scaled = axis * lateral +
                            normal * (Vec3::dot(axis, normal) * (axial - lateral));
        for (int row = 0; row < 3; ++row) {
            matrix.m[row][column] = scaled[row];
        }
    }
}

CarryBody carryBody(const BalloonInteractionTarget& target, Hold& active) {
    CarryBody body{target.position, Vec3(std::max(target.billboard_radius, 0.001f))};
    body.pressure_radius = target.carry_radius;
    const TriangleMesh* mesh = target.visual_mesh;
    if (!mesh || !mesh->geometry) {
        return body;
    }
    auto [entry, inserted] = active.local_visual_bounds.try_emplace(mesh);
    AABB& bounds = entry->second;
    if (inserted) {
        const auto& geometry = *mesh->geometry;
        const Vec3* positions = geometry.get_positions_orig();
        if (!positions) {
            positions = geometry.get_positions();
        }
        if (positions) {
            for (size_t i = 0; i < geometry.get_vertex_count(); ++i) {
                for (int axis = 0; axis < 3; ++axis) {
                    bounds.min[axis] = std::min(bounds.min[axis], positions[i][axis]);
                    bounds.max[axis] = std::max(bounds.max[axis], positions[i][axis]);
                }
            }
        }
    }
    const Vec3 local_radii = (bounds.max - bounds.min) * 0.5f;
    if (!(local_radii.x > 0.0f && local_radii.y > 0.0f && local_radii.z > 0.0f)) {
        return body;
    }
    Matrix4x4 matrix = mesh->transform ? mesh->transform->getFinal()
                                       : Matrix4x4::identity();
    if (target.id == active.id &&
        (active.visual_axial != 1.0f || active.visual_lateral != 1.0f)) {
        // The displayed squash is cosmetic. Sweep the uncompressed shape.
        scaleContactAxes(matrix, active.visual_normal, 1.0f / active.visual_axial,
                         1.0f / active.visual_lateral);
        for (int axis = 0; axis < 3; ++axis) {
            matrix.m[axis][3] += active.visual_normal[axis] * active.visual_offset;
        }
    }
    body.center = matrix.transform_point((bounds.min + bounds.max) * 0.5f);
    const Vec3 x = matrix.transform_vector(Vec3(local_radii.x, 0.0f, 0.0f));
    const Vec3 y = matrix.transform_vector(Vec3(0.0f, local_radii.y, 0.0f));
    const Vec3 z = matrix.transform_vector(Vec3(0.0f, 0.0f, local_radii.z));
    for (int axis = 0; axis < 3; ++axis) {
        const float radius = std::sqrt(x[axis] * x[axis] + y[axis] * y[axis] +
                                       z[axis] * z[axis]);
        body.radii[axis] = std::max(radius, 0.001f);
    }
    return body;
}

Vec3 scaleByInverseRadii(const Vec3& v, const Vec3& radii) {
    return Vec3(v.x / radii.x, v.y / radii.y, v.z / radii.z);
}

void advanceCarryBody(size_t index, Vec3 remaining, std::vector<CarryBody>& bodies,
                      const std::vector<Face>& obstacles, int& budget) {
    CarryBody& body = bodies[index];
    body.advancing = true;
    const Vec3 room_scale = roomScale(body.radii);
    int slides = 0;
    while (budget > 0 && slides < 3 && remaining.length_squared() > 1e-10f) {
        --budget;
        float fraction = 1.0f;
        size_t contact = bodies.size();
        Vec3 normal(0.0f);
        for (size_t j = 0; j < bodies.size(); ++j) {
            if (j == index) {
                continue;
            }
            const Vec3 radii = body.radii + bodies[j].radii;
            const Vec3 offset = scaleByInverseRadii(body.center - bodies[j].center, radii);
            const Vec3 delta = scaleByInverseRadii(remaining, radii);
            const float hit = sphereContact(offset, delta, 1.0f);
            if (hit < fraction) {
                fraction = hit;
                contact = j;
                normal = scaleByInverseRadii(offset + delta * hit, radii).normalize();
            }
        }

        const Vec3 travel = remaining * fraction;
        const RoomSweep room = sweepRoom(body.center, travel, room_scale, obstacles);
        const Vec3 advance = travel * room.fraction;
        body.center = body.center + advance;
        body.displacement = body.displacement + advance;
        remaining = remaining - advance;

        const bool hit_wall = room.fraction < 1.0f;
        if (hit_wall) {
            normal = room.normal;
            body.hit_room = true;
        } else if (contact == bodies.size()) {
            break;
        } else if (!bodies[contact].advancing) {
            // Transfer only motion into the contact. Recheck after a small
            // push so a glancing hit can separate instead of dragging a chain.
            const float into = -Vec3::dot(remaining, normal);
            if (into > 1e-5f) {
                const Vec3 before = bodies[contact].center;
                advanceCarryBody(contact, -normal * std::min(into, 0.08f),
                                 bodies, obstacles, budget);
                if (Vec3::dot(before - bodies[contact].center, normal) > 1e-5f) {
                    continue;
                }
            }
        }

        if (normal.length_squared() < 0.25f) {
            break;
        }
        const float inward = std::max(0.0f, -Vec3::dot(remaining, normal));
        remaining = remaining + normal * inward;
        if (hit_wall && inward > 0.0f) {
            const float tangent = remaining.length();
            const float drag = std::min(0.12f, 0.06f * inward / std::max(tangent, 1e-5f));
            remaining = remaining * (1.0f - drag);
        }
        ++slides;
    }
    body.advancing = false;
}

void moveWithBalloonContacts(const Vec3& desired, Hold& active,
                             const std::vector<BalloonInteractionTarget>& targets,
                             BalloonInteractionResult& result) {
    std::vector<CarryBody> bodies;
    bodies.reserve(targets.size());
    size_t held = 0;
    for (size_t i = 0; i < targets.size(); ++i) {
        if (targets[i].id == active.id) {
            held = i;
        }
        bodies.push_back(carryBody(targets[i], active));
    }

    int budget = 128;
    const Vec3 requested = desired - targets[held].position;
    advanceCarryBody(held, requested, bodies, active.obstacles, budget);

    const auto& carrier = bodies[held];
    Vec3 wall_force(0.0f);
    if (carrier.hit_room) {
        wall_force = roomPressure(carrier.center, roomScale(carrier.radii),
                                  requested - carrier.displacement, active.obstacles);
    }
    const float pressure = wall_force.length() / carrier.pressure_radius;
    const Vec3 normal = wall_force.normalize();
    const Vec3 slide = carrier.displacement -
                       normal * Vec3::dot(carrier.displacement, normal);
    const bool crushing = pressure >= 0.65f && slide.length_squared() < 0.0025f;
    if (!crushing || !active.crushing ||
        Vec3::dot(active.wall_normal, normal) < 0.9f) {
        active.crush_seconds = 0.0f;
    }
    active.compression = std::clamp(pressure * 0.55f, 0.0f, 0.30f);
    if (active.compression > 0.0f) {
        active.wall_normal = normal;
        active.contact_support = (carrier.radii * normal).length();
    }
    active.crushing = crushing;
    result.position = targets[held].position + carrier.displacement;
    for (size_t i = 0; i < bodies.size(); ++i) {
        if (i != held && bodies[i].displacement.length_squared() > 1e-10f) {
            result.pushed.push_back({targets[i].id, targets[i].position + bodies[i].displacement});
        }
    }
}
} // namespace

int pickBalloonTarget(SceneData& scene, float x, float y,
                      const std::vector<BalloonInteractionTarget>& targets) {
    const auto ray = screenRay(scene, x, y);
    const Vec3 forward = screenRay(scene, 0.5f, 0.5f).direction;
    const float ray_forward = Vec3::dot(ray.direction, forward);
    float nearest_hit = std::numeric_limits<float>::max();
    float nearest_miss = std::numeric_limits<float>::max();
    float fallback_depth = std::numeric_limits<float>::max();
    int hit_id = -1;
    int fallback_id = -1;
    for (const auto& target : targets) {
        const Vec3 offset = target.position - ray.origin;
        if (target.visual_mesh) {
            HitRecord hit;
            if (target.visual_mesh->hit(ray, 0.0f, nearest_hit, hit, true)) {
                nearest_hit = hit.t;
                hit_id = target.id;
            }
        } else if (ray_forward > 0.0f) {
            const float depth = Vec3::dot(offset, forward) / ray_forward;
            const float miss_squared = (offset - ray.direction * depth).length_squared();
            if (depth > 0.0f && depth < nearest_hit &&
                miss_squared <= target.billboard_radius * target.billboard_radius) {
                nearest_hit = depth;
                hit_id = target.id;
            }
        }

        const float along = Vec3::dot(offset, ray.direction);
        const float miss_squared = (offset - ray.direction * along).length_squared();
        if (along > 0.0f && miss_squared <= target.radius * target.radius &&
            (miss_squared < nearest_miss ||
             (miss_squared == nearest_miss && along < fallback_depth))) {
            nearest_miss = miss_squared;
            fallback_depth = along;
            fallback_id = target.id;
        }
    }
    return hit_id >= 0 ? hit_id : fallback_id;
}

BalloonInteractionResult balloonPointerCommand(
    SceneData& scene, const json& input, const std::string& mode,
    const std::vector<BalloonInteractionTarget>& targets,
    const std::vector<std::string>& visual_nodes, const std::vector<Vec3>* barrier) {
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
        if (barrier) {
            for (size_t i = 0; i + 2 < barrier->size(); i += 3) {
                auto face = obstacleFace((*barrier)[i], (*barrier)[i + 1], (*barrier)[i + 2]);
                face.compresses = false;
                next.obstacles.push_back(face);
            }
        }
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
    moveWithBalloonContacts(desired, hold, targets, result);
    hold.depth = world ? 0.0f : depth;
    if (!world && wheel != 0.0f) {
        // Wheel pulses can show compression, but only a held pointer goal
        // supplies the sustained pressure needed to burst the balloon.
        hold.crushing = false;
        hold.crush_seconds = 0.0f;
        if ((result.position - desired).length() > 0.001f) {
            const auto centre = screenRay(scene, 0.5f, 0.5f);
            hold.depth = std::clamp(Vec3::dot(result.position - hold.offset - centre.origin,
                                             centre.direction), hold.radius + 0.2f, 50.0f);
        }
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
    return {{"held", hold.id}, {"owner", hold.owner}, {"depth", hold.depth},
            {"compression", hold.compression}};
}

bool balloonIsHeld(int id) {
    return hold.id == id;
}

int heldBalloon() {
    return hold.id;
}

std::vector<Vec3> balloonRoomTriangles(SceneData& scene,
                                       const std::vector<std::string>& visual_nodes) {
    const auto faces = obstacleSnapshot(scene, visual_nodes);
    std::vector<Vec3> vertices;
    vertices.reserve(faces.size() * 3);
    for (const auto& face : faces) {
        vertices.insert(vertices.end(), {face.a, face.b, face.c});
    }
    return vertices;
}

BalloonRoomFront balloonPaintRoomFront(SceneData& scene,
                                       const std::vector<std::string>& surfaces) {
    std::vector<AABB> bounds;
    std::vector<std::string> seen;
    const auto add = [&](const std::string& name) {
        if (std::find(seen.begin(), seen.end(), name) != seen.end()) {
            return;
        }
        seen.push_back(name);
        const auto* mesh = scene.getFlatNodeMesh(name);
        AABB box;
        if (mesh && mesh->bounding_box(0.0f, 0.0f, box)) {
            bounds.push_back(box);
        }
    };
    if (surfaces.empty()) {
        for (const auto& surface : wetPaintGpu().surfaces()) {
            add(surface.object);
        }
    } else {
        for (const auto& name : surfaces) {
            add(name);
        }
    }
    return balloonRoomFront(bounds);
}

void cancelBalloonInteraction() {
    if (hold.visual_axial != 1.0f || hold.visual_lateral != 1.0f) {
        released_compressed_balloon = hold.id;
    }
    hold = Hold{};
}

float balloonCarryCompression() {
    return hold.compression;
}

void applyBalloonCarryCompression(int id, Matrix4x4& transform, float axial, float lateral) {
    if (id == hold.id) {
        if (axial <= 0.0f) {
            axial = 1.0f - hold.compression;
            lateral = 1.0f + 0.18f * hold.compression;
        }
        hold.visual_axial = axial;
        hold.visual_lateral = lateral;
        hold.visual_normal = hold.wall_normal;
        hold.visual_offset = hold.contact_support * (1.0f - axial);
        if (axial == 1.0f && lateral == 1.0f) {
            return;
        }
        scaleContactAxes(transform, hold.visual_normal, axial, lateral);
        for (int axis = 0; axis < 3; ++axis) {
            transform.m[axis][3] -= hold.visual_normal[axis] * hold.visual_offset;
        }
    }
}

int takeReleasedCompressedBalloon() {
    return std::exchange(released_compressed_balloon, -1);
}

int updateBalloonCrush(float dt) {
    if (!hold.crushing) {
        return -1;
    }
    hold.crush_seconds += std::min(dt, 0.05f);
    return hold.crush_seconds >= 0.28f ? hold.id : -1;
}
} // namespace Paint
