#include "HiddenGameObjectsInternal.h"
#include "Input/BodyPoseRuntime.h"
#include "TriangleMesh.h"
#include "scene_data.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace Paint {
namespace {
std::map<std::string, Vec3> history;
using HiddenDetail::Box;

bool contains(const Box& box, const Vec3& p) {
    return p.x > box.low.x && p.x < box.high.x &&
           p.z > box.low.z && p.z < box.high.z;
}
Vec3 eject(const Box& box, Vec3 p) {
    if (!contains(box, p)) {
        return p;
    }
    const float distances[] = {p.x - box.low.x, box.high.x - p.x,
                               p.z - box.low.z, box.high.z - p.z};
    const auto side = std::min_element(distances, distances + 4) - distances;
    if (side == 0) {
        p.x = box.low.x;
    } else if (side == 1) {
        p.x = box.high.x;
    } else if (side == 2) {
        p.z = box.low.z;
    } else {
        p.z = box.high.z;
    }
    return p;
}
// Slab sweep against expanded XZ boxes; no sampling step can tunnel through a box.
float contact(const Box& box, const Vec3& start, const Vec3& delta) {
    float enter = 0, leave = 1;
    for (int axis : {0, 2}) {
        const float p = axis == 0 ? start.x : start.z;
        const float d = axis == 0 ? delta.x : delta.z;
        const float low = axis == 0 ? box.low.x : box.low.z;
        const float high = axis == 0 ? box.high.x : box.high.z;
        if (std::abs(d) < 1e-8f) {
            if (p <= low || p >= high) {
                return 1;
            }
            continue;
        }
        float a = (low - p) / d, b = (high - p) / d;
        if (a > b) {
            std::swap(a, b);
        }
        enter = std::max(enter, a);
        leave = std::min(leave, b);
        if (enter >= leave) {
            return 1;
        }
    }
    return std::clamp(enter, 0.0f, 1.0f);
}
Vec3 sweep(const std::vector<Box>& boxes, const Vec3& start, const Vec3& desired) {
    float fraction = 1;
    const Vec3 delta = desired - start;
    for (const auto& box : boxes) {
        fraction = std::min(fraction, contact(box, start, delta));
    }
    if (fraction < 1) {
        fraction = std::max(0.0f, fraction - 0.001f / std::max(0.001f, delta.length()));
    }
    return start + delta * fraction;
}
} // namespace
void clearHiddenPlayerHistory() {
    history.clear();
}
Vec3 constrainHiddenPlayerAnchor(const std::string& owner, const Vec3& desired,
                                 float radius, float height) {
    auto& state = HiddenDetail::state();
    std::vector<Box> boxes;
    if (state.scene) {
        for (const auto& item : state.entries) {
            const auto& entry = item.second;
            const auto* mesh = state.scene->getFlatNodeMesh(item.first);
            if (!entry.blocks_player || entry.visibility == "hidden" || !mesh || !mesh->visible) {
                continue;
            }
            auto box = HiddenDetail::objectBox(item.first);
            if (box.high.y < desired.y || box.low.y > desired.y + height) {
                continue;
            }
            box.low.x -= radius + 0.01f;
            box.low.z -= radius + 0.01f;
            box.high.x += radius + 0.01f;
            box.high.z += radius + 0.01f;
            boxes.push_back(box);
        }
    }
    if (boxes.empty()) {
        history.erase(owner);
        return desired;
    }
    auto found = history.find(owner);
    Vec3 accepted = found == history.end() ? desired : found->second;
    accepted.y = desired.y;
    // Resolve a first observation or a newly enabled obstacle without retaining penetration.
    for (int pass = 0; pass < 8; ++pass) {
        for (const auto& box : boxes) {
            accepted = eject(box, accepted);
        }
    }
    const auto start = accepted;
    accepted = sweep(boxes, accepted, desired);
    // Two bounded axis slides let a player move along an obstacle after contact.
    accepted = sweep(boxes, accepted, Vec3(desired.x, desired.y, accepted.z));
    accepted = sweep(boxes, accepted, Vec3(accepted.x, desired.y, desired.z));
    for (const auto& box : boxes) {
        if (contains(box, accepted)) {
            accepted = start;
            break;
        }
    }
    if (history.size() >= 32 && found == history.end()) {
        history.erase(history.begin());
    }
    history[owner] = accepted;
    return accepted;
}
void constrainHiddenBodyPerson(Input::BodyPerson& person) {
    const auto& pelvis = person.joints.at("pelvis");
    if (person.confidence < 0.5 || pelvis[3] < 0.5) {
        return;
    }
    const auto& left = person.joints.at("l_ankle");
    const auto& right = person.joints.at("r_ankle");
    const auto& head = person.joints.at("head");
    const float floor = static_cast<float>(std::min(left[1], right[1]));
    const Vec3 desired(static_cast<float>(pelvis[0]), floor, static_cast<float>(pelvis[2]));
    const float height = std::clamp(static_cast<float>(head[1]) - floor + 0.2f, 0.2f, 5.0f);
    const auto accepted = constrainHiddenPlayerAnchor("body:" + std::to_string(person.id),
                                                     desired, 0.25f, height);
    const auto delta = accepted - desired;
    for (auto& joint : person.joints) {
        joint.second[0] += delta.x;
        joint.second[2] += delta.z;
    }
}
} // namespace Paint
