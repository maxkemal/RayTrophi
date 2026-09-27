#include "KinematicColliderSource.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <unordered_set>
#include <unordered_map>
#include <utility>

namespace RayTrophiSim {
namespace {

constexpr float kEpsilon = 1.0e-6f;

bool finite(float value) {
    return std::isfinite(value);
}

bool finite(const Vec3& value) {
    return finite(value.x) && finite(value.y) && finite(value.z);
}

bool finite(const Matrix4x4& value) {
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            if (!finite(value.m[row][column])) {
                return false;
            }
        }
    }
    return true;
}

float lengthSquared(const Vec3& value) {
    return value.x * value.x + value.y * value.y + value.z * value.z;
}

float vectorLength(const Vec3& value) {
    return std::sqrt(lengthSquared(value));
}

float transformScale(const Matrix4x4& transform, const Vec3& axis) {
    const float local_length = vectorLength(axis);
    if (local_length <= kEpsilon) {
        return 1.0f;
    }
    return vectorLength(transform.transform_vector(axis)) / local_length;
}

float maximumBasisScale(const Matrix4x4& transform) {
    return std::max({
        transformScale(transform, Vec3(1.0f, 0.0f, 0.0f)),
        transformScale(transform, Vec3(0.0f, 1.0f, 0.0f)),
        transformScale(transform, Vec3(0.0f, 0.0f, 1.0f))});
}

Vec3 normalized(const Vec3& value, const Vec3& fallback) {
    const float length = vectorLength(value);
    return length > kEpsilon ? value * (1.0f / length) : fallback;
}

std::string canonical(std::string text) {
    std::string result;
    result.reserve(text.size());
    for (char c : text) {
        if (c == ' ' || c == '-' || c == '_') {
            continue;
        }
        result.push_back(static_cast<char>(
            std::tolower(static_cast<unsigned char>(c))));
    }
    return result;
}

bool containsToken(const std::string& value, const char* token) {
    return canonical(value).find(token) != std::string::npos;
}

bool isDetailBone(const std::string& name) {
    return containsToken(name, "thumb") ||
           containsToken(name, "index") ||
           containsToken(name, "middle") ||
           containsToken(name, "ring") ||
           containsToken(name, "pinky") ||
           containsToken(name, "eye") ||
           containsToken(name, "skirt") ||
           containsToken(name, "hair") ||
           containsToken(name, "twist") ||
           containsToken(name, "tongue") ||
           containsToken(name, "jaw") ||
           containsToken(name, "breast") ||
           containsToken(name, "end");
}

bool isBodyAnchor(const std::string& name) {
    return containsToken(name, "head") ||
           containsToken(name, "hand") ||
           containsToken(name, "foot") ||
           containsToken(name, "toe");
}

Matrix4x4 localTransform(const KinematicProxyDesc& proxy) {
    return Matrix4x4::fromTRS(
        proxy.local_position,
        proxy.local_rotation_degrees,
        Vec3(1.0f, 1.0f, 1.0f));
}

Matrix4x4 rotationOnly(const Matrix4x4& source) {
    Matrix4x4 result = Matrix4x4::identity();
    Vec3 x(source.m[0][0], source.m[1][0], source.m[2][0]);
    Vec3 y(source.m[0][1], source.m[1][1], source.m[2][1]);
    x = normalized(x, Vec3(1.0f, 0.0f, 0.0f));
    y = y - x * (x.x * y.x + x.y * y.y + x.z * y.z);
    y = normalized(y, Vec3(0.0f, 1.0f, 0.0f));
    Vec3 z(
        x.y * y.z - x.z * y.y,
        x.z * y.x - x.x * y.z,
        x.x * y.y - x.y * y.x);
    z = normalized(z, Vec3(0.0f, 0.0f, 1.0f));
    result.m[0][0] = x.x;
    result.m[1][0] = x.y;
    result.m[2][0] = x.z;
    result.m[0][1] = y.x;
    result.m[1][1] = y.y;
    result.m[2][1] = y.z;
    result.m[0][2] = z.x;
    result.m[1][2] = z.y;
    result.m[2][2] = z.z;
    return result;
}

Vec3 angularVelocity(const Matrix4x4& previous,
                     const Matrix4x4& current,
                     float inverse_dt) {
    const Matrix4x4 delta = rotationOnly(current) * rotationOnly(previous).transpose();
    return Vec3(
        delta.m[2][1] - delta.m[1][2],
        delta.m[0][2] - delta.m[2][0],
        delta.m[1][0] - delta.m[0][1]) * (0.5f * inverse_dt);
}

KinematicProxyDesc fitLeaf(const KinematicJointPose& joint,
                           const KinematicAutoFitOptions& options) {
    KinematicProxyDesc proxy;
    proxy.name = joint.name;
    proxy.bone = joint.name;
    const float world_scale = std::max(
        maximumBasisScale(joint.world), kEpsilon);
    const float minimum_radius_local = options.minimum_radius / world_scale;
    const float maximum_radius_local = options.maximum_radius / world_scale;
    const bool foot = containsToken(joint.name, "foot") ||
                      containsToken(joint.name, "toe");
    if (foot && joint.has_mesh_bounds) {
        // The ankle pivot is not the foot: the box follows the heel, sole and
        // toes the skin actually has, so a planted foot reaches the ground.
        proxy.shape = KinematicProxyShape::Box;
        proxy.local_position =
            (joint.mesh_bounds_min + joint.mesh_bounds_max) * 0.5f;
        const Vec3 half =
            (joint.mesh_bounds_max - joint.mesh_bounds_min) * 0.5f;
        const float floor_local = minimum_radius_local * 0.5f;
        proxy.half_extents = Vec3(
            std::max(half.x, floor_local),
            std::max(half.y, floor_local),
            std::max(half.z, floor_local));
    } else if (foot) {
        proxy.shape = KinematicProxyShape::Box;
        proxy.half_extents = Vec3(
            minimum_radius_local * 1.5f,
            minimum_radius_local,
            minimum_radius_local * 2.5f);
    } else {
        proxy.shape = KinematicProxyShape::Sphere;
        const bool head = containsToken(joint.name, "head");
        proxy.radius = head
            ? std::min(maximum_radius_local,
                       minimum_radius_local * 3.0f)
            : minimum_radius_local * 1.5f;
    }
    return proxy;
}

} // namespace

const char* kinematicProxyShapeName(KinematicProxyShape shape) {
    switch (shape) {
        case KinematicProxyShape::Sphere:
            return "sphere";
        case KinematicProxyShape::Box:
            return "box";
        case KinematicProxyShape::Capsule:
        default:
            return "capsule";
    }
}

bool parseKinematicProxyShape(const std::string& text,
                              KinematicProxyShape& shape) {
    const std::string key = canonical(text);
    if (key == "sphere") {
        shape = KinematicProxyShape::Sphere;
        return true;
    }
    if (key == "capsule") {
        shape = KinematicProxyShape::Capsule;
        return true;
    }
    if (key == "box" || key == "obb") {
        shape = KinematicProxyShape::Box;
        return true;
    }
    return false;
}

const KinematicProxySet* KinematicColliderRegistry::findSet(uint64_t id) const {
    const auto it = std::find_if(
        sets_.begin(), sets_.end(),
        [id](const KinematicProxySet& set) { return set.id == id; });
    return it == sets_.end() ? nullptr : &*it;
}

KinematicProxySet* KinematicColliderRegistry::findSet(uint64_t id) {
    return const_cast<KinematicProxySet*>(
        static_cast<const KinematicColliderRegistry*>(this)->findSet(id));
}

const KinematicProxySet* KinematicColliderRegistry::findSet(
    const std::string& name) const {
    const auto it = std::find_if(
        sets_.begin(), sets_.end(),
        [&name](const KinematicProxySet& set) { return set.name == name; });
    return it == sets_.end() ? nullptr : &*it;
}

KinematicProxySet* KinematicColliderRegistry::findSet(const std::string& name) {
    return const_cast<KinematicProxySet*>(
        static_cast<const KinematicColliderRegistry*>(this)->findSet(name));
}

bool KinematicColliderRegistry::validateSet(const KinematicProxySet& set,
                                            uint64_t ignore_id,
                                            std::string& error) const {
    if (set.name.empty()) {
        error = "proxy_set_name_required";
        return false;
    }
    if (set.target_character.empty()) {
        error = "target_character_required";
        return false;
    }
    if (!set.target_node_id.empty()) {
        error = "target_node_id_not_supported";
        return false;
    }
    if (!finite(set.friction) || set.friction < 0.0f ||
        !finite(set.restitution) || set.restitution < 0.0f ||
        set.restitution > 1.0f ||
        !finite(set.thickness) || set.thickness < 0.0f) {
        error = "invalid_contact_material";
        return false;
    }
    if ((set.consumer_mask & ~KinematicConsumerAll) != 0u) {
        error = "invalid_consumer_mask";
        return false;
    }
    for (const KinematicProxySet& existing : sets_) {
        if (existing.id != ignore_id && existing.name == set.name) {
            error = "proxy_set_name_exists";
            return false;
        }
    }
    return true;
}

bool KinematicColliderRegistry::validateProxy(const KinematicProxyDesc& proxy,
                                              std::string& error) {
    if (proxy.bone.empty()) {
        error = "proxy_bone_required";
        return false;
    }
    if (!finite(proxy.local_position) ||
        !finite(proxy.local_rotation_degrees) ||
        !finite(proxy.local_axis) ||
        !finite(proxy.radius) ||
        !finite(proxy.half_length) ||
        !finite(proxy.half_extents)) {
        error = "proxy_values_must_be_finite";
        return false;
    }
    if (proxy.radius <= 0.0f || proxy.half_length < 0.0f ||
        proxy.half_extents.x <= 0.0f || proxy.half_extents.y <= 0.0f ||
        proxy.half_extents.z <= 0.0f) {
        error = "proxy_dimensions_must_be_positive";
        return false;
    }
    if (proxy.shape == KinematicProxyShape::Capsule &&
        lengthSquared(proxy.local_axis) <= kEpsilon * kEpsilon) {
        error = "capsule_axis_must_be_nonzero";
        return false;
    }
    return true;
}

bool KinematicColliderRegistry::createSet(const KinematicProxySet& requested,
                                          KinematicProxySet& created,
                                          std::string& error) {
    KinematicProxySet value = requested;
    value.id = next_set_id_;
    value.revision = 1;
    value.proxies.clear();
    if (sets_.size() >= kMaxKinematicProxySets) {
        error = "proxy_set_limit_reached";
        return false;
    }
    if (!validateSet(value, 0, error)) {
        return false;
    }
    ++next_set_id_;
    sets_.push_back(value);
    created = value;
    return true;
}

bool KinematicColliderRegistry::updateSet(uint64_t id,
                                          const KinematicProxySet& requested,
                                          std::string& error) {
    KinematicProxySet* existing = findSet(id);
    if (!existing) {
        error = "unknown_proxy_set";
        return false;
    }
    KinematicProxySet value = requested;
    value.id = id;
    value.proxies = existing->proxies;
    value.revision = existing->revision + 1;
    if (!validateSet(value, id, error)) {
        return false;
    }
    *existing = std::move(value);
    resetMotionHistory(id);
    return true;
}

bool KinematicColliderRegistry::removeSet(uint64_t id, std::string& error) {
    const auto it = std::find_if(
        sets_.begin(), sets_.end(),
        [id](const KinematicProxySet& set) { return set.id == id; });
    if (it == sets_.end()) {
        error = "unknown_proxy_set";
        return false;
    }
    for (const KinematicProxyDesc& proxy : it->proxies) {
        motion_history_.erase(proxy.id);
    }
    sets_.erase(it);
    return true;
}

bool KinematicColliderRegistry::setProxy(uint64_t set_id,
                                         const KinematicProxyDesc& requested,
                                         KinematicProxyDesc& stored,
                                         std::string& error) {
    KinematicProxySet* set = findSet(set_id);
    if (!set) {
        error = "unknown_proxy_set";
        return false;
    }
    KinematicProxyDesc value = requested;
    if (!validateProxy(value, error)) {
        return false;
    }
    if (value.id == 0) {
        if (set->proxies.size() >= kMaxKinematicProxiesPerSet) {
            error = "proxy_limit_reached";
            return false;
        }
        value.id = next_proxy_id_++;
        if (value.name.empty()) {
            value.name = value.bone;
        }
        set->proxies.push_back(value);
    } else {
        const auto it = std::find_if(
            set->proxies.begin(), set->proxies.end(),
            [&value](const KinematicProxyDesc& proxy) {
                return proxy.id == value.id;
            });
        if (it == set->proxies.end()) {
            error = "unknown_proxy";
            return false;
        }
        if (value.name.empty()) {
            value.name = value.bone;
        }
        *it = value;
        motion_history_.erase(value.id);
    }
    ++set->revision;
    stored = value;
    return true;
}

bool KinematicColliderRegistry::removeProxy(uint64_t set_id,
                                            uint64_t proxy_id,
                                            std::string& error) {
    KinematicProxySet* set = findSet(set_id);
    if (!set) {
        error = "unknown_proxy_set";
        return false;
    }
    const auto it = std::find_if(
        set->proxies.begin(), set->proxies.end(),
        [proxy_id](const KinematicProxyDesc& proxy) {
            return proxy.id == proxy_id;
        });
    if (it == set->proxies.end()) {
        error = "unknown_proxy";
        return false;
    }
    motion_history_.erase(proxy_id);
    set->proxies.erase(it);
    ++set->revision;
    return true;
}

bool KinematicColliderRegistry::autoFit(
    uint64_t set_id,
    const std::vector<KinematicJointPose>& joints,
    const KinematicAutoFitOptions& options,
    uint32_t& created_count,
    std::string& error) {
    created_count = 0;
    KinematicProxySet* set = findSet(set_id);
    if (!set) {
        error = "unknown_proxy_set";
        return false;
    }
    if (joints.empty()) {
        error = "character_has_no_joints";
        return false;
    }
    if (!finite(options.radius_fraction) || options.radius_fraction <= 0.0f ||
        !finite(options.minimum_radius) || options.minimum_radius <= 0.0f ||
        !finite(options.maximum_radius) ||
        options.maximum_radius < options.minimum_radius ||
        !finite(options.minimum_bone_length) ||
        options.minimum_bone_length <= 0.0f || options.maximum_proxies == 0 ||
        options.maximum_proxies > kMaxKinematicProxiesPerSet) {
        error = "invalid_auto_fit_options";
        return false;
    }

    std::unordered_map<std::string, std::vector<const KinematicJointPose*>> children;
    for (const KinematicJointPose& joint : joints) {
        if (!joint.parent.empty()) {
            children[joint.parent].push_back(&joint);
        }
    }

    std::vector<const KinematicJointPose*> ordered_joints;
    ordered_joints.reserve(joints.size());
    for (const KinematicJointPose& joint : joints) {
        if (options.weighted_bones_only && !joint.weighted) {
            continue;
        }
        if (!options.include_detail_bones && isDetailBone(joint.name)) {
            continue;
        }
        ordered_joints.push_back(&joint);
    }
    std::stable_sort(
        ordered_joints.begin(),
        ordered_joints.end(),
        [](const KinematicJointPose* left, const KinematicJointPose* right) {
            return isDetailBone(left->name) < isDetailBone(right->name);
        });

    std::vector<KinematicProxyDesc> generated;
    generated.reserve(std::min<std::size_t>(
        ordered_joints.size(), options.maximum_proxies));
    for (const KinematicJointPose* joint_ptr : ordered_joints) {
        const KinematicJointPose& joint = *joint_ptr;
        if (generated.size() >= options.maximum_proxies) {
            break;
        }
        if (isBodyAnchor(joint.name)) {
            generated.push_back(fitLeaf(joint, options));
            continue;
        }
        const auto child_it = children.find(joint.name);
        const KinematicJointPose* best_child = nullptr;
        float best_local_length = 0.0f;
        float best_world_length = 0.0f;
        Vec3 best_local(0.0f);
        if (child_it != children.end()) {
            const Matrix4x4 inverse = joint.world.inverse();
            for (const KinematicJointPose* child : child_it->second) {
                const Vec3 local = inverse.transform_point(
                    child->world.getTranslation());
                const float local_length = vectorLength(local);
                const float world_length = vectorLength(
                    child->world.getTranslation() -
                    joint.world.getTranslation());
                if (world_length > best_world_length) {
                    best_child = child;
                    best_local_length = local_length;
                    best_world_length = world_length;
                    best_local = local;
                }
            }
        }

        if (!best_child || best_world_length < options.minimum_bone_length) {
            generated.push_back(fitLeaf(joint, options));
            continue;
        }

        KinematicProxyDesc proxy;
        proxy.name = joint.name;
        proxy.bone = joint.name;
        proxy.shape = KinematicProxyShape::Capsule;
        proxy.local_position = best_local * 0.5f;
        proxy.local_axis = normalized(best_local, Vec3(0.0f, 1.0f, 0.0f));
        const float world_scale = std::max(
            maximumBasisScale(joint.world), kEpsilon);
        const float world_radius = std::clamp(
            best_world_length * options.radius_fraction,
            options.minimum_radius,
            options.maximum_radius);
        proxy.radius = world_radius / world_scale;
        proxy.half_length = std::max(
            0.0f, best_local_length * 0.5f - proxy.radius);

        // Head, hand, foot and toe joints never reach this point: they are
        // body anchors, fitted by fitLeaf above.
        generated.push_back(proxy);
    }

    if (generated.empty()) {
        error = "auto_fit_found_no_eligible_joints";
        return false;
    }
    if (!options.replace_existing &&
        set->proxies.size() + generated.size() >
            kMaxKinematicProxiesPerSet) {
        error = "proxy_limit_reached";
        return false;
    }
    if (options.replace_existing) {
        for (const KinematicProxyDesc& proxy : set->proxies) {
            motion_history_.erase(proxy.id);
        }
        set->proxies.clear();
    }
    for (KinematicProxyDesc& proxy : generated) {
        proxy.id = next_proxy_id_++;
        set->proxies.push_back(proxy);
        ++created_count;
    }
    ++set->revision;
    return true;
}

bool KinematicColliderRegistry::sampleSet(
    uint64_t set_id,
    float dt,
    bool discontinuity,
    const KinematicBoneResolver& resolver,
    std::vector<KinematicProxySample>& samples,
    std::string& error) {
    samples.clear();
    KinematicProxySet* set = findSet(set_id);
    if (!set) {
        error = "unknown_proxy_set";
        return false;
    }
    if (!resolver) {
        error = "bone_resolver_unavailable";
        return false;
    }
    if (discontinuity) {
        resetMotionHistory(set_id);
    }
    const float inverse_dt = dt > kEpsilon && finite(dt) ? 1.0f / dt : 0.0f;
    samples.reserve(set->proxies.size());
    for (const KinematicProxyDesc& proxy : set->proxies) {
        KinematicProxySample sample;
        sample.set_id = set->id;
        sample.proxy_id = proxy.id;
        sample.set_name = set->name;
        sample.proxy_name = proxy.name;
        sample.target_character = set->target_character;
        sample.bone = proxy.bone;
        sample.shape = proxy.shape;
        sample.consumer_mask = set->consumer_mask;
        sample.friction = set->friction;
        sample.restitution = set->restitution;
        sample.half_extents = proxy.half_extents + Vec3(set->thickness);

        if (!set->enabled || !proxy.enabled) {
            sample.unresolved_reason = !set->enabled
                ? "proxy_set_disabled"
                : "proxy_disabled";
            motion_history_.erase(proxy.id);
            samples.push_back(std::move(sample));
            continue;
        }

        Matrix4x4 bone_world;
        std::string reason;
        if (!resolver(set->target_character, proxy.bone, bone_world, reason)) {
            sample.unresolved_reason = reason.empty()
                ? "bone_unresolved"
                : reason;
            motion_history_.erase(proxy.id);
            samples.push_back(std::move(sample));
            continue;
        }

        sample.world_transform = bone_world * localTransform(proxy);
        sample.center = sample.world_transform.getTranslation();
        sample.radius = proxy.radius * maximumBasisScale(bone_world) +
                        set->thickness;
        if (proxy.shape == KinematicProxyShape::Capsule) {
            const Vec3 world_axis = bone_world.transform_vector(proxy.local_axis);
            const Vec3 axis = normalized(
                world_axis,
                Vec3(0.0f, 1.0f, 0.0f));
            const float world_half_length =
                proxy.half_length * transformScale(bone_world, proxy.local_axis);
            sample.capsule_start = sample.center - axis * world_half_length;
            sample.capsule_end = sample.center + axis * world_half_length;
        }
        sample.resolved = finite(sample.world_transform) &&
                          finite(sample.center) &&
                          finite(sample.radius) &&
                          sample.radius >= 0.0f &&
                          (proxy.shape != KinematicProxyShape::Capsule ||
                           (finite(sample.capsule_start) &&
                            finite(sample.capsule_end)));
        if (!sample.resolved) {
            sample.unresolved_reason = "nonfinite_proxy_transform";
            motion_history_.erase(proxy.id);
            samples.push_back(std::move(sample));
            continue;
        }

        MotionHistory& history = motion_history_[proxy.id];
        if (history.valid && inverse_dt > 0.0f) {
            sample.linear_velocity =
                (sample.center - history.center) * inverse_dt;
            sample.angular_velocity = angularVelocity(
                history.world, sample.world_transform, inverse_dt);
            sample.velocity_valid = finite(sample.linear_velocity) &&
                                    finite(sample.angular_velocity);
            if (!sample.velocity_valid) {
                sample.linear_velocity = Vec3(0.0f);
                sample.angular_velocity = Vec3(0.0f);
            }
        }
        history.world = sample.world_transform;
        history.center = sample.center;
        history.valid = true;
        samples.push_back(std::move(sample));
    }
    return true;
}

void KinematicColliderRegistry::resetMotionHistory() {
    motion_history_.clear();
}

void KinematicColliderRegistry::resetMotionHistory(uint64_t set_id) {
    const KinematicProxySet* set = findSet(set_id);
    if (!set) {
        return;
    }
    for (const KinematicProxyDesc& proxy : set->proxies) {
        motion_history_.erase(proxy.id);
    }
}

bool KinematicColliderRegistry::restoreSets(
    const std::vector<KinematicProxySet>& sets,
    std::string& error) {
    if (sets.size() > kMaxKinematicProxySets) {
        error = "proxy_set_limit_reached";
        return false;
    }

    KinematicColliderRegistry staged;
    std::unordered_set<uint64_t> set_ids;
    std::unordered_set<uint64_t> proxy_ids;
    uint64_t maximum_set_id = 0;
    uint64_t maximum_proxy_id = 0;
    for (KinematicProxySet set : sets) {
        if (set.id == 0 || !set_ids.insert(set.id).second) {
            error = "invalid_or_duplicate_proxy_set_id";
            return false;
        }
        if (set.proxies.size() > kMaxKinematicProxiesPerSet) {
            error = "proxy_limit_reached";
            return false;
        }
        if (!staged.validateSet(set, set.id, error)) {
            return false;
        }
        for (const KinematicProxyDesc& proxy : set.proxies) {
            if (proxy.id == 0 || !proxy_ids.insert(proxy.id).second) {
                error = "invalid_or_duplicate_proxy_id";
                return false;
            }
            if (!validateProxy(proxy, error)) {
                return false;
            }
            maximum_proxy_id = std::max(maximum_proxy_id, proxy.id);
        }
        set.revision = std::max<uint64_t>(1, set.revision);
        maximum_set_id = std::max(maximum_set_id, set.id);
        staged.sets_.push_back(std::move(set));
    }
    if (maximum_set_id == std::numeric_limits<uint64_t>::max() ||
        maximum_proxy_id == std::numeric_limits<uint64_t>::max()) {
        error = "proxy_id_space_exhausted";
        return false;
    }

    staged.next_set_id_ = std::max(next_set_id_, maximum_set_id + 1);
    staged.next_proxy_id_ = std::max(next_proxy_id_, maximum_proxy_id + 1);
    *this = std::move(staged);
    return true;
}

void KinematicColliderRegistry::clear() {
    sets_.clear();
    motion_history_.clear();
    // Stable ids are deliberately not reused after a scene clear. External
    // tools holding an old id must fail instead of editing a new proxy set.
}

} // namespace RayTrophiSim
