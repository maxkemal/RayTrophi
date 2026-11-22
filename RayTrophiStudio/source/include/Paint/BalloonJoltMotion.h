#pragma once

#include "Paint/BalloonInteraction.h"
#include "Paint/BalloonImpactShells.h"
#include "Paint/BalloonRoomFront.h"
#include "../../src/Physics/Jolt/JoltWorld.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace Paint {

inline constexpr float defaultBalloonWindStrength = 0.25f;

inline float balloonWindStrength(const nlohmann::json& input) {
    if (!input.is_number() || !std::isfinite(input.get<double>()) ||
        input.get<double>() < 0.0 || input.get<double>() > 1.0) {
        throw std::runtime_error("wind_strength must be a number in 0..1");
    }
    return input.get<float>();
}

inline void validateBalloonJoltMotion(const nlohmann::json& config) {
    if (config.contains("jolt_motion") && (!config["jolt_motion"].is_boolean() ||
        (config["jolt_motion"] == true && !config.contains("visual_nodes")))) {
        throw std::runtime_error("jolt_motion must be a boolean and true requires mesh visuals");
    }
    if (config.contains("impact_soft_body") && !config["impact_soft_body"].is_boolean()) {
        throw std::runtime_error("impact_soft_body must be a boolean");
    }
    if (config.contains("wind_strength")) {
        (void)balloonWindStrength(config["wind_strength"]);
    }
}

class BalloonJoltMotion {
public:
    BalloonJoltMotion(std::vector<Vec3> room_vertices, size_t count, bool impact_soft_body,
                      float wind_strength, BalloonRoomFront front)
        : bodies_(count), front_(std::move(front)), wind_strength_(wind_strength) {
        using namespace RayTrophiSim::JoltIntegration;
        JoltWorld::Config config;
        config.num_threads = 1;
        config.max_bodies = 64;
        config.max_body_pairs = 2048;
        config.max_contact_constraints = 2048;
        if (!world_.init(config)) {
            throw std::runtime_error("could not initialize Jolt balloon motion");
        }
        JoltBodyDesc room;
        room.shape = JoltShapeType::Mesh;
        room.motion_type = JoltMotionType::Static;
        room.friction = 0.08f;
        room.restitution = 0.26f;
        room_vertices.insert(room_vertices.end(), front_.triangles.begin(), front_.triangles.end());
        room.mesh_vertices = std::move(room_vertices);
        room.mesh_indices.reserve(room.mesh_vertices.size());
        for (size_t i = 0; i < room.mesh_vertices.size(); ++i) {
            room.mesh_indices.push_back(static_cast<uint32_t>(i));
        }
        room_ = world_.createBody(room);
        if (room_ == kInvalidBody) {
            throw std::runtime_error("Jolt balloon motion requires static room mesh geometry");
        }
        world_.setContactCaptureEnabled(true);
        if (impact_soft_body) {
            shells_ = std::make_unique<BalloonImpactShells>(count);
        }
    }

    void spawn(size_t id, const Vec3& position, float radius) {
        auto& body = bodies_[id];
        body.radius = radius;
        body.home = position;
        body.until_drop = 2.0f + static_cast<float>(id) * 0.47f;
        body.fall_remaining = 0.0f;
        body.lift_blend = 1.0f;
        body.phase = static_cast<float>(id) * 1.71f;
        create(body, position, false);
    }

    void remove(size_t id) {
        auto& body = bodies_[id];
        world_.removeBody(body.handle);
        if (shells_) {
            shells_->reset(id);
        }
        body = Body{};
        if (held_ == static_cast<int>(id)) {
            held_ = -1;
        }
    }

    void setHeld(int id, const Vec3& position) {
        if (held_ != id) {
            if (held_ >= 0) {
                auto& previous = bodies_[held_];
                const auto released_at = world_.getBodyPosition(previous.handle);
                create(previous, released_at, false);
                startFall(previous);
            }
            held_ = id;
            if (held_ >= 0) {
                create(bodies_[held_], position, true);
            }
        }
        if (held_ >= 0 &&
            (world_.getBodyPosition(bodies_[held_].handle) - position).length_squared() > 1e-10f) {
            auto& body = bodies_[held_];
            world_.setBodyTransform(body.handle, pose(position, body.rotation));
        }
    }

    void push(size_t id, const Vec3& displacement) {
        auto& body = bodies_[id];
        const auto handle = body.handle;
        const auto kick = displacement * 7.0f;
        auto velocity = world_.getBodyLinearVelocity(handle) + kick;
        const float speed = velocity.length();
        if (speed > 4.0f) {
            velocity = velocity * (4.0f / speed);
        }
        world_.setBodyLinearVelocity(handle, velocity);
        const float kick_speed = kick.length();
        if (kick_speed >= 0.25f) {
            registerImpact(handle, kick / kick_speed, kick_speed);
            if (held_ >= 0) {
                registerImpact(bodies_[held_].handle, kick / kick_speed, kick_speed);
            }
        }
    }

    void update(float dt) {
        using namespace std::chrono;
        const auto start = steady_clock::now();
        constexpr float step = 1.0f / 60.0f;
        clock_ += std::min(dt, 4.0f * step);
        while (clock_ >= step) {
            air_phase_ = std::fmod(air_phase_ + step * 0.65f, 6.2831853f);
            const float wind_scale = wind_strength_ / defaultBalloonWindStrength;
            const Vec3 wind(0.23f * wind_scale * std::sin(air_phase_), 0.0f,
                            0.20f * wind_scale * std::cos(air_phase_));
            for (size_t id = 0; id < bodies_.size(); ++id) {
                if (live(id)) {
                    auto& body = bodies_[id];
                    body.impact_age += step;
                    if (static_cast<int>(id) != held_) {
                        applyLift(body, wind, step);
                    }
                }
            }
            world_.update(step);
            world_.drainContactEvents(contacts_);
            for (const auto& contact : contacts_) {
                if (contact.is_new) {
                    if (contact.body_a == room_ || contact.body_b == room_) {
                        ++room_hits;
                    } else {
                        ++balloon_hits;
                    }
                    registerImpact(contact.body_a, contact.normal, contact.closing_speed);
                    registerImpact(contact.body_b, contact.normal, contact.closing_speed);
                }
            }
            if (shells_) {
                const auto shells_start = steady_clock::now();
                for (size_t id = 0; id < bodies_.size(); ++id) {
                    shells_->load(id, impactAmount(bodies_[id]));
                }
                shells_->step(step);
                for (size_t id = 0; id < bodies_.size(); ++id) {
                    peak_shell_squash = std::max(peak_shell_squash, 1.0f - shells_->scale(id).x);
                }
                shell_ms_max = std::max(shell_ms_max,
                    duration<double, std::milli>(steady_clock::now() - shells_start).count());
            }
            ++steps;
            clock_ -= step;
        }
        last_ms = duration<double, std::milli>(steady_clock::now() - start).count();
        max_ms = std::max(max_ms, last_ms);
    }

    template<class Balloons>
    void syncHold(const Balloons& balloons, int held) {
        setHeld(held, held >= 0 ? balloons[held].position : Vec3(0.0f));
    }

    template<class Balloons>
    void advance(Balloons& balloons, int held, float dt) {
        syncHold(balloons, held);
        update(dt);
        for (size_t id = 0; id < balloons.size(); ++id) {
            if (static_cast<int>(id) != held && live(id)) {
                auto& body = bodies_[id];
                body.rotation = world_.getBodyTransform(body.handle);
                balloons[id].position = Vec3(body.rotation.m[0][3], body.rotation.m[1][3],
                                             body.rotation.m[2][3]);
                for (int axis = 0; axis < 3; ++axis) {
                    body.rotation.m[axis][3] = 0.0f;
                }
                balloons[id].anchor = balloons[id].position;
            }
        }
    }

    bool live(size_t id) const {
        return bodies_[id].handle != RayTrophiSim::JoltIntegration::kInvalidBody;
    }
    size_t count() const { return world_.bodyCount() - 1; }
    size_t shellCount() const { return shells_ ? shells_->count() : 0; }
    const BalloonRoomFront& front() const { return front_; }
    void setWindStrength(float strength) { wind_strength_ = strength; }

    Matrix4x4 visualRotation(size_t id) const { return bodies_[id].rotation; }

    size_t rotatingBodies() const {
        return static_cast<size_t>(std::count_if(bodies_.begin(), bodies_.end(),
            [this](const Body& body) {
                return body.handle != RayTrophiSim::JoltIntegration::kInvalidBody &&
                       world_.getBodyAngularVelocity(body.handle).length_squared() > 0.01f;
            }));
    }

    void applyImpactVisual(size_t id, Matrix4x4& transform) const {
        const auto& body = bodies_[id];
        const auto scale = shells_ ? shells_->scale(id) :
            Vec3(1.0f - impactAmount(body), 1.0f + 0.4f * impactAmount(body), 1.0f);
        const float squash = 1.0f - scale.x;
        if (squash <= 0.0f && scale.y <= 1.0f) {
            return;
        }
        const float lateral = scale.y;
        for (int col = 0; col < 3; ++col) {
            const Vec3 axis(transform.m[0][col], transform.m[1][col], transform.m[2][col]);
            const auto along = body.impact_axis * Vec3::dot(axis, body.impact_axis);
            const auto scaled = (axis - along) * lateral + along * (1.0f - squash);
            for (int row = 0; row < 3; ++row) {
                transform.m[row][col] = scaled[row];
            }
        }
    }

    uint64_t steps = 0;
    uint64_t balloon_hits = 0;
    uint64_t room_hits = 0;
    uint64_t squashes = 0;
    float peak_squash = 0.0f;
    float peak_shell_squash = 0.0f;
    double shell_ms_max = 0.0;
    double last_ms = 0.0;
    double max_ms = 0.0;

private:
    static constexpr float gravity_scale = 0.23f;
    using Handle = RayTrophiSim::JoltIntegration::JoltBodyHandle;
    struct Body {
        Handle handle = RayTrophiSim::JoltIntegration::kInvalidBody;
        float radius = 0.4f;
        Vec3 home = Vec3(0.0f);
        float until_drop = 0.0f;
        float fall_remaining = 0.0f;
        float lift_blend = 1.0f;
        Vec3 impact_axis = Vec3(0.0f);
        float impact_strength = 0.0f;
        float impact_age = 1.0f;
        float phase = 0.0f;
        Matrix4x4 rotation = Matrix4x4::identity();
    };

    static float impactAmount(const Body& body) {
        constexpr float attack = 0.05f;
        constexpr float duration = 0.40f;
        const float envelope = std::clamp(body.impact_age / attack, 0.0f, 1.0f) *
            std::clamp((duration - body.impact_age) / (duration - attack), 0.0f, 1.0f);
        return body.impact_strength * envelope * envelope;
    }

    void registerImpact(Handle handle, const Vec3& axis, float closing_speed) {
        if (handle == room_) {
            return;
        }
        const auto body = std::find_if(bodies_.begin(), bodies_.end(),
            [handle](const Body& entry) { return entry.handle == handle; });
        if (body == bodies_.end()) {
            return;
        }
        const float strength = std::clamp(closing_speed * 0.079f, 0.0f, 0.20f);
        if (body->impact_age == 0.0f && strength > body->impact_strength) {
            body->impact_axis = axis;
            body->impact_strength = strength;
            peak_squash = std::max(peak_squash, strength);
            return;
        }
        if (body->impact_age < 0.08f || impactAmount(*body) > 0.015f) {
            return;
        }
        body->impact_axis = axis;
        body->impact_strength = strength;
        body->impact_age = 0.0f;
        ++squashes;
        peak_squash = std::max(peak_squash, strength);
    }

    static Matrix4x4 pose(const Vec3& position, Matrix4x4 transform) {
        for (int axis = 0; axis < 3; ++axis) {
            transform.m[axis][3] = position[axis];
        }
        return transform;
    }

    void create(Body& body, const Vec3& position, bool held) {
        using namespace RayTrophiSim::JoltIntegration;
        world_.removeBody(body.handle);
        JoltBodyDesc desc;
        desc.shape = JoltShapeType::Sphere;
        desc.motion_type = held ? JoltMotionType::Kinematic : JoltMotionType::Dynamic;
        desc.radius = body.radius;
        desc.transform = pose(position, body.rotation);
        desc.mass = 1.0f;
        desc.linear_damping = 0.32f;
        desc.friction = 0.08f;
        desc.restitution = 0.26f;
        desc.gravity_scale = gravity_scale;
        desc.angular_damping = 0.8f;
        desc.sleep_enabled = false;
        body.handle = world_.createBody(desc);
        if (body.handle == kInvalidBody) {
            throw std::runtime_error("could not create Jolt balloon collider");
        }
    }

    static void startFall(Body& body) {
        body.fall_remaining = 2.15f;
        body.until_drop = 10.0f;
    }

    void applyLift(Body& body, const Vec3& wind, float dt) {
        float target_blend = 1.0f;
        if (body.fall_remaining > 0.0f) {
            body.fall_remaining -= dt;
            target_blend = 0.0f;
        } else {
            body.until_drop -= dt;
            if (body.until_drop <= 0.0f) {
                startFall(body);
            }
        }
        body.lift_blend += std::clamp(target_blend - body.lift_blend, -dt * 4.0f, dt * 1.7f);
        const auto velocity = world_.getBodyLinearVelocity(body.handle);
        const auto center = world_.getBodyPosition(body.handle);
        const auto error = body.home - center;
        const float target_y_speed = std::clamp(error.y * 0.75f, -0.45f, 0.75f);
        const float lift = std::clamp(9.81f * gravity_scale + 2.8f * (target_y_speed - velocity.y),
                                     0.0f, 5.0f);
        Vec3 force(wind.x + std::clamp(error.x * 0.28f, -0.7f, 0.7f),
                   lift * body.lift_blend,
                   wind.z + std::clamp(error.z * 0.28f, -0.7f, 0.7f));
        if (!front_.triangles.empty()) {
            const float near = std::clamp(body.radius + 0.55f - front_.distance(center),
                                          0.0f, 1.0f);
            const float outward_speed = std::max(0.0f, -Vec3::dot(velocity, front_.inward));
            force += front_.inward * (near * (3.0f + 5.0f * outward_speed));
        }
        world_.addForce(body.handle, force, false);
        const float torque = 0.24f * body.radius * body.radius *
            (wind_strength_ / defaultBalloonWindStrength);
        world_.addTorque(body.handle,
            Vec3(std::sin(air_phase_ + body.phase), 0.3f,
                 std::cos(air_phase_ * 0.8f + body.phase)) * torque, false);
    }

    RayTrophiSim::JoltIntegration::JoltWorld world_;
    std::vector<Body> bodies_;
    BalloonRoomFront front_;
    float wind_strength_;
    std::unique_ptr<BalloonImpactShells> shells_;
    std::vector<RayTrophiSim::JoltIntegration::ContactEvent> contacts_;
    Handle room_ = RayTrophiSim::JoltIntegration::kInvalidBody;
    int held_ = -1;
    float clock_ = 0.0f;
    float air_phase_ = 0.0f;
};

inline std::unique_ptr<BalloonJoltMotion> prepareBalloonJoltMotion(
    SceneData& scene, const nlohmann::json& config, std::vector<std::string> excluded_visuals) {
    if (!config.value("jolt_motion", false)) {
        return nullptr;
    }
    const auto visuals = config.at("visual_nodes").get<std::vector<std::string>>();
    excluded_visuals.insert(excluded_visuals.end(), visuals.begin(), visuals.end());
    const auto surfaces = config.value("surfaces", std::vector<std::string>{});
    return std::make_unique<BalloonJoltMotion>(
        balloonRoomTriangles(scene, excluded_visuals), visuals.size(),
                                              config.value("impact_soft_body", true),
                                              config.value("wind_strength", defaultBalloonWindStrength),
                                              balloonPaintRoomFront(scene, surfaces));
}

inline void applyBalloonWind(const nlohmann::json& input, nlohmann::json& config,
                             BalloonJoltMotion* motion) {
    if (!config.contains("visual_nodes") || !input.is_object() ||
        input.size() != 1 || !input.contains("wind_strength")) {
        throw std::runtime_error("wind requires a prepared mesh demo and wind_strength");
    }
    const auto strength = balloonWindStrength(input["wind_strength"]);
    config["wind_strength"] = strength;
    if (motion) {
        motion->setWindStrength(strength);
    }
}

inline nlohmann::json balloonJoltMotionStatus(const BalloonJoltMotion* motion) {
    if (!motion) {
        return {{"enabled", false}};
    }
    const auto& front = motion->front();
    auto barrier = nlohmann::json(nullptr);
    if (!front.triangles.empty()) {
        const auto p = front.triangles.front();
        barrier = {{"point", {p.x, p.y, p.z}},
                   {"inward", {front.inward.x, front.inward.y, front.inward.z}}};
    }
    return {{"enabled", true}, {"bodies", motion->count()}, {"steps", motion->steps},
            {"front_barrier", barrier},
            {"balloon_hits", motion->balloon_hits}, {"room_hits", motion->room_hits},
            {"squashes", motion->squashes}, {"peak_squash", motion->peak_squash},
            {"soft_shells", motion->shellCount()}, {"peak_shell_squash", motion->peak_shell_squash},
            {"shell_step_ms_max", motion->shell_ms_max}, {"rotating_bodies", motion->rotatingBodies()},
            {"last_ms", motion->last_ms}, {"max_ms", motion->max_ms}};
}

} // namespace Paint
