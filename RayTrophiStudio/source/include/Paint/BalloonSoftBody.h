#pragma once

#include "../../src/Physics/Jolt/JoltWorld.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <stdexcept>
#include <utility>

namespace Paint {

inline RayTrophiSim::JoltIntegration::JoltSoftBodyDesc makeBalloonShell() {
    namespace Jolt = RayTrophiSim::JoltIntegration;
    Jolt::JoltSoftBodyDesc shell;
    shell.vertices = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0},
                      {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    constexpr std::array<std::array<uint32_t, 3>, 8> faces = {{
        {0, 2, 4}, {0, 4, 3}, {0, 3, 5}, {0, 5, 2},
        {1, 4, 2}, {1, 3, 4}, {1, 5, 3}, {1, 2, 5}
    }};
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> midpoints;
    const auto midpoint = [&](uint32_t a, uint32_t b) {
        const auto key = std::make_pair(std::min(a, b), std::max(a, b));
        const auto index = static_cast<uint32_t>(shell.vertices.size());
        const auto [entry, inserted] = midpoints.emplace(key, index);
        if (inserted) {
            shell.vertices.push_back((shell.vertices[a] + shell.vertices[b]).normalize());
        }
        return entry->second;
    };
    for (const auto& [a, b, c] : faces) {
        const auto ab = midpoint(a, b);
        const auto bc = midpoint(b, c);
        const auto ca = midpoint(c, a);
        shell.indices.insert(shell.indices.end(),
                             {a, ab, ca, ab, b, bc, ca, bc, c, ab, bc, ca});
    }
    shell.compliance = 0.001f;
    shell.pressure = 2.0f;
    shell.damping = 5.0f;
    shell.gravity_factor = 0.0f;
    shell.vertex_radius = 0.01f;
    return shell;
}

// One unit-radius shell in a contact frame. The carry solver supplies the
// squeeze depth. The Jolt shell supplies the axial and transverse visual scale.
class BalloonSoftBody {
public:
    BalloonSoftBody() {
        namespace Jolt = RayTrophiSim::JoltIntegration;
        Jolt::JoltWorld::Config config;
        config.gravity = Vec3(0.0f);
        config.num_threads = 1;
        config.max_bodies = 8;
        config.max_body_pairs = 16;
        config.max_contact_constraints = 16;
        if (!world_.init(config)) {
            throw std::runtime_error("could not initialize balloon soft-body world");
        }

        const auto shell = makeBalloonShell();
        rest_ = shell.vertices;
        shell_ = world_.createSoftBody(shell);

        Jolt::JoltBodyDesc plate;
        plate.half_extents = Vec3(1.0f, 3.0f, 3.0f);
        plate.motion_type = Jolt::JoltMotionType::Static;
        plate.friction = 0.0f;
        plate.transform = platePose(2.01f);
        const auto wall = world_.createBody(plate);
        plate.motion_type = Jolt::JoltMotionType::Kinematic;
        plate.transform = platePose(-2.01f);
        pusher_ = world_.createBody(plate);
        if (shell_ == Jolt::kInvalidBody || wall == Jolt::kInvalidBody ||
            pusher_ == Jolt::kInvalidBody) {
            throw std::runtime_error("could not create balloon shell or contact plates");
        }
    }

    void update(int held, float compression, float dt) {
        if (held != held_) {
            held_ = held;
            axial_ = lateral_ = 1.0f;
            clock_ = 0.0f;
            travel_ = 0.0f;
            if (held >= 0) {
                world_.setSoftBodyVertices(shell_, rest_);
            }
        }
        if (held < 0 ||
            (compression == 0.0f && travel_ == 0.0f && axial_ == 1.0f && lateral_ == 1.0f)) {
            return;
        }

        constexpr float step = 1.0f / 60.0f;
        clock_ += std::min(dt, 4.0f * step);
        if (clock_ < step) {
            return;
        }
        int steps = 0;
        while (clock_ >= step) {
            travel_ += std::clamp(2.0f * compression - travel_, -0.10f, 0.10f);
            world_.setBodyTransform(pusher_, platePose(-2.01f + travel_));
            world_.activateBody(shell_);
            world_.update(step);
            clock_ -= step;
            ++steps;
        }
        if (!world_.getSoftBodyVertices(shell_, vertices_) || vertices_.empty()) {
            throw std::runtime_error("could not read balloon shell");
        }
        Vec3 lo = vertices_.front();
        Vec3 hi = lo;
        for (const auto& vertex : vertices_) {
            for (int axis = 0; axis < 3; ++axis) {
                if (!std::isfinite(vertex[axis])) {
                    throw std::runtime_error("balloon shell returned a non-finite position");
                }
                lo[axis] = std::min(lo[axis], vertex[axis]);
                hi[axis] = std::max(hi[axis], vertex[axis]);
            }
        }
        const Vec3 width = hi - lo;
        const bool relaxed = compression == 0.0f && travel_ == 0.0f;
        const float axial = relaxed ? 1.0f : std::clamp(width.x * 0.5f, 0.6f, 1.0f);
        const float lateral = relaxed ? 1.0f : std::clamp((width.y + width.z) * 0.25f, 1.0f, 1.25f);
        const float blend = 1.0f - std::exp(-18.0f * step * steps);
        axial_ += (axial - axial_) * blend;
        lateral_ += (lateral - lateral_) * blend;
        if (relaxed && axial_ > 0.999f && lateral_ < 1.001f) {
            axial_ = lateral_ = 1.0f;
        }
    }

    int held() const { return held_; }
    float axial() const { return axial_; }
    float lateral() const { return lateral_; }

private:
    static Matrix4x4 platePose(float x) {
        auto pose = Matrix4x4::identity();
        pose.m[0][3] = x;
        return pose;
    }

    RayTrophiSim::JoltIntegration::JoltWorld world_;
    RayTrophiSim::JoltIntegration::JoltBodyHandle shell_;
    RayTrophiSim::JoltIntegration::JoltBodyHandle pusher_;
    std::vector<Vec3> rest_;
    std::vector<Vec3> vertices_;
    int held_ = -1;
    float axial_ = 1.0f;
    float lateral_ = 1.0f;
    float clock_ = 0.0f;
    float travel_ = 0.0f;
};

} // namespace Paint
