#pragma once

#include "Paint/BalloonSoftBody.h"

namespace Paint {

// Independent contact frames in a shared world. X is the compression axis.
class BalloonImpactShells {
public:
    explicit BalloonImpactShells(size_t count) : lanes_(count) {
        using namespace RayTrophiSim::JoltIntegration;
        JoltWorld::Config config;
        config.gravity = Vec3(0.0f);
        config.num_threads = 1;
        config.max_bodies = 128;
        config.max_body_pairs = 512;
        config.max_contact_constraints = 256;
        if (!world_.init(config)) {
            throw std::runtime_error("could not initialize balloon impact shells");
        }
        const auto unit_shell = makeBalloonShell();
        for (size_t i = 0; i < lanes_.size(); ++i) {
            auto& lane = lanes_[i];
            lane.z = static_cast<float>(i) * 6.0f;
            auto shell = unit_shell;
            for (auto& point : shell.vertices) {
                point.z += lane.z;
            }
            lane.rest = shell.vertices;
            lane.shell = world_.createSoftBody(shell);

            JoltBodyDesc plate;
            plate.half_extents = Vec3(1.0f, 2.0f, 2.0f);
            plate.motion_type = JoltMotionType::Static;
            plate.friction = 0.0f;
            plate.transform = pose(2.01f, lane.z);
            const auto stop = world_.createBody(plate);
            plate.motion_type = JoltMotionType::Kinematic;
            plate.transform = pose(-2.01f, lane.z);
            lane.pusher = world_.createBody(plate);
            if (lane.shell == kInvalidBody || stop == kInvalidBody ||
                lane.pusher == kInvalidBody) {
                throw std::runtime_error("could not create balloon impact shell and plates");
            }
        }
        world_.optimizeBroadPhase();
    }

    void reset(size_t index) {
        auto& lane = lanes_[index];
        if (!lane.active && lane.travel == 0.0f) {
            return;
        }
        lane.active = false;
        lane.travel = 0.0f;
        lane.scale = Vec3(1.0f);
        world_.setSoftBodyVertices(lane.shell, lane.rest);
        world_.setBodyTransform(lane.pusher, pose(-2.01f, lane.z));
    }

    void load(size_t index, float compression) {
        auto& lane = lanes_[index];
        if (compression <= 0.0f && !lane.active) {
            return;
        }
        lane.active = true;
        lane.target = 2.0f * compression;
    }

    void step(float dt) {
        bool active = false;
        for (auto& lane : lanes_) {
            if (!lane.active) {
                continue;
            }
            active = true;
            lane.travel += std::clamp(lane.target - lane.travel, -0.10f, 0.10f);
            world_.setBodyTransform(lane.pusher, pose(-2.01f + lane.travel, lane.z));
            world_.activateBody(lane.shell);
        }
        if (!active) {
            return;
        }
        world_.update(dt);
        for (size_t i = 0; i < lanes_.size(); ++i) {
            auto& lane = lanes_[i];
            if (!lane.active) {
                continue;
            }
            if (!world_.getSoftBodyVertices(lane.shell, sample_) || sample_.empty()) {
                throw std::runtime_error("could not sample balloon impact shell");
            }
            Vec3 lo = sample_.front();
            Vec3 hi = lo;
            for (const auto& point : sample_) {
                for (int axis = 0; axis < 3; ++axis) {
                    if (!std::isfinite(point[axis])) {
                        throw std::runtime_error("non-finite balloon impact shell vertex");
                    }
                    lo[axis] = std::min(lo[axis], point[axis]);
                    hi[axis] = std::max(hi[axis], point[axis]);
                }
            }
            const auto width = hi - lo;
            const bool unloaded = lane.travel == 0.0f;
            const Vec3 measured = unloaded ? Vec3(1.0f) : Vec3(
                std::clamp(width.x * 0.5f, 0.70f, 1.0f),
                std::clamp((width.y + width.z) * 0.25f, 1.0f, 1.18f), 1.0f);
            lane.scale += (measured - lane.scale) * 0.38f;
            if (unloaded && (lane.scale - Vec3(1.0f)).length_squared() < 0.000004f) {
                reset(i);
            }
        }
    }

    Vec3 scale(size_t index) const { return lanes_[index].scale; }
    size_t count() const { return lanes_.size(); }

private:
    using Handle = RayTrophiSim::JoltIntegration::JoltBodyHandle;
    struct Lane {
        Handle shell = RayTrophiSim::JoltIntegration::kInvalidBody;
        Handle pusher = RayTrophiSim::JoltIntegration::kInvalidBody;
        std::vector<Vec3> rest;
        Vec3 scale = Vec3(1.0f);
        float z = 0.0f;
        float target = 0.0f;
        float travel = 0.0f;
        bool active = false;
    };

    static Matrix4x4 pose(float x, float z) {
        auto transform = Matrix4x4::identity();
        transform.m[0][3] = x;
        transform.m[2][3] = z;
        return transform;
    }

    RayTrophiSim::JoltIntegration::JoltWorld world_;
    std::vector<Lane> lanes_;
    std::vector<Vec3> sample_;
};

} // namespace Paint
