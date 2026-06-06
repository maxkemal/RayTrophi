#include "RigidBodySystem.h"

#include "JoltWorld.h"  // pure RT-typed wrapper (no Jolt headers leak in here)

#include <algorithm>
#include <cmath>

namespace RayTrophiSim {

RigidBodySystem::RigidBodySystem() = default;
RigidBodySystem::~RigidBodySystem() = default;  // defined here where JoltWorld is complete

bool RigidBodySystem::ensureWorld() {
    if (!world_) {
        world_ = std::make_unique<JoltIntegration::JoltWorld>();
    }
    if (!world_->isInitialized()) {
        JoltIntegration::JoltWorld::Config cfg;
        cfg.gravity = gravity_;
        if (!world_->init(cfg)) return false;
    }
    return true;
}

void RigidBodySystem::setGravity(const Vec3& g) {
    gravity_ = g;
    if (world_ && world_->isInitialized()) world_->setGravity(g);
}

void RigidBodySystem::ensureBodyCreated(RigidBodyObject& rb) {
    if (rb.created || rb.source_name.empty() || !shape_resolver_) return;

    Matrix4x4 box_pose;
    Vec3 half_extents;
    RigidBodyShape resolved_shape = rb.shape;
    if (!shape_resolver_(rb, box_pose, half_extents, resolved_shape)) {
        return;  // object not resolvable yet (e.g., surface cache not ready)
    }
    rb.shape = resolved_shape;

    Matrix4x4 pivot = Matrix4x4::identity();
    if (pivot_getter_) pivot_getter_(rb.source_name, pivot);

    if (!rb.rest_captured) {
        rb.initial_pivot = pivot;
        rb.rest_half_extents = half_extents;
        rb.rest_captured = true;
    } else {
        half_extents = rb.rest_half_extents;
    }

    if (rb.auto_mass_from_density && rb.motion_type == RigidBodyMotionType::Dynamic) {
        float volume = 1.0f;
        switch (resolved_shape) {
            case RigidBodyShape::Sphere: {
                const float r = std::max({ half_extents.x, half_extents.y, half_extents.z, 1e-4f });
                volume = (4.0f / 3.0f) * 3.1415926535f * r * r * r;
                break;
            }
            case RigidBodyShape::Capsule: {
                const float r = std::max(std::max(half_extents.x, half_extents.z), 1e-4f);
                const float cylinder_half = std::max(1e-4f, half_extents.y - r);
                volume = 3.1415926535f * r * r * (cylinder_half * 2.0f) +
                         (4.0f / 3.0f) * 3.1415926535f * r * r * r;
                break;
            }
            case RigidBodyShape::Box:
            default:
                volume = std::max(1e-6f, (half_extents.x * 2.0f) * (half_extents.y * 2.0f) * (half_extents.z * 2.0f));
                break;
        }
        rb.mass = std::max(0.001f, rb.density * volume);
    }

    JoltIntegration::JoltBodyDesc desc;
    desc.transform = box_pose;
    if (rb.motion_type == RigidBodyMotionType::Dynamic) rb.dynamic = true;
    if (rb.motion_type == RigidBodyMotionType::Static) rb.dynamic = false;
    desc.dynamic = rb.motion_type == RigidBodyMotionType::Dynamic;
    switch (rb.motion_type) {
        case RigidBodyMotionType::Kinematic:
            desc.motion_type = JoltIntegration::JoltMotionType::Kinematic;
            break;
        case RigidBodyMotionType::Static:
            desc.motion_type = JoltIntegration::JoltMotionType::Static;
            break;
        case RigidBodyMotionType::Dynamic:
        default:
            desc.motion_type = JoltIntegration::JoltMotionType::Dynamic;
            break;
    }
    desc.mass = rb.mass;
    desc.friction = rb.friction;
    desc.restitution = rb.restitution;
    desc.initial_linear_velocity = rb.initial_linear_velocity;
    desc.initial_angular_velocity = rb.initial_angular_velocity;
    desc.linear_damping = rb.linear_damping;
    desc.angular_damping = rb.angular_damping;
    desc.gravity_scale = rb.gravity_scale;
    desc.sleep_enabled = rb.sleep_enabled;

    switch (resolved_shape) {
        case RigidBodyShape::Sphere:
            desc.shape = JoltIntegration::JoltShapeType::Sphere;
            desc.radius = std::max({ half_extents.x, half_extents.y, half_extents.z });
            break;
        case RigidBodyShape::Capsule:
            desc.shape = JoltIntegration::JoltShapeType::Capsule;
            desc.radius = std::max(half_extents.x, half_extents.z);
            desc.half_height = std::max(1e-3f, half_extents.y - desc.radius);
            break;
        case RigidBodyShape::Box:
        default:
            desc.shape = JoltIntegration::JoltShapeType::Box;
            desc.half_extents = half_extents;
            break;
    }

    rb.handle = world_->createBody(desc);
    rb.created = (rb.handle != JoltIntegration::kInvalidBody);
    if (rb.created) {
        // B0 = the body's actual world pose right after creation (matches what we
        // seeded, minus scale). P0 = the captured REST pivot. Each step we re-apply
        // the body's motion-since-B0 to P0 so runtime motion never becomes the
        // authoring transform.
        rb.initial_body_xf = world_->getBodyTransform(rb.handle);
        rb.has_written = false;
    }
}

void RigidBodySystem::applyFluidCoupling(RigidBodyObject& rb, float dt) {
    if (!fluid_sampler_ || dt <= 0.0f) return;
    if (rb.motion_type != RigidBodyMotionType::Dynamic) return;

    constexpr float kPi = 3.1415926535f;
    const Vec3 he = rb.rest_half_extents;

    // Total shape volume (matches the auto-mass volume math so a fully submerged
    // body displaces exactly its own volume of fluid).
    float shape_volume;
    switch (rb.shape) {
        case RigidBodyShape::Sphere: {
            const float r = std::max({ he.x, he.y, he.z, 1e-4f });
            shape_volume = (4.0f / 3.0f) * kPi * r * r * r;
            break;
        }
        case RigidBodyShape::Capsule: {
            const float r = std::max(std::max(he.x, he.z), 1e-4f);
            const float ch = std::max(1e-4f, he.y - r);
            shape_volume = kPi * r * r * (ch * 2.0f) + (4.0f / 3.0f) * kPi * r * r * r;
            break;
        }
        case RigidBodyShape::Box:
        default:
            shape_volume = std::max(1e-6f, (he.x * 2.0f) * (he.y * 2.0f) * (he.z * 2.0f));
            break;
    }

    // Body CENTRE pose (rot+trans, no scale). For these primitive shapes the
    // shape centre == centre of mass. NOTE: Matrix4x4::operator*(Vec3) applies
    // only the 3x3 rotation (it drops translation), so world point of a local
    // offset = B_origin + B * local.
    const Matrix4x4 B = world_->getBodyTransform(rb.handle);
    const Vec3 B_origin(B.m[0][3], B.m[1][3], B.m[2][3]);

    // Sample lattice over the oriented bounds, cell-centred over an N-split of
    // [-h, h], keeping only points inside the actual shape.
    constexpr int N = 2;  // N^3 = 8 points
    Vec3 local_pts[N * N * N];
    int count = 0;
    auto insideShape = [&](const Vec3& lp) -> bool {
        switch (rb.shape) {
            case RigidBodyShape::Sphere: {
                const float r = std::max({ he.x, he.y, he.z, 1e-4f });
                return lp.length_squared() <= r * r;
            }
            case RigidBodyShape::Capsule: {
                const float r = std::max(std::max(he.x, he.z), 1e-4f);
                const float ch = std::max(0.0f, he.y - r);
                const float dy = std::max(0.0f, std::abs(lp.y) - ch);
                return (lp.x * lp.x + lp.z * lp.z + dy * dy) <= r * r;
            }
            case RigidBodyShape::Box:
            default:
                return true;  // lattice is already inside the box
        }
    };
    for (int iz = 0; iz < N; ++iz)
        for (int iy = 0; iy < N; ++iy)
            for (int ix = 0; ix < N; ++ix) {
                const Vec3 t((ix + 0.5f) / N * 2.0f - 1.0f,
                             (iy + 0.5f) / N * 2.0f - 1.0f,
                             (iz + 0.5f) / N * 2.0f - 1.0f);
                const Vec3 lp(t.x * he.x, t.y * he.y, t.z * he.z);
                if (insideShape(lp)) local_pts[count++] = lp;
            }
    if (count == 0) return;

    const float sub_volume = shape_volume / static_cast<float>(count);
    // Smooth the surface crossing over ~half the sample spacing so a point does
    // not snap fully wet/dry as the body bobs (kills buoyancy jitter).
    const float min_spacing = std::min({ he.x, he.y, he.z }) * (2.0f / N);
    const float band = std::max(1e-3f, 0.5f * min_spacing);

    // Drag stability cap: the summed drag impulse must not exceed mass*v_rel, or
    // explicit (forward-Euler) drag overshoots and injects energy. Spread the
    // mass/dt budget across all sample points => per-point coefficient ceiling.
    const float mass_eff = std::max(rb.mass, 1.0e-4f);
    const float k_drag_max = (mass_eff / dt) / static_cast<float>(count);

    // Anti-explosion: cap the NET buoyant acceleration at ~kMaxBuoyG * g, even
    // for absurd fluid_density or near-zero-mass bodies, so first contact with
    // the surface never launches the body off-screen in one step. Conservative:
    // scales against the FULL shape volume (worst case = fully submerged), so a
    // normal floater (body density >= rho_fluid/(kMaxBuoyG+1)) is never clamped.
    constexpr float kMaxBuoyG = 15.0f;
    float buoy_scale = rb.buoyancy_scale;
    if (rb.fluid_density * shape_volume > 1.0e-8f) {
        const float cap = (kMaxBuoyG + 1.0f) * mass_eff / (rb.fluid_density * shape_volume);
        if (cap < 1.0f) buoy_scale *= cap;
    }

    int submerged_pts = 0;
    float dbg_buoy_force_y = 0.0f;   // accumulated for diagnostics
    float dbg_drag_force_y = 0.0f;
    float dbg_min_sd = 1.0e30f;
    for (int p = 0; p < count; ++p) {
        const Vec3 wp = B_origin + B * local_pts[p];  // centre + rotated offset
        FluidSample s;
        if (!fluid_sampler_(wp, s) || !s.valid) continue;
        if (!std::isfinite(s.signed_distance)) continue;  // bad sample guard
        dbg_min_sd = std::min(dbg_min_sd, s.signed_distance);

        // Submerged weight 0..1 across the band centred on the surface
        // (signed_distance < 0 inside the fluid).
        float w = 0.5f - s.signed_distance / (2.0f * band);
        w = std::min(1.0f, std::max(0.0f, w));
        if (w <= 0.0f) continue;
        ++submerged_pts;

        const float vsub = sub_volume * w;

        // Buoyancy = -rho * Vsub * g (upward), at the point => restoring torque
        // that levels the body and lets it rock with the surface.
        const Vec3 fb = gravity_ * (-(rb.fluid_density * vsub * buoy_scale));
        world_->addForceAtPoint(rb.handle, fb, wp, /*wake=*/true);
        dbg_buoy_force_y += fb.y;

        // Drag = -k * v_rel, with v_rel the body's velocity AT the point minus
        // the fluid velocity (so rotation is damped through the lever arm too).
        Vec3 fvel = s.velocity;
        if (!std::isfinite(fvel.x) || !std::isfinite(fvel.y) || !std::isfinite(fvel.z))
            fvel = Vec3(0.0f, 0.0f, 0.0f);
        const Vec3 v_rel = world_->getBodyPointVelocity(rb.handle, wp) - fvel;
        float k = rb.fluid_drag * rb.fluid_density * vsub;
        k = std::min(k, k_drag_max);
        if (k > 0.0f) {
            const Vec3 fd = v_rel * (-k);
            world_->addForceAtPoint(rb.handle, fd, wp, /*wake=*/true);
            dbg_drag_force_y += fd.y;
        }
    }

    // Publish diagnostics (accel = force / mass) for the UI readout.
    rb.dbg_coupled = true;
    rb.dbg_sample_count = count;
    rb.dbg_submerged_pts = submerged_pts;
    rb.dbg_min_sd = (dbg_min_sd < 1.0e29f) ? dbg_min_sd : 0.0f;
    rb.dbg_buoy_accel_y = dbg_buoy_force_y / mass_eff;
    rb.dbg_drag_accel_y = dbg_drag_force_y / mass_eff;
    rb.dbg_vel_y = world_->getBodyLinearVelocity(rb.handle).y;
    rb.dbg_body_density = (shape_volume > 1.0e-8f) ? (mass_eff / shape_volume) : 0.0f;

    // Extra rotational damping knob on top of the per-point drag. Capped against
    // a conservative inertia estimate (0.4*m*R^2) so it cannot reverse the spin.
    if (submerged_pts > 0 && rb.fluid_angular_drag > 0.0f && rb.mass > 0.0f) {
        const float frac = static_cast<float>(submerged_pts) / static_cast<float>(count);
        const float Rchar = std::max({ he.x, he.y, he.z, 1e-4f });
        float k = rb.fluid_angular_drag * rb.fluid_density * (shape_volume * frac) * (Rchar * Rchar);
        const float k_max = 0.4f * rb.mass * Rchar * Rchar / dt;
        k = std::min(k, k_max);
        if (k > 0.0f) {
            const Vec3 omega = world_->getBodyAngularVelocity(rb.handle);
            world_->addTorque(rb.handle, omega * (-k), /*wake=*/false);
        }
    }
}

void RigidBodySystem::step(const SimulationContext& ctx) {
    if (!enabled_ || !bodies_ || bodies_->empty()) return;
    if (!ensureWorld()) return;

    // Create any bodies that aren't live yet (new this frame or after a reset).
    bool any_new = false;
    for (RigidBodyObject& rb : *bodies_) {
        if (rb.enabled && !rb.created) {
            ensureBodyCreated(rb);
            any_new = any_new || rb.created;
        }
    }
    if (any_new) world_->optimizeBroadPhase();

    // Advance. Jolt wants dt/collision_steps <= 1/60, so split big ticks.
    float dt = ctx.dt;
    if (dt <= 0.0f) return;
    if (dt > 0.25f) dt = 0.25f;  // guard against huge hitches
    const int collision_steps = std::max(1, (int)std::ceil(dt / (1.0f / 60.0f)));

    // Fluid coupling (buoyancy + drag) — applied as external forces BEFORE the
    // step. Reads the fluid state left by the previous frame (this system runs
    // at order 50, before the fluid solver at 100), a standard one-frame lag.
    bool any_coupled = false;
    if (fluid_sampler_) {
        for (const RigidBodyObject& rb : *bodies_) {
            if (rb.enabled && rb.created && rb.dynamic &&
                rb.motion_type == RigidBodyMotionType::Dynamic && rb.fluid_coupling_enabled) {
                any_coupled = true;
                break;
            }
        }
    }
    if (any_coupled) {
        if (fluid_prepare_) fluid_prepare_();
        for (RigidBodyObject& rb : *bodies_) {
            if (rb.enabled && rb.created && rb.dynamic &&
                rb.motion_type == RigidBodyMotionType::Dynamic && rb.fluid_coupling_enabled) {
                applyFluidCoupling(rb, dt);
            }
        }
    }

    world_->update(dt, collision_steps);

    // Write rigid motion back onto each dynamic source object:
    //   P(t) = B(t) * inverse(B0) * P0
    if (!pivot_setter_) return;
    for (RigidBodyObject& rb : *bodies_) {
        if (!rb.created || !rb.dynamic || !rb.enabled) continue;
        Matrix4x4 Bt = world_->getBodyTransform(rb.handle);
        Matrix4x4 Pt = Bt * rb.initial_body_xf.inverse() * rb.initial_pivot;
        pivot_setter_(rb.source_name, Pt);
        rb.last_written_pivot = Pt;
        rb.has_written = true;
    }
}

void RigidBodySystem::resetRuntime(bool restore_rest_pose) {
    // Restore each driven object to the pose it had when its body was created
    // (P0) BEFORE dropping the Jolt bodies — otherwise the object stays frozen
    // wherever it fell and the next respawn would seed from that fallen pose.
    if (restore_rest_pose && bodies_ && pivot_setter_) {
        for (RigidBodyObject& rb : *bodies_) {
            if (rb.rest_captured && rb.dynamic) pivot_setter_(rb.source_name, rb.initial_pivot);
        }
    }
    if (world_ && world_->isInitialized()) world_->clearBodies();
    if (bodies_) {
        for (RigidBodyObject& rb : *bodies_) {
            rb.created = false;
            rb.handle = JoltIntegration::kInvalidBody;
            rb.has_written = false;
        }
    }
}

} // namespace RayTrophiSim
