#include "RigidBodySystem.h"

#include "JoltWorld.h"  // pure RT-typed wrapper (no Jolt headers leak in here)

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace RayTrophiSim {

namespace {
// Largest s in [0,1] such that |v0 + s*dv| <= v_allow. Returns 1 when the change
// is already within bounds, a smaller factor (down to 0) when applying it fully
// would push the magnitude past v_allow. Makes the fluid coupling dissipative: a
// change that LOWERS the magnitude (drag) is never scaled (s stays 1), only one
// that would raise speed/spin past the allowed ceiling gets reined in.
inline float energyLimitScale(const Vec3& v0, const Vec3& dv, float v_allow) {
    const float v_allow2 = v_allow * v_allow;
    if ((v0 + dv).length_squared() <= v_allow2) return 1.0f;
    const float a = dv.length_squared();
    if (a <= 1.0e-12f) return 0.0f;
    const float b = 2.0f * v0.dot(dv);
    const float c = v0.length_squared() - v_allow2;
    const float disc = b * b - 4.0f * a * c;
    if (disc <= 0.0f) return 0.0f;
    const float s = (-b + std::sqrt(disc)) / (2.0f * a);
    return std::min(1.0f, std::max(0.0f, s));
}
} // namespace

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

void RigidBodySystem::setContactEventsEnabled(bool e) {
    contact_events_enabled_ = e;
    if (world_ && world_->isInitialized()) world_->setContactCaptureEnabled(e);
    if (!e) contact_events_.clear();
}

void RigidBodySystem::ensureBodyCreated(RigidBodyObject& rb) {
    if (rb.created || rb.source_name.empty()) return;
    // Soft / Cloth bodies take the deformable path (Jolt soft body built from the
    // source mesh graph); rigid bodies use the box/shape path below.
    if (rb.kind != BodyKind::Rigid) { ensureSoftBodyCreated(rb); return; }
    if (!shape_resolver_) return;

    Matrix4x4 box_pose;
    Vec3 half_extents;
    RigidBodyShape resolved_shape = rb.shape;
    if (!shape_resolver_(rb, box_pose, half_extents, resolved_shape)) {
        return;  // object not resolvable yet (e.g., surface cache not ready)
    }
    rb.shape = resolved_shape;

    Matrix4x4 pivot = Matrix4x4::identity();
    const bool have_pivot = pivot_getter_ && pivot_getter_(rb.source_name, pivot);

    // Spawn pose = the object's CURRENT pivot at creation time. A body is only
    // (re)created when it actually needs to spawn — after add / reset / a user
    // edit — never mid-sim (step() skips already-created bodies), so reading the
    // live pivot here cannot drift to a simulated pose. This is what makes a gizmo
    // move BEFORE the first play actually relocate the spawn point: previously the
    // pivot was frozen at add-time (rest_captured), so the body was created at the
    // live OBB but the write-back mapped it back to the stale add-time pose and the
    // object teleported to where it was first created on the very first step.
    // (Only overwrite when the getter actually resolved a pose, so a transient
    // lookup miss can't reset a valid spawn to the origin.)
    if (have_pivot) rb.initial_pivot = pivot;
    if (!rb.rest_captured) {
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
        case RigidBodyShape::Mesh: {
            // Exact mesh collision: pull the source mesh's welded world-space rest
            // triangles (same data the soft path uses) and hand them to Jolt. A
            // STATIC body becomes a triangle MeshShape (true boundary, not an OBB);
            // a Dynamic/Kinematic body becomes a ConvexHull of the same points.
            // The verts are world-space at rest, so the body is created at identity
            // (B0 = identity) — the rigid bake delta D = B(t)*inv(B0) then maps a
            // moving body's motion straight onto the rest mesh, identical to soft.
            std::vector<Vec3> verts;
            std::vector<uint32_t> indices;
            if (!soft_resolver_ || !soft_resolver_(rb, verts, indices) ||
                verts.size() < 3 || indices.size() < 3) {
                // Source mesh not ready yet (e.g. surface cache); try again next tick.
                return;
            }
            desc.shape = JoltIntegration::JoltShapeType::Mesh;
            desc.mesh_vertices = std::move(verts);
            desc.mesh_indices = std::move(indices);
            desc.half_extents = half_extents;      // fluid-coupling volume fallback
            desc.transform = Matrix4x4::identity(); // world-space verts carry the pose
            break;
        }
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
        rb.smoothed_fluid_vel = Vec3(0.0f, 0.0f, 0.0f);
        rb.fluid_vel_primed = false;
    }
}

void RigidBodySystem::ensureSoftBodyCreated(RigidBodyObject& rb) {
    if (rb.created || rb.source_name.empty() || !soft_resolver_) return;

    std::vector<Vec3> verts;
    std::vector<uint32_t> indices;
    if (!soft_resolver_(rb, verts, indices)) return;  // source mesh not ready yet
    if (verts.empty() || indices.size() < 3) return;

    // Resolve pins: a rest vertex is pinned if it falls inside any enabled pin
    // region (world-space sphere). Built only when at least one region exists so the
    // common (unpinned) case stays free of an extra allocation.
    std::vector<uint8_t> pinned;
    int pinned_count = 0;
    for (const SoftPinRegion& reg : rb.soft_pins) {
        if (!reg.enabled || reg.radius <= 0.0f) continue;
        if (pinned.empty()) pinned.assign(verts.size(), 0);
        const float r2 = reg.radius * reg.radius;
        for (std::size_t vi = 0; vi < verts.size(); ++vi) {
            if (pinned[vi]) continue;
            if ((verts[vi] - reg.center).length_squared() <= r2) {
                pinned[vi] = 1;
                ++pinned_count;
            }
        }
    }
    rb.dbg_pinned_count = pinned_count;

    JoltIntegration::JoltSoftBodyDesc desc;
    desc.vertices = std::move(verts);
    desc.indices = std::move(indices);
    desc.vertex_pinned = std::move(pinned);
    desc.total_mass = std::max(1.0e-3f, rb.soft_mass);
    // One authored knob folds two Jolt inputs: explicit compliance wins; otherwise
    // derive it from stiffness (1 = stiff => ~0 compliance, 0 = floppy => large).
    desc.compliance = (rb.soft_compliance > 0.0f)
                          ? rb.soft_compliance
                          : (1.0f - std::min(1.0f, std::max(0.0f, rb.soft_stiffness))) * 1.0e-3f;
    desc.pressure = (rb.kind == BodyKind::Cloth) ? 0.0f : rb.soft_pressure;
    desc.damping = rb.soft_damping;
    desc.num_iterations = rb.soft_iterations;
    desc.friction = rb.soft_friction;
    desc.restitution = rb.soft_restitution;
    desc.gravity_factor = rb.soft_gravity_factor;
    desc.vertex_radius = rb.soft_vertex_radius;
    desc.two_sided = (rb.kind == BodyKind::Cloth) ? rb.soft_two_sided : false;

    rb.handle = world_->createSoftBody(desc);
    rb.created = (rb.handle != JoltIntegration::kInvalidBody);
    if (rb.created) {
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
    const float mass_eff = std::max(rb.mass, 1.0e-4f);

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

    // --- Pass 1: sample the fluid once per point, keep only the submerged ones.
    // (The sampler is the expensive call, so we never query a point twice.)
    struct PointHit { Vec3 wp; float w; Vec3 fvel; };
    PointHit hits[N * N * N];
    int submerged_pts = 0;
    Vec3 sum_fvel(0.0f, 0.0f, 0.0f);
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

        Vec3 fvel = s.velocity;
        if (!std::isfinite(fvel.x) || !std::isfinite(fvel.y) || !std::isfinite(fvel.z))
            fvel = Vec3(0.0f, 0.0f, 0.0f);
        sum_fvel += fvel;
        hits[submerged_pts++] = { wp, w, fvel };
    }

    // Drag reads ONE smoothed, speed-clamped fluid velocity for the whole body
    // rather than the raw per-point splash field. The body stamps its own motion
    // into the grid (solid_vel) and splashes the surface; feeding that straight
    // back as drag created a splash->drag->fling feedback loop that hurled light
    // bodies sideways. A per-body average + EMA + speed cap breaks the loop while
    // the per-POINT body velocity (below) still carries rotational drag.
    Vec3 avg_fvel(0.0f, 0.0f, 0.0f);
    if (submerged_pts > 0) avg_fvel = sum_fvel / static_cast<float>(submerged_pts);
    const float vel_alpha = rb.fluid_vel_primed ? 0.35f : 1.0f;
    rb.smoothed_fluid_vel = rb.smoothed_fluid_vel * (1.0f - vel_alpha) + avg_fvel * vel_alpha;
    rb.fluid_vel_primed = true;
    const float vmax = std::max(0.0f, rb.fluid_max_coupling_speed);
    if (vmax > 0.0f) {
        const float sp = rb.smoothed_fluid_vel.length();
        if (sp > vmax) rb.smoothed_fluid_vel = rb.smoothed_fluid_vel * (vmax / sp);
    }
    const Vec3 fluid_vel = rb.smoothed_fluid_vel;

    // Drag stability cap: the summed drag impulse must not exceed mass*v_rel, or
    // explicit (forward-Euler) drag overshoots and injects energy. Spread the
    // mass/dt budget across the SUBMERGED points only — the old per-TOTAL-point
    // budget left a body that touches the surface with just one or two points
    // almost undamped, so it skipped off the surface instead of shedding its
    // impact energy. Per-submerged-point ceiling => the full mass/dt budget is
    // available even at first, shallow contact, while staying energy-stable.
    const float k_drag_max = (submerged_pts > 0)
        ? (mass_eff / dt) / static_cast<float>(submerged_pts)
        : 0.0f;

    // --- Pass 2: compute buoyancy + drag at each submerged point, accumulate the
    // NET linear force, then apply (after an energy-limiting clamp, below).
    Vec3 pt_pos[N * N * N];
    Vec3 pt_force[N * N * N];
    Vec3 F_total(0.0f, 0.0f, 0.0f);
    float dbg_buoy_force_y = 0.0f;   // accumulated for diagnostics
    float dbg_drag_force_y = 0.0f;
    for (int p = 0; p < submerged_pts; ++p) {
        const Vec3& wp = hits[p].wp;
        const float vsub = sub_volume * hits[p].w;

        // Buoyancy = -rho * Vsub * g (upward), at the point => restoring torque
        // that levels the body and lets it rock with the surface.
        const Vec3 fb = gravity_ * (-(rb.fluid_density * vsub * buoy_scale));
        dbg_buoy_force_y += fb.y;

        // Drag = -k * v_rel, with v_rel the body's velocity AT the point minus
        // the smoothed/clamped fluid velocity (so rotation is damped through the
        // lever arm too, while spatial splash noise is filtered out).
        const Vec3 v_rel = world_->getBodyPointVelocity(rb.handle, wp) - fluid_vel;
        float k = rb.fluid_drag * rb.fluid_density * vsub;
        // Form / slam drag ~ 0.5*Cd*rho*A*v^2. Folding the |v_rel| factor into k
        // keeps the same -k*v_rel application but makes the resistance grow with
        // speed (this is what actually stops a body bouncing off the water).
        // Per-point cross-section ~ vsub^(2/3).
        if (rb.fluid_quadratic_drag > 0.0f) {
            const float len = std::cbrt(std::max(vsub, 1e-12f));   // characteristic length
            const float area = len * len;                          // ~ vsub^(2/3)
            k += rb.fluid_quadratic_drag * rb.fluid_density * area * v_rel.length();
        }
        k = std::min(k, k_drag_max);
        Vec3 fd(0.0f, 0.0f, 0.0f);
        if (k > 0.0f) {
            fd = v_rel * (-k);
            dbg_drag_force_y += fd.y;
        }

        pt_pos[p] = wp;
        pt_force[p] = fb + fd;
        F_total += pt_force[p];
    }

    // Split the coupling into a NET linear force at the centre of mass + a NET
    // torque, and energy-clamp each independently. Applied as explicit forces,
    // both buoyancy (a stiff restoring spring F ~ -rho*A*g*depth) and the
    // off-centre buoyancy righting torque overshoot and pump energy — that is why
    // the body first flung linearly and now SPINS away on contact. Water
    // entry/exit is dissipative, so the coupling may not RAISE the body's linear
    // OR angular speed past max(incoming speed, a small size-scaled floor).
    if (submerged_pts > 0) {
        const float g_mag = std::max(1e-4f, gravity_.length());
        const float Rchar = std::max({ he.x, he.y, he.z, 1e-3f });

        // Net torque about the centre of mass (shape centre == COM for these
        // primitives, but use the real COM so an offset-inertia body is correct).
        // NOTE: do NOT temporally filter (EMA) this torque. It is a feedback term
        // (restoring buoyancy + velocity-opposing drag); low-passing it adds phase
        // lag, which turns the damping component into NEGATIVE damping and pumps a
        // plank's roll up instead of bleeding it. Splash jitter is handled by the
        // drag damping itself, not by lagging the torque.
        const Vec3 com = world_->getBodyCenterOfMass(rb.handle);
        Vec3 T_total(0.0f, 0.0f, 0.0f);
        for (int p = 0; p < submerged_pts; ++p)
            T_total += Vec3::cross(pt_pos[p] - com, pt_force[p]);

        // --- Linear clamp, applied to the body's velocity RELATIVE TO the ambient
        // fluid (wave) flow — NOT its absolute speed. Drag legitimately accelerates
        // a body toward the water it sits in (this is how waves drag/carry a ship),
        // so clamping absolute speed throttled wave-dragging at v_floor. Clamping
        // the RELATIVE speed still blocks energy injection (the body can approach
        // the flow but not overshoot it) while letting waves carry it at full flow
        // speed. Buoyancy fling stays bounded because the wave's VERTICAL component
        // is ~0, so vertically this is still ~max(|v0|, v_floor).
        const Vec3 v0 = world_->getBodyLinearVelocity(rb.handle);
        const Vec3 v_rel0 = v0 - fluid_vel;
        const Vec3 dv = F_total * (dt / mass_eff);
        const float v_floor = std::min(8.0f, std::max(1.5f, std::sqrt(2.0f * g_mag * Rchar)));
        const float lin_scale = energyLimitScale(v_rel0, dv, std::max(v_rel0.length(), v_floor));

        // --- Angular clamp. Use the SMALLEST principal inertia of the box (the
        // long-axis "roll" moment for a plank). A scalar 0.4*m*R^2 with R=longest
        // extent grossly OVER-estimates the roll inertia of an elongated body, so
        // domega read tiny and the clamp never engaged — the plank's roll built up
        // unchecked. The smallest principal moment makes domega (and the cap) safe
        // for every axis (real inertia >= this), so the clamp can't be defeated by
        // anisotropy. domega = T*dt/I.
        const float third = 1.0f / 3.0f;
        const float ix = third * mass_eff * (he.y * he.y + he.z * he.z);
        const float iy = third * mass_eff * (he.x * he.x + he.z * he.z);
        const float iz = third * mass_eff * (he.x * he.x + he.y * he.y);
        const float inertia = std::max(1e-6f, std::min({ ix, iy, iz, 0.4f * mass_eff * Rchar * Rchar }));
        const Vec3 w0 = world_->getBodyAngularVelocity(rb.handle);
        const Vec3 dw = T_total * (dt / inertia);
        constexpr float kOmegaFloor = 3.0f;  // rad/s — ~0.5 rev/s righting allowance
        const float ang_scale = energyLimitScale(w0, dw, std::max(w0.length(), kOmegaFloor));

        world_->addForce(rb.handle, F_total * lin_scale, /*wake=*/true);
        world_->addTorque(rb.handle, T_total * ang_scale, /*wake=*/true);
        dbg_buoy_force_y *= lin_scale;
        dbg_drag_force_y *= lin_scale;
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

    // Extra rotational damping knob on top of the per-point drag. The cap MUST use
    // the smallest principal inertia: with a scalar 0.4*m*R^2 (R=longest extent)
    // the cap was far larger than a plank's true roll inertia, so the explicit
    // damping torque over-corrected and REVERSED the spin each step — pumping
    // energy and accelerating the roll instead of bleeding it. k <= I_min/dt
    // guarantees |domega| <= |omega| on every axis (real inertia >= I_min).
    if (submerged_pts > 0 && rb.fluid_angular_drag > 0.0f && rb.mass > 0.0f) {
        const float frac = static_cast<float>(submerged_pts) / static_cast<float>(count);
        const float Rchar = std::max({ he.x, he.y, he.z, 1e-4f });
        const float third = 1.0f / 3.0f;
        const float ix = third * rb.mass * (he.y * he.y + he.z * he.z);
        const float iy = third * rb.mass * (he.x * he.x + he.z * he.z);
        const float iz = third * rb.mass * (he.x * he.x + he.y * he.y);
        const float inertia_min = std::max(1e-6f, std::min({ ix, iy, iz, 0.4f * rb.mass * Rchar * Rchar }));
        float k = rb.fluid_angular_drag * rb.fluid_density * (shape_volume * frac) * (Rchar * Rchar);
        const float k_max = inertia_min / dt;
        k = std::min(k, k_max);
        if (k > 0.0f) {
            const Vec3 omega = world_->getBodyAngularVelocity(rb.handle);
            world_->addTorque(rb.handle, omega * (-k), /*wake=*/false);
        }
    }
}

void RigidBodySystem::applyForceFields(const SimulationContext& ctx, float dt) {
    if (!bodies_ || dt <= 0.0f) return;
    const SimulationForceFieldSnapshot* snap = ctx.force_snapshot;
    if (!snap || snap->empty()) return;
    const float time = ctx.time_seconds;

    // Per-soft-body scratch reused across bodies (force fields evaluate per vertex).
    static std::vector<Vec3> sb_pos;
    static std::vector<Vec3> sb_vel;
    static std::vector<Vec3> sb_dv;

    for (RigidBodyObject& rb : *bodies_) {
        if (!rb.enabled || !rb.created || !rb.force_field_enabled) continue;
        const float scale = rb.force_field_scale;
        if (scale == 0.0f) continue;

        if (rb.kind == BodyKind::Rigid) {
            // Only push dynamic rigids; static/kinematic are driven elsewhere.
            if (rb.motion_type != RigidBodyMotionType::Dynamic || !rb.dynamic) continue;
            const Vec3 com = world_->getBodyCenterOfMass(rb.handle);
            const Vec3 vel = world_->getBodyLinearVelocity(rb.handle);
            // The snapshot returns an ACCELERATION (force per unit mass), matching the
            // particle/fluid consumers; convert to a force at the COM via F = m*a.
            const Vec3 accel = snap->evaluateAt(com, time, vel, SimulationSystemKind::RigidBody);
            const Vec3 force = accel * (rb.mass * scale);
            if (force.length_squared() > 0.0f)
                world_->addForce(rb.handle, force, /*wake=*/true);
        } else {
            // Soft body / cloth: evaluate the field at every (unpinned) vertex and add
            // the integrated velocity push dv = a*dt. Pinned vertices are skipped by
            // the Jolt wrapper (invMass == 0).
            if (!world_->getSoftBodyVertices(rb.handle, sb_pos) || sb_pos.empty()) continue;
            world_->getSoftBodyVertexVelocities(rb.handle, sb_vel);
            const bool have_vel = (sb_vel.size() == sb_pos.size());
            sb_dv.assign(sb_pos.size(), Vec3(0.0f, 0.0f, 0.0f));
            bool any = false;
            for (std::size_t i = 0; i < sb_pos.size(); ++i) {
                const Vec3 v = have_vel ? sb_vel[i] : Vec3(0.0f, 0.0f, 0.0f);
                const Vec3 accel = snap->evaluateAt(sb_pos[i], time, v, SimulationSystemKind::Cloth);
                const Vec3 dv = accel * (dt * scale);
                sb_dv[i] = dv;
                any = any || (dv.length_squared() > 0.0f);
            }
            if (any) world_->addSoftBodyVertexVelocities(rb.handle, sb_dv);
        }
    }
}

void RigidBodySystem::step(const SimulationContext& ctx) {
    if (!enabled_ || !bodies_ || bodies_->empty()) return;
    if (!ensureWorld()) return;
    // Keep the contact listener's capture state in sync (the world may have been
    // (re)created since the flag was last set).
    world_->setContactCaptureEnabled(contact_events_enabled_);

    // Create any bodies that aren't live yet (new this frame or after a reset).
    // Track which SOFT bodies were freshly created so only those get resumed below
    // (an already-live soft body must keep its own state, not be teleported back).
    bool any_new = false;
    std::vector<RigidBodyObject*> resumed_soft;
    for (RigidBodyObject& rb : *bodies_) {
        if (rb.enabled && !rb.created) {
            ensureBodyCreated(rb);
            any_new = any_new || rb.created;
            if (rb.created && rb.kind != BodyKind::Rigid) resumed_soft.push_back(&rb);
        }
    }
    if (any_new) world_->optimizeBroadPhase();

    // Resume soft bodies that were just (re)created from REST onto the cached
    // deformed frame the timeline last replayed, so playing past the RAM cache
    // continues the motion instead of restarting it from rest. The rest-length
    // constraints were already built from the rest mesh at creation; this only moves
    // the live particle state. The provider returns false on a first bake / when the
    // resume frame isn't cached, leaving the body at its (correct) rest.
    if (soft_resume_ && !resumed_soft.empty()) {
        std::vector<Vec3> resume_pos, resume_vel;
        for (RigidBodyObject* rb : resumed_soft) {
            if (soft_resume_(rb->source_name, resume_pos, resume_vel) && !resume_pos.empty()) {
                world_->setSoftBodyVertices(rb->handle, resume_pos,
                                            resume_vel.empty() ? nullptr : &resume_vel);
                rb->has_written = true;
            }
        }
    }

    // Launch freshly-broken fracture shards: when a breakable group shatters, the
    // shards flip Static->Dynamic (recreated above) and carry a one-shot blast
    // velocity set at break time. Apply it now, right after (re)creation.
    for (RigidBodyObject& rb : *bodies_) {
        if (rb.created && rb.has_pending_launch && rb.kind == BodyKind::Rigid) {
            world_->setBodyLinearVelocity(rb.handle, rb.pending_launch_velocity);
            rb.has_pending_launch = false;
            rb.pending_launch_velocity = Vec3(0.0f, 0.0f, 0.0f);
        }
    }

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
            if (rb.enabled && rb.created && rb.dynamic && rb.kind == BodyKind::Rigid &&
                rb.motion_type == RigidBodyMotionType::Dynamic && rb.fluid_coupling_enabled) {
                any_coupled = true;
                break;
            }
        }
    }
    if (any_coupled) {
        if (fluid_prepare_) fluid_prepare_();
        for (RigidBodyObject& rb : *bodies_) {
            if (rb.enabled && rb.created && rb.dynamic && rb.kind == BodyKind::Rigid &&
                rb.motion_type == RigidBodyMotionType::Dynamic && rb.fluid_coupling_enabled) {
                applyFluidCoupling(rb, dt);
            }
        }
    }

    // Force fields drive every body kind (rigid + soft/cloth). Applied as external
    // forces/velocity pushes BEFORE the step, like fluid coupling. Cheap no-op when
    // the scene has no active fields.
    applyForceFields(ctx, dt);

    world_->update(dt, collision_steps);

    // Drain impact events (Faz 1 fracture foundation): map Jolt body handles back
    // to scene-node names so consumers (fracture trigger, UI, audio) can threshold
    // a hard hit. Refilled every step; empty when capture is disabled.
    if (contact_events_enabled_) {
        std::vector<JoltIntegration::ContactEvent> raw;
        world_->drainContactEvents(raw);
        contact_events_.clear();
        auto nameForHandle = [&](JoltIntegration::JoltBodyHandle h) -> std::string {
            if (h == JoltIntegration::kInvalidBody) return std::string();
            for (const RigidBodyObject& rb : *bodies_)
                if (rb.created && rb.handle == h) return rb.source_name;
            return std::string();
        };
        contact_events_.reserve(raw.size());
        for (const auto& e : raw) {
            RigidContactEvent ev;
            ev.source_a = nameForHandle(e.body_a);
            ev.source_b = nameForHandle(e.body_b);
            ev.point = e.point;
            ev.normal = e.normal;
            ev.closing_speed = e.closing_speed;
            ev.impulse = e.impulse;
            ev.is_new = e.is_new;
            contact_events_.push_back(std::move(ev));
        }
    }

    // Write deformed soft-body geometry back onto the source meshes. Independent
    // of the rigid pivot path below — soft bodies own their vertices, not a pose.
    if (soft_writer_) {
        std::vector<Vec3> deformed;
        for (RigidBodyObject& rb : *bodies_) {
            if (!rb.created || rb.kind == BodyKind::Rigid || !rb.enabled) continue;
            if (world_->getSoftBodyVertices(rb.handle, deformed) && !deformed.empty()) {
                soft_writer_(rb.source_name, deformed);
                rb.has_written = true;
            }
        }
    }

    // Write rigid motion back onto each dynamic source object by BAKING the body's
    // world-space rigid delta D = B(t)*inv(B0) into the mesh vertices (rigid_baker_),
    // NOT by moving the transform handle. Moving an imported/non-TRS object's
    // transform corrupted it in the renderer (grew / Y-flipped / lost geometry from
    // frame 0); vertex baking is exactly how soft bodies render imported meshes
    // correctly. P(t) is still tracked for the user-edit / fluid bookkeeping.
    if (!rigid_baker_) return;
    for (RigidBodyObject& rb : *bodies_) {
        if (!rb.created || !rb.dynamic || !rb.enabled || rb.kind != BodyKind::Rigid) continue;
        Matrix4x4 Bt = world_->getBodyTransform(rb.handle);
        Matrix4x4 D = Bt * rb.initial_body_xf.inverse();
        rigid_baker_(rb.source_name, D);
        rb.last_written_pivot = D * rb.initial_pivot;  // P(t) for bookkeeping only
        rb.has_written = true;
    }
}

void RigidBodySystem::resetRuntime(bool restore_rest_pose) {
    // Restore each driven object to the pose it had when its body was created
    // (P0) BEFORE dropping the Jolt bodies — otherwise the object stays frozen
    // wherever it fell and the next respawn would seed from that fallen pose.
    if (restore_rest_pose && bodies_ && rigid_baker_) {
        for (RigidBodyObject& rb : *bodies_) {
            // Bake an IDENTITY delta to restore the source mesh to its rest pose
            // (the baker is keyed off the cached rest verts, so identity == rest).
            if (rb.rest_captured && rb.dynamic && rb.kind == BodyKind::Rigid)
                rigid_baker_(rb.source_name, Matrix4x4::identity());
        }
    }
    // Soft bodies own their geometry, not a pivot — clear any deformation so the
    // next play/respawn rebuilds from the clean rest mesh.
    if (restore_rest_pose && bodies_ && soft_reset_) {
        for (RigidBodyObject& rb : *bodies_) {
            if (rb.kind != BodyKind::Rigid) soft_reset_(rb.source_name);
        }
    }
    if (world_ && world_->isInitialized()) world_->clearBodies();
    if (bodies_) {
        for (RigidBodyObject& rb : *bodies_) {
            rb.created = false;
            rb.handle = JoltIntegration::kInvalidBody;
            rb.has_written = false;
            rb.rest_captured = false;    // force fresh rest capture on next create
            rb.smoothed_fluid_vel = Vec3(0.0f, 0.0f, 0.0f);
            rb.fluid_vel_primed = false;
        }
    }
}

bool RigidBodySystem::destroyBodyForNode(const std::string& node) {
    if (!bodies_ || node.empty()) return false;
    bool found = false;
    const bool world_live = world_ && world_->isInitialized();
    for (RigidBodyObject& rb : *bodies_) {
        if (rb.source_name != node) continue;
        found = true;
        if (rb.created && world_live && rb.handle != JoltIntegration::kInvalidBody) {
            world_->removeBody(rb.handle);
        }
        rb.created = false;
        rb.handle = JoltIntegration::kInvalidBody;
        rb.has_written = false;
    }
    return found;
}

void RigidBodySystem::captureFrameState(std::vector<RigidBodyFrameState>& out) const {
    out.clear();
    if (!bodies_) return;
    out.reserve(bodies_->size());
    const bool world_live = world_ && world_->isInitialized();
    for (const RigidBodyObject& rb : *bodies_) {
        if (rb.motion_type != RigidBodyMotionType::Dynamic || !rb.dynamic || rb.kind != BodyKind::Rigid) continue;
        RigidBodyFrameState s;
        s.source_name = rb.source_name;
        s.pivot = rb.has_written ? rb.last_written_pivot : rb.initial_pivot;
        s.valid = rb.has_written;
        if (rb.created && world_live) {
            s.body_xf = world_->getBodyTransform(rb.handle);
            s.lin_vel = world_->getBodyLinearVelocity(rb.handle);
            s.ang_vel = world_->getBodyAngularVelocity(rb.handle);
        } else {
            s.body_xf = rb.initial_body_xf;
        }
        out.push_back(std::move(s));
    }
}

bool RigidBodySystem::restoreFrameState(const std::vector<RigidBodyFrameState>& in) {
    if (!enabled_ || !bodies_) return false;
    if (!ensureWorld()) return false;

    // Bring any not-yet-live RIGID bodies up so we can both render this frame AND
    // let a forward resume keep stepping. After a resetRuntime the source objects
    // sit at their rest pose, so creation reproduces the same B0 the bake used; the
    // explicit setBodyTransform below then snaps each body to its cached pose.
    //
    // SOFT bodies are deliberately NOT created here. Their cache is mesh-resident
    // (the deformed mesh is replayed by the scene), not a Jolt pose. Creating the
    // soft Jolt body during a replay would freeze it at whatever mesh shape is
    // current — the REST pose right after a loop-back reset — so a live resume
    // would step it from rest and re-animate the whole motion from the start (the
    // "body re-sims from frame 0 while fluid is fine" bug). Leaving it uncreated
    // lets the live resume rebuild it. The rebuild seeds from the REST mesh (so the
    // edge rest-lengths are correct), then step() teleports it onto the last
    // replayed frame via the SoftResumeProvider (setSoftBodyVertices) so playing
    // PAST the cache continues the motion instead of restarting it from rest.
    bool any_new = false;
    for (RigidBodyObject& rb : *bodies_) {
        if (rb.enabled && !rb.created && rb.kind == BodyKind::Rigid) {
            ensureBodyCreated(rb);
            any_new = any_new || rb.created;
        }
    }
    if (any_new) world_->optimizeBroadPhase();

    const bool world_live = world_ && world_->isInitialized();
    for (RigidBodyObject& rb : *bodies_) {
        if (rb.motion_type != RigidBodyMotionType::Dynamic || !rb.dynamic || rb.kind != BodyKind::Rigid) continue;
        const RigidBodyFrameState* s = nullptr;
        for (const auto& cand : in) {
            if (cand.source_name == rb.source_name) { s = &cand; break; }
        }
        if (!s) continue;
        if (rb.created && world_live) {
            world_->setBodyTransform(rb.handle, s->body_xf);
            world_->setBodyLinearVelocity(rb.handle, s->lin_vel);
            world_->setBodyAngularVelocity(rb.handle, s->ang_vel);
        }
        // Re-prime the drag velocity filter on resume (transient per-step state).
        rb.smoothed_fluid_vel = Vec3(0.0f, 0.0f, 0.0f);
        rb.fluid_vel_primed = false;
        // Bake the cached pose into the mesh (delta = cached body_xf * inv(B0)),
        // matching the live write-back path. Falls back to identity (rest) when the
        // frame predates any motion.
        if (rigid_baker_ && s->valid) {
            const Matrix4x4 D = s->body_xf * rb.initial_body_xf.inverse();
            rigid_baker_(rb.source_name, D);
            rb.last_written_pivot = D * rb.initial_pivot;
            rb.has_written = true;
        }
    }
    return true;
}

} // namespace RayTrophiSim
