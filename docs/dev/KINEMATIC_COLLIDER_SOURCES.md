# Kinematic Collider Sources — characters and animated bodies as solver input

> **Durum:** TASLAK — proposed 2026-09-24. Not implemented. Solver-neutral layer
> that lets animated characters (bone-driven) push fluid, gas, granular
> mud/snow and particles through the one collider path they already share.

## 1. Why this is its own layer

A character walking through mud, snow, water or smoke is not a particle
feature. RayTrophi already has strong APIC fluid, gas and granular solvers, and
the high-fidelity version of every one of those interactions belongs to them:

- a footprint in mud or snow is **plastic compaction** in the granular solver
  (`Fluid/GranularConstitutive.h` already carries cohesion, hardening and a
  compaction cap), not a decal and not a separate terrain deformation;
- a leg dragging through water is `FluidGrid::solid_vel` momentum transfer;
- smoke parting around a walking body is `gas_solid_vel_*`.

The particle system (`PARTICLE_SYSTEM_GPU_ROADMAP.md`) is the **real-time /
game-engine tier** of the same scenes. Both tiers must read the **same**
collider input. If the character's body were authored as a particle-system
feature, the fluid and granular solvers would be structurally blind to it.

> Rule: the producer (what the character's body is doing) is authored once;
> every solver is a consumer. Changing the tier never changes the input.

## 2. What exists today (verified 2026-09-24)

- `rtapi::SimulationColliderInfo` (`Api/RtApi.h`) is one collider description
  consumed by particles, fluid (`fluid_collision_enabled`) and gas (`gas_*`),
  plus MSF thermo-chemistry fields.
- `ParticleSimulationSystem` computes per-collider motion **once per step** and
  shares it with every domain (`ParticleSimulation.cpp`, "Moving-collider
  velocity" block):
  - linear velocity = difference of the collider's world centre between steps;
  - angular velocity = skew part of `current * previous.inverse()` from the
    OBB resolver;
  - the voxelizer stamps `v = linear + omega x r` into `FluidGrid::solid_vel`.
- Source modes: `PlaneY, ObjectAABB, ObjectOBB, Sphere, Capsule,
  ObjectMeshSDF, ObjectConvexDecomp, ObjectMeshBVH`.
- `AnimationController::getFinalBoneMatrices()` publishes the evaluated pose;
  `BoneData` carries `boneOffsetMatrices`, `boneParents` and bind-pose locals.

## 3. The gap

**Collider motion is one rigid-body velocity per collider.** That is correct
for a door, a crate or a thrown rock. For a skinned character authored as one
mesh collider it is silently wrong:

- the whole body is treated as a single rigid body moving at its centroid
  velocity;
- a planted foot, a foot lifting out of mud, a kicking leg — their local
  velocities never reach any solver;
- the mud is still pushed and a trough still appears, so the result **looks
  plausible**. This is the failure nobody reports as a bug.

A second, smaller gap: a `Capsule` bound to an object derives its centre from
the object's **bounds**, and a skinned mesh's bounds move with the whole pose,
not with one limb.

## 4. Proposal: bone-attached proxies

A new collider source kind: **proxy set bound to a skeleton**.

```text
KinematicProxySet
  id, name
  target: skinned object (stable node id + name fallback)
  proxies[]
    bone name
    shape: Capsule | Sphere | OBB
    local offset / axis / radius / half-length (in bone space)
    enabled
  shared contact material: friction, restitution, thickness
  consumers: fluid | gas | granular | particles | MSF   (mask)
```

Each proxy is a **rigid** collider whose transform comes from its bone every
step. The existing velocity and voxelization machinery then works unchanged —
it already handles rigid colliders correctly. What is new is only where the
transform comes from.

Why proxies first, and not a per-frame voxelized skinned mesh:

- existing consumers stay untouched (only the producer changes);
- cost is a few dozen matrices per character, no vertex readback, no per-frame
  SDF recook of deforming geometry (the particle roadmap §8.4 already forbids
  silent per-frame recooks);
- it is exactly how game engines represent characters for physics, so the same
  data carries into the game-engine tier.

A **second tier** (skinned mesh with per-vertex velocity, voxelized on GPU) may
come later for close-up hero shots. It is another producer for the same
consumers; nothing downstream changes.

An **auto-fit** helper generates a default proxy set from the skeleton and bind
pose (capsule per long bone, sphere for head/hands, box for feet). Auto-fit is
a convenience that writes ordinary proxies; it does not own state.

## 5. Traps to check before writing code

- **Unit of the bone matrix.** `getFinalBoneMatrices()` returns *skinning*
  matrices (global pose × offset/inverse bind), not bone world transforms.
  Feeding them directly as a proxy transform puts every proxy near the origin
  or in bind space — and it will not crash. Bone world =
  `final * inverse(offset)` (verify against the skinning shader's convention),
  then × the object's world transform.
- **Frame rate vs step rate.** Collider velocity is `Δcentre / dt` with the
  solver's `dt`. If the animation pose advances once per frame but the solver
  takes several substeps, one substep sees the whole frame's motion and the
  rest see zero — a velocity spike followed by silence. Pose must be sampled
  at each substep's time (or interpolated between frame poses). The repo has
  paid for this class twice already (solver locked to frame rate, timeline
  locked to render rate).
- **Teleports.** Clip changes, scrubbing and root-motion resets produce a huge
  `Δcentre`. A per-proxy validity reset (like `prev_collider_center_valid_`)
  must fire on those events, not only on first use.
- **"Missing" ≠ "zero".** A proxy whose bone name no longer resolves must be
  reported as unresolved, not silently emit zero velocity at the origin.

## 6. Scripting and IPC (CLAUDE.md rule 1)

Proxies are values, so they go through IPC from day one:

```text
physics.collider.proxy_set.create / delete / list / get
physics.collider.proxy_set.auto_fit      (skeleton -> default proxies)
physics.collider.proxy.set               (shape, bone, offsets, enabled)
physics.collider.proxy_set.sample        (read-only: world transform and
                                          linear/angular velocity per proxy
                                          at the current step)
```

`sample` is the instrument: without it, "did the foot's velocity reach the
solver?" cannot be answered from outside. Panel editing is required too
(reverse of rule 1): proxies are drawn and editable in the viewport.

Four layers + capability + descriptor overlay, as usual.

## 7. Acceptance scenarios

1. **Proxy follows bone.** Animated skeleton; `proxy_set.sample` world
   positions track the bone within tolerance at every substep.
   *Broken looks like:* proxies at the origin/in bind pose (unit trap).
2. **Velocity is not spiky.** Constant-speed walk; per-substep proxy speed is
   smooth, no one-substep spike per frame.
   *Broken looks like:* speed alternates between large and zero (frame lock).
3. **Mud footprint (granular).** Two animated capsule feet walk across a
   cohesive granular bed. Measure over IPC: footprint depth under each planted
   foot, displaced mass, and that the area between steps is **not** ploughed.
   *Most insidious failure:* a continuous trench instead of separate prints —
   that means the body is still one rigid collider.
4. **Snow compaction.** Same walk on a high-hardening granular bed; print
   depth is shallower on the second pass over the same spot (compacted).
5. **Water wake (APIC).** Leg swing produces a wake whose direction follows
   the leg, not the body's centroid motion.
6. **Smoke parting (gas).** Arm sweep through a smoke column displaces it
   locally.
7. **Particle tier parity.** Same proxy set, particle mud clumps: OnCollision
   events fire at foot strikes, not continuously.
8. **Teleport safety.** Scrub the timeline backwards; no velocity explosion in
   any consumer.

## 8. Relation to other notes

- `PARTICLE_SYSTEM_GPU_ROADMAP.md` §8.5 consumes these proxies; it does not
  define them.
- Surface wetness/dirt on the character (mud sticking to legs) is written by
  the Surface State Deposit output into the MSF, see the particle roadmap
  §10.4.

## 9. Explicitly not in scope

- Ragdoll / two-way coupling where mud pushes the character back. Proxies are
  **kinematic**: they drive solvers, solvers do not drive them. Two-way
  coupling is a later, separate decision (Jolt integration is the natural
  home).
- Cloth or hair as colliders.
