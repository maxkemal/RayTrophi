# Kinematic Collider Sources — characters and animated bodies as solver input

> **Durum:** AKTİF — 2026-09-26. K0 IPC CRUD/validation smoke testi canlıda
> geçti. K1 panel/overlay ve body-first auto-fit canlı rig üzerinde 22/22 proxy
> çözümledi; iki ayağı korudu ve detay kemiklerini dışarıda bıraktı. Dünya/metre
> auto-fit düzeltmesi canlıda bacaklar için yaklaşık 9.4 cm örnek yarıçap verdi.
> K2 CPU snapshot/voxel stamp yolu Vulkan APIC su domaininde altı timeline
> karesi boyunca 6/6 kez çalıştı; hareketli ayakları izlerken 57.800 parçacığı
> korudu. Kullanıcı viewport'ta ayak proxylerini ve yerel su etkileşimini gördü.
> Geri sarma denemesi sahnenin seed tarifinin persistent olmadığını
> (`dropped_seeds`) gösterdiği için ghost-cell kabulü geçerli sayılamadı;
> kalıcılık round-trip ve gas kabulü de bekliyor. Katman, animasyonlu
> karakterlerin fluid, gas, granular mud/snow ve particles alanlarını aynı
> collider girdisiyle sürmesini hedefler.

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
  target: skinned object (character name in K0; stable node id + name fallback
                          after K1 scene identity support)
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

Auto-fit prioritizes body/limb contact coverage. Finger, eye, skirt, twist and
terminal `End` bones are detail bones and are excluded by default; callers can
request them with `include_detail_bones=true`. When details are enabled, body
bones are still processed first so the proxy budget cannot keep fingers while
silently dropping a foot.

### 4.1 Ownership and data flow

```text
Rig evaluated joint globals
          |
          v
KinematicColliderRegistry -- authoring, stable IDs, validation
          |
          v
KinematicColliderSampler  -- one pose cache per character per solver step
          |
          v
KinematicColliderSnapshot -- transforms, endpoints, linear/angular velocity
          |
          +--> APIC fluid / granular / gas
          +--> realtime particles (consumer only)
          +--> MSF deposition/contact
```

`KinematicProxySet` is a separate registry, not another mode squeezed into a
single rigid collider. UI, Python and IPC mutate this registry through the same
core service. Solvers consume immutable sampled snapshots; they do not inspect
the rig, own bone mappings, or repeat pose evaluation.

### 4.2 Cost contract

- Resolve a character's bone-name map once per sampling pass, not per proxy.
- Sample the rig once per character per solver step and fan the result out to
  all consumers.
- Use analytic sphere/capsule/box proxies in the interactive tier; no deforming
  mesh SDF rebuild or CPU vertex readback in the default path.
- Keep authoring visualization analytic and instanced. Selection overlays may
  be richer, but idle viewport cost must scale with visible proxies, not mesh
  triangle count.
- Default auto-fit is 64 proxies and the hard ceiling is 256 proxies per set;
  the registry also accepts at most 256 sets. Imported production rigs and IPC
  clients therefore cannot create an unbounded simulation workload.

## 5. Traps to check before writing code

- **Choose the right pose source.** The current rig path publishes evaluated
  joint globals through `ImportedModelContext::rigJointGlobals`, and
  `RigAuthoring::listBones()` applies `rigSceneTransform`; this is the canonical
  proxy pose. Only a legacy fallback may recover a bone transform from a
  skinning matrix (`final × inverse(offset)` after verifying conventions).
  Feeding a skinning matrix directly to a proxy silently puts it in bind space
  or near the origin.
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
solver?" cannot be answered from outside.

★ 2026-09-26 correction: `sample` answers a different question. Its velocity
is computed against the solver's motion history at READ time, so it reads 0
once the frame loop has caught up and a stale delta otherwise, and nothing in
it says how many cells a proxy occupied. The solver-side instrument is
`physics.collider.proxy_set.solver_stamps`: per proxy and domain, the cells
stamped in the last grid step and the velocity written into `solid_vel`. A
resolved proxy that stamped nothing is listed with `stamped_cells = 0`. The
live probe below counted voxelize calls, which would have reported PASS for a
foot stamping zero cells. Probe:
`scripts/test/rt_probe_kinematic_foot_stamps_ipc.py`. Panel editing is required too
(reverse of rule 1): proxies are drawn and editable in the viewport.

Four layers + capability + descriptor overlay, as usual.

Current source surfaces:

- Core: `KinematicColliderSource` and `KinematicColliderScene`.
- Shared API: `Api/RtApiKinematicCollider`.
- Python: `rt.collider.proxy_set.*`.
- IPC: `physics.collider.proxy_set.*` and `physics.collider.proxy.*`.

### 6.1 Delivery phases and collision boundary

| Phase | Scope | State |
|---|---|---|
| K0 | Registry, validation, auto-fit, rig sampling, motion history, Python/IPC inspection | IPC CRUD/validation verified live; rig sample pending a rigged scene |
| K1 | Scene persistence, contextual authoring panel, analytic viewport proxies | Build and live rig scale/follow verified; save/reopen round-trip pending |
| K2 | One shared CPU snapshot consumed by APIC, granular and gas; exact moving-stamp restore and rewind resets | Vulkan APIC consumption and visual foot/water interaction verified; persistent-seed rewind and gas acceptance pending |
| K3 | Footprint, compaction, wake, smoke and teleport acceptance scenes with measurable IPC checks | Planned |
| K4 | Stable snapshot/consumer handoff to the realtime particle system | Planned; particle roadmap owns GPU-side consumption |
| K5 | Optional hero-quality deforming mesh producer with per-vertex velocity | Future |

K2 adds only narrow integration wiring to `ParticleSimulationSystem`; proxy
authoring, rig resolution and motion history remain in the kinematic modules.
Vulkan gas already uploads the resulting host `solid`/`solid_vel` fields, so
this CPU producer works with both CPU and Vulkan gas solvers. K4 remains the
dedicated realtime-particle consumer; it must not duplicate rig sampling,
proxy authoring or motion history.

K0 resolves targets by character name because the current rig authoring API
does not expose a scene-wide persistent node identity. `target_node_id` is
reserved but rejected when non-empty; K1 must implement identity resolution
before enabling that field. It must never be accepted and silently ignored.

Live IPC probe: `python scripts/test/rt_test_kinematic_collider_ipc.py` from a
separate terminal while the application is open. It uses a deliberately
missing character, verifies CRUD/validation/unresolved-target behavior, and
restores the original registry in a `finally` block. A rigged scene is still
required for the bone-following and auto-fit acceptance cases.

Rig probe: `python scripts/test/rt_probe_kinematic_rig_ipc.py <character>`.
2026-09-26 live result on character `1`: 80 bones, 64 proxies, 64 resolved,
64 moving over 450 ms; left-foot motion 0.0040 m and toe motion 0.0065–0.0077
m. The same run exposed the old arbitrary-order budget bug because right-foot
bones arrived after many finger bones. The source now excludes detail bones by
default and orders all body bones before opt-in details; rebuild verification
must require both left and right foot proxies.

### Open viewport issue — oversized proxy outlines (reported 2026-09-26)

The live walking-character preview made some bone capsules appear much too
large. The overlay already projects a world-space radius through the active
perspective/orthographic camera. Inspection found the concrete unit error:
auto-fit limits are world metres, while stored proxy dimensions are bone-local.
Sampling previously published those local values as metres; the first sampling
fix exposed the inverse authoring error because auto-fit also clamped world
limits directly in local space. Auto-fit now measures bone lengths and clamps
radii in world space, converts the result into bone-local dimensions, and
sampling reapplies the evaluated basis scale. The next build must verify both
halves before changing projection code. A second independent viewport rect
issue remains worth checking:

1. projection currently maps into global ImGui `DisplaySize`; it must use the
   actual viewport content rectangle, including its origin and render extent;
2. compare authored and sampled dimensions with
   `scripts/test/rt_inspect_kinematic_scale_ipc.py` and confirm the ratio agrees
   with the evaluated rig scale.

Acceptance: a sphere of known world radius must cover the same pixels as a
viewport reference gizmo in perspective and orthographic views; changing panel
layout must not resize or offset it. Then compare authored radius, sampled
world radius and the rig scene scale for one thigh and one foot before changing
the fit heuristic.

Live APIC probe: `python scripts/test/rt_probe_kinematic_fluid_live_ipc.py
"1 Kinematic Preview" 6`. The 2026-09-26 run advanced frame 0 to 6, observed
six `sim.fluid.voxelize_colliders` calls in 6.981 ms total, kept all 57,800
particles, and measured 0.26-0.34 m foot-proxy motion. A call count proves the
consumer path ran; the user also confirmed visible foot gizmos and local water
interaction. `rt_probe_kinematic_rewind_ipc.py` must be run only with a
persistent FillLevel/reseed recipe: it deliberately rejects `dropped_seeds`,
because an empty/recreated liquid cannot prove ghost-cell cleanup.

### Weak foot interaction — measured 2026-09-26

User report: splat and granular crushing/dispersion under the feet looked very
weak. Measured on the walking rig (5.88 cm voxels, water ~1 voxel deep, top at
y ≈ 0.078 m) by replicating the voxelizer's cell-centre test per frame:

- Foot and toe proxies stamped 1-2.4 cells per frame on average, sometimes 0.
  Leg capsules stamped ~60 but sat above the water (y ≥ 0.13).
- Cause: foot/toe are `isBodyAnchor` joints, so `fitLeaf` gave a fixed
  3.75 × 2.5 × 6.25 cm box centred on the ANKLE pivot. On a planted foot its
  bottom face stayed at y ≈ 0.07, so it grazed the top of the water; heel and
  sole had no proxy at all. The ankle-to-toe box branch further down `autoFit`
  was unreachable for the same reason (as was the head sphere branch).
- Fix: `collectKinematicJointPoses` measures each joint's bone-local bounds
  over the rest-pose skinned vertices it dominates (weight ≥ 0.5), and
  `fitLeaf` builds foot/toe boxes from them. The dead branches were removed.
  Existing sets keep their old boxes until auto-fit is re-run.
- Not fixed here: a clip loop that teleports the root (observed at frame 61,
  35-40 m/s on every foot) has no discontinuity reset. It only matters for
  looping test clips, not for a character that walks on, so it was left open.
  Water one voxel deep is itself under-resolved; that is a scene choice.

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
