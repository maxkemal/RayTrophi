# Shared particle-domain wall contact

## Domain adhesion — 2026-11-09

Implemented in source: **Wall Adhesion (Pa)** alongside friction and restitution
in the existing Domain Walls panel. Default **0 Pa**, finite range **[0, 1e9]**.
Open/Periodic domains retain the authored value but do not apply wall contact.
This is finite tensile/cohesive traction, separate from viscous wall slip and
contact-angle/wetting simulation. It does not permanently pin a particle.

UI, Python and IPC use the same atomic `patchDomainWallSettings` operation:

```python
rt.fluid.set_wall_contact("Domain", {"adhesion_pa": 10000.0})
print(rt.fluid.get_wall_contact("Domain"))
```

```json
{"method":"fluid.set_wall_contact","params":{"domain":"Domain","settings":{"adhesion_pa":10000}}}
```

Readback adds `adhesion_pa`; existing friction/restitution keys remain unchanged.
Invalid type, negative/nonfinite/over-limit value or unknown key rejects the
whole patch. Both scene serializers persist `wall_adhesion_pa`, missing old-file
values default to zero, and the simulation fingerprint includes it for rewind.

Continuum contact has a velocity budget `adhesion_pa * dt / (rho * h)`.
Separation consumes that budget first; the unused part resists tangential slip.
Separation faster than the budget retains its remaining outward velocity rather
than receiving a reversing impulse. Existing impact/restitution and Coulomb
behavior remain unchanged at zero adhesion. CPU liquid tail and GPU dense/indexed/
sparse variants share the formula. The impulse also constrains the current
liquid advection displacement so grid sampling cannot move a stopped parcel away.
Liquid support is restricted to a one-voxel wall band. MPM applies adhesion once
on boundary MAC faces; its particle escape guard disables adhesion along with
friction/restitution to avoid a second contact impulse. Continuum density is the
resolved lane/domain bulk density (including adopted liquid substance), not a
new per-particle heterogeneous contact-mass solve.

DEM uses `adhesion_pa * pi * radius^2` as the tensile/cohesive force ceiling.
Only an established domain-wall contact can persist across a gap, up to 5% of
the grain radius. Normal tensile force is bounded by separating momentum and
cannot reverse separation. The remaining cohesive force augments the sliding
threshold. Grain-pair/mesh laws are unchanged. Adhesive DEM temporarily disables
sleeping/compact-sleep optimization because existing support forecasts only
model compressive contact; awake force evaluation preserves detachment behavior.
Metadata word `2*N+7` now carries adhesion instead of redundant restitution;
damping ratio still carries restitution and both changes invalidate sleep context.

**Shader delivery completed by Codex under the user's revised policy:** all ten
transitive dependents compiled with glslc (`vulkan1.3 -O`), passed spirv-val,
and were published to `RayTrophiStudio/source/shaders` and
`x64/Release/shaders`; SHA-256 equality checked. The bulk script already discovers
these existing .comp files. Added the missing base `sim_fluid_advect_tail` to
the simulation-only batch, so it now includes all ten.
No new shader entry was introduced. Compiled SPIR-V push ranges checked directly:
base liquid advection **100 B**, indexed advection **104 B**, MPM boundary **52 B**.
DEM shader revision **42**, existing 128-byte push layout retained.

**C++ build/live acceptance pending.** User should rebuild C++ before reopening
or stepping with these new shaders. No C++ build or application launch was run
by Codex. Current source checks PASS: domain-wall contracts, actual CPU/GLSL scalar
rebound/adhesion expressions, Matter GPU ABI/contact checks, grain contracts,
sparse MAC contracts, descriptor check and general shader-struct layout audit.
The former Matter GPU assertions were stale against the preexisting collider
`shear/vn_kept` law; those assertions now verify the current finite cohesive law.

After the user's build, in a paused empty Studio scene from a separate terminal:

```text
python scripts/test/rt_test_domain_wall_adhesion_ipc.py --backend vulkan
python scripts/test/rt_test_domain_wall_adhesion_ipc.py --backend cpu
```

These check default/readback, atomic invalid input refusal, inactive Open behavior
and a free-vs-adhered liquid ceiling slab. They have not run yet. Run the updated
`domain_wall_contact_test.cpp` in a user test target linked with the physics/
substance core and Vec3 for analytic finite traction, no reversal, time subdivision,
interior freedom and validation. Also inspect MPM ceiling hold/release and DEM
wall hold/release at matching physical stress, plus save/reopen and rewind.
Do not infer completed MPM/DEM live acceptance from shader compilation.

---


The domain owns one stationary surface for all six closed walls. Liquid/APIC-FLIP,
MPM (granular, snow and elastic), and DEM read the same `wall_friction` and
`wall_damping` parameters. `wall_damping` now has the explicit UI/API meaning
**restitution**; it is not tangential damping. No scene migration is provided.
Defaults: friction 0.5, restitution 0.15. Restitution controls liquid/DEM impacts;
MPM normal contact uses its constitutive material response. Gas boundary flow remains independent;
a dry Coulomb contact law does not describe gas wall viscosity.

## Authoring

The domain Solvers panel exposes **Domain Walls (Liquid / MPM / DEM)** outside
the liquid solver foldout. The secondary fluid solver panel uses the same widget.
Controls are inactive for Open and Periodic domains. A substance preset does not
overwrite the domain surface. Mesh colliders retain their own contact surface.

Python:

```python
rt.fluid.set_wall_contact("Domain", {"friction": 0.8, "restitution": 0.0})
print(rt.fluid.get_wall_contact("Domain"))
```

IPC methods (use the external `scripts/test/rt_ipc.py` client):

```json
{"method":"fluid.set_wall_contact","params":{"domain":"Domain","settings":{"friction":0.8,"restitution":0.0}}}
{"method":"fluid.get_wall_contact","params":{"domain":"Domain"}}
```

Both read/write operations return `{friction, restitution, active}`. The settings
object is a partial, atomic patch: omitted keys retain their value. Friction must
be a finite number in [0,2]; restitution a finite number in [0,1]. Unknown keys,
booleans, null, strings and invalid ranges fail without modifying either value.
An unknown domain fails. Writes during final rendering fail. The UI and both
bindings use `patchDomainWallSettings`; bindings share the rtapi service.
Scene save/load and the simulation fingerprint include the new friction value.

## Solver behavior

- MPM: constrain MAC grid faces using a read-only velocity snapshot. Tangential
  faces in the first half-cell receive a bounded Coulomb impulse, normal boundary
  faces remove inward motion without an extra restitution response. MPM's
  constitutive stress owns elastic rebound. The particle escape guard is also
  non-rebounding; it must not add the second impact law after G2P.
  Support traction uses the larger of the incoming
  normal velocity correction and inward gravity times physical dt. This supplies
  sustained floor traction even after normal grid velocity has been projected
  to zero. Separating motion is free. The second boundary pass only re-enforces
  constraints, so it does not spend the friction budget twice.
- Liquid: apply the same stationary-plane velocity law in the CPU/GPU advection
  tail at the domain boundary. The dt is the advection substep duration. This is
  particle wall contact, separate from viscous grid no-slip/slip treatment.
- DEM: retain the spring/history contact solve, selecting domain friction and
  restitution-derived damping for wall keys only. Mesh and grain pair contacts
  retain their material law. Rolling resistance is capped by wall friction;
  changing either wall setting invalidates sleeping support. Four extra mass
  metadata words avoid extending the 128-byte grain push constants.

Gravity supplies the sustained support estimate in continuum paths. Other
external accelerations and MPM stress-derived static normal load are not yet
included in that estimate. Full stress-dependent traction therefore still needs
live validation for pressing against a side wall in zero gravity.

GPU ABI: MPM boundary 48 bytes; advection 96 bytes, indexed advection 100 bytes.
DEM shader revision is **41**. All shader variants must be rebuilt together with
the application; old indexed advection binaries use a different lane-word offset.

## Verification

Completed without building or launching the application:

- `python scripts/test/check_domain_wall_support.py`
- `python scripts/test/check_domain_wall_contracts.py`
- `python scripts/test/check_matter_grain_contracts.py`
- `python scripts/test/check_matter_grain_sleep_contracts.py`
- `python scripts/test/check_matter_gpu_contracts.py`
- `python scripts/test/check_sparse_mac_canonical_contracts.py`
- IPC descriptor generation and source capability audit.

Pending user build/runtime checklist:

1. Build shaders using `RayTrophiStudio/source/shaders/compile_sim_shaders.bat`,
   then build RayTrophi Studio in the usual IDE configuration.
2. Build/run `scripts/test/domain_wall_contact_test.cpp` with the project include
   path and `source/src/Math/Vec3.cpp` linked. It covers all six normals, independent
   rebound/slip, support stopping, substep invariance, separating/interior/open
   behavior, and atomic invalid patches. This C++ test was not compiled here.
3. Remove the extra floor plane. Compare snow at friction 0, 0.5, 1 with
   restitution 0: zero should slide, higher values should arrest small slip.
   Repeat on CPU and Vulkan, and with elastic MPM, DEM and liquid carriers.
4. Test a side wall, corner, zero gravity, a falling/bouncing body, and 30/60/120
   fps. Tangential velocity must not reverse just because friction is increased.
5. Change wall settings on a sleeping DEM pile and check wake/re-settle. Test a
   mixed MPM/DEM/fluid domain so only the MPM grid path owns MPM friction.
6. Save/reload and read settings via Python/IPC. Change the substance preset and
   verify wall settings remain authored. Open/Periodic report `active: false`.
7. Send invalid partial patches (negative/out-of-range, unknown keys, boolean,
   null) and verify both settings remain unchanged. Keep IPC probes in a separate
   terminal; do not run them from the embedded script workspace.


## Resting-contact rebound regression (2026-11-07)

The initial shared law returned `e * incoming` even when the entire incoming
normal speed came from gravity in the current substep. For a supported body this
repeats a spurious rebound every tick. CPU and GLSL now use
`rebound = e * max(incoming - support, 0)`, where `support = max(-g.n, 0) * dt`.
The normal impulse includes incoming + rebound; tangential friction still uses
that impulse or the sustained support budget, whichever is larger.

The non-build test evaluates the actual scalar expressions extracted from both
sources. It checks 1000 resting support ticks at multiple dt values and verifies
that independent incoming impacts still rebound. The C++ fixture includes the
same resting-contact regression, but was not built or run by the agent.

External named-pipe inspection of the user's open Snow scene found 63,995 MPM
carriers, wall friction 2, restitution 0.4. A live coefficient A/B was invalid:
a settings change invalidated and cleared the live runtime, so its zero-speed
result is NOT evidence of a successful simulation fix. After explicit user
approval, the original Default_Cube mesh seed (8 particles/cell) was recreated.
The restored 63,995 carriers had zero speed and exactly the original position
hash `1b01ef227fc0cfcf`; wall settings 2 / 0.4 and timeline frame 0 were confirmed.
No project was saved. The running executable/shaders still predate this source
fix, so corrected live dynamics require the user's application/shader rebuild.


## MPM stress-response correction (2026-11-07, supersedes gravity-only fix)

User rebuilt and still observed snow moving upward from floor contact. Subtracting
only gravity was insufficient: MPM normal velocities also include constitutive
stress impulses. The shared restitution law was rebounding that support response.

The MPM MAC-face adapter now uses zero additional restitution on CPU and GPU.
The GPU adapter also preserves the already-constrained normal value instead of
restoring its pre-contact snapshot. Tangential contact still uses the domain's
Coulomb friction. Both CPU and GPU MPM particle escape guards use zero additional
restitution as well, so a residual penetration after G2P cannot reintroduce bounce.
Liquid/DEM restitution is unchanged. The UI tooltip and IPC discovery notes
explicitly describe this solver distinction.

Regression checks include stress-driven normal speeds up to 100 m/s, authored
restitution 0/0.4/1, and particle escape at restitution 1. The normal MPM response
must be zero on approach and preserve separating motion. The C++ regressions
were added but not compiled. Source contracts verify both adapters and both tail
paths use this policy; no build, application launch, or live-state mutation was
performed for this correction.

After rebuilding the application and simulation shaders, replay the original snow
scene from a fresh initial state/cache. Confirm friction still arrests lateral
slip and floor contact does not add upward rebound when the authored restitution
is 0.4 or 1. The snow material can still rebound through its own elastic energy;
this fix removes only the extra wall impact response.
