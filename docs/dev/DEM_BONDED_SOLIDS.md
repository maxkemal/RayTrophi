# DEM bonded solids

2026-11-10. User application build completed. Bounded dry-DEM live acceptance
passed through external named-pipe IPC; full numerical/material acceptance remains
pending. The CPU reference, GPU substep stages, canonical graph, cache,
persistent seed authoring and UI/Python/IPC paths are wired. Ice remains loose
by default. A seed layer or an explicit runtime selection can become a bonded
body. This is not yet a validated/calibrated ice or soil preset.

Latest source-only update: intact-edge compaction and layered-seed runtime reset
are written; the user has not rebuilt this update yet. Its shaders were compiled,
validated and deployed. Live strength A/B below used the already-built executable,
not the unbuilt C++ optimization.

## Reference delivered

`DemBondMechanics.h/.cpp` defines explicit bond creation between two stable
particle identities and a small-strain corotational circular-beam law. The
reference has axial, transverse, bending and torsional resistance, tensile/shear
failure, irreversible broken state and an energy-release ledger. Finite common
rigid rotations do not constitute deformation. Forces balance pairwise;
endpoint torques also balance the moment of the transverse forces. Invalid
input leaves state and response unchanged.

This is a candidate reference for implementation and comparison, not a calibrated
ice/soil model. The abrupt strength criterion
has no independently prescribed fracture energy or process-zone regularization.
Mesh/particle-resolution independence must not be claimed. Released energy is
accounted as dissipation, not injected as velocity or silently added as heat.
Finite-strain energy consistency and integration accuracy still need testing.

`stableStep` supplies a conservative undamped linearized stiffness trace bound
using endpoint masses, inertias and intact graph degree. Runtime combines this
bound with the contact stability limit through reciprocal-square addition.
The existing contact adaptation remains; bond stiffness is never automatically
softened. Insufficient max_substeps holds the step with an explicit error.

User-run C++ regression source: `scripts/test/dem_bond_mechanics_test.cpp`, linked
with only `DemBondMechanics.cpp` and the source include directory. Checks cover
rigid transforms, quaternion sign equivalence, analytic axial/torsional response,
linear/angular momentum balance, irreversible tension/shear fracture, energy
release once, transactional errors and stiffness/degree scaling of the dt bound.
The test has not been compiled or run by Codex.

## Runtime and lifecycle

`FluidParticles::dem_bonds` is authoritative: a sparse undirected graph of stable
endpoint IDs, body IDs, accumulated material quaternions, reference poses,
per-edge material, failure state and energy. A sorted particle view uploads dense
indices and CSR adjacency transiently; sorting cannot heal/recreate a bond.
Each initial source builds only its own new particles into a separate body.
Close bodies and future emitter births do not acquire bonds automatically.
Explicit release can remove a body; rebinding is a new authoring event.

Three new kernels have 10 bindings and 32-byte push constants. Edge rows are
96 bytes, with history at offset 80. One invocation per edge evaluates its load
and failure; one per particle gathers incident loads and applies a velocity/spin
kick before existing MPM contact and DEM contact/integration. Orientation then
advances from the resulting angular velocity. This is a first-order split
integrator requiring dt-convergence acceptance, not a simultaneous constraint
solve. Equal/opposite forces and endpoint moments include the transverse-force
moment. No float atomics accumulate loads. Atomic writes only record invalid
response status; a failed frame cannot publish graph/pose/history changes.

The pending compaction update uploads only intact edges and maps each result
back to its canonical edge index. Broken reference/history/energy rows stay on
the host unchanged. There are three extra dispatches per DEM substep while
constraints remain. Once all bonds are broken, force evaluation and gather/kick
stop on the next frame: only one orientation dispatch remains. Zero transient
quaternions mark nonmembers, so disconnected members still track spin without
needing an active edge. Tiny valid dummy bindings replace retired edge/load/CSR
pools. Contact sleeping can resume when no bonds constrain motion.

There are two transfer batches per graph frame attempt, with no per-substep host
readback. With all bonds broken, only poses/status are read back and poses/status
plus a dummy adjacency word are uploaded. GPU buffers are pooled in the
DEM runtime, refreshed from canonical state on retries/reset, and included in
working-set/transfer/dispatch accounting. Contact sleeping is disabled while
intact bonds exist so dormant support logic cannot ignore tensile/bending loads.
The graph does not add a second volume, mass or fluid-coupling owner: fragments
continue through existing DEM porosity, pressure/drag and gas coupling paths.
The DEM porosity upload invalidates solid_weight_source_valid after weights change.

Important physical limit: the current DEM liquid projection still deposits sphere
volume and the coordinator passes minimum voidage 0.3. A bonded assembly therefore
remains a porous granular body for this coupling. Bonding is not an impermeable
continuum ice volume or a crack-resolved surface representation. A conservative
coherent-body volume/boundary producer and pressure reaction acceptance are a
separate required step before claiming solid-ice displacement/impermeability.
This was reported to Claude through the shared board; its busy pressure files
were not changed during the ongoing live tests.

Host publication follows all existing DEM retry/validity checks. Owner filtering
and the frame-end lifecycle retire missing, frozen or converted endpoints and
account their stored/fracture energy once. Cache v20 adds bounded native binary
records for the graph (32-byte header, 48-byte node, 176-byte edge, 256 MiB cap),
without constructing a per-frame JSON graph. v19 remains readable with empty bonds.
RAM cache copies the canonical
graph and includes its allocation estimate. Explicit reset clears graph and body
allocator; authored seed recipes recreate initial bonds. GPU allocation state is
not authoritative and is never used to infer new bonds after rewind.

Live bind/release invalidates future frames and attempts an exact current-frame
velocity checkpoint, including on a frame outside the usual keyframe stride.
The response reports `checkpoint`; a cache failure does not masquerade as an
authoring rejection after the graph has changed. Seed recipes, not runtime edits,
are the persistent project-level setup.

## Authoring and script/IPC

Matter initial sources expose `bond_body`. The existing seed JSON serializer,
ProjectManager/SceneSerializer hooks and recipe hash carry all its settings.
Old recipes default to disabled. Seed Settle must be off; its pre-settle operator
does not implement bonded-body dynamics. Validation rejects a non-DEM material,
isolated particles, excess graph degree (64), edge/node capacity or graph budget
without replacing the live population. Each source has independent body identity.

Example seed-source settings patch, through the existing update_seed_source API:

```json
{"bond_body": {"enabled": true, "young_pa": 100000,
 "poisson": 0.25, "tensile_strength_pa": 10000,
 "shear_strength_pa": 10000, "reach_ratio": 1.25, "section_ratio": 0.5}}
```

These numbers are a test fixture, not ice/soil physical calibration. Bond section
radius is section_ratio times the smaller endpoint radius. Macroscopic modulus,
strength and anisotropy depend on the assembly and require calibration.

The solver panel also exposes explicit live selection authoring. Shared core:
`ParticleSimulationSystem::demBondOperation`; UI/Python/IPC all call rtapi::demBodies.
Python: `fluid.dem_bodies(domain, action="inspect")`; mutations require pause.
IPC: `fluid.dem_bodies` with `domain` and a `request` object:

```json
{"domain":"Physics Domain 1", "request":{"action":"bind",
 "all_current_grains":true, "material":{"young_pa":100000,"poisson":0.25,
 "tensile_strength_pa":10000,"shear_strength_pa":10000}}}
```

Other actions: inspect (counts, body_ids and energy), snapshot (canonical graph),
remove (body ID). Bind accepts particle_ids instead of all_current_grains.
Unknown fields, wrong types, missing/duplicate/already-bound IDs and invalid
material reject transactionally. Future births stay unbound. Defaults for live
bind are reach_ratio=1.25 and section_ratio=0.5. IPC descriptors were regenerated.

## Verification and exact user checklist

Completed without a C++ build or Studio launch:

- `python scripts/test/check_dem_bond_contracts.py`: PASS static integration/ABI,
  project XML registration, canonical/cache/seed/API wiring and deployed hashes.
- Three new shaders: glslc -O --target-env=vulkan1.2, spirv-val PASS, source/runtime
  copies identical. Disassembly confirms 32-byte push and 96-byte edge layout.
  New kernels registered in compile_sim_shaders.bat; other agent shaders untouched.
- `gen_ipc_descriptors.py --check`: PASS. Layered seed, Matter lanes and compact
  solid-weight source contracts: PASS. Before the build, IPC test syntax alone
  was checked; its later live execution is recorded below.

Claude reconciled the shared suite expectations with its changes. Re-run by
Codex after the user build: check_matter_grain_contracts, check_matter_gpu_contracts,
check_domain_wall_contracts and check_sparse_mac_canonical_contracts all PASS.
The protected porous registry rows remain 12 and 14 bindings.

## Live acceptance, 2026-11-10

External scripts used scripts/test/rt_ipc.py from a separate Python process.
Sandbox pipe access returned Windows error 5; the same transport was re-run
with sandbox escalation. Studio was already open, paused and empty. Only
T_CodexDemBond was created; it was removed afterwards, leaving no domains and
playback paused at frame zero. No C++ or shader changes were needed in this test turn.

Fixture: Matter/Vulkan/Closed, bounds [-.3,0,-.3]..[.3,.6,.3], voxel .05 m,
Ice DEM radius .025 m, thermal liquid disabled, Seed Settle off. One source
[-.1,.25,-.1]..[.1,.45,.1] produced 63 particles and 240 bonds. Bond E=1e5 Pa,
Poisson=.25, reach=1.25, section=.5. These are mechanics test values, not ice calibration.

| Check | Observed result |
|---|---|
| Existing rt_test_dem_bodies_ipc.py | PASS: invalid material and rebind rejection atomicity, bind/release, unchanged position/velocity hashes |
| Strong tensile/shear strength 1e6 Pa, 60 x .005 s | 0/240 broken; elastic response at floor impact, 63 particles retained |
| Weak strength 100 Pa | No cracks through step 45 (free fall); 240/240 broken at step 50; fracture dissipation .00402699228843 J unchanged through step 60 |
| Intermediate strength 10000 Pa | 119/240 broken at step 50, 212/240 at step 60; not an all-or-nothing graph deletion |
| Seed reset after fracture | 240 intact bonds, zero fracture dissipation |
| Bound hold: E=1e10 Pa, max_substeps=2 | Explicit DEM bond stability hold; position and velocity hashes unchanged; 240 intact bonds. Restoring E=1e5 / limit=512 resumed motion |
| Runtime release at timeline frame 7 | checkpoint=true; forward step then scrub to 7 restored released graph (0 bonds), with retired energy 3.201995089824777 J preserved |
| Two adjacent independent source layers | 126 particles, 480 bonds, 2 bodies; every edge's two endpoints belonged to the same source-created body |

Analytic free fall over .2 s, strength 1e6 Pa, no impact: expected drop .1962 m,
speed 1.962 m/s. With frame dt=.005: drop .196345847869 m, speed 1.962022700007;
dt=.0025: drop .196334500989 m, speed 1.961961793521. Maximum position error
was below .15 mm; no broken bonds or particle loss. This checks gravity and
frame partitioning, not a full internal-substep/energy convergence study.

Still pending: standalone assertion-enabled CPU/GPU reference fixture execution,
native disk-cache round trip in a built test target, angular momentum/energy
convergence, calibrated material failure, mixed-phase and impermeable-volume
acceptance, and large-population performance. These are not implied by the dry
drop test. The live RAM checkpoint was tested; native disk codec was not.

## User-scene strength A/B and pending work compaction

2026-11-10. User authorized restoring the previous Ice layer after it and its
domain had been removed. The recorded bounds/settings were restored in a fresh
Matter/Vulkan domain; no other domain was present. Geometry: 59130 particles,
335953 bonds, radius .025 m, voxel .1 m, source bounds
[-1.68,.4389998,-1.365]..[1.68,1.1190002,1.365]. Frame dt=1/24 s, 24 frames.
Young modulus stayed **1e6 Pa**, Poisson .25, reach 1.25 and section .5 throughout.

The first warm-runtime comparison is retained in DEM_BOND_USER_AB.json. It exposed
a reset omission: Apply reset particle state but retained the previous contact
CFL hint. First A used 24 contacts / 678 substeps; warm B used 15 / 596.
The pending successful-Apply hook now invokes invalidateMatterGrainAfterReset,
which already clears contact/CFL/publication state while retaining allocations.
It does not run on failed generation. That source fix has not been live-tested.

The definitive A/B therefore recreated the dedicated test domain before EACH
case. Both first frames used **24 contacts / 678 substeps** and identical initial
position hash **84714c6069a13ea3**. See [raw cold report](DEM_BOND_USER_AB_COLD.json).

| Tensile and shear strength | Broken at frame 24 | Mean full-step host time | Last substeps |
|---|---:|---:|---:|
| 10000 Pa | 335953 / 335953 | 647.97 ms | 262 |
| 1000000 Pa | 0 / 335953 | 814.00 ms | 576 |

This confirms sensitivity to strength in this bounded comparison; it does not
calibrate ice or establish numerical convergence. Intact constraints remain
expensive. Strength does not enter the elastic stability formula directly, but
preserving the graph maintains its constraints and associated substeps. These
timings are **not** a before/after benchmark of the pending optimization.

The scene was left paused at frame zero with the stronger layer, 335953 intact
bonds, zero stored/fracture energy and the same E=1e6 Pa. Baseline settings/digests
were recorded in DEM_BOND_USER_AB_BEFORE.json (not a full simulation snapshot).
The repeatable external client is scripts/test/rt_compare_dem_bond_strength_ipc.py;
its opt-in --cold-runtime mode recreates its selected dedicated test domain and
does not preserve arbitrary extra domain authoring properties.

Pending work-path verification after one user incremental build:

1. Run assertion-enabled dem_bond_compaction_test.cpp: partial graph uploads only
   the intact row; fully broken graph skips force stages on the CPU buffer backend,
   retains all canonical history, and transfers only poses/status/dummy adjacency.
2. Run the expanded Vulkan fixture dem_bond_gpu_reference_test.cpp: a compact row
   must publish to canonical edge 1 while broken edge 0 stays bit-identical;
   fully broken members continue spinning with no velocity kick or energy reset.
3. Repeat a failing/fully broken case and compare counters, history and pose
   validity. The all-broken path has 1 extra orientation dispatch per microstep,
   not 3, and no edge-row uploads/readbacks. The current all-broken 59130/335953
   fixture's graph traffic by layout drops from 36121720 to 946088 upload bytes,
   and 33197572 to 946084 download bytes per frame. These are layout-derived
   expectations, not measured speedups. Large edge/load/CSR device pools trim once.
4. Repeat Apply twice: first-step contact hint must start at 24 for this fixture
   rather than inherit the last run's estimate. Check failure atomicity still holds.

Static DEM/grain/GPU/wall/seed contracts and shader validation/reflection pass.
No C++ test or new-path live timing was run by Codex. GPU bindings/push constants
remain 10/32; canonical graph and native disk-cache schema remain unchanged.

Remaining/repeatable acceptance checklist for the broader model:

1. One user C++ build of the combined tree. Ordinary incremental build with
   dependencies must rebuild affected headers/TUs and add new .cpp modules;
   Clean/Rebuild is not intrinsically required. Do not run an old executable
   against the concurrent agent's changed porous shader ABI.
2. Compile/run assertion-enabled dem_bond_mechanics_test.cpp and
   dem_bond_network_test.cpp. The latter links the bond modules and existing
   particle/substance core. Verify cache round trip, corruption rejection,
   irreversible damage, ID reorder, endpoint retirement and source isolation.
3. Invoke testDemBondGpuReference from dem_bond_gpu_reference_test.cpp in the
   initialized Vulkan test fixture: compare axial/shear kicks, orientation,
   fracture and energy with the double CPU reference. This fixture never edits
   the live scene.
4. Dedicated paused loose DEM population, external process only:
   `python scripts/test/rt_test_dem_bodies_ipc.py "Physics Domain 1"`.
   It binds/releases current grains, validates rejection atomicity and unchanged
   kinematics; it deliberately does not advance time. Do not use an existing
   authored body scene for this test.
5. User-run dynamics: free rigid rotation and zero force, drop without premature
   fracture, tensile/bend/twist failure, separated bodies, thin supports, strong
   impacts, fluid/grain coupling, melting/removal, max_substeps hold/retry and
   cache scrub/reset at a non-stride frame. Compare dt and spatial refinement,
   mass, linear/angular momentum, energy (including fracture/retirement ledger)
   and actual full-step cost. No physical or performance PASS claimed yet.

## Material scope

- Ice block: explicit initial body bonds, brittle failure, no spontaneous heal.
- Loose ice: existing unbonded DEM.
- Wet sand: retain the existing water-volume-dependent capillary bridges; solid
  bonds do not replace the liquid bridge law.
- Soil: cemented bonds can use this skeleton. Pressure-driven bond formation,
  plastic compaction and saturation effects need separate constitutive laws and
  calibration. They are not implemented by raising a cohesion slider.

The reference is informed by the finite-section force/moment and irreversible
tensile/shear failure structure of the
[Itasca parallel-bond model](https://docs.itascacg.com/itasca950/common/contactmodel/linearpbond/doc/manual/cmlinearpbond.html).
It is not an assertion of matching that implementation or its calibration.


## Live cost measurement — 2026-11-10

External IPC, Vulkan, 59130 grains / 335953 initial bonds, dt=1/24 s.
The application was initially empty and paused. Dedicated test domains were
removed afterwards; no source/shader change or build was performed in this run.
Raw data: [DEM_BOND_COST_LIVE.json](DEM_BOND_COST_LIVE.json).
Repeatable client: `scripts/test/rt_measure_dem_bond_cost_ipc.py`.

| State | Samples | Fluid Step mean | DEM host mean | Microsteps |
|---|---:|---:|---:|---:|
| Intact, strength 1 MPa | 22 | 971.41 ms | 953.44 ms | 576–582 |
| All broken, strength 10 kPa | 8 | 485.90 ms | 466.42 ms | 262 |

The running executable **does use live-edge compaction**. At fully broken
state dispatches are 1310 (5 per microstep), versus the previous unfiltered
1834 (7 per microstep). Runtime download is 4494072 bytes, versus 36745560
with all 335953 edges present; upload is 5440188 versus 40615820 bytes.
Partial fracture reduces traffic as well: download follows
4494072 + 96 * intact_edges_at_frame_entry. No grain loss was observed.
This supersedes the earlier pending live counter acceptance above; standalone
CPU/Vulkan reference fixtures and native disk-cache numerical acceptance remain pending.

Host GPU-work/wait phase averages 718.19 ms intact and 369.40 ms fully broken;
CPU preparation averages 150.14 and 27.78 ms respectively. These are host phase
timings, not GPU timestamp measurements. Different states require different
microsteps, so their timing ratio is not an isolated optimization speedup.
High explicit microstep counts and the GPU-work/wait phase remain the main cost.

**Additional reset probe failed:** a small fresh seed used contact hints 24 then
12; after re-Apply the next hint remained 12 rather than resetting to 24.
The proposed Apply reset source wiring is therefore not live-accepted. This
probe alone does not distinguish a stale build from a reset-path defect.
The probe domain was also removed. No automatic source fix is claimed.
