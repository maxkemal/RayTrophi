# Domain field samples — implementation and acceptance

Updated: 2026-11-12. Source integration is present. User C++ build and live
acceptance are pending; this document does not claim production or performance acceptance.

## Active path

`DomainVisualService` publishes a read-only, render-only population after the
physical domain step. The physical `state.particles` remains authoritative for
simulation, Fog and SurfaceSDF. Render consumers get a separate `FluidParticles`
view, containing field samples plus the original parcels that must remain physical.

The explicit route is `cpu_soa_reconstruction`: post-step physical carrier mass,
substance density and velocity are reconstructed separately for each source /
substance / constitutive model / label stream. MPM volume includes the available
elastic determinant and plastic volume. Missing physical mass, density or MPM
volume state is an error. This is **not** the projected solver MAC field and does
not borrow the liquid lane's velocity for MPM. It currently uses a conservative
trilinear cell reconstruction and a one-cell velocity extension from the same
publication. Its CPU cost is reported; GPU sample transport is not delivered.

Samples have persistent UID, position, previous position, represented volume,
birth time and a birth-seeded fixed quaternion. Frozen-velocity RK2 uses local CFL
substeps and the domain's voxel collision mask. No second force integration occurs.
A known-empty region without current velocity support can retire samples; missing
velocity in material support rejects the complete visual update. Retained checkpoint
history is not displayed as a current successful publication after an error.

Hysteresis retains existing samples around the requested density. Cell quotas are
budgeted deterministically; sample weights normalize the represented physical
volume. Underrepresented regions report `unrepresented_volume`; their missing
volume is not silently assigned to another substance. A density/budget edit can
reseed the same physical publication without advancing time.

`DomainSplatRenderBridge.inl` consumes the render view through the existing source
and material planner. Procedural spheres draw one primitive per visual sample;
legacy parent-child expansion is disabled for this primary field route. Raster's
physical resident-position fast path is bypassed, so raster, Vulkan RT and
RayFusion read the same sample positions and sizes. Source objects and near-mesh
palettes use those same samples. New mesh banks are generated directly as flat
`TriangleMesh` / DNA SoA buffers from normalized flat source geometry; no scene
`Triangle` facade collection is used as authoritative input. Source UVs and face
materials survive. Measured exemplar volume correction is applied once.

Birth quaternion is converted through `InstanceTransform::fromMatrix`, preserving
the engine's Euler convention. Sample IDs also drive Object Info random values
across pool compaction and the sphere/mesh split. Source geometry is shared per
bank, not rebuilt per sample.

## Authoring and persistence

Default is disabled. Matter Output > Field samples exposes enable, target grain
diameter (converted to canonical target volume), sample limit, memory budget,
reset and status. Existing splat/source-object controls choose geometry.

UI, Python and IPC call `Api/DomainVisualApi.h` and the same service:

- Python: `rt.fluid.visual_samples(domain, offset=0, limit=0)`;
  `rt.fluid.set_visual_samples(domain, **settings)`;
  `rt.fluid.reset_visual_samples(domain)`.
- IPC: `fluid.visual_samples` with `domain`, optional `offset` and `limit`;
  `fluid.set_visual_samples` with `domain` and a `settings` object;
  `fluid.reset_visual_samples` with `domain`.
- `limit=0` reads metadata only; inspection is capped at 256 entries.
- Settings: `enabled`, `target_volume` (m3), `max_samples`, `seed`, `cfl`,
  `max_substeps`, `reseed_hysteresis`, `working_budget_mb`; schema version 1.
- Writes/reset are undoable, reject invalid patches atomically and do not reset
  physical state. They refresh the particle render bridge directly, rather than
  requesting a physical timeline-cache restore.

Read capability covers metadata/inspection. Writes require SceneWrite and reject
final-render scene locks. Descriptors are generated from dispatcher sources and
`scripts/ipc_descriptor_overlay.json`; the capability audit covers the new methods.

Both project serializers store settings. Immutable checkpoint handles ride with
RAM frame metadata. SimCache **v21** writes clock, stream metadata, actual sample
state and UID cursor; v20 and earlier remain readable for physical state, but
field samples require reset/rebake because visual history is absent. Render
adapter arrays are reconstructed from checkpoints on disk load. RAM cache memory
accounting includes samples and adapter arrays. Saving only an RNG seed is not
used as an advection checkpoint.

## Scope and limits

- CPU reference route, maximum 4,000,000 visual samples. Working budget is
  16..16384 MiB; conservative staging reservation is 768 bytes per maximum sample
  plus physical render staging. Default: 1,000,000 cap / 1024 MiB. Field storage
  has an additional guarded hash-cell estimate. Timeline cache budget is separate.
- Open/closed domains are supported. Periodic boundaries and spatial wet
  appearance are explicitly rejected. DEM contact grains, frozen carriers and
  non-body labels retain physical representation; whitewater retains its existing
  independent route. Canonical Fog/SDF data do not depend on sample count.
- Collision support is the current voxel mask, not exact mesh contact. Visual
  refinement does not increase physical resolution or reproduce extra grain
  contacts. Angular advection is not implemented; birth orientation stays fixed.
- Current source-object selection and distance/budget mesh split are reused.
  Weighted palette authoring, new screen-space LOD/hysteresis, GPU-resident transport
  and shared Fog/SDF snapshot adapters remain roadmap work. Static mesh banks are
  the intended source; arbitrary skinned/deforming source acceptance is not claimed.
- No FPS/VRAM improvement is claimed without device measurements. Reducing physical
  carriers still needs a separate physics-resolution convergence comparison.

## Verification

Completed source checks: MSVC `/Zs` (no object/executable), project/filter XML and
registration checks, Python/PowerShell parsing, descriptor regeneration/capability
coverage and whitespace checks. Three standalone test sources pass `/W4 /WX /Zs`.
These are source checks, not executed numerical or visual acceptance.

User-run C++ numerical tests, from a Developer PowerShell at repository root:

```powershell
./scripts/test/run_domain_visual_core_tests.ps1
```

They cover analytic affine MAC sampling, immutable storage, bad stencils and
budgets, uniform transport, redraw stability, checkpoint replay, atomic failure,
reseed hysteresis, permutation invariance, volume accounting, same-publication
budget changes and strict settings/JSON roundtrips. They do not build Studio.

After the user **full C++ build**, open Studio on a dedicated paused seeded
continuum domain routed to Splat, with valid physical mass state. External tests:

```powershell
python scripts/test/rt_test_domain_visual_samples_ipc.py "T_FieldSamples"
# Also compares embedded Python and a temporary two-object source palette.
python scripts/test/rt_test_domain_visual_samples_ipc.py "T_FieldSamples" --python --mesh-substance Water
# Optional: advances the whole dedicated test scene.
python scripts/test/rt_test_domain_visual_samples_ipc.py "T_FieldSamples" --steps 10 --dt 0.0083333333
```

The paused test restores settings and checks stable pages, atomic validation,
limits, represented+unrepresented volume, and unchanged physical position/velocity
hashes. The optional dynamic path checks new publications and physical clock
progress. `--python` runs a short API-only embedded assertion (no embedded pipe
client); its temporary wrapper is deleted. `--mesh-substance` restores the original
binding and deletes its temporary source objects. Backend mode transitions,
project reopen, RAM/disk seek and device performance still need live acceptance.
At the latest check Studio was closed (named-pipe error 2). The user is taking a
full C++ build. A final normal Build is needed for the last CPU-flat-source and
cross-frame undo corrections (`ParticleRenderBridge.cpp`, `RtApiDomainVisual.cpp`).
Verification captures, if needed, stay in ignored `docs/dev/.captures/` and are
removed after verification. Shader source/ABI did not change in this integration;
there are no affected shader variants to compile.
