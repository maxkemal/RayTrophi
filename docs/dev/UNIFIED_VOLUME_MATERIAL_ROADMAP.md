# Unified volume material — atmosphere, VDB and gas

Date: 2026-10-20. User approved planning and starting implementation.
Status: increment 1 implemented in source; build and live acceptance pending.

Source-fidelity/internal-light continuation: midpoint scalar traces now match on
CPU/GPU; emission-to-smoke field integration is wired for RT and RayFusion with
shared authoring controls. Old cache frames are unchanged. See
[implementation, limitations and user checks](GAS_DETAIL_AND_INTERNAL_LIGHT_IMPLEMENTATION.md).

Gas/VDB implementation continuation: shared validated material authoring, Python
and IPC operations, physical-cloud and fire presets, independent fire intensity,
RT step-dependent sky suppression repair and RayFusion optical-colour correction
are now in source. See [implementation and acceptance](VOLUME_MATERIAL_AUTHORING.md).
The new code has not been built or live-accepted; remaining work is listed there.

2026-10-20 continuation: user reports increment 1 built and confirms saturation
behaviour on white input. Gas/VDB detail audit and canonical ramp-packing repair
are in source; see [detail audit](VOLUME_DETAIL_AUDIT_2026_10_20.md). This repairs
discarded ramp tails above eight stops without a GPU ABI change. Independent
intensity response, temperature units and density shaping remain pending.

## Product direction

Use the Octane Standard Volume panel supplied by the user as a reference for
density, scattering, transmission and emission separation. Keep one vocabulary
and shared evaluation math across atmosphere clouds, imported VDB and gas.
Do not replace the atmosphere's CloudState authority with its derived VDB proxy.
Source fields remain distinct; transport models retain explicit capabilities.
Vulkan RT is the reference priority, followed by RayFusion. Legacy CPU/OptiX
atmosphere approximations are not advertised as equivalent implementations.

All user operations must use the same core through UI, Python and external IPC,
with validation, atomic rejection, serialization, animation and documentation.
Existing files over 2000 lines receive integration wiring only. No agent builds,
shader compilation or Studio launch. Preserve unrelated working-tree changes.

## Delivery sequence

1. **Uniform colour controls — started.** Shared linear-light saturation and
   emission exposure helpers; atmosphere master and per-layer controls; existing
   JSON service supplies Python/IPC, persistence and keyframe interpolation.
   Resolve before GPU upload: no added texture reads, march steps or GPU fields.
   Neutral defaults must preserve existing packets. Test composition, albedo
   bounds, HDR preservation and rejection without committing partial patches.
2. **VDB/gas colour integration and service audit — pending.** Reuse the helper
   after graph evaluation where appropriate. Audit native grids, procedural
   volumes, graph outputs, UI, script/IPC and backend packing. Constant emission
   grading can be uniform; blackbody/ramp grading must occur after colour
   evaluation and cannot be faked by changing only the constant emission colour.
3. **Density shaping — pending.** Input min/max, multiplier, contrast/gamma and
   soft cutoff, then an optional curve. Use identical evaluation for camera,
   shadow and light samples. Preserve conservative majorants and empty-space
   skipping; zero input must remain empty. Define behaviour above input maximum.
   No extra noise fetches or step budget increases in the atmosphere fast path.
4. **Transmission authoring — pending.** Add transmission colour at a reference
   distance as an alternative editor for canonical absorption coefficients:
   sigma_a = -log(T) / distance. Specify units, T=0 handling and migration.
   Coloured absorption in atmosphere needs a transport audit, not just UI.
5. **Fire field semantics — pending.** Explicit Kelvin/Celsius/normalized
   temperature, scale/offset, fixed remap interval, missing-channel policy;
   independent colour ramp and intensity curve; blackbody/ramp blend, ignition
   threshold and density influence. Auto-ranging must be stable over animation.
6. **Transport parity — pending.** Audit Vulkan main/shadow/emission sampling
   and RayFusion consumers for temperature radiance, density coupling, reaction
   emission and hard radiance clamps. Document approximate versus reference
   multiple scattering. Never silently change existing projects' brightness.
7. **Fast cloud lighting — pending.** Expose existing powder/approximate multiple
   scattering only after checking physical controls and current product policy.
   These are fast-model controls, not extra brightness applied to reference RT.
8. **Acceptance — pending.** Old-project neutrality, master/layer composition,
   presets, animation, save/reload, script/IPC rejection and renderer comparison.
   Compare cloud/gas/VDB under fixed lighting and exposure. Measure frame cost;
   unchanged lookup count alone is not proof of unchanged timing.

## Increment 1 contract

Cloud material fields, accepted for master `material` and custom layer material:

| Field | Default | Range | Meaning |
| --- | --- | --- | --- |
| `scattering_saturation` | 1 | 0..2 | Linear Rec.709 luminance blend; clamp albedo to 0..1 |
| `emission_saturation` | 1 | 0..2 | Independent emission colour saturation; remove negative RGB |
| `emission_exposure_ev` | 0 | -16..16 | Multiply emission by 2^EV; preserve HDR |

Zero saturation yields luminance grey. One is exactly neutral. Scatter grading
is energy bounded; saturation above one may clip and does not preserve luminance
after clipping. Exposure affects authored cloud emission only, not sunlight,
ambient illumination, lightning or the composited sky. It is not a general cloud
brightness slider. Master and custom-layer colours are graded independently,
then existing master/layer multiplication and emission addition apply.
Inputs must be finite numbers in range; malformed types and ranges are rejected.
Missing fields retain neutral defaults when loading old files; API overlay
patches preserve fields not sent. Fields interpolate as floats in keyframes.
The existing API uses JSON merge-patch: sending a field as `null` removes that
key and restores its default. Direct serialized material input with a present
non-numeric field (including null) is rejected. This increment preserves that
existing API reset convention.

Existing services need no new entry point:

```python
rt.world.set_clouds(material={
    "scattering_saturation": 0.7,
    "emission_saturation": 1.2,
    "emission_exposure_ev": 1.0,
})
```

External IPC: `world.set_clouds` with the same `material` object; read back using
`world.get_clouds`. UI uses existing World > Clouds / [World] Clouds controls.
No GPU packet layout or shader source changes are needed for this increment.

## User build and acceptance checklist

- Build the C++ application using the normal project configuration. This increment
  does not require shader changes. The agent has not compiled or launched it.
- Build/run `scripts/test/volume_appearance_test.cpp` against the normal Vec3
  implementation; assertions must be enabled. This checks actual C++ helpers.
- Open Studio manually. In a separate terminal run
  `python scripts/test/rt_test_cloud_material_ipc.py` for read-only schema inspection;
  add `--apply` for scene-mutating round-trip, rejection and GPU density checks.
  The probe restores cloud settings in `finally` and resets accumulation.
- In Vulkan RT and RayFusion, compare neutral defaults against an old project;
  test coloured scatter at saturation 0/1/2 and constant emission at EV -1/0/+1.
  Emission strength must be nonzero to see exposure. Compare cloud radiance before
  tone mapping: +1 EV doubles the emission source, not the entire lit cloud.
- Check master plus custom layer material, cirrus, save/reload and keyframes.
  Changing saturation/exposure must not change density queries or shadows.
- Profile both backends under fixed settings. No claim of measured zero cost is
  made until profiling; this increment adds only uniform CPU-side colour math.

## Verification status

`git diff --check` and IPC probe Python AST parsing passed. The existing
`check_cloud_shape_contracts.py` ran 15 tests with five failures: the stale fixed
packet-size expectation and density-wrapper/texture-read expectations. Repeating
with HEAD versions of edited consumers produced the same five failures. Shader
sources and GPU struct declarations are unchanged by this increment. These
pre-existing test failures were not hidden or rewritten as passing checks.
C++ numerical test execution, compilation, GPU visuals, live IPC and timings
remain pending user verification. The full roadmap is not complete.
