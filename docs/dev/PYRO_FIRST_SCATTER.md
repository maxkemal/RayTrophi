# Vulkan pyro: source-sampled first scattering

> **Durum:** ARŞİV — fire lights smoke 2026-11-01 ölçümle söküldü; ayrıntı PYRO_SOURCE_CHANNEL_HANDOFF_20261101.md

Source update 2026-11-01: PYRO_INDEPENDENT_EMISSION_20261101.md supersedes the
initial global proposal with receiver-local sampling and opt-in coefficient j.
The sections below describe the initial 2026-10-31 implementation/history.
Updated source still awaits a user build and GPU acceptance.

Live feedback 2026-11-01: `PYRO_FIRST_SCATTER_LIVE_20261101.md`. New controls
and candidate execution were confirmed after the user's build. The desired
dark-shell/interior-light appearance is not accepted: the existing direct
emission still depends on smoke density, and global source proposals can be
poor for optically local transport. Do not count compilation as visual success.

2026-10-31. Source implementation; user-run C++/shader builds and GPU acceptance
are pending. No app launch, scene edits or new performance capture in this change.
Existing baselines: `rt_pyro_vulkan_cost_20261031.json`,
`rt_pyro_backend_compare_20261031.json`, `rt_gas_integrator_parity_20261028.json`,
`RT_VOLUME_STAGE_SPLIT_COST_20261030.md`. This is a targeted lighting addition,
not a rewrite of Vulkan gas integration.

## Transport

The Vulkan publisher zeros the old dense emitting-block list. Its old march NEE
block was therefore unreachable for published gas, and imported NanoVDBs never
had that proposal. The new source sampler replaces that dead march block.
`internal_light_steps` instead samples a full chord along a
phase-sampled direction with at most 32 taps; thin source support can be missed.

The new path samples **source positions**, then transports emission to smoke.
It does not allocate an RGB voxel grid or persistent cache. Direct emission and
camera transmittance retain their existing definitions.

1. Uniformly pick an active opt-in emitter, including sources outside the camera
   interval. Weight by source count. All selection occurs once per camera segment.
2. Propose points from existing float NanoVDB leaf topology. If available, use the
   smaller density/temperature topology. Only breadth-first fixed-size float
   leaves can be indexed. Temperature **values** never become a new energy proxy.
3. Mix 7/8 sparse-leaf with 1/8 uniform AABB proposals. Leaf cells partition index
   space at `8*k - 0.5`. Evaluate the full mixture PDF, including box-generated
   samples. The box retains support for interpolation halos, tiles and background.
   Non-indexable trees and live dense gas use the box proposal in this version.
4. Evaluate `sampleDensityAcc` and the existing `volumeEmissionAt`. These retain
   remap/curve/cutoff, units, response, ramp/blackbody, mask, flame and source limit.
   The camera integrates source function `S*(1-exp(-sigma_t*dt))`; volume lighting
   needs the corresponding **RGB `j = sigma_t*S`**, not bare S or temperature.
5. Reservoir selection weights luminance(j/PDF). Retain the RGB estimate
   `(j/PDF)*sumWeights/(candidateCount*selectedWeight)*sourceCount`.
   Dark candidates remain in candidateCount. The new term is outside the legacy
   in-scatter cap 8 and receives no extra estimator-radiance clamp.
6. At a smoke sample use `j*phase*T/distance^2`, with receiver scattering/albedo
   supplied by the existing march. Visibility uses physical RGB exp(-tau) through
   all gas contributors, solid probes and the existing SurfaceSDF arbiter.
   Artistic scene-light shadow strength does not replace extinction here.

The selected point stays fixed while visibility follows the existing shadow
stride. Ordinary in-scatter still observes light stride. Material edits use the
existing common accumulation reset/publication. Each new dispatch samples current
fields; no persistent fire cache can retain the previous frame or material.
Source density attempts are charged to the existing segment density counters
without inventing extra rays; visibility density taps go to shadow counters.

## Scope and approximations

- Vulkan **deterministic camera march**, ordinary and general-mixture paths.
  Works without external scene lights. Enable the option on the emitter.
- Actual random walks already gather emission along flights. Adding this term
  there requires MIS. Disable random walk on **all** contributors for acceptance.
  This is first scattering, not full multiscatter.
- Any active gas Volume Graph disables the new estimator in this first version:
  graph optical values must not be silently replaced by raw fields. Density
  noise without a graph still uses the shared sampler.
- OptiX/RayFusion do not implement the new term. Live dense gas currently uses
  box proposals; sparse efficiency applies to NanoVDB, including imported/cache.
- Thin sources can still yield zero candidates and noise at finite spp. More
  candidates improve source selection, not retained shadow chords per update.
- Visibility has jittered `clamp(max(shadow_steps,8),8,64)` taps per intersected
  gas medium. Thin blockers can need more taps. The existing SDF arbiter blocks
  conservatively at its box when it cannot evaluate the surface.
- Near-field inverse-square is regularized at half a topology voxel's
  volume-equivalent width. This finite-voxel approximation, shadow quadrature and
  held visibility/light stride mean the whole render is **not claimed unbiased**.
  The isolated RIS normalization is checked exhaustively.
- No speed/image-quality claim before build/render acceptance. Source sampling
  and visibility add work. No full-domain scan, rebuilt source tree, full-domain
  RGB array or persistent atomic cache is introduced.

## Authoring and compatibility

UI: **Fire lights smoke (Vulkan RT)** and **Fire source candidates**.

| Common material key | Type/range | Default |
| --- | --- | --- |
| `fire_scatter_enabled` | boolean | false |
| `fire_scatter_samples` | integer 1..32 | 8 |

Independent of `response_enabled` and mask. UI, Python and IPC call the existing
`volume_material::configure` through the common material service: strict types,
unknown-key rejection and atomic patches. Saved in `VolumeEmissionResponse`;
legacy defaults off. Presets preserve both controls. Disabled candidate settings
do not split otherwise compatible merged streams. Masked-source separation stays.

Python authoring API (not for live IPC probes):

```python
rt.volume.set_material("gas:Nuclear Gas", fire_scatter_enabled=True,
                       fire_scatter_samples=8)
```

External IPC, separate process with Studio open:

```python
client.call("volume.set_material", target="gas:Nuclear Gas",
            settings={"fire_scatter_enabled": True, "fire_scatter_samples": 8})
```

`volume.get_material` round-trips both controls. The existing publisher packs
`density_curve[11]` as 0=off or 1..32 candidates; this word was reserved.
**VkVolumeInstance stays 688 bytes / 172 words.** The legacy internal integral is
suppressed on opt-in sources in Vulkan RT/preview even with saved nonzero steps.
Legacy controls stay serialized/editable; their UI is disabled while opted in.
The old machinery has not been removed from every backend.

## Verification and user acceptance

Non-build `python scripts/test/check_volume_fire_scatter.py`: exhaustive finite
RIS outcomes with dark/occluded candidates, RGB energy, source count, proposal
support/scale and host/shader/API wiring. Not a GPU render or shader-compilation
substitute. `volume_material_authoring_test.cpp` adds serialization, legacy default,
packet and atomic-invalid-edit cases; agent did not compile it.

Completed non-build verification: 74 math/source-contract tests across fire
sampling, mask, material authoring, temperature, RT optical sampling, stage split,
mixture light cache and step budget passed, plus the light-cache ABI contract.
The post-build IPC probe passed Python syntax inspection only; it was not run
against the older open binary.

After the user's C++ and shader build, from an external terminal:

```powershell
python scripts/test/rt_test_volume_fire_scatter_ipc.py --target "gas:Nuclear Gas"
```

The probe restores changed controls and never saves/resets simulation.

Use the existing explosion, fixed frame/camera/post, Vulkan Rendered, random walk
off and legacy internal steps 0. Compare off/on at the same accumulation target,
initially 8 candidates, keeping source/density settings fixed. Look for smoke
lighting near flame; the reference appearance is not guaranteed by this term.
Check no scene lights, a blocking solid, full mask, frame change, two sources,
scatter=0 on receivers and emission=0 on sources. Opaque smoke must attenuate
the contribution. 1/8/32 candidates should change noise, not systematically
multiply energy. GPU quality, noise and driver acceptance remain unverified.

After this contribution is visible, decide whether to cache incident lighting
with receiver direction kept separate and improve reusable power proposals.
Do not add secondary J (direct emission/external lighting included) to primary;
it would count unrelated terms again.
