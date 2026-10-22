# Gas/VDB material authoring — physical cloud and fire response

Status: implemented in source, not built or live-verified by the agent.
Priority: Vulkan RT, then RayFusion. No application launch or compilation.

## Dark imported clouds

Do not add emission to compensate for missing scattered light. Source inspection
found a camera-step-dependent suppression of sky scattering in the RT march:
an extra squared `sigma_s * step / cutoff` factor after Beer-Lambert integration.
It attenuated thin samples again and changed brightness with primary spacing.
That suppression is removed. The explicit authored density cutoff still applies.

The higher-order sky estimate now uses optical depth of the representative sky
chord, not the primary camera step. Its added visibility is bounded:

`multi * albedo * (1 - T_sky) * [1 - (1 + tau) * exp(-tau)]`,
where `tau = -log(T_sky)`. It vanishes without external sky radiance. This is an
approximation, not a claim to solve multiple scattering exactly.

RayFusion previously multiplied scatter colour into sigma_s itself, changing
extinction when the albedo colour changed. It now uses scalar sigma_s for
extinction and applies colour only to scattered energy, matching RT semantics.
World-mode environment light no longer has the arbitrary 0.18 multiplier.
Representative sky self-shadowing and the same bounded higher-order helper are
used with a minimum eight-sample cache stride. Studio preview environments keep
their separate artistic multiplier. The additional sky shadow march has a cost;
this increment does not claim cost-free lighting or physical RT/RayFusion parity.

`Physical Cloud` opts into the RT multiple-scattering walk, white albedo,
sigma_s 1, sigma_a 0.001, forward g 0.75, no backward-lobe contribution, 128 exact
events, full shadow strength, 24 shadow steps, stride 1, 1024 primary-step ceiling
and half-voxel spacing. It disables emission. It preserves source channels,
density scaling/remap/cutoff and temperature range because their units and values
depend on the imported asset. There is no automatic classification of an imported
VDB as cloud versus smoke/fire.

The physical preset is more expensive. The walk has finite event and sample
budgets; high optical thickness can still lose energy at the event ceiling.
Active spatial graphs, overlapping companion media and emissive volumes retain
the documented fallback march. Scaling imported geometry or density changes
optical depth. A black environment cannot light a cloud. Those conditions need
diagnosis rather than increasing emission. No particular Embergen/Houdini asset
has been measured or visually accepted in this task.

## Shared authoring operations

UI: existing gas/VDB material panel > **Physical volume / fire response**.
All new UI controls and bindings use `volume_material::configure` atomically.
Original material controls remain available. Graph outputs may override them.

Python:

```python
print(rt.volume.list_materials())
# Use an exact returned target: gas:<domain name> or vdb:<volume data id>.
rt.volume.set_material("vdb:3", preset="physical_cloud")
rt.volume.set_material("gas:Fire", preset="artistic_fire",
                       temperature_min=100, temperature_max=2500,
                       density_power=0.5, emission_exposure_ev=-1)
print(rt.volume.get_material("gas:Fire"))
```

IPC uses `volume.list_materials`, `volume.get_material {target}` and
`volume.set_material {target, settings:{...}}`. List/get are SceneRead; set is
SceneWrite and blocked during an active final render. Use `scripts/test/rt_ipc.py`
from an external process, never the embedded workspace for live IPC probes.
World-cloud proxy materials are excluded; use the existing atmosphere service.
Missing/ambiguous targets, unknown fields, bad types, non-finite values, inverted
ranges and invalid ramps fail before material mutation. No partial preset changes
are committed. Successful edits republish the volume packet and restart rendering.
Uninitialized targets can be configured but cannot yet be read as authored material.

Selected fields and accepted ranges:

| Fields | Range |
| --- | --- |
| `preset` | physical_cloud, blackbody_fire, artistic_fire |
| `density_multiplier`, `density_remap_low/high` | 0..10000; high > low |
| `density_cutoff` | 0..1 |
| `scatter_color`, `absorption_color` | RGB 0..1 |
| `scatter_coefficient`, `absorption_coefficient` | 0..10000 per world unit |
| `anisotropy`, `back_anisotropy` | -0.99..0.99; -0.99..0 |
| `lobe_mix`, `multi_scatter`, `shadow_strength` | 0..1 |
| `random_walk` | boolean; RT only |
| `max_scatter_events`, `exact_scatter_events` | integers 8..512; 1..512 |
| `max_steps`, `shadow_steps`, `shadow_stride` | integers 16..1024; 0..48; 1..16 |
| `voxel_step_multiplier` | 0.1..8 |
| `adaptive_stepping` | boolean; enable source-voxel spacing instead of fixed step size |
| `emission_mode` | integer 0 off, 1 constant, 2 blackbody, 3 channel |
| `emission_strength`, `blackbody_intensity` | 0..100000 |
| `emission_color` | RGB 0..10000 |
| `temperature_min/max` | 0..100000, max > min; retain current source semantics |
| `temperature_scale` | 0.001..100; colour mapping |
| `response_enabled`, `ramp_enabled` | boolean |
| `temperature_power`, `density_power` | 0..8; 0..4 |
| `emission_exposure_ev`, `emission_saturation` | -16..16; 0..2 |
| `color_ramp` | 2..256 strictly increasing [position,r,g,b] stops; optional fifth alpha |

Ramp positions are 0..1, RGB 0..10000, stored alpha 0..1. Alpha is preserved on
round-trip but remains unused by GPU emission. The existing eight-stop GPU packing
still approximates larger ramps. Null is rejected; this API is an explicit overlay,
not JSON merge-patch. Omitted settings retain their value. Presets apply first,
then explicit values in the same call; no stale old settings overwrite a new recipe.

## Independent fire intensity

Old scenes default to `response_enabled=false`, retaining their original emission
equation. Opt-in response replaces that equation after blackbody/ramp colour:

`source = graded_color * u^temperature_power * density^density_power * strength * 2^EV`.

Here `u` is clamped temperature within the authored interval, before the colour
temperature scale. At u=0 thermal emission is zero even when power=0. Empty
density is always zero; density power 0 affects only positive cells. Constant
emission omits the thermal term. Saturation uses linear Rec.709 luminance,
removes negative channels and preserves HDR. Flame boost still applies afterward.
The existing 64-luminance limiter remains on temperature-driven emission.

This separates ramp hue from strength. `Blackbody Fire` selects blackbody colour,
power 4, density power 0 (density remains in transport extinction).
`Artistic Fire Ramp` selects eight low-red through white-hot knots, power 1.5 and
density power 1. Both preserve source temperature calibration. The first uses an
RGB blackbody fit and an authored normalized strength law, not calibrated Planck
spectral emission or a complete LTE material. The artistic ramp is explicitly an
artistic alternative. Old ramp defaults are not silently rewritten.

RT camera emission, sampled fire emitters and RayFusion share the response GLSL.
The existing five retired cloud floats at offset 444 carry the response, keeping
the Vulkan volume-table stride 624. C++ host transfer structs grow, so rebuild C++
and shaders together. New response data is saved by VolumeShader and both manual
ProjectManager serialization paths. CPU/OptiX do not yet evaluate the new response;
they are not advertised as matching this Vulkan-first feature. New controls are
not yet exposed as dedicated timeline tracks.

## Verification and remaining roadmap

Agent-run, non-build: `python -B scripts/test/check_volume_material_contracts.py`,
Python AST syntax inspection and targeted `git diff --check`. The source contracts
check raw packet offsets/stride, shader consumers, all three transfer paths,
UI/Python/IPC core wiring, project source registration and manual persistence.
These do not substitute for compiler or render verification.

User checklist:

1. Build C++ and recompile `volume_closesthit.rchit` and
   `material_preview_volume.frag` with `volume_material_response.glsl` available.
2. Compile/run `scripts/test/volume_material_authoring_test.cpp`, linking
   `VolumeMaterialAuthoring.cpp` and the normal Vec3 implementation, assertions on.
   Run the existing ramp-packing regression too.
3. Open Studio manually, then externally run
   `python scripts/test/rt_test_volume_material_ipc.py` to list targets;
   add `--target "vdb:3" --apply` to test overlays, rejection and restoration.
4. Import a known cloud VDB, select Physical Cloud, keep emission off, compare
   RT/RayFusion under non-black world/HDRI light. Save/reload both gas and VDB.
5. With fixed lighting, camera, samples and density, compare primary voxel spacing
   0.5/1/2. Brightness should converge rather than dim from the removed artificial
   factor. Compare higher step/event budgets separately from denoising.
6. On a temperature VDB or live gas, compare legacy response against enabled
   response, blackbody and artistic ramps. Test power 1/2/4, density power 0/1,
   EV -1/0/+1, saturation 0/1/2; test cold/empty regions and emitter-lit smoke.
   Check radiance before tone mapping and profile the new cached sky shadow work.

Still pending: explicit source temperature units/fallback policy, general density
gamma/curves with conservative majorants, transmission-distance authoring,
full-resolution ramp storage, timeline tracks and live physical/visual acceptance.
The overall roadmap is not marked complete.


## Density transfer curve — 2026-10-20

`density_curve` is shared by the material UI, `rt.volume.set_material`, and
`volume.set_material` IPC through `volume_material::configure`. Example:

```python
rt.volume.set_material("gas:Ground Burst Gas", density_curve={
    "enabled": True,
    "values": [0, 0.03, 0.10, 0.25, 0.45, 0.65, 0.85, 1]
})
```

Use the actual target returned by `rt.volume.list_materials()`.
IPC uses the same object under `settings.density_curve`.
The object accepts only `enabled` (boolean) and `values` (eight finite numbers).
Omitted members reset to identity defaults. Invalid patches fail atomically.
Input knots are fixed at i/7. Outputs must be nondecreasing in [0,1], with the
first exactly zero. Linear interpolation avoids overshoot. Above input 1,
output is input times the final knot; the source HDR range is not clipped.

Order: source sampling -> existing low/high normalization -> density curve ->
existing cutoff/feather -> density multiplier. Old projects load with the curve
disabled. The density settings serializer also covers both manual project-save
paths. Presets preserve the density curve and source data.

Vulkan RT camera, surface-pass volume shadows, indexed random-walk sampling,
leaf majorants and dense-block skip maxima use the curve. RayFusion camera,
volume self shadows and dense-block skips share its sampler. Monotonicity is
required so transformed maxima remain conservative. CPU/OptiX do not implement
this new transfer curve yet; do not use them to judge its appearance. Existing
Volume Graph density overrides retain their established behavior.

This changes material extinction, not the grid/cache or solver resolution.
Existing emission equations that depend on sampled material density also respond
to this curve. Independent raw-density emission masking is a later task.
Low curve outputs can fall below cutoff: reduce cutoff for a fair wispy-detail
comparison. A very narrow source-density range should first be expanded with
Density Remap Low/High, rather than crowding all useful values into the first
fixed knot interval.

### Build and verification handoff

No application/shader build was run by Codex. The volume record grew from 624 to
672 bytes (curve starts at byte 624). Rebuild the C++ application and all four RT
stages declaring VkVolumeInstance: volume_closesthit.rchit, closesthit.rchit,
raygen.rgen, volume_intersection.rint. Rebuild material_preview_volume.frag,
material_preview_sdf_surface.frag and every shader including
surface_sdf_shadow_query.glsl. Do not mix old SPIR-V with the new application.

Run the standalone volume_material_authoring_test.cpp with the usual project
include/link environment; it covers serialization, monotonicity, packing,
zero preservation and atomic rejection. External live IPC verification:

```text
python scripts/test/rt_test_volume_material_ipc.py --target "<listed target>"
python scripts/test/rt_test_volume_material_ipc.py --target "<listed target>" --apply
```

The second probe temporarily changes the material and restores it in finally.
Compare the same cached frame in RT and RayFusion with curve disabled, enabled
identity, and the example above. Check shadow changes, thin wisps, multiple
volumes (ABI stride), save/reopen, and an entirely empty source. Identity should
match disabled within render noise. Do not regenerate the simulation for this
comparison. Ten non-build source contracts passed; compiled/runtime validation
remains with the user.

## Deferred core-light plan

Fire presets now select internal_light_steps=0. Existing saved settings are
preserved; setting this field to zero explicitly disables an older active gather.
The expensive stochastic internal gather remains an experimental opt-in.

Next establish stable temperature units/range and an independent scalar
intensity curve. Then add raw-density emission masking with amount and soft
transition, plus a separate hot-core control. Evaluate each stage on the same
cached frame without bloom. Bloom comes after HDR emission and should not be
used to claim smoke illumination. If surrounding smoke still needs actual
incident light, compare imported Houdini scatter data or a bounded deterministic
bake/cache as a separate solution, with measured cost and temporal stability.
Do not re-enable the noisy per-sample RT gather in presets as a shortcut.


### Compact density profile editor

The density UI now shows one enable checkbox, a Reset button and a paintable
profile canvas. Drag with the left mouse button to paint directly on the graph;
fast strokes interpolate across crossed knots. The eight fixed source knots and
canonical density_curve API are unchanged. Painting adjusts neighbouring values
to keep the profile monotone and keeps the origin zero. Interaction help lives
in the graph tooltip. Separate numeric sliders and the explanatory paragraph
were removed. This UI-only revision requires a C++ rebuild; shaders are unchanged
from the density-curve build. Check click/drag, fast strokes in both directions,
reset, disabled state, narrow dock layout and switching materials after rebuild.
Codex did not build or launch the application.
