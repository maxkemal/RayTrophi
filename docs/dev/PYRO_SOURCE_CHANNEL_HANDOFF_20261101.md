# Pyro source channel correction and handoff — 2026-11-01

> **Durum:** AKTİF — pyro tepki zinciri; dört parti canlı ölçüldü, beşinci parti (eklemeli flame, iç ışık + burnout, panel) yazıldı, derlenmedi; duman renk rampası ve koyu nokta bulgusu açık

## Handoff: internal light step fix and cost, 2026-11-01

Step fix MEASURED, in commit 34214c0d: rt_test_pyro_internal_ipc.py PASS on rebuilt
shaders. volume_internal_integral.glsl starts at one voxel and grows segments by
1.35, so steps sets reach, never spacing.

Frame 69, burn material, 32 spp, same process:

| internal steps | ms | p50 | specks |
|---|---|---|---|
| off | 100, 105 | .622 | 38 |
| 1 | 245, 246 | .623 | 33 |
| 2 | 199 | .625 | 35 |
| 8 | 199..265 | .635 | 60..72 |
| 32 | 225..290 | .636 | 67..70 |

Energy no longer depends on the budget: 8 and 32 are .636 and .637 in the test,
frames 12 and 30 identical to three digits. 1 and 2 steps do not reach any fire.
Use 8: same light for less or equal cost.

Cost: one marched voxel already costs about +140 ms, so it is not the march and not
extra rays (camera_volume_rays 3.09 M either way; the gather is not counted). The
one heavy thing left per gather was the nested traceRayEXT in rwSolidDistance, from
inside the volume closest hit. Not proven: inferred by elimination.

MEASURED, confirmed: gather now uses inline rayQuery (volumeInternalSolidDistance,
GL_EXT_ray_query in volume_closesthit.rchit). Same frame and material:

| internal steps | nested trace ms | rayQuery ms | p50 before, after |
|---|---|---|---|
| off | 100, 105 | 105, 109 | .622, .622 |
| 1 | 245 | 110, 113 | .623, .623 |
| 8 | 199..265 | 119..122 | .635, .635 |
| 32 | 225..290 | 124..136 | .636, .636 |

Frame 30, 32 steps: 142 to 80 ms against 73 off. Frame 12: 56 to 54 against 51.
Fire pixels and p50 match to three digits; test PASS. Internal light now costs about
+15% at frame 69 instead of +130..170%. The per-gather cost was the nested trace from
the volume closest hit, and that is the lesson: a nested traceRay in this shader is
not the price of one ray.

Accepted change: alpha-tested geometry blocks fire light as if opaque. A device
without rayQuery fails the pipeline: the capability is optional in VulkanBackend.cpp
and no fallback exists. Not checked: a solid inside the gas (scene has only the
ground), panel, RayFusion.

## Handoff: internal light and additive flame MEASURED, 2026-11-01

User built, agent opened nuke.rtp and measured, project not saved, material
restored. rt_test_pyro_flame_ipc.py PASS. rt_test_pyro_internal_ipc.py gave FAIL
twice, both test faults, both fixed afterwards and NOT rerun: flame 0.1 was too weak
to detect, then frame 69 was checked above the horizon while its flame lies on the
ground. smoke_* view checks passed in both runs.

Burn material (M 3, EV -2.32, power 3, burnout .9 band .2-.6), 32 spp:

| frame | off ms | internal ms | p50 off, on | fire px off, on | specks off, on |
|---|---|---|---|---|---|
| 12 | 29 | 54 | .723, .796 | 10134, 10339 | 0, 0 (first run 51, 4) |
| 30 | 73 | 168 | .757, .792 | 21377, 21586 | 37, 14 (first run 76, 6) |
| 69 | 125 | 319 | .622, .641 | 52986, 61684 | 38, 78 |

Internal light with independent and burnout works and fills the unlit folds. Runs
differ up to 30% between sessions, so use the ratio: about 2.6x at frame 69. It
is cheap only relative to the walk (757 ms, measured earlier on other settings).

Sweep at frame 69, 32 spp:

| steps, stride | ms | p50 | specks |
|---|---|---|---|
| off | 125 | .622 | 38 |
| 32, 32 | 321 | .641 | 78 |
| 16, 32 | 321 | .666 | 267 |
| 8, 32 | 317 | .716 | 701 |
| 4, 32 | 319 | .798 | 919 |
| 32, 8 | 318 | .642 | 62 |
| 8, 4 | 285 | .718 | 528 |

Two defects read from this, neither fixed:

- Step count changes energy, not quality: dt = distance/steps, so few steps land
  far inside hot gas before any attenuation. 32 is the clamp and may not be
  converged. Fix candidate: step by voxels, keep steps as a budget.
- Cost ignores steps and stride. Opaque smoke ends the inner march early and the
  camera ray ends before a stride elapses, so each ray likely pays one gather.
  Cause not measured. rwSolidDistance is a candidate, but the 2026-10-06 walk note
  found cutting those probes by 64% left render time unchanged, so measure first.

Additive flame works and is not a cap control. Flame test, 2.0: frame 12 p95 .763
to .876, frame 30 contrast .364 to .402, frame 69 p95 .803 to .813. The visible
result is a glowing ring on the ground and traces along the stem; the cap barely
moves. That is where burning cells are, which explains the zero thermal base. 0.05
to 0.5 was wrong as a start: 0.1 was undetectable, 0.5 and 2 are visible. Keep 0.

Specks: internal light removes most at frames 12 and 30, doubles the count at 69,
where more of it is likely sparkle from the single direction per ray. The counter
also reads smoke folds as specks, so legacy 340 at 69 means nothing. The legacy test
column is not the user's look: only filtered keys were copied, and it blew out to
white. It proves internal light is on, nothing else.

## Handoff: additive flame, internal light with burnout, panel, 2026-11-01

Written by the agent, built by the user. Numbers in the section above.

User finding that drove it: with independent off, internal light 32/32 and
flame_boost 0, the look is better than the random walk and cheaper, no dark specks
seen. Live material read: M 8.33, EV 4.22, burnout .30 band .99..1, walk off.

Three defects of ours, found reading that material:

1. volumeInternalLighting returned 0 when density_curve[11] > 0. That gate belonged
   to fire scatter; the slot became flame_boost with default 1, so internal light was
   silently off unless boost was 0. The user only had it because they had zeroed it.
   Gate removed in both renderers, and a contract check now forbids it.
2. volumeSootScale read flame without checking independent, against its own comment,
   so legacy materials lost extinction where they burn. Now returns 1 first. The
   preview always had this check. A legacy scene with burnout set changes slightly.
3. Specks were measured in independent mode; the user's view is legacy, so "no
   specks" there does not clear them. Still open.

Changes:

- Additive flame: `flame_boost` renamed `flame_emission`, default 0. Adds
  volumeFlameRadiance (flame * gain * intensity * exposure) with the ramp read at
  max(u, flame), so a burning cell glows on a zero thermal base. Multiplicative gain
  is gone. Old saved key is ignored by design.
- Internal light now works with independent emission, so it and burnout can be on
  together. volumeInternalRay uses volumeEmissionIntegrate (j or S), reads emission
  density through new sampleDensityChannels, and burnout thins its path too. Legacy
  integration is arithmetically unchanged. RT sampleDensity now shares that body.
- smoke_color, smoke_opacity, smoke_tint: appearance view on volume.get_material and
  volume.set_material. Opacity = sigma_s + sigma_a, color is exact per-channel albedo
  (extinction ignores scatter color). First edit normalizes the stored four values
  without changing the look. Raw keys and smoke keys in one call are refused.
- Panel: sections Smoke, Lighting, Fire, Quality, Data (Data only for gas domains).
  Raw coefficients moved to a tree, wall text moved to tooltips, ramp histogram added
  under Measure. Sections, not tabs: the panel theme already uses them.

Decided against for now: smoke color ramp. It needs a LUT buffer and an ABI change,
and internal light plus burnout may change what it should be bound to.

## Handoff: flame batch built and measured, 2026-11-01

Built Release plus shaders by the agent under the user's earlier authorization,
nuke.rtp, material M=3, EV -2.32, power 3, 128 steps, 32 spp, denoiser off.
rt_test_pyro_flame_ipc.py PASS, material restored, project not saved.

Flame is the solver's `interaction` field. Correction to the plan: live gas
ALREADY reached the shader, through the device buffer
(gasGpuFieldView.flame_address, liveDenseGas path). The new third NanoVDB grid
only serves the host path (`up_flame` in scene_data.h, no live device field).
That path was NOT exercised by this test: treat it as unverified.

Removed in this batch: a publish_flame block in syncDomainFogVolume. Matter
domains return early above it, so it could never run. Source edit made after
the measured build, not recompiled.

| frame | burning / active | flame p50 | noflame p95, contrast | boost4 p95, contrast | burnout .9 p95, contrast | ms off, burnout |
|---|---|---|---|---|---|---|
| 12 | 89194 / 91697 | .990 | .810, .332 | .877, .399 | .927, .428 | 48.2, 50.0 |
| 30 | 138480 / 281440 | .013 | .828, .390 | .832, .394 | .933, .489 | 62.6, 70.1 |
| 69 | 25169 / 1102368 | .004 | .801, .414 | .801, .414 | .915, .518 | 84.6, 97.3 |

Reading: burnout is the lever (contrast +.10 at 30 and 69 for +12..15% time).
flame_boost costs nothing, but at 12 flame is saturated, so it acts as flat gain
in the fireball rather than detail, and it fades out by 30.

Why boost is zero at frame 69: not a broken chain. Thin smoke (M .05), unsaturated
exposure (ev -6: 112677 fire px, 0 saturated), boost 40 changed 0 pixels above the
ground. gas.field_stats runs the same formula on the host: boost 0 to 40 moves mean
emission 12.8416 to 12.8420 and max not at all. So burning cells have about zero
base emission, and a multiplicative gain has nothing to multiply. Not established
why (low density fringe is the guess, there is no joint flame/density/T reading).
An additive flame term would be the next step, a design decision, not made.

Measurement traps found here. viewport.status reaches 34 or 35 samples after a
reset, and one extra sample reshuffles only the ground sparkle: signed mean zero,
all of it below y 590, sky clean. That produced a fake non-monotonic boost1 at
frame 69. Compare above the horizon, or hold the sample count. A boost test on a
saturated image proves nothing; check exposure first. Before concluding "does not
reach the shader", use field_stats to see whether the base is zero.

Open finding, not fixed: dark specks on the fireball at 12 and 30. They are field
structure, not sampling noise: 115 of 184 sit in the same place in noflame and
boost4. Counted at 70% of local median: 68 at frame 12, 206 at frame 30, 1 at
frame 69. After rewinding to 0 and stepping to 12 again, 1 (also 0 at 128 spp). So
the first run and the rewound run differ, cause unknown.

fluid.inspect_fog_fields refuses this scene ("exceeds diagnostic memory budget");
166x256x166 at 9+2n floats per cell is over its 128 MiB. Same function guards
syncMatterFogStreams, so check it before trusting Matter stream publication on a
grid this size.

## Handoff: response batches built and measured, 2026-11-01

Built Release plus shaders by the agent on the user's instruction, opened
E:t_k
uke.rtp, stepped to frame 69 (peak T 6.02, reproducible), project not
saved. rt_test_pyro_response_ipc.py PASS twice, material restored.

Field, first real numbers, live material: T p50 7975 K, p95 14578 K, peak
18059 K; ramp position p50 .36, p95 .71; 551281 of 905634 emitting cells (61%)
were over the old limit; optical depth per voxel p50 3.7.

Soft limiter (log above 64, independent j only): response per EV at display L
.65 rose .016 to .048, at .75 rose .007 to .032. CUDA still wraps this in an
outer hard 64 in ray_color.cuh; OptiX is frozen, so it stays.

temperature_gamma plus gas.fit_temperature_range: works, fitted range 874..16688 K,
gamma 1.08, ramp p95 .71 to .85, whiter and p99 .988. No automatic use: EV must
come down after a fit. gas.field_stats is the new measurement.

Fire lights smoke REMOVED: +.0035 mean for 3x cost against walk +.12. Keys are
gone; old projects ignore them. Cold RT pipeline 457 to 287 s (-37%). density_curve[11]
is free.

Random walk on emitters works (routed to the mixture walk). Indirect flights
now soft-limit gathered emission and use the bounce step scale: 993 to 757 ms,
mean unchanged (.6386 to .6353). Noise barely moved (cap .149 to .145): sparks
remain at 64 spp, so the limit knee of 4 is not the main noise source. It costs
about 7x the march and also lifts the stem, so it is a final-render tool, not
accepted as a default. Unmeasured: convergence past 64 spp, SVGF on it.

Not done: RayFusion look of burnout, gamma and limiter; M=0 manual check; flame
publication on the NanoVDB path; burning_cells by frame; smoke appearance panel.
C++ unit tests were edited but not run. Stale source contracts still fail:
check_volume_march_lod_contracts (volumeShadowDue), check_gas_overlap_cache_contracts,
check_volume_optical_block_contracts, check_volume_shared_walk.

## Accepted measurements: bounce cache and heat response, frame 69

Bounce cache: rt_test_pyro_bounce_cache_ipc.py PASS, 64 spp, against the full march.

| Material | Full ms | Cache ms | Cap | Ground drift |
| --- | --- | --- | --- | --- |
| M=3, EV-5, burnout .9 | 188 | 94 | identical | +0.8/255 |
| same, EV-3 | 168 | 101 | identical | +2.1/255 |

407402 cached bounce marches against 61 fallbacks. RayFusion is still unchecked.

Limiter, measured in pixels: display luminance change per EV stop at equal
display brightness, fire pixels of the cap, 64 spp.

| Display L | Live material | M=3 tp3 |
| --- | --- | --- |
| .55-.65 | .031 | .098 |
| .65-.75 | .016 | .093 |
| .75-.85 | .007 | .071 |

The live material responds 6-10x weaker in its bright half. AgX is present in
both columns, so this gap is material-side and consistent with the 64 limiter
plus thick S = j/sigma. Pixel test, not a voxel histogram.

Heat curve on M=3, burnout .9 band 0.2..0.6, EV relative to the EV-5 base:

| Case | ms | Fire p50 / p95 | 15 px detail |
| --- | --- | --- | --- |
| tp .72 | 103 | .589 / .897 | .0705 |
| tp 2, +.94 EV | 102 | .599 / .904 | .0729 |
| tp 3, +1.68 EV | 100 | .611 / .909 | .0750 |
| tp 3, +2.68 EV | 105 | .628 / .915 | .0828 |
| tmax 30000, tp .72 | 92 | .530 / .828 | .0607 |
| tmax 30000, tp 2, +2 EV | 92 | .552 / .854 | .0659 |

temperature_power alone is a small lever. The gain comes with +1 EV staying
under the limiter. tmax 30000 loses yellow and 6-14% detail: rejected. At frame
69 peak T is 6.02 (18060 K), so the 20000 K clamp is inactive there; it only
matters on early frames (frame 21 peak 7.39).

Best so far: M=3, EV 1.9, temperature_power 3, burnout .9, 0.2..0.6, fire off,
max_steps 128. Not applied to the scene.

## Temperature response: hot range is compressed

The limiter table below is the model that the pixel test above confirmed.
Read-only. Frame 21, gas.measure_plume: peak 7.39, mean 3.39 in solver units,
fire_max_temperature 10. Publication multiplies by 3000 to Kelvin, so the field
reaches 22184 K against temperature_max 20000. No voxel histogram exists:
fluid.inspect_fog_fields refuses this domain on its memory budget. The table is
a model of the shader chain at evaluated density .7, not a voxel measurement.

| Solver T | Kelvin | u | Raw j luminance, live EV 4.22 | After limiter 64 |
| --- | --- | --- | --- | --- |
| 2 | 6000 | .26 | 29 | 29 |
| 3 | 9000 | .42 | 133 | 64 |
| 5 | 15000 | .73 | 407 | 64 |
| 7.4 | 20000 (clamped) | 1 | 712 | 64 |

At the live EV the limiter flattens everything above roughly solver T 2.6, a
5:1 raw range, leaving only hue. This corrects the earlier note that the limiter
is not the main cause: that came from cap means under one EV sweep and did not
isolate voxels. At EV-5 the same table peaks at 22, so the tuned material is
below the limiter.

Other compressors: temperature_power .722 is under 1 and flattens the hot end;
ramp red saturates at u .7; temperature_max clamps above solver 6.67; AgX then
compresses the display range. Cooling by frame 69 was not measured.

Next, no code needed: on the tuned material compare temperature_power .72, 2, 3
with EV compensation, and temperature_max 30000 matching fire_max 10. Missing
tool: a bounded temperature histogram over IPC, without it this stays a model.

## Handoff: independent emitters on the bounce radiance cache, written, not built

Follows the measurement below. Shader-only change; no build or live run yet.
Test: scripts/test/rt_test_pyro_bounce_cache_ipc.py, full march is the reference.

- volume_cached_segment.glsl no longer rejects independent emitters.
- For them the source channels hold sigma*inscatter + j, not J, because j/sigma
  has no limit at zero extinction. Legacy materials keep J; the march multiplies
  by sigma per medium, so mixed overlaps add correctly.
- Density channel holds optical density after burnout. A vertex with j > 0 and
  no optical density stores 1e-12 so vrcProvenEmpty cannot discard it.
- The segment weight is now volumeEmissionIntegral with flag 2, the same
  (1 - T) / sigma with a stable vacuum limit, also for legacy materials.
- The cache key hashes the whole VkVolumeInstance, so EV, ramp and burnout edits
  refill without new stamp terms.
- Bounce emission is cell resolution (about 3 voxels here, 2 cells per step) and
  isotropic. Expect softer fire in reflections and ground light, and possibly a
  brightness drift from thick cells, as the 16/4 test hinted (+3.4/255).
- Shared walk stays bypassed for independent emitters: unrelated optimization.

## Acceptance run: soot burnout works, camera cap saves less than predicted

User built and ran rt_test_pyro_soot_steps_ipc.py: PASS, contract all true,
material and LOD restored, legacy gate identical. RayFusion has not been checked.
Same-setting repeat gave 181 and 211 ms, so read GPU times as about +-15%.

Soot burnout, fire off, 32 spp, M=3, EV-5:

| Case | ms | Cap mean | p5..p95 contrast |
| --- | --- | --- | --- |
| burnout 0 | 161 | .4406 | .295 |
| .7, 0.2..0.6 | 170 | .4704 | .429 |
| .9, 0.2..0.6 | 181 | .4923 | .530 |
| .9, 0.05..0.3 | 225 | .5858 | .599 |

0.2..0.6 is the good band: hot cores open up, dark folds and smoke remain, and
contrast rises 45-80% for 6-13%. The low band washes the cap out and costs +40%.
At M=9.7 burnout also works but keeps the skin look. Best candidate so far:
M=3, EV-5, burnout .7-.9, 0.2..0.6, fire off. Not applied to the scene.

Camera cap with authored 128: 195 / 163 / 148 ms at no cap / 64 / 32. That is
-17% at 64, not the predicted half: the earlier 120 ms also capped bounce rays
through material max_steps, and this cap does not. Mean drift +2.1% at 64 and
+8.1% at 32, p99 +9.5% and +19%.

Remaining cost is bounce rays. Emitting independent-j materials return false
in volumeCachedSegment, so they bypass the radiance cache and march in full.
Same tuned material, camera cap 64:

| secondary_max_steps / scale | ms | Cap mean | Ground mean |
| --- | --- | --- | --- |
| 128 / 2 (default) | 181, 211 | .4923 | .3920 |
| 32 / 2 | 117 | .4923 | .3972 |
| 16 / 4 | 104 | .4923 | .4052 |

32 / 2 saves 35-45% with an unchanged cap and ground +1.3/255. The 2026-10-25
default of 128 came from +8/255 ground drift on another material, so re-check
it on a brighter fire before changing the default. Real fix: let independent
emitters use the radiance cache. Not done.

## Handoff: soot burnout and RT camera cap written and built

Both follow the measurement below. The acceptance run is
scripts/test/rt_test_pyro_soot_steps_ipc.py; results are in the section above.

- Camera cap: VolumeMarchLod.camera_max_steps, default 64, 0 = authored, bits
  22-30 of the world word. Vulkan RT ordinary march and fog mixture only;
  RayFusion keeps the authored budget. New materials default to max_steps 128.
  Saved scenes keep theirs, so the open scene runs min(121, 64) in RT.
  Process state, not saved. Panel, render.set_volume_march_lod, Python.
- Soot burnout: soot_burnout, _start, _end in VolumeEmissionResponse. Extinction
  times 1 - strength * smoothstep(start, end, u), where u is the ramp coordinate.
  Rides _accel_reserved[0..2] on non-isosurface volumes; pore fields keep them on
  source_type 4, so the ABI stays 688 bytes. Requires independent_emission.
  Written for ordinary RT march, mixture emission terms and RayFusion.
- Not covered: shadow rays, volume light cache, fire first scatter visibility,
  CPU and OptiX. Shadows reading temperature would add a tree lookup per step
  in the hottest loop; judge the image first.
- Defaults 0.2..0.6 are guesses from ramp colors. The scene's temperature
  distribution was never measured, hence the low band in the test grid.

## Live measurement: quality problem is optical depth, not the sampler

Same scene, frame 69, 1680x945, Vulkan Rendered, SVGF off, 32 spp. Material
was changed and restored by IPC (equality verified); nothing saved. Numbers are
tone-mapped luminance over the cap region and camera RT GPU ms per sample.
Tone mapping compresses differences; treat contribution ratios as approximate.

| Case (M=9.722 unless noted) | Cap mean | GPU ms |
| --- | --- | --- |
| Smoke only (no emission, no fire) | .3790 | 106 |
| Direct emission, fire off | .4223 | 194 |
| Fire on, 8 candidates | .4257 | 463 |
| Fire on, 32 candidates | .4260 | 615 |
| M=3, EV-5, fire off / on | .4443 / .4479 | 230 / 480 |

Fire scattering adds about 8% of direct emission's visible contribution for
+139% cost. 32 candidates change nothing (MAE .0027): this is not unconverged
sampling. The term is mostly noise: 4-to-32-spp residual is 1.5x in the cap,
3x in the ground ring and 7.5x in the stem compared with fire off. At M=3 the
gain is the same size, so a radiance cache of first scattering alone is not
justified by this scene. It would remove noise from a term with little energy.

Why: plateau extinction 28.8/unit means a mean free path of .035 units, one
quarter of a voxel. 73% of cap rays terminate by extinction. The camera sees a
skin of about one voxel, and its visibility to any emitter behind it is nil.
Emission and smoke share one density field, so hot gas is buried in its own
smoke. Lowering M alone blows out because j no longer sits behind extinction:
M and EV must move together. M=2..3 with EV-5..-7 and fire OFF gave a readable
volumetric fireball at about half the current cost, though the dark outer smoke
shell partly disappears, so the look is not final.

Thermal limiter: EV-2 and EV-4 still darken emission, so handoff item 2 is not
the main cause here. Its per-voxel activation remains unmeasured.

Step budget, max_steps. RT is chord/budget on a fixed lattice; step_size needs
about 450 here, so the budget is the real step count, not a ceiling.

| max_steps | Cap mean M=9.7 | ms | Cap mean M=3 | ms |
| --- | --- | --- | --- | --- |
| 32 | .5193 (+26%) | 92 | .4794 (+7%) | 87 |
| 64 | not measured fire off | | .4532 (+1%) | 120 |
| 121 | .4223 (+3%) | 194 | .4443 | 230 |
| 256 | .4106 | 263 | .4472 | 242 |
| 512 | .4104 | 319 | | |

32 looks equally sharp in RT because per-sample jitter accumulates, but it is
biased: a thick step samples deeper, hotter gas. Tuning at 32 and rendering
higher changes the look. RayFusion's jitter is hash31(gl_FragCoord, volume) in
material_preview_volume.frag, with no frame term, so it cannot average over time.
That reading is from source; RayFusion was not re-measured in this batch.

Proposals, none implemented: fire_scatter off by default for this look; extinction
reduced by temperature (soot burnout) so hot gas is not buried; optical-depth
step refinement so the budget becomes a true ceiling; separate RT camera cap.

## Latest user build: visual rejection and startup regression

User compiled the source-channel correction. Same scene, changed material:
fire produces excessive noise and does not meet the intended lighting behavior.
Do NOT label this feature accepted based on the earlier 103 source/math tests.
New read-only snapshot: rt_pyro_noise_startup_20261101.json.

The current SceneLog.txt records:

    RT pipeline cache is for other shaders; starting empty.
    RT pipeline cache saved (13629 KB).
    RT pipeline ready in 398,7 s
    (groups=9, threads=13, cache=unknown, volume=yes, hair=yes, sphere=yes, shadowMiss=yes)

This confirms about 6 min 39 s pipeline wall time on this cold shader build.
It does not isolate the fire sampler's exact share or prove warm-cache restart
time. Shader hash changes intentionally reject the previous pipeline cache.
Fire off in a material is only a runtime branch: its source/visibility tree
remains reachable in the fog closest-hit module and must still be compiled.
Moving source text to includes does not create driver compilation boundaries.
Do not retry the previously rejected all-functions DontInline/Modular mode.

New material: M=9.722, scatter=2.08, absorption=.88, anisotropy=.99,
fire candidates=8, max steps=121, shadow steps=1, stride=5,
independent emission=true, random walk=false, multi_scatter=.819.
Curve=[0,.40728,.73764,.83585,1,1,1,1]; mask center=1.765,width=.955.
Density power=1.417, EV=4.222, blackbody intensity=64.583.
The mask now leaves pre-curve density 0..1 unmasked (transition starts ~1.2875),
so the previous .0886 source-rejection diagnosis is not the explanation for
this new material. Approximate plateau peak sigma is 28.78 per unit under a
unit transform; this is illustrative, not a measured voxel/chord distribution.

Current estimator has structural variance risks: small uniform volume proposals
for a sparse emitter, inverse-square near-field weights, sharply directed HG at
g=.99 even after partial isotropization, actual expensive visibility only after
RIS selection, and early exhaustion of two four-candidate local batches followed
by retained-source reuse deeper in the chord. More candidates are not an accepted
quality/performance solution. First scattering still cannot supply diffusion.

Recommended next architecture: retain independent j/source-density correction;
replace the experimental per-camera source/visibility estimator with a bounded,
separately compiled compute lighting pass and a small spatial incident-radiance
cache consumed by the camera march. Specify first-scatter versus higher-order
energy explicitly, optical-depth-aware spatial resolution, RGB extinction and
solid occlusion, frame/material/topology invalidation, bounded memory/work, and
a fallback for thin hot features before implementation. Direct j remains in the
camera integral. Reuse canonical emission logic, not temperature as energy.
Validate isolated emission/first-scatter/visibility AOVs before visual acceptance.
This is a proposed replacement, not code delivered or measured acceleration.

No scene setting was changed, no app launched, no build run, no new benchmark
matrix performed during this follow-up. The current fire implementation remains
available; it was not silently disabled in the user's scene or in source.

## Live inspection before this correction

User compiled the preceding independent-emission/receiver-aware sampler change.
External named-pipe inspection found Vulkan Rendered, frame 69, one published
Gas NanoVDB (vdb:1 / gas:Nuclear Gas), density and temperature available.
Both authoring targets returned identical material settings. Reviewed a completed
1680x945 / 128 spp viewport capture. Temporary screenshot was deleted afterwards.
Full settings/context and restoration checks: rt_pyro_handoff_live_20261101.json.

Observed: dark, closed mushroom lobes, narrow pink emission strips, bright pink/
white stem and ground ring. User reports small localized fire-light contribution
and high cost; reducing smoke density loses depth. This inspection did not repeat
timing benchmarks or isolate the fire term with an off/on AOV. Therefore narrow
visible strips cannot be attributed specifically to first scattering.

Relevant settings:

| Setting | Value |
| --- | --- |
| Independent emission / response / fire | all enabled |
| Fire candidates | 8 total per camera segment |
| Random walk / legacy internal steps | off / 0 |
| Smoke multiplier | 2.083 |
| Scatter / absorption coefficient | 4.53 / 0.29 |
| Density curve | enabled, [0,1,1,1,1,1,1,1] |
| Density remap / cutoff | 0..1 / 0.007 |
| Emission mask center / width / strength | 0.52 / 0.20 / 1 |
| Emission density power / temperature power | 0 / 0.889 |
| Emission EV / blackbody intensity | 14.445 / 64.583 |
| Temperature range / scale | 1190..20000 / 0.994 |
| Max steps / voxel step | 121 / 0.431 |
| Shadow steps / stride | 1 / 5 |
| Multi scatter | 0.819 |

### Confirmed code interaction

Before this correction independent j removed the smoke multiplier, but retained
the smoke density curve in BOTH emission response and emission-mask input.
The live curve evaluates 7*x over x in [0,1/7], then 1 over [1/7,1].
The inverse mask transitions over evaluated density 0.42..0.62, thus raw remapped
density roughly 0.060..0.0886. Above about 0.0886 emission is zero, including
the entire plateau. Values above 1 follow the curve's linear tail and remain
masked. This is a deterministic source-definition issue, before light transport;
we did not measure the voxel histogram or the fraction of hot voxels rejected.

The plateau also produces scalar peak extinction about 10.04 per world unit
under the shader's density/extinction convention (unit transform assumed).
A hypothetical uniform 1-unit chord then has T about 4.36e-5 and mean free path
about 0.10 units. These are local illustrative estimates, not measured scene
optical depths. Scatter color and colored absorption affect actual RGB energy.

## Source correction now implemented, awaiting user build

With independent_emission enabled, emission and its mask use density BEFORE the
smoke curve and multiplier. Smoke extinction continues using the authored curve.
Remap, material density noise and cutoff remain shared. Cutoff/fade are evaluated
on each channel's own pre-multiplier density. With the flag off, both channels
are exactly the established post-curve density, preserving legacy semantics.

The normalized source density is not reconstructed from post-curve density:
a plateau has no inverse. Instead, one field lookup yields paired optical and
emission densities. No second NanoVDB interpolation/tree lookup or new grid is
introduced. The shared GLSL mapping is volume_density_channels.glsl.

Wiring covers ordinary Vulkan march, prepared NanoVDB path, general mixture,
fire-source RIS and material preview. Shadow-only consumers retain the optical
channel. Graph-written density remains authoritative; arbitrary graphs are
outside this independence guarantee, and fire sampling still rejects gas graphs.
The existing CPU/CUDA camera paths already evaluate pre-curve density for this
response; this correction does not claim full optical parity between backends.

Sparse/dense skip proofs, mixture emission guard and host FogOpticalBounds retain
either channel's support. A zero smoke curve may still contain an independent
emitter. Empty-sample gates check both channels. Source and camera use the same
emission/mask evaluator. GPU record ABI remains 688 bytes / 172 words; no new
material setting, JSON schema, IPC operation or scripting operation was added.
Existing independent_emission UI/API descriptions now document pre-curve input.

Completed: 103 non-build math/source-contract tests. Added live-curve regression,
legacy mapping, zero-curve emitter support and single-field-fetch contracts.
Extended user-run fog_optical_bounds_test.cpp for a zero-curve emitter.
No C++/CUDA build, shader compilation or post-change GPU acceptance was run.

The open scene's material, camera, frame and post settings were unchanged.
Only accumulation was restarted to obtain a completed image; capture was restored
to false. Scene was not saved, simulation was not reset, no application launched.

## Remaining work for the next agent

1. User must build C++ plus the full shader package. Compare the same saved
   settings first. This change restores a broader emitter and can brighten the
   image substantially; old intensity was tuned against a masked/clipped source.
   Freeze optics, retune emission once, then evaluate dark lobes/interior gaps.
   Test independent off for legacy appearance; test zero smoke curve and zero
   multiplier for source support. Do not automatically set shared steps to 32:
   user found Vulkan RT ~32 useful, RayFusion needs >=128.
2. Thermal emission still has the legacy luminance-64 limiter AFTER response
   and mask. Live EV/intensity alone multiply to about 1.44 million before
   temperature/color/mask. Surviving points can saturate this limiter, erasing
   source-power contrast and making exposure changes ineffective. Saturated
   voxel fraction is unmeasured. Diagnose a raw j/mask/clamp AOV before changing
   the limiter; do not compensate by blindly raising emission or candidate count.
3. Fire is first scattering, not multi-bounce diffusion. Its visibility uses
   physical extinction; the live high scatter can block direct core-to-shell
   paths even with low absorption. multi_scatter currently changes phase, not
   propagation through repeated events. This correction cannot by itself make
   optically thick smoke transport light by multiple scattering.
4. Receiver-aware RIS still spends candidates in batches <=4. At budget 8 it
   can exhaust its two batches near the first camera steps, retaining the last
   source deeper in the volume. Audit receiver-distance/PDF/reuse quality and
   budget distribution over the chord. It has not become a persistent cache.
5. Actual fire visibility evaluates at least 8 density samples per medium/chord
   even when shadow_steps=1 (clamp 8..64), plus solid/SDF checks, refreshed at
   stride. Independent emitters bypass old S radiance cache/shared walk. These
   are concrete cost mechanisms, not a new timing attribution. Existing fresh
   performance records remain canonical; do not repeat a broad benchmark matrix.
6. Next useful diagnostic is separate direct-j, fire-first-scatter, visibility,
   mask and limiter activation views. Then choose a bounded spatial radiance/
   transport cache or a measured multiscatter approach if first scattering is
   insufficient. Do not add a temperature-energy proxy: temperature already
   drives the authored ramp/blackbody emission coefficient.

Sources to start with: volume_density_channels.glsl, volume_hit_sampling.glsl,
volume_nano_optical.glsl, volume_internal_rt.glsl, volume_fire_source.glsl,
volume_fire_scatter.glsl, volume_mixture_material_sample.glsl,
volume_mixture_optical_terms.glsl, include/VolumeEmissionTransport.h,
include/Fluid/FogOpticalBounds.h. Previous implementation details remain in
PYRO_INDEPENDENT_EMISSION_20261101.md and historical live/performance notes.
