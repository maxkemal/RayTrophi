# Two-gas mixture: restore persistent scene-light reuse

Follow-up cost breakdown: [RT_GAS_OVERLAP_COST_20261030.md](RT_GAS_OVERLAP_COST_20261030.md).
Its ablation found the largest uncounted term, the per-refresh solid trace that
this note listed as unchanged.

## Problem targeted

The priority is the large overlap penalty: the original controlled gas matrix
measured about 123 ms for A+B against a baseline-corrected additive reference
of 26.847 ms. Its camera density work grew 1.49x but shadow density work grew
27.86x relative to the sum of the single-field runs; persistent light-cache
hits nearly disappeared. Later split/legacy timings around 140 ms did not
resolve that problem and should not redirect the work toward small mode deltas.

Source inspection confirmed a concrete policy discontinuity. Ordinary fog calls
`volumeLightCacheFetch` for fixed scene lights. Mixture `mfDirect` instead always
called `mfShadow`, marching all contributing fields again at shadow refresh.
This is a missing consumer, not evidence that the cache rejected every request.
The live scene's Light_0 is a point light, supported by the existing cache.
Previous shadow-zero ablations still left substantial cost, so the change below
does not claim shadows alone explain or fix the entire overlap penalty.

## Implementation

`volume_mixture_light_cache.glsl` adds a guarded consumer of the existing
per-volume density-integral cache for mixture scene-light visibility. It is
included after the canonical `mfShadow` fallback. Eligibility is prepared once
per mixture invocation, after random-walk mode and the two NanoVDB accessors
have been selected. No new buffer, descriptor, authoring switch or cache format
is introduced.

- Each prepared contributor uses its own volume slot, density remap/cutoff,
  material and cell integrals. No raw density merge or temperature averaging.
- The query is clipped to each medium's ray chord; an outside start queries the
  entry point. Fixed point/spot/directional light identity comes from the existing
  channel mapper. Unsupported lights and clipped finite directional chords fall
  back; no arbitrary ray direction is guessed to be a cache channel.
- Unit-density extinction comes from the canonical `mfExtinctionTerms` core.
  Cached integral times that RGB extinction yields each medium's transmittance;
  its authored 0.92*shadow_strength blend is applied before multiplying media.
  The scalar/luminance `volumeShadowFromIntegral` helper is deliberately not used
  for colored mixtures.
- A cache miss on any contributor rejects the partial aggregate and runs the
  original complete `mfShadow`. Existing atomic claims and four-fill invocation
  budget bound cold-cache work. Zero-strength/disabled contributors are neutral.
- Ambient visibility retains its original direction-dependent march. Random walk,
  material graphs, density noise and additional unprepared contributors retain
  the original path. Geometry probes, light choice, RNG sequence, shadow stride,
  camera integration, adaptive error/budgets and emission behavior are unchanged.

The common lazy fill now samples at least the authored shadow count, clamped to
the same supported 64-step ceiling. Its range is 4..64 rather than 4..32. The
four-cell fill cap is unchanged (worst-case 256 density samples per invocation).
The host key already hashes the full volume record including shadow_steps.
No material quality setting is reduced. This can increase cold-fill work.

The cache is the existing spatially interpolated cell-integral approximation,
not a bit-exact replacement for the old per-ray midpoint quadrature. Higher
fill count alone is not a proof of image parity; image acceptance remains
required. With matching integrals, constant RGB extinction/strength algebra is
preserved, apart from the old march's exp(-16) early-termination tail.

## Verification and status

96 non-build tests passed, including five new cache contracts: 1,500 independent
nonlinear two-field integral cases with RGB/strength variation, colored/black
absorption, zero density, all authored counts 1..64, eligibility/fallback wiring,
solid-probe ordering and ambient preservation. Existing mixture, adaptive,
material, SDF, stage routing and measurement tests also passed. Light-cache and
march-LOD source audits passed. Python math/source checks do not execute GLSL.

### Live acceptance measurement, 2026-10-30

The user compiled and opened Studio. External IPC confirmed the legacy pipeline
and matching source/deployed shader SHA-256
`DB9BAB80749CEC42CF215D7F8290F307D30728B231F0525A0F9F28D2174169FB`.
The following warm measurements use the same paused frame 99, camera, physical
fields, materials and 1680x945 output. Each entry contains two independently
armed GPU captures at sample indices 8..23; counters were collected separately.

| Visible gas | GPU medians, ms |
| --- | --- |
| Hidden | 5.02 / 5.82 |
| A | 23.38 / 24.37 |
| B | 16.03 / 16.15 |
| A+B | 112.56 / 111.39 |
| A+B, repeated after the visibility matrix | 114.78 / 98.54 |

The baseline-corrected additive reference is approximately 34.55 ms; the first
A+B pair averages 111.98 ms, about 3.24 times that reference. The substantial
overlap penalty remains. The repeated A+B pair also shows timing variation;
do not turn these captures into a precise before/after percentage.

The first A+B counter capture reports, per traced path, 1.675 light-cache hits,
6.875 shadow density samples, 13.814 camera density samples and 5.946 majorant
queries. Warm misses, fills and fill samples were zero. The preceding legacy
capture had about 0.00154 cache hits and 19.666 shadow density samples per path.
This confirms the missing cache consumer now executes and reduces shadow work.
Startup-to-startup fields differ, so the historical approximately 140 ms GPU
band is context rather than a controlled speedup estimate. Cold-fill cost and
image parity remain unmeasured. Remaining camera/adaptive integration and
residual shadow work need investigation; shader stage splitting alone is not
the demonstrated solution to this overlap penalty.

Raw measurements are in ignored
`docs/dev/.captures/rt-gas-mixture-cache-matrix-20261030.json` and
`rt-gas-mixture-cache-both-20261030.json`. The matrix verified unchanged physical
scene context and restored both original gas visibility settings. A subsequent
independent IPC read confirmed frame 99 paused, GPU timing disabled and counters
disabled with the original full-frame region.

This is an internal rendering implementation using existing canonical material
operations. UI, Python and IPC controls remain unchanged; no separate UI-only
business logic or new authoring API is introduced. No C++ change is needed for
this optimization. The agent has not compiled, deployed or launched the app.
The installed shader was built by the user.

## User acceptance: focus on the large two-gas cost

With Studio closed, build the currently selected legacy shader only:

```powershell
.\RayTrophiStudio\compile_shaders.bat -VolumeOnly -VolumeOptimization Off
```

Remain in legacy mode for this comparison; the split architecture is not the
performance variable. If testing split later, rebuild its bundle separately
with `-SplitVolumeOnly -VolumeOptimization Off` so the fog variant includes this
change. Restart before testing so cached entries are filled by the new sampler.

Check two gas fields are actually visible/published. Measure hidden/A/B/A+B/A+B
in the same paused session with stable camera, material settings and physical
data, counters off for GPU timings, fixed sample indices, and counters collected
separately. Track total A+B GPU ms, fixed-light cache hits/misses/fill samples
and remaining shadow density work; cache fill work must not be called free.
Warm speed and cold cost should be recorded separately. Small startup-to-startup
voxel differences should not distract from assessing the large overlap penalty.

Image checks: independent RGB absorption and strengths, plain/thermal emission,
lights inside/outside the boxes, point/spot/directional lights, cold cache/misses,
graphs/noise, area lights, random walk and three contributors. Confirm disabled
shadow contributors remain neutral and that the fallback is never unshadowed.
The earlier camera/adaptive overhead remains a separate target if the large
two-gas cost persists after restoring this missing reuse path.
