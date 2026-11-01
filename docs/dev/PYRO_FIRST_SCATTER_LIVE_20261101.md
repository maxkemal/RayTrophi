# First-scatter live feedback and functional check

> **Durum:** ARŞİV — fire lights smoke 2026-11-01 ölçümle söküldü

2026-11-01. User compiled the new path. Reports that Vulkan RT needs little
visible improvement beyond max_steps~32 on this explosion, while RayFusion
gets noisy below 128. These are the user's quality/cost observations; this
turn did not repeat the performance benchmark.

External IPC confirmed Vulkan Rendered, one active NanoVDB gas slot with density
and temperature, random_walk=false, fire_scatter_enabled=true and 32 candidates.
Current max_steps=30, voxel_step_multiplier=.189, density_multiplier=29.861,
density_power=1.861, emission EV=3.778, temperature_power=2.777,
scatter=.13, absorption=.18, mask center=.53/width=.185. This is a different
material configuration from the preceding audit; cross-build appearance
changes must not be credited to the new scattering path alone.

## Three controlled beauty captures

Same camera/frame/post and original density/emission configuration; 128 global
iterations, adaptive sampling state unchanged, not equal-error/speed comparison.
Cases: both paths off; new on/32 candidates/legacy off; new off/legacy 16.
Original material, camera/frame/post, capture enable and counter settings were
restored. No project save, sim reset, app launch or build.

Raw record: `rt_pyro_fire_function_20261101.json`. Temporary images/scripts
were kept in ignored .captures and deleted after inspection.

| Case | JPEG whole-image RGB MAE vs off |
| --- | ---: |
| New first-scatter | 1.2992 / 255 |
| Legacy 16 steps | 1.0476 / 255 |

Both still show the dominant broad bright lobes rather than a dark soot shell
with interior fissure light. The new image has some differences around the
stem/fringe. These beauty differences are not a pure fire-light AOV: changing
the source sampler also changes RNG state and the camera jitter/noise pattern.
They do not quantify the actual lighting contribution or prove zero contribution.

Counters are cumulative across cases. Subtracting preceding snapshots:

- New: 802433505 camera density attempts / 15850594 camera volume rays,
  approximately 50.625 attempts/ray.
- Legacy case: 298310385 / 16208953, approximately 18.404 attempts/ray.
- Baseline: 288458139 / 15516265, approximately 18.590 attempts/ray
  (initial aggregate rather than an exact matched source-free workload).

The approximately 32 additional reads are consistent with the new source
candidate loop executing; it is not simply a missing UI control. Added shadow
samples also occur. This is a functionality check, not a GPU timing result.
The counters do not report selected source radiance or resulting fire in-scatter.

## Main optical problem remains

The existing direct emission source function S still uses multiplied shading
density. With density_power=1.861, doubling the smoke multiplier raises S by
approximately 3.63x before the source limit, in addition to raising opacity.
The new light sampler correctly converts that current S to j=sigma_t*S, so
its unbounded source coefficient scales approximately as multiplier^2.861
(7.26x for doubling). It does not make the existing source independent.

This explains why adding source transport cannot remove the existing bright
surface: direct emission is unchanged. The first implementation must not be
treated as acceptance of the desired explosion look.

The source proposal is also global and power-only. In a dense domain, most
sampled fire points may be optically far from visible smoke. That is a plausible
variance/efficiency limitation from the code, not a measured 3D radiance diagnosis.
For illustration only: raw density .5 at the current multiplier and scalar
scatter+absorption .31 gives a neutral-channel mean free path of about .216 m,
while the published box is about 22 m wide. Actual density and RGB transmittance
along source chords were not measured by these captures.

## Next implementation boundary

1. Define an opt-in independent emission coefficient j from source fields before
   the smoke multiplier. Camera, preview, CPU/CUDA and source lighting must use
   the same meaning. Camera emission is then j*(1-exp(-sigma_t*dt))/sigma_t,
   with stable j*dt in the zero-extinction limit. Do not just remove sigma_t
   from the current NEE estimator or globally change saved-scene semantics.
2. Make source proposals receiver-aware and visibility-aware; retain the full
   proposal PDF/support. Increasing global candidates alone does not solve
   optically local transport. Keep later-scattering caching separate.
3. Treat RT~32 and RayFusion>=128 as distinct starting-quality policies for
   this scene. Do not change the shared material default to 32 and degrade
   RayFusion, or claim 32 sufficient for every gas field.

This inspection changed no runtime source behavior or quality defaults.
