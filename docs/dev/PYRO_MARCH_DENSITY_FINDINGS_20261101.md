# Raymarch budget and density/emission coupling

2026-11-01. User reports no visible difference between raymarch quality 30 and
256 in the open explosion, with a large Vulkan cost difference. This is the
user's observation, not a new agent benchmark. Previous same-setting backend
timings remain observations, but do not establish the irreducible backend gap
at equivalent required visual quality. Do not repeat the broad benchmark sweep.

Read-only external IPC: frame 64, paused, gas:Nuclear Gas and vdb:52 have
identical complete material settings. Current binary does not expose
fire_scatter_enabled yet; the new first-scatter implementation cannot explain
the currently observed density-brightness coupling.

Relevant current settings:

- max_steps=30, adaptive_stepping=true, voxel_step_multiplier=1.643.
- density_multiplier=54.861, density_power=3.778, response_enabled=true.
- emission_exposure_ev=9.778, blackbody_intensity=15.972.
- mask enabled, center=.2, width=.1, strength=1; density curve disabled.
- scatter=.4, absorption=4.52, random_walk=false, internal_light_steps=0.

Source audit:

1. The ordinary gas march sets N=min(ceil(chord/stepSize),budget), then
   baseStep=chord/N. It is a fixed lattice with empty-space skipping; the
   adaptive switch does not give this branch the mixture's variable-step logic.
2. The mixture camera path derives its baseStep from the capped N too.
   Its adaptive span is initially limited to 4*baseStep; where density is
   present and emission_mode!=0, it cannot exceed baseStep. This protects
   independent temperature structure because the bounds only cover density.
   Thus a large budget can mean finer actual work rather than a rarely reached
   emergency ceiling. It does not force exactly 256 samples on every ray.
3. Existing response computes S proportional to evaluatedDensity^density_power.
   Evaluated density already contains density_multiplier. At fixed fields,
   temperature and response, multiplier M therefore affects S as M^3.778
   before the existing source limit. Legacy response-off S is density-linear.
   Camera opacity (1-exp(-sigma_t*dt)) also rises with density.
4. The mask divides evaluated density by the multiplier; it does not cancel
   density in the independent emission response. The source limiter can hide
   the magnitude of this dependence in bright cells.
5. The new, unbuilt first-scatter code correctly converts the current source
   function S to emission coefficient j=sigma_t*S. With the present material
   convention j would scale as M^4.778 before limiting, and as M after S
   saturates. This is consistent with the current camera equation, but it is
   not independent artistic emission authoring.

Recommended follow-up: separate a genuine maximum-work ceiling from step/error
quality and bound the emitting field before claiming adaptive coarsening there.
Do not blindly make 30 the universal default or remove the emission guard.
Separately define emission as an independent coefficient j if changing smoke
opacity must not change emitted power; then camera and transport must both use
that same convention. Merely dropping sigma_t in the new scatter estimator
would disagree with the camera and is not a fix.

No scene parameters, builds, simulation state or shader behavior were changed
during this inspection.
