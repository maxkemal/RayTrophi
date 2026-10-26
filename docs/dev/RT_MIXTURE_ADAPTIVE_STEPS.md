# Adaptive RT mixture steps (2026-10-26)

## Finding

The material already persisted and exposed `quality.adaptive_stepping`, but RT
received only the voxel-relative `primaryStep()`. The switch itself was not
transported to the RT volume record. The general mixture march and random-walk
flight partitioned the whole chord uniformly. Empty-space skipping was a
separate optimization, not density-adaptive quadrature.

No persistent density-triggered fixed/adaptive latch was found in this path.
Raw NanoVDB corner caches are invocation-local and contain raw values, not
material-scaled density. The volume light-cache signature includes slot bytes
and content versions, and material edits republish/reset accumulation. The
reported low/high/low behavior still needs a user build and live check; a
sticky-state runtime defect has not been confirmed.

## Change

`volume_mixture_adaptive.glsl` supplies one stateless policy to both the general
camera march and the random-walk flight. All intersecting contributors, up to
the existing sixteen-slot capacity, constrain each proposed interval:

- Read all eight raw corners of each current trilinear interpolation cell.
  Bound its directional derivative by the largest edge difference on each
  axis, multiplied by the transformed ray direction.
- Apply the current monotone density remap/curve, cutoff-fade derivative bound,
  density multiplier, scattering and absorption. Add contributor bounds;
  opposing gradients do not cancel. The random-walk similarity reduction uses
  the same current event ordinal as the optical evaluator.
- Choose a span using the local midpoint optical-depth error bound
  `L * span^2 / 4`, targeting 0.01, between one quarter and four times the base
  interval. Stop at the nearest interpolation-cell face and volume entry/exit.
  The quarter-base refinement floor limits work: the tolerance is a local
  target, not a strict whole-ray error guarantee.
- Unknown/non-NanoVDB producers, procedural density noise, spatial material
  programs, or a fixed-policy member retain fixed stepping. Nonzero emissive
  fields cannot coarsen based only on density: temperature is independent.
  Proven density-empty cells in an emissive field can still coarsen safely.
- Preserve exponential colored extinction, per-material scattering/emission,
  optical flight targets and contributor selection. The interval used for
  optical integration is also the geometric advance. The flight never redraws
  its exponential target when a segment changes length.
- Existing camera empty-space skipping remains; authored shadow quadrature is
  unchanged. SDF/opaque handoff and ordinary single-volume paths are unchanged.
  Two matching adaptive media select the general path rather than silently
  using the fixed companion path.

The hard adaptive work limit is 8192 intervals. On the final interval any
remaining chord is integrated and the existing `VOLUME_MARCH_STEP_BUDGET`
counter reports the quality fallback. It must not strand a ray short of its
boundary or silently claim that the adaptive error target was met.

## Authoring and packet contract

The existing switch is packed into unused `density_curve[9]`; curve data uses
only `[0..8]`. The shared record stays 672 bytes, with no offset/stride changes.
All three canonical volume packet producers repack the switch on publication;
both render and viewport Vulkan packets already copy the complete twelve-float
block. Other backends ignore the new policy slot and retain their existing
behavior. No optical decision survives a material edit or another ray.

UI: **Ray Marching Quality → Adaptive Volume Steps**. The control calls the
same material configure service as existing Python/IPC operations:

```python
rt.volume.set_material(target, adaptive_stepping=True)
rt.volume.get_material(target)
```

IPC: `volume.set_material` with
`{"target": "<canonical target>", "settings": {"adaptive_stepping": true}}`.
Existing boolean validation and atomic overlays apply. Visibility, physical
simulation state and source identity do not change when this switch changes.

## Verification boundary

`scripts/test/check_volume_mixture_adaptive_contracts.py` exercises random
trilinear derivative/error bounds, cutoff fade bounds, constant colored-medium
integration, optical-target continuity, forward/negative/transformed cell exits,
third-medium bounds, low/high/low reversibility, packet/UI/API wiring, fallback
guards and both integrator call sites. These are numerical/source checks, not
compiled GPU execution or a speedup measurement.

No C++ build, shader compilation, app launch or long live benchmark was run.
Executed: 10 adaptive contracts, 22 optical contracts, 10 material contracts,
10 source-provenance contracts, the existing march-LOD audit, and diff whitespace
checks passed. No runtime performance or image-equivalence claim is made.
User checklist after rebuilding C++ and `volume_closesthit.rchit`:

1. Keep the prepared scene/camera/frame. With random walk off, enable adaptive
   steps on the three independent materials. Check water Fog, vapor and gas
   emission remain present; compare against adaptive off at the same base
   resolution, especially thin source edges and refractive SDF/opaque overlap.
2. On the same material, change density **low → high → original low**. Confirm
   the image and short-sample work counters return near the first state after
   accumulation restarts. Do not compare progressively accumulated images from
   different settings.
3. Repeat with random walk enabled. Compare noise/convergence as well as dispatch
   time: fewer milliseconds per sample does not prove equal image error.
4. Briefly check two matching media, a third fine source, different transforms,
   adaptive off, density curves, emissive temperature detail, and noise/program
   fallbacks. Render and viewport packets must agree on the material switch.
