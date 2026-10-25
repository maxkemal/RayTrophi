# Zero-optics gas cost, 2026-10-25

Live external named-pipe investigation of `Nuclear Gas`, frame 129, 1680x945.
The user held Plane_1 visible and the camera fixed for the accepted captures.
Density multiplier, scattering and absorption were zero; random walk was off.
The simulation reported 3,734,332 active density cells: optically disabled gas,
not an empty physical field. No build or application launch was performed.

## Accepted GPU measurements

Each row contains exactly camera RT dispatch sample indices 0..63; volume atomic
counters were off. These are whole-dispatch GPU timestamps, not CPU FPS or
isolated stage times. Adaptive sampling was retained. Bounce changes alter the
rendering calculation and are diagnostic ablations, not equivalent quality modes.

| Case | Mean GPU ms |
| --- | ---: |
| Original: shadow steps 5, march budget 192, total bounce budget 10 | 227.43 |
| Shadow steps 0, otherwise original | 216.91 |
| Shadow steps 0, march budget 1 | 55.92 |
| Original repeated | 220.30 |
| Shadow steps 0, total bounce budget 1 | 25.10 |

The shadow0/steps1 difference is 3.88x; shadow0/one-bounce is 8.64x.
Original/repeat differ by about 3.2%, so the shadow-steps change is small compared
with the other ablations and should not be interpreted as isolated shadow work.

Raw accepted timings: `rt_gas_zero_optics_plane_visible_20261025.json` and
`rt_gas_zero_optics_bounces_20261025.json`.

## Separate counters

Full-image, short natural-render counter windows, normalized by traced paths:

| Counter | One physical bounce | Original bounce budgets |
| --- | ---: | ---: |
| Density samples / path | 28.434 | 35.130 |
| Volume rays / path | 0.2842 | 0.3578 |
| Solid probes / path | 0.2842 | 0.3578 |

Raw counters: `rt_gas_zero_optics_full_counters_20261025.json`. Counter windows
are separate from GPU captures; their instrumentation overhead is not timed.
The first counter pass sampled only the center (which misses the plane), so its
matching one-bounce/original values are not representative of the entire image.

## Conclusions and limits

- Zero optical settings do not prevent an active gas AABB from reaching RT
  closest-hit. `volume_intersection.rint` rejects inactive records only.
- `volumeNanoDensity` reads NanoVDB corners and remaps density before multiplying
  by density_multiplier. A zero multiplier therefore still incurs memory work.
- The normal march samples density before its zero-extinction rejection.
  Its solid probe runs before the march, even for optically disabled gas.
- RayFusion's camera volume pass clips to the scene depth and integrates camera
  intervals in screen space. The RT path can revisit gas on secondary surface
  rays. This is a real architectural cost difference, but RayFusion GPU time was
  not measured here.
- The 8.64x whole-dispatch bounce difference is much larger than the 24% change
  in density samples/path. It cannot be attributed solely to gas sample counts;
  plane surface shading, secondary-ray execution and GPU scheduling/occupancy
  remain part of that difference. No shader-stage timings are claimed.
- A conservative optically inert gas rejection before traversal/sampling is the
  first optimization candidate. It must preserve material programs that create
  density/extinction/emission, independent emission, overlapping media and SDF
  surfaces. A blanket coefficient-zero rejection would be unsafe.
- A valid gas-hidden or gas-removed baseline was not obtained. The attempted
  `gas.set_param(visible=False)` does not hide grid-only gas in the inspected
  implementation. That raw row is explicitly invalidated and excluded.

The earlier exploratory report `rt_gas_zero_optics_20261025.json` is invalid:
the plane changed during capture and pending work crossed the reset boundary.

All gas shader and render bounce settings were restored and read back equal to
their originals; GPU timing and volume counters were left disabled. Frame 129
and camera were preserved. No performance implementation was changed.
