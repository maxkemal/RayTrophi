# Vulkan Volume Temporal Stability and Instrumentation

> **Durum:** REFERANS — Temporal stabilite ve enstrumantasyon durumu.

## Implementation status

- Temporal color/metadata ping-pong images and fixed asynchronous-safe
  descriptor slots `25..28`: implemented.
- Volume-only history classification, world-position/transmittance rejection,
  neighborhood reprojection and variance-aware blending: implemented.
- Instrumentation SSBO at binding `29`: implemented.
- GPU counters cover volume rays, density samples, skipped empty segments,
  shadow samples, early terminations and temporal acceptance/rejection.
- Host read/reset API synchronizes only when measurements are requested; the
  normal render loop does not add a CPU wait.
- Render Settings exposes the detailed snapshot under `Performance >
  Volumetrics`. `Debug Visualizer` contains only the optional viewport metrics
  overlay. The overlay reads the cached snapshot and never performs GPU
  readback by itself.
- `Copy Metrics Report` places a versioned, paste-ready text report containing
  both raw counters and derived ratios on the clipboard.
- GPU counters default to disabled. They are opt-in for profiling so normal
  frame-time/FPS measurements do not include instrumentation atomics.
- Metrics report `v2` splits march outcomes into extinction termination, step
  budget exhaustion and normal interval completion.
- Metrics report `v3` splits empty-space traversal into topology skips and
  density-leaf skips while retaining the combined skip counter.
- The experimental active-leaf maximum-density test was removed after v3
  measurements showed only `0.046%` of skips came from it.
- The active-leaf hierarchy-span cache experiment was reverted after a
  counter-disabled test regressed to roughly `54 ms/frame`. The retained path
  skips inactive topology tiles and avoids `is_active` for `dim <= 1` leaves.
- Primary and shadow volume marches scale shadow samples from an optical-depth
  hint while retaining minimum coverage in sparse regions.

## Existing baseline

- Bindings `0..24` are occupied by the Vulkan RT pipeline.
- Push constants reserve 256 bytes.
- `raygen.rgen` already writes beauty, variance, first-hit position/material and
  path-stat AOVs.
- NanoVDB primary and shadow marches already skip inactive hierarchy tiles.
- Atmosphere/fog/weather composition happens once, after path integration, in
  `raygen.rgen`.
- VDB and closed-mesh volumes publish measured primary transmittance.

## Descriptor ABI

New bindings are append-only so existing shader records retain their layout:

| Binding | Type | Owner | Contents |
|---:|---|---|---|
| 25 | `rgba16f storage image` | raygen | Previous resolved linear radiance; alpha is history length |
| 26 | `rgba32f storage image` | raygen | Previous world position; alpha packs volume/material classification |
| 27 | `rgba16f storage image` | raygen | Current resolved history target |
| 28 | `rgba32f storage image` | raygen | Current world position, transmittance and classification |
| 29 | storage buffer | volume closest-hit/raygen | Atomic traversal and temporal counters |

History images are ping-ponged after a completed camera trace. They must never be
read and written through aliases during the same dispatch.

## Previous-camera contract

The remaining push-constant space carries:

- previous camera origin;
- previous right, up and forward vectors;
- previous vertical tangent and aspect;
- history-valid flag;
- temporal profile and blend ceiling.

A current world position projects into the previous camera with:

1. `relative = worldPosition - previousOrigin`;
2. previous view depth from `dot(relative, previousForward)`;
3. NDC from right/up projections divided by view depth and field-of-view scale;
4. NDC to previous pixel coordinates.

History is invalid when view depth is non-positive or projected UV leaves the
image.

## Volume-aware rejection

Previous history is accepted only when all applicable checks pass:

- projected coordinate lies on-screen;
- previous/current classification agrees (surface, VDB/gas volume, mesh volume,
  or background);
- material/volume identity agrees;
- relative depth difference is within the profile threshold;
- world-position disagreement is within a voxel-scaled threshold;
- transmittance disagreement is within the profile threshold;
- neither sample contains NaN/Inf;
- the current pixel is not a disocclusion.

An empty volume interval remains background and therefore cannot create a
history boundary or ghost AABB.

Profiles:

| Profile | Max history | Depth tolerance | Transmittance tolerance |
|---|---:|---:|---:|
| Interactive | 8 | 4 voxels | 0.20 |
| Preview | 16 | 2 voxels | 0.12 |
| Final | 32 | 1 voxel | 0.07 |
| Cinema | 48 | 0.5 voxel | 0.04 |

History weight is additionally reduced by luminance disagreement and local
variance. Rejected pixels use the current sample without blending.

## Counters

The stats buffer uses 64-bit counters where supported and otherwise saturating
32-bit counters:

1. primary traversal iterations;
2. primary density samples;
3. primary skipped segments;
4. primary skipped distance in fixed-point millimetres;
5. shadow traversal iterations;
6. shadow density samples;
7. shadow skipped segments;
8. shadow skipped distance;
9. temporal candidates;
10. temporal accepted;
11. temporal rejected by bounds;
12. temporal rejected by classification;
13. temporal rejected by depth/position;
14. temporal rejected by transmittance;
15. NaN/Inf rejections;
16. active volume rays.

Counters are cleared at the beginning of a measured frame, copied asynchronously
to a host-visible ring buffer after the trace, and displayed one completed frame
later. Measurement must not add a queue wait to normal rendering.

## GPU timing

A timestamp-query ring records:

1. camera RT begin/end;
2. volume temporal resolve begin/end;
3. tonemap begin/end;
4. optional photon pass begin/end.

Results are consumed only after the corresponding frame fence signals. The UI
reports milliseconds using `timestampPeriod`; CPU submission time is shown
separately and must not be labelled GPU time.

## Resource lifetime

- Resize recreates all four history images and invalidates history.
- Camera motion preserves history and relies on reprojection.
- Projection/lens changes invalidate history.
- Scene/project load invalidates and clears history.
- Volume topology or transform changes preserve history only when stable volume
  identity and motion data are available; otherwise they invalidate it.
- Backend/device changes destroy history and query resources after the owning
  frame fence completes.

## Acceptance

- Static volume converges without changing expected radiance.
- Camera pan does not smear volume silhouettes.
- Moving smoke leaves no persistent trail after disocclusion.
- Empty volume bounds never appear in beauty or temporal history.
- Enabling instrumentation changes GPU time by less than 2% in Preview.
- Reported skipped distance increases while density samples decrease in sparse
  reference scenes.
- All counters remain bounded and no per-frame synchronous readback is added.
