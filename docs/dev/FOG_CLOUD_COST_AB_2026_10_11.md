# Live fog/cloud cost comparison

Measured through the external named-pipe client on the user's already-open Vulkan
Rendered scene. Camera, geometry, density and light settings were retained. Only
cloud reference/secondary toggles and the fog random-walk toggle were varied and
then read back for each case. Original toggles were restored and verified.

## Results

| Case | Completed samples | Wall ms / completed sample |
| --- | ---: | ---: |
| Reference clouds, full secondary, fog walk | 6 (3 to 9) | 347.85 |
| Reference clouds, capped secondary, fog walk | 6 (3 to 9) | 345.87 |
| Production clouds, fog walk | 6 (3 to 9) | 230.43 |
| Production clouds, fog march | 7 (4 to 11) | 37.62 |
| Restored reference/full-secondary/fog-walk, later samples | 6 (40 to 46) | 325.79 |

These are short, single-window application-wall-time measurements while completed
sample counters advanced, not isolated GPU timestamp timings or a statistical
benchmark. IPC scheduling/readback and frame-loop overhead are included. Adaptive
sampling and different transport paths can change work, so differences are not
additive universal costs and the march comparison is not equal-quality rendering.

In this view, the fog walk is the larger observed cost: changing to the march
under production clouds reduced approximately 193 ms/sample. Switching reference
clouds to production while retaining the walk reduced approximately 117 ms/sample.
The roughly 2 ms secondary-cap difference is too small to treat as significant in
this short comparison. The late baseline was only about 6% faster than the early
baseline: no large adaptive-cost drop was observed by sample 46. This does not
establish whether adaptation is disabled or whether noisy pixels dominate cost.

The first exploratory measurement used `viewport.render_frames(count=6)`. Its
`samples_rendered` field counted loop attempts, while the actual sample count
advanced by only two on Vulkan's asynchronous path. Those timings were discarded.
The retained probe observes natural completed-sample progression instead.

Raw settings, measurements and restoration readback are in
`fog_cloud_cost_live_2026_10_11.json`. The reusable external probe is
`scripts/test/probe_fog_cloud_cost.py`; it temporarily changes settings and resets
accumulation, so run it deliberately with the app open, not inside its embedded
script workspace.

## Shape findings

Only layer zero was enabled in the measured scene. Its type was 0.1 (near the
stratus/stratocumulus end of the four-profile interpolation), thickness 3710 m,
cell scale 2200 m and erosion 1.0. Reference transport changes lighting, not the
density silhouette. Raising its cost cannot add distinct cloud formations.

`cloud_common.glsl` uses the same repeating 3D base texture for both base and
large billows and another periodic detail texture. Layers offset the shared
weather map, but the shape still comes from a common noise basis and four blended
vertical profiles. Repetition therefore has a concrete field-generation source.

A shape improvement should preserve the bounded production renderer and work on
independent layer coordinates, spatially varying multi-scale formations and
distinct type-specific density profiles. Any new authored controls require shared
core logic, scripting and IPC alongside UI and serialization. No new cloud shape
system or permanent quality-setting change was implemented during this audit.

No project build, shader compilation or application launch was performed.
