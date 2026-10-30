# Paused two-gas RT cost matrix, 2026-10-30

External named-pipe measurement; no builds, app launch, physics step or camera
edit. Normal Vulkan pipeline with both volume stages verified present. Frame
99, Pause, 1680 x 945, FOV 40 degrees. Camera position
(11.524720, 8.967054, 22.161875), target (2.239807, 3.464158, 2.849707), up Y.
Orbit azimuth 25.6773 degrees and elevation 14.4027 degrees, computed from
position minus target. The raw full camera and material settings are recorded.

A = Vapor / Unattributed Gas; B = Fuel Jet Pilot #1, under
`Ignited Fuel Jet Matter #1`.

GPU medians use identical progressive sample indices 8..23, instrumentation
counters off. Counters come from a separate full-viewport pass after at least
eight progressive samples, normalized by paths_traced. Their values are workload
indicators, not separate component timings.

| Case | GPU median ms |
|---|---:|
| Both gas fields hidden | 5.138 |
| A only | 19.345 |
| B only | 12.640 |
| A+B | 123.111 |
| A+B repeat | 122.922 |

| Counter / traced path | A-only + B-only | A+B |
|---|---:|---:|
| Volume rays | 0.26577 | 0.17731 |
| Camera density samples | 9.01232 | 13.43590 |
| Secondary density samples | approximately zero | 0.0014 |
| Shadow density samples | 0.67811 | 18.88979 |
| Majorant queries | 3.08506 | 9.28303 |
| Empty segments skipped | 4.94824 | 2.21820 |
| Gas handoffs | 0.00390 | 0.07391 |
| Step budget exhausted | 0 | 0.11255 |
| Solid probes | 0.32860 | 0.24284 |
| Light-cache hits | 0.87892 | 0.00138 |

Baseline-corrected additive estimate A+B-hidden = 26.847 ms, compared with
123.111 ms measured together. This is a diagnostic reference, not a performance
guarantee or a physical linearity law. Camera density sample count increases
1.491x relative to the sum of single-field runs; shadow samples increase 27.857x,
majorant queries 3.009x, while volume ray count is lower than that sum. The
shared path loses nearly all observed light-cache hits. These observations
point toward sampling/traversal and cache-policy differences, not an explosion
in incoming ray count. They do not isolate GPU instruction or memory latency,
nor prove that shadows alone account for the excess time. Earlier shadow-zero
ablations remain relevant and prevent that conclusion.

No travelled-distance/path-length counter was present in the returned payload;
none was inferred from other counts. Geometry/opacity changes can reduce empty
space skip opportunities; the skip count alone does not quantify wasted work.

Camera, frame, all non-visibility material settings, render settings and the
density/temperature float-bit hashes stayed identical across the five cases.
Publication state_version increments on visibility resync; an initial attempt
was stopped for that metadata-only change, verified unchanged voxel hashes,
then repeated with identity based on actual data rather than publication version.
Original visibility, timing and counter settings were restored. No screenshot
was created. Raw report: ignored temporary artifact
`docs/dev/.captures/rt-gas-matrix-20261030.json`.
