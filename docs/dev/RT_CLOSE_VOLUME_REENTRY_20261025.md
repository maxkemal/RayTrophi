# Near-camera mixed-volume cost — 2026-10-25

External IPC probe of the user's fixed near camera, frame 28, Physics Domain 1.
Current phase routes: gas=fog, liquid=sdf, solid=follow. Fog publication includes
Snow solid carriers and the aggregate gas field; a separate liquid SDF is also
visited by the arbiter counters. No build, application launch, camera move,
simulation stepping or performance implementation change was performed.

Raw evidence: [rt_volume_close_reentry_20261025.json](rt_volume_close_reentry_20261025.json).
Reproduction: `scripts/test/rt_probe_volume_reentry_ipc.py --output <report.json>`
from an external Python terminal while the application is open and paused.

## GPU timings

Whole camera dispatch, exactly sample indices 0..7 per case, atomic counters off:

| Physical bounce budget | Mean GPU ms |
| --- | ---: |
| Original: total 10, diffuse 4, transmission 8 | 363.51 |
| Total 1 | 146.03 |
| Total 2, diffuse/transmission 2 | 200.40 |
| Original repeated | 367.63 |

Original/repeat agree within 1.2%. Reducing the budget to one physical bounce
cuts dispatch time by about 60%, but leaves substantial primary/surface work.
Adaptive sampling remains enabled; these ablations change light transport and
are not equal-quality rendering modes or isolated volume stage timings.

## Separate full-image counters, normalized per path

| Counter | One bounce | Original |
| --- | ---: | ---: |
| Recorded gas/fog volume rays | 0.6323 | 0.9119 |
| Recorded camera density samples | 23.406 | 26.131 |
| Recorded shadow density samples | 7.014 | 12.474 |
| Solid probes (gas plus SDF branches) | 0.7526 | 1.2413 |
| Gas-to-solid handoffs | 0.0323 | 0.2025 |
| Skipped empty quadrature segments | 13.163 | 13.219 |
| Paths exhausting free-pass budget | 0 | 0.000000126 |

Recorded fog crossings increase about 44%, solid probes 65%, shadow samples
78%. This confirms additional secondary work, but does not attribute all GPU
time to it. The tiny free-pass-cap rate does not support a widespread runaway
handoff loop in this camera. More than one legitimate crossing per path would
not by itself be a loop: reflection/refraction changes the ray and its interval.

## Source audit: what empty counts do not show

`volumeRecordRay` accumulates gas/fog march density and skipped quadrature
positions. A skipped position is neither a unique voxel nor a saved GPU query;
many positions may be eliminated by one hierarchy lookup, and different rays
may visit the same region independently.

`nearestSurfaceSDFCrossing` walks an overlapping SDF with regular samples and
refinement. It does not use `nanoEmptyIndexStep`, and its raw `sampleIsoField`
reads do not increment camera density/empty counters. The main `source_type==4`
closest-hit branch likewise regularly scans its iso interval and refines the
crossing without reporting those reads through the gas camera sample counters.
A gas-to-SDF handoff can therefore search a surface for ordering and then invoke
the surface's own search. A high empty-skip count is not evidence that these
separate searches were skipped.

The earlier whole-instance zero-density rejection addresses a different case:
a truly optically inert gas need not invoke closest-hit at all. Nonempty media,
SurfaceSDF and nearby splats must still preserve ordering and visibility.

## Next optimization targets

1. Give SDF main/arbiter searches explicit counters before interpreting gas
   density/skip counters as total volume work. Separate intervals, raw samples,
   hierarchy queries and handoffs; collect counts apart from GPU timings.
2. Add conservative SDF empty-space proof using the raw proxy field and its
   trilinear support. A region proven below the iso threshold can skip searching
   for an entry, subject to surface modifiers; interior/exit searches need their
   own bounds. Fog remap/cutoff and optical density must never authorize hiding
   a geometric surface.
3. Investigate sharing surface-distance results on an unchanged ray/interval,
   and limiting SDF searches by a nearer solid. Do not reuse the result after a
   physical reflection/refraction changes direction, or globally mask other
   gas/SDF/splat content to remove repeated work.

No distant-camera, phase-hidden or stage-clock ablation was performed, so no
exclusive root cause or post-fix speedup is claimed. Render settings, camera and
frame were verified restored/unchanged; both diagnostics were left disabled.
