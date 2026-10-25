# Open wet Sand scene: controlled external IPC diagnosis, 2026-10-23

Measurement only. No build, application launch, solver edits or project save.
The open scene had one enabled Matter domain and no flow sources. Initial live
particle count was zero; scene was paused at frame 0.

## Method

`scripts/test/rt_probe_wet_sand_scene_ipc.py` seeds the existing region and runs
96 steps at dt=1/60 (1.6 seconds), recording every eight steps. Comparisons use
temporary Sand-derived materials; original built-in Sand is never edited.
Initial position hashes must agree. Normal restitution changes in these probes
apply to ALL grain contacts; they do not implement wet-only damping.

Domain bounds +/-2.5 m, grain radius 25 mm, real radius 0.25 mm, restitution 0.5,
rolling friction 0.2, water capacity fraction 0.1, seed saturation 1.0.
One 2 m wide cube collider has top y=-0.92627 m and friction 0.15. The seed
footprint is 2.25 m square, so its edges initially overhang the collider.
Capacity truncates the seed at 50,000 grains (requested counter 50,001 is the
truncation sentinel, not a count of all possible grains). This is not a small,
initially supported wet column test.

## Interpretation limits

- Wet bridge counters count directed visits in the last substep, not unique
  persistent bonds, integrated broken bonds or connected clumps.
- The pile repose/scattered metrics are inappropriate as physical acceptance
  while falling over a cube and hitting closed domain walls. Use them only as
  diagnostics; compare mass moments and kinetic energy instead.
- Kinetic energy differences include gravity and contact trajectories, not a
  measured capillary dissipation budget. Contact work telemetry was unavailable
  (`measured=false`), not measured zero.
- Same-material equal-size Sand does not exercise mixed real radii. A per-grain
  real-radius channel cannot alone change this baseline if values remain equal.
- The current shader adds a central attractive bridge force and uses the same
  normal damping ratio from grain restitution for wet and dry contacts.
- These runs test sensitivity, not physical calibration or dt convergence.

## Completed four-arm comparison

All arms started with identical position hash `81ea2412e563ff58`, 50,000
grains, dry mass 8,726.64526 kg and water 327.24927 kg. Water mass remained
constant, with reported balance error zero. All sampled steps completed, with
50,000 DEM owners and no MPM/fluid owners. No sampled substep-cap failure,
manifold truncation or contact-history overflow was reported.

| Arm | Real radius | Restitution | RMS radius at 0.8 s | RMS radius at 1.6 s | Transport KE at 1.6 s |
|---|---|---|---|---|---|
| Original | 0.25 mm | 0.5 | 1.396 m | 2.0923 m | 34,002 J |
| Low rebound | 0.25 mm | 0.1 | 1.365 m | 2.1004 m | 32,254 J |
| Fine radius | 0.10 mm | 0.5 | 1.353 m | 2.0244 m | 25,722 J |
| Both | 0.10 mm | 0.1 | 1.330 m | 2.0592 m | 28,187 J |

At 0.6667 s, low rebound lowered kinetic energy from 64,741 J to 53,498 J
(17.4 percent). It did not meaningfully lower final global spread. Fine radius
lowered final spread 3.2 percent; both changes together lowered it 1.6 percent.
Later samples include closed-wall effects; this is not a monotonic ranking of
physical material quality.

Original active directed bridge visits fell from 572,572 at 0.4 s to 163,160
at 0.8 s; the combined arm retained 242,242 at 0.8 s. These numbers establish
substantial loss of instantaneous connectivity during impact/spreading, but
cannot distinguish isolated grains from disconnected internally bonded clumps.

## Conclusions

1. Wet birth, retained water, and the attractive bridge force work in this scene.
   The single-material real-radius and restitution overrides reach the runtime.
2. The two proposed parameter edits have measurable effects but do not establish
   the requested macroscopic behavior. Adding a wet-only restitution property
   is not supported as a sufficient fix by this test.
3. Per-grain real radii remain a separate mixed-material correctness task, not
   an explanation for failure of equal-radius, single-material Sand here.
4. The current contact model has no separate wet viscous damping. This is a
   verified model limitation, not proof it dominates this scene's spreading.
5. A complete root cause remains open: global radius/energy and bridge visit
   counts do not measure connected clump sizes, bond lifetimes, tensile support,
   or a converged cohesive strength. The falling overhanging block cannot by
   itself calibrate static wet-sand strength. Next discriminatory test is a
   short, supported column with the same material and grain scale, followed by
   matched low/high impact and connected-component measurements. Do not tune
   global cohesion or ship a damping field on this evidence alone.

After the comparison the original Sand default, PPC 16, seed region, saturation
1.0 and collider were preserved/restored. The scene is paused at frame 0 with
zero live particles; temporary diagnostic materials were verified removed.
Raw files: `wet_sand_scene_{baseline,low_rebound,fine_radius,fine_low_rebound}_20261023.json`.

## Follow-up: apparent explosive collider response

User reports that first bottom contact appears to eject the entire block. A
new original-material run sampled every 1/60 s through 0.8 s, followed by a
1/120 s frame run sampled every second step. Raw suffixes: `impact_dt60` and
`impact_dt120`. First sampled support contact was at 0.51667 s in both runs.

In the 1/60 run, non-centroid transport kinetic energy
`K - |P|^2/(2M)` changed from about 6 J at 0.5 s to 17,775 J at 0.5333 s,
26,098 J at 0.55 s and 46,643 J at 0.6 s. Instantaneous bridge visits initially
increased (572,572 to 607,600) while the contact network compressed; their large
decline came afterward. This is a rapid conversion of coherent falling motion
into relative motion, not evidence that all particles receive a direct collider
force simultaneously.

The shader queries nearby triangles independently for each grain, accumulates
at most four deduplicated supports and integrates forces each microstep. No
all-grain collider push was found in this inspected path. No history reset,
contact-history overflow or support-manifold truncation was reported at impact.
Transport KE + gravitational energy fell from approximately 77.3 kJ before
impact to 72.4 kJ at first sampled contact and 42.7 kJ at the next sample.
This is NOT a complete energy ledger: stored contact/bridge/spring energy and
rotational energy must be included before asserting passivity or conservation.
No net transport/gravity energy excess was demonstrated by these samples.

At 0.8 s, horizontal RMS radii were 1.39627 m and 1.39414 m (0.15 percent
difference). Halving frame dt did not remove global spreading. This does NOT
establish DEM integration convergence: adaptive physical microsteps remained
similar (roughly 0.18-0.23 ms). A separate higher-contact-resolution probe would
be needed for that; it was not run after the user steered toward Snow behavior.

## User steering: Snow already has the desired block response

The current Snow source now has an actual Stomakhin-style MPM branch with
elastic/plastic strain, compression hardening and dilation weakening; it is not
the older DP preset. Sand remains force-based DEM with central capillary forces.
Snow's useful behavior is therefore a constitutive-model comparison, not merely
a density comparison. Free-fall acceleration is independent of mass.

Important reuse constraint: `Granular::snowMinPlasticJacobian()` hardcodes ice
density 917 kg/m3. Simply applying the Snow branch at Sand bulk density 1600
clamps minimum plastic Jacobian to 1, eliminating the intended compaction range.
A wet-sand continuum model needs its own packing/solid-density limit, friction
and moisture-dependent cohesion, plus compression and breakup acceptance.
Copying Snow and raising density alone is not a validated solution. No such
material/model change was applied during these measurements.
