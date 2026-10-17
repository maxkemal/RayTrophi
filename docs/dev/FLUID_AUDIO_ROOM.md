# Fluid audio room — 2026-10-17

Saved by external IPC in the user's running Studio; no build or launch performed.

- Project: `x64/Release/assets/fluid_audio_room/FluidAudioRoom.rtp`.
- Before-project: `x64/Release/assets/fluid_audio_room/FluidBeforeAudioRoom.rtp`.
- Incremental setup checkpoint: `x64/Release/assets/fluid_audio_room/session.json`.
- Existing fluid domain/emitter retained. The original cube is a low slab below
  the fluid; six new flat meshes form a 16 x 12 x 60 m acoustic room. Slab and
  room share a derived Stone diagnostic profile with explicit six-band absorption,
  zero transmission and 0.02 scattering. These are test coefficients, not a
  calibrated construction specification. No new simulation colliders were added;
  the existing closed domain provides fluid confinement.
- Camera/listener: [0, 1.7, -10], looking toward the fluid, FOV 65.
- Water source: `Water - solver flow` / `audio_source_2`, existing CC0 mono water
  recording with original pack attribution retained. Real `fluid_flow` telemetry
  controls automatic playback, gain and world position; 1000 W activity input
  maps to full gain (0.45 authored). Activity is not radiated acoustic power.
- Reference: `Clap - echo reference` / `audio_source_4`, dry procedural clap,
  manual playback. It is not a physical splash detector.

## Use

Open the project and use timeline playback for the water reaction. Collection is
explicitly enabled in this test project; it is not a global default. The reference
is left stopped. For an isolated clap audition, use continuous audio mode and play
the reference; return to timeline mode to test timeline-driven water. Stop All
suspends automatic sources; resume automatic audio or restart timeline playback
as documented by the shared transport service.

Live acoustic preview settings are session-only. After reopening, restore them
externally with:

```powershell
python scripts/test/rt_setup_fluid_audio_room.py --resume
```

This command checks the project identity and source IDs before touching previews.
It does not play, enable telemetry, recreate geometry, save, or build. Initial setup
without `--resume` refuses an existing checkpoint to prevent accidental duplication.

## Verified and open

Four real solver samples in the constructed room measured 38.80..178.50 W transport
activity. Each started the water source and reported `reaction_world` positioning.
Both audio sources were exercised through the native playback service. Results are
in `docs/dev/fluid_audio_room_acceptance_20261017.json`.

**Echo/filter acceptance is blocked**, not passed: current flat-only acoustic
coverage rejects a non-flat object in the live fluid scene with
`Non-flat scene geometry is outside acoustic coverage`. Temporarily hiding the
fluid or routing mist to splats did not resolve it; both changes were restored.
The failing path dry-bypasses. Do not describe this project as a working wet room
until the non-flat identity/AS coverage issue is resolved and re-tested. Dynamic
fluid geometry is not yet an acoustic boundary model. No new synthesis, late
reverb, export parity or performance acceptance is claimed.

## Core continuation (requires user rebuild)

The authored-flat-environment policy now distinguishes simulation-owned transient
TriangleMesh/VDB visuals from unsupported authored geometry. RayFusion marks
simulation pool slots without compacting the TLAS-indexed acoustic table. Both
direct and reflected queries, and the surface debug shader, explicitly reject
those slots as boundaries. Opaque triangle auto-commit is disabled for reflection
queries so a display splat cannot hide a real wall before filtering occurs.
Unknown authored surfaces and incomplete AS coverage still fail conservatively.

CPU diagnostic trace uses the same object classification. Existing Python/IPC
`audio.acoustics_trace`, preview-status and debug operations expose the shared
results, without new UI-only logic. Preview reports identify
`coverage=authored_flat_environment`, `simulation_visuals_are_boundaries=false`,
`ignored_simulation_objects` and `ignored_simulation_instances`. CPU trace reports
`ignored_simulation_visuals`. Debug reports the same boundary scope. Snapshot
completeness refers to this environment scope, not to fluid/gas wave propagation.
The RT provider retains its existing stricter AS coverage checks; this change does
not promise support for procedural volume entries in a progressive RT TLAS.

No new test script was added for this core change. User verification checklist:

1. Rebuild C++ and **both** `acoustic_segment.comp` and `acoustic_surface.comp`.
   GPU struct sizes remain unchanged; marker semantics changed, so old shader
   binaries are incompatible with the new classification.
2. Open this saved project, restore session previews with the existing `--resume`
   command, keep Material viewport and run the fluid. Verify completed reports
   for water and clap, nonzero reflected paths and current runtime source pose.
3. Use the clap alone for dry/processed listening; compare room material changes.
   Water activity/position must still follow real solver measurements.
4. Toggle mist/splat visibility and stop/restart flow; these display choices must
   not turn particles into sound barriers. Large candidate/AS budgets still fail
   explicitly; they are not silently expanded.
5. An undefined authored wall must still report unknown material; an unsupported
   non-transient object must still reject coverage. Recheck surface debug and
   Python/IPC report agreement. The saved room file itself was not altered by
   this source-only continuation. No live acceptance is claimed yet.

## Post-build observation

After the user's shader/C++ rebuild, both previews completed in the saved room.
After six real fluid steps, both settled to `clear` / `ready`, with five reflected
paths each, complete authored-environment snapshots and 272 ignored simulation
instances. Water excess delays were 15.6, 19.8, 24.0, 70.0 and 174.9 ms; clap
delays included 69.9 and 168.8 ms. Request-to-publication age was about 505 ms,
which is neither GPU execution cost nor physical propagation time. Water started
automatically in five of six sampled steps and followed measured world positions.

During the stepping sequence, completion invalidation caused `gpu_unavailable`
with the old combined error `Acoustic snapshot expired or scene ownership changed`.
Consequently moving/continuous-simulation acoustic acceptance remains open despite
successful settled-scene coverage. Evidence:
`docs/dev/fluid_audio_core_acceptance_20261017.json`. No native-PCM dry/wet tail
measurement or listening acceptance was performed in this observation.

Source follow-up splits completion errors into coverage unavailable, AS revision,
material revision, audio ownership and timeout (including actual age). No validity
guard was weakened. This diagnostics-only follow-up requires C++ rebuild, with no
additional shader change. Playback mode was restored to its previous timeline
setting and the saved scene's authored settings were not overwritten.

## Completion revision fix — 2026-10-18 (source, not live-accepted)

The rebuilt diagnostic reproduced `Acoustic completion coverage unavailable:
Raster AS has not incorporated current geometry/instance placements` during real
fluid steps. The raster acoustic stale-placement gate included simulation pool
transforms even though those instances are excluded from the boundary model.

The AS now snapshots separate authored-environment geometry/placement signatures
when a TLAS build succeeds. Acoustic coverage compares current environment
signatures with those snapshots. The full renderer placement signature, exact
TLAS-indexed table and counts remain unchanged. A new CPU-only `boundaryRevision`
tracks environment buffers/placements plus geometry generation and skin refits;
completion validates this instead of the full AS identity. Submission still uses
the full revision and its original `stillCurrent` guard. Thus an already completed
summary need not be rejected solely because ignored visual placements changed;
no stale GPU pointers are reused. The stricter progressive RT provider uses its
full revision as boundary identity. Material, owner, age and coverage guards remain.

User build: C++ only, no shader changes. In the saved room, restore previews and
check repeated real fluid steps/continuous playback retain completed reflections.
Then move an authored wall and verify the old result is rejected until its AS
refresh; also check material edits and source remove/rebind. Global geometry or
skin revisions, coverage budgets and the 1000 ms age limit may still invalidate
results. This revision has only source/static checks; no fresh live acceptance or
performance claim is made. No additional test script was created.

### Post-rebuild verification

External IPC verification of the user's next C++ build passed the bounded
regression: eight real 1/60-second fluid steps produced no completion coverage /
revision errors. The first sampled previews were pending with retained paths;
the next seven samples were `clear` / `ready` for both sources with five reflected
paths. Water played automatically in those seven samples. Completion ages were
about 504..615 ms (publication latency, not GPU execution time).

A temporary +3 m X move of `FluidRoom right` changed the clap's first reflected
distance from 23.6411 m to 27.6030 m. Restoring the original transform restored all
five distances exactly at four-decimal reporting precision. This confirms genuine
wall changes still update the environment result. The wall and audio playback mode
were restored, and the saved project was not overwritten. No new test script was
added. Evidence: `docs/dev/fluid_audio_revision_acceptance_20261018.json`.

This closes the reproduced step-driven revision regression. It does not claim
long continuous-timeline/cache-replay stability, native-PCM echo-tail acceptance,
multi-source performance, or dynamic fluid-boundary acoustics.
