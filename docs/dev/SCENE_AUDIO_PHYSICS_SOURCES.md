# Physics-driven spatial sources

2026-10-17 source delivery; C++ build and post-build acceptance pending. No build,
shader compilation, Studio launch or live execution of this revision was performed.

This slice connects existing physics-controlled **asset voices** to the same
spectral transmission and early-reflection path as manual sources. It does not
add real-time fluid/fire/fracture sound synthesis, impact one-shot generation,
multiple-bounce reverb or full offline acoustic export. Solver activity starts,
stops and modulates a bound mono loop; sound content still comes from that asset.

## One runtime authority

`AudioSourceRuntime.cpp` resolves authored data, synchronized parent pose and latest
reaction output. Playback, reflection-distance correction, acoustic GPU queries,
trace/debug inspection, visualizer energy, event capture pose and viewport markers
consume that state. A measured reaction position is already world-space and never
receives the parent transform again. Missing/invalid parents remain unavailable.

Existing `audio.create/update` reaction settings author the binding. Existing
`audio.acoustics_preview` now accepts spatial manual, signal, wind, rain and
domain-controlled sources, including automatic sources. Nonspatial enable still
rejects. Later making an enabled source nonspatial/invalid bypasses that source
with `source_unavailable`; it no longer fails valid siblings in the batch.
Invalid ray coordinates similarly report `invalid_pose` for that source alone.

New Read operation, available identically in Python and IPC:

```python
state = rt.audio.source_runtime(id=source_id)
```

Required `id`: existing source ID string. Unknown parameters/missing/unknown IDs
reject through existing Python ValueError / IPC audio_error semantics. It is not
an authoring/history operation. It reports:

- `position`, `direction`, `valid`, `spatial`, `position_origin`:
  authored_world / parent_world / reaction_world.
- `driver`, `automatic`, `control_ready`, `activity`, `reaction_gain`,
  `effective_gain`, `effective_pitch`, `playing`, `timeline_held`.

Effective gain is source gain times reaction gain **before** distance/cone, mixer,
preview mute and filtering. Invalid poses give zero gain. `control_ready` means
the evaluator has run, not that a physical measurement exists; `audio.domains`
remains measurement authority. Manual sources ignore leftover modulation. Reactive
sources without an evaluated control are quiet. Signal following never rewrites
authored positions, parent transforms, names or asset bindings.

The reaction inspector shows runtime world position. Markers/cones follow it.
Transform handles are omitted while a reaction owns placement; disable Follow
position to place the source manually.

## Real solver bindings

Channels remain mixed_contact, dem_contact, dem_friction, dem_rolling, fluid_flow,
gas_flow, mpm_deformation and mpm_fracture. Existing domain publication owns
measurement/position, freshness, material limitations and rewind semantics.
This slice introduces no new per-particle scan or geometry representation.

Choose driver Domain contact, actual system/domain/channel, Follow position,
Automatic and a suitable full-activity input scale. Flow/deformation use `power_w`;
fracture uses `value` (damage growth 1/s), not invented acoustic watts. Input watts
describe solver transport/contact-work activity, not calibrated radiated acoustic
power; input-to-gain mapping is authored explicitly. Collection remains explicit
and off by default. Unavailable measurements remain quiet under
the ordinary zero-minimum-gain mapping. Binding does not start simulation.

UI and equivalent Python/IPC workflow:

```python
rt.audio.domains()  # choose an actual binding and inspect measurement units
rt.audio.update(id=source_id, loop=True, spatial=True, reaction={
    "driver": "domain_contact", "system_id": system_id, "domain_name": domain_name,
    "domain_channel": "fluid_flow", "input_metric": "power_w",
    "input_min": 0, "input_max": calibrated_full_activity_watts,
    "follow_position": True, "automatic": True, "response_seconds": 0.08})
rt.audio.acoustics_preview(id=source_id, enabled=True, spectral=True, reflections=True)
# Only when the user chooses to collect physics audio telemetry:
rt.audio.telemetry_update(enabled=True)
```

External helper (replace placeholders with actual domain and existing mono asset):

```powershell
python scripts/test/rt_bind_physics_audio.py --system-id 1 --domain Faucet --asset-id audio_asset_1 --channel fluid_flow --input-max 0.5
```

The example is not a universal calibration or a claim that Faucet exists. The
helper leaves collection unchanged unless `--enable-telemetry` is explicitly
supplied. It validates binding/asset, creates an initially nonautomatic source,
configures acoustics, then enables automation. Failure removes that new source
and restores a telemetry toggle made by the helper. IDs/undo history can advance.
It does not save, start simulation or change existing domain data/solver activity.

## Moving-source GPU lifecycle

UI order is transport/parent sync, listener, domain/weather reactions, then
acoustics. Queries sample the same pose playback just consumed. GPU results are
polled every UI frame before throttling submissions to approximately 4 Hz. A
bounded in-flight sample finishes while the source/listener moves; the next query
uses the newest pose. Motion/timeline-frame changes alone no longer cancel every
result before it can be published.

Ownership checks reject changed scene/backend, audio generation/rewind, geometry,
materials, source IDs and preview configuration revisions. Result age above one
second is rejected; stalled moving requests are superseded after that bound.
Invalid source rays/poses are isolated. No synchronous acoustic CPU BVH is built.

Reports expose `pose_policy=sampled_runtime_world`, `position_origin`, actual
sampled source/listener positions, and `result_age_ms` at publication. That age is
request-to-publication time, **not GPU cost or physical travel delay**. Pose drift
between samples is expected; `acoustics_debug.result_pose_matches` exposes exact
agreement with current runtime pose. During refresh, status/solver indicate pending
while `report` retains the applied snapshot; `pending_report` describes the pending
request. Coefficients remain smoothed. First/failed queries retain dry-bypass policy.

Bounds remain 32 previews/resident voices, six sparse first-order candidates per
source, roughly 4 Hz queries and one-second excess delay. Automatic stop currently
releases the voice/node and can truncate echo tails. Capture/export do not reproduce
live spectral filters/echoes; capture shares effective pose/gain/pitch but retains
its previous broadband model.

## User build and acceptance

The first live inspection exposed a descriptor lifetime bug in audio discovery:
the registry retained a loop-local descriptor address. Audio registration now owns
descriptors in a deque with stable addresses. The fix still requires a user C++
rebuild. Before advancing physics, verify repeated external `agent.describe` calls
for `audio.source_runtime` and `audio.list`, plus audio discovery/search, complete
without an access violation and retain distinct method names/parameter schemas.
The inspection was read-only; telemetry remained disabled and no audio was added.

1. Rebuild Studio C++ with new AudioSourceRuntime.cpp/.h (registered in project and
   filters; CMake Scene glob discovers it). No shader/ABI change in this slice.
2. In a disposable geometry/audio-empty scene, run externally:
   `python scripts/test/rt_test_reactive_acoustics_ipc.py --report docs/dev/reactive_acoustics_acceptance.json`.
   It injects controlled flow signals and checks automatic start/expiry/stop/resume,
   world pose without parent double transforms, wall attenuation in native PCM,
   moving-query completion, reflected paths and sibling isolation. This tests the
   bridge, not an actual fluid solver producing sound.
3. With a real small running fluid fixture, explicitly enable collection, inspect
   audio.domains, bind fluid_flow using UI/helper and check measured input/position,
   runtime gain, GPU reports and audible filters/echoes. Stop flow and verify
   silence; disable collection and check stale activity clears. Repeat forward RAM
   replay/rewind. Injected signals/control_ready are not evidence of real measurement.
4. Run existing direct-path and acoustic-playback probes for manual regression.
   Check source/scene changes during in-flight queries, missing/invalid parents,
   remove/rebind, and GPU coverage errors. Use Material after geometry edits if
   Solid retains an old raster AS. Check UI/Python/IPC parity, follow-position
   gizmos/inspector, timeline hold/resume and explicit Stop All suspension.

Full faucet synthesis/events, late tails, export parity and performance acceptance
remain open. No live acceptance is claimed for this revision before the user build.

## Live solver check, 2026-10-17

After the user's rebuild, repeated `agent.describe` calls for distinct audio
methods, capability search and `agent.discover` completed without the previous
registry access violation. The existing matter fixture (system 1, Physics Domain 1)
was advanced by 15 explicit 1/60-second solver steps, without injected signals.
The first 12 samples measured fluid-flow activity from 8.35 to 958.83 W with
16 to 200 carriers. These are kinetic transport activity units, not radiated
acoustic power. Runtime source positions matched each sampled activity center.
Automatic playback, expiry stop, restart on fresh measurement, Stop All suspension
despite a fresh step, explicit resume and stop after disabling collection passed.

The mono wood fixture only verifies transport/control; it is not a water sound
or synthesis acceptance. GPU queries completed after selecting Material viewport,
but returned `unknown_material` (10 crossings), so this scene did not validate
audible spectral transmission or reflections. No geometry or surface material was
edited. The test source/asset remain in the open scene, unsaved; telemetry is off
and playback mode is restored to timeline. Evidence:
`docs/dev/physics_audio_live_20261017.json`. Performance and continuous timeline /
cache replay acceptance remain open.
