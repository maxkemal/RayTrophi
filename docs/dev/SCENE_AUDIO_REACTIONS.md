# Continuous scene audio reactions

2026-10-12: source implementation, pending user build and live acceptance.

Follow-up: [SCENE_AUDIO_AUTOMATION.md](SCENE_AUDIO_AUTOMATION.md) adds opt-in automatic
start/stop, a packaged environment recipe, ground-reaching global cloud rain and the
first published domain-contact-energy adapter. Its lifecycle/binding description
supersedes the manual-only behavior below; detailed material-pair telemetry remains
an external producer contract.

This addition drives existing audio assets from simulation/environment signals.
It does not synthesize new recordings or simulate acoustic propagation. Wind/rain
loops and granular, splash or mud textures must be supplied by the user with valid
provenance. Several sources with different input ranges/material filters can create
layers; this is a small continuous controller, not the stage 2 timeline mixer.

## Canonical operations and editor

`rt.audio.update(id=..., reaction={...})` and `audio.update` accept the same `reaction`
object. The selected source's Reactive Audio inspector invokes this same service.
Configure a loop, select its driver, bind an appropriate asset, then Play. Changing
the driver does not auto-start/decode sounds. Existing play/pause/stop and preview
mute semantics remain. Reaction configuration is saved with the source and is
undoable through the existing audio snapshot commands; old sources default to manual.

| Field | Default / validation |
|---|---|
| driver | manual, wind, rain or signal; default manual |
| input_min / input_max | 0 / 1 (wind maximum defaults to 30); finite nonnegative, max > min |
| gain_min / gain_max | 0 / 1; each 0..1; multiplies authored gain |
| pitch_min / pitch_max | 1 / 1; each 0.5..2; multiplies authored pitch, final ratio clamped 0.125..8 |
| response_seconds | 0.25; finite 0..10; zero means immediate response |
| signal_key | required for signal driver; 1..256 characters |
| source_material / target_material | exact, case-sensitive label filters; empty is wildcard; <=128 characters |
| follow_position | false; use the signal's world position for the actual voice |

Configuration objects replace the previous reaction object; omitted fields use
defaults. Unknown fields and invalid ranges are rejected without changing the source.
The normalized input is clamped to [0,1], exponentially smoothed, then square-root
mapped to the gain range. Pitch uses the smoothed normalized input directly.
Downward mappings are allowed. A nonzero gain_min intentionally keeps a quiet floor.
Manual bypasses all multipliers. Voice loading/32-voice/decoded-memory budgets remain.

Wind reads `World::getClimate().wind_speed_mps` in m/s. Rain reads enabled
`WeatherParams` with type WEATHER_RAIN and its 0..1 intensity. It is a global artist
weather signal; cloud-shaft local precipitation/indoor shelter are not measured by
this bridge. No cloud renderer or atmosphere authority is changed. The controller
runs once per editor control update; device playback stays on miniaudio's thread.
Main-thread stalls delay reaction changes, including signal expiry fades, but do
not freeze the decoded audio stream. Offline/timeline reaction evaluation is not
implemented; the current clock is steady live runtime time, independent of timeline.

```python
rt.audio.update(id=wind_source, loop=True, spatial=False,
                reaction={"driver": "wind", "input_max": 25,
                          "pitch_min": 0.9, "pitch_max": 1.15})
rt.audio.play(id=wind_source)
rt.audio.update(id=rain_source, loop=True, spatial=False,
                reaction={"driver": "rain", "response_seconds": 0.4})
rt.audio.play(id=rain_source)
```

## Domain contact signal contract

The audio layer accepts continuous telemetry; it does not claim to observe domain
contacts automatically. Domain/physics producers should aggregate an actual region's
material transfer and contact speed at a bounded rate (for example 10-30 Hz), then
publish it. No particle readback, per-face Triangle facade or new simulation logic
has been added. Automatically publishing real sand/water/mud contacts remains a
separate adapter task after the authoritative domain telemetry is available.

`audio.signal_set` / `rt.audio.signal_set` accepts:

- key: identity including domain and contact region; <=256 characters.
- Either value (finite nonnegative scalar <=1e9), or both flow_kg_s (0..1e6) and
  speed_mps (0..1e4). Mixing these forms is rejected.
- source_material and target_material: optional producer-owned material labels.
- position: optional finite world-space [x,y,z].
- ttl_seconds: default 0.5, range 0.01..10. Repeated publications replace the same key.

For flow/speed input the scalar is `0.5 * flow_kg_s * speed_mps^2` (watts of kinetic
power proxy, <=1e9). This is an artistic driver, not a claim of acoustic power or
measured loudness. For aggregate point impacts use a producer-defined scalar instead
until a proper event/onset system is added. Multiple contacts need distinct keys;
different material sources can subscribe to one key with different filters.

At most 256 unexpired keys are retained. Expired/missing/nonmatching signals fade
toward input zero. The last valid world position is retained during the fade instead
of teleporting back to the authored transform. The inspector/gizmo edits the authored
source transform; the effective contact position is visible in runtime status.
Position following works for mono spatial sources; stereo ignores spatial placement.

```python
rt.audio.update(id=sand_splash_source, loop=True, spatial=True,
                reaction={"driver": "signal", "signal_key": "domainA/contact0",
                          "source_material": "sand", "target_material": "water",
                          "input_max": 10, "follow_position": True})
rt.audio.play(id=sand_splash_source)
rt.audio.signal_set(key="domainA/contact0", flow_kg_s=2, speed_mps=3,
                    source_material="sand", target_material="water",
                    position=[1, 0, 2], ttl_seconds=0.5)
```

Use another mud-texture source filtered to target_material="mud" for that interaction.
The producer decides material labels from its canonical state; audio never guesses
whether a fluid is water/mud from its rendering appearance or particle count.

`audio.signals_clear` / `rt.audio.signals_clear` clears transient publications and
reaction envelopes. `audio.status.reaction_runtime` exposes live signals and per-source
input/envelope/gain/pitch/effective position. The inspector has an explicit two-second
flow/speed preview and Clear Signals controls through the same operations. Publications
are runtime control, require SceneWrite, do not mark the project modified and are not
saved/undoable. Scene clear/load/undo resets runtime signals/envelopes. Reading status
uses Read. Configuration edits keep the existing SceneWrite policy and undo support.

## Pending build/live acceptance

Build using the normal user workflow; three new modules are registered in VS and
covered by existing CMake Scene/UI globs. No build or new test script was run.

1. Open an old audio scene: driver defaults to manual and existing playback is unchanged.
2. Bind user wind/rain loops; enable loop and Play. Change climate wind 0/10/25 m/s,
   then enabled WEATHER_RAIN intensity 0/0.5/1. Confirm smooth amplitude/pitch in all
   shading modes and unchanged authored gain/pitch. Verify status values through IPC.
3. Configure signal driver with a mono source and follow_position. Publish the example
   above externally through scripts/test/rt_ipc.py. Confirm input=9 and movement to
   [1,0,2], then expiry fades at that position. Use response_seconds=0 for exact endpoints.
4. Change receiver label water -> mud: the water-filtered loop fades; a mud-filtered
   source responds. Confirm missing keys and mismatched labels do not play full volume.
5. Verify invalid/NaN/negative inputs, incomplete flow/speed pairs, reversed ranges,
   excess TTL/keys and budget overflow fail without altering a valid configuration.
6. Save/reopen and undo/redo config edits: config survives; signals are transient.
   Pause/stop/mute do not create duplicate voices or restart sounds every control frame.

Genuine domain contact telemetry, one-shot event deduplication, lightning/thunder,
local rain-field evaluation, occlusion/filtering, waveform/mixer and procedural
noise synthesis remain future work. EXE export remains outside current scope.
