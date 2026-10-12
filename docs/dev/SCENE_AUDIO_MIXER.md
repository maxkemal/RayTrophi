# Scene bus mixer

2026-10-12 source implementation; user build/live acceptance pending. Codex did
not compile, launch Studio or create a test script.

## Shared API

UI, Python and IPC call AudioAuthoring -> AudioService -> AudioMixer/runtime.

| Operation | Parameters | Policy |
| --- | --- | --- |
| audio.mixer_get / rt.audio.mixer_get | none | Read; settings and actual PCM meters, no device initialization. |
| audio.mixer_update / rt.audio.mixer_update | required bus; optional gain, muted, solo | SceneWrite; validated partial patch, saved and undoable. |
| audio.mixer_reset_peaks / rt.audio.mixer_reset_peaks | none | SceneWrite; runtime hold/overload reset, not undoable. |

```python
rt.audio.mixer_update(bus="Weather", gain=0.5)
rt.audio.mixer_update(bus="Weather", solo=True)
rt.audio.mixer_update(bus="Master", muted=True)
rt.audio.mixer_get()
rt.audio.mixer_reset_peaks()
```

```json
{"method":"audio.mixer_update","params":{"bus":"Effects","muted":true}}
```

Fixed buses: Master, Weather, Effects, Ambience, Music. Gain is finite linear
amplitude 0..4, zero silent. UI accepts dB with Enter committing one history edit;
-60 means zero, maximum +12.04. Omitted fields remain unchanged. Unknown bus/key,
invalid bool/type, nonfinite/out-of-range gain and Master solo=true fail through
common audio errors before replacing settings. Master supports gain/mute, no solo.

Any child solo excludes nonsolo children and directly Master-routed sources.
Multiple solos combine; mute overrides solo. Child faders feed Master, so Master
gain/mute affects all scene sources and thunder. Source gain/reaction stay separate
and multiply these gains. Bus changes reroute an existing voice without restarting
its cursor. Thunder routes to Weather. Mute/solo do not stop/restart voices; their
clock still follows the existing timeline transport.

## Real routing/meters

Lazy initialization builds miniaudio groups and meter nodes:
child group -> child meter -> Master group -> Master meter -> engine endpoint.
Direct Master sources feed Master. Spatialization remains at the source; groups
do not spatialize the stereo mix again.

Stable callbacks pass through real stereo float PCM and measure block peak/RMS,
peak hold and blocks at/above full scale. These are not gain-based estimates.
Callback code uses bounded PCM loops and lock-free 32-bit atomic scalars, with
no JSON, scene state, files, allocation or locks. Silent inputs process continuously
so current readings return to silence after Pause/Stop. One miniaudio implementation
remains in AudioService.cpp.

mixer_get reports authored settings, effective_gain (local post-mute/solo fader),
output_gain (including Master), and meter per bus. direct_master_gain is the
child-solo gate for direct Master sources. Meter fields: peak, rms, peak_dbfs,
rms_dbfs, peak_hold_dbfs, overload_blocks, ready. Silence is -120 dBFS. Atomic
fields are independently sampled telemetry, not a coherent multi-field snapshot.
Current values are per-block; hold/count persist until reset/load/clear.

Child readings are post-child-fader/pre-Master; Master measures its aggregate
after Master gain. All are before preview output mute. Asset audition bypasses
scene buses, is excluded from meters, and still obeys preview output mute and
existing exclusive-audition ownership. UI explicitly states this distinction:
Master mute does not silence independent asset audition.

## Persistence/lifecycle

Optional audio schema-1 mixer object stores gain/muted/solo keyed by fixed bus
name. Old scenes default to unity, unmuted, unsoloed buses. Shared snapshot/save/
restore/undo validate the same structure; meters are transient. Settings can be
authored without hardware and are applied before future playback. Read never
opens a device. Graph failures detach partial nodes and report mixer_unavailable.
Voices are destroyed before nodes/groups; mixer shuts down before engine.

Undo/redo follows existing audio snapshot semantics: transient voices/events stop;
automation can restart on later ticks. Ordinary mixer edits retain cursors.
Clip automation, effects/limiting, waveforms and offline rendering remain later work.

## Acceptance after user build

1. Scene Audio > Audio mixer: five unity buses. Before playback, ready=false and
   reading mixer opens no hardware. Play: Weather/Master meters react to wind/rain/
   thunder; empty buses stay silent.
2. Weather gain 0.5 gives about -6.02 dB; Master gain 0.5 adds another -6.02 dB
   only to Master readings. Weather meter remains at its child level.
3. Weather mute silences wind/rain/thunder while cursors advance. Solo Weather
   excludes Effects/Ambience/Music/direct Master. Weather+Effects solos combine;
   muted+solo Weather stays silent. Master mute silences the entire scene mix.
4. Change a playing source's bus without cursor restart. Pause: current meters
   fall to silence, holds remain. Edit mixer while paused, resume with new gains
   and retained cursors. Stop keeps settings but clears voices.
5. Preview output mute preserves bus measurements/cursors. Asset audition bypasses
   Master mute and is excluded from scene meters. Reset clears holds/counts while
   current measurements continue.
6. Save/reopen and undo/redo settings. Check unknown bus/key, gain -1/5/NaN,
   wrong bool and Master solo errors. Read-only IPC can get but not update/reset;
   SceneWrite can edit levels without FilesRead because no file decode occurs.

Live probes use external scripts/test/rt_ipc.py, never the embedded workspace.
