# Reactive audio transport

2026-10-12 source contract. New mode behavior awaits the user's build/live acceptance.
Codex did not compile or launch Studio or create a test script.

## Shared operation

Scene Audio > Audio playback has Continuous preview and Follow timeline (Play /
Pause). UI, Python and IPC use AudioAuthoring -> AudioService:

```python
rt.audio.playback_update(mode="timeline")
rt.audio.playback_update(mode="continuous")
rt.audio.status()
```

```json
{"method":"audio.playback_update","params":{"mode":"timeline"}}
```

Required mode must be exactly continuous or timeline. Invalid/missing/nonstring
mode and unknown parameters fail through existing audio errors. SceneWrite +
FilesRead is required because automation can decode files. The setting is undoable,
marks the project modified, and is idempotent if unchanged. Switching modes clears
scene voices/events/signals and re-arms configured automation in the new mode.

Audio schema 1 stores optional playback_mode, default timeline for old scenes.
Save/load/undo use the common snapshot path. Load/undo resets transient state;
the next editor tick supplies the actual scene clock/play state.

## Behavior

| Action | Continuous preview | Follow timeline |
| --- | --- | --- |
| Pause | Ambience and emitted thunder's wall-time delay can continue. | Hold scene voices/tails, pending thunder, signal TTL and reaction envelopes. |
| Play | Re-arm automation. | Re-arm and resume held voices at current PCM cursors. |
| Timeline Stop | Existing preview behavior. | Common audio.stop_all cancels voices/queue, including when already at start frame. |
| Seek/loop | Existing thunder seek clearing. | Clear voices, queue, signals and controls; automatic sources restart only when playing. |
| Source Play while paused | Ordinary playback. | Hold until timeline Play. Explicit source Pause/Stop cancels its held-resume request. |
| Asset preview | Exclusive audition. | Exclusive audition remains available while paused. |
| Mute | Output mute only. | Output mute only; transport is unaffected. |

The editor feeds scene frame/FPS before weather/domain/thunder updates. Negative
temporary reset sentinels clamp to zero. A pause holds decoded cursors instead of
Stop All. Explicitly paused manual sources stay paused; automatic sources can
be re-armed by Play or audio.resume_automatic. Stop suspends until a new Play or
explicit re-arm; Pause preserves pending events.

Timeline mode uses scene seconds for thunder arrival, runtime signal TTL and
reaction smoothing. Paused weather/domain bridges do not publish/evaluate inputs.
Climate/cloud values and existing published domain telemetry remain authoritative;
this change does not create or advance a separate simulation. On seek, the audio
generation invalidates old domain publication baselines.

Backward seek, loop, stopped scrub and jumps >2 s clear transient timeline audio.
Smaller forward seeks while playing are indistinguishable from normal advances
with the existing timeline interface. Weather seed/rate changes now preserve
already-emitted thunder. Its key includes seed/effective period/lane/cycle.
Current-schedule onset enumeration does not reconstruct historical weather or
camera positions when seeking.

This is reactive audio transport, not a clip editor/offline renderer. Playing
decoded ambience/tails run on the device clock and are held on Pause; their PCM
is not resampled or phase-seeked to every video frame. Event arrivals/TTL use scene
time. Reaction smoothing retains its 0.25 s per-update bound. Offline sample-exact
evaluation, waveform clips and historical seek remain later work.

## Diagnostics and acceptance

audio.status.playback_runtime: mode, playing, paused, time_seconds, dt_seconds,
seek_count. Scene/thunder voices include cursor_frames and timeline_held;
transport_held_sources lists held ids. Thunder UI shows detected identities and
next pending distance/delay. UI explains paused silence and disables its thunder
test while transport is paused.

In the existing build, live IPC found the scene paused at frame 526, then stopped
at zero. On user Play, Automatic thunder was disabled and mute enabled. Enabling
thunder through the service while preserving mute produced real event identities
and pending thunder as frames advanced. This verifies existing event capture,
not the new transport implementation.

After the user's build:

1. Follow timeline while stopped: wind/rain silent. Play: automatic sources react
   to current climate/cloud/domain data; enabled thunder counter increases.
2. Pause active sources and pending thunder. Two external audio.status polls a few
   wall seconds apart must show unchanged PCM cursors and remaining delay, and
   timeline_held=true. Resume: cursors/delay continue with no duplicate events.
3. Publish a short signal while paused: TTL holds until scene time advances. A
   paused domain must not refresh its telemetry signal.
4. Stop clears voices/queue even at start frame. New Play re-arms. Backward seek
   and loop discard old thunder and domain controls; new pass emits once.
5. Continuous preview permits ambience during timeline pause and wall-time thunder
   delays. Mute and asset audition remain distinct from timeline Pause.
6. Save/load/undo/redo mode; test invalid/missing mode and unknown params. Test
   paused asset audition and source Play/Pause overrides.

Probes use external scripts/test/rt_ipc.py, never the embedded script workspace.
No new test helper script is required.

## Default and legacy migration

Follow timeline is now the default for new scenes and legacy audio snapshots.
playback_mode_explicit distinguishes an intentional saved preference from the
previous implicit continuous default. Old snapshots without this flag load in
timeline mode. Choosing a mode via UI/script/IPC saves the flag, so explicitly
chosen continuous preview remains portable. Scene opening starts paused.
Audio reads TimelineWidget.isPlaying directly and the Play/Pause button syncs the
service immediately, instead of relying on the delayed render-settings mirror.
After the next user build: open a legacy scene, confirm timeline mode and silence;
Play, Pause, Resume and Stop; save/reopen and confirm the explicit selection.
