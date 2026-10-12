# Scene audio: lightning and delayed thunder

Implemented source contract, 2026-10-12. Build and live listening acceptance are
pending the user's build. No application launch, project/shader compilation or
new test script was performed by Codex.

## Shared UI, script and IPC operations

All surfaces use `AudioAuthoring` -> `AudioService`.

| IPC method / `rt.audio` function | Parameters | Result |
| --- | --- | --- |
| `thunder_setup` | optional `assets_root` | Atomically import CC0 pack and enable near/far groups. Reuse/re-arm existing complete bindings. |
| `thunder_update` | required `config` object | Merge, validate, stop old thunder, clear queue and re-arm. |
| `thunder_event` | required `key`, `position`; optional `intensity=30`, `age_seconds=0` | Runtime event using active camera; `queued` plus details or skip reason. |
| `list`, `status` | none | Persistent `thunder` settings; pending events/delay, voices, drop count and last decoder/device error. |

Setup/update are undoable, mark the project modified, and require SceneWrite +
FilesRead. Event submission also requires both because it schedules future file
decoding; it is not saved/undoable. Invalid/unknown parameters, nonfinite numbers,
missing ids and stereo bindings fail via existing `audio_error` IPC/script errors.
Skip reasons: `duplicate`, `inaudible`, `disabled_or_suspended`,
`asset_preview_active`, `queue_full`.

```python
rt.audio.thunder_setup()
rt.audio.thunder_update(config={"gain": 0.8, "sound_speed_mps": 343.0})
rt.audio.thunder_event(key="manual/strike/1", position=[1000, 0, 0], intensity=30)
rt.audio.status()
```

```json
{"method":"audio.thunder_event","params":{"key":"manual/strike/1","position":[1000,0,0],"intensity":30}}
```

Position is absolute world metres, not camera-relative. UI's 1 km test uses
camera + [1000,0,0]. Script/IPC can supply custom imported mono asset ids.

## Persisted configuration

| Property | Default | Validation |
| --- | --- | --- |
| `enabled` | false | Boolean; true requires both variation groups. |
| `gain` | 1 | Finite 0..4. |
| `sound_speed_mps` | 343 | Finite 250..450; adjustable constant-speed approximation. |
| `reference_distance_m` | 500 | Finite 1..100000; minimum attenuation distance. |
| `max_distance_m` | 30000 | Finite 1..100000, >= reference distance; skip farther events. |
| `crossover_m` | 3000 | Finite 1..100000; selects near/far group. |
| `near_assets`, `far_assets` | [] | Up to 16 existing mono asset ids per group. |

Audio schema 1 has an optional backward-compatible `thunder` field. Older scenes
restore disabled defaults. Project save uses the existing portable audio file
copy path and retains asset provenance. Missing files keep ids/bindings and report
runtime errors. Referenced assets cannot be removed/relinked to stereo; clear
their array bindings first. Undo/load resets transient events and voices.

## Event and timing contract

Main-thread `AudioThunderBridge` observes timeline frame/FPS, the same clock used
by Main.cpp for cloud lightning. It is independent of RT samples. Camera WEATHER
preview time is a separate clock and is not used for lightning.

Only the four-lane onset clock is mirrored from `CloudLightningParams.h`: effective
period, seed hash, jitter and duration multiplier. All storm eligibility, cell
selection and bolt geometry come from `makeSharedCloudParamsGPU`, sampled halfway
through the flash. Crossed onsets are enumerated, including flashes that ended
between frames. No second storm simulation or geometry/particle scan is added.
Update this adapter if the producer's onset clock changes. Other agent's cloud
and storm files were not modified.

Key = weather seed / lane / cycle. Closest point on the trunk approximates the
first audible wavefront; its onset position is frozen after removing wind drift.
Intensity amplitude = `gain * min(4, sqrt(intensity / 30))`. Current camera drives
inverse-distance attenuation and pan while playing. Distance/closest point use
camera at detection; historical camera interpolation, branch wavefronts, terrain
occlusion and atmospheric refraction are not implemented. Physical context:
[NOAA sound of thunder](https://www.noaa.gov/jetstream/lightning/sound-of-thunder).

Delay = distance / sound speed, minus elapsed visual age since onset. At 343 m/s,
1 km is ~2.915 s, 5 km ~14.577 s. After submission, a monotonic real-time queue
owns arrival. Later camera movement changes pan/attenuation but does not move
the scheduled arrival. A recording's remaining attack/decay remains audible.
Distinct near/far assets use identity hash selection, immediate-repeat avoidance
when possible, and small pitch variation (0.94..1.06). Sound duration/decay and
arrival survive the flash's visual lifetime; thunder never loops.

Pause emits no new events but pending thunder still arrives. Backward time,
stopped-timeline scrubbing, forward gaps over 2 s, seed/rate changes and scene
generation changes clear thunder. Playback advances <=2 s enumerate onsets;
smaller forward seeks during active playback are indistinguishable from advances
with the current timeline API. Timeline looping permits replayed events without
old tails. Render-output jobs suppress new automatic events. Offline audio export
and a shared offline event evaluator remain later-stage work.

Stop All and asset audition clear voices/queue and suspend automatic thunder;
setup/update re-arm. Mute preserves queue/playback. Limits: 64 pending, 128 recent
keys, 8 thunder voices, 32 total voices, existing 512 MiB decoded budget. At most
two queued voices decode per editor update; low editor FPS can delay their start.
Full queues increment a drop counter. Voice-budget/file/device failures drop the
event and retain `thunder_error` until setup/update clears it.

## Pack and acceptance after the user's build

Seven mono 44.1 kHz PCM16 derivatives in `assets/audio/cc0_thunder`, with per-file
sidecars, source/download URLs, author, CC0 dedication and hashes. Publisher says
AI/procedurally generated, not field recordings. [CC0 license](https://sfxmint.com/license)
allows commercial use/modification/redistribution. Full WAV decode, nonzero PCM,
peak/RMS and SHA256 were checked; subjective quality requires scene audition.
VS now copies the full assets/audio tree; CMake already copies all assets.

1. User builds Studio. Scene Audio > Lightning / Thunder > Connect. Confirm seven
   assets, enabled near/far arrays, and repeated Connect reusing ids. Choose the
   Assets folder if the default runtime root differs.
2. Unmute; UI 1 km test should arrive after ~2.9 s. Inspect pending distance, delay,
   variation and remaining time via `audio.status`. External IPC repeated same key
   must report duplicate. Submit 5 km from camera for ~14.6 s delay.
3. Enable a storm, cloud layer and lightning in existing atmosphere UI, then play
   the timeline. Verify one event per flash, near/far pan/level and tails after
   flash disappearance. Compare Raster and RT playback.
4. Pause after distant strike: pending sound arrives. Stop All cancels it; Connect
   re-arms. Backward seek/stopped scrub clears old pending events. A timeline loop
   replays events once per new pass without retaining the previous pass's tails.
5. Save/load .rtp and undo/redo config edit. Settings/ids persist; queue does not.
   Check errors: stereo binding, missing id, invalid speed, unknown config key,
   referenced asset removal. Relink a missing mono WAV and re-arm.
6. Audition all seven variations in scene. IPC probes use external
   `scripts/test/rt_ipc.py`, never the embedded script workspace. No new test script.
