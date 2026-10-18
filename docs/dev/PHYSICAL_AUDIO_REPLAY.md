# Sample-scheduled native physical history replay

Source continuation, 2026-10-18. No Studio build, launch or live test. Unified
domain code/tests and application state are untouched. This is a native replay
route for validated authored/sampled histories, not automatic solver event cache,
full acoustic export parity or final acceptance.

## One evaluator

PhysicalAudioSchedule compiles the validated timeline into <=4096 typed controls
and <=4096 typed energy events outside the callback. It owns initial/current
fixed-size DSP states and prepared vectors. Both renderPhysicalTimeline (bake)
and native replay call its render method. Control/reset/paused transitions and
impacts are applied at their exact integer sample frame, independent of caller
block size. Same-frame excitation coalescing matches the shared DSP model.

No JSON reads, construction, allocation, locking, file access or geometry queries
occur in schedule render. Reset copies fixed DSP state. Once transferred to a
native source, only the callback reads/writes its rendering state. Main-thread
diagnostics use published atomic cursor/end/peak values and immutable frame count.
Prepared arrays are destroyed only after native sound uninitialization.

Offline bake multiplies authored source gain after schedule rendering. Native
sound applies current authored gain through the existing graph; schedule gain
already represents reaction controls, so live reaction gain is bypassed during
replay. Unit source/reaction pitch is required. Authored spatialization, bus routing
and optional existing acoustics remain live; those stages are outside dry schedule
equivalence. Playback on a device is not a bit-identical PCM comparison target.

## Shared Python/IPC service

| Operation | Parameters / result |
|---|---|
| physical_replay_start | id, duration_seconds=2, controls=[], events=[], start_frame=0 |
| physical_replay_status | No params; worker state and active scheduled-source diagnostics |
| physical_replay_apply | No params; consume current prepared schedule and start/hold native source |
| physical_replay_cancel | No params; discard pending/completed preparation, leaving active playback alone |

All operations are the same service in rt.audio and external audio.* IPC. Status
requires Read; other replay operations SceneWrite. No files or scene authoring
records are created. Existing play/pause/stop/all operations control active native
replay; use stop to release it, cancel to discard preparation.

The source must be procedural, unit pitch and nonresident before start. Histories
follow the bake validation contract in [PHYSICAL_AUDIO_BAKE.md](PHYSICAL_AUDIO_BAKE.md),
including 0.05..60 seconds, sample-frame ranges, 4096-row budgets and 32-coincident
impacts. start_frame is an integer in [0, output_frames). Bad requests reject before
replacing the current job. Preparing a replay does not itself start playback or
enable telemetry/simulation.

```python
rt.audio.stop(id=source_id)
rt.audio.physical_replay_start(
    id=source_id, duration_seconds=2, start_frame=24000,
    events=[{"frame": 0, "sequence": 1, "energy_j": 0.1},
            {"frame": 48000, "sequence": 2, "energy_j": 0.05}])
# Poll physical_replay_status from an external process. After completed:
rt.audio.physical_replay_apply()
```

A bounded AudioDesignJobs worker constructs the schedule and prerolls from reset
to start_frame with cancellation checks every <=1024 frames. This reconstructs
modal/noise/filter state from prior controls/events, rather than starting a fresh
tone at the seek point. No long seek work runs on the main thread or callback.
One active and one replaceable pending preparation are allowed. Stop All, scene
clear/restore and timeline discontinuity cancel through the runtime's clear path.

Apply checks scene generation, the full authored source snapshot and nonresidency
again. Changed/removed sources return stale and discard the prepared result. It
transfers ownership before using existing play, so a paused timeline holds the
prepared source. Native start failure cleans up the new source. Individual source
stop does not implicitly cancel a worker job; explicit later apply is still an
intentional playback request if source/generation checks pass.

## End, editing and capture

At schedule end the data source supplies silence rather than immediately detaching
the acoustic node. The existing 1.25 s drain allowance then releases the voice.
Automatic source bindings are suspended on scheduled completion so live telemetry
cannot immediately restart the source as unscheduled sound. Explicit play/resume
automatic can return to the source's usual live-control behavior after retirement.

Pause preserves schedule/DSP position; source stop/remove/recipe change discards
it. Live impact injection and nonunit pitch edits reject while scheduled replay
is active. Gain, pose, routing and other compatible live authoring retain their
existing semantics. source_runtime.generator_runtime includes playback_input
(scheduled_history/live_controls), history_frame, history_frames and history_ended.
The existing effective_gain diagnostic still describes the reaction target, not
the recorded schedule multiplier; playback_input identifies that boundary.

Replay is session-only. It does not add a persistent source/scene timeline binding,
automatic scene-time seeking, looping, material/pose history or solver substep
capture. A different seek is requested by stopping and preparing at a new
start_frame. Sampling an active replay with physical_history fails explicitly;
the original history is already the authoritative bake/replay input. Existing
sampled takes still have sample_exact_capture=false.

## Deferred combined acceptance

The existing reactive acceptance script was extended without running it: worker
prepare versus explicit apply, seek cursor, scheduled-source diagnostics, live
impact/nonunit-pitch rejection and automatic end/drain. At the later user-build
gate additionally compare dry baked PCM with schedule rendering across different
block partitions and seek points; test paused/reset spans, same-frame impacts,
cancel/replacement, source changes in flight, timeline holds/seek invalidation,
Stop All, automatic binding completion and destruction. Native latency, device
quality and acoustic output still require combined measurement/listening.
