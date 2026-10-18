# Shared physical DSP, sampled history and deterministic dry bake

Source continuation, 2026-10-18. No Studio build/launch, shader compilation or live
test was performed. The user requested code development to continue while another
agent tests unified domains; no unified-domain solver or test was edited here.
UI remains deferred. This closes a bounded dry-generator route into existing
assets/clips/export, not full acoustic/simulation replay or whole-system acceptance.

## Shared processing

Subsequent source continuation: [native scheduled replay](PHYSICAL_AUDIO_REPLAY.md)
now shares a compiled typed PhysicalAudioSchedule with the offline evaluator.
Seek/preroll prepares on a worker before explicit native apply. This supersedes
the missing explicit sample-scheduled replay route below, not the remaining
automatic solver cache, scene-time binding or full acoustic parity gates.

PhysicalAudioDsp owns seeded flow/friction/rolling generation and impact modal
state. The existing native data source and offline evaluator call this same core.
No miniaudio device, scene object, JSON parsing, allocation, file access or lock
enters its render loop. Recipe construction remains outside the callback. For an
identical recipe and sample-timed controls/events, this core is deterministic on
the same implementation/platform. Cross-platform bit-identical floating point is
not promised. The unscheduled live impact API still coalesces at callback boundaries.

The offline evaluator divides output at every control/event frame, with internal
blocks <=1024 frames. These boundaries cannot move event onset. It starts from
reset state; paused segments output silence without advancing DSP, and explicit
reset controls reconstruct initial state. No endpoint normalization/fade, clipping
limiter, propagation, source pitch resampling or device-output capture is inserted.
Source gain and recorded gain controls multiply the raw generator samples.

## Explicit bake operations

Python rt.audio and IPC audio.* expose the same service operations:

| Operation | Contract |
|---|---|
| physical_bake_start | id OR scope/ids; absolute new .wav path; optional duration_seconds, controls, events and sample range |
| physical_bake_status | No params; bounded worker queued/running/completed/failed/cancelled/consumed state |
| physical_bake_cancel | No params; invalidate ownership and discard pending/completed output |
| physical_bake_apply | No params; consume the completed current result, check source ownership, publish/import |

### Group output and exact sample ranges

`physical_bake_start` accepts the original `id` or scoped selection, never both.
`scope="selected"` requires one ID; `group` requires 1..32 unique IDs; `scene`
omits IDs and resolves all emitters. The total must be 1..32, and every resolved
source must be procedural with unit source/reaction pitch. Asset sources reject
the whole request rather than disappearing from the output. No telemetry or live
voice is needed. All histories are validated before replacing the pending job.

If duration/controls/events are all omitted, each source uses its own stored
history. An explicit history field selects the same explicit request for every
source, without merging stored fields. Different history lengths are supported:
the output horizon is the longest history and shorter sources contribute silence
after their end. Histories share sample zero; they are not scene timeline clips.

`start_frame` defaults to zero and `end_frame_exclusive` to that horizon. Both are
integer **48 kHz sample indices**, not video frame indices. The range must be
nonempty and within the horizon. DSP preroll preserves resonant/filter/random
state, so a range equals the corresponding samples of a full bake. The worker
renders in stable source-ID order using one output buffer, one schedule and a
1024-sample scratch block; cancellation is checked during preroll and rendering.

The float WAV sums source gain and recorded controls, with no normalization or
limiter. Peaks above 1 are possible and reported. Source mute/solo, bus/master,
spatialization and acoustic processing are excluded (`mixer_processing=false`,
`acoustic_processing=false`). This is a dry production intermediate that can be
placed into the existing asset/clip pipeline. It does not close scene export parity.

Apply validates every captured source before publishing one file/importing one
asset. A removed or edited member makes the entire result stale and its temporary
output is released. Provenance records all timelines and the exact sample range.
Adding an unrelated source after a scene-scope start does not expand that snapshot.

```python
rt.audio.physical_bake_start(scope="group", ids=[water_id, creak_id],
    path="E:/audio/physical_mix.wav", start_frame=2400, end_frame_exclusive=48000)
# Poll physical_bake_status, then explicitly physical_bake_apply.
```

Source-only checkpoint: existing acceptance script extended for permutation
stability, summed gain, exact crop, unequal lengths, invalid request atomicity and
stale group rejection. No build or live test was run.

Controls are <=4096 rows: frame (integer sample index), activity (0..1, default 0),
gain (0..16, default 1), paused and reset (booleans, default false). Frames strictly
increase and start at zero when controls exist. Continuous sources require controls;
impact controls must use zero activity. Gain is the reaction/output multiplier,
additional to the authored source gain, not a dB value.

Events are <=4096 impact-only rows: frame, sequence and energy_j. Frames are
nondecreasing, sequence strictly increases within 0..2^53-1, and at most 32 events
share a frame. Energy validation and reference normalization match physical_impact.
Same-frame amplitudes coalesce and saturate at 1, as in the native excitation ring.
Events in paused regions reject before changing job ownership. Every frame belongs
to [0, round(duration_seconds*48000)); no event/control at the excluded end is valid.
Pitch and reaction pitch range must all equal 1; unsupported pitch rejects rather
than silently baking a different sound. Unknown fields/types/order/ranges reject.

```python
job = rt.audio.physical_bake_start(
    id=source_id, path="E:/AudioOutput/impact.wav", duration_seconds=2,
    events=[{"frame": 0, "sequence": 1, "energy_j": 0.05},
            {"frame": 24000, "sequence": 2, "energy_j": 0.02}])
# Poll from an external process; never block the embedded workspace waiting for UI work.
rt.audio.physical_bake_status()
# After completed, explicitly publish and import the generated asset:
result = rt.audio.physical_bake_apply()
```

One reusable AudioDesignJobs worker has one active and one replaceable pending task.
Worker snapshots contain authored source data and validated history only. Rendering
and WAV writing run off the UI thread with cancellation checks every <=1024 samples.
Maximum PCM is 60*48000 mono floats (~11 MiB); one completed result owns a temporary
WAV, not another retained PCM copy. The worker cannot import assets or play sound.

Output is float32 mono 48 kHz WAV. A uniquely reserved neighboring temporary file
is worker-owned and deleted on cancellation, failure, replacement, stale apply or
service shutdown. AudioDesignJobs results have an optional RAII lifetime attachment
for this cleanup; existing Designer/Arrange results leave it empty. Explicit apply
checks scene generation and the complete authored source snapshot before publishing
with no-replace MoveFileExW. A competing final file is never overwritten. Failed
asset import removes the file just published by this operation. Output publication
currently requires Windows, consistent with the existing synthesis writer.

Apply creates an ordinary asset with the validated timeline/recipe provenance in
source metadata. Existing clips, mixer, project asset copying and offline export
can use it. Apply is the undoable authoring step; undo does not remove the authored
file, matching existing generated assets. Changing/removing the source or seeking
before apply returns stale; clear/restore cancel the worker. Regular play/stop does
not cancel an unrelated bake. Capability is SceneWrite|FilesRead|FilesWrite for
start/apply/cancel (temporary cleanup included); status is Read.

The report exposes frames, peak/RMS, a canonical little-endian PCM FNV-1a hash,
timeline, limited=false and acoustic_processing=false. Hash is a deterministic
comparison aid, not a cryptographic integrity claim. Authored gain can produce
float samples above unity; this path does not silently normalize physical history.

## Sampled take operations

physical_history_start(id, duration_seconds=60) starts one transient take per
AudioService, with 0.05..60 seconds maximum and 4096 controls/events each. The source
must first be stopped/nonresident so the take starts from known DSP state; unit
pitch is required. Existing recordings are replaced only after input validation.
No physics collection or simulation is enabled by this operation.

physical_history_status is a read-only diagnostic. physical_history_cancel discards
the take. physical_history_finish() returns history parameters suitable for
physical_bake_start; adding path starts that worker directly. Finish with a bad
path leaves the take available for retry. History start/cancel require SceneWrite;
status Read; finish conservatively requires file capabilities because it may bake.

The take samples shared runtime activity, reaction gain, pause and residency/reset
edges at main-thread audio ticks, recording only changed controls. Drain state uses
zero excitation and the held gain. Accepted physical_impact calls are recorded;
duplicate/rejected calls are absent. Timeline mode uses transport time; continuous
mode uses monotonic wall time. Held impacts wait until playback resumes before
receiving a take frame. Source edits, generation/mode changes, backwards time,
invalid event epochs and capacity overflow explicitly fail the take. Missing held
sources fail rather than silently replaying stopped events. End-boundary samples
are excluded; finish uses at least 2400 frames (50 ms). Limits stop collection.

```python
rt.audio.stop(id=source_id)
rt.audio.physical_history_start(id=source_id, duration_seconds=10)
# Resume a bound continuous source, or submit explicit physical_impact events.
# After the intended interval:
take = rt.audio.physical_history_finish(path="E:/AudioOutput/take.wav")
# Poll physical_bake_status externally, then physical_bake_apply.
```

This is sampled control reconstruction, with sample_exact_capture=false. It does
not snapshot device resampler/filter phase, recover skipped simulation substeps,
or reproduce the callback's block arrival times. Very short pause/restart edges
between ticks can be missed. It is not speaker PCM recording or a deterministic
solver event cache. Existing asset-envelope capture_start still rejects procedural
sources; the new explicit history route supersedes that missing dry-bake route only.

## Remaining closure and combined gate

Still open: automatic solver contact identity/timestamps/material provenance,
substep-complete event/control cache, sample-scheduled native playback, phase/pitch
checkpoint/seek, full acoustic filtering/reflection export, multi-source spatial
mix capture, and authored history project records beyond baked asset provenance.
Source recipe preservation and shared dry DSP do not close those gates.

The existing reactive acceptance script was extended, not run: deterministic
repeat bake hash/frame count, explicit apply, malformed-request preservation,
source-stale rejection, temporary cleanup and sampled-history-to-asset flow.
At the later combined user-build gate also check cancellation/replacement in flight,
scene/seek/destruction cleanup, paused/reset histories, 4096-row/32-coincident limits,
missing output directories/files racing apply, unit-pitch rejection, save/reload,
undo and inserting a baked asset into the existing offline export path. No new
test script or native/live test round is requested at this checkpoint.
