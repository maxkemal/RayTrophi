# Scene audio: stage 1 implementation

> **Durum:** AKTIF — sahne sesi 1. aşama uygulaması (miniaudio, `audio.*` IPC); derleme ve işitsel kabul bekliyor

2026-10-11. Source implementation added; build and visual/aural acceptance are pending.
This is the first implementation of stage 1 of SCENE_AUDIO_RUNTIME_PLAN_2026_10_11.md.
EXE export, timeline clips, waveform, mixer controls, weather events and offline export
are later stages. No build, application launch or generated test script was performed.

## Editor

Add > Audio Source creates an unbound non-geometric scene object. Scene hierarchy has
an Audio Sources group. The existing Scene properties dock contains a Scene Audio
section for imports, asset preview/relink/removal, source properties and playback.
The bundled CC0 starter files live in RayTrophiStudio/assets/audio/cc0_starter with
per-file metadata, provenance hashes and the license text. Assets can append these
directly; see the pack README for mono/stereo variants and loop suitability.
Import and Relink open the existing native file chooser with WAV/FLAC/MP3 filters;
canceling leaves the scene unchanged. The selected path is shown read-only.
Assets discovers WAV/FLAC/MP3, displays an AUDIO badge and metadata, supports audition
and Append To Scene / viewport drop. Existing license/source/author/tags are retained
on append. Audition does not add an asset to the scene.

Select an icon or hierarchy entry; use the existing translate/rotate modes to move
the source or aim its direction. Distance rings and inner/outer cone guides display
the selected source's attenuation settings. Delete/X and Duplicate use scene history.
Audio is independent of the chosen renderer and continues on miniaudio's device thread.
The active scene camera is the listener; one scene unit equals one audio distance unit.

Parent is an existing flat mesh/spline node name. Position/direction are local when
parent is set, world coordinates otherwise. The existing canonical world transform
API includes the rigid-body delta. Setting/changing a parent keeps the authored local
values. The gizmo converts world edits back to local values. Missing/singular parents
remain repairable in properties and stop their voices; playback returns an error.
No simulation-domain implementation is changed by this feature.

Numeric text properties commit with Enter; gizmo drags produce one undo record.
Undo/redo restores authoring state and stops playback; voices never autoplay on load.
Deleting a referenced audio asset fails until sources have been unbound/deleted.

## Shared script and IPC operations

Continuous wind/rain control and external domain-contact signal inputs are described
in [SCENE_AUDIO_REACTIONS.md](SCENE_AUDIO_REACTIONS.md). Sources accept an optional
`reaction` configuration; signal_set/signals_clear are transient runtime operations.

Python uses `rt.audio.<operation>(**keyword_parameters)`. IPC uses
`audio.<operation>` with the same parameter object. UI, Python and IPC route through
AudioAuthoring and AudioService; asset append is one atomic authoring operation.

| Operation | Parameters | Result / behavior |
|---|---|---|
| list | none | version, next_id, assets[], emitters[] |
| get | id | emitter record |
| select | id | emitter record; sets the editor selection |
| status | none | device, sample_rate, preview_muted, voices, voice_limit, resolved_poses |
| asset_import | path; optional license, source, author, tags[] | persistent asset record |
| asset_relink | id, path; optional provenance/tags | retains asset id; stops voices |
| asset_remove | id | removes an unreferenced asset; undoable |
| append | path; optional name, position, provenance/tags | imports + creates source atomically |
| create | optional emitter properties below | allocates a persistent source id |
| update | id and changed emitter properties | validates complete proposed state first |
| duplicate | id | copies source settings and asset binding; new source id |
| remove | id | stops/deletes source; undoable |
| play | id | starts, resumes paused cursor, or restarts a completed sound |
| pause | id | freezes the voice cursor |
| stop | id | releases the voice; next play starts at zero |
| stop_all | none | releases all source and audition voices |
| preview | id (asset id) | single asset audition |
| preview_file | path | single file audition without changing authoring data |
| preview_stop | none | releases audition voice |
| preview_mute | muted (bool) | silences device output; cursors continue |

Emitter properties: name, asset_id (empty is unbound), parent (empty is world),
position[3], direction[3] (nonzero), gain (0..16), pitch (0.125..8), loop, spatial,
min_distance (0.001..1000000), max_distance (>= min_distance), rolloff (0..10),
cone_inner (0..360 degrees), cone_outer (>= inner, <=360), cone_gain (0..1),
bus (Master / Weather / Effects / Ambience / Music). Bus is a routing label reserved
for stage 2; stage 1 plays through the master engine without bus gain/solo controls.
Unknown emitter properties, invalid ids, malformed vectors, NaN/Infinity and invalid
ranges fail before the source record changes. Mono is required for spatial sources;
stereo ambience uses spatial=False. Empty sources can be authored but cannot play.
When creating a bound source or changing its asset binding without an explicit
spatial parameter, the service selects mono spatial / stereo ambience automatically.
Explicit spatial=True with stereo still fails. UI shows a stereo ambience notice,
disables positional mode for a stereo binding, and highlights errors above controls.
An unbound source shows an explanation and its Play button is disabled.

```python
asset = rt.audio.asset_import(path=r"D:\Sounds\rain.flac", license="CC0")
source = rt.audio.create(name="Rain", asset_id=asset["id"], spatial=False, loop=True)
rt.audio.select(id=source["id"])
rt.audio.play(id=source["id"])
rt.audio.update(id=source["id"], gain=0.4)
rt.audio.pause(id=source["id"])
rt.audio.play(id=source["id"])  # resumes
rt.audio.stop(id=source["id"])
```

Python raises ValueError on service failures. IPC returns the existing error envelope
from `__error`, with code audio_error and the service message. Device failures include
device_unavailable and the miniaudio result code. File decode/timing/channel errors
require relink; no silent stale-metadata playback is accepted. Missing device/file
does not invalidate the scene or prevent image rendering. `status` does not initialize
a device; device initialization is lazy on playback and can be retried after failure.

IPC discovery registers all operations. list/get/status require Read; authoring and
playback control require SceneWrite. append/import/relink/play/preview/preview_file
also require FilesRead. Asset audition stops scene voices. Starting a source stops
audition. Preview mute is runtime state, distinct from authored gain, and is not saved.

## Persistence and bounded playback

The .rtp audio section is schema version 1, with stable asset/source ids and an
allocation counter. Validation completes before replacing audio state. Old projects
without an audio section load empty. Missing files retain their ids/settings and can
be relinked. Saving copies available audio to `<project_stem>_audio/<asset_id>.<ext>`
and writes relative paths; copy that directory with the project and its normal sidecars.
Save As writes a corresponding directory beside the new project. Source audio files
are never modified. A missing asset is marked missing in the saved manifest; its
unresolved path is retained for repair, so a project with missing files is not portable.

The vendored miniaudio engine runs at 48 kHz, stereo float mixing. Stage 1 synchronously
probes/decodes before playback, with no user callback performing file access, scene
locking or geometry work. Import limits are 600 seconds and 128 MiB of decoded 48 kHz
float samples per asset. Resident voices are capped at 32 and estimated decoded data
at 512 MiB. Finished one-shots are reclaimed before allocating another voice. Paused
voices retain their memory. Virtualization, asynchronous loading, device hot-swap and
long-file streaming are not implemented in this stage; large imports can pause the UI.

The only vendor change is miniaudio 0.11.23 from the official tagged distribution.
Source: https://github.com/mackron/miniaudio/tree/0.11.23
License file: RayTrophiStudio/external/miniaudio/LICENSE (included in project manifest).
Header SHA256: 7e4f3f13c8fe66df2080ac3dd12a89193e3c2463cb7f067c798abd7331cd8ee6.
No sound library files were downloaded or bundled.

## User build and acceptance checklist

Live IPC check, 2026-10-12: user-built Studio, initially empty scene. Twenty targeted
checks passed through the external scripts/test/rt_ipc.py client: mono/stereo WAV
metadata, source update, undo/redo, invalid distance/direction/property/id rejection,
referenced-asset removal rejection, positional-stereo rejection, device initialization,
play/pause/resume, preview mute with advancing playback, voice release on stop, stereo
ambience, audition exclusivity, duplication and undo of deletion. Temporary silent WAV
files and scene assets/sources were removed; final assets and emitters were empty,
voices stopped, mute disabled. The allocation counter advanced to 6 and authoring
history/modified state were affected by these checks. No test script was created.
These checks confirm device/service behavior, not audible quality, gizmo visuals,
parent movement, project portability, FLAC playback or remote-token permissions.
A subsequent stereo MP3 check decoded 44.1 kHz audio, bound it to the existing source
with spatial=false, and started playback. The user confirmed hearing it successfully.
The initial silence was an unbound source (empty asset_id), not a path encoding failure.

1. Build the existing VS x64 configuration. New Scene, UI and API translation units
   are registered in vcxproj/filters; CMake already discovers them through existing globs.
2. Import a short mono WAV, FLAC and MP3; inspect duration/channels/sample rate.
   Create/play a source; move and rotate its gizmo while listening. Verify gain,
   pitch, loop, cone and distance response in RT, RayFusion, material and solid views.
3. Import stereo ambience, append/drop it from Assets and confirm spatial=false.
   Verify positional stereo rejection, audition/scene exclusivity, pause/resume,
   stop/restart, and mute while cursors continue. No device should block image rendering.
4. Duplicate/delete/transform and undo/redo through UI and through the operations above.
   Check selection, hierarchy and source settings; undo/redo intentionally stops voices.
5. Parent a source to a moving flat object; check world gizmo/local properties. Delete
   the parent and confirm safe stopped playback, an invalid resolved pose, and repair.
6. Save/reopen and Save As; move the project plus audio directory to a new folder.
   Confirm ids, provenance and settings persist and playback resolves relative paths.
   Rename an audio file, reopen, then relink it without losing the source id/settings.
7. Use scripts/test/rt_ipc.py from a separate terminal with Studio already open to
   exercise audio.list/get/status/import/create/update/select/play/remove. Verify parity
   with Python and UI, and reject invalid ids, NaN, reversed distance/cone ranges,
   malformed files and over-budget inputs. Named-pipe error 5 requires sandbox escalation.
8. Verify Read-only IPC credentials cannot mutate audio or read audio files through
   audition, and SceneWrite without FilesRead cannot import/relink/play/audition.

Remaining checklist items are pending manual checks. Stage 1 acceptance stays
open until the user completes the build and visual/aural verification.
