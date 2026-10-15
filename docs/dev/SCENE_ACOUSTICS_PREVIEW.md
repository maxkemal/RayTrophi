# Scene acoustics: first listening checkpoint

2026-10-15 source delivery. 2026-10-16: after the user built Studio and explicitly
requested measurements, external IPC acceptance passed in the running application.
[Raw report](acoustics_acceptance_20261016.json) records 13 cases, including actual
native mixer callback RMS/peak measurements of a 1000 Hz mono tone. One object gave
0.5 amplitude, two separate objects 0.25, zero transmission gave silence, and
bypass/disable restored 1. Unequal-band expected gain 0.48131763594 measured
0.48131764821 (absolute RMS-ratio error about 1.23e-8).

This measures engine PCM before the OS/device, not microphone SPL or a hardware
loopback. The existing default cube was preserved; test walls/source/asset were
removed and timeline playback mode restored. A derived test material remains
pinned by object-material undo history; history was intentionally preserved.
Playback explicit-mode flag/next_id can advance and the project is left modified.
No project compilation or application launch was performed by Codex.

Reproduce externally with Studio open and an audio-empty scene:
`python scripts/test/rt_test_acoustics_preview_ipc.py --report <report.json>`.
The script writes detailed evidence and performs cleanup in finally.

## Shared controls

Selected Audio Source -> **Scene acoustics / Direct path** -> **Audition scene
obstruction** enables a session-only broadband obstruction preview. It starts off,
is not saved with the project, is cleared by restore/undo/redo, and does not enable
domain simulation. The same service operations are available in Python and IPC:

```python
rt.audio.acoustics_preview(
    id="audio_source_1", enabled=True,
    meters_per_unit=1, boundary_mode="object_barrier")
rt.audio.acoustics_preview_status()
rt.audio.acoustics_preview(id="audio_source_1", enabled=False)
```

Use the actual source ID returned by audio.list. IPC names are
`audio.acoustics_preview` (SceneWrite) and `audio.acoustics_preview_status` (Read).
Status returns an array of enabled entries with settings, latest report, status
and amplitude_gain. Configure requires an existing ID and boolean enabled. Units
must be finite 1e-6..1e6. Unknown parameters are rejected by AudioService. Invalid
configuration leaves existing state unchanged. Python raises ValueError; IPC
returns the existing audio_error envelope. These runtime operations are not undoable.
Disabling restores the source's ordinary gain immediately.

Only spatial sources with manual reaction are eligible. At most four previews
may be enabled. Changing a source to nonspatial/reactive bypasses attenuation;
removing it drops its preview. Sound-file preview, arrangement clips and thunder
voices do not enter this direct-source preview path. Ordinary source gain, mixer,
spatial distance and cone attenuation remain in effect. Recorded source capture
samples include the current preview gain; offline authored export does not trace
scene geometry and does not reproduce this session preview.

The viewport refreshes all enabled paths approximately every 250 ms using bounded
immutable flat TriangleMesh/DNA snapshots (100000 triangles, 2048 objects, 128
crossings). Capture/BVH work is synchronous on the main thread in this first
checkpoint and can cause stalls on large scenes; start with one source and a small
scene. No geometry, JSON or UI enters the audio callback. Pose/material edits can
lag until the next viewport update. This is not a background solver or a hard
realtime update guarantee.

The live amplitude multiplier is sqrt(mean(six transmitted energy fractions)).
This is a declared equal-band broadband approximation, not spectral filtering.
Unknown acoustic profiles, ambiguous boundaries, missing camera, unsupported or
skipped geometry, and budget/validation failures bypass attenuation (gain 1) and
report a status/error. A clear covered segment also gives gain 1.

Boundary modes:

- **object_barrier**: one effective material assembly coefficient per crossed named
  object. Entry and exit do not square the loss. This deliberately assumes each
  named object is one barrier, even with multiple disconnected parts. Author walls
  as separate objects. It does not validate watertightness, compute thickness,
  handle nested media or solve inside/outside endpoint propagation.
- **thin_surface**: every deduplicated crossing is an independent thin boundary.
  Use single planes, not closed wall meshes, for this mode.

Travel delay is reported but not applied. Echo, reflections, diffraction, air loss,
frequency-dependent muffling and physical multilayer wall transmission remain
roadmap work. Built-in material names do not imply measured acoustic data.

## User build and scene acceptance

1. Build the updated project yourself. Confirm AcousticPreview.cpp is included and
   AudioService / RtAudioBindings / AudioAcousticUI compile and link.
2. Create a small scene with one looping **mono** Audio Source, spatial enabled,
   reaction Manual, and one flat cube wall between source and active camera.
   Keep the source and listener outside the wall; avoid edge/corner grazing.
3. Select wall -> Physical material -> Substance. Derive an editable test material
   if necessary. In the material editor's Acoustics group set acoustic_defined=true,
   all six absorption values to 0.2 and all six transmission values to 0.25.
   Absorption + transmission must not exceed 1 in any band. These are synthetic
   acceptance values, not a concrete/wood acoustic preset.
4. Select the source -> Scene acoustics / Direct path, choose **One barrier per
   object**, meters/unit matching scene scale, enable audition and play the source.
   With only the cube in the ray, expect evaluated_object_barrier and gain 0.5,
   despite the trace showing both entry and exit. Gain excludes ordinary distance
   and mixer attenuation. Use rolloff 0 for a level-only comparison if desired.
5. Move wall aside or camera around it: after the next update expect clear and gain
   1. Move it back: expect gain 0.5. Disable audition: ordinary output returns.
6. Put a second separately named wall with the same material in the path: expected
   gain 0.25. In thin-surface mode, one single plane gives gain 0.5. A cube is not
   a thin-surface acceptance asset.
7. Set acoustic_defined=false on a blocking material: expect unknown_material and
   bypass gain 1. Also check invalid unit input reports an error, and removing a
   source / loading another scene leaves no inherited preview attenuation.
8. Compare Python and external IPC configuration/status with the same UI settings.
   Use scripts/test/rt_ipc.py from a separate process while Studio is open; never
   run a named-pipe probe in the embedded script workspace. Do not infer listening
   success from a status response alone.
