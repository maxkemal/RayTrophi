# Live acoustic playback and concurrent sources

2026-10-17 follow-up source slice: [physics-driven source parity](SCENE_AUDIO_PHYSICS_SOURCES.md)
removes the manual-driver restriction, shares evaluated world pose/control state,
and collects moving GPU queries before submitting new poses. User C++ build and
live acceptance remain pending; earlier live reports do not validate that follow-up.

2026-10-17 source delivery; the user built Studio and confirmed concurrent audio.
The [echo-room lab checkpoint](ECHO_ROOMS_LAB.md) verifies live paths and sparse
dry/wet delayed native PCM. Full listening/performance acceptance remains open. Supersedes the
broadband-only playback description in SCENE_ACOUSTICS_PREVIEW.md. Previous live
acceptance reports cover the previous shader/PCM path, not this revision.

## Shared controls

Audio workspace **Play all sources** calls `rt.audio.play_all()` / IPC
`audio.play_all` (SceneWrite|FilesRead). No parameters. Sources without an asset
are skipped. Bound sources share the existing 32 resident voice budget with clips
and thunder and the 512 MiB decoded asset budget. New voices are decoded/prepared
before playback and scheduled at one common engine time, 50 ms ahead. Already
playing sources retain their phase. Continuous mode plays immediately; timeline
mode holds sources until the scene transport resumes. This does not change the
authored playback mode or turn on automatic simulation drivers.

Invalid bindings/parents and capacity overflow fail before starting new voices.
A preparation failure releases newly prepared voices; existing live voices stay
intact. Empty selection succeeds with an empty `ids` array. The result contains
`ok`, participating `ids`, and `transport_held`; inspect `audio.status().voices`
for actual playing/held state and cursor progression. A device/start failure can
still interrupt dispatch. File audition retains its existing exclusive behavior.

Selected source -> **Scene acoustics / Direct path** exposes these same options:

```python
rt.audio.acoustics_preview(
    id=source_id, enabled=True, spectral=True, reflections=True,
    meters_per_unit=1, boundary_mode="object_barrier")
rt.audio.play_all()
rt.audio.acoustics_preview_status()
rt.audio.stop_all()
```

IPC uses the same operation names with `audio.` prefix. `spectral` defaults true,
`reflections` defaults false; both require actual booleans. Invalid configuration
raises Python ValueError / IPC audio_error without modifying the preview entry.
Preview remains session-only and spatial/manual-only, with up to 32 sources per
GPU batch. Concurrent playback shares the 32 resident voice budget. All existing
unit/boundary validation and fallback semantics remain. Pending refresh preserves
the previous response; failed direct queries bypass direct filtering.

## Audible model and bounds

Direct transmission applies square-root energy gains to six complementary bands.
Five first-order low-pass splits at 176.78, 353.55, 707.11, 1414.21 and 2828.43 Hz
form telescoping differences; unity gains reconstruct the input. The outside bands
extend to DC/Nyquist. This is a broad, overlapping envelope approximation, not an
octave-band measurement filter or phase-accurate material transfer function.
Gains approach targets over roughly 10 ms. `spectral=false` supplies the historical
sqrt(mean(energy)) gain uniformly for comparisons. `amplitude_gain` remains that
diagnostic broadband value; `band_energy` carries the actual six spectral inputs.

The GPU samples the closest surface along six world axes from each source, mirrors
the source about each discovered plane, verifies an actual geometry hit at the
image-source point, and checks both segments. Duplicate coplanar candidates are
suppressed. A known material's **specular_reflection** energy, after scattering,
controls each accepted path. Any intervening surface rejects that reflected path;
transmission through reflection blockers is not approximated as a clear segment.
Missing candidates do not prove the absence of room reflections. No CPU geometry
scan/BVH, simulation activation or per-face Triangle source is introduced.

At most six taps per preview are audible, with excess path delay up to one second
at 343 m/s. Dry playback retains its historical zero propagation delay; each echo
is delayed relative to that dry arrival. Path distance corrects the authored inverse
distance attenuation relative to the dry voice. Source cone and stereo pose are
inherited from the direct voice: independent reflected arrival direction/directivity,
diffraction, air absorption, multiple bounces and late reverb remain open. Moving
tap delays interpolate; this is not calibrated physical Doppler. A stop/remove or
completed-voice reclamation can truncate tails. This is live audition, not final
room acoustics or calibrated SPL.

Stable per-voice miniaudio nodes process stereo PCM with lock-free scalar targets;
the callback performs no JSON, geometry work, allocation or file reads. Each node
reserves about 384 KiB of bounded delay storage. Silent taps skip their filters;
ordinary unfiltered sources use a PCM-copy bypass.
GPU instance/ray/result ABI sizes are now 96/48/352 bytes; surface-debug instances
are 160 bytes. Both acoustic shaders must be rebuilt together with C++.

Preview status includes `reflection_paths` (energy, distance_meters,
excess_delay_seconds), and the existing report/solver state. GPU summary readback
is 352 bytes per source. Authored export does not run this session solver. Existing
source-event capture still records broadband gain and does not reproduce spectral
filters or echoes; it must not be used as evidence of this live PCM path.

## User build and acceptance

1. From the repository root run `RayTrophiStudio\compile_shaders.bat` to recompile
   both `acoustic_segment.comp` and `acoustic_surface.comp`, then rebuild Studio
   with the usual Visual Studio configuration. New `AcousticPlayback.cpp/.h` are
   registered in the VS project/filters; CMake discovers Scene sources normally.
   Codex has not run shader compilation, a project build, or Studio.
2. Open a disposable geometry/audio-empty scene, unmute preview, and run externally:
   `python scripts/test/rt_test_acoustic_playback_ipc.py --sources 4 --report docs/dev/acoustic_playback_4.json`
3. Repeat with `--sources 16` and `--sources 32`, recording frame/GPU timings using
   the existing profiler. The probe records dispatch duration, active source
   cursors, mixed PCM, low/high tone filtering, reflected-path delay and PCM change.
   Dispatch duration includes preparation; it is not callback CPU or GPU duration.
4. Run the older direct-path regression externally:
   `python scripts/test/rt_test_acoustics_preview_ipc.py --report docs/dev/acoustic_direct_regression.json`.
   It now explicitly selects broadband mode and expects the revised GPU ABI.
5. Listen with speech/noise: compare spectral on/off, then enable reflections and
   move a broad wall from a valid mirror path to an occluded path. Compare Python,
   IPC and the controls above. Check timeline hold/resume, bus changes, source
   remove/rebind, stop-all, and project reload without inherited responses. Repeat
   on RayFusion and Vulkan RT. No audible/performance success is claimed yet.

The probes leave the project modified and preserve undo history; derived substances
can remain pinned by that history. Do not run IPC from the embedded script workspace.
# Next source slice: diffuse transport

The 2026-10-18 [diffuse transport continuation](SCENE_ACOUSTICS_DIFFUSE.md) adds
an opt-in first-bounce Lambert receiver-energy estimate and approximate playback.
It changes the GPU ABI to 128-byte instances, 752-byte summaries and 192-byte
surface instances; C++ and both shaders must be rebuilt together. The earlier
live acceptance below remains specular-only and does not validate this extension.
