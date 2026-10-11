# RT frame timing and surface guiding rollback

The user reported increased noise and GPU cost after compiling the box-guided
diffuse continuation experiment. It is removed from the shared BSDF and project.
Normal cosine diffuse sampling is restored. The experimental helper/test are
removed and its original note is marked rolled back. Earlier absorption parity
and live-only smoke emitter-list fixes are preserved.

Read-only external named-pipe inspection confirmed the open scene has the
`Nuclear Detonation (Cinematic) #1` system and Vulkan Rendered mode. At inspection
the viewport was already converged at 128 samples. Requesting one more viewport
frame reported zero newly rendered samples, so it provided no fresh RT timing.
Background loop timings and cached raster telemetry must not be mistaken for
current RT rendering performance. No restart, cache clearing or material edits
were performed during inspection.

## FPS correction

The RT/CPU sample blocks wrote their short render-call duration into the full-frame
metrics and window title. The actual presentation-period correction applied only
to Solid/Matcap/RayFusion. Once sample accumulation stopped, the RT title could
retain its last sample-call FPS indefinitely.

`ViewportPresentationStats` now owns frame-time/FPS updates after SDL presentation
for every viewport mode. The period includes simulation, rendering, display work,
UI, presentation and foreground throttle between displayed frames. Sample timing
still drives sample/ETA statistics and cannot overwrite frame metrics.

Frame duration is smoothed with a 500 ms exponential time constant and FPS is its
reciprocal. This avoids independently averaging FPS and milliseconds, which can
give contradictory values. Hidden/skipped presentation resets the interval so
background idle time is not counted as a render frame on focus return. Mode
changes also reset the measurement. The first presentation seeds the clock; the
second provides the first measured period. Title updates are limited to 10 Hz.

The existing script/IPC `viewport.frame_telemetry` display period is supplied by
the same presentation measurement. This is displayed-frame cadence, not isolated
GPU kernel time or simulation timeline FPS. It is not a completed-sample counter
for asynchronous render submissions.

## Validation and user checklist

- `python scripts/test/check_viewport_presentation_stats.py`: non-build audit of
  the single presentation publisher, removed sample writes, reciprocal FPS,
  project registration and disconnected guiding code.
- No project build, shader compilation or application launch was run by Codex.
- Recompile shaders to consume the BSDF rollback (`closesthit`,
  `sphere_closesthit`, `volume_closesthit`). The running app retains the compiled
  experiment until those outputs are loaded.
- Build the application normally to include `ViewportPresentationStats.cpp` and
  the updated Main wiring. A shader-only build cannot fix the FPS display.
- In the same scene and camera, compare noise and frame cost against the guided
  version. This restores baseline noise; it is not a new fire-light estimator.
- During active RT playback or camera movement, check that 200 ms/frame corresponds
  to about 5 FPS and 500 ms/frame to about 2 FPS. Compare the existing external IPC
  `viewport.frame_telemetry` display period over several fresh displayed frames.
- Check CPU, OptiX, Solid and RayFusion, mode transitions and alt-tab/minimize return.
  Confirm the first returning frame does not include the time spent in background.
