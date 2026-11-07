# SVGF: SDF/fog history and project preference

2026-11-07. User reported moving SDF/fog ghost trails while moving solid meshes
looked correct, and requested default-off plus project serialization.
Source implemented; user build and live acceptance remain pending. No C++ tests
were added and no application/shader compilation was run by the agent.

## History correction

SurfaceSDF wrote primary albedo/normal values but never set `PL_PRIMARY_DONE`.
Raygen therefore could treat a visible reconstructed surface as a miss in its
motion record. The surface now publishes a primary-hit record and dominant
composition material ID (or the built-in sentinel) at the actual iso crossing.
The payload layout is unchanged; a volume-depth flag is not used for this surface.

Raygen marks primary procedural hits without a triangle-instance motion mapping
as fully reactive, as it already does for participating volumes. SDF deformation
cannot be reconstructed from a rigid mesh's previous transform. This is a
conservative history policy, not newly implemented fluid advection motion vectors.
It also covers other procedural hits without such a mapping; ordinary mesh
motion-vector handling remains unchanged.

Fully reactive pixels now store zero temporal history length. Both SVGF tap
searches reject previous pixels with that marker, and the camera-stop bridge
requires positive previous history. Previously a newly exposed sky pixel could
accept a previous fog record because both use material 0xFFFF, keeping fog colour
after the fog moved away. The marker prevents that match without changing the
packed motion-record ABI. Current-frame colour is preserved.

The existing reactive-pixel spatial bypass is retained. SDF/fog without reliable
motion therefore trade temporal smoothing for avoiding stale colour; moving
volume/surface noise can remain higher than on well-tracked solid meshes. This
change does not redesign the separate raygen volume temporal stabilizer or FSR.

## Persistence and defaults

`RealtimeDenoiserSettings.enabled` now defaults false. New projects and older
projects lacking the new JSON block load with SVGF off. Explicit saved choices
are restored under `render_settings.realtime_denoiser`:

- enabled
- atrous_iterations (1..5)
- history_max (1..64)
- firefly_clip (0.1..10)
- still_noise (0..0.5)
- scene_motion

The focused `RealtimeDenoiserSettingsJson.h` helper validates types/ranges and
finite floats. ProjectManager only wires the helper. Loading enabled SVGF also
enables its required RT motion AOV; it does not change Halton jitter. GPU history,
statistics and debug_history are runtime-only and are not serialized.

UI, Python and IPC keep the existing `render.set_realtime_denoiser` service.
Persistent setting changes mark the project modified; debug-only changes do not.
Scene-motion policy changes invalidate denoiser history just like quality changes.
OIDN's own default/preference is not changed by this work.

## User build and IPC/manual acceptance

1. Build C++ and all relevant shader variants. Changed shader sources:
   `raygen.rgen`, `realtime_svgf.comp` (regular and RAW32), and the
   `volume_surface_shading.glsl` dependency of `volume_closesthit.rchit`
   (combined and split fog/surface variants). The main all-shader script emits
   these variants. Restart the application afterward.
2. New Project and an older project: `render.realtime_denoiser_stats` should
   report mode off. Enable through the UI or external named-pipe IPC, choose
   different history/iteration/noise settings, save the project, reopen and
   verify the same settings. Repeat after disabling. Debug view should not load.
3. With SVGF explicitly on, move an SDF object/fluid against a sky background,
   then fog against sky and opaque geometry. Check the trailing silhouette and
   newly exposed background while moving, and immediately after stopping.
4. Compare motion debug and SVGF history debug. SDF should have a primary-hit
   record; unsupported procedural/fog pixels should show no reusable history.
   Moving solid meshes should still accumulate valid temporal history.
5. Check raw accumulation and OIDN with SVGF off; final-render bypass must remain.
   Isolate FSR during the ghost test so its own temporal history is not confused
   with SVGF. Re-enable FSR afterward for the combined path acceptance.
6. Existing external API probe: `scripts/test/rt_probe_realtime_denoiser_ipc.py`
   validates setter errors and restores settings, but does not measure ghosting
   or perform project save/reopen. It now marks the project dirty while changing
   persistent settings; do not treat it as a read-only probe.

Project XML and scoped whitespace/source checks passed. No GPU execution or
serialization round-trip has been claimed before the user build.
