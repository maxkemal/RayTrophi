# Single-bounce diffuse transport — source delivery, 2026-10-18

This is the first continuation of the energy-transport work, not a multi-bounce
solver or late reverb. No Studio/shader build, application launch or live probe
of this revision was performed by Codex. The previous fluid-room acceptance is
for specular-only playback; it does not validate this new path.

## Shared material budget

For each of six bands, the existing validated substance defines absorption A,
transmission T and scattering s. The material response already computes:

```
R = 1 - A - T
R_specular = R * (1 - s)
R_diffuse  = R * s
A + T + R_specular + R_diffuse = 1
```

The new solver consumes R_diffuse separately. It never adds total R on top of
R_specular, and does not reinterpret scattering as another absorption loss.
Library validation still rejects A+T>1. The GPU response upload also checks
finite, nonnegative coefficients and T+R_specular+R_diffuse <= 1 within roundoff.
This is conservation at a modeled surface; it does not mean that a point
receiver's summed response must be <=1, or that all emitted energy is captured
by this bounded first-bounce solver.

## Geometry and estimator

The source is an omnidirectional point emitter. Sixty-four deterministic
equal-solid-angle sphere directions (Fibonacci quadrature) find their first
authored-environment surface using the existing Vulkan TLAS. Simulation visual
slots are skipped without compacting TLAS indices. The actual transformed flat
triangle supplies the normal. A next-event ray tests surface-to-listener
visibility. Unknown first-hit material/invalid geometry and query-budget failures
are counted as rejected queries. An intervening surface rejects the listener
connection; transmission along reflected legs is not implemented.

Lambert's reflected-energy density is R_diffuse*cos(theta_out)/pi. Uniform source
sampling has PDF 1/(4*pi). Its surface measure is
cos(theta_in)*dArea/r_source^2, so the incoming cosine and source leg cancel in
the importance weight. The per-sample receiver response, normalized to the
unobstructed direct response, is:

```
E_relative = 4 * R_diffuse * cos(theta_out) * r_direct^2
             / (N * r_listener_leg^2)
```

There is no extra source-distance division after this conversion. N=64. Each
contribution belongs to one of eight bins over 0..1 second of **excess** delay.
Bins sum energy; path length/delay use an energy-weighted centroid. Six specular
candidate slots are retained separately, so diffuse sampling does not consume or
duplicate the six image-source candidates. Diffuse receiver energy is not clamped
to [0,1] like a material coefficient. Samples within 0.1 m of the point listener
are excluded and counted explicitly to bound the point-receiver singular region;
this near-field region is not solved.

This is finite deterministic quadrature, not an unbiased randomized estimator or
a convergence guarantee. Small surfaces and narrow visibility can be missed. A
complete query means no detected query/near-receiver rejection, not complete
physical coverage or numerical convergence. Omitted >1-second energy, higher
bounces, air loss, diffraction and wave phase are not redistributed into bins.

The Lambert high-frequency approximation and the separation of image-source early
paths from scattering transport follow the geometrical-acoustics approach
described in [the JASA overview](https://doi.org/10.1121/1.4926438) and
[Pyroomacoustics' hybrid model](https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.room.html).
No external library or dependency was added.

## Playback approximation

The existing callback now supports six specular plus eight diffuse taps, sharing
the one-second delay history and six complementary approximate bands. Diffuse
energy is already relative to free-field direct response, so playback does not
apply the specular total-path distance correction a second time. The source's
existing distance gain still scales the result: physically inverse-square
intensity is reproduced only where its authored distance law approximates 1/r
amplitude. Soft near-distance clamps, rolloff and maximum-distance clamps retain
their artistic semantics. Outputs are not calibrated SPL.

Diffuse bins have no physical phase. Fixed signed per-bin/per-band reconstruction
avoids treating every bin as a positive-phase copy; it is sparse energetic
auralization, not a wave solution, statistically independent binaural field,
or an exact PCM energy-conservation guarantee. Six overlapping filter bands and
centroid compression are approximations. No additional receiver/directivity,
late diffuse tail or phase interference is claimed. Capture/export still do not
reproduce the live spectral/echo path. Existing automatic stop can truncate tails.

All ray queries/JSON updates remain outside the PCM callback. Callback changes
use fixed arrays, lock-free coefficient atomics and existing history storage;
no geometry traversal, allocation, file access or locking was added there.
The callback computes its active tap list once per block; zero-target, fully
faded taps do not enter the per-sample smoothing/filter loops. This retains
fade-out processing and avoids paying the eight extra diffuse filter loops when
diffuse is disabled. Performance is still unmeasured.

## UI, Python and IPC

The contextual audio acoustics panel adds **Diffuse surface scattering**. Existing
`audio.acoustics_preview` accepts optional boolean `diffuse` (default false) through
the same shared service for UI, Python and IPC:

```python
rt.audio.acoustics_preview(
    id="audio_source_2", enabled=True, spectral=True,
    reflections=True, diffuse=True,
)
```

IPC uses the same method and keyword JSON. Wrong-type values reject before any
preview mutation. Diffuse enablement requires spatial, omnidirectional sources
(both cone angles 360 degrees); unsupported cones reject configuration. If a
source is edited to a directional cone later, the running preview reports
source_unavailable rather than silently applying isotropic transport. Disabling
the preview remains allowed. The source rule is shared by configuration,
submission and completion checks. Settings remain session-only; existing helpers
that omit diffuse retain the default false.

Preview `reflection_paths`/report `reflections` now identify `kind` as specular or
diffuse and `energy_reference` as material_fraction or
receiver_relative_to_free_field_direct. The report's `diffuse` object exposes
enabled, model, samples, contributions, rejected_queries, near_receiver_rejections,
complete, phase_resolved=false, receiver exclusion, delay bins and window.
Partial estimates are labeled in the UI. Existing transmission status continues
to describe the direct path; it is not a diffuse convergence status.

## User build and focused verification

Rebuild C++ and **both** acoustic shaders. acoustic_segment.comp includes the new
acoustic_diffuse.glsl from its source directory. Do not mix old/new binaries:
instance stride is now 128 bytes, summary 752 and surface-debug instance 192.
Compile-time size assertions and backend sizeof-based allocation/copy sizes are
updated. The maximum per-source response is 14 taps; source and instance limits,
4 Hz target submission cadence and 1024-candidate ray budget remain bounded.
Extra ray work is opt-in and has not been performance-measured.

Static checks completed: project XML, existing acceptance-script AST, whitespace,
layout byte arithmetic and source audit of callback arrays / all query paths.
An independent Lambert infinite-plane check (unit height/reference, isotropic
source and coincident listener) gives an analytic normalized return of 0.5;
64-direction quadrature gives 0.499755859375. This checks the factor-four
normalization; it is not a GPU or playback acceptance result.

The existing `scripts/test/rt_test_acoustic_playback_ipc.py` was extended, not
replaced with another test script. Post-build, in its required disposable empty
scene, it verifies zero scattering emits no diffuse paths, full scattering
produces <=8 bins, half scattering halves receiver energy, invalid types reject,
and directional-source rejection does not create a new preview revision. It also
keeps the previous specular/concurrency/PCM cases. This suite was not run against
the old binary.

In the saved fluid room, enable diffuse on a source and vary only the room
profile's scattering at fixed A/T. Check specular energy decreases while diffuse
energy increases, preserve moving-source acceptance, and listen to clap dry /
specular / diffuse / combined states at safe authored gain. Repeat source/scene
changes in flight and check debug shader ABI/parity. Long-run performance,
continuous timeline/cache replay, diffuse PCM reconstruction quality and
quadrature convergence require separate acceptance.

## Bounded live acceptance after user build

The user rebuilt C++ and shaders; external IPC probes in FluidAudioRoom passed:

- At scattering 0, both sources reported 64 samples and zero diffuse energy.
- At scattering 1, each had 62 contributing samples, no rejected/near-receiver
  queries, and zero specular energy. At 0.5, diffuse receiver energy was exactly
  half the scattering-1 result for each source, within numeric tolerance.
- Four real fluid steps retained error-free diffuse reports; after the initial
  pending/start-threshold sample, water played automatically and both previews
  were clear with seven combined specular/bin paths. Moving water had 63 diffuse
  contributions in completed samples, with no reported query rejection.
- Invalid diffuse type and directional-cone enablement rejected without changing
  preview revisions. Original cone angles were restored.
- A temporary continuous mono-water recording on the manual reference, gain 0.08,
  provided nonzero native PCM in dry and diffuse-only arms. Ten meter readings per
  arm gave mean square 2.3982354e-7 vs 2.9294537e-7; overload blocks stayed zero.
  These sparse meter samples are an output smoke check, not an impulse-response,
  listening-quality or quantitative PCM-energy accuracy test.

The profile scattering, source asset/gain/cones and prior timeline playback mode
were restored; the saved project was not overwritten. Diffuse/specular previews
remain enabled in this session. No new test script was added. Evidence:
`docs/dev/acoustic_diffuse_acceptance_20261018.json`. Long continuous playback,
cache replay, convergence, performance and reconstruction-quality checks remain
open. No additional core/source change was needed during this acceptance turn.
