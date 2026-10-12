# Domain contact position and power

## Connect an existing scene sound

Select the scene Audio Source, choose its imported sound in Asset, and open
Reactive Audio. Set Driver=domain_contact, choose Domain binding, enable Loop and
Automatic start / stop, select power_w, and set input_max in watts. Enable Follow
position for a mono positional source. Timeline Play drives the configured source.
Importing a sound into assets alone does not bind it to the scene Audio Source.

A configured binding is separate from an available measurement. A pure fluid
domain does not publish mixed contact energy; water flow rate/splash sensing is
not implemented by this driver. Current production requires a CPU mixed liquid /
granular report. The inspector explicitly explains this when input is unavailable.

Scene Audio's Connect atmosphere wind / rain and Connect scene lightning to
thunder are general scene connections, with the existing shared
audio.environment_setup and audio.thunder_setup script/IPC operations. They use
the bundled licensed library; wind/rain sources accept replacement assets.
The synthetic thunder preview is grouped under Diagnostics / sound preview.
These UI wording/layout changes require a new build; core setup semantics and
validation remain unchanged.

2026-10-12. Source implementation; user build and live acceptance pending.
Continues [SCENE_AUDIO_AUTOMATION.md](SCENE_AUDIO_AUTOMATION.md).

The CPU mixed liquid/granular MAC contact producer now publishes the duration of
its committed solver step and an impulse-weighted world-space contact cell centre.
The audio bridge reads this existing report without particle scans, simulation
steps or GPU readback. Solver impulses and velocities are unchanged by telemetry.

Each contact cell contributes the magnitude of its applied, energy-limited impulse
as a nonnegative weight. The centre is the weighted mean of grid cell centres,
combined over the step's common substeps. It is one aggregate position, not a
particle impact point, per-material position or a set of localized sound sources.
With no position weight, audio retains the documented domain-centre fallback.

`audio.domains` / `rt.audio.domains()` add:

- `contact_power_w`: committed contact energy / committed step duration.
- `power_measured`: finite duration > 0 and power within 0..1e9 W.
- `measurement_duration_seconds`: duration covered by the published energy.
- `position_semantics`: `impulse_weighted_contact_cell_center` or `domain_center`.
- `material_pairs_available`: false until a classified producer exists.

GPU contact currently publishes pair counts but no contact energy/duration. It
reports `contact_energy_not_published` and `measured=false`; zero-initialized energy
is no longer advertised as a measured GPU contact. Dry-only and held reports remain
unavailable. This change does not add GPU telemetry or identify sand/water/mud.

## Shared authoring contract

Reactive Audio > Domain Contact > Contact magnitude selects `energy_j` (legacy
default, joules per published step) or `power_w` (watts). The source's complete
reaction object is saved, restored and undone by the existing audio snapshot path.
UI, Python and IPC use the same validator and core evaluator. `input_min/max` use
the selected units; changing the metric does not silently convert authored bounds.

```python
source = rt.audio.get(id=contact_source)
reaction = dict(source["reaction"])
reaction.update(driver="domain_contact", system_id=system_id,
                domain_name=domain_name, input_metric="power_w",
                input_min=0, input_max=10, follow_position=True, automatic=True)
rt.audio.update(id=contact_source, loop=True, reaction=reaction)
```

IPC uses `audio.update` with the same reaction object. Unknown input metrics and
power metrics on other drivers are rejected. Old scenes without `input_metric`
retain energy mapping. `signal_set` accepts optional finite `power_w` metadata;
ordinary signal drivers still read `value`. Existing Read/SceneWrite/FilesRead
permissions and reserved domain signal namespace remain unchanged.

Unavailable power supplies zero activity and follows existing envelope/automatic
release rules. Authored nonzero gain_min can retain residual gain on manually
playing sources, as with other inactive drivers. The bridge's version deduplication,
TTL and timeline transport remain authoritative; cached reports never refresh TTL.

## After the user builds

1. On a live CPU mixed domain, inspect `audio.domains`: measured duration positive,
   power approximately energy/duration, and contact position inside the reported
   contact region rather than fixed at the domain centre. Bind a mono loop with
   follow_position=true and confirm listener attenuation follows that position.
2. Select power in UI or script/IPC. Adjust bounds in watts; vary simulation step
   duration with a comparable contact rate and inspect average power/activity.
   Physics itself can change with timestep; exact waveform equality is not expected.
3. Pause/Stop, backward seek, held/disabled/deleted domains: no stale publication;
   normal transport holds/releases voices. A GPU domain without energy telemetry
   must be unavailable and silent with automatic playback and gain_min=0.
4. Save/reopen and undo/redo a metric edit. Legacy scenes stay energy_j. Reject an
   unknown metric, power_w on wind, and negative/nonfinite power metadata atomically.

No build, Studio launch or generated test script was performed by the agent.
Non-build checks passed: source-size limits, API parameter/default/unsupported-path
guards, diff whitespace, and numeric power-subdivision/centroid translation and
convex-hull invariants. These checks do not execute the C++ implementation and do
not replace the build/live acceptance above.

## Short live acceptance — 2026-10-12

On the user's new build, external IPC created a small closed CPU Matter domain
with opposing Water/Soil MPM sources and a CC0 mono water-flow loop. Twelve fresh
solver steps reported measured energy, positive duration, power and a moving
impulse-weighted cell centre. Peak sampled power was 6.3593 W; one report gave
0.0582105 J over 0.0166667 s, hence 3.49263 W. In continuous test mode, the automatic
source received 2.02114 W, followed the reported world position, advanced its
cursor and produced nonzero Effects/Master PCM without automatic errors.

The scene was left available for user review, reset to frame zero, with timeline
audio restored. This verifies CPU publication and shared playback, not timestep
invariance, material-pair routing, save/reopen or timeline Play/Pause acceptance.
Gizmos were disabled by the user after an earlier creation attempt crashed in
the legacy FluidObject overlay. That overlay crash remains unresolved; this test
does not establish that a rebuild fixed it.

## RAM cache playback correction — build pending

RAM frame compression keeps APICSolverStats in CachedGridDomain.meta, and
decompression restores that metadata. prepareFluidCacheRestore publishes a fresh
domain version; restoreSimFrame sets the simulation clock to the cached frame.
The contact data was already present, but the audio bridge rejected it because
cached playback has substeps_last_advance=0.

The bridge now accepts a measured report when simulation time advances and its
domain version increases, whether from a live solve or RAM cache restoration.
Repeated redraws or re-restores at the same time do not refresh TTL. Backwards
seek/loop, scene replacement and audio undo/load still invalidate old signals;
subsequent forward frames can publish again. Normal timeline Pause/Stop policies
remain in force. This does not add contact telemetry to older disk bake formats.
When an existing cache restore rebases a manually translated domain, its measured
contact centre now translates with the particles; energy and power stay unchanged.

After rebuilding, run the test scene once to populate RAM frames, rewind and Play
the same range again. Confirm audio.domains still reports measured power/position,
audio.status reaction input and Effects/Master meters react on both passes, and
Pause/Stop silence or hold the voices as before. Same-frame waiting must not create
new publications. Live acceptance of this correction is pending.
