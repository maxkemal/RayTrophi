# Domain contact position and power

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
