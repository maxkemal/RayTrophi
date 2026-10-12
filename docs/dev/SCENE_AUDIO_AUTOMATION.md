# Automatic environment and measured domain audio

2026-10-12. Source implementation; user build and live acceptance pending.
Continues SCENE_AUDIO_REACTIONS.md. No build, Studio launch or generated test script
was performed. Cloud/simulation solver implementations were not changed.

## Environment setup

Scene Audio > Connect starter wind / rain invokes `audio.environment_setup` through
the shared authoring service. The recipe probes/imports the packaged stereo wind/rain
assets and creates two nonspatial loop sources on the Weather routing label, configured
with automatic wind/rain drivers. These are ordinary editable scene sources with
stable asset/source ids, serialization and undo. Weather remains a routing label;
this does not implement the stage 2 mixer.

Calling setup again reuses the sources identified by environment_role and re-arms
them instead of adding duplicate loops. A missing pack/file fails atomically, leaving
the original scene unchanged. Choose environment Assets folder supplies an explicit
root containing audio/cc0_starter; otherwise the existing AssetRegistry default root
is used. Sidecar license/source/author/tags are copied into imported asset metadata;
the original-file hash manifest remains with the starter library.
The starter wind is a short gust, not a guaranteed seamless production ambience;
replace its asset with a suitable seamless loop for a finished scene.

```python
sources = rt.audio.environment_setup()
# Optional explicit library:
# sources = rt.audio.environment_setup(assets_root=r"E:\project\RayTrophiStudio\assets")
rt.audio.update(id=sources["wind_source_id"], gain=2)
```

Wind reads the shared climate authority's surface wind speed. Rain takes the maximum
of enabled WEATHER_RAIN intensity and ground-reaching cloud precipitation:
layer 0 enabled with positive coverage, snow=false, ground_fraction >=0.999, and
rate_mm_h / 30 clamped to [0,1]. Virga and snow do not trigger the rain bed.
This is a global weather/climate control, not a local sample of the cloud core map,
building shelter or ground impact. It never reads GPU image pixels or triggers per
render sample. Listener movement still uses the active camera.

## Automatic source lifecycle

All nonmanual reaction drivers accept automatic (default false), start_threshold
(default 0.02, 0.0001..1) and stop_threshold (default 0.005, 0..1 and strictly below
start). Thresholds apply to the normalized smoothed activity, independently of the
authored gain and pitch ranges. Automatic sources require loop=true and a bound asset.
Existing scenes remain manual unless explicitly configured or set up by the recipe.

On threshold crossing the service starts one resident voice; it does not restart
every control frame. Once activity fades below the off threshold, the voice is
released. Hysteresis prevents near-threshold chatter. At most two new automatic
voices can start in a control tick; existing voice/memory budgets still apply.
Device/decode/budget failures are reported in audio.status.automatic_errors and the
inspector, with a five-second retry backoff. No failure is allowed to abort rendering.

Pause/Stop suspend that source's automation; Stop All suspends all automatic sources.
Play, editing the reaction configuration or changing its asset re-arms it. The
`audio.resume_automatic` / `rt.audio.resume_automatic()` runtime operation re-arms
all configured automatic sources and thunder, and ends asset audition; it requires
SceneWrite + FilesRead. Scene Audio exposes a Resume automatic audio button and
timeline Play transitions call the same service operation. An opted-in source still
stops if its driver is inactive, even after Play. Asset audition owns the engine and
blocks automatic starts until preview ends. Disabling automatic releases an
automatically managed voice. Preview mute silences output while activity/cursors
continue. Suspensions/errors are transient, not saved. Load/undo clears runtime
state; restored automatic sources may start again if their driver is active, while
manual sources never autoplay. This supersedes the earlier manual-only load policy.

Controls run on the main editor tick, while decoded playback runs on miniaudio's
device thread. Initial decoding is still synchronous outside the callback; stalls
delay reaction updates. Offline/timeline mixing, async decode and the standalone
player remain later work.

## First automatic domain adapter

`audio.domains` / `rt.audio.domains()` reports each existing particle-system grid
domain with system_id, system_name, domain_name, enabled/valid/version, measured,
contact_energy_j, contact_pairs, position, measurement semantics, reason and audio_error.
The Domain Contact driver in Reactive Audio binds a selected source to system_id
and domain_name using this same scene service. A new binding must name an existing
domain; a missing domain after reopening remains repairable and silent.

The adapter reads the solver's already published mixed_model_step,
mixed_step_held and mixed_contact_energy_loss statistics. The magnitude is joules of
mixed-contact energy loss in the last published solver step. It is not kg/s, not
watts, and not the sum of every substep skipped by the editor. Unsupported solver
paths report solver_contact_metric_unavailable instead of deriving noise from
particle count, an authored emission rate or stale host particle arrays.

Publication requires simulation time to advance, substeps_last_advance >0 and a new
domain state version. Repeated rendering of the same report never refreshes a
signal. Backwards time/version and scene load/undo invalidate old publications.
Each measured report updates a bounded runtime signal with a 0.3-second TTL; stopped,
deleted, disabled or unmeasured domains fade to silence. The bridge performs no new
GPU readback, particle scan, simulation stepping or geometry/facade work.

The signal position is the domain bounding-box center, explicitly not the precise
contact location. follow_position moves the actual voice there; the authoring gizmo
still displays the authored source transform. source_material/target_material
filters are rejected for this aggregate driver, because its published metric does
not distinguish the contacting substances. A source intended for positional audio
needs a mono asset. A continuous water-flow or contact texture is more suitable than
a short one-shot repeated as a loop.

```python
inputs = rt.audio.domains()
domain = inputs[0]
rt.audio.update(id=contact_source, loop=True,
                reaction={"driver": "domain_contact", "system_id": domain["system_id"],
                          "domain_name": domain["domain_name"], "input_max": 1,
                          "follow_position": True, "automatic": True})
```

External `signal_set` remains available for a producer with real mass-flow/contact
speed, material pair and contact position. The @domain/ namespace is reserved for
automatic publications. Detailed sand->water vs sand->mud routing, dry DEM pour/
friction energy, exact localized impacts, lightning/thunder event deduplication and
propagation delay are not implemented by this first aggregate adapter.

## API/security and pending acceptance

UI, Python and IPC share the source validator, authoring snapshots, environment
recipe and runtime controller. audio.domains is Read. environment_setup is undoable
and requires SceneWrite|FilesRead. create/update/duplicate also require both because
they can persist automatic playback that later reads an asset; this closes an
automatic-playback FilesRead bypass. Other existing controls retain their policy.
environment_role (empty/wind/rain) is reserved for the recipe; duplicates clear it.

After the user's normal build:

1. Connect the starter weather twice. Confirm exactly two sources and two assets,
   one undo for setup, no duplicate loops, preserved source ids on reconnect.
2. Increase climate wind from zero, then return to zero: automatic start, smooth gain,
   one voice and eventual release. Test manual rain intensity and cloud rate with
   full ground_fraction; snow/virga must not start the rain bed.
3. Pause/Stop an active automatic source: it must stay paused/stopped across editor
   frames. Play/reconfigure resumes if active. Stop All holds; audition is exclusive.
4. Rename a source file or remove the device: see an explicit automatic error and
   bounded retries, with working image rendering. Verify scene save/reopen keeps
   config and relative assets, not transient suspensions/errors.
5. Inspect audio.domains on a running mixed-model domain. Bind a mono contact texture,
   confirm source activity tracks measured contact_energy_j and follows the domain
   center. Pause simulation and confirm signal expiry, fade and voice release.
6. Disable/delete/rename a bound domain, seek backwards and replay cached frames.
   Stale reports must not keep resetting TTL. Unsupported/dry-only paths remain
   visibly unavailable. Verify simulation transfer counters are unchanged by audio.
7. Verify Read-only and SceneWrite-without-FilesRead credentials cannot configure
   automatic readers; inspect method discovery and use external scripts/test/rt_ipc.py,
   not the embedded script workspace, for live controls.

These checks are pending, not claimed passing. No new test script was added.
