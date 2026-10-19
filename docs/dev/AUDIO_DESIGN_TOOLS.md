# Sound designer v3: visual production tools

2026-10-18 source continuation: [Parametric EQ](AUDIO_PARAMETRIC_EQ.md) extracts
shared checked coefficients for processing/response, adds wet output trim,
per-band curves, selected-band authoring and explicit pre-output-clip metrics.
Rack order/bypass are clearer. Core/UI changes require the next user build;
the existing external tools test was extended, not run. This is Designer EQ,
not realtime bus inserts or a claim of full professional-audio acceptance.

2026-10-15 continuation: [background paths](AUDIO_BACKGROUND_JOBS.md) now include
Arrange mix, source waveform and history/reuse;
[spectral region editing](AUDIO_SPECTRAL_EDITING.md) adds real PCM masks and Analyze selection.
These latest changes await the final build/test/listening pass requested by the user.

2026-10-14: implemented in source, static checks only. The user confirmed the earlier
v2 Studio build and synthetic preview. This expanded version needs a new user build
and the acceptance below; it has not been compiled or auditioned by Codex.

## Four workflows in the existing Audio editor

- **Generate:** legacy characters, three complete layered recipes, up to eight
  reordered/bypassable oscillator, colored-noise, FM, resonator or imported-sample
  layers. Each has level, pan and start time. Global ADSR has draggable handles;
  amplitude modulation, duration and seed apply to the composite. Legacy character
  controls are disabled when explicit layers replace that engine.
- **Edit:** separate stereo source waveforms, wheel zoom, pan, draggable range edges
  and range movement, processed fade handles, reverse, independent pitch/time,
  formant shifting/compensation and loop crossfade. Source files remain intact.
- **Effects:** up to eight ordered, bypassable effects with dry/wet: parametric EQ,
  resonant filter, compressor, delay, room reverb, chorus, phaser, overdrive and
  lookahead limiter. EQ supports eight bands with draggable frequency/gain and wheel
  Q; filter and dynamics have interactive curves. Parameters remain available in
  the contextual inspector; only the selected effect occupies the bottom graph.
- **Analyze:** selectable FFT size and stereo channel, peak spectrum, spectrogram,
  hover inspection, peak/RMS/dBFS/crest/DC and full-scale sample counts.

The compact toolbar contains audition, original/processed comparison, loop, refresh,
Designer undo/redo and Controls. Generic scene timeline controls are hidden only in
Designer. No permanent shelf is added. Master gain and effect-tail duration are
available on all authoring pages. Save uses a name and the existing project/library
folder; the folder picker and collision suffixes remain available.

## Canonical DSP contract

UI, Python `rt.audio` and external `audio.*` IPC share the same validated service.
`design_catalogue` is the authoritative module field/default/bounds schema. All
unknown fields, wrong types, unsupported modules and out-of-range values fail before
the current design is replaced. Disabled modules still need valid parameters.

| Layer | Controls and bounds |
|---|---|
| Common | enabled; gain -60..6 dB; pan -1..1; start 0..10 s, enabled start before end |
| Oscillator | sine/triangle/saw/square; 20..8000 Hz; detune ±1200 cents; pulse width .1...9 |
| Noise | white/pink/brown; cutoff 20..20000 Hz |
| FM | carrier 20..4000 Hz; ratio .1..8; index 0..10 |
| Resonator | 20..4000 Hz; structure/brightness/damping/position 0..1; burst .001..1 s |
| Sample | existing asset_id; source offsets 0..600 s; selected range .05..10 s |

Layers require synth mode and produce stereo with equal-power pan, short edge fades
and deterministic noise. Global Designer duration is .05..600 s, subject to the PCM budget. An overloaded composite is
normalized before ADSR/output processing. Empty layers retain the legacy engine.
The top-level imported-asset mode still permits a .05..600 s selection, subject to the PCM budget. File decoding
retains the existing 600 s / 128 MiB bound; zoom caches one decoded source by path,
size and modification time. Rendering reads selected sample assets again.

| Effect | Bounds beyond enabled and dry/wet 0..1 |
|---|---|
| EQ | peak/low shelf/high shelf/lowpass/highpass; 20..20000 Hz, ±24 dB, Q .1..10 |
| Filter | low/high/band/notch; 20..16000 Hz; resonance 0...95; drive 0..1 |
| Compressor | threshold -60..0 dB; ratio 1..20; knee 0..24 dB; attack .1..200 ms; release 5..2000 ms; makeup ±12 dB |
| Delay | .001..1.5 s; feedback 0...9; damping 0...95 |
| Room reverb | decay .1..8 s; size .25..1.5; damping 0...95 |
| Chorus | rate .05..10 Hz; depth 0..1; delay 1..40 ms; feedback 0...8 |
| Phaser | rate .05..10 Hz; depth 0..1; frequency 20..8000 Hz; feedback 0...8 |
| Overdrive | drive 0..1 |
| Limiter | ceiling -12..0 dB; release 5..1000 ms; stereo-linked 240-frame lookahead |

EQ gain has no effect on low/highpass bands and is disabled in those UI controls.
Bypass and zero mix skip processing exactly. Compressor and limiter link channel
gain. Room reverb is an authored Schroeder network, not scene acoustic simulation.
Selected DaisySP MIT modules supply oscillator, ADSR, FM, resonator, SVF, chorus,
phaser and overdrive; custom EQ/dynamics/delay/reverb use the same core rack.
Signalsmith supplies pitch/time/formant processing and FFT. Existing pinned
revisions/licenses remain; no DaisySP-LGPL modules are included.

Processing order: source/layers and envelope, reverse, pitch/time/formants, appended
tail, ordered rack, legacy output filters/drive, master gain/fades, sample clamp,
loop seam. Tail is 0..5 s; total result cannot exceed 600 s or the PCM budget. A gain boost after the
limiter can therefore exceed its ceiling. `gain_db` (-60..6) is an input alias;
canonical recipes store linear `gain`. Conflicting supplied gain values fail.
Formant shift is ±12 semitones; base is 20..2000 Hz; compensation defaults false.
Neutral extensions preserve the legacy processing path.

## API additions, history and reuse

| Operation | Parameters | Capability |
|---|---|---|
| design_catalogue | none | Read |
| design_status | details=true; false gives revision/transport/history flags | Read |
| design_analyze | original=false, fft_size=2048, points=256, time_bins=64, channel=mix, continuous=true | Read |
| design_waveform | id, optional start_seconds/end_seconds, points=512 | Read + FilesRead |
| design_effect_visual | effect object | Read |
| design_undo / design_redo | none | SceneWrite + FilesRead |
| design_reuse | id of a saved Designer asset | SceneWrite + FilesRead |
| workspace_update | designer_page, designer_zoom 1..2048, designer_pan 0..1, designer_effect 0..7 | SceneWrite |

Existing render/preview/save/stop/destination operations remain. There are 67 dynamic
audio operations in total. Errors retain audio_error/Python ValueError semantics.
Render/undo/redo/reuse are locked during render-output authoring ownership.

Designer history stores up to 32 recipes independently of scene undo. Undo/redo
rerenders against current source assets; missing or changed inputs can fail without
losing the current output/history. Scene replacement clears transient state.
Saving embeds recipe and original/sample provenance. Reuse accepts saved v2/v3
metadata, resolves referenced asset IDs and rerenders; ordinary imported sounds
are rejected. The saved WAV itself remains usable without its original inputs.
Save imports the exact preview PCM and preserves license/author metadata.

## What graphs measure

Full analysis uses Hann STFT with hop FFT-size/4, covers the entire sound and
aggregates maximum power into time bins. FFT sizes are powers of two 512..8192;
frequency columns 16..1024; time bins 4..128. Mix averages channel powers rather than
cancelling opposite-phase channels. `continuous=false` samples time-bin windows;
the cheap render/status summary uses sampled analysis, not the full spectrogram.

Waveform bins cover exact frame ranges with per-channel minima/maxima/RMS.
Full-scale sample hits mean abs(sample)>=1, not proof of clipping. Processing
reports actual samples clamped before the loop seam separately. No LUFS or
oversampled true-peak claim is made. FNV PCM hashes compare this implementation's
results, not cross-platform bitwise guarantees.

EQ curves are analytical responses; filter curves are finite small-signal impulse
responses. Dynamics/overdrive curves describe steady input/output levels; delay,
reverb and modulation graphs are parameter guides. These guides do not replace
analysis of the actual processed PCM.

Rendering/analysis remain synchronous on the editor thread. Preview updates on
completed edits and restarts audition. Background rendering, uninterrupted live
DSP, plugin hosting and physics-driven procedural synthesis remain future work.

## User build and acceptance

Build Studio only; this feature adds no shaders. CMake users must reconfigure to
discover new source files. Codex does not build or launch Studio.

With the new build open, run from a separate terminal (never embedded scripting):

```powershell
python scripts/test/rt_test_audio_design_tools_ipc.py --output-dir docs/dev
python scripts/test/rt_test_audio_sound_design_ipc.py --output-dir docs/dev
```

The new probe checks layer determinism, each effect and bypass/mix neutrality,
rack order, finite curves, full FFT coverage of a short transient, stereo frequency
peaks, exact waveform ranges, rejected inputs preserving output, history, tail,
preset renders, save/reuse and atomic workspace validation. It creates/removes
only its own temporary files/assets and restores workspace settings; it does not
audition. Tests have not been run against the application yet.

Manual: check compact layout at wide/narrow sizes, each workflow, source zoom/range
handles, ADSR/fades/EQ/filter/dynamics drag controls, rack reordering/bypass, audition
original/processed/loop/Stop, stereo analysis, undo/redo, save by name to the shown
folder and Reuse recipe. Compare the saved WAV to preview and listen for clicks,
tails and level changes. Report stalls on the longest permitted layered/rack sound.

Source verification passed: IPC capability/security mirror audit (667 static routes),
67 unique dynamic audio operations, Python probe syntax, VS project XML and new
module registration, enabled DaisySP include closure, revision-notice parity, focused
C++ delimiter/whitespace checks and git diff whitespace checks. These do not replace
C++ compilation or live DSP/UI acceptance.


## Live core acceptance and integrated UI follow-up, 2026-10-14

User built v3 and left Studio open. Codex ran both external probes successfully:
`rt_test_audio_design_tools_ipc.py` and `rt_test_audio_sound_design_ipc.py`. Layers,
rack order/bypass/curves, full FFT/transient coverage, source ranges, history/reuse,
render/save PCM, errors, safe names, pitch/time and loop contracts passed on this
build. Probes do not verify mouse interaction or audible quality. Their temporary
assets/files were removed and workspace settings restored.

User found graphs useful but wanted controls integrated into the bottom editor.
Follow-up source changes put the same controls/save UI beside the canvas in a
resizable two-column layout; narrow panels use an expandable Controls / Save section.
Opening Designer no longer automatically opens the external inspector. The inspector
points to the integrated editor. Source waveform wheel zoom now also has middle-drag
pan and double-click fit. EQ supports double-click empty graph to add a band and
right-click a band to remove it. All edits use existing canonical recipe/workspace
operations, with the same validation and Designer history.

This UI follow-up is source-only, pending another user build. Verify wide/narrow
layout, resize the divider, use layer/effect add/settings without the inspector,
then test waveform wheel/middle-drag/fit and EQ add/drag/Q/remove. Zoom/pan here
refers to the source waveform; effect plots retain their labelled frequency/dB axes.


## Studio side-panel architecture, 2026-10-14

The next source revision replaces the permanent control column with a large central
canvas and two closeable, resizable bottom-editor panels, following the existing
node/terrain editor side-panel interaction. Tools on the left contains layer/effect
add, remove, reorder and module parameters. Properties on the right contains the
workflow's precise generator/edit/shaping settings and defaults closed. Both can be
hidden for a full-width canvas. Narrow editors use collapsible sections instead.
Input selection, sound name, destination and saving live in the main Audio panel
with its existing asset/source/scene/output sections. Assets / Output opens it.

Canonical workspace fields `designer_tools_open` (default true) and
`designer_properties_open` (default false) are strict booleans exposed through UI,
`rt.audio.workspace_update` and IPC. Layout divider widths use ImGui resizable-table
state. Invalid toggle values are covered by the external probe. These new fields
and UI require a new user build; earlier live passes cover the previous revision.
No builds or app launch were performed for this change.

Next stages remain non-destructive multi-layer clip editing, spectral region
processing/resynthesis, background preview and visual scene-event modulation. The
Photosounder/Vital/RX references guide interaction; the current spectrogram remains
analysis-only and current layers are synthesis ingredients, not a full DAW.


## Screenshot correction: header and main-panel ownership

The next source change removes Assets / Output from the bottom editor; designed
sound input/name/folder/save now appears in the main left Audio Output tab. The
standalone inspector is for source/clip properties and no longer duplicates Designer
saving. Designer navigation and transport share one wrapping row; workflow is a
compact selector and Tools/Properties toggles are in View. Module Parameters and
generator/shaping details default collapsed; selecting visual effect controls no
longer forces numeric controls open. Existing saved ImGui collapse state may still
restore a previously open section. At narrow widths the toolbar wraps as needed.

This correction is source-only, not live-verified. Check the new Studio build for
header height, Output save controls and canvas area with both side panels hidden.
The wider graphical authoring/spectral processing roadmap remains unfinished; the
current UI correction is not acceptance of a complete professional audio editor.


## Visual authoring workspace completion for current DSP modules

Generate now offers a selected-layer canvas with labelled one-period/shape guides
and draggable parameter pads. Oscillator frequency/level, noise cutoff/level, FM
frequency/index, resonator frequency/damping and layer start/pan edit the actual
canonical recipe fields. Start placement is bounded by the current sound duration.
These guides are illustrative; the output waveform and analyzer show rendered PCM.
The legacy character remains available, and layered presets can be chosen directly
from the empty layer canvas.

Tools uses a compact selectable module list and bypass toggles. Add selects the new
module. Right-click provides move up/down/remove instead of permanent action rows.
Properties shows only the selected module's validated fields, alongside collapsed
global recipe sections. `designer_layer` (integer 0..7, default 0) joins the existing
canonical workspace/script/IPC fields; invalid type/range and successful selection
are covered by the external acceptance probe. Effect selection uses designer_effect.

Effects retains EQ/filter/dynamics curves and adds draggable pads for delay
time/feedback, room size/decay, chorus/phaser rate/depth, overdrive drive/mix and
compressor ratio/knee. Pad bounds/defaults come from design_catalogue; edits share
rendering, error handling, release-triggered preview, recipe history and saving.
Edit keeps source selection/zoom/pan, fades, reverse/pitch/time/formant controls;
Analyze keeps actual FFT/spectrogram and channel statistics.

This source revision completes the visual workspace wiring for currently supported
DSP modules, not the future spectral-painting or full multitrack DAW feature set.
Codex ran focused delimiter/size/whitespace checks, VS source-registration/XML checks,
Python probe syntax and the IPC/security audit successfully. No project build, app
launch or live acceptance of this revision was performed. Next user build: check
selected-layer/module properties, pad edits and undo, effect pads, preset selection,
context-menu reorder/remove, compact header and main Output save; then run the
updated external probe. Mouse/zoom/pan fine tuning can follow that UI inspection.


## Shared canvas navigation and silent audition correction

User built the preceding revision and reported absent zoom/pan outside source Edit
and intermittent silent Play. Read-only external IPC initially showed paused timeline
transport, no active voices, suspended weather emitters, two unbound source assets
and a finished Designer cursor. These facts do not prove every reported silent event.
A later check found no rendered Designer sound, so the audible mixer reproduction
was skipped rather than replacing the user's working design.

Source inspection identified separate mechanisms: source Play follows paused scene
transport, whereas Designer audition should be independent; Designer previously
routed through Effects and could be silenced by that child bus mute/another bus solo.
The next source change routes Designer and asset audition directly through Master,
retaining Master gain/mute and global preview mute but bypassing scene child-bus
solo/mute. Designer status reports audition_bus, audition_output_gain and output_muted.
The UI warns about muted output or silent rendered PCM, rerenders if the core design
was cleared despite a clean UI draft, and stops stale playback when choosing another
asset. Source controls now provide explicit Audition asset / Stop audition alongside
scene Play and explain the paused-transport behavior.

Generate, Edit, Effects and Analyze now share canvas magnification (1..8), middle-drag
pan, a compact Fit action and middle-double-click reset. Graph heights scale together
with horizontal content; a viewport-based graph budget reduces heights at fit scale.
The graph axes/data semantics stay unchanged during canvas magnification. Ctrl+wheel
retains precise source-range zoom and EQ Q; Ctrl+middle-drag retains source-range pan.
Canonical workspace fields designer_canvas_zoom (1..8), designer_canvas_pan_x/y
(0..1) expose the same view state through UI/script/IPC. Narrow/very short windows
may still scroll; fit does not remove labels or precise controls.

The routing/navigation change is source-only. Static source/VS/Python checks passed.
After a new user build, run the updated tools probe plus:

```powershell
python scripts/test/rt_test_audio_designer_audition_ipc.py
```

The latter briefly auditions a quiet tone, checks actual Master callback-meter energy
with Effects muted and Music solo, restores mixer/output mute and the prior rendered
recipe, and stops audition. Scene voices stopped by audition are not restarted; run
on a paused scene. Manual acceptance must check wheel zoom, middle pan and Fit on
every page, then source/Designer Play distinctions and original/processed switching.
No live success is claimed for this newly edited revision.


User rebuilt the routing/shared-navigation revision and reported no recurrence of
the issue. Source Zoom/Pan sliders have now been removed as requested; mouse gestures
and top Fit remain. This last small layout edit is source-only. Remaining work is
ordered in AUDIO_SOUND_DESIGN_PLAN_2026_10_14.md; no full-DAW/spectral-edit completion
is claimed.


Arrangement now has canonical multi-selection/batch edits, graphic gain/pan envelopes
and independent bounded mixed audition with loop/range/cursor. See the multitrack
plan for the exact contract and final acceptance checklist. This last source revision
was not built or live-tested; no new test script was written at the user request.

### Designer side panels (2026-10-18)

Tools and Properties default to expanded. Each side has an always-visible arrow rail
for collapsing and restoring it; the View popup and X close buttons are removed.
Expanded layouts retain resizable columns. Narrow layouts use inline collapsible sections.
The existing workspace scripting/IPC fields `designer_tools_open` and
`designer_properties_open` control expansion. Source/arrange is a direct Edit selector.

Manual verification after the user build: collapse/restore both sides independently,
resize expanded columns, switch workflow and source/arrange, and check narrow layouts.
No application build or live verification was performed for this UI adjustment.


2026-10-19: Designer capacity uses the shared 512 MiB estimated PCM working-set
budget (input and output, four float32 banks each), in addition to duration and
existing decode/import limits. design_catalogue exposes max_duration_seconds,
working_pcm_budget_bytes and processing_mode=bounded_memory through Python/IPC.
This is in-memory processing; see AUDIO_DESIGN_CONTEXT_WORKFLOW_PLAN.md for source
identity, save/clip targets, pending streaming work and post-build acceptance.
