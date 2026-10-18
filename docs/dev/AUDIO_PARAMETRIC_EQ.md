# Parametric EQ: processing and authoring controls

2026-10-18 source delivery. No C++ build, Studio launch, live IPC test or listening
acceptance performed. This is Designer EQ/chain work, not realtime mixer-bus EQ.

## Shared core

AudioParametricEq uses the same coefficients for PCM and complex response. The
previous embedded biquad was extracted rather than duplicated in UI. Processing
is fixed 48 kHz, mono/stereo, with independent double-precision channel state.
Coefficients are prepared outside processing; tiny states are cleared. Validation
checks finite coefficients and second-order stability before publishing a design.
EQ sample output is checked for finite values, with channel bounds checked before
processing. Existing recipe limits remain eight bands per EQ and eight effects.

Modes: bell, low/high shelf, low/high cut at 12 dB/octave. Controls: 20..20000 Hz,
-24..24 dB and Q .1..10. Cut gain is retained in recipes but has no DSP effect and
is disabled in UI. New `output_db` is -24..24, default 0; it scales the wet cascade
before dry/wet blending. The response includes complex phase, mix and trim in the
same order. Bypass and mix=0 remain identity. No automatic normalization, gain
compensation, oversampling or linear-phase processing is claimed.

Designer retains its existing final [-1,1] output clipping for compatibility.
Its processing report now adds `peak_before_output_clip` and
`peak_before_output_clip_dbfs` alongside the clipped-sample count. The UI warns
when clipping occurred and suggests reducing EQ/output level. These sample peaks
are measured before loop joining, not true peak or LUFS. An earlier limiter does
not guarantee headroom after later gain increases.

## UI and shared authoring

Designer > Effects has a dedicated EQ editor. Wide layouts place the curve beside
precise controls; narrow layouts stack them. Log frequency ticks, dB grid, numbered
handles and a selected ring identify edits. Solid curve: full EQ including trim,
mix and bypass. Thin curve: selected band's theoretical response, even bypassed.
Display range +/-12/24/48 dB changes the graph only, never the DSP/stored range.

Click selects; drag edits frequency/gain; Ctrl+wheel changes Q. Cut handles stay at
0 dB because gain is inactive. Empty double-click adds a band. Right-click opens
enabled/reset/remove actions instead of immediate deletion. A click alone does
not render; changed gestures complete on release. The draft curve follows edits;
audio still renders/auditions through the existing Designer path, not continuous
realtime coefficient modulation.

Controls expose band enabled/type/frequency/gain/Q, whole-EQ enabled, mix and output
trim. Precise numeric entry remains. Tone setup reads a neutral Bass/Body/Brightness
recipe from the canonical catalogue; it replaces bands, preserving trim/mix. Clear
bands also retains trim. Rack order is labeled top-to-bottom; move controls are
visible, and bypassed modules are explicitly labeled.

Existing UI, Python `rt.audio.*` and IPC `audio.*` use the same `design_render`,
`design_effect_visual`, `design_catalogue`, undo/redo and save paths. Capability and
error semantics remain. Public additions:

- EQ recipe: `output_db`, canonically validated/defaulted.
- Catalogue: `eq_presets`, including the neutral three-band tone setup.
- EQ visual result: `sample_rate`, `band_curves` (band recipe plus points),
  `estimated_max_boost_db`, `linear_response_only`, plus effect/kind/points.
- Processing report: pre-output-clip peak metrics above.

Saved generated assets retain the normalized EQ recipe in provenance and actual
rendered PCM. Invalid fields/types/ranges/coefficients or nonfinite render output
reject before replacing the previous valid design. The sampled response/maximum
boost is a linear guide, not a nonlinear full-chain response or guaranteed headroom.

```python
eq = {"type": "eq", "enabled": True, "mix": 1, "output_db": -3,
      "bands": [{"mode": "peak", "frequency_hz": 1200,
                 "gain_db": 3, "q": 0.8, "enabled": True}]}
curve = rt.audio.design_effect_visual(effect=eq)
# Add eq to the ordered effects array of design_render's complete recipe.
```

## Acceptance after user build

Existing `rt_test_audio_design_tools_ipc.py` extended, not run: mode/frequency/Q/gain
boundary curves; 1 kHz PCM gain; trim and partial wet blend; exported stereo channel
isolation; overload reporting; invalid trim/band types retaining the previous design.
Existing bypass, zero mix, effect order, history and recipe reuse remain. Its stale
catalogue expectation was corrected to include the existing spectral_region effect.

Run externally on a paused disposable scene with an output directory. Check narrow
and wide layouts, selected band, graph/slider agreement, bypass, cut handles, Q wheel,
explicit removal, rack order, trim, clipping warnings, undo and saved recipe reuse.
Listening quality and original/processed level comparison require user acceptance.
Live bus EQ, loudness matching, true-peak measures and broader synth anti-aliasing
remain separate work. Static lexical/registration/AST/whitespace checks are not a
professional-audio quality certification.


### Visual-first EQ update (2026-10-18)

The Effects canvas now gives EQ its full available width and a responsive 200-420 px
plot. Eight consistent band colors connect handles, individual response curves,
filled influence areas and compact band selectors. A white curve shows the complete
EQ including mix, output trim and bypass. Bass/Body/Clarity/Air labels support users
without audio terminology. Only the selected band's precision controls expand on
request; inspector EQ parameters are collapsed by default, with a direct visual-editor
entry. Existing recipe validation, render service, scripting and IPC remain canonical.

Spectrum uses the existing `design_job_start(kind="analysis")` / `design_job_apply`
service with 4096-point FFT and 256 frequency points. It refreshes once per render
revision and original/processed selection, only when the shared worker is idle or
consumed; it does not replace outstanding jobs. Displayed data is measured from the
last rendered original or processed PCM, across the whole clip, not live playback or
an isolated EQ-stage tap. Original means before the complete effect rack. A changed
recipe must render again before its spectrum changes. EQ gain uses the left dB scale;
the spectrum uses an independent right -96..0 dBFS scale. No simulated animation.

Manual acceptance after user build: open Effects and select EQ; drag each colored
handle, Ctrl+wheel Q, double-click empty canvas to add, right-click to change filter
shape/bypass/remove, and confirm tone setup and whole-EQ bypass. Render a non-silent
sound, wait for the background spectrum, switch Before effects, and render an edited
recipe again. Confirm no render is interrupted, spectrum follows the rendered revision,
mono/stereo and silence are stable, and narrow windows remain usable. Check precision
controls, undo/redo and exported recipe round-trip with the existing design-tools IPC
suite. No build/application launch was performed for this UI update.


### Input routing and side controls (2026-10-18)

At widths of at least 560 px the EQ uses a resizable graph/control split. Selected-band
precision controls occupy an independently scrollable child beside the graph; narrower
layouts stack controls. Normal wheel scrolling is restored in the surrounding Designer.
Only the hovered EQ graph owns the wheel: wheel zooms around the pointer, Ctrl+wheel
on a band edits Q, middle drag pans the log-frequency window, and middle double-click
or toolbar Fit restores the full frequency range. Editing/add-band coordinates use the
same transformed frequency window as rendering, so zoom/pan does not move authored
bands. The existing canonical workspace fields `designer_canvas_zoom` (1..8) and
`designer_canvas_pan_x` (0..1) expose this view through scripting and IPC; their generic
Designer meaning becomes EQ frequency zoom/pan while editing EQ. No new authoring API.
Original/processed spectrum follows this frequency transform with its own dBFS scale.

Designer side panels now match Terrain's 13 px full-height seam strips, centered 3.5 px
chevrons, hover highlight and 6 px drag splitters. Expanded widths survive collapsing.
Both sides use existing workspace expansion operations. No View menu or boxed arrows.

User verification: wheel over the waveform, non-graph area, side panels and precision
controls must scroll normally; wheel over EQ must zoom without scrolling its parent.
Check Ctrl+wheel Q, pan to both frequency ends, drag/add a band while zoomed, Fit,
independent left/right collapse, resize, and narrow dock layouts. Builds and live UI
verification remain user-run.


### Compact dials and imported assets (2026-10-19)

Supersedes the duplicated graph-side precision slider layout: the full-width EQ frame
has an integrated compact footer with three band-colored rotary controls (frequency,
gain, Q), band bypass and filter shape. Drag to turn, Shift for fine changes,
double-click for finite bounded numeric entry. Actual drag edits commit on release;
selection alone does not render. Cut filters disable the gain dial. Detailed sliders,
mix and output trim remain in Properties. Graph zoom/pan and real spectrum are retained.

Audio Assets exposes `EQ / Edit` next to Preview. It opens the selected imported asset
in Designer Effects with the canonical flat Bass/Body/Brightness setup. Existing recipe
operations perform all DSP: `design_render` or `design_job_start(kind="render")` with
`mode="asset"`, `asset_id`, source range and ordered EQ effects; `design_preview` auditions;
`design_save` creates a separately numbered file and imports it as a new asset with recipe
metadata; `design_reuse` restores it. These operations are already exposed in Python and
IPC. No UI-only DSP or separate processing implementation was added. Toolbar
`Save as asset...` and inspector saving share the same UI helper and service. Opening a
new source cancels the Designer's outstanding render and clears pending save/preview state.

The existing core limit is 30 seconds including tails/stretch; opening an imported asset
selects up to the first 30 seconds and shows a range warning for longer files. Edit can
select another range. This is not unlimited-length full-file processing. Original files
are preserved. Already processed assets can be edited as audio, or their saved recipe
reopened with Reuse recipe.

Existing design-tools IPC acceptance coverage is extended for imported stereo EQ save:
new asset/path, unchanged original bytes, recipe source and EQ preservation, and exact PCM
hash on reuse. Source checks only were run; the suite awaits the user build. Manual UX
checks: direct asset EQ entry, all three dial drags/Shift/typed input, cut-filter gain
disabled, Properties precision sliders, audition, save from toolbar with inspector hidden,
then Reuse recipe and compare the exported sound.


### Shared selection and gain-axis zoom (2026-10-19)

EQ band selection now uses canonical workspace `designer_eq_band` (integer 0..7).
Properties tree selection or band-value changes select that same band in the canvas;
canvas handles/dials use the same recipe JSON and publish their selection back. The
Properties tree highlights the shared selection. Band gain, effect output trim and
final sound output gain remain distinct: band gain changes its dial and band response;
EQ trim changes the combined white EQ curve; final output gain affects the rendered
sound rather than the EQ transfer function. Shared recipe edits are marked dirty before
subsequent control/render polling.

Visible top-of-graph Y zoom +/- buttons and Shift+wheel over the graph change vertical
half-range, using `designer_eq_range_db` (finite 6..288 dB). `Fit curve` enables
`designer_eq_auto_range` (bool, default true): it fits sampled combined response and
band handles, with margin, to avoid the old fixed +/-24 dB clipping. Wheel still controls
frequency zoom, Ctrl+wheel on a band edits Q, middle drag pans frequency. These are view
operations and do not render or change audio. All three workspace fields are available
through existing Python/IPC workspace_get/update, with validation and atomic rejection.

The existing IPC acceptance script includes valid selection/range boundaries and eight
invalid selection/range/auto-mode cases with unchanged-workspace assertions. Not run;
static checks only. After the user build, edit Band 2 gain in Properties while Band 1 was
selected: the selected graph handle and gain dial should switch to Band 2 and show the
new value during dragging. Then edit via dial and inspect the same band's slider. Check
manual Y zoom, automatic fit with stacked boosts/cut filters, bypass, and frequency zoom.


### Compact toolbar and whole-panel navigation (2026-10-19)

EQ enable, Add band, Tone setup, a single gain-axis Zoom slider, Auto, frame height,
Spectrum and Before/After now share one top toolbar. Narrow widths scroll this row
horizontally instead of multiplying permanent toolbar rows. `designer_eq_height`
(finite 180..600 px, default 280) controls the graph frame and integrated dial footer.
The measured spectrum stays overlaid in the graph; status only adds a line while no
spectrum is available. The separate rendered waveform below Effects is collapsible.

The Designer's main toolbar exposes Panel scale. `designer_panel_zoom` (0.5..1.5,
default 1) scales the center workspace font, EQ frame, dials and rendered views without
scaling either side inspector. Alt+wheel anywhere in the center controls this scale;
Ctrl+middle drag pans the entire center workspace. Canonical normalized scroll fields
`designer_panel_pan_x/y` (0..1) are independent from EQ frequency pan. Existing graph
wheel, Shift+wheel gain zoom and middle-drag frequency pan remain local; Alt+wheel and Ctrl+middle drag reserve
those gestures for the parent panel. Fit restores panel scale/placement and canvas.
All four new fields use workspace_get/update through the shared service/Python/IPC,
with finite bounds and atomic validation. Existing IPC tests cover valid values and
eight malformed/out-of-range requests. Source checks only; no build/live verification.

Manual acceptance: at normal and narrow dock widths check the single-row toolbar,
height drag, whole-panel slider, Alt+wheel and Ctrl+middle pan, then graph-only gestures without Alt.
Check Fit, ordinary wheel scrolling, collapsed waveform, real spectrum visibility and
both side docks. Verify labels/handles/dials remain usable at panel scales .5, 1 and 1.5.


Navigation adjustment: unmodified middle-drag pans the whole Designer center panel.
Shift+middle-drag over EQ pans only the frequency window. Alt+wheel controls panel
scale; graph wheel controls frequency zoom and Shift+wheel controls gain zoom.
This supersedes the Ctrl+middle whole-panel mapping above. Side docks retain normal
wheel scrolling and their own resize/collapse strips; 3D camera controls are unchanged.
