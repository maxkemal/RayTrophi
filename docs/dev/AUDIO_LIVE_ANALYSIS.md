# Live mixer analysis and understandable Analyze workspace

> **Durum:** AKTİF — 2026-10-19 temel canlı analiz kullanıcı build'inde çalışıyor.
> Alternatif görünümler ve mouse zoom/pan kaynakta hazır; yeni build/görsel kabul bekliyor.

## User workflow

The Audio bottom editor's **Analyze** entry defaults to **Live scene / channels**.
The Designer workflow selector also opens Analyze. This view has no
Designer source picker, Play/Loop, Original/Processed, Save or side inspectors.
It shows the current mixer signal without rendering a design. The user chooses
All audio / Master or Weather, Effects, Ambience, Music. Peak/average levels,
a large logarithmic frequency spectrum, six octave bands and a live channel table
share the existing bottom editor. No permanent shelf or new top-level panel.

Clicking the left Master signal bar opens Master analysis. Clicking the Inspector
channel peak bar opens that mixer channel, not an isolated source. Designer/Mixer
navigation remains available. **Designer: last render** explicitly returns to the
existing whole-sound FFT/spectrogram and spectral editing workflow; its banner says
that it does not follow live channels. Existing EQ response editing is unchanged.

The live frequency graph observes sound; it is not a realtime EQ control. Master
includes scene sounds and active asset/Designer/arrangement previews routed through
Master. Child channels include their own gain/pan/mute/solo, before Master gain.
The Master tap is after the Master fader and before engine/output mute. Output-muted
signals are still visible, with a warning. This is not an OS audio-loopback meter.

No device, waiting for a window, silence and no newly received audio are separate
states. No animated placeholder spectrum or fabricated energy is displayed.

## Alternative graphs and navigation

The compact View selector offers Spectrum / line, Spectrum / filled, Six bands,
and Spectrogram / time. Line and filled use the same 160 logarithmic spectrum points;
the six bars use actual octave-band mean-square power, not sampled line heights.
Peak/RMS and all five channel meters remain available below every graph.

Only the graph transforms: **Ctrl+wheel** zooms both axes around the pointer,
**middle-button drag** pans. No zoom/pan buttons or middle-button reset are added.
Zoom out to 1x to restore the full extent. Plain wheel retains ordinary panel scrolling.
The gesture is captured by the graph so Ctrl+wheel does not scroll its parent.
Labels and channel meters retain their size; axes, tooltips and data share the transform.
Zoom is 1..32 on each axis; normalized pan is 0..1 within the available zoomed extent.
The viewport is shared across graph modes and continues to show live updates.

Spectrogram x is time relative to the latest observed window (-10..0 seconds),
y is logarithmic Hz, and color is fixed -96..0 dBFS. Only observed FFT windows are
painted at their real sample-frame positions and 2048-frame widths. Unsampled gaps
stay empty; no retained value is stretched over an unobserved time interval. The
consumer retains at most 128 snapshots per bus, at most 15/sec. Sparse polling can
span longer than ten seconds; older rows outside the visible interval are clipped.
History accumulates while analysis_get is polled, not while Analyze is closed, and
is cleared on device reset. Changing channels uses that channel's own history.
History storage, throttling and queries run on the main thread; callback capture is unchanged.

## Signal contract

The existing stereo mixer nodes copy actual finite output PCM into a bounded
2048-frame window. Three fixed slots use atomic ownership (free/writing/ready/reading).
The callback copies to a free or unread slot, never one owned by a reader. Publication
has at most six compare/exchange attempts per full window; there are no mutexes,
allocations, FFTs, JSON operations or unbounded retries in the new callback path.
The main thread copies the latest completed window and releases slots before FFT.
Each bus retains its latest analysis, computed again only for a newer window.

The UI polls at most about 14 times/second. Capture continues independently of
Designer jobs, transport frame rendering and whether Analyze is visible. At 48 kHz,
one window spans about 42.7 ms and FFT-bin spacing is about 23.4 Hz. This is bounded
live analysis, not a loudness/true-peak meter. The consumer now retains bounded
sampled history for the spectrogram; it is not continuous audio recording.

Stereo FFT powers are averaged, not mono waveforms: opposite-polarity L/R does not
cancel in the spectrum. A Hann window is used. Spectrum values are sine-peak amplitude
dBFS; six octave bands centered at 125/250/500/1000/2000/4000 Hz integrate window-
corrected mean-square power. Each band's bounds are center/sqrt(2)..center*sqrt(2).
The six bands do not cover the entire audible frequency range. RMS and peaks also
report each stereo channel and the combined window; clipping counts samples with
absolute value >=1 in that window. Existing cumulative bus peak-holds are unaffected.

The six live bands are measured mixed-signal content. They are not source energy
in joules or separate direct/reflected/transmitted path energies. That decomposition
still needs consistent path normalization and source-level band weighting. Authored
material fractions and receiver-relative diffuse gains must not be summed together
or drawn as a measured scene-energy breakdown.

## Shared UI / Python / IPC

`rt.audio.analysis_get(bus="Master")` / `audio.analysis_get` is Read-only and does not
initialize a device, start playback, render a design, create files or mutate authoring
history. It also bypasses authoring's scene/physics synchronization. Unknown parameters,
non-string bus, non-boolean details and unknown bus names reject before consuming a snapshot.

Result fields:

- bus, device_ready, ready, fresh, stage=post_bus_fader_before_output_mute.
- output_muted, master_includes_audition, bus_meters (all existing bus meters).
- When ready: sample_rate, window_frames, end_frame, snapshot_age_ms.
- spectrum: 160 [Hz, dBFS] points; bands: frequency_hz/low_hz/high_hz/mean_square/dbfs.
- With details=true and an active device: history_capacity=128,
  history_kind=observed_fft_windows, history rows containing end_frame and
  160 dbfs values on the spectrum frequency grid. Default details=false omits history.
- channels: Left/Right peak_dbfs and rms_dbfs; aggregate peak_dbfs/rms_dbfs,
  silent, clip_samples_window, frequency_resolution_hz and quantity labels.

end_frame advances only when a completed callback window is received. snapshot_age_ms
is elapsed time since the query consumer received its most recent new snapshot; it is
not an absolute sample-capture timestamp. After 1000 ms without receiving a newer
window, fresh=false. A first query after a stopped device can receive a retained
window; the UI explicitly labels retained/stale data on subsequent reads. Repeated
read calls do not reset that age. Device shutdown clears snapshots after nodes detach.

Canonical transient workspace gains `analysis_mode=live|design` (default live), with
atomic type/enum validation. Existing mixer_bus selects the scope. Both fields are
available through workspace_get/update to UI, scripting and IPC. It also exposes
analysis_view=line|filled|bands|spectrogram (default line), analysis_zoom_x/y
(default 1, finite 1..32), and analysis_pan_x/y (default 0, finite 0..1). Invalid
types/enums/ranges reject the entire patch atomically. These are transient display
settings and do not enter scene undo history. Switching view
does not alter DSP, playback, saved audio or the Designer recipe.

The dynamic audio method registry in RtAudioBindings registers the operation, bus
default/schema, read access and documentation for both IPC discovery and Python.
Its C++ security rule and Python capability mirror both return Read. Static generated
descriptors remain generator-owned; no duplicate manual static audio descriptor.

```python
rt.audio.workspace_update(designer_open=True, designer_page="analyze",
                          analysis_mode="live", mixer_bus="Master")
rt.audio.workspace_update(analysis_view="spectrogram", analysis_zoom_x=2,
                          analysis_zoom_y=2, analysis_pan_x=1, analysis_pan_y=0.5)
snapshot = rt.audio.analysis_get(bus="Effects", details=True)
# No design_render call is involved.
```

## Acceptance after user build

Codex does not build or launch Studio. New .cpp files are registered in the VS project
and covered by the existing CMake source glob; no shader change is required by this
feature. Other ongoing domain/shader changes keep their own build requirements.

1. On the open room scene, externally run
   `python scripts/test/rt_test_audio_live_analysis_ipc.py`.
   This checks schema/finite readback, all five buses, invalid parameters and reversible
   workspace selection/atomic rejection. It does not start/stop sounds.
2. On an empty disposable audio scene, externally run the same command with `--signal`.
   This creates a 1 kHz opposite-polarity stereo fixture and checks non-cancelling FFT,
   octave-band RMS, advancing frames, channel gain (-6.02 dB), Master gain while the
   child stays unchanged, silence after mute and no Designer render revision change.
   It temporarily mutes device output, restores mixer/output state and removes its
   source/asset/file. Fixture authoring uses existing history; use a disposable scene.
3. Click Master signal and channel peak. Confirm the right scope opens, live graphs
   update while playing and become silent when that channel stops. Output mute keeps
   pre-mute data visible and displays the warning.
4. Change source tone/file, channel gain, mute and solo without rendering Designer:
   spectrum and channel table follow the actual mix. Verify left/right content and
   device silence at normal/narrow widths; no Designer controls appear in Live mode.
5. Switch to Designer: last render. Its original whole-sound FFT, spectrogram and
   spectral-region edits work; they remain stable while unrelated scene sounds change.
   Return to Live and check the current mixer rather than the rendered sound.

Most deceptive failure: a smooth-looking graph that keeps showing the Designer's
old render or cancels opposite stereo channels. The independent fixture and unchanged
Designer revision checks address those failures; visual animation alone is not proof.

Source checks do not validate C++ compilation, callback scheduling or listening quality.

2026-10-19 static checks PASS: Python AST and runtime test mirror; VS XML and new
module registration; 106 unique dynamic audio operations and explicit Read security
for analysis_get; C++ lexical balance for both platform branches; module sizes and
whitespace. General IPC capability/descriptor audit PASS (677 static dispatched
methods). The new callback/FFT and UI still require the user build and live acceptance.

2026-10-19 extension acceptance: switch through all four Live View modes; inspect
1 kHz signal position, six-band power, actual time/Hz history and stale-window notice.
Ctrl+wheel at several cursor positions must anchor the plotted data except where
extent clamping is necessary. Middle drag must continue outside the plot while held;
plain wheel scrolls the panel, middle click never resets. At 1x the full range returns.
Verify other channel meters/fonts never scale and the Designer last-render view
retains its existing offline behavior. The external acceptance script now checks all
view/viewport bounds, atomic rejections, details typing and history shape/order.

## 2026-10-19 user feedback: stable status layout

The user built the extension and confirmed Ctrl+wheel graph zoom works. During live
silence transitions the conditional notice moved the whole graph by one line. Source
fix: fresh/active, silence, stale snapshots, output mute and clipping now share one
fixed-height, non-wrapping status row. Details are in the hover tooltip; active sound
retains the same row. Peak/RMS, graph origin and available graph height remain stable
across these signal-state transitions. This layout fix still needs user build/visual
acceptance; the feedback does not by itself accept every graph/history/API test.
