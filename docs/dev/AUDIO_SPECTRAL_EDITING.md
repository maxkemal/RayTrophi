# Spectral region editing

2026-10-15 source delivery. User requested test scripts at the end of structural
work; no new test script, build or live listening verification was performed for
this revision. This is real STFT/PCM processing, pending that final acceptance.

Analyze > Processed: drag a time/frequency rectangle, choose gain, edge softness
and Rectangle/Ellipse, then Apply spectral mask. The selection adds one ordinary
`spectral_region` effect to the existing ordered rack. Properties edits its bounds,
channel, dry/wet and bypass; the rack reorders/removes it. Auto preview uses the
existing render worker. Refresh/Play applies edits when automatic preview is off.
Original analysis and a changed/unrendered recipe cannot author masks against
outdated coordinates. Escape cancels an in-progress selection. Each successful
render participates in existing Designer undo/redo, recipe reuse and Save asset.

UI, Python and IPC share `validateDesignDsp` and `applyDesignSpectralRegion` through
the normal design recipe service. There is no separate editor-only mask state;
the transient drag is only a draft. Source files remain untouched. Save asset
writes the same processed PCM and recipe metadata used by normal Designer output.

```python
recipe = {
    "mode": "synth", "preset": "wind", "duration_seconds": 3,
    "effects": [{
        "type": "spectral_region", "start_seconds": 0.5, "end_seconds": 2,
        "low_hz": 800, "high_hz": 4000, "gain_db": -24,
        "shape": "ellipse", "feather": 0.25, "channel": "all", "mix": 1,
    }],
}
job = rt.audio.design_job_start(kind="render", recipe=recipe)
# Poll design_job_status, then design_job_apply when completed.
# IPC uses audio.design_job_start/status/apply with the same parameters.
```

The normal synchronous `design_render(recipe=...)` is also supported. Catalogue
and effect visual metadata expose the module through existing script/IPC methods.

| Field | Default | Validation |
|---|---|---|
| start_seconds / end_seconds | 0 / 1 | Finite 0..60 / 0.001..60; end > start |
| low_hz / high_hz | 200 / 2000 | Finite 20..20000; high > low |
| gain_db | -24 | Finite -96..12 dB |
| feather | 0.25 | Finite 0..1, normalized edge softness |
| shape | rectangle | rectangle or ellipse |
| channel | all | all, left or right; right requires stereo when processed |
| enabled / mix | true / 1 | Boolean / finite 0..1 |

Unknown fields/types, reversed bounds and nonfinite/out-of-range numbers fail via
the existing audio_error/Python error path before replacing current Designer PCM.
The rack keeps its existing eight-effect limit. Masks outside the sound duration
have no selected STFT centers; they are not implicitly moved or clamped in time.
Bypass/zero mix skip processing. Zero dB is an identity operation.

The core uses the pinned Signalsmith Linear real FFT, a 2048-frame Hann window,
512-frame hop and padded weighted overlap-add. Each selected channel reconstructs
to exactly the existing frame count at 48 kHz. DC and Nyquist are preserved.
Frequency coordinates are logarithmic; ellipse and feather follow that space.
Gain modifies complex-bin magnitude while retaining phase. Dry/wet and subsequent
rack processing use the same existing effect chain. Cancellation is checked per
STFT window and output block; workers never access scene/UI/native voices.

This is bounded regional spectral attenuation/boost and reconstruction. Finite
window resolution spreads sharp time/frequency edges; it is not sample-exact
brick-wall isolation. Freehand brushes, large arbitrary masks, spectral healing,
phase vocoding and independent spectral-layer routing are not delivered here.
Multiple region masks can be composed within the current rack limit.

Final acceptance backlog: zero-dB/bypass PCM identity, off-region preservation
within STFT tolerance, selected-band attenuation and gain, mono/stereo channel
isolation, short/edge range reconstruction, sync/worker PCM parity, cancellation,
invalid-request atomicity, UI selection at wide/narrow DPI, undo/redo/reuse and
Save asset/project portability. Add these to the final script/listening pass.

Source-only checks passed: edited C++ delimiter/size checks (Windows branch), VS
project XML and unique source registration, IPC capability/security mirror and
generated descriptor freshness. These do not establish compilation or audible DSP
correctness. The user build for this revision is Studio C++ only; no new shader.
