# Procedural audio v1

Status: legacy v1 build/listening accepted by user on 2026-10-14; Sound Designer v2
Studio build also confirmed, with its new listening/UI/IPC acceptance pending.

UI, `rt.audio.synth_generate(**kwargs)` and IPC `audio.synth_generate` call AudioService.
`audio.synth_presets` / `rt.audio.synth_presets()` expose the same versioned catalogue.
The generated mono float32 48 kHz WAV is imported as a normal persistent asset;
source bindings, clips, waveform, mixer, capture and offline export reuse existing paths.
Generate sound is available from Audio Assets and the Audio timeline. Optional insertion
uses the playhead. Generation does not start playback automatically.

| Parameter | Contract |
|---|---|
| path | Required absolute new .wav file in an existing directory |
| preset | tone, wind, metal, wood; default metal |
| duration_seconds | 0.05..10, default 2; rounded to nearest 48 kHz frame |
| seed | integer 0..4294967295, default 1 |
| frequency_hz | 20..4000, preset default |
| decay_seconds | 0.01..10, preset default; impact damping |
| brightness | 0..1, default 0.5; noise cutoff and higher-mode damping |
| gain | target peak amplitude 0..1, default 0.5 |

Tone is a sine oscillator. Wind is filtered seeded noise with slow modulation.
Metal/wood are three damped oscillators with a short noise transient. Generation
has endpoint fades and peak normalization, and does not claim acoustic calibration
or physically accurate materials. The same parameters/seed reproduce PCM bytes on
this implementation/platform. Preset/version/normalized controls are stored in the
asset's source metadata. No third-party sound recording is used.

The service returns `{asset, synthesis}` including frame count, peak/RMS and recipe.
Invalid controls/unknown keys fail before file creation or scene mutation. Existing
outputs are always rejected. A uniquely reserved neighboring temporary file is
published without replacement; errors remove that temporary file. Failed asset
import removes the generated output. Generation is synchronous and bounded to ten
seconds of audio, on the editor thread, never on the device callback. Windows file
publication is currently required. Long/background jobs remain future work.

Undo removes the imported asset/inserted clip through existing authoring history;
it leaves the authored WAV on disk. Save the scene and retain/package the WAV as
with any imported asset. Generated textures are not seamless loops. Tone/wind do
not use the impact decay control; seed influences noisy presets, not pure tone.

## Next synthesis stage (not implemented)

A runtime generator driven by canonical DEM normal/friction/rolling, liquid/gas and
MPM telemetry: bounded impact scheduling, material/size resonators, continuously
controlled flow/friction textures, gain smoothing, voice budget, deterministic scene
clock/RAM replay, capture recipe and equivalent offline evaluation. Wind/rain/fire
and thunder textures need listening acceptance and dedicated authored presets.
This v1 asset generator must not be described as completion of physics-driven synthesis.

## User verification

Build Studio; no new shader is required. Run externally:

```powershell
python scripts/test/rt_test_audio_synthesis_ipc.py --output-dir E:\YOUR_EXISTING_TEST_FOLDER
```

Checks deterministic native PCM, finite/peak/endpoints, waveform decode, rejected
inputs preserving scene/files, existing-output protection and the same canonical
move/bus/gain/fade/split operations used by timeline gestures. Test creates its own
temporary directory and removes only its generated assets/clips/files. Listening,
interactive gestures, save/reload and undo remain manual acceptance checks.

## Sound designer v2 continuation

The legacy API above is retained. [Sound designer v2](AUDIO_SOUND_DESIGN.md) adds
in-memory render/audition, richer DSP controls and name/folder-based asset creation
through shared UI/script/IPC operations. The user previously accepted v1 synthetic
sound, clip fades and live movement; the v2 Studio build is now confirmed, with new listening/UI/IPC results pending.

## Sound designer v2 continuation

The legacy API above is retained. [Sound designer v2](AUDIO_SOUND_DESIGN.md) adds
in-memory render/audition, richer DSP controls and name/folder-based asset creation
through shared UI/script/IPC operations. The user previously accepted v1 synthetic
sound, clip fades and live movement; the v2 Studio build is now confirmed, with new listening/UI/IPC results pending.
