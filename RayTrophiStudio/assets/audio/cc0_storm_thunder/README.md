# Real storm lightning crack and long thunder

Source: [Thunder 2.aif by davidhrivnak](https://freesound.org/people/davidhrivnak/sounds/328391/).
The author's Freesound page explicitly dedicates this recording under CC0 1.0.
License verified 2026-10-13; the included CC0 text applies to these derivatives.

The packaged source is the publicly available HQ MP3 preview. It is **lossy**, not
the original lossless AIFF (which requires login). This is a real field recording;
the author describes tube-preamp clipping on the loudest original peaks. Rain is
audible behind the thunder. All four variants derive from this single recording.

Near variants preserve the crack and 17–18 second natural rolling tail. Far variants
low-pass the same signal at 850 Hz and lower its pitch for a 20–21 second deeper roll.
No artificial replacement crack or looped tail is added. The initial rain lead-in
is trimmed with 15ms pre-roll. Files are mono PCM16 WAV, 48kHz; peak is scaled to
0.60 before 1ms/1s edge fades, leaving headroom for scene gain. They are not loops.

`manifest.json` and per-file sidecars record provenance, transformations, duration,
PCM measurements and SHA256. `scripts/assets/prepare_storm_thunder.py` reproduces
them from the included HQ MP3 using miniaudio/numpy/scipy authoring tools.

Fresh thunder setup prefers this pack when present and falls back to the legacy
CC0 variations when absent. Existing user-authored near/far bindings are preserved
by setup. Imported scene assets are made portable through the existing project
save/copy workflow. Packaged PCM is a sound-design choice; final quality requires
user listening acceptance in the scene.
