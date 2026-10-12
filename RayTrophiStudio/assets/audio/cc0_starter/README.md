# CC0 audio starter

Ten small audio assets for RayTrophi playback and reactive-audio checks, approximately
4.9 MB total. Each file has its own .asset.json sidecar for the existing Assets browser.
Choose the audio / cc0_starter category or search for rain, wind, water, slime or sand.
Append To Scene selects ambience for stereo and positional playback for mono.

| Files | Intended use | Loop |
|---|---|---|
| wind_gust_stereo.wav, wind_gust_mono.wav | Short wind gust, original stereo and mono derivative | Not seamless |
| rain_loop_stereo.wav, rain_loop_mono.wav | Rain ambience / positional rain test | Source-labeled loop |
| water_flow_stereo.wav, water_flow_mono.wav | Water ambience / continuous contact-position test | Source-labeled loop |
| water_splash_mono.wav | Positional splash | One-shot |
| slime_squelch_mono.wav | Slime texture, artistic mud-contact proxy | One-shot |
| sand_spell_stereo.flac, sand_hiss_mono.wav | Synthetic sand-hiss proxy / FLAC decoder test | One-shot |

The sand effect was made with the author's mouth; it is not an actual sand-pouring
recording. The slime effect is a mud test proxy, not a claim of a mud field recording.
Only the authored rain/water loops are intended as continuous reactive beds. Asset
waveforms were decoded and checked for nonzero RMS; listening/loop-boundary quality
and live application acceptance still require user verification. Levels are retained
from the upstream recordings; these are quieter than a normalized test tone.

## Provenance and license

All three source pages explicitly list CC0. Files are covered by CC0-1.0.txt.
manifest.json records the exact upstream filename, author, source/download URL,
license verification date, source SHA256, packaged SHA256 and conversion for each file.

- remaxim: [Short wind sound](https://opengameart.org/content/short-wind-sound).
- rubberduck: [40 CC0 water / splash / slime SFX](https://opengameart.org/content/40-cc0-water-splash-slime-sfx).
- qubodup: [Sand spell](https://opengameart.org/content/sand-spell).

The stereo wind WAV and stereo sand FLAC are unchanged originals. OGG sources were
decoded to PCM16 WAV because stage 1 accepts WAV/FLAC/MP3. Mono derivatives use the
arithmetic mean of the original channels, retain the sample rate and receive no
gain normalization. Upstream files were not edited. Original archive/files remain
available at the manifest URLs. No unrelated art or executable was included.

## Availability and tests

VS copies this folder, including provenance and CC0 text, beside the built executable
as assets/audio/cc0_starter. CMake already copies the complete assets tree. Before
rebuilding, add RayTrophiStudio/assets as a library in Assets and Refresh to see them.
No build has been performed by Codex.

For wind/rain reactions bind the matching audio, set Driver, enable Loop and Play.
For exact positional tests use the *_mono files. Stereo files are ambience sources;
their 3D checkbox is disabled with an explanation. An explicit spatial=true request
on a stereo asset is rejected by the same service in UI/script/IPC.

Suggested acceptance: audition rain_loop_stereo, append rain_loop_mono and move its
gizmo, decode sand_spell_stereo.flac, verify source creation/binding without an
explicit spatial flag selects stereo ambience, and verify an explicit positional
stereo request reports a useful error. For sand/water/mud reactions use the signal
preview described in docs/dev/SCENE_AUDIO_REACTIONS.md; true domain telemetry is
not automatically connected yet.
