# Echo rooms listening lab — 2026-10-17

## Real voice listening extension

The user requested natural shouting and horror moans. Selected CC0 voice takes
from HaelDB and AntumDeluge are now in
`RayTrophiStudio/assets/audio/cc0_horror_test/`, with originals, legal text,
provenance and derived-file hashes. These recordings have no added test reverb;
original recording coloration can remain. Three 24-second mono sequences space
two yells/moans at 1/13 or 7/19 seconds, leaving room for the physical echoes.

Six independent VoiceLab sources were added; existing sources/geometry and the
user's placement edits are retained. The separate project is
`x64/Release/assets/acoustic_echo_lab/EchoRoomsHorrorVoices.rtp`.
`horror_session.json` beside it preserves pre-test audio/mixer/camera state.
Master and Effects test gains are 1, replacing the previously observed 4×4 gain.
During live audition the corridor shout source was repositioned into the large
room. Its edited position was preserved; a separate corridor listening-point
duplicate was added and the room switcher's source mapping updated. The new
source verified six clear paths, including approximately 251/297 ms delays.

```powershell
python scripts/test/rt_echo_voice_demo.py --room corridor
python scripts/test/rt_echo_voice_demo.py --room corridor --dry
python scripts/test/rt_echo_voice_demo.py --room hard
python scripts/test/rt_echo_voice_demo.py --room soft
```

The corridor alternates shouting and a moan from opposite directions. The other
rooms alternate two moan sources. Switching stops current audio and starts only
that room's VoiceLab pair; source authoring is retained. Run the command again
after project reload to re-enable session acoustic previews. Source preparation
uses numpy/scipy and demo-local soundfile; ordinary room switching needs only
the existing external IPC Python client.

Live corridor reports verified six shout paths (including 250/297 ms) and five
moan paths (including 204/344 ms). The initial 10-second native meter check had
zero overloaded blocks and a -8.61 dBFS held peak. Large-room voice paths were
also ready, with approximately 85–256 ms distinct delays and a -11.35 dBFS held
peak in its 10-second check. These are measured short windows, not full-cycle
acceptance. The soft room's corresponding check held -15.38 dBFS with no overload;
the corridor dry arm held -17.43 dBFS with no overload. These peak summaries do
not measure a continuous room impulse response. Reports are not substitutes for
listening or performance acceptance. Raw per-room voices_*_wet.json reports live
beside the project. The corridor is the final listening position.

## Original procedural fixture checkpoint

The user's built Studio was inspected through the external named-pipe client.
The original band demo had one active slime source and one verified reflection
only 1.092 ms behind its direct sound: mainly coloration, not a separate echo.
The original diagnostic source had moved about 1.5 km away. Existing geometry,
assets and source authoring were preserved; the previously playing slime was paused.

The separate saved project is
`x64/Release/assets/acoustic_echo_lab/EchoRoomsLab.rtp`. It adds 22 flat cube objects,
three synthetic material profiles, three dry mono fixtures and six spatial sources.
No Studio launch or compilation was performed. The original AcousticBandDemo.rtp
was not overwritten. The session checkpoint and per-room reports live beside the
new project in session.json and *_wet_report.json.

| Room | Interior dimensions | Sources | Live reflection result |
| --- | --- | --- | --- |
| hard | 60 × 24 × 80 scene units | clap, wood, chirp | Six paths/source; distinct paths about 89–233 ms |
| soft | same geometry | identical clap | Same timing; frequency-dependent absorption; reflected energy 0.28 down to 0.007 |
| corridor | 12 × 12 × 110; four partial side baffles | clap, chirp | Short side-wall paths, plus 250/320 ms paths on the chirp source |

Materials are exaggerated diagnostic coefficients, not measured real materials.
There are still only first-order reflections, up to six candidate planes/source.
Room complexity does not imply multiple-bounce room reverb.

## Listening

The hard room is left playing with all three sources. Each dry WAV lasts six
seconds: clap-like noise starts at 0.1 s, a modal wooden knock at 2.1 s, and a
90 ms sweep (about 250–2770 Hz) at 4.1 s. Silence separates the events so the
scene-generated echoes can be heard. These fixtures contain no baked-in echo or
reverb and are synthesized reproducibly by the setup script. Existing local CC0
water/slime/weather files were inspected as alternatives; the dry diagnostic
transients avoid confusing source texture with scene reflections.

From a separate terminal at the repository root, while the lab is open:

```powershell
python scripts/test/rt_setup_echo_rooms.py --room hard
python scripts/test/rt_setup_echo_rooms.py --room hard --dry
python scripts/test/rt_setup_echo_rooms.py --room soft
python scripts/test/rt_setup_echo_rooms.py --room corridor
```

Room switching stops the lab voices, moves the listener, selects the room's source,
enables the shared live acoustic operations and starts that room's sources together
in an IPC batch. `--dry` bypasses live processing for the selected room. Session
preview settings are not serialized; run a room command again after reopening.
Use `--setup` only for initial creation, not to switch an existing lab.

Keep Material preview active after adding/moving geometry. Solid mode retained
the previous raster AS in this live session, causing
`Raster AS has not incorporated the current geometry/skin revision` and bypassed
effects. Switching to Material renewed the AS and restored all queries. Turning
debug view 15 on by itself did not resolve this state. The setup script now selects
Material and waits for ready solver reports; this is a scene workflow workaround,
not a source fix for Solid-mode acoustic geometry updates.

## Native PCM checkpoint

`python scripts/test/rt_probe_echo_tail.py` compares the isolated hard-room clap
with live processing disabled/enabled and restores the three hard-room sources.
[Raw measurements](acoustics_echo_tail_20261017.json) record a late-window RMS of
0 for dry and about 0.000316 for wet, with six valid reflection paths. The 190–450 ms
window follows the original transient, so this establishes delayed engine PCM.
IPC sampling was sparse: 13/14 total samples and only one late-window sample per
arm. It does not establish a continuous impulse response, listening quality,
hardware speaker output or performance acceptance. No clipping was observed in
the room setup meter snapshots; that is not a full duration peak scan.
