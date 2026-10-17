# Human voices for scene-acoustic listening tests

Real voice recordings, prepared on 2026-10-17 for the EchoRooms lab. These are
small audition fixtures, not a studio-quality full sound library. The moan sources
are lossy Vorbis originals; WAV conversion does not restore lost recording detail.

Sources and CC0 license evidence:

- HaelDB — [Male Grunt/Yelling sounds](https://opengameart.org/content/male-gruntyelling-sounds).
  The page offers CC0 alongside OGA-BY; this pack uses the CC0 option.
  Archive: https://opengameart.org/sites/default/files/yelling%20sounds.zip
- AntumDeluge — [Undead Moans](https://opengameart.org/content/undead-moans).
  Page and included LICENSE.txt identify CC0 1.0.
  Archive: https://opengameart.org/sites/default/files/undead-moans.zip

Original selected recordings remain in originals/. CC0-1.0.txt retains the
upstream legal text; manifest.json records original and derived SHA256 values.

Derived assets are 48 kHz mono PCM16, each 24 seconds long:

| File | Events |
| --- | --- |
| human_shouts_spaced_mono.wav | Two different yells, at 1 and 13 seconds |
| human_moans_a_spaced_mono.wav | Two moans, at 1 and 13 seconds |
| human_moans_b_spaced_mono.wav | Same moans in reverse order, at 7 and 19 seconds |

Processing: arithmetic mean to mono, trim quiet margins while retaining short
edges, polyphase resampling from 44.1 kHz, event peak normalization to 0.7 and
3 ms edge fades, then silent spacing. No reverb, echo, pitch change or denoising
was added. Original recording coloration/editing can remain; these are not
certified anechoic recordings.

Prepare/import/switch using scripts/test/rt_echo_voice_demo.py. It imports through
the existing audio service, adds independent VoiceLab sources, enables the shared
live spectral/reflection path, and saves a separate EchoRoomsHorrorVoices.rtp on
initial installation. Reopening clears session previews; run a room command again.
