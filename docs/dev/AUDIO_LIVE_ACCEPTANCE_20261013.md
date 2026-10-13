# Audio live acceptance, 2026-10-13

External named-pipe client against the user's new Studio build. Sandbox error 5
was retried with elevated sandbox access. Application was already open.

Passed:

- Existing thunder asset audio_asset_15: 64 waveform bins cover every decoded frame.
  Per-bin min/max/RMS are finite and valid.
- Invalid clip trim rejected without changing the clip collection.
- Pitch-automated clip split preserves envelope offset and advances source phase.
- Offline interval [2040000, 2040024) at 24 fps: stereo float32 48 kHz WAV,
  48000 frames, 384056 bytes. All samples finite; peak 0.1485063732,
  RMS 0.0347443848, nonzero output.
- Existing output rejected when overwrite is omitted; original bytes unchanged.
- Long overwrite export cancellation preserves the previous target bytes.
- Capture start/cancel accepted in timeline mode without creating clips.
- Temporary test clips removed; final clip collection is empty.

Output: docs/dev/audio_acceptance_20261013.wav.

The open storm scene has no matter domains (audio.domains returned an empty array).
DEM/liquid/gas/MPM live and RAM replay acceptance remains pending suitable fixtures.
Recorded playback capture, realtime clip seeking, save/reload and audible comparison
were not established by these probes. No builds or application launches performed.
