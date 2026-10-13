"""Create licensed storm variants from the packaged public HQ MP3 source.

Authoring dependencies: miniaudio, numpy, scipy (not Studio runtime dependencies).
Original lossless AIFF requires a Freesound login; this pack explicitly uses the
public HQ preview, not the original recording's lossless quality.
"""
import hashlib
import json
from pathlib import Path
import wave

import miniaudio
import numpy as np
from scipy.signal import butter, resample_poly, sosfilt


ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "RayTrophiStudio/assets/audio/cc0_storm_thunder"
SOURCE_PAGE = "https://freesound.org/people/davidhrivnak/sounds/328391/"
DOWNLOAD = "https://cdn.freesound.org/previews/328/328391_817470-hq.mp3"
RATE = 48000


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_variant(name, signal, role, description, processing):
    signal = np.asarray(signal, dtype=np.float64).copy()
    # Preserve attack; leave room for scene gain/intensity and overlapping events.
    signal *= 0.60 / max(float(np.max(np.abs(signal))), 1e-12)
    fade_in = min(48, len(signal))
    fade_out = min(RATE, len(signal))
    signal[:fade_in] *= np.linspace(0, 1, fade_in)
    signal[-fade_out:] *= np.linspace(1, 0, fade_out)
    assert np.isfinite(signal).all() and np.max(np.abs(signal)) <= 0.601
    pcm = np.rint(signal * 32767).astype("<i2")
    path = FOLDER / (name + ".wav")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(RATE)
        output.writeframes(pcm.tobytes())
    record = {
        "id": "rt_cc0_" + name, "name": description, "type": "asset",
        "assetKind": "audio", "category": "audio", "subcategory": "storm_thunder",
        "format": "wav", "entry": path.name, "description": description,
        "license": "CC0-1.0", "source": SOURCE_PAGE, "author": "davidhrivnak",
        "tags": ["weather", "thunder", "field-recording", "mono", "cc0", role],
        "version": 1, "duration": len(signal) / RATE, "channels": 1,
        "sampleRate": RATE, "loopSuitable": False, "upstream_download": DOWNLOAD,
        "processing": processing + "; peak scaled to 0.60; 1ms/1s edge fades",
        "packaged_sha256": digest(path),
        "peak_dbfs": float(20 * np.log10(np.max(np.abs(signal)))),
        "rms_dbfs": float(20 * np.log10(np.sqrt(np.mean(signal * signal)))),
    }
    (FOLDER / (name + ".asset.json")).write_text(json.dumps(record, indent=2))
    return record


def main():
    original = FOLDER / "davidhrivnak_328391_hq.mp3"
    decoded = miniaudio.decode_file(str(original),
        output_format=miniaudio.SampleFormat.FLOAT32, nchannels=1, sample_rate=RATE)
    samples = np.asarray(decoded.samples, dtype=np.float64)
    # Find the first significant thunder attack, excluding the preceding rain.
    block = RATE // 1000
    rms = np.sqrt(np.mean(samples[:len(samples) // block * block].reshape(-1, block) ** 2, axis=1))
    onset = int(np.flatnonzero(rms > 0.06)[0]) * block
    start = max(0, onset - int(0.015 * RATE))
    near = samples[start:]
    records = []
    records.append(write_variant("storm_crack_tail_a_mono", near, "near",
        "Storm lightning crack and long natural thunder tail A",
        f"Public HQ preview decoded to mono 48kHz; rain lead-in removed at {start/RATE:.6f}s"))
    records.append(write_variant("storm_crack_tail_b_mono", resample_poly(near, 25, 24), "near",
        "Storm lightning crack and long natural thunder tail B (derived pitch variation)",
        "Same recording as A; duration x25/24 and corresponding lower pitch"))
    # The far group removes the sharp upper spectrum and deepens the natural roll.
    far = sosfilt(butter(3, 850, fs=RATE, output="sos"), near)
    records.append(write_variant("storm_deep_roll_a_mono", resample_poly(far, 25, 21), "far",
        "Storm deep long-distance thunder roll A",
        "Same trimmed recording; 850Hz low-pass, duration x25/21 and corresponding lower pitch"))
    records.append(write_variant("storm_deep_roll_b_mono", resample_poly(far, 8, 7), "far",
        "Storm deep long-distance thunder roll B (derived pitch variation)",
        "Same trimmed recording; 850Hz low-pass, duration x8/7 and corresponding lower pitch"))
    manifest = {
        "pack": "CC0 real storm thunder", "license": "CC0-1.0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "verified_on": "2026-10-13", "source": SOURCE_PAGE, "author": "davidhrivnak",
        "upstream_download": DOWNLOAD, "source_sha256": digest(original),
        "source_quality": "Public lossy HQ MP3 preview; not original lossless AIFF",
        "source_note": "Author reports tube-preamp clipping in the original thunder peaks.",
        "variation_note": "All four files derive from one field recording, not independent strikes.",
        "assets": records,
    }
    (FOLDER / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for record in records:
        print(record["entry"], round(record["duration"], 3), "s", round(record["rms_dbfs"], 2), "dBFS RMS")


if __name__ == "__main__":
    main()
