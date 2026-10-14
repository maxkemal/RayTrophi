"""Post-build audio synthesis/core acceptance. Run from an external terminal.

Creates a new temporary directory beneath --output-dir. Removes only its generated
assets/clips and temporary files. Never runs from Studio's embedded interpreter.
"""
import argparse
import json
import math
from pathlib import Path
import struct
import tempfile
from rt_ipc import RtIpc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    assert args.output_dir.is_dir()
    client = RtIpc()
    assets, clips = [], []
    try:
        presets = client.call("audio.synth_presets")
        assert {p["id"] for p in presets["presets"]} == {"tone", "wind", "wood", "metal"}
        assert client.call("audio.export_status")["state"] != "running", "Export already active"
        with tempfile.TemporaryDirectory(prefix="rt_audio_synth_", dir=args.output_dir) as folder:
            base = Path(folder).resolve()
            first = base / "first.wav"
            second = base / "second.wav"
            previous = client.call("audio.list")
            for params in ({"preset": "unknown"}, {"seed": 1.5}, {"brightness": 2},
                           {"frequency_hz": float("inf")}, {"unexpected": True}):
                try:
                    client.call("audio.synth_generate", path=str(first), **params)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError(f"Invalid synthesis accepted: {params}")
                assert not first.exists()
                assert client.call("audio.list") == previous
            for output in (first, second):
                report = client.call("audio.synth_generate", path=str(output), preset="metal",
                                     duration_seconds=0.4, seed=29, gain=0.25)
                assets.append(report["asset"]["id"])
                assert report["asset"]["channels"] == 1
                assert report["synthesis"]["frames"] == 19200
            assert first.read_bytes() == second.read_bytes(), "Same seed is not deterministic"
            original = first.read_bytes()
            try:
                client.call("audio.synth_generate", path=str(first))
            except RuntimeError:
                pass
            else:
                raise AssertionError("Existing output overwritten")
            assert first.read_bytes() == original
            assert len(original) == 56 + 19200 * 4
            pcm = struct.unpack("<19200f", original[56:])
            assert all(map(math.isfinite, pcm))
            assert abs(max(map(abs, pcm)) - 0.25) < 1e-6
            assert pcm[0] == pcm[-1] == 0
            clip = client.call("audio.clip_create", asset_id=assets[0], start_seconds=85000)
            clips.append(clip["id"])
            # These canonical operations are also the timeline's live gesture path.
            changed = client.call("audio.clip_update", id=clip["id"], bus="Ambience",
                                  start_seconds=85000.1, gain=0.5,
                                  fade_in_seconds=0.05, fade_out_seconds=0.1)
            assert changed["bus"] == "Ambience" and changed["gain"] == 0.5
            assert changed["fade_in_seconds"] == 0.05
            # Restore hidden source without repeating; cross source end only with opt-in.
            short = client.call("audio.clip_resize", id=clip["id"], end_seconds=85000.2)
            assert not short["loop"]
            recovered = client.call("audio.clip_resize", id=clip["id"], end_seconds=85000.4)
            assert not recovered["loop"]
            before_resize = client.call("audio.clip_list")
            try:
                client.call("audio.clip_resize", id=clip["id"], end_seconds=85001.0)
            except RuntimeError:
                pass
            else:
                raise AssertionError("Resize silently looped without opt-in")
            assert client.call("audio.clip_list") == before_resize
            for bad in (1, "true", None):
                try:
                    client.call("audio.clip_resize", id=clip["id"],
                                end_seconds=85001.0, auto_loop=bad)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("Nonboolean auto_loop accepted")
                assert client.call("audio.clip_list") == before_resize
            extended = client.call("audio.clip_resize", id=clip["id"],
                                   end_seconds=85001.0, auto_loop=True)
            assert extended["loop"]
            assert extended["envelope_duration_seconds"] >= extended["duration_seconds"]
            assert extended["gain"] == changed["gain"]
            assert extended["fade_out_seconds"] == changed["fade_out_seconds"]
            cut = client.call("audio.clip_split", id=clip["id"], time_seconds=85000.3)
            clips.append(cut["right"]["id"])
            assert cut["right"]["envelope_offset_seconds"] > 0
            # Independent generated asset waveform verifies the native decoder path.
            wave = client.call("audio.waveform", id=assets[0], points=64)
            assert wave["frames"] == 19200 and len(wave["points"]) == 64
            # Cleanup before the temporary directory removes its files.
            for identity in clips:
                client.call("audio.clip_remove", id=identity)
            clips.clear()
            for identity in assets:
                client.call("audio.asset_remove", id=identity)
            assets.clear()
        print(json.dumps({"passed": True, "checks": ["presets", "atomic invalid input",
            "deterministic PCM", "output protection", "peak and endpoints", "waveform",
            "canonical move/bus/gain/fades/split", "source-aware resize and explicit loop"]}, indent=2))
    finally:
        for identity in clips:
            client.call("audio.clip_remove", id=identity)
        for identity in assets:
            client.call("audio.asset_remove", id=identity)
        client.close()


if __name__ == "__main__":
    main()
