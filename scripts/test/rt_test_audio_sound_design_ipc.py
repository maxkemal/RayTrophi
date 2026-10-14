"""Post-build sound designer acceptance; run in an external terminal.

No device audition unless --audition. Uses only its own temporary files and assets.
Leaves transient design state available for inspection; scene authoring data is cleaned.
"""
import argparse
import math
from pathlib import Path
import struct
import tempfile
import wave

from rt_ipc import RtIpc


def rejected(client, method, **params):
    try:
        client.call(method, **params)
    except RuntimeError:
        return
    raise AssertionError(f"Invalid request accepted: {method} {params}")


def fingerprint(report):
    return report["recipe"], report["processed"]["pcm_hash"], report["processed"]["frames"]


def pcm_hash(data):
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return f"{value:016x}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audition", action="store_true")
    args = parser.parse_args()
    assert args.output_dir.is_dir()
    client = RtIpc()
    assets = []
    initial_scene = client.call("audio.list")
    initial_workspace = client.call("audio.workspace_get")
    try:
        with tempfile.TemporaryDirectory(prefix="rt_sound_design_", dir=args.output_dir) as text:
            folder = Path(text).resolve()
            destination = client.call("audio.design_destination", folder=str(folder / "new"))
            assert Path(destination["folder"]) == folder / "new"
            assert not (folder / "new").exists(), "Destination query wrote a directory"
            assert client.call("audio.design_destination")["folder"]
            recipe = dict(mode="synth", preset="metal", duration_seconds=0.4, seed=29,
                          gain=0.25, fade_in_seconds=0.01, fade_out_seconds=0.02)
            first = client.call("audio.design_render", recipe=recipe)
            second = client.call("audio.design_render", recipe=recipe)
            assert fingerprint(first) == fingerprint(second), "Nondeterministic native PCM"
            assert first["processed"]["frames"] == 19200
            assert first["processed"]["channels"] == 1
            assert len(first["processed"]["waveform"]) == 256
            assert len(first["processed"]["spectrum"]) == 64
            assert all(math.isfinite(point[1]) for point in first["processed"]["spectrum"])
            assert client.call("audio.list") == initial_scene
            assert list(folder.iterdir()) == [], "Render created files"

            for patch in ({"preset": "bad"}, {"seed": 1.5}, {"lowpass_hz": 1},
                          {"sustain": True}, {"unexpected": 1}, {"pitch_semitones": 25},
                          {"stretch_ratio": 0}, {"loop_crossfade_seconds": 0.3},
                          {"highpass_hz": 10000, "lowpass_hz": 1000}):
                rejected(client, "audio.design_render", recipe={**recipe, **patch})
                assert fingerprint(client.call("audio.design_status")) == fingerprint(first)
                assert client.call("audio.list") == initial_scene
            rejected(client, "audio.design_preview", loop=1)
            rejected(client, "audio.design_render", recipe=recipe, extra=True)
            rejected(client, "audio.workspace_update", designer_open=1)
            assert client.call("audio.workspace_get") == initial_workspace
            client.call("audio.workspace_update", designer_open=True)
            assert client.call("audio.workspace_get")["designer_open"]
            client.call("audio.workspace_update", designer_open=initial_workspace["designer_open"])

            for name in ("../escape", "CON", "LPT1.wav", "bad:name", "trailing "):
                rejected(client, "audio.design_save", folder=str(folder / "invalid"), name=name)
                assert not (folder / "invalid").exists()
                assert client.call("audio.list") == initial_scene

            for expected in ("Impact.wav", "Impact_2.wav"):
                saved = client.call("audio.design_save", folder=str(folder), name="Impact")
                assets.append(saved["asset"]["id"])
                path = Path(saved["design"]["path"])
                assert path.name == expected
                data = path.read_bytes()
                assert len(data) == 56 + 19200 * 4
                assert data[:4] == b"RIFF" and data[48:52] == b"data"
                samples = struct.unpack("<19200f", data[56:])
                assert all(map(math.isfinite, samples))
                assert samples[0] == samples[-1] == 0
                assert pcm_hash(data[56:]) == first["processed"]["pcm_hash"]
                assert saved["design"]["pcm_hash"] == first["processed"]["pcm_hash"]
            preserved = (folder / "Impact.wav").read_bytes()
            rejected(client, "audio.design_save", path=str(folder / "Impact.wav"))
            assert (folder / "Impact.wav").read_bytes() == preserved

            transformed = client.call("audio.design_render", recipe=dict(
                mode="asset", asset_id=assets[0], source_out_seconds=0.4,
                pitch_semitones=3, stretch_ratio=1.25, highpass_hz=60, lowpass_hz=6000))
            assert transformed["processed"]["frames"] == 24000
            assert transformed["processed"]["rms"] > 0
            assert transformed["origin"]["id"] == assets[0]
            assert transformed["processed"]["pcm_hash"] != first["processed"]["pcm_hash"]
            again = client.call("audio.design_render", recipe=transformed["recipe"])
            assert fingerprint(again) == fingerprint(transformed)
            rejected(client, "audio.design_render", recipe=dict(mode="asset", asset_id="missing"))
            assert fingerprint(client.call("audio.design_status")) == fingerprint(transformed)

            looped = client.call("audio.design_render", recipe=dict(
                mode="asset", asset_id=assets[0], source_out_seconds=0.4,
                loop_crossfade_seconds=0.05))
            assert looped["processed"]["frames"] == 16800
            stereo_path = folder / "stereo.wav"
            with wave.open(str(stereo_path), "wb") as wav:
                wav.setparams((2, 2, 48000, 0, "NONE", "not compressed"))
                wav.writeframes(b"".join(struct.pack("<hh",
                    int(8000 * math.sin(2 * math.pi * 220 * frame / 48000)),
                    int(4000 * math.sin(2 * math.pi * 440 * frame / 48000)))
                    for frame in range(19200)))
            stereo = client.call("audio.asset_import", path=str(stereo_path),
                                 license="test provenance", author="sound designer probe")
            assets.append(stereo["id"])
            neutral = client.call("audio.design_render", recipe=dict(
                mode="asset", asset_id=stereo["id"], source_out_seconds=0.4,
                fade_in_seconds=0, fade_out_seconds=0))
            assert neutral["processed"]["channels"] == 2
            assert neutral["processed"]["pcm_hash"] == neutral["original"]["pcm_hash"]
            derived = client.call("audio.design_save", folder=str(folder), name="Stereo edited")
            assets.append(derived["asset"]["id"])
            assert derived["asset"]["channels"] == 2
            assert derived["asset"]["license"] == "test provenance"
            assert derived["asset"]["author"] == "sound designer probe"
            if args.audition:
                client.call("audio.design_preview", loop=True)
                assert client.call("audio.design_status")["playing"]
                client.call("audio.design_preview", original=True)
                assert client.call("audio.design_status")["audition_original"]
            client.call("audio.design_stop")
            assert not client.call("audio.design_status")["playing"]
            assert client.call("audio.design_status")["ready"]
            for asset in reversed(assets):
                client.call("audio.asset_remove", id=asset)
            assets.clear()
        print("PASS: render/save PCM, errors, names, waveform, pitch/time, loop and workspace contracts")
    finally:
        client.call("audio.design_stop")
        for asset in reversed(assets):
            client.call("audio.asset_remove", id=asset)
        client.call("audio.workspace_update", **initial_workspace)
        client.close()


if __name__ == "__main__":
    main()
