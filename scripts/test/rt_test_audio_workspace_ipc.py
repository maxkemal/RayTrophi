"""User-build acceptance over external IPC; restores mixer/editor and removes clips.

Requires an existing mono asset and new WAV paths in an existing output directory.
Do not run in Studio's embedded interpreter.
"""
import argparse
import json
import math
from pathlib import Path
import struct
import time
from rt_ipc import RtIpc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    assert args.output_dir.is_dir()
    outputs = [args.output_dir / name for name in ("audio_center.wav", "audio_right.wav")]
    assert not any(path.exists() for path in outputs)
    client = RtIpc()
    previous = client.call("audio.list")
    old_view = client.call("audio.workspace_get")
    old_editor = client.call("editor.get_state")["bottom_editor"]
    identity = None
    try:
        for mode in ("audio", "audio_mixer"):
            client.call("editor.set_bottom_editor", name=mode)
            assert client.call("editor.get_state")["open_editors"] == [mode]
        before = client.call("audio.workspace_get")
        try:
            client.call("audio.workspace_update", selected_clip="missing_clip", snap=False)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Invalid clip selection accepted")
        assert client.call("audio.workspace_get") == before
        asset = next(a for a in previous["assets"] if a["id"] == args.asset_id)
        assert asset["channels"] == 1
        assert not any(c["start_seconds"] < 85001 and
                       c["start_seconds"] + c["duration_seconds"] > 85000
                       for c in previous["clips"]), "Test range must be empty"
        clip = client.call("audio.clip_create", asset_id=args.asset_id, start_seconds=85000,
                           duration_seconds=min(0.5, asset["duration"] / 2), bus="Effects")
        identity = clip["id"]
        updated = client.call("audio.workspace_update", selected_clip=identity,
                              snap=False, waveforms=True)
        assert updated["selected_clip"] == identity and not updated["snap"]
        for bus in previous["mixer"]:
            client.call("audio.mixer_update", bus=bus, gain=1, pan=0, muted=False, solo=False)

        decoded = []
        for pan, output in zip((0, 1), outputs):
            client.call("audio.mixer_update", bus="Effects", pan=pan)
            client.call("audio.export_start", path=str(output.resolve()), start_frame=2040000,
                        end_frame_exclusive=2040024, fps_num=24)
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                report = client.call("audio.export_status")
                if report["state"] != "running":
                    break
                time.sleep(0.1)
            assert report["state"] == "completed", report
            data = output.read_bytes()
            assert len(data) == 56 + 48000 * 8
            values = struct.unpack("<96000f", data[56:])
            assert all(map(math.isfinite, values))
            decoded.append(values)
        center, right = decoded
        assert max(map(abs, center)) > 0
        assert all(value == 0 for value in right[::2]), "Hard-right bus leaked left"
        assert all(math.isclose(a, b, abs_tol=1e-7)
                   for a, b in zip(center[1::2], right[1::2])), "Right amplitude changed"
        client.call("audio.clip_remove", id=identity)
        identity = None
        assert client.call("audio.workspace_get")["selected_clip"] == ""
        print(json.dumps({"passed": True, "checks": [
            "editor modes", "atomic view validation", "shared selection",
            "offline bus balance", "selection cleanup"]}, indent=2))
    finally:
        client.call("audio.export_cancel")
        if identity:
            client.call("audio.clip_remove", id=identity)
        for bus, config in previous["mixer"].items():
            client.call("audio.mixer_update", bus=bus, gain=config["gain"],
                        pan=config.get("pan", 0), muted=config["muted"], solo=config["solo"])
        client.call("audio.workspace_update", **old_view)
        client.call("editor.set_bottom_editor", name=old_editor)
        client.close()


if __name__ == "__main__":
    main()
