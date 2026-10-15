"""Post-build track CRUD, atomic validation and shared offline mixing acceptance.

Run externally with Studio open: --output-dir <existing folder>. Own temporary
assets/tracks/clips/files only; mixer and workspace are restored. No audition.
Uses a reserved 85000-second range and refuses an existing clip overlap.
"""
import argparse
import math
from pathlib import Path
import struct
import tempfile
import time
import wave
from rt_ipc import RtIpc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    assert args.output_dir.is_dir()
    client = RtIpc()
    mixer = client.call("audio.mixer_get")
    workspace = client.call("audio.workspace_get")
    tracks, clips, asset = [], [], None
    initial_order = [t["id"] for t in client.call("audio.track_list")]
    start = 85000

    def rejected(method, **params):
        before = client.call("audio.list")
        try:
            client.call(method, **params)
        except RuntimeError:
            assert client.call("audio.list") == before
            return
        raise AssertionError(f"Invalid call accepted: {method}")

    def export(path):
        client.call("audio.export_start", path=str(path), start_frame=start * 24,
                    end_frame_exclusive=start * 24 + 12, fps_num=24, fps_den=1)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            status = client.call("audio.export_status")
            if status["state"] == "completed":
                break
            assert status["state"] != "failed", status
            time.sleep(0.02)
        else:
            raise AssertionError("Export timed out")
        data = path.read_bytes()
        offset = 12
        while data[offset:offset + 4] != b"data":
            size = struct.unpack_from("<I", data, offset + 4)[0]
            offset += 8 + size + size % 2
        size = struct.unpack_from("<I", data, offset + 4)[0]
        values = struct.unpack_from(f"<{size // 4}f", data, offset + 8)
        return [math.sqrt(sum(v * v for v in values[channel::2]) /
                          len(values[channel::2])) for channel in (0, 1)]

    try:
        scene = client.call("audio.list")
        assert not any(c["start_seconds"] < start + 1 and
                       c["start_seconds"] + c["duration_seconds"] > start
                       for c in scene["clips"])
        for bus in mixer["buses"]:
            client.call("audio.mixer_update", bus=bus["name"], gain=1,
                        muted=False, solo=False, pan=0)
        with tempfile.TemporaryDirectory(prefix="rt_audio_tracks_", dir=args.output_dir) as text:
            folder = Path(text).resolve()
            source = folder / "track_source.wav"
            with wave.open(str(source), "wb") as wav:
                wav.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
                wav.writeframes(b"".join(struct.pack("<h", int(6000 * math.sin(
                    2 * math.pi * 440 * frame / 48000))) for frame in range(24000)))
            asset = client.call("audio.asset_import", path=str(source))["id"]
            first = client.call("audio.track_create", name="Track acceptance A", gain=0.5, solo=True)
            tracks.append(first["id"])
            second = client.call("audio.track_create", name="Track acceptance B", bus="Music")
            tracks.append(second["id"])
            clip = client.call("audio.clip_create", asset_id=asset, track_id=first["id"],
                               start_seconds=start, gain=0.25)
            clips.append(clip["id"])
            assert clip["track_id"] == first["id"] and clip["bus"] == "Effects"
            a = export(folder / "a.wav")
            client.call("audio.track_update", id=first["id"], gain=0.25)
            b = export(folder / "b.wav")
            assert a[0] > 0.001 and abs(b[0] / a[0] - 0.5) < 1e-5
            client.call("audio.track_update", id=first["id"], pan=1)
            pan = export(folder / "pan.wav")
            assert pan[0] < 1e-8 and pan[1] > 0.001
            client.call("audio.track_update", id=first["id"], muted=True)
            assert max(export(folder / "muted.wav")) < 1e-8
            client.call("audio.track_update", id=first["id"], muted=False, pan=0)
            client.call("audio.track_update", id=first["id"], solo=False)
            client.call("audio.track_update", id=second["id"], solo=True)
            assert max(export(folder / "other_solo.wav")) < 1e-8
            client.call("audio.track_update", id=first["id"], solo=True)
            assert min(export(folder / "both_solo.wav")) > 0.001
            for patch in ({"gain": True}, {"gain": 17}, {"pan": -2},
                          {"muted": 1}, {"name": ""}, {"bus": "missing"}):
                rejected("audio.track_update", id=first["id"], **patch)
            rejected("audio.clip_update", id=clip["id"], track_id="missing")
            rejected("audio.track_remove", id=first["id"])
            order = [t["id"] for t in client.call("audio.track_list")]
            rejected("audio.track_reorder", ids=order + [order[0]])
            order.reverse()
            assert [t["id"] for t in client.call("audio.track_reorder", ids=order)] == order
            client.call("audio.track_update", id=first["id"], bus="Weather")
            assert next(c for c in client.call("audio.clip_list") if c["id"] == clip["id"])["bus"] == "Weather"
            client.call("audio.track_remove", id=first["id"], clip_policy="detach")
            tracks.remove(first["id"])
            detached = next(c for c in client.call("audio.clip_list") if c["id"] == clip["id"])
            assert detached["track_id"] == "" and detached["bus"] == "Weather"
            client.call("audio.clip_update", id=clip["id"], track_id=second["id"])
            client.call("audio.track_remove", id=second["id"], clip_policy="delete")
            tracks.clear()
            clips.clear()
            client.call("audio.asset_remove", id=asset)
            asset = None
        print("PASS: tracks, clip binding, routing, gain/pan/mute/solo export, atomic errors and removal")
    finally:
        client.call("audio.export_cancel")
        for identity in reversed(clips):
            client.call("audio.clip_remove", id=identity)
        for identity in reversed(tracks):
            client.call("audio.track_remove", id=identity, clip_policy="detach")
        if asset:
            client.call("audio.asset_remove", id=asset)
        remaining = [t["id"] for t in client.call("audio.track_list")]
        restored_order = [identity for identity in initial_order if identity in remaining]
        restored_order += [identity for identity in remaining if identity not in restored_order]
        client.call("audio.track_reorder", ids=restored_order)
        for bus in mixer["buses"]:
            client.call("audio.mixer_update", bus=bus["name"], gain=bus["gain"],
                        muted=bus["muted"], solo=bus["solo"], pan=bus["pan"])
        client.call("audio.workspace_update", **workspace)
        client.close()


if __name__ == "__main__":
    main()
