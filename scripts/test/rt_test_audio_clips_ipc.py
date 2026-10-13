"""After user build: external IPC clip/waveform/export acceptance.

Uses an existing asset and creates/removes temporary clips at 85000 seconds.
Output must be a new path. Existing scene clips must not overlap that range.
"""
import argparse
import math
from pathlib import Path
import struct
import time
from rt_ipc import RtIpc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    client = RtIpc()
    def call(method, params=None):
        return client.call(method, **(params or {}))

    ids = []
    try:
        waveform = call("audio.waveform", {"id": args.asset_id, "points": 64})
        bins = waveform["points"]
        assert bins[0]["first_frame"] == 0
        assert bins[-1]["end_frame_exclusive"] == waveform["frames"]
        assert all(a["end_frame_exclusive"] == b["first_frame"]
                   for a, b in zip(bins, bins[1:]))
        for item in bins:
            for low, high, rms in zip(item["min"], item["max"], item["rms"]):
                assert all(map(math.isfinite, (low, high, rms)))
                assert low <= high and rms >= 0
        duration = min(0.5, waveform["frames"] / 48000 / 2)
        assert duration > 2 / 48000
        clip = call("audio.clip_create", {
            "asset_id": args.asset_id, "start_seconds": 85000,
            "duration_seconds": duration, "gain": 0.25,
            "pitch_curve": [[0, 1], [duration, 1.25]]})
        ids.append(clip["id"])
        before = client.call("audio.clip_list")
        try:
            call("audio.clip_trim", {
                "id": clip["id"], "start_seconds": 84999, "end_seconds": 85000.1})
        except RuntimeError:
            pass
        else:
            raise AssertionError("Invalid trim accepted")
        assert client.call("audio.clip_list") == before
        split = call("audio.clip_split", {
            "id": clip["id"], "time_seconds": 85000 + duration / 2})
        ids.append(split["right"]["id"])
        assert math.isclose(split["right"]["envelope_offset_seconds"], duration / 2)
        assert split["right"]["phase_seconds"] > 0
        call("audio.export_start", {
            "path": str(args.output.resolve()), "start_frame": 2040000,
            "end_frame_exclusive": 2040024})
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            report = client.call("audio.export_status")
            if report["state"] != "running":
                break
            time.sleep(0.1)
        assert report["state"] == "completed", report
        data = args.output.read_bytes()
        assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"
        assert struct.unpack_from("<H", data, 20)[0] == 3
        assert struct.unpack_from("<I", data, 44)[0] == 48000
        assert len(data) == 56 + 48000 * 8
        print("PASS: waveform coverage, atomic rejection, split, rational WAV length")
    finally:
        client.call("audio.export_cancel")
        for identity in reversed(ids):
            call("audio.clip_remove", {"id": identity})
        client.close()


if __name__ == "__main__":
    main()
