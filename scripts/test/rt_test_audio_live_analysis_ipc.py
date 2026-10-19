"""External post-build live mixer analysis acceptance.

Default: schema/readback and reversible workspace validation, safe for an open scene.
--signal: requires an empty disposable audio scene; creates a quiet temporary stereo
fixture, changes/restores mixer/output mute, and verifies real callback PCM. No build,
launch, embedded scripting, scene save or Designer rendering.
"""
import argparse
import math
from pathlib import Path
import struct
import tempfile
import time
import wave

from rt_ipc import RtIpc, RtIpcError


BUSES = ("Master", "Weather", "Effects", "Ambience", "Music")


def reject(client, method, **params):
    try:
        client.call(method, **params)
    except RtIpcError:
        return
    raise AssertionError(f"Invalid request accepted: {method} {params}")


def validate(report):
    assert report["bus"] in BUSES
    assert report["stage"] == "post_bus_fader_before_output_mute"
    assert isinstance(report["ready"], bool)
    assert set(report["bus_meters"]) == set(BUSES)
    if not report["ready"]:
        return
    assert report["sample_rate"] > 0 and report["window_frames"] == 2048
    assert len(report["spectrum"]) == 160
    assert len(report["channels"]) == 2
    assert [band["frequency_hz"] for band in report["bands"]] == [125, 250, 500, 1000, 2000, 4000]
    for hz, db in report["spectrum"]:
        assert math.isfinite(hz) and hz > 0 and math.isfinite(db)
    for band in report["bands"]:
        assert math.isfinite(band["mean_square"]) and band["mean_square"] >= 0
        assert math.isfinite(band["dbfs"])
    assert math.isfinite(report["peak_dbfs"]) and math.isfinite(report["rms_dbfs"])
    assert report["snapshot_age_ms"] >= 0


def validate_history(report):
    validate(report)
    if not report["device_ready"]:
        return
    assert report["history_kind"] == "observed_fft_windows"
    assert report["history_capacity"] == 128
    rows = report["history"]
    assert len(rows) <= 128
    frames = [row["end_frame"] for row in rows]
    assert frames == sorted(set(frames))
    if frames:
        assert frames[-1] <= report["end_frame"]
        assert frames[0] >= report["window_frames"]
    for previous, current in zip(frames, frames[1:]):
        assert current - previous >= report["sample_rate"] // 15
    for row in rows:
        assert len(row["dbfs"]) == 160
        assert all(math.isfinite(db) for db in row["dbfs"])


def wait_for(client, bus, predicate):
    deadline = time.monotonic() + 8
    latest = None
    while time.monotonic() < deadline:
        latest = client.call("audio.analysis_get", bus=bus)
        validate(latest)
        validate_history(client.call("audio.analysis_get", bus=bus, details=True))
        if latest["ready"] and latest["fresh"] and predicate(latest):
            return latest
        time.sleep(0.06)
    raise AssertionError(f"Live audio did not reach expected state: {bus}, {latest}")


def mixer_state(client):
    view = client.call("audio.mixer_get")
    return ({row["name"]: {key: row[key] for key in ("gain", "pan", "muted", "solo")}
             for row in view["buses"]}, view["output_muted"])


def signal_test(client):
    scene = client.call("audio.list")
    if scene["emitters"] or scene["clips"] or scene["tracks"]:
        raise RuntimeError("--signal requires an empty disposable audio scene")
    design = client.call("audio.design_status", details=False)
    if design["playing"]:
        raise RuntimeError("Stop Designer audition before --signal")
    mixer, muted = mixer_state(client)
    asset_id = source_id = None
    with tempfile.TemporaryDirectory(prefix="rt_live_analysis_") as folder:
        try:
            path = Path(folder) / "opposite_stereo.wav"
            # Opposite stereo polarity must not cancel in a power-averaged spectrum.
            samples = b"".join(struct.pack("<hh", value, -value) for value in (
                int(0.2 * 32767 * math.sin(2 * math.pi * 1000 * frame / 48000))
                for frame in range(48000)))
            with wave.open(str(path), "wb") as output:
                output.setparams((2, 2, 48000, 0, "NONE", "not compressed"))
                output.writeframes(samples)
            client.call("audio.preview_mute", muted=True)
            client.call("audio.mixer_update", bus="Master", gain=1, pan=0, muted=False)
            client.call("audio.mixer_update", bus="Music", gain=1, pan=0, muted=False, solo=True)
            asset_id = client.call("audio.asset_import", path=str(path))["id"]
            source_id = client.call("audio.create", name="Live analyzer fixture",
                                    asset_id=asset_id, bus="Music", spatial=False,
                                    gain=1, loop=True)["id"]
            client.call("audio.play", id=source_id)
            full = wait_for(client, "Music", lambda r: abs(r["peak_dbfs"] + 13.98) < 0.5)
            dominant = max(full["spectrum"], key=lambda point: point[1])[0]
            assert abs(dominant - 1000) < 100, dominant
            assert abs(full["bands"][3]["dbfs"] + 16.99) < 0.6, full["bands"]
            assert full["clip_samples_window"] == 0
            assert full["output_muted"] and not full["silent"]
            advanced = wait_for(client, "Music", lambda r: r["end_frame"] > full["end_frame"])
            assert advanced["sample_rate"] == 48000
            client.call("audio.mixer_update", bus="Music", gain=0.5)
            half = wait_for(client, "Music", lambda r:
                            r["end_frame"] > advanced["end_frame"] and
                            abs(r["peak_dbfs"] - full["peak_dbfs"] + 6.0206) < 0.5)
            assert abs(half["bands"][3]["dbfs"] - full["bands"][3]["dbfs"] + 6.0206) < 0.6
            client.call("audio.mixer_update", bus="Master", gain=0.5)
            master = wait_for(client, "Master", lambda r:
                              abs(r["peak_dbfs"] - half["peak_dbfs"] + 6.0206) < 0.5)
            child = client.call("audio.analysis_get", bus="Music")
            assert abs(child["peak_dbfs"] - half["peak_dbfs"]) < 0.5
            client.call("audio.mixer_update", bus="Music", muted=True)
            wait_for(client, "Music", lambda r:
                     r["end_frame"] > half["end_frame"] and r["silent"] and r["peak_dbfs"] < -110)
            assert client.call("audio.design_status", details=False)["revision"] == design["revision"]
            print("PASS signal: real 1 kHz spectrum, stereo polarity, advancing frames, "
                  "six-band power, channel/Master gain, mute, no Designer render")
        finally:
            if source_id is not None:
                client.call("audio.stop", id=source_id)
                client.call("audio.remove", id=source_id)
            if asset_id is not None:
                client.call("audio.asset_remove", id=asset_id)
            for bus, settings in mixer.items():
                client.call("audio.mixer_update", bus=bus, **settings)
            client.call("audio.preview_mute", muted=muted)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal", action="store_true")
    args = parser.parse_args()
    client = RtIpc()
    workspace = client.call("audio.workspace_get")
    try:
        for bus in BUSES:
            basic = client.call("audio.analysis_get", bus=bus)
            validate(basic)
            assert "history" not in basic
            validate_history(client.call("audio.analysis_get", bus=bus, details=True))
        assert client.call("audio.analysis_get")["bus"] == "Master"
        for params in ({"bus": "invalid"}, {"bus": None}, {"bus": 1}, {"unknown": True},
                       {"details": 1}, {"details": None}, {"details": "yes"}):
            reject(client, "audio.analysis_get", **params)
        for mode in ("live", "design"):
            updated = client.call("audio.workspace_update", analysis_mode=mode)
            assert updated["analysis_mode"] == mode
        for view in ("line", "filled", "bands", "spectrogram"):
            updated = client.call("audio.workspace_update", analysis_view=view,
                                  analysis_zoom_x=4, analysis_zoom_y=2,
                                  analysis_pan_x=0.25, analysis_pan_y=0.75)
            assert updated["analysis_view"] == view
            assert updated["analysis_zoom_x"] == 4 and updated["analysis_zoom_y"] == 2
            assert updated["analysis_pan_x"] == 0.25 and updated["analysis_pan_y"] == 0.75
        stable = client.call("audio.workspace_get")
        for value in ("invalid", None, True, 1):
            reject(client, "audio.workspace_update", analysis_view=value, mixer_bus="Music")
            assert client.call("audio.workspace_get") == stable
        for key in ("analysis_zoom_x", "analysis_zoom_y", "analysis_pan_x", "analysis_pan_y"):
            low, high = (1, 32) if "zoom" in key else (0, 1)
            for value in (low, high):
                assert client.call("audio.workspace_update", **{key: value})[key] == value
            stable = client.call("audio.workspace_get")
            for value in (low - 0.01, high + 0.01, None, True, "2"):
                reject(client, "audio.workspace_update", mixer_bus="Music", **{key: value})
                assert client.call("audio.workspace_get") == stable
        for mode in ("invalid", None, True, 1):
            reject(client, "audio.workspace_update", analysis_mode=mode, mixer_bus="Music")
            assert client.call("audio.workspace_get") == stable
        print("PASS contracts: all buses, finite snapshots, shared mode, atomic invalid requests")
        if args.signal:
            signal_test(client)
    finally:
        client.call("audio.workspace_update", **workspace)
        client.close()


if __name__ == "__main__":
    main()
